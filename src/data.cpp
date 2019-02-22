//
//  data.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//
// most read file methods are adopted from GCTA with modification

#include "data.hpp"
#include <mpi.h>
#include <Eigen/Eigen>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iterator>


#define handle_error(msg)                               \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

Data::Data()
    : ppBedFd(-1)
    , sqNormFd(-1)
    , ppBedMap(nullptr)
    , sqNormMap(nullptr)
    , mappedZ(nullptr, 1, 1)
    , mappedZPZDiag(nullptr, 1)
{
}

void Data::preprocessBedFile(const string &bedFile, const string &preprocessedBedFile, const string &sqNormFile)
{
    cout << "Preprocessing bed file: " << bedFile << endl;
    if (numIncdSnps == 0)
        throw ("Error: No SNP is retained for analysis.");
    if (numKeptInds == 0)
        throw ("Error: No individual is retained for analysis.");

    ifstream BIT(bedFile.c_str(), ios::binary);
    if (!BIT)
        throw ("Error: can not open the file [" + bedFile + "] to read.");

    ofstream ppBedOutput(preprocessedBedFile.c_str(), ios::binary);
    if (!ppBedOutput)
        throw("Error: Unable to open the preprocessed bed file [" + preprocessedBedFile + "] for writing.");
    ofstream sqNormOutput(sqNormFile.c_str(), ios::binary);
    if (!sqNormOutput)
        throw("Error: Unable to open the preprocessed square norm file [" + sqNormFile + "] for writing.");

    cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    char header[3];
    BIT.read(header, 3);
    if (!BIT || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01)
        throw ("Error: Incorrect first three bytes of bed file: " + bedFile);

    // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 10: hetezygote; 01: missing
    for (unsigned int j = 0, snp = 0; j < numSnps; j++) {
        SnpInfo *snpInfo = snpInfoVec[j];
        double sum = 0.0;
        unsigned int nmiss = 0;

        // Create some scratch space to preprocess the raw data
        VectorXf snpData(numKeptInds);
        float sqNorm = 0.0f;

        // Make a note of which individuals have a missing genotype
        vector<long> missingIndices;

        const unsigned int size = (numInds + 3) >> 2;
        if (!snpInfo->included) {
            BIT.ignore(size);
            continue;
        }

        for (unsigned int i = 0, ind = 0; i < numInds;) {
            char ch;
            BIT.read(&ch, 1);
            if (!BIT)
                throw ("Error: problem with the BED file ... has the FAM/BIM file been changed?");

            bitset<8> b = ch;
            unsigned int k = 0;

            while (k < 7 && i < numInds) {
                if (!indInfoVec[i]->kept) {
                    k += 2;
                } else {
                    const unsigned int allele1 = (!b[k++]);
                    const unsigned int allele2 = (!b[k++]);

                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        // Don't store a marker value like this as it requires floating point comparisons later
                        // which are not done properly. Instead, store the index of the individual in a vector and simply
                        // iterate over the collected indices. Also means iterating over far fewer elements which may
                        // make a noticeable difference as this scales up.
                        missingIndices.push_back(ind++);
                        ++nmiss;
                    } else {
                        const auto value = allele1 + allele2;
                        snpData[ind++] = value;
                        sum += value;
                    }
                }
                i++;
            }
        }

        // Fill missing values with the mean
        const double mean = sum / double(numKeptInds - nmiss);
        if (j % 100 == 0) {
            printf("MARKER %6d mean = %12.7f computed on %6.0f with %6d elements (%d - %d)\n",
                   j, mean, sum, numKeptInds-nmiss, numKeptInds, nmiss);
            fflush(stdout);
        }
        if (nmiss) {
            for (const auto index : missingIndices)
                snpData[index] = float(mean);
        }

        // Standardize genotypes
        snpData.array() -= snpData.mean();
        //sqNorm = snpData.squaredNorm();
        float sqn = snpData.squaredNorm();
        float std_ = 1.f / (sqrt(sqn / (float(numKeptInds)-1.0)));
        snpData.array() *= std_;
        // Write out the preprocessed data
        ppBedOutput.write(reinterpret_cast<char *>(&snpData[0]), numInds * sizeof(float));
        sqNormOutput.write(reinterpret_cast<char *>(&sqNorm), sizeof(float));

        // Compute allele frequency and any other required data and write out to file
        //snpInfo->af = 0.5f * float(mean);
        //snp2pq[snp] = 2.0f * snpInfo->af * (1.0f - snpInfo->af);

        if (++snp == numIncdSnps)
            break;
    }
    BIT.clear();
    BIT.close();

    cout << "Genotype data for " << numKeptInds << " individuals and " << numIncdSnps << " SNPs are included from [" + bedFile + "]." << endl;
}

void Data::mapPreprocessBedFile(const string &preprocessedBedFile, const string &sqNormFile)
{
    // Calculate the expected file sizes - cast to size_t so that we don't overflow the unsigned int's
    // that we would otherwise get as intermediate variables!
    const size_t ppBedSize = size_t(numInds) * size_t(numIncdSnps) * sizeof(float);
    const size_t sqNormSize = size_t(numIncdSnps) * sizeof(float);

    // Open and mmap the preprocessed bed file
    ppBedFd = open(preprocessedBedFile.c_str(), O_RDONLY);
    if (ppBedFd == -1)
        throw("Error: Failed to open preprocessed bed file [" + preprocessedBedFile + "]");

    ppBedMap = reinterpret_cast<float *>(mmap(nullptr, ppBedSize, PROT_READ, MAP_SHARED, ppBedFd, 0));
    if (ppBedMap == MAP_FAILED)
        throw("Error: Failed to mmap preprocessed bed file");

    // Open and mmap the sqNorm file
    sqNormFd = open(sqNormFile.c_str(), O_RDONLY);
    if (sqNormFd == -1)
        throw("Error: Failed to open preprocessed square norm file [" + sqNormFile + "]");

    sqNormMap = reinterpret_cast<float *>(mmap(nullptr, sqNormSize, PROT_READ, MAP_SHARED, sqNormFd, 0));
    if (sqNormMap == MAP_FAILED)
        throw("Error: Failed to mmap preprocessed square norm file");

    // Now that the raw data is available, wrap it into the mapped Eigen types using the
    // placement new operator.
    // See https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
    new (&mappedZ) Map<MatrixXf>(ppBedMap, numInds, numIncdSnps);
    new (&mappedZPZDiag) Map<VectorXf>(sqNormMap, numIncdSnps);
}

void Data::unmapPreprocessedBedFile()
{
    // Unmap the data from the Eigen accessors
    new (&mappedZ) Map<MatrixXf>(nullptr, 1, 1);
    new (&mappedZPZDiag) Map<VectorXf>(nullptr, 1);

    const auto ppBedSize = numInds * numIncdSnps * sizeof(float);
    const auto sqNormSize = numIncdSnps * sizeof(float);
    munmap(ppBedMap, ppBedSize);
    munmap(sqNormMap, sqNormSize);

    close(ppBedFd);
    close(sqNormFd);
}

bool SnpInfo::isProximal(const SnpInfo &snp2, const float genWindow) const {
    return chrom == snp2.chrom && fabs(genPos - snp2.genPos) < genWindow;
}

bool SnpInfo::isProximal(const SnpInfo &snp2, const unsigned physWindow) const {
    return chrom == snp2.chrom && abs(physPos - snp2.physPos) < physWindow;
}

void Data::readFamFile(const string &famFile){
    // ignore phenotype column
    ifstream in(famFile.c_str());
    if (!in) throw ("Error: can not open the file [" + famFile + "] to read.");
    if (myMPI::rank==0)
        cout << "Reading PLINK FAM file from [" + famFile + "]." << endl;
    indInfoVec.clear();
    indInfoMap.clear();
    string fid, pid, dad, mom, sex, phen;
    unsigned idx = 0;
    while (in >> fid >> pid >> dad >> mom >> sex >> phen) {
        IndInfo *ind = new IndInfo(idx++, fid, pid, dad, mom, atoi(sex.c_str()));
        indInfoVec.push_back(ind);
        if (indInfoMap.insert(pair<string, IndInfo*>(ind->catID, ind)).second == false) {
            throw ("Error: Duplicate individual ID found: \"" + fid + "\t" + pid + "\".");
        }
    }
    in.close();
    numInds = (unsigned) indInfoVec.size();
    if (myMPI::rank==0)
        cout << numInds << " individuals to be included from [" + famFile + "]." << endl;
}

void Data::readBimFile(const string &bimFile) {
    // Read bim file: recombination rate is defined between SNP i and SNP i-1
    ifstream in(bimFile.c_str());
    if (!in) throw ("Error: can not open the file [" + bimFile + "] to read.");
    if (myMPI::rank==0)
        cout << "Reading PLINK BIM file from [" + bimFile + "]." << endl;
    snpInfoVec.clear();
    snpInfoMap.clear();
    string id, allele1, allele2;
    unsigned chr, physPos;
    float genPos;
    unsigned idx = 0;
    while (in >> chr >> id >> genPos >> physPos >> allele1 >> allele2) {
        SnpInfo *snp = new SnpInfo(idx++, id, allele1, allele2, chr, genPos, physPos);
        snpInfoVec.push_back(snp);
        chromosomes.insert(snp->chrom);
        if (snpInfoMap.insert(pair<string, SnpInfo*>(id, snp)).second == false) {
            throw ("Error: Duplicate SNP ID found: \"" + id + "\".");
        }
    }
    in.close();
    numSnps = (unsigned) snpInfoVec.size();
    if (myMPI::rank==0)
        cout << numSnps << " SNPs to be included from [" + bimFile + "]." << endl;
}


//EO: Method to get a specific SNP data from a bed file using mmap
//----------------------------------------------------------------
void Data::getSnpDataFromBedFileUsingMmap_openmp(const string &bedFile, const size_t snpLenByt, const long memPageSize, const uint snpInd, VectorXf &snpData) {

    struct stat sb;

    snpData.resize(numKeptInds);
    ZPZdiag.resize(numIncdSnps);

    //EO: check how this should be dealt with
    //
    // Early return if SNP is to be ignored
    if (!snpInfoVec[snpInd]->included)
        return;

    int fd = open(bedFile.c_str(), O_RDONLY);
    if (fd == -1)
        handle_error("opening bedFile");

    if (fstat(fd, &sb) == -1)
        handle_error("fstat");
    
    if (!S_ISREG(sb.st_mode))
        handle_error("Not a regular file");
  
    off_t offset    = 3 + snpInd * snpLenByt;
    off_t pa_offset = offset & ~(memPageSize - 1);

    size_t relOffset  = offset - pa_offset;
    size_t bytesToMap = relOffset + snpLenByt;


    char  *addr = static_cast<char*>(mmap(0, bytesToMap, PROT_READ, MAP_PRIVATE, fd, pa_offset));  
    if (addr == MAP_FAILED)
        handle_error("mmap failed");
  
    int   nmiss = 0;
    float mean  = 0.f;
    int   sumi  = 0;

    int N = omp_get_max_threads();
    //if (snpInd%100 == 0)
    //    cout << "Max number of frame N = " << N << endl;
    
    int Nmax = numInds%4 ? numInds/4 + 1 : numInds/4;
    //printf("Nmax = %d\n", Nmax);
    if (Nmax < N)
        N = Nmax;

    omp_set_num_threads(N);

    vector<int> tSizeVec(N);

#pragma omp parallel
    {
        int id, i, Nthrds, istart, iend;

        int itmiss = 0;
        int itsum  = 0;
        vector<float> tSnpData; 

        id = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        //if (snpInd%100 == 0 && id == 0)
        //    cout << "Will use N = " << N << " threads" << endl;
        istart =  id    * (numInds/4) / Nthrds * 4;
        iend   = (id+1) * (numInds/4) / Nthrds * 4 - 1;
        if (id == Nthrds-1) iend = numInds - 1;
        //printf("thread %d [%6d. %6d]\n", id, istart, iend);

        tSnpData.resize(iend-istart+1);

        unsigned allele1=0, allele2=0;
        bitset<8> b;
        int ii    = 0;

        for(i=istart; i<=iend; i+=4) {
            
            b = addr[relOffset + i/4];

            for (int k=0; k<4 && i+k<=iend; ++k) {

                if (indInfoVec[i+k]->kept) {
                    allele1 = (!b[2*k]);
                    allele2 = (!b[2*k+1]);
                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        tSnpData[ii] = -9;
                        ++itmiss;
                    } else {
                        tSnpData[ii] = allele1 + allele2;
                        itsum += tSnpData[ii];
                    }
                    ++ii;
                }
            }
        }

        tSnpData.resize(ii);
        tSizeVec[id] = tSnpData.size();
        

        // Compute total sum and nmiss
#pragma omp critical
        {
            sumi  += itsum;
            nmiss += itmiss;
        }

#pragma omp barrier

        // Then compute the mean
#pragma omp single 
        {
            mean = float(sumi) / float(numKeptInds-nmiss);
        }

#pragma omp barrier

        // Concatenate to output vector
        int absi = 0;
        for (int j=0; j<id; ++j)
            absi += tSizeVec[j];

        for ( auto &it : tSnpData ) {
            if (nmiss && it == -9) {
                snpData[absi] = mean;
            } else {
                snpData[absi] = it;
            }
            ++absi;
        }
    }

    if (munmap(addr, bytesToMap) == -1)
        handle_error("munmap");
    
    if (close(fd) == -1)
        handle_error("closing bedFile");

    /*
    if (snpInd%100 == 0)
        printf("MARKER %6d mean = %12.7f computed on %6d with %6d elements (%d - %d)\n", snpInd, mean, sumi, numKeptInds-nmiss, numKeptInds, nmiss);
    */
 
    snpData.array() -= mean;

    float sqn = snpData.squaredNorm();
    float std_ = 1.f / (sqrt(sqn / (float(numKeptInds)-1.0)));
    snpData.array() *= std_;

    //ZPZdiag[snpInd]  = sqn;
    //We are using the squared norm of the already centered and scaled column
    ZPZdiag[snpInd]  = (float(numKeptInds)-1.0);
    //ZPZdiag[snpInd]  =snpData.squaredNorm() ;
}

void Data::getSnpDataFromBedFileUsingMmap(const string &bedFile, const size_t snpLenByt, const long memPageSize, const uint snpInd, VectorXf &snpData) {

    struct stat sb;

    snpData.resize(numKeptInds);
    ZPZdiag.resize(numIncdSnps);

    //EO: check how this should be dealt with
    //
    // Early return if SNP is to be ignored
    if (!snpInfoVec[snpInd]->included)
        return;

    int fd = open(bedFile.c_str(), O_RDONLY);
    if (fd == -1)
        handle_error("opening bedFile");

    if (fstat(fd, &sb) == -1)
        handle_error("fstat");
    
    if (!S_ISREG(sb.st_mode))
        handle_error("Not a regular file");
  
    off_t offset    = 3 + snpInd * snpLenByt;
    off_t pa_offset = offset & ~(memPageSize - 1);

    size_t relOffset  = offset - pa_offset;
    size_t bytesToMap = relOffset + snpLenByt;


    char  *addr = static_cast<char*>(mmap(0, bytesToMap, PROT_READ, MAP_PRIVATE, fd, pa_offset));  
    if (addr == MAP_FAILED)
        handle_error("mmap failed");
  
    int   nmiss = 0;
    float mean  = 0.f;
    int   sumi  = 0;

    unsigned allele1=0, allele2=0;
    bitset<8> b;
    int ii    = 0;

    for(int i=0; i<numInds; i+=4) {
        
        b = addr[relOffset + i/4];

        for (int k=0; k<4 && i+k<numInds; ++k) {

            if (indInfoVec[i+k]->kept) {
                allele1 = (!b[2*k]);
                allele2 = (!b[2*k+1]);
                if (allele1 == 0 && allele2 == 1) {  // missing genotype
                    snpData[ii] = -9;
                    ++nmiss;
                } else {
                    snpData[ii] = allele1 + allele2;
                    sumi += snpData[ii];
                }
                ++ii;
            }
        }
    }

    mean = float(sumi) / float(numKeptInds-nmiss);
    
    if (nmiss) {
        for (int i=0; i<numKeptInds; ++i) {
            if (snpData[i] == -9) {
                snpData[i] = mean;
            }
        }
    }


    if (munmap(addr, bytesToMap) == -1)
        handle_error("munmap");
    
    if (close(fd) == -1)
        handle_error("closing bedFile");

    /*
    if (snpInd%100 == 0)
        printf("MARKER %6d mean = %12.7f computed on %6d with %6d elements (%d - %d)\n", snpInd, mean, sumi, numKeptInds-nmiss, numKeptInds, nmiss);
    */

    snpData.array() -= mean;

    float sqn = snpData.squaredNorm();
    float std_ = 1.f / (sqrt(sqn / float(numKeptInds))); // assume full pop.
    snpData.array() *= std_;

    ZPZdiag[snpInd]  = sqn;
}


void Data::readBedFile_noMPI(const string &bedFile){
    unsigned i = 0, j = 0, k = 0;
    
    if (numIncdSnps == 0) throw ("Error: No SNP is retained for analysis.");
    if (numKeptInds == 0) throw ("Error: No individual is retained for analysis.");
    
    Z.resize(numKeptInds, numIncdSnps);
    ZPZdiag.resize(numIncdSnps);
    snp2pq.resize(numIncdSnps);
    
    // Read bed file
    char ch[1];
    bitset<8> b;
    unsigned allele1=0, allele2=0;
    ifstream BIT(bedFile.c_str(), ios::binary);
    if (!BIT) throw ("Error: can not open the file [" + bedFile + "] to read.");
    cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    for (i = 0; i < 3; i++) BIT.read(ch, 1); // skip the first three bytes
    SnpInfo *snpInfo = NULL;
    unsigned snp = 0, ind = 0;
    unsigned nmiss = 0;
    float mean = 0.0;

    for (j = 0, snp = 0; j < numSnps; j++) { // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 10: hetezygote; 01: missing
        snpInfo = snpInfoVec[j];
        mean = 0.0;
        nmiss = 0;
        if (!snpInfo->included) {
            for (i = 0; i < numInds; i += 4) BIT.read(ch, 1);
            continue;
        }
        for (i = 0, ind = 0; i < numInds;) {
            BIT.read(ch, 1);
            if (!BIT) throw ("Error: problem with the BED file ... has the FAM/BIM file been changed?");
            b = ch[0];
            k = 0;
            while (k < 7 && i < numInds) {
                if (!indInfoVec[i]->kept) k += 2;
                else {
                    allele1 = (!b[k++]);
                    allele2 = (!b[k++]);
                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        Z(ind++, snp) = -9;
                        ++nmiss;
                    } else {
                        mean += Z(ind++, snp) = allele1 + allele2;
                    }
                }
                i++;
            }
        }

        // fill missing values with the mean
        float sum = mean;
        mean /= float(numKeptInds-nmiss);

        /*
        if (j%100 == 0) {
            printf("MARKER %6d mean = %12.7f computed on %6.0f with %6d elements (%d - %d)\n", j, mean, sum, numKeptInds-nmiss, numKeptInds, nmiss);
            fflush(stdout);
        }
        */

        if (nmiss) {
            for (i=0; i<numKeptInds; ++i) {
                if (Z(i,snp) == -9) Z(i,snp) = mean;
            }
        }
        
        /*
        if (j%100 == 0)
            printf("mean vs mean %13.7f %13.7f sum = %20.7f nmiss=%d\n", mean, Z.col(j).mean(), Z.col(j).sum(), nmiss);
        */


        // compute allele frequency
        snpInfo->af = 0.5f*mean;
        snp2pq[snp] = 2.0f*snpInfo->af*(1.0f-snpInfo->af);

        //cout << "snp " << snp << "     " << Z.col(snp).sum() << endl;

        // standardize genotypes
        Z.col(j).array() -= mean;

        float sqn = Z.col(j).squaredNorm();
        float std_ = 1.f / (sqrt(sqn / float(numKeptInds)));
        Z.col(j).array() *= std_;

        ZPZdiag[j] = sqn;

        if (++snp == numIncdSnps) break;
    }

    BIT.clear();
    BIT.close();
    // standardize genotypes
    for (i=0; i<numIncdSnps; ++i) {
        Z.col(i).array() -= Z.col(i).mean();
        //if (i%10 == 0)
        //  printf("marker %2d new mean = %13.6E computed with %d elements (%d - %d)\n", i, Z.col(i).mean(), Z.col(i).size());
        //Z.col(i).array() /= sqrtf(Gadget::calcVariance(Z.col(i))*numKeptInds);
        ZPZdiag[i] = Z.col(i).squaredNorm();
    }
   


    //cout << "Z" << endl << Z << endl;
    
    cout << "Genotype data for " << numKeptInds << " individuals and " << numIncdSnps << " SNPs are included from [" + bedFile + "]." << endl;
}

void Data::readBedFile(const string &bedFile){
    unsigned i = 0, j = 0;
    
    if (numIncdSnps == 0) throw ("Error: No SNP is retained for analysis.");
    if (numKeptInds == 0) throw ("Error: No individual is retained for analysis.");
    
    Z.resize(numKeptInds, numIncdSnps);
    ZPZdiag.resize(numIncdSnps);
    snp2pq.resize(numIncdSnps);
    
    // Read bed file
    ifstream in(bedFile.c_str(), ios::binary);
    if (!in) throw ("Error: can not open the file [" + bedFile + "] to read.");
    if (myMPI::rank==0)
        cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    char header[3];
    in.read((char *) header, 3);
    if (!in || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01) {
        cerr << "Error: Incorrect first three bytes of bed file: " << bedFile << endl;
        exit(1);
    }

    unsigned numKeptInds_all;
    MPI_Allreduce(&numKeptInds, &numKeptInds_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    
    // Read genotypes
    SnpInfo *snpInfo = NULL;
    IndInfo *indInfo = NULL;
    unsigned snp = 0;
    unsigned nmiss=0, nmiss_all;
    float sum=0.0, sum_all=0.0, mean_all;
    const int bedToGeno[4] = {2, -9, 1, 0};
    int genoValue;
    for (j = 0, snp = 0; j < numSnps; j++) {  // code adopted from BOLT-LMM with modification
        snpInfo = snpInfoVec[j];
        sum = 0.0;
        nmiss = 0;
        
        unsigned size = (numInds+3)>>2;
        
        if (!snpInfo->included) {
            in.ignore(size);
            continue;
        }
 
        char *bedLineIn = new char[size];
        in.read((char *)bedLineIn, size);

        for (i = 0; i < numInds; i++) {
            indInfo = indInfoVec[i];
            if (!indInfo->kept) continue;
            genoValue = bedToGeno[(bedLineIn[i>>2]>>((i&3)<<1))&3];
            
            Z(indInfo->index, snp) = genoValue;
            if (genoValue == -9) ++nmiss;   // missing genotype
            else sum += genoValue;
        }
        delete[] bedLineIn;
        
        MPI_Allreduce(&sum, &sum_all, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&nmiss, &nmiss_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        
        // fill missing values with the mean
        mean_all = sum_all/float(numKeptInds_all - nmiss_all);
        if (nmiss) {
            for (i=0; i<numKeptInds; ++i) {
                if (Z(i,snp) == -9) Z(i,snp) = mean_all;
            }
        }
        
        // compute allele frequency
        snpInfo->af = 0.5f*mean_all;
        snp2pq[snp] = 2.0f*snpInfo->af*(1.0f-snpInfo->af);
        
        //cout << "snp " << snp << "     " << Z.col(snp).sum() << endl;

        if (++snp == numIncdSnps) break;
    }
    in.close();

    
    // standardize genotypes
    VectorXf colsums = Z.colwise().sum();
    VectorXf colsums_all(numIncdSnps);

    MPI_Allreduce(&colsums[0], &colsums_all[0], numIncdSnps, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    Z.rowwise() -= colsums_all.transpose()/numKeptInds_all;  // center
    VectorXf my_ZPZdiag = Z.colwise().squaredNorm();
    
    MPI_Allreduce(&my_ZPZdiag[0], &ZPZdiag[0], numIncdSnps, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    if (myMPI::rank==0)
        cout << "Genotype data for " << numKeptInds_all << " individuals and " << numIncdSnps << " SNPs are included from [" + bedFile + "]." << endl;
}

void Data::readPhenotypeFile(const string &phenFile, const unsigned mphen) {
    // NA: missing phenotype
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
    if (myMPI::rank==0)
        cout << "Reading phenotypes from [" + phenFile + "]." << endl;
    map<string, IndInfo*>::iterator it, end=indInfoMap.end();
    IndInfo *ind = NULL;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    string id;
    unsigned line=0;
    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        id = colData[0] + ":" + colData[1];
        it = indInfoMap.find(id);
        if (it != end && colData[mphen+1] != "NA") {
            ind = it->second;
            ind->phenotype = atof(colData[mphen+1].c_str());
            ++line;
        }
    }
    in.close();
    if (myMPI::rank==0)
        cout << "Non-missing phenotypes of trait " << mphen << " of " << line << " individuals are included from [" + phenFile + "]." << endl;
}

void Data::keepMatchedInd(const string &keepIndFile, const unsigned keepIndMax){  // keepIndFile is optional
    map<string, IndInfo*>::iterator it, end=indInfoMap.end();
    IndInfo *ind = NULL;
    vector<string> keep;
    keep.reserve(numInds);
    unsigned cnt=0;
    for (unsigned i=0; i<numInds; ++i) {
        ind = indInfoVec[i];
        ind->kept = false;
        //We get rid of this misterious -9 condition
        //if (ind->phenotype!=-9) {
            if (keepIndMax > cnt++)
                keep.push_back(ind->catID);
        //}
    }
    
    if (!keepIndFile.empty()) {
        ifstream in(keepIndFile.c_str());
        if (!in) throw ("Error: can not open the file [" + keepIndFile + "] to read.");
        string fid, pid;
        keep.clear();
        while (in >> fid >> pid) {
            keep.push_back(fid + ":" + pid);
        }
        in.close();
    }
    
    unsigned numKeptInds_all = 0;

    if (myMPI::partition == "byrow") {
        unsigned total_size = (unsigned) keep.size();
        unsigned batch_size = total_size/myMPI::clusterSize;
        unsigned my_start = myMPI::rank*batch_size;
        unsigned my_end = (myMPI::rank+1)==myMPI::clusterSize ? total_size : my_start + batch_size;
        unsigned my_size = my_end - my_start;
                
        myMPI::iStart = my_start;
        myMPI::iSize  = my_size;
        
        vector<string>::const_iterator first = keep.begin() + my_start;
        vector<string>::const_iterator last  = keep.begin() + my_end;
        vector<string> my_keep(first, last);
        
        for (unsigned i=0; i<my_size; ++i) {
            it = indInfoMap.find(my_keep[i]);
            if (it == end) {
                Gadget::Tokenizer token;
                token.getTokens(my_keep[i], ":");
                throw("Error: Individual " + token[0] + " " + token[1] + " from file [" + keepIndFile + "] does not exist!");
            } else {
                ind = it->second;
                //if (ind->phenotype != -9) {
                    ind->kept = true;
                //} else {
                  //  throw("Error: Individual " + ind->famID + " " + ind->indID + " from file [" + keepIndFile + "] does not have phenotype!");
                //}
            }
        }
        
        keptIndInfoVec = makeKeptIndInfoVec(indInfoVec);
        numKeptInds =  (unsigned) keptIndInfoVec.size();
        
        y.setZero(numKeptInds);
        for (unsigned i=0; i<numKeptInds; ++i) {
            y[i] = keptIndInfoVec[i]->phenotype;
        }
        
        float my_ypy = ((y.array()-y.mean()).square()).sum();

        MPI_Allreduce(&my_ypy, &ypy, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        MPI_Allreduce(&numKeptInds, &numKeptInds_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        
        if (myMPI::rank==0) {
            cout << numKeptInds_all << " matched individuals are kept." << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        printf("%d individuals assigned to processor %s at rank %d.\n", numKeptInds, myMPI::processorName, myMPI::rank);
        //cout << numKeptInds << " individuals assigned to processor " << myMPI::processorName << " at rank " << myMPI::rank << "." << endl;
    }
    else {
        for (unsigned i=0; i<keep.size(); ++i) {
            it = indInfoMap.find(keep[i]);
            if (it == end) {
                Gadget::Tokenizer token;
                token.getTokens(keep[i], ":");
                throw("Error: Individual " + token[0] + " " + token[1] + " from file [" + keepIndFile + "] does not exist!");
            } else {
                ind = it->second;
                if (ind->phenotype != -9) {
                    ind->kept = true;
                } else {
                    throw("Error: Individual " + ind->famID + " " + ind->indID + " from file [" + keepIndFile + "] does not have phenotype!");
                }
            }
        }
        
        keptIndInfoVec = makeKeptIndInfoVec(indInfoVec);
        numKeptInds =  (unsigned) keptIndInfoVec.size();
        numKeptInds_all = numKeptInds;
        
        y.setZero(numKeptInds);
        for (unsigned i=0; i<numKeptInds; ++i) {
            y[i] = keptIndInfoVec[i]->phenotype;
        }
        ypy = (y.array()-y.mean()).square().sum();
        
        if (myMPI::rank==0) {
            cout << numKeptInds << " matched individuals are kept." << endl;
        }
    }
}

void Data::readCovariateFile(const string &covarFile){
    if (!covarFile.empty()) {
        ifstream in(covarFile.c_str());
        if (!in) throw ("Error: can not open the file [" + covarFile + "] to read.");
        map<string, IndInfo*>::iterator it, end=indInfoMap.end();
        IndInfo *ind = NULL;
        Gadget::Tokenizer colData;
        string inputStr;
        string sep(" \t");
        string id;
        unsigned line=0;
        unsigned numCovariates=0;
        while (getline(in,inputStr)) {
            colData.getTokens(inputStr, sep);
            if (line==0) {
                numCovariates = (unsigned)colData.size() - 2;
                numFixedEffects = numCovariates + 1;
                fixedEffectNames.resize(numFixedEffects);
                fixedEffectNames[0] = "Intercept";
                for (unsigned i=0; i<numCovariates; ++i)
                    fixedEffectNames[i+1] = colData[i+2];
            }
            id = colData[0] + ":" + colData[1];
            it = indInfoMap.find(id);
            if (it != end) {
                ind = it->second;
                ind->covariates.resize(numCovariates + 1);  // plus intercept
                ind->covariates[0] = 1;
                for (unsigned i=2; i<colData.size(); ++i) {
                    ind->covariates[i-1] = atof(colData[i].c_str());
                }
                ++line;
            }
        }
        in.close();
        
        if (myMPI::rank==0)
            cout << "Read " << numCovariates << " covariates from [" + covarFile + "]." << endl;
        
        X.resize(numKeptInds, numFixedEffects);
        for (unsigned i=0; i<numKeptInds; ++i) {
            X.row(i) = keptIndInfoVec[i]->covariates;
        }
        VectorXf my_XPXdiag = X.colwise().squaredNorm();
        XPXdiag.setZero(numFixedEffects);
        MPI_Allreduce(&my_XPXdiag[0], &XPXdiag[0], numFixedEffects, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    else {
        // only intercept for now
        numFixedEffects = 1;
        fixedEffectNames = {"Intercept"};
        X.setOnes(numKeptInds,1);
        XPX.resize(1,1);
        XPXdiag.resize(1);
        XPy.resize(1);
        
        if (myMPI::partition == "byrow") {
            unsigned numKeptInds_all;
            MPI_Allreduce(&numKeptInds, &numKeptInds_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            
            XPX << numKeptInds_all;
            XPXdiag << numKeptInds_all;
            
            float sum = y.sum();
            unsigned sum_all;
            MPI_Allreduce(&sum, &sum_all, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            XPy << sum_all;
        }
        else {
            XPX << numKeptInds;
            XPXdiag << numKeptInds;
            XPy << y.sum();
        }
    }
}

void Data::includeSnp(const string &includeSnpFile){
    ifstream in(includeSnpFile.c_str());
    if (!in) throw ("Error: can not open the file [" + includeSnpFile + "] to read.");
    for (unsigned i=0; i<numSnps; ++i) {
        snpInfoVec[i]->included = false;
    }
    map<string, SnpInfo*>::iterator it, end = snpInfoMap.end();
    string id;
    while (in >> id) {
        it = snpInfoMap.find(id);
        if (it != end) {
            it->second->included = true;
        }
    }
    in.close();
}

void Data::excludeSnp(const string &excludeSnpFile){
    ifstream in(excludeSnpFile.c_str());
    if (!in) throw ("Error: can not open the file [" + excludeSnpFile + "] to read.");
    map<string, SnpInfo*>::iterator it, end = snpInfoMap.end();
    string id;
    while (in >> id) {
        it = snpInfoMap.find(id);
        if (it != end) {
            it->second->included = false;
        }
    }
    in.close();
}

void Data::includeChr(const unsigned chr){
    if (!chr) return;
    for (unsigned i=0; i<numSnps; ++i){
        SnpInfo *snp = snpInfoVec[i];
        if (snp->chrom != chr) snp->included = false;
    }
}

void Data::reindexSnp(vector<SnpInfo*> snpInfoVec){
    SnpInfo *snp;
    for (unsigned i=0, idx=0; i<snpInfoVec.size(); ++i) {
        snp = snpInfoVec[i];
        if (snp->included) {
            snp->index = idx++;
        } else {
            snp->index = -9;
        }
    }
}

void Data::includeMatchedSnp(){
    reindexSnp(snpInfoVec);  // reindex for MPI purpose in terms of full snplist
    fullSnpFlag.resize(numSnps);
    for (int i=0; i<numSnps; ++i) fullSnpFlag[i] = snpInfoVec[i]->included; // for output purpose


    if (myMPI::partition == "bycol") {  // MPI by chromosome
        vector<int> chromVec;
        for (set<int>::iterator it=chromosomes.begin(); it!=chromosomes.end(); ++it) {
            chromVec.push_back(*it);
        }
        unsigned my_chrom = (myMPI::rank+1)==myMPI::clusterSize ? 99 : chromVec[myMPI::rank];

        SnpInfo *snp;
        for (unsigned i=0; i<numSnps; ++i) {  // each core takes one chromosome, if #chrom > #core, the last core takes all rest chroms
            snp = snpInfoVec[i];
            if (my_chrom != 99) {
                if (snp->chrom != my_chrom) snp->included = false;
            } else {
                if (snp->chrom < chromVec[myMPI::rank]) snp->included = false;
            }
        }

        incdSnpInfoVec = makeIncdSnpInfoVec(snpInfoVec);
        numIncdSnps = (unsigned) incdSnpInfoVec.size();
        snp2pq.resize(numIncdSnps);

        // setup MPI for collecting SNP info
        myMPI::iSize = numIncdSnps;
        myMPI::iStart = incdSnpInfoVec[0]->index;
        myMPI::srcounts.resize(myMPI::clusterSize);
        myMPI::displs.resize(myMPI::clusterSize);

        MPI_Allgather(&myMPI::iSize, 1, MPI_INT, &myMPI::srcounts[0], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&myMPI::iStart, 1, MPI_INT, &myMPI::displs[0], 1, MPI_INT, MPI_COMM_WORLD);


        reindexSnp(incdSnpInfoVec);  // reindex based on the snplist for each core

        //cout << "rank " << myMPI::rank << " " << myMPI::iStart << " " << myMPI::iSize << endl;

        unsigned numIncdSnps_all;
        MPI_Reduce(&numIncdSnps, &numIncdSnps_all, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myMPI::rank==0) cout << numIncdSnps_all << " SNPs are included." << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        cout << numIncdSnps << " SNPs assigned to processor " << myMPI::processorName << " at rank " << myMPI::rank << "." << endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        incdSnpInfoVec = makeIncdSnpInfoVec(snpInfoVec);
        numIncdSnps = (unsigned) incdSnpInfoVec.size();
        reindexSnp(incdSnpInfoVec);
        snp2pq.resize(numIncdSnps);
    }

    map<int, vector<SnpInfo*> > chrmap;
    map<int, vector<SnpInfo*> >::iterator it;
    for (unsigned i=0; i<numIncdSnps; ++i) {
        SnpInfo *snp = incdSnpInfoVec[i];
        if (chrmap.find(snp->chrom) == chrmap.end()) {
            chrmap[snp->chrom] = *new vector<SnpInfo*>;
        }
        chrmap[snp->chrom].push_back(snp);
    }
    numChroms = (unsigned) chrmap.size();
    chromInfoVec.clear();
    for (it=chrmap.begin(); it!=chrmap.end(); ++it) {
        int id = it->first;
        vector<SnpInfo*> &vec = it->second;
        ChromInfo *chr = new ChromInfo(id, (unsigned)vec.size(), vec[0]->index, vec.back()->index);
        chromInfoVec.push_back(chr);
        //cout << "size chrom " << id << ": " << vec.back()->physPos - vec[0]->physPos << endl;
    }

    if (myMPI::rank==0) cout << numIncdSnps << " SNPs on " << numChroms << " chromosomes are included." << endl;
}

vector<SnpInfo*> Data::makeIncdSnpInfoVec(const vector<SnpInfo*> &snpInfoVec){
    vector<SnpInfo*> includedSnps;
    includedSnps.reserve(numSnps);
    snpEffectNames.reserve(numSnps);
    SnpInfo *snp = NULL;
    for (unsigned i=0; i<numSnps; ++i) {
        snp = snpInfoVec[i];
        if(snp->included) {
            //snp->index = j++;  // reindex snps
            includedSnps.push_back(snp);
            snpEffectNames.push_back(snp->ID);
        }
    }
    return includedSnps;
}
vector<IndInfo*> Data::makeKeptIndInfoVec(const vector<IndInfo*> &indInfoVec){
    vector<IndInfo*> keptInds;
    keptInds.reserve(numInds);
    IndInfo *ind = NULL;
    for (unsigned i=0, j=0; i<numInds; ++i) {
        ind = indInfoVec[i];
        if(ind->kept) {
            ind->index = j++;  // reindex inds
            keptInds.push_back(ind);
        }
    }
    return keptInds;
}

//TODO Finish function to read the group file
void Data::readGroupFile(const string &groupFile) {
    // NA: missing phenotype
    ifstream in(groupFile.c_str());
    if (!in) throw ("Error: can not open the group file [" + groupFile + "] to read.");
    if (myMPI::rank==0)
        cout << "Reading groups from [" + groupFile + "]." << endl;

    std::istream_iterator<double> start(in), end;
    std::vector<int> numbers(start, end);
    int* ptr =(int*)&numbers[0];
    G=(Eigen::Map<Eigen::VectorXi>(ptr,numbers.size()));

    if (myMPI::rank==0)
        cout << "Groups read from file" << endl;
}


void Data::getSnpDataFromBedFileUsingMmap_new(const int fd, const size_t nb, const long memPageSize, const uint snpInd, double * __restrict__ snpDat) {

    struct stat sb;

    // Early return if SNP is to be ignored
    if (!snpInfoVec[snpInd]->included) return;

    off_t  offset     = 3 + snpInd * nb;
    off_t  pa_offset  = offset & ~(memPageSize - 1);
    size_t relOffset  = offset - pa_offset;
    size_t bytesToMap = relOffset + nb;

    char  *addr = static_cast<char*>(mmap(0, bytesToMap, PROT_READ, MAP_SHARED, fd, pa_offset));  
    if (addr == MAP_FAILED)
        handle_error("mmap failed");

  
    int  nmiss = 0;
    int  sum   = 0;
    char *addr2 = &addr[relOffset];

    int8_t *data = (int8_t*)_mm_malloc(nb * 4 * sizeof(char), 64);

    for (int i=0; i<nb; ++i) {
        for (int ii=0; ii<4; ++ii) {
            data[i*4+ii] = (addr2[i] >> 2*ii) & 0b11;
        }
    }

    for (int i=0; i<numInds; ++i) {
        if (data[i] == 1) {
            data[i] = -1;
        } else {
            data[i] = 2 - ((data[i] & 0b1) + ((data[i] >> 1) & 0b1));
        }
    }

#pragma omp simd reduction(+:sum) reduction(+:nmiss)
    for (int i=0; i<numInds; ++i) {
        if (data[i] < 0) {
            nmiss += data[i];
        } else {
            sum += data[i];
        }
    }
    //printf("sum = %d, N = %d, nmiss = %d\n", sum, numKeptInds, nmiss);
    double mean = double(sum) / double(numKeptInds + nmiss); //EO: nmiss is neg
    //printf("mean = %20.15f\n", mean);


    for (int i=0; i<numKeptInds; ++i) {
        if (data[i] < 0) {
            snpDat[i] = 0.0d;
        } else {
            snpDat[i] = double(data[i]) - mean;
        }
    }

    double sqn  = 0.0d;
    for (int i=0; i<numKeptInds; ++i) {
        sqn  += snpDat[i] * snpDat[i];
    }

    double std_ = sqrt(double(numKeptInds -1) / sqn);
 
    for (int i=0; i<numKeptInds; ++i) {
        snpDat[i] *= std_;
    }

    if (munmap(addr, bytesToMap) == -1)
        handle_error("munmap");
    
    _mm_free(data);

    //ZPZdiag[snpInd]  = sqn;
    //EO: sp -> dp?
    ZPZdiag[snpInd]  = (float(numKeptInds)-1.0);
}


// Overloaded function
void Data::getSnpDataFromBedFileUsingMmap_new(const int fd, const size_t nb, const long memPageSize, const uint snpInd, VectorXd &snpDat) {

    struct stat sb;

    // Early return if SNP is to be ignored
    if (!snpInfoVec[snpInd]->included) return;

    off_t  offset     = 3 + snpInd * nb;
    off_t  pa_offset  = offset & ~(memPageSize - 1);
    size_t relOffset  = offset - pa_offset;
    size_t bytesToMap = relOffset + nb;

    char  *addr = static_cast<char*>(mmap(0, bytesToMap, PROT_READ, MAP_SHARED, fd, pa_offset));  
    if (addr == MAP_FAILED)
        handle_error("mmap failed");

  
    int  nmiss = 0;
    int  sum   = 0;
    char *addr2 = &addr[relOffset];

    int8_t *data = (int8_t*)_mm_malloc(nb * 4 * sizeof(char), 64);

    for (int i=0; i<nb; ++i) {
        for (int ii=0; ii<4; ++ii) {
            data[i*4+ii] = (addr2[i] >> 2*ii) & 0b11;
        }
    }

    for (int i=0; i<numInds; ++i) {
        if (data[i] == 1) {
            data[i] = -1;
        } else {
            data[i] = 2 - ((data[i] & 0b1) + ((data[i] >> 1) & 0b1));
        }
    }

#pragma omp simd reduction(+:sum) reduction(+:nmiss)
    for (int i=0; i<numInds; ++i) {
        if (data[i] < 0) {
            nmiss += data[i];
        } else {
            sum += data[i];
        }
    }
    //printf("sum = %d, N = %d, nmiss = %d\n", sum, numKeptInds, nmiss);
    double mean = double(sum) / double(numKeptInds + nmiss); //EO: nmiss is neg
    //printf("mean = %20.15f\n", mean);

    for (int i=0; i<numKeptInds; ++i) {
        if (data[i] < 0) {
            snpDat[i] = 0.0d;
        } else {
            snpDat[i] = double(data[i]) - mean;
        }
    }

    double sqn  = 0.0d;
    for (int i=0; i<numKeptInds; ++i) {
        sqn  += snpDat[i] * snpDat[i];
    }

    double std_ = sqrt(double(numKeptInds - 1) / sqn);
 
    for (int i=0; i<numKeptInds; ++i) {
        snpDat[i] *= std_;
    }

    if (munmap(addr, bytesToMap) == -1)
        handle_error("munmap");

    _mm_free(data);

    //ZPZdiag[snpInd]  = sqn;
    //EO: sp -> dp?
    ZPZdiag[snpInd]  = (float(numKeptInds)-1.0);
}
