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


#define handle_error(msg)                               \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)


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
void Data::getSnpDataFromBedFileUsingMmap(const string &bedFile, const size_t snpLenByt, const long memPageSize, const uint snpInd, VectorXf &snpData) {

    struct stat sb;

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
  
    // Index of first byte to read
    unsigned ir = relOffset;

    int   nmiss = 0;
    float mean  = 0.f;
    int   sumi  = 0;

    int N = omp_get_max_threads();
    //if (snpInd%100 == 0)
    //    cout << "Max number of frame N = " << N << endl;
    
    vector<vector<int>> snpDataVec(N);

    int Nmax = numInds%4 ? numInds/4 + 1 : numInds/4;
    //printf("Nmax = %d\n", Nmax);
    if (Nmax < N)
        omp_set_num_threads(Nmax);

#pragma omp parallel
    {
        int id, i, Nthrds, istart, iend;
        id = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        //if (snpInd%100 == 0 && id == 0)
        //    cout << "Will use N = " << N << " threads" << endl;
        istart =  id    * (numInds/4) / Nthrds * 4;
        iend   = (id+1) * (numInds/4) / Nthrds * 4;
        if (id == Nthrds-1) iend = numInds;

        snpDataVec[id].reserve((iend-istart)*4+1);

        unsigned allele1=0, allele2=0;
        bitset<8> b;
        uint tmiss = 0;
        uint tsum  = 0;

        for(i=istart; i<iend; i+=4) {
            
            b = addr[relOffset + i/4];

            for (uint k=0; k<4 && i+k<iend; ++k) {

                if (indInfoVec[i+k]->kept) {
                    allele1 = (!b[2*k]);
                    allele2 = (!b[2*k+1]);
                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        snpDataVec[id].push_back(-9);
                    } else {
                        snpDataVec[id].push_back(allele1 + allele2);
                    }
                }
            }
        }
    }


    if (munmap(addr, bytesToMap) == -1)
        handle_error("munmap");
    
    if (close(fd) == -1)
        handle_error("closing bedFile");

#pragma omp parallel for reduction(+:sumi, nmiss)
    for (int i=0; i<N; ++i) {
        for ( auto &it : snpDataVec[i] ) {
            if (it == -9) {
                ++nmiss;
            } else {
                sumi += it;
            }
        }
    }

    mean = float(sumi) / float(numKeptInds-nmiss);

    //if (snpInd%100 == 0)
    //    printf("marker %6d mean = %12.7f computed with sumi = %6d on %6d elements (%d - %d)\n", snpInd, mean, sumi, numKeptInds-nmiss, numKeptInds, nmiss);
    
    snpData.resize(numKeptInds);

    float c2 = 0.f;

#pragma omp parallel for reduction(+:c2) 
    for (int i=0; i<N; ++i) {
        
        int absi = 0;
        for (int j=0; j<i; ++j)
            absi += snpDataVec[j].size();

        for ( auto &it : snpDataVec[i] ) {

            if (nmiss && it == -9) {
                snpData(absi) = mean;
            } else {
                snpData(absi) = it;
            }

            c2 += snpData(absi);
            ++absi;
        }
    }

    
    //if (snpInd%100 == 0)
    //    printf("mean vs mean %13.7f %13.7f sum = %20.7f %13.7f nmiss=%d\n", mean, snpData.mean(), c2, c2/float(numKeptInds), nmiss);

    snpData.array() -= snpData.mean();
    ZPZdiag[snpInd]  = snpData.squaredNorm();
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

    
        //if (j%100 == 0)
        //    printf("MARKER %6d mean = %12.7f computed on %6.0f with %6d elements (%d - %d)\n", j, mean, sum, numKeptInds-nmiss, numKeptInds, nmiss);

        if (nmiss) {
            for (i=0; i<numKeptInds; ++i) {
                if (Z(i,snp) == -9) Z(i,snp) = mean;
            }
        }
        
        //if (j%100 == 0)
        //    printf("mean vs mean %13.7f %13.7f sum = %20.7f nmiss=%d\n", mean, Z.col(j).mean(), Z.col(j).sum(), nmiss);



        // compute allele frequency
        snpInfo->af = 0.5f*mean;
        snp2pq[snp] = 2.0f*snpInfo->af*(1.0f-snpInfo->af);

        //cout << "snp " << snp << "     " << Z.col(snp).sum() << endl;

        if (++snp == numIncdSnps) break;
    }
    BIT.clear();
    BIT.close();
    
    
    // standardize genotypes
    for (i=0; i<numIncdSnps; ++i) {
        Z.col(i).array() -= Z.col(i).mean();
        //if (i%10 == 0)
        //  printf("marker %2d new mean = %13.6E computed with %d elements (%d - %d)\n", i, Z.col(i).mean(), Z.col(i).size());
        //Z.col(i).array() /= sqrtf(gadgets::calcVariance(Z.col(i))*numKeptInds);
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
        if (ind->phenotype!=-9) {
            if (keepIndMax > cnt++)
                keep.push_back(ind->catID);
        }
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
                if (ind->phenotype != -9) {
                    ind->kept = true;
                } else {
                    throw("Error: Individual " + ind->famID + " " + ind->indID + " from file [" + keepIndFile + "] does not have phenotype!");
                }
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

void Data::initVariances(const float heritability){
    float varPhenotypic;
    if (myMPI::partition == "byrow") {
        unsigned numKeptInds_all = 0;
        MPI_Allreduce(&numKeptInds, &numKeptInds_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        varPhenotypic = ypy/numKeptInds_all;
    } else {
        varPhenotypic = ypy/numKeptInds;
    }
    varGenotypic = varPhenotypic * heritability;
    varResidual  = varPhenotypic - varGenotypic;
    //cout <<varPhenotypic<<" " <<varGenotypic << " " <<varResidual << endl;
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

void Data::computeAlleleFreq(const MatrixXf &Z, vector<SnpInfo*> &incdSnpInfoVec, VectorXf &snp2pq){
    if (myMPI::rank==0)
        cout << "Computing allele frequencies ..." << endl;
    snp2pq.resize(numIncdSnps);
    SnpInfo *snp = NULL;
    for (unsigned i=0; i<numIncdSnps; ++i) {
        snp = incdSnpInfoVec[i];
        snp->af = 0.5f*Z.col(i).mean();
        snp2pq[i] = 2.0f*snp->af*(1.0f-snp->af);
    }
}

void Data::getWindowInfo(const vector<SnpInfo*> &incdSnpInfoVec, const unsigned windowWidth, VectorXi &windStart, VectorXi &windSize){
    if (myMPI::rank==0)
        cout << "Creating windows (window width: " + to_string(static_cast<long long>(windowWidth/1e6)) + "Mb) ..." << endl;
    int i=0, j=0;
    windStart.setZero(numIncdSnps);
    windSize.setZero(numIncdSnps);
    SnpInfo *snpi, *snpj;
    for (i=0; i<numIncdSnps; ++i) {
        snpi = incdSnpInfoVec[i];
        snpi->resetWindow();
        for (j=i; j>=0; --j) {
            snpj = incdSnpInfoVec[j];
            if (snpi->isProximal(*snpj, windowWidth/2)) {
                snpi->windStart = snpj->index;
                snpi->windSize++;
            } else break;
        }
        for (j=i+1; j<numIncdSnps; ++j) {
            snpj = incdSnpInfoVec[j];
            if (snpi->isProximal(*snpj, windowWidth/2)) {
                snpi->windSize++;
            } else break;
        }
        if(!(i%10000) && myMPI::rank==0)
            cout << "SNP " << i << " Window Size " << snpi->windSize << endl;
        if (!snpi->windSize) {
            throw("Error: SNP " + snpi->ID + " has zero SNPs in its window!");
        }
        windStart[i] = snpi->windStart;
        windSize [i] = snpi->windSize;
    }
}

void Data::getNonoverlapWindowInfo(const unsigned windowWidth){
    unsigned window = 0;
    unsigned currChr = incdSnpInfoVec[0]->chrom;
    unsigned long startPos = incdSnpInfoVec[0]->physPos;
    vector<int> windStartVec = {0};
    SnpInfo *snp;
    for (unsigned i=0; i<numIncdSnps; ++i) {
        snp = incdSnpInfoVec[i];
        if (snp->physPos - startPos > windowWidth || snp->chrom > currChr) {
            currChr = snp->chrom;
            startPos = snp->physPos;
            windStartVec.push_back(i);
            ++window;
        }
        snp->window = window;
    }
    
    long numberWindows = windStartVec.size();

    windStart = VectorXi::Map(&windStartVec[0], numberWindows);
    windSize.setZero(numberWindows);

    for (unsigned i=0; i<numberWindows; ++i) {
        if (i != numberWindows-1)
            windSize[i] = windStart[i+1] - windStart[i];
        else
            windSize[i] = numIncdSnps - windStart[i];
    }
    
    if (myMPI::rank==0)
        cout << "Created " << numberWindows << " non-overlapping " << windowWidth/1e3 << "kb windows with average size of " << windSize.sum()/float(numberWindows) << " SNPs." << endl;
}

void Data::outputSnpResults(const VectorXf &posteriorMean, const VectorXf &posteriorSqrMean, const VectorXf &pip, const string &filename) const {
    if (myMPI::rank) return;
    ofstream out(filename.c_str());
    out << boost::format("%6s %20s %6s %12s %8s %12s %12s %8s %8s\n")
        % "Id"
        % "Name"
        % "Chrom"
        % "Position"
        % "GeneFrq"
        % "Effect"
        % "SE"
        % "PIP"
        % "Window";
    for (unsigned i=0, idx=0; i<numSnps; ++i) {
        SnpInfo *snp = snpInfoVec[i];
        if(!fullSnpFlag[i]) continue;
        if(snp->isQTL) continue;
        out << boost::format("%6s %20s %6s %12s %8.6f %12.6f %12.6f %8.3f %8s\n")
            % (idx+1)
            % snp->ID
            % snp->chrom
            % snp->physPos
            % snp->af
            % posteriorMean[idx]
            % sqrt(posteriorSqrMean[idx]-posteriorMean[idx]*posteriorMean[idx])
            % pip[idx]
            % snp->window;
        ++idx;
    }
    out.close();
}

void Data::outputWindowResults(const VectorXf &posteriorMean, const string &filename) const {
    if (myMPI::rank) return;
    ofstream out(filename.c_str());
    out << boost::format("%6s %8s\n") %"Id" %"PIP";
    for (unsigned i=0; i<posteriorMean.size(); ++i) {
        out << boost::format("%6s %8.3f\n")
            % (i+1)
            % posteriorMean[i];
    }
    out.close();
}

void Data::readGwasSummaryFile(const string &gwasFile){
    ifstream in(gwasFile.c_str());
    if (!in) throw ("Error: can not open the GWAS summary data file [" + gwasFile + "] to read.");
    if (myMPI::rank==0)
        cout << "Reading GWAS summary data from [" + gwasFile + "]." << endl;
    
    SnpInfo *snp;
    map<string, SnpInfo*>::iterator it;
    string id, allele1, allele2, freq, b, se, pval, n;
    unsigned line=0, match=0;
    while (in >> id >> allele1 >> allele2 >> freq >> b >> se >> pval >> n) {
        ++line;
        it = snpInfoMap.find(id);
        if (it == snpInfoMap.end()) continue;
        snp = it->second;
        if (!snp->included) continue;
        if (allele1 == snp->a1 && allele2 == snp->a2) {
            snp->gwas_b  = atof(b.c_str());
            snp->gwas_af = atof(freq.c_str());
        } else if (allele1 == snp->a2 && allele2 == snp->a1) {
            snp->gwas_b  = -atof(b.c_str());
            snp->gwas_af = 1.0-atof(freq.c_str());
        } else {
            throw("Error: SNP " + id + " has inconsistent allele coding in between the reference and GWAS samples.");
        }
        snp->gwas_se = atof(se.c_str());
        snp->gwas_n  = atof(n.c_str());
        ++match;
    }
    in.close();
    
    for (unsigned i=0; i<numSnps; ++i) {
        snp = snpInfoVec[i];
        if (!snp->included) continue;
        if (snp->gwas_b == -999) {
            snp->included = false;
        }
    }

    if (myMPI::rank==0)
        cout << match << " matched SNPs in the GWAS summary data (in total " << line << " SNPs)." << endl;

}

void Data::readLDmatrixInfoFile(const string &ldmatrixFile){
    ifstream in(ldmatrixFile.c_str());
    if (!in) throw ("Error: can not open the file [" + ldmatrixFile + "] to read.");
    cout << "Reading SNP info from [" + ldmatrixFile + "]." << endl;
    //snpInfoVec.clear();
    //snpInfoMap.clear();
    string id, allele1, allele2;
    unsigned chr, physPos;
    float genPos;
    unsigned idx, windStart, windSize;
    while (in >> chr >> id >> genPos >> physPos >> allele1 >> allele2 >> idx >> windStart >> windSize) {
        SnpInfo *snp = new SnpInfo(idx, id, allele1, allele2, chr, genPos, physPos);
        snp->windStart = windStart;
        snp->windSize = windSize;
        snpInfoVec.push_back(snp);
        if (snpInfoMap.insert(pair<string, SnpInfo*>(id, snp)).second == false) {
            throw ("Error: Duplicate SNP ID found: \"" + id + "\".");
        }
    }
    in.close();
    numSnps = (unsigned) snpInfoVec.size();
    cout << numSnps << " SNPs to be included from [" + ldmatrixFile + "]." << endl;
}

void Data::readLDmatrixInfoFile(const string &ldmatrixFile, vector<SnpInfo*> &vec){
    ifstream in(ldmatrixFile.c_str());
    if (!in) throw ("Error: can not open the file [" + ldmatrixFile + "] to read.");
    cout << "Reading SNP info from [" + ldmatrixFile + "]." << endl;
    string id, allele1, allele2;
    unsigned chr, physPos;
    float genPos;
    unsigned idx, windStart, windSize;
    while (in >> chr >> id >> genPos >> physPos >> allele1 >> allele2 >> idx >> windStart >> windSize) {
        SnpInfo *snp = new SnpInfo(idx, id, allele1, allele2, chr, genPos, physPos);
        snp->windStart = windStart;
        snp->windSize = windSize;
        snpInfoVec.push_back(snp);
        if (snpInfoMap.insert(pair<string, SnpInfo*>(id, snp)).second == false) {
            throw ("Error: Duplicate SNP ID found: \"" + id + "\".");
        }
        vec.push_back(snp);
    }
    in.close();
    numSnps = (unsigned) snpInfoVec.size();
    cout << numSnps << " SNPs to be included from [" + ldmatrixFile + "]." << endl;
}

void Data::readLDmatrixBinFile(const string &ldmatrixFile){
    
    VectorXi windStartLDM(numSnps);
    VectorXi windSizeLDM(numSnps);
    
    windStart.resize(numIncdSnps);
    windSize.resize(numIncdSnps);
    
    SnpInfo *snpi, *snpj;
    
    for (unsigned i=0, inci=0; i<numSnps; ++i) {
        SnpInfo *snpi = snpInfoVec[i];
        windStartLDM[i] = snpi->windStart;
        windSizeLDM[i]  = snpi->windSize;
        //        if (snpi->included) {
        //            int windStarti=0, windSizei=0;
        //            for (unsigned j=0, incj=0; j<snpi->windSize; ++j) {
        //                snpj = snpInfoVec[snpi->windStart+j];
        //                if (snpj->included) {
        //                    if (!incj) windStarti = snpj->index;
        //                    ++incj;
        //                    ++windSizei;
        //                }
        //            }
        //            snpi->windStart = windStart[inci] = windStarti;
        //            snpi->windSize  = windSize [inci] = windSizei;
        //            ++inci;
        //        } else {
        //            snpi->windStart = -1;
        //            snpi->windSize  = 0;
        //        }
    }
    
    //    cout << "windStartLDM " << windStartLDM.transpose() << endl;
    //    cout << "windSizeLDM " << windSizeLDM.transpose() << endl;
    //    cout << "windStart " << windStart.transpose() << endl;
    //    cout << "windSize " << windSize.transpose() << endl;

    FILE *in = fopen(ldmatrixFile.c_str(), "rb");
    if (!in) {
        throw("Error: cannot open LD matrix file " + ldmatrixFile);
    }
    float windowWidth;
    fread(&windowWidth, sizeof(float), 1, in);
    
    getWindowInfo(incdSnpInfoVec, windowWidth*1e6, windStart, windSize);
    
    if (numIncdSnps == 0) throw ("Error: No SNP is retained for analysis.");
    
    cout << "Reading LD matrix from [" + ldmatrixFile + "]..." << endl;
    
    ZPZ.resize(numIncdSnps);
    for (unsigned i=0; i<numIncdSnps; ++i) {
        ZPZ[i].resize(windSize[i]);
    }

    Gadget::Timer timer;
    timer.setTime();
    
    for (unsigned i = 0, inci = 0; i < numSnps; i++) {
        snpi = snpInfoVec[i];
        
        float v[windSizeLDM[i]];
        
        if (!snpi->included) {
            fseek(in, sizeof(v), SEEK_CUR);
            continue;
        }
        
        fread(v, sizeof(v), 1, in);

        for (unsigned j = 0, incj = 0; j<windSizeLDM[i]; ++j) {
            snpj = snpInfoVec[windStartLDM[i]+j];
            //if (inci==0) cout << inci << " " << j << " " << ZPZ[inci].size() << " " << windStartLDM[i] << " " << windSizeLDM[i] << " " << windSize[inci] << " " << v[j] << " " << snpj->included << endl;
            if (snpj->included) {
                //cout << inci << " " << incj << " " << ZPZ[inci].size() << " " << windStartLDM[i] << " " << windSizeLDM[i] << " " << windSize[inci] << " " << v[j] << endl;
                ZPZ[inci][incj++] = v[j];
            }
        }
        
        if (inci++ == numIncdSnps) break;
    }
    
    fclose(in);
    
    timer.getTime();
    
    cout << "Window width " << windowWidth << " Mb." << endl;
    cout << "Average window size " << windSize.sum()/numIncdSnps << "." << endl;
    cout << "Read LD matrix for " << numIncdSnps << " SNPs (time used: " << timer.format(timer.getElapse()) << ")." << endl;
}

void Data::buildSparseMME(){
    snp2pq.resize(numIncdSnps);
    D.resize(numIncdSnps);
    ZPZdiag.resize(numIncdSnps);
    ZPy.resize(numIncdSnps);
    b.resize(numIncdSnps);
    n.resize(numIncdSnps);
    se.resize(numIncdSnps);
    tss.resize(numIncdSnps);
    SnpInfo *snp;
    for (unsigned i=0; i<numIncdSnps; ++i) {
        snp = incdSnpInfoVec[i];
        snp->af = snp->gwas_af;
        snp2pq[i] = 2.0f*snp->gwas_af*(1.0f-snp->gwas_af);
        D[i] = snp2pq[i]*snp->gwas_n;
        b[i] = snp->gwas_b;
        n[i] = snp->gwas_n;
        se[i]= snp->gwas_se;
        tss[i] = D[i]*(n[i]*se[i]*se[i] + b[i]*b[i]);
    }
    
    for (unsigned i=0; i<numIncdSnps; ++i) {
        snp = incdSnpInfoVec[i];
        for (unsigned j=0; j<snp->windSize; ++j) {
            ZPZ[i][j] *= sqrt(D[i]*D[snp->windStart+j]);
        }
        //ZPZdiag[i] = ZPZ[i][i-snp->windStart];
    }
    ZPZdiag = D;
    
    //b.array() -= b.mean();
    ZPy.array() = D.array()*b.array();
    
    //    cout << "ZPZdiag " << ZPZdiag.transpose() << endl;
    //    cout << "ZPZ.back() " << ZPZ.back().transpose() << endl;
    //    cout << "ZPZ.front() " << ZPZ.front().transpose() << endl;
    //    cout << "ZPy " << ZPy.head(100).transpose() << endl;
    //    cout << "b.mean() " << b.mean() << endl;
    
    // estimate ypy
    ypy = (D.array()*(n.array()*se.array().square()+b.array().square())).mean();
    numKeptInds = n.mean();
    
    //cout << ZPZ.size() << " " << ZPy.size() << " " << ypy << endl;
    //    cout << ZPy << endl;
    //    for (unsigned i=0; i<numIncdSnps; ++i) {
    //        cout << D[i] << "\t" << ZPZdiag[i] << endl;
    //    }
    //    
    //    cout << ZPZ << endl;
    
    // no fixed effects
    numFixedEffects = 0;
    fixedEffectNames.resize(0);
    XPX.resize(0,0);
    ZPX.resize(0,0);
    XPy.resize(0);
}

void Data::readMultiLDmatInfoFile(const string &mldmatFile){
    ifstream in(mldmatFile.c_str());
    if (!in) throw ("Error: can not open the file [" + mldmatFile + "] to read.");
    cout << "Reading SNP info from [" + mldmatFile + "]..." << endl;
    
    string inputStr;
    while (getline(in, inputStr)) {
        vector<SnpInfo*> vec;
        readLDmatrixInfoFile(inputStr+".info", vec);
        //        numSnpMldVec.push_back(numSnps);
        mldmVec.push_back(vec);
    }
}


//void Data::readMultiLDmatBinFile(const string &mldmatFile){
//    ifstream in1(mldmatFile.c_str());
//    if (!in1) throw ("Error: can not open the file [" + mldmatFile + "] to read.");
//    cout << "Reading LD matrices from [" + mldmatFile + "]..." << endl;
//    
//    VectorXi windStartLDM(numSnps);
//    VectorXi windSizeLDM(numSnps);
//    
//    for (unsigned i=0; i<numSnps; ++i) {
//        SnpInfo *snp = snpInfoVec[i];
//        windStartLDM[i] = snp->windStart;
//        windSizeLDM[i] = snp->windSize;
//    }
//    
//    Gadget::Timer timer;
//    timer.setTime();
//    
//    string filename;
//    float windowWidth=0;
//    unsigned i=0, incj=0;
//    while (getline(in1, filename)) {
//        FILE *in2 = fopen((filename+".bin").c_str(), "rb");
//        if (!in2) {
//            throw("Error: cannot open LD matrix file " + filename + ".bin");
//        }
//        if (i==0) {
//            fread(&windowWidth, sizeof(float), 1, in2);
//            getWindowInfo(incdSnpInfoVec, windowWidth*1e6, windStart, windSize);
////                cout << "windStart " << windStart.transpose() << endl;
////                cout << "windSize " << windSize.transpose() << endl;
//            ZPZ.resize(numIncdSnps);
//            for (unsigned j=0; j<numIncdSnps; ++j) {
//                ZPZ[j].resize(windSize[j]);
//            }
//        } else {
//            float ww;
//            fread(&ww, sizeof(float), 1, in2);
//            if (ww!=windowWidth) {
//                throw("Error: LD matrix file [" + filename + "] has a different window width (" + to_string(static_cast<long long>(ww/1e6))
//                      + "Mb) than others (" + to_string(static_cast<long long>(windowWidth/1e6)) + "Mb)");
//            }
//        }
//        
//        SnpInfo *snpj = NULL;
//        SnpInfo *snpk = NULL;
//        unsigned start = 0;
//        if (i>0) start = numSnpMldVec[i-1];
//        
//        for (unsigned j = start; j < numSnpMldVec[i]; j++) {
//            snpj = snpInfoVec[j];
//            
//            float v[windSizeLDM[j]];
//            fread(v, sizeof(v), 1, in2);
//            
//            if (!snpj->included) continue;
//
////            cout << windStartLDM[j] << " " << j << " " << snpInfoVec[j]->included << endl;
//            for (unsigned k = 0, inck = 0; k<windSizeLDM[j]; ++k) {
//                snpk = snpInfoVec[start+windStartLDM[j]+k];
//                //if (incj==50 || incj==51) cout << incj << " " << k << " " << ZPZ[incj].size() << " " << start+windStartLDM[j] << " " << windSizeLDM[j] << " " << windSize[incj] << " " << v[k] << " " << snpk->included << endl;
//                if (snpk->included) {
//                    //cout << incj << " " << inck << " " << ZPZ[incj].size() << " " << start+windStartLDM[j] << " " << windSizeLDM[j] << " " << windSize[incj] << " " << v[k] << endl;
//                    ZPZ[incj][inck++] = v[k];
//                }
//            }
//            ++incj;
//        }
//        
//        fclose(in2);
//        cout << "Read LD matrix for " << numSnpMldVec[i] - start << " SNPs from [" << filename << "]." << endl;
//        ++i;
//    }
//    
//    timer.getTime();
//    
//    cout << "Window width " << windowWidth << " Mb." << endl;
//    cout << "Average window size " << windSize.sum()/numIncdSnps << "." << endl;
//    cout << "Read LD matrix for " << numIncdSnps << " SNPs (time used: " << timer.format(timer.getElapse()) << ")." << endl;
//    
//}

