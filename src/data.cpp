#include "data.hpp"
#include <Eigen/Eigen>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iterator>
#include "compression.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_utils.hpp"
#endif

Data::Data()
    : ppBedFd(-1)
    , ppBedMap(nullptr)
    , mappedZ(nullptr, 1, 1)
    , ppbedIndex()
{
}

#ifdef USE_MPI

void Data::sparse_data_get_sizes_from_raw(const char* rawdata, const uint NC, const uint NB, size_t& N1, size_t& N2, size_t& NM) {

    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)malloc(NB * 4 * sizeof(char));

    N1 = 0;
    N2 = 0;
    NM = 0;
    size_t N0 = 0;

    for (uint i=0; i<NC; ++i) {

        uint c0 = 0, c1 = 0, c2 = 0, cm = 0;

        char* locraw = (char*)&rawdata[size_t(i)*size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        for (int ii=0; ii<numInds; ++ii) {
            if      (tmpi[ii] <  0) { cm += 1; NM += 1; }
            else if (tmpi[ii] == 0) { c0 += 1; N0 += 1; }
            else if (tmpi[ii] == 1) { c1 += 1; N1 += 1; }
            else if (tmpi[ii] == 2) { c2 += 1; N2 += 1; }
        }
        
        assert(cm+c0+c1+c2 == numInds);
    }

    free(tmpi);
}


// Screen the raw BED data and count and register number of ones, twos and missing information
// N*S: store the "S"tart of each marker representation 
// N*L: store the "L"ength of each marker representation
// I* : store the indices of the elements
// -------------------------------------------------------------------------------------------
void Data::sparse_data_fill_indices(const char* rawdata, const uint NC, const uint NB,
                                    size_t* N1S, size_t* N1L, uint* I1,
                                    size_t* N2S, size_t* N2L, uint* I2,
                                    size_t* NMS, size_t* NML, uint* IM) {
    
    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)malloc(NB * 4 * sizeof(int8_t));

    size_t i1 = 0, i2 = 0, im = 0;
    size_t N1 = 0, N2 = 0, NM = 0;

    for (int i=0; i<NC; ++i) {

        char* locraw = (char*)&rawdata[size_t(i)*size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        size_t n0 = 0, n1 = 0, n2 = 0, nm = 0;
        
        for (uint ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                IM[im] = ii;
                im += 1;
                nm += -tmpi[ii];
            } else {
                if        (tmpi[ii] == 0) {
                    n0 += 1;
                } else if (tmpi[ii] == 1) {
                    I1[i1] = ii;
                    i1 += 1;
                    n1 += 1;
                } else if (tmpi[ii] == 2) {
                    I2[i2] = ii;
                    i2 += 1;
                    n2 += 1;
                }
            }
        }

        assert(nm + n0 + n1 + n2 == numInds);

        N1S[i] = N1;  N1L[i] = n1;  N1 += n1;
        N2S[i] = N2;  N2L[i] = n2;  N2 += n2;
        NMS[i] = NM;  NML[i] = nm;  NM += nm;
    }

    free(tmpi);
}

size_t Data::get_number_of_elements_from_sparse_files(const std::string basename, const std::string id, const int* MrankS, const int* MrankL,
                                                      size_t* S, size_t* L) {

    MPI_Status status;
    MPI_File   ssfh, slfh;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Number of markers in block handled by task
    const uint M = MrankL[rank];

    const std::string sl = basename + ".sl" + id;
    const std::string ss = basename + ".ss" + id;

    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ssfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);

    // Compute the lengths of ones and twos vectors for all markers in the block
    MPI_Offset offset =  MrankS[rank] * sizeof(size_t);
    
    check_mpi(MPI_File_read_at_all(ssfh, offset, S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slfh, offset, L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);

    // Absolute offsets in 0s, 1s, and 2s
    const size_t nsoff = S[0];

    size_t N = S[M-1] + L[M-1] - nsoff;

    // Close bed and sparse files
    check_mpi(MPI_File_close(&ssfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);

    return N;
}

/*
void Data::sparse_data_get_sizes_from_sparse(size_t* N1S, size_t* N1L,
                                             size_t* N2S, size_t* N2L,
                                             size_t* NMS, size_t* NML,
                                             const int* MrankS, const int* MrankL, const int rank,
                                             const std::string sparseOut,
                                             size_t& N1, size_t& N2, size_t& NM) {
    
    MPI_Offset offset;
    MPI_Status status;

    // Number of markers in block handled by task
    const uint M = MrankL[rank];

    // Read sparse data files
    // Each task is in charge of M markers starting from MrankS[rank]
    // So first we read si1 to get where to read in 
    MPI_File ss1fh, sl1fh, si1fh;
    MPI_File ss2fh, sl2fh, si2fh;
    MPI_File ssmfh, slmfh, simfh;

    // Get bed file directory and basename
    if (rank == 0) printf("INFO   : will read from sparse files with basename: %s\n", sparseOut.c_str());

    const std::string sl1 = sparseOut + ".sl1";
    const std::string ss1 = sparseOut + ".ss1";
    const std::string sl2 = sparseOut + ".sl2";
    const std::string ss2 = sparseOut + ".ss2";
    const std::string slm = sparseOut + ".slm";
    const std::string ssm = sparseOut + ".ssm";

    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, slm.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ssm.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ssmfh), __LINE__, __FILE__);

    // Compute the lengths of ones and twos vectors for all markers in the block
    offset =  MrankS[rank] * sizeof(size_t);
    //printf("rank %4d: offset = %20lu Bytes MrankS = %lu, M = %lu\n", rank, offset, MrankS[rank], M);
    check_mpi(MPI_File_read_at_all(sl1fh, offset, N1L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ss1fh, offset, N1S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(sl2fh, offset, N2L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ss2fh, offset, N2S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slmfh, offset, NML, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ssmfh, offset, NMS, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    
    // Sanity checks
    // -------------
    for (int i=0; i<M-1; i++) {
        if (N1S[i] > N1S[i+1]) {
            printf("rank %4d: Cannot be N1S[%lu] > N1S[%lu] (%20lu > %20lu)\n", rank, i, i+1, N1S[i], N1S[i+1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (N2S[i] > N2S[i+1]) {
            printf("rank %4d: Cannot be N2S[%lu] > N2S[%lu] (%20lu > %20lu)\n", rank, i, i+1, N2S[i], N2S[i+1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (NMS[i] > NMS[i+1]) {
            printf("rank %4d: Cannot be NMS[%lu] > NMS[%lu] (%20lu > %20lu)\n", rank, i, i+1, NMS[i], NMS[i+1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }


    // Absolute offsets in 0s, 1s, and 2s
    const size_t n1soff = N1S[0];
    const size_t n2soff = N2S[0];
    const size_t nmsoff = NMS[0];

    N1 = N1S[M-1] + N1L[M-1] - n1soff;
    N2 = N2S[M-1] + N2L[M-1] - n2soff;
    NM = NMS[M-1] + NML[M-1] - nmsoff;

    // Close bed and sparse files
    check_mpi(MPI_File_close(&sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ssmfh), __LINE__, __FILE__);

    MPI_Barrier(MPI_COMM_WORLD);
}
*/


void Data::read_sparse_data_file(const std::string filename, const size_t N, const size_t OFF, const int NREADS, uint* out) {

    MPI_Offset offset;
    MPI_Status status;
    MPI_File   fh;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    offset = OFF * sizeof(uint);

    mpi_file_read_at_all<uint*>(N, offset, fh, MPI_UNSIGNED, NREADS, out);

    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}


//void Data::get_normalized_marker_data(const char* rawdata, const uint NB)
void Data::get_normalized_marker_data(const char* rawdata, const uint NB, const uint marker, double* Cx, const double mean, const double std_) {

    assert(numInds<=NB*4);

    // Pointer to column in block of raw data
    char* locraw = (char*)&rawdata[size_t(marker) * size_t(NB)];
    
    // temporary array used for translation
    int8_t *tmpi = (int8_t*)malloc(NB * 4 * sizeof(int8_t));

    for (int ii=0; ii<NB; ++ii) {
        for (int iii=0; iii<4; ++iii) {
            tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
        }
    }
       
    for (int ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] == 1) {
            tmpi[ii] = -1;
        } else {
            tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
        }
    }
    
    for (size_t ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] < 0) {
            Cx[ii]  = 0.0;
        } else {
            Cx[ii] = (double(tmpi[ii]) - mean) * std_;
        }
    }

    free(tmpi);
}


//void Data::get_normalized_marker_data(const char* rawdata, const uint NB)
void Data::get_normalized_marker_data(const char* rawdata, const uint NB, const uint marker, double* Cx) {

    assert(numInds<=NB*4);

    // Pointer to column in block of raw data
    char* locraw = (char*)&rawdata[size_t(marker) * size_t(NB)];
    
    // temporary array used for translation
    int8_t *tmpi = (int8_t*)malloc(NB * 4 * sizeof(int8_t));

    for (int ii=0; ii<NB; ++ii) {
        for (int iii=0; iii<4; ++iii) {
            tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
        }
    }
       
    for (int ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] == 1) {
            tmpi[ii] = -1;
        } else {
            tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
        }
    }

    int sum = 0, nmiss = 0;
    //#pragma omp simd reduction(+:sum) reduction(+:nmiss)
    for (int ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] < 0) {
            nmiss += tmpi[ii];
        } else {
            sum   += tmpi[ii];
        }
    }

    double mean = double(sum) / double(numInds + nmiss); //EO: nmiss is neg
    //if (marker < 3)
    //    printf("marker %d  mean = %20.15f\n", marker, mean);

    //printf("rank %d snpInd %2d: sum = %6d, N = %6d, nmiss = %6d, mean = %20.15f\n",
    //       rank, rank*NC+i, sum, numKeptInds, nmiss, mean);
    
    for (size_t ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] < 0) {
            Cx[ii] = 0.0d;
        } else {
            Cx[ii] = double(tmpi[ii]) - mean;
        }
    }

    // Normalize the data
    double sqn  = 0.0d;
    for (size_t ii=0; ii<numInds; ++ii) {
        sqn += Cx[ii] * Cx[ii];
    }
    if (marker < 1)
        printf("marker %d  sqn = %20.15f\n", marker, sqn);

    double std_ = sqrt(double(numInds - 1) / sqn);
    if (marker < 1)
        printf("marker %d  std_ = %20.15f\n", marker, std_);

    for (size_t ii=0; ii<numInds; ++ii)
        Cx[ii] *= std_;

    free(tmpi);
}

// Read raw data loaded in memory to preprocess them (center, scale and cast to double)
void Data::preprocess_data(const char* rawdata, const uint NC, const uint NB, double* ppdata, const int rank) {

    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)malloc(NB * 4 * sizeof(int8_t));

    for (int i=0; i<NC; ++i) {

        char* locraw = (char*)&rawdata[size_t(i) * size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        int sum = 0, nmiss = 0;
        //#pragma omp simd reduction(+:sum) reduction(+:nmiss)
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                nmiss += tmpi[ii];
            } else {
                sum   += tmpi[ii];
            }
        }

        double mean = double(sum) / double(numInds + nmiss); //EO: nmiss is neg
        //printf("rank %d snpInd %2d: sum = %6d, N = %6d, nmiss = %6d, mean = %20.15f\n",
        //       rank, rank*NC+i, sum, numKeptInds, nmiss, mean);

        size_t ppdata_i = size_t(i) * size_t(numInds);
        double *locpp = (double*)&ppdata[ppdata_i];

        for (size_t ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                locpp[ii] = 0.0d;
            } else {
                locpp[ii] = double(tmpi[ii]) - mean;
            }
        }

        double sqn  = 0.0d;
        for (size_t ii=0; ii<numInds; ++ii) {
            sqn += locpp[ii] * locpp[ii];
        }

        double std_ = sqrt(double(numInds - 1) / sqn);

        for (size_t ii=0; ii<numInds; ++ii) {
            locpp[ii] *= std_;
        }
    }
    free(tmpi);
}

#endif



void Data::preprocessBedFile(const string &bedFile, const string &preprocessedBedFile, const string &preprocessedBedIndexFile, bool compress)
{
    cout << "Preprocessing bed file: " << bedFile << ", Compress data = " << (compress ? "yes" : "no") << endl;
    if (numSnps == 0)
        throw ("Error: No SNP is retained for analysis.");
    if (numInds == 0)
        throw ("Error: No individual is retained for analysis.");

    ifstream BIT(bedFile.c_str(), ios::binary);
    if (!BIT)
        throw ("Error: can not open the file [" + bedFile + "] to read.");

    ofstream ppBedOutput(preprocessedBedFile.c_str(), ios::binary);
    if (!ppBedOutput)
        throw("Error: Unable to open the preprocessed bed file [" + preprocessedBedFile + "] for writing.");
    ofstream ppBedIndexOutput(preprocessedBedIndexFile.c_str(), ios::binary);
    if (!ppBedIndexOutput)
        throw("Error: Unable to open the preprocessed bed index file [" + preprocessedBedIndexFile + "] for writing.");

    cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    char header[3];
    BIT.read(header, 3);
    if (!BIT || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01)
        throw ("Error: Incorrect first three bytes of bed file: " + bedFile);

    // How much space do we need to compress the data (if requested)
    const auto maxCompressedOutputSize = compress ? maxCompressedDataSize(numInds) : 0;
    unsigned char *compressedBuffer = nullptr;
    unsigned long pos = 0;
    if (compress)
        compressedBuffer = new unsigned char[maxCompressedOutputSize];

    // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 10: hetezygote; 01: missing
    for (unsigned int j = 0, snp = 0; j < numSnps; j++) {
        SnpInfo *snpInfo = snpInfoVec[j];
        double sum = 0.0;
        unsigned int nmiss = 0;

        // Create some scratch space to preprocess the raw data
        VectorXd snpData(numInds);

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
        const double mean = sum / double(numInds - nmiss);
        if (j % 100 == 0) {
            printf("MARKER %6d mean = %12.7f computed on %6.0f with %6d elements (%d - %d)\n",
                   j, mean, sum, numInds-nmiss, numInds, nmiss);
            fflush(stdout);
        }
        if (nmiss) {
            for (const auto index : missingIndices)
                snpData[index] = mean;
        }

        // Standardize genotypes
        snpData.array() -= snpData.mean();
        const auto sqn = snpData.squaredNorm();
        const auto sigma = 1.0 / (sqrt(sqn / (double(numInds - 1))));
        snpData.array() *= sigma;

        // Write out the preprocessed data
        if (!compress) {
            ppBedOutput.write(reinterpret_cast<char *>(&snpData[0]), numInds * sizeof(double));
        } else {
            const unsigned long compressedSize = compressData(snpData, compressedBuffer, maxCompressedOutputSize);
            ppBedOutput.write(reinterpret_cast<char *>(compressedBuffer), long(compressedSize));

            // Calculate the index data for this column
            ppBedIndexOutput.write(reinterpret_cast<char *>(&pos), sizeof(unsigned long));
            ppBedIndexOutput.write(reinterpret_cast<const char *>(&compressedSize), sizeof(unsigned long));
            pos += compressedSize;
        }

        // Compute allele frequency and any other required data and write out to file
        //snpInfo->af = 0.5f * float(mean);
        //snp2pq[snp] = 2.0f * snpInfo->af * (1.0f - snpInfo->af);

        if (++snp == numSnps)
            break;
    }

    if (compress)
        delete[] compressedBuffer;

    BIT.clear();
    BIT.close();

    cout << "Genotype data for " << numInds << " individuals and " << numSnps << " SNPs are included from [" + bedFile + "]." << endl;
}

void Data::mapPreprocessBedFile(const string &preprocessedBedFile)
{
    // Calculate the expected file sizes - cast to size_t so that we don't overflow the unsigned int's
    // that we would otherwise get as intermediate variables!
    const size_t ppBedSize = size_t(numInds) * size_t(numSnps) * sizeof(double);

    // Open and mmap the preprocessed bed file
    ppBedFd = open(preprocessedBedFile.c_str(), O_RDONLY);
    if (ppBedFd == -1)
        throw("Error: Failed to open preprocessed bed file [" + preprocessedBedFile + "]");

    ppBedMap = reinterpret_cast<double *>(mmap(nullptr, ppBedSize, PROT_READ, MAP_SHARED, ppBedFd, 0));
    if (ppBedMap == MAP_FAILED)
        throw("Error: Failed to mmap preprocessed bed file");

    // Now that the raw data is available, wrap it into the mapped Eigen types using the
    // placement new operator.
    // See https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
    new (&mappedZ) Map<MatrixXd>(ppBedMap, numInds, numSnps);
}

void Data::unmapPreprocessedBedFile()
{
    // Unmap the data from the Eigen accessors
    new (&mappedZ) Map<MatrixXd>(nullptr, 1, 1);

    const auto ppBedSize = numInds * numSnps * sizeof(double);
    munmap(ppBedMap, ppBedSize);
    close(ppBedFd);
}

void Data::mapCompressedPreprocessBedFile(const string &preprocessedBedFile,
                                          const string &indexFile)
{
    // Load the index to the compressed preprocessed bed file
    ppbedIndex.resize(numSnps);
    ifstream indexStream(indexFile, std::ifstream::binary);
    if (!indexStream)
        throw("Error: Failed to open compressed preprocessed bed file index");
    indexStream.read(reinterpret_cast<char *>(ppbedIndex.data()),
                     numSnps * 2 * sizeof(long));

    // Calculate the expected file sizes - cast to size_t so that we don't overflow the unsigned int's
    // that we would otherwise get as intermediate variables!
    const size_t ppBedSize = size_t(ppbedIndex.back().pos + ppbedIndex.back().size);

    // Open and mmap the preprocessed bed file
    ppBedFd = open(preprocessedBedFile.c_str(), O_RDONLY);
    if (ppBedFd == -1)
        throw("Error: Failed to open preprocessed bed file [" + preprocessedBedFile + "]");

    ppBedMap = reinterpret_cast<double *>(mmap(nullptr, ppBedSize, PROT_READ, MAP_SHARED, ppBedFd, 0));
    if (ppBedMap == MAP_FAILED)
        throw("Error: Failed to mmap preprocessed bed file");
}

void Data::unmapCompressedPreprocessedBedFile()
{
    const size_t ppBedSize = size_t(ppbedIndex.back().pos + ppbedIndex.back().size);
    munmap(ppBedMap, ppBedSize);
    close(ppBedFd);
    ppbedIndex.clear();
}


// EO
// Read marker blocks definition file
// Line format:  "%d %d\n"; Nothing else is accepted.
// Gaps are allowed; Overlaps are forbiden
// --------------------------------------------------
void Data::readMarkerBlocksFile(const string &markerBlocksFile) {
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream in(markerBlocksFile.c_str());
    if (!in) throw ("Error: can not open the file [" + markerBlocksFile + "] to read.");
    
    blocksStarts.clear();
    blocksEnds.clear();
    std::string str;
    
	while (std::getline(in, str)) {
        std::vector<std::string> results;
        boost::split(results, str, [](char c){return c == ' ';});
        if (results.size() != 2) {
            printf("Error: line with wrong format: >>%s<<\n", str.c_str());
            printf("       => expected format \"%%d %%d\": two integers separated with a single space, with no leading or trailing space.\n");
            exit(1);
        }
        blocksStarts.push_back(stoi(results[0]));
        blocksEnds.push_back(stoi(results[1]));        
	}
	in.close();

    numBlocks = (unsigned) blocksStarts.size();
    //cout << "Found definitions for " << nbs << " marker blocks." << endl;

    // Neither gaps or overlaps are accepted
    // -------------------------------------
    for (int i=0; i<numBlocks; ++i) {

        if (blocksStarts[i] > blocksEnds[i]) {
            if (rank == 0) {
                printf("FATAL  : block starts beyond end [%d, %d].\n", blocksStarts[i], blocksEnds[i]);
                printf("         => you must correct your marker blocks definition file %s\n", markerBlocksFile.c_str());
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int j=i+1;
        if (j < numBlocks && blocksStarts[j] != blocksEnds[i] + 1) {
            if (rank == 0) {
                printf("FATAL  : block %d ([%d, %d]) and block %d ([%d, %d]) are not contiguous!\n", i, blocksStarts[i], blocksEnds[i], j, blocksStarts[j], blocksEnds[j]);
                printf("         => you must correct your marker blocks definition file %s\n", markerBlocksFile.c_str());
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}


void Data::readFamFile(const string &famFile){
    // ignore phenotype column
    ifstream in(famFile.c_str());
    if (!in) throw ("Error: can not open the file [" + famFile + "] to read.");
#ifndef USE_MPI
    cout << "Reading PLINK FAM file from [" + famFile + "]." << endl;
#endif
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

    //#ifndef USE_MPI
    cout << numInds << " individuals to be included from [" + famFile + "]." << endl;
    //#endif
}

void Data::readBimFile(const string &bimFile) {
    // Read bim file: recombination rate is defined between SNP i and SNP i-1
    ifstream in(bimFile.c_str());
    if (!in) throw ("Error: can not open the file [" + bimFile + "] to read.");
#ifndef USE_MPI
    cout << "Reading PLINK BIM file from [" + bimFile + "]." << endl;
#endif
    snpInfoVec.clear();
    snpInfoMap.clear();
    string id, allele1, allele2;
    unsigned chr, physPos;
    float genPos;
    unsigned idx = 0;
    while (in >> chr >> id >> genPos >> physPos >> allele1 >> allele2) {
        SnpInfo *snp = new SnpInfo(idx++, id, allele1, allele2, chr, genPos, physPos);
        snpInfoVec.push_back(snp);
        if (snpInfoMap.insert(pair<string, SnpInfo*>(id, snp)).second == false) {
            throw ("Error: Duplicate SNP ID found: \"" + id + "\".");
        }
    }
    in.close();
    numSnps = (unsigned) snpInfoVec.size();
#ifndef USE_MPI
    cout << numSnps << " SNPs to be included from [" + bimFile + "]." << endl;
#endif
}


void Data::readBedFile_noMPI(const string &bedFile) {

    unsigned i = 0, j = 0, k = 0;

    cout << "readBedFile_noMPI will read numSnps = " << numSnps << endl;

    if (numSnps == 0) throw ("Error: No SNP is retained for analysis.");
    if (numInds == 0) throw ("Error: No individual is retained for analysis.");
    printf("numInds = %d, numNAs = %d, numSnps = %d\n", numInds, numNAs, numSnps);

    Z.resize(numInds-numNAs, numSnps);
    ZPZdiag.resize(numSnps);
    snp2pq.resize(numSnps);

    // Read bed file
    char ch[1];
    bitset<8> b;
    unsigned allele1=0, allele2=0;
    ifstream BIT(bedFile.c_str(), ios::binary);
    if (!BIT) throw ("Error: can not open the file [" + bedFile + "] to read.");
    //#ifndef USE_MPI
    cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    //#endif

    for (i = 0; i < 3; i++) BIT.read(ch, 1); // skip the first three bytes
    SnpInfo *snpInfo = NULL;
    unsigned snp = 0, ind = 0;
    unsigned nmiss = 0;
    //float mean = 0.0;
    double mean = 0.0;

    // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 10: hetezygote; 01: missing
    // --------------------------------------------------------------------------------------------------
    for (j = 0, snp = 0; j < numSnps; j++) {
        snpInfo = snpInfoVec[j];
        mean = 0.0;
        nmiss = 0;
        if (!snpInfo->included) {
            for (i = 0; i < numInds; i += 4) BIT.read(ch, 1);
            continue;
        }
        
        uint nona = 0;

        for (i = 0, ind = 0; i < numInds;) {
            BIT.read(ch, 1);
            if (!BIT) throw ("Error: problem with the BED file ... has the FAM/BIM file been changed?");
            b = ch[0];
            k = 0;
            while (k < 7 && i < numInds) {
                if (!indInfoVec[i]->kept) {
                    k += 2;
                } else {
                    allele1 = (!b[k++]);
                    allele2 = (!b[k++]);
                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        Z(nona, snp) = -9;
                        ++nmiss;
                    } else {
                        mean += Z(nona, snp) = allele1 + allele2;
                    }
                    nona+=1;
                }
                i += 1;
            }
        }

        if (j==0) printf("numInds vs nona; numNAs: %d vs %d vs %d\n", numInds, nona, numNAs);

        //EO: account for missing phenotypes
        //----------------------------------
        mean /= double(nona-nmiss);
        
        if (nmiss) {
            for (i=0; i<nona; ++i) {
                if (Z(i,snp) == -9) Z(i,snp) = mean;
            }
        }

        // compute allele frequency
        //snpInfo->af = 0.5f*mean;
        //snp2pq[snp] = 2.0f*snpInfo->af*(1.0f-snpInfo->af);

        // standardize genotypes
        Z.col(j).array() -= mean;

        //float sqn = Z.col(j).squaredNorm();
        //float std_ = 1.f / (sqrt(sqn / float(numInds)));
        double sqn = Z.col(j).squaredNorm();
        /*
        double sqn_ = 0.0;
        for (i=0; i<nona; ++i) {
            sqn_ += Z(i,snp) * Z(i,snp);
        }
        printf("sqn vs sqn_  %20.15f vs %20.15f\n", sqn, sqn);
        */
        //double std_ = 1.0 / (sqrt(sqn / double(numInds - 1)));
        double std_ = sqrt(double(nona - 1) / sqn);

        //if (j < 1)
        //printf("OFF: marker %d has mean = %20.15f sqn = %20.15f  std_ = %20.15f\n", j, mean, sqn, std_);

        Z.col(j).array() *= std_;

        ZPZdiag[j] = sqn;

        if (++snp == numSnps) break;
    }
    BIT.clear();
    BIT.close();

    for (i=0; i<numSnps; ++i) {
        //Z.col(i).array() -= Z.col(i).mean(); //EO alread done (so now mean is zero)
        ZPZdiag[i] = Z.col(i).squaredNorm();
    }

#ifndef USE_MPI
    cout << "Genotype data for " << numInds << " individuals and " << numSnps << " SNPs are included from [" + bedFile + "]." << endl;
#endif
}

// EO: overloaded function to be used when processing sparse data
//     In such case we do not read from fam file before
// --------------------------------------------------------------
void Data::readPhenotypeFile(const string &phenFile, const int numberIndividuals) {
    numInds = numberIndividuals;
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
#ifndef USE_MPI
    cout << "Reading phenotypes from [" + phenFile + "]." << endl;
#endif
    uint line = 0, nas = 0, nonas = 0;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    y.setZero(numInds);
    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        if (colData[1+1] != "NA") {
            y[nonas] = double( atof(colData[1+1].c_str()) );
            //if (nonas < 30) printf("read no na on line %d, nonas %d = %15.10f\n", line, nonas, y(nonas));
            nonas += 1;
        } else {
            //cout << "WARNING: found NA on line/individual " << line << endl;
            NAsInds.push_back(line);
            nas += 1;
        }
        line += 1;
    }
    in.close();
    assert(nonas + nas == numInds);

    numNAs = nas;
    y.conservativeResize(numInds-nas);
}

void Data::readPhenotypeFile(const string &phenFile) {
    // NA: missing phenotype
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
#ifndef USE_MPI
    cout << "Reading phenotypes from [" + phenFile + "]." << endl;
#endif
    map<string, IndInfo*>::iterator it, end=indInfoMap.end();
    IndInfo *ind = NULL;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    string id;
    double tmp = 0.0;
    //correct loop to go through numInds
    y.setZero(numInds);
    uint line = 0, nas = 0, nonas = 0;

    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        id = colData[0] + ":" + colData[1];
        it = indInfoMap.find(id);
        // First one corresponded to mphen variable (1+1)
        if (it != end) {
            ind = it->second;
            if (colData[1+1] != "NA") {
                tmp = double(atof(colData[1+1].c_str()));
                ind->phenotype = tmp;
                y[nonas]       = tmp;
                nonas += 1;
            } else {
                //cout << "WARNING; found NA on line/individual " << line << endl;
                ind->kept = false;
                NAsInds.push_back(line);
                nas += 1;
            }
            line += 1;
        }
    }
    in.close();
    assert(nonas + nas == numInds);

    numNAs = nas;
    y.conservativeResize(numInds-nas);
}


//TODO Finish function to read the group file
void Data::readGroupFile(const string &groupFile) {
    // NA: missing phenotype
    ifstream in(groupFile.c_str());
    if (!in) throw ("Error: can not open the group file [" + groupFile + "] to read.");

    cout << "Reading groups from [" + groupFile + "]." << endl;

    std::istream_iterator<double> start(in), end;
    std::vector<int> numbers(start, end);
    int* ptr =(int*)&numbers[0];
    G=(Eigen::Map<Eigen::VectorXi>(ptr,numbers.size()));

    cout << "Groups read from file" << endl;
}

template<typename M>
M Data::readCSVFile (const string &path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    //cout << "rows = " << rows << " vs numInds " << numInds << endl; 
    if (rows != numInds)
        throw(" Error: covariate file has different number of individuals as BED file");
    numFixedEffects = values.size()/rows;
    return Map<const Matrix<typename M::Scalar, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size()/rows);
}

void Data::readCovariateFile(const string &covariateFile ) {
	X = readCSVFile<MatrixXd>(covariateFile);
}
