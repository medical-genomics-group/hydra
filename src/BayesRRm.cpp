/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include <cstdlib>
#include "BayesRRm.h"
#include "data.hpp"
#include "distributions_boost.hpp"
#include "options.hpp"
#include "samplewriter.h"
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <libgen.h>
#include <string.h>

#ifdef USE_MPI
#include <mpi.h>
#include "mpi_utils.hpp"
#endif


BayesRRm::BayesRRm(Data &data, Options &opt, const long memPageSize)
: data(data)
, opt(opt)
, bedFile(opt.bedFile + ".bed")
, memPageSize(memPageSize)
, outputFile(opt.mcmcOut + ".csv")
, seed(opt.seed)
, max_iterations(opt.chainLength)
, burn_in(opt.burnin)
, dist(opt.seed)
, usePreprocessedData(opt.analysisType == "PPBayes")
, showDebug(false)
{
    float* ptr =static_cast<float*>(&opt.S[0]);
    cva = (Eigen::Map<Eigen::VectorXf>(ptr, static_cast<long>(opt.S.size()))).cast<double>();
}

BayesRRm::~BayesRRm()
{
}

#ifdef USE_MPI

// Check malloc in MPI context
// ---------------------------
inline void check_malloc(const void* ptr, const int linenumber, const char* filename) {
    if (ptr == NULL) {
        fprintf(stderr, "#FATAL#: malloc failed on line %d of %s\n", linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


inline void scaadd(double* __restrict__ vout, const double* __restrict__ vin1, const double* __restrict__ vin2, const double dMULT, const int N) {

    if   (dMULT == 0.0) { 
        for (int i=0; i<N; i++) { 
            vout[i] = vin1[i]; 
        } 
    } else { 
        for (int i=0; i<N; i++) { 
            vout[i] = vin1[i] + dMULT * vin2[i];
        }
    }
}


inline void sparse_scaadd(double*       __restrict__ vout,
                          const double  dMULT,
                          const uint* __restrict__ I1, const size_t N1S, const size_t N1L,
                          const uint* __restrict__ I2, const size_t N2S, const size_t N2L,
                          const uint* __restrict__ IM, const size_t NMS, const size_t NML,
                          const double  mu,
                          const double  sig_inv,
                          const int     N) {

    if (dMULT == 0.0) {
        for (int i=0; i<N; i++)
            vout[i] = 0.0;
    } else {

        double aux = mu * sig_inv * dMULT;

        for (int i=0; i<N; i++)
            vout[i] = -aux;

        for (size_t i=NMS; i<NMS+NML; ++i)
            vout[IM[i]] = 0.0;

        aux = dMULT * (1.0 - mu) * sig_inv;
        for (size_t i=N1S; i<N1S+N1L; ++i)
            vout[I1[i]] = aux;

        aux = dMULT * (2.0 - mu) * sig_inv;
        for (size_t i=N2S; i<N2S+N2L; ++i)
            vout[I2[i]] = aux;
    }
}


inline double sparse_dotprod(const double* __restrict__ vin1,
                             const uint*   __restrict__ I1,   const size_t N1S,  const size_t N1L,
                             const uint*   __restrict__ I2,   const size_t N2S,  const size_t N2L,
                             const uint*   __restrict__ IM,   const size_t NMS,  const size_t NML,
                             const double  mu, const double sig_inv, const int N, const int marker) {

    double dp = 0.0, syt = 0.0;

    for (size_t i=N1S; i<N1S+N1L; ++i) {
        //printf("1: %5d %5d %15.10f\n", i, I1[i], vin1[I1[i]]);
        dp +=       vin1[I1[i]];
    }

    for (size_t i=N2S; i<N2S+N2L; ++i) {
        //printf("2: %5d %5d %15.10f\n", i, I2[i], vin1[I2[i]]);
        dp += 2.0 * vin1[I2[i]];
    }

    dp *= sig_inv;

    for (int i=0; i<N; i++) syt += vin1[i];

    //EO: ajust for missing values
    for (size_t i=NMS; i<NMS+NML; ++i) {
        //printf("M: %5d %5d %15.10f\n", i, IM[i], vin1[IM[i]]);
        syt -= vin1[IM[i]];
    }

    dp -= mu * sig_inv * syt;

    return dp;
}


inline void scaadd(double* __restrict__ vout, const double* __restrict__ vin, const double dMULT, const int N) {

    if (dMULT == 0.0) {
        for (int i=0; i<N; i++) {
            vout[i] = 0.0;
        }
    } else {
        for (int i=0; i<N; i++) {
            vout[i] = dMULT * vin[i];
        }
    }
}


inline double dotprod(const double* __restrict__ vec1, const double* __restrict__ vec2, const int N) {

    double dp = 0.0d;

    for (int i=0; i<N; i++) {
        dp += vec1[i] * vec2[i]; 
    }

    return dp;
}


// Define blocks of markers to be processed by each task
// By default processes all markers
// -----------------------------------------------------
void mpi_define_blocks_of_markers(const int Mtot, int* MrankS, int* MrankL, const uint nblocks) {

    const uint modu   = Mtot % nblocks;
    const uint Mrank  = int(Mtot / nblocks);
    uint checkM = 0;
    uint start  = 0;

    for (int i=0; i<nblocks; ++i) {
        MrankL[i] = int(Mtot / nblocks);
        if (modu != 0 && i < modu)
            MrankL[i] += 1;
        MrankS[i] = start;
        //printf("start %d, len %d\n", MrankS[i], MrankL[i]);
        start += MrankL[i];
        checkM += MrankL[i];
    }
    assert(checkM == Mtot);
}


// Get directory and basename of bed file (passed with no extension via command line)
// ----------------------------------------------------------------------------------
std::string BayesRRm::mpi_get_sparse_output_filebase() {

    int rank, nranks, result;
    std::string dir, bsn;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (opt.sparseDir.length() > 0) {
        // Make sure the requested output directory exists
        struct stat stats;
        stat(opt.sparseDir.c_str(), &stats);
        if (!S_ISDIR(stats.st_mode)) { 
            if (rank == 0)
                printf("Fatal: requested directory for sparse output (%s) not found. Must be an existing directory (line %d in %s).\n", opt.sparseDir.c_str(), __LINE__, __FILE__);
            MPI_Abort(MPI_COMM_WORLD, 1); }
        dir = string(opt.sparseDir);
    } else {
        char *cstr = new char[opt.bedFile.length() + 1];
        strcpy(cstr, opt.bedFile.c_str());
        dir = string(dirname(cstr));
    }

    if (opt.sparseBsn.length() > 0) {
        bsn = opt.sparseBsn.c_str();
    } else {
        char *cstr = new char[opt.bedFile.length() + 1];
        strcpy(cstr, opt.bedFile.c_str());
        bsn = string(basename(cstr));
    }
    
    return string(dir) + string("/") + string(bsn);
}


// Sanity check: make sure all elements were set (requires init at UINT_MAX)
// -------------------------------------------------------------------------
void BayesRRm::check_whole_array_was_set(const uint* array, const size_t size, const int linenumber, const char* filename) {

    for (size_t i=0; i<size; i++) { 
        if (array[i] == UINT_MAX) {
            printf("FATAL  : array[%lu] = %d not set at %d of %s!\n", i, array[i], linenumber, filename); 
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}



//EO: This method writes sparse data files out of a BED file
//    Note: will always convert the whole file
//    A two-step process due to RAM limitation for very large BED files:
//      1) Compute the number of ones and twos to be written by each task to compute
//         rank-wise offsets for writing the si1 and si2 files
//      2) Write files with global indexing
// ---------------------------------------------------------------------------------
void BayesRRm::write_sparse_data_files(const uint bpr) {

    int rank, nranks, result;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File   bedfh, dimfh, si1fh, sl1fh, ss1fh, si2fh, sl2fh, ss2fh, simfh, slmfh, ssmfh;
    MPI_Offset offset;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    if (rank == 0) printf("INFO   : will generate sparse data files out of %d ranks and %d blocks per rank.\n", nranks, bpr);

    // Get dimensions of the dataset and define blocks
    // -----------------------------------------------
    const unsigned int N    = data.numInds;
    unsigned int Mtot = data.numSnps;
    if (opt.numberMarkers) Mtot = opt.numberMarkers;

    if (rank == 0) printf("INFO   : full dataset includes %d markers and %d individuals.\n", Mtot, N);

    // Fail if more blocks requested than available markers
    if (nranks * bpr > Mtot) {
        if (rank == 0)
            printf("Fatal: empty tasks defined. Useless and not allowed.\n      Requested %d tasks for %d markers to process.\n", nranks * bpr, Mtot);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Define global marker indexing
    // -----------------------------
    const uint nblocks = nranks * bpr;
    int MrankS[nblocks], MrankL[nblocks];
    mpi_define_blocks_of_markers(Mtot, MrankS, MrankL, nblocks);

    // Length of a column in bytes
    const size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;
    if (rank == 0) printf("INFO   : snpLenByt = %zu bytes.\n", snpLenByt);

    // Get bed file directory and basename
    std::string sparseOut = mpi_get_sparse_output_filebase();
    if (rank == 0) printf("INFO   : will write sparse output files as: %s.{ss1, sl1, si1, ss2, sl2, si2}\n", sparseOut.c_str());

    // Open bed file for reading
    std::string bedfp = opt.bedFile;
    bedfp += ".bed";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh), __LINE__, __FILE__);

    // Create sparse output files
    // --------------------------
    const std::string dim = sparseOut + ".dim";
    const std::string si1 = sparseOut + ".si1";
    const std::string sl1 = sparseOut + ".sl1";
    const std::string ss1 = sparseOut + ".ss1";
    const std::string si2 = sparseOut + ".si2";
    const std::string sl2 = sparseOut + ".sl2";
    const std::string ss2 = sparseOut + ".ss2";
    const std::string sim = sparseOut + ".sim";
    const std::string slm = sparseOut + ".slm";
    const std::string ssm = sparseOut + ".ssm";

    if (rank == 0) { MPI_File_delete(dim.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(si1.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(sl1.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(ss1.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(si2.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(sl2.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(ss2.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(sim.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(slm.c_str(), MPI_INFO_NULL); }
    if (rank == 0) { MPI_File_delete(ssm.c_str(), MPI_INFO_NULL); }

    MPI_Barrier(MPI_COMM_WORLD);
    
    check_mpi(MPI_File_open(MPI_COMM_WORLD, dim.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &dimfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sim.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, slm.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ssm.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ssmfh), __LINE__, __FILE__);

    // Write to dim file (as text)
    char buff[LENBUF];

    if (rank == 0) {
        int  left = snprintf(buff, LENBUF, "%d %d\n", N, Mtot);
        check_mpi(MPI_File_write_at(dimfh, 0, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    // STEP 1: compute rank-wise N1 and N2 (rN1 and rN2)
    // -------------------------------------------------
    size_t rN1 = 0, rN2 = 0, rNM = 0;
    size_t N1  = 0, N2  = 0, NM  = 0;

    for (int i=0; i<bpr; ++i) {

        if (rank == 0) printf("INFO   : reading (1/2) starting block %3d out of %3d\n", i+1, bpr);

        uint globi = rank*bpr + i;
        int  MLi   = MrankL[globi];
        int  MSi   = MrankS[globi];
        //printf("DEBUG  : 1| bpr %i  MLi = %d, MSi = %d\n", i, MLi, MSi);

        // Alloc memory for raw BED data
        const size_t rawdata_n = size_t(MLi) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) malloc(rawdata_n);  check_malloc(rawdata, __LINE__, __FILE__);

        // Gather sizes to determine common number of reads
        size_t rawdata_n_max = 0;
        check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        
        int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (NREADS <= 0) {
            if (rank == 0) printf("FATAL  : NREADS must be >= 1.");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Compute the offset of the section to read from the BED file
        offset = size_t(3) + size_t(MSi) * size_t(snpLenByt) * sizeof(char);
        
        // Read the bed data
        data.mpi_file_read_at_all <char*> (rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata);

        // Get number of ones, twos, and missing
        data.sparse_data_get_sizes_from_raw(rawdata, MLi, snpLenByt, N1, N2, NM);
        //printf("DEBUG  : off rank %d: N1 = %15lu, N2 = %15lu, NM = %15lu\n", rank, N1, N2, NM);

        rN1 += N1;
        rN2 += N2;
        rNM += NM;

        free(rawdata);

        MPI_Barrier(MPI_COMM_WORLD);
    }


    // Gather offsets
    // --------------
    size_t *AllN1 = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllN1, __LINE__, __FILE__);
    size_t *AllN2 = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllN2, __LINE__, __FILE__);
    size_t *AllNM = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllNM, __LINE__, __FILE__);

    check_mpi(MPI_Allgather(&rN1, 1, MPI_UNSIGNED_LONG_LONG, AllN1, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rN2, 1, MPI_UNSIGNED_LONG_LONG, AllN2, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rNM, 1, MPI_UNSIGNED_LONG_LONG, AllNM, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    //printf("DEBUG  : rN1 = %lu and AllN1[0] = %lu\n", rN1, AllN1[0]);

    size_t N1tot = 0, N2tot = 0, NMtot = 0;
    for (int i=0; i<nranks; i++) {
        N1tot += AllN1[i];
        N2tot += AllN2[i];
        NMtot += AllNM[i];
    }
    if (rank ==0 ) printf("INFO   : N1tot = %lu, N2tot = %lu, NMtot = %lu.\n", N1tot, N2tot, NMtot);

    // STEP 2: write sparse structure files
    // ------------------------------------
    if (rank ==0 ) printf("\nINFO   : begining of step 2, writing of the sparse files.\n");
    size_t tN1 = 0, tN2 = 0, tNM = 0;

    for (int i=0; i<bpr; ++i) {

        if (rank == 0) printf("INFO   : reading (2/2) starting block %3d out of %3d\n", i+1, bpr);

        uint globi = rank*bpr + i;
        int  MLi   = MrankL[globi];
        int  MSi   = MrankS[globi];
        //printf("DEBUG  : 2| bpr %i  MLi = %d, MSi = %d\n", i, MLi, MSi);

        // Alloc memory for raw BED data
        const size_t rawdata_n = size_t(MLi) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) malloc(rawdata_n);  check_malloc(rawdata, __LINE__, __FILE__);

        // Gather sizes to determine common number of reads
        size_t rawdata_n_max = 0;
        check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

        int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (NREADS <= 0) {
            if (rank == 0) printf("FATAL  : NREADS must be >= 1.");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Compute the offset of the section to read from the BED file
        offset = size_t(3) + size_t(MSi) * size_t(snpLenByt) * sizeof(char);

        data.mpi_file_read_at_all<char*>(rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata);

        // Alloc memory for sparse representation
        size_t *N1S, *N1L, *N2S, *N2L, *NMS, *NML;
        N1S = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N1S, __LINE__, __FILE__);
        N1L = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N1L, __LINE__, __FILE__);
        N2S = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N2S, __LINE__, __FILE__);
        N2L = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N2L, __LINE__, __FILE__);
        NMS = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(NMS, __LINE__, __FILE__);
        NML = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(NML, __LINE__, __FILE__);

        size_t N1 = 0, N2 = 0, NM = 0;
        data.sparse_data_get_sizes_from_raw(rawdata, uint(MLi), snpLenByt, N1, N2, NM);
        //printf("DEBUG  : N1 = %15lu, N2 = %15lu, NM = %15lu\n", N1, N2, NM);

        // Alloc and build sparse structure
        uint *I1, *I2, *IM;
        I1 = (uint*) malloc(N1 * sizeof(uint));  check_malloc(I1, __LINE__, __FILE__);
        I2 = (uint*) malloc(N2 * sizeof(uint));  check_malloc(I2, __LINE__, __FILE__);
        IM = (uint*) malloc(NM * sizeof(uint));  check_malloc(IM, __LINE__, __FILE__);

        // To check that each element is properly set
        //for (int i=0; i<N1; i++) I1[i] = UINT_MAX;
        //for (int i=0; i<N2; i++) I2[i] = UINT_MAX;
        //for (int i=0; i<NM; i++) IM[i] = UINT_MAX;
        
        data.sparse_data_fill_indices(rawdata, MLi, snpLenByt, N1S, N1L, I1,  N2S, N2L, I2,  NMS, NML, IM);
   
        //check_whole_array_was_set(I1, N1, __LINE__, __FILE__);
        //check_whole_array_was_set(I2, N2, __LINE__, __FILE__);
        //check_whole_array_was_set(IM, NM, __LINE__, __FILE__);

        // Compute the rank offset
        size_t N1Off = 0, N2Off = 0, NMOff = 0;
        for (int ii=0; ii<rank; ++ii) {
            N1Off += AllN1[ii];
            N2Off += AllN2[ii];
            NMOff += AllNM[ii];
        }

        N1Off += tN1;
        N2Off += tN2;
        NMOff += tNM;

        // ss1,2,m files must contain absolute start indices!
        for (int ii=0; ii<MLi; ++ii) {
            N1S[ii] += N1Off;
            N2S[ii] += N2Off;
            NMS[ii] += NMOff;
        }

        size_t N1max = 0, N2max = 0, NMmax = 0;
        check_mpi(MPI_Allreduce(&N1, &N1max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&N2, &N2max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&NM, &NMmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        int    MLimax = 0;
        check_mpi(MPI_Allreduce(&MLi, &MLimax, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        if (rank == 0) printf("INFO   : N1max = %lu, N2max = %lu, NMmax = %lu, MLimax = %lu\n", N1max, N2max, NMmax, MLimax);
        
        int NREADS1   = check_int_overflow(size_t(ceil(double(N1max)/double(INT_MAX/2))),  __LINE__, __FILE__);
        int NREADS2   = check_int_overflow(size_t(ceil(double(N2max)/double(INT_MAX/2))),  __LINE__, __FILE__);
        int NREADSM   = check_int_overflow(size_t(ceil(double(NMmax)/double(INT_MAX/2))),  __LINE__, __FILE__);
        int NREADSMLi = check_int_overflow(size_t(ceil(double(MLimax)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (rank == 0) printf("INFO   : NREADS1 = %d, NREADS2 = %d, NREADSM = %d, NREADSMLi = %d\n", NREADS1, NREADS2, NREADSM, NREADSMLi);

        // Sparse Ones files
        offset = N1Off * sizeof(uint);
        data.mpi_file_write_at_all <uint*>   (N1,  offset, si1fh, MPI_UNSIGNED,           NREADS1,   I1);
        offset = size_t(MSi) * sizeof(size_t);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, sl1fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N1L);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, ss1fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N1S);

        // Sparse Twos files
        offset = N2Off * sizeof(uint) ;
        data.mpi_file_write_at_all <uint*>   (N2,  offset, si2fh, MPI_UNSIGNED,           NREADS2,   I2);
        offset = size_t(MSi) * sizeof(size_t);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, sl2fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N2L);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, ss2fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N2S);

        // Sparse Missing files
        offset = NMOff * sizeof(uint) ;
        data.mpi_file_write_at_all <uint*>   (NM,  offset, simfh, MPI_UNSIGNED,           NREADSM,   IM);
        offset = size_t(MSi) * sizeof(size_t);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, slmfh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, NML);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, ssmfh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, NMS);

        // Free allocated memory
        free(rawdata);
        free(N1S); free(N1L); free(I1);
        free(N2S); free(N2L); free(I2);
        free(NMS); free(NML); free(IM);

        tN1 += N1;
        tN2 += N2;
        tNM += NM;
    }

    free(AllN1);
    free(AllN2);
    free(AllNM);

    // Sync
    MPI_Barrier(MPI_COMM_WORLD);    

    // check size of the written files!
    check_file_size(si1fh, N1tot, sizeof(uint),   __LINE__, __FILE__);
    check_file_size(si2fh, N2tot, sizeof(uint),   __LINE__, __FILE__);
    check_file_size(simfh, NMtot, sizeof(uint),   __LINE__, __FILE__);
    check_file_size(ss1fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(ss2fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(ssmfh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(sl1fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(sl2fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(slmfh, Mtot,  sizeof(size_t), __LINE__, __FILE__);


    // Close files
    check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&dimfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ssmfh), __LINE__, __FILE__);

    // Print approximate time for the conversion
    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to convert the data: " << du2 / double(1000.0) << " seconds." << std::endl;
}


size_t get_file_size(const std::string& filename) {
    struct stat st;
    if(stat(filename.c_str(), &st) != 0) { return 0; }
    return st.st_size;   
}



void mpi_assign_blocks_to_tasks(const uint numBlocks, const vector<int> blocksStarts, const vector<int> blocksEnds, const uint Mtot, const int nranks, const int rank, int* MrankS, int* MrankL) {

    if (numBlocks > 0) {

        if (nranks != numBlocks) {
            if (rank == 0) {
                printf("FATAL  : block definition does not match number of tasks (%d versus %d).\n", numBlocks, nranks);
                printf("        => Provide each task with a block definition\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // First and last markers (continuity is garanteed from reading)
        if (blocksStarts[0] != 1) {
            if (rank == 0) {
                printf("FATAL  : first marker in block definition file should be 1 but is %d\n", blocksStarts[0]);
                printf("        => Adjust block definition file\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (blocksEnds[numBlocks-1] != Mtot) {
            if (rank == 0) {
                printf("FATAL  : last marker in block definition file should be Mtot = %d whereas is %d\n", Mtot, blocksEnds[numBlocks-1]+1);
                printf("        => Adjust block definition file\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Assign to MrankS and MrankL to catch up on logic
        for (int i=0; i<numBlocks; ++i) {
            MrankS[i] = blocksStarts[i] - 1;                  // starts from 0, not 1
            MrankL[i] = blocksEnds[i] - blocksStarts[i] + 1;  // compute length
        }

    } else {
        if (rank == 0)
            printf("INFO   : no marker block definition file used. Will go for even distribution over tasks.\n");
        mpi_define_blocks_of_markers(Mtot, MrankS, MrankL, nranks);
    }
}



//EO: MPI GIBBS
//-------------
int BayesRRm::runMpiGibbs() {

    char   buff[LENBUF]; 
    int    nranks, rank, name_len, result;
    double dalloc = 0.0;

    //MPI_Init(NULL, NULL);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int  processor_name_len;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (opt.checkRam && nranks != 1) {
        printf("#FATAL#: --check-RAM option runs only in single task mode (SIMULATION of --check-RAM-tasks with max --check-RAM-tasks-per-node tasks per node)!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Get_processor_name(processor_name, &processor_name_len);

    MPI_File   bedfh, outfh, betfh, epsfh, cpnfh, acufh;
    MPI_Status status;
    MPI_Info   info;
    MPI_Offset offset, betoff, cpnoff, epsoff;

    // Set up processing options
    // -------------------------
    if (rank < 0) {
        opt.printBanner();
        opt.printProcessingOptions();
    }
    unsigned shuf_mark = opt.shuffleMarkers;
    unsigned sync_rate = opt.MPISyncRate;


    // Initialize MC on each worker
    // ----------------------------
    //cout << "Instantiating dist with seed = " << opt.seed + rank*1000 << endl;
    //Distributions_boost dist((uint)(opt.seed + rank*1000));
    dist.reset_rng((uint)(opt.seed + rank*1000), rank);

    const unsigned int max_it = opt.chainLength;
    
    unsigned int Ntot = opt.numberIndividuals; //data.numInds;
    if (Ntot == 0) 
        throw("FATAL  : opt.numberIndividuals is zero! Set it via --number-individuals in call.");
    if (Ntot != data.numInds - data.numNAs)
        printf("WARNING: opt.numberIndividuals set to %d but will be adjusted to %d - %d = %d due to NAs in phenotype file.\n", Ntot, data.numInds, data.numNAs, data.numInds-data.numNAs);

    unsigned int Mtot   = opt.numberMarkers;     //data.numSnps;
    if (Mtot == 0) throw("FATAL  : opt.numberMarkers is zero! Set it via --number-markers in call.");

    if (rank == 0) printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);
    
    // Block marker definition has precedence over requested number of markers
    if (opt.markerBlocksFile != "" && opt.numberMarkers > 0) {
        opt.numberMarkers = 0;
        if (rank == 0) printf("WARNING: --number-markers option ignored, a marker block definition file was passed!\n");
    }        
    
    if (opt.numberMarkers > 0 && opt.numberMarkers < Mtot) {
        Mtot = opt.numberMarkers;
        if (rank == 0) printf("Option passed to process only %d markers!\n", Mtot);
    }

    // Get name of sparse files to read from (default case)
    // ----------------------------------------------------
    std::string sparseOut = mpi_get_sparse_output_filebase();


    // Alloc memory for sparse representation
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    uint   *I1,         *I2,         *IM;
    size_t  N1=0, N2=0, NM=0;

    N1S = (size_t*)malloc(size_t(Mtot) * sizeof(size_t));  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)malloc(size_t(Mtot) * sizeof(size_t));  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)malloc(size_t(Mtot) * sizeof(size_t));  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)malloc(size_t(Mtot) * sizeof(size_t));  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)malloc(size_t(Mtot) * sizeof(size_t));  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)malloc(size_t(Mtot) * sizeof(size_t));  check_malloc(NML, __LINE__, __FILE__);
    dalloc += 6.0 * double(Mtot) * sizeof(size_t) / 1E9;


    // Estimate RAM usage
    // ------------------
    if (opt.checkRam) {
        if (opt.checkRamTasks <= 0) {
            printf("#FATAL#: --check-RAM-tasks must be strictly positive! Was %d\n", opt.checkRamTasks);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (opt.checkRamTpn <= 0) {
            printf("#FATAL#: --check-RAM-tasks-per-node must be strictly positive! Was %d\n", opt.checkRamTpn);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_File slfh;

        // Read length files once for all
        string sl;
        sl = sparseOut + ".sl1";
        check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);
        check_mpi(MPI_File_read_at_all(slfh, 0, N1L, Mtot, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);
        sl = sparseOut + ".sl2";
        check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);
        check_mpi(MPI_File_read_at_all(slfh, 0, N2L, Mtot, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);
        sl = sparseOut + ".slm";
        check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);
        check_mpi(MPI_File_read_at_all(slfh, 0, NML, Mtot, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);
        

        // Given
        int tpn = opt.checkRamTpn;

        int nranks = opt.checkRamTasks;
        if (opt.markerBlocksFile != "") nranks = data.numBlocks;

        int nnodes = int(ceil(double(nranks)/double(tpn)));
        printf("INFO  : will simulate %d ranks on %d nodes with max %d tasks per node.\n", nranks, nnodes, tpn);

        int proctasks = 0;
                
        printf("Estimation RAM usage when dataset is processed with %2d nodes and %2d tasks per node\n", nnodes, tpn);
        
        int MrankS[nranks], MrankL[nranks];
        mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL);
        //mpi_define_blocks_of_markers(Mtot, MrankS, MrankL, nranks);
        
        int lmax = 0, lmin = 1E9;
        for (int i=0; i<nranks; ++i) {
            if (MrankL[i]>lmax) lmax = MrankL[i];
            if (MrankL[i]<lmin) lmin = MrankL[i];
        }
        printf("INFO   : longest  task has %d markers.\n", lmax);
        printf("INFO   : smallest task has %d markers.\n", lmin);
        double min = 1E9, max = 0.0;
        int nodemin = 0, nodemax = 0;

        // Replicate SLURM block task assignment strategy
        // ----------------------------------------------
        const int nfull = nranks + nnodes * (1 - tpn);
        printf("INFO   : number of nodes fully loaded: %d\n", nfull);

        // Save max
        const int tpnmax = tpn;

        for (int node=0; node<nnodes; node++) {
            
            double ramnode = 0.0;
            
            if (node >= nfull) tpn = tpnmax - 1;

            // Array of pointers allocated memory
            uint *allocs1[tpn], *allocs2[tpn], *allocsm[tpn];
            
            for (int i=0; i<tpn; i++) {
                size_t n1 = 0, n2 = 0, nm = 0;
                for (int m=0; m<MrankL[node*tpn + i]; m++) {
                    n1 += N1L[MrankS[node*tpn + i] + m];
                    n2 += N2L[MrankS[node*tpn + i] + m];
                    nm += NML[MrankS[node*tpn + i] + m];                
                }
                double GB = double((n1+n2+nm)*sizeof(uint))*1E-9;
                ramnode += GB;
                printf("   - t %3d  n %2d will attempt to alloc %.3f + %.3f + %.3f GB of RAM\n", i, node, n1*sizeof(uint)*1E-9,  n2*sizeof(uint)*1E-9,  nm*sizeof(uint)*1E-9);

                allocs1[i] = (uint*) malloc(n1 * sizeof(uint));  check_malloc(allocs1[i], __LINE__, __FILE__);
                allocs2[i] = (uint*) malloc(n2 * sizeof(uint));  check_malloc(allocs2[i], __LINE__, __FILE__);
                allocsm[i] = (uint*) malloc(nm * sizeof(uint));  check_malloc(allocsm[i], __LINE__, __FILE__);

                printf("   - t %3d  n %2d sm %7d  l %6d markers. Number of 1s: %15lu, 2s: %15lu, ms: %15lu => RAM: %7.3f GB; RAM on node: %7.3f with %d tasks\n", i, node, MrankS[node*tpn + i], MrankL[node*tpn + i], n1, n2, nm, GB, ramnode, tpn);

                proctasks++;
            }
            
            // free memory on the node
            for (int i=0; i<tpn; i++) { 
                free(allocs1[i]);
                free(allocs2[i]);
                free(allocsm[i]);
            }
            
            if (ramnode < min) { min = ramnode; nodemin = node; }
            if (ramnode > max) { max = ramnode; nodemax = node; }

        }

        if (proctasks != nranks) {
            printf("#FATAL#: Cannot fit %d tasks on %d nodes with %d x %d + %d x %d tasks per node! Ended up with %d.\n", nranks, nnodes, nfull, tpnmax, nnodes-nfull, tpn, proctasks);
        }
        
        printf("\n");
        printf("    => max RAM required on a node will be %7.3f GB on node %d\n", max, nodemax);
        printf("    => setting up your sbatch with %d tasks and %d tasks per node should work; Will require %d nodes!\n", nranks, tpnmax, nnodes);
        printf("\n");
        
        // Free previously allocated memory
        free(N1S); free(N1L);
        free(N2S); free(N2L);
        free(NMS); free(NML);

        // Do no process anything
        return 0;
    }


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks];
    //mpi_assign_blocks_to_tasks(MrankS, MrankL, data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot);
    mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL);


    int lmax = 0, lmin = 1E9;
    for (int i=0; i<nranks; ++i) {
        if (MrankL[i]>lmax) lmax = MrankL[i];
        if (MrankL[i]<lmin) lmin = MrankL[i];
    }
    if (rank == 0) {
        printf("INFO   : longest  task has %d markers.\n", lmax);
        printf("INFO   : smallest task has %d markers.\n", lmin);
    }

    int M = MrankL[rank];
    //printf("rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);


    const double        sigma0 = 0.0001;
    const double        v0E    = 0.0001;
    const double        s02E   = 0.0001;
    const double        v0G    = 0.0001;
    const double        s02G   = 0.0001;
    const double        v0F    = 0.001;
    const double        s02F   = 1.0;
    const unsigned int  K      = int(cva.size()) + 1;
    const unsigned int  km1    = K - 1;
    std::vector<int>    markerI;
    VectorXi            components(M);
    double              mu;              // mean or intercept
    double              sigmaG;          // genetic variance
    double              sigmaE;          // residuals variance
    double              sigmaF;          // covariates variance if using ridge;
    VectorXd            priorPi(K);      // prior probabilities for each component
    VectorXd            pi(K);           // mixture probabilities
    VectorXd            muk(K);          // mean of k-th component marker effect size
    VectorXd            denom(K-1);      // temporal variable for computing the inflation of the effect variance for a given non-zero component
    VectorXd            cVa(K);          // component-specific variance
    VectorXd            cVaI(K);         // inverse of the component variances
    double              num;             // storing dot product
    //int                 m0;              // total number of markes in model
    double              m0;
    VectorXd            v(K);            // variable storing the component assignment
    VectorXd            sum_v(K);        // To store the sum of v elements over all ranks

    VectorXd            Beta(M);
    VectorXd            Acum(M);
    VectorXd            Gamma(data.numFixedEffects);


    //marion : for annotation code
    /*
    sigmaGG = VectorXd(groupCount); 	//vector with sigmaG (variance) for each annotation
    betasqnG = VectorXd(groupCount);	//vector with sum of beta squared for each annotation
    v = MatrixXd(groupCount,K);         // variable storing the component assignment
   */


    dalloc +=     M * sizeof(int)    / 1E9; // for components
    dalloc += 2 * M * sizeof(double) / 1E9; // for Beta and Acum

    gamma.setZero();

    //fixed effects matrix
    X = data.X;

    //VectorXd            sample(2*M+4+N); // varible containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance


    // Opne BED and output files
    std::string bedfp = opt.bedFile;
    std::string outfp = opt.mcmcOut + ".csv";
    std::string betfp = opt.mcmcOut + ".bet";
    std::string epsfp = opt.mcmcOut + ".eps";
    std::string cpnfp = opt.mcmcOut + ".cpn";
    std::string acufp = opt.mcmcOut + ".acu";

    // Delete old files
    if (rank == 0) {
        MPI_File_delete(outfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(acufp.c_str(), MPI_INFO_NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, acufp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &acufh), __LINE__, __FILE__);

    // First element of the .bet, .cpn and .acu files is the total number of processed markers
    betoff = size_t(0);
    if (rank == 0) {
        check_mpi(MPI_File_write_at(betfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(acufh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    // Preprocess the data
    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();
    

    // READING FROM BED FILE
    // ---------------------
    if (opt.readFromBedFile) {

        bedfp += ".bed";
        check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__);

        // Length of a "column" in bytes
        const size_t snpLenByt = (Ntot % 4) ? Ntot / 4 + 1 : Ntot / 4;
        if (rank==0) printf("INFO   : marker length in bytes (snpLenByt) = %zu bytes.\n", snpLenByt);

        // Alloc memory for raw BED data
        // -----------------------------
        const size_t rawdata_n = size_t(M) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) malloc(rawdata_n);  check_malloc(rawdata, __LINE__, __FILE__);
        //dalloc += rawdata_n / 1E9;
        //printf("rank %d allocation %zu bytes (%.3f GB) for the raw data.\n",           rank, rawdata_n, double(rawdata_n/1E9));

        // Compute the offset of the section to read from the BED file
        // -----------------------------------------------------------
        offset = size_t(3) + size_t(MrankS[rank]) * size_t(snpLenByt) * sizeof(char);

        // Read the BED file
        // -----------------
        MPI_Barrier(MPI_COMM_WORLD);
        const auto st1 = std::chrono::high_resolution_clock::now();

        // Gather the sizes to determine common number of reads
        size_t rawdata_n_max = 0;
        check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

        int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);

        data.mpi_file_read_at_all <char*> (rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata);

        MPI_Barrier(MPI_COMM_WORLD);

        const auto et1 = std::chrono::high_resolution_clock::now();
        const auto dt1 = et1 - st1;
        const auto du1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt1).count();
        if (rank == 0)  std::cout << "INFO   : time to read the BED file: " << du1 / double(1000.0) << " seconds." << std::endl;

        // Close BED file
        check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);
        
        data.sparse_data_get_sizes_from_raw(rawdata, M, snpLenByt, N1, N2, NM);

        // Alloc and build sparse structure
        I1 = (uint*) malloc(N1 * sizeof(uint));  check_malloc(I1, __LINE__, __FILE__);
        I2 = (uint*) malloc(N2 * sizeof(uint));  check_malloc(I2, __LINE__, __FILE__);
        IM = (uint*) malloc(NM * sizeof(uint));  check_malloc(IM, __LINE__, __FILE__);
        dalloc += (N1 + N2 + NM) * sizeof(size_t) / 1E9;

        // To check that each element is properly set
        //for (int i=0; i<N1; i++) I1[i] = UINT_MAX;
        //for (int i=0; i<N2; i++) I2[i] = UINT_MAX;
        //for (int i=0; i<NM; i++) IM[i] = UINT_MAX;

        data.sparse_data_fill_indices(rawdata, M, snpLenByt,
                                      N1S, N1L, I1,
                                      N2S, N2L, I2,
                                      NMS, NML, IM);

        free(rawdata);        
    } 
    // READING FROM SPARSE FILES (DEFAULT CASE)
    // ----------------------------------------
    else {

        // Get sizes to alloc for the task
        N1 = data.get_number_of_elements_from_sparse_files(sparseOut, "1", MrankS, MrankL, N1S, N1L);
        N2 = data.get_number_of_elements_from_sparse_files(sparseOut, "2", MrankS, MrankL, N2S, N2L);
        NM = data.get_number_of_elements_from_sparse_files(sparseOut, "m", MrankS, MrankL, NMS, NML);

        // To gather sizes to determine common number of reads
        //size_t *AllN1 = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllN1, __LINE__, __FILE__);
        //size_t *AllN2 = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllN2, __LINE__, __FILE__);
        //size_t *AllNM = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllNM, __LINE__, __FILE__);

        size_t N1max = 0, N2max = 0, NMmax = 0;
        check_mpi(MPI_Allreduce(&N1, &N1max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&N2, &N2max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&NM, &NMmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

        size_t N1tot = 0, N2tot = 0, NMtot = 0;
        check_mpi(MPI_Allreduce(&N1, &N1tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&N2, &N2tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&NM, &NMtot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

        if (rank == 0) printf("INFO   : rank %3d/%3d  N1max = %15lu, N2max = %15lu, NMmax = %15lu\n", rank, nranks, N1max, N2max, NMmax);
        if (rank == 0) printf("INFO   : rank %3d/%3d  N1tot = %15lu, N2tot = %15lu, NMtot = %15lu\n", rank, nranks, N1tot, N2tot, NMtot);
        if (rank == 0) printf("INFO   : RAM for task %3d/%3d on node %s: %7.3f GB\n", rank, nranks, processor_name, (N1+N2+NM)*sizeof(uint)/1E9);

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("INFO   : Total RAM for storing sparse indices %.3f GB\n", (N1tot+N2tot+NMtot)*sizeof(uint)/1E9);

        I1 = (uint*) malloc(N1 * sizeof(uint));  check_malloc(I1, __LINE__, __FILE__);
        I2 = (uint*) malloc(N2 * sizeof(uint));  check_malloc(I2, __LINE__, __FILE__);
        IM = (uint*) malloc(NM * sizeof(uint));  check_malloc(IM, __LINE__, __FILE__);
        dalloc += (N1 + N2 + NM) * sizeof(uint) / 1E9;

        // To check that each element is properly set
        //for (int i=0; i<N1; i++) I1[i] = UINT_MAX;
        //for (int i=0; i<N2; i++) I2[i] = UINT_MAX;
        //for (int i=0; i<NM; i++) IM[i] = UINT_MAX;

        int NREADS1 = check_int_overflow(size_t(ceil(double(N1max)/double(INT_MAX/2))), __LINE__, __FILE__);
        int NREADS2 = check_int_overflow(size_t(ceil(double(N2max)/double(INT_MAX/2))), __LINE__, __FILE__);
        int NREADSM = check_int_overflow(size_t(ceil(double(NMmax)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (rank == 0) printf("INFO   : number of call to read the sparse files: NREADS1 = %d, NREADS2 = %d, NREADSM = %d\n", NREADS1, NREADS2, NREADSM);

        data.read_sparse_data_file(sparseOut + ".si1", N1, N1S[0], NREADS1, I1);
        data.read_sparse_data_file(sparseOut + ".si2", N2, N2S[0], NREADS2, I2);
        data.read_sparse_data_file(sparseOut + ".sim", NM, NMS[0], NREADSM, IM);

        // Make starts relative to start of block in each task
        const size_t n1soff = N1S[0];  for (int i=0; i<M; ++i) { N1S[i] -= n1soff; }
        const size_t n2soff = N2S[0];  for (int i=0; i<M; ++i) { N2S[i] -= n2soff; }
        const size_t nmsoff = NMS[0];  for (int i=0; i<M; ++i) { NMS[i] -= nmsoff; }
    }

    // Sanity check: make sure each element was set by read
    //check_whole_array_was_set(I1, N1, __LINE__, __FILE__);
    //check_whole_array_was_set(I2, N2, __LINE__, __FILE__);
    //check_whole_array_was_set(IM, NM, __LINE__, __FILE__);


    // Correct each marker for individuals with missing phenotype
    // ----------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);


    const int NNAS = data.numNAs;

    if (rank == 0) printf("INFO   : applying %d corrections to genotype data due to missing phenotype data (NAs in .phen).\n", NNAS);
    
    for (int ii=0; ii<M; ++ii) {

        if (rank >= 0 && ii%100 == 0) printf("INFO   : task %3d applying %3d NA corrections to marker %5d out of %5d\n", rank, NNAS, ii, M);
        
        size_t beg = 0, len = 0, end = 0;

        for (int i=0; i<NNAS; ++i) {

            beg = N1S[ii]; len = N1L[ii];
            if (len > 0) {
                for (size_t iii=beg; iii<beg+len; ++iii) {
                    if (I1[iii] + i == data.NAsInds[i]) { 
                        N1L[ii] -= 1; 
                        for (size_t k = iii; k<beg+N1L[ii]; k++) I1[k] = I1[k + 1] - 1;                        
                        break;
                    } else {
                        if (I1[iii] + i >= data.NAsInds[i]) I1[iii] = I1[iii] - 1;
                    }
                }
            }

            beg = N2S[ii]; len = N2L[ii];
            if (len > 0) {
                for (size_t iii=beg; iii<beg+len; ++iii) {
                    if (I2[iii] + i == data.NAsInds[i]) { 
                        N2L[ii] -= 1;
                        for (size_t k = iii; k<beg+N2L[ii]; k++) I2[k] = I2[k + 1] - 1;
                        break;
                    } else {
                        if (I2[iii] + i >= data.NAsInds[i]) I2[iii] = I2[iii] - 1;
                    }
                }
            }

            beg = NMS[ii]; len = NML[ii];
            if (len > 0) {
                for (size_t iii=beg; iii<beg+len; ++iii) {
                    if (IM[iii] + i == data.NAsInds[i]) { 
                        NML[ii] -= 1;
                        for (size_t k = iii; k<beg+NML[ii]; k++) IM[k] = IM[k + 1] - 1;
                        break;
                    } else {
                        if (IM[iii] + i >= data.NAsInds[i]) IM[iii] = IM[iii] - 1;
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : finished applying NA corrections.\n");
    

    // Adjust N upon number of NAs
    // ---------------------------
    Ntot -= data.numNAs;
    if (rank == 0) printf("INFO   : Ntot adjusted by -%d to account for NAs in phenotype file. Now Ntot=%d\n", data.numNAs, Ntot);

    double dN   = (double) Ntot;
    double dNm1 = (double)(Ntot - 1);


    // Compute statistics (from sparse info)
    // -------------------------------------
    if (rank == 0) printf("INFO   : start computing statistics on Ntot = %d individuals\n", Ntot);
    double *mave, *mstd;
    mave = (double*)malloc(size_t(M) * sizeof(double));  check_malloc(mave, __LINE__, __FILE__);
    mstd = (double*)malloc(size_t(M) * sizeof(double));  check_malloc(mstd, __LINE__, __FILE__);
    dalloc += 2 * size_t(M) * sizeof(double) / 1E9;
    
    double tmp0, tmp1, tmp2;
    for (int i=0; i<M; ++i) {
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (dN - double(NML[i]));        
        tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
        //printf("marker %6d mean %20.15f, std = %20.15f\n", i, mave[i], mstd[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Build list of markers    
    // ---------------------
    for (int i=0; i<M; ++i) markerI.push_back(i);
    //printf("markerI start = %d and end = %d\n", markerI[0], markerI[M-1]);
    //std::iota(markerI.begin(), markerI.end(), 0);


    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();
 
    double *y, *epsilon, *tmpEps, *previt_eps, *deltaEps, *dEpsSum, *deltaSum;
    const size_t NDB = size_t(Ntot) * sizeof(double);
    y          = (double*)malloc(NDB);  check_malloc(y,        __LINE__, __FILE__);
    epsilon    = (double*)malloc(NDB);  check_malloc(epsilon,  __LINE__, __FILE__);
    tmpEps     = (double*)malloc(NDB);  check_malloc(tmpEps,   __LINE__, __FILE__);
    previt_eps = (double*)malloc(NDB);  check_malloc(tmpEps,   __LINE__, __FILE__);
    deltaEps   = (double*)malloc(NDB);  check_malloc(deltaEps, __LINE__, __FILE__);
    dEpsSum    = (double*)malloc(NDB);  check_malloc(dEpsSum,  __LINE__, __FILE__);
    deltaSum   = (double*)malloc(NDB);  check_malloc(deltaSum, __LINE__, __FILE__);
    dalloc += NDB * 7 / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);

    priorPi[0] = 0.5;
    cVa[0]     = 0.0;
    cVaI[0]    = 0.0;
    muk[0]     = 0.0;
    mu         = 0.0;


    //marion : init prior for each annotation
    /*
    m_cVa[0] = 0;
    m_cVaI[0] = 0;

    for(int i=0; i < groupCount; i++){
    	m_priorPi.row(i)(0)=0.5;
        for(int k=1;k<K;k++){
        	m_priorPi.row(i)(k)=0.5/K;
        }
    }

    m_y_tilde.setZero();
    m_beta.setZero();

    for(int i=0; i<groupCount;i++)
       m_sigmaGG[i] = m_dist.beta_rng(1,1);

    m_pi = m_priorPi;
    */


    for (int i=0; i<Ntot; ++i) dEpsSum[i] = 0.0;

    cVa.segment(1,km1)     = cva;
    cVaI.segment(1,km1)    = cVa.segment(1,km1).cwiseInverse();
    priorPi.segment(1,km1) = priorPi[0] * cVa.segment(1,km1).array() / cVa.segment(1,km1).sum();
    sigmaG                 = dist.beta_rng(1.0, 1.0);
    pi                     = priorPi;
    Beta.setZero();
    components.setZero();

    if (opt.covariates) {
    	gamma = VectorXd(data.X.cols());
    	gamma.setZero();
    }
    
    double y_mean = 0.0;
    for (int i=0; i<Ntot; ++i) {
        //if (i<30) printf("y(%d) = %15.10f\n", i, data.y(i));
        y[i]    = data.y(i);
        y_mean += y[i];
    }
    y_mean /= Ntot;
    //printf("rank %4d: y_mean = %20.15f with Ntot = %d\n", rank, y_mean, Ntot);

    for (int i=0; i<Ntot; ++i) y[i] -= y_mean;

    double y_sqn = 0.0d;
    for (int i=0; i<Ntot; ++i) y_sqn += y[i] * y[i];
    y_sqn = sqrt(y_sqn / dNm1);
    //printf("ysqn = %15.10f\n", y_sqn);

    sigmaE = 0.0d;
    for (int i=0; i<Ntot; ++i) {
        y[i]       /= y_sqn;
        epsilon[i]  = y[i]; // - mu but zero
        sigmaE     += epsilon[i] * epsilon[i];
    }
    sigmaE = sigmaE / dN * 0.5;
    //printf("sigmaE = %20.10f with epsilon = y-mu %22.15f\n", sigmaE, mu);

    double   sum_beta_squaredNorm;
    double   sigE_G, sigG_E, i_2sigE;
    double   beta, betaOld, deltaBeta, beta_squaredNorm, p, acum, e_sqn;
    size_t   markoff;
    int      marker, left;
    VectorXd logL(K);
    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    sigmaF = s02F;


    // Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // -----------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    mpi_define_blocks_of_markers(Ntot, IrankS, IrankL, nranks);


    // Adapt the --thin and --save options such that --save >= --thin and --save%--thin = 0
    // ------------------------------------------------------------------------------------
    if (opt.save < opt.thin) {
        opt.save = opt.thin;
        if (rank == 0) printf("WARNING: opt.save was lower that opt.thin ; opt.save reset to opt.thin (%d)\n", opt.thin);
    }
    if (opt.save%opt.thin != 0) {
        if (rank == 0) printf("WARNING: opt.save (= %d) was not a multiple of opt.thin (= %d)\n", opt.save, opt.thin);
        opt.save = int(opt.save/opt.thin) * opt.thin;
        if (rank == 0) printf("         opt.save reset to %d, the closest multiple of opt.thin (%d)\n", opt.save, opt.thin);
    }

    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;


    double   previt_m0 = 0.0;
    double   previt_sg = 0.0;
    double   previt_mu = 0.0;
    double   previt_se = 0.0;
    VectorXd previt_Beta(M);


    // Main iteration loop
    // -------------------
    //bool replay_it = false;

    for (uint iteration=0; iteration < max_it; iteration++) {

        double start_it = MPI_Wtime();
        
        //if (replay_it) {
        //    printf("INFO: replay iteration with m0=%.0f sigG=%15.10f sigE=%15.10f\n", previt_m0, previt_sg, previt_se);
        //    m0        = previt_m0;
        //    sigmaG    = previt_sg;
        //    sigmaE    = previt_se;
        //    mu        = previt_mu;
        //    sync_rate = 0;
        //    for (int i=0; i<Ntot; ++i) epsilon[i] = previt_eps[i];
        //    Beta      = previt_Beta;
        //    replay_it = false;
        //}


        // Store status of iteration to revert back to it if required
        // ----------------------------------------------------------
        previt_m0 = m0;

        //marion : this should probably be a vector with sigmaGG
        previt_sg = sigmaG;
        //----------------

        previt_se = sigmaE;
        previt_mu = mu;
        for (int i=0; i<Ntot; ++i) previt_eps[i]  = epsilon[i];
        for (int i=0; i<M;    ++i) previt_Beta(i) = Beta(i);

        //printf("mu = %15.10f   eps[0] = %15.10f\n", mu, epsilon[0]);
        for (int i=0; i<Ntot; ++i) epsilon[i] += mu;
        
        double epssum  = 0.0;
        for (int i=0; i<Ntot; ++i) epssum += epsilon[i];
        //if (rank == 0) printf("epssum = %20.15f with Ntot=%d elements\n", epssum, Ntot);

        // update mu
        mu = dist.norm_rng(epssum / dN, sigmaE / dN);
        //printf("mu = %15.10f\n", mu);

        // We substract again now epsilon =Y-mu-X*beta
        for (int i=0; i<Ntot; ++i) epsilon[i] -= mu;

        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------
        if (shuf_mark) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
            //std::random_shuffle(markerI.begin(), markerI.end());
        }

        m0 = 0.0d;
        v.setZero();


        //marion : for each marker
        // get sigmaGG : to which annotation the marker belongs
        // then use this sigmaGG as sigmaG for this marker
        /*
        double sigmaG_process; // we could keep sigmaG instead of sigmaG_process
        sigmaG_process = sigmaGG[data->G(marker->i)];

        // set variable cVa
        cVa.segment(1, km1) = cva.row(data->G(marker->i));
        cVaI.segment(1, km1) = cVa.segment(1, km1).cwiseInverse();
        */


        sigE_G  = sigmaE / sigmaG;
        sigG_E  = sigmaG / sigmaE;
        i_2sigE = 1.0 / (2.0 * sigmaE);

        for (int i=0; i<Ntot; ++i) tmpEps[i] = epsilon[i];

        double cumSumDeltaBetas = 0.0;
        int sinceLastSync = 0;


        // Loop over (shuffled) markers
        // ----------------------------
        for (int j = 0; j < lmax; j++) {

            if (j < M) {
                marker  = markerI[j];
                
                beta =  Beta(marker);
                //printf("rank %d, marker %7d: beta = %20.15f, mean = %20.15f, std = %20.15f\n", rank, marker, beta, mave[marker], mstd[marker]);
                
                //we compute the denominator in the variance expression to save computations
                //denom = dNm1 + sigE_G * cVaI.segment(1, km1).array();
                for (int i=1; i<=km1; ++i) {
                    denom(i-1) = dNm1 + sigE_G * cVaI(i);
                    //printf("denom[%d] = %20.15f\n", i-1, denom(i-1));
                }

                num = sparse_dotprod(epsilon,
                                     I1, N1S[marker], N1L[marker],
                                     I2, N2S[marker], N2L[marker],
                                     IM, NMS[marker], NML[marker],
                                     mave[marker],    mstd[marker], Ntot, marker);

                num += beta * double(Ntot - 1);
                //printf("num = %20.15f\n", num);

                //muk for the other components is computed according to equations
                muk.segment(1, km1) = num / denom.array();           
                

                //marion : update the logL for each component of corresponding annotation
                /*
                VectorXd logL(K);
                const double logLScale = sigmaG_process / m_sigmaE * NM1;
                logL = m_pi.row(m_data->G(marker->i)).array().log();
                // First component probabilities remain unchanged
                logL.segment(1, km1) = logL.segment(1, km1).array()
                		- 0.5 * ((logLScale * m_cVa.segment(1, km1).array() + 1).array().log())
                        + 0.5 * (m_muk.segment(1, km1).array() * num) / m_sigmaE;
                */

                //first component probabilities remain unchanged
                logL = pi.array().log();
                
                // Update the log likelihood for each component
                logL.segment(1,km1) = logL.segment(1, km1).array()
                    - 0.5d * (sigG_E * dNm1 * cVa.segment(1,km1).array() + 1.0d).array().log() 
                    + muk.segment(1,km1).array() * num * i_2sigE;
                
                p = dist.unif_rng();
                //printf("%d/%d/%d  p = %15.10f\n", iteration, rank, j, p);
                
                acum = 0.0;
                if(((logL.segment(1,km1).array()-logL[0]).abs().array() > 700 ).any() ){
                    acum = 0.0;
                } else{
                    acum = 1.0 / ((logL.array()-logL[0]).exp().sum());
                }
                //printf("acum = %15.10f\n", acum);
                

                //marion : store marker acum
                /*
                for (int k = 0; k < K; k++) {
                	if (p <= acum) {
                	//if zeroth component
                           if (k == 0) {
                               beta(marker->i) = 0;
                           } else {
                               beta(marker->i) = dist.norm_rng(m_muk[k], sigmaE/denom[k-1]);
                               betasqnG(data->G(marker->i))+= pow(beta(marker->i),2);
                           }
                           v.row(data->G(marker->i))(k)+=1.0;
                           components[marker->i] = k;
                           break;
                       	 } else {
                           //if too big or too small
                           if (((logL.segment(1, km1).array() - logL[k+1]).abs().array() > 700).any()) {
                               acum += 0;
                           } else {
                               acum += 1.0 / ((logL.array() - logL[k+1]).exp().sum());
                           }
                       }
                   }
                */


                // Store marker acum for later dump to file
                Acum(marker) = acum;

                for (int k=0; k<K; k++) {
                    if (p <= acum || k == km1) { //if we p is less than acum or if we are already in the last mixt.
                        if (k==0) {
                            Beta(marker) = 0.0;
                        } else {
                            Beta(marker) = dist.norm_rng(muk[k], sigmaE/denom[k-1]);
                            //printf("@B@ beta update %4d/%4d/%4d muk[%4d] = %15.10f with p=%15.10f <= acum=%15.10f\n", iteration, rank, marker, k, muk[k], p, acum);
                        }
                        v[k] += 1.0d;
                        components[marker] = k;
                        break;
                    } else {
                        //if too big or too small
                        if (k+1 >= K) {
                            printf("FATAL  : iteration %d, marker = %d, p = %15.10f, acum = %15.10f logL overflow with %d => %d\n", iteration, marker, p, acum, k+1, K);
                            MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                        if (((logL.segment(k+1,K-(k+1)).array()-logL[k+1]).abs().array() > 700.0d ).any()) {
                            acum += 0.0d; // we compare next mixture to the others, if to big diff we skip
                        } else{
                            acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum()); //if not , sample
                        }
                    }
                }
                //printf("acum = %15.10f\n", acum);
                
                
                betaOld   = beta;
                beta      = Beta(marker);
                deltaBeta = betaOld - beta;
                
                // Compute delta epsilon
                if (deltaBeta != 0.0) {
                    sparse_scaadd(deltaEps, deltaBeta, 
                                  I1, N1S[marker], N1L[marker], 
                                  I2, N2S[marker], N2L[marker], 
                                  IM, NMS[marker], NML[marker], 
                                  mave[marker], mstd[marker], Ntot);
                    
                    // Update local sum of delta epsilon
                    for (int i=0; i<Ntot; ++i)
                        dEpsSum[i] += deltaEps[i];
                }
            } 

            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else {
                //cout << "rank " << rank << " with M=" << M << " waiting for " << lmax << endl;
                deltaBeta = 0.0;
                for (int i=0; i<Ntot; ++i)
                    deltaEps[i] = 0.0;
            }

            // Check whether we have a non-zero beta somewhere
            // sum of the abs values !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (nranks > 1) { 
                double sumDeltaBetas = 0.0;
                check_mpi(MPI_Allreduce(&deltaBeta, &sumDeltaBetas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                cumSumDeltaBetas += sumDeltaBetas;
            } else {
                cumSumDeltaBetas += deltaBeta;
            } 
            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, beta, cumSumDeltaBetas);
            
            //if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == M-1) && cumSumDeltaBetas != 0.0) {
            if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1) && cumSumDeltaBetas != 0.0) {

                // Update local copy of epsilon
                if (nranks > 1) {
                    check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                    for (int i=0; i<Ntot; ++i) epsilon[i] = tmpEps[i] + deltaSum[i];
                } else {
                    for (int i=0; i<Ntot; ++i) epsilon[i] = tmpEps[i] + dEpsSum[i];
                }

                // Store epsilon state at last synchronization
                for (int i=0; i<Ntot; ++i) tmpEps[i] = epsilon[i];                    
                
                // Reset local sum of delta epsilon
                for (int i=0; i<Ntot; ++i) dEpsSum[i] = 0.0;
                
                // Reset cumulated sum of delta betas
                cumSumDeltaBetas = 0.0;

                sinceLastSync = 0;
            } else {
                sinceLastSync += 1;
            }

        } // END PROCESSING OF ALL MARKERS

        beta_squaredNorm = Beta.squaredNorm();
        //printf("rank %d it %d  beta_squaredNorm = %15.10f\n", rank, iteration, beta_squaredNorm);


        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            check_mpi(MPI_Allreduce(&beta_squaredNorm, &sum_beta_squaredNorm, 1,        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(v.data(),          sum_v.data(),          v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            v                = sum_v;
            beta_squaredNorm = sum_beta_squaredNorm;
        }

        // Update global parameters
        // ------------------------
        m0      = double(Mtot) - v[0];
        sigmaG  = dist.inv_scaled_chisq_rng(v0G+m0, (beta_squaredNorm * m0 + v0G*s02G) /(v0G+m0));
        

        // marion : update sigmaGG
        /*
 	 	for(int i = 0; i < nGroups; i++){
            m_m0 = m_v.row(i).sum() - m_v.row(i)(0);
            m_sigmaGG[i] = m_dist.inv_scaled_chisq_rng(m_v0G + m_m0, (m_betasqnG(i) * m_m0 + m_v0G * m_s02G) / (m_v0G + m_m0));
            m_pi.row(i) = m_dist.dirichilet_rng(m_v.row(i).array() + 1.0);
        }
        */


        // Check iteration
        // 
        // ---------------
        /*
        if (iteration >= 0) {
            
            double max_sg = 0.0;
            MPI_Allreduce(&sigmaG, &max_sg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("INFO   : max sigmaG of iteration %d is %15.10f with sync rate of %d\n", iteration, max_sg, sync_rate);

            if (max_sg > 1.0) {
                if (sync_rate == 0) {
                    if (rank == 0) {
                        printf("CRITICAL: detected task with sigmaG = %15.10f and sync rate of %d\n", max_sg, sync_rate); 
                        printf("          => desperate situation, aborting...\n");
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                } else {
                    if (rank == 0)
                        printf("          => will do an attempt with setting sync_rate to 0\n");
                    replay_it = true;
                    continue;
                }
            }

            //if ( m0 > 1.2 * previt_m0 || m0 < 0.8 * previt_m0) {
            //    printf("CRITICAL: divergence detected! Will cancel iteration and set up a lower sync rate\n");
            //}
        }
        */

        // For the fixed effects
        // ---------------------
        if (opt.covariates) {

            if (rank == 0) {

                std::shuffle(xI.begin(), xI.end(), dist.rng);
                double gamma_old, num_f, denom_f;
                double sigE_sigF = sigmaE / sigmaF;
                //cout << "data.X.cols " << data.X.cols() << endl;
                for (int i=0; i<data.X.cols(); i++) {
                    gamma_old = gamma(xI[i]);
                    num_f     = 0.0;
                    denom_f   = 0.0;
                    
                    for (int k=0; k<Ntot; k++)
                        num_f += data.X(k, xI[i]) * (epsilon[k] + gamma_old * data.X(k, xI[i]));
                    denom_f = dNm1 + sigE_sigF;
                    gamma(i) = dist.norm_rng(num_f/denom_f, sigmaE/denom_f);
                    
                    for (int k = 0; k<Ntot ; k++) {
                        epsilon[k] = epsilon[k] + (gamma_old - gamma(xI[i])) * data.X(k, xI[i]);
                        //cout << "adding " << (gamma_old - gamma(xI[i])) * data.X(k, xI[i]) << endl;
                    }
                }
                //the next line should be uncommented if we want to use ridge for the other covariates.
                //sigmaF = inv_scaled_chisq_rng(0.001 + F, (gamma.squaredNorm() + 0.001)/(0.001+F));
                sigmaF = s02F;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        e_sqn = 0.0d;
        for (int i=0; i<Ntot; ++i) e_sqn += epsilon[i] * epsilon[i];

        sigmaE  = dist.inv_scaled_chisq_rng(v0E+dN, (e_sqn + v0E*s02E)/(v0E+dN));
        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
        if (rank%10==0)
            printf("RESULT: it %4d, rank %4d: sigmaG(%15.10f, %15.10f) = %15.10f, sigmaE = %15.10f, betasq = %15.10f, m0 = %d\n", iteration, rank, v0G+m0,(beta_squaredNorm * m0 + v0G*s02G) /(v0G+m0), sigmaG, sigmaE, beta_squaredNorm, int(m0));
        fflush(stdout);

        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+Ntot,((epsilon).squaredNorm()+v0E*s02E)/(v0E+Ntot));
        //printf("sigmaG = %20.15f, sigmaE = %20.15f, e_sqn = %20.15f\n", sigmaG, sigmaE, e_sqn);
        //printf("it %6d, rank %3d: epsilon[0] = %15.10f, y[0] = %15.10f, m0=%10.1f,  sigE=%15.10f,  sigG=%15.10f [%6d / %6d]\n", iteration, rank, epsilon[0], y[0], m0, sigmaE, sigmaG, markerI[0], markerI[M-1]);

        pi = dist.dirichilet_rng(v.array() + 1.0);


        // Write output files
        // ------------------
        if (iteration%opt.thin == 0) {

            left = snprintf(buff, LENBUF, "%5d, %4d, %15.10f, %15.10f, %15.10f, %15.10f, %7d, %2d", iteration, rank, mu, sigmaG, sigmaE, sigmaG/(sigmaE+sigmaG), int(m0), pi.size());
            //if (rank == 0) { cout << "left on buff: " << left << endl; }

            for (int ii=0; ii<pi.size(); ++ii) {
                left = snprintf(&buff[strlen(buff)], LENBUF-strlen(buff), ", %15.10f", pi(ii));
                //if (rank == 0) { cout << "left on buff " << ii << ": " << left << endl; }
            }
            left = snprintf(&buff[strlen(buff)], LENBUF-strlen(buff), "\n");

            offset = (size_t(n_thinned_saved) * size_t(nranks) + size_t(rank)) * strlen(buff);
            check_mpi(MPI_File_write_at_all(outfh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);

            // Write iteration number
            if (rank == 0) {
                betoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(betfh, betoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                check_mpi(MPI_File_write_at(acufh, betoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                cpnoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpnfh, cpnoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            
            betoff = sizeof(uint) + sizeof(uint) 
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double))
                + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(betfh, betoff, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at_all(acufh, betoff, Acum.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            cpnoff = sizeof(uint) + sizeof(uint)
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int))
                + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh, cpnoff, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            n_thinned_saved += 1;

            //EO: to remove once MPI version fully validated; use the check_marker utility to retrieve
            //    the corresponding values from .bet file
            //    Print a sub-set of non-zero betas, one per rank for validation of the .bet file
            for (int i=0; i<M; ++i) {
                if (Beta(i) != 0.0) {
                    //printf("%4d/%4d global Beta[%8d] = %15.10f, components[%8d] = %2d\n", iteration, rank, MrankS[rank]+i, Beta(i), MrankS[rank]+i, components(i));
                    break;
                }
            }
        }

        // Dump the epsilon vector
        // Note: single line overwritten at each saving iteration
        // Format: uint, uint, double{0 -> N-1}
        //         it,   N,    epsilon[i] i~[0,N-1]
        // ------------------------------------------------------
        if (iteration%opt.save == 0) {
            if (rank == 0) {
                epsoff  = size_t(0);
                check_mpi(MPI_File_write_at(epsfh, epsoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                epsoff += sizeof(uint);
                check_mpi(MPI_File_write_at(epsfh, epsoff, &Ntot,      1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            epsoff = sizeof(uint) + sizeof(uint) + size_t(IrankS[rank])*sizeof(double);
            //printf("%4d/%4d to write epsilon for %5d indiv from %5d (%lu)\n", iteration, rank, IrankL[rank], IrankS[rank], epsoff);
            check_mpi(MPI_File_write_at_all(epsfh, epsoff, &epsilon[IrankS[rank]], IrankL[rank], MPI_DOUBLE, &status), __LINE__, __FILE__);
            //EO: to remove once MPI version fully validated; use the check_epsilon utility to retrieve
            //    the corresponding values from the .eps file
            //    Print only first and last value handled by each task
            //printf("%4d/%4d epsilon[%5d] = %15.10f, epsilon[%5d] = %15.10f\n", iteration, rank, IrankS[rank], epsilon[IrankS[rank]], IrankS[rank]+IrankL[rank]-1, epsilon[IrankS[rank]+IrankL[rank]-1]);
        }

        double end_it = MPI_Wtime();
        if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        MPI_Barrier(MPI_COMM_WORLD);
    }


    // Close output files
    check_mpi(MPI_File_close(&outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&acufh), __LINE__, __FILE__);


    // Release memory
    free(y);
    free(epsilon);
    free(tmpEps);
    free(previt_eps);
    free(deltaEps);
    free(dEpsSum);
    free(deltaSum);
    free(mave);
    free(mstd);
    free(N1S);
    free(N1L);
    free(I1);
    free(N2S); 
    free(N2L);
    free(I2);
    free(NMS); 
    free(NML);
    free(IM);
    

    // Finalize the MPI environment
    //MPI_Finalize();

    const auto et3 = std::chrono::high_resolution_clock::now();
    const auto dt3 = et3 - st3;
    const auto du3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();
    if (rank == 0)
        printf("INFO   : rank %4d, time to process the data: %.3f sec.\n", rank, du3 / double(1000.0));

    return 0;
}

#endif


//
//  ORIGINAL (SEQUENTIAL) VERSION
//

void BayesRRm::init(int K, unsigned int markerCount, unsigned int individualCount, uint missingPhenCount)
{
    // Component variables
    priorPi = VectorXd(K);      // prior probabilities for each component
    pi      = VectorXd(K);      // mixture probabilities
    cVa     = VectorXd(K);      // component-specific variance
    cVaI    = VectorXd(K);      // inverse of the component variances
    logL    = VectorXd(K);      // log likelihood of component
    muk     = VectorXd(K);      // mean of k-th component marker effect size
    denom   = VectorXd(K - 1);  // temporal variable for computing the inflation of the effect variance for a given non-zero component
    m0      = 0;                // total number of markers in model
    v       = VectorXd(K);      // variable storing the component assignment

    // Mean and residual variables
    mu     = 0.0;   // mean or intercept
    sigmaG = 0.0;   // genetic variance
    sigmaE = 0.0;   // residuals variance

    // Linear model variables
    beta    = VectorXd(markerCount);        // effect sizes
    y_tilde = VectorXd(individualCount - missingPhenCount);    // variable containing the adjusted residuals to exclude the effects of a given marker
    epsilon = VectorXd(individualCount - missingPhenCount);    // variable containing the residuals

    //phenotype vector
    y = VectorXd(individualCount - missingPhenCount);

    //SNP column vector
    Cx = VectorXd(individualCount - missingPhenCount);

    // Init the working variables
    const int km1 = K - 1;
    cVa[0] = 0.0;
    cVa.segment(1, km1) = cva;

    //vector with component class for each marker
    components=VectorXd(markerCount);
    components.setZero();

    //set priors for pi parameters
    priorPi[0] = 0.5;
    priorPi.segment(1, km1) = priorPi[0] * cVa.segment(1, km1).array() / cVa.segment(1, km1).sum();

    y_tilde.setZero();

    cVaI[0] = 0;
    cVaI.segment(1, km1) = cVa.segment(1, km1).cwiseInverse();
    beta.setZero();

    //sample from beta distribution
    sigmaG = dist.beta_rng(1.0, 1.0);
    //printf("First sigmaG = %15.10f\n", sigmaG);

    pi = priorPi;

    //scale phenotype vector stored in data.y
    //printf("OFF y mean = %20.15f on %d elements\n", data.y.mean(), data.y.size());
    //y = (data.y.cast<double>().array() - data.y.cast<double>().mean());
    y = (data.y.array() - data.y.mean());
    y /= sqrt(y.squaredNorm() / (double(individualCount - missingPhenCount - 1)));
    //printf(" >>>> ysqn = %15.10f\n", y.squaredNorm());

    //initialize epsilon vector as the phenotype vector
    epsilon    = (y).array() - mu;
    sigmaE     = epsilon.squaredNorm() / (individualCount - missingPhenCount) * 0.5;
    //printf("OFF sigmaE = %20.15f\n", sigmaE);
    betasqn    = 0.0;
    epsilonsum = 0.0;
    ytildesum  = 0.0;
    //gamma      = VectorXd(data.fixedEffectsCount);
    gamma      = VectorXd(data.numFixedEffects);
    gamma.setZero();
    X = data.X; //fixed effects matrix
}


int BayesRRm::runGibbs()
{
    unsigned int M(data.numSnps);
    if (opt.numberMarkers > 0 && opt.numberMarkers < M)
        M = opt.numberMarkers;
    unsigned int N(data.numInds);
    const unsigned int NA(data.numNAs);
    const int K(int(cva.size()) + 1);

    //initialize variables with init member function
    printf("INFO   : Calling init with K=%d, M=%d, N=%d, NA=%d\n", K, M, N, NA);
    init(K, M, N, NA);

    // Adjust N by the number of NAs
    // -----------------------------
    N -= NA;
    printf("INFO   : Adjusted N = %d\n", N);

    const double NM1 = double(N - 1);
    const int    km1 = K - 1;

    //specify how to write samples
    if (1==0) {
        SampleWriter writer;
        writer.setFileName(opt.mcmcOut + ".csv");
        writer.setMarkerCount(M);
        writer.setIndividualCount(N);
        writer.open();
    }

    // Sampler variables
    //VectorXd sample(2*M+4+N); // variable containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
    std::vector<unsigned int> markerI(M);
    std::iota(markerI.begin(), markerI.end(), 0);

    std::cout << "Running Gibbs sampling" << endl;
    const auto t1 = std::chrono::high_resolution_clock::now();

    unsigned shuf_mark = opt.shuffleMarkers;

    //printf("sigmaE = %15.10f\n", sigmaE);

    for (unsigned int iteration = 0; iteration < max_iterations; iteration++) {
        // Output progress
        const auto iterStart = std::chrono::high_resolution_clock::now();
        //if (iteration > 0 && iteration % unsigned(std::ceil(max_iterations / 10)) == 0)
        //std::cout << "iteration: " << iteration << std::endl;
        
        //printf("mu = %15.10f   eps[0] = %15.10f\n", mu, epsilon[0]);
        epsilon = epsilon.array() + mu;//  we substract previous value
        mu = dist.norm_rng(epsilon.sum() / (double)N, sigmaE / (double)N); //update mu
        printf("mu = %15.10f\n", mu);
        epsilon = epsilon.array() - mu;// we substract again now epsilon =Y-mu-X*beta
 
        //cout << shuf_mark << endl;
        //EO: shuffle or not the markers (only tests)
        if (shuf_mark) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
            //std::random_shuffle(markerI.begin(), markerI.end());
        }

        m0 = 0;
        v.setZero();

        // This for should not be parallelized, resulting chain would not be ergodic, still, some times it may converge to the correct solution
        for (unsigned int j = 0; j < M; j++) {

            double acum = 0.0;
            const auto marker = markerI[j];
            double beta_old=beta(marker);
            //printf("beta = %15.10f\n", beta_old);

            //read data for column with member function getSnpData
            Cx = getSnpData(marker);
            
            /*
            double sqn = 0.0;
            for (int i=0; i<N; i++) {
                sqn += Cx[i] * Cx[i];
            }
            printf("beta = %20.15f, mean = %20.15f and sqne = %20.15f sqn = %20.15f\n", beta_old, Cx.mean(), Cx.squaredNorm(), sqn);
            */

            // residual update only if marker was previously included in model
            //EO
            if (components(marker)!=0){
                //printf("%d %d [0]residual update triggered!\n", iteration, marker); 
                y_tilde = epsilon + beta_old*Cx;
            } else {
                y_tilde = epsilon;
            }
            /*
            if (beta(marker) != 0.0) {
                y_tilde = epsilon + beta_old*Cx;
            } else {
                y_tilde = epsilon;
            }
            */


            // muk for the zeroth component=0
            muk[0] = 0.0;
            
            //for (int i=0; i<20; ++i)
            //    printf("%20.15f  %20.15f\n", Cx[i], y_tilde[i]);

            // We compute the denominator in the variance expression to save computations
            const double sigmaEOverSigmaG = sigmaE / sigmaG;
            denom = NM1 + sigmaEOverSigmaG * cVaI.segment(1, km1).array();
            //printf("denom[%d] = %20.15f\n", 0, denom(0));
            //printf("denom[%d] = %20.15f\n", 1, denom(1));

            // We compute the dot product to save computations
            double c2 = 0.0;
            for (int i=0; i<N; i++) {
                c2 += Cx[i] * y_tilde[i];
            }
            const double num = Cx.dot(y_tilde);
            //printf("num = %20.15f REF vs man %20.15f\n", num, c2);

            // muk for the other components is computed according to equations
            muk.segment(1, km1) = num / denom.array();
            //cout << muk << endl;

            // Update the log likelihood for each variance component
            const double logLScale = sigmaG / sigmaE * NM1;
            logL = pi.array().log(); // First component probabilities remain unchanged
            logL.segment(1, km1) = logL.segment(1, km1).array()
                    		        - 0.5 * ((logLScale * cVa.segment(1, km1).array() + 1).array().log())
                    		        + 0.5 * (muk.segment(1, km1).array() * num) / sigmaE;

            double p(dist.unif_rng());

            if (((logL.segment(1, km1).array() - logL[0]).abs().array() > 700).any()) { // this quantity will be exponentiated and go to huge values and we set to infinity, thus the reciprocal is set to zero
                acum = 0;
            } else {
                acum = 1.0 / ((logL.array() - logL[0]).exp().sum());
            }

            for (int k=0; k<K; k++) {
                if (p <= acum || k == km1) { //if we p is less than acum or if we are already in the last mixt.
                    if (k==0) {
                        beta(marker) = 0.0;
                    } else {
                        beta(marker) = dist.norm_rng(muk[k], sigmaE/denom[k-1]);
                        //printf("@B@ beta update %4d/%4d/%4d muk[%4d] = %15.10f with p=%15.10f <= acum=%15.10f\n", iteration, rank, marker, k, muk[k], p, acum);
                    }
                    v[k] += 1.0d;
                    components[marker] = k;
                    break;
                } else {
                    //if too big or too small
                    if (k+1 >= K) {
                        printf("FATAL  : iteration %d, marker = %d, p = %15.10f, acum = %15.10f logL overflow with %d => %d\n", iteration, marker, p, acum, k+1, K);
                        exit(1);
                    }
                    if (((logL.segment(k+1,K-(k+1)).array()-logL[k+1]).abs().array() > 700.0d ).any() ){
                        acum += 0.0d;// we compare next mixture to the others, if to big diff we skip
                    } else{
                        acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum()); //if not , sample
                    }
                }
            }

            betasqn+=beta(marker)*beta(marker)-beta_old*beta_old;

            // residual update only if updated marker is included in model
            // Now epsilon contains Y-mu - X*beta + X.col(marker) * beta(marker)_old - X.col(marker) * beta(marker)_new
            if (components(marker)!=0){
                //printf("%d %d [1]residual update triggered!\n", iteration, marker); 
                epsilon=y_tilde - beta(marker)*Cx;
            } else {
                epsilon=y_tilde;
            }
            /*
            if (beta(marker) != 0.0) {
                epsilon=y_tilde - beta(marker)*Cx;
            } else {
                epsilon=y_tilde;
            }
            */
            //printf("%d %d epssqn = %15.10f\n", iteration, marker, epsilon.squaredNorm());
        }

        //set no. of markers included in the model
        //cout << "M " << M  << endl;
        //cout << "v[0] " << v[0] << endl;
        m0 = int(M) - int(v[0]);
        //cout << "m0 " << m0 << endl;

        //sample sigmaG from inverse gamma
        sigmaG = dist.inv_scaled_chisq_rng(v0G + double(m0), (betasqn * double(m0) + v0G * s02G) / (v0G + double(m0)));

        const double epsilonSqNorm=epsilon.squaredNorm();

        //sample residual variance sigmaE from inverse gamma
        
        sigmaE = dist.inv_scaled_chisq_rng(v0E + double(N), (epsilonSqNorm + v0E * s02E) / (v0E + double(N)));
        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, epsilonSqNorm, v0E, s02E, N, sigmaE);

        //sample hyperparameter pi from dirichlet
        pi = dist.dirichilet_rng(v.array() + 1.0);
        //cout << pi << endl;

        if (showDebug)
            printDebugInfo();

        printf("it %4d, rank %4d: sigmaG(%15.10f, %15.10f) = %15.10f, sigmaE = %15.10f, betasq=%15.10f, m0=%10.1f\n", iteration, 0000, v0G+double(m0),(betasqn * double(m0) + v0G*s02G) /(v0G+double(m0)), sigmaG, sigmaE, betasqn, double(m0));

        //write samples
        //if (iteration >= burn_in && iteration % opt.thin == 0) {
        //        sample << iteration, mu, beta, sigmaE, sigmaG, components, epsilon;
        //        writer.write(sample);
        //    }

        //output time taken for each iteration
        const auto endTime = std::chrono::high_resolution_clock::now();
        const auto dif = endTime - iterStart;
        const auto iterationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(dif).count();
        //std::cout << iterationDuration / double(1000.0) << "s" << std::endl;

        //end of iteration
    }

    //show info on total time spent
    const auto t2 = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "duration: " << duration << "s" << std::endl;

    return 0;
}

VectorXd BayesRRm::getSnpData(unsigned int marker) const
{
    if (!usePreprocessedData) {
        //read column from RAM loaded genotype matrix.
        return data.Z.col(marker);//.cast<double>();
    } else {
        //read column from preprocessed and memory mapped genotype matrix file.
        return data.mappedZ.col(marker).cast<double>();
    }
}

void BayesRRm::printDebugInfo() const
{
    //const unsigned int N(data.numInds);
    const unsigned int N = data.numInds - data.numNAs;

    cout << "inv scaled parameters " << v0G + m0 << "__" << (beta.squaredNorm() * m0 + v0G * s02G) / (v0G + m0);
    cout << "num components: " << opt.S.size();
    cout << "\nMixture components: " << cva[0] << " " << cva[1] << " " << cva[2] << "\n";
    cout << "sigmaG: " << sigmaG << "\n";
    cout << "y mean: " << y.mean() << "\n";
    cout << "y sd: " << sqrt(y.squaredNorm() / (double(N - 1))) << "\n";
    // cout << "x mean " << Cx.mean() << "\n";
    //   cout << "x sd " << sqrt(Cx.squaredNorm() / (double(N - 1))) << "\n";
}
