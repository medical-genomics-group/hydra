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
#include <boost/range/algorithm.hpp>
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <mm_malloc.h>
#include <omp.h>
#include "dotp_lut.h"
#include "dense.hpp"


BayesRRm::BayesRRm(Data &data, Options &opt, const long memPageSize)
    : data(data)
    , opt(opt)
    , bedFile(opt.bedFile + ".bed")
    , memPageSize(memPageSize)
    , seed(opt.seed)
    , max_iterations(opt.chainLength)
    , burn_in(opt.burnin)
    , showDebug(false)
{
    double* ptr = &opt.S[0];
    cva = (Eigen::Map<Eigen::VectorXd>(ptr, static_cast<long>(opt.S.size()))).cast<double>();
}


BayesRRm::~BayesRRm()
{
}


//EO: this to allow reduction on avx256 pd4 datatype with OpenMP
#pragma omp declare reduction \
    (addpd4:__m256d:omp_out+=omp_in) \
    initializer(omp_priv=_mm256_setzero_pd())


inline void sparse_set(double*       __restrict__ vec,
                       const double               val,
                       const uint*   __restrict__ IX, const size_t NXS, const size_t NXL) {
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i=NXS; i<NXS+NXL; ++i) {
        vec[ IX[i] ] = val;
        //if (i==NXS)
        //    cout << ">??> " << i << ", " << IX[i] << ", " << vec[ IX[i] ] << endl;
    }
}


//inline
void BayesRRm::sparse_add(double*       __restrict__ vec,
                       const double               val,
                       const uint*   __restrict__ IX, const size_t NXS, const size_t NXL) {

#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i=NXS; i<NXS+NXL; ++i) {
        vec[ IX[i] ] += val;
        //if (i == NXS)
        //    cout << ">>>> " << i << ", " << IX[i] << ", " << vec[ IX[i] ] << endl;
    }
}


void BayesRRm::sparse_scaadd(double*     __restrict__ vout,
                             const double  dMULT,
                             const uint* __restrict__ I1, const size_t N1S, const size_t N1L,
                             const uint* __restrict__ I2, const size_t N2S, const size_t N2L,
                             const uint* __restrict__ IM, const size_t NMS, const size_t NML,
                             const double  mu,
                             const double  sig_inv,
                             const int     N) {

    if (dMULT == 0.0) {

        set_array(vout, 0.0, N);

    } else {

        double aux = mu * sig_inv * dMULT;
        //printf("sparse_scaadd aux = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", aux, mu, sig_inv * dMULT);
        set_array(vout, -aux, N);

        //cout << "sparse set on M: " << NMS << ", " << NML << endl;
        sparse_set(vout, 0.0, IM, NMS, NML);

        //cout << "sparse set on 1: " << N1S << ", " << N1L << endl;
        aux = dMULT * (1.0 - mu) * sig_inv;
        //printf("1: aux = %15.10f\n", aux);
        sparse_set(vout, aux, I1, N1S, N1L);

        //cout << "sparse set on 2: " << N2S << ", " << N2L << endl;
        aux = dMULT * (2.0 - mu) * sig_inv;
        sparse_set(vout, aux, I2, N2S, N2L);
    }
}


inline double partial_sparse_dotprod(const double* __restrict__ vec,
                                     const uint*   __restrict__ IX,
                                     const size_t               NXS,
                                     const size_t               NXL,
                                     const double               fac) {

    //double t1 = -mysecond();
    //for (int ii=0; ii<1024; ii++) {
    //}
    //t1 += mysecond();
    //printf("kerold 1 BW = %g\n", double(N1L)*sizeof(double) / 1024. / 1024. / t1);

    double dp = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: dp)
#endif
    for (size_t i=NXS; i<NXS+NXL; i++) {
        dp += vec[ IX[i] ];
    }

    dp *= fac;

    //printf("partial %lu -> %lu with fac = %20.15f\n", NXS, NXS+NXL, fac);

    return dp;
}


double BayesRRm::sparse_dotprod(const double* __restrict__ vin1,
                                const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                                const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                                const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                                const double               mu,
                                const double               sig_inv,
                                const int                  N,
                                const int                  marker) {

    double dp  = 0.0;

    dp += partial_sparse_dotprod(vin1, I1, N1S, N1L, 1.0);

    dp += partial_sparse_dotprod(vin1, I2, N2S, N2L, 2.0);

    double syt = sum_array_elements(vin1, N);

    double dsyt = partial_sparse_dotprod(vin1, IM, NMS, NML, 1.0);

    syt -= dsyt;

    dp  -= (mu * syt);

    dp  *= sig_inv;

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


inline void center_and_scale(double* __restrict__ vec, const int N) {

    // Compute mean
    double mean = 0.0;
    for (int i=0; i<N; ++i)  mean += vec[i];
    mean /= N;

    // Center
    for (int i=0; i<N; ++i)  vec[i] -= mean;

    // Compute scale
    double sqn = 0.0;
    for (int i=0; i<N; ++i)  sqn += vec[i] * vec[i];
    sqn = sqrt(double(N-1) / sqn);

    // Scale
    for (int i=0; i<N; ++i)  vec[i] *= sqn;
}


// Define blocks of markers to be processed by each task
// By default processes all markers
// -----------------------------------------------------
void BayesRRm::mpi_define_blocks_of_markers(const int Mtot, int* MrankS, int* MrankL, const uint nblocks) {

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
    std::string sparseOut = mpi_get_sparse_output_filebase(rank);
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

    if (rank == 0) {
        MPI_File_delete(dim.c_str(), MPI_INFO_NULL);
        MPI_File_delete(si1.c_str(), MPI_INFO_NULL);
        MPI_File_delete(sl1.c_str(), MPI_INFO_NULL);
        MPI_File_delete(ss1.c_str(), MPI_INFO_NULL);
        MPI_File_delete(si2.c_str(), MPI_INFO_NULL);
        MPI_File_delete(sl2.c_str(), MPI_INFO_NULL);
        MPI_File_delete(ss2.c_str(), MPI_INFO_NULL);
        MPI_File_delete(sim.c_str(), MPI_INFO_NULL);
        MPI_File_delete(slm.c_str(), MPI_INFO_NULL);
        MPI_File_delete(ssm.c_str(), MPI_INFO_NULL);
    }

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
        char* rawdata = (char*) _mm_malloc(rawdata_n, 64);  check_malloc(rawdata, __LINE__, __FILE__);

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
        size_t bytes = 0;
        data.mpi_file_read_at_all <char*> (rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata, bytes);

        // Get number of ones, twos, and missing
        data.sparse_data_get_sizes_from_raw(rawdata, MLi, snpLenByt, 0, N1, N2, NM);
        //printf("DEBUG  : off rank %d: N1 = %15lu, N2 = %15lu, NM = %15lu\n", rank, N1, N2, NM);

        rN1 += N1;
        rN2 += N2;
        rNM += NM;

        _mm_free(rawdata);

        MPI_Barrier(MPI_COMM_WORLD);
    }


    // Gather offsets
    // --------------
    size_t *AllN1 = (size_t*)_mm_malloc(nranks * sizeof(size_t), 64);  check_malloc(AllN1, __LINE__, __FILE__);
    size_t *AllN2 = (size_t*)_mm_malloc(nranks * sizeof(size_t), 64);  check_malloc(AllN2, __LINE__, __FILE__);
    size_t *AllNM = (size_t*)_mm_malloc(nranks * sizeof(size_t), 64);  check_malloc(AllNM, __LINE__, __FILE__);

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
        char* rawdata = (char*) _mm_malloc(rawdata_n, 64);  check_malloc(rawdata, __LINE__, __FILE__);

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

        size_t bytes = 0;
        data.mpi_file_read_at_all<char*>(rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata, bytes);

        // Alloc memory for sparse representation
        size_t *N1S, *N1L, *N2S, *N2L, *NMS, *NML;
        N1S = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
        N1L = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
        N2S = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
        N2L = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
        NMS = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
        NML = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);

        size_t N1 = 0, N2 = 0, NM = 0;
        data.sparse_data_get_sizes_from_raw(rawdata, uint(MLi), snpLenByt, 0, N1, N2, NM);
        //printf("DEBUG  : N1 = %15lu, N2 = %15lu, NM = %15lu\n", N1, N2, NM);

        // Alloc and build sparse structure
        uint *I1, *I2, *IM;
        I1 = (uint*) _mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
        I2 = (uint*) _mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
        IM = (uint*) _mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);

        // To check that each element is properly set
        //for (int i=0; i<N1; i++) I1[i] = UINT_MAX;
        //for (int i=0; i<N2; i++) I2[i] = UINT_MAX;
        //for (int i=0; i<NM; i++) IM[i] = UINT_MAX;

        data.sparse_data_fill_indices(rawdata, MLi, snpLenByt, 0, N1S, N1L, I1,  N2S, N2L, I2,  NMS, NML, IM);

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
        if (rank == 0) printf("INFO   : N1max = %lu, N2max = %lu, NMmax = %lu, MLimax = %d\n", N1max, N2max, NMmax, MLimax);

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
        _mm_free(rawdata);
        _mm_free(N1S); _mm_free(N1L); _mm_free(I1);
        _mm_free(N2S); _mm_free(N2L); _mm_free(I2);
        _mm_free(NMS); _mm_free(NML); _mm_free(IM);

        tN1 += N1;
        tN2 += N2;
        tNM += NM;
    }

    _mm_free(AllN1);
    _mm_free(AllN2);
    _mm_free(AllNM);

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



void BayesRRm::mpi_assign_blocks_to_tasks(const uint numBlocks, const vector<int> blocksStarts, const vector<int> blocksEnds, const uint Mtot, const int nranks, const int rank, int* MrankS, int* MrankL, int& lmin, int& lmax) {

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
        //if (rank == 0)
        //    printf("INFO   : no marker block definition file used. Will go for even distribution over tasks.\n");
        mpi_define_blocks_of_markers(Mtot, MrankS, MrankL, nranks);
    }

    lmax = 0, lmin = 1E9;
    for (int i=0; i<nranks; ++i) {
        if (MrankL[i]>lmax) lmax = MrankL[i];
        if (MrankL[i]<lmin) lmin = MrankL[i];
    }
}


void BayesRRm::init_from_scratch() {

    iteration_start = 0;
}


//EO: .csv, .bet, .acu, .cpn, .mus         : written every --opt.thin
//    .eps, .mrk, .gam, .xiv, .xbet, .xcpn : last itertion only written every --opt.save
// 
//    one can restart from .bet and .cpn files or .xbet and .xcpn files,
//    by using --restart --ignore-xfiles rather than --restart
//
void BayesRRm::init_from_restart(const int K, const uint M, const uint  Mtot, const uint Ntot,
                                 const int* MrankS, const int* MrankL, const bool use_xfiles_in_restart) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        printf("RESTART: from files: %s.* files\n", opt.mcmcOut.c_str());

    //EO: the .csv files is read to decide where to restart from
    data.read_mcmc_output_csv_file(opt.mcmcOut,
                                   opt.thin, opt.save,
                                   K, sigmaG, sigmaE, estPi,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   first_saved_iteration);

    if (rank == 0) {
        printf("RESTART: Reading .cvs file %s\n", (opt.mcmcOut + ".csv").c_str());
        printf("RESTART: --thin %d  -- save %d\n", opt.thin, opt.save);
        printf("RESTART: iteration_to_restart_from = %d\n", iteration_to_restart_from);
        printf("RESTART: first_thinned_iteration   = %d\n", first_thinned_iteration);
        printf("RESTART: first_saved_iteration     = %d\n", first_saved_iteration);
        fflush(stdout);
    }
    
    //EO: Kill the processing if one's try to restart from iteration 0 as 
    //    we do not save this iteration, as this makes no sense to do so
    if (iteration_to_restart_from == 0) {
        printf("\nFATAL  : There is no point in restarting a chain from iteration 0 (not saved anyway)\n");
        printf("         => restart your analysis from scratch\n\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    MPI_Barrier(MPI_COMM_WORLD);

    // .bet  saved at --opt.thin
    // .xbet saved at --opt.save, single line with last iteration
    data.read_mcmc_output_bet_file(opt.mcmcOut, Mtot,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   opt.thin,
                                   MrankS, MrankL,
                                   use_xfiles_in_restart,
                                   Beta);

    data.read_mcmc_output_cpn_file(opt.mcmcOut, Mtot,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   opt.thin,
                                   MrankS, MrankL,
                                   use_xfiles_in_restart,
                                   components);

    data.read_mcmc_output_mus_file(opt.mcmcOut,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   opt.thin,
                                   mu_restart);

    data.read_mcmc_output_eps_file(opt.mcmcOut, Ntot,
                                   iteration_to_restart_from,
                                   epsilon_restart);

    data.read_mcmc_output_idx_file(opt.mcmcOut, "mrk", M,
                                   iteration_to_restart_from, opt.bayesType,
                                   markerI_restart);

    if (opt.covariates) {

        data.read_mcmc_output_gam_file(opt.mcmcOut, data.X.cols(),
                                       iteration_to_restart_from,
                                       gamma_restart);

        data.read_mcmc_output_idx_file(opt.mcmcOut, "xiv", (uint)data.X.cols(),
                                       iteration_to_restart_from, opt.bayesType,
                                       xI_restart);
    }

    // Adjust starting iteration number.
    iteration_start = iteration_to_restart_from + 1;


    MPI_Barrier(MPI_COMM_WORLD);
}


//EO: MPI GIBBS
//-------------
int BayesRRm::runMpiGibbs() {

    typedef Matrix<bool, Dynamic, 1> VectorXb;

    char   buff[LENBUF]; 
    int    nranks, rank, name_len, result;
    double dalloc = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File   outfh, betfh, epsfh, gamfh, cpnfh, acufh, mrkfh, xivfh, musfh;
    MPI_File   xbetfh, xcpnfh;
    MPI_Status status;
    MPI_Info   info;

    // Display banner if wished
    if (rank < 0) {
        opt.printBanner();
        opt.printProcessingOptions();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        if (rank == 0) {
            int nt = omp_get_num_threads();
            if (omp_get_thread_num() == 0) 
                printf("INFO   : OMP parallel regions will use %d thread(s)\n", nt);
        }
    }


    // Set Ntot and Mtot
    //
    uint Ntot       = set_Ntot(rank);
    const uint Mtot = set_Mtot(rank);
    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);


    // Handle marker groups: by default we assume a single group with all markers belonging
    // to it. In this case -S option is used for the mixture. Otherwise, we expect a group file
    // covering all markers with a group mixture file covering all groups.
    // Warning: passed mixture shall not contain the 0.0 value, it is added by default.

    // First, load the mixture components (adding 0.0 as first element/column)
    if (opt.groupIndexFile != "" && opt.groupMixtureFile != "") {
        data.readGroupFile(opt.groupIndexFile);
        data.readmSFile(opt.groupMixtureFile);
    } else {
        // Define one group for all markers (group 0)
        data.groups.resize(Mtot);
        data.groups.setZero();
        assert(data.numGroups == 1);
        data.mS.resize(data.numGroups, cva.size() + 1);
        data.mS(0, 0) = 0.0;
        for (int i=0; i<cva.size(); i++) {
            if (cva(i) <= 0.0)
                throw("FATAL  : mixture value can only be strictly positive");
            data.mS(0, i + 1) = cva(i);
        }
    }
    assert(data.numGroups == data.mS.rows());

    // Display for control
    if (rank == 0)  data.printGroupMixtureComponents();

    
    // data.mS contains all mixture components, 0.0 included
    const unsigned int  K   = int(data.mS.cols());
    const unsigned int  km1 = K - 1;

    const int numGroups = data.numGroups;

    // Length of a marker in [byte] in BED representation (1 ind is 2 bits, so 1 byte is 4 inds)
    size_t snpLenByt  = (Ntot %  4) ? Ntot /  4 + 1 : Ntot /  4;

    // Length of a marker in [uint] in BED representation (1 ind is 2 bits, so 1 uint is 16 inds)
    size_t snpLenUint = (Ntot % 16) ? Ntot / 16 + 1 : Ntot / 16;

    if (rank == 0)  printf("INFO   : snpLenByt = %zu bytes; snpLenUint = %zu\n", snpLenByt, snpLenUint);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    uint M = MrankL[rank];
    if (rank % 10 == 0) {
        printf("INFO   : rank %3d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);
    }


    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: at this stage Ntot is not yet adjusted for missing phenotypes,
    //       hence the correction in the call
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    mpi_define_blocks_of_markers(Ntot - data.numNAs, IrankS, IrankL, nranks);

    VectorXi groups     = data.groups;
    cVa.resize(numGroups, K);                   // component-specific variance
    cVaI.resize(numGroups, K);                  // inverse of the component variances
    std::vector<int>    markerI;

    VectorXd            muk(K);                 // mean of k-th component marker effect size
    VectorXd            denom(K-1);             // temporal variable for computing the inflation of the effect variance for a given non-zero component
    VectorXi            m0(numGroups);          // non-zero elements per group
    cass.resize(numGroups,K);                   // variable storing the component assignment per group; EO RENAMING was v
    MatrixXi            sum_cass(numGroups,K);  // To store the sum of v elements over all ranks per group;         was sum_v
    VectorXd            Acum(M);
    VectorXb            adaV(M);                // Daniel adaptative scan Vector, ones will be sampled, 0 will be set to 0
    std::vector<int>    mark2sync;
    std::vector<double> dbet2sync;

    dalloc +=     M * sizeof(int)    / 1E9; // for components
    dalloc += 2 * M * sizeof(double) / 1E9; // for Beta and Acum


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


    // Invariant initializations (from scratch / from restart)
    // -------------------------------------------------------
    string lstfp  = opt.mcmcOut + ".lst";
    string outfp  = opt.mcmcOut + ".csv";
    string betfp  = opt.mcmcOut + ".bet";
    string xbetfp = opt.mcmcOut + ".xbet";
    string cpnfp  = opt.mcmcOut + ".cpn";
    string xcpnfp = opt.mcmcOut + ".xcpn";
    string acufp  = opt.mcmcOut + ".acu";
    string rngfp  = opt.mcmcOut + ".rng." + std::to_string(rank);
    string mrkfp  = opt.mcmcOut + ".mrk." + std::to_string(rank);
    string xivfp  = opt.mcmcOut + ".xiv." + std::to_string(rank);
    string epsfp  = opt.mcmcOut + ".eps." + std::to_string(rank);
    string gamfp  = opt.mcmcOut + ".gam." + std::to_string(rank);
    string musfp  = opt.mcmcOut + ".mus." + std::to_string(rank);


    priorPi.resize(numGroups,K);
    priorPi.setZero();

    estPi.resize(numGroups,K);
    estPi.setZero();

    gamma.setZero();

    //fixed effects matrix
    X = data.X;

    priorPi.col(0).array() = 0.5;
    cVa.col(0).array()     = 0.0;
    cVaI.col(0).array()    = 0.0;

    muk[0]     = 0.0;
    mu         = 0.0;

    for (int i=0; i<numGroups; i++) {
        cVa.row(i).segment(1,km1)     = data.mS.row(i).segment(1,km1);
        cVaI.row(i).segment(1,km1)    = cVa.row(i).segment(1,km1).cwiseInverse();
        priorPi.row(i).segment(1,km1) = priorPi(i,0) * cVa.row(i).segment(1,km1).array() / cVa.row(i).segment(1,km1).sum();
    }

    estPi = priorPi;
    //cout << "estPi after init " << estPi << endl;

    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    xI_restart.resize(data.X.cols());

    Beta.resize(M);
    Beta.setZero();


    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FH
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FH
    //----------- FHDT parameters
    double v0L = opt.v0L;
    double v0t = opt.v0t;
    double v0c = opt.v0c;
    double s02c = opt.s02c;
    double hypTau;
    VectorXd lambda_var(Beta.size());
    VectorXd nu_var(Beta.size());
    double tau;
    double tau0 = opt.tau0;
    VectorXd c_slab(numGroups);
    double scaledBSQN;
    
    tau=0;
    c_slab.setZero();;
    lambda_var.setZero();
    nu_var.setZero();
    
    std::cout << "INFO Sampling initial hypTau" <<"\n";
    std::cout << "tau0: "<<tau0 << "\n";
    hypTau = dist.inv_gamma_rate_rng(0.5,1.0/(tau0*tau0));
    std::cout << "hypTau: "<< hypTau <<"\n";
    std::cout << "INFO Sampling initial tau" <<"\n";
    tau = dist.inv_gamma_rate_rng(0.5*v0t,v0t/hypTau);
    std::cout << "tau: " << tau <<"\n";
    std::cout << "INFO Sampling initial slab" <<"\n";
    for(int jj=0;jj<numGroups;++jj)
      c_slab[jj] = dist.inv_scaled_chisq_rng( v0c, s02c);
    std::cout << "c: " << c_slab <<"\n";
    std::cout << "v0L" << v0L << "\n";
    //if using newtonian uncomment this
    //SamplerLocalVar  lambdaSampler(dist,v0L,c_slab/(double)Mtot);
    
    std::cout << "INFO Sampling initial lambda with values: "<< c_slab.sum()/(double)Mtot<<"\n";
    lambda_var.array() = c_slab.sum()/(double)Mtot;
    std::cout << "<------------ FH initialisation finished -------------------->\n";
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FH
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FH
    

    components.resize(M);
    components.setZero();

    epsilon_restart.resize(Ntot - data.numNAs);
    epsilon_restart.setZero();

    gamma_restart.resize(data.X.cols());
    gamma_restart.setZero();

    markerI_restart.resize(M);
    std::fill(markerI_restart.begin(), markerI_restart.end(), 0);

    sigmaG.resize(numGroups);

    sigmaE = 0.0;

    VectorXd dirc(K);
    dirc.array() = 1.0;


    // Build global repartition of markers over the groups
    VectorXi MtotGrp(numGroups);
    MtotGrp.setZero();
    for (int i=0; i<Mtot; i++) {
        MtotGrp[groups[i]] += 1;
    }

    // In case of a restart, we first read the latest dumps
    // ----------------------------------------------------
    if (opt.restart) {

        init_from_restart(K, M, Mtot, Ntot - data.numNAs, MrankS, MrankL, opt.useXfilesInRestart);

        if (rank == 0)
            data.print_restart_banner(opt.mcmcOut.c_str(),  iteration_to_restart_from, iteration_start);

        dist.read_rng_state_from_file(rngfp);

        // Rename output files so that we do not erase from failed job!
        //EO: add a function, to update both Nam and Dir!
        opt.mcmcOutNam += "_rs";
        opt.mcmcOut = opt.mcmcOutDir + "/" + opt.mcmcOutNam;
        lstfp  = opt.mcmcOut + ".lst";
        outfp  = opt.mcmcOut + ".csv";
        betfp  = opt.mcmcOut + ".bet";
        xbetfp = opt.mcmcOut + ".xbet"; // Last saved iteration of bet; .bet has full history
        cpnfp  = opt.mcmcOut + ".cpn";  
        xcpnfp = opt.mcmcOut + ".xcpn"; // Idem
        acufp  = opt.mcmcOut + ".acu";
        rngfp  = opt.mcmcOut + ".rng." + std::to_string(rank);
        mrkfp  = opt.mcmcOut + ".mrk." + std::to_string(rank);
        xivfp  = opt.mcmcOut + ".xiv." + std::to_string(rank);
        epsfp  = opt.mcmcOut + ".eps." + std::to_string(rank);
        gamfp  = opt.mcmcOut + ".gam." + std::to_string(rank);
        musfp  = opt.mcmcOut + ".mus." + std::to_string(rank);

    } else {

        init_from_scratch();        

        dist.reset_rng((uint)(opt.seed + rank*1000));
        
        //EO: sample sigmaG and broadcast from rank 0 to all the others
        for(int i=0; i<numGroups; i++) {
            //sigmaG[i] = dist.unif_rng();
            sigmaG[i] = dist.beta_rng(1.0, 1.0);
        }
        check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
    }

    // Set sigmaG of empty groups to zero
    for (int i=0; i<numGroups; i++)
        if (MtotGrp[i] == 0)  sigmaG[i] = 0.0;

    //cout << "sigmaG = " << sigmaG << endl;


    // Build a list of the files to tar
    // --------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    ofstream listFile;
    listFile.open(lstfp);
    listFile << outfp  << "\n";
    listFile << xbetfp << "\n"; // Only tar the last saved iteration, no need for full history
    listFile << xcpnfp << "\n"; // Idem
    listFile << acufp  << "\n";
    for (int i=0; i<nranks; i++) {
        listFile << opt.mcmcOut + ".rng." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".mrk." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".xiv." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".eps." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".gam." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".mus." + std::to_string(i) << "\n";
    }
    listFile.close();
    MPI_Barrier(MPI_COMM_WORLD);


    // Delete old files (fp appended with "_rs" in case of restart, so that
    // original files are kept untouched) and create new ones
    // --------------------------------------------------------------------
    if (rank == 0) {
        MPI_File_delete(outfp.c_str(),  MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(),  MPI_INFO_NULL);
        MPI_File_delete(xbetfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(),  MPI_INFO_NULL);
        MPI_File_delete(xcpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(acufp.c_str(),  MPI_INFO_NULL);
    }
    MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(mrkfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(xivfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(gamfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(musfp.c_str(), MPI_INFO_NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(),  MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(),  MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xbetfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xbetfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(),  MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xcpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xcpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, acufp.c_str(),  MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &acufh),  __LINE__, __FILE__);
    
    check_mpi(MPI_File_open(MPI_COMM_SELF,  epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  mrkfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &mrkfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  xivfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xivfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  gamfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &gamfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  musfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &musfh), __LINE__, __FILE__);


    // First element of the .bet, .xbet, .cpn, .xcpn, and .acu files is the
    // total number of processed markers
    // -----------------------------------------------------
    MPI_Offset offset = 0;
    if (rank == 0) {
        check_mpi(MPI_File_write_at(betfh,  offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xbetfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh,  offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xcpnfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(acufh,  offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    double tl = -mysecond();


    // Read the data (from sparse representation by default)
    // -----------------------------------------------------
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    N1S = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);
    dalloc += 6.0 * double(M) * sizeof(size_t) / 1E9;


    // Boolean mask for using BED representation or not (SPARSE otherwise)
    // For markers with USEBED == true then the BED representation is 
    // converted on the fly to SPARSE the time for the corresponding marker
    // to be processed
    // --------------------------------------------------------------------
    bool *USEBED;
    USEBED = (bool*)_mm_malloc(M * sizeof(bool), 64);  check_malloc(USEBED, __LINE__, __FILE__);
    for (int i=0; i<M; i++) USEBED[i] = false;
    int nusebed = 0;


    // Sparse indices storage
    uint *I1, *I2, *IM;

    size_t taskBytes = 0;

    string sparseOut = mpi_get_sparse_output_filebase(rank);

    if (opt.readFromBedFile && !opt.readFromSparseFiles) {

        data.load_data_from_bed_file(opt.bedFile, Ntot, M, rank, MrankS[rank],
                                     N1S, N1L, I1,
                                     N2S, N2L, I2,
                                     NMS, NML, IM,
                                     taskBytes);
        //for (int i=0; i<M; i++)
        //    printf("OUT BED bef rank %02d, marker %7d: 1,2,M  S = %6lu, %6lu, %6lu, L = %6lu, %6lu, %6lu\n", rank, i, N1S[i], N2S[i], NMS[i], N1L[i], N2L[i], NML[i]);

    } else if (opt.readFromSparseFiles && !opt.readFromBedFile) {
        
        data.load_data_from_sparse_files(rank, nranks, M, MrankS, MrankL, sparseOut,
                                         N1S, N1L, I1,
                                         N2S, N2L, I2,
                                         NMS, NML, IM,
                                         taskBytes);
        //for (int i=0; i<M; i++)
        //    printf("OUT SPARSE bef rank %02d, marker %7d: 1,2,M  S = %6lu, %6lu, %6lu, L = %6lu, %6lu, %6lu\n", rank, i, N1S[i], N2S[i], NMS[i], N1L[i], N2L[i], NML[i]);

    } else if (opt.mixedRepresentation) {

        // Test on binary reading of BED file
        //EO: be aware that we use an inverted BED format!
        /*
        bitset<8> bo = 0b10011100;
        unsigned allele1, allele2;
        unsigned raw1, raw2;
        int k = 0;
        while (k < 7) {
            int miss = 0;
            raw1    = bo[k];
            allele1 = (!bo[k++]);
            raw2    = bo[k];
            allele2 = (!bo[k++]);
            if (allele1 == 0 && allele2 == 1) miss = 1;
            printf("raw bed = %d %d; allele1, allele2 = %d %d  miss? %d\n", raw1, raw2, allele1, allele2, miss);
        }
        */

        data.load_data_from_mixed_representations(opt.bedFile, sparseOut,
                                                  rank, nranks, Ntot, M, MrankS, MrankL,
                                                  N1S, N1L, I1,
                                                  N2S, N2L, I2,
                                                  NMS, NML, IM,
                                                  opt.threshold_fnz, USEBED,
                                                  taskBytes);

        // Store once for all the number of elements for markers stored as BED 
        for (int i=0; i<M; i++) {
            if (USEBED[i]) {
                nusebed += 1;
                size_t X1 = 0, X2 = 0, XM = 0;
                data.sparse_data_get_sizes_from_raw(reinterpret_cast<char*>(&I1[N1S[i]]), 1, snpLenByt, data.numNAs, X1, X2, XM);
                //printf("data.sparse_data_get_sizes_from_raw => (%2d, %3d): X1 = %9lu, X2 = %9lu, XM = %9lu < << <<<\n", rank, i, X1, X2, XM);

                //EO: N{1,2,M}L structures must not be changed anymore!
                //------------------------------------------------------
                N1L[i] = X1;  N2L[i] = X2;  NML[i] = XM;
            }
        }

    } else {
        printf("FATAL  : either BED, SPARSE or MIXED");
        exit(1);
    }


    tl += mysecond();

    //EO: BW cannot be evaluated properly in case of mixed representation as NA corrections
    //    are applied in the loading routine
    //
    if (opt.mixedRepresentation) {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes and apply NA corrections to BED markers\n", rank, tl, taskBytes);        
    } else {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes  =>  BW = %7.3f GB/s\n", rank, tl, taskBytes, (double)taskBytes * 1E-9 / tl);
    }
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);


    // For mixed representation get global number of markers dealt with as BED
    //
    if (opt.mixedRepresentation) {
        int nusebed_tot = 0;
        MPI_Reduce(&nusebed, &nusebed_tot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("INFO   : there is %d markers using BED representation out of %d (%6.2f%%)\n", nusebed_tot, Mtot, (double)nusebed_tot/(double)Mtot * 100.0);
    }


    // Correct each marker for individuals with missing phenotype
    // ----------------------------------------------------------
    if (rank == 0) 
        printf("INFO   : BEFORE NA correction: Ntot = %d; snpLenByt = %zu [byte]; snpLenUint = %zu [uint]\n", Ntot, snpLenByt, snpLenUint);

    if (data.numNAs > 0) {

        double tna = MPI_Wtime();

        if (rank == 0)
            printf("INFO   : applying %d corrections to genotype data due to missing phenotype data (NAs)...\n", data.numNAs);

        data.sparse_data_correct_for_missing_phenotype(N1S, N1L, I1, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(N2S, N2L, I2, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(NMS, NML, IM, M, USEBED);

        MPI_Barrier(MPI_COMM_WORLD);

        tna = MPI_Wtime() - tna;

        if (rank == 0) {
            if (opt.mixedRepresentation) {
                printf("INFO   : ... finished applying NA corrections in %.2f seconds for SPARSE markers in mixed representation.\n", tna);
            } else {
                printf("INFO   : ... finished applying NA corrections in %.2f seconds.\n", tna);
            }
        }

        // Adjust N upon number of NAs
        Ntot -= data.numNAs;

        // Length of a marker in [byte] in BED representation (1 ind is 2 bits, so 1 byte is 4 inds)
        snpLenByt  = (Ntot %  4) ? Ntot /  4 + 1 : Ntot /  4;

        // Length of a marker in [uint] in BED representation (1 ind is 2 bits, so 1 uint is 16 inds)
        snpLenUint = (Ntot % 16) ? Ntot / 16 + 1 : Ntot / 16;
    }

    if (rank == 0) 
        printf("INFO   : AFTER  NA correction: Ntot = %d; snpLenByt = %zu [byte]; snpLenUint = %zu [uint]\n", Ntot, snpLenByt, snpLenUint);
    fflush(stdout);

    // Compute dalloc increment from NA adjusted structures
    //
    for (int i=0; i<M; i++) {
        if (USEBED[i]) { // Although N2L and NML do contain the info, I2 and IM were not allocated for BED markers! Done on the fly
            dalloc += (double) (snpLenUint * sizeof(uint)) * 1E-9;
        } else {
            dalloc += (double)(N1L[i] + N2L[i] + NML[i]) * (double) sizeof(uint) * 1E-9;
        }
    }


    // Compute statistics (from sparse info)
    //
    double  dN   = (double) Ntot;
    double  dNm1 = (double)(Ntot - 1);
    double* mave = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mave, __LINE__, __FILE__);
    double* mstd = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);
    
    dalloc += 3 * size_t(M) * sizeof(double) * 1E-9;

    double tmp0, tmp1, tmp2;
    for (int i=0; i<M; ++i) {
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (dN - double(NML[i]));
        tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)
        printf("INFO   : overall time to preprocess the data: %.3f seconds\n", (double) du2 / 1000.0);


    // Build list of markers
    // ---------------------
    for (int i=0; i<M; ++i) 
        markerI.push_back(i);

    
    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();

    double   *y, *epsilon, *tmpEps, *deltaEps, *dEpsSum, *deltaSum; //, *previt_eps;
    const size_t NDB = size_t(Ntot) * sizeof(double);
    y          = (double*)_mm_malloc(NDB, 64);  check_malloc(y,          __LINE__, __FILE__);
    epsilon    = (double*)_mm_malloc(NDB, 64);  check_malloc(epsilon,    __LINE__, __FILE__);
    tmpEps     = (double*)_mm_malloc(NDB, 64);  check_malloc(tmpEps,     __LINE__, __FILE__);
    //previt_eps = (double*)malloc(NDB);  check_malloc(previt_eps, __LINE__, __FILE__);
    deltaEps   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaEps,   __LINE__, __FILE__);
    dEpsSum    = (double*)_mm_malloc(NDB, 64);  check_malloc(dEpsSum,    __LINE__, __FILE__);
    deltaSum   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaSum,   __LINE__, __FILE__);
    dalloc += NDB * 6 / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);


    set_array(dEpsSum, 0.0, Ntot);

    // Set gamma and xI, following a restart or not
    if (opt.covariates) {

        if (rank == 0)
            printf("INFO   : using covariate file: %s\n", opt.covariatesFile.c_str());

    	gamma = VectorXd(data.X.cols());
    	gamma.setZero();

        if (opt.restart) {
            for (int i=0; i<data.X.cols(); i++) {
                gamma[i] = gamma_restart[i];
                xI[i]    = xI_restart[i];
            }
        }
    }


    // Copy, center and scale phenotype observations
    for (int i=0; i<Ntot; ++i) y[i] = data.y(i);
    center_and_scale(y, Ntot);


    // In case of restart we reset epsilon to last dumped state (sigmaE as well, see init_from_restart)
    if (opt.restart) {
        for (int i=0; i<Ntot; ++i)  epsilon[i] = epsilon_restart[i];
        markerI = markerI_restart;
        mu      = mu_restart;
    } else {
        for (int i=0; i<Ntot; ++i)  epsilon[i] = y[i];
        sigmaE = 0.0;
        for (int i=0; i<Ntot; ++i)  sigmaE += epsilon[i] * epsilon[i];
        //printf("<sigmaE = %15.13f\n", sigmaE);
        sigmaE = sigmaE / dN * 0.5;

        //EO: no need to broacast here, sigmaE is the same over all tasks as we start from phenotype data
        //EO: broadcast sigmaE from rank 0 to all the others
        //check_mpi(MPI_Bcast(&sigmaE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
    }
    //printf("sigmaE = %20.15f\n", sigmaE);
    //printf("mu = %20.15f\n", mu);


    //EO: for now, we use adav to monitor markers belonging to groups
    //    with sigmaG being null; so no need to dump additional information
    //    in case of restart
    adaV.setOnes();
    for (int i=0; i<M; i++) {
        if (sigmaG[groups[MrankS[rank] + i]] == 0.0) {
            adaV[i] = 0;
        }
    }


    VectorXd sum_beta_squaredNorm(numGroups);
    VectorXd beta_squaredNorm(numGroups);

    logL.resize(K);

    sigmaF = s02F;

    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;

    //double   previt_m0 = 0.0;
    //double   previt_sg = 0.0;
    //double   previt_mu = 0.0;
    //double   previt_se = 0.0;
    //VectorXd previt_Beta(M);


    // Main iteration loop
    // -------------------
    //bool replay_it = false;
    double tot_sync_ar1  = 0.0;
    double tot_sync_ar2  = 0.0;
    int    tot_nsync_ar1 = 0;
    int    tot_nsync_ar2 = 0;
    int    *glob_info, *tasks_len, *tasks_dis, *stats_len, *stats_dis;

    if (opt.bedSync || opt.sparseSync) {
        glob_info = (int*) _mm_malloc(size_t(nranks * 2) * sizeof(int), 64);  check_malloc(glob_info,  __LINE__, __FILE__);
        tasks_len = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(tasks_len,  __LINE__, __FILE__);
        tasks_dis = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(tasks_dis,  __LINE__, __FILE__);
        stats_len = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(stats_len,  __LINE__, __FILE__);
        stats_dis = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(stats_dis,  __LINE__, __FILE__);
    }


    // FH
    if (opt.bayesType == "bayesFHMPI") {
        assert(lambda_var.size() > 0);
        assert(lambda_var.size() == Beta.size());
    }


    for (uint iteration=iteration_start; iteration<opt.chainLength; iteration++) {

        double start_it = MPI_Wtime();
        double it_sync_ar1  = 0.0;
        double it_sync_ar2  = 0.0;
        int    it_nsync_ar1 = 0;
        int    it_nsync_ar2 = 0;

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
        //previt_m0 = m0;
        //marion : this should probably be a vector with sigmaGG
        //previt_sg = sigmaG;
        //previt_se = sigmaE;
        //previt_mu = mu;
        //for (int i=0; i<Ntot; ++i) previt_eps[i]  = epsilon[i];
        //for (int i=0; i<M;    ++i) previt_Beta(i) = Beta(i);

        for (int i=0; i<Ntot; ++i) epsilon[i] += mu;

        double epssum  = 0.0;
        for (int i=0; i<Ntot; ++i) epssum += epsilon[i];
        //printf("epssum = %20.15f with Ntot=%d elements on iteration = %d\n", epssum, Ntot, iteration);

        // update mu
        mu = dist.norm_rng(epssum / dN, sigmaE / dN);
        //printf("it %d, rank %d: mu = %20.15f with dN = %10.1f, sigmaE = %20.15f\n", iteration, rank, mu, dN, sigmaE);

        // We substract again now epsilon =Y-mu-X*beta
        for (int i=0; i<Ntot; ++i) epsilon[i] -= mu;

        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------
        if (opt.shuffleMarkers) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
            //std::random_shuffle(markerI.begin(), markerI.end());
        }

        m0.setZero();
        cass.setZero();


        for (int i=0; i<Ntot; ++i) tmpEps[i] = epsilon[i];

        double cumSumDeltaBetas       = 0.0;
        double task_sum_abs_deltabeta = 0.0;
        int    sinceLastSync          = 0;


        // Loop over (shuffled) markers
        // ----------------------------
        for (int j = 0; j < lmax; j++) {
	 
            sinceLastSync += 1;

            double deltaBeta  = 0.0;

            if (j < M) {

                const int marker  = markerI[j];

                double beta    = Beta(marker);

                double sigE_G  = sigmaE / sigmaG[groups[MrankS[rank] + marker]];
                double sigG_E  = sigmaG[groups[MrankS[rank] + marker]] / sigmaE;
                double i_2sigE = 1.0 / (2.0 * sigmaE);

		//@@@@@@@ FH
		double lambda_tilde = 0.0;
		if (opt.bayesType == "bayesFHMPI") {
		  // Now the variance per mixture changes
		  nu_var(marker) =  dist.inv_gamma_rate_rng(0.5+0.5*v0L,v0L/lambda_var[marker] +1);//sample nu_var in case needed
		  lambda_tilde = tau*c_slab[groups[MrankS[rank] + marker]]/(tau+c_slab[groups[MrankS[rank] + marker]]*lambda_var[marker]);
#ifdef DEBUG
		  std::cout<< "marker:" << marker <<"\n";
		  std::cout << "tau: " << tau << "\n";
		  std::cout << "c: " << c_slab << "\n";
		  std::cout << "local: "<< lambda_var[marker] << "\n";
		  std::cout << "lambda_tilde: "<< lambda_tilde << "\n";
#endif 
		}

                if (adaV[marker]) {
                    
                    //we compute the denominator in the variance expression to save computations
                    //denom = dNm1 + sigE_G * cVaI.segment(1, km1).array();
                    for (int i=1; i<=km1; ++i) {

		        //@@@@@@@ FH
		      	if (opt.bayesType == "bayesFHMPI") {
			  denom(i-1) = dNm1 + sigmaE/lambda_tilde;
			} else {
			  denom(i-1) = dNm1 + sigE_G * cVaI(groups[MrankS[rank] + marker], i);
			}
                        //printf("it %d, rank %d, m %d: denom[%d] = %20.15f, cvai = %20.15f\n", iteration, rank, marker, i-1, denom(i-1), cVaI(groups[MrankS[rank] + marker], i));
                    }

                    double num  = 0.0;
 
                    if (USEBED[marker]) {
                        
                        const uint8_t* rawdata = reinterpret_cast<uint8_t*>(&I1[N1S[marker]]);                            
                            
                        const int dp_version = 1;
                        //const int dp_version = 2;  // Replicates exactly original sparse_dotprod
                        
                        if (dp_version == 1) {
                            
                            double c1 = 0.0, c2 = 0.0;
                            double s1 = 0.0, s2 = 0.0;

                            // main + remainder to avoid a test on idx < Ntot 
                            const int fullb = Ntot / 4;
                            int idx = 0;

                            
                            __m256d vsum1 = _mm256_setzero_pd();
                            __m256d vsum2 = _mm256_setzero_pd();
#ifdef _OPENMP
#pragma omp parallel for reduction(addpd4:vsum1,vsum2)
#endif                              
                            for (int ii=0; ii<fullb; ++ii) {

                                __m256d p4c1  = _mm256_loadu_pd(&(dotp_lut_a[rawdata[ii] * 4]));
                                __m256d p4c2  = _mm256_loadu_pd(&(dotp_lut_b[rawdata[ii] * 4]));
                                __m256d p4eps = _mm256_loadu_pd(&(epsilon[ii * 4]));

                                __m256d p4sum = _mm256_mul_pd(p4c2, p4eps);
                                
                                vsum2 = _mm256_add_pd(vsum2, p4sum);

                                p4sum = _mm256_mul_pd(p4sum, p4c1);                                
                                vsum1 = _mm256_add_pd(vsum1, p4sum);
                            }

                            //EO: to double-check but no reduction available as in avx512
                            s1 = vsum1[0] + vsum1[1] + vsum1[2] + vsum1[3];
                            s2 = vsum2[0] + vsum2[1] + vsum2[2] + vsum2[3];
      
                            // remainder
                            if (Ntot % 4 != 0) {
                                int ii = fullb;
                                for (int iii = 0; iii < Ntot - fullb * 4; iii++) {
                                    idx = rawdata[ii] * 4 + iii;
                                    c1  = dotp_lut_a[idx];
                                    c2  = dotp_lut_b[idx];
                                    s1 += c1 * (c2 * epsilon[ii * 4 + iii]);
                                    s2 +=      (c2 * epsilon[ii * 4 + iii]);
                                }
                            }

                            num = mstd[marker] * (s1 - mave[marker] * s2);

                        } else if (dp_version == 2) {

                            throw("need to adapt if Ntot%4 != 0");
                            exit(1);
                            double c1  = 0.0, c2 = 0.0;
                            double dp1 = 0.0, dp2 = 0.0, dpm = 0.0;
                            
                            double syt = sum_array_elements(epsilon, Ntot);
                            
                            for (int ii=0; ii<snpLenByt; ++ii) {
                                for (int iii=0; iii<4; ++iii) {                                    
                                    c1 = dotp_lut_a[rawdata[ii] * 4 + iii];
                                    c2 = dotp_lut_b[rawdata[ii] * 4 + iii];
                                    if (ii*4 + iii < Ntot) {
                                        if (c1 == 1.0) dp1 +=       epsilon[ii * 4 + iii];
                                        if (c1 == 2.0) dp2 += 2.0 * epsilon[ii * 4 + iii];
                                        dpm  += (1.0 - c2) * epsilon[ii * 4 + iii];
                                    }
                                }
                            }
                            
                            syt -= dpm;
                            num  = dp1 + dp2;
                            num -= mave[marker] * syt;
                            num *= mstd[marker];
                            
                        } else {
                            
                            throw("FATAL  : something wrong with your selection");
                        }

                    } else {
                        
                        num = sparse_dotprod(epsilon,
                                             I1, N1S[marker], N1L[marker],
                                             I2, N2S[marker], N2L[marker],
                                             IM, NMS[marker], NML[marker],
                                             mave[marker],    mstd[marker], Ntot, marker);
                    }
                    
                    //if (j < 10)  printf("it %2d, rank %2d, m %3d: num = %22.15f  USEBED=%d\n", iteration, rank, marker, num, USEBED[marker]);
                    
                    //continue;
                    
                    num += beta * double(Ntot - 1);
                    //printf("it %d, rank %d, mark %d: num = %22.15f, %20.15f, %20.15f\n", iteration, rank, marker, num, mave[marker], mstd[marker]);

                    //muk for the other components is computed according to equations
                    muk.segment(1, km1) = num / denom.array();                    
                    //cout << "muk = " << endl << muk << endl;
                    
                    //first component probabilities remain unchanged
                    for (int i=0; i<K; i++)
                        logL[i] = log(estPi(groups[MrankS[rank] + marker], i));

                    // Update the log likelihood for each component
                    for (int i=1; i<1+km1; i++) {
		        //@@@@@ FH
		        if (opt.bayesType == "bayesFHMPI") {
			    logL[i] = logL[i]
			      - 0.5 * log((lambda_tilde / sigmaE) * dNm1 + 1.0)
			      + muk[i] * num * i_2sigE;
			} else {
			    logL[i] = logL[i]
			      - 0.5 * log(sigG_E * dNm1 * cVa(groups[MrankS[rank] + marker], i) + 1.0)
			      +  muk[i] * num * i_2sigE;
			}
		    }

                    double prob = dist.unif_rng();
                    //printf("%d/%d/%d  prob = %15.10f\n", iteration, rank, j, prob);

                    double acum = 0.0;
                    if (((logL.segment(1,km1).array()-logL[0]).abs().array() > 700.0).any()) {
                        acum = 0.0;
                    } else{
                        acum = 1.0 / ((logL.array()-logL[0]).exp().sum());
                    }
                    //printf("it %d, marker %d, acum = %20.15f, prob = %20.15f\n", iteration, marker, acum, prob);
                    //continue;

                    Acum(marker) = acum;

                    for (int k=0; k<K; k++) {

                        if (prob <= acum || k == km1) { // if prob is less than acum or if we are already in the last mixt.

                            if (k==0) {
                                Beta(marker) = 0.0;
                            } else {
                                Beta(marker) = dist.norm_rng(muk[k], sigmaE/denom[k-1]);
                                //printf("@B@ beta update %4d/%4d/%4d muk[%4d] = %15.10f with prob=%15.10f <= acum = %15.10f, denom = %15.10f, sigmaE = %15.10f: beta = %15.10f\n", iteration, rank, marker, k, muk[k], prob, acum, denom[k-1], sigmaE, Beta(marker));
                            }
                            cass(groups[MrankS[rank] + marker], k) += 1;
                            components[marker]  = k;
                            break;

                        } else {
                            //if too big or too small
                            if (k+1 >= K) {
                                printf("FATAL  : iteration %d, marker = %d, prob = %15.10f, acum = %15.10f logL overflow with %d => %d\n", iteration, marker, prob, acum, k+1, K);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }

                            if (((logL.segment(k+1,K-(k+1)).array()-logL[k+1]).abs().array() > 700.0).any()) {
                                acum += 0.0d; // we compare next mixture to the others, if to big diff we skip
                            } else{
                                acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum()); //if not , sample
                            }
                        }
                    }

		} else { // end of adapative if daniel
                    Beta(marker) = 0.0;
                    Acum(marker) = 1.0; // probability of beta being 0 equals 1.0
                }

                fflush(stdout);


                double betaOld = beta;
                beta           = Beta(marker);
                deltaBeta      = betaOld - beta;
                //printf("deltaBeta = %20.15f\n", deltaBeta);
                

		//@@@@@@@@@@@ FH
		//---- FHDT sample horseshoe local parameters
		// this does not depend on other betas, and the rest of the loop does
		// not depend on the results of this, so could be done in a different
		// thread
		if (opt.bayesType == "bayesFHMPI") {
#ifdef DEBUG
		  std::cout << "<----- sampling local variance -------->\n";
		  std::cout << "tau: "<< tau << "\n";
		  std::cout << "c: " << c << "\n";
		  std::cout << "beta^2: " << beta*beta <<"\n";
#endif
		  //if doing newtonian montecarlo use this
		  // lambda_var(marker) =
		  //   lambdaSampler.sampleLocalVar(10, tau, c, beta*beta );
		  lambda_var(marker)=  dist.inv_gamma_rate_rng(0.5+0.5*v0L,0.5*beta*beta/tau + v0L/nu_var(marker));
#ifdef DEBUG
		  std::cout << "lambda_var: " << lambda_var(marker) << "\n"; 
		  
#endif
		}



                //continue;
                

                // Compute delta epsilon
                if (deltaBeta != 0.0) {

                    //printf("it %d, task %3d, marker %5d has non-zero deltaBeta = %15.10f (%15.10f, %15.10f) => %15.10f) 1,2,M: %lu, %lu, %lu\n", iteration, rank, marker, deltaBeta, mave[marker], mstd[marker],  deltaBeta * mstd[marker], N1L[marker], N2L[marker], NML[marker]);

                    if ((opt.bedSync || opt.sparseSync) && nranks > 1) {

                        mark2sync.push_back(marker);
                        dbet2sync.push_back(deltaBeta);
                        
                    } else {

                        if (USEBED[marker]) {
                                
                            const uint8_t* rawdata = reinterpret_cast<uint8_t*>(&I1[N1S[marker]]);
                            
                            double c1 = 0.0, c2 = 0.0;
                            
                            const double sigdb = mstd[marker] * deltaBeta;

                            const int fullb = Ntot / 4;

                            int idx = 0;

                            // main
#ifdef _OPENMP
#pragma omp parallel for
#endif                               
                            for (int ii=0; ii<fullb; ++ii) {
                                for (int iii=0; iii<4; iii++) {
                                    idx = rawdata[ii] * 4 + iii;
                                    c1 = dotp_lut_a[idx];
                                    c2 = dotp_lut_b[idx];
                                    deltaEps[ii * 4 + iii]  = (c1 - mave[marker]) * c2 * sigdb;
                                }
                            }
                                    
                            // remainder
                            if (Ntot % 4 != 0) {
                                int ii = fullb;
                                for (int iii = 0; iii < Ntot - fullb * 4; iii++) {
                                    idx = rawdata[ii] * 4 + iii;
                                    c1 = dotp_lut_a[idx];
                                    c2 = dotp_lut_b[idx];
                                    deltaEps[ii * 4 + iii]  = (c1 - mave[marker]) * c2 * sigdb;
                                }
                            }

                        } else {
                            
                            sparse_scaadd(deltaEps, deltaBeta, 
                                          I1,  N1S[marker], N1L[marker],
                                          I2,  N2S[marker], N2L[marker],
                                          IM,  NMS[marker], NML[marker],
                                          mave[marker], mstd[marker], Ntot);
                        }

                        // Update local sum of delta epsilon
                        add_arrays(dEpsSum, deltaEps, Ntot);
                    }
                }
            }

            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else {
                //cout << "rank " << rank << " with M=" << M << " waiting for " << lmax << endl;
                deltaBeta = 0.0;

                set_array(deltaEps, 0.0, Ntot);
            }

            task_sum_abs_deltabeta += fabs(deltaBeta);

            
            //continue;


            // Check whether we have a non-zero beta somewhere
            //
            if (nranks > 1 && (sinceLastSync >= opt.syncRate || j == lmax-1)) {

                //EO: watch out this one for the time reported in sync stats
                //MPI_Barrier(MPI_COMM_WORLD);
                
                double tb = MPI_Wtime();
                
                check_mpi(MPI_Allreduce(&task_sum_abs_deltabeta, &cumSumDeltaBetas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

                double te = MPI_Wtime();
                tot_sync_ar1  += te - tb;
                it_sync_ar1   += te - tb;
                tot_nsync_ar1 += 1;
                it_nsync_ar1  += 1;

            } else {

                cumSumDeltaBetas = task_sum_abs_deltabeta;
            }

            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, beta, cumSumDeltaBetas);
            //fflush(stdout);

            //continue;

            if ( (sinceLastSync >= opt.syncRate || j == lmax-1) && cumSumDeltaBetas != 0.0) {

                // Update local copy of epsilon
                //MPI_Barrier(MPI_COMM_WORLD);

                if (nranks > 1) {

                    double tb = MPI_Wtime();

                    // Bed synchronization
                    //
                    if (opt.bedSync) {

                        // 1. Get overall number of markers contributing to synchronization
                        // EO: check types below but should be fine as numbers should be relatively small
                        //                        
                        int task_m2s = (int) mark2sync.size();

                        // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
                        //printf("task %d has %d m2s\n", rank, task_m2s);
                        double* task_stat = (double*) _mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
                        check_malloc(task_stat, __LINE__, __FILE__);
                        for (int i=0; i<task_m2s; i++) {
                            task_stat[2 * i + 0] = mave[ mark2sync[i] ];
                            task_stat[2 * i + 1] = mstd[ mark2sync[i] ] * dbet2sync[i];
                        }

                        check_mpi(MPI_Allgather(&task_m2s, 1, MPI_INT, tasks_len, 1, MPI_INT, MPI_COMM_WORLD), __LINE__, __FILE__);

                        int tdisp_ = 0, sdisp_ = 0, glob_m2s = 0, glob_size = 0;

                        for (int i=0; i<nranks; i++) {
                            glob_m2s     += tasks_len[i];     // in number of markers
                            stats_len[i]  = tasks_len[i] * 2;
                            stats_dis[i]  = sdisp_;
                            sdisp_       += stats_len[i];
                            tasks_len[i] *= snpLenByt;        // each marker is same length in BED
                            tasks_dis[i]  = tdisp_;           // in bytes
                            tdisp_       += tasks_len[i];     // now in bytes
                        }
                        glob_size = tdisp_; // in bytes
                        
                        /*
                        if (rank == 0) {
                            for (int i=0; i<nranks; i++) 
                                printf("| %d:%d", i, tasks_len[i]/(int)snpLenByt);
                            printf(" |  => Tot = %d\n", glob_m2s);
                        }
                        fflush(stdout);
                        */

                        // Alloc to store all task's markers in BED format
                        //
                        
                        char* task_bed = (char*)_mm_malloc(snpLenByt * task_m2s, 64);  check_malloc(task_bed, __LINE__, __FILE__);
                        
                        for (int i=0; i<mark2sync.size(); i++) {
                            
                            int m2si = mark2sync[i];
                            
                            if (USEBED[m2si]) {
                                //printf("rank %d will sync m %d => ISBED\n", rank, m2si);
                                memcpy(&task_bed[i * snpLenByt], reinterpret_cast<char*>(&I1[N1S[m2si]]), snpLenByt);
                            
                            } else {
                                //printf("rank %d will sync m %d => NEED TO CONVERT\n", rank, m2si);
                                //cout << ">ntot = " << Ntot << endl;
                                data.get_bed_marker_from_sparse(&task_bed[(size_t)i * snpLenByt],
                                                                Ntot,
                                                                N1S[m2si], N1L[m2si], &I1[N1S[m2si]],
                                                                N2S[m2si], N2L[m2si], &I2[N2S[m2si]],
                                                                NMS[m2si], NML[m2si], &IM[NMS[m2si]]);
                                // local check ;-)
                                //size_t X1 = 0, X2 = 0, XM = 0;
                                //data.sparse_data_get_sizes_from_raw(&task_bed[(size_t) i * snpLenByt], 1, snpLenByt, data.numNAs, X1, X2, XM);
                                //printf("data.sparse_data_get_sizes_from_raw => (%2d, %3d): X1 = %9lu, X2 = %9lu, XM = %9lu #?# vs %9lu %9lu %9lu\n", rank, i, X1, X2, XM, N1L[m2si], N2L[m2si], NML[m2si]);
                                //fflush(stdout);

                            }
                        }


                        // Collect BED data from all markers to sync from all tasks
                        //
                        char* glob_bed = (char*)_mm_malloc(snpLenByt * glob_m2s, 64);  check_malloc(task_bed, __LINE__, __FILE__);
                        
                        check_mpi(MPI_Allgatherv(task_bed, tasks_len[rank], MPI_CHAR,
                                                 glob_bed, tasks_len, tasks_dis, MPI_CHAR, MPI_COMM_WORLD), __LINE__, __FILE__);

                        double* glob_stats = (double*) _mm_malloc(size_t(glob_m2s * 2) * sizeof(double), 64);
                        check_malloc(glob_stats, __LINE__, __FILE__);
                        
                        check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                                 glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD), __LINE__, __FILE__); 

                        // Now apply corrections from each marker to sync
                        //
                        for (int i=0; i<glob_m2s; i++) {

                            double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                            //printf("rank %d, %2d lambda0 = %20.15f\n", rank, i, lambda0);
                            //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f %d/%d\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1], i, glob_m2s);
                                                       
                            
                            // Get sizes from BED data
                            // note: locally available from markers local to task but ignored for now
                            //
                            size_t X1 = 0, X2 = 0, XM = 0;
                            data.sparse_data_get_sizes_from_raw(&glob_bed[(size_t) i * snpLenByt], 1, snpLenByt, data.numNAs, X1, X2, XM);
                            //printf("data.sparse_data_get_sizes_from_raw => (%2d, %3d): X1 = %9lu, X2 = %9lu, XM = %9lu ###\n", rank, i, X1, X2, XM);
                            //fflush(stdout);
                            // Allocate sparse structure
                            //
                            uint* XI1 = (uint*)_mm_malloc(X1 * sizeof(uint), 64);  check_malloc(XI1, __LINE__, __FILE__);
                            uint* XI2 = (uint*)_mm_malloc(X2 * sizeof(uint), 64);  check_malloc(XI2, __LINE__, __FILE__);
                            uint* XIM = (uint*)_mm_malloc(XM * sizeof(uint), 64);  check_malloc(XIM, __LINE__, __FILE__);
                            
                            
                            // Fill the structure
                            size_t fake_n1s = 0, fake_n2s = 0, fake_nms = 0;
                            size_t fake_n1l = 0, fake_n2l = 0, fake_nml = 0;
                            
                            //EO: bed data already adjusted for NAs
                            data.sparse_data_fill_indices(&glob_bed[(size_t) i * snpLenByt], 1, snpLenByt, data.numNAs,
                                                          &fake_n1s, &fake_n1l, XI1,
                                                          &fake_n2s, &fake_n2l, XI2,
                                                          &fake_nms, &fake_nml, XIM);

                            // Use it
                            // Set all to 0 contribution
                            if (i == 0) {
                                set_array(deltaSum, lambda0, Ntot);
                            } else {
                                offset_array(deltaSum, lambda0, Ntot);
                            }
                            
                            // M -> revert lambda 0 (so that equiv to add 0.0)
                            sparse_add(deltaSum, -lambda0, XIM, 0, XM);
                            
                            // 1 -> add dbet * sig * ( 1.0 - mu)
                            double lambda = glob_stats[2 * i + 1] * (1.0 - glob_stats[2 * i]);
                            sparse_add(deltaSum, lambda - lambda0, XI1, 0, X1);
                            
                            // 2 -> add dbet * sig * ( 2.0 - mu)
                            lambda = glob_stats[2 * i + 1] * (2.0 - glob_stats[2 * i]);
                            sparse_add(deltaSum, lambda - lambda0, XI2, 0, X2);

                            // Free memory and reset pointers
                            _mm_free(XI1);  XI1 = NULL;
                            _mm_free(XI2);  XI2 = NULL;
                            _mm_free(XIM);  XIM = NULL;
                        }
                        //fflush(stdout);

                        //MPI_Barrier(MPI_COMM_WORLD);

                        _mm_free(glob_stats);
                        _mm_free(glob_bed);                       
                        _mm_free(task_bed);
                        _mm_free(task_stat);

                        mark2sync.clear();
                        dbet2sync.clear();                            
                    }

                    // Sparse synchronization
                    // ----------------------
                    else if (opt.sparseSync) {

                        uint task_m2s = (uint) mark2sync.size();
                        //printf("task %3d has %3d markers to share at %d\n", rank, task_m2s, sinceLastSync);
                        //fflush(stdout);

                        // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
                        double* task_stat = (double*) _mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
                        check_malloc(task_stat, __LINE__, __FILE__);

                        // Compute total number of elements to be sent by each task
                        uint task_size = 0;
                        for (int i=0; i<task_m2s; i++) {
                            if (USEBED[mark2sync[i]]) {
                                task_size += snpLenUint + 3;
                            } else {
                                task_size += (N1L[ mark2sync[i] ] + N2L[ mark2sync[i] ] + NML[ mark2sync[i] ] + 3);
                            }
                            task_stat[2 * i + 0] = mave[ mark2sync[i] ];
                            task_stat[2 * i + 1] = mstd[ mark2sync[i] ] * dbet2sync[i];
                            //printf("Task %3d, m2s %d/%d: 1: %8lu, 2: %8lu, m: %8lu, info: 3); stats are (%15.10f, %15.10f)\n", rank, i, task_m2s, N1L[ mark2sync[i] ], N2L[ mark2sync[i] ], NML[ mark2sync[i] ], task_stat[2 * i + 0], task_stat[2 * i + 1]);
                        }
                        //printf("Task %3d final task_size = %8d elements to send from task_m2s = %d markers to sync.\n", rank, task_size, task_m2s);
                        //fflush(stdout);

                        // Get the total numbers of markers and corresponding indices to gather

                        const int NEL = 2;
                        uint task_info[NEL] = {};
                        task_info[0] = task_m2s;
                        task_info[1] = task_size;

                        check_mpi(MPI_Allgather(task_info, NEL, MPI_UNSIGNED, glob_info, NEL, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);

                        int tdisp_ = 0, sdisp_ = 0, glob_m2s = 0, glob_size = 0;
                        for (int i=0; i<nranks; i++) {
                            tasks_len[i]  = glob_info[2 * i + 1];
                            tasks_dis[i]  = tdisp_;
                            tdisp_       += tasks_len[i];
                            stats_len[i]  = glob_info[2 * i] * 2; // number of markers in task i times 2 for stat1 and stat2
                            stats_dis[i]  = sdisp_;
                            sdisp_       += glob_info[2 * i] * 2;
                            glob_size    += tasks_len[i];
                            glob_m2s     += glob_info[2 * i];
                        }
                        //printf("glob_info: markers to sync: %d, with glob_size = %7d elements (sum of all task_size)\n", glob_m2s, glob_size);
                        //fflush(stdout);

                        // Build task's array to spread: | marker 1                             | marker 2
                        //                               | n1 | n2 | nm | data1 | data2 | datam | n1 | n2 | nm | data1 | ...
                        // -------------------------------------------------------------------------------------------------
                        uint* task_dat = (uint*) _mm_malloc(size_t(task_size) * sizeof(uint), 64);
                        check_malloc(task_dat, __LINE__, __FILE__);

                        int loc = 0;

                        for (int i=0; i<task_m2s; i++) {

                            int m2si = mark2sync[i];
                            
                            if (USEBED[m2si]) {

                                task_dat[loc++] = snpLenUint;  // bed data "as is" stored in uint
                                task_dat[loc++] = UINT_MAX;    // switch to detect a bed stored marker
                                task_dat[loc++] = 0;

                                const uint* rawdata = reinterpret_cast<uint*>(&I1[N1S[m2si]]);

                                for (int ii=0; ii<snpLenUint; ii++) {
                                    task_dat[loc++] = rawdata[ii];
                                }

                            } else {

                                task_dat[loc++] = N1L[m2si];
                                task_dat[loc++] = N2L[m2si];
                                task_dat[loc++] = NML[m2si];

                                //cout << "1: " << loc << ", " << N1L[m2si] << endl;
                                for (uint ii = 0; ii < N1L[m2si]; ii++) {
                                    task_dat[loc] = I1[ N1S[m2si] + ii ];  loc += 1;
                                }  
                                //cout << "2: " << loc << ", " << N2L[m2si] << endl;                                                                
                                for (uint ii = 0; ii < N2L[m2si]; ii++) {
                                    task_dat[loc] = I2[ N2S[m2si] + ii ];  loc += 1;
                                }
                                //cout << "M: " << loc << ", " << NML[m2si] << endl;
                                for (uint ii = 0; ii < NML[m2si]; ii++) {
                                    task_dat[loc] = IM[ NMS[m2si] + ii ];  loc += 1;
                                }
                            }
                        }
                        //printf("loc vs task_size = %d vs %d\n", loc, task_size);
                        assert(loc == task_size);

                        
                        // Allocate receive buffer for all the data
                        //if (rank == 0)
                        //    printf("glob_size = %d\n", glob_size);
                        uint* glob_dat = (uint*) _mm_malloc(size_t(glob_size) * sizeof(uint), 64);
                        check_malloc(glob_dat, __LINE__, __FILE__);

                        check_mpi(MPI_Allgatherv(task_dat, task_size, MPI_UNSIGNED,
                                                 glob_dat, tasks_len, tasks_dis, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);
                        _mm_free(task_dat);
                        //cout << "glob_size = " << glob_size << endl;

                        double* glob_stats = (double*) _mm_malloc(size_t(glob_size * 2) * sizeof(double), 64);
                        check_malloc(glob_stats, __LINE__, __FILE__);

                        check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                                 glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD), __LINE__, __FILE__);
                        _mm_free(task_stat);


                        // Compute global delta epsilon deltaSum
                        //
                        size_t loci = 0;
                        double c1 = 0.0, c2 = 0.0;

                        for (int i=0; i<glob_m2s ; i++) {

                            //printf("m2s %d/%d (loci = %lu): %d, %d, %d\n", i, glob_m2s, loci, glob_dat[loci], glob_dat[loci + 1], glob_dat[loci + 2]);
                            double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                            //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1]);

                            if (glob_dat[loci + 1] == UINT_MAX) {

                                // Reset vector
                                if (i == 0)  set_array(deltaSum, 0.0, Ntot);

                                const uint8_t* rawdata = reinterpret_cast<uint8_t*>(&glob_dat[loci + 3]);

                                // main + remainder to avoid a test on idx < Ntot 
                                const int fullb = Ntot / 4;
                                int idx = 0;

                                // main
                                
                                double mu_ = glob_stats[2 * i];
                                double si_ = glob_stats[2 * i + 1];

                                __m256d vmu_ = _mm256_set1_pd(mu_);
                                __m256d vsi_ = _mm256_set1_pd(si_);
                                
#ifdef _OPENMP
#pragma omp parallel for
#endif                                
                                for (int ii=0; ii<fullb; ++ii) {
                                
                                    __m256d p4c1  = _mm256_loadu_pd(&(dotp_lut_a[rawdata[ii] * 4]));
                                    __m256d p4c2  = _mm256_loadu_pd(&(dotp_lut_b[rawdata[ii] * 4]));
                                    
                                    p4c1 = _mm256_sub_pd(p4c1, vmu_);
                                    p4c2 = _mm256_mul_pd(p4c2, vsi_);
                                    p4c2 = _mm256_mul_pd(p4c2, p4c1);

                                    __m256d p4dls = _mm256_loadu_pd(&(deltaSum[ii * 4])); 
                                    
                                    p4dls = _mm256_add_pd(p4dls, p4c2);

                                    _mm256_store_pd(&(deltaSum[ii * 4]), p4dls);

                                }

                                // remainder
                                if (Ntot % 4 != 0) {
                                    int ii = fullb;
                                    for (int iii = 0; iii < Ntot - fullb * 4; iii++) {
                                        idx = rawdata[ii] * 4 + iii;
                                        c1  = dotp_lut_a[idx];
                                        c2  = dotp_lut_b[idx];
                                        deltaSum[ii * 4 + iii] += (c1 - glob_stats[2 * i]) * c2 * glob_stats[2 * i + 1];
                                    }
                                }
                                
                                loci += 3 + glob_dat[loci];  // + 1 contains UINT_MAX to mark BED data

                            } else {
                                
                                if (i == 0) {
                                    set_array(deltaSum, lambda0, Ntot);
                                } else {
                                    offset_array(deltaSum, lambda0, Ntot);
                                }

                                // M -> revert lambda 0 (so that equiv to add 0.0)
                                size_t S = loci + (size_t) (3 + glob_dat[loci] + glob_dat[loci + 1]);
                                size_t L = glob_dat[loci + 2];
                                //cout << "task " << rank << " M: start = " << S << ", len = " << L <<  endl;
                                sparse_add(deltaSum, -lambda0, glob_dat, S, L);
                                
                                // 1 -> add dbet * sig * ( 1.0 - mu)
                                double lambda = glob_stats[2 * i + 1] * (1.0 - glob_stats[2 * i]);
                                //printf("1: lambda = %15.10f, l-l0 = %15.10f\n", lambda, lambda - lambda0);
                                S = loci + 3;
                                L = glob_dat[loci];
                                //cout << "1: start = " << S << ", len = " << L <<  endl;
                                sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);
                                
                                // 2 -> add dbet * sig * ( 2.0 - mu)
                                lambda = glob_stats[2 * i + 1] * (2.0 - glob_stats[2 * i]);
                                S = loci + 3 + glob_dat[loci];
                                L = glob_dat[loci + 1];
                                //cout << "2: start = " << S << ", len = " << L <<  endl;
                                sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);

                                loci += 3 + glob_dat[loci] + glob_dat[loci + 1] + glob_dat[loci + 2];
                            }

                        }

                        _mm_free(glob_stats);
                        _mm_free(glob_dat);

                        mark2sync.clear();
                        dbet2sync.clear();

                    } else {

                        check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

                    }

                    add_arrays(epsilon, tmpEps, deltaSum, Ntot);

                    double te = MPI_Wtime();
                    tot_sync_ar2  += te - tb;
                    it_sync_ar2   += te - tb;
                    tot_nsync_ar2 += 1;
                    it_nsync_ar2  += 1;

                } else { // case nranks == 1

                    //cout << "COME HERE" << endl;
                    add_arrays(epsilon, tmpEps, dEpsSum, Ntot);
                }

                double end_sync = MPI_Wtime();
                //printf("INFO   : synchronization time = %8.3f ms\n", (end_sync - beg_sync) * 1000.0);

                // Store epsilon state at last synchronization
                copy_array(tmpEps, epsilon, Ntot);

                // Reset local sum of delta epsilon
                set_array(dEpsSum, 0.0, Ntot);

                // Reset cumulated sum of delta betas
                cumSumDeltaBetas       = 0.0;
                task_sum_abs_deltabeta = 0.0;
                
                sinceLastSync = 0;                
            }

        } // END PROCESSING OF ALL MARKERS

        //continue;
        

        // Update beta squared norm by group
        beta_squaredNorm.setZero();
        for (int i=0; i<M; i++) {
            beta_squaredNorm[groups[MrankS[rank] + i]] += Beta[i] * Beta[i];
        }
        //printf("rank %d it %d  beta_squaredNorm = %15.10f\n", rank, iteration, beta_squaredNorm);
        //printf("==> after eps sync it %d, rank %d, epsilon[0] = %15.10f %15.10f\n", iteration, rank, epsilon[0], epsilon[Ntot-1]);

	//@@@@@@@ FH
	// Scaled sum of squares
	double scaledBSQN = 0.0;
	if (opt.bayesType == "bayesFHMPI") {
	    for (int i = 0; i < M; i++) {
	        scaledBSQN +=  Beta[i] * Beta[i] / lambda_var[i];
	    }
	}


        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            check_mpi(MPI_Allreduce(beta_squaredNorm.data(), sum_beta_squaredNorm.data(), beta_squaredNorm.size(), MPI_DOUBLE,  MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(cass.data(),             sum_cass.data(),             cass.size(),             MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            beta_squaredNorm = sum_beta_squaredNorm;
            cass             = sum_cass;
        }

        // Update global parameters
        // ------------------------
        for (int i=0; i<numGroups; i++) {

            // Skip empty groups 
            if (MtotGrp[i] == 0)   continue;

            //printf("%d: m0 = %d - %d, v0G = %20.15f\n", i, MtotGrp[i], cass(i, 0), v0G);
            m0[i] = MtotGrp[i] - cass(i, 0);

            // Skip groups with m0 being null or empty cass (adaV in action)
            if (m0[i] == 0 || cass.row(i).sum() == 0) {
                for (int ii=0; ii<M; ii++) {
                    if (groups[MrankS[rank] + ii] == i) {
                        adaV[ii] = 0;
                    }
                }
                sigmaG[i] = 0.0;
                continue;
            }

            // AH: naive check that v0G and s02G were supplied from file
            if (opt.priorsFile != "") {
                v0G  = data.priors(i, 0);
                s02G = data.priors(i, 1);
            }

            // AH: similarly for dirichlet priors
            if (opt.dPriorsFile != "") {
                // get vector of parameters for the current group
                dirc = data.dPriors.row(i).transpose().array();
            }
           

	    if (opt.bayesType == "bayesFHMPI") {
	        //----------------------------FHDT sample hyper parameters
	        hypTau    = dist.inv_gamma_rate_rng(0.5 + 0.5 * v0t, 1.0 / (tau0 * tau0) + 1.0 / tau);
		tau       = dist.inv_gamma_rate_rng(0.5 * (m0[i] + v0t), v0t / hypTau + (0.5 * scaledBSQN));
		c_slab[i] = dist.inv_scaled_chisq_rng(v0c + (double)m0[i],
						      (beta_squaredNorm[i] * (double)m0[i] + v0c * s02G) / (v0c + (double)m0[i]));
		//--------------------------------
		
		sigmaG[i] = beta_squaredNorm[i];
		
		//DT scaling for ishwaran rao
		// sigmaG[i] = SamplerV.sampleGroupVar(1, (double)m0[i] * beta_squaredNorm[i], (double)m0[i], i);
	    } else {
	      sigmaG[i] = dist.inv_scaled_chisq_rng(v0G + (double) m0[i], (beta_squaredNorm[i] * (double) m0[i] + v0G * s02G) / (v0G + (double) m0[i]));
	    }         

            //printf("???? %d: %d, bs %20.15f, m0 %d -> sigmaG[i] = %20.15f, call(%20.15f, %20.15f)\n", rank, i, beta_squaredNorm[i], m0[i], sigmaG[i], v0G + (double) m0[i],  (beta_squaredNorm[i] * (double) m0[i] + v0G * s02G) / (v0G + (double) m0[i]));

            //we moved the pi update here to use the same loop
            VectorXd dirin = cass.row(i).transpose().array().cast<double>() + dirc.array();
            estPi.row(i) = dist.dirichlet_rng(dirin);
        }

        //fflush(stdout);

        //printf("rank %d own sigmaG[0] = %20.15f with Mtot = %d and m0[0] = %d\n", rank, sigmaG[0], Mtot, int(m0[0]));

        // Broadcast sigmaG of rank 0
        check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        //printf("rank %d has sigmaG = %15.10f\n", rank, sigmaG);
        //cout << "sigmaG = " << sigmaG << endl;


        if (rank == 0) {
            printf("\nINFO   : global cass on iteration %d:\n", iteration);
            for (int i=0; i<numGroups; i++) {
                printf("         MtotGrp[%3d] = %8d  | ", i, MtotGrp[i]);
                if (MtotGrp[i] == 0) {
                    printf(" (empty group)");
                } else if (sigmaG[i] == 0.0) {
                    printf(" excluded (sigmaG set to zero)");
                } else {
                    printf(" cass:");
                    for (int ii=0; ii<K; ii++) {
                        printf(" %8d", cass(i, ii));
                    }
                    assert(cass.row(i).sum() == MtotGrp[i]);
                }
                printf("\n");
            }
        }
        fflush(stdout);




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
        uint gamma_length = 0;
        
        if (opt.covariates) {
            
            if (iteration == 0 && rank == 0)
                cout << "COVARIATES with X of size " << data.X.rows() << "x" << data.X.cols() << endl;

            std::shuffle(xI.begin(), xI.end(), dist.rng);

            double gamma_old, num_f, denom_f;
            double sigE_sigF = sigmaE / sigmaF;

            gamma_length = data.X.cols();

            for (int i=0; i<gamma_length; i++) {

                gamma_old = gamma(xI[i]);
                num_f     = 0.0;
                denom_f   = 0.0;

                for (int k=0; k<Ntot; k++) {
                    num_f += data.X(k, xI[i]) * (epsilon[k] + gamma_old * data.X(k, xI[i]));
                }

                denom_f      = dNm1 + sigE_sigF;
                gamma(xI[i]) = dist.norm_rng(num_f/denom_f, sigmaE/denom_f);

                for (int k = 0; k<Ntot ; k++) {
                    epsilon[k] = epsilon[k] + (gamma_old - gamma(xI[i])) * data.X(k, xI[i]);
                    //cout << "adding " << (gamma_old - gamma(xI[i])) * data.X(k, xI[i]) << endl;
                }
            }
            //the next line should be uncommented if we want to use ridge for the other cvoariates.
            //sigmaF = inv_scaled_chisq_rng(0.001 + F, (gamma.squaredNorm() + 0.001)/(0.001+F));
            sigmaF = s02F;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double e_sqn = 0.0d;
        for (int i=0; i<Ntot; ++i) e_sqn += epsilon[i] * epsilon[i];
        //printf("e_sqn = %20.15f, v0E = %20.15f, s02E = %20.15f\n", e_sqn, v0E, s02E);

        //EO: sample sigmaE and broadcast the one from rank 0 to all the others
	sigmaE  = dist.inv_scaled_chisq_rng(v0E+dN, (e_sqn + v0E*s02E)/(v0E+dN));

	//@@@@@@@ FH
	//DT needs to check here
	//EO: do we need the #idef? sigmaE not updated here so leave sigmaE outside for now
	if (opt.bayesType == "bayesFHMPI") {
#ifdef NEWTONIAN
	  //DT scaling like in ishwaran and rao
	  // sigmaE = SamplerE.sampleEpsVar(1, e_sqn);
#endif
	} else {
	  // 
	}


        check_mpi(MPI_Bcast(&sigmaE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        //printf("sigmaE = %20.15f\n", sigmaE);

        double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);


        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
        if (rank%10==0) {
            printf("RESULT : it %4d, rank %4d: proc = %9.3f s, sync = %9.3f (%9.3f + %9.3f), n_sync = %8d (%8d + %8d) (%7.3f / %7.3f), sigmaG = %15.10f, sigmaE = %15.10f, betasq = %15.10f, m0 = %10d\n",
                   iteration, rank, end_it-start_it,
                   it_sync_ar1  + it_sync_ar2,  it_sync_ar1,  it_sync_ar2,
                   it_nsync_ar1 + it_nsync_ar2, it_nsync_ar1, it_nsync_ar2,
                   (it_sync_ar1) / double(it_nsync_ar1) * 1000.0,
                   (it_sync_ar2) / double(it_nsync_ar2) * 1000.0,
                   sigmaG.sum(), sigmaE, beta_squaredNorm.sum(), m0.sum());
            fflush(stdout);
        }

        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+Ntot,((epsilon).squaredNorm()+v0E*s02E)/(v0E+Ntot));
        //printf("sigmaG = %20.15f, sigmaE = %20.15f, e_sqn = %20.15f\n", sigmaG, sigmaE, e_sqn);
        //printf("it %6d, rank %3d: epsilon[0] = %15.10f, y[0] = %15.10f, m0=%10.1f,  sigE=%15.10f,  sigG=%15.10f [%6d / %6d]\n", iteration, rank, epsilon[0], y[0], m0, sigmaE, sigmaG, markerI[0], markerI[M-1]);

        // Broadcast estPi from rank 0 (update moved up)
        check_mpi(MPI_Bcast(estPi.data(), estPi.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);


        // Write output files
        // ------------------
        if (iteration%opt.thin == 0) {

            //TODO adjust for multiple sigmaG and matrix PI
            //EO: now only global parameters to out
            if (rank == 0) {

                int cx = snprintf(buff, LENBUF, "%5d, %4d", iteration, (int) sigmaG.size());
                assert(cx >= 0 && cx < LENBUF);

                for(int jj = 0; jj < sigmaG.size(); ++jj){
                    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", sigmaG(jj));
                    assert(cx >= 0 && cx < LENBUF - strlen(buff));
                }
                
                cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f, %20.15f, %7d, %4d, %2d",  sigmaE, sigmaG.sum()/(sigmaE+sigmaG.sum()), m0.sum(), int(estPi.rows()), int(estPi.cols()));
                assert(cx >= 0 && cx < LENBUF - strlen(buff));

                for (int ii=0; ii<estPi.rows(); ++ii) {
                    for(int kk = 0; kk < estPi.cols(); ++kk) {
                        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", estPi(ii,kk));
                        assert(cx >= 0 && cx < LENBUF - strlen(buff));
                    }
                }

                cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), "\n");
                assert(cx >= 0 && cx < LENBUF - strlen(buff));

                offset = size_t(n_thinned_saved) * strlen(buff);
                check_mpi(MPI_File_write_at(outfh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);

                
                // Write iteration number in .bet, acu, cpn
                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(betfh,  offset,  &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                check_mpi(MPI_File_write_at(acufh,  offset,  &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpnfh,  offset,  &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            
            offset = sizeof(uint) + sizeof(uint) 
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double))
                + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(betfh,  offset,  Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at_all(acufh,  offset,  Acum.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint)
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int))
                + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh,  offset,  components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);
            
            //EO: only iteration number and mu value (double) for the task-wise mus files
            offset  = size_t(n_thinned_saved) * ( sizeof(uint) + sizeof(double) );
            check_mpi(MPI_File_write_at(musfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            offset += sizeof(uint);
            check_mpi(MPI_File_write_at(musfh, offset, &mu,        1, MPI_DOUBLE,   &status), __LINE__, __FILE__);

            n_thinned_saved += 1;
        }


        // Dump the epsilon vector and the marker indexing one
        // Note: single line overwritten at each saving iteration
        // .eps format: uint, uint, double[0, N-1] (it, Ntot, [eps])
        // .mrk format: uint, uint, int[0, M-1]    (it, M,    <mrk>)
        // ------------------------------------------------------
        if (iteration > 0 && iteration%opt.save == 0) {

            // Each task writes its own rng file
            dist.write_rng_state_to_file(rngfp);

            offset = 0;
            check_mpi(MPI_File_write_at(epsfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            if (opt.covariates) {
                check_mpi(MPI_File_write_at(gamfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                check_mpi(MPI_File_write_at(xivfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }

            offset = sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh,  offset, &Ntot,      1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh,  offset, &M,         1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(xbetfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);                
            check_mpi(MPI_File_write_at(xcpnfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            if (opt.covariates) {
                check_mpi(MPI_File_write_at(gamfh, offset, &gamma_length, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                check_mpi(MPI_File_write_at(xivfh, offset, &gamma_length, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            
            offset = sizeof(uint) + sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh, offset, epsilon,        Ntot,           MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, offset, markerI.data(), markerI.size(), MPI_INT,    &status), __LINE__, __FILE__);
            if (opt.covariates) {
                check_mpi(MPI_File_write_at(gamfh, offset, gamma.data(),   gamma_length,  MPI_DOUBLE, &status), __LINE__, __FILE__);
                check_mpi(MPI_File_write_at(xivfh, offset, xI.data(),      gamma_length,  MPI_INT,    &status), __LINE__, __FILE__);
            }

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(xbetfh, offset, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(xcpnfh, offset, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);



            //if (iteration == 0) {
            //    printf("rank %d dumping eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot-1]);
            //}
            //EO: to remove once MPI version fully validated; use the check_epsilon utility to retrieve
            //    the corresponding values from the .eps file
            //    Print only first and last value handled by each task
            //printf("%4d/%4d epsilon[%5d] = %15.10f, epsilon[%5d] = %15.10f\n", iteration, rank, IrankS[rank], epsilon[IrankS[rank]], IrankS[rank]+IrankL[rank]-1, epsilon[IrankS[rank]+IrankL[rank]-1]);


            //EO system call to create a tarball of the dump
            //TODO: quite rough, make it more selective...
            //----------------------------------------------
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                time_t now = time(0);
                tm *   ltm = localtime(&now);
                int    n   = 0;
                char tar[LENBUF];

                n=sprintf(tar, "dump_%s_%05d__%4d-%02d-%02d_%02d-%02d-%02d.tar",
                          opt.mcmcOutNam.c_str(), iteration,
                          1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                          ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
                assert(n > 0);

                printf("INFO   : will create tarball %s in %s with file listed in %s.\n",
                       tar, opt.mcmcOutDir.c_str(), lstfp.c_str());
                //std::system(("ls " + opt.mcmcOut + ".*").c_str());
                //string cmd = "tar -czf " + opt.mcmcOutDir + "/tarballs/" + targz + " -T " + lstfp;
                string cmd = "tar -cf " + opt.mcmcOutDir + "/tarballs/" + tar + " -T " + lstfp;
                //cout << "cmd >>" << cmd << "<<" << endl;
                std::system(cmd.c_str());
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        //double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Close output files
    check_mpi(MPI_File_close(&outfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xbetfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&epsfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&cpnfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xcpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&acufh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&mrkfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xivfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&gamfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_close(&musfh),  __LINE__, __FILE__);


    // Release memory
    _mm_free(y);
    _mm_free(epsilon);
    _mm_free(tmpEps);
    //free(previt_eps);
    _mm_free(deltaEps);
    _mm_free(dEpsSum);
    _mm_free(deltaSum);
    _mm_free(mave);
    _mm_free(mstd);
    _mm_free(USEBED);
    _mm_free(N1S);
    _mm_free(N1L);
    _mm_free(I1);
    _mm_free(N2S);
    _mm_free(N2L);
    _mm_free(I2);
    _mm_free(NMS);
    _mm_free(NML);
    _mm_free(IM);

    if (opt.bedSync || opt.sparseSync) {
        _mm_free(glob_info);
        _mm_free(tasks_len);
        _mm_free(tasks_dis);
        _mm_free(stats_len);
        _mm_free(stats_dis);
    }

    const auto et3 = std::chrono::high_resolution_clock::now();
    const auto dt3 = et3 - st3;
    const auto du3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();
    if (rank == 0)
        printf("INFO   : rank %4d, time to process the data: %.3f sec, with %.3f (%.3f, %.3f) = %4.1f%% spent on allred (%d, %d)\n",
               rank, du3 / double(1000.0),
               tot_sync_ar1 + tot_sync_ar2, tot_sync_ar1, tot_sync_ar2,
               (tot_sync_ar1 + tot_sync_ar2) / (du3 / double(1000.0)) * 100.0,
               tot_nsync_ar1, tot_nsync_ar2);

    return 0;
}


//EO: rough check on RAM requirements.
//    Simulates the RAM usage on a single node to see if would fit,
//    assuming at least the same amount of RAM is available on the nodes
//    that would be assigned to the job
//----------------------------------------------------------------------
int BayesRRm::checkRamUsage() {

    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (opt.checkRam && nranks != 1) {
        printf("#FATAL#: --check-RAM option runs only in single task mode (SIMULATION of --check-RAM-tasks with max --check-RAM-tasks-per-node tasks per node)!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (opt.checkRamTasks <= 0) {
        printf("#FATAL#: --check-RAM-tasks must be strictly positive! Was %d\n", opt.checkRamTasks);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (opt.checkRamTpn <= 0) {
        printf("#FATAL#: --check-RAM-tasks-per-node must be strictly positive! Was %d\n", opt.checkRamTpn);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint Mtot = set_Mtot(rank);

    // Alloc memory for sparse representation
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    N1S = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);

    MPI_File   slfh;
    MPI_Status status;

    string sparseOut = mpi_get_sparse_output_filebase(rank);
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
    int tpn = 0;
    tpn    = opt.checkRamTpn;
    nranks = opt.checkRamTasks;
    if (opt.markerBlocksFile != "") nranks = data.numBlocks;

    int nnodes = int(ceil(double(nranks)/double(tpn)));
    printf("INFO  : will simulate %d ranks on %d nodes with max %d tasks per node.\n", nranks, nnodes, tpn);

    int proctasks = 0;

    printf("Estimation RAM usage when dataset is processed with %2d nodes and %2d tasks per node\n", nnodes, tpn);

    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);
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

            allocs1[i] = (uint*)_mm_malloc(n1 * sizeof(uint), 64);  check_malloc(allocs1[i], __LINE__, __FILE__);
            allocs2[i] = (uint*)_mm_malloc(n2 * sizeof(uint), 64);  check_malloc(allocs2[i], __LINE__, __FILE__);
            allocsm[i] = (uint*)_mm_malloc(nm * sizeof(uint), 64);  check_malloc(allocsm[i], __LINE__, __FILE__);

            printf("   - t %3d  n %2d sm %7d  l %6d markers. Number of 1s: %15lu, 2s: %15lu, ms: %15lu => RAM: %7.3f GB; RAM on node: %7.3f with %d tasks\n", i, node, MrankS[node*tpn + i], MrankL[node*tpn + i], n1, n2, nm, GB, ramnode, tpn);

            proctasks++;
        }

        // free memory on the node
        for (int i=0; i<tpn; i++) {
            _mm_free(allocs1[i]);
            _mm_free(allocs2[i]);
            _mm_free(allocsm[i]);
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
    _mm_free(N1S); _mm_free(N1L);
    _mm_free(N2S); _mm_free(N2L);
    _mm_free(NMS); _mm_free(NML);

    return 0;
}


// Get directory and basename of bed file (passed with no extension via command line)
// ----------------------------------------------------------------------------------
string BayesRRm::mpi_get_sparse_output_filebase(const int rank) {

    std::string dir, bsn;

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



uint BayesRRm::set_Ntot(const int rank) {

    uint Ntot = opt.numberIndividuals;

    if (Ntot == 0) {
        printf("FATAL  : opt.numberIndividuals is zero! Set it via --number-individuals in call.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (Ntot != data.numInds - data.numNAs) {
        if (rank == 0)
            printf("WARNING: opt.numberIndividuals set to %d but will be adjusted to %d - %d = %d due to NAs in phenotype file.\n", Ntot, data.numInds, data.numNAs, data.numInds-data.numNAs);
    }

    return Ntot;
}


uint BayesRRm::set_Mtot(const int rank) {

    uint Mtot = opt.numberMarkers;

    if (Mtot == 0) throw("FATAL  : opt.numberMarkers is zero! Set it via --number-markers in call.");

    // Block marker definition has precedence over requested number of markers
    if (opt.markerBlocksFile != "" && opt.numberMarkers > 0) {
        opt.numberMarkers = 0;
        if (rank == 0)
            printf("WARNING: --number-markers option ignored, a marker block definition file was passed!\n");
    }

    if (opt.numberMarkers > 0 && opt.numberMarkers < Mtot) {
        Mtot = opt.numberMarkers;
        if (rank == 0)
            printf("INFO   : Option passed to process only %d markers!\n", Mtot);
    }

    return Mtot;
}
