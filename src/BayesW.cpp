/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include <cstdlib>
#include "BayesW.hpp"
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
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_utils.hpp"
#endif

#include <omp.h>
#include "BayesW_arms.h"
#include <math.h>

/* Pre-calculate used constants */
#define PI 3.14159
#define PI2 6.283185
#define sqrtPI 1.77245385090552
#define EuMasc 0.577215664901532

/*
BayesW::BayesW(Data &data, Options &opt, const long memPageSize)
    : data(data)
    , opt(opt)
    , bedFile(opt.bedFile + ".bed")
    , memPageSize(memPageSize)
    , seed(opt.seed)
    , max_iterations(opt.chainLength)
    , thinning(opt.thin)
    , burn_in(opt.burnin)
    //, dist(opt.seed) //Was commented out
    , usePreprocessedData(opt.analysisType == "PPBayes")
    , showDebug(false)
{
    float* ptr =static_cast<float*>(&opt.S[0]);
}
*/
BayesW::~BayesW()
{
}

/*
void BayesW::offset_vector_f64(double* __restrict__ vec, const double offset, const int N) {
#ifdef __INTEL_COMPILER
    __assume_aligned(vec,   64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        vec[i] += offset;
    }
}


void BayesW::set_vector_f64(double* __restrict__ vec, const double val, const int N) {
  
    const int N8 = (N/8) * 8;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume(N8%8==0);
#endif
    //#pragma unroll(8)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N8; i++) {
        vec[i] = val;
    }

    for (int i=N8; i<N; ++i) {
        vec[i] = val;
    }
}
*/

/*
  inline void copy_vector_f64(double* __restrict__ dest, const double* __restrict__ source, const int N) {

  const int N8 = (N/8) * 8;

  __assume_aligned(dest,   64);
  __assume_aligned(source, 64);
  __assume(N8%8==0);

  #pragma unroll(8)
  for (int i=0; i<N8; i++) {
  dest[i] = source[i];
  }

  for (int i=N8; i<N; ++i) {
  dest[i] = source[i];
  }
  }
*/

/*
void BayesW::copy_vector_f64(double* __restrict__ dest, const double* __restrict__ source, const int N) {
#ifdef __INTEL_COMPILER
    __assume_aligned(dest,   64);
    __assume_aligned(source, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        dest[i] = source[i];
    }
}


double BayesW::sum_vector_elements_f64_base(const double* __restrict__ vec, const int N) {

    const int N8 = (N/8) * 8;
    double sum = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume(N8%8==0);
#endif
#pragma unroll(8)
    for (int i=0; i<N8; i++) {
        sum += vec[i];
    }

    for (int i=N8; i<N; ++i) {
        sum += vec[i];
    }

    return sum;
}


double BayesW::sum_vector_elements_f64(const double* __restrict__ vec, const int N) {

    double sum = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
#endif
    //#pragma unroll
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum)
#endif
    for (int i=0; i<N; i++) {
        sum += vec[i];
    }

    return sum;
}


void BayesW::sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2,
                               const int N) {
#ifdef __INTEL_COMPILER
    __assume_aligned(in1, 64);
    __assume_aligned(in2, 64);
    __assume_aligned(out, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        out[i] = in1[i] + in2[i];
    }
}
*/

/*
  inline void sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2,
  const int N) {

  const int N8 = (N/8) * 8;

  __assume_aligned(in1, 64);
  __assume_aligned(in2, 64);
  __assume_aligned(out, 64);
  __assume(N8%8==0);

  //#pragma unroll(8)
  #pragma omp parallel for
  for (int i=0; i<N8; i++) {
  out[i] = in1[i] + in2[i];
  }

  for (int i=N8; i<N; ++i) {
  out[i] = in1[i] + in2[i];
  }
  }
*/

/*
void BayesW::sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const int N) {
    
    const int N8 = (N/8) * 8;
#ifdef __INTEL_COMPILER
    __assume_aligned(in1, 64);
    __assume_aligned(out, 64);
    __assume(N8%8==0);
#endif
    //#pragma unroll(8)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N8; i++) {
        out[i] += in1[i];
    }

    for (int i=N8; i<N; ++i) {
        out[i] += in1[i];
    }
}


inline void sum_vectors_f64_ref(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2,
                                const int N) {
#ifdef __INTEL_COMPILER
    __assume_aligned(in1, 64);
    __assume_aligned(in2, 64);
    __assume_aligned(out, 64);
#endif
    for (int i=0; i<N; ++i) {
        out[i] = in1[i] + in2[i];
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


inline void sparse_add(double*       __restrict__ vec,
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


void BayesW::sparse_scaadd(double*     __restrict__ vout,
                             const double  dMULT,
                             const uint* __restrict__ I1, const size_t N1S, const size_t N1L,
                             const uint* __restrict__ I2, const size_t N2S, const size_t N2L,
                             const uint* __restrict__ IM, const size_t NMS, const size_t NML,
                             const double  mu,
                             const double  sig_inv,
                             const int     N) {
    
    if (dMULT == 0.0) {
        set_vector_f64(vout, 0.0, N);

    } else {
        double aux = mu * sig_inv * dMULT;
        //printf("sparse_scaadd aux = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", aux, mu, sig_inv * dMULT);
        set_vector_f64(vout, -aux, N);

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
        dp += vec[ IX[i] ] * fac;
    }

    return dp;
}


double BayesW::sparse_dotprod(const double* __restrict__ vin1,
                                const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                                const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                                const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                                const double               mu, 
                                const double               sig_inv,
                                const int                  N,
                                const int                  marker) {
    
    double dp  = 0.0;
    double syt = 0.0;

    dp += partial_sparse_dotprod(vin1, I1, N1S, N1L, 1.0);

    dp += partial_sparse_dotprod(vin1, I2, N2S, N2L, 2.0);

    dp *= sig_inv;

    syt += sum_vector_elements_f64(vin1, N);
    
    syt += partial_sparse_dotprod(vin1, IM, NMS, NML, -1.0);

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

#ifdef USE_MPI


// Define blocks of markers to be processed by each task
// By default processes all markers
// -----------------------------------------------------
void BayesW::mpi_define_blocks_of_markers(const int Mtot, int* MrankS, int* MrankL, const uint nblocks) {

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
void BayesW::check_whole_array_was_set(const uint* array, const size_t size, const int linenumber, const char* filename) {

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
void BayesW::write_sparse_data_files(const uint bpr) {

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
        data.sparse_data_get_sizes_from_raw(rawdata, MLi, snpLenByt, N1, N2, NM);
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
        data.sparse_data_get_sizes_from_raw(rawdata, uint(MLi), snpLenByt, N1, N2, NM);
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
*/

//Already defined in BayesRRm.cpp
/*
size_t get_file_size(const std::string& filename) {
    struct stat st;
    if(stat(filename.c_str(), &st) != 0) { return 0; }
    return st.st_size;   
}
*/

/*
void BayesW::mpi_assign_blocks_to_tasks(const uint numBlocks, const vector<int> blocksStarts, const vector<int> blocksEnds, const uint Mtot, const int nranks, const int rank, int* MrankS, int* MrankL, int& lmin, int& lmax) {
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
*/

/* Function to check if ARS resulted with error*/
inline void errorCheck(int err){
	if(err>0){
		cout << "Error code = " << err << endl;
		exit(1);
	}
}


/* Function for the log density of mu */
inline double mu_dens(double x, void *norm_data)
/* We are sampling mu (denoted by x here) */
{
	double y;

	/* In C++ we need to do a static cast for the void data */
	pars p = *(static_cast<pars *>(norm_data));

	/* cast voided pointer into pointer to struct norm_parm */
	y = - p.alpha * x * p.d - (( (p.epsilon).array()  - x) * p.alpha - EuMasc).exp().sum() - x*x/(2*p.sigma_mu);
	return y;
};

/* Function for the log density of some "fixed" covariate effect */
inline double theta_dens(double x, void *norm_data)
/* We are sampling beta (denoted by x here) */
{
	double y;
	/* In C++ we need to do a static cast for the void data */
	pars p = *(static_cast<pars *>(norm_data));

	/* cast voided pointer into pointer to struct norm_parm */
	y = - p.alpha * x * p.sum_failure - (((p.epsilon -  p.X_j * x)* p.alpha).array() - EuMasc).exp().sum() - x*x/(2*p.sigma_mu); // Prior is the same currently for intercepts and fixed effects
	return y;
};

/* Function for the log density of alpha */
inline double alpha_dens(double x, void *norm_data)
/* We are sampling alpha (denoted by x here) */
{
	double y;

	/* In C++ we need to do a static cast for the void data */
	pars_alpha p = *(static_cast<pars_alpha *>(norm_data));
	y = (p.alpha_0 + p.d - 1) * log(x) + x * ((p.epsilon.array() * p.failure_vector.array()).sum() - p.kappa_0) -
			((p.epsilon * x).array() - EuMasc).exp().sum() ;
	return y;
};

/* Sparse version for function for the log density of beta: uses mixture component from the structure norm_data */
inline double beta_dens(double x, void *norm_data)
/* We are sampling beta (denoted by x here) */
{
	double y;
	/* In C++ we need to do a static cast for the void data */
	pars_beta_sparse p = *(static_cast<pars_beta_sparse *>(norm_data));

	y = -p.alpha * x * p.sum_failure -
			exp(p.alpha*x*p.mean_sd_ratio)* (p.vi_0 + p.vi_1 * exp(-p.alpha*x/p.sd) + p.vi_2 * exp(-2*p.alpha*x/p.sd))
			-x * x / (2 * p.mixture_classes(p.used_mixture) * p.sigma_b) ;
	return y;
};




//The function for integration
inline double gh_integrand_adaptive(double s,double alpha, double dj, double sqrt_2Ck_sigmab,
		double vi_sum, double vi_2, double vi_1, double vi_0, double mean, double sd, double mean_sd_ratio){
	//vi is a vector of exp(vi)
	double temp = -alpha *s*dj*sqrt_2Ck_sigmab +
			vi_sum - exp(alpha*mean_sd_ratio*s*sqrt_2Ck_sigmab) *
			(vi_0 + vi_1 * exp(-alpha * s*sqrt_2Ck_sigmab/sd) + vi_2* exp(-2 * alpha * s*sqrt_2Ck_sigmab/sd))
			-pow(s,2);
	return exp(temp);
}


//Calculate the value of the integral using Adaptive Gauss-Hermite quadrature
//Let's assume that mu is always 0 for speed
double BayesW::gauss_hermite_adaptive_integral(int k, double sigma, string n, double vi_sum, double vi_2, double vi_1, double vi_0,
		double mean, double sd, double mean_sd_ratio){

	double temp = 0;
	double sqrt_2ck_sigma = sqrt(2*used_data.mixture_classes(k)*used_data_beta.sigma_b);

	if(n == "3"){
		double x1,x2;
		double w1,w2,w3;

		x1 = 1.2247448713916;
		x2 = -x1;

		w1 = 1.3239311752136;
		w2 = w1;

		w3 = 1.1816359006037;

		x1 = sigma*x1;
		x2 = sigma*x2;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
				vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
						w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
								vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
								w3;
	}
	// n=5
	else if(n == "5"){
		double x1,x2,x3,x4;//x5;
		double w1,w2,w3,w4,w5; //These are adjusted weights

		x1 = 2.0201828704561;
		x2 = -x1;
		w1 = 1.181488625536;
		w2 = w1;

		x3 = 0.95857246461382;
		x4 = -x3;
		w3 = 0.98658099675143;
		w4 = w3;

		//	x5 = 0.0;
		w5 = 0.94530872048294;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		//x5 = sigma*x5;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
				vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
						w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
								vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
								w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
										vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
										w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
												vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
												w5 ;//* gh_integrand_adaptive(x5,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j); // This part is just 1
	}else if(n == "7"){
		double x1,x2,x3,x4,x5,x6;
		double w1,w2,w3,w4,w5,w6,w7; //These are adjusted weights

		x1 = 2.6519613568352;
		x2 = -x1;
		w1 = 1.1013307296103;
		w2 = w1;

		x3 = 1.6735516287675;
		x4 = -x3;
		w3 = 0.8971846002252;
		w4 = w3;

		x5 = 0.81628788285897;
		x6 = -x5;
		w5 = 0.8286873032836;
		w6 = w5;

		w7 = 0.81026461755681;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
				vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
						w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
								vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
								w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
										vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
										w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
												vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
												w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
														vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
														w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
																vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
																w7;
	}else if(n == "9"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9; //These are adjusted weights

		x1 = 3.1909932017815;
		x2 = -x1;
		w1 = 1.0470035809767;
		w2 = w1;

		x3 = 2.2665805845318;
		x4 = -x3;
		w3 = 0.84175270147867;
		w4 = w3;

		x5 = 1.4685532892167;
		x6 = -x3;
		w5 = 0.7646081250946;
		w6 = w5;

		x7 = 0.72355101875284;
		x8 = -x7;
		w7 = 0.73030245274509;
		w8 = w7;

	//	x9 = 0;
		w9 = 0.72023521560605;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w9 ;//* gh_integrand_adaptive(x9,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "11"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11; //These are adjusted weights

		x1 = 3.6684708465596;
		x2 = -x1;
		w1 = 1.0065267861724;
		w2 = w1;

		x3 = 2.7832900997817;
		x4 = -x3;
		w3 = 0.802516868851;
		w4 = w3;

		x5 = 2.0259480158258;
		x6 = -x3;
		w5 = 0.721953624728;
		w6 = w5;

		x7 = 1.3265570844949;
		x8 = -x7;
		w7 = 0.6812118810667;
		w8 = w7;

		x9 = 0.6568095668821;
		x10 = -x9;
		w9 = 0.66096041944096;
		w10 = w9;

		//x11 = 0.0;
		w11 = 0.65475928691459;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		//	x11 = sigma*x11;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w11 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "13"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13; //These are adjusted weights

		x1 = 4.1013375961786;
		x2 = -x1;
		w1 = 0.97458039564;
		w2 = w1;

		x3 = 3.2466089783724;
		x4 = -x3;
		w3 = 0.7725808233517;
		w4 = w3;

		x5 = 2.5197356856782;
		x6 = -x3;
		w5 = 0.6906180348378;
		w6 = w5;

		x7 = 1.8531076516015;
		x8 = -x7;
		w7 = 0.6467594633158;
		w8 = w7;

		x9 = 1.2200550365908;
		x10 = -x9;
		w9 = 0.6217160552868;
		w10 = w9;

		x11 = 0.60576387917106;
		x12 = -x11;
		w11 = 0.60852958370332;
		w12 = w11;

		//x13 = 0.0;
		w13 = 0.60439318792116;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		x11 = sigma*x11;
		x12 = sigma*x12;


		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w13 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "15"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15; //These are adjusted weights

		x1 = 4.4999907073094;
		x2 = -x1;
		w1 = 0.94836897082761;
		w2 = w1;

		x3 = 3.6699503734045;
		x4 = -x3;
		w3 = 0.7486073660169;
		w4 = w3;

		x5 = 2.9671669279056;
		x6 = -x3;
		w5 = 0.666166005109;
		w6 = w5;

		x7 = 2.3257324861739;
		x8 = -x7;
		w7 = 0.620662603527;
		w8 = w7;

		x9 = 1.7199925751865;
		x10 = -x9;
		w9 = 0.5930274497642;
		w10 = w9;

		x11 = 1.1361155852109;
		x12 = -x11;
		w11 = 0.5761933502835;
		w12 = w11;

		x13 = 0.5650695832556;
		x14 = -x13;
		w13 = 0.5670211534466;
		w14 = w13;

		//x15 = 0.0;
		w15 = 0.56410030872642;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		x11 = sigma*x11;
		x12 = sigma*x12;
		x13 = sigma*x13;
		x14 = sigma*x14;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
				w15 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else{
		cout << "Possible number of quad_points = 3,5,7,9,11,13,15" << endl;
		exit(1);
	}

	return sigma*temp;
}


//Pass the vector post_marginals of marginal likelihoods by reference
void BayesW::marginal_likelihood_vec_calc(VectorXd prior_prob, VectorXd &post_marginals, string n,
		double vi_sum, double vi_2, double vi_1, double vi_0, double mean, double sd, double mean_sd_ratio){
	double exp_sum = (vi_1 * (1 - 2 * mean) + 4 * (1-mean) * vi_2 + vi_sum * mean * mean) /(sd*sd) ;

	for(int i=0; i < used_data_beta.mixture_classes.size(); i++){
		//Calculate the sigma for the adaptive G-H
		double sigma = 1.0/sqrt(1 + used_data_beta.alpha * used_data_beta.alpha * used_data_beta.sigma_b * used_data_beta.mixture_classes(i) * exp_sum);
		post_marginals(i+1) = prior_prob(i+1) * gauss_hermite_adaptive_integral(i, sigma, n, vi_sum,  vi_2,  vi_1,  vi_0,
				mean, sd, mean_sd_ratio);
	}
}


// Function for sampling fixed effect (theta_i)
void BayesW::sampleTheta(int fix_i){
	// ARS parameters
	int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 4 ;
	int neval;
	double xsamp[0], xcent[10], qcent[10] = {5., 30., 70., 95.};
	double convex = 1.0;
	int dometrop = 0;
	double xprev = 0.0;
	double xinit[4] = {theta(fix_i)-0.01, theta(fix_i),  theta(fix_i)+0.005, theta(fix_i)+0.01};     // Initial abscissae
	double *p_xinit = xinit;

	double xl = -2;
	double xr = 2;			  // Initial left and right (pseudo) extremes

	used_data.X_j = data.X.col(fix_i).cast<double>();  //Take from the fixed effects matrix
	used_data.sum_failure = sum_failure_fix(fix_i);

	used_data.epsilon = epsilon.array() + (used_data.X_j * theta(fix_i)).array(); // Adjust residual

	// Sample using ARS
	err = arms(xinit,ninit,&xl,&xr,theta_dens,&used_data,&convex,
			npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
	errorCheck(err);

	theta(fix_i) = xsamp[0];  // Save the new result
	epsilon = used_data.epsilon - used_data.X_j * theta(fix_i); // Adjust residual
}

void BayesW::init(unsigned int markerCount, unsigned int individualCount, unsigned int fixedCount)
{
	// Read the failure indicator vector
	data.readFailureFile(opt.failureFile);

	// Component variables
	pi_L = VectorXd(K);           		 // prior mixture probabilities
	marginal_likelihoods = VectorXd(K);  // likelihood for each mixture component
	v = VectorXd(K);            		 // vector storing the component assignment

	// Linear model variables
	Beta = VectorXd(markerCount);           // effect sizes
	theta = VectorXd(fixedCount);

	sum_failure = VectorXd(markerCount);	// Vector to sum SNP data vector * failure vector per SNP
	sum_failure_fix = VectorXd(fixedCount); // Vector to sum fixed vector * failure vector per fixed effect

	//phenotype vector
	y = VectorXd();
	//residual vector
	epsilon = VectorXd();

	//vi vector
	vi = VectorXd(individualCount);

	// Resize the vectors in the structure
	used_data.X_j = VectorXd(individualCount);
	used_data.epsilon.resize(individualCount);
	used_data_alpha.epsilon.resize(individualCount);

	// Init the working variables
	const int km1 = K - 1;

	//vector with component class for each marker
	components = VectorXi(markerCount);
	components.setZero();

	//set priors for pi parameters
	//Give all mixtures (except 0 class) equal initial probabilities
	pi_L(0) = 0.99;
	pi_L.segment(1,km1).setConstant((1-pi_L(0))/km1);

	marginal_likelihoods.setOnes();   //Initialize with just ones

	Beta.setZero();
	theta.setZero();

	//initialize epsilon vector as the phenotype vector
	y = data.y.cast<double>().array();

	epsilon = y;
	mu = y.mean();       // mean or intercept
	// Initialize the variables in structures
	//Save variance classes
	used_data.mixture_classes.resize(km1);
	used_data_beta.mixture_classes.resize(km1);  //The future solution


	for(int i=0;i<(km1);i++){
		used_data.mixture_classes(i) = opt.S[i];   //Save the mixture data (C_k)
		used_data_beta.mixture_classes(i) = opt.S[i];
	}

	//Store the vector of failures only in the structure used for sampling alpha
	used_data_alpha.failure_vector = data.fail.cast<double>();

	double denominator = (6 * ((y.array() - mu).square()).sum()/(y.size()-1));
	used_data.alpha = PI/sqrt(denominator);    // The shape parameter initial value
	used_data_beta.alpha = PI/sqrt(denominator);    // The shape parameter initial value


	for(int i=0; i<(y.size()); ++i){
		(used_data.epsilon)[i] = y[i] - mu ; // Initially, all the BETA elements are set to 0, XBeta = 0
		epsilon[i] = y[i] - mu;
	}

	used_data_beta.sigma_b = PI2/ (6 * pow(used_data_beta.alpha,2) * markerCount ) ;


	/* Prior value selection for the variables */
	/* At the moment we set them to be weakly informative (in .hpp file) */
	/* alpha */
	used_data_alpha.alpha_0 = alpha_0;
	used_data_alpha.kappa_0 = kappa_0;
	/* mu */
	used_data.sigma_mu = sigma_mu;
	/* sigma_b */
	used_data.alpha_sigma = alpha_sigma;
	used_data.beta_sigma = beta_sigma;

	// Save the number of events
	used_data.d = used_data_alpha.failure_vector.array().sum();
	used_data_alpha.d = used_data.d;

	// Reading the Xj*failure sum in sparse format:
	/*for(int marker=0; marker < markerCount; marker++){
		std::vector<int> oneIndices = data.Zones[marker]; //Take the vector of indices
		std::vector<int> twoIndices = data.Ztwos[marker]; //Take the vector of indices

		int temp_sum = 0;
		for(int i=0; i < oneIndices.size(); i++){
			temp_sum += used_data_alpha.failure_vector(oneIndices[i]);
		}
		for(int i=0; i < twoIndices.size(); i++){
			temp_sum += 2*used_data_alpha.failure_vector(twoIndices[i]);
		}

		sum_failure(marker) = (temp_sum - data.means(marker) * used_data_alpha.failure_vector.array().sum()) / data.sds(marker);
	}*/

	//If there are fixed effects, find the same values for them
	if(fixedCount > 0){
		for(int fix_i=0; fix_i < fixedCount; fix_i++){
			sum_failure_fix(fix_i) = ((data.X.col(fix_i).cast<double>()).array() * used_data_alpha.failure_vector.array()).sum();
		}
	}
}

//EO: MPI GIBBS
//-------------
int BayesW::runMpiGibbs_bW() {

#ifdef _OPENMP
#warning "using OpenMP"
#endif

	const unsigned int numFixedEffects(data.numFixedEffects);
	const int km1 = K - 1;

    char   buff[LENBUF]; 
    int    nranks, rank, name_len, result;
    double dalloc = 0.0;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File   outfh, betfh, epsfh, gamfh, cpnfh, acufh, mrkfh, xivfh;
    MPI_Status status;
    MPI_Info   info;
    MPI_Offset offset, betoff, cpnoff, epsoff;

    // Set up processing options
    // -------------------------
    if (rank < 0) {
        opt.printBanner();
        opt.printProcessingOptions();
    }

    // Set Ntot and Mtot
    // -----------------
    uint Ntot = set_Ntot(rank);
    const uint Mtot = set_Mtot(rank);
    init(Mtot, Ntot, numFixedEffects);
    //Reset the dist
    dist.reset_rng((uint)(opt.seed + rank*1000));

	
    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    uint M = MrankL[rank];
    if (rank % 10 == 0) {
        printf("INFO   : rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);
    }


    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: at this stage Ntot is not yet adjusted for missing phenotypes,
    //       hence the correction in the call
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    mpi_define_blocks_of_markers(Ntot - data.numNAs, IrankS, IrankL, nranks);


    std::vector<int>    markerI;
    VectorXd            sum_v(K);        // To store the sum of v elements over all ranks

    std::vector<int>     mark2sync;
    std::vector<double>  dbet2sync;


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
    string lstfp = opt.mcmcOut + ".lst";
    string outfp = opt.mcmcOut + ".csv";
    string betfp = opt.mcmcOut + ".bet";
    string cpnfp = opt.mcmcOut + ".cpn";
    string acufp = opt.mcmcOut + ".acu";
    string rngfp = opt.mcmcOut + ".rng." + std::to_string(rank);
    string mrkfp = opt.mcmcOut + ".mrk." + std::to_string(rank);
    string xivfp = opt.mcmcOut + ".xiv." + std::to_string(rank);
    string epsfp = opt.mcmcOut + ".eps." + std::to_string(rank);
    string gamfp = opt.mcmcOut + ".gam." + std::to_string(rank);

 
    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);

//Deleted the restart part
        
    //    dist.reset_rng((uint)(opt.seed + rank*1000));

    // Build a list of the files to tar
    // --------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    ofstream listFile;
    listFile.open(lstfp);
    listFile << outfp << "\n";
    listFile << betfp << "\n";
    listFile << cpnfp << "\n";
    listFile << acufp << "\n";
    for (int i=0; i<nranks; i++) {
        listFile << opt.mcmcOut + ".rng." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".mrk." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".xiv." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".eps." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".gam." + std::to_string(i) << "\n";
    }
    listFile.close();
    MPI_Barrier(MPI_COMM_WORLD);


    // Delete old files (fp appended with "_rs" in case of restart, so that
    // original files are kept untouched) and create new ones
    // --------------------------------------------------------------------
    if (rank == 0) {
        MPI_File_delete(outfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(acufp.c_str(), MPI_INFO_NULL);
    }
    MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(mrkfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(xivfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(gamfp.c_str(), MPI_INFO_NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, acufp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &acufh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_SELF,  epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  mrkfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &mrkfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  xivfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xivfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  gamfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &gamfh), __LINE__, __FILE__);


    // First element of the .bet, .cpn and .acu files is the
    // total number of processed markers
    // -----------------------------------------------------
    betoff = size_t(0);
    if (rank == 0) {
        check_mpi(MPI_File_write_at(betfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(acufh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();
    
    double tl = -mysecond();

    // Read the data (from sparse representation by default)
    // -----------------------------------------------------
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    N1S = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);
    dalloc += 6.0 * double(Mtot) * sizeof(size_t) / 1E9;

    uint *I1, *I2, *IM;
    size_t totalBytes = 0;

    if (opt.readFromBedFile) {
        data.load_data_from_bed_file(opt.bedFile, Ntot, M, rank, MrankS[rank],
                                     dalloc,
                                     N1S, N1L, I1,
                                     N2S, N2L, I2,
                                     NMS, NML, IM);
    } else {
        string sparseOut = mpi_get_sparse_output_filebase(rank);
        data.load_data_from_sparse_files(rank, nranks, M, MrankS, MrankL, sparseOut,
                                         dalloc,
                                         N1S, N1L, I1,
                                         N2S, N2L, I2,
                                         NMS, NML, IM,
                                         totalBytes);
    }
	

    MPI_Barrier(MPI_COMM_WORLD);

    tl += mysecond();

    if (rank == 0) {
        printf("INFO   : Total time to load the data: %lu bytes in %.3f seconds => BW = %7.3f GB/s\n", totalBytes, tl, (double)totalBytes * 1E-9 / tl);
        fflush(stdout);
    }


    // Compute statistics (from sparse info)
    // -------------------------------------
    //if (rank == 0) printf("INFO   : start computing statistics on Ntot = %d individuals\n", Ntot);
    double dN   = (double) Ntot;
    double dNm1 = (double)(Ntot - 1);
    double *mave, *mstd;
    mave = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mave, __LINE__, __FILE__);
    mstd = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);
    dalloc += 2 * size_t(M) * sizeof(double) / 1E9;

    double tmp0, tmp1, tmp2;
    for (int i=0; i<M; ++i) {
	// For now use the old way to compute means
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (dN - double(NML[i]));        
//Old method for calculating mean
//        mave[i] = data.means(i);

	tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        //TODO At some point we need to turn sd to 1/sd for speed
	//mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
        mstd[i] = sqrt( (tmp0+tmp1+tmp2)/double(Ntot - 1));

	// Old sd
	//mstd[i] = data.sds(i);
        //printf("marker %6d mean %20.15f, std = %20.15f (%.1f / %.15f)  (%15.10f, %15.10f, %15.10f)\n", i, mave[i], mstd[i], double(Ntot - 1), tmp0+tmp1+tmp2, tmp1, tmp2, tmp0);
    }

        // Reading the Xj*failure sum in sparse format:
        for(int marker=0; marker < Mtot; marker++){
                int temp_sum = 0;
		for(int i = N1S[marker]; i < (N1S[marker] + N1L[marker]) ; i++){
                //for(int i=0; i < oneIndices.size(); i++){
                        temp_sum += used_data_alpha.failure_vector(I1[i]);
                }
                for(int i = N2S[marker]; i < (N2S[marker] + N2L[marker]) ; i++){
	        //for(int i=0; i < twoIndices.size(); i++){
                        temp_sum += 2*used_data_alpha.failure_vector(I2[i]);
                }

                sum_failure(marker) = (temp_sum - mave[marker] * used_data_alpha.failure_vector.array().sum()) / mstd[marker];
        }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Build list of markers    
    // ---------------------
    for (int i=0; i<M; ++i) markerI.push_back(i);
    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();
 
    //double *y, *epsilon, *tmpEps, *previt_eps, *deltaEps, *dEpsSum, *deltaSum;
    double *y, *tmpEps, *deltaEps, *dEpsSum, *deltaSum, *epsilon ,*vi , *tmp_vi, *tmpEps_vi, *tmp_deltaEps;
    const size_t NDB = size_t(Ntot) * sizeof(double);
    y          = (double*)_mm_malloc(NDB, 64);  check_malloc(y,          __LINE__, __FILE__);
    epsilon    = (double*)_mm_malloc(NDB, 64);  check_malloc(epsilon,    __LINE__, __FILE__);
    vi    = (double*)_mm_malloc(NDB, 64);  check_malloc(vi,    __LINE__, __FILE__);

    tmpEps_vi    = (double*)_mm_malloc(NDB, 64);  check_malloc(tmpEps_vi,    __LINE__, __FILE__);
    tmp_vi    = (double*)_mm_malloc(NDB, 64);  check_malloc(tmp_vi,    __LINE__, __FILE__);

    tmpEps     = (double*)_mm_malloc(NDB, 64);  check_malloc(tmpEps,     __LINE__, __FILE__);
    //previt_eps = (double*)malloc(NDB);  check_malloc(previt_eps, __LINE__, __FILE__);
    tmp_deltaEps   = (double*)_mm_malloc(NDB, 64);  check_malloc(tmp_deltaEps,   __LINE__, __FILE__);

    deltaEps   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaEps,   __LINE__, __FILE__);
    dEpsSum    = (double*)_mm_malloc(NDB, 64);  check_malloc(dEpsSum,    __LINE__, __FILE__);
    deltaSum   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaSum,   __LINE__, __FILE__);
    dalloc += NDB * 6 / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);


    set_vector_f64(dEpsSum, 0.0, Ntot);

    // Copy, center and scale phenotype observations
    // In bW we are not scaling and centering phenotypes
    for (int i=0; i<Ntot; ++i) y[i] = data.y(i);
    for (int i=0; i<Ntot; ++i)  epsilon[i] = y[i] - mu;	


    double   sum_beta_squaredNorm;
    double   beta, betaOld, deltaBeta, beta_squaredNorm, p, acum, e_sqn;
    size_t   markoff;
    int      marker, left;

    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;

    // Main iteration loop
    // -------------------
    //bool replay_it = false;
    double tot_sync_ar1  = 0.0;
    double tot_sync_ar2  = 0.0;
    int    tot_nsync_ar1 = 0;
    int    tot_nsync_ar2 = 0;
    int    *glob_info, *tasks_len, *tasks_dis, *stats_len, *stats_dis;
    if (opt.sparseSync) {
        glob_info  = (int*)    _mm_malloc(size_t(nranks * 2) * sizeof(int),    64);  check_malloc(glob_info,  __LINE__, __FILE__);
        tasks_len  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(tasks_len,  __LINE__, __FILE__);
        tasks_dis  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(tasks_dis,  __LINE__, __FILE__);
        stats_len  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(stats_len,  __LINE__, __FILE__);
        stats_dis  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(stats_dis,  __LINE__, __FILE__);
    }

   //Set iteration_start=0
    for (uint iteration=0; iteration<opt.chainLength; iteration++) {
        double start_it = MPI_Wtime();
        double it_sync_ar1  = 0.0;
        double it_sync_ar2  = 0.0;
        int    it_nsync_ar1 = 0;
        int    it_nsync_ar2 = 0;

	/* 1. Intercept (mu) */
 	//Removed sampleMu function on its own 
	int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 4 ;
        int neval;
        double xsamp[0], xcent[10], qcent[10] = {5., 30., 70., 95.};
        double convex = 1.0;
        int dometrop = 0;
        double xprev = 0.0;
        double xinit[4] = {0.95*mu, mu,  1.005*mu, 1.01*mu};     // Initial abscissae
        double *p_xinit = xinit;

        double xl = 2;
        double xr = 5;   //xl and xr and the maximum and minimum values between which we sample
	

	//Update before sampling
	for(int mu_ind=0; mu_ind < Ntot; mu_ind++){
		(used_data.epsilon)[mu_ind] = epsilon[mu_ind] + mu;// we add to epsilon =Y+mu-X*beta
	}
        // Use ARS to sample mu (with density mu_dens, using parameters from used_data)
        err = arms(xinit,ninit,&xl,&xr,mu_dens,&used_data,&convex,
                        npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);

        errorCheck(err); // If there is error, stop the program
        mu = xsamp[0];   // Save the sampled value
        //Update after sampling
        for(int mu_ind=0; mu_ind < Ntot; mu_ind++){
                epsilon[mu_ind] = (used_data.epsilon)[mu_ind] - mu;// we add to epsilon =Y+mu-X*beta
        }

////////// End sampling mu
        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------
        
	// Calculate the vector of exponent of the adjusted residuals
	for(int i=0; i<Ntot; ++i){
		vi[i] = exp(used_data.alpha * epsilon[i] - EuMasc);
	}
	if (opt.shuffleMarkers) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
	}
        m0 = 0.0;
        v.setOnes();

        for (int i=0; i<Ntot; ++i) tmpEps[i] = epsilon[i];

        double cumSumDeltaBetas = 0.0;
        double task_sum_abs_deltabeta = 0.0;
        int    sinceLastSync    = 0;

        // Loop over (shuffled) markers
        // ----------------------------
     	
	// First element for the marginal likelihoods is always is pi_0 *sqrt(pi) for
	marginal_likelihoods(0) = pi_L(0) * sqrtPI;  
	for (int j = 0; j < lmax; j++) {
	    sinceLastSync += 1; 
            if (j < M) {
                marker  = markerI[j];
                beta =  Beta(marker);

/////////////////////////////////////////////////////////
	//Replace the sampleBeta function with the inside of the function        
        double vi_sum = 0.0;
        double vi_1 = 0.0;
        double vi_2 = 0.0;

        //Change the residual vector only if the previous beta was non-zero
        if(Beta(marker) != 0){

                /*for(int i=0; i<Ntot; ++i){
                        tmpEps_vi[i] = epsilon[i] - data.mean_sd_ratio(marker) * Beta(marker);
                 }
		//And adjust even further for specific 1 and 2 allele values
                for(int i=0; i < data.Zones[marker].size(); i++){
                	tmpEps_vi[data.Zones[marker][i]] += Beta(marker)/mstd[marker];
                }
                for(int i=0; i < data.Ztwos[marker].size(); i++){
                        tmpEps_vi[data.Ztwos[marker][i]] += 2*Beta(marker)/mstd[marker];
                } */

		//Calculate the change in epsilon if we remove the previous marker effect (-Beta(marker))
		
		set_vector_f64(tmp_deltaEps, 0.0, Ntot);
		sparse_scaadd(tmp_deltaEps, Beta(marker),
                                      I1, N1S[marker], N1L[marker],
                                      I2, N2S[marker], N2L[marker],
                                      IM, NMS[marker], NML[marker],
                                      mave[marker], 1/mstd[marker] , Ntot);
        	//Create the temporary vector to store the vector without the last Beta(marker)
                sum_vectors_f64(tmpEps_vi, epsilon, tmp_deltaEps,  Ntot);

	        //Also find the transformed residuals
		for(int i=0; i<Ntot; ++i){
                        tmp_vi[i] = exp(used_data.alpha * tmpEps_vi[i] - EuMasc);
                        vi_sum += tmp_vi[i];
		}

	        for (int i = N2S[marker]; i < (N2S[marker] + N2L[marker]) ; i++){
//                for (int i=0; i < data.Zones[marker].size(); i++){
			vi_2 += tmp_vi[I2[i]];
          	      //vi_2 += tmp_vi[data.Ztwos[marker][i]];
        	}
//        	for (int i=0; i < data.Zones[marker].size(); i++){
                for (int i = N1S[marker]; i < (N1S[marker] + N1L[marker]) ; i++){
                        vi_1 += tmp_vi[I1[i]];
                	//vi_1 += tmp_vi[data.Zones[marker][i]];
        	}
        }else{
		// Calculate the sums of vi elements
        	for (int i=0; i < Ntot; i++){
                	vi_sum += vi[i];
        	}
//        	for (int i=0; i < data.Ztwos[marker].size(); i++){
                for (int i = N2S[marker]; i < (N2S[marker] + N2L[marker]) ; i++){
//		       	vi_2 += vi[data.Ztwos[marker][i]];
                        vi_2 += vi[I2[i]];
        	}
//        	for (int i=0; i < data.Zones[marker].size(); i++){
                for (int i = N1S[marker]; i < (N1S[marker] + N1L[marker]) ; i++){
//                        vi_1 += vi[data.Zones[marker][i]];
                        vi_1 += vi[I1[i]];
        	}

	}

        double vi_0 = vi_sum - vi_1 - vi_2;

        /* Calculate the mixture probability */
        double p = dist.unif_rng();  //Generate number from uniform distribution (for sampling from categorical distribution)    
  
	// Calculate the (ratios of) marginal likelihoods
        marginal_likelihood_vec_calc(pi_L, marginal_likelihoods, quad_points, vi_sum, vi_2, vi_1, vi_0, mave[marker],mstd[marker], mave[marker]/mstd[marker]);
        // Calculate the probability that marker is 0
        double acum = marginal_likelihoods(0)/marginal_likelihoods.sum();
        //Loop through the possible mixture classes
        for (int k = 0; k < K; k++) {
                if (p <= acum) {
                        //if zeroth component
                        if (k == 0) {
                                Beta(marker) = 0;
                                v[k] += 1.0;
                                components[marker] = k;
                        }
                        // If is not 0th component then sample using ARS
                        else {
		 		used_data_beta.sum_failure = sum_failure(marker);
				used_data_beta.mean = mave[marker];
        			used_data_beta.sd = mstd[marker];
        			used_data_beta.mean_sd_ratio = mave[marker]/mstd[marker];
                                used_data_beta.used_mixture = k-1;

                                used_data_beta.vi_0 = vi_0;
                                used_data_beta.vi_1 = vi_1;
                                used_data_beta.vi_2 = vi_2;

                                double safe_limit = 2 * sqrt(used_data_beta.sigma_b * used_data_beta.mixture_classes(k-1));

                                // ARS parameters
       				 neval = 0;
			         xsamp[0] = 0;
			         convex = 1.0;
        			 dometrop = 0;
			         xprev = 0.0;
			         xinit[0] = Beta(marker) - safe_limit/10;     // Initial abscissae
			         xinit[1] = Beta(marker);
			         xinit[2] = Beta(marker) + safe_limit/20;
        			 xinit[3] = Beta(marker) + safe_limit/10;
   			        
				// Initial left and right (pseudo) extremes

                                xl = Beta(marker) - safe_limit  ; //Construct the hull around previous beta value
                                xr = Beta(marker) + safe_limit;
                                // Sample using ARS
                                err = arms(xinit,ninit,&xl,&xr,beta_dens,&used_data_beta,&convex,
                                                npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
	                        errorCheck(err);

                                Beta(marker) = xsamp[0];  // Save the new result
 
                                v[k] += 1.0;
                                components[marker] = k;
                        }
                        break;
                } else {
                        if((k+1) == (K-1)){
                                acum = 1; // In the end probability will be 1
                        }else{
                                acum += marginal_likelihoods(k+1)/marginal_likelihoods.sum();
                        }
                }
        }

                betaOld   = beta;
                beta      = Beta(marker);
                deltaBeta = betaOld - beta;
                //printf("deltaBeta = %15.10f\n", deltaBeta);

                // Compute delta epsilon
                if (deltaBeta != 0.0) {
                    //printf("it %d, task %3d, marker %5d has non-zero deltaBeta = %15.10f (%15.10f, %15.10f) => %15.10f) 1,2,M: %lu, %lu, %lu\n", iteration, rank, marker, deltaBeta, mave[marker], mstd[marker],  deltaBeta * mstd[marker], N1L[marker], N2L[marker], NML[marker]);

                    if (opt.sparseSync && nranks > 1) {

                        mark2sync.push_back(marker);
                        dbet2sync.push_back(deltaBeta);

                    } else {
                        sparse_scaadd(deltaEps, deltaBeta, 
                                      I1, N1S[marker], N1L[marker],
                                      I2, N2S[marker], N2L[marker],
                                      IM, NMS[marker], NML[marker],
                                      mave[marker], 1/mstd[marker] , Ntot); //Use here 1/sd
                        
                        // Update local sum of delta epsilon
                        sum_vectors_f64(dEpsSum, deltaEps, Ntot);
                    }
                }	
}

                        

            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else {
                //cout << "rank " << rank << " with M=" << M << " waiting for " << lmax << endl;
                deltaBeta = 0.0;
                
                set_vector_f64(deltaEps, 0.0, Ntot);
            }

            task_sum_abs_deltabeta += fabs(deltaBeta);

            // Check whether we have a non-zero beta somewhere
            //if (nranks > 1 && (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1)) {
            if (nranks > 1 && (sinceLastSync >= opt.syncRate || j == lmax-1)) {    
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

 //         if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1) && cumSumDeltaBetas != 0.0) {
            if ( cumSumDeltaBetas != 0.0 && (sinceLastSync >= opt.syncRate || j == lmax-1)) {

                // Update local copy of epsilon
                //MPI_Barrier(MPI_COMM_WORLD);

                if (nranks > 1) {
                    double tb = MPI_Wtime();
                    
                    // Sparse synchronization
                    // ----------------------
                    if (opt.sparseSync) {
                            
                        uint task_m2s = (uint) mark2sync.size();
                        
                        // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
                        double* task_stat = (double*) _mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
                        check_malloc(task_stat, __LINE__, __FILE__);
                        
                        // Compute total number of elements to be sent by each task
                        uint task_size = 0;
                        for (int i=0; i<task_m2s; i++) {
                            task_size += (N1L[ mark2sync[i] ] + N2L[ mark2sync[i] ] + NML[ mark2sync[i] ] + 3);
                            task_stat[2 * i + 0] = mave[ mark2sync[i] ];
                            task_stat[2 * i + 1] = mstd[ mark2sync[i] ] * dbet2sync[i]; //CHANGE mstd later!
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
                            stats_len[i]  = glob_info[2 * i] * 2;
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
                            task_dat[loc] = N1L[ mark2sync[i] ];                 loc += 1;
                            task_dat[loc] = N2L[ mark2sync[i] ];                 loc += 1;
                            task_dat[loc] = NML[ mark2sync[i] ];                 loc += 1;
                            for (uint ii = 0; ii < N1L[ mark2sync[i] ]; ii++) {
                                task_dat[loc] = I1[ N1S[ mark2sync[i] ] + ii ];  loc += 1;
                            }
                            for (uint ii = 0; ii < N2L[ mark2sync[i] ]; ii++) {
                                task_dat[loc] = I2[ N2S[ mark2sync[i] ] + ii ];  loc += 1;
                            }
                            for (uint ii = 0; ii < NML[ mark2sync[i] ]; ii++) {
                                task_dat[loc] = IM[ NMS[ mark2sync[i] ] + ii ];  loc += 1;
                            }
                        }                        
                        assert(loc == task_size);
                            
                        // Allocate receive buffer for all the data
                        uint* glob_dat = (uint*) _mm_malloc(size_t(glob_size) * sizeof(uint), 64);
                        check_malloc(glob_dat, __LINE__, __FILE__);
                        
                        check_mpi(MPI_Allgatherv(task_dat, task_size, MPI_UNSIGNED,
                                                 glob_dat, tasks_len, tasks_dis, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);
                        _mm_free(task_dat);
                        
                        double* glob_stats = (double*) _mm_malloc(size_t(glob_size * 2) * sizeof(double), 64);
                        check_malloc(glob_stats, __LINE__, __FILE__);
                        
                        check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                                 glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD), __LINE__, __FILE__);                        
                        _mm_free(task_stat);
                        
                         
                        // Compute global delta epsilon deltaSum
                        size_t loci = 0;
                        for (int i=0; i<glob_m2s ; i++) {
                            
                            //printf("m2s %d/%d (loci = %d): %d, %d, %d\n", i, glob_m2s, loci, glob_dat[loci], glob_dat[loci + 1], glob_dat[loci + 2]);
                            
                            double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                            //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1]);
                            
                            // Set all to 0 contribution
                            if (i == 0) {
                                set_vector_f64(deltaSum, lambda0, Ntot);
                            } else {
                                offset_vector_f64(deltaSum, lambda0, Ntot);
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
                        
                        _mm_free(glob_stats);
                        _mm_free(glob_dat);                        
                        
                        mark2sync.clear();
                        dbet2sync.clear();                            
                        
                    } else {
                        
                        check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                    
                    }
                    
                    sum_vectors_f64(epsilon, tmpEps, deltaSum, Ntot);
                    
                    double te = MPI_Wtime();
                    tot_sync_ar2  += te - tb;
                    it_sync_ar2   += te - tb;
                    tot_nsync_ar2 += 1;
                    it_nsync_ar2  += 1;    

                } else { // case nranks == 1    
                     if(opt.deltaUpdate == true){
			sum_vectors_f64(epsilon, tmpEps, dEpsSum,  Ntot);
 		     }else{	
			for(int i=0; i < Ntot; i++){
		                        epsilon[i] = epsilon[i] -  betaOld * mave[marker]/mstd[marker];
                                        epsilon[i] = epsilon[i] + beta * mave[marker]/mstd[marker];
                                }
                                //And adjust even further for specific 1 and 2 allele values
                              //  for(int i=0; i < data.Zones[marker].size(); i++){
		                for (int i = N1S[marker]; i < (N1S[marker] + N1L[marker]) ; i++){
                                        //epsilon[data.Zones[marker][i]] += betaOld/mstd[marker];
                                        //epsilon[data.Zones[marker][i]] -= beta/mstd[marker];
                                        epsilon[I1[i]] += betaOld/mstd[marker];
					epsilon[I1[i]] += betaOld/mstd[marker];
                                }
                            //    for(int i=0; i < data.Ztwos[marker].size(); i++){
                                for (int i = N2S[marker]; i < (N2S[marker] + N2L[marker]) ; i++){
                                        //epsilon[data.Ztwos[marker][i]] += 2*betaOld/mstd[marker];
					//epsilon[data.Ztwos[marker][i]] -= 2*beta/mstd[marker];
					epsilon[I2[i]] += 2*betaOld/mstd[marker];
                                        epsilon[I2[i]] -= 2*beta/mstd[marker];
                                }
			}
		}
   
		// Do a update currently locally for vi vector
		for(int vi_ind=0; vi_ind < Ntot; vi_ind++){
			vi[vi_ind] = exp(used_data.alpha * epsilon[vi_ind] - EuMasc);
		}
                double end_sync = MPI_Wtime();
                //printf("INFO   : synchronization time = %8.3f ms\n", (end_sync - beg_sync) * 1000.0);
                
                // Store epsilon state at last synchronization
                copy_vector_f64(tmpEps, epsilon, Ntot);
                
                // Reset local sum of delta epsilon
                set_vector_f64(dEpsSum, 0.0, Ntot);
                
                // Reset cumulated sum of delta betas
                cumSumDeltaBetas       = 0.0;
                task_sum_abs_deltabeta = 0.0;
                
                sinceLastSync = 0;
                
            }// else {
             //   sinceLastSync += 1;
                //task_sum_abs_deltabeta += fabs(deltaBeta);
            //}

        } // END PROCESSING OF ALL MARKERS

        //PROFILE
        //continue;

        //printf("rank %d it %d  beta_squaredNorm = %15.10f\n", rank, iteration, beta_squaredNorm);

        //printf("==> after eps sync it %d, rank %d, epsilon[0] = %15.10f %15.10f\n", iteration, rank, epsilon[0], epsilon[Ntot-1]);

        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            check_mpi(MPI_Allreduce(&beta_squaredNorm, &sum_beta_squaredNorm, 1,        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(v.data(),          sum_v.data(),          v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            v                = sum_v;
            beta_squaredNorm = sum_beta_squaredNorm;
        }

        // Update global parameters
        // ------------------------
        m0      = double(Mtot) - v[0];

	// ARS parameters
        neval = 0;
	xsamp[0] = 0;
        convex = 1.0;
        dometrop = 0;
        xprev = 0.0;
        xinit[0] = (used_data.alpha)*0.5;     // Initial abscissae
        xinit[1] =  used_data.alpha;
        xinit[2] = (used_data.alpha)*1.15;
        xinit[3] = (used_data.alpha)*1.5; 
	// double *p_xinit = xinit;

        // Initial left and right (pseudo) extremes
        xl = 0.0;
        xr = 30.0;

        //Give the residual to alpha structure
        //used_data_alpha.epsilon = epsilon;
        for(int alpha_ind=0; alpha_ind < Ntot; alpha_ind++){
                (used_data_alpha.epsilon)[alpha_ind] = epsilon[alpha_ind];
        }

        //Sample using ARS
        err = arms(xinit,ninit,&xl,&xr,alpha_dens,&used_data_alpha,&convex,
                        npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
        errorCheck(err);

        check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

	used_data.alpha = xsamp[0];
        used_data_beta.alpha = xsamp[0];
 
        MPI_Barrier(MPI_COMM_WORLD);

	// 4. Sample sigma_b
	used_data_beta.sigma_b = dist.inv_gamma_rng((double) (used_data.alpha_sigma + 0.5 * (M - v[0]+1)),
	(double)(used_data.beta_sigma + 0.5 * (M - v[0]+1) * Beta.squaredNorm()));
        check_mpi(MPI_Bcast(&(used_data_beta.sigma_b), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);


	//Update the sqrt(2sigmab) variable
	used_data.sqrt_2sigmab = sqrt(2*used_data_beta.sigma_b);
	//Print results
        cout << iteration << ". " << Mtot - v[0] +1 <<"; " <<"; "<< setprecision(17) << mu << "; " <<  used_data.alpha << "; " << used_data_beta.sigma_b << endl;
	
	// 5. Sample prior mixture component probability from Dirichlet distribution
	pi_L = dist.dirichilet_rng(v.array());
	check_mpi(MPI_Bcast(pi_L.data(), pi_L.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);


        double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
  /*      if (rank%10==0) {
            printf("RESULT : it %4d, rank %4d: proc = %9.3f s, sync = %9.3f (%9.3f + %9.3f), n_sync = %8d (%8d + %8d) (%7.3f / %7.3f), sigmaG = %15.10f, sigmaE = %15.10f, betasq = %15.10f, m0 = %10d\n",
                   iteration, rank, end_it-start_it,
                   it_sync_ar1  + it_sync_ar2,  it_sync_ar1,  it_sync_ar2,
                   it_nsync_ar1 + it_nsync_ar2, it_nsync_ar1, it_nsync_ar2,
                   (it_sync_ar1) / double(it_nsync_ar1) * 1000.0,
                   (it_sync_ar2) / double(it_nsync_ar2) * 1000.0,
                   sigmaG, sigmaE, beta_squaredNorm, int(m0));
            fflush(stdout);
        }*/
 
        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+Ntot,((epsilon).squaredNorm()+v0E*s02E)/(v0E+Ntot));
        //printf("sigmaG = %20.15f, sigmaE = %20.15f, e_sqn = %20.15f\n", sigmaG, sigmaE, e_sqn);
        //printf("it %6d, rank %3d: epsilon[0] = %15.10f, y[0] = %15.10f, m0=%10.1f,  sigE=%15.10f,  sigG=%15.10f [%6d / %6d]\n", iteration, rank, epsilon[0], y[0], m0, sigmaE, sigmaG, markerI[0], markerI[M-1]);

        // Write output files
        // ------------------
        if (iteration%opt.thin == 0) {
            left = snprintf(buff, LENBUF, "%5d, %4d, %20.15f, %20.15f, %20.15f, %20.15f, %7d, %2d", iteration, rank, mu, used_data.alpha, used_data_beta.sigma_b ,  used_data_beta.sigma_b/(PI2*6*used_data.alpha*used_data.alpha) , int(m0), K);
            assert(left > 0);

            for (int ii=0; ii < K; ++ii) {
                left = snprintf(&buff[strlen(buff)], LENBUF-strlen(buff), ", %20.15f", pi_L(ii));
                assert(left > 0);
            }
            left = snprintf(&buff[strlen(buff)], LENBUF-strlen(buff), "\n");
            assert(left > 0);

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
    //        check_mpi(MPI_File_write_at_all(acufh, betoff, Acum.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            cpnoff = sizeof(uint) + sizeof(uint)
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int))
                + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh, cpnoff, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            //if (iteration == 0) {
            //    printf("rank %d dumping bet: %15.10f %15.10f\n", rank, Beta[0], Beta[MrankL[rank]-1]);
            //    printf("rank %d dumping cpn: %d %d\n", rank, components[0], components[MrankL[rank]-1]);
            //}

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

        // Dump the epsilon vector and the marker indexing one
        // Note: single line overwritten at each saving iteration
        // .eps format: uint, uint, double[0, N-1] (it, Ntot, [eps])
        // .mrk format: uint, uint, int[0, M-1]    (it, M,    <mrk>)
        // ------------------------------------------------------
        if (iteration > 0 && iteration%opt.save == 0) {

            // Each task writes its own rng file
            dist.write_rng_state_to_file(rngfp);
            epsoff  = size_t(0);
            check_mpi(MPI_File_write_at(epsfh, epsoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, epsoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            epsoff += sizeof(uint);
 
           check_mpi(MPI_File_write_at(epsfh, epsoff, &Ntot,         1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, epsoff, &M,            1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            epsoff = sizeof(uint) + sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh, epsoff, epsilon,        Ntot,           MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, epsoff, markerI.data(), markerI.size(), MPI_INT,    &status), __LINE__, __FILE__);


            //if (iteration == 0) {
            //    printf("rank %d dumping eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot-1]);
            //}
            //EO: to remove once MPI version fully validated; use the check_epsilon utility to retrieve
            //    the corresponding values from the .eps file
            //    Print only first and last value handled by each task
            //printf("%4d/%4d epsilon[%5d] = %15.10f, epsilon[%5d] = %15.10f\n", iteration, rank, IrankS[rank], epsilon[IrankS[rank]], IrankS[rank]+IrankL[rank]-1, epsilon[IrankS[rank]+IrankL[rank]-1]);

#if 1
            //EO system call to create a tarball of the dump
            //TODO: quite rough, make it more selective...
            //----------------------------------------------
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                time_t now = time(0);
                tm *   ltm = localtime(&now);
                int    n   = 0;
                char targz[LENBUF];

                n=sprintf(targz, "dump_%s_%05d__%4d-%02d-%02d_%02d-%02d-%02d.tgz",
                          opt.mcmcOutNam.c_str(), iteration,
                          1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                          ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
                assert(n > 0);

                printf("INFO   : will create tarball %s in %s with file listed in %s.\n",
                       targz, opt.mcmcOutDir.c_str(), lstfp.c_str());

                //std::system(("ls " + opt.mcmcOut + ".*").c_str());
                string cmd = "tar -czf " + opt.mcmcOutDir + "/tarballs/" + targz + " -T " + lstfp;

                std::system(cmd.c_str());

            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        //double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Close output files
    check_mpi(MPI_File_close(&outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&acufh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&mrkfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xivfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&gamfh), __LINE__, __FILE__);


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
    _mm_free(N1S);
    _mm_free(N1L);
    _mm_free(I1);
    _mm_free(N2S); 
    _mm_free(N2L);
    _mm_free(I2);
    _mm_free(NMS);
    _mm_free(NML);
    _mm_free(IM);
        
    if (opt.sparseSync) {
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


// Get directory and basename of bed file (passed with no extension via command line)
// ----------------------------------------------------------------------------------
/*
string BayesW::mpi_get_sparse_output_filebase(const int rank) {

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



uint BayesW::set_Ntot(const int rank) {

    uint Ntot = opt.numberIndividuals; //data.numInds;

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

uint BayesW::set_Mtot(const int rank) {

    uint Mtot = opt.numberMarkers;     //data.numSnps;

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


#endif
*/

//  ORIGINAL (SEQUENTIAL) VERSION
/*
VectorXd BayesW::getSnpData(unsigned int marker) const
{
    if (!usePreprocessedData) {
        //read column from RAM loaded genotype matrix.
        return data.Z.col(marker);//.cast<double>();
    } else {
        //read column from preprocessed and memory mapped genotype matrix file.
        return data.mappedZ.col(marker).cast<double>();
    }
}

void BayesW::printDebugInfo() const
{
    //const unsigned int N(data.numInds);
    // cout << "x mean " << Cx.mean() << "\n";
    //   cout << "x sd " << sqrt(Cx.squaredNorm() / (double(N - 1))) << "\n";
}
*/
