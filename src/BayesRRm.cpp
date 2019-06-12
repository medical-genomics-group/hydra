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

void BayesRRm::init(int K, unsigned int markerCount, unsigned int individualCount)
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
    y_tilde = VectorXd(individualCount);    // variable containing the adjusted residuals to exclude the effects of a given marker
    epsilon = VectorXd(individualCount);    // variable containing the residuals

    //phenotype vector
    y = VectorXd(individualCount);

    //SNP column vector
    Cx = VectorXd(individualCount);

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
    y = (data.y.cast<double>().array() - data.y.cast<double>().mean());
    y /= sqrt(y.squaredNorm() / (double(individualCount - 1)));
    //printf(" >>>> ysqn = %15.10f\n", y.squaredNorm());

    //initialize epsilon vector as the phenotype vector
    epsilon    = (y).array() - mu;
    sigmaE     = epsilon.squaredNorm() / individualCount * 0.5;
    betasqn    = 0.0;
    epsilonsum = 0.0;
    ytildesum  = 0.0;
    //gamma      = VectorXd(data.fixedEffectsCount);
    gamma      = VectorXd(data.numFixedEffects);
    gamma.setZero();
    X = data.X; //fixed effects matrix
}


#ifdef USE_MPI


// Check MPI call returned value. If error print message and call MPI_Abort()
// --------------------------------------------------------------------------
inline void check_mpi(const int error, const int linenumber, const char* filename) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "MPI error %d at line %d of file %s\n", error, linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Check malloc in MPI context
// ---------------------------
inline void check_malloc(const void* ptr, const int linenumber, const char* filename) {
    if (ptr == NULL) {
        fprintf(stderr, "Fatal: malloc failed on line %d of %s\n", linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Check for integer overflow
// --------------------------
inline void check_int_overflow(const int n, const int linenumber, const char* filename) {
    if (n > pow(2,(sizeof(int)*8)-1)) {
        fprintf(stderr, "Fatal: integer overflow detected on line %d of %s\n", linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


inline void scaadd(double* __restrict__ vout, const double* __restrict__ vin1, const double* __restrict__ vin2, const double dMULT, const int N) {

    if (dMULT == 0.0) {
        for (int i=0; i<N; i++) {
            vout[i] = vin1[i];
        }
    } else {
        for (int i=0; i<N; i++) {
            vout[i] = vin1[i] + dMULT * vin2[i];
        }
    }

}

/*
inline void sparse_scaadd(double*       __restrict__ vout,
                          const double* __restrict__ vin1,
                          const double  dMULT,
                          const size_t* __restrict__ I1, const size_t N1S, const size_t N1L,
                          const size_t* __restrict__ I2, const size_t N2S, const size_t N2L,
                          const size_t* __restrict__ IM, const size_t NMS, const size_t NML,
                          const double  mu,
                          const double  sig_inv,
                          const int     N) {

    if (dMULT == 0.0) {
        cout << "dmult is 0" << endl;
        for (int i=0; i<N; i++)
            vout[i] = vin1[i];

        for (size_t i=N1S; i<N1S+N1L; ++i)
            vout[I1[i]] = vin1[I1[i]];

        for (size_t i=N2S; i<N2S+N2L; ++i)
            vout[I2[i]] = vin1[I2[i]];
    } else {
        cout << "dmult is NOT 0" << endl;
        double aux = mu * sig_inv * dMULT;
        for (int i=0; i<N; i++)
            vout[i] = vin1[i] - aux;

        for (size_t i=NMS; i<NMS+NML; ++i)
            vout[IM[i]] = vin1[IM[i]];

        aux = dMULT * (1.0 - mu) * sig_inv;
        for (size_t i=N1S; i<N1S+N1L; ++i)
            vout[I1[i]] = vin1[I1[i]] + aux;

        aux = dMULT * (2.0 - mu) * sig_inv;
        for (size_t i=N2S; i<N2S+N2L; ++i)
            vout[I2[i]] = vin1[I2[i]] + aux;
    }
}
*/

inline void sparse_scaadd(double*       __restrict__ vout,
                          const double  dMULT,
                          const size_t* __restrict__ I1, const size_t N1S, const size_t N1L,
                          const size_t* __restrict__ I2, const size_t N2S, const size_t N2L,
                          const size_t* __restrict__ IM, const size_t NMS, const size_t NML,
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
                             const size_t* __restrict__ I1, const size_t N1S, const size_t N1L,
                             const size_t* __restrict__ I2, const size_t N2S, const size_t N2L,
                             const size_t* __restrict__ IM, const size_t NMS, const size_t NML,
                             const double  mu, const double  sig_inv, const int     N) {

    double dp = 0.0, syt = 0.0;
    
    for (size_t i=N1S; i<N1S+N1L; ++i)
        dp += vin1[I1[i]];

    for (size_t i=N2S; i<N2S+N2L; ++i)
        dp += 2.0 * vin1[I2[i]];

    dp *= sig_inv;

    for (int i=0; i<N; i++)
        syt += vin1[i];

    //EO: ajust for missing values
    for (size_t i=NMS; i<NMS+NML; ++i)
        syt -= vin1[IM[i]];

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


//EO: This method writes sparse data files out of a BED file
//    Note: will always convert the whole file
//    A two-step process due to RAM limitation for very large BED files:
//      1) Compute the number of ones and twos to be written by each task to compute
//         rank-wise offsets for writing the si1 and si2 files
//      2) Write files with global indexing
// ---------------------------------------------------------------------------------
void BayesRRm::write_sparse_data_files(const uint bpr) { //CHECK_ALLOC

    int rank, nranks, result;

    //MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File   bedfh, si1fh, sl1fh, ss1fh, si2fh, sl2fh, ss2fh, simfh, slmfh, ssmfh;
    MPI_Offset offset;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    if (rank == 0)
        printf("Will generate sparse data files out of %d ranks and %d blocks per rank.\n", nranks, bpr);

    // Get dimensions of the dataset and define blocks
    // -----------------------------------------------
    const unsigned int N    = data.numInds;
    const unsigned int Mtot = data.numSnps;
    if (rank == 0)
        printf("Full dataset includes %d markers and %d individuals.\n", Mtot, N);

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
    if (rank==0)
        printf("snpLenByt = %zu bytes.\n", snpLenByt);

    // Get bed file directory and basename
    std::string sparseOut = mpi_get_sparse_output_filebase();
    if (rank == 0)
        printf("INFO   : will write sparse output files as: %s.{ss1, sl1, si1, ss2, sl2, si2}\n", sparseOut.c_str());

    // Open bed file for reading
    std::string bedfp = opt.bedFile;
    bedfp += ".bed";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh), __LINE__, __FILE__);

    // Create sparse output files
    // --------------------------
    const std::string si1 = sparseOut + ".si1";
    const std::string sl1 = sparseOut + ".sl1";
    const std::string ss1 = sparseOut + ".ss1";
    const std::string si2 = sparseOut + ".si2";
    const std::string sl2 = sparseOut + ".sl2";
    const std::string ss2 = sparseOut + ".ss2";
    const std::string sim = sparseOut + ".sim";
    const std::string slm = sparseOut + ".slm";
    const std::string ssm = sparseOut + ".ssm";

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
    
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sim.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, slm.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ssm.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ssmfh), __LINE__, __FILE__);


    // STEP 1: compute rank-wise N1 and N2 (rN1 and rN2)
    // -------------------------------------------------
    size_t rN1 = 0, rN2 = 0, rNM = 0;
    size_t N1  = 0, N2  = 0, NM  = 0;

    for (int i=0; i<bpr; ++i) {

        uint globi = rank*bpr + i;
        int  MLi   = MrankL[globi];
        int  MSi   = MrankS[globi];

        // Alloc memory for raw BED data
        const size_t rawdata_n = size_t(MLi) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) malloc(rawdata_n);  check_malloc(rawdata, __LINE__, __FILE__);
        
        // Print information
        printf("rank %4d / block %4d will handle a block of %6d markers starting at %8d, raw = %7.3f GB\n", rank, i, MLi, MSi, rawdata_n * 1E-9);
        
        // Compute the offset of the section to read from the BED file
        offset = size_t(3) + size_t(MSi) * size_t(snpLenByt) * sizeof(char);
        
        // As it does not matter here, limit number of elements to int capacity
        check_int_overflow(rawdata_n, __LINE__, __FILE__);

        check_mpi(MPI_File_read_at(bedfh, offset, rawdata, rawdata_n, MPI_CHAR, &status), __LINE__, __FILE__);

        data.sparse_data_get_sizes(rawdata, MLi, snpLenByt, &N1, &N2, &NM);

        // Check for integer overflow
        check_int_overflow(N1, __LINE__, __FILE__);
        check_int_overflow(N2, __LINE__, __FILE__);
        check_int_overflow(NM, __LINE__, __FILE__);

        rN1 += N1;
        rN2 += N2;
        rNM += NM;

        free(rawdata);
    }

    // Gather offsets
    size_t *AllN1 = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllN1, __LINE__, __FILE__);
    size_t *AllN2 = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllN2, __LINE__, __FILE__);
    size_t *AllNM = (size_t*) malloc(nranks * sizeof(size_t));  check_malloc(AllNM, __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rN1, 1, MPI_UNSIGNED_LONG_LONG, AllN1, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rN2, 1, MPI_UNSIGNED_LONG_LONG, AllN2, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rNM, 1, MPI_UNSIGNED_LONG_LONG, AllNM, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);


    // STEP 2: write sparse structure files
    // ------------------------------------    
    size_t tN1 = 0, tN2 = 0, tNM = 0;

    for (int i=0; i<bpr; ++i) {

        uint globi = rank*bpr + i;
        int  MLi   = MrankL[globi];
        int  MSi   = MrankS[globi];

        // Alloc memory for raw BED data
        const size_t rawdata_n = size_t(MLi) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) malloc(rawdata_n);  check_malloc(rawdata, __LINE__, __FILE__);
        
        // Compute the offset of the section to read from the BED file
        offset = size_t(3) + size_t(MSi) * size_t(snpLenByt) * sizeof(char);

        // As it does not matter here, limit number of elements to int capacity
        check_int_overflow(rawdata_n, __LINE__, __FILE__);
        check_mpi(MPI_File_read_at(bedfh, offset, rawdata, rawdata_n, MPI_CHAR, &status), __LINE__, __FILE__);

        // Alloc memory for sparse representation
        size_t *N1S, *N1L, *N2S, *N2L, *NMS, *NML;
        N1S = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N1S, __LINE__, __FILE__);
        N1L = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N1L, __LINE__, __FILE__);
        N2S = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N2S, __LINE__, __FILE__);
        N2L = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(N2L, __LINE__, __FILE__);
        NMS = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(NMS, __LINE__, __FILE__);
        NML = (size_t*) malloc(size_t(MLi) * sizeof(size_t));  check_malloc(NML, __LINE__, __FILE__);

        size_t N1, N2, NM;
        data.sparse_data_get_sizes(rawdata, MLi, snpLenByt, &N1, &N2, &NM);

        // Check overflow on number of elements (limit my int in the MPI call)
        check_int_overflow(N1, __LINE__, __FILE__);
        check_int_overflow(N2, __LINE__, __FILE__);
        check_int_overflow(NM, __LINE__, __FILE__);

        // Alloc and build sparse structure
        size_t *I1, *I2, *IM;
        I1 = (size_t*) malloc(N1 * sizeof(size_t));  check_malloc(I1, __LINE__, __FILE__);
        I2 = (size_t*) malloc(N2 * sizeof(size_t));  check_malloc(I2, __LINE__, __FILE__);
        IM = (size_t*) malloc(NM * sizeof(size_t));  check_malloc(IM, __LINE__, __FILE__);

        data.sparse_data_fill_indices(rawdata, MLi, snpLenByt, N1S, N1L, I1,  N2S, N2L, I2,  NMS, NML, IM);
        
        size_t N1Off = 0, N2Off = 0, NMOff = 0;
        for (int ii=0; ii<rank; ++ii) {
            N1Off += AllN1[ii];
            N2Off += AllN2[ii];
            NMOff += AllNM[ii];
        }
        N1Off += tN1;
        N2Off += tN2;
        NMOff += tNM;

        //printf("rank/prg %4d/%4d as N1 = %15lu and AllN1 = %15lu; Will dump at N1Off %15lu\n", rank, i, N1, AllN1[rank], N1Off);
        //printf("rank/prg %4d/%4d as N2 = %15lu and AllN2 = %15lu; Will dump at N2Off %15lu\n", rank, i, N2, AllN2[rank], N2Off);
        //printf("rank/prg %4d/%4d as NM = %15lu and AllNM = %15lu; Will dump at NMOff %15lu\n", rank, i, NM, AllNM[rank], NMOff);

        // ss1,2,m files must contain absolute start indices
        for (int ii=0; ii<MLi; ++ii) {
            N1S[ii] += N1Off;
            N2S[ii] += N2Off;
            NMS[ii] += NMOff;
        }

        // Sparse Index Ones file (si1)
        offset = N1Off * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(si1fh, offset, I1,   N1, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        // Sparse Length Ones file (sl1)
        offset = size_t(MSi) * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(sl1fh, offset, N1L, MLi, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        // Sparse Start Ones file (ss1)
        offset = size_t(MSi) * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(ss1fh, offset, N1S, MLi, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        
        // Sparse Index Twos file (si2)
        offset = N2Off * sizeof(size_t) ;
        check_mpi(MPI_File_write_at_all(si2fh, offset, I2,   N2, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        // Sparse Length Twos file (sl2)
        offset = size_t(MSi) * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(sl2fh, offset, N2L, MLi, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        // Sparse Start Ones file (ss2)
        offset = size_t(MSi) * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(ss2fh, offset, N2S, MLi, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);

        // Sparse Index Missing file (sim)
        offset = NMOff * sizeof(size_t) ;
        check_mpi(MPI_File_write_at_all(simfh, offset, IM,   NM, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        // Sparse Length Missing file (slm)
        offset = size_t(MSi) * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(slmfh, offset, NML, MLi, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
        // Sparse Start Missing file (ssm)
        offset = size_t(MSi) * sizeof(size_t);
        check_mpi(MPI_File_write_at_all(ssmfh, offset, NMS, MLi, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);


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


    // Close bed and sparse files
    check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__); 
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
    if (rank == 0)
        std::cout << "INFO   : time to convert the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Finalize the MPI environment
    //MPI_Finalize();
}


size_t get_file_size(const std::string& filename) {
    struct stat st;
    if(stat(filename.c_str(), &st) != 0) { return 0; }
    return st.st_size;   
}


void BayesRRm::read_sparse_data_files(size_t*& N1S, size_t*& N1L, size_t*& I1, 
                                      size_t*& N2S, size_t*& N2L, size_t*& I2,
                                      size_t*& NMS, size_t*& NML, size_t*& IM,
                                      int* MrankS, int* MrankL, const int rank) {
    
    MPI_Offset offset;
    MPI_Status status;

    // Number of markers in block handled by task
    const uint M = MrankL[rank];

    // Alloc memory for sparse representation
    N1S = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(NML, __LINE__, __FILE__);

    // Read sparse data files
    // Each task is in charge of M markers starting from MrankS[rank]
    // So first we read si1 to get where to read in 
    MPI_File ss1fh, sl1fh, si1fh;
    MPI_File ss2fh, sl2fh, si2fh;
    MPI_File ssmfh, slmfh, simfh;

    // Get bed file directory and basename
    std::string sparseOut = mpi_get_sparse_output_filebase();
    if (rank == 0)
        printf("INFO   : will read from sparse files with basename: %s\n", sparseOut.c_str());

    const std::string si1 = sparseOut + ".si1";
    const std::string sl1 = sparseOut + ".sl1";
    const std::string ss1 = sparseOut + ".ss1";
    const std::string si2 = sparseOut + ".si2";
    const std::string sl2 = sparseOut + ".sl2";
    const std::string ss2 = sparseOut + ".ss2";
    const std::string sim = sparseOut + ".sim";
    const std::string slm = sparseOut + ".slm";
    const std::string ssm = sparseOut + ".ssm";

    check_mpi(MPI_File_open(MPI_COMM_WORLD, si1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sim.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, slm.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ssm.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ssmfh), __LINE__, __FILE__);

    // Compute the lengths of ones and twos vectors for all markers in the block
    offset =  MrankS[rank] * sizeof(size_t);
    check_mpi(MPI_File_read_at_all(sl1fh, offset, N1L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ss1fh, offset, N1S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(sl2fh, offset, N2L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ss2fh, offset, N2S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slmfh, offset, NML, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ssmfh, offset, NMS, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);

    // Absolute offsets in 0s, 1s, and 2s
    const size_t n1soff = N1S[0];
    const size_t n2soff = N2S[0];
    const size_t nmsoff = NMS[0];

    size_t N1 = N1S[M-1] + N1L[M-1] - n1soff;
    size_t N2 = N2S[M-1] + N2L[M-1] - n2soff;
    size_t NM = NMS[M-1] + NML[M-1] - nmsoff;
    //printf("rank %d: N1 = %20lu, N2 = %20lu, NM = %20lu\n", rank, N1, N2, NM);

    // Check for integer overflow
    check_int_overflow(N1, __LINE__, __FILE__);
    check_int_overflow(N2, __LINE__, __FILE__);
    check_int_overflow(NM, __LINE__, __FILE__);

    // Alloc and build sparse structure
    I1 = (size_t*) malloc(N1 * sizeof(size_t));  check_malloc(I1, __LINE__, __FILE__);
    I2 = (size_t*) malloc(N2 * sizeof(size_t));  check_malloc(I2, __LINE__, __FILE__);
    IM = (size_t*) malloc(NM * sizeof(size_t));  check_malloc(IM, __LINE__, __FILE__);

    // Make starts relative to start of block in each task
    for (int i=0; i<M; ++i) { N1S[i] -= n1soff; }
    for (int i=0; i<M; ++i) { N2S[i] -= n2soff; }
    for (int i=0; i<M; ++i) { NMS[i] -= nmsoff; }

    // Read the indices of 1s and 2s
    offset =  n1soff * sizeof(size_t);
    check_mpi(MPI_File_read_at_all(si1fh, offset, I1, N1, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    offset =  n2soff * sizeof(size_t);
    check_mpi(MPI_File_read_at_all(si2fh, offset, I2, N2, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    offset =  nmsoff * sizeof(size_t);
    check_mpi(MPI_File_read_at_all(simfh, offset, IM, NM, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);

    // Close bed and sparse files
    check_mpi(MPI_File_close(&si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ssmfh), __LINE__, __FILE__);
}


void mpi_assign_blocks_to_tasks(int* MrankS, int* MrankL, const uint numBlocks, const vector<int> blocksStarts, const vector<int> blocksEnds, const uint Mtot) {

    int rank, nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    const size_t LENBUF=200;

    char buff[LENBUF]; 
    int  nranks, rank, name_len, result;
    double dalloc = 0.0;

    //MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File   bedfh, outfh, betfh, epsfh, cpnfh;
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
    const unsigned int N      = data.numInds;
    unsigned int       Mtot   = data.numSnps;

    if (rank == 0) printf("INFO   : Full dataset includes %d markers and %d individuals.\n", Mtot, N);
    
    // Block marker definition has precedence over requested number of markers
    if (opt.markerBlocksFile != "" && opt.numberMarkers > 0) {
        opt.numberMarkers = 0;
        if (rank == 0) printf("WARNING: --number-markers option ignored, a marker block definition file was passed!\n");
    }        
    
    if (opt.numberMarkers > 0 && opt.numberMarkers < Mtot) {
        Mtot = opt.numberMarkers;
        if (rank == 0) printf("Option passed to process only %d markers!\n", Mtot);
    }

    // Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // -----------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    mpi_define_blocks_of_markers(N, IrankS, IrankL, nranks);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks];
    mpi_assign_blocks_to_tasks(MrankS, MrankL, data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot);


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
    double              dNm1   = (double)(N - 1);
    double              dN     = (double) N;
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
    VectorXd            beta(M);
    VectorXd            gamma;

    dalloc += M * sizeof(double) / 1E9; // for components
    dalloc += M * sizeof(double) / 1E9; // for beta

    //gamma = VectorXd(data.fixedEffectsCount);
    gamma = VectorXd(data.numFixedEffects);
    gamma.setZero();
    X = data.X; //fixed effects matrix

    //VectorXd            sample(2*M+4+N); // varible containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance


    // Length of a column in bytes
    const size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;
    if (rank==0) printf("INFO   : marker length in bytes (snpLenByt) = %zu bytes.\n", snpLenByt);

    // Opne BED and output files
    std::string bedfp = opt.bedFile;
    std::string outfp = opt.mcmcOut + ".csv";
    std::string betfp = opt.mcmcOut + ".bet";
    std::string epsfp = opt.mcmcOut + ".eps";
    std::string cpnfp = opt.mcmcOut + ".cpn";

    // Delete old files
    if (rank == 0) {
        MPI_File_delete(outfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(), MPI_INFO_NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);

    // First element of the .bet file is the total number of processed markers
    // Idem for .cpn file
    betoff = size_t(0);
    if (rank == 0) {
        check_mpi(MPI_File_write_at(betfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    // Preprocess the data
    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    // Alloc memory for sparse representation
    size_t *N1S, *N1L, *I1,  *N2S, *N2L, *I2, *NMS, *NML, *IM;

    if (opt.readFromBedFile) {

        bedfp += ".bed";
        check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__);

        // Alloc memory for raw BED data
        // -----------------------------
        const size_t rawdata_n = size_t(M) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) malloc(rawdata_n);  check_malloc(rawdata, __LINE__, __FILE__);
        dalloc += rawdata_n / 1E9;
        //printf("rank %d allocation %zu bytes (%.3f GB) for the raw data.\n",           rank, rawdata_n, double(rawdata_n/1E9));

        //EO: leftover from previous implementation, but keep it in for now
        //EO  can be useful for checks
        //-----------------------------------------------------------------
        //const size_t ppdata_n  = size_t(M) * size_t(data.numInds) * sizeof(double);
        //double* ppdata    = (double*)malloc(ppdata_n);  if (ppdata  == NULL) { printf("malloc ppdata failed.\n");  exit (1); }    
        //printf("rank %d allocation %zu bytes (%.3f GB) for the pre-processed data.\n", rank, ppdata_n,  double(ppdata_n/1E9));

        // Compute the offset of the section to read from the BED file
        // -----------------------------------------------------------
        offset = size_t(3) + size_t(MrankS[rank]) * size_t(snpLenByt) * sizeof(char);

        // Read the BED file
        // -----------------
        MPI_Barrier(MPI_COMM_WORLD);
        const auto st1 = std::chrono::high_resolution_clock::now();

        // Check how many calls are needed (limited by the int type of the number of elements to read!)
        uint nmpiread = 1;
        if (rawdata_n >= pow(2,(sizeof(int)*8)-1)) {   
            printf("MPI_file_read_at capacity exceeded. Asking to read %zu elements vs max %12.0f\n", 
                   rawdata_n, pow(2,(sizeof(int)*8)-1));
            nmpiread = ceil(double(rawdata_n) / double(pow(2,(sizeof(int)*8)-1)));       
        }
        assert(nmpiread >= 1);
        //cout << "Will need " << nmpiread << " calls to MPI_file_read_at to load all the data." << endl;

        if (nmpiread == 1) {
            check_mpi(result = MPI_File_read_at(bedfh, offset, rawdata, rawdata_n, MPI_CHAR, &status), __LINE__, __FILE__);
        } else {
            cout << "rawdata_n = " << rawdata_n << endl;
            size_t chunk    = size_t(double(rawdata_n)/double(nmpiread));
            size_t checksum = 0;
            for (int i=0; i<nmpiread; ++i) {
                size_t chunk_ = chunk;        
                if (i==nmpiread-1)
                    chunk_ = rawdata_n - (i * chunk);
                checksum += chunk_;
                printf("rank %03d: chunk %02d: read at %zu a chunk of %zu.\n", rank, i, i*chunk*sizeof(char), chunk_);
                check_mpi(MPI_File_read_at(bedfh, offset + size_t(i)*chunk*sizeof(char), &rawdata[size_t(i)*chunk], chunk_, MPI_CHAR, &status), __LINE__, __FILE__);
            }            
            if (checksum != rawdata_n) {
                cout << "FATAL!! checksum not equal to rawdata_n: " << checksum << " vs " << rawdata_n << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const auto et1 = std::chrono::high_resolution_clock::now();
        const auto dt1 = et1 - st1;
        const auto du1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt1).count();
        //std::cout << "rank " << rank << ", time to read the BED file: " << du1 / double(1000.0) << " s." << std::endl;
        if (rank == 0)
            std::cout << "INFO   : time to read the BED file: " << du1 / double(1000.0) << " seconds." << std::endl;

        // Close BED file
        check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);


        //EO: leftover from previous implementation but keep it in for now
        //data.preprocess_data(rawdata, M, snpLenByt, ppdata, rank);

    
        //cout << " *** READ FROM BED FILE" << endl;
        N1S = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N1S, __LINE__, __FILE__);
        N1L = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N1L, __LINE__, __FILE__);
        N2S = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N2S, __LINE__, __FILE__);
        N2L = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(N2L, __LINE__, __FILE__);
        NMS = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(NMS, __LINE__, __FILE__);
        NML = (size_t*)malloc(size_t(M) * sizeof(size_t));  check_malloc(NML, __LINE__, __FILE__);
        dalloc += 6.0 * double(M) * sizeof(double) / 1E9;
        
        size_t N1, N2, NM;
        data.sparse_data_get_sizes(rawdata, M, snpLenByt, &N1, &N2, &NM);
        //printf("OFF rank %d N1 = %10lu, N2 = %10lu, NM = %10lu\n", rank, N1, N2, NM);

        // Alloc and build sparse structure
        I1 = (size_t*) malloc(N1 * sizeof(size_t));  check_malloc(I1, __LINE__, __FILE__);
        I2 = (size_t*) malloc(N2 * sizeof(size_t));  check_malloc(I2, __LINE__, __FILE__);
        IM = (size_t*) malloc(NM * sizeof(size_t));  check_malloc(IM, __LINE__, __FILE__);
        dalloc += (N1 + N2 + NM) * sizeof(size_t) / 1E9;
        
        data.sparse_data_fill_indices(rawdata, M, snpLenByt,
                                      N1S, N1L, I1,
                                      N2S, N2L, I2,
                                      NMS, NML, IM);
        
        //EO: to clean
        //data.compute_markers_statistics(rawdata, M, snpLenByt, mave, mstd, msum);

        free(rawdata);
        
    } else {
        
        //cout << " *** READ FROM SPARSE REPRESENTATION FILES" << endl;
        read_sparse_data_files(N1S, N1L, I1,
                               N2S, N2L, I2,
                               NMS, NML, IM,
                               MrankS, MrankL, rank);
    }


    // Compute statistics (from sparse info)
    // -------------------------------------
    double *mave, *mstd;
    mave = (double*)malloc(size_t(M) * sizeof(double));  check_malloc(mave, __LINE__, __FILE__);
    mstd = (double*)malloc(size_t(M) * sizeof(double));  check_malloc(mstd, __LINE__, __FILE__);
    dalloc += 2 * size_t(M) * sizeof(double) / 1E9;
    dalloc +=     size_t(M) * sizeof(uint)   / 1E9;
    
    double tmp0, tmp1, tmp2;
    for (int i=0; i<M; ++i) {
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (double(N) - double(NML[i]));
        //if (rank == 0 && i < 3)
        //    printf("mave[%d] = %20.15f\n", i, mave[i]);
        tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(N - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        mstd[i] = sqrt(double(N - 1) / (tmp0+tmp1+tmp2));
    }


    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)
        std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Build list of markers    
    // ---------------------
    for (int i=0; i<M; ++i)
        markerI.push_back(i);
    //printf("markerI start = %d and end = %d\n", markerI[0], markerI[M-1]);
    //std::iota(markerI.begin(), markerI.end(), 0);


    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();
 
    double *y, *epsilon, *tmpEps, *deltaEps, *dEpsSum, *deltaSum;
    const size_t NDB = size_t(N) * sizeof(double);
    y        = (double*)malloc(NDB);  check_malloc(y,        __LINE__, __FILE__);
    epsilon  = (double*)malloc(NDB);  check_malloc(epsilon,  __LINE__, __FILE__);
    tmpEps   = (double*)malloc(NDB);  check_malloc(tmpEps,   __LINE__, __FILE__);
    deltaEps = (double*)malloc(NDB);  check_malloc(deltaEps, __LINE__, __FILE__);
    dEpsSum  = (double*)malloc(NDB);  check_malloc(dEpsSum,  __LINE__, __FILE__);
    deltaSum = (double*)malloc(NDB);  check_malloc(deltaSum, __LINE__, __FILE__);
    dalloc += NDB * 8 / 1E9;

    double totalloc = 0.0;
    //printf("rank %02d dalloc = %.3f GB\n", rank, dalloc);
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);

    priorPi[0] = 0.5;
    cVa[0]     = 0.0;
    cVaI[0]    = 0.0;
    muk[0]     = 0.0;
    mu         = 0.0;

    for (int i=0; i<N; ++i)
        dEpsSum[i] = 0.0;

    cVa.segment(1,km1)     = cva;
    cVaI.segment(1,km1)    = cVa.segment(1,km1).cwiseInverse();
    priorPi.segment(1,km1) = priorPi[0] * cVa.segment(1,km1).array() / cVa.segment(1,km1).sum();
    sigmaG                 = dist.beta_rng(1.0, 1.0);
    //printf("First sigmaG = %15.10f\n", sigmaG);
    pi                     = priorPi;
    beta.setZero();
    components.setZero();

    if (opt.covariates) {
    	gamma = VectorXd(data.X.cols());
    	gamma.setZero();
    }
    
    double y_mean = 0.0;
    for (int i=0; i<N; ++i) {
        y[i]    = (double)data.y(i);
        y_mean += y[i];
    }
    y_mean /= N;
    //printf("rank %d: y_mean = %20.15f\n", rank, y_mean);

    for (int i=0; i<N; ++i) {
        y[i] -= y_mean;
    }

    double y_sqn = 0.0d;
    for (int i=0; i<N; ++i) {
        y_sqn += y[i] * y[i];
    }
    y_sqn = sqrt(y_sqn / dNm1);
    //printf("ysqn = %15.10f\n", y_sqn);

    sigmaE = 0.0d;
    for (int i=0; i<N; ++i) {
        y[i]       /= y_sqn;
        epsilon[i]  = y[i]; // - mu but zero
        sigmaE     += epsilon[i] * epsilon[i];
    }
    sigmaE = sigmaE / dN * 0.5d;
    //printf("sigmaE = %20.10f with epsilon = y-mu %22.15f\n", sigmaE, mu);

    double   sum_beta_squaredNorm;
    double   sigE_G, sigG_E, i_2sigE;
    double   bet, betaOld, deltaBeta, beta_squaredNorm, p, acum, e_sqn;
    size_t   markoff;
    int      marker, left;
    VectorXd logL(K);
    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    sigmaF = s02F;


    // Adapt the --thin and --save options such that --save >= --thin and --save%--thin = 0
    // ------------------------------------------------------------------------------------
    if (opt.save < opt.thin) {
        opt.save = opt.thin;
        if (rank == 0)
            printf("WARNING: opt.save was lower that opt.thin ; opt.save reset to opt.thin (%d)\n", opt.thin);
    }
    if (opt.save%opt.thin != 0) {
        if (rank == 0)
            printf("WARNING: opt.save (= %d) was not a multiple of opt.thin (= %d)\n", opt.save, opt.thin);
        opt.save = int(opt.save/opt.thin) * opt.thin;
        if (rank == 0)
            printf("         opt.save reset to %d, the closest multiple of opt.thin (%d)\n", opt.save, opt.thin);
    }


    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;

    // Main iteration loop
    // -------------------
    for (uint iteration=0; iteration < max_it; iteration++) {

        double start_it = MPI_Wtime();

        //printf("mu = %15.10f   eps[0] = %15.10f\n", mu, epsilon[0]);
        for (int i=0; i<N; ++i)
            epsilon[i] += mu;
        
        double epssum  = 0.0;
        for (int i=0; i<N; ++i)
            epssum += epsilon[i];
        // update mu (mean eps is mu)
        //mu = dist.norm_rng(mu, sigmaE/dN);
        mu = dist.norm_rng(epssum / dN, sigmaE / dN); //update mu
        //printf("mu = %15.10f\n", mu);

        // We substract again now epsilon =Y-mu-X*beta
        for (int i=0; i<N; ++i)
            epsilon[i] -= mu;

        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------
        if (shuf_mark) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
            //std::random_shuffle(markerI.begin(), markerI.end());
        }

        m0 = 0.0d;
        v.setZero();

        sigE_G  = sigmaE / sigmaG;
        sigG_E  = sigmaG / sigmaE;
        i_2sigE = 1.0 / (2.0 * sigmaE);

        for (int i=0; i<N; ++i)
            tmpEps[i] = epsilon[i];

        double cumSumDeltaBetas = 0.0;
        int sinceLastSync = 0;

        // Loop over (shuffled) markers
        
        //for (int j = 0; j < M; j++) {
        for (int j = 0; j < lmax; j++) {

            if (j < M) {
                marker  = markerI[j];
                //printf("marker %d %d\n", j, marker);
                
                //EO: leftover from previous implementation but keep it in for now
                //markoff = size_t(marker) * size_t(N);
                //double* Cxx      = &ppdata[markoff];
                // Convert marker data from BED in RAM as normalized DP
                // ----------------------------------------------------
                //VectorXd Cx(N);
                //Cx = getSnpData(marker);
                //data.get_normalized_marker_data(rawdata, snpLenByt, marker, Cxx);
                //data.get_normalized_marker_data(rawdata, snpLenByt, marker, Cx, mave[marker], mstd[marker]);
                //printf("%d/%d/%d: Cx[0] = %20.15f / %20.15f\n", iteration, rank, marker, Cx[0], ppdata[markoff]);
                
                bet =  beta(marker);
                //printf("beta = %20.15f, mean = %20.15f, std = %20.15f\n", bet, mave[marker], mstd[marker]);
                
                //we compute the denominator in the variance expression to save computations
                //denom = dNm1 + sigE_G * cVaI.segment(1, km1).array();
                for (int i=1; i<=km1; ++i) {
                    denom(i-1) = dNm1 + sigE_G * cVaI(i);
                    //printf("denom[%d] = %20.15f\n", i-1, denom(i-1));
                }
                
                //for (int i=0; i<5; ++i)
                //    printf("(%20.15f) %20.15f  %20.15f\n", Cx[i], Cxx[i], epsilon[i]);
                
                //EO: leftover from previous implementation but keep it in for now
                //double num2 = dotprod(epsilon, Cxx, N);
                
                //we compute the dot product to save computations
                //cout << "Marker " << marker << ": " <<  N1S[marker] << ", " << N2S[marker] << ", " <<  NMS[marker] << endl;
                //cout << "Marker " << marker << ": " <<  N1L[marker] << ", " << N2L[marker] << ", " <<  NML[marker] << endl;

                num = sparse_dotprod(epsilon,
                                     I1, N1S[marker], N1L[marker],
                                     I2, N2S[marker], N2L[marker],
                                     IM, NMS[marker], NML[marker],
                                     mave[marker],    mstd[marker], N);
                num += bet * double(N - 1);
                //printf("num = %20.15f\n", num);
                
                //muk for the other components is computed according to equations
                muk.segment(1, km1) = num / denom.array();           
                //cout << muk << endl;
                
                //first component probabilities remain unchanged
                logL = pi.array().log();
                
                // Update the log likelihood for each component
                logL.segment(1,km1) = logL.segment(1, km1).array()
                    - 0.5d * (sigG_E * dNm1 * cVa.segment(1,km1).array() + 1.0d).array().log() 
                    + muk.segment(1,km1).array() * num * i_2sigE;
                
                // I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later
                //p = dist.beta_rng(1.0, 1.0);
                p = dist.unif_rng();
                //printf("%d/%d/%d  p = %15.10f\n", iteration, rank, j, p);
                
                acum = 0.d;
                if(((logL.segment(1,km1).array()-logL[0]).abs().array() > 700 ).any() ){
                    acum = 0.0d;
                } else{
                    acum = 1.0d / ((logL.array()-logL[0]).exp().sum());
                }
                //printf("acum = %15.10f\n", acum);
                
                //EO: K -> K-1 by Daniel on 20190219!
                //-----------------------------------
                //for (int k=0; k<K-1; k++) {
                for (int k=0; k<K; k++) {
                    if (p <= acum) {
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
                        if (((logL.segment(1,km1).array()-logL[k+1]).abs().array() > 700.0d ).any() ){
                            acum += 0.0d;
                        } else{
                        acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum());
                        }
                    }
                }
                //printf("acum = %15.10f\n", acum);
                
                betaOld   = bet;
                bet       = beta(marker);
                deltaBeta = betaOld - bet;
                //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f\n", iteration, rank, marker, deltaBeta, betaOld, bet);
                //fflush(stdout);
                //MPI_Barrier(MPI_COMM_WORLD);
                
                // Compute delta epsilon
                //scaadd(deltaEps, Cx, deltaBeta, N);
                if (deltaBeta != 0.0) {
                    //printf("Trigger update!!\n");
                    sparse_scaadd(deltaEps, deltaBeta, 
                                  I1, N1S[marker], N1L[marker], 
                                  I2, N2S[marker], N2L[marker], 
                                  IM, NMS[marker], NML[marker], 
                                  mave[marker], mstd[marker], N);
                    
                    // Update local sum of delta epsilon
                    for (int i=0; i<N; ++i)
                        dEpsSum[i] += deltaEps[i];
                }
            } 

            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else {
                //cout << "rank " << rank << " with M=" << M << " waiting for " << lmax << endl;
                deltaBeta = 0.0;
                for (int i=0; i<N; ++i)
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
            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, bet, cumSumDeltaBetas);
            
            //if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == M-1) && cumSumDeltaBetas != 0.0) {
            if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1) && cumSumDeltaBetas != 0.0) {

                // Update local copy of epsilon
                if (nranks > 1) {
                    check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                    for (int i=0; i<N; ++i)
                        epsilon[i] = tmpEps[i] + deltaSum[i];
                } else {
                    for (int i=0; i<N; ++i)
                        epsilon[i] = tmpEps[i] + dEpsSum[i];
                }

                // Store epsilon state at last synchronization
                for (int i=0; i<N; ++i)
                    tmpEps[i] = epsilon[i];                    
                
                // Reset local sum of delta epsilon
                for (int i=0; i<N; ++i)
                    dEpsSum[i] = 0.0;
                
                // Reset cumulated sum of delta betas
                cumSumDeltaBetas = 0.0;

                sinceLastSync = 0;
            } else {
                sinceLastSync += 1;
            }

        } // END PROCESSING OF ALL MARKERS

        beta_squaredNorm = beta.squaredNorm();
        //printf("rank %d it %d  beta_squaredNorm = %15.10f\n", rank, iteration, beta_squaredNorm);


        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            check_mpi(MPI_Allreduce(&beta_squaredNorm, &sum_beta_squaredNorm, 1,        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(v.data(),          sum_v.data(),          v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            v                = sum_v;
            beta_squaredNorm = sum_beta_squaredNorm;
        }

        m0 = double(Mtot) - v[0];

        sigmaG  = dist.inv_scaled_chisq_rng(v0G+m0, (beta_squaredNorm * m0 + v0G*s02G) /(v0G+m0));


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
                    
                    for (int k=0; k<N; k++)
                        num_f += data.X(k, xI[i]) * (epsilon[k] + gamma_old * data.X(k, xI[i]));
                    denom_f = dNm1 + sigE_sigF;
                    gamma(i) = dist.norm_rng(num_f/denom_f, sigmaE/denom_f);
                    
                    for (int k = 0; k<N ; k++) {
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
        for (int i=0; i<N; ++i)
            e_sqn += epsilon[i] * epsilon[i];

        sigmaE  = dist.inv_scaled_chisq_rng(v0E+double(N),(e_sqn + v0E*s02E) /(v0E+double(N)));
        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, N, sigmaE);
        printf("it %4d, rank %4d: sigmaG(%15.10f, %15.10f) = %15.10f, sigmaE = %15.10f, betasq = %15.10f, m0 = %d\n", iteration, rank, v0G+m0,(beta_squaredNorm * m0 + v0G*s02G) /(v0G+m0), sigmaG, sigmaE, beta_squaredNorm, int(m0));
        fflush(stdout);

        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));
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

            if (rank == 0) {
                betoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(betfh, betoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                cpnoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpnfh, cpnoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            betoff = sizeof(uint) + sizeof(uint) 
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double))
                + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(betfh, betoff, beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            cpnoff = sizeof(uint) + sizeof(uint)
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int))
                + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh, cpnoff, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            n_thinned_saved += 1;

            //EO: to remove once MPI version fully validated; use the check_marker utility to retrieve
            //    the corresponding values from .bet file
            //    Print a sub-set of non-zero betas, one per rank for validation of the .bet file
            for (int i=0; i<M; ++i) {
                if (beta(i) != 0.0) {
                    //printf("%4d/%4d global beta[%8d] = %15.10f, components[%8d] = %2d\n", iteration, rank, MrankS[rank]+i, beta(i), MrankS[rank]+i, components(i));
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
                check_mpi(MPI_File_write_at(epsfh, epsoff, &N,         1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
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
        //if (rank == 0) { printf("Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it); }
    }


    // Close output files
    check_mpi(MPI_File_close(&outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);

    // Release memory
    free(y);
    free(epsilon);
    free(tmpEps);
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


int BayesRRm::runGibbs()
{
    //const unsigned int M(data.numSnps);
    unsigned int M(data.numSnps);
    if (opt.numberMarkers > 0 && opt.numberMarkers < M)
        M = opt.numberMarkers;
    const unsigned int N(data.numInds);
    const double NM1 = double(N - 1);
    const int K(int(cva.size()) + 1);
    const int km1 = K - 1;

    //initialize variables with init member function
    init(K, M, N);

    //specify how to write samples
    if (1==0) {
        SampleWriter writer;
        writer.setFileName(opt.mcmcOut + ".csv");
        writer.setMarkerCount(M);
        writer.setIndividualCount(N);
        writer.open();
    }

    // Sampler variables
    VectorXd sample(2*M+4+N); // variable containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
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
        //printf("mu = %15.10f\n", mu);
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
            //double c2 = 0.0;
            //for (int i=0; i<N; i++) {
            //    c2 += Cx[i] * y_tilde[i];
            //}
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

            for (int k = 0; k < K; k++) {
                if (p <= acum) {
                    //if zeroth component
                    if (k == 0) {
                        beta(marker) = 0.0;
                    } else {
                        beta(marker) = dist.norm_rng(muk[k], sigmaE/denom[k-1]);
                        //printf("@B@ beta update %4d/%4d/%4d muk[%4d] = %15.10f with p=%15.10f <= acum=%15.10f\n", iteration, 0, marker, k, muk[k], p, acum);
                    }
                    v[k] += 1.0;
                    components[marker] = k;
                    break;
                } else {
                    if (((logL.segment(1, km1).array() - logL[k+1]).abs().array() > 700).any()) {//again controlling for large values for the exponent
                        acum += 0;
                    } else {
                        acum += 1.0 / ((logL.array() - logL[k+1]).exp().sum());
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
    const unsigned int N(data.numInds);
    cout << "inv scaled parameters " << v0G + m0 << "__" << (beta.squaredNorm() * m0 + v0G * s02G) / (v0G + m0);
    cout << "num components: " << opt.S.size();
    cout << "\nMixture components: " << cva[0] << " " << cva[1] << " " << cva[2] << "\n";
    cout << "sigmaG: " << sigmaG << "\n";
    cout << "y mean: " << y.mean() << "\n";
    cout << "y sd: " << sqrt(y.squaredNorm() / (double(N - 1))) << "\n";
    // cout << "x mean " << Cx.mean() << "\n";
    //   cout << "x sd " << sqrt(Cx.squaredNorm() / (double(N - 1))) << "\n";
}
