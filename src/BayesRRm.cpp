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

#ifdef USE_MPI
#include <mpi.h>
#endif



BayesRRm::BayesRRm(Data &data, Options &opt, const long memPageSize)
: data(data)
, opt(opt)
, bedFile(opt.bedFile + ".bed")
, memPageSize(memPageSize)
, outputFile(opt.mcmcSampleFile)
, seed(opt.seed)
, max_iterations(opt.chainLength)
, burn_in(opt.burnin)
, thinning(opt.thin)
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
}


#ifdef USE_MPI

struct Lineout {
    double sigmaE,sigmaG;
    int    iteration, rank;
} lineout;


void sample_error(int error, const char *string)
{
  fprintf(stderr, "Error %d in %s\n", error, string);
  MPI_Finalize();
  exit(-1);
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


inline void sparse_scaadd(double*       __restrict__ vout,
                          const double* __restrict__ vin1,
                          const double  dMULT,
                          const size_t* __restrict__ I1, const size_t N1S, const size_t N1L,
                          const size_t* __restrict__ I2, const size_t N2S, const size_t N2L,
                          const double  mu,
                          const double  sig_inv,
                          const int     N) {

    if (dMULT == 0.0) {
        for (int i=0; i<N; i++)
            vout[i] = vin1[i];

        for (size_t i=N1S; i<N1S+N1L; ++i)
            vout[I1[i]] = vin1[I1[i]];

        for (size_t i=N2S; i<N2S+N2L; ++i)
            vout[I2[i]] = vin1[I2[i]];
    } else {
        double aux = mu * sig_inv * dMULT;
        for (int i=0; i<N; i++)
            vout[i] = vin1[i] - aux;

        aux = dMULT * (1.0 - mu) * sig_inv;
        for (size_t i=N1S; i<N1S+N1L; ++i)
            vout[I1[i]] = vin1[I1[i]] + aux;

        aux = dMULT * (2.0 - mu) * sig_inv;
        for (size_t i=N2S; i<N2S+N2L; ++i)
            vout[I2[i]] = vin1[I2[i]] + aux;
    }
}


inline void sparse_scaadd(double*       __restrict__ vout,
                          const double  dMULT,
                          const size_t* __restrict__ I1, const size_t N1S, const size_t N1L,
                          const size_t* __restrict__ I2, const size_t N2S, const size_t N2L,
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

        aux = dMULT * (1.0 - mu) * sig_inv;
        for (size_t i=N1S; i<N1S+N1L; ++i)
            vout[I1[i]] = aux;

        aux = dMULT * (2.0 - mu) * sig_inv;
        for (size_t i=N2S; i<N2S+N2L; ++i)
            vout[I2[i]] = aux;
    }
}


inline double sparse_dotprod(const double* __restrict__ vin1, // y_tilde
                             const size_t* __restrict__ I1, const size_t N1S, const size_t N1L,
                             const size_t* __restrict__ I2, const size_t N2S, const size_t N2L,
                             const double  mu,
                             const double  sig_inv,
                             const int     N) {

    double dp = 0.0, syt = 0.0;
    
    for (size_t i=N1S; i<N1S+N1L; ++i)
        dp += vin1[I1[i]];

    for (size_t i=N2S; i<N2S+N2L; ++i)
        dp += 2.0 * vin1[I2[i]];

    dp *= sig_inv;

    for (int i=0; i<N; i++)
        syt += vin1[i];
    
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
void mpi_define_blocks_of_markers(const int Mtot, int* MrankS, int* MrankL) {

    int nranks, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int modu   = Mtot % nranks;
    int Mrank  = int(Mtot / nranks);
    int checkM = 0;
    int start  = 0;

    /*
#ifdef USEALLMARKERS
    */
    for (int i=0; i<nranks; ++i) {
        MrankL[i] = int(Mtot / nranks);
        if (modu != 0 && i < modu)
            MrankL[i] += 1;
        MrankS[i] = start;
        //printf("start %d, len %d\n", MrankS[i], MrankL[i]);
        start += MrankL[i];
        checkM += MrankL[i];
    }
    assert(checkM == Mtot);

    /*
#else
    // Accept loosing M%nranks markers but easier to sync
    for (int i=0; i<nranks; ++i) {
        MrankL[i] = int(Mtot / nranks);
        MrankS[i] = start; //
        //printf("start %d, len %d\n", MrankS[i], MrankL[i]);
        start  += MrankL[i];
        checkM += MrankL[i];
    }
#endif
    */

    if (rank == 0)
        printf("checkM vs Mtot: %d vs %d. Will sacrify %d markers!\n", checkM, Mtot, Mtot-checkM);

}



//EO: This method writes sparse data files out of the BED one 
//    Note: will always convert the whole file
//---------------------------------------------------------------------
void BayesRRm::write_sparse_data_files() {

    int rank, nranks, result;
    double dalloc;
    const size_t LENBUF=200;
    char buff[LENBUF];

    // Initialize MPI environment
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File   bedfh, si1fh, sl1fh, ss1fh, si2fh, sl2fh, ss2fh;
    MPI_Offset offset;
    MPI_Status status;

    if (rank == 0)
        printf("Will generate sparse data files out of %d ranks\n", nranks);

    // Get dimensions of the dataset
    const unsigned int N    = data.numInds;
    const unsigned int Mtot = data.numSnps;
    if (rank == 0)
        printf("Full dataset includes %d markers and %d individuals.\n", Mtot, N);

    // Length of a column in bytes
    const size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;
    if (rank==0)
        printf("snpLenByt = %zu bytes.\n", snpLenByt);

    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks];
    mpi_define_blocks_of_markers(Mtot, MrankS, MrankL);

    /*
    int MrankS[nranks];
    int MrankL[nranks];
    int modu  = Mtot % nranks;
    int Mrank = int(Mtot / nranks);
    int checkM = 0;
    int start = 0;

    for (int i=0; i<nranks; ++i) {
        MrankL[i] = int(Mtot / nranks);
        if (modu != 0 && i < modu)
            MrankL[i] += 1;
        MrankS[i] = start;
        //printf("start %d, len %d\n", MrankS[i], MrankL[i]);
        start += MrankL[i];
        checkM += MrankL[i];
    }
    assert(checkM == Mtot);
    if (rank == 0)
        printf("checkM vs Mtot: %d vs %d. Will sacrify %d markers!\n", checkM, Mtot, Mtot-checkM);
    */

    int M = MrankL[rank];


    // Alloc memory for raw BED data
    const size_t rawdata_n = size_t(M) * size_t(snpLenByt) * sizeof(char);
    char* rawdata = (char*) malloc(rawdata_n); if (rawdata == NULL) { printf("malloc rawdata failed.\n"); exit (1); }
    dalloc += rawdata_n / 1E9;

    // Print information
    printf("rank %4d will handle a block of %6d markers starting at %7d, raw = %7.3f GB\n", rank, MrankL[rank], MrankS[rank], dalloc);


    // Read the BED file
    // -----------------
    std::string bedfp = opt.bedFile;
    bedfp += ".bed";
    result = MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh);
    if(result != MPI_SUCCESS) 
        sample_error(result, "MPI_File_open bed file");

    // Compute the offset of the section to read from the BED file
    offset = size_t(3) + size_t(MrankS[rank]) * size_t(snpLenByt) * sizeof(char);

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
        result = MPI_File_read_at(bedfh, offset, rawdata, rawdata_n, MPI_CHAR, &status);
        if(result != MPI_SUCCESS) 
            sample_error(result, "MPI_File_read_at");
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
            result = MPI_File_read_at(bedfh, offset + size_t(i)*chunk*sizeof(char), &rawdata[size_t(i)*chunk], chunk_, MPI_CHAR, &status);
            if(result != MPI_SUCCESS) 
                sample_error(result, "MPI_File_read_at");
        }
        if (checksum != rawdata_n) {
            cout << "FATAL!! checksum not equal to rawdata_n: " << checksum << " vs " << rawdata_n << endl; 
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et1 = std::chrono::high_resolution_clock::now();
    const auto dt1 = et1 - st1;
    const auto du1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt1).count();
    //std::cout << "rank " << rank << ", time to read the BED file: " << du1 / double(1000.0) << " s." << std::endl;
    if (rank == 0)
        std::cout << "Time to read the BED file: " << du1 / double(1000.0) << " seconds." << std::endl;

    // Close BED file
    result = MPI_File_close(&bedfh);
    if(result != MPI_SUCCESS) 
        sample_error(result, "MPI_File_close");


    // Preprocess the data
    // -------------------
    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    // Alloc memory for sparse representation
    size_t *N1S, *N1L, *N2S, *N2L;
    N1S = (size_t*) malloc(size_t(M) * sizeof(size_t)); if (N1S == NULL) { printf("malloc N1S failed.\n"); exit (1); }
    N1L = (size_t*) malloc(size_t(M) * sizeof(size_t)); if (N1L == NULL) { printf("malloc N1L failed.\n"); exit (1); }
    N2S = (size_t*) malloc(size_t(M) * sizeof(size_t)); if (N2S == NULL) { printf("malloc N2S failed.\n"); exit (1); }
    N2L = (size_t*) malloc(size_t(M) * sizeof(size_t)); if (N2L == NULL) { printf("malloc N2L failed.\n"); exit (1); }
    dalloc += 4.0 * double(M) * sizeof(double) / 1E9;

    size_t N1, N2;
    data.sparse_data_get_sizes(rawdata, M, snpLenByt, &N1, &N2);

    // Check how many calls are needed (limited by the int type of the number of elements to read!)
    assert(N1 < pow(2,(sizeof(int)*8)-1));
    assert(N2 < pow(2,(sizeof(int)*8)-1));

    // Alloc and build sparse structure
    size_t *I1, *I2;
    printf("rank %3d allocates %10.3f GB for I1\n", rank, double(N1 * sizeof(size_t))*1E-9);
    printf("rank %3d allocates %10.3f GB for I2\n", rank, double(N2 * sizeof(size_t))*1E-9);
    I1 = (size_t*) malloc( N1 * sizeof(size_t) ); if (I1 == NULL) { printf("malloc I1 failed.\n"); exit (1); }
    I2 = (size_t*) malloc( N2 * sizeof(size_t) ); if (I2 == NULL) { printf("malloc I2 failed.\n"); exit (1); }
    dalloc += N1 * sizeof(size_t) / 1E9;
    dalloc += N2 * sizeof(size_t) / 1E9;

    data.sparse_data_fill_indices(rawdata, M, snpLenByt, N1S, N1L, N2S, N2L, N1, N2, I1, I2);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)
        std::cout << "Time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Write sparse structure to file
    // ------------------------------

    // Gather sizes of I1, I2 over ranks
    size_t *AllN1 = (size_t*) malloc( nranks * sizeof(size_t) ); if (AllN1 == NULL) { printf("malloc AllN1 failed.\n"); exit (1); }
    size_t *AllN2 = (size_t*) malloc( nranks * sizeof(size_t) ); if (AllN2 == NULL) { printf("malloc AllN2 failed.\n"); exit (1); }
    MPI_Allgather(&N1, 1, MPI_UNSIGNED_LONG_LONG, AllN1, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&N2, 1, MPI_UNSIGNED_LONG_LONG, AllN2, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    size_t N1Off = 0, N2Off = 0;
    for (int i=0; i<rank; ++i) {
        N1Off += AllN1[i];
        N2Off += AllN2[i];
    }
    printf("rank %4d as N1 = %10lu and AllN1 = %10lu; Will dump at N1S offset %10lu  | N2 = %10lu and AllN2 = %10lu; Will dump at %10lu\n",
           rank, N1, AllN1[rank], N1Off, N2, AllN2[rank], N2Off);

    // ss1,2 files must contain absolute start indices
    // -----------------------------------------------
    for (int i=0; i<M; ++i) {
        N1S[i] += N1Off;
        N2S[i] += N2Off;
    }

    // Sparse Index Ones file (si1)
    std::string si1 = opt.bedFile + ".si1";
    result = MPI_File_open(MPI_COMM_WORLD, si1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open si1"); }
    offset = N1Off * sizeof(size_t);
    result = MPI_File_write_at_all(si1fh, offset, I1, N1, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all si1"); }
    result = MPI_File_close(&si1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close si1"); }
    if (rank == 0) { printf("INFO: wrote si1 file %s\n", si1.c_str()); }

    // Sparse Length Ones file (sl1)
    std::string sl1 = opt.bedFile + ".sl1";
    result = MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open sl1"); }
    offset = size_t(MrankS[rank]) * sizeof(size_t);
    result = MPI_File_write_at_all(sl1fh, offset, N1L, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all sl1"); }
    result = MPI_File_close(&sl1fh); 
    if(result != MPI_SUCCESS)  { sample_error(result, "MPI_File_close sl1"); }
    if (rank == 0) { printf("INFO: wrote sl1 file %s\n", sl1.c_str()); }

    // Sparse Start Ones file (ss1)
    std::string ss1 = opt.bedFile + ".ss1";
    result = MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open ss1"); }
    offset = size_t(MrankS[rank]) * sizeof(size_t);
    //cout << "Writing at " << offset << " el N1S[0] = " << N1S[0] << endl;
    result = MPI_File_write_at_all(ss1fh, offset, N1S, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all ss1"); }
    result = MPI_File_close(&ss1fh); 
    if (result != MPI_SUCCESS)  { sample_error(result, "MPI_File_close ss1"); }
    if (rank == 0) { printf("INFO: wrote ss1 file %s\n", ss1.c_str()); }


    // Sparse Index Twos file (si2)
    std::string si2 = opt.bedFile + ".si2";
    result = MPI_File_open(MPI_COMM_WORLD, si2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open si1"); }
    offset = N2Off * sizeof(size_t) ;
    result = MPI_File_write_at_all(si2fh, offset, I2, N2, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all si2"); }
    result = MPI_File_close(&si2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close si2"); }
    if (rank == 0) { printf("INFO: wrote si2 file %s\n", si2.c_str()); }

    // Sparse Length Twos file (sl2)
    std::string sl2 = opt.bedFile + ".sl2";
    result = MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open sl2"); }
    offset = size_t(MrankS[rank]) * sizeof(size_t);
    result = MPI_File_write_at_all(sl2fh, offset, N2L, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all s21"); }
    result = MPI_File_close(&sl2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close sl2"); }
    if (rank == 0) { printf("INFO: wrote sl2 file %s\n", sl2.c_str()); }

    // Sparse Start Ones file (ss2)
    std::string ss2 = opt.bedFile + ".ss2";
    result = MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open ss2"); }
    offset = size_t(MrankS[rank]) * sizeof(size_t);
    result = MPI_File_write_at_all(ss2fh, offset, N2S, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all ss2"); }
    result = MPI_File_close(&ss2fh); 
    if (result != MPI_SUCCESS)  { sample_error(result, "MPI_File_close ss2"); }
    if (rank == 0) { printf("INFO: wrote ss2 file %s\n", ss2.c_str()); }


    // Free allocated memory
    free(rawdata);
    free(N1S);   free(N1L); free(I1);
    free(N2S);   free(N2L); free(I2);
    free(AllN1); free(AllN2);

    // Finalize the MPI environment
    MPI_Finalize();
}


size_t get_file_size(const std::string& filename) {
    struct stat st;
    if(stat(filename.c_str(), &st) != 0) {
        return 0;
    }
    return st.st_size;   
}


void BayesRRm::read_sparse_data_files(size_t*& I1, size_t*& I2, size_t*& N1S, size_t*& N1L,  size_t*& N2S, size_t*& N2L, int* MrankS, int* MrankL) {

    int rank, nranks, result;
    double dalloc;
    const size_t LENBUF=200;
    char buff[LENBUF];

    // Initialize MPI environment
    //MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Offset offset;
    MPI_Status status;

    // Get dimensions of the dataset
    //const unsigned int N_    = data.numInds;
    //const unsigned int Mtot_ = data.numSnps;
    //if (rank == 0)
    //    printf("Full dataset includes %d markers and %d individuals.\n", Mtot_, N_);

    // From the size of .sl1 and .sl2 compute the number of markers
    // ------------------------------------------------------------
    //size_t nbysl1 = get_file_size(opt.bedFile + ".sl1");
    //int mpisizeofull;
    //MPI_Type_size(MPI_Datatype MPI_UNSIGNED_LONG_LONG, &mpisizeofull);
    //assert(nbysl1%mpisizeofull == 0);
    //uint Mtot = nbysl1 / mpisizeofull;
    //if (rank == 0)
    //    std::cout << "File size = " << nbysl1 << " bytes => Mtot = " << Mtot << endl;


    const uint M = MrankL[rank];

    // Alloc memory for sparse representation
    //size_t *N1S, *N1L, *N2S, *N2L;
    N1S = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N1S == NULL) { printf("malloc N1S failed.\n"); exit (1); }
    N1L = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N1L == NULL) { printf("malloc N1L failed.\n"); exit (1); }
    N2S = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N2S == NULL) { printf("malloc N2S failed.\n"); exit (1); }
    N2L = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N2L == NULL) { printf("malloc N2L failed.\n"); exit (1); }

    // Read sparse data files
    // Each task is in charge of M markers starting from MrankS[rank]
    // So first we read si1 to get where to read in 
    MPI_File si1fh, si2fh, sl1fh, sl2fh, ss1fh, ss2fh;

    // Get the lengths of ones for each marker in the block
    offset =  MrankS[rank] * sizeof(size_t);
    std::string sl1 = opt.bedFile + ".sl1";
    result = MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open sl1"); }
    result = MPI_File_read_at_all(sl1fh, offset, N1L, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all sl1"); }
    result = MPI_File_close(&sl1fh); if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close sl1"); }

    std::string ss1 = opt.bedFile + ".ss1";
    result = MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open ss1"); }
    result = MPI_File_read_at_all(ss1fh, offset, N1S, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all ss1"); }
    result = MPI_File_close(&ss1fh); if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close ss1"); }


    // Get the lengths of ones for each marker in the block
    std::string sl2 = opt.bedFile + ".sl2";
    result = MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open sl2"); }
    result = MPI_File_read_at_all(sl2fh, offset, N2L, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all sl2"); }
    result = MPI_File_close(&sl2fh); if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close sl2"); }

    std::string ss2 = opt.bedFile + ".ss2";
    result = MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open ss2"); }
    result = MPI_File_read_at_all(ss2fh, offset, N2S, M, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_write_at_all ss2"); }
    result = MPI_File_close(&ss2fh); if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close ss2"); }


    size_t is1 = N1S[0];
    size_t ie1 = N1S[M-1] + N1L[M-1] - 1;
    size_t is2 = N2S[0];
    size_t ie2 = N2S[M-1] + N2L[M-1] - 1;
    size_t N1  = ie1 - is1 + 1;
    size_t N2  = ie2 - is2 + 1;

    // Check how many calls are needed (limited by the int type of the number of elements to read!)
    assert(N1 < pow(2,(sizeof(int)*8)-1));
    assert(N2 < pow(2,(sizeof(int)*8)-1));

    // Alloc and build sparse structure
    //size_t *I1, *I2;
    I1 = (size_t*) malloc( N1 * sizeof(size_t) ); if (I1 == NULL) { printf("malloc I1 failed.\n"); exit (1); }
    I2 = (size_t*) malloc( N2 * sizeof(size_t) ); if (I2 == NULL) { printf("malloc I2 failed.\n"); exit (1); }
    //dalloc += N1 * sizeof(size_t) / 1E9;
    //dalloc += N2 * sizeof(size_t) / 1E9;


    // Make starts relative to start of block in each task
    // ---------------------------------------------------
    size_t n1soff = N1S[0];
    for (int i=0; i<M; ++i)
        N1S[i] -= n1soff;

    size_t n2soff = N2S[0];
    for (int i=0; i<M; ++i)
        N2S[i] -= n2soff;


    // Read the indices of 1s
    std::string si1 = opt.bedFile + ".si1";
    result = MPI_File_open(MPI_COMM_WORLD, si1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &si1fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open si1"); }
    offset =  is1 * sizeof(size_t);
    result = MPI_File_read_at_all(si1fh, offset, I1, N1, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_read_at_all si1"); }
    result = MPI_File_close(&si1fh); if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close si1"); }


    // Read the indices of 2s
    std::string si2 = opt.bedFile + ".si2";
    result = MPI_File_open(MPI_COMM_WORLD, si2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &si2fh);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_open si2"); }
    offset =  is2 * sizeof(size_t);
    result = MPI_File_read_at_all(si2fh, offset, I2, N2, MPI_UNSIGNED_LONG_LONG, &status);
    if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_read_at_all si2"); }
    result = MPI_File_close(&si2fh); if (result != MPI_SUCCESS) { sample_error(result, "MPI_File_close si2"); }

    
    //free(N1S); free(N1L);
    //free(N2S); free(N2L);
    //free(I1);  free(I2);

    // Finalize the MPI environment
    //MPI_Finalize();
}


void mpi_assign_blocks_to_tasks(int* MrankS, int* MrankL, const uint numBlocks, const vector<int> blocksStarts, const vector<int> blocksEnds, const uint Mtot) {

    int rank, nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numBlocks > 0) {

        if (rank == 0)
            printf("INFO: Using distribution over tasks.\n");

        if (nranks != numBlocks) {
            printf("FATAL: block definition does not match number of tasks (%d versus %d).\n", numBlocks, nranks);
            printf("      => Provide each task with a block definition\n");
            exit(1);
        }

        // Make sure last marker is not greater than Mtot
        if (blocksEnds[numBlocks-1] > Mtot) {
            printf("FATAL: block definition goes beyond the number of markers to be processed (%d > Mtot = %d).\n", blocksEnds[numBlocks-1], Mtot);
            printf("      => Adjust block definition file\n");
            exit(1);
        }

        // Assign to MrankS and MrankL to catch up on logic
        for (int i=0; i<numBlocks; ++i) {
            MrankS[i] = blocksStarts[i] - 1;                  // starts from 0, not 1
            MrankL[i] = blocksEnds[i] - blocksStarts[i] + 1;  // compute length
        }

    } else {
        if (rank == 0)
            printf("INFO: no marker block definition file used. Will go for even distribution over tasks.\n");
        mpi_define_blocks_of_markers(Mtot, MrankS, MrankL);
    }
}



//EO: MPI GIBBS
//-------------
int BayesRRm::runMpiGibbs() {

    const size_t LENBUF=200;

    char buff[LENBUF]; 
    int  nranks, rank, name_len, result;
    double dalloc = 0.0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File   bedfh, outfh, betfh;
    MPI_Status status;
    MPI_Offset offset, betoff;

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

    if (rank == 0)
        printf("Full dataset includes %d markers and %d individuals.\n", Mtot, N);
    if (opt.numberMarkers > 0 && opt.numberMarkers < Mtot)
        Mtot = opt.numberMarkers;
    if (rank == 0)
        printf("Option passed to process only %d markers!\n", Mtot);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks];
    mpi_assign_blocks_to_tasks(MrankS, MrankL, data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot);


    int lmax = 0;
    for (int i=0; i<nranks; ++i)
        if (MrankL[i]>lmax) { lmax = MrankL[i]; }
    if (rank == 0)
        printf("Longest tasks has %d markers.\n", lmax);

    int M = MrankL[rank];
    printf("rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);


    const double	    sigma0 = 0.0001;
    const double	    v0E    = 0.0001;
    const double        s02E   = 0.0001;
    const double        v0G    = 0.0001;
    const double        s02G   = 0.0001;
    const unsigned int  K      = int(cva.size()) + 1;
    const unsigned int  km1    = K - 1;
    double              dNm1   = (double)(N - 1);
    double              dN     = (double) N;
    std::vector<int>    markerI;
    VectorXd            components(M);
    double              mu;              // mean or intercept
    double              sigmaG;          // genetic variance
    double              sigmaE;          // residuals variance
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

    dalloc += M * sizeof(double) / 1E9; // for components
    dalloc += M * sizeof(double) / 1E9; // for beta

    //VectorXd            sample(2*M+4+N); // varible containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance


    // Length of a column in bytes
    const size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;
    if (rank==0)
        printf("snpLenByt = %zu bytes.\n", snpLenByt);


    // Read the BED file
    // -----------------
    std::string bedfp = opt.bedFile;
    bedfp += ".bed";
    result = MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh);
    if(result != MPI_SUCCESS) 
        sample_error(result, "MPI_File_open bed file");


    // Output files for global variables and betas
    // -------------------------------------------
    std::string outfp = opt.mcmcSampleFile;
    result = MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh);
    if (result != MPI_SUCCESS) {
        int lc = sprintf(buff, "FATAL: MPI_File_open failed to open file %s", outfp.c_str());
        sample_error(result, buff);
    }
    
    std::string betfp = opt.mcmcBetFile;
    result = MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh);
    if (result != MPI_SUCCESS) {
        int lc = sprintf(buff, "FATAL: MPI_File_open failed to open file %s", betfp.c_str());
        sample_error(result, buff);
    }

    
    // First element of the .bet file is the total number of processed markers
    // -----------------------------------------------------------------------
    betoff = size_t(0);
    if (rank == 0) {
        result = MPI_File_write_at(betfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status);
        if(result != MPI_SUCCESS) 
            sample_error(result, "MPI_File_write_at number of markers");
    }


    //EO: not used but keep it in for now
    //-----------------------------------
    //int          blockcounts[2] = {2, 2};
    //MPI_Datatype types[2]       = {MPI_INT, MPI_DOUBLE}; 
    //MPI_Aint     displs[2];
    //MPI_Datatype typeout;
    //MPI_Get_address(&lineout.sigmaE,    &displs[0]);
    //MPI_Get_address(&lineout.iteration, &displs[1]);
    //for (int i = 1; i >= 0; i--)
    //    displs[i] -= displs[0];
    //MPI_Type_create_struct(2, blockcounts, displs, types, &typeout);
    //MPI_Type_commit(&typeout);


    // Alloc memory for raw BED data
    // -----------------------------
    const size_t rawdata_n = size_t(M) * size_t(snpLenByt) * sizeof(char);
    char* rawdata = (char*) malloc(rawdata_n); if (rawdata == NULL) { printf("malloc rawdata failed.\n"); exit (1); }
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
        result = MPI_File_read_at(bedfh, offset, rawdata, rawdata_n, MPI_CHAR, &status);
        if(result != MPI_SUCCESS) 
            sample_error(result, "MPI_File_read_at");
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
            result = MPI_File_read_at(bedfh, offset + size_t(i)*chunk*sizeof(char), &rawdata[size_t(i)*chunk], chunk_, MPI_CHAR, &status);
            if(result != MPI_SUCCESS) 
                sample_error(result, "MPI_File_read_at");
        }
        if (checksum != rawdata_n) {
            cout << "FATAL!! checksum not equal to rawdata_n: " << checksum << " vs " << rawdata_n << endl; 
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et1 = std::chrono::high_resolution_clock::now();
    const auto dt1 = et1 - st1;
    const auto du1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt1).count();
    //std::cout << "rank " << rank << ", time to read the BED file: " << du1 / double(1000.0) << " s." << std::endl;
    if (rank == 0)
        std::cout << "Time to read the BED file: " << du1 / double(1000.0) << " seconds." << std::endl;


    // Close BED file
    // --------------
    result = MPI_File_close(&bedfh);
    if(result != MPI_SUCCESS) 
        sample_error(result, "MPI_File_close");


    //EO: leftover from previous implementation but keep it in for now
    //----------------------------------------------------------------
    //data.preprocess_data(rawdata, M, snpLenByt, ppdata, rank);


    // Preprocess the data
    // -------------------
    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    // Alloc memory for sparse representation
    size_t *N1S,  *N1L,  *N2S,  *N2L,  *I1,  *I2;    

    if (opt.readFromBedFile) {
    
        //cout << " *** READ FROM BED FILE" << endl;
        N1S = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N1S == NULL) { printf("malloc N1S failed.\n"); exit (1); }
        N1L = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N1L == NULL) { printf("malloc N1L failed.\n"); exit (1); }
        N2S = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N2S == NULL) { printf("malloc N2S failed.\n"); exit (1); }
        N2L = (size_t*)malloc(size_t(M) * sizeof(size_t)); if (N2L == NULL) { printf("malloc N2L failed.\n"); exit (1); }
        dalloc += 4.0 * double(M) * sizeof(double) / 1E9;
        
        size_t N1, N2;
        data.sparse_data_get_sizes(rawdata, M, snpLenByt, &N1, &N2);
        printf("OFF rank %d N1 = %10lu, N2 = %10lu\n", rank, N1, N2);

        // Alloc and build sparse structure
        I1 = (size_t*) malloc( N1 * sizeof(size_t) ); if (I1 == NULL) { printf("malloc I1 failed.\n"); exit (1); }
        I2 = (size_t*) malloc( N2 * sizeof(size_t) ); if (I2 == NULL) { printf("malloc I2 failed.\n"); exit (1); }
        dalloc += N1 * sizeof(size_t) / 1E9;
        dalloc += N2 * sizeof(size_t) / 1E9;
        
        data.sparse_data_fill_indices(rawdata, M, snpLenByt, N1S, N1L, N2S, N2L, N1, N2, I1, I2);

    } else {
        
        //cout << " *** READ FROM SPARSE REPRESENTATION FILES" << endl;
        read_sparse_data_files(I1, I2, N1S, N1L, N2S, N2L, MrankS, MrankL);
    }

    // Compute markers' statistics
    double *mave, *mstd;
    uint   *msum;
    mave = (double*)malloc(size_t(M) * sizeof(double)); if (mave == NULL) { printf("malloc mave failed.\n"); exit (1); }
    mstd = (double*)malloc(size_t(M) * sizeof(double)); if (mstd == NULL) { printf("malloc mstd failed.\n"); exit (1); }
    msum = (uint*)  malloc(size_t(M) * sizeof(uint));   if (msum == NULL) { printf("malloc msum failed.\n"); exit (1); }
    dalloc += 2 * size_t(M) * sizeof(double) / 1E9;
    dalloc +=     size_t(M) * sizeof(uint)   / 1E9;
    
    data.compute_markers_statistics(rawdata, M, snpLenByt, mave, mstd, msum);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)
        std::cout << "Time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


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
    y        = (double*)malloc(NDB); if (y        == NULL) { printf("malloc y failed.\n");        exit (1); }
    epsilon  = (double*)malloc(NDB); if (epsilon  == NULL) { printf("malloc epsilon failed.\n");  exit (1); }
    tmpEps   = (double*)malloc(NDB); if (tmpEps   == NULL) { printf("malloc tmpEps failed.\n");   exit (1); }
    deltaEps = (double*)malloc(NDB); if (deltaEps == NULL) { printf("malloc deltaEps failed.\n"); exit (1); }
    dEpsSum  = (double*)malloc(NDB); if (dEpsSum  == NULL) { printf("malloc dEpsSum failed.\n");  exit (1); }
    deltaSum = (double*)malloc(NDB); if (deltaSum == NULL) { printf("malloc deltaSum failed.\n"); exit (1); }
    dalloc += NDB * 8 / 1E9;

    double totalloc = 0.0;
    //printf("rank %02d dalloc = %.3f GB\n", rank, dalloc);
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("OVERALL ALLOC %.3f GB\n", totalloc);
    }

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


    // Main iteration loop
    // -------------------
    for (int iteration=0; iteration < max_it; iteration++) {

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
                num = sparse_dotprod(epsilon, I1, N1S[marker], N1L[marker], I2, N2S[marker], N2L[marker], mave[marker], mstd[marker], N);
                //printf("num = %20.15f\n", num);
                num += bet * double(N - 1);
                //printf("num = %15.10f vs num2 = %15.10f\n", num, num2);
                
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
                    
                    sparse_scaadd(deltaEps, deltaBeta, I1, N1S[marker], N1L[marker], I2, N2S[marker], N2L[marker], mave[marker], mstd[marker], N);
                    
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
                MPI_Allreduce(&deltaBeta, &sumDeltaBetas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                cumSumDeltaBetas += sumDeltaBetas;
            } else {
                cumSumDeltaBetas += deltaBeta;
            } 
            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, bet, cumSumDeltaBetas);
            
            //if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == M-1) && cumSumDeltaBetas != 0.0) {
            if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1) && cumSumDeltaBetas != 0.0) {

                // Update local copy of epsilon
                if (nranks > 1) {
                    //EO
                    //MPI_Barrier(MPI_COMM_WORLD);
                    //double start = MPI_Wtime();
                    MPI_Allreduce(&dEpsSum[0], &deltaSum[0], N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    //double end = MPI_Wtime();
                    //std::cout << "MPI_Allreduce took " << end - start << " seconds to run." << std::endl;
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
            MPI_Allreduce(&beta_squaredNorm, &sum_beta_squaredNorm, 1,        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(v.data(),          sum_v.data(),          v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            v                = sum_v;
            beta_squaredNorm = sum_beta_squaredNorm;
        }

        m0 = double(Mtot) - v[0];

        sigmaG  = dist.inv_scaled_chisq_rng(v0G+m0, (beta_squaredNorm * m0 + v0G*s02G) /(v0G+m0));

        e_sqn = 0.0d;
        for (int i=0; i<N; ++i)
            e_sqn += epsilon[i] * epsilon[i];

        sigmaE  = dist.inv_scaled_chisq_rng(v0E+double(N),(e_sqn + v0E*s02E) /(v0E+double(N)));
        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, N, sigmaE);
        printf("it %4d, rank %4d: sigmaG(%15.10f, %15.10f) = %15.10f, sigmaE = %15.10f, betasq=%15.10f, m0=%10.1f\n", iteration, rank, v0G+m0,(beta_squaredNorm * m0 + v0G*s02G) /(v0G+m0), sigmaG, sigmaE, beta_squaredNorm, m0);
        fflush(stdout);

        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));
        //printf("sigmaG = %20.15f, sigmaE = %20.15f, e_sqn = %20.15f\n", sigmaG, sigmaE, e_sqn);
        //printf("it %6d, rank %3d: epsilon[0] = %15.10f, y[0] = %15.10f, m0=%10.1f,  sigE=%15.10f,  sigG=%15.10f [%6d / %6d]\n", iteration, rank, epsilon[0], y[0], m0, sigmaE, sigmaG, markerI[0], markerI[M-1]);

        pi = dist.dirichilet_rng(v.array() + 1.0);
        //cout << pi << endl;

        // Write to output file
        //Lineout lineout;
        //lineout.sigmaE    = sigmaE;
        //lineout.sigmaG    = sigmaG;
        //lineout.iteration = iteration;
        //lineout.rank      = rank;
        //offset = size_t(iteration) * size_t(nranks) + size_t(rank) * sizeof(lineout);
        //result = MPI_File_write_at_all(outfh, offset, &lineout, 1, typeout, &status);

        left = snprintf(buff, LENBUF, "%5d, %4d, %15.10f, %15.10f, %15.10f\n", iteration, rank, sigmaE, sigmaG, sigmaG/(sigmaE+sigmaG));
        //printf("letf = %d\n", left);
        offset = (size_t(iteration) * size_t(nranks) + size_t(rank)) * strlen(buff);
        result = MPI_File_write_at_all(outfh, offset, &buff, strlen(buff), MPI_CHAR, &status);
        if (result != MPI_SUCCESS) 
            sample_error(result, "MPI_File_write_at_all");

        // Dump the betas
        // --------------
        betoff = sizeof(uint) + (size_t(iteration) * size_t(Mtot) + size_t(MrankS[rank])) * sizeof(double);
        //printf("%d/%d betoff = %d\n", iteration, rank, betoff);
        result = MPI_File_write_at_all(betfh, betoff, beta.data(), beta.size(), MPI_DOUBLE, &status);
        if(result != MPI_SUCCESS) 
            sample_error(result, "MPI_File_write_at_all");

        //EO: to remove once MPI version fully validated; use the check_marker utility to retrieve
        //    the corresponding value from .bet files
        // Print a sub-set of non-zero betas, one per rank for validation of the .bet file
        for (int i=0; i<M; ++i) {
            if (beta(i) != 0.0) {
                printf("%4d/%4d beta(%8d -> %8d) = %15.10f\n", iteration, rank, i, rank*M+i, beta(i));
                break;
            }
        }

        double end_it = MPI_Wtime();
        if (rank == 0) 
            printf("Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);
    }

    result = MPI_File_close(&outfh);
    if (result != MPI_SUCCESS) 
        sample_error(result, "MPI_File_close outfh");
    
    result = MPI_File_close(&betfh);
    if (result != MPI_SUCCESS) 
        sample_error(result, "MPI_File_close betfh");


    //MPI_Type_free(&typeout);

    free(y);
    free(epsilon);
    free(tmpEps);
    free(deltaEps);
    free(dEpsSum);
    free(deltaSum);
    free(rawdata);
    free(mave);
    free(mstd);
    free(msum);
    free(N1S);
    free(N1L);
    free(N2S); 
    free(N2L);
    free(I1);
    free(I2);
    

    // Finalize the MPI environment
    MPI_Finalize();

    const auto et3 = std::chrono::high_resolution_clock::now();
    const auto dt3 = et3 - st3;
    const auto du3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();
    std::cout << "rank " << rank << ", time to process the data: " << du3 / double(1000.0) << " s." << std::endl;

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
        writer.setFileName(outputFile);
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
            //printf("beta = %15.10f, mean = %15.10f\n", beta_old, Cx.mean());

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
            const double num = Cx.dot(y_tilde);
            //printf("num = %15.10f\n", num);

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
        //if (iteration >= burn_in && iteration % thinning == 0) {
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
