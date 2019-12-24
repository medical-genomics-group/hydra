/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#ifndef SRC_BAYESRRM_H_
#define SRC_BAYESRRM_H_

#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"

#include <Eigen/Eigen>

class BayesRRm
{
    
 public:

    Data               &data;            // data matrices
    Options            &opt;
    const string       bedFile;          // bed file
    const long         memPageSize;      // size of memory
    const unsigned int seed;
    const unsigned int max_iterations;
    const unsigned int burn_in;
    const double	   sigma0  = 0.0001;
    const double	   v0E     = 0.0001;
    const double       s02E    = 0.0001;
    const double       v0G     = 0.0001;
    const double       s02G    = 0.0001;
    const double       s02F    = 1.0;
    const size_t       LENBUF  = 50000;

    //VectorXd  cva;     //EO: move v
    MatrixXd  cVa;       // component-specific variance
    MatrixXd  cVaI;      // inverse of the component variances

    Distributions_boost dist;
    Distributions_boost dist8[8];
    bool usePreprocessedData;
    bool showDebug;

    MatrixXd X;         // "fixed effects" matrix.
    VectorXd gamma;     // fixed effects coefficients

    // Component variables
    MatrixXd priorPi;   // prior probabilities for each component
    MatrixXd estPi;        // mixture probabilities
    //MatrixXd pi8;
    VectorXd logL;      // log likelihood of component
    VectorXd muk;       // mean of k-th component marker effect size
    VectorXd denom;     // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
    int      m0;        // total number of markers in model
    VectorXd cass;      // variable storing the component assignment //EO RENAMING: was v
    MatrixXd cass8;

    // Mean and residual variables
    double mu;          // mean or intercept
    VectorXd sigmaG;    // genetic variance
    double sigmaE;      // residuals variance
    double sigmaF;      // covariates variance if using ridge;

    double mu8[8];
    double sigmaG8[8];
    double sigmaE8[8];
    double sigmaF8[8];

    // Linear model variables
    VectorXd Beta;       // effect sizes
    VectorXd y_tilde;    // variable containing the adjusted residuals to exclude the effects of a given marker
    VectorXd epsilon;    // variable containing the residuals
    VectorXi components;
    MatrixXi components8;

    VectorXd         epsilon_restart; //EO:to store epsilon found in .eps* file when restarting a chain
    std::vector<int> markerI_restart; //   idem for markerI found in .mrk* file
    VectorXd         gamma_restart;   //   idem for gamma vector found in .gam* file
    std::vector<int> xI_restart;      //   idem for xI vector found in .xiv
    double           mu_restart;      //   to store task-wise mu read back from .mus.rank file

    VectorXd y;
    VectorXd Cx;

    double epsilonsum;
    double ytildesum;

    uint iteration_restart = 0;
    uint iteration_start   = 0;

    BayesRRm(Data &data, Options &opt, const long memPageSize);
    virtual ~BayesRRm();

    void   setDebugEnabled(bool enabled) { showDebug = enabled; }
    bool   isDebugEnabled() const { return showDebug; }    
    void   offset_vector_f64(double* __restrict__ vec, const double offset, const int N);
    void   set_vector_f64(double* __restrict__ vec, const double val, const int N);
    void   sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2, const int N);
    void   sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const int N);
    double sum_vector_elements_f64_base(const double* __restrict__ vec, const int N);
    double sum_vector_elements_f64(const double* __restrict__ vec, const int N);
    void   copy_vector_f64(double* __restrict__ dest, const double* __restrict__ source, const int N);

    void   sparse_scaadd(double*     __restrict__ vout,
                         const double  dMULT,
                         const uint* __restrict__ I1, const size_t N1S, const size_t N1L,
                         const uint* __restrict__ I2, const size_t N2S, const size_t N2L,
                         const uint* __restrict__ IM, const size_t NMS, const size_t NML,
                         const double  mu,
                         const double  sig_inv,
                         const int     N);
    double sparse_dotprod(const double* __restrict__ vin1,
                          const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                          const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                          const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                          const double               mu, 
                          const double               sig_inv,
                          const int                  N,
                          const int                  marker);

#ifdef USE_MPI
    void   init_from_restart(const int K, const uint M, const uint Mtot, const uint Ntotc,
                             const int* MrankS, const int* MrankL);
    void   init_from_scratch();

    string mpi_get_sparse_output_filebase(const int rank);
    void   write_sparse_data_files(const uint bpr);
    int    checkRamUsage();
    uint   set_Ntot(const int rank);
    uint   set_Mtot(const int rank);
    int    runMpiGibbs();
    int    runMpiGibbsMultiTraits();
    void   check_whole_array_was_set(const uint* array, const size_t size, const int linenumber, const char* filename);
    void   mpi_assign_blocks_to_tasks(const uint numBlocks, const vector<int> blocksStarts, const vector<int> blocksEnds, const uint Mtot, const int nranks, const int rank, int* MrankS, int* MrankL, int& lmin, int& lmax);
    void   mpi_define_blocks_of_markers(const int Mtot, int* MrankS, int* MrankL, const uint nblocks);


#endif
    
private:

};

#endif /* SRC_BAYESRRM_H_ */
