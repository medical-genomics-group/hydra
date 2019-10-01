#ifndef SRC_BAYESRRM_MT_H_
#define SRC_BAYESRRM_MT_H_

#include "BayesRRm.h"
#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"

#include <Eigen/Eigen>

class BayesRRm_mt : public BayesRRm
{
    /*
    Data               &data;            // data matrices
    Options            &opt;
    const string       bedFile;          // bed file
    const long         memPageSize;      // size of memory
    const string       outputFile;
    const unsigned int seed;
    const unsigned int max_iterations;
    const unsigned int burn_in;
    const double	   sigma0  = 0.0001;
    const double	   v0E     = 0.0001;
    const double       s02E    = 0.0001;
    const double       v0G     = 0.0001;
    const double       s02G    = 0.0001;
    const double       s02F    = 1.0;
    const size_t       LENBUF  = 300;

    VectorXd    cva;

    Distributions_boost dist;
    Distributions_boost dist8[8];
    bool usePreprocessedData;
    bool showDebug;
    double betasqn;

    MatrixXd X;         // "fixed effects" matrix.
    VectorXd gamma;     // fixed effects coefficients

    // Component variables
    VectorXd priorPi;   // prior probabilities for each component
    VectorXd pi;        // mixture probabilities
    MatrixXd pi8;
    VectorXd cVa;       // component-specific variance
    VectorXd logL;      // log likelihood of component
    VectorXd muk;       // mean of k-th component marker effect size
    VectorXd denom;     // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
    int      m0;        // total number of markers in model
    //VectorXd v;         // variable storing the component assignment
    VectorXd cass;      // variable storing the component assignment //EO RENAMING
    MatrixXd cass8;
    VectorXd cVaI;      // inverse of the component variances

    // Mean and residual variables
    double mu;          // mean or intercept
    double sigmaG;      // genetic variance
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

    VectorXd         epsilon_restart; //EO to store epsilon found in .eps file when restarting a chain
    std::vector<int> markerI_restart; // idem for markerI found in .mrk file

    VectorXd y;
    VectorXd Cx;

    double epsilonsum;
    double ytildesum;

    uint iteration_restart = 0;
    uint iteration_start   = 0;
    */

public:
 BayesRRm_mt(Data &data, Options &opt, const long memPageSize)
     : BayesRRm(data, opt, memPageSize) 
        {
        }
     virtual ~BayesRRm_mt();

    //int  runGibbs(); // where we run Gibbs sampling over the parametrised model
    //void setDebugEnabled(bool enabled) { showDebug = enabled; }
    //bool isDebugEnabled() const { return showDebug; }
    
    //void sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2,
    //                       const int N);
    //void sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const int N);

     void set_mt_vector_f64(double* __restrict__ vec,
                            const double*        val,
                            const int            NT,
                            const int            N,
                            const bool           interleave);
 
     void sum_mt_vector_elements_f64(const double* __restrict__ vec,
                                     const int                  NT,
                                     const int                  N,
                                     const bool                 interleave,
                                     double*       __restrict__ syt8);
     
     void sparse_dotprod_mt(const double* __restrict__ vin1, const uint8_t* __restrict__ mask,
                            const uint*   __restrict__ I1,   const size_t __restrict__ N1S,  const size_t __restrict__ N1L,
                            const uint*   __restrict__ I2,   const size_t __restrict__ N2S,  const size_t __restrict__ N2L,
                            const uint*   __restrict__ IM,   const size_t NMS,  const size_t NML,
                            const double  mu, const double sig_inv, const int Ntot, const int marker,
                            double* __restrict__ m8, const int NT, const bool interleave);

     void sparse_scaadd_mt(double*       __restrict__ vout,
                           const double* __restrict__ dMULT,
                           const uint*   __restrict__ I1,    const size_t N1S, const size_t N1L,
                           const uint*   __restrict__ I2,    const size_t N2S, const size_t N2L,
                           const uint*   __restrict__ IM,    const size_t NMS, const size_t NML,
                           const double  mu,                 const double sig_inv,
                           const int     Ntot,               const int NT,     const bool interleave);
     
    
#ifdef USE_MPI
    //void   init_from_restart(const int K, const uint Mtot, const uint Ntotc,
    //                         const int* MrankS, const int* MrankL);
    //void   init_from_scratch();
    //string mpi_get_sparse_output_filebase(const int rank);
    //void   write_sparse_data_files(const uint bpr);
    //int    checkRamUsage();
    //uint   set_Ntot(const int rank);
    //uint   set_Mtot(const int rank);
    //int    runMpiGibbs();
    int    runMpiGibbsMultiTraits();
    //void   check_whole_array_was_set(const uint* array, const size_t size, const int linenumber, const char* filename);
#endif

private:
    /*
    void     init(int K, unsigned int markerCount, unsigned int individualCount, unsigned int missingPhenCount);
    VectorXd getSnpData(unsigned int marker) const;
    void     printDebugInfo() const;
    */
};

#endif /* SRC_BAYESRRM_MT_H_ */
