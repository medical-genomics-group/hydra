/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#ifndef SRC_BAYESW_HPP_
#define SRC_BAYESW_HPP_

#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"

#include <Eigen/Eigen>


// Three structures for ARS
struct pars{
	/* Common parameters for the densities */
	VectorXd epsilon;			// epsilon per subject (before each sampling, need to remove the effect of the sampled parameter and then carry on
	VectorXd mixture_classes; // Vector to store mixture component C_k values
	int used_mixture; //Write the index of the mixture we decide to use
	/* Store the current variables */
	double alpha;

	/* Beta_j - specific variables */
	VectorXd X_j;

	/*  of sum(X_j*failure) */
	double sum_failure;

	/* Mu-specific variables */
	double sigma_mu;
	/* sigma_b-specific variables */
	double alpha_sigma, beta_sigma;

	/* Number of events (sum of failure indicators) */
	double d;

	/* Help variable for storing sqrt(2sigma_b)	 */
	double sqrt_2sigmab;

};

struct pars_beta_sparse{
	/* Common parameters for the densities */

	VectorXd mixture_classes; // Vector to store mixture component C_k values

	int used_mixture; //Write the index of the mixture we decide to use

	/* Store the current variables */
	double alpha, sigma_b;

	/* Beta_j - specific variables */
	double vi_0, vi_1, vi_2; // Sums of vi elements

	// Mean, std dev and their ratio for snp j
	double mean, sd, mean_sd_ratio;

	/*  of sum(X_j*failure) */
	double sum_failure;

	/* Number of events (sum of failure indicators) */
	double d;

};

struct pars_alpha{
	VectorXd failure_vector;
	VectorXd epsilon;			// epsilon per subject (before each sampling, need to remove the effect of the sampled parameter and then carry on

	/* Alpha-specific variables */
	double alpha_0, kappa_0;  /*  Prior parameters */

	/* Number of events (sum of failure indicators) */
	double d;
};


class BayesW
{
    
 public:

    Data               &data;            // data matrices
    Options            &opt;
    const string       bedFile;          // bed file
    const long         memPageSize;      // size of memory
    const unsigned int seed;
    const unsigned int max_iterations;
    const unsigned int burn_in;
    const int       thinning = 1;
    const double	alpha_0  = 0.01;
    const double	kappa_0     = 0.01;
    const double    sigma_mu    = 100;
    const double    alpha_sigma  = 1;
    const double    beta_sigma   = 0.0001;
    const string 	quad_points = opt.quad_points;  // Number of Gaussian quadrature points
    const int 		K = opt.S.size()+1;  //number of mixtures + 0 class 

    const size_t       LENBUF  = 300;


    Distributions_boost dist;
    Distributions_boost dist8[8];
    bool usePreprocessedData;
    bool showDebug;
    double betasqn;

	// The ARS structures
	struct pars used_data;
	struct pars_beta_sparse used_data_beta;
	struct pars_alpha used_data_alpha;


    int      m0;        // total number of markers in model
    //VectorXd v;         // variable storing the component assignment
    VectorXd cass;      // variable storing the component assignment //EO RENAMING
    MatrixXd cass8;

	// Component variables
	VectorXd pi_L;        // mixture probabilities
	VectorXd marginal_likelihoods;      // likelihood for each mixture component
	VectorXd v;         // variable storing the component assignment
	VectorXi components; // Indicator vector stating to which mixture SNP belongs to

/*  double mu8[8];
    double sigmaG8[8];
    double sigmaE8[8];
    double sigmaF8[8];
*/
    // Linear model variables
    VectorXd Beta;       // effect sizes
    VectorXd theta;
    VectorXd vi;
//    VectorXd y_tilde;    // variable containing the adjusted residuals to exclude the effects of a given marker
    VectorXd epsilon;    // variable containing the residuals
    MatrixXi components8;

    VectorXd y;
	VectorXd sum_failure;
	VectorXd sum_failure_fix;
    
    //Sampled variables that are not kept in structure
    double mu;

    //double epsilonsum;
    //double ytildesum;

    BayesW(Data &data, Options &opt, const long memPageSize);
    virtual ~BayesW();

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
			 const double mu_sig_ratio,
                         const int     N);
    double sparse_dotprod(const double* __restrict__ vin1,
                          const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                          const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                          const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                          const double               mu, 
                          const double               sig_inv,
                          const int                  N,
                          const int                  marker);

    int    runGibbs();

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
    void init(unsigned int markerCount, unsigned int individualCount, unsigned int fixedCount);
	void sampleMu();
	void sampleTheta(int fix_i);
	void sampleBeta(int marker);
	void sampleAlpha();

	void marginal_likelihood_vec_calc(VectorXd prior_prob, VectorXd &post_marginals, string n, double vi_sum, double vi_2,double vi_1, double vi_0,double mean, double sd, double mean_sd_ratio);
		double gauss_hermite_adaptive_integral(int k, double sigma, string n, double vi_sum, double vi_2, double vi_1, double vi_0, double mean, double sd, double mean_sd_ratio);

	VectorXd getSnpData(unsigned int marker) const;
    void     printDebugInfo() const;
};

#endif /* SRC_BAYESW_HPP_ */
