/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */
 
#ifndef SRC_BAYESW_HPP_
#define SRC_BAYESW_HPP_

#include "BayesRRm.h"
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

	/* Help variable for storing sqrt(2sigmaG)	 */
	//double sqrt_2sigmaG;

};

struct pars_beta_sparse{
	/* Common parameters for the densities */

	VectorXd mixture_classes; // Vector to store mixture component C_k values

	int used_mixture; //Write the index of the mixture we decide to use

	/* Store the current variables */
	double alpha, sigmaG;

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


class BayesW : public BayesRRm
{
public:
    const double	alpha_0  = 0.01;
    const double	kappa_0     = 0.01;
    const double    sigma_mu    = 100;
    const double    alpha_sigma  = 1;
    const double    beta_sigma   = 0.0001;
    const string 	quad_points = opt.quad_points;  // Number of Gaussian quadrature points
    const int 		K = opt.S.size()+1;  //number of mixtures + 0 class
    const size_t       LENBUF_gamma  = 3500; //Not more than 160 "fixed effects can be used at the moment 

	// The ARS structures
	struct pars used_data;
	struct pars_beta_sparse used_data_beta;
	struct pars_alpha used_data_alpha;

	// Component variables
	MatrixXd pi_L;        // mixture probabilities
	VectorXd marginal_likelihoods;      // likelihood for each mixture component (for specific group)
        VectorXd marginal_likelihood_0;      // 0th likelihood for each group component

    int numGroups;
  VectorXi groups;
    // Linear model variables
    VectorXd vi;
    

    //VectorXd y;

     BayesW(Data &data, Options &opt, const long memPageSize) : BayesRRm(data, opt, memPageSize)
	{
	};
    virtual ~BayesW();


#ifdef USE_MPI
    int    runMpiGibbs_bW();
#endif
    
private:
        void init(unsigned int individualCount, unsigned int Mtot, unsigned int fixedCount);
	void init_from_restart(const int K, const uint M, const uint  Mtot, const uint Ntot, const uint fixtot,
                                 const int* MrankS, const int* MrankL, const bool use_xfiles_in_restart);
	void marginal_likelihood_vec_calc(VectorXd prior_prob, VectorXd &post_marginals, string n, double vi_sum, double vi_2,double vi_1, double vi_0,double mean, double sd, double mean_sd_ratio);
	double gauss_hermite_adaptive_integral(int k, double sigma, string n, double vi_sum, double vi_2, double vi_1, double vi_0, double mean, double sd, double mean_sd_ratio);
	double partial_sum(const double* __restrict__ vec, const uint*   __restrict__ IX, const size_t  NXS, const size_t NXL);

//	VectorXd getSnpData(unsigned int marker) const;
//    void     printDebugInfo() const;
};

#endif /* SRC_BAYESW_HPP_ */
