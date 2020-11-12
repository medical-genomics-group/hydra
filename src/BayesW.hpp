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
#include "ars.hpp"
#include <Eigen/Eigen>



class BayesW : public BayesRRm
{
public:
    const double	alpha_0      = 0.01;
    const double	kappa_0      = 0.01;
    const double      sigma_mu     = 100;
    const double      sigma_covariate = 100;

    double    alpha_sigma  = 1;
    double    beta_sigma   = 0.0001;
    const string 	quad_points  = opt.quad_points;   // Number of Gaussian quadrature points
    unsigned int 	K            = opt.S.size() + 1;  // Number of mixtures + 0 class
    unsigned int    km1          = opt.S.size();      // Number of mixtures 
    const size_t    LENBUF_gamma = 3500;              // Not more than 160 "fixed effects can be used at the moment 

    double tau = log(61.0); // Later we need to read from the data what is the breaking point


	// The ARS structures
	struct pars used_data;
	struct pars_beta_sparse used_data_beta;
	struct pars_alpha used_data_alpha;

	// Component variables
	MatrixXd pi_L, pi_L2;                   // mixture probabilities
	VectorXd marginal_likelihoods;   // likelihood for each mixture component (for specific group)
    VectorXd marginal_likelihood_0;  // 0th likelihood for each group component
    VectorXd marginal_likelihood2_0;  // 0th likelihood for each group component
    
    int numGroups;
    VectorXi groups;

    MatrixXi cass2;

    // Linear model variables
    VectorXd vi1;
    VectorXd vi2;
    VectorXd vi3;
    VectorXd vi4;

    VectorXd epsilon2;
    VectorXd epsilon3;
    VectorXd epsilon4;

    //Variables for the second epoch
    VectorXd Beta2;       // effect sizes
    VectorXi components2;
    VectorXd y2;

    VectorXd sigmaG2;    // Genetic variance for epoch 2
    VectorXd Rho;    // Correlation parameters


    BayesW(Data &data, Options &opt) : BayesRRm(data, opt)
	{
	};

    virtual ~BayesW();

    int runMpiGibbs_bW();
    
private:
    void init(unsigned int individualCount, unsigned int individualCount2, unsigned int Mtot, unsigned int fixedCount);

	void init_from_restart(const int K, const uint M, const uint  Mtot, const uint Ntot1, const uint Ntot2, const uint fixtot,
                           const int* MrankS, const int* MrankL, const bool use_xfiles_in_restart);

	void marginal_likelihood_vec_calc(VectorXd prior_prob,
                                      VectorXd &post_marginals,
                                      string   n,
                                      double   vi_sum,
                                      double   vi_2,
                                      double   vi_1,
                                      double   vi_0,
                                      double   vi_tau_sum,
                                      double   vi_tau_2,
                                      double   vi_tau_1,
                                      double   vi_tau_0,
                                      double   mean,
                                      double   sd,
                                      double   mean_sd_ratio,
                                      unsigned int group_index,
                                      const pars_beta_sparse used_data_beta);

 


};

#endif /* SRC_BAYESW_HPP_ */
