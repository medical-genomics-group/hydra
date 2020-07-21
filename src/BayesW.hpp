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
    const double    sigma_mu     = 100;
    const double    alpha_sigma  = 1;
    const double    beta_sigma   = 0.0001;
    const string 	quad_points  = opt.quad_points;   // Number of Gaussian quadrature points
    unsigned int 	K            = opt.S.size() + 1;  // Number of mixtures + 0 class
    unsigned int    km1          = opt.S.size();      // Number of mixtures 
    const size_t    LENBUF_gamma = 3500;              // Not more than 160 "fixed effects can be used at the moment 

	// The ARS structures
	struct pars used_data;
	struct pars_beta_sparse used_data_beta;
	struct pars_alpha used_data_alpha;

	// Component variables
	MatrixXd pi_L;                   // mixture probabilities
	VectorXd marginal_likelihoods;   // likelihood for each mixture component (for specific group)
    VectorXd marginal_likelihood_0;  // 0th likelihood for each group component
    
    int numGroups;
    VectorXi groups;

    // Linear model variables
    VectorXd vi;
    

    BayesW(Data &data, Options &opt) : BayesRRm(data, opt)
	{
	};

    virtual ~BayesW();

    int runMpiGibbs_bW();
    
private:
    void init(unsigned int individualCount, unsigned int Mtot, unsigned int fixedCount);

	void init_from_restart(const int K, const uint M, const uint  Mtot, const uint Ntot, const uint fixtot,
                           const int* MrankS, const int* MrankL, const bool use_xfiles_in_restart);

	void marginal_likelihood_vec_calc(VectorXd prior_prob,
                                      VectorXd &post_marginals,
                                      string   n,
                                      double   vi_sum,
                                      double   vi_2,
                                      double   vi_1,
                                      double   vi_0,
                                      double   mean,
                                      double   sd,
                                      double   mean_sd_ratio,
                                      unsigned int group_index,
                                      const pars_beta_sparse used_data_beta);
};

#endif /* SRC_BAYESW_HPP_ */
