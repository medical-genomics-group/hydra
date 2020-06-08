#ifndef HYDRA_DENSITIES_HPP_
#define HYDRA_DENSITIES_HPP_

#include "constants.hpp"



//inline 
double mu_dens(double x, void *norm_data)
{
	double y;
    
    pars p = *(static_cast<pars *>(norm_data));

	return -p.alpha * x * p.d - (( (p.epsilon).array()  - x) * p.alpha - EuMasc).exp().sum() - x * x / (2.0 * p.sigma_mu);
};


// Function for the log density of some "fixed" covariate effect (gamma denoted by x here
//inline
double gamma_dens2(double x, void *norm_data) {

    double sum = 0.0;

    pars p = *(static_cast<pars *>(norm_data));

#ifdef __INTEL_COMPILER
    __assume_aligned(&p.epsilon, 64);
    __assume_aligned(&p.X_j, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum)
#endif
    for (size_t i=0; i < p.epsilon.size(); i++) {
        sum += exp((p.epsilon[i] - p.X_j[i] * x)* p.alpha - EuMasc);
    }

    /* cast voided pointer into pointer to struct norm_parm */
    return -p.alpha * x * p.sum_failure - sum - x*x/(2*p.sigma_mu); // Prior is the same currently for intercepts and fixed effects
};


// Function for the log density of some "fixed" covariate effect
//inline 
double gamma_dens(double x, void *norm_data)
{
	double y;

	pars p = *(static_cast<pars *>(norm_data));

	// Cast voided pointer into pointer to struct norm_parm
    // Prior is the same currently for intercepts and fixed effects
	return -p.alpha * x * p.sum_failure - (((p.epsilon -  p.X_j * x)* p.alpha).array() - EuMasc).exp().sum() - x*x/(2*p.sigma_mu);

};


// Function for the log density of alpha
// We are sampling alpha (denoted by x here)
//inline
double alpha_dens(double x, void *norm_data) {

	pars_alpha p = *(static_cast<pars_alpha *>(norm_data));
	return (p.alpha_0 + p.d - 1) * log(x) + x * ((p.epsilon.array() * p.failure_vector.array()).sum() - p.kappa_0) -
        ((p.epsilon * x).array() - EuMasc).exp().sum() ;
};


// Sparse version for function for the log density of beta: uses mixture
// component from the structure norm_data
// We are sampling beta (denoted by x here)
//inline
double beta_dens(double x, void *norm_data) {

    pars_beta_sparse p = *(static_cast<pars_beta_sparse *>(norm_data));

	return -p.alpha * x * p.sum_failure -
        exp(p.alpha*x*p.mean_sd_ratio)* (p.vi_0 + p.vi_1 * exp(-p.alpha*x/p.sd) + p.vi_2 * exp(-2.0 * p.alpha * x / p.sd))
        -x * x / (2 * p.mixture_value * p.sigmaG);
};


#endif
