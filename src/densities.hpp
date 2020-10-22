#ifndef HYDRA_DENSITIES_HPP_
#define HYDRA_DENSITIES_HPP_

#include "constants.hpp"



//inline 
double mu_dens(double x, void *norm_data)
{
	double y;
    
    pars p = *(static_cast<pars *>(norm_data));

	return -p.alpha * x * p.d + expmEuMasc * ( 
	-(( (p.epsilon).array()  - x) * p.alpha).exp().sum() 
	-(( (p.epsilon2).array()  - x) * p.alpha).exp().sum()  
	+(( (p.epsilon4).array()  - x) * p.alpha).exp().sum() 
	-(( (p.epsilon3).array()  - x) * p.alpha).exp().sum() )	- x * x / (2.0 * p.sigma_mu);
};


// Function for the log density of some "fixed" covariate effect
//inline 
double gamma_dens(double x, void *norm_data)
{
	double y;

	pars p = *(static_cast<pars *>(norm_data));

	// Cast voided pointer into pointer to struct norm_parm
    // Prior is the same currently for intercepts and fixed effects
	return -p.alpha * x * p.sum_failure - (((p.epsilon -  p.X_j * x)* p.alpha).array() - EuMasc).exp().sum() - x*x/(2*p.sigma_covariate);

};


// Function for the log density of alpha
// We are sampling alpha (denoted by x here)
//inline
double alpha_dens(double x, void *norm_data) {

	pars_alpha p = *(static_cast<pars_alpha *>(norm_data));
	return (p.alpha_0 + p.d - 1) * log(x) - p.kappa_0 * x   
	+ x * ( (p.epsilon.array() * p.failure_vector.array()).sum() )
	+ x * ( (p.epsilon2.array() * p.failure_vector2.array()).sum() )
	+ expmEuMasc* ( -(x*p.epsilon.array()).exp().sum() - (x*p.epsilon2.array()).exp().sum() + (x*p.epsilon4.array()).exp().sum() - (x*p.epsilon3.array()).exp().sum() );
};


// Sparse version for function for the log density of beta: uses mixture
// component from the structure norm_data
// We are sampling beta (denoted by x here)
//inline

double beta_dens(double x, void *norm_data) {
    pars_beta_sparse p = *(static_cast<pars_beta_sparse *>(norm_data));
	double H,s, exp_sums;
	H=(-p.alpha * x / p.sd).exp();
	if (p.beta_ind==1){
		if (p.mixture_value_other!=0) {
			s= p.rho*p.sigmaG1 * (p.mixture_value).sqrt() * p.beta_other / p.sigmaG2 / (p.mixture_value_other).sqrt();
		} 
		else {
			s=0;
		}
		exp_sums= expmEuMasc * H.pow(-p.mean) (p.vi_0 + p.vi_tau_0 + H *( p.vi_1 + p.vi_tau_1 + H * (p.vi_2 + p.vi_tau_2) ) ); 
		return -p.alpha * x * p.sum_failure - exp_sums -(x - s)*(x - s) / (2 * p.mixture_value * p.sigmaG1* (1- p.rho*p.rho));
	} else if (p.beta_ind==2) {
		if (p.mixture_value_other!=0) {
			s= p.rho*p.sigmaG2 * (p.mixture_value).sqrt() * p.beta_other / p.sigmaG1 / (p.mixture_value_other).sqrt();
		} 
		else {
			s=0;
		}
		exp_sums= expmEuMasc * H.pow(-p.mean) (p.vi_0 - p.vi_tau_0 + H *( p.vi_1 - p.vi_tau_1 + H * (p.vi_2 - p.vi_tau_2) ) );
		return -p.alpha * x * p.sum_failure - exp_sums -(x - s)*(x - s) / (2 * p.mixture_value * p.sigmaG2* (1- p.rho*p.rho));
	}
	else {
		throw std::invalid_argument( "b_ind must be either 1 or 2" );
	}
	/*return -p.alpha * x * p.sum_failure -
        exp(p.alpha*x*p.mean_sd_ratio)* (p.vi_0 + p.vi_1 * exp(-p.alpha*x/p.sd) + p.vi_2 * exp(-2.0 * p.alpha * x / p.sd))
        -x * x / (2 * p.mixture_value * p.sigmaG);*/
};

#endif
