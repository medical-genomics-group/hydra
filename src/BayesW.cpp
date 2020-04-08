/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include <cstdlib>
#include "BayesW.hpp"
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
#include <boost/range/algorithm.hpp>
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <mm_malloc.h>
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_utils.hpp"
#endif

#include <omp.h>
#include "BayesW_arms.h"
#include <math.h>

/* Pre-calculate used constants */
#define PI 3.14159265359
#define PI_squared 9.86960440109
#define PI2 6.28318530718
#define sqrtPI 1.77245385090552
#define EuMasc 0.577215664901532

BayesW::~BayesW()
{
}

/* Function that finds the sum across the sum across individuals who have marker with specified value */
double BayesW::partial_sum(const double* __restrict__ vec,
                           const uint*   __restrict__ IX,
                           const size_t               NXS,
                           const size_t               NXL) {
    double sum = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum)
#endif
    for (size_t i=NXS; i < NXS + NXL; i++) {
        sum += vec[IX[i]];
    }
    return sum;
}

/* Function to check if ARS resulted with error*/
inline void errorCheck(int err){
	if(err>0){
		cout << "Error code = " << err << endl;
		exit(1);
	}
}


/* Function for the log density of mu */
inline double mu_dens(double x, void *norm_data)
/* We are sampling mu (denoted by x here) */
{
	double y;

	/* In C++ we need to do a static cast for the void data */
	pars p = *(static_cast<pars *>(norm_data));

	/* cast voided pointer into pointer to struct norm_parm */
	y = - p.alpha * x * p.d - (( (p.epsilon).array()  - x) * p.alpha - EuMasc).exp().sum() - x*x/(2*p.sigma_mu);
	return y;
};


/* Function for the log density of some "fixed" covariate effect */
inline double gamma_dens2(double x, void *norm_data)
/* We are sampling gamma (denoted by x here) */
{
    double y;
    double sum = 0.0;

    /* In C++ we need to do a static cast for the void data */
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
    y = - p.alpha * x * p.sum_failure - sum - x*x/(2*p.sigma_mu); // Prior is the same currently for intercepts and fixed effects
    return y;
};


/* Function for the log density of some "fixed" covariate effect */
inline double gamma_dens(double x, void *norm_data)
/* We are sampling gamma (denoted by x here) */
{
	double y;
	/* In C++ we need to do a static cast for the void data */
	pars p = *(static_cast<pars *>(norm_data));

	/* cast voided pointer into pointer to struct norm_parm */
	y = - p.alpha * x * p.sum_failure - (((p.epsilon -  p.X_j * x)* p.alpha).array() - EuMasc).exp().sum() - x*x/(2*p.sigma_mu); // Prior is the same currently for intercepts and fixed effects
	return y;
};

/* Function for the log density of alpha */
inline double alpha_dens(double x, void *norm_data)
/* We are sampling alpha (denoted by x here) */
{
	double y;

	/* In C++ we need to do a static cast for the void data */
	pars_alpha p = *(static_cast<pars_alpha *>(norm_data));
	y = (p.alpha_0 + p.d - 1) * log(x) + x * ((p.epsilon.array() * p.failure_vector.array()).sum() - p.kappa_0) -
        ((p.epsilon * x).array() - EuMasc).exp().sum() ;
	return y;
};

/* Sparse version for function for the log density of beta: uses mixture component from the structure norm_data */
inline double beta_dens(double x, void *norm_data)
/* We are sampling beta (denoted by x here) */
{
	double y;
	/* In C++ we need to do a static cast for the void data */
	pars_beta_sparse p = *(static_cast<pars_beta_sparse *>(norm_data));

	y = -p.alpha * x * p.sum_failure -
        exp(p.alpha*x*p.mean_sd_ratio)* (p.vi_0 + p.vi_1 * exp(-p.alpha*x/p.sd) + p.vi_2 * exp(-2*p.alpha*x/p.sd))
        -x * x / (2 * p.mixture_value * p.sigmaG) ;
	return y;
};



//The function for integration
inline double gh_integrand_adaptive(double s,double alpha, double dj, double sqrt_2Ck_sigmaG,
                                    double vi_sum, double vi_2, double vi_1, double vi_0, double mean, double sd, double mean_sd_ratio){
	//vi is a vector of exp(vi)
	double temp = -alpha *s*dj*sqrt_2Ck_sigmaG +
        vi_sum - exp(alpha*mean_sd_ratio*s*sqrt_2Ck_sigmaG) *
        (vi_0 + vi_1 * exp(-alpha * s*sqrt_2Ck_sigmaG/sd) + vi_2* exp(-2 * alpha * s*sqrt_2Ck_sigmaG/sd))
        -pow(s,2);
	return exp(temp);
}


//Calculate the value of the integral using Adaptive Gauss-Hermite quadrature
//Let's assume that mu is always 0 for speed
double BayesW::gauss_hermite_adaptive_integral(double C_k, double sigma, string n, double vi_sum, double vi_2, double vi_1, double vi_0,
                                               double mean, double sd, double mean_sd_ratio){

	double temp = 0;
	double sqrt_2ck_sigma = sqrt(2* C_k * used_data_beta.sigmaG);

	if(n == "3"){
		double x1,x2;
		double w1,w2,w3;

		x1 = 1.2247448713916;
		x2 = -x1;

		w1 = 1.3239311752136;
		w2 = w1;

		w3 = 1.1816359006037;

		x1 = sigma*x1;
		x2 = sigma*x2;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                           vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3;
	}
	// n=5
	else if(n == "5"){
		double x1,x2,x3,x4;//x5;
		double w1,w2,w3,w4,w5; //These are adjusted weights

		x1 = 2.0201828704561;
		x2 = -x1;
		w1 = 1.181488625536;
		w2 = w1;

		x3 = 0.95857246461382;
		x4 = -x3;
		w3 = 0.98658099675143;
		w4 = w3;

		//	x5 = 0.0;
		w5 = 0.94530872048294;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		//x5 = sigma*x5;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                           vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 ;//* gh_integrand_adaptive(x5,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j); // This part is just 1
	}else if(n == "7"){
		double x1,x2,x3,x4,x5,x6;
		double w1,w2,w3,w4,w5,w6,w7; //These are adjusted weights

		x1 = 2.6519613568352;
		x2 = -x1;
		w1 = 1.1013307296103;
		w2 = w1;

		x3 = 1.6735516287675;
		x4 = -x3;
		w3 = 0.8971846002252;
		w4 = w3;

		x5 = 0.81628788285897;
		x6 = -x5;
		w5 = 0.8286873032836;
		w6 = w5;

		w7 = 0.81026461755681;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                           vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7;
	}else if(n == "9"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9; //These are adjusted weights

		x1 = 3.1909932017815;
		x2 = -x1;
		w1 = 1.0470035809767;
		w2 = w1;

		x3 = 2.2665805845318;
		x4 = -x3;
		w3 = 0.84175270147867;
		w4 = w3;

		x5 = 1.4685532892167;
		x6 = -x5;
		w5 = 0.7646081250946;
		w6 = w5;

		x7 = 0.72355101875284;
		x8 = -x7;
		w7 = 0.73030245274509;
		w8 = w7;

        //	x9 = 0;
		w9 = 0.72023521560605;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 ;//* gh_integrand_adaptive(x9,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "11"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11; //These are adjusted weights

		x1 = 3.6684708465596;
		x2 = -x1;
		w1 = 1.0065267861724;
		w2 = w1;

		x3 = 2.7832900997817;
		x4 = -x3;
		w3 = 0.802516868851;
		w4 = w3;

		x5 = 2.0259480158258;
		x6 = -x3;
		w5 = 0.721953624728;
		w6 = w5;

		x7 = 1.3265570844949;
		x8 = -x7;
		w7 = 0.6812118810667;
		w8 = w7;

		x9 = 0.6568095668821;
		x10 = -x9;
		w9 = 0.66096041944096;
		w10 = w9;

		//x11 = 0.0;
		w11 = 0.65475928691459;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		//	x11 = sigma*x11;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "13"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13; //These are adjusted weights

		x1 = 4.1013375961786;
		x2 = -x1;
		w1 = 0.97458039564;
		w2 = w1;

		x3 = 3.2466089783724;
		x4 = -x3;
		w3 = 0.7725808233517;
		w4 = w3;

		x5 = 2.5197356856782;
		x6 = -x3;
		w5 = 0.6906180348378;
		w6 = w5;

		x7 = 1.8531076516015;
		x8 = -x7;
		w7 = 0.6467594633158;
		w8 = w7;

		x9 = 1.2200550365908;
		x10 = -x9;
		w9 = 0.6217160552868;
		w10 = w9;

		x11 = 0.60576387917106;
		x12 = -x11;
		w11 = 0.60852958370332;
		w12 = w11;

		//x13 = 0.0;
		w13 = 0.60439318792116;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		x11 = sigma*x11;
		x12 = sigma*x12;


		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "15"){
		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15; //These are adjusted weights

		x1 = 4.4999907073094;
		x2 = -x1;
		w1 = 0.94836897082761;
		w2 = w1;

		x3 = 3.6699503734045;
		x4 = -x3;
		w3 = 0.7486073660169;
		w4 = w3;

		x5 = 2.9671669279056;
		x6 = -x3;
		w5 = 0.666166005109;
		w6 = w5;

		x7 = 2.3257324861739;
		x8 = -x7;
		w7 = 0.620662603527;
		w8 = w7;

		x9 = 1.7199925751865;
		x10 = -x9;
		w9 = 0.5930274497642;
		w10 = w9;

		x11 = 1.1361155852109;
		x12 = -x11;
		w11 = 0.5761933502835;
		w12 = w11;

		x13 = 0.5650695832556;
		x14 = -x13;
		w13 = 0.5670211534466;
		w14 = w13;

		//x15 = 0.0;
		w15 = 0.56410030872642;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		x11 = sigma*x11;
		x12 = sigma*x12;
		x13 = sigma*x13;
		x14 = sigma*x14;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);
	}else if(n == "17"){
        double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;//,x17;
        double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17; //These are adjusted weights

        x1 = 4.8713451936744;
        x2 = -x1;
        w1 = 0.92625413999;
        w2 = w1;

        x3 = 4.0619466758755;
        x4 = -x3;
        w3 = 0.728748370587;
        w4 = w3;

        x5 = 3.3789320911415;
        x6 = -x3;
        w5 = 0.6462917002129;
        w6 = w5;

        x7 = 2.7577629157039;
        x8 = -x7;
        w7 = 0.5998927326678;
        w8 = w7;

        x9 = 2.1735028266666;
        x10 = -x9;
        w9 = 0.5707392941245;
        w10 = w9;

        x11 = 1.6129243142212;
        x12 = -x11;
        w11 = 0.55177735307817;
        w12 = w11;

        x13 = 1.0676487257435;
        x14 = -x13;
        w13 = 0.5397631139085;
        w14 = w13;

        x15 = 0.53163300134266;
        x16 = -x15;
        w15 = 0.5330706545736;
        w16 = w15;

        w17 = 0.53091793762486;

        x1 = sigma*x1;
        x2 = sigma*x2;
        x3 = sigma*x3;
        x4 = sigma*x4;
        x5 = sigma*x5;
        x6 = sigma*x6;
        x7 = sigma*x7;
        x8 = sigma*x8;
        x9 = sigma*x9;
        x10 = sigma*x10;
        x11 = sigma*x11;
        x12 = sigma*x12;
        x13 = sigma*x13;
        x14 = sigma*x14;
        x15 = sigma*x15;
        x16 = sigma*x16;

        temp =  w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 * gh_integrand_adaptive(x15,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w16 * gh_integrand_adaptive(x16,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w17 ;//* gh_integrand_adaptive(0,...)= 1
    }else if(n == "25"){
        double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24;
        double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25; //These are adjusted weights

        x1 = 6.1642724340525;
        x2 = -x1;
        w1 = 0.862401988731;
        w2 = w1;

        x3 = 5.41363635528;
        x4 = -x3;
        w3 = 0.673022290222;
        w4 = w3;

        x5 = 4.7853203673522;
        x6 = -x3;
        w5 = 0.5920816930865;
        w6 = w5;
        x7 = 4.2186094443866;
        x8 = -x7;
        w7 = 0.5449177721944;
        w8 = w7;

        x9 = 3.690282876998;
        x10 = -x9;
        w9 = 0.513655789775;
        w10 = w9;

        x11 = 3.1882949244251;
        x12 = -x11;
        w11 = 0.4915068818876;
        w12 = w11;

        x13 = 2.705320237173;
        x14 = -x13;
        w13 = 0.4752497380022;
        w14 = w13;

        x15 = 2.2364201302673;
        x16 = -x15;
        w15 = 0.463141046575;
        w16 = w15;

        x17 = 1.7780011243372;
        x18 = -x17;
        w17 = 0.45415588552762;
        w18 = w17;

        x19 = 1.3272807020731;
        x20 = -x19;
        w19 = 0.4476612565874;
        w20 = w19;

        x21 = 0.88198275621382;
        x22 = -x21;
        w21 = 0.44325918925185;
        w22 = w21;

        x23 = 0.44014729864531;
        x24 = -x23;
        w23 = 0.44070582891206;
        w24 = w23;
        //x25 = 0.0;
        w25 = 0.43986872216949;

        x1 = sigma*x1;
        x2 = sigma*x2;
        x3 = sigma*x3;
        x4 = sigma*x4;
        x5 = sigma*x5;
        x6 = sigma*x6;
        x7 = sigma*x7;
        x8 = sigma*x8;
        x9 = sigma*x9;
        x10 = sigma*x10;
        x11 = sigma*x11;
        x12 = sigma*x12;
        x13 = sigma*x13;
        x14 = sigma*x14;
        x15 = sigma*x15;
        x16 = sigma*x16;
        x17 = sigma*x17;
        x18 = sigma*x18;
        x19 = sigma*x19;
        x20 = sigma*x20;
        x21 = sigma*x21;
        x22 = sigma*x22;
        x23 = sigma*x23;
        x24 = sigma*x24;

        temp =  w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 * gh_integrand_adaptive(x15,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w16 * gh_integrand_adaptive(x16,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w17 * gh_integrand_adaptive(x17,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w18 * gh_integrand_adaptive(x18,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w19 * gh_integrand_adaptive(x19,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w20 * gh_integrand_adaptive(x20,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w21 * gh_integrand_adaptive(x21,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w22 * gh_integrand_adaptive(x22,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w23 * gh_integrand_adaptive(x23,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w24 * gh_integrand_adaptive(x24,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w25 ;//* gh_integrand_adaptive(0,...)= 1
    }else{
		cout << "Possible number of quad_points = 3,5,7,9,11,13,15,17,25" << endl;
		exit(1);
	}

	return sigma*temp;
}


//Pass the vector post_marginals of marginal likelihoods by reference
void BayesW::marginal_likelihood_vec_calc(VectorXd prior_prob, VectorXd &post_marginals, string n,
                                          double vi_sum, double vi_2, double vi_1, double vi_0, double mean, double sd, double mean_sd_ratio, unsigned int group_index){
	double exp_sum = (vi_1 * (1 - 2 * mean) + 4 * (1-mean) * vi_2 + vi_sum * mean * mean) /(sd*sd) ;

	for(int i=0; i < km1; i++){
		//Calculate the sigma for the adaptive G-H
		double sigma = 1.0/sqrt(1 + used_data_beta.alpha * used_data_beta.alpha * used_data_beta.sigmaG * cVa(group_index,i) * exp_sum);
		post_marginals(i+1) = prior_prob(i+1) * gauss_hermite_adaptive_integral(cVa(group_index,i), sigma, n, vi_sum,  vi_2,  vi_1,  vi_0,  //(i+1) because 0th is already pre-calculated
                                                                                mean, sd, mean_sd_ratio);
	}
}

void BayesW::init(unsigned int individualCount, unsigned int Mtot, unsigned int fixedCount)
{
	// Read the failure indicator vector
	if(individualCount != (data.fail).size()){
		cout << "Number of phenotypes "<< individualCount << " was different from the number of failures " << (data.fail).size() << endl;
		exit(1);
	}

	// Linear model variables
	gamma = VectorXd(fixedCount);

	//phenotype vector
	y = VectorXd();
	//residual vector
	epsilon = VectorXd();

	//vi vector
	vi = VectorXd(individualCount);

	// Resize the vectors in the structure
	used_data.X_j = VectorXd(individualCount);
	used_data.epsilon.resize(individualCount);
	used_data_alpha.epsilon.resize(individualCount);

	//Init the group variables
	 data.groups.resize(Mtot);
	 data.groups.setZero();
	 const int Kt   = cva.size() + 1;			//Temporary K
	 const int Ktm1 = Kt - 1; 

    	 data.mS.resize(Mtot, Ktm1);

	 for (int i=0; i<Mtot; i++)
	   data.mS.row(i) = cva;
	
	 if (opt.groupIndexFile != "" && opt.groupMixtureFile != "") {
	   data.readGroupFile(opt.groupIndexFile);
	   data.readmSFile(opt.groupMixtureFile);
	 }

	 printf("numGroups = %d, data.groups.size() = %lu, Mtot = %d\n", data.numGroups, data.groups.size(), Mtot);

         numGroups = data.numGroups;
    	 K  = int(data.mS.cols()) + 1;  //Mixtures + 0th component. 
  	 km1 = K - 1;		    //Just mixtures
         sigmaG.resize(numGroups);
         sigmaG.setZero();

	 assert(data.groups.size() == Mtot);
	 groups     = data.groups;
	 cVa.resize(numGroups, km1);    // component-specific variance

	 //Populate cVa. We store only km1 values for mixtures
    	 for (int i=0; i < numGroups; i++) {
        	cVa.row(i) = data.mS.row(i).segment(0,km1);
    	 }
	
        // Component variables
        pi_L.resize(numGroups,K);                        // prior mixture probabilities
        marginal_likelihoods = VectorXd(K);  // likelihood for each mixture component

        // Vector to store the 0th component of the marginal likelihood for each group  
        marginal_likelihood_0 = VectorXd(numGroups);

	//set priors for pi parameters
	//Give only the first mixture some initial probability of entering
	pi_L.setConstant(1.0/Mtot);
	pi_L.col(0).array() = 0.99;
	pi_L.col(1).array() = 1 - pi_L.col(0).array() - (km1 - 1)/Mtot;

	marginal_likelihoods.setOnes();   //Initialize with just ones
        marginal_likelihood_0.setOnes();

	Beta.setZero();
	gamma.setZero();

	//initialize epsilon vector as the phenotype vector
	y = data.y.cast<double>().array();

	epsilon = y;
	mu = y.mean();       // mean or intercept
	// Initialize the variables in structures
	//Save variance classes

	//Store the vector of failures only in the structure used for sampling alpha
	used_data_alpha.failure_vector = data.fail.cast<double>();

	double denominator = (6 * ((y.array() - mu).square()).sum()/(y.size()-1));
	used_data.alpha = PI/sqrt(denominator);    // The shape parameter initial value
	used_data_beta.alpha = PI/sqrt(denominator);    // The shape parameter initial value


	for(int i=0; i<(y.size()); ++i){
		(used_data.epsilon)[i] = y[i] - mu ; // Initially, all the BETA elements are set to 0, XBeta = 0
		epsilon[i] = y[i] - mu;
	}
	// Use h2 = 0.5 for the inital estimate// divided  by the number of groups
	sigmaG.array() = PI_squared/ (6 * pow(used_data_beta.alpha,2))/numGroups;

    //Restart variables
    epsilon_restart.resize(individualCount);
    epsilon_restart.setZero();

    gamma_restart.resize(fixedCount);
    gamma_restart.setZero();

    xI_restart.resize(fixedCount);

	/* Prior value selection for the variables */
	/* At the moment we set them to be weakly informative (in .hpp file) */
	/* alpha */
	used_data_alpha.alpha_0 = alpha_0;
	used_data_alpha.kappa_0 = kappa_0;
	/* mu */
	used_data.sigma_mu = sigma_mu;
	/* sigmaG */
	used_data.alpha_sigma = alpha_sigma;
	used_data.beta_sigma = beta_sigma;

	// Save the number of events
	used_data.d = used_data_alpha.failure_vector.array().sum();
	used_data_alpha.d = used_data.d;
}


void BayesW::init_from_restart(const int K, const uint M, const uint  Mtot, const uint Ntot, const uint fixtot,
                               const int* MrankS, const int* MrankL, const bool use_xfiles_in_restart) {
    //Use the regular bW initialisation
    init(Ntot,Mtot, fixtot);    

    //TODO @@@DT change this function to read the csv file from restart in groups 
    data.read_mcmc_output_csv_file_bW(opt.mcmcOut, opt.save, K, mu, sigmaG, used_data.alpha, pi_L, iteration_restart);
    
    // Set new random seed for the ARS in case of restart. In long run we should use dist object for simulating from uniform distribution
    srand(opt.seed + iteration_restart);

    //Carry the values to the other structures
    used_data_beta.alpha = used_data.alpha;

    MPI_Barrier(MPI_COMM_WORLD);

    data.read_mcmc_output_bet_file(opt.mcmcOut, Mtot, iteration_restart, opt.thin,
                                   MrankS, MrankL, use_xfiles_in_restart,
                                   Beta);

    data.read_mcmc_output_cpn_file(opt.mcmcOut, Mtot, iteration_restart, opt.thin,
                                   MrankS, MrankL, use_xfiles_in_restart,
                                   components);

    data.read_mcmc_output_eps_file(opt.mcmcOut, Ntot, iteration_restart,
                                   epsilon_restart);
    
    data.read_mcmc_output_idx_file(opt.mcmcOut, "mrk", M, iteration_restart,
                                   markerI_restart);

    if (opt.covariates) {
        data.read_mcmc_output_gam_file_bW(opt.mcmcOut, opt.save, fixtot, gamma_restart, iteration_restart);

        data.read_mcmc_output_idx_file_bW(opt.mcmcOut, "xiv", fixtot, iteration_restart, xI_restart);
    }

    // Adjust starting iteration number.
    iteration_start = iteration_restart + 1;
             
    MPI_Barrier(MPI_COMM_WORLD);
}
    


//EO: MPI GIBBS
//-------------
int BayesW::runMpiGibbs_bW() {

    //#ifdef _OPENMP
    //#warning "using OpenMP"
    //#endif

	const unsigned int numFixedEffects(data.numFixedEffects);

    char   buff[LENBUF];
    char   buff_gamma[LENBUF_gamma]; 
    int    nranks, rank, name_len, result;
    double dalloc = 0.0;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File   outfh, betfh, epsfh, gamfh, cpnfh, mrkfh, xivfh; 
    MPI_File   xbetfh, xcpnfh;
    MPI_Status status;
    MPI_Info   info;

    // Set up processing options
    // -------------------------
    if (rank < 0) {
        opt.printBanner();
        opt.printProcessingOptions();
    }

    // Set Ntot and Mtot
    // -----------------
    uint Ntot = set_Ntot(rank);
    const uint Mtot = set_Mtot(rank);
    //Reset the dist
    dist.reset_rng((uint)(opt.seed + rank*1000));

	
    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    uint M = MrankL[rank];
    if (rank % 10 == 0) {
        printf("INFO   : rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);
    }


    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: at this stage Ntot is not yet adjusted for missing phenotypes,
    //       hence the correction in the call
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    mpi_define_blocks_of_markers(Ntot - data.numNAs, IrankS, IrankL, nranks);

    Beta.resize(M);
    Beta.setZero();

    components.resize(M);
    components.setZero();

    std::vector<int>    markerI;

    markerI_restart.resize(M);
    std::fill(markerI_restart.begin(), markerI_restart.end(), 0);

    std::vector<int>     mark2sync;
    std::vector<double>  dbet2sync;

    dalloc +=     M * sizeof(int)    / 1E9; // for components
    dalloc += 2 * M * sizeof(double) / 1E9; // for Beta and Acum

    // Adapt the --thin and --save options such that --save >= --thin and --save%--thin = 0
    // ------------------------------------------------------------------------------------
    if (opt.save < opt.thin) {
        opt.save = opt.thin;
        if (rank == 0) printf("WARNING: opt.save was lower that opt.thin ; opt.save reset to opt.thin (%d)\n", opt.thin);
    }
    if (opt.save%opt.thin != 0) {
        if (rank == 0) printf("WARNING: opt.save (= %d) was not a multiple of opt.thin (= %d)\n", opt.save, opt.thin);
        opt.save = int(opt.save/opt.thin) * opt.thin;
        if (rank == 0) printf("         opt.save reset to %d, the closest multiple of opt.thin (%d)\n", opt.save, opt.thin);
    }


    // Invariant initializations (from scratch / from restart)
    // -------------------------------------------------------
    string lstfp = opt.mcmcOut + ".lst";
    string outfp = opt.mcmcOut + ".csv";
    string betfp = opt.mcmcOut + ".bet";
    string xbetfp = opt.mcmcOut + ".xbet";
    string cpnfp = opt.mcmcOut + ".cpn";
    string xcpnfp = opt.mcmcOut + ".xcpn";
    string gamfp = opt.mcmcOut + ".gam";
    string xivfp = opt.mcmcOut + ".xiv";
    string rngfp = opt.mcmcOut + ".rng." + std::to_string(rank);
    string mrkfp = opt.mcmcOut + ".mrk." + std::to_string(rank);
    string epsfp = opt.mcmcOut + ".eps." + std::to_string(rank);

    if(opt.restart){
        init_from_restart(K, M, Mtot, Ntot - data.numNAs, numFixedEffects, MrankS, MrankL, use_xfiles_in_restart);
        if (rank == 0)
            data.print_restart_banner(opt.mcmcOut.c_str(),  iteration_restart, iteration_start);

        dist.read_rng_state_from_file(rngfp);

        // Rename output files so that we do not erase from failed job!
        //EO: add a function, to update both Nam and Dir!
        opt.mcmcOutNam += "_rs";
        opt.mcmcOut = opt.mcmcOutDir + "/" + opt.mcmcOutNam;
        lstfp  = opt.mcmcOut + ".lst";
        outfp  = opt.mcmcOut + ".csv";
        betfp  = opt.mcmcOut + ".bet";
        xbetfp = opt.mcmcOut + ".xbet"; // Last saved iteration of bet; .bet has full history
        cpnfp  = opt.mcmcOut + ".cpn";
        xcpnfp = opt.mcmcOut + ".xcpn"; // Idem
        rngfp  = opt.mcmcOut + ".rng." + std::to_string(rank);
        mrkfp  = opt.mcmcOut + ".mrk." + std::to_string(rank);
        epsfp  = opt.mcmcOut + ".eps." + std::to_string(rank);
        gamfp  = opt.mcmcOut + ".gam";
        xivfp  = opt.mcmcOut + ".xiv";

    }else{
        // Set new random seed for the ARS in case of restart. In long run we should use dist object for simulating from uniform distribution
        srand(opt.seed);

        init(Ntot - data.numNAs, Mtot,numFixedEffects);
    }
    cass.resize(numGroups,K); //rows are groups columns are mixtures
    MatrixXi sum_cass(numGroups,K);  // To store the sum of cass elements over all ranks

   // Build global repartition of markers over the groups
    VectorXi MtotGrp(numGroups);
    MtotGrp.setZero();
    for (int i=0; i < Mtot; i++) {
        MtotGrp[groups[i]] += 1;
    }
    VectorXi m0(numGroups); // non-zero elements per group

    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    xI_restart.resize(data.X.cols());
 
    //    dist.reset_rng((uint)(opt.seed + rank*1000));

    // Build a list of the files to tar
    // --------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    ofstream listFile;
    listFile.open(lstfp);
    listFile << outfp << "\n";
    listFile << betfp << "\n";
    listFile << xbetfp << "\n";
    listFile << cpnfp << "\n";
    listFile << xcpnfp << "\n";
    listFile << gamfp << "\n";
    listFile << xivfp << "\n";
    for (int i=0; i<nranks; i++) {
        listFile << opt.mcmcOut + ".rng." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".mrk." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".eps." + std::to_string(i) << "\n";
    }
    listFile.close();
    MPI_Barrier(MPI_COMM_WORLD);


    // Delete old files (fp appended with "_rs" in case of restart, so that
    // original files are kept untouched) and create new ones
    // --------------------------------------------------------------------
    if (rank == 0) {
        MPI_File_delete(outfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xbetfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xcpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(gamfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xivfp.c_str(), MPI_INFO_NULL);
    }
    MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(mrkfp.c_str(), MPI_INFO_NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    
    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xbetfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xbetfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xcpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xcpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD,  gamfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &gamfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD,  xivfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xivfh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_SELF,  epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  mrkfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &mrkfh), __LINE__, __FILE__);


    // First element of the .bet, .cpn and .acu files is the
    // total number of processed markers
    // -----------------------------------------------------
    MPI_Offset offset = 0;

    if (rank == 0) {
        check_mpi(MPI_File_write_at(betfh,  offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xbetfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh,  offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xcpnfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        // check_mpi(MPI_File_write_at(acufh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();
    
    double tl = -mysecond();

    // Read the data (from sparse representation by default)
    // -----------------------------------------------------
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    N1S = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);
    dalloc += 6.0 * double(M) * sizeof(size_t) / 1E9;


    // Boolean mask for using BED representation or not (SPARSE otherwise)
    // For markers with USEBED == true then the BED representation is 
    // converted on the fly to SPARSE the time for the corresponding marker
    // to be processed
    // --------------------------------------------------------------------
    bool *USEBED;
    USEBED = (bool*)_mm_malloc(M * sizeof(bool), 64);  check_malloc(USEBED, __LINE__, __FILE__);
    for (int i=0; i<M; i++) USEBED[i] = false;
    int nusebed = 0;


    uint *I1, *I2, *IM;
    size_t taskBytes = 0;

    if (opt.readFromBedFile) {
        data.load_data_from_bed_file(opt.bedFile, Ntot, M, rank, MrankS[rank],
                                     N1S, N1L, I1,
                                     N2S, N2L, I2,
                                     NMS, NML, IM,
                                     taskBytes);
    } else {
        string sparseOut = mpi_get_sparse_output_filebase(rank);
        data.load_data_from_sparse_files(rank, nranks, M, MrankS, MrankL, sparseOut,
                                         N1S, N1L, I1,
                                         N2S, N2L, I2,
                                         NMS, NML, IM,
                                         taskBytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tl += mysecond();

    if (rank == 0) {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes  =>  BW = %7.3f GB/s\n", rank, tl, taskBytes, (double)taskBytes * 1E-9 / tl);
        fflush(stdout);
    }


    // Correct each marker for individuals with missing phenotype
    // ----------------------------------------------------------
    if (data.numNAs > 0) {

        if (rank == 0)
            printf("INFO   : applying %d corrections to genotype data due to missing phenotype data (NAs in .phen).\n", data.numNAs);

        data.sparse_data_correct_for_missing_phenotype(N1S, N1L, I1, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(N2S, N2L, I2, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(NMS, NML, IM, M, USEBED);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("INFO   : finished applying NA corrections.\n");

        // Adjust N upon number of NAs
        Ntot -= data.numNAs;
        if (rank == 0 && data.numNAs > 0)
            printf("INFO   : Ntot adjusted by -%d to account for NAs in phenotype file. Now Ntot=%d\n", data.numNAs, Ntot);
    }

    // Compute statistics (from sparse info)
    // -------------------------------------
    //if (rank == 0) printf("INFO   : start computing statistics on Ntot = %d individuals\n", Ntot);
    double dN   = (double) Ntot;
    double dNm1 = (double)(Ntot - 1);
    double *mave, *mstd, *sum_failure, *sum_failure_fix; 

    mave = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mave, __LINE__, __FILE__);
    mstd = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);
    sum_failure = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);

    sum_failure_fix = (double*)_mm_malloc(size_t(numFixedEffects) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);

    dalloc += 2 * size_t(M) * sizeof(double) / 1E9;

    double tmp0, tmp1, tmp2;
    double temp_fail_sum = used_data_alpha.failure_vector.array().sum();
    for (int i=0; i<M; ++i) {
        // For now use the old way to compute means
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (dN - double(NML[i]));        

        tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        //TODO At some point we need to turn sd to 1/sd for speed
        //mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
        mstd[i] = sqrt( (tmp0+tmp1+tmp2)/double(Ntot - 1));

        int temp_sum = 0;
        for(size_t ii = N1S[i]; ii < (N1S[i] + N1L[i]) ; ii++){
            temp_sum += used_data_alpha.failure_vector(I1[ii]);
        }
        for(size_t ii = N2S[i]; ii < (N2S[i] + N2L[i]) ; ii++){
            temp_sum += 2*used_data_alpha.failure_vector(I2[ii]);
        }
        sum_failure[i] = (temp_sum - mave[i] * temp_fail_sum) / mstd[i];

        //printf("marker %6d mean %20.15f, std = %20.15f (%.1f / %.15f)  (%15.10f, %15.10f, %15.10f)\n", i, mave[i], mstd[i], double(Ntot - 1), tmp0+tmp1+tmp2, tmp1, tmp2, tmp0);
    }
    //If there are fixed effects, find the same values for them
    if(opt.covariates){
        for(int fix_i=0; fix_i < numFixedEffects; fix_i++){
            sum_failure_fix[fix_i] = ((data.X.col(fix_i).cast<double>()).array() * used_data_alpha.failure_vector.array()).sum();
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Build list of markers    
    // ---------------------
    for (int i=0; i<M; ++i) markerI.push_back(i);
    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();
    //double *y, *epsilon, *tmpEps, *previt_eps, *deltaEps, *dEpsSum, *deltaSum;
    double *y, *tmpEps, *deltaEps, *dEpsSum, *deltaSum, *epsilon ,*vi , *tmp_vi, *tmpEps_vi, *tmp_deltaEps;
    const size_t NDB = size_t(Ntot) * sizeof(double);
    y          = (double*)_mm_malloc(NDB, 64);  check_malloc(y,          __LINE__, __FILE__);
    epsilon    = (double*)_mm_malloc(NDB, 64);  check_malloc(epsilon,    __LINE__, __FILE__);
    vi    = (double*)_mm_malloc(NDB, 64);  check_malloc(vi,    __LINE__, __FILE__);

    tmpEps_vi    = (double*)_mm_malloc(NDB, 64);  check_malloc(tmpEps_vi,    __LINE__, __FILE__);
    tmp_vi    = (double*)_mm_malloc(NDB, 64);  check_malloc(tmp_vi,    __LINE__, __FILE__);

    tmpEps     = (double*)_mm_malloc(NDB, 64);  check_malloc(tmpEps,     __LINE__, __FILE__);
    //previt_eps = (double*)malloc(NDB);  check_malloc(previt_eps, __LINE__, __FILE__);
    tmp_deltaEps   = (double*)_mm_malloc(NDB, 64);  check_malloc(tmp_deltaEps,   __LINE__, __FILE__);

    deltaEps   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaEps,   __LINE__, __FILE__);
    dEpsSum    = (double*)_mm_malloc(NDB, 64);  check_malloc(dEpsSum,    __LINE__, __FILE__);
    deltaSum   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaSum,   __LINE__, __FILE__);
    dalloc += NDB * 6 / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);

    set_vector_f64(dEpsSum, 0.0, Ntot);

    // Copy, center and scale phenotype observations
    // In bW we are not scaling and centering phenotypes
    if(opt.restart){
        for (int i=0; i<Ntot; ++i){
            epsilon[i] = epsilon_restart[i];
        }
        markerI = markerI_restart;
        if (opt.covariates) {
            for (int i=0; i < numFixedEffects; i++) {
                gamma[i] = gamma_restart[i];
                xI[i]    = xI_restart[i];
            }
        }
    }else{
        for (int i=0; i<Ntot; ++i) y[i] = data.y(i);
        for (int i=0; i<Ntot; ++i)  epsilon[i] = y[i] - mu;
    }
    VectorXd sum_beta_squaredNorm;
    double   beta, betaOld, deltaBeta, p, acum;
    VectorXd beta_squaredNorm;
    size_t   markoff;
    int      marker, cx;

    beta_squaredNorm.resize(numGroups);
    sum_beta_squaredNorm.resize(numGroups);
    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;

    // Main iteration loop
    // -------------------
    //bool replay_it = false;
    double tot_sync_ar1  = 0.0;
    double tot_sync_ar2  = 0.0;
    int    tot_nsync_ar1 = 0;
    int    tot_nsync_ar2 = 0;
    int    *glob_info, *tasks_len, *tasks_dis, *stats_len, *stats_dis;

    if (opt.sparseSync) {
        glob_info  = (int*)    _mm_malloc(size_t(nranks * 2) * sizeof(int),    64);  check_malloc(glob_info,  __LINE__, __FILE__);
        tasks_len  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(tasks_len,  __LINE__, __FILE__);
        tasks_dis  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(tasks_dis,  __LINE__, __FILE__);
        stats_len  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(stats_len,  __LINE__, __FILE__);
        stats_dis  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(stats_dis,  __LINE__, __FILE__);
    }

    //Set iteration_start=0
    for (uint iteration=0; iteration<opt.chainLength; iteration++) {

        double start_it = MPI_Wtime();
        double it_sync_ar1  = 0.0;
        double it_sync_ar2  = 0.0;
        int    it_nsync_ar1 = 0;
        int    it_nsync_ar2 = 0;

        /* 1. Intercept (mu) */
        //Removed sampleMu function on its own 
        int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 4 ;
        int neval;
        double xsamp[0], xcent[10], qcent[10] = {5., 30., 70., 95.};
        double convex = 1.0;
        int dometrop = 0;
        double xprev = 0.0;
        double xinit[4] = {0.95*mu, mu,  1.005*mu, 1.01*mu};     // Initial abscissae
        double *p_xinit = xinit;

        double xl = 2;
        double xr = 5;   //xl and xr and the maximum and minimum values between which we sample

        //Update before sampling
        for(int mu_ind=0; mu_ind < Ntot; mu_ind++){
            (used_data.epsilon)[mu_ind] = epsilon[mu_ind] + mu;// we add to epsilon =Y+mu-X*beta
        }

        // Use ARS to sample mu (with density mu_dens, using parameters from used_data)
        err = arms(xinit,ninit,&xl,&xr,mu_dens,&used_data,&convex,
                   npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);

        errorCheck(err); // If there is error, stop the program
        check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        mu = xsamp[0];   // Save the sampled value
        //Update after sampling
        for(int mu_ind=0; mu_ind < Ntot; mu_ind++){
            epsilon[mu_ind] = (used_data.epsilon)[mu_ind] - mu;// we add to epsilon =Y+mu-X*beta
        }
        ////////// End sampling mu
        /* 1a. Fixed effects (gammas) */
        if(opt.covariates){

            double gamma_old = 0;
            std::shuffle(xI.begin(), xI.end(), dist.rng);    
    		//Use only rank 0 shuffling
            check_mpi(MPI_Bcast(xI.data(), xI.size(), MPI_INT, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
	        MPI_Barrier(MPI_COMM_WORLD);


            for(int fix_i = 0; fix_i < numFixedEffects; fix_i++){
                gamma_old = gamma(xI[fix_i]);

                neval = 0;
                xsamp[0] = 0;
                convex = 1.0;
                dometrop = 0;
                xprev = 0.0;

                xinit[0] = gamma_old - 0.075/30 ;     // Initial abscissae
                xinit[1] = gamma_old; 	
                xinit[2] = gamma_old + 0.075/60;  
                xinit[3] = gamma_old + 0.075/30;  

                xl = gamma_old - 0.075;
                xr = gamma_old + 0.075;			  // Initial left and right (pseudo) extremes

                used_data.X_j = data.X.col(xI[fix_i]).cast<double>();  //Take from the fixed effects matrix
                used_data.sum_failure = sum_failure_fix[xI[fix_i]];


                for(int k = 0; k < Ntot; k++){
                    (used_data.epsilon)[k] = epsilon[k] + used_data.X_j[k] * gamma_old;// we adjust the residual with the respect to the previous gamma value
        	}
                // Sample using ARS
                err = arms(xinit,ninit,&xl,&xr, gamma_dens,&used_data,&convex,
                           npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
                errorCheck(err);

                //Use only rank 0
		        check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
		
                gamma(xI[fix_i]) = xsamp[0];  // Save the new result
                for(int k = 0; k < Ntot; k++){
                    epsilon[k] = (used_data.epsilon)[k] - used_data.X_j[k] * gamma(xI[fix_i]);// we adjust the residual with the respect to the previous gamma value
                }
		        MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        ////////// End sampling gamma
        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------

        // ARS parameters
        neval = 0;
        xsamp[0] = 0;
        convex = 1.0;
        dometrop = 0;
        xprev = 0.0;
        xinit[0] = (used_data.alpha)*0.5;     // Initial abscissae
        xinit[1] =  used_data.alpha;
        xinit[2] = (used_data.alpha)*1.05;
        xinit[3] = (used_data.alpha)*1.10;

        // Initial left and right (pseudo) extremes
        xl = 0.0;
        xr = 40.0;

        //Give the residual to alpha structure
        //used_data_alpha.epsilon = epsilon;
        for(int alpha_ind=0; alpha_ind < Ntot; alpha_ind++){
            (used_data_alpha.epsilon)[alpha_ind] = epsilon[alpha_ind];
        }

        //Sample using ARS
        err = arms(xinit,ninit,&xl,&xr,alpha_dens,&used_data_alpha,&convex,
                   npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
        errorCheck(err);
        check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        used_data.alpha = xsamp[0];
        used_data_beta.alpha = xsamp[0];

        MPI_Barrier(MPI_COMM_WORLD);


        // Calculate the vector of exponent of the adjusted residuals
        for(int i = 0; i < Ntot; ++i){
            vi[i] = exp(used_data.alpha * epsilon[i] - EuMasc);
        }

        if (opt.shuffleMarkers) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
        }
        m0.array() = 0;
	cass.setZero();

        for (int i=0; i<Ntot; ++i) tmpEps[i] = epsilon[i];

        double cumSumDeltaBetas = 0.0;
        double task_sum_abs_deltabeta = 0.0;
        int    sinceLastSync    = 0;
     
        // First element for the marginal likelihoods is always is pi_0 *sqrt(pi) for
        //marginal_likelihoods(0) = pi_L(0) * sqrtPI;  
	//Precalculate the product already before for each group
	for(int gg = 0; gg < numGroups; gg++){
		marginal_likelihood_0(gg) = pi_L(gg,0) * sqrtPI ;
	}
	//Set the sum of beta squared 0
        beta_squaredNorm.setZero();

        // Loop over (shuffled) markers
        // ----------------------------
        for (int j = 0; j < lmax; j++) {
            sinceLastSync += 1; 
            
            if (j < M) {
                marker  = markerI[j];
                beta =  Beta(marker);

		unsigned int cur_group = groups[MrankS[rank] + marker];
                /////////////////////////////////////////////////////////
                //Replace the sampleBeta function with the inside of the function        
                double vi_sum = 0.0;
                double vi_1 = 0.0;
                double vi_2 = 0.0;

                used_data_beta.sigmaG = sigmaG[cur_group];

		marginal_likelihoods(0) = marginal_likelihood_0(cur_group);  //Each group has now different marginal likelihood at 0

                //Change the residual vector only if the previous beta was non-zero
                if(Beta(marker) != 0){
                    //Calculate the change in epsilon if we remove the previous marker effect (-Beta(marker))
                    set_vector_f64(tmp_deltaEps, 0.0, Ntot);
                    sparse_scaadd(tmp_deltaEps, Beta(marker),
                                  I1, N1S[marker], N1L[marker],
                                  I2, N2S[marker], N2L[marker],
                                  IM, NMS[marker], NML[marker],
                                  mave[marker], 1/mstd[marker] , Ntot);
                    //Create the temporary vector to store the vector without the last Beta(marker)
                    sum_vectors_f64(tmpEps_vi, epsilon, tmp_deltaEps,  Ntot);
                    //Also find the transformed residuals
                    for(uint i=0; i<Ntot; ++i){
                        tmp_vi[i] = exp(used_data.alpha * tmpEps_vi[i] - EuMasc);
                    }
                    vi_sum = sum_vector_elements_f64(tmp_vi, Ntot);
                    vi_2 = partial_sum(tmp_vi, I2, N2S[marker], N2L[marker]);
                    vi_1 = partial_sum(tmp_vi, I1, N1S[marker], N1L[marker]);

                }else{
                    // Calculate the sums of vi elements
                    vi_sum = sum_vector_elements_f64(vi, Ntot);
                    vi_2 = partial_sum(vi, I2, N2S[marker], N2L[marker]);
                    vi_1 = partial_sum(vi, I1, N1S[marker], N1L[marker]);

                }

                double vi_0 = vi_sum - vi_1 - vi_2;

                /* Calculate the mixture probability */
                double p = dist.unif_rng();  //Generate number from uniform distribution (for sampling from categorical distribution)    
 
                // Calculate the (ratios of) marginal likelihoods
                used_data_beta.sum_failure = sum_failure[marker];
                marginal_likelihood_vec_calc(pi_L.row(cur_group) , marginal_likelihoods, quad_points, vi_sum, vi_2, vi_1, vi_0,
                                             mave[marker],mstd[marker], mave[marker]/mstd[marker], cur_group);

                // Calculate the probability that marker is 0
                double acum = marginal_likelihoods(0)/marginal_likelihoods.sum();

                //Loop through the possible mixture classes
                for (int k = 0; k < K; k++) {
                    if (p <= acum) {
                        //if zeroth component
                        if (k == 0) {
                            Beta(marker) = 0;
                            cass(cur_group, 0) += 1;
                            components[marker]  = k;

                        }
                        // If is not 0th component then sample using ARS
                        else {
                            //used_data_beta.sum_failure = sum_failure(marker);
                            used_data_beta.mean = mave[marker];
                            used_data_beta.sd = mstd[marker];
                            used_data_beta.mean_sd_ratio = mave[marker]/mstd[marker];
                            //used_data_beta.used_mixture = k-1;
			    used_data_beta.mixture_value = cVa(cur_group, k-1); //k-1 because cVa stores only non-zero in bW

                            used_data_beta.vi_0 = vi_0;
                            used_data_beta.vi_1 = vi_1;
                            used_data_beta.vi_2 = vi_2;

                           // double safe_limit = 2 * sqrt(used_data_beta.sigmaG * used_data_beta.mixture_classes(k-1));
                            double safe_limit = 2 * sqrt(used_data_beta.sigmaG * used_data_beta.mixture_value); // Need to think is this safe enough if we have groups
		 	    // ARS parameters
                            neval = 0;
                            xsamp[0] = 0;
                            convex = 1.0;
                            dometrop = 0;
                            xprev = 0.0;
                            xinit[0] = Beta(marker) - safe_limit/10;     // Initial abscissae
                            xinit[1] = Beta(marker);
                            xinit[2] = Beta(marker) + safe_limit/20;
                            xinit[3] = Beta(marker) + safe_limit/10;
		        
                            // Initial left and right (pseudo) extremes
                            xl = Beta(marker) - safe_limit  ; //Construct the hull around previous beta value
                            xr = Beta(marker) + safe_limit;
                            // Sample using ARS
                            err = arms(xinit,ninit,&xl,&xr,beta_dens,&used_data_beta,&convex,
                                       npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
	                        errorCheck(err);

                            Beta(marker) = xsamp[0];  // Save the new result

                            cass(cur_group, k) += 1;
                            components[marker] = k;
			    // Write the sum of the beta squared to the vector
		            beta_squaredNorm[groups[MrankS[rank] + marker]] += Beta[marker] * Beta[marker];

                        }
                        break;
                    } else {
                        if((k+1) == km1){
                            acum = 1; // In the end probability will be 1
                        }else{
                            acum += marginal_likelihoods(k+1)/marginal_likelihoods.sum();
                        }
                    }
                }

                betaOld   = beta;
                beta      = Beta(marker);
                deltaBeta = betaOld - beta;
                //printf("deltaBeta = %15.10f\n", deltaBeta);

                // Compute delta epsilon
                if (deltaBeta != 0.0) {
                    //printf("it %d, task %3d, marker %5d has non-zero deltaBeta = %15.10f (%15.10f, %15.10f) => %15.10f) 1,2,M: %lu, %lu, %lu\n", iteration, rank, marker, deltaBeta, mave[marker], mstd[marker],  deltaBeta * mstd[marker], N1L[marker], N2L[marker], NML[marker]);

                    if (opt.sparseSync && nranks > 1) {

                        mark2sync.push_back(marker);
                        dbet2sync.push_back(deltaBeta);

                    } else {
                        sparse_scaadd(deltaEps, deltaBeta, 
                                      I1, N1S[marker], N1L[marker],
                                      I2, N2S[marker], N2L[marker],
                                      IM, NMS[marker], NML[marker],
                                      mave[marker], 1/mstd[marker] , Ntot); //Use here 1/sd
                        
                        // Update local sum of delta epsilon
                        sum_vectors_f64(dEpsSum, deltaEps, Ntot);
                    }
                }	
            }

                        

            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else {
                //cout << "rank " << rank << " with M=" << M << " waiting for " << lmax << endl;
                deltaBeta = 0.0;
                
                set_vector_f64(deltaEps, 0.0, Ntot);
            }

            task_sum_abs_deltabeta += fabs(deltaBeta);

            // Check whether we have a non-zero beta somewhere
            //if (nranks > 1 && (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1)) {
            if (nranks > 1 && (sinceLastSync >= opt.syncRate || j == lmax-1)) {    
                //MPI_Barrier(MPI_COMM_WORLD);
                double tb = MPI_Wtime();                
                check_mpi(MPI_Allreduce(&task_sum_abs_deltabeta, &cumSumDeltaBetas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

                double te = MPI_Wtime();
                tot_sync_ar1  += te - tb;
                it_sync_ar1   += te - tb;
                tot_nsync_ar1 += 1;
                it_nsync_ar1  += 1;

            } else {
                cumSumDeltaBetas = task_sum_abs_deltabeta;
            }
            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, beta, cumSumDeltaBetas);

            //         if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1) && cumSumDeltaBetas != 0.0) {
            if ( cumSumDeltaBetas != 0.0 && (sinceLastSync >= opt.syncRate || j == lmax-1)) {

                // Update local copy of epsilon
                //MPI_Barrier(MPI_COMM_WORLD);

                if (nranks > 1) {
                    double tb = MPI_Wtime();
                    
                    // Sparse synchronization
                    // ----------------------
                    if (opt.sparseSync) {
                            
                        uint task_m2s = (uint) mark2sync.size();
                        
                        // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
                        double* task_stat = (double*) _mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
                        check_malloc(task_stat, __LINE__, __FILE__);
                        
                        // Compute total number of elements to be sent by each task
                        uint task_size = 0;
                        for (int i=0; i<task_m2s; i++) {
                            task_size += (N1L[ mark2sync[i] ] + N2L[ mark2sync[i] ] + NML[ mark2sync[i] ] + 3);
                            task_stat[2 * i + 0] = mave[ mark2sync[i] ];
                            task_stat[2 * i + 1] = mstd[ mark2sync[i] ] * dbet2sync[i]; //CHANGE mstd later!
                            //printf("Task %3d, m2s %d/%d: 1: %8lu, 2: %8lu, m: %8lu, info: 3); stats are (%15.10f, %15.10f)\n", rank, i, task_m2s, N1L[ mark2sync[i] ], N2L[ mark2sync[i] ], NML[ mark2sync[i] ], task_stat[2 * i + 0], task_stat[2 * i + 1]);
                        }
                        //printf("Task %3d final task_size = %8d elements to send from task_m2s = %d markers to sync.\n", rank, task_size, task_m2s);
                        //fflush(stdout);
                        
                        // Get the total numbers of markers and corresponding indices to gather
                        
                        const int NEL = 2;
                        uint task_info[NEL] = {};                        
                        task_info[0] = task_m2s;
                        task_info[1] = task_size;
                        
                        check_mpi(MPI_Allgather(task_info, NEL, MPI_UNSIGNED, glob_info, NEL, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);
                        
                        int tdisp_ = 0, sdisp_ = 0, glob_m2s = 0, glob_size = 0;
                        for (int i=0; i<nranks; i++) {
                            tasks_len[i]  = glob_info[2 * i + 1];
                            tasks_dis[i]  = tdisp_;
                            tdisp_       += tasks_len[i];
                            stats_len[i]  = glob_info[2 * i] * 2;
                            stats_dis[i]  = sdisp_;
                            sdisp_       += glob_info[2 * i] * 2;
                            glob_size    += tasks_len[i];
                            glob_m2s     += glob_info[2 * i];
                        }
                        //printf("glob_info: markers to sync: %d, with glob_size = %7d elements (sum of all task_size)\n", glob_m2s, glob_size);
                        //fflush(stdout);
                        

                        // Build task's array to spread: | marker 1                             | marker 2
                        //                               | n1 | n2 | nm | data1 | data2 | datam | n1 | n2 | nm | data1 | ...
                        // -------------------------------------------------------------------------------------------------
                        uint* task_dat = (uint*) _mm_malloc(size_t(task_size) * sizeof(uint), 64);
                        check_malloc(task_dat, __LINE__, __FILE__);
                        
                        int loc = 0;
                        for (int i=0; i<task_m2s; i++) {
                            task_dat[loc] = N1L[ mark2sync[i] ];                 loc += 1;
                            task_dat[loc] = N2L[ mark2sync[i] ];                 loc += 1;
                            task_dat[loc] = NML[ mark2sync[i] ];                 loc += 1;
                            for (uint ii = 0; ii < N1L[ mark2sync[i] ]; ii++) {
                                task_dat[loc] = I1[ N1S[ mark2sync[i] ] + ii ];  loc += 1;
                            }
                            for (uint ii = 0; ii < N2L[ mark2sync[i] ]; ii++) {
                                task_dat[loc] = I2[ N2S[ mark2sync[i] ] + ii ];  loc += 1;
                            }
                            for (uint ii = 0; ii < NML[ mark2sync[i] ]; ii++) {
                                task_dat[loc] = IM[ NMS[ mark2sync[i] ] + ii ];  loc += 1;
                            }
                        }                        
                        assert(loc == task_size);
                            
                        // Allocate receive buffer for all the data
                        uint* glob_dat = (uint*) _mm_malloc(size_t(glob_size) * sizeof(uint), 64);
                        check_malloc(glob_dat, __LINE__, __FILE__);
                        
                        check_mpi(MPI_Allgatherv(task_dat, task_size, MPI_UNSIGNED,
                                                 glob_dat, tasks_len, tasks_dis, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);
                        _mm_free(task_dat);
                        
                        double* glob_stats = (double*) _mm_malloc(size_t(glob_size * 2) * sizeof(double), 64);
                        check_malloc(glob_stats, __LINE__, __FILE__);
                        
                        check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                                 glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD), __LINE__, __FILE__);                        
                        _mm_free(task_stat);
                        
                         
                        // Compute global delta epsilon deltaSum
                        size_t loci = 0;
                        for (int i=0; i<glob_m2s ; i++) {
                            
                            //printf("m2s %d/%d (loci = %d): %d, %d, %d\n", i, glob_m2s, loci, glob_dat[loci], glob_dat[loci + 1], glob_dat[loci + 2]);
                            
                            double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                            //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1]);
                            
                            // Set all to 0 contribution
                            if (i == 0) {
                                set_vector_f64(deltaSum, lambda0, Ntot);
                            } else {
                                offset_vector_f64(deltaSum, lambda0, Ntot);
                            }
                            
                            // M -> revert lambda 0 (so that equiv to add 0.0)
                            size_t S = loci + (size_t) (3 + glob_dat[loci] + glob_dat[loci + 1]);
                            size_t L = glob_dat[loci + 2];
                            //cout << "task " << rank << " M: start = " << S << ", len = " << L <<  endl;
                            sparse_add(deltaSum, -lambda0, glob_dat, S, L);
                            
                            // 1 -> add dbet * sig * ( 1.0 - mu)
                            double lambda = glob_stats[2 * i + 1] * (1.0 - glob_stats[2 * i]);
                            //printf("1: lambda = %15.10f, l-l0 = %15.10f\n", lambda, lambda - lambda0);
                            S = loci + 3;
                            L = glob_dat[loci];
                            //cout << "1: start = " << S << ", len = " << L <<  endl;
                            sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);
                            
                            // 2 -> add dbet * sig * ( 2.0 - mu)
                            lambda = glob_stats[2 * i + 1] * (2.0 - glob_stats[2 * i]);
                            S = loci + 3 + glob_dat[loci];
                            L = glob_dat[loci + 1];
                            //cout << "2: start = " << S << ", len = " << L <<  endl;
                            sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);
                            
                            loci += 3 + glob_dat[loci] + glob_dat[loci + 1] + glob_dat[loci + 2];
                        }
                        
                        _mm_free(glob_stats);
                        _mm_free(glob_dat);                        
                        
                        mark2sync.clear();
                        dbet2sync.clear();                            
                        
                    } else {
                        
                        check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                    
                    }
                    
                    sum_vectors_f64(epsilon, tmpEps, deltaSum, Ntot);
                    
                    double te = MPI_Wtime();
                    tot_sync_ar2  += te - tb;
                    it_sync_ar2   += te - tb;
                    tot_nsync_ar2 += 1;
                    it_nsync_ar2  += 1;    

                } else { // case nranks == 1    
                    if(opt.deltaUpdate == true){
                        sum_vectors_f64(epsilon, tmpEps, dEpsSum,  Ntot);
                    }else{	
                        for(uint i=0; i < Ntot; i++){
                            epsilon[i] = epsilon[i] -  betaOld * mave[marker]/mstd[marker];
                            epsilon[i] = epsilon[i] + beta * mave[marker]/mstd[marker];
                        }
                        //And adjust even further for specific 1 and 2 allele values
		                for (size_t i = N1S[marker]; i < (N1S[marker] + N1L[marker]) ; i++){
                            epsilon[I1[i]] += betaOld/mstd[marker];
                            epsilon[I1[i]] -= beta/mstd[marker];
                        }
                        for (size_t i = N2S[marker]; i < (N2S[marker] + N2L[marker]) ; i++){
                            epsilon[I2[i]] += 2*betaOld/mstd[marker];
                            epsilon[I2[i]] -= 2*beta/mstd[marker];
                        }
                    }
                }
   
                // Do a update currently locally for vi vector
                for(int vi_ind=0; vi_ind < Ntot; vi_ind++){
                    vi[vi_ind] = exp(used_data.alpha * epsilon[vi_ind] - EuMasc);
                }
                double end_sync = MPI_Wtime();
                //printf("INFO   : synchronization time = %8.3f ms\n", (end_sync - beg_sync) * 1000.0);
                
                // Store epsilon state at last synchronization
                copy_vector_f64(tmpEps, epsilon, Ntot);
                
                // Reset local sum of delta epsilon
                set_vector_f64(dEpsSum, 0.0, Ntot);
                
                // Reset cumulated sum of delta betas
                cumSumDeltaBetas       = 0.0;
                task_sum_abs_deltabeta = 0.0;
                
                sinceLastSync = 0;
                
            }

        } // END PROCESSING OF ALL MARKERS

        //PROFILE
        //continue;

       
        //printf("rank %d it %d  beta_squaredNorm = %15.10f\n", rank, iteration, beta_squaredNorm);

        //printf("==> after eps sync it %d, rank %d, epsilon[0] = %15.10f %15.10f\n", iteration, rank, epsilon[0], epsilon[Ntot-1]);

        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            check_mpi(MPI_Allreduce(beta_squaredNorm.data(), sum_beta_squaredNorm.data(), beta_squaredNorm.size(),  MPI_DOUBLE,  MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(cass.data(),       sum_cass.data(),       cass.size(), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            cass             = sum_cass;
            beta_squaredNorm = sum_beta_squaredNorm;
        }
        if (rank == 0) {

            printf("\nINFO   : global cass on iteration %d:\n", iteration);
            for (int i=0; i<numGroups; i++) {
                printf("         Mtot[%3d] = %8d  | cass:", i, MtotGrp[i]);
                for (int ii=0; ii<K; ii++) {
                    printf(" %8d", cass(i, ii));
                }
                printf(" -> sum = %8d\n", cass.row(i).sum());
            }
        }

        // Update global parameters
        // ------------------------
	for(int gg = 0; gg < numGroups ; gg++){
	        m0[gg] = MtotGrp[gg] - cass(gg,0);
	}

        MPI_Barrier(MPI_COMM_WORLD);
 
        // 4. Sample sigmaG
	for(int gg=0; gg < numGroups ; gg++){
		sigmaG[gg]  = dist.inv_gamma_rng((double) (used_data.alpha_sigma + 0.5 * m0[gg]),(double)(used_data.beta_sigma + 0.5 * double(m0[gg])) * beta_squaredNorm(gg));
        }
	 check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
      	
        // 5. Sample prior mixture component probability from Dirichlet distribution
       
	for(int gg =0; gg < numGroups; gg++){
            	VectorXi dirin = cass.row(gg).array() + 1;  //For now use +1 as prior
            	pi_L.row(gg) = dist.dirichlet_rng(dirin);

	}

        check_mpi(MPI_Bcast(pi_L.data(), pi_L.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

        //Print results
        if(rank == 0){
	  cout << iteration << ". " << m0.sum() <<"; "<< setprecision(7) << mu << "; " <<  used_data.alpha << "; " << sigmaG.sum()  << endl;
        }

        double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
        if (rank == 0) {
            printf("RESULT : it %4d, rank %4d: proc = %9.3f s, sync = %9.3f (%9.3f + %9.3f), n_sync = %8d (%8d + %8d) (%7.3f / %7.3f), betasq = %15.10f, m0 = %10d\n",
                   iteration, rank, end_it-start_it,
                   it_sync_ar1  + it_sync_ar2,  it_sync_ar1,  it_sync_ar2,
                   it_nsync_ar1 + it_nsync_ar2, it_nsync_ar1, it_nsync_ar2,
                   (it_sync_ar1) / double(it_nsync_ar1) * 1000.0,
                   (it_sync_ar2) / double(it_nsync_ar2) * 1000.0,
                   beta_squaredNorm.sum(), int(m0.sum()));
            fflush(stdout);
        }
 
        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+Ntot,((epsilon).squaredNorm()+v0E*s02E)/(v0E+Ntot));
        //printf("sigmaG = %20.15f, sigmaE = %20.15f, e_sqn = %20.15f\n", sigmaG, sigmaE, e_sqn);
        //printf("it %6d, rank %3d: epsilon[0] = %15.10f, y[0] = %15.10f, m0=%10.1f,  sigE=%15.10f,  sigG=%15.10f [%6d / %6d]\n", iteration, rank, epsilon[0], y[0], m0, sigmaE, sigmaG, markerI[0], markerI[M-1]);

        // Write output files
        // ------------------

        if (iteration%opt.thin == 0) {

            if(rank == 0){
                //Save the hyperparameters
	        cx = snprintf(buff, LENBUF, "%5d, %20.15f, %20.15f, %20.15f, %20.15f, %7d, %7d, %2d", iteration, mu, sigmaG.sum() , used_data.alpha, sigmaG.sum()/(sigmaG.sum() + PI_squared / (6 * used_data.alpha*used_data.alpha)) , int(m0.sum()), int(pi_L.rows()), int(pi_L.cols()));
            //	assert(left > 0);
		assert(cx >= 0 && cx < LENBUF);  //We also use the condition cx < LENBUF for now


             //   cx = snprintf(&buff, LENBUF - strlen(buff), "%5d, %4d", iteration, (int) sigmaG.size());
             //   assert(cx >= 0 && cx < LENBUF);

                for(int jj = 0; jj < sigmaG.size(); ++jj){
                    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", sigmaG(jj));
                    assert(cx >= 0 && cx < LENBUF - strlen(buff));
                }


                for (int ii=0; ii < pi_L.rows(); ++ii) {
                    for(int kk = 0; kk < pi_L.cols(); ++kk) {
                        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", pi_L(ii,kk));
                        assert(cx >= 0 && cx < LENBUF - strlen(buff));
                    }
                }

                cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), "\n");
                assert(cx >= 0 && cx < LENBUF - strlen(buff));

                offset = size_t(n_thinned_saved) * strlen(buff);
                check_mpi(MPI_File_write_at(outfh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);

                //Save the covariates
                if(opt.covariates){ 
                    cx = snprintf(buff_gamma, LENBUF_gamma, "%5d", iteration);
                    for (int ii=0; ii < numFixedEffects; ++ii) {
                        cx = snprintf(&buff_gamma[strlen(buff_gamma)], LENBUF_gamma-strlen(buff_gamma), ", %20.17f", gamma(ii));
                        assert(cx > 0);
                	}
                	cx = snprintf(&buff_gamma[strlen(buff_gamma)], LENBUF_gamma - strlen(buff_gamma), "\n");
                	assert(cx > 0);
	                offset = size_t(n_thinned_saved) * strlen(buff_gamma);

                	check_mpi(MPI_File_write_at(gamfh, offset, &buff_gamma, strlen(buff_gamma), MPI_CHAR, &status), __LINE__, __FILE__);
                    //Save the order of the covariates
                    if (iteration > 0 && iteration%opt.save == 0){
                        offset = 0;
	                	check_mpi(MPI_File_write_at(xivfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                        offset = sizeof(uint);
                        check_mpi(MPI_File_write_at(xivfh, offset, &numFixedEffects, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
				
                        offset = sizeof(uint) + sizeof(uint);
	                	check_mpi(MPI_File_write_at(xivfh, offset, xI.data(), numFixedEffects,  MPI_INT,    &status), __LINE__, __FILE__);

                    }

                }

            }

            // Write iteration number
            if (rank == 0) {
                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(betfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpnfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            
            offset = sizeof(uint) + sizeof(uint) 
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double))
                + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(betfh, offset, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint)
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int))
                + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh, offset, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            //if (iteration == 0) {
            //    printf("rank %d dumping bet: %15.10f %15.10f\n", rank, Beta[0], Beta[MrankL[rank]-1]);
            //    printf("rank %d dumping cpn: %d %d\n", rank, components[0], components[MrankL[rank]-1]);
            //}

            n_thinned_saved += 1;
        }

        // Dump the epsilon vector and the marker indexing one
        // Note: single line overwritten at each saving iteration
        // .eps format: uint, uint, double[0, N-1] (it, Ntot, [eps])
        // .mrk format: uint, uint, int[0, M-1]    (it, M,    <mrk>)
        // ------------------------------------------------------
        if (iteration > 0 && iteration%opt.save == 0) {
            srand(opt.seed + iteration);
            // Each task writes its own rng file
            dist.write_rng_state_to_file(rngfp);
            offset  = 0;
            check_mpi(MPI_File_write_at(epsfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            
            offset = sizeof(uint);
 
            check_mpi(MPI_File_write_at(epsfh, offset, &Ntot,         1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, offset, &M,            1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(xbetfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);                
            check_mpi(MPI_File_write_at(xcpnfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh, offset, epsilon,        Ntot,           MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, offset, markerI.data(), markerI.size(), MPI_INT,    &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(xbetfh, offset, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(xcpnfh, offset, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            //if (iteration == 0) {
            //    printf("rank %d dumping eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot-1]);
            //}
            //EO: to remove once MPI version fully validated; use the check_epsilon utility to retrieve
            //    the corresponding values from the .eps file
            //    Print only first and last value handled by each task
            //printf("%4d/%4d epsilon[%5d] = %15.10f, epsilon[%5d] = %15.10f\n", iteration, rank, IrankS[rank], epsilon[IrankS[rank]], IrankS[rank]+IrankL[rank]-1, epsilon[IrankS[rank]+IrankL[rank]-1]);

#if 1
            //EO system call to create a tarball of the dump
            //TODO: quite rough, make it more selective...
            //----------------------------------------------
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                time_t now = time(0);
                tm *   ltm = localtime(&now);
                int    n   = 0;
                char targz[LENBUF];

                n=sprintf(targz, "dump_%s_%05d__%4d-%02d-%02d_%02d-%02d-%02d.tgz",
                          opt.mcmcOutNam.c_str(), iteration,
                          1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                          ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
                assert(n > 0);

                printf("INFO   : will create tarball %s in %s with file listed in %s.\n",
                       targz, opt.mcmcOutDir.c_str(), lstfp.c_str());

                //std::system(("ls " + opt.mcmcOut + ".*").c_str());
                string cmd = "tar -czf " + opt.mcmcOutDir + "/tarballs/" + targz + " -T " + lstfp;

                std::system(cmd.c_str());

            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        //double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Close output files
    check_mpi(MPI_File_close(&outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xbetfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xcpnfh), __LINE__, __FILE__);
    // check_mpi(MPI_File_close(&acufh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&mrkfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xivfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&gamfh), __LINE__, __FILE__);

    // Release memory
    _mm_free(y);
    _mm_free(epsilon);
    _mm_free(tmpEps);
    //free(previt_eps);
    _mm_free(deltaEps);
    _mm_free(dEpsSum);
    _mm_free(deltaSum);
    _mm_free(mave);
    _mm_free(mstd);
    _mm_free(USEBED);
    _mm_free(sum_failure);
    _mm_free(N1S);
    _mm_free(N1L);
    _mm_free(I1);
    _mm_free(N2S); 
    _mm_free(N2L);
    _mm_free(I2);
    _mm_free(NMS);
    _mm_free(NML);
    _mm_free(IM);

    if (opt.sparseSync) {
        _mm_free(glob_info);
        _mm_free(tasks_len);
        _mm_free(tasks_dis);
        _mm_free(stats_len);
        _mm_free(stats_dis);
    }

    const auto et3 = std::chrono::high_resolution_clock::now();
    const auto dt3 = et3 - st3;
    const auto du3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();
    if (rank == 0)
        printf("INFO   : rank %4d, time to process the data: %.3f sec, with %.3f (%.3f, %.3f) = %4.1f%% spent on allred (%d, %d)\n",
               rank, du3 / double(1000.0),
               tot_sync_ar1 + tot_sync_ar2, tot_sync_ar1, tot_sync_ar2,
               (tot_sync_ar1 + tot_sync_ar2) / (du3 / double(1000.0)) * 100.0,
               tot_nsync_ar1, tot_nsync_ar2);

    return 0;
}


// Get directory and basename of bed file (passed with no extension via command line)
// ----------------------------------------------------------------------------------

//  ORIGINAL (SEQUENTIAL) VERSION
/*
  VectorXd BayesW::getSnpData(unsigned int marker) const
  {
  if (!usePreprocessedData) {
  //read column from RAM loaded genotype matrix.
  return data.Z.col(marker);//.cast<double>();
  } else {
  //read column from preprocessed and memory mapped genotype matrix file.
  return data.mappedZ.col(marker).cast<double>();
  }
  }

  void BayesW::printDebugInfo() const
  {
  //const unsigned int N(data.numInds);
  // cout << "x mean " << Cx.mean() << "\n";
  //   cout << "x sd " << sqrt(Cx.squaredNorm() / (double(N - 1))) << "\n";
  }
*/
