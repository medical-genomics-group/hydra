/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#ifndef HYDRA_BAYESRRM_H_
#define HYDRA_BAYESRRM_H_

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
    double             v0G     = 0.0001;
    double             s02G    = 0.0001;
    const double       s02F    = 1.0;

    uint iteration_start           = 0;
    uint iteration_to_restart_from = 0;
    uint first_thinned_iteration   = 0;
    uint first_saved_iteration     = 0;

    VectorXd  cva;
    MatrixXd  cVa;       // component-specific variance
    MatrixXd  cVaI;      // inverse of the component variances

    Distributions_boost dist;
    Distributions_boost dist8[8];
    bool showDebug;

    MatrixXd X;         // "fixed effects" matrix.
    VectorXd gamma;     // fixed effects coefficients

    // Component variables
    MatrixXd priorPi;   // prior probabilities for each component
    MatrixXd estPi;     // mixture probabilities
    VectorXd logL;      // log likelihood of component
    VectorXd muk;       // mean of k-th component marker effect size
    VectorXd denom;     // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
    VectorXi m0;        // total number of markers in model
    MatrixXi cass;
    MatrixXi cass8;

    // Mean and residual variables
    double mu;          // mean or intercept
    VectorXd sigmaG;    // genetic variance
    double sigmaE;      // residuals variance
    double sigmaF;      // covariates variance if using ridge;

    //EO: for sequential
    double   seq_mu,  seq_sigmaG, seq_sigmaE,  betasqn;
    VectorXd seq_cVa, seq_cVaI,   seq_priorPi, seq_estPi;

    double mu8[8];
    double sigmaG8[8];
    double sigmaE8[8];
    double sigmaF8[8];

    // Linear model variables
    //EO@@@ check for double declaration...
    //DT@@@ solved, deleted double declaration and related functions.
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

    BayesRRm(Data &data, Options &opt, const long memPageSize);
    virtual ~BayesRRm();

    void setDebugEnabled(bool enabled) { showDebug = enabled; }

    bool isDebugEnabled() const { return showDebug; }    

    int  runGibbs();

    void init_from_restart(const int K, const uint M, const uint Mtot, const uint Ntotc,
                             const int* MrankS, const int* MrankL, const bool use_xbet);
    
    void init_from_scratch();

    int  checkRamUsage();

    int  runMpiGibbs();

    int  runMpiGibbsMultiTraits();
};

#endif /* SRC_BAYESRRM_H_ */
