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
    Data               &data; // data matrices
    Options            &opt;
    const string       bedFile; // bed file
    const long         memPageSize; // size of memory
    const string       outputFile;
    const unsigned int seed;
    const unsigned int max_iterations;
    const unsigned int burn_in;
    const double	   sigma0  = 0.0001;
    const double	   v0E     = 0.0001;
    const double       s02E    = 0.0001;
    const double       v0G     = 0.0001;
    const double       s02G    = 0.0001;
    Eigen::VectorXd    cva;
    Distributions_boost dist;
    bool usePreprocessedData;
    bool showDebug;
    double betasqn;

    MatrixXd X;         // "fixed effects" matrix.
    VectorXd gamma;     // fixed effects coefficients

    // Component variables
    VectorXd priorPi;   // prior probabilities for each component
    VectorXd pi;        // mixture probabilities
    VectorXd cVa;       // component-specific variance
    VectorXd logL;      // log likelihood of component
    VectorXd muk;       // mean of k-th component marker effect size
    VectorXd denom;     // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
    int m0;             // total number of markers in model
    VectorXd v;         // variable storing the component assignment
    VectorXd cVaI;      // inverse of the component variances
    VectorXd components;

    // Mean and residual variables
    double mu;          // mean or intercept
    double sigmaG;      // genetic variance
    double sigmaE;      // residuals variance

    // Linear model variables
    VectorXd beta;       // effect sizes
    VectorXd y_tilde;    // variable containing the adjusted residuals to exclude the effects of a given marker
    VectorXd epsilon;    // variable containing the residuals

    VectorXd y;
    VectorXd Cx;

    double epsilonsum;
    double ytildesum;

    const size_t LENBUF = 200;

public:
    BayesRRm(Data &data, Options &opt, const long memPageSize);
    virtual ~BayesRRm();

    int  runGibbs(); // where we run Gibbs sampling over the parametrised model
    void setDebugEnabled(bool enabled) { showDebug = enabled; }
    bool isDebugEnabled() const { return showDebug; }

#ifdef USE_MPI
    std::string mpi_get_sparse_output_filebase(void);
    void write_sparse_data_files(const uint bpr);
    int  runMpiGibbs();
    void check_whole_array_was_set(const uint* array, const size_t size, const int linenumber, const char* filename);
#endif
    
private:
    void init(int K, unsigned int markerCount, unsigned int individualCount);
    VectorXd getSnpData(unsigned int marker) const;
    void printDebugInfo() const;
};

#endif /* SRC_BAYESRRM_H_ */
