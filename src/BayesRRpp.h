/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#ifndef SRC_BAYESRRPP_H_
#define SRC_BAYESRRPP_H_
#include "data.hpp"
#include <Eigen/Eigen>
#include "options.hpp"
#include "distributions_boost.hpp"

class BayesRRpp
{
    Data            &data; // Data matrices
    Options         &opt;
    const string    bedFile;//bed file
    const long      memPageSize;//size of memory
    const string    outputFile;
    const int       seed;
    const int       max_iterations;
    const int		burn_in;
    const int       thinning;
    const double	sigma0 = 0.0001;
    const double	v0E = 0.0001;
    const double    s02E = 0.0001;
    const double    v0G = 0.0001;
    const double    s02G = 0.0001;
    Eigen::VectorXd cva;
    Distributions_boost dist;

public:
    BayesRRpp(Data &data, Options &opt, const long memPageSize);
    virtual ~BayesRRpp();
    int runGibbs(); // Where we run Gibbs sampling over the paramaterised model
};

#endif /* SRC_BAYESRRM_H_ */
