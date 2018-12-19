/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#ifndef SRC_BAYESRRM_EO_H_
#define SRC_BAYESRRM_EO_H_

#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"

#include <Eigen/Eigen>

class BayesRRm_eo
{
    Data            &data; // data matrices
    Options         &opt;
    const string    bedFile; // bed file
    const long      memPageSize; // size of memory
    const string    outputFile;
    const int       seed;
    const int       max_iterations;
    const int		burn_in;
    const int       thinning;
    const double	sigma0  = 0.0001;
    const double	v0E     = 0.0001;
    const double    s02E    = 0.0001;
    const double    v0G     = 0.0001;
    const double    s02G    = 0.0001;
    Eigen::VectorXd cva;
    Distributions_boost dist;

    const unsigned int M    = 5000;

public:
    BayesRRm_eo(Data &data, Options &opt, const long memPageSize);
    virtual ~BayesRRm_eo();
    int  runGibbs();
    void runTest(int numKeptInds, size_t snpLenByt);
    void runTest_moody(int numKeptInds, size_t snpLenByt);
    int  open_bed_file_for_reading(const string &bedFile);
};

#endif /* SRC_BAYESRRM_H_ */
