/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#ifndef SRC_BAYESRRM_H_
#define SRC_BAYESRRM_H_
#include "data.hpp"
#include <Eigen/Eigen>
class BayesRRm {
  Data          &data;//data matrices
  const string  bedFile;//bed file
  const long    memPageSize;//size of memory
  const string  outputFile="bayesOutput.csv";
  const int     seed=1;
  const int 	max_iterations=10;
  const int		burn_in=5;
  const int 	thinning=1;
  const double	sigma0=0.0001;
  const double	v0E=0.0001;
  const double  s02E=0.0001;
  const double  v0G=0.0001;
  const double  s02G=0.0001;
  Eigen::VectorXd cva;

public:
	BayesRRm(Data &data, const string bedFile, const long memPageSize);
	virtual ~BayesRRm();
	int runGibbs(); //where we run Gibbs sampling over the parametrised model
};

#endif /* SRC_BAYESRRM_H_ */
