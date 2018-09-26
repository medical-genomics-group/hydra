/*
 * BayesRRg.h
 *
 *  Created on: 25 Sep 2018
 *      Author: admin
 */

#ifndef BAYESRRG_H_
#define BAYESRRG_H_
#include "data.hpp"
#include <Eigen/Eigen>
#include "options.hpp"
#include "distributions_boost.hpp"
class BayesRRg {

	  Data          &data;//data matrices
	  Options       &opt;
	  const string  bedFile;//bed file
	  const long    memPageSize;//size of memory
	  const string  outputFile;
	  const int     seed;
	  const int 	max_iterations;
	  const int		burn_in;
	  const int 	thinning;
	  const double	sigma0=0.0001;
	  const double	v0E=0.0001;
	  const double  s02E=0.0001;
	  const double  v0G=0.0001;
	  const double  s02G=0.0001;
	  Eigen::MatrixXd cva;
	  //TODO maybe it is better to leave the group assignment to be handled by Data
	  //Eigen::VectorXi groupAssignment;
	  const int numberGroups;
	  Distributions_boost dist;

public:
	BayesRRg(Data &data, Options &opt, const long memPageSize);
	virtual ~BayesRRg();
	void runGibbs();
};

#endif /* BAYESRRG_H_ */
