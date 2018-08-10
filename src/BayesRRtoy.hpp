/*
 * BayesRRtoy.h
 *
 *  Created on: 10 Aug 2018
 *      Author: admin
 */

#ifndef BAYESRRTOY_HPP_
#define BAYESRRTOY_HPP_
#include "data.hpp"
#include <Eigen/Eigen>

class BayesRRtoy {
	const Data &data;
public:
	BayesRRtoy(Data &data);
	virtual ~BayesRRtoy();

	void runToyExample(int samples);
};

#endif /* BAYESRRTOY_HPP_ */
