/*
 * BayesRMmapToy.h
 *
 *  Created on: 24 Aug 2018
 *      Author: E. Orliac
 */

#ifndef BAYESRMMAPTOY_HPP_
#define BAYESRMMAPTOY_HPP_
#include "data.hpp"
#include <Eigen/Eigen>

class BayesRMmapToy {

  Data          &data;
  const string  bedFile;
  const long    memPageSize;

public:
  BayesRMmapToy(Data &data, const string bedFile, const long memPageSize);
  virtual ~BayesRMmapToy();
  
  void runToyExample(int samples);
};

#endif /* BAYESRRTOY_HPP_ */
