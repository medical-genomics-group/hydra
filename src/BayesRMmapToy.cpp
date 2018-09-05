/*
 * BayesRMmapToy.cpp
 *
 *  Created on: 13 Aug 2018
 *      Author: E. Orliac
 */

#include "BayesRMmapToy.hpp"
#include "data.hpp"

BayesRMmapToy::BayesRMmapToy(Data &data, const string bedFile, const long memPageSize)
  :data(data), bedFile(bedFile), memPageSize(memPageSize) {}

BayesRMmapToy::~BayesRMmapToy() {
  // TODO Auto-generated destructor stub
}

void BayesRMmapToy::runToyExample(int samples ){

  std::vector<int> markerI;
  int marker;
  VectorXf normedSnpData(data.numKeptInds);

  std::cout<<"running toy example ";
  std::cout << "Sampling " << data.numIncdSnps <<" snps\n";

  // Compute the SNP data length in bytes
  size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;

  for (int i=0; i < data.numIncdSnps; ++i) {
    markerI.push_back(i);
  }

  for(int i=0; i<samples; ++i) {

    std::random_shuffle(markerI.begin(), markerI.end());

    for(int j=0; j < data.numIncdSnps; ++j) {

      marker= markerI[j];
      data.getSnpDataFromBedFileUsingMmap(bedFile, snpLenByt, memPageSize, marker, normedSnpData);

      if(marker%100 == 0) {
          printf("  -> marker %6i has mean %13.6f on %d elements [%13.6f, %13.6f]  Sq. Norm = %13.6f\n", marker, normedSnpData.mean(), normedSnpData.size(), normedSnpData.minCoeff(), normedSnpData.maxCoeff(), data.ZPZdiag[marker]);
          fflush(stdout);
      }
    }
  }
  std::cout << "BayesRMmapToy success" << endl;
}
