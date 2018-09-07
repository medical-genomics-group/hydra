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


// Calling mmap based reading function (which does base on OpenMP)
// Rather, OpenMP is used to parallelize calls to the function.
// ---------------------------------------------------------------
void BayesRMmapToy::runToyExample(int samples ) {

  std::vector<int> markerI;
  //int marker;
  //VectorXf normedSnpData(data.numKeptInds);

  std::cout<<"running toy example ";
  std::cout << "Sampling " << data.numIncdSnps <<" snps\n";

  // Compute the SNP data length in bytes
  size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;

  for (int i=0; i < data.numIncdSnps; ++i) {
    markerI.push_back(i);
  }

  for (int i=0; i<samples; ++i) {

    std::random_shuffle(markerI.begin(), markerI.end());

#pragma omp parallel for
    for (int j=0; j < data.numIncdSnps; ++j) {
        
        int marker= markerI[j];
        VectorXf normedSnpData(data.numKeptInds);
        data.getSnpDataFromBedFileUsingMmap(bedFile, snpLenByt, memPageSize, marker, normedSnpData);

        if(marker%100 == 0) {
            printf("  -> marker %6i has mean %13.6f on %d elements [%13.6f, %13.6f]  Sq. Norm = %13.6f, Var = %9.7f\n", marker, normedSnpData.mean(), normedSnpData.size(), normedSnpData.minCoeff(), normedSnpData.maxCoeff(), data.ZPZdiag[marker], normedSnpData.squaredNorm()/normedSnpData.size());
            fflush(stdout);
        }
    }
  }
  std::cout << "BayesRMmapToy success" << endl;
}


// Calling mmap + OpenMP based reading function
// --------------------------------------------
void BayesRMmapToy::runToyExample2(int samples) {

  std::vector<int> markerI;

  std::cout<<"running toy example ";
  std::cout << "Sampling " << data.numIncdSnps <<" snps\n";

  // Compute the SNP data length in bytes
  size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;

  for (int i=0; i < data.numIncdSnps; ++i) {
    markerI.push_back(i);
  }

  for (int i=0; i<samples; ++i) {

    std::random_shuffle(markerI.begin(), markerI.end());

    for (int j=0; j < data.numIncdSnps; ++j) {

        int marker= markerI[j];
        VectorXf normedSnpData(data.numKeptInds);
        data.getSnpDataFromBedFileUsingMmap_openmp(bedFile, snpLenByt, memPageSize, marker, normedSnpData);

        if(marker%100 == 0) {
            printf("  -> marker %6i has mean %13.6f on %d elements [%13.6f, %13.6f]  Sq. Norm = %13.6f, Var = %9.7f\n", marker, normedSnpData.mean(), normedSnpData.size(), normedSnpData.minCoeff(), normedSnpData.maxCoeff(), data.ZPZdiag[marker], normedSnpData.squaredNorm()/normedSnpData.size());
            fflush(stdout);
        }
    }
  }
  std::cout << "BayesRMmapToy success" << endl;
}
