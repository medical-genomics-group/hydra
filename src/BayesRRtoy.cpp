/*
 * BayesRRtoy.cpp
 *
 *  Created on: 10 Aug 2018
 *      Author: Daniel Trejo Banos
 */

#include "BayesRRtoy.hpp"

BayesRRtoy::BayesRRtoy(Data &data):data(data) {}

BayesRRtoy::~BayesRRtoy() {
    // TODO Auto-generated destructor stub
}

void BayesRRtoy::runToyExample(int samples ){

    std::vector<int> markerI;
    int M;
    int marker;
    std::cout<<"running toy example ";

    M=data.Z.cols();
    std::cout << "Sampling " << M <<" snps\n";

    for (int i=0; i<M; ++i) {
        markerI.push_back(i);
    }

    for(int i=0;  i<samples; i++){

        std::random_shuffle(markerI.begin(), markerI.end());

        for(int j=0; j < M; ++j) {

            marker= markerI[j];

            if (marker%100 == 0)
                printf("  -> marker %6i has mean %13.6f on %d elements [%13.6f, %13.6f]  Sq. Norm = %13.6f\n", marker, data.Z.col(marker).mean(), data.Z.col(marker).size(), data.Z.col(marker).minCoeff(), data.Z.col(marker).maxCoeff(), data.ZPZdiag[marker]);
        }
    }

    std::cout << "BayesRRtoy success" << endl;
}
