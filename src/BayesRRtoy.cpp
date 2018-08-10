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
	std::cout<<"sampling " << M <<" snps\n";

    for (int i=0; i<M; ++i) {
	   markerI.push_back(i);
	}
	int i;
    for(i=0;  i<samples; i++){
    	std::random_shuffle(markerI.begin(), markerI.end());
    	for(int j=0; j < M; j++){
    		marker= markerI[j];
    		data.Z.col(marker);
    		std::cout<<"column: " << std::to_string(marker)<< "mean: " <<data.Z.col(marker).mean() << "\n";
    	}

    }
    std::cout<<"success";

}
