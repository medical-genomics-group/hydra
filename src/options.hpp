//
//  options.hpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef options_hpp
#define options_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <string>
#include <limits.h>
#include <mpi.h>
#include <boost/format.hpp>
#include "mympi.hpp"
#include "gadgets.hpp"
#include <Eigen/Eigen>

using namespace std;
using namespace boost;


const unsigned Megabase = 1e6;

class Options {
public:
    unsigned chainLength;
    unsigned burnin;
    unsigned outputFreq;
    unsigned seed;
    unsigned numThread;
    unsigned keepIndMax;  // the maximum number of individuals kept for analysis
    unsigned thin;  // save every this th sampled value in MCMC
    unsigned includeChr;  // chromosome to include
    float varS; // prior variance of S in BayesS and BayesNS
    vector<float> S;    // starting value of S in BayesS and BayesNS

    bool outputResults;


    unsigned int numGroups;
    Eigen::MatrixXd mS;
    string groupFile;


    string title;
    string analysisType;
    string bayesType;
    string algorithm;
    string optionFile;
    string phenotypeFile;
    string bedFile;
    string alleleFreqFile;
    string includeSnpFile;
    string excludeSnpFile;
    string keepIndFile;
    string snpResFile;
    string mcmcSampleFile;


    Options(){
        chainLength             = 10000;
        burnin                  = 5000;
        outputFreq              = 100;
        seed                    = static_cast<unsigned int>(std::time(0));
        numThread               = 1;
        keepIndMax              = UINT_MAX;
        thin                    = 5;
        includeChr              = 0;

        varS                    = 1.0;
        S.resize(3);
        S[0]                    = 0.01;
        S[1]                    = 0.001;
        S[2]                    = 0.0001;

        outputResults           = true;

        title                   = "brr";
        analysisType            = "Bayes";
        bayesType               = "C";
        algorithm               = "";
        optionFile              = "";
        phenotypeFile           = "";
        bedFile                 = "";
        alleleFreqFile          = "";
        includeSnpFile          = "";
        excludeSnpFile          = "";
        keepIndFile             = "";
        mcmcSampleFile          = "bayesOutput.csv";
        numGroups				=2;
    }

    void inputOptions(const int argc, const char* argv[]);

private:
    void readFile(const string &file);
    void makeTitle(void);
    void seedEngine(void);
};

#endif /* options_hpp */
