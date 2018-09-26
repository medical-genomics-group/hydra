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
    unsigned mphen; // triat order id in the phenotype file for analysis
    unsigned windowWidth; // in mega-base unit
    unsigned keepIndMax;  // the maximum number of individuals kept for analysis
    unsigned snpFittedPerWindow;    // for BayesN
    unsigned thin;  // save every this th sampled value in MCMC
    unsigned includeChr;  // chromosome to include
    
    float probFixed;
    float heritability;
//    float varGenotypic;
//    float varResidual;
    float varS; // prior variance of S in BayesS and BayesNS
    vector<float> S;    // starting value of S in BayesS and BayesNS
    


    bool estimatePi;
    bool estimateScale;
    bool writeBinPosterior;
    bool outputResults;
    bool multiLDmat;
    

    unsigned int numGroups;
    Eigen::MatrixXd mS;
    string groupFile;


    string title;
    string analysisType;
    string bayesType;
    string algorithm;
    string optionFile;
    string phenotypeFile;
    string covariateFile;
    string bedFile;
    string alleleFreqFile;
    string includeSnpFile;
    string excludeSnpFile;
    string keepIndFile;
    string snpResFile;
    string mcmcSampleFile;
    string gwasSummaryFile;
    string ldmatrixFile;
    
    Options(){
        chainLength             = 10000;
        burnin                  = 5000;
        outputFreq              = 100;
        seed                    = static_cast<unsigned int>(std::time(0));
        numThread               = 1;
        mphen                   = 1;
        keepIndMax              = UINT_MAX;
        snpFittedPerWindow      = 2;
        thin                    = 5;
        includeChr              = 0;
                
        windowWidth             = Megabase;
        probFixed               = 0.05;
        heritability            = 0.5;
//        varGenotypic            = 1.0;
//        varResidual             = 1.0;
        varS                    = 1.0;
        S.resize(3);
        S[0]                    = 0.01;
        S[1]                    = 0.001;
        S[2]                    = 0.0001;
        estimatePi              = true;
        estimateScale           = false;
        writeBinPosterior       = true;
        outputResults           = true;
        multiLDmat              = false;
        
        title                   = "brr";
        analysisType            = "Bayes";
        bayesType               = "C";
        algorithm               = "";
        optionFile              = "";
        phenotypeFile           = "";
        covariateFile           = "";
        bedFile                 = "";
        alleleFreqFile          = "";
        includeSnpFile          = "";
        excludeSnpFile          = "";
        keepIndFile             = "";
        snpResFile              = "";
        mcmcSampleFile          = "bayesOutput.csv";
        gwasSummaryFile         = "";
        ldmatrixFile            = "";
        numGroups				=2;
    }
    
    void inputOptions(const int argc, const char* argv[]);
    
private:
    void readFile(const string &file);
    void makeTitle(void);
    void seedEngine(void);
};

#endif /* options_hpp */
