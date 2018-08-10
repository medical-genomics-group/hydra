//
//  gctb.hpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef amber_hpp
#define amber_hpp

#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#include "../src/data.hpp"
#include "../src/options.hpp"
#include "model.hpp"
#include "mcmc.hpp"
#include "hsq.hpp"
#include "mympi.hpp"

class GCTB {
public:
    Options &opt;

    GCTB(Options &options): opt(options){};
    
    void inputIndInfo(Data &data, const string &bedFile, const string &phenotypeFile, const string &keepIndFile,
                      const unsigned keepIndMax, const unsigned mphen, const string &covariateFile);
    void inputSnpInfo(Data &data, const string &bedFile, const string &includeSnpFile, const string &excludeSnpFile,
                      const unsigned includeChr, const bool readGenotypes);
    void inputSnpInfo(Data &data, const string &includeSnpFile, const string &excludeSnpFile,
                      const string &gwasSummaryFile, const string &ldmatrixFile, const unsigned includeChr, const bool multiLDmatrix);

    Model* buildModel(Data &data, const string &bedFile, const string &gwasFile, const string &bayesType, const unsigned windowWidth,
                      const float heritability, const float probFixed, const bool estimatePi,
                      const string &algorithm, const unsigned snpFittedPerWindow, const float varS, const vector<float> &S);
    vector<McmcSamples*> runMcmc(Model &model, const unsigned chainLength, const unsigned burnin, const unsigned thin, const unsigned outputFreq, const string &title, const bool writeBinPosterior);
    void saveMcmcSamples(const vector<McmcSamples*> &mcmcSampleVec, const string &filename);
    void outputResults(const Data &data, const vector<McmcSamples*> &mcmcSampleVec, const string &filename);

    McmcSamples* inputMcmcSamples(const string &mcmcSampleFile, const string &label, const string &fileformat);
    void estimateHsq(const Data &data, const McmcSamples &snpEffects, const McmcSamples &resVar, const string &filename);

    void inputSnpResults(Data &data, const string &snpResFile);
    void predict(const Data &data, const string &filename);

    void clearGenotypes(Data &data);
};

#endif /* amber_hpp */
