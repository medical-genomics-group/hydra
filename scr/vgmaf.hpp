//
//  vgmaf.hpp
//  gctb
//
//  Created by Jian Zeng on 23/11/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef vgmaf_hpp
#define vgmaf_hpp

#include <stdio.h>
#include "data.hpp"
#include "mcmc.hpp"

class VGMAF {
public:
    unsigned numBins;
    unsigned numSnps;
    
    vector<float> mafbin;
    vector<vector<int> > snpIndex;
    MatrixXf cumVarGen;
    MatrixXf cumPi;
    MatrixXf meanBeta;
    MatrixXf nnz;
    VectorXf cumVarGenMean;
    VectorXf cumVarGenSD;
    VectorXf cumPiMean;
    VectorXf cumPiSD;
    VectorXf meanBetaMean;
    VectorXf meanBetaSD;
    VectorXf nnzMean;
    VectorXf nnzSD;
    VectorXf auc;
    
    void makeMafBin(const Data &data, vector<float> &mafbin, vector<vector<int> > &snpIndex);
    void calcCumVarGen(const MatrixXf &Z, const vector<SnpInfo*> &snpVec, const McmcSamples &snpEffects, const unsigned burnin, const unsigned thin,
                       const vector<vector<int> > &snpIndex, MatrixXf &cumVarGen, VectorXf &cumVarGenMean);
    void outputRes(const string &title);
    void compute(const Data &data, const McmcSamples &snpEffects, const unsigned burnin, const unsigned thin, const string &title);
    void simulate(const Data &data, const string &title);
};


#endif /* vgmaf_hpp */
