//
//  xci.hpp
//  gctb
//
//  Created by Jian Zeng on 27/10/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef xci_hpp
#define xci_hpp

#include <stdio.h>
#include "gctb.hpp"

class XCI {
public:
    unsigned numKeptMales;
    unsigned numKeptFemales;
    
    XCI(){
        numKeptMales   = 0;
        numKeptFemales = 0;
    }
    
    void inputIndInfo(Data &data, const string &bedFile, const string &phenotypeFile, const string &keepIndFile,
                      const unsigned keepIndMax, const unsigned mphen, const string &covariateFile);
    void sortIndBySex(vector<IndInfo*> &indInfoVec);
    void restoreFamFileOrder(vector<IndInfo*> &indInfoVec);
    Model* buildModel(Data &data, const float heritability, const float probFixed, const bool estimatePi);
    void simu(Data &data, const unsigned numQTL, const float heritability, const float probNDC, const bool removeQTL);
    void outputResults(const Data &data, const vector<McmcSamples*> &mcmcSampleVec, const string &title);
};


class BayesXCI : public BayesC {
public:
    // y = mu + sum_j Z_j beta_j delta_j + e
    // For males,   Z_mj = X_mj
    // For females, Z_fj = X_fj with prob. p (under NDC model) or 0.5*X_fj with prob. 1-p (under FDC model)
    // p ~ U(0,1) or Beta(a,b) is the prob. of NDC model, in other word, the proportion of SNPs that escape from XCI
    // beta_j ~ N(0, sigma^2); sigma^2 ~ scaled-inverse chi-square
    // delta_j ~ Bernoulli(pi)
    
    class ProbNDC : public Parameter, public Stat::Beta {
    public:
        const float alpha;
        const float beta;
        
        ProbNDC(const float p): Parameter("ProbNDC"), alpha(1), beta(1){  // conditional probability on nonzero SNPs, uniform prior
            value = p;
        }
        
        void sampleFromFC(const unsigned numSnps, const unsigned numNDC);
    };
        
    class Gamma : public ParamSet {
    public:
        Gamma(const vector<string> &header): ParamSet("Gamma", header){};
    };
    
    class SnpEffects : public BayesC::SnpEffects {
    public:
        SnpEffects(const vector<string> &header): BayesC::SnpEffects(header, "Gibbs"){};
        
        void sampleFromFC(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag, const VectorXf &ZPZdiagMale,
                          const VectorXf &ZPZdiagFemale, const unsigned nmale, const unsigned nfemale, const float p,
                          const float sigmaSq, const float pi, const float vare, VectorXf &gamma, VectorXf &ghat);
    };
    
    class ScaleVar : public BayesC::ScaleVar {
    public:
        const float sum2pq;
        
        ScaleVar(const float sum2pq, const float val): BayesC::ScaleVar(val), sum2pq(sum2pq){}
        
        void compute(const float vg, const float pi, float &scaleVar){
            value = 0.5f*vg/(sum2pq*(1.0f-pi));
            scaleVar = value;
        };
    };
    
    class Rounding : public BayesC::Rounding {
    public:
        Rounding(): BayesC::Rounding(){}
        void computeYcorr(const VectorXf &y, const MatrixXf &X, const MatrixXf &Z,
                          const VectorXf &gamma, const unsigned nmale, const unsigned nfemale,
                          const VectorXf &fixedEffects, const VectorXf &snpEffects,
                          VectorXf &ycorr);
    };
    
    unsigned nmale, nfemale;
    float genVarPrior;
    float piPrior;
    
    ProbNDC p;
    Gamma gamma;   // indicator variable with 1: NDC, 0: FDC
    SnpEffects snpEffects;
    ScaleVar scale;
    Rounding rounding;
    
    VectorXf ZPZdiagMale;
    VectorXf ZPZdiagFemale;
    
    BayesXCI(const Data &data, const float varGenotypic, const float varResidual, const float probFixed, const bool estimatePi, const unsigned nmale, const unsigned nfemale):
    BayesC(data, varGenotypic, varResidual, probFixed, estimatePi, "Gibbs", false),
    p(0.15),
    gamma(data.snpEffectNames),
    snpEffects(data.snpEffectNames),
    scale(data.snp2pq.sum(), sigmaSq.scale),
    genVarPrior(varGenotypic),
    piPrior(probFixed),
    nmale(nmale), nfemale(nfemale) {
        
        // MPI
        ZPZdiagMale.setZero(data.numIncdSnps);
        ZPZdiagFemale.setZero(data.numIncdSnps);
        VectorXf my_ZPZdiagMale   = data.Z.block(0, 0, nmale, data.numIncdSnps).colwise().squaredNorm();
        VectorXf my_ZPZdiagFemale = data.Z.block(nmale, 0, nfemale, data.numIncdSnps).colwise().squaredNorm();
        MPI_Allreduce(&my_ZPZdiagMale[0], &ZPZdiagMale[0], data.numIncdSnps, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&my_ZPZdiagFemale[0], &ZPZdiagFemale[0], data.numIncdSnps, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        gamma.values.setZero(data.numIncdSnps);
        paramSetVec = {&snpEffects, &gamma, &fixedEffects};
        paramVec = {&pi, &nnzSnp, &sigmaSq, &scale, &p, &vare, &varg, &hsq};
        paramToPrint = {&pi, &nnzSnp, &sigmaSq, &scale, &p, &vare, &varg, &hsq, &rounding};
        if (myMPI::rank==0)
            cout << "\nBayesXCI model fitted." << endl;
    }
    
    void sampleUnknowns(void);
};


#endif /* xci_hpp */
