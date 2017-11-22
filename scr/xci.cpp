//
//  xci.cpp
//  gctb
//
//  Created by Jian Zeng on 27/10/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "xci.hpp"

using namespace std;

void XCI::sortIndBySex(vector<IndInfo*> &indInfoVec){
    vector<IndInfo*> male, female;
    long numInds = indInfoVec.size();
    IndInfo *ind;
    for (unsigned i=0; i<numInds; ++i) {
        ind = indInfoVec[i];
        if (ind->sex == 1) male.push_back(ind);
        else if (ind->sex == 2) female.push_back(ind);
    }
    indInfoVec.resize(0);
    indInfoVec.reserve(male.size() + female.size());
    indInfoVec.insert(indInfoVec.end(), male.begin(), male.end());
    indInfoVec.insert(indInfoVec.end(), female.begin(), female.end());
}

void XCI::restoreFamFileOrder(vector<IndInfo*> &indInfoVec){
    long numInds = indInfoVec.size();
    vector<IndInfo*> vec(numInds);
    IndInfo *ind;
    for (unsigned i=0; i<numInds; ++i) {
        ind = indInfoVec[i];
        vec[ind->famFileOrder] = ind;
    }
    indInfoVec = vec;
}

void XCI::inputIndInfo(Data &data, const string &bedFile, const string &phenotypeFile, const string &keepIndFile,
                       const unsigned keepIndMax, const unsigned mphen, const string &covariateFile){
    data.readFamFile(bedFile + ".fam");
    sortIndBySex(data.indInfoVec);
    data.readPhenotypeFile(phenotypeFile, mphen);
    data.keepMatchedInd(keepIndFile, keepIndMax);
    
    numKeptMales   = 0;
    numKeptFemales = 0;
    IndInfo *ind;
    for (unsigned i=0; i<data.numKeptInds; ++i){
        ind = data.keptIndInfoVec[i];
        if (ind->sex == 1) ++numKeptMales;
        else if (ind->sex == 2) ++numKeptFemales;
    }
    
    // MPI
    unsigned numKeptMales_all, numKeptFemales_all;
    MPI_Allreduce(&numKeptMales, &numKeptMales_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&numKeptFemales, &numKeptFemales_all, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
   if (myMPI::rank==0)
        cout << "Matched " << numKeptMales_all << " males and " << numKeptFemales_all << " females." << endl;
    
    data.readCovariateFile(covariateFile);
    restoreFamFileOrder(data.indInfoVec);
}

Model* XCI::buildModel(Data &data, const float heritability, const float probFixed, const bool estimatePi){
    data.initVariances(heritability);
    return new BayesXCI(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, numKeptMales, numKeptFemales);
}

void XCI::simu(Data &data, const unsigned numQTL, const float heritability, const float probNDC, const bool removeQTL){
    vector<unsigned> indices(data.numIncdSnps);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());
    vector<SnpInfo*> QTLvec(numQTL);
    MatrixXf Q(data.numKeptInds, numQTL);
    VectorXf alpha(numQTL);
    SnpInfo *qtl;
    unsigned numFDC = 0;
    for (unsigned j=0; j<numQTL; ++j) {
        qtl = data.incdSnpInfoVec[indices[j]];
        qtl->isQTL = true;
        QTLvec[j] = qtl;
        Q.col(j) = data.Z.col(qtl->index);
        alpha[j] = Stat::snorm();
        if (Stat::ranf() < 1.0f - probNDC) {
            for (unsigned i=numKeptMales; i<data.numKeptInds; ++i) {
                if ( Q(i,j) == 1 ) Q(i,j) = Stat::ranf() < 0.5 ? 1 : 0;
                else Q(i,j) *= 0.5f;
            }
            //Q.col(j).tail(numKeptFemales) *= 0.5f;
            ++numFDC;
        }
    }
    
    VectorXf g = Q*alpha;
    
    // calculate genetic variance with MPI
    float my_sumg = g.sum();
    float my_ssg  = g.squaredNorm();
    float sumg, ssg;
    unsigned ng;
    MPI_Allreduce(&my_sumg, &sumg, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&my_ssg, &ssg, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&data.numKeptInds, &ng, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    
    float genVar = ssg/float(ng) - sumg*sumg/float(ng*ng);
    float resVar = genVar*(1.0-heritability)/heritability;
    float resSD  = sqrt(resVar);
    
    for (unsigned i=0; i<data.numKeptInds; ++i) {
        data.y[i] = g[i] + Stat::snorm()*resSD;
    }
    
    data.varGenotypic = genVar;
    data.varResidual  = resVar;
    
    if (removeQTL) {
        MatrixXf Ztmp(data.numKeptInds, data.numIncdSnps-numQTL);
        VectorXf ZPZdiagTmp(data.numIncdSnps-numQTL);
        VectorXf snp2pqTmp(data.numIncdSnps-numQTL);
        vector<string> snpEffectNamesTmp(data.numIncdSnps-numQTL);
        for (unsigned j=0, k=0; j<data.numIncdSnps; ++j) {
            if (data.incdSnpInfoVec[j]->isQTL) continue;
            Ztmp.col(k) = data.Z.col(j);
            ZPZdiagTmp[k] = data.ZPZdiag[j];
            snp2pqTmp[k] = data.snp2pq[j];
            snpEffectNamesTmp[k] = data.snpEffectNames[j];
            ++k;
        }
        data.Z = Ztmp;
        data.ZPZdiag = ZPZdiagTmp;
        data.snp2pq = snp2pqTmp;
        data.snpEffectNames = snpEffectNamesTmp;
        data.numIncdSnps -= numQTL;
    }
    
    if (!myMPI::rank) {
        cout << "\nSimulated " << numQTL << " QTL with " << numQTL - numFDC << " QTL escaped from XCI." << endl;
        cout << "Simulated genotypic variance: " << genVar << endl;
        cout << "Simulated residual  variance: " << resVar << endl;
        if (removeQTL)
            cout << "QTL removed from the analysis." << endl;
    }
}

void XCI::outputResults(const Data &data, const vector<McmcSamples*> &mcmcSampleVec, const string &title){
    McmcSamples *snpEffects = NULL;
    McmcSamples *gamma = NULL;
    for (unsigned i=0; i<mcmcSampleVec.size(); ++i) {
        if (mcmcSampleVec[i]->label == "SnpEffects") snpEffects = mcmcSampleVec[i];
        if (mcmcSampleVec[i]->label == "Gamma") gamma = mcmcSampleVec[i];
    }
    if (myMPI::rank) return;
    string filename = title + ".snpRes";
    ofstream out(filename.c_str());
    out << boost::format("%6s %20s %6s %12s %8s %12s %8s %8s\n")
    % "Id"
    % "Name"
    % "Chrom"
    % "Position"
    % "GeneFrq"
    % "Effect"
    % "PIP"
    % "PrNDC";
    for (unsigned i=0, idx=0; i<data.numSnps; ++i) {
        SnpInfo *snp = data.snpInfoVec[i];
        if(!data.fullSnpFlag[i]) continue;
        if(snp->isQTL) continue;
        out << boost::format("%6s %20s %6s %12s %8.3f %12.6f %8.3f %8.3f\n")
        % (idx+1)
        % snp->ID
        % snp->chrom
        % snp->physPos
        % snp->af
        % snpEffects->posteriorMean[idx]
        % snpEffects->pip[idx]
        % gamma->posteriorMean[idx];
        ++idx;
    }
    out.close();
}


void BayesXCI::SnpEffects::sampleFromFC(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag, const VectorXf &ZPZdiagMale,
                                             const VectorXf &ZPZdiagFemale, const unsigned nmale, const unsigned nfemale, const float p,
                                             const float sigmaSq, const float pi, const float vare, VectorXf &gamma, VectorXf &ghat){
    // sample beta, delta, gamma jointly
    // f(beta, delta, gamma) propto f(beta | delta, gamma) f(delta | gamma) f(gamma)
    
    sumSq = 0.0;
    numNonZeros = 0;
    
    ghat.setZero(ycorr.size());
    
    Vector2f rhsFemale;     // 0: FDC, 1: NDC, corresponding to gamma
    Vector2f my_rhs, rhs;   // 0: FDC, 1: NDC, corresponding to gamma
    Vector2f invLhs;        // 0: FDC, 1: NDC, corresponding to gamma
    Vector2f uhat;          // 0: FDC, 1: NDC, corresponding to gamma
    Vector2f logGamma;      // 0: FDC, 1: NDC, corresponding to gamma
    
    float oldSample, sample;
    float logDelta0, logDelta1, probDelta1;
    float logPi = log(pi);
    float logPiComp = log(1.0-pi);
    float logSigmaSq = log(sigmaSq);
    float invVare = 1.0f/vare;
    float invSigmaSq = 1.0f/sigmaSq;
    float logP = log(p);
    float logPcomp = log(1.0f-p);
    float rhsMale;
    float probGamma1;
    float sampleGamma;
    
    for (unsigned i=0; i<size; ++i) {
        oldSample = values[i];
        ycorr.head(nmale) += Z.col(i).head(nmale) * oldSample;
        if (gamma[i])
            ycorr.tail(nfemale) += Z.col(i).tail(nfemale) * oldSample;
        else
            ycorr.tail(nfemale) += Z.col(i).tail(nfemale) * oldSample * 0.5f;
        
        rhsMale = Z.col(i).head(nmale).dot(ycorr.head(nmale)) * invVare;
        rhsFemale[1] = Z.col(i).tail(nfemale).dot(ycorr.tail(nfemale)) * invVare;
        rhsFemale[0] = rhsFemale[1] * 0.5f;
        
        my_rhs[1] = rhsMale + rhsFemale[1];
        my_rhs[0] = rhsMale + rhsFemale[0];
        
        MPI_Allreduce(&my_rhs[0], &rhs[0], 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        invLhs[1] = 1.0f/((ZPZdiagMale[i] + ZPZdiagFemale[i]      )*invVare + invSigmaSq);
        invLhs[0] = 1.0f/((ZPZdiagMale[i] + ZPZdiagFemale[i]*0.25f)*invVare + invSigmaSq);
        
        uhat.array() = invLhs.array()*rhs.array();
        
        //sample gamma
        
        logGamma[1] = 0.5f*rhs[1]*uhat[1] + logf(sqrt(invLhs[1])*(1.0f-pi) + expf(0.5f*(logSigmaSq-rhs[1]*uhat[1]))*pi) + logP;
        logGamma[0] = 0.5f*rhs[0]*uhat[0] + logf(sqrt(invLhs[0])*(1.0f-pi) + expf(0.5f*(logSigmaSq-rhs[0]*uhat[0]))*pi) + logPcomp;
        probGamma1 = 1.0f/(1.0f + expf(logGamma[0] - logGamma[1]));
        sampleGamma = bernoulli.sample(probGamma1);
        gamma[i] = sampleGamma;
        
        
        // sample delta
        
        logDelta1 = 0.5*(logf(invLhs[sampleGamma]) + uhat[sampleGamma]*rhs[sampleGamma]) + logPiComp;
        logDelta0 = 0.5*logSigmaSq + logPi;
        probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
        
        if (bernoulli.sample(probDelta1)) {
            
            // sample effect
            
            sample = normal.sample(uhat[sampleGamma], invLhs[sampleGamma]);
            values[i] = sample;
            sumSq += sample * sample;
            ++numNonZeros;
            
            ycorr.head(nmale) -= Z.col(i).head(nmale) * sample;
            ghat .head(nmale) += Z.col(i).head(nmale) * sample;
            
            if (gamma[i]) {
                ycorr.tail(nfemale) -= Z.col(i).tail(nfemale) * sample;
                ghat .tail(nfemale) += Z.col(i).tail(nfemale) * sample;
            } else {
                ycorr.tail(nfemale) -= Z.col(i).tail(nfemale) * sample * 0.5f;
                ghat .tail(nfemale) += Z.col(i).tail(nfemale) * sample * 0.5f;
            }
        }
        else {
            values[i] = 0.0;
        }
    }
}

void BayesXCI::ProbNDC::sampleFromFC(const unsigned numSnps, const unsigned numNDC){
    //cout << numSnps << " " << numNDC << endl;
    float alphaTilde = numNDC + alpha;
    float betaTilde  = numSnps - numNDC + beta;
    value = Beta::sample(alphaTilde, betaTilde);
}

void BayesXCI::Rounding::computeYcorr(const VectorXf &y, const MatrixXf &X, const MatrixXf &Z,
                                           const VectorXf &gamma, const unsigned int nmale, const unsigned int nfemale,
                                           const VectorXf &fixedEffects, const VectorXf &snpEffects, VectorXf &ycorr){
    if (count++ % 100) return;
    VectorXf oldYcorr = ycorr;
    ycorr = y - X*fixedEffects;
    for (unsigned i=0; i<snpEffects.size(); ++i) {
        if (snpEffects[i]) {
            if (gamma[i]) {
                ycorr -= Z.col(i)*snpEffects[i];
            } else {
                ycorr.head(nmale) -= Z.col(i).head(nmale)*snpEffects[i];
                ycorr.tail(nfemale) -= Z.col(i).tail(nfemale)*snpEffects[i]*0.5f;
            }
        }
    }
    float my_ss = (ycorr - oldYcorr).squaredNorm();
    float ss;
    MPI_Allreduce(&my_ss, &ss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    value = sqrt(ss);
}

void BayesXCI::sampleUnknowns(){
    fixedEffects.sampleFromFC(ycorr, data.X, data.XPXdiag, vare.value);
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(ycorr, data.Z, data.ZPZdiag, ZPZdiagMale, ZPZdiagFemale,
                                nmale, nfemale, p.value, sigmaSq.value, pi.value, vare.value, gamma.values, ghat);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);
    p.sampleFromFC(snpEffects.size, gamma.values.sum());
    sigmaSq.sampleFromFC(snpEffects.sumSq, snpEffects.numNonZeros);
    if(estimatePi) pi.sampleFromFC(snpEffects.size, snpEffects.numNonZeros);
    vare.sampleFromFC(ycorr);
    
    varg.compute(ghat);
    hsq.compute(varg.value, vare.value);
    
    rounding.computeYcorr(data.y, data.X, data.Z, gamma.values, nmale, nfemale, fixedEffects.values, snpEffects.values, ycorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
    
    static unsigned iter = 0;
    if (++iter < 5000) {
        genVarPrior += (varg.value - genVarPrior)/iter;
        piPrior += (pi.value - piPrior)/iter;
        scale.compute(genVarPrior, piPrior, sigmaSq.scale);
    }
}

