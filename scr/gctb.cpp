//
//  gctb.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "gctb.hpp"

void GCTB::inputIndInfo(Data &data, const string &bedFile, const string &phenotypeFile, const string &keepIndFile, const unsigned keepIndMax, const unsigned mphen, const string &covariateFile){
    data.readFamFile(bedFile + ".fam");
    data.readPhenotypeFile(phenotypeFile, mphen);
    data.keepMatchedInd(keepIndFile, keepIndMax);
    data.readCovariateFile(covariateFile);
}

void GCTB::inputSnpInfo(Data &data, const string &bedFile, const string &includeSnpFile, const string &excludeSnpFile, const unsigned includeChr, const bool readGenotypes){
    data.readBimFile(bedFile + ".bim");
    if (!includeSnpFile.empty()) data.includeSnp(includeSnpFile);
    if (!excludeSnpFile.empty()) data.excludeSnp(excludeSnpFile);
    data.includeChr(includeChr);
    data.includeMatchedSnp();
    if (readGenotypes) data.readBedFile(bedFile + ".bed");
}

void GCTB::inputSnpInfo(Data &data, const string &includeSnpFile, const string &excludeSnpFile, const string &gwasSummaryFile, const string &ldmatrixFile, const unsigned includeChr, const bool multiLDmat){
    if (multiLDmat)
        data.readMultiLDmatInfoFile(ldmatrixFile);
    else
        data.readLDmatrixInfoFile(ldmatrixFile + ".info");
    if (!includeSnpFile.empty()) data.includeSnp(includeSnpFile);
    if (!excludeSnpFile.empty()) data.excludeSnp(excludeSnpFile);
    data.includeChr(includeChr);
    data.readGwasSummaryFile(gwasSummaryFile);
    data.includeMatchedSnp();
    if (multiLDmat)
        data.readMultiLDmatBinFile(ldmatrixFile);
    else
        data.readLDmatrixBinFile(ldmatrixFile + ".bin");
    data.buildSparseMME();
}

Model* GCTB::buildModel(Data &data, const string &bedFile, const string &gwasFile, const string &bayesType, const unsigned windowWidth,
                         const float heritability, const float probFixed, const bool estimatePi,
                         const string &algorithm, const unsigned snpFittedPerWindow, const float varS, const vector<float> &S){
    data.initVariances(heritability);
    if (!gwasFile.empty()) {
        if (bayesType == "C")
            return new ApproxBayesC(data, data.varGenotypic, data.varResidual, probFixed, estimatePi);
        else if (bayesType == "S")
            return new ApproxBayesS(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, varS, S, algorithm);
        else
            throw(" Error: Wrong bayes type: " + bayesType + " in the summary-data-based Bayes analysis.");
    }
    if (bayesType == "C") {
        data.readBedFile(bedFile + ".bed");
        return new BayesC(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, algorithm);
    }
    else if (bayesType == "S") {
        data.readBedFile(bedFile + ".bed");
        return new BayesS(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, varS, S, algorithm);
    }
    else if (bayesType == "N") {
        data.readBedFile(bedFile + ".bed");
        data.getNonoverlapWindowInfo(windowWidth);
        return new BayesN(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, snpFittedPerWindow);
    }
    else if (bayesType == "NS") {
        data.readBedFile(bedFile + ".bed");
        data.getNonoverlapWindowInfo(windowWidth);
        return new BayesNS(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, varS, S, snpFittedPerWindow, algorithm);
    }
    else if (bayesType == "Cap") {
        //data.readBedFile(bedFile + ".bed");
        data.buildSparseMME(bedFile + ".bed", windowWidth);
        return new ApproxBayesC(data, data.varGenotypic, data.varResidual, probFixed, estimatePi);
    }
    else if (bayesType == "Sap") {
        data.buildSparseMME(bedFile + ".bed", windowWidth);
        return new ApproxBayesS(data, data.varGenotypic, data.varResidual, probFixed, estimatePi, varS, S, algorithm);
    }
    else {
        throw(" Error: Wrong bayes type: " + bayesType);
    }
}

vector<McmcSamples*> GCTB::runMcmc(Model &model, const unsigned chainLength, const unsigned burnin, const unsigned thin, const unsigned outputFreq, const string &title, const bool writeBinPosterior){
    MCMC mcmc;
    return mcmc.run(model, chainLength, burnin, thin, outputFreq, title, writeBinPosterior);
}

void GCTB::saveMcmcSamples(const vector<McmcSamples*> &mcmcSampleVec, const string &filename){
    for (unsigned i=0; i<mcmcSampleVec.size(); ++i) {
        McmcSamples *mcmcSamples = mcmcSampleVec[i];
        if (mcmcSamples->label == "SnpEffects" )  continue;
        if (mcmcSamples->label == "WindowDelta") continue;
        mcmcSamples->writeDataTxt(filename);
    }
}

void GCTB::outputResults(const Data &data, const vector<McmcSamples*> &mcmcSampleVec, const string &filename){
    for (unsigned i=0; i<mcmcSampleVec.size(); ++i) {
        McmcSamples *mcmcSamples = mcmcSampleVec[i];
        if (mcmcSamples->label == "SnpEffects") {
            //mcmcSamples->readDataBin(mcmcSamples->filename);
            data.outputSnpResults(mcmcSamples->posteriorMean, mcmcSamples->pip, filename + ".snpRes");
        }
        if (mcmcSamples->label == "FixedEffects") {
            data.outputFixedEffects(mcmcSamples->datMat, filename + ".fxdRes");
        }
        if (mcmcSamples->label == "WindowDelta") {
            //mcmcSamples->readDataBin(mcmcSamples->filename);
            data.outputWindowResults(mcmcSamples->posteriorMean, filename + ".window");
        }
    }
}

McmcSamples* GCTB::inputMcmcSamples(const string &mcmcSampleFile, const string &label, const string &fileformat){
    if (myMPI::rank==0) cout << "reading MCMC samples for " << label << endl;
    McmcSamples *mcmcSamples = new McmcSamples(label);
    if (fileformat == "bin") mcmcSamples->readDataBin(mcmcSampleFile + "." + label);
    if (fileformat == "txt") mcmcSamples->readDataTxt(mcmcSampleFile + "." + label);
    return mcmcSamples;
}

void GCTB::estimateHsq(const Data &data, const McmcSamples &snpEffects, const McmcSamples &resVar, const string &filename){
    Heritability hsq;
    hsq.getEstimate(data, snpEffects, resVar);
    hsq.writeRes(filename);
    hsq.writeMcmcSamples(filename);
}

void GCTB::inputSnpResults(Data &data, const string &snpResFile){
    
}

void GCTB::predict(const Data &data, const string &filename){
    
}

void GCTB::clearGenotypes(Data &data){
    data.X.resize(0,0);
}
