//
//  main.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include <iostream>
#include "gctb.hpp"
#include "xci.hpp"
#include "vgmaf.hpp"

using namespace std;


int main(int argc, const char * argv[]) {
    
    MPI_Init(NULL, NULL);
    
    MPI_Comm_size(MPI_COMM_WORLD, &myMPI::clusterSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myMPI::rank);
    MPI_Get_processor_name(myMPI::processorName, &myMPI::processorNameLength);
    
    if (myMPI::rank==0) {
        cout << "***********************************************\n";
        cout << "* GCTB 1.9                                    *\n";
        cout << "* Genome-wide Complex Trait Bayesian analysis *\n";
        cout << "* Author: Jian Zeng                           *\n";
        cout << "* MIT License                                 *\n";
        cout << "***********************************************\n";
        if (myMPI::clusterSize > 1)
            cout << "\nGCTB is using MPI with " << myMPI::clusterSize << " processors" << endl;
    }

    //printf("Hello from processor %s, rank %d\n", myMPI::processorName, myMPI::rank);

//    omp_set_num_threads(2);
//    #pragma omp parallel
//    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());

    Gadget::Timer timer;
    timer.setTime();
    if (myMPI::rank==0) cout << "\nAnalysis started: " << timer.getDate();
    
    if (argc < 2){
        if (myMPI::rank==0) cerr << " \nDid you forget to give the input parameters?\n" << endl;
        exit(1);
    }
    
    try {
        
        Options opt;
        opt.inputOptions(argc, argv);
        
        if (opt.seed) Stat::seedEngine(opt.seed);
        else          Stat::seedEngine(011415);  // fix the random seed if not given due to the use of MPI
                
        Data data;
        bool readGenotypes;
        
        GCTB gctb(opt);


        if (opt.analysisType == "Bayes") {
            readGenotypes = false;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            
            Model *model = gctb.buildModel(data, opt.bedFile, "", opt.bayesType, opt.windowWidth,
                                            opt.heritability, opt.probFixed, opt.estimatePi,
                                            opt.algorithm, opt.snpFittedPerWindow, opt.varS, opt.S);
            vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, opt.chainLength, opt.burnin, opt.thin,
                                                               opt.outputFreq, opt.title, opt.writeBinPosterior);
            //gctb.saveMcmcSamples(mcmcSampleVec, opt.title);
            gctb.clearGenotypes(data);
            if (opt.outputResults) gctb.outputResults(data, mcmcSampleVec, opt.title);
        }
        else if (opt.analysisType == "LDmatrix") {
            readGenotypes = false;
            gctb.inputIndInfo(data, opt.bedFile, opt.bedFile + ".fam", opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            data.makeLDmatrix(opt.bedFile + ".bed", opt.windowWidth, opt.title);
        }
        else if (opt.analysisType == "SBayes") {
            gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.gwasSummaryFile, opt.ldmatrixFile, opt.includeChr, opt.multiLDmat);
            
            Model *model = gctb.buildModel(data, "", opt.gwasSummaryFile, opt.bayesType, opt.windowWidth,
                                            opt.heritability, opt.probFixed, opt.estimatePi,
                                            opt.algorithm, opt.snpFittedPerWindow, opt.varS, opt.S);
            vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, opt.chainLength, opt.burnin, opt.thin,
                                                               opt.outputFreq, opt.title, opt.writeBinPosterior);
            if (opt.outputResults) gctb.outputResults(data, mcmcSampleVec, opt.title);
        }
        else if (opt.analysisType == "hsq") {
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            McmcSamples *resVar     = gctb.inputMcmcSamples(opt.mcmcSampleFile, "ResVar", "txt");
            gctb.estimateHsq(data, *snpEffects, *resVar, opt.title);
        }
        else if (opt.analysisType == "Predict") {
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            
            gctb.inputSnpResults(data, opt.snpResFile);
            gctb.predict(data, opt.title);
        }
        else if (opt.analysisType == "Summarize") {  // ad hoc method for producing summary from binary MCMC samples of SNP effects
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            gctb.clearGenotypes(data);
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            data.summarizeSnpResults(snpEffects->datMatSp, opt.title + ".snpRes");
        }
        else if (opt.analysisType == "XCI") {  // ad hoc method for X chromosome inactivation project
            XCI xci;
            readGenotypes = true;
            xci.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                             opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            if (opt.bayesType == "Simu") {
                xci.simu(data, 1000, 0.02, 0.15, false);  // ad hoc simulation to test BayesXCI method
            }
            Model *model = xci.buildModel(data, opt.heritability, opt.probFixed, opt.estimatePi);
            vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, opt.chainLength, opt.burnin, opt.thin,
                                                               opt.outputFreq, opt.title, opt.writeBinPosterior);
            gctb.saveMcmcSamples(mcmcSampleVec, opt.title);
            gctb.clearGenotypes(data);
            gctb.outputResults(data, mcmcSampleVec, opt.title);
            xci.outputResults(data, mcmcSampleVec, opt.title);
        }
        else if (opt.analysisType == "VGMAF") {  // ad hoc method for cumulative Vg against MAF to detect selection
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
            VGMAF vgmaf;
            if (opt.bayesType == "Simu") {
                vgmaf.simulate(data, opt.title);
            } else {
                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
                vgmaf.compute(data, *snpEffects, opt.burnin, opt.thin, opt.title);
            }
        }
        
        else if (opt.analysisType == "OutputEffectSamples") { // for now an ad hoc method to output the MCMC SNP effect samples in text file
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            data.outputSnpEffectSamples(snpEffects->datMatSp, opt.burnin, opt.outputFreq, opt.snpResFile, opt.title + ".snpEffectSamples");
        }
        
        else {
            throw(" Error: Wrong analysis type: " + opt.analysisType);
        }
    }
    catch (const string &err_msg) {
        if (myMPI::rank==0) cerr << "\n" << err_msg << endl;
    }
    catch (const char *err_msg) {
        if (myMPI::rank==0) cerr << "\n" << err_msg << endl;
    }
    
    timer.getTime();
    
    if (myMPI::rank==0) {
        cout << "\nAnalysis finished: " << timer.getDate();
        cout << "Computational time: "  << timer.format(timer.getElapse()) << endl;
    }
    
    MPI_Finalize();

    return 0;
}
