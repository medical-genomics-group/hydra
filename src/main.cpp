//
//  main.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//
// Modified by Daniel Trejo Banos for memory mapped filed implementation

#include <iostream>
#include "gctb.hpp"
#include "xci.hpp"
#include "vgmaf.hpp"

using namespace std;


int main(int argc, const char * argv[]) {

    

    Gadget::Timer timer;
    timer.setTime();
    
    if (argc < 2){
        cerr << " \nDid you forget to give the input parameters?\n" << endl;
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
            //gctb.inputSnpInfo already called data.readbedfiles


            gctb.clearGenotypes(data);
        }
        
        else {
            throw(" Error: Wrong analysis type: " + opt.analysisType);
        }
    }
    catch (const string &err_msg) {
        cerr << "\n" << err_msg << endl;
    }
    catch (const char *err_msg) {
        cerr << "\n" << err_msg << endl;
    }
    
    timer.getTime();
    

        cout << "\nAnalysis finished: " << timer.getDate();
        cout << "Computational time: "  << timer.format(timer.getElapse()) << endl;
    

    return 0;
}
