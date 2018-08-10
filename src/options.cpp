//
//  options.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "options.hpp"

void Options::inputOptions(const int argc, const char* argv[]){
    stringstream ss;
    for (unsigned i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "--inp-file")) {
            optionFile = argv[++i];
            readFile(optionFile);
            return;
        } else {
            if (i==1) ss << "\nOptions:\n\n";
        }
        if (!strcmp(argv[i], "--bayes")) {
            analysisType = "Bayes";
            bayesType = argv[++i];
            ss << "--bayes " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--sbayes")) {
            analysisType = "SBayes";
            bayesType = argv[++i];
            ss << "--sbayes " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--make-ldm")) {
            analysisType = "LDmatrix";
            ss << "--make-ldm " << "\n";
        }
        else if (!strcmp(argv[i], "--alg")) {
            algorithm = argv[++i];
            ss << "--alg " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--predict")) {
            analysisType = "Predict";
            ss << "--predict " << "\n";
        }
        else if (!strcmp(argv[i], "--bfile")) {
            bedFile = argv[++i];
            ss << "--bfile " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--pheno")) {
            phenotypeFile = argv[++i];
            ss << "--pheno " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--covar")) {
            covariateFile = argv[++i];
            ss << "--covar " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--mpheno")) {
            mphen = atoi(argv[++i]);
            ss << "--mpheno " << mphen << "\n";
        }
        else if (!strcmp(argv[i], "--keep")) {
            keepIndFile = argv[++i];
            ss << "--keep " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--keep-max")) {
            keepIndMax = atoi(argv[++i]);
            ss << "--keep-max " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--extract")) {
            includeSnpFile = argv[++i];
            ss << "--extract " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--exclude")) {
            excludeSnpFile = argv[++i];
            ss << "--exclude " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--mcmc-samples")) {
            mcmcSampleFile = argv[++i];
            ss << "--mcmc-samples " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--gwas-summary")) {
            gwasSummaryFile = argv[++i];
            ss << "--gwas-summary " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--ldm")) {
            ldmatrixFile = argv[++i];
            ss << "--ldm " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--mldm")) {
            ldmatrixFile = argv[++i];
            multiLDmat = true;
            ss << "--mldm " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--snp-res")) {
            snpResFile = argv[++i];
            ss << "--snp-res " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--wind")) {
            windowWidth = unsigned(atof(argv[++i]) * Megabase);
            ss << "--wind " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--pi")) {
            probFixed = atof(argv[++i]);
            ss << "--pi " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--hsq")) {
            heritability = atof(argv[++i]);
            ss << "--hsq " << argv[i] << "\n";
        }
//        else if (!strcmp(argv[i], "--varg")) {
//            varGenotypic = atof(argv[++i]);
//            ss << "--varg " << argv[i] << "\n";
//        }
//        else if (!strcmp(argv[i], "--vare")) {
//            varResidual = atof(argv[++i]);
//            ss << "--vare " << argv[i] << "\n";
//        }
        else if (!strcmp(argv[i], "--chain-length")) {
            chainLength = atoi(argv[++i]);
            ss << "--chain-length " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--burn-in")) {
            burnin = atoi(argv[++i]);
            ss << "--burn-in " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--out-freq")) {
            outputFreq = atoi(argv[++i]);
            ss << "--out-freq " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--seed")) {
            seed = atoi(argv[++i]);
            ss << "--seed " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--wind-nnz")) {
            snpFittedPerWindow = atoi(argv[++i]);
            ss << "--wind-nnz " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--out")) {
            title = argv[++i];
            ss << "--out " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--no-mcmc-bin")) {
            writeBinPosterior = false;
            ss << "--no-mcmc-bin " << "\n";
        }
        else if (!strcmp(argv[i], "--thin")) {
            thin = atoi(argv[++i]);
            ss << "--thin " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--fix-pi")) {
            estimatePi = false;
            ss << "--fix-pi " << "\n";
        }
        else if (!strcmp(argv[i], "--varS")) {
            varS = atof(argv[++i]);
            ss << "--varS " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--S")) {
            Gadget::Tokenizer strvec;
            strvec.getTokens(argv[++i], " ,");
            S.resize(strvec.size());
            for (unsigned j=0; j<strvec.size(); ++j) {
                S[j] = stof(strvec[j]);
            }
            ss << "--S " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--thread")) {
            numThread = atoi(argv[++i]);
            ss << "--thread " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--chr")) {
            includeChr = atoi(argv[++i]);
            ss << "--chr " << argv[i] << "\n";
        }
        else {
            stringstream errmsg;
            errmsg << "\nError: invalid option \"" << argv[i] << "\".\n";
            throw (errmsg.str());
        }
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myMPI::rank);
    if(myMPI::rank==0) cout << ss.str() << endl;
    
    //if (bayesType == "Cap" || bayesType == "Sap") myMPI::partition = "bycol";
    


}

void Options::readFile(const string &file){  // input options from file
    optionFile = file;
    stringstream ss;
    ss << "\nOptions:\n\n";
    ss << boost::format("%20s %-1s %-20s\n") %"optionFile" %":" %file;
    makeTitle();
    
    ifstream in(file.c_str());
    if (!in) throw ("Error: can not open the file [" + file + "] to read.");
    
    string key, value;
    while (in >> key >> value) {
        if (key == "bedFile") {
            bedFile = value;
        } else if (key == "phenotypeFile") {
            phenotypeFile = value;
        } else if (key == "mpheno") {
            mphen = stoi(value);
        } else if (key == "bedFile") {
            bedFile = value;
        } else if (key == "covariateFile") {
            covariateFile = value;
        } else if (key == "analysisType") {
            analysisType = value;
        } else if (key == "bayesType") {
            bayesType = value;
        } else if (key == "algorithm") {
            algorithm = value;
        } else if (key == "keepIndFile") {
            keepIndFile = value;
        } else if (key == "keepIndMax") {
            keepIndMax = stoi(value);
        } else if (key == "includeSnpFile") {
            includeSnpFile = value;
        } else if (key == "excludeSnpFile") {
            excludeSnpFile = value;
        } else if (key == "mcmcSampleFile") {
            mcmcSampleFile = value;
        } else if (key == "gwasSummaryFile") {
            gwasSummaryFile = value;
        } else if (key == "LDmatrixFile") {
            ldmatrixFile = value;
        } else if (key == "multiLDmatrixFile") {
            ldmatrixFile = value;
            multiLDmat = true;
        } else if (key == "snpResFile") {
            snpResFile = value;
        } else if (key == "windowWidth") {
            windowWidth = unsigned(stof(value) * Megabase);
        } else if (key == "probFixed") {
            probFixed = stof(value);
        } else if (key == "heritability") {
            heritability = stof(value);
//        } else if (key == "varGenotypic") {
//            varGenotypic = stof(value);
//        } else if (key == "varResidual") {
//            varResidual = stof(value);
        } else if (key == "chainLength") {
            chainLength = stoi(value);
        } else if (key == "burnin") {
            burnin = stoi(value);
        } else if (key == "outputFreq") {
            outputFreq = stoi(value);
        } else if (key == "seed") {
            seed = stoi(value);
        } else if (key == "snpFittedPerWindow") {
            snpFittedPerWindow = stoi(value);
        } else if (key == "writeBinPosterior" && value == "No") {
            writeBinPosterior = false;
        } else if (key == "thin") {
            thin = stoi(value);
        } else if (key == "estimatePi" && value == "No") {
            estimatePi = false;
        } else if (key == "outputResults" && value == "No") {
            outputResults = false;
        } else if (key == "varS") {
            varS = stof(value);
        } else if (key == "S") {
            Gadget::Tokenizer strvec;
            strvec.getTokens(value, " ,");
            S.resize(strvec.size());
            for (unsigned j=0; j<strvec.size(); ++j) {
                S[j] = stof(strvec[j]);
            }
        } else if (key == "numThread") {
            numThread = stoi(value);
        } else if (key == "includeChr") {
            includeChr = stoi(value);
        } else if (key.substr(0,2) == "//" ||
                   key.substr(0,1) == "#") {
            continue;
        } else {
            throw("\nError: invalid option " + key + " " + value + "\n");
        }
        ss << boost::format("%20s %-1s %-20s\n") %key %":" %value;
    }
    in.close();
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myMPI::rank);
    if(myMPI::rank==0) cout << ss.str() << endl;
    
    //if (bayesType == "Cap" || bayesType == "Sap") myMPI::partition = "bycol";
    

}

void Options::makeTitle(void){
    title = optionFile;
    size_t pos = optionFile.rfind('.');
    if (pos != string::npos) {
        title = optionFile.substr(0,pos);
    }
}
