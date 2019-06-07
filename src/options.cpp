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
            analysisType = "RAMBayes";
            bayesType = argv[++i];
            ss << "--bayes " << argv[i] << "\n";
        }
#ifdef USE_MPI
        else if (!strcmp(argv[i], "--mpibayes")) {
            analysisType = "RAM";
            bayesType = argv[++i];
            ss << "--mpibayes " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--bed-to-sparse")) {
            bedToSparse = true;
            ss << "--bed-to-sparse " << "\n";
        }
        else if (!strcmp(argv[i], "--read-from-bed-file")) {
            readFromBedFile = true;
            ss << "--read-from-bed-file " << "\n";
        }
        else if (!strcmp(argv[i], "--blocks-per-rank")) {
            blocksPerRank = atoi(argv[++i]);
            ss << "--blocks-per-rank " << "\n";
        }
#endif
        else if (!strcmp(argv[i], "--ppbayes")) {
            analysisType = "PPBayes";
            bayesType = argv[++i];
            ss << "--ppbayes " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--ppasyncbayes")) {
            analysisType = "PPAsyncBayes";
            bayesType = argv[++i];
            ss << "--ppasyncbayes " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--preprocess")) {
            analysisType = "Preprocess";
            ss << "--preprocess " << "\n";
        }
        else if (!strcmp(argv[i], "--compress")) {
            compress = true;
            ss << "--compress " << "\n";
        }
        else if (!strcmp(argv[i], "--bfile")) {
            bedFile = argv[++i];
            ss << "--bfile " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--pheno")) {
            phenotypeFile = argv[++i];
            ss << "--pheno " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--mcmc-out")) {
            mcmcOut = argv[++i];
            ss << "--mcmc-out " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--shuf-mark")) {    //EO
            shuffleMarkers = atoi(argv[++i]);
            ss << "--shuf-mark " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--marker-blocks-file")) {
            markerBlocksFile = argv[++i];
            ss << "--marker-blocks-file " << argv[i] << "\n";
        }
#ifdef USE_MPI
        else if (!strcmp(argv[i], "--mpi-sync-rate")) {    //EO
            MPISyncRate = atoi(argv[++i]);
            ss << "--mpi-sync-rate " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--sparse-dir")) {    //EO
            sparseDir = argv[++i];
            ss << "--sparse-dir " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--sparse-basename")) {    //EO
            sparseBsn = argv[++i];
            ss << "--sparse-basename " << argv[i] << "\n";
        }
#endif
        else if (!strcmp(argv[i], "--number-markers")) {    //EO
            numberMarkers = atoi(argv[++i]);
            ss << "--number-markers " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--chain-length")) {
            chainLength = atoi(argv[++i]);
            ss << "--chain-length " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--burn-in")) {
            burnin = atoi(argv[++i]);
            ss << "--burn-in " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--seed")) {
            seed = (uint)atoi(argv[++i]);
            ss << "--seed " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--out")) {
            title = argv[++i];
            ss << "--out " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--thin")) {
            thin = atoi(argv[++i]);
            ss << "--thin " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--save")) {
            save = atoi(argv[++i]);
            ss << "--save " << argv[i] << "\n";
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
        //Daniel, include variance components matrix for groups
        else if (!strcmp(argv[i], "--mS")) {
            Gadget::Tokenizer strvec;
            Gadget::Tokenizer strT;
            strvec.getTokens(argv[++i], " ;");
            strT.getTokens(strvec[0],",");
            mS=Eigen::MatrixXd(strvec.size(),strT.size());
            numGroups=strvec.size();
            for (unsigned j=0; j<strvec.size(); ++j) {
                strT.getTokens(strvec[j],",");
                for(unsigned k=0;k<strT.size();++k)
                    mS(j,k) = stod(strT[k]);
            }
            ss << "--mS " << argv[i] << "\n";
        }
        //Daniel group assignment file
        else if (!strcmp(argv[i], "--group")) {
            groupFile = argv[++i];
            ss << "--group " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--thread")) {
            numThread = atoi(argv[++i]);
            ss << "--thread " << argv[i] << "\n";
        }
        else if(!strcmp(argv[i], "--thread-spawned")) {
            numThreadSpawned = atoi(argv[++i]);
            ss << "--thread-spawned " << argv[i] << "\n";
        }
        else if(!strcmp(argv[i], "--covariates")) {
            covariates = true;
            covariatesFile = argv[++i];
            ss << "--covariates " << argv[i] << "\n";
        }
        else {
            stringstream errmsg;
            errmsg << "\nError: invalid option \"" << argv[i] << "\".\n";
            throw (errmsg.str());
        }
    }

    options_s = ss.str();
    //cout << ss.str() << endl;
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
        } else if (key == "bedFile") {
            bedFile = value;
        } else if (key == "analysisType") {
            analysisType = value;
        } else if (key == "bayesType") {
            bayesType = value;
        } else if (key == "mcmcOut") {
            mcmcOut = value;
        } else if (key == "shuffleMarkers") {
            shuffleMarkers = stoi(value);
#ifdef USE_MPI
        } else if (key == "MPISyncRate") {
            MPISyncRate = stoi(value);
        } else if (key == "blocksPerRank") {
            blocksPerRank = stoi(value);
        }
#endif
        else if (key == "numberMarkers") {
            numberMarkers = stoi(value);
        }
        else if (key == "chainLength") {
            chainLength = stoi(value);
        } else if (key == "burnin") {
            burnin = stoi(value);
        } else if (key == "seed") {
            seed = (uint)stoi(value);
        } else if (key == "thin") {
            thin = stoi(value);
        } else if (key == "save") {
            save = stoi(value);
        } else if (key == "S") {
            Gadget::Tokenizer strvec;
            strvec.getTokens(value, " ,");
            S.resize(strvec.size());
            for (unsigned j=0; j<strvec.size(); ++j) {
                S[j] = stof(strvec[j]);
            }
        } else if (key == "numThread") {
            numThread = stoi(value);
        } else if (key.substr(0,2) == "//" ||
                key.substr(0,1) == "#") {
            continue;
        } else {
            throw("\nError: invalid option " + key + " " + value + "\n");
        }
        ss << boost::format("%20s %-1s %-20s\n") %key %":" %value;
    }
    in.close();
    cout << ss.str() << endl;
}


void Options::printBanner(void) {
    cout << "\n";
    cout << "***********************************************\n";
    cout << "* BayesRRcmd                                  *\n";
    cout << "* Complex Trait Genetics group UNIL           *\n";
    cout << "*                                             *\n";
    cout << "* MIT License                                 *\n";
    cout << "***********************************************\n\n";
}


void Options::printProcessingOptions(void) {
    cout << options_s << endl;
}


void Options::makeTitle(void){
    title = optionFile;
    size_t pos = optionFile.rfind('.');
    if (pos != string::npos) {
        title = optionFile.substr(0,pos);
    }
}

