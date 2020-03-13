#include "options.hpp"
#include <sys/stat.h>


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
        else if (!strcmp(argv[i], "--restart")) {
            restart = true;
            ss << "--restart " << "\n";
        }
        else if (!strcmp(argv[i], "--sparse-sync")) {
            sparseSync = true;
            ss << "--sparse-sync " << "\n";
        }
        else if (!strcmp(argv[i], "--mpibayes")) {
            analysisType = "RAM";
            bayesType = argv[++i];
            ss << "--mpibayes " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--bed-to-sparse")) {
            bedToSparse = true;
            ss << "--bed-to-sparse " << "\n";
        }
        else if (!strcmp(argv[i], "--blocks-per-rank")) {
            blocksPerRank = atoi(argv[++i]);
            ss << "--blocks-per-rank " << "\n";
        }
        else if (!strcmp(argv[i], "--check-RAM")) {
            checkRam = true;
            ss << "--check-RAM " << "\n";
        }
        else if (!strcmp(argv[i], "--restart")) {
            restart = true;
            ss << "--restart " << "\n";
        }
        else if (!strcmp(argv[i], "--raw-update")) {
            deltaUpdate = false;
            ss << "--raw-update " << "\n";
        }
        else if (!strcmp(argv[i], "--check-RAM-tasks")) {
            int tmp = atoi(argv[++i]);
            if (tmp < 0) { printf("FATAL: --check-RAM-tasks passed a negative number!\n"); exit(1); }
            checkRamTasks = tmp;
            ss << "--check-RAM-tasks " << "\n";
        }
        else if (!strcmp(argv[i], "--check-RAM-tasks-per-node")) {
            int tmp = atoi(argv[++i]);
            if (tmp < 0) { printf("FATAL: --check-RAM-tasks-per-node passed a negative number!\n"); exit(1); }
            checkRamTpn = tmp;
            ss << "--check-RAM-tasks-per-node " << "\n";
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
            readFromBedFile = true;
            bedFile         = argv[++i];
            ss << "--bfile " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--threshold-fnz")) {
            threshold_fnz = atof(argv[++i]);
            ss << "--threshold-fnz " << argv[i] << "\n";
        }        
        else if (!strcmp(argv[i], "--pheno")) {
            phenotypeFile = argv[++i];
            ss << "--pheno " << argv[i] << "\n";
            string substr;
            stringstream stream(phenotypeFile);
            while(stream.good()) {
                getline(stream, substr, ',');
                //cout << "adding phenotype file: " << substr << endl;
                phenotypeFiles.push_back(substr);
            }
            
            int nphen = phenotypeFiles.size();
            //cout << nphen << " phenotype files passed on CL option" << endl;
            if (nphen != 1 && nphen != 2 && nphen != 4 && nphen != 8) {
                cout << "Passed " << nphen << " phenotype files via CL option --pheno; Only accepts either 1, 2, 4 or 8 input phenotypes! (to fill evenly up to 8)" << endl;
                exit(0);
            }

            //EO: trick: if 2 or more passed, fill up to 8 and activate multi-phen option. Otherwise assume single-phen processing.
            // Disabled for now!!
            if (nphen > 1) {
                /*
                for (int i=0; i < 8/nphen - 1; i++) {
                    for (int j=0; j<nphen; j++) {
                        phenotypeFiles.push_back(phenotypeFiles[j]);
                    }
                }
                assert(phenotypeFiles.size() == 8);
                */
                multi_phen = true;
                //nphen = 8;
               
            } else {
                multi_phen = false;
            }
            
            //Print for record
            //EO TODO: get rank and print only for 0
            //cout << "Final list of phenotype file(s) to be processed:" << endl;
            //for (int i=0; i<nphen; i++)
            //    cout << " " << i << ": " << phenotypeFiles[i] << endl;
        }
        // Failure vector file
        else if (!strcmp(argv[i], "--failure")) {
        	failureFile = argv[++i];
        	ss << "--failure " << argv[i] << "\n";
		}
        //Number of quadrature points        
        else if (!strcmp(argv[i], "--quad_points")) {
        	quad_points = argv[++i];
            ss << "--quad_points " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--interleave-phenotypes")) {
            interleave = true;
            ss << "--interleave-phenotypes " << "\n";
        }
        else if (!strcmp(argv[i], "--mcmc-out-dir")) {
            mcmcOutDir = argv[++i];
            ss << "--mcmc-out-dir " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--mcmc-out-name")) {
            mcmcOutNam = argv[++i];
            ss << "--mcmc-out-nam " << argv[i] << "\n";
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
        else if (!strcmp(argv[i], "--sync-rate")) {    //EO
            syncRate = atoi(argv[++i]);
            ss << "--sync-rate " << argv[i] << "\n";
        }
#endif
        else if (!strcmp(argv[i], "--sparse-dir")) {    //EO
            readFromSparseFiles = true;
            sparseDir = argv[++i];
            ss << "--sparse-dir " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--sparse-basename")) {    //EO
            sparseBsn = argv[++i];
            ss << "--sparse-basename " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--number-markers")) {    //EO
            numberMarkers = atoi(argv[++i]);
            ss << "--number-markers " << argv[i] << "\n";
        }
        else if (!strcmp(argv[i], "--number-individuals")) {    //EO
            numberIndividuals = atoi(argv[++i]);
            ss << "--number-individuals " << argv[i] << "\n";
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
                S[j] = stod(strvec[j]);
            }
            ss << "--S " << argv[i] << "\n";
        }
        //marion : include mS for annotations
        else if (!strcmp(argv[i], "--mS")) {
        	mSfile = argv[++i];
            ss << "--mS " << argv[i] << "\n";
        }
        //marion : include annotation index file
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
            //cout << "covariatesFile = " << covariatesFile << endl;
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


    //EO: check output directory exists or can be created
    int ldir = mcmcOutDir.length();
    if (ldir == 0)
        throw "--mcmc-out-dir CL option has to be set!";

    int lnam = mcmcOutNam.length();
    if (lnam == 0)
        throw "--mcmc-out-nam CL option has to be set!";

    struct stat buffer;
    if (stat (mcmcOutDir.c_str(), &buffer) != 0) {
        if (system(("mkdir -p " + mcmcOutDir).c_str()) != 0)
            throw "could not create output directory --mcmc-out-dir " + mcmcOutDir;
    }
    
    string dir_tar = mcmcOutDir + "/tarballs";
    if (stat (dir_tar.c_str(), &buffer) != 0) {
        if (system(("mkdir -p " + dir_tar).c_str()) != 0)
            throw "could not create tarballs subdirectory " + dir_tar;
    }
    
    mcmcOut = mcmcOutDir + "/" + mcmcOutNam;


    //EO: sparseDir and sparseBsn must be either both set or unset
    //------------------------------------------------------------
    if ( (!sparseBsn.empty() && sparseDir.empty()) || (sparseBsn.empty() && !sparseDir.empty()))
        throw "--sparse-dir and --sparse-basename must either be both set or unset";

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
        } else if (key == "analysisType") {
            analysisType = value;
        } else if (key == "bayesType") {
            bayesType = value;
        } else if (key == "mcmcOut") {
            mcmcOut = value;
        } else if (key == "shuffleMarkers") {
            shuffleMarkers = stoi(value);
#ifdef USE_MPI
        } else if (key == "syncRate") {
            syncRate = stoi(value);
        } else if (key == "blocksPerRank") {
            blocksPerRank = stoi(value);
        }
#endif
        else if (key == "numberMarkers") {
            numberMarkers = stoi(value);
        } else if (key == "numberIndividuals") {
            numberIndividuals = stoi(value);
        } else if (key == "chainLength") {
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

