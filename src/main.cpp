#include <iostream>
#include <string>
#include "BayesRRm.h"
#include "BayesW.hpp"
#include "BayesRRm_mt.h"
#include "data.hpp"
#include "options.hpp"
#ifndef USE_MPI
#include "BayesRRmz.hpp"
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace std;

int main(int argc, const char * argv[]) {

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    Gadget::Timer timer;
    timer.setTime();
    cout << "\nAnalysis started: " << timer.getDate();
#endif

    if (argc < 2){
        cerr << " \nDid you forget to give the input parameters?\n" << endl;
        exit(1);
    }

    try {

        Options opt;

#ifndef USE_MPI
        opt.printBanner();
#endif
        opt.inputOptions(argc, argv);

        Data data;

        if (opt.bedToSparse || opt.checkRam) {

            data.readFamFile(opt.bedFile + ".fam");
            data.readBimFile(opt.bedFile + ".bim");

            BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));

            if (opt.bedToSparse) {
                analysis.write_sparse_data_files(opt.blocksPerRank);
            } else if (opt.checkRam) {
                analysis.checkRamUsage();
            }

        } else if ((opt.bayesType == "bayesMPI" || opt.bayesType == "bayesWMPI") && opt.analysisType == "RAM") {


            // Reading from BED file
            // ---------------------
            if (opt.readFromBedFile && !opt.readFromSparseFiles) {

                data.readFamFile(opt.bedFile + ".fam");
                data.readBimFile(opt.bedFile + ".bim");

                //EO: no usable for now
                if (opt.multi_phen) {
                    throw("EO: multi-trait disabled for now.");
                    data.readPhenotypeFiles(opt.phenotypeFiles, opt.numberIndividuals, data.phenosData);
                } else {
                    // Read in covariates file if passed
                    if (opt.covariates) {
                        if (opt.bayesType == "bayesWMPI") {
                            data.readPhenFailCovFiles(opt.phenotypeFiles[0], opt.covariatesFile, opt.failureFile, opt.numberIndividuals, data.y, data.fail, rank);
                        } else {
                            data.readPhenCovFiles(opt.phenotypeFiles[0], opt.covariatesFile, opt.numberIndividuals, data.y, rank);
                        }
                    } else {
                        if (opt.bayesType == "bayesWMPI") {
                            data.readPhenFailFiles(opt.phenotypeFiles[0], opt.failureFile, opt.numberIndividuals, data.y, data.fail, rank);
                        } else {
                            data.readPhenotypeFile(opt.phenotypeFile);
                        }
                    }
                }
            }

            // Read from sparse representation files
            // -------------------------------------
            else if (opt.readFromSparseFiles && !opt.readFromBedFile) {

                if (opt.multi_phen) {
                    throw("EO: Disabled for now");
                    data.readPhenotypeFiles(opt.phenotypeFiles, opt.numberIndividuals, data.phenosData);
                } else {

                    // Read in covariates file if passed
                    if (opt.covariates) {
                        if (opt.bayesType == "bayesWMPI") {
                            data.readPhenFailCovFiles(opt.phenotypeFiles[0], opt.covariatesFile, opt.failureFile, opt.numberIndividuals, data.y, data.fail, rank);
                        } else {
                            data.readPhenCovFiles(opt.phenotypeFiles[0], opt.covariatesFile, opt.numberIndividuals, data.y, rank);
                        }
                    } else {
                        if (opt.bayesType == "bayesWMPI") {
                            data.readPhenFailFiles(opt.phenotypeFiles[0], opt.failureFile, opt.numberIndividuals, data.y, data.fail, rank);
                        } else {
                            data.readPhenotypeFile(opt.phenotypeFiles[0], opt.numberIndividuals, data.y);
                        }
                    }
                }
            }

            //EO@@@ mixed should be using sparse + option 

            // Mixing bed in memory + sparse representations
            // ---------------------------------------------
            else if (opt.readFromSparseFiles && opt.readFromBedFile) {

                cout << "EO: CHANGE BEHAVIOUR HERE!" << endl;
                
                //cout << "WARNING: mixed-representation processing type requested!" << endl;
                opt.mixedRepresentation = true;

                data.readPhenotypeFile(opt.phenotypeFiles[0], opt.numberIndividuals, data.y);

            } else {
                cerr << "FATAL: either go for BED, SPARSE or BOTH" << endl;
                exit(1);
            }
            
            if (opt.markerBlocksFile != "") {
                data.readMarkerBlocksFile(opt.markerBlocksFile);
            }

            //EO: groups
            //    by default a single group with all the markers in (zero file passed)
            //    if only one file -> crash
            //    if two files -> define groups
            //
            if ( (opt.groupIndexFile == "" && opt.groupMixtureFile != "") || (opt.groupIndexFile != "" && opt.groupMixtureFile == "") ) {
                throw("FATAL   : you need to activate both --groupIndexFile and --groupMixtureFile");
            }            

            if (opt.priorsFile != "") {
                data.read_group_priors(opt.priorsFile);
            }

            if (opt.dPriorsFile != "") {
                data.read_dirichlet_priors(opt.dPriorsFile);
            }

            if (opt.multi_phen) {
                throw("EO: Disabled for now");
                //BayesRRm_mt analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                //analysis.runMpiGibbsMultiTraits();

            } else if (opt.bayesType == "bayesWMPI") {

                BayesW analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                analysis.runMpiGibbs_bW();

            } else {

                BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                analysis.runMpiGibbs();
            }

        } else {
            throw(" Error: Wrong analysis requested: " + opt.analysisType + " + " + opt.bayesType);
        }
    } 
    catch (const string &err_msg) {
        cerr << "\n" << err_msg << endl;
    }
    catch (const char *err_msg) {
        cerr << "\n" << err_msg << endl;
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#else
    timer.getTime();
    cout << "\nAnalysis finished: " << timer.getDate();
    cout << "Computational time: "  << timer.format(timer.getElapse()) << endl;
#endif
    
    return 0;
}
