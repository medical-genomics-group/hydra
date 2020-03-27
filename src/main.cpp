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
//#include "tbb/task_scheduler_init.h"
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

#ifdef USE_MPI

        if (opt.bedToSparse || opt.checkRam) {

            data.readFamFile(opt.bedFile + ".fam");
            data.readBimFile(opt.bedFile + ".bim");

            BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));

            if (opt.bedToSparse) {
                analysis.write_sparse_data_files(opt.blocksPerRank);
            } else if (opt.checkRam) {
                analysis.checkRamUsage();
            }


            //EO@@@ } else if ((opt.bayesType == "bayesMPI" && opt.analysisType == "RAM") || opt.mpiBayesGroups) {
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
                        }else{
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
                //BayesRRm_mt analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                //analysis.runMpiGibbsMultiTraits();

            } else if(opt.bayesType == "bayesWMPI"){
                //data.readBedFile_noMPI_unstandardised(opt.bedFile+".bed"); // This part to read the non-standardised data
                BayesW analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                analysis.runMpiGibbs_bW();

            } else {
                BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                analysis.runMpiGibbs();
            }

        }
        
        else if (opt.analysisType == "RAMBayes" && ( opt.bayesType == "bayes" || opt.bayesType == "bayesMmap" || opt.bayesType == "horseshoe")) {
#else
        // RAM solution (analysisType = RAMBayes)
        if (opt.analysisType == "RAMBayes" && ( opt.bayesType == "bayes" || opt.bayesType == "bayesMmap" || opt.bayesType == "horseshoe")) {
#endif
            clock_t start = clock();
            
            // Read input files
            data.readFamFile(opt.bedFile + ".fam");
            data.readBimFile(opt.bedFile + ".bim");
            data.readPhenotypeFile(opt.phenotypeFile);
            
            // Limit number of markers to process
            if (opt.numberMarkers > 0 && opt.numberMarkers < data.numSnps)
                data.numSnps = opt.numberMarkers;
            
            data.readBedFile_noMPI(opt.bedFile+".bed");
            
            // Option bayesType="bayesMmap" is going to be deprecated
            if (opt.bayesType == "bayesMmap" || opt.bayesType == "bayes"){
                BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                analysis.runGibbs();
            } else if (opt.bayesType == "horseshoe") {
                //TODO Finish horseshoe
            } else if (opt.bayesType == "bayesW") {
                //TODO Add BayesW
            } else if (opt.bayesType == "bayesG") {
                //TODO add Bayes groups
            }
            
            clock_t end   = clock();
            printf("OVERALL read+compute time = %.3f sec.\n", (float)(end - start) / CLOCKS_PER_SEC);
        }
        
        // Pre-processing the data (centering and scaling)
        else if (opt.analysisType == "Preprocess") {
            cout << "Start preprocessing " << opt.bedFile + ".bed" << endl;
            
            clock_t start_bed = clock();
            data.preprocessBedFile(opt.bedFile + ".bed",
                                   opt.bedFile + ".ppbed",
                                   opt.bedFile + ".ppbedindex",
                                   opt.compress);
            
            clock_t end = clock();
            printf("Finished preprocessing the bed file in %.3f sec.\n", double(end - start_bed) / double(CLOCKS_PER_SEC));
            cout << endl;
        }

#ifndef USE_MPI
        else if (opt.analysisType == "PPBayes" || opt.analysisType == "PPAsyncBayes") {
            clock_t start = clock();
            data.readPhenotypeFile(opt.phenotypeFile);
            // Run analysis using mapped data files
            if (opt.compress) {
                cout << "Start reading preprocessed bed file: " << opt.bedFile + ".ppbed" << endl;
                clock_t start_bed = clock();
                data.mapCompressedPreprocessBedFile(opt.bedFile + ".ppbed",
                                                    opt.bedFile + ".ppbedindex");
                clock_t end = clock();
                printf("Finished reading preprocessed bed file in %.3f sec.\n", double(end - start_bed) / double(CLOCKS_PER_SEC));
                cout << endl;
                
                std::unique_ptr<tbb::task_scheduler_init> taskScheduler { nullptr };
                if (opt.numThreadSpawned > 0)
                    taskScheduler = std::make_unique<tbb::task_scheduler_init>(opt.numThreadSpawned);
                
                BayesRRmz analysis(data, opt);
                analysis.runGibbs();
                data.unmapCompressedPreprocessedBedFile();
            } else {
                cout << "Start reading preprocessed bed file: " << opt.bedFile + ".ppbed" << endl;
                clock_t start_bed = clock();
                data.mapPreprocessBedFile(opt.bedFile + ".ppbed");
                clock_t end = clock();
                printf("Finished reading preprocessed bed file in %.3f sec.\n", double(end - start_bed) / double(CLOCKS_PER_SEC));
                cout << endl;
                
                BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                //here we check if we want to restart a chain
                if(opt.restart){
                    //TODO function to read output file
                    //TODO function to run restarted mpi
                    
                }
                else{
                    analysis.runGibbs();
                }
                data.unmapPreprocessedBedFile();
                end = clock();
                printf("OVERALL read+compute time = %.3f sec.\n", double(end - start) / double(CLOCKS_PER_SEC));
            }
        }
#endif
        else {
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
