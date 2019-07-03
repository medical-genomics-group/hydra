#include <iostream>
#include <string>
#include "BayesRRm.h"
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

        if (opt.bedToSparse == true) {
            data.readFamFile(opt.bedFile + ".fam");
            data.readBimFile(opt.bedFile + ".bim");
            //data.readPhenotypeFile(opt.phenotypeFile);
            BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));
            analysis.write_sparse_data_files(opt.blocksPerRank);

        } else if (opt.bayesType == "bayesMPI" && opt.analysisType == "RAM") {
            
            if (opt.readFromBedFile) {
                data.readFamFile(opt.bedFile + ".fam");
                data.readBimFile(opt.bedFile + ".bim");
                data.readPhenotypeFile(opt.phenotypeFile);
                // Read in covariates file if passed
               if (opt.covariates) {
		 std::cout << "reading covariates file: "  << opt.covariatesFile;
                   data.readCovariateFile(opt.covariatesFile);
               }
            } else {
                data.readPhenotypeFile(opt.phenotypeFile, opt.numberIndividuals);
                if (opt.covariates) {

                   std::cout << "reading covariates file: "  << opt.covariatesFile;
                    data.readCovariateFile(opt.covariatesFile);
                }

            }

            BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));

            if (opt.markerBlocksFile != "") {
                data.readMarkerBlocksFile(opt.markerBlocksFile);
            } else {
                if (rank == 0) cout << "INFO   : no request to read a block markers definition file." << endl;
            }

            analysis.runMpiGibbs();

        } else if (opt.analysisType == "RAMBayes" && ( opt.bayesType == "bayes" || opt.bayesType == "bayesMmap" || opt.bayesType == "horseshoe")) {

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
                analysis.runGibbs();

                data.unmapPreprocessedBedFile();
                end = clock();
                printf("OVERALL read+compute time = %.3f sec.\n", double(end - start) / double(CLOCKS_PER_SEC));
            }
        }
#endif
        else {
            throw(" Error: Wrong analysis requested: " + opt.analysisType + " + " + opt.bayesType);
        }

        //#endif

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
