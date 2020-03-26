#include <iostream>
#include <string>
#include "BayesRRm.h"
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

        } else if ((opt.bayesType == "bayesMPI" && opt.analysisType == "RAM") || opt.mpiBayesGroups) {
            
            if (opt.readFromBedFile) {
                //printf("INFO   : reading from BED file\n");
                data.readFamFile(opt.bedFile + ".fam");
                data.readBimFile(opt.bedFile + ".bim");
                data.readPhenotypeFile(opt.phenotypeFile);

                // Read in covariates file if passed
                if (opt.covariates) {
                    //std::cout << "reading covariates file: "  << opt.covariatesFile << endl;
                    data.readCovariateFile(opt.covariatesFile);
                }

            } else { // Read from sparse representation files

                if (opt.multi_phen) {
                    throw("EO: Disabled for now");
                    data.readPhenotypeFiles(opt.phenotypeFiles, opt.numberIndividuals, data.phenosData);
                } else {
                    if (opt.covariates) { // Then combine reading of the .phen & .cov
                        data.readPhenCovFiles(opt.phenotypeFiles[0], opt.covariatesFile, opt.numberIndividuals, data.y, rank);
                    } else {
                        data.readPhenotypeFile(opt.phenotypeFiles[0], opt.numberIndividuals, data.y);
                    }
                }
                //if (opt.covariates) {
                //    std::cout << "reading covariates file: "  << opt.covariatesFile << endl;
                //    data.readCovariateFile(opt.covariatesFile);
                //}
            }

            if (opt.markerBlocksFile != "") {
                data.readMarkerBlocksFile(opt.markerBlocksFile);
            }

            if (opt.mpiBayesGroups) {
                //printf("MPI BAYES GROUPS\n");
                if (opt.groupIndexFile == "") throw("with --mpiBayesGroups activated you must use the --groupIndexFile!");
                data.readGroupFile(opt.groupIndexFile);
                if (opt.groupMixtureFile == "") throw("with --mpiBayesGroups activated you must use the --groupMixtureFile!");
                data.readmSFile(opt.groupMixtureFile);
                // TODO: group priors file should be optional
                data.read_group_priors(opt.priorsFile);
                data.read_dirichlet_priors(opt.dPriorsFile);
            }

            if (opt.multi_phen) {
                //BayesRRm_mt analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                //analysis.runMpiGibbsMultiTraits();
            } else {
                BayesRRm analysis(data, opt, sysconf(_SC_PAGE_SIZE));
                analysis.runMpiGibbs();
            }

        } 
#else
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
