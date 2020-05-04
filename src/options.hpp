#ifndef options_hpp
#define options_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <string>
#include <limits.h>
#include <boost/format.hpp>
#include "gadgets.hpp"
#include <Eigen/Eigen>

using namespace std;
using namespace boost;

const unsigned Megabase = 1e6;

class Options {
public:

    unsigned shuffleMarkers      = 1;
#ifdef USE_MPI
    //bool     mpiBayesGroups    = false;
    string   groupIndexFile    = "";
    string   groupMixtureFile  = "";
    bool     restart             = false;
    bool     sparseSync          = false;
    bool     bedSync             = false;
    unsigned syncRate            = 1;
    bool     bedToSparse         = false;
    bool     readFromBedFile     = false;
    bool     readFromSparseFiles = false;
    bool     mixedRepresentation = false;
    unsigned blocksPerRank       = 1;     //EO: for bed -> sparse conversion, to split blocks if too large
    bool     checkRam            = false;
    unsigned checkRamTasks       = 0;
    unsigned checkRamTpn         = 0;
#endif
    unsigned numberMarkers       = 0;
    unsigned numberIndividuals   = 0;
    unsigned chainLength;
    unsigned burnin;
    unsigned seed;
    unsigned thin;                       // save every this th sampled value in MCMC
    unsigned save;                       // sampling rate of the epsilon vector
    vector<double> S;                    // variance components
 
    //marion : include annotation variables
    unsigned int numGroups;
    Eigen::MatrixXd mS;
    string groupFile;
    string mSfile;
    string failureFile;
    string bayesW_version;
    string quad_points;
    string priorsFile;
    string dPriorsFile;

    string title;
    string analysisType;
    string bayesType;

    string phenotypeFile;
    bool   multi_phen = false;
    vector<string> phenotypeFiles;
    bool   interleave = false;

    string markerBlocksFile;
    string bedFile;
    string mcmcOutDir  = "";
    string mcmcOutNam  = "default_output_name";
    string mcmcOut     = "default_output_name";
    string sparseDir   = "";
    string sparseBsn   = "";
    string optionFile;
    string covariatesFile;         // for extra covariates.
    bool   covariates  = false;    // for extra covatiates.
    bool   compress    = false;
    bool   deltaUpdate = true;     // Use the delta epsilon to pass the message in mpi

    // Use BED representation over SPARSE if fraction of non-zero elements (fnz)
    // is greater than this threshold:
    double threshold_fnz = 0.06;

    double betaA;
    double betaB;
    string options_s;

    Options(){
        chainLength             = 10000;
        burnin                  = 5000;
        seed                    = static_cast<unsigned int>(std::time(0));
        thin                    = 5;
        save                    = 10;
        S.resize(3);
        S[0]                    = 0.01;
        S[1]                    = 0.001;
        S[2]                    = 0.0001;
        title                   = "brr";
        analysisType            = "Bayes";
        bayesType               = "C";
        phenotypeFile           = "";
        markerBlocksFile        = "";
        bedFile                 = "";
        sparseDir               = "";
        sparseBsn               = "";
        optionFile				= "";
        //EO@@@ check for default value
	//DT@@@ should be one group the default allowing for no group files
        numGroups				= 1;
        groupFile               = "";
        priorsFile              = "";
        dPriorsFile             = "";
        mSfile                  = "";
        betaA = 1.0;
        betaB = 1.0;
    }

    void inputOptions(const int argc, const char* argv[]);

    void printBanner(void);
    void printProcessingOptions(void);

private:
    void readFile(const string &file);
    void makeTitle(void);
    void seedEngine(void);
};

#endif /* options_hpp */
