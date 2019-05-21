#ifndef data_hpp
#define data_hpp

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <set>
#include <bitset>
#include <iomanip>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <boost/format.hpp>
#include "gadgets.hpp"


using namespace std;
using namespace Eigen;

class SnpInfo {
public:
    const string ID;
    const string a1; // the referece allele
    const string a2; // the coded allele
    const int chrom;
    const float genPos;
    const int physPos;

    int index;
    int window;
    int windStart;  // for window surrounding the SNP
    int windSize;   // for window surrounding the SNP
    float af;       // allele frequency
    bool included;  // flag for inclusion in panel
    bool isQTL;     // for simulation

    SnpInfo(const int idx, const string &id, const string &allele1, const string &allele2,
            const int chr, const float gpos, const int ppos)
    : ID(id), index(idx), a1(allele1), a2(allele2), chrom(chr), genPos(gpos), physPos(ppos) {
        window = 0;
        windStart = -1;
        windSize  = 0;
        af = -1;
        included = true;
        isQTL = false;
    };

    void resetWindow(void) {windStart = -1; windSize = 0;};
};

class IndInfo {
public:
    const string famID;
    const string indID;
    const string catID;    // catenated family and individual ID
    const string fatherID;
    const string motherID;
    const int famFileOrder; // original fam file order
    const int sex;  // 1: male, 2: female

    int index;
    bool kept;

    float phenotype;

    VectorXf covariates;  // covariates for fixed effects

    IndInfo(const int idx, const string &fid, const string &pid, const string &dad, const string &mom, const int sex)
    : famID(fid), indID(pid), catID(fid+":"+pid), fatherID(dad), motherID(mom), index(idx), famFileOrder(idx), sex(sex) {
        phenotype = -9;
        kept = true;
    }
};

// An entry for the index to the compressed preprocessed bed file
struct IndexEntry {
    long pos;
    long size;
};

using PpBedIndex = std::vector<IndexEntry>;

class Data {
public:
    Data();

    // mmap related data
    int ppBedFd;
    double *ppBedMap;
    Map<MatrixXd> mappedZ;
    PpBedIndex ppbedIndex;

    // Original data
    MatrixXf X;              // coefficient matrix for fixed effects
    //MatrixXf Z;              // coefficient matrix for SNP effects
    MatrixXd Z;
    VectorXf D;              // 2pqn
    //VectorXf y;              // phenotypes
    VectorXd y;              // phenotypes
    VectorXi G; // groups

    MatrixXf XPX;            // X'X the MME lhs
    MatrixXf ZPX;            // Z'X the covariance matrix of SNPs and fixed effects
    VectorXf XPXdiag;        // X'X diagonal
    VectorXf ZPZdiag;        // Z'Z diagonal
    VectorXf XPy;            // X'y the MME rhs for fixed effects
    VectorXf ZPy;            // Z'y the MME rhs for snp effects

    VectorXf snp2pq;         // 2pq of SNPs
    VectorXf se;             // se from GWAS summary data
    VectorXf tss;            // total ss (ypy) for every SNP
    VectorXf b;              // beta from GWAS summary data
    VectorXf n;              // sample size for each SNP in GWAS

    vector<SnpInfo*> snpInfoVec;
    vector<IndInfo*> indInfoVec;

    map<string, SnpInfo*> snpInfoMap;
    map<string, IndInfo*> indInfoMap;

    vector<SnpInfo*> incdSnpInfoVec;
    vector<IndInfo*> keptIndInfoVec;

    vector<string> fixedEffectNames;
    vector<string> snpEffectNames;


    vector<bool> fullSnpFlag;
    vector<vector<SnpInfo*> > mldmVec;

    unsigned numFixedEffects;
    unsigned numSnps;
    unsigned numInds;

    vector<int> blocksStarts;
    vector<int> blocksEnds;
    uint        numBlocks;


#ifdef USE_MPI
    void sparse_data_get_sizes(const char* rawdata, const uint NC, const uint NB, size_t* N1, size_t* N2);
    void sparse_data_fill_indices(const char* rawdata, const uint NC, const uint NB, size_t* N1S, size_t* N1L, size_t* N2S, size_t* N2L, size_t N1, size_t N2, size_t* I1, size_t* I2);
    void compute_markers_statistics(const char* rawdata, const uint M, const uint NB, double* mave, double* mstd, uint* msum);
    void get_normalized_marker_data(const char* rawdata, const uint NB, const uint marker, double* Cx);
    void get_normalized_marker_data(const char* rawdata, const uint NB, const uint marker, double* Cx, const double mean, const double std_);
    void preprocess_data(const char* data, const uint NC, const uint NB, double* ppdata, const int rank);
#endif
    //EO to read definitions of blocks of markers to process
    void readMarkerBlocksFile(const string &markerBlocksFile);

    void preprocessBedFile(const string &bedFile, const string &preprocessedBedFile, const string &preprocessedBedIndexFile, bool compress);
    void mapPreprocessBedFile(const string &preprocessedBedFile);
    void unmapPreprocessedBedFile();

    void mapCompressedPreprocessBedFile(const string &preprocessedBedFile, const string &indexFile);
    void unmapCompressedPreprocessedBedFile();

    void readFamFile(const string &famFile);
    void readBimFile(const string &bimFile);
    void readBedFile_noMPI(const string &bedFile);
    void readPhenotypeFile(const string &phenFile);
    void readGroupFile(const string &groupFile);
};

#endif /* data_hpp */
