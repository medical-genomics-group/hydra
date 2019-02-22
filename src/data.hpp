//
//  data.hpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

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
#include <mpi.h>
#include "gadgets.hpp"
#include "mympi.hpp"

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

    VectorXf genotypes; // temporary storage of genotypes of individuals used for building sparse Z'Z

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
    bool isProximal(const SnpInfo &snp2, const float genWindow) const;
    bool isProximal(const SnpInfo &snp2, const unsigned physWindow) const;
};

class ChromInfo {
public:
    const int id;
    const unsigned size;
    const int startSnpIdx;
    const int endSnpIdx;

    ChromInfo(const int id, const unsigned size, const int startSnp, const int endSnp): id(id), size(size), startSnpIdx(startSnp), endSnpIdx(endSnp){}
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

class Data {
public:
    Data();

    // mmap related data
    int ppBedFd;
    int sqNormFd;
    float *ppBedMap;
    float *sqNormMap;
    Map<MatrixXf> mappedZ;
    Map<VectorXf> mappedZPZDiag;

    // Original data
    MatrixXf X;              // coefficient matrix for fixed effects
    MatrixXf Z;              // coefficient matrix for SNP effects
    VectorXf D;              // 2pqn
    VectorXf y;              // phenotypes
    VectorXi G; 			 //groups

    //SparseMatrix<float> ZPZ; // sparse Z'Z because LE is assumed for distant SNPs
    vector<VectorXf> ZPZ;
    SparseMatrix<float> ZPZinv;

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

    float ypy;               // y'y the total sum of squares adjusted for the mean
    float varGenotypic;
    float varResidual;

    vector<SnpInfo*> snpInfoVec;
    vector<IndInfo*> indInfoVec;

    map<string, SnpInfo*> snpInfoMap;
    map<string, IndInfo*> indInfoMap;

    vector<SnpInfo*> incdSnpInfoVec;
    vector<IndInfo*> keptIndInfoVec;

    vector<string> fixedEffectNames;
    vector<string> snpEffectNames;

    set<int> chromosomes;
    vector<ChromInfo*> chromInfoVec;

    vector<bool> fullSnpFlag;
    vector<vector<SnpInfo*> > mldmVec;

    unsigned numFixedEffects;
    unsigned numSnps;
    unsigned numInds;
    unsigned numIncdSnps;
    unsigned numKeptInds;
    unsigned numChroms;

    void preprocessBedFile(const string &bedFile, const string &preprocessedBedFile, const string &sqNormFile);
    void mapPreprocessBedFile(const string &preprocessedBedFile, const string &sqNormFile);
    void unmapPreprocessedBedFile();

    void readFamFile(const string &famFile);
    void readBimFile(const string &bimFile);
    void readBedFile_noMPI(const string &bedFile);
    void readBedFile(const string &bedFile);

    void getSnpDataFromBedFileUsingMmap_openmp(const string &bedFile, const size_t snpLenByt, const long memPageSize, const uint spnInd, VectorXf &snpData);
    void getSnpDataFromBedFileUsingMmap(const string &bedFile, const size_t snpLenByt, const long memPageSize, const uint spnInd, VectorXf &snpData);
    void getSnpDataFromBedFileUsingMmap_new(const int fd, const size_t nb, const long memPageSize, const uint snpInd, double   *snpDat);
    void getSnpDataFromBedFileUsingMmap_new(const int fd, const size_t nb, const long memPageSize, const uint snpInd, VectorXd &snpDat);

    void readPhenotypeFile(const string &phenFile, const unsigned mphen);
    void readCovariateFile(const string &covarFile);
    void keepMatchedInd(const string &keepIndFile, const unsigned keepIndMax);
    void includeSnp(const string &includeSnpFile);
    void excludeSnp(const string &excludeSnpFile);
    void includeChr(const unsigned chr);
    void includeMatchedSnp(void);

    vector<SnpInfo*> makeIncdSnpInfoVec(const vector<SnpInfo*> &snpInfoVec);
    vector<IndInfo*> makeKeptIndInfoVec(const vector<IndInfo*> &indInfoVec);

    void reindexSnp(vector<SnpInfo*> snpInfoVec);
    void readGroupFile(const string &groupFile);
};

#endif /* data_hpp */
