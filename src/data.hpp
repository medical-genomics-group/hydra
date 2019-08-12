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
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_utils.hpp"
#endif


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
    MatrixXd X;              // coefficient matrix for fixed effects
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

    unsigned numSnps = 0;
    unsigned numInds = 0;
    unsigned numNAs  = 0;
    vector<uint> NAsInds;
    vector<int> blocksStarts;
    vector<int> blocksEnds;
    uint        numBlocks = 0;


#ifdef USE_MPI
    
    void sparse_data_correct_NA_OLD(const size_t* N1S, const size_t* N2S, const size_t* NMS, 
                                    size_t*       N1L,       size_t* N2L,       size_t* NML,
                                    uint*         I1,        uint*   I2,        uint*   IM,
                                    const int M);

    void sparse_data_correct_for_missing_phenotype(const size_t* NS, size_t* NL, uint* I, const int M);

    void sparse_data_get_sizes_from_raw(const char* rawdata, const uint NC, const uint NB, size_t& N1, size_t& N2, size_t& NM);

    void sparse_data_fill_indices(const char* rawdata, const uint NC, const uint NB,
                                  size_t* N1S, size_t* N1L, uint* I1,
                                  size_t* N2S, size_t* N2L, uint* I2,
                                  size_t* NMS, size_t* NML, uint* IM);

    size_t get_number_of_elements_from_sparse_files(const std::string basename, const std::string id, const int* MrankS, const int* MrankL,
                                                    size_t* S, size_t* L);
    
    /*
    void sparse_data_get_sizes_from_sparse(size_t* N1S, size_t* N1L,
    size_t* N2S, size_t* N2L,
                                           size_t* NMS, size_t* NML,
                                           const int* MrankS, const int* MrankL, const int rank,
                                           const std::string sparseOut,
                                           size_t& N1, size_t& N2, size_t& NM);
    */

    void read_sparse_data_file(const std::string filename, const size_t N, const size_t OFF, const int NREAD, uint* out);

    /*
    void sparse_data_read_files(const size_t N1, const size_t N1SOFF, uint* I1,
                                const size_t N2, const size_t N2SOFF, uint* I2,
                                const size_t NM, const size_t NMSOFF, uint* IM,
                                const std::string sparseOut);
    */

    // MPI_File_read_at_all handling count argument larger than INT_MAX
    template <typename T>
    void mpi_file_read_at_all(const size_t N, const MPI_Offset offset, const MPI_File fh, const MPI_Datatype MPI_DT, const int NREADS, T buffer) {

        int rank, dtsize;
        MPI_Status status;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Type_size(MPI_DT, &dtsize);
        assert(dtsize == sizeof(buffer[0]));

        if (NREADS == 0) return;

        int SPLIT_ON = check_int_overflow(size_t(ceil(double(N)/double(NREADS))), __LINE__, __FILE__);
        int count = SPLIT_ON;

        for (uint i=0; i<NREADS; ++i) {

            const size_t iim = size_t(i) * size_t(SPLIT_ON);

            // Last iteration takes only the leftover
            if (i == NREADS-1) count = check_int_overflow(N - iim, __LINE__, __FILE__);

            check_mpi(MPI_File_read_at_all(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
        }
    }


    // MPI_File_write_at_all handling count argument larger than INT_MAX
    template <typename T>
    void mpi_file_write_at_all(const size_t N, MPI_Offset offset, MPI_File fh, MPI_Datatype MPI_DT, const int NWRITES, T buffer) 
    {
        int rank, dtsize;
        MPI_Status status;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Type_size(MPI_DT, &dtsize);
        assert(dtsize == sizeof(buffer[0]));

        if (NWRITES == 0) return;

        int SPLIT_ON = check_int_overflow(size_t(ceil(double(N)/double(NWRITES))), __LINE__, __FILE__);
        int count = SPLIT_ON;

        for (uint i=0; i<NWRITES; ++i) {

            const size_t iim = size_t(i) * size_t(SPLIT_ON);

            // Last iteration takes only the leftover
            if (i == NWRITES-1) count = check_int_overflow(N - iim, __LINE__, __FILE__);

            check_mpi(MPI_File_write_at_all(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
        }
    }


    //void compute_markers_statistics(const char* rawdata, const uint M, const uint NB, double* mave, double* mstd, uint* msum);
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
    void readPhenotypeFile(const string &phenFile, const int numberIndividuals);
    void readGroupFile(const string &groupFile);
    template<typename M>
    M    readCSVFile(const string &covariateFile);
    void readCovariateFile(const string &covariateFile);
};

#endif /* data_hpp */
