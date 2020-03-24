#include "data.hpp"
#include <Eigen/Eigen>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iterator>
#include "compression.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <mm_malloc.h>
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_utils.hpp"
#endif

Data::Data()
    : ppBedFd(-1)
    , ppBedMap(nullptr)
    , mappedZ(nullptr, 1, 1)
    , ppbedIndex()
{
}

#ifdef USE_MPI


void Data::print_restart_banner(const string mcmcOut, const uint iteration_restart, 
                                const uint iteration_start) {
    printf("INFO   : %s\n", string(100, '*').c_str());
    printf("INFO   : RESTART DETECTED\n");
    printf("INFO   : restarting from: %s.* files\n", mcmcOut.c_str());
    printf("INFO   : last saved iteration:        %d\n", iteration_restart);
    printf("INFO   : will restart from iteration: %d\n", iteration_start);
    printf("INFO   : %s\n", string(100, '*').c_str());
}


void Data::read_mcmc_output_idx_file(const string mcmcOut, const string ext, const uint length, const uint iteration_restart,
                                     std::vector<int>& markerI)  {
    
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    MPI_File   fh;

    const string fp = mcmcOut + "." + ext + "." + std::to_string(rank);
    check_mpi(MPI_File_open(MPI_COMM_SELF, fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    // 1. get and validate iteration number that we are about to read
    MPI_Offset off = size_t(0);
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at(fh, off, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_restart) {
        printf("Mismatch between expected and read mrk iteration: %d vs %d\n", iteration_restart, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate M (against size of markerI)
    uint M_ = 0;
    off = sizeof(uint);
    check_mpi(MPI_File_read_at(fh, off, &M_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    uint M = markerI.size();
    if (M_ != M) {
        printf("Mismatch between expected and read mrk M: %d vs %d\n", M, M_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the M_ coefficients
    off = sizeof(uint) + sizeof(uint);
    check_mpi(MPI_File_read_at(fh, off, markerI.data(), M_, MPI_INT, &status), __LINE__, __FILE__);


    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}


//EO: .gam files only contain a dump of last saved iteration (no history)
//  : format is: iter, length, vector
void Data::read_mcmc_output_gam_file(const string mcmcOut, const int gamma_length, const uint iteration_restart,
                                     VectorXd& gamma) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const string gamfp = mcmcOut + ".gam." + std::to_string(rank);

    MPI_Status status;
    
    MPI_File gamfh;
    check_mpi(MPI_File_open(MPI_COMM_SELF, gamfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &gamfh), __LINE__, __FILE__);


    // 1. get and validate iteration number that we are about to read
    MPI_Offset gamoff = size_t(0);
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(gamfh, gamoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_restart) {
        printf("Mismatch between expected and read gamma iteration: %d vs %d\n", iteration_restart, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate gamma_length
    uint gamma_length_ = 0;
    gamoff = sizeof(uint);
    check_mpi(MPI_File_read_at_all(gamfh, gamoff, &gamma_length_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (gamma_length_ != gamma_length) {
        printf("Mismatch between expected and read length of gamma vector: %d vs %d\n", gamma_length, gamma_length_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the gamma_length_ coefficients
    gamoff = sizeof(uint) + sizeof(uint);
    check_mpi(MPI_File_read_at_all(gamfh, gamoff, gamma.data(), gamma_length_, MPI_DOUBLE, &status), __LINE__, __FILE__);

    //if (rank%50==0)
    //    printf("rank %d reading back gam: %15.10f %15.10f\n", rank, gamma[0], gamma[gamma_length_-1]);

    check_mpi(MPI_File_close(&gamfh), __LINE__, __FILE__);
}


//EO: .eps files only contain a dump of last saved iteration (no history)
//
void Data::read_mcmc_output_eps_file(const string mcmcOut,  const uint Ntotc, const uint iteration_restart,
                                     VectorXd& epsilon) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const string epsfp = mcmcOut + ".eps." + std::to_string(rank);
    //const string epsfp = mcmcOut + ".eps";

    MPI_Status status;
    
    MPI_File epsfh;
    check_mpi(MPI_File_open(MPI_COMM_SELF, epsfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);


    // 1. get and validate iteration number that we are about to read
    MPI_Offset epsoff = size_t(0);
    uint iteration_   = UINT_MAX;
    check_mpi(MPI_File_read_at_all(epsfh, epsoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_restart) {
        printf("Mismatch between expected and read eps iteration: %d vs %d\n", iteration_restart, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate Ntot_ (against size of epsilon, adjusted for NAs)
    uint Ntot_ = 0;
    epsoff = sizeof(uint);
    check_mpi(MPI_File_read_at_all(epsfh, epsoff, &Ntot_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    uint Ntot = epsilon.size();
    assert(Ntot == Ntotc);
    if (Ntot_ != Ntot) {
        printf("Mismatch between expected and read eps Ntot: %d vs %d\n", Ntot, Ntot_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the Ntot_ coefficients
    epsoff = sizeof(uint) + sizeof(uint);
    check_mpi(MPI_File_read_at_all(epsfh, epsoff, epsilon.data(), Ntot_, MPI_DOUBLE, &status), __LINE__, __FILE__);

    //if (rank%50==0)
    //    printf("rank %d reading back eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot_-1]);

    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
}


//EO: Watch out the saving frequency of the betas (--thin)
//    Each line contains simple the iteration (uint) and the value of mu (double)
void Data::read_mcmc_output_mus_file(const string mcmcOut, const uint  iteration_restart, const int thin, double& mu) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const string musfp = mcmcOut + ".mus." + std::to_string(rank);

    MPI_Status status;

    MPI_File musfh;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, musfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &musfh), __LINE__, __FILE__);

    MPI_Offset musoff = size_t(0);

    // 1. get and validate iteration number that we are about to read
    assert(iteration_restart%thin == 0);
    int n_thinned_saved = iteration_restart / thin;
    musoff = size_t(n_thinned_saved) * (sizeof(uint) + sizeof(double));
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(musfh, musoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_restart) {
        printf("Mismatch between expected and read mus iteration: %d vs %d\n", iteration_restart, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. read the value of mu
    musoff += sizeof(uint);
    check_mpi(MPI_File_read_at(musfh, musoff, &mu, 1, MPI_DOUBLE, &status), __LINE__, __FILE__);
    //printf("reading back mu = %15.10f at iteration %d\n", mu, iteration_);

    check_mpi(MPI_File_close(&musfh), __LINE__, __FILE__);
}




//EO: Watch out the saving frequency of the betas (--thin)
void Data::read_mcmc_output_cpn_file(const string mcmcOut, const uint Mtot, 
                                     const uint  iteration_restart, const int thin,
                                     const int*   MrankS,  const int* MrankL,
                                     VectorXi& components) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const string cpnfp = mcmcOut + ".cpn";

    MPI_Status status;

    MPI_File cpnfh;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);

    // 1. first element of the .bet, .cpn and .acu files is the total number of processed markers
    uint Mtot_ = 0;
    MPI_Offset cpnoff = size_t(0);
    check_mpi(MPI_File_read_at_all(cpnfh, cpnoff, &Mtot_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (Mtot_ != Mtot) {
        printf("Mismatch between expected and read cpn Mtot: %d vs %d\n", Mtot, Mtot_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate iteration number that we are about to read
    assert(iteration_restart%thin == 0);
    int n_thinned_saved = iteration_restart / thin;
    cpnoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot_) * sizeof(int));
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(cpnfh, cpnoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_restart) {
        printf("Mismatch between expected and read cpn iteration: %d vs %d\n", iteration_restart, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the Mtot_ coefficients
    cpnoff = sizeof(uint) + sizeof(uint) 
        + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot_) * sizeof(int))
        + size_t(MrankS[rank]) * sizeof(int);
    check_mpi(MPI_File_read_at_all(cpnfh, cpnoff, components.data(), MrankL[rank], MPI_INTEGER, &status), __LINE__, __FILE__);

    //printf("reading back cpn: %d %d\n", components[0], components[MrankL[rank]-1]);

    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
}


//EO: Watch out the saving frequency of the betas (--thin)
void Data::read_mcmc_output_bet_file(const string mcmcOut, const uint Mtot,
                                     const uint  iteration_restart, const int thin,
                                     const int*   MrankS,  const int* MrankL,
                                     VectorXd& Beta) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const string betfp = mcmcOut + ".bet";

    MPI_Status status;

    MPI_File betfh;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);

    // 1. first element of the .bet, .cpn and .acu files is the total number of processed markers
    uint Mtot_ = 0;
    MPI_Offset betoff = size_t(0);
    check_mpi(MPI_File_read_at_all(betfh, betoff, &Mtot_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (Mtot_ != Mtot) {
        printf("Mismatch between expected and read bet Mtot: %d vs %d\n", Mtot, Mtot_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate iteration number that we are about to read
    assert(iteration_restart%thin == 0);
    int n_thinned_saved = iteration_restart / thin;
    betoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot_) * sizeof(double));
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(betfh, betoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_restart) {
        printf("Mismatch between expected and read bet iteration: %d vs %d\n", iteration_restart, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the Mtot_ coefficients
    betoff = sizeof(uint) + sizeof(uint) 
        + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot_) * sizeof(double))
        + size_t(MrankS[rank]) * sizeof(double);
    check_mpi(MPI_File_read_at_all(betfh, betoff, Beta.data(), MrankL[rank], MPI_DOUBLE, &status), __LINE__, __FILE__);

    //printf("rank %d reading back bet: %15.10f %15.10f\n", rank, Beta[0], Beta[MrankL[rank]-1]);
        

    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
}


//EO: consider moving the csv output file from ASCII to BIN
void Data::read_mcmc_output_csv_file(const string mcmcOut, const uint optSave, const int K,
                                     VectorXd& sigmaG, double& sigmaE, MatrixXd& pi, uint& iteration_restart) {
    
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string        csv = mcmcOut + ".csv";
    std::ifstream file(csv);
    int           it_ = 1E9, rank_ = -1, m0_ = -1, pirows_ = -1, picols_ = -1, nchar = 0, ngrp_ = -1;
    int           lastSavedIt = 0;
    double        mu_, rat_, tmp;
    VectorXd      sigg_(numGroups);
    MatrixXd      pipi(numGroups,K);
    
    if (file.is_open()) {

        std::string str;

        while (std::getline(file, str)) {

            if (str.length() > 0) {

                int nread = sscanf(str.c_str(), "%5d, %4d", &it_, &ngrp_);
                //printf("%5d | %4d with optSave = %d\n", it_, ngrp_, optSave);

                if (it_ % optSave == 0) {

                    lastSavedIt = it_;

                    char cstr[str.length()+1];
                    strcpy(cstr, str.c_str());
                    nread = sscanf(cstr, "%5d, %4d, %n", &it_, &ngrp_, &nchar);
                    string remain_s = str.substr(nchar, str.length() - nchar);
                    char   remain_c[remain_s.length() + 1];
                    strcpy(remain_c, remain_s.c_str());

                    // remove commas
                    for (int i=0; i<remain_s.length(); ++i) {
                        if (remain_c[i] == ',') remain_c[i] = ' ';
                    }

                    char* prc = remain_c;
                    for (int i=0; i<numGroups; i++) {
                        sigmaG[i] = strtod(prc, &prc);
                        //printf("sigmaG[%d] = %20.15f\n", i, sigmaG[i]); 
                    }

                    sigmaE = strtod(prc, &prc);
                    //printf("sigmaE = %20.15f\n", sigmaE);

                    rat_ = strtod(prc, &prc);
                    //printf("rat_ = %20.15f\n", rat_);
                    
                    m0_ = std::strtol(prc, &prc, 10);
                    //printf("m0_ = %d\n", m0_);

                    pirows_ = std::strtol(prc, &prc, 10);
                    //printf("pirows_ = %d\n", pirows_);
                    assert(pirows_ == ngrp_);
                    assert(pi.rows() == pirows_);

                    picols_ = std::strtol(prc, &prc, 10);
                    //printf("picols_ = %d\n", picols_);
                    assert(pi.cols() == picols_);


                    for (int i=0; i<pirows_; i++) {
                        for (int j=0; j<picols_; j++) {
                            pi(i, j) = strtod(prc, &prc);
                        }
                    }
                }
            }
        }
    } else {
        printf("*FATAL*: failed to open csv file %s!\n", csv.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Sanity checks
    assert(lastSavedIt % optSave == 0);
    assert(lastSavedIt >= 0);
    assert(lastSavedIt <= UINT_MAX);

    iteration_restart = lastSavedIt;
}


void Data::sparse_data_correct_NA_OLD(const size_t* N1S, const size_t* N2S, const size_t* NMS, 
                                      size_t*       N1L,       size_t* N2L,       size_t* NML,
                                      uint*         I1,        uint*   I2,        uint*   IM,
                                      const int M) {
    
    for (int ii=0; ii<M; ++ii) {
        
        size_t beg = 0, len = 0;
        
        for (int i=0; i<numNAs; ++i) {

            beg = N1S[ii]; len = N1L[ii];
            if (len > 0) {
                for (size_t iii=beg; iii<beg+len; ++iii) {
                    if (I1[iii] + i == NAsInds[i]) { 
                        N1L[ii] -= 1; 
                        for (size_t k = iii; k<beg+N1L[ii]; k++) I1[k] = I1[k + 1] - 1;                        
                        break;
                    } else {
                        if (I1[iii] + i >= NAsInds[i]) I1[iii] = I1[iii] - 1;
                    }
                }
            }

            beg = N2S[ii]; len = N2L[ii];
            if (len > 0) {
                for (size_t iii=beg; iii<beg+len; ++iii) {
                    if (I2[iii] + i == NAsInds[i]) { 
                        N2L[ii] -= 1;
                        for (size_t k = iii; k<beg+N2L[ii]; k++) I2[k] = I2[k + 1] - 1;
                        break;
                    } else {
                        if (I2[iii] + i >= NAsInds[i]) I2[iii] = I2[iii] - 1;
                    }
                }
            }

            beg = NMS[ii]; len = NML[ii];
            if (len > 0) {
                for (size_t iii=beg; iii<beg+len; ++iii) {
                    if (IM[iii] + i == NAsInds[i]) { 
                        NML[ii] -= 1;
                        for (size_t k = iii; k<beg+NML[ii]; k++) IM[k] = IM[k + 1] - 1;
                        break;
                    } else {
                        if (IM[iii] + i >= NAsInds[i]) IM[iii] = IM[iii] - 1;
                    }
                }
            }
        }
    }
}

//EO: load data from a bed file
// ----------------------------
void Data::load_data_from_bed_file(string bedfp, const uint Ntot, const int M, const int rank, const int start, 
                                   double& dalloc,
                                   size_t* N1S, size_t* N1L, uint*& I1,
                                   size_t* N2S, size_t* N2L, uint*& I2,
                                   size_t* NMS, size_t* NML, uint*& IM) {

    MPI_File   bedfh;
    MPI_Offset offset;

    bedfp += ".bed";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__);

    // Length of a "column" in bytes
    const size_t snpLenByt = (Ntot % 4) ? Ntot / 4 + 1 : Ntot / 4;
    //if (rank==0) printf("INFO   : marker length in bytes (snpLenByt) = %zu bytes.\n", snpLenByt);

    // Alloc memory for raw BED data
    // -----------------------------
    const size_t rawdata_n = size_t(M) * size_t(snpLenByt) * sizeof(char);
    char* rawdata = (char*)_mm_malloc(rawdata_n, 64);  check_malloc(rawdata, __LINE__, __FILE__);
    dalloc += rawdata_n / 1E9;
    //printf("rank %d allocation %zu bytes (%.3f GB) for the raw data.\n", rank, rawdata_n, double(rawdata_n/1E9));

    // Compute the offset of the section to read from the BED file
    // -----------------------------------------------------------
    //offset = size_t(3) + size_t(MrankS[rank]) * size_t(snpLenByt) * sizeof(char);
    offset = size_t(3) + size_t(start) * size_t(snpLenByt) * sizeof(char);

    // Read the BED file
    // -----------------
    MPI_Barrier(MPI_COMM_WORLD);
    //const auto st1 = std::chrono::high_resolution_clock::now();

    // Gather the sizes to determine common number of reads
    size_t rawdata_n_max = 0;
    check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);
    size_t bytes = 0;
    mpi_file_read_at_all <char*> (rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata, bytes);

    MPI_Barrier(MPI_COMM_WORLD);
    //const auto et1 = std::chrono::high_resolution_clock::now();
    //const auto dt1 = et1 - st1;
    //const auto du1 = std::chrono::duration_cast<std::chrono::milliseconds>(dt1).count();
    //if (rank == 0)  std::cout << "INFO   : time to read the BED file: " << du1 / double(1000.0) << " seconds." << std::endl;

    // Close BED file
    check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);
   

    size_t N1 = 0, N2 = 0, NM = 0;
    sparse_data_get_sizes_from_raw(rawdata, M, snpLenByt, N1, N2, NM);
    //printf("read from bed: N1 = %lu, N2 = %lu, NM = %lu\n", N1, N2, NM);

    // Alloc and build sparse structure
    I1 = (uint*)_mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
    I2 = (uint*)_mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
    IM = (uint*)_mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);
    dalloc += (N1 + N2 + NM) * sizeof(size_t) / 1E9;
    
    sparse_data_fill_indices(rawdata, M, snpLenByt,
                             N1S, N1L, I1,
                             N2S, N2L, I2,
                             NMS, NML, IM);

    _mm_free(rawdata);        
}


void Data::load_data_from_sparse_files(const int rank, const int nranks, const int M,
                                       const int* MrankS, const int* MrankL,
                                       const string sparseOut,
                                       double& dalloc,
                                       size_t* N1S,   size_t* N1L, uint*& I1,
                                       size_t* N2S,   size_t* N2L, uint*& I2,
                                       size_t* NMS,   size_t* NML, uint*& IM,
                       size_t& totalBytes) {

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int  processor_name_len;
    MPI_Get_processor_name(processor_name, &processor_name_len);

    // Get sizes to alloc for the task
    size_t N1 = get_number_of_elements_from_sparse_files(sparseOut, "1", MrankS, MrankL, N1S, N1L);
    size_t N2 = get_number_of_elements_from_sparse_files(sparseOut, "2", MrankS, MrankL, N2S, N2L);
    size_t NM = get_number_of_elements_from_sparse_files(sparseOut, "m", MrankS, MrankL, NMS, NML);

    size_t N1max = 0, N2max = 0, NMmax = 0;
    check_mpi(MPI_Allreduce(&N1, &N1max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&N2, &N2max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NM, &NMmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    size_t N1tot = 0, N2tot = 0, NMtot = 0;
    check_mpi(MPI_Allreduce(&N1, &N1tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&N2, &N2tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NM, &NMtot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

    totalBytes =  (N1tot + N2tot + NMtot) * sizeof(uint);

    if (rank % 10 == 0) {
        printf("INFO   : rank %3d/%3d  N1max = %15lu, N2max = %15lu, NMmax = %15lu\n", rank, nranks, N1max, N2max, NMmax);
        printf("INFO   : rank %3d/%3d  N1tot = %15lu, N2tot = %15lu, NMtot = %15lu\n", rank, nranks, N1tot, N2tot, NMtot);
        printf("INFO   : RAM for task %3d/%3d on node %s: %7.3f GB\n", rank, nranks, processor_name, (N1 + N2 + NM) * sizeof(uint) / 1E9);
    }
    if (rank == 0) 
        printf("INFO   : Total RAM for storing sparse indices %.3f GB\n", double(totalBytes) * 1E-9);
    fflush(stdout);

    I1 = (uint*)_mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
    I2 = (uint*)_mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
    IM = (uint*)_mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);
    dalloc += double(N1 + N2 + NM) * 1E-9;

    //EO: base the number of read calls on a max buffer size of 2 GiB
    //    rather than count be lower that MAX_INT/2
    //----------------------------------------------------------------------
    int NREADS1 = int(ceil(double(N1 * sizeof(uint)) / double(2147483648)));
    int NREADS2 = int(ceil(double(N2 * sizeof(uint)) / double(2147483648)));
    int NREADSM = int(ceil(double(NM * sizeof(uint)) / double(2147483648)));

    /*
    int NREADS1 = check_int_overflow(size_t(ceil(double(N1max)/double(INT_MAX/2))), __LINE__, __FILE__);
    int NREADS2 = check_int_overflow(size_t(ceil(double(N2max)/double(INT_MAX/2))), __LINE__, __FILE__);
    int NREADSM = check_int_overflow(size_t(ceil(double(NMmax)/double(INT_MAX/2))), __LINE__, __FILE__);
    */
    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("INFO   : rank %d, number of calls to read the sparse files: NREADS1 = %d, NREADS2 = %d, NREADSM = %d\n", rank, NREADS1, NREADS2, NREADSM);
    //fflush(stdout);
   
    //EO: to keep the read_at_all in, we need to get the max nreads
    int MAX_NREADS1 = 0, MAX_NREADS2 = 0, MAX_NREADSM = 0;
    check_mpi(MPI_Allreduce(&NREADS1, &MAX_NREADS1, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NREADS2, &MAX_NREADS2, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NREADSM, &MAX_NREADSM, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    if (rank % 10 ==0) {
        printf("INFO   : rank %d, numbers of calls to read the sparse files: NREADS1,2,M = %3d, %3d, %3d vs MAX_NREADS1,2,M = %3d, %3d, %3d\n",
               rank, NREADS1, NREADS2, NREADSM, MAX_NREADS1, MAX_NREADS2, MAX_NREADSM);
        fflush(stdout);
    }
    //read_sparse_data_file(sparseOut + ".si1", N1, N1S[0], NREADS1, I1);
    //read_sparse_data_file(sparseOut + ".si2", N2, N2S[0], NREADS2, I2);
    //read_sparse_data_file(sparseOut + ".sim", NM, NMS[0], NREADSM, IM);    
    read_sparse_data_file(sparseOut + ".si1", N1, N1S[0], MAX_NREADS1, I1);
    read_sparse_data_file(sparseOut + ".si2", N2, N2S[0], MAX_NREADS2, I2);
    read_sparse_data_file(sparseOut + ".sim", NM, NMS[0], MAX_NREADSM, IM);    

    // Make starts relative to start of block in each task
    const size_t n1soff = N1S[0];  for (int i=0; i<M; ++i) { N1S[i] -= n1soff; }
    const size_t n2soff = N2S[0];  for (int i=0; i<M; ++i) { N2S[i] -= n2soff; }
    const size_t nmsoff = NMS[0];  for (int i=0; i<M; ++i) { NMS[i] -= nmsoff; }
}


// EO: Apply corrections to the sparse structures (1,2,m)
//     Watch out that NAs have to be considered globally accross the structures
// ---------------------------------------------------------------------------
void Data::sparse_data_correct_for_missing_phenotype(const size_t* NS, size_t* NL, uint* I, const int M) {

    // Alloc one tmp vector large enough
    uint max = 0;
    for (int i=0; i<M; ++i)
        if (NL[i] > max) max = NL[i];

    uint* tmp = (uint*)_mm_malloc(max * sizeof(uint), 64);  check_malloc(tmp, __LINE__, __FILE__);

    for (int i=0; i<M; ++i) {

        //cout << "dealing with marker " << i << " out of " << M << endl;

        const size_t beg = NS[i], len = NL[i];
        size_t k   = 0;
        uint   nas = 0;

        if (len > 0) {

            // Make a tmp copy of the original data
            for (size_t iii=beg; iii<beg+len; ++iii) tmp[iii-beg] = I[iii];
            
            for (size_t iii=beg; iii<beg+len; ++iii) {
                bool isna   = false;
                uint allnas = 0;
                for (int ii=0; ii<numNAs; ++ii) {
                    if (NAsInds[ii] > tmp[iii-beg]) break;
                    if (NAsInds[ii] <= tmp[iii-beg]) allnas += 1;
                    if (tmp[iii-beg] == NAsInds[ii]) { // NA found
                        isna = true;
                        nas += 1;
                        break;
                    }
                }
                if (isna) continue;
                I[beg+k] = tmp[iii-beg]  - allnas;
                k += 1;
            }
        }
        NL[i] -= nas;
    }
    _mm_free(tmp);
}


void Data::sparse_data_get_sizes_from_raw(const char* rawdata, const uint NC, const uint NB, size_t& N1, size_t& N2, size_t& NM) {

    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(char), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    N1 = 0;
    N2 = 0;
    NM = 0;
    size_t N0 = 0;

    for (uint i=0; i<NC; ++i) {

        uint c0 = 0, c1 = 0, c2 = 0, cm = 0;

        char* locraw = (char*)&rawdata[size_t(i)*size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        for (int ii=0; ii<numInds; ++ii) {
            if      (tmpi[ii] <  0) { cm += 1; NM += 1; }
            else if (tmpi[ii] == 0) { c0 += 1; N0 += 1; }
            else if (tmpi[ii] == 1) { c1 += 1; N1 += 1; }
            else if (tmpi[ii] == 2) { c2 += 1; N2 += 1; }
        }
        
        assert(cm+c0+c1+c2 == numInds);
    }

    _mm_free(tmpi);
}


// Screen the raw BED data and count and register number of ones, twos and missing information
// N*S: store the "S"tart of each marker representation 
// N*L: store the "L"ength of each marker representation
// I* : store the indices of the elements
// -------------------------------------------------------------------------------------------
void Data::sparse_data_fill_indices(const char* rawdata, const uint NC, const uint NB,
                                    size_t* N1S, size_t* N1L, uint* I1,
                                    size_t* N2S, size_t* N2L, uint* I2,
                                    size_t* NMS, size_t* NML, uint* IM) {
    
    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(int8_t), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    size_t i1 = 0, i2 = 0, im = 0;
    size_t N1 = 0, N2 = 0, NM = 0;

    for (int i=0; i<NC; ++i) {

        char* locraw = (char*)&rawdata[size_t(i)*size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        size_t n0 = 0, n1 = 0, n2 = 0, nm = 0;
        
        for (uint ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                IM[im] = ii;
                im += 1;
                nm += -tmpi[ii];
            } else {
                if        (tmpi[ii] == 0) {
                    n0 += 1;
                } else if (tmpi[ii] == 1) {
                    I1[i1] = ii;
                    i1 += 1;
                    n1 += 1;
                } else if (tmpi[ii] == 2) {
                    I2[i2] = ii;
                    i2 += 1;
                    n2 += 1;
                }
            }
        }

        assert(nm + n0 + n1 + n2 == numInds);

        N1S[i] = N1;  N1L[i] = n1;  N1 += n1;
        N2S[i] = N2;  N2L[i] = n2;  N2 += n2;
        NMS[i] = NM;  NML[i] = nm;  NM += nm;
    }

    _mm_free(tmpi);
}

size_t Data::get_number_of_elements_from_sparse_files(const std::string basename, const std::string id, const int* MrankS, const int* MrankL,
                                                      size_t* S, size_t* L) {

    MPI_Status status;
    MPI_File   ssfh, slfh;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Number of markers in block handled by task
    const uint M = MrankL[rank];

    const std::string sl = basename + ".sl" + id;
    const std::string ss = basename + ".ss" + id;

    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ssfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);

    // Compute the lengths of ones and twos vectors for all markers in the block
    MPI_Offset offset =  MrankS[rank] * sizeof(size_t);
    check_mpi(MPI_File_read_at_all(ssfh, offset, S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slfh, offset, L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);

    // Absolute offsets in 0s, 1s, and 2s
    const size_t nsoff = S[0];

    size_t N = S[M-1] + L[M-1] - nsoff;

    // Close bed and sparse files
    check_mpi(MPI_File_close(&ssfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);

    return N;
}

/*
void Data::sparse_data_get_sizes_from_sparse(size_t* N1S, size_t* N1L,
                                             size_t* N2S, size_t* N2L,
                                             size_t* NMS, size_t* NML,
                                             const int* MrankS, const int* MrankL, const int rank,
                                             const std::string sparseOut,
                                             size_t& N1, size_t& N2, size_t& NM) {
    
    MPI_Offset offset;
    MPI_Status status;

    // Number of markers in block handled by task
    const uint M = MrankL[rank];

    // Read sparse data files
    // Each task is in charge of M markers starting from MrankS[rank]
    // So first we read si1 to get where to read in 
    MPI_File ss1fh, sl1fh, si1fh;
    MPI_File ss2fh, sl2fh, si2fh;
    MPI_File ssmfh, slmfh, simfh;

    // Get bed file directory and basename
    if (rank == 0) printf("INFO   : will read from sparse files with basename: %s\n", sparseOut.c_str());

    const std::string sl1 = sparseOut + ".sl1";
    const std::string ss1 = sparseOut + ".ss1";
    const std::string sl2 = sparseOut + ".sl2";
    const std::string ss2 = sparseOut + ".ss2";
    const std::string slm = sparseOut + ".slm";
    const std::string ssm = sparseOut + ".ssm";

    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, slm.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ssm.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ssmfh), __LINE__, __FILE__);

    // Compute the lengths of ones and twos vectors for all markers in the block
    offset =  MrankS[rank] * sizeof(size_t);
    //printf("rank %4d: offset = %20lu Bytes MrankS = %lu, M = %lu\n", rank, offset, MrankS[rank], M);
    check_mpi(MPI_File_read_at_all(sl1fh, offset, N1L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ss1fh, offset, N1S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(sl2fh, offset, N2L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ss2fh, offset, N2S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slmfh, offset, NML, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(ssmfh, offset, NMS, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    
    // Sanity checks
    // -------------
    for (int i=0; i<M-1; i++) {
        if (N1S[i] > N1S[i+1]) {
            printf("rank %4d: Cannot be N1S[%lu] > N1S[%lu] (%20lu > %20lu)\n", rank, i, i+1, N1S[i], N1S[i+1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (N2S[i] > N2S[i+1]) {
            printf("rank %4d: Cannot be N2S[%lu] > N2S[%lu] (%20lu > %20lu)\n", rank, i, i+1, N2S[i], N2S[i+1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (NMS[i] > NMS[i+1]) {
            printf("rank %4d: Cannot be NMS[%lu] > NMS[%lu] (%20lu > %20lu)\n", rank, i, i+1, NMS[i], NMS[i+1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }


    // Absolute offsets in 0s, 1s, and 2s
    const size_t n1soff = N1S[0];
    const size_t n2soff = N2S[0];
    const size_t nmsoff = NMS[0];

    N1 = N1S[M-1] + N1L[M-1] - n1soff;
    N2 = N2S[M-1] + N2L[M-1] - n2soff;
    NM = NMS[M-1] + NML[M-1] - nmsoff;

    // Close bed and sparse files
    check_mpi(MPI_File_close(&sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ssmfh), __LINE__, __FILE__);

    MPI_Barrier(MPI_COMM_WORLD);
}
*/


void Data::read_sparse_data_file(const std::string filename, const size_t N, const size_t OFF, const int NREADS, uint* out) {

    MPI_Offset offset;
    MPI_Status status;
    MPI_File   fh;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    offset = OFF * sizeof(uint);
    size_t bytes = 0;

    mpi_file_read_at_all<uint*>(N, offset, fh, MPI_UNSIGNED, NREADS, out, bytes);

    //EO: above call not collective anymore...
    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}


//void Data::get_normalized_marker_data(const char* rawdata, const uint NB)
void Data::get_normalized_marker_data(const char* rawdata, const uint NB, const uint marker, double* Cx, const double mean, const double std_) {

    assert(numInds<=NB*4);

    // Pointer to column in block of raw data
    char* locraw = (char*)&rawdata[size_t(marker) * size_t(NB)];
    
    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(int8_t), 64); check_malloc(tmpi, __LINE__, __FILE__);

    for (int ii=0; ii<NB; ++ii) {
        for (int iii=0; iii<4; ++iii) {
            tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
        }
    }
       
    for (int ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] == 1) {
            tmpi[ii] = -1;
        } else {
            tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
        }
    }
    
    for (size_t ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] < 0) {
            Cx[ii]  = 0.0;
        } else {
            Cx[ii] = (double(tmpi[ii]) - mean) * std_;
        }
    }

    _mm_free(tmpi);
}


//void Data::get_normalized_marker_data(const char* rawdata, const uint NB)
void Data::get_normalized_marker_data(const char* rawdata, const uint NB, const uint marker, double* Cx) {

    assert(numInds<=NB*4);

    // Pointer to column in block of raw data
    char* locraw = (char*)&rawdata[size_t(marker) * size_t(NB)];
    
    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(int8_t), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    for (int ii=0; ii<NB; ++ii) {
        for (int iii=0; iii<4; ++iii) {
            tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
        }
    }
       
    for (int ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] == 1) {
            tmpi[ii] = -1;
        } else {
            tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
        }
    }

    int sum = 0, nmiss = 0;
    //#pragma omp simd reduction(+:sum) reduction(+:nmiss)
    for (int ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] < 0) {
            nmiss += tmpi[ii];
        } else {
            sum   += tmpi[ii];
        }
    }

    double mean = double(sum) / double(numInds + nmiss); //EO: nmiss is neg
    //if (marker < 3)
    //    printf("marker %d  mean = %20.15f\n", marker, mean);

    //printf("rank %d snpInd %2d: sum = %6d, N = %6d, nmiss = %6d, mean = %20.15f\n",
    //       rank, rank*NC+i, sum, numKeptInds, nmiss, mean);
    
    for (size_t ii=0; ii<numInds; ++ii) {
        if (tmpi[ii] < 0) {
            Cx[ii] = 0.0d;
        } else {
            Cx[ii] = double(tmpi[ii]) - mean;
        }
    }

    // Normalize the data
    double sqn  = 0.0d;
    for (size_t ii=0; ii<numInds; ++ii) {
        sqn += Cx[ii] * Cx[ii];
    }
    if (marker < 1)
        printf("marker %d  sqn = %20.15f\n", marker, sqn);

    double std_ = sqrt(double(numInds - 1) / sqn);
    if (marker < 1)
        printf("marker %d  std_ = %20.15f\n", marker, std_);

    for (size_t ii=0; ii<numInds; ++ii)
        Cx[ii] *= std_;

    _mm_free(tmpi);
}

// Read raw data loaded in memory to preprocess them (center, scale and cast to double)
void Data::preprocess_data(const char* rawdata, const uint NC, const uint NB, double* ppdata, const int rank) {

    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(int8_t), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    for (int i=0; i<NC; ++i) {

        char* locraw = (char*)&rawdata[size_t(i) * size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        int sum = 0, nmiss = 0;
        //#pragma omp simd reduction(+:sum) reduction(+:nmiss)
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                nmiss += tmpi[ii];
            } else {
                sum   += tmpi[ii];
            }
        }

        double mean = double(sum) / double(numInds + nmiss); //EO: nmiss is neg
        //printf("rank %d snpInd %2d: sum = %6d, N = %6d, nmiss = %6d, mean = %20.15f\n",
        //       rank, rank*NC+i, sum, numKeptInds, nmiss, mean);

        size_t ppdata_i = size_t(i) * size_t(numInds);
        double *locpp = (double*)&ppdata[ppdata_i];

        for (size_t ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                locpp[ii] = 0.0d;
            } else {
                locpp[ii] = double(tmpi[ii]) - mean;
            }
        }

        double sqn  = 0.0d;
        for (size_t ii=0; ii<numInds; ++ii) {
            sqn += locpp[ii] * locpp[ii];
        }

        double std_ = sqrt(double(numInds - 1) / sqn);

        for (size_t ii=0; ii<numInds; ++ii) {
            locpp[ii] *= std_;
        }
    }

    _mm_free(tmpi);
}

#endif



void Data::preprocessBedFile(const string &bedFile, const string &preprocessedBedFile, const string &preprocessedBedIndexFile, bool compress)
{
    cout << "Preprocessing bed file: " << bedFile << ", Compress data = " << (compress ? "yes" : "no") << endl;
    if (numSnps == 0)
        throw ("Error: No SNP is retained for analysis.");
    if (numInds == 0)
        throw ("Error: No individual is retained for analysis.");

    ifstream BIT(bedFile.c_str(), ios::binary);
    if (!BIT)
        throw ("Error: can not open the file [" + bedFile + "] to read.");

    ofstream ppBedOutput(preprocessedBedFile.c_str(), ios::binary);
    if (!ppBedOutput)
        throw("Error: Unable to open the preprocessed bed file [" + preprocessedBedFile + "] for writing.");
    ofstream ppBedIndexOutput(preprocessedBedIndexFile.c_str(), ios::binary);
    if (!ppBedIndexOutput)
        throw("Error: Unable to open the preprocessed bed index file [" + preprocessedBedIndexFile + "] for writing.");

    cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    char header[3];
    BIT.read(header, 3);
    if (!BIT || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01)
        throw ("Error: Incorrect first three bytes of bed file: " + bedFile);

    // How much space do we need to compress the data (if requested)
    const auto maxCompressedOutputSize = compress ? maxCompressedDataSize(numInds) : 0;
    unsigned char *compressedBuffer = nullptr;
    unsigned long pos = 0;
    if (compress)
        compressedBuffer = new unsigned char[maxCompressedOutputSize];

    // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 10: hetezygote; 01: missing
    for (unsigned int j = 0, snp = 0; j < numSnps; j++) {
        SnpInfo *snpInfo = snpInfoVec[j];
        double sum = 0.0;
        unsigned int nmiss = 0;

        // Create some scratch space to preprocess the raw data
        VectorXd snpData(numInds);

        // Make a note of which individuals have a missing genotype
        vector<long> missingIndices;

        const unsigned int size = (numInds + 3) >> 2;
        if (!snpInfo->included) {
            BIT.ignore(size);
            continue;
        }

        for (unsigned int i = 0, ind = 0; i < numInds;) {
            char ch;
            BIT.read(&ch, 1);
            if (!BIT)
                throw ("Error: problem with the BED file ... has the FAM/BIM file been changed?");

            bitset<8> b = ch;
            unsigned int k = 0;

            while (k < 7 && i < numInds) {
                if (!indInfoVec[i]->kept) {
                    k += 2;
                } else {
                    const unsigned int allele1 = (!b[k++]);
                    const unsigned int allele2 = (!b[k++]);

                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        // Don't store a marker value like this as it requires floating point comparisons later
                        // which are not done properly. Instead, store the index of the individual in a vector and simply
                        // iterate over the collected indices. Also means iterating over far fewer elements which may
                        // make a noticeable difference as this scales up.
                        missingIndices.push_back(ind++);
                        ++nmiss;
                    } else {
                        const auto value = allele1 + allele2;
                        snpData[ind++] = value;
                        sum += value;
                    }
                }
                i++;
            }
        }

        // Fill missing values with the mean
        const double mean = sum / double(numInds - nmiss);
        if (j % 100 == 0) {
            printf("MARKER %6d mean = %12.7f computed on %6.0f with %6d elements (%d - %d)\n",
                   j, mean, sum, numInds-nmiss, numInds, nmiss);
            fflush(stdout);
        }
        if (nmiss) {
            for (const auto index : missingIndices)
                snpData[index] = mean;
        }

        // Standardize genotypes
        snpData.array() -= snpData.mean();
        const auto sqn = snpData.squaredNorm();
        const auto sigma = 1.0 / (sqrt(sqn / (double(numInds - 1))));
        snpData.array() *= sigma;

        // Write out the preprocessed data
        if (!compress) {
            ppBedOutput.write(reinterpret_cast<char *>(&snpData[0]), numInds * sizeof(double));
        } else {
            const unsigned long compressedSize = compressData(snpData, compressedBuffer, maxCompressedOutputSize);
            ppBedOutput.write(reinterpret_cast<char *>(compressedBuffer), long(compressedSize));

            // Calculate the index data for this column
            ppBedIndexOutput.write(reinterpret_cast<char *>(&pos), sizeof(unsigned long));
            ppBedIndexOutput.write(reinterpret_cast<const char *>(&compressedSize), sizeof(unsigned long));
            pos += compressedSize;
        }

        // Compute allele frequency and any other required data and write out to file
        //snpInfo->af = 0.5f * float(mean);
        //snp2pq[snp] = 2.0f * snpInfo->af * (1.0f - snpInfo->af);

        if (++snp == numSnps)
            break;
    }

    if (compress)
        delete[] compressedBuffer;

    BIT.clear();
    BIT.close();

    cout << "Genotype data for " << numInds << " individuals and " << numSnps << " SNPs are included from [" + bedFile + "]." << endl;
}

void Data::mapPreprocessBedFile(const string &preprocessedBedFile)
{
    // Calculate the expected file sizes - cast to size_t so that we don't overflow the unsigned int's
    // that we would otherwise get as intermediate variables!
    const size_t ppBedSize = size_t(numInds) * size_t(numSnps) * sizeof(double);

    // Open and mmap the preprocessed bed file
    ppBedFd = open(preprocessedBedFile.c_str(), O_RDONLY);
    if (ppBedFd == -1)
        throw("Error: Failed to open preprocessed bed file [" + preprocessedBedFile + "]");

    ppBedMap = reinterpret_cast<double *>(mmap(nullptr, ppBedSize, PROT_READ, MAP_SHARED, ppBedFd, 0));
    if (ppBedMap == MAP_FAILED)
        throw("Error: Failed to mmap preprocessed bed file");

    // Now that the raw data is available, wrap it into the mapped Eigen types using the
    // placement new operator.
    // See https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
    new (&mappedZ) Map<MatrixXd>(ppBedMap, numInds, numSnps);
}

void Data::unmapPreprocessedBedFile()
{
    // Unmap the data from the Eigen accessors
    new (&mappedZ) Map<MatrixXd>(nullptr, 1, 1);

    const auto ppBedSize = numInds * numSnps * sizeof(double);
    munmap(ppBedMap, ppBedSize);
    close(ppBedFd);
}

void Data::mapCompressedPreprocessBedFile(const string &preprocessedBedFile,
                                          const string &indexFile)
{
    // Load the index to the compressed preprocessed bed file
    ppbedIndex.resize(numSnps);
    ifstream indexStream(indexFile, std::ifstream::binary);
    if (!indexStream)
        throw("Error: Failed to open compressed preprocessed bed file index");
    indexStream.read(reinterpret_cast<char *>(ppbedIndex.data()),
                     numSnps * 2 * sizeof(long));

    // Calculate the expected file sizes - cast to size_t so that we don't overflow the unsigned int's
    // that we would otherwise get as intermediate variables!
    const size_t ppBedSize = size_t(ppbedIndex.back().pos + ppbedIndex.back().size);

    // Open and mmap the preprocessed bed file
    ppBedFd = open(preprocessedBedFile.c_str(), O_RDONLY);
    if (ppBedFd == -1)
        throw("Error: Failed to open preprocessed bed file [" + preprocessedBedFile + "]");

    ppBedMap = reinterpret_cast<double *>(mmap(nullptr, ppBedSize, PROT_READ, MAP_SHARED, ppBedFd, 0));
    if (ppBedMap == MAP_FAILED)
        throw("Error: Failed to mmap preprocessed bed file");
}

void Data::unmapCompressedPreprocessedBedFile()
{
    const size_t ppBedSize = size_t(ppbedIndex.back().pos + ppbedIndex.back().size);
    munmap(ppBedMap, ppBedSize);
    close(ppBedFd);
    ppbedIndex.clear();
}


// EO
// Read marker blocks definition file
// Line format:  "%d %d\n"; Nothing else is accepted.
// Gaps are allowed; Overlaps are forbiden
// --------------------------------------------------
void Data::readMarkerBlocksFile(const string &markerBlocksFile) {
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream in(markerBlocksFile.c_str());
    if (!in) throw ("Error: can not open the file [" + markerBlocksFile + "] to read.");
    
    blocksStarts.clear();
    blocksEnds.clear();
    std::string str;
    
    while (std::getline(in, str)) {
        std::vector<std::string> results;
        boost::split(results, str, [](char c){return c == ' ';});
        if (results.size() != 2) {
            printf("Error: line with wrong format: >>%s<<\n", str.c_str());
            printf("       => expected format \"%%d %%d\": two integers separated with a single space, with no leading or trailing space.\n");
            exit(1);
        }
        blocksStarts.push_back(stoi(results[0]));
        blocksEnds.push_back(stoi(results[1]));        
    }
    in.close();

    numBlocks = (unsigned) blocksStarts.size();
    //cout << "Found definitions for " << nbs << " marker blocks." << endl;

    // Neither gaps or overlaps are accepted
    // -------------------------------------
    for (int i=0; i<numBlocks; ++i) {

        if (blocksStarts[i] > blocksEnds[i]) {
            if (rank == 0) {
                printf("FATAL  : block starts beyond end [%d, %d].\n", blocksStarts[i], blocksEnds[i]);
                printf("         => you must correct your marker blocks definition file %s\n", markerBlocksFile.c_str());
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int j=i+1;
        if (j < numBlocks && blocksStarts[j] != blocksEnds[i] + 1) {
            if (rank == 0) {
                printf("FATAL  : block %d ([%d, %d]) and block %d ([%d, %d]) are not contiguous!\n", i, blocksStarts[i], blocksEnds[i], j, blocksStarts[j], blocksEnds[j]);
                printf("         => you must correct your marker blocks definition file %s\n", markerBlocksFile.c_str());
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}


void Data::readFamFile(const string &famFile){
    // ignore phenotype column
    ifstream in(famFile.c_str());
    if (!in) throw ("Error: can not open the file [" + famFile + "] to read.");
#ifndef USE_MPI
    cout << "Reading PLINK FAM file from [" + famFile + "]." << endl;
#endif
    indInfoVec.clear();
    indInfoMap.clear();
    string fid, pid, dad, mom, sex, phen;
    unsigned idx = 0;
    while (in >> fid >> pid >> dad >> mom >> sex >> phen) {
        IndInfo *ind = new IndInfo(idx++, fid, pid, dad, mom, atoi(sex.c_str()));
        indInfoVec.push_back(ind);
        if (indInfoMap.insert(pair<string, IndInfo*>(ind->catID, ind)).second == false) {
            throw ("Error: Duplicate individual ID found: \"" + fid + "\t" + pid + "\".");
        }
    }
    in.close();
    numInds = (unsigned) indInfoVec.size();

#ifndef USE_MPI
    cout << numInds << " individuals to be included from [" + famFile + "]." << endl;
#endif
}

void Data::readBimFile(const string &bimFile) {
    // Read bim file: recombination rate is defined between SNP i and SNP i-1
    ifstream in(bimFile.c_str());
    if (!in) throw ("Error: can not open the file [" + bimFile + "] to read.");
#ifndef USE_MPI
    cout << "Reading PLINK BIM file from [" + bimFile + "]." << endl;
#endif
    snpInfoVec.clear();
    snpInfoMap.clear();
    string id, allele1, allele2;
    unsigned chr, physPos;
    float genPos;
    unsigned idx = 0;
    while (in >> chr >> id >> genPos >> physPos >> allele1 >> allele2) {
        SnpInfo *snp = new SnpInfo(idx++, id, allele1, allele2, chr, genPos, physPos);
        snpInfoVec.push_back(snp);
        if (snpInfoMap.insert(pair<string, SnpInfo*>(id, snp)).second == false) {
            throw ("Error: Duplicate SNP ID found: \"" + id + "\".");
        }
    }
    in.close();
    numSnps = (unsigned) snpInfoVec.size();
#ifndef USE_MPI
    cout << numSnps << " SNPs to be included from [" + bimFile + "]." << endl;
#endif
}


void Data::readBedFile_noMPI(const string &bedFile) {

    unsigned i = 0, j = 0, k = 0;

    cout << "readBedFile_noMPI will read numSnps = " << numSnps << endl;

    if (numSnps == 0) throw ("Error: No SNP is retained for analysis.");
    if (numInds == 0) throw ("Error: No individual is retained for analysis.");
    printf("numInds = %d, numNAs = %d, numSnps = %d\n", numInds, numNAs, numSnps);

    Z.resize(numInds-numNAs, numSnps);
    ZPZdiag.resize(numSnps);
    snp2pq.resize(numSnps);

    // Read bed file
    char ch[1];
    bitset<8> b;
    unsigned allele1=0, allele2=0;
    ifstream BIT(bedFile.c_str(), ios::binary);
    if (!BIT) throw ("Error: can not open the file [" + bedFile + "] to read.");
    //#ifndef USE_MPI
    cout << "Reading PLINK BED file from [" + bedFile + "] in SNP-major format ..." << endl;
    //#endif

    for (i = 0; i < 3; i++) BIT.read(ch, 1); // skip the first three bytes
    SnpInfo *snpInfo = NULL;
    unsigned snp = 0, ind = 0;
    unsigned nmiss = 0;
    //float mean = 0.0;
    double mean = 0.0;

    // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 10: hetezygote; 01: missing
    // --------------------------------------------------------------------------------------------------
    for (j = 0, snp = 0; j < numSnps; j++) {
        snpInfo = snpInfoVec[j];
        mean = 0.0;
        nmiss = 0;
        if (!snpInfo->included) {
            for (i = 0; i < numInds; i += 4) BIT.read(ch, 1);
            continue;
        }
        
        uint nona = 0;

        for (i = 0, ind = 0; i < numInds;) {
            BIT.read(ch, 1);
            if (!BIT) throw ("Error: problem with the BED file ... has the FAM/BIM file been changed?");
            b = ch[0];
            k = 0;
            while (k < 7 && i < numInds) {
                if (!indInfoVec[i]->kept) {
                    k += 2;
                } else {
                    allele1 = (!b[k++]);
                    allele2 = (!b[k++]);
                    if (allele1 == 0 && allele2 == 1) {  // missing genotype
                        Z(nona, snp) = -9;
                        ++nmiss;
                    } else {
                        mean += Z(nona, snp) = allele1 + allele2;
                    }
                    nona+=1;
                }
                i += 1;
            }
        }

        //if (j==0) printf("numInds vs nona; numNAs: %d vs %d vs %d\n", numInds, nona, numNAs);

        //EO: account for missing phenotypes
        //----------------------------------
        mean /= double(nona-nmiss);
        
        if (nmiss) {
            for (i=0; i<nona; ++i) {
                if (Z(i,snp) == -9) Z(i,snp) = mean;
            }
        }

        // compute allele frequency
        //snpInfo->af = 0.5f*mean;
        //snp2pq[snp] = 2.0f*snpInfo->af*(1.0f-snpInfo->af);

        // standardize genotypes
        Z.col(j).array() -= mean;

        //float sqn = Z.col(j).squaredNorm();
        //float std_ = 1.f / (sqrt(sqn / float(numInds)));
        double sqn = Z.col(j).squaredNorm();
        /*
        double sqn_ = 0.0;
        for (i=0; i<nona; ++i) {
            sqn_ += Z(i,snp) * Z(i,snp);
        }
        printf("sqn vs sqn_  %20.15f vs %20.15f\n", sqn, sqn);
        */
        //double std_ = 1.0 / (sqrt(sqn / double(numInds - 1)));
        double std_ = sqrt(double(nona - 1) / sqn);

        //if (j < 1)
        //printf("OFF: marker %d has mean = %20.15f sqn = %20.15f  std_ = %20.15f\n", j, mean, sqn, std_);

        Z.col(j).array() *= std_;

        ZPZdiag[j] = sqn;

        if (++snp == numSnps) break;
    }
    BIT.clear();
    BIT.close();

    for (i=0; i<numSnps; ++i) {
        //Z.col(i).array() -= Z.col(i).mean(); //EO alread done (so now mean is zero)
        ZPZdiag[i] = Z.col(i).squaredNorm();
    }

#ifndef USE_MPI
    cout << "Genotype data for " << numInds << " individuals and " << numSnps << " SNPs are included from [" + bedFile + "]." << endl;
#endif
}


void Data::center_and_scale(double* __restrict__ vec, int* __restrict__ mask, const uint N, const uint nas) {

    // Compute mean
    double mean = 0.0;
    uint nonas = N - nas;

    for (int i=0; i<N; ++i)
        mean += vec[i] * mask[i];

    mean /= nonas;
    //cout << "mean = " << mean << endl;

    // Center
    for (int i=0; i<N; ++i)
        vec[i] -= mean;

    // Compute scale
    double sqn = 0.0;
    for (int i=0; i<N; ++i) {
        if (mask[i] == 1) {
            sqn += vec[i] * vec[i];
        }
    }
    sqn = sqrt(double(nonas-1) / sqn);

    // Scale
    for (int i=0; i<N; ++i) {
        if (mask[i] == 1) {
            vec[i] *= sqn;
        } else {
            vec[i] = 0.0;
        }
    }
}


// EO: overloaded function to be used when processing sparse data
//     In such case we do not read from fam file before
// --------------------------------------------------------------
void Data::readPhenotypeFiles(const vector<string> &phenFiles, const int numberIndividuals, MatrixXd& dest) {

    const int NT = phenFiles.size();
    int ninds = -1;

    for (int i=0; i<NT; i++) {
        //cout << "reading phen " << i << ": " << phenFiles[i] << endl;
        VectorXd phen;
        VectorXi mask;
        uint nas = 0;
        readPhenotypeFileAndSetNanMask(phenFiles[i], numberIndividuals, phen, mask, nas);
        cout << "read phen file of length: " << phen.size() << " with " << nas << " NAs" << endl;

        // Make sure that all phenotypes cover the same number of individuals
        if (ninds < 0) {
            ninds = phen.size();

            phenosData.resize(NT, ninds);
            phenosNanMasks.resize(NT, ninds);
            phenosNanNum.resize(NT);
            cout << "data phenosNanNum " << phenosNanNum.size() << endl; 

            phenosData.setZero();
            phenosNanMasks.setZero();
            phenosNanNum.setZero();
        }
        assert(ninds == phen.size());
        assert(ninds == mask.size());

        center_and_scale(phen.data(), mask.data(), ninds, nas);

        for (int j=0; j<ninds; j++) {
            phenosData(i, j)   = phen(j);
            phenosNanMasks(i, j) = mask(j);
        }

        phenosNanNum(i) = nas;
    }
}

void Data::readPhenotypeFileAndSetNanMask(const string &phenFile, const int numberIndividuals, VectorXd& dest, VectorXi& mask, uint& nas) {

    numInds = numberIndividuals;
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
#ifndef USE_MPI
    //cout << "Reading phenotypes from [" + phenFile + "], and setting NAn" << endl;
#endif
    uint line = 0, nonas = 0;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    dest.setZero(numInds);
    mask.setZero(numInds);
    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        if (colData[1+1] != "NA") {
            dest(line) = double( atof(colData[1+1].c_str()) );
            mask(line) = 1;
            nonas += 1;
        } else {
            dest(line) = 1E30; //0./0.; //EO: generate a nan
            mask(line) = 0;
            nas += 1;
        }
        line += 1;
    }
    in.close();
    assert(nonas + nas == numInds);
    assert(line == numInds);
    //printf("nonas = %d, nas = %d\n", nonas, nas);
}

//EO: combined reading of a .phen and .cov files
//    Assume .cov and .phen to be consistent with .fam and .bed!
//--------------------------------------------------------------
void Data::readPhenCovFiles(const string &phenFile, const string covFile, const int numberIndividuals, VectorXd& dest, const int rank) {

    numInds = numberIndividuals;
    dest.setZero(numInds);

    ifstream inp(phenFile.c_str());
    if (!inp)
        throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");

    ifstream inc(covFile.c_str());
    if (!inc)
        throw ("Error: can not open the covariates file [" + covFile + "] to read.");

    uint line = 0, nas = 0, nonas = 0;
    string sep(" \t");
    Gadget::Tokenizer colDataP,  colDataC;
    string            inputStrP, inputStrC;
    std::vector<double> values;

    while (getline(inp, inputStrP)) {

        getline(inc, inputStrC);

        colDataP.getTokens(inputStrP, sep);
        colDataC.getTokens(inputStrC, sep);

        bool naC = false;
        for (int i=2; i<colDataC.size(); i++) {
            if (colDataC[i] == "NA") {
                naC = true;
                break;
            }
        }
 
        if (colDataP[1+1] != "NA" && naC == false) {
            dest[nonas] = double( atof(colDataP[1+1].c_str()) );
            for (int i=2; i<colDataC.size(); i++) {
                values.push_back(std::stod(colDataC[i]));
            }
            nonas += 1;
        } else {
            if (rank == 0)
                cout << "NA(s) detected on line " << line << ": naC? " << naC << ", naP? " << colDataP[1+1] << endl;
            NAsInds.push_back(line);
            nas += 1;
        }
        line += 1;
    }
    inp.close();
    inc.close();
    
    assert(nonas + nas == numInds);
    assert(line == numInds);

    numFixedEffects = values.size() / (line - nas);
    cout << "numFixedEffect = " << numFixedEffects << endl;

    X = Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), (line - nas), numFixedEffects);

    numNAs = nas;

    dest.conservativeResize(numInds-nas);
}



void Data::readPhenotypeFile(const string &phenFile, const int numberIndividuals, VectorXd& dest) {
    numInds = numberIndividuals;
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
#ifndef USE_MPI
    //cout << "Reading phenotypes from [" + phenFile + "]." << endl;
#endif
    uint line = 0, nas = 0, nonas = 0;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    //y.setZero(numInds);
    dest.setZero(numInds);
    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        if (colData[1+1] != "NA") {
            //y[nonas] = double( atof(colData[1+1].c_str()) );
            dest[nonas] = double( atof(colData[1+1].c_str()) );
            //if (nonas < 30) printf("read no na on line %d, nonas %d = %15.10f\n", line, nonas, y(nonas));
            nonas += 1;
        } else {
            //cout << "WARNING: found NA on line/individual " << line << endl;
            NAsInds.push_back(line);
            nas += 1;
        }
        line += 1;
    }
    in.close();
    assert(nonas + nas == numInds);

    numNAs = nas;
    //y.conservativeResize(numInds-nas);
    dest.conservativeResize(numInds-nas);
}

void Data::readPhenotypeFile(const string &phenFile) {
    // NA: missing phenotype
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
#ifndef USE_MPI
    //cout << "Reading phenotypes from [" + phenFile + "]." << endl;
#endif
    map<string, IndInfo*>::iterator it, end=indInfoMap.end();
    IndInfo *ind = NULL;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    string id;
    double tmp = 0.0;
    //correct loop to go through numInds
    y.setZero(numInds);
    uint line = 0, nas = 0, nonas = 0;

    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        id = colData[0] + ":" + colData[1];
        it = indInfoMap.find(id);
        // First one corresponded to mphen variable (1+1)
        if (it != end) {
            ind = it->second;
            if (colData[1+1] != "NA") {
                tmp = double(atof(colData[1+1].c_str()));
                ind->phenotype = tmp;
                y[nonas]       = tmp;
                nonas += 1;
            } else {
                //cout << "WARNING; found NA on line/individual " << line << endl;
                ind->kept = false;
                NAsInds.push_back(line);
                nas += 1;
            }
            line += 1;
        }
    }
    in.close();
    //printf("nonas = %d + nas = %d = numInds = %d\n", nonas, nas, numInds);
    assert(nonas + nas == numInds);

    numNAs = nas;
    y.conservativeResize(numInds-nas);
}


template<typename M>
M Data::readCSVFile (const string &path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    //cout << "rows = " << rows << " vs numInds " << numInds << endl; 
    if (rows != numInds)
        throw(" Error: covariate file has different number of individuals as BED file");
    numFixedEffects = values.size()/rows;
    return Map<const Matrix<typename M::Scalar, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size()/rows);
}

void Data::readCovariateFile(const string &covariateFile ) {
    X = readCSVFile<MatrixXd>(covariateFile);
}



//TODO Finish function to read the group file
void Data::readGroupFile(const string &groupFile) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    ifstream in(groupFile.c_str());
    if (!in) throw ("Error: can not open the group file [" + groupFile + "] to read. Use the --groupIndexFile option!");

    if (rank == 0)
        cout << "Reading groups from [" + groupFile + "]." << endl;

    std::istream_iterator<double> start(in), end;
    std::vector<int> numbers(start, end);
    int* ptr =(int*)&numbers[0];
    G=(Eigen::Map<Eigen::VectorXi>(ptr, numbers.size()));
}

//marion : read annotation file
//group index starts from 0
void Data::readGroupFile_new(const string& groupFile){

    ifstream input(groupFile);
    vector<int> tmp;
    string col1;
    int col2;

    if(!input.is_open()){
        cout<<"Error opening the file"<< endl;
        return;
    }

    while(true){
        input >> col1 >> col2;
        if(input.eof()) break;
        tmp.push_back(col2);
    }

    G=Eigen::VectorXi::Map(tmp.data(), tmp.size());
    
    cout << "Groups read from file" << endl;
}


//marion : read mS (mixtures) for each group
//save as Eigen Matrix
void Data::readmSFile(const string& mSfile){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream in(mSfile);

    if(!in.is_open()){
        cout<<"Error opening the file"<< endl;
        return;
    }

    else if(in.is_open()){

        string whole_text{ istreambuf_iterator<char>(in), istreambuf_iterator<char>() };

        Gadget::Tokenizer strvec;
        Gadget::Tokenizer strT;

        strvec.getTokens(whole_text, ";");
        strT.getTokens(strvec[0],",");
        
        mS=Eigen::MatrixXd(strvec.size(),strT.size());
        numGroups=strvec.size();
        //cout << "numGroups = " << numGroups << endl;
        for (unsigned j=0; j<strvec.size(); ++j) {
            strT.getTokens(strvec[j],",");
            for(unsigned k=0; k<strT.size(); ++k)
                mS(j,k) = stod(strT[k]);
        }
    }

    if (rank == 0)
        cout << "Mixtures read from file" << endl;
}

/*
 * Reads priors v0, s0 for groups from file
 * in : path to file (expected format as "v0,s0; v0,s0; ...")
 * out: void
 */
void Data::read_group_priors(const string& file){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        ifstream in(file);
        string whole_text{ istreambuf_iterator<char>(in), istreambuf_iterator<char>() };

        Gadget::Tokenizer strvec;
        Gadget::Tokenizer strT;
        // get element sizes to instantiate result vector
        strvec.getTokens(whole_text, ";");
        strT.getTokens(strvec[0], ",");
        priors = Eigen::MatrixXd(strvec.size(), strT.size());
        numGroups = strvec.size();
        cout << "numGroups = " << numGroups << endl;
        for (unsigned j=0; j<strvec.size(); ++j) {
            strT.getTokens(strvec[j], ",");
            for (unsigned k=0; k<2; ++k) {
                priors(j, k) = stod(strT[k]);
            }
        }
    } catch (const ifstream::failure& e) {
        cout<<"Error opening the file"<< endl;
    }
    if (rank == 0) {
        cout << "Mixtures read from file" << endl;
    }
}
