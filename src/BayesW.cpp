/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include <cstdlib>
#include <Eigen/Eigen>

#include "BayesW.hpp"
#include "BayesRRm.h"

#include "data.hpp"
#include "distributions_boost.hpp"
#include "options.hpp"
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <libgen.h>
#include <string.h>
#include <boost/range/algorithm.hpp>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <mm_malloc.h>
//#include <mpi.h>
#include <omp.h>
#include <cmath>
#include "BayesW_arms.h"
#include "dense.hpp"
#include "sparse.hpp"
#include "gauss_hermite.hpp"
#include "ars.hpp"
#include "densities.hpp"
#include "utils.hpp"

BayesW::~BayesW() {}

// Pass the vector post_marginals of marginal likelihoods by reference
void BayesW::marginal_likelihood_vec_calc(VectorXd prior_prob, VectorXd &post_marginals, string n,
                                          double vi_sum, double vi_2, double vi_1, double vi_0,
                                          double vi_tau_sum, double vi_tau_2, double vi_tau_1, double vi_tau_0,
                                          double mean, double sd,
                                          double mean_sd_ratio, unsigned int group_index,
                                          const pars_beta_sparse used_data_beta)
{

    double s = 0;
    double M = 1; //auxiliary variables
    double M_pow_mean = 1;
    if (used_data_beta.mixture_value_other != 0) // SEO : before it was  == 0 - is it now correct?
    {
        s = used_data_beta.alpha * used_data_beta.rho * sqrt(used_data_beta.mixture_value * used_data_beta.sigmaG1 / used_data_beta.mixture_value_other/ used_data_beta.sigmaG2)  * used_data_beta.beta_other / sd;
        M = exp(s);
        M_pow_mean = exp(-mean*s);
    }

    double exp_sum = (mean * mean * (vi_0 + vi_tau_0) + M * ((1 - mean) * (1 - mean) * (vi_1 + vi_tau_1) + M * (4 - mean) * (4 - mean) * (vi_2 + vi_tau_2))) / (sd * sd);
    double C_k_other = used_data_beta.mixture_value_other;

    for (int i = 0; i < km1; i++)
    {
        //Calculate the sigma for the adaptive G-H
        double sigma = 1.0 / sqrt(1.0 + used_data_beta.alpha * used_data_beta.alpha * used_data_beta.sigmaG1 * M_pow_mean * (1 - used_data_beta.rho * used_data_beta.rho) * used_data_beta.mixture_value * exp_sum);

        post_marginals(i + 1) = prior_prob(i + 1) * gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, vi_tau_2, vi_tau_1, vi_tau_0, mean, sd, mean_sd_ratio,
                                                                                    used_data_beta, used_data_beta.sigmaG1, used_data_beta.sigmaG2, used_data_beta.beta_other, C_k_other, vi_tau_sum);
    }
}

void BayesW::marginal_likelihood_vec_calc_test(VectorXd prior_prob, VectorXd &post_marginals, string n,
                                               double vi_sum, double vi_2, double vi_1, double vi_0,
                                               double vi_tau_sum, double vi_tau_2, double vi_tau_1, double vi_tau_0,
                                               double mean, double sd,
                                               double mean_sd_ratio, unsigned int group_index,
                                               const pars_beta_sparse used_data_beta)
{
    double s = 0;
    double M = 1; //auxiliary variables
    double M_pow_mean = 1;

    if (used_data_beta.mixture_value_other != 0) // SEO : before it was  == 0 - is it now correct?
    {
        s = used_data_beta.alpha * used_data_beta.rho * used_data_beta.sigmaG1 * sqrt(used_data_beta.mixture_value / used_data_beta.mixture_value_other) / used_data_beta.sigmaG2 * used_data_beta.beta_other / sd;
        M = exp(s);
        M_pow_mean = exp(-mean * s);
    }

    double exp_sum = (mean * mean * (vi_0 + vi_tau_0) + M * ((1 - mean) * (1 - mean) * (vi_1 + vi_tau_1) + M * (4 - mean) * (4 - mean) * (vi_2 + vi_tau_2))) / (sd * sd);
    cout << "exp_sum = " << exp_sum << endl;

    double C_k_other = used_data_beta.mixture_value_other;

    for (int i = 0; i < km1; i++)
    {
        //Calculate the sigma for the adaptive G-H
        double sigma = 1.0 / sqrt(1.0 + used_data_beta.alpha * used_data_beta.alpha * used_data_beta.sigmaG1 * used_data_beta.sigmaG1 * M_pow_mean * (1 - used_data_beta.rho * used_data_beta.rho) * used_data_beta.mixture_value * exp_sum);
        cout << "C_k = " << cVa(group_index, i) << endl;
 
        cout << gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, vi_tau_2, vi_tau_1, vi_tau_0, mean, sd, mean_sd_ratio,
                                                used_data_beta, used_data_beta.sigmaG1, used_data_beta.sigmaG2, used_data_beta.beta_other, C_k_other, vi_tau_sum) << endl;

        post_marginals(i + 1) = prior_prob(i + 1) * gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, vi_tau_2, vi_tau_1, vi_tau_0, mean, sd, mean_sd_ratio,
                                                                                    used_data_beta, used_data_beta.sigmaG1, used_data_beta.sigmaG2, used_data_beta.beta_other, C_k_other, vi_tau_sum);
    }
}

void BayesW::marginal_likelihood_vec_calc2(VectorXd prior_prob, VectorXd &post_marginals, string n,
                                           double vi_sum, double vi_2, double vi_1, double vi_0,
                                           double vi_tau_sum, double vi_tau_2, double vi_tau_1, double vi_tau_0,
                                           double mean, double sd,
                                           double mean_sd_ratio, unsigned int group_index,
                                           const pars_beta_sparse used_data_beta)
{
    double M = 1;
    double s = 0; //auxiliary variables
    double M_pow_mean = 1;

    if (used_data_beta.mixture_value_other != 0) //SEO - changed from ==
    {
        s = used_data_beta.alpha * used_data_beta.rho  * sqrt(used_data_beta.mixture_value * used_data_beta.sigmaG2/ used_data_beta.mixture_value_other/ used_data_beta.sigmaG1 ) * used_data_beta.beta_other / sd;
        M = exp(s);
        M_pow_mean = exp(- mean * s);
    }

    double exp_sum = (mean * mean * (vi_0 - vi_tau_0) + M * ((1 - mean) * (1 - mean) * (vi_1 - vi_tau_1) + M * (4 - mean) * (4 - mean) * (vi_2 - vi_tau_2))) / (sd * sd);
    double C_k_other = used_data_beta.mixture_value_other;

    for (int i = 0; i < km1; i++)
    {
        //Calculate the sigma for the adaptive G-H
        double sigma = 1.0 / sqrt(1.0 + used_data_beta.alpha * used_data_beta.alpha * used_data_beta.sigmaG2 * used_data_beta.sigmaG2 * M_pow_mean * (1 - used_data_beta.rho * used_data_beta.rho) * used_data_beta.mixture_value * exp_sum);

        // Pass the tau-residual with opposite sign
        post_marginals(i + 1) = prior_prob(i + 1) * gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, -vi_tau_2, -vi_tau_1, -vi_tau_0, mean, sd, mean_sd_ratio,
                                                                                    used_data_beta, used_data_beta.sigmaG2, used_data_beta.sigmaG1, used_data_beta.beta_other, C_k_other, -vi_tau_sum);

       // post_marginals(i + 1) = prior_prob(i + 1) * gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, vi_tau_2, vi_tau_1, vi_tau_0, mean, sd, mean_sd_ratio,
       //                                                                             used_data_beta, used_data_beta.sigmaG2, used_data_beta.sigmaG1, used_data_beta.beta_other, C_k_other, vi_tau_sum);

    }
}

void BayesW::marginal_likelihood_vec_calc2_test(VectorXd prior_prob, VectorXd &post_marginals, string n,
                                           double vi_sum, double vi_2, double vi_1, double vi_0,
                                           double vi_tau_sum, double vi_tau_2, double vi_tau_1, double vi_tau_0,
                                           double mean, double sd,
                                           double mean_sd_ratio, unsigned int group_index,
                                           const pars_beta_sparse used_data_beta)
{
    double M = 1;
    double s = 0; //auxiliary variables
    double M_pow_mean = 1;

    if (used_data_beta.mixture_value_other != 0) //SEO - changed from ==
    {
        s = used_data_beta.alpha * used_data_beta.rho * used_data_beta.sigmaG2 * sqrt(used_data_beta.mixture_value / used_data_beta.mixture_value_other) / used_data_beta.sigmaG1 * used_data_beta.beta_other / sd;
        M = exp(s);
        M_pow_mean = exp(- mean * s);
    }

    double exp_sum = (mean * mean * (vi_0 - vi_tau_0) + M * ((1 - mean) * (1 - mean) * (vi_1 - vi_tau_1) + M * (4 - mean) * (4 - mean) * (vi_2 - vi_tau_2))) / (sd * sd);
    double C_k_other = used_data_beta.mixture_value_other;
    cout << "exp_sum = " << exp_sum << endl;

    for (int i = 0; i < km1; i++)
    {
        //Calculate the sigma for the adaptive G-H
        double sigma = 1.0 / sqrt(1.0 + used_data_beta.alpha * used_data_beta.alpha * used_data_beta.sigmaG2 * used_data_beta.sigmaG2 * M_pow_mean * (1 - used_data_beta.rho * used_data_beta.rho) * used_data_beta.mixture_value * exp_sum);
        cout << "C_k = " << cVa(group_index, i) << endl;

        cout << gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, -vi_tau_2, -vi_tau_1, -vi_tau_0, mean, sd, mean_sd_ratio,
                                                used_data_beta, used_data_beta.sigmaG1, used_data_beta.sigmaG2, used_data_beta.beta_other, C_k_other, -vi_tau_sum) << endl;
        // Pass the tau-residual with opposite sign
        post_marginals(i + 1) = prior_prob(i + 1) * gauss_hermite_adaptive_integral(cVa(group_index, i), sigma, n, vi_sum, vi_2, vi_1, vi_0, -vi_tau_2, -vi_tau_1, -vi_tau_0, mean, sd, mean_sd_ratio,
                                                                                    used_data_beta, used_data_beta.sigmaG2, used_data_beta.sigmaG1, used_data_beta.beta_other, C_k_other, -vi_tau_sum);
    }
}


void BayesW::init(unsigned int individualCount, unsigned int individualCount2, unsigned int Mtot, unsigned int fixedCount)
{
    // Read the failure indicator vector
    if (individualCount != (data.fail).size())
    {
        cout << "Number of phenotypes " << individualCount << " was different from the number of failures " << (data.fail).size() << endl;
        exit(1);
    }

    // Linear model variables
    gamma = VectorXd(fixedCount);

    //phenotype vector
    y = VectorXd();
    y2 = VectorXd();

    //residual vectors
    epsilon = VectorXd();  //Summing done across first epoch individuals
    epsilon2 = VectorXd(); //For the rest we sum across second epoch individuals
    epsilon3 = VectorXd();
    epsilon4 = VectorXd();

    // Resize the vectors in the structure
    used_data.X_j = VectorXd(individualCount);
    used_data.X_j2 = VectorXd(individualCount2);

    //Four residual vectors
    used_data.epsilon.resize(individualCount);
    used_data_alpha.epsilon.resize(individualCount);

    used_data.epsilon2.resize(individualCount2);
    used_data_alpha.epsilon2.resize(individualCount2);

    used_data.epsilon3.resize(individualCount2);
    used_data_alpha.epsilon3.resize(individualCount2);

    used_data.epsilon4.resize(individualCount2);
    used_data_alpha.epsilon4.resize(individualCount2);

    //Init the group variables
    data.groups.resize(Mtot);
    data.groups.setZero();
    const int Kt = cva.size() + 1; //Temporary K
    const int Ktm1 = Kt - 1;

    data.mS.resize(Mtot, Ktm1);

    for (int i = 0; i < Mtot; i++)
        data.mS.row(i) = cva;

    if (opt.groupIndexFile != "" && opt.groupMixtureFile != "")
    {
        data.readGroupFile(opt.groupIndexFile);
        data.readmSFile(opt.groupMixtureFile);
    }

    //printf("numGroups = %d, data.groups.size() = %lu, Mtot = %d\n", data.numGroups, data.groups.size(), Mtot);

    numGroups = data.numGroups;
    K = int(data.mS.cols()); //Mixtures + 0th component.
    km1 = K - 1;             //Just mixtures
    sigmaG.resize(numGroups);
    sigmaG.setZero();

    sigmaG2.resize(numGroups);
    sigmaG2.setZero();

    Rho.resize(numGroups);
    Rho.setZero();

    assert(data.groups.size() == Mtot);
    groups = data.groups;
    cVa.resize(numGroups, km1); // component-specific variance   //SEO : Added -1 so that we only store non-zero mixtures
    //Populate cVa. We store only km1 values for mixtures
    for (int i = 0; i < numGroups; i++)
    {
        cVa.row(i) = data.mS.row(i).segment(1, km1); //SEO: Take the segment from 1 (not 0) to exclude the 0th
    }

    // Component variables
    pi_L.resize(numGroups, K);  // prior mixture probabilities (also the 0th)
    pi_L2.resize(numGroups, K); // prior mixture probabilities (also the 0th)

    marginal_likelihoods = VectorXd(K); // likelihood for each mixture component

    // Vector to store the 0th component of the marginal likelihood for each group
    marginal_likelihood_0 = VectorXd(numGroups);
    marginal_likelihood2_0 = VectorXd(numGroups);

    //set priors for pi parameters
    //Give only the first mixture some initial probability of entering
    pi_L.setConstant(1.0 / Mtot);
    pi_L.col(0).array() = 0.99;
    pi_L.col(1).array() = 1 - pi_L.col(0).array() - (km1 - 1) / Mtot;

    // Second epoch
    pi_L2.setConstant(1.0 / Mtot);
    pi_L2.col(0).array() = 0.99;
    pi_L2.col(1).array() = 1 - pi_L2.col(0).array() - (km1 - 1) / Mtot;

    marginal_likelihoods.setOnes(); //Initialize with just ones
    marginal_likelihood_0.setOnes();

    Beta.setZero();
    Beta2.setZero();

    gamma.setZero();

    //initialize epsilon vector as the phenotype vector
    y = data.y.cast<double>().array();
    y2 = data.y2.cast<double>().array();

    epsilon = y;
    epsilon2 = y2;
    epsilon3 = y2;
    epsilon4 = y2;

    mu = (y.sum() + y2.sum())/(individualCount + individualCount2) ; // mean or intercept
    // Initialize the variables in structures
    //Save variance classes

    //Store the vector of failures only in the structure used for sampling alpha
    used_data_alpha.failure_vector = data.fail.cast<double>();
    used_data_alpha.failure_vector2 = data.fail2.cast<double>();

    double denominator = (6 * ((y.array() - mu).square()).sum() / (y.size() - 1));
    used_data.alpha = PI / sqrt(denominator);      // The shape parameter initial value
    used_data_beta.alpha = PI / sqrt(denominator); // The shape parameter initial value

    //Initialise epoch 1 individuals
    for (int i = 0; i < (y.size()); ++i)
    {
        (used_data.epsilon)[i] = y[i] - mu; // Initially, all the BETA elements are set to 0, XBeta = 0
        epsilon[i] = y[i] - mu;
    }

    // and epoch 2 individuals
    for (int i = 0; i < (y2.size()); ++i)
    {
        (used_data.epsilon2)[i] = y2[i] - mu; // Initially, all the BETA elements are set to 0, XBeta = 0
        epsilon2[i] = y2[i] - mu;

        (used_data.epsilon3)[i] = tau - mu;
        epsilon3[i] = tau - mu;

        (used_data.epsilon4)[i] = tau - mu;
        epsilon4[i] = tau - mu;
    }
    // Use h2 = 0.5 for the inital estimate// divided  by the number of groups
    sigmaG.array() = PI_squared / (6 * pow(used_data_beta.alpha, 2)) / numGroups;
    //Also initialise similarly for the second epoch
    sigmaG2.array() = PI_squared / (6 * pow(used_data_beta.alpha, 2)) / numGroups;

    //Restart variables
    epsilon_restart.resize(individualCount);
    epsilon_restart.setZero();

    gamma_restart.resize(fixedCount);
    gamma_restart.setZero();

    xI_restart.resize(fixedCount);

    /* Prior value selection for the variables */
    /* We set them to be weakly informative (in .hpp file) by default */
    if (opt.hypPriorsFile == "")
    {
        /* alpha */
        used_data_alpha.alpha_0 = alpha_0;
        used_data_alpha.kappa_0 = kappa_0;
        /* mu */
        used_data.sigma_mu = sigma_mu;
        /* covariates */
        used_data.sigma_covariate = sigma_covariate;
    }
    else
    {
        /* alpha */
        used_data_alpha.alpha_0 = data.sigmaEPriors(0);
        used_data_alpha.kappa_0 = data.sigmaEPriors(1);
        /* mu */
        used_data.sigma_mu = data.muPrior;
        /* covariates */
        used_data.sigma_covariate = data.covarPrior;
    }

    // Save the number of events
    used_data.d = used_data_alpha.failure_vector.array().sum() + used_data_alpha.failure_vector2.array().sum();
    used_data_alpha.d = used_data.d;
}

void BayesW::init_from_restart(const int K, const uint M, const uint Mtot, const uint Ntot1, const uint Ntot2, const uint fixtot,
                               const int *MrankS, const int *MrankL, const bool use_xfiles_in_restart)
{
    //Use the regular bW initialisation
    init(Ntot1, Ntot2, Mtot, fixtot);

    //TODO SEO - fix for epoch restarts
    // data.read_mcmc_output_csv_file_bW(opt.mcmcOut, opt.thin, opt.save, K, mu, sigmaG, used_data.alpha, pi_L,
    //                                   iteration_to_restart_from, first_thinned_iteration, first_saved_iteration);

    // Set new random seed for the ARS in case of restart. In long run we should use dist object for simulating from uniform distribution
    //@@@EO srand(opt.seed + iteration_to_restart_from);

    //Carry the values to the other structures
    used_data_beta.alpha = used_data.alpha;

    MPI_Barrier(MPI_COMM_WORLD);

    data.read_mcmc_output_bet_file(opt.mcmcOut,
                                   Mtot, iteration_to_restart_from, first_thinned_iteration, opt.thin,
                                   MrankS, MrankL, use_xfiles_in_restart,
                                   Beta);

    data.read_mcmc_output_cpn_file(opt.mcmcOut,
                                   Mtot, iteration_to_restart_from, first_thinned_iteration, opt.thin,
                                   MrankS, MrankL, use_xfiles_in_restart,
                                   components);
    //TODO - Fix restart
    //   data.read_mcmc_output_eps_file(opt.mcmcOut, Ntot, iteration_to_restart_from,
    //                                  epsilon_restart);

    cout << opt.mcmcOut << ".mrk" << endl;

    data.read_mcmc_output_idx_file(opt.mcmcOut, "mrk", M, iteration_to_restart_from, opt.bayesType,
                                   markerI_restart);

    if (opt.covariates)
    {
        data.read_mcmc_output_gam_file_bW(opt.mcmcOut, opt.save, fixtot, gamma_restart);

        data.read_mcmc_output_idx_file(opt.mcmcOut, "xiv", fixtot, iteration_to_restart_from, opt.bayesType,
                                       xI_restart);
    }

    // Adjust starting iteration number.
    iteration_start = iteration_to_restart_from + 1;

    MPI_Barrier(MPI_COMM_WORLD);
}

//EO: MPI GIBBS
//-------------
int BayesW::runMpiGibbs_bW()
{

    const unsigned int numFixedEffects(data.numFixedEffects);

    char buff[LENBUF];
    char buff_gamma[LENBUF_gamma];
    int nranks, rank, name_len, result;
    double dalloc = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File outfh, betfh, bet2fh, epsfh, eps2fh, eps3fh, eps4fh, gamfh, cpnfh, cpn2fh, mrkfh, xivfh;
    MPI_File xbetfh, xcpnfh, xbet2fh, xcpn2fh;
    MPI_Status status;
    MPI_Info info;

    // Set up processing options
    // -------------------------
    if (rank < 0)
    {
        opt.printBanner();
        opt.printProcessingOptions();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        if (rank == 0)
        {
            int nt = omp_get_num_threads();
            if (omp_get_thread_num() == 0)
                printf("INFO   : OMP parallel regions will use %d thread(s)\n", nt);
        }
    }

    //TODO: Write new functions for Ntot1 and Ntot2 to read from different epochs
    uint Ntot1 = data.set_Ntot1(rank, opt);
    uint Ntot2 = data.set_Ntot2(rank, opt);

    const uint Mtot = data.set_Mtot(rank, opt);

    //Reset the dist
    //@@@EO WHY THIS ONE ??? dist.reset_rng((uint)(opt.seed + rank*1000));

    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot1=%d epoch 1 individuals, Ntot2=%d epoch 2 individuals.\n", Mtot, Ntot1, Ntot2);

    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    uint M = MrankL[rank];
    if (rank % 10 == 0)
    {
        printf("INFO   : rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);
    }

    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: at this stage Ntot is not yet adjusted for missing phenotypes,
    //       hence the correction in the call
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    define_blocks_of_markers(Ntot1 + Ntot2 - data.numNAs, IrankS, IrankL, nranks); //TODO: SEO - rethink

    Beta.resize(M);
    Beta.setZero();

    //Similarly for the second epoch
    Beta2.resize(M);
    Beta2.setZero();

    components.resize(M);
    components.setZero();
    components2.resize(M);
    components2.setZero();

    std::vector<int> markerI;

    markerI_restart.resize(M);
    std::fill(markerI_restart.begin(), markerI_restart.end(), 0);

    std::vector<int> mark2sync;
    std::vector<double> dbet2sync;

    dalloc += M * sizeof(int) / 1E9;        // for components
    dalloc += 2 * M * sizeof(double) / 1E9; // for Beta and Acum

    //Same for epoch 2
    dalloc += M * sizeof(int) / 1E9;        // for components
    dalloc += 2 * M * sizeof(double) / 1E9; // for Beta and Acum

    // Adapt the --thin and --save options such that --save >= --thin and --save%--thin = 0
    // ------------------------------------------------------------------------------------
    if (opt.save < opt.thin)
    {
        opt.save = opt.thin;
        if (rank == 0)
            printf("WARNING: opt.save was lower that opt.thin ; opt.save reset to opt.thin (%d)\n", opt.thin);
    }
    if (opt.save % opt.thin != 0)
    {
        if (rank == 0)
            printf("WARNING: opt.save (= %d) was not a multiple of opt.thin (= %d)\n", opt.save, opt.thin);
        opt.save = int(opt.save / opt.thin) * opt.thin;
        if (rank == 0)
            printf("         opt.save reset to %d, the closest multiple of opt.thin (%d)\n", opt.save, opt.thin);
    }

    // Invariant initializations (from scratch / from restart)
    // -------------------------------------------------------
    string lstfp = opt.mcmcOut + ".lst";
    string outfp = opt.mcmcOut + ".csv";
    string betfp = opt.mcmcOut + ".bet";
    string xbetfp = opt.mcmcOut + ".xbet";
    string cpnfp = opt.mcmcOut + ".cpn";
    string xcpnfp = opt.mcmcOut + ".xcpn";

    string bet2fp = opt.mcmcOut + ".bet2";
    string xbet2fp = opt.mcmcOut + ".xbet2";
    string cpn2fp = opt.mcmcOut + ".cpn2";
    string xcpn2fp = opt.mcmcOut + ".xcpn2";

    string gamfp = opt.mcmcOut + ".gam";
    string xivfp = opt.mcmcOut + ".xiv";
    string rngfp = opt.mcmcOut + ".rng." + std::to_string(rank);
    string mrkfp = opt.mcmcOut + ".mrk." + std::to_string(rank);
    string epsfp = opt.mcmcOut + ".eps." + std::to_string(rank);
    string eps2fp = opt.mcmcOut + ".eps2." + std::to_string(rank);
    string eps3fp = opt.mcmcOut + ".eps3." + std::to_string(rank);
    string eps4fp = opt.mcmcOut + ".eps4." + std::to_string(rank);

    if (opt.restart)
    {
        //TODO: Fix restart
        init_from_restart(K, M, Mtot, Ntot1, Ntot2, numFixedEffects, MrankS, MrankL, opt.useXfilesInRestart);
        if (rank == 0)
            data.print_restart_banner(opt.mcmcOut.c_str(), iteration_to_restart_from, iteration_start);

        dist.read_rng_state_from_file(rngfp);

        // Rename output files so that we do not erase from failed job!
        //EO: add a function, to update both Nam and Dir!
        opt.mcmcOutNam += "_rs";
        opt.mcmcOut = opt.mcmcOutDir + "/" + opt.mcmcOutNam;
        lstfp = opt.mcmcOut + ".lst";
        outfp = opt.mcmcOut + ".csv";
        betfp = opt.mcmcOut + ".bet";
        xbetfp = opt.mcmcOut + ".xbet"; // Last saved iteration of bet; .bet has full history
        cpnfp = opt.mcmcOut + ".cpn";
        xcpnfp = opt.mcmcOut + ".xcpn"; // Idem

        bet2fp = opt.mcmcOut + ".bet2";
        xbet2fp = opt.mcmcOut + ".xbet2"; // Last saved iteration of bet; .bet has full history
        cpn2fp = opt.mcmcOut + ".cpn2";
        xcpn2fp = opt.mcmcOut + ".xcpn2"; // Idem

        rngfp = opt.mcmcOut + ".rng." + std::to_string(rank);
        mrkfp = opt.mcmcOut + ".mrk." + std::to_string(rank);
        epsfp = opt.mcmcOut + ".eps." + std::to_string(rank);
        eps2fp = opt.mcmcOut + ".eps2." + std::to_string(rank);
        eps3fp = opt.mcmcOut + ".eps3." + std::to_string(rank);
        eps4fp = opt.mcmcOut + ".eps4." + std::to_string(rank);

        gamfp = opt.mcmcOut + ".gam";
        xivfp = opt.mcmcOut + ".xiv";
    }
    else
    {
        // Set new random seed for the ARS in case of restart. In long run we should use dist object for simulating from uniform distribution
        //@@@EO srand(opt.seed);
        //TODO: SEO - fix NAs

        init(Ntot1, Ntot2, Mtot, numFixedEffects); //- data.numNAs
    }

    cass.resize(numGroups, K); //rows are groups columns are mixtures
    cass2.resize(numGroups, K);

    MatrixXi sum_cass(numGroups, K);  // To store the sum of cass elements over all ranks
    MatrixXi sum_cass2(numGroups, K); // To store the sum of cass elements over all ranks

    // Initialise the vector of prior inclusion probability (pi) hyperparameters
    VectorXd dirc(K);
    dirc.array() = 1.0;

    // Define sumSigmaG for creating "safe limit"
    double sumSigmaG = sigmaG.sum();
    double sumSigmaG2 = sigmaG2.sum(); // For epoch 2

    // Build global repartition of markers over the groups
    VectorXi MtotGrp(numGroups);
    MtotGrp.setZero();
    for (int i = 0; i < Mtot; i++)
    {
        MtotGrp[groups[i]] += 1;
    }
    VectorXi m0(numGroups);   // non-zero elements per group
    VectorXi m0_2(numGroups); // non-zero elements per group (in epoch 2)

    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    xI_restart.resize(data.X.cols());

    //    dist.reset_rng((uint)(opt.seed + rank*1000));

    // Build a list of the files to tar
    // --------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    ofstream listFile;
    listFile.open(lstfp);
    listFile << outfp << "\n";
    listFile << betfp << "\n";
    listFile << xbetfp << "\n";
    listFile << cpnfp << "\n";
    listFile << xcpnfp << "\n";

    listFile << bet2fp << "\n";
    listFile << xbet2fp << "\n";
    listFile << cpn2fp << "\n";
    listFile << xcpn2fp << "\n";

    listFile << gamfp << "\n";
    listFile << xivfp << "\n";
    for (int i = 0; i < nranks; i++)
    {
        listFile << opt.mcmcOut + ".rng." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".mrk." + std::to_string(i) << "\n";
        listFile << opt.mcmcOut + ".eps." + std::to_string(i) << "\n";
    }
    listFile.close();
    MPI_Barrier(MPI_COMM_WORLD);

    // Delete old files (fp appended with "_rs" in case of restart, so that
    // original files are kept untouched) and create new ones
    // --------------------------------------------------------------------
    if (rank == 0)
    {
        MPI_File_delete(outfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xbetfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xcpnfp.c_str(), MPI_INFO_NULL);

        MPI_File_delete(bet2fp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xbet2fp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpn2fp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xcpn2fp.c_str(), MPI_INFO_NULL);

        MPI_File_delete(gamfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(xivfp.c_str(), MPI_INFO_NULL);
    }
    MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(eps2fp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(eps3fp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(eps4fp.c_str(), MPI_INFO_NULL);

    MPI_File_delete(mrkfp.c_str(), MPI_INFO_NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xbetfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xbetfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xcpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xcpnfh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, bet2fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &bet2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xbet2fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xbet2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpn2fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpn2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xcpn2fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xcpn2fh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, gamfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &gamfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, xivfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &xivfh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_SELF, epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF, eps2fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &eps2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF, eps3fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &eps3fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF, eps4fp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &eps4fh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_SELF, mrkfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &mrkfh), __LINE__, __FILE__);

    // First element of the .bet, .cpn and .acu files is the
    // total number of processed markers
    // -----------------------------------------------------
    MPI_Offset offset = 0;

    if (rank == 0)
    {
        check_mpi(MPI_File_write_at(betfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xbetfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xcpnfh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

        check_mpi(MPI_File_write_at(bet2fh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xbet2fh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpn2fh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(xcpn2fh, offset, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    double tl = -mysecond();

    // Read the data (from sparse representation by default)
    // -----------------------------------------------------
    // SEO: Now do it by two bed files (in one people had the event in epoch one and in the other one they had it in epoch two)
    size_t *N1S, *N1L, *N2S, *N2L, *NMS, *NML;
    N1S = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(NML, __LINE__, __FILE__);
    //    dalloc += 6.0 * double(M) * sizeof(size_t) / 1E9;

    size_t *N1S_2, *N1L_2, *N2S_2, *N2L_2, *NMS_2, *NML_2;
    N1S_2 = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N1S_2, __LINE__, __FILE__);
    N1L_2 = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N1L_2, __LINE__, __FILE__);
    N2S_2 = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N2S_2, __LINE__, __FILE__);
    N2L_2 = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(N2L_2, __LINE__, __FILE__);
    NMS_2 = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(NMS_2, __LINE__, __FILE__);
    NML_2 = (size_t *)_mm_malloc(size_t(M) * sizeof(size_t), 64);
    check_malloc(NML_2, __LINE__, __FILE__);
    dalloc += 12.0 * double(M) * sizeof(size_t) / 1E9; //SEO: Is this correct?

    // Boolean mask for using BED representation or not (SPARSE otherwise)
    // For markers with USEBED == true then the BED representation is
    // converted on the fly to SPARSE the time for the corresponding marker
    // to be processed
    // --------------------------------------------------------------------
    bool *USEBED;
    USEBED = (bool *)_mm_malloc(M * sizeof(bool), 64);
    check_malloc(USEBED, __LINE__, __FILE__);
    for (int i = 0; i < M; i++)
        USEBED[i] = false;
    int nusebed = 0;

    uint *I1, *I2, *IM;
    uint *I1_2, *I2_2, *IM_2; //For epoch 2

    size_t taskBytes = 0;

    if (opt.readFromBedFile)
    {
        //Read the data sets for two epochs
        data.load_data_from_bed_file(opt.bedFile, Ntot1, M, rank, MrankS[rank],
                                     N1S, N1L, I1,
                                     N2S, N2L, I2,
                                     NMS, NML, IM,
                                     taskBytes);

        data.load_data_from_bed_file(opt.bedFile2, Ntot2, M, rank, MrankS[rank],
                                     N1S_2, N1L_2, I1_2,
                                     N2S_2, N2L_2, I2_2,
                                     NMS_2, NML_2, IM_2,
                                     taskBytes);
    }
    else
    { //TODO : Implement two epochs for sparse solution
        cout << "Sparse solution not yet implemented for two epochs" << endl;
        exit(1000);
        string sparseOut = opt.get_sparse_output_filebase(rank);
        data.load_data_from_sparse_files(rank, nranks, M, MrankS, MrankL, sparseOut,
                                         N1S, N1L, I1,
                                         N2S, N2L, I2,
                                         NMS, NML, IM,
                                         taskBytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tl += mysecond();

    if (rank == 0)
    {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes  =>  BW = %7.3f GB/s\n", rank, tl, taskBytes, (double)taskBytes * 1E-9 / tl);
        fflush(stdout);
    }

    // Correct each marker for individuals with missing phenotype
    // ----------------------------------------------------------
    //TODO: Implement NA correction for two epochs
    if (data.numNAs > 0)
    {

        if (rank == 0)
            printf("INFO   : applying %d corrections to genotype data due to missing phenotype data (NAs in .phen).\n", data.numNAs);

        data.sparse_data_correct_for_missing_phenotype(N1S, N1L, I1, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(N2S, N2L, I2, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(NMS, NML, IM, M, USEBED);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            printf("INFO   : finished applying NA corrections.\n");

        // Adjust N upon number of NAs
        //Ntot -= data.numNAs; //TODO - Fix NA-s
        if (rank == 0 && data.numNAs > 0)
            printf("INFO   : Ntot adjusted by -%d to account for NAs in phenotype file. Now Ntot=%d\n", data.numNAs, Ntot1 + Ntot2);
    }

    // Compute statistics (from sparse info)
    // -------------------------------------
    //if (rank == 0) printf("INFO   : start computing statistics on Ntot = %d individuals\n", Ntot);
    double dN = (double)(Ntot1 + Ntot2);
    double dNm1 = (double)(Ntot1 + Ntot2 - 1);
    double *mave, *mstd, *sum_failure, *sum_failure2, *sum_failure_fix;

    mave = (double *)_mm_malloc(size_t(M) * sizeof(double), 64);
    check_malloc(mave, __LINE__, __FILE__);
    mstd = (double *)_mm_malloc(size_t(M) * sizeof(double), 64);
    check_malloc(mstd, __LINE__, __FILE__);
    sum_failure = (double *)_mm_malloc(size_t(M) * sizeof(double), 64);
    check_malloc(sum_failure, __LINE__, __FILE__);
    sum_failure2 = (double *)_mm_malloc(size_t(M) * sizeof(double), 64);
    check_malloc(sum_failure2, __LINE__, __FILE__);

    sum_failure_fix = (double *)_mm_malloc(size_t(numFixedEffects) * sizeof(double), 64);
    check_malloc(sum_failure_fix, __LINE__, __FILE__);

    dalloc += 5 * size_t(M) * sizeof(double) / 1E9;   //SEO - Is this correct? Previously it was 2

    double tmp0, tmp1, tmp2;
    double temp_fail_sum = used_data_alpha.failure_vector.array().sum();
    double temp_fail_sum2 = used_data_alpha.failure_vector2.array().sum();

    for (int i = 0; i < M; ++i)
    {
        // For now use the old way to compute means
        mave[i] = (double(N1L[i]) + double(N1L_2[i]) + 2.0 * (double(N2L[i]) + double(N2L_2[i]))) / (dN - double(NML[i] - double(NML_2[i])));

        tmp1 = (double(N1L[i]) + double(N1L_2[i])) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = (double(N2L[i]) + double(N2L_2[i])) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot1 + Ntot2 - N1L[i] - N1L_2[i] - N2L[i] - N2L_2[i] - NML[i] - NML_2[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        //TODO At some point we need to turn sd to 1/sd for speed
        //mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
        mstd[i] = sqrt((tmp0 + tmp1 + tmp2) / double(Ntot1 + Ntot2 - 1));

        //TODO: This part has to be thought through
        int temp_sum = 0;
        for (size_t ii = N1S[i]; ii < (N1S[i] + N1L[i]); ii++)
        {
            temp_sum += used_data_alpha.failure_vector(I1[ii]);
        }
        for (size_t ii = N2S[i]; ii < (N2S[i] + N2L[i]); ii++)
        {
            temp_sum += 2 * used_data_alpha.failure_vector(I2[ii]);
        }
        sum_failure[i] = (temp_sum - mave[i] * temp_fail_sum) / mstd[i];

        temp_sum = 0;
        for (size_t ii = N1S_2[i]; ii < (N1S_2[i] + N1L_2[i]); ii++)
        {
            temp_sum += used_data_alpha.failure_vector2(I1_2[ii]);
        }
        for (size_t ii = N2S_2[i]; ii < (N2S_2[i] + N2L_2[i]); ii++)
        {
            temp_sum += 2 * used_data_alpha.failure_vector2(I2_2[ii]);
        }
        sum_failure2[i] = (temp_sum - mave[i] * temp_fail_sum2) / mstd[i];

        //printf("marker %6d mean %20.15f, std = %20.15f (%.1f / %.15f)  (%15.10f, %15.10f, %15.10f)\n", i, mave[i], mstd[i], double(Ntot - 1), tmp0+tmp1+tmp2, tmp1, tmp2, tmp0);
    }

    //If there are fixed effects, find the same values for them
    //TODO: Implement covariates later for the two epochs
    if (opt.covariates)
    {
        for (int fix_i = 0; fix_i < numFixedEffects; fix_i++)
        {
            sum_failure_fix[fix_i] = ((data.X.col(fix_i).cast<double>()).array() * used_data_alpha.failure_vector.array()).sum();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)
        std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;

    // Build list of markers
    // ---------------------
    for (int i = 0; i < M; ++i)
        markerI.push_back(i);

    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();

    //First epoch has the individuals that experienced event there
    double *y, *tmpEps, *deltaEps, *dEpsSum, *deltaSum, *epsilon, *tmpEps_vi, *tmp_deltaEps;
    const size_t NDB = size_t(Ntot1) * sizeof(double);
    y = (double *)_mm_malloc(NDB, 64);
    check_malloc(y, __LINE__, __FILE__);
    epsilon = (double *)_mm_malloc(NDB, 64);
    check_malloc(epsilon, __LINE__, __FILE__);
    tmpEps_vi = (double *)_mm_malloc(NDB, 64);
    check_malloc(tmpEps_vi, __LINE__, __FILE__);
    tmpEps = (double *)_mm_malloc(NDB, 64);
    check_malloc(tmpEps, __LINE__, __FILE__);
    tmp_deltaEps = (double *)_mm_malloc(NDB, 64);
    check_malloc(tmp_deltaEps, __LINE__, __FILE__);
    deltaEps = (double *)_mm_malloc(NDB, 64);
    check_malloc(deltaEps, __LINE__, __FILE__);
    dEpsSum = (double *)_mm_malloc(NDB, 64);
    check_malloc(dEpsSum, __LINE__, __FILE__);
    deltaSum = (double *)_mm_malloc(NDB, 64);
    check_malloc(deltaSum, __LINE__, __FILE__);

    //We also keep in memory residuals 2,3,4
    double *y2, *tmpEps2, *deltaEps2, *dEpsSum2, *deltaSum2, *epsilon2, *tmpEps_vi2, *tmp_deltaEps2;
    const size_t NDB_2 = size_t(Ntot2) * sizeof(double);
    y2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(y2, __LINE__, __FILE__);
    epsilon2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(epsilon2, __LINE__, __FILE__);
    tmpEps_vi2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmpEps_vi2, __LINE__, __FILE__);
    tmpEps2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmpEps2, __LINE__, __FILE__);
    tmp_deltaEps2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmp_deltaEps2, __LINE__, __FILE__);
    deltaEps2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(deltaEps2, __LINE__, __FILE__);
    dEpsSum2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(dEpsSum2, __LINE__, __FILE__);
    deltaSum2 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(deltaSum2, __LINE__, __FILE__);

    double *tmpEps3, *deltaEps3, *dEpsSum3, *deltaSum3, *epsilon3, *tmpEps_vi3, *tmp_deltaEps3;
    epsilon3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(epsilon3, __LINE__, __FILE__);
    tmpEps_vi3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmpEps_vi3, __LINE__, __FILE__);
    tmpEps3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmpEps3, __LINE__, __FILE__);
    tmp_deltaEps3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmp_deltaEps3, __LINE__, __FILE__);
    deltaEps3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(deltaEps3, __LINE__, __FILE__);
    dEpsSum3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(dEpsSum3, __LINE__, __FILE__);
    deltaSum3 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(deltaSum3, __LINE__, __FILE__);

    double *tmpEps4, *deltaEps4, *dEpsSum4, *deltaSum4, *epsilon4, *tmpEps_vi4, *tmp_deltaEps4;
    epsilon4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(epsilon4, __LINE__, __FILE__);
    tmpEps_vi4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmpEps_vi4, __LINE__, __FILE__);
    tmpEps4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmpEps4, __LINE__, __FILE__);
    tmp_deltaEps4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(tmp_deltaEps4, __LINE__, __FILE__);
    deltaEps4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(deltaEps4, __LINE__, __FILE__);
    dEpsSum4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(dEpsSum4, __LINE__, __FILE__);
    deltaSum4 = (double *)_mm_malloc(NDB_2, 64);
    check_malloc(deltaSum4, __LINE__, __FILE__);

    cout << "sizeof(double) = " << sizeof(double) << " vs sizeof(long double) = " << sizeof(long double) << endl;

    const size_t NLDB = size_t(Ntot1) * sizeof(long double);
    const size_t NLDB_2 = size_t(Ntot2) * sizeof(long double);

    long double *tmp_vi = (long double *)_mm_malloc(NLDB, 64);
    check_malloc(tmp_vi, __LINE__, __FILE__);
    long double *vi = (long double *)_mm_malloc(NLDB, 64);
    check_malloc(vi, __LINE__, __FILE__);

    long double *tmp_vi2 = (long double *)_mm_malloc(NLDB_2, 64);
    check_malloc(tmp_vi2, __LINE__, __FILE__);
    long double *vi2 = (long double *)_mm_malloc(NLDB_2, 64);
    check_malloc(vi2, __LINE__, __FILE__);

    long double *tmp_vi3 = (long double *)_mm_malloc(NLDB_2, 64);
    check_malloc(tmp_vi3, __LINE__, __FILE__);
    long double *vi3 = (long double *)_mm_malloc(NLDB_2, 64);
    check_malloc(vi3, __LINE__, __FILE__);

    long double *tmp_vi4 = (long double *)_mm_malloc(NLDB_2, 64);
    check_malloc(tmp_vi4, __LINE__, __FILE__);
    long double *vi4 = (long double *)_mm_malloc(NLDB_2, 64);
    check_malloc(vi4, __LINE__, __FILE__);

    dalloc += NDB * 10 / 1E9;
  //  dalloc += NDB_2 * 10 / 1E9;
    dalloc += NDB_2 * 10 * 3 / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("INFO   : overall allocation %.3f GB\n", totalloc);

    set_array(dEpsSum, 0.0, Ntot1);
    set_array(dEpsSum2, 0.0, Ntot2);
    set_array(dEpsSum3, 0.0, Ntot2);
    set_array(dEpsSum4, 0.0, Ntot2);

    // Copy, center and scale phenotype observations
    // In bW we are not scaling and centering phenotypes
    //TODO: Fix the restart for two epochs
    if (opt.restart)
    {

        for (int i = 0; i < Ntot1; ++i)
        {
            epsilon[i] = epsilon_restart[i];
        }
        //TODO - Fix restart
        /*for (int i=0; i<Ntot2; ++i){
            epsilon2[i] = epsilon_restart2[i];
        }*/

        markerI = markerI_restart;

        if (opt.covariates)
        {
            for (int i = 0; i < numFixedEffects; i++)
            {
                gamma[i] = gamma_restart[i];
                xI[i] = xI_restart[i];
            }
        }
    }
    else
    {
        for (int i = 0; i < Ntot1; ++i)
        {
            y[i] = data.y(i);
            epsilon[i] = y[i] - mu;
        }

        for (int i = 0; i < Ntot2; ++i)
        {
            y2[i] = data.y2(i);
            epsilon2[i] = y2[i] - mu;
            epsilon3[i] = tau - mu;
            epsilon4[i] = epsilon3[i];
        }
    }

    VectorXd sum_beta_squaredNorm, sum_beta_squaredNorm2, sum_beta1_beta2;

    double beta, betaOld, deltaBeta, p, acum;
    double beta2, betaOld2, deltaBeta2, p2;

    // beta1_beta2 is the vector of covariances
    VectorXd beta_squaredNorm, beta_squaredNorm2, beta1_beta2;
    size_t markoff;
    int marker, cx;

    beta_squaredNorm.resize(numGroups);
    sum_beta_squaredNorm.resize(numGroups);

    beta_squaredNorm2.resize(numGroups);
    sum_beta_squaredNorm2.resize(numGroups);

    beta1_beta2.resize(numGroups);
    sum_beta1_beta2.resize(numGroups);
    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;

    // Main iteration loop
    // -------------------
    //bool replay_it = false;
    double tot_sync_ar1 = 0.0;
    double tot_sync_ar2 = 0.0;
    int tot_nsync_ar1 = 0;
    int tot_nsync_ar2 = 0;
    int *glob_info, *tasks_len, *tasks_dis, *stats_len, *stats_dis;

    if (opt.sparseSync)
    {
        glob_info = (int *)_mm_malloc(size_t(nranks * 2) * sizeof(int), 64);
        check_malloc(glob_info, __LINE__, __FILE__);
        tasks_len = (int *)_mm_malloc(size_t(nranks) * sizeof(int), 64);
        check_malloc(tasks_len, __LINE__, __FILE__);
        tasks_dis = (int *)_mm_malloc(size_t(nranks) * sizeof(int), 64);
        check_malloc(tasks_dis, __LINE__, __FILE__);
        stats_len = (int *)_mm_malloc(size_t(nranks) * sizeof(int), 64);
        check_malloc(stats_len, __LINE__, __FILE__);
        stats_dis = (int *)_mm_malloc(size_t(nranks) * sizeof(int), 64);
        check_malloc(stats_dis, __LINE__, __FILE__);
    }

    const int ninit = 4;
    const int npoint = 100;
    const int nsamp = 1;
    const int ncent = 4;

    //Set iteration_start=0
    for (uint iteration = iteration_start; iteration < opt.chainLength; iteration++)
    {

        double start_it = MPI_Wtime();
        double it_sync_ar1 = 0.0;
        double it_sync_ar2 = 0.0;
        int it_nsync_ar1 = 0;
        int it_nsync_ar2 = 0;

        /* 1. Intercept (mu) */
        //Removed sampleMu function on its own
        int err = 0;

        int neval;
        double xsamp[0], xcent[10], qcent[10] = {5., 30., 70., 95.};
        double convex = 1.0;
        int dometrop = 0;
        double xprev = 0.0;
        double xinit[4] = {0.95 * mu, mu, 1.005 * mu, 1.01 * mu}; // Initial abscissae
        double *p_xinit = xinit;

        double xl = 2;
        double xr = 5; //xl and xr and the maximum and minimum values between which we sample

        //Update before sampling
        for (int mu_ind = 0; mu_ind < Ntot1; mu_ind++)
        {
            (used_data.epsilon)[mu_ind] = epsilon[mu_ind] + mu; // we add to epsilon =Y+mu-X*beta
        }
        for (int mu_ind = 0; mu_ind < Ntot2; mu_ind++)
        {
            (used_data.epsilon2)[mu_ind] = epsilon2[mu_ind] + mu;
            (used_data.epsilon3)[mu_ind] = epsilon3[mu_ind] + mu;
            (used_data.epsilon4)[mu_ind] = epsilon4[mu_ind] + mu;
        }

        // Use ARS to sample mu (with density mu_dens, using parameters from used_data)
        err = arms(xinit, ninit, &xl, &xr, mu_dens, &used_data, &convex, npoint,
                   dometrop, &xprev, xsamp, nsamp, qcent, xcent, ncent, &neval, dist);

        errorCheck(err); // If there is error, stop the program

        check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        mu = xsamp[0]; // Save the sampled value
        //Update after sampling
        for (int mu_ind = 0; mu_ind < Ntot1; mu_ind++)
        {
            epsilon[mu_ind] = (used_data.epsilon)[mu_ind] - mu; // we add to epsilon =Y+mu-X*beta
        }
        for (int mu_ind = 0; mu_ind < Ntot2; mu_ind++)
        {
            epsilon2[mu_ind] = (used_data.epsilon2)[mu_ind] - mu;
            epsilon3[mu_ind] = (used_data.epsilon3)[mu_ind] - mu;
            epsilon4[mu_ind] = (used_data.epsilon4)[mu_ind] - mu;
        }

        ////////// End sampling mu
        /* 1a. Fixed effects (gammas) */
        if (opt.covariates)
        {

            double gamma_old = 0;
            //std::shuffle(xI.begin(), xI.end(), dist.rng);
            boost::uniform_int<> unii(0, numFixedEffects - 1);
            boost::variate_generator<boost::mt19937 &, boost::uniform_int<>> generator(dist.rng, unii);
            if (opt.shuffleMarkers)
            {
                boost::range::random_shuffle(xI, generator);
            }

            //Use only rank 0 shuffling
            check_mpi(MPI_Bcast(xI.data(), xI.size(), MPI_INT, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

            MPI_Barrier(MPI_COMM_WORLD);

            for (int fix_i = 0; fix_i < numFixedEffects; fix_i++)
            {
                gamma_old = gamma(xI[fix_i]);

                neval = 0;
                xsamp[0] = 0;
                convex = 1.0;
                dometrop = 0;
                xprev = 0.0;

                xinit[0] = gamma_old - 0.075 / 30; // Initial abscissae
                xinit[1] = gamma_old;
                xinit[2] = gamma_old + 0.075 / 60;
                xinit[3] = gamma_old + 0.075 / 30;

                xl = gamma_old - 0.075;
                xr = gamma_old + 0.075; // Initial left and right (pseudo) extremes

                used_data.X_j = data.X.col(xI[fix_i]).cast<double>(); //Take from the fixed effects matrix
                used_data.sum_failure = sum_failure_fix[xI[fix_i]];

                for (int k = 0; k < Ntot1; k++)
                {
                    (used_data.epsilon)[k] = epsilon[k] + used_data.X_j[k] * gamma_old; // we adjust the residual with the respect to the previous gamma value
                }
                for (int k = 0; k < Ntot2; k++)
                {
                    (used_data.epsilon2)[k] = epsilon2[k] + used_data.X_j[k] * gamma_old; // we adjust the residual with the respect to the previous gamma value
                }
                //TODO - Fix covariates by concatenating epsilons here
                // Sample using ARS
                err = arms(xinit, ninit, &xl, &xr, gamma_dens, &used_data, &convex,
                           npoint, dometrop, &xprev, xsamp, nsamp, qcent, xcent, ncent, &neval, dist);
                errorCheck(err);

                //Use only rank 0
                check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

                gamma(xI[fix_i]) = xsamp[0]; // Save the new result
                for (int k = 0; k < Ntot1; k++)
                {
                    epsilon[k] = (used_data.epsilon)[k] - used_data.X_j[k] * gamma(xI[fix_i]); // we adjust the residual with the respect to the previous gamma value
                }
                for (int k = 0; k < Ntot2; k++)
                {
                    epsilon2[k] = (used_data.epsilon2)[k] - used_data.X_j[k] * gamma(xI[fix_i]); // we adjust the residual with the respect to the previous gamma value
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        ////////// End sampling gamma

        // ARS parameters
        neval = 0;
        xsamp[0] = 0;
        convex = 1.0;
        dometrop = 0;
        xprev = 0.0;
        xinit[0] = (used_data.alpha) * 0.5; // Initial abscissae
        xinit[1] = used_data.alpha;
        xinit[2] = (used_data.alpha) * 1.05;
        xinit[3] = (used_data.alpha) * 1.10;

        // Initial left and right (pseudo) extremes
        xl = 0.0;
        xr = 40.0;

        //Give the residual to alpha structure
        //used_data_alpha.epsilon = epsilon;
        for (int alpha_ind = 0; alpha_ind < Ntot1; alpha_ind++)
        {
            (used_data_alpha.epsilon)[alpha_ind] = epsilon[alpha_ind];
        }
        for (int alpha_ind = 0; alpha_ind < Ntot2; alpha_ind++)
        {
            (used_data_alpha.epsilon2)[alpha_ind] = epsilon2[alpha_ind];
            (used_data_alpha.epsilon3)[alpha_ind] = epsilon3[alpha_ind];
            (used_data_alpha.epsilon4)[alpha_ind] = epsilon4[alpha_ind];
        }

        //Sample using ARS
        err = arms(xinit, ninit, &xl, &xr, alpha_dens, &used_data_alpha, &convex,
                   npoint, dometrop, &xprev, xsamp, nsamp, qcent, xcent, ncent, &neval, dist);

        errorCheck(err);

        check_mpi(MPI_Bcast(&xsamp[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

        used_data.alpha = xsamp[0];
        used_data_beta.alpha = xsamp[0];
        MPI_Barrier(MPI_COMM_WORLD);

        // Calculate the vector of exponent of the adjusted residuals
        for (int i = 0; i < Ntot1; ++i)
        {
            vi[i] = expl((long double)(used_data.alpha * epsilon[i] - EuMasc));
        }
        for (int i = 0; i < Ntot2; ++i)
        {
            vi2[i] = expl((long double)(used_data.alpha * epsilon2[i] - EuMasc));
            vi3[i] = expl((long double)(used_data.alpha * epsilon3[i] - EuMasc));
            vi4[i] = expl((long double)(used_data.alpha * epsilon4[i] - EuMasc));
        }

        boost::uniform_int<> unii(0, M - 1);
        boost::variate_generator<boost::mt19937 &, boost::uniform_int<>> generator(dist.rng, unii);
        if (opt.shuffleMarkers)
        {
            //std::shuffle(markerI.begin(), markerI.end(), dist.rng);
            boost::range::random_shuffle(markerI, generator);
        }

        m0.setZero();
        m0_2.setZero();

        cass.setZero();
        cass2.setZero();

        for (int i = 0; i < Ntot1; ++i)
            tmpEps[i] = epsilon[i];

        for (int i = 0; i < Ntot2; ++i)
        {
            tmpEps2[i] = epsilon2[i];
            tmpEps3[i] = epsilon3[i];
            tmpEps4[i] = epsilon4[i];
        }

        double cumSumDeltaBetas = 0.0;
        double cumSumDeltaBetas2 = 0.0;

        double task_sum_abs_deltabeta = 0.0;
        double task_sum_abs_deltabeta2 = 0.0;

        int sinceLastSync = 0;

        // First element for the marginal likelihoods is always is pi_0 *sqrt(pi) for
        //marginal_likelihoods(0) = pi_L(0) * sqrtPI;
        //Precalculate the product already before for each group
        for (int gg = 0; gg < numGroups; gg++)
        {
            marginal_likelihood_0(gg) = pi_L(gg, 0) * sqrtPI;
            marginal_likelihood2_0(gg) = pi_L2(gg, 0) * sqrtPI; // For the second epoch
        }
        //Set the sum of beta squared 0
        beta_squaredNorm.setZero();
        beta_squaredNorm2.setZero();
        beta1_beta2.setZero();

        // Loop over (shuffled) markers
        // ----------------------------

        for (int j = 0; j < lmax; j++)
        {
            sinceLastSync += 1;

            if (j < M)
            {
                marker = markerI[j];
                beta = Beta(marker);
                beta2 = Beta2(marker);
                unsigned int cur_group = groups[MrankS[rank] + marker];
                /////////////////////////////////////////////////////////
                //Replace the sampleBeta function with the inside of the function
                double vi_sum = 0.0;
                double vi_1 = 0.0;
                double vi_2 = 0.0;

                double vi2_sum = 0.0;
                double vi2_1 = 0.0;
                double vi2_2 = 0.0;

                double vi3_sum = 0.0;
                double vi3_1 = 0.0;
                double vi3_2 = 0.0;

                double vi4_sum = 0.0;
                double vi4_1 = 0.0;
                double vi4_2 = 0.0;

                used_data_beta.sigmaG1 = sigmaG[cur_group];
                used_data_beta.sigmaG2 = sigmaG2[cur_group];
                used_data_beta.rho = Rho[cur_group];

                marginal_likelihoods(0) = marginal_likelihood_0(cur_group); //Each group has now different marginal likelihood at 0

                //Change the residual vector only if the previous beta was non-zero

                if (Beta(marker) != 0.0)
                {
                    //Calculate the change in epsilon if we remove the previous marker effect (-Beta(marker))
                    //Epsilon and epsilon3 are using beta1
                    set_array(tmp_deltaEps, 0.0, Ntot1);
                    set_array(tmp_deltaEps3, 0.0, Ntot2);

                    sparse_scaadd(tmp_deltaEps, Beta(marker),
                                  I1, N1S[marker], N1L[marker],
                                  I2, N2S[marker], N2L[marker],
                                  IM, NMS[marker], NML[marker],
                                  mave[marker], 1.0 / mstd[marker], Ntot1);
                    sparse_scaadd(tmp_deltaEps3, Beta(marker),
                                  I1_2, N1S_2[marker], N1L_2[marker],
                                  I2_2, N2S_2[marker], N2L_2[marker],
                                  IM_2, NMS_2[marker], NML_2[marker],
                                  mave[marker], 1.0 / mstd[marker], Ntot2);

                    //Create the temporary vector to store the vector without the last Beta(marker)
                    add_arrays(tmpEps_vi, epsilon, tmp_deltaEps, Ntot1);
                    add_arrays(tmpEps_vi3, epsilon3, tmp_deltaEps3, Ntot2);

                    //Also find the transformed residuals
                    for (uint i = 0; i < Ntot1; ++i)
                    {
                        tmp_vi[i] = expl((long double)(used_data.alpha * tmpEps_vi[i] - EuMasc));
                    }
                    for (uint i = 0; i < Ntot2; ++i)
                    {
                        tmp_vi3[i] = expl((long double)(used_data.alpha * tmpEps_vi3[i] - EuMasc));
                    }

                    vi_sum = sum_array_elements(tmp_vi, Ntot1);
                    vi_2 = sparse_partial_sum(tmp_vi, I2, N2S[marker], N2L[marker]);
                    vi_1 = sparse_partial_sum(tmp_vi, I1, N1S[marker], N1L[marker]);

                    vi3_sum = sum_array_elements(tmp_vi3, Ntot2);
                    vi3_2 = sparse_partial_sum(tmp_vi3, I2_2, N2S_2[marker], N2L_2[marker]);
                    vi3_1 = sparse_partial_sum(tmp_vi3, I1_2, N1S_2[marker], N1L_2[marker]);
                    //printf("iter %3d  A: vi_sum _1 _2 = %25.16f %25.16f %25.16f  %20.17f\n", iteration, vi_sum, vi_1, vi_2, Beta(marker));
                }
                else
                {
                    // Calculate the sums of vi elements
                    vi_sum = sum_array_elements(vi, Ntot1);
                    vi_2 = sparse_partial_sum(vi, I2, N2S[marker], N2L[marker]);
                    vi_1 = sparse_partial_sum(vi, I1, N1S[marker], N1L[marker]);

                    vi3_sum = sum_array_elements(vi3, Ntot2);
                    vi3_2 = sparse_partial_sum(vi3, I2_2, N2S_2[marker], N2L_2[marker]);
                    vi3_1 = sparse_partial_sum(vi3, I1_2, N1S_2[marker], N1L_2[marker]);
                    //printf("iter %3d  B: vi_sum _1 _2 = %25.16f %25.16f %25.16f  %20.17f\n", iteration, vi_sum, vi_1, vi_2, Beta(marker));
                }

                double vi_0 = vi_sum - vi_1 - vi_2;
                double vi3_0 = vi3_sum - vi3_1 - vi3_2;
                if (Beta2(marker) != 0.0)
                {
                    //Calculate the change in epsilon if we remove the previous marker effect (-Beta(marker))
                    //Epsilon and epsilon3 are using beta1
                    set_array(tmp_deltaEps2, 0.0, Ntot1);
                    set_array(tmp_deltaEps4, 0.0, Ntot2);

                    sparse_scaadd(tmp_deltaEps2, Beta2(marker),
                                  I1_2, N1S_2[marker], N1L_2[marker],
                                  I2_2, N2S_2[marker], N2L_2[marker],
                                  IM, NMS_2[marker], NML_2[marker],
                                  mave[marker], 1.0 / mstd[marker], Ntot2);
                    sparse_scaadd(tmp_deltaEps4, Beta2(marker),
                                  I1_2, N1S_2[marker], N1L_2[marker],
                                  I2_2, N2S_2[marker], N2L_2[marker],
                                  IM_2, NMS_2[marker], NML_2[marker],
                                  mave[marker], 1.0 / mstd[marker], Ntot2);

                    //Create the temporary vector to store the vector without the last Beta(marker)
                    add_arrays(tmpEps_vi2, epsilon2, tmp_deltaEps2, Ntot2);
                    add_arrays(tmpEps_vi4, epsilon4, tmp_deltaEps4, Ntot2);

                    //Also find the transformed residuals
                    for (uint i = 0; i < Ntot2; ++i)
                    {
                        tmp_vi2[i] = expl((long double)(used_data.alpha * tmpEps_vi2[i] - EuMasc));
                    }
                    for (uint i = 0; i < Ntot2; ++i)
                    {
                        tmp_vi4[i] = expl((long double)(used_data.alpha * tmpEps_vi4[i] - EuMasc));
                    }

                    vi2_sum = sum_array_elements(tmp_vi2, Ntot2);
                    vi2_2 = sparse_partial_sum(tmp_vi2, I2_2, N2S_2[marker], N2L_2[marker]);
                    vi2_1 = sparse_partial_sum(tmp_vi2, I1_2, N1S_2[marker], N1L_2[marker]);

                    vi4_sum = sum_array_elements(tmp_vi4, Ntot2);
                    vi4_2 = sparse_partial_sum(tmp_vi4, I2_2, N2S_2[marker], N2L_2[marker]);
                    vi4_1 = sparse_partial_sum(tmp_vi4, I1_2, N1S_2[marker], N1L_2[marker]);
                    //printf("iter %3d  A: vi_sum _1 _2 = %25.16f %25.16f %25.16f  %20.17f\n", iteration, vi_sum, vi_1, vi_2, Beta(marker));
                }
                else
                {
                    // Calculate the sums of vi elements
                    vi2_sum = sum_array_elements(vi2, Ntot2);
                    vi2_2 = sparse_partial_sum(vi2, I2_2, N2S_2[marker], N2L_2[marker]);
                    vi2_1 = sparse_partial_sum(vi2, I1_2, N1S_2[marker], N1L_2[marker]);

                    vi4_sum = sum_array_elements(vi4, Ntot2);
                    vi4_2 = sparse_partial_sum(vi4, I2_2, N2S_2[marker], N2L_2[marker]);
                    vi4_1 = sparse_partial_sum(vi4, I1_2, N1S_2[marker], N1L_2[marker]);
                    //printf("iter %3d  B: vi_sum _1 _2 = %25.16f %25.16f %25.16f  %20.17f\n", iteration, vi_sum, vi_1, vi_2, Beta(marker));
                }

                double vi2_0 = vi2_sum - vi2_1 - vi2_2;
                double vi4_0 = vi4_sum - vi4_1 - vi4_2;
                /* Calculate the mixture probability */
                double p = dist.unif_rng(); //Generate number from uniform distribution (for sampling from categorical distribution)
                double p2 = dist.unif_rng(); //Generate number from uniform distribution (for sampling from categorical distribution)

                // Calculate the (ratios of) marginal likelihoods

                //EO!!!!!!!!!!!!!!
                used_data_beta.sum_failure = sum_failure[marker];
                //printf("m %3d -> sum fail = %20.15f\n", marker, sum_failure[marker]);

                used_data_beta.beta_other = Beta2(marker);

                if (components2[marker] != 0)
                {
                    used_data_beta.mixture_value_other = cVa(cur_group, components2[marker] - 1);
                }
                else
                {
                    used_data_beta.mixture_value_other = 0;
                }
                if (false)
                {
                    cout << "C block" << endl;
                    cout << "Marker "  << j << " reached " << "C1" << endl; 

                    cout << "marginal likelihoods size = " << marginal_likelihoods.size() << endl;
                    cout << marginal_likelihoods(0) << "," << marginal_likelihoods(1) << endl;
                    cout << "vi pars = " << vi_sum << ", " << vi_2 << "," << vi_1 << ", " << vi_0 << endl;
                    cout << "vi_tau pars = " << vi3_sum << ", " << vi3_2 << "," << vi3_1 << ", " << vi3_0 << endl;
                    cout << "pi values = " << pi_L(cur_group, 0) << ", " << pi_L(cur_group, 1) << "," << pi_L(cur_group, 2) << endl;

                    marginal_likelihood_vec_calc_test(pi_L.row(cur_group), marginal_likelihoods, quad_points, vi_sum, vi_2, vi_1, vi_0,
                                                      vi3_sum, vi3_2, vi3_1, vi3_0,
                                                      mave[marker], mstd[marker], mave[marker] / mstd[marker], cur_group, used_data_beta);
                    cout << marginal_likelihoods(0) << "," << marginal_likelihoods(1) << "," << marginal_likelihoods(2) << endl;

                }

                marginal_likelihood_vec_calc(pi_L.row(cur_group), marginal_likelihoods, quad_points, vi_sum, vi_2, vi_1, vi_0,
                                             vi3_sum, vi3_2, vi3_1, vi3_0,
                                             mave[marker], mstd[marker], mave[marker] / mstd[marker], cur_group, used_data_beta);

                // Calculate the probability that marker is 0
                double acum = marginal_likelihoods(0) / marginal_likelihoods.sum();

                //Loop through the possible mixture classes
                for (int k = 0; k < K; k++)
                {
                    if (p <= acum)
                    {
                        //if zeroth component
                        if (k == 0)
                        {
                            Beta(marker) = 0.0;
                            cass(cur_group, 0) += 1;
                            components[marker] = k;
                            //if(j % 1 == 0) cout << "Epoch 1 no sample " << j << "; p=" << p << "M likelihoods: "<< marginal_likelihoods(0) << "," << marginal_likelihoods(1)  << endl;

                        }
                        // If is not 0th component then sample using ARS
                        else
                        {
           // cout << "Epoch 1 sample " << j << "; p=" << p << "M likelihoods: "<< marginal_likelihoods(0) << "," << marginal_likelihoods(1)  << endl;
                            //used_data_beta.sum_failure = sum_failure(marker);
                            used_data_beta.mean = mave[marker];
                            used_data_beta.sd = mstd[marker];
                            used_data_beta.mean_sd_ratio = mave[marker] / mstd[marker];

                            used_data_beta.mixture_value = cVa(cur_group, k - 1); //k-1 because cVa stores only non-zero in bW

                            used_data_beta.vi_0 = vi_0;
                            used_data_beta.vi_1 = vi_1;
                            used_data_beta.vi_2 = vi_2;

                            used_data_beta.vi_tau_0 = vi3_0;
                            used_data_beta.vi_tau_1 = vi3_1;
                            used_data_beta.vi_tau_2 = vi3_2;

                            //printf("safe_limit: 2.0 * sqrt(%20.17f * %20.17f)\n", sumSigmaG, used_data_beta.mixture_value);
                            double safe_limit = 2.0 * std::sqrt(sumSigmaG * used_data_beta.mixture_value); // Need to think is this safe enough
                            // ARS parameters
                            neval = 0;
                            xsamp[0] = 0.0;
                            convex = 1.0;
                            dometrop = 0;
                            xprev = 0.0;

                            xinit[0] = Beta(marker) - safe_limit / 10.0; // Initial abscissae
                            xinit[1] = Beta(marker);
                            xinit[2] = Beta(marker) + safe_limit / 20.0;
                            xinit[3] = Beta(marker) + safe_limit / 10.0;

                            // Initial left and right (pseudo) extremes
                            xl = Beta(marker) - safe_limit; //Construct the hull around previous beta value
                            xr = Beta(marker) + safe_limit;

                            // Sample using ARS
                            //printf("beta:  %20.17f, safe_limit = %20.17f\n", Beta(marker), safe_limit);
                            //printf("xinit: %20.17f %20.17f %20.17f %20.17f\n", xinit[0], xinit[1], xinit[2], xinit[3]);

                            err = arms(xinit, ninit, &xl, &xr, beta_dens, &used_data_beta, &convex,
                                       npoint, dometrop, &xprev, xsamp, nsamp, qcent, xcent, ncent, &neval, dist);

                            errorCheck(err);

                            Beta(marker) = xsamp[0]; // Save the new result
                            //cout << "Sampled beta1 = " <<  Beta(marker) << endl;
                            cass(cur_group, k) += 1;
                            components[marker] = k;

                            // Write the sum of the beta squared to the vector
                            beta_squaredNorm[groups[MrankS[rank] + marker]] += Beta[marker] * Beta[marker];
                        }
                        break;
                    }
                    else
                    {
                        if ((k + 1) == km1)
                        {
                            acum = 1; // In the end probability will be 1
                        }
                        else
                        {
                            acum += marginal_likelihoods(k + 1) / marginal_likelihoods.sum();
                        }
                    }
                }

                //////////////////////////
                //Do same for second epoch
                marginal_likelihoods(0) = marginal_likelihood2_0(cur_group); //Each group has now different marginal likelihood at 0
                used_data_beta.sum_failure = sum_failure2[marker];
                //printf("m %3d -> sum fail = %20.15f\n", marker, sum_failure[marker]);

                used_data_beta.beta_other = Beta(marker);

                if (components[marker] != 0)
                {
                    used_data_beta.mixture_value_other = cVa(cur_group, components[marker] - 1);
                }
                else
                {
                    used_data_beta.mixture_value_other = 0;
                }

                if (false)
                {
                    cout << "EPOCH 2!" << endl;
                    cout << "marginal likelihoods size = " << marginal_likelihoods.size() << endl;
                    cout << marginal_likelihoods(0) << "," << marginal_likelihoods(1) << endl;
                    cout << "vi pars = " << vi2_sum << ", " << vi2_2 << "," << vi2_1 << ", " << vi2_0 << endl;
                    cout << "vi_tau pars = " << vi4_sum << ", " << vi4_2 << "," << vi4_1 << ", " << vi4_0 << endl;
                    cout << "pi values = " << pi_L2(cur_group, 0) << ", " << pi_L2(cur_group, 1) << "," << pi_L2(cur_group, 2) << endl;

                    marginal_likelihood_vec_calc2_test(pi_L.row(cur_group), marginal_likelihoods, quad_points, vi2_sum, vi2_2, vi2_1, vi2_0,
                                                      vi4_sum, vi4_2, vi4_1, vi4_0,
                                                      mave[marker], mstd[marker], mave[marker] / mstd[marker], cur_group, used_data_beta);
                    cout << marginal_likelihoods(0) << "," << marginal_likelihoods(1) << "," << marginal_likelihoods(2) << endl;

                }

                // Now we use residuals 2 and 4
                marginal_likelihood_vec_calc2(pi_L2.row(cur_group), marginal_likelihoods, quad_points, vi2_sum, vi2_2, vi2_1, vi2_0,
                                              vi4_sum, vi4_2, vi4_1, vi4_0,
                                              mave[marker], mstd[marker], mave[marker] / mstd[marker], cur_group, used_data_beta);
                // cout << marginal_likelihoods(0) << "," << marginal_likelihoods(1) << endl;
                // Calculate the probability that marker is 0
                acum = marginal_likelihoods(0) / marginal_likelihoods.sum();
       
                //Loop through the possible mixture classes
                for (int k = 0; k < K; k++)
                {
                    if (p2 <= acum)
                    {
                        //if zeroth component
                        if (k == 0)
                        {
                            Beta2(marker) = 0.0;
                            cass2(cur_group, 0) += 1;
                            components2[marker] = k;

                            //if(j % 1 == 0) cout << "Epoch 2 no sample " << j << "; p=" << p2 << "M likelihoods: "<< marginal_likelihoods(0) << "," << marginal_likelihoods(1)  << endl;

                        }
                        // If is not 0th component then sample using ARS
                        else
                        {
                            //used_data_beta.sum_failure = sum_failure(marker);
                            //cout << "Epoch 2 sample " << j << "; p=" << p2 << "M likelihoods: "<< marginal_likelihoods(0) << "," << marginal_likelihoods(1)  << endl;
                            used_data_beta.mean = mave[marker];
                            used_data_beta.sd = mstd[marker];
                            used_data_beta.mean_sd_ratio = mave[marker] / mstd[marker];

                            used_data_beta.mixture_value = cVa(cur_group, k - 1); //k-1 because cVa stores only non-zero in bW

                            used_data_beta.vi_0 = vi2_0;
                            used_data_beta.vi_1 = vi2_1;
                            used_data_beta.vi_2 = vi2_2;

                            used_data_beta.vi_tau_0 = vi4_0;
                            used_data_beta.vi_tau_1 = vi4_1;
                            used_data_beta.vi_tau_2 = vi4_2;

                            //printf("safe_limit: 2.0 * sqrt(%20.17f * %20.17f)\n", sumSigmaG, used_data_beta.mixture_value);
                            double safe_limit = 2.0 * std::sqrt(sumSigmaG2 * used_data_beta.mixture_value); // Need to think is this safe enough
                            // ARS parameters
                            neval = 0;
                            xsamp[0] = 0.0;
                            convex = 1.0;
                            dometrop = 0;
                            xprev = 0.0;

                            xinit[0] = Beta2(marker) - safe_limit / 10.0; // Initial abscissae
                            xinit[1] = Beta2(marker);
                            xinit[2] = Beta2(marker) + safe_limit / 20.0;
                            xinit[3] = Beta2(marker) + safe_limit / 10.0;

                            // Initial left and right (pseudo) extremes
                            xl = Beta2(marker) - safe_limit; //Construct the hull around previous beta value
                            xr = Beta2(marker) + safe_limit;

                            // Sample using ARS
                            //printf("beta:  %20.17f, safe_limit = %20.17f\n", Beta2(marker), safe_limit);
                            //printf("xinit: %20.17f %20.17f %20.17f %20.17f\n", xinit[0], xinit[1], xinit[2], xinit[3]);

                            err = arms(xinit, ninit, &xl, &xr, beta_dens2, &used_data_beta, &convex,
                                       npoint, dometrop, &xprev, xsamp, nsamp, qcent, xcent, ncent, &neval, dist);

                            errorCheck(err);

                            Beta2(marker) = xsamp[0]; // Save the new result
                           // cout << "Sampled beta2 = " <<  Beta2(marker) << endl;

                            cass2(cur_group, k) += 1;
                            components2[marker] = k;

                            // Write the sum of the beta squared to the vector
                            beta_squaredNorm2[groups[MrankS[rank] + marker]] += Beta2[marker] * Beta2[marker];
                            //If the first epoch marker was also included, add the product to the covariance

                            if (Beta[marker] != 0)
                            {
                                beta1_beta2[groups[MrankS[rank] + marker]] += Beta[marker] * Beta2[marker];
                            }
                        }
                        break;
                    }
                    else
                    {
                        if ((k + 1) == km1)
                        {
                            acum = 1; // In the end probability will be 1
                        }
                        else
                        {
                            acum += marginal_likelihoods(k + 1) / marginal_likelihoods.sum();
                        }
                    }
                }

                betaOld = beta;
                beta = Beta(marker);
                deltaBeta = betaOld - beta;
                // And the same for second epoch
                betaOld2 = beta2;
                beta2 = Beta2(marker);
                deltaBeta2 = betaOld2 - beta2;

                //continue;

                //if (deltaBeta != 0.0)
                //printf("deltaBeta = %20.16f\n", deltaBeta);

                // Compute delta epsilon
                // beta will affect residuals 1 and 3
                if (deltaBeta != 0.0)
                {
                    //printf("it %d, task %3d, marker %5d has non-zero deltaBeta = %15.10f (%15.10f, %15.10f) => %15.10f) 1,2,M: %lu, %lu, %lu\n", iteration, rank, marker, deltaBeta, mave[marker], mstd[marker],  deltaBeta * mstd[marker], N1L[marker], N2L[marker], NML[marker]);

                    if (opt.sparseSync && nranks > 1)
                    {
                        //TODO - Check if it's feasible to use for multiepoch bW
                        mark2sync.push_back(marker);
                        dbet2sync.push_back(deltaBeta);
                    }
                    else
                    {
                        // Change in residual 1 (across Ntot1 individuals)
                        sparse_scaadd(deltaEps, deltaBeta,
                                      I1, N1S[marker], N1L[marker],
                                      I2, N2S[marker], N2L[marker],
                                      IM, NMS[marker], NML[marker],
                                      mave[marker], 1 / mstd[marker], Ntot1); //Use here 1/sd

                        // Update local sum of delta epsilon
                        add_arrays(dEpsSum, deltaEps, Ntot1);

                        // Change in residual 3 (across Ntot2 individuals)
                        sparse_scaadd(deltaEps3, deltaBeta,
                                      I1_2, N1S_2[marker], N1L_2[marker],
                                      I2_2, N2S_2[marker], N2L_2[marker],
                                      IM_2, NMS_2[marker], NML_2[marker],
                                      mave[marker], 1 / mstd[marker], Ntot2); //Use here 1/sd

                        // Update local sum of delta epsilon
                        add_arrays(dEpsSum3, deltaEps3, Ntot2);
                    }
                }

                if (deltaBeta2 != 0.0)
                {
                    //printf("it %d, task %3d, marker %5d has non-zero deltaBeta = %15.10f (%15.10f, %15.10f) => %15.10f) 1,2,M: %lu, %lu, %lu\n", iteration, rank, marker, deltaBeta, mave[marker], mstd[marker],  deltaBeta * mstd[marker], N1L[marker], N2L[marker], NML[marker]);

                    if (opt.sparseSync && nranks > 1)
                    {
                        //TODO - Check if it's feasible to use for multiepoch bW
                        mark2sync.push_back(marker);
                        dbet2sync.push_back(deltaBeta);
                    }
                    else
                    {
                        // Change in residual 2 (across Ntot2 individuals)
                        sparse_scaadd(deltaEps2, deltaBeta2,
                                      I1_2, N1S_2[marker], N1L_2[marker],
                                      I2_2, N2S_2[marker], N2L_2[marker],
                                      IM_2, NMS_2[marker], NML_2[marker],
                                      mave[marker], 1 / mstd[marker], Ntot2); //Use here 1/sd

                        // Update local sum of delta epsilon
                        add_arrays(dEpsSum2, deltaEps2, Ntot2);

                        // Change in residual 4 (across Ntot2 individuals)
                        sparse_scaadd(deltaEps4, deltaBeta2,
                                      I1_2, N1S_2[marker], N1L_2[marker],
                                      I2_2, N2S_2[marker], N2L_2[marker],
                                      IM_2, NMS_2[marker], NML_2[marker],
                                      mave[marker], 1 / mstd[marker], Ntot2); //Use here 1/sd

                        // Update local sum of delta epsilon
                        add_arrays(dEpsSum4, deltaEps4, Ntot2);
                    }
                }
            }

            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else
            {
                //cout << "rank " << rank << " with M=" << M << " waiting for " << lmax << endl;
                deltaBeta = 0.0;
                deltaBeta2 = 0.0;

                set_array(deltaEps, 0.0, Ntot1);
                set_array(deltaEps2, 0.0, Ntot2);
                set_array(deltaEps3, 0.0, Ntot2);
                set_array(deltaEps4, 0.0, Ntot2);
            }

            //continue;

            task_sum_abs_deltabeta += fabs(deltaBeta);
            task_sum_abs_deltabeta2 += fabs(deltaBeta2);

            // Check whether we have a non-zero beta somewhere
            //if (nranks > 1 && (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1)) {
            if (nranks > 1 && (sinceLastSync >= opt.syncRate || j == lmax - 1))
            {
                //MPI_Barrier(MPI_COMM_WORLD);
                double tb = MPI_Wtime();
                check_mpi(MPI_Allreduce(&task_sum_abs_deltabeta, &cumSumDeltaBetas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                check_mpi(MPI_Allreduce(&task_sum_abs_deltabeta2, &cumSumDeltaBetas2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

                double te = MPI_Wtime();
                tot_sync_ar1 += te - tb;
                it_sync_ar1 += te - tb;
                tot_nsync_ar1 += 1;
                it_nsync_ar1 += 1;
            }
            else
            {
                cumSumDeltaBetas = task_sum_abs_deltabeta;
                cumSumDeltaBetas2 = task_sum_abs_deltabeta2;
            }
            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, beta, cumSumDeltaBetas);

            if (cumSumDeltaBetas != 0.0 && (sinceLastSync >= opt.syncRate || j == lmax - 1))
            {

                // Update local copy of epsilon
                //MPI_Barrier(MPI_COMM_WORLD);

                if (nranks > 1)
                {
                    double tb = MPI_Wtime();

                    // Sparse synchronization
                    // ----------------------
                    //TODO - sparse sync yet to be thought and implemented
                    if (opt.sparseSync)
                    {

                        uint task_m2s = (uint)mark2sync.size();

                        // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
                        double *task_stat = (double *)_mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
                        check_malloc(task_stat, __LINE__, __FILE__);

                        // Compute total number of elements to be sent by each task
                        uint task_size = 0;
                        for (int i = 0; i < task_m2s; i++)
                        {
                            task_size += (N1L[mark2sync[i]] + N2L[mark2sync[i]] + NML[mark2sync[i]] + 3);
                            task_stat[2 * i + 0] = mave[mark2sync[i]];
                            task_stat[2 * i + 1] = mstd[mark2sync[i]] * dbet2sync[i]; //CHANGE mstd later!
                            //printf("Task %3d, m2s %d/%d: 1: %8lu, 2: %8lu, m: %8lu, info: 3); stats are (%15.10f, %15.10f)\n", rank, i, task_m2s, N1L[ mark2sync[i] ], N2L[ mark2sync[i] ], NML[ mark2sync[i] ], task_stat[2 * i + 0], task_stat[2 * i + 1]);
                        }
                        //printf("Task %3d final task_size = %8d elements to send from task_m2s = %d markers to sync.\n", rank, task_size, task_m2s);
                        //fflush(stdout);

                        // Get the total numbers of markers and corresponding indices to gather

                        const int NEL = 2;
                        uint task_info[NEL] = {};
                        task_info[0] = task_m2s;
                        task_info[1] = task_size;

                        check_mpi(MPI_Allgather(task_info, NEL, MPI_UNSIGNED, glob_info, NEL, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);

                        int tdisp_ = 0, sdisp_ = 0, glob_m2s = 0, glob_size = 0;
                        for (int i = 0; i < nranks; i++)
                        {
                            tasks_len[i] = glob_info[2 * i + 1];
                            tasks_dis[i] = tdisp_;
                            tdisp_ += tasks_len[i];
                            stats_len[i] = glob_info[2 * i] * 2;
                            stats_dis[i] = sdisp_;
                            sdisp_ += glob_info[2 * i] * 2;
                            glob_size += tasks_len[i];
                            glob_m2s += glob_info[2 * i];
                        }
                        //printf("glob_info: markers to sync: %d, with glob_size = %7d elements (sum of all task_size)\n", glob_m2s, glob_size);
                        //fflush(stdout);

                        // Build task's array to spread: | marker 1                             | marker 2
                        //                               | n1 | n2 | nm | data1 | data2 | datam | n1 | n2 | nm | data1 | ...
                        // -------------------------------------------------------------------------------------------------
                        uint *task_dat = (uint *)_mm_malloc(size_t(task_size) * sizeof(uint), 64);
                        check_malloc(task_dat, __LINE__, __FILE__);

                        int loc = 0;
                        for (int i = 0; i < task_m2s; i++)
                        {
                            task_dat[loc] = N1L[mark2sync[i]];
                            loc += 1;
                            task_dat[loc] = N2L[mark2sync[i]];
                            loc += 1;
                            task_dat[loc] = NML[mark2sync[i]];
                            loc += 1;
                            for (uint ii = 0; ii < N1L[mark2sync[i]]; ii++)
                            {
                                task_dat[loc] = I1[N1S[mark2sync[i]] + ii];
                                loc += 1;
                            }
                            for (uint ii = 0; ii < N2L[mark2sync[i]]; ii++)
                            {
                                task_dat[loc] = I2[N2S[mark2sync[i]] + ii];
                                loc += 1;
                            }
                            for (uint ii = 0; ii < NML[mark2sync[i]]; ii++)
                            {
                                task_dat[loc] = IM[NMS[mark2sync[i]] + ii];
                                loc += 1;
                            }
                        }
                        assert(loc == task_size);

                        // Allocate receive buffer for all the data
                        uint *glob_dat = (uint *)_mm_malloc(size_t(glob_size) * sizeof(uint), 64);
                        check_malloc(glob_dat, __LINE__, __FILE__);

                        check_mpi(MPI_Allgatherv(task_dat, task_size, MPI_UNSIGNED,
                                                 glob_dat, tasks_len, tasks_dis, MPI_UNSIGNED, MPI_COMM_WORLD),
                                  __LINE__, __FILE__);
                        _mm_free(task_dat);

                        double *glob_stats = (double *)_mm_malloc(size_t(glob_size * 2) * sizeof(double), 64);
                        check_malloc(glob_stats, __LINE__, __FILE__);

                        check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                                 glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD),
                                  __LINE__, __FILE__);
                        _mm_free(task_stat);

                        // Compute global delta epsilon deltaSum
                        size_t loci = 0;
                        for (int i = 0; i < glob_m2s; i++)
                        {

                            //printf("m2s %d/%d (loci = %d): %d, %d, %d\n", i, glob_m2s, loci, glob_dat[loci], glob_dat[loci + 1], glob_dat[loci + 2]);

                            double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                            //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1]);

                            // Set all to 0 contribution
                            if (i == 0)
                            {
                                //TODO - fix sparse sync
                                //set_array(deltaSum, lambda0, Ntot);
                            }
                            else
                            {
                                //offset_array(deltaSum, lambda0, Ntot);
                            }

                            // M -> revert lambda 0 (so that equiv to add 0.0)
                            size_t S = loci + (size_t)(3 + glob_dat[loci] + glob_dat[loci + 1]);
                            size_t L = glob_dat[loci + 2];
                            //cout << "task " << rank << " M: start = " << S << ", len = " << L <<  endl;
                            sparse_add(deltaSum, -lambda0, glob_dat, S, L);

                            // 1 -> add dbet * sig * ( 1.0 - mu)
                            double lambda = glob_stats[2 * i + 1] * (1.0 - glob_stats[2 * i]);
                            //printf("1: lambda = %15.10f, l-l0 = %15.10f\n", lambda, lambda - lambda0);
                            S = loci + 3;
                            L = glob_dat[loci];
                            //cout << "1: start = " << S << ", len = " << L <<  endl;
                            sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);

                            // 2 -> add dbet * sig * ( 2.0 - mu)
                            lambda = glob_stats[2 * i + 1] * (2.0 - glob_stats[2 * i]);
                            S = loci + 3 + glob_dat[loci];
                            L = glob_dat[loci + 1];
                            //cout << "2: start = " << S << ", len = " << L <<  endl;
                            sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);

                            loci += 3 + glob_dat[loci] + glob_dat[loci + 1] + glob_dat[loci + 2];
                        }

                        _mm_free(glob_stats);
                        _mm_free(glob_dat);

                        mark2sync.clear();
                        dbet2sync.clear();
                    }
                    else
                    {

                        check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                        check_mpi(MPI_Allreduce(&dEpsSum2[0], &deltaSum2[0], Ntot2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                        check_mpi(MPI_Allreduce(&dEpsSum3[0], &deltaSum3[0], Ntot2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                        check_mpi(MPI_Allreduce(&dEpsSum4[0], &deltaSum4[0], Ntot2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                    }

                    add_arrays(epsilon, tmpEps, deltaSum, Ntot1);
                    add_arrays(epsilon2, tmpEps2, deltaSum2, Ntot2);
                    add_arrays(epsilon3, tmpEps3, deltaSum3, Ntot2);
                    add_arrays(epsilon4, tmpEps4, deltaSum4, Ntot2);

                    double te = MPI_Wtime();
                    tot_sync_ar2 += te - tb;
                    it_sync_ar2 += te - tb;
                    tot_nsync_ar2 += 1;
                    it_nsync_ar2 += 1;
                }
                else
                { // case nranks == 1
                    add_arrays(epsilon, tmpEps, dEpsSum, Ntot1);
                    add_arrays(epsilon2, tmpEps2, dEpsSum2, Ntot2);
                    add_arrays(epsilon3, tmpEps3, dEpsSum3, Ntot2);
                    add_arrays(epsilon4, tmpEps4, dEpsSum4, Ntot2);
                }

                // Do a update currently locally for vi vector
                for (int vi_ind = 0; vi_ind < Ntot1; vi_ind++)
                {
                    vi[vi_ind] = expl((long double)(used_data.alpha * epsilon[vi_ind] - EuMasc));
                }

                for (int vi_ind = 0; vi_ind < Ntot2; vi_ind++)
                {
                    vi2[vi_ind] = expl((long double)(used_data.alpha * epsilon2[vi_ind] - EuMasc));
                    vi3[vi_ind] = expl((long double)(used_data.alpha * epsilon3[vi_ind] - EuMasc));
                    vi4[vi_ind] = expl((long double)(used_data.alpha * epsilon4[vi_ind] - EuMasc));
                }
                double end_sync = MPI_Wtime();
                //printf("INFO   : synchronization time = %8.3f ms\n", (end_sync - beg_sync) * 1000.0);

                // Store epsilon state at last synchronization
                copy_array(tmpEps, epsilon, Ntot1);
                copy_array(tmpEps2, epsilon2, Ntot2);
                copy_array(tmpEps3, epsilon3, Ntot2);
                copy_array(tmpEps4, epsilon4, Ntot2);

                // Reset local sum of delta epsilon
                set_array(dEpsSum, 0.0, Ntot1);
                set_array(dEpsSum2, 0.0, Ntot2);
                set_array(dEpsSum3, 0.0, Ntot2);
                set_array(dEpsSum4, 0.0, Ntot2);

                // Reset cumulated sum of delta betas
                cumSumDeltaBetas = 0.0;
                task_sum_abs_deltabeta = 0.0;

                cumSumDeltaBetas2 = 0.0;
                task_sum_abs_deltabeta2 = 0.0;

                sinceLastSync = 0;
            }

        } // END PROCESSING OF ALL MARKERS

        //PROFILE
        //continue;

        printf("rank %d it %d  beta_squaredNorm[0] = %20.15f, beta_squaredNorm2[0] = %20.15f, beta1_beta2[0] = %20.15f\n",
               rank, iteration, beta_squaredNorm[0], beta_squaredNorm2[0], beta1_beta2[0]);

        //printf("==> after eps sync it %d, rank %d, epsilon[0] = %15.10f %15.10f\n", iteration, rank, epsilon[0], epsilon[Ntot-1]);

        // Transfer global to local
        // ------------------------
        if (nranks > 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            check_mpi(MPI_Allreduce(beta_squaredNorm.data(), sum_beta_squaredNorm.data(), beta_squaredNorm.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(beta_squaredNorm2.data(), sum_beta_squaredNorm2.data(), beta_squaredNorm2.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(beta1_beta2.data(), sum_beta1_beta2.data(), beta_squaredNorm.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

            check_mpi(MPI_Allreduce(cass.data(), sum_cass.data(), cass.size(), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(cass2.data(), sum_cass2.data(), cass2.size(), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

            cass = sum_cass;
            cass2 = sum_cass2;

            beta_squaredNorm = sum_beta_squaredNorm;
            beta_squaredNorm2 = sum_beta_squaredNorm2;
            beta1_beta2 = sum_beta1_beta2;
        }

        if (rank < 0)
        {

            printf("\nINFO   : global cass on iteration %d:\n", iteration);
            for (int i = 0; i < numGroups; i++)
            {
                printf("         Mtot[%3d] = %8d  | cass:", i, MtotGrp[i]);
                for (int ii = 0; ii < K; ii++)
                {
                    printf(" %8d", cass(i, ii));
                }
                printf(" -> sum = %8d\n", cass.row(i).sum());
            }
        }
        // Update global parameters
        // ------------------------
        for (int gg = 0; gg < numGroups; gg++)
        {
            m0[gg] = MtotGrp[gg] - cass(gg, 0);
            m0_2[gg] = MtotGrp[gg] - cass2(gg, 0);

            if (opt.priorsFile != "")
            {
                // Prior parameters for sigmaG
                alpha_sigma = data.priors(gg, 0); //First column is for alpha_sigma, second for beta_sigma hyperparameter
                beta_sigma = data.priors(gg, 1);
            }

            if (opt.dPriorsFile != "")
            {
                // get vector of parameters for the current group
                dirc = data.dPriors.row(gg).transpose().array();
            }
            // 4. Sample sigmaG
            // TODO - Figure out how to sample
            //sigmaG[gg] = dist.inv_gamma_rng((double)(alpha_sigma + 0.5 * m0[gg]), (double)(beta_sigma + 0.5 * double(m0[gg])) * beta_squaredNorm(gg));
            sigmaG[gg] = beta_squaredNorm(gg) + 0.001;
            sigmaG2[gg] = beta_squaredNorm2(gg) + 0.001;
            Rho[gg] = beta1_beta2(gg) / sqrt(sigmaG[gg] * sigmaG2[gg]+ 1);  // + 1 for now

            // 5. Sample prior mixture component probability from Dirichlet distribution
            VectorXd dirin = cass.row(gg).transpose().array().cast<double>() + dirc.array();
            VectorXd dirin2 = cass2.row(gg).transpose().array().cast<double>() + dirc.array();
            pi_L.row(gg) = dist.dirichlet_rng(dirin);
            pi_L2.row(gg) = dist.dirichlet_rng(dirin);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Synchronise the global parameters
        check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Bcast(sigmaG2.data(), sigmaG2.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Bcast(Rho.data(), Rho.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

        check_mpi(MPI_Bcast(pi_L.data(), pi_L.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Bcast(pi_L2.data(), pi_L2.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

        sumSigmaG = sigmaG.sum();   // Keep in memory for safe limit calculations
        sumSigmaG2 = sigmaG2.sum(); // similarly for epoch 2

        //Print results
        if(rank == 0){
            cout << iteration << ". " << m0.sum() <<"; "<< setprecision(7) << mu << "; " <<  used_data.alpha << "; " << sumSigmaG << ": " <<  sumSigmaG2 << "; "<< beta1_beta2(0) << endl;
        }

        double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
        if (rank == 0)
        {
            printf("RESULT : it %4d, rank %4d: proc = %9.3f s, sync = %9.3f (%9.3f + %9.3f), n_sync = %8d (%8d + %8d) (%7.3f / %7.3f), betasq = %20.15f, m0 = %10d\n",
                   iteration, rank, end_it - start_it,
                   it_sync_ar1 + it_sync_ar2, it_sync_ar1, it_sync_ar2,
                   it_nsync_ar1 + it_nsync_ar2, it_nsync_ar1, it_nsync_ar2,
                   (it_sync_ar1) / double(it_nsync_ar1) * 1000.0,
                   (it_sync_ar2) / double(it_nsync_ar2) * 1000.0,
                   beta_squaredNorm.sum(), int(m0.sum()));
            fflush(stdout);
        }

        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        //printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (Beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
        //sigmaE = dist.inv_scaled_chisq_rng(v0E+Ntot,((epsilon).squaredNorm()+v0E*s02E)/(v0E+Ntot));
        //printf("sigmaG = %20.15f, sigmaE = %20.15f, e_sqn = %20.15f\n", sigmaG, sigmaE, e_sqn);
        //printf("it %6d, rank %3d: epsilon[0] = %15.10f, y[0] = %15.10f, m0=%10.1f,  sigE=%15.10f,  sigG=%15.10f [%6d / %6d]\n", iteration, rank, epsilon[0], y[0], m0, sigmaE, sigmaG, markerI[0], markerI[M-1]);

        // Write output files
        // ------------------


        if (iteration % opt.thin == 0)
        {

            if (rank == 0)
            {
                //Save the hyperparameters
                cx = snprintf(buff, LENBUF, "%5d, %22.17f, %22.17f, %22.17f, %22.17f, %7d, %7d, %2d", iteration, mu, sigmaG.sum(), used_data.alpha, sigmaG.sum() / (sigmaG.sum() + PI_squared / (6.0 * used_data.alpha * used_data.alpha)), int(m0.sum()), int(pi_L.rows()), int(pi_L.cols()));
                //	assert(left > 0);
                assert(cx >= 0 && cx < LENBUF); //We also use the condition cx < LENBUF for now

                //   cx = snprintf(&buff, LENBUF - strlen(buff), "%5d, %4d", iteration, (int) sigmaG.size());
                //   assert(cx >= 0 && cx < LENBUF);

                for (int jj = 0; jj < sigmaG.size(); ++jj)
                {
                    //sigmaG(jj) = (double)((int)(sigmaG(jj) * 1E15)) * 1E-15;   // Doesn't work with -O0
                    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %22.17f", sigmaG(jj));
                    assert(cx >= 0 && cx < LENBUF - strlen(buff));
                }
                

                for (int ii = 0; ii < pi_L.rows(); ++ii)
                {
                    for (int kk = 0; kk < pi_L.cols(); ++kk)
                    {
                        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %22.17f", pi_L(ii, kk));
                        assert(cx >= 0 && cx < LENBUF - strlen(buff));
                    }
                }

                //Same for epoch 2
                for (int jj = 0; jj < sigmaG2.size(); ++jj)
                {
                    //sigmaG2(jj) = (double)((int)(sigmaG2(jj) * 1E15)) * 1E-15; // Doesn't work with -O0
                    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %22.17f", sigmaG2(jj));
                    assert(cx >= 0 && cx < LENBUF - strlen(buff));
                }
  
                for (int ii = 0; ii < pi_L2.rows(); ++ii)
                {
                    for (int kk = 0; kk < pi_L2.cols(); ++kk)
                    {
                        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %22.17f", pi_L2(ii, kk));
                        assert(cx >= 0 && cx < LENBUF - strlen(buff));
                    }
                }

                // Rho parameters
                for (int jj = 0; jj < Rho.size(); ++jj)
                {
                    //Rho(jj) = (double)((int)(Rho(jj) * 1E15)) * 1E-15;   // Doesn't work with -O0
                    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %22.17f", Rho(jj));
                    assert(cx >= 0 && cx < LENBUF - strlen(buff));
                }

                cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), "\n");
                assert(cx >= 0 && cx < LENBUF - strlen(buff));

                offset = size_t(n_thinned_saved) * strlen(buff);
                check_mpi(MPI_File_write_at(outfh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);

                //Save the covariates
                if (opt.covariates)
                {
                    cx = snprintf(buff_gamma, LENBUF_gamma, "%5d", iteration);
                    for (int ii = 0; ii < numFixedEffects; ++ii)
                    {
                        cx = snprintf(&buff_gamma[strlen(buff_gamma)], LENBUF_gamma - strlen(buff_gamma), ", %20.17f", gamma(ii));
                        assert(cx > 0);
                    }
                    cx = snprintf(&buff_gamma[strlen(buff_gamma)], LENBUF_gamma - strlen(buff_gamma), "\n");
                    assert(cx > 0);
                    offset = size_t(n_thinned_saved) * strlen(buff_gamma);

                    check_mpi(MPI_File_write_at(gamfh, offset, &buff_gamma, strlen(buff_gamma), MPI_CHAR, &status), __LINE__, __FILE__);
                    //Save the order of the covariates
                    if (iteration > 0 && iteration % opt.save == 0)
                    {
                        offset = 0;
                        check_mpi(MPI_File_write_at(xivfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                        offset = sizeof(uint);
                        check_mpi(MPI_File_write_at(xivfh, offset, &numFixedEffects, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

                        offset = sizeof(uint) + sizeof(uint);
                        check_mpi(MPI_File_write_at(xivfh, offset, xI.data(), numFixedEffects, MPI_INT, &status), __LINE__, __FILE__);
                    }
                }
            }

            // Write iteration number
            if (rank == 0)
            {
                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(betfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpnfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

                // Same for the second epoch
                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(bet2fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

                offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpn2fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }

            offset = sizeof(uint) + sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double)) + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(betfh, offset, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int)) + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh, offset, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            // Epoch 2
            offset = sizeof(uint) + sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double)) + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(bet2fh, offset, Beta2.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int)) + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpn2fh, offset, components2.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            //if (iteration == 0) {
            //    printf("rank %d dumping bet: %15.10f %15.10f\n", rank, Beta[0], Beta[MrankL[rank]-1]);
            //    printf("rank %d dumping cpn: %d %d\n", rank, components[0], components[MrankL[rank]-1]);
            //}

            n_thinned_saved += 1;
        }

        // Dump the epsilon vector and the marker indexing one
        // Note: single line overwritten at each saving iteration
        // .eps format: uint, uint, double[0, N-1] (it, Ntot, [eps])
        // .mrk format: uint, uint, int[0, M-1]    (it, M,    <mrk>)
        // ------------------------------------------------------
        if (iteration > 0 && iteration % opt.save == 0)
        {

            //@@@EO srand(opt.seed + iteration);

            // Each task writes its own rng file
            dist.write_rng_state_to_file(rngfp);
            offset = 0;
            check_mpi(MPI_File_write_at(epsfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps2fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps3fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps4fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            check_mpi(MPI_File_write_at(mrkfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            offset = sizeof(uint);

            check_mpi(MPI_File_write_at(epsfh, offset, &Ntot1, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps2fh, offset, &Ntot2, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps3fh, offset, &Ntot2, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps4fh, offset, &Ntot2, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            check_mpi(MPI_File_write_at(mrkfh, offset, &M, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(xbetfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(xcpnfh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            check_mpi(MPI_File_write_at(xbet2fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(xcpn2fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh, offset, epsilon, Ntot1, MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps2fh, offset, epsilon2, Ntot2, MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps3fh, offset, epsilon3, Ntot2, MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(eps4fh, offset, epsilon4, Ntot2, MPI_DOUBLE, &status), __LINE__, __FILE__);

            check_mpi(MPI_File_write_at(mrkfh, offset, markerI.data(), markerI.size(), MPI_INT, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(xbetfh, offset, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(xcpnfh, offset, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(xbet2fh, offset, Beta2.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            offset = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(xcpn2fh, offset, components2.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            //if (iteration == 0) {
            //    printf("rank %d dumping eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot-1]);
            //}
            //EO: to remove once MPI version fully validated; use the check_epsilon utility to retrieve
            //    the corresponding values from the .eps file
            //    Print only first and last value handled by each task
            //printf("%4d/%4d epsilon[%5d] = %15.10f, epsilon[%5d] = %15.10f\n", iteration, rank, IrankS[rank], epsilon[IrankS[rank]], IrankS[rank]+IrankL[rank]-1, epsilon[IrankS[rank]+IrankL[rank]-1]);

#if 1
            //EO system call to create a tarball of the dump
            //TODO: quite rough, make it more selective...
            //----------------------------------------------
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0)
            {
                time_t now = time(0);
                tm *ltm = localtime(&now);
                int n = 0;
                char targz[LENBUF];

                n = sprintf(targz, "dump_%s_%05d__%4d-%02d-%02d_%02d-%02d-%02d.tgz",
                            opt.mcmcOutNam.c_str(), iteration,
                            1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                            ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
                assert(n > 0);

                //printf("INFO   : will create tarball %s in %s with file listed in %s.\n",
                //       targz, opt.mcmcOutDir.c_str(), lstfp.c_str());

                //std::system(("ls " + opt.mcmcOut + ".*").c_str());
                string cmd = "tar -czf " + opt.mcmcOutDir + "/tarballs/" + targz + " -T " + lstfp + " 2>/dev/null";

                std::system(cmd.c_str());
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        //double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);
 
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Close output files
    check_mpi(MPI_File_close(&outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&bet2fh), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&xbetfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xbet2fh), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&eps2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&eps3fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&eps4fh), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xcpnfh), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&cpn2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xcpn2fh), __LINE__, __FILE__);

    // check_mpi(MPI_File_close(&acufh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&mrkfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&xivfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&gamfh), __LINE__, __FILE__);

    // Release memory
    _mm_free(y);
    _mm_free(epsilon);
    _mm_free(tmpEps);
    _mm_free(deltaEps);
    _mm_free(dEpsSum);
    _mm_free(deltaSum);
    _mm_free(mave);
    _mm_free(mstd);
    _mm_free(USEBED);
    _mm_free(sum_failure);
    _mm_free(sum_failure2);
    _mm_free(N1S);
    _mm_free(N1L);
    _mm_free(I1);
    _mm_free(N2S);
    _mm_free(N2L);
    _mm_free(I2);
    _mm_free(NMS);
    _mm_free(NML);
    _mm_free(IM);

    _mm_free(y2);
    _mm_free(epsilon2);
    _mm_free(epsilon3);
    _mm_free(epsilon4);

    _mm_free(tmpEps2);
    _mm_free(deltaEps2);
    _mm_free(dEpsSum2);
    _mm_free(deltaSum2);

    _mm_free(tmpEps3);
    _mm_free(deltaEps3);
    _mm_free(dEpsSum3);
    _mm_free(deltaSum3);

    _mm_free(tmpEps4);
    _mm_free(deltaEps4);
    _mm_free(dEpsSum4);
    _mm_free(deltaSum4);

    _mm_free(N1S_2);
    _mm_free(N1L_2);
    _mm_free(I1_2);
    _mm_free(N2S_2);
    _mm_free(N2L_2);
    _mm_free(I2_2);
    _mm_free(NMS_2);
    _mm_free(NML_2);
    _mm_free(IM_2);

    _mm_free(vi);
    _mm_free(tmp_vi);
    _mm_free(tmpEps_vi);

    _mm_free(vi2);
    _mm_free(tmp_vi2);
    _mm_free(tmpEps_vi2);

    _mm_free(vi3);
    _mm_free(tmp_vi3);
    _mm_free(tmpEps_vi3);

    _mm_free(vi4);
    _mm_free(tmp_vi4);
    _mm_free(tmpEps_vi4);

    if (opt.sparseSync)
    {
        _mm_free(glob_info);
        _mm_free(tasks_len);
        _mm_free(tasks_dis);
        _mm_free(stats_len);
        _mm_free(stats_dis);
    }

    const auto et3 = std::chrono::high_resolution_clock::now();
    const auto dt3 = et3 - st3;
    const auto du3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();
    if (rank == 0)
        printf("INFO   : rank %4d, time to process the data: %.3f sec, with %.3f (%.3f, %.3f) = %4.1f%% spent on allred (%d, %d)\n",
               rank, du3 / double(1000.0),
               tot_sync_ar1 + tot_sync_ar2, tot_sync_ar1, tot_sync_ar2,
               (tot_sync_ar1 + tot_sync_ar2) / (du3 / double(1000.0)) * 100.0,
               tot_nsync_ar1, tot_nsync_ar2);

    return 0;
}
