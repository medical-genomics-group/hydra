/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include "BayesRRpp.h"
#include "data.hpp"
#include "distributions_boost.hpp"
#include "concurrentqueue.h"

BayesRRpp::BayesRRpp(Data &data,Options &opt, const long memPageSize)
    : seed(opt.seed)
    , data(data)
    , opt(opt)
    , memPageSize(memPageSize)
    , max_iterations(opt.chainLength)
    , thinning(opt.thin)
    , burn_in(opt.burnin)
    , outputFile(opt.mcmcSampleFile)
    , bedFile(opt.bedFile + ".bed")
    , dist(opt.seed)
{
    float* ptr =(float*)&opt.S[0];
    cva = (Eigen::Map<Eigen::VectorXf>(ptr, opt.S.size())).cast<double>();
}

BayesRRpp::~BayesRRpp()
{
}

int BayesRRpp::runGibbs()
{
    int flag;
    moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
    unsigned int M(data.numIncdSnps);
    unsigned int N(data.numKeptInds);
    std::vector<int> markerI;
    const int K(int(cva.size()) + 1);
    VectorXd components(M);
    VectorXf normedSnpData(data.numKeptInds);

    flag = 0;
    std::cout << "Running Gibbs sampling" << std::endl;
    std::cout << "Performing " << max_iterations << " iterations" << std::endl;

    // Compute the SNP data length in bytes
    size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;
    omp_set_nested(1); // 1 - enables nested parallelism; 0 - disables nested parallelism.

#pragma omp parallel  shared(flag,q,M,N)
    {
#pragma omp sections
        {
            {
                // Mean and residual variables
                double mu; // mean or intercept
                double sigmaG; //genetic variance
                double sigmaE; // residuals variance

                // Component variables
                VectorXd priorPi(K); // prior probabilities for each component
                VectorXd pi(K); // mixture probabilities
                VectorXd cVa(K); //component-specific variance
                VectorXd logL(K); // log likelihood of component
                VectorXd muk(K); // mean of k-th component marker effect size
                VectorXd denom(K-1); // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
                double num;//storing dot product
                int m0; // total num ber of markes in model
                VectorXd v(K); //variable storing the component assignment
                VectorXd cVaI(K);// inverse of the component variances

                // Linear model variables
                MatrixXd beta(M,1); // effect sizes
                VectorXd y_tilde(N); // variable containing the adjusted residuals to exclude the effects of a given marker
                VectorXd epsilon(N); // variable containing the residuals

                // Sampler variables
                VectorXd sample(2*M+4+N); // varible containg a sambple of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
                std::vector<int> markerI(M);
                std::iota(markerI.begin(), markerI.end(), 0);

                int marker;
                double acum;

                VectorXd y;
                VectorXd Cx;
                priorPi[0] = 0.5;

                // Just do this once
                const int km1 = K - 1;

                cVa.segment(1, km1) = cva;
                priorPi.segment(1, km1) = priorPi[0] * cVa.segment(1, km1).array() / cVa.segment(1, km1).sum();
                y_tilde.setZero();
                cVa[0] = 0;

                cVaI[0] = 0;
                cVaI.segment(1, km1) = cVa.segment(1, km1).cwiseInverse();

                beta.setZero();
                mu = 0;
                sigmaG = dist.beta_rng(1, 1);
                pi = priorPi;

                components.setZero();
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

                y = (data.y.cast<double>().array() - data.y.cast<double>().mean());
                y /= sqrt(y.squaredNorm() / (double(N)-1.0));

                epsilon = y;
                sigmaE = epsilon.squaredNorm() / N * 0.5;

                // THIS FOR MUST NOT BE PARALLELIZED, its the markov chain
                for (int iteration = 0; iteration < max_iterations; iteration++) {
                    if (iteration > 0) {
                        if (iteration % int(std::ceil(max_iterations / 10)) == 0)
                            std::cout << "iteration: " << iteration << std::endl;
                    }

                    epsilon = epsilon.array() + mu; // We substract previous value
                    mu = dist.norm_rng(epsilon.sum() / double(N), sigmaE / double(N)); // Update mu
                    epsilon = epsilon.array() - mu;// we substract again now epsilon =Y-mu-X*beta

                    std::random_shuffle(markerI.begin(), markerI.end());

                    m0 = 0;
                    v.setZero();

                    // THIS FOR should not be parallelized, resulting chain is not ergodic, but it may converge to result sometimes if parallelized
                    for (int j = 0; j < M; j++ ) {
                        marker = markerI[j];
                        // TODO mappedZ column should be scaled, that is, column = (column - mean(column)) / sd(column), maybe that can be done in preprocess
                        Cx = data.mappedZ.col(marker).cast<double>();

                        y_tilde = epsilon.array()+(Cx * beta(marker, 0)).array(); // Now y_tilde = Y-mu-X*beta+ X.col(marker)*beta(marker)_old

                        muk[0] = 0.0; // muk for the zeroth component=0

                        // We compute the denominator in the variance expression to save computations
                        denom = (double(N) - 1.0) + (sigmaE / sigmaG) * cVaI.segment(1, km1).array();

                        // We compute the dot product to save computations
                        num = (Cx.cwiseProduct(y_tilde)).sum();

                        // muk for the other components is computed according to equaitons
                        muk.segment(1, km1) = num / denom.array();

                        logL = pi.array().log(); // First component probabilities remain unchanged

                        // Update the log likelihood for each component
                        logL.segment(1, km1) = logL.segment(1, km1).array() - 0.5 * ((((sigmaG / sigmaE) * ((double(N) - 1.0))) * cVa.segment(1, km1).array() + 1).array().log()) + 0.5 * (muk.segment(1, km1).array() * num) / sigmaE;
                        double p(dist.beta_rng(1, 1)); // I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later

                        if (((logL.segment(1, km1).array() - logL[0]).abs().array() > 700 ).any()) {
                            acum = 0.0;
                        } else {
                            acum = 1.0 / ((logL.array() - logL[0]).exp().sum());
                        }

                        for (int k = 0; k < K; k++) {
                            if (p <= acum) {
                                // If zeroth component
                                if (k == 0) {
                                    beta(marker, 0) = 0;
                                } else {
                                    beta(marker,0 ) = dist.norm_rng(muk[k], sigmaE / denom[k-1]);
                                }
                                v[k] += 1.0;
                                components[marker] = k;
                                break;
                            } else {
                                // If too big or too small
                                if (((logL.segment(1, km1).array() - logL[k + 1]).abs().array() > 700 ).any()) {
                                    acum += 0;
                                } else {
                                    acum += 1.0 / ((logL.array() - logL[k + 1]).exp().sum());
                                }
                            }
                        }
                        epsilon = y_tilde - (Cx) * beta(marker, 0); // Now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new
                    }

                    m0 = int(M) - int(v[0]);
                    //cout << "inv scaled parameters " << v0G + m0 << "__" << (beta.squaredNorm() * m0 + v0G * s02G) / (v0G + m0);
                    sigmaG = dist.inv_scaled_chisq_rng(v0G + m0, (beta.squaredNorm() * m0 + v0G * s02G) / (v0G + m0));
                    sigmaE = dist.inv_scaled_chisq_rng(v0E + N, ((epsilon).squaredNorm() + v0E * s02E) / (v0E + N));

                    // cout << "sigmaE " << sigmaE << "\n";
                    // cout << "sigmaG " << sigmaG << "\n";
                    // cout << "mu " << mu << "\n";

                    pi = dist.dirichilet_rng(v.array() + 1.0);

                    if (iteration >= burn_in) {
                        if (iteration % thinning == 0) {
                            sample << iteration, mu, beta, sigmaE, sigmaG, components, epsilon;
                            q.enqueue(sample);
                        }
                    }
                }

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
                std::cout << "duration: "<< duration << "s\n";
                flag = 1;
            }
#pragma omp section
            {
                bool queueFull;
                unsigned int i;
                queueFull = 0;
                std::ofstream outFile;
                outFile.open(outputFile);
                VectorXd sampleq(2 * M + 4 + N);
                IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
                outFile << "iteration," << "mu,";
                for (unsigned int i = 0; i < M; ++i) {
                    outFile << "beta[" << (i + 1) << "],";
                }
                outFile << "sigmaE," << "sigmaG,";
                for (unsigned int i = 0; i < M; ++i) {
                    outFile << "comp[" << (i+1) << "],";
                }

                outFile << "epsilon[" << (i+1) << "]";
                outFile<<"\n";

                while(!flag) {
                    if (q.try_dequeue(sampleq))
                        outFile << sampleq.transpose().format(CommaInitFmt) << "\n";
                }
            }
        }
    }

    return 0;
}
