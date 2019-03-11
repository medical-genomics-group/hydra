/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include "BayesRRmz.hpp"
#include "compression.h"
#include "data.hpp"
#include "distributions_boost.hpp"
#include "options.hpp"
#include "limitsequencegraph.hpp"
#include "parallelgraph.hpp"
#include "samplewriter.h"

#include <chrono>
#include <numeric>
#include <random>
#include <mutex>

BayesRRmz::BayesRRmz(Data &data, Options &opt)
    : m_flowGraph(nullptr)
    , m_data(data)
    , m_opt(opt)
    , m_bedFile(opt.bedFile + ".bed")
    , m_outputFile(opt.mcmcSampleFile)
    , m_seed(opt.seed)
    , m_maxIterations(opt.chainLength)
    , m_burnIn(opt.burnin)
    , m_thinning(opt.thin)
    , m_dist(opt.seed)
    , m_usePreprocessedData(opt.analysisType == "PPBayes")
    , m_showDebug(false)
{
    float* ptr =static_cast<float*>(&opt.S[0]);
    m_cva = (Eigen::Map<Eigen::VectorXf>(ptr, static_cast<long>(opt.S.size()))).cast<double>();

    if (opt.analysisType == "PPAsyncBayes") {
        m_flowGraph.reset(new ParallelGraph(this, opt.numThread));
    } else {
        m_flowGraph.reset(new LimitSequenceGraph(this, opt.numThread));
    }
}

BayesRRmz::~BayesRRmz()
{
}

void BayesRRmz::init(int K, unsigned int markerCount, unsigned int individualCount)
{
    // Component variables
    m_priorPi = VectorXd(K);      // prior probabilities for each component
    m_pi = VectorXd(K);           // mixture probabilities
    m_cVa = VectorXd(K);          // component-specific variance
    m_muk = VectorXd (K);         // mean of k-th component marker effect size
    m_denom = VectorXd(K - 1);    // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
    m_m0 = 0;                     // total num ber of markes in model
    m_v = VectorXd(K);            // variable storing the component assignment
    m_cVaI = VectorXd(K);         // inverse of the component variances

    // Mean and residual variables
    m_mu = 0.0;       // mean or intercept
    m_sigmaG = 0.0;   // genetic variance
    m_sigmaE = 0.0;   // residuals variance

    // Linear model variables
    m_beta = VectorXd(markerCount);           // effect sizes
    m_y_tilde = VectorXd(individualCount);    // variable containing the adjusted residuals to exclude the effects of a given marker
    m_epsilon = VectorXd(individualCount);    // variable containing the residuals
    m_async_epsilon = VectorXd(individualCount);

    m_y = VectorXd();
    //Cx = VectorXd();

    // Init the working variables
    const int km1 = K - 1;
    m_cVa[0] = 0;
    m_cVa.segment(1, km1) = m_cva;
    m_priorPi[0] = 0.5;
    m_priorPi.segment(1, km1) = m_priorPi[0] * m_cVa.segment(1, km1).array() / m_cVa.segment(1, km1).sum();
    m_y_tilde.setZero();

    m_cVaI[0] = 0;
    m_cVaI.segment(1, km1) = m_cVa.segment(1, km1).cwiseInverse();
    m_beta.setZero();
    m_sigmaG = m_dist.beta_rng(1,1);

    m_pi = m_priorPi;

    m_y = (m_data.y.cast<double>().array() - m_data.y.cast<double>().mean());
    m_y /= sqrt(m_y.squaredNorm() / (double(individualCount - 1)));

    m_epsilon = (m_y).array() - m_mu;
    m_sigmaE = m_epsilon.squaredNorm() / individualCount * 0.5;
}

int BayesRRmz::runGibbs()
{
    const unsigned int M(m_data.numSnps);
    const unsigned int N(m_data.numInds);
    const int K(int(m_cva.size()) + 1);

    init(K, M, N);

    SampleWriter writer;
    writer.setFileName(m_outputFile);
    writer.setMarkerCount(M);
    writer.setIndividualCount(N);
    writer.open();

    // Sampler variables
    VectorXd sample(2*M+4+N); // varible containg a sambple of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
    std::vector<unsigned int> markerI(M);
    std::iota(markerI.begin(), markerI.end(), 0);

    std::cout << "Running Gibbs sampling" << endl;
    const auto t1 = std::chrono::high_resolution_clock::now();

    // This for MUST NOT BE PARALLELIZED, IT IS THE MARKOV CHAIN
    m_components.resize(M);
    m_components.setZero();

    long meanIterationTime = 0;
    long meanFlowGraphIterationTime = 0;

    for (unsigned int iteration = 0; iteration < m_maxIterations; iteration++) {
        // Output progress
        const auto startTime = std::chrono::high_resolution_clock::now();
        //if (iteration > 0 && iteration % unsigned(std::ceil(max_iterations / 10)) == 0)
            std::cout << "iteration " << iteration << ": ";

         m_epsilon = m_epsilon.array() + m_mu;//  we substract previous value
         m_mu = m_dist.norm_rng(m_epsilon.sum() / (double)N, m_sigmaE / (double)N); //update mu
         m_epsilon = m_epsilon.array() - m_mu;// we substract again now epsilon =Y-mu-X*beta

         std::memcpy(m_async_epsilon.data(), m_epsilon.data(), static_cast<size_t>(m_epsilon.size()) * sizeof(double));

        std::random_shuffle(markerI.begin(), markerI.end());

        m_m0 = 0;
        m_v.setZero();

        // This for should not be parallelized, resulting chain would not be ergodic, still, some times it may converge to the correct solution.
        // The flow graph is constructed to allow the data to be decompressed in parallel for enforce sequential processing of each column
        // in turn. HOwever, within each column we make use of Intel TBB's parallel_for to parallelise the operations on the large vectors
        // of data.
        const auto flowGraphStartTime = std::chrono::high_resolution_clock::now();
        m_flowGraph->exec(N, M, markerI);
        const auto flowGraphEndTime = std::chrono::high_resolution_clock::now();

        m_m0 = int(M) - int(m_v[0]);
        m_sigmaG = m_dist.inv_scaled_chisq_rng(m_v0G + m_m0, (m_betasqn * m_m0 + m_v0G * m_s02G) / (m_v0G + m_m0));

        if (m_showDebug)
            printDebugInfo();
        const double epsilonSqNorm = m_epsilon.squaredNorm();
        m_sigmaE = m_dist.inv_scaled_chisq_rng(m_v0E + N, (epsilonSqNorm + m_v0E * m_s02E) / (m_v0E + N));
        m_pi = m_dist.dirichilet_rng(m_v.array() + 1.0);

        if (iteration >= m_burnIn && iteration % m_thinning == 0) {
            sample << iteration, m_mu, m_beta, m_sigmaE, m_sigmaG, m_components, m_epsilon;
            writer.write(sample);
        }

        const auto endTime = std::chrono::high_resolution_clock::now();
        const auto iterationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        const auto flowGraphDuration = std::chrono::duration_cast<std::chrono::milliseconds>(flowGraphEndTime - flowGraphStartTime).count();
        std::cout << static_cast<double>(iterationDuration) / 1000.0 << "s (" << static_cast<double>(flowGraphDuration) / 1000.0 << "s)" << std::endl;
        meanIterationTime += iterationDuration;
        meanFlowGraphIterationTime += flowGraphDuration;
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "duration: " << duration << "s" << std::endl;
    const double meanIterationDuration = (static_cast<double>(meanIterationTime) / 1000.0) / static_cast<double>(m_maxIterations);
    const double meanFlowGraphIterationDuration = (static_cast<double>(meanFlowGraphIterationTime) / 1000.0) / static_cast<double>(m_maxIterations);
    std::cout << "mean iteration duration: " << meanIterationDuration  << "s" << std::endl
              << "mean flowgraph duration: " << meanFlowGraphIterationDuration << "s" << std::endl;

    return 0;
}

void BayesRRmz::processColumn(unsigned int marker, const Map<VectorXd> &Cx)
{
    const unsigned int N(m_data.numInds);
    const double NM1 = double(N - 1);
    const int K(int(m_cva.size()) + 1);
    const int km1 = K - 1;
    double acum = 0.0;
    double beta_old;

    beta_old = m_beta(marker);

    // Now y_tilde = Y-mu - X * beta + X.col(marker) * beta(marker)_old
    if (m_components(marker) != 0.0) {
        m_y_tilde = m_epsilon + beta_old * Cx;
    } else {
        m_y_tilde = m_epsilon;
    }
    // muk for the zeroth component=0
    m_muk[0] = 0.0;

    // We compute the denominator in the variance expression to save computations
    const double sigmaEOverSigmaG = m_sigmaE / m_sigmaG;
    m_denom = NM1 + sigmaEOverSigmaG * m_cVaI.segment(1, km1).array();

    // We compute the dot product to save computations
    // We compute the dot product to save computations
      const double num = Cx.dot(m_y_tilde);
    // muk for the other components is computed according to equaitons
    m_muk.segment(1, km1) = num / m_denom.array();

    // Update the log likelihood for each component
    VectorXd logL(K);
    const double logLScale = m_sigmaG / m_sigmaE * NM1;
    logL = m_pi.array().log(); // First component probabilities remain unchanged
    logL.segment(1, km1) = logL.segment(1, km1).array()
            - 0.5 * ((logLScale * m_cVa.segment(1, km1).array() + 1).array().log())
            + 0.5 * (m_muk.segment(1, km1).array() * num) / m_sigmaE;

    double p(m_dist.unif_rng());

    if (((logL.segment(1, km1).array() - logL[0]).abs().array() > 700).any()) {
        acum = 0;
    } else {
        acum = 1.0 / ((logL.array() - logL[0]).exp().sum());
    }

    for (int k = 0; k < K; k++) {
        if (p <= acum) {
            //if zeroth component
            if (k == 0) {
                m_beta(marker) = 0;
            } else {
                m_beta(marker) = m_dist.norm_rng(m_muk[k], m_sigmaE/m_denom[k-1]);
            }
            m_v[k] += 1.0;
            m_components[marker] = k;
            break;
        } else {
            //if too big or too small
            if (((logL.segment(1, km1).array() - logL[k+1]).abs().array() > 700).any()) {
                acum += 0;
            } else {
                acum += 1.0 / ((logL.array() - logL[k+1]).exp().sum());
            }
        }
    }
    m_betasqn += m_beta(marker) * m_beta(marker) - beta_old * beta_old;

    if (m_components(marker) != 0.0) {
        m_epsilon = m_y_tilde - m_beta(marker) * Cx;
    } else {
        m_epsilon = m_y_tilde;
    }
    // Now epsilon contains Y-mu - X*beta + X.col(marker) * beta(marker)_old - X.col(marker) * beta(marker)_new
}

std::tuple<double, double> BayesRRmz::processColumnAsync(unsigned int marker, const Map<VectorXd> &Cx)
{
    // Lock and take local copies of needed variabls
    // [*] m_beta(marker) rwr - used, updated, then used - per column, could take a copy and update at end
    // [*] m_betasqn w - updated here, used in BayezRRmz::runGibbs
    // [*] m_components(marker) rwr - used, updated, then used - per column, could take a copy and update at end
    // [*] m_epsilon rw - used throughout, then updated, used in BayezRRmz::runGibbs
    // [*] m_v w - updated here, used in BayezRRmz::runGibbs

    // [*] m_dist r - the engine is not thread safe

    // Temporaries
    // - cost of locking vs allocating per iteration?
    // [*] m_denom wr - computed from m_cVaI
    // [*] m_muk wr - computed from m_cVaI

    // m_data.numInds r - could be a member?
    // m_cva.size() r - could be a member?
    // m_sigmaE r - calculated in BayesRRmz::init
    // m_sigmaG r - calculated in BayesRRmz::init, updated in BayezRRmz::runGibbs
    // m_pi r - calculated in BayesRRmz::init, updated in BayezRRmz::runGibbs
    // m_cVa r - calculated in BayesRRmz::init
    // m_cVaI r - calculated in BayesRRmz::init

    double beta = 0;
    double component = 0;
    VectorXd y_tilde(m_data.numInds);
    VectorXd epsilon(m_data.numInds);

    {
        // Use a shared lock to allow multiple threads to read updates
        std::shared_lock lock(m_mutex);

        // std::memcpy is faster than epsilon = m_epsilon which compiles down to a loop over pairs of
        // doubles and uses _mm_load_pd(source) SIMD intrinsics. Just be careful if we change the type
        // contained in the vector back to floats.
        std::memcpy(y_tilde.data(), m_async_epsilon.data(), static_cast<size_t>(epsilon.size()) * sizeof(double));
        beta = m_beta(marker);
        component = m_components(marker);
    }
    const double beta_old = beta;

    // Note that we assign y_tilde = m_epsilon above with the memcpy.
    // Now y_tilde = Y-mu - X * beta + X.col(marker) * beta(marker)_old
    if (component != 0.0)
        y_tilde += beta_old * Cx;

    // We compute the dot product to save computations
    const double num = Cx.dot(y_tilde);

    // Do work

    // We compute the denominator in the variance expression to save computations
    const double sigmaEOverSigmaG = m_sigmaE / m_sigmaG;

    const double NM1 = double(m_data.numInds - 1);
    const int K(int(m_cva.size()) + 1);
    const int km1 = K - 1;
    VectorXd denom = NM1 + sigmaEOverSigmaG * m_cVaI.segment(1, km1).array();

    // muk for the zeroth component=0
    VectorXd muk(K);
    muk[0] = 0.0;
    // muk for the other components is computed according to equaitons
    muk.segment(1, km1) = num / denom.array();

    // Update the log likelihood for each component
    VectorXd logL(K);
    const double logLScale = m_sigmaG / m_sigmaE * NM1;
    logL = m_pi.array().log(); // First component probabilities remain unchanged
    logL.segment(1, km1) = logL.segment(1, km1).array()
            - 0.5 * ((logLScale * m_cVa.segment(1, km1).array() + 1).array().log())
            + 0.5 * (muk.segment(1, km1).array() * num) / m_sigmaE;


    double acum = 0.0;
    if (((logL.segment(1, km1).array() - logL[0]).abs().array() > 700).any()) {
        acum = 0;
    } else {
        acum = 1.0 / ((logL.array() - logL[0]).exp().sum());
    }

    double p = 0;
    std::vector<double> randomNumbers(static_cast<std::vector<double>::size_type>(K), 0);
    {
        // Generate all the numbers we are going to need in one go.
        // Use a unique lock to ensure only one thread can use the random number engine
        // at a time.
        std::unique_lock lock(m_rngMutex);
        p = m_dist.unif_rng();

        auto beginItr = randomNumbers.begin();
        std::advance(beginItr, 1);
        std::generate(beginItr, randomNumbers.end(), [&, k = 0] () mutable {
            ++k;
            return m_dist.norm_rng(muk[k], m_sigmaE/denom[k-1]);
        });
    }
    VectorXd v = VectorXd(K);
    v.setZero();
    for (int k = 0; k < K; k++) {
        if (p <= acum) {
            //if zeroth component
            if (k == 0) {
                beta = 0;
            } else {
                beta = randomNumbers.at(static_cast<std::vector<double>::size_type>(k));
            }
            v[k] += 1.0;
            component = k;
            break;
        } else {
            //if too big or too small
            if (((logL.segment(1, km1).array() - logL[k+1]).abs().array() > 700).any()) {
                acum += 0;
            } else {
                acum += 1.0 / ((logL.array() - logL[k+1]).exp().sum());
            }
        }
    }

    // Only update m_epsilon if required
    const bool skipUpdate = beta_old == 0.0 && beta == 0.0;

    // Update our local copy of epsilon to minimise the amount of time we need to hold the unique lock for.
    if (!skipUpdate) {
        y_tilde -= beta * Cx;
    }
    // Now y_tilde contains Y-mu - X*beta + X.col(marker) * beta(marker)_old - X.col(marker) * beta(marker)_new

    // Lock to write updates (at end, or perhaps as updates are computed)
    {
        // Use a unique lock to ensure only one thread can write updates
        std::unique_lock lock(m_mutex);
        if (!skipUpdate) {
            std::memcpy(m_async_epsilon.data(), y_tilde.data(), static_cast<size_t>(y_tilde.size()) * sizeof(double));
            m_betasqn += beta * beta - beta_old * beta_old;
        }
        m_v += v;
    }

    // These updates do not need to be atomic
    m_beta(marker) = beta;
    m_components(marker) = component;

    return {beta_old, beta};
}

void BayesRRmz::updateGlobal(double beta_old, double beta, const Map<VectorXd> &Cx)
{
    // No mutex required here whilst m_globalComputeNode uses the serial policy
    m_epsilon -= Cx * (beta - beta_old);
}

void BayesRRmz::printDebugInfo() const
{
    const unsigned int N(m_data.numInds);
    cout << "inv scaled parameters " << m_v0G + m_m0 << "__" << (m_beta.squaredNorm() * m_m0 + m_v0G * m_s02G) / (m_v0G + m_m0);
    cout << "num components: " << m_opt.S.size();
    cout << "\nMixture components: " << m_cva[0] << " " << m_cva[1] << " " << m_cva[2] << "\n";
    cout << "sigmaG: " << m_sigmaG << "\n";
    cout << "y mean: " << m_y.mean() << "\n";
    cout << "y sd: " << sqrt(m_y.squaredNorm() / (double(N - 1))) << "\n";
//    cout << "x mean " << Cx.mean() << "\n";
//    cout << "x sd " << sqrt(Cx.squaredNorm() / (double(N - 1))) << "\n";
}
