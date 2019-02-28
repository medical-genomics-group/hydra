#ifndef SRC_BAYESRRMZ_H_
#define SRC_BAYESRRMZ_H_

#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"


#include <Eigen/Eigen>
#include <memory>
#include <shared_mutex>

class AnalysisGraph;

class BayesRRmz
{
    friend class LimitSequenceGraph;
    std::unique_ptr<AnalysisGraph> m_flowGraph;
    Data                &m_data; // data matrices
    Options             &m_opt;
    const string        m_bedFile; // bed file
    const string        m_outputFile;
    const unsigned int  m_seed;
    const unsigned int  m_maxIterations;
    const unsigned int  m_burnIn;
    const unsigned int  m_thinning;
    const double        m_sigma0  = 0.0001;
    const double        m_v0E     = 0.0001;
    const double        m_s02E    = 0.0001;
    const double        m_v0G     = 0.0001;
    const double        m_s02G    = 0.0001;
    Eigen::VectorXd     m_cva;
    Distributions_boost m_dist;
    bool m_usePreprocessedData;
    bool m_showDebug;

    // Component variables
    VectorXd m_priorPi;   // prior probabilities for each component
    VectorXd m_pi;        // mixture probabilities
    VectorXd m_cVa;       // component-specific variance
    VectorXd m_muk;       // mean of k-th component marker effect size
    VectorXd m_denom;     // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
    int m_m0;             // total number of markers in model
    VectorXd m_v;         // variable storing the component assignment
    VectorXd m_cVaI;      // inverse of the component variances
    VectorXd Cx;
    //unsigned char *decompressBuffer = nullptr;
    //unsigned int colSize ;
    // Mean and residual variables
    double m_mu;          // mean or intercept
    double m_sigmaG;      // genetic variance
    double m_sigmaE;      // residuals variance

    // Linear model variables
    VectorXd m_beta;       // effect sizes
    VectorXd m_y_tilde;    // variable containing the adjusted residuals to exclude the effects of a given marker
    VectorXd m_epsilon;    // variable containing the residuals
    double m_betasqn = 0.0;

    VectorXd m_y;
    VectorXd m_components;

public:
    BayesRRmz(Data &m_data, Options &m_opt);
    virtual ~BayesRRmz();
    int runGibbs(); // where we run Gibbs sampling over the parametrised model
    void processColumn(unsigned int marker,  const Map<VectorXf> &Cx);

    void setDebugEnabled(bool enabled) { m_showDebug = enabled; }
    bool isDebugEnabled() const { return m_showDebug; }

private:
    void init(int K, unsigned int markerCount, unsigned int individualCount);
    void printDebugInfo() const;
};

#endif /* SRC_BAYESRRM_H_ */
