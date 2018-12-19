/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include "BayesRRm_eo.h"
#include "data.hpp"
#include "distributions_boost.hpp"
#include "concurrentqueue.h"
#include "options.hpp"
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <numeric>
#include <random>
#include <fcntl.h>


#define handle_error(msg)                               \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)


BayesRRm_eo::BayesRRm_eo(Data &data, Options &opt, const long memPageSize)
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
    cva=(Eigen::Map<Eigen::VectorXf>(ptr,opt.S.size())).cast<double>();
}


BayesRRm_eo::~BayesRRm_eo() 
{
}


inline void scaadd(double* __restrict__ vout, double* __restrict__ vin1, double* __restrict__ vin2, const double dMULT, const int N) {
    
    __assume_aligned(vout, 64);
    __assume_aligned(vin1, 64);
    __assume_aligned(vin2, 64);

    for (int i=0; i<N; i++) {
        vout[i] = vin1[i] + dMULT * vin2[i];
    }
}

inline double dotprod(double* __restrict__ vec1, double* __restrict__ vec2, const int N) {

    __assume_aligned(vec1, 64);
    __assume_aligned(vec2, 64);

    double dp = 0.0d;

    for (int i=0; i<N; i++) {
        dp += vec1[i] * vec2[i]; 
    }

    return dp;
}


int BayesRRm_eo::runGibbs()
{
    int flag;
    moodycamel::ConcurrentQueue<Eigen::VectorXd> q;//lock-free queue
    //const unsigned int M(data.numIncdSnps);
    //const unsigned int M(10000); // EO
    const unsigned int N(data.numKeptInds);
    const int K(int(cva.size()) + 1);
    const int km1 = K - 1;
    VectorXd components(M);
    VectorXf normedSnpData(data.numKeptInds);
    const bool usePreprocessedData = (opt.analysisType == "PPBayes");

    flag = 0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Running Gibbs sampling" << endl;

    // Compute the SNP data length in bytes
    size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;

    omp_set_nested(1); // 1 - enables nested parallelism; 0 - disables nested parallelism.

    //Eigen::initParallel();

#pragma omp parallel shared(flag,q)
    {
#pragma omp sections
        {

            {
                //mean and residual variables
                double mu; // mean or intercept
                double sigmaG; //genetic variance
                double sigmaE; // residuals variance

                //component variables
                VectorXd priorPi(K); // prior probabilities for each component
                VectorXd pi(K); // mixture probabilities
                VectorXd cVa(K); //component-specific variance
                VectorXd logL(K); // log likelihood of component
                VectorXd muk(K); // mean of k-th component marker effect size
                VectorXd denom(km1); // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
                double num;//storing dot product
                int m0; // total num ber of markes in model
                VectorXd v(K); //variable storing the component assignment
                VectorXd cVaI(K);// inverse of the component variances

                //linear model variables
                VectorXd beta(M); // effect sizes
                VectorXd y_tilde(N); // variable containing the adjusted residuals to exclude the effects of a given marker
                VectorXd epsilon(N); // variable containing the residuals

                //sampler variables
                VectorXd sample(2*M+4+N); // varible containg a sambple of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
                std::vector<int> markerI(M);
                std::iota(markerI.begin(), markerI.end(), 0);

                int marker;
                double acum;

                VectorXd y;
                VectorXd Cx;

                priorPi[0] = 0.5;
                cVa[0] = 0;
                cVaI[0] = 0;
                //EO
                muk[0] = 0.0; //muk for the zeroth component=0

                cout << "EO: check this one, order is wrong I guess..." << endl;
                //priorPi.segment(1, km1) = priorPi[0] * cVa.segment(1, km1).array() / cVa.segment(1, km1).sum();

                y_tilde.setZero();
                cVa.segment(1, km1) = cva;
                cVaI.segment(1, km1) = cVa.segment(1, km1).cwiseInverse();
                priorPi.segment(1, km1) = priorPi[0] * cVa.segment(1, km1).array() / cVa.segment(1, km1).sum();

                beta.setZero();
                mu=0;
                sigmaG = dist.beta_rng(1,1);
                //printf("sigmaG = %20.10f\n", sigmaG);

                pi = priorPi;

                components.setZero();
                //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                y = (data.y.cast<double>().array() - data.y.cast<double>().mean());
                y /= sqrt(y.squaredNorm() / ((double)N - 1.0));

                epsilon = (y).array() - mu;
                sigmaE = epsilon.squaredNorm() / N * 0.5;

                // This for MUST NOT BE PARALLELIZED, IT IS THE MARKOV CHAIN
                for (int iteration = 0; iteration < max_iterations; iteration++) {

                    //if (iteration > 0) {
                        //if (iteration % (int)std::ceil(max_iterations / 10) == 0)
                    std::cout << "iteration: " << iteration + 1 <<"\n";
                    //}

                    epsilon = epsilon.array() + mu;//  we substract previous value
                    mu = dist.norm_rng(epsilon.sum() / (double)N, sigmaE / (double)N); //update mu
                    epsilon = epsilon.array() - mu;// we substract again now epsilon =Y-mu-X*beta
                    //printf("mu = %20.10f\n", mu);
                    

                    //EO
                    //std::random_shuffle(markerI.begin(), markerI.end());

                    m0 = 0;
                    v.setZero();

                    // This for should not be parallelized, resulting chain would not be ergodic, still, some times it may converge to the correct solution
                    for (int j = 0; j < M; j++) {
                        marker = markerI[j];

                        if (!usePreprocessedData) {
                            data.getSnpDataFromBedFileUsingMmap_openmp(bedFile, snpLenByt, memPageSize, marker, normedSnpData);
                            //printf("normedSnpData.sum()= %20.10f\n", normedSnpData.sum());
                            //I use a temporal variable to do the cast, there should be better ways to do this.
                            Cx = normedSnpData.cast<double>();
                        }
                        else{
                            Cx = data.mappedZ.col(marker).cast<double>();
                        }

                        //printf("bet = %22.15f\n", beta(marker));

                        y_tilde = epsilon.array() + (Cx * beta(marker)).array(); //now y_tilde = Y-mu-X*beta+ X.col(marker)*beta(marker)_old
                        
                        muk[0] = 0.0;//muk for the zeroth component=0

                        //we compute the denominator in the variance expression to save computations
                        denom = ((double)N-1) + (sigmaE/sigmaG) * cVaI.segment(1, km1).array();
                        //printf("denom[0] = %20.10f\n", denom(0));
                        //printf("denom[1] = %20.10f\n", denom(1));
                        //printf("denom[2] = %20.10f\n", denom(2));

                        //we compute the dot product to save computations
                        num = (Cx.cwiseProduct(y_tilde)).sum();
                        //printf("num = %20.10f\n", num);

                        //muk for the other components is computed according to equations
                        muk.segment(1, km1) = num / denom.array();

                        /*
                        if (marker%1000 == 0)
                            printf("__marker = %8d normedSnpData.sum() = %20.10f\n", marker, normedSnpData.sum());
                        */

                        logL = pi.array().log(); //first component probabilities remain unchanged

                        //update the log likelihood for each component
                        logL.segment(1, km1) = logL.segment(1, km1).array() - 0.5 * ((((sigmaG / sigmaE) * (((double)N-1))) * cVa.segment(1, km1).array() + 1).array().log()) + 0.5 * (muk.segment(1, km1).array() * num) / sigmaE;

                        double p(dist.beta_rng(1,1)); //I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later

                        if (((logL.segment(1, km1).array() - logL[0]).abs().array() > 700).any()) {
                            acum = 0;
                        } else {
                            acum = 1.0 / ((logL.array() - logL[0]).exp().sum());
                        }

                        for (int k = 0; k < K; k++) {
                            if (p <= acum) {
                                //if zeroth component
                                if (k == 0) {
                                    beta(marker) = 0;
                                } else {
                                    beta(marker) = dist.norm_rng(muk[k], sigmaE/denom[k-1]);
                                }
                                v[k] += 1.0;
                                components[marker] = k;
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
                        epsilon = y_tilde - Cx * beta(marker); //now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new
                    }

                    m0 = M - v[0];
                    //cout<< "inv scaled parameters "<< v0G+m0 << "__" << (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
                    printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));

                    //cout<< "num components"<< opt.S.size();
                    //cout<< "\nMixture components : "<<cva[0]<<""<<cva[1]<<" "<<cva[2]<<"\n";
                    sigmaG = dist.inv_scaled_chisq_rng(v0G + m0, (beta.col(0).squaredNorm() * m0 + v0G * s02G) / (v0G + m0));
                    //cout<<"sigmaG: "<<sigmaG<<"\n";
                    //cout<<"y mean: "<<y.mean()<<"\n";
                    // cout<<"y sd: "<< sqrt(y.squaredNorm()/((double)N-1.0))<< "\n";
                    //cout<<"x mean "<<Cx.mean()<<"\n";
                    //cout<<"x sd "<<sqrt(Cx.squaredNorm()/((double)N-1.0))<<"\n";

                    sigmaE = dist.inv_scaled_chisq_rng(v0E + N, ((epsilon).squaredNorm() + v0E * s02E) / (v0E + N));
                    pi = dist.dirichilet_rng(v.array() + 1.0);

                    /*
                    if (iteration >= burn_in) {
                        if (iteration % thinning == 0) {
                            sample << iteration, mu, beta, sigmaE, sigmaG, components, epsilon;
                            q.enqueue(sample);
                        }
                    }
                    */
                }

                //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
                //std::cout << "duration: "<< duration << "s\n";
                flag = 1;

            }

            /*
            //this thread saves in the output file using the lock-free queue
#pragma omp section
            {
                bool queueFull;
                queueFull = 0;
                std::ofstream outFile;
                outFile.open(outputFile);
                VectorXd sampleq(2 * M + 4 + N);
                IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
                outFile<< "iteration," << "mu,";
                for (unsigned int i = 0; i < M; ++i) {
                    outFile << "beta[" << (i+1) << "],";
                }
                outFile << "sigmaE," << "sigmaG,";
                for (unsigned int i = 0; i < M; ++i) {
                    outFile << "comp[" << (i+1) << "],";
                }
                unsigned int i;
                for (i = 0; i < (N-1); ++i) {
                    outFile << "epsilon[" << (i+1) << "],";
                }
                outFile << "epsilon[" << (i+1) << "]";
                outFile << "\n";

                while (!flag) {
                    if (q.try_dequeue(sampleq))
                        outFile << sampleq.transpose().format(CommaInitFmt) << "\n";
                }
            }
            */
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    printf("END RUN after %d milliseconds\n", duration);

    return 0;
}


//
/// With moodycamel queue, more or less Eigen free
//
void BayesRRm_eo::runTest_moody(int numKeptInds, const size_t snpLenByt) {

    cout << "\n### runTest_moody @@@ WITH MOODYCAMEL @@@"  << endl;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //unsigned int M(data.numIncdSnps);
    const unsigned int N(data.numKeptInds);
    double dNm1 = (double)(N - 1);
    double dN   = (double) N;
     
    struct cx {
        double* dat;
        int marker;
    };


	std::vector<int> markerI;
    VectorXd components(M);

    const int K(int(cva.size()) + 1);
    const int km1 = K - 1;

    // Mean and residual variables
    double   mu;              // mean or intercept
    double   sigmaG;          // genetic variance
    double   sigmaE;          // residuals variance

    // Component variables
    VectorXd priorPi(K);      // prior probabilities for each component
    VectorXd pi(K);           // mixture probabilities
    VectorXd muk(K);          // mean of k-th component marker effect size
    VectorXd denom(K-1);      // temporal variable for computing the inflation of the effect variance for a given non-zero component
    //VectorXd logL(K);         // log likelihood of component
    VectorXd cVa(K);          // component-specific variance
    VectorXd cVaI(K);         // inverse of the component variances
    double   num;             // storing dot product
    int      m0;              // total num ber of markes in model
    VectorXd v(K);            // variable storing the component assignment

    // Linear model variables
    MatrixXd beta(M,1);       // effect sizes

    double *y, *y_tilde, *epsilon;
    y =       (double*)_mm_malloc(N * sizeof(double), 64);  if (y == NULL)       exit (1);
    y_tilde = (double*)_mm_malloc(N * sizeof(double), 64);  if (y_tilde == NULL) exit (1);
    epsilon = (double*) _mm_malloc(N * sizeof(double), 64); if (epsilon == NULL) exit (1);

    // Sampler variables
    VectorXd sample(2*M+4+N); // varible containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance

    data.ZPZdiag.resize(data.numIncdSnps);

    for (int i=0; i<M; ++i) {
        markerI.push_back(i);
    }

    priorPi[0] = 0.5d;
    cVa[0]     = 0.0d;
    cVaI[0]    = 0.0d;
    muk[0]     = 0.0d;
    mu         = 0.0d;

    for (int i=0; i<N; ++i) {
        y_tilde[i] = 0.0d;
    }

    cVa.segment(1,km1)     = cva;
    cVaI.segment(1,km1)    = cVa.segment(1,km1).cwiseInverse();
    priorPi.segment(1,km1) = priorPi[0] * cVa.segment(1,km1).array() / cVa.segment(1,km1).sum();
    sigmaG                 = dist.beta_rng(1, 1);
    //printf("sigmaG = %20.10f\n", sigmaG);
    pi                     = priorPi;
    beta.setZero();
    components.setZero();

    double y_mean = 0.0d;
    for (int i=0; i<N; ++i) {
        y[i]    = (double)data.y(i);
        y_mean += y[i];
    }
    y_mean /= N;

    for (int i=0; i<N; ++i) {
        y[i] -= y_mean;
    }

    double y_sqn = 0.0d;
    for (int i=0; i<N; ++i) {
        y_sqn += y[i] * y[i];
    }
    y_sqn = sqrt(y_sqn / dNm1);

    sigmaE = 0.0d;
    for (int i=0; i<N; ++i) {
        y[i]       /= y_sqn;
        epsilon[i]  = y[i] - mu;
        sigmaE     += epsilon[i] * epsilon[i];
    }
    sigmaE = sigmaE / dN * 0.5d;


    // Open BED file for reading
    int fd  = open_bed_file_for_reading(bedFile.c_str());


    // OpenMP settings
    omp_set_nested(1);
    omp_set_dynamic(0);


    //This for MUST NOT BE PARALLELIZED, IT IS THE MARKOV CHAIN

    for (int iteration=0; iteration < max_iterations; iteration++) {
        printf("iteration = %d / %d\n", iteration+1, max_iterations);

        moodycamel::ConcurrentQueue<std::unique_ptr<cx>> q;

        double sum_eps = 0.0d;
        for (int i=0; i<N; ++i) {
            sum_eps += epsilon[i];
        }

        // we substract previous value (?)
        double sum_mu = 0.0d;
        for (int i=0; i<N; ++i) {
            epsilon[i] += mu;
            sum_mu     += epsilon[i];
        }

        // update mu
        mu = dist.norm_rng(sum_mu/dN, sigmaE/dN);

        // We substract again; now epsilon = Y - mu - X*beta
        for (int i=0; i<N; ++i) {
            epsilon[i] -= mu;
        }

        // Shuffle the markers
        //std::random_shuffle(markerI.begin(), markerI.end());
        
        m0 = 0;
        v.setZero();


#pragma omp parallel num_threads(2)
        {
            
            // READING SECTION: multithreaded but ordered -- if needed
            if (omp_get_thread_num() == 0) {
                
                omp_set_num_threads(2);
                
#pragma omp parallel for schedule(static, 1) //ordered // not needed as we shuffle the order of the markers
                for (int j = 0; j < M; j++) {
                    
                    int marker = markerI[j];
                    
                    //EO: keep the queue small
                    while (q.size_approx() > 2) {
                        //spin
                    }

                    std::unique_ptr<cx> cx1 (new cx);
                    cx1->marker = marker;
                    cx1->dat = (double *)_mm_malloc(N * sizeof(double), 64);
                    
                    data.getSnpDataFromBedFileUsingMmap_new(fd, snpLenByt, memPageSize, marker, cx1->dat);
                    
                    q.enqueue(std::move(cx1));
                }

            } else {

                // Single consuming thread!!
                omp_set_num_threads(1);

                double sigE_G  = sigmaE / sigmaG;
                double sigG_E  = sigmaG / sigmaE;
                double i_2sigE = 1.d / (2.d * sigmaE);
                
                for (int j = 0; j < M; j++) {
                    
                    std::unique_ptr<cx> cx2 (new cx);
                    
                    bool found = false;
                    while (found == false) { 
                        found = q.try_dequeue(cx2);
                    }

                    int marker = cx2->marker;
                    double bet =  beta(marker,0);
                    double* Cx = cx2->dat;
                    
                    scaadd(y_tilde, epsilon, Cx, bet, N);
                    
                    //we compute the denominator in the variance expression to save computations
                    //denom = dNm1 + sigE_G * cVaI.segment(1, km1).array();
                    for (int i=1; i<=km1; ++i) {
                        denom(i-1) = dNm1 + sigE_G * cVaI(i);
                    }
                    
                    //we compute the dot product to save computations
                    num = dotprod(Cx, y_tilde, N);
                    
                    //muk for the other components is computed according to equations
                    muk.segment(1, km1) = num / denom.array();
                    
                    VectorXd logL(K);
                    logL = pi.array().log();//first component probabilities remain unchanged
                    
                    // Update the log likelihood for each component
                    logL.segment(1,km1) = logL.segment(1, km1).array()
                        - 0.5d * (sigG_E * dNm1 * cVa.segment(1,km1).array() + 1.0d).array().log() 
                        + muk.segment(1,km1).array() * num * i_2sigE;
                    
                    double p(dist.beta_rng(1,1)); // I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later
                    //printf("p = %22.15f\n", p);
                    
                    double acum;
                    if(((logL.segment(1,km1).array()-logL[0]).abs().array() > 700 ).any() ){
                        acum = 0.0d;
                    } else{
                        acum = 1.0d / ((logL.array()-logL[0]).exp().sum());
                    }
                    
                    for (int k=0; k<K; k++) {
                        if (p <= acum) {
                            if (k==0) {
                                beta(marker,0) = 0.0d;
                            } else {
                                beta(marker,0) = dist.norm_rng(muk[k],sigmaE/denom[k-1]);
                            }
                            v[k] += 1.0d;
                            components[marker] = k;
                            break;
                        } else {
                            //if too big or too small
                            if (((logL.segment(1,km1).array()-logL[k+1]).abs().array() > 700.0d ).any() ){
                                acum += 0.0d;
                            } else{
                                acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum());
                            }
                        }
                    }
                    
                    //epsilon = y_tilde - Cx * beta(marker,0);
                    //now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new
                    
                    bet = beta(marker,0);
                    
                    scaadd(epsilon, y_tilde, Cx, -bet, N);
                    
                    _mm_free(cx2->dat);
                }
                        
                m0 = M - v[0];
                //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
                printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
                
                sigmaG = dist.inv_scaled_chisq_rng(v0G+m0,(beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));
                
                //sigmaE = dist.inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));
                double e_sqn = 0.0d;
                for (int i=0; i<N; ++i) {
                    e_sqn += epsilon[i] * epsilon[i];
                }
                sigmaE = dist.inv_scaled_chisq_rng(v0E+N,(e_sqn + v0E*s02E)/(v0E+N));
            
                pi=dist.dirichilet_rng(v.array() + 1.0);
            }
        }
    }

    _mm_free(y);
    _mm_free(y_tilde);
    _mm_free(epsilon);

    close(fd);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    printf("END RUN after %d milliseconds\n", duration);
}



//
/// No moodycamel queue, more or less Eigen free
//
void BayesRRm_eo::runTest(int numKeptInds, const size_t snpLenByt) {

    cout << "\n### runTest"  << endl;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //unsigned int M(data.numIncdSnps);
    unsigned int N(data.numKeptInds);
    double dNm1 = (double)(N - 1);
    double dN   = (double) N;

	std::vector<int> markerI;
    VectorXd components(M);

    const int K(int(cva.size()) + 1);
    const int km1 = K - 1;

    // Mean and residual variables
    double   mu;              // mean or intercept
    double   sigmaG;          // genetic variance
    double   sigmaE;          // residuals variance

    // Component variables
    VectorXd priorPi(K);      // prior probabilities for each component
    VectorXd pi(K);           // mixture probabilities
    VectorXd muk(K);          // mean of k-th component marker effect size
    VectorXd denom(K-1);      // temporal variable for computing the inflation of the effect variance for a given non-zero component
    //VectorXd logL(K);         // log likelihood of component
    VectorXd cVa(K);          // component-specific variance
    VectorXd cVaI(K);         // inverse of the component variances
    double   num;             // storing dot product
    int      m0;              // total num ber of markes in model
    VectorXd v(K);            // variable storing the component assignment

    // Linear model variables
    MatrixXd beta(M,1);       // effect sizes

    double *y, *y_tilde, *epsilon, *Cx;
    y       = (double*)_mm_malloc(N * sizeof(double), 64); if (y == NULL)       exit (1);
    y_tilde = (double*)_mm_malloc(N * sizeof(double), 64); if (y_tilde == NULL) exit (1);
    epsilon = (double*)_mm_malloc(N * sizeof(double), 64); if (epsilon == NULL) exit (1);
    Cx      = (double*)_mm_malloc(N * sizeof(double), 64); if (Cx == NULL)      exit (1);

    // Sampler variables
    VectorXd sample(2*M+4+N); // varible containg a sample of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance

    data.ZPZdiag.resize(data.numIncdSnps);

    for (int i=0; i<M; ++i) {
        markerI.push_back(i);
    }

    priorPi[0] = 0.5d;
    cVa[0]     = 0.0d;
    cVaI[0]    = 0.0d;
    muk[0]     = 0.0d;
    mu         = 0.0d;

    for (int i=0; i<N; ++i) {
        y_tilde[i] = 0.0d;
    }

    cVa.segment(1,km1)     = cva;
    cVaI.segment(1,km1)    = cVa.segment(1,km1).cwiseInverse();
    priorPi.segment(1,km1) = priorPi[0] * cVa.segment(1,km1).array() / cVa.segment(1,km1).sum();
    sigmaG                 = dist.beta_rng(1, 1);
    //printf("sigmaG = %20.10f\n", sigmaG);
    pi                     = priorPi;
    beta.setZero();
    components.setZero();

    double y_mean = 0.0d;
    for (int i=0; i<N; ++i) {
        y[i]    = (double)data.y(i);
        y_mean += y[i];
    }
    y_mean /= N;
    //printf("mean = %20.15f\n", y_mean);

    for (int i=0; i<N; ++i) {
        y[i] -= y_mean;
    }

    double y_sqn = 0.0d;
    for (int i=0; i<N; ++i) {
        y_sqn += y[i] * y[i];
    }
    y_sqn = sqrt(y_sqn / dNm1);

    sigmaE = 0.0d;
    for (int i=0; i<N; ++i) {
        y[i]       /= y_sqn;
        epsilon[i]  = y[i] - mu;
        sigmaE     += epsilon[i] * epsilon[i];
    }
    sigmaE = sigmaE / dN * 0.5d;
    //printf("sigmaE = %20.10f\n", sigmaE);


    // Open BED file for reading
    int fd  = open_bed_file_for_reading(bedFile.c_str());


    //This for MUST NOT BE PARALLELIZED, IT IS THE MARKOV CHAIN

    for (int iteration=0; iteration < max_iterations; iteration++) {

        printf("iteration = %d / %d\n", iteration+1, max_iterations);

        double sum_eps = 0.0d;
        for (int i=0; i<N; ++i) {
            sum_eps += epsilon[i];
        }
        //printf("mu = %20.10f %20.10f\n", mu, sum_eps);

        //epsilon = epsilon.array() + mu;                                     // we substract previous value
        double sum_mu = 0.0d;
        for (int i=0; i<N; ++i) {
            epsilon[i] += mu;
            sum_mu     += epsilon[i];
        }

        // update mu
        mu = dist.norm_rng(sum_mu/dN, sigmaE/dN);

        // We substract again now epsilon =Y-mu-X*beta
        for (int i=0; i<N; ++i) {
            epsilon[i] -= mu;
        }
        //printf("mu = %20.10f\n", mu);

        
        //std::random_shuffle(markerI.begin(), markerI.end());

        m0 = 0;
        v.setZero();

        for (int j = 0; j < M; j++) {

            int marker = markerI[j];
                    
            //Cx = (double *) _mm_malloc(N * sizeof(double), 64);
            //if (Cx == NULL) exit (1);
            data.getSnpDataFromBedFileUsingMmap_new(fd, snpLenByt, memPageSize, marker, Cx);
            //printf("<<%5d: el %5d of marker %6d = %22.15f\n", j, marker, marker, Cx[marker]);

            double sigE_G  = sigmaE / sigmaG;
            double sigG_E  = sigmaG / sigmaE;
            double i_2sigE = 1.d / (2.d * sigmaE);
            
            double bet =  beta(marker,0);

            scaadd(y_tilde, epsilon, Cx, bet, N);
            
            //we compute the denominator in the variance expression to save computations
            //denom = dNm1 + sigE_G * cVaI.segment(1, km1).array();
            for (int i=1; i<=km1; ++i) {
                denom(i-1) = dNm1 + sigE_G * cVaI(i);
            }

            //we compute the dot product to save computations
            num = dotprod(Cx, y_tilde, N);
            //printf("num = %22.15f\n", num);
            
            //muk for the other components is computed according to equations
            muk.segment(1, km1) = num / denom.array();
            
            VectorXd logL(K);
            logL = pi.array().log();//first component probabilities remain unchanged
            //printf("logL = %22.15f\n", logL[0]);
            
            // Update the log likelihood for each component
            logL.segment(1,km1) = logL.segment(1, km1).array()
                - 0.5d * (sigG_E * dNm1 * cVa.segment(1,km1).array() + 1.0d).array().log() 
                + muk.segment(1,km1).array() * num * i_2sigE;
            
            double p(dist.beta_rng(1,1)); // I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later
            //printf("p = %22.15f\n", p);
            
            double acum;
            if(((logL.segment(1,km1).array()-logL[0]).abs().array() > 700 ).any() ){
                acum = 0.0d;
            } else{
                acum = 1.0d / ((logL.array()-logL[0]).exp().sum());
            }
            //printf("acum = %22.15f\n", acum);            
            
            for (int k=0; k<K; k++) {
                if (p <= acum) {
                    if (k==0) {
                        beta(marker,0) = 0.0d;
                    } else {
                        beta(marker,0) = dist.norm_rng(muk[k],sigmaE/denom[k-1]);
                    }
                    v[k] += 1.0d;
                    components[marker] = k;
                    break;
                } else {
                    //if too big or too small
                    if (((logL.segment(1,km1).array()-logL[k+1]).abs().array() > 700.0d ).any() ){
                        acum += 0.0d;
                    } else{
                        acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum());
                    }
                }
            }
            //printf("acum = %22.15f bet = %22.15f\n", acum, beta(marker,0));
            
            //epsilon = y_tilde - Cx * beta(marker,0);
            //now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new
            bet = beta(marker,0);
            scaadd(epsilon, y_tilde, Cx, -bet, N);
        }

        //printf("v[0] = %22.15f %22.15f\n", v[0], beta.squaredNorm());
        m0 = M - v[0];
        //cout<< "inv scaled parameters "<< v0G+m0 << "__"<< (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0) << endl;
        printf("inv scaled parameters %20.15f __ %20.15f\n", v0G+m0, (beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));

        sigmaG = dist.inv_scaled_chisq_rng(v0G+m0,(beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));

        //sigmaE = dist.inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));
        double e_sqn = 0.0d;
        for (int i=0; i<N; ++i) {
            e_sqn += epsilon[i] * epsilon[i];
        }
        sigmaE = dist.inv_scaled_chisq_rng(v0E+N,(e_sqn + v0E*s02E)/(v0E+N));
        
        pi=dist.dirichilet_rng(v.array() + 1.0);
    }

    _mm_free(y);
    _mm_free(y_tilde);
    _mm_free(epsilon);
    _mm_free(Cx);

    close(fd);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    printf("END RUN after %d milliseconds\n", duration);
}
            
int BayesRRm_eo::open_bed_file_for_reading(const string &bedFile) {
    
    struct stat sb;

    int fd = open(bedFile.c_str(), O_RDONLY);

    if (fd == -1)             handle_error("opening bedFile");
    if (fstat(fd, &sb) == -1) handle_error("fstat");
    if (!S_ISREG(sb.st_mode)) handle_error("Not a regular file");

    return fd;
}
