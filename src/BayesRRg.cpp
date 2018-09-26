/*
 * BayesRRg.cpp
 *
 *  Created on: 25 Sep 2018
 *      Author: admin
 */
#include "data.hpp"
#include <Eigen/Core>
#include <random>
#include "distributions_boost.hpp"
#include "concurrentqueue.h"
#include "options.hpp"
#include "BayesRRg.h"

BayesRRg::BayesRRg(Data &data,Options &opt, const long memPageSize):seed(opt.seed),
																	data(data),
																	opt(opt),
																	memPageSize(memPageSize),
																	max_iterations(opt.chainLength),
																	thinning(opt.thin),
																	burn_in(opt.burnin),
																	outputFile(opt.mcmcSampleFile),
																	bedFile(opt.bedFile + ".bed"),
																	dist(opt.seed),
																	cva(opt.mS) ,
																	numberGroups(opt.numGroups){
	// TODO Auto-generated constructor stub

}

BayesRRg::~BayesRRg() {
	// TODO Auto-generated destructor stub
}
//TODO adapt this to command line
void BayesRRg::runGibbs() {
	int flag;
	  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
	  flag=0;
	  int N(data.numKeptInds);
	  int M(data.numIncdSnps);
	  VectorXd components(M);

	  int K(cva.cols()+1);
	  ////////////validate inputs

	  if(max_iterations < burn_in || max_iterations<1 || burn_in<1) //validations related to mcmc burnin and iterations
	  {
	    std::cout<<"error: burn_in has to be a positive integer and smaller than the maximum number of iterations ";
	    return;
	  }
	  if(sigma0 < 0 || v0E < 0 || s02E < 0 || v0G < 0||  s02G < 0 )//validations related to hyperparameters
	  {
	    std::cout<<"error: hyper parameters have to be positive";
	    //return;
	  }
	  if((cva.array()==0).any() )//validations related to hyperparameters
	  {
	    std::cout<<"error: the zero component is already included in the model by default";
	    //return;
	  }
	  if((cva.array()<0).any() )//validations related to hyperparameters
	  {
	    std::cout<<"error: the variance of the components should be positive";
	    //return;
	  }
	  /////end of declarations//////

	  omp_set_nested(1);
	  Eigen::initParallel();
	  double sum_beta_sqr;


	#pragma omp parallel  shared(flag,q,M,N)
	{
	#pragma omp sections
	{

	  {

	    //mean and residual variables
	    double mu; // mean or intercept
	    double sigmaG; //genetic variance
	    double sigmaE; // residuals variance
	    VectorXd sigmaGG(numberGroups);

	    //component variables
	    MatrixXd priorPi(numberGroups,K); // prior probabilities for each component
	    MatrixXd pi(numberGroups,K); // mixture probabilities
	    VectorXd cVa(K); //component-specific variance
	    VectorXd logL(K); // log likelihood of component
	    VectorXd muk(K); // mean of k-th component marker effect size
	    VectorXd denom(K-1); // temporal variable for computing the inflation of the effect variance for a given non-zero componnet
	    int m0; // total num ber of markes in model
	    MatrixXd v(numberGroups,K); //variable storing the component assignment
	    VectorXd cVaI(K);// inverse of the component variances

	    //linear model variables
	    MatrixXd beta(M,1); // effect sizes
	    VectorXd y_tilde(N); // variable containing the adjusted residuals to exclude the effects of a given marker
	    VectorXd epsilon(N); // variable containing the residuals

	    //sampler variables
	    VectorXd sample(2*M+3+numberGroups+N); // varible containg a sambple of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
	    std::vector<int> markerI;
	    for (int i=0; i<M; ++i) {
	      markerI.push_back(i);
	    }
	    VectorXd xsquared(M);
	    double num;

	    int marker;
	    double acum;
	    VectorXd betaAcum(numberGroups);


        VectorXd Y;
        VectorXd cX;


	    for(int i=0; i < numberGroups; i++)
	    {
	      priorPi.row(i)(0)=0;
	      priorPi.row(i).segment(1,(K-1))=priorPi.row(i)(0)*cVa.segment(1,(K-1)).segment(1,(K-1)).array()/cVa.segment(1,(K-1)).segment(1,(K-1)).sum();
	    }

	    y_tilde.setZero();
	    //cVa[0] = 0;
	    //cVa.segment(1,(K-1))=cva;

	    //cVaI[0] = 0;
	    //cVaI.segment(1,(K-1))=cVa.segment(1,(K-1)).cwiseInverse();
	    //beta=beta.setRandom();

	    //beta=(beta.array().abs() > 1e-6  ).select(beta, MatrixXd::Zero(M,1));
	    beta.setZero();

	    //mu=norm_rng(0,1);
	    mu=0;


	    betaAcum.setZero();
	    // sigmaG=(1*cVa).sum()/M;
	    for(int i=0; i<groups;i++)
	      sigmaGG[i]=beta_rng(1,1);

	    pi=priorPi;

	    components.setZero();
	    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	    Y=(data.y.cast<double>().array()-data.y.cast<double>().mean());
	    Y/=sqrt(Y.squaredNorm()/((double)N-1.0));
	    epsilon= Y.array();
	    sigmaE=epsilon.squaredNorm()/N*0.5;
        //xsquared=X.colwise().squaredNorm();
	    for(int iteration=0; iteration < max_iterations; iteration++){

	      if(iteration>0)
	        if( iteration % (int)std::ceil(max_iterations/10) ==0)
	          std::cout << "iteration: "<<iteration <<"\n";

	        epsilon= epsilon.array()+mu;//  we substract previous value
	        mu = norm_rng(epsilon.sum()/(double)N, sigmaE/(double)N); //update mu
	        epsilon= epsilon.array()-mu;// we substract again now epsilon =Y-mu-X*beta


	        std::random_shuffle(markerI.begin(), markerI.end());

	        m0=0;
	        v.setZero();
	        betaAcum.setZero();
	        for(int j=0; j < M; j++){

	          marker= markerI[j];
	          sigmaG=sigmaGG[data.G(marker)];

	          cVa[0]=0;
	          cVaI[0]=0;
	          cVa.segment(1,(K-1))=cva.row(data.G(marker));
	          cVaI.segment(1,(K-1))=(cVa.segment(1,(K-1))).cwiseInverse();


	          y_tilde= epsilon.array()+(X.col(marker)*beta(marker,0)).array();//now y_tilde= Y-mu-X*beta+ X.col(marker)*beta(marker)_old




	          muk[0]=0.0;//muk for the zeroth component=0

	          // std::cout<< muk;
	          //we compute the denominator in the variance expression to save computations
	          denom=xsquared(marker)+(sigmaE/sigmaG)*cVaI.segment(1,(K-1)).array();
	          //muk for the other components is computed according to equaitons
	          num=(X.col(marker).cwiseProduct(y_tilde)).sum();
	          muk.segment(1,(K-1))= num/denom.array();



	          logL= pi.row(gAssign(marker)).array().log();//first component probabilities remain unchanged


	          // Here we reproduce the fortran code
	          logL.segment(1,(K-1))=logL.segment(1,(K-1)).array() - 0.5*((((sigmaG/sigmaE)*(xsquared(marker))*cVa.segment(1,(K-1)).array() + 1).array().log()))+
	            0.5*( muk.segment(1,(K-1)).array()*num)/sigmaE;

	          double p(beta_rng(1,1));//I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later


	          if(((logL.segment(1,(K-1)).array()-logL[0]).abs().array() >700 ).any() ){
	            acum=0;
	          }else{
	            acum=1.0/((logL.array()-logL[0]).exp().sum());
	          }

	          for(int k=0;k<K;k++){
	            if(p<=acum){
	              if(k==0){
	                beta(marker,0)=0;
	              }else{
	                beta(marker,0)=norm_rng(muk[k],sigmaE/denom[k-1]);
	                betaAcum(gAssign(marker))+= pow(beta(marker,0),2);
	                // beta(marker,0)=norm_rng(rhs/denom[k-1],sigmaE/denom[k-1]);
	              }
	              v.row(gAssign(marker))(k)+=1.0;
	              components[marker]=k;
	              break;
	            }else{
	              if(((logL.segment(1,(K-1)).array()-logL[k+1]).abs().array() >700 ).any() ){
	                acum+=0;
	              }
	              else{
	                acum+=1.0/((logL.array()-logL[k+1]).exp().sum());
	              }
	            }
	          }
	          epsilon=y_tilde-X.col(marker)*beta(marker,0);//now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new


	        }



	        sigmaE=inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));


	        for(int i=0; i<groups; i++){
	          m0=v.row(i).sum()-v.row(i)(0);
	          sigmaGG[i]=inv_scaled_chisq_rng(v0G+m0,(betaAcum(i)*m0+v0G*s02G)/(v0G+m0));
	          pi.row(i)=dirichilet_rng(v.row(i).array() + 1.0);

	        }

	        if(iteration >= burn_in)
	        {
	          if(iteration % thinning == 0){
	            sample<< iteration,mu,beta,sigmaE,components,sigmaGG,epsilon;
	            q.enqueue(sample);
	          }

	        }

	    }

	    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
	    std::cout << "duration: "<<duration << "s\n";
	    flag=1;
	  }
	#pragma omp section
	{
	  bool queueFull;
	  queueFull=0;
	  std::ofstream outFile;
	  outFile.open(outputFile);
	  VectorXd sampleq(2*M+3+groups+N);
	  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
	  outFile<< "iteration,"<<"mu,";
	  for(unsigned int i = 0; i < M; ++i){
	    outFile << "beta[" << (i+1) << "],";

	  }
	  outFile<<"sigmaE,";
	  for(unsigned int i = 0; i < M; ++i){
	    outFile << "comp[" << (i+1) << "],";
	  }
	  for(unsigned int i = 0; i < groups; ++i){
	    outFile << "sigmaG[" << (i+1) << "],";
	  }
	  for(unsigned int i = 0; i < N; ++i){
	    outFile << "epsilon[" << (i+1) << "],";
	  }
	  outFile<<"\n";

	  while(!flag ){
	    if(q.try_dequeue(sampleq))
	      outFile<< sampleq.transpose().format(CommaInitFmt) << "\n";
	  }
	}

	}
	}

}
