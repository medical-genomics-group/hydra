/*
 * BayesRRhp.cpp
 *
 *  Created on: 21 Sep 2018
 *      Author: admin
 */

#include "BayesRRhp.h"
#include <Eigen/Core>
#include <random>
#include "distributions_boost.hpp"
#include "concurrentqueue.h"
#include "options.hpp"
#include "BayesRRm.h"
#include "data.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

double  rgamma(double shape, double scale){
	boost::mt19937 rn;
	boost::random::gamma_distribution<double> myGamma(shape, scale);
	boost::random::variate_generator<boost::mt19937&, boost::random::gamma_distribution<> > rand_gamma(rn, myGamma);
	return rand_gamma();
}

double inv_gamma_rng(double shape,double scale){

  return (1.0 / rgamma(shape, 1.0/scale));
}
// [[Rcpp::export]]
double gamma_rng(double shape,double scale){
  return rgamma(shape, scale);
}
double gamma_rate_rng(double shape,double rate){
  return rgamma(shape,1.0/rate);
}

// [[Rcpp::export]]
double inv_gamma_rate_rng(double shape,double rate){
  return 1.0 / gamma_rate_rng(shape, rate);
}
// [[Rcpp::export]]

template<typename Scalar>
struct inv_gamma_functor
{
  inv_gamma_functor(const Scalar& vd):m_a(vd){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rate_rng(0.5 + 0.5*m_a,x); }
  Scalar  m_a;

};


template<typename Scalar>
struct inv_gamma_functor_init_v
{
  inv_gamma_functor_init_v(const Scalar& vd):m_a(vd){}

  const Scalar operator()(const Scalar& x)const {return inv_gamma_rate_rng(0.5*m_a,m_a*x);}
  Scalar  m_a;
};


BayesRRhp::BayesRRhp(Data &data,Options &opt, const long memPageSize):
seed(opt.seed),data(data),
opt(opt),
memPageSize(memPageSize),
max_iterations(opt.chainLength),
thinning(opt.thin),burn_in(opt.burnin),
outputFile(opt.mcmcSampleFile),
bedFile(opt.bedFile + ".bed"),
dist(opt.seed) {

}


BayesRRhp::~BayesRRhp() {
	// TODO Auto-generated destructor stub
}

void BayesRRhp::runGibbs(){
	 int flag;
	  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
	  flag=0;
	  static unsigned int M(data.numIncdSnps);
	  static unsigned int N(data.numKeptInds);
	  VectorXd components(M);
		VectorXf normedSnpData(data.numKeptInds);

	  ////////////validate inputs

	  if(max_iterations < burn_in || max_iterations<1 || burn_in<1) //validations related to mcmc burnin and iterations
	  {
	    std::cout<<"error: burn_in has to be a positive integer and smaller than the maximum number of iterations ";
	    return;
	  }

	  omp_set_nested(1); // 1 - enables nested parallelism; 0 - disables nested parallelism.


	  Eigen::initParallel();
	  double sum_beta_sqr;
	  size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;

	#pragma omp parallel  shared(flag,q,M,N)
	{
	#pragma omp sections
	{

	  {

	    //mean and residual variables
	    double mu; // mean or intercept
	    double sigmaG; //genetic variance
	    double sigmaE; // residuals variance

	    //component variables
	    VectorXd lambda(M);
	    VectorXd v(M);
	    VectorXd phi(M);
	    VectorXd chi(M);
	    double tau;
	    double eta;
	    //linear model variables
	    MatrixXd beta(M,1); // effect sizes
	    VectorXd y_tilde(N); // variable containing the adjusted residuals to exclude the effects of a given marker
	    VectorXd epsilon(N); // variable containing the residuals

	    //sampler variables
	    VectorXd sample(2*M+4+N); // varible containg a sambple of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
	    std::vector<int> markerI;
	    for (int i=0; i<M; ++i) {
	      markerI.push_back(i);
	    }


	    int marker;
	    double acum;
	    VectorXd y;
	    VectorXd cX;

	    y_tilde.setZero();

	    beta.setZero();
	    tau=dist.beta_rng(1,1);

	    mu=0;

	    eta=dist.inv_gamma_rate_rng(0.5,1/pow(A,2));
	    eta=0.00001;
	    std::cout<< "initial eta " << eta<<"\n";
	    tau=(1.0/eta)*dist.inv_gamma_rate_rng(0.5*vT,vT);

	    tau=A;
	    // tau=1/A;
	    std::cout<< "initial tau " << tau<<"\n";

	    v=(v.setOnes().array()).unaryExpr(inv_gamma_functor<double>(0));
	    v.setOnes();
	    //std::cout<< "initial v" << eta;
	    lambda=v.unaryExpr(inv_gamma_functor_init_v<double>(vL));
	    lambda.setOnes();


	    y=(data.y.cast<double>().array()-data.y.cast<double>().mean());
	    y/=sqrt(y.squaredNorm()/((double)N-1.0));

	    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	    epsilon= y.array();
	    sigmaE=epsilon.squaredNorm()/N*0.5;
	    for(int iteration=0; iteration < max_iterations; iteration++){

	      if(iteration>0)
	        if( iteration % (int)std::ceil(max_iterations/10) ==0)
	        {
	          std::cout << "iteration: "<<iteration <<"\n";
	          std::cout<< " tau " << tau<<"\n";
	          std::cout<< " eta " << eta<<"\n";
	          std::cout<< "sigmaE " << sigmaE<<"\n";
	          std::cout<<"VarE "<< beta.squaredNorm()<<"\n";
	          std::cout<<"Mu "<< mu <<"\n";
	          std::cout<<"C2 "<< c2 <<"\n";
	        }


	      epsilon= epsilon.array()+mu;//  we substract previous value
	      mu = dist.norm_rng(epsilon.sum()/(double)N, sigmaE/(double)N); //update mu
	      mu=0;//TODO remove this test value
	      epsilon= epsilon.array()-mu;// we substract again now epsilon =Y-mu-X*beta


	      std::random_shuffle(markerI.begin(), markerI.end());

	      eta = dist.inv_gamma_rate_rng(0.5+0.5*vT,(1.0/(A*A))+vT/tau);
	      v=(vL/(lambda).array()+1.0).unaryExpr(inv_gamma_functor<double>(vL));
	      for(int j=0; j < M; j++){

	        marker= markerI[j];
	        data.getSnpDataFromBedFileUsingMmap_openmp(bedFile, snpLenByt, memPageSize, marker, normedSnpData);
            cX=normedSnpData.cast<double>();
	        y_tilde= epsilon.array()+(cX*beta(marker,0)).array();//now y_tilde= Y-mu-X*beta+ X.col(marker)*beta(marker)_old



	       // std::cout<< muk;
	        //we compute the denominator in the variance expression to save computations
	        //denom=X.col(marker).squaredNorm()+(sigmaE/(tau*c2*lambda[marker]/(tau*lambda[marker]+c2)));
	        //muk for the other components is computed according to equaitons
	        //muk= (X.col(marker).cwiseProduct(y_tilde)).sum()/denom;
	        //beta(marker,0)=norm_rng(muk,sigmaE/denom);
	        beta(marker,0)=(cX.cwiseProduct(y_tilde)).sum()/((double(N)-1.0)+(sigmaE/(tau*c2*lambda[marker]/(tau*lambda[marker]+c2))))+sqrt(sigmaE/((double(N)-1.0)+(sigmaE/(tau*c2*lambda[marker]/(tau*lambda[marker]+c2)))))*dist.norm_rng(0,1);



	        epsilon=y_tilde-cX*beta(marker,0);//now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new

	      }

	      lambda=(vL*v.cwiseInverse()+(0.5*beta.cwiseProduct(beta)*(1.0/tau))).unaryExpr(inv_gamma_functor<double>(vL));
	    //  if(iteration==0)
	      //  std::cout<< " lambda " << lambda<<"\n";
	      tau= dist.inv_gamma_rate_rng(0.5*((double)M+vT),vT/eta+((0.5)*(double)((beta.array().pow(2))/lambda.array()).sum()));

	      //tau=A;
	      //double vC(10.0),sC(100);
	      //c2=inv_gamma_rate_rng(0.5*vC+0.5*M,vC*sC*0.5+0.5*beta.squaredNorm());
	    //  c2=sC;



	      sigmaE=dist.inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));

	      if(iteration >= burn_in)
	      {
	        if(iteration % thinning == 0){
	          sample<< iteration,mu,beta,sigmaE,sigmaG,lambda,epsilon;
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
	  VectorXd sampleq(2*M+4+N);
	  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
	  outFile<< "iteration,"<<"mu,";
	  for(unsigned int i = 0; i < M; ++i){
	    outFile << "beta[" << (i+1) << "],";

	  }
	  outFile<<"sigmaE,"<<"sigmaG,";
	  for(unsigned int i = 0; i < M; ++i){
	    outFile << "comp[" << (i+1) << "],";
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


