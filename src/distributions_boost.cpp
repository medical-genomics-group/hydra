/*
 * distributions_boost.cpp
 *
 *  Created on: 7 Sep 2018
 *      Author: admin
 */
#include <Eigen/Eigen>
#include <math.h>
#include "distributions_boost.hpp"
#include <boost/random/gamma_distribution.hpp>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/uniform_01.hpp>

double rgamma(double shape, double scale){
	boost::mt19937 rng;
	boost::random::gamma_distribution<double> myGamma(shape, scale);
	boost::random::variate_generator<boost::mt19937&, boost::random::gamma_distribution<> > rand_gamma(rng, myGamma);
	return rand_gamma();
}

Eigen::VectorXd dirichilet_rng(Eigen::VectorXd alpha) {
  int len;
  len=alpha.size();
  Eigen::VectorXd result(len);
  for(int i=0;i<len;i++)
    result[i]=rgamma(alpha[i],(double)1.0);
  result/=result.sum();
  return result;
}

double inv_gamma_rng(double shape,double scale){
  return ((double)1.0 / rgamma(shape, 1.0/scale));
}
double gamma_rng(double shape,double scale){
  return rgamma(shape, scale);
}
double inv_gamma_rate_rng(double shape,double rate){
  return (double)1.0 / gamma_rate_rng(shape, rate);
}
double gamma_rate_rng(double shape,double rate){
  return rgamma(shape,(double)1.0/rate);
}

double inv_scaled_chisq_rng(double dof,double scale){
  return inv_gamma_rng((double)0.5*dof, (double)0.5*dof*scale);
}
double norm_rng(double mean,double sigma2){
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	boost::normal_distribution<double> nd(mean, std::sqrt((double)sigma2));
	boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > var_nor(rng, nd);
  return var_nor();
}

inline double runif(){
	boost::mt19937 rng;
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();
}

double component_probs(double b2,Eigen::VectorXd pi){
  double sum;
  double p;
  p=runif();
  sum= pi[0]*exp((-0.5*b2)/(5e-2))/sqrt(5e-2)+pi[1]*exp((-0.5*b2));
  if(p<=(pi[0]*exp((-0.5*b2)/(5e-2))/sqrt(5e-2))/sum)
    return 5e-2;
  else
    return 1;
}




inline double bernoulli_rng(double probs0,double probs1,double cats0,double cats1){
  double p;
  p=runif();
  if(p<= probs0/(probs0+probs1))
    return cats0;
  else
    return cats1;
}

double beta_rng(double a,double b){
	boost::mt19937 rng;
    boost::random::beta_distribution<double> mybeta(a, b);
    boost::random::variate_generator<boost::mt19937&, boost::random::beta_distribution<> > rand_beta(rng, mybeta);
  return rand_beta();
}

