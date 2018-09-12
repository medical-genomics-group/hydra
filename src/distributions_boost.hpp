/*
 * distributions_boost.hpp
 *
 *  Created on: 7 Sep 2018
 *      Author: admin
 */

#ifndef SRC_DISTRIBUTIONS_BOOST_HPP_
#define SRC_DISTRIBUTIONS_BOOST_HPP_


#include <random>
#include <Eigen/Eigen>

Eigen::VectorXd dirichilet_rng(Eigen::VectorXd alpha);
double inv_gamma_rng(double shape,double scale);
double gamma_rng(double shape,double scale);
double inv_gamma_rate_rng(double shape,double rate);
double gamma_rate_rng(double shape,double rate);
double inv_scaled_chisq_rng(double dof,double scale);
double norm_rng(double mu, double sigma2);
double component_probs(double b,Eigen::VectorXd pi);
double categorical(Eigen::VectorXd probs);
double beta_rng(double a, double b);
double exp_rng(double a);


#endif /* SRC_DISTRIBUTIONS_BOOST_HPP_ */
