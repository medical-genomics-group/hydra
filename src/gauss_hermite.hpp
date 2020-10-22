#ifndef HYDRA_GAUSS_HERMITE_H
#define HYDRA_GAUSS_HERMITE_H


//
/*double gauss_hermite_adaptive_integral(double C_k,
                                       double sigma,
                                       string n,
                                       double vi_sum,
                                       double vi_2,
                                       double vi_1,
                                       double vi_0,
                                       double mean,
                                       double sd,
                                       double mean_sd_ratio,
                                       const pars_beta_sparse used_data_beta);

*/

double gauss_hermite_adaptive_integral(double C_k,
                                       double sigma,
                                       std::string n,
                                       double vi_sum,
                                       double vi_2,
                                       double vi_1,
                                       double vi_0,
                                       double vi_tau_2,
                                       double vi_tau_1,
                                       double vi_tau_0,
                                       double mean,
                                       double sd,
                                       double mean_sd_ratio,
                                       const pars_beta_sparse used_data_beta,
                                       double sigmaG,
                                       double sigmaG_other,
                                       double beta_other, 
                                       double C_k_other,
                                       double vi_sum_tau);
#endif
