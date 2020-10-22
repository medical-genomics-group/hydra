#ifndef HYDRA_ARS_HPP_
#define HYDRA_ARS_HPP_

// Three structures for ARS


struct pars {

	// Common parameters for the densities

    // epsilon per subject (before each sampling, need to remove 
    // the effect of the sampled parameter and then carry on
    Eigen::VectorXd epsilon;		
    Eigen::VectorXd epsilon2;  // Additional residuals for second epoch
    Eigen::VectorXd epsilon3;
    Eigen::VectorXd epsilon4;
	

	//Probably unnecessary to store mixture classes
	//VectorXd mixture_classes; // Vector to store mixture component C_k values
	//int used_mixture; //Write the index of the mixture we decide to use

	// Store the current variables
	double alpha;

	// Beta_j - specific variables
    Eigen::VectorXd X_j;
    Eigen::VectorXd X_j2;

	//  of sum(X_j*failure)
	double sum_failure;

	// Mu-specific variables
	double sigma_mu;

	// Covariate-specific variables
	double sigma_covariate;

	// Number of events (sum of failure indicators)
	double d;
};

//Use this one structure for both epochs
struct pars_beta_sparse {

	// Common parameters for the densities
    
	// Instead of storing the vector of mixtures and the corresponding 
    // index, we keep only the mixture value in the structure
	double mixture_value; 

	// Store the current variables
	double alpha, sigmaG1, sigmaG2;

	// Parameter for inter-epoch correlation
	double rho;

	// indictes the epoch on which beta is being updated
	int beta_ind;

	// Beta_j - specific variables
	// Sums of exponentiated residuals - one for difference from y, one for difference from tau
	double vi_0, vi_1, vi_2; // Sums of vi elements
	double vi_tau_0, vi_tau_1, vi_tau_2;

	// Mean, std dev and their ratio for snp j
	double mean, sd, mean_sd_ratio;

	//  of sum(X_j*failure)
	double sum_failure;

	// Number of events (sum of failure indicators)
	double d;   

	// values of beta and mixture value for other epoch (sd, beta for snp j)
	double beta_other, mixture_value_other;   
};


struct pars_alpha {

    Eigen::VectorXd failure_vector;
    Eigen::VectorXd failure_vector2;  // failure indicators for second epoch 

    // epsilon per subject (before each sampling, need to remove
    // the effect of the sampled parameter and then carry on
    Eigen::VectorXd epsilon;			
    Eigen::VectorXd epsilon2;
    Eigen::VectorXd epsilon3;
    Eigen::VectorXd epsilon4;

	// Alpha-specific variable; prior parameters
	double alpha_0, kappa_0;

	// Number of events (sum of failure indicators)
	double d;
};


#endif
