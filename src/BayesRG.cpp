// this code skeleton only highlights which would be the proposal of code changes
//Additional variables
// VectorXd means; //vector that contains the mean of each column of the bed file matrix
// VectorXd sds; //vector that contains the sd of each column of the bed file matrix
// VectorXd sqrdZ; //vector that contains the sum of squares of each column the bed file matrix
// VectorXd Zsum;  //vector that contains the sum of squares of each columnof the bed file matrix
// double epsilonSum //acumulator that updates the sum of epsilon vector


//SparseMatrix<double> Zg; //this is a sparse matrix that contains the uncentered and unscaled elements of the bed matrix
//OR
//std::vector<std:vector<int>> Zones(M)//vector containing the vectors the indexes of elements of the bed matrix which are one for each column
//std::vector<std:vector<int>> Ztwos(M)//vector containing the vectors the indexes of elements of the bed matrix which are two for each column
//

BayesRRG:BayesRRmz() // BayesRG would inherit from BayesRmz, or not

BsyesRRG::init(){
    epsilonSum = y.sum();// we initialise with the current sum of y elements
}

BayesRRG:runGibbs(){
//Unlike the BayesRRmz, we would not need to compress and decompress the matrix, but we would like to have the sequential and async execution as options 
}


void BayesRRG::processColumn(unsigned int marker, const Map<VectorXd> &Cx)
{
    const unsigned int N(m_data.numInds);
    const double NM1 = double(N - 1);
    const int K(int(m_cva.size()) + 1);
    const int km1 = K - 1;
    double acum = 0.0;
    double beta_old;

    beta_old = m_beta(marker);

  
    // muk for the zeroth component=0
    m_muk[0] = 0.0;

    // We compute the denominator in the variance expression to save computations
    const double sigmaEOverSigmaG = m_sigmaE / m_sigmaG;
    m_denom = NM1 + sigmaEOverSigmaG * m_cVaI.segment(1, km1).array();

    //DANIEL here we either use the column of the sparse matrix or the two index vectors
    //num= means(marker)*epsilonSum/sds(marker)+beta_old*sqrdZ(marker)-N*means(marker)/sds(marker) +Zg.col(marker).dot(epsilon)/sds(marker)
    //OR the indexing solution, which using the development branch of eigen should be this
    //num = means(marker)*epsilonSum/sds(marker)+beta_old*sqrdZ(marker)-N*means(marker)/sds(marker) +(epsilon(Zones[marker]).sum()+2*epsilon(Ztwos[marker]).sum())/sds(marker)
    //maybe you can come up with a better way to index the elements of epsilon  
      
    //The rest of the algorithm remains the same
    
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
    
    //until here
    //we skip update if old and new beta equals zero
     const bool skipUpdate = beta_old == 0.0 && beta == 0.0;
     if (!skipUpdate) {
        //Either
        //epsilon+=(beta_old-beta_new)*Zsum(marker)/sds(marker)+(beta_new-beta_old)*means(marker)/sds(marker)*ONES;
    
       //OR
       //epsilon(Zones[marker])+=(beta_old-beta_new)/sds(marker)+(beta_new-beta_old)*means(marker)/sds(marker);
       //epsilon(Ztwos[marker])+=2*(beta_old-beta_new)/sds(marker)+(beta_new-beta_old)*means(marker)/sds(marker);  
       //Regardless of which scheme, the update of epsilonSum is the same
         epsilonSum+= (beta_old-beta_new)*Zsum(marker)
       
       }
       // Now epsilon contains Y-mu - X*beta + X.col(marker) * beta(marker)_old - X.col(marker) * beta(marker)_new
}

void BayesRRmz::updateGlobal(double beta_old, double beta, const Map<VectorXd> &Cx)
{
    // No mutex required here whilst m_globalComputeNode uses the serial policy
    //Either
    
    
    //OR
    //epsilon(Zones[marker])+=(beta_old-beta_new)/sds(marker)+(beta_new-beta_old)*means(marker)/sds(marker);
    //epsilon(Ztwos[marker])+=2*(beta_old-beta_new)/sds(marker)+(beta_new-beta_old)*means(marker)/sds(marker);
    //Regardless of which scheme, the update of epsilonSum is the same
    epsilonSum+= (beta_old-beta_new)*Zsum(marker)
    
}
