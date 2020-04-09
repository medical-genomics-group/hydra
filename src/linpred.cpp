#include "linpred.hpp"
#include "data.hpp"
#include "options.hpp"

LinPred::LinPred(Data &data, Options &opt)
    : data(data)
    , opt(opt)
{
}

/* Computes prediction for each individual from pre-computed effect estimates
 * pre  : binary PLINK files and .bet files have been read and processed
 * post : NxI prediction matrix is stored in pred member variable
 */
void LinPred::predict_genetic_values() {
    // perform prediction
    data.pred.resize(data.Z_common.rows(), data.predBet.cols());
    for (uint i = 0; i < data.predBet.cols(); i++) {
        data.pred.col(i) = (data.Z_common * data.predBet.col(i)).rowwise().sum();
    }
    // TODO: write prediction matrix to disk
}
