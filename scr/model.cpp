//
//  model.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "model.hpp"


void BayesC::FixedEffects::sampleFromFC(VectorXf &ycorr, const MatrixXf &X,
                                        const VectorXf &XPXdiag, const float vare){
    float rhs;
    for (unsigned i=0; i<size; ++i) {
        float oldSample = values[i];
        float my_rhs = X.col(i).dot(ycorr);
        MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        rhs += XPXdiag[i]*oldSample;
        float invLhs = 1.0f/XPXdiag[i];
        float bhat = invLhs*rhs;
        values[i] = Normal::sample(bhat, invLhs*vare);
        ycorr += X.col(i) * (oldSample - values[i]);
    }
}

void BayesC::SnpEffects::sampleFromFC(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag,
const float sigmaSq, const float pi, const float vare, VectorXf &ghat){
    if (algorithm == gibbs) {
        gibbsSampler(ycorr, Z, ZPZdiag, sigmaSq, pi, vare, ghat);
    } else if (algorithm == hmc) {
        hmcSampler(ycorr, Z, ZPZdiag, sigmaSq, pi, vare, ghat);
    }
}

void BayesC::SnpEffects::gibbsSampler(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag,
                                      const float sigmaSq, const float pi, const float vare, VectorXf &ghat){
    sumSq = 0.0;
    numNonZeros = 0;
    
    ghat.setZero(ycorr.size());
    
    float oldSample;
    float my_rhs, rhs, invLhs, uhat;
    float logDelta0, logDelta1, probDelta1;
    float logPi = log(pi);
    float logPiComp = log(1.0-pi);
    float logSigmaSq = log(sigmaSq);
    float invVare = 1.0f/vare;
    float invSigmaSq = 1.0f/sigmaSq;
    
    for (unsigned i=0; i<size; ++i) {
        oldSample = values[i];
        my_rhs = Z.col(i).dot(ycorr);
        MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        rhs += ZPZdiag[i]*oldSample;
        rhs *= invVare;
        invLhs = 1.0f/(ZPZdiag[i]*invVare + invSigmaSq);
        uhat = invLhs*rhs;
        logDelta1 = 0.5*(logf(invLhs) - logSigmaSq + uhat*rhs) + logPi;
        //logDelta1 = rhs*oldSample - 0.5*ZPZdiag[i]*oldSample*oldSample/vare + logPiComp;
        logDelta0 = logPiComp;
        probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
        
        //cout << i << " rhs " << rhs << " invLhs " << invLhs << " uhat " << uhat << endl;

        if (bernoulli.sample(probDelta1)) {
            values[i] = normal.sample(uhat, invLhs);
            ycorr += Z.col(i) * (oldSample - values[i]);
            ghat  += Z.col(i) * values[i];
            sumSq += values[i]*values[i];
            ++numNonZeros;
        } else {
            if (oldSample) ycorr += Z.col(i) * oldSample;
            values[i] = 0.0;
        }
    }
}

void BayesC::SnpEffects::hmcSampler(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag, const float sigmaSq, const float pi, const float vare, VectorXf &ghat){
    // Hamiltonian Monte Carlo
    // Only BayesC0 model available
    
    float stepSize = 0.1;
    unsigned numSteps = 10;
    
    ycorr += Z*values;
    
    static MatrixXf ZPZ;
    if (cnt==0) ZPZ = Z.transpose()*Z;
    VectorXf ypZ = ycorr.transpose()*Z;
    
    VectorXf curr = values;
    
    ArrayXf curr_p(size);
    for (unsigned i=0; i<size; ++i) {
        curr_p[i] = Stat::snorm();
    }
    
    VectorXf cand = curr;
    // Make a half step for momentum at the beginning
    ArrayXf cand_p = curr_p - 0.5*stepSize * gradientU(curr, ZPZ, ypZ, sigmaSq, vare);
    
    for (unsigned i=0; i<numSteps; ++i) {
        cand.array() += stepSize * cand_p;
        if (i < numSteps-1) {
            cand_p -= stepSize * gradientU(cand, ZPZ, ypZ, sigmaSq, vare);
        } else {
            cand_p -= 0.5*stepSize * gradientU(cand, ZPZ, ypZ, sigmaSq, vare);
        }
    }
    
    float curr_H = computeU(curr, ZPZ, ypZ, sigmaSq, vare) + 0.5*curr_p.matrix().squaredNorm();
    float cand_H = computeU(cand, ZPZ, ypZ, sigmaSq, vare) + 0.5*cand_p.matrix().squaredNorm();
    
    if (Stat::ranf() < exp(curr_H-cand_H)) {  // accept
        values = cand;
        ghat = Z*values;
        ++mhr;
    }
    
    if (!(++cnt % 100) && myMPI::rank==0) {
        float ar = mhr/float(cnt);
        if      (ar < 0.5) cout << "Warning: acceptance rate for SNP effects is too low "  << ar << endl;
        else if (ar > 0.9) cout << "Warning: acceptance rate for SNP effects is too high " << ar << endl;
    }
    
    numNonZeros = size;
    sumSq = values.squaredNorm();
    
    ycorr -= Z*values;
}

ArrayXf BayesC::SnpEffects::gradientU(const VectorXf &alpha, const MatrixXf &ZPZ, const VectorXf &ypZ, const float sigmaSq, const float vare){
    return 1.0/vare*(ZPZ*alpha - ypZ) + 1/sigmaSq*alpha;
}

float BayesC::SnpEffects::computeU(const VectorXf &alpha, const MatrixXf &ZPZ, const VectorXf &ypZ, const float sigmaSq, const float vare){
    return 0.5/vare*(alpha.transpose()*ZPZ*alpha + vare/sigmaSq*alpha.squaredNorm() - 2.0*ypZ.dot(alpha));
}

void BayesC::SnpEffects::sampleFromFC_omp(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag,
                                          const float sigmaSq, const float pi, const float vare, VectorXf &ghat){
    // speed-enhanced single site Gibbs sampling due to the use of parallel computing on SNPs with zero effect
    
    unsigned blockSize = 1; //omp_get_num_threads();
    //cout << blockSize << endl;
    
    sumSq = 0.0;
    numNonZeros = 0;
    
    ghat.setZero(ycorr.size());
    
    float oldSample;
    float logPi = log(pi);
    float logPiComp = log(1.0-pi);
    float logSigmaSq = log(sigmaSq);
    float invVare = 1.0f/vare;
    float invSigmaSq = 1.0f/sigmaSq;
    
    vector<int> deltaVec(blockSize);
    vector<float> invLhsVec(blockSize);
    vector<float> uhatVec(blockSize);
    
    unsigned blocki;
    unsigned i, j;
    bool breakFlag;
    
    for (i=0; i<size; ) {
        
        if (blockSize + i < size) {
            blocki = blockSize;
        } else {
            blocki = size - i;
            deltaVec.resize(blocki);
            invLhsVec.resize(blocki);
            uhatVec.resize(blocki);
        }
        
        #pragma omp parallel for
        for (j=0; j<blocki; ++j) {
            float rhsj = (Z.col(i+j).dot(ycorr) + ZPZdiag[i+j]*values[i+j])*invVare;
            invLhsVec[j] = 1.0f/(ZPZdiag[i+j]*invVare + invSigmaSq);
            uhatVec[j] = invLhsVec[j]*rhsj;
            float logDelta0minusDelta1j = logPiComp - (0.5f*(logf(invLhsVec[j]) - logSigmaSq + uhatVec[j]*rhsj) + logPi);
            deltaVec[j] = bernoulli.sample(1.0f/(1.0f + expf(logDelta0minusDelta1j)));
        }
        
        breakFlag = false;
        for (j=0; j<blocki; ++j) {
            if (values[i+j] || deltaVec[j]) {   // need to update ycorr for the first snp who is in the model at either last or this iteration
                i += j;
                breakFlag = true;
                break;
            }
        }
        
        if (breakFlag) {
            oldSample = values[i];
            if (deltaVec[j]) {
                values[i] = normal.sample(uhatVec[j], invLhsVec[j]);
                ycorr += Z.col(i) * (oldSample - values[i]);
                ghat  += Z.col(i) * values[i];
                sumSq += values[i]*values[i];
                ++numNonZeros;
            } else {
                if (oldSample) ycorr += Z.col(i) * oldSample;
                values[i] = 0.0;
            }
            ++i;
        }
        else {
            i += blocki;
        }
    }
}

void BayesC::VarEffects::sampleFromFC(const float snpEffSumSq, const unsigned numSnpEff){
    float dfTilde = df + numSnpEff;
    float scaleTilde = snpEffSumSq + df*scale;
    value = InvChiSq::sample(dfTilde, scaleTilde);
    //cout << "snpEffSumSq " << snpEffSumSq << " scale " << scale << " scaleTilde " << scaleTilde << " dfTilde " << dfTilde << " value " << value << endl;
}

void BayesC::ScaleVar::sampleFromFC(const float sigmaSq, const float df, float &scaleVar){
    float shapeTilde = shape + 0.5*df;
    float scaleTilde = 1.0/(1.0/scale + 0.5*df/sigmaSq);
    value = Gamma::sample(shapeTilde, scaleTilde);
    scaleVar = value;
}

void BayesC::ProbFixed::sampleFromFC(const unsigned numSnps, const unsigned numSnpEff){
    float alphaTilde = numSnpEff + alpha;
    float betaTilde  = numSnps - numSnpEff + beta;
    value = Beta::sample(alphaTilde, betaTilde);
}

void BayesC::ResidualVar::sampleFromFC(VectorXf &ycorr){
    float sse;
    float my_sse = ycorr.squaredNorm();
    MPI_Allreduce(&my_sse, &sse, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float dfTilde = df + nobs;
    float scaleTilde = sse + df*scale;
    value = InvChiSq::sample(dfTilde, scaleTilde);
}

void BayesC::GenotypicVar::compute(const VectorXf &ghat){
    //value = Gadget::calcVariance(ghat);
    float my_sum = ghat.sum();
    float my_ssq = ghat.squaredNorm();
    unsigned my_size = (unsigned)ghat.size();
    float sum, ssq;
    unsigned size;
    MPI_Allreduce(&my_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&my_ssq, &ssq, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&my_size, &size, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    float mean = sum/size;
    value = ssq/size - mean*mean;
}

void BayesC::Rounding::computeYcorr(const VectorXf &y, const MatrixXf &X, const MatrixXf &Z,
                                    const VectorXf &fixedEffects, const VectorXf &snpEffects,
                                    VectorXf &ycorr){
    if (count++ % 100) return;
    VectorXf oldYcorr = ycorr;
    ycorr = y - X*fixedEffects;
    for (unsigned i=0; i<snpEffects.size(); ++i) {
        if (snpEffects[i]) ycorr -= Z.col(i)*snpEffects[i];
    }
    float my_ss = (ycorr - oldYcorr).squaredNorm();
    float ss;
    MPI_Allreduce(&my_ss, &ss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    value = sqrt(ss);
}

void BayesC::sampleUnknowns(){
    fixedEffects.sampleFromFC(ycorr, data.X, data.XPXdiag, vare.value);
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(ycorr, data.Z, data.ZPZdiag, sigmaSq.value, pi.value, vare.value, ghat);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);
    sigmaSq.sampleFromFC(snpEffects.sumSq, snpEffects.numNonZeros);
    //scale.sampleFromFC(sigmaSq.value, sigmaSq.df, sigmaSq.scale);
    if (estimatePi) pi.sampleFromFC(snpEffects.size, snpEffects.numNonZeros);
    vare.sampleFromFC(ycorr);
    
    varg.compute(ghat);
    hsq.compute(varg.value, vare.value);
    
    rounding.computeYcorr(data.y, data.X, data.Z, fixedEffects.values, snpEffects.values, ycorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
}


void BayesN::SnpEffects::sampleFromFC(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag,
                                      const float sigmaSq, const float pi, const float vare, VectorXf &ghat){
    sumSq = 0.0;
    numNonZeros = 0;
    numNonZeroWind = 0;
    
    ghat.setZero(ycorr.size());
    
    float oldSample;
    float my_rhs, rhs, invLhs, uhat;
    float logDelta0, logDelta1, probDelta1;
    float logPi = log(pi);
    float logPiComp = log(1.0-pi);
    float logSigmaSq = log(sigmaSq);
    float invVare = 1.0f/vare;
    float invSigmaSq = 1.0f/sigmaSq;
    float diffQuadSum;
    float logDelta0MinusLogDelta1;
    
    unsigned start, end;
    
    for (unsigned i=0; i<numWindows; ++i) {
        start = windStart[i];
        end = i+1 < numWindows ? windStart[i] + windSize[i] : size;
        
        // sample window delta
        diffQuadSum = 0.0;
        if (windDelta[i]) {
            for (unsigned j=start; j<end; ++j) {
                if (snpDelta[j]) {
                    my_rhs = Z.col(j).dot(ycorr);
                    MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    diffQuadSum += 2.0f*beta[j]*rhs + beta[j]*beta[j]*ZPZdiag[j];
                }
            }
        } else {
            for (unsigned j=start; j<end; ++j) {
                if (snpDelta[j]) {
                    my_rhs = Z.col(j).dot(ycorr);
                    MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    diffQuadSum += 2.0f*beta[j]*rhs - beta[j]*beta[j]*ZPZdiag[j];
                }
            }
        }
        
        diffQuadSum *= invVare;
        logDelta0MinusLogDelta1 = -0.5f*diffQuadSum + logPiComp - logPi;
        probDelta1 = 1.0f/(1.0f + expf(logDelta0MinusLogDelta1));
        
        if (bernoulli.sample(probDelta1)) {
            if (!windDelta[i]) {
                for (unsigned j=start; j<end; ++j) {
                    if (snpDelta[j]) {
                        ycorr -= Z.col(j) * beta[j];
                    }
                }
            }
            windDelta[i] = 1.0;
            ++numNonZeroWind;
            
            for (unsigned j=start; j<end; ++j) {
                oldSample = beta[j]*snpDelta[j];
                my_rhs = Z.col(j).dot(ycorr);
                
                MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                
                rhs += ZPZdiag[j]*oldSample;
                rhs *= invVare;
                invLhs = 1.0f/(ZPZdiag[j]*invVare + invSigmaSq);
                uhat = invLhs*rhs;
                logDelta1 = 0.5*(logf(invLhs) - logSigmaSq + uhat*rhs) + logLocalPi[i];
                logDelta0 = logLocalPiComp[i];
                probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
                if (bernoulli.sample(probDelta1)) {
                    values[j] = beta[j] = normal.sample(uhat, invLhs);
                    ycorr += Z.col(j) * (oldSample - values[j]);
                    ghat  += Z.col(j) * values[j];
                    sumSq += values[j]*values[j];
                    snpDelta[j] = 1.0;
                    ++cumDelta[j];
                    ++numNonZeros;
                } else {
                    if (oldSample) ycorr += Z.col(j) * oldSample;
                    beta[j] = normal.sample(0.0, sigmaSq);
                    snpDelta[j] = 0.0;
                    values[j] = 0.0;
                }
                //sumSq += beta[j]*beta[j];
            }
        }
        else {
//            unsigned windSize = end-start;
//            float localSum = cumDelta.segment(start,windSize).sum();
            for (unsigned j=start; j<end; ++j) {
                beta[j] = normal.sample(0.0, sigmaSq);
                snpDelta[j] = bernoulli.sample(localPi[i]);
//                float seudopi = (localPi[i]/(windSize-1)+cumDelta[j])/(localPi[i]+localSum-cumDelta[j]);
//                snpDelta[j] = bernoulli.sample(seudopi);
                if (values[j]) ycorr += Z.col(j) * values[j];
                values[j] = 0.0;
                //sumSq += beta[j]*beta[j];
            }
            windDelta[i] = 0.0;
        }
    }
}

void BayesN::sampleUnknowns(){
    fixedEffects.sampleFromFC(ycorr, data.X, data.XPXdiag, vare.value);
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(ycorr, data.Z, data.ZPZdiag, sigmaSq.value, pi.value, vare.value, ghat);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);
    sigmaSq.sampleFromFC(snpEffects.sumSq, snpEffects.numNonZeros);
    //scale.sampleFromFC(sigmaSq.value, sigmaSq.df, sigmaSq.scale);
    if (estimatePi) pi.sampleFromFC(snpEffects.numWindows, snpEffects.numNonZeroWind);
    vare.sampleFromFC(ycorr);
    
    varg.compute(ghat);
    hsq.compute(varg.value, vare.value);
    
    rounding.computeYcorr(data.y, data.X, data.Z, fixedEffects.values, snpEffects.values, ycorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
    nnzWind.getValue(snpEffects.numNonZeroWind);
    windDelta.getValues(snpEffects.windDelta);
}


void BayesS::AcceptanceRate::count(const bool state, const float lower, const float upper){
    accepted += state;
    value = accepted/float(++cnt);
    if (!state) ++consecRej;
    else consecRej = 0;
//    if (!(cnt % 100) && myMPI::rank==0) {
//        if      (value < lower) cout << "Warning: acceptance rate is too low  " << value << endl;
//        else if (value > upper) cout << "Warning: acceptance rate is too high " << value << endl;
//    }
}

void BayesS::Sp::sampleFromFC(const float snpEffWtdSumSq, const unsigned numNonZeros, const float sigmaSq, const VectorXf &snpEffects,
                              const VectorXf &snp2pq, ArrayXf &snp2pqPowS, const ArrayXf &logSnp2pq,
                              const float vg, float &scale, float &sum2pqOneMinusS){
    if (algorithm == random_walk) {
        randomWalkMHsampler(snpEffWtdSumSq, numNonZeros, sigmaSq, snpEffects, snp2pq, snp2pqPowS, logSnp2pq, vg, scale, sum2pqOneMinusS);
    } else if (algorithm == hmc) {
        hmcSampler(numNonZeros, sigmaSq, snpEffects, snp2pq, snp2pqPowS, logSnp2pq, vg, scale, sum2pqOneMinusS);
    }
}

void BayesS::Sp::randomWalkMHsampler(const float snpEffWtdSumSq, const unsigned numNonZeros, const float sigmaSq, const VectorXf &snpEffects,
                                     const VectorXf &snp2pq, ArrayXf &snp2pqPowS, const ArrayXf &logSnp2pq,
                                     const float vg, float &scale, float &sum2pqOneMinusS){
    // Random walk Mentroplis-Hastings
    // note that the scale factor of sigmaSq will be simultaneously updated
    
    float curr = value;
    float cand = sample(value, varProp);
    
    float sumLog2pq = 0;
    float snpEffWtdSumSqCurr = snpEffWtdSumSq;
    float snpEffWtdSumSqCand = 0;
    float snp2pqCand = 0;
    float sum2pqOneMinusCand = 0;
    for (unsigned i=0; i<numSnps; ++i) {
        if (snpEffects[i]) {
            sumLog2pq += logf(snp2pq[i]);
            snp2pqCand = powf(snp2pq[i], cand);
            snpEffWtdSumSqCand += snpEffects[i]*snpEffects[i]*snp2pqCand;
            sum2pqOneMinusCand += snp2pq[i]/snp2pqCand;
        }
    }
    
    float logCurr = -0.5f*(-curr*sumLog2pq + snpEffWtdSumSqCurr/sigmaSq + curr*curr/var);
    float logCand = -0.5f*(-cand*sumLog2pq + snpEffWtdSumSqCand/sigmaSq + cand*cand/var);
    
    //cout << "curr " << curr << " logCurr " << logCurr << " cand " << cand << " logCand " << logCand << " sigmaSq " << sigmaSq << endl;

    float scaleCurr = scale;
    float scaleCand = 0.5f*vg/sum2pqOneMinusCand; // based on the mean of scaled inverse chisq distribution

    // terms due to scale factor of scaled-inverse chi-square distribution
//    float logChisqCurr = 2.0f*log(scaleCurr) - 2.0f*scaleCurr/sigmaSq;
//    float logChisqCand = 2.0f*log(scaleCand) - 2.0f*scaleCand/sigmaSq;
    
    //cout << "curr " << curr << " logChisqCurr " << logChisqCurr << " cand " << cand << " logChisqCand " <<  logChisqCand << endl;
    //cout << "scaleCurr " << scaleCurr << " scaleCand " << scaleCand << endl;
    
//    if (abs(logCand-logCurr) > abs(logChisqCand-logChisqCurr)*10) {  // to avoid the prior of variance dominating the posterior when number of nonzeros are very small
//        logCurr += logChisqCurr;
//        logCand += logChisqCand;
//    }
    
    //cout << "prob " << exp(logCand-logCurr) << endl;
    
    if (Stat::ranf() < exp(logCand-logCurr)) {  // accept
        value = cand;
        scale = scaleCand;
        snp2pqPowS = snp2pq.array().pow(cand);
        sum2pqOneMinusS = sum2pqOneMinusCand;
        ar.count(1, 0.1, 0.5);
    } else {
        ar.count(0, 0.1, 0.5);
    }
    
    if (!(ar.cnt % 10)) {
        if      (ar.value < 0.2) varProp *= 0.8;
        else if (ar.value > 0.5) varProp *= 1.2;
    }
    
    tuner.value = varProp;
}

void BayesS::Sp::hmcSampler(const unsigned numNonZeros, const float sigmaSq, const VectorXf &snpEffects,
                            const VectorXf &snp2pq, ArrayXf &snp2pqPowS, const ArrayXf &logSnp2pq,
                            const float vg, float &scale, float &sum2pqOneMinusS){
    // Hamiltonian Monte Carlo
    // note that the scale factor of sigmaSq will be simultaneously updated
    
    // Prepare
    ArrayXf snpEffectDelta1(numNonZeros);
    ArrayXf snp2pqDelta1(numNonZeros);
    ArrayXf logSnp2pqDelta1(numNonZeros);
    
    for (unsigned i=0, j=0; i<numSnps; ++i) {
        if (snpEffects[i]) {
            snpEffectDelta1[j] = snpEffects[i];
            snp2pqDelta1[j] = snp2pq[i];
            logSnp2pqDelta1[j] = logSnp2pq[i];
            ++j;
        }
    }
    
    float snp2pqLogSumDelta1 = logSnp2pqDelta1.sum();
    
    float curr = value;
    float curr_p = Stat::snorm();
    
    float cand = curr;
    // Make a half step for momentum at the beginning
    float cand_p = curr_p - 0.5*stepSize * gradientU(curr,  snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq, vg);
    
    for (unsigned i=0; i<numSteps; ++i) {
        // Make a full step for the position
        cand += stepSize * cand_p;
        if (i < numSteps-1) {
            // Make a full step for the momentum, except at end of trajectory
            cand_p -= stepSize * gradientU(cand, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq, vg);
        } else {
            // Make a half step for momentum at the end
            cand_p -= 0.5*stepSize * gradientU(cand, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq, vg);
        }
        //cout << i << " " << cand << endl;
    }

    // Evaluate potential (negative log posterior) and kinetic energies at start and end of trajectory
    float scaleCurr, scaleCand;
    float curr_U_chisq, cand_U_chisq;
    float curr_H = computeU(curr, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq, vg, scaleCurr, curr_U_chisq) + 0.5*curr_p*curr_p;
    float cand_H = computeU(cand, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq, vg, scaleCand, cand_U_chisq) + 0.5*cand_p*cand_p;
    
//    if (abs(curr_H-cand_H) > abs(curr_U_chisq-cand_U_chisq)*10) { // temporary fix to avoid the prior of variance dominating the posterior (especially when number of nonzeros are very small)
//        curr_H += curr_U_chisq;
//        cand_H += cand_U_chisq;
//    }
    
    //cout << " curr " << curr << " curr_H " << curr_H << " curr_U " << curr_H - 0.5*curr_p*curr_p << " curr_p " << 0.5*curr_p*curr_p << " curr_scale " << scaleCurr << " sigmaSq " << sigmaSq << endl;
    //cout << " cand " << cand << " cand_H " << cand_H << " cand_U " << cand_H - 0.5*cand_p*cand_p << " curr_p " << 0.5*cand_p*cand_p << " cand_scale " << scaleCand << endl;
    //cout << "curr_H-cand_H " << curr_H-cand_H << endl;
    
    if (Stat::ranf() < exp(curr_H-cand_H)) {  // accept
        value = cand;
        scale = scaleCand;
        snp2pqPowS = snp2pq.array().pow(cand);
        sum2pqOneMinusS = snp2pqDelta1.pow(1.0+value).sum();
        ar.count(1, 0.5, 0.9);
    } else {
        ar.count(0, 0.5, 0.9);
    }
    
    if (!(ar.cnt % 10)) {
        if      (ar.value < 0.6) stepSize *= 0.8;
        else if (ar.value > 0.8) stepSize *= 1.2;
    }

    if (ar.consecRej > 20) stepSize *= 0.8;

    tuner.value = stepSize;
}

float BayesS::Sp::gradientU(const float S, const ArrayXf &snpEffects, const float snp2pqLogSum, const ArrayXf &snp2pq, const ArrayXf &logSnp2pq, const float sigmaSq, const float vg){
    // compute the first derivative of the negative log posterior

    long size = snp2pq.size();
    long chunkSize = size/omp_get_num_threads();
    ArrayXf snp2pqPowS(size);
#pragma omp parallel for schedule(dynamic, chunkSize)
    for (unsigned i=0; i<size; ++i) {
        snp2pqPowS[i] = powf(snp2pq[i], S);
    }
//    ArrayXf snp2pqPowS = snp2pq.pow(S);
    float constantA = snp2pqLogSum;
    float constantB = (snpEffects.square()*logSnp2pq/snp2pqPowS).sum();
    //float constantC = (snp2pq/snp2pqPowS).sum();
    //float constantD = (logSnp2pq*snp2pq/snp2pqPowS).sum();
    float ret = 0.5*constantA - 0.5/sigmaSq*constantB + S/var;
    //float dchisq = - 2.0/constantC*constantD + vg/(sigmaSq*constantC*constantC)*constantD;
    //ret += dchisq;
    //cout << ret << " " << dchisq << endl;
    return ret;
}

float BayesS::Sp::computeU(const float S, const ArrayXf &snpEffects, const float snp2pqLogSum, const ArrayXf &snp2pq, const ArrayXf &logSnp2pq, const float sigmaSq, const float vg, float &scale, float &U_chisq){
    // compute negative log posterior and scale
    ArrayXf snp2pqPowS = snp2pq.pow(S);
    float constantA = snp2pqLogSum;
    float constantB = (snpEffects.square()/snp2pqPowS).sum();
    float constantC = (snp2pq*snp2pqPowS).sum();
    scale = 0.5*vg/constantC;
    float ret = 0.5*S*constantA + 0.5/sigmaSq*constantB + 0.5*S*S/var;
    U_chisq = 2.0*logf(constantC) + scale/sigmaSq;
    //cout << abs(ret) << " " << dchisq << endl;
    //if (abs(ret) > abs(dchisq)) ret += dchisq;
    return ret;
}

void BayesS::SnpEffects::sampleFromFC(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag,
                                      const float sigmaSq, const float pi, const float vare,
                                      const ArrayXf &snp2pqPowS, const VectorXf &snp2pq,
                                      const float vg, float &scale, VectorXf &ghat){
    wtdSumSq = 0.0;
    numNonZeros = 0;

    ghat.setZero(ycorr.size());

    float oldSample;
    float my_rhs, rhs, invLhs, uhat;
    float logDelta0, logDelta1, probDelta1;
    float logPi = log(pi);
    float logPiComp = log(1.0-pi);
    float invVare = 1.0f/vare;
    float invSigmaSq = 1.0f/sigmaSq;
//    float invSigmaSq = sum2pqOneMinusS/sigmaSq;
//    float snp2pqOneMinusS;
    
    for (unsigned i=0; i<size; ++i) {
        oldSample = values[i];
        my_rhs = Z.col(i).dot(ycorr);
        
        MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        rhs += ZPZdiag[i]*oldSample;
        rhs *= invVare;
        invLhs = 1.0f/(ZPZdiag[i]*invVare + invSigmaSq/snp2pqPowS[i]);
        uhat = invLhs*rhs;
        logDelta1 = 0.5*(logf(invLhs) - logf(snp2pqPowS[i]*sigmaSq) + uhat*rhs) + logPi;
        logDelta0 = logPiComp;
        
        //cout << i << " " << sum2pqOneMinusS << " " << invLhs << endl;
        
        // terms due to the presence of delta in the scale factor for the effect variance
//        snp2pqOneMinusS =  snp2pq[i]/snp2pqPowS[i];
//        if (oldSample) sum2pqOneMinusS -= snp2pqOneMinusS;
//        if (sum2pqOneMinusS > 0) {
//            logDelta1 += -2.0f*logf(sum2pqOneMinusS+snp2pqOneMinusS) - vg/(sigmaSq*(sum2pqOneMinusS+snp2pqOneMinusS));
//            logDelta0 += -2.0f*logf(sum2pqOneMinusS) - vg/(sigmaSq*sum2pqOneMinusS);
//        }
        // end
        
        probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
        
        if (bernoulli.sample(probDelta1)) {
            values[i] = normal.sample(uhat, invLhs);
            ycorr += Z.col(i) * (oldSample - values[i]);
            ghat  += Z.col(i) * values[i];
            wtdSumSq += values[i]*values[i]/snp2pqPowS[i];
//            sum2pqOneMinusS += snp2pqOneMinusS;
            ++numNonZeros;
        } else {
            if (oldSample) ycorr += Z.col(i) * oldSample;
            values[i] = 0.0;
        }
    }
    
    //scale = 0.5f*vg/sum2pqOneMinusS;
}

void BayesS::sampleUnknowns(){
    static unsigned iter = 0;
    
    fixedEffects.sampleFromFC(ycorr, data.X, data.XPXdiag, vare.value);
    
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(ycorr, data.Z, data.ZPZdiag, sigmaSq.value, pi.value, vare.value, snp2pqPowS, data.snp2pq, genVarPrior, sigmaSq.scale, ghat);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);
    
    sigmaSq.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros);
    
    if (estimatePi) pi.sampleFromFC(snpEffects.size, snpEffects.numNonZeros);
    vare.sampleFromFC(ycorr);
    S.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros, sigmaSq.value, snpEffects.values, data.snp2pq, snp2pqPowS, logSnp2pq, genVarPrior, sigmaSq.scale, snpEffects.sum2pqOneMinusS);
    if (iter >= 2000) sigmaSq.scale = scalePrior;
    scale.getValue(sigmaSq.scale);
    
    varg.compute(ghat);
    hsq.compute(varg.value, vare.value);
    
    rounding.computeYcorr(data.y, data.X, data.Z, fixedEffects.values, snpEffects.values, ycorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
    
    if (++iter < 2000) {
        genVarPrior += (varg.value - genVarPrior)/iter;
        scalePrior += (sigmaSq.scale - scalePrior)/iter;
    }
}

void BayesS::findStartValueForS(const vector<float> &val){
    long size = val.size();
    float start;
    if (size == 1) start = val[0];
    else {
        cout << "Finding the optimal starting value for S ..." << endl;
        float loglike=0, topLoglike=0, optimal=0;
        unsigned idx = 0;
        for (unsigned i=0; i<size; ++i) {
            vector<float> cand = {val[i]};
            BayesS *model = new BayesS(data, varg.value, vare.value, pi.value, estimatePi, S.var, cand, "", false);
            unsigned numiter = 100;
            for (unsigned iter=0; iter<numiter; ++iter) {
                model->sampleUnknownsWarmup();
            }
            loglike = model->computeLogLikelihood();
            if (i==0) {
                topLoglike = loglike;
                optimal = model->S.value;
            }
            if (loglike > topLoglike) {
                idx = i;
                topLoglike = loglike;
                optimal = model->S.value;
            }
            //cout << val[i] <<" " << loglike << " " << model->S.value << endl;
            delete model;
        }
        start = optimal;
        cout << "The optimal starting value for S is " << start << endl;
    }
    S.value = start;
}

float BayesS::computeLogLikelihood(){
    float sse;
    float my_sse = ycorr.squaredNorm();
    MPI_Allreduce(&my_sse, &sse, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return -0.5f*data.numKeptInds*log(vare.value) - 0.5f*sse/vare.value;
}

void BayesS::sampleUnknownsWarmup(){
    fixedEffects.sampleFromFC(ycorr, data.X, data.XPXdiag, vare.value);
    snpEffects.sampleFromFC(ycorr, data.Z, data.ZPZdiag, sigmaSq.value, pi.value, vare.value, snp2pqPowS, data.snp2pq, varg.value, sigmaSq.scale, ghat);
    sigmaSq.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros);
    if (estimatePi) pi.sampleFromFC(snpEffects.size, snpEffects.numNonZeros);
    vare.sampleFromFC(ycorr);
    S.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros, sigmaSq.value, snpEffects.values, data.snp2pq, snp2pqPowS, logSnp2pq, varg.value, sigmaSq.scale, snpEffects.sum2pqOneMinusS);
    scale.getValue(sigmaSq.scale);
    varg.compute(ghat);
}


void BayesNS::SnpEffects::sampleFromFC(VectorXf &ycorr, const MatrixXf &Z, const VectorXf &ZPZdiag,
                                       const float sigmaSq, const float pi, const float vare,
                                       const ArrayXf &snp2pqPowS, const VectorXf &snp2pq,
                                       const float vg, float &scale, VectorXf &ghat){
    wtdSumSq = 0.0;
    numNonZeros = 0;
    numNonZeroWind = 0;
    
    static unsigned iter = 0;
    static unsigned burnin = 2000;
    //if (++iter < burnin) varPseudoPrior += (sigmaSq/snp2pqPowS - varPseudoPrior)/iter;
    
    ghat.setZero(ycorr.size());
    
    float oldSample;
    float my_rhs, rhs, invLhs, uhat;
    float logDelta0, logDelta1, probDelta1;
    float logPi = log(pi);
    float logPiComp = log(1.0-pi);
    float invVare = 1.0f/vare;
    float invSigmaSq = 1.0f/sigmaSq;
    float diffQuadSum;
    float logDelta0MinusLogDelta1;
    float snp2pqOneMinusS;
    
    unsigned start, end;
    
    for (unsigned i=0; i<numWindows; ++i) {
        start = windStart[i];
        end = i+1 < numWindows ? windStart[i] + windSize[i] : size;
        
        // sample window delta
        diffQuadSum = 0.0;
        snp2pqOneMinusS = 0.0;
        if (windDelta[i]) {
            for (unsigned j=start; j<end; ++j) {
                if (snpDelta[j]) {
                    my_rhs = Z.col(j).dot(ycorr);
                    MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    diffQuadSum += 2.0f*beta[j]*rhs + beta[j]*beta[j]*ZPZdiag[j];
//                    snp2pqOneMinusS += snp2pq[j]/snp2pqPowS[j];
//                    sum2pqOneMinusS -= snp2pq[j]/snp2pqPowS[j];
                }
            }
        } else {
            for (unsigned j=start; j<end; ++j) {
                if (snpDelta[j]) {
                    my_rhs = Z.col(j).dot(ycorr);
                    MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    diffQuadSum += 2.0f*beta[j]*rhs - beta[j]*beta[j]*ZPZdiag[j];
//                    snp2pqOneMinusS += snp2pq[j]/snp2pqPowS[j];
                }
            }
        }
        
        diffQuadSum *= invVare;
        logDelta0MinusLogDelta1 = -0.5f*diffQuadSum + logPiComp - logPi;
        
        // terms due to the presence of delta in the scale factor for the effect variance
//        if (sum2pqOneMinusS > 0) {
//            logDelta0MinusLogDelta1 += -2.0f*logf(sum2pqOneMinusS) - vg/(sigmaSq*sum2pqOneMinusS);
//            logDelta0MinusLogDelta1 -= -2.0f*logf(sum2pqOneMinusS+snp2pqOneMinusS) - vg/(sigmaSq*(sum2pqOneMinusS+snp2pqOneMinusS));
//        }
        // end
        
        probDelta1 = 1.0f/(1.0f + expf(logDelta0MinusLogDelta1));
        
        if (bernoulli.sample(probDelta1)) {
            if (!windDelta[i]) {
                for (unsigned j=start; j<end; ++j) {
                    if (snpDelta[j]) {
                        ycorr -= Z.col(j) * beta[j];
                    }
                }
            }
//            sum2pqOneMinusS += snp2pqOneMinusS;
            windDelta[i] = 1.0;
            ++numNonZeroWind;

            for (unsigned j=start; j<end; ++j) {
                oldSample = beta[j]*snpDelta[j];
                my_rhs = Z.col(j).dot(ycorr);
                
                MPI_Allreduce(&my_rhs, &rhs, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                
                rhs += ZPZdiag[j]*oldSample;
                rhs *= invVare;
                invLhs = 1.0f/(ZPZdiag[j]*invVare + invSigmaSq/snp2pqPowS[j]);
                uhat = invLhs*rhs;
                logDelta1 = 0.5*(logf(invLhs) - logf(snp2pqPowS[j]*sigmaSq) + uhat*rhs) + logLocalPi[i];
                logDelta0 = logLocalPiComp[i];
                
                // terms due to the presence of delta in the scale factor for the effect variance
//                snp2pqOneMinusS =  snp2pq[i]/snp2pqPowS[i];
//                if (oldSample) sum2pqOneMinusS -= snp2pqOneMinusS;
//                if (sum2pqOneMinusS > 0) {
//                    logDelta1 += -2.0f*logf(sum2pqOneMinusS+snp2pqOneMinusS) - vg/(sigmaSq*(sum2pqOneMinusS+snp2pqOneMinusS));
//                    logDelta0 += -2.0f*logf(sum2pqOneMinusS) - vg/(sigmaSq*sum2pqOneMinusS);
//                }
                // end

                probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
                
                if (bernoulli.sample(probDelta1)) {
                    values[j] = beta[j] = normal.sample(uhat, invLhs);
                    ycorr += Z.col(j) * (oldSample - values[j]);
                    ghat  += Z.col(j) * values[j];
                    wtdSumSq += values[j]*values[j]/snp2pqPowS[j];
//                    sum2pqOneMinusS += snp2pqOneMinusS;
                    snpDelta[j] = 1.0;
                    ++cumDelta[j];
                    ++numNonZeros;
                } else {
                    if (oldSample) ycorr += Z.col(j) * oldSample;
                    beta[j] = normal.sample(0.0, snp2pqPowS[j]*sigmaSq);
//                    if (iter < burnin) beta[j] = normal.sample(0.0, sigmaSq/snp2pqPowS[j]);
//                    else beta[j] = normal.sample(0.0, varPseudoPrior[j]);
                    snpDelta[j] = 0.0;
                    values[j] = 0.0;
                }
            }
        }
        else {
//            unsigned windSize = end-start;
//            float localSum = cumDelta.segment(start,windSize).sum();
            for (unsigned j=start; j<end; ++j) {
                beta[j] = normal.sample(0.0, snp2pqPowS[j]*sigmaSq);
//                if (iter < burnin) beta[j] = normal.sample(0.0, sigmaSq/snp2pqPowS[j]);
//                else beta[j] = normal.sample(0.0, varPseudoPrior[j]);
                snpDelta[j] = bernoulli.sample(localPi[i]);
//                float seudopi = (localPi[i]/(windSize-1)+cumDelta[j])/(localPi[i]+localSum-cumDelta[j]);
//                snpDelta[j] = bernoulli.sample(seudopi);
                if (values[j]) ycorr += Z.col(j) * values[j];
                values[j] = 0.0;
            }
            windDelta[i] = 0.0;
        }
    }
}

void BayesNS::Sp::sampleFromFC(const unsigned numNonZeros, const float sigmaSq, const VectorXf &snpEffects, const VectorXf &snp2pq, ArrayXf &snp2pqPowS, const ArrayXf &logSnp2pq){
    // do not update scale factor of sigmaSq
    
    // Prepare
    ArrayXf snpEffectDelta1(numNonZeros);
    ArrayXf snp2pqDelta1(numNonZeros);
    ArrayXf logSnp2pqDelta1(numNonZeros);
    
    for (unsigned i=0, j=0; i<numSnps; ++i) {
        if (snpEffects[i]) {
            snpEffectDelta1[j] = snpEffects[i];
            snp2pqDelta1[j] = snp2pq[i];
            logSnp2pqDelta1[j] = logSnp2pq[i];
            ++j;
        }
    }
    
    float snp2pqLogSumDelta1 = logSnp2pqDelta1.sum();
    
    float curr = value;
    float curr_p = Stat::snorm();
    
    float cand = curr;
    // Make a half step for momentum at the beginning
    float cand_p = curr_p - 0.5*stepSize * gradientU(curr,  snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq);
    
    for (unsigned i=0; i<numSteps; ++i) {
        // Make a full step for the position
        cand += stepSize * cand_p;
        if (i < numSteps-1) {
            // Make a full step for the momentum, except at end of trajectory
            cand_p -= stepSize * gradientU(cand, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq);
        } else {
            // Make a half step for momentum at the end
            cand_p -= 0.5*stepSize * gradientU(cand, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq);
        }
    }

    // Evaluate potential (negative log posterior) and kinetic energies at start and end of trajectory
    float curr_H = computeU(curr, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq) + 0.5*curr_p*curr_p;
    float cand_H = computeU(cand, snpEffectDelta1, snp2pqLogSumDelta1, snp2pqDelta1, logSnp2pqDelta1, sigmaSq) + 0.5*cand_p*cand_p;
    
    if (Stat::ranf() < exp(curr_H-cand_H)) {  // accept
        value = cand;
        snp2pqPowS = snp2pq.array().pow(cand);
        ar.count(1, 0.5, 0.9);
    } else {
        ar.count(0, 0.5, 0.9);
    }
    
    if      (ar.value < 0.5) stepSize *= 0.8;
    else if (ar.value > 0.9) stepSize *= 1.2;
    
    tuner.value = stepSize;
}

float BayesNS::Sp::gradientU(const float S, const ArrayXf &snpEffects, const float snp2pqLogSum,
                             const ArrayXf &snp2pq, const ArrayXf &logSnp2pq, const float sigmaSq){
    // compute the first derivative of the negative log posterior
    return 0.5*snp2pqLogSum - 0.5/sigmaSq*(snpEffects.square()*logSnp2pq/snp2pq.pow(S)).sum() + S/var;
}

float BayesNS::Sp::computeU(const float S, const ArrayXf &snpEffects, const float snp2pqLogSum,
                            const ArrayXf &snp2pq, const ArrayXf &logSnp2pq, const float sigmaSq){
    // compute negative log posterior
//    return -0.5*S*snp2pqLogSum + 0.5/sigmaSq*(snpEffects.square()*logSnp2pq*snp2pq.pow(S)).sum() + 0.5*S*S;
    return 0.5*S*snp2pqLogSum + 0.5/sigmaSq*(snpEffects.square()/snp2pq.pow(S)).sum() + 0.5*S*S/var;
}

void BayesNS::sampleUnknowns(){
    static unsigned iter = 0;
    
    fixedEffects.sampleFromFC(ycorr, data.X, data.XPXdiag, vare.value);
    
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(ycorr, data.Z, data.ZPZdiag, sigmaSq.value, pi.value, vare.value, snp2pqPowS, data.snp2pq, genVarPrior, sigmaSq.scale, ghat);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);
    
    sigmaSq.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros);
    
    //scale.sampleFromFC(sigmaSq.value, sigmaSq.df, sigmaSq.scale);
    
    if (estimatePi) pi.sampleFromFC(snpEffects.numWindows, snpEffects.numNonZeroWind);
    vare.sampleFromFC(ycorr);
    
    //S.sampleFromFC(snpEffects.numNonZeros, sigmaSq.value, snpEffects.values, data.snp2pq, snp2pqPowS, logSnp2pq);
    S.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros, sigmaSq.value, snpEffects.values, data.snp2pq, snp2pqPowS, logSnp2pq, genVarPrior, sigmaSq.scale, snpEffects.sum2pqOneMinusS);
    if (iter >= 2000) sigmaSq.scale = scalePrior;
    scale.getValue(sigmaSq.scale);
    
    varg.compute(ghat);
    hsq.compute(varg.value, vare.value);
    
    rounding.computeYcorr(data.y, data.X, data.Z, fixedEffects.values, snpEffects.values, ycorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
    nnzWind.getValue(snpEffects.numNonZeroWind);
    windDelta.getValues(snpEffects.windDelta);
    
    if (++iter < 2000) {
        genVarPrior += (varg.value - genVarPrior)/iter;
        scalePrior += (sigmaSq.scale - scalePrior)/iter;
    }
}


void ApproxBayesC::FixedEffects::sampleFromFC(const MatrixXf &XPX, const VectorXf &XPXdiag,
                                              const MatrixXf &ZPX, const VectorXf &XPy,
                                              const VectorXf &snpEffects, const float vare,
                                              VectorXf &rcorr){
    for (unsigned i=0; i<size; ++i) {
        float oldSample = values[i];
        float XPZa = ZPX.col(i).dot(snpEffects);
        float rhs = XPy[i] - XPZa - XPX.row(i).dot(values) + XPXdiag[i]*values[i];
        float invLhs = 1.0f/XPXdiag[i];
        float bhat = invLhs*rhs;
        values[i] = Normal::sample(bhat, invLhs*vare);
        rcorr += ZPX.col(i) * (oldSample - values[i]);
    }

}

void ApproxBayesC::SnpEffects::sampleFromFC(VectorXf &rcorr,const vector<VectorXf> &ZPZ, const VectorXf &ZPZdiag, const VectorXf &ZPy,
                                            const VectorXi &windStart, const VectorXi &windSize, const vector<ChromInfo*> &chromInfoVec,
                                            const VectorXf &se, VectorXf &sse, const VectorXf &n, const VectorXf &snp2pq,
                                            const float sigmaSq, const float pi, const float vare){
    
    static unsigned iter = 0;
    long numChr = chromInfoVec.size();
    
    VectorXf ssq, s2pq, nnz;
    ssq.setZero(numChr);
    s2pq.setZero(numChr);
    nnz.setZero(numChr);

#pragma omp parallel for
    for (unsigned chr=0; chr<numChr; ++chr) {
        //cout << " thread " << omp_get_thread_num() << " chr " << chr << endl;
        
        ChromInfo *chromInfo = chromInfoVec[chr];
        unsigned chrStart = chromInfo->startSnpIdx;
        unsigned chrEnd   = chromInfo->endSnpIdx;
        unsigned windEnd, j;
        
        float oldSample;
        float rhs, invLhs, uhat;
        float logDelta0, logDelta1, probDelta1;
        float logPi = log(pi);
        float logPiComp = log(1.0-pi);
        float logSigmaSq = log(sigmaSq);
        float varei;
        
        for (unsigned i=chrStart; i<=chrEnd; ++i) {
            //cout << i << " " << chrStart << " " << chrEnd << " " << ssq << " " << nnz << endl;
            if (!(iter % 100)) {
                //float varei = (sse[i] - values.segment(windStart[i], windSize[i]).dot(ZPy.segment(windStart[i], windSize[i]) + rcorr.segment(windStart[i], windSize[i])))/n[i];
                windEnd = windStart[i] + windSize[i];
                varei = sse[i];
                for (j=windStart[i]; j<windEnd; ++j) {
                    if (values[j]) varei -= values[j]*(ZPy[j] + rcorr[j]);
                }
                varei /= n[i];
            } else {
                varei = sse[i]/n[i];
            }
//            varei = se[i]*se[i]*ZPZdiag[i];
            oldSample = values[i];
            rhs = rcorr[i] + ZPZdiag[i]*oldSample;
            rhs /= varei;
            invLhs = 1.0f/(ZPZdiag[i]/varei + 1.0f/sigmaSq);
            uhat = invLhs*rhs;
            logDelta1 = 0.5*(logf(invLhs) - logSigmaSq + uhat*rhs) + logPi;
            logDelta0 = logPiComp;
            probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
            //cout << rhs << " " << invLhs << " " << logDelta1 << " " << logSigmaSq << " " << sigmaSq << endl;
            if (bernoulli.sample(probDelta1)) {
                values[i] = normal.sample(uhat, invLhs);
                rcorr.segment(windStart[i], windSize[i]) += ZPZ[i]*(oldSample - values[i]);
                ssq[chr] += values[i]*values[i];
                s2pq[chr] += snp2pq[i];
                ++nnz[chr];
            } else {
                if (oldSample) rcorr.segment(windStart[i], windSize[i]) += ZPZ[i]*oldSample;
                values[i] = 0.0;
            }
        }
    }
    
    //cout << ssq << " " << nnz << endl;
    
    sumSq = ssq.sum();
    sum2pq = s2pq.sum();
    numNonZeros = nnz.sum();
    ++iter;
}

void ApproxBayesC::SnpEffects::hmcSampler(VectorXf &rcorr, const VectorXf &ZPy, const vector<VectorXf> &ZPZ,
                                            const VectorXi &windStart, const VectorXi &windSize, const vector<ChromInfo*> &chromInfoVec,
                                            const float sigmaSq, const float pi, const float vare){
    
    float stepSize = 0.01;
    unsigned numSteps = 100;
    
    
#pragma omp parallel for
    for (unsigned chr=0; chr<chromInfoVec.size(); ++chr) {
        //cout << " thread " << omp_get_thread_num() << " chr " << chr << endl;
        
        ChromInfo *chromInfo = chromInfoVec[chr];
        unsigned chrStart = chromInfo->startSnpIdx;
        unsigned chrEnd   = chromInfo->endSnpIdx;
        unsigned chrSize  = chromInfo->size;
        
        VectorXf chrZPy = ZPy.segment(chrStart, chrSize);
        VectorXi chrWindStart = windStart.segment(chrStart, chrSize);
        VectorXi chrWindSize = windSize.segment(chrStart, chrSize);
        chrWindStart.array() -= chrStart;
        
        VectorXf curr = values.segment(chrStart, chrSize);
        VectorXf curr_p(chrSize);
        
        for (unsigned i=0; i<chrSize; ++i) {
            curr_p[i] = Stat::snorm();
        }
        
        VectorXf cand = curr;
        // Make a half step for momentum at the beginning
        VectorXf rc = chrZPy;
        VectorXf cand_p = curr_p - 0.5*stepSize * gradientU(curr, rc, chrZPy, ZPZ, chrWindStart, chrWindSize, chrStart, chrSize, sigmaSq, vare);
        
        for (unsigned i=0; i<numSteps; ++i) {
            cand += stepSize * cand_p;
            if (i < numSteps-1) {
                cand_p -= stepSize * gradientU(cand, rc, chrZPy, ZPZ, chrWindStart, chrWindSize, chrStart, chrSize, sigmaSq, vare);
            } else {
                cand_p -= 0.5* stepSize * gradientU(cand, rc, chrZPy, ZPZ, chrWindStart, chrWindSize, chrStart, chrSize, sigmaSq, vare);
            }
        }
        
        float curr_H = computeU(curr, rcorr.segment(chrStart, chrSize), chrZPy, sigmaSq, vare) + 0.5*curr_p.squaredNorm();
        float cand_H = computeU(cand, rc, chrZPy, sigmaSq, vare) + 0.5*cand_p.squaredNorm();
        
        if (Stat::ranf() < exp(curr_H-cand_H)) {  // accept
            values.segment(chrStart, chrSize) = cand;
            rcorr.segment(chrStart, chrSize) = rc;
            ++mhr;
        }
    }
    
    sumSq = values.squaredNorm();
    numNonZeros = size;
    
    if (!(++cnt % 100) && myMPI::rank==0) {
        float ar = mhr/float(cnt*22);
        if      (ar < 0.5) cout << "Warning: acceptance rate for SNP effects is too low "  << ar << endl;
        else if (ar > 0.9) cout << "Warning: acceptance rate for SNP effects is too high " << ar << endl;
    }

}

VectorXf ApproxBayesC::SnpEffects::gradientU(const VectorXf &effects, VectorXf &rcorr, const VectorXf &ZPy, const vector<VectorXf> &ZPZ, 
                                             const VectorXi &windStart, const VectorXi &windSize, const unsigned chrStart, const unsigned chrSize,
                                             const float sigmaSq, const float vare){
    rcorr = ZPy;
    for (unsigned i=0; i<chrSize; ++i) {
        if (effects[i]) {
            rcorr.segment(windStart[i], windSize[i]) -= ZPZ[chrStart+i]*effects[i];
        }
    }
    return -rcorr/vare + effects/sigmaSq;
}

float ApproxBayesC::SnpEffects::computeU(const VectorXf &effects, const VectorXf &rcorr, const VectorXf &ZPy,                                             const float sigmaSq, const float vare){
    return -0.5f/vare*effects.dot(ZPy+rcorr) + 0.5/sigmaSq*effects.squaredNorm();
}

//void ApproxBayesC::ResidualVar::sampleFromFC(VectorXf &rcorr, const SparseMatrix<float> &ZPZinv){ // this would not work if ZPZ has no full rank, e.g. exist SNPs in complete LD.
//    float sse = rcorr.transpose()*ZPZinv*rcorr;
//    float dfTilde = df + nobs;
//    float scaleTilde = sse + df*scale;
//    value = InvChiSq::sample(dfTilde, scaleTilde);
//}

void ApproxBayesC::ResidualVar::sampleFromFC(const float ypy, const VectorXf &effects, const VectorXf &ZPy, const VectorXf &rcorr){
    float modelSS = effects.dot(ZPy) - effects.dot(rcorr);
    float sse = ypy - modelSS;
    if (sse < 0) sse = 0.0;
    if (sse > ypy) sse = ypy;
    float dfTilde = df + nobs;
    float scaleTilde = sse + df*scale;
    value = InvChiSq::sample(dfTilde, scaleTilde);
}

void ApproxBayesC::GenotypicVar::compute(const VectorXf &effects, const VectorXf &ZPy, const VectorXf &rcorr){
    float modelSS = effects.dot(ZPy) - effects.dot(rcorr);
    value = modelSS/nobs;
}

void ApproxBayesC::Rounding::computeRcorr(const VectorXf &ZPy, const vector<VectorXf> &ZPZ,
                                          const VectorXi &windStart, const VectorXi &windSize, const vector<ChromInfo*> &chromInfoVec,
                                          const VectorXf &snpEffects, VectorXf &rcorr){
    if (count++ % 100) return;
    VectorXf rcorrOld = rcorr;
    rcorr = ZPy;
#pragma omp parallel for
    for (unsigned chr=0; chr<chromInfoVec.size(); ++chr) {
        ChromInfo *chromInfo = chromInfoVec[chr];
        unsigned chrStart = chromInfo->startSnpIdx;
        unsigned chrEnd   = chromInfo->endSnpIdx;
        for (unsigned i=chrStart; i<=chrEnd; ++i) {
            rcorr.segment(windStart[i], windSize[i]) -= ZPZ[i]*snpEffects[i];
        }
    }
    value = sqrt(Gadget::calcVariance(rcorrOld-rcorr));
}


void ApproxBayesC::sampleUnknowns(){
    fixedEffects.sampleFromFC(data.XPX, data.XPXdiag, data.ZPX, data.XPy, snpEffects.values, vare.value, rcorr);
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(rcorr, data.ZPZ, data.ZPZdiag, data.ZPy, data.windStart, data.windSize, data.chromInfoVec, data.se, sse, data.n, data.snp2pq, sigmaSq.value, pi.value, vare.value);
        //snpEffects.hmcSampler(rcorr, data.ZPy, data.ZPZ, data.windStart, data.windSize, data.chromInfoVec, sigmaSq.value, pi.value, vare.value);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);
    sigmaSq.sampleFromFC(snpEffects.sumSq, snpEffects.numNonZeros);
    if (estimatePi) pi.sampleFromFC(data.numIncdSnps, snpEffects.numNonZeros);
    vare.sampleFromFC(data.ypy, snpEffects.values, data.ZPy, rcorr);
    varg.compute(snpEffects.values, data.ZPy, rcorr);
    hsq.compute(varg.value, vare.value);
    rounding.computeRcorr(data.ZPy, data.ZPZ, data.windStart, data.windSize, data.chromInfoVec, snpEffects.values, rcorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
    sigmaSqG.compute(sigmaSq.value, snpEffects.sum2pq);
}



void ApproxBayesS::SnpEffects::sampleFromFC(VectorXf &rcorr,const vector<VectorXf> &ZPZ, const VectorXf &ZPZdiag, const VectorXf &ZPy,
                                            const VectorXi &windStart, const VectorXi &windSize, const vector<ChromInfo*> &chromInfoVec,
                                            const float sigmaSq, const float pi, const float vare,
                                            const VectorXf &snp2pqPowS, const VectorXf &snp2pq,
                                            const VectorXf &se, VectorXf &sse, const VectorXf &n,
                                            const float vg, float &scale){
    static unsigned iter = 0;
    long numChr = chromInfoVec.size();
    
    float ssq[numChr], nnz[numChr];
    memset(ssq,0,sizeof(float)*numChr);
    memset(nnz,0, sizeof(float)*numChr);
    //ssq.setZero(numChr);
    //nnz.setZero(numChr);
    
    for (unsigned chr=0; chr<numChr; ++chr) {
        ChromInfo *chromInfo = chromInfoVec[chr];
        unsigned chrStart = chromInfo->startSnpIdx;
        unsigned chrEnd   = chromInfo->endSnpIdx;
        if (iter==0) {
            cout << "chr " << chr+1 << " start " << chrStart << " end " << chrEnd << endl;
        }
    }
    
    float *valueTmp = values.data();
    
    vector<float> urnd(size), nrnd(size);
    for (unsigned i=0; i<size; ++i) {
        urnd[i] = Stat::ranf();
        nrnd[i] = Stat::snorm();
    }
    
#pragma omp parallel for
    for (unsigned chr=0; chr<numChr; ++chr) {
        //cout << " thread " << omp_get_thread_num() << " chr " << chr << endl;
        
        ChromInfo *chromInfo = chromInfoVec[chr];
        unsigned chrStart = chromInfo->startSnpIdx;
        unsigned chrEnd   = chromInfo->endSnpIdx;
        unsigned chrSize  = chrEnd - chrStart + 1;
        unsigned windEnd, j;
        
        float oldSample;
        float rhs, invLhs, uhat;
        float logDelta0, logDelta1, probDelta1;
        float logPi = log(pi);
        float logPiComp = log(1.0-pi);
        float invSigmaSq = 1.0f/sigmaSq;
        float varei;
        //float snp2pqOneMinusS;
        
        for (unsigned i=chrStart; i<=chrEnd; ++i) {
            oldSample = valueTmp[i];
            
            //float varei = se[i]*se[i]*ZPZdiag[i];
            
            if (!(iter % 100)) {
                //float varei = (sse[i] - values.segment(windStart[i], windSize[i]).dot(ZPy.segment(windStart[i], windSize[i]) + rcorr.segment(windStart[i], windSize[i])))/n[i];
                windEnd = windStart[i] + windSize[i];
                varei = sse[i];
                for (j=windStart[i]; j<windEnd; ++j) {
                    if (values[j]) varei -= valueTmp[j]*(ZPy[j] + rcorr[j]);
                }
                varei /= n[i];
            } else {
                varei = sse[i]/n[i];
            }
                
            //cout << i << " " << varei << " " << se[i]*se[i]*ZPZdiag[i] << " " << (sse[i] - values.segment(windStart[i], windSize[i]).dot(ZPy.segment(windStart[i], windSize[i]) + rcorr.segment(windStart[i], windSize[i])))/n[i] << " " << values.segment(windStart[i], windSize[i]).dot(ZPy.segment(windStart[i], windSize[i]))
            //<< " " << values.segment(windStart[i], windSize[i]).dot(rcorr.segment(windStart[i], windSize[i])) << endl;

            rhs  = rcorr[i] + ZPZdiag[i]*oldSample;
            rhs /= varei;
            invLhs = 1.0f/(ZPZdiag[i]/varei + invSigmaSq/snp2pqPowS[i]);
            uhat = invLhs*rhs;
            logDelta1 = 0.5*(logf(invLhs) - logf(snp2pqPowS[i]*sigmaSq) + uhat*rhs) + logPi;
            logDelta0 = logPiComp;
            
            //        // terms due to the presence of delta in the scale factor for the effect variance
            //            snp2pqOneMinusS =  snp2pq[i]/snp2pqPowS[i];
            //            if (oldSample) sum2pqOneMinusS -= snp2pqOneMinusS;
            //        logDelta1 += -2.0f*logf(sum2pqOneMinusS+snp2pqOneMinusS) - vg/(sigmaSq*(sum2pqOneMinusS+snp2pqOneMinusS));
            //        logDelta0 += -2.0f*logf(sum2pqOneMinusS) - vg/(sigmaSq*sum2pqOneMinusS);
            //        // end
            
            probDelta1 = 1.0f/(1.0f + expf(logDelta0-logDelta1));
            //if(i==0) cout << i << " chrStart " << chrStart << " chrEnd " << chrEnd << " windStart " << windStart[i] << " windSize " << windSize[i] << " rcorr " << rcorr[i] << " ZPZdiag " << ZPZdiag[i] << " vare " << vare << " sigmaSq " << sigmaSq <<  " snp2pq " << snp2pq[i] <<  " logDelta0 " << logDelta0 << " logDelta1 " << logDelta1 << " probDelta1 " << probDelta1 << endl;
            
            //if (bernoulli.sample(probDelta1)) {
            if(urnd[i] < probDelta1) {
                //valueTmp[i] = normal.sample(uhat, invLhs);
                valueTmp[i] = uhat + nrnd[i]*sqrtf(invLhs);
                rcorr.segment(windStart[i], windSize[i]) += ZPZ[i]*(oldSample - values[i]);
                //sse.segment(windStart[i], windSize[i]) += (ZPy.segment(windStart[i], windSize[i]) + rcorr.segment(windStart[i], windSize[i]))*(oldSample - values[i]);
                //sum2pqOneMinusS += snp2pqOneMinusS;
                ssq[chr] += valueTmp[i]*valueTmp[i]/snp2pqPowS[i];
                ++nnz[chr];
            } else {
                if (oldSample) {
                    rcorr.segment(windStart[i], windSize[i]) += ZPZ[i]*oldSample;
                    //sse.segment(windStart[i], windSize[i]) += (ZPy.segment(windStart[i], windSize[i]) + rcorr.segment(windStart[i], windSize[i]))*oldSample;
                }
                valueTmp[i] = 0.0;
            }
        }
    }
    
    //wtdSumSq = ssq.sum();
    //numNonZeros = nnz.sum();
    wtdSumSq = 0.0;
    numNonZeros = 0;
    for (unsigned i=0; i<numChr; ++i) {
        wtdSumSq += ssq[i];
        numNonZeros += nnz[i];
    }
    ++iter;
    
    values = VectorXf::Map(valueTmp, size);
}


void ApproxBayesS::sampleUnknowns(){
    static unsigned iter = 0;

    fixedEffects.sampleFromFC(data.XPX, data.XPXdiag, data.ZPX, data.XPy, snpEffects.values, vare.value, rcorr);
    
    unsigned cnt=0;
    do {
        snpEffects.sampleFromFC(rcorr, data.ZPZ, data.ZPZdiag, data.ZPy, data.windStart, data.windSize, data.chromInfoVec, sigmaSq.value, pi.value, vare.value, snp2pqPowS, data.snp2pq, data.se, sse, data.n, genVarPrior, sigmaSq.scale);
        if (++cnt == 100) throw("Error: Zero SNP effect in the model for 100 cycles of sampling");
    } while (snpEffects.numNonZeros == 0);

    if(estimatePi) pi.sampleFromFC(data.numIncdSnps, snpEffects.numNonZeros);
    
    sigmaSq.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros);
    vare.sampleFromFC(data.ypy, snpEffects.values, data.ZPy, rcorr);
    
//    VectorXf ZPZb(data.numIncdSnps);
//    for (unsigned i=0; i<data.numIncdSnps; ++i) {
//        ZPZb.segment(data.windStart[i], data.windSize[i]) += data.ZPZ[i]*snpEffects.values[i];
//    }
//    float modelSS = 2.0f*snpEffects.values.dot(data.ZPy) - snpEffects.values.dot(ZPZb);
//    float sse = data.ypy - modelSS;
//    if (sse < 0) sse = 0.0;
//    if (sse > data.ypy) sse = data.ypy;
//    float dfTilde = vare.df + vare.nobs;
//    float scaleTilde = sse + vare.df*vare.scale;
//    vare.value = vare.InvChiSq::sample(dfTilde, scaleTilde);
//    varg.value = modelSS/varg.nobs;
    
    varg.compute(snpEffects.values, data.ZPy, rcorr);
    hsq.compute(varg.value, vare.value);
    
    S.sampleFromFC(snpEffects.wtdSumSq, snpEffects.numNonZeros, sigmaSq.value, snpEffects.values, data.snp2pq, snp2pqPowS, logSnp2pq, genVarPrior, sigmaSq.scale, snpEffects.sum2pqOneMinusS);

    if (iter >= 2000) sigmaSq.scale = scalePrior;
    scale.getValue(sigmaSq.scale);

    rounding.computeRcorr(data.ZPy, data.ZPZ, data.windStart, data.windSize, data.chromInfoVec, snpEffects.values, rcorr);
    nnzSnp.getValue(snpEffects.numNonZeros);
    sigmaSqG.compute(sigmaSq.value, snpEffects.sum2pqOneMinusS);
    
    if (++iter < 2000) {
        genVarPrior += (varg.value - genVarPrior)/iter;
        scalePrior += (sigmaSq.scale - scalePrior)/iter;
    }
}

