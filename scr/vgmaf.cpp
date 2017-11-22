//
//  vgmaf.cpp
//  gctb
//
//  Created by Jian Zeng on 23/11/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "vgmaf.hpp"

void VGMAF::makeMafBin(const Data &data, vector<float> &mafbin, vector<vector<int> > &snpIndex){
    // seq(0,0.5,0.01)
    mafbin.clear();
    for (float i=0.01; i<=0.5; i+=0.01) {
        mafbin.push_back(i);
    }
    
    numBins = (unsigned) mafbin.size();
    numSnps = data.numIncdSnps;
    snpIndex.resize(numBins);
    
    SnpInfo *snp = NULL;
    for (unsigned i=0; i<data.numIncdSnps; ++i) {
        snp = data.incdSnpInfoVec[i];
        float maf = snp->af < 0.5 ? snp->af : 1.0 - snp->af;
        for (unsigned j=0; j<numBins; ++j) {
            if (maf < mafbin[j]) {
                snpIndex[j].push_back(snp->index);
                break;
            }
        }
    }
}

void VGMAF::calcCumVarGen(const MatrixXf &Z, const vector<SnpInfo*> &snpVec, const McmcSamples &snpEffects, const unsigned burnin, const unsigned thin,
                          const vector<vector<int> > &snpIndex, MatrixXf &cumVarGen, VectorXf &cumVarGenMean){
    long nobs = Z.rows();
    long niter = snpEffects.datMatSp.rows();
    unsigned burninThined = burnin/thin;
    long postburnin = niter - burninThined;
    cout << " thin " << thin << " chainLength: " << niter << " burnin: " << burninThined << " postburnin: " << postburnin << endl;
    VectorXf ghat;
    cumVarGen.setZero(postburnin, numBins);
    cumPi.setZero(postburnin, numBins);
    meanBeta.setZero(postburnin, numBins);
    nnz.setZero(postburnin, numBins);
    auc.setZero(postburnin);
    for (unsigned iter=0; iter<niter; ++iter) {
        if (iter < burninThined) continue;
        unsigned iterPostburnin = iter-burninThined;
        ghat.setZero(nobs);
        float nonzero = 0;
        for (unsigned i=0; i<numBins; ++i) {
            float nonzeroBini = 0;
            long binSize = snpIndex[i].size();
            for (unsigned j=0; j<binSize; ++j) {
                int idx = snpIndex[i][j];
                float sample = snpEffects.datMatSp.coeff(iter, idx);
                if (sample) {
                    ghat += Z.col(idx) * sample;
                    if (snpVec[idx]->af < 0.5) meanBeta(iterPostburnin, i) += sample;
                    else meanBeta(iterPostburnin, i) += -sample;
                    ++nonzeroBini;
                }
            }
            nonzero += nonzeroBini;
            nnz(iterPostburnin, i) = nonzeroBini;
            cumPi(iterPostburnin, i) = nonzero;
            meanBeta(iterPostburnin, i) /= float(binSize);
            //cumVarGen(iterPostburnin, i) = Gadget::calcVariance(ghat);  // TODO MPI
            float my_sum = ghat.sum();
            float my_ssq = ghat.squaredNorm();
            unsigned my_size = (unsigned)ghat.size();
            float sum, ssq;
            unsigned size;
            MPI_Allreduce(&my_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&my_ssq, &ssq, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&my_size, &size, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            float mean = sum/size;
            cumVarGen(iterPostburnin, i) = ssq/size - mean*mean;
            float area = 0.01*cumVarGen(iterPostburnin, i);
            if (i==0) area *= 0.5;
            else area -= 0.5*0.01*(cumVarGen(iterPostburnin, i) - cumVarGen(iterPostburnin, i-1));
            auc[iterPostburnin] += area;
        }
        auc[iterPostburnin] /= 0.5*cumVarGen(iterPostburnin, numBins-1);
        cumVarGen.row(iterPostburnin) /= cumVarGen(iterPostburnin, numBins-1);
        cumPi.row(iterPostburnin) /= nonzero;
    }
    cumVarGenMean = cumVarGen.colwise().mean();
    //cout << cumVarGen.block(0, 0, nsamples, 10) << endl;
    cumVarGenSD   = (cumVarGen.colwise().squaredNorm().array()/float(postburnin) - cumVarGenMean.array().square().transpose()).sqrt().matrix();
    
    cumPiMean = cumPi.colwise().mean();
    cumPiSD = ((cumPi.rowwise() - cumPiMean.transpose()).colwise().squaredNorm()/float(postburnin)).cwiseSqrt();
    
    meanBetaMean = meanBeta.colwise().mean();
    meanBetaSD = ((meanBeta.rowwise() - meanBetaMean.transpose()).colwise().squaredNorm()/float(postburnin)).cwiseSqrt();
    
    nnzMean = nnz.colwise().mean();
    nnzSD = ((nnz.rowwise() - nnzMean.transpose()).colwise().squaredNorm()/float(postburnin)).cwiseSqrt();
    
    cout << " AUC with SD: " << auc.mean() << "  " << sqrt(Gadget::calcVariance(auc)) << endl;
}

void VGMAF::outputRes(const string &title){
    if (myMPI::rank) return;
    string filename = title + ".vgmaf";
    ofstream out(filename);
    out << boost::format("%8s %8s %8s %6s %6s %8s %8s %8s %8s %8s %8s\n")
    %"Maf" %"NumSNPs" %"CumDist" %"NNZ" %"NNZSD" %"CumPi" %"CumPiSD" %"CumVarMean" %"CumVarSD" %"MeanBeta" %"MeanBetaSD";
    unsigned cumsum = 0;
    for (unsigned i=0; i<numBins; ++i) {
        cumsum += snpIndex[i].size();
        out << boost::format("%8.3f %8s %8.3f %6.1f %6.1f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n")
        % mafbin[i]
        % snpIndex[i].size()
        % (cumsum/float(numSnps))
        % nnzMean[i]
        % nnzSD[i]
        % cumPiMean[i]
        % cumPiSD[i]
        % cumVarGenMean[i]
        % cumVarGenSD[i]
        % meanBetaMean[i]
        % meanBetaSD[i];
    }
    out.close();
    cout << "Results are saved in [" << filename << "]" << endl;
    
    filename = title + ".auc";
    ofstream out2(filename);
    out2 << auc.mean() << "\t" << sqrt(Gadget::calcVariance(auc)) << endl;
}

void VGMAF::compute(const Data &data, const McmcSamples &snpEffects, const unsigned burnin, const unsigned thin, const string &title){
    if (myMPI::rank==0) cout << "Calculating cumulative genetic variances for MAF bins ..." << endl;
    makeMafBin(data, mafbin, snpIndex);
    calcCumVarGen(data.Z, data.incdSnpInfoVec, snpEffects, burnin, thin, snpIndex, cumVarGen, cumVarGenMean);
    outputRes(title);
}

void VGMAF::simulate(const Data &data, const string &title){
    if (myMPI::rank==0) cout << "Simulating data and calculating cumulative genetic variances for MAF bins ..." << endl;
    makeMafBin(data, mafbin, snpIndex);
    vector<float> svalue = {-1.0, -0.5, 0.0, 0.5, 1.0};
    //vector<float> svalue = {0};
    long numSValues = svalue.size();
    cumVarGen.setZero(numBins, numSValues);
    MatrixXf simEffect(data.numIncdSnps, numSValues);
    for (unsigned i=0; i<data.numIncdSnps; ++i) {
        float rn = Stat::snorm();
        for (unsigned j=0; j<numSValues; ++j) {
            //float rn = Stat::snorm();
            simEffect(i,j) = rn*powf(data.snp2pq[i], -0.5*svalue[j]);
        }
    }
    
    long nobs = data.Z.rows();
    VectorXf ghat;
    for (unsigned col=0; col<numSValues; ++col) {
        ghat.setZero(nobs);
        //float sumVar = 0;
        for (unsigned i=0; i<numBins; ++i) {
            long binSize = snpIndex[i].size();
            //cout << endl << i << " " << binSize;
            for (unsigned j=0; j<binSize; ++j) {
                int idx = snpIndex[i][j];
                //cout << " " << idx;
                ghat += data.Z.col(idx) * simEffect(idx, col);
                //sumVar += data.snp2pq[idx] * simEffect(idx, col)*simEffect(idx, col);
            }
            float my_sum = ghat.sum();
            float my_ssq = ghat.squaredNorm();
            unsigned my_size = (unsigned)ghat.size();
            float sum, ssq;
            unsigned size;
            MPI_Allreduce(&my_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&my_ssq, &ssq, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&my_size, &size, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            float mean = sum/size;
            cumVarGen(i, col) = ssq/size - mean*mean;
            
            //cumVarGen(i, col) = sumVar;
        }
        cumVarGen.col(col) /= cumVarGen(numBins-1, col);
        //cout << endl << cumVarGen.col(col).transpose() << endl;
    }
    
    if (myMPI::rank) return;
    string filename = title + ".sim.vgmaf";
    ofstream out(filename);
    out << boost::format("%8s %8s %6s %8s\n") %"MAF" %"NumSNPs" %"S" %"CumVar";
    for (unsigned i=0; i<numSValues; ++i) {
        for (unsigned j=0; j<numBins; ++j) {
            out << boost::format("%8.3f %8s %6.1f %8.6f\n")
            % mafbin[j]
            % snpIndex[j].size()
            % svalue[i]
            % cumVarGen(j,i);
        }
    }
    out.close();
    cout << "Results are saved in [" << filename << "]" << endl;
}
