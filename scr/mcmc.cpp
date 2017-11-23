//
//  mcmc.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "mcmc.hpp"

void McmcSamples::getSample(const unsigned iter, const VectorXf &sample, bool writeBinPosterior){
    if (storageMode == dense) {
        tout << sample.transpose() << endl;
    }
    if (iter % thin) return;
    unsigned thin_iter = iter/thin;
    unsigned thin_iter_post_burnin = thin_iter - burnin/thin;
    if (storageMode == dense) {
        tout << sample.transpose() << endl;
        if (iter >= burnin) {
            datMat.row(thin_iter_post_burnin) = sample;
            posteriorMean.array() += (sample - posteriorMean).array()/(thin_iter_post_burnin+1);
            posteriorSqrMean.array() += (sample.array().square() - posteriorSqrMean.array())/(thin_iter_post_burnin+1);
        }
    } else if (storageMode == sparse) {
        //SparseVector<float>::InnerIterator it(sample.sparseView());
        SparseVector<float> spvec = sample.sparseView();
        if (writeBinPosterior) {
            for (SparseVector<float>::InnerIterator it(spvec); it; ++it) {
                unsigned rc[2] = {thin_iter, (unsigned)it.index()};
                fwrite(rc, sizeof(unsigned), 2, bout);
                float val = it.value();
                fwrite(&val, sizeof(float), 1, bout);
                //cout << it.index() << " " << it.value() << endl;
            }
        }
        //nnz += sample.sparseView().nonZeros();
        //cout << nnz << " " << sample.sparseView().nonZeros() <<endl;
        ArrayXf delta;
        delta.setZero(sample.size());
        for (SparseVector<float>::InnerIterator it(spvec); it; ++it) {
            delta[it.index()] = 1;
        }
        if (iter >= burnin) {
            pip.array() += (delta - pip.array())/(thin_iter_post_burnin+1);
            posteriorMean.array() += (sample - posteriorMean).array()/(thin_iter_post_burnin+1);
            posteriorSqrMean.array() += (sample.array().square() - posteriorSqrMean.array())/(thin_iter_post_burnin+1);
        }
    }
}

void McmcSamples::getSample(const unsigned iter, const float sample, ofstream &out){
    out << boost::format("%12s ") %sample;
    if (iter % thin) return;
    unsigned thin_iter_post_burnin = iter/thin - burnin/thin;
    if (iter >= burnin) {
        datMat(thin_iter_post_burnin,0) = sample;
        posteriorMean.array() += (sample - posteriorMean.array())/(thin_iter_post_burnin+1);
        posteriorSqrMean.array() += (sample*sample - posteriorSqrMean.array())/(thin_iter_post_burnin+1);
    }
}

VectorXf McmcSamples::mean(){
    if (storageMode == dense) {
        return VectorXf::Ones(nrow).transpose()*datMat/nrow;
    } else {
        return VectorXf::Ones(nrow).transpose()*datMatSp/nrow;
    }
}

VectorXf McmcSamples::sd(){
    VectorXf res(ncol);
    if (storageMode == dense) {
        for (unsigned i=0; i<ncol; ++i) {
            res[i] = std::sqrt(Gadget::calcVariance(datMat.col(i)));
        }
    } else {
        for (unsigned i=0; i<ncol; ++i) {
            res[i] = std::sqrt(Gadget::calcVariance(datMatSp.col(i)));
        }
    }
    return res;
}

void McmcSamples::initBinFile(const string &title){
    if (myMPI::rank) return;
    filename = title + ".mcmcsamples." + label;
    bout = fopen(filename.c_str(), "wb");
    if (!bout) {
        throw("Error: cannot open file " + filename);
    }
    nnz = 0;
    unsigned xyn[3] = {chainLength/thin, ncol, nnz};
    fwrite(xyn, sizeof(unsigned), 3, bout);
}

void McmcSamples::initTxtFile(const string &title){
    if (myMPI::rank) return;
    filename = title + ".mcmcsamples." + label;
    tout.open(filename.c_str());
    if (!tout) {
        throw("Error: cannot open file " + filename);
    }
}

void McmcSamples::writeDataBin(const string &title){
    if (myMPI::rank) return;
    filename = title+ ".mcmcsamples." + label ;
    FILE *out = fopen(filename.c_str(), "wb");
    if (!out) {
        throw("Error: cannot open file " + filename);
    }
    
    int xyn[3] = {datMatSp.rows(), datMatSp.cols(), datMatSp.nonZeros()};
    fwrite(xyn, sizeof(unsigned), 3, out);
    
    for (int i=0; i < datMatSp.outerSize(); ++i) {
        SparseMatrix<float>::InnerIterator it(datMatSp, i);
        for (; it; ++it) {
            unsigned rc[2] = {(unsigned)it.row(), (unsigned)it.col()};
            fwrite(rc, sizeof(unsigned), 2, out);
            float v = it.value();
            fwrite(&v, sizeof(float), 1, out);
        }
    }
    fclose(out);
}

void McmcSamples::readDataBin(const string &filename){
    FILE *in = fopen(filename.c_str(), "rb");
    if (!in) {
        throw("Error: cannot open file " + filename);
    }
    
    unsigned xyn[3];
    fread(xyn, sizeof(unsigned), 3, in);
    
    // read with MPI
    unsigned batch_size = xyn[0]/myMPI::clusterSize;
    unsigned my_start = myMPI::rank*batch_size;
    unsigned my_end = myMPI::rank+1 == myMPI::clusterSize ? xyn[0] : my_start + batch_size;
    unsigned my_size = my_end - my_start;
    
    datMatSp.resize(my_size, xyn[1]);
//    vector<Triplet<float>> trips(xyn[2]);
    vector<Triplet<float> > trips;
    
    //for (int i=0; i < trips.size(); ++i){
    while (!feof(in)) {
        unsigned rc[2];
        fread(rc, sizeof(unsigned), 2, in);
        float v;
        fread(&v, sizeof(float), 1, in);
        
        if (rc[0] < my_start) continue;
        else if (rc[0]>= my_end) break;
        
        if(rc[0]>xyn[0] || rc[1]>xyn[1]) continue;
        
        //trips[i] = Triplet<float>(rc[0], rc[1], v);
        trips.push_back(Triplet<float>(rc[0]-my_start, rc[1], v));
    }
    fclose(in);
    
    datMatSp.setFromTriplets(trips.begin(), trips.end());
    datMatSp.makeCompressed();
    
    //cout << datMatSp.nonZeros() << " " << nnz << endl;
}

void McmcSamples::readDataTxt(const string &filename){
    ifstream in(filename.c_str());
    string inputStr;
    vector<float> tmp;
    while (in >> inputStr) {
        tmp.push_back(stof(inputStr));
    }
    in.close();
    datMat.resize(tmp.size(), 1);
    datMat.col(0) = Eigen::Map<VectorXf>(&tmp[0], tmp.size());
}

void McmcSamples::writeDataTxt(const string &title){
    if (myMPI::rank) return;
    filename = title+ ".mcmcsamples." + label ;
    ofstream out(filename);
    out << datMat << endl;
    out.close();
}

void MCMC::initTxtFile(const vector<Parameter*> &paramVec, const string &title){
    if (myMPI::rank) return;
    outfilename = title + ".mcmcsamples.Par";
    out.open(outfilename.c_str());
    if (!out) {
        throw("Error: cannot open file " + outfilename);
    }
    for (unsigned i=0; i<paramVec.size(); ++i) {
        Parameter *par = paramVec[i];
        out << boost::format("%12s ") %par->label;
    }
    out << endl;
}

vector<McmcSamples*> MCMC::initMcmcSamples(const Model &model, const unsigned chainLength, const unsigned burnin,
                                           const unsigned thin, const string &title, const bool writeBinPosterior){
    vector<McmcSamples*> mcmcSampleVec;
    for (unsigned i=0; i<model.paramSetVec.size(); ++i) {
        ParamSet *parSet = model.paramSetVec[i];
        McmcSamples *mcmcSamples;
        if (parSet->label == "SnpEffects") {
            if (myMPI::partition=="bycol")
                mcmcSamples = new McmcSamples(parSet->label, chainLength, burnin, thin, model.numSnps, "sparse");
            else
                mcmcSamples = new McmcSamples(parSet->label, chainLength, burnin, thin, parSet->size, "sparse");
            if (writeBinPosterior) mcmcSamples->initBinFile(title);
        } else if (parSet->label == "WindowDelta") {
            mcmcSamples = new McmcSamples(parSet->label, chainLength, burnin, thin, parSet->size, "sparse");
            if (writeBinPosterior) mcmcSamples->initBinFile(title);
        } else {
            mcmcSamples = new McmcSamples(parSet->label, chainLength, burnin, thin, parSet->size);
            mcmcSamples->initTxtFile(title);
        }
        mcmcSampleVec.push_back(mcmcSamples);
    }
    for (unsigned i=0; i<model.paramVec.size(); ++i) {
        Parameter *par = model.paramVec[i];
        McmcSamples *mcmcSamples = new McmcSamples(par->label, chainLength, burnin, thin, 1);
        //mcmcSamples->initTxtFile(title);
        mcmcSampleVec.push_back(mcmcSamples);
    }
    initTxtFile(model.paramVec, title);
    return mcmcSampleVec;
}

void MCMC::collectSamples(const Model &model, vector<McmcSamples*> &mcmcSampleVec, const unsigned iteration, const bool writeBinPosterior){
    unsigned i = 0;
    for (unsigned j=0; j<model.paramSetVec.size(); ++j) {
        McmcSamples *mcmcSamples = mcmcSampleVec[i++];
        ParamSet *parSet = model.paramSetVec[j];
        mcmcSamples->getSample(iteration, parSet->values, writeBinPosterior);
    }
    for (unsigned j=0; j<model.paramVec.size(); ++j) {
        McmcSamples *mcmcSamples = mcmcSampleVec[i++];
        Parameter *par = model.paramVec[j];
        mcmcSamples->getSample(iteration, par->value, out);
    }
    out << endl;
}

void MCMC::printStatus(const vector<Parameter*> &paramToPrint, const unsigned thisIter, const unsigned outputFreq, const string &timeLeft){
    if (thisIter==outputFreq) {
        cout << boost::format("%=10s ") % "Iter";
        for (unsigned i=0; i<paramToPrint.size(); ++i) {
            cout << boost::format("%=12s ") % paramToPrint[i]->label;
        }
        cout << boost::format("%=12s\n") % "TimeLeft";
    }
    cout << boost::format("%=10s ") % thisIter;
    for (unsigned i=0; i<paramToPrint.size(); ++i) {
        Parameter *par = paramToPrint[i];
        if (par->label=="NNZsnp" || par->label=="NNZwind")
            cout << boost::format("%=12.0f ") % par->value;
        else
            cout << boost::format("%=12.4f ") % paramToPrint[i]->value;
    }
    cout << boost::format("%=12s\n") % timeLeft;
    
    cout.flush();
}

void MCMC::printSummary(const vector<Parameter*> &paramToPrint, const vector<McmcSamples*> &mcmcSampleVec, const string &filename){
    ofstream out;
    out.open(filename.c_str());
    if (!out) {
        throw("Error: cannot open file " + filename);
    }
    cout << "\nPosterior statistics from MCMC samples:\n\n";
    cout << boost::format("%13s %-15s %-15s\n") %"" % "Mean" % "SD ";
    out << "Posterior statistics from MCMC samples:\n\n";
    out << boost::format("%13s %-15s %-15s\n") %"" % "Mean" % "SD ";
    for (unsigned i=0; i<paramToPrint.size(); ++i) {
        Parameter *par = paramToPrint[i];
        for (unsigned j=0; j<mcmcSampleVec.size(); ++j) {
            McmcSamples *mcmcSamples = mcmcSampleVec[j];
            if (mcmcSamples->label == par->label) {
                cout << boost::format("%10s %2s %-15.6f %-15.6f\n")
                % par->label
                % ""
                % mcmcSamples->mean()
                % mcmcSamples->sd();
                out << boost::format("%10s %2s %-15.6f %-15.6f\n")
                % par->label
                % ""
                % mcmcSamples->mean()
                % mcmcSamples->sd();
                break;
            }
        }
    }
    out.close();
}

vector<McmcSamples*> MCMC::run(Model &model, const unsigned chainLength, const unsigned burnin, const unsigned thin,
                               const unsigned outputFreq, const string &title, const bool writeBinPosterior){
    if (myMPI::rank==0) cout << "MCMC lauched ...\n" << endl;

    vector<McmcSamples*> mcmcSampleVec = initMcmcSamples(model, chainLength, burnin, thin, title, writeBinPosterior);
    
    Gadget::Timer timer;
    timer.setTime();
    
    for (unsigned iteration=0; iteration<chainLength; ++iteration) {
        unsigned thisIter = iteration + 1;
        
        model.sampleUnknowns();
        
        if (myMPI::rank==0) {
            collectSamples(model, mcmcSampleVec, iteration, writeBinPosterior);
        }
        
        if (!(thisIter % outputFreq)) {
            timer.getTime();
            time_t timeToFinish = (chainLength-thisIter)*timer.getElapse()/thisIter; // remaining iterations multiplied by average time per iteration in seconds
            if (myMPI::rank==0) {
                printStatus(model.paramToPrint, thisIter, outputFreq, timer.format(timeToFinish));
            }
        }
    }
    
    // save the samples in the last iteration for potential continual run
    

    if (myMPI::rank==0) {
        cout << "\nMCMC cycles completed." << endl;
        printSummary(model.paramToPrint, mcmcSampleVec, title + ".parRes");
    }
    
    return mcmcSampleVec;
}
