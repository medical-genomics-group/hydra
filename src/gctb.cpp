//
//  gctb.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "gctb.hpp"

void GCTB::inputIndInfo(Data &data, const string &bedFile, const string &phenotypeFile, const string &keepIndFile, const unsigned keepIndMax, const unsigned mphen, const string &covariateFile){
    data.readFamFile(bedFile + ".fam");
    data.readPhenotypeFile(phenotypeFile, mphen);
    data.keepMatchedInd(keepIndFile, keepIndMax);
    data.readCovariateFile(covariateFile);
}

void GCTB::inputSnpInfo(Data &data, const string &bedFile, const string &includeSnpFile, const string &excludeSnpFile, const unsigned includeChr, const bool readGenotypes){
    data.readBimFile(bedFile + ".bim");
    if (!includeSnpFile.empty()) data.includeSnp(includeSnpFile);
    if (!excludeSnpFile.empty()) data.excludeSnp(excludeSnpFile);
    data.includeChr(includeChr);
    data.includeMatchedSnp();
    if (readGenotypes) data.readBedFile(bedFile + ".bed");
}

void GCTB::inputSnpInfo(Data &data, const string &includeSnpFile, const string &excludeSnpFile, const string &gwasSummaryFile, const string &ldmatrixFile, const unsigned includeChr, const bool multiLDmat){
    if (multiLDmat)
        data.readMultiLDmatInfoFile(ldmatrixFile);
    else
        data.readLDmatrixInfoFile(ldmatrixFile + ".info");
    if (!includeSnpFile.empty()) data.includeSnp(includeSnpFile);
    if (!excludeSnpFile.empty()) data.excludeSnp(excludeSnpFile);
    data.includeChr(includeChr);
    data.readGwasSummaryFile(gwasSummaryFile);
    data.includeMatchedSnp();
    data.buildSparseMME();
}

void GCTB::inputSnpResults(Data &data, const string &snpResFile){
    
}

void GCTB::predict(const Data &data, const string &filename){
    
}

void GCTB::clearGenotypes(Data &data){
    data.X.resize(0,0);
}
