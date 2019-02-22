//
//  gctb.hpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef amber_hpp
#define amber_hpp

#include <stdio.h>
#include <mpi.h>

#include "data.hpp"
#include "options.hpp"

#include "mympi.hpp"

class GCTB {
public:
    Options &opt;

    GCTB(Options &options): opt(options){};
    
    void inputIndInfo(Data &data, const string &bedFile, const string &phenotypeFile, const string &keepIndFile,
                      const unsigned keepIndMax, const unsigned mphen);
    void inputSnpInfo(Data &data, const string &bedFile, const string &includeSnpFile, const string &excludeSnpFile,
                      const unsigned includeChr, const bool readGenotypes);

};

#endif /* amber_hpp */
