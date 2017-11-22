//
//  hsq.hpp
//  gctb
//
//  Created by Jian Zeng on 20/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef hsq_hpp
#define hsq_hpp

#include <stdio.h>
#include "data.hpp"
#include "mcmc.hpp"


class Heritability {
public:
    float varGenotypic;
    float varResidual;
    float hsq;
    unsigned popSize;
    
    VectorXf hsqMcmc;
    
    void getEstimate(const Data &data, const McmcSamples &snpEffects, const McmcSamples &resVar);
    void writeRes(const string &filename);
    void writeMcmcSamples(const string &filename);
};

#endif /* hsq_hpp */
