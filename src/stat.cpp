//
//  stat.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "stat.hpp"

void Stat::seedEngine(const int seed){
    if (seed) {
        srand(seed);
        engine.seed(seed);
    } else {
        srand((int)time(NULL));
        engine.seed((int)time(NULL));
    }
}

float Stat::Normal::sample(const float mean, const float variance){
    return mean + snorm()*sqrtf(variance);
}

float Stat::InvChiSq::sample(const float df, const float scale){
    //inverse_chi_squared_distribution invchisq(df, scale);
    //return boost::math::quantile(invchisq, ranf());   // don't know why this is not correct
    
    gamma_generator sgamma(engine, gamma_distribution(0.5f*df, 1));
    return scale/(2.0f*sgamma());
}

float Stat::Gamma::sample(const float shape, const float scale){
    gamma_generator sgamma(engine, gamma_distribution(shape, scale));
    return sgamma();
}

float Stat::Beta::sample(const float a, const float b){
    beta_distribution beta(a,b);
    return boost::math::quantile(beta,ranf());
}

unsigned Stat::Bernoulli::sample(const float p){
    return ranf() < p ? 1:0;
}

float Stat::NormalZeroMixture::sample(const float mean, const float variance, const float p){
    return bernoulli.sample(p) ? normal.sample(mean, variance) : 0;
}
