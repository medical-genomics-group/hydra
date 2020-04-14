#include <cstdlib>
#include <stdio.h>
#include "dense.h"

inline 
double partial_sparse_dotprod(const double* __restrict__ vec,
                              const uint*   __restrict__ IX,
                              const size_t               NXS,
                              const size_t               NXL,
                              const double               fac) {
    
    //double t1 = -mysecond();
    //for (int ii=0; ii<1024; ii++) {
    //}
    //t1 += mysecond();
    //printf("kerold 1 BW = %g\n", double(N1L)*sizeof(double) / 1024. / 1024. / t1);

    double dp = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: dp)
#endif
    for (size_t i=NXS; i<NXS+NXL; i++) {
        dp += vec[ IX[i] ] * fac;
    }

    printf("partial %lu -> %lu with fac = %20.15f, dp = %20.15f REF\n", NXS, NXS+NXL, fac, dp);

    return dp;
}


double sparse_dotprod(const double* __restrict__ vin1,
                                const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                                const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                                const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                                const double               mu,
                                const double               sig_inv,
                                const int                  N,
                                const int                  marker) {

    double dp  = 0.0;
    double syt = 0.0;

    dp += partial_sparse_dotprod(vin1, I1, N1S, N1L, 1.0);

    dp += partial_sparse_dotprod(vin1, I2, N2S, N2L, 2.0);

    dp *= sig_inv;

    syt += sum_vector_elements_f64(vin1, N);

    syt += partial_sparse_dotprod(vin1, IM, NMS, NML, -1.0);

    dp -= mu * sig_inv * syt;

    return dp;
}


// NEW NA

inline 
double partial_sparse_dotprod_new(const double*        __restrict__  vec,
                                  const uint*          __restrict__  IX,
                                  const size_t                       NXS,
                                  const size_t                       NXL,
                                  const double                       fac,
                                  const unsigned char* __restrict__  mask) {
    
    //double t1 = -mysecond();
    //for (int ii=0; ii<1024; ii++) {
    //}
    //t1 += mysecond();
    //printf("kerold 1 BW = %g\n", double(N1L)*sizeof(double) / 1024. / 1024. / t1);

    double dp = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: dp)
#endif
    for (size_t i=NXS; i<NXS+NXL; i++) {
        double mask_dp = (double) ((mask[IX[i] / 8] >> (IX[i] % 8)) & 1);
        //printf("mask_dp at %6d %6d = %20.15f\n", i, IX[i], mask_dp);
        dp += vec[ IX[i] ] * fac * mask_dp;
    }

    printf("partial %lu -> %lu with fac = %20.15f, dp = %20.15f\n", NXS, NXS+NXL, fac, dp);

    return dp;
}


double sparse_dotprod_new(const double* __restrict__ vin1,
                          const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                          const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                          const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                          const double               mu,
                          const double               sig_inv,
                          const int                  N,
                          const int                  marker,
                          const unsigned char* __restrict__ mask) {

    double dp  = 0.0;
    double syt = 0.0;

    dp += partial_sparse_dotprod_new(vin1, I1, N1S, N1L, 1.0, mask);

    dp += partial_sparse_dotprod_new(vin1, I2, N2S, N2L, 2.0, mask);

    dp *= sig_inv;

    syt += sum_vector_elements_f64(vin1, N);

    syt += partial_sparse_dotprod_new(vin1, IM, NMS, NML, -1.0, mask);

    dp -= mu * sig_inv * syt;

    return dp;
}
