#ifndef HYDRA_DENSE_H
#define HYDRA_DENSE_H

#include <cstdio>

inline
void offset_array(double* __restrict__ in,
                  const double OFFSET,
                  const int SIZE) {

    in = (double*)__builtin_assume_aligned(in, 64);

#ifdef _OPENMP
#pragma omp parallel for simd aligned(in:64)
#endif
    for (int i=0; i<SIZE; i++)
        in[i] += OFFSET;
}

inline
void set_array(double* __restrict__ in,
               const double VALUE,
               const int SIZE) {
    in = (double*) __builtin_assume_aligned(in, 64);
#ifdef _OPENMP
#pragma omp parallel for simd aligned(in:64)
#endif
    for (int i=0; i<SIZE; i++)
        in[i] = VALUE;
}

inline
void copy_array(double*       __restrict__ dst,
                const double* __restrict__ src,
                const int SIZE) {
    dst = (double*) __builtin_assume_aligned(dst, 64);
    src = (double*) __builtin_assume_aligned(src, 64);
#ifdef _OPENMP
#pragma omp parallel for simd aligned(dst,src:64)
#endif
    for (int i=0; i<SIZE; i++)
        dst[i] = src[i];
}


double sum_array_elements(const double* __restrict__ in,
                          const int N);
double sum_array_elements(const long double* __restrict__ in,
                          const int N);

void add_arrays(double*       __restrict__ pout,
                const double* __restrict__ pin1,
                const double* __restrict__ pin2,
                const int SIZE);

void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in1,
                const int N);

void center_and_scale(double* __restrict__ vec,
                      const int            N);


void avx_bed_dot_product(uint*         I1_data, 
                         const double* epsilon,
                         const uint    Ntot,
                         const size_t  snpLenByt,
                         const double  mave,
                         const double  mstd,
                         double&       num);

void bed_dot_product(uint*         I1_data, 
                     const double* epsilon,
                     const uint    Ntot,
                     const size_t  snpLenByt,
                     const double  mave,
                     const double  mstd,
                     double&       num);

void bed_scaadd(uint*        I1_data,
                const uint   Ntot,
                const double deltaBeta,
                const double mave,
                const double mstd,
                double*      deltaEps);


#endif //#define HYDRA_DENSE_H
