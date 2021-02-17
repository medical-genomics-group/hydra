#include <math.h>
#include <omp.h>
#include <cassert>
#include <immintrin.h>
#include "dense.hpp"
#include "hydra.h"
#include "dotp_lut.h"


void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in1,
                const double* __restrict__ in2,
                const int SIZE) {
    out = (double*) __builtin_assume_aligned(out, 64);
    in1 = (double*) __builtin_assume_aligned(in1, 64);
    in2 = (double*) __builtin_assume_aligned(in2, 64);
#ifdef _OPENMP
#pragma omp parallel for simd aligned(in1,in2,out:64) simdlen(8)
#endif
    for (int i=0; i<SIZE; i++) {
        out[i]  = in1[i] + in2[i];
    }
}

void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in,
                const int N) {
    out = (double*) __builtin_assume_aligned(out, 64);
    in  = (double*) __builtin_assume_aligned(in,  64);
#ifdef _OPENMP
#pragma omp parallel for simd aligned(in,out:64)
#endif
    for (int i=0; i<N; i++)
        out[i] = out[i] + in[i];
}



//#pragma omp parallel default(none) shared(sum, array, N)
//EO: see https://gcc.gnu.org/gcc-9/porting_to.html#ompdatasharing
//
double sum_array_elements(const double* __restrict__ in,
                          const int N) {

    in = (double*) __builtin_assume_aligned(in, 64);

    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel default(none) shared(sum, in) firstprivate(N)
    {
        //int ID = omp_get_thread_num();
        double partial_sum = 0.0;
#pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            partial_sum += in[i];
        }
#pragma omp for ordered schedule(static,1)
        for (int t=0; t<omp_get_num_threads(); ++t) {
            //assert( t==ID );
#pragma omp ordered
            {
                sum += partial_sum;
            }
        }
    }
#else
    for (int i=0; i<N; i++)
        sum += in[i];
#endif
    return sum;
}

double sum_array_elements(const long double* __restrict__ in,
                          const int N) {

    in = (long double*) __builtin_assume_aligned(in, 64);

    long double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel default(none) shared(sum, in) firstprivate(N)
    {
        //int ID = omp_get_thread_num();
        long double partial_sum = 0.0;

#pragma omp for schedule(static)

        for (int i=0; i<N; i++) {
            partial_sum += in[i];
        }

#pragma omp for ordered schedule(static,1)
        for (int t=0; t<omp_get_num_threads(); ++t) {
            //assert( t==ID );
#pragma omp ordered
            {
                sum += partial_sum;
            }
        }
    }
#else
    for (int i=0; i<N; i++)
        sum += in[i];
#endif

    return (double) sum;
}


void center_and_scale(double* __restrict__ in,
                      const int N) {

    in = (double*) __builtin_assume_aligned(in, 64);

    double mean = sum_array_elements(in, N) / N;

    offset_array(in, -mean, N);

    double sqn = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sqn)
#endif
    for (int i=0; i<N; ++i)
        sqn += in[i] * in[i];

    sqn = sqrt(double(N-1) / sqn);

#ifdef _OPENMP
#pragma omp parallel for simd aligned(in:64)
#endif
    for (int i=0; i<N; ++i)
        in[i] *= sqn;
}


//EO: this to allow reduction on avx256 pd4 datatype with OpenMP
#ifdef _OPENMP
#pragma omp declare reduction \
    (addpd4:__m256d:omp_out+=omp_in) \
    initializer(omp_priv=_mm256_setzero_pd())
#endif

void avx_bed_dot_product(uint* I1_data,
                         const double* __restrict__ epsilon,
                         const uint Ntot,
                         const size_t snpLenByt,
                         const double mave,
                         const double mstd,
                         double &num) {

    epsilon = (double*) __builtin_assume_aligned(epsilon, 64);

    //const uint8_t* rawdata = reinterpret_cast<uint8_t*>(&I1[N1S[marker]]);
    const uint8_t* rawdata = reinterpret_cast<uint8_t*>(I1_data);
                        
#if 1
    double c1 = 0.0, c2 = 0.0;
    double s1 = 0.0, s2 = 0.0;
    
    // main + remainder to avoid a test on idx < Ntot 
    const int fullb = Ntot / 4;
    int idx = 0;
    
    
    __m256d vsum1 = _mm256_setzero_pd();
    __m256d vsum2 = _mm256_setzero_pd();
#ifdef _OPENMP
#pragma omp parallel for reduction(addpd4:vsum1,vsum2)
#endif                              
    for (int ii=0; ii<fullb; ++ii) {
        
        __m256d p4c1  = _mm256_load_pd(&(dotp_lut_a[rawdata[ii] * 4]));
        __m256d p4c2  = _mm256_load_pd(&(dotp_lut_b[rawdata[ii] * 4]));
        __m256d p4eps = _mm256_load_pd(&(epsilon[ii * 4]));
        
        __m256d p4sum = _mm256_mul_pd(p4c2, p4eps);
        
        vsum2 = _mm256_add_pd(vsum2, p4sum);
        
        p4sum = _mm256_mul_pd(p4sum, p4c1);                                
        vsum1 = _mm256_add_pd(vsum1, p4sum);
    }
    
    //EO: to double-check but no reduction available as in avx512
    s1 = vsum1[0] + vsum1[1] + vsum1[2] + vsum1[3];
    s2 = vsum2[0] + vsum2[1] + vsum2[2] + vsum2[3];
    
    // remainder
    if (Ntot % 4 != 0) {
        int ii = fullb;
        for (int iii = 0; iii < Ntot - fullb * 4; iii++) {
            idx = rawdata[ii] * 4 + iii;
            c1  = dotp_lut_a[idx];
            c2  = dotp_lut_b[idx];
            s1 += c1 * (c2 * epsilon[ii * 4 + iii]);
            s2 +=      (c2 * epsilon[ii * 4 + iii]);
        }
    }
    
    //num = mstd[marker] * (s1 - mave[marker] * s2);
    num = mstd * (s1 - mave * s2);

#else
    throw("need to adapt if Ntot%4 != 0");
    exit(1);
    double c1  = 0.0, c2 = 0.0;
    double dp1 = 0.0, dp2 = 0.0, dpm = 0.0;
                            
    double syt = sum_array_elements(epsilon, Ntot);
                            
    for (int ii=0; ii<snpLenByt; ++ii) {
        for (int iii=0; iii<4; ++iii) {                                    
            c1 = dotp_lut_a[rawdata[ii] * 4 + iii];
            c2 = dotp_lut_b[rawdata[ii] * 4 + iii];
            if (ii*4 + iii < Ntot) {
                if (c1 == 1.0) dp1 +=       epsilon[ii * 4 + iii];
                if (c1 == 2.0) dp2 += 2.0 * epsilon[ii * 4 + iii];
                dpm  += (1.0 - c2) * epsilon[ii * 4 + iii];
            }
        }
    }
                            
    syt -= dpm;
    num  = dp1 + dp2;
    //num -= mave[marker] * syt;
    //num *= mstd[marker];
    num -= mave * syt;
    num *= mstd;
#endif
}


void bed_scaadd(uint* I1_data,
                const uint Ntot,
                const double deltaBeta,
                const double mave,
                const double mstd,
                double* deltaEps) {

    const uint8_t* rawdata = reinterpret_cast<uint8_t*>(I1_data);
                            
    double c1 = 0.0, c2 = 0.0;
                            
    const double sigdb = mstd * deltaBeta;

    const int fullb = Ntot / 4;

    int idx = 0;

    // main
#ifdef _OPENMP
#pragma omp parallel for
#endif                               
    for (int ii=0; ii<fullb; ++ii) {
        for (int iii=0; iii<4; iii++) {
            idx = rawdata[ii] * 4 + iii;
            c1 = dotp_lut_a[idx];
            c2 = dotp_lut_b[idx];
            deltaEps[ii * 4 + iii]  = (c1 - mave) * c2 * sigdb;
        }
    }
                                    
    // remainder
    if (Ntot % 4 != 0) {
        int ii = fullb;
        for (int iii = 0; iii < Ntot - fullb * 4; iii++) {
            idx = rawdata[ii] * 4 + iii;
            c1 = dotp_lut_a[idx];
            c2 = dotp_lut_b[idx];
            deltaEps[ii * 4 + iii]  = (c1 - mave) * c2 * sigdb;
        }
    }
}
