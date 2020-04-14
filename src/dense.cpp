#include <cstdlib>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>


double sum_vector_elements_f64(const double* __restrict__ vec, const int N) {

    double sum = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
#endif
    //#pragma unroll
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum)
#endif
    for (int i=0; i<N; i++) {
        sum += vec[i];
    }

    return sum;
}


void sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2, const int N) {

#ifdef __INTEL_COMPILER
    __assume_aligned(in1, 64);
    __assume_aligned(in2, 64);
    __assume_aligned(out, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        out[i] = in1[i] + in2[i];
    }
}


void sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const int N) {

    const int N8 = (N/8) * 8;
#ifdef __INTEL_COMPILER
    __assume_aligned(in1, 64);
    __assume_aligned(out, 64);
    __assume(N8%8==0);
#endif
    //#pragma unroll(8)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N8; i++) {
        out[i] += in1[i];
    }

    for (int i=N8; i<N; ++i) {
        out[i] += in1[i];
    }
}


void offset_vector_f64(double* __restrict__ vec, const double offset, const int N) {
#ifdef __INTEL_COMPILER
    __assume_aligned(vec,   64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        vec[i] += offset;
    }
}


void set_vector_f64(double* __restrict__ vec, const double val, const int N) {

    const int N8 = (N/8) * 8;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume(N8%8==0);
#endif
    //#pragma unroll(8)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N8; i++) {
        vec[i] = val;
    }

    for (int i=N8; i<N; ++i) {
        vec[i] = val;
    }
}


void copy_vector_f64(double* __restrict__ dest, const double* __restrict__ source, const int N) {
#ifdef __INTEL_COMPILER
    __assume_aligned(dest,   64);
    __assume_aligned(source, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        dest[i] = source[i];
    }
}


void scaadd(double* __restrict__ vout, const double* __restrict__ vin1, const double* __restrict__ vin2, const double dMULT, const int N) {

    if   (dMULT == 0.0) {
        for (int i=0; i<N; i++) {
            vout[i] = vin1[i];
        }
    } else {
        for (int i=0; i<N; i++) {
            vout[i] = vin1[i] + dMULT * vin2[i];
        }
    }
}


//
// Functions based on Intel AVX512 intrinsics
//

void CenterAndScaleAvx(const uint N, const uint NAS, const unsigned char* __restrict__ mask, double* __restrict__ vec) {

    __m512d sumx8 = _mm512_setzero_pd();
    
    for (int i=0; i<N/8; ++i) {

        __m512d p8  = _mm512_loadu_pd(&(vec[i * 8]));
        __m512d s8  = _mm512_maskz_add_pd(mask[i], p8, p8);

        sumx8 = _mm512_add_pd(sumx8, s8);
    }

    // EO: divide by 2 because using masked add vec + vec
    double mean = _mm512_reduce_add_pd(sumx8) / 2.0;

    // Remaining elements
    for (int i=N/8*8; i<N; ++i) {
        if ((mask[i / 8] >> (i % 8)) & 1) {
            mean += vec[i];
        }
    }

    mean /= (N - NAS);


    // Center; check that this auto-vec; don't care about NA values here
    for (int i=0; i<N; ++i)  vec[i] -= mean;


    sumx8 = _mm512_setzero_pd();
    
    for (int i=0; i<N/8; ++i) {

        __m512d p8  = _mm512_loadu_pd(&(vec[i * 8]));
        __m512d s8  = _mm512_maskz_mul_pd(mask[i], p8, p8);

        sumx8 = _mm512_add_pd(sumx8, s8);
    }

    double sqn = _mm512_reduce_add_pd(sumx8);
    sqn = sqrt(double(N - NAS - 1) / sqn);
    printf("sqn = %20.15f avx\n", sqn);

    // Scale; idem don't care about NAs; check auto-ver
    for (int i=0; i<N; ++i)  vec[i] *= sqn;
}


void CenterAndScale(const uint N, const unsigned char* __restrict__ mask, double* __restrict__ vec) {

    // Compute mean
    double mean  = 0.0;
    uint   n_nas = 0;
 
    for (int i=0; i<N; ++i) {
        if ((mask[i / 8] >> (i % 8)) & 1) {
            mean += vec[i];
        } else {
            n_nas += 1;
        }
    }
    printf("mean = %20.15f ??\n", mean);

    mean /= (N - n_nas);
    //printf("mean = %20.15f\n", mean);


    // Center
    for (int i=0; i<N; ++i)  vec[i] -= mean;

    // Compute scale
    double sqn = 0.0;
    for (int i=0; i<N; ++i)  sqn += vec[i] * vec[i];
    sqn = sqrt(double(N-1) / sqn);
    printf("sqn = %20.15f\n", sqn);


    // Scale
    for (int i=0; i<N; ++i)  vec[i] *= sqn;
}


double SquaredNormAvx(const uint N, const unsigned char* __restrict__ mask, double* __restrict__ vec) {

    __m512d sumx8 = _mm512_setzero_pd();
    
    for (int i=0; i<N/8; ++i) {

        __m512d p8  = _mm512_loadu_pd(&(vec[i * 8]));
        __m512d s8  = _mm512_maskz_mul_pd(mask[i], p8, p8);

        sumx8 = _mm512_add_pd(sumx8, s8);
    }

    double sqn = _mm512_reduce_add_pd(sumx8);
    
    for (int i=N/8*8; i<N; ++i) {
        if ((mask[i / 8] >> (i % 8)) & 1) {
            sqn += vec[i] * vec[i];
        }
    }

    return sqn;
}


double SumVectorElementsAvx(const uint N, const unsigned char* __restrict__ mask, const double* __restrict__ vec) {

    __m512d sumx8 = _mm512_setzero_pd();
    //__m512d zero  = _mm512_setzero_pd();
    __m512d zero  = _mm512_set1_pd(0.0);

#ifdef __INTEL_COMPILER
    __assume_aligned(vec,   64);
#endif
    for (int i=0; i<N/8; ++i) {

        __m512d p8  = _mm512_loadu_pd(&(vec[i * 8]));
        __m512d s8  = _mm512_maskz_add_pd(mask[i], p8, zero);
        //__m512d s8  = _mm512_maskz_add_pd(mask[i], p8, p8);
        //__m512d s8  = _mm512_add_pd(p8, p8);

        sumx8 = _mm512_add_pd(sumx8, s8);
    }
    double sum = _mm512_reduce_add_pd(sumx8);

    for (int i=(N/8)*8; i<N; ++i) {
        if ((mask[i / 8] >> (i % 8)) & 1) {
            sum += vec[i];
        }
    }

    return sum;
}

double SumVectorElements(const uint N, const unsigned char* __restrict__ mask, const double* __restrict__ vec) {

    // Compute mean
    double sum  = 0.0;
 
    for (int i=0; i<N; ++i) {
        if ((mask[i / 8] >> (i % 8)) & 1) {
            sum += vec[i];
        }
    }
    
    return sum;
}
