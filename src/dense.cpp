#include <math.h>


// Add offset to each element of the array
//
void offset_array(double* __restrict__ array,
                  const double         offset,
                  const int            N) {

#ifdef __INTEL_COMPILER
    __assume_aligned(array,   64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) { 
        array[i] += offset;
    }
}


// Set all elements of array to val
//
void set_array(double* __restrict__ array,
               const double         val,
               const int            N) {

#ifdef __INTEL_COMPILER
    __assume_aligned(array, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        array[i] = val;
    }
}


void copy_array(double*       __restrict__ dest,
                const double* __restrict__ source,
                const int                  N) {

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


double sum_array_elements(const double* __restrict__ array, const int N) {

    double sum = 0.0;

#ifdef __INTEL_COMPILER
    __assume_aligned(array, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum)
#endif
    for (int i=0; i<N; i++) {
        sum += array[i];
    }

    return sum;
}


void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in1,
                const double* __restrict__ in2,
                const int                  N) {

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


void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in1,
                const int                  N) {

#ifdef __INTEL_COMPILER
    __assume_aligned(in1, 64);
    __assume_aligned(out, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<N; i++) {
        out[i] += in1[i];
    }
}


void center_and_scale(double* __restrict__ vec,
                      const int            N) {

    double mean = sum_array_elements(vec, N) / N;

    // Center
    offset_array(vec, -mean, N);

    // Compute scale
    double sqn = 0.0;
    for (int i=0; i<N; ++i)  sqn += vec[i] * vec[i];
    sqn = sqrt(double(N-1) / sqn);

    // Scale
    for (int i=0; i<N; ++i)  vec[i] *= sqn;
}
