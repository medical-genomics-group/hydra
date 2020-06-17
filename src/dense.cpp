#include <math.h>
#include <omp.h>
#include <cassert>

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

#ifdef __INTEL_COMPILER
    __assume_aligned(array, 64);
#endif

    double sum = 0.0;

#ifdef _OPENMP

#pragma omp parallel default(none) shared(sum, array, N)
    {
        //int ID = omp_get_thread_num();
        double partial_sum = 0.0;
        
#pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            partial_sum += array[i];
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

    for (int i=0; i<N; i++) {
        sum += array[i];
    }

#endif

    return sum;
}

double sum_array_elements(const long double* __restrict__ array, const int N) {

#ifdef __INTEL_COMPILER
    __assume_aligned(array, 64);
#endif

    long double sum = 0.0;

#ifdef _OPENMP

#pragma omp parallel default(none) shared(sum, array, N)
    {
        //int ID = omp_get_thread_num();
        long double partial_sum = 0.0;
        
#pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            partial_sum += array[i];
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

    for (int i=0; i<N; i++) {
        sum += array[i];
    }

#endif

    return (double) sum;
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
