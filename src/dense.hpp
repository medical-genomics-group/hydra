#ifndef HYDRA_DENSE_H
#define HYDRA_DENSE_H


// Add an offset to each element of the input array
void offset_array(double* __restrict__ array,
                  const double         offset,
                  const int            N);


void set_array(double* __restrict__ array,
               const double         val,
               const int            N);


void copy_array(double*       __restrict__ dest,
                const double* __restrict__ source,
                const int N);


double sum_array_elements(const      double* __restrict__ array, const int N);
double sum_array_elements(const long double* __restrict__ array, const int N);


void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in1,
                const double* __restrict__ in2,
                const int N);


void add_arrays(double*       __restrict__ out,
                const double* __restrict__ in1,
                const int N);


void center_and_scale(double* __restrict__ vec,
                      const int            N);


#endif //#define HYDRA_DENSE_H
