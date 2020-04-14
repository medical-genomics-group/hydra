#ifndef DENSE_H
#define DENSE_H


double sum_vector_elements_f64(const double* __restrict__ vec, const int N);

void   sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const double* __restrict__ in2, const int N);

void   sum_vectors_f64(double* __restrict__ out, const double* __restrict__ in1, const int N);

void   set_vector_f64(double* __restrict__ vec, const double val, const int N);

void   copy_vector_f64(double* __restrict__ dest, const double* __restrict__ source, const int N);

void   offset_vector_f64(double* __restrict__ vec, const double offset, const int N);


void   CenterAndScaleAvx(const uint N, const uint NA, const unsigned char* __restrict__ mask, double* __restrict__ vec);

void   CenterAndScale(const uint N, const unsigned char* __restrict__ mask, double* __restrict__ vec);


double SquaredNormAvx(const uint N, const unsigned char* __restrict__ mask, double* __restrict__ vec);

double SumVectorElementsAvx(const uint N, const unsigned char* __restrict__ mask, const double* __restrict__ vec);
double SumVectorElements(   const uint N, const unsigned char* __restrict__ mask, const double* __restrict__ vec);

#endif
