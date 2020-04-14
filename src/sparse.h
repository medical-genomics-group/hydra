#ifndef SPARSE_H
#define SPARSE_H


double sparse_dotprod(const double* __restrict__ vin1,
                      const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                      const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                      const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                      const double               mu, 
                      const double               sig_inv,
                      const int                  N,
                      const int                  marker);

double sparse_dotprod_new(const double* __restrict__ vin1,
                          const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                          const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                          const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                          const double               mu, 
                          const double               sig_inv,
                          const int                  N,
                          const int                  marker,
                          const unsigned char* __restrict__ mask);

#endif
