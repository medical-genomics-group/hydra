#pragma once

void delta_epsilon_exchange(const bool opt_bedSync,
                            const bool opt_sparseSync,
                            std::vector<int>& mark2sync,
                            std::vector<double>& dbet2sync,
                            const double* mave,
                            const double* mstd,
                            const size_t snpLenByt,
                            const size_t snpLenUint,
                            const bool* USEBED,
                            const sparse_info_t* sparse_info,
                            const uint Ntot,
                            const Data* data,
                            const double* __restrict__ dEpsSum,
                            const double* __restrict__ tmpEps,
                            double* __restrict__ deltaSum,
                            double* __restrict__ epsilon);
