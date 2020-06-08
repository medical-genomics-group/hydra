#ifndef HYDRA_MPI_UTILS_H_
#define HYDRA_MPI_UTILS_H_

#include <mpi.h>
#include <ctime>
#include <sys/time.h>

inline double mysecond() {
    struct timeval  tp;
    struct timezone tzp;
    int i;
    i = gettimeofday(&tp, &tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


void check_malloc(const void* ptr, const int linenumber, const char* filename);

void check_mpi(const int error, const int linenumber, const char* filename);

int check_int_overflow(const size_t n, const int linenumber, const char* filename);

void check_file_size(const MPI_File fh, const size_t N, const size_t DTSIZE, const int linenumber, const char* filename);

#endif 
