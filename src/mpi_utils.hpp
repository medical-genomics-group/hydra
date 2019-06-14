#ifndef mpi_utils
#define mpi_utils

#include <mpi.h>

// Check MPI call returned value. If error print message and call MPI_Abort()
// --------------------------------------------------------------------------
inline void check_mpi(const int error, const int linenumber, const char* filename) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "MPI error %d at line %d of file %s\n", error, linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Check for integer overflow
// --------------------------
inline void check_int_overflow(const int n, const int linenumber, const char* filename) {
    if (n > pow(2,(sizeof(int)*8)-1)) {
        fprintf(stderr, "Fatal: integer overflow detected on line %d of %s\n", linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


#endif 
