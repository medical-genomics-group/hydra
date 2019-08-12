#ifndef mpi_utils
#define mpi_utils

#include <mpi.h>


// Check malloc in MPI context
// ---------------------------
inline void check_malloc(const void* ptr, const int linenumber, const char* filename) {
    if (ptr == NULL) {
        fprintf(stderr, "#FATAL#: malloc failed on line %d of %s\n", linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Check MPI call returned value. If error print message and call MPI_Abort()
// --------------------------------------------------------------------------
inline void check_mpi(const int error, const int linenumber, const char* filename) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "*FATAL*: MPI error %d at line %d of file %s\n", error, linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Check if size_can be casted to int or would overflow
// ------------------------------------------------------
inline int check_int_overflow(const size_t n, const int linenumber, const char* filename) {

    if (n > INT_MAX) {
        fprintf(stderr, "FATAL  : integer overflow detected on line %d of %s. %lu does not fit in type int.\n", linenumber, filename, n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(n);
}


// Check file size is the expected one
// -----------------------------------
inline int check_file_size(const MPI_File fh, const size_t N, const size_t DTSIZE, const int linenumber, const char* filename) {

    //printf("DEBUG  : expected size = %lu x %lu bytes\n", N, DTSIZE);
    size_t     expected_file_size = N * DTSIZE;

    MPI_Offset actual_file_size = 0;

    check_mpi(MPI_File_get_size(fh, &actual_file_size), __LINE__, __FILE__);

    if (actual_file_size != expected_file_size) {
        fprintf(stderr, "FATAL  : expected file size in bytes: %lu; actual file size: %lu from call on line %d of %s.\n", expected_file_size, actual_file_size, linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


#endif 
