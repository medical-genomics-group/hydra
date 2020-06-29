#pragma once

#include <mpi.h>
#include "utils.hpp"


// file layout:  iter | size | data
//
template <class T>
void write_ofile_t1 (MPI_File fh, uint iteration, uint size, T data, MPI_Datatype mpi_type) {

    MPI_Status status;

    MPI_Offset offset = 0;
   
    check_mpi(MPI_File_write_at(fh, offset, &iteration, 1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, &size,      1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, data,       size, mpi_type,     &status), __LINE__, __FILE__);
}


template <class T>
void write_ofile_t2 (MPI_File fh, const int rank, const int mranks, const uint Mtot, const uint iteration, const uint size, const T* data, const MPI_Datatype mpi_type) {

    MPI_Status status;

    MPI_Offset offset = 0;
   
    if (rank == 0)
        check_mpi(MPI_File_write_at(fh, offset, &Mtot,      1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);

    
    offset += sizeof(uint);

    if (rank == 0)
        check_mpi(MPI_File_write_at(fh, offset, &iteration, 1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);

   
    offset += sizeof(uint);

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    offset += size_t(mranks) * mpi_type_size;

    check_mpi(MPI_File_write_at(fh, offset, data,       size, mpi_type,     &status), __LINE__, __FILE__);
}

