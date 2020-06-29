#pragma once

#include <mpi.h>
#include "utils.hpp"


// History file with layout: Mtot | [ iteration | rank_0_data ... rank_N_data ] 
template <class T>
void write_ofile_h1 (MPI_File fh, const uint rank, const uint Mtot, const uint iteration, const uint n_thinned_saved, const uint mranks, const uint size, const T* data, const MPI_Datatype mpi_type) {

    MPI_Status status;
    MPI_Offset offset;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    if (rank == 0) {

        if (n_thinned_saved == 0) {
            check_mpi(MPI_File_write_at(fh, size_t(0), &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        }

        offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * mpi_type_size);
        check_mpi(MPI_File_write_at(fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }
   
    offset = sizeof(uint) + sizeof(uint)
        + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * mpi_type_size)
        + size_t(mranks) * mpi_type_size;
    
    check_mpi(MPI_File_write_at(fh, offset, data, size, mpi_type, &status), __LINE__, __FILE__);
}


// History file with layout: iteration | double
template <class T>
void write_ofile_s1 (MPI_File fh, const uint iteration, const uint n_thinned_saved, const T data, const MPI_Datatype mpi_type) {

    MPI_Status status;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    MPI_Offset offset = size_t(n_thinned_saved) * ( sizeof(uint) + mpi_type_size );

    check_mpi(MPI_File_write_at(fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, &data,        1, MPI_DOUBLE,   &status), __LINE__, __FILE__);
}


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

