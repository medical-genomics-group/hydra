#ifndef HYDRA_H
#define HYDRA_H

#include <cstddef>
#include <mpi.h>
#include <map>
#include <vector>
#include <string>


#ifdef __GNUG__
/* Old compatibility names for C types.  */
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
#endif


const size_t LENBUF  = 50000;


using fp_it = std::vector<std::string>::iterator;

using fh_it = std::map<std::string, MPI_File>::iterator;

#endif
