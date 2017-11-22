//
//  mympi.hpp
//  gctb
//
//  Created by Jian Zeng on 23/08/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef mympi_hpp
#define mympi_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <mpi.h>

namespace myMPI {
    extern int clusterSize;
    extern int rank;
    extern char processorName[MPI_MAX_PROCESSOR_NAME];
    extern int processorNameLength;
    extern std::string partition;
    extern int iSize;
    extern int iStart;
    extern std::vector<int> srcounts;
    extern std::vector<int> displs;
}

#endif /* mympi_hpp */
