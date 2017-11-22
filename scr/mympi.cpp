//
//  mympi.cpp
//  gctb
//
//  Created by Jian Zeng on 23/08/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include "mympi.hpp"

int myMPI::clusterSize = 1;
int myMPI::rank = 0;
char myMPI::processorName[MPI_MAX_PROCESSOR_NAME];
int myMPI::processorNameLength;
std::string myMPI::partition = "byrow";
int myMPI::iSize;
int myMPI::iStart;
std::vector<int> myMPI::srcounts;
std::vector<int> myMPI::displs;
