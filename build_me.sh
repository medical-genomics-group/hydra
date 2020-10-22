#!/bin/bash

# Author : E. Orliac, DCSR, UNIL
# Date   : 26.03.2019
# Purpose: Handle compilation for MPI versions of the code
# Note   : to force compilation, use: sh build_me -B

echo "First arg is: >>$1<<"

BUILD_GCC=0
BUILD_INTEL=1

if [ $BUILD_GCC == 1 ]; then
    echo 
    echo "====================================="
    echo "     COMPILING WITH GCC/MVAPCIH2     "
    echo "====================================="
    echo

    module purge
    module load gcc mvapich2 boost eigen
    
    cd src
    make $1 -f Makefile_G || exit 1
    cd ..
fi


if [ $BUILD_INTEL == 1 ]; then
    
    echo 
    echo "========================================"
    echo "     COMPILING WITH INTEL/INTEL-MPI     "
    echo "========================================"
    echo
    module purge
    module load intel intel-mpi boost eigen
    
    cd src
    make $1 || exit 1
    cd ..

fi

module purge
