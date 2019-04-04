#!/bin/bash

# Author : E. Orliac, DCSR, UNIL
# Date   : 26.03.2019
# Purpose: Handle compilation for sequential and MPI versions of the code

echo "First arg is: >>$1<<"

echo 
echo "==========================================="
echo "       COMPILING THE MPI VERSION (GNU)     "
echo "==========================================="
echo

module purge
module load gcc/7.3.0 mvapich2 openblas boost eigen zlib

cd src
make $1 -f Makefile_G
cd ..

#echo "__EARLY_EXIT__"
#exit

echo 
echo "==========================================="
echo "       COMPILING THE MPI VERSION (INTEL)   "
echo "==========================================="
echo
module purge
module load intel intel-mpi intel-mkl boost eigen zlib

cd src
make $1

if [ $? -ne 0 ] ; then
    echo "make failed."
    exit 1
fi
cd ..
module purge

exit 0

echo 
echo "======================================"
echo "   COMPILING THE SEQUENTIAL VERSION   "
echo "======================================"
echo
module purge
module load gcc/7.3.0 mvapich2/2.3rc2 cmake boost eigen
module list
cmake -G "CodeBlocks - Ninja" -DCMAKE_BUILD_TYPE=Release ../BayesRRcmd
ninja
