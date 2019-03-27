#!/bin/bash

# Author : E. Orliac, DCSR, UNIL
# Date   : 26.03.2019
# Purpose: Handle compilation for sequential and MPI versions of the code

set +x


echo 
echo "==============================="
echo "   COMPILING THE MPI VERSION   "
echo "==============================="
echo
module purge
module load intel intel-mpi intel-mkl intel-tbb boost eigen zlib
env | grep TBB
module list

cd src

make
# -B

#exit 0

#
## COMPILATION OF THE SEQUENTIAL VERSION
#

cd ..

module purge
module load gcc/7.3.0 mvapich2/2.3rc2 cmake boost eigen
module list

cmake -G "CodeBlocks - Ninja" -DCMAKE_BUILD_TYPE=Release ../BayesRRcmd
ninja
