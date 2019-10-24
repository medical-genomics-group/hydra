#!/bin/bash

module purge
module load intel intel-mpi boost eigen
module list

#export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH

NAM=mpi_gibbs
cd ./src
make EXE=$NAM $1 -f Makefile || exit 1;
cd ..

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

PWD=`pwd`
export EXE=$PWD/src/$NAM
