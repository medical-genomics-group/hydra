#!/bin/bash

module purge
module load intel intel-mpi boost eigen
module list

NAM=hydra

cd ./src

make EXE=$NAM $1 -f Makefile || exit 1;

cd ..

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

PWD=`pwd`
export EXE=$PWD/src/$NAM
