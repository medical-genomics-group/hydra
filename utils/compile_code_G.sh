#!/bin/bash

module purge
module load gcc mvapich2 boost eigen
module list

#LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH

NAM=hydra_G

cd ./src

make EXE=$NAM $1 -f Makefile_G || exit 1;

cd ..

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

PWD=`pwd`
export EXE=$PWD/src/$NAM
