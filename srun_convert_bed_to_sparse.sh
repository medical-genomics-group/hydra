#!/bin/bash

# To be run within a SLURM allocation (salloc)

module purge
module load intel intel-mpi intel-mkl boost eigen zlib
module list

EXE=./src/mpi_gibbs

# COMPILATION
cd ./src
echo REMOVE THAT ONE
touch main.cpp 
B='-B'
B=''
make $B -f Makefile || exit 1;
cd ..

# DATASET
datadir=""
dataset=""
sparsedir=""
sparsebsn=""
phen=""

DS=2

if [ $DS == 0 ]; then 
    datadir=./test/data
    dataset=uk10k_chr1_1mb
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    phen=test
elif [ $DS == 1 ]; then
    datadir=/scratch/orliac/testM100K_N5K_missing
    dataset=memtest_M100K_N5K_missing0.01
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    phen=memtest_M100K_N5K_missing0.01
elif [ $DS == 2 ]; then
    datadir=/scratch/orliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint_test
fi

echo datadir: $datadir

N=1
TPN=30
BPR=5

echo
echo Convert BED to SPARSE
echo

srun -N $N --ntasks-per-node=$TPN $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank $BPR --sparse-dir $sparsedir --sparse-basename $sparsebsn
