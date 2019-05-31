#!/bin/bash

# To be run within a SLURM allocation (salloc)

module purge
module load intel intel-mpi intel-mkl boost eigen zlib
module list

EXE=./src/mpi_gibbs

# COMPILATION
cd ./src
B='-B'
B=''
make $B -f Makefile || exit 1;
cd ..

# DATASET
datadir=./test/data
dataset=uk10k_chr1_1mb
phen=test

datadir=/scratch/orliac/testN500K
dataset=testN500K
phen=$dataset

si1=$datadir/$dataset.si1; rm -v $si1
sl1=$datadir/$dataset.sl1; rm -v $sl1
ss1=$datadir/$dataset.ss1; rm -v $ss1
si2=$datadir/$dataset.si2; rm -v $si2
sl2=$datadir/$dataset.sl2; rm -v $sl2
ss2=$datadir/$dataset.ss2; rm -v $ss2

echo
echo Convert BED to SPARSE
srun --ntasks-per-node=36 $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank 33
