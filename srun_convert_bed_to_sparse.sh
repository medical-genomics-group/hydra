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

#datadir=/scratch/orliac/testN500K
#dataset=testN500K
#phen=$dataset

echo
echo Convert BED to SPARSE
echo
srun -N 1 --ntasks-per-node=2 $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank 3 --sparse-dir /scratch/orliac/ABC123 --sparse-basename iamsparse2
