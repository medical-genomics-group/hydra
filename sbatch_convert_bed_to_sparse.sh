#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem 180G
#SBATCH -p debug
#SBATCH -t 01:00:00

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
srun $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank 2 --sparse-dir /scratch/orliac/ABC123 --sparse-basename iamsparse3
