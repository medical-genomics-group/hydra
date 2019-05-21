#!/bin/bash

#SBATCH -N 4
#SBATCH --ntasks-per-node 36
#SBATCH --mem 180G
# SBATCH -p debug
#SBATCH -t 01:00:00

module purge
module load intel intel-mpi intel-mkl boost eigen zlib
module list

EXE=./src/mpi_gibbs


# DATASET
datadir=./test/data
dataset=uk10k_chr1_1mb
phen=test
S="1.0,0.1"

datadir=/scratch/orliac/testN500K
dataset=testN500K
phen=$dataset
S="1.0,0.1"

si1=$datadir/$dataset.si1; rm -v $si1
sl1=$datadir/$dataset.sl1; rm -v $sl1
ss1=$datadir/$dataset.ss1; rm -v $ss1
si2=$datadir/$dataset.si2; rm -v $si2
sl2=$datadir/$dataset.sl2; rm -v $sl2
ss2=$datadir/$dataset.ss2; rm -v $ss2

echo
echo Convert BED to SPARSE
srun $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen
