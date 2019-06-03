#!/bin/bash
#
# Author : E. Orliac, DCSR, UNIL
# Date   : 2019/05/21
# Purpose: Compare solutions when reading from BED file or SPARSE representation files.
#          Results should be strictly equivalent.
#
# Warning: needs to be in an active slurm allocation, execution via srun!
#
# Warning: the block definition file (.blk) and job setup must match (wrt the number of tasks)
#


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

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

# DATASET
datadir=./test/data
dataset=uk10k_chr1_1mb
phen=test
S="1.0,0.1"

#datadir=/scratch/orliac/testN500K
#dataset=testN500K
#phen=$dataset
#S="1.0,0.1"

echo 
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo "datadir:" $datadir
echo "dataset:" $dataset
echo "S      :" $S
echo "======================================"
echo

CL=3
SEED=5
SR=0
SM=0
NM=100

N=1
TPN=3

echo "@@@ Solution reading from  BED file @@@"
sol=bed
rm $sol.csv
rm $sol.bet
srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples $sol.csv --mcmc-betas $sol.bet --mcmc-epsilon $sol.epsilon --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file --covariates scaled_covariates.csv
#  --number-markers $NM
echo; echo

#echo __EARLY_EXIT__
#exit 0


echo "@@@ Solution reading from SPARSE files @@@"
sol=sparse
rm $sol.csv
rm $sol.bet
srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples $sol.csv --mcmc-betas $sol.bet --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir /scratch/orliac/ABC123 --sparse-basename iamsparse3
# --marker-blocks-file $datadir/${dataset}.blk --number-markers $NM
