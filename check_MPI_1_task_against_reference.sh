#!/bin/bash

#
# ATTENTION: need to be in an active slurm allocation!
#

module purge
module load intel intel-mpi intel-mkl boost eigen zlib

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

echo 
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo $datadir / $dataset / $S
echo "======================================"
echo

CL=100
SEED=5
SR=0
SM=0
NM=10000

sol=devel
rm $sol.csv
rm $sol.bet

echo 
echo
echo "DEVEL SOLUTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo
srun -N 1 --ntasks 1 --ntasks-per-node=1 $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples $sol.csv --mcmc-betas $sol.bet --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --number-markers $NM
## --m2skip $M2S --mstart $MS

#exit 0

echo 
echo
echo "OFFICIAL SOLUTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo
$EXE --bayes bayesMmap --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples ref.csv --shuf-mark $SM --seed $SEED --S $S --number-markers $NM
