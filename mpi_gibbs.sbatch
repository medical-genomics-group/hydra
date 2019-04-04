#!/bin/bash

#SBATCH --partition debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 10G
#SBATCH -t 00:10:00

set -x

env | grep SLURM

module load intel intel-mpi eigen boost

datadir=test/data
dataset=uk10k_chr1_1mb

datadir=~/CADMOS/Matthew/BayesRRcmd/test/data/testdata_msp_constpop_Ne10K_M100K_N10K
dataset=testdata_msp_constpop_Ne10K_M100K_N10K

#srun ./src/mpi_gibbs --mpibayes bayesMPI --bfile test/data/$dataset --pheno test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001

#srun ./src/mpi_gibbs --mpibayes bayesMPI --bfile test/data/$dataset --pheno test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --seed 1 --shuf-mark 1 --mpi-sync-rate 8 --number-markers 20000 --S 0.01,0.001

#srun ./src/mpi_gibbs --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${dataset}.phen --chain-length 100 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --seed 1 --shuf-mark 1 --mpi-sync-rate 8 --number-markers 100 --S 0.01,0.001

CSV=./bayesOutput.csv

if [ -f $CSV ]; then 
    rm -v $CSV;
fi

srun ./src/mpi_gibbs --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${dataset}.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples $CSV --seed 1 --shuf-mark 1 --mpi-sync-rate 1 --number-markers 100 --S 1.0,0.1
