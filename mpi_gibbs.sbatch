#!/bin/bash

#SBATCH --partition debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 10G

set +x

env | grep SLURM

module load intel intel-mpi eigen boost

dataset=uk10k_chr1_1mb

#srun ./src/mpi_gibbs --mpibayes bayesMPI --bfile test/data/$dataset --pheno test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001

srun ./src/mpi_gibbs --mpibayes bayesMPI --bfile test/data/$dataset --pheno test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --seed 1 --shuf-mark 1 --mpi-sync-rate 8 --number-markers 2000 --S 0.01,0.001
