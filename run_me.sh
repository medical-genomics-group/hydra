#!/bin/bash

sh build_me.sh

dataset=uk10k_chr1_1mb

src/brr --bayes bayesMmap --bfile test/data/$dataset --pheno test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001

rm -v bayesOutput.csv; sbatch mpi_gibbs.sbatch
