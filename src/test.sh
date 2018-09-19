#!/bin/bash

# E. Orliac
# 04 Sep 2018
#
# Run reference solution

dataset=ukb_imp_v3_UKB_EST_clumpLD09_chr22

out=$dataset.mmap

echo Running mmap solution on dataset $dataset. Output in $out

<<<<<<< HEAD
./brr --bayes bayesMmap --bfile ../test/data/$dataset --pheno ../test/data/test.phen --chain-length 10 --burn-in 5 --thin 2 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001 > $out
=======
./brr --bayes bayesMmap --bfile ../../../etienne/$dataset --pheno ../../../etienne/$dataset.phen --chain-length 10 --burn-in 5 --thin 2 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001 > $out
>>>>>>> branch 'master' of https://github.com/ctggroup/BayesRRcmd

out=$dataset.pp
echo Running preprocessing solution on dataset $dataset. Output in $out

#./brr --preprocess  --bfile ../test/data/$dataset --pheno ../test/data/test.phen > $out
#./brr --ppbayes  bayes --bfile ../test/data/$dataset --pheno ../test/data/test.phen>>$out
echo done
