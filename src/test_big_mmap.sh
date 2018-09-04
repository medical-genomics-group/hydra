#!/bin/bash

# E. Orliac
# 04 Sep 2018
#
# Run mmap-based solution

dataset=ukb_imp_v3_UKB_EST_clumpLD09_chr22

out=$dataset.sol

echo Running mmap-based solution on dataset $dataset. Output in $out

./gctb --bayes bayesMmap --bfile /scratch/local/monthly/etienne/$dataset --pheno /scratch/local/monthly/etienne/$dataset.phen > $out

echo done
