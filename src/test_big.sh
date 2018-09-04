#!/bin/bash

# E. Orliac
# 04 Sep 2018
#
# Run reference solution

dataset=ukb_imp_v3_UKB_EST_clumpLD09_chr22

out=$dataset.ref

echo Running reference solution on dataset $dataset. Output in $out

./gctb --bayes bayes --bfile /scratch/local/monthly/etienne/$dataset --pheno /scratch/local/monthly/etienne/$dataset.phen > $out

echo done
