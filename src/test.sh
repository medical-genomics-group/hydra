#!/bin/bash

# E. Orliac
# 04 Sep 2018
#
# Run reference solution

dataset=uk10k_chr1_1mb

out=$dataset.ref

echo Running reference solution on dataset $dataset. Output in $out

./gctb --bayes bayes --bfile ../test/data/$dataset --pheno ../test/data/test.phen > $out

echo done
