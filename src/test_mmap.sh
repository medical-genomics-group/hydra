#!/bin/bash

# E. Orliac
# 04 Sep 2018
#
# Run mmap-based solution

dataset=uk10k_chr1_1mb

out=$dataset.sol

echo Running mmap-based solution on dataset $dataset. Output in $out

./gctb --bayes bayesMmap --bfile ../test/data/$dataset --pheno ../test/data/test.phen > $out

echo done
