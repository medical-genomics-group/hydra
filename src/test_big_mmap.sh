#!/bin/bash

# E. Orliac
# 04 Sep 2018
#
# Run mmap-based solution


me=`basename "$0"`

# Make sure OMP_NUM_THREADS is properly set 
# -----------------------------------------
if [ -z "$OMP_NUM_THREADS" ]; then
    echo "ERROR: you forgot to define environment variable OMP_NUM_THREADS!"
    exit
fi
echo Will run $me with OMP_NUM_THREADS set to $OMP_NUM_THREADS

dataset=ukb_imp_v3_UKB_EST_clumpLD09_chr22

sel=$1

if [ "$sel" != "1" ] && [ "$sel" != "2" ] && [ "$sel" != "12" ]; then
    echo "ERROR: no valid selection passed! Valid options are: 1, 2, or 12."
    echo "Launch as e.g.: sh $me 12"
fi

if [ "$sel" = "1" ] || [ "$sel" = "12" ]; then
    out=$dataset.sol
    echo Running mmap-based solution on dataset $dataset. Output in $out
    ./gctb --bayes bayesMmap --bfile /scratch/local/monthly/etienne/$dataset --pheno /scratch/local/monthly/etienne/$dataset.phen > $out
fi

if [ "$sel" = "2" ] || [ "$sel" = "12" ]; then
    out=$dataset.sol2
    echo Running mmap+OpenMP-based solution on dataset $dataset. Output in $out
    ./gctb --bayes bayesMmap2 --bfile /scratch/local/monthly/etienne/$dataset --pheno /scratch/local/monthly/etienne/$dataset.phen > $out
fi
