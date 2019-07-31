#!/bin/bash

#example to run brr with annotations with current solution
#marion 31.07.19

##########################################################
# test.groups					 	 #
# two annotations : g1 and g2				 #
#							 #
# sigmaG1 = 0.417105106755495				 #
# sigmaG2 = 0.12378336810673				 #
########################################################## 


export LD_LIBRARY_PATH=/software/Development/gcc/8.2.1/lib64/:$LD_LIBRARY_PATH
module add Development/gcc/8.2.1


#directory
cd BayesRRcmd/

#preprocess bed file
src/brr --preprocess --sparse-data ragged --data-file test.annot/data/uk10k_chr1_1mb.bed --pheno test.annot/phen/test.phen

#run brr
src/brr --analysis-type ppbayes --sparse-data ragged --data-file test.annot/data/uk10k_chr1_1mb.bed --S test.annot/test.cva --group test.annot/test.group --pheno test.annot/phen/test.phen --chain-length 1000 --burn-in 1 --thin 1 --mcmc-samples test.annot/output/output.bayes.csv > test.annot/output/bayes.log






