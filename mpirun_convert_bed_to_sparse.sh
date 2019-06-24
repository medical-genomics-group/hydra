#!/bin/bash

module add UHTS/Analysis/plink/1.90
module add Development/mpi/openmpi-x86_64

EXE=~/BayesRRcmd/src/mpi_gibbs

# COMPILATION
cd ./src
B='-B'
#B=''
make $B -f Makefile_G_Unil || exit 1;
cd ..

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

#mpirun -n 80 -mca btl ^openib $EXE --bed-to-sparse --bfile /scratch/local/yearly/UKBgen/epfl_test_data --pheno /scratch/local/yearly/UKBgen/epfl_test_data.phen --sparse-dir /scratch/temporary/eorliac/ --sparse-basename epfl_test_data_sparse_TEST --blocks-per-rank 2 --number-markers 100000

mpirun -n 160 -mca btl ^openib $EXE --bed-to-sparse --bfile /scratch/local/yearly/UKBgen/epfl_test_data --pheno /scratch/local/yearly/UKBgen/epfl_test_data.phen --sparse-dir /scratch/temporary/eorliac/ --sparse-basename epfl_test_data_sparse --blocks-per-rank 100 > convert_UKBgen.log
# --number-markers 1000000

# was 7000
