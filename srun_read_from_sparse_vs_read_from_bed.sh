#!/bin/bash
#
# Author : E. Orliac, DCSR, UNIL
# Date   : 2019/05/21
# Purpose: Compare solutions when reading from BED file or SPARSE representation files.
#          Results should be strictly equivalent.
#
# Warning: needs to be in an active slurm allocation, execution via srun!
#
# Warning: the block definition file (.blk) and job setup must match (wrt the number of tasks)
#


module purge
module load intel intel-mpi intel-mkl boost eigen zlib
module list

icc beta_converter.c       -o beta_converter
icc epsilon_converter.c    -o epsilon_converter
icc components_converter.c -o components_converter

EXE=./src/mpi_gibbs

# COMPILATION
cd ./src
B='-B'
B=''
make $B -f Makefile || exit 1;
cd ..

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

# DATASET
datadir=./test/data
dataset=uk10k_chr1_1mb
phen=test
S="1.0,0.1"

datadir=/scratch/orliac/testM100K_N5K_missing
dataset=memtest_M100K_N5K_missing0.01
phen=memtest_M100K_N5K_missing0.01
sparsedir=/scratch/orliac/CTGG/memtest_M100K_N5K_missing0_01
sparsebsn=memtest_M100K_N5K_missing0_01

#datadir=/scratch/orliac/testN500K
#dataset=testN500K
#phen=$dataset
#S="1.0,0.1"

echo 
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo "datadir:   " $datadir
echo "dataset:   " $dataset
echo "sparse dir:" $sparsedir
echo "sparse bsn:" $sparsebsn
echo "S         :" $S
echo "======================================"
echo

CL=100
SEED=10
SR=0
SM=1
NM=1000
THIN=3
SAVE=3
TOCONV_T=$(($CL / $THIN))
echo TOCONV_T $TOCONV_T
N=1
TPN=17

echo "@@@ Solution reading from  BED file @@@"
sol=from_bed
srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file 
# --covariates scaled_covariates.csv
# --number-markers $NM

./beta_converter       $sol".bet" $TOCONV_T > $sol".bet.txt"
./epsilon_converter    $sol".eps"           > $sol".eps.txt"
./components_converter $sol".cpn" $TOCONV_T > $sol".cpn.txt"

echo; echo

#echo __EARLY_EXIT__
#exit 0

echo "@@@ Solution reading from SPARSE files @@@"
sol2=from_sparse
srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol2 --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn  
# --covariates scaled_covariates.csv
# --marker-blocks-file $datadir/${dataset}.blk --number-markers $NM

./beta_converter       $sol2".bet" $TOCONV_T > $sol2".bet.txt"
./epsilon_converter    $sol2".eps"           > $sol2".eps.txt"
./components_converter $sol2".cpn" $TOCONV_T > $sol2".cpn.txt"

#exit
#echo; echo

echo diff bin .bet
diff $sol".bet" $sol2".bet"
echo diff bin .eps
diff $sol".eps" $sol2".eps"
echo diff bin .cpn
diff $sol".cpn" $sol2".cpn"
echo diff txt .bet
diff $sol".bet.txt" $sol2".bet.txt"
echo diff txt .eps
diff $sol".eps.txt" $sol2".eps.txt"
echo diff txt .cpn
diff $sol".cpn.txt" $sol2".cpn.txt"
