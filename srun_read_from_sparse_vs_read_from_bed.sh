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
module load intel intel-mpi intel-mkl boost eigen
module list

if [ 1  == 1 ]; then 
    echo compiling utilities
    icc beta_converter.c       -o beta_converter
    icc epsilon_converter.c    -o epsilon_converter
    icc components_converter.c -o components_converter
    echo end utilities compilation
fi    

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

S="1.0,0.1"

DS=3

if [ $DS == 0 ]; then
    datadir=./test/data
    dataset=uk10k_chr1_1mb
    phen=test
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=3642
    NUMSNPS=6717
elif [ $DS == 1 ]; then
    datadir=/scratch/orliac/testM100K_N5K_missing
    dataset=memtest_M100K_N5K_missing0.01
    phen=memtest_M100K_N5K_missing0.01
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=5000
    NUMSNPS=117148
elif [ $DS == 2 ]; then
    datadir=/scratch/orliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint_test
    NUMINDS=500000
    NUMSNPS=1270420
    NUMSNPS=10000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=1000
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 4 ]; then
    datadir=/scratch/orliac/testNA/
    dataset=test_nm
    phen=test_m
    NUMINDS=20000
    NUMSNPS=50000
    S="0.00001,0.0001,0.001,0.01"
fi


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

CL=5
SEED=10
SR=0
SM=1
THIN=1
SAVE=1
TOCONV_T=$((($CL - 1) / $THIN))
echo TOCONV_T $TOCONV_T
N=1
TPN=1

# Set what to run
run_bed=0; run_sparse=1; run_comp=0;

COV="--covariates $datadir/scaled_covariates.csv"
COV=""
BLK="--marker-blocks-file $datadir/${dataset}.blk"
BLK=""

if [ $run_bed == 1 ]; then
    echo; echo
    echo "@@@ Solution reading from  BED file @@@"
    echo
    sol=test_mnm2
    cmd="-N $N --ntasks-per-node=$TPN  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file $COV $BLK"
    echo $cmd
    srun $cmd || exit 1
    rm $sol".bet.txt" $sol".eps.txt" $sol".cpn.txt"
    ./beta_converter       $sol".bet" $TOCONV_T > $sol".bet.txt"
    ./epsilon_converter    $sol".eps"           > $sol".eps.txt"
    ./components_converter $sol".cpn" $TOCONV_T > $sol".cpn.txt"
fi


if [ $run_sparse == 1 ]; then
    echo; echo
    echo "@@@ Solution reading from SPARSE files @@@"
    echo
    sol2=from_sparse2
    cmd="-N $N --ntasks-per-node=$TPN  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol2 --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $COV $BLK"
    echo $cmd
    srun $cmd || exit 1

    rm $sol2".bet.txt" $sol2".eps.txt" $sol2".cpn.txt" 
    ./beta_converter       $sol2".bet" $TOCONV_T > $sol2".bet.txt"
    ./epsilon_converter    $sol2".eps"           > $sol2".eps.txt"
    ./components_converter $sol2".cpn" $TOCONV_T > $sol2".cpn.txt"
fi

echo; echo

if [ $run_comp == 1 ] && [ $run_bed == 1 ] && [ $run_sparse == 1 ]; then
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
fi
