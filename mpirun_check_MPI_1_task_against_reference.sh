#!/bin/bash

module add UHTS/Analysis/plink/1.90
module add Development/mpi/openmpi-x86_64

EXE=./src/mpi_gibbs

# COMPILATION
cd ./src
B='-B'
B=''
make $B -f Makefile_G_Unil || exit 1;
cd ..

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

DS=4

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
    sparsebsn=epfl_test_data_sparse_V2
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=500000
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 4 ]; then
    datadir=/scratch/local/yearly/UKBgen/
    dataset=epfl_test_data
    sparsedir=/scratch/temporary/eorliac/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=2341
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

CL=10
SEED=10
SR=0
SM=0
THIN=3
SAVE=3
TOCONV_T=$((($CL - 1) / $THIN))
echo TOCONV_T $TOCONV_T

# WARNING:
# If you change those, do not expect compatibility! ###
N=1                                                   #
TPN=1                                                 #
#######################################################

echo 
echo
echo "@@@ Official (sequential) solution (reading from BED file) @@@"
echo
sol=seq_bed
cmd="-n $TPN  $EXE --bayes bayesMmap --number-markers $NUMSNPS   --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file $COV $BLK"
echo "$cmd"; echo
mpirun -mca btl ^openib $cmd |& tee -a $sol.log || exit 1

#REF# mpirun -N $N --ntasks-per-node=$TPN $EXE --bayes bayesMmap --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-out ref --shuf-mark $SM --seed $SEED --S $S --number-markers $NM 
#--covariates $datadir/scaled_covariates.csv

echo
echo
echo "@@@ Solution reading from  BED file @@@"
sol=mpi1tbed
cmd="-n $TPN $EXE --mpibayes bayesMPI --number-individuals $NUMINDS --number-markers $NUMSNPS --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1 --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file"
echo "$cmd"; echo
mpirun -mca btl ^openib $cmd |& tee -a $sol.log || exit 1

#REF# srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1 --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file --number-markers $NM 
#--covariates $datadir/scaled_covariates.csv
echo; echo


echo "@@@ Solution reading from SPARSE files @@@"
sol=mpi1tsparse
cmd="-n $TPN $EXE --mpibayes bayesMPI --number-individuals $NUMINDS --number-markers $NUMSNPS --pheno $datadir/${phen}.phen --chain-length $CL --thin 1 --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir --sparse-basename $sparsebsn"
echo "$cmd"; echo
mpirun -mca btl ^openib $cmd |& tee -a $sol.log || exit 1

#REF# srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1 --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --number-markers $NM --sparse-dir $sparsedir --sparse-basename $sparsebsn
#--marker-blocks-file $datadir/${dataset}.blk_1 
#--covariates $datadir/scaled_covariates.csv

