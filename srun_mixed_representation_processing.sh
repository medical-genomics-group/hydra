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


# COMPILATION
cd ./src
B='-B'
B=''
export EXE=hydra
make $B -f Makefile || exit 1;
cd ..

EXE=./src/$EXE

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

S="1.0,0.1"

DS=5

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
    phen=${dataset}_NA
    phen=${dataset}
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=500000
    NUMSNPS=1270420
    NUMSNPS=60000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=50
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 4 ]; then
    datadir=/scratch/orliac/testNA/
    dataset=test_nm
    phen=test_m
    NUMINDS=20000
    NUMSNPS=100000
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 5 ]; then
    datadir=/scratch/orliac/ukb_sparse
    dataset=ukb_imp_v3_UKB_EST_oct19_unrelated
    phen=height_noNA
    sparsedir=$datadir
    sparsebsn=$dataset
    NUMINDS=382466
    NUMSNPS=8430446
    NUMSNPS=100000
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

CL=30
SEED=10
SR=5
SM=1
FNZ=0.060
THIN=1
SAVE=100
TOCONV_T=$((($CL - 1) / $THIN))
echo TOCONV_T $TOCONV_T
N=1
TPN=6

CPT=6 # CPUs per task
export OMP_NUM_THREADS=$CPT
export KMP_AFFINITY=verbose
export KMP_AFFINITY=noverbose

# Select what to run
run_bed_sync_dp=0;      run_sparse_sync_dp=0;      run_mixed_sync_dp=0;
run_bed_sync_sparse=0;  run_sparse_sync_sparse=0;  run_mixed_sync_sparse=1;
run_bed_sync_bed=0;     run_sparse_sync_bed=0;     run_mixed_sync_bed=0;


COV="--covariates $datadir/scaled_covariates.csv"
COV=""
BLK="--marker-blocks-file $datadir/${dataset}.blk"
BLK=""

outdir=/home/orliac/DCSR/CTGG/BayesRRcmd/output_tests/


### BED data processing

if [ $run_bed_sync_dp == 1 ]; then
    echo; echo
    echo "@@@ BED + DP @@@"
    echo
    sol=test_mnm2
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_bed_sync_sparse == 1 ]; then
    echo; echo
    echo "@@@ BED + SPARSE @@@"
    echo
    sol=test_mnm2
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-sync $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_bed_sync_bed == 1 ]; then
    echo; echo
    echo "@@@ BED + BED @@@"
    echo
    sol=test_mnm2
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bed-sync $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


### SPARSE data processing

if [ $run_sparse_sync_dp == 1 ]; then
    echo; echo; echo "@@@ SPARSE + DP @@@"
    echo
    sol2=sparse_sync_dp
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync  $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_sparse_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ SPARSE + SPARSE @@@"; echo
    sol2=sparse_sync_sparse
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync  $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_sparse_sync_bed == 1 ]; then
    echo; echo; echo "@@@ SPARSE + BED @@@"; echo
    sol2=sparse_sync_bed
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn --bed-sync  $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


### MIXED-representation data processing

if [ $run_mixed_sync_dp == 1 ]; then
    echo; echo; echo "@@@ MIXED + DP @@@"; echo
    sol2=mixed_sync_dp
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz $FNZ $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


AMPLIFIER=0;  ampdir=/scratch/orliac/tmp_vtune_profile;
ADVISOR=0;    advdir=/scratch/orliac/tmp_advis_profile;

if [ $run_mixed_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ MIXED + SPARSE @@@"; echo
    sol2=mixed_sync_sparse

     CMD="$EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz $FNZ --sparse-sync $COV $BLK"

     PRE="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT "

    if [ $AMPLIFIER == 1 ]; then
        echo "@@@ AMPLIFIER"
        [ -d $ampdir ] && rm -r $ampdir/*
        cmd_prefix=$PRE" amplxe-cl –c hotspots –r $ampdir -data-limit=1000 -- "

        cmd=$cmd_prefix$CMD
        echo $cmd; echo
        $cmd || exit 1

        amplxe-gui $ampdir/tmp_vtune_profile.amplxeproj &

    elif [ $ADVISOR == 1 ]; then
        echo "@@@ ADVISOR"
        source ~/load_Intel_Advisor.sh
        [ -d $advdir ] && rm -r $advdir/*

        cmd_prefix="advixe-cl -v -collect survey           --project-dir=$advdir -no-auto-finalize -data-limit=0 --search-dir all:=./src -- "
        cmd=$PRE$cmd_prefix$CMD
        echo $cmd; echo
        $cmd || exit 1

        cmd_prefix="advixe-cl -v -collect tripcounts -flop --project-dir=$advdir -no-auto-finalize -data-limit=0 --search-dir all:=./src -- "
        cmd=$PRE$cmd_prefix$CMD
        echo $cmd; echo
        $cmd || exit 1
        
        #cmd_prefix="advixe-cl --collect=roofline          --project-dir=$advdir  --no-auto-finalize                  --search-dir all:=./src -- "
        #cmd=$PRE$cmd_prefix$CMD
        #echo $cmd; echo
        #$cmd || exit 1

        echo "advixe-gui $advdir/tmp_advis_profile.advixeproj &"

    else
        cmd=$PRE$CMD
        echo $cmd; echo
        $cmd || exit 1
    fi
fi

if [ $run_mixed_sync_bed == 1 ]; then
    echo; echo; echo "@@@ MIXED + BED @@@"; echo
    #cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  cpuinfo"
    #echo $cmd
    #$cmd || exit 1

    #cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  ${HOME}/DCSR/Affinity/xthi_mpi"
    #echo $cmd
    #$cmd || exit 1

    sol2=mixed_sync_bed
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz $FNZ --bed-sync $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi
