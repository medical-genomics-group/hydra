#!/bin/bash

env | grep SLURM

module purge

#set -x

echo
sh build_me.sh
##sh build_me.sh -B
echo
echo

datadir=test/data
dataset=uk10k_chr1_1mb
phen=test
S="1.0,0.1"

#datadir=~/CADMOS/Matthew/BayesRRcmd/test/data/testdata_msp_constpop_Ne10K_M100K_N10K
#dataset=testdata_msp_constpop_Ne10K_M100K_N10K
#phen=$dataset
#S="1.0,0.1"

#datadir=/scratch/orliac/testN500K
#dataset=testN500K
#phen=$dataset
#S="1.0,0.1"

echo 
echo "======================================"
echo "        RUNNING THE APPLICATION       "
echo "======================================"
echo "On the following:"
echo $datadir / $dataset / $S


# Set options here
# ----------------
NM=100000
SM=1
CL=5000
SKIP_INTEL=0
SKIP_GNU=0

for ntasks in 1 4 8 16
do
    ntasks=$(printf '%02d' "$ntasks")

    ### INTEL
    if [ $SKIP_INTEL == 0 ]; then
        module purge
        module load intel intel-mpi intel-mkl boost eigen zlib
        module list

        csv=./bayesOutput_${ntasks}.csv
        out=./bayesOutput_${ntasks}.out
        err=./bayesOutput_${ntasks}.err
        log=./bayesOutput_${ntasks}.log
        
        if [ -f $csv ]; then rm -v $csv; fi
        
        start_time="$(date -u +%s)"
        srun -o $out -e $err --ntasks $ntasks ./src/mpi_gibbs --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples $csv --seed 1 --shuf-mark $SM --mpi-sync-rate 1 --number-markers $NM --S $S
        end_time="$(date -u +%s)"
        elapsed="$(($end_time-$start_time))"
        echo "Total time in sec: $elapsed" > $log
    fi

    ### GNU
    if [ $SKIP_GNU == 0 ]; then
        module purge
        module load gcc/7.3.0 mvapich2 openblas boost eigen zlib
        module list

        csv_G=./bayesOutput_${ntasks}.csv_G
        out_G=./bayesOutput_${ntasks}.out_G
        err_G=./bayesOutput_${ntasks}.err_G
        log_G=./bayesOutput_${ntasks}.log_G
        
        if [ -f $csv_G ]; then rm -v $csv_G; fi
        
        start_time="$(date -u +%s)"
        srun -o $out_G -e $err_G --ntasks $ntasks ./src/mpi_gibbs_G --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-samples $csv_G --seed 1 --shuf-mark $SM --mpi-sync-rate 1 --number-markers $NM --S $S
        end_time="$(date -u +%s)"
        elapsed="$(($end_time-$start_time))"
        echo "Total time in sec: $elapsed" > $log_G
    fi
    
    module purge
done
