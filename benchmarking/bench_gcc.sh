#!/bin/bash

set -e
#set -x

source ../compile_with_gcc.sh $1

echo HYDRA_ROOT = $HYDRA_ROOT
echo HYDRA_EXE  = $HYDRA_EXE

BENCH_DIR=/work/ext-unil-ctgg/etienne/data_bench/
[ -d $BENCH_DIR ] || (echo "fatal: bench directory not found! $BENCH_DIR" && exit)
echo BENCH_DIR = $BENCH_DIR

OUT_DIR=/scratch/orliac/bench_hydra

NTASKS=2
NTHREADS_PER_TASK=1

export OMP_NUM_THREADS=$NTHREADS_PER_TASK

export MV2_ENABLE_AFFINITY=0
export PAMID_COLLECTIVE_ALLGATHERV=GLUE_BCAST

CMD_BASE="srun -p build -n $NTASKS --cpus-per-task $NTHREADS_PER_TASK -t 00:10:00 --mem=0 --cpu-bind=verbose,sockets \
    $HYDRA_ROOT/bin/$HYDRA_EXE \
    --pheno $BENCH_DIR/test.phen \
    --mcmc-out-dir $OUT_DIR \
    --mcmc-out-name bench_hydra_epfl_gcc \
    --mpibayes bayesMPI \
    --shuf-mark 0 \
    --seed 123 \
    --number-markers 200 \
    --number-individuals 500000 \
    --verbosity 0 \
    --chain-length 3"

# Add groups & mixtures
CMD_GROUPS="--groupIndexFile $BENCH_DIR/test.gri_hydra --groupMixtureFile $BENCH_DIR/test.grm_hydra"
#CMD_GROUPS="--groupIndexFile $BENCH_DIR/test1.gri_hydra --groupMixtureFile $BENCH_DIR/test1.grm_hydra"
CMD_BASE=${CMD_BASE}" "${CMD_GROUPS}

#BED
CMD=${CMD_BASE}" \
    --bfile $BENCH_DIR/test \
    --mcmc-out-name bench_hydra_epfl_gcc_bed"
#echo CMD = $CMD
#$CMD

#exit 0
#SPARSE
CMD=${CMD_BASE}"
    --bfile $BENCH_DIR/test \
    --mcmc-out-name bench_hydra_epfl_gcc_sparse"
#echo CMD = $CMD
#$CMD

#MIXED
CMD=${CMD_BASE}"
    --bfile $BENCH_DIR/test \
    --sparse-dir $BENCH_DIR \
    --sparse-basename test \
    --mcmc-out-name bench_hydra_epfl_gcc_mix"
CMD+=" --sparse-sync --threshold-fnz 0.06"


$CMD




