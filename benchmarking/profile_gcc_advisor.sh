#!/bin/bash

#SBATCH --partition build
#SBATCH --mem 10GB
#SBATCH --cpus-per-task 4
#SBATCH --ntasks 2

set -e
#set -x

source ../compile_with_gcc.sh $1
echo HYDRA_ROOT = $HYDRA_ROOT
echo HYDRA_EXE  = $HYDRA_EXE

ADVIXE_VARS=/ssoft/spack/external/intel/2018.4/advisor/advixe-vars.sh
[ -f $ADVIXE_VARS ] || (echo "fatal: $ADVIXE_VARS not found." && exit 1)
source $ADVIXE_VARS
export MODULEPATH=/ssoft/spack/humagne/v1/share/spack/lmod/linux-rhel7-x86_S6g1_Mellanox/intel/18.0.5:$MODULEPATH

export INTEL_LICENSE_FILE=/ssoft/spack/external/intel/License:$INTEL_LICENSE_FILE

BENCH_DIR=/work/ext-unil-ctgg/etienne/data_bench/
[ -d $BENCH_DIR ] || (echo "fatal: bench directory not found! $BENCH_DIR" && exit)
echo BENCH_DIR = $BENCH_DIR

OUT_DIR=/scratch/orliac/bench_hydra2
[ -d $OUT_DIR ] && rm -rv $OUT_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

env | grep SLURM_
env | grep OMP_


SEARCH_DIRS="--search-dir src:=$HYDRA_ROOT/src --search-dir sym:=$HYDRA_ROOT/build_gcc --search-dir bin:=$HYDRA_ROOT/bin"

ADVIXE1="advixe-cl --collect=survey            --project-dir=$OUT_DIR/advisor -no-auto-finalize $SEARCH_DIRS -data-limit=0 --"
ADVIXE2="advixe-cl --collect=tripcounts --flop --project-dir=$OUT_DIR/advisor -no-auto-finalize $SEARCH_DIRS -data-limit=0 --"

CMD_BASE="srun -t 00:10:00 --cpu-bind=verbose,sockets"

echo $HYDRA_ROOT

CMD_TAIL="$HYDRA_ROOT/bin/$HYDRA_EXE \
    --bfile $BENCH_DIR/test \
    --pheno $BENCH_DIR/test.phen \
    --mcmc-out-dir $OUT_DIR \
    --mcmc-out-name bench_hydra_epfl_gcc \
    --mpibayes bayesMPI \
    --shuf-mark 0 \
    --seed 123 \
    --number-markers 50 \
    --number-individuals 500000 \
    --S 0.0001,0.001,0.01 \
    --verbosity 0 \
    --chain-length 1"

CMD_TAIL="${CMD_TAIL} \
    --sparse-dir $BENCH_DIR \
    --sparse-basename test \
    --mcmc-out-name bench_hydra_epfl_gcc_mix \
    --sparse-sync"

CMD="${CMD_BASE} ${ADVIXE1} ${CMD_TAIL}"
echo CMD ?1 $CMD
$CMD

CMD="${CMD_BASE} ${ADVIXE2} ${CMD_TAIL}"
echo CMD ?2 $CMD
$CMD

