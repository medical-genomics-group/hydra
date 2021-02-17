#!/bin/bash

#SBATCH --partition debug
#SBATCH --mem 0
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 4
#SBATCH --time 00:30:00

set -e
#set -x

COMPILER=gcc

print_help() {
cat <<-HELP

Call the script like this:

sbatch $0 -c|--compiler {gcc|intel}, default is gcc

HELP
exit 0
}

# Parse Command Line Arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --compiler*|-c*)
            if [[ "$1" != *=* ]]; then shift; fi
            COMPILER="${1#*=}"
            ;;
        --help|-h)
            print_help;;
        *)
            printf "************************************************************\n"
            printf "* Error: Invalid argument, run --help for valid arguments. *\n"
            printf "************************************************************\n"
            exit 1
            ;;
    esac
    shift
done

if [ $COMPILER == "gcc" ]; then
    echo "GCC"
    source ../compile_with_gcc.sh $1
    #export GOMP_CPU_AFFINITY=verbose
elif [ $COMPILER == "intel" ]; then
    echo "INTEL"
    source ../compile_with_intel.sh $1
    #export KMP_AFFINITY=verbose
else 
    echo "fatal: unknown compiler $COMPILER. Check --help."
fi

PGM_ROOT=$HYDRA_ROOT
PGM_EXE=$HYDRA_EXE


#source /ssoft/spack/external/intel/2018.4/vtune_amplifier/amplxe-vars.sh
AMPLXE_VARS=/ssoft/spack/external/intel/2018.4/vtune_amplifier/amplxe-vars.sh
[ -f $AMPLEX_VARS ] || (echo "fatal: $AMPLXE_VARS not found." && exit 1)
source $AMPLXE_VARS
#source /ssoft/spack/external/intel/2018.4/advisor/advixe-vars.sh
ADVIXE_VARS=/ssoft/spack/external/intel/2018.4/advisor/advixe-vars.sh
[ -f $ADVIXE_VARS ] || (echo "fatal: $ADVIXE_VARS not found." && exit 1)
source $ADVIXE_VARS

export MODULEPATH=/ssoft/spack/humagne/v1/share/spack/lmod/linux-rhel7-x86_S6g1_Mellanox/intel/18.0.5:$MODULEPATH
export INTEL_LICENSE_FILE=/ssoft/spack/external/intel/License:$INTEL_LICENSE_FILE

BENCH_DIR=/work/ext-unil-ctgg/etienne/data_bench/
[ -d $BENCH_DIR ] || (echo "fatal: bench directory not found! $BENCH_DIR" && exit)
echo BENCH_DIR = $BENCH_DIR

OUT_DIR=/scratch/orliac/bench_${PGM_EXE}
[ -d $OUT_DIR ] && rm -rv $OUT_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEARCH_DIRS="--search-dir src:=$PGM_ROOT/src --search-dir sym:=$PGM_ROOT/build_${COMPILER} --search-dir bin:=$PGM_ROOT/bin"

AMPLXE1="amplxe-cl -c hotspots    -r $OUT_DIR/vtune  -data-limit=0 -- " #-no-auto-finalize $SEARCH_DIRS"
#AMPLXE1="amplxe-cl -c concurrency -r $OUT_DIR/vtune  -data-limit=0 -- " #-no-auto-finalize $SEARCH_DIRS"

ADVIXE1="advixe-cl --collect survey           -no-auto-finalize -project-dir=$OUT_DIR/advisor  $SEARCH_DIRS -data-limit=0 -- "
ADVIXE2="advixe-cl --collect tripcounts -flop -no-auto-finalize -project-dir=$OUT_DIR/advisor  $SEARCH_DIRS -data-limit=0 -- "

CMD_BASE="srun --cpu-bind=verbose,sockets"

CMD_TAIL="$PGM_ROOT/bin/$PGM_EXE \
--bfile $BENCH_DIR/test.bed \
--pheno $BENCH_DIR/test1.phen \
--mcmc-out-dir $OUT_DIR \
--mcmc-out-name bench_${PGM_EXE}_epfl \
--mpibayes bayesMPI \
--shuf-mark 0 \
--seed 123 \
--number-markers 200 \
--number-individuals 500000 \
--S 0.0001,0.001,0.01 \
--verbosity 0 \
--chain-length 1"

CMD_TAIL="${CMD_TAIL} \
    --sparse-dir $BENCH_DIR \
    --sparse-basename test \
    --mcmc-out-name bench_${PGM_EXE}_epfl_mix \
    --sparse-sync --threshold-fnz 0.06"

#BED
env | grep SLURM_
env | grep OMP_

CMD="${CMD_BASE} ${AMPLXE1} ${CMD_TAIL}"
echo CMD = $CMD
#$CMD

CMD="${CMD_BASE} ${ADVIXE1} ${CMD_TAIL}"
echo CMD = $CMD
$CMD
CMD="${CMD_BASE} ${ADVIXE2} ${CMD_TAIL}"
echo CMD = $CMD
$CMD

echo 
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "To visualize the results:"
echo
echo "module load intel"
echo "source $AMPLXE_VARS"
echo "source $ADVIXE_VARS"
echo
echo "amplxe-gui ${OUT_DIR}/vtune/vtune.amplxe &"
echo "advixe-gui ${OUT_DIR}/advisor/advisor.advixeproj &"
echo " ^^^ SELECT e001 not e000 for roofline ;-)" 
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
