#!/bin/bash

# E. Orliac
# 04 Sep 2018
#

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--environment)
    ENV="$2"
    shift # past argument
    shift # past value
    ;;
    -B|--force_recomp)
    B="-B"
    shift
    ;;
    -s|--solution)
    SOLUTION="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ -z ${ENV} ]; then
    echo "ENVironment not specified. Use -e|--environment."
    exit
fi
if [ ${ENV} != "EPFL" ] && [ ${ENV} != "UNIL" ] ; then
    echo "Environment can only be: EPFL or UNIL."
    exit
fi
if [ -z ${SOLUTION} ]; then
    echo "Solution not specified. Use -s|--solution."
    exit
fi
if [ ${SOLUTION} != "ref" ] && [ ${SOLUTION} != "sol" ] && [ ${SOLUTION} != "both" ]; then
    echo "Requested solution can only be: ref/sol/both"
    exit
fi
if [ -z ${DATASET} ]; then
    echo "Dataset not specified. Use -d|--dataset. E.g. -d ukb_imp_v3_UKB_EST_22."
    exit
fi

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
    tail -1 "$1"
fi

out=${DATASET}_${SOLUTION}.mmap
#echo output file is $out

datadir=""

if [ $ENV == "EPFL" ]; then
    source ./SET_EPFL_ENV.sh
    make $B -f Makefile_EPFL || exit 1
    datadir=$HOME/CADMOS/Matthew/data

    if [ ${DATASET} == "sim2" ]; then
        datadir=/scratch/orliac/BED
    fi
else
    source ./SET_VITENV.sh
    make $B -f Makefile || exit 1
    datadir=/scratch/local/monthly/Etienne
fi

### ./brr --bayes bayesMmap --bfile ../test/data/$dataset --pheno ../test/data/test.phen --chain-length 10 --burn-in 5 --thin 2 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001 > $out

chainlen=1
seed="--seed 1"
#seed=""

if [ $SOLUTION == "ref" ] || [ $SOLUTION == "both" ]; then
    echo
    echo "REFERENCE"
    echo "--------------------------------------------------------------------------------------------------------"
    ./brr --bayes bayesMmap_eo_ref   --bfile $datadir/$DATASET --pheno $datadir/$DATASET.phen --chain-length $chainlen $seed # > $out
fi

if [ $SOLUTION == "sol" ] || [ $SOLUTION == "both" ]; then
    #./brr --bayes bayesMmap2  --bfile $HOME/CADMOS/Matthew/data/$DATASET --pheno $HOME/CADMOS/Matthew/data/$DATASET.phen # > $out
    echo
    echo "NEW SOLUTION"
    echo "--------------------------------------------------------------------------------------------------------"
    #numactl --cpubind=0 --membind=0 ./brr --bayes bayesMmap_eo  --bfile $datadir/$DATASET --pheno $datadir/$DATASET.phen --chain-length $chainlen $seed # > $out
    ./brr --bayes bayesMmap_eo  --bfile $datadir/$DATASET --pheno $datadir/$DATASET.phen --chain-length $chainlen $seed # > $out
fi

#./brr --preprocess  --bfile ../test/data/$dataset --pheno ../test/data/test.phen > $out
#./brr --ppbayes  bayes --bfile ../test/data/$dataset --pheno ../test/data/test.phen>>$out

echo done
