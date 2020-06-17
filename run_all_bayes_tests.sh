#!/bin/bash

module load intel intel-mpi eigen boost

source ./compile_code.sh

export OMP_NUM_THREADS=4

NT=1

export SRUN_OPTS="-n $NT -c $OMP_NUM_THREADS"

echo; echo "@@@@@@ BAYES R"
#sh srun_bayes_test.sh --bayes bayesMPI


echo; echo "@@@@@@ BAYES FH"
#sh srun_bayes_test.sh --bayes bayesFHMPI


echo; echo "@@@@@@ BAYES W"
sh srun_temp_bayesW_test.sh --bayes bayesWMPI
