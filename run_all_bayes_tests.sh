#!/bin/bash

module load intel intel-mpi eigen boost

source ./compile_code.sh

export OMP_NUM_THREADS=4

NT=4

export SRUN_OPTS="-n $NT -c $OMP_NUM_THREADS"

sh srun_bayes_test.sh --bayes bayesMPI

sh srun_bayes_test.sh --bayes bayesFHMPI

#sh srun_bayes_test.sh --bayes bayesWMPI
