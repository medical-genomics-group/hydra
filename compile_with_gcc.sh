#!/bin/bash

set -e

# before loading modules!!!
export HYDRA_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo HYDRA_ROOT = $HYDRA_ROOT
[[ "$HYDRA_ROOT" == *hydra ]] || (echo "HYDRA_ROOT ($HYDRA_ROOT) expected to end with \"hydra\"" && exit 1)

module purge
module load gcc/8.3.0 mvapich2 boost #eigen
module list

# Local install of eigen to be able to use gcc/8.3.0
export EIGEN_ROOT=/home/orliac/inc/Eigen-3.3.7

cd $HYDRA_ROOT
HYDRA_EXE=hydra_g
EXEC=$HYDRA_EXE make -j8 $1 || exit 1
export HYDRA_EXE=$HYDRA_EXE
cd -
