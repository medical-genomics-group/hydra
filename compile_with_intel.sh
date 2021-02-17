#!/bin/bash

set -e

# before loading modules!!!
export HYDRA_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo HYDRA_ROOT = $HYDRA_ROOT
[[ "$HYDRA_ROOT" == *hydra ]] || (echo "HYDRA_ROOT ($HYDRA_ROOT) expected to end with \"hydra\"" && exit 1)

module purge
module load intel intel-mpi boost eigen
module list

cd $HYDRA_ROOT
HYDRA_EXE=hydra_i
EXEC=$HYDRA_EXE make -j8 $1 || exit 1
export HYDRA_EXE=$HYDRA_EXE
cd -
