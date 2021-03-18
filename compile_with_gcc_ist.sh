
#!/bin/bash

module purge

module load openmpi/3.1.2-intel2018 eigen/3.3.7 boost/1.75.0 intel2018/compiler2018 
export EIGEN_ROOT="/mnt/nfs/clustersw/Debian/buster/eigen/3.3.7"


module list

make -f Makefile_G clean
make -f Makefile_G

