
module purge
#module load cuda/10.1.243
#module load gcc/10.2
module load eigen/3.3.7
module load boost/1.75.0
module load intel/oneApi/20210114
module load impi/2021.1.1

#module load gcc/9.3
#module load intelmpi/impi-2018.2.199
#module load openmpi/4.1.0b
module list


#export EIGEN_ROOT="/mnt/nfs/clustersw/Debian/buster/eigen/3.3.7b"
export EIGEN_ROOT="/mnt/nfs/clustersw/Debian/buster/eigen/3.3.9"

rm *.d *.o hydra

make -f Makefile_I clean
make -f Makefile_I

#mpic++ -Wall -Wextra -Ofast -std=c++17 -D USE_MPI -fopenmp \
#-I/mnt/nfs/clustersw/Debian/buster/eigen/3.3.9/include/eigen3/Eigen \
#-I/include -lz  -MMD -MP -c main.cpp -o main.o
