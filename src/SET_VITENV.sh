module add UHTS/Analysis/plink/1.90
module add Development/mpi/openmpi-x86_64
export LDFLAGS="-Wl,--rpath=/software/lib64/ -Wl,--dynamic-linker=/software/lib64/ld-linux-x86-64.so.2 -L/lib/"
export CC=clang
export CXX=clang++
export MPICXX=clang++
export MPICC=clang
