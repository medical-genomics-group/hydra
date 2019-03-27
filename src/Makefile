#
# Purpose: Makefile for compile the MPI based asynchronous Gibbs sampler
# Author : E. Orliac
# Date   : 22.01.2019
#

EXE      = mpi_gibbs
CXX      = mpiicpc

CXXFLAGS  = -g -O3 -Wall 
#CXXFLAGS += -std=c++11 -qopenmp
CXXFLAGS += -std=c++17

#CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high
#CXXFLAGS += -xCORE-AVX2, -axCORE-AVX512
#EO: Set this back!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CXXFLAGS += -xCORE-AVX2 
CXXFLAGS += -qopenmp-simd
#CXXFLAGS += -fp-model precise -fp-model source
#CXXFLAGS += -DISABLE_MPI

CXXFLAGS += -D USE_MPI

INCLUDE  = -I/home/orliac/external_soft/pcg-c/include
INCLUDE += -I$(EIGEN_ROOT)/include/eigen3
INCLUDE += -I$(BOOST_ROOT)/include
INCLUDE += -I$(INTEL_TBB_ROOT)/include

LIBS     = -L$(INTEL_TBB_ROOT)/lib -ltbb -lz

#INCLUDE  += -I/home/orliac/DCSR/CTGG/BayesRRcmd/tbb/include/

#SRC = $(wildcard *.cpp)

SRC = data.cpp BayesRRm.cpp gadgets.cpp distributions_boost.cpp main.cpp options.cpp samplewriter.cpp compression.cpp

OBJ = $(SRC:.cpp = .o)

all:	$(EXE)

$(EXE):	$(OBJ)
		$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -o $(EXE) $(OBJ)

clean:
	/usr/bin/rm -rvf	\
	*.o

