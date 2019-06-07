#
# Purpose: Makefile for compile the MPI based asynchronous Gibbs sampler
# Author : E. Orliac
# Date   : 22.01.2019
#

EXE       = mpi_gibbs

CXX       = mpiicpc

CXXFLAGS  = -O3
#CXXFLAGS  = -O0
#CXXFLAGS += -g
CXXFLAGS += -std=c++17
CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high
#CXXFLAGS += -xCORE-AVX2
#CXXFLAGS += -xCORE-AVX2, -axCORE-AVX512 -qopt-zmm-usage=high
CXXFLAGS += -D USE_MPI  
#CXXFLAGS += -D EIGEN_DONT_VECTORIZE

#EO: set this back when switching to production
#    This will allow to fully use vectorization on both fidis and gacrux
#CXXFLAGS += -xCORE-AVX2, -axCORE-AVX512
##CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high

#EO: enable the following for a strict comparison with Eigen
#CXXFLAGS += -fp-model precise 
#CXXFLAGS += -fp-model source

INCLUDE  = -I$(EIGEN_ROOT)/include/eigen3
INCLUDE += -I$(BOOST_ROOT)/include
INCLUDE += -I../tbb/include

LIBS     = -lz #-lstdc++fs #-lboost_system -lboost_filesystem

SRC = data.cpp BayesRRm.cpp gadgets.cpp distributions_boost.cpp main.cpp options.cpp samplewriter.cpp compression.cpp

OBJ = $(SRC:.cpp = .o)

all:	$(EXE)

$(EXE):	$(OBJ)
		$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -o $(EXE) $(OBJ)

clean:
	/usr/bin/rm -rvf	\
	*.o
