#
# Purpose: Makefile for compile the MPI based asynchronous Gibbs sampler
# Author : E. Orliac
# Date   : 22.01.2019
#

EXE      ?= hydra

CXX       = mpiicpc

CXXFLAGS  = -Ofast
#CXXFLAGS += -g
CXXFLAGS += -std=c++17
CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high
#CXXFLAGS += -xCORE-AVX2
CXXFLAGS += -D USE_MPI
CXXFLAGS += -qopenmp

INCLUDE  = -I$(EIGEN_ROOT)/include/eigen3
INCLUDE += -I$(BOOST_ROOT)/include

LIBS     = -lz 

#EO: remove multi-trait version for now
SOURCES = main.cpp data.cpp BayesRRm.cpp BayesW_arms.cpp BayesW.cpp gadgets.cpp distributions_boost.cpp options.cpp samplewriter.cpp compression.cpp


OBJECTS := $(patsubst %.cpp,%.o,$(SOURCES))

DEPENDS := $(patsubst %.cpp,%.d,$(SOURCES))

WARNING := -Wall -Wextra

RM := /usr/bin/rm

.PHONY: all clean

all: $(EXE)

clean:
	$(RM) $(OBJECTS) $(DEPENDS) $(EXE)

$(EXE): $(OBJECTS)
	$(CXX) $(WARNING) $(CXXFLAGS) $(INCLUDE) $(LIBS) $^ -o $@

-include $(DEPENDS)

%.o: %.cpp Makefile
	$(CXX) $(WARNING) $(CXXFLAGS) $(INCLUDE) $(LIBS) -MMD -MP -c $< -o $@
