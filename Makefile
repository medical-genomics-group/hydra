.PHONY: all clean help info

SOURCEDIR = ./src
BINDIR    = ./bin

SOURCES  := $(wildcard $(SOURCEDIR)/*.cpp)

SRC_EXCL  =  $(SOURCEDIR)/BayesRRm_mt.cpp
SRC_EXCL +=  $(SOURCEDIR)/mk_lut.cpp

SOURCES  := $(filter-out $(SRC_EXCL),$(SOURCES))

#CXXFLAGS  = -Ofast
CXXFLAGS  = -O1
CXXFLAGS += -g
CXXFLAGS += -std=c++17

INCLUDE   = -I$(SOURCEDIR)
INCLUDE  += -I$(EIGEN_ROOT)/include/eigen3
INCLUDE  += -I$(BOOST_ROOT)/include

ifeq ($(CXX),g++)

EXEC     ?= hydra_g
CXX       = mpic++
BUILDDIR  = build_gcc
CXXFLAGS += -fopenmp
CXXFLAGS += -march=native

else ifeq ($(CXX),icpc)

EXEC     ?= hydra_i
CXX       = mpiicpc
BUILDDIR  = build_intel
CXXFLAGS += -qopenmp
CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high
#CXXFLAGS += -xCORE-AVX2, -axCORE-AVX512 -qopt-zmm-usage=high

else
	@echo "Neither GCC nor Intel compiler available." 1>&2 && false

endif


ifeq (, $(shell which $(CXX)))
$(error "no $(CXX) in $(PATH), please load relevant modules.")
endif


OBJ      := $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

LIBS      = -lz


all: dir $(BINDIR)/$(EXEC)

$(BINDIR)/$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@

$(OBJ): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

dir:
	mkdir -p $(BUILDDIR)
	mkdir -p $(BINDIR)

clean:
	rm -vf $(BUILDDIR)/*.o $(BINDIR)/$(EXEC)

help:
	@echo "Usage: make [ all | clean | help ]"
