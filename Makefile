.PHONY: all clean help info

SOURCEDIR = ./src
BINDIR    = ./bin

SOURCES  := $(wildcard $(SOURCEDIR)/*.cpp)

SRC_EXCL  =  $(SOURCEDIR)/BayesRRm_mt.cpp
SRC_EXCL +=  $(SOURCEDIR)/mk_lut.cpp

SOURCES  := $(filter-out $(SRC_EXCL),$(SOURCES))

CXXFLAGS  = -Ofast
#CXXFLAGS = -O1
CXXFLAGS += -g
CXXFLAGS += -std=c++17

INCLUDE   = -I$(SOURCEDIR)
#EO: with locally installed eigen (for gcc 8.3.0 @ SCITAS)
#    EIGEN_ROOT must be exported manually (see compile_with_gcc.sh)
INCLUDE  += -I$(EIGEN_ROOT) #/include/eigen3
#    otherwise loaded with Spack
INCLUDE  += -I$(EIGEN_ROOT)/include/eigen3
INCLUDE  += -I$(BOOST_ROOT)/include

ifeq ($(CXX),g++)

EXEC     ?= hydra_g
CXX       = mpic++
BUILDDIR  = build_gcc
CXXFLAGS += -fopenmp
CXXFLAGS += -march=skylake-avx512 -mprefer-vector-width=512
#CXXFLAGS += -march=native
#CXXFLAGS += -mavx2
#CXXFLAGS += -fopt-info-vec-missed=gcc_vec_missed.txt
CXXFLAGS += -fopt-info-vec=gcc_vec.txt


else ifeq ($(CXX),icpc)

EXEC     ?= hydra_i
CXX       = mpiicpc
BUILDDIR  = build_intel
CXXFLAGS += -qopenmp
CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high
#CXXFLAGS += -xCORE-AVX2, -axCORE-AVX512 -qopt-zmm-usage=high
CXXFLAGS += -qopt-report=2

else
	@echo "Neither GCC nor Intel compiler available." 1>&2 && false

endif

ifeq (, $(shell which $(CXX)))
$(error "no $(CXX) in $(PATH), please load relevant modules.")
endif

OBJS	:= $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))
DEPS	:= $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDIR)/%.d, $(SOURCES))

LIBS      = -lz

all: create_path $(BINDIR)/$(EXEC)

$(BINDIR)/$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp Makefile
	$(CXX) $(CXXFLAGS) $(INCLUDE) -MMD -MP -c $< -o $@

-include $(DEPS)

create_path:
	mkdir -p $(BUILDDIR)
	mkdir -p $(BINDIR)

clean:
	rm -vf $(BUILDDIR)/*.o $(BUILDDIR)/*.d $(BINDIR)/$(EXEC)

help:
	@echo "Usage: make [ all | clean | help ]"
