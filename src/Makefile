
# -----------------------------------------------------------------
#   Makefile for testing repo
#
#   Supported platforms
#       Unix / Linux                UNIX
#       MacOS                       MAC
# ---------------------------------------------------------------------

# Set this variable to either LINUX, MAC or WIN
OUTPUT = brr

# Put C++ compiler here
CXX_UNIX= mpicc #g++
#CXX_MAC = ""/usr/local/opt/llvm/bin/clang"   #g++

# CXXFLAGS = -I../ -I../../ -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/   -ldl -idirafter /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/
CXXFLAGS  = -I/software/include/Eigen -I/software/include/boost -DUNIX -m64 -ldl -O3
CXXFLAGS += -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./
# CXXFLAGS += -I../ -I../../  -I/usr/local/include -I/usr/local/include/Eigen -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DUNIX -m64 -ldl
#CXXFLAGS += -I/software/include/mpich-x86_64
CXXFLAGS += -fopenmp
#CXXFLAGS += -mavx512f -mavx512cd -fopt-info-vec-all
#CXXFLAGS += -g
#CXXFLAGS += -Wall
CXXFLAGS += -march=native
#CXXFLAGS += -S

# LIB += -lz -lmpi -lm  -lc++
LIB     += -lz -lmpi -lm  

#LDFLAGS = -L/usr/local/opt/llvm/lib
LDFLAGS= -Wl,--rpath=/software/lib64/  -Wl,--dynamic-linker=/software/lib64/ld-linux-x86-64.so.2 -L/lib/
#LDFLAGS+= -L/software/lib64/openmpi/lib/
#LDFLAGS+= -L/software/lib64/

CXX = $(CXX_UNIX)


HDR =  gctb.hpp \
      gadgets.hpp \
      options.hpp \
      mympi.hpp \
      BayesRRtoy.hpp \
      BayesRMmapToy.hpp \
      BayesRRm.h \
      distributions_boost.hpp \
      data.hpp 
	#  BayesRRpp.h
 
SRC = gctb.cpp \
      gadgets.cpp \
      main.cpp \
      options.cpp \
      mympi.cpp \
      BayesRRm.cpp \
      distributions_boost.cpp \
      data.cpp 
	 # BayesRRpp.cpp

OBJ = $(SRC:.cpp=.o)

all : $(OUTPUT)

$(OUTPUT) :
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LDFLAGS) $(LIB) 

$(OBJ) : $(HDR)

.cpp.o :
	$(CXX) $(CXXFLAGS) -c $*.cpp

.SUFFIXES : .cpp .c .o $(SUFFIXES)

$(OUTPUT) : $(OBJ)

FORCE:

clean:
	rm -f *.o *~
