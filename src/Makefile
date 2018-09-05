
# -----------------------------------------------------------------
#   Makefile for testing repo 
#   
#   Supported platforms
#       Unix / Linux                UNIX
#       MacOS                       MAC
# ---------------------------------------------------------------------

# Set this variable to either LINUX, MAC or WIN
OUTPUT = gctb

# Put C++ compiler here
CXX_UNIX = mpicxx   #g++
#CXX_MAC = gcc   #g++

# Any other compiler flags here ( -Wall, -g, etc)
#CXXFLAGS = -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./
#CXXFLAGS = -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./
# CXXFLAGS += -I../ -I../../  -I/usr/local/include -I/usr/local/include/Eigen -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DUNIX -m64 -ldl

CXXFLAGS  = -I/software/include/Eigen -I/software/include/boost -DUNIX -m64 -ldl -O3
CXXFLAGS += -fopenmp
#CXXFLAGS += -mavx512f -mavx512cd -fopt-info-vec-all
#CXXFLAGS += -g
CXXFLAGS += -march=native

# LIB += -lz -lmpi -lm  -lc++
LIB     += -lz -lmpi -lm

LDFLAGS += -L/usr/local/opt/llvm/lib

CXX = $(CXX_UNIX)


HDR = gctb.hpp \
      data.hpp \
      gadgets.hpp \
      options.hpp \
      mympi.hpp \
      BayesRRtoy.hpp \
      BayesRMmapToy.hpp

SRC = gctb.cpp \
      data.cpp \
      gadgets.cpp \
      main.cpp \
      options.cpp \
      mympi.cpp \
      BayesRRtoy.cpp \
      BayesRMmapToy.cpp


OBJ = $(SRC:.cpp=.o)

all : $(OUTPUT) 

$(OUTPUT) :
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LIB) 

$(OBJ) : $(HDR)

.cpp.o : 
	$(CXX) $(CXXFLAGS) -c $*.cpp

.SUFFIXES : .cpp .c .o $(SUFFIXES)

$(OUTPUT) : $(OBJ)

FORCE:

clean:
	rm -f *.o *~
