
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
CXX_UNIX = g++
#CXX_MAC = gcc   #g++

#CXXFLAGS += -I../ -I../../ -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/   -ldl -idirafter /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/
CXXFLAGS  = -I/software/include/Eigen -I/software/include/boost -DUNIX -m64 -ldl -O3
CXXFLAGS = -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./ -I../tbb/include -I/home/thanasis/anaconda3/include
# CXXFLAGS += -I../ -I../../  -I/usr/local/include -I/usr/local/include/Eigen -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DUNIX -m64 -ldl

#CXXFLAGS += -fopenmp
#CXXFLAGS += -mavx512f -mavx512cd -fopt-info-vec-all
#CXXFLAGS += -g
#CXXFLAGS += -Wall
CXXFLAGS += -march=native
#CXXFLAGS += -S

# LIB += -lz -lmpi -lm  -lc++
LIB     += -lz -lm 

LDFLAGS = -L/usr/local/opt/llvm/lib

CXX = $(CXX_UNIX)


HDR = BayesRRm.h \
    BayesRRmz.hpp \
    data.hpp \
    distributions_boost.hpp \
    gadgets.hpp \
    options.hpp \
    samplewriter.h \
    compression.h \
    limitsequencegraph.hpp \
    analysisgraph.hpp 
 
SRC = BayesRRm.cpp \
    BayesRRmz.cpp \
    data.cpp \
    distributions_boost.cpp \
    gadgets.cpp \
    options.cpp \
    samplewriter.cpp \
    compression.cpp \
    limitsequencegraph.cpp \
    analysisgraph.cpp 

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
