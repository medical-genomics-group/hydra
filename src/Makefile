
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
CXX_UNIX = "/usr/local/opt/llvm/bin/clang"  #g++
#CXX_MAC = gcc   #g++

# Any other compiler flags here ( -Wall, -g, etc)
CXXFLAGS = -w -O3  -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./
#CXXFLAGS = -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./
<<<<<<< HEAD
 CXXFLAGS += -I../ -I../../ -I/usr/local/include/boost   -ldl -idirafter /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/
#CXXFLAGS  = -I/software/include/Eigen -I/software/include/boost -DUNIX -m64 -ldl -O3
=======
#CXXFLAGS = -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./
# CXXFLAGS += -I../ -I../../  -I/usr/local/include -I/usr/local/include/Eigen -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DUNIX -m64 -ldl

CXXFLAGS  = -I/usr/include/eigen3 -I/usr/include/boost -DUNIX -m64 -ldl -O3
>>>>>>> refs/heads/kdab
CXXFLAGS += -fopenmp
#CXXFLAGS += -mavx512f -mavx512cd -fopt-info-vec-all
#CXXFLAGS += -g
#CXXFLAGS += -Wall
CXXFLAGS += -march=native
#CXXFLAGS += -S

# LIB += -lz -lmpi -lm  -lc++
LIB     += -lz -lmpi -lm -lc++

LDFLAGS = -L/usr/local/opt/llvm/lib

CXX = $(CXX_UNIX)


HDR = data.hpp \
      gctb.hpp \
      gadgets.hpp \
      options.hpp \
      mympi.hpp \
      BayesRRtoy.hpp \
      BayesRMmapToy.hpp \
<<<<<<< HEAD
      BayesRRm.h \
      distributions_boost.hpp
=======
			BayesPPToy.hpp
>>>>>>> refs/heads/kdab

SRC = data.cpp \
      gctb.cpp \
      gadgets.cpp \
      main.cpp \
      options.cpp \
      mympi.cpp \
      BayesRRtoy.cpp \
      BayesRMmapToy.cpp \
<<<<<<< HEAD
      BayesRRm.cpp \
      distributions_boost.cpp
=======
			BayesPPToy.cpp

>>>>>>> refs/heads/kdab

OBJ = $(SRC:.cpp=.o)

all : $(OUTPUT)

$(OUTPUT) :
<<<<<<< HEAD
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LDFLAGS) $(LIB) 
=======
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LIB)
>>>>>>> refs/heads/kdab

$(OBJ) : $(HDR)

.cpp.o :
	$(CXX) $(CXXFLAGS) -c $*.cpp

.SUFFIXES : .cpp .c .o $(SUFFIXES)

$(OUTPUT) : $(OBJ)

FORCE:

clean:
	rm -f *.o *~
