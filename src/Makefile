
# -----------------------------------------------------------------
#   Makefile for gctb 
#   
#   Supported platforms
#       Unix / Linux                UNIX
#       MacOS                       MAC
# ---------------------------------------------------------------------

# Set this variable to either LINUX, MAC or WIN
SYS = MAC
OUTPUT = gctb

# Put C++ compiler here
CXX_UNIX = mpicxx   #g++
CXX_MAC = gcc   #g++

# Any other compiler flags here ( -Wall, -g, etc)
CXXFLAGS = -w -O3 -m64 -DEIGEN_NO_DEBUG -msse2 -std=c++11 -I./

 CXXFLAGS += -I../ -I../../  -I/usr/local/include -I/usr/local/include/Eigen -I/usr/local/include/boost -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -DUNIX -m64 -ldl
 LIB += -lz -lmpi -lm  -lc++
 LDFLAGS+= -L/usr/local/opt/llvm/lib
 CXX = $(CXX_MAC)


HDR = gctb.hpp \
      data.hpp \
      gadgets.hpp \
      options.hpp \
      mympi.hpp \
      BayesRRtoy.hpp

SRC = gctb.cpp \
      data.cpp \
      gadgets.cpp \
      main.cpp \
      options.cpp \
      mympi.cpp \
      BayesRRtoy.cpp

	   
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
