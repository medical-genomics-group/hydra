# GCTB
a tool for Genome-wide Complex Trait Bayesian analysis


### How to compile GCTB 1.9
GCTB 1.9 has dependencies on both MPI and OpenMP libraries as well as two C++ libraries i.e. Eigen and Boost.

To compile GCTB 1.9 on a Linux system, follow below steps:

1. Download Eigen and Boost;
2. Edit their path in the enclosed Makefile (these two libraries themselves do not need to be compiled);
3. Load an appropriate implementation of MPI library (e.g. openMPI, intelMPI, MPICH2, etc) and OpenMP library (it is best to consult with the system administrator);
4. Switch "SYS" to "LINUX" in the Makefile;
5. Execute make command.


To compile GCTB 1.9 on a Mac system, follow below steps:

1. Download Eigen and Boost;
2. Download Xcode;
3. Open Terminal and run "xcode-select --install";
4. Install clang-omp using homebrew: "brew install clang-omp";
5. Install open-mpi using homebrew: "brew install open-mpi";
6. Switch "SYS" to "MAC" in the Makefile;
7. Execute make command.


We have also included a Xcode (v7.3.1) project file for Mac system. To compile in the Xcode, follow below steps:

1. Click on the project in Xcode and then click on Build Settings
2. Add “/usr/local/include” to “Search Paths – Header Search Paths”
3. Add the paths of Eigen and Boost libraries to “Search Paths – Header Search Paths”
4. Add”/usr/local/lib” to “Search Paths – Library Search Paths”
5. Add “-lmpi -lm” to “Linking – Other Linker Flags”
6. Add a new user-defined setting "CC" with the value "/usr/local/bin/clang-omp"
7. Add "-fopenmp" to "Other C Flags"
8. Set Enable Modules (C and Objective-C) to "No"
9. Add "/usr/local/include/libiomp" to “Search Paths – Header Search Paths”
10. Click on the Build Phases and add "/usr/local/lib/libiomp5.dylib" to Link Binary With Libraries
11. Add "/usr/local/lib/libmpi_cxx.1.dylib" to Link Binary With Libraries
12. Use Command+B to compile


To run in the Xcode, follow these steps:

1. Edit the “schemes” of the project: change the “Executable” to mpiexec, which is located at “/usr/local/bin”, an alias of “orterun”. 
   Note that “/usr/local/bin” is by default hidden in the Finder. To choose “orterun”, press “command + shift + G” then type in “/usr/local/bin” and press “Go”.
2. Click on “Arguments” in the “schemes” of the project, for running with two processors, add “-n 2” and “\$BUILT_PRODUCTS_DIR/$EXECUTABLE_PATH” to “Arguments Passed On Launch”.
