# Code for a test example
Much of the code is borrowed from GCTB: a tool for Genome-wide Complex Trait Bayesian analysis


### How to compile
We have set dependencies on both MPI and OpenMP libraries as well as two C++ libraries i.e. Eigen and Boost. Please change the compiler and directories of these libraries in the Makefile.

### Aims
After compiling run ./test.sh

This will read in a data file that is in a specific format, select a column at random, and call Eigen to calculate the mean value of each column. The output will print that the data has been read in and then print the means to screen.

In the file data.cpp line 45 reads in the binary file that contains the data. We need Eigen to operate on a memory mapped version of this file, so that only one column is read into memory at any time, rather than the entire file.

