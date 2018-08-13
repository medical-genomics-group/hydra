# Code for a test example
Much of the code is borrowed from GCTB: a tool for Genome-wide Complex Trait Bayesian analysis


### How to compile
We have set dependencies on both MPI and OpenMP libraries as well as two C++ libraries i.e. Eigen and Boost. In /src/Makefile please change the compiler and directories of these libraries in the Makefile to suit your operating system.

### Aims
After compiling run /src/test.sh

This will read in a data file that is in a specific format, select a column at random, and call Eigen to calculate the mean value of each column. Output will be printed as the data are read and then the means will print to screen.

On line 45 the file /src/data.cpp the binary file that contains the data is read into memory. Instead of this, we would like Eigen to operate on a memory mapped version of this file, so that only one randomly selected column is read into memory at any time, rather than the entire file.

Our overall aim is to apply this to a datset that has 470,000 rows and 38 million columns, with each data entry consisting of 8 bytes. The computations we are doing require only one randomly selected column at a time and so we hope to reduce the RAM requirements as much as possible through memory mapping, though other suggestions are more than welcome also.

