#BayesRRcmd

Software for performing Bayesian penalized regression for complex trait analysis.

For the moment the software is compatible with Linux OS and we are working into making it compatible with Mac OS.

## Quick start

### Install prerequisites
The software has some pre-requisites to be installed.   

eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)
boost (https://www.boost.org/)
cmake (https://cmake.org/)
ninja (https://ninja-build.org/manual.html)

###Clone or download

clone
```
git clone https://github.com/ctggroup/BayesRRcmd.git
```

download

```wget https://github.com/ctggroup/BayesRRcmd/archive/master.zip
unzip master.zip
```

###Compile

You can compile by using CMake & ninja and the following commands:

```
cd BayesRRcmd
cmake -G "CodeBlocks - Ninja" -DCMAKE_BUILD_TYPE=Release ../BayesRRcmd
ninja

```

###Test run

You can do a test run on a small provided dataset as follows:

```
cd BayesRRcmd/src
dataset=uk10k_chr1_1mb

./brr --bayes bayesMmap --bfile ../test/data/$dataset --pheno ../test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001

```

You should get messages in standard output for the reading of the dataset files, the running of Gibbs sampling, time taken for each iteration and finishing with "Analysis finished" message.
