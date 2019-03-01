# BayesRRcmd

Software for performing Bayesian penalized regression for complex trait analysis.

For the moment the software is compatible with Linux OS and we are working into making it compatible with Mac OS.

## Quick start

Follow instructions to deploy software.

### 1. Install prerequisites
The software has some pre-requisites to be installed.   

eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)   
boost (https://www.boost.org/)   
cmake (https://cmake.org/)   
ninja (https://ninja-build.org/manual.html)   

these can be easily installed in Linux:

```
sudo apt-get install libeigen3-dev libboost-all-dev cmake ninja-build 

```


### 2. Clone or download

Clone

```
git clone https://github.com/ctggroup/BayesRRcmd.git
```

Download

```
wget https://github.com/ctggroup/BayesRRcmd/archive/master.zip
unzip master.zip
```

### 3. Compile

You can compile by using CMake & ninja and the following commands:

```
cd BayesRRcmd
cmake -G "CodeBlocks - Ninja" -DCMAKE_BUILD_TYPE=Release ../BayesRRcmd
ninja

```

### 4. Test run

You can do a test run on a small provided dataset as follows:

```
cd BayesRRcmd/src
dataset=uk10k_chr1_1mb

./brr --bayes bayesMmap --bfile ../test/data/$dataset --pheno ../test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001

```

You should get messages in standard output for the reading of the dataset files, the running of Gibbs sampling, time taken for each iteration and finishing with "Analysis finished" message.
