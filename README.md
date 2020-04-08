# hydra

-- BELOW IS OLD STUFF

Software for performing Bayesian penalized regression for complex trait analysis.

For the moment the software is compatible with Linux OS and we are working into making it compatible with Mac OS.

In the README you will find installation instructions, go to the wiki (https://github.com/ctggroup/BayesRRcmd/wiki) for more information on the algorithms, analysis types available and options. 

## Quick start

Follow instructions to deploy software.

### 1. Install prerequisites
The software has some pre-requisites to be installed.   

eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)   
boost (https://www.boost.org/)   
ZLIB  (https://www.zlib.net/)	
cmake (https://cmake.org/)   
ninja (https://ninja-build.org/manual.html)   

these can be easily installed in Linux:

```
sudo apt-get install libeigen3-dev libboost-all-dev zlib1g-dev cmake ninja-build 

```

Additionally you will need to download Threading Building Blocks (TBB) software (see below in section 2.)

### 2. Clone or download

Clone

```
git clone https://github.com/ctggroup/BayesRRcmd.git
```

or Download

```
wget https://github.com/ctggroup/BayesRRcmd/archive/master.zip
unzip master.zip
mv BayesRRcmd-master BayesRRcmd
```

At the moment we are using Threading Building Blocks software which should be installed as follows:

if you cloned BayesRRcmd:
```
cd BayesRRcmd
git submodule init
git submodule update
```

if you downloaded BayesRRcmd:

```
cd BayesRRcmd
wget https://github.com/01org/tbb/archive/tbb_2019.zip
unzip tbb_2019.zip
mv tbb-tbb_2019/* tbb
rm -r tbb-tbb_2019 tbb_2019.zip
```

### 3. Compile

You can compile by using CMake & ninja within the BayesRRcmd directory awith the following commands:

```
cmake -G "CodeBlocks - Ninja" -DCMAKE_BUILD_TYPE=Release ../BayesRRcmd
ninja

```
You should obtain the executable brr in src folder.

### 4. Test run

You can do a test run within the BayesRRcmd directory on a small provided dataset as follows:

```
dataset=uk10k_chr1_1mb

src/brr --bayes bayesMmap --bfile test/data/$dataset --pheno test/data/test.phen --chain-length 10 --burn-in 0 --thin 1 --mcmc-samples ./bayesOutput.csv --S 0.01,0.001,0.0001

```

You should get messages in standard output for the reading of the dataset files, the running of Gibbs sampling, time taken for each iteration and finishing with "Analysis finished" message and the time taken to run.


## MPI GIBBS (experimental)