# hydra

Software for performing Bayesian penalized regression for complex trait analysis using hybrid-parallel algorithms. 

For the moment the software is is only available in architectures AVX2 and AVX512 and is compatible with gcc and intel compilers with respectively mvapich2 and intel mpi libraries, we are working to make it compatible with clang.

In the README you will find installation instructions, go to the wiki (https://github.com/medical-genomics-group/hydra/wiki) for more information on the algorithms, analysis types available and options. 


## Quick start

Follow instructions to deploy software.

### 1. Install prerequisites
In addition to the mvapich2/intel mpi libraries of the compilers, the software has two pre-requisites to be installed.   

eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)   
boost (https://www.boost.org/)

these can be easily installed in Linux:

```
sudo apt-get install libeigen3-dev libboost-all-dev 
```

### 2. Clone or download

Clone

```
git clone https://github.com/medical-genomics-group/hydra
```

or Download

```
wget https://github.com/medical-genomics-group/hydra/archive/master.zip
unzip master.zip
mv hydra-master hydra
```


### 3. Load modules

You need to load the modules boost, eigen, mpi and gcc/intel 

If you are using gcc compiler, for example, you need to type
```
module load gcc mvapich2 boost eigen
```

If you are using intel compiler, you need to type

```
module load intel intel-mpi boost eigen
```

However, given the specifications in your cluster, the modules might be called slightly differently.

### 4. Compile

You can compile using intel simply using make :

```
cd hydra/src
make
```

You can compile using gcc simply using make -f Makefile_G
```
cd hydra/src
make -f Makefile_G
```

You should obtain the executable `hydra` if used intel compiler or `hydra_G` if used gcc compiler in src folder.
