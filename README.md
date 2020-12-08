# hydra

Software for performing Bayesian penalized regression for complex trait analysis using hybrid-parallel algorithms. 

For the moment the software is is only available in architectures AVX2 and AVX512 and is compatible with gcc and intel compilers, we are working to make it compatible with clang. In the README you will find installation instructions, go to the wiki (https://github.com/medical-genomics-group/hydra/wiki) for more information on the algorithms, analysis types available and options. 


## Quick start

Follow instructions to deploy software.

### 1. Install prerequisites
The software has some pre-requisites to be installed.   

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


### 3. Compile

You can compile by simply using make :

```
cd hydra/src
make
```

You should obtain the executable hydra in src folder.
