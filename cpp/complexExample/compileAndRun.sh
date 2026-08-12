#!/bin/bash
# compile and run ComplexExample.cpp three ways, plotting each result
set -e # stop at the first failure


CXX=g++
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nvc++ # GPU only: g++ cannot build it
GPUARCH=cc120 # RTX 5060 Ti. `nvaccelinfo | grep Target` if this moves

# what keeps the three outputs bit for bit identical: no FMA fusion, and no
# non-IEEE division/sqrt (-Kieee, enough on its own to break it if missed)
CPUREPRO="-ffp-contract=off"
GPUREPRO="-Kieee -Mnofma"

INC=-I.. # physics.h, geometry.h and types.h live a level up

# 1. serial
$CXX -Wall -O2 $CPUREPRO $INC -o box.out ComplexExample.cpp
time ./box.out > box_serial.dat
python3 ../../plotBox.py box_serial.dat --out box_serial.png

# 2. OpenMP across the CPU cores
# -foffload=disable: ComplexExample.cpp has an "omp target" region g++ cannot build device code for
$CXX -Wall -O2 $CPUREPRO $INC -fopenmp -foffload=disable -o box_omp.out ComplexExample.cpp
time ./box_omp.out > box_omp.dat
python3 ../../plotBox.py box_omp.dat --out box_omp.png 

# 3. OpenMP offloaded to the GPU
# MANDATORY: fail loudly rather than fall back to the host and look like a slow GPU
$NV -O2 $GPUREPRO -mp=gpu -gpu=$GPUARCH $INC -o box_gpu.out ComplexExample.cpp
time OMP_TARGET_OFFLOAD=MANDATORY ./box_gpu.out > box_gpu.dat
python3 ../../plotBox.py box_gpu.dat --out box_gpu.png


# bitwise check
echo
cmp -s box_serial.dat box_omp.dat \
    && echo "serial == omp : bitwise identical" \
    || echo "serial != omp : REGRESSION, the parallel loop is not independent"
cmp -s box_serial.dat box_gpu.dat \
    && echo "serial == gpu : bitwise identical" \
    || echo "serial != gpu : REGRESSION, the device arithmetic has drifted"

sha1sum *.dat