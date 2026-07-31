#!/bin/bash
# compile and run Box.cpp three ways, plotting each result
set -e # stop at the first failure

# 1. serial
g++ -Wall -O2 -o box.out Box.cpp
time ./box.out > box_serial.dat
python3 plotBox.py box_serial.dat --out box_serial.png

# 2. OpenMP on the CPU cores
# (LD_LIBRARY_PATH is cleared because the HPC SDK puts an old libgomp first;
#  -foffload=disable keeps this build on the CPU even with offload installed)
g++ -Wall -O2 -fopenmp -foffload=disable -o box_omp.out Box.cpp
time env LD_LIBRARY_PATH= ./box_omp.out > box_omp.dat
python3 plotBox.py box_omp.dat --out box_omp.png

# 3. OpenMP offloaded to the GPU
# (MANDATORY makes the run fail loudly instead of silently falling back to CPU)
nvc++ -O2 -mp=gpu -gpu=cc89 -o box_gpu.out Box.cpp
time OMP_TARGET_OFFLOAD=MANDATORY ./box_gpu.out > box_gpu.dat
python3 plotBox.py box_gpu.dat --out box_gpu.png
