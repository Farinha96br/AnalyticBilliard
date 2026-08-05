#!/bin/bash
# long-run checks. kept out of compileAndRun.sh because this takes minutes.
set -e

NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nvc++
GPUARCH=cc120
REPRO="-Mnofma"

# 1. 1e8 bounces per particle, energy / containment / stalls / recorded time.
# one thread per particle: they are independent, so this only changes which core
# does the work, never the arithmetic
$NV -O2 $REPRO -mp=multicore -o testStability.out testStability.cpp
time ./testStability.out

# 2. the bitwise agreement, at a length worth calling long. the output is hashed
# as it streams rather than landing on disk: same comparison, no GB of temporaries.
#
# Niter is deliberately NOT pushed to 1e7 here. Box.cpp runs five particles, and
# five threads leave a GPU built for tens of thousands almost entirely idle -- the
# offload build is ~190x slower than the CPU one at this width, so 1e7 turns a
# two minute check into the better part of an hour. it is the same arithmetic
# either way, so length buys nothing the CPU legs do not already give
echo
echo "=== bitwise agreement at Niter = 1e6 ==="
sed 's/int Niter = 1e6;/int Niter = 1000000;/' Box.cpp > BoxLong.cpp
$NV -O2 $REPRO -o box_long_ser.out BoxLong.cpp
$NV -O2 $REPRO -mp=multicore -o box_long_omp.out BoxLong.cpp
$NV -O2 $REPRO -mp=gpu -gpu=$GPUARCH -o box_long_gpu.out BoxLong.cpp

SER=$(./box_long_ser.out | md5sum | cut -d' ' -f1)
OMP=$(./box_long_omp.out | md5sum | cut -d' ' -f1)
GPU=$(OMP_TARGET_OFFLOAD=MANDATORY ./box_long_gpu.out | md5sum | cut -d' ' -f1)
echo "  serial $SER"
echo "  omp    $OMP"
echo "  gpu    $GPU"
[ "$SER" = "$OMP" ] && echo "serial == omp : bitwise identical" \
                    || echo "serial != omp : REGRESSION"
[ "$SER" = "$GPU" ] && echo "serial == gpu : bitwise identical" \
                    || echo "serial != gpu : REGRESSION"

rm -f BoxLong.cpp box_long_*.out
