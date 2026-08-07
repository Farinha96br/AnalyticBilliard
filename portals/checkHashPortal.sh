#!/bin/bash
# compile and run every portal case three ways, and check the three agree
set -e


CXX=g++
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nvc++ # GPU only: g++ cannot build it
GPUARCH=cc120 # RTX 5060 Ti. `nvaccelinfo | grep Target` if this moves

# what keeps the three outputs bit for bit identical: no FMA fusion, and no
# non-IEEE division/sqrt (-Kieee, easy to miss and enough on its own to break it)
CPUREPRO="-ffp-contract=off"
GPUREPRO="-Kieee -Mnofma"

INC=-I.. # physics.h, geometry.h and types.h live a level up

# case:bounces -- a window as long as the run makes the two panels one picture
CASES="PortalPlain:12 PortalFlipTangent:12 PortalFlipNormal:12
       PortalFlipBoth:12 PortalReversedEnds:12 PortalSegment:8"

fail=0

for entry in $CASES; do
    src=${entry%%:*}
    bounces=${entry##*:}
    name=$(echo "$src" | tr 'A-Z' 'a-z')

    echo
    echo "=== $src ==="

    # 1. serial. the only one whose stderr is kept: the other two say the same
    $CXX -Wall -O2 $CPUREPRO $INC -o ${name}_serial.out $src.cpp
    ./${name}_serial.out > ${name}_serial.dat

    # 2. OpenMP across the CPU cores
    # -foffload=disable: g++ cannot generate device code for the target region
    $CXX -Wall -O2 $CPUREPRO -fopenmp -foffload=disable $INC -o ${name}_omp.out $src.cpp
    ./${name}_omp.out > ${name}_omp.dat 2>/dev/null

    # 3. OpenMP offloaded to the GPU.
    #
    # this lane is worth nothing unless the work reaches the card, and it can
    # fail to in two ways. MANDATORY only covers the second:
    #
    #   - no device kernel is built at all. drop the pragma and nvc++ still
    #     produces a fine host binary, with no target region for MANDATORY to
    #     enforce: the run passes, the hashes match trivially, and the GPU was
    #     never involved. -Minfo=mp names the kernel it makes out of main.
    #   - a kernel is built but the runtime falls back to the host, which is
    #     what MANDATORY aborts on.
    #
    # NVCOMPILER_ACC_NOTIFY then has the runtime say where it actually went
    if ! $NV -O2 $GPUREPRO -mp=gpu -gpu=$GPUARCH -Minfo=mp $INC \
         -o ${name}_gpu.out $src.cpp 2> ${name}_gpu.build; then
        cat ${name}_gpu.build
        exit 1
    fi

    kernel=$(grep -o 'nvkernel_main[A-Za-z0-9_]*' ${name}_gpu.build | head -1)
    if [ -n "$kernel" ]; then
        echo "  gpu build : device kernel $kernel"
    else
        echo "  gpu build : NO DEVICE KERNEL -- main's target region did not compile"
        echo "              for the GPU, so this lane would test only host code"
        fail=1
    fi

    # MANDATORY aborts when it cannot offload, which under set -e would take the
    # script down before it could say why
    if ! NVCOMPILER_ACC_NOTIFY=1 OMP_TARGET_OFFLOAD=MANDATORY ./${name}_gpu.out \
           > ${name}_gpu.dat 2> ${name}_gpu.log; then
        echo "  gpu run   : ABORTED -- no usable device, and MANDATORY refused to"
        echo "              fall back to the host"
        head -3 ${name}_gpu.log | sed 's/^/              /'
        fail=1
        continue
    fi

    launch=$(grep -m1 'launch CUDA kernel.*function=main' ${name}_gpu.log || true)
    if [ -n "$launch" ]; then
        echo "  gpu run   : launched on $(echo "$launch" | grep -o 'device=[0-9]*'),"\
             "$(echo "$launch" | grep -o 'block=<<<[0-9]*' | grep -o '[0-9]*') threads"
    else
        echo "  gpu run   : NEVER REACHED THE DEVICE"
        fail=1
    fi

done


# three builds that agree have one distinct hash between them, so a case that
# prints more than a single line has a build out of step
echo
echo "sha1 over serial, omp and gpu:"
for entry in $CASES; do
    name=$(echo "${entry%%:*}" | tr 'A-Z' 'a-z')

    hashes=$(sha1sum ${name}_serial.dat ${name}_omp.dat ${name}_gpu.dat |
             awk '{print $1}' | sort -u)

    if [ "$(echo "$hashes" | wc -l)" -eq 1 ]; then
        printf "  %-22s %s\n" "$name" "$hashes"
    else
        printf "  %-22s MISMATCH, %s distinct:\n" "$name" "$(echo "$hashes" | wc -l)"
        sha1sum ${name}_serial.dat ${name}_omp.dat ${name}_gpu.dat | sed 's/^/    /'
        fail=1
    fi
done

echo
if [ $fail -eq 0 ]; then
    echo "all six cases: serial == omp == gpu"
else
    echo "REGRESSION above"
fi

# has to come before the exit, or it never runs at all
rm -f *.out *.png *.dat *.build *.log

exit $fail


