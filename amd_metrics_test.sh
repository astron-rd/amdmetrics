#!/bin/bash
# amd_metrics build + test on a gpu* node (Radeon AI PRO R9700 / gfx1201).
# ROCm 7.14.0 lives in /opt/rocm-7.14.0 and has NO environment module, so PATH /
# LD_LIBRARY_PATH are set explicitly here. Run inside an srun allocation:
#   srun -p rbq -w gpu03 --gres=gpu:R9700:1 --time=01:30:00 -c8 -n1 bash -lc '~/amd_metrics_test.sh'
set -u
export ROCM_PATH=/opt/rocm-7.14.0
export PATH=$ROCM_PATH/bin:$PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$SCRIPT_DIR/build/lib:${LD_LIBRARY_PATH:-}

echo "=== ENV ==="
hipcc --version 2>/dev/null | head -1
PYB=$(find /var/software/spack -name pybind11Config.cmake -path '*py-pybind11*' 2>/dev/null | head -1)
echo "pybind11_cmake_dir=$(dirname "$PYB")"

echo "=== CONFIGURE ==="
cmake -S . -B build -DCMAKE_PREFIX_PATH=$ROCM_PATH -Dpybind11_DIR="$(dirname "$PYB")" -DROCM_PATH=$ROCM_PATH -Wno-dev > build/cmake.log 2>&1
echo "configure exit: $?"
echo "=== BUILD ==="
cmake --build build -j8 > build/build.log 2>&1
echo "build exit: $?"
ls -1 build/lib/libamd_metrics.so build/demo/demo build/python/amdametrics*.so 2>&1

echo "=== RUN C++ DEMO ==="
./build/demo/demo
echo "demo exit: $?"

echo "=== PYTHON DEMO ==="
hipcc -O2 -fPIC -shared --offload-arch=gfx1101 --offload-arch=gfx1201 python/kernel_helper.cpp -o python/libkernels.so
PYTHONPATH=$SCRIPT_DIR/build/python LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$SCRIPT_DIR/build/lib python3 python/demo.py
echo "python exit: $?"
