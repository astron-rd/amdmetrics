#!/bin/bash
# amd_metrics build + test. Run on a machine with an AMD GPU (Radeon AI PRO
# R9700 / gfx1201 or Radeon Pro W7700 / gfx1101 verified) and a ROCm install
# with rocprofiler-sdk. ROCM_PATH may be overridden; the default below is
# /opt/rocm-7.14.0, where ROCm is installed on the machines this was developed
# on.
set -u
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.14.0}
export PATH=$ROCM_PATH/bin:$PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$SCRIPT_DIR/build/lib:${LD_LIBRARY_PATH:-}

echo "=== ENV ==="
hipcc --version 2>/dev/null | head -1

# Locate pybind11's CMake config: prefer the Python package, fall back to
# whatever find_package() picks up from CMAKE_PREFIX_PATH.
PYB="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)"
echo "pybind11_cmake_dir=${PYB:-<find_package default>}"

echo "=== CONFIGURE ==="
mkdir -p build
CMAKE_ARGS=(-S . -B build -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DROCM_PATH="$ROCM_PATH" -Wno-dev)
[ -n "$PYB" ] && CMAKE_ARGS+=(-Dpybind11_DIR="$PYB")
cmake "${CMAKE_ARGS[@]}" > build/cmake.log 2>&1
echo "configure exit: $?"
echo "=== BUILD ==="
cmake --build build -j8 > build/build.log 2>&1
echo "build exit: $?"
ls -1 build/lib/libamd_metrics.so build/demo/demo build/python/amdametrics*.so 2>&1

echo "=== RUN C++ DEMO ==="
./build/demo/demo
echo "demo exit: $?"

echo "=== PYTHON DEMO ==="
# compile the kernel helper for every GPU arch you want to measure
hipcc -O2 -fPIC -shared --offload-arch=gfx1101 --offload-arch=gfx1201 python/kernel_helper.cpp -o python/libkernels.so
PYTHONPATH=$SCRIPT_DIR/build/python LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$SCRIPT_DIR/build/lib python3 python/demo.py
echo "python exit: $?"
