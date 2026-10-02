# Python bindings (`amdametrics`)

pybind11 module around `libamd_metrics` (see the top-level README for how the
rocprofiler tool works). Importing the module does **not** initialise the
profiler; the first API call self-initialises it, so it works whether the module
is imported before or after HIP is up.

## API

```python
import amdametrics

info = amdametrics.query_counters()   # [{"name", "description", "is_derived"}, ...]

vals = amdametrics.measure(["Wavefronts", "GPUBusy"])   # one-shot window

amdametrics.measure_start(["Wavefronts", "SQ_WAVES", "GPUBusy", "GRBM_COUNT"])
# ... launch kernels here (any path: torch, hip-python, ctypes, ...) ...
vals = amdametrics.measure_end()      # one value per requested name, summed
```

`measure_end()` returns floats in request order, summed over every kernel
dispatch that started inside the window (runtime-internal dispatches, e.g.
`hipMalloc` fills, count too — subtract or ignore them if you need pure kernel
numbers; they show up as e.g. GPUBusy=300 when three dispatches were counted).

## demo.py + kernel_helper.cpp

`demo.py` measures a HIP kernel launched through `ctypes` (the pycuda analog):

```bash
# on a gpu* node, after cmake --build (see top-level README)
hipcc -O2 -fPIC -shared --offload-arch=gfx1101 --offload-arch=gfx1201 \
    python/kernel_helper.cpp -o python/libkernels.so
PYTHONPATH=<build-dir>/python python3 python/demo.py
```

`kernel_helper.cpp` must be compiled for **every GPU arch you want to measure**
(`--offload-arch=...` per arch, see `amd_metrics_test.sh`). A single-arch build
silently launches nothing on the other arch, which looks exactly like
zero-valued counters.
