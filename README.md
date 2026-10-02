# amd_metrics

Hardware-counter measurement for AMD GPUs, built on the ROCm **rocprofiler-sdk**
(the AMD analog of `nv_metrics`, which uses CUPTI's range profiler). Developed
and verified on the Radeon AI PRO **R9700** (gfx1201 / RDNA4) and **Radeon Pro
W7700** (gfx1101) with ROCm **7.14.0**.

The library is an in-process rocprofiler tool (`rocprofiler_configure` embedded
in `libamd_metrics.so`), self-initialising on first use. It collects
per-dispatch counter records through the SDK's **buffered dispatch-counting
service** and sums each requested counter over every GPU dispatch that lands
inside a measurement window. No `LD_PRELOAD`, no `rocprofv3` wrapper. It works
both when linked into an executable and when loaded mid-process (Python import /
dlopen).

## Layout

```
lib/        amd_metrics.h (public API) + amd_metrics.cpp (rocprofiler tool)
python/     pybind11 module "amdametrics" + demo.py + kernel_helper.cpp
            (ctypes-launchable HIP kernel, the pycuda analog) — see python/README.md
demo/       demo.cpp — HIP kernel, query + measure end-to-end
amd_metrics_test.sh   configure + build + run C++ demo + run Python demo
```

## Public API (`lib/amd_metrics.h`)

```cpp
namespace amd_metrics {
    struct CounterInfo { std::string name, description; bool is_derived; };
    // list the GPU's supported HW counters (143 on gfx1201)
    int query_counters(std::vector<CounterInfo>& out);
    // one-shot: measure `names` for kernels launched in this window
    int measure(const std::vector<std::string>& names, std::vector<double>& vals);
    // split window control (iterative, e.g. around kernels launched from Python)
    int measure_start(const std::vector<std::string>& names,
                      std::vector<std::string>* missing = nullptr);
    int measure_end(std::vector<double>& vals);   // one value per requested name
} // namespace amd_metrics
```

Python (see `python/README.md`):
`import amdametrics; amdametrics.query_counters()`,
`amdametrics.measure(["Wavefronts","SQ_WAVES","GPUBusy","GRBM_COUNT"])`, or
`measure_start(names)` … `measure_end()`.

Requirements: CMake >= 3.20, a HIP-capable ROCm install (rocprofiler-sdk), and
`pybind11` for the Python module. On our cluster ROCm lives in
`/opt/rocm-7.14.0` with no environment module (PATH/LD_LIBRARY_PATH are set
explicitly in `amd_metrics_test.sh`), and `pybind11` comes from the Spack module
`py-pybind11`. Any stock ROCm >= 7.x with `find_package` support should work
similarly; adjust `ROCM_PATH` in the script.

## Build & run (on a gpu\\\* node, inside an srun allocation)

```bash
srun -p rbq -w gpu03 --gres=gpu:R9700:1 --time=01:30:00 -c8 -n1 \
    bash -lc "$HOME/src/amd_metrics/amd_metrics_test.sh"
```

That configures (CMake, `pybind11` from the Spack `py-pybind11` module, HIP +
rocprofiler-sdk from `$ROCM_PATH`), builds, runs the C++ demo and the Python
demo (script is location-independent; the `srun` line above is cluster-specific
— anywhere else just run `amd_metrics_test.sh` on a node with a GPU, or
configure/build manually with CMake). Example output:

```
device:        AMD Radeon AI PRO R9700 (32 CUs, wavefront size 32)
143 supported counters (84 hardware, 59 agent-property constants)
  CU_NUM ...

mix kernel: n=1048576, grid=4096, block=256 (wall 0.435 ms)
  Wavefronts                      33280   expected ~ n / 32 = 32768 (plus ramp-up waves)
  SQ_WAVES                        33280   waves seen by the sequencers
  SQ_BUSY_CYCLES            2.47144e+06   cycles with active waves in the SQs
  GRBM_COUNT                      95781   cycles while the GPU was measured
  GPUBusy                           300   100*GUI_ACTIVE/GRBM_COUNT, per counted dispatch (>100 = several counted dispatches)
```

## Counter reality on gfx1201 (proven, important)

Through the SDK dispatch-counting service this GPU only returns real values for
the **wavefront / activity family**: `Wavefronts` (= launched_threads/32,
exactly 32768 for a 1M-thread kernel), `SQ_WAVES`, `SQ_BUSY_CYCLES`, `GPUBusy`,
`GRBM_COUNT`. Per-opcode instruction counters (`SQ_INSTS_VALU/LDS/SALU`,
`VALUInsts`, `SALUInsts`), occupancy and L2 (`GL2C_*`) read **0** even with 50
dispatches on RDNA4/gfx1201 — a hardware+SDK reality, not a bug, so the demos
use the working set and `rocprofv3 --list-avail` shows the full counter
inventory for your GPU.

Note: when a measurement window contains more than one kernel dispatch
(runtime-internal dispatches like `hipMalloc` fills count too), per-dispatch
percentages like `GPUBusy` are summed, so values above 100 mean several
dispatches were counted, not an error.
