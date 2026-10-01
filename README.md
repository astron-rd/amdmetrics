# amd_metrics

Hardware-counter measurement for AMD GPUs, built on the ROCm **rocprofiler-sdk**
(the AMD analog of `nv_metrics`, which uses CUPTI's range profiler). Target: the
Radeon AI PRO **R9700** (gfx1201 / RDNA4) on the `gpu*` nodes, ROCm **7.14.0**
at `/opt/rocm-7.14.0` (no environment module — PATH/LD_LIBRARY_PATH are set
explicitly, see `amd_metrics_test.sh`).

The library is an in-process rocprofiler tool (`rocprofiler_configure` embedded
in `libamd_metrics.so`), self-initialising on first use. It collects
per-dispatch counter records through the SDK's **buffered dispatch-counting
service** and sums each requested counter over every GPU dispatch that lands
inside a measurement window. No `LD_PRELOAD`, no `rocprofv3` wrapper.

## Layout

```
lib/        amd_metrics.h (public API) + amd_metrics.cpp (rocprofiler tool)
python/     pyamdmetrics.cpp (pybind11 module "amdametrics") + demo.py +
            kernel_helper.cpp (ctypes-launchable HIP kernel, the pycuda analog)
demo/       demo.cpp — HIP fadd/fmul/fma kernel, query + measure end-to-end
spike/      standalone spike_tool + main used to prove the pipeline (RP_WANT/NK)
amd_metrics_test.sh   configure + build + run C++ demo + run Python demo
```

## Public API (`lib/amd_metrics.h`)

```cpp
namespace amd_metrics {
    struct CounterInfo { std::string name, description; bool is_derived; };
    // queryMetrics analog: list the GPU's supported HW counters (143 on gfx1201)
    int query_counters(std::vector<CounterInfo>& out);
    // one-shot: measure `names` for kernels launched in this window
    int measure(const std::vector<std::string>& names, std::vector<double>& vals);
    // split window control (iterative / from Python around ctypes-launched kernels)
    int measure_start(const std::vector<std::string>& names,
                      std::vector<std::string>* missing = nullptr);
    int measure_end(std::vector<double>& vals);   // one value per requested name
} // namespace amd_metrics
```

Python: `import amdametrics; amdametrics.query_counters()`,
`amdametrics.measure(["Wavefronts","SQ_WAVES","GPUBusy","GRBM_COUNT"])`, or
`measure_start(names)` … `measure_end()`.

## Build & run (on a gpu\* node, inside an srun allocation)

```bash
srun -p rbq -w gpu03 --gres=gpu:R9700:1 --time=01:30:00 -c8 -n1 \
    bash -lc '~/src/amd_metrics/amd_metrics_test.sh'
```

That configures (CMake, `pybind11` from the Spack `py-pybind11` module, HIP +
rocprofiler-sdk from `$ROCM_PATH`), builds, runs the C++ demo and the Python
demo. Example output:

```
device:        AMD Radeon AI PRO R9700 (32 CUs, wavefront size 32)
143 supported counters (84 hardware, 59 agent-property constants)
  CU_NUM ...

mix kernel: n=1048576, grid=4096, block=256 (wall 0.435 ms)
  Wavefronts                      33280   expected ~ n / 32 = 32768 (plus ramp-up waves)
  SQ_WAVES                        33280   waves seen by the sequencers
  SQ_BUSY_CYCLES            2.47144e+06   cycles with active waves in the SQs
  GRBM_COUNT                      95781   cycles while the GPU was measured
  GPUBusy                           300   busy percent, summed over GPU instances
```

## Counter reality on gfx1201 (proven, important)

Through the SDK dispatch-counting service this GPU only returns real values for
the **wavefront / activity family**: `Wavefronts` (= launched_threads/32,
exactly 32768 for a 1M-thread kernel), `SQ_WAVES`, `SQ_BUSY_CYCLES`, `GPUBusy`,
`GRBM_COUNT`. Per-opcode instruction counters (`SQ_INSTS_VALU/LDS/SALU`,
`VALUInsts`, `SALUInsts`), occupancy and L2 (`GL2C_*`) read **0** even with 50
dispatches on RDNA4/gfx1201 — a hardware+SDK reality, not a bug, so the demos
use the working set and `rocprofv3 -d 0 list --pmc` shows the full 143-counter
inventory.
