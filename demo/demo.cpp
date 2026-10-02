// demo.cpp — amd_metrics demo:
//   1. query the GPU's supported HW counters
//   2. run a float fadd/fmul/fma mix kernel and collect a useful counter set
//      for every dispatch between measure_start()/measure_end().
// Built as a HIP binary (LANGUAGE HIP in CMake); compiles with hipcc under the
// hood, links libamd_metrics.so. No LD_PRELOAD needed (self-initialising).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "amd_metrics.h"

__global__ void mix(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    float y = b[i];
    float z = x * y + x; // fma
    float w = x + y;     // fadd
    float u = x * y;     // fmul
    c[i] = z + w * 0.5f + u * 0.25f;
  }
}

int main(int argc, char **argv) {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    fprintf(stderr, "hipGetDeviceProperties failed\n");
    return 1;
  }

  // ---- what does the GPU expose? ----
  std::vector<amd_metrics::CounterInfo> info;
  if (amd_metrics::query_counters(info) != 0 || info.empty()) {
    fprintf(stderr, "query_counters failed\n");
    return 1;
  }
  // The counter list also contains read-only agent property constants
  // ("Constant value ... from agent properties"); those never measure anything.
  std::vector<const amd_metrics::CounterInfo *> hw;
  for (auto &c : info)
    if (c.description.find("Constant value") == std::string::npos)
      hw.push_back(&c);

  printf("device:        %s (%d CUs, wavefront size %d)\n", prop.name,
         prop.multiProcessorCount, prop.warpSize);
  printf(
      "%zu supported counters (%zu hardware, %zu agent-property constants)\n",
      info.size(), hw.size(), info.size() - hw.size());
  for (size_t i = 0; i < hw.size() && i < 12; ++i)
    printf("  %-24s %s%s\n", hw[i]->name.c_str(), hw[i]->description.c_str(),
           hw[i]->is_derived ? "  [derived]" : "");
  if (hw.size() > 12)
    printf("  ... (%zu more)\n", hw.size() - 12);

  // ---- measurement window around the kernel ----
  // Counters that do return values for dispatches on RDNA4; the expected column
  // documents what the numbers should be for this launch.
  const int n = 1 << 20;
  const int grid = (n + 255) / 256;
  // Profile on the R9700 when this process sees several GPUs (mirrors the
  // agent preference inside the library); HIP picks device 0 otherwise.
  for (int d = 0, nd = 1; hipGetDeviceCount(&nd) == hipSuccess && d < nd; ++d) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, d) != hipSuccess)
      continue;
    if (strstr(p.name, "R9700") || strstr(p.name, "gfx1201")) {
      if (hipSetDevice(d) == hipSuccess) {
        prop = p;
        break;
      }
    }
  }
  const int ws = prop.warpSize > 0 ? prop.warpSize : 32;
  char exp_wf[64];
  snprintf(exp_wf, sizeof(exp_wf),
           "expected ~ n / %d = %d (plus ramp-up waves)", ws, n / ws);
  // The counters the dispatch-counting service on gfx1201 returns real values
  // for (per-opcode/occupancy/L2 counters exist there but read as zero).
  struct Row {
    const char *name;
    const char *expected;
  };
  std::vector<Row> want = {
      {"Wavefronts", exp_wf},
      {"SQ_WAVES", "waves seen by the sequencers"},
      {"SQ_BUSY_CYCLES", "cycles with active waves in the SQs"},
      {"GRBM_COUNT", "cycles while the GPU was measured"},
      {"GPUBusy", "100*GUI_ACTIVE/GRBM_COUNT, per dispatch "
                  "(>100 = several counted dispatches)"}};
  for (int i = 1; i < argc; ++i)
    want.push_back({argv[i], "user requested"});

  std::vector<std::string> names;
  for (auto &w : want)
    names.push_back(w.name);
  std::vector<std::string> missing;

  // Arm the window FIRST. The counting service binds a queue's counter config
  // at that queue's first dispatch; hipMemset dispatches a fill kernel, so the
  // buffers are touched inside the window (their waves add ~512 to Wavefronts,
  // visible below). Dispatching anything before measure_start() instead binds
  // the queue to the default config and then zeroes GRBM_COUNT/GPUBusy for the
  // measured dispatches — verified on gfx1201.
  if (amd_metrics::measure_start(names, &missing) != 0) {
    fprintf(stderr, "measure_start failed\n");
    return 1;
  }
  for (auto &m : missing)
    fprintf(stderr, "  (skipped unknown counter: %s)\n", m.c_str());

  float *a = nullptr, *b = nullptr, *c = nullptr;
  (void)hipMalloc(&a, n * sizeof(float));
  (void)hipMalloc(&b, n * sizeof(float));
  (void)hipMalloc(&c, n * sizeof(float));
  (void)hipMemset(a, 1, n * sizeof(float));
  (void)hipMemset(b, 2, n * sizeof(float));

  auto t0 = std::chrono::steady_clock::now();
  mix<<<grid, 256>>>(a, b, c, n);
  hipError_t launch_err = hipGetLastError();
  (void)hipDeviceSynchronize();
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

  std::vector<double> vals;
  if (amd_metrics::measure_end(vals) != 0) {
    fprintf(stderr, "measure_end failed\n");
    return 1;
  }
  (void)hipFree(a);
  (void)hipFree(b);
  (void)hipFree(c);
  if (launch_err != hipSuccess) {
    fprintf(stderr, "kernel launch failed: %s\n",
            hipGetErrorString(launch_err));
    return 1;
  }

  printf("\nmix kernel: n=%d, grid=%d, block=256 (wall %.3f ms)\n", n, grid,
         ms);
  for (size_t i = 0; i < want.size() && i < vals.size(); ++i)
    printf("  %-24s %12g   %s\n", want[i].name, vals[i], want[i].expected);
  return 0;
}
