// kernel_helper.cpp — tiny HIP kernel launcher callable from Python via ctypes,
// so the Python demo can measure a real kernel between
// measure_start/measure_end. Built separately with hipcc.
#include <hip/hip_runtime.h>

#include <cstdio>

namespace {
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
} // namespace

extern "C" int launch_mix(int n) {
  static float *a = nullptr, *b = nullptr, *c = nullptr;
  if (a == nullptr) {
    if (hipMalloc(&a, (size_t)n * sizeof(float)) != hipSuccess ||
        hipMalloc(&b, (size_t)n * sizeof(float)) != hipSuccess ||
        hipMalloc(&c, (size_t)n * sizeof(float)) != hipSuccess)
      return -1;
    if (hipMemset(a, 1, (size_t)n * sizeof(float)) != hipSuccess)
      return -1;
    if (hipMemset(b, 2, (size_t)n * sizeof(float)) != hipSuccess)
      return -1;
  }
  mix<<<(n + 255) / 256, 256>>>(a, b, c, n);
  return (hipDeviceSynchronize() == hipSuccess) ? 0 : 1;
}
