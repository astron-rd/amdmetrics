#!/usr/bin/env python3
"""amd_metrics Python demo:
   1. query the GPU's supported HW counters
   2. measure Wavefronts/SQ_WAVES/GPUBusy/GRBM_COUNT around a HIP kernel launched
      through kernel_helper.so (loaded with ctypes), three separate windows.
"""
import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# find the amdametrics extension in the repo's build dir when not installed
# (setting PYTHONPATH overrides this)
_build_py = os.path.join(_HERE, "..", "build", "python")
if os.path.isdir(_build_py):
    sys.path.insert(0, _build_py)

import amdametrics

LIB = os.path.join(_HERE, "libkernels.so")
lib = ctypes.CDLL(LIB)
lib.launch_mix.argtypes = [ctypes.c_int]
lib.launch_mix.restype = ctypes.c_int

print("== query_counters ==")
info = amdametrics.query_counters()
print("GPU supports %d HW counters:" % len(info))
for c in info[:8]:
    print(
        "  %-28s %s%s"
        % (c["name"], c["description"], "  [derived]" if c["is_derived"] else "")
    )
if len(info) > 8:
    print("  ... (%d more)" % (len(info) - 8))

names = ["Wavefronts", "SQ_WAVES", "SQ_BUSY_CYCLES", "GRBM_COUNT", "GPUBusy"]
print("== measure ==")
for it in range(3):
    amdametrics.measure_start(names)
    rc = lib.launch_mix(1 << 20)
    assert rc == 0, "launch_mix failed"
    vals = amdametrics.measure_end()
    print(
        "window %d:" % (it + 1),
        "  ".join("%s=%g" % (n, v) for n, v in zip(names, vals)),
    )
print("done")
