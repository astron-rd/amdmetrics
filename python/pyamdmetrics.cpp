// pyamdmetrics.cpp — Python bindings for amd_metrics (mirrors nv_metrics'
// pynvmetrics).
//
// API mirrors the nv_metrics Python module:
//   query_counters()                -> list of
//   {"name","description","is_derived"} measure(counters: list[str])    ->
//   list[float]   (single measurement window) measure_start(...) /
//   measure_end() -> iterative window control
//
// Importing the module does NOT initialise the rocprofiler tool; the first call
// to any function self-initialises it (see ensure_init_once in
// amd_metrics.cpp).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "amd_metrics.h"

namespace py = pybind11;
using namespace pybind11::literals;

static py::list py_query_counters() {
  std::vector<amd_metrics::CounterInfo> info;
  amd_metrics::query_counters(info);
  py::list out;
  for (auto &c : info)
    out.append(py::dict("name"_a = c.name, "description"_a = c.description,
                        "is_derived"_a = c.is_derived));
  return out;
}

static py::list py_measure(const std::vector<std::string> &names) {
  std::vector<double> vals;
  amd_metrics::measure(names, vals);
  return py::cast(vals);
}

static int py_measure_start(const std::vector<std::string> &names) {
  return amd_metrics::measure_start(names);
}

static py::list py_measure_end() {
  std::vector<double> vals;
  amd_metrics::measure_end(vals);
  return py::cast(vals);
}

PYBIND11_MODULE(amdametrics, m) {
  m.doc() = "AMD GPU hardware-counter measurement (ROCm rocprofiler-sdk)";
  m.def("query_counters", &py_query_counters,
        "List HW counters the AMD GPU supports");
  m.def("measure", &py_measure,
        "Measure counters for kernels launched in this window");
  m.def("measure_start", &py_measure_start, "Begin a measurement window");
  m.def("measure_end", &py_measure_end,
        "End the window and return per-counter sums");
}
