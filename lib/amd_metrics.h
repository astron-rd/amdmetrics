#pragma once

#include <string>
#include <vector>

namespace amd_metrics {

struct CounterInfo {
  std::string name;
  std::string description;
  bool is_derived = false;
};

// Query every hardware counter supported by the AMD GPU in this process.
// Returns 0 on success, -1 on failure (e.g. no AMD agent / SDK init failed).
int query_counters(std::vector<CounterInfo> &out);

// Begin measuring: the named counters are collected for every kernel dispatch
// until measure_end(). Counter names that the GPU does not support are skipped
// and appended to out_missing (optional). Returns 0 on success, -1 on error.
int measure_start(const std::vector<std::string> &names,
                  std::vector<std::string> *out_missing = nullptr);

// Stop measuring and return one value per requested counter, in request order,
// summed over all measured dispatches (and per-instance values). Requires a
// prior measure_start(). Returns 0 on success, -1 on error.
int measure_end(std::vector<double> &out);

// Convenience wrapper around measure_start()/measure_end().
int measure(const std::vector<std::string> &names, std::vector<double> &out);

} // namespace amd_metrics
