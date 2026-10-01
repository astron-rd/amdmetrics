// amd_metrics.cpp — in-process rocprofiler-sdk counter tool + measurement API.
//
// Self-initializes via the registration API (no LD_PRELOAD required); also runs
// as a normal preload tool.
#include "amd_metrics.h"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define AMD_CK(fn, msg)                                                        \
  do {                                                                         \
    rocprofiler_status_t s_ = (fn);                                            \
    if (s_ != ROCPROFILER_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "[amd_metrics] %s: %s failed: %s\n", msg, #fn,           \
              rocprofiler_get_status_string(s_));                              \
      return -1;                                                               \
    }                                                                          \
  } while (0)

namespace {

// ---- counter configuration (per agent) -----------------------------------
std::map<uint64_t, rocprofiler_counter_config_id_t>
    g_configs;                                     // agent -> config
std::unordered_map<uint64_t, std::string> g_cname; // counter id -> name
std::unordered_map<uint64_t, size_t> g_pos;        // counter id -> request idx
std::vector<std::string> g_requested;
std::vector<rocprofiler_agent_v0_t> g_agents;

// ---- measurement window state -------------------------------------------
std::mutex g_mutex;
std::atomic<bool> g_measuring{false};
std::set<uint64_t> g_measured; // dispatch ids in window
std::vector<double> g_accum;   // per-request sums
std::atomic<uint64_t> g_drained{0};

static rocprofiler_context_id_t g_ctx{0};
static rocprofiler_buffer_id_t g_buf{0};

// default config used until measure_start picks a set (the counters proven to
// return real values on gfx1201 through the SDK dispatch-counting service).
const char *const kDefault[4] = {"Wavefronts", "SQ_WAVES", "GPUBusy",
                                 "GRBM_COUNT"};

// forward decl of the tool entry so we can register it too
extern "C" rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *);

} // namespace

namespace amd_metrics {

// internal (implementation detail, but needed by query/measure_start)
namespace impl {

int collect_supported(rocprofiler_agent_id_t agent,
                      std::vector<rocprofiler_counter_id_t> &out) {
  return rocprofiler_iterate_agent_supported_counters(
             agent,
             [](rocprofiler_agent_id_t, rocprofiler_counter_id_t *c, size_t n,
                void *ud) {
               auto *v =
                   static_cast<std::vector<rocprofiler_counter_id_t> *>(ud);
               for (size_t i = 0; i < n; ++i)
                 v->push_back(c[i]);
               return ROCPROFILER_STATUS_SUCCESS;
             },
             &out) == ROCPROFILER_STATUS_SUCCESS
             ? 0
             : -1;
}

// names of every counter this agent supports (for target selection and
// per-agent resolution on heterogeneous nodes where GPUs differ in arch).
int supported_names(rocprofiler_agent_id_t agent, std::set<std::string> &out) {
  std::vector<rocprofiler_counter_id_t> sup;
  if (collect_supported(agent, sup) != 0)
    return -1;
  for (auto &c : sup) {
    rocprofiler_counter_info_v0_t info;
    if (rocprofiler_query_counter_info(c, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                       &info) == 0 &&
        info.name)
      out.insert(info.name);
  }
  return 0;
}

} // namespace impl

// ---------------------------------------------------------------------------
// dispatch_cb: (synchronously on kernel enqueue) hand the per-agent counter
// config to the dispatch-counting service, and mark the dispatch if a window is
// armed.
// ---------------------------------------------------------------------------
static void dispatch_cb(rocprofiler_dispatch_counting_service_data_t data,
                        rocprofiler_counter_config_id_t *config,
                        rocprofiler_user_data_t *, void *) {
  auto it = g_configs.find(data.dispatch_info.agent_id.handle);
  *config =
      (it != g_configs.end()) ? it->second : rocprofiler_counter_config_id_t{0};
  if (g_measuring.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_measured.insert(data.dispatch_info.dispatch_id);
  }
}

// ---------------------------------------------------------------------------
// buffered_cb: (callback thread) accumulate counter values for measured
// dispatches.
// ---------------------------------------------------------------------------
static void buffered_cb(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                        rocprofiler_record_header_t **headers,
                        size_t num_headers, void *, uint64_t) {
  static uint64_t cur_disp = 0; // dispatch whose records we are currently in
  bool locksum = false;
  for (size_t i = 0; i < num_headers; ++i) {
    auto *h = headers[i];
    if (h->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS)
      continue;
    if (h->kind ==
        ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER) {
      auto *r = static_cast<rocprofiler_dispatch_counting_service_record_t *>(
          h->payload);
      cur_disp = r->dispatch_info.dispatch_id;
      locksum = false; // (re)evaluated per VALUE record below
    } else if (h->kind == ROCPROFILER_COUNTER_RECORD_VALUE) {
      auto *r = static_cast<rocprofiler_counter_record_t *>(h->payload);
      if (!locksum) {
        std::lock_guard<std::mutex> lk(g_mutex);
        locksum = g_measured.count(cur_disp) > 0;
      }
      if (!locksum)
        continue;
      rocprofiler_counter_id_t cid{0};
      if (rocprofiler_query_record_counter_id(r->id, &cid) !=
          ROCPROFILER_STATUS_SUCCESS)
        continue;
      auto it = g_pos.find(cid.handle);
      if (it == g_pos.end())
        continue;
      std::lock_guard<std::mutex> lk(g_mutex);
      g_accum[it->second] += r->counter_value;
    }
  }
  g_drained.fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// build a counter config for one agent from an ordered list of counter ids.
// ---------------------------------------------------------------------------
static int rebuild_config(const rocprofiler_agent_v0_t &agent,
                          std::vector<rocprofiler_counter_id_t> &ids) {
  rocprofiler_counter_config_id_t cfg{0};
  AMD_CK(
      rocprofiler_create_counter_config(agent.id, ids.data(), ids.size(), &cfg),
      "create counter config");
  auto it = g_configs.find(agent.id.handle);
  if (it != g_configs.end())
    rocprofiler_destroy_counter_config(it->second); // free old config, if any
  g_configs[agent.id.handle] = cfg;
  return 0;
}

// resolve names -> (ordered ids + position map) against ONE agent's supported
// list.
static int resolve_names(rocprofiler_agent_id_t agent,
                         const std::vector<std::string> &names,
                         std::vector<rocprofiler_counter_id_t> &ids,
                         std::vector<std::string> &missing) {
  std::vector<rocprofiler_counter_id_t> supported;
  if (impl::collect_supported(agent, supported) != 0)
    return -1;
  std::unordered_map<std::string, rocprofiler_counter_id_t> byname;
  for (auto &c : supported) {
    rocprofiler_counter_info_v0_t info;
    if (rocprofiler_query_counter_info(c, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                       &info) == 0)
      byname[info.name] = c;
  }
  g_pos.clear();
  ids.clear();
  missing.clear();
  for (size_t i = 0; i < names.size(); ++i) {
    auto it = byname.find(names[i]);
    if (it == byname.end()) {
      missing.push_back(names[i]);
      continue;
    }
    g_pos[it->second.handle] = i;
    ids.push_back(it->second);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// tool entry / lifecycle
// ---------------------------------------------------------------------------
static int tool_init(rocprofiler_client_finalize_t, void *) {
  std::vector<rocprofiler_agent_v0_t> gpus;
  rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0,
      [](rocprofiler_agent_version_t, const void **arr, size_t n, void *ud) {
        auto *v = static_cast<std::vector<rocprofiler_agent_v0_t> *>(ud);
        for (size_t i = 0; i < n; ++i) {
          auto *a = static_cast<const rocprofiler_agent_v0_t *>(arr[i]);
          if (a->type == ROCPROFILER_AGENT_TYPE_GPU)
            v->push_back(*a);
        }
        return ROCPROFILER_STATUS_SUCCESS;
      },
      sizeof(rocprofiler_agent_t), &gpus);
  if (gpus.empty()) {
    fprintf(stderr, "[amd_metrics] no AMD GPU agent found\n");
    return -1;
  }
  // GPUs are enumerated in board order; on multi-GPU nodes (gpu04 also carries
  // a W7700/gfx1101) the first GPU may not be the R9700/gfx1201 we profile.
  // Prefer the agent that names the R9700/gfx1201 explicitly, then fall back
  // to the first GPU that supports the default counter set.
  auto hasstr = [](const char *s, const char *needle) {
    return s && std::string(s).find(needle) != std::string::npos;
  };
  auto is_target = [&](const rocprofiler_agent_v0_t &a) {
    return hasstr(a.name, "gfx1201") || hasstr(a.product_name, "R9700") ||
           hasstr(a.model_name, "gfx1201");
  };
  const std::vector<std::string> def{kDefault, kDefault + 4};
  for (size_t i = 0; i < gpus.size(); ++i)
    if (is_target(gpus[i])) {
      std::swap(gpus[0], gpus[i]);
      break;
    }
  for (size_t i = 1; i < gpus.size(); ++i) {
    std::set<std::string> names;
    if (impl::supported_names(gpus[i].id, names) != 0)
      continue;
    if (std::all_of(def.begin(), def.end(),
                    [&](const std::string &n) { return names.count(n) > 0; })) {
      std::swap(gpus[0], gpus[i]);
      break;
    }
  }
  g_agents = std::move(gpus);

  AMD_CK(rocprofiler_create_context(&g_ctx), "create context");
  AMD_CK(rocprofiler_create_buffer(g_ctx, 4096, 2048,
                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                   buffered_cb, nullptr, &g_buf),
         "create buffer");

  // default config per agent (working counters on gfx1201), resolved against
  // EACH agent: a different-arch GPU on the same node simply gets no config.
  for (auto &a : g_agents) {
    std::vector<rocprofiler_counter_id_t> a_ids;
    std::vector<std::string> a_missing;
    if (resolve_names(a.id, def, a_ids, a_missing) != 0)
      return -1;
    if (a_ids.empty())
      continue;
    if (rebuild_config(a, a_ids) != 0)
      return -1;
    for (auto &c : a_ids) {
      rocprofiler_counter_info_v0_t info;
      rocprofiler_query_counter_info(c, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                     &info);
      g_cname[c.handle] = info.name;
    }
  }

  rocprofiler_callback_thread_t th;
  AMD_CK(rocprofiler_create_callback_thread(&th), "create callback thread");
  AMD_CK(rocprofiler_assign_callback_thread(g_buf, th),
         "assign callback thread");
  AMD_CK(rocprofiler_configure_buffer_dispatch_counting_service(
             g_ctx, g_buf, dispatch_cb, nullptr),
         "configure dispatch counting service");
  AMD_CK(rocprofiler_start_context(g_ctx), "start context");
  return 0;
}

static void tool_fini(void *) {
  rocprofiler_flush_buffer(g_buf);
  rocprofiler_stop_context(g_ctx);
}

// ensure the SDK + tool are up (works preloaded or linked / imported).
static void ensure_init_once() {
  static std::once_flag once;
  std::call_once(once,
                 [] { rocprofiler_force_configure(rocprofiler_configure); });
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------
int query_counters(std::vector<CounterInfo> &out) {
  ensure_init_once();
  if (g_agents.empty())
    return -1;
  std::vector<rocprofiler_counter_id_t> supported;
  if (impl::collect_supported(g_agents.front().id, supported) != 0)
    return -1;
  out.clear();
  out.reserve(supported.size());
  for (auto &c : supported) {
    rocprofiler_counter_info_v0_t info;
    if (rocprofiler_query_counter_info(c, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                       &info) != 0)
      continue;
    out.push_back({info.name ? info.name : "",
                   info.description ? info.description : "",
                   info.is_derived != 0});
  }
  return 0;
}

int measure_start(const std::vector<std::string> &names,
                  std::vector<std::string> *out_missing) {
  ensure_init_once();
  if (g_agents.empty())
    return -1;

  // resolve against the target (primary) agent: sets g_pos + authoritative
  // missing list.
  std::vector<rocprofiler_counter_id_t> ids;
  std::vector<std::string> missing;
  if (resolve_names(g_agents.front().id, names, ids, missing) != 0)
    return -1;
  if (out_missing)
    *out_missing = missing;

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_accum.assign(names.size(), 0.0);
    g_requested = names;
    g_measured.clear();
  }
  // per-agent configs: use each agent's own counter ids; an agent that does
  // not expose the requested counters (different arch) simply gets no config.
  for (auto &a : g_agents) {
    if (a.id.handle == g_agents.front().id.handle) {
      if (rebuild_config(a, ids) != 0)
        return -1;
      continue;
    }
    std::vector<rocprofiler_counter_id_t> a_ids;
    std::vector<std::string> a_missing;
    if (resolve_names(a.id, names, a_ids, a_missing) != 0)
      return -1;
    if (a_ids.empty())
      continue;
    if (rebuild_config(a, a_ids) != 0)
      return -1;
  }
  for (auto &c : ids) {
    rocprofiler_counter_info_v0_t info;
    rocprofiler_query_counter_info(c, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                   &info);
    g_cname[c.handle] = info.name;
  }

  g_measuring.store(true, std::memory_order_release);
  return 0;
}

int measure_end(std::vector<double> &out) {
  g_measuring.store(false, std::memory_order_release);

  // drain: flush the buffer, wait for the callback to process the final batch.
  uint64_t before = g_drained.load(std::memory_order_acquire);
  rocprofiler_flush_buffer(g_buf);
  for (int spins = 0; spins < 2000; ++spins) {
    if (g_drained.load(std::memory_order_acquire) > before)
      break;
    usleep(1000); // up to ~2s
  }

  std::lock_guard<std::mutex> lk(g_mutex);
  out = g_accum;
  g_accum.assign(g_accum.size(), 0.0);
  g_measured.clear();
  return 0;
}

int measure(const std::vector<std::string> &names, std::vector<double> &out) {
  int rc = measure_start(names, nullptr);
  if (rc != 0)
    return rc;
  return measure_end(out);
}

} // namespace amd_metrics

// ---------------------------------------------------------------------------
// rocprofiler-sdk tool entry (also reachable via the registration API above)
// ---------------------------------------------------------------------------
extern "C" rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t priority,
                      rocprofiler_client_id_t *id) {
  if (priority > 0)
    return nullptr;
  id->name = "amd_metrics";
  static rocprofiler_tool_configure_result_t cfg = {
      sizeof(rocprofiler_tool_configure_result_t), &amd_metrics::tool_init,
      &amd_metrics::tool_fini, nullptr};
  return &cfg;
}
