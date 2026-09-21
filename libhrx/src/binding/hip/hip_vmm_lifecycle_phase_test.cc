// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "binding/hip/api.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

constexpr size_t kDeterministicPoolBytes = size_t{64} * 1024 * 1024;
constexpr int kAllocationRetireReservedPhase = 22;

template <typename T>
T Resolve(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class Notification {
 public:
  void Notify() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      notified_ = true;
    }
    condition_.notify_all();
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return notified_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool notified_ = false;
};

using HipInitFn = hipError_t (*)(unsigned int);
using HipHALDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int*);
using HipGetDeviceCountFn = hipError_t (*)(int*);
using HipDeviceResetFn = hipError_t (*)(void);
using HipDevicePrimaryCtxReleaseFn = hipError_t (*)(hipDevice_t);
using HipDevicePrimaryCtxResetFn = hipError_t (*)(hipDevice_t);
using HipDevicePrimaryCtxRetainFn = hipError_t (*)(hipCtx_t*, hipDevice_t);
using HipCtxCreateFn = hipError_t (*)(hipCtx_t*, unsigned int, hipDevice_t);
using HipCtxDestroyFn = hipError_t (*)(hipCtx_t);
using HipCtxGetCurrentFn = hipError_t (*)(hipCtx_t*);
using HipCtxSetCurrentFn = hipError_t (*)(hipCtx_t);
using HipCtxPushCurrentFn = hipError_t (*)(hipCtx_t);
using HipCtxPopCurrentFn = hipError_t (*)(hipCtx_t*);
using HipDeviceGetDevResourceFn = hipError_t (*)(hipDevice_t, hipDevResource*,
                                                 hipDevResourceType);
using HipDevResourceGenerateDescFn = hipError_t (*)(hipDevResourceDesc_t*,
                                                    hipDevResource*,
                                                    unsigned int);
using HipGreenCtxCreateFn = hipError_t (*)(hipExecutionCtx_t*,
                                           hipDevResourceDesc_t, int,
                                           unsigned int);
using HipExecutionCtxDestroyFn = hipError_t (*)(hipExecutionCtx_t);
using HipSetDeviceFn = hipError_t (*)(int);
using HipDeviceGetAttributeFn = hipError_t (*)(int*, hipDeviceAttribute_t, int);
using HipMemGetAllocationGranularityFn = hipError_t (*)(
    size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags);
using HipMemAddressReserveFn = hipError_t (*)(void**, size_t, size_t, void*,
                                              unsigned long long);
using HipMemAddressFreeFn = hipError_t (*)(void*, size_t);
using HipMemCreateFn = hipError_t (*)(hipMemGenericAllocationHandle_t*, size_t,
                                      const hipMemAllocationProp*,
                                      unsigned long long);
using HipMemReleaseFn = hipError_t (*)(hipMemGenericAllocationHandle_t);
using HipMemMapFn = hipError_t (*)(void*, size_t, size_t,
                                   hipMemGenericAllocationHandle_t,
                                   unsigned long long);
using HipMemUnmapFn = hipError_t (*)(void*, size_t);
using HipMemSetAccessFn = hipError_t (*)(void*, size_t, const hipMemAccessDesc*,
                                         size_t);
using HipMemGetAccessFn = hipError_t (*)(unsigned long long*,
                                         const hipMemLocation*, void*);
using HipMemcpyFn = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind);
using HipMemGetInfoFn = hipError_t (*)(size_t*, size_t*);
using HipMemPoolCreateFn = hipError_t (*)(hipMemPool_t*,
                                          const hipMemPoolProps*);
using HipMemPoolDestroyFn = hipError_t (*)(hipMemPool_t);
using HipMemPoolTrimToFn = hipError_t (*)(hipMemPool_t, size_t);
using HipMemPoolSetAttributeFn = hipError_t (*)(hipMemPool_t,
                                                hipMemPool_attribute, void*);
using HipDeviceGetDefaultMemPoolFn = hipError_t (*)(hipMemPool_t*, int);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t*, size_t);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t);
using HipMallocAsyncFn = hipError_t (*)(void**, size_t, hipStream_t);
using HipMallocFromPoolAsyncFn = hipError_t (*)(void**, size_t, hipMemPool_t,
                                                hipStream_t);
using HipFreeAsyncFn = hipError_t (*)(void*, hipStream_t);
using HipStreamCreateFn = hipError_t (*)(hipStream_t*);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t*, unsigned int);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t*, unsigned int);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t);
using HipGraphAddHostNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                             const hipGraphNode_t*, size_t,
                                             const void*);
using HipGraphAddMemcpyNode1DFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                                 const hipGraphNode_t*, size_t,
                                                 void*, const void*, size_t,
                                                 hipMemcpyKind);
using HipGraphAddMemcpyNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                               const hipGraphNode_t*, size_t,
                                               const void*);
using HipGraphAddMemsetNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                               const hipGraphNode_t*, size_t,
                                               const void*);
using HipGraphAddMemAllocNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                                 const hipGraphNode_t*, size_t,
                                                 void*);
using HipGraphAddMemFreeNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                                const hipGraphNode_t*, size_t,
                                                void*);
using HipGraphAddEventRecordNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                                    const hipGraphNode_t*,
                                                    size_t, hipEvent_t);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t, hipGraphNode_t*, size_t*);
using HipGraphGetEdgesFn = hipError_t (*)(hipGraph_t, hipGraphNode_t*,
                                          hipGraphNode_t*, size_t*);
using HipGraphGetRootNodesFn = hipError_t (*)(hipGraph_t, hipGraphNode_t*,
                                              size_t*);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t*, hipGraph_t,
                                             hipGraphNode_t*, char*, size_t);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t, hipStream_t);
using HipGraphExecUpdateFn = hipError_t (*)(hipGraphExec_t, hipGraph_t,
                                            hipGraphNode_t*,
                                            hipGraphExecUpdateResult*);
using HipGraphExecMemcpyNodeSetParams1DFn = hipError_t (*)(
    hipGraphExec_t, hipGraphNode_t, void*, const void*, size_t, hipMemcpyKind);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t);

using PhaseObserverFn = void (*)(int, void*, void*);
using SetPhaseObserverFn = hipError_t (*)(PhaseObserverFn, void*);
using WaitPhaseObserverClosingFn = hipError_t (*)(void);
using NotifyPhaseForTestFn = void (*)(int);
using LowerAddressFreeObserverFn = void (*)(bool, void*, size_t, void*);
using SetLowerAddressFreeObserverFn = hipError_t (*)(LowerAddressFreeObserverFn,
                                                     void*);
using ProbeAcceptedAccessFn = hipError_t (*)(void*, size_t, int);
using FailOnceFn = void (*)(void);
using FailAtFn = void (*)(int);
using SetIntFn = void (*)(int);
using SetBoolIntFn = void (*)(bool, int);
using QueryU64Fn = uint64_t (*)(void);
using QueryBoolU64Fn = uint64_t (*)(bool);
using QueryBoolU32Fn = uint32_t (*)(bool);
using QueryIntFn = int (*)(int);
using DeviceOperationFn = hipError_t (*)(int);
using QueryGraphU64Fn = uint64_t (*)(hipGraph_t);
using QueryContextU64Fn = uint64_t (*)(hipCtx_t);
using QueryPointerU64Fn = uint64_t (*)(void*);
using QueryOrdinalU64Fn = uint64_t (*)(int);
using QueryU32Fn = uint32_t (*)(void);
using QueryRegistryStateFn = hipError_t (*)(size_t*, size_t*, size_t*);
using SetEpochExhaustionFn = hipError_t (*)(int, int, bool);
using InstallGraphExecAllocatorFn = hipError_t (*)(hipGraph_t);
using FailGraphExecAllocationAtFn = hipError_t (*)(int);
using InjectTlsFailuresFn = void (*)(int);

constexpr int kTlsTestFailureSet = 1 << 1;
constexpr int kTlsTestFailureClear = 1 << 2;

struct Api {
  void* library = nullptr;
  HipInitFn init = nullptr;
  HipHALDeinitFn hal_deinit = nullptr;
  HipGetDeviceFn get_device = nullptr;
  HipGetDeviceCountFn get_device_count = nullptr;
  HipDeviceResetFn device_reset = nullptr;
  HipDevicePrimaryCtxReleaseFn device_primary_ctx_release = nullptr;
  HipDevicePrimaryCtxResetFn device_primary_ctx_reset = nullptr;
  HipDevicePrimaryCtxRetainFn device_primary_ctx_retain = nullptr;
  HipCtxCreateFn ctx_create = nullptr;
  HipCtxDestroyFn ctx_destroy = nullptr;
  HipCtxGetCurrentFn ctx_get_current = nullptr;
  HipCtxSetCurrentFn ctx_set_current = nullptr;
  HipCtxPushCurrentFn ctx_push_current = nullptr;
  HipCtxPopCurrentFn ctx_pop_current = nullptr;
  HipDeviceGetDevResourceFn device_get_resource = nullptr;
  HipDevResourceGenerateDescFn generate_descriptor = nullptr;
  HipGreenCtxCreateFn green_ctx_create = nullptr;
  HipExecutionCtxDestroyFn execution_ctx_destroy = nullptr;
  HipSetDeviceFn set_device = nullptr;
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  HipMemGetAllocationGranularityFn mem_get_allocation_granularity = nullptr;
  HipMemAddressReserveFn mem_address_reserve = nullptr;
  HipMemAddressFreeFn mem_address_free = nullptr;
  HipMemCreateFn mem_create = nullptr;
  HipMemReleaseFn mem_release = nullptr;
  HipMemMapFn mem_map = nullptr;
  HipMemUnmapFn mem_unmap = nullptr;
  HipMemSetAccessFn mem_set_access = nullptr;
  HipMemGetAccessFn mem_get_access = nullptr;
  HipMemcpyFn memcpy = nullptr;
  HipMemGetInfoFn mem_get_info = nullptr;
  HipMemPoolCreateFn mem_pool_create = nullptr;
  HipMemPoolDestroyFn mem_pool_destroy = nullptr;
  HipMemPoolTrimToFn mem_pool_trim_to = nullptr;
  HipMemPoolSetAttributeFn mem_pool_set_attribute = nullptr;
  HipDeviceGetDefaultMemPoolFn device_get_default_mem_pool = nullptr;
  HipMallocFn malloc = nullptr;
  HipFreeFn free = nullptr;
  HipMallocAsyncFn malloc_async = nullptr;
  HipMallocFromPoolAsyncFn malloc_from_pool_async = nullptr;
  HipFreeAsyncFn free_async = nullptr;
  HipStreamCreateFn stream_create = nullptr;
  HipStreamDestroyFn stream_destroy = nullptr;
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  HipEventCreateWithFlagsFn event_create_with_flags = nullptr;
  HipEventDestroyFn event_destroy = nullptr;
  HipGraphCreateFn graph_create = nullptr;
  HipGraphDestroyFn graph_destroy = nullptr;
  HipGraphAddHostNodeFn graph_add_host_node = nullptr;
  HipGraphAddMemcpyNodeFn graph_add_memcpy_node = nullptr;
  HipGraphAddMemcpyNode1DFn graph_add_memcpy_node_1d = nullptr;
  HipGraphAddMemsetNodeFn graph_add_memset_node = nullptr;
  HipGraphAddMemAllocNodeFn graph_add_mem_alloc_node = nullptr;
  HipGraphAddMemFreeNodeFn graph_add_mem_free_node = nullptr;
  HipGraphAddEventRecordNodeFn graph_add_event_record_node = nullptr;
  HipGraphGetNodesFn graph_get_nodes = nullptr;
  HipGraphGetEdgesFn graph_get_edges = nullptr;
  HipGraphGetRootNodesFn graph_get_root_nodes = nullptr;
  HipGraphInstantiateFn graph_instantiate = nullptr;
  HipGraphLaunchFn graph_launch = nullptr;
  HipGraphExecUpdateFn graph_exec_update = nullptr;
  HipGraphExecMemcpyNodeSetParams1DFn graph_exec_memcpy_node_set_params_1d =
      nullptr;
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  SetPhaseObserverFn set_phase_observer = nullptr;
  WaitPhaseObserverClosingFn wait_phase_observer_closing = nullptr;
  NotifyPhaseForTestFn notify_phase_for_test = nullptr;
  SetLowerAddressFreeObserverFn set_lower_address_free_observer = nullptr;
  SetLowerAddressFreeObserverFn clear_lower_address_free_observer = nullptr;
  FailOnceFn fail_next_lower_observer_clear = nullptr;
  ProbeAcceptedAccessFn probe_accepted_access = nullptr;
  FailOnceFn fail_green_context_after_retain_once = nullptr;
  FailOnceFn fail_lower_virtual_reserve_after_native_once = nullptr;
  FailOnceFn fail_hrx_virtual_reserve_after_native_once = nullptr;
  QueryU64Fn last_virtual_reserve_address = nullptr;
  QueryU64Fn last_virtual_reserve_alignment = nullptr;
  FailOnceFn fail_address_reserve_after_native_once = nullptr;
  FailOnceFn fail_physical_create_after_native_once = nullptr;
  SetBoolIntFn fail_cleanup_count = nullptr;
  QueryU64Fn quarantine_count = nullptr;
  QueryBoolU64Fn cleanup_attempt_count = nullptr;
  QueryBoolU32Fn last_cleanup_status = nullptr;
  FailOnceFn arm_quarantine_drain_pause = nullptr;
  FailOnceFn wait_quarantine_drain_paused = nullptr;
  FailOnceFn release_quarantine_drain_pause = nullptr;
  QueryIntFn quarantine_drain_attempt_kind = nullptr;
  DeviceOperationFn quarantine_drain_without_writer = nullptr;
  FailAtFn fail_plan_allocation_at = nullptr;
  FailOnceFn fail_plan_revalidation_once = nullptr;
  FailAtFn fail_native_operation_at = nullptr;
  QueryU64Fn native_operation_count = nullptr;
  QueryU64Fn native_operation_cursor = nullptr;
  QueryOrdinalU64Fn native_operation_attempt_count = nullptr;
  QueryU32Fn first_native_error = nullptr;
  QueryU64Fn raw_native_call_count = nullptr;
  QueryRegistryStateFn query_allocation_registry = nullptr;
  FailAtFn fail_access_preflight_at = nullptr;
  FailOnceFn fail_graph_exec_memcpy_prepare_once = nullptr;
  FailOnceFn fail_graph_exec_rebuild_once = nullptr;
  FailAtFn fail_graph_add_at = nullptr;
  SetIntFn set_graph_peer_transfer_failure = nullptr;
  QueryGraphU64Fn graph_internal_node_count = nullptr;
  QueryGraphU64Fn graph_internal_root_count = nullptr;
  QueryGraphU64Fn graph_internal_dependency_count = nullptr;
  QueryGraphU64Fn graph_owned_host_allocation_count = nullptr;
  QueryGraphU64Fn graph_user_object_reference_count = nullptr;
  QueryContextU64Fn context_reference_count = nullptr;
  QueryPointerU64Fn pointer_buffer_reference_count = nullptr;
  FailOnceFn reset_observability = nullptr;
  SetEpochExhaustionFn set_epoch_exhaustion = nullptr;
  InstallGraphExecAllocatorFn install_graph_exec_allocator = nullptr;
  FailGraphExecAllocationAtFn fail_graph_exec_allocation_at = nullptr;
  InjectTlsFailuresFn inject_tls_failures = nullptr;
};

class ScopedEpochExhaustion {
 public:
  ScopedEpochExhaustion(SetEpochExhaustionFn setter, int device, int source)
      : setter_(setter), device_(device), source_(source) {
    enable_result_ = setter_(device_, source_, /*enabled=*/true);
    armed_ = enable_result_ == hipSuccess;
  }

  ScopedEpochExhaustion(const ScopedEpochExhaustion&) = delete;
  ScopedEpochExhaustion& operator=(const ScopedEpochExhaustion&) = delete;

  ~ScopedEpochExhaustion() {
    if (armed_) {
      EXPECT_EQ(hipSuccess, setter_(device_, source_, /*enabled=*/false));
    }
  }

  hipError_t enable_result() const { return enable_result_; }

  hipError_t Restore() {
    if (!armed_) return hipErrorIllegalState;
    const hipError_t result = setter_(device_, source_, /*enabled=*/false);
    if (result == hipSuccess) armed_ = false;
    return result;
  }

 private:
  SetEpochExhaustionFn setter_ = nullptr;
  int device_ = -1;
  int source_ = -1;
  hipError_t enable_result_ = hipErrorUnknown;
  bool armed_ = false;
};

class ScopedPhaseObserver {
 public:
  ScopedPhaseObserver(SetPhaseObserverFn setter, PhaseObserverFn observer,
                      void* user_data)
      : setter_(setter) {
    EXPECT_EQ(hipSuccess, setter_(observer, user_data));
  }

  ScopedPhaseObserver(const ScopedPhaseObserver&) = delete;
  ScopedPhaseObserver& operator=(const ScopedPhaseObserver&) = delete;

  ~ScopedPhaseObserver() { Reset(); }

  void Reset() {
    if (setter_) {
      EXPECT_EQ(hipSuccess, setter_(nullptr, nullptr));
      setter_ = nullptr;
    }
  }

 private:
  SetPhaseObserverFn setter_ = nullptr;
};

class ScopedEnvironmentVariable {
 public:
  ScopedEnvironmentVariable(const char* name, const std::string& value)
      : name_(name) {
    const char* old_value = std::getenv(name);
    if (old_value) {
      had_old_value_ = true;
      old_value_ = old_value;
    }
    EXPECT_EQ(0, setenv(name_.c_str(), value.c_str(), /*overwrite=*/1));
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) =
      delete;

  ~ScopedEnvironmentVariable() {
    if (had_old_value_) {
      EXPECT_EQ(0, setenv(name_.c_str(), old_value_.c_str(), /*overwrite=*/1));
    } else {
      EXPECT_EQ(0, unsetenv(name_.c_str()));
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool had_old_value_ = false;
};

struct PhaseState {
  Notification writer_pending;
  Notification reader_waiting;
  Notification access_pre_drain;
  Notification before_device_writer;
  Notification release_writer;
  Notification release_pre_drain;
  Notification release_before_device_writer;
  Notification graph_exec_active_launch_waiting;
  Notification graph_peer_transfer_waiting;
  Notification release_graph_peer_transfer;
  Notification teardown_stream_waiting;
  std::atomic<int> teardown_stream_wait_count{0};
};

void ObservePhase(int phase, void* object, void* user_data) {
  (void)object;
  auto* state = static_cast<PhaseState*>(user_data);
  switch (phase) {
    case 1:
      state->writer_pending.Notify();
      state->release_writer.Wait();
      break;
    case 2:
      state->reader_waiting.Notify();
      break;
    case 3:
      state->access_pre_drain.Notify();
      state->release_pre_drain.Wait();
      break;
    case 4:
      state->before_device_writer.Notify();
      state->release_before_device_writer.Wait();
      break;
    case 18:
      state->graph_exec_active_launch_waiting.Notify();
      break;
    case 19:
      state->graph_peer_transfer_waiting.Notify();
      state->release_graph_peer_transfer.Wait();
      break;
    case 20:
      state->teardown_stream_wait_count.fetch_add(1, std::memory_order_acq_rel);
      state->teardown_stream_waiting.Notify();
      break;
  }
}

struct MixedResetPhaseState {
  // Signals that reset has closed lifecycle admission before taking its VMM
  // reservation snapshot.
  Notification writer_pending;
  // Allows reset to take its VMM reservation snapshot after the callback's
  // access probe has completed.
  Notification release_writer;
  // Signals that the exact foreign incident context reached a non-empty
  // accepted stream frontier wait.
  Notification incident_context_stream_waiting;
  // Device B's current context, used to select its incident stream-wait phase.
  void* expected_incident_context = nullptr;
  // Context object carried by the matching incident stream-wait phase.
  std::atomic<uintptr_t> observed_incident_context{0};
};

void ObserveMixedResetPhase(int phase, void* object, void* user_data) {
  auto* state = static_cast<MixedResetPhaseState*>(user_data);
  switch (phase) {
    case 1:
      state->writer_pending.Notify();
      state->release_writer.Wait();
      break;
    case 23:
      if (object != state->expected_incident_context) break;
      state->observed_incident_context.store(
          reinterpret_cast<uintptr_t>(object), std::memory_order_release);
      state->incident_context_stream_waiting.Notify();
      break;
  }
}

struct ObserverRegistryState {
  int target_phase = 0;
  Notification entered;
  Notification release;
  SetPhaseObserverFn setter = nullptr;
  bool block_callback = false;
  bool clear_reentrantly = false;
  std::atomic<int> callback_count{0};
  std::atomic<int> reentrant_clear_result{hipErrorUnknown};
};

struct RegistryState {
  size_t count = 0;
  size_t pending = 0;
  size_t capacity = 0;
};

RegistryState QueryRegistryState(QueryRegistryStateFn query) {
  RegistryState state;
  EXPECT_EQ(hipSuccess, query(&state.count, &state.pending, &state.capacity));
  return state;
}

void ExpectValidRegistryOccupancy(const RegistryState& state) {
  ASSERT_LE(state.count, state.capacity);
  EXPECT_LE(state.pending, state.capacity - state.count);
}

void ObserveRegistryPhase(int phase, void* object, void* user_data) {
  (void)object;
  auto* state = static_cast<ObserverRegistryState*>(user_data);
  if (phase != state->target_phase) return;
  state->callback_count.fetch_add(1, std::memory_order_acq_rel);
  if (state->clear_reentrantly) {
    state->reentrant_clear_result.store(state->setter(nullptr, nullptr),
                                        std::memory_order_release);
  }
  state->entered.Notify();
  if (state->block_callback) state->release.Wait();
}

void IgnoreLowerAddressFree(bool entering, void* address, size_t size,
                            void* user_data) {
  (void)entering;
  (void)address;
  (void)size;
  (void)user_data;
}

struct HostState {
  Notification entered;
  Notification release;
  std::atomic<int> probe_result{hipErrorUnknown};
  ProbeAcceptedAccessFn probe = nullptr;
  void* ptr = nullptr;
  size_t size = 0;
  int device = 0;
};

struct CallbackReentryState {
  int target_phase = 16;
  void* containing_ptr = nullptr;
  size_t containing_range = 0;
  std::atomic<bool> entered{false};
  std::atomic<int> match_count{0};
  std::atomic<uintptr_t> matched_object{0};
  std::atomic<int> init_result{hipErrorUnknown};
  std::atomic<int> deinit_result{hipErrorUnknown};
  std::atomic<int> access_result{hipErrorUnknown};
  bool probe_pool_apis = false;
  std::atomic<int> pool_trim_result{hipErrorUnknown};
  std::atomic<int> pool_destroy_result{hipErrorUnknown};
  std::atomic<int> pool_allocate_result{hipErrorUnknown};
  HipInitFn init = nullptr;
  HipHALDeinitFn deinit = nullptr;
  HipMemGetAccessFn get_access = nullptr;
  HipMemPoolTrimToFn pool_trim = nullptr;
  HipMemPoolDestroyFn pool_destroy = nullptr;
  HipMallocFromPoolAsyncFn malloc_from_pool_async = nullptr;
  hipMemPool_t pool = nullptr;
  void* ptr = nullptr;
  hipMemLocation location = {};
};

void ObserveCallbackReentry(int phase, void* object, void* user_data) {
  auto* state = static_cast<CallbackReentryState*>(user_data);
  if (phase != state->target_phase) return;
  if (state->containing_ptr) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(object);
    const uintptr_t ptr = reinterpret_cast<uintptr_t>(state->containing_ptr);
    if (ptr < base || ptr - base >= state->containing_range) return;
  }
  const int prior_match_count =
      state->match_count.fetch_add(1, std::memory_order_acq_rel);
  if (prior_match_count != 0) return;
  state->matched_object.store(reinterpret_cast<uintptr_t>(object),
                              std::memory_order_release);
  unsigned long long flags = ~0ull;
  state->init_result.store(state->init(0), std::memory_order_release);
  state->deinit_result.store(state->deinit(), std::memory_order_release);
  state->access_result.store(
      state->get_access(&flags, &state->location, state->ptr),
      std::memory_order_release);
  if (state->probe_pool_apis) {
    state->pool_trim_result.store(state->pool_trim(state->pool, 0),
                                  std::memory_order_release);
    state->pool_destroy_result.store(state->pool_destroy(state->pool),
                                     std::memory_order_release);
    void* nested_allocation = nullptr;
    state->pool_allocate_result.store(
        state->malloc_from_pool_async(&nested_allocation, /*size=*/4096,
                                      state->pool, /*stream=*/nullptr),
        std::memory_order_release);
  }
  state->entered.store(true, std::memory_order_release);
}

void ExpectCallbackReentryRejected(const CallbackReentryState& state) {
  EXPECT_GE(state.match_count.load(std::memory_order_acquire), 1);
  ASSERT_TRUE(state.entered.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorNotInitialized,
            static_cast<hipError_t>(
                state.init_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipErrorNotInitialized,
            static_cast<hipError_t>(
                state.deinit_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipErrorNotInitialized,
            static_cast<hipError_t>(
                state.access_result.load(std::memory_order_acquire)));
}

void ExpectSingleCallbackReentryRejected(const CallbackReentryState& state) {
  ExpectCallbackReentryRejected(state);
  EXPECT_EQ(1, state.match_count.load(std::memory_order_acquire));
}

void ExpectPoolCallbackReentryRejected(const CallbackReentryState& state) {
  EXPECT_EQ(hipErrorNotInitialized,
            static_cast<hipError_t>(
                state.pool_trim_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipErrorNotInitialized,
            static_cast<hipError_t>(
                state.pool_destroy_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipErrorNotInitialized,
            static_cast<hipError_t>(
                state.pool_allocate_result.load(std::memory_order_acquire)));
}

void ExpectSingleContainingAddressFreeCallback(
    const CallbackReentryState& state, const void* ptr, size_t range) {
  ExpectSingleCallbackReentryRejected(state);
  const uintptr_t base = state.matched_object.load(std::memory_order_acquire);
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  ASSERT_NE(0u, base);
  ASSERT_GE(address, base);
  EXPECT_LT(address - base, range);
}

void AcceptedHostCallback(void* user_data) {
  auto* state = static_cast<HostState*>(user_data);
  state->entered.Notify();
  state->release.Wait();
  state->probe_result.store(
      state->probe(state->ptr, state->size, state->device),
      std::memory_order_release);
}

struct MixedResetHostState {
  // Signals that the graph host callback is running on device B's queue.
  Notification entered;
  // Allows the callback to probe the target-owned first mapping.
  Notification release_probe;
  // Signals that the callback has published the probe result.
  Notification probe_complete;
  // Keeps the callback, and therefore the following memset, queued until the
  // reset reaches incident-context quiescence.
  Notification release_return;
  // Result of probing the access accepted before writer admission closed.
  std::atomic<int> probe_result{hipErrorUnknown};
  // Instrumented access probe loaded from the test DSO.
  ProbeAcceptedAccessFn probe = nullptr;
  // Target-owned mapping covered by device B's accepted access capability.
  void* ptr = nullptr;
  // Number of bytes the queued operation will access.
  size_t size = 0;
  // Device whose access capability must resolve the mapping.
  int device = 0;
};

void MixedResetHostCallback(void* user_data) {
  auto* state = static_cast<MixedResetHostState*>(user_data);
  state->entered.Notify();
  state->release_probe.Wait();
  state->probe_result.store(
      state->probe(state->ptr, state->size, state->device),
      std::memory_order_release);
  state->probe_complete.Notify();
  state->release_return.Wait();
}

struct BlockingHostState {
  Notification entered;
  Notification release;
};

void BlockingHostCallback(void* user_data) {
  auto* state = static_cast<BlockingHostState*>(user_data);
  state->entered.Notify();
  state->release.Wait();
}

class HipVmmLifecyclePhaseTest : public testing::Test {
 protected:
  struct MappedVmm {
    int device = -1;
    size_t size = 0;
    void* reservation = nullptr;
    hipMemGenericAllocationHandle_t allocation = nullptr;
  };

  static void SetUpTestSuite() {
    const char* path = std::getenv("HRX_TEST_LIBAMDHIP64_PHASE");
    ASSERT_NE(path, nullptr);
    api_.library = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    ASSERT_NE(api_.library, nullptr) << dlerror();
#define RESOLVE(name, field) \
  api_.field = Resolve<decltype(api_.field)>(api_.library, name)
    RESOLVE("hipInit", init);
    RESOLVE("hipHALDeinit", hal_deinit);
    RESOLVE("hipGetDevice", get_device);
    RESOLVE("hipGetDeviceCount", get_device_count);
    RESOLVE("hipDeviceReset", device_reset);
    RESOLVE("hipDevicePrimaryCtxRelease", device_primary_ctx_release);
    RESOLVE("hipDevicePrimaryCtxReset", device_primary_ctx_reset);
    RESOLVE("hipDevicePrimaryCtxRetain", device_primary_ctx_retain);
    RESOLVE("hipCtxCreate", ctx_create);
    RESOLVE("hipCtxDestroy", ctx_destroy);
    RESOLVE("hipCtxGetCurrent", ctx_get_current);
    RESOLVE("hipCtxSetCurrent", ctx_set_current);
    RESOLVE("hipCtxPushCurrent", ctx_push_current);
    RESOLVE("hipCtxPopCurrent", ctx_pop_current);
    RESOLVE("hipDeviceGetDevResource", device_get_resource);
    RESOLVE("hipDevResourceGenerateDesc", generate_descriptor);
    RESOLVE("hipGreenCtxCreate", green_ctx_create);
    RESOLVE("hipExecutionCtxDestroy", execution_ctx_destroy);
    RESOLVE("hipSetDevice", set_device);
    RESOLVE("hipDeviceGetAttribute", device_get_attribute);
    RESOLVE("hipMemGetAllocationGranularity", mem_get_allocation_granularity);
    RESOLVE("hipMemAddressReserve", mem_address_reserve);
    RESOLVE("hipMemAddressFree", mem_address_free);
    RESOLVE("hipMemCreate", mem_create);
    RESOLVE("hipMemRelease", mem_release);
    RESOLVE("hipMemMap", mem_map);
    RESOLVE("hipMemUnmap", mem_unmap);
    RESOLVE("hipMemSetAccess", mem_set_access);
    RESOLVE("hipMemGetAccess", mem_get_access);
    RESOLVE("hipMemcpy", memcpy);
    RESOLVE("hipMemGetInfo", mem_get_info);
    RESOLVE("hipMemPoolCreate", mem_pool_create);
    RESOLVE("hipMemPoolDestroy", mem_pool_destroy);
    RESOLVE("hipMemPoolTrimTo", mem_pool_trim_to);
    RESOLVE("hipMemPoolSetAttribute", mem_pool_set_attribute);
    RESOLVE("hipDeviceGetDefaultMemPool", device_get_default_mem_pool);
    RESOLVE("hipMalloc", malloc);
    RESOLVE("hipFree", free);
    RESOLVE("hipMallocAsync", malloc_async);
    RESOLVE("hipMallocFromPoolAsync", malloc_from_pool_async);
    RESOLVE("hipFreeAsync", free_async);
    RESOLVE("hipStreamCreate", stream_create);
    RESOLVE("hipStreamDestroy", stream_destroy);
    RESOLVE("hipStreamSynchronize", stream_synchronize);
    RESOLVE("hipEventCreateWithFlags", event_create_with_flags);
    RESOLVE("hipEventDestroy", event_destroy);
    RESOLVE("hipGraphCreate", graph_create);
    RESOLVE("hipGraphDestroy", graph_destroy);
    RESOLVE("hipGraphAddHostNode", graph_add_host_node);
    RESOLVE("hipGraphAddMemcpyNode", graph_add_memcpy_node);
    RESOLVE("hipGraphAddMemcpyNode1D", graph_add_memcpy_node_1d);
    RESOLVE("hipGraphAddMemsetNode", graph_add_memset_node);
    RESOLVE("hipGraphAddMemAllocNode", graph_add_mem_alloc_node);
    RESOLVE("hipGraphAddMemFreeNode", graph_add_mem_free_node);
    RESOLVE("hipGraphAddEventRecordNode", graph_add_event_record_node);
    RESOLVE("hipGraphGetNodes", graph_get_nodes);
    RESOLVE("hipGraphGetEdges", graph_get_edges);
    RESOLVE("hipGraphGetRootNodes", graph_get_root_nodes);
    RESOLVE("hipGraphInstantiate", graph_instantiate);
    RESOLVE("hipGraphLaunch", graph_launch);
    RESOLVE("hipGraphExecUpdate", graph_exec_update);
    RESOLVE("hipGraphExecMemcpyNodeSetParams1D",
            graph_exec_memcpy_node_set_params_1d);
    RESOLVE("hipGraphExecDestroy", graph_exec_destroy);
    RESOLVE("iree_hip_vmm_test_set_phase_observer", set_phase_observer);
    RESOLVE("iree_hip_vmm_test_wait_phase_observer_closing",
            wait_phase_observer_closing);
    RESOLVE("iree_hip_vmm_test_notify_phase_for_test", notify_phase_for_test);
    RESOLVE("iree_hip_vmm_test_set_lower_address_free_observer",
            set_lower_address_free_observer);
    RESOLVE("iree_hip_vmm_test_clear_lower_address_free_observer",
            clear_lower_address_free_observer);
    RESOLVE("iree_hip_vmm_test_fail_next_lower_observer_clear",
            fail_next_lower_observer_clear);
    RESOLVE("iree_hip_vmm_test_probe_accepted_access", probe_accepted_access);
    RESOLVE("iree_hip_vmm_test_fail_green_context_after_retain_once",
            fail_green_context_after_retain_once);
    RESOLVE("iree_hip_vmm_test_fail_lower_virtual_reserve_after_native_once",
            fail_lower_virtual_reserve_after_native_once);
    RESOLVE("iree_hip_vmm_test_fail_hrx_virtual_reserve_after_native_once",
            fail_hrx_virtual_reserve_after_native_once);
    RESOLVE("iree_hip_vmm_test_last_virtual_reserve_address",
            last_virtual_reserve_address);
    RESOLVE("iree_hip_vmm_test_last_virtual_reserve_alignment",
            last_virtual_reserve_alignment);
    RESOLVE("iree_hip_vmm_test_fail_address_reserve_after_native_once",
            fail_address_reserve_after_native_once);
    RESOLVE("iree_hip_vmm_test_fail_physical_create_after_native_once",
            fail_physical_create_after_native_once);
    RESOLVE("iree_hip_vmm_test_fail_cleanup_count", fail_cleanup_count);
    RESOLVE("iree_hip_vmm_test_quarantine_count", quarantine_count);
    RESOLVE("iree_hip_vmm_test_cleanup_attempt_count", cleanup_attempt_count);
    RESOLVE("iree_hip_vmm_test_last_cleanup_status", last_cleanup_status);
    RESOLVE("iree_hip_vmm_test_arm_quarantine_drain_pause",
            arm_quarantine_drain_pause);
    RESOLVE("iree_hip_vmm_test_wait_quarantine_drain_paused",
            wait_quarantine_drain_paused);
    RESOLVE("iree_hip_vmm_test_release_quarantine_drain_pause",
            release_quarantine_drain_pause);
    RESOLVE("iree_hip_vmm_test_quarantine_drain_attempt_kind",
            quarantine_drain_attempt_kind);
    RESOLVE("iree_hip_vmm_test_quarantine_drain_without_lifecycle_writer",
            quarantine_drain_without_writer);
    RESOLVE("iree_hip_vmm_test_fail_plan_allocation_at",
            fail_plan_allocation_at);
    RESOLVE("iree_hip_vmm_test_fail_plan_revalidation_once",
            fail_plan_revalidation_once);
    RESOLVE("iree_hip_vmm_test_fail_native_operation_at",
            fail_native_operation_at);
    RESOLVE("iree_hip_vmm_test_native_operation_count", native_operation_count);
    RESOLVE("iree_hip_vmm_test_native_operation_cursor",
            native_operation_cursor);
    RESOLVE("iree_hip_vmm_test_native_operation_attempt_count",
            native_operation_attempt_count);
    RESOLVE("iree_hip_vmm_test_first_native_error", first_native_error);
    RESOLVE("iree_hip_vmm_test_raw_native_call_count", raw_native_call_count);
    RESOLVE("iree_hip_vmm_test_query_allocation_registry",
            query_allocation_registry);
    RESOLVE("iree_hip_vmm_test_fail_access_preflight_at",
            fail_access_preflight_at);
    RESOLVE("iree_hip_vmm_test_fail_graph_exec_memcpy_prepare_once",
            fail_graph_exec_memcpy_prepare_once);
    RESOLVE("iree_hip_vmm_test_fail_graph_exec_rebuild_once",
            fail_graph_exec_rebuild_once);
    RESOLVE("iree_hip_vmm_test_fail_graph_add_at", fail_graph_add_at);
    RESOLVE("iree_hip_vmm_test_set_graph_peer_transfer_failure",
            set_graph_peer_transfer_failure);
    RESOLVE("iree_hip_vmm_test_graph_internal_node_count",
            graph_internal_node_count);
    RESOLVE("iree_hip_vmm_test_graph_internal_root_count",
            graph_internal_root_count);
    RESOLVE("iree_hip_vmm_test_graph_internal_dependency_count",
            graph_internal_dependency_count);
    RESOLVE("iree_hip_vmm_test_graph_owned_host_allocation_count",
            graph_owned_host_allocation_count);
    RESOLVE("iree_hip_vmm_test_graph_user_object_reference_count",
            graph_user_object_reference_count);
    RESOLVE("iree_hip_vmm_test_context_reference_count",
            context_reference_count);
    RESOLVE("iree_hip_vmm_test_pointer_buffer_reference_count",
            pointer_buffer_reference_count);
    RESOLVE("iree_hip_vmm_test_reset_observability", reset_observability);
    RESOLVE("iree_hip_vmm_test_set_epoch_exhaustion", set_epoch_exhaustion);
    RESOLVE("iree_hip_vmm_test_install_graph_exec_allocator",
            install_graph_exec_allocator);
    RESOLVE("iree_hip_vmm_test_fail_graph_exec_allocation_at",
            fail_graph_exec_allocation_at);
    RESOLVE("iree_hal_streaming_tls_test_inject_failures", inject_tls_failures);
#undef RESOLVE
    ASSERT_NE(api_.init, nullptr);
    ASSERT_NE(api_.hal_deinit, nullptr);
    ASSERT_NE(api_.get_device, nullptr);
    ASSERT_NE(api_.get_device_count, nullptr);
    ASSERT_NE(api_.device_reset, nullptr);
    ASSERT_NE(api_.device_primary_ctx_release, nullptr);
    ASSERT_NE(api_.device_primary_ctx_reset, nullptr);
    ASSERT_NE(api_.device_primary_ctx_retain, nullptr);
    ASSERT_NE(api_.ctx_create, nullptr);
    ASSERT_NE(api_.ctx_destroy, nullptr);
    ASSERT_NE(api_.ctx_get_current, nullptr);
    ASSERT_NE(api_.ctx_set_current, nullptr);
    ASSERT_NE(api_.ctx_push_current, nullptr);
    ASSERT_NE(api_.ctx_pop_current, nullptr);
    ASSERT_NE(api_.device_get_resource, nullptr);
    ASSERT_NE(api_.generate_descriptor, nullptr);
    ASSERT_NE(api_.green_ctx_create, nullptr);
    ASSERT_NE(api_.execution_ctx_destroy, nullptr);
    ASSERT_NE(api_.set_device, nullptr);
    ASSERT_NE(api_.device_get_attribute, nullptr);
    ASSERT_NE(api_.mem_get_allocation_granularity, nullptr);
    ASSERT_NE(api_.mem_address_reserve, nullptr);
    ASSERT_NE(api_.mem_address_free, nullptr);
    ASSERT_NE(api_.mem_create, nullptr);
    ASSERT_NE(api_.mem_release, nullptr);
    ASSERT_NE(api_.mem_map, nullptr);
    ASSERT_NE(api_.mem_unmap, nullptr);
    ASSERT_NE(api_.mem_set_access, nullptr);
    ASSERT_NE(api_.mem_get_access, nullptr);
    ASSERT_NE(api_.memcpy, nullptr);
    ASSERT_NE(api_.mem_get_info, nullptr);
    ASSERT_NE(api_.mem_pool_create, nullptr);
    ASSERT_NE(api_.mem_pool_destroy, nullptr);
    ASSERT_NE(api_.mem_pool_trim_to, nullptr);
    ASSERT_NE(api_.mem_pool_set_attribute, nullptr);
    ASSERT_NE(api_.device_get_default_mem_pool, nullptr);
    ASSERT_NE(api_.malloc, nullptr);
    ASSERT_NE(api_.free, nullptr);
    ASSERT_NE(api_.malloc_async, nullptr);
    ASSERT_NE(api_.malloc_from_pool_async, nullptr);
    ASSERT_NE(api_.free_async, nullptr);
    ASSERT_NE(api_.stream_create, nullptr);
    ASSERT_NE(api_.stream_destroy, nullptr);
    ASSERT_NE(api_.stream_synchronize, nullptr);
    ASSERT_NE(api_.event_create_with_flags, nullptr);
    ASSERT_NE(api_.event_destroy, nullptr);
    ASSERT_NE(api_.graph_create, nullptr);
    ASSERT_NE(api_.graph_destroy, nullptr);
    ASSERT_NE(api_.graph_add_host_node, nullptr);
    ASSERT_NE(api_.graph_add_memcpy_node, nullptr);
    ASSERT_NE(api_.graph_add_memcpy_node_1d, nullptr);
    ASSERT_NE(api_.graph_add_memset_node, nullptr);
    ASSERT_NE(api_.graph_add_mem_alloc_node, nullptr);
    ASSERT_NE(api_.graph_add_mem_free_node, nullptr);
    ASSERT_NE(api_.graph_add_event_record_node, nullptr);
    ASSERT_NE(api_.graph_get_nodes, nullptr);
    ASSERT_NE(api_.graph_get_edges, nullptr);
    ASSERT_NE(api_.graph_get_root_nodes, nullptr);
    ASSERT_NE(api_.graph_instantiate, nullptr);
    ASSERT_NE(api_.graph_launch, nullptr);
    ASSERT_NE(api_.graph_exec_update, nullptr);
    ASSERT_NE(api_.graph_exec_memcpy_node_set_params_1d, nullptr);
    ASSERT_NE(api_.graph_exec_destroy, nullptr);
    ASSERT_NE(api_.set_phase_observer, nullptr);
    ASSERT_NE(api_.wait_phase_observer_closing, nullptr);
    ASSERT_NE(api_.notify_phase_for_test, nullptr);
    ASSERT_NE(api_.set_lower_address_free_observer, nullptr);
    ASSERT_NE(api_.clear_lower_address_free_observer, nullptr);
    ASSERT_NE(api_.fail_next_lower_observer_clear, nullptr);
    ASSERT_NE(api_.probe_accepted_access, nullptr);
    ASSERT_NE(api_.fail_green_context_after_retain_once, nullptr);
    ASSERT_NE(api_.fail_lower_virtual_reserve_after_native_once, nullptr);
    ASSERT_NE(api_.fail_hrx_virtual_reserve_after_native_once, nullptr);
    ASSERT_NE(api_.last_virtual_reserve_address, nullptr);
    ASSERT_NE(api_.last_virtual_reserve_alignment, nullptr);
    ASSERT_NE(api_.fail_address_reserve_after_native_once, nullptr);
    ASSERT_NE(api_.fail_physical_create_after_native_once, nullptr);
    ASSERT_NE(api_.fail_cleanup_count, nullptr);
    ASSERT_NE(api_.quarantine_count, nullptr);
    ASSERT_NE(api_.cleanup_attempt_count, nullptr);
    ASSERT_NE(api_.last_cleanup_status, nullptr);
    ASSERT_NE(api_.arm_quarantine_drain_pause, nullptr);
    ASSERT_NE(api_.wait_quarantine_drain_paused, nullptr);
    ASSERT_NE(api_.release_quarantine_drain_pause, nullptr);
    ASSERT_NE(api_.quarantine_drain_attempt_kind, nullptr);
    ASSERT_NE(api_.quarantine_drain_without_writer, nullptr);
    ASSERT_NE(api_.fail_plan_allocation_at, nullptr);
    ASSERT_NE(api_.fail_plan_revalidation_once, nullptr);
    ASSERT_NE(api_.fail_native_operation_at, nullptr);
    ASSERT_NE(api_.native_operation_count, nullptr);
    ASSERT_NE(api_.native_operation_cursor, nullptr);
    ASSERT_NE(api_.native_operation_attempt_count, nullptr);
    ASSERT_NE(api_.first_native_error, nullptr);
    ASSERT_NE(api_.raw_native_call_count, nullptr);
    ASSERT_NE(api_.query_allocation_registry, nullptr);
    ASSERT_NE(api_.fail_access_preflight_at, nullptr);
    ASSERT_NE(api_.fail_graph_exec_memcpy_prepare_once, nullptr);
    ASSERT_NE(api_.fail_graph_exec_rebuild_once, nullptr);
    ASSERT_NE(api_.fail_graph_add_at, nullptr);
    ASSERT_NE(api_.set_graph_peer_transfer_failure, nullptr);
    ASSERT_NE(api_.graph_internal_node_count, nullptr);
    ASSERT_NE(api_.graph_internal_root_count, nullptr);
    ASSERT_NE(api_.graph_internal_dependency_count, nullptr);
    ASSERT_NE(api_.graph_owned_host_allocation_count, nullptr);
    ASSERT_NE(api_.graph_user_object_reference_count, nullptr);
    ASSERT_NE(api_.context_reference_count, nullptr);
    ASSERT_NE(api_.pointer_buffer_reference_count, nullptr);
    ASSERT_NE(api_.reset_observability, nullptr);
    ASSERT_NE(api_.set_epoch_exhaustion, nullptr);
    ASSERT_NE(api_.install_graph_exec_allocator, nullptr);
    ASSERT_NE(api_.fail_graph_exec_allocation_at, nullptr);
    ASSERT_NE(api_.inject_tls_failures, nullptr);
  }

  void SetUp() override {
    ASSERT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));
    api_.inject_tls_failures(/*failures=*/0);
    api_.reset_observability();
  }

  bool IsVmmSupported(int device) {
    int supported = 0;
    EXPECT_EQ(hipSuccess,
              api_.device_get_attribute(
                  &supported,
                  hipDeviceAttributeVirtualMemoryManagementSupported, device));
    return supported != 0;
  }

  size_t MinimumGranularity(int device) {
    hipMemAllocationProp properties = {};
    properties.type = hipMemAllocationTypePinned;
    properties.requestedHandleType = hipMemHandleTypeNone;
    properties.location = {hipMemLocationTypeDevice, device};
    size_t granularity = 0;
    EXPECT_EQ(hipSuccess, api_.mem_get_allocation_granularity(
                              &granularity, &properties,
                              hipMemAllocationGranularityMinimum));
    return granularity;
  }

  void CreateMappedVmm(int device, size_t page_count, bool grant_access,
                       MappedVmm* out_vmm) {
    ASSERT_NE(out_vmm, nullptr);
    ASSERT_GT(page_count, 0u);
    ASSERT_EQ(hipSuccess, api_.set_device(device));
    const size_t granularity = MinimumGranularity(device);
    ASSERT_NE(granularity, 0u);
    ASSERT_LE(page_count, SIZE_MAX / granularity);
    const size_t size = page_count * granularity;
    hipMemAllocationProp properties = {};
    properties.type = hipMemAllocationTypePinned;
    properties.requestedHandleType = hipMemHandleTypeNone;
    properties.location = {hipMemLocationTypeDevice, device};
    void* reservation = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
    hipMemGenericAllocationHandle_t allocation = nullptr;
    ASSERT_EQ(hipSuccess, api_.mem_create(&allocation, size, &properties, 0));
    ASSERT_EQ(hipSuccess, api_.mem_map(reservation, size, 0, allocation, 0));
    if (grant_access) {
      hipMemAccessDesc descriptor = {};
      descriptor.location = {hipMemLocationTypeDevice, device};
      descriptor.flags = hipMemAccessFlagsProtReadWrite;
      ASSERT_EQ(hipSuccess,
                api_.mem_set_access(reservation, size, &descriptor, 1));
    }
    *out_vmm = {device, size, reservation, allocation};
  }

  void DestroyMappedVmm(const MappedVmm& vmm) {
    ASSERT_EQ(hipSuccess, api_.set_device(vmm.device));
    ASSERT_EQ(hipSuccess, api_.mem_unmap(vmm.reservation, vmm.size));
    ASSERT_EQ(hipSuccess, api_.mem_release(vmm.allocation));
    ASSERT_EQ(hipSuccess, api_.mem_address_free(vmm.reservation, vmm.size));
  }

  void ExpectEmptyStagedGraph(hipGraph_t graph) {
    size_t count = 1;
    EXPECT_EQ(hipSuccess, api_.graph_get_nodes(graph, nullptr, &count));
    EXPECT_EQ(0u, count);
    count = 1;
    EXPECT_EQ(hipSuccess, api_.graph_get_root_nodes(graph, nullptr, &count));
    EXPECT_EQ(0u, count);
    count = 1;
    EXPECT_EQ(hipSuccess,
              api_.graph_get_edges(graph, nullptr, nullptr, &count));
    EXPECT_EQ(0u, count);
    EXPECT_EQ(0u, api_.graph_internal_node_count(graph));
    EXPECT_EQ(0u, api_.graph_internal_root_count(graph));
    EXPECT_EQ(0u, api_.graph_internal_dependency_count(graph));
    EXPECT_EQ(0u, api_.graph_owned_host_allocation_count(graph));
    EXPECT_EQ(0u, api_.graph_user_object_reference_count(graph));
  }

  template <typename AddFn>
  void ExpectStagedAddFailureAtomic(
      hipGraph_t graph, int checkpoint_count,
      uint64_t expected_committed_user_reference_count,
      hipCtx_t observed_context_a, hipCtx_t observed_context_b,
      void* observed_pointer_a, void* observed_pointer_b, AddFn add,
      hipGraphNode_t* out_node) {
    ASSERT_NE(nullptr, graph);
    ASSERT_GT(checkpoint_count, 0);
    ASSERT_NE(nullptr, out_node);
    *out_node = nullptr;

    const uint64_t context_a_reference_count =
        observed_context_a ? api_.context_reference_count(observed_context_a)
                           : 0;
    const uint64_t context_b_reference_count =
        observed_context_b ? api_.context_reference_count(observed_context_b)
                           : 0;
    const uint64_t pointer_a_reference_count =
        observed_pointer_a
            ? api_.pointer_buffer_reference_count(observed_pointer_a)
            : 0;
    const uint64_t pointer_b_reference_count =
        observed_pointer_b
            ? api_.pointer_buffer_reference_count(observed_pointer_b)
            : 0;

    for (int ordinal = 1; ordinal <= checkpoint_count; ++ordinal) {
      SCOPED_TRACE(testing::Message() << "failure ordinal " << ordinal);
      hipGraphNode_t node = reinterpret_cast<hipGraphNode_t>(uintptr_t{0x1234});
      api_.fail_graph_add_at(ordinal);
      ASSERT_EQ(hipErrorOutOfMemory, add(&node));
      EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{0x1234}), node);
      ExpectEmptyStagedGraph(graph);
      if (observed_context_a) {
        EXPECT_EQ(context_a_reference_count,
                  api_.context_reference_count(observed_context_a));
      }
      if (observed_context_b) {
        EXPECT_EQ(context_b_reference_count,
                  api_.context_reference_count(observed_context_b));
      }
      if (observed_pointer_a) {
        EXPECT_EQ(pointer_a_reference_count,
                  api_.pointer_buffer_reference_count(observed_pointer_a));
      }
      if (observed_pointer_b) {
        EXPECT_EQ(pointer_b_reference_count,
                  api_.pointer_buffer_reference_count(observed_pointer_b));
      }

      hipGraphExec_t empty_exec = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_instantiate(&empty_exec, graph, nullptr,
                                                   nullptr, 0));
      ASSERT_NE(nullptr, empty_exec);
      ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(empty_exec));
    }

    api_.fail_graph_add_at(checkpoint_count + 1);
    ASSERT_EQ(hipSuccess, add(out_node));
    ASSERT_NE(nullptr, *out_node);
    size_t count = 0;
    ASSERT_EQ(hipSuccess, api_.graph_get_nodes(graph, nullptr, &count));
    EXPECT_EQ(1u, count);
    ASSERT_EQ(hipSuccess, api_.graph_get_root_nodes(graph, nullptr, &count));
    EXPECT_EQ(1u, count);
    ASSERT_EQ(hipSuccess,
              api_.graph_get_edges(graph, nullptr, nullptr, &count));
    EXPECT_EQ(0u, count);
    EXPECT_EQ(2u, api_.graph_internal_node_count(graph));
    EXPECT_EQ(1u, api_.graph_internal_root_count(graph));
    EXPECT_EQ(1u, api_.graph_internal_dependency_count(graph));
    EXPECT_EQ(1u, api_.graph_owned_host_allocation_count(graph));
    EXPECT_EQ(expected_committed_user_reference_count,
              api_.graph_user_object_reference_count(graph));
  }

  void CreateGraphMemoryHostGraph(int device, BlockingHostState* host_state,
                                  hipGraph_t* out_graph,
                                  hipEvent_t* tail_events = nullptr,
                                  size_t tail_event_count = 0) {
    ASSERT_NE(host_state, nullptr);
    ASSERT_NE(out_graph, nullptr);
    *out_graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(out_graph, /*flags=*/0));

    hipMemAllocNodeParams allocation_parameters = {};
    allocation_parameters.poolProps.allocType = hipMemAllocationTypePinned;
    allocation_parameters.poolProps.location = {hipMemLocationTypeDevice,
                                                device};
    allocation_parameters.bytesize = 4096;
    hipGraphNode_t allocation_node = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_add_mem_alloc_node(
                  &allocation_node, *out_graph, /*dependencies=*/nullptr,
                  /*dependency_count=*/0, &allocation_parameters));
    ASSERT_NE(allocation_parameters.dptr, nullptr);

    hipHostNodeParams host_parameters = {BlockingHostCallback, host_state};
    hipGraphNode_t host_node = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_add_host_node(&host_node, *out_graph, &allocation_node,
                                       1, &host_parameters));
    hipGraphNode_t tail = host_node;
    for (size_t i = 0; i < tail_event_count; ++i) {
      ASSERT_NE(tail_events, nullptr);
      ASSERT_EQ(hipSuccess, api_.event_create_with_flags(
                                &tail_events[i], hipEventDisableTiming));
      hipGraphNode_t event_node = nullptr;
      ASSERT_EQ(hipSuccess,
                api_.graph_add_event_record_node(&event_node, *out_graph, &tail,
                                                 1, tail_events[i]));
      tail = event_node;
    }
    hipGraphNode_t free_node = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_add_mem_free_node(&free_node, *out_graph, &tail, 1,
                                           allocation_parameters.dptr));
  }

  void ExpectDeinitWinsBeforeDeviceWriter(
      hipError_t (*operation)(hipDevice_t)) {
    ASSERT_EQ(hipSuccess, api_.init(0));
    int device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&device));

    PhaseState phase_state;
    phase_state.release_writer.Notify();
    phase_state.release_pre_drain.Notify();
    ASSERT_EQ(hipSuccess, api_.set_phase_observer(ObservePhase, &phase_state));
    std::atomic<int> operation_result{hipErrorUnknown};
    std::thread worker([&] {
      operation_result.store(operation(device), std::memory_order_release);
    });
    phase_state.before_device_writer.Wait();

    const hipError_t deinit_result = api_.hal_deinit();
    phase_state.release_before_device_writer.Notify();
    worker.join();
    ASSERT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));

    ASSERT_EQ(hipSuccess, deinit_result);
    EXPECT_EQ(hipErrorNotInitialized,
              static_cast<hipError_t>(
                  operation_result.load(std::memory_order_acquire)));
    ASSERT_EQ(hipSuccess, api_.init(0));
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
  }

  void ExpectFailedResetCleanupAllowsExplicitTeardown(bool whole_device) {
    ASSERT_EQ(hipSuccess, api_.init(0));
    int device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&device));
    int supported = 0;
    ASSERT_EQ(hipSuccess,
              api_.device_get_attribute(
                  &supported,
                  hipDeviceAttributeVirtualMemoryManagementSupported, device));
    if (!supported) {
      EXPECT_EQ(hipSuccess, api_.hal_deinit());
      GTEST_SKIP() << "device does not support HIP VMM";
    }

    hipCtx_t explicit_context = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.ctx_create(&explicit_context, /*flags=*/0, device));
    ASSERT_NE(nullptr, explicit_context);
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
    ASSERT_EQ(hipSuccess, api_.set_device(device));

    hipCtx_t stale_primary = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.device_primary_ctx_retain(&stale_primary, device));
    ASSERT_NE(nullptr, stale_primary);

    hipMemAllocationProp properties = {};
    properties.type = hipMemAllocationTypePinned;
    properties.requestedHandleType = hipMemHandleTypeNone;
    properties.location = {hipMemLocationTypeDevice, device};
    size_t granularity = 0;
    ASSERT_EQ(hipSuccess, api_.mem_get_allocation_granularity(
                              &granularity, &properties,
                              hipMemAllocationGranularityMinimum));
    void* reservation = nullptr;
    ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&reservation, granularity, 0,
                                                   nullptr, 0));
    hipMemGenericAllocationHandle_t handle = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.mem_create(&handle, granularity, &properties, 0));
    ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));
    hipMemAccessDesc descriptor = {};
    descriptor.location = {hipMemLocationTypeDevice, device};
    descriptor.flags = hipMemAccessFlagsProtReadWrite;
    ASSERT_EQ(hipSuccess,
              api_.mem_set_access(reservation, granularity, &descriptor, 1));

    // Fail the first native operation after public reset commit. This makes
    // the old context epoch and all primary retains irreversibly consumed. An
    // exact surviving explicit context may resume the journal while destroying
    // itself; unknown and consumed primary handles must not advance it.
    api_.fail_native_operation_at(1);
    const hipError_t reset_result = whole_device
                                        ? api_.device_reset()
                                        : api_.device_primary_ctx_reset(device);
    EXPECT_EQ(hipErrorUnknown, reset_result);

    const hipCtx_t unknown_handle =
        reinterpret_cast<hipCtx_t>(uintptr_t{0x1234});
    ASSERT_EQ(0u, api_.native_operation_cursor());
    ASSERT_EQ(1u, api_.native_operation_attempt_count(1));
    EXPECT_EQ(hipErrorInvalidContext, api_.ctx_destroy(unknown_handle));
    EXPECT_EQ(0u, api_.native_operation_cursor());
    EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
    EXPECT_EQ(hipErrorInvalidContext, api_.ctx_destroy(stale_primary));
    EXPECT_EQ(0u, api_.native_operation_cursor());
    EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
    ASSERT_EQ(hipSuccess, api_.ctx_destroy(explicit_context));
    EXPECT_EQ(hipErrorInvalidContext, api_.ctx_destroy(explicit_context));

    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    ASSERT_EQ(hipSuccess, api_.init(0));
    EXPECT_EQ(hipErrorInvalidContext, api_.ctx_destroy(explicit_context));
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
  }

  static Api api_;
};

Api HipVmmLifecyclePhaseTest::api_;

TEST_F(HipVmmLifecyclePhaseTest,
       AddressReserveForwardsExactPlacementHintsToNativeCall) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  int vmm_supported = 0;
  ASSERT_EQ(hipSuccess,
            api_.device_get_attribute(
                &vmm_supported,
                hipDeviceAttributeVirtualMemoryManagementSupported, device));
  if (vmm_supported == 0) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypePinned;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location = {hipMemLocationTypeDevice, device};
  size_t minimum_granularity = 0;
  ASSERT_EQ(hipSuccess, api_.mem_get_allocation_granularity(
                            &minimum_granularity, &properties,
                            hipMemAllocationGranularityMinimum));
  size_t recommended_granularity = 0;
  ASSERT_EQ(hipSuccess, api_.mem_get_allocation_granularity(
                            &recommended_granularity, &properties,
                            hipMemAllocationGranularityRecommended));
  ASSERT_GT(minimum_granularity, 0u);
  ASSERT_EQ(0u, minimum_granularity & (minimum_granularity - 1));
  ASSERT_GT(recommended_granularity, 0u);
  ASSERT_EQ(0u, recommended_granularity & (recommended_granularity - 1));
  const size_t backend_default_alignment =
      minimum_granularity > recommended_granularity ? minimum_granularity
                                                    : recommended_granularity;
  ASSERT_LE(backend_default_alignment, SIZE_MAX / 2);
  const size_t explicit_alignment = backend_default_alignment * 2;
  ASSERT_GT(explicit_alignment, minimum_granularity);
  ASSERT_GT(explicit_alignment, recommended_granularity);
  ASSERT_EQ(0u, explicit_alignment & (explicit_alignment - 1));

  void* occupied = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&occupied, minimum_granularity,
                                     explicit_alignment, nullptr, /*flags=*/0));
  ASSERT_NE(nullptr, occupied);
  ASSERT_EQ(0u, reinterpret_cast<uintptr_t>(occupied) % explicit_alignment);

  api_.reset_observability();
  void* fallback = nullptr;
  const hipError_t reserve_result = api_.mem_address_reserve(
      &fallback, minimum_granularity, explicit_alignment, occupied,
      /*flags=*/0);
  EXPECT_EQ(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(occupied)),
            api_.last_virtual_reserve_address());
  EXPECT_EQ(static_cast<uint64_t>(explicit_alignment),
            api_.last_virtual_reserve_alignment());
  EXPECT_EQ(hipSuccess, reserve_result);
  if (reserve_result == hipSuccess) {
    EXPECT_NE(nullptr, fallback);
    EXPECT_NE(occupied, fallback);
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(fallback) % explicit_alignment);
    EXPECT_EQ(hipSuccess, api_.mem_address_free(fallback, minimum_granularity));
  }
  EXPECT_EQ(hipSuccess, api_.mem_address_free(occupied, minimum_granularity));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       ObserverClearWaitsForInFlightCallbackAndStopsNewCallbacks) {
  ObserverRegistryState state;
  state.target_phase = 101;
  state.block_callback = true;
  ASSERT_EQ(hipSuccess, api_.set_phase_observer(ObserveRegistryPhase, &state));

  std::thread notifier([&] { api_.notify_phase_for_test(state.target_phase); });
  state.entered.Wait();

  std::atomic<int> clear_result{hipErrorUnknown};
  std::atomic<bool> clear_returned{false};
  std::thread clearer([&] {
    clear_result.store(api_.set_phase_observer(nullptr, nullptr),
                       std::memory_order_release);
    clear_returned.store(true, std::memory_order_release);
  });
  ASSERT_EQ(hipSuccess, api_.wait_phase_observer_closing());
  EXPECT_FALSE(clear_returned.load(std::memory_order_acquire));

  state.release.Notify();
  notifier.join();
  clearer.join();
  EXPECT_EQ(hipSuccess, static_cast<hipError_t>(
                            clear_result.load(std::memory_order_acquire)));
  EXPECT_TRUE(clear_returned.load(std::memory_order_acquire));
  EXPECT_EQ(1, state.callback_count.load(std::memory_order_acquire));

  api_.notify_phase_for_test(state.target_phase);
  EXPECT_EQ(1, state.callback_count.load(std::memory_order_acquire));
}

TEST_F(HipVmmLifecyclePhaseTest, ObserverRejectsReplacementAndReentrantClear) {
  ObserverRegistryState state;
  state.target_phase = 102;
  state.setter = api_.set_phase_observer;
  state.clear_reentrantly = true;
  ASSERT_EQ(hipSuccess, api_.set_phase_observer(ObserveRegistryPhase, &state));
  EXPECT_EQ(hipErrorIllegalState,
            api_.set_phase_observer(ObserveRegistryPhase, &state));

  api_.notify_phase_for_test(state.target_phase);
  EXPECT_EQ(1, state.callback_count.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorIllegalState,
            static_cast<hipError_t>(
                state.reentrant_clear_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));
}

TEST_F(HipVmmLifecyclePhaseTest,
       ObserverPropagatesLowerInstallAndClearFailures) {
  int lower_user_data = 0;
  ASSERT_EQ(hipSuccess, api_.set_lower_address_free_observer(
                            IgnoreLowerAddressFree, &lower_user_data));

  ObserverRegistryState state;
  state.target_phase = 103;
  EXPECT_EQ(hipErrorIllegalState,
            api_.set_phase_observer(ObserveRegistryPhase, &state));
  ASSERT_EQ(hipSuccess, api_.clear_lower_address_free_observer(
                            IgnoreLowerAddressFree, &lower_user_data));

  ASSERT_EQ(hipSuccess, api_.set_phase_observer(ObserveRegistryPhase, &state));
  api_.fail_next_lower_observer_clear();
  EXPECT_EQ(hipErrorUnknown, api_.set_phase_observer(nullptr, nullptr));
  api_.notify_phase_for_test(state.target_phase);
  EXPECT_EQ(1, state.callback_count.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));
}

TEST_F(HipVmmLifecyclePhaseTest,
       AllocationReleaseReservesRollbackCapacityAcrossConcurrentCreate) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(0u, size);
  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypePinned;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location = {hipMemLocationTypeDevice, device};

  RegistryState state = QueryRegistryState(api_.query_allocation_registry);
  ASSERT_EQ(0u, state.count);
  ASSERT_EQ(0u, state.pending);
  std::vector<hipMemGenericAllocationHandle_t> allocations;
  if (state.capacity == 0) {
    hipMemGenericAllocationHandle_t allocation = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.mem_create(&allocation, size, &properties, /*flags=*/0));
    allocations.push_back(allocation);
    state = QueryRegistryState(api_.query_allocation_registry);
  }
  ASSERT_GT(state.capacity, 0u);
  while (state.count < state.capacity) {
    hipMemGenericAllocationHandle_t allocation = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.mem_create(&allocation, size, &properties, /*flags=*/0));
    allocations.push_back(allocation);
    state = QueryRegistryState(api_.query_allocation_registry);
    ASSERT_EQ(0u, state.pending);
  }
  ASSERT_EQ(state.capacity, state.count);
  ASSERT_EQ(state.count, allocations.size());
  const size_t full_capacity = state.capacity;

  api_.fail_cleanup_count(/*physical_memory=*/true, /*failure_count=*/1);
  ObserverRegistryState observer_state;
  observer_state.target_phase = kAllocationRetireReservedPhase;
  observer_state.block_callback = true;
  hipError_t release_result = hipErrorUnknown;
  hipMemGenericAllocationHandle_t concurrent_allocation = nullptr;
  {
    ScopedPhaseObserver observer(api_.set_phase_observer, ObserveRegistryPhase,
                                 &observer_state);
    std::thread release_thread(
        [&] { release_result = api_.mem_release(allocations.front()); });
    observer_state.entered.Wait();

    state = QueryRegistryState(api_.query_allocation_registry);
    EXPECT_EQ(full_capacity - 1, state.count);
    EXPECT_EQ(1u, state.pending);
    EXPECT_EQ(full_capacity, state.capacity);
    ExpectValidRegistryOccupancy(state);

    const hipError_t create_result =
        api_.mem_create(&concurrent_allocation, size, &properties, /*flags=*/0);
    EXPECT_EQ(hipSuccess, create_result);
    state = QueryRegistryState(api_.query_allocation_registry);
    EXPECT_EQ(full_capacity, state.count);
    EXPECT_EQ(1u, state.pending);
    EXPECT_GT(state.capacity, full_capacity);
    ExpectValidRegistryOccupancy(state);

    observer_state.release.Notify();
    release_thread.join();
  }

  EXPECT_EQ(hipErrorOutOfMemory, release_result);
  state = QueryRegistryState(api_.query_allocation_registry);
  EXPECT_EQ(full_capacity + 1, state.count);
  EXPECT_EQ(0u, state.pending);
  ExpectValidRegistryOccupancy(state);
  EXPECT_EQ(hipSuccess, api_.mem_release(allocations.front()));
  for (size_t i = 1; i < allocations.size(); ++i) {
    EXPECT_EQ(hipSuccess, api_.mem_release(allocations[i]));
  }
  if (concurrent_allocation) {
    EXPECT_EQ(hipSuccess, api_.mem_release(concurrent_allocation));
  }
  state = QueryRegistryState(api_.query_allocation_registry);
  EXPECT_EQ(0u, state.count);
  EXPECT_EQ(0u, state.pending);
  ExpectValidRegistryOccupancy(state);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       TlsInstallFailuresDoNotPublishContextOrAllocationOutputs) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  ASSERT_GT(device_count, 0);
  const int device = 0;

  const hipCtx_t sentinel_context =
      reinterpret_cast<hipCtx_t>(uintptr_t{0x1234});
  hipCtx_t context = sentinel_context;
  api_.inject_tls_failures(kTlsTestFailureSet);
  EXPECT_EQ(hipErrorNotReady, api_.ctx_create(&context, /*flags=*/0, device));
  EXPECT_EQ(sentinel_context, context);

  hipCtx_t current = sentinel_context;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  EXPECT_EQ(nullptr, current);

  void* allocation = reinterpret_cast<void*>(uintptr_t{0x5678});
  api_.inject_tls_failures(kTlsTestFailureSet);
  EXPECT_EQ(hipErrorNotReady, api_.malloc(&allocation, /*size=*/4096));
  EXPECT_EQ(nullptr, allocation);
  current = sentinel_context;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  EXPECT_EQ(nullptr, current);

  api_.inject_tls_failures(kTlsTestFailureSet);
  EXPECT_EQ(hipErrorNotReady, api_.set_device(device));
  current = sentinel_context;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  EXPECT_EQ(nullptr, current);
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));

  context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context, /*flags=*/0, device));
  ASSERT_NE(nullptr, context);
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
  const uint64_t reference_count = api_.context_reference_count(context);
  api_.inject_tls_failures(kTlsTestFailureSet);
  EXPECT_EQ(hipErrorNotReady, api_.ctx_set_current(context));
  EXPECT_EQ(reference_count, api_.context_reference_count(context));
  current = sentinel_context;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  EXPECT_EQ(nullptr, current);
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       TlsClearFailuresPreserveCurrentContextAndResetRetryability) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  hipCtx_t current = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  ASSERT_NE(nullptr, current);
  const uint64_t reference_count = api_.context_reference_count(current);
  api_.inject_tls_failures(kTlsTestFailureClear);
  EXPECT_EQ(hipErrorNotReady, api_.ctx_set_current(nullptr));
  hipCtx_t observed_current = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&observed_current));
  EXPECT_EQ(current, observed_current);
  EXPECT_EQ(reference_count, api_.context_reference_count(current));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));

  ASSERT_EQ(hipSuccess, api_.set_device(device));
  current = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  ASSERT_NE(nullptr, current);
  const uint64_t reset_reference_count = api_.context_reference_count(current);
  api_.inject_tls_failures(kTlsTestFailureClear);
  EXPECT_EQ(hipErrorNotReady, api_.device_reset());
  observed_current = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&observed_current));
  EXPECT_EQ(current, observed_current);
  EXPECT_EQ(reset_reference_count, api_.context_reference_count(current));
  ASSERT_EQ(hipSuccess, api_.device_reset());

  ASSERT_EQ(hipSuccess, api_.set_device(device));
  current = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  ASSERT_NE(nullptr, current);
  const uint64_t deinit_reference_count = api_.context_reference_count(current);
  api_.inject_tls_failures(kTlsTestFailureClear);
  EXPECT_EQ(hipErrorNotReady, api_.hal_deinit());
  observed_current = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&observed_current));
  EXPECT_EQ(current, observed_current);
  EXPECT_EQ(deinit_reference_count, api_.context_reference_count(current));
  int device_count = 0;
  EXPECT_EQ(hipSuccess, api_.get_device_count(&device_count));
  EXPECT_GT(device_count, 0);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       TlsClearFailureLeavesExplicitContextDestroyRetryable) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  hipCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context, /*flags=*/0, device));
  ASSERT_NE(nullptr, context);

  api_.inject_tls_failures(kTlsTestFailureClear);
  EXPECT_EQ(hipErrorNotReady, api_.ctx_destroy(context));
  EXPECT_EQ(hipSuccess, api_.ctx_destroy(context));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       RetiredCurrentDiscardPropagatesOneShotTlsClearFailure) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  // Preserve a second TLS owner on the stack, then retire its public context
  // while leaving that stack owner for hipCtxPopCurrent to discard.
  hipCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api_.ctx_push_current(context));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context));
  const uint64_t retired_reference_count =
      api_.context_reference_count(context);
  ASSERT_NE(0u, retired_reference_count);

  const hipCtx_t sentinel = reinterpret_cast<hipCtx_t>(uintptr_t{0x1234});
  hipCtx_t popped = sentinel;
  api_.inject_tls_failures(kTlsTestFailureClear);
  EXPECT_EQ(hipErrorNotReady, api_.ctx_pop_current(&popped));
  EXPECT_EQ(nullptr, popped);

  EXPECT_EQ(retired_reference_count, api_.context_reference_count(context));
  hipCtx_t current = reinterpret_cast<hipCtx_t>(uintptr_t{0x5678});
  EXPECT_EQ(hipErrorInvalidContext, api_.ctx_get_current(&current));
  EXPECT_EQ(nullptr, current);
  EXPECT_EQ(hipErrorContextIsDestroyed, api_.ctx_pop_current(&popped));
  EXPECT_EQ(nullptr, popped);
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&current));
  EXPECT_EQ(nullptr, current);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       RetiredCurrentDiscardStopsOnPersistentTlsClearFailures) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api_.ctx_push_current(context));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context));
  const uint64_t retired_reference_count =
      api_.context_reference_count(context);
  ASSERT_NE(0u, retired_reference_count);

  hipCtx_t popped = reinterpret_cast<hipCtx_t>(uintptr_t{0x1234});
  for (int attempt = 0; attempt < 3; ++attempt) {
    api_.inject_tls_failures(kTlsTestFailureClear);
    EXPECT_EQ(hipErrorNotReady, api_.ctx_pop_current(&popped));
    EXPECT_EQ(nullptr, popped);
    EXPECT_EQ(retired_reference_count, api_.context_reference_count(context));
    hipCtx_t current = reinterpret_cast<hipCtx_t>(uintptr_t{0x5678});
    EXPECT_EQ(hipErrorInvalidContext, api_.ctx_get_current(&current));
    EXPECT_EQ(nullptr, current);
  }

  EXPECT_EQ(hipErrorContextIsDestroyed, api_.ctx_pop_current(&popped));
  EXPECT_EQ(nullptr, popped);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PrimaryResetStartsWriterBeforeRegistryResolution) {
  ExpectDeinitWinsBeforeDeviceWriter(api_.device_primary_ctx_reset);
}

TEST_F(HipVmmLifecyclePhaseTest,
       InvalidPrimaryReleaseStartsWriterBeforeRegistryResolution) {
  ExpectDeinitWinsBeforeDeviceWriter(api_.device_primary_ctx_release);
}

TEST_F(HipVmmLifecyclePhaseTest,
       GreenContextPostRetainFailureRollsBackWithoutChangingError) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipDevResource full_resource = {};
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));

  api_.fail_green_context_after_retain_once();
  hipExecutionCtx_t untouched_context =
      reinterpret_cast<hipExecutionCtx_t>(uintptr_t{0x1234});
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.green_ctx_create(&untouched_context, descriptor, device, 0));
  EXPECT_EQ(reinterpret_cast<hipExecutionCtx_t>(uintptr_t{0x1234}),
            untouched_context);
  EXPECT_EQ(hipErrorInvalidContext, api_.device_primary_ctx_release(device));

  hipExecutionCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.green_ctx_create(&context, descriptor, device, 0));
  ASSERT_NE(nullptr, context);
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
  ASSERT_EQ(hipSuccess, api_.execution_ctx_destroy(context));
  EXPECT_EQ(hipErrorInvalidContext, api_.device_primary_ctx_release(device));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  ASSERT_EQ(hipSuccess, api_.init(0));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PlanAllocationFailureHasNoPublicOrNativeMutation) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }

  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  api_.reset_observability();
  // The fifth non-empty durable-plan allocation is the native-segment snapshot
  // for this one-reservation/one-mapping graph.
  api_.fail_plan_allocation_at(5);
  EXPECT_EQ(hipErrorOutOfMemory, api_.device_reset());
  EXPECT_EQ(0u, api_.raw_native_call_count());

  const hipMemLocation location = {hipMemLocationTypeDevice, device};
  unsigned long long flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_get_access(&flags, &location, vmm.reservation));
  EXPECT_EQ(hipMemAccessFlagsProtReadWrite, flags);
  const uint32_t expected = 0xA110CA7Eu;
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(vmm.reservation, &expected,
                                    sizeof(expected), hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, vmm.reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(expected, actual);

  DestroyMappedVmm(vmm);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PlanRevalidationFailureHasNoPublicOrNativeMutation) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }

  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  api_.reset_observability();
  api_.fail_plan_revalidation_once();
  EXPECT_EQ(hipErrorInvalidValue, api_.device_reset());
  EXPECT_EQ(0u, api_.raw_native_call_count());

  const hipMemLocation location = {hipMemLocationTypeDevice, device};
  unsigned long long flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_get_access(&flags, &location, vmm.reservation));
  EXPECT_EQ(hipMemAccessFlagsProtReadWrite, flags);
  DestroyMappedVmm(vmm);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       EveryNativeJournalOrdinalResumesExactlyWithoutReplay) {
  for (int failed_ordinal = 1; failed_ordinal <= 3; ++failed_ordinal) {
    SCOPED_TRACE(failed_ordinal);
    ASSERT_EQ(hipSuccess, api_.init(0));
    int device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&device));
    if (!IsVmmSupported(device)) {
      ASSERT_EQ(hipSuccess, api_.hal_deinit());
      GTEST_SKIP() << "device does not support HIP VMM";
    }

    MappedVmm vmm;
    CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
    // Make the physical allocation natural transaction-owned cleanup: the map
    // remains its only edge when device reset retires the reservation.
    ASSERT_EQ(hipSuccess, api_.mem_release(vmm.allocation));
    api_.fail_native_operation_at(failed_ordinal);

    EXPECT_EQ(hipErrorUnknown, api_.device_reset());
    ASSERT_EQ(3u, api_.native_operation_count());
    EXPECT_EQ(static_cast<uint64_t>(failed_ordinal - 1),
              api_.native_operation_cursor());
    EXPECT_EQ(static_cast<uint64_t>(failed_ordinal - 1),
              api_.raw_native_call_count());
    EXPECT_EQ(UINT32_MAX - 1, api_.first_native_error());
    for (int ordinal = 1; ordinal <= 3; ++ordinal) {
      EXPECT_EQ(ordinal <= failed_ordinal ? 1u : 0u,
                api_.native_operation_attempt_count(ordinal));
    }

    // A destructive failure consumed the reset transaction and deliberately
    // closes ordinary/same-API admission. hipHALDeinit is the recovery API that
    // owns the retained cursor and completes the preserved public ledger.
    EXPECT_EQ(hipErrorNotInitialized, api_.device_reset());
    for (int ordinal = 1; ordinal <= 3; ++ordinal) {
      EXPECT_EQ(ordinal <= failed_ordinal ? 1u : 0u,
                api_.native_operation_attempt_count(ordinal));
    }
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    EXPECT_EQ(3u, api_.raw_native_call_count());
    for (int ordinal = 1; ordinal <= 3; ++ordinal) {
      EXPECT_EQ(ordinal == failed_ordinal ? 2u : 1u,
                api_.native_operation_attempt_count(ordinal));
    }
    // A later successful retry never overwrites the first native error.
    EXPECT_EQ(UINT32_MAX - 1, api_.first_native_error());
  }
}

TEST_F(HipVmmLifecyclePhaseTest,
       LaterAccessPreflightFailureCallsNoNativeMutation) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t granularity = MinimumGranularity(device);
  ASSERT_LE(granularity, SIZE_MAX / 2);
  const size_t reservation_size = granularity * 2;
  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypePinned;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location = {hipMemLocationTypeDevice, device};
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&reservation, reservation_size,
                                                 0, nullptr, 0));
  hipMemGenericAllocationHandle_t allocations[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(hipSuccess,
              api_.mem_create(&allocations[i], granularity, &properties, 0));
    ASSERT_EQ(hipSuccess,
              api_.mem_map(static_cast<uint8_t*>(reservation) + i * granularity,
                           granularity, 0, allocations[i], 0));
  }

  api_.fail_access_preflight_at(2);
  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, device};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  EXPECT_EQ(hipErrorNotSupported,
            api_.mem_set_access(reservation, reservation_size, &descriptor, 1));
  EXPECT_EQ(0u, api_.raw_native_call_count());
  const hipMemLocation location = {hipMemLocationTypeDevice, device};
  for (size_t i = 0; i < 2; ++i) {
    unsigned long long flags = ~0ull;
    ASSERT_EQ(hipSuccess,
              api_.mem_get_access(
                  &flags, &location,
                  static_cast<uint8_t*>(reservation) + i * granularity));
    EXPECT_EQ(hipMemAccessFlagsProtNone, flags);
  }

  api_.reset_observability();
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, reservation_size, &descriptor, 1));
  ASSERT_GT(api_.raw_native_call_count(), 0u);
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, reservation_size));
  for (auto allocation : allocations) {
    ASSERT_EQ(hipSuccess, api_.mem_release(allocation));
  }
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, reservation_size));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       BothEpochDomainsRejectMaxMinusOneBeforeNativeMutation) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  for (int source = 0; source < 2; ++source) {
    SCOPED_TRACE(source == 0 ? "VMM epoch" : "common epoch");
    api_.reset_observability();
    ScopedEpochExhaustion forced_epoch(api_.set_epoch_exhaustion, device,
                                       source);
    ASSERT_EQ(hipSuccess, forced_epoch.enable_result());
    EXPECT_EQ(hipErrorOutOfMemory, api_.device_reset());
    EXPECT_EQ(0u, api_.raw_native_call_count());
    ASSERT_EQ(hipSuccess, forced_epoch.Restore());
  }
  EXPECT_EQ(hipSuccess, api_.device_reset());
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       TeardownAddressFreeCallbackWindowRejectsSynchronousHipReentry) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  ASSERT_EQ(hipSuccess, api_.mem_release(vmm.allocation));

  CallbackReentryState state;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.ptr = vmm.reservation;
  state.location = {hipMemLocationTypeDevice, device};
  {
    ScopedPhaseObserver observer(api_.set_phase_observer,
                                 ObserveCallbackReentry, &state);
    ASSERT_EQ(hipSuccess, api_.device_reset());
  }

  ExpectCallbackReentryRejected(state);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       DirectAddressFreeCallbackWindowRejectsSynchronousHipReentry) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));

  CallbackReentryState state;
  state.target_phase = 17;
  state.containing_ptr = reservation;
  state.containing_range = size;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.ptr = reservation;
  state.location = {hipMemLocationTypeDevice, device};
  hipError_t free_result = hipErrorUnknown;
  {
    ScopedPhaseObserver observer(api_.set_phase_observer,
                                 ObserveCallbackReentry, &state);
    free_result = api_.mem_address_free(reservation, size);
  }

  EXPECT_EQ(hipSuccess, free_result);
  if (free_result != hipSuccess) {
    EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, size));
  }
  ExpectSingleContainingAddressFreeCallback(state, reservation, size);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       AddressReserveCallbackWindowRejectsSynchronousHipReentryAndRecovers) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);

  CallbackReentryState state;
  state.target_phase = 17;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.location = {hipMemLocationTypeDevice, device};
  api_.fail_address_reserve_after_native_once();
  void* reservation = reinterpret_cast<void*>(uintptr_t{1});
  hipError_t reserve_result = hipErrorUnknown;
  {
    ScopedPhaseObserver observer(api_.set_phase_observer,
                                 ObserveCallbackReentry, &state);
    reserve_result =
        api_.mem_address_reserve(&reservation, size, 0, nullptr, 0);
  }

  if (reserve_result == hipSuccess && reservation) {
    const hipError_t cleanup_result = api_.mem_address_free(reservation, size);
    EXPECT_EQ(hipSuccess, cleanup_result);
    if (cleanup_result != hipSuccess) {
      EXPECT_EQ(hipSuccess, api_.hal_deinit());
      return;
    }
    reservation = nullptr;
  }
  EXPECT_EQ(hipErrorOutOfMemory, reserve_result);
  EXPECT_EQ(nullptr, reservation);
  ExpectSingleCallbackReentryRejected(state);
  EXPECT_NE(0u, state.matched_object.load(std::memory_order_acquire));

  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  ASSERT_NE(nullptr, reservation);
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, size));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       LowerReserveRollbackCleanupFailureIsRetainedAcrossResetRetry) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);

  api_.fail_cleanup_count(/*physical_memory=*/false, /*failure_count=*/2);
  api_.fail_lower_virtual_reserve_after_native_once();
  void* reservation = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(nullptr, reservation);
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(1u, api_.cleanup_attempt_count(/*physical_memory=*/false));
  EXPECT_NE(0u, api_.last_cleanup_status(/*physical_memory=*/false));

  // The first writer-held drain fails before reset commits and leaves the
  // runtime, allocator state, and exact native owner retryable.
  EXPECT_EQ(hipErrorOutOfMemory, api_.device_reset());
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(2u, api_.cleanup_attempt_count(/*physical_memory=*/false));
  EXPECT_EQ(hipSuccess, api_.device_reset());
  EXPECT_EQ(0u, api_.quarantine_count());
  EXPECT_EQ(3u, api_.cleanup_attempt_count(/*physical_memory=*/false));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       HrxWrapperRollbackCleanupFailureIsRetainedAcrossDeinitRetry) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);

  api_.fail_cleanup_count(/*physical_memory=*/false, /*failure_count=*/2);
  api_.fail_hrx_virtual_reserve_after_native_once();
  void* reservation = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(nullptr, reservation);
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(1u, api_.cleanup_attempt_count(/*physical_memory=*/false));

  EXPECT_EQ(hipErrorOutOfMemory, api_.hal_deinit());
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(2u, api_.cleanup_attempt_count(/*physical_memory=*/false));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
  EXPECT_EQ(0u, api_.quarantine_count());
  EXPECT_EQ(3u, api_.cleanup_attempt_count(/*physical_memory=*/false));
}

TEST_F(HipVmmLifecyclePhaseTest,
       HipReservationPublicationFailureTransfersSoleOwnerToDeinitDrain) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);

  api_.fail_cleanup_count(/*physical_memory=*/false, /*failure_count=*/1);
  api_.fail_address_reserve_after_native_once();
  void* reservation = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(nullptr, reservation);
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(1u, api_.cleanup_attempt_count(/*physical_memory=*/false));

  EXPECT_EQ(hipSuccess, api_.hal_deinit());
  EXPECT_EQ(0u, api_.quarantine_count());
  EXPECT_EQ(2u, api_.cleanup_attempt_count(/*physical_memory=*/false));
}

TEST_F(HipVmmLifecyclePhaseTest,
       HipPhysicalPublicationFailureIsRetainedAcrossResetRetry) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);
  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypePinned;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location = {hipMemLocationTypeDevice, device};

  api_.fail_cleanup_count(/*physical_memory=*/true, /*failure_count=*/2);
  api_.fail_physical_create_after_native_once();
  hipMemGenericAllocationHandle_t allocation =
      reinterpret_cast<hipMemGenericAllocationHandle_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.mem_create(&allocation, size, &properties, 0));
  EXPECT_EQ(nullptr, allocation);
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(1u, api_.cleanup_attempt_count(/*physical_memory=*/true));

  EXPECT_EQ(hipErrorOutOfMemory, api_.device_reset());
  EXPECT_EQ(1u, api_.quarantine_count());
  EXPECT_EQ(2u, api_.cleanup_attempt_count(/*physical_memory=*/true));
  EXPECT_EQ(hipSuccess, api_.device_reset());
  EXPECT_EQ(0u, api_.quarantine_count());
  EXPECT_EQ(3u, api_.cleanup_attempt_count(/*physical_memory=*/true));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       DetachedDrainRequeuesFailedOldOwnerAheadOfConcurrentNewOwner) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  ASSERT_NE(size, 0u);

  // Seed one old reservation owner in quarantine.
  api_.fail_cleanup_count(/*physical_memory=*/false, /*failure_count=*/1);
  api_.fail_address_reserve_after_native_once();
  void* reservation = nullptr;
  ASSERT_EQ(hipErrorOutOfMemory,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  ASSERT_EQ(1u, api_.quarantine_count());

  // Detach the old list, then enqueue a new physical owner while the drain is
  // paused outside the domain mutex. Both attempts fail once.
  api_.fail_cleanup_count(/*physical_memory=*/false, /*failure_count=*/1);
  api_.arm_quarantine_drain_pause();
  hipError_t first_drain_result = hipSuccess;
  std::thread drain_thread([&] {
    first_drain_result = api_.quarantine_drain_without_writer(device);
  });
  api_.wait_quarantine_drain_paused();

  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypePinned;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location = {hipMemLocationTypeDevice, device};
  api_.fail_cleanup_count(/*physical_memory=*/true, /*failure_count=*/1);
  api_.fail_physical_create_after_native_once();
  hipMemGenericAllocationHandle_t allocation = nullptr;
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.mem_create(&allocation, size, &properties, 0));
  EXPECT_EQ(nullptr, allocation);
  EXPECT_EQ(2u, api_.quarantine_count());

  api_.release_quarantine_drain_pause();
  drain_thread.join();
  EXPECT_EQ(hipErrorOutOfMemory, first_drain_result);
  EXPECT_EQ(2u, api_.quarantine_count());
  EXPECT_EQ(1, api_.quarantine_drain_attempt_kind(1));

  // The failed detached reservation is prepended ahead of the concurrently
  // enqueued physical owner: retry order is reservation, then physical.
  EXPECT_EQ(hipSuccess, api_.quarantine_drain_without_writer(device));
  EXPECT_EQ(0u, api_.quarantine_count());
  EXPECT_EQ(1, api_.quarantine_drain_attempt_kind(2));
  EXPECT_EQ(2, api_.quarantine_drain_attempt_kind(3));
  EXPECT_EQ(0, api_.quarantine_drain_attempt_kind(4));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PoolSlabAcquisitionRollbackRejectsExactAddressFreeReentry) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  size_t free_memory = 0;
  size_t total_memory = 0;
  ASSERT_EQ(hipSuccess, api_.mem_get_info(&free_memory, &total_memory));
  EXPECT_LE(free_memory, total_memory);
  constexpr size_t kFailureMargin = size_t{4} * 1024 * 1024 * 1024;
  if (total_memory > SIZE_MAX - kFailureMargin) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "reported device memory leaves no oversized test range";
  }
  const size_t failing_slab_size = total_memory + kFailureMargin;

  hipMemPoolProps properties = {};
  properties.allocType = hipMemAllocationTypePinned;
  properties.location = {hipMemLocationTypeDevice, device};
  hipMemPool_t pool = nullptr;
  void* allocation = nullptr;
  hipError_t allocation_result = hipErrorUnknown;
  CallbackReentryState state;
  state.target_phase = 17;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.location = {hipMemLocationTypeDevice, device};
  state.probe_pool_apis = true;
  state.pool_trim = api_.mem_pool_trim_to;
  state.pool_destroy = api_.mem_pool_destroy;
  state.malloc_from_pool_async = api_.malloc_from_pool_async;
  {
    ScopedEnvironmentVariable generic_pool_bytes(
        "HRX_MEM_POOL_BYTES", std::to_string(failing_slab_size));
    ScopedEnvironmentVariable hip_pool_bytes("HRX_HIP_POOL_BYTES",
                                             std::to_string(failing_slab_size));
    ASSERT_EQ(hipSuccess, api_.mem_pool_create(&pool, &properties));
    state.pool = pool;
    {
      ScopedPhaseObserver observer(api_.set_phase_observer,
                                   ObserveCallbackReentry, &state);
      allocation_result = api_.malloc_from_pool_async(
          &allocation, /*size=*/4096, pool, /*stream=*/nullptr);
    }
    EXPECT_NE(hipSuccess, allocation_result);
    if (allocation_result == hipSuccess) {
      EXPECT_EQ(hipSuccess, api_.free_async(allocation, /*stream=*/nullptr));
      EXPECT_EQ(hipSuccess, api_.stream_synchronize(/*stream=*/nullptr));
    }
    EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool));
  }

  ExpectSingleCallbackReentryRejected(state);
  ExpectPoolCallbackReentryRejected(state);
  const size_t size = MinimumGranularity(device);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, size));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PoolTrimRejectsExactAddressFreeReentryAndClearsTls) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  ScopedEnvironmentVariable generic_pool_bytes(
      "HRX_MEM_POOL_BYTES", std::to_string(kDeterministicPoolBytes));
  ScopedEnvironmentVariable hip_pool_bytes(
      "HRX_HIP_POOL_BYTES", std::to_string(kDeterministicPoolBytes));
  hipMemPoolProps properties = {};
  properties.allocType = hipMemAllocationTypePinned;
  properties.location = {hipMemLocationTypeDevice, device};
  hipMemPool_t pool = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_pool_create(&pool, &properties));
  int allow_opportunistic = 0;
  ASSERT_EQ(hipSuccess, api_.mem_pool_set_attribute(
                            pool, hipMemPoolAttrReuseAllowOpportunistic,
                            &allow_opportunistic));
  uint64_t release_threshold = kDeterministicPoolBytes;
  ASSERT_EQ(hipSuccess,
            api_.mem_pool_set_attribute(pool, hipMemPoolAttrReleaseThreshold,
                                        &release_threshold));
  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.malloc_from_pool_async(&allocation, /*size=*/4096, pool,
                                        /*stream=*/nullptr));
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.free_async(allocation, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(/*stream=*/nullptr));

  CallbackReentryState state;
  state.target_phase = 17;
  state.containing_ptr = allocation;
  state.containing_range = kDeterministicPoolBytes;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.ptr = allocation;
  state.location = {hipMemLocationTypeDevice, device};
  state.probe_pool_apis = true;
  state.pool_trim = api_.mem_pool_trim_to;
  state.pool_destroy = api_.mem_pool_destroy;
  state.malloc_from_pool_async = api_.malloc_from_pool_async;
  state.pool = pool;
  hipError_t trim_result = hipErrorUnknown;
  {
    ScopedPhaseObserver observer(api_.set_phase_observer,
                                 ObserveCallbackReentry, &state);
    trim_result = api_.mem_pool_trim_to(pool, /*minBytesToKeep=*/0);
  }

  EXPECT_EQ(hipSuccess, trim_result);
  ExpectSingleContainingAddressFreeCallback(state, allocation,
                                            kDeterministicPoolBytes);
  ExpectPoolCallbackReentryRejected(state);
  EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool));

  const size_t size = MinimumGranularity(device);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, size));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       DeviceResetPoolReleaseRejectsExactAddressFreeReentryAndRecovers) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  ScopedEnvironmentVariable generic_pool_bytes(
      "HRX_MEM_POOL_BYTES", std::to_string(kDeterministicPoolBytes));
  ScopedEnvironmentVariable hip_pool_bytes(
      "HRX_HIP_POOL_BYTES", std::to_string(kDeterministicPoolBytes));
  hipMemPool_t default_pool = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_get_default_mem_pool(&default_pool, device));
  ASSERT_NE(nullptr, default_pool);
  int allow_opportunistic = 0;
  ASSERT_EQ(hipSuccess, api_.mem_pool_set_attribute(
                            default_pool, hipMemPoolAttrReuseAllowOpportunistic,
                            &allow_opportunistic));
  uint64_t release_threshold = kDeterministicPoolBytes;
  ASSERT_EQ(hipSuccess, api_.mem_pool_set_attribute(
                            default_pool, hipMemPoolAttrReleaseThreshold,
                            &release_threshold));

  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc_async(&allocation, /*size=*/4096,
                                          /*stream=*/nullptr));
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.free_async(allocation, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(/*stream=*/nullptr));

  CallbackReentryState state;
  state.target_phase = 17;
  state.containing_ptr = allocation;
  state.containing_range = kDeterministicPoolBytes;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.ptr = allocation;
  state.location = {hipMemLocationTypeDevice, device};
  state.probe_pool_apis = true;
  state.pool_trim = api_.mem_pool_trim_to;
  state.pool_destroy = api_.mem_pool_destroy;
  state.malloc_from_pool_async = api_.malloc_from_pool_async;
  state.pool = default_pool;
  hipError_t reset_result = hipErrorUnknown;
  {
    ScopedPhaseObserver observer(api_.set_phase_observer,
                                 ObserveCallbackReentry, &state);
    reset_result = api_.device_reset();
  }

  EXPECT_EQ(hipSuccess, reset_result);
  ExpectSingleContainingAddressFreeCallback(state, allocation,
                                            kDeterministicPoolBytes);
  ExpectPoolCallbackReentryRejected(state);
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, size));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       GlobalDeinitPoolReleaseRejectsExactAddressFreeReentryAndReinitializes) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  ScopedEnvironmentVariable generic_pool_bytes(
      "HRX_MEM_POOL_BYTES", std::to_string(kDeterministicPoolBytes));
  ScopedEnvironmentVariable hip_pool_bytes(
      "HRX_HIP_POOL_BYTES", std::to_string(kDeterministicPoolBytes));
  hipMemPool_t default_pool = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_get_default_mem_pool(&default_pool, device));
  ASSERT_NE(nullptr, default_pool);
  int allow_opportunistic = 0;
  ASSERT_EQ(hipSuccess, api_.mem_pool_set_attribute(
                            default_pool, hipMemPoolAttrReuseAllowOpportunistic,
                            &allow_opportunistic));
  uint64_t release_threshold = kDeterministicPoolBytes;
  ASSERT_EQ(hipSuccess, api_.mem_pool_set_attribute(
                            default_pool, hipMemPoolAttrReleaseThreshold,
                            &release_threshold));

  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc_async(&allocation, /*size=*/4096,
                                          /*stream=*/nullptr));
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.free_async(allocation, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(/*stream=*/nullptr));

  CallbackReentryState state;
  state.target_phase = 17;
  state.containing_ptr = allocation;
  state.containing_range = kDeterministicPoolBytes;
  state.init = api_.init;
  state.deinit = api_.hal_deinit;
  state.get_access = api_.mem_get_access;
  state.ptr = allocation;
  state.location = {hipMemLocationTypeDevice, device};
  state.probe_pool_apis = true;
  state.pool_trim = api_.mem_pool_trim_to;
  state.pool_destroy = api_.mem_pool_destroy;
  state.malloc_from_pool_async = api_.malloc_from_pool_async;
  state.pool = default_pool;
  hipError_t deinit_result = hipErrorUnknown;
  {
    ScopedPhaseObserver observer(api_.set_phase_observer,
                                 ObserveCallbackReentry, &state);
    deinit_result = api_.hal_deinit();
  }

  EXPECT_EQ(hipSuccess, deinit_result);
  ExpectSingleContainingAddressFreeCallback(state, allocation,
                                            kDeterministicPoolBytes);
  ExpectPoolCallbackReentryRejected(state);
  ASSERT_EQ(hipSuccess, api_.init(0));
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const size_t size = MinimumGranularity(device);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, size, 0, nullptr, 0));
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, size));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest, ExplicitContextDestroyRemovesOnlyItsAlias) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);

  hipCtx_t explicit_context = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.ctx_create(&explicit_context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(explicit_context));
  const uint32_t expected = 0xE71C17A5u;
  ASSERT_EQ(hipSuccess, api_.memcpy(vmm.reservation, &expected,
                                    sizeof(expected), hipMemcpyHostToDevice));

  api_.reset_observability();
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(explicit_context));
  EXPECT_EQ(0u, api_.native_operation_count());
  EXPECT_EQ(0u, api_.raw_native_call_count());

  ASSERT_EQ(hipSuccess, api_.set_device(device));
  const hipMemLocation location = {hipMemLocationTypeDevice, device};
  unsigned long long flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_get_access(&flags, &location, vmm.reservation));
  EXPECT_EQ(hipMemAccessFlagsProtReadWrite, flags);
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, vmm.reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(expected, actual);
  DestroyMappedVmm(vmm);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       DirectLastPrimaryFailureRecoversThroughGlobalTeardown) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  hipCtx_t primary_context = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_primary_ctx_retain(&primary_context, device));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(primary_context));
  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  ASSERT_EQ(hipSuccess, api_.mem_release(vmm.allocation));

  api_.fail_native_operation_at(2);
  EXPECT_EQ(hipErrorUnknown, api_.device_primary_ctx_release(device));
  ASSERT_EQ(3u, api_.native_operation_count());
  EXPECT_EQ(1u, api_.native_operation_cursor());
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(0u, api_.native_operation_attempt_count(3));

  // The public retain was consumed by the committed release. Repeating the
  // same call must not consume it twice or replay the journal.
  EXPECT_EQ(hipErrorNotInitialized, api_.device_primary_ctx_release(device));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(0u, api_.native_operation_attempt_count(3));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(2u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(3));
  ASSERT_EQ(hipSuccess, api_.init(0));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PartitionedLastPrimaryFailureRecoversThroughGlobalTeardown) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  hipDevResource full_resource = {};
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));
  hipExecutionCtx_t execution_context = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.green_ctx_create(&execution_context, descriptor, device, 0));

  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  ASSERT_EQ(hipSuccess, api_.mem_release(vmm.allocation));
  api_.fail_native_operation_at(2);
  EXPECT_EQ(hipErrorUnknown, api_.execution_ctx_destroy(execution_context));
  ASSERT_EQ(3u, api_.native_operation_count());
  EXPECT_EQ(1u, api_.native_operation_cursor());

  // The partitioned handle and its primary retain were consumed exactly once
  // at the committed boundary; only global teardown may resume the ledger.
  EXPECT_EQ(hipErrorNotInitialized,
            api_.execution_ctx_destroy(execution_context));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(0u, api_.native_operation_attempt_count(3));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(2u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(3));
  ASSERT_EQ(hipSuccess, api_.init(0));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       GlobalDeinitFailureResumesBeforeReinitialization) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  ASSERT_EQ(hipSuccess, api_.mem_release(vmm.allocation));

  api_.fail_native_operation_at(2);
  EXPECT_EQ(hipErrorUnknown, api_.hal_deinit());
  ASSERT_EQ(3u, api_.native_operation_count());
  EXPECT_EQ(1u, api_.native_operation_cursor());
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(0u, api_.native_operation_attempt_count(3));

  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  EXPECT_EQ(1u, api_.native_operation_attempt_count(1));
  EXPECT_EQ(2u, api_.native_operation_attempt_count(2));
  EXPECT_EQ(1u, api_.native_operation_attempt_count(3));
  ASSERT_EQ(hipSuccess, api_.init(0));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       DeviceResetClipsMixedForeignGrantAndPreservesForeignQueue) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2 || !IsVmmSupported(0) || !IsVmmSupported(1)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "requires two HIP VMM devices";
  }

  const size_t target_granularity = MinimumGranularity(0);
  const size_t foreign_granularity = MinimumGranularity(1);
  const size_t granularity_gcd =
      std::gcd(target_granularity, foreign_granularity);
  ASSERT_NE(granularity_gcd, 0u);
  const size_t foreign_factor = foreign_granularity / granularity_gcd;
  ASSERT_LE(target_granularity, SIZE_MAX / foreign_factor);
  const size_t chunk = target_granularity * foreign_factor;
  ASSERT_NE(chunk, 0u);
  ASSERT_LE(chunk, SIZE_MAX / 2);
  const size_t reservation_size = chunk * 2;

  ASSERT_EQ(hipSuccess, api_.set_device(1));
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&reservation, reservation_size,
                                                 0, nullptr, 0));
  auto* first = static_cast<uint8_t*>(reservation);
  auto* second = first + chunk;

  hipMemAllocationProp target_properties = {};
  target_properties.type = hipMemAllocationTypePinned;
  target_properties.requestedHandleType = hipMemHandleTypeNone;
  target_properties.location = {hipMemLocationTypeDevice, 0};
  hipMemAllocationProp foreign_properties = target_properties;
  foreign_properties.location.id = 1;
  hipMemGenericAllocationHandle_t target_allocation = nullptr;
  hipMemGenericAllocationHandle_t foreign_allocation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&target_allocation, chunk, &target_properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_create(&foreign_allocation, chunk,
                                        &foreign_properties, 0));

  // This map is the cross-logical-device atomic-cell case: reservation queue
  // family ordinals belong to device 1 while the physical pool belongs to
  // device 0. The backend must query the pool with the reservation's exact HSA
  // agent identity rather than reinterpreting its local ordinal in topology 0.
  ASSERT_EQ(hipSuccess, api_.mem_map(first, chunk, 0, target_allocation, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(second, chunk, 0, foreign_allocation, 0));

  hipMemAccessDesc foreign_access = {};
  foreign_access.location = {hipMemLocationTypeDevice, 1};
  foreign_access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess, api_.mem_set_access(reservation, reservation_size,
                                            &foreign_access, 1));
  hipMemAccessDesc target_access = {};
  target_access.location = {hipMemLocationTypeDevice, 0};
  target_access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess, api_.mem_set_access(first, chunk, &target_access, 1));

  hipCtx_t foreign_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&foreign_context));
  ASSERT_NE(nullptr, foreign_context);

  MixedResetHostState host_state;
  host_state.probe = api_.probe_accepted_access;
  host_state.ptr = first;
  host_state.size = sizeof(uint32_t);
  host_state.device = 1;
  hipHostNodeParams host_params = {MixedResetHostCallback, &host_state};
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, 0));
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&host_node, graph, nullptr, 0,
                                                 &host_params));
  hipMemsetParams memset_params = {};
  memset_params.dst = first;
  memset_params.value = 0x5A;
  memset_params.elementSize = 1;
  memset_params.width = sizeof(uint32_t);
  memset_params.height = 1;
  hipGraphNode_t memset_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_memset_node(&memset_node, graph, &host_node, 1,
                                       &memset_params));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  host_state.entered.Wait();

  MixedResetPhaseState phase_state;
  phase_state.expected_incident_context = foreign_context;
  ASSERT_EQ(hipSuccess,
            api_.set_phase_observer(ObserveMixedResetPhase, &phase_state));
  std::atomic<bool> reset_done{false};
  std::atomic<int> reset_result{hipErrorUnknown};
  std::thread reset_thread([&] {
    hipError_t result = api_.set_device(0);
    if (result == hipSuccess) result = api_.device_reset();
    reset_result.store(result, std::memory_order_release);
    reset_done.store(true, std::memory_order_release);
  });
  phase_state.writer_pending.Wait();
  EXPECT_FALSE(reset_done.load(std::memory_order_acquire));

  host_state.release_probe.Notify();
  host_state.probe_complete.Wait();
  const hipError_t probe_result = static_cast<hipError_t>(
      host_state.probe_result.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, probe_result);
  if (probe_result == hipSuccess) {
    phase_state.release_writer.Notify();
    phase_state.incident_context_stream_waiting.Wait();
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(foreign_context),
        phase_state.observed_incident_context.load(std::memory_order_acquire));
    EXPECT_FALSE(reset_done.load(std::memory_order_acquire));
    host_state.release_return.Notify();
  } else {
    host_state.release_return.Notify();
    phase_state.release_writer.Notify();
  }
  reset_thread.join();
  ASSERT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));
  ASSERT_EQ(hipSuccess, static_cast<hipError_t>(
                            reset_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipSuccess, probe_result);

  const hipMemLocation foreign_location = {hipMemLocationTypeDevice, 1};
  unsigned long long flags = ~0ull;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.mem_get_access(&flags, &foreign_location, first));
  ASSERT_EQ(hipSuccess, api_.mem_get_access(&flags, &foreign_location, second));
  EXPECT_EQ(hipMemAccessFlagsProtReadWrite, flags);
  const hipMemLocation target_location = {hipMemLocationTypeDevice, 0};
  flags = ~0ull;
  ASSERT_EQ(hipSuccess, api_.mem_get_access(&flags, &target_location, second));
  EXPECT_EQ(hipMemAccessFlagsProtNone, flags);

  // Device B's context and access alias survived the target reset. Exercise a
  // fresh public transfer path without relaunching the graph that names first.
  ASSERT_EQ(hipSuccess, api_.set_device(1));
  const uint32_t expected = 0x5A5A5A5Au;
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(second, &expected, sizeof(expected),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, second, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(expected, actual);

  EXPECT_EQ(hipErrorInvalidValue, api_.mem_release(target_allocation));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(second, chunk));
  ASSERT_EQ(hipSuccess, api_.mem_release(foreign_allocation));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, reservation_size));
  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       GraphExecUpdateWaitsForActiveGraphMemoryLaunch) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  BlockingHostState old_host_state;
  BlockingHostState new_host_state;
  new_host_state.release.Notify();
  hipGraph_t old_graph = nullptr;
  hipGraph_t new_graph = nullptr;
  CreateGraphMemoryHostGraph(device, &old_host_state, &old_graph);
  CreateGraphMemoryHostGraph(device, &new_host_state, &new_graph);

  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, old_graph, nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  old_host_state.entered.Wait();

  PhaseState phase_state;
  ScopedPhaseObserver observer(api_.set_phase_observer, ObservePhase,
                               &phase_state);
  std::atomic<bool> update_done{false};
  std::atomic<int> update_status{hipErrorUnknown};
  std::atomic<int> update_result{hipGraphExecUpdateError};
  std::thread update_thread([&] {
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    update_status.store(
        api_.graph_exec_update(exec, new_graph, &error_node, &result),
        std::memory_order_release);
    update_result.store(result, std::memory_order_release);
    update_done.store(true, std::memory_order_release);
  });

  // This notification is issued while update holds executable state, directly
  // before it drops that state to wait on the old launch's stream timeline.
  // The blocked host callback prevents that timeline from reaching the named
  // value, so update cannot have retired the old compiled state or graph here.
  phase_state.graph_exec_active_launch_waiting.Wait();
  EXPECT_FALSE(update_done.load(std::memory_order_acquire));
  old_host_state.release.Notify();
  update_thread.join();
  observer.Reset();

  ASSERT_EQ(hipSuccess, static_cast<hipError_t>(
                            update_status.load(std::memory_order_acquire)));
  ASSERT_EQ(hipGraphExecUpdateSuccess,
            static_cast<hipGraphExecUpdateResult>(
                update_result.load(std::memory_order_acquire)));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  new_host_state.entered.Wait();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(new_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(old_graph));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       FailedGraphMemoryLaunchPublishesAcceptedPrefix) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }
  ASSERT_EQ(hipSuccess, api_.set_device(device));

  static constexpr size_t kEventCount = 17;
  std::array<hipEvent_t, kEventCount> old_events = {};
  std::array<hipEvent_t, kEventCount> new_events = {};
  BlockingHostState old_host_state;
  BlockingHostState new_host_state;
  new_host_state.release.Notify();
  hipGraph_t old_graph = nullptr;
  hipGraph_t new_graph = nullptr;
  CreateGraphMemoryHostGraph(device, &old_host_state, &old_graph,
                             old_events.data(), old_events.size());
  CreateGraphMemoryHostGraph(device, &new_host_state, &new_graph,
                             new_events.data(), new_events.size());

  // Sixteen event release cells live in the launching frame. Installing this
  // allocator before instantiation gives the private executable snapshot its
  // own wrapper, then arming it afterwards makes the 17th event cell the first
  // and only failing launch allocation. The graph-memory allocation, blocked
  // host call, and sixteen event blocks have already been accepted by then.
  ASSERT_EQ(hipSuccess, api_.install_graph_exec_allocator(old_graph));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, old_graph, nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.fail_graph_exec_allocation_at(1));
  ASSERT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));
  old_host_state.entered.Wait();

  PhaseState phase_state;
  ScopedPhaseObserver observer(api_.set_phase_observer, ObservePhase,
                               &phase_state);
  std::atomic<bool> update_done{false};
  std::atomic<int> update_status{hipErrorUnknown};
  std::atomic<int> update_result{hipGraphExecUpdateError};
  std::thread update_thread([&] {
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    update_status.store(
        api_.graph_exec_update(exec, new_graph, &error_node, &result),
        std::memory_order_release);
    update_result.store(result, std::memory_order_release);
    update_done.store(true, std::memory_order_release);
  });

  // Update owns executable state here but cannot retire the snapshot, its
  // graph-memory allocation slot, or any accepted event reference until the
  // blocked prefix reaches the exact completion frontier retained by launch.
  phase_state.graph_exec_active_launch_waiting.Wait();
  EXPECT_FALSE(update_done.load(std::memory_order_acquire));
  old_host_state.release.Notify();
  update_thread.join();
  observer.Reset();

  ASSERT_EQ(hipSuccess, static_cast<hipError_t>(
                            update_status.load(std::memory_order_acquire)));
  ASSERT_EQ(hipGraphExecUpdateSuccess,
            static_cast<hipGraphExecUpdateResult>(
                update_result.load(std::memory_order_acquire)));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  new_host_state.entered.Wait();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(new_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(old_graph));
  for (hipEvent_t event : new_events) {
    ASSERT_EQ(hipSuccess, api_.event_destroy(event));
  }
  for (hipEvent_t event : old_events) {
    ASSERT_EQ(hipSuccess, api_.event_destroy(event));
  }
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       CrossContextHostGraphLaunchRejectedBeforeOwnerTeardown) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipCtx_t context_a = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_a, /*flags=*/0, device));

  // Host-to-host memcpy lowers to a host call whose copied payload lives in
  // the executable arena. This is the state that owner-context teardown must
  // never free while a foreign stream can still have the callback queued.
  const uint32_t source = 0x13572468u;
  uint32_t target = 0xA5A5A5A5u;
  hipGraph_t owner_graph = nullptr;
  hipGraphNode_t owner_node = nullptr;
  hipGraphExec_t owner_exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&owner_graph, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_add_memcpy_node_1d(
                            &owner_node, owner_graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &target, &source,
                            sizeof(source), hipMemcpyHostToHost));
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&owner_exec, owner_graph,
                                               nullptr, nullptr, 0));

  hipCtx_t context_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_b, /*flags=*/0, device));
  hipStream_t stream_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_b));

  // Hold B's explicit stream at a deterministic callback boundary. A broken
  // implementation would accept A's graph behind this callback and leave its
  // arena payload exposed to A's otherwise unrelated context teardown.
  BlockingHostState blocker;
  hipHostNodeParams blocker_params = {BlockingHostCallback, &blocker};
  hipGraph_t blocker_graph = nullptr;
  hipGraphNode_t blocker_node = nullptr;
  hipGraphExec_t blocker_exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&blocker_graph, /*flags=*/0));
  ASSERT_EQ(hipSuccess,
            api_.graph_add_host_node(&blocker_node, blocker_graph,
                                     /*dependencies=*/nullptr,
                                     /*dependency_count=*/0, &blocker_params));
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&blocker_exec, blocker_graph,
                                               nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(blocker_exec, stream_b));
  blocker.entered.Wait();

  const uint64_t context_a_references = api_.context_reference_count(context_a);
  const uint64_t context_b_references = api_.context_reference_count(context_b);
  const hipError_t launch_result = api_.graph_launch(owner_exec, stream_b);
  EXPECT_EQ(hipErrorInvalidContext, launch_result);

  // Keep the test safe on a regressed implementation: drain a mistakenly
  // accepted foreign launch before touching A's arena-owning executable. The
  // failed expectation above still makes the regression visible.
  if (launch_result != hipErrorInvalidContext) {
    blocker.release.Notify();
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_b));
    EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(blocker_exec));
    EXPECT_EQ(hipSuccess, api_.graph_destroy(blocker_graph));
    EXPECT_EQ(hipSuccess, api_.ctx_set_current(context_a));
    EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(owner_exec));
    EXPECT_EQ(hipSuccess, api_.graph_destroy(owner_graph));
    EXPECT_EQ(hipSuccess, api_.ctx_destroy(context_a));
    EXPECT_EQ(hipSuccess, api_.ctx_set_current(context_b));
    EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_b));
    EXPECT_EQ(hipSuccess, api_.ctx_destroy(context_b));
    EXPECT_EQ(hipSuccess, api_.set_device(device));
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    return;
  }

  // Rejection must unwind both retained operation guards without publishing
  // stream work or changing either context's ownership count.
  EXPECT_EQ(context_a_references, api_.context_reference_count(context_a));
  EXPECT_EQ(context_b_references, api_.context_reference_count(context_b));
  EXPECT_EQ(0xA5A5A5A5u, target);

  // Destroy A while B remains inside its callback. Since no A work reached B,
  // owner teardown has no foreign frontier to wait on and cannot expose an
  // executable-arena payload to B after returning.
  const hipError_t set_a_result = api_.ctx_set_current(context_a);
  EXPECT_EQ(hipSuccess, set_a_result);
  const hipError_t destroy_a_result = set_a_result == hipSuccess
                                          ? api_.ctx_destroy(context_a)
                                          : hipErrorInvalidContext;
  EXPECT_EQ(hipSuccess, destroy_a_result);

  // Always release the callback before any fatal cleanup assertion so a test
  // failure cannot strand B's queue service thread.
  blocker.release.Notify();
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
  if (destroy_a_result == hipSuccess) {
    EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_destroy(owner_exec));
    EXPECT_EQ(hipErrorInvalidValue, api_.graph_destroy(owner_graph));
  }
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_b));
  EXPECT_EQ(0xA5A5A5A5u, target);

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(blocker_exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(blocker_graph));
  ASSERT_EQ(hipSuccess, api_.stream_destroy(stream_b));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_b));
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       GraphExecMemcpyStagingCommitAndFailureRollback) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  if (!IsVmmSupported(device)) {
    ASSERT_EQ(hipSuccess, api_.hal_deinit());
    GTEST_SKIP() << "device does not support HIP VMM";
  }

  MappedVmm vmm;
  CreateMappedVmm(device, /*page_count=*/1, /*grant_access=*/true, &vmm);
  const uint32_t initial_value = 0x13572468u;
  const uint32_t committed_value = 0x24681357u;
  const uint32_t prepare_failure_value = 0xA5A5A5A5u;
  const uint32_t rebuild_failure_value = 0x5A5A5A5Au;

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t memcpy_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_memcpy_node_1d(
                &memcpy_node, graph, /*dependencies=*/nullptr,
                /*dependency_count=*/0, vmm.reservation, &initial_value,
                sizeof(initial_value), hipMemcpyHostToDevice));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

  ASSERT_EQ(hipSuccess,
            api_.graph_exec_memcpy_node_set_params_1d(
                exec, memcpy_node, vmm.reservation, &committed_value,
                sizeof(committed_value), hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, vmm.reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(committed_value, actual);

  // Fail after the replacement staging allocation exists but before the
  // callback or memcpy attributes are replaced. Rollback must dispose the
  // unpublished allocation after dropping executable state and preserve the
  // last committed executable/template state.
  api_.fail_graph_exec_memcpy_prepare_once();
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.graph_exec_memcpy_node_set_params_1d(
                exec, memcpy_node, vmm.reservation, &prepare_failure_value,
                sizeof(prepare_failure_value), hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
  actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, vmm.reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(committed_value, actual);

  // Fail after the template transaction is complete but before compiling it.
  // Rollback must restore all old buffer/callback/staging ownership and leave
  // the previously compiled executable launchable.
  api_.fail_graph_exec_rebuild_once();
  EXPECT_EQ(hipErrorOutOfMemory,
            api_.graph_exec_memcpy_node_set_params_1d(
                exec, memcpy_node, vmm.reservation, &rebuild_failure_value,
                sizeof(rebuild_failure_value), hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
  actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, vmm.reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(committed_value, actual);

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  DestroyMappedVmm(vmm);
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       CompoundStagedMemcpyAddFailuresLeaveReusableEmptyGraph) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipCtx_t context_a = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_a, /*flags=*/0, device));
  constexpr size_t kElementCount = 32;
  constexpr size_t kBytes = kElementCount * sizeof(uint32_t);
  void* device_a = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&device_a, kBytes));

  uint32_t host_input[kElementCount];
  uint32_t host_output[kElementCount];
  for (size_t i = 0; i < kElementCount; ++i) {
    host_input[i] = static_cast<uint32_t>(0x12000000u + i);
    host_output[i] = 0;
  }

  {
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/3,
        /*expected_committed_user_reference_count=*/0, context_a, nullptr,
        device_a, nullptr,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node_1d(
              out_node, graph, /*dependencies=*/nullptr,
              /*dependency_count=*/0, device_a, host_input, kBytes,
              hipMemcpyHostToDevice);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_a, kBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  }

  {
    for (size_t i = 0; i < kElementCount; ++i) {
      host_input[i] = static_cast<uint32_t>(0x34000000u + i);
      host_output[i] = 0;
    }
    ASSERT_EQ(hipSuccess,
              api_.memcpy(device_a, host_input, kBytes, hipMemcpyHostToDevice));
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/3,
        /*expected_committed_user_reference_count=*/0, context_a, nullptr,
        device_a, nullptr,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node_1d(
              out_node, graph, /*dependencies=*/nullptr,
              /*dependency_count=*/0, host_output, device_a, kBytes,
              hipMemcpyDeviceToHost);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  }

  constexpr size_t kWidth = 4 * sizeof(uint32_t);
  constexpr size_t kPitch = 6 * sizeof(uint32_t);
  constexpr size_t kHeight = 3;
  constexpr size_t kPitchedBytes = kPitch * kHeight;
  {
    for (size_t i = 0; i < kElementCount; ++i) {
      host_input[i] = static_cast<uint32_t>(0x56000000u + i);
      host_output[i] = 0;
    }
    hipMemcpy3DParms params = {};
    params.srcPtr = {host_input, kPitch, kWidth, kHeight};
    params.dstPtr = {device_a, kPitch, kWidth, kHeight};
    params.extent = {kWidth, kHeight, 1};
    params.kind = hipMemcpyHostToDevice;
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/4,
        /*expected_committed_user_reference_count=*/0, context_a, nullptr,
        device_a, nullptr,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node(out_node, graph,
                                            /*dependencies=*/nullptr,
                                            /*dependency_count=*/0, &params);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_a, kPitchedBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t row = 0; row < kHeight; ++row) {
      const size_t row_offset = row * kPitch / sizeof(uint32_t);
      for (size_t column = 0; column < kWidth / sizeof(uint32_t); ++column) {
        EXPECT_EQ(host_input[row_offset + column],
                  host_output[row_offset + column]);
      }
    }
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  }

  {
    for (size_t i = 0; i < kElementCount; ++i) {
      host_input[i] = static_cast<uint32_t>(0x78000000u + i);
      host_output[i] = 0;
    }
    ASSERT_EQ(hipSuccess, api_.memcpy(device_a, host_input, kPitchedBytes,
                                      hipMemcpyHostToDevice));
    hipMemcpy3DParms params = {};
    params.srcPtr = {device_a, kPitch, kWidth, kHeight};
    params.dstPtr = {host_output, kPitch, kWidth, kHeight};
    params.extent = {kWidth, kHeight, 1};
    params.kind = hipMemcpyDeviceToHost;
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/3,
        /*expected_committed_user_reference_count=*/0, context_a, nullptr,
        device_a, nullptr,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node(out_node, graph,
                                            /*dependencies=*/nullptr,
                                            /*dependency_count=*/0, &params);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    for (size_t row = 0; row < kHeight; ++row) {
      const size_t row_offset = row * kPitch / sizeof(uint32_t);
      for (size_t column = 0; column < kWidth / sizeof(uint32_t); ++column) {
        EXPECT_EQ(host_input[row_offset + column],
                  host_output[row_offset + column]);
      }
    }
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  }

  hipCtx_t context_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_b, /*flags=*/0, device));
  void* device_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&device_b, kBytes));

  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  for (size_t i = 0; i < kElementCount; ++i) {
    host_input[i] = static_cast<uint32_t>(0x9A000000u + i);
    host_output[i] = 0;
  }
  ASSERT_EQ(hipSuccess,
            api_.memcpy(device_a, host_input, kBytes, hipMemcpyHostToDevice));
  {
    const uint64_t context_a_reference_count =
        api_.context_reference_count(context_a);
    const uint64_t context_b_reference_count =
        api_.context_reference_count(context_b);
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/5,
        /*expected_committed_user_reference_count=*/1, context_a, context_b,
        device_a, device_b,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node_1d(
              out_node, graph, /*dependencies=*/nullptr,
              /*dependency_count=*/0, device_b, device_a, kBytes,
              hipMemcpyDeviceToDevice);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_b, kBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
    EXPECT_EQ(context_a_reference_count,
              api_.context_reference_count(context_a));
    EXPECT_EQ(context_b_reference_count,
              api_.context_reference_count(context_b));
  }

  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
  for (size_t i = 0; i < kElementCount; ++i) {
    host_input[i] = static_cast<uint32_t>(0xBC000000u + i);
    host_output[i] = 0;
  }
  ASSERT_EQ(hipSuccess,
            api_.memcpy(device_b, host_input, kBytes, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  {
    const uint64_t context_a_reference_count =
        api_.context_reference_count(context_a);
    const uint64_t context_b_reference_count =
        api_.context_reference_count(context_b);
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/5,
        /*expected_committed_user_reference_count=*/1, context_a, context_b,
        device_a, device_b,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node_1d(
              out_node, graph, /*dependencies=*/nullptr,
              /*dependency_count=*/0, device_a, device_b, kBytes,
              hipMemcpyDeviceToDevice);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_a, kBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
    EXPECT_EQ(context_a_reference_count,
              api_.context_reference_count(context_a));
    EXPECT_EQ(context_b_reference_count,
              api_.context_reference_count(context_b));
  }

  // With a third context owning the graph, both peer endpoints are remote.
  // The hidden download and visible upload must submit asynchronously in
  // order; every add checkpoint and both executable-setter rollback points
  // preserve the old graph and all context/buffer references.
  hipCtx_t context_c = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_c, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  for (size_t i = 0; i < kElementCount; ++i) {
    host_input[i] = static_cast<uint32_t>(0xDE000000u + i);
    host_output[i] = 0;
  }
  ASSERT_EQ(hipSuccess,
            api_.memcpy(device_a, host_input, kBytes, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_c));
  {
    const uint64_t context_a_reference_count =
        api_.context_reference_count(context_a);
    const uint64_t context_b_reference_count =
        api_.context_reference_count(context_b);
    const uint64_t context_c_reference_count =
        api_.context_reference_count(context_c);
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    hipGraphNode_t node = nullptr;
    ExpectStagedAddFailureAtomic(
        graph, /*checkpoint_count=*/7,
        /*expected_committed_user_reference_count=*/2, context_a, context_b,
        device_a, device_b,
        [&](hipGraphNode_t* out_node) {
          return api_.graph_add_memcpy_node_1d(
              out_node, graph, /*dependencies=*/nullptr,
              /*dependency_count=*/0, device_b, device_a, kBytes,
              hipMemcpyDeviceToDevice);
        },
        &node);
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_b, kBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }

    // Reverse both remote endpoints through the executable setter and prove
    // the two deferred callbacks retain ordering through their shared staging.
    for (size_t i = 0; i < kElementCount; ++i) {
      host_input[i] = static_cast<uint32_t>(0xEF000000u + i);
      host_output[i] = 0;
    }
    ASSERT_EQ(hipSuccess,
              api_.memcpy(device_b, host_input, kBytes, hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_c));
    ASSERT_EQ(hipSuccess, api_.graph_exec_memcpy_node_set_params_1d(
                              exec, node, device_a, device_b, kBytes,
                              hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_a, kBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }

    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_c));
    api_.fail_graph_exec_memcpy_prepare_once();
    EXPECT_EQ(hipErrorOutOfMemory, api_.graph_exec_memcpy_node_set_params_1d(
                                       exec, node, device_b, device_a, kBytes,
                                       hipMemcpyDeviceToDevice));
    api_.fail_graph_exec_rebuild_once();
    EXPECT_EQ(hipErrorOutOfMemory, api_.graph_exec_memcpy_node_set_params_1d(
                                       exec, node, device_b, device_a, kBytes,
                                       hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
    memset(host_output, 0, sizeof(host_output));
    ASSERT_EQ(hipSuccess, api_.memcpy(host_output, device_a, kBytes,
                                      hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kElementCount; ++i) {
      EXPECT_EQ(host_input[i], host_output[i]);
    }

    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_c));
    ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
    EXPECT_EQ(context_a_reference_count,
              api_.context_reference_count(context_a));
    EXPECT_EQ(context_b_reference_count,
              api_.context_reference_count(context_b));
    EXPECT_EQ(context_c_reference_count,
              api_.context_reference_count(context_c));
  }

  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
  ASSERT_EQ(hipSuccess, api_.free(device_b));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_b));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  ASSERT_EQ(hipSuccess, api_.free(device_a));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_a));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_c));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_c));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       PeerTransferSubmissionAndAsyncFailureTerminateExactlyOnce) {
  for (int failure_mode : {1, 2}) {
    SCOPED_TRACE(testing::Message() << "peer failure mode " << failure_mode);
    ASSERT_EQ(hipSuccess, api_.init(0));
    int device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&device));

    hipCtx_t context_a = nullptr;
    hipCtx_t context_b = nullptr;
    ASSERT_EQ(hipSuccess, api_.ctx_create(&context_a, /*flags=*/0, device));
    ASSERT_EQ(hipSuccess, api_.ctx_create(&context_b, /*flags=*/0, device));
    constexpr size_t kBytes = 64;
    void* device_a = nullptr;
    void* device_b = nullptr;
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
    ASSERT_EQ(hipSuccess, api_.malloc(&device_a, kBytes));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
    ASSERT_EQ(hipSuccess, api_.malloc(&device_b, kBytes));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));

    hipGraph_t graph = nullptr;
    hipGraphNode_t node = nullptr;
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
    ASSERT_EQ(hipSuccess, api_.graph_add_memcpy_node_1d(
                              &node, graph, /*dependencies=*/nullptr,
                              /*dependency_count=*/0, device_b, device_a,
                              kBytes, hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess,
              api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

    api_.set_graph_peer_transfer_failure(failure_mode);
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    const hipError_t expected_error =
        failure_mode == 1 ? hipErrorOutOfMemory : hipErrorInvalidContext;
    EXPECT_EQ(expected_error, api_.stream_synchronize(nullptr));

    // Both ownership paths are terminal: the callback owns failure directly
    // before DEFERRED for mode 1, while the remote queue owns and fails the
    // cloned frontier after DEFERRED for mode 2. Teardown proves the exact
    // accepted frontier terminal and must not wait for a second owner or report
    // the already-observed execution error as a cleanup failure.
    EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
    EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
    ASSERT_EQ(hipSuccess, api_.set_device(device));
    PhaseState phase_state;
    phase_state.release_writer.Notify();
    phase_state.release_pre_drain.Notify();
    phase_state.release_before_device_writer.Notify();
    {
      ScopedPhaseObserver observer(api_.set_phase_observer, ObservePhase,
                                   &phase_state);
      EXPECT_EQ(hipSuccess, api_.device_reset());
    }
    EXPECT_GT(
        phase_state.teardown_stream_wait_count.load(std::memory_order_acquire),
        0);
    ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_b));
    ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_a));
    ASSERT_EQ(hipSuccess, api_.set_device(device));
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
  }
}

TEST_F(HipVmmLifecyclePhaseTest,
       ActivePeerTransferSerializesExecMutationAndDestroy) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipCtx_t context_a = nullptr;
  hipCtx_t context_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_a, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_b, /*flags=*/0, device));
  constexpr size_t kBytes = 64;
  void* device_a = nullptr;
  void* device_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  ASSERT_EQ(hipSuccess, api_.malloc(&device_a, kBytes));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
  ASSERT_EQ(hipSuccess, api_.malloc(&device_b, kBytes));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_add_memcpy_node_1d(
                            &node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, device_b, device_a, kBytes,
                            hipMemcpyDeviceToDevice));
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

  {
    PhaseState phase_state;
    phase_state.release_writer.Notify();
    phase_state.release_pre_drain.Notify();
    phase_state.release_before_device_writer.Notify();
    ScopedPhaseObserver observer(api_.set_phase_observer, ObservePhase,
                                 &phase_state);
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    phase_state.graph_peer_transfer_waiting.Wait();

    std::atomic<int> setter_result{hipErrorUnknown};
    std::atomic<bool> setter_finished{false};
    std::thread setter([&] {
      setter_result.store(
          api_.graph_exec_memcpy_node_set_params_1d(
              exec, node, device_b, device_a, kBytes, hipMemcpyDeviceToDevice),
          std::memory_order_release);
      setter_finished.store(true, std::memory_order_release);
    });
    phase_state.graph_exec_active_launch_waiting.Wait();
    EXPECT_FALSE(setter_finished.load(std::memory_order_acquire));
    phase_state.release_graph_peer_transfer.Notify();
    setter.join();
    EXPECT_EQ(hipSuccess, static_cast<hipError_t>(
                              setter_result.load(std::memory_order_acquire)));
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(nullptr));
  }

  {
    PhaseState phase_state;
    phase_state.release_writer.Notify();
    phase_state.release_pre_drain.Notify();
    phase_state.release_before_device_writer.Notify();
    ScopedPhaseObserver observer(api_.set_phase_observer, ObservePhase,
                                 &phase_state);
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    phase_state.graph_peer_transfer_waiting.Wait();

    std::atomic<int> destroy_result{hipErrorUnknown};
    std::atomic<bool> destroy_finished{false};
    std::thread destroyer([&] {
      destroy_result.store(api_.graph_exec_destroy(exec),
                           std::memory_order_release);
      destroy_finished.store(true, std::memory_order_release);
    });
    phase_state.writer_pending.Wait();
    EXPECT_FALSE(destroy_finished.load(std::memory_order_acquire));
    phase_state.release_graph_peer_transfer.Notify();
    destroyer.join();
    EXPECT_EQ(hipSuccess, static_cast<hipError_t>(
                              destroy_result.load(std::memory_order_acquire)));
    exec = nullptr;
  }

  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
  ASSERT_EQ(hipSuccess, api_.free(device_b));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_b));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  ASSERT_EQ(hipSuccess, api_.free(device_a));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_a));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       ActiveTerminalPeerFailureResetWaitsForExactFrontier) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipCtx_t context_a = nullptr;
  hipCtx_t context_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_a, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api_.ctx_create(&context_b, /*flags=*/0, device));
  constexpr size_t kBytes = 64;
  void* device_a = nullptr;
  void* device_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));
  ASSERT_EQ(hipSuccess, api_.malloc(&device_a, kBytes));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_b));
  ASSERT_EQ(hipSuccess, api_.malloc(&device_b, kBytes));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(context_a));

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_add_memcpy_node_1d(
                            &node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, device_b, device_a, kBytes,
                            hipMemcpyDeviceToDevice));
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

  PhaseState phase_state;
  phase_state.release_writer.Notify();
  phase_state.release_pre_drain.Notify();
  phase_state.release_before_device_writer.Notify();
  {
    ScopedPhaseObserver observer(api_.set_phase_observer, ObservePhase,
                                 &phase_state);
    api_.set_graph_peer_transfer_failure(/*mode=*/2);
    ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
    phase_state.graph_peer_transfer_waiting.Wait();
    ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));

    std::atomic<int> reset_result{hipErrorUnknown};
    std::atomic<bool> reset_finished{false};
    std::thread resetter([&] {
      hipError_t result = api_.set_device(device);
      if (result == hipSuccess) result = api_.device_reset();
      reset_result.store(result, std::memory_order_release);
      reset_finished.store(true, std::memory_order_release);
    });
    phase_state.teardown_stream_waiting.Wait();
    EXPECT_FALSE(reset_finished.load(std::memory_order_acquire));
    phase_state.release_graph_peer_transfer.Notify();
    resetter.join();
    EXPECT_EQ(hipSuccess, static_cast<hipError_t>(
                              reset_result.load(std::memory_order_acquire)));
  }

  // Whole-device reset consumes the graph registry only after the accepted
  // frontier above is terminal; the old public handles are no longer live.
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_destroy(exec));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_destroy(graph));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_b));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(context_a));
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  EXPECT_EQ(hipSuccess, api_.hal_deinit());
}

TEST_F(HipVmmLifecyclePhaseTest,
       FailedPrimaryResetConsumesEqualEpochRetiredExplicitContext) {
  ExpectFailedResetCleanupAllowsExplicitTeardown(/*whole_device=*/false);
}

TEST_F(HipVmmLifecyclePhaseTest,
       FailedWholeDeviceResetConsumesAdvancedEpochRetiredExplicitContext) {
  ExpectFailedResetCleanupAllowsExplicitTeardown(/*whole_device=*/true);
}

TEST_F(HipVmmLifecyclePhaseTest,
       WriterClosesAdmissionAndDrainsWithPublishedUnlockedAlias) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  ASSERT_EQ(hipSuccess, api_.set_device(device));
  int supported = 0;
  ASSERT_EQ(hipSuccess,
            api_.device_get_attribute(
                &supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                device));
  if (!supported) GTEST_SKIP() << "device does not support HIP VMM";

  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypePinned;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location = {hipMemLocationTypeDevice, device};
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess,
            api_.mem_get_allocation_granularity(
                &granularity, &properties, hipMemAllocationGranularityMinimum));
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));
  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, device};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  const uint64_t reservation_reference_count_baseline =
      api_.pointer_buffer_reference_count(reservation);
  ASSERT_NE(UINT64_MAX, reservation_reference_count_baseline)
      << "mapped reservation must resolve to a streaming buffer";

  HostState host_state;
  host_state.probe = api_.probe_accepted_access;
  host_state.ptr = reservation;
  host_state.size = sizeof(uint32_t);
  host_state.device = device;
  hipHostNodeParams host_params = {AcceptedHostCallback, &host_state};
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, 0));
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&host_node, graph, nullptr, 0,
                                                 &host_params));
  hipMemsetParams memset_params = {};
  memset_params.dst = reservation;
  memset_params.value = 0x5A;
  memset_params.elementSize = 1;
  memset_params.width = sizeof(uint32_t);
  memset_params.height = 1;
  hipGraphNode_t memset_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_memset_node(&memset_node, graph, &host_node, 1,
                                       &memset_params));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  host_state.entered.Wait();

  PhaseState phase_state;
  ASSERT_EQ(hipSuccess, api_.set_phase_observer(ObservePhase, &phase_state));
  std::atomic<int> revoke_result{hipErrorUnknown};
  descriptor.flags = hipMemAccessFlagsProtNone;
  std::thread writer([&] {
    revoke_result.store(
        api_.mem_set_access(reservation, granularity, &descriptor, 1),
        std::memory_order_release);
  });
  phase_state.writer_pending.Wait();

  std::atomic<bool> reader_done{false};
  std::atomic<int> reader_result{hipErrorUnknown};
  std::thread reader([&] {
    reader_result.store(api_.graph_launch(exec, nullptr),
                        std::memory_order_release);
    reader_done.store(true, std::memory_order_release);
  });
  phase_state.reader_waiting.Wait();
  EXPECT_FALSE(reader_done.load(std::memory_order_acquire));
  phase_state.release_writer.Notify();
  phase_state.access_pre_drain.Wait();
  phase_state.release_pre_drain.Notify();
  host_state.release.Notify();

  writer.join();
  reader.join();
  ASSERT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));
  ASSERT_EQ(hipSuccess, static_cast<hipError_t>(
                            revoke_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipSuccess, static_cast<hipError_t>(host_state.probe_result.load(
                            std::memory_order_acquire)));
  EXPECT_EQ(
      hipErrorInvalidValue,
      static_cast<hipError_t>(reader_result.load(std::memory_order_acquire)));

  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(0x5A5A5A5Au, actual);

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  EXPECT_EQ(reservation_reference_count_baseline,
            api_.pointer_buffer_reference_count(reservation))
      << "graph teardown must release the template and executable MEMSET "
         "destination references";
  descriptor.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
}

}  // namespace
