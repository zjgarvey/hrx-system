// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_H_
#define IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/device_event.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/feedback_channel.h"

typedef struct iree_thread_t iree_thread_t;
typedef struct iree_hal_amdgpu_logical_device_options_t
    iree_hal_amdgpu_logical_device_options_t;
typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;
typedef struct iree_hal_amdgpu_feedback_source_hold_t
    iree_hal_amdgpu_feedback_source_hold_t;
typedef struct iree_hal_amdgpu_feedback_source_batch_t
    iree_hal_amdgpu_feedback_source_batch_t;

enum {
  IREE_HAL_AMDGPU_FEEDBACK_SOURCE_LIST_INLINE_CAPACITY = 8,
};

// Host-only immutable sidecar identifying every executable that can produce
// feedback for a recorded command buffer. Entries are borrowed under the HAL
// command-buffer resource-lifetime contract, including UNRETAINED mode; queue
// admission converts them to independent retained source-owner holds before
// publishing native work.
typedef struct iree_hal_amdgpu_feedback_source_list_t {
  // Borrowed executable pointers in first-recorded order.
  iree_hal_executable_t** values;
  // Number of occupied entries in |values|.
  iree_host_size_t count;
  // Allocated entry capacity of |values|.
  iree_host_size_t capacity;
  // Inline storage used before the distinct source count spills.
  iree_hal_executable_t*
      inline_values[IREE_HAL_AMDGPU_FEEDBACK_SOURCE_LIST_INLINE_CAPACITY];
} iree_hal_amdgpu_feedback_source_list_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_feedback_state_t
//===----------------------------------------------------------------------===//

// Takes ownership of |status| produced by a feedback service thread.
typedef void(IREE_API_PTR* iree_hal_amdgpu_feedback_error_handler_fn_t)(
    void* user_data, iree_status_t status);

// Per-physical-device feedback service state.
typedef struct iree_hal_amdgpu_feedback_device_state_t {
  // Owning logical-device feedback state.
  struct iree_hal_amdgpu_feedback_state_t* parent;

  // Physical device ordinal associated with this feedback channel.
  iree_host_size_t physical_device_ordinal;

  // Serializes drains between the feedback service thread and queue retirement.
  // It protects runner admission and the source-owner ledger only and is never
  // held while a device event sink, error handler, or resource release runs.
  iree_slim_mutex_t drain_mutex;

  // True while one caller owns channel draining and callback claims.
  bool runner_active;

  // True after device teardown has closed new drain/owner admission.
  bool is_stopping;

  // True after malformed channel data or an unmatched source identity makes
  // further packet consumption unsafe.
  bool is_poisoned;

  // Wakes teardown after the active runner and callback claim scope exits.
  iree_notification_t runner_notification;

  // Intrusive channel-visible source-owner holds. Every entry retains the
  // executable owning its source context and is installed before AQL publish.
  iree_hal_amdgpu_feedback_source_hold_t* source_hold_head;

  // Host-owned channel shared with device producers.
  iree_hal_amdgpu_feedback_channel_t channel;

  // Thread draining ready packets from |channel|.
  iree_thread_t* service_thread;

  // HSA signal used to request |service_thread| shutdown.
  hsa_signal_t stop_signal;
} iree_hal_amdgpu_feedback_device_state_t;

// Logical-device feedback state shared by executable and service paths.
typedef struct iree_hal_amdgpu_feedback_state_t {
  // True when feedback support is enabled for the logical device.
  uint32_t is_enabled : 1;

  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for state-owned allocations.
  iree_allocator_t host_allocator;

  // Borrowed HAL device used for event source attribution.
  iree_hal_device_t* device;

  // Borrowed stable device identifier used for event source attribution.
  iree_string_view_t device_id;

  // Programmatic sink receiving decoded feedback events.
  iree_hal_device_event_sink_t event_sink;

  // Policy applied after a valid ASAN report is emitted.
  iree_hal_amdgpu_asan_report_policy_t asan_report_policy;

  // Policy applied after a valid TSAN report is emitted.
  iree_hal_amdgpu_tsan_report_policy_t tsan_report_policy;

  // Number of entries in |device_states|.
  iree_host_size_t device_state_count;

  // Per-physical-device feedback channels and service threads.
  iree_hal_amdgpu_feedback_device_state_t* device_states;

  // Callback consuming service-thread errors. Owns the passed status.
  iree_hal_amdgpu_feedback_error_handler_fn_t error_handler;

  // User data passed to |error_handler|.
  void* error_handler_user_data;
} iree_hal_amdgpu_feedback_state_t;

// Initializes |out_state| from logical-device options.
//
// When feedback is disabled this leaves |out_state| zeroed and returns OK.
// Sanitizers currently enable feedback because report packets need a serviced
// device-to-host transport; other feedback clients can enable the same state
// once they have packet schemas and handlers.
iree_status_t iree_hal_amdgpu_feedback_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_hal_device_t* device, iree_string_view_t device_id,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amdgpu_feedback_error_handler_fn_t error_handler,
    void* error_handler_user_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_feedback_state_t* out_state);

// Stops service threads and releases any feedback resources owned by |state|.
void iree_hal_amdgpu_feedback_state_deinitialize(
    iree_hal_amdgpu_feedback_state_t* state);

// Returns true when |state| owns enabled feedback resources.
bool iree_hal_amdgpu_feedback_state_is_enabled(
    const iree_hal_amdgpu_feedback_state_t* state);

// Initializes an empty borrowed executable sidecar.
void iree_hal_amdgpu_feedback_source_list_initialize(
    iree_hal_amdgpu_feedback_source_list_t* out_list);

// Releases sidecar storage without releasing borrowed executable entries.
void iree_hal_amdgpu_feedback_source_list_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_feedback_source_list_t* list);

// Inserts |executable| if not already present. The sidecar never retains it.
iree_status_t iree_hal_amdgpu_feedback_source_list_insert(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_feedback_source_list_t* list,
    iree_hal_executable_t* executable);

// Allocates a feedback source-owner batch and retains each distinct AMDGPU
// executable in |executables|. The batch is private until explicitly installed
// and may be cancelled without touching the channel ledger.
iree_status_t iree_hal_amdgpu_feedback_source_batch_prepare(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal, iree_host_size_t executable_count,
    iree_hal_executable_t* const* executables,
    iree_hal_amdgpu_feedback_source_batch_t** out_batch);

// Atomically installs every hold in |batch| with one acquired lower
// reservation bound. This is infallible and must run after all fallible packet
// construction but before the first packet header/doorbell is published.
void iree_hal_amdgpu_feedback_source_batch_install(
    iree_hal_amdgpu_feedback_source_batch_t* batch);

// Releases an uninstalled batch. Once installed, ownership belongs to the
// device-stable feedback ledger and cancellation is forbidden.
void iree_hal_amdgpu_feedback_source_batch_cancel(
    iree_hal_amdgpu_feedback_source_batch_t* batch);

// Closes an installed batch after the associated queue epoch has been observed
// with an HSA system acquire. Captures the channel reservation head as the
// exclusive upper sequence bound and wakes the device feedback service for
// eventual retirement. Queue failure without an HSA acquire must leave the
// batch OPEN for device teardown to release.
void iree_hal_amdgpu_feedback_source_batch_close(
    iree_hal_amdgpu_feedback_source_batch_t* batch);

// Drains ready packets from the channel for |physical_device_ordinal|.
//
// This reports channel or packet handling errors through the state error
// handler. It may be called by the feedback service thread after a device
// notification or by host queue retirement before queue-owned resources and
// signal semaphores are released.
void iree_hal_amdgpu_feedback_state_drain_physical_device(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal);

// Populates |out_config| with the device-visible feedback configuration.
//
// Callers must only call this when feedback is enabled.
// |physical_device_ordinal| selects the physical device whose executable global
// is being assigned.
iree_status_t iree_hal_amdgpu_feedback_state_populate_config(
    const iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_feedback_config_t* out_config);

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
typedef enum iree_hal_amdgpu_feedback_test_phase_e {
  IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_BATCH_INSTALLED = 0,
  IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_TARGET_CLOSED = 1,
  IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_CLAIM_ENTERED = 2,
  IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_CLAIM_RETURNED = 3,
  IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_BATCH_RETIRED = 4,
} iree_hal_amdgpu_feedback_test_phase_t;

typedef void(IREE_API_PTR* iree_hal_amdgpu_feedback_test_phase_observer_t)(
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_feedback_test_phase_t phase, uint64_t source_identity,
    uint64_t sequence, void* user_data);

// Test-only observer compiled exclusively into the instrumented companion.
void iree_hal_amdgpu_feedback_test_set_phase_observer(
    iree_hal_amdgpu_feedback_test_phase_observer_t observer, void* user_data);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_H_
