// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/vmm.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/context.h"
#include "common/graph.h"
#include "common/internal.h"
#include "common/memory.h"
#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
typedef struct iree_hip_vmm_allocation_t iree_hip_vmm_allocation_t;
typedef struct iree_hip_vmm_teardown_plan_t iree_hip_vmm_teardown_plan_t;

typedef enum iree_hip_vmm_allocation_ownership_e {
  // The physical allocation is retired by reset of its creating device.
  IREE_HIP_VMM_ALLOCATION_OWNERSHIP_DEVICE = 0,
} iree_hip_vmm_allocation_ownership_t;

typedef struct iree_hip_vmm_mapping_t {
  // Offset within the virtual reservation in bytes.
  size_t virtual_offset;
  // Offset within the physical allocation in bytes.
  size_t physical_offset;
  // Length of this mapping in bytes.
  size_t size;
  // Physical allocation retained by the mapping registry.
  iree_hip_vmm_allocation_t* allocation;
} iree_hip_vmm_mapping_t;

typedef struct iree_hip_vmm_access_range_t {
  // HIP location class whose permissions this record describes.
  hipMemLocationType location_type;
  // HIP location ordinal within |location_type|.
  int location_id;
  // Offset within the virtual reservation in bytes.
  size_t offset;
  // Length of the permission range in bytes.
  size_t size;
  // Permissions last applied to this range.
  hipMemAccessFlags flags;
  // Never-reused identity of this device-scoped access grant. Every
  // context-local alias realizing the grant carries this same identity.
  uint64_t capability_id;
} iree_hip_vmm_access_range_t;

typedef struct iree_hip_vmm_access_binding_t {
  hipMemLocationType location_type;
  int location_id;
  size_t offset;
  size_t size;
  hipMemAccessFlags flags;
  uint64_t generation;
  uint64_t target_device_epoch;
  uint64_t capability_id;
  iree_hal_streaming_buffer_t* streaming_buffer;
} iree_hip_vmm_access_binding_t;

typedef struct iree_hip_vmm_reservation_t {
  // Registry and in-flight operation references.
  iree_atomic_ref_count_t ref_count;
  // Serializes mapping and access-state mutation for this reservation.
  iree_slim_mutex_t mutex;
  // Device retained while allocator-owned reservation state is live.
  hrx_device_t device;
  // Borrowed allocator owned by |device|.
  hrx_allocator_t allocator;
  // HRX virtual address reservation.
  hrx_buffer_t virtual_buffer;
  // Exact context that created the reservation, retained for process-global
  // metadata until reset or explicit release.
  iree_hal_streaming_context_t* owner_context;
  // Runtime generation and reservation-owner device epoch.
  uint64_t generation;
  uint64_t owner_device_epoch;
  // Serial that invalidates access wrappers staged across native calls.
  uint64_t mutation_serial;
  // True when native rollback could not prove the old permission state.
  bool poisoned;
  // Base device address returned to HIP callers.
  uintptr_t base_address;
  // Total reservation length in bytes.
  size_t size;
  // Minimum page size for mapping and protection operations.
  size_t granularity;
  // Device ordinal owning the virtual address space.
  int device_ordinal;
  // True after removal from the process registry begins.
  bool retiring;
  // Sorted, non-overlapping physical mappings.
  iree_hip_vmm_mapping_t* mappings;
  // Number of live entries in |mappings|.
  size_t mapping_count;
  // Allocated entry capacity of |mappings|.
  size_t mapping_capacity;
  // Exact ROCr mapping units. These may split one logical mapping at access
  // boundaries because ROCr unmap/protect operations require exact map units.
  iree_hip_vmm_mapping_t* native_segments;
  // Number of confirmed live entries in |native_segments|.
  size_t native_segment_count;
  // Allocated entry capacity of |native_segments|.
  size_t native_segment_capacity;
  // Sorted, normalized access ranges.
  iree_hip_vmm_access_range_t* access_ranges;
  // Number of live entries in |access_ranges|.
  size_t access_range_count;
  // Exact-context operational wrappers for non-ProtNone device access ranges.
  iree_hip_vmm_access_binding_t* access_bindings;
  // Number of live entries in |access_bindings|.
  size_t access_binding_count;
} iree_hip_vmm_reservation_t;

struct iree_hip_vmm_allocation_t {
  // Registry and in-flight operation references.
  iree_atomic_ref_count_t ref_count;
  // Serializes public and mapping reference transitions.
  iree_slim_mutex_t mutex;
  // Device retained while physical memory is live.
  hrx_device_t device;
  // Device ordinal whose allocator owns the physical handle.
  int device_ordinal;
  // Runtime generation and physical-owner device epoch.
  uint64_t generation;
  uint64_t owner_device_epoch;
  // Context coherently reported with the physical owner device ordinal.
  iree_hal_streaming_context_t* owner_context;
  // Explicit policy selecting the lifecycle transaction that owns retirement.
  iree_hip_vmm_allocation_ownership_t ownership;
  // Borrowed allocator owned by |device|.
  hrx_allocator_t allocator;
  // Allocator-owned physical memory handle.
  hrx_physical_memory_t physical_memory;
  // Properties reported through the HIP handle query API.
  hipMemAllocationProp properties;
  // Physical allocation length in bytes.
  size_t size;
  // Minimum aligned subrange accepted by the physical owner's backend.
  size_t granularity;
  // Monotonic opaque key exposed as the HIP handle value.
  uintptr_t handle_key;
  // Stable public HIP pointer BUFFER_ID width and identity.
  uint32_t pointer_attribute_buffer_id;
  // Number of public HIP handle references.
  uint64_t public_reference_count;
  // Number of mapping records retaining this allocation.
  uint64_t mapping_reference_count;
  // True while the final native free is in progress.
  bool retiring;
};

typedef enum iree_hip_vmm_teardown_scope_e {
  IREE_HIP_VMM_TEARDOWN_EXPLICIT_CONTEXT_ALIASES_ONLY = 0,
  IREE_HIP_VMM_TEARDOWN_PRIMARY_CONTEXT = 1,
  IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT = 2,
  IREE_HIP_VMM_TEARDOWN_PROCESS_ALL = 3,
} iree_hip_vmm_teardown_scope_t;

typedef enum iree_hip_vmm_teardown_state_e {
  IREE_HIP_VMM_TEARDOWN_PREPARED = 0,
  IREE_HIP_VMM_TEARDOWN_SEALED = 1,
  IREE_HIP_VMM_TEARDOWN_PUBLIC_COMMIT = 2,
  IREE_HIP_VMM_TEARDOWN_NATIVE = 3,
  IREE_HIP_VMM_TEARDOWN_FAILED_CLOSED = 4,
  IREE_HIP_VMM_TEARDOWN_FINALIZING = 5,
  IREE_HIP_VMM_TEARDOWN_COMPLETE = 6,
} iree_hip_vmm_teardown_state_t;

typedef struct iree_hip_vmm_reservation_plan_t {
  // Exact registry object pinned through external finalization.
  iree_hip_vmm_reservation_t* reservation;
  uint64_t expected_mutation_serial;
  bool affected;
  bool retire;

  // Exact PREPARED graph used for post-wait revalidation.
  iree_hip_vmm_mapping_t* expected_mappings;
  size_t expected_mapping_count;
  iree_hip_vmm_mapping_t* expected_native_segments;
  size_t expected_native_segment_count;
  iree_hip_vmm_access_range_t* expected_access_ranges;
  size_t expected_access_range_count;
  iree_hip_vmm_access_binding_t* expected_access_bindings;
  size_t expected_access_binding_count;

  // Fully allocated graph to install after the native journal completes.
  iree_hip_vmm_mapping_t* final_mappings;
  size_t final_mapping_count;
  iree_hip_vmm_mapping_t* final_native_segments;
  size_t final_native_segment_count;
  iree_hip_vmm_access_range_t* final_access_ranges;
  size_t final_access_range_count;
  iree_hip_vmm_access_binding_t* final_access_bindings;
  size_t final_access_binding_count;
  // True when final bindings are newly prepared aliases rather than shallow
  // ownership transfers from the expected array.
  bool final_bindings_are_new;

  // Exact wrapper ownership removed at PUBLIC_COMMIT. Entries are shallow
  // copies until finalization transfers survivor ownership to final_bindings.
  iree_hip_vmm_access_binding_t* removed_bindings;
  size_t removed_binding_count;
  size_t removed_binding_reserved_count;
} iree_hip_vmm_reservation_plan_t;

typedef struct iree_hip_vmm_allocation_plan_t {
  // Exact registry object pinned through external finalization.
  iree_hip_vmm_allocation_t* allocation;
  uint64_t expected_public_reference_count;
  uint64_t expected_mapping_reference_count;
  hrx_physical_memory_t expected_physical_memory;
  uint64_t removed_mapping_reference_count;
  bool retire;
} iree_hip_vmm_allocation_plan_t;

struct iree_hip_vmm_teardown_plan_t {
  iree_hip_vmm_teardown_scope_t scope;
  iree_hip_vmm_teardown_state_t state;
  uint64_t generation;
  int target_device_ordinal;
  iree_hal_streaming_context_t* target_context;

  bool advances_device_epoch;
  uint64_t expected_vmm_device_epoch;
  uint64_t next_vmm_device_epoch;
  uint64_t expected_common_device_epoch;
  uint64_t next_common_device_epoch;

  iree_hip_vmm_reservation_plan_t* reservations;
  size_t reservation_count;
  iree_hip_vmm_allocation_plan_t* allocations;
  size_t allocation_count;

  // Exact reader set derived from every affected graph edge. Foreign readers
  // are synchronized during PREPARED but never added to the permanent seal
  // set owned by the caller's context-teardown token.
  iree_hal_streaming_context_t** incident_contexts;
  size_t incident_context_count;
  size_t incident_context_capacity;

  // Dependency-ordered raw libhsa journal. Only this single executor mutates
  // cursor/state; retained slots merely make an inactive retry discoverable.
  hrx_vmm_native_operation_t* operations;
  size_t operation_count;
  size_t cursor;
  bool has_first_error;
  hrx_vmm_native_status_t first_error;
};

typedef struct iree_hip_vmm_registry_t {
  // Serializes lifecycle writers in arrival order.
  iree_slim_mutex_t transition_mutex;
  // Protects lifecycle admission state only.
  iree_slim_mutex_t lifecycle_state_mutex;
  // Wakes readers after a writer reopens admission and writers after drain.
  iree_notification_t lifecycle_notification;
  // True after a writer closes admission and until its transaction completes.
  bool writer_pending;
  // Writers announce before waiting for transition ownership so readers
  // cannot reopen in the handoff gap between consecutive writers.
  size_t writer_waiter_count;
  // Admission state to restore when a writer aborts before its commit point.
  bool writer_started_active;
  // Number of top-level admitted reader threads.
  size_t reader_count;
  // Guards registry vectors and handle generation only.
  iree_slim_mutex_t mutex;
  // True while one initialized HIP runtime generation may accept VMM calls.
  bool active;
  // Monotonic process runtime generation.
  uint64_t generation;
  // Per-device reset epochs for the active generation.
  uint64_t* device_epochs;
  // Number of entries in |device_epochs|.
  size_t device_epoch_count;
  // Reservations sorted by base device address.
  iree_hip_vmm_reservation_t** reservations;
  // Number of entries in |reservations|.
  size_t reservation_count;
  // Allocated entry capacity of |reservations|.
  size_t reservation_capacity;
  // Slots preallocated by in-flight creators before native acquisition.
  size_t pending_reservation_slots;
  // Allocations sorted by monotonic handle key.
  iree_hip_vmm_allocation_t** allocations;
  // Number of entries in |allocations|.
  size_t allocation_count;
  // Allocated entry capacity of |allocations|.
  size_t allocation_capacity;
  // Slots preallocated by in-flight creators before native acquisition.
  size_t pending_allocation_slots;
  // Next never-reused opaque handle key.
  uintptr_t next_handle_key;
  // Next never-reused operational access capability identity.
  iree_atomic_uint64_t next_capability_id;
  // Retryable plan published after complete graph revalidation and before the
  // first queue seal.
  iree_hip_vmm_teardown_plan_t* prepared_cleanup;
  // Fail-closed plan retaining the exact first incomplete native ordinal.
  iree_hip_vmm_teardown_plan_t* failed_cleanup;
} iree_hip_vmm_registry_t;

static iree_once_flag iree_hip_vmm_registry_once = IREE_ONCE_FLAG_INIT;
static iree_hip_vmm_registry_t iree_hip_vmm_registry;
static IREE_THREAD_LOCAL uint32_t iree_hip_vmm_reader_depth = 0;
static IREE_THREAD_LOCAL uint32_t iree_hip_vmm_native_callback_window_depth = 0;
static void iree_hip_vmm_registry_initialize(void);
static iree_hip_vmm_registry_t* iree_hip_vmm_registry_lock(void);
static void iree_hip_vmm_registry_unlock(void);
static hipError_t iree_hip_vmm_from_hrx_status(hrx_status_t status);
static hipError_t iree_hip_vmm_from_iree_status(iree_status_t status);

#if defined(IREE_HIP_VMM_TESTING)
enum {
  IREE_HIP_VMM_TEST_PHASE_WRITER_PENDING = 1,
  IREE_HIP_VMM_TEST_PHASE_READER_WAITING = 2,
  IREE_HIP_VMM_TEST_PHASE_ACCESS_PRE_DRAIN = 3,
  IREE_HIP_VMM_TEST_PHASE_BEFORE_DEVICE_WRITER = 4,
  IREE_HIP_VMM_TEST_PHASE_CONTEXT_DESTROY_PINNED = 5,
  IREE_HIP_VMM_TEST_PHASE_EXECUTION_CONTEXT_QUEUE_RELEASE = 6,
  IREE_HIP_VMM_TEST_PHASE_EXECUTION_CONTEXT_TAKEN = 7,
  IREE_HIP_VMM_TEST_PHASE_BORROWED_DEVICE_RESOLVED = 8,
  IREE_HIP_VMM_TEST_PHASE_PRIMARY_RELEASE_PREPARE = 9,
  IREE_HIP_VMM_TEST_PHASE_WRITER_QUEUED = 10,
  IREE_HIP_VMM_TEST_PHASE_EXPLICIT_CONTEXT_QUEUE_RELEASE = 11,
  IREE_HIP_VMM_TEST_PHASE_EXPLICIT_CONTEXT_UNREGISTERED = 12,
  IREE_HIP_VMM_TEST_PHASE_ALLOCATION_OUTPUT = 13,
  IREE_HIP_VMM_TEST_PHASE_GRAPH_MUTATION_ADMITTED = 14,
  IREE_HIP_VMM_TEST_PHASE_CONTEXT_RESOLVED = 15,
  IREE_HIP_VMM_TEST_PHASE_NATIVE_CALLBACK_WINDOW = 16,
  IREE_HIP_VMM_TEST_PHASE_ADDRESS_FREE_CALLBACK_WINDOW = 17,
  IREE_HIP_VMM_TEST_PHASE_GRAPH_EXEC_ACTIVE_LAUNCH_WAITING = 18,
  IREE_HIP_VMM_TEST_PHASE_GRAPH_PEER_TRANSFER_WAITING = 19,
  IREE_HIP_VMM_TEST_PHASE_TEARDOWN_STREAM_WAITING = 20,
  IREE_HIP_VMM_TEST_PHASE_POOL_API_ADMITTED = 21,
  IREE_HIP_VMM_TEST_PHASE_ALLOCATION_RETIRE_RESERVED = 22,
  IREE_HIP_VMM_TEST_PHASE_INCIDENT_CONTEXT_STREAM_WAITING = 23,
};
typedef void (*iree_hip_vmm_test_phase_observer_t)(int phase, void* object,
                                                   void* user_data);
typedef enum iree_hip_vmm_test_phase_observer_state_e {
  IREE_HIP_VMM_TEST_PHASE_OBSERVER_EMPTY = 0,
  IREE_HIP_VMM_TEST_PHASE_OBSERVER_INSTALLING = 1,
  IREE_HIP_VMM_TEST_PHASE_OBSERVER_ACTIVE = 2,
  IREE_HIP_VMM_TEST_PHASE_OBSERVER_CLOSING = 3,
} iree_hip_vmm_test_phase_observer_state_t;
typedef struct iree_hip_vmm_test_phase_observer_registry_t {
  // Serializes registration state and callback acquisition.
  iree_slim_mutex_t mutex;
  // Wakes clear after the final acquired callback completes.
  iree_notification_t notification;
  // Registration state controlling whether new callbacks may be acquired.
  iree_hip_vmm_test_phase_observer_state_t state;
  // Callback published while |state| is ACTIVE or CLOSING.
  iree_hip_vmm_test_phase_observer_t observer;
  // User data whose lifetime is protected by |in_flight|.
  void* user_data;
  // Number of callbacks holding a registry snapshot.
  size_t in_flight;
} iree_hip_vmm_test_phase_observer_registry_t;
typedef struct iree_hip_vmm_test_phase_observer_snapshot_t {
  // Acquired callback, or NULL when observation is inactive.
  iree_hip_vmm_test_phase_observer_t observer;
  // User data paired atomically with |observer|.
  void* user_data;
} iree_hip_vmm_test_phase_observer_snapshot_t;
static iree_once_flag iree_hip_vmm_test_phase_observer_registry_once =
    IREE_ONCE_FLAG_INIT;
static iree_hip_vmm_test_phase_observer_registry_t
    iree_hip_vmm_test_phase_observer_registry;
static IREE_THREAD_LOCAL uint32_t
    iree_hip_vmm_test_phase_observer_callback_depth;
static iree_atomic_int32_t iree_hip_vmm_test_fail_green_post_retain =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_primary_release_prepare =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t
    iree_hip_vmm_test_fail_binding_rollback_prepare_ordinal =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_plan_allocation_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_plan_allocation_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_plan_revalidation =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_native_operation_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_native_operation_attempts[64];
static iree_atomic_int64_t iree_hip_vmm_test_native_operation_count_value =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int64_t iree_hip_vmm_test_native_operation_cursor_value =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int64_t iree_hip_vmm_test_first_native_error_value =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int64_t iree_hip_vmm_test_raw_native_call_count_value =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_access_preflight_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_address_reserve_after_native =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_physical_create_after_native =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_graph_exec_memcpy_prepare =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_graph_exec_rebuild =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_fail_graph_add_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_graph_add_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_graph_peer_transfer_failure =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hip_vmm_test_access_preflight_ordinal =
    IREE_ATOMIC_VAR_INIT(0);
static bool iree_hip_vmm_test_epoch_is_forced = false;
static bool iree_hip_vmm_test_epoch_is_applied = false;
static int iree_hip_vmm_test_forced_epoch_source = -1;
static int iree_hip_vmm_test_forced_epoch_device = -1;
static uint64_t iree_hip_vmm_test_saved_epoch = 0;

static void iree_hip_vmm_test_native_address_free_observer(bool entering,
                                                           void* address,
                                                           size_t size,
                                                           void* user_data);

static void iree_hip_vmm_test_phase_observer_registry_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_vmm_test_phase_observer_registry.mutex);
  iree_notification_initialize(
      &iree_hip_vmm_test_phase_observer_registry.notification);
}

static bool iree_hip_vmm_test_phase_observer_registry_is_drained(
    void* user_data) {
  iree_hip_vmm_test_phase_observer_registry_t* registry =
      (iree_hip_vmm_test_phase_observer_registry_t*)user_data;
  iree_slim_mutex_lock(&registry->mutex);
  const bool is_drained = registry->in_flight == 0;
  iree_slim_mutex_unlock(&registry->mutex);
  return is_drained;
}

static bool iree_hip_vmm_test_phase_observer_registry_is_closing(
    void* user_data) {
  iree_hip_vmm_test_phase_observer_registry_t* registry =
      (iree_hip_vmm_test_phase_observer_registry_t*)user_data;
  iree_slim_mutex_lock(&registry->mutex);
  const bool is_closing =
      registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_CLOSING;
  iree_slim_mutex_unlock(&registry->mutex);
  return is_closing;
}

HIPAPI hipError_t iree_hip_vmm_test_wait_phase_observer_closing(void) {
  iree_call_once(&iree_hip_vmm_test_phase_observer_registry_once,
                 iree_hip_vmm_test_phase_observer_registry_initialize);
  iree_notification_await(
      &iree_hip_vmm_test_phase_observer_registry.notification,
      iree_hip_vmm_test_phase_observer_registry_is_closing,
      &iree_hip_vmm_test_phase_observer_registry, iree_infinite_timeout());
  return hipSuccess;
}

static hipError_t iree_hip_vmm_test_observer_status_to_hip(
    iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;
  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      return hipErrorInvalidValue;
    case IREE_STATUS_ALREADY_EXISTS:
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorIllegalState;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    default:
      return hipErrorUnknown;
  }
}

HIPAPI hipError_t iree_hip_vmm_test_set_lower_address_free_observer(
    iree_hip_vmm_test_lower_address_free_observer_t observer, void* user_data) {
  return iree_hip_vmm_test_observer_status_to_hip(
      iree_hal_amdgpu_libhsa_set_vmem_address_free_observer(observer,
                                                            user_data));
}

HIPAPI hipError_t iree_hip_vmm_test_clear_lower_address_free_observer(
    iree_hip_vmm_test_lower_address_free_observer_t observer, void* user_data) {
  return iree_hip_vmm_test_observer_status_to_hip(
      iree_hal_amdgpu_libhsa_clear_vmem_address_free_observer(observer,
                                                              user_data));
}

HIPAPI void iree_hip_vmm_test_fail_next_lower_observer_clear(void) {
  iree_hal_amdgpu_libhsa_test_fail_next_observer_clear();
}

HIPAPI hipError_t iree_hip_vmm_test_set_phase_observer(
    iree_hip_vmm_test_phase_observer_t observer, void* user_data) {
  if (iree_hip_vmm_test_phase_observer_callback_depth != 0) {
    return hipErrorIllegalState;
  }
  iree_call_once(&iree_hip_vmm_test_phase_observer_registry_once,
                 iree_hip_vmm_test_phase_observer_registry_initialize);
  iree_hip_vmm_test_phase_observer_registry_t* registry =
      &iree_hip_vmm_test_phase_observer_registry;
  if (observer) {
    iree_slim_mutex_lock(&registry->mutex);
    if (registry->state != IREE_HIP_VMM_TEST_PHASE_OBSERVER_EMPTY) {
      iree_slim_mutex_unlock(&registry->mutex);
      return hipErrorIllegalState;
    }
    registry->state = IREE_HIP_VMM_TEST_PHASE_OBSERVER_INSTALLING;
    iree_slim_mutex_unlock(&registry->mutex);

    iree_status_t status =
        iree_hal_amdgpu_libhsa_set_vmem_address_free_observer(
            iree_hip_vmm_test_native_address_free_observer,
            /*user_data=*/NULL);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&registry->mutex);
      IREE_ASSERT(registry->state ==
                  IREE_HIP_VMM_TEST_PHASE_OBSERVER_INSTALLING);
      registry->state = IREE_HIP_VMM_TEST_PHASE_OBSERVER_EMPTY;
      iree_slim_mutex_unlock(&registry->mutex);
      return iree_hip_vmm_test_observer_status_to_hip(status);
    }

    iree_slim_mutex_lock(&registry->mutex);
    IREE_ASSERT(registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_INSTALLING);
    registry->observer = observer;
    registry->user_data = user_data;
    registry->state = IREE_HIP_VMM_TEST_PHASE_OBSERVER_ACTIVE;
    iree_slim_mutex_unlock(&registry->mutex);
    return hipSuccess;
  }

  iree_slim_mutex_lock(&registry->mutex);
  if (registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_EMPTY) {
    iree_slim_mutex_unlock(&registry->mutex);
    return hipSuccess;
  }
  if (registry->state != IREE_HIP_VMM_TEST_PHASE_OBSERVER_ACTIVE) {
    iree_slim_mutex_unlock(&registry->mutex);
    return hipErrorIllegalState;
  }
  registry->state = IREE_HIP_VMM_TEST_PHASE_OBSERVER_CLOSING;
  iree_slim_mutex_unlock(&registry->mutex);
  iree_notification_post(&registry->notification, IREE_ALL_WAITERS);

  iree_status_t status =
      iree_hal_amdgpu_libhsa_clear_vmem_address_free_observer(
          iree_hip_vmm_test_native_address_free_observer,
          /*user_data=*/NULL);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&registry->mutex);
    IREE_ASSERT(registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_CLOSING);
    registry->state = IREE_HIP_VMM_TEST_PHASE_OBSERVER_ACTIVE;
    iree_slim_mutex_unlock(&registry->mutex);
    return iree_hip_vmm_test_observer_status_to_hip(status);
  }

  iree_notification_await(&registry->notification,
                          iree_hip_vmm_test_phase_observer_registry_is_drained,
                          registry, iree_infinite_timeout());
  iree_slim_mutex_lock(&registry->mutex);
  IREE_ASSERT(registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_CLOSING);
  IREE_ASSERT(registry->in_flight == 0);
  registry->observer = NULL;
  registry->user_data = NULL;
  registry->state = IREE_HIP_VMM_TEST_PHASE_OBSERVER_EMPTY;
  iree_slim_mutex_unlock(&registry->mutex);
  return hipSuccess;
}

HIPAPI void iree_hip_vmm_test_fail_green_context_after_retain_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_green_post_retain, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_primary_release_prepare_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_primary_release_prepare, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_binding_rollback_prepare_at(
    int binding_ordinal) {
  IREE_ASSERT(binding_ordinal > 0);
  iree_atomic_store(&iree_hip_vmm_test_fail_binding_rollback_prepare_ordinal,
                    binding_ordinal, iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_plan_allocation_at(int allocation_ordinal) {
  IREE_ASSERT(allocation_ordinal > 0);
  iree_atomic_store(&iree_hip_vmm_test_plan_allocation_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_plan_allocation_ordinal,
                    allocation_ordinal, iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_plan_revalidation_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_plan_revalidation, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_native_operation_at(int operation_ordinal) {
  IREE_ASSERT(operation_ordinal > 0 && operation_ordinal <= 64);
  for (size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hip_vmm_test_native_operation_attempts); ++i) {
    iree_atomic_store(&iree_hip_vmm_test_native_operation_attempts[i], 0,
                      iree_memory_order_release);
  }
  iree_atomic_store(&iree_hip_vmm_test_native_operation_count_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_native_operation_cursor_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_first_native_error_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_raw_native_call_count_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_native_operation_ordinal,
                    operation_ordinal, iree_memory_order_release);
}

HIPAPI uint64_t iree_hip_vmm_test_native_operation_count(void) {
  return (uint64_t)iree_atomic_load(
      &iree_hip_vmm_test_native_operation_count_value,
      iree_memory_order_acquire);
}

HIPAPI uint64_t iree_hip_vmm_test_native_operation_cursor(void) {
  return (uint64_t)iree_atomic_load(
      &iree_hip_vmm_test_native_operation_cursor_value,
      iree_memory_order_acquire);
}

HIPAPI uint64_t
iree_hip_vmm_test_native_operation_attempt_count(int operation_ordinal) {
  if (operation_ordinal <= 0 || operation_ordinal > 64) return 0;
  return (uint64_t)iree_atomic_load(
      &iree_hip_vmm_test_native_operation_attempts[operation_ordinal - 1],
      iree_memory_order_acquire);
}

HIPAPI uint32_t iree_hip_vmm_test_first_native_error(void) {
  return (uint32_t)iree_atomic_load(&iree_hip_vmm_test_first_native_error_value,
                                    iree_memory_order_acquire);
}

HIPAPI uint64_t iree_hip_vmm_test_raw_native_call_count(void) {
  return (uint64_t)iree_atomic_load(
      &iree_hip_vmm_test_raw_native_call_count_value,
      iree_memory_order_acquire);
}

HIPAPI hipError_t iree_hip_vmm_test_query_allocation_registry(
    size_t* out_count, size_t* out_pending, size_t* out_capacity) {
  if (!out_count || !out_pending || !out_capacity) {
    return hipErrorInvalidValue;
  }
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  *out_count = registry->allocation_count;
  *out_pending = registry->pending_allocation_slots;
  *out_capacity = registry->allocation_capacity;
  iree_hip_vmm_registry_unlock();
  return hipSuccess;
}

HIPAPI void iree_hip_vmm_test_fail_address_reserve_after_native_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_address_reserve_after_native, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_physical_create_after_native_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_physical_create_after_native, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_lower_virtual_reserve_after_native_once(
    void) {
  hrx_allocator_test_fail_lower_virtual_reserve_after_native_once();
}

HIPAPI void iree_hip_vmm_test_fail_hrx_virtual_reserve_after_native_once(void) {
  hrx_allocator_test_fail_virtual_reserve_after_hal_native_once();
}

HIPAPI uint64_t iree_hip_vmm_test_last_virtual_reserve_address(void) {
  return hrx_allocator_test_last_virtual_reserve_address();
}

HIPAPI uint64_t iree_hip_vmm_test_last_virtual_reserve_alignment(void) {
  return hrx_allocator_test_last_virtual_reserve_alignment();
}

HIPAPI void iree_hip_vmm_test_fail_cleanup_count(bool physical_memory,
                                                 int failure_count) {
  hrx_allocator_test_fail_lower_cleanup_count(physical_memory, failure_count);
}

HIPAPI uint64_t iree_hip_vmm_test_quarantine_count(void) {
  return hrx_allocator_test_vmm_quarantine_count();
}

HIPAPI uint64_t iree_hip_vmm_test_cleanup_attempt_count(bool physical_memory) {
  return hrx_allocator_test_vmm_cleanup_attempt_count(physical_memory);
}

HIPAPI uint32_t iree_hip_vmm_test_last_cleanup_status(bool physical_memory) {
  return hrx_allocator_test_vmm_last_cleanup_status(physical_memory);
}

HIPAPI void iree_hip_vmm_test_arm_quarantine_drain_pause(void) {
  hrx_allocator_test_arm_vmm_quarantine_drain_pause();
}

HIPAPI void iree_hip_vmm_test_wait_quarantine_drain_paused(void) {
  hrx_allocator_test_wait_vmm_quarantine_drain_paused();
}

HIPAPI void iree_hip_vmm_test_release_quarantine_drain_pause(void) {
  hrx_allocator_test_release_vmm_quarantine_drain_pause();
}

HIPAPI int iree_hip_vmm_test_quarantine_drain_attempt_kind(
    int attempt_ordinal) {
  return hrx_allocator_test_vmm_quarantine_drain_attempt_kind(attempt_ordinal);
}

HIPAPI void iree_hip_vmm_test_fail_access_preflight_at(int access_ordinal) {
  IREE_ASSERT(access_ordinal > 0);
  iree_atomic_store(&iree_hip_vmm_test_access_preflight_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_raw_native_call_count_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_access_preflight_ordinal,
                    access_ordinal, iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_graph_exec_memcpy_prepare_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_graph_exec_memcpy_prepare, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_graph_exec_rebuild_once(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_graph_exec_rebuild, 1,
                    iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_fail_graph_add_at(int operation_ordinal) {
  IREE_ASSERT(operation_ordinal > 0);
  iree_atomic_store(&iree_hip_vmm_test_graph_add_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_graph_add_ordinal,
                    operation_ordinal, iree_memory_order_release);
}

HIPAPI void iree_hip_vmm_test_set_graph_peer_transfer_failure(int mode) {
  IREE_ASSERT(mode >= 0 && mode <= 2);
  iree_atomic_store(&iree_hip_vmm_test_graph_peer_transfer_failure, mode,
                    iree_memory_order_release);
}

HIPAPI uint64_t
iree_hip_vmm_test_graph_owned_host_allocation_count(hipGraph_t graph) {
  if (!graph) return UINT64_MAX;
  uint64_t count = 0;
  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  for (iree_hal_streaming_graph_owned_host_allocation_t* allocation =
           stream_graph->owned_host_allocations;
       allocation; allocation = allocation->next) {
    ++count;
  }
  return count;
}

HIPAPI uint64_t iree_hip_vmm_test_graph_internal_node_count(hipGraph_t graph) {
  return graph ? ((iree_hal_streaming_graph_t*)graph)->node_count : UINT64_MAX;
}

HIPAPI uint64_t iree_hip_vmm_test_graph_internal_root_count(hipGraph_t graph) {
  return graph ? ((iree_hal_streaming_graph_t*)graph)->root_count : UINT64_MAX;
}

HIPAPI uint64_t
iree_hip_vmm_test_graph_internal_dependency_count(hipGraph_t graph) {
  if (!graph) return UINT64_MAX;
  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  uint64_t count = stream_graph->additional_edge_count;
  for (iree_hal_streaming_node_block_t* block = stream_graph->node_blocks;
       block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      if (UINT64_MAX - count < block->nodes[i]->dependency_count) {
        return UINT64_MAX;
      }
      count += block->nodes[i]->dependency_count;
    }
  }
  return count;
}

HIPAPI uint64_t
iree_hip_vmm_test_graph_user_object_reference_count(hipGraph_t graph) {
  if (!graph) return UINT64_MAX;
  uint64_t count = 0;
  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  for (iree_hal_streaming_graph_user_object_ref_t* reference =
           stream_graph->user_object_refs;
       reference; reference = reference->next) {
    if (UINT64_MAX - count < reference->count) return UINT64_MAX;
    count += reference->count;
  }
  return count;
}

HIPAPI uint64_t iree_hip_vmm_test_context_reference_count(hipCtx_t context) {
  if (!context) return UINT64_MAX;
  const int32_t count = iree_atomic_ref_count_load(
      &((iree_hal_streaming_context_t*)context)->ref_count);
  return count < 0 ? UINT64_MAX : (uint64_t)count;
}

HIPAPI uint64_t iree_hip_vmm_test_pointer_buffer_reference_count(void* ptr) {
  if (!ptr) return UINT64_MAX;
  iree_hal_streaming_context_t* context = NULL;
  iree_hal_streaming_buffer_ref_t ref = {0};
  iree_status_t status = iree_hal_streaming_memory_lookup_range_across_contexts(
      (iree_hal_streaming_deviceptr_t)ptr, 1, &context, &ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return UINT64_MAX;
  }
  const int32_t count = iree_atomic_ref_count_load(&ref.buffer->ref_count);
  iree_hal_streaming_context_release(context);
  return count < 0 ? UINT64_MAX : (uint64_t)count;
}

HIPAPI void iree_hip_vmm_test_reset_observability(void) {
  iree_atomic_store(&iree_hip_vmm_test_fail_plan_allocation_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_plan_allocation_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_plan_revalidation, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_native_operation_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_access_preflight_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_address_reserve_after_native, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_physical_create_after_native, 0,
                    iree_memory_order_release);
  hrx_allocator_test_reset_vmm_quarantine_observability();
  iree_atomic_store(&iree_hip_vmm_test_fail_graph_exec_memcpy_prepare, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_graph_exec_rebuild, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_fail_graph_add_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_graph_add_ordinal, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_graph_peer_transfer_failure, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_access_preflight_ordinal, 0,
                    iree_memory_order_release);
  for (size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hip_vmm_test_native_operation_attempts); ++i) {
    iree_atomic_store(&iree_hip_vmm_test_native_operation_attempts[i], 0,
                      iree_memory_order_release);
  }
  iree_atomic_store(&iree_hip_vmm_test_native_operation_count_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_native_operation_cursor_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_first_native_error_value, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_raw_native_call_count_value, 0,
                    iree_memory_order_release);
}

// Arms a writer-held injection that temporarily forces the selected epoch
// source to the final usable value after exact device resolution. Delaying the
// mutation until that boundary preserves the production invariant that the
// VMM and common epochs agree during writer admission while still exercising
// each prepare guard independently. Source 0 is the HIP VMM registry epoch and
// source 1 is the common streaming device epoch.
HIPAPI hipError_t iree_hip_vmm_test_set_epoch_exhaustion(int device_ordinal,
                                                         int source,
                                                         bool enabled) {
  if (device_ordinal < 0 || (source != 0 && source != 1)) {
    return hipErrorInvalidValue;
  }
  if (enabled) {
    if (iree_hip_vmm_test_epoch_is_forced) return hipErrorIllegalState;
    if (source == 0) {
      iree_hip_vmm_registry_t* registry = &iree_hip_vmm_registry;
      iree_slim_mutex_lock(&registry->mutex);
      if ((size_t)device_ordinal >= registry->device_epoch_count) {
        iree_slim_mutex_unlock(&registry->mutex);
        return hipErrorInvalidDevice;
      }
      iree_slim_mutex_unlock(&registry->mutex);
    } else {
      iree_hal_streaming_device_t* device =
          iree_hal_streaming_device_entry(device_ordinal);
      if (!device) return hipErrorInvalidDevice;
    }
    iree_hip_vmm_test_epoch_is_forced = true;
    iree_hip_vmm_test_epoch_is_applied = false;
    iree_hip_vmm_test_forced_epoch_source = source;
    iree_hip_vmm_test_forced_epoch_device = device_ordinal;
    return hipSuccess;
  }

  if (!iree_hip_vmm_test_epoch_is_forced ||
      iree_hip_vmm_test_forced_epoch_source != source ||
      iree_hip_vmm_test_forced_epoch_device != device_ordinal) {
    return hipErrorInvalidValue;
  }
  if (iree_hip_vmm_test_epoch_is_applied && source == 0) {
    iree_hip_vmm_registry_t* registry = &iree_hip_vmm_registry;
    iree_slim_mutex_lock(&registry->mutex);
    if ((size_t)device_ordinal >= registry->device_epoch_count ||
        registry->device_epochs[device_ordinal] != UINT64_MAX - 1) {
      iree_slim_mutex_unlock(&registry->mutex);
      return hipErrorIllegalState;
    }
    registry->device_epochs[device_ordinal] = iree_hip_vmm_test_saved_epoch;
    iree_slim_mutex_unlock(&registry->mutex);
  } else if (iree_hip_vmm_test_epoch_is_applied) {
    iree_hal_streaming_device_t* device =
        iree_hal_streaming_device_entry(device_ordinal);
    if (!device ||
        iree_atomic_load(&device->reset_epoch, iree_memory_order_acquire) !=
            UINT64_MAX - 1) {
      return hipErrorIllegalState;
    }
    iree_atomic_store(&device->reset_epoch, iree_hip_vmm_test_saved_epoch,
                      iree_memory_order_release);
  }
  iree_hip_vmm_test_epoch_is_forced = false;
  iree_hip_vmm_test_epoch_is_applied = false;
  iree_hip_vmm_test_forced_epoch_source = -1;
  iree_hip_vmm_test_forced_epoch_device = -1;
  iree_hip_vmm_test_saved_epoch = 0;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_test_apply_epoch_exhaustion(int device_ordinal) {
  if (!iree_hip_vmm_test_epoch_is_forced ||
      iree_hip_vmm_test_forced_epoch_device != device_ordinal) {
    return hipSuccess;
  }
  if (iree_hip_vmm_test_epoch_is_applied) return hipErrorIllegalState;

  if (iree_hip_vmm_test_forced_epoch_source == 0) {
    iree_hip_vmm_registry_t* registry = &iree_hip_vmm_registry;
    iree_slim_mutex_lock(&registry->mutex);
    if ((size_t)device_ordinal >= registry->device_epoch_count) {
      iree_slim_mutex_unlock(&registry->mutex);
      return hipErrorInvalidDevice;
    }
    iree_hip_vmm_test_saved_epoch = registry->device_epochs[device_ordinal];
    registry->device_epochs[device_ordinal] = UINT64_MAX - 1;
    iree_slim_mutex_unlock(&registry->mutex);
  } else {
    iree_hal_streaming_device_t* device =
        iree_hal_streaming_device_entry(device_ordinal);
    if (!device) return hipErrorInvalidDevice;
    iree_hip_vmm_test_saved_epoch = iree_atomic_exchange(
        &device->reset_epoch, UINT64_MAX - 1, iree_memory_order_acq_rel);
  }
  iree_hip_vmm_test_epoch_is_applied = true;
  return hipSuccess;
}

static void iree_hip_vmm_test_notify_phase(int phase, void* object) {
  iree_call_once(&iree_hip_vmm_test_phase_observer_registry_once,
                 iree_hip_vmm_test_phase_observer_registry_initialize);
  iree_hip_vmm_test_phase_observer_registry_t* registry =
      &iree_hip_vmm_test_phase_observer_registry;
  iree_hip_vmm_test_phase_observer_snapshot_t snapshot = {0};
  iree_slim_mutex_lock(&registry->mutex);
  if (registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_ACTIVE) {
    snapshot.observer = registry->observer;
    snapshot.user_data = registry->user_data;
    ++registry->in_flight;
  }
  iree_slim_mutex_unlock(&registry->mutex);
  if (!snapshot.observer) return;

  IREE_ASSERT(iree_hip_vmm_test_phase_observer_callback_depth != UINT32_MAX);
  ++iree_hip_vmm_test_phase_observer_callback_depth;
  snapshot.observer(phase, object, snapshot.user_data);
  IREE_ASSERT(iree_hip_vmm_test_phase_observer_callback_depth > 0);
  --iree_hip_vmm_test_phase_observer_callback_depth;

  iree_slim_mutex_lock(&registry->mutex);
  IREE_ASSERT(registry->in_flight > 0);
  --registry->in_flight;
  const bool notify =
      registry->state == IREE_HIP_VMM_TEST_PHASE_OBSERVER_CLOSING &&
      registry->in_flight == 0;
  iree_slim_mutex_unlock(&registry->mutex);
  if (notify) {
    iree_notification_post(&registry->notification, IREE_ALL_WAITERS);
  }
}

typedef struct iree_hip_vmm_test_incident_context_wait_t {
  // Exact incident context being quiesced.
  iree_hal_streaming_context_t* context;
  // First stream whose non-empty accepted frontier is observed.
  iree_hal_streaming_stream_t* stream;
  // Accepted frontier value captured before its teardown wait.
  uint64_t accepted_value;
} iree_hip_vmm_test_incident_context_wait_t;

static void iree_hip_vmm_test_notify_incident_context_stream_waiting(
    void* user_data, iree_hal_streaming_stream_t* stream) {
  iree_hip_vmm_test_incident_context_wait_t* wait =
      (iree_hip_vmm_test_incident_context_wait_t*)user_data;
  if (!wait->stream) {
    iree_hal_streaming_stream_retain(stream);
    iree_slim_mutex_lock(&stream->mutex);
    wait->accepted_value = stream->pending_value;
    iree_slim_mutex_unlock(&stream->mutex);
    wait->stream = stream;
  }
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_INCIDENT_CONTEXT_STREAM_WAITING, wait->context);
}

static iree_status_t iree_hip_vmm_test_verify_incident_context_stream_wait(
    iree_hip_vmm_test_incident_context_wait_t* wait) {
  if (!wait->stream) return iree_ok_status();

  uint64_t current_value = 0;
  iree_status_t query_status = iree_hal_semaphore_query(
      wait->stream->timeline_semaphore, &current_value);
  iree_hal_streaming_stream_release(wait->stream);
  wait->stream = NULL;
  if (!iree_status_is_ok(query_status)) {
    // A terminal timeline is quiescent even though it has no numeric value.
    iree_status_ignore(query_status);
    return iree_ok_status();
  }
  if (current_value < wait->accepted_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "teardown returned before the accepted stream frontier completed");
  }
  return iree_ok_status();
}

HIPAPI void iree_hip_vmm_test_notify_phase_for_test(int phase) {
  iree_hip_vmm_test_notify_phase(phase, NULL);
}

HIPAPI uint32_t iree_hip_vmm_test_reader_depth(void) {
  return iree_hip_vmm_reader_depth;
}

HIPAPI uint64_t iree_hip_vmm_test_reader_count(void) {
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  const uint64_t reader_count = iree_hip_vmm_registry.reader_count;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  return reader_count;
}

HIPAPI void iree_hip_vmm_test_run_native_callback_window(
    iree_hip_vmm_test_callback_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(callback);
  IREE_ASSERT(iree_hip_vmm_native_callback_window_depth != UINT32_MAX);
  ++iree_hip_vmm_native_callback_window_depth;
  callback(user_data);
  IREE_ASSERT(iree_hip_vmm_native_callback_window_depth > 0);
  --iree_hip_vmm_native_callback_window_depth;
}
#else
#define iree_hip_vmm_test_notify_phase(phase, object) \
  do {                                                \
  } while (0)
#endif

#if defined(IREE_HIP_VMM_TESTING)
void iree_hip_vmm_test_notify_execution_context_queue_release(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_EXECUTION_CONTEXT_QUEUE_RELEASE, object);
}

void iree_hip_vmm_test_notify_execution_context_taken(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_EXECUTION_CONTEXT_TAKEN, object);
}

void iree_hip_vmm_test_notify_borrowed_device_resolved(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_BORROWED_DEVICE_RESOLVED, object);
}

void iree_hip_vmm_test_notify_explicit_context_queue_release(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_EXPLICIT_CONTEXT_QUEUE_RELEASE, object);
}

void iree_hip_vmm_test_notify_explicit_context_unregistered(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_EXPLICIT_CONTEXT_UNREGISTERED, object);
}

void iree_hip_vmm_test_notify_allocation_output(void* object) {
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_ALLOCATION_OUTPUT,
                                 object);
}

void iree_hip_vmm_test_notify_graph_mutation_admitted(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_GRAPH_MUTATION_ADMITTED, object);
}

void iree_hip_vmm_test_notify_graph_exec_active_launch_waiting(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_GRAPH_EXEC_ACTIVE_LAUNCH_WAITING, object);
}

void iree_hip_vmm_test_notify_graph_peer_transfer_waiting(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_GRAPH_PEER_TRANSFER_WAITING, object);
}

void iree_hip_vmm_test_notify_teardown_stream_waiting(void* object) {
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_TEARDOWN_STREAM_WAITING, object);
}

void iree_hip_vmm_test_notify_context_resolved(void* object) {
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_CONTEXT_RESOLVED,
                                 object);
}

void iree_hip_vmm_test_notify_pool_api_admitted(void* object) {
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_POOL_API_ADMITTED,
                                 object);
}

bool iree_hip_vmm_test_consume_graph_exec_memcpy_prepare_failure(void) {
  return iree_atomic_exchange(&iree_hip_vmm_test_fail_graph_exec_memcpy_prepare,
                              0, iree_memory_order_acq_rel) != 0;
}

bool iree_hip_vmm_test_consume_graph_exec_rebuild_failure(void) {
  return iree_atomic_exchange(&iree_hip_vmm_test_fail_graph_exec_rebuild, 0,
                              iree_memory_order_acq_rel) != 0;
}

bool iree_hip_vmm_test_consume_graph_add_failure(void) {
  const int32_t ordinal =
      iree_atomic_fetch_add(&iree_hip_vmm_test_graph_add_ordinal, 1,
                            iree_memory_order_acq_rel) +
      1;
  int32_t expected = ordinal;
  return iree_atomic_compare_exchange_strong(
      &iree_hip_vmm_test_fail_graph_add_ordinal, &expected, 0,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

int iree_hip_vmm_test_consume_graph_peer_transfer_failure(void) {
  return iree_atomic_exchange(&iree_hip_vmm_test_graph_peer_transfer_failure, 0,
                              iree_memory_order_acq_rel);
}

hipError_t iree_hip_vmm_green_context_post_retain_checkpoint(void) {
  if (iree_atomic_exchange(&iree_hip_vmm_test_fail_green_post_retain, 0,
                           iree_memory_order_acq_rel) != 0) {
    return hipErrorOutOfMemory;
  }
  return hipSuccess;
}
#endif  // IREE_HIP_VMM_TESTING

// The instrumented DSO keeps deterministic failure and observability helpers.
// Production forms are macros so unoptimized builds contain no helper bodies or
// calls.
#if defined(IREE_HIP_VMM_TESTING)
static bool iree_hip_vmm_test_take_primary_release_prepare_failure(void) {
  return iree_atomic_exchange(&iree_hip_vmm_test_fail_primary_release_prepare,
                              0, iree_memory_order_acq_rel) != 0;
}

static bool iree_hip_vmm_test_take_binding_rollback_prepare_failure(
    size_t binding_ordinal) {
  if (binding_ordinal > INT32_MAX) return false;
  int32_t expected = (int32_t)binding_ordinal;
  return iree_atomic_compare_exchange_strong(
      &iree_hip_vmm_test_fail_binding_rollback_prepare_ordinal, &expected, 0,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static bool iree_hip_vmm_test_take_plan_allocation_failure(void) {
  const int32_t ordinal =
      iree_atomic_fetch_add(&iree_hip_vmm_test_plan_allocation_ordinal, 1,
                            iree_memory_order_acq_rel) +
      1;
  int32_t expected = ordinal;
  return iree_atomic_compare_exchange_strong(
      &iree_hip_vmm_test_fail_plan_allocation_ordinal, &expected, 0,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static bool iree_hip_vmm_test_take_plan_revalidation_failure(void) {
  return iree_atomic_exchange(&iree_hip_vmm_test_fail_plan_revalidation, 0,
                              iree_memory_order_acq_rel) != 0;
}

static bool iree_hip_vmm_test_take_address_reserve_after_native_failure(void) {
  return iree_atomic_exchange(
             &iree_hip_vmm_test_fail_address_reserve_after_native, 0,
             iree_memory_order_acq_rel) != 0;
}

static bool iree_hip_vmm_test_take_physical_create_after_native_failure(void) {
  return iree_atomic_exchange(
             &iree_hip_vmm_test_fail_physical_create_after_native, 0,
             iree_memory_order_acq_rel) != 0;
}

static bool iree_hip_vmm_test_take_access_preflight_failure(void) {
  const int32_t ordinal =
      iree_atomic_fetch_add(&iree_hip_vmm_test_access_preflight_ordinal, 1,
                            iree_memory_order_acq_rel) +
      1;
  int32_t expected = ordinal;
  return iree_atomic_compare_exchange_strong(
      &iree_hip_vmm_test_fail_access_preflight_ordinal, &expected, 0,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hip_vmm_test_record_raw_native_call(void) {
  iree_atomic_fetch_add(&iree_hip_vmm_test_raw_native_call_count_value, 1,
                        iree_memory_order_acq_rel);
}

static void iree_hip_vmm_test_record_native_journal_begin(
    size_t operation_count, size_t cursor) {
  iree_atomic_store(&iree_hip_vmm_test_native_operation_count_value,
                    (int64_t)operation_count, iree_memory_order_release);
  iree_atomic_store(&iree_hip_vmm_test_native_operation_cursor_value,
                    (int64_t)cursor, iree_memory_order_release);
}

static bool iree_hip_vmm_test_take_native_operation_failure(size_t cursor) {
  if (cursor >= IREE_ARRAYSIZE(iree_hip_vmm_test_native_operation_attempts)) {
    return false;
  }
  iree_atomic_fetch_add(&iree_hip_vmm_test_native_operation_attempts[cursor], 1,
                        iree_memory_order_acq_rel);
  int32_t expected = (int32_t)cursor + 1;
  return iree_atomic_compare_exchange_strong(
      &iree_hip_vmm_test_fail_native_operation_ordinal, &expected, 0,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hip_vmm_test_record_native_operation_success(size_t cursor) {
  iree_atomic_store(&iree_hip_vmm_test_native_operation_cursor_value,
                    (int64_t)cursor, iree_memory_order_release);
}

static void iree_hip_vmm_test_record_first_native_error(
    hrx_vmm_native_status_t native_status) {
  int64_t expected = 0;
  iree_atomic_compare_exchange_strong(
      &iree_hip_vmm_test_first_native_error_value, &expected,
      (int64_t)native_status, iree_memory_order_acq_rel,
      iree_memory_order_acquire);
}
#else
#define iree_hip_vmm_test_take_primary_release_prepare_failure() (false)
#define iree_hip_vmm_test_take_binding_rollback_prepare_failure(ordinal) (false)
#define iree_hip_vmm_test_take_plan_allocation_failure() (false)
#define iree_hip_vmm_test_take_plan_revalidation_failure() (false)
#define iree_hip_vmm_test_take_address_reserve_after_native_failure() (false)
#define iree_hip_vmm_test_take_physical_create_after_native_failure() (false)
#define iree_hip_vmm_test_take_access_preflight_failure() (false)
#define iree_hip_vmm_test_record_raw_native_call() ((void)0)
#define iree_hip_vmm_test_record_native_journal_begin(count, cursor) ((void)0)
#define iree_hip_vmm_test_take_native_operation_failure(cursor) (false)
#define iree_hip_vmm_test_record_native_operation_success(cursor) ((void)0)
#define iree_hip_vmm_test_record_first_native_error(status) ((void)0)
#endif  // IREE_HIP_VMM_TESTING

#if defined(IREE_HIP_VMM_TESTING)
static void iree_hip_vmm_test_native_address_free_observer(bool entering,
                                                           void* address,
                                                           size_t size,
                                                           void* user_data) {
  (void)size;
  (void)user_data;
  if (entering) {
    iree_hip_vmm_test_notify_phase(
        IREE_HIP_VMM_TEST_PHASE_ADDRESS_FREE_CALLBACK_WINDOW, address);
  }
}
#endif  // IREE_HIP_VMM_TESTING

static void iree_hip_vmm_native_callback_window_begin(void* object) {
  IREE_ASSERT(iree_hip_vmm_native_callback_window_depth != UINT32_MAX);
  ++iree_hip_vmm_native_callback_window_depth;
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_NATIVE_CALLBACK_WINDOW,
                                 object);
}

static void iree_hip_vmm_native_callback_window_end(void) {
  IREE_ASSERT(iree_hip_vmm_native_callback_window_depth > 0);
  --iree_hip_vmm_native_callback_window_depth;
}

static hrx_status_t iree_hip_vmm_virtual_memory_release(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer) {
  iree_hip_vmm_native_callback_window_begin(virtual_buffer);
  hrx_status_t status =
      hrx_allocator_virtual_memory_release(allocator, virtual_buffer);
  iree_hip_vmm_native_callback_window_end();
  return status;
}

static hrx_status_t iree_hip_vmm_virtual_memory_release_or_quarantine(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer) {
  iree_hip_vmm_native_callback_window_begin(virtual_buffer);
  hrx_status_t status = hrx_allocator_virtual_memory_release_or_quarantine(
      allocator, virtual_buffer);
  iree_hip_vmm_native_callback_window_end();
  return status;
}

static hipError_t iree_hip_vmm_quarantine_drain_device(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  // A quarantined address release may synchronously invoke a native callback.
  // The lifecycle writer is held, and the callback-window guard rejects HIP
  // reentry before it can attempt reader admission.
  iree_hip_vmm_native_callback_window_begin(/*object=*/NULL);
  hrx_status_t status = hrx_allocator_vmm_quarantine_drain(
      hrx_device_allocator(device->hrx_device));
  iree_hip_vmm_native_callback_window_end();
  return iree_hip_vmm_from_hrx_status(status);
}

static hipError_t iree_hip_vmm_quarantine_drain_all_devices(void) {
  iree_host_size_t device_count = 0;
  hipError_t result = iree_hip_vmm_from_iree_status(
      iree_hal_streaming_device_count(&device_count));
  if (result != hipSuccess) return result;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(i);
    if (!device) return hipErrorInvalidDevice;
    result = iree_hip_vmm_quarantine_drain_device(device);
    if (result != hipSuccess) return result;
  }
  return hipSuccess;
}

#if defined(IREE_HIP_VMM_TESTING)
HIPAPI hipError_t iree_hip_vmm_test_quarantine_drain_without_lifecycle_writer(
    int device_ordinal) {
  iree_hal_streaming_device_t* device =
      device_ordinal >= 0
          ? iree_hal_streaming_device_entry((iree_host_size_t)device_ordinal)
          : NULL;
  return device ? iree_hip_vmm_quarantine_drain_device(device)
                : hipErrorInvalidDevice;
}
#endif  // IREE_HIP_VMM_TESTING

static iree_status_t iree_hip_vmm_resolve_pointer(
    void* user_data, iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_memory_access_t required_access,
    iree_hal_streaming_buffer_ref_t* out_ref, uint64_t* out_capability_id);
static hipError_t iree_hip_vmm_resume_failed_teardown(void);

static void iree_hip_vmm_registry_initialize(void) {
  memset(&iree_hip_vmm_registry, 0, sizeof(iree_hip_vmm_registry));
  iree_slim_mutex_initialize(&iree_hip_vmm_registry.transition_mutex);
  iree_slim_mutex_initialize(&iree_hip_vmm_registry.lifecycle_state_mutex);
  iree_notification_initialize(&iree_hip_vmm_registry.lifecycle_notification);
  iree_slim_mutex_initialize(&iree_hip_vmm_registry.mutex);
  iree_hip_vmm_registry.next_handle_key = 1;
  iree_atomic_store(&iree_hip_vmm_registry.next_capability_id, 1,
                    iree_memory_order_relaxed);
}

static bool iree_hip_vmm_reader_may_enter(void* user_data) {
  iree_hip_vmm_registry_t* registry = (iree_hip_vmm_registry_t*)user_data;
  iree_slim_mutex_lock(&registry->lifecycle_state_mutex);
  const bool may_enter =
      !registry->active ||
      (!registry->writer_pending && registry->writer_waiter_count == 0);
  iree_slim_mutex_unlock(&registry->lifecycle_state_mutex);
  return may_enter;
}

static bool iree_hip_vmm_readers_are_drained(void* user_data) {
  iree_hip_vmm_registry_t* registry = (iree_hip_vmm_registry_t*)user_data;
  iree_slim_mutex_lock(&registry->lifecycle_state_mutex);
  const bool drained = registry->reader_count == 0;
  iree_slim_mutex_unlock(&registry->lifecycle_state_mutex);
  return drained;
}

static hipError_t iree_hip_vmm_reader_begin(void) {
  if (IREE_UNLIKELY(iree_hip_vmm_native_teardown_callback_window_is_active())) {
    return hipErrorNotInitialized;
  }
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  if (iree_hip_vmm_reader_depth > 0) {
    ++iree_hip_vmm_reader_depth;
    return hipSuccess;
  }
  while (true) {
    iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    if (!iree_hip_vmm_registry.active) {
      iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
      return hipErrorNotInitialized;
    }
    if (!iree_hip_vmm_registry.writer_pending &&
        iree_hip_vmm_registry.writer_waiter_count == 0) {
      ++iree_hip_vmm_registry.reader_count;
      iree_hip_vmm_reader_depth = 1;
      iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
      return hipSuccess;
    }
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_READER_WAITING,
                                   NULL);
    iree_notification_await(&iree_hip_vmm_registry.lifecycle_notification,
                            iree_hip_vmm_reader_may_enter,
                            &iree_hip_vmm_registry, iree_infinite_timeout());
  }
}

static void iree_hip_vmm_reader_end(void) {
  IREE_ASSERT(iree_hip_vmm_reader_depth > 0);
  if (--iree_hip_vmm_reader_depth > 0) return;
  bool notify_writer = false;
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  IREE_ASSERT(iree_hip_vmm_registry.reader_count > 0);
  --iree_hip_vmm_registry.reader_count;
  notify_writer = iree_hip_vmm_registry.reader_count == 0;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  if (notify_writer) {
    iree_notification_post(&iree_hip_vmm_registry.lifecycle_notification,
                           IREE_ALL_WAITERS);
  }
}

hipError_t iree_hip_vmm_launch_begin(void) {
  return iree_hip_vmm_reader_begin();
}

void iree_hip_vmm_launch_end(void) { iree_hip_vmm_reader_end(); }

static hipError_t iree_hip_vmm_writer_begin_with_snapshot(
    bool require_active, bool* out_observed_active,
    uint64_t* out_observed_generation) {
  if (IREE_UNLIKELY(iree_hip_vmm_native_teardown_callback_window_is_active())) {
    return hipErrorNotInitialized;
  }
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  if (IREE_UNLIKELY(iree_hip_vmm_reader_depth != 0)) {
    return hipErrorIllegalState;
  }
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  if (out_observed_active) {
    *out_observed_active = iree_hip_vmm_registry.active;
  }
  if (out_observed_generation) {
    *out_observed_generation = iree_hip_vmm_registry.generation;
  }
  ++iree_hip_vmm_registry.writer_waiter_count;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);

  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_WRITER_QUEUED, NULL);

  iree_slim_mutex_lock(&iree_hip_vmm_registry.transition_mutex);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  IREE_ASSERT(iree_hip_vmm_registry.writer_waiter_count > 0);
  --iree_hip_vmm_registry.writer_waiter_count;
  if (require_active && !iree_hip_vmm_registry.active) {
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.transition_mutex);
    // This writer may have been the final waiter keeping later readers
    // closed. A reader can consume the prior deinit writer's notification,
    // observe this waiter, and sleep again; wake it after removing the waiter
    // even though the runtime itself remains inactive.
    iree_notification_post(&iree_hip_vmm_registry.lifecycle_notification,
                           IREE_ALL_WAITERS);
    return hipErrorNotInitialized;
  }
  IREE_ASSERT(!iree_hip_vmm_registry.writer_pending);
  iree_hip_vmm_registry.writer_started_active = iree_hip_vmm_registry.active;
  iree_hip_vmm_registry.writer_pending = true;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_WRITER_PENDING, NULL);
  iree_notification_await(&iree_hip_vmm_registry.lifecycle_notification,
                          iree_hip_vmm_readers_are_drained,
                          &iree_hip_vmm_registry, iree_infinite_timeout());
  return hipSuccess;
}

static hipError_t iree_hip_vmm_writer_begin(bool require_active) {
  return iree_hip_vmm_writer_begin_with_snapshot(
      require_active, /*out_observed_active=*/NULL,
      /*out_observed_generation=*/NULL);
}

// Announces a writer while the caller's only reader admission is still held,
// then consumes that reader and acquires the transition. This preserves writer
// preference across the handoff: no new reader or creator can enter between
// pinning a public object and exact writer-held revalidation.
static hipError_t iree_hip_vmm_writer_begin_from_reader(
    bool* out_writer_started_active, uint64_t* out_generation) {
  if (IREE_UNLIKELY(iree_hip_vmm_native_teardown_callback_window_is_active())) {
    return hipErrorNotInitialized;
  }
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  if (IREE_UNLIKELY(iree_hip_vmm_reader_depth != 1)) {
    return hipErrorIllegalState;
  }

  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  const bool observed_active = iree_hip_vmm_registry.active;
  const uint64_t observed_generation = iree_hip_vmm_registry.generation;
  ++iree_hip_vmm_registry.writer_waiter_count;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);

  // The queued-writer count is visible before the reader drains.
  iree_hip_vmm_reader_end();
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_WRITER_QUEUED, NULL);

  iree_slim_mutex_lock(&iree_hip_vmm_registry.transition_mutex);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  IREE_ASSERT(iree_hip_vmm_registry.writer_waiter_count > 0);
  --iree_hip_vmm_registry.writer_waiter_count;
  IREE_ASSERT(!iree_hip_vmm_registry.writer_pending);
  iree_hip_vmm_registry.writer_started_active = iree_hip_vmm_registry.active;
  iree_hip_vmm_registry.writer_pending = true;
  const bool writer_started_active =
      iree_hip_vmm_registry.writer_started_active;
  const uint64_t current_generation = iree_hip_vmm_registry.generation;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_WRITER_PENDING, NULL);
  iree_notification_await(&iree_hip_vmm_registry.lifecycle_notification,
                          iree_hip_vmm_readers_are_drained,
                          &iree_hip_vmm_registry, iree_infinite_timeout());

  if (out_writer_started_active) {
    *out_writer_started_active = observed_active && writer_started_active;
  }
  if (out_generation) {
    *out_generation =
        observed_generation == current_generation ? current_generation : 0;
  }
  return hipSuccess;
}

static void iree_hip_vmm_writer_end(void) {
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  iree_hip_vmm_registry.writer_pending = false;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  iree_notification_post(&iree_hip_vmm_registry.lifecycle_notification,
                         IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.transition_mutex);
}

hipError_t iree_hip_vmm_module_unload_begin(uint64_t* out_generation) {
  bool observed_active = false;
  uint64_t observed_generation = 0;
  hipError_t result = iree_hip_vmm_writer_begin_with_snapshot(
      /*require_active=*/false, &observed_active, &observed_generation);
  if (result != hipSuccess) return result;
  if (observed_active || iree_hip_vmm_registry.writer_started_active ||
      iree_hip_vmm_registry.generation != observed_generation) {
    iree_hip_vmm_writer_end();
    return hipErrorInvalidResourceHandle;
  }
  if (out_generation) *out_generation = observed_generation;
  return hipSuccess;
}

hipError_t iree_hip_vmm_module_unload_handoff_begin(
    bool* out_writer_started_active, uint64_t* out_generation) {
  return iree_hip_vmm_writer_begin_from_reader(out_writer_started_active,
                                               out_generation);
}

void iree_hip_vmm_module_unload_end(void) { iree_hip_vmm_writer_end(); }

hipError_t iree_hip_vmm_object_destroy_begin(uint64_t* out_generation) {
  return iree_hip_vmm_module_unload_begin(out_generation);
}

hipError_t iree_hip_vmm_object_destroy_handoff_begin(
    bool* out_writer_started_active, uint64_t* out_generation) {
  return iree_hip_vmm_writer_begin_from_reader(out_writer_started_active,
                                               out_generation);
}

void iree_hip_vmm_object_destroy_end(void) { iree_hip_vmm_writer_end(); }

static iree_status_t iree_hip_vmm_context_operation_begin(
    void* user_data, iree_hal_streaming_context_t* context) {
  (void)user_data;
  const hipError_t result = iree_hip_vmm_reader_begin();
  if (result != hipSuccess) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HIP VMM lifecycle is not active");
  }
  if (!iree_hal_streaming_context_is_current(context)) {
    iree_hip_vmm_reader_end();
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "context belongs to a retired runtime epoch");
  }
  return iree_ok_status();
}

static void iree_hip_vmm_context_operation_end(void* user_data) {
  (void)user_data;
  iree_hip_vmm_reader_end();
}

static uint64_t iree_hip_vmm_device_epoch(int device_ordinal) {
  IREE_ASSERT(device_ordinal >= 0);
  IREE_ASSERT((size_t)device_ordinal <
              iree_hip_vmm_registry.device_epoch_count);
  return iree_hip_vmm_registry.device_epochs[device_ordinal];
}

// Resolves one common device only after lifecycle writer admission has made the
// raw inline registry stable for the complete caller transaction.
static hipError_t iree_hip_vmm_resolve_device_under_writer(
    int device_ordinal, iree_hal_streaming_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  *out_device = NULL;

  if (device_ordinal < 0 ||
      (size_t)device_ordinal >= iree_hip_vmm_registry.device_epoch_count) {
    return hipErrorInvalidDevice;
  }
  const uint64_t device_epoch = iree_hip_vmm_device_epoch(device_ordinal);
  if (device_epoch == UINT64_MAX) return hipErrorNotInitialized;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry ||
      device_registry->runtime_generation != iree_hip_vmm_registry.generation ||
      device_registry->device_count !=
          iree_hip_vmm_registry.device_epoch_count) {
    return hipErrorNotInitialized;
  }

  iree_hal_streaming_device_t* device =
      &device_registry->devices[device_ordinal];
  if (device->runtime_generation != iree_hip_vmm_registry.generation ||
      iree_atomic_load(&device->reset_epoch, iree_memory_order_acquire) !=
          device_epoch) {
    return hipErrorNotInitialized;
  }
  *out_device = device;
  return hipSuccess;
}

hipError_t iree_hip_vmm_initialize(size_t device_count, uint64_t generation) {
  if (IREE_UNLIKELY(iree_hip_vmm_native_teardown_callback_window_is_active())) {
    return hipErrorNotInitialized;
  }
  if (device_count == 0) return hipErrorNoDevice;
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.transition_mutex);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  if (iree_hip_vmm_registry.active) {
    const hipError_t result =
        iree_hip_vmm_registry.device_epoch_count == device_count &&
                iree_hip_vmm_registry.generation == generation
            ? hipSuccess
            : hipErrorNotInitialized;
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.transition_mutex);
    return result;
  }
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);

  size_t allocation_size = 0;
  hipError_t result = iree_hip_vmm_registry.reservation_count == 0 &&
                              iree_hip_vmm_registry.allocation_count == 0 &&
                              !iree_hip_vmm_registry.prepared_cleanup &&
                              !iree_hip_vmm_registry.failed_cleanup
                          ? hipSuccess
                          : hipErrorNotInitialized;
  if (result == hipSuccess) {
    if (!iree_host_size_checked_mul(device_count, sizeof(uint64_t),
                                    &allocation_size)) {
      result = hipErrorOutOfMemory;
    } else {
      iree_status_t status =
          iree_allocator_realloc(iree_allocator_system(), allocation_size,
                                 (void**)&iree_hip_vmm_registry.device_epochs);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        result = hipErrorOutOfMemory;
      }
    }
  }
  if (result == hipSuccess) {
    if (generation == 0) {
      result = hipErrorNotInitialized;
    } else {
      memset(iree_hip_vmm_registry.device_epochs, 0, allocation_size);
      iree_hip_vmm_registry.device_epoch_count = device_count;
      iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
      iree_hip_vmm_registry.generation = generation;
      iree_hip_vmm_registry.active = true;
      iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    }
  }
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.transition_mutex);
  if (result == hipSuccess) {
    iree_hal_streaming_set_lifecycle_hooks(iree_hip_vmm_context_operation_begin,
                                           iree_hip_vmm_context_operation_end,
                                           iree_hip_vmm_resolve_pointer,
                                           &iree_hip_vmm_registry);
  }
  return result;
}

bool iree_hip_vmm_native_teardown_callback_window_is_active(void) {
  return iree_hip_vmm_native_callback_window_depth != 0 ||
         iree_hal_amdgpu_libhsa_vmem_address_free_callback_window_is_active();
}

static iree_hip_vmm_registry_t* iree_hip_vmm_registry_lock(void) {
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
  return &iree_hip_vmm_registry;
}

static void iree_hip_vmm_registry_unlock(void) {
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);
}

static hipError_t iree_hip_vmm_from_hrx_status(hrx_status_t status) {
  if (hrx_status_is_ok(status)) return hipSuccess;
  const hrx_status_code_t code = hrx_status_code(status);
  hrx_status_ignore(status);
  switch (code) {
    case HRX_STATUS_INVALID_ARGUMENT:
    case HRX_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case HRX_STATUS_OUT_OF_MEMORY:
      return hipErrorOutOfMemory;
    case HRX_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case HRX_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case HRX_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case HRX_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    default:
      return hipErrorUnknown;
  }
}

static hipError_t iree_hip_vmm_from_iree_status(iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;
  const iree_status_code_t code = iree_status_code(status);
  iree_status_ignore(status);
  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

static hipError_t iree_hip_vmm_grow_pointer_array(void** values,
                                                  size_t* capacity,
                                                  size_t minimum_capacity) {
  if (*capacity >= minimum_capacity) return hipSuccess;
  size_t new_capacity = *capacity ? *capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) return hipErrorOutOfMemory;
    new_capacity *= 2;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(new_capacity, sizeof(void*),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status =
      iree_allocator_realloc(iree_allocator_system(), allocation_size, values);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  *capacity = new_capacity;
  return hipSuccess;
}

static void iree_hip_vmm_reservation_retain(
    iree_hip_vmm_reservation_t* reservation) {
  if (!reservation) return;
  iree_atomic_ref_count_inc(&reservation->ref_count);
}

static void iree_hip_vmm_reservation_release(
    iree_hip_vmm_reservation_t* reservation) {
  if (!reservation) return;
  if (iree_atomic_ref_count_dec(&reservation->ref_count) != 1) return;
  IREE_ASSERT(reservation->virtual_buffer == NULL);
  for (size_t i = 0; i < reservation->access_binding_count; ++i) {
    iree_hal_streaming_memory_release_wrapped_buffer(
        reservation->access_bindings[i].streaming_buffer);
  }
  iree_allocator_free(iree_allocator_system(), reservation->access_bindings);
  iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
  iree_allocator_free(iree_allocator_system(), reservation->native_segments);
  iree_allocator_free(iree_allocator_system(), reservation->mappings);
  iree_slim_mutex_deinitialize(&reservation->mutex);
  iree_hal_streaming_context_release(reservation->owner_context);
  hrx_device_release(reservation->device);
  iree_allocator_free(iree_allocator_system(), reservation);
}

static void iree_hip_vmm_allocation_retain(
    iree_hip_vmm_allocation_t* allocation) {
  if (!allocation) return;
  iree_atomic_ref_count_inc(&allocation->ref_count);
}

static void iree_hip_vmm_allocation_release(
    iree_hip_vmm_allocation_t* allocation) {
  if (!allocation) return;
  if (iree_atomic_ref_count_dec(&allocation->ref_count) != 1) return;
  iree_slim_mutex_deinitialize(&allocation->mutex);
  iree_hal_streaming_context_release(allocation->owner_context);
  hrx_device_release(allocation->device);
  iree_allocator_free(iree_allocator_system(), allocation);
}

static size_t iree_hip_vmm_reservation_lower_bound(
    const iree_hip_vmm_registry_t* registry, uintptr_t base_address) {
  size_t begin = 0;
  size_t end = registry->reservation_count;
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (registry->reservations[middle]->base_address < base_address) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static hipError_t iree_hip_vmm_registry_insert_reservation_locked(
    iree_hip_vmm_registry_t* registry,
    iree_hip_vmm_reservation_t* reservation) {
  if (registry->reservation_count == SIZE_MAX ||
      registry->pending_reservation_slots >
          SIZE_MAX - registry->reservation_count - 1) {
    return hipErrorOutOfMemory;
  }
  hipError_t result = iree_hip_vmm_grow_pointer_array(
      (void**)&registry->reservations, &registry->reservation_capacity,
      registry->reservation_count + registry->pending_reservation_slots + 1);
  if (result != hipSuccess) return result;
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, reservation->base_address);
  if (position < registry->reservation_count &&
      registry->reservations[position]->base_address ==
          reservation->base_address) {
    return hipErrorInvalidValue;
  }
  memmove(&registry->reservations[position + 1],
          &registry->reservations[position],
          (registry->reservation_count - position) *
              sizeof(registry->reservations[0]));
  registry->reservations[position] = reservation;
  ++registry->reservation_count;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_registry_reserve_reservation_slot_locked(
    iree_hip_vmm_registry_t* registry) {
  if (registry->pending_reservation_slots == SIZE_MAX ||
      registry->reservation_count >
          SIZE_MAX - registry->pending_reservation_slots - 1) {
    return hipErrorOutOfMemory;
  }
  hipError_t result = iree_hip_vmm_grow_pointer_array(
      (void**)&registry->reservations, &registry->reservation_capacity,
      registry->reservation_count + registry->pending_reservation_slots + 1);
  if (result == hipSuccess) ++registry->pending_reservation_slots;
  return result;
}

static void iree_hip_vmm_registry_cancel_reservation_slot_locked(
    iree_hip_vmm_registry_t* registry) {
  IREE_ASSERT(registry->pending_reservation_slots > 0);
  --registry->pending_reservation_slots;
}

static hipError_t iree_hip_vmm_registry_insert_reserved_reservation_slot_locked(
    iree_hip_vmm_registry_t* registry,
    iree_hip_vmm_reservation_t* reservation) {
  IREE_ASSERT(registry->pending_reservation_slots > 0);
  IREE_ASSERT(registry->reservation_count +
                  registry->pending_reservation_slots <=
              registry->reservation_capacity);
  --registry->pending_reservation_slots;
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, reservation->base_address);
  if (position < registry->reservation_count &&
      registry->reservations[position]->base_address ==
          reservation->base_address) {
    return hipErrorInvalidValue;
  }
  memmove(&registry->reservations[position + 1],
          &registry->reservations[position],
          (registry->reservation_count - position) *
              sizeof(registry->reservations[0]));
  registry->reservations[position] = reservation;
  ++registry->reservation_count;
  return hipSuccess;
}

static bool iree_hip_vmm_registry_remove_reservation_locked(
    iree_hip_vmm_registry_t* registry,
    iree_hip_vmm_reservation_t* reservation) {
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, reservation->base_address);
  if (position >= registry->reservation_count ||
      registry->reservations[position] != reservation) {
    return false;
  }
  memmove(&registry->reservations[position],
          &registry->reservations[position + 1],
          (registry->reservation_count - position - 1) *
              sizeof(registry->reservations[0]));
  --registry->reservation_count;
  return true;
}

static iree_hip_vmm_reservation_t* iree_hip_vmm_lookup_reservation(
    uintptr_t address, bool require_base) {
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, address);
  iree_hip_vmm_reservation_t* reservation = NULL;
  if (position < registry->reservation_count &&
      registry->reservations[position]->base_address == address) {
    reservation = registry->reservations[position];
  } else if (!require_base && position > 0) {
    iree_hip_vmm_reservation_t* candidate =
        registry->reservations[position - 1];
    if (address >= candidate->base_address &&
        address - candidate->base_address < candidate->size) {
      reservation = candidate;
    }
  }
  if (reservation &&
      (reservation->generation != iree_hip_vmm_registry.generation ||
       reservation->device_ordinal < 0 ||
       (size_t)reservation->device_ordinal >=
           iree_hip_vmm_registry.device_epoch_count ||
       reservation->owner_device_epoch !=
           iree_hip_vmm_device_epoch(reservation->device_ordinal))) {
    reservation = NULL;
  }
  if (reservation) iree_hip_vmm_reservation_retain(reservation);
  iree_hip_vmm_registry_unlock();
  return reservation;
}

#if defined(IREE_HIP_VMM_TESTING)
HIPAPI hipError_t iree_hip_vmm_test_probe_accepted_access(void* device_ptr,
                                                          size_t size,
                                                          int device_ordinal) {
  if (!device_ptr || size == 0) return hipErrorInvalidValue;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)device_ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;

  hipError_t result = hipErrorInvalidValue;
  iree_hal_streaming_context_t* context = NULL;
  uint64_t expected_capability_id = 0;
  if (!iree_slim_mutex_try_lock(&reservation->mutex)) {
    iree_hip_vmm_reservation_release(reservation);
    return hipErrorIllegalState;
  }
  const size_t offset = (uintptr_t)device_ptr - reservation->base_address;
  for (size_t i = 0; i < reservation->access_binding_count; ++i) {
    const iree_hip_vmm_access_binding_t* binding =
        &reservation->access_bindings[i];
    iree_hal_streaming_buffer_t* buffer = binding->streaming_buffer;
    const size_t binding_offset =
        offset >= binding->offset ? offset - binding->offset : SIZE_MAX;
    if (!buffer || binding->location_type != hipMemLocationTypeDevice ||
        binding->location_id != device_ordinal ||
        binding_offset > binding->size ||
        size > binding->size - binding_offset ||
        binding->flags != hipMemAccessFlagsProtReadWrite ||
        !buffer->is_virtual_memory_access || !buffer->is_published ||
        buffer->virtual_memory_capability_id != binding->capability_id) {
      continue;
    }
    context = buffer->context;
    expected_capability_id = binding->capability_id;
    iree_hal_streaming_context_retain(context);
    break;
  }
  iree_slim_mutex_unlock(&reservation->mutex);

  if (context) {
    iree_hal_streaming_buffer_ref_t ref;
    uint64_t capability_id = 0;
    iree_status_t status = iree_hal_streaming_memory_lookup_range_with_access(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, size,
        IREE_HAL_MEMORY_ACCESS_WRITE, &ref, &capability_id);
    if (iree_status_is_ok(status) && capability_id == expected_capability_id) {
      result = hipSuccess;
    }
    iree_status_ignore(status);
    iree_hal_streaming_context_release(context);
  }
  iree_hip_vmm_reservation_release(reservation);
  return result;
}
#endif

static size_t iree_hip_vmm_allocation_lower_bound(
    const iree_hip_vmm_registry_t* registry, uintptr_t handle_key) {
  size_t begin = 0;
  size_t end = registry->allocation_count;
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (registry->allocations[middle]->handle_key < handle_key) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static hipError_t iree_hip_vmm_registry_reserve_allocation_slot_locked(
    iree_hip_vmm_registry_t* registry) {
  if (registry->pending_allocation_slots == SIZE_MAX ||
      registry->allocation_count >
          SIZE_MAX - registry->pending_allocation_slots - 1) {
    return hipErrorOutOfMemory;
  }
  hipError_t result = iree_hip_vmm_grow_pointer_array(
      (void**)&registry->allocations, &registry->allocation_capacity,
      registry->allocation_count + registry->pending_allocation_slots + 1);
  if (result == hipSuccess) ++registry->pending_allocation_slots;
  return result;
}

static void iree_hip_vmm_registry_cancel_allocation_slot_locked(
    iree_hip_vmm_registry_t* registry) {
  IREE_ASSERT(registry->pending_allocation_slots > 0);
  --registry->pending_allocation_slots;
}

static void iree_hip_vmm_registry_insert_reserved_allocation_slot_locked(
    iree_hip_vmm_registry_t* registry, iree_hip_vmm_allocation_t* allocation) {
  IREE_ASSERT(registry->pending_allocation_slots > 0);
  IREE_ASSERT(registry->allocation_count + registry->pending_allocation_slots <=
              registry->allocation_capacity);
  --registry->pending_allocation_slots;
  const size_t position =
      iree_hip_vmm_allocation_lower_bound(registry, allocation->handle_key);
  memmove(&registry->allocations[position + 1],
          &registry->allocations[position],
          (registry->allocation_count - position) *
              sizeof(registry->allocations[0]));
  registry->allocations[position] = allocation;
  ++registry->allocation_count;
}

static bool iree_hip_vmm_registry_remove_allocation_locked(
    iree_hip_vmm_registry_t* registry, iree_hip_vmm_allocation_t* allocation) {
  const size_t position =
      iree_hip_vmm_allocation_lower_bound(registry, allocation->handle_key);
  if (position >= registry->allocation_count ||
      registry->allocations[position] != allocation) {
    return false;
  }
  memmove(&registry->allocations[position],
          &registry->allocations[position + 1],
          (registry->allocation_count - position - 1) *
              sizeof(registry->allocations[0]));
  --registry->allocation_count;
  return true;
}

// Transfers one published entry into an infallible rollback slot without
// changing the registry's accounted occupancy.
static bool iree_hip_vmm_registry_remove_allocation_and_reserve_rollback_locked(
    iree_hip_vmm_registry_t* registry, iree_hip_vmm_allocation_t* allocation) {
  const bool removed =
      iree_hip_vmm_registry_remove_allocation_locked(registry, allocation);
  if (!removed) return false;
  IREE_ASSERT(registry->pending_allocation_slots <
              registry->allocation_capacity);
  ++registry->pending_allocation_slots;
  IREE_ASSERT(registry->allocation_count + registry->pending_allocation_slots <=
              registry->allocation_capacity);
  return true;
}

static iree_hip_vmm_allocation_t* iree_hip_vmm_lookup_allocation(
    hipMemGenericAllocationHandle_t handle) {
  const uintptr_t handle_key = (uintptr_t)handle;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const size_t position =
      iree_hip_vmm_allocation_lower_bound(registry, handle_key);
  iree_hip_vmm_allocation_t* allocation = NULL;
  if (position < registry->allocation_count &&
      registry->allocations[position]->handle_key == handle_key) {
    iree_hip_vmm_allocation_t* candidate = registry->allocations[position];
    if (candidate->generation == iree_hip_vmm_registry.generation &&
        candidate->device_ordinal >= 0 &&
        (size_t)candidate->device_ordinal <
            iree_hip_vmm_registry.device_epoch_count &&
        candidate->owner_device_epoch ==
            iree_hip_vmm_device_epoch(candidate->device_ordinal)) {
      allocation = candidate;
      iree_hip_vmm_allocation_retain(allocation);
    }
  }
  iree_hip_vmm_registry_unlock();
  return allocation;
}

static bool iree_hip_vmm_is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static bool iree_hip_vmm_range_is_valid(size_t total_size, size_t offset,
                                        size_t size, size_t granularity) {
  return size != 0 && granularity != 0 && size % granularity == 0 &&
         offset % granularity == 0 && offset <= total_size &&
         size <= total_size - offset;
}

static hipError_t iree_hip_vmm_validate_properties(
    const hipMemAllocationProp* properties) {
  if (!properties || (properties->type != hipMemAllocationTypePinned &&
                      properties->type != hipMemAllocationTypeUncached)) {
    return hipErrorInvalidValue;
  }
  switch (properties->requestedHandleType) {
    case hipMemHandleTypeNone:
    case hipMemHandleTypePosixFileDescriptor:
    case hipMemHandleTypeWin32:
    case hipMemHandleTypeWin32Kmt:
      break;
    default:
      return hipErrorInvalidValue;
  }
  int device_count = 0;
  hipError_t result = hipGetDeviceCount(&device_count);
  if (result != hipSuccess) return result;
  if (properties->location.type == hipMemLocationTypeHost) {
    return hipSuccess;
  }
  if (properties->location.type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  return properties->location.id >= 0 && properties->location.id < device_count
             ? hipSuccess
             : hipErrorInvalidDevice;
}

static hrx_memory_type_t iree_hip_vmm_memory_type(
    const hipMemAllocationProp* properties) {
  const hrx_memory_type_t cache_type =
      properties->type == hipMemAllocationTypeUncached
          ? HRX_MEMORY_TYPE_DEVICE_UNCACHED
          : HRX_MEMORY_TYPE_NONE;
  if (properties->location.type == hipMemLocationTypeHost) {
    return HRX_MEMORY_TYPE_DEVICE_LOCAL | HRX_MEMORY_TYPE_HOST_VISIBLE |
           HRX_MEMORY_TYPE_HOST_COHERENT | cache_type;
  }
  return HRX_MEMORY_TYPE_DEVICE_LOCAL | cache_type;
}

static hipError_t iree_hip_vmm_get_device_for_properties(
    const hipMemAllocationProp* properties, int* out_device_ordinal,
    hrx_device_t* out_device) {
  int device_ordinal = properties->location.id;
  if (properties->location.type == hipMemLocationTypeHost) {
    hipError_t result = hipGetDevice(&device_ordinal);
    if (result != hipSuccess) return result;
  }
  hrx_device_t device = NULL;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) return result;
  *out_device_ordinal = device_ordinal;
  *out_device = device;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_query_granularity(hrx_device_t device,
                                                 hrx_memory_type_t memory_type,
                                                 size_t* out_minimum,
                                                 size_t* out_recommended) {
  bool supported = false;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_allocator_query_virtual_memory(
          hrx_device_allocator(device), memory_type, &supported, out_minimum,
          out_recommended));
  if (result != hipSuccess) return result;
  return supported && *out_minimum != 0 ? hipSuccess : hipErrorNotSupported;
}

static size_t iree_hip_vmm_mapping_lower_bound(
    const iree_hip_vmm_reservation_t* reservation, size_t offset) {
  size_t begin = 0;
  size_t end = reservation->mapping_count;
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (reservation->mappings[middle].virtual_offset < offset) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static size_t iree_hip_vmm_mapping_containing(
    const iree_hip_vmm_reservation_t* reservation, size_t offset) {
  const size_t position = iree_hip_vmm_mapping_lower_bound(reservation, offset);
  if (position < reservation->mapping_count &&
      reservation->mappings[position].virtual_offset == offset) {
    return position;
  }
  if (position == 0) return SIZE_MAX;
  const iree_hip_vmm_mapping_t* candidate =
      &reservation->mappings[position - 1];
  return offset - candidate->virtual_offset < candidate->size ? position - 1
                                                              : SIZE_MAX;
}

static bool iree_hip_vmm_mapped_range_is_complete(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  size_t position = iree_hip_vmm_mapping_containing(reservation, offset);
  if (position == SIZE_MAX) return false;
  size_t cursor = offset;
  const size_t end = offset + size;
  while (position < reservation->mapping_count && cursor < end) {
    const iree_hip_vmm_mapping_t* mapping = &reservation->mappings[position];
    if (cursor < mapping->virtual_offset ||
        cursor - mapping->virtual_offset >= mapping->size) {
      return false;
    }
    const size_t mapping_end = mapping->virtual_offset + mapping->size;
    cursor = mapping_end < end ? mapping_end : end;
    ++position;
  }
  return cursor == end;
}

static bool iree_hip_vmm_mapping_range_overlaps(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  const size_t position = iree_hip_vmm_mapping_lower_bound(reservation, offset);
  if (position < reservation->mapping_count &&
      reservation->mappings[position].virtual_offset < offset + size) {
    return true;
  }
  if (position == 0) return false;
  const iree_hip_vmm_mapping_t* previous = &reservation->mappings[position - 1];
  return offset < previous->virtual_offset + previous->size;
}

static hipError_t iree_hip_vmm_reserve_mapping_capacity(
    iree_hip_vmm_reservation_t* reservation, size_t minimum_capacity) {
  if (reservation->mapping_capacity >= minimum_capacity) return hipSuccess;
  size_t new_capacity =
      reservation->mapping_capacity ? reservation->mapping_capacity : 8;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) return hipErrorOutOfMemory;
    new_capacity *= 2;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(
          new_capacity, sizeof(reservation->mappings[0]), &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status = iree_allocator_realloc(
      iree_allocator_system(), allocation_size, (void**)&reservation->mappings);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  reservation->mapping_capacity = new_capacity;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_reserve_native_segment_capacity(
    iree_hip_vmm_reservation_t* reservation, size_t minimum_capacity) {
  if (reservation->native_segment_capacity >= minimum_capacity) {
    return hipSuccess;
  }
  size_t new_capacity = reservation->native_segment_capacity
                            ? reservation->native_segment_capacity
                            : 8;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) return hipErrorOutOfMemory;
    new_capacity *= 2;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(new_capacity,
                                  sizeof(reservation->native_segments[0]),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status =
      iree_allocator_realloc(iree_allocator_system(), allocation_size,
                             (void**)&reservation->native_segments);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  reservation->native_segment_capacity = new_capacity;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_retire_allocation(
    iree_hip_vmm_allocation_t* allocation, bool restore_public_on_failure) {
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const bool removed =
      iree_hip_vmm_registry_remove_allocation_and_reserve_rollback_locked(
          registry, allocation);
  iree_hip_vmm_registry_unlock();
  if (!removed) return hipErrorInvalidValue;
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_ALLOCATION_RETIRE_RESERVED, allocation);

  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_allocator_physical_memory_free(
          allocation->allocator, allocation->physical_memory));
  if (result == hipSuccess) {
    registry = iree_hip_vmm_registry_lock();
    iree_hip_vmm_registry_cancel_allocation_slot_locked(registry);
    iree_hip_vmm_registry_unlock();
    allocation->physical_memory = NULL;
    iree_hip_vmm_allocation_release(allocation);  // Registry reference.
    return hipSuccess;
  }

  registry = iree_hip_vmm_registry_lock();
  iree_hip_vmm_registry_insert_reserved_allocation_slot_locked(registry,
                                                               allocation);
  iree_hip_vmm_registry_unlock();
  iree_slim_mutex_lock(&allocation->mutex);
  allocation->retiring = false;
  if (restore_public_on_failure) ++allocation->public_reference_count;
  iree_slim_mutex_unlock(&allocation->mutex);
  return result;
}

static hipError_t iree_hip_vmm_mapping_reference_add(
    iree_hip_vmm_allocation_t* allocation, bool require_public_reference) {
  iree_slim_mutex_lock(&allocation->mutex);
  hipError_t result = hipSuccess;
  if (allocation->retiring ||
      (require_public_reference && allocation->public_reference_count == 0) ||
      allocation->mapping_reference_count == UINT64_MAX) {
    result = hipErrorInvalidValue;
  } else {
    ++allocation->mapping_reference_count;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  return result;
}

static hipError_t iree_hip_vmm_mapping_reference_remove(
    iree_hip_vmm_allocation_t* allocation) {
  bool retire = false;
  iree_slim_mutex_lock(&allocation->mutex);
  if (allocation->mapping_reference_count == 0) {
    iree_slim_mutex_unlock(&allocation->mutex);
    return hipErrorInvalidValue;
  }
  --allocation->mapping_reference_count;
  if (allocation->mapping_reference_count == 0 &&
      allocation->public_reference_count == 0 && !allocation->retiring) {
    allocation->retiring = true;
    retire = true;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  return retire ? iree_hip_vmm_retire_allocation(
                      allocation, /*restore_public_on_failure=*/false)
                : hipSuccess;
}

static int iree_hip_vmm_compare_access_ranges(const void* lhs,
                                              const void* rhs) {
  const iree_hip_vmm_access_range_t* a =
      (const iree_hip_vmm_access_range_t*)lhs;
  const iree_hip_vmm_access_range_t* b =
      (const iree_hip_vmm_access_range_t*)rhs;
  if (a->location_type != b->location_type) {
    return a->location_type < b->location_type ? -1 : 1;
  }
  if (a->location_id != b->location_id) {
    return a->location_id < b->location_id ? -1 : 1;
  }
  if (a->offset == b->offset) return 0;
  return a->offset < b->offset ? -1 : 1;
}

static bool iree_hip_vmm_access_location_matches(
    const iree_hip_vmm_access_range_t* range, const hipMemLocation* location) {
  const int location_id =
      location->type == hipMemLocationTypeHost ? 0 : location->id;
  return range->location_type == location->type &&
         range->location_id == location_id;
}

static uint64_t iree_hip_vmm_allocate_capability_id(void);

static hipError_t iree_hip_vmm_build_access_update(
    const iree_hip_vmm_reservation_t* reservation,
    const hipMemLocation* location, size_t offset, size_t size,
    hipMemAccessFlags flags, bool insert_update,
    iree_hip_vmm_access_range_t** out_ranges, size_t* out_count) {
  *out_ranges = NULL;
  *out_count = 0;
  size_t maximum_count = 0;
  if (!iree_host_size_checked_mul(reservation->access_range_count, 2,
                                  &maximum_count) ||
      !iree_host_size_checked_add(maximum_count, insert_update ? 1 : 0,
                                  &maximum_count)) {
    return hipErrorOutOfMemory;
  }
  if (maximum_count == 0) return hipSuccess;

  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(maximum_count,
                                  sizeof(iree_hip_vmm_access_range_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_access_range_t* ranges = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), allocation_size, (void**)&ranges);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }

  const size_t update_end = offset + size;
  size_t count = 0;
  for (size_t i = 0; i < reservation->access_range_count; ++i) {
    const iree_hip_vmm_access_range_t* old = &reservation->access_ranges[i];
    const size_t old_end = old->offset + old->size;
    if ((location && !iree_hip_vmm_access_location_matches(old, location)) ||
        old_end <= offset || old->offset >= update_end) {
      ranges[count++] = *old;
      continue;
    }
    if (old->offset < offset) {
      ranges[count++] = (iree_hip_vmm_access_range_t){
          .location_type = old->location_type,
          .location_id = old->location_id,
          .offset = old->offset,
          .size = offset - old->offset,
          .flags = old->flags,
          .capability_id = old->capability_id,
      };
    }
    if (old_end > update_end) {
      ranges[count++] = (iree_hip_vmm_access_range_t){
          .location_type = old->location_type,
          .location_id = old->location_id,
          .offset = update_end,
          .size = old_end - update_end,
          .flags = old->flags,
          .capability_id = old->capability_id,
      };
    }
  }
  if (insert_update && flags != hipMemAccessFlagsProtNone) {
    IREE_ASSERT_ARGUMENT(location);
    const uint64_t capability_id = location->type == hipMemLocationTypeDevice
                                       ? iree_hip_vmm_allocate_capability_id()
                                       : 0;
    if (location->type == hipMemLocationTypeDevice && capability_id == 0) {
      iree_allocator_free(iree_allocator_system(), ranges);
      return hipErrorOutOfMemory;
    }
    ranges[count++] = (iree_hip_vmm_access_range_t){
        .location_type = location->type,
        .location_id =
            location->type == hipMemLocationTypeHost ? 0 : location->id,
        .offset = offset,
        .size = size,
        .flags = flags,
        .capability_id = capability_id,
    };
  }
  qsort(ranges, count, sizeof(ranges[0]), iree_hip_vmm_compare_access_ranges);

  size_t merged_count = 0;
  for (size_t i = 0; i < count;) {
    const size_t run_begin = i;
    const iree_hip_vmm_access_range_t* first = &ranges[run_begin];
    size_t run_end = i + 1;
    size_t run_size = first->size;
    while (run_end < count) {
      const iree_hip_vmm_access_range_t* candidate = &ranges[run_end];
      if (first->location_type != candidate->location_type ||
          first->location_id != candidate->location_id ||
          first->flags != candidate->flags ||
          first->offset + run_size != candidate->offset) {
        break;
      }
      run_size += candidate->size;
      ++run_end;
    }

    iree_hip_vmm_access_range_t* merged = &ranges[merged_count++];
    *merged = *first;
    merged->size = run_size;
    if (run_end - run_begin > 1) {
      const uint64_t capability_id =
          first->location_type == hipMemLocationTypeDevice
              ? iree_hip_vmm_allocate_capability_id()
              : 0;
      if (first->location_type == hipMemLocationTypeDevice &&
          capability_id == 0) {
        iree_allocator_free(iree_allocator_system(), ranges);
        return hipErrorOutOfMemory;
      }
      merged->capability_id = capability_id;
    }
    i = run_end;
  }
  *out_ranges = ranges;
  *out_count = merged_count;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_validate_access_location(
    const hipMemLocation* location) {
  if (!location) return hipErrorInvalidValue;
  if (location->type == hipMemLocationTypeHost) {
    return hipSuccess;
  }
  if (location->type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  int device_count = 0;
  hipError_t result = hipGetDeviceCount(&device_count);
  if (result != hipSuccess) return result;
  return location->id >= 0 && location->id < device_count
             ? hipSuccess
             : hipErrorInvalidValue;
}

static hipError_t iree_hip_vmm_validate_access_descriptor(
    const hipMemAccessDesc* descriptor) {
  hipError_t result =
      iree_hip_vmm_validate_access_location(&descriptor->location);
  if (result != hipSuccess) return result;
  switch (descriptor->flags) {
    case hipMemAccessFlagsProtNone:
    case hipMemAccessFlagsProtRead:
    case hipMemAccessFlagsProtReadWrite:
      return hipSuccess;
    default:
      return hipErrorInvalidValue;
  }
}

static hrx_memory_protection_t iree_hip_vmm_protection(
    hipMemAccessFlags flags) {
  switch (flags) {
    case hipMemAccessFlagsProtRead:
      return HRX_MEMORY_PROTECTION_READ;
    case hipMemAccessFlagsProtReadWrite:
      return HRX_MEMORY_PROTECTION_READ_WRITE;
    default:
      return HRX_MEMORY_PROTECTION_NONE;
  }
}

static hrx_memory_access_t iree_hip_vmm_allowed_access(
    hipMemAccessFlags flags) {
  return flags == hipMemAccessFlagsProtRead
             ? HRX_MEMORY_ACCESS_READ
             : HRX_MEMORY_ACCESS_READ | HRX_MEMORY_ACCESS_WRITE;
}

static uint64_t iree_hip_vmm_allocate_capability_id(void) {
  uint64_t capability_id = 0;
  return iree_hal_streaming_atomic_allocate_id(
             &iree_hip_vmm_registry.next_capability_id, UINT64_MAX - 1,
             &capability_id)
             ? capability_id
             : 0;
}

static void iree_hip_vmm_release_access_bindings(
    iree_hip_vmm_access_binding_t* bindings, size_t count) {
  if (!bindings) return;
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_memory_release_wrapped_buffer(
        bindings[i].streaming_buffer);
  }
  iree_allocator_free(iree_allocator_system(), bindings);
}

static hipError_t iree_hip_vmm_build_access_bindings(
    iree_hip_vmm_reservation_t* reservation,
    const iree_hip_vmm_access_range_t* ranges, size_t range_count,
    iree_hip_vmm_access_binding_t** out_bindings, size_t* out_count) {
  *out_bindings = NULL;
  *out_count = 0;
  size_t device_range_count = 0;
  for (size_t i = 0; i < range_count; ++i) {
    if (ranges[i].location_type == hipMemLocationTypeDevice &&
        ranges[i].flags != hipMemAccessFlagsProtNone) {
      ++device_range_count;
    }
  }
  if (device_range_count == 0) return hipSuccess;

  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(device_range_count,
                                  sizeof(iree_hip_vmm_access_binding_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_access_binding_t* bindings = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), allocation_size, (void**)&bindings);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(bindings, 0, allocation_size);

  hipError_t result = hipSuccess;
  size_t count = 0;
  for (size_t i = 0; i < range_count && result == hipSuccess; ++i) {
    const iree_hip_vmm_access_range_t* range = &ranges[i];
    if (range->location_type != hipMemLocationTypeDevice ||
        range->flags == hipMemAccessFlagsProtNone) {
      continue;
    }
    if (range->location_id < 0 ||
        (size_t)range->location_id >=
            iree_hip_vmm_registry.device_epoch_count) {
      result = hipErrorInvalidDevice;
      break;
    }
    iree_hal_streaming_device_t* target_device =
        iree_hal_streaming_device_entry(range->location_id);
    if (!target_device) {
      result = hipErrorInvalidDevice;
      break;
    }
    iree_hal_streaming_context_t* target_context = NULL;
    result = iree_hip_vmm_from_iree_status(
        iree_hal_streaming_device_get_or_create_primary_context(
            target_device, &target_context));
    if (result != hipSuccess) break;

    hrx_buffer_t alias_buffer = NULL;
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_alias(
        hrx_device_allocator(target_device->hrx_device),
        reservation->virtual_buffer, range->offset, range->size,
        iree_hip_vmm_allowed_access(range->flags), &alias_buffer));
    if (result != hipSuccess) break;

    iree_hal_streaming_buffer_t* streaming_buffer = NULL;
    status = iree_hal_streaming_memory_prepare_virtual_alias(
        target_context, alias_buffer, &streaming_buffer);
    hrx_buffer_release(alias_buffer);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_vmm_from_iree_status(status);
      break;
    }
    if (range->capability_id == 0) {
      iree_hal_streaming_memory_release_wrapped_buffer(streaming_buffer);
      result = hipErrorOutOfMemory;
      break;
    }
    const uint64_t target_device_epoch =
        iree_hip_vmm_device_epoch(range->location_id);
    streaming_buffer->virtual_memory_allowed_access =
        (iree_hal_memory_access_t)iree_hip_vmm_allowed_access(range->flags);
    streaming_buffer->virtual_memory_capability_id = range->capability_id;
    streaming_buffer->virtual_memory_generation =
        iree_hip_vmm_registry.generation;
    streaming_buffer->virtual_memory_device_epoch = target_device_epoch;
    bindings[count++] = (iree_hip_vmm_access_binding_t){
        .location_type = range->location_type,
        .location_id = range->location_id,
        .offset = range->offset,
        .size = range->size,
        .flags = range->flags,
        .generation = iree_hip_vmm_registry.generation,
        .target_device_epoch = target_device_epoch,
        .capability_id = range->capability_id,
        .streaming_buffer = streaming_buffer,
    };
  }

  if (result == hipSuccess) {
    *out_bindings = bindings;
    *out_count = count;
  } else {
    iree_hip_vmm_release_access_bindings(bindings, count);
  }
  return result;
}

static hipError_t iree_hip_vmm_reserve_binding_rollback(
    iree_hip_vmm_access_binding_t* bindings, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_buffer_t* buffer = bindings[i].streaming_buffer;
    // A published wrapper cannot be in the middle of publication. Normalize
    // any unused capacity left by an older failed rollback before acquiring
    // capacity owned by this exact preparation attempt.
    if (buffer->is_published && buffer->has_reserved_insert) {
      iree_hal_streaming_memory_cancel_wrapped_buffer_publication(buffer);
    }
    IREE_ASSERT(!buffer->has_reserved_insert);
    if (iree_hip_vmm_test_take_binding_rollback_prepare_failure(i + 1)) {
      for (size_t j = 0; j < i; ++j) {
        iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
            bindings[j].streaming_buffer);
      }
      return hipErrorOutOfMemory;
    }
    iree_status_t status =
        iree_hal_streaming_memory_reserve_wrapped_buffer_publication(buffer);
    if (!iree_status_is_ok(status)) {
      // Preparation is transactional: capacity acquired by this attempt must
      // not escape when a later binding fails.
      for (size_t j = 0; j < i; ++j) {
        iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
            bindings[j].streaming_buffer);
      }
      return iree_hip_vmm_from_iree_status(status);
    }
  }
  return hipSuccess;
}

static hipError_t iree_hip_vmm_unpublish_bindings(
    iree_hip_vmm_access_binding_t* bindings, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    iree_status_t status = iree_hal_streaming_memory_unpublish_wrapped_buffer(
        bindings[i].streaming_buffer);
    if (!iree_status_is_ok(status)) {
      return iree_hip_vmm_from_iree_status(status);
    }
  }
  return hipSuccess;
}

static hipError_t iree_hip_vmm_publish_bindings(
    iree_hip_vmm_access_binding_t* bindings, size_t count) {
  size_t published_count = 0;
  for (; published_count < count; ++published_count) {
    const iree_hip_vmm_access_binding_t* binding = &bindings[published_count];
    if (binding->location_type != hipMemLocationTypeDevice ||
        binding->location_id < 0 ||
        (size_t)binding->location_id >=
            iree_hip_vmm_registry.device_epoch_count ||
        binding->generation != iree_hip_vmm_registry.generation ||
        binding->target_device_epoch !=
            iree_hip_vmm_device_epoch(binding->location_id)) {
      for (size_t i = 0; i < published_count; ++i) {
        iree_status_ignore(iree_hal_streaming_memory_unpublish_wrapped_buffer(
            bindings[i].streaming_buffer));
      }
      for (size_t i = published_count; i < count; ++i) {
        iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
            bindings[i].streaming_buffer);
      }
      return hipErrorInvalidDevice;
    }
    if (binding->streaming_buffer->is_published) {
      // A partially failed unpublish leaves the untouched suffix published.
      // Its rollback reservation is unused and must not leak into a later
      // transaction or context-destroy preparation.
      iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
          binding->streaming_buffer);
      continue;
    }
    iree_status_t status = iree_hal_streaming_memory_publish_wrapped_buffer(
        binding->streaming_buffer);
    if (!iree_status_is_ok(status)) {
      for (size_t i = 0; i < published_count; ++i) {
        iree_status_ignore(iree_hal_streaming_memory_unpublish_wrapped_buffer(
            bindings[i].streaming_buffer));
      }
      for (size_t i = published_count; i < count; ++i) {
        iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
            bindings[i].streaming_buffer);
      }
      return iree_hip_vmm_from_iree_status(status);
    }
  }
  return hipSuccess;
}

hipError_t iree_hip_vmm_context_destroy_begin(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t** out_retained_context,
    iree_hip_context_teardown_t** out_teardown) {
  if (!context) return hipErrorInvalidValue;
  IREE_ASSERT_ARGUMENT(out_retained_context);
  IREE_ASSERT_ARGUMENT(out_teardown);
  *out_retained_context = NULL;
  *out_teardown = NULL;

  iree_hal_streaming_context_t* retained_context = NULL;
  bool observed_active = false;
  uint64_t observed_generation = 0;
  uint64_t pinned_generation = 0;
  bool inactive_entry = false;

  // In the active runtime, resolve and pin the exact incarnation while a
  // lifecycle reader still prevents any writer from removing its public list
  // edge. The pin remains owned while this caller releases the reader and
  // queues for exclusive writer admission, preventing same-address reuse
  // across an overlapping destroy handoff.
  hipError_t result = iree_hip_vmm_reader_begin();
  if (result == hipSuccess) {
    retained_context = iree_hal_streaming_context_lookup_retain(context);
    if (!retained_context) {
      iree_hip_vmm_reader_end();
      return hipErrorInvalidContext;
    }
    iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    pinned_generation = iree_hip_vmm_registry.generation;
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    iree_hip_vmm_test_notify_phase(
        IREE_HIP_VMM_TEST_PHASE_CONTEXT_DESTROY_PINNED, retained_context);
    bool handoff_started_active = false;
    uint64_t handoff_generation = 0;
    result = iree_hip_vmm_writer_begin_from_reader(&handoff_started_active,
                                                   &handoff_generation);
    if (result != hipSuccess) {
      iree_hal_streaming_context_release(retained_context);
      return result;
    }
    if (!handoff_started_active || handoff_generation != pinned_generation ||
        iree_hip_vmm_registry.generation != pinned_generation) {
      iree_hal_streaming_context_release(retained_context);
      iree_hip_vmm_writer_end();
      return hipErrorInvalidContext;
    }
  } else if (result == hipErrorNotInitialized) {
    // Fail-closed teardown has no creators or ordinary readers. Acquire the
    // writer first, bind the lookup to the exact inactive generation, and only
    // then inspect the raw address.
    result = iree_hip_vmm_writer_begin_with_snapshot(
        /*require_active=*/false, &observed_active, &observed_generation);
    if (result != hipSuccess) return result;
    if (observed_active || iree_hip_vmm_registry.writer_started_active ||
        iree_hip_vmm_registry.generation != observed_generation) {
      iree_hip_vmm_writer_end();
      return hipErrorInvalidContext;
    }
    inactive_entry = true;
    retained_context = iree_hal_streaming_context_lookup_retain(context);
    if (!retained_context) {
      iree_hip_vmm_writer_end();
      return hipErrorInvalidContext;
    }
    iree_hip_vmm_test_notify_phase(
        IREE_HIP_VMM_TEST_PHASE_CONTEXT_DESTROY_PINNED, retained_context);
  } else {
    return result;
  }

  // Only an exact surviving explicit context from this inactive generation may
  // drive the retained retry ledger. Reject unknown handles before lookup and
  // stale primary handles here, before either can advance the native cursor.
  if (inactive_entry) {
    iree_hal_streaming_device_t* inactive_device = NULL;
    const hipError_t inactive_device_result =
        retained_context->runtime_generation == iree_hip_vmm_registry.generation
            ? iree_hip_vmm_resolve_device_under_writer(
                  (int)retained_context->device_ordinal, &inactive_device)
            : hipErrorInvalidContext;
    const uint64_t inactive_device_epoch =
        inactive_device ? iree_atomic_load(&inactive_device->reset_epoch,
                                           iree_memory_order_acquire)
                        : 0;
    const bool exact_inactive_explicit =
        inactive_device_result == hipSuccess &&
        retained_context->device_entry == inactive_device &&
        retained_context->device_epoch <= inactive_device_epoch &&
        !retained_context->is_primary &&
        iree_atomic_load(&retained_context->accepting_work,
                         iree_memory_order_acquire) == 0;
    if (!exact_inactive_explicit) {
      iree_hal_streaming_context_release(retained_context);
      iree_hip_vmm_writer_end();
      return hipErrorInvalidContext;
    }
  }

  // Resume the cursor before any new graph scan; the exact explicit-context
  // pin above keeps this incarnation alive if the completed plan releases
  // other publication edges.
  result = iree_hip_vmm_resume_failed_teardown();
  if (result != hipSuccess) {
    iree_hal_streaming_context_release(retained_context);
    iree_hip_vmm_writer_end();
    return result;
  }

  const bool writer_started_active =
      iree_hip_vmm_registry.writer_started_active;

  // Revalidate exact list membership after writer admission. The owning pin
  // prevents address reuse, so equality proves this is the same incarnation,
  // not merely a newly allocated context at the same raw address.
  iree_hal_streaming_context_t* revalidated_context =
      iree_hal_streaming_context_lookup_retain(context);
  const bool exact_incarnation =
      retained_context && revalidated_context == retained_context;
  iree_hal_streaming_context_release(revalidated_context);
  if (!exact_incarnation) {
    iree_hal_streaming_context_release(retained_context);
    iree_hip_vmm_writer_end();
    return hipErrorInvalidContext;
  }

  iree_hal_streaming_device_t* device = NULL;
  const hipError_t device_result =
      retained_context->runtime_generation == iree_hip_vmm_registry.generation
          ? iree_hip_vmm_resolve_device_under_writer(
                (int)retained_context->device_ordinal, &device)
          : hipErrorInvalidContext;
  const uint64_t current_device_epoch =
      device ? iree_atomic_load(&device->reset_epoch, iree_memory_order_acquire)
             : 0;
  const bool valid_explicit_context =
      device_result == hipSuccess && retained_context->device_entry == device &&
      retained_context->device_epoch <= current_device_epoch &&
      !retained_context->is_primary &&
      (writer_started_active ||
       iree_atomic_load(&retained_context->accepting_work,
                        iree_memory_order_acquire) == 0);
  if (!valid_explicit_context) {
    iree_hal_streaming_context_release(retained_context);
    iree_hip_vmm_writer_end();
    return hipErrorInvalidContext;
  }

  const bool context_is_current =
      iree_hal_streaming_context_is_current(retained_context);
  const bool teardown_certified =
      iree_hal_streaming_context_is_teardown_certified(retained_context);
  if (writer_started_active && !context_is_current && !teardown_certified) {
    iree_hal_streaming_context_release(retained_context);
    iree_hip_vmm_writer_end();
    return hipErrorInvalidContext;
  }

  // Snapshot every binding-private owner, dynamic queue, and deferred graph
  // callback before the transactional VMM eviction can mutate anything. This
  // also performs the only fallible context synchronization and rejects an
  // active capture without changing public state.
  iree_hip_context_teardown_t* teardown = NULL;
  iree_hip_vmm_teardown_plan_t* vmm_plan = NULL;
  bool teardown_committed = false;
  result = iree_hip_context_teardown_prepare(
      &retained_context, 1, /*device_filter=*/-1,
      /*abort_captures=*/false,
      /*invalidate_stream_handles=*/false,
      /*seal_device_owned_queues=*/false,
      /*allow_retired_synchronization=*/!writer_started_active, &teardown);
  if (result == hipSuccess) {
    result = iree_hip_vmm_prepare_explicit_context_destroy(retained_context,
                                                           &vmm_plan);
  }
  if (result == hipSuccess) {
    iree_hip_vmm_test_notify_explicit_context_queue_release(retained_context);
    iree_hip_context_teardown_seal_queues(teardown);
    iree_hip_context_teardown_commit(teardown);
    teardown_committed = true;
    result = iree_hip_vmm_commit_teardown_plan(vmm_plan);
  }
  if (result != hipSuccess && !teardown_committed) {
    iree_hip_vmm_cancel_teardown_plan(vmm_plan);
    iree_hip_context_teardown_cancel(teardown);
    iree_hal_streaming_context_release(retained_context);
    iree_hip_vmm_writer_end();
    return result;
  }
  if (result != hipSuccess) {
    // Queue/context retirement crossed the irreversible boundary, while the
    // exact VMM cursor remains in failed_cleanup. Keep the public context/list
    // edges as the inactive retry handle, but release this call's pin and
    // finish deferred teardown only after reopening writer admission.
    iree_hal_streaming_context_release(retained_context);
    iree_hip_vmm_writer_end();
    iree_hip_context_teardown_finish(teardown);
    return result;
  }
  *out_retained_context = retained_context;
  *out_teardown = teardown;
  return hipSuccess;
}

void iree_hip_vmm_context_destroy_end(
    iree_hal_streaming_context_t* retained_context) {
  iree_hal_streaming_context_release(retained_context);
  iree_hip_vmm_writer_end();
}

static bool iree_hip_vmm_access_flags_allow(
    hipMemAccessFlags flags, iree_hal_memory_access_t required_access) {
  const iree_hal_memory_access_t required =
      required_access &
      (IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
  const iree_hal_memory_access_t allowed =
      flags == hipMemAccessFlagsProtRead ? IREE_HAL_MEMORY_ACCESS_READ
      : flags == hipMemAccessFlagsProtReadWrite
          ? IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE
          : IREE_HAL_MEMORY_ACCESS_NONE;
  return required != IREE_HAL_MEMORY_ACCESS_NONE &&
         iree_all_bits_set(allowed, required);
}

static iree_status_t iree_hip_vmm_resolve_pointer(
    void* user_data, iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_memory_access_t required_access,
    iree_hal_streaming_buffer_ref_t* out_ref, uint64_t* out_capability_id) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  IREE_ASSERT_ARGUMENT(out_capability_id);
  memset(out_ref, 0, sizeof(*out_ref));
  *out_capability_id = 0;
  if (user_data != &iree_hip_vmm_registry ||
      !iree_hal_streaming_context_is_current(context)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "context belongs to a retired runtime epoch");
  }
  if (device_ptr == 0 || size == 0 ||
      (iree_device_size_t)(size_t)size != size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid VMM pointer range");
  }

  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)device_ptr,
                                      /*require_base=*/false);
  if (!reservation) return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  const size_t offset = (uintptr_t)device_ptr - reservation->base_address;
  const size_t range_size = (size_t)size;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&reservation->mutex);

  const iree_hip_vmm_access_range_t* access_range = NULL;
  if (reservation->retiring || reservation->poisoned ||
      !iree_hip_vmm_mapped_range_is_complete(reservation, offset, range_size)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VMM pointer range is not mapped");
  }
  for (size_t i = 0;
       i < reservation->access_range_count && iree_status_is_ok(status); ++i) {
    const iree_hip_vmm_access_range_t* candidate =
        &reservation->access_ranges[i];
    const size_t candidate_offset =
        offset >= candidate->offset ? offset - candidate->offset : SIZE_MAX;
    if (candidate->location_type != hipMemLocationTypeDevice ||
        candidate->location_id != (int)context->device_ordinal ||
        candidate_offset > candidate->size ||
        range_size > candidate->size - candidate_offset ||
        !iree_hip_vmm_access_flags_allow(candidate->flags, required_access)) {
      continue;
    }
    access_range = candidate;
    break;
  }
  if (iree_status_is_ok(status) && !access_range) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VMM pointer range has no device access");
  }

  for (size_t i = 0; i < reservation->access_binding_count && access_range &&
                     iree_status_is_ok(status);
       ++i) {
    iree_hip_vmm_access_binding_t* binding = &reservation->access_bindings[i];
    iree_hal_streaming_buffer_t* buffer = binding->streaming_buffer;
    const size_t binding_offset =
        offset >= binding->offset ? offset - binding->offset : SIZE_MAX;
    if (!buffer || buffer->context != context ||
        binding->location_type != hipMemLocationTypeDevice ||
        binding->location_id != (int)context->device_ordinal ||
        binding->generation != iree_hip_vmm_registry.generation ||
        binding->target_device_epoch != context->device_epoch ||
        binding_offset > binding->size ||
        range_size > binding->size - binding_offset ||
        !iree_hip_vmm_access_flags_allow(binding->flags, required_access)) {
      continue;
    }
    if (!buffer->is_virtual_memory_access || !buffer->is_published ||
        buffer->virtual_memory_capability_id != binding->capability_id ||
        buffer->virtual_memory_generation != binding->generation ||
        buffer->virtual_memory_device_epoch != binding->target_device_epoch) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "VMM access capability is not active");
      break;
    }
    out_ref->buffer = buffer;
    out_ref->offset = offset - binding->offset;
    *out_capability_id = binding->capability_id;
    access_range = NULL;
  }

  if (access_range && iree_status_is_ok(status)) {
    iree_hal_streaming_device_t* target_device =
        iree_hal_streaming_device_entry(context->device_ordinal);
    if (!target_device || target_device != context->device_entry) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "VMM target device is retired");
    }

    hrx_buffer_t alias_buffer = NULL;
    if (iree_status_is_ok(status)) {
      status = hrx_to_iree_status(hrx_allocator_virtual_memory_alias(
          hrx_device_allocator(target_device->hrx_device),
          reservation->virtual_buffer, access_range->offset, access_range->size,
          iree_hip_vmm_allowed_access(access_range->flags), &alias_buffer));
    }
    iree_hal_streaming_buffer_t* streaming_buffer = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_memory_prepare_virtual_alias(
          context, alias_buffer, &streaming_buffer);
    }
    hrx_buffer_release(alias_buffer);

    const uint64_t capability_id = access_range->capability_id;
    if (iree_status_is_ok(status) && capability_id == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "VMM capability identity exhausted");
    }

    size_t bindings_size = 0;
    if (iree_status_is_ok(status) &&
        (!iree_host_size_checked_mul(reservation->access_binding_count + 1,
                                     sizeof(reservation->access_bindings[0]),
                                     &bindings_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "VMM access binding table overflow");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_realloc(iree_allocator_system(), bindings_size,
                                      (void**)&reservation->access_bindings);
    }

    if (iree_status_is_ok(status)) {
      streaming_buffer->virtual_memory_allowed_access =
          (iree_hal_memory_access_t)iree_hip_vmm_allowed_access(
              access_range->flags);
      streaming_buffer->virtual_memory_capability_id = capability_id;
      streaming_buffer->virtual_memory_generation =
          iree_hip_vmm_registry.generation;
      streaming_buffer->virtual_memory_device_epoch = context->device_epoch;
      status =
          iree_hal_streaming_memory_publish_wrapped_buffer(streaming_buffer);
    }
    if (iree_status_is_ok(status)) {
      reservation->access_bindings[reservation->access_binding_count++] =
          (iree_hip_vmm_access_binding_t){
              .location_type = hipMemLocationTypeDevice,
              .location_id = (int)context->device_ordinal,
              .offset = access_range->offset,
              .size = access_range->size,
              .flags = access_range->flags,
              .generation = iree_hip_vmm_registry.generation,
              .target_device_epoch = context->device_epoch,
              .capability_id = capability_id,
              .streaming_buffer = streaming_buffer,
          };
      out_ref->buffer = streaming_buffer;
      out_ref->offset = offset - access_range->offset;
      *out_capability_id = capability_id;
    } else {
      iree_hal_streaming_memory_release_wrapped_buffer(streaming_buffer);
    }
  }

  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return status;
}

static hipError_t iree_hip_vmm_is_supported_impl(int device_ordinal,
                                                 bool* out_supported) {
  if (!out_supported) return hipErrorInvalidValue;
  *out_supported = false;
  hrx_device_t device = NULL;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) return result;
  return iree_hip_vmm_from_hrx_status(hrx_allocator_query_virtual_memory(
      hrx_device_allocator(device), HRX_MEMORY_TYPE_DEVICE_LOCAL, out_supported,
      /*min_page_size=*/NULL,
      /*recommended_page_size=*/NULL));
}

static hipError_t iree_hip_vmm_address_reserve_impl(void** ptr, size_t size,
                                                    size_t alignment,
                                                    void* address,
                                                    unsigned long long flags) {
  if (!ptr || flags != 0) return hipErrorInvalidValue;
  *ptr = NULL;
  if (alignment != 0 && !iree_hip_vmm_is_power_of_two(alignment)) {
    return hipErrorInvalidValue;
  }

  int device_ordinal = 0;
  hipError_t result = hipGetDevice(&device_ordinal);
  if (result != hipSuccess) return result;
  hrx_device_t device = NULL;
  result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) return result;
  size_t granularity = 0;
  size_t recommended = 0;
  result = iree_hip_vmm_query_granularity(device, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                          &granularity, &recommended);
  if (result != hipSuccess ||
      !iree_hip_vmm_range_is_valid(size, 0, size, granularity)) {
    return result == hipSuccess ? hipErrorInvalidValue : result;
  }
  if (alignment != 0 && alignment < granularity) {
    return hipErrorInvalidValue;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) return hipErrorInvalidContext;

  // Preallocate both wrapper ownership and a registry publication slot before
  // any native reservation. Concurrent creators account for their own pending
  // slots and cannot consume this capacity.
  iree_hip_vmm_reservation_t* reservation = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*reservation), (void**)&reservation);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(reservation, 0, sizeof(*reservation));
  iree_atomic_ref_count_init(&reservation->ref_count);
  iree_slim_mutex_initialize(&reservation->mutex);
  reservation->device = device;
  hrx_device_retain(device);
  reservation->allocator = hrx_device_allocator(device);
  reservation->owner_context = context;
  iree_hal_streaming_context_retain(context);
  reservation->generation = iree_hip_vmm_registry.generation;
  reservation->owner_device_epoch = iree_hip_vmm_device_epoch(device_ordinal);
  reservation->mutation_serial = 1;
  reservation->size = size;
  reservation->granularity = granularity;
  reservation->device_ordinal = device_ordinal;

  bool registry_slot_reserved = false;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  result = iree_hip_vmm_registry_reserve_reservation_slot_locked(registry);
  if (result == hipSuccess) registry_slot_reserved = true;
  iree_hip_vmm_registry_unlock();
  if (result != hipSuccess) {
    iree_hip_vmm_reservation_release(reservation);
    return result;
  }

  hrx_allocator_t allocator = hrx_device_allocator(device);
  hrx_buffer_t virtual_buffer = NULL;
  iree_hip_vmm_native_callback_window_begin(/*object=*/NULL);
  hrx_status_t reserve_status = hrx_allocator_virtual_memory_reserve_at(
      allocator, /*affinity=*/0, size, alignment, (uintptr_t)address,
      &virtual_buffer);
  iree_hip_vmm_native_callback_window_end();
  result = iree_hip_vmm_from_hrx_status(reserve_status);
  if (result != hipSuccess) goto fail;
  reservation->virtual_buffer = virtual_buffer;

  if (iree_hip_vmm_test_take_address_reserve_after_native_failure()) {
    result = hipErrorOutOfMemory;
  }

  void* device_ptr = NULL;
  if (result == hipSuccess) {
    result = iree_hip_vmm_from_hrx_status(
        hrx_buffer_get_device_ptr(virtual_buffer, &device_ptr));
  }
  if (result == hipSuccess && alignment != 0 &&
      (uintptr_t)device_ptr % alignment != 0) {
    result = hipErrorNotSupported;
  }

  if (result == hipSuccess) {
    reservation->base_address = (uintptr_t)device_ptr;
    registry = iree_hip_vmm_registry_lock();
    result = iree_hip_vmm_registry_insert_reserved_reservation_slot_locked(
        registry, reservation);
    registry_slot_reserved = false;
    iree_hip_vmm_registry_unlock();
  }
  if (result != hipSuccess) goto fail;
  *ptr = device_ptr;
  return hipSuccess;

fail:
  if (virtual_buffer) {
    // The lower call consumes the exact owner on both success and failure;
    // failure is recorded in its persistent retry entry while |result| keeps
    // the publication error's public precedence.
    hrx_status_ignore(iree_hip_vmm_virtual_memory_release_or_quarantine(
        allocator, virtual_buffer));
    virtual_buffer = NULL;
    reservation->virtual_buffer = NULL;
  }
  if (registry_slot_reserved) {
    registry = iree_hip_vmm_registry_lock();
    iree_hip_vmm_registry_cancel_reservation_slot_locked(registry);
    iree_hip_vmm_registry_unlock();
  }
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_address_free_impl(void* device_ptr,
                                                 size_t size) {
  if (!device_ptr || size == 0) return hipErrorInvalidValue;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)device_ptr,
                                      /*require_base=*/true);
  if (!reservation) return hipErrorInvalidValue;

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || reservation->size != size ||
      reservation->mapping_count != 0 ||
      reservation->native_segment_count != 0) {
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
    return hipErrorInvalidValue;
  }
  reservation->retiring = true;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const bool removed =
      iree_hip_vmm_registry_remove_reservation_locked(registry, reservation);
  iree_hip_vmm_registry_unlock();
  if (!removed) {
    reservation->retiring = false;
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
    return hipErrorInvalidValue;
  }
  hipError_t result =
      iree_hip_vmm_from_hrx_status(iree_hip_vmm_virtual_memory_release(
          reservation->allocator, reservation->virtual_buffer));
  if (result == hipSuccess) {
    reservation->virtual_buffer = NULL;
  } else {
    registry = iree_hip_vmm_registry_lock();
    const hipError_t insert_result =
        iree_hip_vmm_registry_insert_reservation_locked(registry, reservation);
    iree_hip_vmm_registry_unlock();
    reservation->retiring = false;
    if (insert_result != hipSuccess) result = insert_result;
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  if (removed && result == hipSuccess) {
    iree_hip_vmm_reservation_release(reservation);  // Registry reference.
  }
  iree_hip_vmm_reservation_release(reservation);  // Operation reference.
  return result;
}

static hipError_t iree_hip_vmm_create_impl(
    hipMemGenericAllocationHandle_t* handle, size_t size,
    const hipMemAllocationProp* properties, unsigned long long flags) {
  if (!handle || size == 0 || flags != 0) return hipErrorInvalidValue;
  *handle = NULL;
  hipError_t result = iree_hip_vmm_validate_properties(properties);
  if (result != hipSuccess) return result;
  if (properties->requestedHandleType != hipMemHandleTypeNone) {
    return hipErrorNotSupported;
  }

  int device_ordinal = 0;
  hrx_device_t device = NULL;
  result = iree_hip_vmm_get_device_for_properties(properties, &device_ordinal,
                                                  &device);
  if (result != hipSuccess) return result;
  const hrx_memory_type_t memory_type = iree_hip_vmm_memory_type(properties);
  size_t granularity = 0;
  size_t recommended = 0;
  result = iree_hip_vmm_query_granularity(device, memory_type, &granularity,
                                          &recommended);
  if (result != hipSuccess ||
      !iree_hip_vmm_range_is_valid(size, 0, size, granularity)) {
    return result == hipSuccess ? hipErrorInvalidValue : result;
  }

  iree_hal_streaming_context_t* owner_context =
      iree_hal_streaming_context_current();
  if (!owner_context ||
      owner_context->device_ordinal != (iree_host_size_t)device_ordinal ||
      !iree_hal_streaming_context_is_current(owner_context)) {
    iree_hal_streaming_device_t* owner_device =
        iree_hal_streaming_device_entry(device_ordinal);
    if (!owner_device) {
      result = hipErrorInvalidDevice;
    } else {
      result = iree_hip_vmm_from_iree_status(
          iree_hal_streaming_device_get_or_create_primary_context(
              owner_device, &owner_context));
    }
  }
  if (result != hipSuccess) return result;

  iree_hip_vmm_allocation_t* allocation = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*allocation), (void**)&allocation);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(allocation, 0, sizeof(*allocation));
  iree_atomic_ref_count_init(&allocation->ref_count);
  iree_slim_mutex_initialize(&allocation->mutex);
  allocation->device = device;
  hrx_device_retain(device);
  allocation->device_ordinal = device_ordinal;
  allocation->generation = iree_hip_vmm_registry.generation;
  allocation->owner_device_epoch = iree_hip_vmm_device_epoch(device_ordinal);
  allocation->owner_context = owner_context;
  iree_hal_streaming_context_retain(owner_context);
  allocation->ownership = IREE_HIP_VMM_ALLOCATION_OWNERSHIP_DEVICE;
  allocation->allocator = hrx_device_allocator(device);
  allocation->properties = *properties;
  allocation->size = size;
  allocation->granularity = granularity;
  allocation->public_reference_count = 1;

  status = iree_hal_streaming_allocate_pointer_buffer_id(
      &allocation->pointer_attribute_buffer_id);
  if (!iree_status_is_ok(status)) {
    result = iree_hip_vmm_from_iree_status(status);
    iree_hip_vmm_allocation_release(allocation);
    return result;
  }

  bool registry_slot_reserved = false;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  result = iree_hip_vmm_registry_reserve_allocation_slot_locked(registry);
  if (result == hipSuccess) {
    registry_slot_reserved = true;
    if (registry->next_handle_key == UINTPTR_MAX) {
      iree_hip_vmm_registry_cancel_allocation_slot_locked(registry);
      registry_slot_reserved = false;
      result = hipErrorOutOfMemory;
    } else {
      allocation->handle_key = registry->next_handle_key;
      ++registry->next_handle_key;
    }
  }
  iree_hip_vmm_registry_unlock();
  if (result != hipSuccess) {
    iree_hip_vmm_allocation_release(allocation);
    return result;
  }

  hrx_physical_memory_t physical_memory = NULL;
  result = iree_hip_vmm_from_hrx_status(hrx_allocator_physical_memory_allocate(
      allocation->allocator, memory_type, size, &physical_memory));
  if (result != hipSuccess) goto create_fail;
  allocation->physical_memory = physical_memory;

  if (iree_hip_vmm_test_take_physical_create_after_native_failure()) {
    result = hipErrorOutOfMemory;
    goto create_fail;
  }

  registry = iree_hip_vmm_registry_lock();
  iree_hip_vmm_registry_insert_reserved_allocation_slot_locked(registry,
                                                               allocation);
  registry_slot_reserved = false;
  iree_hip_vmm_registry_unlock();
  *handle = (hipMemGenericAllocationHandle_t)allocation->handle_key;
  return hipSuccess;

create_fail:
  if (physical_memory) {
    hrx_status_ignore(hrx_allocator_physical_memory_free_or_quarantine(
        allocation->allocator, physical_memory));
    physical_memory = NULL;
    allocation->physical_memory = NULL;
  }
  if (registry_slot_reserved) {
    registry = iree_hip_vmm_registry_lock();
    iree_hip_vmm_registry_cancel_allocation_slot_locked(registry);
    iree_hip_vmm_registry_unlock();
  }
  iree_hip_vmm_allocation_release(allocation);
  return result;
}

static hipError_t iree_hip_vmm_release_impl(
    hipMemGenericAllocationHandle_t handle) {
  if (!handle) return hipErrorInvalidValue;
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!allocation) return hipErrorInvalidValue;

  bool retire = false;
  iree_slim_mutex_lock(&allocation->mutex);
  if (allocation->retiring || allocation->public_reference_count == 0) {
    iree_slim_mutex_unlock(&allocation->mutex);
    iree_hip_vmm_allocation_release(allocation);
    return hipErrorInvalidValue;
  }
  --allocation->public_reference_count;
  if (allocation->public_reference_count == 0 &&
      allocation->mapping_reference_count == 0) {
    allocation->retiring = true;
    retire = true;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  hipError_t result = retire
                          ? iree_hip_vmm_retire_allocation(
                                allocation, /*restore_public_on_failure=*/true)
                          : hipSuccess;
  iree_hip_vmm_allocation_release(allocation);
  return result;
}

static hipError_t iree_hip_vmm_apply_access(
    iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size,
    const hipMemLocation* location, hipMemAccessFlags flags) {
  if (location->type == hipMemLocationTypeDevice) {
    hrx_device_t access_device = NULL;
    hipError_t result = iree_hip_vmm_from_hrx_status(
        hrx_gpu_device_get(location->id, &access_device));
    if (result != hipSuccess) return result;
    iree_hip_vmm_test_record_raw_native_call();
    return iree_hip_vmm_from_hrx_status(
        hrx_allocator_virtual_memory_protect_peer(
            reservation->allocator, hrx_device_allocator(access_device),
            reservation->virtual_buffer, offset, size,
            iree_hip_vmm_protection(flags)));
  }
  // Host access is inherently system-scoped and therefore remains rooted in
  // the reservation owner. Device access above carries both identities and is
  // never widened through the reservation's placement affinity.
  iree_hip_vmm_test_record_raw_native_call();
  return iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_protect(
      reservation->allocator, reservation->virtual_buffer, offset, size,
      HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST, iree_hip_vmm_protection(flags)));
}

static hipError_t iree_hip_vmm_native_map_segment(
    iree_hip_vmm_reservation_t* reservation,
    const iree_hip_vmm_mapping_t* segment) {
  iree_hip_vmm_test_record_raw_native_call();
  return iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_map(
      segment->allocation->allocator, reservation->virtual_buffer,
      segment->virtual_offset, segment->allocation->physical_memory,
      segment->physical_offset, segment->size));
}

static hipError_t iree_hip_vmm_native_unmap_segment(
    iree_hip_vmm_reservation_t* reservation,
    const iree_hip_vmm_mapping_t* segment) {
  iree_hip_vmm_test_record_raw_native_call();
  return iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_unmap(
      reservation->allocator, reservation->virtual_buffer,
      segment->virtual_offset, segment->size));
}

static bool iree_hip_vmm_mapping_equals(const iree_hip_vmm_mapping_t* lhs,
                                        const iree_hip_vmm_mapping_t* rhs) {
  return lhs->virtual_offset == rhs->virtual_offset &&
         lhs->physical_offset == rhs->physical_offset &&
         lhs->size == rhs->size && lhs->allocation == rhs->allocation;
}

static bool iree_hip_vmm_mapping_array_contains(
    const iree_hip_vmm_mapping_t* mappings, size_t count,
    const iree_hip_vmm_mapping_t* target) {
  for (size_t i = 0; i < count; ++i) {
    if (iree_hip_vmm_mapping_equals(&mappings[i], target)) return true;
  }
  return false;
}

static bool iree_hip_vmm_mapping_array_remove(
    iree_hip_vmm_mapping_t* mappings, size_t* count,
    const iree_hip_vmm_mapping_t* target) {
  for (size_t i = 0; i < *count; ++i) {
    if (!iree_hip_vmm_mapping_equals(&mappings[i], target)) continue;
    memmove(&mappings[i], &mappings[i + 1],
            (*count - i - 1) * sizeof(mappings[0]));
    --*count;
    return true;
  }
  return false;
}

static hipError_t iree_hip_vmm_build_native_segments(
    const iree_hip_vmm_mapping_t* mappings, size_t mapping_count,
    const iree_hip_vmm_access_range_t* access_ranges, size_t access_range_count,
    iree_hip_vmm_mapping_t** out_segments, size_t* out_segment_count) {
  *out_segments = NULL;
  *out_segment_count = 0;
  if (mapping_count == 0) return hipSuccess;

  size_t boundary_count = 0;
  size_t maximum_count = 0;
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(access_range_count, 2, &boundary_count) ||
      !iree_host_size_checked_add(boundary_count, 1, &boundary_count) ||
      !iree_host_size_checked_mul(mapping_count, boundary_count,
                                  &maximum_count) ||
      !iree_host_size_checked_mul(maximum_count, sizeof(iree_hip_vmm_mapping_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_mapping_t* segments = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), allocation_size, (void**)&segments);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }

  size_t segment_count = 0;
  for (size_t i = 0; i < mapping_count; ++i) {
    const iree_hip_vmm_mapping_t* mapping = &mappings[i];
    const size_t mapping_end = mapping->virtual_offset + mapping->size;
    size_t cursor = mapping->virtual_offset;
    while (cursor < mapping_end) {
      size_t next = mapping_end;
      for (size_t j = 0; j < access_range_count; ++j) {
        const iree_hip_vmm_access_range_t* range = &access_ranges[j];
        const size_t range_end = range->offset + range->size;
        if (range_end <= cursor || range->offset >= mapping_end) continue;
        if (range->offset > cursor && range->offset < next) {
          next = range->offset;
        }
        if (range_end > cursor && range_end < next) next = range_end;
      }
      IREE_ASSERT(next > cursor);
      segments[segment_count++] = (iree_hip_vmm_mapping_t){
          .virtual_offset = cursor,
          .physical_offset =
              mapping->physical_offset + cursor - mapping->virtual_offset,
          .size = next - cursor,
          .allocation = mapping->allocation,
      };
      cursor = next;
    }
  }
  *out_segments = segments;
  *out_segment_count = segment_count;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_native_apply_segment_access(
    iree_hip_vmm_reservation_t* reservation,
    const iree_hip_vmm_mapping_t* segment,
    const iree_hip_vmm_access_range_t* access_ranges,
    size_t access_range_count) {
  const size_t segment_end = segment->virtual_offset + segment->size;
  for (size_t i = 0; i < access_range_count; ++i) {
    const iree_hip_vmm_access_range_t* range = &access_ranges[i];
    const size_t range_end = range->offset + range->size;
    if (range_end <= segment->virtual_offset || range->offset >= segment_end) {
      continue;
    }
    if (range->offset > segment->virtual_offset || range_end < segment_end) {
      return hipErrorUnknown;
    }
    const hipMemLocation location = {
        .type = range->location_type,
        .id = range->location_id,
    };
    hipError_t result =
        iree_hip_vmm_apply_access(reservation, segment->virtual_offset,
                                  segment->size, &location, range->flags);
    if (result != hipSuccess) return result;
  }
  return hipSuccess;
}

// Validates the complete future native access graph without issuing a native
// mutation. This closes the multi-segment failure hole where an unsupported
// later reservation/physical-pool/access-target tuple was discovered only
// after earlier segments had already been unmapped or remapped.
static hipError_t iree_hip_vmm_prevalidate_native_segment_access(
    iree_hip_vmm_reservation_t* reservation,
    const iree_hip_vmm_mapping_t* segments, size_t segment_count,
    const iree_hip_vmm_access_range_t* access_ranges,
    size_t access_range_count) {
  for (size_t i = 0; i < segment_count; ++i) {
    const iree_hip_vmm_mapping_t* segment = &segments[i];
    if (segment->virtual_offset > SIZE_MAX - segment->size ||
        !segment->allocation || !segment->allocation->physical_memory) {
      return hipErrorInvalidValue;
    }
    const size_t segment_end = segment->virtual_offset + segment->size;
    for (size_t j = 0; j < access_range_count; ++j) {
      const iree_hip_vmm_access_range_t* range = &access_ranges[j];
      if (range->offset > SIZE_MAX - range->size) {
        return hipErrorInvalidValue;
      }
      const size_t range_end = range->offset + range->size;
      if (range_end <= segment->virtual_offset ||
          range->offset >= segment_end) {
        continue;
      }
      if (range->offset > segment->virtual_offset || range_end < segment_end) {
        return hipErrorInvalidValue;
      }
      if (iree_hip_vmm_test_take_access_preflight_failure()) {
        return hipErrorNotSupported;
      }

      hrx_allocator_t access_allocator = NULL;
      hrx_virtual_memory_access_scope_t access_scope =
          HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST;
      if (range->location_type == hipMemLocationTypeDevice) {
        hrx_device_t access_device = NULL;
        hipError_t result = iree_hip_vmm_from_hrx_status(
            hrx_gpu_device_get(range->location_id, &access_device));
        if (result != hipSuccess) return result;
        access_allocator = hrx_device_allocator(access_device);
        access_scope = HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE;
      } else if (range->location_type != hipMemLocationTypeHost) {
        return hipErrorInvalidValue;
      }

      hrx_physical_memory_t physical_memory =
          segment->allocation->physical_memory;
      hrx_vmm_native_operation_t operation = NULL;
      hipError_t result = iree_hip_vmm_from_hrx_status(
          hrx_allocator_vmm_native_operation_prepare_access(
              reservation->allocator, access_allocator,
              reservation->virtual_buffer, segment->virtual_offset,
              segment->size, /*affinity=*/0, access_scope,
              iree_hip_vmm_protection(range->flags),
              /*physical_memory_count=*/1, &physical_memory, &operation));
      hrx_vmm_native_operation_destroy(operation);
      if (result != hipSuccess) return result;
    }
  }
  return hipSuccess;
}

// Replaces the exact ROCr mapping layout while the lifecycle gate and
// reservation mutex exclude observers. On rollback failure the ledger is
// replaced with only mappings confirmed live by successful native calls.
static hipError_t iree_hip_vmm_replace_native_layout_locked(
    iree_hip_vmm_reservation_t* reservation,
    iree_hip_vmm_mapping_t* new_segments, size_t new_segment_count,
    const iree_hip_vmm_access_range_t* old_access_ranges,
    size_t old_access_range_count,
    const iree_hip_vmm_access_range_t* new_access_ranges,
    size_t new_access_range_count, bool* out_replaced) {
  *out_replaced = false;
  size_t maximum_active_count = 0;
  size_t allocation_size = 0;
  if (!iree_host_size_checked_add(reservation->native_segment_count,
                                  new_segment_count, &maximum_active_count) ||
      !iree_host_size_checked_mul(maximum_active_count,
                                  sizeof(iree_hip_vmm_mapping_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_mapping_t* active_segments = NULL;
  if (allocation_size != 0) {
    iree_status_t status = iree_allocator_malloc(
        iree_allocator_system(), allocation_size, (void**)&active_segments);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorOutOfMemory;
    }
  }
  size_t active_segment_count = reservation->native_segment_count;
  if (active_segment_count != 0) {
    memcpy(active_segments, reservation->native_segments,
           active_segment_count * sizeof(active_segments[0]));
  }

  hipError_t forward_result = hipSuccess;
  for (size_t i = 0;
       i < reservation->native_segment_count && forward_result == hipSuccess;
       ++i) {
    const iree_hip_vmm_mapping_t* segment = &reservation->native_segments[i];
    forward_result = iree_hip_vmm_native_unmap_segment(reservation, segment);
    if (forward_result == hipSuccess) {
      IREE_ASSERT(iree_hip_vmm_mapping_array_remove(
          active_segments, &active_segment_count, segment));
    }
  }

  size_t new_mapped_count = 0;
  while (new_mapped_count < new_segment_count && forward_result == hipSuccess) {
    forward_result = iree_hip_vmm_native_map_segment(
        reservation, &new_segments[new_mapped_count]);
    if (forward_result == hipSuccess) {
      active_segments[active_segment_count++] = new_segments[new_mapped_count];
      ++new_mapped_count;
    }
  }
  for (size_t i = 0; i < new_segment_count && forward_result == hipSuccess;
       ++i) {
    forward_result = iree_hip_vmm_native_apply_segment_access(
        reservation, &new_segments[i], new_access_ranges,
        new_access_range_count);
  }

  if (forward_result == hipSuccess) {
    iree_allocator_free(iree_allocator_system(), active_segments);
    iree_allocator_free(iree_allocator_system(), reservation->native_segments);
    reservation->native_segments = new_segments;
    reservation->native_segment_count = new_segment_count;
    reservation->native_segment_capacity = new_segment_count;
    *out_replaced = true;
    return hipSuccess;
  }

  bool rollback_ok = true;
  while (new_mapped_count > 0) {
    --new_mapped_count;
    const iree_hip_vmm_mapping_t* segment = &new_segments[new_mapped_count];
    const hipError_t result =
        iree_hip_vmm_native_unmap_segment(reservation, segment);
    if (result == hipSuccess) {
      IREE_ASSERT(iree_hip_vmm_mapping_array_remove(
          active_segments, &active_segment_count, segment));
    } else {
      rollback_ok = false;
    }
  }

  for (size_t i = 0; i < reservation->native_segment_count; ++i) {
    const iree_hip_vmm_mapping_t* segment = &reservation->native_segments[i];
    if (iree_hip_vmm_mapping_array_contains(active_segments,
                                            active_segment_count, segment)) {
      continue;
    }
    const hipError_t map_result =
        iree_hip_vmm_native_map_segment(reservation, segment);
    if (map_result != hipSuccess) {
      rollback_ok = false;
      continue;
    }
    active_segments[active_segment_count++] = *segment;
    if (iree_hip_vmm_native_apply_segment_access(
            reservation, segment, old_access_ranges, old_access_range_count) !=
        hipSuccess) {
      rollback_ok = false;
    }
  }

  if (rollback_ok &&
      active_segment_count == reservation->native_segment_count) {
    for (size_t i = 0; i < reservation->native_segment_count; ++i) {
      if (!iree_hip_vmm_mapping_array_contains(
              active_segments, active_segment_count,
              &reservation->native_segments[i])) {
        rollback_ok = false;
        break;
      }
    }
  } else {
    rollback_ok = false;
  }
  if (rollback_ok) {
    iree_allocator_free(iree_allocator_system(), active_segments);
    return forward_result;
  }

  iree_allocator_free(iree_allocator_system(), reservation->native_segments);
  reservation->native_segments = active_segments;
  reservation->native_segment_count = active_segment_count;
  reservation->native_segment_capacity = maximum_active_count;
  reservation->poisoned = true;
  return hipErrorUnknown;
}

static hipError_t iree_hip_vmm_map_impl(void* ptr, size_t size, size_t offset,
                                        hipMemGenericAllocationHandle_t handle,
                                        unsigned long long flags) {
  if (!ptr || !handle || size == 0 || offset != 0 || flags != 0) {
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!reservation || !allocation) {
    iree_hip_vmm_reservation_release(reservation);
    iree_hip_vmm_allocation_release(allocation);
    return hipErrorInvalidValue;
  }
  const size_t virtual_offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  hipError_t result = hipSuccess;
  if (reservation->retiring || reservation->poisoned ||
      !iree_hip_vmm_range_is_valid(reservation->size, virtual_offset, size,
                                   reservation->granularity) ||
      reservation->granularity % allocation->granularity != 0 ||
      !iree_hip_vmm_range_is_valid(allocation->size, offset, size,
                                   allocation->granularity) ||
      iree_hip_vmm_mapping_range_overlaps(reservation, virtual_offset, size)) {
    result = hipErrorInvalidValue;
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_reserve_mapping_capacity(
        reservation, reservation->mapping_count + 1);
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_reserve_native_segment_capacity(
        reservation, reservation->native_segment_count + 1);
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_mapping_reference_add(
        allocation, /*require_public_reference=*/true);
  }
  if (result == hipSuccess) {
    const iree_hip_vmm_mapping_t mapping = {
        .virtual_offset = virtual_offset,
        .physical_offset = offset,
        .size = size,
        .allocation = allocation,
    };
    // Map through the physical owner's allocator. The backend independently
    // validates that the foreign reservation shares the same retained native
    // HSA domain.
    result = iree_hip_vmm_native_map_segment(reservation, &mapping);
    if (result == hipSuccess) {
      const size_t position =
          iree_hip_vmm_mapping_lower_bound(reservation, virtual_offset);
      memmove(&reservation->mappings[position + 1],
              &reservation->mappings[position],
              (reservation->mapping_count - position) *
                  sizeof(reservation->mappings[0]));
      reservation->mappings[position] = mapping;
      ++reservation->mapping_count;

      size_t native_position = 0;
      while (native_position < reservation->native_segment_count &&
             reservation->native_segments[native_position].virtual_offset <
                 virtual_offset) {
        ++native_position;
      }
      memmove(&reservation->native_segments[native_position + 1],
              &reservation->native_segments[native_position],
              (reservation->native_segment_count - native_position) *
                  sizeof(reservation->native_segments[0]));
      reservation->native_segments[native_position] = mapping;
      ++reservation->native_segment_count;
      ++reservation->mutation_serial;
    } else {
      (void)iree_hip_vmm_mapping_reference_remove(allocation);
    }
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_allocation_release(allocation);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_build_mapping_unmap(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size,
    iree_hip_vmm_mapping_t** out_mappings, size_t* out_count) {
  *out_mappings = NULL;
  *out_count = 0;
  size_t maximum_count = 0;
  if (!iree_host_size_checked_mul(reservation->mapping_count, 2,
                                  &maximum_count)) {
    return hipErrorOutOfMemory;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(maximum_count, sizeof(iree_hip_vmm_mapping_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_mapping_t* mappings = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), allocation_size, (void**)&mappings);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }

  const size_t unmap_end = offset + size;
  size_t count = 0;
  for (size_t i = 0; i < reservation->mapping_count; ++i) {
    const iree_hip_vmm_mapping_t* old = &reservation->mappings[i];
    const size_t old_end = old->virtual_offset + old->size;
    if (old_end <= offset || old->virtual_offset >= unmap_end) {
      mappings[count++] = *old;
      continue;
    }
    if (old->virtual_offset < offset) {
      mappings[count++] = (iree_hip_vmm_mapping_t){
          .virtual_offset = old->virtual_offset,
          .physical_offset = old->physical_offset,
          .size = offset - old->virtual_offset,
          .allocation = old->allocation,
      };
    }
    if (old_end > unmap_end) {
      mappings[count++] = (iree_hip_vmm_mapping_t){
          .virtual_offset = unmap_end,
          .physical_offset =
              old->physical_offset + unmap_end - old->virtual_offset,
          .size = old_end - unmap_end,
          .allocation = old->allocation,
      };
    }
  }
  *out_mappings = mappings;
  *out_count = count;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_unmap_impl(void* ptr, size_t size) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  if (!ptr || size == 0) return hipErrorInvalidValue;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_hip_vmm_mapping_t* updated_mappings = NULL;
  size_t updated_mapping_count = 0;
  iree_hip_vmm_access_range_t* updated_access_ranges = NULL;
  size_t updated_access_range_count = 0;
  size_t retained_mapping_count = 0;
  uint64_t mutation_serial = 0;
  hipError_t result = hipSuccess;

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || reservation->poisoned ||
      !iree_hip_vmm_range_is_valid(reservation->size, offset, size,
                                   reservation->granularity) ||
      !iree_hip_vmm_mapped_range_is_complete(reservation, offset, size)) {
    result = hipErrorInvalidValue;
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_mapping_unmap(
        reservation, offset, size, &updated_mappings, &updated_mapping_count);
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_access_update(
        reservation, /*location=*/NULL, offset, size, hipMemAccessFlagsProtNone,
        /*insert_update=*/false, &updated_access_ranges,
        &updated_access_range_count);
  }
  for (size_t i = 0; i < updated_mapping_count && result == hipSuccess; ++i) {
    result =
        iree_hip_vmm_mapping_reference_add(updated_mappings[i].allocation,
                                           /*require_public_reference=*/false);
    if (result == hipSuccess) ++retained_mapping_count;
  }
  mutation_serial = reservation->mutation_serial;
  iree_slim_mutex_unlock(&reservation->mutex);

  iree_hip_vmm_mapping_t* updated_native_segments = NULL;
  size_t updated_native_segment_count = 0;
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_native_segments(
        updated_mappings, updated_mapping_count, updated_access_ranges,
        updated_access_range_count, &updated_native_segments,
        &updated_native_segment_count);
  }

  iree_hip_vmm_access_binding_t* updated_bindings = NULL;
  size_t updated_binding_count = 0;
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_access_bindings(
        reservation, updated_access_ranges, updated_access_range_count,
        &updated_bindings, &updated_binding_count);
  }

  iree_hip_vmm_mapping_t* old_mappings = NULL;
  size_t old_mapping_count = 0;
  iree_hip_vmm_access_range_t* old_access_ranges = NULL;
  iree_hip_vmm_access_binding_t* old_bindings = NULL;
  size_t old_binding_count = 0;
  bool committed = false;
  bool release_old_bindings = false;

  // Writer admission is already closed. Drain work accepted before the
  // transition while its old capabilities are still published; accepted host
  // callbacks may resolve VMM pointers and must be allowed to complete before
  // any alias is removed. No context/queue wait may occur under the
  // reservation mutex.
  if (result == hipSuccess) {
    result = iree_hip_vmm_from_iree_status(
        iree_hal_streaming_context_synchronize_all());
  }

  if (result == hipSuccess) {
    iree_slim_mutex_lock(&reservation->mutex);
    if (reservation->retiring || reservation->poisoned ||
        reservation->mutation_serial != mutation_serial ||
        !iree_hip_vmm_mapped_range_is_complete(reservation, offset, size)) {
      result = hipErrorInvalidValue;
    }
    if (result == hipSuccess) {
      result = iree_hip_vmm_reserve_binding_rollback(
          reservation->access_bindings, reservation->access_binding_count);
    }
    bool old_unpublished = false;
    if (result == hipSuccess) {
      old_unpublished = true;
      result = iree_hip_vmm_unpublish_bindings(
          reservation->access_bindings, reservation->access_binding_count);
    }
    bool native_replaced = false;
    if (result == hipSuccess) {
      result = iree_hip_vmm_replace_native_layout_locked(
          reservation, updated_native_segments, updated_native_segment_count,
          reservation->access_ranges, reservation->access_range_count,
          updated_access_ranges, updated_access_range_count, &native_replaced);
      if (native_replaced) updated_native_segments = NULL;
    }

    if (!native_replaced) {
      if (reservation->poisoned) {
        old_bindings = reservation->access_bindings;
        old_binding_count = reservation->access_binding_count;
        reservation->access_bindings = NULL;
        reservation->access_binding_count = 0;
        release_old_bindings = true;
        result = hipErrorUnknown;
      } else if (old_unpublished &&
                 iree_hip_vmm_publish_bindings(
                     reservation->access_bindings,
                     reservation->access_binding_count) != hipSuccess) {
        reservation->poisoned = true;
        old_bindings = reservation->access_bindings;
        old_binding_count = reservation->access_binding_count;
        reservation->access_bindings = NULL;
        reservation->access_binding_count = 0;
        release_old_bindings = true;
        result = hipErrorUnknown;
      }
    } else {
      hipError_t publish_result = iree_hip_vmm_publish_bindings(
          updated_bindings, updated_binding_count);
      old_mappings = reservation->mappings;
      old_mapping_count = reservation->mapping_count;
      old_access_ranges = reservation->access_ranges;
      old_bindings = reservation->access_bindings;
      old_binding_count = reservation->access_binding_count;

      reservation->mappings = updated_mappings;
      reservation->mapping_count = updated_mapping_count;
      reservation->mapping_capacity = updated_mapping_count;
      updated_mappings = NULL;
      retained_mapping_count = 0;
      reservation->access_ranges = updated_access_ranges;
      reservation->access_range_count = updated_access_range_count;
      updated_access_ranges = NULL;
      reservation->access_bindings =
          publish_result == hipSuccess ? updated_bindings : NULL;
      reservation->access_binding_count =
          publish_result == hipSuccess ? updated_binding_count : 0;
      if (publish_result == hipSuccess) {
        updated_bindings = NULL;
        updated_binding_count = 0;
      } else {
        reservation->poisoned = true;
        result = hipErrorUnknown;
      }
      ++reservation->mutation_serial;
      committed = true;
      release_old_bindings = true;
    }
    iree_slim_mutex_unlock(&reservation->mutex);
  }

  if (!committed) {
    for (size_t i = 0; i < retained_mapping_count; ++i) {
      const hipError_t remove_result =
          iree_hip_vmm_mapping_reference_remove(updated_mappings[i].allocation);
      if (result == hipSuccess) result = remove_result;
    }
  } else {
    for (size_t i = 0; i < old_mapping_count; ++i) {
      const hipError_t remove_result =
          iree_hip_vmm_mapping_reference_remove(old_mappings[i].allocation);
      if (result == hipSuccess) result = remove_result;
    }
  }
  if (release_old_bindings) {
    iree_hip_vmm_release_access_bindings(old_bindings, old_binding_count);
  }
  iree_hip_vmm_release_access_bindings(updated_bindings, updated_binding_count);
  iree_allocator_free(iree_allocator_system(), old_access_ranges);
  iree_allocator_free(iree_allocator_system(), old_mappings);
  iree_allocator_free(iree_allocator_system(), updated_access_ranges);
  iree_allocator_free(iree_allocator_system(), updated_native_segments);
  iree_allocator_free(iree_allocator_system(), updated_mappings);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_set_access_impl(
    void* ptr, size_t size, const hipMemAccessDesc* descriptors, size_t count) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  if (!ptr || size == 0 || !descriptors || count == 0) {
    return hipErrorInvalidValue;
  }
  for (size_t i = 0; i < count; ++i) {
    hipError_t result =
        iree_hip_vmm_validate_access_descriptor(&descriptors[i]);
    if (result != hipSuccess) return result;
  }

  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_hip_vmm_access_range_t* updated_ranges = NULL;
  size_t updated_count = 0;
  bool updated_ranges_owned = false;
  uint64_t mutation_serial = 0;
  hipError_t result = hipSuccess;
  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || reservation->poisoned ||
      !iree_hip_vmm_range_is_valid(reservation->size, offset, size,
                                   reservation->granularity) ||
      !iree_hip_vmm_mapped_range_is_complete(reservation, offset, size)) {
    result = hipErrorInvalidValue;
  }
  for (size_t i = 0; i < count && result == hipSuccess; ++i) {
    iree_hip_vmm_reservation_t view = *reservation;
    view.access_ranges =
        updated_ranges_owned ? updated_ranges : reservation->access_ranges;
    view.access_range_count =
        updated_ranges_owned ? updated_count : reservation->access_range_count;
    iree_hip_vmm_access_range_t* next_ranges = NULL;
    size_t next_count = 0;
    result = iree_hip_vmm_build_access_update(
        &view, &descriptors[i].location, offset, size, descriptors[i].flags,
        /*insert_update=*/true, &next_ranges, &next_count);
    if (updated_ranges_owned) {
      iree_allocator_free(iree_allocator_system(), updated_ranges);
    }
    updated_ranges = next_ranges;
    updated_count = next_count;
    updated_ranges_owned = result == hipSuccess;
  }
  mutation_serial = reservation->mutation_serial;
  iree_slim_mutex_unlock(&reservation->mutex);

  iree_hip_vmm_mapping_t* updated_native_segments = NULL;
  size_t updated_native_segment_count = 0;
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_native_segments(
        reservation->mappings, reservation->mapping_count, updated_ranges,
        updated_count, &updated_native_segments, &updated_native_segment_count);
  }

  iree_hip_vmm_access_binding_t* updated_bindings = NULL;
  size_t updated_binding_count = 0;
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_access_bindings(
        reservation, updated_ranges, updated_count, &updated_bindings,
        &updated_binding_count);
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_prevalidate_native_segment_access(
        reservation, updated_native_segments, updated_native_segment_count,
        updated_ranges, updated_count);
  }

  iree_hip_vmm_access_range_t* old_ranges = NULL;
  iree_hip_vmm_access_binding_t* old_bindings = NULL;
  size_t old_binding_count = 0;
  bool release_old_bindings = false;

  // Keep old access bindings published until every operation admitted before
  // this writer has become terminal. Accepted graph host callbacks resolve at
  // execution time and therefore require the old capability during the drain.
  if (result == hipSuccess) {
    iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_ACCESS_PRE_DRAIN,
                                   reservation);
    result = iree_hip_vmm_from_iree_status(
        iree_hal_streaming_context_synchronize_all());
  }
  if (result == hipSuccess) {
    iree_slim_mutex_lock(&reservation->mutex);
    if (reservation->retiring || reservation->poisoned ||
        reservation->mutation_serial != mutation_serial ||
        !iree_hip_vmm_mapped_range_is_complete(reservation, offset, size)) {
      result = hipErrorInvalidValue;
    }
    if (result == hipSuccess) {
      result = iree_hip_vmm_reserve_binding_rollback(
          reservation->access_bindings, reservation->access_binding_count);
    }
    bool old_unpublished = false;
    if (result == hipSuccess) {
      old_unpublished = true;
      result = iree_hip_vmm_unpublish_bindings(
          reservation->access_bindings, reservation->access_binding_count);
    }
    bool native_replaced = false;
    if (result == hipSuccess) {
      result = iree_hip_vmm_replace_native_layout_locked(
          reservation, updated_native_segments, updated_native_segment_count,
          reservation->access_ranges, reservation->access_range_count,
          updated_ranges, updated_count, &native_replaced);
      if (native_replaced) updated_native_segments = NULL;
    }
    if (!native_replaced) {
      if (reservation->poisoned) {
        old_bindings = reservation->access_bindings;
        old_binding_count = reservation->access_binding_count;
        reservation->access_bindings = NULL;
        reservation->access_binding_count = 0;
        release_old_bindings = true;
        result = hipErrorUnknown;
      } else if (old_unpublished &&
                 iree_hip_vmm_publish_bindings(
                     reservation->access_bindings,
                     reservation->access_binding_count) != hipSuccess) {
        reservation->poisoned = true;
        old_bindings = reservation->access_bindings;
        old_binding_count = reservation->access_binding_count;
        reservation->access_bindings = NULL;
        reservation->access_binding_count = 0;
        release_old_bindings = true;
        result = hipErrorUnknown;
      }
    } else {
      const hipError_t publish_result = iree_hip_vmm_publish_bindings(
          updated_bindings, updated_binding_count);
      old_ranges = reservation->access_ranges;
      old_bindings = reservation->access_bindings;
      old_binding_count = reservation->access_binding_count;
      reservation->access_ranges = updated_ranges;
      reservation->access_range_count = updated_count;
      updated_ranges = NULL;
      updated_ranges_owned = false;
      reservation->access_bindings =
          publish_result == hipSuccess ? updated_bindings : NULL;
      reservation->access_binding_count =
          publish_result == hipSuccess ? updated_binding_count : 0;
      if (publish_result == hipSuccess) {
        updated_bindings = NULL;
        updated_binding_count = 0;
      } else {
        reservation->poisoned = true;
        result = hipErrorUnknown;
      }
      ++reservation->mutation_serial;
      release_old_bindings = true;
    }
    iree_slim_mutex_unlock(&reservation->mutex);
  }

  if (release_old_bindings) {
    iree_hip_vmm_release_access_bindings(old_bindings, old_binding_count);
  }
  iree_hip_vmm_release_access_bindings(updated_bindings, updated_binding_count);
  iree_allocator_free(iree_allocator_system(), old_ranges);
  iree_allocator_free(iree_allocator_system(), updated_native_segments);
  if (updated_ranges_owned) {
    iree_allocator_free(iree_allocator_system(), updated_ranges);
  }
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_get_access_impl(unsigned long long* flags,
                                               const hipMemLocation* location,
                                               void* ptr) {
  if (!flags || !ptr) return hipErrorInvalidValue;
  hipError_t result = iree_hip_vmm_validate_access_location(location);
  if (result != hipSuccess) return result;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || reservation->poisoned ||
      iree_hip_vmm_mapping_containing(reservation, offset) == SIZE_MAX) {
    result = hipErrorInvalidValue;
  } else {
    *flags = hipMemAccessFlagsProtNone;
    for (size_t i = 0; i < reservation->access_range_count; ++i) {
      const iree_hip_vmm_access_range_t* range = &reservation->access_ranges[i];
      if (iree_hip_vmm_access_location_matches(range, location) &&
          offset >= range->offset && offset - range->offset < range->size) {
        *flags = range->flags;
        break;
      }
    }
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_get_allocation_granularity_impl(
    size_t* granularity, const hipMemAllocationProp* properties,
    hipMemAllocationGranularity_flags option) {
  if (!granularity) return hipErrorInvalidValue;
  hipError_t result = iree_hip_vmm_validate_properties(properties);
  if (result == hipErrorInvalidDevice) result = hipErrorInvalidValue;
  if (result != hipSuccess) return result;
  if (option != hipMemAllocationGranularityMinimum &&
      option != hipMemAllocationGranularityRecommended) {
    return hipErrorInvalidValue;
  }
  int device_ordinal = 0;
  hrx_device_t device = NULL;
  result = iree_hip_vmm_get_device_for_properties(properties, &device_ordinal,
                                                  &device);
  if (result != hipSuccess) return result;
  size_t minimum = 0;
  size_t recommended = 0;
  result = iree_hip_vmm_query_granularity(
      device, iree_hip_vmm_memory_type(properties), &minimum, &recommended);
  if (result == hipSuccess) {
    *granularity =
        option == hipMemAllocationGranularityMinimum ? minimum : recommended;
  }
  return result;
}

static hipError_t iree_hip_vmm_get_allocation_properties_impl(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle) {
  if (!properties || !handle) return hipErrorInvalidValue;
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!allocation) return hipErrorInvalidValue;
  iree_slim_mutex_lock(&allocation->mutex);
  hipError_t result = hipSuccess;
  if (allocation->retiring || allocation->public_reference_count == 0) {
    result = hipErrorInvalidValue;
  } else {
    *properties = allocation->properties;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  iree_hip_vmm_allocation_release(allocation);
  return result;
}

static hipError_t iree_hip_vmm_retain_allocation_handle_impl(
    hipMemGenericAllocationHandle_t* handle, void* address) {
  if (!handle || !address) return hipErrorInvalidValue;
  *handle = NULL;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)address,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)address - reservation->base_address;
  iree_slim_mutex_lock(&reservation->mutex);
  const size_t mapping_position =
      iree_hip_vmm_mapping_containing(reservation, offset);
  hipError_t result = hipSuccess;
  if (reservation->retiring || reservation->poisoned ||
      mapping_position == SIZE_MAX) {
    result = hipErrorInvalidValue;
  } else {
    iree_hip_vmm_allocation_t* allocation =
        reservation->mappings[mapping_position].allocation;
    iree_slim_mutex_lock(&allocation->mutex);
    if (allocation->retiring ||
        allocation->public_reference_count == UINT64_MAX) {
      result = hipErrorInvalidValue;
    } else {
      ++allocation->public_reference_count;
      *handle = (hipMemGenericAllocationHandle_t)allocation->handle_key;
    }
    iree_slim_mutex_unlock(&allocation->mutex);
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_plan_allocate_array(size_t count,
                                                   size_t element_size,
                                                   void** out_array) {
  *out_array = NULL;
  if (count == 0) return hipSuccess;
  if (iree_hip_vmm_test_take_plan_allocation_failure()) {
    return hipErrorOutOfMemory;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(count, element_size, &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               allocation_size, out_array);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(*out_array, 0, allocation_size);
  return hipSuccess;
}

static hipError_t iree_hip_vmm_plan_copy_array(const void* source, size_t count,
                                               size_t element_size,
                                               void** out_array) {
  if (count != 0 && !source) {
    *out_array = NULL;
    return hipErrorInvalidValue;
  }
  hipError_t result =
      iree_hip_vmm_plan_allocate_array(count, element_size, out_array);
  if (result == hipSuccess && count != 0) {
    size_t copy_size = 0;
    if (!iree_host_size_checked_mul(count, element_size, &copy_size)) {
      iree_allocator_free(iree_allocator_system(), *out_array);
      *out_array = NULL;
      return hipErrorOutOfMemory;
    }
    memcpy(*out_array, source, copy_size);
  }
  return result;
}

static iree_hip_vmm_allocation_plan_t* iree_hip_vmm_plan_find_allocation(
    iree_hip_vmm_teardown_plan_t* plan, iree_hip_vmm_allocation_t* allocation) {
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    if (plan->allocations[i].allocation == allocation) {
      return &plan->allocations[i];
    }
  }
  return NULL;
}

static const iree_hip_vmm_allocation_plan_t*
iree_hip_vmm_plan_find_allocation_const(
    const iree_hip_vmm_teardown_plan_t* plan,
    const iree_hip_vmm_allocation_t* allocation) {
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    if (plan->allocations[i].allocation == allocation) {
      return &plan->allocations[i];
    }
  }
  return NULL;
}

static hipError_t iree_hip_vmm_plan_add_incident_context(
    iree_hip_vmm_teardown_plan_t* plan, iree_hal_streaming_context_t* context) {
  if (!context) return hipSuccess;
  for (size_t i = 0; i < plan->incident_context_count; ++i) {
    if (plan->incident_contexts[i] == context) return hipSuccess;
  }
  hipError_t result = iree_hip_vmm_grow_pointer_array(
      (void**)&plan->incident_contexts, &plan->incident_context_capacity,
      plan->incident_context_count + 1);
  if (result != hipSuccess) return result;
  iree_hal_streaming_context_retain(context);
  plan->incident_contexts[plan->incident_context_count++] = context;
  return hipSuccess;
}

static bool iree_hip_vmm_plan_retires_reservation(
    const iree_hip_vmm_teardown_plan_t* plan,
    const iree_hip_vmm_reservation_t* reservation) {
  switch (plan->scope) {
    case IREE_HIP_VMM_TEARDOWN_PRIMARY_CONTEXT:
      return reservation->owner_context == plan->target_context;
    case IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT:
      return reservation->device_ordinal == plan->target_device_ordinal;
    case IREE_HIP_VMM_TEARDOWN_PROCESS_ALL:
      return true;
    case IREE_HIP_VMM_TEARDOWN_EXPLICIT_CONTEXT_ALIASES_ONLY:
    default:
      return false;
  }
}

static bool iree_hip_vmm_plan_retires_allocation(
    const iree_hip_vmm_teardown_plan_t* plan,
    const iree_hip_vmm_allocation_t* allocation) {
  if (plan->scope == IREE_HIP_VMM_TEARDOWN_PROCESS_ALL) return true;
  return plan->scope == IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT &&
         allocation->ownership == IREE_HIP_VMM_ALLOCATION_OWNERSHIP_DEVICE &&
         allocation->device_ordinal == plan->target_device_ordinal;
}

static bool iree_hip_vmm_plan_removes_mapping(
    const iree_hip_vmm_teardown_plan_t* plan,
    const iree_hip_vmm_reservation_plan_t* reservation_plan,
    const iree_hip_vmm_mapping_t* mapping) {
  if (reservation_plan->retire) return true;
  const iree_hip_vmm_allocation_plan_t* allocation_plan =
      iree_hip_vmm_plan_find_allocation_const(plan, mapping->allocation);
  return allocation_plan && allocation_plan->retire;
}

static bool iree_hip_vmm_plan_removes_access_range(
    const iree_hip_vmm_teardown_plan_t* plan,
    const iree_hip_vmm_reservation_plan_t* reservation_plan,
    const iree_hip_vmm_access_range_t* range) {
  if (reservation_plan->retire) return true;
  return plan->scope == IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT &&
         range->location_type == hipMemLocationTypeDevice &&
         range->location_id == plan->target_device_ordinal;
}

static bool iree_hip_vmm_plan_removes_binding(
    const iree_hip_vmm_teardown_plan_t* plan,
    const iree_hip_vmm_reservation_plan_t* reservation_plan,
    const iree_hip_vmm_access_binding_t* binding) {
  if (reservation_plan->retire) return true;
  if ((plan->scope == IREE_HIP_VMM_TEARDOWN_EXPLICIT_CONTEXT_ALIASES_ONLY ||
       plan->scope == IREE_HIP_VMM_TEARDOWN_PRIMARY_CONTEXT) &&
      binding->streaming_buffer &&
      binding->streaming_buffer->context == plan->target_context) {
    return true;
  }
  return plan->scope == IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT &&
         binding->location_type == hipMemLocationTypeDevice &&
         binding->location_id == plan->target_device_ordinal;
}

static void iree_hip_vmm_plan_destroy_storage(
    iree_hip_vmm_teardown_plan_t* plan, bool cancel_publication_reservations) {
  if (!plan) return;

  // Native operations are destroyed before any reservation/allocation pin is
  // released: release/free operations retain borrowed record addresses.
  for (size_t i = 0; i < plan->operation_count; ++i) {
    hrx_vmm_native_operation_destroy(plan->operations[i]);
  }
  iree_allocator_free(iree_allocator_system(), plan->operations);

  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    if (cancel_publication_reservations) {
      for (size_t j = 0; j < reservation_plan->removed_binding_reserved_count;
           ++j) {
        iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
            reservation_plan->removed_bindings[j].streaming_buffer);
      }
    }
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->expected_mappings);
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->expected_native_segments);
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->expected_access_ranges);
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->expected_access_bindings);
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->final_mappings);
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->final_native_segments);
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->final_access_ranges);
    if (reservation_plan->final_bindings_are_new) {
      iree_hip_vmm_release_access_bindings(
          reservation_plan->final_access_bindings,
          reservation_plan->final_access_binding_count);
    } else {
      iree_allocator_free(iree_allocator_system(),
                          reservation_plan->final_access_bindings);
    }
    iree_allocator_free(iree_allocator_system(),
                        reservation_plan->removed_bindings);
    iree_hip_vmm_reservation_release(reservation_plan->reservation);
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    iree_hip_vmm_allocation_release(plan->allocations[i].allocation);
  }
  for (size_t i = 0; i < plan->incident_context_count; ++i) {
    iree_hal_streaming_context_release(plan->incident_contexts[i]);
  }
  iree_hal_streaming_context_release(plan->target_context);
  iree_allocator_free(iree_allocator_system(), plan->incident_contexts);
  iree_allocator_free(iree_allocator_system(), plan->reservations);
  iree_allocator_free(iree_allocator_system(), plan->allocations);
  iree_allocator_free(iree_allocator_system(), plan);
}

static hipError_t iree_hip_vmm_plan_snapshot_allocation(
    iree_hip_vmm_teardown_plan_t* plan,
    iree_hip_vmm_allocation_plan_t* allocation_plan) {
  iree_hip_vmm_allocation_t* allocation = allocation_plan->allocation;
  iree_slim_mutex_lock(&allocation->mutex);
  hipError_t result = hipSuccess;
  if (allocation->retiring || !allocation->physical_memory ||
      allocation->generation != plan->generation ||
      allocation->device_ordinal < 0 ||
      (size_t)allocation->device_ordinal >=
          iree_hip_vmm_registry.device_epoch_count ||
      allocation->owner_device_epoch !=
          iree_hip_vmm_device_epoch(allocation->device_ordinal)) {
    result = hipErrorInvalidValue;
  } else {
    allocation_plan->expected_public_reference_count =
        allocation->public_reference_count;
    allocation_plan->expected_mapping_reference_count =
        allocation->mapping_reference_count;
    allocation_plan->expected_physical_memory = allocation->physical_memory;
    allocation_plan->retire =
        iree_hip_vmm_plan_retires_allocation(plan, allocation);
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  if (result == hipSuccess && allocation_plan->retire) {
    result =
        iree_hip_vmm_plan_add_incident_context(plan, allocation->owner_context);
  }
  return result;
}

static bool iree_hip_vmm_access_ranges_can_merge(
    const iree_hip_vmm_access_range_t* lhs,
    const iree_hip_vmm_access_range_t* rhs) {
  return lhs->location_type == rhs->location_type &&
         lhs->location_id == rhs->location_id && lhs->flags == rhs->flags &&
         lhs->capability_id == rhs->capability_id &&
         lhs->offset <= SIZE_MAX - lhs->size &&
         lhs->offset + lhs->size == rhs->offset;
}

static hipError_t iree_hip_vmm_plan_build_final_access_ranges(
    const iree_hip_vmm_teardown_plan_t* plan,
    iree_hip_vmm_reservation_plan_t* reservation_plan) {
  const bool clips_removed_mappings = reservation_plan->final_mapping_count !=
                                      reservation_plan->expected_mapping_count;
  size_t maximum_count = reservation_plan->expected_access_range_count;
  if (clips_removed_mappings &&
      !iree_host_size_checked_mul(reservation_plan->expected_access_range_count,
                                  reservation_plan->final_mapping_count,
                                  &maximum_count)) {
    return hipErrorOutOfMemory;
  }
  hipError_t result = iree_hip_vmm_plan_allocate_array(
      maximum_count, sizeof(reservation_plan->final_access_ranges[0]),
      (void**)&reservation_plan->final_access_ranges);
  if (result != hipSuccess) return result;

  for (size_t i = 0; i < reservation_plan->expected_access_range_count; ++i) {
    const iree_hip_vmm_access_range_t* range =
        &reservation_plan->expected_access_ranges[i];
    if (iree_hip_vmm_plan_removes_access_range(plan, reservation_plan, range)) {
      continue;
    }
    if (range->offset > SIZE_MAX - range->size) return hipErrorInvalidValue;
    const size_t range_end = range->offset + range->size;

    const size_t mapping_count =
        clips_removed_mappings ? reservation_plan->final_mapping_count : 1;
    for (size_t j = 0; j < mapping_count; ++j) {
      size_t fragment_offset = range->offset;
      size_t fragment_end = range_end;
      if (clips_removed_mappings) {
        const iree_hip_vmm_mapping_t* mapping =
            &reservation_plan->final_mappings[j];
        if (mapping->virtual_offset > SIZE_MAX - mapping->size) {
          return hipErrorInvalidValue;
        }
        const size_t mapping_end = mapping->virtual_offset + mapping->size;
        fragment_offset = range->offset > mapping->virtual_offset
                              ? range->offset
                              : mapping->virtual_offset;
        fragment_end = range_end < mapping_end ? range_end : mapping_end;
        if (fragment_offset >= fragment_end) continue;
      }

      iree_hip_vmm_access_range_t fragment = *range;
      fragment.offset = fragment_offset;
      fragment.size = fragment_end - fragment_offset;
      if (reservation_plan->final_access_range_count != 0 &&
          iree_hip_vmm_access_ranges_can_merge(
              &reservation_plan->final_access_ranges
                   [reservation_plan->final_access_range_count - 1],
              &fragment)) {
        reservation_plan
            ->final_access_ranges[reservation_plan->final_access_range_count -
                                  1]
            .size += fragment.size;
      } else {
        IREE_ASSERT(reservation_plan->final_access_range_count < maximum_count);
        reservation_plan->final_access_ranges
            [reservation_plan->final_access_range_count++] = fragment;
      }
    }
  }
  return hipSuccess;
}

static hipError_t iree_hip_vmm_plan_snapshot_reservation(
    iree_hip_vmm_teardown_plan_t* plan,
    iree_hip_vmm_reservation_plan_t* reservation_plan) {
  iree_hip_vmm_reservation_t* reservation = reservation_plan->reservation;
  hipError_t result = hipSuccess;
  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || !reservation->virtual_buffer ||
      reservation->generation != plan->generation ||
      reservation->device_ordinal < 0 ||
      (size_t)reservation->device_ordinal >=
          iree_hip_vmm_registry.device_epoch_count ||
      reservation->owner_device_epoch !=
          iree_hip_vmm_device_epoch(reservation->device_ordinal)) {
    result = hipErrorInvalidValue;
    goto unlock;
  }

  reservation_plan->expected_mutation_serial = reservation->mutation_serial;
  reservation_plan->retire =
      iree_hip_vmm_plan_retires_reservation(plan, reservation);
  reservation_plan->expected_mapping_count = reservation->mapping_count;
  reservation_plan->expected_native_segment_count =
      reservation->native_segment_count;
  reservation_plan->expected_access_range_count =
      reservation->access_range_count;
  reservation_plan->expected_access_binding_count =
      reservation->access_binding_count;
  result = iree_hip_vmm_plan_copy_array(
      reservation->mappings, reservation->mapping_count,
      sizeof(reservation->mappings[0]),
      (void**)&reservation_plan->expected_mappings);
  if (result != hipSuccess) goto unlock;
  result = iree_hip_vmm_plan_copy_array(
      reservation->native_segments, reservation->native_segment_count,
      sizeof(reservation->native_segments[0]),
      (void**)&reservation_plan->expected_native_segments);
  if (result != hipSuccess) goto unlock;
  result = iree_hip_vmm_plan_copy_array(
      reservation->access_ranges, reservation->access_range_count,
      sizeof(reservation->access_ranges[0]),
      (void**)&reservation_plan->expected_access_ranges);
  if (result != hipSuccess) goto unlock;
  result = iree_hip_vmm_plan_copy_array(
      reservation->access_bindings, reservation->access_binding_count,
      sizeof(reservation->access_bindings[0]),
      (void**)&reservation_plan->expected_access_bindings);
  if (result != hipSuccess) goto unlock;

  size_t final_mapping_count = 0;
  for (size_t i = 0; i < reservation->mapping_count; ++i) {
    if (!iree_hip_vmm_plan_removes_mapping(plan, reservation_plan,
                                           &reservation->mappings[i])) {
      ++final_mapping_count;
    }
  }
  result = iree_hip_vmm_plan_allocate_array(
      final_mapping_count, sizeof(reservation->mappings[0]),
      (void**)&reservation_plan->final_mappings);
  if (result != hipSuccess) goto unlock;
  for (size_t i = 0; i < reservation->mapping_count; ++i) {
    iree_hip_vmm_mapping_t mapping = reservation->mappings[i];
    if (iree_hip_vmm_plan_removes_mapping(plan, reservation_plan, &mapping)) {
      iree_hip_vmm_allocation_plan_t* allocation_plan =
          iree_hip_vmm_plan_find_allocation(plan, mapping.allocation);
      if (!allocation_plan ||
          allocation_plan->removed_mapping_reference_count == UINT64_MAX) {
        result = hipErrorInvalidValue;
        goto unlock;
      }
      ++allocation_plan->removed_mapping_reference_count;
    } else {
      reservation_plan
          ->final_mappings[reservation_plan->final_mapping_count++] = mapping;
    }
  }

  size_t final_native_segment_count = 0;
  for (size_t i = 0; i < reservation->native_segment_count; ++i) {
    if (!iree_hip_vmm_plan_removes_mapping(plan, reservation_plan,
                                           &reservation->native_segments[i])) {
      ++final_native_segment_count;
    }
  }
  result = iree_hip_vmm_plan_allocate_array(
      final_native_segment_count, sizeof(reservation->native_segments[0]),
      (void**)&reservation_plan->final_native_segments);
  if (result != hipSuccess) goto unlock;
  for (size_t i = 0; i < reservation->native_segment_count; ++i) {
    iree_hip_vmm_mapping_t segment = reservation->native_segments[i];
    if (!iree_hip_vmm_plan_removes_mapping(plan, reservation_plan, &segment)) {
      reservation_plan->final_native_segments
          [reservation_plan->final_native_segment_count++] = segment;
    }
  }

  result = iree_hip_vmm_plan_build_final_access_ranges(plan, reservation_plan);
  if (result != hipSuccess) goto unlock;

  size_t final_binding_count = 0;
  size_t removed_binding_count = 0;
  const bool rebuild_bindings =
      reservation_plan->final_mapping_count != reservation->mapping_count &&
      !reservation_plan->retire;
  for (size_t i = 0; i < reservation->access_binding_count; ++i) {
    if (rebuild_bindings ||
        iree_hip_vmm_plan_removes_binding(plan, reservation_plan,
                                          &reservation->access_bindings[i])) {
      ++removed_binding_count;
    } else {
      ++final_binding_count;
    }
  }
  reservation_plan->final_bindings_are_new = rebuild_bindings;
  if (!rebuild_bindings) {
    result = iree_hip_vmm_plan_allocate_array(
        final_binding_count, sizeof(reservation->access_bindings[0]),
        (void**)&reservation_plan->final_access_bindings);
    if (result != hipSuccess) goto unlock;
  }
  result = iree_hip_vmm_plan_allocate_array(
      removed_binding_count, sizeof(reservation->access_bindings[0]),
      (void**)&reservation_plan->removed_bindings);
  if (result != hipSuccess) goto unlock;
  for (size_t i = 0; i < reservation->access_binding_count; ++i) {
    iree_hip_vmm_access_binding_t binding = reservation->access_bindings[i];
    if (rebuild_bindings ||
        iree_hip_vmm_plan_removes_binding(plan, reservation_plan, &binding)) {
      reservation_plan
          ->removed_bindings[reservation_plan->removed_binding_count++] =
          binding;
    } else {
      reservation_plan->final_access_bindings
          [reservation_plan->final_access_binding_count++] = binding;
    }
  }
  reservation_plan->affected =
      reservation_plan->retire ||
      reservation_plan->final_mapping_count != reservation->mapping_count ||
      reservation_plan->final_native_segment_count !=
          reservation->native_segment_count ||
      reservation_plan->final_access_range_count !=
          reservation->access_range_count ||
      reservation_plan->removed_binding_count != 0;
  for (size_t i = 0;
       i < reservation_plan->removed_binding_count && result == hipSuccess;
       ++i) {
    iree_hal_streaming_buffer_t* buffer =
        reservation_plan->removed_bindings[i].streaming_buffer;
    if (!buffer || buffer->has_reserved_insert) {
      result = hipErrorInvalidValue;
      break;
    }
    if (iree_hip_vmm_test_take_binding_rollback_prepare_failure(i + 1)) {
      result = hipErrorOutOfMemory;
      break;
    }
    iree_status_t status =
        iree_hal_streaming_memory_reserve_wrapped_buffer_publication(buffer);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_vmm_from_iree_status(status);
      break;
    }
    ++reservation_plan->removed_binding_reserved_count;
  }

unlock:
  iree_slim_mutex_unlock(&reservation->mutex);
  if (result != hipSuccess || !reservation_plan->affected) return result;

  if (reservation_plan->final_bindings_are_new) {
    result = iree_hip_vmm_build_access_bindings(
        reservation, reservation_plan->final_access_ranges,
        reservation_plan->final_access_range_count,
        &reservation_plan->final_access_bindings,
        &reservation_plan->final_access_binding_count);
    if (result != hipSuccess) return result;
  }

  result =
      iree_hip_vmm_plan_add_incident_context(plan, reservation->owner_context);
  for (size_t i = 0;
       i < reservation_plan->expected_mapping_count && result == hipSuccess;
       ++i) {
    result = iree_hip_vmm_plan_add_incident_context(
        plan, reservation_plan->expected_mappings[i].allocation->owner_context);
  }
  for (size_t i = 0; i < reservation_plan->expected_access_binding_count &&
                     result == hipSuccess;
       ++i) {
    iree_hal_streaming_buffer_t* buffer =
        reservation_plan->expected_access_bindings[i].streaming_buffer;
    result = iree_hip_vmm_plan_add_incident_context(
        plan, buffer ? buffer->context : NULL);
  }
  for (size_t i = 0;
       i < reservation_plan->final_access_binding_count && result == hipSuccess;
       ++i) {
    iree_hal_streaming_buffer_t* buffer =
        reservation_plan->final_access_bindings[i].streaming_buffer;
    result = iree_hip_vmm_plan_add_incident_context(
        plan, buffer ? buffer->context : NULL);
  }
  return result;
}

static bool iree_hip_vmm_plan_ranges_overlap(size_t lhs_offset, size_t lhs_size,
                                             size_t rhs_offset,
                                             size_t rhs_size) {
  if (lhs_size == 0 || rhs_size == 0) return false;
  return lhs_offset <= rhs_offset ? rhs_offset - lhs_offset < lhs_size
                                  : lhs_offset - rhs_offset < rhs_size;
}

static hipError_t iree_hip_vmm_plan_count_operation(size_t* operation_count) {
  if (*operation_count == SIZE_MAX) return hipErrorOutOfMemory;
  ++*operation_count;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_plan_prepare_native_journal(
    iree_hip_vmm_teardown_plan_t* plan) {
  size_t operation_count = 0;

  // Phase 1: exact access revokes on native segments that survive. Native
  // segments are already split at every access boundary.
  if (plan->scope == IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT) {
    for (size_t i = 0; i < plan->reservation_count; ++i) {
      iree_hip_vmm_reservation_plan_t* reservation_plan =
          &plan->reservations[i];
      if (reservation_plan->retire) continue;
      for (size_t j = 0; j < reservation_plan->expected_native_segment_count;
           ++j) {
        const iree_hip_vmm_mapping_t* segment =
            &reservation_plan->expected_native_segments[j];
        const iree_hip_vmm_allocation_plan_t* allocation_plan =
            iree_hip_vmm_plan_find_allocation_const(plan, segment->allocation);
        if (!allocation_plan || allocation_plan->retire) continue;
        for (size_t k = 0; k < reservation_plan->expected_access_range_count;
             ++k) {
          const iree_hip_vmm_access_range_t* range =
              &reservation_plan->expected_access_ranges[k];
          if (!iree_hip_vmm_plan_removes_access_range(plan, reservation_plan,
                                                      range) ||
              !iree_hip_vmm_plan_ranges_overlap(segment->virtual_offset,
                                                segment->size, range->offset,
                                                range->size)) {
            continue;
          }
          // build_native_segments includes every access boundary, so an
          // overlap with this target grant must cover the full native unit.
          if (segment->virtual_offset < range->offset ||
              segment->virtual_offset - range->offset > range->size ||
              segment->size >
                  range->size - (segment->virtual_offset - range->offset)) {
            return hipErrorInvalidValue;
          }
          hipError_t result =
              iree_hip_vmm_plan_count_operation(&operation_count);
          if (result != hipSuccess) return result;
        }
      }
    }
  }

  // Phase 2: exact native segments incident to a retiring reservation or
  // physical allocation.
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    for (size_t j = 0; j < reservation_plan->expected_native_segment_count;
         ++j) {
      const iree_hip_vmm_mapping_t* segment =
          &reservation_plan->expected_native_segments[j];
      const iree_hip_vmm_allocation_plan_t* allocation_plan =
          iree_hip_vmm_plan_find_allocation_const(plan, segment->allocation);
      if (!allocation_plan) return hipErrorInvalidValue;
      if (reservation_plan->retire || allocation_plan->retire) {
        hipError_t result = iree_hip_vmm_plan_count_operation(&operation_count);
        if (result != hipSuccess) return result;
      }
    }
  }
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    if (plan->reservations[i].retire) {
      hipError_t result = iree_hip_vmm_plan_count_operation(&operation_count);
      if (result != hipSuccess) return result;
    }
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    if (plan->allocations[i].retire) {
      hipError_t result = iree_hip_vmm_plan_count_operation(&operation_count);
      if (result != hipSuccess) return result;
    }
  }

  hipError_t result = iree_hip_vmm_plan_allocate_array(
      operation_count, sizeof(plan->operations[0]), (void**)&plan->operations);
  if (result != hipSuccess) return result;
  plan->operation_count = operation_count;
  size_t operation_index = 0;

  if (plan->scope == IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT) {
    hrx_device_t access_device = NULL;
    result = iree_hip_vmm_from_hrx_status(
        hrx_gpu_device_get(plan->target_device_ordinal, &access_device));
    if (result != hipSuccess) return result;
    for (size_t i = 0; i < plan->reservation_count; ++i) {
      iree_hip_vmm_reservation_plan_t* reservation_plan =
          &plan->reservations[i];
      if (reservation_plan->retire) continue;
      for (size_t j = 0; j < reservation_plan->expected_native_segment_count;
           ++j) {
        const iree_hip_vmm_mapping_t* segment =
            &reservation_plan->expected_native_segments[j];
        const iree_hip_vmm_allocation_plan_t* allocation_plan =
            iree_hip_vmm_plan_find_allocation_const(plan, segment->allocation);
        if (!allocation_plan || allocation_plan->retire) continue;
        for (size_t k = 0; k < reservation_plan->expected_access_range_count;
             ++k) {
          const iree_hip_vmm_access_range_t* range =
              &reservation_plan->expected_access_ranges[k];
          if (!iree_hip_vmm_plan_removes_access_range(plan, reservation_plan,
                                                      range) ||
              !iree_hip_vmm_plan_ranges_overlap(segment->virtual_offset,
                                                segment->size, range->offset,
                                                range->size)) {
            continue;
          }
          hrx_physical_memory_t physical_memory =
              segment->allocation->physical_memory;
          result = iree_hip_vmm_from_hrx_status(
              hrx_allocator_vmm_native_operation_prepare_access(
                  reservation_plan->reservation->allocator,
                  hrx_device_allocator(access_device),
                  reservation_plan->reservation->virtual_buffer,
                  segment->virtual_offset, segment->size,
                  /*affinity=*/0, HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
                  HRX_MEMORY_PROTECTION_NONE, /*physical_memory_count=*/1,
                  &physical_memory, &plan->operations[operation_index]));
          if (result != hipSuccess) return result;
          ++operation_index;
        }
      }
    }
  }

  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    for (size_t j = 0; j < reservation_plan->expected_native_segment_count;
         ++j) {
      const iree_hip_vmm_mapping_t* segment =
          &reservation_plan->expected_native_segments[j];
      const iree_hip_vmm_allocation_plan_t* allocation_plan =
          iree_hip_vmm_plan_find_allocation_const(plan, segment->allocation);
      if (!allocation_plan) return hipErrorInvalidValue;
      if (!reservation_plan->retire && !allocation_plan->retire) continue;
      result = iree_hip_vmm_from_hrx_status(
          hrx_allocator_vmm_native_operation_prepare_unmap(
              reservation_plan->reservation->allocator,
              reservation_plan->reservation->virtual_buffer,
              segment->virtual_offset, segment->size,
              &plan->operations[operation_index]));
      if (result != hipSuccess) return result;
      ++operation_index;
    }
  }
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    if (!reservation_plan->retire) continue;
    result = iree_hip_vmm_from_hrx_status(
        hrx_allocator_vmm_native_operation_prepare_release_reservation(
            reservation_plan->reservation->allocator,
            reservation_plan->reservation->virtual_buffer,
            &plan->operations[operation_index]));
    if (result != hipSuccess) return result;
    ++operation_index;
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    iree_hip_vmm_allocation_plan_t* allocation_plan = &plan->allocations[i];
    if (!allocation_plan->retire) continue;
    result = iree_hip_vmm_from_hrx_status(
        hrx_allocator_vmm_native_operation_prepare_free_physical(
            allocation_plan->allocation->allocator,
            allocation_plan->allocation->physical_memory,
            &plan->operations[operation_index]));
    if (result != hipSuccess) return result;
    ++operation_index;
  }
  IREE_ASSERT(operation_index == plan->operation_count);
  return operation_index == plan->operation_count ? hipSuccess
                                                  : hipErrorUnknown;
}

static bool iree_hip_vmm_plan_array_matches(const void* expected,
                                            const void* current, size_t count,
                                            size_t element_size) {
  return count == 0 || memcmp(expected, current, count * element_size) == 0;
}

static bool iree_hip_vmm_plan_revalidate(iree_hip_vmm_teardown_plan_t* plan) {
  if (iree_hip_vmm_test_take_plan_revalidation_failure()) return false;
  bool valid = true;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  valid = registry->generation == plan->generation &&
          registry->reservation_count == plan->reservation_count &&
          registry->allocation_count == plan->allocation_count &&
          !registry->prepared_cleanup && !registry->failed_cleanup;
  for (size_t i = 0; valid && i < plan->reservation_count; ++i) {
    valid = registry->reservations[i] == plan->reservations[i].reservation;
  }
  for (size_t i = 0; valid && i < plan->allocation_count; ++i) {
    valid = registry->allocations[i] == plan->allocations[i].allocation;
  }
  if (valid && plan->advances_device_epoch) {
    valid =
        plan->target_device_ordinal >= 0 &&
        (size_t)plan->target_device_ordinal < registry->device_epoch_count &&
        registry->device_epochs[plan->target_device_ordinal] ==
            plan->expected_vmm_device_epoch;
  }
  iree_hip_vmm_registry_unlock();
  if (!valid) return false;

  if (plan->advances_device_epoch) {
    iree_hal_streaming_device_t* device =
        iree_hal_streaming_device_entry(plan->target_device_ordinal);
    if (!device ||
        iree_atomic_load(&device->reset_epoch, iree_memory_order_acquire) !=
            plan->expected_common_device_epoch) {
      return false;
    }
  }

  for (size_t i = 0; i < plan->allocation_count; ++i) {
    iree_hip_vmm_allocation_plan_t* allocation_plan = &plan->allocations[i];
    iree_hip_vmm_allocation_t* allocation = allocation_plan->allocation;
    iree_slim_mutex_lock(&allocation->mutex);
    valid = !allocation->retiring &&
            allocation->generation == plan->generation &&
            allocation->public_reference_count ==
                allocation_plan->expected_public_reference_count &&
            allocation->mapping_reference_count ==
                allocation_plan->expected_mapping_reference_count &&
            allocation->physical_memory ==
                allocation_plan->expected_physical_memory;
    iree_slim_mutex_unlock(&allocation->mutex);
    if (!valid) return false;
  }

  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    iree_hip_vmm_reservation_t* reservation = reservation_plan->reservation;
    iree_slim_mutex_lock(&reservation->mutex);
    valid = !reservation->retiring && reservation->virtual_buffer &&
            reservation->generation == plan->generation &&
            reservation->mutation_serial ==
                reservation_plan->expected_mutation_serial &&
            reservation->mapping_count ==
                reservation_plan->expected_mapping_count &&
            reservation->native_segment_count ==
                reservation_plan->expected_native_segment_count &&
            reservation->access_range_count ==
                reservation_plan->expected_access_range_count &&
            reservation->access_binding_count ==
                reservation_plan->expected_access_binding_count &&
            iree_hip_vmm_plan_array_matches(
                reservation_plan->expected_mappings, reservation->mappings,
                reservation->mapping_count, sizeof(reservation->mappings[0])) &&
            iree_hip_vmm_plan_array_matches(
                reservation_plan->expected_native_segments,
                reservation->native_segments, reservation->native_segment_count,
                sizeof(reservation->native_segments[0])) &&
            iree_hip_vmm_plan_array_matches(
                reservation_plan->expected_access_ranges,
                reservation->access_ranges, reservation->access_range_count,
                sizeof(reservation->access_ranges[0])) &&
            iree_hip_vmm_plan_array_matches(
                reservation_plan->expected_access_bindings,
                reservation->access_bindings, reservation->access_binding_count,
                sizeof(reservation->access_bindings[0]));
    for (size_t j = 0; valid && j < reservation_plan->removed_binding_count;
         ++j) {
      iree_hal_streaming_buffer_t* buffer =
          reservation_plan->removed_bindings[j].streaming_buffer;
      valid =
          buffer && buffer->has_reserved_insert &&
          iree_hal_streaming_memory_is_wrapped_buffer_published_exactly_once(
              buffer);
    }
    valid = valid && reservation_plan->removed_binding_reserved_count ==
                         reservation_plan->removed_binding_count;
    for (size_t j = 0; valid && reservation_plan->final_bindings_are_new &&
                       j < reservation_plan->final_access_binding_count;
         ++j) {
      iree_hal_streaming_buffer_t* buffer =
          reservation_plan->final_access_bindings[j].streaming_buffer;
      valid = buffer && !buffer->is_published && buffer->has_reserved_insert;
    }
    iree_slim_mutex_unlock(&reservation->mutex);
    if (!valid) return false;
  }
  return true;
}

static hipError_t iree_hip_vmm_plan_synchronize_incident_readers(
    iree_hip_vmm_teardown_plan_t* plan) {
  iree_status_t status = iree_ok_status();
  iree_status_t execution_status = iree_ok_status();
  for (size_t i = 0; i < plan->incident_context_count; ++i) {
    iree_status_t context_execution_status = iree_ok_status();
#if defined(IREE_HIP_VMM_TESTING)
    iree_hip_vmm_test_incident_context_wait_t wait = {
        .context = plan->incident_contexts[i],
    };
    iree_status_t context_status =
        iree_hal_streaming_context_quiesce_for_teardown(
            plan->incident_contexts[i],
            iree_hip_vmm_test_notify_incident_context_stream_waiting, &wait,
            &context_execution_status);
    context_status = iree_status_join(
        context_status,
        iree_hip_vmm_test_verify_incident_context_stream_wait(&wait));
#else
    iree_status_t context_status =
        iree_hal_streaming_context_quiesce_for_teardown(
            plan->incident_contexts[i], NULL, NULL, &context_execution_status);
#endif
    status = iree_status_join(status, context_status);
    execution_status =
        iree_status_join(execution_status, context_execution_status);
  }
  // A verified terminal stream error is an execution result, not evidence of
  // work still accessing VMM state. Teardown reports only failures that kept it
  // from proving every incident context quiescent.
  iree_status_ignore(execution_status);
  return iree_hip_vmm_from_iree_status(status);
}

static hipError_t iree_hip_vmm_prepare_teardown_plan(
    iree_hip_vmm_teardown_scope_t scope, int target_device_ordinal,
    iree_hal_streaming_context_t* target_context,
    iree_hip_vmm_teardown_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  *out_plan = NULL;

  iree_hip_vmm_teardown_plan_t* plan = NULL;
  hipError_t result =
      iree_hip_vmm_plan_allocate_array(1, sizeof(*plan), (void**)&plan);
  if (result != hipSuccess) return result;
  plan->scope = scope;
  plan->state = IREE_HIP_VMM_TEARDOWN_PREPARED;
  plan->generation = iree_hip_vmm_registry.generation;
  plan->target_device_ordinal = target_device_ordinal;
  plan->target_context = target_context;
  iree_hal_streaming_context_retain(target_context);

  if (scope == IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT) {
    plan->advances_device_epoch = true;
    if (target_device_ordinal < 0 ||
        (size_t)target_device_ordinal >=
            iree_hip_vmm_registry.device_epoch_count) {
      result = hipErrorInvalidDevice;
      goto fail;
    }
    plan->expected_vmm_device_epoch =
        iree_hip_vmm_registry.device_epochs[target_device_ordinal];
    if (plan->expected_vmm_device_epoch >= UINT64_MAX - 1) {
      result = hipErrorOutOfMemory;
      goto fail;
    }
    plan->next_vmm_device_epoch = plan->expected_vmm_device_epoch + 1;
    iree_status_t status = iree_hal_streaming_device_prepare_epoch_advance(
        target_device_ordinal, &plan->expected_common_device_epoch,
        &plan->next_common_device_epoch);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_vmm_from_iree_status(status);
      goto fail;
    }
  }

  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  if (registry->prepared_cleanup || registry->failed_cleanup ||
      registry->generation != plan->generation) {
    iree_hip_vmm_registry_unlock();
    result = hipErrorNotInitialized;
    goto fail;
  }
  plan->reservation_count = registry->reservation_count;
  plan->allocation_count = registry->allocation_count;
  iree_hip_vmm_registry_unlock();

  result = iree_hip_vmm_plan_allocate_array(plan->reservation_count,
                                            sizeof(plan->reservations[0]),
                                            (void**)&plan->reservations);
  if (result != hipSuccess) goto fail;
  result = iree_hip_vmm_plan_allocate_array(plan->allocation_count,
                                            sizeof(plan->allocations[0]),
                                            (void**)&plan->allocations);
  if (result != hipSuccess) goto fail;

  registry = iree_hip_vmm_registry_lock();
  if (registry->prepared_cleanup || registry->failed_cleanup ||
      registry->generation != plan->generation ||
      registry->reservation_count != plan->reservation_count ||
      registry->allocation_count != plan->allocation_count) {
    iree_hip_vmm_registry_unlock();
    result = hipErrorInvalidValue;
    goto fail;
  }
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    plan->reservations[i].reservation = registry->reservations[i];
    iree_hip_vmm_reservation_retain(plan->reservations[i].reservation);
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    plan->allocations[i].allocation = registry->allocations[i];
    iree_hip_vmm_allocation_retain(plan->allocations[i].allocation);
  }
  iree_hip_vmm_registry_unlock();

  for (size_t i = 0; i < plan->allocation_count && result == hipSuccess; ++i) {
    result = iree_hip_vmm_plan_snapshot_allocation(plan, &plan->allocations[i]);
  }
  for (size_t i = 0; i < plan->reservation_count && result == hipSuccess; ++i) {
    result =
        iree_hip_vmm_plan_snapshot_reservation(plan, &plan->reservations[i]);
  }
  if (result != hipSuccess) goto fail;

  // Physical allocations whose final logical mapping and public-handle counts
  // are both zero are part of this exact transaction even when their declared
  // device policy did not independently select them.
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    iree_hip_vmm_allocation_plan_t* allocation_plan = &plan->allocations[i];
    if (allocation_plan->removed_mapping_reference_count >
        allocation_plan->expected_mapping_reference_count) {
      result = hipErrorInvalidValue;
      goto fail;
    }
    if (!allocation_plan->retire &&
        allocation_plan->expected_public_reference_count == 0 &&
        allocation_plan->removed_mapping_reference_count ==
            allocation_plan->expected_mapping_reference_count) {
      allocation_plan->retire = true;
    }
    if (allocation_plan->retire &&
        allocation_plan->removed_mapping_reference_count !=
            allocation_plan->expected_mapping_reference_count) {
      result = hipErrorInvalidValue;
      goto fail;
    }
    if (allocation_plan->retire) {
      result = iree_hip_vmm_plan_add_incident_context(
          plan, allocation_plan->allocation->owner_context);
      if (result != hipSuccess) goto fail;
    }
  }

  // A natural last-edge retirement discovered above may make a stale native
  // segment newly incident. Compact the already allocated final vectors; the
  // original expected vectors remain untouched for revalidation and journal
  // preparation.
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    size_t write_index = 0;
    for (size_t j = 0; j < reservation_plan->final_native_segment_count; ++j) {
      iree_hip_vmm_mapping_t segment =
          reservation_plan->final_native_segments[j];
      const iree_hip_vmm_allocation_plan_t* allocation_plan =
          iree_hip_vmm_plan_find_allocation_const(plan, segment.allocation);
      if (!allocation_plan) {
        result = hipErrorInvalidValue;
        goto fail;
      }
      if (!allocation_plan->retire) {
        reservation_plan->final_native_segments[write_index++] = segment;
      }
    }
    if (write_index != reservation_plan->final_native_segment_count) {
      reservation_plan->affected = true;
      reservation_plan->final_native_segment_count = write_index;
    }
  }

  result = iree_hip_vmm_plan_synchronize_incident_readers(plan);
  if (result != hipSuccess) goto fail;
  result = iree_hip_vmm_plan_prepare_native_journal(plan);
  if (result != hipSuccess) goto fail;
  if (!iree_hip_vmm_plan_revalidate(plan)) {
    result = hipErrorInvalidValue;
    goto fail;
  }

  registry = iree_hip_vmm_registry_lock();
  if (registry->prepared_cleanup || registry->failed_cleanup ||
      registry->generation != plan->generation) {
    iree_hip_vmm_registry_unlock();
    result = hipErrorInvalidValue;
    goto fail;
  }
  registry->prepared_cleanup = plan;
  iree_hip_vmm_registry_unlock();
  *out_plan = plan;
  return hipSuccess;

fail:
  iree_hip_vmm_plan_destroy_storage(plan,
                                    /*cancel_publication_reservations=*/true);
  return result;
}

hipError_t iree_hip_vmm_prepare_device_reset(
    int device_ordinal, iree_hal_streaming_context_t* primary_context,
    bool whole_device, iree_hip_vmm_teardown_plan_t** out_plan) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  if (!out_plan) return hipErrorInvalidValue;
  *out_plan = NULL;
  iree_hal_streaming_device_t* device = NULL;
  hipError_t result =
      iree_hip_vmm_resolve_device_under_writer(device_ordinal, &device);
  if (result != hipSuccess) return result;
  if (primary_context && primary_context->device_entry != device) {
    return hipErrorInvalidContext;
  }
  if (whole_device) {
#if defined(IREE_HIP_VMM_TESTING)
    result = iree_hip_vmm_test_apply_epoch_exhaustion(device_ordinal);
    if (result != hipSuccess) return result;
#endif
  }
  return iree_hip_vmm_prepare_teardown_plan(
      whole_device ? IREE_HIP_VMM_TEARDOWN_DEVICE_INCIDENT
                   : IREE_HIP_VMM_TEARDOWN_PRIMARY_CONTEXT,
      device_ordinal, primary_context, out_plan);
}

hipError_t iree_hip_vmm_prepare_explicit_context_destroy(
    iree_hal_streaming_context_t* context,
    iree_hip_vmm_teardown_plan_t** out_plan) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  if (!context || !out_plan) return hipErrorInvalidValue;
  return iree_hip_vmm_prepare_teardown_plan(
      IREE_HIP_VMM_TEARDOWN_EXPLICIT_CONTEXT_ALIASES_ONLY,
      (int)context->device_ordinal, context, out_plan);
}

hipError_t iree_hip_vmm_prepare_process_cleanup(
    iree_hip_vmm_teardown_plan_t** out_plan) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  if (!out_plan) return hipErrorInvalidValue;
  return iree_hip_vmm_prepare_teardown_plan(IREE_HIP_VMM_TEARDOWN_PROCESS_ALL,
                                            /*target_device_ordinal=*/-1,
                                            /*target_context=*/NULL, out_plan);
}

void iree_hip_vmm_cancel_teardown_plan(iree_hip_vmm_teardown_plan_t* plan) {
  if (!plan) return;
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  IREE_ASSERT(plan->state == IREE_HIP_VMM_TEARDOWN_PREPARED);
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  IREE_ASSERT(registry->prepared_cleanup == plan);
  if (registry->prepared_cleanup == plan) registry->prepared_cleanup = NULL;
  iree_hip_vmm_registry_unlock();
  iree_hip_vmm_plan_destroy_storage(plan,
                                    /*cancel_publication_reservations=*/true);
}

static void iree_hip_vmm_plan_commit_public_state(
    iree_hip_vmm_teardown_plan_t* plan) {
  IREE_ASSERT(plan->state == IREE_HIP_VMM_TEARDOWN_SEALED);
  if (plan->advances_device_epoch) {
    IREE_ASSERT(
        iree_hip_vmm_registry.device_epochs[plan->target_device_ordinal] ==
        plan->expected_vmm_device_epoch);
    if (iree_hip_vmm_registry.device_epochs[plan->target_device_ordinal] ==
        plan->expected_vmm_device_epoch) {
      iree_hip_vmm_registry.device_epochs[plan->target_device_ordinal] =
          plan->next_vmm_device_epoch;
    } else {
      // An exclusive-writer invariant violation must never publish a reusable
      // identity in a release build.
      iree_hip_vmm_registry.device_epochs[plan->target_device_ordinal] =
          UINT64_MAX;
    }
    iree_hal_streaming_device_commit_epoch_advance(
        plan->target_device_ordinal, plan->expected_common_device_epoch,
        plan->next_common_device_epoch);
  }

  // Every exact wrapper was validated and owns rollback capacity. Commit is
  // allocation-free and cannot fail after the caller's first queue seal.
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    IREE_ASSERT(reservation_plan->removed_binding_reserved_count ==
                reservation_plan->removed_binding_count);
    for (size_t j = 0; j < reservation_plan->removed_binding_count; ++j) {
      iree_hal_streaming_memory_commit_unpublish_wrapped_buffer(
          reservation_plan->removed_bindings[j].streaming_buffer);
    }
  }
  plan->state = IREE_HIP_VMM_TEARDOWN_PUBLIC_COMMIT;
}

static void iree_hip_vmm_plan_fail_closed(
    iree_hip_vmm_teardown_plan_t* plan, hrx_vmm_native_status_t native_status) {
  if (!plan->has_first_error) {
    plan->first_error = native_status;
    plan->has_first_error = true;
    iree_hip_vmm_test_record_first_native_error(native_status);
  }
  plan->state = IREE_HIP_VMM_TEARDOWN_FAILED_CLOSED;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  IREE_ASSERT(!registry->failed_cleanup || registry->failed_cleanup == plan);
  registry->failed_cleanup = plan;
  iree_hip_vmm_registry_unlock();

  // A partially consumed graph is teardown-only process-wide. Preserve every
  // registry/device/context edge as the externally visible retry ledger.
  iree_hal_streaming_context_retire_all();
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  iree_hip_vmm_registry.active = false;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
}

static bool iree_hip_vmm_plan_commit_publish_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  if (buffer->is_published) return true;
  if (!buffer->has_reserved_insert) return false;
  hrx_status_t status = hrx_buffer_table_insert_reserved(
      &buffer->context->buffer_table, buffer->device_ptr, buffer->host_ptr,
      buffer->size, buffer->hrx_buf, buffer);
  const bool succeeded = hrx_status_is_ok(status);
  hrx_status_ignore(status);
  if (succeeded) {
    buffer->has_reserved_insert = false;
    buffer->is_published = true;
  }
  return succeeded;
}

static bool iree_hip_vmm_plan_finalize(iree_hip_vmm_teardown_plan_t* plan) {
  IREE_ASSERT(plan->cursor == plan->operation_count);

  // Split aliases were fully allocated and their table insertion capacity was
  // reserved before seal. Publish them only after all native mutations have
  // succeeded. An invariant-level insertion error retains the cursor-at-end
  // plan so inactive retry never replays native operations.
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    if (!reservation_plan->final_bindings_are_new) continue;
    for (size_t j = 0; j < reservation_plan->final_access_binding_count; ++j) {
      if (!iree_hip_vmm_plan_commit_publish_wrapped_buffer(
              reservation_plan->final_access_bindings[j].streaming_buffer)) {
        return false;
      }
    }
  }

  plan->state = IREE_HIP_VMM_TEARDOWN_FINALIZING;

  // Operations retain their native domain and may borrow reservation/physical
  // records. Destroy every operation before consuming any owning wrapper.
  for (size_t i = 0; i < plan->operation_count; ++i) {
    hrx_vmm_native_operation_destroy(plan->operations[i]);
    plan->operations[i] = NULL;
  }
  iree_allocator_free(iree_allocator_system(), plan->operations);
  plan->operations = NULL;
  plan->operation_count = 0;
  plan->cursor = 0;

  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    if (!reservation_plan->retire) continue;
    IREE_ASSERT(reservation_plan->reservation->virtual_buffer);
    hrx_allocator_virtual_memory_dispose_consumed_reservation(
        reservation_plan->reservation->allocator,
        reservation_plan->reservation->virtual_buffer);
    reservation_plan->reservation->virtual_buffer = NULL;
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    iree_hip_vmm_allocation_plan_t* allocation_plan = &plan->allocations[i];
    if (!allocation_plan->retire) continue;
    IREE_ASSERT(allocation_plan->allocation->physical_memory);
    hrx_allocator_physical_memory_dispose_consumed(
        allocation_plan->allocation->allocator,
        allocation_plan->allocation->physical_memory);
    allocation_plan->allocation->physical_memory = NULL;
  }

  for (size_t i = 0; i < plan->reservation_count; ++i) {
    iree_hip_vmm_reservation_plan_t* reservation_plan = &plan->reservations[i];
    if (!reservation_plan->affected) continue;
    iree_hip_vmm_reservation_t* reservation = reservation_plan->reservation;
    iree_slim_mutex_lock(&reservation->mutex);
    IREE_ASSERT(reservation->mutation_serial ==
                reservation_plan->expected_mutation_serial);
    iree_hip_vmm_mapping_t* old_mappings = reservation->mappings;
    iree_hip_vmm_mapping_t* old_native_segments = reservation->native_segments;
    iree_hip_vmm_access_range_t* old_access_ranges = reservation->access_ranges;
    iree_hip_vmm_access_binding_t* old_access_bindings =
        reservation->access_bindings;

    reservation->mappings = reservation_plan->final_mappings;
    reservation->mapping_count = reservation_plan->final_mapping_count;
    reservation->mapping_capacity = reservation_plan->final_mapping_count;
    reservation_plan->final_mappings = NULL;
    reservation_plan->final_mapping_count = 0;
    reservation->native_segments = reservation_plan->final_native_segments;
    reservation->native_segment_count =
        reservation_plan->final_native_segment_count;
    reservation->native_segment_capacity =
        reservation_plan->final_native_segment_count;
    reservation_plan->final_native_segments = NULL;
    reservation_plan->final_native_segment_count = 0;
    reservation->access_ranges = reservation_plan->final_access_ranges;
    reservation->access_range_count =
        reservation_plan->final_access_range_count;
    reservation_plan->final_access_ranges = NULL;
    reservation_plan->final_access_range_count = 0;
    reservation->access_bindings = reservation_plan->final_access_bindings;
    reservation->access_binding_count =
        reservation_plan->final_access_binding_count;
    reservation_plan->final_access_bindings = NULL;
    reservation_plan->final_access_binding_count = 0;
    reservation_plan->final_bindings_are_new = false;
    reservation->retiring = reservation_plan->retire;
    reservation->poisoned = false;
    ++reservation->mutation_serial;
    iree_slim_mutex_unlock(&reservation->mutex);

    // Survivor wrapper ownership moved into the final array. The old backing
    // array is freed without releasing those shallow entries; only exact
    // removed wrappers are released.
    iree_allocator_free(iree_allocator_system(), old_access_bindings);
    iree_allocator_free(iree_allocator_system(), old_access_ranges);
    iree_allocator_free(iree_allocator_system(), old_native_segments);
    iree_allocator_free(iree_allocator_system(), old_mappings);
    iree_hip_vmm_release_access_bindings(
        reservation_plan->removed_bindings,
        reservation_plan->removed_binding_count);
    reservation_plan->removed_bindings = NULL;
    reservation_plan->removed_binding_count = 0;
    reservation_plan->removed_binding_reserved_count = 0;
  }

  for (size_t i = 0; i < plan->allocation_count; ++i) {
    iree_hip_vmm_allocation_plan_t* allocation_plan = &plan->allocations[i];
    iree_hip_vmm_allocation_t* allocation = allocation_plan->allocation;
    iree_slim_mutex_lock(&allocation->mutex);
    IREE_ASSERT(allocation->mapping_reference_count ==
                allocation_plan->expected_mapping_reference_count);
    allocation->mapping_reference_count =
        allocation_plan->expected_mapping_reference_count -
        allocation_plan->removed_mapping_reference_count;
    if (allocation_plan->retire) {
      IREE_ASSERT(allocation->mapping_reference_count == 0);
      allocation->public_reference_count = 0;
      allocation->retiring = true;
    }
    iree_slim_mutex_unlock(&allocation->mutex);
  }

  // Registry edges are removed only after every native dependency completed
  // and all final arrays are installed. The plan still pins each exact node.
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    if (!plan->reservations[i].retire) continue;
    const bool removed = iree_hip_vmm_registry_remove_reservation_locked(
        registry, plan->reservations[i].reservation);
    IREE_ASSERT(removed);
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    if (!plan->allocations[i].retire) continue;
    const bool removed = iree_hip_vmm_registry_remove_allocation_locked(
        registry, plan->allocations[i].allocation);
    IREE_ASSERT(removed);
  }
  if (registry->failed_cleanup == plan) registry->failed_cleanup = NULL;
  IREE_ASSERT(registry->prepared_cleanup != plan);
  iree_hip_vmm_registry_unlock();

  // Balance the registry references independently of the plan pins.
  for (size_t i = 0; i < plan->reservation_count; ++i) {
    if (plan->reservations[i].retire) {
      iree_hip_vmm_reservation_release(plan->reservations[i].reservation);
    }
  }
  for (size_t i = 0; i < plan->allocation_count; ++i) {
    if (plan->allocations[i].retire) {
      iree_hip_vmm_allocation_release(plan->allocations[i].allocation);
    }
  }

  plan->state = IREE_HIP_VMM_TEARDOWN_COMPLETE;
  iree_hip_vmm_plan_destroy_storage(plan,
                                    /*cancel_publication_reservations=*/false);
  return true;
}

hipError_t iree_hip_vmm_commit_teardown_plan(
    iree_hip_vmm_teardown_plan_t* plan) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  if (!plan) return hipErrorInvalidValue;

  if (plan->state == IREE_HIP_VMM_TEARDOWN_PREPARED) {
    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    if (registry->prepared_cleanup != plan || registry->failed_cleanup) {
      iree_hip_vmm_registry_unlock();
      return hipErrorInvalidValue;
    }
    registry->prepared_cleanup = NULL;
    iree_hip_vmm_registry_unlock();
    // The caller invokes commit only after sealing its exact target queues.
    plan->state = IREE_HIP_VMM_TEARDOWN_SEALED;
    iree_hip_vmm_plan_commit_public_state(plan);
  } else if (plan->state == IREE_HIP_VMM_TEARDOWN_FAILED_CLOSED) {
    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    const bool exact_retry = registry->failed_cleanup == plan;
    iree_hip_vmm_registry_unlock();
    if (!exact_retry) return hipErrorInvalidValue;
  } else {
    return hipErrorInvalidValue;
  }

  plan->state = IREE_HIP_VMM_TEARDOWN_NATIVE;
  iree_hip_vmm_test_record_native_journal_begin(plan->operation_count,
                                                plan->cursor);
  while (plan->cursor < plan->operation_count) {
    hrx_vmm_native_operation_t operation = plan->operations[plan->cursor];
    const bool callback_window =
        hrx_vmm_native_operation_may_invoke_callbacks(operation);
    if (callback_window) {
      iree_hip_vmm_native_callback_window_begin(operation);
    }
    const bool inject_failure =
        iree_hip_vmm_test_take_native_operation_failure(plan->cursor);
    const hrx_vmm_native_status_t native_status =
        inject_failure ? UINT32_MAX - 1
                       : hrx_vmm_native_operation_apply(operation);
    if (!inject_failure) iree_hip_vmm_test_record_raw_native_call();
    if (callback_window) {
      iree_hip_vmm_native_callback_window_end();
    }
    if (!hrx_vmm_native_status_is_success(native_status)) {
      iree_hip_vmm_plan_fail_closed(plan, native_status);
      return hipErrorUnknown;
    }
    ++plan->cursor;
    iree_hip_vmm_test_record_native_operation_success(plan->cursor);
  }

  if (!iree_hip_vmm_plan_finalize(plan)) {
    iree_hip_vmm_plan_fail_closed(plan, UINT32_MAX);
    return hipErrorUnknown;
  }
  return hipSuccess;
}

static hipError_t iree_hip_vmm_resume_failed_teardown(void) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  iree_hip_vmm_teardown_plan_t* plan = registry->failed_cleanup;
  iree_hip_vmm_registry_unlock();
  return plan ? iree_hip_vmm_commit_teardown_plan(plan) : hipSuccess;
}

hipError_t iree_hip_vmm_device_reset_begin(
    int device_ordinal, iree_hal_streaming_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_BEFORE_DEVICE_WRITER,
                                 NULL);
  hipError_t result = iree_hip_vmm_writer_begin(/*require_active=*/true);
  if (result != hipSuccess) return result;
  result = iree_hip_vmm_resolve_device_under_writer(device_ordinal, out_device);
  if (result == hipSuccess) {
    result = iree_hip_vmm_quarantine_drain_device(*out_device);
    if (result != hipSuccess) *out_device = NULL;
  }
  if (result != hipSuccess) iree_hip_vmm_writer_end();
  return result;
}

hipError_t iree_hip_vmm_device_reset_handoff_begin(
    int device_ordinal, iree_hal_streaming_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  iree_hip_vmm_test_notify_phase(IREE_HIP_VMM_TEST_PHASE_BEFORE_DEVICE_WRITER,
                                 NULL);
  bool writer_started_active = false;
  uint64_t generation = 0;
  hipError_t result = iree_hip_vmm_writer_begin_from_reader(
      &writer_started_active, &generation);
  if (result != hipSuccess) return result;
  if (!writer_started_active || generation == 0) {
    iree_hip_vmm_writer_end();
    return hipErrorNotInitialized;
  }
  result = iree_hip_vmm_resolve_device_under_writer(device_ordinal, out_device);
  if (result == hipSuccess) {
    result = iree_hip_vmm_quarantine_drain_device(*out_device);
    if (result != hipSuccess) *out_device = NULL;
  }
  if (result != hipSuccess) iree_hip_vmm_writer_end();
  return result;
}

void iree_hip_vmm_device_reset_end(bool cleanup_committed,
                                   bool reset_succeeded) {
  if (cleanup_committed && !reset_succeeded) {
    // VMM admission is process-wide, so a destructive failure on one device
    // makes every context in this generation teardown-only.
    iree_hal_streaming_context_retire_all();
    iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    iree_hip_vmm_registry.active = false;
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  }
  iree_hip_vmm_writer_end();
}

hipError_t iree_hip_vmm_prepare_primary_context_release(
    int device_ordinal, iree_hal_streaming_context_t* expected_primary_context,
    iree_hip_vmm_primary_release_t* out_release) {
  IREE_ASSERT_ARGUMENT(out_release);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  memset(out_release, 0, sizeof(*out_release));
  iree_hip_vmm_test_notify_phase(
      IREE_HIP_VMM_TEST_PHASE_PRIMARY_RELEASE_PREPARE,
      expected_primary_context);

  iree_hal_streaming_device_t* device = NULL;
  hipError_t result =
      iree_hip_vmm_resolve_device_under_writer(device_ordinal, &device);
  if (result != hipSuccess) return result;

  iree_hal_streaming_context_t* context = NULL;
  int32_t expected_ref_count = 0;
  iree_slim_mutex_lock(&device->primary_context_mutex);
  context = device->primary_context;
  expected_ref_count = device->primary_context_ref_count;
  if (context && expected_ref_count > 0 &&
      (!expected_primary_context || context == expected_primary_context)) {
    iree_hal_streaming_context_retain(context);
  } else {
    context = NULL;
  }
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  if (!context || expected_ref_count <= 0) return hipErrorInvalidContext;

  iree_hip_context_teardown_t* teardown = NULL;
  iree_hip_vmm_teardown_plan_t* vmm_plan = NULL;
  if (expected_ref_count == 1) {
    // The last retain owns the entire primary context, including ordinary and
    // CU-mask streams that are not members of the execution context initiating
    // this release. Capture cleanup can release graphs and pageable staging,
    // so require the caller to end/abort every capture while the transaction
    // is still fully retryable.
    if (iree_hip_vmm_test_take_primary_release_prepare_failure()) {
      result = hipErrorUnknown;
    } else {
      result = iree_hip_context_teardown_prepare(
          &context, 1, /*device_filter=*/-1,
          /*abort_captures=*/false,
          /*invalidate_stream_handles=*/false,
          /*seal_device_owned_queues=*/false,
          /*allow_retired_synchronization=*/false, &teardown);
      if (result == hipSuccess) {
        result = iree_hip_vmm_prepare_device_reset(
            device_ordinal, context, /*whole_device=*/false, &vmm_plan);
      }
    }
  }

  // Bind preparation to the exact state that was quiesced. Lifecycle writer
  // admission excludes normal retain/release operations; this validation also
  // fails closed if an internal caller violated that contract.
  if (result == hipSuccess) {
    iree_slim_mutex_lock(&device->primary_context_mutex);
    const bool exact_state =
        device->primary_context == context &&
        device->primary_context_ref_count == expected_ref_count;
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    if (!exact_state) result = hipErrorInvalidContext;
  }
  if (result != hipSuccess) {
    iree_hip_vmm_cancel_teardown_plan(vmm_plan);
    iree_hip_context_teardown_cancel(teardown);
    iree_hal_streaming_context_release(context);
    return result;
  }

  out_release->device = device;
  out_release->context = context;
  out_release->expected_ref_count = expected_ref_count;
  out_release->teardown = teardown;
  out_release->vmm_plan = vmm_plan;
  return hipSuccess;
}

void iree_hip_vmm_cancel_primary_context_release(
    iree_hip_vmm_primary_release_t* release) {
  if (!release || !release->context) return;
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  iree_hip_vmm_cancel_teardown_plan(release->vmm_plan);
  iree_hip_context_teardown_cancel(release->teardown);
  iree_hal_streaming_context_release(release->context);
  memset(release, 0, sizeof(*release));
}

hipError_t iree_hip_vmm_commit_primary_context_release(
    iree_hip_vmm_primary_release_t* release, bool* out_cleanup_committed,
    iree_hip_context_teardown_t** out_committed_teardown) {
  IREE_ASSERT_ARGUMENT(release);
  IREE_ASSERT_ARGUMENT(out_cleanup_committed);
  IREE_ASSERT_ARGUMENT(out_committed_teardown);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  *out_cleanup_committed = false;
  *out_committed_teardown = NULL;
  if (!release->device || !release->context ||
      release->expected_ref_count <= 0) {
    return hipErrorInvalidContext;
  }

  hipError_t result = hipSuccess;
  if (release->expected_ref_count == 1) {
    IREE_ASSERT(release->teardown);
    IREE_ASSERT(release->vmm_plan);
    iree_hip_context_teardown_seal_queues(release->teardown);
    iree_hip_context_teardown_commit(release->teardown);
    *out_cleanup_committed = true;
    *out_committed_teardown = release->teardown;
    release->teardown = NULL;
    result = iree_hip_vmm_commit_teardown_plan(release->vmm_plan);
    release->vmm_plan = NULL;
  }

  if (result != hipSuccess) {
    iree_hip_vmm_assert_preserved_context_bindings(release->context);
  }

  iree_status_t status =
      result == hipSuccess
          ? iree_hal_streaming_device_commit_primary_context_release(
                release->device, release->context, release->expected_ref_count)
          : iree_hal_streaming_device_commit_primary_context_release_preserving_ledger(
                release->device, release->context, release->expected_ref_count);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    if (result == hipSuccess) result = hipErrorInvalidContext;
  }
  iree_hal_streaming_context_release(release->context);
  memset(release, 0, sizeof(*release));
  return result;
}

hipError_t iree_hip_vmm_release_primary_context(int device_ordinal) {
  iree_hal_streaming_device_t* ignored_device = NULL;
  hipError_t result =
      iree_hip_vmm_device_reset_begin(device_ordinal, &ignored_device);
  if (result != hipSuccess) return result;

  iree_hip_vmm_primary_release_t release = {0};
  iree_hip_context_teardown_t* teardown = NULL;
  bool cleanup_committed = false;
  result = iree_hip_vmm_prepare_primary_context_release(
      device_ordinal, /*expected_primary_context=*/NULL, &release);
  if (result == hipSuccess) {
    result = iree_hip_vmm_commit_primary_context_release(
        &release, &cleanup_committed, &teardown);
  }
  iree_hip_vmm_device_reset_end(cleanup_committed, result == hipSuccess);
  iree_hip_context_teardown_finish(teardown);
  return result;
}

hipError_t iree_hip_vmm_deinitialize_begin(void) {
  // Also admits cleanup retry after a prior reset/deinitialize transaction
  // failed closed with quarantined registry objects still present.
  hipError_t result = iree_hip_vmm_writer_begin(/*require_active=*/false);
  if (result != hipSuccess) return result;
  result = iree_hip_vmm_resume_failed_teardown();
  if (result == hipSuccess) {
    result = iree_hip_vmm_quarantine_drain_all_devices();
  }
  if (result != hipSuccess) iree_hip_vmm_writer_end();
  return result;
}

hipError_t iree_hip_vmm_deinitialize_prepare(
    iree_hip_vmm_teardown_plan_t** out_plan) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  return iree_hip_vmm_prepare_process_cleanup(out_plan);
}

void iree_hip_vmm_deinitialize_commit(void) {
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  // hipHALDeinit proved every registered context's accepted frontier complete
  // or terminal before this irreversible transition. Certify that fact while
  // admission remains closed so final releases cannot perform a second wait.
  iree_hal_streaming_context_retire_all();
  iree_hal_streaming_context_mark_all_teardown_quiesced();
  iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  iree_hip_vmm_registry.active = false;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
}

void iree_hip_vmm_deinitialize_end(bool cleanup_committed,
                                   bool teardown_succeeded) {
  if (teardown_succeeded) {
    iree_hip_vmm_registry.device_epoch_count = 0;
  } else if (!cleanup_committed) {
    // Synchronization failed before any destructive transition; the old
    // generation is still coherent and may resume.
    iree_slim_mutex_lock(&iree_hip_vmm_registry.lifecycle_state_mutex);
    iree_hip_vmm_registry.active = iree_hip_vmm_registry.writer_started_active;
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.lifecycle_state_mutex);
  }
  iree_hip_vmm_writer_end();
}

size_t iree_hip_vmm_context_owned_reference_count(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  size_t count = 0;
  // Retirement takes reservation->mutex before registry.mutex. Snapshot and
  // pin one reservation at a time so ownership validation never establishes
  // the inverse nested order. Writer admission keeps registry membership
  // stable for the duration of this scan.
  for (size_t i = 0;; ++i) {
    iree_hip_vmm_reservation_t* reservation = NULL;
    iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
    if (i < iree_hip_vmm_registry.reservation_count) {
      reservation = iree_hip_vmm_registry.reservations[i];
      iree_hip_vmm_reservation_retain(reservation);
    }
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);
    if (!reservation) break;

    iree_slim_mutex_lock(&reservation->mutex);
    if (reservation->owner_context == context) ++count;
    for (size_t j = 0; j < reservation->access_binding_count; ++j) {
      iree_hal_streaming_buffer_t* buffer =
          reservation->access_bindings[j].streaming_buffer;
      if (buffer && buffer->context == context) ++count;
    }
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
  }

  iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
  for (size_t i = 0; i < iree_hip_vmm_registry.allocation_count; ++i) {
    if (iree_hip_vmm_registry.allocations[i]->owner_context == context) {
      ++count;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);
  return count;
}

static size_t iree_hip_vmm_count_live_binding_names(
    iree_hal_streaming_buffer_t* wrapper) {
  size_t count = 0;
  // Pin one reservation under the registry mutex, then inspect it after
  // dropping that mutex. This never nests registry->reservation and therefore
  // cannot invert the retirement path's reservation->registry order.
  for (size_t i = 0;; ++i) {
    iree_hip_vmm_reservation_t* reservation = NULL;
    iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
    if (i < iree_hip_vmm_registry.reservation_count) {
      reservation = iree_hip_vmm_registry.reservations[i];
      iree_hip_vmm_reservation_retain(reservation);
    }
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);
    if (!reservation) break;

    iree_slim_mutex_lock(&reservation->mutex);
    for (size_t j = 0; j < reservation->access_binding_count; ++j) {
      if (reservation->access_bindings[j].streaming_buffer == wrapper) ++count;
    }
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
  }
  return count;
}

void iree_hip_vmm_assert_preserved_context_bindings(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT(iree_hip_vmm_registry.writer_pending);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
  const bool has_failed_plan = iree_hip_vmm_registry.failed_cleanup != NULL;
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);

  // Check binding -> wrapper/table. Snapshot each wrapper with its own retain
  // while holding the reservation lock, then drop that lock before acquiring
  // the pointer-table or other reservation locks. Exclusive lifecycle writer
  // admission makes the snapshot immutable apart from these owned pins.
  for (size_t r = 0;; ++r) {
    iree_hip_vmm_reservation_t* reservation = NULL;
    iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
    if (r < iree_hip_vmm_registry.reservation_count) {
      reservation = iree_hip_vmm_registry.reservations[r];
      iree_hip_vmm_reservation_retain(reservation);
    }
    iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);
    if (!reservation) break;

    for (size_t b = 0;; ++b) {
      iree_hal_streaming_buffer_t* wrapper = NULL;
      iree_slim_mutex_lock(&reservation->mutex);
      if (b < reservation->access_binding_count) {
        iree_hal_streaming_buffer_t* candidate =
            reservation->access_bindings[b].streaming_buffer;
        if (candidate && candidate->context == context) {
          wrapper = candidate;
          iree_hal_streaming_buffer_retain(wrapper);
        }
      }
      const bool at_end = b >= reservation->access_binding_count;
      iree_slim_mutex_unlock(&reservation->mutex);
      if (at_end) break;
      if (!wrapper) continue;

      IREE_ASSERT(wrapper->is_virtual_memory_access,
                  "context VMM binding must name a VMM wrapper");
      if (wrapper->is_published) {
        IREE_ASSERT(
            iree_hal_streaming_memory_is_wrapped_buffer_published_exactly_once(
                wrapper),
            "each published context VMM binding must have exactly one "
            "pointer-table name");
      } else {
        IREE_ASSERT(has_failed_plan && wrapper->has_reserved_insert,
                    "an unpublished VMM binding must belong to the retained "
                    "fail-closed plan");
      }
      IREE_ASSERT(iree_hip_vmm_count_live_binding_names(wrapper) == 1,
                  "each context VMM wrapper must have exactly one live "
                  "reservation binding");
      iree_hal_streaming_buffer_release(wrapper);
    }
    iree_hip_vmm_reservation_release(reservation);
  }

  // Check wrapper/table -> binding. Never hold the table lock while scanning
  // reservations: retain the exact table entry, unlock, and then count names.
  for (size_t i = 0;; ++i) {
    iree_hal_streaming_buffer_t* wrapper = NULL;
    iree_slim_mutex_lock(&context->buffer_table.mutex);
    if (i < context->buffer_table.count) {
      iree_hal_streaming_buffer_t* candidate =
          (iree_hal_streaming_buffer_t*)context->buffer_table.entries[i]
              .user_data;
      IREE_ASSERT(candidate && candidate->context == context);
      if (candidate->is_virtual_memory_access && candidate->is_published) {
        wrapper = candidate;
        iree_hal_streaming_buffer_retain(wrapper);
      }
    }
    const bool at_end = i >= context->buffer_table.count;
    iree_slim_mutex_unlock(&context->buffer_table.mutex);
    if (at_end) break;
    if (!wrapper) continue;

    IREE_ASSERT(iree_hip_vmm_count_live_binding_names(wrapper) == 1,
                "each preserved VMM wrapper must have exactly one live "
                "reservation binding");
    iree_hal_streaming_buffer_release(wrapper);
  }
}

hipError_t iree_hip_vmm_query_pointer_under_admission(
    const void* ptr, iree_hip_vmm_pointer_snapshot_t* out_snapshot) {
  if (!ptr || !out_snapshot) return hipErrorInvalidValue;
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  IREE_ASSERT(iree_hip_vmm_reader_depth > 0);
  hipError_t result = hipSuccess;

  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) {
    return hipErrorInvalidValue;
  }

  const size_t reservation_offset = (uintptr_t)ptr - reservation->base_address;
  iree_slim_mutex_lock(&reservation->mutex);
  const size_t mapping_position =
      iree_hip_vmm_mapping_containing(reservation, reservation_offset);
  if (reservation->retiring || reservation->poisoned ||
      mapping_position == SIZE_MAX) {
    result = hipErrorInvalidValue;
  } else {
    const iree_hip_vmm_mapping_t* mapping =
        &reservation->mappings[mapping_position];
    out_snapshot->mapped_base =
        reservation->base_address + mapping->virtual_offset;
    out_snapshot->mapped_size = mapping->size;
    out_snapshot->offset = reservation_offset - mapping->virtual_offset;
    out_snapshot->physical_device_ordinal = mapping->allocation->device_ordinal;
    out_snapshot->physical_owner_context =
        (hipCtx_t)mapping->allocation->owner_context;
    out_snapshot->buffer_id = mapping->allocation->pointer_attribute_buffer_id;
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

hipError_t iree_hip_vmm_query_pointer(
    const void* ptr, iree_hip_vmm_pointer_snapshot_t* out_snapshot) {
  hipError_t result = iree_hip_vmm_reader_begin();
  if (result != hipSuccess) return result;
  result = iree_hip_vmm_query_pointer_under_admission(ptr, out_snapshot);
  iree_hip_vmm_reader_end();
  return result;
}

hipError_t iree_hip_vmm_validate_kernel_pointer(
    iree_hal_streaming_context_t* context, uint64_t device_pointer) {
  if (!context || device_pointer == 0) return hipErrorInvalidValue;

  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)device_pointer,
                                      /*require_base=*/false);
  // A pointer outside every live VMM reservation is an ordinary external
  // device pointer and keeps the existing raw-argument behavior.
  if (!reservation) return hipSuccess;
  iree_hip_vmm_reservation_release(reservation);

  // Known VMM pointers must resolve through an operational alias in the exact
  // launch context. The resolver lazily realizes device-scoped grants for
  // explicit contexts and rejects holes, ProtNone ranges, retired epochs, and
  // revoked/replaced capabilities. The surrounding launch reader keeps that
  // capability live through queue acceptance.
  iree_hal_streaming_buffer_ref_t ref;
  uint64_t capability_id = 0;
  iree_status_t status = iree_hal_streaming_memory_lookup_range_with_access(
      context, device_pointer, 1, IREE_HAL_MEMORY_ACCESS_READ, &ref,
      &capability_id);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorInvalidValue;
  }
  return capability_id != 0 ? hipSuccess : hipErrorInvalidValue;
}

static hipError_t iree_hip_vmm_call_begin(void) {
  return iree_hip_vmm_reader_begin();
}

#define IREE_HIP_VMM_GUARDED_CALL(expression)            \
  do {                                                   \
    hipError_t begin_result = iree_hip_vmm_call_begin(); \
    if (begin_result != hipSuccess) return begin_result; \
    hipError_t result = (expression);                    \
    iree_hip_vmm_reader_end();                           \
    return result;                                       \
  } while (0)

#define IREE_HIP_VMM_WRITER_GUARDED_CALL(expression)        \
  do {                                                      \
    hipError_t begin_result =                               \
        iree_hip_vmm_writer_begin(/*require_active=*/true); \
    if (begin_result != hipSuccess) return begin_result;    \
    hipError_t result = (expression);                       \
    iree_hip_vmm_writer_end();                              \
    return result;                                          \
  } while (0)

hipError_t iree_hip_vmm_is_supported(int device_ordinal, bool* out_supported) {
  IREE_HIP_VMM_GUARDED_CALL(
      iree_hip_vmm_is_supported_impl(device_ordinal, out_supported));
}

hipError_t iree_hip_vmm_address_reserve(void** ptr, size_t size,
                                        size_t alignment, void* address,
                                        unsigned long long flags) {
  IREE_HIP_VMM_GUARDED_CALL(
      iree_hip_vmm_address_reserve_impl(ptr, size, alignment, address, flags));
}

hipError_t iree_hip_vmm_address_free(void* device_ptr, size_t size) {
  IREE_HIP_VMM_WRITER_GUARDED_CALL(
      iree_hip_vmm_address_free_impl(device_ptr, size));
}

hipError_t iree_hip_vmm_create(hipMemGenericAllocationHandle_t* handle,
                               size_t size,
                               const hipMemAllocationProp* properties,
                               unsigned long long flags) {
  IREE_HIP_VMM_GUARDED_CALL(
      iree_hip_vmm_create_impl(handle, size, properties, flags));
}

hipError_t iree_hip_vmm_release(hipMemGenericAllocationHandle_t handle) {
  IREE_HIP_VMM_GUARDED_CALL(iree_hip_vmm_release_impl(handle));
}

hipError_t iree_hip_vmm_map(void* ptr, size_t size, size_t offset,
                            hipMemGenericAllocationHandle_t handle,
                            unsigned long long flags) {
  IREE_HIP_VMM_GUARDED_CALL(
      iree_hip_vmm_map_impl(ptr, size, offset, handle, flags));
}

hipError_t iree_hip_vmm_unmap(void* ptr, size_t size) {
  IREE_HIP_VMM_WRITER_GUARDED_CALL(iree_hip_vmm_unmap_impl(ptr, size));
}

hipError_t iree_hip_vmm_set_access(void* ptr, size_t size,
                                   const hipMemAccessDesc* descriptors,
                                   size_t count) {
  IREE_HIP_VMM_WRITER_GUARDED_CALL(
      iree_hip_vmm_set_access_impl(ptr, size, descriptors, count));
}

hipError_t iree_hip_vmm_get_access(unsigned long long* flags,
                                   const hipMemLocation* location, void* ptr) {
  IREE_HIP_VMM_GUARDED_CALL(iree_hip_vmm_get_access_impl(flags, location, ptr));
}

hipError_t iree_hip_vmm_get_allocation_granularity(
    size_t* granularity, const hipMemAllocationProp* properties,
    hipMemAllocationGranularity_flags option) {
  IREE_HIP_VMM_GUARDED_CALL(iree_hip_vmm_get_allocation_granularity_impl(
      granularity, properties, option));
}

hipError_t iree_hip_vmm_get_allocation_properties(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle) {
  IREE_HIP_VMM_GUARDED_CALL(
      iree_hip_vmm_get_allocation_properties_impl(properties, handle));
}

hipError_t iree_hip_vmm_retain_allocation_handle(
    hipMemGenericAllocationHandle_t* handle, void* address) {
  IREE_HIP_VMM_GUARDED_CALL(
      iree_hip_vmm_retain_allocation_handle_impl(handle, address));
}

#undef IREE_HIP_VMM_GUARDED_CALL
#undef IREE_HIP_VMM_WRITER_GUARDED_CALL
