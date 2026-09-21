// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/virtual_memory.h"

#include <stdint.h>

#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

// ROCr is the authoritative owner of reservation, mapping, access, and handle
// state. Shared native-domain and reservation records keep every pointer used
// by delayed VMM cleanup independent of allocator/device inline storage.

typedef struct iree_hal_amdgpu_virtual_memory_domain_t
    iree_hal_amdgpu_virtual_memory_domain_t;

typedef enum iree_hal_amdgpu_vmm_quarantine_kind_e {
  IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION = 0,
  IREE_HAL_AMDGPU_VMM_QUARANTINE_PHYSICAL = 1,
} iree_hal_amdgpu_vmm_quarantine_kind_t;

// Intrusive cleanup retry node embedded in the exact native owner record. No
// allocation is required after a native cleanup failure.
typedef struct iree_hal_amdgpu_vmm_quarantine_entry_t {
  struct iree_hal_amdgpu_vmm_quarantine_entry_t* next;
  iree_hal_amdgpu_vmm_quarantine_kind_t kind;
  bool enqueued;
  hsa_status_t last_status;
  uint64_t attempt_count;
} iree_hal_amdgpu_vmm_quarantine_entry_t;

typedef struct iree_hal_amdgpu_virtual_memory_reservation_t {
  // Must remain first so a detached quarantine entry can recover its owner
  // without allocating a separate list node.
  iree_hal_amdgpu_vmm_quarantine_entry_t quarantine;
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdgpu_virtual_memory_domain_t* domain;
  iree_allocator_t host_allocator;
  // Transferred HAL buffer ownership for a failed explicit rollback. NULL for
  // a reserve-time failure before a buffer wrapper was created.
  iree_hal_buffer_t* quarantined_buffer;
  void* base_ptr;
  iree_device_size_t size;
  iree_hal_queue_family_affinity_t queue_family_affinity;
  iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells;
  // Exact HSA identities represented by |queue_family_affinity| in the
  // reservation owner's topology. Queue-family ordinals are local to a
  // logical device and therefore cannot be applied to a foreign physical
  // allocation's topology.
  uint32_t source_agent_count;
  hsa_agent_t source_agents[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
} iree_hal_amdgpu_virtual_memory_reservation_t;

typedef struct iree_hal_amdgpu_virtual_memory_range_t {
  void* base_ptr;
  iree_device_size_t size;
  iree_hal_queue_family_affinity_t queue_family_affinity;
  iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells;
  uint32_t source_agent_count;
  const hsa_agent_t* source_agents;
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation;
} iree_hal_amdgpu_virtual_memory_range_t;

struct iree_hal_amdgpu_virtual_memory_domain_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_allocator_t host_allocator;
  // Protects only quarantine list ownership. Native HSA calls are never made
  // while this mutex is held.
  iree_slim_mutex_t quarantine_mutex;
  iree_hal_amdgpu_vmm_quarantine_entry_t* quarantine_head;
  iree_hal_amdgpu_vmm_quarantine_entry_t* quarantine_tail;
  size_t quarantine_count;
  bool quarantine_drain_active;
  iree_hal_amdgpu_virtual_memory_domain_t* next;
};

struct iree_hal_physical_memory_t {
  // Must remain first; see the reservation quarantine node above.
  iree_hal_amdgpu_vmm_quarantine_entry_t quarantine;
  iree_allocator_t host_allocator;
  iree_hal_amdgpu_virtual_memory_state_t* creator_state;
  iree_hal_amdgpu_virtual_memory_domain_t* domain;
  // Exact HAL device identity that selected the physical allocation pool.
  iree_hal_device_t* creator_device;
  // Exact immutable topology used to resolve the creator device's HSA agents.
  const iree_hal_amdgpu_topology_t* creator_topology;
  // HSA pool that owns the native physical allocation.
  hsa_amd_memory_pool_t memory_pool;
  // Flags supplied to hsa_amd_vmem_handle_create for this pool allocation.
  uint32_t allocation_flags;
  hsa_amd_vmem_alloc_handle_t allocation_handle;
  iree_device_size_t allocation_size;
  iree_device_size_t minimum_granule;
  iree_hal_memory_type_t memory_type;
  iree_hal_amdgpu_atomic_memory_source_masks_t atomic_memory_source_masks;
};

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
static iree_atomic_int32_t
    iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t
    iree_hal_amdgpu_vmm_test_fail_reservation_cleanup_count =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t
    iree_hal_amdgpu_vmm_test_fail_physical_cleanup_count =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int64_t
    iree_hal_amdgpu_vmm_test_reservation_cleanup_attempt_count =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int64_t
    iree_hal_amdgpu_vmm_test_physical_cleanup_attempt_count =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int64_t iree_hal_amdgpu_vmm_test_quarantine_owner_count =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t
    iree_hal_amdgpu_vmm_test_last_reservation_cleanup_status =
        IREE_ATOMIC_VAR_INIT(HSA_STATUS_SUCCESS);
static iree_atomic_int32_t
    iree_hal_amdgpu_vmm_test_last_physical_cleanup_status =
        IREE_ATOMIC_VAR_INIT(HSA_STATUS_SUCCESS);
static iree_once_flag iree_hal_amdgpu_vmm_test_drain_pause_once =
    IREE_ONCE_FLAG_INIT;
static iree_notification_t iree_hal_amdgpu_vmm_test_drain_pause_notification;
static iree_atomic_int32_t iree_hal_amdgpu_vmm_test_drain_pause_armed =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hal_amdgpu_vmm_test_drain_pause_entered =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hal_amdgpu_vmm_test_drain_pause_released =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hal_amdgpu_vmm_test_drain_attempt_count =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_int32_t iree_hal_amdgpu_vmm_test_drain_attempt_kinds[64];

static void iree_hal_amdgpu_vmm_test_drain_pause_initialize(void) {
  iree_notification_initialize(
      &iree_hal_amdgpu_vmm_test_drain_pause_notification);
}

static bool iree_hal_amdgpu_vmm_test_drain_pause_has_entered(void* user_data) {
  (void)user_data;
  return iree_atomic_load(&iree_hal_amdgpu_vmm_test_drain_pause_entered,
                          iree_memory_order_acquire) != 0;
}

static bool iree_hal_amdgpu_vmm_test_drain_pause_is_released(void* user_data) {
  (void)user_data;
  return iree_atomic_load(&iree_hal_amdgpu_vmm_test_drain_pause_released,
                          iree_memory_order_acquire) != 0;
}

static bool iree_hal_amdgpu_vmm_test_take_counted_failure(
    iree_atomic_int32_t* failure_count) {
  int32_t remaining =
      iree_atomic_load(failure_count, iree_memory_order_acquire);
  while (remaining > 0) {
    int32_t expected = remaining;
    if (iree_atomic_compare_exchange_strong(
            failure_count, &expected, remaining - 1, iree_memory_order_acq_rel,
            iree_memory_order_acquire)) {
      return true;
    }
    remaining = expected;
  }
  return false;
}
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

struct iree_hal_amdgpu_virtual_memory_state_t {
  iree_hal_amdgpu_virtual_memory_domain_t* domain;
  const iree_hal_amdgpu_topology_t* topology;
  iree_hal_device_t* device;
  iree_hal_allocator_statistics_t* statistics;
  iree_allocator_t host_allocator;
};

typedef enum iree_hal_amdgpu_vmm_native_operation_kind_e {
  IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_ACCESS = 0,
  IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_UNMAP = 1,
  IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_RELEASE_RESERVATION = 2,
  IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_FREE_PHYSICAL = 3,
} iree_hal_amdgpu_vmm_native_operation_kind_t;

struct iree_hal_amdgpu_vmm_native_operation_t {
  // Host allocator owning this prepared operation.
  iree_allocator_t host_allocator;
  // Retained native domain used by the raw apply call.
  iree_hal_amdgpu_virtual_memory_domain_t* domain;
  // Exact native mutation represented by this entry.
  iree_hal_amdgpu_vmm_native_operation_kind_t kind;
  // True after the raw mutation completed successfully. Prepared operations
  // have exactly one external executor; this field is not atomic and merely
  // makes retry after an observed successful ordinal idempotent.
  bool completed;
  // Exact native address for access, unmap, or reservation release.
  void* address;
  // Exact byte length consumed by the raw mutation.
  size_t size;
  // Reservation record made inert by successful address release.
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation;
  // Physical record made inert by successful handle release.
  iree_hal_physical_memory_t* physical_memory;
  // Number of initialized trailing access descriptors.
  iree_host_size_t access_descriptor_count;
  // Access descriptors copied during retryable preparation.
  hsa_amd_memory_access_desc_t access_descriptors[];
};

typedef struct iree_hal_amdgpu_virtual_memory_domain_registry_t {
  iree_slim_mutex_t mutex;
  iree_hal_amdgpu_virtual_memory_domain_t* head;
} iree_hal_amdgpu_virtual_memory_domain_registry_t;

static iree_once_flag iree_hal_amdgpu_virtual_memory_domain_registry_once =
    IREE_ONCE_FLAG_INIT;
static iree_hal_amdgpu_virtual_memory_domain_registry_t
    iree_hal_amdgpu_virtual_memory_domain_registry;

static void iree_hal_amdgpu_virtual_memory_domain_registry_initialize(void) {
  iree_slim_mutex_initialize(
      &iree_hal_amdgpu_virtual_memory_domain_registry.mutex);
}

static bool iree_hal_amdgpu_virtual_memory_domain_matches(
    const iree_hal_amdgpu_virtual_memory_domain_t* domain,
    const iree_hal_amdgpu_libhsa_t* libhsa) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)domain;
  (void)libhsa;
  return true;
#else
  // Independently loaded wrappers around the same ROCr implementation have
  // distinct dynamic-library wrapper addresses but resolve the same HSA entry
  // points. Symbol identity therefore defines the shared native domain.
  return domain->libhsa.hsa_init == libhsa->hsa_init;
#endif
}

static iree_status_t iree_hal_amdgpu_virtual_memory_domain_acquire(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_memory_domain_t** out_domain) {
  *out_domain = NULL;
  iree_call_once(&iree_hal_amdgpu_virtual_memory_domain_registry_once,
                 iree_hal_amdgpu_virtual_memory_domain_registry_initialize);
  iree_hal_amdgpu_virtual_memory_domain_registry_t* registry =
      &iree_hal_amdgpu_virtual_memory_domain_registry;
  iree_slim_mutex_lock(&registry->mutex);
  for (iree_hal_amdgpu_virtual_memory_domain_t* domain = registry->head; domain;
       domain = domain->next) {
    if (iree_hal_amdgpu_virtual_memory_domain_matches(domain, libhsa)) {
      iree_atomic_ref_count_inc(&domain->ref_count);
      *out_domain = domain;
      iree_slim_mutex_unlock(&registry->mutex);
      return iree_ok_status();
    }
  }

  iree_hal_amdgpu_virtual_memory_domain_t* domain = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*domain), (void**)&domain);
  if (iree_status_is_ok(status)) {
    memset(domain, 0, sizeof(*domain));
    iree_atomic_ref_count_init(&domain->ref_count);
    domain->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&domain->quarantine_mutex);
    status = iree_hal_amdgpu_libhsa_copy(libhsa, &domain->libhsa);
  }
  if (iree_status_is_ok(status)) {
    domain->next = registry->head;
    registry->head = domain;
    *out_domain = domain;
  } else {
    if (domain) iree_slim_mutex_deinitialize(&domain->quarantine_mutex);
    iree_allocator_free(host_allocator, domain);
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return status;
}

static void iree_hal_amdgpu_virtual_memory_domain_retain(
    iree_hal_amdgpu_virtual_memory_domain_t* domain) {
  iree_atomic_ref_count_inc(&domain->ref_count);
}

static void iree_hal_amdgpu_virtual_memory_domain_release(
    iree_hal_amdgpu_virtual_memory_domain_t* domain) {
  if (!domain) return;
  iree_hal_amdgpu_virtual_memory_domain_registry_t* registry =
      &iree_hal_amdgpu_virtual_memory_domain_registry;
  iree_slim_mutex_lock(&registry->mutex);
  if (iree_atomic_ref_count_dec(&domain->ref_count) != 1) {
    iree_slim_mutex_unlock(&registry->mutex);
    return;
  }
  iree_hal_amdgpu_virtual_memory_domain_t** link = &registry->head;
  while (*link && *link != domain) link = &(*link)->next;
  IREE_ASSERT(*link == domain);
  *link = domain->next;
  iree_slim_mutex_unlock(&registry->mutex);

  IREE_ASSERT(domain->quarantine_head == NULL);
  IREE_ASSERT(domain->quarantine_tail == NULL);
  IREE_ASSERT(domain->quarantine_count == 0);
  IREE_ASSERT(!domain->quarantine_drain_active);
  const iree_allocator_t host_allocator = domain->host_allocator;
  iree_slim_mutex_deinitialize(&domain->quarantine_mutex);
  iree_hal_amdgpu_libhsa_deinitialize(&domain->libhsa);
  iree_allocator_free(host_allocator, domain);
}

static void iree_hal_amdgpu_virtual_memory_reservation_release_record(
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation) {
  if (iree_atomic_ref_count_dec(&reservation->ref_count) != 1) return;
  const iree_allocator_t host_allocator = reservation->host_allocator;
  iree_hal_amdgpu_virtual_memory_domain_release(reservation->domain);
  iree_allocator_free(host_allocator, reservation);
}

static bool iree_hal_amdgpu_virtual_memory_is_aligned(
    iree_device_size_t value, iree_device_size_t alignment) {
  return alignment != 0 && value % alignment == 0;
}

static bool iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
    iree_device_size_t total_size, iree_device_size_t offset,
    iree_device_size_t size) {
  return offset <= total_size && size <= total_size - offset;
}

static bool iree_hal_amdgpu_virtual_memory_range_is_valid(
    iree_device_size_t total_size, iree_device_size_t offset,
    iree_device_size_t size, iree_device_size_t alignment) {
  return size != 0 &&
         iree_hal_amdgpu_virtual_memory_is_aligned(offset, alignment) &&
         iree_hal_amdgpu_virtual_memory_is_aligned(size, alignment) &&
         iree_hal_amdgpu_virtual_memory_range_is_in_bounds(total_size, offset,
                                                           size);
}

static iree_status_t iree_hal_amdgpu_virtual_memory_validate_placement(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement) {
  if (IREE_UNLIKELY(!placement.memory_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no HSA memory pool");
  }
  if (IREE_UNLIKELY(placement.device != state->device)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement belongs to another device");
  }
  if (IREE_UNLIKELY(
          placement.minimum_granule == 0 ||
          !iree_device_size_is_power_of_two(placement.minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU VMM placement has invalid minimum granule %" PRIu64,
        (uint64_t)placement.minimum_granule);
  }
  if (IREE_UNLIKELY(
          placement.recommended_granule < placement.minimum_granule ||
          !iree_device_size_is_power_of_two(placement.recommended_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU VMM placement has invalid recommended granule %" PRIu64,
        (uint64_t)placement.recommended_granule);
  }
  if (IREE_UNLIKELY(placement.max_allocation_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no allocation limit");
  }
  return iree_ok_status();
}

static hsa_access_permission_t
iree_hal_amdgpu_virtual_memory_translate_protection(
    iree_hal_memory_protection_t protection, bool* out_is_valid) {
  *out_is_valid = true;
  switch (protection) {
    case IREE_HAL_MEMORY_PROTECTION_NONE:
      return HSA_ACCESS_PERMISSION_NONE;
    case IREE_HAL_MEMORY_PROTECTION_READ:
      return HSA_ACCESS_PERMISSION_RO;
    case IREE_HAL_MEMORY_PROTECTION_WRITE:
      return HSA_ACCESS_PERMISSION_WO;
    case IREE_HAL_MEMORY_PROTECTION_READ_WRITE:
      return HSA_ACCESS_PERMISSION_RW;
    default:
      *out_is_valid = false;
      return HSA_ACCESS_PERMISSION_NONE;
  }
}

static void iree_hal_amdgpu_virtual_memory_release_reservation(
    void* user_data, iree_hal_buffer_t* virtual_buffer) {
  (void)virtual_buffer;
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      (iree_hal_amdgpu_virtual_memory_reservation_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (reservation->base_ptr) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_address_free_raw(&reservation->domain->libhsa,
                                           reservation->base_ptr,
                                           (size_t)reservation->size));
  }
  iree_hal_amdgpu_virtual_memory_reservation_release_record(reservation);
  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_buffer_release_callback_t
iree_hal_amdgpu_virtual_memory_reservation_release_callback(
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation) {
  return (iree_hal_buffer_release_callback_t){
      .fn = iree_hal_amdgpu_virtual_memory_release_reservation,
      .user_data = reservation,
  };
}

static hsa_status_t iree_hal_amdgpu_vmm_cleanup_reservation_raw(
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_atomic_fetch_add(
      &iree_hal_amdgpu_vmm_test_reservation_cleanup_attempt_count, 1,
      iree_memory_order_acq_rel);
  if (iree_hal_amdgpu_vmm_test_take_counted_failure(
          &iree_hal_amdgpu_vmm_test_fail_reservation_cleanup_count)) {
    iree_atomic_store(&iree_hal_amdgpu_vmm_test_last_reservation_cleanup_status,
                      HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                      iree_memory_order_release);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  const hsa_status_t status = iree_hsa_amd_vmem_address_free_raw(
      &reservation->domain->libhsa, reservation->base_ptr,
      (size_t)reservation->size);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_last_reservation_cleanup_status,
                    status, iree_memory_order_release);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  return status;
}

static hsa_status_t iree_hal_amdgpu_vmm_cleanup_physical_raw(
    iree_hal_physical_memory_t* physical_memory) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_atomic_fetch_add(
      &iree_hal_amdgpu_vmm_test_physical_cleanup_attempt_count, 1,
      iree_memory_order_acq_rel);
  if (iree_hal_amdgpu_vmm_test_take_counted_failure(
          &iree_hal_amdgpu_vmm_test_fail_physical_cleanup_count)) {
    iree_atomic_store(&iree_hal_amdgpu_vmm_test_last_physical_cleanup_status,
                      HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                      iree_memory_order_release);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  const hsa_status_t status = iree_hsa_amd_vmem_handle_release_raw(
      &physical_memory->domain->libhsa, physical_memory->allocation_handle);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_last_physical_cleanup_status,
                    status, iree_memory_order_release);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  return status;
}

static void iree_hal_amdgpu_vmm_quarantine_enqueue(
    iree_hal_amdgpu_virtual_memory_domain_t* domain,
    iree_hal_amdgpu_vmm_quarantine_entry_t* entry, hsa_status_t status) {
  IREE_ASSERT(!entry->enqueued);
  IREE_ASSERT(entry->next == NULL);
  entry->last_status = status;
  entry->attempt_count = 1;
  entry->enqueued = true;
  iree_slim_mutex_lock(&domain->quarantine_mutex);
  if (domain->quarantine_tail) {
    domain->quarantine_tail->next = entry;
  } else {
    domain->quarantine_head = entry;
  }
  domain->quarantine_tail = entry;
  ++domain->quarantine_count;
  iree_slim_mutex_unlock(&domain->quarantine_mutex);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_atomic_fetch_add(&iree_hal_amdgpu_vmm_test_quarantine_owner_count, 1,
                        iree_memory_order_acq_rel);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
}

static iree_status_t iree_hal_amdgpu_vmm_cleanup_status(
    iree_hal_amdgpu_vmm_quarantine_kind_t kind, hsa_status_t status) {
  if (kind == IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION &&
      status == HSA_STATUS_ERROR_RESOURCE_FREE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU virtual-memory reservation still has mapped ranges");
  }
  return iree_status_from_hsa_status(
      __FILE__, __LINE__, status,
      kind == IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION
          ? "hsa_amd_vmem_address_free"
          : "hsa_amd_vmem_handle_release",
      NULL);
}

static iree_status_t iree_hal_amdgpu_virtual_memory_free_address(
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation) {
  return iree_hal_amdgpu_vmm_cleanup_status(
      IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION,
      iree_hal_amdgpu_vmm_cleanup_reservation_raw(reservation));
}

static iree_status_t iree_hal_amdgpu_virtual_memory_resolve_range(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer,
    iree_hal_amdgpu_virtual_memory_range_t* out_range) {
  *out_range = (iree_hal_amdgpu_virtual_memory_range_t){0};
  void* user_data = NULL;
  if (!iree_hal_amdgpu_buffer_query_release_callback(
          virtual_buffer, iree_hal_amdgpu_virtual_memory_release_reservation,
          &user_data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer is not an AMDGPU virtual reservation");
  }
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      (iree_hal_amdgpu_virtual_memory_reservation_t*)user_data;
  if (IREE_UNLIKELY(reservation->domain != state->domain)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "virtual reservation belongs to another native HSA domain");
  }
  if (IREE_UNLIKELY(reservation->base_ptr !=
                        iree_hal_amdgpu_buffer_device_pointer(virtual_buffer) ||
                    reservation->size !=
                        iree_hal_buffer_allocation_size(virtual_buffer))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "virtual reservation metadata is inconsistent");
  }

  *out_range = (iree_hal_amdgpu_virtual_memory_range_t){
      .base_ptr = reservation->base_ptr,
      .size = reservation->size,
      .queue_family_affinity = reservation->queue_family_affinity,
      .atomic_memory_cells = reservation->atomic_memory_cells,
      .source_agent_count = reservation->source_agent_count,
      .source_agents = reservation->source_agents,
      .reservation = reservation,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_virtual_memory_state_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_hal_device_t* device,
    iree_hal_allocator_statistics_t* statistics,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_memory_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  iree_hal_amdgpu_virtual_memory_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  iree_status_t status = iree_hal_amdgpu_virtual_memory_domain_acquire(
      libhsa, host_allocator, &state->domain);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, state);
    return status;
  }
  state->topology = topology;
  state->device = device;
  state->statistics = statistics;
  state->host_allocator = host_allocator;
  *out_state = state;
  return iree_ok_status();
}

void iree_hal_amdgpu_virtual_memory_state_destroy(
    iree_hal_amdgpu_virtual_memory_state_t* state) {
  if (!state) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&state->domain->quarantine_mutex);
  IREE_ASSERT(state->domain->quarantine_count == 0);
  IREE_ASSERT(!state->domain->quarantine_drain_active);
  iree_slim_mutex_unlock(&state->domain->quarantine_mutex);
  const iree_allocator_t host_allocator = state->host_allocator;
  iree_hal_amdgpu_virtual_memory_domain_release(state->domain);
  iree_allocator_free(host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_virtual_memory_reserve(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_device_size_t minimum_alignment,
    iree_device_size_t requested_address,
    iree_hal_buffer_t** out_virtual_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_virtual_buffer);
  *out_virtual_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_memory_validate_placement(state, placement));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          size, /*offset=*/0, size, placement.minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory reservation size %" PRIu64
        " must be a non-zero multiple of minimum VMM granule %" PRIu64,
        (uint64_t)size, (uint64_t)placement.minimum_granule);
  }
  if (IREE_UNLIKELY(minimum_alignment != 0 &&
                    (minimum_alignment < placement.minimum_granule ||
                     !iree_device_size_is_power_of_two(minimum_alignment)))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory reservation alignment %" PRIu64
        " must be zero or a power of two no smaller than minimum VMM "
        "granule %" PRIu64,
        (uint64_t)minimum_alignment, (uint64_t)placement.minimum_granule);
  }

  iree_hal_amdgpu_virtual_memory_reservation_t* reservation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator, sizeof(*reservation), (void**)&reservation));
  memset(reservation, 0, sizeof(*reservation));
  iree_atomic_ref_count_init(&reservation->ref_count);
  reservation->domain = state->domain;
  iree_hal_amdgpu_virtual_memory_domain_retain(reservation->domain);
  reservation->host_allocator = state->host_allocator;
  reservation->size = size;
  reservation->queue_family_affinity = placement.queue_family_affinity;
  reservation->atomic_memory_cells = placement.atomic_memory_cells;
  iree_hal_amdgpu_access_agent_list_t source_agents;
  iree_status_t status =
      iree_hal_amdgpu_access_agent_list_resolve_queue_family_agents(
          state->topology, placement.queue_family_affinity, &source_agents);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_virtual_memory_reservation_release_record(reservation);
    return status;
  }
  IREE_ASSERT(source_agents.count <= IREE_HAL_AMDGPU_MAX_GPU_AGENT);
  reservation->source_agent_count = source_agents.count;
  memcpy(reservation->source_agents, source_agents.values,
         source_agents.count * sizeof(source_agents.values[0]));

  const iree_device_size_t reservation_alignment =
      minimum_alignment ? minimum_alignment
      : iree_hal_amdgpu_virtual_memory_is_aligned(size,
                                                  placement.recommended_granule)
          ? placement.recommended_granule
          : placement.minimum_granule;
  void* base_ptr = NULL;
  status = iree_hsa_amd_vmem_address_reserve_align(
      IREE_LIBHSA(&state->domain->libhsa), &base_ptr, size, requested_address,
      reservation_alignment, /*flags=*/0);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (iree_status_is_ok(status) &&
      iree_atomic_exchange(
          &iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native, 0,
          iree_memory_order_acq_rel) != 0) {
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "injected AMDGPU VMM failure after native reservation");
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  if (iree_status_is_ok(status) &&
      (uintptr_t)base_ptr % reservation_alignment != 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "AMDGPU VMM reservation has insufficient alignment");
  }
  reservation->base_ptr = base_ptr;

  iree_hal_buffer_t* virtual_buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = placement.device,
        .queue_family_affinity = placement.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        &state->domain->libhsa, buffer_placement, placement.memory_type,
        IREE_HAL_MEMORY_ACCESS_ALL, placement.buffer_usage,
        placement.atomic_memory_cells, size, size, base_ptr,
        iree_hal_amdgpu_virtual_memory_reservation_release_callback(
            reservation),
        state->host_allocator, &virtual_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_virtual_buffer = virtual_buffer;
  } else {
    if (base_ptr) {
      const hsa_status_t cleanup_status =
          iree_hal_amdgpu_vmm_cleanup_reservation_raw(reservation);
      if (cleanup_status == HSA_STATUS_SUCCESS) {
        reservation->base_ptr = NULL;
      } else {
        iree_hal_amdgpu_vmm_quarantine_enqueue(
            reservation->domain, &reservation->quarantine, cleanup_status);
      }
      status = iree_status_join(
          status,
          iree_hal_amdgpu_vmm_cleanup_status(
              IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION, cleanup_status));
    }
    if (!reservation->quarantine.enqueued) {
      iree_hal_amdgpu_virtual_memory_reservation_release_record(reservation);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_release(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_memory_free_address(range.reservation));

  const iree_hal_buffer_release_callback_t release_callback =
      iree_hal_amdgpu_virtual_memory_reservation_release_callback(
          range.reservation);
  iree_hal_amdgpu_buffer_disarm_storage(virtual_buffer, release_callback);
  iree_hal_amdgpu_virtual_memory_reservation_release_record(range.reservation);
  iree_hal_buffer_release(virtual_buffer);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_virtual_memory_release_or_quarantine(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  const hsa_status_t cleanup_status =
      iree_hal_amdgpu_vmm_cleanup_reservation_raw(range.reservation);
  if (cleanup_status == HSA_STATUS_SUCCESS) {
    range.reservation->base_ptr = NULL;
    iree_hal_amdgpu_virtual_memory_dispose_consumed_reservation(virtual_buffer);
    return iree_ok_status();
  }

  // Transfer the caller's exact HAL buffer ownership into the embedded retry
  // node. Its release callback keeps the reservation and native domain alive.
  range.reservation->quarantined_buffer = virtual_buffer;
  iree_hal_amdgpu_vmm_quarantine_enqueue(range.reservation->domain,
                                         &range.reservation->quarantine,
                                         cleanup_status);
  return iree_hal_amdgpu_vmm_cleanup_status(
      IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION, cleanup_status);
}

typedef struct iree_hal_amdgpu_virtual_memory_alias_data_t {
  iree_allocator_t host_allocator;
  iree_hal_buffer_t* virtual_buffer;
} iree_hal_amdgpu_virtual_memory_alias_data_t;

static void iree_hal_amdgpu_virtual_memory_release_alias(
    void* user_data, iree_hal_buffer_t* alias_buffer) {
  (void)alias_buffer;
  iree_hal_amdgpu_virtual_memory_alias_data_t* alias_data =
      (iree_hal_amdgpu_virtual_memory_alias_data_t*)user_data;
  const iree_allocator_t host_allocator = alias_data->host_allocator;
  iree_hal_buffer_release(alias_data->virtual_buffer);
  iree_allocator_free(host_allocator, alias_data);
}

iree_status_t iree_hal_amdgpu_virtual_memory_alias(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_t** out_alias_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(out_alias_buffer);
  *out_alias_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_memory_validate_placement(state, placement));

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(size == 0 ||
                    !iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory alias exceeds reservation");
  }

  iree_hal_amdgpu_virtual_memory_alias_data_t* alias_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator, sizeof(*alias_data), (void**)&alias_data));
  alias_data->host_allocator = state->host_allocator;
  alias_data->virtual_buffer = virtual_buffer;
  iree_hal_buffer_retain(virtual_buffer);

  const iree_hal_buffer_placement_t buffer_placement = {
      .device = placement.device,
      .queue_family_affinity = placement.queue_family_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  void* alias_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  iree_status_t status = iree_hal_amdgpu_buffer_create(
      &state->domain->libhsa, buffer_placement, placement.memory_type,
      allowed_access, placement.buffer_usage, placement.atomic_memory_cells,
      size, size, alias_ptr,
      (iree_hal_buffer_release_callback_t){
          .fn = iree_hal_amdgpu_virtual_memory_release_alias,
          .user_data = alias_data,
      },
      state->host_allocator, out_alias_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(alias_data->virtual_buffer);
    iree_allocator_free(alias_data->host_allocator, alias_data);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_physical_memory_allocate(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_physical_memory);
  *out_physical_memory = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_memory_validate_placement(state, placement));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          size, /*offset=*/0, size, placement.minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical-memory allocation size %" PRIu64
        " must be a non-zero multiple of minimum VMM granule %" PRIu64,
        (uint64_t)size, (uint64_t)placement.minimum_granule);
  }
  if (IREE_UNLIKELY(size > placement.max_allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical-memory allocation size %" PRIu64
                            " exceeds pool allocation limit %" PRIu64,
                            (uint64_t)size,
                            (uint64_t)placement.max_allocation_size);
  }

  // HIP requires VMM backing allocations to remain resident and ROCr expresses
  // that contract with MEMORY_TYPE_PINNED for both host and device pools.
  hsa_amd_memory_type_t hsa_memory_type = MEMORY_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_translate_memory_type(
      IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST, &hsa_memory_type));

  iree_hal_physical_memory_t* physical_memory = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*physical_memory), (void**)&physical_memory));
  *physical_memory = (iree_hal_physical_memory_t){
      .quarantine.kind = IREE_HAL_AMDGPU_VMM_QUARANTINE_PHYSICAL,
      .host_allocator = host_allocator,
      .creator_state = state,
      .domain = state->domain,
      .creator_device = state->device,
      .creator_topology = state->topology,
      .memory_pool = placement.memory_pool,
      .allocation_flags = 0,
      .allocation_size = size,
      .minimum_granule = placement.minimum_granule,
      .memory_type = placement.memory_type,
  };
  iree_hal_amdgpu_virtual_memory_domain_retain(physical_memory->domain);
  if (placement.atomic_memory_source_masks) {
    physical_memory->atomic_memory_source_masks =
        *placement.atomic_memory_source_masks;
  }

  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(&physical_memory->domain->libhsa), placement.memory_pool,
      size, hsa_memory_type, /*flags=*/0, &physical_memory->allocation_handle);
  if (iree_status_is_ok(status)) {
    IREE_STATISTICS({
      if (state->statistics) {
        iree_hal_allocator_statistics_record_alloc(
            state->statistics, physical_memory->memory_type,
            physical_memory->allocation_size);
      }
    });
    *out_physical_memory = physical_memory;
  } else {
    iree_hal_amdgpu_virtual_memory_domain_release(physical_memory->domain);
    iree_allocator_free(host_allocator, physical_memory);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_physical_memory_free(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_physical_memory_t* physical_memory) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (physical_memory->creator_state != state) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "physical memory must be freed by its creating AMDGPU allocator");
  }

  // The HAL contract requires callers to unmap every alias before freeing.
  // ROCr retains mapped references internally, so no duplicate mapping ledger
  // is needed here to preserve native allocation lifetime.
  const hsa_status_t cleanup_status =
      iree_hal_amdgpu_vmm_cleanup_physical_raw(physical_memory);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_cleanup_status(
      IREE_HAL_AMDGPU_VMM_QUARANTINE_PHYSICAL, cleanup_status));
  physical_memory->allocation_handle = (hsa_amd_vmem_alloc_handle_t){0};
  IREE_STATISTICS({
    if (state->statistics) {
      iree_hal_allocator_statistics_record_free(
          state->statistics, physical_memory->memory_type,
          physical_memory->allocation_size);
    }
  });
  const iree_allocator_t host_allocator = physical_memory->host_allocator;
  iree_hal_amdgpu_virtual_memory_domain_release(physical_memory->domain);
  iree_allocator_free(host_allocator, physical_memory);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_physical_memory_free_or_quarantine(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_physical_memory_t* physical_memory) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (physical_memory->creator_state != state) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "physical memory must be freed by its creating AMDGPU allocator");
  }
  const hsa_status_t cleanup_status =
      iree_hal_amdgpu_vmm_cleanup_physical_raw(physical_memory);
  if (cleanup_status == HSA_STATUS_SUCCESS) {
    physical_memory->allocation_handle = (hsa_amd_vmem_alloc_handle_t){0};
    iree_hal_amdgpu_physical_memory_dispose_consumed(state, physical_memory);
    return iree_ok_status();
  }
  iree_hal_amdgpu_vmm_quarantine_enqueue(
      physical_memory->domain, &physical_memory->quarantine, cleanup_status);
  return iree_hal_amdgpu_vmm_cleanup_status(
      IREE_HAL_AMDGPU_VMM_QUARANTINE_PHYSICAL, cleanup_status);
}

iree_status_t iree_hal_amdgpu_virtual_memory_map(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (physical_memory->creator_state != state) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "physical memory must be mapped by its creating AMDGPU allocator");
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          physical_memory->allocation_size, physical_offset, size,
          physical_memory->minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical-memory mapping range is not a valid "
        "minimum-granule subrange");
  }

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          range.size, virtual_offset, size,
          physical_memory->minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory map range is not a valid minimum-granule "
        "physical mapping");
  }

  // Query the physical pool using the reservation owner's exact HSA agent
  // identities. Queue-family bits are ordinal within one logical-device
  // topology, so applying |range.queue_family_affinity| to the physical
  // creator's precomputed masks could silently select different GPUs.
  iree_hal_amdgpu_topology_t source_topology;
  memset(&source_topology, 0, sizeof(source_topology));
  source_topology.gpu_agent_count = range.source_agent_count;
  memcpy(source_topology.gpu_agents, range.source_agents,
         range.source_agent_count * sizeof(range.source_agents[0]));
  iree_hal_amdgpu_atomic_memory_source_masks_t source_masks;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_atomic_memory_query_source_masks(
      &physical_memory->domain->libhsa, &source_topology,
      physical_memory->memory_pool, physical_memory->allocation_flags,
      &source_masks));
  const iree_hal_amdgpu_gpu_agent_mask_t source_agent_mask =
      range.source_agent_count == IREE_HAL_AMDGPU_MAX_GPU_AGENT
          ? UINT64_MAX
          : (((iree_hal_amdgpu_gpu_agent_mask_t)1 << range.source_agent_count) -
             1);
  const iree_hal_amdgpu_atomic_memory_cell_flags_t reservation_cells =
      range.atomic_memory_cells;
  const iree_hal_amdgpu_atomic_memory_cell_flags_t physical_cells =
      iree_hal_amdgpu_atomic_memory_select_device_cells(&source_masks,
                                                        source_agent_mask);
  if (IREE_UNLIKELY(!iree_all_bits_set(physical_cells, reservation_cells))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical backing atomic cells 0x%08" PRIx32
        " do not satisfy virtual reservation cells 0x%08" PRIx32
        " for the reservation's exact HSA agents",
        physical_cells, reservation_cells);
  }

  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  if (IREE_UNLIKELY((uintptr_t)map_ptr % physical_memory->minimum_granule !=
                    0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory map address is not aligned to the physical "
        "allocation granule");
  }
  return iree_hsa_amd_vmem_map(IREE_LIBHSA(&physical_memory->domain->libhsa),
                               map_ptr, size, physical_offset,
                               physical_memory->allocation_handle, /*flags=*/0);
}

iree_status_t iree_hal_amdgpu_virtual_memory_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(size == 0 ||
                    !iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory unmap range exceeds the "
                            "reservation");
  }

  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  return iree_hsa_amd_vmem_unmap(
      IREE_LIBHSA(&range.reservation->domain->libhsa), map_ptr, size);
}

iree_status_t iree_hal_amdgpu_virtual_memory_protect(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  bool is_valid_protection = false;
  const hsa_access_permission_t permissions =
      iree_hal_amdgpu_virtual_memory_translate_protection(protection,
                                                          &is_valid_protection);
  if (!is_valid_protection) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU virtual-memory protection flags");
  }

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(size == 0 ||
                    !iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory protection range exceeds "
                            "the reservation");
  }

  if (iree_hal_queue_family_affinity_is_any(queue_family_affinity)) {
    queue_family_affinity = range.queue_family_affinity;
  } else if (IREE_UNLIKELY(iree_hal_queue_family_affinity_is_empty(
                               queue_family_affinity) ||
                           !iree_all_bits_set(range.queue_family_affinity,
                                              queue_family_affinity))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory protection family affinity 0x%016" PRIx64
        " exceeds reservation affinity 0x%016" PRIx64,
        queue_family_affinity, range.queue_family_affinity);
  }
  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      state->topology, queue_family_affinity, access_scope, &agent_list));

  hsa_amd_memory_access_desc_t access_descs[IREE_HAL_AMDGPU_MAX_CPU_AGENT +
                                            IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  for (uint32_t i = 0; i < agent_list.count; ++i) {
    access_descs[i] = (hsa_amd_memory_access_desc_t){
        .permissions = permissions,
        .agent_handle = agent_list.values[i],
    };
  }
  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  return iree_hsa_amd_vmem_set_access(
      IREE_LIBHSA(&range.reservation->domain->libhsa), map_ptr, size,
      access_descs, agent_list.count);
}

iree_status_t iree_hal_amdgpu_virtual_memory_protect_peer(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_amdgpu_virtual_memory_state_t* access_state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_memory_protection_t protection) {
  IREE_ASSERT_ARGUMENT(reservation_state);
  IREE_ASSERT_ARGUMENT(access_state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  bool is_valid_protection = false;
  const hsa_access_permission_t permissions =
      iree_hal_amdgpu_virtual_memory_translate_protection(protection,
                                                          &is_valid_protection);
  if (!is_valid_protection) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU virtual-memory protection flags");
  }

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      reservation_state, virtual_buffer, &range));
  if (IREE_UNLIKELY(access_state->domain != range.reservation->domain)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "peer access device belongs to another native HSA domain");
  }
  if (IREE_UNLIKELY(size == 0 ||
                    !iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory protection range exceeds "
                            "the reservation");
  }

  // ANY is scoped only within the explicitly supplied access logical device;
  // unlike the generic protection path it is never expanded through the
  // reservation owner's placement affinity.
  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      access_state->topology, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, &agent_list));
  if (IREE_UNLIKELY(agent_list.count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "peer access device has no GPU agents");
  }

  hsa_amd_memory_access_desc_t access_descs[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  for (uint32_t i = 0; i < agent_list.count; ++i) {
    access_descs[i] = (hsa_amd_memory_access_desc_t){
        .permissions = permissions,
        .agent_handle = agent_list.values[i],
    };
  }
  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  return iree_hsa_amd_vmem_set_access(
      IREE_LIBHSA(&range.reservation->domain->libhsa), map_ptr, size,
      access_descs, agent_list.count);
}

static bool iree_hal_amdgpu_vmm_native_status_is_success(hsa_status_t status) {
  return status == HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_vmm_device_size_to_host_size(
    iree_device_size_t value, size_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  const size_t converted_value = (size_t)value;
  if (IREE_UNLIKELY((iree_device_size_t)converted_value != value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native VMM range exceeds host address width");
  }
  *out_value = converted_value;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_vmm_native_operation_allocate(
    iree_hal_amdgpu_virtual_memory_domain_t* domain,
    iree_hal_amdgpu_vmm_native_operation_kind_t kind,
    iree_host_size_t access_descriptor_count, iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation) {
  *out_operation = NULL;
  iree_host_size_t descriptor_size = 0;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(access_descriptor_count,
                                  sizeof(hsa_amd_memory_access_desc_t),
                                  &descriptor_size) ||
      !iree_host_size_checked_add(
          sizeof(iree_hal_amdgpu_vmm_native_operation_t), descriptor_size,
          &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native VMM operation size overflow");
  }
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, allocation_size,
                                             (void**)&operation));
  memset(operation, 0, allocation_size);
  operation->host_allocator = host_allocator;
  operation->domain = domain;
  iree_hal_amdgpu_virtual_memory_domain_retain(domain);
  operation->kind = kind;
  operation->access_descriptor_count = access_descriptor_count;
  *out_operation = operation;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_vmm_validate_physical_access(
    iree_hal_amdgpu_virtual_memory_domain_t* reservation_domain,
    const iree_hal_amdgpu_access_agent_list_t* agent_list,
    iree_host_size_t physical_memory_count,
    iree_hal_physical_memory_t* const* physical_memories) {
  if (IREE_UNLIKELY(physical_memory_count == 0 || !physical_memories)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared VMM access requires mapped physical memory");
  }
  for (iree_host_size_t i = 0; i < physical_memory_count; ++i) {
    const iree_hal_physical_memory_t* physical_memory = physical_memories[i];
    if (IREE_UNLIKELY(!physical_memory ||
                      physical_memory->domain != reservation_domain ||
                      !physical_memory->creator_state ||
                      physical_memory->creator_device !=
                          physical_memory->creator_state->device ||
                      physical_memory->creator_topology !=
                          physical_memory->creator_state->topology ||
                      physical_memory->memory_pool.handle == 0 ||
                      physical_memory->allocation_handle.handle == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "physical memory has stale VMM owner or topology identity");
    }
    for (uint32_t j = 0; j < agent_list->count; ++j) {
      hsa_amd_memory_pool_access_t access =
          HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
      IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
          IREE_LIBHSA(&reservation_domain->libhsa), agent_list->values[j],
          physical_memory->memory_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
          &access));
      if (IREE_UNLIKELY(access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "VMM access agent cannot access a mapped physical pool");
      }
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_access(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_amdgpu_virtual_memory_state_t* access_state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection,
    iree_host_size_t physical_memory_count,
    iree_hal_physical_memory_t* const* physical_memories,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation) {
  IREE_ASSERT_ARGUMENT(reservation_state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(out_operation);
  *out_operation = NULL;

  bool is_valid_protection = false;
  const hsa_access_permission_t permissions =
      iree_hal_amdgpu_virtual_memory_translate_protection(protection,
                                                          &is_valid_protection);
  if (IREE_UNLIKELY(!is_valid_protection)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid prepared VMM protection flags");
  }

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      reservation_state, virtual_buffer, &range));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size) ||
                    size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared VMM access exceeds reservation");
  }

  iree_hal_amdgpu_virtual_memory_state_t* target_state =
      access_state ? access_state : reservation_state;
  if (IREE_UNLIKELY(target_state->domain != range.reservation->domain)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared VMM access target belongs to another native domain");
  }
  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      target_state->topology, queue_family_affinity, access_scope,
      &agent_list));
  if (IREE_UNLIKELY(agent_list.count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared VMM access has no target agents");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_validate_physical_access(
      range.reservation->domain, &agent_list, physical_memory_count,
      physical_memories));

  size_t host_virtual_offset = 0;
  size_t host_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_device_size_to_host_size(
      virtual_offset, &host_virtual_offset));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_vmm_device_size_to_host_size(size, &host_size));

  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_native_operation_allocate(
      range.reservation->domain, IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_ACCESS,
      agent_list.count, host_allocator, &operation));
  operation->address = (uint8_t*)range.base_ptr + host_virtual_offset;
  operation->size = host_size;
  for (uint32_t i = 0; i < agent_list.count; ++i) {
    operation->access_descriptors[i] = (hsa_amd_memory_access_desc_t){
        .permissions = permissions,
        .agent_handle = agent_list.values[i],
    };
  }
  *out_operation = operation;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation) {
  IREE_ASSERT_ARGUMENT(reservation_state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(out_operation);
  *out_operation = NULL;
  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      reservation_state, virtual_buffer, &range));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size) ||
                    size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared VMM unmap exceeds reservation");
  }
  size_t host_virtual_offset = 0;
  size_t host_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_device_size_to_host_size(
      virtual_offset, &host_virtual_offset));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_vmm_device_size_to_host_size(size, &host_size));
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_native_operation_allocate(
      range.reservation->domain, IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_UNMAP, 0,
      host_allocator, &operation));
  operation->address = (uint8_t*)range.base_ptr + host_virtual_offset;
  operation->size = host_size;
  *out_operation = operation;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_release_reservation(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_buffer_t* virtual_buffer, iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation) {
  IREE_ASSERT_ARGUMENT(reservation_state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(out_operation);
  *out_operation = NULL;
  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      reservation_state, virtual_buffer, &range));
  size_t host_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_vmm_device_size_to_host_size(range.size, &host_size));
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_native_operation_allocate(
      range.reservation->domain,
      IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_RELEASE_RESERVATION, 0,
      host_allocator, &operation));
  operation->address = range.base_ptr;
  operation->size = host_size;
  operation->reservation = range.reservation;
  *out_operation = operation;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_free_physical(
    iree_hal_amdgpu_virtual_memory_state_t* creator_state,
    iree_hal_physical_memory_t* physical_memory,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation) {
  IREE_ASSERT_ARGUMENT(creator_state);
  IREE_ASSERT_ARGUMENT(physical_memory);
  IREE_ASSERT_ARGUMENT(out_operation);
  *out_operation = NULL;
  if (IREE_UNLIKELY(physical_memory->creator_state != creator_state ||
                    physical_memory->creator_device != creator_state->device ||
                    physical_memory->creator_topology !=
                        creator_state->topology ||
                    physical_memory->domain != creator_state->domain ||
                    physical_memory->allocation_handle.handle == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "physical memory has stale VMM creator identity");
  }
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmm_native_operation_allocate(
      physical_memory->domain,
      IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_FREE_PHYSICAL, 0, host_allocator,
      &operation));
  operation->physical_memory = physical_memory;
  *out_operation = operation;
  return iree_ok_status();
}

hsa_status_t iree_hal_amdgpu_vmm_native_operation_apply(
    iree_hal_amdgpu_vmm_native_operation_t* operation) {
  IREE_ASSERT_ARGUMENT(operation);
  if (operation->completed) return HSA_STATUS_SUCCESS;

  hsa_status_t status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_ACCESS:
      status = iree_hsa_amd_vmem_set_access_raw(
          &operation->domain->libhsa, operation->address, operation->size,
          operation->access_descriptors,
          (size_t)operation->access_descriptor_count);
      break;
    case IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_UNMAP:
      status = iree_hsa_amd_vmem_unmap_raw(&operation->domain->libhsa,
                                           operation->address, operation->size);
      break;
    case IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_RELEASE_RESERVATION:
      status = iree_hsa_amd_vmem_address_free_raw(
          &operation->domain->libhsa, operation->address, operation->size);
      if (iree_hal_amdgpu_vmm_native_status_is_success(status)) {
        operation->reservation->base_ptr = NULL;
      }
      break;
    case IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_FREE_PHYSICAL:
      status = iree_hsa_amd_vmem_handle_release_raw(
          &operation->domain->libhsa,
          operation->physical_memory->allocation_handle);
      if (iree_hal_amdgpu_vmm_native_status_is_success(status)) {
        operation->physical_memory->allocation_handle =
            (hsa_amd_vmem_alloc_handle_t){0};
      }
      break;
  }
  if (iree_hal_amdgpu_vmm_native_status_is_success(status)) {
    operation->completed = true;
  }
  return status;
}

bool iree_hal_amdgpu_vmm_native_operation_may_invoke_callbacks(
    const iree_hal_amdgpu_vmm_native_operation_t* operation) {
  return operation &&
         operation->kind ==
             IREE_HAL_AMDGPU_VMM_NATIVE_OPERATION_RELEASE_RESERVATION;
}

void iree_hal_amdgpu_vmm_native_operation_destroy(
    iree_hal_amdgpu_vmm_native_operation_t* operation) {
  if (!operation) return;
  const iree_allocator_t host_allocator = operation->host_allocator;
  iree_hal_amdgpu_virtual_memory_domain_release(operation->domain);
  iree_allocator_free(host_allocator, operation);
}

void iree_hal_amdgpu_virtual_memory_dispose_consumed_reservation(
    iree_hal_buffer_t* virtual_buffer) {
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  void* user_data = NULL;
  const bool has_reservation = iree_hal_amdgpu_buffer_query_release_callback(
      virtual_buffer, iree_hal_amdgpu_virtual_memory_release_reservation,
      &user_data);
  IREE_ASSERT(has_reservation);
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      (iree_hal_amdgpu_virtual_memory_reservation_t*)user_data;
  IREE_ASSERT(reservation->base_ptr == NULL);
  const iree_hal_buffer_release_callback_t release_callback =
      iree_hal_amdgpu_virtual_memory_reservation_release_callback(reservation);
  iree_hal_amdgpu_buffer_disarm_storage(virtual_buffer, release_callback);
  iree_hal_amdgpu_virtual_memory_reservation_release_record(reservation);
  iree_hal_buffer_release(virtual_buffer);
}

void iree_hal_amdgpu_physical_memory_dispose_consumed(
    iree_hal_amdgpu_virtual_memory_state_t* creator_state,
    iree_hal_physical_memory_t* physical_memory) {
  IREE_ASSERT_ARGUMENT(creator_state);
  IREE_ASSERT_ARGUMENT(physical_memory);
  IREE_ASSERT(physical_memory->creator_state == creator_state);
  IREE_ASSERT(physical_memory->allocation_handle.handle == 0);
  IREE_STATISTICS({
    if (creator_state->statistics) {
      iree_hal_allocator_statistics_record_free(
          creator_state->statistics, physical_memory->memory_type,
          physical_memory->allocation_size);
    }
  });
  const iree_allocator_t host_allocator = physical_memory->host_allocator;
  iree_hal_amdgpu_virtual_memory_domain_release(physical_memory->domain);
  iree_allocator_free(host_allocator, physical_memory);
}

iree_status_t iree_hal_amdgpu_vmm_quarantine_drain(
    iree_hal_amdgpu_virtual_memory_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);
  iree_hal_amdgpu_virtual_memory_domain_t* domain = state->domain;
  iree_slim_mutex_lock(&domain->quarantine_mutex);
  if (domain->quarantine_drain_active) {
    iree_slim_mutex_unlock(&domain->quarantine_mutex);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU VMM quarantine drain is already active");
  }
  domain->quarantine_drain_active = true;
  iree_hal_amdgpu_vmm_quarantine_entry_t* entry = domain->quarantine_head;
  domain->quarantine_head = NULL;
  domain->quarantine_tail = NULL;
  iree_slim_mutex_unlock(&domain->quarantine_mutex);

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_call_once(&iree_hal_amdgpu_vmm_test_drain_pause_once,
                 iree_hal_amdgpu_vmm_test_drain_pause_initialize);
  if (iree_atomic_exchange(&iree_hal_amdgpu_vmm_test_drain_pause_armed, 0,
                           iree_memory_order_acq_rel) != 0) {
    iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_entered, 1,
                      iree_memory_order_release);
    iree_notification_post(&iree_hal_amdgpu_vmm_test_drain_pause_notification,
                           IREE_ALL_WAITERS);
    iree_notification_await(&iree_hal_amdgpu_vmm_test_drain_pause_notification,
                            iree_hal_amdgpu_vmm_test_drain_pause_is_released,
                            NULL, iree_infinite_timeout());
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  iree_hal_amdgpu_vmm_quarantine_entry_t* failed_head = NULL;
  iree_hal_amdgpu_vmm_quarantine_entry_t* failed_tail = NULL;
  size_t success_count = 0;
  hsa_status_t first_failure = HSA_STATUS_SUCCESS;
  iree_hal_amdgpu_vmm_quarantine_kind_t first_failure_kind =
      IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION;
  while (entry) {
    iree_hal_amdgpu_vmm_quarantine_entry_t* next = entry->next;
    entry->next = NULL;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
    const int32_t attempt_ordinal =
        iree_atomic_fetch_add(&iree_hal_amdgpu_vmm_test_drain_attempt_count, 1,
                              iree_memory_order_acq_rel);
    if (attempt_ordinal >= 0 &&
        (size_t)attempt_ordinal <
            IREE_ARRAYSIZE(iree_hal_amdgpu_vmm_test_drain_attempt_kinds)) {
      iree_atomic_store(
          &iree_hal_amdgpu_vmm_test_drain_attempt_kinds[attempt_ordinal],
          (int32_t)entry->kind + 1, iree_memory_order_release);
    }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
    hsa_status_t cleanup_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if (entry->kind == IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION) {
      cleanup_status = iree_hal_amdgpu_vmm_cleanup_reservation_raw(
          (iree_hal_amdgpu_virtual_memory_reservation_t*)entry);
    } else {
      cleanup_status = iree_hal_amdgpu_vmm_cleanup_physical_raw(
          (iree_hal_physical_memory_t*)entry);
    }
    entry->last_status = cleanup_status;
    ++entry->attempt_count;
    if (cleanup_status == HSA_STATUS_SUCCESS) {
      entry->enqueued = false;
      ++success_count;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      iree_atomic_fetch_sub(&iree_hal_amdgpu_vmm_test_quarantine_owner_count, 1,
                            iree_memory_order_acq_rel);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
      if (entry->kind == IREE_HAL_AMDGPU_VMM_QUARANTINE_RESERVATION) {
        iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
            (iree_hal_amdgpu_virtual_memory_reservation_t*)entry;
        reservation->base_ptr = NULL;
        if (reservation->quarantined_buffer) {
          iree_hal_buffer_t* virtual_buffer = reservation->quarantined_buffer;
          reservation->quarantined_buffer = NULL;
          iree_hal_amdgpu_virtual_memory_dispose_consumed_reservation(
              virtual_buffer);
        } else {
          iree_hal_amdgpu_virtual_memory_reservation_release_record(
              reservation);
        }
      } else {
        iree_hal_physical_memory_t* physical_memory =
            (iree_hal_physical_memory_t*)entry;
        physical_memory->allocation_handle = (hsa_amd_vmem_alloc_handle_t){0};
        iree_hal_amdgpu_physical_memory_dispose_consumed(
            physical_memory->creator_state, physical_memory);
      }
    } else {
      if (first_failure == HSA_STATUS_SUCCESS) {
        first_failure = cleanup_status;
        first_failure_kind = entry->kind;
      }
      if (failed_tail) {
        failed_tail->next = entry;
      } else {
        failed_head = entry;
      }
      failed_tail = entry;
    }
    entry = next;
  }

  // Older failed owners remain ahead of owners enqueued during the detached
  // drain so every retry observes stable FIFO ownership.
  iree_slim_mutex_lock(&domain->quarantine_mutex);
  IREE_ASSERT(domain->quarantine_count >= success_count);
  domain->quarantine_count -= success_count;
  if (failed_head) {
    failed_tail->next = domain->quarantine_head;
    domain->quarantine_head = failed_head;
    if (!domain->quarantine_tail) domain->quarantine_tail = failed_tail;
  }
  domain->quarantine_drain_active = false;
  iree_slim_mutex_unlock(&domain->quarantine_mutex);

  return first_failure == HSA_STATUS_SUCCESS
             ? iree_ok_status()
             : iree_hal_amdgpu_vmm_cleanup_status(first_failure_kind,
                                                  first_failure);
}

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
void iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native_once(void) {
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native,
                    1, iree_memory_order_release);
}

uint64_t iree_hal_amdgpu_vmm_test_last_virtual_reserve_address(void) {
  return iree_hal_amdgpu_libhsa_test_last_vmem_address_reserve_address();
}

uint64_t iree_hal_amdgpu_vmm_test_last_virtual_reserve_alignment(void) {
  return iree_hal_amdgpu_libhsa_test_last_vmem_address_reserve_alignment();
}

void iree_hal_amdgpu_vmm_test_fail_cleanup_count(bool physical_memory,
                                                 int failure_count) {
  IREE_ASSERT(failure_count >= 0);
  iree_atomic_store(
      physical_memory
          ? &iree_hal_amdgpu_vmm_test_fail_physical_cleanup_count
          : &iree_hal_amdgpu_vmm_test_fail_reservation_cleanup_count,
      failure_count, iree_memory_order_release);
}

uint64_t iree_hal_amdgpu_vmm_test_quarantine_count(void) {
  return (uint64_t)iree_atomic_load(
      &iree_hal_amdgpu_vmm_test_quarantine_owner_count,
      iree_memory_order_acquire);
}

uint64_t iree_hal_amdgpu_vmm_test_cleanup_attempt_count(bool physical_memory) {
  return (uint64_t)iree_atomic_load(
      physical_memory
          ? &iree_hal_amdgpu_vmm_test_physical_cleanup_attempt_count
          : &iree_hal_amdgpu_vmm_test_reservation_cleanup_attempt_count,
      iree_memory_order_acquire);
}

uint32_t iree_hal_amdgpu_vmm_test_last_cleanup_status(bool physical_memory) {
  return (uint32_t)iree_atomic_load(
      physical_memory
          ? &iree_hal_amdgpu_vmm_test_last_physical_cleanup_status
          : &iree_hal_amdgpu_vmm_test_last_reservation_cleanup_status,
      iree_memory_order_acquire);
}

void iree_hal_amdgpu_vmm_test_arm_quarantine_drain_pause(void) {
  iree_call_once(&iree_hal_amdgpu_vmm_test_drain_pause_once,
                 iree_hal_amdgpu_vmm_test_drain_pause_initialize);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_entered, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_released, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_armed, 1,
                    iree_memory_order_release);
}

void iree_hal_amdgpu_vmm_test_wait_quarantine_drain_paused(void) {
  iree_call_once(&iree_hal_amdgpu_vmm_test_drain_pause_once,
                 iree_hal_amdgpu_vmm_test_drain_pause_initialize);
  iree_notification_await(&iree_hal_amdgpu_vmm_test_drain_pause_notification,
                          iree_hal_amdgpu_vmm_test_drain_pause_has_entered,
                          NULL, iree_infinite_timeout());
}

void iree_hal_amdgpu_vmm_test_release_quarantine_drain_pause(void) {
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_released, 1,
                    iree_memory_order_release);
  iree_notification_post(&iree_hal_amdgpu_vmm_test_drain_pause_notification,
                         IREE_ALL_WAITERS);
}

int iree_hal_amdgpu_vmm_test_drain_attempt_kind(int attempt_ordinal) {
  if (attempt_ordinal <= 0 ||
      (size_t)attempt_ordinal >
          IREE_ARRAYSIZE(iree_hal_amdgpu_vmm_test_drain_attempt_kinds)) {
    return 0;
  }
  const int32_t attempt_count = iree_atomic_load(
      &iree_hal_amdgpu_vmm_test_drain_attempt_count, iree_memory_order_acquire);
  if (attempt_ordinal > attempt_count) return 0;
  return iree_atomic_load(
      &iree_hal_amdgpu_vmm_test_drain_attempt_kinds[attempt_ordinal - 1],
      iree_memory_order_acquire);
}

void iree_hal_amdgpu_vmm_test_reset_quarantine_observability(void) {
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native,
                    0, iree_memory_order_release);
  iree_hal_amdgpu_libhsa_test_reset_vmem_address_reserve_observability();
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_fail_reservation_cleanup_count, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_fail_physical_cleanup_count, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_reservation_cleanup_attempt_count,
                    0, iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_physical_cleanup_attempt_count, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_last_reservation_cleanup_status,
                    HSA_STATUS_SUCCESS, iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_last_physical_cleanup_status,
                    HSA_STATUS_SUCCESS, iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_armed, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_entered, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_pause_released, 0,
                    iree_memory_order_release);
  iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_attempt_count, 0,
                    iree_memory_order_release);
  for (size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_vmm_test_drain_attempt_kinds); ++i) {
    iree_atomic_store(&iree_hal_amdgpu_vmm_test_drain_attempt_kinds[i], 0,
                      iree_memory_order_release);
  }
}
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

iree_status_t iree_hal_amdgpu_virtual_memory_advise(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_advice_t advice) {
  (void)advice;
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
          range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory advice range exceeds the "
                            "reservation");
  }
  if (!iree_hal_queue_family_affinity_is_any(queue_family_affinity) &&
      IREE_UNLIKELY(
          iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
          !iree_all_bits_set(range.queue_family_affinity,
                             queue_family_affinity))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory advice family affinity 0x%016" PRIx64
        " exceeds reservation affinity 0x%016" PRIx64,
        queue_family_affinity, range.queue_family_affinity);
  }
  return iree_ok_status();
}
