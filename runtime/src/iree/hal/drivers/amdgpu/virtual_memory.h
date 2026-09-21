// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;
typedef struct iree_hal_amdgpu_virtual_memory_state_t
    iree_hal_amdgpu_virtual_memory_state_t;
typedef struct iree_hal_amdgpu_vmm_native_operation_t
    iree_hal_amdgpu_vmm_native_operation_t;

// Placement and capability information selected by the AMDGPU allocator.
// The virtual-memory component consumes this representation without owning the
// allocator's broader pool-selection policy.
typedef struct iree_hal_amdgpu_virtual_memory_placement_t {
  // HSA memory pool used for the physical allocation.
  hsa_amd_memory_pool_t memory_pool;

  // HAL device that owns virtual-buffer placement metadata.
  iree_hal_device_t* device;

  // HAL queue families permitted to access the virtual address range.
  iree_hal_queue_family_affinity_t queue_family_affinity;

  // HAL memory type exposed by the virtual buffer and physical allocation.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage exposed by the virtual buffer.
  iree_hal_buffer_usage_t buffer_usage;

  // Atomic memory cells supported by every family in
  // |queue_family_affinity|.
  iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells;

  // Unowned source GPU masks for allocations from |memory_pool|.
  const iree_hal_amdgpu_atomic_memory_source_masks_t*
      atomic_memory_source_masks;

  // Smallest legal HSA VMM allocation and mapping multiple.
  iree_device_size_t minimum_granule;

  // Preferred HSA VMM allocation multiple for reducing fragmentation.
  iree_device_size_t recommended_granule;

  // Maximum physical allocation size accepted by this memory pool.
  iree_device_size_t max_allocation_size;
} iree_hal_amdgpu_virtual_memory_placement_t;

// Creates the virtual-memory context for one AMDGPU allocator.
iree_status_t iree_hal_amdgpu_virtual_memory_state_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_hal_device_t* device,
    iree_hal_allocator_statistics_t* statistics,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_memory_state_t** out_state);

// Destroys an allocator-owned VMM context after all handles are released.
void iree_hal_amdgpu_virtual_memory_state_destroy(
    iree_hal_amdgpu_virtual_memory_state_t* state);

// Reserves a virtual buffer with no physical backing or access permissions.
// |minimum_alignment| may be zero to select the placement default. A non-zero
// value must be a power of two no smaller than |placement.minimum_granule|.
// |requested_address| may be zero when no address is preferred. The requested
// address is advisory and the native allocator may select another range.
iree_status_t iree_hal_amdgpu_virtual_memory_reserve(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_device_size_t minimum_alignment,
    iree_device_size_t requested_address,
    iree_hal_buffer_t** out_virtual_buffer);

// Releases an unmapped virtual buffer returned by virtual_memory_reserve.
iree_status_t iree_hal_amdgpu_virtual_memory_release(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer);

// Consumes |virtual_buffer| even when native cleanup fails by transferring its
// sole ownership into the domain retry quarantine.
iree_status_t iree_hal_amdgpu_virtual_memory_release_or_quarantine(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer);

// Creates a target-device HAL alias over a mapped portion of a reservation.
// The alias only retains the reservation buffer; it never owns the native
// address, mapping, physical handle, or access permission.
iree_status_t iree_hal_amdgpu_virtual_memory_alias(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_t** out_alias_buffer);

// Creates a physical allocation that can be mapped into one or more virtual
// buffers associated with this state.
iree_status_t iree_hal_amdgpu_physical_memory_allocate(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory);

// Releases an unmapped physical allocation.
iree_status_t iree_hal_amdgpu_physical_memory_free(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_physical_memory_t* physical_memory);

// Consumes |physical_memory| even when native cleanup fails by transferring
// its exact owner record into the domain retry quarantine.
iree_status_t iree_hal_amdgpu_physical_memory_free_or_quarantine(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_physical_memory_t* physical_memory);

// Retries every owner present at drain entry without holding the domain mutex
// across HSA. Failed owners remain queued; success consumes their wrappers.
iree_status_t iree_hal_amdgpu_vmm_quarantine_drain(
    iree_hal_amdgpu_virtual_memory_state_t* state);

// Maps physical memory into an unmapped virtual address range.
iree_status_t iree_hal_amdgpu_virtual_memory_map(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size);

// Unmaps a fully mapped virtual address range.
iree_status_t iree_hal_amdgpu_virtual_memory_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size);

// Sets the requested access permissions on a mapped virtual address range.
iree_status_t iree_hal_amdgpu_virtual_memory_protect(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection);

// Applies device access through the reservation-owning state to exactly the
// GPU agents represented by |access_state|. The states may differ but must
// share one native HSA domain.
iree_status_t iree_hal_amdgpu_virtual_memory_protect_peer(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_amdgpu_virtual_memory_state_t* access_state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_memory_protection_t protection);

// Prepares one exact native access mutation after validating the reservation,
// exact access-agent group, and every mapped physical pool named by
// |physical_memories|. Preparation performs no native VMM mutation.
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
    iree_hal_amdgpu_vmm_native_operation_t** out_operation);

// Prepares one exact native unmap without mutating the reservation.
iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation);

// Prepares release of one exact, already-unmapped virtual reservation.
iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_release_reservation(
    iree_hal_amdgpu_virtual_memory_state_t* reservation_state,
    iree_hal_buffer_t* virtual_buffer, iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation);

// Prepares release of one exact, already-unmapped physical allocation.
iree_status_t iree_hal_amdgpu_vmm_native_operation_prepare_free_physical(
    iree_hal_amdgpu_virtual_memory_state_t* creator_state,
    iree_hal_physical_memory_t* physical_memory,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_vmm_native_operation_t** out_operation);

// Applies one prepared native mutation without allocating or constructing an
// IREE status. Successful ownership-consuming operations atomically make their
// normal release callbacks inert before returning.
hsa_status_t iree_hal_amdgpu_vmm_native_operation_apply(
    iree_hal_amdgpu_vmm_native_operation_t* operation);

// Returns true when applying |operation| may synchronously invoke registered
// HSA deallocation callbacks on the calling thread.
bool iree_hal_amdgpu_vmm_native_operation_may_invoke_callbacks(
    const iree_hal_amdgpu_vmm_native_operation_t* operation);

// Releases a prepared operation. This never applies or rolls back it.
void iree_hal_amdgpu_vmm_native_operation_destroy(
    iree_hal_amdgpu_vmm_native_operation_t* operation);

// Disposes wrappers whose native reservation/physical handle was consumed by
// a successful prepared operation. Both functions are allocation-free and
// issue no native VMM call.
void iree_hal_amdgpu_virtual_memory_dispose_consumed_reservation(
    iree_hal_buffer_t* virtual_buffer);
void iree_hal_amdgpu_physical_memory_dispose_consumed(
    iree_hal_amdgpu_virtual_memory_state_t* creator_state,
    iree_hal_physical_memory_t* physical_memory);

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
void iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native_once(void);
uint64_t iree_hal_amdgpu_vmm_test_last_virtual_reserve_address(void);
uint64_t iree_hal_amdgpu_vmm_test_last_virtual_reserve_alignment(void);
void iree_hal_amdgpu_vmm_test_fail_cleanup_count(bool physical_memory,
                                                 int failure_count);
uint64_t iree_hal_amdgpu_vmm_test_quarantine_count(void);
uint64_t iree_hal_amdgpu_vmm_test_cleanup_attempt_count(bool physical_memory);
uint32_t iree_hal_amdgpu_vmm_test_last_cleanup_status(bool physical_memory);
void iree_hal_amdgpu_vmm_test_reset_quarantine_observability(void);
void iree_hal_amdgpu_vmm_test_arm_quarantine_drain_pause(void);
void iree_hal_amdgpu_vmm_test_wait_quarantine_drain_paused(void);
void iree_hal_amdgpu_vmm_test_release_quarantine_drain_pause(void);
int iree_hal_amdgpu_vmm_test_drain_attempt_kind(int attempt_ordinal);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

// Validates an advisory virtual-address range and otherwise performs no work.
iree_status_t iree_hal_amdgpu_virtual_memory_advise(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_advice_t advice);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_MEMORY_H_
