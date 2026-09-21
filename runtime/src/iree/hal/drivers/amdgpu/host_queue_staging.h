// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_STAGING_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_STAGING_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/notification_ring.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default byte length of one file staging slot.
#define IREE_HAL_AMDGPU_STAGING_SLOT_SIZE_DEFAULT (16 * 1024 * 1024)

// Required byte alignment for the staging allocation base and every slot.
#define IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT (2 * 1024 * 1024)

// Default number of file staging slots per physical device.
#define IREE_HAL_AMDGPU_STAGING_SLOT_COUNT_DEFAULT 4u

typedef struct iree_hal_amdgpu_host_queue_t iree_hal_amdgpu_host_queue_t;
typedef struct iree_hal_amdgpu_staging_transfer_t
    iree_hal_amdgpu_staging_transfer_t;
typedef struct iree_hal_amdgpu_staging_pool_waiter_t
    iree_hal_amdgpu_staging_pool_waiter_t;

// Options controlling the per-physical-device queue_read/queue_write staging
// pool.
typedef struct iree_hal_amdgpu_staging_pool_options_t {
  // Byte length of each staging slot. Must be a non-zero power of two and at
  // least IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT so every slot begins on a
  // large-page-friendly boundary.
  iree_host_size_t slot_size;
  // Number of staging slots. Must be non-zero and a power of two.
  uint32_t slot_count;
  // Forces the staging allocation to use the fine-grained host pool instead of
  // the preferred coarse-grained host pool.
  uint32_t force_fine_host_memory : 1;
  // Reserved for future staging allocation policy bits.
  uint32_t reserved : 31;
} iree_hal_amdgpu_staging_pool_options_t;

// Fixed-size host/device-visible staging pool shared by one physical device.
typedef struct iree_hal_amdgpu_staging_pool_t {
  // Host allocator used for the free-slot ring.
  iree_allocator_t host_allocator;
  // HAL buffer wrapping the whole staging allocation.
  iree_hal_buffer_t* buffer;
  // Host pointer to the first staging byte.
  uint8_t* host_base;
  // Byte length of each staging slot.
  iree_host_size_t slot_size;
  // Number of staging slots in |buffer|.
  uint32_t slot_count;
  // Mask used to wrap free-slot ring indices.
  uint32_t slot_mask;
  // Serializes free-slot and waiter FIFO state.
  iree_slim_mutex_t mutex;
  // Number of entries currently available in |free_slots|.
  uint32_t available_count;
  // Read index into |free_slots|.
  uint32_t free_read;
  // Write index into |free_slots|.
  uint32_t free_write;
  // Ring of available slot ordinals.
  uint32_t* free_slots;
  // First waiter blocked on slot availability.
  iree_hal_amdgpu_staging_pool_waiter_t* waiter_head;
  // Last waiter blocked on slot availability.
  iree_hal_amdgpu_staging_pool_waiter_t* waiter_tail;
  // Number of waiters removed from the FIFO whose callbacks have not yet
  // returned. Protected by |mutex|.
  uint32_t claimed_waiter_count;
} iree_hal_amdgpu_staging_pool_t;

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
// Test-only observer invoked after a staging waiter has atomically
// claimed a slot and the pool mutex has been released, immediately before its
// callback runs. Present only in the explicitly instrumented test library.
typedef void(IREE_API_PTR* iree_hal_amdgpu_staging_waiter_claimed_observer_t)(
    void* user_data);

// Sets the process-wide staging-waiter observer. Intended for deterministic
// lifecycle tests; production callers leave it NULL.
void iree_hal_amdgpu_host_queue_staging_set_waiter_claimed_observer(
    iree_hal_amdgpu_staging_waiter_claimed_observer_t observer,
    void* user_data);

// Test/diagnostic observer invoked after a completed chunk has captured and
// cleared its exact slot under the transfer mutex, immediately before that
// captured ordinal is returned to the staging pool.
typedef void(
    IREE_API_PTR* iree_hal_amdgpu_staging_chunk_slot_release_observer_t)(
    void* user_data, uint32_t captured_slot_ordinal);

void iree_hal_amdgpu_host_queue_staging_set_chunk_slot_release_observer(
    iree_hal_amdgpu_staging_chunk_slot_release_observer_t observer,
    void* user_data);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

// Initializes |out_options| to its default values.
void iree_hal_amdgpu_staging_pool_options_initialize(
    iree_hal_amdgpu_staging_pool_options_t* out_options);

// Verifies |options| for use by a staging pool.
iree_status_t iree_hal_amdgpu_staging_pool_options_verify(
    const iree_hal_amdgpu_staging_pool_options_t* options);

// Initializes a fixed-size staging pool in caller-owned storage.
iree_status_t iree_hal_amdgpu_staging_pool_initialize(
    iree_hal_device_t* logical_device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    const iree_hal_amdgpu_staging_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_amdgpu_staging_pool_t* out_pool);

// Deinitializes |pool| and releases its fixed staging allocation.
void iree_hal_amdgpu_staging_pool_deinitialize(
    iree_hal_amdgpu_staging_pool_t* pool);

// Captures a borrowed host-to-device transfer without scheduling it.
iree_status_t iree_hal_amdgpu_staging_transfer_create_upload(
    iree_hal_amdgpu_host_queue_t* queue, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length,
    iree_hal_amdgpu_staging_transfer_t** out_transfer);

// Captures a device-to-borrowed-host transfer without scheduling it.
iree_status_t iree_hal_amdgpu_staging_transfer_create_download(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, void* target, iree_device_size_t length,
    iree_hal_amdgpu_staging_transfer_t** out_transfer);

// Starts a captured host transfer. The optional |completion_resource| keeps
// |completion_action.user_data| live until the action executes. The completion
// action runs outside notification-ring drain with a NULL reclaim entry.
iree_status_t iree_hal_amdgpu_staging_transfer_start(
    iree_hal_amdgpu_staging_transfer_t* transfer,
    iree_hal_amdgpu_reclaim_action_t completion_action,
    iree_hal_resource_t* completion_resource);

// Releases a captured host transfer.
void iree_hal_amdgpu_staging_transfer_release(
    iree_hal_amdgpu_staging_transfer_t* transfer);

// Detaches every staged/proactor publisher registered with |queue| and
// requests cancellation. Queue admission must already be permanently closed.
// This does not wait: hardware and post-drain completions may be required for
// a publisher to reach its terminal callback.
void iree_hal_amdgpu_staging_transfer_cancel_all(
    iree_hal_amdgpu_host_queue_t* queue);

// Joins all publishers detached by
// iree_hal_amdgpu_staging_transfer_cancel_all, then consumes their queue-owned
// registry edges. Hardware completions and post-drain actions must be drained
// before this call. Returns only when no staged/proactor callback can touch
// |queue| again.
void iree_hal_amdgpu_staging_transfer_await_all(
    iree_hal_amdgpu_host_queue_t* queue);

// Submits a chunked fd-backed queue_read through the staging pool.
iree_status_t iree_hal_amdgpu_host_queue_submit_staged_read(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length);

// Submits a chunked fd-backed queue_write through the staging pool.
iree_status_t iree_hal_amdgpu_host_queue_submit_staged_write(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_STAGING_H_
