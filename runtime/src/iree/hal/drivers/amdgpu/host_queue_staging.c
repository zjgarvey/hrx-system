// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_staging.h"

#include <string.h>

#include "iree/async/operations/file.h"
#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/slab_provider.h"

typedef struct iree_hal_amdgpu_staging_slot_t {
  // Slot ordinal in the physical-device staging pool.
  uint32_t ordinal;
  // Byte offset of the slot inside the staging buffer.
  iree_device_size_t buffer_offset;
  // Host-accessible bytes for file I/O.
  iree_byte_span_t host_span;
  // HAL buffer wrapping the complete staging allocation. Borrowed from the
  // pool; queue copy submissions retain it while the GPU owns the slot.
  iree_hal_buffer_t* buffer;
} iree_hal_amdgpu_staging_slot_t;

typedef struct iree_hal_amdgpu_staging_allocation_t {
  // Borrowed HSA API table used to free |allocation_base|.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Host allocator used to allocate this release state.
  iree_allocator_t host_allocator;
  // Original pointer returned by hsa_amd_memory_pool_allocate.
  void* allocation_base;
} iree_hal_amdgpu_staging_allocation_t;

typedef enum iree_hal_amdgpu_staging_pool_waiter_state_e {
  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE = 0,
  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_QUEUED = 1,
  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED = 2,
  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CALLBACK = 3,
} iree_hal_amdgpu_staging_pool_waiter_state_t;

// Callback invoked when a staging slot may be available.
typedef void(IREE_API_PTR* iree_hal_amdgpu_staging_pool_waiter_fn_t)(
    void* user_data);

// Callback invoked after one claimed waiter callback has returned and its
// next stable state has been published. |may_finish| is false when the waiter
// immediately claimed another available slot and another callback follows.
typedef void(IREE_API_PTR* iree_hal_amdgpu_staging_pool_waiter_after_fn_t)(
    void* user_data, bool may_finish);

// Intrusive waiter used by transfers that cannot acquire a staging slot.
struct iree_hal_amdgpu_staging_pool_waiter_t {
  // Pool owning this waiter after its first publication.
  iree_hal_amdgpu_staging_pool_t* pool;
  // Next waiter in the pool-owned FIFO list.
  iree_hal_amdgpu_staging_pool_waiter_t* next;
  // Callback invoked after the waiter is removed from the FIFO list.
  iree_hal_amdgpu_staging_pool_waiter_fn_t fn;
  // Callback consuming the reference associated with one waiter callback.
  iree_hal_amdgpu_staging_pool_waiter_after_fn_t after_fn;
  // User data passed to |fn|.
  void* user_data;
  // Slot reserved for this waiter when it is dequeued.
  iree_hal_amdgpu_staging_slot_t slot;
  // Wakes cancellation after a CLAIMED/CALLBACK waiter reaches a stable state.
  iree_notification_t state_notification;
  // Exact lifecycle state protected by |pool->mutex|.
  iree_hal_amdgpu_staging_pool_waiter_state_t state;
  // True when the callback requested another waiter publication. Protected by
  // |pool->mutex| and consumed after the callback returns.
  bool requeue_requested;
};

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
static iree_hal_amdgpu_staging_waiter_claimed_observer_t
    iree_hal_amdgpu_staging_waiter_claimed_observer = NULL;
static void* iree_hal_amdgpu_staging_waiter_claimed_observer_user_data = NULL;
static iree_hal_amdgpu_staging_chunk_slot_release_observer_t
    iree_hal_amdgpu_staging_chunk_slot_release_observer = NULL;
static void* iree_hal_amdgpu_staging_chunk_slot_release_observer_user_data =
    NULL;

void iree_hal_amdgpu_host_queue_staging_set_waiter_claimed_observer(
    iree_hal_amdgpu_staging_waiter_claimed_observer_t observer,
    void* user_data) {
  iree_hal_amdgpu_staging_waiter_claimed_observer = observer;
  iree_hal_amdgpu_staging_waiter_claimed_observer_user_data = user_data;
}

void iree_hal_amdgpu_host_queue_staging_set_chunk_slot_release_observer(
    iree_hal_amdgpu_staging_chunk_slot_release_observer_t observer,
    void* user_data) {
  iree_hal_amdgpu_staging_chunk_slot_release_observer = observer;
  iree_hal_amdgpu_staging_chunk_slot_release_observer_user_data = user_data;
}
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
static void iree_hal_amdgpu_staging_waiter_tail_before_wake(void* user_data);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

typedef enum iree_hal_amdgpu_staging_pool_wait_result_e {
  // The waiter was newly queued and will receive a future callback.
  IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED = 0,
  // The waiter was already queued by an earlier pump attempt.
  IREE_HAL_AMDGPU_STAGING_POOL_WAIT_ALREADY_QUEUED = 1,
  // A slot became available before the waiter was queued; retry acquire.
  IREE_HAL_AMDGPU_STAGING_POOL_WAIT_RETRY = 2,
} iree_hal_amdgpu_staging_pool_wait_result_t;

void iree_hal_amdgpu_staging_pool_options_initialize(
    iree_hal_amdgpu_staging_pool_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->slot_size = IREE_HAL_AMDGPU_STAGING_SLOT_SIZE_DEFAULT;
  out_options->slot_count = IREE_HAL_AMDGPU_STAGING_SLOT_COUNT_DEFAULT;
}

iree_status_t iree_hal_amdgpu_staging_pool_options_verify(
    const iree_hal_amdgpu_staging_pool_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  if (options->slot_size == 0 ||
      !iree_host_size_is_power_of_two(options->slot_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "staging slot size must be a non-zero power of two (got %" PRIhsz ")",
        options->slot_size);
  }
  if (options->slot_size < IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "staging slot size must be at least %" PRIhsz
        " bytes to preserve slot "
        "alignment (got %" PRIhsz ")",
        (iree_host_size_t)IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
        options->slot_size);
  }
  if (options->slot_count == 0 ||
      !iree_host_size_is_power_of_two(options->slot_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "staging slot count must be a non-zero power of two (got %u)",
        options->slot_count);
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(options->slot_size, options->slot_count,
                                  &total_size) ||
      total_size > (iree_host_size_t)IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staging pool size overflows (slot_size=%" PRIhsz
                            ", slot_count=%u)",
                            options->slot_size, options->slot_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_staging_pool_resolve_access_agents(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  return iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
      topology, queue_family_affinity, out_agent_list);
}

static void iree_hal_amdgpu_staging_allocation_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_amdgpu_staging_allocation_t* allocation =
      (iree_hal_amdgpu_staging_allocation_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (allocation->allocation_base) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(allocation->libhsa,
                                          allocation->allocation_base));
  }
  iree_allocator_free(allocation->host_allocator, allocation);
  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_amdgpu_staging_pool_waiter_t*
iree_hal_amdgpu_staging_pool_pop_waiter_locked(
    iree_hal_amdgpu_staging_pool_t* pool,
    const iree_hal_amdgpu_staging_slot_t* slot) {
  iree_hal_amdgpu_staging_pool_waiter_t* waiter = pool->waiter_head;
  if (waiter) {
    pool->waiter_head = waiter->next;
    if (!pool->waiter_head) {
      pool->waiter_tail = NULL;
    }
    waiter->next = NULL;
    waiter->slot = *slot;
    IREE_ASSERT(waiter->state ==
                IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_QUEUED);
    waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED;
    ++pool->claimed_waiter_count;
  }
  return waiter;
}

static bool iree_hal_amdgpu_staging_pool_try_acquire(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_slot_t* out_slot) {
  bool did_acquire = false;
  iree_slim_mutex_lock(&pool->mutex);
  if (pool->available_count > 0) {
    const uint32_t slot_ordinal =
        pool->free_slots[pool->free_read++ & pool->slot_mask];
    --pool->available_count;
    out_slot->ordinal = slot_ordinal;
    out_slot->buffer_offset =
        (iree_device_size_t)slot_ordinal * pool->slot_size;
    out_slot->host_span = iree_make_byte_span(
        pool->host_base + (iree_host_size_t)out_slot->buffer_offset,
        pool->slot_size);
    out_slot->buffer = pool->buffer;
    did_acquire = true;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return did_acquire;
}

static bool iree_hal_amdgpu_staging_pool_take_waiter_slot(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_pool_waiter_t* waiter,
    iree_hal_amdgpu_staging_slot_t* out_slot) {
  bool did_take = false;
  iree_slim_mutex_lock(&pool->mutex);
  if (waiter->slot.buffer) {
    *out_slot = waiter->slot;
    memset(&waiter->slot, 0, sizeof(waiter->slot));
    did_take = true;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return did_take;
}

static iree_hal_amdgpu_staging_pool_wait_result_t
iree_hal_amdgpu_staging_pool_queue_waiter(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_pool_waiter_t* waiter,
    iree_hal_amdgpu_staging_pool_waiter_fn_t fn,
    iree_hal_amdgpu_staging_pool_waiter_after_fn_t after_fn, void* user_data) {
  iree_hal_amdgpu_staging_pool_wait_result_t result =
      IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED;
  iree_slim_mutex_lock(&pool->mutex);
  IREE_ASSERT(!waiter->pool || waiter->pool == pool);
  waiter->pool = pool;
  if (waiter->state == IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_QUEUED ||
      waiter->state == IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED) {
    result = IREE_HAL_AMDGPU_STAGING_POOL_WAIT_ALREADY_QUEUED;
  } else if (waiter->state ==
             IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CALLBACK) {
    if (waiter->requeue_requested) {
      result = IREE_HAL_AMDGPU_STAGING_POOL_WAIT_ALREADY_QUEUED;
    } else {
      waiter->fn = fn;
      waiter->after_fn = after_fn;
      waiter->user_data = user_data;
      waiter->requeue_requested = true;
    }
  } else if (pool->available_count > 0) {
    result = IREE_HAL_AMDGPU_STAGING_POOL_WAIT_RETRY;
  } else {
    IREE_ASSERT(waiter->state ==
                IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE);
    waiter->next = NULL;
    waiter->fn = fn;
    waiter->after_fn = after_fn;
    waiter->user_data = user_data;
    waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_QUEUED;
    if (pool->waiter_tail) {
      pool->waiter_tail->next = waiter;
    } else {
      pool->waiter_head = waiter;
    }
    pool->waiter_tail = waiter;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return result;
}

static bool iree_hal_amdgpu_staging_pool_waiter_is_stable(void* user_data) {
  iree_hal_amdgpu_staging_pool_waiter_t* waiter =
      (iree_hal_amdgpu_staging_pool_waiter_t*)user_data;
  iree_hal_amdgpu_staging_pool_t* pool = waiter->pool;
  iree_slim_mutex_lock(&pool->mutex);
  const bool is_stable =
      waiter->state != IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED &&
      waiter->state != IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CALLBACK;
  iree_slim_mutex_unlock(&pool->mutex);
  return is_stable;
}

static bool iree_hal_amdgpu_staging_pool_cancel_waiter(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_pool_waiter_t* waiter) {
  IREE_ASSERT(!waiter->pool || waiter->pool == pool);
  for (;;) {
    iree_slim_mutex_lock(&pool->mutex);
    if (waiter->state == IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE) {
      iree_slim_mutex_unlock(&pool->mutex);
      return false;
    }
    if (waiter->state == IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_QUEUED) {
      iree_hal_amdgpu_staging_pool_waiter_t* previous = NULL;
      for (iree_hal_amdgpu_staging_pool_waiter_t* current = pool->waiter_head;
           current != NULL; current = current->next) {
        if (current == waiter) {
          if (previous) {
            previous->next = current->next;
          } else {
            pool->waiter_head = current->next;
          }
          if (pool->waiter_tail == current) {
            pool->waiter_tail = previous;
          }
          waiter->next = NULL;
          waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE;
          waiter->requeue_requested = false;
          break;
        }
        previous = current;
      }
      IREE_ASSERT(waiter->state ==
                  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE);
      iree_slim_mutex_unlock(&pool->mutex);
      iree_notification_post(&waiter->state_notification, IREE_ALL_WAITERS);
      return true;
    }
    IREE_ASSERT(
        waiter->state == IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED ||
        waiter->state == IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CALLBACK);
    iree_slim_mutex_unlock(&pool->mutex);
    iree_notification_await(&waiter->state_notification,
                            iree_hal_amdgpu_staging_pool_waiter_is_stable,
                            waiter, iree_infinite_timeout());
  }
}

static void iree_hal_amdgpu_staging_pool_release(
    iree_hal_amdgpu_staging_pool_t* pool, uint32_t slot_ordinal) {
  iree_hal_amdgpu_staging_slot_t slot = {
      .ordinal = slot_ordinal,
      .buffer_offset = (iree_device_size_t)slot_ordinal * pool->slot_size,
      .host_span = iree_make_byte_span(
          pool->host_base + (iree_host_size_t)slot_ordinal * pool->slot_size,
          pool->slot_size),
      .buffer = pool->buffer,
  };
  iree_hal_amdgpu_staging_pool_waiter_t* waiter = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  waiter = iree_hal_amdgpu_staging_pool_pop_waiter_locked(pool, &slot);
  if (!waiter) {
    pool->free_slots[pool->free_write++ & pool->slot_mask] = slot_ordinal;
    ++pool->available_count;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  while (waiter) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
    if (iree_hal_amdgpu_staging_waiter_claimed_observer) {
      iree_hal_amdgpu_staging_waiter_claimed_observer(
          iree_hal_amdgpu_staging_waiter_claimed_observer_user_data);
    }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

    iree_slim_mutex_lock(&pool->mutex);
    IREE_ASSERT(waiter->state ==
                IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED);
    waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CALLBACK;
    iree_slim_mutex_unlock(&pool->mutex);

    waiter->fn(waiter->user_data);

    bool has_immediate_callback = false;
    iree_slim_mutex_lock(&pool->mutex);
    IREE_ASSERT(waiter->state ==
                IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CALLBACK);
    IREE_ASSERT(pool->claimed_waiter_count > 0);
    --pool->claimed_waiter_count;
    if (waiter->requeue_requested) {
      waiter->requeue_requested = false;
      if (pool->available_count > 0) {
        const uint32_t next_slot_ordinal =
            pool->free_slots[pool->free_read++ & pool->slot_mask];
        --pool->available_count;
        waiter->slot = (iree_hal_amdgpu_staging_slot_t){
            .ordinal = next_slot_ordinal,
            .buffer_offset =
                (iree_device_size_t)next_slot_ordinal * pool->slot_size,
            .host_span = iree_make_byte_span(
                pool->host_base +
                    (iree_host_size_t)next_slot_ordinal * pool->slot_size,
                pool->slot_size),
            .buffer = pool->buffer,
        };
        waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_CLAIMED;
        ++pool->claimed_waiter_count;
        has_immediate_callback = true;
      } else {
        waiter->next = NULL;
        waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_QUEUED;
        if (pool->waiter_tail) {
          pool->waiter_tail->next = waiter;
        } else {
          pool->waiter_head = waiter;
        }
        pool->waiter_tail = waiter;
      }
    } else {
      waiter->state = IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE;
    }
    // Cache the tail callback before publishing a stable waiter state. Its
    // transfer-owned reference keeps the embedded waiter and pool alive, and
    // the callback performs every remaining transfer touch after this
    // function's final waiter access.
    iree_hal_amdgpu_staging_pool_waiter_after_fn_t after_fn = waiter->after_fn;
    void* after_user_data = waiter->user_data;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
    iree_hal_amdgpu_staging_waiter_tail_before_wake(after_user_data);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
    // State publication and wake form one lock-linearized operation. A
    // canceller cannot observe the stable state and free waiter storage before
    // the wake has completed.
    iree_notification_post(&waiter->state_notification, IREE_ALL_WAITERS);
    iree_slim_mutex_unlock(&pool->mutex);
    after_fn(after_user_data, !has_immediate_callback);
    if (!has_immediate_callback) break;
  }
}

iree_status_t iree_hal_amdgpu_staging_pool_initialize(
    iree_hal_device_t* logical_device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    const iree_hal_amdgpu_staging_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_amdgpu_staging_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(host_memory_pools);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options->slot_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options->slot_count);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_staging_pool_options_verify(options));

  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->host_allocator = host_allocator;
  out_pool->slot_size = options->slot_size;
  out_pool->slot_count = options->slot_count;
  out_pool->slot_mask = options->slot_count - 1u;
  iree_slim_mutex_initialize(&out_pool->mutex);

  hsa_amd_memory_pool_t memory_pool = host_memory_pools->coarse_pool;
  if (options->force_fine_host_memory || !memory_pool.handle) {
    memory_pool = host_memory_pools->fine_pool;
  }
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!memory_pool.handle)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU staging requires a host memory pool");
  }

  iree_hal_amdgpu_slab_provider_memory_pool_properties_t memory_pool_properties;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
        libhsa, memory_pool, &memory_pool_properties);
  }

  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(options->slot_size, options->slot_count,
                                  &total_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "staging pool size overflows (slot_size=%" PRIhsz
                              ", slot_count=%u)",
                              options->slot_size, options->slot_count);
  }

  iree_host_size_t free_slots_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(options->slot_count, sizeof(uint32_t),
                                  &free_slots_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "staging free-slot table size overflows");
  }

  iree_host_size_t allocation_size = total_size;
  if (iree_status_is_ok(status) && memory_pool_properties.allocation_alignment <
                                       IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT) {
    if (!iree_host_size_checked_add(total_size,
                                    IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT - 1,
                                    &allocation_size)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "staging aligned allocation size overflows");
    }
  }

  uint32_t* free_slots = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, free_slots_size,
                                   (void**)&free_slots);
  }

  void* allocation_base = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(libhsa), memory_pool, allocation_size,
        HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &allocation_base);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_access_agent_list_t access_agents;
    status = iree_hal_amdgpu_staging_pool_resolve_access_agents(
        topology, queue_family_affinity, &access_agents);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_access_allow_agent_list(libhsa, &access_agents,
                                                       allocation_base);
    }
  }

  void* host_ptr = NULL;
  if (iree_status_is_ok(status)) {
    const uintptr_t allocation_begin = (uintptr_t)allocation_base;
    const uintptr_t allocation_end = allocation_begin + allocation_size;
    iree_host_size_t aligned_host_base = 0;
    if (!iree_host_size_checked_align((iree_host_size_t)allocation_begin,
                                      IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
                                      &aligned_host_base)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HSA staging allocation base overflowed while aligning to %" PRIhsz
          " bytes (base=%p)",
          (iree_host_size_t)IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
          allocation_base);
    }
    if (iree_status_is_ok(status)) {
      const uintptr_t aligned_base = (uintptr_t)aligned_host_base;
      const uintptr_t aligned_end = aligned_base + total_size;
      if (allocation_end < allocation_begin ||
          aligned_base < allocation_begin || aligned_end < aligned_base ||
          aligned_end > allocation_end ||
          !iree_host_ptr_has_alignment(
              (void*)aligned_base, IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "HSA staging allocation could not satisfy %" PRIhsz
            "-byte alignment (base=%p, allocation_size=%" PRIhsz
            ", total_size=%" PRIhsz ")",
            (iree_host_size_t)IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
            allocation_base, allocation_size, total_size);
      } else {
        host_ptr = (void*)aligned_base;
      }
    }
  }

  iree_hal_amdgpu_staging_allocation_t* release_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*release_state),
                                   (void**)&release_state);
  }
  if (iree_status_is_ok(status)) {
    release_state->libhsa = libhsa;
    release_state->host_allocator = host_allocator;
    release_state->allocation_base = allocation_base;
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = logical_device,
        .queue_family_affinity = queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        libhsa, placement,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE,
        (iree_device_size_t)total_size, (iree_device_size_t)total_size,
        host_ptr,
        (iree_hal_buffer_release_callback_t){
            .fn = iree_hal_amdgpu_staging_allocation_release,
            .user_data = release_state,
        },
        host_allocator, &buffer);
  }

  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < options->slot_count; ++i) {
      free_slots[i] = i;
    }
    out_pool->buffer = buffer;
    out_pool->host_base = (uint8_t*)host_ptr;
    out_pool->available_count = options->slot_count;
    out_pool->free_slots = free_slots;
    out_pool->free_write = options->slot_count;
    release_state = NULL;
    allocation_base = NULL;
  } else {
    iree_hal_buffer_release(buffer);
    if (allocation_base) {
      status = iree_status_join(
          status,
          iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), allocation_base));
    }
    iree_allocator_free(host_allocator, release_state);
    iree_allocator_free(host_allocator, free_slots);
    iree_slim_mutex_deinitialize(&out_pool->mutex);
    memset(out_pool, 0, sizeof(*out_pool));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_staging_pool_deinitialize(
    iree_hal_amdgpu_staging_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  if (!pool->buffer && !pool->free_slots && pool->slot_count == 0) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&pool->mutex);
  IREE_ASSERT(pool->waiter_head == NULL && pool->waiter_tail == NULL,
              "staging pool cannot be destroyed with queued waiters");
  IREE_ASSERT(pool->claimed_waiter_count == 0,
              "staging pool cannot be destroyed during a waiter callback");
  IREE_ASSERT(pool->available_count == pool->slot_count,
              "staging pool cannot be destroyed with checked-out slots");
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_buffer_release(pool->buffer);
  iree_allocator_free(pool->host_allocator, pool->free_slots);
  iree_slim_mutex_deinitialize(&pool->mutex);
  memset(pool, 0, sizeof(*pool));
  IREE_TRACE_ZONE_END(z0);
}

typedef enum iree_hal_amdgpu_staging_transfer_kind_e {
  // File data flows from the proactor into staging and then GPU copy writes the
  // target buffer.
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ = 0,
  // GPU copy writes staging and then file data flows from staging into the
  // proactor.
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_WRITE = 1,
  // Borrowed host data is copied into staging and then into the target buffer.
  IREE_HAL_AMDGPU_STAGING_TRANSFER_UPLOAD = 2,
  // GPU copy writes staging and then staging is copied into borrowed host data.
  IREE_HAL_AMDGPU_STAGING_TRANSFER_DOWNLOAD = 3,
} iree_hal_amdgpu_staging_transfer_kind_t;

static bool iree_hal_amdgpu_staging_transfer_uses_file(
    iree_hal_amdgpu_staging_transfer_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ ||
         kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_WRITE;
}

static bool iree_hal_amdgpu_staging_transfer_copies_to_device(
    iree_hal_amdgpu_staging_transfer_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ ||
         kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_UPLOAD;
}

static iree_hal_profile_queue_event_type_t
iree_hal_amdgpu_staging_transfer_profile_event_type(
    iree_hal_amdgpu_staging_transfer_kind_t kind) {
  switch (kind) {
    case IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ;
    case IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_WRITE:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE;
    case IREE_HAL_AMDGPU_STAGING_TRANSFER_UPLOAD:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
    case IREE_HAL_AMDGPU_STAGING_TRANSFER_DOWNLOAD:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
    default:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
  }
}

typedef uint32_t iree_hal_amdgpu_staging_transfer_flags_t;
enum iree_hal_amdgpu_staging_transfer_flag_bits_e {
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING = 1u << 0,
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED = 1u << 1,
};

typedef enum iree_hal_amdgpu_staging_chunk_state_e {
  // Chunk is available for a new file subrange.
  IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE = 0,
  // Chunk has an in-flight async file read into its staging slot.
  IREE_HAL_AMDGPU_STAGING_CHUNK_READING = 1,
  // Chunk has an in-flight GPU copy from staging to the user buffer.
  IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_DEVICE = 2,
  // Chunk has an in-flight GPU copy from the user buffer to staging.
  IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_HOST = 3,
  // Chunk has an in-flight async file write from its staging slot.
  IREE_HAL_AMDGPU_STAGING_CHUNK_WRITING = 4,
  // The async file-read callback owns the operation storage and chunk.
  IREE_HAL_AMDGPU_STAGING_CHUNK_READ_CALLBACK = 5,
  // The async file-write callback owns the operation storage and chunk.
  IREE_HAL_AMDGPU_STAGING_CHUNK_WRITE_CALLBACK = 6,
} iree_hal_amdgpu_staging_chunk_state_t;

typedef struct iree_hal_amdgpu_staging_transfer_t
    iree_hal_amdgpu_staging_transfer_t;

typedef struct iree_hal_amdgpu_staging_chunk_t {
  // Owning transfer.
  iree_hal_amdgpu_staging_transfer_t* transfer;
  // Current lifecycle state.
  iree_hal_amdgpu_staging_chunk_state_t state;
  // Staging slot owned by this chunk while |state| is not IDLE.
  iree_hal_amdgpu_staging_slot_t slot;
  // Byte offset from the transfer start.
  iree_device_size_t transfer_offset;
  // Byte length assigned to this chunk.
  iree_host_size_t length;
  // Bytes completed by the current partial file operation.
  iree_host_size_t file_progress;
  // Owned status captured by the GPU copy pre-signal action for post-drain use.
  iree_status_t copy_status;
  // Owned status captured by a proactor callback for queue-owned safe
  // continuation. The callback does not touch the transfer after enqueue.
  iree_status_t file_callback_status;
  // Post-drain continuation queued by the GPU copy pre-signal action.
  iree_hal_amdgpu_host_queue_post_drain_action_t post_drain_action;
  // Async read operation storage.
  iree_async_file_read_operation_t read_op;
  // Async write operation storage.
  iree_async_file_write_operation_t write_op;
} iree_hal_amdgpu_staging_chunk_t;

struct iree_hal_amdgpu_staging_transfer_t {
  // Resource header retained by host actions, async file callbacks, and GPU
  // copy reclaim entries.
  iree_hal_resource_t resource;
  // Host allocator used for this transfer and cloned semaphore-list storage.
  iree_allocator_t host_allocator;
  // Serializes transfer counters and terminal status ownership.
  iree_slim_mutex_t mutex;
  // Wakes queue sealing after an in-progress proactor submission finishes or
  // the transfer's terminal callback returns.
  iree_notification_t state_notification;
  // True after the terminal callback has returned and this transfer can no
  // longer dereference |queue|. Published before notifying seal waiters.
  iree_atomic_int32_t terminal_complete;
  // Number of threads between initializing an async operation and returning
  // from proactor submit. Cancellation first closes transfer admission and
  // waits for this count to reach zero before inspecting operation storage.
  uint32_t io_submit_count;
  // Queue used for internal GPU copies and final user signal publication.
  iree_hal_amdgpu_host_queue_t* queue;
  // Pins queue/device state from active publication until the terminal safe
  // epilogue consumes all callback-dependent resources.
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;
  // Intrusive membership in queue->active_staging_transfer_head. The queue
  // owns one transfer resource reference while linked. Protected by the
  // queue submission mutex.
  iree_hal_amdgpu_staging_transfer_t* active_next;
  iree_hal_amdgpu_staging_transfer_t** active_prev_next;
  bool is_registered;
  // Physical-device staging pool used by this transfer.
  iree_hal_amdgpu_staging_pool_t* pool;
  // File being read or written.
  iree_hal_file_t* file;
  // Async file handle borrowed from |file|.
  iree_async_file_t* async_file;
  // Borrowed host range used by upload/download transfers.
  uint8_t* host_base;
  // User buffer being copied to or from.
  iree_hal_buffer_t* buffer;
  // File byte offset for the first requested byte.
  uint64_t file_offset;
  // User buffer byte offset for the first requested byte.
  iree_device_size_t buffer_offset;
  // Total requested transfer length.
  iree_device_size_t requested_length;
  // Number of bytes assigned to chunks.
  iree_device_size_t submitted_length;
  // Number of bytes fully transferred through all stages.
  iree_device_size_t completed_length;
  // Number of chunks currently owning a staging slot or in-flight operation.
  uint32_t active_chunk_count;
  // Number of proactor callbacks handed to queue-owned safe execution but not
  // yet fully consumed. Protected by |mutex| and joined by queue seal.
  uint32_t safe_action_count;
  // True from entry to a claimed slot callback until its final waiter-owned
  // resource reference has been consumed. While true no path may publish
  // terminal completion or synchronously cancel this same waiter.
  bool slot_waiter_callback_active;
  // Number of chunk records in |chunks|.
  uint32_t chunk_count;
  // Number of wait semaphores supplied to the queue_read/write operation.
  uint32_t profile_wait_count;
  // Direction of this transfer.
  iree_hal_amdgpu_staging_transfer_kind_t kind;
  // Transfer lifecycle flags from iree_hal_amdgpu_staging_transfer_flags_t.
  iree_hal_amdgpu_staging_transfer_flags_t flags;
  // Owned aggregate of transfer failures.
  iree_status_t failure_status;
  // True when |failure_status| is the queue shutdown reason installed by seal
  // rather than an operation-local failure.
  bool failure_status_is_shutdown;
  // Waiter queued when all staging slots are temporarily unavailable.
  iree_hal_amdgpu_staging_pool_waiter_t slot_waiter;
  // Completion-thread retry queued when the final signal barrier is blocked by
  // temporary queue capacity pressure.
  iree_hal_amdgpu_host_queue_post_drain_action_t signal_capacity_retry;
  // Queue-owned terminal epilogue and owned status transferred to it.
  iree_hal_amdgpu_host_queue_post_drain_action_t terminal_action;
  iree_status_t terminal_status;
  bool terminal_action_queued;
  // Cloned signal list published after the transfer completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Optional completion action used instead of publishing signal semaphores.
  iree_hal_amdgpu_reclaim_action_t completion_action;
  // Resource retaining |completion_action.user_data| through completion.
  iree_hal_resource_t* completion_resource;
  // Chunk records used to pipeline file I/O and GPU copies.
  iree_hal_amdgpu_staging_chunk_t* chunks;
};

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
static void iree_hal_amdgpu_staging_waiter_tail_before_wake(void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE,
      /*value0=*/0, /*value1=*/0);
}
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

static void iree_hal_amdgpu_staging_transfer_pump(
    iree_hal_amdgpu_staging_transfer_t* transfer);

static void iree_hal_amdgpu_staging_copy_post_drain(void* user_data);
static void iree_hal_amdgpu_staging_copy_capacity_post_drain(void* user_data);
static void iree_hal_amdgpu_staging_signal_capacity_post_drain(void* user_data);

static void iree_hal_amdgpu_staging_transfer_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(!transfer->lifetime_claim.queue_retained &&
                  !transfer->lifetime_claim.device_retained,
              "staging transfer destroyed with a live queue/device claim");
  IREE_ASSERT(iree_status_is_ok(transfer->terminal_status),
              "staging transfer destroyed with an unconsumed terminal status");
  IREE_ASSERT(iree_status_is_ok(transfer->failure_status),
              "staging transfer destroyed with an unconsumed failure status");
  IREE_ASSERT(!transfer->failure_status_is_shutdown,
              "staging transfer destroyed with a tagged shutdown status");
  IREE_ASSERT(transfer->safe_action_count == 0,
              "staging transfer destroyed with live safe actions");
  if (!iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                                 transfer->host_allocator);
  }
  iree_hal_buffer_release(transfer->buffer);
  iree_hal_file_release(transfer->file);
  iree_hal_resource_release(transfer->completion_resource);
  iree_notification_deinitialize(&transfer->slot_waiter.state_notification);
  iree_notification_deinitialize(&transfer->state_notification);
  iree_slim_mutex_deinitialize(&transfer->mutex);
  iree_allocator_free(transfer->host_allocator, transfer);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_staging_transfer_vtable = {
        .destroy = iree_hal_amdgpu_staging_transfer_destroy,
};

static bool iree_hal_amdgpu_staging_transfer_is_terminal(void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  return iree_atomic_load(&transfer->terminal_complete,
                          iree_memory_order_acquire) != 0;
}

static bool iree_hal_amdgpu_staging_transfer_is_terminal_or_handed_off(
    void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  if (iree_hal_amdgpu_staging_transfer_is_terminal(transfer)) return true;
  iree_slim_mutex_lock(&transfer->mutex);
  const bool is_handed_off =
      transfer->safe_action_count != 0 || transfer->terminal_action_queued;
  iree_slim_mutex_unlock(&transfer->mutex);
  return is_handed_off;
}

static bool iree_hal_amdgpu_staging_transfer_io_submit_is_idle(
    void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  iree_slim_mutex_lock(&transfer->mutex);
  const bool is_idle = transfer->io_submit_count == 0;
  iree_slim_mutex_unlock(&transfer->mutex);
  return is_idle;
}

// Registers the exact queue ownership edge before any async publisher starts.
// The queue-owned resource reference keeps |transfer| and its intrusive node
// valid until normal terminal completion or queue seal consumes the edge.
static iree_status_t iree_hal_amdgpu_staging_transfer_register(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_hal_amdgpu_host_queue_t* queue = transfer->queue;
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;
  if (!iree_hal_amdgpu_host_queue_lifetime_try_acquire(queue,
                                                       &lifetime_claim)) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_host_queue_revalidate_submission_locked(queue);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
    return status;
  }
  IREE_ASSERT(!transfer->is_registered && !transfer->active_prev_next);
  iree_hal_resource_retain(&transfer->resource);
  transfer->active_next = queue->active_staging_transfer_head;
  transfer->active_prev_next = &queue->active_staging_transfer_head;
  if (transfer->active_next) {
    transfer->active_next->active_prev_next = &transfer->active_next;
  }
  queue->active_staging_transfer_head = transfer;
  transfer->is_registered = true;
  transfer->lifetime_claim = lifetime_claim;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return iree_ok_status();
}

// Publishes terminal completion and drops the queue-owned registry edge unless
// queue seal has already detached and assumed ownership of it.
static void iree_hal_amdgpu_staging_transfer_publish_terminal(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(transfer->active_chunk_count == 0,
              "terminal staged transfer retains active chunks");
  IREE_ASSERT(transfer->io_submit_count == 0,
              "terminal staged transfer retains proactor submissions");
  IREE_ASSERT(transfer->safe_action_count == 0,
              "terminal staged transfer retains callback continuations");
  IREE_ASSERT(iree_status_is_ok(transfer->failure_status),
              "terminal staged transfer retains an unconsumed status");
  IREE_ASSERT(!transfer->failure_status_is_shutdown,
              "terminal staged transfer retains a tagged shutdown status");
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_slim_mutex_lock(&transfer->pool->mutex);
  IREE_ASSERT(transfer->slot_waiter.state ==
                  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_STATE_IDLE,
              "terminal staged transfer retains a slot waiter");
  iree_slim_mutex_unlock(&transfer->pool->mutex);

  bool release_registry_ref = false;
  iree_hal_amdgpu_host_queue_t* queue = transfer->queue;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  if (transfer->is_registered) {
    *transfer->active_prev_next = transfer->active_next;
    if (transfer->active_next) {
      transfer->active_next->active_prev_next = transfer->active_prev_next;
    }
    transfer->active_next = NULL;
    transfer->active_prev_next = NULL;
    transfer->is_registered = false;
    release_registry_ref = true;
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  const int32_t previous = iree_atomic_exchange(&transfer->terminal_complete, 1,
                                                iree_memory_order_acq_rel);
  IREE_ASSERT(previous == 0, "staging transfer completed more than once");
  iree_notification_post(&transfer->state_notification, IREE_ALL_WAITERS);
  if (release_registry_ref) {
    iree_hal_resource_release(&transfer->resource);
  }
}

static bool iree_hal_amdgpu_staging_transfer_cancel_requested(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_slim_mutex_lock(&transfer->mutex);
  const bool is_cancel_requested = iree_any_bit_set(
      transfer->flags, IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED);
  iree_slim_mutex_unlock(&transfer->mutex);
  return is_cancel_requested;
}

static void iree_hal_amdgpu_staging_transfer_record_failure(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  iree_slim_mutex_lock(&transfer->mutex);
  const bool cancel_requested = iree_any_bit_set(
      transfer->flags, IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED);
  if (cancel_requested && transfer->failure_status_is_shutdown &&
      iree_status_code(status) == IREE_STATUS_CANCELLED) {
    // Proactor cancellation reports generic CANCELLED. The queue shutdown
    // reason already installed by seal is more specific and remains owned by
    // the transfer.
    iree_status_free(status);
  } else if (cancel_requested && transfer->failure_status_is_shutdown) {
    // A non-cancellation operation error that completed before/during seal
    // remains the first operation-local failure and outranks queue closure.
    iree_status_free(transfer->failure_status);
    transfer->failure_status = status;
    transfer->failure_status_is_shutdown = false;
  } else {
    transfer->failure_status =
        iree_status_join(transfer->failure_status, status);
  }
  iree_slim_mutex_unlock(&transfer->mutex);
}

static iree_status_t iree_hal_amdgpu_staging_transfer_submit_signal_barrier(
    iree_hal_amdgpu_staging_transfer_t* transfer, bool* out_deferred) {
  *out_deferred = false;
  if (iree_hal_amdgpu_staging_transfer_cancel_requested(transfer)) {
    return iree_hal_amdgpu_host_queue_clone_shutdown_status(transfer->queue);
  }
  if (iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_clone_error_status(transfer->queue));

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PUBLISHER_SUBMISSION_REVALIDATION,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_PUBLISHER_SUBMISSION_LOCK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_STAGING_SIGNAL_BARRIER,
      /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_lock(&transfer->queue->locks.submission_mutex);
  bool ready = false;
  uint64_t submission_id = 0;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
      .type =
          iree_hal_amdgpu_staging_transfer_profile_event_type(transfer->kind),
      .payload_length = transfer->requested_length,
      .operation_count = 1,
  };
  const iree_hal_amdgpu_host_queue_profile_event_info_t*
      profile_event_info_ptr =
          profile_event_info.type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE
              ? &profile_event_info
              : NULL;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_revalidate_submission_locked(transfer->queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_submit_barrier(
        transfer->queue, &resolution, transfer->signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){0},
        /*operation_resources=*/NULL, /*operation_resource_count=*/0,
        profile_event_info_ptr,
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready,
        &submission_id);
  }
  if (iree_status_is_ok(status) && ready && profile_event_info_ptr) {
    iree_hal_amdgpu_wait_resolution_t profile_resolution = resolution;
    profile_resolution.wait_count = transfer->profile_wait_count;
    profile_event_info.submission_id = submission_id;
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        transfer->queue, &profile_resolution, transfer->signal_semaphore_list,
        &profile_event_info);
  }
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&transfer->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        transfer->queue, &transfer->signal_capacity_retry,
        iree_hal_amdgpu_staging_signal_capacity_post_drain, transfer);
    *out_deferred = true;
  }
  iree_slim_mutex_unlock(&transfer->queue->locks.submission_mutex);
  return status;
}

static void iree_hal_amdgpu_staging_transfer_fail_signals(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
}

static void iree_hal_amdgpu_staging_transfer_fail_signals_with_borrowed_status(
    iree_hal_amdgpu_staging_transfer_t* transfer, const iree_status_t status) {
  if (iree_status_is_ok(status) ||
      iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list,
                               iree_status_clone(status));
}

static void iree_hal_amdgpu_staging_transfer_finish_terminal(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim =
      transfer->lifetime_claim;
  memset(&transfer->lifetime_claim, 0, sizeof(transfer->lifetime_claim));
  iree_hal_amdgpu_staging_transfer_publish_terminal(transfer);
  iree_hal_resource_release(&transfer->resource);
  (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
}

static void iree_hal_amdgpu_staging_transfer_complete_safe(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (transfer->completion_action.fn) {
    transfer->completion_action.fn(
        /*entry=*/NULL, transfer->completion_action.user_data, status);
    iree_status_free(status);
    iree_hal_amdgpu_staging_transfer_finish_terminal(transfer);
    return;
  }
  bool deferred = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_staging_transfer_submit_signal_barrier(transfer,
                                                                    &deferred);
  }
  iree_hal_amdgpu_staging_transfer_fail_signals(transfer, status);
  if (!deferred) {
    iree_hal_amdgpu_staging_transfer_finish_terminal(transfer);
  } else {
    iree_hal_resource_release(&transfer->resource);
  }
}

static void iree_hal_amdgpu_staging_terminal_post_drain(void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(transfer->terminal_action_queued);
  transfer->terminal_action_queued = false;
  iree_status_t status = transfer->terminal_status;
  transfer->terminal_status = iree_ok_status();
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_hal_amdgpu_staging_transfer_complete_safe(transfer, status);
}

static void iree_hal_amdgpu_staging_transfer_schedule_terminal(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(!transfer->terminal_action_queued,
              "staging transfer terminal epilogue scheduled twice");
  transfer->terminal_status = status;
  transfer->terminal_action_queued = true;
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action_and_notify(
      transfer->queue, &transfer->terminal_action,
      iree_hal_amdgpu_staging_terminal_post_drain, transfer,
      &transfer->state_notification);
}

// Claims terminal completion while holding |transfer->mutex|. The caller must
// perform waiter cancellation and callback-capable completion after unlock.
static bool iree_hal_amdgpu_staging_transfer_prepare_finish_locked(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t* out_status) {
  const bool has_failure = !iree_status_is_ok(transfer->failure_status);
  const bool is_complete =
      transfer->completed_length == transfer->requested_length;
  if (transfer->slot_waiter_callback_active ||
      iree_any_bit_set(transfer->flags,
                       IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING) ||
      transfer->active_chunk_count != 0 || (!has_failure && !is_complete)) {
    return false;
  }
  transfer->flags |= IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING;
  *out_status = transfer->failure_status;
  transfer->failure_status = iree_ok_status();
  transfer->failure_status_is_shutdown = false;
  return true;
}

static void iree_hal_amdgpu_staging_transfer_complete_prepared_finish(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_hal_amdgpu_staging_pool_cancel_waiter(transfer->pool,
                                                 &transfer->slot_waiter)) {
    iree_hal_resource_release(&transfer->resource);
  }
  iree_hal_amdgpu_staging_transfer_schedule_terminal(transfer, status);
}

static void iree_hal_amdgpu_staging_signal_capacity_post_drain(
    void* user_data) {
  iree_hal_amdgpu_staging_transfer_complete_safe(
      (iree_hal_amdgpu_staging_transfer_t*)user_data, iree_ok_status());
}

static void iree_hal_amdgpu_staging_transfer_try_finish(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  bool should_complete = false;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&transfer->mutex);
  should_complete =
      iree_hal_amdgpu_staging_transfer_prepare_finish_locked(transfer, &status);
  iree_slim_mutex_unlock(&transfer->mutex);

  if (should_complete) {
    iree_hal_amdgpu_staging_transfer_complete_prepared_finish(transfer, status);
  }
}

static void iree_hal_amdgpu_staging_chunk_finish(
    iree_hal_amdgpu_staging_chunk_t* chunk, bool did_transfer_bytes) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(chunk->state != IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE);
  IREE_ASSERT(chunk->slot.buffer,
              "active staging chunk must own an exact pool slot");
  const uint32_t slot_ordinal = chunk->slot.ordinal;
  if (did_transfer_bytes) {
    transfer->completed_length += chunk->length;
  }
  memset(&chunk->slot, 0, sizeof(chunk->slot));
  chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE;
  chunk->length = 0;
  chunk->file_progress = 0;
  --transfer->active_chunk_count;
  iree_slim_mutex_unlock(&transfer->mutex);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (iree_hal_amdgpu_staging_chunk_slot_release_observer) {
    iree_hal_amdgpu_staging_chunk_slot_release_observer(
        iree_hal_amdgpu_staging_chunk_slot_release_observer_user_data,
        slot_ordinal);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_hal_amdgpu_staging_pool_release(transfer->pool, slot_ordinal);
  iree_hal_amdgpu_staging_transfer_pump(transfer);
  iree_hal_amdgpu_staging_transfer_try_finish(transfer);
}

static void iree_hal_amdgpu_staging_chunk_fail(
    iree_hal_amdgpu_staging_chunk_t* chunk, iree_status_t status) {
  iree_hal_amdgpu_staging_transfer_record_failure(chunk->transfer, status);
  iree_hal_amdgpu_staging_chunk_finish(chunk, /*did_transfer_bytes=*/false);
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_read(
    iree_hal_amdgpu_staging_chunk_t* chunk);

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_write(
    iree_hal_amdgpu_staging_chunk_t* chunk);

static void iree_hal_amdgpu_staging_copy_pre_signal(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  chunk->copy_status =
      iree_status_is_ok(status) ? iree_ok_status() : iree_status_clone(status);
  iree_hal_resource_retain(&chunk->transfer->resource);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      chunk->transfer->queue, &chunk->post_drain_action,
      iree_hal_amdgpu_staging_copy_post_drain, chunk);
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_copy(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_clone_error_status(transfer->queue));

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;

  iree_hal_buffer_t* source_buffer = NULL;
  iree_device_size_t source_offset = 0;
  iree_hal_buffer_t* target_buffer = NULL;
  iree_device_size_t target_offset = 0;
  iree_hsa_fence_scope_t minimum_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t minimum_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hal_amdgpu_staging_chunk_state_t next_state;
  if (iree_hal_amdgpu_staging_transfer_copies_to_device(transfer->kind)) {
    source_buffer = chunk->slot.buffer;
    source_offset = chunk->slot.buffer_offset;
    target_buffer = transfer->buffer;
    target_offset = transfer->buffer_offset + chunk->transfer_offset;
    minimum_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
    next_state = IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_DEVICE;
  } else {
    source_buffer = transfer->buffer;
    source_offset = transfer->buffer_offset + chunk->transfer_offset;
    target_buffer = chunk->slot.buffer;
    target_offset = chunk->slot.buffer_offset;
    minimum_release_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
    next_state = IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_HOST;
  }

  iree_slim_mutex_lock(&transfer->mutex);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED)) {
    iree_slim_mutex_unlock(&transfer->mutex);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  chunk->state = next_state;
  iree_slim_mutex_unlock(&transfer->mutex);

  iree_hal_resource_t* extra_resources[1] = {&transfer->resource};
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PUBLISHER_SUBMISSION_REVALIDATION,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_PUBLISHER_SUBMISSION_LOCK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_STAGING_COPY,
      /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_lock(&transfer->queue->locks.submission_mutex);
  bool ready = false;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_revalidate_submission_locked(transfer->queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_copy_with_action(
        transfer->queue, &resolution, iree_hal_semaphore_list_empty(),
        source_buffer, source_offset, target_buffer, target_offset,
        chunk->length, IREE_HAL_COPY_FLAG_NONE, minimum_acquire_scope,
        minimum_release_scope,
        (iree_hal_amdgpu_reclaim_action_t){
            .fn = iree_hal_amdgpu_staging_copy_pre_signal,
            .user_data = chunk,
        },
        extra_resources, IREE_ARRAYSIZE(extra_resources),
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready);
  }
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&transfer->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        transfer->queue, &chunk->post_drain_action,
        iree_hal_amdgpu_staging_copy_capacity_post_drain, chunk);
  }
  iree_slim_mutex_unlock(&transfer->queue->locks.submission_mutex);
  return status;
}

// Leaves one queue-owned safe continuation after every callback-dependent
// transfer touch and the callback-held resource release have completed. The
// zero predicate and wake are one transfer-mutex transaction; after the final
// unlock this continuation must not touch |transfer| or |queue| again.
static void iree_hal_amdgpu_staging_safe_action_leave(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  // The transfer's queue registry/self references and lifetime claim keep the
  // transfer live while this callback-owned reference is consumed.
  iree_hal_resource_release(&transfer->resource);

  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(transfer->safe_action_count > 0);
  --transfer->safe_action_count;
  const bool is_idle = transfer->safe_action_count == 0;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        transfer->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE,
        /*value0=*/0, /*value1=*/0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_notification_post(&transfer->state_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        transfer->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT,
        /*value0=*/0, /*value1=*/0);
  }
#else
  (void)is_idle;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&transfer->mutex);
}

static void iree_hal_amdgpu_staging_read_safe(void* user_data) {
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_BEGIN,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_READ_CALLBACK);
  iree_status_t status = chunk->file_callback_status;
  chunk->file_callback_status = iree_ok_status();
  const bool cancel_requested = iree_any_bit_set(
      transfer->flags, IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED);
  iree_slim_mutex_unlock(&transfer->mutex);
  if (iree_status_is_ok(status) && cancel_requested) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  if (iree_status_is_ok(status) && chunk->read_op.bytes_read > 0) {
    chunk->file_progress += chunk->read_op.bytes_read;
    if (chunk->file_progress < chunk->length) {
      status = iree_hal_amdgpu_staging_chunk_submit_read(chunk);
      if (iree_status_is_ok(status)) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
        iree_hal_amdgpu_host_queue_test_notify_phase(
            transfer->queue,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_END,
            /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
        iree_hal_amdgpu_staging_safe_action_leave(transfer);
        return;
      }
    }
  } else if (iree_status_is_ok(status) &&
             chunk->file_progress < chunk->length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short read: requested %" PRIhsz
                              " bytes, got %" PRIhsz,
                              chunk->length, chunk->file_progress);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_staging_chunk_submit_copy(chunk);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  }
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_END,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_hal_amdgpu_staging_safe_action_leave(transfer);
}

static void iree_hal_amdgpu_staging_write_safe(void* user_data) {
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_BEGIN,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_WRITE_CALLBACK);
  iree_status_t status = chunk->file_callback_status;
  chunk->file_callback_status = iree_ok_status();
  const bool cancel_requested = iree_any_bit_set(
      transfer->flags, IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED);
  iree_slim_mutex_unlock(&transfer->mutex);
  if (iree_status_is_ok(status) && cancel_requested) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  if (iree_status_is_ok(status) && chunk->write_op.bytes_written > 0) {
    chunk->file_progress += chunk->write_op.bytes_written;
    if (chunk->file_progress < chunk->length) {
      status = iree_hal_amdgpu_staging_chunk_submit_write(chunk);
      if (iree_status_is_ok(status)) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
        iree_hal_amdgpu_host_queue_test_notify_phase(
            transfer->queue,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_END,
            /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
        iree_hal_amdgpu_staging_safe_action_leave(transfer);
        return;
      }
    }
  } else if (iree_status_is_ok(status) &&
             chunk->file_progress < chunk->length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short write: requested %" PRIhsz
                              " bytes, wrote %" PRIhsz,
                              chunk->length, chunk->file_progress);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  } else {
    iree_hal_amdgpu_staging_chunk_finish(chunk, /*did_transfer_bytes=*/true);
  }
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_END,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_hal_amdgpu_staging_safe_action_leave(transfer);
}

static void iree_hal_amdgpu_staging_read_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;

  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_READING);
  chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_READ_CALLBACK;
  IREE_ASSERT(iree_status_is_ok(chunk->file_callback_status));
  chunk->file_callback_status = status;
  ++transfer->safe_action_count;
  iree_slim_mutex_unlock(&transfer->mutex);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_HANDOFF_INSTALLED,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  // This enqueue unlock is the proactor callback's final queue/transfer touch.
  // The generic async callback contract permits operation storage to be
  // released from the callback; the queue continuation owns the callback ref.
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action_and_notify(
      transfer->queue, &chunk->post_drain_action,
      iree_hal_amdgpu_staging_read_safe, chunk, &transfer->state_notification);
}

static void iree_hal_amdgpu_staging_write_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;

  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_WRITING);
  chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_WRITE_CALLBACK;
  IREE_ASSERT(iree_status_is_ok(chunk->file_callback_status));
  chunk->file_callback_status = status;
  ++transfer->safe_action_count;
  iree_slim_mutex_unlock(&transfer->mutex);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_HANDOFF_INSTALLED,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  // See the read callback above: enqueue is the final callback-side touch and
  // transfers the callback-owned reference to queue-owned safe execution.
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action_and_notify(
      transfer->queue, &chunk->post_drain_action,
      iree_hal_amdgpu_staging_write_safe, chunk, &transfer->state_notification);
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_read(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_clone_error_status(transfer->queue));

  iree_slim_mutex_lock(&transfer->mutex);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED)) {
    iree_slim_mutex_unlock(&transfer->mutex);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  ++transfer->io_submit_count;
  chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_READING;

  iree_async_operation_zero(&chunk->read_op.base, sizeof(chunk->read_op));
  iree_async_operation_initialize(&chunk->read_op.base,
                                  IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_hal_amdgpu_staging_read_complete, chunk);
  chunk->read_op.file = transfer->async_file;
  chunk->read_op.offset =
      transfer->file_offset + chunk->transfer_offset + chunk->file_progress;
  chunk->read_op.buffer = iree_async_span_from_ptr(
      chunk->slot.host_span.data + chunk->file_progress,
      chunk->length - chunk->file_progress);
  iree_hal_resource_retain(&transfer->resource);
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_status_t status = iree_async_proactor_submit_one(
      transfer->queue->proactor, &chunk->read_op.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&transfer->resource);
  }
  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(transfer->io_submit_count > 0);
  --transfer->io_submit_count;
  if (!iree_status_is_ok(status)) {
    IREE_ASSERT(chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_READING);
    chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_READ_CALLBACK;
  }
  const bool submit_tail_is_idle = transfer->io_submit_count == 0;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        transfer->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_READ_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_ZERO_BEFORE_WAKE,
        /*value0=*/0, /*value1=*/0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  // The stable zero predicate and wake are one transfer-mutex transaction.
  iree_notification_post(&transfer->state_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        transfer->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_READ_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_LEFT,
        /*value0=*/0, /*value1=*/0);
  }
#else
  (void)submit_tail_is_idle;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&transfer->mutex);
  return status;
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_write(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_clone_error_status(transfer->queue));

  iree_slim_mutex_lock(&transfer->mutex);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED)) {
    iree_slim_mutex_unlock(&transfer->mutex);
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  ++transfer->io_submit_count;
  chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_WRITING;

  iree_async_operation_zero(&chunk->write_op.base, sizeof(chunk->write_op));
  iree_async_operation_initialize(
      &chunk->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amdgpu_staging_write_complete,
      chunk);
  chunk->write_op.file = transfer->async_file;
  chunk->write_op.offset =
      transfer->file_offset + chunk->transfer_offset + chunk->file_progress;
  chunk->write_op.buffer = iree_async_span_from_ptr(
      chunk->slot.host_span.data + chunk->file_progress,
      chunk->length - chunk->file_progress);
  iree_hal_resource_retain(&transfer->resource);
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_status_t status = iree_async_proactor_submit_one(
      transfer->queue->proactor, &chunk->write_op.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&transfer->resource);
  }
  iree_slim_mutex_lock(&transfer->mutex);
  IREE_ASSERT(transfer->io_submit_count > 0);
  --transfer->io_submit_count;
  if (!iree_status_is_ok(status)) {
    IREE_ASSERT(chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_WRITING);
    chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_WRITE_CALLBACK;
  }
  const bool submit_tail_is_idle = transfer->io_submit_count == 0;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        transfer->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_WRITE_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_ZERO_BEFORE_WAKE,
        /*value0=*/0, /*value1=*/0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_notification_post(&transfer->state_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        transfer->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_WRITE_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_LEFT,
        /*value0=*/0, /*value1=*/0);
  }
#else
  (void)submit_tail_is_idle;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&transfer->mutex);
  return status;
}

static void iree_hal_amdgpu_staging_copy_post_drain(void* user_data) {
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_status_t status = chunk->copy_status;
  chunk->copy_status = iree_ok_status();

  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_staging_transfer_cancel_requested(chunk->transfer)) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  if (iree_status_is_ok(status) &&
      chunk->transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_WRITE) {
    status = iree_hal_amdgpu_staging_chunk_submit_write(chunk);
    if (iree_status_is_ok(status)) {
      iree_hal_resource_release(&chunk->transfer->resource);
      return;
    }
  } else if (iree_status_is_ok(status) &&
             chunk->transfer->kind ==
                 IREE_HAL_AMDGPU_STAGING_TRANSFER_DOWNLOAD) {
    memcpy(chunk->transfer->host_base + chunk->transfer_offset,
           chunk->slot.host_span.data, chunk->length);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  } else {
    iree_hal_amdgpu_staging_chunk_finish(chunk, /*did_transfer_bytes=*/true);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static void iree_hal_amdgpu_staging_copy_capacity_post_drain(void* user_data) {
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_status_t status = iree_hal_amdgpu_staging_chunk_submit_copy(chunk);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static void iree_hal_amdgpu_staging_transfer_slot_available(void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  iree_slim_mutex_lock(&transfer->mutex);
  transfer->slot_waiter_callback_active = true;
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_hal_amdgpu_staging_transfer_pump(transfer);
}

static void iree_hal_amdgpu_staging_transfer_slot_callback_finished(
    void* user_data, bool may_finish) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;

  // Consume the reference associated with this callback while the callback
  // tail gate still blocks terminal publication. The transfer-owned self
  // reference remains until complete() publishes terminal state.
  iree_hal_resource_release(&transfer->resource);

  bool should_complete = false;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&transfer->mutex);
  if (may_finish) {
    IREE_ASSERT(transfer->slot_waiter_callback_active);
    transfer->slot_waiter_callback_active = false;
    should_complete = iree_hal_amdgpu_staging_transfer_prepare_finish_locked(
        transfer, &status);
  }
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      transfer->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT, may_finish ? 1 : 0,
      should_complete ? 1 : 0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&transfer->mutex);

  if (should_complete) {
    iree_hal_amdgpu_staging_transfer_complete_prepared_finish(transfer, status);
  }
}

static iree_hal_amdgpu_staging_chunk_t*
iree_hal_amdgpu_staging_transfer_find_idle_chunk(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  for (uint32_t i = 0; i < transfer->chunk_count; ++i) {
    if (transfer->chunks[i].state == IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE) {
      return &transfer->chunks[i];
    }
  }
  return NULL;
}

static void iree_hal_amdgpu_staging_transfer_pump(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  for (;;) {
    iree_hal_amdgpu_staging_slot_t slot;
    memset(&slot, 0, sizeof(slot));
    const bool has_waiter_slot = iree_hal_amdgpu_staging_pool_take_waiter_slot(
        transfer->pool, &transfer->slot_waiter, &slot);

    iree_slim_mutex_lock(&transfer->mutex);
    const bool can_submit_more =
        !iree_any_bit_set(
            transfer->flags,
            IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING |
                IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED) &&
        iree_status_is_ok(transfer->failure_status) &&
        transfer->submitted_length < transfer->requested_length;
    iree_slim_mutex_unlock(&transfer->mutex);
    if (!can_submit_more) {
      if (has_waiter_slot) {
        iree_hal_amdgpu_staging_pool_release(transfer->pool, slot.ordinal);
      }
      return;
    }

    if (!has_waiter_slot &&
        !iree_hal_amdgpu_staging_pool_try_acquire(transfer->pool, &slot)) {
      // Retain before publishing the waiter: a concurrent slot release may
      // dequeue and invoke the callback immediately after the pool unlocks.
      iree_hal_resource_retain(&transfer->resource);
      iree_hal_amdgpu_staging_pool_wait_result_t wait_result =
          iree_hal_amdgpu_staging_pool_queue_waiter(
              transfer->pool, &transfer->slot_waiter,
              iree_hal_amdgpu_staging_transfer_slot_available,
              iree_hal_amdgpu_staging_transfer_slot_callback_finished,
              transfer);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      if (wait_result == IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED) {
        // The pool helper has released pool->mutex and transfer->mutex is not
        // held. The observer must remain nonblocking and must not reenter the
        // queue. value0/value1 identify the exact transfer and shared pool.
        iree_hal_amdgpu_host_queue_test_notify_phase(
            transfer->queue,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_STAGING_WAITER_QUEUED,
            (uint64_t)(uintptr_t)transfer, (uint64_t)(uintptr_t)transfer->pool);
      }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
      if (wait_result != IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED) {
        iree_hal_resource_release(&transfer->resource);
      }
      if (wait_result != IREE_HAL_AMDGPU_STAGING_POOL_WAIT_RETRY) {
        return;
      }
      continue;
    }

    iree_hal_amdgpu_staging_chunk_t* chunk = NULL;
    iree_device_size_t chunk_offset = 0;
    iree_host_size_t chunk_length = 0;
    bool should_release_slot = false;
    iree_slim_mutex_lock(&transfer->mutex);
    const bool has_failure = !iree_status_is_ok(transfer->failure_status);
    const bool has_more_bytes =
        transfer->submitted_length < transfer->requested_length;
    if (!iree_any_bit_set(
            transfer->flags,
            IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING |
                IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED) &&
        !has_failure && has_more_bytes) {
      chunk = iree_hal_amdgpu_staging_transfer_find_idle_chunk(transfer);
    }
    if (chunk) {
      const iree_device_size_t remaining_length =
          transfer->requested_length - transfer->submitted_length;
      chunk_length = (iree_host_size_t)iree_min(
          (iree_device_size_t)transfer->pool->slot_size, remaining_length);
      chunk_offset = transfer->submitted_length;
      if (transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ) {
        chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_READING;
      } else if (iree_hal_amdgpu_staging_transfer_copies_to_device(
                     transfer->kind)) {
        chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_DEVICE;
      } else {
        chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_HOST;
      }
      chunk->slot = slot;
      chunk->transfer_offset = chunk_offset;
      chunk->length = chunk_length;
      chunk->file_progress = 0;
      chunk->copy_status = iree_ok_status();
      transfer->submitted_length += chunk_length;
      ++transfer->active_chunk_count;
    } else {
      should_release_slot = true;
    }
    iree_slim_mutex_unlock(&transfer->mutex);

    if (should_release_slot) {
      iree_hal_amdgpu_staging_pool_release(transfer->pool, slot.ordinal);
      return;
    }

    iree_status_t status = iree_ok_status();
    if (transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ) {
      status = iree_hal_amdgpu_staging_chunk_submit_read(chunk);
    } else {
      if (transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_UPLOAD) {
        memcpy(chunk->slot.host_span.data,
               transfer->host_base + chunk->transfer_offset, chunk->length);
      }
      status = iree_hal_amdgpu_staging_chunk_submit_copy(chunk);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amdgpu_staging_chunk_fail(chunk, status);
      return;
    }
  }
}

iree_status_t iree_hal_amdgpu_staging_transfer_start(
    iree_hal_amdgpu_staging_transfer_t* transfer,
    iree_hal_amdgpu_reclaim_action_t completion_action,
    iree_hal_resource_t* completion_resource) {
  transfer->completion_action = completion_action;
  transfer->completion_resource = completion_resource;
  iree_hal_resource_retain(completion_resource);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_clone_error_status(transfer->queue));
  // The transfer buffer may be a queue_alloca result whose backing is only
  // staged when the operation is submitted. Validate the device pointer here,
  // after the queue wait set has been satisfied.
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(transfer->buffer);
  if (IREE_UNLIKELY(!iree_hal_amdgpu_buffer_device_pointer(allocated_buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "staged AMDGPU transfer buffer was not backed by an AMDGPU "
        "allocation after queue waits completed");
  }
  // The caller owns |transfer| only through this call. Keep a transfer-owned
  // self reference across the asynchronous host/GPU copy pipeline; the
  // terminal completion path releases it.
  iree_hal_resource_retain(&transfer->resource);
  iree_status_t register_status =
      iree_hal_amdgpu_staging_transfer_register(transfer);
  if (!iree_status_is_ok(register_status)) {
    iree_hal_resource_release(&transfer->resource);
    return register_status;
  }
  iree_hal_amdgpu_staging_transfer_pump(transfer);
  iree_hal_amdgpu_staging_transfer_try_finish(transfer);
  return iree_ok_status();
}

static void iree_hal_amdgpu_staging_transfer_execute(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_transfer_fail_signals_with_borrowed_status(transfer,
                                                                       status);
    return;
  }

  iree_status_t start_status = iree_hal_amdgpu_staging_transfer_start(
      transfer, (iree_hal_amdgpu_reclaim_action_t){0},
      /*completion_resource=*/NULL);
  if (!iree_status_is_ok(start_status)) {
    iree_hal_amdgpu_staging_transfer_fail_signals(transfer, start_status);
  }
}

static iree_status_t iree_hal_amdgpu_staging_transfer_validate_buffer(
    iree_hal_amdgpu_staging_transfer_kind_t kind, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(buffer, buffer_offset, length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer),
      iree_hal_amdgpu_staging_transfer_copies_to_device(kind)
          ? IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET
          : IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer),
      iree_hal_amdgpu_staging_transfer_copies_to_device(kind)
          ? IREE_HAL_MEMORY_ACCESS_WRITE
          : IREE_HAL_MEMORY_ACCESS_READ));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_staging_transfer_create(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_staging_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, void* host_base, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t profile_wait_count,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_staging_transfer_t** out_transfer) {
  if (IREE_UNLIKELY(!queue->staging_pool || !queue->staging_pool->buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU queue staging pool is not initialized");
  }
  if (iree_hal_amdgpu_staging_transfer_uses_file(kind) &&
      IREE_UNLIKELY(!file || !iree_hal_file_async_handle(file))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU staged queue file transfers require a proactor-backed async "
        "file handle");
  }
  if (!iree_hal_amdgpu_staging_transfer_uses_file(kind) &&
      IREE_UNLIKELY(!host_base)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU staged host transfer range is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_staging_transfer_validate_buffer(
      kind, buffer, buffer_offset, length));

  iree_host_size_t chunks_size = 0;
  if (!iree_host_size_checked_mul(queue->staging_pool->slot_count,
                                  sizeof(iree_hal_amdgpu_staging_chunk_t),
                                  &chunks_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staging transfer chunk table size overflows");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_amdgpu_staging_transfer_t),
                                  chunks_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staging transfer allocation size overflows");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, queue->staging_pool->slot_count);
  iree_hal_amdgpu_staging_transfer_t* transfer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, total_size,
                                (void**)&transfer));
  memset(transfer, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_staging_transfer_vtable,
                               &transfer->resource);
  transfer->host_allocator = queue->host_allocator;
  iree_slim_mutex_initialize(&transfer->mutex);
  iree_notification_initialize(&transfer->state_notification);
  iree_notification_initialize(&transfer->slot_waiter.state_notification);
  iree_atomic_store(&transfer->terminal_complete, 0, iree_memory_order_relaxed);
  transfer->queue = queue;
  transfer->pool = queue->staging_pool;
  transfer->file = file;
  iree_hal_file_retain(file);
  transfer->async_file = file ? iree_hal_file_async_handle(file) : NULL;
  transfer->host_base = (uint8_t*)host_base;
  transfer->buffer = buffer;
  iree_hal_buffer_retain(transfer->buffer);
  transfer->file_offset = file_offset;
  transfer->buffer_offset = buffer_offset;
  transfer->requested_length = length;
  transfer->chunk_count = queue->staging_pool->slot_count;
  transfer->profile_wait_count = profile_wait_count;
  transfer->kind = kind;
  transfer->chunks = (iree_hal_amdgpu_staging_chunk_t*)(transfer + 1);
  for (uint32_t i = 0; i < transfer->chunk_count; ++i) {
    transfer->chunks[i].transfer = transfer;
  }

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, transfer->host_allocator,
      &transfer->signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_transfer = transfer;
  } else {
    iree_hal_resource_release(&transfer->resource);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_staged_transfer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_staging_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  iree_hal_amdgpu_staging_transfer_t* transfer = NULL;
  const uint32_t profile_wait_count =
      iree_hal_amdgpu_host_queue_profile_semaphore_count(wait_semaphore_list);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_staging_transfer_create(
      queue, kind, file, file_offset, /*host_base=*/NULL, buffer, buffer_offset,
      length, profile_wait_count, signal_semaphore_list, &transfer));

  iree_hal_resource_t* resources[1] = {&transfer->resource};
  iree_status_t status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_staging_transfer_execute,
          .user_data = transfer,
      },
      resources, IREE_ARRAYSIZE(resources));
  iree_hal_resource_release(&transfer->resource);
  return status;
}

iree_status_t iree_hal_amdgpu_staging_transfer_create_upload(
    iree_hal_amdgpu_host_queue_t* queue, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length,
    iree_hal_amdgpu_staging_transfer_t** out_transfer) {
  iree_hal_amdgpu_staging_transfer_t* transfer = NULL;
  iree_status_t status = iree_hal_amdgpu_staging_transfer_create(
      queue, IREE_HAL_AMDGPU_STAGING_TRANSFER_UPLOAD, /*file=*/NULL,
      /*file_offset=*/0, (void*)source, target_buffer, target_offset, length,
      /*profile_wait_count=*/0, iree_hal_semaphore_list_empty(), &transfer);
  if (iree_status_is_ok(status)) {
    *out_transfer = transfer;
  }
  return status;
}

iree_status_t iree_hal_amdgpu_staging_transfer_create_download(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, void* target, iree_device_size_t length,
    iree_hal_amdgpu_staging_transfer_t** out_transfer) {
  iree_hal_amdgpu_staging_transfer_t* transfer = NULL;
  iree_status_t status = iree_hal_amdgpu_staging_transfer_create(
      queue, IREE_HAL_AMDGPU_STAGING_TRANSFER_DOWNLOAD, /*file=*/NULL,
      /*file_offset=*/0, target, source_buffer, source_offset, length,
      /*profile_wait_count=*/0, iree_hal_semaphore_list_empty(), &transfer);
  if (iree_status_is_ok(status)) {
    *out_transfer = transfer;
  }
  return status;
}

void iree_hal_amdgpu_staging_transfer_release(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_hal_resource_release((iree_hal_resource_t*)transfer);
}

static void iree_hal_amdgpu_staging_transfer_request_cancel(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_slim_mutex_lock(&transfer->mutex);
  const bool already_finishing = iree_any_bit_set(
      transfer->flags, IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING);
  transfer->flags |= IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_CANCEL_REQUESTED;
  if (!already_finishing && iree_status_is_ok(transfer->failure_status)) {
    transfer->failure_status =
        iree_hal_amdgpu_host_queue_clone_shutdown_status(transfer->queue);
    transfer->failure_status_is_shutdown = true;
  }
  iree_slim_mutex_unlock(&transfer->mutex);

  // A submitter that passed the cancellation check owns the operation storage
  // until submit returns. Wait without holding transfer/queue locks so
  // callbacks and submitters can make progress.
  iree_notification_await(&transfer->state_notification,
                          iree_hal_amdgpu_staging_transfer_io_submit_is_idle,
                          transfer, iree_infinite_timeout());

  // File cancellation is optional. Backends that advertise it return an
  // allocation-free status on this path; unsupported threaded backends are
  // joined through natural completion instead.
  const bool can_cancel_file_ops = iree_any_bit_set(
      iree_async_proactor_query_capabilities(transfer->queue->proactor),
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  if (can_cancel_file_ops) {
    for (uint32_t i = 0; i < transfer->chunk_count; ++i) {
      iree_hal_amdgpu_staging_chunk_t* chunk = &transfer->chunks[i];
      iree_async_operation_t* operation = NULL;
      iree_slim_mutex_lock(&transfer->mutex);
      if (chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_READING) {
        operation = &chunk->read_op.base;
      } else if (chunk->state == IREE_HAL_AMDGPU_STAGING_CHUNK_WRITING) {
        operation = &chunk->write_op.base;
      }
      iree_slim_mutex_unlock(&transfer->mutex);
      if (operation) {
        iree_status_ignore(
            iree_async_proactor_cancel(transfer->queue->proactor, operation));
      }
    }
  }

  if (iree_hal_amdgpu_staging_pool_cancel_waiter(transfer->pool,
                                                 &transfer->slot_waiter)) {
    iree_hal_resource_release(&transfer->resource);
  }
  iree_hal_amdgpu_staging_transfer_try_finish(transfer);
}

void iree_hal_amdgpu_staging_transfer_cancel_all(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);

  // Atomically detach the exact active set after queue admission has closed.
  // Each detached node still owns the registry reference installed at start,
  // so callbacks may race without invalidating this traversal.
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->is_shutting_down,
              "queue admission must close before publisher cancellation");
  iree_hal_amdgpu_staging_transfer_t* transfer =
      queue->active_staging_transfer_head;
  queue->active_staging_transfer_head = NULL;
  IREE_ASSERT(queue->shutdown_staging_transfer_head == NULL,
              "publisher cancellation can begin only once");
  queue->shutdown_staging_transfer_head = transfer;
  for (iree_hal_amdgpu_staging_transfer_t* current = transfer; current;
       current = current->active_next) {
    current->is_registered = false;
    current->active_prev_next = NULL;
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  // Request cancellation for the whole detached set before waiting for any
  // one transfer. This lets independent proactor callbacks drain in parallel.
  for (iree_hal_amdgpu_staging_transfer_t* current = transfer; current;
       current = current->active_next) {
    iree_hal_amdgpu_staging_transfer_request_cancel(current);
  }
}

void iree_hal_amdgpu_staging_transfer_await_all(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_hal_amdgpu_staging_transfer_t* transfer =
      queue->shutdown_staging_transfer_head;
  while (transfer) {
    iree_hal_amdgpu_staging_transfer_t* next = transfer->active_next;
    transfer->active_next = NULL;
    while (!iree_hal_amdgpu_staging_transfer_is_terminal(transfer)) {
      iree_notification_await(
          &transfer->state_notification,
          iree_hal_amdgpu_staging_transfer_is_terminal_or_handed_off, transfer,
          iree_infinite_timeout());
      if (!iree_hal_amdgpu_staging_transfer_is_terminal(transfer)) {
        // The completion service is already joined on this seal path. Consume
        // callback and terminal continuations from the external sealer while
        // normal completion-runner admission remains open.
        iree_hal_amdgpu_host_queue_drain_completions(queue);
      }
    }
    // Consume the queue-owned registry edge detached above.
    iree_hal_resource_release(&transfer->resource);
    transfer = next;
  }
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  queue->shutdown_staging_transfer_head = NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
}

iree_status_t iree_hal_amdgpu_host_queue_submit_staged_read(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  return iree_hal_amdgpu_host_queue_submit_staged_transfer(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_READ, source_file, source_offset,
      target_buffer, target_offset, length);
}

iree_status_t iree_hal_amdgpu_host_queue_submit_staged_write(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length) {
  return iree_hal_amdgpu_host_queue_submit_staged_transfer(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_STAGING_TRANSFER_FILE_WRITE, target_file, target_offset,
      source_buffer, source_offset, length);
}
