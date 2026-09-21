// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "common/internal.h"
#include "common/stream.h"
#include "common/tls.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/call_once.h"

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Stable OS-thread identity used for capture and per-thread-stream ownership.
// Context state itself is stored through the binding TLS key below; on Windows
// that makes it coherently fiber-local with its FLS destructor.
static IREE_THREAD_LOCAL int iree_hal_streaming_thread_token_storage;

// Counts owning context references held in every TLS current slot and push/pop
// stack, plus short-lived teardown sentinels. Global teardown uses this to
// reject while another execution context could still release an old-generation
// context through a retired registry.
static iree_atomic_int32_t iree_hal_streaming_tls_context_reference_count =
    IREE_ATOMIC_VAR_INIT(0);

typedef struct iree_hal_streaming_context_stack_t {
  iree_hal_streaming_context_t** contexts;
  iree_host_size_t depth;
  iree_host_size_t capacity;
  iree_allocator_t allocator;
} iree_hal_streaming_context_stack_t;

// Complete context state associated with one binding TLS value. Keeping the
// current slot, stack, and reference ledger in the same object is required on
// Windows, where FLS destructors can run on fiber deletion as well as thread
// exit. A destructor must never act on unrelated compiler thread-local state.
typedef struct iree_hal_streaming_context_tls_state_t {
  iree_hal_streaming_context_t* current_context;
  iree_hal_streaming_context_stack_t stack;
  int32_t context_reference_count;
} iree_hal_streaming_context_tls_state_t;

#if !defined(IREE_PLATFORM_WINDOWS) || IREE_SYNCHRONIZATION_DISABLE_UNSAFE
// pthread TLS is used only as the exit callback trigger. Ordinary context
// lookup remains a single compiler-TLS load on platforms where a thread and
// the destructor execution context have the same lifetime.
static IREE_THREAD_LOCAL iree_hal_streaming_context_tls_state_t
    iree_hal_streaming_context_local_state;
#endif  // !IREE_PLATFORM_WINDOWS || IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// A non-NULL state in this key arranges to release every current/stack context
// reference when its execution context exits. Without this cleanup a
// short-lived submission thread or fiber could permanently pin an old runtime
// generation and make process deinitialization either unsafe or impossible.
static iree_once_flag iree_hal_streaming_context_tls_key_mutex_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hal_streaming_context_tls_key_mutex;
static iree_hal_streaming_tls_key_t iree_hal_streaming_context_tls_key =
    IREE_HAL_STREAMING_TLS_KEY_INVALID;

static void iree_hal_streaming_context_tls_state_cleanup_detached(
    iree_hal_streaming_context_tls_state_t* state);

static void iree_hal_streaming_context_tls_destroy(void* value) {
  iree_hal_streaming_context_tls_state_cleanup_detached(
      (iree_hal_streaming_context_tls_state_t*)value);
}

static void iree_hal_streaming_context_tls_key_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hal_streaming_context_tls_key_mutex);
}

iree_status_t iree_hal_streaming_context_tls_initialize(void) {
  iree_call_once(&iree_hal_streaming_context_tls_key_mutex_once,
                 iree_hal_streaming_context_tls_key_mutex_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_context_tls_key_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_streaming_context_tls_key ==
      IREE_HAL_STREAMING_TLS_KEY_INVALID) {
    status = iree_hal_streaming_tls_key_create(
        &iree_hal_streaming_context_tls_key,
        iree_hal_streaming_context_tls_destroy);
  }
  iree_slim_mutex_unlock(&iree_hal_streaming_context_tls_key_mutex);
  return status;
}

iree_status_t iree_hal_streaming_context_tls_deinitialize(void) {
  iree_call_once(&iree_hal_streaming_context_tls_key_mutex_once,
                 iree_hal_streaming_context_tls_key_mutex_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_context_tls_key_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_streaming_context_tls_key !=
      IREE_HAL_STREAMING_TLS_KEY_INVALID) {
    IREE_ASSERT(iree_hal_streaming_context_tls_reference_count() == 0);
    status =
        iree_hal_streaming_tls_key_delete(iree_hal_streaming_context_tls_key);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_context_tls_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
    }
  }
  iree_slim_mutex_unlock(&iree_hal_streaming_context_tls_key_mutex);
  return status;
}

static iree_hal_streaming_context_tls_state_t*
iree_hal_streaming_context_tls_state(void) {
#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  if (iree_hal_streaming_context_tls_key ==
      IREE_HAL_STREAMING_TLS_KEY_INVALID) {
    return NULL;
  }
  return (iree_hal_streaming_context_tls_state_t*)iree_hal_streaming_tls_get(
      iree_hal_streaming_context_tls_key);
#else
  return iree_hal_streaming_context_local_state.context_reference_count > 0
             ? &iree_hal_streaming_context_local_state
             : NULL;
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
}

// Returns the installed state, or an uninstalled empty state the caller must
// either commit with reference_add or free with state_discard_empty.
static iree_status_t iree_hal_streaming_context_tls_state_prepare(
    iree_hal_streaming_context_tls_state_t** out_state,
    bool* out_state_is_new) {
  IREE_ASSERT_ARGUMENT(out_state);
  IREE_ASSERT_ARGUMENT(out_state_is_new);
  *out_state = iree_hal_streaming_context_tls_state();
  *out_state_is_new = *out_state == NULL;
  if (!*out_state_is_new) return iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(**out_state), (void**)out_state));
  memset(*out_state, 0, sizeof(**out_state));
#else
  *out_state = &iree_hal_streaming_context_local_state;
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  return iree_ok_status();
}

static void iree_hal_streaming_context_tls_state_discard_empty(
    iree_hal_streaming_context_tls_state_t* state) {
  IREE_ASSERT(state->current_context == NULL);
  IREE_ASSERT(state->stack.depth == 0);
  IREE_ASSERT(state->context_reference_count == 0);
  iree_allocator_free(state->stack.allocator, state->stack.contexts);
#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  iree_allocator_free(iree_allocator_system(), state);
#else
  *state = (iree_hal_streaming_context_tls_state_t){0};
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
}

// Releases only the backing object for a state that was detached and reset
// before invoking callbacks. Compiler-TLS state may already have been reused
// reentrantly and must not be inspected or cleared here.
static void iree_hal_streaming_context_tls_state_free_detached(
    iree_hal_streaming_context_tls_state_t* state) {
#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  iree_allocator_free(iree_allocator_system(), state);
#else
  (void)state;
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
}

static iree_status_t iree_hal_streaming_context_tls_reference_add(
    iree_hal_streaming_context_tls_state_t* state, bool state_is_new) {
  IREE_ASSERT(state);
  IREE_ASSERT(state_is_new == (state->context_reference_count == 0));
  if (state_is_new) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_tls_set(iree_hal_streaming_context_tls_key, state));
  }
  ++state->context_reference_count;
  iree_atomic_fetch_add(&iree_hal_streaming_tls_context_reference_count, 1,
                        iree_memory_order_acq_rel);
  return iree_ok_status();
}

static void iree_hal_streaming_context_tls_reference_remove_committed(
    iree_hal_streaming_context_tls_state_t* state) {
  IREE_ASSERT(state->context_reference_count > 0);
  --state->context_reference_count;
  iree_atomic_fetch_sub(&iree_hal_streaming_tls_context_reference_count, 1,
                        iree_memory_order_acq_rel);
}

// Removes one TLS reference. When this is the last reference, a teardown
// sentinel keeps global cleanup excluded until the caller has released the
// detached context and freed the detached state.
static iree_status_t iree_hal_streaming_context_tls_reference_remove(
    iree_hal_streaming_context_tls_state_t* state,
    bool* out_teardown_sentinel_held) {
  IREE_ASSERT(state->context_reference_count > 0);
  IREE_ASSERT_ARGUMENT(out_teardown_sentinel_held);
  *out_teardown_sentinel_held = false;
  if (state->context_reference_count == 1) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_tls_set(iree_hal_streaming_context_tls_key, NULL));
    iree_atomic_fetch_add(&iree_hal_streaming_tls_context_reference_count, 1,
                          iree_memory_order_acq_rel);
    *out_teardown_sentinel_held = true;
  }
  iree_hal_streaming_context_tls_reference_remove_committed(state);
  return iree_ok_status();
}

static void iree_hal_streaming_context_tls_release_teardown_sentinel(void) {
  iree_atomic_fetch_sub(&iree_hal_streaming_tls_context_reference_count, 1,
                        iree_memory_order_acq_rel);
}

// Stream IDs identify timeline dependencies that can outlive the context that
// created them. Keeping the namespace process-wide prevents a dependency from
// being mistaken for one recorded by a different context.
static iree_atomic_uint64_t iree_hal_streaming_next_stream_id =
    IREE_ATOMIC_VAR_INIT(1);

static void iree_hal_streaming_context_stack_free_storage(
    iree_hal_streaming_context_stack_t* stack) {
  IREE_ASSERT(stack->depth == 0);
  iree_allocator_free(stack->allocator, stack->contexts);
  *stack = (iree_hal_streaming_context_stack_t){0};
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_context_destroy(
    iree_hal_streaming_context_t* context);
static iree_status_t iree_hal_streaming_context_synchronize_streams(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams,
    bool flush_before_wait);

iree_hal_streaming_timestamp_domain_t iree_hal_streaming_query_timestamp_domain(
    const iree_hal_device_spec_t* spec) {
  const iree_hal_streaming_timestamp_domain_t none = {0};
  if (!spec) return none;
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  if (!iree_all_bits_set(timing->flags,
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS)) {
    return none;
  }
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  if (queues->family_count != 1) return none;
  const iree_hal_queue_family_spec_t* family = &queues->families[0];
  // Exactly one physical device: the header guarantees a single comparable
  // domain only for one family covering one physical device.
  if (iree_math_count_ones_u64(family->physical_device_affinity) != 1) {
    return none;
  }
  // The flag comes from the device-scope summary and the numbers from the
  // family, which is sound only while the two describe the same domain. On a
  // single-family device they must, so facets that disagree contradict the
  // device's own facts and name no domain to convert with.
  if (timing->timestamp_frequency_hz != family->timestamp_frequency_hz ||
      timing->timestamp_valid_bits != family->timestamp_valid_bits) {
    return none;
  }
  if (family->timestamp_frequency_hz == 0 ||
      family->timestamp_valid_bits == 0 || family->timestamp_valid_bits > 64) {
    return none;
  }
  const iree_hal_streaming_timestamp_domain_t domain = {
      .frequency_hz = family->timestamp_frequency_hz,
      .valid_bits = family->timestamp_valid_bits,
  };
  return domain;
}

iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device_entry);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "global symbol registry failed to initialize");
  }

  iree_hal_streaming_context_t* context = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*context),
                                (void**)&context));
  iree_atomic_ref_count_init(&context->ref_count);
  iree_atomic_store(&context->accepting_work, 1, iree_memory_order_relaxed);
  iree_atomic_store(&context->teardown_quiesced, 0, iree_memory_order_relaxed);
  context->device = device_entry->hal_device;
  context->device_ordinal = device_entry->ordinal;
  context->device_entry = device_entry;
  context->is_primary = false;
  context->runtime_generation = device_entry->runtime_generation;
  context->device_epoch =
      iree_atomic_load(&device_entry->reset_epoch, iree_memory_order_acquire);
  context->queue = NULL;
  context->device_allocator =
      iree_hal_device_allocator(device_entry->hal_device);
  context->timestamp_domain = iree_hal_streaming_query_timestamp_domain(
      iree_hal_device_spec(device_entry->hal_device));
  context->flags = flags;
  context->default_stream = NULL;
  context->next_capture_id = 1;
  context->peer_contexts = NULL;
  context->peer_count = 0;
  context->peer_capacity = 0;
  memset(&context->symbol_map, 0, sizeof(context->symbol_map));
  memset(&context->buffer_table, 0, sizeof(context->buffer_table));
  context->pending_free_head = NULL;
  context->pageable_h2d_staging_buffer = NULL;
  context->pageable_h2d_staging_size = 0;
  iree_atomic_store(&context->capture_stream_count, 0,
                    iree_memory_order_relaxed);
  context->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context->mutex);
  iree_slim_mutex_initialize(&context->pending_free_mutex);

  // Initialize global list pointers.
  context->context_list_entry.next = NULL;
  context->context_list_entry.prev = NULL;

  // Initialize stream tracking BEFORE creating default stream.
  iree_slim_mutex_initialize(&context->stream_list_mutex);
  context->stream_count = 0;
  context->stream_capacity =
      8;  // Pre-allocate for default stream + user streams.
  context->streams = NULL;
  context->stream_wait_frontier = NULL;
  context->event_record_timeline = (iree_hal_streaming_operation_timeline_t){0};
  iree_slim_mutex_initialize(&context->event_record_mutex);

  // Initialize default limits.
  // These are typical defaults matching CUDA/HIP behavior.
  context->limits.stack_size = 1024;                        // 1KB default
  context->limits.printf_fifo_size = 1024 * 1024;           // 1MB
  context->limits.malloc_heap_size = 8 * 1024 * 1024;       // 8MB
  context->limits.dev_runtime_sync_depth = 128;             // 128 levels
  context->limits.dev_runtime_pending_launch_count = 2048;  // 2048 launches
  context->limits.max_l2_fetch_granularity = 128;           // 128 bytes
  context->limits.persisting_l2_cache_size = 0;             // 0 = default
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  context->lifecycle_begin =
      device_registry ? device_registry->lifecycle_begin : NULL;
  context->lifecycle_end =
      device_registry ? device_registry->lifecycle_end : NULL;
  context->pointer_resolver =
      device_registry ? device_registry->pointer_resolver : NULL;
  context->lifecycle_user_data =
      device_registry ? device_registry->lifecycle_user_data : NULL;

  // Retain the HAL device.
  iree_hal_device_retain(context->device);
  iree_hal_allocator_retain(context->device_allocator);

  // Initialize buffer mapping table.
  hrx_buffer_table_initialize(&context->buffer_table);

  // Initialize the device timestamp slot pool.
  iree_hal_streaming_event_timestamp_pool_initialize(
      context->device_allocator, host_allocator, &context->timestamp_pool);

  iree_status_t status = iree_hal_streaming_device_select_primary_queue(
      device_entry, &context->queue);

  // Create the context-wide event record timeline. Resource-partitioned
  // binding contexts share this timeline so primary-context synchronization
  // includes their context-level records without relying on queue FIFO order.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &context->event_record_timeline.semaphore);
  }

  // Initialize symbol map with global registry as the backing store.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_symbol_map_initialize(
        context, /*initial_capacity=*/16, registry, host_allocator,
        &context->symbol_map);
  }

  // Allocate stream tracking array.
  if (iree_status_is_ok(status)) {
    iree_host_size_t stream_array_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(context->stream_capacity,
                                                  sizeof(context->streams[0]),
                                                  &stream_array_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream list capacity overflow");
    } else {
      status = iree_allocator_malloc(host_allocator, stream_array_size,
                                     (void**)&context->streams);
    }
  }

  // Create default stream.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_create(
        context, context->queue, /*flags=*/0, /*priority=*/0, host_allocator,
        &context->default_stream);
  }

  if (iree_status_is_ok(status)) {
    // Register with global list.
    iree_hal_streaming_register_context(context);
    *out_context = context;
  } else {
    iree_hal_streaming_context_destroy(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_context_destroy(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unregister from global list.
  iree_hal_streaming_unregister_context(context);

  // Clean up peer contexts array.
  if (context->peer_contexts) {
    for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
      iree_hal_streaming_context_release(context->peer_contexts[i]);
    }
    iree_allocator_free(context->host_allocator, context->peer_contexts);
  }

  // Prepared teardown certifies this wait while the last public owner is still
  // published. Other destruction paths retain the conservative wait.
  if (iree_atomic_load(&context->teardown_quiesced,
                       iree_memory_order_acquire) == 0) {
    iree_status_t status = iree_hal_streaming_context_synchronize(context);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
  }
  iree_status_t status =
      iree_hal_streaming_memory_release_terminal_async_frees(context);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  iree_hal_streaming_memory_release_pageable_staging(context);

  // Deinitialize symbol map and unload any statically-registered modules that
  // were on-demand loaded for this context.
  iree_hal_streaming_context_symbol_map_deinitialize(&context->symbol_map);

  // Deinitialize buffer mapping table.
  hrx_buffer_table_deinitialize(&context->buffer_table);
  IREE_ASSERT(context->pending_free_head == NULL,
              "pending asynchronous frees must be drained before context "
              "destruction");
  iree_slim_mutex_deinitialize(&context->pending_free_mutex);

  // Release default stream.
  // This releases the context's reference but not the list's reference.
  iree_hal_streaming_stream_t* default_stream = context->default_stream;
  context->default_stream = NULL;

  // Emptying the list under its own mutex is what takes the streams out of
  // reach of every other reader: they all walk the list under this lock and now
  // find nothing. It also disarms unregistration, whose scan of the emptied
  // list matches no stream and so writes nothing, which leaves registration as
  // the only writer that could still reach the array - and that cannot run
  // either. A context reaches destruction in one of two states, and neither
  // admits a registration: unpublished, which is how
  // iree_hal_streaming_context_create disposes of a context it failed to finish
  // building and where no thread but that one can name it; or published with
  // its last reference gone, and registering runs under a reference. The array
  // is therefore stable for the walk below.
  iree_slim_mutex_lock(&context->stream_list_mutex);
  const iree_host_size_t detached_stream_count = context->stream_count;
  context->stream_count = 0;
  iree_hal_fence_t* stream_wait_frontier = context->stream_wait_frontier;
  context->stream_wait_frontier = NULL;
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Detach under each stream's own mutex, which is the lock
  // iree_hal_streaming_stream_retain_context reads the context under, so no
  // reader can be part way through that read when the field is cleared. A
  // reader that gets there first is refused anyway: it retains through
  // iree_hal_streaming_context_try_retain, which fails once the last reference
  // is gone, and an unpublished context has no such reader to refuse. The list
  // mutex is not held: end_capture holds a stream mutex while walking the list,
  // and taking these in the other order deadlocks against it. The list still
  // holds its reference to every stream, so none can be destroyed while the
  // loop runs; those references are released afterwards, outside both locks,
  // because the last one destroys the stream. Queue references are released
  // during detachment so a dynamically acquired queue cannot outlive the HAL
  // device retained by this context.
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    iree_hal_queue_t* queue = NULL;
    iree_hal_queue_t* cooperative_queue = NULL;
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context == context) {
      queue = stream->queue;
      cooperative_queue = stream->cooperative_queue;
      stream->queue = NULL;
      stream->cooperative_queue = NULL;
      stream->context = NULL;
    }
    iree_slim_mutex_unlock(&stream->mutex);
    iree_hal_queue_release(cooperative_queue);
    iree_hal_queue_release(queue);
  }
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_release(context->streams[i]);
  }
  iree_hal_fence_release(stream_wait_frontier);

  // Now release the context's reference to default stream.
  iree_hal_streaming_stream_release(default_stream);

  // Free stream tracking resources.
  if (context->streams) {
    iree_allocator_free(context->host_allocator, context->streams);
  }
  iree_slim_mutex_deinitialize(&context->stream_list_mutex);

  iree_hal_semaphore_release(context->event_record_timeline.semaphore);
  iree_slim_mutex_deinitialize(&context->event_record_mutex);

  // A record draws its slot from the pool of the recording event's own context
  // and every event retains that context, so reaching here means every event
  // that could hold a slot from this pool is gone and every slot is back.
  iree_hal_streaming_event_timestamp_pool_deinitialize(
      &context->timestamp_pool);

  iree_status_ignore(context->loop_status);
  iree_hal_allocator_release(context->device_allocator);
  iree_hal_device_release(context->device);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&context->mutex);

  // Free context memory.
  const iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_context_retain(iree_hal_streaming_context_t* context) {
  if (context) {
    iree_atomic_ref_count_inc(&context->ref_count);
  }
}

bool iree_hal_streaming_context_try_retain(
    iree_hal_streaming_context_t* context) {
  if (!context) return false;
  int32_t reference_count = iree_atomic_ref_count_load(&context->ref_count);
  while (reference_count > 0) {
    if (iree_atomic_compare_exchange_weak(
            &context->ref_count, &reference_count, reference_count + 1,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context) {
  if (context && iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_hal_streaming_context_destroy(context);
  }
}

void iree_hal_streaming_context_discard_unpublished(
    iree_hal_streaming_context_t* context) {
  if (!context) return;
  // Context creation publishes an owning global-list edge before returning.
  // A caller that fails before publishing its native handle must remove that
  // edge before dropping the creator edge or the list will pin an unreachable
  // context indefinitely.
  iree_hal_streaming_unregister_context(context);
  iree_hal_streaming_context_release(context);
}

iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->flags;
}

iree_hal_streaming_context_t* iree_hal_streaming_context_current(void) {
  iree_hal_streaming_context_tls_state_t* state =
      iree_hal_streaming_context_tls_state();
  return state ? state->current_context : NULL;
}

uintptr_t iree_hal_streaming_current_thread_token(void) {
  return (uintptr_t)&iree_hal_streaming_thread_token_storage;
}

iree_status_t iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_tls_state_t* state =
      iree_hal_streaming_context_tls_state();
  iree_hal_streaming_context_t* old_context =
      state ? state->current_context : NULL;
  bool state_is_new = false;
  bool teardown_sentinel_held = false;
  iree_status_t status = iree_ok_status();
  if (!old_context && context) {
    status =
        iree_hal_streaming_context_tls_state_prepare(&state, &state_is_new);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_streaming_context_tls_reference_add(state, state_is_new);
    }
  } else if (old_context && !context) {
    status = iree_hal_streaming_context_tls_reference_remove(
        state, &teardown_sentinel_held);
  }
  if (!iree_status_is_ok(status)) {
    if (state_is_new) {
      iree_hal_streaming_context_tls_state_discard_empty(state);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (context) {
    iree_hal_streaming_context_retain(context);
  }
  iree_hal_streaming_context_stack_t detached_stack = {0};
  if (teardown_sentinel_held) {
    detached_stack = state->stack;
    state->current_context = NULL;
    state->stack = (iree_hal_streaming_context_stack_t){0};
  } else if (state) {
    state->current_context = context;
  }
  iree_hal_streaming_context_release(old_context);
  if (teardown_sentinel_held) {
    iree_hal_streaming_context_stack_free_storage(&detached_stack);
    iree_hal_streaming_context_tls_state_free_detached(state);
    iree_hal_streaming_context_tls_release_teardown_sentinel();
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_tls_state_t* state = NULL;
  bool state_is_new = false;
  iree_status_t status =
      iree_hal_streaming_context_tls_state_prepare(&state, &state_is_new);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Grow stack if needed.
  if (state->stack.depth >= state->stack.capacity) {
    iree_host_size_t new_capacity =
        state->stack.capacity ? state->stack.capacity * 2 : 8;
    const iree_allocator_t stack_allocator = state->stack.capacity
                                                 ? state->stack.allocator
                                                 : context->host_allocator;
    status = iree_allocator_realloc(
        stack_allocator, new_capacity * sizeof(iree_hal_streaming_context_t*),
        (void**)&state->stack.contexts);
    if (!iree_status_is_ok(status)) {
      if (state_is_new) {
        iree_hal_streaming_context_tls_state_discard_empty(state);
      }
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    state->stack.capacity = new_capacity;
    state->stack.allocator = stack_allocator;
  }

  // Install the thread-exit marker before adding the first owning TLS
  // reference. A failure leaves the current slot and stack unchanged.
  status = iree_hal_streaming_context_tls_reference_add(state, state_is_new);
  if (!iree_status_is_ok(status)) {
    if (state_is_new) {
      iree_hal_streaming_context_tls_state_discard_empty(state);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Push current context onto stack.
  if (state->current_context) {
    state->stack.contexts[state->stack.depth++] = state->current_context;
  }

  // Set new current context.
  iree_hal_streaming_context_retain(context);
  state->current_context = context;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_pop(
    iree_hal_streaming_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_context) *out_context = NULL;

  iree_hal_streaming_context_tls_state_t* state =
      iree_hal_streaming_context_tls_state();
  if (!state) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Release current context.
  if (state->current_context) {
    iree_hal_streaming_context_t* popped_context = state->current_context;
    bool teardown_sentinel_held = false;
    iree_status_t status = iree_hal_streaming_context_tls_reference_remove(
        state, &teardown_sentinel_held);
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    if (state->stack.depth > 0) {
      state->current_context = state->stack.contexts[--state->stack.depth];
    } else {
      state->current_context = NULL;
    }
    if (out_context) {
      // Public context handles are non-owning values backed by their normal
      // registry/public ownership. Popping only removes the TLS ownership; it
      // must not silently transfer that reference to an untracked raw handle.
      *out_context = popped_context;
    }
    iree_hal_streaming_context_stack_t detached_stack = {0};
    if (teardown_sentinel_held) {
      detached_stack = state->stack;
      state->stack = (iree_hal_streaming_context_stack_t){0};
    }
    iree_hal_streaming_context_release(popped_context);
    if (teardown_sentinel_held) {
      iree_hal_streaming_context_stack_free_storage(&detached_stack);
      iree_hal_streaming_context_tls_state_free_detached(state);
      iree_hal_streaming_context_tls_release_teardown_sentinel();
    }
  } else if (state->stack.depth > 0) {
    // A caller may explicitly clear the current slot while pushed contexts
    // remain. Move the top stack ownership into the current slot without
    // changing the number of TLS references.
    state->current_context = state->stack.contexts[--state->stack.depth];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_host_size_t iree_hal_streaming_context_tls_reference_count(void) {
  const int32_t count =
      iree_atomic_load(&iree_hal_streaming_tls_context_reference_count,
                       iree_memory_order_acquire);
  IREE_ASSERT(count >= 0);
  return (iree_host_size_t)count;
}

iree_host_size_t iree_hal_streaming_context_current_thread_tls_reference_count(
    void) {
  iree_hal_streaming_context_tls_state_t* state =
      iree_hal_streaming_context_tls_state();
  IREE_ASSERT(!state || state->context_reference_count > 0);
  return state ? (iree_host_size_t)state->context_reference_count : 0;
}

iree_host_size_t
iree_hal_streaming_context_current_thread_tls_reference_count_for(
    const iree_hal_streaming_context_t* context) {
  if (!context) return 0;
  iree_hal_streaming_context_tls_state_t* state =
      iree_hal_streaming_context_tls_state();
  if (!state) return 0;
  iree_host_size_t count = state->current_context == context ? 1 : 0;
  for (iree_host_size_t i = 0; i < state->stack.depth; ++i) {
    if (state->stack.contexts[i] == context) ++count;
  }
  return count;
}

iree_status_t iree_hal_streaming_context_snapshot_all(
    iree_hal_streaming_context_t*** out_contexts,
    iree_host_size_t* out_context_count) {
  IREE_ASSERT_ARGUMENT(out_contexts);
  IREE_ASSERT_ARGUMENT(out_context_count);
  *out_contexts = NULL;
  *out_context_count = 0;
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_device_registry();
  if (!registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_host_size_t count = 0;
  iree_slim_mutex_lock(&registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context = registry->context_list.head;
       context; context = context->context_list_entry.next) {
    ++count;
  }
  iree_hal_streaming_context_t** contexts = NULL;
  iree_status_t status = iree_ok_status();
  if (count > 0) {
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(count, sizeof(contexts[0]),
                                                  &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "context snapshot size overflow");
    } else {
      status = iree_allocator_malloc(iree_allocator_system(), allocation_size,
                                     (void**)&contexts);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t index = 0;
    for (iree_hal_streaming_context_t* context = registry->context_list.head;
         context; context = context->context_list_entry.next) {
      contexts[index++] = context;
      iree_hal_streaming_context_retain(context);
    }
    *out_contexts = contexts;
    *out_context_count = index;
  }
  iree_slim_mutex_unlock(&registry->context_list.mutex);
  return status;
}

void iree_hal_streaming_context_release_snapshot_all(
    iree_hal_streaming_context_t** contexts, iree_host_size_t context_count) {
  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  iree_allocator_free(iree_allocator_system(), contexts);
}

iree_status_t iree_hal_streaming_context_clear_current_thread(void) {
  iree_hal_streaming_context_tls_state_t* state =
      iree_hal_streaming_context_tls_state();
  if (!state) return iree_ok_status();

  // This is the only fallible transition. Until the complete state has been
  // detached from the key, no current pointer, stack entry, owning reference,
  // count, or allocation is mutated, so a clear failure is exactly retryable.
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_tls_set(iree_hal_streaming_context_tls_key, NULL));
  iree_hal_streaming_context_tls_state_cleanup_detached(state);
  return iree_ok_status();
}

static void iree_hal_streaming_context_tls_state_cleanup_detached(
    iree_hal_streaming_context_tls_state_t* state) {
  if (!state) return;

  const int32_t reference_count = state->context_reference_count;
  IREE_ASSERT(reference_count > 0);
  IREE_ASSERT((iree_host_size_t)reference_count ==
              state->stack.depth + (state->current_context ? 1 : 0));

  // The old references still exclude global teardown while the sentinel is
  // installed. Subtract them only after the sentinel is visible, ensuring the
  // global count cannot pass through zero before release/destruction finishes.
  iree_atomic_fetch_add(&iree_hal_streaming_tls_context_reference_count, 1,
                        iree_memory_order_acq_rel);

  iree_hal_streaming_context_t* current_context = state->current_context;
  iree_hal_streaming_context_stack_t stack = state->stack;
  state->current_context = NULL;
  state->stack = (iree_hal_streaming_context_stack_t){0};
  state->context_reference_count = 0;
  iree_atomic_fetch_sub(&iree_hal_streaming_tls_context_reference_count,
                        reference_count, iree_memory_order_acq_rel);

  // The TLS key and complete local state were detached before any release.
  // Reentrant callbacks therefore create and install a distinct valid state.
  iree_hal_streaming_context_release(current_context);
  while (stack.depth > 0) {
    iree_hal_streaming_context_release(stack.contexts[--stack.depth]);
  }
  iree_hal_streaming_context_stack_free_storage(&stack);
  iree_hal_streaming_context_tls_state_free_detached(state);

  // Keep teardown excluded through context destruction and stack/state frees.
  iree_hal_streaming_context_tls_release_teardown_sentinel();
}

#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
bool iree_hal_streaming_context_tls_test_marker_is_set(void) {
  return iree_hal_streaming_context_tls_state() != NULL;
}

iree_status_t iree_hal_streaming_context_tls_test_reset(void) {
  IREE_ASSERT(iree_hal_streaming_context_tls_state() == NULL);
  IREE_ASSERT(iree_hal_streaming_context_tls_reference_count() == 0);
  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_NONE);
  return iree_hal_streaming_context_tls_deinitialize();
}
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

iree_status_t iree_hal_streaming_context_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t* out_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;

  // Return the limit value from context.
  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      *out_value = context->limits.stack_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      *out_value = context->limits.printf_fifo_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      *out_value = context->limits.malloc_heap_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      *out_value = context->limits.dev_runtime_sync_depth;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      *out_value = context->limits.dev_runtime_pending_launch_count;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      *out_value = context->limits.max_l2_fetch_granularity;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      *out_value = context->limits.persisting_l2_cache_size;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid limit type %d", limit);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_set_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate the limit value first without holding the lock.
  iree_status_t status = iree_ok_status();
  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      // Stack size must be at least 512 bytes.
      if (value < 512) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack size must be at least 512 bytes");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      // Printf FIFO must be at least 4KB.
      if (value < 4096) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "printf FIFO size must be at least 4KB");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      // Heap size must be at least 4KB.
      if (value < 4096) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malloc heap size must be at least 4KB");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      // Must be at least 1.
      if (value < 1) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "runtime sync depth must be at least 1");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      // Must be at least 1.
      if (value < 1) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "runtime pending launch count must be at least 1");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      // Must be 0, 32, 64, or 128 bytes.
      if (value != 0 && value != 32 && value != 64 && value != 128) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "L2 fetch granularity must be 0, 32, 64, or 128");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      // No specific validation for cache size.
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid limit type %d", limit);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Now take the lock and set the value.
  iree_slim_mutex_lock(&context->mutex);

  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      context->limits.stack_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      context->limits.printf_fifo_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      context->limits.malloc_heap_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      context->limits.dev_runtime_sync_depth = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      context->limits.dev_runtime_pending_launch_count = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      context->limits.max_l2_fetch_granularity = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      context->limits.persisting_l2_cache_size = value;
      break;
    default:
      // Already validated above, should not reach here.
      break;
  }

  iree_slim_mutex_unlock(&context->mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_enable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(peer_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);

  // Check if already enabled.
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();  // Already enabled.
    }
  }

  // Grow peer array if needed.
  if (context->peer_count >= context->peer_capacity) {
    const iree_host_size_t new_capacity =
        context->peer_capacity ? context->peer_capacity * 2 : 4;
    iree_status_t status = iree_allocator_realloc(
        context->host_allocator,
        new_capacity * sizeof(iree_hal_streaming_context_t*),
        (void**)&context->peer_contexts);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    context->peer_capacity = new_capacity;
  }

  // Add peer context.
  iree_hal_streaming_context_retain(peer_context);
  context->peer_contexts[context->peer_count++] = peer_context;

  // Update P2P topology if we have the registry.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (device_registry && device_registry->p2p_topology) {
    const iree_host_size_t src_ordinal = context->device_ordinal;
    const iree_host_size_t dst_ordinal = peer_context->device_ordinal;
    const iree_host_size_t device_count = device_registry->device_count;
    if (src_ordinal < device_count && dst_ordinal < device_count) {
      // Find the link in topology.
      const iree_host_size_t link_index =
          src_ordinal * device_count + dst_ordinal;
      iree_hal_streaming_p2p_link_t* link =
          &device_registry->p2p_topology[link_index];
      // Enable P2P access.
      link->access_supported = true;
      // TODO: Query actual P2P capabilities.
    }
  }

  iree_slim_mutex_unlock(&context->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_disable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(peer_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);

  // Find and remove peer.
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      const iree_host_size_t dst_ordinal = peer_context->device_ordinal;

      // Release peer context.
      iree_hal_streaming_context_release(peer_context);

      // Shift remaining peers.
      for (iree_host_size_t j = i + 1; j < context->peer_count; ++j) {
        context->peer_contexts[j - 1] = context->peer_contexts[j];
      }
      context->peer_count--;

      // Update P2P topology.
      iree_hal_streaming_device_registry_t* device_registry =
          iree_hal_streaming_device_registry();
      if (device_registry && device_registry->p2p_topology) {
        const iree_host_size_t src_ordinal = context->device_ordinal;
        const iree_host_size_t device_count = device_registry->device_count;
        if (src_ordinal < device_count && dst_ordinal < device_count) {
          // Find the link in topology.
          const iree_host_size_t link_index =
              src_ordinal * device_count + dst_ordinal;
          iree_hal_streaming_p2p_link_t* link =
              &device_registry->p2p_topology[link_index];
          // Disable P2P access.
          link->access_supported = false;
        }
      }

      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  iree_slim_mutex_unlock(&context->mutex);

  // Peer not found.
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_NOT_FOUND, "peer context not found");
}

iree_status_t iree_hal_streaming_context_register_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  iree_hal_fence_t* wait_frontier = NULL;

  iree_slim_mutex_lock(&context->stream_list_mutex);

  // Grow array if needed (double capacity).
  if (context->stream_count >= context->stream_capacity) {
    iree_host_size_t new_capacity = 0;
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(context->stream_capacity, 2,
                                                  &new_capacity) ||
                      !iree_host_size_checked_mul(new_capacity,
                                                  sizeof(context->streams[0]),
                                                  &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream list capacity overflow");
    } else {
      status = iree_allocator_realloc(context->host_allocator, allocation_size,
                                      (void**)&context->streams);
    }
    if (iree_status_is_ok(status)) {
      context->stream_capacity = new_capacity;
    }
  }

  if (iree_status_is_ok(status)) {
    uint64_t stream_id = iree_atomic_load(&iree_hal_streaming_next_stream_id,
                                          iree_memory_order_relaxed);
    while (stream_id != 0 && stream_id != UINT64_MAX &&
           !iree_atomic_compare_exchange_weak(
               &iree_hal_streaming_next_stream_id, &stream_id, stream_id + 1,
               iree_memory_order_relaxed, iree_memory_order_relaxed)) {
    }
    if (stream_id == 0 || stream_id == UINT64_MAX) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream identifier space exhausted");
    } else {
      stream->stream_id = stream_id;
    }
  }

  if (iree_status_is_ok(status)) {
    // Retain the stream - the context's stream list owns a reference.
    iree_hal_streaming_stream_retain(stream);
    context->streams[context->stream_count++] = stream;
    wait_frontier = context->stream_wait_frontier;
    iree_hal_fence_retain(wait_frontier);
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  if (iree_status_is_ok(status) && wait_frontier) {
    status = iree_hal_streaming_stream_wait_semaphores(
        stream, iree_hal_fence_semaphore_list(wait_frontier));
  }
  iree_hal_fence_release(wait_frontier);
  if (!iree_status_is_ok(status) && stream->stream_id != 0) {
    iree_hal_streaming_context_unregister_stream(context, stream);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_allocate_capture_id(
    iree_hal_streaming_context_t* context, unsigned long long* out_capture_id) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_capture_id);
  *out_capture_id = 0;

  iree_slim_mutex_lock(&context->stream_list_mutex);
  if (context->next_capture_id == 0) {
    iree_slim_mutex_unlock(&context->stream_list_mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream capture identifier space exhausted");
  }
  *out_capture_id = context->next_capture_id++;
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  return iree_ok_status();
}

void iree_hal_streaming_context_unregister_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  if (!context || !stream) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  bool found = false;
  iree_slim_mutex_lock(&context->stream_list_mutex);

  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    if (context->streams[i] == stream) {
      // Swap with last and remove.
      context->streams[i] = context->streams[context->stream_count - 1];
      --context->stream_count;
      found = true;
      break;
    }
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Release the list's reference after unlinking. The caller holds another
  // reference while requesting unregister, so the stream cannot be destroyed
  // out from under this operation.
  if (found) {
    iree_hal_streaming_stream_release(stream);
  }

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_streaming_context_has_peer_contexts(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return false;

  bool has_peer = false;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* candidate =
           device_registry->context_list.head;
       candidate; candidate = candidate->context_list_entry.next) {
    if (candidate != context) {
      has_peer = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return has_peer;
}

// Takes a retained snapshot while the caller holds |stream_list_mutex|.
static iree_status_t iree_hal_streaming_context_snapshot_streams_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_streams);
  IREE_ASSERT_ARGUMENT(out_count);

  const iree_host_size_t count = context->stream_count;
  iree_hal_streaming_stream_t** streams = NULL;
  iree_status_t status = iree_ok_status();
  if (count > 0) {
    iree_host_size_t streams_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(count, sizeof(streams[0]),
                                                  &streams_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream snapshot size overflow");
    } else {
      status = iree_allocator_malloc(context->host_allocator, streams_size,
                                     (void**)&streams);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < count; ++i) {
        streams[i] = context->streams[i];
        if (streams[i]) {
          iree_hal_streaming_stream_retain(streams[i]);
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_streams = streams;
    *out_count = count;
  }
  return status;
}

iree_status_t iree_hal_streaming_context_snapshot_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams,
    iree_host_size_t* out_stream_count) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_streams);
  IREE_ASSERT_ARGUMENT(out_stream_count);

  iree_slim_mutex_lock(&context->stream_list_mutex);
  iree_status_t status = iree_hal_streaming_context_snapshot_streams_locked(
      context, out_streams, out_stream_count);
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  return status;
}

void iree_hal_streaming_context_release_stream_snapshot(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t** streams, iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_release(streams[i]);
  }
  if (streams) {
    iree_allocator_free(context->host_allocator, streams);
  }
}

iree_status_t iree_hal_streaming_context_record_event(
    iree_hal_streaming_context_t* context, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(event);
  if (IREE_UNLIKELY(event->context != context)) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "event belongs to a different context");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams, &stream_count);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_event_record_after_streams(
        event, streams, stream_count, /*additional_timeline=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_flush(context->queue);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams,
                                                     stream_count);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_wait_frontier_extend(
    iree_hal_fence_t* previous_frontier, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_frontier) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_frontier);
  const iree_hal_semaphore_list_t previous_points =
      iree_hal_fence_semaphore_list(previous_frontier);
  iree_host_size_t capacity = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(previous_points.count, 1, &capacity))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream wait frontier capacity overflow");
  }

  iree_hal_fence_t* frontier = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_create(capacity, host_allocator, &frontier));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < previous_points.count && iree_status_is_ok(status); ++i) {
    uint64_t current_value = 0;
    status =
        iree_hal_semaphore_query(previous_points.semaphores[i], &current_value);
    if (iree_status_is_ok(status) &&
        current_value < previous_points.payload_values[i]) {
      status = iree_hal_fence_insert(frontier, previous_points.semaphores[i],
                                     previous_points.payload_values[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_fence_insert(frontier, semaphore, value);
  }

  if (iree_status_is_ok(status)) {
    *out_frontier = frontier;
  } else {
    iree_hal_fence_release(frontier);
  }
  return status;
}

iree_status_t iree_hal_streaming_context_wait_event(
    iree_hal_streaming_context_t* context, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(event);
  if (IREE_UNLIKELY(iree_hal_streaming_event_has_capture_graph(event))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "captured event has no submitted point");
  }

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  if (!recorded_point.semaphore) {
    iree_hal_streaming_event_release_recorded_point(&recorded_point);
    return iree_ok_status();
  }

  uint64_t current_value = 0;
  iree_status_t status =
      iree_hal_semaphore_query(recorded_point.semaphore, &current_value);
  if (!iree_status_is_ok(status) || current_value >= recorded_point.value) {
    iree_hal_streaming_event_release_recorded_point(&recorded_point);
    return status;
  }

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_hal_fence_t* new_frontier = NULL;
  iree_hal_fence_t* old_frontier = NULL;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  status = iree_hal_streaming_wait_frontier_extend(
      context->stream_wait_frontier, recorded_point.semaphore,
      recorded_point.value, context->host_allocator, &new_frontier);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_snapshot_streams_locked(
        context, &streams, &stream_count);
  }
  if (iree_status_is_ok(status)) {
    old_frontier = context->stream_wait_frontier;
    context->stream_wait_frontier = new_frontier;
    new_frontier = NULL;
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t wait = {
        .count = 1,
        .semaphores = &recorded_point.semaphore,
        .payload_values = &recorded_point.value,
    };
    for (iree_host_size_t i = 0; i < stream_count; ++i) {
      status = iree_status_join(
          status, iree_hal_streaming_stream_wait_semaphores(streams[i], wait));
    }
  }

  iree_hal_fence_release(new_frontier);
  iree_hal_fence_release(old_frontier);
  iree_hal_streaming_context_release_stream_snapshot(context, streams,
                                                     stream_count);
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  return status;
}

iree_status_t iree_hal_streaming_context_wait_idle(
    iree_hal_streaming_context_t* context, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(context);
  (void)timeout;
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/true,
      /*flush_before_wait=*/true);
}

iree_status_t iree_hal_streaming_context_quiesce_for_teardown(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_teardown_wait_observer_t wait_observer,
    void* wait_observer_user_data, iree_status_t* out_execution_status) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_execution_status);
  *out_execution_status = iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Exclusive binding teardown prevents new submissions. Retain the complete
  // stream set before flushing so destruction cannot invalidate a wait target.
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Flush every stream even if one flush fails. A flush error remains fatal:
  // unlike a verified terminal wait result it does not prove that accepted
  // but unsubmitted work has reached a terminal state.
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    if (streams[i]) {
      status =
          iree_status_join(status, iree_hal_streaming_stream_flush(streams[i]));
    }
  }
  if (context->default_stream) {
    status = iree_status_join(
        status, iree_hal_streaming_stream_flush(context->default_stream));
  }

  // Wait every captured frontier regardless of prior stream results. This is
  // teardown: returning the first error early could leave another stream
  // executing against storage the caller is about to retire.
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    if (!streams[i]) continue;
    iree_status_t execution_status = iree_ok_status();
    status = iree_status_join(
        status, iree_hal_streaming_stream_wait_submitted_or_terminal(
                    streams[i], wait_observer, wait_observer_user_data,
                    &execution_status));
    *out_execution_status =
        iree_status_join(*out_execution_status, execution_status);
  }
  if (context->default_stream) {
    iree_status_t execution_status = iree_ok_status();
    status = iree_status_join(
        status, iree_hal_streaming_stream_wait_submitted_or_terminal(
                    context->default_stream, wait_observer,
                    wait_observer_user_data, &execution_status));
    *out_execution_status =
        iree_status_join(*out_execution_status, execution_status);
  }

  // Context event records are outside the ordinary stream list but may still
  // reference teardown resources. Their terminal failure has the same
  // completed-frontier meaning as a stream timeline failure.
  iree_hal_semaphore_t* event_semaphore = NULL;
  uint64_t event_value = 0;
  iree_slim_mutex_lock(&context->event_record_mutex);
  if (context->event_record_timeline.pending_value > 0) {
    event_semaphore = context->event_record_timeline.semaphore;
    event_value = context->event_record_timeline.pending_value;
    iree_hal_semaphore_retain(event_semaphore);
  }
  iree_slim_mutex_unlock(&context->event_record_mutex);
  if (event_semaphore) {
    iree_status_t wait_status = iree_hal_semaphore_wait(
        event_semaphore, event_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    if (!iree_status_is_ok(wait_status)) {
      uint64_t current_value = 0;
      iree_status_t query_status =
          iree_hal_semaphore_query(event_semaphore, &current_value);
      if (iree_status_is_ok(query_status)) {
        status = iree_status_join(status, wait_status);
      } else {
        iree_status_ignore(wait_status);
        *out_execution_status =
            iree_status_join(*out_execution_status, query_status);
      }
    }
  }
  iree_hal_semaphore_release(event_semaphore);

  iree_hal_streaming_context_release_stream_snapshot(context, streams,
                                                     stream_count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_quiesce_all_for_teardown(
    iree_status_t* out_execution_status) {
  IREE_ASSERT_ARGUMENT(out_execution_status);
  *out_execution_status = iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_count = 0;
  iree_status_t status =
      iree_hal_streaming_context_snapshot_all(&contexts, &context_count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_status_t execution_status = iree_ok_status();
    status = iree_status_join(status,
                              iree_hal_streaming_context_quiesce_for_teardown(
                                  contexts[i], NULL, NULL, &execution_status));
    *out_execution_status =
        iree_status_join(*out_execution_status, execution_status);
  }

  iree_hal_streaming_context_release_snapshot_all(contexts, context_count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_flush(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t count = 0;
  iree_status_t status =
      iree_hal_streaming_context_snapshot_streams(context, &streams, &count);

  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    if (streams[i]) {
      status = iree_hal_streaming_stream_flush(streams[i]);
    }
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams, count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  if (context->default_stream) {
    status = iree_hal_streaming_stream_flush(context->default_stream);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_flush_all(void) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_capacity = 0;
  iree_host_size_t context_count = 0;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    ++context_capacity;
  }
  if (context_capacity > 0) {
    iree_host_size_t contexts_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            context_capacity, sizeof(contexts[0]), &contexts_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "context snapshot size overflow");
    } else {
      status = iree_allocator_malloc(device_registry->host_allocator,
                                     contexts_size, (void**)&contexts);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t index = 0;
    for (iree_hal_streaming_context_t* context =
             device_registry->context_list.head;
         context; context = context->context_list_entry.next) {
      contexts[index++] = context;
      iree_hal_streaming_context_retain(context);
    }
    context_count = index;
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  for (iree_host_size_t i = 0; i < context_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_context_flush(contexts[i]);
  }

  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  if (contexts) {
    iree_allocator_free(device_registry->host_allocator, contexts);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_context_synchronize_streams(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams,
    bool flush_before_wait) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (flush_before_wait) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_context_flush(context));
  }

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Synchronize streams from the retained snapshot. Legacy default stream
  // ordering excludes non-blocking streams, while device/context-wide
  // synchronization includes them.
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_streaming_stream_t* stream = streams_copy[i];
    if (!stream) continue;
    if (!include_non_blocking_streams &&
        (stream->flags & IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
    status = iree_hal_streaming_stream_synchronize_flushed(stream);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Also synchronize the default stream, which may not be in the streams list.
  // The legacy default stream always participates in its own ordering.
  if (context->default_stream) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_stream_synchronize_flushed(context->default_stream));
  }

  if (include_non_blocking_streams) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_context_synchronize_event_records(context));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_synchronize_event_records(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);

  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t value = 0;
  iree_slim_mutex_lock(&context->event_record_mutex);
  if (context->event_record_timeline.pending_value > 0) {
    semaphore = context->event_record_timeline.semaphore;
    value = context->event_record_timeline.pending_value;
    iree_hal_semaphore_retain(semaphore);
  }
  iree_slim_mutex_unlock(&context->event_record_mutex);

  iree_status_t status = iree_ok_status();
  if (semaphore) {
    status = iree_hal_semaphore_wait(semaphore, value, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context) {
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/true,
      /*flush_before_wait=*/true);
}

iree_status_t iree_hal_streaming_context_synchronize_legacy_default(
    iree_hal_streaming_context_t* context) {
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/false,
      /*flush_before_wait=*/true);
}

iree_status_t iree_hal_streaming_context_synchronize_all(void) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_capacity = 0;
  iree_host_size_t context_count = 0;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    ++context_capacity;
  }
  if (context_capacity > 0) {
    iree_host_size_t contexts_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            context_capacity, sizeof(contexts[0]), &contexts_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "context snapshot size overflow");
    } else {
      status = iree_allocator_malloc(device_registry->host_allocator,
                                     contexts_size, (void**)&contexts);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t index = 0;
    for (iree_hal_streaming_context_t* context =
             device_registry->context_list.head;
         context; context = context->context_list_entry.next) {
      contexts[index++] = context;
      iree_hal_streaming_context_retain(context);
    }
    context_count = index;
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < context_count;
       ++i) {
    status = iree_hal_streaming_context_flush(contexts[i]);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < context_count;
       ++i) {
    status = iree_hal_streaming_context_synchronize_streams(
        contexts[i], /*include_non_blocking_streams=*/true,
        /*flush_before_wait=*/false);
  }

  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  if (contexts) {
    iree_allocator_free(device_registry->host_allocator, contexts);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_streaming_context_is_current(
    const iree_hal_streaming_context_t* context) {
  if (!context || iree_atomic_load(&context->accepting_work,
                                   iree_memory_order_acquire) == 0) {
    return false;
  }
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_device_registry();
  // Common-layer tests and embedders may construct a context directly from an
  // unregistered device entry. Runtime generations start at one, so zero is
  // the explicit standalone domain with no process reset epoch to compare.
  if (!registry && context->runtime_generation == 0) return true;
  if (!registry ||
      context->runtime_generation != registry->runtime_generation ||
      context->device_ordinal >= registry->device_count) {
    return false;
  }
  const iree_hal_streaming_device_t* device =
      &registry->devices[context->device_ordinal];
  return context->device_epoch ==
         iree_atomic_load(&device->reset_epoch, iree_memory_order_acquire);
}

void iree_hal_streaming_context_retire(iree_hal_streaming_context_t* context) {
  if (context) {
    iree_atomic_store(&context->accepting_work, 0, iree_memory_order_release);
  }
}

void iree_hal_streaming_context_mark_teardown_quiesced(
    iree_hal_streaming_context_t* context) {
  if (!context) return;
  IREE_ASSERT(iree_atomic_load(&context->accepting_work,
                               iree_memory_order_acquire) == 0);
  iree_atomic_store(&context->teardown_quiesced, 1, iree_memory_order_release);
}

bool iree_hal_streaming_context_is_teardown_certified(
    iree_hal_streaming_context_t* context) {
  if (!context ||
      iree_atomic_load(&context->accepting_work, iree_memory_order_acquire) !=
          0 ||
      iree_atomic_load(&context->teardown_quiesced,
                       iree_memory_order_acquire) == 0 ||
      iree_atomic_load(&context->capture_stream_count,
                       iree_memory_order_acquire) != 0) {
    return false;
  }

  iree_slim_mutex_lock(&context->stream_list_mutex);
  const bool streams_detached = context->stream_count == 0 &&
                                context->default_stream == NULL &&
                                context->stream_wait_frontier == NULL;
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  return streams_detached;
}

void iree_hal_streaming_context_detach_streams_quiesced(
    iree_hal_streaming_context_t* context, bool abort_captures) {
  if (!context) return;
  IREE_ASSERT(iree_atomic_load(&context->accepting_work,
                               iree_memory_order_acquire) == 0);
  IREE_ASSERT(iree_atomic_load(&context->teardown_quiesced,
                               iree_memory_order_acquire) != 0);

  iree_slim_mutex_lock(&context->stream_list_mutex);
  const iree_host_size_t stream_count = context->stream_count;
  context->stream_count = 0;
  iree_hal_streaming_stream_t* default_stream = context->default_stream;
  context->default_stream = NULL;
  iree_hal_fence_t* stream_wait_frontier = context->stream_wait_frontier;
  context->stream_wait_frontier = NULL;
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    if (abort_captures) {
      iree_hal_streaming_stream_abort_capture_quiesced(stream);
    }
    iree_hal_streaming_stream_detach_quiesced(context, stream);
  }
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_release(context->streams[i]);
  }
  iree_hal_fence_release(stream_wait_frontier);
  iree_hal_streaming_stream_release(default_stream);
}

iree_status_t iree_hal_streaming_context_operation_begin(
    iree_hal_streaming_context_t* context) {
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "context is required");
  }
  if (context->lifecycle_begin) {
    return context->lifecycle_begin(context->lifecycle_user_data, context);
  }
  return iree_hal_streaming_context_is_current(context)
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "context belongs to a retired runtime epoch");
}

void iree_hal_streaming_context_operation_end(
    iree_hal_streaming_context_t* context) {
  if (context && context->lifecycle_end) {
    context->lifecycle_end(context->lifecycle_user_data);
  }
}

iree_status_t iree_hal_streaming_device_prepare_epoch_advance(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    uint64_t* out_expected_epoch, uint64_t* out_next_epoch) {
  IREE_ASSERT_ARGUMENT(out_expected_epoch);
  IREE_ASSERT_ARGUMENT(out_next_epoch);
  *out_expected_epoch = 0;
  *out_next_epoch = 0;
  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_ordinal);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid device ordinal");
  }
  const uint64_t expected_epoch =
      iree_atomic_load(&device->reset_epoch, iree_memory_order_acquire);
  // UINT64_MAX is the permanently invalid sentinel. Never publish it as a
  // successful next epoch: the final usable value is UINT64_MAX - 1 and a
  // reset starting there is already exhausted.
  if (expected_epoch >= UINT64_MAX - 1) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device reset epoch is exhausted");
  }
  *out_expected_epoch = expected_epoch;
  *out_next_epoch = expected_epoch + 1;
  return iree_ok_status();
}

void iree_hal_streaming_device_commit_epoch_advance(
    iree_hal_streaming_device_ordinal_t device_ordinal, uint64_t expected_epoch,
    uint64_t next_epoch) {
  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_ordinal);
  IREE_ASSERT(device);
  IREE_ASSERT(expected_epoch < UINT64_MAX - 1);
  IREE_ASSERT(next_epoch == expected_epoch + 1);
  IREE_ASSERT(next_epoch != UINT64_MAX);
  uint64_t observed_epoch = expected_epoch;
  const bool exchanged = iree_atomic_compare_exchange_strong(
      &device->reset_epoch, &observed_epoch, next_epoch,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
  IREE_ASSERT(exchanged,
              "device reset epoch changed after teardown preparation");
  if (IREE_UNLIKELY(!exchanged)) {
    iree_atomic_store(&device->reset_epoch, UINT64_MAX,
                      iree_memory_order_release);
  }
}

iree_status_t iree_hal_streaming_context_synchronize_device(
    iree_hal_streaming_device_ordinal_t device_ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_device_registry();
  if (!registry || device_ordinal >= registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid device ordinal");
  }

  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_count = 0;
  iree_host_size_t retained_count = 0;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context = registry->context_list.head;
       context; context = context->context_list_entry.next) {
    if (context->runtime_generation == registry->runtime_generation &&
        context->device_ordinal == device_ordinal) {
      ++context_count;
    }
  }
  if (context_count > 0) {
    iree_host_size_t contexts_size = 0;
    if (!iree_host_size_checked_mul(context_count, sizeof(contexts[0]),
                                    &contexts_size)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "context snapshot size overflow");
    } else {
      status = iree_allocator_malloc(registry->host_allocator, contexts_size,
                                     (void**)&contexts);
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_hal_streaming_context_t* context = registry->context_list.head;
         context; context = context->context_list_entry.next) {
      if (context->runtime_generation != registry->runtime_generation ||
          context->device_ordinal != device_ordinal) {
        continue;
      }
      contexts[retained_count++] = context;
      iree_hal_streaming_context_retain(context);
    }
    IREE_ASSERT(retained_count == context_count);
  }
  iree_slim_mutex_unlock(&registry->context_list.mutex);

  for (iree_host_size_t i = 0; i < retained_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_context_flush(contexts[i]);
  }
  for (iree_host_size_t i = 0; i < retained_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_context_wait_idle(contexts[i],
                                                  iree_infinite_timeout());
  }
  for (iree_host_size_t i = 0; i < retained_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  iree_allocator_free(registry->host_allocator, contexts);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_wait_blocking_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_host_size_t source_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_t* source_stream = streams_copy[i];
    if (!source_stream || source_stream == stream ||
        source_stream == context->default_stream ||
        iree_any_bit_set(source_stream->flags,
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING) ||
        source_stream->capture_status !=
            IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      continue;
    }
    // Partition selected sources into the front while preserving every
    // retained entry for snapshot release below.
    iree_hal_streaming_stream_t* displaced_stream = streams_copy[source_count];
    streams_copy[source_count++] = source_stream;
    streams_copy[i] = displaced_stream;
  }
  if (source_count > 0) {
    status = iree_hal_streaming_stream_wait_streams(stream, streams_copy,
                                                    source_count);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_query(
    iree_hal_streaming_context_t* context, int* status) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(status);
  *status = 0;

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t query_status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(query_status);
       ++i) {
    // The legacy default stream does not order with non-blocking streams.
    if (iree_any_bit_set(streams_copy[i]->flags,
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
    int stream_status = 0;
    query_status =
        iree_hal_streaming_stream_query(streams_copy[i], &stream_status);
    if (iree_status_is_ok(query_status) && stream_status != 0) {
      *status = 1;
      break;
    }
  }
  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  return query_status;
}

iree_status_t iree_hal_streaming_context_wait_all_submitted(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Wait for submitted work on all streams (doesn't flush).
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (streams_copy[i]) {
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_stream_wait_submitted(streams_copy[i]);
      }
    }
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // For the default stream, synchronize fully since caller needs it complete.
  if (context->default_stream) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(context->default_stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
