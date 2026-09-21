// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_file.h"

#include <string.h>

#include "iree/async/operations/file.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_staging.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

typedef enum iree_hal_amdgpu_file_action_kind_e {
  IREE_HAL_AMDGPU_FILE_ACTION_READ,
  IREE_HAL_AMDGPU_FILE_ACTION_WRITE,
} iree_hal_amdgpu_file_action_kind_t;

typedef enum iree_hal_amdgpu_file_action_io_state_e {
  IREE_HAL_AMDGPU_FILE_ACTION_IO_NONE = 0,
  IREE_HAL_AMDGPU_FILE_ACTION_IO_READING = 1,
  IREE_HAL_AMDGPU_FILE_ACTION_IO_WRITING = 2,
} iree_hal_amdgpu_file_action_io_state_t;

static iree_hal_profile_queue_event_type_t
iree_hal_amdgpu_file_action_profile_event_type(
    iree_hal_amdgpu_file_action_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_FILE_ACTION_READ
             ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
             : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE;
}

typedef struct iree_hal_amdgpu_file_action_state_t {
  // Resource header retained by the queue reclaim entry and async completion.
  iree_hal_resource_t resource;

  // Host allocator used for this state and cloned semaphore-list storage.
  iree_allocator_t host_allocator;

  // Serializes cancellation, proactor submission publication, and I/O state.
  iree_slim_mutex_t mutex;
  // Wakes seal after a submit handoff or terminal callback completes.
  iree_notification_t state_notification;
  // True after the terminal callback and all mapping cleanup has returned.
  iree_atomic_int32_t terminal_complete;
  // Number of threads handing an operation to the proactor.
  uint32_t io_submit_count;
  // Operation currently owned by the proactor, if any.
  iree_hal_amdgpu_file_action_io_state_t io_state;
  // Permanently set by queue seal before it snapshots/cancels the operation.
  bool cancel_requested;

  // Proactor used for async file I/O. Borrowed from the logical device.
  iree_async_proactor_t* proactor;

  // Queue used to publish final signal semaphores after async file I/O.
  iree_hal_amdgpu_host_queue_t* queue;

  // Pins the queue and logical device from active-registry publication until
  // the queue-owned terminal epilogue has consumed every callback-dependent
  // resource. Never released from a proactor callback.
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;

  // Queue-owned active-publisher registry membership. Protected by the queue
  // submission mutex.
  iree_hal_amdgpu_file_action_state_t* active_next;
  iree_hal_amdgpu_file_action_state_t** active_prev_next;
  bool is_registered;

  // File being read or written. Retained while the action is pending.
  iree_hal_file_t* file;

  // Async file handle borrowed from |file|.
  iree_async_file_t* async_file;

  // Buffer being read into or written from. Retained while the action is
  // pending.
  iree_hal_buffer_t* buffer;

  // File byte offset for the next transfer.
  uint64_t file_offset;

  // Buffer byte offset for the mapped range.
  iree_device_size_t buffer_offset;

  // Total requested transfer length.
  iree_host_size_t requested_length;

  // Number of wait semaphores supplied to the queue_read/write operation.
  uint32_t profile_wait_count;

  // Total bytes transferred by completed async file operations.
  iree_host_size_t completed_length;

  // Direction of the file action.
  iree_hal_amdgpu_file_action_kind_t kind;

  // Scoped mapping of |buffer| used by async file operations.
  iree_hal_buffer_mapping_t mapping;

  // Cloned signal list published by a final queue barrier after file I/O.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Completion-thread retry queued when the final signal barrier is blocked by
  // temporary queue capacity pressure.
  iree_hal_amdgpu_host_queue_post_drain_action_t signal_capacity_retry;

  // Queue-owned terminal epilogue. Proactor callbacks transfer status and
  // their self reference here as their final state/queue operation.
  iree_hal_amdgpu_host_queue_post_drain_action_t terminal_action;
  iree_status_t terminal_status;
  bool terminal_action_queued;

  // Async read operation reused across partial completions.
  iree_async_file_read_operation_t read_op;

  // Async write operation reused across partial completions.
  iree_async_file_write_operation_t write_op;
} iree_hal_amdgpu_file_action_state_t;

static void iree_hal_amdgpu_file_action_state_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(!state->lifetime_claim.queue_retained &&
                  !state->lifetime_claim.device_retained,
              "file action destroyed with a live queue/device claim");
  IREE_ASSERT(iree_status_is_ok(state->terminal_status),
              "file action destroyed with an unconsumed terminal status");
  if (!iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(state->signal_semaphore_list,
                                 state->host_allocator);
  }
  iree_hal_buffer_release(state->buffer);
  iree_hal_file_release(state->file);
  iree_notification_deinitialize(&state->state_notification);
  iree_slim_mutex_deinitialize(&state->mutex);
  iree_allocator_free(state->host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_file_action_state_vtable = {
        .destroy = iree_hal_amdgpu_file_action_state_destroy,
};

static bool iree_hal_amdgpu_file_action_is_terminal(void* user_data) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;
  return iree_atomic_load(&state->terminal_complete,
                          iree_memory_order_acquire) != 0;
}

static bool iree_hal_amdgpu_file_action_is_terminal_or_handed_off(
    void* user_data) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;
  if (iree_hal_amdgpu_file_action_is_terminal(state)) return true;
  iree_slim_mutex_lock(&state->mutex);
  const bool is_handed_off = state->terminal_action_queued;
  iree_slim_mutex_unlock(&state->mutex);
  return is_handed_off;
}

static bool iree_hal_amdgpu_file_action_io_submit_is_idle(void* user_data) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;
  iree_slim_mutex_lock(&state->mutex);
  const bool is_idle = state->io_submit_count == 0;
  iree_slim_mutex_unlock(&state->mutex);
  return is_idle;
}

static bool iree_hal_amdgpu_file_action_cancel_requested(
    iree_hal_amdgpu_file_action_state_t* state) {
  iree_slim_mutex_lock(&state->mutex);
  const bool cancel_requested = state->cancel_requested;
  iree_slim_mutex_unlock(&state->mutex);
  return cancel_requested;
}

// Replaces success or generic proactor cancellation with the queue's shutdown
// reason. A distinct operation-local error remains first and is returned
// unchanged. Takes ownership of |status| and returns ownership to the caller.
static iree_status_t iree_hal_amdgpu_file_action_resolve_cancellation_status(
    iree_hal_amdgpu_file_action_state_t* state, iree_status_t status) {
  const bool use_shutdown_status =
      iree_hal_amdgpu_file_action_cancel_requested(state) &&
      (iree_status_is_ok(status) ||
       iree_status_code(status) == IREE_STATUS_CANCELLED);
  if (use_shutdown_status) {
    iree_status_free(status);
    return iree_hal_amdgpu_host_queue_clone_shutdown_status(state->queue);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_file_action_register(
    iree_hal_amdgpu_file_action_state_t* state) {
  iree_hal_amdgpu_host_queue_t* queue = state->queue;
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
  IREE_ASSERT(!state->is_registered && !state->active_prev_next);
  iree_hal_resource_retain(&state->resource);
  state->active_next = queue->active_file_action_head;
  state->active_prev_next = &queue->active_file_action_head;
  if (state->active_next) {
    state->active_next->active_prev_next = &state->active_next;
  }
  queue->active_file_action_head = state;
  state->is_registered = true;
  state->lifetime_claim = lifetime_claim;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return iree_ok_status();
}

static void iree_hal_amdgpu_file_action_publish_terminal(
    iree_hal_amdgpu_file_action_state_t* state) {
  bool release_registry_ref = false;
  iree_hal_amdgpu_host_queue_t* queue = state->queue;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  if (state->is_registered) {
    *state->active_prev_next = state->active_next;
    if (state->active_next) {
      state->active_next->active_prev_next = state->active_prev_next;
    }
    state->active_next = NULL;
    state->active_prev_next = NULL;
    state->is_registered = false;
    release_registry_ref = true;
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  const int32_t previous = iree_atomic_exchange(&state->terminal_complete, 1,
                                                iree_memory_order_acq_rel);
  IREE_ASSERT(previous == 0, "file action completed more than once");
  iree_notification_post(&state->state_notification, IREE_ALL_WAITERS);
  if (release_registry_ref) {
    iree_hal_resource_release(&state->resource);
  }
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_direct_file_buffer(
    iree_hal_buffer_t* buffer, const char* operation_name,
    iree_device_size_t length) {
  if (IREE_UNLIKELY(length > (iree_device_size_t)IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s length %" PRIdsz
                            " exceeds host addressable size %" PRIhsz,
                            operation_name, length, IREE_HOST_SIZE_MAX);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                       IREE_HAL_MEMORY_TYPE_HOST_VISIBLE))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_%s for non-host-visible buffers requires bounded "
        "chunked staging",
        operation_name);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                       IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_%s for non-mappable buffers requires bounded chunked "
        "staging",
        operation_name);
  }
  return iree_ok_status();
}

static bool iree_hal_amdgpu_host_queue_file_buffer_supports_direct_io(
    iree_hal_buffer_t* buffer) {
  return iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                           IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                           IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED);
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_direct_file_handle(
    iree_hal_file_t* file, const char* operation_name) {
  if (IREE_UNLIKELY(!iree_hal_file_async_handle(file))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_%s for non-memory files requires a proactor-backed "
        "async file handle",
        operation_name);
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_file_action_fail_with_borrowed_status(
    iree_hal_amdgpu_file_action_state_t* state, const iree_status_t status) {
  if (iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    return;
  }
  iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                               iree_status_clone(status));
}

static void iree_hal_amdgpu_file_action_signal_capacity_post_drain(
    void* user_data);
static void iree_hal_amdgpu_file_action_terminal_post_drain(void* user_data);

// Publishes terminal state, consumes the caller-owned action reference, then
// releases the publisher lifetime claim. The claim is copied before the state
// releases because file/buffer finalizers may consume the last state reference.
static void iree_hal_amdgpu_file_action_finish_terminal(
    iree_hal_amdgpu_file_action_state_t* state) {
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim =
      state->lifetime_claim;
  memset(&state->lifetime_claim, 0, sizeof(state->lifetime_claim));
  iree_hal_amdgpu_file_action_publish_terminal(state);
  iree_hal_resource_release(&state->resource);
  (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
}

static iree_status_t iree_hal_amdgpu_file_action_submit_signal_barrier(
    iree_hal_amdgpu_file_action_state_t* state, bool* out_deferred) {
  *out_deferred = false;
  if (iree_hal_amdgpu_file_action_cancel_requested(state)) {
    return iree_hal_amdgpu_host_queue_clone_shutdown_status(state->queue);
  }
  if (iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_clone_error_status(state->queue));

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      state->queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PUBLISHER_SUBMISSION_REVALIDATION,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_PUBLISHER_SUBMISSION_LOCK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_DIRECT_FILE_SIGNAL_BARRIER,
      /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_lock(&state->queue->locks.submission_mutex);
  bool ready = false;
  uint64_t submission_id = 0;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
      .type = iree_hal_amdgpu_file_action_profile_event_type(state->kind),
      .payload_length = state->requested_length,
      .operation_count = 1,
  };
  iree_status_t status =
      iree_hal_amdgpu_host_queue_revalidate_submission_locked(state->queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_submit_barrier(
        state->queue, &resolution, state->signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){0},
        /*operation_resources=*/NULL, /*operation_resource_count=*/0,
        &profile_event_info,
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready,
        &submission_id);
  }
  if (iree_status_is_ok(status) && ready) {
    iree_hal_amdgpu_wait_resolution_t profile_resolution = resolution;
    profile_resolution.wait_count = state->profile_wait_count;
    profile_event_info.submission_id = submission_id;
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        state->queue, &profile_resolution, state->signal_semaphore_list,
        &profile_event_info);
  }
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&state->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        state->queue, &state->signal_capacity_retry,
        iree_hal_amdgpu_file_action_signal_capacity_post_drain, state);
    *out_deferred = true;
  }
  iree_slim_mutex_unlock(&state->queue->locks.submission_mutex);
  return status;
}

static void iree_hal_amdgpu_file_action_signal_capacity_post_drain(
    void* user_data) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;
  bool deferred = false;
  iree_status_t status =
      iree_hal_amdgpu_file_action_submit_signal_barrier(state, &deferred);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }
  if (!deferred) {
    iree_hal_amdgpu_file_action_finish_terminal(state);
  } else {
    iree_hal_resource_release(&state->resource);
  }
}

static void iree_hal_amdgpu_file_action_complete_safe(
    iree_hal_amdgpu_file_action_state_t* state, iree_status_t status) {
  const bool cancel_requested =
      iree_hal_amdgpu_file_action_cancel_requested(state);
  status =
      iree_hal_amdgpu_file_action_resolve_cancellation_status(state, status);
  if (iree_status_is_ok(status) &&
      state->kind == IREE_HAL_AMDGPU_FILE_ACTION_READ &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(state->mapping.buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_status_join(
        status, iree_hal_buffer_mapping_flush_range(&state->mapping, 0,
                                                    state->requested_length));
  }
  if (state->mapping.buffer) {
    iree_status_t unmap_status = iree_hal_buffer_unmap_range(&state->mapping);
    if (cancel_requested) {
      iree_status_free(unmap_status);
    } else {
      status = iree_status_join(status, unmap_status);
    }
  }

  bool deferred = false;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_file_action_submit_signal_barrier(state, &deferred);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }
  if (!deferred) {
    iree_hal_amdgpu_file_action_finish_terminal(state);
  } else {
    iree_hal_resource_release(&state->resource);
  }
}

static void iree_hal_amdgpu_file_action_terminal_post_drain(void* user_data) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;
  iree_slim_mutex_lock(&state->mutex);
  IREE_ASSERT(state->terminal_action_queued);
  state->terminal_action_queued = false;
  iree_status_t status = state->terminal_status;
  state->terminal_status = iree_ok_status();
  iree_slim_mutex_unlock(&state->mutex);
  iree_hal_amdgpu_file_action_complete_safe(state, status);
}

// Transfers terminal work from a proactor callback to queue-owned execution.
// The enqueue call is the callback's final state/queue access; the completion
// service or external sealer owns every potentially-final release afterward.
static void iree_hal_amdgpu_file_action_schedule_terminal(
    iree_hal_amdgpu_file_action_state_t* state, iree_status_t status) {
  iree_slim_mutex_lock(&state->mutex);
  IREE_ASSERT(!state->terminal_action_queued,
              "file action terminal epilogue scheduled twice");
  state->terminal_status = status;
  state->terminal_action_queued = true;
  iree_slim_mutex_unlock(&state->mutex);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action_and_notify(
      state->queue, &state->terminal_action,
      iree_hal_amdgpu_file_action_terminal_post_drain, state,
      &state->state_notification);
}

static iree_status_t iree_hal_amdgpu_file_action_submit_next_read(
    iree_hal_amdgpu_file_action_state_t* state);

static iree_status_t iree_hal_amdgpu_file_action_submit_next_write(
    iree_hal_amdgpu_file_action_state_t* state);

static void iree_hal_amdgpu_file_action_read_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;

  iree_slim_mutex_lock(&state->mutex);
  state->io_state = IREE_HAL_AMDGPU_FILE_ACTION_IO_NONE;
  iree_slim_mutex_unlock(&state->mutex);
  status =
      iree_hal_amdgpu_file_action_resolve_cancellation_status(state, status);

  bool should_complete = true;
  if (iree_status_is_ok(status) && state->read_op.bytes_read > 0) {
    state->completed_length += state->read_op.bytes_read;
    if (state->completed_length < state->requested_length) {
      status = iree_hal_amdgpu_file_action_submit_next_read(state);
      should_complete = !iree_status_is_ok(status);
    }
  } else if (iree_status_is_ok(status) &&
             state->completed_length < state->requested_length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short read: requested %" PRIhsz
                              " bytes, got %" PRIhsz,
                              state->requested_length, state->completed_length);
  }

  if (should_complete) {
    iree_hal_amdgpu_file_action_schedule_terminal(state, status);
  }
}

static void iree_hal_amdgpu_file_action_write_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;

  iree_slim_mutex_lock(&state->mutex);
  state->io_state = IREE_HAL_AMDGPU_FILE_ACTION_IO_NONE;
  iree_slim_mutex_unlock(&state->mutex);
  status =
      iree_hal_amdgpu_file_action_resolve_cancellation_status(state, status);

  bool should_complete = true;
  if (iree_status_is_ok(status) && state->write_op.bytes_written > 0) {
    state->completed_length += state->write_op.bytes_written;
    if (state->completed_length < state->requested_length) {
      status = iree_hal_amdgpu_file_action_submit_next_write(state);
      should_complete = !iree_status_is_ok(status);
    }
  } else if (iree_status_is_ok(status) &&
             state->completed_length < state->requested_length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short write: requested %" PRIhsz
                              " bytes, wrote %" PRIhsz,
                              state->requested_length, state->completed_length);
  }

  if (should_complete) {
    iree_hal_amdgpu_file_action_schedule_terminal(state, status);
  }
}

static iree_status_t iree_hal_amdgpu_file_action_submit_next_read(
    iree_hal_amdgpu_file_action_state_t* state) {
  const iree_host_size_t remaining_length =
      state->requested_length - state->completed_length;
  iree_slim_mutex_lock(&state->mutex);
  if (state->cancel_requested) {
    iree_slim_mutex_unlock(&state->mutex);
    return iree_hal_amdgpu_host_queue_clone_shutdown_status(state->queue);
  }
  ++state->io_submit_count;
  state->io_state = IREE_HAL_AMDGPU_FILE_ACTION_IO_READING;
  iree_async_operation_zero(&state->read_op.base, sizeof(state->read_op));
  iree_async_operation_initialize(
      &state->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amdgpu_file_action_read_complete,
      state);
  state->read_op.file = state->async_file;
  state->read_op.offset = state->file_offset + state->completed_length;
  state->read_op.buffer = iree_async_span_from_ptr(
      state->mapping.contents.data + state->completed_length, remaining_length);
  iree_slim_mutex_unlock(&state->mutex);
  iree_status_t status =
      iree_async_proactor_submit_one(state->proactor, &state->read_op.base);
  iree_slim_mutex_lock(&state->mutex);
  IREE_ASSERT(state->io_submit_count > 0);
  --state->io_submit_count;
  if (!iree_status_is_ok(status)) {
    state->io_state = IREE_HAL_AMDGPU_FILE_ACTION_IO_NONE;
  }
  const bool submit_tail_is_idle = state->io_submit_count == 0;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        state->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_DIRECT_READ_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_ZERO_BEFORE_WAKE,
        /*value0=*/0, /*value1=*/0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  // Publish the zero submit-tail predicate and its wake under one lock. Seal's
  // predicate takes this same mutex and cannot free operation/notification
  // storage in the old zero-before-post window.
  iree_notification_post(&state->state_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        state->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_DIRECT_READ_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_LEFT,
        /*value0=*/0, /*value1=*/0);
  }
#else
  (void)submit_tail_is_idle;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&state->mutex);
  return status;
}

static iree_status_t iree_hal_amdgpu_file_action_submit_next_write(
    iree_hal_amdgpu_file_action_state_t* state) {
  const iree_host_size_t remaining_length =
      state->requested_length - state->completed_length;
  iree_slim_mutex_lock(&state->mutex);
  if (state->cancel_requested) {
    iree_slim_mutex_unlock(&state->mutex);
    return iree_hal_amdgpu_host_queue_clone_shutdown_status(state->queue);
  }
  ++state->io_submit_count;
  state->io_state = IREE_HAL_AMDGPU_FILE_ACTION_IO_WRITING;
  iree_async_operation_zero(&state->write_op.base, sizeof(state->write_op));
  iree_async_operation_initialize(
      &state->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_amdgpu_file_action_write_complete, state);
  state->write_op.file = state->async_file;
  state->write_op.offset = state->file_offset + state->completed_length;
  state->write_op.buffer = iree_async_span_from_ptr(
      state->mapping.contents.data + state->completed_length, remaining_length);
  iree_slim_mutex_unlock(&state->mutex);
  iree_status_t status =
      iree_async_proactor_submit_one(state->proactor, &state->write_op.base);
  iree_slim_mutex_lock(&state->mutex);
  IREE_ASSERT(state->io_submit_count > 0);
  --state->io_submit_count;
  if (!iree_status_is_ok(status)) {
    state->io_state = IREE_HAL_AMDGPU_FILE_ACTION_IO_NONE;
  }
  const bool submit_tail_is_idle = state->io_submit_count == 0;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        state->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_DIRECT_WRITE_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_ZERO_BEFORE_WAKE,
        /*value0=*/0, /*value1=*/0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_notification_post(&state->state_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (submit_tail_is_idle) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        state->queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_DIRECT_WRITE_SUBMIT,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_LEFT,
        /*value0=*/0, /*value1=*/0);
  }
#else
  (void)submit_tail_is_idle;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&state->mutex);
  return status;
}

static iree_status_t iree_hal_amdgpu_file_action_start_async(
    iree_hal_amdgpu_file_action_state_t* state) {
  // The host action owns |state| only through this call. Keep an independent
  // self reference across the async callback and publish a queue-owned edge
  // before any publisher can escape this thread.
  iree_hal_resource_retain(&state->resource);
  iree_status_t status = iree_hal_amdgpu_file_action_register(state);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&state->resource);
    return status;
  }

  iree_hal_memory_access_t mapping_access = IREE_HAL_MEMORY_ACCESS_READ;
  if (state->kind == IREE_HAL_AMDGPU_FILE_ACTION_READ) {
    mapping_access = IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;
  }
  status = iree_hal_buffer_map_range(
      state->buffer, IREE_HAL_MAPPING_MODE_SCOPED, mapping_access,
      state->buffer_offset, state->requested_length, &state->mapping);

  if (iree_status_is_ok(status) &&
      state->kind == IREE_HAL_AMDGPU_FILE_ACTION_WRITE &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(state->mapping.buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_invalidate_range(&state->mapping, 0,
                                                      state->requested_length);
  }

  status =
      iree_hal_amdgpu_file_action_resolve_cancellation_status(state, status);

  if (iree_status_is_ok(status)) {
    if (state->kind == IREE_HAL_AMDGPU_FILE_ACTION_READ) {
      status = iree_hal_amdgpu_file_action_submit_next_read(state);
    } else {
      status = iree_hal_amdgpu_file_action_submit_next_write(state);
    }
    if (iree_status_is_ok(status)) {
      return iree_ok_status();
    }
  }

  // Mapping cleanup, semaphore dispatch, imported file/buffer release, and the
  // queue/device publisher claim all move to the safe epilogue. Return OK so
  // the reclaim action does not also fail the same signal list.
  iree_hal_amdgpu_file_action_schedule_terminal(state, status);
  return iree_ok_status();
}

static void iree_hal_amdgpu_file_action_execute(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;

  if (iree_status_is_ok(status)) {
    iree_status_t start_status = iree_hal_amdgpu_file_action_start_async(state);
    if (!iree_status_is_ok(start_status)) {
      iree_hal_semaphore_list_fail(state->signal_semaphore_list, start_status);
    }
  } else {
    iree_hal_amdgpu_file_action_fail_with_borrowed_status(state, status);
  }
}

static iree_status_t iree_hal_amdgpu_file_action_state_create(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_file_action_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t profile_wait_count,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_file_action_state_t** out_state) {
  *out_state = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  iree_hal_amdgpu_file_action_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, sizeof(*state),
                                (void**)&state));
  memset(state, 0, sizeof(*state));
  iree_hal_resource_initialize(&iree_hal_amdgpu_file_action_state_vtable,
                               &state->resource);
  state->host_allocator = queue->host_allocator;
  iree_slim_mutex_initialize(&state->mutex);
  iree_notification_initialize(&state->state_notification);
  iree_atomic_store(&state->terminal_complete, 0, iree_memory_order_relaxed);
  state->proactor = queue->proactor;
  state->queue = queue;
  state->file = file;
  iree_hal_file_retain(state->file);
  state->async_file = iree_hal_file_async_handle(file);
  state->buffer = buffer;
  iree_hal_buffer_retain(state->buffer);
  state->file_offset = file_offset;
  state->buffer_offset = buffer_offset;
  state->requested_length = (iree_host_size_t)length;
  state->profile_wait_count = profile_wait_count;
  state->kind = kind;

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, state->host_allocator,
      &state->signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_state = state;
  } else {
    iree_hal_resource_release(&state->resource);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_file_action_request_cancel(
    iree_hal_amdgpu_file_action_state_t* state) {
  iree_slim_mutex_lock(&state->mutex);
  IREE_ASSERT(!state->cancel_requested);
  state->cancel_requested = true;
  iree_slim_mutex_unlock(&state->mutex);

  // A submitter that passed the cancellation check owns the operation storage
  // until submit returns. Wait without holding action/queue locks so callbacks
  // and submitters can make progress.
  iree_notification_await(&state->state_notification,
                          iree_hal_amdgpu_file_action_io_submit_is_idle, state,
                          iree_infinite_timeout());

  const bool can_cancel_file_ops = iree_any_bit_set(
      iree_async_proactor_query_capabilities(state->proactor),
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  if (can_cancel_file_ops) {
    iree_async_operation_t* operation = NULL;
    iree_slim_mutex_lock(&state->mutex);
    if (state->io_state == IREE_HAL_AMDGPU_FILE_ACTION_IO_READING) {
      operation = &state->read_op.base;
    } else if (state->io_state == IREE_HAL_AMDGPU_FILE_ACTION_IO_WRITING) {
      operation = &state->write_op.base;
    }
    iree_slim_mutex_unlock(&state->mutex);
    if (operation) {
      iree_status_ignore(
          iree_async_proactor_cancel(state->proactor, operation));
    }
  }
}

void iree_hal_amdgpu_file_action_cancel_all(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);

  // Atomically detach the exact active set after queue admission has closed.
  // Each detached node keeps the queue-owned registry reference installed at
  // start so callbacks may race without invalidating this traversal.
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->is_shutting_down,
              "queue admission must close before publisher cancellation");
  iree_hal_amdgpu_file_action_state_t* action = queue->active_file_action_head;
  queue->active_file_action_head = NULL;
  IREE_ASSERT(queue->shutdown_file_action_head == NULL,
              "publisher cancellation can begin only once");
  queue->shutdown_file_action_head = action;
  for (iree_hal_amdgpu_file_action_state_t* current = action; current;
       current = current->active_next) {
    current->is_registered = false;
    current->active_prev_next = NULL;
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  for (iree_hal_amdgpu_file_action_state_t* current = action; current;
       current = current->active_next) {
    iree_hal_amdgpu_file_action_request_cancel(current);
  }
}

void iree_hal_amdgpu_file_action_await_all(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_hal_amdgpu_file_action_state_t* action =
      queue->shutdown_file_action_head;
  while (action) {
    iree_hal_amdgpu_file_action_state_t* next = action->active_next;
    action->active_next = NULL;
    while (!iree_hal_amdgpu_file_action_is_terminal(action)) {
      iree_notification_await(
          &action->state_notification,
          iree_hal_amdgpu_file_action_is_terminal_or_handed_off, action,
          iree_infinite_timeout());
      if (!iree_hal_amdgpu_file_action_is_terminal(action)) {
        // The completion service is already joined on this seal path. Claim
        // the normal runner and consume the handed-off terminal action from
        // the external sealer; this also joins any runner that won the race.
        iree_hal_amdgpu_host_queue_drain_completions(queue);
      }
    }
    // Consume the queue-owned registry edge detached above.
    iree_hal_resource_release(&action->resource);
    action = next;
  }
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  queue->shutdown_file_action_head = NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_direct_file_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_file_action_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  iree_hal_amdgpu_file_action_state_t* state = NULL;
  const uint32_t profile_wait_count =
      iree_hal_amdgpu_host_queue_profile_semaphore_count(wait_semaphore_list);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_file_action_state_create(
      queue, kind, file, file_offset, buffer, buffer_offset, length,
      profile_wait_count, signal_semaphore_list, &state));

  iree_hal_resource_t* resources[1] = {&state->resource};
  iree_status_t status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_file_action_execute,
          .user_data = state,
      },
      resources, IREE_ARRAYSIZE(resources));
  iree_hal_resource_release(&state->resource);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_read_file(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  (void)flags;

  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  if (!storage_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_direct_file_handle(
        source_file, "read"));
    if (iree_hal_amdgpu_host_queue_file_buffer_supports_direct_io(
            target_buffer)) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_host_queue_validate_direct_file_buffer(
              target_buffer, "read", length));
      return iree_hal_amdgpu_host_queue_submit_direct_file_action(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_FILE_ACTION_READ, source_file, source_offset,
          target_buffer, target_offset, length);
    }
    return iree_hal_amdgpu_host_queue_submit_staged_read(
        queue, wait_semaphore_list, signal_semaphore_list, source_file,
        source_offset, target_buffer, target_offset, length);
  }
  IREE_ASSERT(source_offset <= IREE_DEVICE_SIZE_MAX);
  return iree_hal_amdgpu_host_queue_copy_buffer(
      queue, wait_semaphore_list, signal_semaphore_list, storage_buffer,
      (iree_device_size_t)source_offset, target_buffer, target_offset, length,
      IREE_HAL_COPY_FLAG_NONE, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ);
}

iree_status_t iree_hal_amdgpu_host_queue_write_file(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  (void)flags;

  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(target_file);
  if (!storage_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_direct_file_handle(
        target_file, "write"));
    if (iree_hal_amdgpu_host_queue_file_buffer_supports_direct_io(
            source_buffer)) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_host_queue_validate_direct_file_buffer(
              source_buffer, "write", length));
      return iree_hal_amdgpu_host_queue_submit_direct_file_action(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_FILE_ACTION_WRITE, target_file, target_offset,
          source_buffer, source_offset, length);
    }
    return iree_hal_amdgpu_host_queue_submit_staged_write(
        queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
        source_offset, target_file, target_offset, length);
  }
  IREE_ASSERT(target_offset <= IREE_DEVICE_SIZE_MAX);
  return iree_hal_amdgpu_host_queue_copy_buffer(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, storage_buffer, (iree_device_size_t)target_offset, length,
      IREE_HAL_COPY_FLAG_NONE, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE);
}
