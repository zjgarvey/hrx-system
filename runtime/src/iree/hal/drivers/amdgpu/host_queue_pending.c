// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending_operation.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"

//===----------------------------------------------------------------------===//
// Pending operations (deferred submission)
//===----------------------------------------------------------------------===//
//
// LOCKING PROTOCOL
//
// submission_mutex protects all submission-path state: AQL ring reservation,
// kernarg allocation, packet emission, commit_signals, frontier mutation,
// notification ring push, and the pending list (link/unlink).
//
// The completion thread (drain, error check) does NOT acquire
// submission_mutex. It reads the notification ring (SPSC consumer) and the
// atomic error_status.
//
// Deferred operations use a two-phase protocol:
//
//   Phase 1 (under submission_mutex): resolve_waits, allocate pending_op,
//     capture operation parameters, link to pending list.
//
//   Phase 2 (WITHOUT submission_mutex): register timepoints via enqueue_waits.
//     Timepoint callbacks may fire synchronously during acquire_timepoint
//     (when the semaphore value is already reached or the semaphore is already
//     failed). The last callback to fire calls pending_op_issue or
//     pending_op_fail, both of which acquire submission_mutex internally.
//     This is safe because Phase 1 released the mutex before Phase 2 began.
//
// pending_op_issue: acquires submission_mutex to emit AQL packets, transfer
//   retained resources to the reclaim ring, and atomically unlink. It releases
//   the mutex before invoking callbacks or releasing any retained object.
//
// pending_op_fail: acquires submission_mutex to unlink. Semaphore failure
//   and resource release happen outside the lock.
//
// pending_op_discard_under_lock: for capture-time failures (arena allocation
//   errors after pending_op_allocate). Caller already holds submission_mutex.
//   Unlinks under the mutex, temporarily unlocks for callback-capable cleanup,
//   and re-acquires the mutex before returning to its caller.

// Per-wait timepoint entry, arena-allocated one per unsatisfied wait. The
// timepoint callback decrements the operation's atomic wait counter; the last
// callback to fire issues or fails the operation.
struct iree_hal_amdgpu_wait_entry_t {
  // Async semaphore timepoint registration owned by this wait entry.
  iree_async_semaphore_timepoint_t timepoint;
  // Pending operation whose wait_count is decremented by this callback.
  iree_hal_amdgpu_pending_op_t* operation;
  // Set to 1 after the callback's final access to this entry/op completes.
  // Queue shutdown spins on this for callbacks that were already detached from
  // the semaphore before cancel_timepoint() ran.
  iree_atomic_int32_t callback_complete;
};

typedef enum iree_hal_amdgpu_alloca_memory_wait_kind_e {
  // No active memory wait or held reservation transaction.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE = 0,
  // Waiting for a copied pool death frontier while holding reservations.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  // Performing cold pool backing growth before retrying reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_GROWTH = 2,
  // Waiting for a pool release notification before retrying reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 3,
} iree_hal_amdgpu_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside a pending op's arena
// only after user semaphore waits have resolved and the pool cannot produce
// immediately-usable bytes.
struct iree_hal_amdgpu_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_amdgpu_alloca_memory_wait_kind_t kind;

  // Set to 1 after the callback's final access to this wait/op completes.
  iree_atomic_int32_t callback_complete;

  // Held-reservation wait state blocked on a pool death frontier.
  struct {
    // Tracker waiter storage for the transaction's merged wait frontier.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // Cold pool backing-growth retry state for reservation attempts.
  struct {
    // Queue-order frontier snapshot used for cold reservation pre-growth.
    iree_hal_amdgpu_fixed_frontier_t requester_frontier;

  } pool_growth;

  // Pool notification retry state for reservation attempts.
  struct {
    // Borrowed notification returned by the pool.
    iree_async_notification_t* notification;

    // Notification epoch observed before the reservation retry.
    uint32_t wait_token;

    // Whether the pre-submit observation scope is still held. Once submit
    // returns, the submitted wait operation owns its own observation scope and
    // this bridge scope is released.
    bool pre_submit_observation_held;

    // Wait operations rotated so a callback can arm a retry before returning.
    iree_async_notification_wait_operation_t wait_ops[2];

    // Index of the active wait operation in |wait_ops|.
    uint8_t wait_slot;
  } pool_notification;
};

static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op);
static void iree_hal_amdgpu_pending_op_capacity_post_drain(void* user_data);
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status);
// Links a pending op into the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_link(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  op->next = queue->pending_head;
  op->prev_next = &queue->pending_head;
  if (queue->pending_head) {
    queue->pending_head->prev_next = &op->next;
  }
  queue->pending_head = op;
}

// Unlinks a pending op from the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_unlink(
    iree_hal_amdgpu_pending_op_t* op) {
  *op->prev_next = op->next;
  if (op->next) {
    op->next->prev_next = op->prev_next;
  }
  op->next = NULL;
  op->prev_next = NULL;
}

// Retains a resource and appends it to the pending op's retained_resources
// array. The caller must have allocated sufficient capacity in the array
// via the max_resource_count parameter to pending_op_allocate.
void iree_hal_amdgpu_pending_op_retain(iree_hal_amdgpu_pending_op_t* op,
                                       iree_hal_resource_t* resource) {
  if (IREE_LIKELY(resource)) {
    iree_hal_resource_retain(resource);
    op->retained_resources[op->retained_resource_count++] = resource;
  }
}

// Releases all retained HAL resources in the flat array. Used on failure,
// cancellation, and success paths where the submit helper retained the
// resources it needs instead of consuming this pending op's refs.
void iree_hal_amdgpu_pending_op_release_retained(
    iree_hal_amdgpu_pending_op_t* op) {
  for (uint16_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  op->retained_resource_count = 0;
}

static void iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type == IREE_HAL_AMDGPU_PENDING_OP_EXECUTE) {
    iree_hal_resource_set_free(op->execute.binding_resource_set);
    op->execute.binding_resource_set = NULL;
  }
}

static void iree_hal_amdgpu_pending_op_fail_host_action(
    iree_hal_amdgpu_pending_op_t* op, const iree_status_t status) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION ||
      !op->host_action.action.fn) {
    return;
  }
  op->host_action.action.fn(/*entry=*/NULL, op->host_action.action.user_data,
                            status);
  op->host_action.action.fn = NULL;
  op->host_action.action.user_data = NULL;
}

// Releases any queue-owned alloca memory-readiness reservation. This runs only
// on failure/cancellation paths or after ownership has not transferred into the
// transient buffer.
static void iree_hal_amdgpu_pending_op_release_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (wait) wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  iree_hal_amdgpu_host_queue_release_alloca_transaction(
      op->alloca_op.pool, &op->alloca_op.transaction);
}

// Clears the queued marker for a deferred dealloca that never published a
// completion epoch. Successful deallocas transfer ownership to the reclaim ring
// and must not call this.
static void iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA) return;
  for (iree_host_size_t i = 0; i < op->dealloca.transaction.buffer_count; ++i) {
    iree_hal_amdgpu_transient_buffer_abort_dealloca(
        op->dealloca.transaction.buffers[i]);
  }
}

// Destroys a terminal pending operation after it has been unlinked. No queue
// lock may be held: arbitrary host actions, semaphore timepoint callbacks, and
// final resource releases can synchronously reenter this or another queue.
static void iree_hal_amdgpu_pending_op_destroy_unlocked(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  IREE_ASSERT(op->has_submission_epilogue_token);
  op->has_submission_epilogue_token = false;
  const bool failed = !iree_status_is_ok(status);
  if (failed) {
    iree_hal_amdgpu_pending_op_fail_host_action(op, status);
    // Failure callbacks may immediately resolve another pending operation on
    // this queue, so this must remain outside submission_mutex.
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
    iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  }
  iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
  // Successful payload helpers clear resources whose ownership transferred to
  // the reclaim ring. Execute and host-call helpers leave their independently
  // retained capture resources here for post-submit release.
  iree_hal_amdgpu_pending_op_release_retained(op);
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_notification_deinitialize(&op->callback_notification);
  iree_slim_mutex_deinitialize(&op->callback_mutex);
  iree_arena_deinitialize(&op->arena);
  // The op and arena are gone. Releasing the scalar token is the terminal
  // queue access for this cleanup scope; a failed lifetime claim means the
  // destructor waiting on this token may now reclaim queue storage.
  iree_hal_amdgpu_host_queue_leave_submission_epilogue(queue);
}

// Cleans up an operation that failed while being captured. Signal semaphores
// are deliberately not failed because the queue operation was never accepted.
// As with terminal cleanup, this must run without submission_mutex held.
static void iree_hal_amdgpu_pending_op_discard_unlocked(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
  iree_hal_amdgpu_pending_op_release_retained(op);
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_notification_deinitialize(&op->callback_notification);
  iree_slim_mutex_deinitialize(&op->callback_mutex);
  iree_arena_deinitialize(&op->arena);
}

static bool iree_hal_amdgpu_alloca_memory_wait_callback_is_complete(
    void* user_data) {
  iree_hal_amdgpu_pending_op_t* op = (iree_hal_amdgpu_pending_op_t*)user_data;
  iree_slim_mutex_lock(&op->callback_mutex);
  const bool is_complete =
      iree_atomic_load(&op->alloca_op.memory_wait->callback_complete,
                       iree_memory_order_acquire) != 0;
  iree_slim_mutex_unlock(&op->callback_mutex);
  return is_complete;
}

static void iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
    iree_hal_amdgpu_pending_op_t* op, bool is_cancelling_loser) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_slim_mutex_lock(&op->callback_mutex);
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_release);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  const iree_hal_amdgpu_host_queue_test_subject_t test_subject =
      is_cancelling_loser
          ? IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_CANCELLING_LOSER
          : IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_ALLOCA_CALLBACK;
  iree_hal_amdgpu_host_queue_test_notify_phase(
      op->queue, test_subject,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE,
      /*value0=*/0, /*value1=*/0);
#else
  (void)is_cancelling_loser;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_notification_post(&op->callback_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      op->queue, test_subject, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  // Final access for a callback that did not claim lifecycle completion.
  iree_slim_mutex_unlock(&op->callback_mutex);
}

// Publishes a prepared memory-readiness wait as ARMING. The release store on
// lifecycle_state makes the initialized sidecar fields visible to the callback
// or cancellation path that observes the state transition.
static void iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t* wait,
    iree_hal_amdgpu_alloca_memory_wait_kind_t kind) {
  wait->kind = kind;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT,
                    iree_memory_order_release);
}

static iree_status_t iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t** out_wait) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) {
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&op->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
    op->alloca_op.memory_wait = wait;
    IREE_TRACE_ZONE_END(z0);
  }
  *out_wait = wait;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (IREE_UNLIKELY(op->alloca_op.transaction.wait_frontier->entry_count ==
                    0)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca waitable pool reservation did not provide a frontier");
  }

  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_pool_growth(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_async_frontier_t* requester_frontier) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_async_frontier_t* growth_frontier =
      iree_hal_amdgpu_fixed_frontier_as_frontier(
          &wait->pool_growth.requester_frontier);
  iree_async_frontier_initialize(growth_frontier,
                                 requester_frontier->entry_count);
  memcpy(
      growth_frontier->entries, requester_frontier->entries,
      requester_frontier->entry_count * sizeof(requester_frontier->entries[0]));
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_GROWTH);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op, iree_async_notification_t* notification,
    uint32_t wait_token) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  wait->pool_notification.notification = notification;
  wait->pool_notification.wait_token = wait_token;
  wait->pool_notification.pre_submit_observation_held = true;
  wait->pool_notification.wait_slot =
      (uint8_t)((wait->pool_notification.wait_slot + 1u) & 1u);
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION);
  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_pool_notification_end_observe(
    iree_hal_amdgpu_alloca_memory_wait_t* wait) {
  if (wait->pool_notification.pre_submit_observation_held) {
    wait->pool_notification.pre_submit_observation_held = false;
    iree_async_notification_end_observe(wait->pool_notification.notification);
  }
}

// Cancels any active alloca memory-readiness wait before destroying the op.
static void iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER: {
      const bool cancelled = iree_async_frontier_tracker_cancel_wait(
          op->queue->frontier_tracker, &wait->frontier.waiter);
      if (!cancelled) {
        iree_notification_await(
            &op->callback_notification,
            iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, op,
            iree_infinite_timeout());
      }
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION: {
      iree_hal_amdgpu_alloca_pool_notification_end_observe(wait);
      // Shutdown is allowed to prod the pool notification: it is a broad wake,
      // but prevents teardown from depending on a future dealloca. The callback
      // observes the CANCELLING lifecycle state and only publishes completion.
      iree_async_notification_signal(wait->pool_notification.notification,
                                     INT32_MAX);
      iree_notification_await(
          &op->callback_notification,
          iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, op,
          iree_infinite_timeout());
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_GROWTH:
      iree_hal_amdgpu_host_queue_release_alloca_transaction(
          op->alloca_op.pool, &op->alloca_op.transaction);
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static bool iree_hal_amdgpu_wait_entry_callback_is_complete(void* user_data) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;
  iree_slim_mutex_lock(&op->callback_mutex);
  const bool is_complete = iree_atomic_load(&entry->callback_complete,
                                            iree_memory_order_acquire) != 0;
  iree_slim_mutex_unlock(&op->callback_mutex);
  return is_complete;
}

static void iree_hal_amdgpu_wait_entry_publish_callback_complete(
    iree_hal_amdgpu_wait_entry_t* entry, bool is_cancelling_loser) {
  iree_hal_amdgpu_pending_op_t* op = entry->operation;
  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  const iree_hal_amdgpu_host_queue_test_subject_t test_subject =
      is_cancelling_loser
          ? IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_CANCELLING_LOSER
          : IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK;
  iree_hal_amdgpu_host_queue_test_notify_phase(
      op->queue, test_subject,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE,
      /*value0=*/0, /*value1=*/0);
#else
  (void)is_cancelling_loser;
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_notification_post(&op->callback_notification, IREE_ALL_WAITERS);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      op->queue, test_subject, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  // Final access for a callback that did not claim lifecycle completion.
  iree_slim_mutex_unlock(&op->callback_mutex);
}

static bool iree_hal_amdgpu_pending_op_wait_callbacks_are_complete(
    void* user_data) {
  iree_hal_amdgpu_pending_op_t* op = (iree_hal_amdgpu_pending_op_t*)user_data;
  for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    if (!iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) {
      return false;
    }
  }
  return true;
}

// Records the first asynchronous wait failure. Takes ownership of |status|,
// storing it for the completion owner or dropping it if another failure won.
static void iree_hal_amdgpu_pending_op_record_error_status(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &op->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static void iree_hal_amdgpu_pending_op_complete_resolved_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_notification_await(
      &op->callback_notification,
      iree_hal_amdgpu_pending_op_wait_callbacks_are_complete, op,
      iree_infinite_timeout());
  iree_status_t error = (iree_status_t)iree_atomic_exchange(
      &op->error_status, 0, iree_memory_order_acquire);
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_pending_op_fail(op, error);
  } else {
    iree_hal_amdgpu_pending_op_issue(op);
  }
}

// Discards a pending operation that failed during capture. Caller MUST hold
// submission_mutex; the op is linked to the pending list by allocate and needs
// the mutex for unlinking. Signal semaphores remain untouched because the
// operation was never successfully captured.
//
// Unlike pending_op_fail (which acquires submission_mutex internally), this
// function is entered and returns with the mutex held. It temporarily releases
// the mutex after unlinking so no final release occurs while submission state
// is locked.
void iree_hal_amdgpu_pending_op_discard_under_lock(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  IREE_ASSERT(op->has_submission_epilogue_token);
  op->has_submission_epilogue_token = false;
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_pending_op_discard_unlocked(op);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  // Capture failed while the public queue call still owns submission
  // serialization. Keep seal joined through unlocked arena cleanup and the
  // relock, then consume the operation's token before returning to the caller.
  iree_hal_amdgpu_host_queue_leave_submission_epilogue(queue);
}

// Timepoint callback fired when a wait semaphore reaches its target value or
// fails. The last resolved wait claims completion, waits for all callbacks to
// finish touching arena-owned entries, and then issues or fails the operation.
static void iree_hal_amdgpu_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;

  iree_hal_amdgpu_pending_op_record_error_status(op, status);

  // Serialize the final count transition, lifecycle claim, completion marker,
  // and notification wake. Cancellation checks the marker under this same
  // mutex and therefore cannot deinitialize the arena in the old window
  // between callback_complete=1 and notification_post().
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  // No queue lock is held at this test-only boundary. The observer must not
  // mutate |op| or reenter the originating queue; it may only gate readiness.
  iree_hal_amdgpu_host_queue_test_notify_phase(
      op->queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_PENDING_CALLBACK_MUTEX,
      (uint64_t)(uintptr_t)op, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_lock(&op->callback_mutex);
  const int32_t previous_count =
      iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel);
  bool owns_completion = false;
  int32_t observed_state =
      iree_atomic_load(&op->lifecycle_state, iree_memory_order_acquire);
  if (previous_count == 1) {
    int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
    owns_completion = iree_atomic_compare_exchange_strong(
        &op->lifecycle_state, &expected_state,
        IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
        iree_memory_order_acq_rel, iree_memory_order_acquire);
    observed_state = owns_completion
                         ? IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING
                         : expected_state;
  }

  // This helper posts and drops callback_mutex as the callback-entry tail's
  // final access. A lifecycle completion owner may then continue using |op|;
  // cancellation cannot take an op in COMPLETING state.
  iree_hal_amdgpu_wait_entry_publish_callback_complete(
      entry, observed_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING);
  if (owns_completion) {
    iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
  }
}

// Registers timepoints for all waits in the operation's wait semaphore list.
// Sets wait_count and registers one timepoint per wait; callbacks may fire
// synchronously during registration. When all waits are satisfied (the last
// callback fires), the operation is issued or failed.
//
// The wait_semaphore_list on the op (cloned into the arena by allocate)
// retains all semaphores for the lifetime of the op. Wait entries do not
// independently retain semaphores.
static iree_status_t iree_hal_amdgpu_pending_op_enqueue_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_semaphore_list_t wait_semaphores = op->wait_semaphore_list;
  if (wait_semaphores.count == 0) {
    iree_atomic_store(&op->lifecycle_state,
                      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
                      iree_memory_order_release);
    iree_hal_amdgpu_pending_op_issue(op);
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, wait_semaphores.count);

  iree_host_size_t wait_entry_bytes = 0;
  iree_status_t status =
      IREE_STRUCT_LAYOUT(0, &wait_entry_bytes,
                         IREE_STRUCT_FIELD(wait_semaphores.count,
                                           iree_hal_amdgpu_wait_entry_t, NULL));
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  status = iree_arena_allocate(&op->arena, wait_entry_bytes,
                               (void**)&op->wait_entries);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  memset(op->wait_entries, 0, wait_entry_bytes);
  // Unregistered entries never receive callbacks, so they start complete.
  // Active registrations flip their entry incomplete until the callback exits.
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    op->wait_entries[i].operation = op;
    iree_atomic_store(&op->wait_entries[i].callback_complete, 1,
                      iree_memory_order_relaxed);
  }

  // Set wait_count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint.
  iree_atomic_store(&op->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    iree_atomic_store(&entry->callback_complete, 0, iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_amdgpu_wait_entry_resolved;
    entry->timepoint.user_data = entry;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously; we cannot
      // destroy the op here. Record the error and subtract the unregistered
      // count so the existing callbacks drain and destroy the op.
      iree_hal_amdgpu_pending_op_record_error_status(op, status);
      const int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      iree_slim_mutex_lock(&op->callback_mutex);
      iree_atomic_store(&entry->callback_complete, 1,
                        iree_memory_order_release);
      const int32_t previous_count = iree_atomic_fetch_sub(
          &op->wait_count, unregistered, iree_memory_order_acq_rel);
      bool owns_completion = false;
      if (previous_count == unregistered) {
        int32_t expected_state =
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_WAITS;
        owns_completion = iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
            iree_memory_order_acq_rel, iree_memory_order_acquire);
        IREE_ASSERT(owns_completion);
      } else {
        int32_t expected_state =
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_WAITS;
        const bool published_pending = iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
            iree_memory_order_release, iree_memory_order_acquire);
        IREE_ASSERT(published_pending);
      }
      iree_slim_mutex_unlock(&op->callback_mutex);
      if (owns_completion) {
        iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
      }
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
    if (i + 1 < wait_semaphores.count) {
      // Exact ARMING_WAITS boundary: this registration has fully returned
      // (including any synchronous callback), at least one later entry is not
      // registered, and cancellation is still ineligible to take the op.
      iree_hal_amdgpu_host_queue_test_notify_phase(
          op->queue,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_ARMING_WAITS_PARTIAL_REGISTERED,
          i + 1, wait_semaphores.count);
    }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  }

  // Publish cancellation eligibility only after every timepoint registration
  // has returned. Synchronous callbacks are serialized by callback_mutex: if
  // they already resolved all waits then this thread claims completion;
  // otherwise the last future callback transitions PENDING to COMPLETING.
  iree_slim_mutex_lock(&op->callback_mutex);
  const bool all_waits_resolved =
      iree_atomic_load(&op->wait_count, iree_memory_order_acquire) == 0;
  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_WAITS;
  const bool transitioned = iree_atomic_compare_exchange_strong(
      &op->lifecycle_state, &expected_state,
      all_waits_resolved ? IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING
                         : IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
      iree_memory_order_release, iree_memory_order_acquire);
  IREE_ASSERT(transitioned);
  iree_slim_mutex_unlock(&op->callback_mutex);
  if (all_waits_resolved) {
    iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_memory_wait_resolved(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  const bool resolved_successfully = iree_status_is_ok(status);

  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_record_error_status(op, status);
    if (resolved_successfully &&
        wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION) {
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
    }
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
        op, /*is_cancelling_loser=*/false);
    return;
  }

  expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    if (resolved_successfully &&
        wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION) {
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
    }
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
        op, /*is_cancelling_loser=*/false);
    if (!resolved_successfully) {
      iree_hal_amdgpu_pending_op_fail(op, status);
    } else {
      iree_hal_amdgpu_pending_op_issue(op);
    }
    return;
  }

  iree_hal_amdgpu_pending_op_record_error_status(op, status);
  iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
      op, expected_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING);
}

static void iree_hal_amdgpu_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_status_is_ok(status)) {
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return;
    }
    if (expected_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING) {
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_hal_amdgpu_pending_op_fail(op, error);
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
    }
    return;
  }

  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return;
  }
  iree_hal_amdgpu_pending_op_record_error_status(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      queue->frontier_tracker, op->alloca_op.transaction.wait_frontier,
      iree_hal_amdgpu_alloca_frontier_wait_resolved, op,
      &wait->frontier.waiter);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_async_notification_wait_operation_t* wait_op =
      &wait->pool_notification.wait_ops[wait->pool_notification.wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_amdgpu_alloca_pool_notification_wait_resolved, op);
  wait_op->notification = wait->pool_notification.notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = wait->pool_notification.wait_token;

  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status =
      iree_async_proactor_submit_one(op->queue->proactor, &wait_op->base);
  iree_hal_amdgpu_alloca_pool_notification_end_observe(wait);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static iree_status_t iree_hal_amdgpu_pending_op_grow_alloca_pool(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_hal_amdgpu_alloca_transaction_t* transaction =
      &op->alloca_op.transaction;
  const iree_async_frontier_t* requester_frontier =
      iree_hal_amdgpu_fixed_frontier_as_frontier(
          &wait->pool_growth.requester_frontier);
  const iree_hal_pool_reserve_flags_t reserve_flags =
      op->alloca_op.reserve_flags & ~IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH;

  transaction->acquire_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  transaction->reservations_held = false;
  transaction->backing_buffers_held = false;
  iree_async_frontier_initialize(transaction->wait_frontier, 0);
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservations(
      op->alloca_op.pool, transaction->request_count, transaction->requests,
      requester_frontier, reserve_flags, transaction->reservations,
      transaction->acquire_infos, &transaction->acquire_result));
  transaction->reservations_held =
      transaction->acquire_result == IREE_HAL_POOL_ACQUIRE_OK ||
      transaction->acquire_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
      transaction->acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;

  switch (transaction->acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      return iree_hal_amdgpu_host_queue_materialize_alloca_transaction(
          op->queue, op->alloca_op.pool, transaction);
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT: {
      bool merged = true;
      for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
        const iree_async_frontier_t* item_frontier =
            transaction->acquire_infos[i].wait_frontier;
        if (item_frontier &&
            !iree_async_frontier_merge(transaction->wait_frontier, UINT8_MAX,
                                       item_frontier)) {
          merged = false;
          break;
        }
      }
      const bool has_wait_frontier =
          transaction->wait_frontier->entry_count != 0;
      if (merged && has_wait_frontier) {
        iree_hal_pool_release_reservations(
            op->alloca_op.pool, transaction->request_count,
            transaction->reservations, transaction->wait_frontier);
      } else {
        for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
          iree_hal_pool_release_reservations(
              op->alloca_op.pool, 1, &transaction->reservations[i],
              transaction->acquire_infos[i].wait_frontier);
        }
      }
      transaction->reservations_held = false;
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      if (IREE_UNLIKELY(merged && !has_wait_frontier)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "waitable pool reservation transaction provided an empty "
            "frontier");
      }
      return iree_ok_status();
    }
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "queue_alloca cold pool growth did not produce a reservation "
          "(result=%u)",
          transaction->acquire_result);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pool acquire result %u",
                              transaction->acquire_result);
  }
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_pool_growth(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_hal_amdgpu_pending_op_grow_alloca_pool(op);
  iree_hal_amdgpu_alloca_memory_wait_resolved(op, status);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
      op, iree_ok_status());
}

void iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_GROWTH:
      iree_hal_amdgpu_pending_op_enqueue_alloca_pool_growth(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      iree_hal_amdgpu_pending_op_fail(
          op, iree_make_status(IREE_STATUS_INTERNAL,
                               "pending alloca has no memory wait to enqueue"));
      break;
  }
}

static void iree_hal_amdgpu_pending_op_enqueue_capacity_retry(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
                    iree_memory_order_release);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      op->queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_START_HANDOFF,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_CAPACITY_COMPLETING_BEFORE_ENQUEUE,
      (uint64_t)(uintptr_t)op, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      op->queue, &op->capacity_retry,
      iree_hal_amdgpu_pending_op_capacity_post_drain, op);
}

static void iree_hal_amdgpu_pending_op_capacity_post_drain(void* user_data) {
  iree_hal_amdgpu_pending_op_issue((iree_hal_amdgpu_pending_op_t*)user_data);
}

iree_status_t iree_hal_amdgpu_pending_op_start(iree_hal_amdgpu_pending_op_t* op,
                                               bool wait_for_capacity) {
  if (wait_for_capacity) {
    iree_hal_amdgpu_pending_op_enqueue_capacity_retry(op);
    return iree_ok_status();
  }
  return iree_hal_amdgpu_pending_op_enqueue_waits(op);
}

// Allocates and initializes a pending operation from a fresh arena.
// Clones the wait semaphore list. Allocates the retained_resources array
// with |max_resource_count| capacity and populates the first entries with
// signal semaphores (retained). The signal_semaphore_list.semaphores pointer
// aliases into retained_resources so that commit_signals and
// semaphore_list_fail can use it directly.
//
// The caller must push operation-specific resources into retained_resources
// (via the returned op) before calling enqueue_waits.
//
// On failure, the arena is cleaned up and *out_op is set to NULL.
iree_status_t iree_hal_amdgpu_pending_op_allocate(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_amdgpu_pending_op_type_t type, uint16_t max_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, type);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_resource_count);

  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_arena_allocate(&arena, sizeof(*op), (void**)&op);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  memset(op, 0, sizeof(*op));
  memcpy(&op->arena, &arena, sizeof(arena));
  op->queue = queue;
  op->type = type;
  iree_atomic_store(&op->wait_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_WAITS,
                    iree_memory_order_relaxed);
  iree_atomic_store(&op->error_status, 0, iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&op->callback_mutex);
  iree_notification_initialize(&op->callback_notification);

  // Install the seal-join token before any fallible capture step can require
  // unlocked cleanup. On success it transfers with the linked op; on failure
  // this function consumes it after arena teardown and relocking.
  iree_hal_amdgpu_host_queue_enter_submission_epilogue(queue);
  op->has_submission_epilogue_token = true;

  iree_allocator_t arena_allocator = iree_arena_allocator(&op->arena);

  // Clone the wait semaphore list (retains each wait semaphore).
  status = iree_hal_semaphore_list_clone(wait_semaphore_list, arena_allocator,
                                         &op->wait_semaphore_list);

  // Allocate the retained resources array and the signal payload values.
  if (iree_status_is_ok(status) && max_resource_count > 0) {
    iree_host_size_t retained_resource_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &retained_resource_size,
        IREE_STRUCT_FIELD(max_resource_count, iree_hal_resource_t*, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, retained_resource_size,
                                   (void**)&op->retained_resources);
    }
  }
  uint64_t* signal_payload_values = NULL;
  if (iree_status_is_ok(status) && signal_semaphore_list->count > 0) {
    iree_host_size_t signal_payload_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &signal_payload_size,
        IREE_STRUCT_FIELD(signal_semaphore_list->count, uint64_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, signal_payload_size,
                                   (void**)&signal_payload_values);
    }
  }

  if (iree_status_is_ok(status)) {
    // Signal semaphores occupy the first entries of retained_resources.
    // The signal_semaphore_list.semaphores pointer aliases this region.
    for (iree_host_size_t i = 0; i < signal_semaphore_list->count; ++i) {
      op->retained_resources[i] =
          (iree_hal_resource_t*)signal_semaphore_list->semaphores[i];
      iree_hal_resource_retain(op->retained_resources[i]);
      signal_payload_values[i] = signal_semaphore_list->payload_values[i];
    }
    op->retained_resource_count = (uint16_t)signal_semaphore_list->count;
    op->signal_semaphore_list.count = signal_semaphore_list->count;
    op->signal_semaphore_list.semaphores =
        (iree_hal_semaphore_t**)op->retained_resources;
    op->signal_semaphore_list.payload_values = signal_payload_values;

    iree_hal_amdgpu_pending_op_link(op);
    *out_op = op;
  } else {
    // Allocation failure can release the last wait-semaphore or arena-block
    // reference; temporarily leave submission serialization before cleanup.
    op->has_submission_epilogue_token = false;
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_slim_mutex_deinitialize(&op->callback_mutex);
    iree_arena_deinitialize(&op->arena);
    iree_slim_mutex_lock(&queue->locks.submission_mutex);
    iree_hal_amdgpu_host_queue_leave_submission_epilogue(queue);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Issues a deferred operation after all waits are satisfied. All waits are
// tier 0 (timeline_value >= waited_value); the GPU work producing those
// values has completed. No barriers are needed.
//
// Called from the last wait_entry callback (any thread). Acquires
// submission_mutex to emit AQL packets and commit signals.
static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;
  (void)iree_hal_amdgpu_host_queue_lifetime_try_acquire(queue, &lifetime_claim);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);

  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_payload_issue_t issue = {
      .ready = true,
      .memory_wait_op = NULL,
  };
  // A recorded failure outranks closed admission. The terminal transition
  // closes admission and then flushes the post-drain retries that resume
  // capacity-parked operations, so consulting the flag first would report the
  // shutdown to an operation resumed there while every operation the same
  // transition cancels reports the fault that caused it.
  status = iree_hal_amdgpu_host_queue_clone_error_status(queue);
  if (iree_status_is_ok(status) && queue->is_shutting_down) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  if (iree_status_is_ok(status)) {
    // All waits are tier 0; emit operation packets with no dependency
    // barriers.
    iree_hal_amdgpu_wait_resolution_t resolution;
    resolution.barrier_count = 0;
    resolution.needs_deferral = false;
    memset(resolution.reserved, 0, sizeof(resolution.reserved));
    resolution.wait_count = op->wait_semaphore_list.count > UINT32_MAX
                                ? UINT32_MAX
                                : (uint32_t)op->wait_semaphore_list.count;
    resolution.profile_event_flags =
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
    resolution.inline_acquire_scope = op->wait_semaphore_list.count > 0
                                          ? IREE_HSA_FENCE_SCOPE_SYSTEM
                                          : IREE_HSA_FENCE_SCOPE_NONE;
    resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
    status = iree_hal_amdgpu_pending_op_issue_payload(op, &resolution, &issue);

    // Payload-specific cleanup can invoke arbitrary resource destructors or
    // pool callbacks. Keep this COMPLETING operation linked while the queue
    // lock is dropped so teardown owns stable queue/op storage and waits for
    // the epilogue instead of freeing either beneath a callback.
    if (issue.cleanup_resource ||
        (op->type == IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA &&
         op->dealloca.transaction.reservations_detached)) {
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      iree_hal_resource_release(issue.cleanup_resource);
      issue.cleanup_resource = NULL;
      if (op->type == IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA) {
        iree_hal_amdgpu_host_queue_release_dealloca_transaction(
            queue, &op->dealloca.transaction);
      }
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
    }

    if (iree_status_is_ok(status) && issue.memory_wait_op) {
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
          issue.memory_wait_op);
      // The rearm may synchronously terminalize the operation. Releasing the
      // claim is therefore the outermost action and may destroy queue/device.
      (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
      return;
    }
  }

  if (iree_status_is_ok(status) && !issue.ready) {
    iree_hal_amdgpu_pending_op_enqueue_capacity_retry(op);
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
    return;
  }

  // Terminalize and unlink while the submission state is serialized, then
  // drop the lock before any callback or last release. In particular,
  // semaphore failure below may synchronously issue another pending operation
  // on this same queue.
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_pending_op_destroy_unlocked(op, status);
  // May synchronously destroy the queue/device. No queue or op access follows.
  (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
}

// Fails a deferred operation. Propagates the error to all signal semaphores
// so downstream waiters receive the failure instead of hanging. Takes
// ownership of |status|.
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;
  (void)iree_hal_amdgpu_host_queue_lifetime_try_acquire(queue, &lifetime_claim);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_pending_op_destroy_unlocked(op, status);
  // May synchronously destroy the queue/device. No queue or op access follows.
  (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
}

void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, const iree_status_t failure_status) {
  for (;;) {
    iree_hal_amdgpu_pending_op_t* op = NULL;
    iree_slim_mutex_lock(&queue->locks.submission_mutex);
    for (iree_hal_amdgpu_pending_op_t* candidate = queue->pending_head;
         candidate != NULL; candidate = candidate->next) {
      int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
      if (iree_atomic_compare_exchange_strong(
              &candidate->lifecycle_state, &expected_state,
              IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        iree_hal_amdgpu_pending_op_unlink(candidate);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
        // submission_mutex remains held. The observer must not reenter the
        // queue or acquire any queue-owned lock.
        iree_hal_amdgpu_host_queue_test_notify_phase(
            queue,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_CANCELLING_LOSER,
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PENDING_CANCELLING_CLAIMED,
            (uint64_t)(uintptr_t)candidate, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
        op = candidate;
        break;
      }
    }
    bool has_pending_ops = queue->pending_head != NULL;
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);

    if (op == NULL) {
      if (!has_pending_ops) break;
      iree_thread_yield();
      continue;
    }

    for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
      iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
      if (iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) continue;
      if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                                &entry->timepoint)) {
        continue;
      }
      iree_notification_await(&op->callback_notification,
                              iree_hal_amdgpu_wait_entry_callback_is_complete,
                              entry, iree_infinite_timeout());
    }
    iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(op);

    iree_status_t op_status = (iree_status_t)iree_atomic_exchange(
        &op->error_status, 0, iree_memory_order_acquire);
    if (iree_status_is_ok(op_status)) {
      op_status = iree_status_clone(failure_status);
    }
    iree_hal_amdgpu_pending_op_destroy_unlocked(op, op_status);
  }
}

//===----------------------------------------------------------------------===//
// Alloca memory-readiness waits
//===----------------------------------------------------------------------===//

// Returns the first queue terminal condition observed while submission_mutex is
// held. Callers use this after every unlocked pool-vtable window before they
// publish transaction ownership or queue work.
static iree_status_t
iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_clone_error_status(queue);
  if (iree_status_is_ok(status) && queue->is_shutting_down) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return status;
}

// Captures an immutable by-value requester frontier while submission_mutex is
// held. Pool callbacks may reenter and advance the queue while it is unlocked;
// the older snapshot remains conservative for death-frontier reuse.
static const iree_async_frontier_t*
iree_hal_amdgpu_host_queue_snapshot_alloca_requester_frontier_locked(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_fixed_frontier_t* storage) {
  const iree_async_frontier_t* requester_frontier =
      iree_hal_amdgpu_host_queue_pool_requester_frontier(queue, resolution,
                                                         storage);
  const iree_async_frontier_t* snapshot =
      iree_hal_amdgpu_fixed_frontier_as_frontier(storage);
  if (requester_frontier != snapshot) {
    memcpy(storage, requester_frontier, sizeof(*storage));
  }
  return snapshot;
}

// Entered and returned with submission_mutex held. Arbitrary pool callbacks and
// final backing-buffer releases run in the unlocked interval.
static void iree_hal_amdgpu_host_queue_release_alloca_transaction_outside_lock(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction) {
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_host_queue_release_alloca_transaction(allocation_pool,
                                                        transaction);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
}

// Publishes an already materialized transaction or rolls it back outside the
// queue lock. Entered and returned with submission_mutex held.
static iree_status_t
iree_hal_amdgpu_host_queue_submit_materialized_alloca_locked(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  iree_status_t status =
      iree_hal_amdgpu_host_queue_submit_alloca_materialization(
          queue, transaction, signal_semaphore_list, allocation_pool,
          submission_flags, out_ready);
  if (iree_status_is_ok(status) && *out_ready) {
    // Pool statistics and profile sinks are public vtables. The transaction is
    // committed and independently retained by the reclaim entry before this
    // unlocked epilogue begins.
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    iree_hal_amdgpu_host_queue_record_committed_alloca(queue, allocation_pool,
                                                       transaction);
    iree_slim_mutex_lock(&queue->locks.submission_mutex);
  } else {
    iree_hal_amdgpu_host_queue_release_alloca_transaction_outside_lock(
        queue, allocation_pool, transaction);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_alloca_memory_wait_t* memory_wait, bool* out_ready) {
  transaction->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
  transaction->wait_resolution = *resolution;
  memory_wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_host_queue_materialize_alloca_transaction(
          queue, allocation_pool, transaction);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(queue);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_release_alloca_transaction_outside_lock(
        queue, allocation_pool, transaction);
    return status;
  }
  return iree_hal_amdgpu_host_queue_submit_materialized_alloca_locked(
      queue, signal_semaphore_list, allocation_pool, transaction,
      submission_flags, out_ready);
}

static iree_status_t
iree_hal_amdgpu_host_queue_submit_alloca_held_growth_materialization(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_alloca_memory_wait_t* memory_wait, bool* out_ready) {
  transaction->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
  transaction->wait_resolution = *resolution;
  memory_wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(queue));
  return iree_hal_amdgpu_host_queue_submit_materialized_alloca_locked(
      queue, signal_semaphore_list, allocation_pool, transaction,
      submission_flags, out_ready);
}

static iree_status_t iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_memory_wait_op = NULL;
  if (pending_op) {
    *out_memory_wait_op = pending_op;
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty_wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_defer_alloca(
      queue, &empty_wait_list, &signal_semaphore_list, allocation_pool,
      transaction->request_count, transaction->requests, transaction->buffers,
      reserve_flags, out_memory_wait_op));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_move_alloca_acquisition(
    iree_hal_amdgpu_alloca_transaction_t* source,
    iree_hal_amdgpu_alloca_transaction_t* target) {
  if (source == target) return iree_ok_status();
  IREE_ASSERT_TRUE(source->request_count == target->request_count);
  IREE_ASSERT_TRUE(source->reservations_held);
  memcpy(target->reservations, source->reservations,
         source->request_count * sizeof(*source->reservations));
  memcpy(target->acquire_infos, source->acquire_infos,
         source->request_count * sizeof(*source->acquire_infos));
  iree_host_size_t frontier_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_frontier_size(
      source->wait_frontier->entry_count, &frontier_size));
  memcpy(target->wait_frontier, source->wait_frontier, frontier_size);
  target->readiness = source->readiness;
  target->acquire_result = source->acquire_result;
  target->wait_resolution = source->wait_resolution;
  target->reservations_held = true;
  source->reservations_held = false;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, transaction, reserve_flags,
      pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_move_alloca_acquisition(
        transaction, &memory_wait_op->alloca_op.transaction);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(memory_wait_op);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else {
    if (!pending_op && memory_wait_op) {
      iree_hal_amdgpu_pending_op_discard_under_lock(memory_wait_op);
    }
  }

  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_pool_growth(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, transaction, reserve_flags,
      pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_pool_growth(
        memory_wait_op, requester_frontier);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else if (!pending_op && memory_wait_op) {
    iree_hal_amdgpu_pending_op_discard_under_lock(memory_wait_op);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_async_notification_t* notification, uint32_t wait_token,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op,
    bool* out_observation_transferred) {
  *out_observation_transferred = false;
  bool observation_transferred = false;
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, transaction, reserve_flags,
      pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
        memory_wait_op, notification, wait_token);
    observation_transferred = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else if (!pending_op && memory_wait_op) {
    iree_hal_amdgpu_pending_op_discard_under_lock(memory_wait_op);
  }
  *out_observation_transferred = observation_transferred;
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op, bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  *out_memory_wait_op = NULL;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_hal_amdgpu_host_queue_clone_shutdown_status(queue);
  }

  iree_hal_amdgpu_alloca_memory_wait_t* memory_wait =
      pending_op ? pending_op->alloca_op.memory_wait : NULL;
  if (memory_wait &&
      memory_wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER) {
    return iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
        queue, resolution, signal_semaphore_list, allocation_pool, transaction,
        submission_flags, memory_wait, out_ready);
  }
  if (memory_wait &&
      memory_wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_GROWTH &&
      transaction->backing_buffers_held) {
    return iree_hal_amdgpu_host_queue_submit_alloca_held_growth_materialization(
        queue, resolution, signal_semaphore_list, allocation_pool, transaction,
        submission_flags, memory_wait, out_ready);
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(queue));

  iree_hal_amdgpu_fixed_frontier_t requester_frontier_storage;
  const iree_async_frontier_t* requester_frontier =
      iree_hal_amdgpu_host_queue_snapshot_alloca_requester_frontier_locked(
          queue, resolution, &requester_frontier_storage);

  // Observe before checking pool state so a release concurrent with acquire
  // cannot be lost. Pools that never expose a notification may still satisfy
  // the request; NULL is an error only when the returned state needs a wake.
  iree_async_notification_t* notification = NULL;
  uint32_t wait_token = 0;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  notification = iree_hal_pool_notification(allocation_pool);
  if (notification) {
    wait_token = iree_async_notification_begin_observe(notification);
  }
  iree_status_t status = iree_hal_amdgpu_host_queue_acquire_alloca_transaction(
      queue, resolution, requester_frontier, allocation_pool, reserve_flags,
      transaction);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(queue);
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    if (notification) iree_async_notification_end_observe(notification);
    iree_hal_amdgpu_host_queue_release_alloca_transaction(allocation_pool,
                                                          transaction);
    iree_slim_mutex_lock(&queue->locks.submission_mutex);
    return status;
  }

  // Returned death frontiers are integrated only after the pool callback has
  // returned and submission serialization is reacquired. The pre-callback
  // requester snapshot is intentionally conservative across reentrant queue
  // publications.
  if (transaction->readiness ==
      IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT) {
    iree_hal_amdgpu_wait_resolution_t candidate_resolution = *resolution;
    if (iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
            queue, requester_frontier, transaction->wait_frontier,
            &candidate_resolution)) {
      transaction->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
      transaction->wait_resolution = candidate_resolution;
    }
  }

  switch (transaction->readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY: {
      // Materialization and observation release may invoke arbitrary pool or
      // resource callbacks. Newly created backing remains transaction-private
      // until the relocked submission commit attaches it to transient buffers.
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      if (notification) iree_async_notification_end_observe(notification);
      status = iree_hal_amdgpu_host_queue_materialize_alloca_transaction(
          queue, allocation_pool, transaction);
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(
            queue);
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amdgpu_host_queue_release_alloca_transaction_outside_lock(
            queue, allocation_pool, transaction);
        return status;
      }
      return iree_hal_amdgpu_host_queue_submit_materialized_alloca_locked(
          queue, signal_semaphore_list, allocation_pool, transaction,
          submission_flags, out_ready);
    }
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      if (notification) iree_async_notification_end_observe(notification);
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
      status =
          iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(queue);
      if (!iree_status_is_ok(status)) {
        iree_hal_amdgpu_host_queue_release_alloca_transaction_outside_lock(
            queue, allocation_pool, transaction);
        return status;
      }
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, transaction,
          reserve_flags, pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_GROWTH:
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      if (notification) iree_async_notification_end_observe(notification);
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
      status =
          iree_hal_amdgpu_host_queue_alloca_revalidate_admission_locked(queue);
      if (!iree_status_is_ok(status)) return status;
      return iree_hal_amdgpu_host_queue_defer_alloca_pool_growth(
          queue, signal_semaphore_list, requester_frontier, allocation_pool,
          transaction, reserve_flags, pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION: {
      if (IREE_UNLIKELY(!notification)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "queue_alloca exhausted pool did not provide a notification");
      }
      bool observation_transferred = false;
      status = iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
          queue, signal_semaphore_list, allocation_pool, transaction,
          reserve_flags, pending_op, notification, wait_token,
          out_memory_wait_op, &observation_transferred);
      if (!observation_transferred) {
        iree_slim_mutex_unlock(&queue->locks.submission_mutex);
        iree_async_notification_end_observe(notification);
        iree_slim_mutex_lock(&queue->locks.submission_mutex);
      }
      return status;
    }
    default:
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      if (notification) iree_async_notification_end_observe(notification);
      iree_hal_amdgpu_host_queue_release_alloca_transaction(allocation_pool,
                                                            transaction);
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              transaction->readiness);
  }
}
