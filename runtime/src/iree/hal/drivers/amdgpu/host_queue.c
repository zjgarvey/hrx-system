// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/device/tsan.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/feedback_state.h"
#include "iree/hal/drivers/amdgpu/host_queue_atomic.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_scratch.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/host_queue_file.h"
#include "iree/hal/drivers/amdgpu/host_queue_host_call.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/host_queue_staging.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"
#include "iree/hal/drivers/amdgpu/host_queue_transfer.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/hsa_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system_event.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/tsan_state.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/hal/utils/resource_set.h"

static void iree_hal_amdgpu_host_queue_destroy(iree_hal_queue_t* base_queue);
static const iree_hal_queue_vtable_t iree_hal_amdgpu_host_queue_vtable;

// Stack-owned identity of a queue completion service. The thread trampoline
// copies its entry argument before calling us, so a self-finalizing queue may
// detach/free its iree_thread_t and queue/device storage as long as this record
// is the only state inspected before the entry point returns.
typedef struct iree_hal_amdgpu_host_queue_completion_service_t {
  iree_hal_amdgpu_host_queue_t* queue;
  struct iree_hal_amdgpu_host_queue_completion_service_t* previous;
  bool exit_requested;
} iree_hal_amdgpu_host_queue_completion_service_t;

static IREE_THREAD_LOCAL iree_hal_amdgpu_host_queue_completion_service_t*
    iree_hal_amdgpu_host_queue_current_completion_service = NULL;

bool iree_hal_amdgpu_host_queue_lifetime_try_acquire(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_lifetime_claim_t* out_claim) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_claim);
  memset(out_claim, 0, sizeof(*out_claim));

  // The device is acquired first so every queue retain continues to satisfy
  // the HAL queue contract requiring its parent to remain live. Both retains
  // are non-resurrecting: failure means a destructor already owns lifetime and
  // will join the independently stable callback/runner before freeing storage.
  iree_hal_device_t* device = queue->logical_device;
  if (!iree_hal_resource_try_retain(device)) return false;
  out_claim->device = device;
  out_claim->device_retained = true;
  if (!iree_hal_resource_try_retain(&queue->base)) {
    out_claim->device = NULL;
    out_claim->device_retained = false;
    iree_hal_device_release(device);
    return false;
  }
  out_claim->queue = queue;
  out_claim->queue_retained = true;
  return true;
}

bool iree_hal_amdgpu_host_queue_lifetime_release(
    iree_hal_amdgpu_host_queue_lifetime_claim_t* claim) {
  IREE_ASSERT_ARGUMENT(claim);
  iree_hal_amdgpu_host_queue_t* queue = claim->queue;
  iree_hal_device_t* device = claim->device;
  const bool queue_retained = claim->queue_retained;
  const bool device_retained = claim->device_retained;
  const bool is_dedicated = queue && queue->storage.parent_device != NULL;
  memset(claim, 0, sizeof(*claim));

  // A dedicated queue has a permanent queue->device edge. Drop the temporary
  // device reference first and the queue reference last; its destructor can
  // then consume the permanent edge after freeing queue storage. Provisioned
  // and cached-cooperative queues are owned by the device, so return the queue
  // to its parent-only reference before the final device release.
  if (is_dedicated) {
    if (device_retained) iree_hal_device_release(device);
    if (queue_retained) iree_hal_queue_release(&queue->base);
  } else {
    if (queue_retained) iree_hal_queue_release(&queue->base);
    if (device_retained) iree_hal_device_release(device);
  }

  iree_hal_amdgpu_host_queue_completion_service_t* current_service =
      iree_hal_amdgpu_host_queue_current_completion_service;
  return current_service && current_service->exit_requested;
}

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
static iree_hal_amdgpu_host_queue_test_phase_observer_t
    iree_hal_amdgpu_host_queue_test_phase_observer = NULL;
static void* iree_hal_amdgpu_host_queue_test_phase_observer_user_data = NULL;

void iree_hal_amdgpu_host_queue_test_set_phase_observer(
    iree_hal_amdgpu_host_queue_test_phase_observer_t observer,
    void* user_data) {
  iree_hal_amdgpu_host_queue_test_phase_observer = observer;
  iree_hal_amdgpu_host_queue_test_phase_observer_user_data = user_data;
}

void iree_hal_amdgpu_host_queue_test_notify_phase(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_test_subject_t subject,
    iree_hal_amdgpu_host_queue_test_phase_t phase, uint64_t value0,
    uint64_t value1) {
  if (iree_hal_amdgpu_host_queue_test_phase_observer) {
    iree_hal_amdgpu_host_queue_test_phase_observer(
        queue, subject, phase, value0, value1,
        iree_hal_amdgpu_host_queue_test_phase_observer_user_data);
  }
}
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

static iree_status_t iree_hal_amdgpu_host_queue_submit_tsan_state_initialize(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_tsan_queue_initialize_args_t* initialize_args) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  iree_hal_amdgpu_host_queue_kernel_submission_t submission;
  bool ready = false;
  uint64_t submission_epoch = 0;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
      queue, &resolution, iree_hal_semaphore_list_empty(),
      /*operation_resource_count=*/0, /*payload_packet_count=*/1,
      (uint32_t)iree_host_size_ceil_div(
          sizeof(iree_hal_amdgpu_tsan_queue_initialize_args_t),
          sizeof(iree_hal_amdgpu_kernarg_block_t)),
      &ready, &submission);
  if (iree_status_is_ok(status) && !ready) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue had insufficient capacity for TSAN state initialization");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
        &queue->aql_ring, submission.first_packet_id);
    iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
        &queue->transfer_context->kernels
             ->iree_hal_amdgpu_device_tsan_initialize_queue_state,
        initialize_args, queue->transfer_context->max_workgroup_count,
        &packet->dispatch, submission.kernargs.blocks->data);
    packet->dispatch.completion_signal =
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring);
    const uint16_t setup = packet->dispatch.setup;
    const iree_hsa_fence_scope_t acquire_scope =
        iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
            IREE_HSA_FENCE_SCOPE_AGENT);
    const uint16_t header = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_packet_control_barrier(acquire_scope,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));

    iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, &resolution,
                                                             &submission);
    submission_epoch = iree_hal_amdgpu_host_queue_finish_kernel_submission(
        queue, &resolution, iree_hal_semaphore_list_empty(),
        /*operation_resources=*/NULL, /*operation_resource_count=*/0,
        /*inout_resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &submission);
    iree_hal_amdgpu_host_queue_publish_submission_kernargs(queue, &submission);
    iree_hal_amdgpu_host_queue_publish_submission_epoch(queue,
                                                        submission_epoch);
    iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
    iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                      submission.first_packet_id);
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_wait_for_setup_epoch(queue,
                                                             submission_epoch);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent,
    hsa_amd_memory_pool_t pm4_ib_pool, uint32_t aql_queue_capacity,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, aql_queue_capacity);
  iree_host_size_t pm4_ib_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &pm4_ib_size,
              IREE_STRUCT_FIELD(aql_queue_capacity,
                                iree_hal_amdgpu_pm4_ib_slot_t, NULL)));
  if (IREE_UNLIKELY(!pm4_ib_pool.handle)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "PM4 IB memory pool is required"));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, pm4_ib_size);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slots = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(libhsa), pm4_ib_pool, pm4_ib_size,
              HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, (void**)&pm4_ib_slots));
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(libhsa), /*num_agents=*/1, &gpu_agent, /*flags=*/NULL,
      pm4_ib_slots);
  if (iree_status_is_ok(status)) {
    memset(pm4_ib_slots, 0, pm4_ib_size);
    out_queue->pm4_ib_slots = pm4_ib_slots;
  } else {
    status = iree_status_join(status, iree_hsa_amd_memory_pool_free(
                                          IREE_LIBHSA(libhsa), pm4_ib_slots));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_initialize_tsan_state(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_tsan_memory_policy_t* memory_policy,
    iree_device_size_t workgroup_shadow_stride,
    iree_device_size_t dispatch_shadow_stride, uint32_t workgroup_capacity,
    uint32_t shadow_entry_size, uint32_t memory_granule_shift,
    uint32_t shadow_slot_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, shadow_slot_count);

  if (IREE_UNLIKELY(queue->tsan.allocation_base)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "AMDGPU host queue TSAN state is already "
                             "initialized"));
  }
  if (IREE_UNLIKELY(queue->device_ordinal > UINT32_MAX)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU TSAN queue ordinals exceed device ABI "
                             "limits"));
  }
  if (IREE_UNLIKELY(queue->aql_ring.mask >= UINT32_MAX)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU TSAN AQL ring mask %u"
                             " exceeds payload slot derivation ABI limits",
                             queue->aql_ring.mask));
  }

  iree_device_size_t shadow_size = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul(
          dispatch_shadow_stride, shadow_slot_count, &shadow_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "AMDGPU TSAN queue shadow size overflow: "
                "dispatch_shadow_stride=%" PRIu64 ", shadow_slot_count=%u",
                (uint64_t)dispatch_shadow_stride, shadow_slot_count));
  }
  if (IREE_UNLIKELY(shadow_size == 0 ||
                    (iree_device_size_t)(iree_host_size_t)shadow_size !=
                        shadow_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU TSAN queue shadow allocation size is "
                             "invalid: %" PRIu64,
                             (uint64_t)shadow_size));
  }

  iree_host_size_t queue_state_offset = 0;
  iree_host_size_t shadow_offset = 0;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &allocation_size,
              IREE_STRUCT_FIELD(1, iree_hal_amdgpu_tsan_queue_state_t,
                                &queue_state_offset),
              IREE_STRUCT_FIELD_ALIGNED((iree_host_size_t)shadow_size, uint8_t,
                                        64, &shadow_offset)));

  void* allocation_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(queue->libhsa), memory_policy->memory_pool,
              allocation_size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
              &allocation_base));
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(queue->libhsa), memory_policy->access_agent_count,
      memory_policy->access_agents, /*flags=*/NULL, allocation_base);
  if (iree_status_is_ok(status)) {
    uint8_t* storage = (uint8_t*)allocation_base;
    iree_hal_amdgpu_tsan_queue_state_t* queue_state =
        (iree_hal_amdgpu_tsan_queue_state_t*)(storage + queue_state_offset);
    void* shadow_base = storage + shadow_offset;
    const iree_hal_amdgpu_tsan_queue_state_t host_state = {
        .record_length = sizeof(iree_hal_amdgpu_tsan_queue_state_t),
        .abi_version = IREE_HAL_AMDGPU_TSAN_QUEUE_STATE_ABI_VERSION_0,
        .flags = IREE_HAL_AMDGPU_TSAN_QUEUE_STATE_FLAG_NONE,
        .queue_ordinal = iree_async_axis_queue_index(queue->axis),
        .physical_device_ordinal = (uint32_t)queue->device_ordinal,
        .physical_queue_ordinal = queue->physical_queue_ordinal,
        .reserved0 = 0,
        .shadow_slot_count = shadow_slot_count,
        .aql_ring_base = (uint64_t)(uintptr_t)queue->aql_ring.base,
        .aql_ring_mask = queue->aql_ring.mask,
        .reserved1 = 0,
        .shadow_base = (uint64_t)(uintptr_t)shadow_base,
        .shadow_size = shadow_size,
        .dispatch_shadow_stride = dispatch_shadow_stride,
        .workgroup_shadow_stride = workgroup_shadow_stride,
        .workgroup_capacity = workgroup_capacity,
        .shadow_entry_size = shadow_entry_size,
        .memory_granule_shift = memory_granule_shift,
        .reserved2 = 0,
        .reserved3 = 0,
    };
    const iree_hal_amdgpu_tsan_queue_initialize_args_t initialize_args = {
        .queue_state = queue_state,
        .shadow_base = shadow_base,
        .shadow_size = shadow_size,
        .queue_state_template_value = host_state,
    };
    status = iree_hal_amdgpu_host_queue_submit_tsan_state_initialize(
        queue, &initialize_args);
    if (iree_status_is_ok(status)) {
      queue->tsan.allocation_base = allocation_base;
      queue->tsan.allocation_size = allocation_size;
      queue->tsan.host_state = host_state;
      queue->tsan.queue_state = queue_state;
      queue->tsan.shadow_base = shadow_base;
      queue->tsan.shadow_size = shadow_size;
    } else {
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(queue->libhsa),
                                                allocation_base));
    }
  } else {
    status = iree_status_join(
        status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(queue->libhsa),
                                              allocation_base));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize_tsan_state(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->tsan.allocation_base) return;
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_memory_pool_free_raw(
      queue->libhsa, queue->tsan.allocation_base));
  memset(&queue->tsan, 0, sizeof(queue->tsan));
}

static void iree_hal_amdgpu_host_queue_retire_reclaim_entry(
    iree_hal_amdgpu_reclaim_entry_t* entry,
    iree_hal_amdgpu_reclaim_retire_flags_t flags, void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  if (iree_any_bit_set(flags, IREE_HAL_AMDGPU_RECLAIM_RETIRE_FLAG_FAILED)) {
    return;
  }
  iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation = {
      .first_event_position = entry->profile_event_first_position,
      .event_count = entry->profile_event_count,
  };
  iree_hal_amdgpu_host_queue_retire_profile_dispatch_events(queue, reservation);
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      queue_device_reservation = {
          .first_event_position = entry->queue_device_event_first_position,
          .event_count = entry->queue_device_event_count,
      };
  iree_hal_amdgpu_host_queue_retire_profile_queue_device_events(
      queue, queue_device_reservation);
}

static void iree_hal_amdgpu_host_queue_reclaim_retired(
    iree_hal_amdgpu_reclaim_entry_t* entry, uint64_t epoch,
    iree_hal_amdgpu_reclaim_retire_flags_t flags, void* user_data) {
  (void)epoch;
  iree_hal_amdgpu_host_queue_retire_reclaim_entry(entry, flags, user_data);
}

static void iree_hal_amdgpu_host_queue_reclaim_retired_with_feedback(
    iree_hal_amdgpu_reclaim_entry_t* entry, uint64_t epoch,
    iree_hal_amdgpu_reclaim_retire_flags_t flags, void* user_data) {
  (void)epoch;
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  iree_hal_amdgpu_feedback_source_batch_t* source_batch =
      entry->feedback_source_batch;
  entry->feedback_source_batch = NULL;
  if (source_batch &&
      !iree_any_bit_set(flags, IREE_HAL_AMDGPU_RECLAIM_RETIRE_FLAG_FAILED)) {
    // Lane A acquired this completed epoch before Lane C reached this hook.
    // The acquired reservation target closes the exact source interval; the
    // device feedback service owns eventual read-tail retirement independently
    // of this queue and may outlive the reclaim entry.
    iree_hal_amdgpu_feedback_source_batch_close(source_batch);
  }
  // A force-failed epoch has no matching HSA completion acquire. Its installed
  // batch deliberately remains OPEN in the device ledger until feedback/device
  // teardown, preventing a late packet from dereferencing a released source.
  iree_hal_amdgpu_host_queue_retire_reclaim_entry(entry, flags, queue);
}

static iree_hal_amdgpu_reclaim_retire_fn_t
iree_hal_amdgpu_host_queue_reclaim_retire_fn(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->feedback_state
             ? iree_hal_amdgpu_host_queue_reclaim_retired_with_feedback
             : iree_hal_amdgpu_host_queue_reclaim_retired;
}

static void iree_hal_amdgpu_host_queue_reclaim_queue_owned_positions(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_reclaim_positions_t reclaim_positions) {
  if (reclaim_positions.kernarg_write_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(
        &queue->kernarg_ring, reclaim_positions.kernarg_write_position);
  }
  if (reclaim_positions.queue_upload_write_position > 0) {
    IREE_ASSERT(queue->queue_upload_ring.base,
                "queue upload bytes retired without an initialized upload "
                "ring");
    iree_hal_amdgpu_queue_upload_ring_reclaim(
        &queue->queue_upload_ring,
        reclaim_positions.queue_upload_write_position);
  }
}

//===----------------------------------------------------------------------===//
// Initialization / deinitialization
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_host_queue_enqueue_post_drain_action_and_notify(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_post_drain_action_t* action,
    iree_hal_amdgpu_host_queue_post_drain_fn_t fn, void* user_data,
    iree_notification_t* enqueued_notification) {
  action->next = NULL;
  action->fn = fn;
  action->user_data = user_data;

  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  if (queue->post_drain.tail) {
    queue->post_drain.tail->next = action;
  } else {
    queue->post_drain.head = action;
  }
  queue->post_drain.tail = action;
  if (enqueued_notification) {
    iree_notification_post(enqueued_notification, IREE_ALL_WAITERS);
  }
  if (queue->completion.work_signal.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(queue->libhsa),
                                    queue->completion.work_signal, 1);
  }
  // Posting both wakes while holding post_drain_mutex makes the unlock the
  // producer's final queue/action access. A service or sealer cannot dequeue
  // and free embedded action storage until after this function stops touching
  // it, and a stopped service can observe |enqueued_notification|.
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
}

void iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_post_drain_action_t* action,
    iree_hal_amdgpu_host_queue_post_drain_fn_t fn, void* user_data) {
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action_and_notify(
      queue, action, fn, user_data, /*enqueued_notification=*/NULL);
}

static void iree_hal_amdgpu_host_queue_run_post_drain_actions(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  if (queue->post_drain.runner_active) {
    // The active runner owns exactly one detached batch. Ordinary drainers
    // must not wait here: a callback may reenter a waiter drain on the same
    // thread. Work appended behind the detached batch keeps the completion
    // work signal raised for a later service pass, while terminal sealers use
    // await_post_drain_idle to drive all batches to a fixed point.
    iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
    return;
  }
  queue->post_drain.runner_active = true;

  // A normal completion epilogue owns one snapshot, not a drain-to-fixed-point
  // loop. A callback that consumes newly reclaimed notification capacity may
  // fill the ring again before a later callback retries. Yielding after this
  // snapshot lets notification drain run between those attempts.
  iree_hal_amdgpu_host_queue_post_drain_action_t* action =
      queue->post_drain.head;
  queue->post_drain.head = NULL;
  queue->post_drain.tail = NULL;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (action) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_BATCH_DETACHED_BEFORE_CALLBACK,
        /*value0=*/0, /*value1=*/0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  while (action) {
    iree_hal_amdgpu_host_queue_post_drain_action_t* next_action = action->next;
    action->next = NULL;
    action->fn(action->user_data);
    action = next_action;
  }

  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  IREE_ASSERT((queue->post_drain.head == NULL) ==
              (queue->post_drain.tail == NULL));
  queue->post_drain.runner_active = false;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  const bool has_queued_batch = queue->post_drain.head != NULL;
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_BATCH_YIELDED_BEFORE_WAKE,
      has_queued_batch ? 1 : 0, /*value1=*/0);
  if (!has_queued_batch) {
    iree_hal_amdgpu_host_queue_test_notify_phase(
        queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_IDLE_BEFORE_WAKE, 0,
        0);
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_notification_post(&queue->post_drain_notification, IREE_ALL_WAITERS);
  // This unlock is the runner's final queue access. Posting while the mutex is
  // still held prevents a woken sealer from observing the yielded state and
  // freeing the notification storage before the post completes.
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
}

static bool iree_hal_amdgpu_host_queue_post_drain_runner_is_inactive(
    void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  const bool is_inactive = !queue->post_drain.runner_active;
  IREE_ASSERT((queue->post_drain.head == NULL) ==
              (queue->post_drain.tail == NULL));
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
  return is_inactive;
}

static void iree_hal_amdgpu_host_queue_await_post_drain_idle(
    iree_hal_amdgpu_host_queue_t* queue) {
  // The sealer owns liveness after completion service shutdown. Repeatedly
  // claim one batch or join its current owner, then recheck under the queue
  // mutex. This is deliberately the fixed-point counterpart to the
  // fairness-limited normal service pass above. Callers reach this only from
  // terminal ownership outside a post-drain callback: completion/waiter
  // lifetime claims defer self-seal until their epilogue (and its batch) has
  // returned, so joining an active runner cannot wait on the calling thread.
  for (;;) {
    iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
    iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
    const bool is_idle =
        !queue->post_drain.runner_active && queue->post_drain.head == NULL;
    IREE_ASSERT((queue->post_drain.head == NULL) ==
                (queue->post_drain.tail == NULL));
    iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
    if (is_idle) return;
    iree_notification_await(
        &queue->post_drain_notification,
        iree_hal_amdgpu_host_queue_post_drain_runner_is_inactive, queue,
        iree_infinite_timeout());
  }
}

// Drains every owner-sensitive deferred path while the queue's public failure
// and resource ledgers are still published. Admission must already be closed.
// The first pass resumes capacity-parked operations so they observe shutdown;
// cancellation then waits out semaphore callbacks and releases their retained
// resources; the final fixed-point drain consumes cleanup actions produced by
// either path, including actions requeued by earlier detached snapshots.
static void iree_hal_amdgpu_host_queue_drain_shutdown_deferred(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
  iree_status_t shutdown_status =
      iree_hal_amdgpu_host_queue_clone_shutdown_status(queue);
  iree_hal_amdgpu_host_queue_cancel_pending(queue, shutdown_status);
  iree_status_free(shutdown_status);
  iree_hal_amdgpu_host_queue_await_post_drain_idle(queue);
}

static bool iree_hal_amdgpu_host_queue_post_drain_is_empty(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  const bool is_empty =
      queue->post_drain.head == NULL && !queue->post_drain.runner_active;
  IREE_ASSERT(is_empty == (queue->post_drain.tail == NULL));
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
  return is_empty;
}

void iree_hal_amdgpu_host_queue_query_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_scope_t* out_scope) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_scope);

  *out_scope = (iree_hal_amdgpu_queue_scope_t){
      .physical_device_ordinal = queue->device_ordinal,
      .physical_queue_ordinal = queue->physical_queue_ordinal,
      .aql_ring_base = (uint64_t)(uintptr_t)queue->aql_ring.base,
      .aql_ring_mask = queue->aql_ring.mask,
      .tsan =
          {
              .queue_state_base = (uint64_t)(uintptr_t)queue->tsan.queue_state,
              .shadow_base = queue->tsan.host_state.shadow_base,
              .shadow_size = queue->tsan.host_state.shadow_size,
              .dispatch_shadow_stride =
                  queue->tsan.host_state.dispatch_shadow_stride,
              .workgroup_shadow_stride =
                  queue->tsan.host_state.workgroup_shadow_stride,
              .workgroup_capacity = queue->tsan.host_state.workgroup_capacity,
              .shadow_entry_size = queue->tsan.host_state.shadow_entry_size,
              .memory_granule_shift =
                  queue->tsan.host_state.memory_granule_shift,
              .shadow_slot_count = queue->tsan.host_state.shadow_slot_count,
          },
  };
}

typedef struct iree_hal_amdgpu_host_queue_completion_runner_t {
  iree_hal_amdgpu_host_queue_t* queue;
  iree_hal_amdgpu_notification_ring_claim_t claim;
  struct iree_hal_amdgpu_host_queue_completion_runner_t* previous;
  uint64_t frontier_dispatched_epoch;
  uint32_t dispatch_depth;
  uint32_t retire_depth;
  bool frontier_failed;
  bool profile_events_cleared;
  bool release_in_progress;
} iree_hal_amdgpu_host_queue_completion_runner_t;

// Direct waiters may assist from unmanaged threads, so runner identity is a
// TLS stack instead of an iree_thread_t. The stack also supports a callback on
// one queue draining another queue and then reentering its predecessor.
static IREE_THREAD_LOCAL iree_hal_amdgpu_host_queue_completion_runner_t*
    iree_hal_amdgpu_host_queue_current_completion_runner = NULL;

static iree_hal_amdgpu_host_queue_completion_runner_t*
iree_hal_amdgpu_host_queue_find_current_completion_runner(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_host_queue_completion_runner_t* runner =
      iree_hal_amdgpu_host_queue_current_completion_runner;
  while (runner && runner->queue != queue) runner = runner->previous;
  return runner;
}

// Runs A-E against the runner's private cursors. Reentrant calls share the
// same claim. A nested call may extend and dispatch later completed values but
// never publishes producer-visible reuse cursors; only the outer owner commits.
static iree_host_size_t iree_hal_amdgpu_host_queue_completion_runner_pump(
    iree_hal_amdgpu_host_queue_completion_runner_t* runner) {
  iree_hal_amdgpu_host_queue_t* queue = runner->queue;
  iree_hal_amdgpu_notification_ring_claim_t* claim = &runner->claim;
  const uint64_t initial_read = claim->dispatched_read;

  // Feedback sinks are forbidden from reentering their originating device.
  // Refuse to mutate the shared private claim if an internal diagnostic or
  // malformed sink nevertheless reenters while lane C owns an entry.
  if (runner->retire_depth != 0) return 0;

  for (;;) {
    const intptr_t error_status =
        iree_hal_amdgpu_host_queue_load_error_status_raw(queue);
    const bool force_failure = error_status != 0;
    const iree_status_code_t failure_code =
        force_failure ? iree_status_code((iree_status_t)error_status)
                      : IREE_STATUS_OK;
    const uint64_t target_epoch =
        iree_hal_amdgpu_notification_ring_claim_query_target(
            &queue->notification_ring, force_failure);

    if (target_epoch > claim->transitioned_epoch) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      iree_hal_amdgpu_host_queue_test_notify_phase(
          queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED,
          target_epoch, force_failure ? 1 : 0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

      const iree_status_t transition_status =
          force_failure ? iree_status_from_code(failure_code)
                        : iree_ok_status();
      iree_hal_amdgpu_notification_ring_claim_transition(
          &queue->notification_ring, claim, target_epoch, transition_status);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      iree_hal_amdgpu_host_queue_test_notify_phase(
          queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TRANSITIONS_DONE,
          claim->transitioned_epoch, 0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

      iree_hal_amdgpu_notification_ring_claim_prepare(
          &queue->notification_ring, claim, target_epoch, force_failure,
          failure_code, /*fallback_frontier=*/NULL);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      iree_hal_amdgpu_host_queue_test_notify_phase(
          queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_VALUES_PREPARED,
          claim->prepared_read, target_epoch);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

      ++runner->retire_depth;
      iree_hal_amdgpu_notification_ring_claim_retire(
          &queue->notification_ring, claim, target_epoch, force_failure,
          iree_hal_amdgpu_host_queue_reclaim_retire_fn(queue), queue);
      --runner->retire_depth;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      iree_hal_amdgpu_host_queue_test_notify_phase(
          queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_FEEDBACK_TARGET_CLOSED,
          claim->retire_completed_epoch, 0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
    }

    // Dispatch may synchronously invoke timepoint callbacks. Nested drains
    // share this claim and may extend A-D, but E remains blocked until the
    // complete callback stack has returned.
    ++runner->dispatch_depth;
    (void)iree_hal_amdgpu_notification_ring_claim_dispatch(
        &queue->notification_ring, claim);
    --runner->dispatch_depth;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
    iree_hal_amdgpu_host_queue_test_notify_phase(
        queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED,
        claim->dispatched_read, 0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

    // Frontier failure/advance also synchronously dispatches callbacks and is
    // therefore lane D work, not scalar bookkeeping after cursor publication.
    // Mark the private cursor first so a nested same-thread drain extends the
    // claim without recursively dispatching the same frontier transition.
    if (runner->dispatch_depth == 0) {
      if (force_failure && !runner->profile_events_cleared) {
        runner->profile_events_cleared = true;
        ++runner->dispatch_depth;
        iree_hal_amdgpu_host_queue_clear_profile_events(queue);
        --runner->dispatch_depth;
        continue;
      }
      if (force_failure && !runner->frontier_failed) {
        runner->frontier_failed = true;
        ++runner->dispatch_depth;
        iree_async_frontier_tracker_fail_axis(
            queue->frontier_tracker, queue->axis,
            iree_status_from_code(failure_code));
        --runner->dispatch_depth;
        continue;
      }
      if (!runner->frontier_failed &&
          claim->retire_completed_epoch > runner->frontier_dispatched_epoch) {
        const uint64_t frontier_epoch = claim->retire_completed_epoch;
        runner->frontier_dispatched_epoch = frontier_epoch;
        ++runner->dispatch_depth;
        iree_async_frontier_tracker_advance(queue->frontier_tracker,
                                            queue->axis, frontier_epoch);
        --runner->dispatch_depth;
        continue;
      }
    }

    // A generic release may synchronously reenter this queue. Advance the
    // private epoch first, but prevent the nested pump from starting a second
    // E action until this release has returned. It may still run A-D so waits
    // on later completed values make progress.
    if (runner->dispatch_depth == 0 && !runner->release_in_progress &&
        claim->released_epoch < claim->retire_completed_epoch) {
      runner->release_in_progress = true;
      const bool did_release =
          iree_hal_amdgpu_notification_ring_claim_release_one(
              &queue->notification_ring, claim);
      runner->release_in_progress = false;
      IREE_ASSERT(did_release);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
      iree_hal_amdgpu_host_queue_test_notify_phase(
          queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_RELEASES_DONE,
          claim->released_epoch, 0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
      continue;
    }

    // Reclaim queue-owned ring positions as part of lane E. Detach the
    // positions first and block nested E while the reclaim implementation may
    // synchronously release resources or reenter the queue.
    if (runner->dispatch_depth == 0 && !runner->release_in_progress &&
        (claim->reclaim_positions.kernarg_write_position != 0 ||
         claim->reclaim_positions.queue_upload_write_position != 0)) {
      const iree_hal_amdgpu_reclaim_positions_t reclaim_positions =
          claim->reclaim_positions;
      claim->reclaim_positions = (iree_hal_amdgpu_reclaim_positions_t){0};
      runner->release_in_progress = true;
      iree_hal_amdgpu_host_queue_reclaim_queue_owned_positions(
          queue, reclaim_positions);
      runner->release_in_progress = false;
      continue;
    }

    // Recheck after callback-capable D/E work. GPU completion or a reentrant
    // submission may have extended the ready prefix while public reuse cursors
    // intentionally remained frozen.
    const intptr_t final_error_status =
        iree_hal_amdgpu_host_queue_load_error_status_raw(queue);
    const uint64_t final_target =
        iree_hal_amdgpu_notification_ring_claim_query_target(
            &queue->notification_ring, final_error_status != 0);
    if (final_target > claim->claimed_epoch) continue;
    if (claim->dispatched_read < claim->prepared_read) continue;
    if (runner->dispatch_depth == 0 && !runner->release_in_progress &&
        claim->released_epoch < claim->retire_completed_epoch) {
      continue;
    }
    if (runner->dispatch_depth == 0 && !runner->release_in_progress &&
        (claim->reclaim_positions.kernarg_write_position != 0 ||
         claim->reclaim_positions.queue_upload_write_position != 0)) {
      continue;
    }
    break;
  }

  return (iree_host_size_t)(claim->dispatched_read - initial_read);
}

static bool iree_hal_amdgpu_host_queue_completion_runner_is_idle(
    void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  const bool is_idle = !queue->completion.runner_active;
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  return is_idle;
}

static bool iree_hal_amdgpu_host_queue_completion_is_quiescent(
    void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  const bool is_quiescent =
      !queue->completion.runner_active && queue->completion.epilogue_count == 0;
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  return is_quiescent;
}

static bool iree_hal_amdgpu_host_queue_completion_runner_is_closed_and_idle(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  const bool is_closed_and_idle =
      queue->completion.runner_state ==
          IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_CLOSED &&
      !queue->completion.runner_active && queue->completion.epilogue_count == 0;
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  return is_closed_and_idle;
}

static void iree_hal_amdgpu_host_queue_close_completion_runner(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  queue->completion.runner_state =
      IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_CLOSED;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  // completion_drain_mutex remains held. The observer must not reenter the
  // queue or acquire any queue-owned lock.
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_RUNNER_CLOSED_INSTALLED,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  const bool is_quiescent =
      !queue->completion.runner_active && queue->completion.epilogue_count == 0;
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  if (!is_quiescent) {
    iree_notification_await(&queue->completion.runner_notification,
                            iree_hal_amdgpu_host_queue_completion_is_quiescent,
                            queue, iree_infinite_timeout());
  }
}

static iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions_serialized(
    iree_hal_amdgpu_host_queue_t* queue, bool allow_closed,
    bool* out_has_epilogue_token) {
  *out_has_epilogue_token = false;
  iree_hal_amdgpu_host_queue_completion_runner_t* current_runner =
      iree_hal_amdgpu_host_queue_find_current_completion_runner(queue);
  if (current_runner) {
    return iree_hal_amdgpu_host_queue_completion_runner_pump(current_runner);
  }

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_RUNNER_CLAIM,
      allow_closed ? 1 : 0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  // Acquire sole runner ownership. A competing thread waits for the complete
  // outer cursor commit; callbacks on this thread reenter through TLS above.
  for (;;) {
    iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
    const bool is_closed = queue->completion.runner_state ==
                           IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_CLOSED;
    if (is_closed && !allow_closed) {
      iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
      return 0;
    }
    IREE_ASSERT(!allow_closed || is_closed,
                "terminal runner claim requires CLOSED admission");
    if (!queue->completion.runner_active) {
      queue->completion.runner_active = true;
      iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
      break;
    }
    iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
    iree_notification_await(
        &queue->completion.runner_notification,
        iree_hal_amdgpu_host_queue_completion_runner_is_idle, queue,
        iree_infinite_timeout());
  }

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_CLAIM,
      allow_closed ? 1 : 0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  iree_hal_amdgpu_host_queue_completion_runner_t runner = {
      .queue = queue,
      .previous = iree_hal_amdgpu_host_queue_current_completion_runner,
  };
  iree_hal_amdgpu_notification_ring_claim_initialize(&queue->notification_ring,
                                                     &runner.claim);
  runner.frontier_dispatched_epoch = runner.claim.initial_epoch;
  iree_hal_amdgpu_host_queue_current_completion_runner = &runner;
  const uint64_t initial_read = runner.claim.initial_read;
  (void)iree_hal_amdgpu_host_queue_completion_runner_pump(&runner);
  const iree_host_size_t count =
      (iree_host_size_t)(runner.claim.dispatched_read - initial_read);

  // Publish epoch/hot/frontier reuse as one producer-serialized logical
  // commit. No callback or generic release runs under submission_mutex.
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_notification_ring_claim_commit(&queue->notification_ring,
                                                 &runner.claim);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PUBLIC_CURSORS_FINALIZED,
      runner.claim.released_epoch, runner.claim.dispatched_read);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  iree_hal_amdgpu_host_queue_current_completion_runner = runner.previous;
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  // Install the wrapper-tail token before publishing runner inactivity. Seal
  // therefore observes either the runner or its admitted post-drain epilogue.
  ++queue->completion.epilogue_count;
  queue->completion.runner_active = false;
  iree_notification_post(&queue->completion.runner_notification,
                         IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  *out_has_epilogue_token = true;
  return count;
}

static void iree_hal_amdgpu_host_queue_run_completion_epilogue_and_leave(
    iree_hal_amdgpu_host_queue_t* queue) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SAFE_EPILOGUE,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_RELEASE_BEFORE_EPILOGUE,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SAFE_EPILOGUE,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_EPILOGUE_TOKEN_DROP,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  // Wake and count publication are one lock-linearized final queue operation.
  // A normal wrapper touches no queue storage after this unlock.
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  IREE_ASSERT(queue->completion.epilogue_count > 0);
  --queue->completion.epilogue_count;
  iree_notification_post(&queue->completion.runner_notification,
                         IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
}

// Registers an unlocked callback/resource-release epilogue before the
// submission that owns it leaves submission_mutex. Queue teardown closes
// submission admission before it closes the completion runner, so an accepted
// submission always installs this token before the sealer can certify the
// queue. Asynchronous terminal scopes consume the token as their final direct
// queue access; capture scopes may consume it after regaining the submission
// serialization still owned by their public queue call.
void iree_hal_amdgpu_host_queue_enter_submission_epilogue(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  IREE_ASSERT(queue->completion.runner_state ==
              IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_RUNNING);
  ++queue->completion.epilogue_count;
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
}

// Consumes a token installed by
// iree_hal_amdgpu_host_queue_enter_submission_epilogue. Unlocking the mutex is
// the final queue access owned by the token scope.
void iree_hal_amdgpu_host_queue_leave_submission_epilogue(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  IREE_ASSERT(queue->completion.epilogue_count > 0);
  --queue->completion.epilogue_count;
  iree_notification_post(&queue->completion.runner_notification,
                         IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
}

static iree_host_size_t
iree_hal_amdgpu_host_queue_drain_completions_with_lifetime(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;
  (void)iree_hal_amdgpu_host_queue_lifetime_try_acquire(queue, &lifetime_claim);
  bool has_epilogue_token = false;
  const iree_host_size_t count =
      iree_hal_amdgpu_host_queue_drain_completions_serialized(
          queue, /*allow_closed=*/false, &has_epilogue_token);
  if (has_epilogue_token) {
    iree_hal_amdgpu_host_queue_run_completion_epilogue_and_leave(queue);
  }
  // This is the outer safe point. It may synchronously destroy |queue| and
  // its logical device on the current completion service; no queue access is
  // permitted after the release.
  (void)iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
  return count;
}

iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions(
    iree_hal_amdgpu_host_queue_t* queue) {
  return iree_hal_amdgpu_host_queue_drain_completions_with_lifetime(queue);
}

iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions_for_waiter(
    iree_hal_amdgpu_host_queue_t* queue) {
  return iree_hal_amdgpu_host_queue_drain_completions_with_lifetime(queue);
}

static bool iree_hal_amdgpu_host_queue_store_error(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t error) {
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_FAILURE_ADMISSION_LOCK,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  // All queue publishers hold submission_mutex from admission through epoch
  // publication/doorbell. Installing the first error and closing admission in
  // the same critical section therefore gives failure one exact linearization
  // against both already-admitted and not-yet-admitted publishers.
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool won = iree_hal_amdgpu_host_queue_load_error_status_raw(queue) == 0;
  if (won) {
    iree_atomic_store(&queue->error_status, (intptr_t)error,
                      iree_memory_order_release);
  }
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  if (!won) iree_status_free(error);
  return won;
}

static void iree_hal_amdgpu_host_queue_request_completion_thread_stop(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (queue->completion.stop_signal.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(queue->libhsa),
                                    queue->completion.stop_signal, 1);
  }
}

// Permanently closes the queue to further submission. Idempotent, and the only
// writer of |is_shutting_down|.
//
// Every notification epoch is published under submission_mutex, so taking the
// mutex here waits out any publisher already inside the critical section and
// keeps any later one out. A drain that runs after this returns therefore sees
// a published epoch that can no longer advance.
static void iree_hal_amdgpu_host_queue_close_submission(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
}

void iree_hal_amdgpu_host_queue_record_failure(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(queue);
  if (iree_hal_amdgpu_host_queue_store_error(queue, status)) {
    iree_hal_amdgpu_host_queue_request_completion_thread_stop(queue);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_wait_for_setup_epoch(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t epoch) {
  IREE_ASSERT_ARGUMENT(queue);
  if (epoch == 0) return iree_ok_status();
  if (!queue->hardware_queue || !queue->notification_ring.epoch.signal.handle) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU host queue cannot wait for epoch %" PRIu64
                            " without an active hardware queue",
                            epoch);
  }
  if (IREE_UNLIKELY(epoch > (uint64_t)IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU host queue epoch %" PRIu64
                            " exceeds representable HSA signal range",
                            epoch);
  }

  hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  hsa_signal_t stop_signal = queue->completion.stop_signal;
  const hsa_signal_value_t target_signal_value =
      (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - epoch);
  const hsa_signal_value_t compare_value = target_signal_value + 1;

  if (stop_signal.handle) {
    enum {
      IREE_HAL_AMDGPU_EPOCH_WAIT_EPOCH_SIGNAL = 0,
      IREE_HAL_AMDGPU_EPOCH_WAIT_STOP_SIGNAL = 1,
      IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT = 2,
    };
    hsa_signal_t signals[IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT] = {
        epoch_signal,
        stop_signal,
    };
    hsa_signal_condition_t conditions[IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT] =
        {
            HSA_SIGNAL_CONDITION_LT,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT] = {
        compare_value,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(queue->libhsa), IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT,
        signals, conditions, values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);
    if (IREE_UNLIKELY(signal_index >=
                      IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT)) {
      iree_status_t error = iree_status_from_code(IREE_STATUS_INTERNAL);
      iree_hal_amdgpu_host_queue_record_failure(queue,
                                                iree_status_clone(error));
      return error;
    } else if (signal_index == IREE_HAL_AMDGPU_EPOCH_WAIT_STOP_SIGNAL) {
      iree_status_t error =
          iree_hal_amdgpu_host_queue_clone_error_status(queue);
      if (!iree_status_is_ok(error)) return error;
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
  } else {
    (void)iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(queue->libhsa), epoch_signal, HSA_SIGNAL_CONDITION_LT,
        compare_value, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_clone_error_status(queue));
  iree_hal_amdgpu_host_queue_drain_completions_for_waiter(queue);
  return iree_hal_amdgpu_host_queue_clone_error_status(queue);
}

static hsa_signal_value_t iree_hal_amdgpu_host_queue_last_drained_signal_value(
    iree_hal_amdgpu_host_queue_t* queue) {
  const uint64_t last_drained_epoch = (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_acquire);
  return (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                              last_drained_epoch);
}

static bool iree_hal_amdgpu_host_queue_epoch_is_complete_or_terminal(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t epoch) {
  if (iree_hal_amdgpu_host_queue_has_error(queue) || epoch == 0) return true;
  if (!queue->hardware_queue || !queue->notification_ring.epoch.signal.handle ||
      epoch > (uint64_t)IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE) {
    return false;
  }
  const hsa_signal_value_t target_signal_value =
      (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - epoch);
  const hsa_signal_value_t signal_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(queue->libhsa),
      iree_hal_amdgpu_notification_ring_epoch_signal(
          &queue->notification_ring));
  return signal_value <= target_signal_value;
}

// Requires submission_mutex and a permanently closed queue. Returns true only
// when all callback-capable notification/reclaim storage has been consumed.
static bool iree_hal_amdgpu_host_queue_notification_storage_is_inert_locked(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_notification_ring_t* ring = &queue->notification_ring;
  if ((uint64_t)iree_atomic_load(&ring->write, iree_memory_order_acquire) !=
          (uint64_t)iree_atomic_load(&ring->read, iree_memory_order_acquire) ||
      (uint64_t)iree_atomic_load(&ring->frontier_ring.write,
                                 iree_memory_order_acquire) !=
          (uint64_t)iree_atomic_load(&ring->frontier_ring.read,
                                     iree_memory_order_acquire) ||
      (uint64_t)iree_atomic_load(&ring->epoch.last_drained,
                                 iree_memory_order_acquire) !=
          (uint64_t)iree_atomic_load(&ring->epoch.last_published,
                                     iree_memory_order_acquire) ||
      (uint64_t)iree_atomic_load(&ring->epoch.last_published,
                                 iree_memory_order_acquire) !=
          ring->epoch.next_submission) {
    return false;
  }
  for (uint32_t i = 0; i < ring->capacity; ++i) {
    const iree_hal_amdgpu_reclaim_entry_t* entry = &ring->reclaim_entries[i];
    if (entry->resources || entry->resource_set || entry->count != 0 ||
        entry->signal_semaphore_count != 0 || entry->pre_signal_action.fn) {
      return false;
    }
  }
  return true;
}

// Requires submission_mutex. A certificate is usable only while it names the
// exact permanently closed frontier and that frontier remains complete or
// terminal. The outcome check makes corrupted or prematurely published
// certificates fail closed rather than suppressing the defensive wait.
static bool iree_hal_amdgpu_host_queue_has_idle_certificate_locked(
    iree_hal_amdgpu_host_queue_t* queue) {
  return queue->is_shutting_down && queue->idle_certificate_valid &&
         iree_hal_amdgpu_host_queue_completion_runner_is_closed_and_idle(
             queue) &&
         queue->hardware_queue_retired && queue->hardware_queue == NULL &&
         queue->completion.thread == NULL &&
         queue->system_event_target == NULL &&
         queue->epoch_registration_table == NULL &&
         queue->frontier_tracker == NULL &&
         iree_hal_amdgpu_host_queue_load_error_status_raw(queue) == 0 &&
         queue->pending_head == NULL &&
         queue->active_staging_transfer_head == NULL &&
         queue->shutdown_staging_transfer_head == NULL &&
         queue->active_file_action_head == NULL &&
         queue->shutdown_file_action_head == NULL &&
         queue->idle_certificate_epoch ==
             queue->notification_ring.epoch.next_submission &&
         iree_hal_amdgpu_host_queue_notification_storage_is_inert_locked(
             queue) &&
         iree_hal_amdgpu_host_queue_post_drain_is_empty(queue);
}

static bool iree_hal_amdgpu_host_queue_seal_is_complete(void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool is_complete =
      iree_hal_amdgpu_host_queue_has_idle_certificate_locked(queue);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return is_complete;
}

static void iree_hal_amdgpu_host_queue_record_invalid_idle_state(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t epoch) {
  (void)epoch;
  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_status_from_code(IREE_STATUS_INTERNAL));
}

// Retires every producer that can still publish an owner-sensitive callback,
// destroys the native queue, and consumes all notification/reclaim state. The
// final epoch has already been proven complete or terminal and the completion
// thread/publisher registries have already joined.
static void iree_hal_amdgpu_host_queue_retire_native_and_callbacks(
    iree_hal_amdgpu_host_queue_t* queue) {
  // System-event delivery holds its registry mutex across record_failure.
  // Exact retirement is therefore a quiescence barrier for in-flight delivery.
  iree_hal_amdgpu_system_event_agent_target_t* system_event_target =
      queue->system_event_target;
  iree_hal_amdgpu_system_event_retire_queue_target(system_event_target, queue);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->system_event_target == system_event_target);
  queue->system_event_target = NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  // HSA queue destruction quiesces its native error callback. Keep the
  // notification/error storage alive until after that callback can no longer
  // run, then publish native retirement under submission_mutex.
  hsa_queue_t* hardware_queue = queue->hardware_queue;
  if (hardware_queue) {
    iree_hal_amdgpu_hsa_queue_destroy(queue->libhsa, hardware_queue);
  }
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->hardware_queue == hardware_queue);
  queue->hardware_queue = NULL;
  queue->hardware_queue_retired = true;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  // No native or registry producer can install another queue error. The
  // sealer is the sole runner admitted after CLOSED and consumes every
  // remaining normal/failure callback through the same A-E claim without
  // holding completion_drain_mutex.
  bool has_epilogue_token = false;
  (void)iree_hal_amdgpu_host_queue_drain_completions_serialized(
      queue, /*allow_closed=*/true, &has_epilogue_token);
  IREE_ASSERT(has_epilogue_token,
              "terminal sealer claim must own an exact epilogue token");
  iree_hal_amdgpu_host_queue_run_completion_epilogue_and_leave(queue);

  // Once every hot entry has been consumed no future reader needs cold
  // frontier snapshots. Collapse any trailing wrap/stale snapshots so the
  // certificate directly proves the complete ring is empty.
  const uint64_t frontier_write = (uint64_t)iree_atomic_load(
      &queue->notification_ring.frontier_ring.write, iree_memory_order_acquire);
  iree_atomic_store(&queue->notification_ring.frontier_ring.read,
                    (int64_t)frontier_write, iree_memory_order_release);

  // Reclaim callbacks may enqueue a final capacity/failure continuation.
  // Keep the sticky queue failure published until every such continuation has
  // observed it and the terminal owner has reached a post-drain fixed point.
  iree_hal_amdgpu_host_queue_drain_shutdown_deferred(queue);

  // Retire remaining shared publication while its parent device and callback
  // machinery are still live. The notification ring signal remains allocated
  // until inert storage destruction, but no new peer lookup can discover it.
  iree_hal_amdgpu_epoch_signal_table_t* epoch_registration_table =
      queue->epoch_registration_table;
  if (epoch_registration_table) {
    iree_hal_amdgpu_epoch_signal_table_deregister(
        epoch_registration_table, iree_async_axis_queue_index(queue->axis));
  }
  iree_async_frontier_tracker_t* frontier_tracker = queue->frontier_tracker;
  const iree_async_axis_t axis = queue->axis;
  if (frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        frontier_tracker, axis, iree_status_from_code(IREE_STATUS_CANCELLED));
  }
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->epoch_registration_table == epoch_registration_table);
  IREE_ASSERT(queue->frontier_tracker == frontier_tracker);
  queue->epoch_registration_table = NULL;
  queue->frontier_tracker = NULL;
  queue->axis = 0;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  // Native, completion, post-drain, and frontier producers are now quiescent.
  // Take and free the exact queue-owned sticky status before certification.
  iree_status_t error = (iree_status_t)iree_atomic_exchange(
      &queue->error_status, 0, iree_memory_order_acq_rel);
  iree_status_free(error);
}

void iree_hal_amdgpu_host_queue_wait_idle_before_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->is_shutting_down,
              "queue admission must close before the idle wait");
  if (iree_hal_amdgpu_host_queue_has_idle_certificate_locked(queue)) {
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    return;
  }
  if (queue->seal_in_progress) {
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    iree_notification_await(&queue->seal_notification,
                            iree_hal_amdgpu_host_queue_seal_is_complete, queue,
                            iree_infinite_timeout());
    return;
  }
  IREE_ASSERT(!queue->hardware_queue_retired,
              "a retired native queue cannot rebuild an idle certificate");
  queue->seal_in_progress = true;
#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  // Test-only readiness boundary. The observer runs under submission_mutex and
  // therefore must not reenter the queue or acquire queue-owned locks.
  iree_hal_amdgpu_host_queue_test_notify_phase(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SEAL,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SEAL_OWNER_ACQUIRED,
      /*value0=*/0, /*value1=*/0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  // Active file/proactor publishers are not represented by pending_head or
  // the notification ring. Close their exact publisher set before draining
  // deferred work. Joining happens after hardware and post-drain callbacks run
  // so a capacity-parked or GPU-copy-backed transfer cannot deadlock seal.
  iree_hal_amdgpu_staging_transfer_cancel_all(queue);
  iree_hal_amdgpu_file_action_cancel_all(queue);
  iree_hal_amdgpu_host_queue_drain_shutdown_deferred(queue);
  uint64_t final_submitted_epoch = 0;
  for (;;) {
    iree_slim_mutex_lock(&queue->locks.submission_mutex);
    queue->idle_certificate_valid = false;
    const uint64_t submitted_epoch =
        queue->notification_ring.epoch.next_submission;
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);

    if (!iree_hal_amdgpu_host_queue_epoch_is_complete_or_terminal(
            queue, submitted_epoch)) {
      if (!queue->hardware_queue ||
          !queue->notification_ring.epoch.signal.handle ||
          submitted_epoch > (uint64_t)IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE) {
        iree_hal_amdgpu_host_queue_record_invalid_idle_state(queue,
                                                             submitted_epoch);
      } else {
        hsa_signal_t epoch_signal =
            iree_hal_amdgpu_notification_ring_epoch_signal(
                &queue->notification_ring);
        hsa_signal_t stop_signal = queue->completion.stop_signal;
        const hsa_signal_value_t idle_epoch_signal_value =
            (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                                 submitted_epoch);
        const hsa_signal_value_t idle_compare_value =
            idle_epoch_signal_value + 1;

        // Submission is shut out. Wait only for hardware to publish the final
        // submitted epoch so no late CP write can race with HSA queue/signal
        // teardown; the completion thread still owns notification draining.
        if (stop_signal.handle) {
          enum {
            IREE_HAL_AMDGPU_IDLE_WAIT_EPOCH_SIGNAL = 0,
            IREE_HAL_AMDGPU_IDLE_WAIT_STOP_SIGNAL = 1,
            IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT = 2,
          };
          hsa_signal_t signals[IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT] = {
              epoch_signal,
              stop_signal,
          };
          hsa_signal_condition_t
              conditions[IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT] = {
                  HSA_SIGNAL_CONDITION_LT,
                  HSA_SIGNAL_CONDITION_NE,
              };
          hsa_signal_value_t values[IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT] = {
              idle_compare_value,
              0,
          };
          const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
              IREE_LIBHSA(queue->libhsa),
              IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT, signals, conditions,
              values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
              /*satisfying_value=*/NULL);
          if (IREE_UNLIKELY(signal_index >=
                            IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT)) {
            iree_hal_amdgpu_host_queue_record_failure(
                queue, iree_status_from_code(IREE_STATUS_INTERNAL));
          } else if (IREE_UNLIKELY(
                         signal_index ==
                             IREE_HAL_AMDGPU_IDLE_WAIT_STOP_SIGNAL &&
                         !iree_hal_amdgpu_host_queue_has_error(queue))) {
            iree_hal_amdgpu_host_queue_record_failure(
                queue, iree_status_from_code(IREE_STATUS_INTERNAL));
          }
        } else {
          (void)iree_hsa_signal_wait_scacquire(
              IREE_LIBHSA(queue->libhsa), epoch_signal, HSA_SIGNAL_CONDITION_LT,
              idle_compare_value, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        }
      }
    }

    iree_slim_mutex_lock(&queue->locks.submission_mutex);
    const bool epoch_is_unchanged =
        queue->notification_ring.epoch.next_submission == submitted_epoch;
    const bool epoch_is_complete_or_terminal =
        iree_hal_amdgpu_host_queue_epoch_is_complete_or_terminal(
            queue, submitted_epoch);
    if (epoch_is_unchanged && epoch_is_complete_or_terminal) {
      final_submitted_epoch = submitted_epoch;
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      break;
    }
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  }

  // No hardware epoch can advance after the exact closed frontier completed.
  // Stop and join the completion service before the final notification and
  // deferred drains so no callback can be published behind the certificate.
  iree_thread_t* completion_thread = queue->completion.thread;
  if (completion_thread) {
    iree_hal_amdgpu_host_queue_request_completion_thread_stop(queue);
    iree_hal_amdgpu_host_queue_completion_service_t* current_service =
        iree_hal_amdgpu_host_queue_current_completion_service;
    const bool is_current_service =
        current_service && current_service->queue == queue;
    if (is_current_service) {
      // A callback consumed the last public owner while this service held its
      // safe-scope references. Transfer teardown to this outer safe point:
      // remove the current handle from every future join set, mark the copied
      // trampoline for immediate return, and self-release/detach the wrapper.
      // The generic pthread/Darwin/Win32 thread implementations guarantee no
      // wrapper access after the entry point returns.
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
      IREE_ASSERT(queue->completion.thread == completion_thread);
      queue->completion.thread = NULL;
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      current_service->exit_requested = true;
      iree_thread_release(completion_thread);
    } else {
      iree_thread_release(completion_thread);
      iree_slim_mutex_lock(&queue->locks.submission_mutex);
      IREE_ASSERT(queue->completion.thread == completion_thread);
      queue->completion.thread = NULL;
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    }
  }
  iree_hal_amdgpu_host_queue_drain_completions(queue);
  iree_hal_amdgpu_host_queue_drain_shutdown_deferred(queue);
  iree_hal_amdgpu_staging_transfer_await_all(queue);
  iree_hal_amdgpu_file_action_await_all(queue);
  // A terminal callback can enqueue one final failure-delivery continuation.
  iree_hal_amdgpu_host_queue_drain_shutdown_deferred(queue);
  // Close direct-waiter/completion claim admission and join any claim that
  // began before the close. The sealer will make the sole terminal claim after
  // retiring native and system-event producers.
  iree_hal_amdgpu_host_queue_close_completion_runner(queue);
  iree_hal_amdgpu_host_queue_retire_native_and_callbacks(queue);

  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(queue->pending_head == NULL,
              "sealed queue must not retain deferred operations");
  IREE_ASSERT(queue->active_staging_transfer_head == NULL,
              "sealed queue must not retain staged/proactor publishers");
  IREE_ASSERT(queue->shutdown_staging_transfer_head == NULL,
              "sealed queue must join staged/proactor publishers");
  IREE_ASSERT(queue->active_file_action_head == NULL,
              "sealed queue must not retain direct-file publishers");
  IREE_ASSERT(queue->shutdown_file_action_head == NULL,
              "sealed queue must join direct-file publishers");
  IREE_ASSERT(queue->notification_ring.epoch.next_submission ==
              final_submitted_epoch);
  IREE_ASSERT(
      iree_hal_amdgpu_host_queue_completion_runner_is_closed_and_idle(queue),
      "sealed queue must have CLOSED and joined completion-runner admission");
  IREE_ASSERT(queue->hardware_queue_retired && !queue->hardware_queue,
              "sealed queue must retire its native queue");
  IREE_ASSERT(!queue->system_event_target,
              "sealed queue must retire exact system-event delivery");
  IREE_ASSERT(iree_hal_amdgpu_host_queue_load_error_status_raw(queue) == 0,
              "sealed queue must consume its terminal error status");
  IREE_ASSERT(
      iree_hal_amdgpu_host_queue_notification_storage_is_inert_locked(queue),
      "sealed queue must consume notification/reclaim storage");
  IREE_ASSERT(iree_hal_amdgpu_host_queue_post_drain_is_empty(queue),
              "sealed queue must not retain post-drain callbacks");
  queue->idle_certificate_epoch = final_submitted_epoch;
  queue->idle_certificate_valid = true;
  queue->seal_in_progress = false;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_notification_post(&queue->seal_notification, IREE_ALL_WAITERS);
}

// Completion thread entry point. Blocks in HSA until either the queue epoch
// signal changes or teardown/error signals the stop signal. Completion wakeups
// drain normally; stop/error wakeups perform one final drain/fail before exit.
static int iree_hal_amdgpu_host_queue_completion_thread_main(void* entry_arg) {
  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_host_queue_completion_thread_start");
    IREE_TRACE_ZONE_END(z0);
  }
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)entry_arg;
  iree_hal_amdgpu_host_queue_completion_service_t service = {
      .queue = queue,
      .previous = iree_hal_amdgpu_host_queue_current_completion_service,
  };
  iree_hal_amdgpu_host_queue_current_completion_service = &service;

  enum {
    IREE_HAL_AMDGPU_COMPLETION_WAIT_EPOCH_SIGNAL = 0,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_WORK_SIGNAL = 1,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_STOP_SIGNAL = 2,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT = 3,
  };

  hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  hsa_signal_t work_signal = queue->completion.work_signal;
  hsa_signal_t stop_signal = queue->completion.stop_signal;
  hsa_signal_value_t last_epoch_value =
      iree_hal_amdgpu_host_queue_last_drained_signal_value(queue);

  bool keep_running = true;
  while (keep_running) {
    hsa_signal_t signals[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
        epoch_signal,
        work_signal,
        stop_signal,
    };
    hsa_signal_condition_t
        conditions[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
            HSA_SIGNAL_CONDITION_NE,
            HSA_SIGNAL_CONDITION_NE,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
        last_epoch_value,
        0,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(queue->libhsa),
        IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT, signals, conditions,
        values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);

    {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z0, "iree_hal_amdgpu_host_queue_completion_thread_pump");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_index);

      // Hold both queue and device across every callback-capable lane and the
      // pending-operation failure tail below. User/resource callbacks may drop
      // their final public references, but destruction cannot begin until this
      // outer safe point. If destruction already owns lifetime the borrowed
      // service is still joined by that destructor.
      iree_hal_amdgpu_host_queue_lifetime_claim_t lifetime_claim;
      (void)iree_hal_amdgpu_host_queue_lifetime_try_acquire(queue,
                                                            &lifetime_claim);

      if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_WORK_SIGNAL) {
        // Reset before draining. A producer that enqueues after the exchange
        // stores 1 again and forces a later wake; an action enqueued before the
        // exchange is already visible through post_drain_mutex.
        (void)iree_hsa_signal_exchange_scacquire(IREE_LIBHSA(queue->libhsa),
                                                 work_signal, 0);
      }

      if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_EPOCH_SIGNAL ||
          signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_WORK_SIGNAL) {
        iree_hal_amdgpu_host_queue_drain_completions(queue);
        // Arm the next wait from the epoch we actually drained, not from a raw
        // HSA signal load. A GPU completion can race with the drain and update
        // the signal after drain() sampled it; observing that newer value here
        // would mark an undrained epoch as already seen and could sleep forever
        // with a user semaphore still pending.
        last_epoch_value =
            iree_hal_amdgpu_host_queue_last_drained_signal_value(queue);
      }

      // An out-of-range index means every signal handle passed was null or
      // invalid, so nothing can ever wake this wait again. Fail the queue and
      // take the terminal path below rather than spinning on a dead wait.
      if (IREE_UNLIKELY(signal_index >=
                        IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT)) {
        iree_hal_amdgpu_host_queue_record_failure(
            queue, iree_status_from_code(IREE_STATUS_INTERNAL));
      }

      if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_STOP_SIGNAL ||
          iree_hal_amdgpu_host_queue_has_error(queue)) {
        // Close admission before the final drain so no epoch can be published
        // behind it. On the teardown path admission is already closed and this
        // is a no-op; on the failure path it is what makes the drain final.
        iree_hal_amdgpu_host_queue_close_submission(queue);
        iree_hal_amdgpu_host_queue_drain_completions(queue);
        const intptr_t error_status =
            iree_hal_amdgpu_host_queue_load_error_status_raw(queue);
        if (error_status != 0) {
          // Capacity-parked pending operations are retried by post-drain
          // callbacks and a detached pass can enqueue more. Run one fairness
          // pass before cancellation, then let the failure owner drain the
          // closed queue to a fixed point while its lifetime claim is active.
          iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
          iree_hal_amdgpu_host_queue_cancel_pending(
              queue, (iree_status_t)error_status);
          // Cancellation can enqueue a final capacity/failure continuation.
          // Consume every detached batch while the outer lifetime claim is
          // still active; the completion service exits after this branch.
          iree_hal_amdgpu_host_queue_await_post_drain_idle(queue);
        }
        keep_running = false;
      }

      IREE_TRACE_ZONE_END(z0);

      // May synchronously self-seal and free queue/device storage. This must be
      // the last operation in the iteration; only the stack-owned service flag
      // may be inspected afterward.
      const bool exit_now =
          iree_hal_amdgpu_host_queue_lifetime_release(&lifetime_claim);
      if (exit_now || service.exit_requested) {
        iree_hal_amdgpu_host_queue_current_completion_service =
            service.previous;
        return 0;
      }
    }
  }

  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_host_queue_completion_thread_exit");
    IREE_TRACE_ZONE_END(z0);
  }
  iree_hal_amdgpu_host_queue_current_completion_service = service.previous;
  return 0;
}

// HSA queue error callback, invoked by the HSA runtime on an internal thread
// when this queue takes an unrecoverable error: an invalid or unsupported AQL
// packet, an illegal instruction, a wave exception, a memory aperture
// violation, exhausted scratch or registers, or a device-level fault such as
// ECC.
//
// A plain VM memory fault does not arrive here. The runtime installs one of two
// queue handlers depending on whether the kernel driver reports support for
// exception debugging: the one it installs with support recognizes a wave
// memory violation, stamps the queue and hands the fault to its process-wide VM
// fault path without calling back here, and the one it installs without support
// carries no memory-fault code in its mapping at all. Either way the fault is
// delivered to the process-wide system event handlers, which is where this
// driver observes it - see system_event.c. Both callbacks can fail the same
// queue, so both converge on the same terminal transition and handling must
// stay idempotent.
static void iree_hal_amdgpu_host_queue_error_callback(hsa_status_t status,
                                                      hsa_queue_t* source,
                                                      void* data) {
  iree_hal_amdgpu_host_queue_t* queue = (iree_hal_amdgpu_host_queue_t*)data;

  // Convert the HSA error to an IREE status with diagnostic information.
  iree_status_t error = iree_status_from_hsa_status(
      __FILE__, __LINE__, status, "hsa_queue_error_callback",
      "GPU queue encountered an unrecoverable error");

  // First-error-wins: store the error with release semantics so the status
  // payload (heap-allocated string, backtrace) is visible to any thread that
  // loads with acquire. If another error already won the race, free ours.
  iree_hal_amdgpu_host_queue_record_failure(queue, error);
}

static iree_status_t iree_hal_amdgpu_host_queue_map_native_priority(
    iree_hal_queue_priority_t priority,
    hsa_amd_queue_priority_t* out_native_priority) {
  hsa_amd_queue_priority_t native_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  switch (priority) {
    case -1:
      native_priority = HSA_AMD_QUEUE_PRIORITY_LOW;
      break;
    case IREE_HAL_QUEUE_PRIORITY_NORMAL:
      native_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
      break;
    case 1:
      native_priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AMDGPU queue priority %" PRId32,
                              priority);
  }
  *out_native_priority = native_priority;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_amdgpu_host_queue_params_t* params,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  if (IREE_UNLIKELY(!params->hardware.execution_resource_topology)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU host queue requires an execution-resource topology");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_execution_resource_topology_verify(
      params->hardware.execution_resource_topology));
  if (!iree_host_size_is_power_of_two(params->capacity.aql_packet_count) ||
      !iree_host_size_is_power_of_two(params->capacity.notification_count) ||
      !iree_host_size_is_power_of_two(params->capacity.kernarg_block_count) ||
      (params->capacity.upload_byte_count != 0 &&
       !iree_host_size_is_power_of_two(params->capacity.upload_byte_count))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "all enabled capacities must be powers of two");
  }
  if (params->capacity.kernarg_block_count / 2u <
      params->capacity.aql_packet_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ring capacity must be at least 2x the AQL ring capacity "
        "to cover one tail-padding gap at wrap (got kernarg_blocks=%u, "
        "aql_packets=%u)",
        params->capacity.kernarg_block_count,
        params->capacity.aql_packet_count);
  }
  const bool is_cooperative =
      iree_any_bit_set(params->identity.params.features,
                       IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH);
  if (IREE_UNLIKELY(is_cooperative &&
                    params->identity.params.features !=
                        IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU cooperative queue request has unsupported feature bits");
  }
  if (IREE_UNLIKELY(is_cooperative && params->identity.params.priority !=
                                          IREE_HAL_QUEUE_PRIORITY_NORMAL)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU cooperative queues require normal "
                            "scheduling priority");
  }
  if (IREE_UNLIKELY(is_cooperative &&
                    params->identity.params.execution_resources.count != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU cooperative queues require the complete execution-resource "
        "set");
  }
  if (IREE_UNLIKELY(is_cooperative &&
                    params->hardware.grid_sync_strategy ==
                        IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_NONE)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU cooperative grid synchronization is unavailable for this "
        "target");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));
  iree_hal_queue_initialize(params->identity.family, &params->identity.params,
                            &iree_hal_amdgpu_host_queue_vtable,
                            &out_queue->base);
  out_queue->libhsa = params->hardware.libhsa;
  out_queue->logical_device = params->coordination.logical_device;
  out_queue->system_event_target = params->coordination.system_event_target;
  out_queue->execution_resource_topology =
      params->hardware.execution_resource_topology;
  out_queue->dispatch_concurrency_capabilities =
      params->hardware.dispatch_concurrency_capabilities;
  out_queue->hostcall_buffer = params->hardware.hostcall_buffer;
  out_queue->proactor = params->coordination.proactor;
  out_queue->frontier_tracker = params->coordination.frontier_tracker;
  out_queue->host_allocator = params->host_allocator;
  out_queue->epoch_table = params->coordination.epoch_table;

  // Submission pipeline state.
  iree_slim_mutex_initialize(&out_queue->locks.submission_mutex);
  iree_slim_mutex_initialize(&out_queue->locks.completion_drain_mutex);
  out_queue->completion.runner_state =
      IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_RUNNING;
  iree_slim_mutex_initialize(&out_queue->locks.post_drain_mutex);
  iree_notification_initialize(&out_queue->completion.runner_notification);
  iree_notification_initialize(&out_queue->post_drain_notification);
  iree_notification_initialize(&out_queue->seal_notification);
  iree_slim_mutex_initialize(&out_queue->profiling.event_mutex);
  out_queue->profiling.memory = params->memory.profiling;
  out_queue->axis = params->identity.axis;
  out_queue->wait_barrier_strategy = params->hardware.wait_barrier_strategy;
  out_queue->grid_sync_strategy = params->hardware.grid_sync_strategy;
  out_queue->vendor_packet_capabilities =
      params->hardware.vendor_packet_capabilities;
  out_queue->pm4_timestamp_strategy = params->hardware.pm4_timestamp_strategy;
  out_queue->physical_queue_ordinal = params->identity.physical_queue_ordinal;
  out_queue->last_signal.semaphore = NULL;
  out_queue->last_signal.epoch = 0;
  out_queue->feedback_state = iree_hal_amdgpu_feedback_state_is_enabled(
                                  params->coordination.feedback_state)
                                  ? params->coordination.feedback_state
                                  : NULL;
  out_queue->block_pool = params->memory.block_pool;
  out_queue->can_publish_frontier = true;
  out_queue->transfer_context = params->memory.transfer_context;
  out_queue->default_pool_set = params->memory.default_pool_set;
  out_queue->default_pool = params->memory.default_pool;
  out_queue->transient_buffer_pool = params->memory.transient_buffer_pool;
  out_queue->staging_pool = params->memory.staging_pool;
  out_queue->device_ordinal = params->identity.device_ordinal;
  out_queue->pending_head = NULL;
  iree_async_frontier_initialize(iree_hal_amdgpu_host_queue_frontier(out_queue),
                                 /*entry_count=*/0);

  // The optional tracker semaphore is an iree_async_semaphore_t bridge for
  // CPU-side wait integration. The queue's GPU-visible HSA epoch signal is
  // created by the notification ring below and registered in the epoch table.
  iree_status_t status = iree_async_frontier_tracker_register_axis(
      params->coordination.frontier_tracker, params->identity.axis,
      /*semaphore=*/NULL);

  // Create the host-only stop signal before the hardware queue so the HSA error
  // callback always has a valid signal to wake if queue creation races with an
  // asynchronous fault.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(params->hardware.libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_queue->completion.stop_signal);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(params->hardware.libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_queue->completion.work_signal);
  }

  // Create the HSA hardware AQL queue.
  //
  // HSA_QUEUE_TYPE_MULTI is required (not just an optimization). Once command
  // buffers start performing device-side enqueue, the CP itself becomes a
  // concurrent producer alongside the host submission path, so the queue must
  // permit multiple concurrent producers. The host-side reserve already uses
  // an atomic fetch_add on the write index, which is well-defined only on
  // MULTI queues.
  hsa_queue_t* hardware_queue = NULL;
  hsa_amd_queue_priority_t native_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  const uint32_t native_mask_bit_count =
      iree_hal_amdgpu_queue_execution_resource_mask_bit_count(
          params->hardware.execution_resource_topology);
  const iree_host_size_t native_mask_word_count = native_mask_bit_count / 32u;
  uint32_t* native_mask = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_map_native_priority(
        params->identity.params.priority, &native_priority);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        params->host_allocator, native_mask_word_count, sizeof(*native_mask),
        (void**)&native_mask);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_queue_execution_resource_write_mask(
        params->hardware.execution_resource_topology,
        params->identity.params.execution_resources, native_mask_bit_count,
        native_mask);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_hsa_queue_params_t hsa_queue_params = {
        .libhsa = params->hardware.libhsa,
        .agent = params->hardware.gpu_agent,
        .packet_count = params->capacity.aql_packet_count,
        .type =
            is_cooperative ? HSA_QUEUE_TYPE_COOPERATIVE : HSA_QUEUE_TYPE_MULTI,
        .priority = native_priority,
        .compute_unit_mask_bit_count = native_mask_bit_count,
        .compute_unit_mask = native_mask,
        .error_callback = iree_hal_amdgpu_host_queue_error_callback,
        .error_callback_data = out_queue,
    };
    status =
        iree_hal_amdgpu_hsa_queue_create(&hsa_queue_params, &hardware_queue);
  }
  iree_allocator_free(params->host_allocator, native_mask);

  // Initialize the AQL ring from the hardware queue. HSA may return a queue
  // with a different capacity than requested, notably for cooperative queues,
  // so every AQL-indexed sidecar must use the achieved capacity.
  uint32_t actual_aql_packet_count = params->capacity.aql_packet_count;
  if (iree_status_is_ok(status)) {
    out_queue->hardware_queue = hardware_queue;
    actual_aql_packet_count = hardware_queue->size;
    if (IREE_UNLIKELY(
            !iree_host_size_is_power_of_two(actual_aql_packet_count))) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HSA returned an invalid AQL queue capacity of %u packets",
          actual_aql_packet_count);
    } else if (IREE_UNLIKELY(params->capacity.kernarg_block_count / 2u <
                             actual_aql_packet_count)) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "HSA returned an AQL queue larger than the queue-owned kernarg "
          "ring can support (kernarg_blocks=%u, aql_packets=%u)",
          params->capacity.kernarg_block_count, actual_aql_packet_count);
    } else {
      iree_hal_amdgpu_aql_ring_initialize(
          params->hardware.libhsa, (iree_amd_queue_t*)hardware_queue,
          params->hardware.aql_execution_mode, &out_queue->aql_ring);
    }
  }

  // Initialize the kernarg ring from the selected HSA memory pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_kernarg_ring_initialize(
        params->hardware.libhsa, &params->memory.kernarg,
        params->capacity.kernarg_block_count, &out_queue->kernarg_ring);
  }

  // Initialize the optional queue-control upload ring from the same
  // host-visible memory policy as queue-owned kernargs. A zero capacity keeps
  // future device-side fixup storage opt-in and avoids charging every queue for
  // an unused allocation.
  if (iree_status_is_ok(status) && params->capacity.upload_byte_count != 0) {
    const iree_hal_amdgpu_queue_upload_ring_memory_t upload_memory = {
        .memory_pool = params->memory.kernarg.memory_pool,
        .access_agents = params->memory.kernarg.access_agents,
        .access_agent_count = params->memory.kernarg.access_agent_count,
        .publication = params->memory.kernarg.publication,
    };
    status = iree_hal_amdgpu_queue_upload_ring_initialize(
        params->hardware.libhsa, &upload_memory,
        params->capacity.upload_byte_count, &out_queue->queue_upload_ring);
  }

  // Initialize the optional PM4 IB slot buffer. Capability-driven allocation
  // keeps dynamic PM4 storage available on CDNA queues that use BARRIER_VALUE
  // for waits but still support AQL PM4-IB snippets for other features. The
  // buffer is indexed by AQL packet id and inherits AQL ring
  // backpressure/reuse; there is no separate PM4 producer or reclaim position.
  if (iree_status_is_ok(status) &&
      (params->hardware.vendor_packet_capabilities &
       IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB)) {
    status = iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
        params->hardware.libhsa, params->hardware.gpu_agent,
        params->memory.pm4_ib_pool, actual_aql_packet_count, out_queue);
  }

  // Initialize the notification ring (creates epoch signal + entry buffer).
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_notification_ring_initialize(
        params->hardware.libhsa, params->memory.block_pool,
        params->capacity.notification_count, params->host_allocator,
        &out_queue->notification_ring);
  }

  // Provisioned queues publish their epoch signals for device-side cross-queue
  // barriers. Independently releasable queues must not publish: an already
  // submitted peer packet can retain the raw signal beyond HAL queue release.
  // They still use epoch_table above to look up provisioned peer signals and
  // resolve every other wait through software deferral.
  if (iree_status_is_ok(status) &&
      params->coordination.epoch_registration_table) {
    iree_hal_amdgpu_epoch_signal_table_register(
        params->coordination.epoch_registration_table,
        iree_async_axis_queue_index(params->identity.axis),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &out_queue->notification_ring));
    out_queue->epoch_registration_table =
        params->coordination.epoch_registration_table;
  }

  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t thread_params;
    memset(&thread_params, 0, sizeof(thread_params));
    char thread_name[32] = {0};
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name), "amdgpu-d%uq%u-c",
             (unsigned)iree_async_axis_device_index(params->identity.axis),
             (unsigned)iree_async_axis_queue_index(params->identity.axis));
    thread_params.name = iree_make_cstring_view(thread_name);
    thread_params.initial_affinity =
        params->coordination.completion_thread_affinity;
    status = iree_thread_create(
        iree_hal_amdgpu_host_queue_completion_thread_main, out_queue,
        thread_params, params->host_allocator, &out_queue->completion.thread);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_deinitialize(out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_allocate(
    const iree_hal_amdgpu_host_queue_params_t* params,
    iree_hal_amdgpu_system_event_agent_target_t* system_event_target,
    iree_hal_amdgpu_host_queue_release_slot_callback_t release_slot,
    bool retain_parent_device, iree_hal_amdgpu_host_queue_t** out_queue) {
  iree_host_size_t total_size = 0;
  iree_host_size_t resource_ordinals_offset = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(iree_hal_amdgpu_host_queue_t), &total_size,
                         IREE_STRUCT_FIELD_ALIGNED(
                             params->identity.params.execution_resources.count,
                             iree_hal_queue_execution_resource_ordinal_t, 1,
                             &resource_ordinals_offset)));

  iree_hal_amdgpu_host_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_aligned(params->host_allocator, total_size,
                                    iree_alignof(iree_hal_amdgpu_host_queue_t),
                                    /*offset=*/0, (void**)&queue));

  iree_hal_amdgpu_host_queue_params_t owned_params = *params;
  IREE_ASSERT(!owned_params.coordination.system_event_target ||
              owned_params.coordination.system_event_target ==
                  system_event_target);
  owned_params.coordination.system_event_target = system_event_target;
  const iree_host_size_t resource_count =
      params->identity.params.execution_resources.count;
  if (resource_count) {
    iree_hal_queue_execution_resource_ordinal_t* resource_ordinals =
        (iree_hal_queue_execution_resource_ordinal_t*)((uint8_t*)queue +
                                                       resource_ordinals_offset);
    memcpy(resource_ordinals,
           params->identity.params.execution_resources.ordinals,
           resource_count * sizeof(*resource_ordinals));
    owned_params.identity.params.execution_resources.ordinals =
        resource_ordinals;
  }

  iree_status_t status =
      iree_hal_amdgpu_host_queue_initialize(&owned_params, queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_system_event_publish_queue_targets(
        system_event_target, queue, /*live_queue_count=*/1);
    if (iree_status_is_ok(status)) {
      queue->storage.allocator = params->host_allocator;
      queue->storage.system_event_target = system_event_target;
      queue->storage.release_slot = release_slot;
      if (retain_parent_device) {
        queue->storage.parent_device = params->coordination.logical_device;
        iree_hal_device_retain(queue->storage.parent_device);
      }
      *out_queue = queue;
    } else {
      iree_hal_amdgpu_host_queue_deinitialize(queue);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free_aligned(params->host_allocator, queue);
  }
  return status;
}

void iree_hal_amdgpu_host_queue_begin_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_hal_amdgpu_host_queue_close_submission(queue);
}

void iree_hal_amdgpu_host_queue_seal(iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_hal_amdgpu_host_queue_begin_deinitialize(queue);
  iree_hal_amdgpu_host_queue_wait_idle_before_deinitialize(queue);
}

bool iree_hal_amdgpu_host_queue_isa(iree_hal_queue_t* base_queue) {
  return iree_hal_resource_is(base_queue, &iree_hal_amdgpu_host_queue_vtable);
}

void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_host_queue_seal(queue);
  iree_hal_queue_release(&queue->base);
}

void iree_hal_amdgpu_host_queue_finish_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(!queue->completion.thread,
              "queue completion service must stop before final teardown");
  IREE_ASSERT(queue->pending_head == NULL,
              "queue pending operations must drain before final teardown");
  IREE_ASSERT(queue->active_staging_transfer_head == NULL,
              "queue staged/proactor publishers must drain before teardown");
  IREE_ASSERT(queue->shutdown_staging_transfer_head == NULL,
              "queue staged/proactor publishers must join before teardown");
  IREE_ASSERT(queue->active_file_action_head == NULL,
              "queue direct-file publishers must drain before teardown");
  IREE_ASSERT(queue->shutdown_file_action_head == NULL,
              "queue direct-file publishers must join before teardown");
  IREE_ASSERT(iree_hal_amdgpu_host_queue_post_drain_is_empty(queue),
              "queue callbacks must drain before final teardown");
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  IREE_ASSERT(iree_hal_amdgpu_host_queue_has_idle_certificate_locked(queue),
              "queue finalization requires an exact inert idle certificate");
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  // Everything below only destroys inert host-side storage. All callbacks,
  // retained operation resources, native queue/error producers, and failure
  // delivery were consumed by the sole sealer before certificate publication.
  IREE_ASSERT(!queue->epoch_registration_table && !queue->frontier_tracker,
              "queue publication must retire before inert finalization");

  iree_hal_amdgpu_notification_ring_deinitialize(&queue->notification_ring);

  if (queue->queue_upload_ring.base) {
    iree_hal_amdgpu_queue_upload_ring_deinitialize(queue->libhsa,
                                                   &queue->queue_upload_ring);
  }

  iree_hal_amdgpu_host_queue_deinitialize_tsan_state(queue);

  iree_hal_amdgpu_kernarg_ring_deinitialize(queue->libhsa,
                                            &queue->kernarg_ring);

  if (queue->pm4_ib_slots) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(queue->libhsa, queue->pm4_ib_slots));
    queue->pm4_ib_slots = NULL;
  }

  iree_hal_amdgpu_host_queue_deallocate_profiling_completion_signals(queue);
  iree_hal_amdgpu_host_queue_deallocate_profile_events(queue);

  if (queue->command_buffer_scratch) {
    iree_allocator_free(queue->host_allocator, queue->command_buffer_scratch);
    queue->command_buffer_scratch = NULL;
  }

  if (queue->completion.stop_signal.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
        queue->libhsa, queue->completion.stop_signal));
    queue->completion.stop_signal.handle = 0;
  }
  if (queue->completion.work_signal.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
        queue->libhsa, queue->completion.work_signal));
    queue->completion.work_signal.handle = 0;
  }

  iree_notification_deinitialize(&queue->seal_notification);
  iree_notification_deinitialize(&queue->post_drain_notification);
  iree_notification_deinitialize(&queue->completion.runner_notification);
  iree_slim_mutex_deinitialize(&queue->locks.post_drain_mutex);
  iree_slim_mutex_deinitialize(&queue->locks.completion_drain_mutex);
  iree_slim_mutex_deinitialize(&queue->profiling.event_mutex);
  iree_slim_mutex_deinitialize(&queue->locks.submission_mutex);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(
    iree_hal_amdgpu_host_queue_t* queue, bool enabled) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, enabled ? 1 : 0);

  if (enabled) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_host_queue_ensure_profile_event_storage(queue));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_host_queue_ensure_profiling_completion_signals(queue));
    iree_hal_amdgpu_host_queue_clear_profile_events(queue);
  }

  iree_status_t status = iree_hsa_amd_profiling_set_profiler_enabled(
      IREE_LIBHSA(queue->libhsa), queue->hardware_queue, enabled ? 1 : 0);
  if (iree_status_is_ok(status)) {
    queue->profiling.hsa_queue_timestamps_enabled = enabled ? 1 : 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_trim(iree_hal_amdgpu_host_queue_t* queue) {
  (void)queue;
}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_host_queue_op_submission_t {
  // Queue whose submission_mutex is held between begin/end.
  iree_hal_amdgpu_host_queue_t* queue;

  // Wait resolution computed while holding submission_mutex.
  iree_hal_amdgpu_wait_resolution_t resolution;

  // Deferred operation captured while holding submission_mutex, if any.
  iree_hal_amdgpu_pending_op_t* deferred_op;

  // Number of input waits. Capacity retries only need post-drain resubmission
  // when no semantic waits are available to naturally re-enter the queue.
  iree_host_size_t wait_semaphore_count;

  // Whether the direct submit helper found enough queue capacity.
  bool ready;

  // Whether |deferred_op| should retry after notification-ring drain.
  bool wait_for_capacity;
} iree_hal_amdgpu_host_queue_op_submission_t;

// Begins one direct/deferred queue operation attempt. The caller must pair this
// with iree_hal_amdgpu_host_queue_op_submission_end exactly once.
static inline iree_status_t iree_hal_amdgpu_host_queue_op_submission_begin(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_host_queue_op_submission_t* out_submission) {
  out_submission->queue = queue;
  out_submission->deferred_op = NULL;
  out_submission->wait_semaphore_count = wait_semaphore_list.count;
  out_submission->ready = true;
  out_submission->wait_for_capacity = false;

  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t terminal_status =
      iree_hal_amdgpu_host_queue_clone_error_status(queue);
  if (IREE_UNLIKELY(!iree_status_is_ok(terminal_status) ||
                    queue->is_shutting_down)) {
    // Terminal closure is the admission linearization point for every direct
    // and deferred operation. Keep the lock held for submission_end, but do
    // not resolve waits, capture resources, link a pending op, or install an
    // epilogue token after the completion service has closed admission.
    memset(&out_submission->resolution, 0, sizeof(out_submission->resolution));
    return iree_status_is_ok(terminal_status)
               ? iree_status_from_code(IREE_STATUS_CANCELLED)
               : terminal_status;
  }
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &out_submission->resolution);
  return iree_ok_status();
}

// Marks a captured pending op as retrying after notification-ring drain because
// direct submission ran out of queue capacity.
static inline void iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(
    iree_hal_amdgpu_host_queue_op_submission_t* submission) {
  submission->wait_for_capacity = submission->wait_semaphore_count == 0;
}

// Ends one direct/deferred queue operation attempt. Capacity-only retries are
// made visible while submission_mutex remains held; wait-backed operations
// start after releasing the lock because registration may invoke callbacks.
static inline iree_status_t iree_hal_amdgpu_host_queue_op_submission_end(
    iree_hal_amdgpu_host_queue_op_submission_t* submission,
    iree_status_t status) {
  // Capacity retry publication is callback-free and only takes
  // post_drain_mutex. Publish its COMPLETING state and queued continuation
  // before releasing submission admission so terminal failure cannot run its
  // first post-drain/cancellation pass in between the two operations. A
  // terminal owner subsequently drives queued snapshots to a fixed point.
  if (iree_status_is_ok(status) && submission->deferred_op &&
      submission->wait_for_capacity) {
    status = iree_hal_amdgpu_pending_op_start(submission->deferred_op,
                                              /*wait_for_capacity=*/true);
    submission->deferred_op = NULL;
  }
  iree_slim_mutex_unlock(&submission->queue->locks.submission_mutex);

  if (iree_status_is_ok(status) && submission->deferred_op) {
    status = iree_hal_amdgpu_pending_op_start(submission->deferred_op,
                                              /*wait_for_capacity=*/false);
  }
  return status;
}

typedef enum iree_hal_amdgpu_host_queue_buffer_state_e {
  IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY = 0,
  IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_UNSTAGED = 1,
  IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED = 2,
} iree_hal_amdgpu_host_queue_buffer_state_t;

// Classifies a queue_alloca wrapper for queue submission. The caller must hold
// submission_mutex so staging and decommit cannot race the query. Unstaged live
// wrappers must wait on their alloca signal through the host pending path;
// deallocated wrappers are terminal and cannot be submitted again.
static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_buffer_state(iree_hal_buffer_t* buffer) {
  if (!buffer) return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_amdgpu_transient_buffer_isa(allocated_buffer)) {
    return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  }
  if (iree_hal_amdgpu_transient_buffer_backing_buffer(allocated_buffer)) {
    return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  }
  if (iree_hal_amdgpu_transient_buffer_is_deallocated(allocated_buffer)) {
    return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED;
  }
  return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_UNSTAGED;
}

static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_merge_buffer_states(
    iree_hal_amdgpu_host_queue_buffer_state_t lhs,
    iree_hal_amdgpu_host_queue_buffer_state_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_binding_table_state(
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_amdgpu_host_queue_buffer_state_t state =
      IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  if (!binding_table.bindings) return state;
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    state = iree_hal_amdgpu_host_queue_merge_buffer_states(
        state, iree_hal_amdgpu_host_queue_buffer_state(
                   binding_table.bindings[i].buffer));
    if (state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED) break;
  }
  return state;
}

static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_binding_refs_state(
    iree_hal_buffer_ref_list_t bindings) {
  iree_hal_amdgpu_host_queue_buffer_state_t state =
      IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  if (!bindings.values) return state;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    state = iree_hal_amdgpu_host_queue_merge_buffer_states(
        state,
        iree_hal_amdgpu_host_queue_buffer_state(bindings.values[i].buffer));
    if (state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED) break;
  }
  return state;
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_buffer_state(
    iree_hal_amdgpu_host_queue_buffer_state_t state,
    iree_hal_semaphore_list_t wait_semaphore_list, bool* out_needs_staging) {
  *out_needs_staging =
      state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_UNSTAGED;
  if (state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue operation references a deallocated queue_alloca buffer");
  }
  if (*out_needs_staging && wait_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue operation references an unstaged queue_alloca buffer without "
        "waiting on its allocation signal");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_signal_empty_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_hal_amdgpu_host_queue_clone_error_status(queue);
  if (iree_status_is_ok(status) && IREE_UNLIKELY(queue->is_shutting_down)) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  if (iree_status_is_ok(status)) {
    // Signal outside submission_mutex: semaphore signaling dispatches satisfied
    // timepoints, and those callbacks may submit additional queue work.
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_execute_flags(flags));

  if (!command_buffer && wait_semaphore_list.count == 0) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    }
    return iree_hal_amdgpu_host_queue_signal_empty_barrier(
        queue, signal_semaphore_list);
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  // Command-buffer helpers may construct partial binding/replay ownership
  // while submission is serialized. Keep those cleanup handles through the
  // transaction and release them only after op_submission_end unlocks.
  iree_hal_resource_set_t* execute_cleanup_resource_set = NULL;
  iree_hal_resource_t* execute_cleanup_resource = NULL;
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        iree_hal_amdgpu_host_queue_binding_table_state(binding_table),
        wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_execute(
        queue, &wait_semaphore_list, &signal_semaphore_list, command_buffer,
        binding_table, flags, &submission.deferred_op);
  } else if (iree_status_is_ok(status) && !command_buffer) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    } else {
      uint64_t submission_id = 0;
      iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
          .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
          .operation_count = 0,
      };
      status = iree_hal_amdgpu_host_queue_try_submit_barrier(
          queue, &submission.resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){0},
          /*operation_resources=*/NULL,
          /*operation_resource_count=*/0, &profile_event_info,
          iree_hal_amdgpu_host_queue_post_commit_callback_null(),
          /*resource_set=*/NULL,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
          &submission.ready, &submission_id);
      if (iree_status_is_ok(status) && submission.ready) {
        profile_event_info.submission_id = submission_id;
        iree_hal_amdgpu_host_queue_record_profile_queue_event(
            queue, &submission.resolution, signal_semaphore_list,
            &profile_event_info);
      }
      if (iree_status_is_ok(status) && !submission.ready) {
        status = iree_hal_amdgpu_host_queue_defer_execute(
            queue, &wait_semaphore_list, &signal_semaphore_list,
            /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
            flags, &submission.deferred_op);
        iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(
            &submission);
      }
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_command_buffer(
        queue, &submission.resolution, signal_semaphore_list, command_buffer,
        binding_table, flags, &execute_cleanup_resource_set,
        &execute_cleanup_resource, &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list, command_buffer,
          binding_table, flags, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  status = iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
  iree_hal_resource_release(execute_cleanup_resource);
  iree_hal_resource_set_free(execute_cleanup_resource_set);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_atomic(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        iree_hal_amdgpu_host_queue_buffer_state(operation->target_buffer),
        wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_atomic(
        queue, &wait_semaphore_list, &signal_semaphore_list, operation,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_atomic(
        queue, &submission.resolution, signal_semaphore_list, operation,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_atomic(
          queue, &wait_semaphore_list, &signal_semaphore_list, operation,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_atomic_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  const iree_hal_amdgpu_host_queue_atomic_operation_t operation = {
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .kind = IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT,
      .params.wait = params,
  };
  return iree_hal_amdgpu_host_queue_enqueue_atomic(
      queue, wait_semaphore_list, signal_semaphore_list, &operation);
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_atomic_store(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  const iree_hal_amdgpu_host_queue_atomic_operation_t operation = {
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .kind = IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE,
      .params.store = params,
  };
  return iree_hal_amdgpu_host_queue_enqueue_atomic(
      queue, wait_semaphore_list, signal_semaphore_list, &operation);
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_atomic_rmw(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  const iree_hal_amdgpu_host_queue_atomic_operation_t operation = {
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .kind = IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW,
      .params.rmw = params,
  };
  return iree_hal_amdgpu_host_queue_enqueue_atomic(
      queue, wait_semaphore_list, signal_semaphore_list, &operation);
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(queue->block_pool, &scratch_arena);

  iree_hal_amdgpu_alloca_transaction_t transaction = {
      .request_count = request_count,
  };
  iree_hal_pool_reservation_request_t* canonical_requests = NULL;
  iree_hal_buffer_t** buffers = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, request_count, sizeof(*canonical_requests),
      (void**)&canonical_requests);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, request_count,
                                       sizeof(*buffers), (void**)&buffers);
    if (iree_status_is_ok(status)) {
      memset(buffers, 0, request_count * sizeof(*buffers));
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, request_count,
                                       sizeof(*transaction.reservations),
                                       (void**)&transaction.reservations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, request_count,
                                       sizeof(*transaction.acquire_infos),
                                       (void**)&transaction.acquire_infos);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, request_count,
                                       sizeof(*transaction.backing_buffers),
                                       (void**)&transaction.backing_buffers);
  }
  iree_host_size_t frontier_size = 0;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_size(UINT8_MAX, &frontier_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate(&scratch_arena, frontier_size,
                                 (void**)&transaction.wait_frontier);
  }
  if (iree_status_is_ok(status)) {
    memset(transaction.reservations, 0,
           request_count * sizeof(*transaction.reservations));
    memset(transaction.acquire_infos, 0,
           request_count * sizeof(*transaction.acquire_infos));
    memset(transaction.backing_buffers, 0,
           request_count * sizeof(*transaction.backing_buffers));
    iree_async_frontier_initialize(transaction.wait_frontier, 0);
    status = iree_hal_amdgpu_host_queue_prepare_alloca_buffers(
        queue, pool, request_count, requests, canonical_requests, buffers);
  }
  transaction.requests = canonical_requests;
  transaction.buffers = buffers;

  // Always ask the pool to surface waitable death-frontier candidates so the
  // queue can distinguish true pool pressure from an internal dependency.
  // Disallow growth while submission_mutex is held; growable pools report that
  // as a cold retry instead of calling into their slab provider on the
  // serialized queue path.
  const iree_hal_pool_reserve_flags_t reserve_flags =
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER |
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH;

  iree_hal_amdgpu_host_queue_op_submission_t submission = {0};
  iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
  bool has_alloca_epilogue = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_op_submission_begin(
        queue, wait_semaphore_list, &submission);
  }
  if (iree_status_is_ok(status) && submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_alloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, pool,
        request_count, canonical_requests, buffers, reserve_flags,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    // The direct transaction deliberately unlocks submission_mutex around
    // pool vtable calls and profiling. Register that entire lock-split window
    // before the first unlock so a concurrent sealer cannot certify and tear
    // down queue-owned state while the transaction is reentering the pool.
    iree_hal_amdgpu_host_queue_enter_submission_epilogue(queue);
    has_alloca_epilogue = true;
    status = iree_hal_amdgpu_host_queue_submit_alloca(
        queue, &submission.resolution, signal_semaphore_list, pool,
        &transaction, reserve_flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        /*pending_op=*/NULL, &memory_wait_op, &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready && !memory_wait_op) {
      status = iree_hal_amdgpu_host_queue_defer_alloca(
          queue, &wait_semaphore_list, &signal_semaphore_list, pool,
          request_count, canonical_requests, buffers, reserve_flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  if (submission.queue) {
    status = iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
  }
  if (iree_status_is_ok(status) && memory_wait_op) {
    iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
  }

  if (iree_status_is_ok(status)) {
    memcpy(out_buffers, buffers, request_count * sizeof(*out_buffers));
  } else if (buffers) {
    iree_hal_amdgpu_host_queue_release_alloca_transaction(pool, &transaction);
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  if (has_alloca_epilogue) {
    // Final queue access after relocked publication/rollback, any pending-op
    // handoff, output/failure cleanup, and scratch block-pool release.
    iree_hal_amdgpu_host_queue_leave_submission_epilogue(queue);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers) {
  for (iree_host_size_t i = 0; i < buffer_count; ++i) {
    if (IREE_UNLIKELY(!iree_hal_amdgpu_transient_buffer_isa(buffers[i]))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "deallocation buffer %" PRIhsz
                              " is not an AMDGPU queue allocation root",
                              i);
    }
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(buffers[i]);
    if (IREE_UNLIKELY(placement.device != queue->logical_device)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "deallocation buffer %" PRIhsz " belongs to another device", i);
    }
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(queue->block_pool, &scratch_arena);
  iree_hal_pool_reservation_t* reservations = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, buffer_count,
                                sizeof(*reservations), (void**)&reservations);
  if (iree_status_is_ok(status)) {
    memset(reservations, 0, buffer_count * sizeof(*reservations));
  }

  iree_hal_pool_t* source_pool = NULL;
  iree_host_size_t marked_count = 0;
  while (marked_count < buffer_count && iree_status_is_ok(status)) {
    iree_hal_pool_t* buffer_pool = NULL;
    status = iree_hal_amdgpu_transient_buffer_begin_dealloca(
        buffers[marked_count], &buffer_pool);
    if (iree_status_is_ok(status)) {
      if (marked_count == 0) {
        source_pool = buffer_pool;
      } else if (IREE_UNLIKELY(source_pool != buffer_pool)) {
        iree_hal_amdgpu_transient_buffer_abort_dealloca(buffers[marked_count]);
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "deallocation transaction spans multiple source pools");
        break;
      }
      ++marked_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < marked_count; ++i) {
      iree_hal_amdgpu_transient_buffer_abort_dealloca(buffers[i]);
    }
    iree_arena_deinitialize(&scratch_arena);
    return status;
  }

  iree_hal_amdgpu_dealloca_transaction_t transaction = {
      .buffer_count = buffer_count,
      .buffers = buffers,
      .pool = source_pool,
      .reservations = reservations,
  };
  iree_hal_amdgpu_host_queue_op_submission_t submission;
  status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  if (iree_status_is_ok(status) && submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_dealloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, buffer_count,
        buffers, source_pool, &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_dealloca(
        queue, &submission.resolution, signal_semaphore_list, &transaction,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_dealloca(
          queue, &wait_semaphore_list, &signal_semaphore_list, buffer_count,
          buffers, source_pool, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  const bool has_release_epilogue = transaction.reservations_detached;
  if (has_release_epilogue) {
    // Install the seal-joined token before dropping submission_mutex. This
    // closes the accepted-submission -> unlocked-pool-callback window.
    iree_hal_amdgpu_host_queue_enter_submission_epilogue(queue);
  }
  status = iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
  iree_hal_amdgpu_host_queue_release_dealloca_transaction(queue, &transaction);
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < buffer_count; ++i) {
      iree_hal_amdgpu_transient_buffer_abort_dealloca(buffers[i]);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  if (has_release_epilogue) {
    // Final queue access: wakes any sealer after pool callbacks, profiling,
    // buffer rollback, and arena/block-pool cleanup have all returned.
    iree_hal_amdgpu_host_queue_leave_submission_epilogue(queue);
  }
  return status;
}

// Queue fill entry point. Resolves waits under submission_mutex and captures a
// pending operation when waits, an unstaged target or submission capacity
// require deferral.
iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        iree_hal_amdgpu_host_queue_buffer_state(target_buffer),
        wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_fill(
        queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
        target_offset, length, pattern_bits, pattern_length, flags,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_fill(
        queue, &submission.resolution, signal_semaphore_list, target_buffer,
        target_offset, length, pattern_bits, pattern_length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_fill(
          queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
          target_offset, length, pattern_bits, pattern_length, flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

// Queue timestamp entry point. Resolves waits under submission_mutex and
// captures a pending operation when waits, target staging, or submission
// capacity require deferral.
static iree_status_t iree_hal_amdgpu_host_queue_enqueue_timestamp(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  // Reject unsupported flags synchronously at enqueue, before the
  // immediate/deferred split, so the error is deterministic regardless of
  // wait-list state (mirrors the up-front flag validation in queue_execute).
  if (IREE_UNLIKELY(flags != IREE_HAL_TIMESTAMP_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported timestamp flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        iree_hal_amdgpu_host_queue_buffer_state(target_buffer),
        wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_timestamp(
        queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
        target_offset, flags, &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_timestamp(
        queue, &submission.resolution, signal_semaphore_list, target_buffer,
        target_offset, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_timestamp(
          queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
          target_offset, flags, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

iree_status_t iree_hal_amdgpu_host_queue_copy_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type) {
  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  iree_hal_amdgpu_host_queue_buffer_state_t buffer_state =
      iree_hal_amdgpu_host_queue_buffer_state(source_buffer);
  buffer_state = iree_hal_amdgpu_host_queue_merge_buffer_states(
      buffer_state, iree_hal_amdgpu_host_queue_buffer_state(target_buffer));
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        buffer_state, wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_copy(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        profile_event_type, &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_copy(
        queue, &submission.resolution, signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        profile_event_type,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_copy(
          queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
          source_offset, target_buffer, target_offset, length, flags,
          profile_event_type, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

// Queue copy entry point. The shared copy path is also used by file read/write
// staging so all copy-shaped operations use the same wait/backpressure path.
iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_hal_amdgpu_host_queue_copy_buffer(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY);
}

// Queue update entry point. Immediate updates copy into queue-owned kernarg
// memory; deferred updates copy into the pending-op arena.
iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        iree_hal_amdgpu_host_queue_buffer_state(target_buffer),
        wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_update(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_update(
        queue, &submission.resolution, signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_update(
          queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
          source_offset, target_buffer, target_offset, length, flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static bool iree_hal_amdgpu_host_queue_is_noop_dispatch(
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  return !iree_hal_dispatch_uses_indirect_parameters(flags) &&
         (config.workgroup_count[0] | config.workgroup_count[1] |
          config.workgroup_count[2]) == 0;
}

// Queue dispatch entry point. Empty direct dispatches route through the barrier
// path so they still signal semaphores and profile as dispatch submissions.
static iree_status_t iree_hal_amdgpu_host_queue_enqueue_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  const bool is_noop_dispatch =
      iree_hal_amdgpu_host_queue_is_noop_dispatch(config, flags);

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  iree_hal_amdgpu_host_queue_buffer_state_t buffer_state =
      iree_hal_amdgpu_host_queue_binding_refs_state(bindings);
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    buffer_state = iree_hal_amdgpu_host_queue_merge_buffer_states(
        buffer_state, iree_hal_amdgpu_host_queue_buffer_state(
                          config.workgroup_count_ref.buffer));
  }
  bool needs_staging = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_validate_buffer_state(
        buffer_state, wait_semaphore_list, &needs_staging);
  }
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    if (is_noop_dispatch) {
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list,
          /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE, &submission.deferred_op);
    } else {
      status = iree_hal_amdgpu_host_queue_defer_dispatch(
          queue, &wait_semaphore_list, &signal_semaphore_list, executable,
          export_ordinal, config, constants, bindings, flags,
          &submission.deferred_op);
    }
  } else if (iree_status_is_ok(status) && is_noop_dispatch) {
    uint64_t submission_id = 0;
    iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
        .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH,
        .operation_count = 0,
    };
    status = iree_hal_amdgpu_host_queue_try_submit_barrier(
        queue, &submission.resolution, signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){0},
        /*operation_resources=*/NULL,
        /*operation_resource_count=*/0, &profile_event_info,
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready, &submission_id);
    if (iree_status_is_ok(status) && submission.ready) {
      profile_event_info.submission_id = submission_id;
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, &submission.resolution, signal_semaphore_list,
          &profile_event_info);
    }
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list,
          /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_dispatch(
        queue, &submission.resolution, signal_semaphore_list, executable,
        export_ordinal, config, constants, bindings, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_dispatch(
          queue, &wait_semaphore_list, &signal_semaphore_list, executable,
          export_ordinal, config, constants, bindings, flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

iree_status_t iree_hal_amdgpu_host_queue_submit_read(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_hal_amdgpu_host_queue_read_file(
      queue, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags);
}

iree_status_t iree_hal_amdgpu_host_queue_submit_write(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_hal_amdgpu_host_queue_write_file(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags);
}

iree_status_t iree_hal_amdgpu_host_queue_enqueue_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count) {
  if (IREE_UNLIKELY(!action.fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host action callback must be non-null");
  }
  if (IREE_UNLIKELY(operation_resource_count > 0 && !operation_resources)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host action resources must be non-null");
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  // Host actions execute on CPU threads and must observe device-produced
  // host-visible memory even when a semaphore edge itself is device-local.
  submission.resolution.inline_acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          submission.resolution.inline_acquire_scope,
          IREE_HSA_FENCE_SCOPE_SYSTEM);
  submission.resolution.barrier_acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          submission.resolution.barrier_acquire_scope,
          IREE_HSA_FENCE_SCOPE_SYSTEM);
  if (iree_status_is_ok(status) && submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_host_action(
        queue, &wait_semaphore_list, action, operation_resources,
        operation_resource_count, &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_submit_barrier(
        queue, &submission.resolution, iree_hal_semaphore_list_empty(), action,
        operation_resources, operation_resource_count,
        /*profile_event_info=*/NULL,
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready, /*out_submission_id=*/NULL);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_host_action(
          queue, &wait_semaphore_list, action, operation_resources,
          operation_resource_count, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_host_call(call, args, flags));

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_op_submission_begin(
      queue, wait_semaphore_list, &submission);
  if (iree_status_is_ok(status) && submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_host_call(
        queue, &wait_semaphore_list, &signal_semaphore_list, call, args, flags,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_host_call(
        queue, &submission.resolution, signal_semaphore_list, call, args, flags,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_host_call(
          queue, &wait_semaphore_list, &signal_semaphore_list, call, args,
          flags, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

//===----------------------------------------------------------------------===//
// HAL queue resource
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_destroy(iree_hal_queue_t* base_queue) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  const iree_hal_amdgpu_host_queue_storage_t storage = queue->storage;
  if (!iree_allocator_is_null(storage.allocator)) {
    iree_hal_amdgpu_host_queue_seal(queue);
    iree_hal_amdgpu_system_event_retire_queue_target(
        storage.system_event_target, queue);
  }
  iree_hal_amdgpu_host_queue_finish_deinitialize(queue);
  if (storage.release_slot.fn) {
    storage.release_slot.fn(storage.release_slot.user_data,
                            storage.release_slot.queue_index);
  }
  if (!iree_allocator_is_null(storage.allocator)) {
    iree_allocator_free_aligned(storage.allocator, queue);
  }
  // Must be last: dropping the dedicated queue's parent edge may destroy the
  // logical/physical device state borrowed by every earlier teardown step.
  iree_hal_device_release(storage.parent_device);
}

static iree_status_t iree_hal_amdgpu_host_queue_check_device_failure(
    iree_hal_amdgpu_host_queue_t* queue) {
  return iree_hal_amdgpu_logical_device_check_failure(
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device);
}

static iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_host_call(
      queue, wait_semaphore_list, signal_semaphore_list, call, args, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_query_dispatch_concurrency(
    iree_hal_queue_t* base_queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  (void)flags;

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue_ordinal(
          executable, function, queue->physical_queue_ordinal, &descriptor));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_dispatch_limits_validate_workgroup_size(
          &descriptor->limits, params.workgroup_size));

  const iree_hal_amdgpu_dispatch_concurrency_inputs_t inputs = {
      .capabilities = queue->dispatch_concurrency_capabilities,
      .execution_resource_topology = queue->execution_resource_topology,
      .execution_resources = iree_hal_queue_execution_resources(base_queue),
      .queue_features = iree_hal_queue_features(base_queue),
      .kernel_descriptor = descriptor->kernel_descriptor,
      .workgroup_cluster_size =
          {
              descriptor->kernel_args.workgroup_cluster_size[0],
              descriptor->kernel_args.workgroup_cluster_size[1],
              descriptor->kernel_args.workgroup_cluster_size[2],
          },
      .maximum_dynamic_workgroup_local_memory_size =
          descriptor->limits.maximum_dynamic_workgroup_local_memory_size,
  };
  iree_hal_queue_dispatch_concurrency_t concurrency;
  iree_status_t status = iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, params, &concurrency);
  if (iree_status_is_ok(status)) {
    *out_concurrency = concurrency;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_dispatch(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_dispatch(
      queue, wait_semaphore_list, signal_semaphore_list, executable, function,
      config, constants, bindings, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_wait(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_atomic_wait(
      queue, wait_semaphore_list, signal_semaphore_list, target_buffer,
      target_offset, params);
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_store(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_atomic_store(
      queue, wait_semaphore_list, signal_semaphore_list, target_buffer,
      target_offset, params);
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_rmw(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_atomic_rmw(
      queue, wait_semaphore_list, signal_semaphore_list, target_buffer,
      target_offset, params);
}

static iree_status_t iree_hal_amdgpu_host_queue_timestamp(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_timestamp(
      queue, wait_semaphore_list, signal_semaphore_list, target_buffer,
      target_offset, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_barrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  (void)flags;
  return iree_hal_amdgpu_host_queue_enqueue_execute(
      queue, wait_semaphore_list, signal_semaphore_list,
      /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_execute(
      queue, wait_semaphore_list, signal_semaphore_list, command_buffer,
      binding_table, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_flush(
    iree_hal_queue_t* base_queue) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_alloca(
      queue, wait_semaphore_list, signal_semaphore_list, pool, request_count,
      requests, out_buffers);
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_dealloca(
      queue, wait_semaphore_list, signal_semaphore_list, buffer_count, buffers);
}

static iree_status_t iree_hal_amdgpu_host_queue_transfer(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_enqueue_transfer(
      queue, wait_semaphore_list, signal_semaphore_list, operation_count,
      operations);
}

static iree_status_t iree_hal_amdgpu_host_queue_read(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_submit_read(
      queue, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_HAL_ASSERT_TYPE(base_queue, &iree_hal_amdgpu_host_queue_vtable);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_device_failure(queue));
  return iree_hal_amdgpu_host_queue_submit_write(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags);
}

static const iree_hal_queue_vtable_t iree_hal_amdgpu_host_queue_vtable = {
    .destroy = iree_hal_amdgpu_host_queue_destroy,
    .barrier = iree_hal_amdgpu_host_queue_barrier,
    .execute = iree_hal_amdgpu_host_queue_execute,
    .host_call = iree_hal_amdgpu_host_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amdgpu_host_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amdgpu_host_queue_dispatch,
    .atomic_wait = iree_hal_amdgpu_host_queue_atomic_wait,
    .atomic_store = iree_hal_amdgpu_host_queue_atomic_store,
    .atomic_rmw = iree_hal_amdgpu_host_queue_atomic_rmw,
    .timestamp = iree_hal_amdgpu_host_queue_timestamp,
    .flush = iree_hal_amdgpu_host_queue_flush,
    .alloca = iree_hal_amdgpu_host_queue_alloca,
    .dealloca = iree_hal_amdgpu_host_queue_dealloca,
    .transfer = iree_hal_amdgpu_host_queue_transfer,
    .read = iree_hal_amdgpu_host_queue_read,
    .write = iree_hal_amdgpu_host_queue_write,
};
