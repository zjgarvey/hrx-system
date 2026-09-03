// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/replay/execute_object.h"
#include "iree/hal/replay/execute_operation.h"
#include "iree/hal/replay/execute_state.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/format.h"

typedef enum iree_hal_replay_plan_record_kind_e {
  IREE_HAL_REPLAY_PLAN_RECORD_KIND_GENERIC = 0u,
  IREE_HAL_REPLAY_PLAN_RECORD_KIND_SKIP = 1u,
  IREE_HAL_REPLAY_PLAN_RECORD_KIND_SCOPE = 2u,
  IREE_HAL_REPLAY_PLAN_RECORD_KIND_COMMAND_BUFFER_EXECUTION_BARRIER = 3u,
  IREE_HAL_REPLAY_PLAN_RECORD_KIND_COMMAND_BUFFER_DISPATCH = 4u,
  IREE_HAL_REPLAY_PLAN_RECORD_KIND_QUEUE_EXECUTE = 5u,
} iree_hal_replay_plan_record_kind_t;

typedef struct iree_hal_replay_plan_scope_t {
  // Type of scope event to publish during execution.
  iree_hal_replay_scope_event_type_t event_type;
  // Scope name borrowed from the replay file.
  iree_string_view_t name;
} iree_hal_replay_plan_scope_t;

typedef struct iree_hal_replay_plan_command_buffer_execution_barrier_t {
  // Captured command buffer object id.
  iree_hal_replay_object_id_t command_buffer_id;
  // Source execution stages for the barrier.
  iree_hal_execution_stage_t source_stage_mask;
  // Target execution stages for the barrier.
  iree_hal_execution_stage_t target_stage_mask;
  // Barrier flags captured from the original call.
  iree_hal_execution_barrier_flags_t flags;
  // Number of serialized memory barriers.
  iree_host_size_t memory_barrier_count;
  // Serialized memory barriers borrowed from the replay file.
  const iree_hal_replay_memory_barrier_payload_t* memory_barrier_payloads;
  // Number of serialized buffer barriers.
  iree_host_size_t buffer_barrier_count;
  // Serialized buffer barriers borrowed from the replay file.
  const iree_hal_replay_buffer_barrier_payload_t* buffer_barrier_payloads;
} iree_hal_replay_plan_command_buffer_execution_barrier_t;

typedef struct iree_hal_replay_plan_command_buffer_dispatch_t {
  // Captured command buffer object id.
  iree_hal_replay_object_id_t command_buffer_id;
  // Captured executable object id.
  iree_hal_replay_object_id_t executable_id;
  // Captured function ordinal to resolve at replay execution time.
  uint32_t function_ordinal;
  // Dispatch grid and indirect-count configuration.
  iree_hal_dispatch_config_t config;
  // Serialized indirect workgroup count buffer reference.
  iree_hal_replay_buffer_ref_payload_t workgroup_count_ref;
  // Inline constant bytes borrowed from the replay file.
  iree_const_byte_span_t constants;
  // Number of serialized buffer refs in |binding_payloads|.
  iree_host_size_t binding_count;
  // Serialized dispatch bindings borrowed from the replay file.
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads;
  // Dispatch flags captured from the original call.
  iree_hal_dispatch_flags_t flags;
} iree_hal_replay_plan_command_buffer_dispatch_t;

typedef struct iree_hal_replay_plan_queue_execute_t {
  // Captured device object id receiving the queue operation.
  iree_hal_replay_object_id_t device_id;
  // Captured command buffer object id, or NONE for a queue barrier.
  iree_hal_replay_object_id_t command_buffer_id;
  // Queue affinity captured from the original call.
  iree_hal_queue_affinity_t queue_affinity;
  // Queue execution flags captured from the original call.
  iree_hal_execute_flags_t flags;
  // Number of serialized wait semaphore timepoints.
  iree_host_size_t wait_semaphore_count;
  // Serialized wait semaphore timepoints borrowed from the replay file.
  const iree_hal_replay_semaphore_timepoint_payload_t* wait_semaphore_payloads;
  // Number of serialized signal semaphore timepoints.
  iree_host_size_t signal_semaphore_count;
  // Serialized signal semaphore timepoints borrowed from the replay file.
  const iree_hal_replay_semaphore_timepoint_payload_t*
      signal_semaphore_payloads;
  // Number of serialized queue binding entries.
  iree_host_size_t binding_count;
  // Serialized queue binding entries borrowed from the replay file.
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads;
} iree_hal_replay_plan_queue_execute_t;

// Static completion information for one queue submission. This
// proves that replay completion waits and VMM synchronization points are
// reachable before any backend operation is issued.
typedef struct iree_hal_replay_plan_queue_dependency_t {
  // True when the record submits work to a device queue.
  bool is_submission;
  // True when completion can depend on a device-memory predicate.
  bool may_wait_on_device_memory;
  // Captured command buffer, or NONE for direct queue operations.
  iree_hal_replay_object_id_t command_buffer_id;
  // Number of captured wait semaphore timepoints.
  iree_host_size_t wait_semaphore_count;
  // Captured wait semaphore timepoints borrowed from the replay file.
  const iree_hal_replay_semaphore_timepoint_payload_t* wait_semaphore_payloads;
  // Number of captured signal semaphore timepoints.
  iree_host_size_t signal_semaphore_count;
  // Captured signal semaphore timepoints borrowed from the replay file.
  const iree_hal_replay_semaphore_timepoint_payload_t*
      signal_semaphore_payloads;
} iree_hal_replay_plan_queue_dependency_t;

typedef struct iree_hal_replay_plan_record_t {
  // Parsed replay record borrowed from the original replay file.
  iree_hal_replay_file_record_t file_record;
  // Prepared interpreter opcode for this record.
  iree_hal_replay_plan_record_kind_t kind;
  // Prepared payload for hot replay operations.
  union {
    // Prepared scope marker payload.
    iree_hal_replay_plan_scope_t scope;
    // Prepared command buffer execution barrier payload.
    iree_hal_replay_plan_command_buffer_execution_barrier_t
        command_buffer_execution_barrier;
    // Prepared command buffer dispatch payload.
    iree_hal_replay_plan_command_buffer_dispatch_t command_buffer_dispatch;
    // Prepared queue execute payload.
    iree_hal_replay_plan_queue_execute_t queue_execute;
  } payload;
} iree_hal_replay_plan_record_t;

struct iree_hal_replay_plan_t {
  // Original replay file bytes borrowed by this plan.
  iree_const_byte_span_t file_contents;
  // Host allocator used for the plan allocations.
  iree_allocator_t host_allocator;
  // Dense session-local object table capacity required during execution.
  iree_host_size_t object_capacity;
  // Number of entries in |records|.
  iree_host_size_t record_count;
  // Prepared replay records owned by this plan.
  iree_hal_replay_plan_record_t* records;
};

static iree_status_t iree_hal_replay_plan_prepare_scope(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_scope_event_type_t event_type,
    iree_hal_replay_plan_scope_t* out_scope) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      sizeof(iree_hal_replay_scope_payload_t)));
  iree_hal_replay_scope_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.flags != IREE_HAL_REPLAY_SCOPE_FLAG_NONE ||
                    payload.reserved0 != 0 || payload.reserved1 != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload reserved fields must be "
                            "zero");
  }
  if (IREE_UNLIKELY(payload.name_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) + (iree_host_size_t)payload.name_length !=
                        record->payload.data_length ||
                    payload.name_length == 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload name length mismatch");
  }
  out_scope->event_type = event_type;
  out_scope->name =
      iree_make_string_view((const char*)record->payload.data + sizeof(payload),
                            (iree_host_size_t)payload.name_length);
  return iree_ok_status();
}

static iree_status_t
iree_hal_replay_plan_prepare_command_buffer_execution_barrier(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_plan_command_buffer_execution_barrier_t* out_barrier) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER,
      sizeof(iree_hal_replay_command_buffer_execution_barrier_payload_t)));
  iree_hal_replay_command_buffer_execution_barrier_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t memory_payloads_size = 0;
  iree_host_size_t buffer_payloads_size = 0;
  iree_host_size_t total_payload_size = 0;
  if (IREE_UNLIKELY(
          payload.memory_barrier_count > IREE_HOST_SIZE_MAX ||
          payload.buffer_barrier_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.memory_barrier_count,
              sizeof(iree_hal_replay_memory_barrier_payload_t),
              &memory_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.buffer_barrier_count,
              sizeof(iree_hal_replay_buffer_barrier_payload_t),
              &buffer_payloads_size) ||
          !iree_host_size_checked_add(sizeof(payload), memory_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, buffer_payloads_size,
                                      &total_payload_size) ||
          total_payload_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay execution barrier payload length mismatch");
  }
  out_barrier->command_buffer_id = record->header.object_id;
  out_barrier->source_stage_mask = payload.source_stage_mask;
  out_barrier->target_stage_mask = payload.target_stage_mask;
  out_barrier->flags = payload.flags;
  out_barrier->memory_barrier_count =
      (iree_host_size_t)payload.memory_barrier_count;
  out_barrier->memory_barrier_payloads =
      (const iree_hal_replay_memory_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload));
  out_barrier->buffer_barrier_count =
      (iree_host_size_t)payload.buffer_barrier_count;
  out_barrier->buffer_barrier_payloads =
      (const iree_hal_replay_buffer_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        memory_payloads_size);
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_plan_prepare_command_buffer_dispatch(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_plan_command_buffer_dispatch_t* out_dispatch) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      sizeof(iree_hal_replay_dispatch_payload_t)));
  iree_hal_replay_dispatch_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (payload.wait_semaphore_count != 0 ||
      payload.signal_semaphore_count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "command buffer dispatch semaphore lists are "
                            "reserved for immediate dispatch");
  }
  iree_host_size_t wait_offset = 0;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_offset = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t binding_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_dispatch_layout(
      record, &payload, &wait_offset, &wait_size, &signal_offset, &signal_size,
      &constants_offset, &binding_offset, &binding_size));
  (void)wait_offset;
  (void)wait_size;
  (void)signal_offset;
  (void)signal_size;
  (void)binding_size;
  out_dispatch->command_buffer_id = record->header.object_id;
  out_dispatch->executable_id = payload.executable_id;
  out_dispatch->function_ordinal = payload.function_ordinal;
  memset(&out_dispatch->config, 0, sizeof(out_dispatch->config));
  memcpy(out_dispatch->config.workgroup_size, payload.workgroup_size,
         sizeof(out_dispatch->config.workgroup_size));
  memcpy(out_dispatch->config.workgroup_count, payload.workgroup_count,
         sizeof(out_dispatch->config.workgroup_count));
  out_dispatch->config.dynamic_workgroup_local_memory =
      payload.dynamic_workgroup_local_memory;
  out_dispatch->workgroup_count_ref = payload.workgroup_count_ref;
  out_dispatch->constants =
      iree_make_const_byte_span(record->payload.data + constants_offset,
                                (iree_host_size_t)payload.constants_length);
  out_dispatch->binding_count = (iree_host_size_t)payload.binding_count;
  out_dispatch->binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                    binding_offset);
  out_dispatch->flags = payload.flags;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_plan_prepare_queue_execute(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_plan_queue_execute_t* out_execute) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE,
      sizeof(iree_hal_replay_device_queue_execute_payload_t)));
  iree_hal_replay_device_queue_execute_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t binding_size = 0;
  iree_host_size_t expected_size = 0;
  if (IREE_UNLIKELY(payload.wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.binding_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.wait_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &wait_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.signal_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &signal_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.binding_count,
                        sizeof(iree_hal_replay_buffer_ref_payload_t),
                        &binding_size) ||
                    !iree_host_size_checked_add(sizeof(payload), wait_size,
                                                &expected_size) ||
                    !iree_host_size_checked_add(expected_size, signal_size,
                                                &expected_size) ||
                    !iree_host_size_checked_add(expected_size, binding_size,
                                                &expected_size) ||
                    expected_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue execute payload length mismatch");
  }
  const uint8_t* cursor = record->payload.data + sizeof(payload);
  out_execute->device_id = record->header.object_id;
  out_execute->command_buffer_id = payload.command_buffer_id;
  out_execute->queue_affinity = payload.queue_affinity;
  out_execute->flags = payload.flags;
  out_execute->wait_semaphore_count =
      (iree_host_size_t)payload.wait_semaphore_count;
  out_execute->wait_semaphore_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)cursor;
  cursor += wait_size;
  out_execute->signal_semaphore_count =
      (iree_host_size_t)payload.signal_semaphore_count;
  out_execute->signal_semaphore_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)cursor;
  cursor += signal_size;
  out_execute->binding_count = (iree_host_size_t)payload.binding_count;
  out_execute->binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)cursor;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_plan_prepare_queue_dependency_lists(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t header_length,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count,
    uint64_t trailing_payload_length,
    iree_hal_replay_plan_queue_dependency_t* out_dependency) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, payload_type, header_length));
  if (IREE_UNLIKELY(wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    trailing_payload_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue dependency length exceeds host size");
  }
  const iree_host_size_t wait_count = (iree_host_size_t)wait_semaphore_count;
  const iree_host_size_t signal_count =
      (iree_host_size_t)signal_semaphore_count;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t expected_length = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              wait_count, sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
              &wait_size) ||
          !iree_host_size_checked_mul(
              signal_count,
              sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
              &signal_size) ||
          !iree_host_size_checked_add(header_length, wait_size,
                                      &expected_length) ||
          !iree_host_size_checked_add(expected_length, signal_size,
                                      &expected_length) ||
          !iree_host_size_checked_add(expected_length,
                                      (iree_host_size_t)trailing_payload_length,
                                      &expected_length) ||
          expected_length != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue dependency payload length mismatch");
  }
  const uint8_t* cursor = record->payload.data + header_length;
  out_dependency->is_submission = true;
  out_dependency->wait_semaphore_count = wait_count;
  out_dependency->wait_semaphore_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)cursor;
  cursor += wait_size;
  out_dependency->signal_semaphore_count = signal_count;
  out_dependency->signal_semaphore_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)cursor;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_plan_prepare_queue_dependency(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_plan_queue_dependency_t* out_dependency) {
  memset(out_dependency, 0, sizeof(*out_dependency));
  if (record->header.record_type !=
          IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION ||
      record->header.status_code != IREE_STATUS_OK) {
    return iree_ok_status();
  }
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA: {
      iree_hal_replay_device_queue_alloca_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA: {
      iree_hal_replay_device_queue_dealloca_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL: {
      iree_hal_replay_device_queue_fill_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.pattern_length,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE: {
      iree_hal_replay_device_queue_update_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.data_length, out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY: {
      iree_hal_replay_device_queue_copy_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ: {
      iree_hal_replay_device_queue_read_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.captured_data_length,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE: {
      iree_hal_replay_device_queue_write_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT: {
      iree_hal_replay_device_queue_atomic_wait_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      out_dependency->may_wait_on_device_memory = true;
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE: {
      iree_hal_replay_device_queue_atomic_store_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_RMW: {
      iree_hal_replay_device_queue_atomic_rmw_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW,
          sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_hal_replay_plan_prepare_queue_dependency_lists(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW,
          sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          out_dependency);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH: {
      iree_hal_replay_dispatch_payload_t payload;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
          record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH, sizeof(payload)));
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t constants_offset = 0;
      iree_host_size_t binding_offset = 0;
      iree_host_size_t binding_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_dispatch_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &constants_offset, &binding_offset, &binding_size));
      (void)wait_size;
      (void)signal_size;
      (void)constants_offset;
      (void)binding_offset;
      (void)binding_size;
      out_dependency->is_submission = true;
      out_dependency->wait_semaphore_count =
          (iree_host_size_t)payload.wait_semaphore_count;
      out_dependency->wait_semaphore_payloads =
          (const iree_hal_replay_semaphore_timepoint_payload_t*)(record->payload
                                                                     .data +
                                                                 wait_offset);
      out_dependency->signal_semaphore_count =
          (iree_host_size_t)payload.signal_semaphore_count;
      out_dependency->signal_semaphore_payloads =
          (const iree_hal_replay_semaphore_timepoint_payload_t*)(record->payload
                                                                     .data +
                                                                 signal_offset);
      return iree_ok_status();
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE: {
      iree_hal_replay_plan_queue_execute_t execute;
      memset(&execute, 0, sizeof(execute));
      IREE_RETURN_IF_ERROR(
          iree_hal_replay_plan_prepare_queue_execute(record, &execute));
      out_dependency->is_submission = true;
      out_dependency->command_buffer_id = execute.command_buffer_id;
      out_dependency->wait_semaphore_count = execute.wait_semaphore_count;
      out_dependency->wait_semaphore_payloads = execute.wait_semaphore_payloads;
      out_dependency->signal_semaphore_count = execute.signal_semaphore_count;
      out_dependency->signal_semaphore_payloads =
          execute.signal_semaphore_payloads;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

typedef struct iree_hal_replay_plan_semaphore_state_t {
  bool is_defined;
  uint64_t known_value;
} iree_hal_replay_plan_semaphore_state_t;

static bool iree_hal_replay_plan_queue_dependency_can_complete(
    const iree_hal_replay_plan_queue_dependency_t* dependency,
    const iree_hal_replay_plan_semaphore_state_t* semaphore_states,
    iree_host_size_t object_capacity,
    const bool* command_buffer_may_wait_on_device_memory) {
  if (dependency->may_wait_on_device_memory) return false;
  if (dependency->command_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
      command_buffer_may_wait_on_device_memory[dependency->command_buffer_id]) {
    return false;
  }
  for (iree_host_size_t i = 0; i < dependency->wait_semaphore_count; ++i) {
    const iree_hal_replay_semaphore_timepoint_payload_t wait =
        dependency->wait_semaphore_payloads[i];
    IREE_ASSERT(wait.semaphore_id < object_capacity);
    if (semaphore_states[wait.semaphore_id].known_value < wait.value) {
      return false;
    }
  }
  return true;
}

static void iree_hal_replay_plan_publish_queue_signals(
    const iree_hal_replay_plan_queue_dependency_t* dependency,
    iree_hal_replay_plan_semaphore_state_t* semaphore_states) {
  for (iree_host_size_t i = 0; i < dependency->signal_semaphore_count; ++i) {
    const iree_hal_replay_semaphore_timepoint_payload_t signal =
        dependency->signal_semaphore_payloads[i];
    if (semaphore_states[signal.semaphore_id].known_value < signal.value) {
      semaphore_states[signal.semaphore_id].known_value = signal.value;
    }
  }
}

static bool iree_hal_replay_plan_is_completion_boundary(
    const iree_hal_replay_file_record_t* record) {
  if (record->header.record_type !=
          IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION ||
      record->header.status_code != IREE_STATUS_OK) {
    return false;
  }
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT:
      return true;
    default:
      return false;
  }
}

static iree_status_t iree_hal_replay_plan_verify_completion_dependencies(
    const iree_hal_replay_plan_t* plan, iree_allocator_t host_allocator) {
  iree_hal_replay_plan_semaphore_state_t* semaphore_states = NULL;
  bool* command_buffer_may_wait_on_device_memory = NULL;
  iree_hal_replay_plan_queue_dependency_t* pending_dependencies = NULL;
  iree_status_t status = iree_ok_status();
  if (plan->object_capacity != 0) {
    iree_host_size_t semaphore_states_size = 0;
    iree_host_size_t command_buffer_flags_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(plan->object_capacity,
                                                  sizeof(*semaphore_states),
                                                  &semaphore_states_size) ||
                      !iree_host_size_checked_mul(
                          plan->object_capacity,
                          sizeof(*command_buffer_may_wait_on_device_memory),
                          &command_buffer_flags_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "replay dependency table size overflow");
    }
    status = iree_allocator_malloc(host_allocator, semaphore_states_size,
                                   (void**)&semaphore_states);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(
          host_allocator, command_buffer_flags_size,
          (void**)&command_buffer_may_wait_on_device_memory);
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, semaphore_states);
      return status;
    }
    memset(semaphore_states, 0, semaphore_states_size);
    memset(command_buffer_may_wait_on_device_memory, 0,
           command_buffer_flags_size);
  }
  if (plan->record_count != 0) {
    iree_host_size_t pending_dependencies_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            plan->record_count, sizeof(*pending_dependencies),
            &pending_dependencies_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay pending dependency table overflow");
    } else {
      status = iree_allocator_malloc(host_allocator, pending_dependencies_size,
                                     (void**)&pending_dependencies);
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator,
                          command_buffer_may_wait_on_device_memory);
      iree_allocator_free(host_allocator, semaphore_states);
      return status;
    }
  }

  iree_host_size_t pending_dependency_count = 0;
  for (iree_host_size_t i = 0;
       i < plan->record_count && iree_status_is_ok(status); ++i) {
    const iree_hal_replay_file_record_t* record = &plan->records[i].file_record;
    if (record->header.record_type ==
            IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION &&
        record->header.status_code == IREE_STATUS_OK) {
      if (record->header.operation_code ==
          IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE) {
        if (record->payload.data_length >=
                sizeof(iree_hal_replay_semaphore_object_payload_t) &&
            record->header.related_object_id < plan->object_capacity) {
          iree_hal_replay_semaphore_object_payload_t payload;
          memcpy(&payload, record->payload.data, sizeof(payload));
          iree_hal_replay_plan_semaphore_state_t* semaphore_state =
              &semaphore_states[record->header.related_object_id];
          semaphore_state->is_defined = true;
          semaphore_state->known_value = payload.initial_value;
        }
      } else if (
          record->header.operation_code ==
              IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_WAIT &&
          record->header.object_id < plan->object_capacity) {
        command_buffer_may_wait_on_device_memory[record->header.object_id] =
            true;
      }
    }

    iree_hal_replay_plan_queue_dependency_t dependency;
    status = iree_hal_replay_plan_prepare_queue_dependency(record, &dependency);
    if (!iree_status_is_ok(status)) break;
    if (dependency.is_submission) {
      if (dependency.command_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
          dependency.command_buffer_id >= plan->object_capacity) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "replay queue command buffer id is invalid");
      }
      for (iree_host_size_t j = 0;
           j < dependency.wait_semaphore_count && iree_status_is_ok(status);
           ++j) {
        const iree_hal_replay_object_id_t semaphore_id =
            dependency.wait_semaphore_payloads[j].semaphore_id;
        if (semaphore_id >= plan->object_capacity ||
            !semaphore_states[semaphore_id].is_defined) {
          status =
              iree_make_status(IREE_STATUS_DATA_LOSS,
                               "replay queue wait semaphore id is not defined");
        }
      }
      for (iree_host_size_t j = 0;
           j < dependency.signal_semaphore_count && iree_status_is_ok(status);
           ++j) {
        const iree_hal_replay_object_id_t semaphore_id =
            dependency.signal_semaphore_payloads[j].semaphore_id;
        if (semaphore_id >= plan->object_capacity ||
            !semaphore_states[semaphore_id].is_defined) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "replay queue signal semaphore id is not defined");
        }
      }
      if (!iree_status_is_ok(status)) break;

      // Device identities and affinities do not establish independent queue
      // domains, so every unresolved submission is a possible predecessor.
      const bool can_complete =
          pending_dependency_count == 0 &&
          iree_hal_replay_plan_queue_dependency_can_complete(
              &dependency, semaphore_states, plan->object_capacity,
              command_buffer_may_wait_on_device_memory);
      if (can_complete) {
        iree_hal_replay_plan_publish_queue_signals(&dependency,
                                                   semaphore_states);
      } else if (dependency.signal_semaphore_count != 0) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay signaled queue operation at sequence %" PRIu64
            " has an unresolved completion dependency or FIFO predecessor",
            record->header.sequence_ordinal);
        break;
      } else {
        pending_dependencies[pending_dependency_count++] = dependency;
      }

      // Compact oldest-first. Once an unresolved submission is retained, all
      // later submissions remain behind that possible FIFO predecessor.
      iree_host_size_t retained_dependency_count = 0;
      for (iree_host_size_t j = 0; j < pending_dependency_count; ++j) {
        const bool pending_can_complete =
            retained_dependency_count == 0 &&
            iree_hal_replay_plan_queue_dependency_can_complete(
                &pending_dependencies[j], semaphore_states,
                plan->object_capacity,
                command_buffer_may_wait_on_device_memory);
        if (!pending_can_complete) {
          pending_dependencies[retained_dependency_count++] =
              pending_dependencies[j];
        }
      }
      pending_dependency_count = retained_dependency_count;
    }

    if (iree_hal_replay_plan_is_completion_boundary(record)) {
      if (pending_dependency_count != 0) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay completion synchronization at sequence %" PRIu64
            " follows queue work with unresolved completion dependencies",
            record->header.sequence_ordinal);
      }
    }
  }
  if (iree_status_is_ok(status) && pending_dependency_count != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay end-of-file follows queue work with unresolved completion "
        "dependencies");
  }

  iree_allocator_free(host_allocator, command_buffer_may_wait_on_device_memory);
  iree_allocator_free(host_allocator, pending_dependencies);
  iree_allocator_free(host_allocator, semaphore_states);
  return status;
}

static iree_status_t iree_hal_replay_plan_prepare_record(
    iree_hal_replay_plan_record_t* plan_record) {
  const iree_hal_replay_file_record_t* record = &plan_record->file_record;
  plan_record->kind = IREE_HAL_REPLAY_PLAN_RECORD_KIND_GENERIC;
  if (record->header.record_type !=
      IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION) {
    return iree_ok_status();
  }
  if (record->header.status_code != IREE_STATUS_OK) {
    plan_record->kind = IREE_HAL_REPLAY_PLAN_RECORD_KIND_SKIP;
    return iree_ok_status();
  }
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN:
      plan_record->kind = IREE_HAL_REPLAY_PLAN_RECORD_KIND_SCOPE;
      return iree_hal_replay_plan_prepare_scope(
          record, IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_BEGIN,
          &plan_record->payload.scope);
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END:
      plan_record->kind = IREE_HAL_REPLAY_PLAN_RECORD_KIND_SCOPE;
      return iree_hal_replay_plan_prepare_scope(
          record, IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_END,
          &plan_record->payload.scope);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER:
      plan_record->kind =
          IREE_HAL_REPLAY_PLAN_RECORD_KIND_COMMAND_BUFFER_EXECUTION_BARRIER;
      return iree_hal_replay_plan_prepare_command_buffer_execution_barrier(
          record, &plan_record->payload.command_buffer_execution_barrier);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH:
      plan_record->kind =
          IREE_HAL_REPLAY_PLAN_RECORD_KIND_COMMAND_BUFFER_DISPATCH;
      return iree_hal_replay_plan_prepare_command_buffer_dispatch(
          record, &plan_record->payload.command_buffer_dispatch);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE:
      plan_record->kind = IREE_HAL_REPLAY_PLAN_RECORD_KIND_QUEUE_EXECUTE;
      return iree_hal_replay_plan_prepare_queue_execute(
          record, &plan_record->payload.queue_execute);
    default:
      return iree_ok_status();
  }
}

IREE_API_EXPORT void iree_hal_replay_plan_destroy(
    iree_hal_replay_plan_t* plan) {
  if (!plan) return;
  iree_allocator_t host_allocator = plan->host_allocator;
  iree_allocator_free(host_allocator, plan->records);
  iree_allocator_free(host_allocator, plan);
}

IREE_API_EXPORT iree_status_t iree_hal_replay_plan_create(
    iree_const_byte_span_t file_contents, iree_allocator_t host_allocator,
    iree_hal_replay_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t first_record_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_file_parse_header(
      file_contents, &file_header, &first_record_offset));
  iree_const_byte_span_t valid_contents = file_contents;
  if (file_header.file_length != 0) {
    valid_contents.data_length = (iree_host_size_t)file_header.file_length;
  }

  iree_host_size_t record_count = 0;
  iree_hal_replay_object_id_t max_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_host_size_t offset = first_record_offset;
  uint64_t expected_sequence_ordinal = 0;
  while (offset < valid_contents.data_length) {
    iree_hal_replay_file_record_t record;
    IREE_RETURN_IF_ERROR(iree_hal_replay_file_parse_record(
        valid_contents, offset, &record, &offset));
    if (IREE_UNLIKELY(record.header.sequence_ordinal !=
                      expected_sequence_ordinal++)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "replay record sequence ordinal mismatch");
    }
    if (record.header.object_id > max_object_id) {
      max_object_id = record.header.object_id;
    }
    if (record.header.related_object_id > max_object_id) {
      max_object_id = record.header.related_object_id;
    }
    ++record_count;
  }
  if (IREE_UNLIKELY(max_object_id >= IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay object id exceeds host size");
  }

  iree_hal_replay_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->file_contents = valid_contents;
  plan->host_allocator = host_allocator;
  plan->object_capacity = (iree_host_size_t)max_object_id + 1;
  plan->record_count = record_count;

  iree_status_t status = iree_ok_status();
  if (record_count != 0) {
    iree_host_size_t records_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            record_count, sizeof(*plan->records), &records_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay plan record table size overflow");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(host_allocator, records_size,
                                     (void**)&plan->records);
    }
    if (iree_status_is_ok(status)) {
      memset(plan->records, 0, records_size);
    }
  }

  offset = first_record_offset;
  for (iree_host_size_t i = 0; i < record_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_plan_record_t* plan_record = &plan->records[i];
    status = iree_hal_replay_file_parse_record(
        valid_contents, offset, &plan_record->file_record, &offset);
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_plan_prepare_record(plan_record);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    iree_hal_replay_plan_destroy(plan);
  }
  return status;
}

static iree_status_t iree_hal_replay_plan_execute_scope(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_plan_record_t* record) {
  iree_hal_replay_scope_event_callback_t callback =
      executor->options->scope_event_callback;
  if (!callback.fn) return iree_ok_status();
  const iree_hal_replay_plan_scope_t* scope = &record->payload.scope;
  iree_hal_replay_scope_event_t event = {
      .sequence_ordinal = record->file_record.header.sequence_ordinal,
      .type = scope->event_type,
      .name = scope->name,
  };
  return callback.fn(callback.user_data, &event);
}

static iree_status_t
iree_hal_replay_plan_execute_command_buffer_execution_barrier(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_plan_command_buffer_execution_barrier_t* barrier) {
  iree_hal_memory_barrier_t inline_memory_barriers
      [IREE_HAL_REPLAY_INLINE_MEMORY_BARRIER_LIST_CAPACITY];
  iree_hal_memory_barrier_t* memory_barriers = NULL;
  bool memory_barriers_allocated = false;
  if (barrier->memory_barrier_count <=
      IREE_HAL_REPLAY_INLINE_MEMORY_BARRIER_LIST_CAPACITY) {
    memory_barriers = inline_memory_barriers;
  } else {
    iree_host_size_t memory_barriers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(barrier->memory_barrier_count,
                                                  sizeof(*memory_barriers),
                                                  &memory_barriers_size))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay execution barrier memory barrier count overflow");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(executor->host_allocator,
                                               memory_barriers_size,
                                               (void**)&memory_barriers));
    memory_barriers_allocated = true;
  }
  for (iree_host_size_t i = 0; i < barrier->memory_barrier_count; ++i) {
    memory_barriers[i].source_scope =
        barrier->memory_barrier_payloads[i].source_scope;
    memory_barriers[i].target_scope =
        barrier->memory_barrier_payloads[i].target_scope;
  }

  iree_hal_buffer_barrier_t inline_buffer_barriers
      [IREE_HAL_REPLAY_INLINE_BUFFER_BARRIER_LIST_CAPACITY];
  iree_hal_buffer_barrier_t* buffer_barriers = NULL;
  iree_status_t status = iree_ok_status();
  bool buffer_barriers_allocated = false;
  if (barrier->buffer_barrier_count <=
      IREE_HAL_REPLAY_INLINE_BUFFER_BARRIER_LIST_CAPACITY) {
    buffer_barriers = inline_buffer_barriers;
  } else {
    iree_host_size_t buffer_barriers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(barrier->buffer_barrier_count,
                                                  sizeof(*buffer_barriers),
                                                  &buffer_barriers_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay execution barrier buffer barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, buffer_barriers_size,
                                (void**)&buffer_barriers);
    }
    buffer_barriers_allocated = iree_status_is_ok(status);
  }
  for (iree_host_size_t i = 0;
       i < barrier->buffer_barrier_count && iree_status_is_ok(status); ++i) {
    buffer_barriers[i].source_scope =
        barrier->buffer_barrier_payloads[i].source_scope;
    buffer_barriers[i].target_scope =
        barrier->buffer_barrier_payloads[i].target_scope;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &barrier->buffer_barrier_payloads[i].buffer_ref,
        &buffer_barriers[i].buffer_ref);
  }

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, barrier->command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_execution_barrier(
        command_buffer_entry->value.command_buffer, barrier->source_stage_mask,
        barrier->target_stage_mask, barrier->flags,
        barrier->memory_barrier_count, memory_barriers,
        barrier->buffer_barrier_count, buffer_barriers);
  }
  if (buffer_barriers_allocated) {
    iree_allocator_free(executor->host_allocator, buffer_barriers);
  }
  if (memory_barriers_allocated) {
    iree_allocator_free(executor->host_allocator, memory_barriers);
  }
  return status;
}

static iree_status_t iree_hal_replay_plan_execute_command_buffer_dispatch(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_plan_command_buffer_dispatch_t* dispatch) {
  iree_hal_replay_buffer_ref_list_storage_t binding_storage = {0};
  IREE_RETURN_IF_ERROR(iree_hal_replay_buffer_ref_list_storage_initialize(
      executor, dispatch->binding_count, &binding_storage));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < dispatch->binding_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &dispatch->binding_payloads[i], &binding_storage.values[i]);
  }

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, dispatch->command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  iree_hal_replay_object_entry_t* executable_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, dispatch->executable_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE, &executable_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_dispatch_config_t config = dispatch->config;
    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &dispatch->workgroup_count_ref, &config.workgroup_count_ref);
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_executor_resolve_function(
          dispatch->executable_id, &executable_entry->value.executable,
          dispatch->function_ordinal, &function);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_dispatch(
          command_buffer_entry->value.command_buffer,
          executable_entry->value.executable.handle, function, config,
          dispatch->constants, binding_storage.list, dispatch->flags);
    }
  }
  iree_hal_replay_buffer_ref_list_storage_deinitialize(
      &binding_storage, executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_plan_execute_queue_execute(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_plan_queue_execute_t* queue_execute) {
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(
          (const uint8_t*)queue_execute->wait_semaphore_payloads,
          queue_execute->wait_semaphore_count *
              sizeof(*queue_execute->wait_semaphore_payloads)),
      queue_execute->wait_semaphore_count, /*additional_capacity=*/0,
      &wait_storage));
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(
          (const uint8_t*)queue_execute->signal_semaphore_payloads,
          queue_execute->signal_semaphore_count *
              sizeof(*queue_execute->signal_semaphore_payloads)),
      queue_execute->signal_semaphore_count, /*additional_capacity=*/1,
      &signal_storage);

  iree_hal_replay_buffer_binding_table_storage_t binding_storage = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_buffer_binding_table_storage_initialize(
        executor, queue_execute->binding_count, &binding_storage);
  }
  for (iree_host_size_t i = 0;
       i < queue_execute->binding_count && iree_status_is_ok(status); ++i) {
    iree_hal_buffer_ref_t ref;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &queue_execute->binding_payloads[i], &ref);
    if (iree_status_is_ok(status)) {
      binding_storage.bindings[i] = (iree_hal_buffer_binding_t){
          .buffer = ref.buffer,
          .offset = ref.offset,
          .length = ref.length,
      };
    }
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, queue_execute->device_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                             &device_entry);
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status) &&
      queue_execute->command_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    status = iree_hal_replay_executor_lookup(
        executor, queue_execute->command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  iree_hal_replay_queue_completion_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_prepare_queue_completion(
        executor, device_entry->value.device, queue_execute->queue_affinity,
        &signal_storage, &completion);
  }
  if (iree_status_is_ok(status)) {
    if (command_buffer_entry) {
      status = iree_hal_device_queue_execute(
          device_entry->value.device, queue_execute->queue_affinity,
          wait_storage.list, signal_storage.list,
          command_buffer_entry->value.command_buffer, binding_storage.table,
          queue_execute->flags);
    } else if (queue_execute->binding_count == 0) {
      status = iree_hal_device_queue_barrier(
          device_entry->value.device, queue_execute->queue_affinity,
          wait_storage.list, signal_storage.list, queue_execute->flags);
    } else {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "replay queue barrier payload unexpectedly has bindings");
    }
  }
  status = iree_hal_replay_executor_finalize_queue_completion(
      executor, completion, status);

  iree_hal_replay_buffer_binding_table_storage_deinitialize(
      &binding_storage, executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_plan_execute_record(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_plan_record_t* plan_record) {
  const iree_hal_replay_file_record_t* record = &plan_record->file_record;
  switch (record->header.record_type) {
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
      return iree_ok_status();
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
      return iree_hal_replay_executor_replay_object(executor, record);
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
      switch (plan_record->kind) {
        case IREE_HAL_REPLAY_PLAN_RECORD_KIND_SKIP:
          return iree_ok_status();
        case IREE_HAL_REPLAY_PLAN_RECORD_KIND_SCOPE:
          return iree_hal_replay_plan_execute_scope(executor, plan_record);
        case IREE_HAL_REPLAY_PLAN_RECORD_KIND_COMMAND_BUFFER_EXECUTION_BARRIER:
          return iree_hal_replay_plan_execute_command_buffer_execution_barrier(
              executor, &plan_record->payload.command_buffer_execution_barrier);
        case IREE_HAL_REPLAY_PLAN_RECORD_KIND_COMMAND_BUFFER_DISPATCH:
          return iree_hal_replay_plan_execute_command_buffer_dispatch(
              executor, &plan_record->payload.command_buffer_dispatch);
        case IREE_HAL_REPLAY_PLAN_RECORD_KIND_QUEUE_EXECUTE:
          return iree_hal_replay_plan_execute_queue_execute(
              executor, &plan_record->payload.queue_execute);
        case IREE_HAL_REPLAY_PLAN_RECORD_KIND_GENERIC:
        default:
          return iree_hal_replay_executor_replay_operation(executor, record);
      }
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
      return iree_hal_replay_executor_replay_unsupported(record);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED, "replay record type %s is not executable",
          iree_hal_replay_file_record_type_string(record->header.record_type));
  }
}

IREE_API_EXPORT iree_status_t iree_hal_replay_plan_execute(
    const iree_hal_replay_plan_t* plan, iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(device_group);
  iree_hal_replay_execute_options_t default_options =
      iree_hal_replay_execute_options_default();
  if (!options) options = &default_options;
  if (IREE_UNLIKELY(options->flags != IREE_HAL_REPLAY_EXECUTE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay execute flags are unsupported");
  }
  if (IREE_UNLIKELY(options->file_path_remap_count != 0 &&
                    !options->file_path_remaps)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replay execute file path remaps require a remap list");
  }

  IREE_RETURN_IF_ERROR(iree_hal_replay_plan_verify_completion_dependencies(
      plan, host_allocator));

  iree_hal_replay_executor_t executor;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_initialize(
      &executor, plan->file_contents, plan->object_capacity, device_group,
      options, host_allocator));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < plan->record_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_plan_execute_record(&executor, &plan->records[i]);
  }
  if (iree_status_is_ok(status) &&
      executor.next_device_index !=
          iree_hal_device_group_device_count(executor.device_group)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay captured %" PRIhsz
        " device(s) but the target device group contains %" PRIhsz " device(s)",
        executor.next_device_index,
        iree_hal_device_group_device_count(executor.device_group));
  }

  if (!iree_status_is_ok(status) && executor.queue_completion_count != 0) {
    // Some accepted queue work may depend on records that were not submitted
    // after the execution error. Waiting could block forever, while releasing
    // replay objects could race that work. Retain the executor-owned state for
    // process-lifetime safety; a recoverable cleanup API can be added later.
    return status;
  }

  return iree_status_join(status,
                          iree_hal_replay_executor_deinitialize(&executor));
}

IREE_API_EXPORT iree_status_t iree_hal_replay_execute_file(
    iree_const_byte_span_t file_contents, iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator) {
  iree_hal_replay_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_plan_create(file_contents, host_allocator, &plan));
  iree_status_t status =
      iree_hal_replay_plan_execute(plan, device_group, options, host_allocator);
  iree_hal_replay_plan_destroy(plan);
  return status;
}
