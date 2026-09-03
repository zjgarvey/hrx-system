// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/hal/replay/recorder.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static iree_hal_device_t* CreateTaskDevice() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

static iree_hal_device_group_t* CreateDeviceGroup(
    iree_hal_device_t* const* devices, iree_host_size_t device_count) {
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    IREE_CHECK_OK(
        iree_hal_device_group_builder_add_device(&builder, devices[i]));
  }

  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  return group;
}

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  iree_hal_device_t* device = CreateTaskDevice();
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group =
      CreateDeviceGroup(devices, IREE_ARRAYSIZE(devices));
  iree_hal_device_release(device);
  return group;
}

static iree_hal_device_group_t* CreateMockExecutableDeviceGroup() {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view("mock-executable-device");
  options.executable_loading_enabled = true;

  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group =
      CreateDeviceGroup(devices, IREE_ARRAYSIZE(devices));
  iree_hal_device_release(device);
  return group;
}

static iree_hal_replay_recorder_t* CreateHostAllocationRecorder(
    std::vector<uint8_t>* storage,
    const iree_hal_replay_recorder_options_t* options = nullptr) {
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage->data(), storage->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_recorder_t* recorder = nullptr;
  IREE_CHECK_OK(iree_hal_replay_recorder_create(
      file_handle, options, iree_allocator_system(), &recorder));
  iree_io_file_handle_release(file_handle);
  return recorder;
}

static iree_const_byte_span_t GetCapturedFileContents(
    const std::vector<uint8_t>& storage) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_LE(file_header.file_length, storage.size());
  return iree_make_const_byte_span(storage.data(),
                                   (iree_host_size_t)file_header.file_length);
}

static void AppendReplayRecord(
    iree_hal_replay_file_writer_t* writer,
    const iree_hal_replay_file_record_metadata_t& metadata,
    std::initializer_list<iree_const_byte_span_t> payload_iovecs) {
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, payload_iovecs.size(), payload_iovecs.begin(),
      /*out_payload_range=*/nullptr));
}

static void AppendQueueBarrierDependencyRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_replay_object_id_t wait_semaphore_id,
    iree_hal_replay_object_id_t signal_semaphore_id) {
  const iree_hal_replay_device_queue_execute_payload_t payload = {
      /*.command_buffer_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.queue_affinity=*/queue_affinity,
      /*.flags=*/IREE_HAL_EXECUTE_FLAG_NONE,
      /*.wait_semaphore_count=*/
      wait_semaphore_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ? 0u : 1u,
      /*.signal_semaphore_count=*/
      signal_semaphore_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ? 0u : 1u,
      /*.binding_count=*/0,
  };
  const iree_hal_replay_semaphore_timepoint_payload_t wait_timepoint = {
      /*.semaphore_id=*/wait_semaphore_id,
      /*.value=*/1,
  };
  const iree_hal_replay_semaphore_timepoint_payload_t signal_timepoint = {
      /*.semaphore_id=*/signal_semaphore_id,
      /*.value=*/1,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      /*.sequence_ordinal=*/sequence_ordinal,
      /*.thread_id=*/0,
      /*.device_id=*/device_id,
      /*.object_id=*/device_id,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/
      IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE,
      /*.status_code=*/IREE_STATUS_OK,
  };
  iree_const_byte_span_t payload_iovecs[3];
  iree_host_size_t payload_iovec_count = 0;
  payload_iovecs[payload_iovec_count++] =
      iree_make_const_byte_span(&payload, sizeof(payload));
  if (payload.wait_semaphore_count != 0) {
    payload_iovecs[payload_iovec_count++] =
        iree_make_const_byte_span(&wait_timepoint, sizeof(wait_timepoint));
  }
  if (payload.signal_semaphore_count != 0) {
    payload_iovecs[payload_iovec_count++] =
        iree_make_const_byte_span(&signal_timepoint, sizeof(signal_timepoint));
  }
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, payload_iovec_count, payload_iovecs,
      /*out_payload_range=*/nullptr));
}

static void AppendQueueAllocaDependencyRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_replay_object_id_t buffer_id) {
  iree_hal_replay_device_queue_alloca_payload_t payload = {};
  payload.allocation.allocation_size = 4;
  payload.queue_affinity = queue_affinity;
  const iree_hal_replay_file_record_metadata_t metadata = {
      /*.sequence_ordinal=*/sequence_ordinal,
      /*.thread_id=*/0,
      /*.device_id=*/device_id,
      /*.object_id=*/device_id,
      /*.related_object_id=*/buffer_id,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendPreferOriginDeallocaDependencyRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_replay_object_id_t buffer_id,
    iree_hal_replay_object_id_t signal_semaphore_id) {
  iree_hal_replay_device_queue_dealloca_payload_t payload = {};
  payload.buffer_ref.buffer_id = buffer_id;
  payload.buffer_ref.length = 4;
  payload.queue_affinity = queue_affinity;
  payload.flags = IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN;
  payload.signal_semaphore_count = 1;
  const iree_hal_replay_semaphore_timepoint_payload_t signal_timepoint = {
      /*.semaphore_id=*/signal_semaphore_id,
      /*.value=*/1,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      /*.sequence_ordinal=*/sequence_ordinal,
      /*.thread_id=*/0,
      /*.device_id=*/device_id,
      /*.object_id=*/device_id,
      /*.related_object_id=*/buffer_id,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, metadata,
      {iree_make_const_byte_span(&payload, sizeof(payload)),
       iree_make_const_byte_span(&signal_timepoint, sizeof(signal_timepoint))});
}

static void AppendQueueBarrierRecord(iree_hal_replay_file_writer_t* writer,
                                     uint64_t sequence_ordinal, bool is_wait) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kSemaphoreId = 2;
  AppendQueueBarrierDependencyRecord(
      writer, sequence_ordinal, kDeviceId, /*queue_affinity=*/1,
      is_wait ? kSemaphoreId : IREE_HAL_REPLAY_OBJECT_ID_NONE,
      is_wait ? IREE_HAL_REPLAY_OBJECT_ID_NONE : kSemaphoreId);
}

static void AppendQueueAtomicWaitRecord(iree_hal_replay_file_writer_t* writer,
                                        uint64_t sequence_ordinal) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  iree_hal_replay_device_queue_atomic_wait_payload_t payload = {};
  payload.target_ref.buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  payload.target_ref.length = 4;
  payload.queue_affinity = 1;
  payload.params.value = 1;
  payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  payload.params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL;
  const iree_hal_replay_file_record_metadata_t metadata = {
      /*.sequence_ordinal=*/sequence_ordinal,
      /*.thread_id=*/0,
      /*.device_id=*/kDeviceId,
      /*.object_id=*/kDeviceId,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static std::vector<uint8_t> MakeVmmDependencyReplay(
    bool signal_before_wait, bool include_vmm_boundary = true,
    bool signal_after_wait = true, bool atomic_wait = false) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kSemaphoreId = 2;
  constexpr iree_hal_replay_object_id_t kAllocatorId = 3;
  constexpr iree_hal_replay_object_id_t kVirtualBufferId = 4;
  std::vector<uint8_t> storage(32768, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  uint64_t sequence_ordinal = 0;
  const iree_hal_replay_scope_payload_t scope_payload = {
      /*.name_length=*/7,
      /*.flags=*/IREE_HAL_REPLAY_SCOPE_FLAG_NONE,
  };
  const iree_hal_replay_file_record_metadata_t scope_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, scope_metadata,
      {iree_make_const_byte_span(&scope_payload, sizeof(scope_payload)),
       iree_make_const_byte_span("started", 7)});

  const iree_hal_replay_file_record_metadata_t device_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/kDeviceId,
      /*.object_id=*/kDeviceId,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_NONE,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, device_metadata, {});

  const iree_hal_replay_semaphore_object_payload_t semaphore_payload = {
      /*.queue_affinity=*/1,
      /*.initial_value=*/0,
      /*.flags=*/IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
  };
  const iree_hal_replay_file_record_metadata_t semaphore_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/kDeviceId,
      /*.object_id=*/kDeviceId,
      /*.related_object_id=*/kSemaphoreId,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, semaphore_metadata,
                     {iree_make_const_byte_span(&semaphore_payload,
                                                sizeof(semaphore_payload))});

  if (signal_before_wait) {
    AppendQueueBarrierRecord(writer, sequence_ordinal++, /*is_wait=*/false);
  }
  if (atomic_wait) {
    AppendQueueAtomicWaitRecord(writer, sequence_ordinal++);
  } else {
    AppendQueueBarrierRecord(writer, sequence_ordinal++, /*is_wait=*/true);
  }

  if (include_vmm_boundary) {
    const iree_hal_replay_allocator_virtual_memory_protect_payload_t
        protect_payload = {
            /*.virtual_buffer_id=*/kVirtualBufferId,
            /*.virtual_offset=*/0,
            /*.size=*/4096,
            /*.queue_affinity=*/1,
            /*.access_scope=*/IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
            /*.reserved0=*/0,
            /*.protection=*/IREE_HAL_MEMORY_PROTECTION_READ,
        };
    const iree_hal_replay_file_record_metadata_t protect_metadata = {
        /*.sequence_ordinal=*/sequence_ordinal++,
        /*.thread_id=*/0,
        /*.device_id=*/kDeviceId,
        /*.object_id=*/kAllocatorId,
        /*.related_object_id=*/kVirtualBufferId,
        /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
        /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
        /*.payload_type=*/
        IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
        /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
        /*.operation_code=*/
        IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
        /*.status_code=*/IREE_STATUS_OK,
    };
    AppendReplayRecord(
        writer, protect_metadata,
        {iree_make_const_byte_span(&protect_payload, sizeof(protect_payload))});
  }

  if (!signal_before_wait && signal_after_wait && !atomic_wait) {
    AppendQueueBarrierRecord(writer, sequence_ordinal++, /*is_wait=*/false);
  }
  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

enum class QueueDependencyOperation {
  kExecute,
  kAlloca,
  kPreferOriginDealloca,
};

struct QueueDependencyRecord {
  // Captured device object id receiving the queue operation.
  iree_hal_replay_object_id_t device_id;
  // Captured affinity selecting possible queues.
  iree_hal_queue_affinity_t queue_affinity;
  // Optional semaphore waited on by the queue operation.
  iree_hal_replay_object_id_t wait_semaphore_id;
  // Optional semaphore signaled by the queue operation.
  iree_hal_replay_object_id_t signal_semaphore_id;
  // Queue operation kind; execute is the default for compact test graphs.
  QueueDependencyOperation operation = QueueDependencyOperation::kExecute;
  // Buffer produced or consumed by allocation operations.
  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
};

static std::vector<uint8_t> MakeQueueCompletionDependencyReplay(
    std::initializer_list<QueueDependencyRecord> dependencies) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kFirstSemaphoreId = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreCount = 3;
  std::vector<uint8_t> storage(32768, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  uint64_t sequence_ordinal = 0;
  const iree_hal_replay_scope_payload_t scope_payload = {
      /*.name_length=*/7,
      /*.flags=*/IREE_HAL_REPLAY_SCOPE_FLAG_NONE,
  };
  const iree_hal_replay_file_record_metadata_t scope_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, scope_metadata,
      {iree_make_const_byte_span(&scope_payload, sizeof(scope_payload)),
       iree_make_const_byte_span("started", 7)});

  const iree_hal_replay_semaphore_object_payload_t semaphore_payload = {
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.initial_value=*/0,
      /*.flags=*/IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
  };
  for (iree_hal_replay_object_id_t i = 0; i < kSemaphoreCount; ++i) {
    const iree_hal_replay_object_id_t semaphore_id = kFirstSemaphoreId + i;
    const iree_hal_replay_file_record_metadata_t semaphore_metadata = {
        /*.sequence_ordinal=*/sequence_ordinal++,
        /*.thread_id=*/0,
        /*.device_id=*/kDeviceId,
        /*.object_id=*/kDeviceId,
        /*.related_object_id=*/semaphore_id,
        /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
        /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
        /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT,
        /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
        /*.operation_code=*/
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE,
        /*.status_code=*/IREE_STATUS_OK,
    };
    AppendReplayRecord(writer, semaphore_metadata,
                       {iree_make_const_byte_span(&semaphore_payload,
                                                  sizeof(semaphore_payload))});
  }
  for (const QueueDependencyRecord& dependency : dependencies) {
    switch (dependency.operation) {
      case QueueDependencyOperation::kExecute:
        AppendQueueBarrierDependencyRecord(
            writer, sequence_ordinal++, dependency.device_id,
            dependency.queue_affinity, dependency.wait_semaphore_id,
            dependency.signal_semaphore_id);
        break;
      case QueueDependencyOperation::kAlloca:
        AppendQueueAllocaDependencyRecord(
            writer, sequence_ordinal++, dependency.device_id,
            dependency.queue_affinity, dependency.buffer_id);
        break;
      case QueueDependencyOperation::kPreferOriginDealloca:
        AppendPreferOriginDeallocaDependencyRecord(
            writer, sequence_ordinal++, dependency.device_id,
            dependency.queue_affinity, dependency.buffer_id,
            dependency.signal_semaphore_id);
        break;
    }
  }
  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static std::vector<uint8_t> MakeDispatchThenHostDataReplay(
    bool command_buffer_dispatch) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kCommandBufferId = 2;
  constexpr iree_hal_replay_object_id_t kBufferId = 3;
  constexpr iree_hal_replay_object_id_t kExecutableId = 4;
  std::vector<uint8_t> storage(32768, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  uint64_t sequence_ordinal = 0;
  const iree_hal_replay_scope_payload_t scope_payload = {
      /*.name_length=*/7,
      /*.flags=*/IREE_HAL_REPLAY_SCOPE_FLAG_NONE,
  };
  const iree_hal_replay_file_record_metadata_t scope_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, scope_metadata,
      {iree_make_const_byte_span(&scope_payload, sizeof(scope_payload)),
       iree_make_const_byte_span("started", 7)});

  iree_hal_replay_dispatch_payload_t dispatch_payload = {};
  dispatch_payload.executable_id = kExecutableId;
  dispatch_payload.queue_affinity = command_buffer_dispatch ? 0 : 1;
  dispatch_payload.workgroup_size[0] = 1;
  dispatch_payload.workgroup_size[1] = 1;
  dispatch_payload.workgroup_size[2] = 1;
  dispatch_payload.workgroup_count[0] = 1;
  dispatch_payload.workgroup_count[1] = 1;
  dispatch_payload.workgroup_count[2] = 1;
  const iree_hal_replay_file_record_metadata_t dispatch_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/kDeviceId,
      /*.object_id=*/command_buffer_dispatch ? kCommandBufferId : kDeviceId,
      /*.related_object_id=*/kExecutableId,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/
      command_buffer_dispatch
          ? IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH
          : IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, dispatch_metadata,
      {iree_make_const_byte_span(&dispatch_payload, sizeof(dispatch_payload))});

  if (command_buffer_dispatch) {
    const iree_hal_replay_device_queue_execute_payload_t execute_payload = {
        /*.command_buffer_id=*/kCommandBufferId,
        /*.queue_affinity=*/1,
        /*.flags=*/IREE_HAL_EXECUTE_FLAG_NONE,
        /*.wait_semaphore_count=*/0,
        /*.signal_semaphore_count=*/0,
        /*.binding_count=*/0,
    };
    const iree_hal_replay_file_record_metadata_t execute_metadata = {
        /*.sequence_ordinal=*/sequence_ordinal++,
        /*.thread_id=*/0,
        /*.device_id=*/kDeviceId,
        /*.object_id=*/kDeviceId,
        /*.related_object_id=*/kCommandBufferId,
        /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
        /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
        /*.payload_type=*/
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE,
        /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
        /*.operation_code=*/
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE,
        /*.status_code=*/IREE_STATUS_OK,
    };
    AppendReplayRecord(
        writer, execute_metadata,
        {iree_make_const_byte_span(&execute_payload, sizeof(execute_payload))});
  }

  constexpr uint32_t kData = 0xA5A5A5A5u;
  const iree_hal_replay_buffer_range_data_payload_t data_payload = {
      /*.byte_offset=*/0,
      /*.byte_length=*/sizeof(kData),
      /*.data_length=*/sizeof(kData),
      /*.mapping_mode=*/0,
      /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_WRITE,
      /*.reserved0=*/0,
      /*.reserved1=*/0,
  };
  const iree_hal_replay_file_record_metadata_t data_metadata = {
      /*.sequence_ordinal=*/sequence_ordinal++,
      /*.thread_id=*/0,
      /*.device_id=*/kDeviceId,
      /*.object_id=*/kBufferId,
      /*.related_object_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.record_type=*/IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      /*.record_flags=*/IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      /*.payload_type=*/IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA,
      /*.object_type=*/IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      /*.operation_code=*/IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE,
      /*.status_code=*/IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, data_metadata,
      {iree_make_const_byte_span(&data_payload, sizeof(data_payload)),
       iree_make_const_byte_span(&kData, sizeof(kData))});

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static iree_status_t NoopHostCall(void* user_data, const uint64_t args[4],
                                  iree_hal_host_call_context_t* context) {
  (void)user_data;
  (void)args;
  (void)context;
  return iree_ok_status();
}

typedef struct ReplayScopeCallbackState {
  // Ordered scope event descriptions observed during replay.
  std::vector<std::string>* events;
} ReplayScopeCallbackState;

static iree_status_t RecordReplayScopeEvent(
    void* user_data, const iree_hal_replay_scope_event_t* event) {
  ReplayScopeCallbackState* state = (ReplayScopeCallbackState*)user_data;
  const char* prefix = nullptr;
  switch (event->type) {
    case IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_BEGIN:
      prefix = "begin:";
      break;
    case IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_END:
      prefix = "end:";
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown replay scope event type");
  }
  std::string text(prefix);
  text.append(event->name.data, event->name.size);
  state->events->push_back(text);
  return iree_ok_status();
}

static std::vector<std::string> ExecuteQueueDependencyPreflight(
    const std::vector<uint8_t>& storage) {
  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  return events;
}

TEST(ReplayExecuteTest, RejectsSignalBehindPossibleSameQueuePredecessor) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kSemaphoreS = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreT = 3;
  std::vector<uint8_t> storage = MakeQueueCompletionDependencyReplay({
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/kSemaphoreS,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreT},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/2,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreS},
  });

  EXPECT_TRUE(ExecuteQueueDependencyPreflight(storage).empty());
}

TEST(ReplayExecuteTest, RejectsDistinctAffinitiesThatMayAlias) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kSemaphoreS = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreT = 3;
  std::vector<uint8_t> storage = MakeQueueCompletionDependencyReplay({
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/kSemaphoreS,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/2,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreT},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/2,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreS},
  });

  EXPECT_TRUE(ExecuteQueueDependencyPreflight(storage).empty());
}

TEST(ReplayExecuteTest, RejectsSignalBehindDifferentDeviceIdPredecessor) {
  constexpr iree_hal_replay_object_id_t kDeviceA = 1;
  constexpr iree_hal_replay_object_id_t kDeviceB = 5;
  constexpr iree_hal_replay_object_id_t kSemaphoreS = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreT = 3;
  std::vector<uint8_t> storage = MakeQueueCompletionDependencyReplay({
      {/*.device_id=*/kDeviceA, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/kSemaphoreS,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE},
      {/*.device_id=*/kDeviceB, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreT},
      {/*.device_id=*/kDeviceB, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreS},
  });

  EXPECT_TRUE(ExecuteQueueDependencyPreflight(storage).empty());
}

TEST(ReplayExecuteTest, RejectsUnknownQueueAffinityAsPossiblePredecessor) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kSemaphoreS = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreT = 3;
  std::vector<uint8_t> storage = MakeQueueCompletionDependencyReplay({
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/0,
       /*.wait_semaphore_id=*/kSemaphoreS,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/2,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreT},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/2,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreS},
  });

  EXPECT_TRUE(ExecuteQueueDependencyPreflight(storage).empty());
}

TEST(ReplayExecuteTest, RejectsPreferOriginDeallocaWithPendingOriginQueue) {
  constexpr iree_hal_replay_object_id_t kCallsiteDevice = 1;
  constexpr iree_hal_replay_object_id_t kOriginDevice = 5;
  constexpr iree_hal_replay_object_id_t kBufferId = 6;
  constexpr iree_hal_replay_object_id_t kSemaphoreS = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreT = 3;
  std::vector<uint8_t> storage = MakeQueueCompletionDependencyReplay({
      {/*.device_id=*/kOriginDevice, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.operation=*/QueueDependencyOperation::kAlloca,
       /*.buffer_id=*/kBufferId},
      {/*.device_id=*/kOriginDevice, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/kSemaphoreS,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE},
      {/*.device_id=*/kCallsiteDevice, /*.queue_affinity=*/2,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreT,
       /*.operation=*/QueueDependencyOperation::kPreferOriginDealloca,
       /*.buffer_id=*/kBufferId},
      {/*.device_id=*/kCallsiteDevice, /*.queue_affinity=*/4,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreS},
  });

  EXPECT_TRUE(ExecuteQueueDependencyPreflight(storage).empty());
}

TEST(ReplayExecuteTest, AllowsSignalAfterResolvedSubmission) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kSemaphoreS = 2;
  constexpr iree_hal_replay_object_id_t kSemaphoreT = 3;
  std::vector<uint8_t> storage = MakeQueueCompletionDependencyReplay({
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreS},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/kSemaphoreS,
       /*.signal_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE},
      {/*.device_id=*/kDeviceId, /*.queue_affinity=*/1,
       /*.wait_semaphore_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
       /*.signal_semaphore_id=*/kSemaphoreT},
  });

  std::vector<std::string> events = ExecuteQueueDependencyPreflight(storage);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0], "begin:started");
}

TEST(ReplayExecuteTest, RejectsForwardQueueDependencyBeforeVmmBoundary) {
  std::vector<uint8_t> storage =
      MakeVmmDependencyReplay(/*signal_before_wait=*/false);
  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
  EXPECT_TRUE(events.empty());
}

TEST(ReplayExecuteTest, AllowsSatisfiedQueueDependencyBeforeVmmBoundary) {
  std::vector<uint8_t> storage =
      MakeVmmDependencyReplay(/*signal_before_wait=*/true);
  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  // The dependency preflight passes and execution reaches the intentionally
  // undefined allocator used as the terminal VMM boundary fixture.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0], "begin:started");
}
TEST(ReplayExecuteTest, RejectsSameQueueForwardDependencyAtEndOfFile) {
  std::vector<uint8_t> storage = MakeVmmDependencyReplay(
      /*signal_before_wait=*/false, /*include_vmm_boundary=*/false);
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, RejectsUnresolvedQueueDependencyAtEndOfFile) {
  std::vector<uint8_t> storage = MakeVmmDependencyReplay(
      /*signal_before_wait=*/false, /*include_vmm_boundary=*/false,
      /*signal_after_wait=*/false);
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, RejectsDeviceMemoryWaitAtEndOfFile) {
  std::vector<uint8_t> storage = MakeVmmDependencyReplay(
      /*signal_before_wait=*/false, /*include_vmm_boundary=*/false,
      /*signal_after_wait=*/false, /*atomic_wait=*/true);
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, AllowsDirectDispatchBeforeCapturedHostData) {
  std::vector<uint8_t> storage =
      MakeDispatchThenHostDataReplay(/*command_buffer_dispatch=*/false);
  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0], "begin:started");
}

TEST(ReplayExecuteTest, AllowsCommandBufferDispatchBeforeCapturedHostData) {
  std::vector<uint8_t> storage =
      MakeDispatchThenHostDataReplay(/*command_buffer_dispatch=*/true);
  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0], "begin:started");
}

typedef struct MockExecutableFunctionRecord {
  // Number of 32-bit constant words reflected for the function.
  uint8_t constant_count;
  // Number of buffer bindings reflected for the function.
  uint8_t binding_count;
  // Executable function flags byte.
  uint8_t flags;
  // Static workgroup size reflected for the function.
  uint8_t workgroup_size[3];
  // Byte length of the function name in the trailing name storage.
  uint8_t name_length;
  // Native ABI byte offset for one optional reflected buffer binding.
  uint8_t native_abi_offset;
  // Native ABI byte size for the optional reflected buffer binding.
  uint16_t parameter_size;
} MockExecutableFunctionRecord;

static_assert(sizeof(MockExecutableFunctionRecord) == 10);

static std::vector<uint8_t> MakeMockExecutableData(
    uint8_t constant_count, uint8_t binding_count, uint8_t workgroup_size_x,
    uint8_t native_abi_offset = 0, uint16_t parameter_size = 0) {
  const char name[] = "main";
  if (native_abi_offset != 0 && parameter_size == 0) {
    parameter_size = sizeof(void*);
  }
  std::vector<uint8_t> data(
      4 + sizeof(MockExecutableFunctionRecord) + sizeof(name) - 1, 0);
  const uint32_t function_count = 1;
  std::memcpy(data.data(), &function_count, sizeof(function_count));
  const MockExecutableFunctionRecord record = {
      /*.constant_count=*/constant_count,
      /*.binding_count=*/binding_count,
      /*.flags=*/0,
      /*.workgroup_size=*/{workgroup_size_x, 1, 1},
      /*.name_length=*/sizeof(name) - 1,
      /*.native_abi_offset=*/native_abi_offset,
      /*.parameter_size=*/parameter_size,
  };
  std::memcpy(data.data() + 4, &record, sizeof(record));
  std::memcpy(data.data() + 4 + sizeof(record), name, sizeof(name) - 1);
  return data;
}

typedef struct MockExecutableFunction {
  // Function name stored in the executable's trailing name storage.
  const char* name;
  // Number of 32-bit constant words reflected for the function.
  uint8_t constant_count;
  // Number of buffer bindings reflected for the function.
  uint8_t binding_count;
  // Static X dimension reflected for the function's workgroup size.
  uint8_t workgroup_size_x;
} MockExecutableFunction;

static std::vector<uint8_t> MakeNamedMockExecutableData(
    std::initializer_list<MockExecutableFunction> functions) {
  const uint32_t function_count = (uint32_t)functions.size();
  std::vector<uint8_t> data(
      4 + functions.size() * sizeof(MockExecutableFunctionRecord), 0);
  std::memcpy(data.data(), &function_count, sizeof(function_count));

  size_t function_ordinal = 0;
  size_t name_offset = data.size();
  for (const MockExecutableFunction& function_record : functions) {
    const size_t name_length = strlen(function_record.name);
    if (name_length > UINT8_MAX) {
      ADD_FAILURE() << "mock function name too long";
      return {};
    }
    data.resize(data.size() + name_length);

    const MockExecutableFunctionRecord record = {
        /*.constant_count=*/function_record.constant_count,
        /*.binding_count=*/function_record.binding_count,
        /*.flags=*/0,
        /*.workgroup_size=*/{function_record.workgroup_size_x, 1, 1},
        /*.name_length=*/(uint8_t)name_length,
        /*.native_abi_offset=*/0,
        /*.parameter_size=*/0,
    };
    std::memcpy(data.data() + 4 + function_ordinal * sizeof(record), &record,
                sizeof(record));
    if (name_length != 0) {
      std::memcpy(data.data() + name_offset, function_record.name, name_length);
      name_offset += name_length;
    }
    ++function_ordinal;
  }
  return data;
}

static void CaptureMockExecutableLoad(iree_const_byte_span_t executable_data,
                                      std::vector<uint8_t>* storage) {
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(storage, nullptr);

  iree_hal_device_group_t* source_group = CreateMockExecutableDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  const iree_hal_executable_target_selection_t target_selection = {
      /*.family=*/IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_FAMILY),
      /*.target_key=*/IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_KEY),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_VIRTUAL,
      /*.physical_device_affinity=*/1,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(wrapped_device), &target_selection);
  ASSERT_EQ(target_result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data = executable_data;

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_device_load_executable(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, target_result.target,
      &load_params, &executable));

  iree_hal_executable_release(executable);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
  iree_hal_replay_recorder_release(recorder);
}

static void CorruptFirstCapturedExecutableData(std::vector<uint8_t>* storage) {
  iree_const_byte_span_t file_contents =
      iree_make_const_byte_span(storage->data(), storage->size());
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t record_offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_header(file_contents, &file_header,
                                                   &record_offset));
  file_contents = iree_make_const_byte_span(
      storage->data(), static_cast<iree_host_size_t>(file_header.file_length));

  while (record_offset < file_contents.data_length) {
    iree_hal_replay_file_record_t record;
    iree_host_size_t next_record_offset = record_offset;
    IREE_ASSERT_OK(iree_hal_replay_file_parse_record(
        file_contents, record_offset, &record, &next_record_offset));
    if (record.header.payload_type !=
        IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_LOAD) {
      record_offset = next_record_offset;
      continue;
    }

    iree_hal_replay_executable_load_payload_t payload;
    ASSERT_GE(record.payload.data_length, sizeof(payload));
    std::memcpy(&payload, record.payload.data, sizeof(payload));
    const iree_host_size_t data_offset = sizeof(payload) +
                                         payload.target_family_length +
                                         payload.target_key_length;
    ASSERT_GE(payload.executable_data_length, sizeof(uint32_t));
    ASSERT_LE(data_offset + sizeof(uint32_t), record.payload.data_length);
    auto* mutable_payload =
        storage->data() + (record.payload.data - file_contents.data);
    const uint32_t impossible_function_count = 0xFFFFFFFFu;
    std::memcpy(mutable_payload + data_offset, &impossible_function_count,
                sizeof(impossible_function_count));
    return;
  }

  FAIL() << "expected an executable load record";
}

typedef struct TestExecutableSubstitutionState {
  iree_string_view_t source;
  iree_const_byte_span_t executable_data;
  iree_host_size_t invocation_count;
  iree_hal_replay_object_id_t executable_id;
} TestExecutableSubstitutionState;

static iree_status_t TestExecutableSubstitutionCallback(
    void* user_data,
    const iree_hal_replay_executable_substitution_request_t* request,
    iree_hal_replay_executable_substitution_t* out_substitution) {
  TestExecutableSubstitutionState* state =
      (TestExecutableSubstitutionState*)user_data;
  ++state->invocation_count;
  state->executable_id = request->executable_id;
  EXPECT_EQ(request->device_id, 1u);
  EXPECT_EQ(request->executable_id, 2u);
  EXPECT_EQ(std::string_view(request->captured_target->family.data,
                             request->captured_target->family.size),
            std::string_view(IREE_HAL_MOCK_EXECUTABLE_TARGET_FAMILY));
  EXPECT_EQ(std::string_view(request->captured_target->target_key.data,
                             request->captured_target->target_key.size),
            std::string_view(IREE_HAL_MOCK_EXECUTABLE_TARGET_KEY));
  EXPECT_EQ(request->captured_target->kind_flags,
            IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_VIRTUAL);
  memset(out_substitution, 0, sizeof(*out_substitution));
  out_substitution->substitute = true;
  out_substitution->source = state->source;
  out_substitution->executable_data = state->executable_data;
  return iree_ok_status();
}

TEST(ReplayExecuteTest, ObservesScopeEvents) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_begin(
      recorder, iree_make_cstring_view("execute")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_end(
      recorder, iree_make_cstring_view("execute")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);

  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0], "begin:execute");
  EXPECT_EQ(events[1], "end:execute");
}

TEST(ReplayExecuteTest, ExecutesPreparedPlanRepeatedly) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_begin(
      recorder, iree_make_cstring_view("execute")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_end(
      recorder, iree_make_cstring_view("execute")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_replay_plan_t* plan = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_plan_create(GetCapturedFileContents(storage),
                                             iree_allocator_system(), &plan));

  std::vector<std::string> events;
  ReplayScopeCallbackState callback_state = {
      /*.events=*/&events,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = RecordReplayScopeEvent;
  options.scope_event_callback.user_data = &callback_state;

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_plan_execute(plan, replay_group, &options,
                                              iree_allocator_system()));
  IREE_EXPECT_OK(iree_hal_replay_plan_execute(plan, replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_plan_destroy(plan);

  ASSERT_EQ(events.size(), 4u);
  EXPECT_EQ(events[0], "begin:execute");
  EXPECT_EQ(events[1], "end:execute");
  EXPECT_EQ(events[2], "begin:execute");
  EXPECT_EQ(events[3], "end:execute");
}

TEST(ReplayExecuteTest, SubstitutesRecordedExecutablePayload) {
  std::vector<uint8_t> captured_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> replacement_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutableLoad(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  EXPECT_EQ(substitution_state.executable_id, 2u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, UsesRecordedExecutableMetadataForSubstitution) {
  std::vector<uint8_t> captured_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> replacement_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutableLoad(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);
  CorruptFirstCapturedExecutableData(&storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, SubstitutesRecordedExecutablePayloadByName) {
  std::vector<uint8_t> captured_data = MakeNamedMockExecutableData({
      {
          /*.name=*/"first",
          /*.constant_count=*/2,
          /*.binding_count=*/3,
          /*.workgroup_size_x=*/4,
      },
      {
          /*.name=*/"second",
          /*.constant_count=*/5,
          /*.binding_count=*/6,
          /*.workgroup_size_x=*/7,
      },
  });
  std::vector<uint8_t> replacement_data = MakeNamedMockExecutableData({
      {
          /*.name=*/"second",
          /*.constant_count=*/5,
          /*.binding_count=*/6,
          /*.workgroup_size_x=*/7,
      },
      {
          /*.name=*/"first",
          /*.constant_count=*/2,
          /*.binding_count=*/3,
          /*.workgroup_size_x=*/4,
      },
  });
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutableLoad(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);
  CorruptFirstCapturedExecutableData(&storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, RejectsExecutableSubstitutionAbiMismatch) {
  std::vector<uint8_t> captured_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> replacement_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/4,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutableLoad(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest,
     RejectsExecutableSubstitutionNativeParameterLayoutMismatch) {
  std::vector<uint8_t> captured_data = MakeMockExecutableData(
      /*constant_count=*/2, /*binding_count=*/3, /*workgroup_size_x=*/4,
      /*native_abi_offset=*/8);
  std::vector<uint8_t> replacement_data = MakeMockExecutableData(
      /*constant_count=*/2, /*binding_count=*/3, /*workgroup_size_x=*/4,
      /*native_abi_offset=*/16);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutableLoad(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);
  CorruptFirstCapturedExecutableData(&storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, PreservesWideNativeParameterSize) {
  std::vector<uint8_t> captured_data = MakeMockExecutableData(
      /*constant_count=*/2, /*binding_count=*/3, /*workgroup_size_x=*/4,
      /*native_abi_offset=*/8, /*parameter_size=*/512);
  std::vector<uint8_t> replacement_data = MakeMockExecutableData(
      /*constant_count=*/2, /*binding_count=*/3, /*workgroup_size_x=*/4,
      /*native_abi_offset=*/8, /*parameter_size=*/512);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutableLoad(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, RejectsTargetDeviceCountMismatch) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_t* replay_device_a = CreateTaskDevice();
  iree_hal_device_t* replay_device_b = CreateTaskDevice();
  iree_hal_device_t* replay_devices[] = {replay_device_a, replay_device_b};
  iree_hal_device_group_t* replay_group =
      CreateDeviceGroup(replay_devices, IREE_ARRAYSIZE(replay_devices));
  iree_hal_device_release(replay_device_a);
  iree_hal_device_release(replay_device_b);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));

  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
iree::testing::TempFilePath WriteTempFile(iree_const_byte_span_t contents) {
  iree::testing::TempFilePath path("iree_hal_replay_file");
  IREE_EXPECT_OK(iree_io_file_contents_write(path.path_view(), contents,
                                             iree_allocator_system()));
  return path;
}

void RenameToUniquePath(iree::testing::TempFilePath* path) {
  iree::testing::TempFilePath new_path("iree_hal_replay_file");
  EXPECT_EQ(0, rename(path->path().c_str(), new_path.path().c_str()));
  *path = std::move(new_path);
}

static void CaptureFdBackedQueueRead(
    iree_string_view_t source_path,
    const iree_hal_replay_recorder_options_t* recorder_options,
    std::vector<uint8_t>* storage) {
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(storage, recorder_options);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS, source_path,
      iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
  iree_hal_replay_recorder_release(recorder);
}
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

TEST(ReplayExecuteTest, ExecutesRecordedMappedBufferWrite) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, 16, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, 16, &span));
  const uint8_t contents[16] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  std::memcpy(span.data, contents, sizeof(contents));
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&mapping, 0, 16));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, RejectsAmbiguousQueueDependencyAcrossCapturedHostData) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t value = 1;
  const iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphores),
                                                    semaphores, &value};

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, semaphore_list,
      iree_hal_semaphore_list_empty(), /*command_buffer=*/nullptr,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, 16, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, 16, &span));
  const uint8_t contents[16] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  std::memcpy(span.data, contents, sizeof(contents));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), semaphore_list,
      /*command_buffer=*/nullptr, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedHostAllocationFileRead) {
  uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(file_contents, sizeof(file_contents)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedFdBackedQueueRead) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesCapturedFdBackedQueueReadWithoutSourceFile) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_options_t recorder_options =
      iree_hal_replay_recorder_options_default();
  recorder_options.external_file_policy =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_ALL;
  CaptureFdBackedQueueRead(source_file.path_view(), &recorder_options,
                           &storage);

  RenameToUniquePath(&source_file);
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest,
     ExecutesRangeCapturedFdBackedQueueReadWithoutSourceFile) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_options_t recorder_options =
      iree_hal_replay_recorder_options_default();
  recorder_options.external_file_policy =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_RANGES;
  CaptureFdBackedQueueRead(source_file.path_view(), &recorder_options,
                           &storage);

  RenameToUniquePath(&source_file);
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesRemappedFdBackedQueueRead) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  std::string captured_path = source_file.path();

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  RenameToUniquePath(&source_file);
  iree_hal_replay_file_path_remap_t file_path_remap = {
      iree_make_string_view(captured_path.data(), captured_path.size()),
      source_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, CopiedFdBackedQueueReadFailsIdentityValidation) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  iree::testing::TempFilePath copied_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  CaptureFdBackedQueueRead(source_file.path_view(),
                           /*recorder_options=*/nullptr, &storage);

  iree_hal_replay_file_path_remap_t file_path_remap = {
      source_file.path_view(),
      copied_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesDigestValidatedCopiedFdBackedQueueRead) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  iree::testing::TempFilePath copied_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_options_t recorder_options =
      iree_hal_replay_recorder_options_default();
  recorder_options.external_file_validation =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
  CaptureFdBackedQueueRead(source_file.path_view(), &recorder_options,
                           &storage);

  iree_hal_replay_file_path_remap_t file_path_remap = {
      source_file.path_view(),
      copied_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, DigestValidatedFdBackedQueueReadRejectsWrongBytes) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  uint8_t wrong_file_contents[32];
  std::memcpy(wrong_file_contents, file_contents, sizeof(file_contents));
  wrong_file_contents[7] ^= 0xFF;
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  iree::testing::TempFilePath wrong_file =
      WriteTempFile(iree_make_const_byte_span(wrong_file_contents,
                                              sizeof(wrong_file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_options_t recorder_options =
      iree_hal_replay_recorder_options_default();
  recorder_options.external_file_validation =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
  CaptureFdBackedQueueRead(source_file.path_view(), &recorder_options,
                           &storage);

  iree_hal_replay_file_path_remap_t file_path_remap = {
      source_file.path_view(),
      wrong_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesRecordedQueueAlloca) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      signal_values,
  };

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, /*pool=*/nullptr, params,
      16, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(signal_semaphore);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, RejectsUnsupportedHostCallRecord) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_host_call_t call =
      iree_hal_make_host_call(NoopHostCall, /*user_data=*/nullptr);
  const uint64_t args[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_device_queue_host_call(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(), call,
      args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesHostAllocationImportedBufferRecord) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  alignas(64) uint8_t imported_storage[16] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  external_buffer.size = sizeof(imported_storage);
  external_buffer.handle.host_allocation.ptr = imported_storage;

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* imported_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer,
      iree_hal_buffer_release_callback_null(), &imported_buffer));

  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, sizeof(imported_storage), &target_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, imported_buffer,
      /*source_offset=*/0, target_buffer, /*target_offset=*/0,
      sizeof(imported_storage), IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(imported_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, nullptr,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, SkipsFailedOperationRecords) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_channel_params_t channel_params = {};
  iree_hal_channel_t* channel = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_channel_create(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
                              channel_params, &channel));
  EXPECT_EQ(nullptr, channel);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, nullptr,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, SkipsFailedUnsupportedImportedBufferRecord) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD;
  external_buffer.size = 16;

  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* imported_buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_allocator_import_buffer(allocator, params, &external_buffer,
                                       iree_hal_buffer_release_callback_null(),
                                       &imported_buffer));
  EXPECT_EQ(nullptr, imported_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, nullptr,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedQueueTransfersAndDealloca) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_STORAGE;

  iree_hal_buffer_t* source_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 32,
                                                    &source_buffer));
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 32,
                                                    &target_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};

  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  const uint8_t update_data[12] = {
      0xF0, 0xF1, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0xF2, 0xF3,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, update_data,
      /*source_offset=*/2, source_buffer, /*target_offset=*/0, /*length=*/8,
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t fill_pattern = 0xA5A5A5A5u;
  signal_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, target_buffer,
      /*target_offset=*/0, /*length=*/16, &fill_pattern, sizeof(fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  signal_value = 3;
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, source_buffer,
      /*source_offset=*/0, target_buffer, /*target_offset=*/8, /*length=*/8,
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_t* transient_buffer = nullptr;
  signal_value = 4;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, /*pool=*/nullptr, params,
      16, IREE_HAL_ALLOCA_FLAG_NONE, &transient_buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t wait_value = 4;
  iree_hal_semaphore_list_t wait_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &wait_value,
  };
  signal_value = 5;
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(transient_buffer);
  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedQueueBarrier) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};

  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, /*command_buffer=*/nullptr,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedCommandBufferTransfers) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_STORAGE;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 32, &buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  uint32_t fill_pattern = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, iree_hal_make_buffer_ref(buffer, 0, 16), &fill_pattern,
      sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));
  const iree_hal_memory_barrier_t transfer_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &transfer_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  const uint8_t update_data[8] = {
      0xE0, 0x20, 0x21, 0x22, 0x23, 0xE1, 0xE2, 0xE3,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, update_data, /*source_offset=*/1,
      iree_hal_make_buffer_ref(buffer, 4, 4), IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedIndirectCommandBufferBindings) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_STORAGE;
  iree_hal_buffer_t* source_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &source_buffer));
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, iree_hal_make_buffer_ref(source_buffer, 0, 16),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, 0, 16),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  iree_hal_buffer_binding_t binding = {
      target_buffer,
      0,
      16,
  };
  iree_hal_buffer_binding_table_t binding_table = {
      1,
      &binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

}  // namespace
