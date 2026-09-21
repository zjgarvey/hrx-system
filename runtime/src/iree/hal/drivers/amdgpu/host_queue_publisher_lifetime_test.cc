// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/file.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/support/feedback.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/feedback_state.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profiling_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending_operation.h"
#include "iree/hal/drivers/amdgpu/host_queue_staging.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

constexpr iree_device_size_t kTransferSize = 4096;
constexpr iree_device_size_t kStagingSlotSize =
    IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT;
constexpr char kStickyShutdownMessage[] =
    "annotated sticky queue shutdown failure";
// Proves that a callback is not executing under |mutex| while tolerating an
// unrelated thread that transiently owns it. If the callback's own call stack
// owns the nonrecursive mutex then the outer test timeout detects the
// deadlock; a transient owner simply releases the mutex normally.
static bool AcquireAndReleaseMutex(iree_slim_mutex_t* mutex) {
  iree_slim_mutex_lock(mutex);
  iree_slim_mutex_unlock(mutex);
  return true;
}

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
class ScopedHsaWaitCounter {
 public:
  explicit ScopedHsaWaitCounter(iree_hal_amdgpu_libhsa_t* libhsa)
      : libhsa_(libhsa),
        signal_wait_scacquire_(libhsa->hsa_signal_wait_scacquire),
        signal_wait_any_(libhsa->hsa_amd_signal_wait_any) {
    IREE_ASSERT(instance_ == nullptr);
    instance_ = this;
    libhsa_->hsa_signal_wait_scacquire = SignalWaitScacquire;
    libhsa_->hsa_amd_signal_wait_any = SignalWaitAny;
  }

  ~ScopedHsaWaitCounter() {
    libhsa_->hsa_signal_wait_scacquire = signal_wait_scacquire_;
    libhsa_->hsa_amd_signal_wait_any = signal_wait_any_;
    instance_ = nullptr;
  }

  uint64_t count() const { return count_.load(); }

 private:
  using SignalWaitScacquireFn =
      decltype(std::declval<iree_hal_amdgpu_libhsa_t>()
                   .hsa_signal_wait_scacquire);
  using SignalWaitAnyFn = decltype(std::declval<iree_hal_amdgpu_libhsa_t>()
                                       .hsa_amd_signal_wait_any);

  static hsa_signal_value_t HSA_API
  SignalWaitScacquire(hsa_signal_t signal, hsa_signal_condition_t condition,
                      hsa_signal_value_t compare_value, uint64_t timeout_hint,
                      hsa_wait_state_t wait_state_hint) {
    IREE_ASSERT(instance_ != nullptr);
    instance_->count_.fetch_add(1);
    return instance_->signal_wait_scacquire_(signal, condition, compare_value,
                                             timeout_hint, wait_state_hint);
  }

  static uint32_t HSA_API SignalWaitAny(uint32_t signal_count,
                                        hsa_signal_t* signals,
                                        hsa_signal_condition_t* conditions,
                                        hsa_signal_value_t* values,
                                        uint64_t timeout_hint,
                                        hsa_wait_state_t wait_state_hint,
                                        hsa_signal_value_t* satisfying_value) {
    IREE_ASSERT(instance_ != nullptr);
    instance_->count_.fetch_add(1);
    return instance_->signal_wait_any_(signal_count, signals, conditions,
                                       values, timeout_hint, wait_state_hint,
                                       satisfying_value);
  }

  static ScopedHsaWaitCounter* instance_;
  iree_hal_amdgpu_libhsa_t* libhsa_;
  SignalWaitScacquireFn signal_wait_scacquire_;
  SignalWaitAnyFn signal_wait_any_;
  std::atomic<uint64_t> count_{0};
};

ScopedHsaWaitCounter* ScopedHsaWaitCounter::instance_ = nullptr;
#else
class ScopedHsaWaitCounter {
 public:
  explicit ScopedHsaWaitCounter(iree_hal_amdgpu_libhsa_t* libhsa) {
    (void)libhsa;
  }
  uint64_t count() const { return 0; }
};
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

class ControlledProactorController;

struct ControlledProactor {
  iree_async_proactor_t base;
  ControlledProactorController* controller = nullptr;
  iree_async_proactor_capabilities_t capabilities =
      IREE_ASYNC_PROACTOR_CAPABILITY_NONE;
};

struct ControlledProactorSnapshot {
  uint64_t submit_attempt_count = 0;
  uint64_t submit_count = 0;
  uint64_t submit_failure_count = 0;
  uint64_t cancel_count = 0;
  uint64_t callback_entry_count = 0;
  uint64_t callback_exit_count = 0;
  uint64_t capability_query_count = 0;
  uint64_t file_import_count = 0;
  uint64_t file_destroy_count = 0;
  uint64_t proactor_destroy_count = 0;
  uint64_t calls_after_seal_count = 0;
  uint64_t destroy_with_pending_count = 0;
  uint64_t release_without_pending_count = 0;
  uint64_t finalizer_return_event = 0;
  uint64_t callback_exit_event = 0;
  uint64_t proactor_destroy_event = 0;
  size_t pending_count = 0;
};

class ControlledProactorController {
 public:
  explicit ControlledProactorController(
      iree_async_proactor_capabilities_t capabilities)
      : capabilities_(capabilities) {}

  iree_async_proactor_capabilities_t capabilities() const {
    return capabilities_;
  }

  void Attach(ControlledProactor* proactor) {
    (void)proactor;
    std::lock_guard<std::mutex> lock(mutex_);
    condition_.notify_all();
  }

  void RecordCapabilityQuery() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.capability_query_count;
    if (sealed_) ++snapshot_.calls_after_seal_count;
    condition_.notify_all();
  }

  iree_status_t Submit(iree_async_operation_list_t operations) {
    std::unique_lock<std::mutex> lock(mutex_);
    snapshot_.submit_attempt_count += operations.count;
    if (sealed_) snapshot_.calls_after_seal_count += operations.count;
    if (fail_next_submit_code_ != IREE_STATUS_OK) {
      const iree_status_code_t status_code = fail_next_submit_code_;
      fail_next_submit_code_ = IREE_STATUS_OK;
      snapshot_.submit_failure_count += operations.count;
      condition_.notify_all();
      return iree_status_from_code(status_code);
    }
    for (iree_host_size_t i = 0; i < operations.count; ++i) {
      iree_async_operation_t* operation = operations.values[i];
      for (const PendingOperation& pending : pending_operations_) {
        if (pending.operation == operation) {
          return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                  "operation is already pending");
        }
      }
      iree_async_operation_retain_resources(operation);
      pending_operations_.push_back({operation, Completion::kParked, false});
      ++snapshot_.submit_count;
    }
    condition_.notify_all();
    return iree_ok_status();
  }

  iree_status_t Cancel(iree_async_operation_t* operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.cancel_count;
    if (sealed_) ++snapshot_.calls_after_seal_count;
    if (!iree_any_bit_set(
            capabilities_,
            IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION)) {
      condition_.notify_all();
      return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
    }
    for (PendingOperation& pending : pending_operations_) {
      if (pending.operation == operation) {
        pending.cancel_requested = true;
        condition_.notify_all();
        return iree_ok_status();
      }
    }
    condition_.notify_all();
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }

  iree_status_t Poll(iree_host_size_t* out_completed_count) {
    iree_async_operation_t* operation = nullptr;
    Completion completion = Completion::kParked;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [&] {
        if (wake_requested_) return true;
        for (const PendingOperation& pending : pending_operations_) {
          if (pending.completion != Completion::kParked) return true;
        }
        return false;
      });
      for (auto it = pending_operations_.begin();
           it != pending_operations_.end(); ++it) {
        if (it->completion == Completion::kParked) continue;
        operation = it->operation;
        completion = it->completion;
        pending_operations_.erase(it);
        ++snapshot_.callback_entry_count;
        if (sealed_) ++snapshot_.calls_after_seal_count;
        break;
      }
      if (!operation) {
        wake_requested_ = false;
        if (out_completed_count) *out_completed_count = 0;
        return iree_ok_status();
      }
    }

    CompleteOperation(operation, completion);
    if (out_completed_count) *out_completed_count = 1;
    return iree_ok_status();
  }

  void Wake() {
    std::lock_guard<std::mutex> lock(mutex_);
    wake_requested_ = true;
    condition_.notify_all();
  }

  void RecordFileImport() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.file_import_count;
  }

  void RecordFileDestroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.file_destroy_count;
  }

  void RecordFinalizerReturn() {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.finalizer_return_event = ++lifecycle_event_count_;
    condition_.notify_all();
  }

  void RecordProactorDestroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_operations_.empty()) {
      ++snapshot_.destroy_with_pending_count;
    }
    ++snapshot_.proactor_destroy_count;
    snapshot_.proactor_destroy_event = ++lifecycle_event_count_;
    condition_.notify_all();
  }

  void WaitForSubmitCount(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return snapshot_.submit_count >= count; });
  }

  void WaitForCancelCount(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return snapshot_.cancel_count >= count; });
  }

  void WaitForCallbackEntryCount(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [&] { return snapshot_.callback_entry_count >= count; });
  }

  void WaitForCallbackExitCount(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [&] { return snapshot_.callback_exit_count >= count; });
  }

  void WaitForFileDestroyCount(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [&] { return snapshot_.file_destroy_count >= count; });
  }

  void WaitForProactorDestroyCount(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [&] { return snapshot_.proactor_destroy_count >= count; });
  }

  void WaitForCapabilityQueryAfter(uint64_t previous_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] {
      return snapshot_.capability_query_count > previous_count;
    });
  }

  void FailNextSubmit(iree_status_code_t status_code) {
    std::lock_guard<std::mutex> lock(mutex_);
    IREE_ASSERT(status_code != IREE_STATUS_OK);
    IREE_ASSERT(fail_next_submit_code_ == IREE_STATUS_OK);
    fail_next_submit_code_ = status_code;
  }

  void ReleaseNextCancelled() { ReleaseNext(Completion::kCancelled, true); }

  void ReleaseNextSuccessful() { ReleaseNext(Completion::kSuccess, false); }

  void ReleaseNextOperationFailure() {
    ReleaseNext(Completion::kOperationFailure, true);
  }

  // Releases the Nth currently parked operation. This is used only when a
  // test must complete a later submission before an earlier one without
  // changing the production proactor callback path.
  bool ReleaseSuccessfulAt(size_t parked_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t current_index = 0;
    for (PendingOperation& pending : pending_operations_) {
      if (pending.completion != Completion::kParked) continue;
      if (current_index++ != parked_index) continue;
      pending.completion = Completion::kSuccess;
      condition_.notify_all();
      return true;
    }
    ++snapshot_.release_without_pending_count;
    condition_.notify_all();
    return false;
  }

  // Completes one parked operation synchronously on the calling thread. This
  // models a second proactor poller while another callback is deliberately
  // paused by a test observer.
  bool CompleteNextSuccessfulInline() {
    iree_async_operation_t* operation = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = pending_operations_.begin();
           it != pending_operations_.end(); ++it) {
        if (it->completion != Completion::kParked) continue;
        operation = it->operation;
        pending_operations_.erase(it);
        ++snapshot_.callback_entry_count;
        if (sealed_) ++snapshot_.calls_after_seal_count;
        break;
      }
      if (!operation) {
        ++snapshot_.release_without_pending_count;
        condition_.notify_all();
        return false;
      }
    }
    CompleteOperation(operation, Completion::kSuccess);
    return true;
  }

  void MarkSealed() {
    std::lock_guard<std::mutex> lock(mutex_);
    sealed_ = true;
  }

  ControlledProactorSnapshot Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ControlledProactorSnapshot result = snapshot_;
    result.pending_count = pending_operations_.size();
    return result;
  }

 private:
  enum class Completion {
    kParked,
    kSuccess,
    kCancelled,
    kOperationFailure,
  };

  struct PendingOperation {
    iree_async_operation_t* operation;
    Completion completion;
    bool cancel_requested;
  };

  void CompleteOperation(iree_async_operation_t* operation,
                         Completion completion) {
    if (completion == Completion::kSuccess) {
      if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
        auto* read_operation =
            reinterpret_cast<iree_async_file_read_operation_t*>(operation);
        read_operation->bytes_read = read_operation->buffer.length;
      } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_WRITE) {
        auto* write_operation =
            reinterpret_cast<iree_async_file_write_operation_t*>(operation);
        write_operation->bytes_written = write_operation->buffer.length;
      }
    }

    iree_async_operation_release_resources(operation);
    iree_status_t status = iree_ok_status();
    if (completion == Completion::kCancelled) {
      status = iree_status_from_code(IREE_STATUS_CANCELLED);
    } else if (completion == Completion::kOperationFailure) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
    iree_async_completion_flags_t flags = IREE_ASYNC_COMPLETION_FLAG_NONE;
    status = iree_async_operation_resolve_completion(operation, status, &flags);
    operation->completion_fn(operation->user_data, operation, status, flags);

    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.callback_exit_count;
    snapshot_.callback_exit_event = ++lifecycle_event_count_;
    condition_.notify_all();
  }

  void ReleaseNext(Completion completion, bool require_cancel) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (PendingOperation& pending : pending_operations_) {
      if (pending.completion != Completion::kParked) continue;
      if (require_cancel && !pending.cancel_requested) continue;
      pending.completion = completion;
      condition_.notify_all();
      return;
    }
    ++snapshot_.release_without_pending_count;
    condition_.notify_all();
  }

  const iree_async_proactor_capabilities_t capabilities_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<PendingOperation> pending_operations_;
  ControlledProactorSnapshot snapshot_;
  bool wake_requested_ = false;
  bool sealed_ = false;
  iree_status_code_t fail_next_submit_code_ = IREE_STATUS_OK;
  uint64_t lifecycle_event_count_ = 0;
};

static std::atomic<ControlledProactorController*> g_controller = nullptr;

static ControlledProactor* ControlledProactorCast(
    iree_async_proactor_t* base_proactor) {
  return reinterpret_cast<ControlledProactor*>(base_proactor);
}

static void ControlledProactorDestroy(iree_async_proactor_t* base_proactor) {
  ControlledProactor* proactor = ControlledProactorCast(base_proactor);
  proactor->controller->RecordProactorDestroy();
  delete proactor;
}

static iree_async_proactor_capabilities_t ControlledProactorQueryCapabilities(
    iree_async_proactor_t* base_proactor) {
  ControlledProactor* proactor = ControlledProactorCast(base_proactor);
  proactor->controller->RecordCapabilityQuery();
  return proactor->capabilities;
}

static iree_status_t ControlledProactorSubmit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  return ControlledProactorCast(base_proactor)->controller->Submit(operations);
}

static iree_status_t ControlledProactorPoll(
    iree_async_proactor_t* base_proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  (void)timeout;
  return ControlledProactorCast(base_proactor)
      ->controller->Poll(out_completed_count);
}

static void ControlledProactorWake(iree_async_proactor_t* base_proactor) {
  ControlledProactorCast(base_proactor)->controller->Wake();
}

static iree_status_t ControlledProactorCancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation) {
  return ControlledProactorCast(base_proactor)->controller->Cancel(operation);
}

static iree_status_t ControlledProactorImportFile(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  *out_file = nullptr;
  iree_async_file_t* file = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(base_proactor->allocator,
                                             sizeof(*file),
                                             reinterpret_cast<void**>(&file)));
  std::memset(file, 0, sizeof(*file));
  iree_atomic_ref_count_init(&file->ref_count);
  file->proactor = base_proactor;
  file->primitive = primitive;
  file->fixed_file_index = -1;
  ControlledProactorCast(base_proactor)->controller->RecordFileImport();
  *out_file = file;
  return iree_ok_status();
}

static void ControlledProactorDestroyFile(iree_async_proactor_t* base_proactor,
                                          iree_async_file_t* file) {
  ControlledProactorCast(base_proactor)->controller->RecordFileDestroy();
  iree_async_primitive_close(&file->primitive);
  iree_allocator_free(base_proactor->allocator, file);
}

struct ControlledNotification {
  iree_async_notification_t base;
  iree_notification_t wake_notification;
};

static ControlledNotification* ControlledNotificationCast(
    iree_async_notification_t* notification) {
  return reinterpret_cast<ControlledNotification*>(notification);
}

static iree_status_t ControlledProactorCreateNotification(
    iree_async_proactor_t* base_proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  *out_notification = nullptr;
  ControlledNotification* notification = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(base_proactor->allocator, sizeof(*notification),
                            reinterpret_cast<void**>(&notification)));
  std::memset(notification, 0, sizeof(*notification));
  iree_atomic_ref_count_init(&notification->base.ref_count);
  notification->base.proactor = base_proactor;
  iree_atomic_store(&notification->base.epoch, 0, iree_memory_order_release);
  notification->base.epoch_ptr = &notification->base.epoch;
  iree_atomic_store(&notification->base.observer_count, 0,
                    iree_memory_order_release);
  notification->base.flags = flags;
  notification->base.mode = IREE_ASYNC_NOTIFICATION_MODE_FUTEX;
  iree_notification_initialize(&notification->wake_notification);
  *out_notification = &notification->base;
  return iree_ok_status();
}

static iree_status_t ControlledProactorCreateNotificationShared(
    iree_async_proactor_t* base_proactor,
    const iree_async_notification_shared_options_t* options,
    iree_async_notification_t** out_notification) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->epoch_address);
  IREE_RETURN_IF_ERROR(ControlledProactorCreateNotification(
      base_proactor, IREE_ASYNC_NOTIFICATION_FLAG_SHARED, out_notification));
  (*out_notification)->epoch_ptr = options->epoch_address;
  return iree_ok_status();
}

static void ControlledProactorDestroyNotification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* base_notification) {
  ControlledNotification* notification =
      ControlledNotificationCast(base_notification);
  iree_notification_deinitialize(&notification->wake_notification);
  iree_allocator_free(base_proactor->allocator, notification);
}

static void ControlledProactorSignalNotification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* base_notification, int32_t wake_count) {
  (void)base_proactor;
  ControlledNotification* notification =
      ControlledNotificationCast(base_notification);
  iree_notification_post(&notification->wake_notification, wake_count);
}

struct ControlledNotificationWaitCondition {
  iree_async_notification_t* notification;
  uint32_t wait_token;
};

static bool ControlledNotificationEpochChanged(void* user_data) {
  auto* condition =
      static_cast<ControlledNotificationWaitCondition*>(user_data);
  return iree_atomic_load(condition->notification->epoch_ptr,
                          iree_memory_order_acquire) != condition->wait_token;
}

static bool ControlledProactorWaitNotification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* base_notification, uint32_t wait_token,
    iree_timeout_t timeout) {
  (void)base_proactor;
  ControlledNotification* notification =
      ControlledNotificationCast(base_notification);
  ControlledNotificationWaitCondition condition = {
      /*.notification=*/base_notification,
      /*.wait_token=*/wait_token,
  };
  return iree_notification_await(&notification->wake_notification,
                                 ControlledNotificationEpochChanged, &condition,
                                 timeout);
}

static const iree_async_proactor_vtable_t kControlledProactorVtable = [] {
  iree_async_proactor_vtable_t vtable = {};
  vtable.destroy = ControlledProactorDestroy;
  vtable.query_capabilities = ControlledProactorQueryCapabilities;
  vtable.submit = ControlledProactorSubmit;
  vtable.poll = ControlledProactorPoll;
  vtable.wake = ControlledProactorWake;
  vtable.cancel = ControlledProactorCancel;
  vtable.import_file = ControlledProactorImportFile;
  vtable.destroy_file = ControlledProactorDestroyFile;
  vtable.create_notification = ControlledProactorCreateNotification;
  vtable.create_notification_shared =
      ControlledProactorCreateNotificationShared;
  vtable.destroy_notification = ControlledProactorDestroyNotification;
  vtable.notification_signal = ControlledProactorSignalNotification;
  vtable.notification_wait = ControlledProactorWaitNotification;
  return vtable;
}();

static iree_status_t ControlledProactorCreate(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  *out_proactor = nullptr;
  ControlledProactorController* controller = g_controller.load();
  if (!controller) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "controlled proactor has no controller");
  }
  auto* proactor = new ControlledProactor;
  iree_async_proactor_initialize(&kControlledProactorVtable, options.debug_name,
                                 allocator, &proactor->base);
  proactor->controller = controller;
  proactor->capabilities =
      controller->capabilities() & options.allowed_capabilities;
  controller->Attach(proactor);
  *out_proactor = &proactor->base;
  return iree_ok_status();
}

class ThreadGate {
 public:
  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++ready_count_;
    condition_.notify_all();
    condition_.wait(lock, [&] { return is_open_; });
  }

  void WaitForReadyCount(uint32_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return ready_count_ >= count; });
  }

  void Open() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_open_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t ready_count_ = 0;
  bool is_open_ = false;
};

class ThreadCompletion {
 public:
  void MarkDone() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_done_ = true;
    condition_.notify_all();
  }

  bool IsDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_done_;
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return is_done_; });
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool is_done_ = false;
};

// Records the exact lock-linearized point at which the sole sealer owns queue
// teardown. The production observer runs under submission_mutex, so this
// callback only updates test-local state and never reenters the queue.
class SealOwnerLatch {
 public:
  explicit SealOwnerLatch(iree_hal_amdgpu_host_queue_t* queue)
      : queue_(queue) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value0;
    (void)value1;
    auto* latch = static_cast<SealOwnerLatch*>(user_data);
    if (queue != latch->queue_ ||
        subject != IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SEAL ||
        phase != IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SEAL_OWNER_ACQUIRED) {
      return;
    }
    std::lock_guard<std::mutex> lock(latch->mutex_);
    ++latch->entry_count_;
    latch->condition_.notify_all();
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entry_count_ != 0; });
  }

  uint32_t entry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entry_count_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t entry_count_ = 0;
};

// Immutable test-local fanout for tests that need to observe both their target
// phase and seal-owner readiness through the process-wide guarded observer.
class QueuePhaseObserverGroup {
 public:
  using Observer = iree_hal_amdgpu_host_queue_test_phase_observer_t;

  void Add(Observer observer, void* user_data) {
    IREE_ASSERT(entry_count_ < entries_.size());
    entries_[entry_count_++] = {observer, user_data};
  }

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    auto* group = static_cast<QueuePhaseObserverGroup*>(user_data);
    for (size_t i = 0; i < group->entry_count_; ++i) {
      group->entries_[i].observer(queue, subject, phase, value0, value1,
                                  group->entries_[i].user_data);
    }
  }

 private:
  struct Entry {
    Observer observer = nullptr;
    void* user_data = nullptr;
  };
  std::array<Entry, 3> entries_ = {};
  size_t entry_count_ = 0;
};

class QueuePhaseLatch {
 public:
  QueuePhaseLatch(iree_hal_amdgpu_host_queue_t* queue,
                  iree_hal_amdgpu_host_queue_test_subject_t subject,
                  iree_hal_amdgpu_host_queue_test_phase_t phase,
                  uint32_t blocked_occurrence = 1, uint64_t minimum_value0 = 0,
                  const std::atomic<uint32_t>* prerequisite_count = nullptr)
      : queue_(queue),
        subject_(subject),
        phase_(phase),
        blocked_occurrence_(blocked_occurrence),
        minimum_value0_(minimum_value0),
        prerequisite_count_(prerequisite_count) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    auto* latch = static_cast<QueuePhaseLatch*>(user_data);
    if (queue != latch->queue_ || subject != latch->subject_ ||
        phase != latch->phase_ || value0 < latch->minimum_value0_ ||
        (latch->prerequisite_count_ &&
         latch->prerequisite_count_->load(std::memory_order_acquire) == 0)) {
      return;
    }
    std::unique_lock<std::mutex> lock(latch->mutex_);
    ++latch->occurrence_count_;
    latch->value0_ = value0;
    latch->value1_ = value1;
    latch->condition_.notify_all();
    if (latch->occurrence_count_ == latch->blocked_occurrence_) {
      latch->condition_.wait(lock, [&] { return latch->released_; });
    }
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [&] { return occurrence_count_ >= blocked_occurrence_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

  uint32_t occurrence_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return occurrence_count_;
  }

  uint64_t value0() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value0_;
  }

  uint64_t value1() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value1_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  const iree_hal_amdgpu_host_queue_test_subject_t subject_;
  const iree_hal_amdgpu_host_queue_test_phase_t phase_;
  const uint32_t blocked_occurrence_;
  const uint64_t minimum_value0_;
  const std::atomic<uint32_t>* const prerequisite_count_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t occurrence_count_ = 0;
  uint64_t value0_ = 0;
  uint64_t value1_ = 0;
  bool released_ = false;
};

// Blocks one exact publisher admission boundary. Unlike QueuePhaseLatch this
// matches value0 exactly because several publisher paths share the guarded
// subject/phase and may execute earlier in the same operation.
class PublisherSubmissionLatch {
 public:
  PublisherSubmissionLatch(
      iree_hal_amdgpu_host_queue_t* queue,
      iree_hal_amdgpu_host_queue_test_publisher_path_t path)
      : queue_(queue), path_(path) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value1;
    auto* latch = static_cast<PublisherSubmissionLatch*>(user_data);
    if (queue != latch->queue_ ||
        subject !=
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PUBLISHER_SUBMISSION_REVALIDATION ||
        phase !=
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_PUBLISHER_SUBMISSION_LOCK ||
        value0 != static_cast<uint64_t>(latch->path_)) {
      return;
    }
    std::unique_lock<std::mutex> lock(latch->mutex_);
    ++latch->occurrence_count_;
    latch->condition_.notify_all();
    if (latch->occurrence_count_ == 1) {
      latch->condition_.wait(lock, [&] { return latch->released_; });
    }
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return occurrence_count_ != 0; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

  uint32_t occurrence_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return occurrence_count_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  const iree_hal_amdgpu_host_queue_test_publisher_path_t path_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t occurrence_count_ = 0;
  bool released_ = false;
};

// Records one exact queue phase without blocking the producer. This is used
// for transitions that may occur synchronously inside the operation being
// observed and whose postcondition remains stable until a later test action.
class QueuePhaseSignal {
 public:
  QueuePhaseSignal(iree_hal_amdgpu_host_queue_t* queue,
                   iree_hal_amdgpu_host_queue_test_subject_t subject,
                   iree_hal_amdgpu_host_queue_test_phase_t phase)
      : queue_(queue), subject_(subject), phase_(phase) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    auto* signal = static_cast<QueuePhaseSignal*>(user_data);
    if (queue != signal->queue_ || subject != signal->subject_ ||
        phase != signal->phase_) {
      return;
    }
    std::lock_guard<std::mutex> lock(signal->mutex_);
    ++signal->occurrence_count_;
    signal->value0_ = value0;
    signal->value1_ = value1;
    signal->condition_.notify_all();
  }

  void WaitUntilObserved() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return occurrence_count_ != 0; });
  }

  uint32_t occurrence_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return occurrence_count_;
  }

  uint64_t value0() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value0_;
  }

  uint64_t value1() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value1_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  const iree_hal_amdgpu_host_queue_test_subject_t subject_;
  const iree_hal_amdgpu_host_queue_test_phase_t phase_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t occurrence_count_ = 0;
  uint64_t value0_ = 0;
  uint64_t value1_ = 0;
};

// Gates the detached and exact-idle boundaries of one post-drain batch. The
// completion service and explicit waiters legitimately race for ownership, so
// tests synchronize on production runner phases instead of assuming a
// particular caller executes the batch.
class PostDrainBoundaryLatch {
 public:
  explicit PostDrainBoundaryLatch(iree_hal_amdgpu_host_queue_t* queue)
      : queue_(queue) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value0;
    (void)value1;
    auto* latch = static_cast<PostDrainBoundaryLatch*>(user_data);
    if (queue != latch->queue_ ||
        subject != IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER) {
      return;
    }
    if (phase ==
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_BATCH_DETACHED_BEFORE_CALLBACK) {
      latch->detached_count_.fetch_add(1);
      latch->detached_gate_.ArriveAndWait();
    } else if (
        phase ==
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_IDLE_BEFORE_WAKE) {
      latch->idle_before_wake_count_.fetch_add(1);
      latch->idle_before_wake_gate_.ArriveAndWait();
    }
  }

  void WaitForDetached() { detached_gate_.WaitForReadyCount(1); }
  void ReleaseDetached() { detached_gate_.Open(); }
  void WaitForIdleBeforeWake() { idle_before_wake_gate_.WaitForReadyCount(1); }
  void ReleaseIdleBeforeWake() { idle_before_wake_gate_.Open(); }

  uint32_t detached_count() const { return detached_count_.load(); }
  uint32_t idle_before_wake_count() const {
    return idle_before_wake_count_.load();
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  ThreadGate detached_gate_;
  ThreadGate idle_before_wake_gate_;
  std::atomic<uint32_t> detached_count_{0};
  std::atomic<uint32_t> idle_before_wake_count_{0};
};

// Observes the complete admission/epilogue lifecycle while blocking one
// selected normal-runner boundary. The thread stopped at that boundary becomes
// the target, allowing the test to distinguish its later phases from the
// completion service and the terminal sealer runner.
class CompletionRunnerLifecycleLatch {
 public:
  CompletionRunnerLifecycleLatch(
      iree_hal_amdgpu_host_queue_t* queue,
      iree_hal_amdgpu_host_queue_test_subject_t blocked_subject,
      iree_hal_amdgpu_host_queue_test_phase_t blocked_phase,
      int blocked_allow_closed)
      : queue_(queue),
        blocked_subject_(blocked_subject),
        blocked_phase_(blocked_phase),
        blocked_allow_closed_(blocked_allow_closed) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value1;
    auto* latch = static_cast<CompletionRunnerLifecycleLatch*>(user_data);
    if (queue != latch->queue_) return;

    std::unique_lock<std::mutex> lock(latch->mutex_);
    const bool allow_closed_matches =
        latch->blocked_allow_closed_ < 0 ||
        value0 == static_cast<uint64_t>(latch->blocked_allow_closed_);
    const bool matches_block = subject == latch->blocked_subject_ &&
                               phase == latch->blocked_phase_ &&
                               allow_closed_matches;
    bool must_block = false;
    if (matches_block) {
      ++latch->blocked_occurrence_count_;
      if (latch->blocked_occurrence_count_ == 1) {
        latch->blocked_thread_ = std::this_thread::get_id();
        latch->entered_ = true;
        if (subject == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SAFE_EPILOGUE &&
            phase ==
                IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_EPILOGUE_TOKEN_DROP) {
          latch->target_after_runner_release_count_ = static_cast<uint32_t>(
              std::count(latch->after_runner_release_threads_.begin(),
                         latch->after_runner_release_threads_.end(),
                         latch->blocked_thread_));
        }
        must_block = true;
      }
    }

    const bool is_target_thread =
        latch->entered_ && std::this_thread::get_id() == latch->blocked_thread_;
    if (subject == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER) {
      if (phase == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_RUNNER_CLAIM) {
        if (value0) {
          ++latch->terminal_before_claim_count_;
        } else {
          ++latch->normal_before_claim_count_;
        }
      } else if (phase ==
                 IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_CLAIM) {
        if (value0) {
          ++latch->terminal_after_claim_count_;
        } else {
          ++latch->normal_after_claim_count_;
        }
        if (is_target_thread) ++latch->target_after_claim_count_;
      }
    } else if (subject ==
               IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SAFE_EPILOGUE) {
      if (phase ==
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_RELEASE_BEFORE_EPILOGUE) {
        ++latch->after_runner_release_count_;
        latch->after_runner_release_threads_.push_back(
            std::this_thread::get_id());
        if (is_target_thread) ++latch->target_after_runner_release_count_;
      } else if (
          phase ==
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_EPILOGUE_TOKEN_DROP) {
        ++latch->before_token_drop_count_;
        if (is_target_thread) ++latch->target_before_token_drop_count_;
      }
    }
    latch->condition_.notify_all();
    if (must_block) {
      latch->condition_.wait(lock, [&] { return latch->released_; });
    }
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

  uint32_t terminal_before_claim_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return terminal_before_claim_count_;
  }

  uint32_t terminal_after_claim_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return terminal_after_claim_count_;
  }

  uint32_t target_after_claim_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return target_after_claim_count_;
  }

  uint32_t target_after_runner_release_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return target_after_runner_release_count_;
  }

  uint32_t target_before_token_drop_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return target_before_token_drop_count_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  const iree_hal_amdgpu_host_queue_test_subject_t blocked_subject_;
  const iree_hal_amdgpu_host_queue_test_phase_t blocked_phase_;
  const int blocked_allow_closed_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::thread::id blocked_thread_;
  uint32_t blocked_occurrence_count_ = 0;
  uint32_t normal_before_claim_count_ = 0;
  uint32_t terminal_before_claim_count_ = 0;
  uint32_t normal_after_claim_count_ = 0;
  uint32_t terminal_after_claim_count_ = 0;
  uint32_t after_runner_release_count_ = 0;
  uint32_t before_token_drop_count_ = 0;
  std::vector<std::thread::id> after_runner_release_threads_;
  uint32_t target_after_claim_count_ = 0;
  uint32_t target_after_runner_release_count_ = 0;
  uint32_t target_before_token_drop_count_ = 0;
  bool entered_ = false;
  bool released_ = false;
};

// Holds an admitted publisher at the final publication point while recording
// when terminal-failure admission reaches its pre-lock boundary.
class SubmissionPublicationRaceLatch {
 public:
  explicit SubmissionPublicationRaceLatch(iree_hal_amdgpu_host_queue_t* queue)
      : queue_(queue) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value1;
    auto* latch = static_cast<SubmissionPublicationRaceLatch*>(user_data);
    if (queue != latch->queue_) return;
    std::unique_lock<std::mutex> lock(latch->mutex_);
    if (subject ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SUBMISSION_PUBLICATION &&
        phase ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_SUBMISSION_PUBLICATION) {
      ++latch->publication_count_;
      latch->publication_epoch_ = value0;
      latch->condition_.notify_all();
      if (latch->publication_count_ == 1) {
        latch->condition_.wait(lock, [&] { return latch->released_; });
      }
    } else if (
        subject == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION &&
        phase ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_FAILURE_ADMISSION_LOCK) {
      ++latch->failure_attempt_count_;
      latch->condition_.notify_all();
    }
  }

  void WaitForPublication() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return publication_count_ != 0; });
  }

  void WaitForFailureAttempt() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return failure_attempt_count_ != 0; });
  }

  void ReleasePublication() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

  uint32_t publication_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return publication_count_;
  }

  uint64_t publication_epoch() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return publication_epoch_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t publication_count_ = 0;
  uint32_t failure_attempt_count_ = 0;
  uint64_t publication_epoch_ = 0;
  bool released_ = false;
};

// Pauses the capacity-only pending-op handoff after COMPLETING is visible but
// before its post-drain retry is queued, and records a concurrent terminal
// failure's admission attempt. Correct production code invokes the paused
// phase while still owning submission_mutex, so failure cannot close admission
// or run its cancellation pass until the retry is visible.
class CapacityHandoffRaceLatch {
 public:
  explicit CapacityHandoffRaceLatch(iree_hal_amdgpu_host_queue_t* queue)
      : queue_(queue) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value1;
    auto* latch = static_cast<CapacityHandoffRaceLatch*>(user_data);
    if (queue != latch->queue_) return;
    if (subject == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER &&
        phase == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_CLAIM &&
        value0 == 0) {
      std::unique_lock<std::mutex> lock(latch->mutex_);
      ++latch->runner_claim_count_;
      latch->condition_.notify_all();
      if (latch->runner_claim_count_ == 1) {
        latch->condition_.wait(lock,
                               [&] { return latch->runner_claim_released_; });
      }
      return;
    }
    if (subject !=
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_START_HANDOFF &&
        subject != IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION) {
      return;
    }
    if (subject ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_START_HANDOFF &&
        phase ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_CAPACITY_COMPLETING_BEFORE_ENQUEUE) {
      std::unique_lock<std::mutex> lock(latch->mutex_);
      ++latch->handoff_count_;
      latch->condition_.notify_all();
      if (latch->handoff_count_ == 1) {
        latch->condition_.wait(lock, [&] { return latch->released_; });
      }
    } else if (
        subject == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION &&
        phase ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_FAILURE_ADMISSION_LOCK) {
      std::unique_lock<std::mutex> lock(latch->mutex_);
      ++latch->failure_attempt_count_;
      latch->condition_.notify_all();
    }
  }

  void WaitForHandoff() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return handoff_count_ != 0; });
  }

  void WaitForRunnerClaim() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return runner_claim_count_ != 0; });
  }

  void WaitForFailureAttempt() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return failure_attempt_count_ != 0; });
  }

  void ReleaseHandoff() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

  void ReleaseRunnerClaim() {
    std::lock_guard<std::mutex> lock(mutex_);
    runner_claim_released_ = true;
    condition_.notify_all();
  }

  uint32_t handoff_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return handoff_count_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t handoff_count_ = 0;
  uint32_t failure_attempt_count_ = 0;
  uint32_t runner_claim_count_ = 0;
  bool released_ = false;
  bool runner_claim_released_ = false;
};

class PublisherTailLatch {
 public:
  PublisherTailLatch(iree_hal_amdgpu_host_queue_t* queue,
                     iree_hal_amdgpu_host_queue_test_subject_t subject)
      : queue_(queue), subject_(subject) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    auto* latch = static_cast<PublisherTailLatch*>(user_data);
    if (queue != latch->queue_ || subject != latch->subject_) return;
    std::unique_lock<std::mutex> lock(latch->mutex_);
    bool* released = nullptr;
    if (phase == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE ||
        phase ==
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_ZERO_BEFORE_WAKE) {
      ++latch->zero_before_wake_count_;
      released = &latch->zero_before_wake_released_;
    } else if (phase == IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT ||
               phase ==
                   IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SUBMIT_TAIL_LEFT) {
      ++latch->tail_left_count_;
      latch->tail_left_value0_ = value0;
      latch->tail_left_value1_ = value1;
      released = &latch->tail_left_released_;
    } else {
      return;
    }
    latch->condition_.notify_all();
    latch->condition_.wait(lock, [&] { return *released; });
  }

  void WaitForZeroBeforeWake() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return zero_before_wake_count_ != 0; });
  }

  void ReleaseZeroBeforeWake() {
    std::lock_guard<std::mutex> lock(mutex_);
    zero_before_wake_released_ = true;
    condition_.notify_all();
  }

  void WaitForTailLeft() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return tail_left_count_ != 0; });
  }

  void ReleaseTailLeft() {
    std::lock_guard<std::mutex> lock(mutex_);
    tail_left_released_ = true;
    condition_.notify_all();
  }

  uint32_t zero_before_wake_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return zero_before_wake_count_;
  }

  uint32_t tail_left_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tail_left_count_;
  }

  uint64_t tail_left_value0() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tail_left_value0_;
  }

  uint64_t tail_left_value1() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tail_left_value1_;
  }

 private:
  iree_hal_amdgpu_host_queue_t* const queue_;
  const iree_hal_amdgpu_host_queue_test_subject_t subject_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t zero_before_wake_count_ = 0;
  uint32_t tail_left_count_ = 0;
  uint64_t tail_left_value0_ = 0;
  uint64_t tail_left_value1_ = 0;
  bool zero_before_wake_released_ = false;
  bool tail_left_released_ = false;
};

// Serializes the complete staged proactor-callback handoff into queue-owned
// safe execution. Each boundary is independently released so a sealer racing
// the poll callback cannot conflate installed ownership, callback return,
// safe-action execution, or the final lock-linearized tail wake.
class StagedSafeActionLifecycleLatch {
 public:
  enum class Boundary : uint32_t {
    kHandoffInstalled = 0,
    kSafeEpilogueBegin = 1,
    kSafeEpilogueEnd = 2,
    kTailZeroBeforeWake = 3,
    kTailLeft = 4,
    kCount = 5,
  };

  explicit StagedSafeActionLifecycleLatch(iree_hal_amdgpu_host_queue_t* queue)
      : queue_(queue) {}

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value0;
    (void)value1;
    auto* latch = static_cast<StagedSafeActionLifecycleLatch*>(user_data);
    if (queue != latch->queue_ ||
        subject !=
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK) {
      return;
    }

    Boundary boundary;
    switch (phase) {
      case IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_HANDOFF_INSTALLED:
        boundary = Boundary::kHandoffInstalled;
        break;
      case IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_BEGIN:
        boundary = Boundary::kSafeEpilogueBegin;
        break;
      case IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_SAFE_EPILOGUE_END:
        boundary = Boundary::kSafeEpilogueEnd;
        break;
      case IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE:
        boundary = Boundary::kTailZeroBeforeWake;
        break;
      case IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT:
        boundary = Boundary::kTailLeft;
        break;
      default:
        return;
    }

    const size_t index = static_cast<size_t>(boundary);
    std::unique_lock<std::mutex> lock(latch->mutex_);
    ++latch->entry_counts_[index];
    latch->condition_.notify_all();
    latch->condition_.wait(lock, [&] { return latch->released_[index]; });
  }

  void WaitFor(Boundary boundary) {
    const size_t index = static_cast<size_t>(boundary);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entry_counts_[index] != 0; });
  }

  void Release(Boundary boundary) {
    const size_t index = static_cast<size_t>(boundary);
    std::lock_guard<std::mutex> lock(mutex_);
    released_[index] = true;
    condition_.notify_all();
  }

  uint32_t entry_count(Boundary boundary) const {
    const size_t index = static_cast<size_t>(boundary);
    std::lock_guard<std::mutex> lock(mutex_);
    return entry_counts_[index];
  }

 private:
  static constexpr size_t kBoundaryCount =
      static_cast<size_t>(Boundary::kCount);
  iree_hal_amdgpu_host_queue_t* const queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::array<uint32_t, kBoundaryCount> entry_counts_ = {};
  std::array<bool, kBoundaryCount> released_ = {};
};

struct SlowFeedbackSink {
  struct Metadata {
    iree_hal_device_event_type_t type = IREE_HAL_DEVICE_EVENT_TYPE_NONE;
    uint64_t sequence = 0;
    iree_hal_device_t* device = nullptr;
    uint64_t executable_id = 0;
    iree_host_size_t payload_length = 0;
    iree_host_size_t implementation_payload_length = 0;
    uint32_t packet_record_length = 0;
    uint64_t packet_source_context = 0;
  };

  static Metadata Capture(const iree_hal_device_event_t* event) {
    Metadata metadata = {
        .type = event->type,
        .sequence = event->sequence,
        .device = event->source.device,
        .executable_id = event->source.executable_id,
        .payload_length = event->payload.data_length,
        .implementation_payload_length =
            event->implementation_payload.data_length,
    };
    if (event->implementation_payload.data_length >=
        sizeof(iree_hal_amdgpu_feedback_packet_t)) {
      const auto* packet =
          reinterpret_cast<const iree_hal_amdgpu_feedback_packet_t*>(
              event->implementation_payload.data);
      metadata.packet_record_length = packet->record_length;
      metadata.packet_source_context = packet->source_context;
    }
    return metadata;
  }

  static bool Equal(const Metadata& lhs, const Metadata& rhs) {
    return lhs.type == rhs.type && lhs.sequence == rhs.sequence &&
           lhs.device == rhs.device && lhs.executable_id == rhs.executable_id &&
           lhs.payload_length == rhs.payload_length &&
           lhs.implementation_payload_length ==
               rhs.implementation_payload_length &&
           lhs.packet_record_length == rhs.packet_record_length &&
           lhs.packet_source_context == rhs.packet_source_context;
  }

  static void Publish(void* user_data, const iree_hal_device_event_t* event) {
    auto* state = static_cast<SlowFeedbackSink*>(user_data);
    state->entry_count.fetch_add(1);
    if (state->drain_mutex) {
      const bool acquired = iree_slim_mutex_try_lock(state->drain_mutex);
      state->drain_lock_was_available.store(acquired);
      if (acquired) iree_slim_mutex_unlock(state->drain_mutex);
    }
    const Metadata before = Capture(event);
    state->gate.ArriveAndWait();
    state->metadata_stable.store(Equal(before, Capture(event)));
    state->metadata = before;
    state->exit_count.fetch_add(1);
    state->exited.MarkDone();
  }

  iree_hal_device_event_sink_t sink() {
    return iree_hal_device_event_sink_t{
        /*.fn=*/Publish,
        /*.user_data=*/this,
    };
  }

  iree_slim_mutex_t* drain_mutex = nullptr;
  ThreadGate gate;
  ThreadCompletion exited;
  Metadata metadata;
  std::atomic<uint32_t> entry_count{0};
  std::atomic<uint32_t> exit_count{0};
  std::atomic<bool> drain_lock_was_available{false};
  std::atomic<bool> metadata_stable{false};
};

class FeedbackPhaseRecorder {
 public:
  static void Observe(iree_host_size_t physical_device_ordinal,
                      iree_hal_amdgpu_feedback_test_phase_t phase,
                      uint64_t source_identity, uint64_t sequence,
                      void* user_data) {
    auto* recorder = static_cast<FeedbackPhaseRecorder*>(user_data);
    recorder->physical_device_ordinal.store(physical_device_ordinal);
    ThreadGate* gate = nullptr;
    switch (phase) {
      case IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_BATCH_INSTALLED:
        recorder->installed_source_identity.store(source_identity);
        recorder->installed_sequence.store(sequence);
        recorder->batch_installed_count.fetch_add(1);
        gate = recorder->batch_installed_gate;
        break;
      case IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_TARGET_CLOSED:
        recorder->closed_source_identity.store(source_identity);
        recorder->closed_sequence.store(sequence);
        recorder->target_closed_count.fetch_add(1);
        break;
      case IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_CLAIM_ENTERED:
        recorder->claimed_source_identity.store(source_identity);
        recorder->claimed_sequence.store(sequence);
        recorder->claim_entered_count.fetch_add(1);
        break;
      case IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_CLAIM_RETURNED:
        recorder->claim_returned_count.fetch_add(1);
        break;
      case IREE_HAL_AMDGPU_FEEDBACK_TEST_PHASE_BATCH_RETIRED:
        recorder->batch_retired_count.fetch_add(1);
        break;
    }
    {
      std::lock_guard<std::mutex> lock(recorder->mutex);
      recorder->condition.notify_all();
    }
    if (gate) gate->ArriveAndWait();
  }

  void WaitForPhaseCount(const std::atomic<uint32_t>& phase_count,
                         uint32_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return phase_count.load() >= expected_count; });
  }

  std::atomic<iree_host_size_t> physical_device_ordinal{IREE_HOST_SIZE_MAX};
  std::atomic<uint32_t> batch_installed_count{0};
  std::atomic<uint32_t> target_closed_count{0};
  std::atomic<uint32_t> claim_entered_count{0};
  std::atomic<uint32_t> claim_returned_count{0};
  std::atomic<uint32_t> batch_retired_count{0};
  std::atomic<uint64_t> installed_source_identity{0};
  std::atomic<uint64_t> installed_sequence{0};
  std::atomic<uint64_t> closed_source_identity{0};
  std::atomic<uint64_t> closed_sequence{0};
  std::atomic<uint64_t> claimed_source_identity{0};
  std::atomic<uint64_t> claimed_sequence{0};
  ThreadGate* batch_installed_gate = nullptr;
  std::mutex mutex;
  std::condition_variable condition;
};

class ScopedFeedbackPhaseObserver {
 public:
  explicit ScopedFeedbackPhaseObserver(FeedbackPhaseRecorder* recorder) {
    iree_hal_amdgpu_feedback_test_set_phase_observer(
        FeedbackPhaseRecorder::Observe, recorder);
  }
  ~ScopedFeedbackPhaseObserver() {
    iree_hal_amdgpu_feedback_test_set_phase_observer(nullptr, nullptr);
  }

  ScopedFeedbackPhaseObserver(const ScopedFeedbackPhaseObserver&) = delete;
  ScopedFeedbackPhaseObserver& operator=(const ScopedFeedbackPhaseObserver&) =
      delete;
};

static bool FeedbackIsPoisoned(void* user_data) {
  auto* device_state =
      static_cast<iree_hal_amdgpu_feedback_device_state_t*>(user_data);
  iree_slim_mutex_lock(&device_state->drain_mutex);
  const bool is_poisoned = device_state->is_poisoned;
  iree_slim_mutex_unlock(&device_state->drain_mutex);
  return is_poisoned;
}

static void WaitForFeedbackPoisoned(
    iree_hal_amdgpu_feedback_device_state_t* device_state) {
  iree_notification_await(&device_state->runner_notification,
                          FeedbackIsPoisoned, device_state,
                          iree_infinite_timeout());
}

static bool FeedbackLedgerIsEmpty(void* user_data) {
  auto* device_state =
      static_cast<iree_hal_amdgpu_feedback_device_state_t*>(user_data);
  iree_slim_mutex_lock(&device_state->drain_mutex);
  const bool is_empty =
      device_state->source_hold_head == nullptr && !device_state->runner_active;
  iree_slim_mutex_unlock(&device_state->drain_mutex);
  return is_empty;
}

static void WaitForFeedbackLedgerEmpty(
    iree_hal_amdgpu_feedback_device_state_t* device_state) {
  iree_notification_await(&device_state->runner_notification,
                          FeedbackLedgerIsEmpty, device_state,
                          iree_infinite_timeout());
}

#if IREE_FILE_IO_ENABLE
struct PollRunnerFileFinalizer {
  static void Release(void* user_data,
                      iree_io_file_handle_primitive_t primitive) {
    (void)primitive;
    auto* state = static_cast<PollRunnerFileFinalizer*>(user_data);
    state->entry_count.fetch_add(1);
    if (state->gate) state->gate->ArriveAndWait();
    iree_io_file_handle_t* source_handle = state->source_handle;
    state->source_handle = nullptr;
    iree_io_file_handle_release(source_handle);
    iree_hal_queue_t* queue = state->queue;
    state->queue = nullptr;
    if (queue) iree_hal_queue_release(queue);
    state->return_count.fetch_add(1);
    if (state->controller) state->controller->RecordFinalizerReturn();
    state->returned.MarkDone();
  }

  iree_io_file_handle_t* source_handle = nullptr;
  iree_hal_queue_t* queue = nullptr;
  ControlledProactorController* controller = nullptr;
  ThreadGate* gate = nullptr;
  std::atomic<uint32_t> entry_count{0};
  std::atomic<uint32_t> return_count{0};
  ThreadCompletion returned;
};
#endif  // IREE_FILE_IO_ENABLE

class WaiterClaimLatch {
 public:
  static void Observe(void* user_data) {
    auto* latch = static_cast<WaiterClaimLatch*>(user_data);
    std::unique_lock<std::mutex> lock(latch->mutex_);
    latch->entered_ = true;
    latch->condition_.notify_all();
    latch->condition_.wait(lock, [&] { return latch->released_; });
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool released_ = false;
};

class ChunkSlotReleaseLatch {
 public:
  explicit ChunkSlotReleaseLatch(size_t blocked_capture_count = 1)
      : blocked_capture_count_(blocked_capture_count) {}

  static void Observe(void* user_data, uint32_t captured_slot_ordinal) {
    auto* latch = static_cast<ChunkSlotReleaseLatch*>(user_data);
    std::unique_lock<std::mutex> lock(latch->mutex_);
    latch->captured_ordinals_.push_back(captured_slot_ordinal);
    latch->condition_.notify_all();
    if (latch->captured_ordinals_.size() == latch->blocked_capture_count_) {
      latch->condition_.wait(lock, [&] { return latch->blocked_released_; });
    }
  }

  void WaitForCapturedCount(size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return captured_ordinals_.size() >= count; });
  }

  std::vector<uint32_t> CapturedOrdinals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return captured_ordinals_;
  }

  void ReleaseBlocked() {
    std::lock_guard<std::mutex> lock(mutex_);
    blocked_released_ = true;
    condition_.notify_all();
  }

 private:
  const size_t blocked_capture_count_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<uint32_t> captured_ordinals_;
  bool blocked_released_ = false;
};

struct PostDrainActionCounter {
  static void Run(void* user_data) {
    auto* state = static_cast<PostDrainActionCounter*>(user_data);
    if (state->gate) state->gate->ArriveAndWait();
    state->call_count.fetch_add(1);
  }

  iree_hal_amdgpu_host_queue_post_drain_action_t action = {};
  ThreadGate* gate = nullptr;
  std::atomic<uint32_t> call_count{0};
};

// Requeues the same intrusive action once. The first callback models a
// capacity retry that cannot make progress until a later notification-drain
// turn; the second callback proves the later service pass consumes it.
struct RequeuedPostDrainAction {
  static void Run(void* user_data) {
    auto* state = static_cast<RequeuedPostDrainAction*>(user_data);
    const uint32_t call_count = state->call_count.fetch_add(1) + 1;
    if (call_count == 1) {
      iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
          state->queue, &state->action, RequeuedPostDrainAction::Run, state);
    } else {
      IREE_ASSERT(call_count == 2);
      state->completion->MarkDone();
    }
  }

  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  iree_hal_amdgpu_host_queue_post_drain_action_t action = {};
  ThreadCompletion* completion = nullptr;
  std::atomic<uint32_t> call_count{0};
};

// Requeues itself from the terminal completion epilogue, then clones the
// queue failure from the sealer-owned fixed-point pass. This reaches the exact
// window in which an early error-slot exchange loses sticky-error precedence.
struct RequeuedShutdownStatusProbe {
  static void Run(void* user_data) {
    auto* state = static_cast<RequeuedShutdownStatusProbe*>(user_data);
    const uint32_t call_count = state->call_count.fetch_add(1) + 1;
    if (call_count == 1) {
      iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
          state->queue, &state->action, RequeuedShutdownStatusProbe::Run,
          state);
      return;
    }
    IREE_ASSERT(call_count == 2);
    state->observed_status =
        iree_hal_amdgpu_host_queue_clone_error_status(state->queue);
  }

  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  iree_hal_amdgpu_host_queue_post_drain_action_t action = {};
  iree_status_t observed_status = iree_ok_status();
  std::atomic<uint32_t> call_count{0};
};

struct ReentrantHostCallState {
  static iree_status_t Call(void* user_data, const uint64_t args[4],
                            iree_hal_host_call_context_t* context) {
    (void)args;
    (void)context;
    auto* state = static_cast<ReentrantHostCallState*>(user_data);
    state->call_count.fetch_add(1);
    iree_status_t status = iree_hal_semaphore_wait(
        state->later_semaphore, state->later_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    state->status_code.store(iree_status_code(status));
    return status;
  }

  iree_hal_semaphore_t* later_semaphore = nullptr;
  uint64_t later_value = 0;
  std::atomic<uint32_t> call_count{0};
  std::atomic<iree_status_code_t> status_code{IREE_STATUS_UNKNOWN};
};

// Drops one caller-owned queue or device edge from inside a host-call
// completion callback, then pauses immediately before returning. External
// counters and the gate remain valid even when the release eventually frees
// the queue, device, and completion-thread wrapper.
struct CompletionSelfReleaseHostCallState {
  static iree_status_t Call(void* user_data, const uint64_t args[4],
                            iree_hal_host_call_context_t* context) {
    (void)args;
    (void)context;
    auto* state = static_cast<CompletionSelfReleaseHostCallState*>(user_data);
    state->call_count.fetch_add(1);
    if (state->queue) {
      iree_hal_queue_t* queue = state->queue;
      state->queue = nullptr;
      iree_hal_queue_release(queue);
    }
    if (state->device) {
      iree_hal_device_t* device = state->device;
      state->device = nullptr;
      iree_hal_device_release(device);
    }
    state->release_return_count.fetch_add(1);
    state->after_release_gate.ArriveAndWait();
    state->callback_return_count.fetch_add(1);
    return iree_ok_status();
  }

  iree_hal_queue_t* queue = nullptr;
  iree_hal_device_t* device = nullptr;
  ThreadGate after_release_gate;
  std::atomic<uint32_t> call_count{0};
  std::atomic<uint32_t> release_return_count{0};
  std::atomic<uint32_t> callback_return_count{0};
};

struct QueueReentrantResourceObservations {
  std::atomic<uint32_t> destroy_count{0};
  std::atomic<bool> completion_lock_was_available{false};
  std::atomic<bool> submission_lock_was_available{false};
  ThreadCompletion destroyed;
};

struct QueueReentrantResource {
  iree_hal_resource_t resource;
  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  QueueReentrantResourceObservations* observations = nullptr;
};

static void DestroyQueueReentrantResource(iree_hal_resource_t* base_resource) {
  auto* state = reinterpret_cast<QueueReentrantResource*>(base_resource);
  const bool acquired =
      iree_slim_mutex_try_lock(&state->queue->locks.completion_drain_mutex);
  state->observations->completion_lock_was_available.store(acquired);
  if (acquired) {
    iree_slim_mutex_unlock(&state->queue->locks.completion_drain_mutex);
  }
  const bool submission_acquired =
      iree_slim_mutex_try_lock(&state->queue->locks.submission_mutex);
  state->observations->submission_lock_was_available.store(submission_acquired);
  if (submission_acquired) {
    iree_slim_mutex_unlock(&state->queue->locks.submission_mutex);
  }

  // Exercise recursive waiter-drain and queue-ref release from the terminal
  // resource callback. Both would deadlock or use freed state if invoked under
  // the outer completion drain or after queue finalization.
  iree_hal_amdgpu_host_queue_drain_completions_for_waiter(state->queue);
  iree_hal_queue_release(&state->queue->base);
  state->observations->destroy_count.fetch_add(1);
  state->observations->destroyed.MarkDone();
  delete state;
}

static const iree_hal_resource_vtable_t kQueueReentrantResourceVtable = {
    /*.destroy=*/DestroyQueueReentrantResource,
};

static QueueReentrantResource* CreateQueueReentrantResource(
    iree_hal_amdgpu_host_queue_t* queue,
    QueueReentrantResourceObservations* observations) {
  auto* state = new QueueReentrantResource;
  iree_hal_resource_initialize(&kQueueReentrantResourceVtable,
                               &state->resource);
  state->queue = queue;
  state->observations = observations;
  iree_hal_queue_retain(&queue->base);
  return state;
}

struct RingActionObservations {
  static void Run(iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
                  const iree_status_t status) {
    (void)entry;
    auto* state = static_cast<RingActionObservations*>(user_data);
    const bool acquired =
        iree_slim_mutex_try_lock(&state->queue->locks.completion_drain_mutex);
    state->completion_lock_was_available.store(acquired);
    if (acquired) {
      iree_slim_mutex_unlock(&state->queue->locks.completion_drain_mutex);
    }
    if (state->sequence) {
      state->action_sequence.store(state->sequence->fetch_add(1) + 1);
    }
    state->status_code.store(iree_status_code(status));
    state->call_count.fetch_add(1);
  }

  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  std::atomic<uint64_t>* sequence = nullptr;
  std::atomic<uint32_t> call_count{0};
  std::atomic<uint64_t> action_sequence{0};
  std::atomic<bool> completion_lock_was_available{false};
  std::atomic<iree_status_code_t> status_code{IREE_STATUS_UNKNOWN};
};

struct CompletionPhaseObservations {
  static constexpr size_t kPhaseCount =
      static_cast<size_t>(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_CAPACITY_COMPLETING_BEFORE_ENQUEUE) +
      1;

  CompletionPhaseObservations() {
    for (size_t i = 0; i < kPhaseCount; ++i) {
      counts[i].store(0);
      first_sequences[i].store(0);
      last_value0[i].store(0);
      last_value1[i].store(0);
    }
  }

  static void Observe(iree_hal_amdgpu_host_queue_t* queue,
                      iree_hal_amdgpu_host_queue_test_subject_t subject,
                      iree_hal_amdgpu_host_queue_test_phase_t phase,
                      uint64_t value0, uint64_t value1, void* user_data) {
    (void)value0;
    (void)value1;
    auto* state = static_cast<CompletionPhaseObservations*>(user_data);
    if (queue != state->queue ||
        subject != IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER) {
      return;
    }
    const size_t phase_index = static_cast<size_t>(phase);
    if (phase_index >= kPhaseCount) return;
    const uint64_t sequence = state->NextSequence();
    uint64_t expected = 0;
    state->first_sequences[phase_index].compare_exchange_strong(expected,
                                                                sequence);
    state->counts[phase_index].fetch_add(1);
    state->last_value0[phase_index].store(value0);
    state->last_value1[phase_index].store(value1);
    std::lock_guard<std::mutex> lock(state->mutex);
    state->condition.notify_all();
  }

  uint64_t NextSequence() { return sequence.fetch_add(1) + 1; }

  uint32_t count(iree_hal_amdgpu_host_queue_test_phase_t phase) const {
    return counts[static_cast<size_t>(phase)].load();
  }

  uint64_t first_sequence(iree_hal_amdgpu_host_queue_test_phase_t phase) const {
    return first_sequences[static_cast<size_t>(phase)].load();
  }

  uint64_t value0(iree_hal_amdgpu_host_queue_test_phase_t phase) const {
    return last_value0[static_cast<size_t>(phase)].load();
  }

  uint64_t value1(iree_hal_amdgpu_host_queue_test_phase_t phase) const {
    return last_value1[static_cast<size_t>(phase)].load();
  }

  void WaitForCount(iree_hal_amdgpu_host_queue_test_phase_t phase,
                    uint32_t expected_count) const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return count(phase) >= expected_count; });
  }

  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  std::atomic<uint64_t> sequence{0};
  std::array<std::atomic<uint32_t>, kPhaseCount> counts;
  std::array<std::atomic<uint64_t>, kPhaseCount> first_sequences;
  std::array<std::atomic<uint64_t>, kPhaseCount> last_value0;
  std::array<std::atomic<uint64_t>, kPhaseCount> last_value1;
  mutable std::mutex mutex;
  mutable std::condition_variable condition;
};

struct CompletionResourceObservations {
  std::atomic<uint32_t> destroy_count{0};
  ThreadCompletion destroyed;
};

struct CompletionResource {
  iree_hal_resource_t resource;
  CompletionResourceObservations* observations = nullptr;
};

static void DestroyCompletionResource(iree_hal_resource_t* base_resource) {
  auto* resource = reinterpret_cast<CompletionResource*>(base_resource);
  resource->observations->destroy_count.fetch_add(1);
  resource->observations->destroyed.MarkDone();
  delete resource;
}

static const iree_hal_resource_vtable_t kCompletionResourceVtable = {
    /*.destroy=*/DestroyCompletionResource,
};

static CompletionResource* CreateCompletionResource(
    CompletionResourceObservations* observations) {
  auto* resource = new CompletionResource;
  iree_hal_resource_initialize(&kCompletionResourceVtable, &resource->resource);
  resource->observations = observations;
  return resource;
}

struct PendingCleanupResourceObservations {
  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  iree_hal_queue_t* owned_queue = nullptr;
  iree_hal_device_t* owned_device = nullptr;
  ThreadGate gate;
  ThreadCompletion returned;
  std::atomic<uint32_t> entry_count{0};
  std::atomic<uint32_t> owner_release_return_count{0};
  std::atomic<uint32_t> return_count{0};
  std::atomic<bool> completion_lock_was_available{false};
  std::atomic<bool> submission_lock_was_available{false};
};

struct PendingCleanupResource {
  iree_hal_resource_t resource;
  PendingCleanupResourceObservations* observations = nullptr;
};

static void DestroyPendingCleanupResource(iree_hal_resource_t* base_resource) {
  auto* resource = reinterpret_cast<PendingCleanupResource*>(base_resource);
  PendingCleanupResourceObservations* observations = resource->observations;
  observations->entry_count.fetch_add(1);
  observations->completion_lock_was_available.store(AcquireAndReleaseMutex(
      &observations->queue->locks.completion_drain_mutex));
  observations->submission_lock_was_available.store(
      AcquireAndReleaseMutex(&observations->queue->locks.submission_mutex));

  // Optional caller-owned edges are consumed only after both queue locks have
  // been proven absent. A successful pending completion's lifetime claim keeps
  // them alive; cancellation/destructor paths instead rely on the scalar
  // epilogue token owned by the already-running sealer.
  iree_hal_device_t* owned_device = observations->owned_device;
  observations->owned_device = nullptr;
  if (owned_device) iree_hal_device_release(owned_device);
  iree_hal_queue_t* owned_queue = observations->owned_queue;
  observations->owned_queue = nullptr;
  if (owned_queue) iree_hal_queue_release(owned_queue);
  observations->owner_release_return_count.fetch_add(1);

  observations->gate.ArriveAndWait();
  observations->return_count.fetch_add(1);
  observations->returned.MarkDone();
  delete resource;
}

static const iree_hal_resource_vtable_t kPendingCleanupResourceVtable = {
    /*.destroy=*/DestroyPendingCleanupResource,
};

static PendingCleanupResource* CreatePendingCleanupResource(
    PendingCleanupResourceObservations* observations) {
  auto* resource = new PendingCleanupResource;
  iree_hal_resource_initialize(&kPendingCleanupResourceVtable,
                               &resource->resource);
  resource->observations = observations;
  return resource;
}

struct BindingBufferReleaseObservations {
  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  bool reenter_queue = false;
  std::atomic<uint32_t> release_count{0};
  std::atomic<uint32_t> reentry_count{0};
  std::atomic<iree_status_code_t> reentry_status_code{IREE_STATUS_OK};
  std::atomic<uint64_t> epoch_observed_under_lock{UINT64_MAX};
  std::atomic<uint64_t> write_observed_under_lock{UINT64_MAX};
  std::atomic<bool> all_submission_lock_checks_passed{true};
};

struct BindingBufferReleaseToken {
  static void Release(void* user_data, iree_hal_buffer_t* buffer) {
    (void)buffer;
    auto* token = static_cast<BindingBufferReleaseToken*>(user_data);
    const bool lock_available = iree_slim_mutex_try_lock(
        &token->observations->queue->locks.submission_mutex);
    if (lock_available) {
      token->observations->epoch_observed_under_lock.store(
          token->observations->queue->notification_ring.epoch.next_submission);
      token->observations->write_observed_under_lock.store(
          static_cast<uint64_t>(iree_atomic_load(
              &token->observations->queue->notification_ring.write,
              iree_memory_order_acquire)));
      iree_slim_mutex_unlock(
          &token->observations->queue->locks.submission_mutex);
    } else {
      token->observations->all_submission_lock_checks_passed.store(false);
    }
    if (lock_available && token->observations->reenter_queue) {
      iree_status_t status = iree_hal_queue_barrier(
          &token->observations->queue->base, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
      token->observations->reentry_status_code.store(iree_status_code(status));
      iree_status_free(status);
      token->observations->reentry_count.fetch_add(1);
    }
    token->observations->release_count.fetch_add(1);
  }

  BindingBufferReleaseObservations* observations = nullptr;
};

struct FailSecondBlockAllocator {
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* state = static_cast<FailSecondBlockAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC) {
      const uint32_t allocation_ordinal =
          state->allocation_count.fetch_add(1) + 1;
      if (allocation_ordinal == state->fail_at_allocation) {
        for (iree_host_size_t i = 0; i < state->release_prefix_count; ++i) {
          iree_hal_buffer_t*& buffer = (*state->caller_buffers)[i];
          IREE_ASSERT(buffer);
          iree_hal_buffer_release(buffer);
          buffer = nullptr;
          state->caller_release_count.fetch_add(1);
        }
        *inout_ptr = nullptr;
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
    }
    return state->backing.ctl(state->backing.self, command, params, inout_ptr);
  }

  iree_allocator_t allocator() {
    return iree_allocator_t{
        /*.self=*/this,
        /*.ctl=*/Control,
    };
  }

  iree_allocator_t backing = iree_allocator_system();
  std::vector<iree_hal_buffer_t*>* caller_buffers = nullptr;
  iree_host_size_t release_prefix_count = 0;
  uint32_t fail_at_allocation = 2;
  std::atomic<uint32_t> allocation_count{0};
  std::atomic<uint32_t> caller_release_count{0};
};

struct alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) BindingBufferStorage {
  uint8_t data[IREE_HAL_HEAP_BUFFER_ALIGNMENT];
};

enum class CommandBufferOwnerCaptureCase {
  kDirectAql,
  kRetainedAqlReplay,
  kUnretainedAqlReplay,
  kPm4DynamicFixup,
  kPm4Profiled,
};

struct alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) OwnerCaptureBufferStorage {
  uint8_t data[4096];
};

static void WaitForHardwareEpoch(const iree_hal_amdgpu_libhsa_t* libhsa,
                                 iree_hal_amdgpu_host_queue_t* queue,
                                 uint64_t epoch);

struct QueueAxisFrontierWaiter {
  static uint64_t LoadCursor(const iree_atomic_int64_t* cursor) {
    return static_cast<uint64_t>(
        iree_atomic_load(cursor, iree_memory_order_acquire));
  }

  bool PublicCursorsAreFrozen() const {
    return LoadCursor(&queue->notification_ring.epoch.last_drained) ==
               initial_epoch &&
           LoadCursor(&queue->notification_ring.read) == initial_read &&
           LoadCursor(&queue->notification_ring.frontier_ring.read) ==
               initial_frontier_read;
  }

  static void Callback(void* user_data, iree_status_t status) {
    auto* state = static_cast<QueueAxisFrontierWaiter*>(user_data);
    state->status_code.store(iree_status_code(status));
    iree_status_free(status);
    state->callback_sequence.store(state->phases->NextSequence());

    const bool drain_lock_available =
        iree_slim_mutex_try_lock(&state->queue->locks.completion_drain_mutex);
    state->drain_lock_was_available.store(drain_lock_available);
    if (drain_lock_available) {
      iree_slim_mutex_unlock(&state->queue->locks.completion_drain_mutex);
    }
    const bool submission_lock_available =
        iree_slim_mutex_try_lock(&state->queue->locks.submission_mutex);
    state->submission_lock_was_available.store(submission_lock_available);
    if (submission_lock_available) {
      iree_slim_mutex_unlock(&state->queue->locks.submission_mutex);
    }
    state->resource_was_live_at_entry.store(
        state->resource_observations->destroy_count.load() == 0);
    state->cursors_were_frozen_at_entry.store(state->PublicCursorsAreFrozen());
    state->releases_had_not_started.store(
        state->phases->count(
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_RELEASES_DONE) == 0);
    state->public_commit_had_not_started.store(
        state->phases->count(
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PUBLIC_CURSORS_FINALIZED) ==
        0);
    state->feedback_phase_count_at_entry.store(state->phases->count(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_FEEDBACK_TARGET_CLOSED));
    state->dispatch_phase_count_at_entry.store(state->phases->count(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED));

    if (state->blocker_signal.handle) {
      iree_hsa_signal_store_screlease(IREE_LIBHSA(state->libhsa),
                                      state->blocker_signal, 0);
      WaitForHardwareEpoch(state->libhsa, state->queue, state->later_epoch);
      const uint32_t claimed_count = state->phases->count(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED);
      const uint32_t transitioned_count = state->phases->count(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TRANSITIONS_DONE);
      const uint32_t prepared_count = state->phases->count(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_VALUES_PREPARED);
      const uint32_t feedback_count = state->phases->count(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_FEEDBACK_TARGET_CLOSED);
      const uint32_t dispatch_count = state->phases->count(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED);
      state->nested_drain_count.store(
          iree_hal_amdgpu_host_queue_drain_completions_for_waiter(
              state->queue));
      state->nested_claimed_phase_delta.store(
          state->phases->count(
              IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED) -
          claimed_count);
      state->nested_transition_phase_delta.store(
          state->phases->count(
              IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TRANSITIONS_DONE) -
          transitioned_count);
      state->nested_prepare_phase_delta.store(
          state->phases->count(
              IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_VALUES_PREPARED) -
          prepared_count);
      state->nested_feedback_phase_delta.store(
          state->phases->count(
              IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_FEEDBACK_TARGET_CLOSED) -
          feedback_count);
      state->nested_dispatch_phase_delta.store(
          state->phases->count(
              IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED) -
          dispatch_count);
      state->nested_claimed_target.store(state->phases->value0(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED));
      state->nested_prepared_read.store(state->phases->value0(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_VALUES_PREPARED));
      state->nested_dispatched_read.store(state->phases->value0(
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED));
    }
    iree_status_t wait_status = iree_hal_semaphore_wait(
        state->later_semaphore, state->later_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    state->nested_wait_status_code.store(iree_status_code(wait_status));
    iree_status_free(wait_status);

    state->resource_was_live_after_nested_wait.store(
        state->resource_observations->destroy_count.load() == 0);
    state->cursors_were_frozen_after_nested_wait.store(
        state->PublicCursorsAreFrozen());
    state->call_count.fetch_add(1);
    state->completed.MarkDone();
  }

  const iree_hal_amdgpu_libhsa_t* libhsa = nullptr;
  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  CompletionPhaseObservations* phases = nullptr;
  CompletionResourceObservations* resource_observations = nullptr;
  iree_hal_semaphore_t* later_semaphore = nullptr;
  uint64_t later_value = 0;
  uint64_t later_epoch = 0;
  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  uint64_t initial_epoch = 0;
  uint64_t initial_read = 0;
  uint64_t initial_frontier_read = 0;
  iree_async_frontier_waiter_t waiter = {};
  std::atomic<uint32_t> call_count{0};
  std::atomic<iree_host_size_t> nested_drain_count{0};
  std::atomic<uint32_t> feedback_phase_count_at_entry{0};
  std::atomic<uint32_t> dispatch_phase_count_at_entry{0};
  std::atomic<uint32_t> nested_claimed_phase_delta{0};
  std::atomic<uint32_t> nested_transition_phase_delta{0};
  std::atomic<uint32_t> nested_prepare_phase_delta{0};
  std::atomic<uint32_t> nested_feedback_phase_delta{0};
  std::atomic<uint32_t> nested_dispatch_phase_delta{0};
  std::atomic<uint64_t> nested_claimed_target{0};
  std::atomic<uint64_t> nested_prepared_read{0};
  std::atomic<uint64_t> nested_dispatched_read{0};
  std::atomic<uint64_t> callback_sequence{0};
  std::atomic<iree_status_code_t> status_code{IREE_STATUS_UNKNOWN};
  std::atomic<iree_status_code_t> nested_wait_status_code{IREE_STATUS_UNKNOWN};
  std::atomic<bool> drain_lock_was_available{false};
  std::atomic<bool> submission_lock_was_available{false};
  std::atomic<bool> resource_was_live_at_entry{false};
  std::atomic<bool> resource_was_live_after_nested_wait{false};
  std::atomic<bool> cursors_were_frozen_at_entry{false};
  std::atomic<bool> cursors_were_frozen_after_nested_wait{false};
  std::atomic<bool> releases_had_not_started{false};
  std::atomic<bool> public_commit_had_not_started{false};
  ThreadCompletion completed;
};

enum class ObservingPoolMethod : uint32_t {
  kAcquire = 0,
  kRelease,
  kMaterialize,
  kQueryCapabilities,
  kQueryStats,
  kTrim,
  kNotification,
  kCount,
};

enum class ObservingPoolAcquireBehavior : uint32_t {
  kDelegate = 0,
  kFail,
  kMalformedWait,
  kExhausted,
};

enum class AllocaUnlockedWindow : uint32_t {
  kNotification = 0,
  kAcquire,
  kMaterialize,
};

struct ObservingPool {
  static ObservingPool* Cast(iree_hal_pool_t* base_pool) {
    return reinterpret_cast<ObservingPool*>(base_pool);
  }

  static const ObservingPool* Cast(const iree_hal_pool_t* base_pool) {
    return reinterpret_cast<const ObservingPool*>(base_pool);
  }

  static void Destroy(iree_hal_pool_t* base_pool) {
    ObservingPool* pool = Cast(base_pool);
    iree_hal_pool_release(pool->delegate);
    delete pool;
  }

  static iree_status_t AcquireReservations(
      iree_hal_pool_t* base_pool, iree_host_size_t request_count,
      const iree_hal_pool_reservation_request_t* requests,
      const iree_async_frontier_t* requester_frontier,
      iree_hal_pool_reserve_flags_t flags,
      iree_hal_pool_reservation_t* out_reservations,
      iree_hal_pool_acquire_info_t* out_infos,
      iree_hal_pool_acquire_result_t* out_result) {
    ObservingPool* pool = Cast(base_pool);
    pool->Observe(ObservingPoolMethod::kAcquire);
    if (pool->acquire_behavior == ObservingPoolAcquireBehavior::kFail) {
      return iree_status_from_code(pool->acquire_failure_code);
    }
    IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservations(
        pool->delegate, request_count, requests, requester_frontier, flags,
        out_reservations, out_infos, out_result));
    if (pool->acquire_after_delegate_gate) {
      pool->acquire_after_delegate_gate->ArriveAndWait();
    }
    if (pool->acquire_behavior ==
        ObservingPoolAcquireBehavior::kMalformedWait) {
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        out_infos[i].result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
        out_infos[i].wait_frontier = nullptr;
      }
      *out_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
    } else if (pool->acquire_behavior ==
               ObservingPoolAcquireBehavior::kExhausted) {
      if (*out_result == IREE_HAL_POOL_ACQUIRE_OK ||
          *out_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
          *out_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
        for (iree_host_size_t i = 0; i < request_count; ++i) {
          iree_hal_pool_release_reservations(pool->delegate, 1,
                                             &out_reservations[i],
                                             out_infos[i].wait_frontier);
        }
      }
      std::memset(out_reservations, 0,
                  request_count * sizeof(*out_reservations));
      std::memset(out_infos, 0, request_count * sizeof(*out_infos));
      *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    }
    return iree_ok_status();
  }

  static void ReleaseReservations(
      iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
      const iree_hal_pool_reservation_t* reservations,
      const iree_async_frontier_t* death_frontier) {
    ObservingPool* pool = Cast(base_pool);
    pool->Observe(ObservingPoolMethod::kRelease);
    if (pool->release_before_delegate_gate) {
      pool->release_before_delegate_gate->ArriveAndWait();
    }
    iree_hal_pool_release_reservations(pool->delegate, reservation_count,
                                       reservations, death_frontier);
  }

  static iree_status_t MaterializeReservations(
      iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
      const iree_hal_pool_reservation_request_t* requests,
      const iree_hal_pool_reservation_t* reservations,
      iree_hal_pool_materialize_flags_t flags,
      iree_hal_buffer_t** out_buffers) {
    ObservingPool* pool = Cast(base_pool);
    pool->Observe(ObservingPoolMethod::kMaterialize);
    IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
        pool->delegate, reservation_count, requests, reservations, flags,
        out_buffers));
    if (pool->materialize_after_delegate_gate) {
      pool->materialize_after_delegate_gate->ArriveAndWait();
    }
    if (pool->materialize_failure_code != IREE_STATUS_OK) {
      for (iree_host_size_t i = 0; i < reservation_count; ++i) {
        iree_hal_buffer_release(out_buffers[i]);
        out_buffers[i] = nullptr;
      }
      return iree_status_from_code(pool->materialize_failure_code);
    }
    return iree_ok_status();
  }

  static void QueryCapabilities(
      const iree_hal_pool_t* base_pool,
      iree_hal_pool_capabilities_t* out_capabilities) {
    ObservingPool* pool = const_cast<ObservingPool*>(Cast(base_pool));
    pool->Observe(ObservingPoolMethod::kQueryCapabilities);
    iree_hal_pool_query_capabilities(pool->delegate, out_capabilities);
  }

  static void QueryStats(const iree_hal_pool_t* base_pool,
                         iree_hal_pool_stats_t* out_stats) {
    ObservingPool* pool = const_cast<ObservingPool*>(Cast(base_pool));
    pool->Observe(ObservingPoolMethod::kQueryStats);
    if (pool->query_stats_gate) pool->query_stats_gate->ArriveAndWait();
    iree_hal_pool_query_stats(pool->delegate, out_stats);
  }

  static iree_status_t Trim(iree_hal_pool_t* base_pool) {
    ObservingPool* pool = Cast(base_pool);
    pool->Observe(ObservingPoolMethod::kTrim);
    return iree_hal_pool_trim(pool->delegate);
  }

  static iree_async_notification_t* Notification(iree_hal_pool_t* base_pool) {
    ObservingPool* pool = Cast(base_pool);
    pool->Observe(ObservingPoolMethod::kNotification);
    iree_async_notification_t* notification =
        pool->return_null_notification
            ? nullptr
            : iree_hal_pool_notification(pool->delegate);
    if (pool->notification_after_delegate_gate) {
      pool->notification_after_delegate_gate->ArriveAndWait();
    }
    return notification;
  }

  void Observe(ObservingPoolMethod method) {
    const size_t index = static_cast<size_t>(method);
    call_counts[index].fetch_add(1);
    all_submission_lock_checks_passed[index].store(
        AcquireAndReleaseMutex(&queue->locks.submission_mutex));

    if (!reenter_queue) return;
    iree_status_t status = iree_hal_queue_barrier(
        &queue->base, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    if (!iree_status_is_ok(status)) {
      reentry_status_code.store(iree_status_code(status));
      iree_status_free(status);
    }
    reentry_counts[index].fetch_add(1);
  }

  uint32_t call_count(ObservingPoolMethod method) const {
    return call_counts[static_cast<size_t>(method)].load();
  }

  uint32_t reentry_count(ObservingPoolMethod method) const {
    return reentry_counts[static_cast<size_t>(method)].load();
  }

  bool submission_lock_was_available(ObservingPoolMethod method) const {
    return all_submission_lock_checks_passed[static_cast<size_t>(method)]
        .load();
  }

  static const iree_hal_pool_vtable_t vtable;
  iree_hal_pool_t base;
  iree_hal_pool_t* delegate = nullptr;
  iree_hal_amdgpu_host_queue_t* queue = nullptr;
  std::array<std::atomic<uint32_t>,
             static_cast<size_t>(ObservingPoolMethod::kCount)>
      call_counts;
  std::array<std::atomic<uint32_t>,
             static_cast<size_t>(ObservingPoolMethod::kCount)>
      reentry_counts;
  std::array<std::atomic<bool>,
             static_cast<size_t>(ObservingPoolMethod::kCount)>
      all_submission_lock_checks_passed;
  std::atomic<iree_status_code_t> reentry_status_code{IREE_STATUS_OK};
  ObservingPoolAcquireBehavior acquire_behavior =
      ObservingPoolAcquireBehavior::kDelegate;
  iree_status_code_t acquire_failure_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  iree_status_code_t materialize_failure_code = IREE_STATUS_OK;
  ThreadGate* notification_after_delegate_gate = nullptr;
  ThreadGate* acquire_after_delegate_gate = nullptr;
  ThreadGate* materialize_after_delegate_gate = nullptr;
  ThreadGate* release_before_delegate_gate = nullptr;
  ThreadGate* query_stats_gate = nullptr;
  bool return_null_notification = false;
  bool reenter_queue = true;
};

const iree_hal_pool_vtable_t ObservingPool::vtable = {
    /*.destroy=*/ObservingPool::Destroy,
    /*.acquire_reservations=*/ObservingPool::AcquireReservations,
    /*.release_reservations=*/ObservingPool::ReleaseReservations,
    /*.materialize_reservations=*/ObservingPool::MaterializeReservations,
    /*.query_capabilities=*/ObservingPool::QueryCapabilities,
    /*.query_stats=*/ObservingPool::QueryStats,
    /*.trim=*/ObservingPool::Trim,
    /*.notification=*/ObservingPool::Notification,
};

static ObservingPool* CreateObservingPool(iree_hal_pool_t* delegate,
                                          iree_hal_amdgpu_host_queue_t* queue) {
  auto* pool = new ObservingPool;
  iree_hal_pool_initialize(&ObservingPool::vtable, &pool->base);
  pool->delegate = delegate;
  pool->queue = queue;
  for (size_t i = 0; i < static_cast<size_t>(ObservingPoolMethod::kCount);
       ++i) {
    pool->call_counts[i].store(0);
    pool->reentry_counts[i].store(0);
    pool->all_submission_lock_checks_passed[i].store(true);
  }
  return pool;
}

static void ExpectPoolMethodUnlocked(ObservingPool* pool,
                                     ObservingPoolMethod method,
                                     uint32_t expected_count) {
  EXPECT_EQ(pool->call_count(method), expected_count);
  EXPECT_TRUE(pool->submission_lock_was_available(method));
  EXPECT_EQ(pool->reentry_count(method), expected_count);
}

struct NeverCalledHostCallState {
  static iree_status_t Call(void* user_data, const uint64_t args[4],
                            iree_hal_host_call_context_t* context) {
    (void)args;
    (void)context;
    auto* state = static_cast<NeverCalledHostCallState*>(user_data);
    state->call_count.fetch_add(1);
    return iree_ok_status();
  }

  std::atomic<uint32_t> call_count{0};
};

struct CountingHostCallState {
  static iree_status_t Call(void* user_data, const uint64_t args[4],
                            iree_hal_host_call_context_t* context) {
    (void)args;
    (void)context;
    auto* state = static_cast<CountingHostCallState*>(user_data);
    state->call_count.fetch_add(1);
    return iree_ok_status();
  }

  std::atomic<uint32_t> call_count{0};
};

static iree_status_t EnqueueCleanupOnlyDeferredHostCall(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t& wait_list, iree_hal_host_call_t call,
    iree_hal_resource_t* cleanup_resource, bool force_capacity_retry,
    iree_hal_amdgpu_pending_op_t** out_op) {
  *out_op = nullptr;
  iree_hal_amdgpu_pending_op_t* op = nullptr;
  const iree_hal_semaphore_list_t signal_list = iree_hal_semaphore_list_empty();
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_hal_amdgpu_pending_op_allocate(
      queue, &wait_list, &signal_list, IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL,
      /*max_resource_count=*/1, &op);
  if (iree_status_is_ok(status)) {
    op->host_call.call = call;
    memset(op->host_call.args, 0, sizeof(op->host_call.args));
    op->host_call.flags = IREE_HAL_HOST_CALL_FLAG_NONE;
    iree_hal_amdgpu_pending_op_retain(op, cleanup_resource);
    if (force_capacity_retry) {
      // Match production op_submission_end: make the COMPLETING state and
      // post-drain retry visible before submission admission is released.
      status = iree_hal_amdgpu_pending_op_start(op, /*wait_for_capacity=*/true);
    }
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  if (!iree_status_is_ok(status)) return status;
  *out_op = op;
  return force_capacity_retry ? iree_ok_status()
                              : iree_hal_amdgpu_pending_op_start(
                                    op, /*wait_for_capacity=*/false);
}

struct CompletionRunnerIdleState {
  iree_hal_amdgpu_host_queue_t* queue;
  uint32_t expected_epilogue_count;
};

static bool CompletionRunnerIsIdleWithEpilogueCount(void* user_data) {
  auto* state = static_cast<CompletionRunnerIdleState*>(user_data);
  iree_slim_mutex_lock(&state->queue->locks.completion_drain_mutex);
  const bool has_expected_state =
      !state->queue->completion.runner_active &&
      state->queue->completion.epilogue_count == state->expected_epilogue_count;
  iree_slim_mutex_unlock(&state->queue->locks.completion_drain_mutex);
  return has_expected_state;
}

static void WaitForCompletionRunnerIdleWithEpilogueCount(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t expected_epilogue_count) {
  CompletionRunnerIdleState state = {
      .queue = queue,
      .expected_epilogue_count = expected_epilogue_count,
  };
  iree_notification_await(&queue->completion.runner_notification,
                          CompletionRunnerIsIdleWithEpilogueCount, &state,
                          iree_infinite_timeout());
}

static bool StagingPoolHasQueuedWaiter(iree_hal_amdgpu_staging_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  const bool has_waiter = pool->waiter_head != nullptr;
  iree_slim_mutex_unlock(&pool->mutex);
  return has_waiter;
}

static bool DynamicQueueSlotIsLive(
    iree_hal_amdgpu_logical_device_t* logical_device, uint8_t queue_index) {
  const uint32_t word_index = queue_index / 64u;
  const uint64_t bit = UINT64_C(1) << (queue_index % 64u);
  iree_slim_mutex_lock(&logical_device->dynamic_queue_slots.mutex);
  const bool is_live =
      (logical_device->dynamic_queue_slots.live_bits[word_index] & bit) != 0;
  iree_slim_mutex_unlock(&logical_device->dynamic_queue_slots.mutex);
  return is_live;
}

class DynamicQueueSlotReleaseLatch {
 public:
  void Install(iree_hal_amdgpu_host_queue_t* queue) {
    IREE_ASSERT(original_.fn == nullptr);
    original_ = queue->storage.release_slot;
    IREE_ASSERT(original_.fn != nullptr);
    queue->storage.release_slot.fn = Release;
    queue->storage.release_slot.user_data = this;
  }

  void Wait() { released_.Wait(); }

  uint32_t release_count() const { return release_count_.load(); }
  uint8_t queue_index() const { return queue_index_.load(); }

 private:
  static void Release(void* user_data, uint8_t queue_index) {
    auto* latch = static_cast<DynamicQueueSlotReleaseLatch*>(user_data);
    latch->original_.fn(latch->original_.user_data, queue_index);
    latch->queue_index_.store(queue_index);
    latch->release_count_.fetch_add(1);
    latch->released_.MarkDone();
  }

  iree_hal_amdgpu_host_queue_release_slot_callback_t original_ = {};
  ThreadCompletion released_;
  std::atomic<uint32_t> release_count_{0};
  std::atomic<uint8_t> queue_index_{UINT8_MAX};
};

static size_t CountPendingOperations(iree_hal_amdgpu_host_queue_t* queue) {
  size_t count = 0;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  for (iree_hal_amdgpu_pending_op_t* operation = queue->pending_head; operation;
       operation = operation->next) {
    ++count;
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return count;
}

static bool StatusMentions(const iree_status_t status, const char* text) {
  if (iree_status_is_ok(status)) return false;
  iree_allocator_t host_allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t buffer_length = 0;
  if (!iree_status_to_string(status, &host_allocator, &buffer,
                             &buffer_length)) {
    return false;
  }
  const bool mentions = std::strstr(buffer, text) != nullptr;
  iree_allocator_free(host_allocator, buffer);
  return mentions;
}

static void ExpectOwnedStatus(iree_status_t status,
                              iree_status_code_t expected_code,
                              const char* expected_message = nullptr) {
  EXPECT_EQ(iree_status_code(status), expected_code);
  if (expected_message) {
    EXPECT_TRUE(StatusMentions(status, expected_message));
  }
  iree_status_free(status);
}

struct QueuePublicationSnapshot {
  uint64_t next_submission = 0;
  uint64_t last_published = 0;
  uint64_t notification_write = 0;
  uint64_t frontier_write = 0;
  int64_t aql_write = 0;
};

static QueuePublicationSnapshot SnapshotQueuePublication(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const QueuePublicationSnapshot snapshot = {
      /*.next_submission=*/queue->notification_ring.epoch.next_submission,
      /*.last_published=*/
      QueueAxisFrontierWaiter::LoadCursor(
          &queue->notification_ring.epoch.last_published),
      /*.notification_write=*/
      QueueAxisFrontierWaiter::LoadCursor(&queue->notification_ring.write),
      /*.frontier_write=*/
      QueueAxisFrontierWaiter::LoadCursor(
          &queue->notification_ring.frontier_ring.write),
      /*.aql_write=*/
      iree_atomic_load(queue->aql_ring.write_dispatch_id,
                       iree_memory_order_acquire),
  };
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return snapshot;
}

static void ExpectQueuePublicationUnchanged(
    const QueuePublicationSnapshot& before,
    const QueuePublicationSnapshot& after) {
  EXPECT_EQ(after.next_submission, before.next_submission);
  EXPECT_EQ(after.last_published, before.last_published);
  EXPECT_EQ(after.notification_write, before.notification_write);
  EXPECT_EQ(after.frontier_write, before.frontier_write);
  EXPECT_EQ(after.aql_write, before.aql_write);
}

static void WaitForSubmittedEpoch(const iree_hal_amdgpu_libhsa_t* libhsa,
                                  iree_hal_amdgpu_host_queue_t* queue) {
  const uint64_t submitted_epoch =
      queue->notification_ring.epoch.next_submission;
  if (submitted_epoch == 0) return;
  const hsa_signal_value_t compare_value =
      static_cast<hsa_signal_value_t>(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                                      submitted_epoch) +
      1;
  const hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  (void)iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), epoch_signal,
                                       HSA_SIGNAL_CONDITION_LT, compare_value,
                                       UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
}

static void WaitForHardwareEpoch(const iree_hal_amdgpu_libhsa_t* libhsa,
                                 iree_hal_amdgpu_host_queue_t* queue,
                                 uint64_t epoch) {
  const hsa_signal_value_t target_value = static_cast<hsa_signal_value_t>(
      IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - epoch);
  const hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  (void)iree_hsa_signal_wait_scacquire(
      IREE_LIBHSA(libhsa), epoch_signal, HSA_SIGNAL_CONDITION_LT,
      target_value + 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
}

static void WaitForSignalNonzero(const iree_hal_amdgpu_libhsa_t* libhsa,
                                 hsa_signal_t signal) {
  (void)iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), signal,
                                       HSA_SIGNAL_CONDITION_NE, 0, UINT64_MAX,
                                       HSA_WAIT_STATE_BLOCKED);
}

static void ExpectQueueCertifiedAndEmpty(iree_hal_amdgpu_host_queue_t* queue,
                                         iree_hal_amdgpu_staging_pool_t* pool) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  EXPECT_TRUE(queue->idle_certificate_valid);
  EXPECT_FALSE(queue->seal_in_progress);
  EXPECT_EQ(queue->pending_head, nullptr);
  EXPECT_EQ(queue->active_staging_transfer_head, nullptr);
  EXPECT_EQ(queue->shutdown_staging_transfer_head, nullptr);
  EXPECT_EQ(queue->active_file_action_head, nullptr);
  EXPECT_EQ(queue->shutdown_file_action_head, nullptr);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  EXPECT_FALSE(queue->post_drain.runner_active);
  EXPECT_EQ(queue->post_drain.head, nullptr);
  EXPECT_EQ(queue->post_drain.tail, nullptr);
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);

  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  EXPECT_EQ(queue->completion.runner_state,
            IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_CLOSED);
  EXPECT_FALSE(queue->completion.runner_active);
  EXPECT_EQ(queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);

  iree_slim_mutex_lock(&pool->mutex);
  EXPECT_EQ(pool->waiter_head, nullptr);
  EXPECT_EQ(pool->waiter_tail, nullptr);
  EXPECT_EQ(pool->claimed_waiter_count, 0u);
  EXPECT_EQ(pool->available_count, pool->slot_count);
  iree_slim_mutex_unlock(&pool->mutex);
}

static void ExpectControllerUnchanged(const ControlledProactorSnapshot& before,
                                      const ControlledProactorSnapshot& after) {
  EXPECT_EQ(after.submit_attempt_count, before.submit_attempt_count);
  EXPECT_EQ(after.submit_count, before.submit_count);
  EXPECT_EQ(after.submit_failure_count, before.submit_failure_count);
  EXPECT_EQ(after.cancel_count, before.cancel_count);
  EXPECT_EQ(after.callback_entry_count, before.callback_entry_count);
  EXPECT_EQ(after.callback_exit_count, before.callback_exit_count);
  EXPECT_EQ(after.calls_after_seal_count, before.calls_after_seal_count);
  EXPECT_EQ(after.release_without_pending_count, 0u);
  EXPECT_EQ(after.pending_count, 0u);
}

static iree_status_t EnqueueRawBlockingBarrier(
    iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t blocker_signal) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, 1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dependency_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dependency_signals,
      IREE_ARRAYSIZE(dependency_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(),
      iree_hsa_signal_null());
  iree_hal_amdgpu_aql_ring_commit(packet, header, 0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
  return iree_ok_status();
}

// Makes a raw hardware blocker failure-safe for tests with assertions between
// publication and the normal release point. On an early return it clears any
// test observer, releases the hardware dependency, seals the still-live queue,
// and only then destroys the signal.
class RawQueueBlockerGuard {
 public:
  RawQueueBlockerGuard(const iree_hal_amdgpu_libhsa_t* libhsa,
                       iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t signal)
      : libhsa_(libhsa), queue_(queue), signal_(signal) {}

  ~RawQueueBlockerGuard() {
    if (!signal_.handle) return;
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
    Release();
    iree_hal_amdgpu_host_queue_seal(queue_);
    iree_status_ignore(iree_hsa_signal_destroy(IREE_LIBHSA(libhsa_), signal_));
  }

  void Release() {
    if (released_) return;
    iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa_), signal_, 0);
    released_ = true;
  }

  iree_status_t DestroyAfterSeal() {
    Release();
    hsa_signal_t signal = signal_;
    signal_ = iree_hsa_signal_null();
    return iree_hsa_signal_destroy(IREE_LIBHSA(libhsa_), signal);
  }

 private:
  const iree_hal_amdgpu_libhsa_t* const libhsa_;
  iree_hal_amdgpu_host_queue_t* const queue_;
  hsa_signal_t signal_;
  bool released_ = false;
};

// Keeps a publisher observer and its blocking latches alive through queue
// teardown. Cleanup opens both latches before sealing so an observer already
// inside either callback can leave, and seal joins all queue work before the
// observer registration and backing storage are released.
class PublisherSubmissionObserverGuard {
 public:
  PublisherSubmissionObserverGuard(iree_hal_amdgpu_host_queue_t* queue,
                                   PublisherSubmissionLatch* publisher_latch,
                                   QueuePhaseLatch* failure_latch,
                                   QueuePhaseObserverGroup* phase_observers)
      : queue_(queue),
        publisher_latch_(publisher_latch),
        failure_latch_(failure_latch) {
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        QueuePhaseObserverGroup::Observe, phase_observers);
  }

  ~PublisherSubmissionObserverGuard() { Reset(); }

  PublisherSubmissionObserverGuard(const PublisherSubmissionObserverGuard&) =
      delete;
  PublisherSubmissionObserverGuard& operator=(
      const PublisherSubmissionObserverGuard&) = delete;

  void Reset() {
    if (!queue_) return;
    publisher_latch_->Release();
    failure_latch_->Release();
    iree_hal_amdgpu_host_queue_seal(queue_);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
    queue_ = nullptr;
  }

 private:
  iree_hal_amdgpu_host_queue_t* queue_;
  PublisherSubmissionLatch* const publisher_latch_;
  QueuePhaseLatch* const failure_latch_;
};

static iree_status_t ReserveFeedback(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_host_size_t physical_device_ordinal, uint64_t source_context,
    iree_hal_amdgpu_feedback_packet_kind_t kind, size_t payload_length,
    iree_hal_amdgpu_feedback_config_t* out_config,
    iree_hal_amdgpu_feedback_packet_t** out_packet) {
  *out_packet = nullptr;
  iree_hal_amdgpu_feedback_config_t config;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_feedback_state_populate_config(
      &logical_device->feedback, physical_device_ordinal, &config));
  config.source_context = source_context;

  if (!iree_hal_amdgpu_feedback_try_reserve(
          &config, kind, IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
          payload_length, out_packet)) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  *out_config = config;
  return iree_ok_status();
}

static void InitializeAsanFeedbackReport(
    iree_hal_amdgpu_feedback_packet_t* packet) {
  auto* report = static_cast<iree_hal_amdgpu_asan_report_t*>(
      iree_hal_amdgpu_feedback_packet_payload(packet));
  *report = {
      .record_length = sizeof(*report),
      .abi_version = IREE_HAL_AMDGPU_ASAN_REPORT_ABI_VERSION_0,
      .access_kind = IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
      .flags = IREE_HAL_AMDGPU_ASAN_REPORT_FLAG_NONE,
      .fault_address = 0xCAFE0000u,
      .access_size = 4,
      .site_id = 0,
      .shadow_address = 0,
      .shadow_value = 0,
      .reserved = {0},
  };
}

static iree_status_t ReserveAsanFeedback(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_host_size_t physical_device_ordinal, uint64_t source_context,
    iree_hal_amdgpu_feedback_config_t* out_config,
    iree_hal_amdgpu_feedback_packet_t** out_packet) {
  IREE_RETURN_IF_ERROR(ReserveFeedback(
      logical_device, physical_device_ordinal, source_context,
      IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
      sizeof(iree_hal_amdgpu_asan_report_t), out_config, out_packet));
  InitializeAsanFeedbackReport(*out_packet);
  return iree_ok_status();
}

static iree_status_t PublishAsanFeedback(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_host_size_t physical_device_ordinal, uint64_t source_context) {
  iree_hal_amdgpu_feedback_config_t config;
  iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
  IREE_RETURN_IF_ERROR(ReserveAsanFeedback(logical_device,
                                           physical_device_ordinal,
                                           source_context, &config, &packet));
  iree_hal_amdgpu_feedback_publish(&config, packet);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&logical_device->system->libhsa),
                                  config.notify_signal, 1);
  return iree_ok_status();
}

class HostQueuePublisherLifetimeTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  class TestLogicalDevice {
   public:
    ~TestLogicalDevice() {
      iree_hal_device_release(base_device_);
      iree_hal_device_group_release(device_group_);
      iree_async_frontier_tracker_release(frontier_tracker_);
      iree_async_proactor_pool_release(proactor_pool_);
      ControlledProactorController* expected = controller_;
      g_controller.compare_exchange_strong(expected, nullptr);
    }

    iree_status_t Initialize(
        ControlledProactorController* controller,
        const iree_hal_amdgpu_logical_device_options_t* options,
        uint32_t staging_slot_count,
        iree_hal_device_event_sink_t event_sink =
            iree_hal_device_event_sink_discard()) {
      controller_ = controller;
      ControlledProactorController* expected = nullptr;
      if (!g_controller.compare_exchange_strong(expected, controller)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "controlled proactor fixture is already active");
      }

      iree_async_proactor_pool_options_t pool_options =
          iree_async_proactor_pool_options_default();
      pool_options.proactor_create = ControlledProactorCreate;
      IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
          1, nullptr, pool_options, host_allocator_, &proactor_pool_));

      iree_async_frontier_tracker_options_t frontier_options =
          iree_async_frontier_tracker_options_default();
      frontier_options.axis_table_capacity = 256;
      IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
          frontier_options, host_allocator_, &frontier_tracker_));

      iree_hal_device_create_params_t create_params =
          iree_hal_device_create_params_default();
      create_params.proactor_pool = proactor_pool_;
      create_params.event_sink = event_sink;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
          IREE_SV("amdgpu"), options, &libhsa_, &topology_, &create_params,
          host_allocator_, &base_device_));
      IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
          base_device_, frontier_tracker_, host_allocator_, &device_group_));

      iree_hal_amdgpu_physical_device_t* physical_device =
          first_physical_device();
      if (!physical_device) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "test device has no physical devices");
      }
      iree_hal_amdgpu_staging_pool_deinitialize(
          &physical_device->file_staging_pool);
      iree_hal_amdgpu_staging_pool_options_t staging_options;
      iree_hal_amdgpu_staging_pool_options_initialize(&staging_options);
      staging_options.slot_size = kStagingSlotSize;
      staging_options.slot_count = staging_slot_count;
      return iree_hal_amdgpu_staging_pool_initialize(
          base_device_, &logical_device()->system->libhsa,
          &logical_device()->system->topology,
          &physical_device->host_memory_pools,
          iree_hal_make_queue_family_affinity(
              static_cast<iree_hal_queue_family_ordinal_t>(
                  physical_device->device_ordinal)),
          &staging_options, host_allocator_,
          &physical_device->file_staging_pool);
    }

    iree_status_t AcquireDynamicQueue(iree_hal_queue_t** out_queue) {
      const iree_hal_queue_family_t* queue_family =
          iree_hal_device_queue_family(base_device_, 0);
      if (!queue_family) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "test device has no queue family");
      }
      iree_hal_queue_params_t params;
      iree_hal_queue_params_initialize(&params);
      return iree_hal_device_acquire_queue(base_device_, queue_family, &params,
                                           out_queue);
    }

    iree_status_t AcquireCooperativeQueue(iree_hal_queue_t** out_queue) {
      const iree_hal_queue_family_t* queue_family =
          iree_hal_device_queue_family(base_device_, 0);
      if (!queue_family) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "test device has no queue family");
      }
      iree_hal_queue_params_t params;
      iree_hal_queue_params_initialize(&params);
      params.features = IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
      return iree_hal_device_acquire_queue(base_device_, queue_family, &params,
                                           out_queue);
    }

    iree_hal_queue_t* provisioned_queue() const {
      return iree_hal_device_queue(base_device_, 0, 0);
    }

    void TakeDeviceOwnership(iree_hal_device_t** out_device,
                             iree_hal_device_group_t** out_device_group) {
      IREE_ASSERT_ARGUMENT(out_device);
      IREE_ASSERT_ARGUMENT(out_device_group);
      *out_device = base_device_;
      *out_device_group = device_group_;
      base_device_ = nullptr;
      device_group_ = nullptr;
    }

    void ReleaseFixtureProactorPoolReference() {
      iree_async_proactor_pool_release(proactor_pool_);
      proactor_pool_ = nullptr;
    }

    iree_hal_device_t* base_device() const { return base_device_; }

    iree_hal_allocator_t* allocator() const {
      return iree_hal_device_allocator(base_device_);
    }

    iree_hal_amdgpu_logical_device_t* logical_device() const {
      return reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(base_device_);
    }

    iree_hal_amdgpu_physical_device_t* first_physical_device() const {
      if (!logical_device() || logical_device()->physical_device_count == 0) {
        return nullptr;
      }
      return logical_device()->physical_devices[0];
    }

    iree_hal_amdgpu_staging_pool_t* staging_pool() const {
      return &first_physical_device()->file_staging_pool;
    }

   private:
    ControlledProactorController* controller_ = nullptr;
    iree_async_proactor_pool_t* proactor_pool_ = nullptr;
    iree_async_frontier_tracker_t* frontier_tracker_ = nullptr;
    iree_hal_device_t* base_device_ = nullptr;
    iree_hal_device_group_t* device_group_ = nullptr;
  };

  iree_status_t CreateTestDevice(ControlledProactorController* controller,
                                 TestLogicalDevice* out_device,
                                 uint32_t notification_capacity = 0,
                                 uint32_t staging_slot_count = 1) {
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.preallocate_pools = 0;
    if (notification_capacity != 0) {
      options.host_queues.aql_capacity = 64;
      options.host_queues.notification_capacity = notification_capacity;
      options.host_queues.kernarg_capacity = 128;
    }
    return out_device->Initialize(controller, &options, staging_slot_count);
  }

  iree_status_t CreateFeedbackTestDevice(
      ControlledProactorController* controller,
      iree_hal_device_event_sink_t event_sink, TestLogicalDevice* out_device) {
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.preallocate_pools = 0;
    options.feedback.enabled = true;
    return out_device->Initialize(controller, &options,
                                  /*staging_slot_count=*/1, event_sink);
  }

  iree_status_t CreateNativeFile(iree_hal_device_t* device,
                                 iree_hal_memory_access_t access,
                                 iree::testing::TempFilePath* out_path,
                                 iree_hal_file_t** out_file,
                                 iree_host_size_t file_size = kTransferSize) {
    *out_path = iree::testing::TempFilePath("iree_amdgpu_publisher_lifetime");
    std::vector<uint8_t> contents(file_size, 0x5A);
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        out_path->path_view(),
        iree_make_const_byte_span(contents.data(), contents.size()),
        host_allocator_));
    iree_io_file_mode_t mode =
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_ASYNC |
        IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_SHARE_WRITE;
    if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      mode |= IREE_IO_FILE_MODE_WRITE;
    }
    iree_io_file_handle_t* handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(mode, out_path->path_view(),
                                                  host_allocator_, &handle));
    iree_status_t status = iree_hal_file_import(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

#if IREE_FILE_IO_ENABLE
  iree_status_t CreatePollRunnerFinalizerFile(
      iree_hal_device_t* device, iree_hal_memory_access_t access,
      PollRunnerFileFinalizer* finalizer, iree::testing::TempFilePath* out_path,
      iree_hal_file_t** out_file) {
    *out_path =
        iree::testing::TempFilePath("iree_amdgpu_poll_runner_finalizer");
    std::vector<uint8_t> contents(kTransferSize, 0x3C);
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        out_path->path_view(),
        iree_make_const_byte_span(contents.data(), contents.size()),
        host_allocator_));

    iree_io_file_mode_t mode =
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_ASYNC |
        IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_SHARE_WRITE;
    iree_io_file_access_t io_access = IREE_IO_FILE_ACCESS_READ;
    if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      mode |= IREE_IO_FILE_MODE_WRITE;
      io_access |= IREE_IO_FILE_ACCESS_WRITE;
    }

    iree_io_file_handle_t* source_handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
        mode, out_path->path_view(), host_allocator_, &source_handle));
    finalizer->source_handle = source_handle;
    iree_io_file_handle_t* wrapped_handle = nullptr;
    iree_status_t status = iree_io_file_handle_wrap(
        io_access, mode, iree_io_file_handle_primitive(source_handle),
        iree_io_file_handle_release_callback_t{
            /*.fn=*/PollRunnerFileFinalizer::Release,
            /*.user_data=*/finalizer,
        },
        host_allocator_, &wrapped_handle);
    if (!iree_status_is_ok(status)) {
      finalizer->source_handle = nullptr;
      iree_io_file_handle_release(source_handle);
      return status;
    }

    status = iree_hal_file_import(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                  access, wrapped_handle,
                                  IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
    iree_io_file_handle_release(wrapped_handle);
    return status;
  }
#endif  // IREE_FILE_IO_ENABLE

  iree_status_t CreateBuffer(iree_hal_allocator_t* allocator, bool host_visible,
                             iree_hal_buffer_t** out_buffer,
                             iree_device_size_t length = kTransferSize) {
    iree_hal_buffer_params_t params = {};
    if (host_visible) {
      params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      params.usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    } else {
      params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
      params.usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
    }
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    return iree_hal_allocator_allocate_buffer(allocator, params, length,
                                              out_buffer);
  }

  iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                     0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                     out_semaphore);
  }

  iree_status_t CreateFixedBlockPool(iree_hal_device_t* device,
                                     const iree_hal_queue_family_t* family,
                                     iree_device_size_t block_size,
                                     iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {};
    IREE_RETURN_IF_ERROR(
        iree_hal_device_query_queue_pool_backend(device, family, &backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete backend bundle");
    }
    iree_hal_fixed_block_pool_options_t options = {};
    options.block_allocator_options.block_size = block_size;
    options.block_allocator_options.block_count = 1;
    options.block_allocator_options.frontier_capacity = 2;
    return iree_hal_fixed_block_pool_create(
        options, backend.slab_provider, backend.notification,
        iree_hal_pool_epoch_query_null(), host_allocator_, out_pool);
  }

  static iree_hal_pool_reservation_request_t PoolRequest(
      const iree_hal_queue_family_t* family,
      iree_device_size_t allocation_size = kTransferSize) {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
    params.queue_family_affinity = iree_hal_make_queue_family_affinity(
        iree_hal_queue_family_ordinal(family));
    return iree_hal_pool_reservation_request_t{
        /*.params=*/params,
        /*.allocation_size=*/allocation_size,
    };
  }

  static iree_status_t WarmPool(
      iree_hal_pool_t* pool,
      const iree_hal_pool_reservation_request_t& request) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
        pool, request.params, request.allocation_size,
        /*requester_frontier=*/nullptr, iree_infinite_timeout(), &buffer));
    iree_hal_buffer_release(buffer);
    return iree_ok_status();
  }

  static iree_hal_semaphore_list_t SemaphoreList(
      iree_hal_semaphore_t** semaphore, uint64_t* value) {
    return {1, semaphore, value};
  }

  static QueuePublicationSnapshot InstallStickyFailureWhilePublisherPaused(
      iree_hal_amdgpu_host_queue_t* host_queue,
      PublisherSubmissionLatch* publisher_latch,
      QueuePhaseLatch* failure_latch) {
    publisher_latch->WaitUntilEntered();
    const QueuePublicationSnapshot before =
        SnapshotQueuePublication(host_queue);

    ThreadCompletion failure_completion;
    std::thread failure([&] {
      iree_hal_amdgpu_host_queue_record_failure(
          host_queue, iree_status_from_code(IREE_STATUS_DATA_LOSS));
      failure_completion.MarkDone();
    });
    failure_latch->WaitUntilEntered();
    EXPECT_FALSE(failure_completion.IsDone());
    failure_latch->Release();
    failure_completion.Wait();
    failure.join();

    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    EXPECT_TRUE(host_queue->is_shutting_down);
    EXPECT_NE(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue), 0);
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
    return before;
  }

  static iree_hal_amdgpu_host_queue_t* HostQueue(iree_hal_queue_t* queue) {
    EXPECT_TRUE(iree_hal_amdgpu_host_queue_isa(queue));
    return reinterpret_cast<iree_hal_amdgpu_host_queue_t*>(queue);
  }

  void RunQueueAxisFrontierCallbackBeforeReleaseTest(bool force_failure) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device,
                                    /*notification_capacity=*/8));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
    Ref<iree_hal_semaphore_t> span_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), span_signal.out()));

    hsa_signal_t blocker_signal = iree_hsa_signal_null();
    IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                              nullptr, 0, &blocker_signal));

    CompletionPhaseObservations phases;
    phases.queue = host_queue;
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        CompletionPhaseObservations::Observe, &phases);

    CompletionResourceObservations resource_observations;
    CompletionResource* resource =
        CreateCompletionResource(&resource_observations);
    RingActionObservations action_observations;
    action_observations.queue = host_queue;
    action_observations.sequence = &phases.sequence;
    iree_hal_resource_t* operation_resources[1] = {&resource->resource};

    uint64_t first_value = 1;
    uint64_t later_value = 2;
    iree_hal_semaphore_t* span_signal_pointer = span_signal.get();
    const iree_hal_semaphore_list_t first_signal_list =
        SemaphoreList(&span_signal_pointer, &first_value);
    const iree_hal_semaphore_list_t later_signal_list =
        SemaphoreList(&span_signal_pointer, &later_value);
    const uint32_t first_pattern = 0x46524F4Eu;
    const uint32_t later_pattern = 0x54494552u;

    QueueAxisFrontierWaiter waiter_state;
    waiter_state.libhsa = &libhsa_;
    waiter_state.queue = host_queue;
    waiter_state.phases = &phases;
    waiter_state.resource_observations = &resource_observations;
    waiter_state.later_semaphore = span_signal.get();
    waiter_state.later_value = later_value;
    waiter_state.blocker_signal = blocker_signal;
    iree_async_single_frontier_t callback_frontier;

    bool action_enqueued = false;
    iree_status_t setup_status = iree_ok_status();
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    setup_status = iree_hal_amdgpu_host_queue_enqueue_host_action(
        host_queue, iree_hal_semaphore_list_empty(),
        iree_hal_amdgpu_reclaim_action_t{
            /*.fn=*/RingActionObservations::Run,
            /*.user_data=*/&action_observations,
        },
        operation_resources, IREE_ARRAYSIZE(operation_resources));
    action_enqueued = iree_status_is_ok(setup_status);
    if (iree_status_is_ok(setup_status)) {
      setup_status = iree_hal_amdgpu_host_queue_fill(
          host_queue, iree_hal_semaphore_list_empty(), first_signal_list,
          buffer, 0, 4 * sizeof(first_pattern), first_pattern,
          sizeof(first_pattern), IREE_HAL_FILL_FLAG_NONE);
    }
    const uint64_t callback_epoch =
        host_queue->notification_ring.epoch.next_submission;
    if (iree_status_is_ok(setup_status)) {
      setup_status = EnqueueRawBlockingBarrier(host_queue, blocker_signal);
    }
    if (iree_status_is_ok(setup_status)) {
      setup_status = iree_hal_amdgpu_host_queue_fill(
          host_queue, iree_hal_semaphore_list_empty(), later_signal_list,
          buffer, 0, 4 * sizeof(later_pattern), later_pattern,
          sizeof(later_pattern), IREE_HAL_FILL_FLAG_NONE);
    }
    const uint64_t final_epoch =
        host_queue->notification_ring.epoch.next_submission;
    const uint64_t final_notification_write =
        QueueAxisFrontierWaiter::LoadCursor(
            &host_queue->notification_ring.write);
    bool later_notification_is_published = final_notification_write != 0;
    uint64_t later_notification_epoch = 0;
    uint64_t later_notification_value = 0;
    bool later_notification_semaphore_matches = false;
    if (later_notification_is_published) {
      const iree_hal_amdgpu_notification_entry_t* later_notification =
          &host_queue->notification_ring
               .entries[(final_notification_write - 1) &
                        (host_queue->notification_ring.capacity - 1)];
      later_notification_epoch = later_notification->submission_epoch;
      later_notification_value = later_notification->timeline_value;
      later_notification_semaphore_matches =
          later_notification->semaphore ==
          reinterpret_cast<iree_async_semaphore_t*>(span_signal.get());
      later_notification_is_published =
          later_notification_epoch == final_epoch &&
          later_notification_semaphore_matches &&
          later_notification_value == later_value;
    }
    const size_t pending_after_later_submission =
        CountPendingOperations(host_queue);
    if (iree_status_is_ok(setup_status) &&
        (final_epoch <= callback_epoch || pending_after_later_submission != 0 ||
         !later_notification_is_published)) {
      setup_status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "later frontier-test fill and notification were not published "
          "behind the blocker (callback_epoch=%" PRIu64 ", final_epoch=%" PRIu64
          ", pending=%zu, write=%" PRIu64 ", entry_epoch=%" PRIu64
          ", entry_value=%" PRIu64 ", semaphore_matches=%d)",
          callback_epoch, final_epoch, pending_after_later_submission,
          final_notification_write, later_notification_epoch,
          later_notification_value,
          later_notification_semaphore_matches ? 1 : 0);
    }
    waiter_state.later_epoch = final_epoch;
    waiter_state.initial_epoch = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.epoch.last_drained);
    waiter_state.initial_read = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.read);
    waiter_state.initial_frontier_read = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.frontier_ring.read);
    iree_async_single_frontier_initialize(&callback_frontier, host_queue->axis,
                                          callback_epoch);
    if (iree_status_is_ok(setup_status)) {
      setup_status = iree_async_frontier_tracker_wait(
          host_queue->frontier_tracker,
          iree_async_single_frontier_as_const_frontier(&callback_frontier),
          QueueAxisFrontierWaiter::Callback, &waiter_state,
          &waiter_state.waiter);
    }
    if (action_enqueued) {
      // The ring now owns the only remaining reference. Its destruction is a
      // direct witness for the first lane-E release.
      iree_hal_resource_release(&resource->resource);
      resource = nullptr;
    }
    if (iree_status_is_ok(setup_status)) {
      WaitForHardwareEpoch(&libhsa_, host_queue, callback_epoch);
      if (force_failure) {
        iree_hal_amdgpu_host_queue_record_failure(
            host_queue, iree_status_from_code(IREE_STATUS_DATA_LOSS));
      }
    }
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

    if (!iree_status_is_ok(setup_status)) {
      if (resource) iree_hal_resource_release(&resource->resource);
      iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
      iree_hal_amdgpu_host_queue_seal(host_queue);
      iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
      IREE_EXPECT_OK(
          iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
      IREE_ASSERT_OK(setup_status);
      return;
    }

    // The outer test harness is the deadlock detector for the callback's
    // nested semaphore wait.
    waiter_state.completed.Wait();
    resource_observations.destroyed.Wait();
    phases.WaitForCount(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PUBLIC_CURSORS_FINALIZED, 1);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    const iree_status_code_t expected_status =
        force_failure ? IREE_STATUS_DATA_LOSS : IREE_STATUS_OK;
    EXPECT_EQ(waiter_state.call_count.load(), 1u);
    EXPECT_EQ(waiter_state.status_code.load(), expected_status);
    EXPECT_EQ(waiter_state.nested_wait_status_code.load(), expected_status);
    EXPECT_GT(waiter_state.feedback_phase_count_at_entry.load(), 0u);
    EXPECT_GT(waiter_state.dispatch_phase_count_at_entry.load(), 0u);
    if (force_failure) {
      // Terminal failure prepared and dispatched the entire published prefix
      // before failing the queue axis. Reentry still pumps the shared claim,
      // but there is no newly dispatched hot-ring entry to count.
      EXPECT_EQ(waiter_state.nested_drain_count.load(), 0u);
      EXPECT_EQ(waiter_state.nested_claimed_phase_delta.load(), 0u);
      EXPECT_EQ(waiter_state.nested_transition_phase_delta.load(), 0u);
      EXPECT_EQ(waiter_state.nested_prepare_phase_delta.load(), 0u);
      EXPECT_EQ(waiter_state.nested_feedback_phase_delta.load(), 0u);
      EXPECT_EQ(waiter_state.nested_dispatch_phase_delta.load(), 1u);
    } else {
      // The blocker leaves exactly the later fill outside the outer claim.
      // Reentry extends lanes A-D through that now-complete epoch while lane E
      // and public cursor publication remain deferred until callback return.
      EXPECT_EQ(waiter_state.nested_drain_count.load(), 1u);
      EXPECT_EQ(waiter_state.nested_claimed_phase_delta.load(), 1u);
      EXPECT_EQ(waiter_state.nested_transition_phase_delta.load(), 1u);
      EXPECT_EQ(waiter_state.nested_prepare_phase_delta.load(), 1u);
      EXPECT_EQ(waiter_state.nested_feedback_phase_delta.load(), 1u);
      EXPECT_EQ(waiter_state.nested_dispatch_phase_delta.load(), 1u);
      EXPECT_EQ(waiter_state.nested_claimed_target.load(), final_epoch);
      EXPECT_EQ(waiter_state.nested_prepared_read.load(),
                final_notification_write);
      EXPECT_EQ(waiter_state.nested_dispatched_read.load(),
                final_notification_write);
    }
    EXPECT_TRUE(waiter_state.drain_lock_was_available.load());
    EXPECT_TRUE(waiter_state.submission_lock_was_available.load());
    EXPECT_TRUE(waiter_state.resource_was_live_at_entry.load());
    EXPECT_TRUE(waiter_state.resource_was_live_after_nested_wait.load());
    EXPECT_TRUE(waiter_state.cursors_were_frozen_at_entry.load());
    EXPECT_TRUE(waiter_state.cursors_were_frozen_after_nested_wait.load());
    EXPECT_TRUE(waiter_state.releases_had_not_started.load());
    EXPECT_TRUE(waiter_state.public_commit_had_not_started.load());
    EXPECT_EQ(resource_observations.destroy_count.load(), 1u);
    EXPECT_EQ(action_observations.call_count.load(), 1u);
    EXPECT_EQ(action_observations.status_code.load(), expected_status);
    EXPECT_TRUE(action_observations.completion_lock_was_available.load());

    const uint64_t claimed_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED);
    const uint64_t transition_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TRANSITIONS_DONE);
    const uint64_t prepared_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_VALUES_PREPARED);
    const uint64_t feedback_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_FEEDBACK_TARGET_CLOSED);
    const uint64_t dispatch_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED);
    const uint64_t release_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_RELEASES_DONE);
    const uint64_t commit_sequence = phases.first_sequence(
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PUBLIC_CURSORS_FINALIZED);
    EXPECT_GT(claimed_sequence, 0u);
    EXPECT_LT(claimed_sequence, action_observations.action_sequence.load());
    EXPECT_LT(action_observations.action_sequence.load(), transition_sequence);
    EXPECT_LT(transition_sequence, prepared_sequence);
    EXPECT_LT(prepared_sequence, feedback_sequence);
    EXPECT_LT(feedback_sequence, dispatch_sequence);
    EXPECT_LT(dispatch_sequence, waiter_state.callback_sequence.load());
    EXPECT_LT(waiter_state.callback_sequence.load(), release_sequence);
    EXPECT_LT(release_sequence, commit_sequence);
    EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                  &host_queue->notification_ring.epoch.last_drained),
              final_epoch);
    EXPECT_GT(QueueAxisFrontierWaiter::LoadCursor(
                  &host_queue->notification_ring.read),
              waiter_state.initial_read);

    if (force_failure) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                            iree_hal_semaphore_wait(span_signal, later_value,
                                                    iree_infinite_timeout(),
                                                    IREE_ASYNC_WAIT_FLAG_NONE));
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(span_signal, later_value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }

    iree_hal_amdgpu_host_queue_seal(host_queue);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    IREE_EXPECT_OK(
        iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunCompletionEpilogueSealJoinTest(
      iree_hal_amdgpu_host_queue_test_phase_t blocked_phase) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    CompletionRunnerLifecycleLatch lifecycle_latch(
        host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_SAFE_EPILOGUE,
        blocked_phase, /*blocked_allow_closed=*/-1);
    QueuePhaseLatch closed_latch(
        host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_RUNNER_CLOSED_INSTALLED);
    closed_latch.Release();
    SealOwnerLatch seal_owner_latch(host_queue);
    QueuePhaseObserverGroup phase_observers;
    phase_observers.Add(CompletionRunnerLifecycleLatch::Observe,
                        &lifecycle_latch);
    phase_observers.Add(QueuePhaseLatch::Observe, &closed_latch);
    phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        QueuePhaseObserverGroup::Observe, &phase_observers);

    std::atomic<iree_host_size_t> drain_count{IREE_HOST_SIZE_MAX};
    ThreadCompletion runner_completion;
    std::thread runner([&] {
      drain_count.store(
          iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue));
      runner_completion.MarkDone();
    });
    lifecycle_latch.WaitUntilEntered();

    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_FALSE(host_queue->completion.runner_active);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
    EXPECT_FALSE(runner_completion.IsDone());

    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    seal_owner_latch.WaitUntilEntered();
    closed_latch.WaitUntilEntered();
    EXPECT_FALSE(runner_completion.IsDone());
    EXPECT_FALSE(seal_completion.IsDone());
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_EQ(host_queue->completion.runner_state,
              IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_CLOSED);
    EXPECT_FALSE(host_queue->completion.runner_active);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

    if (blocked_phase ==
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_RELEASE_BEFORE_EPILOGUE) {
      EXPECT_EQ(lifecycle_latch.target_after_runner_release_count(), 1u);
      EXPECT_EQ(lifecycle_latch.target_before_token_drop_count(), 0u);
    } else {
      EXPECT_EQ(
          blocked_phase,
          IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_EPILOGUE_TOKEN_DROP);
      EXPECT_EQ(lifecycle_latch.target_after_runner_release_count(), 1u);
      EXPECT_EQ(lifecycle_latch.target_before_token_drop_count(), 1u);
    }

    lifecycle_latch.Release();
    runner_completion.Wait();
    seal_completion.Wait();
    runner.join();
    sealer.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    EXPECT_EQ(drain_count.load(), 0u);
    EXPECT_EQ(lifecycle_latch.target_after_runner_release_count(), 1u);
    EXPECT_EQ(lifecycle_latch.target_before_token_drop_count(), 1u);
    EXPECT_EQ(lifecycle_latch.terminal_before_claim_count(), 1u);
    EXPECT_EQ(lifecycle_latch.terminal_after_claim_count(), 1u);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunFinalDeviceReleaseFromCompletionTest(bool cooperative_queue) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> cooperative_queue_ref;
    iree_hal_queue_t* queue = nullptr;
    if (cooperative_queue) {
      IREE_ASSERT_OK(
          test_device.AcquireCooperativeQueue(cooperative_queue_ref.out()));
      queue = cooperative_queue_ref.get();
    } else {
      queue = test_device.provisioned_queue();
      ASSERT_NE(queue, nullptr);
    }
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue);

    // Keep the callback behind a real queue dependency while every external
    // queue/device/pool edge is transferred or dropped.
    hsa_signal_t blocker_signal = iree_hsa_signal_null();
    IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                              nullptr, 0, &blocker_signal));
    IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

    CompletionSelfReleaseHostCallState release_state;
    CompletionPhaseObservations phase_observations;
    phase_observations.queue = host_queue;
    const uint64_t args[4] = {0, 0, 0, 0};
    IREE_ASSERT_OK(iree_hal_queue_host_call(
        queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
        iree_hal_make_host_call(CompletionSelfReleaseHostCallState::Call,
                                &release_state),
        args, IREE_HAL_HOST_CALL_FLAG_NONE));
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        CompletionPhaseObservations::Observe, &phase_observations);

    iree_hal_device_t* device = nullptr;
    iree_hal_device_group_t* device_group = nullptr;
    test_device.TakeDeviceOwnership(&device, &device_group);
    release_state.device = device;
    cooperative_queue_ref.reset();
    test_device.ReleaseFixtureProactorPoolReference();
    iree_hal_device_group_release(device_group);

    iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
    release_state.after_release_gate.WaitForReadyCount(1);
    EXPECT_EQ(release_state.call_count.load(), 1u);
    EXPECT_EQ(release_state.release_return_count.load(), 1u);
    EXPECT_EQ(release_state.callback_return_count.load(), 0u);
    // The completion service's non-resurrecting lifetime claim must keep the
    // device, all parent-owned queue storage, and the proactor pool live until
    // the callback has actually returned.
    EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 0u);

    release_state.after_release_gate.Open();
    controller.WaitForProactorDestroyCount(1);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
    EXPECT_EQ(release_state.callback_return_count.load(), 1u);
    EXPECT_GE(phase_observations.count(
                  IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED),
              1u);
    EXPECT_GE(
        phase_observations.count(
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PUBLIC_CURSORS_FINALIZED),
        1u);

    const ControlledProactorSnapshot final_snapshot = controller.Snapshot();
    EXPECT_EQ(final_snapshot.proactor_destroy_count, 1u);
    EXPECT_EQ(final_snapshot.destroy_with_pending_count, 0u);
    EXPECT_EQ(final_snapshot.pending_count, 0u);
    IREE_EXPECT_OK(
        iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
  }

  void RunCommandBufferOwnerCaptureCleanupTest(
      CommandBufferOwnerCaptureCase test_case) {
    const bool is_replay =
        test_case == CommandBufferOwnerCaptureCase::kRetainedAqlReplay ||
        test_case == CommandBufferOwnerCaptureCase::kUnretainedAqlReplay;
    const bool is_pm4 =
        test_case == CommandBufferOwnerCaptureCase::kPm4DynamicFixup ||
        test_case == CommandBufferOwnerCaptureCase::kPm4Profiled;
    const bool is_profiled =
        test_case == CommandBufferOwnerCaptureCase::kPm4Profiled;

    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.preallocate_pools = 0;
    options.command_buffer_mode = is_pm4
                                      ? IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4
                                      : IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
    if (is_replay) {
      options.host_block_pools.command_buffer.usable_block_size =
          IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
    }
    if (is_pm4) options.host_queues.upload_capacity = 64 * 1024;

    TestLogicalDevice test_device;
    IREE_ASSERT_OK(test_device.Initialize(&controller, &options,
                                          /*staging_slot_count=*/1));
    iree_hal_amdgpu_physical_device_t* physical_device =
        test_device.first_physical_device();
    ASSERT_NE(physical_device, nullptr);
    if (is_pm4 &&
        !iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            physical_device->vendor_packet_capabilities)) {
      GTEST_SKIP() << "PM4 dispatch command buffers are not supported on this "
                      "physical device";
    }
    if (is_profiled && !iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
                           physical_device->pm4_timestamp_strategy)) {
      GTEST_SKIP() << "PM4 dispatch timestamp packets are not supported on "
                      "this physical device";
    }

    test::CommandBufferProfileSink profile_sink = {};
    test::CommandBufferProfileSinkInitialize(&profile_sink);
    test::DeviceProfilingScope profiling(test_device.base_device());
    if (is_profiled) {
      iree_status_t status =
          profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                          test::CommandBufferProfileSinkAsBase(&profile_sink));
      if (!iree_status_is_ok(status) &&
          test::IsProfilingUnsupported(iree_status_code(status))) {
        iree_status_free(status);
        GTEST_SKIP() << "dispatch profiling is not supported on this device";
      }
      IREE_ASSERT_OK(status);
    }

    Ref<iree_hal_queue_t> dynamic_queue;
    iree_hal_queue_t* queue = nullptr;
    if (is_profiled) {
      queue = test_device.provisioned_queue();
      ASSERT_NE(queue, nullptr);
    } else {
      IREE_ASSERT_OK(test_device.AcquireDynamicQueue(dynamic_queue.out()));
      queue = dynamic_queue.get();
    }
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue);

    // The intentionally foreign heap binding must pass the generic HAL
    // binding validator so the backend first captures its cleanup owner and
    // then fails during AMDGPU pointer resolution at the observed boundary.
    iree_hal_command_buffer_mode_t command_buffer_mode =
        IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    if (test_case == CommandBufferOwnerCaptureCase::kUnretainedAqlReplay) {
      command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
    }
    if (is_profiled) {
      command_buffer_mode |=
          IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA;
    }

    Ref<iree_hal_executable_t> executable;
    if (is_pm4) {
      IREE_ASSERT_OK(test::LoadCtsExecutable(
          test_device.base_device(), iree_hal_queue_family(queue),
          IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
          executable.out()));
    }

    const iree_host_size_t binding_count = is_pm4 ? 2 : 1;
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        test_device.base_device(), iree_hal_queue_family(queue),
        command_buffer_mode,
        is_pm4 ? IREE_HAL_COMMAND_CATEGORY_DISPATCH
               : IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        binding_count, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    if (is_pm4) {
      iree_hal_buffer_ref_t dispatch_bindings[2] = {
          iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                            4 * sizeof(uint32_t)),
          iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/0,
                                            4 * sizeof(uint32_t)),
      };
      const iree_hal_buffer_ref_list_t dispatch_binding_list = {
          /*.count=*/IREE_ARRAYSIZE(dispatch_bindings),
          /*.values=*/dispatch_bindings,
      };
      IREE_ASSERT_OK(test::AppendConstantsBindingsDispatch(
          command_buffer, executable, dispatch_binding_list));
    } else {
      const uint32_t fill_count = is_replay ? 32 : 1;
      for (uint32_t i = 0; i < fill_count; ++i) {
        const uint32_t pattern = 0xC4A70000u | i;
        IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
            command_buffer,
            iree_hal_make_indirect_buffer_ref(
                /*buffer_slot=*/0, i * sizeof(pattern), sizeof(pattern)),
            &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
      }
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    iree_hal_amdgpu_host_queue_test_command_buffer_owner_path_t expected_path;
    if (is_pm4) {
      ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
      const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
          iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
      ASSERT_NE(fixup_plan, nullptr);
      ASSERT_NE(fixup_plan->entries, nullptr);
      ASSERT_GT(fixup_plan->entry_count, 0u);
      if (is_profiled) {
        const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile_plan =
            iree_hal_amdgpu_pm4_command_buffer_profile_plan(
                command_buffer, host_queue->physical_queue_ordinal);
        ASSERT_NE(profile_plan, nullptr);
        ASSERT_NE(profile_plan->entries, nullptr);
        ASSERT_GT(profile_plan->entry_count, 0u);
        expected_path =
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_COMMAND_BUFFER_OWNER_PATH_PM4_PROFILED;
      } else {
        expected_path =
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_COMMAND_BUFFER_OWNER_PATH_PM4_DYNAMIC_FIXUP;
      }
    } else {
      ASSERT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
      const iree_hal_amdgpu_aql_program_t* program =
          iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
      ASSERT_NE(program, nullptr);
      if (is_replay) {
        ASSERT_GT(program->block_count, 1u);
        expected_path =
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_COMMAND_BUFFER_OWNER_PATH_AQL_REPLAY;
      } else {
        ASSERT_EQ(program->block_count, 1u);
        ASSERT_GT(program->max_block_aql_packet_count, 0u);
        expected_path =
            IREE_HAL_AMDGPU_HOST_QUEUE_TEST_COMMAND_BUFFER_OWNER_PATH_DIRECT_AQL;
      }
    }

    OwnerCaptureBufferStorage storage = {};
    BindingBufferReleaseObservations release_observations;
    release_observations.queue = host_queue;
    release_observations.reenter_queue = true;
    BindingBufferReleaseToken release_token;
    release_token.observations = &release_observations;
    iree_hal_buffer_t* binding_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        sizeof(storage.data),
        iree_make_byte_span(storage.data, sizeof(storage.data)),
        iree_hal_buffer_release_callback_t{
            /*.fn=*/BindingBufferReleaseToken::Release,
            /*.user_data=*/&release_token,
        },
        host_allocator_, &binding_buffer));
    std::array<iree_hal_buffer_binding_t, 2> bindings = {};
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      bindings[i] = iree_hal_buffer_binding_t{
          /*.buffer=*/binding_buffer,
          /*.offset=*/0,
          /*.length=*/IREE_HAL_WHOLE_BUFFER,
      };
    }
    const iree_hal_buffer_binding_table_t binding_table = {
        /*.count=*/binding_count,
        /*.bindings=*/bindings.data(),
    };

    const uint64_t initial_epoch =
        host_queue->notification_ring.epoch.next_submission;
    const uint64_t initial_write = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.write);
    QueuePhaseLatch capture_latch(
        host_queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMMAND_BUFFER_OWNER_CAPTURE,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_OWNER_CAPTURED_BEFORE_VALIDATION);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseLatch::Observe,
                                                       &capture_latch);
    std::atomic<iree_status_code_t> execute_status_code{IREE_STATUS_OK};
    ThreadCompletion execute_completion;
    iree_hal_command_buffer_t* command_buffer_ptr = command_buffer.get();
    std::thread execute_thread([&] {
      iree_status_t status = iree_hal_queue_execute(
          queue, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer_ptr, binding_table,
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      execute_status_code.store(iree_status_code(status));
      iree_status_free(status);
      execute_completion.MarkDone();
    });

    // The binding set (and replay object for multi-block AQL) has captured all
    // ownership while submission_mutex remains held. Dropping the caller edges
    // here must not run either finalizer before post-validation cleanup is
    // handed back to the unlocked queue epilogue.
    capture_latch.WaitUntilEntered();
    EXPECT_EQ(capture_latch.value0(), static_cast<uint64_t>(expected_path));
    EXPECT_EQ(capture_latch.value1(),
              static_cast<uint64_t>(command_buffer_mode));
    iree_hal_buffer_release(binding_buffer);
    binding_buffer = nullptr;
    if (is_replay) command_buffer.reset();
    EXPECT_EQ(release_observations.release_count.load(), 0u);
    EXPECT_FALSE(execute_completion.IsDone());

    capture_latch.Release();
    execute_completion.Wait();
    execute_thread.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    EXPECT_EQ(execute_status_code.load(), IREE_STATUS_PERMISSION_DENIED);
    EXPECT_EQ(capture_latch.occurrence_count(), 1u);
    EXPECT_EQ(release_observations.release_count.load(), 1u);
    EXPECT_EQ(release_observations.reentry_count.load(), 1u);
    EXPECT_EQ(release_observations.reentry_status_code.load(), IREE_STATUS_OK);
    EXPECT_TRUE(release_observations.all_submission_lock_checks_passed.load());
    EXPECT_EQ(release_observations.epoch_observed_under_lock.load(),
              initial_epoch);
    EXPECT_EQ(release_observations.write_observed_under_lock.load(),
              initial_write);

    command_buffer.reset();
    executable.reset();
    if (is_profiled) {
      IREE_ASSERT_OK(profiling.End());
    }
    iree_hal_amdgpu_host_queue_seal(host_queue);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    EXPECT_EQ(release_observations.release_count.load(), 1u);
    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    controller.MarkSealed();
    dynamic_queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunAllocaUnlockedWindowCloseTest(AllocaUnlockedWindow window) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
    ASSERT_NE(queue_family, nullptr);
    const iree_hal_pool_reservation_request_t request =
        PoolRequest(queue_family);

    iree_hal_pool_t* delegate_pool = nullptr;
    IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                        kTransferSize, &delegate_pool));
    IREE_ASSERT_OK(WarmPool(delegate_pool, request));
    iree_hal_pool_stats_t initial_stats;
    iree_hal_pool_query_stats(delegate_pool, &initial_stats);
    ASSERT_EQ(initial_stats.reservation_count, 0u);
    ObservingPool* observing_pool =
        CreateObservingPool(delegate_pool, host_queue);

    Ref<iree_hal_semaphore_t> readiness_semaphore;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), readiness_semaphore.out()));
    iree_hal_semaphore_t* readiness_semaphore_pointer =
        readiness_semaphore.get();
    uint64_t readiness_value = 1;
    const iree_hal_semaphore_list_t readiness_signal_list =
        SemaphoreList(&readiness_semaphore_pointer, &readiness_value);

    ThreadGate callback_gate;
    switch (window) {
      case AllocaUnlockedWindow::kNotification:
        observing_pool->notification_after_delegate_gate = &callback_gate;
        break;
      case AllocaUnlockedWindow::kAcquire:
        observing_pool->acquire_after_delegate_gate = &callback_gate;
        break;
      case AllocaUnlockedWindow::kMaterialize:
        observing_pool->materialize_after_delegate_gate = &callback_gate;
        break;
    }

    auto* const untouched = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{1});
    iree_hal_buffer_t* buffer = untouched;
    std::atomic<iree_status_code_t> alloca_status_code{IREE_STATUS_UNKNOWN};
    ThreadCompletion alloca_completion;
    std::thread alloca_thread([&] {
      iree_status_t status =
          iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(),
                                readiness_signal_list, &observing_pool->base,
                                /*request_count=*/1, &request, &buffer);
      alloca_status_code.store(iree_status_code(status));
      iree_status_free(status);
      alloca_completion.MarkDone();
    });

    callback_gate.WaitForReadyCount(1);
    EXPECT_EQ(buffer, untouched);
    EXPECT_FALSE(alloca_completion.IsDone());
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

    iree_hal_pool_stats_t gated_stats;
    iree_hal_pool_query_stats(delegate_pool, &gated_stats);
    if (window == AllocaUnlockedWindow::kNotification) {
      EXPECT_EQ(gated_stats.reservation_count, 0u);
      EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kAcquire), 0u);
    } else {
      EXPECT_EQ(gated_stats.reservation_count, 1u);
      EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kAcquire), 1u);
    }
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize),
              window == AllocaUnlockedWindow::kMaterialize ? 1u : 0u);
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 0u);

    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    const uint64_t gated_epoch =
        host_queue->notification_ring.epoch.next_submission;
    const uint64_t gated_write = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.write);
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

    SealOwnerLatch seal_owner_latch(host_queue);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(SealOwnerLatch::Observe,
                                                       &seal_owner_latch);
    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    seal_owner_latch.WaitUntilEntered();
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_FALSE(alloca_completion.IsDone());
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

    callback_gate.Open();
    alloca_completion.Wait();
    seal_completion.Wait();
    alloca_thread.join();
    sealer.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    EXPECT_EQ(alloca_status_code.load(), IREE_STATUS_CANCELLED);
    EXPECT_EQ(buffer, untouched);
    uint64_t readiness_semaphore_value = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(readiness_semaphore,
                                            &readiness_semaphore_value));
    EXPECT_EQ(readiness_semaphore_value, 0u);
    EXPECT_EQ(host_queue->notification_ring.epoch.next_submission, gated_epoch);
    EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                  &host_queue->notification_ring.write),
              gated_write);
    ExpectPoolMethodUnlocked(observing_pool,
                             ObservingPoolMethod::kQueryCapabilities, 1);
    ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kNotification,
                             1);
    ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kAcquire, 1);
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize),
              window == AllocaUnlockedWindow::kMaterialize ? 1u : 0u);
    if (window == AllocaUnlockedWindow::kMaterialize) {
      ExpectPoolMethodUnlocked(observing_pool,
                               ObservingPoolMethod::kMaterialize, 1);
    }
    ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kRelease, 1);
    EXPECT_EQ(observing_pool->reentry_status_code.load(),
              IREE_STATUS_CANCELLED);

    iree_hal_pool_stats_t final_stats;
    iree_hal_pool_query_stats(delegate_pool, &final_stats);
    EXPECT_EQ(final_stats.reservation_count, 0u);
    EXPECT_EQ(final_stats.release_count, initial_stats.release_count + 1);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    iree_hal_pool_release(&observing_pool->base);
    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunAllocaAcquireFailureRaceTest() {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
    ASSERT_NE(queue_family, nullptr);
    const iree_hal_pool_reservation_request_t request =
        PoolRequest(queue_family);

    iree_hal_pool_t* delegate_pool = nullptr;
    IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                        kTransferSize, &delegate_pool));
    IREE_ASSERT_OK(WarmPool(delegate_pool, request));
    iree_hal_pool_stats_t initial_stats;
    iree_hal_pool_query_stats(delegate_pool, &initial_stats);
    ASSERT_EQ(initial_stats.reservation_count, 0u);
    ObservingPool* observing_pool =
        CreateObservingPool(delegate_pool, host_queue);
    ThreadGate acquire_gate;
    observing_pool->acquire_after_delegate_gate = &acquire_gate;

    Ref<iree_hal_semaphore_t> readiness_semaphore;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), readiness_semaphore.out()));
    iree_hal_semaphore_t* readiness_semaphore_pointer =
        readiness_semaphore.get();
    uint64_t readiness_value = 1;
    const iree_hal_semaphore_list_t readiness_signal_list =
        SemaphoreList(&readiness_semaphore_pointer, &readiness_value);

    auto* const untouched = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{1});
    iree_hal_buffer_t* buffer = untouched;
    std::atomic<iree_status_code_t> alloca_status_code{IREE_STATUS_UNKNOWN};
    ThreadCompletion alloca_completion;
    std::thread alloca_thread([&] {
      iree_status_t status =
          iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(),
                                readiness_signal_list, &observing_pool->base,
                                /*request_count=*/1, &request, &buffer);
      alloca_status_code.store(iree_status_code(status));
      iree_status_free(status);
      alloca_completion.MarkDone();
    });

    acquire_gate.WaitForReadyCount(1);
    EXPECT_EQ(buffer, untouched);
    EXPECT_FALSE(alloca_completion.IsDone());
    iree_hal_pool_stats_t acquired_stats;
    iree_hal_pool_query_stats(delegate_pool, &acquired_stats);
    EXPECT_EQ(acquired_stats.reservation_count, 1u);
    EXPECT_EQ(acquired_stats.release_count, initial_stats.release_count);
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    const uint64_t acquired_epoch =
        host_queue->notification_ring.epoch.next_submission;
    const uint64_t acquired_write = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.write);
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

    iree_hal_amdgpu_host_queue_record_failure(
        host_queue, iree_status_from_code(IREE_STATUS_DATA_LOSS));
    EXPECT_FALSE(alloca_completion.IsDone());
    acquire_gate.Open();
    alloca_completion.Wait();
    alloca_thread.join();

    EXPECT_EQ(alloca_status_code.load(), IREE_STATUS_DATA_LOSS);
    EXPECT_EQ(buffer, untouched);
    uint64_t readiness_semaphore_value = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(readiness_semaphore,
                                            &readiness_semaphore_value));
    EXPECT_EQ(readiness_semaphore_value, 0u);
    EXPECT_EQ(host_queue->notification_ring.epoch.next_submission,
              acquired_epoch);
    EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                  &host_queue->notification_ring.write),
              acquired_write);
    ExpectPoolMethodUnlocked(observing_pool,
                             ObservingPoolMethod::kQueryCapabilities, 1);
    ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kNotification,
                             1);
    ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kAcquire, 1);
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize),
              0u);
    ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kRelease, 1);
    // Failure closes submission admission before the rollback callback. A
    // same-queue reentry must be lock-safe and preserve the recorded terminal
    // error ahead of the generic closed-admission status.
    EXPECT_EQ(observing_pool->reentry_status_code.load(),
              IREE_STATUS_DATA_LOSS);
    iree_hal_pool_stats_t final_stats;
    iree_hal_pool_query_stats(delegate_pool, &final_stats);
    EXPECT_EQ(final_stats.reservation_count, 0u);
    EXPECT_EQ(final_stats.release_count, initial_stats.release_count + 1);

    iree_hal_amdgpu_host_queue_seal(host_queue);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    iree_hal_pool_release(&observing_pool->base);
    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunDeallocaUnlockedEpilogueSealJoinTest() {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    test::CommandBufferProfileSink profile_sink = {};
    test::CommandBufferProfileSinkInitialize(&profile_sink);
    test::DeviceProfilingScope profiling(test_device.base_device());
    iree_status_t profiling_status =
        profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS,
                        test::CommandBufferProfileSinkAsBase(&profile_sink));
    if (!iree_status_is_ok(profiling_status) &&
        test::IsProfilingUnsupported(iree_status_code(profiling_status))) {
      iree_status_free(profiling_status);
      GTEST_SKIP() << "memory-event profiling is not supported on this device";
    }
    IREE_ASSERT_OK(profiling_status);

    iree_hal_queue_t* queue = test_device.provisioned_queue();
    ASSERT_NE(queue, nullptr);
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue);
    const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
    ASSERT_NE(queue_family, nullptr);
    const iree_hal_pool_reservation_request_t request =
        PoolRequest(queue_family);

    iree_hal_pool_t* delegate_pool = nullptr;
    IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                        kTransferSize, &delegate_pool));
    IREE_ASSERT_OK(WarmPool(delegate_pool, request));
    iree_hal_pool_stats_t initial_stats;
    iree_hal_pool_query_stats(delegate_pool, &initial_stats);
    ASSERT_EQ(initial_stats.reservation_count, 0u);
    ObservingPool* observing_pool =
        CreateObservingPool(delegate_pool, host_queue);

    Ref<iree_hal_semaphore_t> semaphore;
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), semaphore.out()));
    iree_hal_semaphore_t* semaphore_ptr = semaphore.get();
    uint64_t alloca_value = 1;
    uint64_t dealloca_value = 2;
    const iree_hal_semaphore_list_t alloca_signal_list =
        SemaphoreList(&semaphore_ptr, &alloca_value);
    const iree_hal_semaphore_list_t dealloca_signal_list =
        SemaphoreList(&semaphore_ptr, &dealloca_value);

    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_queue_alloca(
        queue, iree_hal_semaphore_list_empty(), alloca_signal_list,
        &observing_pool->base, /*request_count=*/1, &request, &buffer));
    ASSERT_NE(buffer, nullptr);
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, alloca_value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_pool_stats_t allocated_stats;
    iree_hal_pool_query_stats(delegate_pool, &allocated_stats);
    ASSERT_EQ(allocated_stats.reservation_count, 1u);
    ASSERT_EQ(allocated_stats.release_count, initial_stats.release_count);
    const uint32_t query_stats_count_before_dealloca =
        observing_pool->call_count(ObservingPoolMethod::kQueryStats);

    // Keep the accepted dealloca behind a real hardware dependency until its
    // unlocked release callback is parked. Completing the hardware epoch
    // before seal starts isolates the submission epilogue token as the only
    // reason seal remains blocked.
    hsa_signal_t blocker_signal = iree_hsa_signal_null();
    IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                              nullptr, 0, &blocker_signal));
    IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

    ThreadGate release_gate;
    ThreadGate profile_stats_gate;
    observing_pool->release_before_delegate_gate = &release_gate;
    observing_pool->query_stats_gate = &profile_stats_gate;
    std::atomic<iree_status_code_t> dealloca_status_code{IREE_STATUS_UNKNOWN};
    ThreadCompletion dealloca_completion;
    std::thread dealloca_thread([&] {
      iree_status_t status = iree_hal_queue_dealloca(
          queue, alloca_signal_list, dealloca_signal_list,
          /*buffer_count=*/1, &buffer);
      dealloca_status_code.store(iree_status_code(status));
      iree_status_free(status);
      dealloca_completion.MarkDone();
    });

    release_gate.WaitForReadyCount(1);
    EXPECT_FALSE(dealloca_completion.IsDone());
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 1u);
    EXPECT_TRUE(observing_pool->submission_lock_was_available(
        ObservingPoolMethod::kRelease));
    EXPECT_EQ(observing_pool->reentry_count(ObservingPoolMethod::kRelease), 1u);
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

    iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
    IREE_EXPECT_OK(iree_hal_semaphore_wait(semaphore, dealloca_value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    WaitForCompletionRunnerIdleWithEpilogueCount(host_queue, 1u);

    SealOwnerLatch seal_owner_latch(host_queue);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(SealOwnerLatch::Observe,
                                                       &seal_owner_latch);
    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    seal_owner_latch.WaitUntilEntered();
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_FALSE(dealloca_completion.IsDone());

    release_gate.Open();
    profile_stats_gate.WaitForReadyCount(1);
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_FALSE(dealloca_completion.IsDone());
    iree_hal_pool_stats_t released_stats;
    iree_hal_pool_query_stats(delegate_pool, &released_stats);
    EXPECT_EQ(released_stats.reservation_count, 0u);
    EXPECT_EQ(released_stats.release_count, initial_stats.release_count + 1);
    iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
    EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
    iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

    profile_stats_gate.Open();
    dealloca_completion.Wait();
    seal_completion.Wait();
    dealloca_thread.join();
    sealer.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    EXPECT_EQ(dealloca_status_code.load(), IREE_STATUS_OK);
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 1u);
    EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kQueryStats),
              query_stats_count_before_dealloca + 2);
    EXPECT_TRUE(observing_pool->submission_lock_was_available(
        ObservingPoolMethod::kQueryStats));
    EXPECT_EQ(observing_pool->reentry_count(ObservingPoolMethod::kQueryStats),
              observing_pool->call_count(ObservingPoolMethod::kQueryStats));
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    IREE_ASSERT_OK(profiling.End());
    EXPECT_GE(profile_sink.memory_events.size(), 5u);
    const iree_hal_profile_memory_event_t* pool_release_event = nullptr;
    const iree_hal_profile_memory_event_t* queue_dealloca_event = nullptr;
    uint32_t pool_release_event_count = 0;
    uint32_t queue_dealloca_event_count = 0;
    for (const iree_hal_profile_memory_event_t& event :
         profile_sink.memory_events) {
      if (event.pool_id != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                               &observing_pool->base))) {
        continue;
      }
      if (event.type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE) {
        pool_release_event = &event;
        ++pool_release_event_count;
      } else if (event.type ==
                 IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
        queue_dealloca_event = &event;
        ++queue_dealloca_event_count;
      }
    }
    ASSERT_EQ(pool_release_event_count, 1u);
    ASSERT_EQ(queue_dealloca_event_count, 1u);
    ASSERT_NE(pool_release_event, nullptr);
    ASSERT_NE(queue_dealloca_event, nullptr);
    EXPECT_NE(pool_release_event->submission_id, 0u);
    EXPECT_EQ(queue_dealloca_event->submission_id,
              pool_release_event->submission_id);
    EXPECT_EQ(queue_dealloca_event->frontier_entry_count,
              pool_release_event->frontier_entry_count);
    EXPECT_GT(pool_release_event->frontier_entry_count, 0u);
    EXPECT_EQ(queue_dealloca_event->backing_id, pool_release_event->backing_id);

    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(&observing_pool->base);
    IREE_EXPECT_OK(
        iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
    controller.MarkSealed();
  }

  void RunRecordedFeedbackOwnerTest(bool use_pm4, bool unretained) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.preallocate_pools = 0;
    options.feedback.enabled = true;
    options.command_buffer_mode = use_pm4
                                      ? IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4
                                      : IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
    if (use_pm4) options.host_queues.upload_capacity = 64 * 1024;

    SlowFeedbackSink sink;
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(test_device.Initialize(&controller, &options,
                                          /*staging_slot_count=*/1,
                                          sink.sink()));
    iree_hal_amdgpu_physical_device_t* physical_device =
        test_device.first_physical_device();
    ASSERT_NE(physical_device, nullptr);
    if (use_pm4 &&
        !iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            physical_device->vendor_packet_capabilities)) {
      GTEST_SKIP() << "native PM4 command-buffer execution is not supported on "
                      "this physical device";
    }

    iree_hal_amdgpu_logical_device_t* logical_device =
        test_device.logical_device();
    iree_hal_amdgpu_feedback_device_state_t* feedback_device =
        &logical_device->feedback.device_states[0];
    sink.drain_mutex = &feedback_device->drain_mutex;
    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

    Ref<iree_hal_executable_t> executable;
    IREE_ASSERT_OK(test::LoadCtsExecutable(
        test_device.base_device(), iree_hal_queue_family(queue),
        IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
        executable.out()));
    const iree_hal_amdgpu_source_context_t* source_context =
        iree_hal_amdgpu_executable_source_context(executable);
    ASSERT_NE(source_context, nullptr);
    const uint64_t source_identity =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(source_context));
    const uint64_t executable_id = iree_hal_amdgpu_executable_id(executable);

    Ref<iree_hal_buffer_t> input_buffer;
    Ref<iree_hal_buffer_t> output_buffer;
    IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
        test_device.allocator(), 4 * sizeof(uint32_t), input_buffer.out()));
    IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
        test_device.allocator(), 4 * sizeof(uint32_t), output_buffer.out()));
    const uint32_t input_values[4] = {1, 2, 3, 4};
    IREE_ASSERT_OK(iree_hal_buffer_map_write(
        input_buffer, /*target_offset=*/0, input_values, sizeof(input_values)));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                                 sizeof(input_values)),
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 sizeof(input_values)),
    };
    const iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };

    iree_hal_command_buffer_mode_t command_buffer_mode =
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
    if (unretained) {
      command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
    }
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        test_device.base_device(), iree_hal_queue_family(queue),
        command_buffer_mode, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*binding_capacity=*/0, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(test::AppendConstantsBindingsDispatch(command_buffer,
                                                         executable, bindings));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    iree_host_size_t feedback_source_count = 0;
    iree_hal_executable_t* const* feedback_sources =
        use_pm4 ? iree_hal_amdgpu_pm4_command_buffer_feedback_sources(
                      command_buffer, &feedback_source_count)
                : iree_hal_amdgpu_aql_command_buffer_feedback_sources(
                      command_buffer, &feedback_source_count);
    ASSERT_EQ(feedback_source_count, 1u);
    ASSERT_EQ(feedback_sources[0], executable.get());

    Ref<iree_hal_semaphore_t> signal;
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
    iree_hal_semaphore_t* signal_pointer = signal.get();
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_list =
        SemaphoreList(&signal_pointer, &signal_value);

    ThreadGate installed_gate;
    FeedbackPhaseRecorder feedback_phases;
    feedback_phases.batch_installed_gate = &installed_gate;
    ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
    const uint64_t initial_last_published = QueueAxisFrontierWaiter::LoadCursor(
        &host_queue->notification_ring.epoch.last_published);
    std::atomic<iree_status_code_t> execute_status_code{IREE_STATUS_UNKNOWN};
    ThreadCompletion execute_completion;
    std::thread submitter([&] {
      iree_status_t status = iree_hal_queue_execute(
          queue, iree_hal_semaphore_list_empty(), signal_list, command_buffer,
          iree_hal_buffer_binding_table_empty(),
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      execute_status_code.store(iree_status_code(status));
      iree_status_free(status);
      execute_completion.MarkDone();
    });

    installed_gate.WaitForReadyCount(1);
    EXPECT_EQ(feedback_phases.batch_installed_count.load(), 1u);
    EXPECT_EQ(feedback_phases.installed_source_identity.load(),
              source_identity);
    EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                  &host_queue->notification_ring.epoch.last_published),
              initial_last_published);
    IREE_ASSERT_OK(PublishAsanFeedback(logical_device, 0, source_identity));
    sink.gate.WaitForReadyCount(1);
    EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
    EXPECT_FALSE(execute_completion.IsDone());

    installed_gate.Open();
    execute_completion.Wait();
    submitter.join();
    EXPECT_EQ(execute_status_code.load(), IREE_STATUS_OK);
    IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    feedback_phases.WaitForPhaseCount(feedback_phases.target_closed_count, 1u);

    // In UNRETAINED mode the command buffer never owned the executable. In
    // retained mode both ordinary queue ownership and the command buffer end
    // at completion. Either way the independently installed feedback hold is
    // the sole source-metadata owner while the callback remains active.
    command_buffer.reset();
    executable.reset();
    input_buffer.reset();
    output_buffer.reset();
    iree_hal_amdgpu_host_queue_seal(host_queue);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
    EXPECT_EQ(sink.exit_count.load(), 0u);

    sink.gate.Open();
    sink.exited.Wait();
    feedback_phases.WaitForPhaseCount(feedback_phases.batch_retired_count, 1u);
    WaitForFeedbackLedgerEmpty(feedback_device);
    EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
    EXPECT_TRUE(sink.metadata_stable.load());
    EXPECT_EQ(sink.metadata.executable_id, executable_id);
    EXPECT_EQ(sink.metadata.packet_source_context, source_identity);

    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunPm4FeedbackSidecarHostOnlyTest(bool unretained) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.preallocate_pools = 0;
    options.feedback.enabled = true;
    options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;
    options.host_queues.upload_capacity = 64 * 1024;

    SlowFeedbackSink sink;
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(test_device.Initialize(&controller, &options,
                                          /*staging_slot_count=*/1,
                                          sink.sink()));
    iree_hal_amdgpu_physical_device_t* physical_device =
        test_device.first_physical_device();
    ASSERT_NE(physical_device, nullptr);
    iree_hal_amdgpu_logical_device_t* logical_device =
        test_device.logical_device();
    iree_hal_amdgpu_feedback_device_state_t* feedback_device =
        &logical_device->feedback.device_states[0];
    sink.drain_mutex = &feedback_device->drain_mutex;

    Ref<iree_hal_executable_t> executable;
    IREE_ASSERT_OK(test::LoadCtsExecutable(
        test_device.base_device(),
        iree_hal_queue_family(test_device.provisioned_queue()),
        IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
        executable.out()));
    const iree_hal_amdgpu_source_context_t* source_context =
        iree_hal_amdgpu_executable_source_context(executable);
    ASSERT_NE(source_context, nullptr);
    const uint64_t source_identity =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(source_context));
    const uint64_t executable_id = iree_hal_amdgpu_executable_id(executable);

    Ref<iree_hal_buffer_t> input_buffer;
    Ref<iree_hal_buffer_t> output_buffer;
    IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
        test_device.allocator(), 4 * sizeof(uint32_t), input_buffer.out()));
    IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
        test_device.allocator(), 4 * sizeof(uint32_t), output_buffer.out()));
    const uint32_t input_values[4] = {1, 2, 3, 4};
    IREE_ASSERT_OK(iree_hal_buffer_map_write(
        input_buffer, /*target_offset=*/0, input_values, sizeof(input_values)));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                                 sizeof(input_values)),
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 sizeof(input_values)),
    };
    const iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };

    iree_hal_command_buffer_mode_t command_buffer_mode =
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
    if (unretained) {
      command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
    }

    // gfx942 deliberately does not advertise general resident PM4 command
    // buffers. Qualify only construction/recording with the complete packet
    // set so this host-only test exercises the real PM4 sidecar without ever
    // publishing or executing the generated native stream.
    constexpr iree_hal_amdgpu_vendor_packet_capability_flags_t
        kHostOnlyPm4RecordingCapabilities =
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM |
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10 |
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT |
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_INDIRECT;
    const iree_hal_amdgpu_vendor_packet_capability_flags_t
        original_capabilities = physical_device->vendor_packet_capabilities;
    physical_device->vendor_packet_capabilities =
        (original_capabilities &
         ~IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9) |
        kHostOnlyPm4RecordingCapabilities;
    Ref<iree_hal_command_buffer_t> command_buffer;
    iree_status_t create_status = iree_hal_command_buffer_create(
        test_device.base_device(),
        iree_hal_queue_family(test_device.provisioned_queue()),
        command_buffer_mode, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*binding_capacity=*/0, command_buffer.out());
    physical_device->vendor_packet_capabilities = original_capabilities;
    IREE_ASSERT_OK(create_status);
    ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(test::AppendConstantsBindingsDispatch(command_buffer,
                                                         executable, bindings));
    IREE_ASSERT_OK(test::AppendConstantsBindingsDispatch(command_buffer,
                                                         executable, bindings));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    iree_host_size_t feedback_source_count = 0;
    iree_hal_executable_t* const* feedback_sources =
        iree_hal_amdgpu_pm4_command_buffer_feedback_sources(
            command_buffer, &feedback_source_count);
    ASSERT_EQ(feedback_source_count, 1u);
    ASSERT_EQ(feedback_sources[0], executable.get());

    FeedbackPhaseRecorder feedback_phases;
    ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);

    // A prepared batch is private and reversible until publication. This
    // covers the PM4 precommit rollback path without native execution.
    iree_hal_amdgpu_feedback_source_batch_t* cancelled_batch = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
        &logical_device->feedback, /*physical_device_ordinal=*/0,
        feedback_source_count, feedback_sources, &cancelled_batch));
    ASSERT_NE(cancelled_batch, nullptr);
    iree_hal_amdgpu_feedback_source_batch_cancel(cancelled_batch);
    EXPECT_EQ(feedback_phases.batch_installed_count.load(), 0u);
    iree_slim_mutex_lock(&feedback_device->drain_mutex);
    EXPECT_EQ(feedback_device->source_hold_head, nullptr);
    iree_slim_mutex_unlock(&feedback_device->drain_mutex);

    // Install the exact deduplicated PM4 sidecar before reserving/publishing a
    // packet. The command buffer and caller ownership can then disappear while
    // the installed batch alone keeps source metadata valid through the sink.
    iree_hal_amdgpu_feedback_source_batch_t* source_batch = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
        &logical_device->feedback, /*physical_device_ordinal=*/0,
        feedback_source_count, feedback_sources, &source_batch));
    iree_hal_amdgpu_feedback_source_batch_install(source_batch);
    ASSERT_EQ(feedback_phases.batch_installed_count.load(), 1u);
    EXPECT_EQ(feedback_phases.installed_source_identity.load(),
              source_identity);

    iree_hal_amdgpu_feedback_config_t config;
    iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
    IREE_ASSERT_OK(ReserveAsanFeedback(logical_device, 0, source_identity,
                                       &config, &packet));
    const uint64_t packet_sequence = packet->sequence;
    EXPECT_GE(packet_sequence, feedback_phases.installed_sequence.load());
    iree_hal_amdgpu_feedback_publish(&config, packet);
    iree_hsa_signal_store_screlease(
        IREE_LIBHSA(&logical_device->system->libhsa), config.notify_signal, 1);
    sink.gate.WaitForReadyCount(1);
    EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
    EXPECT_EQ(feedback_phases.claimed_source_identity.load(), source_identity);

    iree_hal_amdgpu_feedback_source_batch_close(source_batch);
    EXPECT_EQ(feedback_phases.target_closed_count.load(), 1u);
    EXPECT_EQ(feedback_phases.closed_source_identity.load(), source_identity);
    EXPECT_GT(feedback_phases.closed_sequence.load(), packet_sequence);
    command_buffer.reset();
    executable.reset();
    input_buffer.reset();
    output_buffer.reset();
    EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
    EXPECT_EQ(sink.exit_count.load(), 0u);

    sink.gate.Open();
    sink.exited.Wait();
    feedback_phases.WaitForPhaseCount(feedback_phases.batch_retired_count, 1u);
    WaitForFeedbackLedgerEmpty(feedback_device);
    EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
    EXPECT_TRUE(sink.drain_lock_was_available.load());
    EXPECT_TRUE(sink.metadata_stable.load());
    EXPECT_EQ(sink.metadata.executable_id, executable_id);
    EXPECT_EQ(sink.metadata.packet_source_context, source_identity);
  }

  void RunSealerPendingCancellationStatusTest(bool record_failure) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_NONE);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    Ref<iree_hal_semaphore_t> wait_semaphore;
    Ref<iree_hal_semaphore_t> signal_semaphore;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
    iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
    iree_hal_semaphore_t* signal_pointer = signal_semaphore.get();
    uint64_t wait_value = 1;
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t wait_list =
        SemaphoreList(&wait_pointer, &wait_value);
    const iree_hal_semaphore_list_t signal_list =
        SemaphoreList(&signal_pointer, &signal_value);

    NeverCalledHostCallState call_state;
    const uint64_t args[4] = {0, 0, 0, 0};
    IREE_ASSERT_OK(iree_hal_queue_host_call(
        queue, wait_list, signal_list,
        iree_hal_make_host_call(NeverCalledHostCallState::Call, &call_state),
        args, IREE_HAL_HOST_CALL_FLAG_NONE));
    ASSERT_EQ(CountPendingOperations(host_queue), 1u);

    // On the failure variant, hold the completion service before its claim so
    // the external sealer deterministically wins the pending cancellation.
    QueuePhaseLatch runner_latch(
        host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_RUNNER_CLAIM);
    QueuePhaseLatch cancellation_latch(
        host_queue,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_CANCELLING_LOSER,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PENDING_CANCELLING_CLAIMED);
    QueuePhaseObserverGroup phase_observers;
    if (record_failure) {
      phase_observers.Add(QueuePhaseLatch::Observe, &runner_latch);
    }
    phase_observers.Add(QueuePhaseLatch::Observe, &cancellation_latch);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        QueuePhaseObserverGroup::Observe, &phase_observers);

    if (record_failure) {
      iree_hal_amdgpu_host_queue_record_failure(
          host_queue,
          iree_make_status(IREE_STATUS_DATA_LOSS, kStickyShutdownMessage));
      runner_latch.WaitUntilEntered();
      const iree_status_t sticky_status = (iree_status_t)iree_atomic_load(
          &host_queue->error_status, iree_memory_order_acquire);
      EXPECT_EQ(iree_status_code(sticky_status), IREE_STATUS_DATA_LOSS);
      EXPECT_TRUE(StatusMentions(sticky_status, kStickyShutdownMessage));
    }

    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    cancellation_latch.WaitUntilEntered();
    if (record_failure) {
      const iree_status_t sticky_status = (iree_status_t)iree_atomic_load(
          &host_queue->error_status, iree_memory_order_acquire);
      EXPECT_EQ(iree_status_code(sticky_status), IREE_STATUS_DATA_LOSS);
      EXPECT_TRUE(StatusMentions(sticky_status, kStickyShutdownMessage));
    } else {
      EXPECT_EQ(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue),
                0);
    }
    cancellation_latch.Release();
    if (record_failure) runner_latch.Release();
    seal_completion.Wait();
    sealer.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    const iree_status_code_t expected_status =
        record_failure ? IREE_STATUS_DATA_LOSS : IREE_STATUS_CANCELLED;
    ExpectOwnedStatus(iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE),
                      expected_status,
                      record_failure ? kStickyShutdownMessage : nullptr);
    EXPECT_EQ(call_state.call_count.load(), 0u);
    EXPECT_EQ(CountPendingOperations(host_queue), 0u);
    EXPECT_EQ(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue), 0);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());

    const ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

#if IREE_FILE_IO_ENABLE
  void RunFileSealCancellationStatusTest(bool staged,
                                         bool supports_cancellation,
                                         bool record_failure,
                                         iree_status_code_t completion_code,
                                         iree_status_code_t expected_status) {
    const iree_async_proactor_capabilities_t capabilities =
        supports_cancellation
            ? IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION
            : IREE_ASYNC_PROACTOR_CAPABILITY_NONE;
    ControlledProactorController controller(capabilities);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    iree::testing::TempFilePath path;
    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(CreateNativeFile(test_device.base_device(),
                                    IREE_HAL_MEMORY_ACCESS_READ, &path,
                                    file.out()));
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(CreateBuffer(test_device.allocator(),
                                /*host_visible=*/!staged, buffer.out()));
    Ref<iree_hal_semaphore_t> signal;
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
    iree_hal_semaphore_t* signal_pointer = signal.get();
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_list =
        SemaphoreList(&signal_pointer, &signal_value);

    IREE_ASSERT_OK(iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(),
                                       signal_list, file, 0, buffer, 0,
                                       kTransferSize, IREE_HAL_READ_FLAG_NONE));
    controller.WaitForSubmitCount(1);
    const uint64_t capability_query_count =
        controller.Snapshot().capability_query_count;

    if (record_failure) {
      iree_hal_amdgpu_host_queue_record_failure(
          host_queue,
          iree_make_status(IREE_STATUS_DATA_LOSS, kStickyShutdownMessage));
    }

    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    if (supports_cancellation) {
      controller.WaitForCancelCount(1);
    } else {
      controller.WaitForCapabilityQueryAfter(capability_query_count);
    }

    if (record_failure) {
      const iree_status_t sticky_status = (iree_status_t)iree_atomic_load(
          &host_queue->error_status, iree_memory_order_acquire);
      EXPECT_EQ(iree_status_code(sticky_status), IREE_STATUS_DATA_LOSS);
      EXPECT_TRUE(StatusMentions(sticky_status, kStickyShutdownMessage));
    } else {
      EXPECT_EQ(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue),
                0);
    }

    if (completion_code == IREE_STATUS_CANCELLED) {
      controller.ReleaseNextCancelled();
    } else if (completion_code == IREE_STATUS_ABORTED) {
      controller.ReleaseNextOperationFailure();
    } else {
      ASSERT_EQ(completion_code, IREE_STATUS_OK);
      controller.ReleaseNextSuccessful();
    }
    seal_completion.Wait();
    sealer.join();

    ExpectOwnedStatus(
        iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE),
        expected_status,
        expected_status == IREE_STATUS_DATA_LOSS ? kStickyShutdownMessage
                                                 : nullptr);
    EXPECT_EQ(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue), 0);
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    file.reset();
    buffer.reset();
    controller.WaitForFileDestroyCount(1);

    const ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    EXPECT_EQ(sealed_snapshot.submit_count, 1u);
    EXPECT_EQ(sealed_snapshot.cancel_count, supports_cancellation ? 1u : 0u);
    EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
    EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
    EXPECT_EQ(sealed_snapshot.pending_count, 0u);
    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunFileSubmitTailSealJoinTest(bool staged, bool is_write) {
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    iree::testing::TempFilePath path;
    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(CreateNativeFile(
        test_device.base_device(),
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, &path,
        file.out()));
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(CreateBuffer(test_device.allocator(),
                                /*host_visible=*/!staged, buffer.out()));
    Ref<iree_hal_semaphore_t> signal;
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
    iree_hal_semaphore_t* signal_pointer = signal.get();
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_list =
        SemaphoreList(&signal_pointer, &signal_value);

    const iree_hal_amdgpu_host_queue_test_subject_t subject =
        staged
            ? (is_write
                   ? IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_WRITE_SUBMIT
                   : IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_READ_SUBMIT)
            : (is_write
                   ? IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_DIRECT_WRITE_SUBMIT
                   : IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_DIRECT_READ_SUBMIT);
    PublisherTailLatch tail_latch(host_queue, subject);
    SealOwnerLatch seal_owner_latch(host_queue);
    QueuePhaseObserverGroup phase_observers;
    phase_observers.Add(PublisherTailLatch::Observe, &tail_latch);
    phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        QueuePhaseObserverGroup::Observe, &phase_observers);

    ThreadCompletion submit_completion;
    std::atomic<iree_status_code_t> submit_status_code{IREE_STATUS_UNKNOWN};
    std::thread submitter([&] {
      iree_status_t status =
          is_write
              ? iree_hal_queue_write(queue, iree_hal_semaphore_list_empty(),
                                     signal_list, buffer, 0, file, 0,
                                     kTransferSize, IREE_HAL_WRITE_FLAG_NONE)
              : iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(),
                                    signal_list, file, 0, buffer, 0,
                                    kTransferSize, IREE_HAL_READ_FLAG_NONE);
      submit_status_code.store(iree_status_code(status));
      iree_status_free(status);
      submit_completion.MarkDone();
    });

    // The proactor owns the operation, but the publisher has only published
    // io_submit_count == 0. Its notification wake and final mutex unlock have
    // not happened, so seal may move the exact publisher to its shutdown
    // ledger but cannot cancel, join, or free it.
    tail_latch.WaitForZeroBeforeWake();
    ControlledProactorSnapshot parked_snapshot = controller.Snapshot();
    EXPECT_EQ(parked_snapshot.submit_attempt_count, 1u);
    EXPECT_EQ(parked_snapshot.submit_count, 1u);
    EXPECT_EQ(parked_snapshot.pending_count, 1u);
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    if (staged) {
      EXPECT_NE(host_queue->active_staging_transfer_head, nullptr);
      EXPECT_EQ(host_queue->shutdown_staging_transfer_head, nullptr);
    } else {
      EXPECT_NE(host_queue->active_file_action_head, nullptr);
      EXPECT_EQ(host_queue->shutdown_file_action_head, nullptr);
    }
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    seal_owner_latch.WaitUntilEntered();
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    if (staged) {
      EXPECT_NE(host_queue->active_staging_transfer_head == nullptr,
                host_queue->shutdown_staging_transfer_head == nullptr);
    } else {
      EXPECT_NE(host_queue->active_file_action_head == nullptr,
                host_queue->shutdown_file_action_head == nullptr);
    }
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_EQ(controller.Snapshot().cancel_count, 0u);

    // The wake has now happened, but TAIL_LEFT is still inside the same
    // publisher mutex. Unlock must be the submitter's final state access before
    // seal can observe the idle predicate and issue cancellation.
    tail_latch.ReleaseZeroBeforeWake();
    tail_latch.WaitForTailLeft();
    EXPECT_EQ(tail_latch.zero_before_wake_count(), 1u);
    EXPECT_EQ(tail_latch.tail_left_count(), 1u);
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_EQ(controller.Snapshot().cancel_count, 0u);

    tail_latch.ReleaseTailLeft();
    submit_completion.Wait();
    submitter.join();
    EXPECT_EQ(submit_status_code.load(), IREE_STATUS_OK);
    controller.WaitForCancelCount(1);
    EXPECT_GT(controller.Snapshot().capability_query_count,
              parked_snapshot.capability_query_count);
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    if (staged) {
      EXPECT_EQ(host_queue->active_staging_transfer_head, nullptr);
      EXPECT_NE(host_queue->shutdown_staging_transfer_head, nullptr);
    } else {
      EXPECT_EQ(host_queue->active_file_action_head, nullptr);
      EXPECT_NE(host_queue->shutdown_file_action_head, nullptr);
    }
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
    controller.ReleaseNextCancelled();
    controller.WaitForCallbackExitCount(1);
    seal_completion.Wait();
    sealer.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_CANCELLED,
        iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    file.reset();
    buffer.reset();
    controller.WaitForFileDestroyCount(1);
    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    EXPECT_EQ(sealed_snapshot.submit_attempt_count, 1u);
    EXPECT_EQ(sealed_snapshot.submit_count, 1u);
    EXPECT_EQ(sealed_snapshot.submit_failure_count, 0u);
    EXPECT_EQ(sealed_snapshot.cancel_count, 1u);
    EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
    EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
    EXPECT_EQ(sealed_snapshot.file_import_count, 1u);
    EXPECT_EQ(sealed_snapshot.file_destroy_count, 1u);
    EXPECT_EQ(sealed_snapshot.pending_count, 0u);

    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunStagedSafeActionSealJoinTest(bool is_write) {
    using Boundary = StagedSafeActionLifecycleLatch::Boundary;

    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    iree::testing::TempFilePath path;
    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(CreateNativeFile(
        test_device.base_device(),
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, &path,
        file.out()));
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(CreateBuffer(test_device.allocator(),
                                /*host_visible=*/false, buffer.out()));
    Ref<iree_hal_semaphore_t> signal;
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
    iree_hal_semaphore_t* signal_pointer = signal.get();
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_list =
        SemaphoreList(&signal_pointer, &signal_value);

    StagedSafeActionLifecycleLatch lifecycle_latch(host_queue);
    SealOwnerLatch seal_owner_latch(host_queue);
    QueuePhaseObserverGroup phase_observers;
    phase_observers.Add(StagedSafeActionLifecycleLatch::Observe,
                        &lifecycle_latch);
    phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
    iree_hal_amdgpu_host_queue_test_set_phase_observer(
        QueuePhaseObserverGroup::Observe, &phase_observers);
    IREE_ASSERT_OK(
        is_write ? iree_hal_queue_write(queue, iree_hal_semaphore_list_empty(),
                                        signal_list, buffer, 0, file, 0,
                                        kTransferSize, IREE_HAL_WRITE_FLAG_NONE)
                 : iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(),
                                       signal_list, file, 0, buffer, 0,
                                       kTransferSize, IREE_HAL_READ_FLAG_NONE));
    controller.WaitForSubmitCount(1);

    // The proactor callback has made its queue-owned continuation visible in
    // safe_action_count but has not yet performed its final notified enqueue.
    // Seal can close admission and move the transfer to the shutdown ledger,
    // but it must join this explicit owner rather than certify an empty queue.
    controller.ReleaseNextSuccessful();
    lifecycle_latch.WaitFor(Boundary::kHandoffInstalled);
    ControlledProactorSnapshot handoff_snapshot = controller.Snapshot();
    EXPECT_EQ(handoff_snapshot.callback_entry_count, 1u);
    EXPECT_EQ(handoff_snapshot.callback_exit_count, 0u);
    EXPECT_EQ(handoff_snapshot.pending_count, 0u);

    ThreadCompletion seal_completion;
    std::thread sealer([&] {
      iree_hal_amdgpu_host_queue_seal(host_queue);
      seal_completion.MarkDone();
    });
    seal_owner_latch.WaitUntilEntered();
    controller.WaitForCapabilityQueryAfter(
        handoff_snapshot.capability_query_count);
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    EXPECT_EQ(host_queue->active_staging_transfer_head, nullptr);
    EXPECT_NE(host_queue->shutdown_staging_transfer_head, nullptr);
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_EQ(controller.Snapshot().cancel_count, 0u);

    // Once the final callback-side enqueue returns, the poll callback itself
    // must be free to return even while the queue-owned safe action is paused.
    lifecycle_latch.Release(Boundary::kHandoffInstalled);
    lifecycle_latch.WaitFor(Boundary::kSafeEpilogueBegin);
    controller.WaitForCallbackExitCount(1);
    EXPECT_FALSE(seal_completion.IsDone());
    EXPECT_EQ(controller.Snapshot().callback_exit_count, 1u);

    // The safe action consumes the callback result and settles the chunk. Its
    // end marker still precedes release of the callback-owned transfer ref and
    // therefore cannot be interpreted as publisher terminal completion.
    lifecycle_latch.Release(Boundary::kSafeEpilogueBegin);
    lifecycle_latch.WaitFor(Boundary::kSafeEpilogueEnd);
    EXPECT_FALSE(seal_completion.IsDone());
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    EXPECT_NE(host_queue->shutdown_staging_transfer_head, nullptr);
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

    // The final safe-action count, notification post, and unlock are one
    // transfer-mutex transaction. Neither the zero publication nor the wake is
    // alone a certificate that the callback tail has stopped touching state.
    lifecycle_latch.Release(Boundary::kSafeEpilogueEnd);
    lifecycle_latch.WaitFor(Boundary::kTailZeroBeforeWake);
    EXPECT_FALSE(seal_completion.IsDone());
    lifecycle_latch.Release(Boundary::kTailZeroBeforeWake);
    lifecycle_latch.WaitFor(Boundary::kTailLeft);
    EXPECT_FALSE(seal_completion.IsDone());
    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    EXPECT_NE(host_queue->shutdown_staging_transfer_head, nullptr);
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

    lifecycle_latch.Release(Boundary::kTailLeft);
    seal_completion.Wait();
    sealer.join();
    iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

    EXPECT_EQ(lifecycle_latch.entry_count(Boundary::kHandoffInstalled), 1u);
    EXPECT_EQ(lifecycle_latch.entry_count(Boundary::kSafeEpilogueBegin), 1u);
    EXPECT_EQ(lifecycle_latch.entry_count(Boundary::kSafeEpilogueEnd), 1u);
    EXPECT_EQ(lifecycle_latch.entry_count(Boundary::kTailZeroBeforeWake), 1u);
    EXPECT_EQ(lifecycle_latch.entry_count(Boundary::kTailLeft), 1u);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_CANCELLED,
        iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    file.reset();
    buffer.reset();
    controller.WaitForFileDestroyCount(1);

    ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    EXPECT_EQ(sealed_snapshot.submit_attempt_count, 1u);
    EXPECT_EQ(sealed_snapshot.submit_count, 1u);
    EXPECT_EQ(sealed_snapshot.submit_failure_count, 0u);
    EXPECT_EQ(sealed_snapshot.cancel_count, 0u);
    EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
    EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
    EXPECT_EQ(sealed_snapshot.file_import_count, 1u);
    EXPECT_EQ(sealed_snapshot.file_destroy_count, 1u);
    EXPECT_EQ(sealed_snapshot.pending_count, 0u);

    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }

  void RunFilePublisherStickyFailureRaceTest(bool staged, bool gate_copy) {
    ASSERT_FALSE(gate_copy && !staged);
    ControlledProactorController controller(
        IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

    Ref<iree_hal_queue_t> queue;
    IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
    iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
    iree::testing::TempFilePath path;
    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(CreateNativeFile(
        test_device.base_device(),
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, &path,
        file.out()));
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(CreateBuffer(test_device.allocator(),
                                /*host_visible=*/!staged, buffer.out()));
    Ref<iree_hal_semaphore_t> signal;
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
    iree_hal_semaphore_t* signal_pointer = signal.get();
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_list =
        SemaphoreList(&signal_pointer, &signal_value);

    const iree_hal_amdgpu_host_queue_test_publisher_path_t publisher_path =
        gate_copy ? IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_STAGING_COPY
        : staged
            ? IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_STAGING_SIGNAL_BARRIER
            : IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_DIRECT_FILE_SIGNAL_BARRIER;
    PublisherSubmissionLatch publisher_latch(host_queue, publisher_path);
    QueuePhaseLatch failure_latch(
        host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION,
        IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_FAILURE_ADMISSION_LOCK);
    QueuePhaseObserverGroup phase_observers;
    phase_observers.Add(PublisherSubmissionLatch::Observe, &publisher_latch);
    phase_observers.Add(QueuePhaseLatch::Observe, &failure_latch);
    PublisherSubmissionObserverGuard observer_guard(
        host_queue, &publisher_latch, &failure_latch, &phase_observers);

    IREE_ASSERT_OK(iree_hal_queue_write(
        queue, iree_hal_semaphore_list_empty(), signal_list, buffer, 0, file, 0,
        kTransferSize, IREE_HAL_WRITE_FLAG_NONE));
    if (!gate_copy) {
      controller.WaitForSubmitCount(1);
      controller.ReleaseNextSuccessful();
    }

    const QueuePublicationSnapshot before =
        InstallStickyFailureWhilePublisherPaused(host_queue, &publisher_latch,
                                                 &failure_latch);
    publisher_latch.Release();

    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    if (!gate_copy) {
      controller.WaitForCallbackExitCount(1);
    }
    EXPECT_EQ(publisher_latch.occurrence_count(), 1u);
    ExpectQueuePublicationUnchanged(before,
                                    SnapshotQueuePublication(host_queue));

    observer_guard.Reset();
    ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
    file.reset();
    buffer.reset();
    controller.WaitForFileDestroyCount(1);

    const ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
    const uint64_t expected_file_submit_count = gate_copy ? 0u : 1u;
    EXPECT_EQ(sealed_snapshot.submit_attempt_count, expected_file_submit_count);
    EXPECT_EQ(sealed_snapshot.submit_count, expected_file_submit_count);
    EXPECT_EQ(sealed_snapshot.submit_failure_count, 0u);
    EXPECT_EQ(sealed_snapshot.cancel_count, 0u);
    EXPECT_EQ(sealed_snapshot.callback_entry_count, expected_file_submit_count);
    EXPECT_EQ(sealed_snapshot.callback_exit_count, expected_file_submit_count);
    EXPECT_EQ(sealed_snapshot.file_import_count, 1u);
    EXPECT_EQ(sealed_snapshot.file_destroy_count, 1u);
    EXPECT_EQ(sealed_snapshot.pending_count, 0u);

    controller.MarkSealed();
    queue.reset();
    ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  }
#endif  // IREE_FILE_IO_ENABLE

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueuePublisherLifetimeTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueuePublisherLifetimeTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueuePublisherLifetimeTest::topology_;

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsDetachedPostDrainRunnerBeforeCertification) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  PostDrainBoundaryLatch post_drain_latch(host_queue);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(PostDrainBoundaryLatch::Observe, &post_drain_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);

  ThreadGate callback_gate;
  PostDrainActionCounter action_state;
  action_state.gate = &callback_gate;
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      host_queue, &action_state.action, PostDrainActionCounter::Run,
      &action_state);

  ThreadCompletion runner_completion;
  std::thread runner([&] {
    iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue);
    runner_completion.MarkDone();
  });
  post_drain_latch.WaitForDetached();

  iree_slim_mutex_lock(&host_queue->locks.post_drain_mutex);
  EXPECT_TRUE(host_queue->post_drain.runner_active);
  EXPECT_EQ(host_queue->post_drain.head, nullptr);
  EXPECT_EQ(host_queue->post_drain.tail, nullptr);
  iree_slim_mutex_unlock(&host_queue->locks.post_drain_mutex);
  EXPECT_EQ(action_state.call_count.load(), 0u);

  // A concurrent waiter must leave the detached batch with its sole runner.
  // The sealer must instead join that runner before publishing a certificate.
  ThreadCompletion waiter_completion;
  std::thread waiter([&] {
    iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue);
    waiter_completion.MarkDone();
  });
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  waiter_completion.Wait();
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(action_state.call_count.load(), 0u);
  EXPECT_EQ(post_drain_latch.detached_count(), 1u);
  EXPECT_EQ(post_drain_latch.idle_before_wake_count(), 0u);

  post_drain_latch.ReleaseDetached();
  callback_gate.WaitForReadyCount(1);
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(action_state.call_count.load(), 0u);
  callback_gate.Open();
  post_drain_latch.WaitForIdleBeforeWake();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(action_state.call_count.load(), 1u);
  EXPECT_EQ(post_drain_latch.detached_count(), 1u);
  EXPECT_EQ(post_drain_latch.idle_before_wake_count(), 1u);

  // The observer runs after runner_active becomes false but while the runner
  // still owns post_drain_mutex. Publication of idle and the wake are one
  // lock-linearized transition, so the sealer cannot certify in this window.
  post_drain_latch.ReleaseIdleBeforeWake();
  runner_completion.Wait();
  seal_completion.Wait();
  runner.join();
  waiter.join();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(action_state.call_count.load(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealerDrainsRequeuedPostDrainActionAfterFairnessYield) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  // Block at the exact lock-linearized yield where the first callback has
  // requeued itself, runner ownership is inactive, and no waiter has yet been
  // woken. An implementation that immediately loops over the appended action
  // never publishes this queued-batch yield.
  QueuePhaseLatch yield_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_BATCH_YIELDED_BEFORE_WAKE,
      /*blocked_occurrence=*/1, /*minimum_value0=*/1);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(QueuePhaseLatch::Observe, &yield_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);

  ThreadCompletion action_completion;
  RequeuedPostDrainAction action_state;
  action_state.queue = host_queue;
  action_state.completion = &action_completion;
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      host_queue, &action_state.action, RequeuedPostDrainAction::Run,
      &action_state);

  ThreadCompletion first_pass_completion;
  std::thread first_pass([&] {
    iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue);
    first_pass_completion.MarkDone();
  });
  yield_latch.WaitUntilEntered();
  EXPECT_EQ(yield_latch.value0(), 1u);
  EXPECT_EQ(action_state.call_count.load(), 1u);

  // Make the terminal owner responsible for joining or running every batch
  // left after normal service yields. It must not certify the queue while the
  // callback-enqueued snapshot remains pending.
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());

  yield_latch.Release();
  first_pass_completion.Wait();
  first_pass.join();
  action_completion.Wait();
  seal_completion.Wait();
  sealer.join();
  EXPECT_EQ(action_state.call_count.load(), 2u);
  EXPECT_EQ(yield_latch.occurrence_count(), 1u);

  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());

  const ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       FailureSealerPendingCancellationPreservesAnnotatedStickyStatus) {
  RunSealerPendingCancellationStatusTest(/*record_failure=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       OrdinarySealerPendingCancellationReportsCancelled) {
  RunSealerPendingCancellationStatusTest(/*record_failure=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       TerminalFixedPointKeepsAnnotatedStickyStatusPublished) {
  ControlledProactorController controller(IREE_ASYNC_PROACTOR_CAPABILITY_NONE);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  QueuePhaseLatch closed_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_RUNNER_CLOSED_INSTALLED);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseLatch::Observe,
                                                     &closed_latch);

  iree_hal_amdgpu_host_queue_record_failure(
      host_queue,
      iree_make_status(IREE_STATUS_DATA_LOSS, kStickyShutdownMessage));
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  closed_latch.WaitUntilEntered();

  // Completion-runner admission is now closed and every earlier deferred pass
  // is complete. Queue a probe for the terminal sealer claim: its first pass
  // requeues itself, so only the final fixed-point pass can observe status.
  const iree_status_t published_status = (iree_status_t)iree_atomic_load(
      &host_queue->error_status, iree_memory_order_acquire);
  EXPECT_EQ(iree_status_code(published_status), IREE_STATUS_DATA_LOSS);
  EXPECT_TRUE(StatusMentions(published_status, kStickyShutdownMessage));
  RequeuedShutdownStatusProbe probe;
  probe.queue = host_queue;
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      host_queue, &probe.action, RequeuedShutdownStatusProbe::Run, &probe);
  closed_latch.Release();

  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(probe.call_count.load(), 2u);
  iree_status_t observed_status = probe.observed_status;
  probe.observed_status = iree_ok_status();
  ExpectOwnedStatus(observed_status, IREE_STATUS_DATA_LOSS,
                    kStickyShutdownMessage);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue), 0);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());

  const ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsDirectWaiterAfterCompletionRunnerClaim) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  CompletionRunnerLifecycleLatch lifecycle_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_CLAIM,
      /*blocked_allow_closed=*/0);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(CompletionRunnerLifecycleLatch::Observe,
                      &lifecycle_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);

  std::atomic<iree_host_size_t> drain_count{IREE_HOST_SIZE_MAX};
  ThreadCompletion waiter_completion;
  std::thread waiter([&] {
    drain_count.store(
        iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue));
    waiter_completion.MarkDone();
  });
  lifecycle_latch.WaitUntilEntered();
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_TRUE(host_queue->completion.runner_active);
  EXPECT_EQ(host_queue->completion.runner_state,
            IREE_HAL_AMDGPU_COMPLETION_RUNNER_STATE_RUNNING);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(waiter_completion.IsDone());
  EXPECT_FALSE(seal_completion.IsDone());

  lifecycle_latch.Release();
  waiter_completion.Wait();
  seal_completion.Wait();
  waiter.join();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(drain_count.load(), 0u);
  EXPECT_EQ(lifecycle_latch.target_after_claim_count(), 1u);
  EXPECT_EQ(lifecycle_latch.target_after_runner_release_count(), 1u);
  EXPECT_EQ(lifecycle_latch.target_before_token_drop_count(), 1u);
  EXPECT_EQ(lifecycle_latch.terminal_before_claim_count(), 1u);
  EXPECT_EQ(lifecycle_latch.terminal_after_claim_count(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       ClosedCompletionAdmissionDeniesPreClaimWaiterWithoutEpilogue) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  CompletionRunnerLifecycleLatch lifecycle_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_RUNNER_CLAIM,
      /*blocked_allow_closed=*/0);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      CompletionRunnerLifecycleLatch::Observe, &lifecycle_latch);

  std::atomic<iree_host_size_t> drain_count{IREE_HOST_SIZE_MAX};
  ThreadCompletion waiter_completion;
  std::thread waiter([&] {
    drain_count.store(
        iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue));
    waiter_completion.MarkDone();
  });
  lifecycle_latch.WaitUntilEntered();

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_completion.Wait();
  sealer.join();
  EXPECT_FALSE(waiter_completion.IsDone());
  EXPECT_EQ(lifecycle_latch.target_after_claim_count(), 0u);
  EXPECT_EQ(lifecycle_latch.target_after_runner_release_count(), 0u);
  EXPECT_EQ(lifecycle_latch.target_before_token_drop_count(), 0u);
  EXPECT_EQ(lifecycle_latch.terminal_before_claim_count(), 1u);
  EXPECT_EQ(lifecycle_latch.terminal_after_claim_count(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());

  // The paused ordinary waiter now observes CLOSED and returns an explicit
  // zero-count denial. It must not claim the runner or touch post-drain state.
  lifecycle_latch.Release();
  waiter_completion.Wait();
  waiter.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(drain_count.load(), 0u);
  EXPECT_EQ(lifecycle_latch.target_after_claim_count(), 0u);
  EXPECT_EQ(lifecycle_latch.target_after_runner_release_count(), 0u);
  EXPECT_EQ(lifecycle_latch.target_before_token_drop_count(), 0u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsCompletionWrapperBeforePostDrainEntry) {
  RunCompletionEpilogueSealJoinTest(
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_AFTER_RUNNER_RELEASE_BEFORE_EPILOGUE);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsCompletionWrapperBeforeEpilogueTokenDrop) {
  RunCompletionEpilogueSealJoinTest(
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_EPILOGUE_TOKEN_DROP);
}

TEST_F(HostQueuePublisherLifetimeTest,
       FailureAdmissionBeforeSubmitRejectsWithoutReservation) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  const uint32_t pattern = 0x434C4F53u;

  const uint64_t initial_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t initial_last_published = QueueAxisFrontierWaiter::LoadCursor(
      &host_queue->notification_ring.epoch.last_published);
  const uint64_t initial_notification_write =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  const int64_t initial_aql_write = iree_atomic_load(
      host_queue->aql_ring.write_dispatch_id, iree_memory_order_acquire);

  ThreadGate submit_gate;
  ThreadCompletion submit_completion;
  std::atomic<iree_status_code_t> submit_status_code{IREE_STATUS_UNKNOWN};
  std::thread submitter([&] {
    submit_gate.ArriveAndWait();
    iree_status_t status = iree_hal_queue_fill(
        queue, iree_hal_semaphore_list_empty(), signal_list, buffer, 0,
        sizeof(pattern), &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
    submit_status_code.store(iree_status_code(status));
    iree_status_free(status);
    submit_completion.MarkDone();
  });
  submit_gate.WaitForReadyCount(1);

  QueuePhaseLatch failure_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_FAILURE_ADMISSION_LOCK);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseLatch::Observe,
                                                     &failure_latch);
  ThreadCompletion failure_completion;
  std::thread failure([&] {
    iree_hal_amdgpu_host_queue_record_failure(
        host_queue, iree_status_from_code(IREE_STATUS_DATA_LOSS));
    failure_completion.MarkDone();
  });
  failure_latch.WaitUntilEntered();
  EXPECT_FALSE(failure_completion.IsDone());
  EXPECT_FALSE(submit_completion.IsDone());
  failure_latch.Release();
  failure_completion.Wait();
  failure.join();

  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_TRUE(host_queue->is_shutting_down);
  EXPECT_NE(iree_hal_amdgpu_host_queue_load_error_status_raw(host_queue), 0);
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  submit_gate.Open();
  submit_completion.Wait();
  submitter.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  // The recorded fatal status outranks generic closed admission even after the
  // failure service has completed. No reservation or packet was published.
  EXPECT_EQ(submit_status_code.load(), IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission, initial_epoch);
  EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.epoch.last_published),
            initial_last_published);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      initial_notification_write);
  EXPECT_EQ(iree_atomic_load(host_queue->aql_ring.write_dispatch_id,
                             iree_memory_order_acquire),
            initial_aql_write);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       AdmittedPublicationPrecedesFailureLinearizationAndClosesLaterSubmit) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  const uint32_t pattern = 0x5055424Cu;

  const uint64_t initial_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const int64_t initial_aql_write = iree_atomic_load(
      host_queue->aql_ring.write_dispatch_id, iree_memory_order_acquire);

  SubmissionPublicationRaceLatch race_latch(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      SubmissionPublicationRaceLatch::Observe, &race_latch);
  ThreadCompletion publication_completion;
  std::atomic<iree_status_code_t> publication_status_code{IREE_STATUS_UNKNOWN};
  std::thread publisher([&] {
    iree_status_t status = iree_hal_queue_fill(
        queue, iree_hal_semaphore_list_empty(), signal_list, buffer, 0,
        sizeof(pattern), &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
    publication_status_code.store(iree_status_code(status));
    iree_status_free(status);
    publication_completion.MarkDone();
  });
  race_latch.WaitForPublication();
  EXPECT_EQ(race_latch.publication_epoch(), initial_epoch + 1);
  EXPECT_FALSE(publication_completion.IsDone());

  ThreadCompletion failure_completion;
  std::thread failure([&] {
    iree_hal_amdgpu_host_queue_record_failure(
        host_queue, iree_status_from_code(IREE_STATUS_DATA_LOSS));
    failure_completion.MarkDone();
  });
  race_latch.WaitForFailureAttempt();
  // The publisher owns submission_mutex from admission through this hook, so
  // failure cannot linearize until the epoch is published and doorbelled.
  EXPECT_FALSE(publication_completion.IsDone());
  EXPECT_FALSE(failure_completion.IsDone());

  race_latch.ReleasePublication();
  publication_completion.Wait();
  failure_completion.Wait();
  publisher.join();
  failure.join();
  EXPECT_EQ(publication_status_code.load(), IREE_STATUS_OK);
  EXPECT_EQ(race_latch.publication_count(), 1u);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission,
            initial_epoch + 1);
  EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.epoch.last_published),
            initial_epoch + 1);
  EXPECT_GT(iree_atomic_load(host_queue->aql_ring.write_dispatch_id,
                             iree_memory_order_acquire),
            initial_aql_write);

  const uint64_t closed_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t closed_notification_write =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  const int64_t closed_aql_write = iree_atomic_load(
      host_queue->aql_ring.write_dispatch_id, iree_memory_order_acquire);
  iree_status_t later_status = iree_hal_queue_fill(
      queue, iree_hal_semaphore_list_empty(), signal_list, buffer, 0,
      sizeof(pattern), &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  // Sticky fatal status continues to outrank generic closure for every later
  // submission while the queue remains terminal.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, later_status);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission, closed_epoch);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      closed_notification_write);
  EXPECT_EQ(iree_atomic_load(host_queue->aql_ring.write_dispatch_id,
                             iree_memory_order_acquire),
            closed_aql_write);
  EXPECT_EQ(race_latch.publication_count(), 1u);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       CapacityPendingStartPublishesRetryBeforeFailureAdmission) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateTestDevice(&controller, &test_device, /*notification_capacity=*/1));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  ASSERT_EQ(host_queue->notification_ring.capacity, 1u);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  NeverCalledHostCallState call_state;
  const uint64_t host_call_args[4] = {0, 0, 0, 0};

  // Occupy the sole notification slot without completing it. The following
  // no-wait host call must capture a linked capacity-deferred pending
  // host-call operation through the public queue entry point.
  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

  CapacityHandoffRaceLatch handoff_latch(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      CapacityHandoffRaceLatch::Observe, &handoff_latch);
  ThreadCompletion drainer_completion;
  std::thread drainer([&] {
    (void)iree_hal_amdgpu_host_queue_drain_completions(host_queue);
    drainer_completion.MarkDone();
  });
  handoff_latch.WaitForRunnerClaim();
  EXPECT_FALSE(drainer_completion.IsDone());

  // Publish one epoch behind the raw hardware blocker. With notification
  // capacity one, this occupies the reclaim slot even though no signal
  // transition is requested and forces the next fill into the capacity-only
  // pending path. A length greater than 8 bypasses the PM4 WRITE_DATA fast
  // path, ensuring that the occupying fill is ordered behind the raw AQL
  // blocker.
  constexpr iree_device_size_t kCapacityRaceFillLength = 16;
  const uint32_t occupying_pattern = 0x4F434355u;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_fill(
      host_queue, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), buffer, 0, kCapacityRaceFillLength,
      occupying_pattern, sizeof(occupying_pattern), IREE_HAL_FILL_FLAG_NONE));
  ASSERT_EQ(host_queue->notification_ring.epoch.next_submission, 1u);
  const uint64_t last_drained = static_cast<uint64_t>(
      iree_atomic_load(&host_queue->notification_ring.epoch.last_drained,
                       iree_memory_order_acquire));
  ASSERT_EQ(host_queue->notification_ring.epoch.next_submission - last_drained,
            1u);

  std::atomic<iree_status_code_t> submit_status_code{IREE_STATUS_UNKNOWN};
  ThreadCompletion submit_completion;
  std::thread submitter([&] {
    iree_status_t status = iree_hal_queue_host_call(
        queue, iree_hal_semaphore_list_empty(), signal_list,
        iree_hal_make_host_call(NeverCalledHostCallState::Call, &call_state),
        host_call_args, IREE_HAL_HOST_CALL_FLAG_NONE);
    submit_status_code.store(iree_status_code(status));
    iree_status_free(status);
    submit_completion.MarkDone();
  });
  handoff_latch.WaitForHandoff();
  const bool submission_lock_available =
      iree_slim_mutex_try_lock(&host_queue->locks.submission_mutex);
  if (submission_lock_available) {
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  }
  EXPECT_FALSE(submission_lock_available);
  EXPECT_FALSE(submit_completion.IsDone());
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  ThreadCompletion failure_completion;
  std::thread failure([&] {
    iree_hal_amdgpu_host_queue_record_failure(
        host_queue,
        iree_make_status(IREE_STATUS_DATA_LOSS, "controlled handoff race"));
    failure_completion.MarkDone();
  });
  handoff_latch.WaitForFailureAttempt();
  EXPECT_FALSE(failure_completion.IsDone());

  // Publishing the retry while admission is still locked gives the terminal
  // service a continuation to drain before it can observe COMPLETING in the
  // cancellation scan. The cross-thread lock probe above runs while the
  // handoff observer is parked and is the exact ordering discriminator; the
  // concurrent failure additionally proves correct code blocks terminal
  // admission until the retry is visible.
  handoff_latch.ReleaseHandoff();
  submit_completion.Wait();
  failure_completion.Wait();
  submitter.join();
  failure.join();
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  handoff_latch.ReleaseRunnerClaim();
  drainer_completion.Wait();
  drainer.join();
  EXPECT_EQ(submit_status_code.load(), IREE_STATUS_OK);
  EXPECT_EQ(handoff_latch.handoff_count(), 1u);
  EXPECT_EQ(call_state.call_count.load(), 0u);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  iree_slim_mutex_lock(&host_queue->locks.post_drain_mutex);
  EXPECT_EQ(host_queue->post_drain.head, nullptr);
  EXPECT_EQ(host_queue->post_drain.tail, nullptr);
  EXPECT_FALSE(host_queue->post_drain.runner_active);
  iree_slim_mutex_unlock(&host_queue->locks.post_drain_mutex);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       PostTerminalUnresolvedWaitRejectsBeforePendingCapture) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
  uint64_t wait_value = 1;
  const iree_hal_semaphore_list_t wait_list =
      SemaphoreList(&wait_pointer, &wait_value);

  CompletionResourceObservations resource_observations;
  CompletionResource* resource =
      CreateCompletionResource(&resource_observations);
  iree_hal_resource_t* operation_resources[] = {&resource->resource};
  RingActionObservations action_observations;
  action_observations.queue = host_queue;

  // Gate the terminal sealer's sole CLOSED runner claim. Seal has already
  // joined and removed the completion thread and retired the hardware queue at
  // this point, but has not yet consumed the sticky failure status.
  CompletionRunnerLifecycleLatch terminal_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_RUNNER_CLAIM,
      /*blocked_allow_closed=*/1);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      CompletionRunnerLifecycleLatch::Observe, &terminal_latch);
  iree_hal_amdgpu_host_queue_record_failure(
      host_queue,
      iree_make_status(IREE_STATUS_DATA_LOSS, "controlled terminal state"));
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  terminal_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_EQ(host_queue->completion.thread, nullptr);
  EXPECT_TRUE(host_queue->hardware_queue_retired);
  const uint64_t closed_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t closed_write =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  iree_status_t enqueue_status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      host_queue, wait_list,
      iree_hal_amdgpu_reclaim_action_t{
          /*.fn=*/RingActionObservations::Run,
          /*.user_data=*/&action_observations,
      },
      operation_resources, IREE_ARRAYSIZE(operation_resources));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, enqueue_status);
  EXPECT_EQ(action_observations.call_count.load(), 0u);
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission, closed_epoch);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      closed_write);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  iree_slim_mutex_lock(&host_queue->locks.post_drain_mutex);
  EXPECT_EQ(host_queue->post_drain.head, nullptr);
  EXPECT_EQ(host_queue->post_drain.tail, nullptr);
  iree_slim_mutex_unlock(&host_queue->locks.post_drain_mutex);

  // Admission rejection must not retain the operation resource. The caller's
  // release is therefore its single finalization, before terminal seal resumes.
  iree_hal_resource_release(&resource->resource);
  resource_observations.destroyed.Wait();
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);

  terminal_latch.Release();
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(terminal_latch.terminal_before_claim_count(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       PostTerminalClosedAdmissionRejectsUnresolvedWaitBeforeCapture) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
  uint64_t wait_value = 1;
  const iree_hal_semaphore_list_t wait_list =
      SemaphoreList(&wait_pointer, &wait_value);

  CompletionResourceObservations resource_observations;
  CompletionResource* resource =
      CreateCompletionResource(&resource_observations);
  iree_hal_resource_t* operation_resources[] = {&resource->resource};
  RingActionObservations action_observations;
  action_observations.queue = host_queue;

  // Gate the terminal sealer after admission has closed, completion service
  // has exited, and the hardware queue has retired, but before its sole CLOSED
  // drain. A later unresolved wait must be rejected synchronously instead of
  // linking a token that no ordinary service thread can observe.
  CompletionRunnerLifecycleLatch terminal_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_RUNNER_CLAIM,
      /*blocked_allow_closed=*/1);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      CompletionRunnerLifecycleLatch::Observe, &terminal_latch);
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  terminal_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_TRUE(host_queue->is_shutting_down);
  EXPECT_TRUE(iree_status_is_ok(host_queue->error_status));
  EXPECT_EQ(host_queue->completion.thread, nullptr);
  EXPECT_TRUE(host_queue->hardware_queue_retired);
  const uint64_t closed_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t closed_write =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

  iree_status_t enqueue_status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      host_queue, wait_list,
      iree_hal_amdgpu_reclaim_action_t{
          /*.fn=*/RingActionObservations::Run,
          /*.user_data=*/&action_observations,
      },
      operation_resources, IREE_ARRAYSIZE(operation_resources));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, enqueue_status);
  EXPECT_EQ(action_observations.call_count.load(), 0u);
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission, closed_epoch);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      closed_write);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  iree_slim_mutex_lock(&host_queue->locks.post_drain_mutex);
  EXPECT_EQ(host_queue->post_drain.head, nullptr);
  EXPECT_EQ(host_queue->post_drain.tail, nullptr);
  iree_slim_mutex_unlock(&host_queue->locks.post_drain_mutex);

  // Rejection did not retain the resource; the caller's release is its one
  // finalization while the terminal drain is still parked.
  iree_hal_resource_release(&resource->resource);
  resource_observations.destroyed.Wait();
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);

  terminal_latch.Release();
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(terminal_latch.terminal_before_claim_count(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       FinalDynamicQueueRefReleaseFromOwnCompletionSelfFinalizes) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  const uint8_t queue_index = host_queue->storage.release_slot.queue_index;
  DynamicQueueSlotReleaseLatch slot_release_latch;
  slot_release_latch.Install(host_queue);
  uint32_t queue_incarnation = 0;
  iree_slim_mutex_lock(&logical_device->dynamic_queue_slots.mutex);
  queue_incarnation =
      logical_device->dynamic_queue_slots.incarnations[queue_index];
  iree_slim_mutex_unlock(&logical_device->dynamic_queue_slots.mutex);
  EXPECT_TRUE(DynamicQueueSlotIsLive(logical_device, queue_index));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

  CompletionSelfReleaseHostCallState release_state;
  const uint64_t args[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      iree_hal_make_host_call(CompletionSelfReleaseHostCallState::Call,
                              &release_state),
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  CompletionPhaseObservations phase_observations;
  phase_observations.queue = host_queue;
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      CompletionPhaseObservations::Observe, &phase_observations);

  // Transfer the only caller-owned queue edge into the callback. The dedicated
  // queue's parent-device edge remains internal and the fixture keeps the
  // device itself alive so slot retirement is externally observable.
  release_state.queue = queue.release();
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  release_state.after_release_gate.WaitForReadyCount(1);
  EXPECT_EQ(release_state.call_count.load(), 1u);
  EXPECT_EQ(release_state.release_return_count.load(), 1u);
  EXPECT_EQ(release_state.callback_return_count.load(), 0u);
  EXPECT_TRUE(DynamicQueueSlotIsLive(logical_device, queue_index));

  release_state.after_release_gate.Open();
  slot_release_latch.Wait();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(release_state.callback_return_count.load(), 1u);
  EXPECT_EQ(slot_release_latch.release_count(), 1u);
  EXPECT_EQ(slot_release_latch.queue_index(), queue_index);
  EXPECT_FALSE(DynamicQueueSlotIsLive(logical_device, queue_index));
  EXPECT_GE(phase_observations.count(
                IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_COMPLETION_CLAIMED),
            1u);
  EXPECT_GE(phase_observations.count(
                IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PUBLIC_CURSORS_FINALIZED),
            1u);
  iree_slim_mutex_lock(&logical_device->dynamic_queue_slots.mutex);
  EXPECT_EQ(logical_device->dynamic_queue_slots.incarnations[queue_index],
            queue_incarnation);
  iree_slim_mutex_unlock(&logical_device->dynamic_queue_slots.mutex);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 0u);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
}

TEST_F(HostQueuePublisherLifetimeTest,
       FinalDeviceRefReleaseFromProvisionedQueueCompletionSelfFinalizes) {
  RunFinalDeviceReleaseFromCompletionTest(/*cooperative_queue=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       FinalDeviceRefReleaseFromCachedCooperativeCompletionSelfFinalizes) {
  RunFinalDeviceReleaseFromCompletionTest(/*cooperative_queue=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SuccessfulDeferredCleanupFinalOwnerReleaseUsesLifetimeClaim) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  PendingCleanupResourceObservations cleanup_observations;
  cleanup_observations.queue = host_queue;
  PendingCleanupResource* cleanup_resource =
      CreatePendingCleanupResource(&cleanup_observations);
  CountingHostCallState call_state;
  QueuePhaseLatch retry_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_BATCH_DETACHED_BEFORE_CALLBACK);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseLatch::Observe,
                                                     &retry_latch);

  iree_hal_amdgpu_pending_op_t* operation = nullptr;
  IREE_ASSERT_OK(EnqueueCleanupOnlyDeferredHostCall(
      host_queue, iree_hal_semaphore_list_empty(),
      iree_hal_make_host_call(CountingHostCallState::Call, &call_state),
      &cleanup_resource->resource, /*force_capacity_retry=*/true, &operation));
  iree_hal_resource_release(&cleanup_resource->resource);
  retry_latch.WaitUntilEntered();
  ASSERT_NE(operation, nullptr);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);
  EXPECT_EQ(
      iree_atomic_load(&operation->lifecycle_state, iree_memory_order_acquire),
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  // One token belongs to the pending op and one to the completion wrapper
  // whose post-drain runner owns this detached retry.
  EXPECT_EQ(host_queue->completion.epilogue_count, 2u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  // Transfer every caller-owned queue/device edge into the cleanup resource.
  // The retry callback acquires its non-resurrecting claim before this
  // resource is released, so the releases must return without starting seal
  // on the same cleanup stack.
  cleanup_observations.owned_queue = queue.release();
  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  cleanup_observations.owned_device = device;
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);

  retry_latch.Release();
  cleanup_observations.gate.WaitForReadyCount(1);
  EXPECT_EQ(cleanup_observations.entry_count.load(), 1u);
  EXPECT_EQ(cleanup_observations.owner_release_return_count.load(), 1u);
  EXPECT_EQ(cleanup_observations.return_count.load(), 0u);
  EXPECT_TRUE(cleanup_observations.completion_lock_was_available.load());
  EXPECT_TRUE(cleanup_observations.submission_lock_was_available.load());
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 2u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 0u);

  cleanup_observations.gate.Open();
  cleanup_observations.returned.Wait();
  controller.WaitForProactorDestroyCount(1);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(cleanup_observations.return_count.load(), 1u);
  EXPECT_EQ(call_state.call_count.load(), 1u);
  const ControlledProactorSnapshot final_snapshot = controller.Snapshot();
  EXPECT_EQ(final_snapshot.proactor_destroy_count, 1u);
  EXPECT_EQ(final_snapshot.destroy_with_pending_count, 0u);
  EXPECT_EQ(final_snapshot.pending_count, 0u);
}

TEST_F(HostQueuePublisherLifetimeTest,
       CancellationSealJoinsPostUnlinkCleanupThroughTokenDrop) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  iree_hal_amdgpu_staging_pool_t* staging_pool = test_device.staging_pool();
  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
  uint64_t wait_value = 1;
  const iree_hal_semaphore_list_t wait_list =
      SemaphoreList(&wait_pointer, &wait_value);
  PendingCleanupResourceObservations cleanup_observations;
  cleanup_observations.queue = host_queue;
  PendingCleanupResource* cleanup_resource =
      CreatePendingCleanupResource(&cleanup_observations);
  NeverCalledHostCallState call_state;
  iree_hal_amdgpu_pending_op_t* operation = nullptr;
  IREE_ASSERT_OK(EnqueueCleanupOnlyDeferredHostCall(
      host_queue, wait_list,
      iree_hal_make_host_call(NeverCalledHostCallState::Call, &call_state),
      &cleanup_resource->resource, /*force_capacity_retry=*/false, &operation));
  iree_hal_resource_release(&cleanup_resource->resource);
  ASSERT_EQ(
      iree_atomic_load(&operation->lifecycle_state, iree_memory_order_acquire),
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);
  wait_semaphore.reset();

  iree_hal_queue_t* final_queue = queue.release();
  iree_hal_device_t* final_device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&final_device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);

  ThreadGate sealed_before_final_release;
  ThreadCompletion owner_thread_completion;
  std::thread owner_thread([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    sealed_before_final_release.ArriveAndWait();
    iree_hal_device_release(final_device);
    iree_hal_queue_release(final_queue);
    owner_thread_completion.MarkDone();
  });

  // Cancellation has unlinked the op and entered callback/resource cleanup,
  // but notification, callback mutex, arena, and the scalar token are still
  // live. Seal must remain on this stack until all of them retire.
  cleanup_observations.gate.WaitForReadyCount(1);
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  EXPECT_EQ(cleanup_observations.entry_count.load(), 1u);
  EXPECT_EQ(cleanup_observations.return_count.load(), 0u);
  EXPECT_TRUE(cleanup_observations.completion_lock_was_available.load());
  EXPECT_TRUE(cleanup_observations.submission_lock_was_available.load());
  EXPECT_FALSE(owner_thread_completion.IsDone());
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 0u);

  cleanup_observations.gate.Open();
  cleanup_observations.returned.Wait();
  sealed_before_final_release.WaitForReadyCount(1);
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 0u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  ExpectQueueCertifiedAndEmpty(host_queue, staging_pool);
  EXPECT_EQ(call_state.call_count.load(), 0u);
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();

  sealed_before_final_release.Open();
  owner_thread_completion.Wait();
  owner_thread.join();
  controller.WaitForProactorDestroyCount(1);
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       AsyncFailedWaitDestructorWinsBeforeLifetimeClaimUsesTokenFallback) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  const uint8_t queue_index = host_queue->storage.release_slot.queue_index;
  DynamicQueueSlotReleaseLatch slot_release_latch;
  slot_release_latch.Install(host_queue);
  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
  uint64_t wait_value = 1;
  const iree_hal_semaphore_list_t wait_list =
      SemaphoreList(&wait_pointer, &wait_value);
  PendingCleanupResourceObservations cleanup_observations;
  cleanup_observations.queue = host_queue;
  PendingCleanupResource* cleanup_resource =
      CreatePendingCleanupResource(&cleanup_observations);
  NeverCalledHostCallState call_state;
  iree_hal_amdgpu_pending_op_t* operation = nullptr;
  IREE_ASSERT_OK(EnqueueCleanupOnlyDeferredHostCall(
      host_queue, wait_list,
      iree_hal_make_host_call(NeverCalledHostCallState::Call, &call_state),
      &cleanup_resource->resource, /*force_capacity_retry=*/false, &operation));
  iree_hal_resource_release(&cleanup_resource->resource);
  ASSERT_EQ(
      iree_atomic_load(&operation->lifecycle_state, iree_memory_order_acquire),
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING);

  QueuePhaseLatch callback_tail_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(QueuePhaseLatch::Observe, &callback_tail_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);
  ThreadCompletion failer_completion;
  std::thread failer([&] {
    iree_hal_semaphore_fail(
        wait_semaphore,
        iree_make_status(IREE_STATUS_DATA_LOSS, "controlled async failure"));
    failer_completion.MarkDone();
  });
  callback_tail_latch.WaitUntilEntered();
  EXPECT_EQ(
      iree_atomic_load(&operation->lifecycle_state, iree_memory_order_acquire),
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);

  // Drop the final queue edge while the callback still owns its entry tail.
  // The destructor closes admission and owns storage before pending_op_fail can
  // attempt its non-resurrecting claim.
  iree_hal_queue_t* final_queue = queue.release();
  ThreadCompletion release_completion;
  std::thread releaser([&] {
    iree_hal_queue_release(final_queue);
    release_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(failer_completion.IsDone());
  EXPECT_FALSE(release_completion.IsDone());
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  callback_tail_latch.Release();
  cleanup_observations.gate.WaitForReadyCount(1);
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  EXPECT_FALSE(release_completion.IsDone());
  EXPECT_EQ(cleanup_observations.entry_count.load(), 1u);
  EXPECT_EQ(cleanup_observations.return_count.load(), 0u);
  EXPECT_TRUE(cleanup_observations.completion_lock_was_available.load());
  EXPECT_TRUE(cleanup_observations.submission_lock_was_available.load());
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  EXPECT_EQ(host_queue->completion.epilogue_count, 1u);
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);

  cleanup_observations.gate.Open();
  cleanup_observations.returned.Wait();
  failer_completion.Wait();
  release_completion.Wait();
  failer.join();
  releaser.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(cleanup_observations.return_count.load(), 1u);
  EXPECT_EQ(call_state.call_count.load(), 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_wait(wait_semaphore, wait_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  slot_release_latch.Wait();
  EXPECT_EQ(slot_release_latch.release_count(), 1u);
  EXPECT_EQ(slot_release_latch.queue_index(), queue_index);
  EXPECT_FALSE(DynamicQueueSlotIsLive(logical_device, queue_index));
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  wait_semaphore.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsOrdinaryPendingWaitCallbackStoreBeforeWake) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_semaphore_t> wait_semaphore;
  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
  iree_hal_semaphore_t* signal_pointer = signal_semaphore.get();
  uint64_t wait_value = 1;
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t wait_list =
      SemaphoreList(&wait_pointer, &wait_value);
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  CompletionResourceObservations resource_observations;
  CompletionResource* resource =
      CreateCompletionResource(&resource_observations);
  NeverCalledHostCallState call_state;
  const uint64_t args[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queue, wait_list, signal_list,
      iree_hal_make_host_call_with_resource(NeverCalledHostCallState::Call,
                                            &call_state, &resource->resource),
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  iree_hal_resource_release(&resource->resource);
  ASSERT_EQ(CountPendingOperations(host_queue), 1u);

  QueuePhaseLatch tail_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(QueuePhaseLatch::Observe, &tail_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);
  ThreadCompletion signal_completion;
  std::atomic<iree_status_code_t> signal_status_code{IREE_STATUS_UNKNOWN};
  std::thread signaler([&] {
    iree_status_t status = iree_hal_semaphore_signal(wait_semaphore, wait_value,
                                                     /*frontier=*/nullptr);
    signal_status_code.store(iree_status_code(status));
    iree_status_free(status);
    signal_completion.MarkDone();
  });
  tail_latch.WaitUntilEntered();
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);
  EXPECT_FALSE(signal_completion.IsDone());

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);

  tail_latch.Release();
  signal_completion.Wait();
  seal_completion.Wait();
  signaler.join();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(signal_status_code.load(), IREE_STATUS_OK);
  EXPECT_EQ(tail_latch.occurrence_count(), 1u);
  EXPECT_EQ(call_state.call_count.load(), 0u);
  resource_observations.destroyed.Wait();
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsAllocaMemoryWaitCallbackStoreBeforeWake) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
  ASSERT_NE(queue_family, nullptr);
  const iree_hal_pool_reservation_request_t request = PoolRequest(queue_family);

  iree_hal_pool_t* delegate_pool = nullptr;
  IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                      kTransferSize, &delegate_pool));
  IREE_ASSERT_OK(WarmPool(delegate_pool, request));
  ObservingPool* observing_pool =
      CreateObservingPool(delegate_pool, host_queue);
  observing_pool->acquire_behavior = ObservingPoolAcquireBehavior::kExhausted;

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  iree_hal_semaphore_t* signal_pointer = signal_semaphore.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(),
                                       signal_list, &observing_pool->base,
                                       /*request_count=*/1, &request, &buffer));
  ASSERT_NE(buffer, nullptr);
  controller.WaitForSubmitCount(1);
  ASSERT_EQ(CountPendingOperations(host_queue), 1u);

  QueuePhaseLatch tail_latch(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_ALLOCA_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(QueuePhaseLatch::Observe, &tail_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);
  controller.ReleaseNextSuccessful();
  tail_latch.WaitUntilEntered();
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(controller.Snapshot().callback_exit_count, 0u);

  tail_latch.Release();
  controller.WaitForCallbackExitCount(1);
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(tail_latch.occurrence_count(), 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(&observing_pool->base);
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       CancellingLoserCallbackTailPrecedesPendingOpDestruction) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_semaphore_t> wait_semaphore;
  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  iree_hal_semaphore_t* wait_pointer = wait_semaphore.get();
  iree_hal_semaphore_t* signal_pointer = signal_semaphore.get();
  uint64_t wait_value = 1;
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t wait_list =
      SemaphoreList(&wait_pointer, &wait_value);
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  CompletionResourceObservations resource_observations;
  CompletionResource* resource =
      CreateCompletionResource(&resource_observations);
  NeverCalledHostCallState call_state;
  const uint64_t args[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queue, wait_list, signal_list,
      iree_hal_make_host_call_with_resource(NeverCalledHostCallState::Call,
                                            &call_state, &resource->resource),
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  iree_hal_resource_release(&resource->resource);

  iree_hal_amdgpu_pending_op_t* operation = nullptr;
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  operation = host_queue->pending_head;
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  ASSERT_NE(operation, nullptr);
  ASSERT_NE(operation->wait_entries, nullptr);
  ASSERT_EQ(
      iree_atomic_load(&operation->lifecycle_state, iree_memory_order_acquire),
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING);

  // Pause after the semaphore has detached the timepoint but before the
  // callback can take callback_mutex. Cancellation must then claim and unlink
  // this exact operation before the callback resumes as the loser.
  QueuePhaseLatch callback_entry_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_PENDING_CALLBACK_MUTEX);
  QueuePhaseLatch cancellation_latch(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_CANCELLING_LOSER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_PENDING_CANCELLING_CLAIMED);
  QueuePhaseLatch tail_latch(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_CANCELLING_LOSER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_ZERO_BEFORE_WAKE);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(QueuePhaseLatch::Observe, &callback_entry_latch);
  phase_observers.Add(QueuePhaseLatch::Observe, &cancellation_latch);
  phase_observers.Add(QueuePhaseLatch::Observe, &tail_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);
  ThreadCompletion signal_completion;
  std::atomic<iree_status_code_t> signal_status_code{IREE_STATUS_UNKNOWN};
  std::thread signaler([&] {
    iree_status_t status = iree_hal_semaphore_signal(wait_semaphore, wait_value,
                                                     /*frontier=*/nullptr);
    signal_status_code.store(iree_status_code(status));
    iree_status_free(status);
    signal_completion.MarkDone();
  });
  callback_entry_latch.WaitUntilEntered();
  EXPECT_EQ(callback_entry_latch.value0(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(operation)));

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  cancellation_latch.WaitUntilEntered();
  EXPECT_EQ(cancellation_latch.value0(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(operation)));
  EXPECT_EQ(
      iree_atomic_load(&operation->lifecycle_state, iree_memory_order_acquire),
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING);
  EXPECT_FALSE(signal_completion.IsDone());
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);
  callback_entry_latch.Release();

  tail_latch.WaitUntilEntered();
  const bool callback_mutex_available =
      iree_slim_mutex_try_lock(&operation->callback_mutex);
  if (callback_mutex_available) {
    iree_slim_mutex_unlock(&operation->callback_mutex);
  }
  EXPECT_FALSE(callback_mutex_available);
  EXPECT_FALSE(signal_completion.IsDone());
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);

  cancellation_latch.Release();
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  EXPECT_FALSE(signal_completion.IsDone());
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);

  tail_latch.Release();
  signal_completion.Wait();
  seal_completion.Wait();
  signaler.join();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(signal_status_code.load(), IREE_STATUS_OK);
  EXPECT_EQ(tail_latch.occurrence_count(), 1u);
  EXPECT_EQ(call_state.call_count.load(), 0u);
  resource_observations.destroyed.Wait();
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealCannotCancelPartiallyRegisteredArmingWaits) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_semaphore_t> ready_semaphore;
  Ref<iree_hal_semaphore_t> blocked_semaphore;
  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), ready_semaphore.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), blocked_semaphore.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  constexpr uint64_t kWaitValue = 1;
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready_semaphore, kWaitValue,
                                           /*frontier=*/nullptr));
  iree_hal_semaphore_t* wait_semaphores[2] = {ready_semaphore.get(),
                                              blocked_semaphore.get()};
  uint64_t wait_values[2] = {kWaitValue, kWaitValue};
  const iree_hal_semaphore_list_t wait_list = {IREE_ARRAYSIZE(wait_semaphores),
                                               wait_semaphores, wait_values};
  iree_hal_semaphore_t* signal_pointer = signal_semaphore.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  CompletionResourceObservations resource_observations;
  CompletionResource* resource =
      CreateCompletionResource(&resource_observations);
  NeverCalledHostCallState call_state;
  const uint64_t args[4] = {0, 0, 0, 0};
  QueuePhaseLatch arming_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_WAIT_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_ARMING_WAITS_PARTIAL_REGISTERED);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(QueuePhaseLatch::Observe, &arming_latch);
  phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &phase_observers);
  ThreadCompletion submit_completion;
  std::atomic<iree_status_code_t> submit_status_code{IREE_STATUS_UNKNOWN};
  std::thread submitter([&] {
    iree_status_t status = iree_hal_queue_host_call(
        queue, wait_list, signal_list,
        iree_hal_make_host_call_with_resource(NeverCalledHostCallState::Call,
                                              &call_state, &resource->resource),
        args, IREE_HAL_HOST_CALL_FLAG_NONE);
    submit_status_code.store(iree_status_code(status));
    iree_status_free(status);
    submit_completion.MarkDone();
  });
  arming_latch.WaitUntilEntered();
  iree_hal_resource_release(&resource->resource);
  EXPECT_EQ(arming_latch.value0(), 1u);
  EXPECT_EQ(arming_latch.value1(), 2u);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);
  iree_hal_amdgpu_pending_op_t* operation = nullptr;
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  operation = host_queue->pending_head;
  const int32_t lifecycle_state =
      operation ? iree_atomic_load(&operation->lifecycle_state,
                                   iree_memory_order_acquire)
                : -1;
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  ASSERT_NE(operation, nullptr);
  EXPECT_EQ(lifecycle_state, IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_WAITS);

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(submit_completion.IsDone());
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(resource_observations.destroy_count.load(), 0u);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);

  arming_latch.Release();
  submit_completion.Wait();
  seal_completion.Wait();
  submitter.join();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(submit_status_code.load(), IREE_STATUS_OK);
  EXPECT_EQ(arming_latch.occurrence_count(), 1u);
  EXPECT_EQ(call_state.call_count.load(), 0u);
  resource_observations.destroyed.Wait();
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       HostCallCanWaitForLaterCompletedSameQueueSignal) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_semaphore_t> host_call_signal;
  Ref<iree_hal_semaphore_t> later_signal;
  Ref<iree_hal_semaphore_t> ready_wait_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), host_call_signal.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), later_signal.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), ready_wait_signal.out()));

  uint64_t host_call_value = 1;
  uint64_t later_value = 1;
  uint64_t ready_wait_value = 1;
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready_wait_signal, ready_wait_value,
                                           /*frontier=*/nullptr));
  iree_hal_semaphore_t* host_call_signal_pointer = host_call_signal.get();
  iree_hal_semaphore_t* later_signal_pointer = later_signal.get();
  iree_hal_semaphore_t* ready_wait_signal_pointer = ready_wait_signal.get();
  const iree_hal_semaphore_list_t host_call_signal_list =
      SemaphoreList(&host_call_signal_pointer, &host_call_value);
  const iree_hal_semaphore_list_t later_signal_list =
      SemaphoreList(&later_signal_pointer, &later_value);
  const iree_hal_semaphore_list_t ready_wait_list =
      SemaphoreList(&ready_wait_signal_pointer, &ready_wait_value);
  ReentrantHostCallState call_state = {
      .later_semaphore = later_signal.get(),
      .later_value = later_value,
  };
  const uint64_t args[4] = {0, 0, 0, 0};

  const uint64_t initial_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t initial_notification_write =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  std::atomic<uint32_t> submissions_ready{0};
  QueuePhaseLatch dispatch_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_COMPLETION_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TIMEPOINTS_DISPATCHED,
      /*blocked_occurrence=*/1,
      /*minimum_value0=*/initial_notification_write + 1, &submissions_ready);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseLatch::Observe,
                                                     &dispatch_latch);

  // Keep the completion service out until both queue epochs have completed.
  // The first reclaim callback may then wait on the signal published by the
  // later entry only if drain prepares and dispatches the whole batch before
  // running post-drain user code. A public barrier with an already-satisfied
  // nonempty wait list is forced through the hardware submission path without
  // introducing an unrelated buffer-allocation readiness wait.
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  iree_status_t host_call_status = iree_hal_queue_host_call(
      queue, iree_hal_semaphore_list_empty(), host_call_signal_list,
      iree_hal_make_host_call(ReentrantHostCallState::Call, &call_state), args,
      IREE_HAL_HOST_CALL_FLAG_NONE);
  iree_status_t barrier_status = iree_ok_status();
  if (iree_status_is_ok(host_call_status)) {
    barrier_status =
        iree_hal_queue_barrier(queue, ready_wait_list, later_signal_list,
                               IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  }
  if (iree_status_is_ok(host_call_status) &&
      iree_status_is_ok(barrier_status)) {
    EXPECT_EQ(host_queue->notification_ring.epoch.next_submission,
              initial_epoch + 2);
    EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                  &host_queue->notification_ring.write),
              initial_notification_write + 1);
    WaitForSubmittedEpoch(&libhsa_, host_queue);
    submissions_ready.store(1, std::memory_order_release);
  }
  if (!iree_status_is_ok(barrier_status)) {
    iree_hal_semaphore_fail(later_signal, iree_status_clone(barrier_status));
  }
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  IREE_ASSERT_OK(host_call_status);
  IREE_ASSERT_OK(barrier_status);

  dispatch_latch.WaitUntilEntered();
  uint64_t observed_later_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(later_signal, &observed_later_value));
  EXPECT_EQ(observed_later_value, later_value);
  uint64_t observed_host_call_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(host_call_signal, &observed_host_call_value));
  EXPECT_EQ(observed_host_call_value, 0u);
  EXPECT_EQ(call_state.call_count.load(), 0u);
  dispatch_latch.Release();

  IREE_ASSERT_OK(iree_hal_semaphore_wait(host_call_signal, host_call_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(call_state.call_count.load(), 1u);
  EXPECT_EQ(call_state.status_code.load(), IREE_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(later_signal, later_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_amdgpu_host_queue_seal(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       NormalReclaimCallbacksAndFinalResourceReleaseRunOutsideDrainLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  QueueReentrantResourceObservations resource_observations;
  QueueReentrantResource* resource =
      CreateQueueReentrantResource(host_queue, &resource_observations);
  RingActionObservations action_observations;
  action_observations.queue = host_queue;
  iree_hal_resource_t* operation_resources[1] = {&resource->resource};

  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  iree_status_t enqueue_status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      host_queue, iree_hal_semaphore_list_empty(),
      iree_hal_amdgpu_reclaim_action_t{
          /*.fn=*/RingActionObservations::Run,
          /*.user_data=*/&action_observations,
      },
      operation_resources, IREE_ARRAYSIZE(operation_resources));
  if (iree_status_is_ok(enqueue_status)) {
    WaitForSubmittedEpoch(&libhsa_, host_queue);
  }
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  iree_hal_resource_release(&resource->resource);
  IREE_ASSERT_OK(enqueue_status);

  resource_observations.destroyed.Wait();
  EXPECT_EQ(action_observations.call_count.load(), 1u);
  EXPECT_EQ(action_observations.status_code.load(), IREE_STATUS_OK);
  EXPECT_TRUE(action_observations.completion_lock_was_available.load());
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);
  EXPECT_TRUE(resource_observations.completion_lock_was_available.load());
  EXPECT_TRUE(resource_observations.submission_lock_was_available.load());

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       FaultedReclaimFailurePropagatesPendingChainOutsideDrainLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  host_queue->wait_barrier_strategy =
      IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
  Ref<iree_hal_semaphore_t> first_signal;
  Ref<iree_hal_semaphore_t> second_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), first_signal.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), second_signal.out()));

  uint64_t first_value = 1;
  uint64_t second_value = 1;
  iree_hal_semaphore_t* first_signal_pointer = first_signal.get();
  iree_hal_semaphore_t* second_signal_pointer = second_signal.get();
  const iree_hal_semaphore_list_t first_signal_list =
      SemaphoreList(&first_signal_pointer, &first_value);
  const iree_hal_semaphore_list_t second_signal_list =
      SemaphoreList(&second_signal_pointer, &second_value);
  const uint32_t pattern = 0xFA117EDu;
  const uint64_t args[4] = {0, 0, 0, 0};
  NeverCalledHostCallState call_state;
  QueueReentrantResourceObservations resource_observations;
  QueueReentrantResource* resource =
      CreateQueueReentrantResource(host_queue, &resource_observations);

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

  // Keep A unresolved while pending B arms on A. The 16-byte fill is forced
  // through AQL so the raw barrier orders it; tiny 4/8-byte fills may use a
  // PM4 WRITE_DATA path that the AQL blocker does not serialize.
  iree_status_t first_status = iree_hal_queue_fill(
      queue, iree_hal_semaphore_list_empty(), first_signal_list, buffer, 0,
      4 * sizeof(pattern), &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  iree_status_t second_status = iree_ok_status();
  if (iree_status_is_ok(first_status)) {
    second_status = iree_hal_queue_host_call(
        queue, first_signal_list, second_signal_list,
        iree_hal_make_host_call_with_resource(NeverCalledHostCallState::Call,
                                              &call_state, &resource->resource),
        args, IREE_HAL_HOST_CALL_FLAG_NONE);
  }
  const size_t pending_before_failure =
      iree_status_is_ok(second_status) ? CountPendingOperations(host_queue) : 0;

  // Complete A in hardware while excluding the completion service. The
  // recorded queue failure makes the outer ring drain fail A; A's timepoint
  // callback synchronously terminalizes B. B's final retained resource
  // recursively drains and drops a queue ref, proving all user/reentrant work
  // runs after completion_drain_mutex is released.
  iree_slim_mutex_lock(&host_queue->locks.completion_drain_mutex);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  if (iree_status_is_ok(first_status) && iree_status_is_ok(second_status)) {
    WaitForSubmittedEpoch(&libhsa_, host_queue);
    iree_hal_amdgpu_host_queue_record_failure(
        host_queue, iree_status_from_code(IREE_STATUS_DATA_LOSS));
  }
  iree_slim_mutex_unlock(&host_queue->locks.completion_drain_mutex);
  iree_hal_resource_release(&resource->resource);
  IREE_ASSERT_OK(first_status);
  IREE_ASSERT_OK(second_status);
  ASSERT_EQ(pending_before_failure, 1u);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_wait(first_signal, first_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_wait(second_signal, second_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  resource_observations.destroyed.Wait();
  EXPECT_EQ(call_state.call_count.load(), 0u);
  EXPECT_EQ(resource_observations.destroy_count.load(), 1u);
  EXPECT_TRUE(resource_observations.completion_lock_was_available.load());
  EXPECT_TRUE(resource_observations.submission_lock_was_available.load());
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       NormalFrontierCallbackPrecedesReleaseAndCursorCommit) {
  RunQueueAxisFrontierCallbackBeforeReleaseTest(/*force_failure=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       FailureFrontierCallbackPrecedesReleaseAndCursorCommit) {
  RunQueueAxisFrontierCallbackBeforeReleaseTest(/*force_failure=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsUnlockedAllocaNotificationAndRollsBackExactlyOnce) {
  RunAllocaUnlockedWindowCloseTest(AllocaUnlockedWindow::kNotification);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsUnlockedAllocaAcquireAndRollsBackExactlyOnce) {
  RunAllocaUnlockedWindowCloseTest(AllocaUnlockedWindow::kAcquire);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsUnlockedAllocaMaterializeAndRollsBackExactlyOnce) {
  RunAllocaUnlockedWindowCloseTest(AllocaUnlockedWindow::kMaterialize);
}

TEST_F(HostQueuePublisherLifetimeTest,
       QueueFailureDuringUnlockedAllocaAcquireRollsBackExactlyOnce) {
  RunAllocaAcquireFailureRaceTest();
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsDeallocaPoolAndProfileEpilogue) {
  RunDeallocaUnlockedEpilogueSealJoinTest();
}

TEST_F(HostQueuePublisherLifetimeTest,
       CompatiblePoolCallbacksAndDeallocaReleaseRunOutsideSubmissionLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
  ASSERT_NE(queue_family, nullptr);

  iree_hal_pool_t* delegate_pool = nullptr;
  IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                      kTransferSize, &delegate_pool));
  const iree_hal_pool_reservation_request_t request = PoolRequest(queue_family);
  IREE_ASSERT_OK(WarmPool(delegate_pool, request));
  ObservingPool* observing_pool =
      CreateObservingPool(delegate_pool, host_queue);

  Ref<iree_hal_semaphore_t> semaphore;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), semaphore.out()));
  iree_hal_semaphore_t* semaphore_pointer = semaphore.get();
  uint64_t alloca_value = 1;
  uint64_t dealloca_value = 2;
  const iree_hal_semaphore_list_t alloca_signal_list =
      SemaphoreList(&semaphore_pointer, &alloca_value);
  const iree_hal_semaphore_list_t dealloca_signal_list =
      SemaphoreList(&semaphore_pointer, &dealloca_value);

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queue, iree_hal_semaphore_list_empty(), alloca_signal_list,
      &observing_pool->base, /*request_count=*/1, &request, &buffer));
  ASSERT_NE(buffer, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, alloca_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_queue_dealloca(queue, alloca_signal_list,
                                         dealloca_signal_list,
                                         /*buffer_count=*/1, &buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, dealloca_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // The source-pool callback runs synchronously after the dealloca packet is
  // committed. It must be able to reenter queue completion handling and must
  // never observe submission_mutex held by the outer queue API.
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kQueryCapabilities),
            1u);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kNotification), 1u);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kAcquire), 1u);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize), 1u);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 1u);
  for (ObservingPoolMethod method : {
           ObservingPoolMethod::kQueryCapabilities,
           ObservingPoolMethod::kNotification,
           ObservingPoolMethod::kAcquire,
           ObservingPoolMethod::kMaterialize,
           ObservingPoolMethod::kRelease,
       }) {
    EXPECT_TRUE(observing_pool->submission_lock_was_available(method));
    EXPECT_EQ(observing_pool->reentry_count(method),
              observing_pool->call_count(method));
  }
  EXPECT_EQ(observing_pool->reentry_status_code.load(), IREE_STATUS_OK);

  iree_hal_buffer_release(buffer);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 1u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(&observing_pool->base, &stats);
  IREE_ASSERT_OK(iree_hal_pool_trim(&observing_pool->base));
  for (ObservingPoolMethod method : {
           ObservingPoolMethod::kQueryStats,
           ObservingPoolMethod::kTrim,
       }) {
    EXPECT_EQ(observing_pool->call_count(method), 1u);
    EXPECT_TRUE(observing_pool->submission_lock_was_available(method));
    EXPECT_EQ(observing_pool->reentry_count(method), 1u);
  }
  EXPECT_EQ(observing_pool->reentry_status_code.load(), IREE_STATUS_OK);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();

  iree_hal_pool_release(&observing_pool->base);
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       CompatiblePoolAcquireFailureRunsOutsideSubmissionLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
  ASSERT_NE(queue_family, nullptr);
  const iree_hal_pool_reservation_request_t request = PoolRequest(queue_family);

  iree_hal_pool_t* delegate_pool = nullptr;
  IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                      kTransferSize, &delegate_pool));
  IREE_ASSERT_OK(WarmPool(delegate_pool, request));
  ObservingPool* observing_pool =
      CreateObservingPool(delegate_pool, host_queue);
  observing_pool->acquire_behavior = ObservingPoolAcquireBehavior::kFail;
  observing_pool->acquire_failure_code = IREE_STATUS_UNAVAILABLE;

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  auto* const untouched = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{1});
  iree_hal_buffer_t* buffer = untouched;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(), signal_list,
                            &observing_pool->base, /*request_count=*/1,
                            &request, &buffer));
  EXPECT_EQ(buffer, untouched);
  uint64_t queried_value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal, &queried_value));
  EXPECT_EQ(queried_value, 0u);

  ExpectPoolMethodUnlocked(observing_pool,
                           ObservingPoolMethod::kQueryCapabilities, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kAcquire, 1);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize), 0u);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 0u);
  EXPECT_EQ(observing_pool->reentry_status_code.load(), IREE_STATUS_OK);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  iree_hal_pool_release(&observing_pool->base);
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       MalformedWaitReservationRollbackRunsOutsideSubmissionLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
  ASSERT_NE(queue_family, nullptr);
  const iree_hal_pool_reservation_request_t request = PoolRequest(queue_family);

  iree_hal_pool_t* delegate_pool = nullptr;
  IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                      kTransferSize, &delegate_pool));
  IREE_ASSERT_OK(WarmPool(delegate_pool, request));
  ObservingPool* observing_pool =
      CreateObservingPool(delegate_pool, host_queue);
  observing_pool->acquire_behavior =
      ObservingPoolAcquireBehavior::kMalformedWait;

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(), signal_list,
                            &observing_pool->base, /*request_count=*/1,
                            &request, &buffer));
  EXPECT_EQ(buffer, nullptr);

  ExpectPoolMethodUnlocked(observing_pool,
                           ObservingPoolMethod::kQueryCapabilities, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kAcquire, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kRelease, 1);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize), 0u);
  EXPECT_EQ(observing_pool->reentry_status_code.load(), IREE_STATUS_OK);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  iree_hal_pool_release(&observing_pool->base);
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       MaterializeFailureRollbackRunsOutsideSubmissionLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
  ASSERT_NE(queue_family, nullptr);
  const iree_hal_pool_reservation_request_t request = PoolRequest(queue_family);

  iree_hal_pool_t* delegate_pool = nullptr;
  IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                      kTransferSize, &delegate_pool));
  IREE_ASSERT_OK(WarmPool(delegate_pool, request));
  ObservingPool* observing_pool =
      CreateObservingPool(delegate_pool, host_queue);
  observing_pool->materialize_failure_code = IREE_STATUS_ABORTED;

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(), signal_list,
                            &observing_pool->base, /*request_count=*/1,
                            &request, &buffer));
  EXPECT_EQ(buffer, nullptr);

  ExpectPoolMethodUnlocked(observing_pool,
                           ObservingPoolMethod::kQueryCapabilities, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kAcquire, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kMaterialize,
                           1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kRelease, 1);
  EXPECT_EQ(observing_pool->reentry_status_code.load(), IREE_STATUS_OK);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  iree_hal_pool_release(&observing_pool->base);
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       ExhaustedPoolNotificationRunsOutsideSubmissionLock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  const iree_hal_queue_family_t* queue_family = iree_hal_queue_family(queue);
  ASSERT_NE(queue_family, nullptr);
  const iree_hal_pool_reservation_request_t request = PoolRequest(queue_family);

  iree_hal_pool_t* delegate_pool = nullptr;
  IREE_ASSERT_OK(CreateFixedBlockPool(test_device.base_device(), queue_family,
                                      kTransferSize, &delegate_pool));
  IREE_ASSERT_OK(WarmPool(delegate_pool, request));
  ObservingPool* observing_pool =
      CreateObservingPool(delegate_pool, host_queue);
  observing_pool->acquire_behavior = ObservingPoolAcquireBehavior::kExhausted;
  observing_pool->return_null_notification = true;

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(), signal_list,
                            &observing_pool->base, /*request_count=*/1,
                            &request, &buffer));
  EXPECT_EQ(buffer, nullptr);

  ExpectPoolMethodUnlocked(observing_pool,
                           ObservingPoolMethod::kQueryCapabilities, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kAcquire, 1);
  ExpectPoolMethodUnlocked(observing_pool, ObservingPoolMethod::kNotification,
                           1);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kMaterialize), 0u);
  EXPECT_EQ(observing_pool->call_count(ObservingPoolMethod::kRelease), 0u);
  EXPECT_EQ(observing_pool->reentry_status_code.load(), IREE_STATUS_OK);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  iree_hal_pool_release(&observing_pool->base);
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       PartialBindingSetFailureReleasesInsertedResourcesAfterUnlock) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  FailSecondBlockAllocator fail_allocator;
  iree_arena_block_pool_t binding_block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   fail_allocator.allocator(),
                                   &binding_block_pool);
  const iree_host_size_t inline_chunk_storage =
      binding_block_pool.usable_block_size -
      iree_host_align(sizeof(iree_hal_resource_set_t), iree_max_align_t);
  ASSERT_GT(inline_chunk_storage, sizeof(iree_hal_resource_set_chunk_t));
  const iree_host_size_t inline_capacity = std::min<iree_host_size_t>(
      (inline_chunk_storage - sizeof(iree_hal_resource_set_chunk_t)) /
          sizeof(iree_hal_resource_t*),
      IREE_HAL_RESOURCE_SET_CHUNK_MAX_CAPACITY);
  ASSERT_GT(inline_capacity, 8u);
  const iree_host_size_t binding_count = inline_capacity + 1;

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, static_cast<uint32_t>(binding_count),
      command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t pattern = 0x50415254u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(
          static_cast<uint32_t>(binding_count - 1), /*offset=*/0,
          sizeof(pattern)),
      &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  BindingBufferReleaseObservations release_observations;
  release_observations.queue = host_queue;
  std::vector<BindingBufferStorage> storage(binding_count);
  std::vector<BindingBufferReleaseToken> release_tokens(binding_count);
  std::vector<iree_hal_buffer_t*> caller_buffers(binding_count, nullptr);
  std::vector<iree_hal_buffer_binding_t> bindings(binding_count);
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    release_tokens[i].observations = &release_observations;
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        sizeof(storage[i].data),
        iree_make_byte_span(storage[i].data, sizeof(storage[i].data)),
        iree_hal_buffer_release_callback_t{
            /*.fn=*/BindingBufferReleaseToken::Release,
            /*.user_data=*/&release_tokens[i],
        },
        host_allocator_, &caller_buffers[i]));
    bindings[i] = iree_hal_buffer_binding_t{
        /*.buffer=*/caller_buffers[i],
        /*.offset=*/0,
        /*.length=*/IREE_HAL_WHOLE_BUFFER,
    };
  }

  constexpr iree_host_size_t kFinalReleaseWitnessCount = 8;
  fail_allocator.caller_buffers = &caller_buffers;
  fail_allocator.release_prefix_count = kFinalReleaseWitnessCount;
  iree_arena_block_pool_t* original_block_pool = host_queue->block_pool;
  const uint64_t initial_epoch =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t initial_write =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  host_queue->block_pool = &binding_block_pool;
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

  const iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/binding_count,
      /*.bindings=*/bindings.data(),
  };
  iree_status_t execute_status = iree_hal_queue_execute(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      command_buffer, binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);

  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_EQ(host_queue->block_pool, &binding_block_pool);
  host_queue->block_pool = original_block_pool;
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, execute_status);
  EXPECT_EQ(fail_allocator.allocation_count.load(), 2u);
  EXPECT_EQ(fail_allocator.caller_release_count.load(),
            kFinalReleaseWitnessCount);
  EXPECT_EQ(release_observations.release_count.load(),
            kFinalReleaseWitnessCount);
  EXPECT_TRUE(release_observations.all_submission_lock_checks_passed.load());
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission, initial_epoch);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      initial_write);

  for (iree_hal_buffer_t*& buffer : caller_buffers) {
    if (!buffer) continue;
    iree_hal_buffer_release(buffer);
    buffer = nullptr;
  }
  EXPECT_EQ(release_observations.release_count.load(), binding_count);
  EXPECT_TRUE(release_observations.all_submission_lock_checks_passed.load());
  iree_arena_block_pool_deinitialize(&binding_block_pool);

  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       DirectAqlPostCaptureFailureCleansUpAfterSubmissionUnlock) {
  RunCommandBufferOwnerCaptureCleanupTest(
      CommandBufferOwnerCaptureCase::kDirectAql);
}

TEST_F(HostQueuePublisherLifetimeTest,
       RetainedAqlReplayPostCaptureFailureCleansUpAfterSubmissionUnlock) {
  RunCommandBufferOwnerCaptureCleanupTest(
      CommandBufferOwnerCaptureCase::kRetainedAqlReplay);
}

TEST_F(HostQueuePublisherLifetimeTest,
       UnretainedAqlReplayPostCaptureFailureCleansUpAfterSubmissionUnlock) {
  RunCommandBufferOwnerCaptureCleanupTest(
      CommandBufferOwnerCaptureCase::kUnretainedAqlReplay);
}

TEST_F(HostQueuePublisherLifetimeTest,
       Pm4DynamicFixupPostCaptureFailureCleansUpAfterSubmissionUnlock) {
  RunCommandBufferOwnerCaptureCleanupTest(
      CommandBufferOwnerCaptureCase::kPm4DynamicFixup);
}

TEST_F(HostQueuePublisherLifetimeTest,
       ProfiledPm4PostCaptureFailureCleansUpAfterSubmissionUnlock) {
  RunCommandBufferOwnerCaptureCleanupTest(
      CommandBufferOwnerCaptureCase::kPm4Profiled);
}

TEST_F(HostQueuePublisherLifetimeTest,
       DeviceDestructionJoinsUnlockedSlowFeedbackSink) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_TRUE(
      iree_hal_amdgpu_feedback_state_is_enabled(&logical_device->feedback));
  ASSERT_GT(logical_device->feedback.device_state_count, 0u);
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];
  sink.drain_mutex = &feedback_device->drain_mutex;

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(),
      iree_hal_queue_family(test_device.provisioned_queue()),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const iree_hal_amdgpu_source_context_t* source_context =
      iree_hal_amdgpu_executable_source_context(executable);
  ASSERT_NE(source_context, nullptr);
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(source_context));
  const uint64_t executable_id = iree_hal_amdgpu_executable_id(executable);
  ASSERT_NE(executable_id, 0u);

  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  iree_hal_amdgpu_feedback_source_batch_t* source_batch = nullptr;
  iree_hal_executable_t* source_executables[] = {executable.get()};
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
      &logical_device->feedback, /*physical_device_ordinal=*/0,
      IREE_ARRAYSIZE(source_executables), source_executables, &source_batch));
  ASSERT_NE(source_batch, nullptr);
  iree_hal_amdgpu_feedback_source_batch_install(source_batch);
  EXPECT_EQ(feedback_phases.batch_installed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.installed_source_identity.load(), source_identity);
  EXPECT_EQ(feedback_phases.physical_device_ordinal.load(), 0u);

  // The installed ledger hold is now the sole executable/source owner. The
  // sink must be able to read this metadata until its callback claim returns.
  executable.reset();
  IREE_ASSERT_OK(PublishAsanFeedback(logical_device, 0, source_identity));
  sink.gate.WaitForReadyCount(1);
  EXPECT_EQ(sink.entry_count.load(), 1u);
  EXPECT_EQ(sink.exit_count.load(), 0u);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 0u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
  EXPECT_EQ(feedback_phases.claimed_source_identity.load(), source_identity);
  EXPECT_GE(feedback_phases.claimed_sequence.load(),
            feedback_phases.installed_sequence.load());

  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);

  ThreadGate destroy_gate;
  ThreadCompletion destroy_completion;
  std::thread destroyer([&] {
    destroy_gate.ArriveAndWait();
    iree_hal_device_release(device);
    destroy_completion.MarkDone();
  });
  destroy_gate.WaitForReadyCount(1);
  destroy_gate.Open();
  WaitForSignalNonzero(&logical_device->system->libhsa,
                       feedback_device->stop_signal);
  EXPECT_FALSE(destroy_completion.IsDone());

  sink.gate.Open();
  sink.exited.Wait();
  destroy_completion.Wait();
  destroyer.join();

  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 1u);
  EXPECT_EQ(sink.entry_count.load(), 1u);
  EXPECT_EQ(sink.exit_count.load(), 1u);
  EXPECT_TRUE(sink.drain_lock_was_available.load());
  EXPECT_TRUE(sink.metadata_stable.load());
  EXPECT_EQ(sink.metadata.type, IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT);
  EXPECT_EQ(sink.metadata.device, device);
  EXPECT_EQ(sink.metadata.executable_id, executable_id);
  EXPECT_GE(sink.metadata.payload_length,
            sizeof(iree_hal_device_asan_report_t));
  EXPECT_GE(sink.metadata.implementation_payload_length,
            sizeof(iree_hal_amdgpu_feedback_packet_t));
  EXPECT_EQ(sink.metadata.packet_source_context, source_identity);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 1u);
}

TEST_F(HostQueuePublisherLifetimeTest,
       DirectBorrowedDispatchFeedbackOwnerPrecedesPublicationAndOutlivesQueue) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];
  sink.drain_mutex = &feedback_device->drain_mutex;

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const iree_hal_amdgpu_source_context_t* source_context =
      iree_hal_amdgpu_executable_source_context(executable);
  ASSERT_NE(source_context, nullptr);
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(source_context));
  const uint64_t executable_id = iree_hal_amdgpu_executable_id(executable);

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), input_buffer.out()));
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), output_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               sizeof(input_values)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               sizeof(input_values)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  ThreadGate installed_gate;
  FeedbackPhaseRecorder feedback_phases;
  feedback_phases.batch_installed_gate = &installed_gate;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  const uint64_t initial_last_published = QueueAxisFrontierWaiter::LoadCursor(
      &host_queue->notification_ring.epoch.last_published);
  std::atomic<iree_status_code_t> dispatch_status_code{IREE_STATUS_UNKNOWN};
  ThreadCompletion dispatch_completion;
  std::thread dispatcher([&] {
    iree_status_t status = iree_hal_queue_dispatch(
        queue, iree_hal_semaphore_list_empty(), signal_list, executable,
        iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_make_const_byte_span(constant_values, sizeof(constant_values)),
        bindings, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES);
    dispatch_status_code.store(iree_status_code(status));
    iree_status_free(status);
    dispatch_completion.MarkDone();
  });

  // The independent source owner is visible before the first AQL packet or
  // public epoch can be committed. A packet arriving in this window must be
  // able to claim that owner even though the dispatch publisher is paused.
  installed_gate.WaitForReadyCount(1);
  EXPECT_EQ(feedback_phases.batch_installed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.installed_source_identity.load(), source_identity);
  EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.epoch.last_published),
            initial_last_published);
  IREE_ASSERT_OK(PublishAsanFeedback(logical_device, 0, source_identity));
  sink.gate.WaitForReadyCount(1);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 0u);
  EXPECT_FALSE(dispatch_completion.IsDone());

  installed_gate.Open();
  dispatch_completion.Wait();
  dispatcher.join();
  EXPECT_EQ(dispatch_status_code.load(), IREE_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  feedback_phases.WaitForPhaseCount(feedback_phases.target_closed_count, 1u);
  EXPECT_EQ(feedback_phases.closed_source_identity.load(), source_identity);

  // BORROW_RESOURCE_LIFETIMES permits caller ownership to end at signal
  // completion. The queue can also seal while the unrelated feedback runner
  // remains in its compliant sink, but the executable-owned metadata must
  // remain valid through callback return.
  executable.reset();
  input_buffer.reset();
  output_buffer.reset();
  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
  EXPECT_EQ(sink.exit_count.load(), 0u);

  sink.gate.Open();
  sink.exited.Wait();
  feedback_phases.WaitForPhaseCount(feedback_phases.batch_retired_count, 1u);
  WaitForFeedbackLedgerEmpty(feedback_device);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
  EXPECT_TRUE(sink.metadata_stable.load());
  EXPECT_EQ(sink.metadata.executable_id, executable_id);
  EXPECT_EQ(sink.metadata.packet_source_context, source_identity);

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       RetainedAqlRecordedFeedbackOwnerOutlivesQueueResources) {
  RunRecordedFeedbackOwnerTest(/*use_pm4=*/false, /*unretained=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       UnretainedAqlRecordedFeedbackOwnerOutlivesQueueResources) {
  RunRecordedFeedbackOwnerTest(/*use_pm4=*/false, /*unretained=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       RetainedPm4RecordedFeedbackOwnerOutlivesQueueResources) {
  RunRecordedFeedbackOwnerTest(/*use_pm4=*/true, /*unretained=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       UnretainedPm4RecordedFeedbackOwnerOutlivesQueueResources) {
  RunRecordedFeedbackOwnerTest(/*use_pm4=*/true, /*unretained=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       RetainedPm4HostOnlySidecarTransfersExactFeedbackOwner) {
  RunPm4FeedbackSidecarHostOnlyTest(/*unretained=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       UnretainedPm4HostOnlySidecarTransfersExactFeedbackOwner) {
  RunPm4FeedbackSidecarHostOnlyTest(/*unretained=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       MalformedFeedbackSourcePoisonsWithoutDereferenceOrSinkCall) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(),
      iree_hal_queue_family(test_device.provisioned_queue()),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  iree_hal_amdgpu_feedback_source_batch_t* source_batch = nullptr;
  iree_hal_executable_t* sources[] = {executable.get()};
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
      &logical_device->feedback, 0, IREE_ARRAYSIZE(sources), sources,
      &source_batch));
  iree_hal_amdgpu_feedback_source_batch_install(source_batch);
  executable.reset();

  // This deliberately invalid non-null value must remain an integer lookup
  // key. No source-context load or sink call is permitted before a retained
  // identity+sequence owner has been claimed.
  constexpr uint64_t kInvalidSourceIdentity = 1;
  IREE_ASSERT_OK(
      PublishAsanFeedback(logical_device, 0, kInvalidSourceIdentity));
  WaitForFeedbackPoisoned(feedback_device);
  EXPECT_EQ(sink.entry_count.load(), 0u);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 0u);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 0u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);

  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);
  iree_hal_device_release(device);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 1u);
  EXPECT_EQ(sink.entry_count.load(), 0u);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 1u);
}

TEST_F(HostQueuePublisherLifetimeTest,
       UnknownFeedbackKindReturnsClaimAndPoisonsWithoutSinkCall) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(),
      iree_hal_queue_family(test_device.provisioned_queue()),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(executable)));
  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  iree_hal_amdgpu_feedback_source_batch_t* source_batch = nullptr;
  iree_hal_executable_t* sources[] = {executable.get()};
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
      &logical_device->feedback, 0, IREE_ARRAYSIZE(sources), sources,
      &source_batch));
  iree_hal_amdgpu_feedback_source_batch_install(source_batch);
  executable.reset();

  iree_hal_amdgpu_feedback_config_t config;
  iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
  IREE_ASSERT_OK(ReserveFeedback(logical_device, 0, source_identity,
                                 IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
                                 /*payload_length=*/0, &config, &packet));
  iree_hal_amdgpu_feedback_publish(&config, packet);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&logical_device->system->libhsa),
                                  config.notify_signal, 1);
  WaitForFeedbackPoisoned(feedback_device);
  feedback_phases.WaitForPhaseCount(feedback_phases.claim_returned_count, 1u);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
  EXPECT_EQ(feedback_phases.claimed_source_identity.load(), source_identity);
  EXPECT_EQ(sink.entry_count.load(), 0u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);

  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);
  iree_hal_device_release(device);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 1u);
  EXPECT_EQ(sink.entry_count.load(), 0u);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 1u);
}

TEST_F(HostQueuePublisherLifetimeTest,
       StaleFeedbackSequenceBelowReusedSourceIntervalPoisonsWithoutClaim) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(),
      iree_hal_queue_family(test_device.provisioned_queue()),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(executable)));

  // Reserve the stale packet first but leave it unpublished. Installing the
  // new owner afterward gives it a strictly greater lower sequence bound while
  // preserving a drainable packet sequence at the channel read tail.
  iree_hal_amdgpu_feedback_config_t config;
  iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
  IREE_ASSERT_OK(ReserveAsanFeedback(logical_device, 0, source_identity,
                                     &config, &packet));
  const uint64_t stale_sequence = packet->sequence;
  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  iree_hal_amdgpu_feedback_source_batch_t* source_batch = nullptr;
  iree_hal_executable_t* sources[] = {executable.get()};
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
      &logical_device->feedback, 0, IREE_ARRAYSIZE(sources), sources,
      &source_batch));
  iree_hal_amdgpu_feedback_source_batch_install(source_batch);
  ASSERT_GT(feedback_phases.installed_sequence.load(), stale_sequence);
  executable.reset();

  iree_hal_amdgpu_feedback_publish(&config, packet);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&logical_device->system->libhsa),
                                  config.notify_signal, 1);
  WaitForFeedbackPoisoned(feedback_device);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 0u);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 0u);
  EXPECT_EQ(sink.entry_count.load(), 0u);

  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);
  iree_hal_device_release(device);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 1u);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 1u);
}

TEST_F(HostQueuePublisherLifetimeTest,
       DeviceTeardownDrainsReadyFeedbackPrefixBeforeOwnerRelease) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];
  sink.drain_mutex = &feedback_device->drain_mutex;

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(),
      iree_hal_queue_family(test_device.provisioned_queue()),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(executable)));
  const uint64_t executable_id = iree_hal_amdgpu_executable_id(executable);
  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  iree_hal_amdgpu_feedback_source_batch_t* source_batch = nullptr;
  iree_hal_executable_t* sources[] = {executable.get()};
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
      &logical_device->feedback, 0, IREE_ARRAYSIZE(sources), sources,
      &source_batch));
  iree_hal_amdgpu_feedback_source_batch_install(source_batch);
  executable.reset();

  iree_hal_amdgpu_feedback_config_t config;
  iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
  IREE_ASSERT_OK(ReserveAsanFeedback(logical_device, 0, source_identity,
                                     &config, &packet));
  iree_hal_amdgpu_feedback_publish(&config, packet);
  // Intentionally do not wake the ordinary service thread. Device teardown
  // must consume this maximal READY prefix after stopping and joining it.
  EXPECT_EQ(sink.entry_count.load(), 0u);

  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);
  ThreadCompletion destroy_completion;
  std::thread destroyer([&] {
    iree_hal_device_release(device);
    destroy_completion.MarkDone();
  });

  sink.gate.WaitForReadyCount(1);
  EXPECT_FALSE(destroy_completion.IsDone());
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 0u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
  sink.gate.Open();
  sink.exited.Wait();
  destroy_completion.Wait();
  destroyer.join();

  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 1u);
  EXPECT_TRUE(sink.drain_lock_was_available.load());
  EXPECT_TRUE(sink.metadata_stable.load());
  EXPECT_EQ(sink.metadata.executable_id, executable_id);
  EXPECT_EQ(sink.metadata.packet_source_context, source_identity);
  EXPECT_EQ(controller.Snapshot().proactor_destroy_count, 1u);
}

TEST_F(HostQueuePublisherLifetimeTest,
       ConcurrentSameSourceFeedbackBatchesRetireAfterExactClaimReturns) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];
  sink.drain_mutex = &feedback_device->drain_mutex;
  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(executable)));
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), input_buffer.out()));
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), output_buffer.out()));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, 0, 4 * sizeof(uint32_t)),
      iree_hal_make_buffer_ref(output_buffer, 0, 4 * sizeof(uint32_t)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};

  Ref<iree_hal_semaphore_t> first_signal;
  Ref<iree_hal_semaphore_t> second_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), first_signal.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), second_signal.out()));
  iree_hal_semaphore_t* first_signal_pointer = first_signal.get();
  iree_hal_semaphore_t* second_signal_pointer = second_signal.get();
  uint64_t first_signal_value = 1;
  uint64_t second_signal_value = 1;
  const iree_hal_semaphore_list_t first_signal_list =
      SemaphoreList(&first_signal_pointer, &first_signal_value);
  const iree_hal_semaphore_list_t second_signal_list =
      SemaphoreList(&second_signal_pointer, &second_signal_value);

  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  for (iree_hal_semaphore_list_t signal_list :
       {first_signal_list, second_signal_list}) {
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, iree_hal_semaphore_list_empty(), signal_list, executable,
        iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_make_const_byte_span(constant_values, sizeof(constant_values)),
        bindings, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES));
  }
  ASSERT_EQ(feedback_phases.batch_installed_count.load(), 2u);
  IREE_ASSERT_OK(PublishAsanFeedback(logical_device, 0, source_identity));
  sink.gate.WaitForReadyCount(1);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(first_signal, first_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(second_signal, second_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  feedback_phases.WaitForPhaseCount(feedback_phases.target_closed_count, 2u);
  executable.reset();
  input_buffer.reset();
  output_buffer.reset();
  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);

  sink.gate.Open();
  sink.exited.Wait();
  feedback_phases.WaitForPhaseCount(feedback_phases.batch_retired_count, 2u);
  WaitForFeedbackLedgerEmpty(feedback_device);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
  EXPECT_EQ(feedback_phases.claimed_source_identity.load(), source_identity);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       QueueSealIgnoresUnrelatedEarlierFeedbackReservationButOwnerDoesNot) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateFeedbackTestDevice(&controller, sink.sink(), &test_device));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];
  sink.drain_mutex = &feedback_device->drain_mutex;
  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  Ref<iree_hal_executable_t> earlier_executable;
  Ref<iree_hal_executable_t> queue_executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      earlier_executable.out()));
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      queue_executable.out()));
  const uint64_t earlier_source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(earlier_executable)));
  const uint64_t queue_source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(queue_executable)));
  ASSERT_NE(earlier_source_identity, queue_source_identity);

  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  iree_hal_amdgpu_feedback_source_batch_t* earlier_batch = nullptr;
  iree_hal_executable_t* earlier_sources[] = {earlier_executable.get()};
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_source_batch_prepare(
      &logical_device->feedback, 0, IREE_ARRAYSIZE(earlier_sources),
      earlier_sources, &earlier_batch));
  iree_hal_amdgpu_feedback_source_batch_install(earlier_batch);
  iree_hal_amdgpu_feedback_config_t earlier_config;
  iree_hal_amdgpu_feedback_packet_t* earlier_packet = nullptr;
  IREE_ASSERT_OK(ReserveAsanFeedback(logical_device, 0, earlier_source_identity,
                                     &earlier_config, &earlier_packet));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), input_buffer.out()));
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), output_buffer.out()));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, 0, 4 * sizeof(uint32_t)),
      iree_hal_make_buffer_ref(output_buffer, 0, 4 * sizeof(uint32_t)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue, iree_hal_semaphore_list_empty(), signal_list, queue_executable,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_make_const_byte_span(constant_values, sizeof(constant_values)),
      bindings, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  feedback_phases.WaitForPhaseCount(feedback_phases.target_closed_count, 1u);
  ASSERT_EQ(feedback_phases.batch_installed_count.load(), 2u);

  // The queue batch has an empty interval beginning after the unrelated
  // RESERVED packet. It cannot retire until read_tail crosses that packet, but
  // queue sealing is independent of this device-stable feedback ownership.
  queue_executable.reset();
  input_buffer.reset();
  output_buffer.reset();
  iree_hal_amdgpu_host_queue_seal(host_queue);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);

  iree_hal_amdgpu_feedback_source_batch_close(earlier_batch);
  earlier_executable.reset();
  iree_hal_amdgpu_feedback_publish(&earlier_config, earlier_packet);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&logical_device->system->libhsa),
                                  earlier_config.notify_signal, 1);
  sink.gate.WaitForReadyCount(1);
  EXPECT_EQ(feedback_phases.claimed_source_identity.load(),
            earlier_source_identity);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
  sink.gate.Open();
  sink.exited.Wait();
  feedback_phases.WaitForPhaseCount(feedback_phases.batch_retired_count, 2u);
  WaitForFeedbackLedgerEmpty(feedback_device);
  EXPECT_EQ(feedback_phases.closed_source_identity.load(),
            earlier_source_identity);
  EXPECT_NE(queue_source_identity, earlier_source_identity);

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       PartialAqlReplayClosesOnlyPublishedFeedbackBatchBeforeSeal) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  options.feedback.enabled = true;
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.host_queues.aql_capacity = 64;
  options.host_queues.notification_capacity = 1;
  options.host_queues.kernarg_capacity = 128;
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;

  SlowFeedbackSink sink;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&controller, &options,
                                        /*staging_slot_count=*/1, sink.sink()));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_feedback_device_state_t* feedback_device =
      &logical_device->feedback.device_states[0];
  sink.drain_mutex = &feedback_device->drain_mutex;
  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(test::LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_SV("command_buffer_dispatch_constants_bindings_test.bin"),
      executable.out()));
  const uint64_t source_identity =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          iree_hal_amdgpu_executable_source_context(executable)));
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), input_buffer.out()));
  IREE_ASSERT_OK(test::CreateHostVisibleDispatchBuffer(
      test_device.allocator(), 4 * sizeof(uint32_t), output_buffer.out()));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, 0, 4 * sizeof(uint32_t)),
      iree_hal_make_buffer_ref(output_buffer, 0, 4 * sizeof(uint32_t)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  for (uint32_t i = 0; i < 32; ++i) {
    IREE_ASSERT_OK(test::AppendConstantsBindingsDispatch(command_buffer,
                                                         executable, bindings));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  ASSERT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program, nullptr);
  ASSERT_GT(program->block_count, 1u);

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);
  FeedbackPhaseRecorder feedback_phases;
  ScopedFeedbackPhaseObserver feedback_observer(&feedback_phases);
  const uint64_t first_block_epoch =
      host_queue->notification_ring.epoch.next_submission + 1;
  QueuePhaseLatch second_block_retry_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_POST_DRAIN_RUNNER,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_POST_DRAIN_BATCH_DETACHED_BEFORE_CALLBACK,
      /*blocked_occurrence=*/1, /*minimum_value0=*/0,
      &feedback_phases.target_closed_count);
  SealOwnerLatch seal_owner_latch(host_queue);
  QueuePhaseObserverGroup queue_phase_observers;
  queue_phase_observers.Add(QueuePhaseLatch::Observe,
                            &second_block_retry_latch);
  queue_phase_observers.Add(SealOwnerLatch::Observe, &seal_owner_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &queue_phase_observers);
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queue, iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  iree_slim_mutex_lock(&host_queue->locks.post_drain_mutex);
  const bool has_second_block_retry = host_queue->post_drain.head != nullptr;
  iree_slim_mutex_unlock(&host_queue->locks.post_drain_mutex);
  ASSERT_TRUE(has_second_block_retry);
  ASSERT_EQ(host_queue->notification_ring.epoch.next_submission,
            first_block_epoch);
  ASSERT_EQ(feedback_phases.batch_installed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.installed_source_identity.load(), source_identity);

  IREE_ASSERT_OK(PublishAsanFeedback(logical_device, 0, source_identity));
  sink.gate.WaitForReadyCount(1);
  executable.reset();

  // Finish block 1 normally. Lane C closes its exact published feedback
  // target. Fairness-limited post-drain passes yield between earlier capacity
  // retries; the phase prerequisite pauses the first later detached retry
  // after that close, before its callback can take submission_mutex and
  // publish block 2. This distinguishes normal target close from seal's
  // OPEN-batch force-retirement path.
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  WaitForHardwareEpoch(&libhsa_, host_queue, first_block_epoch);
  ThreadCompletion drain_completion;
  std::thread drainer([&] {
    (void)iree_hal_amdgpu_host_queue_drain_completions_for_waiter(host_queue);
    drain_completion.MarkDone();
  });
  second_block_retry_latch.WaitUntilEntered();
  EXPECT_EQ(feedback_phases.target_closed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.batch_installed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(feedback_phases.batch_installed_count.load(), 1u);

  // Admission is now CLOSED. Releasing the detached retry lets it attempt
  // block 2, which must be rejected without installing a second source batch
  // or reclassifying block 1.
  second_block_retry_latch.Release();
  drain_completion.Wait();
  drainer.join();
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(feedback_phases.batch_installed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.target_closed_count.load(), 1u);
  EXPECT_EQ(feedback_phases.batch_retired_count.load(), 0u);
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  command_buffer.reset();
  input_buffer.reset();
  output_buffer.reset();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  sink.gate.Open();
  sink.exited.Wait();
  feedback_phases.WaitForPhaseCount(feedback_phases.batch_retired_count, 1u);
  WaitForFeedbackLedgerEmpty(feedback_device);
  EXPECT_EQ(feedback_phases.claim_entered_count.load(), 1u);
  EXPECT_EQ(feedback_phases.claim_returned_count.load(), 1u);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       NativeTransferSignalBarrierRacingFatalFailurePreservesStickyStatus) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_pointer = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  PublisherSubmissionLatch publisher_latch(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PUBLISHER_PATH_TRANSFER_SIGNAL_BARRIER);
  QueuePhaseLatch failure_latch(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_FAILURE_ADMISSION,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_BEFORE_FAILURE_ADMISSION_LOCK);
  QueuePhaseObserverGroup phase_observers;
  phase_observers.Add(PublisherSubmissionLatch::Observe, &publisher_latch);
  phase_observers.Add(QueuePhaseLatch::Observe, &failure_latch);
  PublisherSubmissionObserverGuard observer_guard(
      host_queue, &publisher_latch, &failure_latch, &phase_observers);

  const uint32_t fill_pattern = 0x53544943u;
  iree_hal_transfer_operation_t operation = {};
  operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  operation.fill.target_buffer = buffer;
  operation.fill.length = kTransferSize;
  operation.fill.pattern = &fill_pattern;
  operation.fill.pattern_length = sizeof(fill_pattern);
  IREE_ASSERT_OK(iree_hal_queue_transfer(queue, iree_hal_semaphore_list_empty(),
                                         signal_list,
                                         /*operation_count=*/1, &operation));

  const QueuePublicationSnapshot before =
      InstallStickyFailureWhilePublisherPaused(host_queue, &publisher_latch,
                                               &failure_latch);
  publisher_latch.Release();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(publisher_latch.occurrence_count(), 1u);
  ExpectQueuePublicationUnchanged(before, SnapshotQueuePublication(host_queue));

  observer_guard.Reset();
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  const ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 0u);
  EXPECT_EQ(sealed_snapshot.cancel_count, 0u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

#if IREE_FILE_IO_ENABLE

TEST_F(HostQueuePublisherLifetimeTest,
       DirectFileCancelledCompletionPreservesAnnotatedStickyStatus) {
  RunFileSealCancellationStatusTest(
      /*staged=*/false, /*supports_cancellation=*/true,
      /*record_failure=*/true, /*completion_code=*/IREE_STATUS_CANCELLED,
      /*expected_status=*/IREE_STATUS_DATA_LOSS);
}

TEST_F(HostQueuePublisherLifetimeTest,
       StagedFileNaturalCompletionPreservesAnnotatedStickyStatus) {
  RunFileSealCancellationStatusTest(
      /*staged=*/true, /*supports_cancellation=*/false,
      /*record_failure=*/true, /*completion_code=*/IREE_STATUS_OK,
      /*expected_status=*/IREE_STATUS_DATA_LOSS);
}

TEST_F(HostQueuePublisherLifetimeTest, DirectFileOrdinarySealReportsCancelled) {
  RunFileSealCancellationStatusTest(
      /*staged=*/false, /*supports_cancellation=*/true,
      /*record_failure=*/false, /*completion_code=*/IREE_STATUS_CANCELLED,
      /*expected_status=*/IREE_STATUS_CANCELLED);
}

TEST_F(HostQueuePublisherLifetimeTest,
       DirectFileOperationFailureOutranksStickyShutdown) {
  RunFileSealCancellationStatusTest(
      /*staged=*/false, /*supports_cancellation=*/true,
      /*record_failure=*/true, /*completion_code=*/IREE_STATUS_ABORTED,
      /*expected_status=*/IREE_STATUS_ABORTED);
}

TEST_F(HostQueuePublisherLifetimeTest,
       StagedFileOperationFailureOutranksStickyShutdown) {
  RunFileSealCancellationStatusTest(
      /*staged=*/true, /*supports_cancellation=*/true,
      /*record_failure=*/true, /*completion_code=*/IREE_STATUS_ABORTED,
      /*expected_status=*/IREE_STATUS_ABORTED);
}

TEST_F(HostQueuePublisherLifetimeTest,
       StagingCopyRacingFatalFailurePreservesStickyStatus) {
  RunFilePublisherStickyFailureRaceTest(/*staged=*/true,
                                        /*gate_copy=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       StagingSignalBarrierRacingFatalFailurePreservesStickyStatus) {
  RunFilePublisherStickyFailureRaceTest(/*staged=*/true,
                                        /*gate_copy=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       DirectFileSignalBarrierRacingFatalFailurePreservesStickyStatus) {
  RunFilePublisherStickyFailureRaceTest(/*staged=*/false,
                                        /*gate_copy=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsDirectReadSubmitTailWakeBeforeDestruction) {
  RunFileSubmitTailSealJoinTest(/*staged=*/false, /*is_write=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsDirectWriteSubmitTailWakeBeforeDestruction) {
  RunFileSubmitTailSealJoinTest(/*staged=*/false, /*is_write=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsStagedReadSubmitTailWakeBeforeDestruction) {
  RunFileSubmitTailSealJoinTest(/*staged=*/true, /*is_write=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsStagedWriteSubmitTailWakeBeforeDestruction) {
  RunFileSubmitTailSealJoinTest(/*staged=*/true, /*is_write=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsStagedReadSafeHandoffAndCallbackTail) {
  RunStagedSafeActionSealJoinTest(/*is_write=*/false);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsStagedWriteSafeHandoffAndCallbackTail) {
  RunStagedSafeActionSealJoinTest(/*is_write=*/true);
}

TEST_F(HostQueuePublisherLifetimeTest,
       ConcurrentSealJoinsParkedDirectFileOnDynamicQueue) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());

  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(test_device.base_device(),
                                  IREE_HAL_MEMORY_ACCESS_READ, &path,
                                  file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), true, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_pointer = signal.get();
  iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  IREE_ASSERT_OK(iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(),
                                     signal_list, file, 0, buffer, 0,
                                     kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);

  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_NE(host_queue->active_file_action_head, nullptr);
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

  ThreadGate gate;
  ThreadCompletion first_completion;
  ThreadCompletion second_completion;
  std::thread first_sealer([&] {
    gate.ArriveAndWait();
    iree_hal_amdgpu_host_queue_seal(host_queue);
    first_completion.MarkDone();
  });
  std::thread second_sealer([&] {
    gate.ArriveAndWait();
    iree_hal_amdgpu_host_queue_seal(host_queue);
    second_completion.MarkDone();
  });
  gate.WaitForReadyCount(2);
  gate.Open();

  controller.WaitForCancelCount(1);
  EXPECT_FALSE(first_completion.IsDone());
  EXPECT_FALSE(second_completion.IsDone());
  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackExitCount(1);
  first_completion.Wait();
  second_completion.Wait();
  first_sealer.join();
  second_sealer.join();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.cancel_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealCancelsAndJoinsParkedStagedFileCallback) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(test_device.base_device(),
                                  IREE_HAL_MEMORY_ACCESS_READ, &path,
                                  file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_pointer = signal.get();
  iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  IREE_ASSERT_OK(iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(),
                                     signal_list, file, 0, buffer, 0,
                                     kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  controller.WaitForCancelCount(1);
  EXPECT_FALSE(seal_completion.IsDone());
  controller.ReleaseNextCancelled();
  controller.WaitForCallbackExitCount(1);
  seal_completion.Wait();
  sealer.join();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.cancel_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsDirectFileTerminalCallbackTail) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  ThreadGate callback_tail_gate;
  PollRunnerFileFinalizer finalizer;
  finalizer.gate = &callback_tail_gate;
  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreatePollRunnerFinalizerFile(test_device.base_device(),
                                               IREE_HAL_MEMORY_ACCESS_READ,
                                               &finalizer, &path, file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), true, buffer.out()));

  IREE_ASSERT_OK(iree_hal_queue_read(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      file, 0, buffer, 0, kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);
  file.reset();
  buffer.reset();

  controller.ReleaseNextSuccessful();
  callback_tail_gate.WaitForReadyCount(1);
  controller.WaitForFileDestroyCount(1);
  EXPECT_EQ(finalizer.entry_count.load(), 1u);
  EXPECT_EQ(finalizer.return_count.load(), 0u);

  SealOwnerLatch seal_owner_latch(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(SealOwnerLatch::Observe,
                                                     &seal_owner_latch);
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());

  callback_tail_gate.Open();
  finalizer.returned.Wait();
  controller.WaitForCallbackExitCount(1);
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(finalizer.entry_count.load(), 1u);
  EXPECT_EQ(finalizer.return_count.load(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(sealed_snapshot.file_import_count, 1u);
  EXPECT_EQ(sealed_snapshot.file_destroy_count, 1u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       PollRunnerDefersLastQueueDeviceAndPoolReleasePastCallbackReturn) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  ThreadGate finalizer_gate;
  PollRunnerFileFinalizer finalizer;
  finalizer.controller = &controller;
  finalizer.gate = &finalizer_gate;
  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreatePollRunnerFinalizerFile(test_device.base_device(),
                                               IREE_HAL_MEMORY_ACCESS_READ,
                                               &finalizer, &path, file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), true, buffer.out()));

  IREE_ASSERT_OK(iree_hal_queue_read(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      file, 0, buffer, 0, kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);

  // Move the only caller-owned dynamic-queue reference into the imported-file
  // finalizer. The queue's parent edge is then the only remaining device edge,
  // and the device's edge is the only remaining proactor-pool edge.
  finalizer.queue = queue.release();
  file.reset();
  buffer.reset();
  iree_hal_device_t* device = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  test_device.TakeDeviceOwnership(&device, &device_group);
  test_device.ReleaseFixtureProactorPoolReference();
  iree_hal_device_group_release(device_group);
  iree_hal_device_release(device);

  controller.ReleaseNextSuccessful();
  finalizer_gate.WaitForReadyCount(1);
  ControlledProactorSnapshot parked_snapshot = controller.Snapshot();
  EXPECT_EQ(parked_snapshot.callback_entry_count, 1u);
  // The proactor callback must return before the handed-off finalizer can
  // release the last queue/device/pool edge.
  EXPECT_EQ(parked_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(parked_snapshot.proactor_destroy_count, 0u);
  EXPECT_EQ(finalizer.entry_count.load(), 1u);
  EXPECT_EQ(finalizer.return_count.load(), 0u);

  finalizer_gate.Open();
  finalizer.returned.Wait();
  controller.WaitForCallbackExitCount(1);
  controller.WaitForProactorDestroyCount(1);

  const ControlledProactorSnapshot final_snapshot = controller.Snapshot();
  EXPECT_EQ(finalizer.entry_count.load(), 1u);
  EXPECT_EQ(finalizer.return_count.load(), 1u);
  EXPECT_EQ(final_snapshot.submit_count, 1u);
  EXPECT_EQ(final_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(final_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(final_snapshot.file_import_count, 1u);
  EXPECT_EQ(final_snapshot.file_destroy_count, 1u);
  EXPECT_EQ(final_snapshot.proactor_destroy_count, 1u);
  EXPECT_EQ(final_snapshot.destroy_with_pending_count, 0u);
  EXPECT_EQ(final_snapshot.pending_count, 0u);
  EXPECT_GT(final_snapshot.finalizer_return_event, 0u);
  EXPECT_GT(final_snapshot.finalizer_return_event,
            final_snapshot.callback_exit_event);
  EXPECT_GT(final_snapshot.proactor_destroy_event,
            final_snapshot.callback_exit_event);
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsStagedFileTerminalCallbackTail) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  ThreadGate callback_tail_gate;
  PollRunnerFileFinalizer finalizer;
  finalizer.gate = &callback_tail_gate;
  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreatePollRunnerFinalizerFile(test_device.base_device(),
                                               IREE_HAL_MEMORY_ACCESS_READ,
                                               &finalizer, &path, file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer.out()));

  IREE_ASSERT_OK(iree_hal_queue_read(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      file, 0, buffer, 0, kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);
  file.reset();
  buffer.reset();

  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackExitCount(1);
  callback_tail_gate.WaitForReadyCount(1);
  controller.WaitForFileDestroyCount(1);
  EXPECT_EQ(finalizer.entry_count.load(), 1u);
  EXPECT_EQ(finalizer.return_count.load(), 0u);

  SealOwnerLatch seal_owner_latch(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(SealOwnerLatch::Observe,
                                                     &seal_owner_latch);
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());

  callback_tail_gate.Open();
  finalizer.returned.Wait();
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);

  EXPECT_EQ(finalizer.entry_count.load(), 1u);
  EXPECT_EQ(finalizer.return_count.load(), 1u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(sealed_snapshot.file_import_count, 1u);
  EXPECT_EQ(sealed_snapshot.file_destroy_count, 1u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealNaturallyJoinsUnsupportedFileCancellation) {
  ControlledProactorController controller(IREE_ASYNC_PROACTOR_CAPABILITY_NONE);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(test_device.base_device(),
                                  IREE_HAL_MEMORY_ACCESS_READ, &path,
                                  file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), true, buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_pointer = signal.get();
  iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  IREE_ASSERT_OK(iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(),
                                     signal_list, file, 0, buffer, 0,
                                     kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);
  const uint64_t previous_query_count =
      controller.Snapshot().capability_query_count;

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  controller.WaitForCapabilityQueryAfter(previous_query_count);
  EXPECT_EQ(controller.Snapshot().cancel_count, 0u);
  EXPECT_FALSE(seal_completion.IsDone());
  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackExitCount(1);
  seal_completion.Wait();
  sealer.join();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.cancel_count, 0u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsOneSlotWaiterClaimedBeforeCallback) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> waiter_queue;
  Ref<iree_hal_queue_t> slot_owner_queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(waiter_queue.out()));
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(slot_owner_queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(waiter_queue.get());
  iree_hal_amdgpu_host_queue_t* slot_owner_host_queue =
      HostQueue(slot_owner_queue.get());
  ASSERT_NE(host_queue, slot_owner_host_queue);
  iree_hal_amdgpu_staging_pool_t* staging_pool = test_device.staging_pool();
  ASSERT_EQ(staging_pool->slot_count, 1u);

  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(
      test_device.base_device(),
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, &path,
      file.out(), 2 * kTransferSize));
  Ref<iree_hal_buffer_t> first_buffer;
  Ref<iree_hal_buffer_t> second_buffer;
  IREE_ASSERT_OK(
      CreateBuffer(test_device.allocator(), false, first_buffer.out()));
  IREE_ASSERT_OK(
      CreateBuffer(test_device.allocator(), false, second_buffer.out()));

  IREE_ASSERT_OK(
      iree_hal_queue_write(slot_owner_queue, iree_hal_semaphore_list_empty(),
                           iree_hal_semaphore_list_empty(), first_buffer, 0,
                           file, 0, kTransferSize, IREE_HAL_WRITE_FLAG_NONE));
  controller.WaitForSubmitCount(1);
  QueuePhaseSignal queued_waiter(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_STAGING_WAITER_QUEUED);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseSignal::Observe,
                                                     &queued_waiter);
  IREE_ASSERT_OK(iree_hal_queue_write(
      waiter_queue, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), second_buffer, 0, file, kTransferSize,
      kTransferSize, IREE_HAL_WRITE_FLAG_NONE));
  queued_waiter.WaitUntilObserved();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(queued_waiter.occurrence_count(), 1u);
  EXPECT_EQ(queued_waiter.value1(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staging_pool)));
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_EQ(queued_waiter.value0(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                host_queue->active_staging_transfer_head)));
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  ASSERT_TRUE(StagingPoolHasQueuedWaiter(staging_pool));

  WaiterClaimLatch claim_latch;
  iree_hal_amdgpu_host_queue_staging_set_waiter_claimed_observer(
      WaiterClaimLatch::Observe, &claim_latch);
  SealOwnerLatch seal_owner_latch(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(SealOwnerLatch::Observe,
                                                     &seal_owner_latch);
  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackEntryCount(1);
  claim_latch.WaitUntilEntered();

  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_EQ(staging_pool->waiter_head, nullptr);
  EXPECT_EQ(staging_pool->waiter_tail, nullptr);
  EXPECT_EQ(staging_pool->claimed_waiter_count, 1u);
  EXPECT_EQ(staging_pool->available_count, 0u);
  iree_slim_mutex_unlock(&staging_pool->mutex);

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());

  claim_latch.Release();
  controller.WaitForCallbackExitCount(1);
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  iree_hal_amdgpu_host_queue_staging_set_waiter_claimed_observer(nullptr,
                                                                 nullptr);

  ExpectQueueCertifiedAndEmpty(host_queue, staging_pool);
  iree_hal_amdgpu_host_queue_seal(slot_owner_host_queue);
  ExpectQueueCertifiedAndEmpty(slot_owner_host_queue, staging_pool);
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  waiter_queue.reset();
  slot_owner_queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealJoinsOneSlotWaiterWhileCallbackFinishesFailedSubmit) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device));

  Ref<iree_hal_queue_t> waiter_queue;
  Ref<iree_hal_queue_t> slot_owner_queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(waiter_queue.out()));
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(slot_owner_queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(waiter_queue.get());
  iree_hal_amdgpu_host_queue_t* slot_owner_host_queue =
      HostQueue(slot_owner_queue.get());
  ASSERT_NE(host_queue, slot_owner_host_queue);
  iree_hal_amdgpu_staging_pool_t* staging_pool = test_device.staging_pool();
  ASSERT_EQ(staging_pool->slot_count, 1u);

  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(test_device.base_device(),
                                  IREE_HAL_MEMORY_ACCESS_READ, &path,
                                  file.out()));
  Ref<iree_hal_buffer_t> first_buffer;
  Ref<iree_hal_buffer_t> second_buffer;
  IREE_ASSERT_OK(
      CreateBuffer(test_device.allocator(), false, first_buffer.out()));
  IREE_ASSERT_OK(
      CreateBuffer(test_device.allocator(), false, second_buffer.out()));
  Ref<iree_hal_semaphore_t> first_signal;
  Ref<iree_hal_semaphore_t> second_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), first_signal.out()));
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), second_signal.out()));
  uint64_t first_signal_value = 1;
  uint64_t second_signal_value = 1;
  iree_hal_semaphore_t* first_signal_pointer = first_signal.get();
  iree_hal_semaphore_t* second_signal_pointer = second_signal.get();
  iree_hal_semaphore_list_t first_signal_list =
      SemaphoreList(&first_signal_pointer, &first_signal_value);
  iree_hal_semaphore_list_t second_signal_list =
      SemaphoreList(&second_signal_pointer, &second_signal_value);

  IREE_ASSERT_OK(iree_hal_queue_read(
      slot_owner_queue, iree_hal_semaphore_list_empty(), first_signal_list,
      file, 0, first_buffer, 0, kTransferSize, IREE_HAL_READ_FLAG_NONE));
  controller.WaitForSubmitCount(1);
  QueuePhaseSignal queued_waiter(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_STAGING_WAITER_QUEUED);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseSignal::Observe,
                                                     &queued_waiter);
  IREE_ASSERT_OK(iree_hal_queue_read(
      waiter_queue, iree_hal_semaphore_list_empty(), second_signal_list, file,
      0, second_buffer, 0, kTransferSize, IREE_HAL_READ_FLAG_NONE));
  queued_waiter.WaitUntilObserved();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(queued_waiter.occurrence_count(), 1u);
  EXPECT_EQ(queued_waiter.value1(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staging_pool)));
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_EQ(queued_waiter.value0(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                host_queue->active_staging_transfer_head)));
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  ASSERT_TRUE(StagingPoolHasQueuedWaiter(staging_pool));

  // The second transfer's waiter callback consumes the first transfer's slot.
  // Reject its file submit so production decrements io_submit_count, enters
  // chunk_finish, and can be paused while the waiter remains in CALLBACK.
  controller.FailNextSubmit(IREE_STATUS_RESOURCE_EXHAUSTED);
  ChunkSlotReleaseLatch callback_latch(/*blocked_capture_count=*/2);
  PublisherTailLatch tail_latch(
      host_queue,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK);
  iree_hal_amdgpu_host_queue_staging_set_chunk_slot_release_observer(
      ChunkSlotReleaseLatch::Observe, &callback_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      PublisherTailLatch::Observe, &tail_latch);
  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackExitCount(1);
  callback_latch.WaitForCapturedCount(2);

  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_EQ(staging_pool->waiter_head, nullptr);
  EXPECT_EQ(staging_pool->waiter_tail, nullptr);
  EXPECT_EQ(staging_pool->claimed_waiter_count, 1u);
  EXPECT_EQ(staging_pool->available_count, 0u);
  iree_slim_mutex_unlock(&staging_pool->mutex);
  ControlledProactorSnapshot callback_snapshot = controller.Snapshot();
  EXPECT_EQ(callback_snapshot.submit_attempt_count, 2u);
  EXPECT_EQ(callback_snapshot.submit_count, 1u);
  EXPECT_EQ(callback_snapshot.submit_failure_count, 1u);
  EXPECT_EQ(callback_snapshot.pending_count, 0u);
  const uint64_t previous_query_count =
      callback_snapshot.capability_query_count;

  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  controller.WaitForCapabilityQueryAfter(previous_query_count);
  EXPECT_FALSE(seal_completion.IsDone());

  callback_latch.ReleaseBlocked();
  tail_latch.WaitForZeroBeforeWake();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(tail_latch.zero_before_wake_count(), 1u);

  // The stable waiter-state publication is not a callback join. The wake and
  // pool unlock must complete before after_fn consumes the waiter-held transfer
  // reference, and terminal publication remains blocked through that tail.
  tail_latch.ReleaseZeroBeforeWake();
  tail_latch.WaitForTailLeft();
  EXPECT_FALSE(seal_completion.IsDone());
  EXPECT_EQ(tail_latch.tail_left_count(), 1u);
  EXPECT_EQ(tail_latch.tail_left_value0(), 1u);
  EXPECT_EQ(tail_latch.tail_left_value1(), 1u);

  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  EXPECT_TRUE(host_queue->active_staging_transfer_head != nullptr ||
              host_queue->shutdown_staging_transfer_head != nullptr);
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);

  tail_latch.ReleaseTailLeft();
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  iree_hal_amdgpu_host_queue_staging_set_chunk_slot_release_observer(nullptr,
                                                                     nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OK, iree_hal_semaphore_wait(first_signal, first_signal_value,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_semaphore_wait(second_signal, second_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectQueueCertifiedAndEmpty(host_queue, staging_pool);
  iree_hal_amdgpu_host_queue_seal(slot_owner_host_queue);
  ExpectQueueCertifiedAndEmpty(slot_owner_host_queue, staging_pool);
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_attempt_count, 2u);
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.submit_failure_count, 1u);
  EXPECT_EQ(sealed_snapshot.cancel_count, 0u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(sealed_snapshot.release_without_pending_count, 0u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  waiter_queue.reset();
  slot_owner_queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       ChunkFinishReturnsCapturedSlotBeforeIdleChunkReuse) {
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&controller, &test_device,
                                  /*notification_capacity=*/0,
                                  /*staging_slot_count=*/2));

  Ref<iree_hal_queue_t> queue_a;
  Ref<iree_hal_queue_t> queue_b;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue_a.out()));
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue_b.out()));
  iree_hal_amdgpu_host_queue_t* host_queue_a = HostQueue(queue_a.get());
  iree_hal_amdgpu_host_queue_t* host_queue_b = HostQueue(queue_b.get());
  ASSERT_NE(host_queue_a, host_queue_b);
  iree_hal_amdgpu_staging_pool_t* staging_pool = test_device.staging_pool();
  ASSERT_EQ(staging_pool->slot_count, 2u);

  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(
      test_device.base_device(),
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, &path,
      file.out(), 3 * kStagingSlotSize));
  Ref<iree_hal_buffer_t> buffer_a;
  Ref<iree_hal_buffer_t> buffer_b;
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer_a.out(),
                              2 * kStagingSlotSize));
  IREE_ASSERT_OK(CreateBuffer(test_device.allocator(), false, buffer_b.out(),
                              kStagingSlotSize));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_pointer = signal.get();
  iree_hal_semaphore_list_t signal_list =
      SemaphoreList(&signal_pointer, &signal_value);

  // Queue B claims slot 0 first and parks its file write. Queue A then claims
  // slot 1 for its first chunk and queues its second chunk on the shared pool.
  // Separate queue runners are required here: safe actions on one queue are
  // intentionally serialized and cannot expose this cross-queue pool race.
  IREE_ASSERT_OK(iree_hal_queue_write(
      queue_b, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      buffer_b, 0, file, 0, kStagingSlotSize, IREE_HAL_WRITE_FLAG_NONE));
  controller.WaitForSubmitCount(1);
  QueuePhaseSignal queued_waiter(
      host_queue_a,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_STAGING_WAITER_QUEUED);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseSignal::Observe,
                                                     &queued_waiter);
  IREE_ASSERT_OK(iree_hal_queue_write(
      queue_a, iree_hal_semaphore_list_empty(), signal_list, buffer_a, 0, file,
      kStagingSlotSize, 2 * kStagingSlotSize, IREE_HAL_WRITE_FLAG_NONE));
  controller.WaitForSubmitCount(2);
  queued_waiter.WaitUntilObserved();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  ASSERT_EQ(controller.Snapshot().pending_count, 2u);
  EXPECT_EQ(queued_waiter.occurrence_count(), 1u);
  EXPECT_EQ(queued_waiter.value1(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staging_pool)));
  iree_slim_mutex_lock(&host_queue_a->locks.submission_mutex);
  EXPECT_EQ(queued_waiter.value0(),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                host_queue_a->active_staging_transfer_head)));
  iree_slim_mutex_unlock(&host_queue_a->locks.submission_mutex);
  ASSERT_TRUE(StagingPoolHasQueuedWaiter(staging_pool));

  iree_slim_mutex_lock(&host_queue_a->locks.submission_mutex);
  EXPECT_NE(host_queue_a->active_staging_transfer_head, nullptr);
  iree_slim_mutex_unlock(&host_queue_a->locks.submission_mutex);
  iree_slim_mutex_lock(&host_queue_b->locks.submission_mutex);
  EXPECT_NE(host_queue_b->active_staging_transfer_head, nullptr);
  iree_slim_mutex_unlock(&host_queue_b->locks.submission_mutex);
  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_NE(staging_pool->waiter_head, nullptr);
  EXPECT_NE(staging_pool->waiter_tail, nullptr);
  EXPECT_EQ(staging_pool->claimed_waiter_count, 0u);
  EXPECT_EQ(staging_pool->available_count, 0u);
  iree_slim_mutex_unlock(&staging_pool->mutex);

  ChunkSlotReleaseLatch release_latch;
  iree_hal_amdgpu_host_queue_staging_set_chunk_slot_release_observer(
      ChunkSlotReleaseLatch::Observe, &release_latch);
  QueuePhaseLatch waiter_tail_latch(
      host_queue_a,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGING_WAITER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT);
  QueuePhaseLatch transfer_tail_latch(
      host_queue_a,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_STAGED_TRANSFER_CALLBACK,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_TAIL_LEFT);
  QueuePhaseObserverGroup queue_phase_observers;
  queue_phase_observers.Add(QueuePhaseLatch::Observe, &waiter_tail_latch);
  queue_phase_observers.Add(QueuePhaseLatch::Observe, &transfer_tail_latch);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(
      QueuePhaseObserverGroup::Observe, &queue_phase_observers);

  // Complete queue A's later parked operation first. Its safe runner clears
  // the chunk that owned slot 1 and pauses before returning that captured
  // ordinal, leaving the chunk IDLE while the pool still has no free slots.
  ASSERT_TRUE(controller.ReleaseSuccessfulAt(/*parked_index=*/1));
  release_latch.WaitForCapturedCount(1);
  std::vector<uint32_t> first_capture = release_latch.CapturedOrdinals();
  ASSERT_EQ(first_capture.size(), 1u);
  EXPECT_EQ(first_capture[0], 1u);
  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_NE(staging_pool->waiter_head, nullptr);
  EXPECT_EQ(staging_pool->claimed_waiter_count, 0u);
  EXPECT_EQ(staging_pool->available_count, 0u);
  iree_slim_mutex_unlock(&staging_pool->mutex);
  iree_slim_mutex_lock(&host_queue_a->locks.submission_mutex);
  const uint64_t reuse_epoch_before =
      host_queue_a->notification_ring.epoch.next_submission;
  iree_slim_mutex_unlock(&host_queue_a->locks.submission_mutex);
  const int64_t reuse_aql_write_before = iree_atomic_load(
      host_queue_a->aql_ring.write_dispatch_id, iree_memory_order_acquire);

  // Complete queue B inline. Its independent queue-safe runner captures and
  // returns slot 0, then synchronously pumps queue A's waiter. The resulting
  // queue-A copy epoch is proof that its now-IDLE chunk was reused with slot 0
  // while queue A's original slot-1 release remains paused. Its later file
  // submit cannot occur yet because queue A's completion runner is the thread
  // deliberately paused by the observer.
  std::atomic<bool> inline_completion_succeeded{false};
  ThreadCompletion inline_completion;
  std::thread inline_completer([&] {
    inline_completion_succeeded.store(
        controller.CompleteNextSuccessfulInline());
    inline_completion.MarkDone();
  });
  release_latch.WaitForCapturedCount(2);
  waiter_tail_latch.WaitUntilEntered();
  // The proactor callback must have returned after handing the work to the
  // queue-safe runner even though that runner is still blocked in the waiter
  // callback tail below.
  inline_completion.Wait();
  inline_completer.join();
  EXPECT_TRUE(inline_completion_succeeded.load());
  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_EQ(staging_pool->waiter_head, nullptr);
  EXPECT_EQ(staging_pool->waiter_tail, nullptr);
  EXPECT_EQ(staging_pool->claimed_waiter_count, 0u);
  EXPECT_EQ(staging_pool->available_count, 0u);
  iree_slim_mutex_unlock(&staging_pool->mutex);
  iree_slim_mutex_lock(&host_queue_a->locks.submission_mutex);
  const uint64_t reuse_epoch_after =
      host_queue_a->notification_ring.epoch.next_submission;
  iree_slim_mutex_unlock(&host_queue_a->locks.submission_mutex);
  const int64_t reuse_aql_write_after = iree_atomic_load(
      host_queue_a->aql_ring.write_dispatch_id, iree_memory_order_acquire);
  EXPECT_GT(reuse_epoch_after, reuse_epoch_before);
  EXPECT_GT(reuse_aql_write_after, reuse_aql_write_before);
  std::vector<uint32_t> first_two_ordinals = release_latch.CapturedOrdinals();
  ASSERT_EQ(first_two_ordinals.size(), 2u);
  EXPECT_EQ(first_two_ordinals[0], 1u);
  EXPECT_EQ(first_two_ordinals[1], 0u);

  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_EQ(staging_pool->waiter_head, nullptr);
  EXPECT_EQ(staging_pool->waiter_tail, nullptr);
  EXPECT_EQ(staging_pool->claimed_waiter_count, 0u);
  EXPECT_EQ(staging_pool->available_count, 0u);
  iree_slim_mutex_unlock(&staging_pool->mutex);
  waiter_tail_latch.Release();

  // Queue A must return the slot-1 ordinal captured before its transfer unlock,
  // not reread the reused chunk's current slot 0. The old reread mutant returns
  // slot 0 twice and this exact free-list assertion fails.
  release_latch.ReleaseBlocked();
  transfer_tail_latch.WaitUntilEntered();

  uint32_t returned_ordinal = UINT32_MAX;
  iree_slim_mutex_lock(&staging_pool->mutex);
  EXPECT_EQ(staging_pool->available_count, 1u);
  if (staging_pool->available_count == 1) {
    returned_ordinal =
        staging_pool
            ->free_slots[staging_pool->free_read & staging_pool->slot_mask];
  }
  iree_slim_mutex_unlock(&staging_pool->mutex);
  EXPECT_EQ(returned_ordinal, 1u);
  transfer_tail_latch.Release();
  controller.WaitForSubmitCount(3);

  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackExitCount(3);
  release_latch.WaitForCapturedCount(3);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  iree_hal_amdgpu_host_queue_staging_set_chunk_slot_release_observer(nullptr,
                                                                     nullptr);
  std::vector<uint32_t> captured_ordinals = release_latch.CapturedOrdinals();
  EXPECT_EQ(captured_ordinals.size(), 3u);
  EXPECT_EQ(captured_ordinals[2], 0u);

  std::vector<uint32_t> free_ordinals;
  iree_slim_mutex_lock(&staging_pool->mutex);
  for (uint32_t i = 0; i < staging_pool->available_count; ++i) {
    free_ordinals.push_back(
        staging_pool->free_slots[(staging_pool->free_read + i) &
                                 staging_pool->slot_mask]);
  }
  iree_slim_mutex_unlock(&staging_pool->mutex);
  std::sort(free_ordinals.begin(), free_ordinals.end());
  EXPECT_EQ(free_ordinals, (std::vector<uint32_t>{0u, 1u}));

  iree_hal_amdgpu_host_queue_seal(host_queue_a);
  iree_hal_amdgpu_host_queue_seal(host_queue_b);
  ExpectQueueCertifiedAndEmpty(host_queue_a, staging_pool);
  ExpectQueueCertifiedAndEmpty(host_queue_b, staging_pool);
  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  EXPECT_EQ(sealed_snapshot.submit_count, 3u);
  EXPECT_EQ(sealed_snapshot.cancel_count, 0u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 3u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 3u);
  EXPECT_EQ(sealed_snapshot.release_without_pending_count, 0u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  queue_a.reset();
  queue_b.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
}

TEST_F(HostQueuePublisherLifetimeTest,
       SealDrainsCapacityRetryAndFinalDestructionIsInert) {
  ScopedHsaWaitCounter wait_counter(&libhsa_);
  ControlledProactorController controller(
      IREE_ASYNC_PROACTOR_CAPABILITY_FILE_OPERATION_CANCELLATION);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      CreateTestDevice(&controller, &test_device, /*notification_capacity=*/1));

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(test_device.AcquireDynamicQueue(queue.out()));
  iree_hal_amdgpu_host_queue_t* host_queue = HostQueue(queue.get());
  iree::testing::TempFilePath path;
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(CreateNativeFile(
      test_device.base_device(),
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, &path,
      file.out()));
  Ref<iree_hal_buffer_t> source_buffer;
  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(
      CreateBuffer(test_device.allocator(), false, source_buffer.out()));
  IREE_ASSERT_OK(
      CreateBuffer(test_device.allocator(), false, pressure_buffer.out()));

  Ref<iree_hal_semaphore_t> write_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), write_signal.out()));
  uint64_t write_signal_value = 1;
  iree_hal_semaphore_t* write_signal_pointer = write_signal.get();
  iree_hal_semaphore_list_t write_signal_list =
      SemaphoreList(&write_signal_pointer, &write_signal_value);
  IREE_ASSERT_OK(iree_hal_queue_write(
      queue, iree_hal_semaphore_list_empty(), write_signal_list, source_buffer,
      0, file, 0, kTransferSize, IREE_HAL_WRITE_FLAG_NONE));
  controller.WaitForSubmitCount(1);

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(host_queue, blocker_signal));
  RawQueueBlockerGuard blocker_guard(&libhsa_, host_queue, blocker_signal);

  // Publish one AQL fill behind the raw blocker so the sole notification slot
  // is occupied. A length greater than eight bypasses the PM4 WRITE_DATA fast
  // path; the following public fill must therefore enter the capacity-only
  // pending path rather than publish another packet.
  constexpr iree_device_size_t kPressureFillLength = 16;
  const uint32_t occupying_pattern = 0x4F434355u;
  const uint64_t occupying_epoch_before =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t occupying_last_published_before =
      QueueAxisFrontierWaiter::LoadCursor(
          &host_queue->notification_ring.epoch.last_published);
  const uint64_t occupying_notification_write_before =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  const uint64_t occupying_frontier_write_before =
      QueueAxisFrontierWaiter::LoadCursor(
          &host_queue->notification_ring.frontier_ring.write);
  const int64_t occupying_aql_write_before = iree_atomic_load(
      host_queue->aql_ring.write_dispatch_id, iree_memory_order_acquire);
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_fill(
      host_queue, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), pressure_buffer, 0, kPressureFillLength,
      occupying_pattern, sizeof(occupying_pattern), IREE_HAL_FILL_FLAG_NONE));
  ASSERT_EQ(host_queue->notification_ring.epoch.next_submission,
            occupying_epoch_before + 1);
  ASSERT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.epoch.last_published),
            occupying_last_published_before + 1);
  ASSERT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      occupying_notification_write_before);
  ASSERT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.frontier_ring.write),
            occupying_frontier_write_before);
  ASSERT_EQ(iree_atomic_load(host_queue->aql_ring.write_dispatch_id,
                             iree_memory_order_acquire),
            occupying_aql_write_before + 1);
  const uint64_t pressure_last_drained = static_cast<uint64_t>(
      iree_atomic_load(&host_queue->notification_ring.epoch.last_drained,
                       iree_memory_order_acquire));
  ASSERT_EQ(host_queue->notification_ring.epoch.next_submission -
                pressure_last_drained,
            1u);

  const uint64_t pressure_epoch_before =
      host_queue->notification_ring.epoch.next_submission;
  const uint64_t pressure_last_published_before =
      QueueAxisFrontierWaiter::LoadCursor(
          &host_queue->notification_ring.epoch.last_published);
  const uint64_t pressure_notification_write_before =
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write);
  const int64_t pressure_aql_write_before = iree_atomic_load(
      host_queue->aql_ring.write_dispatch_id, iree_memory_order_acquire);
  QueuePhaseSignal capacity_retry(
      host_queue, IREE_HAL_AMDGPU_HOST_QUEUE_TEST_SUBJECT_PENDING_START_HANDOFF,
      IREE_HAL_AMDGPU_HOST_QUEUE_TEST_PHASE_CAPACITY_COMPLETING_BEFORE_ENQUEUE);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(QueuePhaseSignal::Observe,
                                                     &capacity_retry);
  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_pointer = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list =
      SemaphoreList(&pressure_signal_pointer, &pressure_signal_value);
  const uint32_t pressure_pattern = 0xA5A55A5Au;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_fill(
      host_queue, iree_hal_semaphore_list_empty(), pressure_signal_list,
      pressure_buffer, 0, kPressureFillLength, pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE));
  capacity_retry.WaitUntilObserved();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  EXPECT_EQ(capacity_retry.occurrence_count(), 1u);
  ASSERT_EQ(CountPendingOperations(host_queue), 1u);
  iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
  ASSERT_NE(host_queue->pending_head, nullptr);
  EXPECT_EQ(capacity_retry.value0(),
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(host_queue->pending_head)));
  iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission,
            pressure_epoch_before);
  EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.epoch.last_published),
            pressure_last_published_before);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      pressure_notification_write_before);
  EXPECT_EQ(iree_atomic_load(host_queue->aql_ring.write_dispatch_id,
                             iree_memory_order_acquire),
            pressure_aql_write_before);

  controller.ReleaseNextSuccessful();
  controller.WaitForCallbackExitCount(1);
  EXPECT_EQ(CountPendingOperations(host_queue), 1u);

  SealOwnerLatch seal_owner_latch(host_queue);
  iree_hal_amdgpu_host_queue_test_set_phase_observer(SealOwnerLatch::Observe,
                                                     &seal_owner_latch);
  ThreadCompletion seal_completion;
  std::thread sealer([&] {
    iree_hal_amdgpu_host_queue_seal(host_queue);
    seal_completion.MarkDone();
  });
  seal_owner_latch.WaitUntilEntered();
  EXPECT_FALSE(seal_completion.IsDone());

  // The operation was accepted before closure but never published. While the
  // raw blocker keeps native queue storage live, seal must drain its capacity
  // continuation so pending_op_issue observes CLOSED, fails the signal, and
  // retires every retained edge without emitting a packet or advancing a
  // public cursor.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(pressure_signal, pressure_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  EXPECT_EQ(host_queue->notification_ring.epoch.next_submission,
            pressure_epoch_before);
  EXPECT_EQ(QueueAxisFrontierWaiter::LoadCursor(
                &host_queue->notification_ring.epoch.last_published),
            pressure_last_published_before);
  EXPECT_EQ(
      QueueAxisFrontierWaiter::LoadCursor(&host_queue->notification_ring.write),
      pressure_notification_write_before);
  EXPECT_EQ(iree_atomic_load(host_queue->aql_ring.write_dispatch_id,
                             iree_memory_order_acquire),
            pressure_aql_write_before);

  blocker_guard.Release();
  seal_completion.Wait();
  sealer.join();
  iree_hal_amdgpu_host_queue_test_set_phase_observer(nullptr, nullptr);
  IREE_EXPECT_OK(blocker_guard.DestroyAfterSeal());

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(write_signal, write_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(CountPendingOperations(host_queue), 0u);
  ExpectQueueCertifiedAndEmpty(host_queue, test_device.staging_pool());

  ControlledProactorSnapshot sealed_snapshot = controller.Snapshot();
  const uint64_t sealed_wait_count = wait_counter.count();
  EXPECT_EQ(sealed_snapshot.submit_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_entry_count, 1u);
  EXPECT_EQ(sealed_snapshot.callback_exit_count, 1u);
  EXPECT_EQ(sealed_snapshot.pending_count, 0u);

  controller.MarkSealed();
  queue.reset();
  ExpectControllerUnchanged(sealed_snapshot, controller.Snapshot());
  EXPECT_EQ(wait_counter.count(), sealed_wait_count);
}

#else

TEST_F(HostQueuePublisherLifetimeTest, FileIoDisabled) {
  GTEST_SKIP() << "file I/O is disabled";
}

#endif  // IREE_FILE_IO_ENABLE

}  // namespace
}  // namespace iree::hal::amdgpu
