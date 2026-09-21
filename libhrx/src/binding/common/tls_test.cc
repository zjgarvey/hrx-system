// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/tls.h"

#include <atomic>
#include <future>
#include <thread>

#include "common/internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
#include <windows.h>
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if !defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
#error "tls_test requires the instrumented common provider"
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

namespace {

struct CountingAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  std::atomic<int> allocation_count{0};
  std::atomic<int> free_count{0};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<CountingAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      allocator->allocation_count.fetch_add(1, std::memory_order_acq_rel);
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      allocator->free_count.fetch_add(1, std::memory_order_acq_rel);
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &CountingAllocator::Control};
  }
};

struct ReentrantCleanupAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  iree_hal_streaming_context_t* reentrant_context = nullptr;
  std::atomic<bool> observe_next_free{true};
  std::atomic<int> global_count_before_free{-1};
  std::atomic<int> global_count_after_free{-1};
  std::atomic<int> local_count_before_reentry{-1};
  std::atomic<int> global_count_after_reentry{-1};
  std::atomic<int> local_count_after_reentry{-1};
  std::atomic<int> reentry_status{IREE_STATUS_UNKNOWN};
  std::atomic<bool> current_was_detached{false};
  std::atomic<bool> marker_was_detached{false};
  std::atomic<bool> marker_after_reentry{false};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ReentrantCleanupAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE ||
        !allocator->observe_next_free.exchange(false,
                                               std::memory_order_acq_rel)) {
      return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                     inout_ptr);
    }

    allocator->global_count_before_free.store(
        static_cast<int>(iree_hal_streaming_context_tls_reference_count()),
        std::memory_order_release);
    allocator->local_count_before_reentry.store(
        static_cast<int>(
            iree_hal_streaming_context_current_thread_tls_reference_count()),
        std::memory_order_release);
    allocator->current_was_detached.store(
        iree_hal_streaming_context_current() == nullptr,
        std::memory_order_release);
    allocator->marker_was_detached.store(
        !iree_hal_streaming_context_tls_test_marker_is_set(),
        std::memory_order_release);

    iree_status_t status = allocator->delegate.ctl(allocator->delegate.self,
                                                   command, params, inout_ptr);
    allocator->global_count_after_free.store(
        static_cast<int>(iree_hal_streaming_context_tls_reference_count()),
        std::memory_order_release);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_streaming_context_set_current(allocator->reentrant_context);
    }
    allocator->reentry_status.store(iree_status_code(status),
                                    std::memory_order_release);
    iree_status_ignore(status);
    allocator->global_count_after_reentry.store(
        static_cast<int>(iree_hal_streaming_context_tls_reference_count()),
        std::memory_order_release);
    allocator->local_count_after_reentry.store(
        static_cast<int>(
            iree_hal_streaming_context_current_thread_tls_reference_count()),
        std::memory_order_release);
    allocator->marker_after_reentry.store(
        iree_hal_streaming_context_tls_test_marker_is_set(),
        std::memory_order_release);
    return iree_ok_status();
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &ReentrantCleanupAllocator::Control};
  }
};

struct ReinstallingDestructorState {
  iree_hal_streaming_tls_key_t key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  std::atomic<int> call_count{0};
  std::atomic<int> reinstall_status{IREE_STATUS_OK};
};

static void ReinstallingDestructor(void* value) {
  auto* state = static_cast<ReinstallingDestructorState*>(value);
  const int call_count =
      state->call_count.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (call_count < 4) {
    iree_status_t status = iree_hal_streaming_tls_set(state->key, state);
    state->reinstall_status.store(iree_status_code(status),
                                  std::memory_order_release);
    iree_status_ignore(status);
  }
}

#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
struct WindowsFiberIsolationState {
  void* caller_fiber = nullptr;
  ReinstallingDestructorState caller_state;
  ReinstallingDestructorState target_state;
  std::atomic<int> target_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> caller_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> caller_clear_status{IREE_STATUS_UNKNOWN};
  std::atomic<DWORD> win32_error{ERROR_SUCCESS};
  void* caller_value_after_target_delete = nullptr;
};

static VOID WINAPI WindowsTlsTargetFiber(void* raw_state) {
  auto* state = static_cast<WindowsFiberIsolationState*>(raw_state);
  iree_status_t status =
      iree_hal_streaming_tls_set(state->target_state.key, &state->target_state);
  state->target_set_status.store(iree_status_code(status),
                                 std::memory_order_release);
  iree_status_ignore(status);
  SwitchToFiber(state->caller_fiber);
}

enum class WindowsNestedValueKind {
  kCallerA,
  kCallerB,
  kTargetA,
  kTargetB,
  kNestedA,
};

struct WindowsNestedFrameState;
struct WindowsNestedValue {
  WindowsNestedFrameState* state;
  WindowsNestedValueKind kind;
};

struct WindowsNestedFrameState {
  iree_hal_streaming_tls_key_t key_a = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  iree_hal_streaming_tls_key_t key_b = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  void* caller_fiber = nullptr;
  void* target_b_fiber = nullptr;
  WindowsNestedValue caller_a{this, WindowsNestedValueKind::kCallerA};
  WindowsNestedValue caller_b{this, WindowsNestedValueKind::kCallerB};
  WindowsNestedValue target_a{this, WindowsNestedValueKind::kTargetA};
  WindowsNestedValue target_b{this, WindowsNestedValueKind::kTargetB};
  WindowsNestedValue nested_a{this, WindowsNestedValueKind::kNestedA};
  std::atomic<int> caller_a_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> caller_b_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> target_a_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> target_b_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> nested_a_set_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> caller_a_clear_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> caller_b_clear_status{IREE_STATUS_UNKNOWN};
  std::atomic<int> nested_a_destructor_count{0};
  std::atomic<bool> nested_a_get_matched{false};
  std::atomic<bool> target_b_deleted_by_outer_callback{false};
  std::atomic<DWORD> win32_error{ERROR_SUCCESS};
  void* caller_a_after_nested_delete = nullptr;
  void* caller_b_after_nested_delete = nullptr;
};

static void WindowsNestedADestructor(void* raw_value) {
  auto* value = static_cast<WindowsNestedValue*>(raw_value);
  WindowsNestedFrameState* state = value->state;
  if (value->kind == WindowsNestedValueKind::kTargetA) {
    void* target_b_fiber = state->target_b_fiber;
    state->target_b_fiber = nullptr;
    DeleteFiber(target_b_fiber);
    state->target_b_deleted_by_outer_callback.store(true,
                                                    std::memory_order_release);
  } else if (value->kind == WindowsNestedValueKind::kNestedA) {
    state->nested_a_destructor_count.fetch_add(1, std::memory_order_acq_rel);
  }
}

static void WindowsNestedBDestructor(void* raw_value) {
  auto* value = static_cast<WindowsNestedValue*>(raw_value);
  WindowsNestedFrameState* state = value->state;
  if (value->kind != WindowsNestedValueKind::kTargetB) return;
  iree_status_t status =
      iree_hal_streaming_tls_set(state->key_a, &state->nested_a);
  state->nested_a_set_status.store(iree_status_code(status),
                                   std::memory_order_release);
  iree_status_ignore(status);
  state->nested_a_get_matched.store(
      iree_hal_streaming_tls_get(state->key_a) == &state->nested_a,
      std::memory_order_release);
}

static VOID WINAPI WindowsNestedTargetFiber(void* raw_value) {
  auto* value = static_cast<WindowsNestedValue*>(raw_value);
  WindowsNestedFrameState* state = value->state;
  const bool is_a = value->kind == WindowsNestedValueKind::kTargetA;
  iree_status_t status =
      iree_hal_streaming_tls_set(is_a ? state->key_a : state->key_b, value);
  (is_a ? state->target_a_set_status : state->target_b_set_status)
      .store(iree_status_code(status), std::memory_order_release);
  iree_status_ignore(status);
  SwitchToFiber(state->caller_fiber);
}
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

static void ExpectIdentityAllocatorSaturates(uint64_t initial,
                                             uint64_t max_valid) {
  iree_atomic_uint64_t next_id = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_store(&next_id, initial, iree_memory_order_relaxed);

  uint64_t id = 0;
  ASSERT_TRUE(iree_hal_streaming_atomic_allocate_id(&next_id, max_valid, &id));
  EXPECT_EQ(initial, id);
  ASSERT_TRUE(iree_hal_streaming_atomic_allocate_id(&next_id, max_valid, &id));
  EXPECT_EQ(max_valid, id);

  // Exhaustion is latched. Repeated calls never advance through zero and can
  // therefore never return the process's first identity again.
  EXPECT_FALSE(iree_hal_streaming_atomic_allocate_id(&next_id, max_valid, &id));
  EXPECT_EQ(0u, id);
  EXPECT_FALSE(iree_hal_streaming_atomic_allocate_id(&next_id, max_valid, &id));
  EXPECT_EQ(0u, id);
  EXPECT_EQ(0u, iree_atomic_load(&next_id, iree_memory_order_relaxed));
}

TEST(StreamingIdentityTest, RuntimeGenerationSaturatesPermanently) {
  ExpectIdentityAllocatorSaturates(UINT64_MAX - 1, UINT64_MAX);
}

TEST(StreamingIdentityTest, PointerBufferIdSaturatesPermanently) {
  ExpectIdentityAllocatorSaturates(UINT32_MAX - 1, UINT32_MAX);
}

TEST(StreamingIdentityTest, CapabilityIdReservesTerminalSentinel) {
  ExpectIdentityAllocatorSaturates(UINT64_MAX - 2, UINT64_MAX - 1);
}

class StreamingContextTlsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_streaming_context_tls_test_reset());
    context_ = {};
    iree_atomic_ref_count_init(&context_.ref_count);
    context_.host_allocator = iree_allocator_system();
  }

  void TearDown() override {
    iree_hal_streaming_tls_test_inject_failures(
        IREE_HAL_STREAMING_TLS_TEST_FAILURE_NONE);
    IREE_EXPECT_OK(iree_hal_streaming_context_clear_current_thread());
    EXPECT_EQ(1, iree_atomic_ref_count_load(&context_.ref_count));
    IREE_EXPECT_OK(iree_hal_streaming_context_tls_test_reset());
  }

  void ExpectUnreferenced() {
    EXPECT_EQ(nullptr, iree_hal_streaming_context_current());
    EXPECT_EQ(0u,
              iree_hal_streaming_context_current_thread_tls_reference_count());
    EXPECT_EQ(0u, iree_hal_streaming_context_tls_reference_count());
    EXPECT_EQ(0u,
              iree_hal_streaming_context_current_thread_tls_reference_count_for(
                  &context_));
    EXPECT_EQ(1, iree_atomic_ref_count_load(&context_.ref_count));
    EXPECT_FALSE(iree_hal_streaming_context_tls_test_marker_is_set());
  }

  void ExpectOneReference() {
    EXPECT_EQ(&context_, iree_hal_streaming_context_current());
    EXPECT_EQ(1u,
              iree_hal_streaming_context_current_thread_tls_reference_count());
    EXPECT_EQ(1u, iree_hal_streaming_context_tls_reference_count());
    EXPECT_EQ(1u,
              iree_hal_streaming_context_current_thread_tls_reference_count_for(
                  &context_));
    EXPECT_EQ(2, iree_atomic_ref_count_load(&context_.ref_count));
    EXPECT_TRUE(iree_hal_streaming_context_tls_test_marker_is_set());
  }

  iree_hal_streaming_context_t context_ = {};
};

TEST_F(StreamingContextTlsTest, KeyCreationFailureIsReturnedAndRetryable) {
  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_CREATE);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_tls_initialize());
  ExpectUnreferenced();

  IREE_EXPECT_OK(iree_hal_streaming_context_tls_initialize());
  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       SetAndClearFailuresPreserveCurrentReferenceTransaction) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());

  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_SET);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_set_current(&context_));
  ExpectUnreferenced();

  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
  ExpectOneReference();

  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_set_current(nullptr));
  ExpectOneReference();

  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(nullptr));
  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       ClearCurrentThreadFailurePreservesCurrentReferenceTransaction) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
  ExpectOneReference();

  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_clear_current_thread());
  ExpectOneReference();

  IREE_ASSERT_OK(iree_hal_streaming_context_clear_current_thread());
  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       StackedClearFailurePreservesCompleteStateTransaction) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  CountingAllocator allocator;
  context_.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_context_t second_context = {};
  iree_atomic_ref_count_init(&second_context.ref_count);
  second_context.host_allocator = allocator.AsAllocator();

  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
  IREE_ASSERT_OK(iree_hal_streaming_context_push(&second_context));
  ASSERT_EQ(&second_context, iree_hal_streaming_context_current());
  ASSERT_EQ(2u,
            iree_hal_streaming_context_current_thread_tls_reference_count());
  ASSERT_EQ(2u, iree_hal_streaming_context_tls_reference_count());
  ASSERT_EQ(1u,
            iree_hal_streaming_context_current_thread_tls_reference_count_for(
                &context_));
  ASSERT_EQ(1u,
            iree_hal_streaming_context_current_thread_tls_reference_count_for(
                &second_context));
  ASSERT_EQ(2, iree_atomic_ref_count_load(&context_.ref_count));
  ASSERT_EQ(2, iree_atomic_ref_count_load(&second_context.ref_count));
  ASSERT_EQ(0, allocator.free_count.load(std::memory_order_acquire));

  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_clear_current_thread());

  EXPECT_EQ(&second_context, iree_hal_streaming_context_current());
  EXPECT_EQ(2u,
            iree_hal_streaming_context_current_thread_tls_reference_count());
  EXPECT_EQ(2u, iree_hal_streaming_context_tls_reference_count());
  EXPECT_EQ(1u,
            iree_hal_streaming_context_current_thread_tls_reference_count_for(
                &context_));
  EXPECT_EQ(1u,
            iree_hal_streaming_context_current_thread_tls_reference_count_for(
                &second_context));
  EXPECT_EQ(2, iree_atomic_ref_count_load(&context_.ref_count));
  EXPECT_EQ(2, iree_atomic_ref_count_load(&second_context.ref_count));
  EXPECT_EQ(0, allocator.free_count.load(std::memory_order_acquire));
  EXPECT_TRUE(iree_hal_streaming_context_tls_test_marker_is_set());

  IREE_ASSERT_OK(iree_hal_streaming_context_clear_current_thread());
  ExpectUnreferenced();
  EXPECT_EQ(1, iree_atomic_ref_count_load(&second_context.ref_count));
  EXPECT_EQ(1, allocator.free_count.load(std::memory_order_acquire));
}

TEST_F(StreamingContextTlsTest,
       PushAndPopFailuresPreserveStackReferenceTransaction) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  CountingAllocator allocator;
  context_.host_allocator = allocator.AsAllocator();

  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_SET);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_push(&context_));
  ExpectUnreferenced();
  EXPECT_EQ(1, allocator.allocation_count.load(std::memory_order_acquire));
  EXPECT_EQ(1, allocator.free_count.load(std::memory_order_acquire));

  IREE_ASSERT_OK(iree_hal_streaming_context_push(&context_));
  ExpectOneReference();
  EXPECT_EQ(2, allocator.allocation_count.load(std::memory_order_acquire));
  EXPECT_EQ(1, allocator.free_count.load(std::memory_order_acquire));

  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  iree_hal_streaming_context_t* popped_context =
      reinterpret_cast<iree_hal_streaming_context_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_context_pop(&popped_context));
  EXPECT_EQ(nullptr, popped_context);
  ExpectOneReference();
  EXPECT_EQ(1, allocator.free_count.load(std::memory_order_acquire));

  IREE_ASSERT_OK(iree_hal_streaming_context_pop(&popped_context));
  EXPECT_EQ(&context_, popped_context);
  ExpectUnreferenced();
  EXPECT_EQ(2, allocator.free_count.load(std::memory_order_acquire));
}

TEST_F(StreamingContextTlsTest, ThreadExitUsesAlreadyClearedMarkerCleanupPath) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());

  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
    EXPECT_TRUE(iree_hal_streaming_context_tls_test_marker_is_set());
    EXPECT_EQ(1u,
              iree_hal_streaming_context_current_thread_tls_reference_count());
  });
  thread.join();

  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       ThreadExitFreesStackWithItsAllocatingContextAllocator) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  CountingAllocator allocator;
  context_.host_allocator = allocator.AsAllocator();

  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_context_push(&context_));
    EXPECT_TRUE(iree_hal_streaming_context_tls_test_marker_is_set());
  });
  thread.join();

  ExpectUnreferenced();
  EXPECT_EQ(1, allocator.allocation_count.load(std::memory_order_acquire));
  EXPECT_EQ(1, allocator.free_count.load(std::memory_order_acquire));
}

TEST_F(StreamingContextTlsTest,
       ThreadExitHoldsGateAndSupportsReentrantCleanup) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  iree_hal_streaming_context_t reentrant_context = {};
  iree_atomic_ref_count_init(&reentrant_context.ref_count);
  reentrant_context.host_allocator = iree_allocator_system();
  ReentrantCleanupAllocator allocator;
  allocator.reentrant_context = &reentrant_context;
  context_.host_allocator = allocator.AsAllocator();

  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_context_push(&context_));
    EXPECT_EQ(1u,
              iree_hal_streaming_context_current_thread_tls_reference_count());
  });
  thread.join();

  EXPECT_EQ(1,
            allocator.global_count_before_free.load(std::memory_order_acquire));
  EXPECT_EQ(1,
            allocator.global_count_after_free.load(std::memory_order_acquire));
  EXPECT_EQ(
      0, allocator.local_count_before_reentry.load(std::memory_order_acquire));
  EXPECT_TRUE(allocator.current_was_detached.load(std::memory_order_acquire));
  EXPECT_TRUE(allocator.marker_was_detached.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            allocator.reentry_status.load(std::memory_order_acquire));
  EXPECT_EQ(
      2, allocator.global_count_after_reentry.load(std::memory_order_acquire));
  EXPECT_EQ(
      1, allocator.local_count_after_reentry.load(std::memory_order_acquire));
  EXPECT_TRUE(allocator.marker_after_reentry.load(std::memory_order_acquire));
  EXPECT_EQ(1, iree_atomic_ref_count_load(&context_.ref_count));
  EXPECT_EQ(1, iree_atomic_ref_count_load(&reentrant_context.ref_count));
  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       CleanupRejectsSetAndPushOwnershipPublicationInFlight) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  std::promise<void> admitted_promise;
  std::future<void> admitted = admitted_promise.get_future();
  std::promise<void> proceed_promise;
  std::shared_future<void> proceed = proceed_promise.get_future().share();
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_context_publication_begin());
    admitted_promise.set_value();
    proceed.wait();

    // CUDA holds the admission pin across both of these ownership-add
    // transitions. Cleanup must reject before either transition is allowed to
    // cross its ownership snapshot.
    IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
    IREE_ASSERT_OK(iree_hal_streaming_context_push(&context_));
    IREE_EXPECT_OK(iree_hal_streaming_context_clear_current_thread());
    iree_hal_streaming_context_publication_end();
  });
  admitted.wait();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_cleanup_global());
  EXPECT_EQ(0u, iree_hal_streaming_context_tls_reference_count());
  EXPECT_EQ(1, iree_atomic_ref_count_load(&context_.ref_count));

  proceed_promise.set_value();
  thread.join();
  ExpectUnreferenced();

  // A rejected cleanup reopens the gate exactly; a later CUDA ownership-add
  // transition can retry instead of inheriting a permanently closed runtime.
  IREE_ASSERT_OK(iree_hal_streaming_context_publication_begin());
  iree_hal_streaming_context_publication_end();
}

TEST_F(StreamingContextTlsTest, CleanupRejectsForeignTlsOwnerPrecommit) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  std::promise<void> ready_promise;
  std::future<void> ready = ready_promise.get_future();
  std::promise<void> release_promise;
  std::shared_future<void> release = release_promise.get_future().share();
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
    ready_promise.set_value();
    release.wait();
    IREE_EXPECT_OK(iree_hal_streaming_context_clear_current_thread());
  });
  ready.wait();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_cleanup_global());
  EXPECT_EQ(1u, iree_hal_streaming_context_tls_reference_count());
  EXPECT_EQ(0u,
            iree_hal_streaming_context_current_thread_tls_reference_count());
  EXPECT_EQ(2, iree_atomic_ref_count_load(&context_.ref_count));

  release_promise.set_value();
  thread.join();
  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       CleanupReturnsMarkerClearFailureAndRemainsRetryable) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(&context_));
  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_cleanup_global());
  ExpectOneReference();

  IREE_EXPECT_OK(iree_hal_streaming_cleanup_global());
  ExpectUnreferenced();
}

TEST_F(StreamingContextTlsTest,
       CleanupKeyDeleteFailureKeepsRegistryAndKeyRetryable) {
  IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  iree_hal_streaming_device_registry_t registry = {};
  iree_hal_streaming_test_install_device_registry(&registry,
                                                  iree_allocator_system());
  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_DELETE);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_cleanup_global());
  EXPECT_EQ(&registry, iree_hal_streaming_device_registry());
  EXPECT_FALSE(iree_hal_streaming_context_tls_test_marker_is_set());

  // The failed delete retained the native key, so ordinary TLS use remains
  // valid before a cleanup retry.
  IREE_EXPECT_OK(iree_hal_streaming_context_set_current(&context_));
  ExpectOneReference();
  IREE_EXPECT_OK(iree_hal_streaming_context_clear_current_thread());
  ExpectUnreferenced();

  iree_hal_streaming_test_remove_device_registry(&registry);
  IREE_EXPECT_OK(iree_hal_streaming_cleanup_global());
}

TEST(StreamingTlsTest, SetGetCurrentThread) {
  iree_hal_streaming_tls_key_t key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&key, nullptr));

  int main_value = 1;
  int thread_value = 2;
  IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &main_value));
  EXPECT_EQ(&main_value, iree_hal_streaming_tls_get(key));

  std::thread thread([&]() {
    EXPECT_EQ(nullptr, iree_hal_streaming_tls_get(key));
    IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &thread_value));
    EXPECT_EQ(&thread_value, iree_hal_streaming_tls_get(key));
  });
  thread.join();

  EXPECT_EQ(&main_value, iree_hal_streaming_tls_get(key));
  IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(key));
}

TEST(StreamingTlsTest, DestructorRunsOnThreadExit) {
  iree_hal_streaming_tls_key_t key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(
      &key, +[](void* value) {
        auto* destructor_count = static_cast<std::atomic<int>*>(value);
        destructor_count->fetch_add(1, std::memory_order_acq_rel);
      }));

  std::atomic<int> destructor_count{0};
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &destructor_count));
    EXPECT_EQ(&destructor_count, iree_hal_streaming_tls_get(key));
  });
  thread.join();

  EXPECT_EQ(1, destructor_count.load(std::memory_order_acquire));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(key));
}

TEST(StreamingTlsTest, DestructorReinstallRunsFourPasses) {
  ReinstallingDestructorState state;
  IREE_ASSERT_OK(
      iree_hal_streaming_tls_key_create(&state.key, ReinstallingDestructor));

  std::thread thread(
      [&]() { IREE_ASSERT_OK(iree_hal_streaming_tls_set(state.key, &state)); });
  thread.join();

  EXPECT_EQ(4, state.call_count.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.reinstall_status.load(std::memory_order_acquire));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(state.key));
}

#if defined(IREE_PLATFORM_WINDOWS) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
TEST(StreamingTlsTest, NoncurrentFiberDestructorDoesNotMutateCallerFiber) {
  WindowsFiberIsolationState state;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&state.target_state.key,
                                                   ReinstallingDestructor));
  state.caller_state.key = state.target_state.key;
  state.caller_state.call_count.store(4, std::memory_order_release);

  std::thread thread([&]() {
    state.caller_fiber = ConvertThreadToFiber(nullptr);
    if (!state.caller_fiber) {
      state.win32_error.store(GetLastError(), std::memory_order_release);
      return;
    }

    iree_status_t status =
        iree_hal_streaming_tls_set(state.caller_state.key, &state.caller_state);
    state.caller_set_status.store(iree_status_code(status),
                                  std::memory_order_release);
    const bool caller_set = iree_status_is_ok(status);
    iree_status_ignore(status);

    void* target_fiber = nullptr;
    if (caller_set) {
      target_fiber =
          CreateFiber(/*dwStackSize=*/0, WindowsTlsTargetFiber, &state);
      if (!target_fiber) {
        state.win32_error.store(GetLastError(), std::memory_order_release);
      } else {
        SwitchToFiber(target_fiber);
        DeleteFiber(target_fiber);
        state.caller_value_after_target_delete =
            iree_hal_streaming_tls_get(state.caller_state.key);
      }
    }

    status =
        iree_hal_streaming_tls_set(state.caller_state.key, /*value=*/nullptr);
    state.caller_clear_status.store(iree_status_code(status),
                                    std::memory_order_release);
    iree_status_ignore(status);
    if (!ConvertFiberToThread()) {
      state.win32_error.store(GetLastError(), std::memory_order_release);
    }
  });
  thread.join();

  EXPECT_EQ(ERROR_SUCCESS, state.win32_error.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.caller_set_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.target_set_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.caller_clear_status.load(std::memory_order_acquire));
  EXPECT_EQ(&state.caller_state, state.caller_value_after_target_delete);
  EXPECT_EQ(4, state.caller_state.call_count.load(std::memory_order_acquire));
  EXPECT_EQ(4, state.target_state.call_count.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK, state.target_state.reinstall_status.load(
                                std::memory_order_acquire));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(state.target_state.key));
}

TEST(StreamingTlsTest, NestedFiberCallbacksFindOuterKeyFrame) {
  WindowsNestedFrameState state;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&state.key_a,
                                                   WindowsNestedADestructor));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&state.key_b,
                                                   WindowsNestedBDestructor));

  std::thread thread([&]() {
    state.caller_fiber = ConvertThreadToFiber(nullptr);
    if (!state.caller_fiber) {
      state.win32_error.store(GetLastError(), std::memory_order_release);
      return;
    }

    iree_status_t status =
        iree_hal_streaming_tls_set(state.key_a, &state.caller_a);
    state.caller_a_set_status.store(iree_status_code(status),
                                    std::memory_order_release);
    const bool caller_a_set = iree_status_is_ok(status);
    iree_status_ignore(status);
    status = iree_hal_streaming_tls_set(state.key_b, &state.caller_b);
    state.caller_b_set_status.store(iree_status_code(status),
                                    std::memory_order_release);
    const bool caller_b_set = iree_status_is_ok(status);
    iree_status_ignore(status);

    void* target_a_fiber = nullptr;
    if (caller_a_set && caller_b_set) {
      state.target_b_fiber = CreateFiber(
          /*dwStackSize=*/0, WindowsNestedTargetFiber, &state.target_b);
      if (state.target_b_fiber) {
        SwitchToFiber(state.target_b_fiber);
        target_a_fiber = CreateFiber(/*dwStackSize=*/0,
                                     WindowsNestedTargetFiber, &state.target_a);
      }
      if (!state.target_b_fiber || !target_a_fiber) {
        state.win32_error.store(GetLastError(), std::memory_order_release);
      } else {
        SwitchToFiber(target_a_fiber);
        DeleteFiber(target_a_fiber);
        state.caller_a_after_nested_delete =
            iree_hal_streaming_tls_get(state.key_a);
        state.caller_b_after_nested_delete =
            iree_hal_streaming_tls_get(state.key_b);
      }
    }

    // Keep failure cleanup deterministic if the outer callback did not delete
    // the target-B fiber as expected.
    if (state.target_b_fiber) {
      DeleteFiber(state.target_b_fiber);
      state.target_b_fiber = nullptr;
    }
    status = iree_hal_streaming_tls_set(state.key_a, /*value=*/nullptr);
    state.caller_a_clear_status.store(iree_status_code(status),
                                      std::memory_order_release);
    iree_status_ignore(status);
    status = iree_hal_streaming_tls_set(state.key_b, /*value=*/nullptr);
    state.caller_b_clear_status.store(iree_status_code(status),
                                      std::memory_order_release);
    iree_status_ignore(status);
    if (!ConvertFiberToThread()) {
      state.win32_error.store(GetLastError(), std::memory_order_release);
    }
  });
  thread.join();

  EXPECT_EQ(ERROR_SUCCESS, state.win32_error.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.caller_a_set_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.caller_b_set_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.target_a_set_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.target_b_set_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.nested_a_set_status.load(std::memory_order_acquire));
  EXPECT_TRUE(state.nested_a_get_matched.load(std::memory_order_acquire));
  EXPECT_TRUE(
      state.target_b_deleted_by_outer_callback.load(std::memory_order_acquire));
  EXPECT_EQ(1, state.nested_a_destructor_count.load(std::memory_order_acquire));
  EXPECT_EQ(&state.caller_a, state.caller_a_after_nested_delete);
  EXPECT_EQ(&state.caller_b, state.caller_b_after_nested_delete);
  EXPECT_EQ(IREE_STATUS_OK,
            state.caller_a_clear_status.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK,
            state.caller_b_clear_status.load(std::memory_order_acquire));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(state.key_b));
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(state.key_a));
}
#endif  // IREE_PLATFORM_WINDOWS && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

TEST(StreamingTlsTest, KeyDeleteFailurePreservesValueForRetry) {
  iree_hal_streaming_tls_key_t key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&key, nullptr));
  int value = 1;
  IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &value));
  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_DELETE);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_tls_key_delete(key));
  EXPECT_EQ(&value, iree_hal_streaming_tls_get(key));

  IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(key));
  EXPECT_EQ(nullptr, iree_hal_streaming_tls_get(key));
}

class StreamingPrimaryContextFailureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_streaming_context_tls_test_reset());
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));
    iree_hal_streaming_test_install_device_registry(&registry_,
                                                    iree_allocator_system());

    memset(&device_, 0, sizeof(device_));
    registry_.device_count = 1;
    device_.ordinal = 0;
    device_.hrx_device = hrx_device;
    device_.hal_device = hrx_device_hal(hrx_device);
    device_.runtime_generation = registry_.runtime_generation;
    device_.total_memory = 1024;
    iree_atomic_store(&device_.free_memory, device_.total_memory,
                      iree_memory_order_relaxed);
    device_.primary_context_flags.scheduling_mode =
        IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    iree_slim_mutex_initialize(&device_.primary_context_mutex);
    IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());
  }

  void TearDown() override {
    iree_hal_streaming_tls_test_inject_failures(
        IREE_HAL_STREAMING_TLS_TEST_FAILURE_NONE);
    IREE_EXPECT_OK(iree_hal_streaming_context_clear_current_thread());

    while (device_.retired_primary_context_ref_count > 0) {
      IREE_EXPECT_OK(
          iree_hal_streaming_device_release_primary_context(&device_));
    }
    while (device_.primary_context_ref_count > 0) {
      IREE_EXPECT_OK(
          iree_hal_streaming_device_release_primary_context(&device_));
    }
    if (device_.primary_context) {
      IREE_EXPECT_OK(iree_hal_streaming_device_reset_primary_context(&device_));
    }
    iree_hal_streaming_context_release(detached_context_);
    detached_context_ = nullptr;

    // Keep teardown robust when an assertion observes a poisoned list: remove
    // each leaked list edge so the real context destructor can run while the
    // test registry and device entry are still alive.
    while (registry_.context_list.head) {
      iree_hal_streaming_unregister_context(registry_.context_list.head);
    }
    hrx_mem_pool_release(device_.current_mem_pool);
    device_.current_mem_pool = nullptr;
    hrx_mem_pool_release(device_.default_mem_pool);
    device_.default_mem_pool = nullptr;
    iree_slim_mutex_deinitialize(&device_.primary_context_mutex);
    registry_.device_count = 0;
    iree_hal_streaming_test_remove_device_registry(&registry_);
    IREE_EXPECT_OK(iree_hal_streaming_context_tls_test_reset());
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  iree_hal_streaming_device_registry_t registry_ = {};
  iree_hal_streaming_device_t& device_ = registry_.devices[0];
  iree_hal_streaming_context_t* detached_context_ = nullptr;
};

TEST_F(StreamingPrimaryContextFailureTest,
       DefaultPoolFailureUnregistersAndRetryDoesNotPoisonList) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    iree_hal_streaming_device_test_fail_next_default_mem_pool();
    iree_hal_streaming_context_t* context =
        reinterpret_cast<iree_hal_streaming_context_t*>(uintptr_t{1});
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_streaming_device_get_or_create_primary_context(&device_,
                                                                &context));
    EXPECT_EQ(nullptr, context);
    EXPECT_EQ(nullptr, device_.primary_context);
    EXPECT_EQ(nullptr, registry_.context_list.head);
    EXPECT_EQ(nullptr, registry_.context_list.tail);
  }

  iree_hal_streaming_context_t* context = nullptr;
  iree_hal_streaming_context_flags_t flags = {};
  flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_, flags, iree_allocator_system(), &context));
  ASSERT_TRUE(iree_hal_streaming_context_is_registered(context));
  iree_hal_streaming_context_discard_unpublished(context);
  EXPECT_EQ(nullptr, registry_.context_list.head);
  EXPECT_EQ(nullptr, registry_.context_list.tail);
}

TEST_F(StreamingPrimaryContextFailureTest,
       CleanupRejectsPrimaryCreateAndRetainPublicationInFlight) {
  std::promise<void> admitted_promise;
  std::future<void> admitted = admitted_promise.get_future();
  std::promise<void> proceed_promise;
  std::shared_future<void> proceed = proceed_promise.get_future().share();
  std::atomic<int> retain_status{IREE_STATUS_UNKNOWN};
  std::thread thread([&]() {
    iree_status_t status = iree_hal_streaming_context_publication_begin();
    admitted_promise.set_value();
    if (iree_status_is_ok(status)) {
      proceed.wait();
      iree_hal_streaming_context_t* context = nullptr;
      status =
          iree_hal_streaming_device_retain_primary_context(&device_, &context);
      iree_hal_streaming_context_publication_end();
    }
    retain_status.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
  });
  admitted.wait();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_cleanup_global());
  EXPECT_EQ(nullptr, device_.primary_context);
  EXPECT_EQ(nullptr, registry_.context_list.head);

  proceed_promise.set_value();
  thread.join();
  ASSERT_EQ(IREE_STATUS_OK, retain_status.load(std::memory_order_acquire));
  ASSERT_NE(nullptr, device_.primary_context);
  EXPECT_EQ(1, device_.primary_context_ref_count);
  ASSERT_NE(nullptr, registry_.context_list.head);

  IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  EXPECT_EQ(nullptr, device_.primary_context);
  EXPECT_EQ(0, device_.primary_context_ref_count);
  EXPECT_EQ(nullptr, registry_.context_list.head);
  EXPECT_EQ(nullptr, registry_.context_list.tail);
}

TEST_F(StreamingPrimaryContextFailureTest,
       CleanupRejectsUnexpectedPublishedContextOwnerPrecommit) {
  iree_hal_streaming_context_flags_t flags = {};
  flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  iree_hal_streaming_context_t* context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_, flags, iree_allocator_system(), &context));
  ASSERT_TRUE(iree_hal_streaming_context_is_registered(context));
  ASSERT_EQ(2, iree_atomic_ref_count_load(&context->ref_count));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_cleanup_global());
  EXPECT_EQ(&registry_, iree_hal_streaming_device_registry());
  EXPECT_TRUE(iree_hal_streaming_context_is_registered(context));
  EXPECT_EQ(2, iree_atomic_ref_count_load(&context->ref_count));

  iree_hal_streaming_context_discard_unpublished(context);
  EXPECT_EQ(nullptr, registry_.context_list.head);
  EXPECT_EQ(nullptr, registry_.context_list.tail);
}

TEST_F(StreamingPrimaryContextFailureTest,
       ResetFailureIsAtomicAndRetiredRetainsPrecedeNewGeneration) {
  iree_hal_streaming_context_t* old_context = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_retain_primary_context(&device_, &old_context));
  iree_hal_streaming_context_t* second_retain = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_retain_primary_context(
      &device_, &second_retain));
  ASSERT_EQ(old_context, second_retain);
  ASSERT_EQ(2, device_.primary_context_ref_count);

  iree_hal_streaming_buffer_t* allocation = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      old_context, 64, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &allocation));
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(1u, old_context->buffer_table.count);
  ASSERT_EQ(960u,
            iree_atomic_load(&device_.free_memory, iree_memory_order_relaxed));
  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(old_context));

  hrx_mem_pool_t old_current_pool = device_.current_mem_pool;
  hrx_mem_pool_t old_default_pool = device_.default_mem_pool;
  iree_hal_streaming_tls_test_inject_failures(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_streaming_device_reset_primary_context(&device_));
  EXPECT_EQ(old_context, device_.primary_context);
  EXPECT_EQ(2, device_.primary_context_ref_count);
  EXPECT_EQ(0, device_.retired_primary_context_ref_count);
  EXPECT_EQ(old_context, iree_hal_streaming_context_current());
  EXPECT_EQ(old_context, registry_.context_list.head);
  EXPECT_EQ(old_context, registry_.context_list.tail);
  EXPECT_EQ(1u, old_context->buffer_table.count);
  EXPECT_EQ(old_current_pool, device_.current_mem_pool);
  EXPECT_EQ(old_default_pool, device_.default_mem_pool);
  EXPECT_EQ(960u,
            iree_atomic_load(&device_.free_memory, iree_memory_order_relaxed));
  EXPECT_EQ(1, iree_atomic_load(&old_context->accepting_work,
                                iree_memory_order_acquire));

  // Keep the retired allocation itself live for exact post-reset inspection;
  // this pin is not part of either logical primary-retain ledger.
  iree_hal_streaming_context_retain(old_context);
  detached_context_ = old_context;
  IREE_ASSERT_OK(iree_hal_streaming_device_reset_primary_context(&device_));
  EXPECT_EQ(nullptr, device_.primary_context);
  EXPECT_EQ(0, device_.primary_context_ref_count);
  EXPECT_EQ(2, device_.retired_primary_context_ref_count);
  EXPECT_EQ(nullptr, iree_hal_streaming_context_current());
  EXPECT_EQ(nullptr, registry_.context_list.head);
  EXPECT_EQ(nullptr, registry_.context_list.tail);
  EXPECT_EQ(nullptr, device_.current_mem_pool);
  EXPECT_EQ(nullptr, device_.default_mem_pool);
  EXPECT_EQ(0u, old_context->buffer_table.count);
  EXPECT_EQ(1024u,
            iree_atomic_load(&device_.free_memory, iree_memory_order_relaxed));
  EXPECT_FALSE(iree_hal_streaming_context_is_registered(old_context));
  EXPECT_TRUE(iree_hal_streaming_context_is_teardown_certified(old_context));

  iree_hal_streaming_context_t* new_context = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_retain_primary_context(&device_, &new_context));
  ASSERT_NE(old_context, new_context);
  ASSERT_EQ(1, device_.primary_context_ref_count);
  ASSERT_EQ(2, device_.retired_primary_context_ref_count);
  ASSERT_EQ(new_context, registry_.context_list.head);
  ASSERT_EQ(new_context, registry_.context_list.tail);

  // Generationless releases always consume reset-retired logical retains
  // before they are allowed to affect a lazily recreated primary.
  IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  EXPECT_EQ(new_context, device_.primary_context);
  EXPECT_EQ(1, device_.primary_context_ref_count);
  EXPECT_EQ(0, device_.retired_primary_context_ref_count);
  EXPECT_EQ(new_context, registry_.context_list.head);

  IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  EXPECT_EQ(nullptr, device_.primary_context);
  EXPECT_EQ(nullptr, registry_.context_list.head);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_device_release_primary_context(&device_));

  iree_hal_streaming_context_release(detached_context_);
  detached_context_ = nullptr;
}

TEST_F(StreamingPrimaryContextFailureTest,
       RepeatedResetAccumulatesRetiredLedgerAndCapsTotalRetains) {
  iree_hal_streaming_context_t* first_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_retain_primary_context(
      &device_, &first_context));
  IREE_ASSERT_OK(iree_hal_streaming_device_reset_primary_context(&device_));
  EXPECT_EQ(1, device_.retired_primary_context_ref_count);

  iree_hal_streaming_context_t* second_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_retain_primary_context(
      &device_, &second_context));
  iree_hal_streaming_context_t* second_context_again = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_retain_primary_context(
      &device_, &second_context_again));
  ASSERT_EQ(second_context, second_context_again);
  IREE_ASSERT_OK(iree_hal_streaming_device_reset_primary_context(&device_));
  EXPECT_EQ(3, device_.retired_primary_context_ref_count);

  iree_hal_streaming_context_t* final_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_retain_primary_context(
      &device_, &final_context));
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  }
  EXPECT_EQ(final_context, device_.primary_context);
  EXPECT_EQ(1, device_.primary_context_ref_count);
  EXPECT_EQ(0, device_.retired_primary_context_ref_count);
  IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  EXPECT_EQ(nullptr, device_.primary_context);

  device_.retired_primary_context_ref_count = INT32_MAX;
  iree_hal_streaming_context_t* unchanged =
      reinterpret_cast<iree_hal_streaming_context_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_device_retain_primary_context(&device_, &unchanged));
  EXPECT_EQ(reinterpret_cast<iree_hal_streaming_context_t*>(uintptr_t{1}),
            unchanged);
  EXPECT_EQ(nullptr, device_.primary_context);
  EXPECT_EQ(0, device_.primary_context_ref_count);
  device_.retired_primary_context_ref_count = 0;
}

TEST_F(StreamingPrimaryContextFailureTest,
       ResetRetiresDuplicateTlsStackUntilPopped) {
  iree_hal_streaming_context_t* old_context = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_retain_primary_context(&device_, &old_context));
  IREE_ASSERT_OK(iree_hal_streaming_context_set_current(old_context));
  IREE_ASSERT_OK(iree_hal_streaming_context_push(old_context));
  ASSERT_EQ(2u,
            iree_hal_streaming_context_current_thread_tls_reference_count());

  IREE_ASSERT_OK(iree_hal_streaming_device_reset_primary_context(&device_));
  EXPECT_EQ(nullptr, iree_hal_streaming_context_current());
  EXPECT_EQ(1u,
            iree_hal_streaming_context_current_thread_tls_reference_count());
  EXPECT_EQ(1, iree_atomic_ref_count_load(&old_context->ref_count));
  EXPECT_EQ(nullptr, registry_.context_list.head);
  EXPECT_EQ(1, device_.retired_primary_context_ref_count);

  iree_hal_streaming_context_t* popped_context =
      reinterpret_cast<iree_hal_streaming_context_t*>(uintptr_t{1});
  IREE_ASSERT_OK(iree_hal_streaming_context_pop(&popped_context));
  EXPECT_EQ(nullptr, popped_context);
  EXPECT_EQ(old_context, iree_hal_streaming_context_current());
  EXPECT_EQ(1u,
            iree_hal_streaming_context_current_thread_tls_reference_count());
  EXPECT_FALSE(iree_hal_streaming_context_is_current(old_context));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_streaming_context_operation_begin(old_context));

  IREE_ASSERT_OK(iree_hal_streaming_context_pop(&popped_context));
  EXPECT_EQ(old_context, popped_context);
  EXPECT_EQ(nullptr, iree_hal_streaming_context_current());
  EXPECT_EQ(0u,
            iree_hal_streaming_context_current_thread_tls_reference_count());
  IREE_ASSERT_OK(iree_hal_streaming_device_release_primary_context(&device_));
  EXPECT_EQ(0, device_.retired_primary_context_ref_count);
}

TEST_F(StreamingPrimaryContextFailureTest,
       ZeroRetainResetConsumesAllocationsWithoutCreatingRetiredLedger) {
  iree_hal_streaming_context_t* old_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_get_or_create_primary_context(
      &device_, &old_context));
  ASSERT_EQ(0, device_.primary_context_ref_count);
  iree_hal_streaming_context_retain(old_context);
  detached_context_ = old_context;

  iree_hal_streaming_buffer_t* allocation = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      old_context, 64, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &allocation));
  ASSERT_EQ(1u, old_context->buffer_table.count);
  IREE_ASSERT_OK(iree_hal_streaming_device_reset_primary_context(&device_));

  EXPECT_EQ(nullptr, device_.primary_context);
  EXPECT_EQ(0, device_.primary_context_ref_count);
  EXPECT_EQ(0, device_.retired_primary_context_ref_count);
  EXPECT_EQ(nullptr, registry_.context_list.head);
  EXPECT_EQ(0u, old_context->buffer_table.count);
  EXPECT_EQ(1024u,
            iree_atomic_load(&device_.free_memory, iree_memory_order_relaxed));
  EXPECT_TRUE(iree_hal_streaming_context_is_teardown_certified(old_context));

  iree_hal_streaming_context_release(detached_context_);
  detached_context_ = nullptr;
}

TEST_F(StreamingPrimaryContextFailureTest,
       CleanupRejectsUnreleasedResetRetainLedgerPrecommit) {
  device_.retired_primary_context_ref_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_cleanup_global());
  EXPECT_EQ(&registry_, iree_hal_streaming_device_registry());
  EXPECT_EQ(1, device_.retired_primary_context_ref_count);
  device_.retired_primary_context_ref_count = 0;
}

TEST(StreamingTlsTest, CapacityExhausted) {
  iree_hal_streaming_tls_key_t keys[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {
      IREE_HAL_STREAMING_TLS_KEY_INVALID};
  for (iree_host_size_t i = 0; i < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&keys[i], nullptr));
  }

  iree_hal_streaming_tls_key_t extra_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  iree_status_t status = iree_hal_streaming_tls_key_create(&extra_key, nullptr);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(IREE_HAL_STREAMING_TLS_KEY_INVALID, extra_key);

  for (iree_host_size_t i = 0; i < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_tls_key_delete(keys[i]));
  }
}

class StreamingDirectCleanupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_streaming_context_tls_test_reset());
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*registry_),
                                         reinterpret_cast<void**>(&registry_)));
    iree_hal_streaming_test_install_device_registry(registry_,
                                                    iree_allocator_system());

    memset(&device_, 0, sizeof(device_));
    device_.hrx_device = hrx_device;
    device_.hal_device = hrx_device_hal(hrx_device);
    device_.runtime_generation = registry_->runtime_generation;
    iree_slim_mutex_initialize(&device_.primary_context_mutex);
    IREE_ASSERT_OK(iree_hal_streaming_context_tls_initialize());

    iree_hal_streaming_context_flags_t flags = {};
    flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_, flags, allocator_.AsAllocator(), &context_));
    context_->is_primary = true;
    device_.primary_context = context_;
  }

  void TearDown() override {
    iree_hal_streaming_tls_test_inject_failures(
        IREE_HAL_STREAMING_TLS_TEST_FAILURE_NONE);
    IREE_EXPECT_OK(iree_hal_streaming_context_clear_current_thread());

    // The creator edge stands in for a device's primary publication. Drop it
    // while the test device entry is still alive.
    device_.primary_context = nullptr;
    iree_hal_streaming_context_release(context_);
    context_ = nullptr;

    if (registry_ && iree_hal_streaming_device_registry() == registry_) {
      while (registry_->context_list.head) {
        iree_hal_streaming_unregister_context(registry_->context_list.head);
      }
      iree_hal_streaming_test_remove_device_registry(registry_);
      iree_allocator_free(iree_allocator_system(), registry_);
    }
    registry_ = nullptr;
    iree_slim_mutex_deinitialize(&device_.primary_context_mutex);
    IREE_EXPECT_OK(iree_hal_streaming_context_tls_test_reset());
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  CountingAllocator allocator_;
  iree_hal_streaming_device_registry_t* registry_ = nullptr;
  iree_hal_streaming_device_t device_ = {};
  iree_hal_streaming_context_t* context_ = nullptr;
};

TEST_F(StreamingDirectCleanupTest,
       CommonCleanupQuiescesAndDetachesCudaStyleContext) {
  ASSERT_NE(nullptr, context_->default_stream);
  ASSERT_EQ(1, iree_atomic_load(&context_->accepting_work,
                                iree_memory_order_acquire));
  ASSERT_EQ(0, iree_atomic_load(&context_->teardown_quiesced,
                                iree_memory_order_acquire));

  IREE_ASSERT_OK(iree_hal_streaming_cleanup_global());
  // cleanup_global consumed and freed this heap registry.
  registry_ = nullptr;

  EXPECT_EQ(0, iree_atomic_load(&context_->accepting_work,
                                iree_memory_order_acquire));
  EXPECT_EQ(1, iree_atomic_load(&context_->teardown_quiesced,
                                iree_memory_order_acquire));
  EXPECT_TRUE(iree_hal_streaming_context_is_teardown_certified(context_));
  EXPECT_EQ(nullptr, context_->default_stream);
  EXPECT_EQ(1, iree_atomic_ref_count_load(&context_->ref_count));
}

}  // namespace
