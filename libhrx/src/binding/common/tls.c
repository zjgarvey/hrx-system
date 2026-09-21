// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/tls.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/internal/atomics.h"

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_PLATFORM_WINDOWS)
#include <pthread.h>
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !IREE_PLATFORM_WINDOWS

typedef enum iree_hal_streaming_tls_slot_state_e {
  IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY = 0,
  IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING = 1,
  IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED = 2,
} iree_hal_streaming_tls_slot_state_t;

typedef struct iree_hal_streaming_tls_slot_t {
  // Allocation state for this process-global slot.
  iree_atomic_int32_t state;
  // Destructor registered for values stored in this slot.
  iree_hal_streaming_tls_destructor_t destructor;
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_PLATFORM_WINDOWS)
  // Native pthread key backing this slot.
  pthread_key_t pthread_key;
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !IREE_PLATFORM_WINDOWS
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && defined(IREE_PLATFORM_WINDOWS)
  // Native FLS index backing this slot.
  DWORD fls_index;
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && IREE_PLATFORM_WINDOWS
} iree_hal_streaming_tls_slot_t;

static iree_hal_streaming_tls_slot_t
    iree_hal_streaming_tls_slots[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {{0}};

#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
static iree_atomic_int32_t iree_hal_streaming_tls_test_failures =
    IREE_ATOMIC_VAR_INIT(IREE_HAL_STREAMING_TLS_TEST_FAILURE_NONE);

static bool iree_hal_streaming_tls_test_consume_failure(
    iree_hal_streaming_tls_test_failure_bits_t failure) {
  int32_t failures = iree_atomic_load(&iree_hal_streaming_tls_test_failures,
                                      iree_memory_order_acquire);
  while ((failures & failure) != 0) {
    const int32_t remaining_failures = failures & ~failure;
    if (iree_atomic_compare_exchange_weak(&iree_hal_streaming_tls_test_failures,
                                          &failures, remaining_failures,
                                          iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

#if defined(IREE_PLATFORM_WINDOWS)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif  // IREE_PLATFORM_WINDOWS
void iree_hal_streaming_tls_test_inject_failures(
    iree_hal_streaming_tls_test_failure_bits_t failures) {
  iree_atomic_store(&iree_hal_streaming_tls_test_failures, failures,
                    iree_memory_order_release);
}

static iree_status_t iree_hal_streaming_tls_test_injected_failure(
    iree_hal_streaming_tls_test_failure_bits_t failure) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "injected binding TLS operation failure: %u",
                          (unsigned)failure);
}

#define IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(failure)      \
  do {                                                              \
    if (iree_hal_streaming_tls_test_consume_failure(failure)) {     \
      return iree_hal_streaming_tls_test_injected_failure(failure); \
    }                                                               \
  } while (0)
#else
#define IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(failure) \
  do {                                                         \
  } while (0)
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

static bool iree_hal_streaming_tls_slot_is_allocated(
    iree_hal_streaming_tls_key_t key) {
  return key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY &&
         iree_atomic_load(&iree_hal_streaming_tls_slots[key].state,
                          iree_memory_order_acquire) ==
             IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED;
}

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

static void*
    iree_hal_streaming_tls_values[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {0};

IREE_API_EXPORT iree_status_t iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_CREATE);
  for (iree_hal_streaming_tls_key_t key = 0;
       key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
    int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY;
    if (!iree_atomic_compare_exchange_strong(
            &iree_hal_streaming_tls_slots[key].state, &expected_state,
            IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    iree_hal_streaming_tls_slots[key].destructor = destructor;
    iree_hal_streaming_tls_values[key] = NULL;
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    *out_key = key;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "binding TLS key capacity exhausted");
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_key_delete(iree_hal_streaming_tls_key_t key) {
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_DELETE);
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return iree_ok_status();
  iree_hal_streaming_tls_values[key] = NULL;
  iree_hal_streaming_tls_slots[key].destructor = NULL;
  iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                    IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                    iree_memory_order_release);
  return iree_ok_status();
}

IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key) {
  return iree_hal_streaming_tls_slot_is_allocated(key)
             ? iree_hal_streaming_tls_values[key]
             : NULL;
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value) {
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      value ? IREE_HAL_STREAMING_TLS_TEST_FAILURE_SET
            : IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  if (IREE_UNLIKELY(!iree_hal_streaming_tls_slot_is_allocated(key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid binding TLS key");
  }
  iree_hal_streaming_tls_values[key] = value;
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_WINDOWS)

// FLS stores a cell instead of the caller's value so one callback can recover
// the owning key and its destructor. The cell is replaced transactionally by
// each set, leaving the prior value intact if allocation or FlsSetValue fails.
typedef struct iree_hal_streaming_tls_fls_cell_t {
  iree_hal_streaming_tls_key_t key;
  DWORD fls_index;
  iree_hal_streaming_tls_destructor_t destructor;
  void* value;
} iree_hal_streaming_tls_fls_cell_t;

// FlsFree and DeleteFiber may invoke a callback for a fiber other than the
// calling fiber. FlsGetValue/FlsSetValue always address the calling fiber, so
// callbacks must model destructor reinsertion without touching native FLS.
// The compiler-TLS frame is only active while user destructor code runs and
// preserves nested callback behavior on the calling OS thread.
typedef struct iree_hal_streaming_tls_fls_callback_frame_t {
  struct iree_hal_streaming_tls_fls_callback_frame_t* previous;
  iree_hal_streaming_tls_key_t key;
  DWORD fls_index;
  iree_hal_streaming_tls_fls_cell_t* pending_cell;
} iree_hal_streaming_tls_fls_callback_frame_t;

static IREE_THREAD_LOCAL iree_hal_streaming_tls_fls_callback_frame_t*
    iree_hal_streaming_tls_fls_callback_frame = NULL;

static iree_hal_streaming_tls_fls_callback_frame_t*
iree_hal_streaming_tls_find_fls_callback_frame(iree_hal_streaming_tls_key_t key,
                                               DWORD fls_index) {
  for (iree_hal_streaming_tls_fls_callback_frame_t* frame =
           iree_hal_streaming_tls_fls_callback_frame;
       frame; frame = frame->previous) {
    if (frame->key == key && frame->fls_index == fls_index) return frame;
  }
  return NULL;
}

static void iree_hal_streaming_tls_fls_cell_free(
    iree_hal_streaming_tls_fls_cell_t* cell) {
  if (cell) HeapFree(GetProcessHeap(), 0, cell);
}

static VOID NTAPI iree_hal_streaming_tls_fls_callback(void* raw_cell) {
  iree_hal_streaming_tls_fls_cell_t* cell =
      (iree_hal_streaming_tls_fls_cell_t*)raw_cell;

  if (!cell) return;
  const iree_hal_streaming_tls_key_t key = cell->key;
  const DWORD fls_index = cell->fls_index;
  if (key >= IREE_HAL_STREAMING_TLS_KEY_CAPACITY ||
      iree_atomic_load(&iree_hal_streaming_tls_slots[key].state,
                       iree_memory_order_acquire) !=
          IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED ||
      iree_hal_streaming_tls_slots[key].fls_index != fls_index) {
    // FlsFree marks the slot non-allocated before invoking callbacks for live
    // values. Free the binding wrapper without running user destructors.
    iree_hal_streaming_tls_fls_cell_free(cell);
    return;
  }

  iree_hal_streaming_tls_fls_callback_frame_t frame = {
      .previous = iree_hal_streaming_tls_fls_callback_frame,
      .key = key,
      .fls_index = fls_index,
      .pending_cell = NULL,
  };
  iree_hal_streaming_tls_fls_callback_frame = &frame;

  // Match the established four-pass destructor contract. Clear each cell
  // before invoking user code. A same-key set is intercepted by |frame| so a
  // destructor can reinstall for the callback's target fiber without reading
  // or modifying the unrelated calling/current fiber's native FLS value.
  for (int iteration = 0; cell && iteration < 4; ++iteration) {
    if (!iree_hal_streaming_tls_slot_is_allocated(key) ||
        iree_hal_streaming_tls_slots[key].fls_index != fls_index) {
      iree_hal_streaming_tls_fls_cell_free(cell);
      cell = NULL;
      break;
    }

    iree_hal_streaming_tls_destructor_t destructor = cell->destructor;
    void* value = cell->value;
    iree_hal_streaming_tls_fls_cell_free(cell);
    if (destructor) destructor(value);
    cell = frame.pending_cell;
    frame.pending_cell = NULL;
  }

  // A fifth value is outside the destructor-iteration contract. Free only the
  // callback-local wrapper so a continually reinstalling destructor cannot
  // create an unbounded callback loop.
  iree_hal_streaming_tls_fls_cell_free(cell);
  iree_hal_streaming_tls_fls_callback_frame = frame.previous;
}

IREE_API_EXPORT iree_status_t iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_CREATE);
  for (iree_hal_streaming_tls_key_t key = 0;
       key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
    int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY;
    if (!iree_atomic_compare_exchange_strong(
            &iree_hal_streaming_tls_slots[key].state, &expected_state,
            IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    DWORD fls_index = FlsAlloc(iree_hal_streaming_tls_fls_callback);
    if (fls_index == FLS_OUT_OF_INDEXES) {
      const DWORD error = GetLastError();
      iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                        IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                        iree_memory_order_release);
      return error ? iree_make_status(iree_status_code_from_win32_error(error),
                                      "FlsAlloc failed: %lu",
                                      (unsigned long)error)
                   : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                      "FLS index capacity exhausted");
    }
    iree_hal_streaming_tls_slots[key].fls_index = fls_index;
    iree_hal_streaming_tls_slots[key].destructor = destructor;
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    *out_key = key;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "binding TLS key capacity exhausted");
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_key_delete(iree_hal_streaming_tls_key_t key) {
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_DELETE);
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return iree_ok_status();
  int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED;
  if (!iree_atomic_compare_exchange_strong(
          &iree_hal_streaming_tls_slots[key].state, &expected_state,
          IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "binding FLS key deletion already in progress");
  }
  const DWORD fls_index = iree_hal_streaming_tls_slots[key].fls_index;
  // FlsFree synchronously invokes the callback for every live cell. The slot
  // is INITIALIZING during those callbacks, so they free only the wrapper and
  // never invoke user destructors. Calling FlsFree directly is also critical
  // for rollback: if it fails, no current cell was detached or freed.
  if (!FlsFree(fls_index)) {
    const DWORD error = GetLastError();
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "FlsFree failed: %lu", (unsigned long)error);
  }
  iree_hal_streaming_tls_slots[key].fls_index = FLS_OUT_OF_INDEXES;
  iree_hal_streaming_tls_slots[key].destructor = NULL;
  iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                    IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                    iree_memory_order_release);
  return iree_ok_status();
}

IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key) {
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return NULL;
  iree_hal_streaming_tls_slot_t* slot = &iree_hal_streaming_tls_slots[key];
  iree_hal_streaming_tls_fls_callback_frame_t* frame =
      iree_hal_streaming_tls_find_fls_callback_frame(key, slot->fls_index);
  if (frame) {
    return frame->pending_cell ? frame->pending_cell->value : NULL;
  }
  iree_hal_streaming_tls_fls_cell_t* cell =
      (iree_hal_streaming_tls_fls_cell_t*)FlsGetValue(slot->fls_index);
  return cell ? cell->value : NULL;
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value) {
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      value ? IREE_HAL_STREAMING_TLS_TEST_FAILURE_SET
            : IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  if (IREE_UNLIKELY(!iree_hal_streaming_tls_slot_is_allocated(key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid binding TLS key");
  }
  iree_hal_streaming_tls_slot_t* slot = &iree_hal_streaming_tls_slots[key];
  iree_hal_streaming_tls_fls_cell_t* new_cell = NULL;
  if (value) {
    new_cell = (iree_hal_streaming_tls_fls_cell_t*)HeapAlloc(
        GetProcessHeap(), 0, sizeof(*new_cell));
    if (!new_cell) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "binding FLS value allocation failed");
    }
    new_cell->key = key;
    new_cell->fls_index = slot->fls_index;
    new_cell->destructor = slot->destructor;
    new_cell->value = value;
  }
  iree_hal_streaming_tls_fls_callback_frame_t* frame =
      iree_hal_streaming_tls_find_fls_callback_frame(key, slot->fls_index);
  if (frame) {
    iree_hal_streaming_tls_fls_cell_t* old_cell = frame->pending_cell;
    frame->pending_cell = new_cell;
    iree_hal_streaming_tls_fls_cell_free(old_cell);
    return iree_ok_status();
  }
  iree_hal_streaming_tls_fls_cell_t* old_cell =
      (iree_hal_streaming_tls_fls_cell_t*)FlsGetValue(slot->fls_index);
  if (!FlsSetValue(slot->fls_index, new_cell)) {
    const DWORD error = GetLastError();
    iree_hal_streaming_tls_fls_cell_free(new_cell);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "FlsSetValue failed: %lu", (unsigned long)error);
  }
  iree_hal_streaming_tls_fls_cell_free(old_cell);
  return iree_ok_status();
}

#else

IREE_API_EXPORT iree_status_t iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_CREATE);
  for (iree_hal_streaming_tls_key_t key = 0;
       key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
    int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY;
    if (!iree_atomic_compare_exchange_strong(
            &iree_hal_streaming_tls_slots[key].state, &expected_state,
            IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    int result =
        pthread_key_create(&iree_hal_streaming_tls_slots[key].pthread_key,
                           (void (*)(void*))destructor);
    if (result != 0) {
      iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                        IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                        iree_memory_order_release);
      return iree_make_status(iree_status_code_from_errno(result),
                              "pthread_key_create failed: %d", result);
    }
    iree_hal_streaming_tls_slots[key].destructor = destructor;
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    *out_key = key;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "binding TLS key capacity exhausted");
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_key_delete(iree_hal_streaming_tls_key_t key) {
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_DELETE);
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return iree_ok_status();
  int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED;
  if (!iree_atomic_compare_exchange_strong(
          &iree_hal_streaming_tls_slots[key].state, &expected_state,
          IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "binding pthread TLS key deletion already in "
                            "progress");
  }
  int result =
      pthread_key_delete(iree_hal_streaming_tls_slots[key].pthread_key);
  if (result != 0) {
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    return iree_make_status(iree_status_code_from_errno(result),
                            "pthread_key_delete failed: %d", result);
  }
  iree_hal_streaming_tls_slots[key].destructor = NULL;
  iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                    IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                    iree_memory_order_release);
  return iree_ok_status();
}

IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key) {
  return iree_hal_streaming_tls_slot_is_allocated(key)
             ? pthread_getspecific(
                   iree_hal_streaming_tls_slots[key].pthread_key)
             : NULL;
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value) {
  IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE(
      value ? IREE_HAL_STREAMING_TLS_TEST_FAILURE_SET
            : IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR);
  if (IREE_UNLIKELY(!iree_hal_streaming_tls_slot_is_allocated(key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid binding TLS key");
  }
  int result =
      pthread_setspecific(iree_hal_streaming_tls_slots[key].pthread_key, value);
  if (IREE_UNLIKELY(result != 0)) {
    return iree_make_status(iree_status_code_from_errno(result),
                            "pthread_setspecific failed: %d", result);
  }
  return iree_ok_status();
}

#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#undef IREE_HAL_STREAMING_TLS_RETURN_IF_TEST_FAILURE
