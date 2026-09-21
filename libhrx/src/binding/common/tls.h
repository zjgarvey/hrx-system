// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Binding-local thread-specific storage keys with optional thread-exit
// destructors.

#ifndef IREE_EXPERIMENTAL_STREAMING_TLS_H_
#define IREE_EXPERIMENTAL_STREAMING_TLS_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum process-global TLS keys that can be allocated through this binding
// API. The fixed capacity keeps key allocation bounded and lets get/set remain
// direct slot operations.
#define IREE_HAL_STREAMING_TLS_KEY_CAPACITY 128

// Invalid key value used for uninitialized key storage.
#define IREE_HAL_STREAMING_TLS_KEY_INVALID ((iree_hal_streaming_tls_key_t) - 1)

// Process-global TLS key. Keys are small indexes into an internal fixed table.
typedef iree_host_size_t iree_hal_streaming_tls_key_t;

// Invoked with a non-NULL thread-local value when a thread exits.
typedef void(IREE_API_PTR* iree_hal_streaming_tls_destructor_t)(void* value);

// Creates a process-global TLS key with an optional thread-exit |destructor|.
//
// Keys are usually created once and kept for process lifetime. Deleting a key
// is only safe after all threads that could read, write, or destruct values for
// the key are gone or externally synchronized. Deletion does not invoke
// destructors.
//
// Windows values are fiber-local and destructors run from the native FLS
// callback on fiber deletion or thread exit, outside DllMain teardown. A
// destructor may reinstall a value for up to four callback passes. The module
// containing the callback must remain loaded until the key is deleted; binding
// global cleanup deletes its permanent key before the module can be unloaded.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor);

// Deletes a key previously created with iree_hal_streaming_tls_key_create.
// Failure leaves the key and every associated value valid for retry.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_tls_key_delete(iree_hal_streaming_tls_key_t key);

// Returns the value associated with |key| on the current thread.
//
// Returns NULL for valid unset keys and invalid/deleted keys.
IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key);

// Sets the value associated with |key| on the current thread.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value);

#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
// TLS operations whose next matching invocation may be failed by tests. The
// injection is process-global and callers must externally serialize changes
// with the operations they exercise.
typedef enum iree_hal_streaming_tls_test_failure_bits_e {
  IREE_HAL_STREAMING_TLS_TEST_FAILURE_NONE = 0,
  IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_CREATE = 1u << 0,
  IREE_HAL_STREAMING_TLS_TEST_FAILURE_SET = 1u << 1,
  IREE_HAL_STREAMING_TLS_TEST_FAILURE_CLEAR = 1u << 2,
  IREE_HAL_STREAMING_TLS_TEST_FAILURE_KEY_DELETE = 1u << 3,
} iree_hal_streaming_tls_test_failure_bits_t;

// Replaces the set of one-shot TLS failures armed for tests. Each matching
// operation consumes its bit before returning an injected error.
#if defined(IREE_PLATFORM_WINDOWS)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif  // IREE_PLATFORM_WINDOWS
void iree_hal_streaming_tls_test_inject_failures(
    iree_hal_streaming_tls_test_failure_bits_t failures);
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_TLS_H_
