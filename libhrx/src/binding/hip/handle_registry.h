// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_HANDLE_REGISTRY_H_
#define HRX_BINDING_HIP_HANDLE_REGISTRY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Number of independently locked handle-table shards. A power of two keeps
// shard selection to one mask on lookup paths.
#define IREE_HIP_HANDLE_REGISTRY_SHARD_COUNT 16

typedef struct iree_hip_handle_registry_shard_t {
  // Serializes operations in this shard.
  iree_slim_mutex_t mutex;
  // Open-addressed table of live handle values.
  uintptr_t* handles;
  // Slot state array parallel to |handles|.
  uint8_t* states;
  // Power-of-two slot count in |handles| and |states|.
  iree_host_size_t capacity;
  // Number of live handles.
  iree_host_size_t count;
  // Number of removed slots awaiting reuse or rehashing.
  iree_host_size_t tombstone_count;
} iree_hip_handle_registry_shard_t;

// Tracks live opaque API handles without dereferencing caller-provided values.
typedef struct iree_hip_handle_registry_t {
  // Independently locked tables selected by the handle hash.
  iree_hip_handle_registry_shard_t shards[IREE_HIP_HANDLE_REGISTRY_SHARD_COUNT];
} iree_hip_handle_registry_t;

// Retains a handle while its registry entry is locked and known to be live.
typedef void (*iree_hip_handle_registry_retain_fn_t)(uintptr_t handle);

// Returns true when |handle| belongs in a retained registry snapshot.
// Invoked while all registry shard locks are held; implementations must not
// call back into this registry.
typedef bool (*iree_hip_handle_registry_match_fn_t)(uintptr_t handle,
                                                    void* user_data);

void iree_hip_handle_registry_initialize(iree_hip_handle_registry_t* registry);

// Requires the registry to be empty.
void iree_hip_handle_registry_deinitialize(
    iree_hip_handle_registry_t* registry);

iree_status_t iree_hip_handle_registry_insert(
    iree_hip_handle_registry_t* registry, uintptr_t handle);

// Looks up and retains a live handle before releasing the registry lock.
bool iree_hip_handle_registry_lookup_retain(
    iree_hip_handle_registry_t* registry, uintptr_t handle,
    iree_hip_handle_registry_retain_fn_t retain_fn);

// Removes a live handle, transferring its existing ownership to the caller.
bool iree_hip_handle_registry_remove(iree_hip_handle_registry_t* registry,
                                     uintptr_t handle);

// Atomically snapshots and retains every live handle matching |match_fn|.
// Locks all shards in increasing order across count, allocation, and retain so
// the result is exact even without an external registry-stability gate. The
// caller owns the system-allocator array and one reference per handle.
iree_status_t iree_hip_handle_registry_snapshot_retain_if(
    iree_hip_handle_registry_t* registry,
    iree_hip_handle_registry_match_fn_t match_fn, void* user_data,
    iree_hip_handle_registry_retain_fn_t retain_fn, uintptr_t** out_handles,
    iree_host_size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_HANDLE_REGISTRY_H_
