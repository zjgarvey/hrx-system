// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_MEMORY_H_
#define IREE_EXPERIMENTAL_STREAMING_MEMORY_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;

// Allocates queue-visible host staging memory.
iree_status_t iree_hal_streaming_memory_allocate_host_staging(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

// Prepares a detached exact-context wrapper for a target-placement VMM alias
// and reserves pointer-table capacity for allocation-free publication.
iree_status_t iree_hal_streaming_memory_prepare_virtual_alias(
    iree_hal_streaming_context_t* context, hrx_buffer_t alias_buffer,
    iree_hal_streaming_buffer_t** out_buffer);

// Reserves one allocation-free publication slot for rollback of a currently
// published wrapper.
iree_status_t iree_hal_streaming_memory_reserve_wrapped_buffer_publication(
    iree_hal_streaming_buffer_t* buffer);

// Cancels a previously reserved publication slot without changing the
// wrapper's current publication or ownership.
void iree_hal_streaming_memory_cancel_wrapped_buffer_publication(
    iree_hal_streaming_buffer_t* buffer);

// Returns true only when |buffer| is marked published and appears exactly once
// in its owning context's pointer table. Used to validate a prepared no-fail
// teardown transaction before any entry is removed.
bool iree_hal_streaming_memory_is_wrapped_buffer_published_exactly_once(
    iree_hal_streaming_buffer_t* buffer);

// Removes an exactly validated wrapper from pointer lookup without failure or
// release. Requires exclusive lifecycle admission and a prior successful
// exactly-once validation while no table mutation can intervene.
void iree_hal_streaming_memory_commit_unpublish_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Publishes a prepared wrapper using its reserved insertion.
iree_status_t iree_hal_streaming_memory_publish_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Removes a wrapper from pointer lookup without releasing it.
iree_status_t iree_hal_streaming_memory_unpublish_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Removes and releases an externally owned buffer wrapper, cancelling any
// unused insertion reservation.
void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Releases a graph's independent pin on an exact wrapper and, when that same
// wrapper is still published, consumes the pointer-table ownership edge too.
// |pointer_buffer_id| must be the never-reused identity captured with the pin.
// The caller has either never submitted work using the allocation or has
// already quiesced every such submission. This function never waits, allocates,
// or performs pointer-only lookup.
void iree_hal_streaming_memory_release_graph_owned_buffer_quiesced(
    iree_hal_streaming_buffer_t* buffer, uint64_t pointer_buffer_id);

// Detaches and releases every published wrapper whose context reference is
// table-owned. Structurally owned borrowed wrappers remain until their module
// or other owner releases them. The caller must first exclude new operations
// and quiesce every queue that can reference the context.
void iree_hal_streaming_memory_release_context_allocations(
    iree_hal_streaming_context_t* context);

// Detaches and releases only ordinary table-owned allocations. Borrowed
// wrappers remain with their structural owners, and virtual address-space
// wrappers remain published because the VMM registry names them until the
// owning bindings have been removed successfully.
void iree_hal_streaming_memory_release_context_owned_ordinary_allocations(
    iree_hal_streaming_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_MEMORY_H_
