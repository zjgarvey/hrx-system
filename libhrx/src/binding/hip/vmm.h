// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_VMM_H_
#define LIBHRX_SRC_BINDING_HIP_VMM_H_

#include "binding/hip/api.h"

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hip_context_teardown_t iree_hip_context_teardown_t;
typedef struct iree_hip_vmm_teardown_plan_t iree_hip_vmm_teardown_plan_t;
typedef struct iree_hip_mipmapped_array_teardown_t
    iree_hip_mipmapped_array_teardown_t;

typedef bool (*iree_hip_array_match_fn_t)(hipArray_t array, void* user_data);

// Prepares/exact-takes binding-private mipmapped-array wrappers whose level
// arrays all belong to one target teardown. These helpers run under lifecycle
// writer admission; commit invalidates only the wrapper while the caller owns
// exact teardown of each level array.
hipError_t iree_hip_mipmapped_array_prepare_teardown(
    iree_hip_array_match_fn_t match, void* user_data,
    iree_hip_mipmapped_array_teardown_t** out_teardown);
void iree_hip_mipmapped_array_cancel_teardown(
    iree_hip_mipmapped_array_teardown_t* teardown);
void iree_hip_mipmapped_array_commit_teardown(
    iree_hip_mipmapped_array_teardown_t* teardown);

// Internal array/mipmap entry points use one outer lifecycle reader across a
// complete multi-object publication or destruction transaction.
hipError_t iree_hip_internal_ensure_context_admitted(
    iree_hal_streaming_context_t** out_context);
void iree_hip_internal_context_admission_end(void);
hipError_t iree_hip_array_create_admitted(iree_hal_streaming_context_t* context,
                                          hipArray_t* array,
                                          const hipChannelFormatDesc* desc,
                                          hipExtent extent, unsigned int flags);
hipError_t iree_hip_array3d_create_admitted(
    iree_hal_streaming_context_t* context, hipArray_t* array,
    const HIP_ARRAY3D_DESCRIPTOR* descriptor);
hipError_t iree_hip_array_free_admitted(hipArray_t array);

typedef struct iree_hip_vmm_pointer_snapshot_t {
  uintptr_t mapped_base;
  size_t mapped_size;
  size_t offset;
  int physical_device_ordinal;
  hipCtx_t physical_owner_context;
  uint32_t buffer_id;
} iree_hip_vmm_pointer_snapshot_t;

// A zero-or-prepared transaction for consuming one exact primary retain. A
// prepared token owns |context| and is bound to all three state coordinates;
// it must be committed or cancelled while the lifecycle writer remains held.
typedef struct iree_hip_vmm_primary_release_t {
  iree_hal_streaming_device_t* device;
  iree_hal_streaming_context_t* context;
  int32_t expected_ref_count;
  iree_hip_context_teardown_t* teardown;
  iree_hip_vmm_teardown_plan_t* vmm_plan;
} iree_hip_vmm_primary_release_t;

// Returns true only while the current thread is inside a native release
// operation that may synchronously invoke an HSA deallocation callback. This
// combines binding-owned outer scopes with the central raw address-free scope.
// Public HIP initialization and lifecycle entries fail before acquiring any
// lock while this window is active.
bool iree_hip_vmm_native_teardown_callback_window_is_active(void);

// Starts a new process runtime generation after device discovery.
hipError_t iree_hip_vmm_initialize(size_t device_count, uint64_t generation);

// Holds the process VMM lifecycle gate across kernel pointer validation and
// queue acceptance. Launch paths must release with iree_hip_vmm_launch_end.
hipError_t iree_hip_vmm_launch_begin(void);
void iree_hip_vmm_launch_end(void);

// Prepares the complete binding-private ownership drain for an exact context
// set while the lifecycle writer is held. |device_filter| selects one device,
// or is negative to include every supplied context. Preparation owns exact
// retained registry snapshots, all fallible synchronization, queue pins, and
// deferred user-callback storage without mutating public state.
// |allow_retired_synchronization| is reserved for inactive recovery and global
// teardown: it permits an exact registered context whose admission is already
// closed, but which has not yet been quiesced, to perform its fallible wait
// before this transaction's first queue seal.
hipError_t iree_hip_context_teardown_prepare(
    iree_hal_streaming_context_t* const* contexts, size_t context_count,
    int device_filter, bool abort_captures, bool invalidate_stream_handles,
    bool seal_device_owned_queues, bool allow_retired_synchronization,
    iree_hip_context_teardown_t** out_teardown);

// Crosses the fail-closed boundary by permanently closing and certifying every
// exact prepared queue. All public/context/device/failure-delivery ledgers must
// still be published. Once called, cancellation, reopening, and rollback are
// forbidden. A later native cleanup failure may return a public error only
// after the exact generation and dependent ownership ledger have become
// fail-closed and teardown-visible for an inactive-writer retry.
void iree_hip_context_teardown_seal_queues(
    iree_hip_context_teardown_t* teardown);

// Retires and detaches target contexts and exact-takes every prepared graph
// exec, graph, module, and event after queues have been sealed. This is
// allocation-free, no-wait, and no-fail.
void iree_hip_context_teardown_commit(iree_hip_context_teardown_t* teardown);

// Cancels an uncommitted preparation without changing public state.
void iree_hip_context_teardown_cancel(iree_hip_context_teardown_t* teardown);

// Invokes callbacks deferred by commit. Must run only after the lifecycle
// writer has ended and destroys the transaction object.
void iree_hip_context_teardown_finish(iree_hip_context_teardown_t* teardown);

// Drains lifecycle readers and excludes new work while one public module is
// synchronized, invalidated, and removed from the binding-private live-module
// registry. No runtime generation state is changed.
// Acquires an inactive, generation-bound writer for fail-closed cleanup. No
// raw module lookup is permitted before this succeeds.
hipError_t iree_hip_vmm_module_unload_begin(uint64_t* out_generation);
// Converts the caller's single lifecycle-reader admission into queued writer
// admission without reopening a reader/creator window. The reader is consumed
// on every return.
hipError_t iree_hip_vmm_module_unload_handoff_begin(
    bool* out_writer_started_active, uint64_t* out_generation);
void iree_hip_vmm_module_unload_end(void);

// Generic public-object destruction gate used by graph handles whose backing
// resources retain runtime-owned context/device state. The inactive entry is
// generation-bound and permits lookup only after writer acquisition. The
// handoff entry consumes the caller's single reader admission after it has
// pinned the exact object and closes the reader/creator window first.
hipError_t iree_hip_vmm_object_destroy_begin(uint64_t* out_generation);
hipError_t iree_hip_vmm_object_destroy_handoff_begin(
    bool* out_writer_started_active, uint64_t* out_generation);
void iree_hip_vmm_object_destroy_end(void);

// Instrumented builds report exact cross-translation-unit commit ordering
// through the phase observer. Production call sites compile away completely;
// no observer shim is emitted into the production library. Phase observers are
// instrumentation-only and must not reenter HIP: graph mutation notifications
// may run while graph or executable state serialization is held.
#if defined(IREE_HIP_VMM_TESTING)
void iree_hip_vmm_test_notify_execution_context_queue_release(void* object);
void iree_hip_vmm_test_notify_execution_context_taken(void* object);
void iree_hip_vmm_test_notify_borrowed_device_resolved(void* object);
void iree_hip_vmm_test_notify_explicit_context_queue_release(void* object);
void iree_hip_vmm_test_notify_explicit_context_unregistered(void* object);
void iree_hip_vmm_test_notify_allocation_output(void* object);
void iree_hip_vmm_test_notify_graph_mutation_admitted(void* object);
void iree_hip_vmm_test_notify_graph_exec_active_launch_waiting(void* object);
void iree_hip_vmm_test_notify_graph_peer_transfer_waiting(void* object);
void iree_hip_vmm_test_notify_teardown_stream_waiting(void* object);
void iree_hip_vmm_test_notify_context_resolved(void* object);
void iree_hip_vmm_test_notify_pool_api_admitted(void* object);
uint32_t iree_hip_vmm_test_reader_depth(void);
uint64_t iree_hip_vmm_test_reader_count(void);
typedef void (*iree_hip_vmm_test_callback_t)(void* user_data);
void iree_hip_vmm_test_run_native_callback_window(
    iree_hip_vmm_test_callback_t callback, void* user_data);
typedef void (*iree_hip_vmm_test_lower_address_free_observer_t)(
    bool entering, void* address, size_t size, void* user_data);
hipError_t iree_hip_vmm_test_set_lower_address_free_observer(
    iree_hip_vmm_test_lower_address_free_observer_t observer, void* user_data);
hipError_t iree_hip_vmm_test_clear_lower_address_free_observer(
    iree_hip_vmm_test_lower_address_free_observer_t observer, void* user_data);
void iree_hip_vmm_test_fail_next_lower_observer_clear(void);
bool iree_hip_vmm_test_consume_graph_exec_memcpy_prepare_failure(void);
bool iree_hip_vmm_test_consume_graph_exec_rebuild_failure(void);
bool iree_hip_vmm_test_consume_graph_add_failure(void);
int iree_hip_vmm_test_consume_graph_peer_transfer_failure(void);
void iree_hip_vmm_test_fail_lower_virtual_reserve_after_native_once(void);
void iree_hip_vmm_test_fail_hrx_virtual_reserve_after_native_once(void);
void iree_hip_vmm_test_fail_address_reserve_after_native_once(void);
uint64_t iree_hip_vmm_test_last_virtual_reserve_address(void);
uint64_t iree_hip_vmm_test_last_virtual_reserve_alignment(void);
void iree_hip_vmm_test_fail_physical_create_after_native_once(void);
void iree_hip_vmm_test_fail_cleanup_count(bool physical_memory,
                                          int failure_count);
uint64_t iree_hip_vmm_test_quarantine_count(void);
uint64_t iree_hip_vmm_test_cleanup_attempt_count(bool physical_memory);
uint32_t iree_hip_vmm_test_last_cleanup_status(bool physical_memory);
hipError_t iree_hip_vmm_test_query_allocation_registry(size_t* out_count,
                                                       size_t* out_pending,
                                                       size_t* out_capacity);
void iree_hip_vmm_test_arm_quarantine_drain_pause(void);
void iree_hip_vmm_test_wait_quarantine_drain_paused(void);
void iree_hip_vmm_test_release_quarantine_drain_pause(void);
int iree_hip_vmm_test_quarantine_drain_attempt_kind(int attempt_ordinal);
hipError_t iree_hip_vmm_test_quarantine_drain_without_lifecycle_writer(
    int device_ordinal);
#else
#define iree_hip_vmm_test_notify_execution_context_queue_release(object) \
  ((void)0)
#define iree_hip_vmm_test_notify_execution_context_taken(object) ((void)0)
#define iree_hip_vmm_test_notify_borrowed_device_resolved(object) ((void)0)
#define iree_hip_vmm_test_notify_explicit_context_queue_release(object) \
  ((void)0)
#define iree_hip_vmm_test_notify_explicit_context_unregistered(object) ((void)0)
#define iree_hip_vmm_test_notify_allocation_output(object) ((void)0)
#define iree_hip_vmm_test_notify_graph_mutation_admitted(object) ((void)0)
#define iree_hip_vmm_test_notify_graph_exec_active_launch_waiting(object) \
  ((void)0)
#define iree_hip_vmm_test_notify_graph_peer_transfer_waiting(object) ((void)0)
#define iree_hip_vmm_test_notify_teardown_stream_waiting(object) ((void)0)
#define iree_hip_vmm_test_notify_context_resolved(object) ((void)0)
#define iree_hip_vmm_test_notify_pool_api_admitted(object) ((void)0)
#define iree_hip_vmm_test_consume_graph_exec_memcpy_prepare_failure() false
#define iree_hip_vmm_test_consume_graph_exec_rebuild_failure() false
#define iree_hip_vmm_test_consume_graph_add_failure() false
#define iree_hip_vmm_test_consume_graph_peer_transfer_failure() 0
#endif  // IREE_HIP_VMM_TESTING

// Exclusively retires one explicit context after quiescing its queues and
// evicting every context-local realization of device-scoped VMM access. An
// exact surviving explicit handle may also enter an inactive generation and
// resume its retained native cursor; unknown and primary/stale handles cannot.
hipError_t iree_hip_vmm_context_destroy_begin(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t** out_retained_context,
    iree_hip_context_teardown_t** out_teardown);
void iree_hip_vmm_context_destroy_end(
    iree_hal_streaming_context_t* retained_context);

// Classifies one formal kernel pointer while the launch gate is held. Unknown
// external pointers preserve raw HIP behavior; addresses in a known VMM
// reservation require a mapped extent and a published access alias in exactly
// |context|.
hipError_t iree_hip_vmm_validate_kernel_pointer(
    iree_hal_streaming_context_t* context, uint64_t device_pointer);

// Begins VMM cleanup for a device reset and holds the lifecycle gate until the
// matching end call. On failure the gate is released and device teardown must
// not proceed.
hipError_t iree_hip_vmm_device_reset_begin(
    int device_ordinal, iree_hal_streaming_device_t** out_device);

// Converts the caller's single lifecycle-reader admission into the same
// device-reset writer used above. The reader must already own the exact public
// object being destroyed; it is consumed on every return. New readers and
// creators remain closed from the pin through writer-held revalidation.
hipError_t iree_hip_vmm_device_reset_handoff_begin(
    int device_ordinal, iree_hal_streaming_device_t** out_device);

// Prepares the complete VMM graph transaction while public/device/context
// ledgers remain published. Whole-device scope removes every graph edge
// incident to |device_ordinal|; primary scope removes reservations and aliases
// owned by |primary_context| while preserving process-global physical handles.
hipError_t iree_hip_vmm_prepare_device_reset(
    int device_ordinal, iree_hal_streaming_context_t* primary_context,
    bool whole_device, iree_hip_vmm_teardown_plan_t** out_plan);

// Prepares alias-only retirement for an explicit context. Mappings, access
// grants, reservations, allocations, and public VMM handles are preserved.
hipError_t iree_hip_vmm_prepare_explicit_context_destroy(
    iree_hal_streaming_context_t* context,
    iree_hip_vmm_teardown_plan_t** out_plan);

// Prepares the complete process graph for global deinitialization.
hipError_t iree_hip_vmm_prepare_process_cleanup(
    iree_hip_vmm_teardown_plan_t** out_plan);

// Cancels a PREPARED plan without changing public or native state.
void iree_hip_vmm_cancel_teardown_plan(iree_hip_vmm_teardown_plan_t* plan);

// Consumes a prepared plan after the caller has crossed its first queue-seal
// boundary. Success installs the precomputed final graph and destroys the
// plan. Native failure retains the exact cursor and graph as the inactive
// generation's retry ledger.
hipError_t iree_hip_vmm_commit_teardown_plan(
    iree_hip_vmm_teardown_plan_t* plan);
void iree_hip_vmm_device_reset_end(bool cleanup_committed,
                                   bool reset_succeeded);

// Internal checkpoint immediately after a green context retains its primary
// context. The instrumented test DSO can inject a deterministic failure before
// any ownership is published; production compiles the checkpoint away.
#if defined(IREE_HIP_VMM_TESTING)
hipError_t iree_hip_vmm_green_context_post_retain_checkpoint(void);
#else
#define iree_hip_vmm_green_context_post_retain_checkpoint() (hipSuccess)
#endif  // IREE_HIP_VMM_TESTING

// Releases one primary-context retain under a self-owned lifecycle writer. A
// post-seal native failure has already consumed that public retain and closes
// ordinary admission; repeating this API must not consume it again.
// hipHALDeinit is the recovery owner that resumes the retained native cursor
// and completes the preserved device/context publication ledger.
hipError_t iree_hip_vmm_release_primary_context(int device_ordinal);

// Prepares one exact release while the lifecycle writer is already held. A
// last retain is quiesced here, before a binding-private owner is unpublished.
// Failure leaves the device/context/count and binding owner unchanged.
hipError_t iree_hip_vmm_prepare_primary_context_release(
    int device_ordinal, iree_hal_streaming_context_t* expected_primary_context,
    iree_hip_vmm_primary_release_t* out_release);

// Commits a prepared release without another fallible precommit wait. Reports
// whether native/context retirement crossed its destructive boundary.
hipError_t iree_hip_vmm_commit_primary_context_release(
    iree_hip_vmm_primary_release_t* release, bool* out_cleanup_committed,
    iree_hip_context_teardown_t** out_committed_teardown);

// Drops a prepared transaction pin without changing primary ownership.
void iree_hip_vmm_cancel_primary_context_release(
    iree_hip_vmm_primary_release_t* release);

// Begins process-wide VMM cleanup and holds the lifecycle gate across global
// runtime teardown. This is the recovery entry when a post-seal reset or
// last-primary failure consumed its public retain/handle and no exact surviving
// explicit-context teardown handle is available. A surviving explicit context
// may instead resume that cursor through its exact destroy path. This entry
// admits the inactive generation and resumes the exact retained cursor before
// preparing any new process cleanup. On failure the gate is released and
// teardown must stop.
hipError_t iree_hip_vmm_deinitialize_begin(void);
hipError_t iree_hip_vmm_deinitialize_prepare(
    iree_hip_vmm_teardown_plan_t** out_plan);
void iree_hip_vmm_deinitialize_commit(void);
void iree_hip_vmm_deinitialize_end(bool cleanup_committed,
                                   bool teardown_succeeded);

// Counts references to |context| owned by live VMM reservation/allocation
// metadata and exact-context access aliases. Requires exclusive lifecycle
// writer admission; the returned count is stable until that writer ends.
size_t iree_hip_vmm_context_owned_reference_count(
    iree_hal_streaming_context_t* context);

// Debug/test invariant for committed cleanup failure: every preserved
// published VMM wrapper in |context|'s table is named exactly once by a live
// reservation binding. Requires exclusive lifecycle writer admission.
void iree_hip_vmm_assert_preserved_context_bindings(
    iree_hal_streaming_context_t* context);

// Copies process-global mapped-extent metadata without exposing registry
// objects or requiring an access grant in the current context.
hipError_t iree_hip_vmm_query_pointer(
    const void* ptr, iree_hip_vmm_pointer_snapshot_t* out_snapshot);

// Same query for callers already holding lifecycle reader admission across a
// larger public operation. This never opens or closes admission itself.
hipError_t iree_hip_vmm_query_pointer_under_admission(
    const void* ptr, iree_hip_vmm_pointer_snapshot_t* out_snapshot);

hipError_t iree_hip_vmm_is_supported(int device_ordinal, bool* out_supported);
hipError_t iree_hip_vmm_address_reserve(void** ptr, size_t size,
                                        size_t alignment, void* address,
                                        unsigned long long flags);
hipError_t iree_hip_vmm_address_free(void* device_ptr, size_t size);
hipError_t iree_hip_vmm_create(hipMemGenericAllocationHandle_t* handle,
                               size_t size,
                               const hipMemAllocationProp* properties,
                               unsigned long long flags);
hipError_t iree_hip_vmm_release(hipMemGenericAllocationHandle_t handle);
hipError_t iree_hip_vmm_map(void* ptr, size_t size, size_t offset,
                            hipMemGenericAllocationHandle_t handle,
                            unsigned long long flags);
hipError_t iree_hip_vmm_unmap(void* ptr, size_t size);
hipError_t iree_hip_vmm_set_access(void* ptr, size_t size,
                                   const hipMemAccessDesc* descriptors,
                                   size_t count);
hipError_t iree_hip_vmm_get_access(unsigned long long* flags,
                                   const hipMemLocation* location, void* ptr);
hipError_t iree_hip_vmm_get_allocation_granularity(
    size_t* granularity, const hipMemAllocationProp* properties,
    hipMemAllocationGranularity_flags option);
hipError_t iree_hip_vmm_get_allocation_properties(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle);
hipError_t iree_hip_vmm_retain_allocation_handle(
    hipMemGenericAllocationHandle_t* handle, void* address);

#endif  // LIBHRX_SRC_BINDING_HIP_VMM_H_
