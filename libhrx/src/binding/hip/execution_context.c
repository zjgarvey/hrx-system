// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/execution_context.h"

#include "binding/hip/execution_resource.h"
#include "binding/hip/execution_resource_descriptor.h"
#include "binding/hip/stream.h"
#include "binding/hip/vmm.h"
#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/threading/call_once.h"

typedef enum iree_hip_execution_context_kind_e {
  IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY = 0,
  IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED = 1,
} iree_hip_execution_context_kind_t;

// A HIP execution context is a binding-private scheduling domain layered over
// the device primary streaming context. Hardware queues are realized lazily by
// streams; this control-plane object does not itself schedule work.
struct ihipExecutionCtx_t {
  // Reference count including the live registry or public-handle ownership.
  iree_atomic_ref_count_t ref_count;

  // Determines whether the context is device-managed or resource-partitioned.
  iree_hip_execution_context_kind_t kind;

  // Serializes liveness and execution-context stream membership.
  iree_slim_mutex_t mutex;

  // Host allocator owning this context allocation.
  iree_allocator_t host_allocator;

  // Borrowed device registry entry used while the HIP runtime is initialized.
  iree_hal_streaming_device_t* device;

  // Stable ordinal of |device| used without dereferencing the registry entry.
  hipDevice_t device_ordinal;

  // Owning primary-context reference paired with a device usage count.
  iree_hal_streaming_context_t* primary_context;

  // Resource descriptor consumed when this context was created.
  iree_hip_execution_resource_descriptor_t* descriptor;

  // Canonical union of the descriptor SM resources returned by queries.
  hipDevResource sm_resource;

  // Stable process-unique context identifier.
  unsigned long long context_id;

  // Intrusive list of retained streams created on this execution context.
  hipStream_t stream_head;

  // Outstanding execution-context event waits inherited by newly created
  // streams. Immutable once published and guarded by |mutex|.
  iree_hal_fence_t* stream_wait_frontier;

  // Timeline covering context-wide event records for this exact partitioned
  // scheduling domain. Primary contexts use the common context timeline.
  // Guarded by |mutex|.
  iree_hal_streaming_operation_timeline_t event_record_timeline;

  // Next context in the live-handle registry.
  struct ihipExecutionCtx_t* next_live_context;
};

typedef struct iree_hip_execution_context_stream_snapshot_t {
  // Common primary streaming context retained for the snapshot lifetime.
  iree_hal_streaming_context_t* common_context;

  // Retained common streams participating in the execution context.
  iree_hal_streaming_stream_t** streams;

  // Number of retained entries in |streams|.
  iree_host_size_t stream_count;
} iree_hip_execution_context_stream_snapshot_t;

typedef struct iree_hip_execution_context_registry_t {
  // Serializes live-handle lookup, publication, and invalidation.
  iree_slim_mutex_t mutex;

  // Intrusive list of live public execution-context handles.
  hipExecutionCtx_t head;
} iree_hip_execution_context_registry_t;

static iree_once_flag iree_hip_execution_context_registry_once =
    IREE_ONCE_FLAG_INIT;
static iree_hip_execution_context_registry_t
    iree_hip_execution_context_registry;

static hipError_t iree_hip_execution_context_destroy_preflight(
    hipExecutionCtx_t context);

// Next process-unique context identifier. Zero permanently marks exhaustion.
static iree_atomic_uint64_t iree_hip_next_execution_context_id =
    IREE_ATOMIC_VAR_INIT(1);

static void iree_hip_execution_context_registry_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_execution_context_registry.mutex);
  iree_hip_execution_context_registry.head = NULL;
}

static hipError_t iree_hip_execution_context_consume_status(
    iree_status_t status) {
  const iree_status_code_t status_code = iree_status_code(status);
  iree_status_free(status);
  switch (status_code) {
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

static iree_status_t iree_hip_execution_context_allocate_id(
    unsigned long long* out_context_id) {
  uint64_t current = iree_atomic_load(&iree_hip_next_execution_context_id,
                                      iree_memory_order_relaxed);
  while (current != 0) {
    const uint64_t next = current + 1;
    if (iree_atomic_compare_exchange_weak(
            &iree_hip_next_execution_context_id, &current, next,
            iree_memory_order_relaxed, iree_memory_order_relaxed)) {
      *out_context_id = current;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "HIP execution-context ID space is exhausted");
}

static void iree_hip_execution_context_retain(hipExecutionCtx_t context) {
  if (!context) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

static void iree_hip_execution_context_release(hipExecutionCtx_t context) {
  if (!context) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) != 1) return;
  IREE_ASSERT(context->device == NULL);
  IREE_ASSERT(context->primary_context == NULL);
  IREE_ASSERT(context->stream_head == NULL);
  IREE_ASSERT(context->stream_wait_frontier == NULL);
  IREE_ASSERT(context->event_record_timeline.semaphore == NULL);
  IREE_ASSERT(context->event_record_timeline.pending_value == 0);
  iree_hip_execution_resource_descriptor_release(context->descriptor);
  iree_slim_mutex_deinitialize(&context->mutex);
  iree_allocator_free(context->host_allocator, context);
}

// Detaches an execution context after its caller has synchronously quiesced
// every stream and event record while the handle remained published.
static void iree_hip_execution_context_deinitialize(hipExecutionCtx_t context,
                                                    bool abort_captures) {
  iree_slim_mutex_lock(&context->mutex);
  iree_hal_fence_t* stream_wait_frontier = context->stream_wait_frontier;
  iree_hal_streaming_operation_timeline_t event_record_timeline =
      context->event_record_timeline;
  context->device = NULL;
  context->primary_context = NULL;
  context->stream_wait_frontier = NULL;
  context->event_record_timeline = (iree_hal_streaming_operation_timeline_t){0};

  // Remove all members as one state transition. Each list entry owns a stream
  // reference and each association owns a context reference, keeping both
  // objects live while blocking work is drained outside the lock.
  hipStream_t stream_head = context->stream_head;
  context->stream_head = NULL;
  for (hipStream_t stream = stream_head; stream;
       stream = stream->next_execution_context_stream) {
    iree_slim_mutex_lock(&stream->mutex);
    IREE_ASSERT(stream->execution_context == context);
    stream->execution_context = NULL;
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_slim_mutex_unlock(&context->mutex);
  iree_hal_fence_release(stream_wait_frontier);

  while (stream_head) {
    hipStream_t stream = stream_head;
    stream_head = stream->next_execution_context_stream;
    stream->next_execution_context_stream = NULL;
    iree_hal_streaming_stream_t* common_stream = NULL;
    iree_hal_streaming_context_t* common_context = NULL;
    if (iree_hip_stream_detach(stream, &common_stream, &common_context)) {
      if (common_context) {
        if (abort_captures) {
          iree_hal_streaming_stream_abort_capture_quiesced(common_stream);
        }
        // A final dynamic-queue release may synchronously wait in the backend.
        // This entire detach runs before the public execution handle is taken.
        iree_hip_vmm_test_notify_execution_context_queue_release(common_stream);
        iree_hal_streaming_stream_detach_quiesced(common_context,
                                                  common_stream);
      }
      iree_hal_streaming_stream_release(common_stream);
      iree_hal_streaming_context_release(common_context);
    }
    iree_hip_execution_context_release(context);
    iree_hip_stream_release(stream);
  }

  iree_hal_semaphore_release(event_record_timeline.semaphore);
}

// Resolves |handle| against the live registry and returns a retained context.
static hipExecutionCtx_t iree_hip_execution_context_resolve_live(
    hipExecutionCtx_t handle) {
  if (!handle) return NULL;
  iree_call_once(&iree_hip_execution_context_registry_once,
                 iree_hip_execution_context_registry_initialize);

  hipExecutionCtx_t retained_context = NULL;
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  for (hipExecutionCtx_t current = iree_hip_execution_context_registry.head;
       current; current = current->next_live_context) {
    if (current == handle) {
      iree_atomic_ref_count_inc(&current->ref_count);
      retained_context = current;
      break;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);
  return retained_context;
}

static void iree_hip_execution_context_publish(hipExecutionCtx_t context) {
  iree_call_once(&iree_hip_execution_context_registry_once,
                 iree_hip_execution_context_registry_initialize);
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  context->next_live_context = iree_hip_execution_context_registry.head;
  iree_hip_execution_context_registry.head = context;
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);
}

static hipExecutionCtx_t iree_hip_execution_context_take_partitioned(
    hipExecutionCtx_t handle, hipExecutionCtx_t expected_context) {
  if (!handle) return NULL;
  iree_call_once(&iree_hip_execution_context_registry_once,
                 iree_hip_execution_context_registry_initialize);

  hipExecutionCtx_t owned_context = NULL;
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  hipExecutionCtx_t* link = &iree_hip_execution_context_registry.head;
  while (*link && *link != handle) {
    link = &(*link)->next_live_context;
  }
  if (*link == expected_context &&
      (*link)->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED) {
    owned_context = *link;
    *link = owned_context->next_live_context;
    owned_context->next_live_context = NULL;
  }
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);
  return owned_context;
}

hipError_t iree_hip_execution_context_primary(
    iree_hal_streaming_device_t* device, hipExecutionCtx_t* out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  iree_call_once(&iree_hip_execution_context_registry_once,
                 iree_hip_execution_context_registry_initialize);

  hipError_t result = hipSuccess;
  hipExecutionCtx_t context = NULL;
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  for (hipExecutionCtx_t current = iree_hip_execution_context_registry.head;
       current; current = current->next_live_context) {
    if (current->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY &&
        current->device_ordinal == (hipDevice_t)device->ordinal) {
      context = current;
      break;
    }
  }

  iree_hal_streaming_context_t* primary_context = NULL;
  if (!context) {
    iree_status_t status =
        iree_hal_streaming_device_get_or_create_primary_context(
            device, &primary_context);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  iree_hal_queue_t* primary_queue = NULL;
  if (!context && result == hipSuccess) {
    iree_status_t status =
        iree_hal_streaming_device_select_primary_queue(device, &primary_queue);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  hipDevResource sm_resource = {0};
  if (!context && result == hipSuccess) {
    iree_status_t status = iree_hip_execution_resource_create_sm(
        device, iree_hal_queue_family(primary_queue),
        (iree_hal_queue_execution_resource_list_t){0},
        hipDevSmResourceGroupDefault, &sm_resource);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  unsigned long long context_id = 0;
  if (!context && result == hipSuccess) {
    iree_status_t status = iree_hip_execution_context_allocate_id(&context_id);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  const iree_allocator_t host_allocator = iree_allocator_system();
  hipExecutionCtx_t new_context = NULL;
  if (!context && result == hipSuccess) {
    iree_status_t status = iree_allocator_malloc(
        host_allocator, sizeof(*new_context), (void**)&new_context);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }
  if (new_context) {
    iree_atomic_ref_count_init(&new_context->ref_count);
    iree_slim_mutex_initialize(&new_context->mutex);
    new_context->kind = IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY;
    new_context->host_allocator = host_allocator;
    new_context->device = device;
    new_context->device_ordinal = (hipDevice_t)device->ordinal;
    new_context->primary_context = NULL;
    new_context->descriptor = NULL;
    new_context->sm_resource = sm_resource;
    new_context->context_id = context_id;
    new_context->stream_head = NULL;
    new_context->stream_wait_frontier = NULL;
    new_context->event_record_timeline =
        (iree_hal_streaming_operation_timeline_t){0};
    new_context->next_live_context = iree_hip_execution_context_registry.head;
    iree_hip_execution_context_registry.head = new_context;
    context = new_context;
  }
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);

  if (result == hipSuccess) {
    *out_context = context;
  }
  return result;
}

hipError_t iree_hip_execution_context_create(
    iree_hal_streaming_device_t* device, hipDevResourceDesc_t descriptor_handle,
    hipExecutionCtx_t* out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);

  iree_hip_execution_resource_descriptor_t* retained_descriptor = NULL;
  if (!iree_hip_execution_resource_descriptor_lookup_retain(
          descriptor_handle, &retained_descriptor)) {
    return hipErrorInvalidValue;
  }

  hipError_t result = hipSuccess;
  const uint64_t table_incarnation =
      iree_hal_streaming_execution_resource_table_incarnation(
          &device->execution_resource_table);
  if (retained_descriptor->device_ordinal != device->ordinal) {
    result = hipErrorInvalidDevice;
  } else if (retained_descriptor->table_incarnation != table_incarnation) {
    result = hipErrorInvalidResourceConfiguration;
  }

  const iree_hal_queue_family_t* queue_family = NULL;
  const iree_hal_streaming_execution_resource_set_t* sm_resource_set = NULL;
  if (result == hipSuccess) {
    queue_family = iree_hal_device_queue_family(
        device->hal_device, retained_descriptor->queue_family_ordinal);
    sm_resource_set = iree_hal_streaming_execution_resource_table_resolve(
        &device->execution_resource_table,
        retained_descriptor->sm_resource_set_id);
    if (!queue_family || !sm_resource_set ||
        sm_resource_set->queue_family_ordinal !=
            retained_descriptor->queue_family_ordinal) {
      result = hipErrorInvalidResourceConfiguration;
    }
  }

  iree_hal_streaming_context_t* primary_context = NULL;
  if (result == hipSuccess) {
    iree_status_t status = iree_hal_streaming_device_retain_primary_context(
        device, &primary_context);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_green_context_post_retain_checkpoint();
  }

  iree_hal_semaphore_t* event_record_semaphore = NULL;
  if (result == hipSuccess) {
    iree_status_t status = iree_hal_semaphore_create(
        primary_context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &event_record_semaphore);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  hipDevResource sm_resource = {0};
  if (result == hipSuccess) {
    iree_status_t status = iree_hip_execution_resource_create_sm(
        device, queue_family, sm_resource_set->resources,
        hipDevSmResourceGroupDefault, &sm_resource);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  unsigned long long context_id = 0;
  if (result == hipSuccess) {
    iree_status_t status = iree_hip_execution_context_allocate_id(&context_id);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  const iree_allocator_t host_allocator = iree_allocator_system();
  hipExecutionCtx_t context = NULL;
  if (result == hipSuccess) {
    iree_status_t status = iree_allocator_malloc(
        host_allocator, sizeof(*context), (void**)&context);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  iree_hip_execution_resource_descriptor_t* owned_descriptor = NULL;
  if (result == hipSuccess && !iree_hip_execution_resource_descriptor_take(
                                  descriptor_handle, &owned_descriptor)) {
    result = hipErrorInvalidValue;
  }
  if (result == hipSuccess) {
    iree_atomic_ref_count_init(&context->ref_count);
    iree_slim_mutex_initialize(&context->mutex);
    context->kind = IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED;
    context->host_allocator = host_allocator;
    context->device = device;
    context->device_ordinal = (hipDevice_t)device->ordinal;
    context->primary_context = primary_context;
    context->descriptor = owned_descriptor;
    context->sm_resource = sm_resource;
    context->context_id = context_id;
    context->stream_head = NULL;
    context->stream_wait_frontier = NULL;
    context->event_record_timeline = (iree_hal_streaming_operation_timeline_t){
        .semaphore = event_record_semaphore,
        .pending_value = 0,
    };
    context->next_live_context = NULL;
    iree_hip_execution_context_publish(context);
    *out_context = context;
    iree_hip_execution_resource_descriptor_release(retained_descriptor);
    return hipSuccess;
  }

  iree_allocator_free(host_allocator, context);
  iree_hal_semaphore_release(event_record_semaphore);
  if (primary_context) {
    iree_hal_streaming_device_rollback_primary_context_retain(device,
                                                              primary_context);
  }
  iree_hip_execution_resource_descriptor_release(owned_descriptor);
  iree_hip_execution_resource_descriptor_release(retained_descriptor);
  return result;
}

hipError_t iree_hip_execution_context_destroy(hipExecutionCtx_t context) {
  hipError_t result = iree_hip_vmm_launch_begin();
  if (result != hipSuccess) return result;
  hipExecutionCtx_t retained_context =
      iree_hip_execution_context_resolve_live(context);
  if (!retained_context ||
      retained_context->kind !=
          IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED) {
    iree_hip_execution_context_release(retained_context);
    iree_hip_vmm_launch_end();
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_test_notify_context_resolved(retained_context);
  const hipDevice_t device_ordinal = retained_context->device_ordinal;
  iree_hal_streaming_device_t* admitted_device = NULL;
  result =
      iree_hip_vmm_device_reset_handoff_begin(device_ordinal, &admitted_device);
  if (result != hipSuccess) {
    iree_hip_execution_context_release(retained_context);
    return result;
  }
  IREE_ASSERT((hipDevice_t)admitted_device->ordinal == device_ordinal);

  // The raw handle was pinned before joining the writer queue. A preceding
  // destroy may have consumed its public mapping while this caller waited, so
  // revalidate that the registry still names this exact retained incarnation
  // before reading or mutating any of its teardown state.
  hipExecutionCtx_t revalidated_context =
      iree_hip_execution_context_resolve_live(context);
  const bool exact_public_incarnation =
      revalidated_context == retained_context &&
      revalidated_context->kind ==
          IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED;
  iree_hip_execution_context_release(revalidated_context);
  if (!exact_public_incarnation) {
    iree_hip_execution_context_release(retained_context);
    iree_hip_vmm_device_reset_end(/*cleanup_committed=*/false,
                                  /*reset_succeeded=*/false);
    return hipErrorInvalidValue;
  }

  // A captured member may own a graph whose destruction performs fallible
  // pageable-host cleanup. Reject it before any wait, release preparation, or
  // state mutation so the caller can end/abort capture and retry. Lifecycle
  // writer admission keeps this capture-free snapshot stable through take.
  result = iree_hip_execution_context_destroy_preflight(retained_context);
  if (result != hipSuccess) {
    iree_hip_execution_context_release(retained_context);
    iree_hip_vmm_device_reset_end(/*cleanup_committed=*/false,
                                  /*reset_succeeded=*/false);
    return result;
  }

  iree_hal_streaming_context_t* expected_primary_context = NULL;
  iree_slim_mutex_lock(&retained_context->mutex);
  expected_primary_context = retained_context->primary_context;
  iree_slim_mutex_unlock(&retained_context->mutex);

  // Every fallible stream and event wait is precommit and runs with the exact
  // public handle still discoverable. Failure leaves its handle, membership,
  // streams, primary retain, and runtime admission unchanged.
  result = iree_hip_execution_context_synchronize(retained_context);
  if (result != hipSuccess) {
    iree_hip_execution_context_release(retained_context);
    iree_hip_vmm_device_reset_end(/*cleanup_committed=*/false,
                                  /*reset_succeeded=*/false);
    return result;
  }

  // Quiesce a possible last retain while the exact handle and all of its
  // streams remain published. A preparation failure therefore leaves every
  // observable execution-context and primary-context field unchanged.
  iree_hip_vmm_primary_release_t primary_release = {0};
  result = iree_hip_vmm_prepare_primary_context_release(
      device_ordinal, expected_primary_context, &primary_release);
  if (result != hipSuccess) {
    iree_hip_execution_context_release(retained_context);
    iree_hip_vmm_device_reset_end(/*cleanup_committed=*/false,
                                  /*reset_succeeded=*/false);
    return result;
  }

  // A last-primary commit may release the primary/default queue and enter the
  // same backend path. Queue sealing is the first irreversible transition;
  // keep the exact execution handle and all member ownership published until
  // the primary/VMM plan has crossed that boundary.
  iree_hip_vmm_test_notify_execution_context_queue_release(
      primary_release.context);
  bool cleanup_committed = false;
  iree_hip_context_teardown_t* teardown = NULL;
  result = iree_hip_vmm_commit_primary_context_release(
      &primary_release, &cleanup_committed, &teardown);

  // The prepared primary teardown has now sealed/detached target queues (or a
  // non-last retain required no seal). Consume the partitioned member ledger
  // without placing a queue release ahead of the fail-closed boundary.
  iree_hip_execution_context_deinitialize(retained_context,
                                          /*abort_captures=*/false);

  hipExecutionCtx_t owned_context =
      iree_hip_execution_context_take_partitioned(context, retained_context);
  iree_hip_vmm_test_notify_execution_context_taken(owned_context);
  IREE_ASSERT(owned_context == retained_context,
              "writer-held execution handle changed during commit");
  if (!owned_context) {
    // This is unreachable under writer admission. Fail closed rather than
    // restoring an already-detached public object and consumed ownership.
    iree_hip_execution_context_release(retained_context);
    iree_hip_vmm_device_reset_end(/*cleanup_committed=*/true,
                                  /*reset_succeeded=*/false);
    iree_hip_context_teardown_finish(teardown);
    return hipErrorInvalidValue;
  }
  iree_hip_execution_context_release(retained_context);
  // Taking the handle completes binding-state commit. All potentially waiting
  // execution stream, queue, timeline, and primary teardown happened while
  // the handle remained published. Only reference release remains.
  iree_hip_execution_context_release(owned_context);
  iree_hip_vmm_device_reset_end(/*cleanup_committed=*/true,
                                result == hipSuccess);
  iree_hip_context_teardown_finish(teardown);
  return result;
}

hipError_t iree_hip_execution_context_stream_create(
    hipExecutionCtx_t context_handle, unsigned int flags, int priority,
    hipStream_t* out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  if (flags & ~hipStreamNonBlocking) return hipErrorInvalidValue;

  hipExecutionCtx_t context =
      iree_hip_execution_context_resolve_live(context_handle);
  if (!context) return hipErrorInvalidValue;

  hipError_t result = hipSuccess;
  iree_hal_streaming_context_t* common_context = NULL;
  iree_hal_queue_t* queue = NULL;
  hipStream_t stream = NULL;

  iree_slim_mutex_lock(&context->mutex);
  iree_hal_streaming_device_t* device = context->device;
  if (!device) {
    result = hipErrorInvalidValue;
  } else if (context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY) {
    iree_status_t status =
        iree_hal_streaming_device_get_or_create_primary_context(
            device, &common_context);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  } else {
    common_context = context->primary_context;
    if (!common_context) result = hipErrorInvalidValue;
  }

  const iree_hal_streaming_execution_resource_set_t* resource_set = NULL;
  if (result == hipSuccess) {
    result = iree_hip_execution_resource_resolve_sm_for_device(
        &context->sm_resource, device, &resource_set);
  }

  const iree_hal_queue_family_t* queue_family = NULL;
  if (result == hipSuccess) {
    queue_family = iree_hal_device_queue_family(
        device->hal_device, resource_set->queue_family_ordinal);
    if (!queue_family) result = hipErrorInvalidResourceConfiguration;
  }

  int hip_priority = 0;
  iree_hal_queue_priority_t queue_priority = IREE_HAL_QUEUE_PRIORITY_NORMAL;
  if (result == hipSuccess) {
    queue_priority = iree_hip_queue_family_select_priority(
        queue_family, priority, &hip_priority);
  }

  if (result == hipSuccess) {
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    queue_params.priority = queue_priority;
    queue_params.execution_resources = resource_set->resources;
    iree_status_t status = iree_hal_device_acquire_queue(
        device->hal_device, queue_family, &queue_params, &queue);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }

  if (result == hipSuccess) {
    const iree_hal_semaphore_list_t initial_wait_semaphores =
        iree_hal_fence_semaphore_list(context->stream_wait_frontier);
    iree_status_t status =
        iree_hip_stream_create(common_context, queue, hipStreamNonBlocking,
                               hip_priority, initial_wait_semaphores, &stream);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_context_consume_status(status);
    }
  }
  iree_hal_queue_release(queue);

  if (result == hipSuccess) {
    iree_hip_execution_context_retain(context);
    iree_hip_stream_retain(stream);
    iree_slim_mutex_lock(&stream->mutex);
    IREE_ASSERT(stream->execution_context == NULL);
    IREE_ASSERT(stream->next_execution_context_stream == NULL);
    stream->execution_context = context;
    stream->next_execution_context_stream = context->stream_head;
    context->stream_head = stream;
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_slim_mutex_unlock(&context->mutex);

  iree_hip_execution_context_release(context);
  if (result == hipSuccess) *out_stream = stream;
  return result;
}

void iree_hip_execution_context_unregister_stream(hipStream_t stream) {
  if (!stream) return;

  // The association itself owns a context reference. Retain it while holding
  // the stream mutex so context teardown cannot clear and release the
  // association between the load and retain.
  iree_slim_mutex_lock(&stream->mutex);
  hipExecutionCtx_t context = stream->execution_context;
  iree_hip_execution_context_retain(context);
  iree_slim_mutex_unlock(&stream->mutex);
  if (!context) return;

  bool was_removed = false;
  iree_slim_mutex_lock(&context->mutex);
  iree_slim_mutex_lock(&stream->mutex);
  if (stream->execution_context == context) {
    hipStream_t* link = &context->stream_head;
    while (*link && *link != stream) {
      link = &(*link)->next_execution_context_stream;
    }
    IREE_ASSERT(*link == stream);
    if (*link == stream) {
      *link = stream->next_execution_context_stream;
      stream->execution_context = NULL;
      stream->next_execution_context_stream = NULL;
      was_removed = true;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&context->mutex);

  if (was_removed) iree_hip_execution_context_release(context);
  if (was_removed) iree_hip_stream_release(stream);
  iree_hip_execution_context_release(context);
}

// Retains the common primary streaming context underlying |context| while the
// caller holds |context->mutex|. The output is unchanged on failure.
static iree_status_t iree_hip_execution_context_retain_common_locked(
    hipExecutionCtx_t context,
    iree_hal_streaming_context_t** out_common_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_common_context);

  iree_hal_streaming_device_t* device = context->device;
  if (IREE_UNLIKELY(!device)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "execution context has been destroyed");
  }

  iree_hal_streaming_context_t* common_context = NULL;
  iree_status_t status = iree_ok_status();
  if (context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY) {
    status = iree_hal_streaming_device_get_or_create_primary_context(
        device, &common_context);
  } else {
    common_context = context->primary_context;
    if (IREE_UNLIKELY(!common_context)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "execution context has been destroyed");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_context_retain(common_context);
    *out_common_context = common_context;
  }
  return status;
}

// Takes a retained snapshot of the common streams tracked by |context| while
// the caller holds |context->mutex|. Primary contexts include every stream in
// the common context; resource-partitioned contexts include only their member
// streams. The output is unchanged on failure.
static iree_status_t iree_hip_execution_context_snapshot_locked(
    hipExecutionCtx_t context,
    iree_hip_execution_context_stream_snapshot_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_snapshot);

  iree_hal_streaming_context_t* common_context = NULL;
  IREE_RETURN_IF_ERROR(iree_hip_execution_context_retain_common_locked(
      context, &common_context));

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status = iree_ok_status();
  if (context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY) {
    status = iree_hal_streaming_context_snapshot_streams(
        common_context, &streams, &stream_count);
  } else {
    iree_host_size_t stream_capacity = 0;
    for (hipStream_t stream = context->stream_head; stream;
         stream = stream->next_execution_context_stream) {
      if (IREE_UNLIKELY(!iree_host_size_checked_add(stream_capacity, 1,
                                                    &stream_capacity))) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "execution-context stream count overflow");
        break;
      }
    }

    if (iree_status_is_ok(status) && stream_capacity > 0) {
      iree_host_size_t streams_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              stream_capacity, sizeof(*streams), &streams_size))) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "execution-context stream snapshot size overflow");
      } else {
        status = iree_allocator_malloc(common_context->host_allocator,
                                       streams_size, (void**)&streams);
      }
    }

    for (hipStream_t stream = context->stream_head;
         stream && iree_status_is_ok(status);
         stream = stream->next_execution_context_stream) {
      iree_slim_mutex_lock(&stream->mutex);
      iree_hal_streaming_stream_t* common_stream = stream->stream;
      if (IREE_LIKELY(common_stream)) {
        iree_hal_streaming_stream_retain(common_stream);
        streams[stream_count++] = common_stream;
      } else {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "execution-context member stream is detached");
      }
      iree_slim_mutex_unlock(&stream->mutex);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_snapshot = (iree_hip_execution_context_stream_snapshot_t){
        .common_context = common_context,
        .streams = streams,
        .stream_count = stream_count,
    };
  } else {
    iree_hal_streaming_context_release_stream_snapshot(common_context, streams,
                                                       stream_count);
    iree_hal_streaming_context_release(common_context);
  }
  return status;
}

static void iree_hip_execution_context_release_snapshot(
    iree_hip_execution_context_stream_snapshot_t* snapshot) {
  if (!snapshot->common_context) return;
  iree_hal_streaming_context_release_stream_snapshot(
      snapshot->common_context, snapshot->streams, snapshot->stream_count);
  iree_hal_streaming_context_release(snapshot->common_context);
  *snapshot = (iree_hip_execution_context_stream_snapshot_t){0};
}

static hipError_t iree_hip_execution_context_destroy_preflight(
    hipExecutionCtx_t context) {
  iree_hip_execution_context_stream_snapshot_t snapshot = {0};
  iree_slim_mutex_lock(&context->mutex);
  iree_status_t status =
      iree_hip_execution_context_snapshot_locked(context, &snapshot);
  iree_slim_mutex_unlock(&context->mutex);
  if (!iree_status_is_ok(status)) {
    return iree_hip_execution_context_consume_status(status);
  }

  bool has_capture = false;
  for (iree_host_size_t i = 0; i < snapshot.stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = snapshot.streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    has_capture =
        stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE ||
        stream->capture_graph || stream->capture_graph_owned;
    iree_slim_mutex_unlock(&stream->mutex);
    if (has_capture) break;
  }
  iree_hip_execution_context_release_snapshot(&snapshot);
  return has_capture ? hipErrorStreamCaptureUnsupported : hipSuccess;
}

// Invalidates every capture represented in |snapshot| and returns whether any
// active or already-invalidated capture was present.
static bool iree_hip_execution_context_invalidate_captures(
    const iree_hip_execution_context_stream_snapshot_t* snapshot) {
  bool had_capture = false;
  for (iree_host_size_t i = 0; i < snapshot->stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = snapshot->streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      had_capture = true;
      if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
        iree_hal_streaming_stream_set_capture_status(
            stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      }
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  return had_capture;
}

// Invalidates streams participating in |capture_graph|. The event retaining
// the graph also retains |common_context| for the duration of the call.
static iree_status_t iree_hip_execution_context_invalidate_capture_graph(
    iree_hal_streaming_context_t* common_context,
    iree_hal_streaming_graph_t* capture_graph) {
  IREE_ASSERT_ARGUMENT(common_context);
  IREE_ASSERT_ARGUMENT(capture_graph);

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      common_context, &streams, &stream_count);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < stream_count; ++i) {
      iree_hal_streaming_stream_t* stream = streams[i];
      iree_slim_mutex_lock(&stream->mutex);
      if (stream->capture_graph == capture_graph &&
          stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
        iree_hal_streaming_stream_set_capture_status(
            stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      }
      iree_slim_mutex_unlock(&stream->mutex);
    }
  }
  iree_hal_streaming_context_release_stream_snapshot(common_context, streams,
                                                     stream_count);
  return status;
}

hipError_t iree_hip_execution_context_record_event(
    hipExecutionCtx_t context_handle, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  hipExecutionCtx_t context =
      iree_hip_execution_context_resolve_live(context_handle);
  if (!context) return hipErrorInvalidValue;

  iree_hip_execution_context_stream_snapshot_t snapshot = {0};
  iree_slim_mutex_lock(&context->mutex);
  iree_status_t status =
      iree_hip_execution_context_snapshot_locked(context, &snapshot);

  hipError_t result = hipSuccess;
  if (!iree_status_is_ok(status)) {
    result = iree_hip_execution_context_consume_status(status);
  } else if (event->context != snapshot.common_context) {
    result = hipErrorInvalidHandle;
  } else if (iree_hip_execution_context_invalidate_captures(&snapshot)) {
    result = hipErrorStreamCaptureUnsupported;
  } else {
    iree_hal_streaming_operation_timeline_t* additional_timeline =
        context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED
            ? &context->event_record_timeline
            : NULL;
    status = iree_hal_streaming_event_record_after_streams(
        event, snapshot.streams, snapshot.stream_count, additional_timeline);
    if (iree_status_code(status) == IREE_STATUS_FAILED_PRECONDITION &&
        iree_hip_execution_context_invalidate_captures(&snapshot)) {
      iree_status_free(status);
      result = hipErrorStreamCaptureUnsupported;
    } else {
      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_flush(snapshot.common_context->queue);
      }
      result = iree_status_is_ok(status)
                   ? hipSuccess
                   : iree_hip_execution_context_consume_status(status);
    }
  }
  iree_slim_mutex_unlock(&context->mutex);

  iree_hip_execution_context_release_snapshot(&snapshot);
  iree_hip_execution_context_release(context);
  return result;
}

hipError_t iree_hip_execution_context_wait_event(
    hipExecutionCtx_t context_handle, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  hipExecutionCtx_t context =
      iree_hip_execution_context_resolve_live(context_handle);
  if (!context) return hipErrorInvalidValue;

  iree_hal_streaming_graph_t* capture_graph =
      iree_hal_streaming_event_acquire_capture_graph(event);
  iree_status_t status = iree_ok_status();
  const bool event_is_captured = capture_graph != NULL;
  if (capture_graph) {
    status = iree_hip_execution_context_invalidate_capture_graph(event->context,
                                                                 capture_graph);
  }
  iree_hal_streaming_graph_release(capture_graph);

  iree_hip_execution_context_stream_snapshot_t snapshot = {0};
  iree_hal_streaming_recorded_point_t recorded_point = {0};
  iree_hal_fence_t* new_frontier = NULL;
  iree_hal_fence_t* old_frontier = NULL;
  bool target_is_captured = false;
  bool fan_out_wait = false;
  iree_slim_mutex_lock(&context->mutex);
  if (iree_status_is_ok(status)) {
    status = iree_hip_execution_context_snapshot_locked(context, &snapshot);
  }
  if (iree_status_is_ok(status)) {
    target_is_captured =
        iree_hip_execution_context_invalidate_captures(&snapshot);
  }

  if (iree_status_is_ok(status) && !event_is_captured && !target_is_captured &&
      context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY) {
    iree_slim_mutex_unlock(&context->mutex);
    status =
        iree_hal_streaming_context_wait_event(snapshot.common_context, event);
    iree_slim_mutex_lock(&context->mutex);
  } else if (iree_status_is_ok(status) && !event_is_captured &&
             !target_is_captured) {
    iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
    uint64_t current_value = 0;
    if (recorded_point.semaphore) {
      status =
          iree_hal_semaphore_query(recorded_point.semaphore, &current_value);
    }
    if (iree_status_is_ok(status) && recorded_point.semaphore &&
        current_value < recorded_point.value) {
      status = iree_hal_streaming_wait_frontier_extend(
          context->stream_wait_frontier, recorded_point.semaphore,
          recorded_point.value, context->host_allocator, &new_frontier);
    }
    if (iree_status_is_ok(status) && new_frontier) {
      // Publish before dropping the domain lock so a stream is either in the
      // retained snapshot or inherits this wait during creation.
      old_frontier = context->stream_wait_frontier;
      context->stream_wait_frontier = new_frontier;
      new_frontier = NULL;
      fan_out_wait = true;
    }
  }
  iree_slim_mutex_unlock(&context->mutex);

  if (iree_status_is_ok(status) && fan_out_wait) {
    const iree_hal_semaphore_list_t wait = {
        .count = 1,
        .semaphores = &recorded_point.semaphore,
        .payload_values = &recorded_point.value,
    };
    for (iree_host_size_t i = 0; i < snapshot.stream_count; ++i) {
      status = iree_status_join(
          status,
          iree_hal_streaming_stream_wait_semaphores(snapshot.streams[i], wait));
    }
  }
  if (iree_status_code(status) == IREE_STATUS_FAILED_PRECONDITION) {
    target_is_captured |=
        iree_hip_execution_context_invalidate_captures(&snapshot);
  }

  hipError_t result = hipSuccess;
  if (!iree_status_is_ok(status)) {
    if (target_is_captured) {
      iree_status_free(status);
      result = hipErrorStreamCaptureUnsupported;
    } else {
      result = iree_hip_execution_context_consume_status(status);
    }
  } else if (event_is_captured || target_is_captured) {
    result = hipErrorStreamCaptureUnsupported;
  }

  iree_hal_fence_release(new_frontier);
  iree_hal_fence_release(old_frontier);
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  iree_hip_execution_context_release_snapshot(&snapshot);
  iree_hip_execution_context_release(context);
  return result;
}

hipError_t iree_hip_execution_context_synchronize(
    hipExecutionCtx_t context_handle) {
  hipExecutionCtx_t context =
      iree_hip_execution_context_resolve_live(context_handle);
  if (!context) return hipErrorInvalidValue;

  iree_hip_execution_context_stream_snapshot_t snapshot = {0};
  iree_hal_semaphore_t* event_record_semaphore = NULL;
  uint64_t event_record_value = 0;
  iree_slim_mutex_lock(&context->mutex);
  iree_status_t status =
      iree_hip_execution_context_snapshot_locked(context, &snapshot);
  if (iree_status_is_ok(status) &&
      context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_RESOURCE_PARTITIONED &&
      context->event_record_timeline.pending_value > 0) {
    event_record_semaphore = context->event_record_timeline.semaphore;
    event_record_value = context->event_record_timeline.pending_value;
    iree_hal_semaphore_retain(event_record_semaphore);
  }
  iree_slim_mutex_unlock(&context->mutex);

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < snapshot.stream_count; ++i) {
      status = iree_status_join(
          status, iree_hal_streaming_stream_flush(snapshot.streams[i]));
    }
    for (iree_host_size_t i = 0; i < snapshot.stream_count; ++i) {
      status = iree_status_join(
          status,
          iree_hal_streaming_stream_synchronize_flushed(snapshot.streams[i]));
    }
    if (context->kind == IREE_HIP_EXECUTION_CONTEXT_KIND_PRIMARY) {
      status = iree_status_join(
          status, iree_hal_streaming_context_synchronize_event_records(
                      snapshot.common_context));
    } else if (event_record_semaphore) {
      status = iree_status_join(
          status, iree_hal_semaphore_wait(
                      event_record_semaphore, event_record_value,
                      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }

  const hipError_t result =
      iree_status_is_ok(status)
          ? hipSuccess
          : iree_hip_execution_context_consume_status(status);
  iree_hal_semaphore_release(event_record_semaphore);
  iree_hip_execution_context_release_snapshot(&snapshot);
  iree_hip_execution_context_release(context);
  return result;
}

hipError_t iree_hip_execution_context_get_resource(
    hipExecutionCtx_t context, hipDevResourceType type,
    hipDevResource* out_resource) {
  IREE_ASSERT_ARGUMENT(out_resource);
  if (type != hipDevResourceTypeSm) return hipErrorInvalidResourceType;
  hipExecutionCtx_t retained_context =
      iree_hip_execution_context_resolve_live(context);
  if (!retained_context) return hipErrorInvalidValue;
  const hipDevResource resource = retained_context->sm_resource;
  iree_hip_execution_context_release(retained_context);
  *out_resource = resource;
  return hipSuccess;
}

hipError_t iree_hip_execution_context_get_device(hipExecutionCtx_t context,
                                                 hipDevice_t* out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  hipExecutionCtx_t retained_context =
      iree_hip_execution_context_resolve_live(context);
  if (!retained_context) return hipErrorInvalidValue;
  const hipDevice_t device = retained_context->device_ordinal;
  iree_hip_execution_context_release(retained_context);
  *out_device = device;
  return hipSuccess;
}

hipError_t iree_hip_execution_context_get_id(
    hipExecutionCtx_t context, unsigned long long* out_context_id) {
  IREE_ASSERT_ARGUMENT(out_context_id);
  hipExecutionCtx_t retained_context =
      iree_hip_execution_context_resolve_live(context);
  if (!retained_context) return hipErrorInvalidValue;
  const unsigned long long context_id = retained_context->context_id;
  iree_hip_execution_context_release(retained_context);
  *out_context_id = context_id;
  return hipSuccess;
}

static hipError_t iree_hip_execution_context_prepare_reset_matching(
    bool reset_all, hipDevice_t device,
    iree_hip_execution_context_reset_t* out_reset) {
  IREE_ASSERT_ARGUMENT(out_reset);
  *out_reset = (iree_hip_execution_context_reset_t){0};
  iree_call_once(&iree_hip_execution_context_registry_once,
                 iree_hip_execution_context_registry_initialize);

  // Prepare a complete owning snapshot before mutating any handle. Allocation
  // failure therefore leaves registry visibility, streams, queues, and primary
  // retains unchanged.
  iree_host_size_t reset_count = 0;
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  for (hipExecutionCtx_t context = iree_hip_execution_context_registry.head;
       context; context = context->next_live_context) {
    if (!reset_all && context->device_ordinal != device) {
      continue;
    }
    if (IREE_UNLIKELY(
            !iree_host_size_checked_add(reset_count, 1, &reset_count))) {
      iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);
      return hipErrorOutOfMemory;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);

  size_t allocation_size = 0;
  hipExecutionCtx_t* reset_contexts = NULL;
  if (reset_count > 0) {
    if (!iree_host_size_checked_mul(reset_count, sizeof(reset_contexts[0]),
                                    &allocation_size)) {
      return hipErrorOutOfMemory;
    }
    iree_status_t allocation_status = iree_allocator_malloc(
        iree_allocator_system(), allocation_size, (void**)&reset_contexts);
    if (!iree_status_is_ok(allocation_status)) {
      iree_status_ignore(allocation_status);
      return hipErrorOutOfMemory;
    }
  }

  iree_host_size_t snapshot_count = 0;
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  for (hipExecutionCtx_t context = iree_hip_execution_context_registry.head;
       context; context = context->next_live_context) {
    if (!reset_all && context->device_ordinal != device) continue;
    IREE_ASSERT(snapshot_count < reset_count);
    iree_hip_execution_context_retain(context);
    reset_contexts[snapshot_count++] = context;
  }
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);
  IREE_ASSERT(snapshot_count == reset_count,
              "lifecycle writer must stabilize execution handles");

  out_reset->contexts = reset_contexts;
  out_reset->count = reset_count;
  return hipSuccess;
}

void iree_hip_execution_context_commit_reset(
    iree_hip_execution_context_reset_t* reset) {
  if (!reset) return;
  hipExecutionCtx_t* reset_contexts = reset->contexts;
  const iree_host_size_t reset_count = reset->count;
  // Dynamic queue destruction may synchronously enter a backend wait. Drain
  // every such release while the exact public handle is still registered as
  // the teardown ledger.
  for (iree_host_size_t i = 0; i < reset_count; ++i) {
    iree_hip_execution_context_deinitialize(reset_contexts[i],
                                            /*abort_captures=*/true);
  }

  // With preparation and potentially waiting teardown complete, atomically
  // consume only the exact pinned incarnations. Writer admission makes a
  // mismatch impossible; assert instead of restoring detached state.
  iree_slim_mutex_lock(&iree_hip_execution_context_registry.mutex);
  for (iree_host_size_t i = 0; i < reset_count; ++i) {
    hipExecutionCtx_t* link = &iree_hip_execution_context_registry.head;
    while (*link && *link != reset_contexts[i]) {
      link = &(*link)->next_live_context;
    }
    IREE_ASSERT(*link == reset_contexts[i],
                "prepared execution handle changed under lifecycle writer");
    if (*link == reset_contexts[i]) {
      *link = reset_contexts[i]->next_live_context;
      reset_contexts[i]->next_live_context = NULL;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_execution_context_registry.mutex);

  for (iree_host_size_t i = 0; i < reset_count; ++i) {
    iree_hip_vmm_test_notify_execution_context_taken(reset_contexts[i]);
    // Drop the transferred registry ownership and the preparation pin.
    iree_hip_execution_context_release(reset_contexts[i]);
    iree_hip_execution_context_release(reset_contexts[i]);
  }
  iree_allocator_free(iree_allocator_system(), reset_contexts);
  *reset = (iree_hip_execution_context_reset_t){0};
}

void iree_hip_execution_context_cancel_reset(
    iree_hip_execution_context_reset_t* reset) {
  if (!reset) return;
  for (size_t i = 0; i < reset->count; ++i) {
    iree_hip_execution_context_release(reset->contexts[i]);
  }
  iree_allocator_free(iree_allocator_system(), reset->contexts);
  *reset = (iree_hip_execution_context_reset_t){0};
}

hipError_t iree_hip_execution_context_prepare_reset_device(
    hipDevice_t device, iree_hip_execution_context_reset_t* out_reset) {
  return iree_hip_execution_context_prepare_reset_matching(
      /*reset_all=*/false, device, out_reset);
}

hipError_t iree_hip_execution_context_prepare_reset_all(
    iree_hip_execution_context_reset_t* out_reset) {
  return iree_hip_execution_context_prepare_reset_matching(
      /*reset_all=*/true, 0, out_reset);
}
