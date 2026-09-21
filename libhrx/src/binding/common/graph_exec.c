// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"
#include "common/internal.h"
#include "common/kernel_arguments.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_exec_t (instantiation)
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_graph_block_type_e {
  // iree_hal_queue_barrier
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  // iree_hal_queue_fill
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL,
  // iree_hal_queue_copy
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY,
  // iree_hal_queue_host_call
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
  // Event record node.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD,
  // Event wait node.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT,
  // iree_hal_queue_dispatch
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
  // iree_hal_queue_execute
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
  // Nested graph executable.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
} iree_hal_streaming_graph_block_type_t;

typedef void (*iree_hal_streaming_host_callback_t)(void* user_data);

#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
static iree_hal_streaming_graph_test_queue_submission_observer_t
    iree_hal_streaming_graph_test_queue_submission_observer = NULL;
static void* iree_hal_streaming_graph_test_queue_submission_observer_user_data =
    NULL;

void iree_hal_streaming_graph_test_set_queue_submission_observer(
    iree_hal_streaming_graph_test_queue_submission_observer_t observer,
    void* user_data) {
  iree_hal_streaming_graph_test_queue_submission_observer = observer;
  iree_hal_streaming_graph_test_queue_submission_observer_user_data = user_data;
}
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
typedef struct iree_hal_streaming_graph_barrier_block_attrs_t {
  iree_hal_queue_barrier_flags_t flags;
  // Exact one-node graph allocation barrier represented by this block, or
  // NULL. An accepted unmatched allocation transfers its independent wrapper
  // pin to the context allocation registry at this block's commit point.
  iree_hal_streaming_graph_node_t* source_mem_alloc_node;
  bool transfer_mem_alloc_on_accept;
} iree_hal_streaming_graph_barrier_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL
typedef struct iree_hal_streaming_graph_fill_block_attrs_t {
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint64_t pattern;
  iree_host_size_t pattern_length;
  iree_hal_fill_flags_t flags;
} iree_hal_streaming_graph_fill_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY
typedef struct iree_hal_streaming_graph_copy_block_attrs_t {
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  iree_hal_copy_flags_t flags;
} iree_hal_streaming_graph_copy_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
typedef struct iree_hal_streaming_graph_host_call_block_attrs_t {
  // Host callback invoked when the queued host call reaches this block.
  iree_hal_streaming_host_callback_t fn;
  // Callback that transfers terminal completion to asynchronous work.
  iree_hal_streaming_graph_deferred_host_call_fn_t deferred_fn;
  // User data passed to the host callback.
  void* user_data;
  // Persistent queue-host-call argument storage referenced by the device queue.
  uint64_t args[4];
  // Host-call submission flags.
  iree_hal_host_call_flags_t flags;
} iree_hal_streaming_graph_host_call_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD/WAIT
typedef struct iree_hal_streaming_graph_event_block_attrs_t {
  // Source graph node this block was instantiated from.
  iree_hal_streaming_graph_node_t* source_node;
  // Event handle to record or wait on.
  iree_hal_streaming_event_t* event;
} iree_hal_streaming_graph_event_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
typedef struct iree_hal_streaming_graph_dispatch_block_attrs_t {
  iree_hal_executable_t* executable;
  iree_host_size_t entry_point;
  iree_hal_dispatch_config_t config;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
} iree_hal_streaming_graph_dispatch_block_attrs_t;

// One immutable launch-time validation record copied from the exact kernel
// bytes that form an executable graph.
typedef struct iree_hal_streaming_graph_pointer_validation_t {
  iree_hal_streaming_parameter_info_t parameters;
  iree_const_byte_span_t constants;
  iree_hal_streaming_device_pointer_validator_t validator;
  void* user_data;
} iree_hal_streaming_graph_pointer_validation_t;

// One VMM capability required by a compiled copy/fill node. The executable
// retains no authority through this record: every instantiate, update, and
// launch resolves the numeric address in the participating context and
// requires the same never-reused device access-grant identity.
typedef struct iree_hal_streaming_graph_memory_validation_t {
  // Exact context whose device-scoped capability must be resolved. NULL uses
  // the graph launch context for ordinary copy/fill nodes.
  iree_hal_streaming_context_t* context;
  iree_hal_streaming_deviceptr_t device_ptr;
  iree_device_size_t size;
  iree_hal_memory_access_t required_access;
  uint64_t capability_id;
} iree_hal_streaming_graph_memory_validation_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
typedef struct iree_hal_streaming_graph_execute_block_attrs_t {
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_queue_execute_flags_t flags;
} iree_hal_streaming_graph_execute_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
typedef struct iree_hal_streaming_graph_child_graph_block_attrs_t {
  iree_hal_streaming_graph_exec_t* exec;
} iree_hal_streaming_graph_child_graph_block_attrs_t;

// Block-specific data stored at the end of the block allocation.
typedef union iree_hal_streaming_graph_block_attrs_t {
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
  iree_hal_streaming_graph_barrier_block_attrs_t barrier;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL
  iree_hal_streaming_graph_fill_block_attrs_t fill;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY
  iree_hal_streaming_graph_copy_block_attrs_t copy;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
  iree_hal_streaming_graph_host_call_block_attrs_t host_call;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD/WAIT
  iree_hal_streaming_graph_event_block_attrs_t event;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
  iree_hal_streaming_graph_dispatch_block_attrs_t dispatch;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
  iree_hal_streaming_graph_execute_block_attrs_t execute;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
  iree_hal_streaming_graph_child_graph_block_attrs_t child_graph;
} iree_hal_streaming_graph_block_attrs_t;

// Represents an atomically executable block of work in a graph.
typedef struct iree_hal_streaming_graph_block_t {
  iree_hal_streaming_graph_block_type_t type;

  // First node in sorted array.
  uint32_t node_start_index;
  // Number of nodes in this block.
  uint32_t node_count;

  // Semaphore synchronization.
  uint16_t wait_semaphore_count;
  uint16_t signal_semaphore_count;

  // Variable-length data follows:
  // - uint16_t wait_semaphore_indices[wait_semaphore_count]
  // - uint32_t wait_payload_deltas[wait_semaphore_count]
  // - uint16_t signal_semaphore_indices[signal_semaphore_count]
  // - uint32_t signal_payload_deltas[signal_semaphore_count]
  // - iree_hal_streaming_graph_block_attrs_t attrs (based on type)
} iree_hal_streaming_graph_block_t;

// Pointers to all variable-length arrays in a block.
typedef struct iree_hal_streaming_graph_block_ptrs_t {
  uint16_t* wait_semaphore_indices;
  uint32_t* wait_payload_deltas;
  uint16_t* signal_semaphore_indices;
  uint32_t* signal_payload_deltas;
  iree_hal_streaming_graph_block_attrs_t* attrs;
} iree_hal_streaming_graph_block_ptrs_t;

typedef struct iree_hal_streaming_graph_exec_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  iree_hal_streaming_context_t* context;  // retained
  iree_hal_streaming_graph_t* graph;      // retained
  // True after the public HIP graph-exec handle has been destroyed.
  bool is_destroyed;
  // True while destruction has exclusive ownership but has not yet proven the
  // accepted launch frontier quiescent. Cleared on any retryable wait failure.
  bool destroy_pending;

  // Arena allocator used for block allocations and inlined data.
  iree_arena_allocator_t arena_allocator;

  // Immutable block list created during instantiate.
  iree_hal_streaming_graph_block_t** blocks;
  uint32_t block_count;
  // True when |blocks|, or the blocks of a child-graph executable they launch,
  // hold an event record. Every such record names an event of |context|, so a
  // launch on a stream of any other context would have every one of its
  // records refused.
  bool records_events;
  // Number of graph nodes present when this executable was instantiated.
  iree_host_size_t instantiated_node_count;
  // Number of HIP-visible graph nodes present at instantiation/update time.
  iree_host_size_t instantiated_visible_node_count;
  // Exact formal-pointer metadata and native bytes for compiled kernel nodes.
  iree_hal_streaming_graph_pointer_validation_t* pointer_validations;
  iree_host_size_t pointer_validation_count;
  iree_host_size_t pointer_validation_capacity;
  // VMM copy/fill address and access-generation metadata. Buffer wrappers
  // retained by graph templates and command buffers are inert storage only.
  iree_hal_streaming_graph_memory_validation_t* memory_validations;
  iree_host_size_t memory_validation_count;
  iree_host_size_t memory_validation_capacity;

  // Semaphore pool for internal synchronization.
  uint32_t semaphore_count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* semaphore_base_values;

  // One completion timeline per block. Every accepted block advances its own
  // timeline so a later synchronous rejection still leaves an exact terminal
  // frontier for the accepted prefix.
  iree_hal_semaphore_t** block_completion_semaphores;
  // Last value assigned on each block completion timeline.
  uint64_t* block_completion_base_values;
  // Scratch arrays large enough to collect every block completion in this
  // executable and all recursively compiled children during one launch.
  iree_hal_semaphore_t** launch_frontier_semaphores;
  uint64_t* launch_frontier_values;
  iree_host_size_t launch_frontier_capacity;

  // Resource set for automatic cleanup.
  iree_hal_resource_set_t* resource_set;

  // Graph-memory accounting entries retained while this exec is alive.
  struct iree_hal_streaming_graph_memory_contribution_t*
      graph_memory_contributions;
  // Number of entries in |graph_memory_contributions|.
  uint32_t graph_memory_contribution_count;
  // True when an unmatched graph alloc node remains live after launch.
  bool has_unfreed_graph_alloc_nodes;
  // Number of successful launches of this exec.
  uint64_t launch_count;

  // True if this exec contributes to graph memory-node instantiation limits.
  bool uses_graph_memory_nodes;

  // Stream retained while the most recent launch is in flight.
  iree_hal_streaming_stream_t* active_launch_stream;
  // Timeline value signaled by |active_launch_stream|.
  uint64_t active_launch_value;
  // Accepted block completions being synchronously drained because submission
  // of their aggregate stream closure failed.
  iree_host_size_t active_frontier_count;
  // True while one waiter owns the drain of |active_frontier_count|.
  iree_atomic_int32_t active_frontier_draining;
  // Wakes other executable operations after the frontier owner finishes.
  iree_notification_t active_frontier_notification;

  unsigned long long flags;

  // Mutex needed for launch/update.
  iree_slim_mutex_t mutex;
} iree_hal_streaming_graph_exec_t;

typedef struct iree_hal_streaming_graph_exec_deferred_cleanup_t {
  iree_hal_streaming_graph_exec_t compiled_state;
  bool has_compiled_state;
  iree_hal_streaming_graph_t* graph;
  bool release_graph_memory_slot;
} iree_hal_streaming_graph_exec_deferred_cleanup_t;

typedef struct iree_hal_streaming_graph_memory_contribution_t {
  // Allocation size represented by this contribution.
  iree_device_size_t size;
  // Number of same-sized allocations represented by this contribution.
  uint32_t count;
  // True when this contribution can be shared by graph alloc/free pairs.
  bool reusable;
} iree_hal_streaming_graph_memory_contribution_t;

static iree_status_t iree_hal_streaming_graph_record_memcpy_node(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_streaming_graph_memcpy_node_attrs_t* attrs) {
  const iree_device_size_t width =
      attrs->execution_extent_width
          ? attrs->execution_extent_width
          : (attrs->hip_extent_width ? attrs->hip_extent_width : attrs->size);
  const iree_device_size_t height =
      attrs->execution_extent_height
          ? attrs->execution_extent_height
          : (attrs->hip_extent_height ? attrs->hip_extent_height : 1);
  const iree_device_size_t depth =
      attrs->execution_extent_depth
          ? attrs->execution_extent_depth
          : (attrs->hip_extent_depth ? attrs->hip_extent_depth : 1);
  if (width == 0 || height == 0 || depth == 0) return iree_ok_status();

  const iree_device_size_t src_pitch =
      attrs->execution_src_pitch
          ? attrs->execution_src_pitch
          : (attrs->hip_src_pitch ? attrs->hip_src_pitch : width);
  const iree_device_size_t dst_pitch =
      attrs->execution_dst_pitch
          ? attrs->execution_dst_pitch
          : (attrs->hip_dst_pitch ? attrs->hip_dst_pitch : width);
  const iree_device_size_t src_rows_per_slice =
      attrs->execution_src_ysize
          ? attrs->execution_src_ysize
          : (attrs->hip_src_ysize ? attrs->hip_src_ysize : height);
  const iree_device_size_t dst_rows_per_slice =
      attrs->execution_dst_ysize
          ? attrs->execution_dst_ysize
          : (attrs->hip_dst_ysize ? attrs->hip_dst_ysize : height);

  iree_device_size_t total_size = 0;
  iree_device_size_t src_slice_pitch = 0;
  iree_device_size_t dst_slice_pitch = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(width, height, &total_size) ||
          !iree_device_size_checked_mul(total_size, depth, &total_size) ||
          !iree_device_size_checked_mul(src_pitch, src_rows_per_slice,
                                        &src_slice_pitch) ||
          !iree_device_size_checked_mul(dst_pitch, dst_rows_per_slice,
                                        &dst_slice_pitch))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "graph memcpy node geometry overflows");
  }

  const bool compact_layout = src_pitch == width && dst_pitch == width &&
                              src_rows_per_slice == height &&
                              dst_rows_per_slice == height;
  if (compact_layout) {
    if (attrs->src_ref.buffer == attrs->dst_ref.buffer &&
        attrs->src_ref.offset == attrs->dst_ref.offset) {
      return iree_ok_status();
    }
    return iree_hal_command_buffer_copy_buffer(
        command_buffer,
        iree_hal_streaming_convert_range_buffer_ref(attrs->src_ref, total_size),
        iree_hal_streaming_convert_range_buffer_ref(attrs->dst_ref, total_size),
        attrs->flags);
  }

  for (iree_device_size_t z = 0; z < depth; ++z) {
    iree_device_size_t src_slice_offset = 0;
    iree_device_size_t dst_slice_offset = 0;
    if (IREE_UNLIKELY(!iree_device_size_checked_mul(z, src_slice_pitch,
                                                    &src_slice_offset) ||
                      !iree_device_size_checked_mul(z, dst_slice_pitch,
                                                    &dst_slice_offset))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "graph memcpy slice offset overflows");
    }
    for (iree_device_size_t y = 0; y < height; ++y) {
      iree_device_size_t src_row_offset = 0;
      iree_device_size_t dst_row_offset = 0;
      iree_device_size_t src_offset = 0;
      iree_device_size_t dst_offset = 0;
      if (IREE_UNLIKELY(
              !iree_device_size_checked_mul(y, src_pitch, &src_row_offset) ||
              !iree_device_size_checked_mul(y, dst_pitch, &dst_row_offset) ||
              !iree_device_size_checked_add(src_slice_offset, src_row_offset,
                                            &src_offset) ||
              !iree_device_size_checked_add(dst_slice_offset, dst_row_offset,
                                            &dst_offset) ||
              !iree_device_size_checked_add(attrs->src_ref.offset, src_offset,
                                            &src_offset) ||
              !iree_device_size_checked_add(attrs->dst_ref.offset, dst_offset,
                                            &dst_offset))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "graph memcpy row offset overflows");
      }
      if (attrs->src_ref.buffer == attrs->dst_ref.buffer &&
          src_offset == dst_offset) {
        continue;
      }
      iree_hal_streaming_buffer_ref_t src_ref = attrs->src_ref;
      src_ref.offset = src_offset;
      iree_hal_streaming_buffer_ref_t dst_ref = attrs->dst_ref;
      dst_ref.offset = dst_offset;
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_copy_buffer(
          command_buffer,
          iree_hal_streaming_convert_range_buffer_ref(src_ref, width),
          iree_hal_streaming_convert_range_buffer_ref(dst_ref, width),
          attrs->flags));
    }
  }
  return iree_ok_status();
}

static inline void iree_hal_streaming_graph_block_get_ptrs(
    iree_hal_streaming_graph_block_t* block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs);
static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec);
static void iree_hal_streaming_graph_exec_release_child_blocks(
    iree_hal_streaming_graph_exec_t* exec);
static void iree_hal_streaming_graph_exec_deinitialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec);
static iree_status_t
iree_hal_streaming_graph_exec_wait_for_active_launch_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_active_launch_wait_callback_t wait_callback,
    void* wait_callback_user_data, iree_status_t* out_execution_status,
    bool fail_if_destroyed);
static iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template_locked(
    iree_hal_streaming_graph_exec_state_guard_t* guard);
static iree_host_size_t iree_hal_streaming_graph_visible_node_count(
    const iree_hal_streaming_graph_t* graph);

static void iree_hal_streaming_graph_normalize_compound_enabled_state(
    iree_hal_streaming_graph_t* graph) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (!node->compound_owner_node) continue;
      if (node->compound_owner_node->flags &
          IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED) {
        node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED;
      } else {
        node->flags &= ~IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED;
      }
    }
  }
}

static void iree_hal_streaming_graph_memory_add_high_water(
    iree_hal_streaming_device_t* device) {
  device->graph_memory_used_high = iree_max(device->graph_memory_used_high,
                                            device->graph_memory_used_current);
  device->graph_memory_reserved_high =
      iree_max(device->graph_memory_reserved_high,
               device->graph_memory_reserved_current);
}

static iree_hal_streaming_graph_memory_size_entry_t*
iree_hal_streaming_graph_memory_find_reusable_size_entry(
    iree_hal_streaming_device_t* device, iree_device_size_t size,
    iree_hal_streaming_graph_memory_size_entry_t*** out_previous_next) {
  iree_hal_streaming_graph_memory_size_entry_t** previous_next =
      &device->graph_memory_reusable_size_entries;
  while (*previous_next) {
    if ((*previous_next)->size == size) {
      if (out_previous_next) *out_previous_next = previous_next;
      return *previous_next;
    }
    previous_next = &(*previous_next)->next;
  }
  if (out_previous_next) *out_previous_next = previous_next;
  return NULL;
}

static iree_status_t iree_hal_streaming_graph_memory_add_contribution(
    iree_hal_streaming_graph_memory_contribution_t* contributions,
    uint32_t* contribution_count, uint32_t contribution_capacity,
    iree_device_size_t size, bool reusable) {
  if (reusable) {
    for (uint32_t i = 0; i < *contribution_count; ++i) {
      if (contributions[i].reusable && contributions[i].size == size) {
        return iree_ok_status();
      }
    }
  } else {
    for (uint32_t i = 0; i < *contribution_count; ++i) {
      if (!contributions[i].reusable && contributions[i].size == size) {
        ++contributions[i].count;
        return iree_ok_status();
      }
    }
  }
  if (*contribution_count >= contribution_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph memory contribution table overflow");
  }
  contributions[*contribution_count] =
      (iree_hal_streaming_graph_memory_contribution_t){
          .size = size,
          .count = 1,
          .reusable = reusable,
      };
  ++*contribution_count;
  return iree_ok_status();
}

static bool iree_hal_streaming_graph_has_free_node_for_pointer(
    iree_hal_streaming_graph_t* graph, void* dptr) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE &&
          node->attrs.mem_free.dptr == dptr) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t iree_hal_streaming_graph_memory_build_contributions(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec->graph->has_graph_memory_nodes || exec->graph->node_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t contribution_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          exec->graph->node_count, sizeof(*exec->graph_memory_contributions),
          &contribution_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph memory contribution size overflow");
  }
  iree_hal_streaming_graph_memory_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &exec->arena_allocator, contribution_size, (void**)&contributions));

  uint32_t contribution_count = 0;
  const uint32_t contribution_capacity = (uint32_t)exec->graph->node_count;
  for (iree_hal_streaming_node_block_t* block = exec->graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC ||
          node->attrs.mem_alloc.bytesize == 0) {
        continue;
      }
      const bool has_matching_free =
          iree_hal_streaming_graph_has_free_node_for_pointer(
              exec->graph, node->attrs.mem_alloc.dptr);
      exec->has_unfreed_graph_alloc_nodes |= !has_matching_free;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_memory_add_contribution(
          contributions, &contribution_count, contribution_capacity,
          node->attrs.mem_alloc.bytesize, has_matching_free));
    }
  }
  exec->graph_memory_contributions = contributions;
  exec->graph_memory_contribution_count = contribution_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_memory_retain_exec(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_streaming_context_t* context = exec->context;
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  iree_status_t status = iree_ok_status();
  uint32_t retained_contribution_count = 0;
  for (uint32_t i = 0;
       iree_status_is_ok(status) && i < exec->graph_memory_contribution_count;
       ++i) {
    iree_hal_streaming_graph_memory_contribution_t* contribution =
        &exec->graph_memory_contributions[i];
    const uint64_t bytes = (uint64_t)contribution->size * contribution->count;
    if (contribution->reusable) {
      iree_hal_streaming_graph_memory_size_entry_t* entry =
          iree_hal_streaming_graph_memory_find_reusable_size_entry(
              device, contribution->size, NULL);
      if (entry) {
        if (entry->reference_count == 0) {
          device->graph_memory_used_current += bytes;
          iree_hal_streaming_graph_memory_add_high_water(device);
        }
        ++entry->reference_count;
        retained_contribution_count = i + 1;
      } else {
        status = iree_allocator_malloc(context->host_allocator, sizeof(*entry),
                                       (void**)&entry);
        if (iree_status_is_ok(status)) {
          entry->next = device->graph_memory_reusable_size_entries;
          entry->size = contribution->size;
          entry->reference_count = 1;
          device->graph_memory_reusable_size_entries = entry;
          device->graph_memory_used_current += bytes;
          device->graph_memory_reserved_current += bytes;
          iree_hal_streaming_graph_memory_add_high_water(device);
          retained_contribution_count = i + 1;
        }
      }
    } else {
      device->graph_memory_used_current += bytes;
      device->graph_memory_reserved_current += bytes;
      iree_hal_streaming_graph_memory_add_high_water(device);
      retained_contribution_count = i + 1;
    }
  }
  if (!iree_status_is_ok(status)) {
    exec->graph_memory_contribution_count = retained_contribution_count;
  }
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return status;
}

static void iree_hal_streaming_graph_memory_release_exec(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_streaming_context_t* context = exec->context;
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  for (uint32_t i = 0; i < exec->graph_memory_contribution_count; ++i) {
    iree_hal_streaming_graph_memory_contribution_t* contribution =
        &exec->graph_memory_contributions[i];
    const uint64_t bytes = (uint64_t)contribution->size * contribution->count;
    if (contribution->reusable) {
      iree_hal_streaming_graph_memory_size_entry_t* entry =
          iree_hal_streaming_graph_memory_find_reusable_size_entry(
              device, contribution->size, NULL);
      if (entry && entry->reference_count > 1) {
        --entry->reference_count;
      } else if (entry && entry->reference_count == 1) {
        entry->reference_count = 0;
        device->graph_memory_used_current -=
            iree_min(device->graph_memory_used_current, bytes);
      }
    } else {
      device->graph_memory_used_current -=
          iree_min(device->graph_memory_used_current, bytes);
      device->graph_memory_reserved_current -=
          iree_min(device->graph_memory_reserved_current, bytes);
    }
  }
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
}

uint64_t iree_hal_streaming_graph_memory_used_current(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_used_current;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

uint64_t iree_hal_streaming_graph_memory_used_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_used_high;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

uint64_t iree_hal_streaming_graph_memory_reserved_current(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_reserved_current;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

uint64_t iree_hal_streaming_graph_memory_reserved_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_reserved_high;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

void iree_hal_streaming_graph_memory_reset_used_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  device->graph_memory_used_high = 0;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
}

void iree_hal_streaming_graph_memory_reset_reserved_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  device->graph_memory_reserved_high = 0;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
}

void iree_hal_streaming_graph_memory_trim(iree_hal_streaming_device_t* device) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  iree_allocator_t host_allocator = device_registry
                                        ? device_registry->host_allocator
                                        : iree_allocator_system();
  iree_hal_streaming_graph_memory_size_entry_t* entry = NULL;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  iree_hal_streaming_graph_memory_size_entry_t** previous_next =
      &device->graph_memory_reusable_size_entries;
  while (*previous_next) {
    iree_hal_streaming_graph_memory_size_entry_t* current_entry =
        *previous_next;
    if (current_entry->reference_count > 0) {
      previous_next = &current_entry->next;
      continue;
    }
    *previous_next = current_entry->next;
    device->graph_memory_reserved_current -= iree_min(
        device->graph_memory_reserved_current, (uint64_t)current_entry->size);
    current_entry->next = entry;
    entry = current_entry;
  }
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  while (entry) {
    iree_hal_streaming_graph_memory_size_entry_t* next_entry = entry->next;
    iree_allocator_free(host_allocator, entry);
    entry = next_entry;
  }
}

static iree_status_t iree_hal_streaming_graph_memory_acquire_exec_slot(
    iree_hal_streaming_graph_t* graph) {
  if (!graph->has_graph_memory_nodes) return iree_ok_status();
  graph =
      graph->graph_memory_owner_graph ? graph->graph_memory_owner_graph : graph;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&graph->graph_memory_state_mutex);
  if (graph->has_transferred_unfreed_allocation) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "graph has a live allocation transferred by an earlier launch");
  } else if (graph->active_graph_memory_exec_count != 0) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "graphs with memory allocation nodes support one live executable");
  } else {
    ++graph->active_graph_memory_exec_count;
  }
  iree_slim_mutex_unlock(&graph->graph_memory_state_mutex);
  return status;
}

static void iree_hal_streaming_graph_memory_release_exec_slot(
    iree_hal_streaming_graph_t* graph) {
  if (!graph || !graph->has_graph_memory_nodes) return;
  graph =
      graph->graph_memory_owner_graph ? graph->graph_memory_owner_graph : graph;
  iree_slim_mutex_lock(&graph->graph_memory_state_mutex);
  IREE_ASSERT(graph->active_graph_memory_exec_count > 0);
  --graph->active_graph_memory_exec_count;
  iree_slim_mutex_unlock(&graph->graph_memory_state_mutex);
}

// Internal: Create an exec object (called by graph.c).
iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*exec), (void**)&exec));

  iree_atomic_ref_count_init(&exec->ref_count);
  exec->host_allocator = host_allocator;
  exec->context = context;
  iree_hal_streaming_context_retain(exec->context);
  exec->graph = graph;
  iree_hal_streaming_graph_retain(exec->graph);
  exec->is_destroyed = false;
  exec->destroy_pending = false;
  iree_arena_initialize(&context->device_entry->block_pool,
                        &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->instantiated_node_count = 0;
  exec->instantiated_visible_node_count = 0;
  exec->pointer_validations = NULL;
  exec->pointer_validation_count = 0;
  exec->pointer_validation_capacity = 0;
  exec->memory_validations = NULL;
  exec->memory_validation_count = 0;
  exec->memory_validation_capacity = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->block_completion_semaphores = NULL;
  exec->block_completion_base_values = NULL;
  exec->launch_frontier_semaphores = NULL;
  exec->launch_frontier_values = NULL;
  exec->launch_frontier_capacity = 0;
  exec->resource_set = NULL;
  exec->graph_memory_contributions = NULL;
  exec->graph_memory_contribution_count = 0;
  exec->has_unfreed_graph_alloc_nodes = false;
  exec->launch_count = 0;
  exec->uses_graph_memory_nodes = false;
  exec->active_launch_stream = NULL;
  exec->active_launch_value = 0;
  exec->active_frontier_count = 0;
  iree_atomic_store(&exec->active_frontier_draining, 0,
                    iree_memory_order_relaxed);
  iree_notification_initialize(&exec->active_frontier_notification);
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->mutex);

  // Create resource set for automatic cleanup.
  iree_status_t status =
      iree_hal_streaming_graph_memory_acquire_exec_slot(graph);
  if (iree_status_is_ok(status)) {
    exec->uses_graph_memory_nodes = graph->has_graph_memory_nodes;
    status = iree_hal_resource_set_allocate(&context->device_entry->block_pool,
                                            &exec->resource_set);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_build_contributions(exec);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_retain_exec(exec);
  }

  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    iree_hal_streaming_graph_exec_destroy(exec);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_deinitialize_compiled_state(exec);

  if (exec->uses_graph_memory_nodes) {
    iree_hal_streaming_graph_memory_release_exec_slot(exec->graph);
  }
  iree_hal_streaming_stream_release(exec->active_launch_stream);
  exec->active_launch_stream = NULL;
  exec->active_launch_value = 0;
  exec->active_frontier_count = 0;

  iree_hal_streaming_graph_release(exec->graph);
  iree_hal_streaming_context_release(exec->context);
  iree_notification_deinitialize(&exec->active_frontier_notification);
  iree_slim_mutex_deinitialize(&exec->mutex);

  iree_allocator_t host_allocator = exec->host_allocator;
  iree_allocator_free(host_allocator, exec);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_streaming_graph_exec_release_child_blocks(
    iree_hal_streaming_graph_exec_t* exec) {
  for (uint32_t i = 0; exec->blocks && i < exec->block_count; ++i) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[i];
    if (!block) continue;
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);

    switch (block->type) {
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH:
        iree_hal_streaming_graph_exec_release(ptrs.attrs->child_graph.exec);
        ptrs.attrs->child_graph.exec = NULL;
        break;
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
        iree_hal_streaming_event_release(ptrs.attrs->event.event);
        ptrs.attrs->event.event = NULL;
        break;
      default:
        break;
    }
  }
}

static void iree_hal_streaming_graph_exec_initialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_arena_initialize(&exec->context->device_entry->block_pool,
                        &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->instantiated_node_count = 0;
  exec->instantiated_visible_node_count = 0;
  exec->pointer_validations = NULL;
  exec->pointer_validation_count = 0;
  exec->pointer_validation_capacity = 0;
  exec->memory_validations = NULL;
  exec->memory_validation_count = 0;
  exec->memory_validation_capacity = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->block_completion_semaphores = NULL;
  exec->block_completion_base_values = NULL;
  exec->launch_frontier_semaphores = NULL;
  exec->launch_frontier_values = NULL;
  exec->launch_frontier_capacity = 0;
  exec->resource_set = NULL;
  exec->graph_memory_contributions = NULL;
  exec->graph_memory_contribution_count = 0;
  exec->has_unfreed_graph_alloc_nodes = false;
}

static void iree_hal_streaming_graph_exec_deinitialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_streaming_graph_exec_release_child_blocks(exec);
  if (exec->resource_set) {
    iree_hal_resource_set_free(exec->resource_set);
    exec->resource_set = NULL;
  }
  if (exec->graph_memory_contribution_count > 0) {
    iree_hal_streaming_graph_memory_release_exec(exec);
  }
  iree_arena_deinitialize(&exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->instantiated_node_count = 0;
  exec->instantiated_visible_node_count = 0;
  exec->pointer_validations = NULL;
  exec->pointer_validation_count = 0;
  exec->pointer_validation_capacity = 0;
  exec->memory_validations = NULL;
  exec->memory_validation_count = 0;
  exec->memory_validation_capacity = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->block_completion_semaphores = NULL;
  exec->block_completion_base_values = NULL;
  exec->launch_frontier_semaphores = NULL;
  exec->launch_frontier_values = NULL;
  exec->launch_frontier_capacity = 0;
  exec->graph_memory_contributions = NULL;
  exec->graph_memory_contribution_count = 0;
  exec->has_unfreed_graph_alloc_nodes = false;
}

static void iree_hal_streaming_graph_exec_move_compiled_state(
    iree_hal_streaming_graph_exec_t* target,
    iree_hal_streaming_graph_exec_t* source) {
  target->arena_allocator = source->arena_allocator;
  target->blocks = source->blocks;
  target->block_count = source->block_count;
  target->records_events = source->records_events;
  target->instantiated_node_count = source->instantiated_node_count;
  target->instantiated_visible_node_count =
      source->instantiated_visible_node_count;
  target->pointer_validations = source->pointer_validations;
  target->pointer_validation_count = source->pointer_validation_count;
  target->pointer_validation_capacity = source->pointer_validation_capacity;
  target->memory_validations = source->memory_validations;
  target->memory_validation_count = source->memory_validation_count;
  target->memory_validation_capacity = source->memory_validation_capacity;
  target->semaphores = source->semaphores;
  target->semaphore_count = source->semaphore_count;
  target->semaphore_base_values = source->semaphore_base_values;
  target->block_completion_semaphores = source->block_completion_semaphores;
  target->block_completion_base_values = source->block_completion_base_values;
  target->launch_frontier_semaphores = source->launch_frontier_semaphores;
  target->launch_frontier_values = source->launch_frontier_values;
  target->launch_frontier_capacity = source->launch_frontier_capacity;
  target->resource_set = source->resource_set;
  target->graph_memory_contributions = source->graph_memory_contributions;
  target->graph_memory_contribution_count =
      source->graph_memory_contribution_count;
  target->has_unfreed_graph_alloc_nodes = source->has_unfreed_graph_alloc_nodes;

  source->blocks = NULL;
  source->block_count = 0;
  source->records_events = false;
  source->instantiated_node_count = 0;
  source->instantiated_visible_node_count = 0;
  source->pointer_validations = NULL;
  source->pointer_validation_count = 0;
  source->pointer_validation_capacity = 0;
  source->memory_validations = NULL;
  source->memory_validation_count = 0;
  source->memory_validation_capacity = 0;
  source->semaphores = NULL;
  source->semaphore_count = 0;
  source->semaphore_base_values = NULL;
  source->block_completion_semaphores = NULL;
  source->block_completion_base_values = NULL;
  source->launch_frontier_semaphores = NULL;
  source->launch_frontier_values = NULL;
  source->launch_frontier_capacity = 0;
  source->resource_set = NULL;
  source->graph_memory_contributions = NULL;
  source->graph_memory_contribution_count = 0;
  source->has_unfreed_graph_alloc_nodes = false;
}

void iree_hal_streaming_graph_exec_retain(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec) {
    iree_atomic_ref_count_inc(&exec->ref_count);
  }
}

void iree_hal_streaming_graph_exec_release(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec && iree_atomic_ref_count_dec(&exec->ref_count) == 1) {
    iree_hal_streaming_graph_exec_destroy(exec);
  }
}

bool iree_hal_streaming_graph_exec_try_retain_live(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec) return false;
  iree_slim_mutex_lock(&exec->mutex);
  const bool is_live = !exec->is_destroyed && !exec->destroy_pending;
  if (is_live) {
    iree_hal_streaming_graph_exec_retain(exec);
  }
  iree_slim_mutex_unlock(&exec->mutex);
  return is_live;
}

// Clears active-launch stream ownership throughout one compiled executable
// tree after the lifecycle writer has quiesced its exact context. Blocks and
// child exec references are immutable while the root is closing, so recursion
// needs no allocation and every stream release happens before registry take.
static void iree_hal_streaming_graph_exec_clear_active_launch_streams_quiesced(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_streaming_stream_t* active_stream = NULL;
  iree_slim_mutex_lock(&exec->mutex);
  active_stream = exec->active_launch_stream;
  exec->active_launch_stream = NULL;
  exec->active_launch_value = 0;
  exec->active_frontier_count = 0;
  iree_atomic_store(&exec->active_frontier_draining, 0,
                    iree_memory_order_release);
  iree_slim_mutex_unlock(&exec->mutex);

  for (uint32_t i = 0; exec->blocks && i < exec->block_count; ++i) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[i];
    if (!block ||
        block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      continue;
    }
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);
    if (ptrs.attrs->child_graph.exec) {
      iree_hal_streaming_graph_exec_clear_active_launch_streams_quiesced(
          ptrs.attrs->child_graph.exec);
    }
  }

  iree_hal_streaming_stream_release(active_stream);
}

static iree_status_t iree_hal_streaming_graph_exec_begin_destroy(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed || exec->destroy_pending) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  exec->destroy_pending = true;
  iree_slim_mutex_unlock(&exec->mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_destroy_handle(
    iree_hal_streaming_graph_exec_t* exec) {
  return iree_hal_streaming_graph_exec_destroy_handle_with_wait_callback(
      exec, NULL, NULL);
}

iree_status_t iree_hal_streaming_graph_exec_destroy_handle_with_wait_callback(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_active_launch_wait_callback_t wait_callback,
    void* wait_callback_user_data) {
  if (!exec) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_begin_destroy(exec));
  // Every accepted launch publishes either a stream closure plus its exact
  // fallback frontier or only that frontier. Drain those points directly:
  // synchronizing the whole context can report a sticky stream failure before
  // proving the executable's accepted prefix terminal, and would then leave
  // the public handle permanently indestructible. The execution error remains
  // observable on its stream; destruction consumes only object ownership.
  iree_slim_mutex_lock(&exec->mutex);
  iree_status_t execution_status = iree_ok_status();
  iree_status_t wait_status =
      iree_hal_streaming_graph_exec_wait_for_active_launch_locked(
          exec, wait_callback, wait_callback_user_data, &execution_status,
          /*fail_if_destroyed=*/false);
  const bool launch_is_quiescent =
      !exec->active_launch_stream && exec->active_frontier_count == 0 &&
      iree_atomic_load(&exec->active_frontier_draining,
                       iree_memory_order_acquire) == 0;
  if (launch_is_quiescent) {
    IREE_ASSERT(exec->destroy_pending);
    exec->is_destroyed = true;
  }
  exec->destroy_pending = false;
  iree_slim_mutex_unlock(&exec->mutex);
  if (!launch_is_quiescent) {
    iree_status_ignore(execution_status);
    if (iree_status_is_ok(wait_status)) {
      wait_status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "graph exec wait returned with a live accepted frontier");
    }
    return wait_status;
  }
  // Terminal execution failures remain observable on their stream but prove
  // that the accepted executable frontier can no longer access its snapshot.
  // Only the state above, never the status code alone, permits destruction.
  iree_status_ignore(execution_status);
  IREE_ASSERT(iree_status_is_ok(wait_status));
  iree_status_ignore(wait_status);
  iree_hal_streaming_graph_exec_release(exec);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_destroy_handle_quiesced(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_begin_destroy(exec));
  IREE_ASSERT(iree_hal_streaming_context_is_teardown_certified(exec->context),
              "no-wait graph-exec destroy requires certified context");
  iree_hal_streaming_graph_exec_clear_active_launch_streams_quiesced(exec);
  iree_slim_mutex_lock(&exec->mutex);
  IREE_ASSERT(exec->destroy_pending);
  exec->is_destroyed = true;
  exec->destroy_pending = false;
  iree_slim_mutex_unlock(&exec->mutex);
  iree_hal_streaming_graph_exec_release(exec);
  return iree_ok_status();
}

iree_hal_streaming_graph_instantiate_flags_t
iree_hal_streaming_graph_exec_flags(iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  if (!iree_status_is_ok(
          iree_hal_streaming_graph_exec_state_begin(exec, &guard))) {
    return IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE;
  }
  const iree_hal_streaming_graph_instantiate_flags_t flags =
      iree_hal_streaming_graph_exec_state_flags(&guard);
  iree_hal_streaming_graph_exec_state_end(&guard);
  return flags;
}

iree_hal_streaming_context_t* iree_hal_streaming_graph_exec_retain_context(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec) return NULL;
  iree_hal_streaming_context_retain(exec->context);
  return exec->context;
}

static iree_status_t iree_hal_streaming_graph_add_user_callback_capacity(
    const iree_hal_streaming_graph_t* graph, iree_host_size_t* inout_capacity) {
  if (!graph) return iree_ok_status();
  for (const iree_hal_streaming_graph_user_object_ref_t* ref =
           graph->user_object_refs;
       ref; ref = ref->next) {
    if (ref->count == 0) continue;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_add(*inout_capacity, 1, inout_capacity))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph user callback capacity overflow");
    }
  }
  for (const iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      const iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH &&
          node->attrs.child_graph.graph) {
        IREE_RETURN_IF_ERROR(
            iree_hal_streaming_graph_add_user_callback_capacity(
                node->attrs.child_graph.graph, inout_capacity));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_add_user_callback_capacity(
    iree_hal_streaming_graph_exec_t* exec, iree_host_size_t* inout_capacity) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(inout_capacity);
  // The lifecycle writer has drained every operation that could update this
  // executable. Its graph and recursively compiled child blocks are therefore
  // immutable for the duration of this allocation-free capacity walk.
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_user_callback_capacity(
      exec->graph, inout_capacity));
  for (uint32_t i = 0; exec->blocks && i < exec->block_count; ++i) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[i];
    if (!block ||
        block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      continue;
    }
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);
    if (ptrs.attrs->child_graph.exec) {
      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_graph_exec_add_user_callback_capacity(
              ptrs.attrs->child_graph.exec, inout_capacity));
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_state_begin(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_state_guard_t* out_guard) {
  IREE_ASSERT_ARGUMENT(out_guard);
  *out_guard = (iree_hal_streaming_graph_exec_state_guard_t){0};
  if (!exec) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed || exec->destroy_pending) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  out_guard->exec = exec;
  return iree_ok_status();
}

bool iree_hal_streaming_graph_exec_state_try_begin(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_state_guard_t* out_guard) {
  IREE_ASSERT_ARGUMENT(out_guard);
  *out_guard = (iree_hal_streaming_graph_exec_state_guard_t){0};
  if (!exec || !iree_slim_mutex_try_lock(&exec->mutex)) return false;
  if (exec->is_destroyed || exec->destroy_pending) {
    iree_slim_mutex_unlock(&exec->mutex);
    return false;
  }
  out_guard->exec = exec;
  return true;
}

iree_status_t iree_hal_streaming_graph_exec_rebuild_state_begin(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_active_launch_wait_callback_t wait_callback,
    void* wait_callback_user_data,
    iree_hal_streaming_graph_exec_state_guard_t* out_guard) {
  IREE_ASSERT_ARGUMENT(out_guard);
  *out_guard = (iree_hal_streaming_graph_exec_state_guard_t){0};
  if (!exec) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);

  iree_hal_streaming_graph_exec_deferred_cleanup_t* cleanup = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      exec->host_allocator, sizeof(*cleanup), (void**)&cleanup));
  memset(cleanup, 0, sizeof(*cleanup));
  cleanup->compiled_state.host_allocator = exec->host_allocator;
  cleanup->compiled_state.context = exec->context;

  iree_status_t status =
      iree_hal_streaming_graph_exec_state_begin(exec, out_guard);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(exec->host_allocator, cleanup);
    return status;
  }
  out_guard->deferred_cleanup = cleanup;
  iree_status_t execution_status = iree_ok_status();
  status = iree_hal_streaming_graph_exec_wait_for_active_launch_locked(
      exec, wait_callback, wait_callback_user_data, &execution_status,
      /*fail_if_destroyed=*/true);
  if (iree_status_is_ok(status)) {
    status = execution_status;
  } else {
    iree_status_ignore(execution_status);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_exec_state_end(out_guard);
    return status;
  }
  return iree_ok_status();
}

void iree_hal_streaming_graph_exec_state_end(
    iree_hal_streaming_graph_exec_state_guard_t* guard) {
  if (!guard || !guard->exec) return;
  iree_hal_streaming_graph_exec_t* exec = guard->exec;
  iree_hal_streaming_graph_exec_deferred_cleanup_t* cleanup =
      (iree_hal_streaming_graph_exec_deferred_cleanup_t*)
          guard->deferred_cleanup;
  *guard = (iree_hal_streaming_graph_exec_state_guard_t){0};
  iree_slim_mutex_unlock(&exec->mutex);

  if (!cleanup) return;
  if (cleanup->has_compiled_state) {
    iree_hal_streaming_graph_exec_deinitialize_compiled_state(
        &cleanup->compiled_state);
  }
  if (cleanup->release_graph_memory_slot) {
    iree_hal_streaming_graph_memory_release_exec_slot(cleanup->graph);
  }
  iree_hal_streaming_graph_release(cleanup->graph);
  iree_allocator_free(exec->host_allocator, cleanup);
}

iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_exec_state_resolve_node(
    iree_hal_streaming_graph_exec_state_guard_t* guard,
    const void* node_address) {
  if (!guard || !guard->exec || !node_address) return NULL;
  iree_hal_streaming_graph_exec_t* exec = guard->exec;
  for (iree_hal_streaming_node_block_t* block = exec->graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->executable_source_node == node_address &&
          node->node_index < exec->instantiated_node_count) {
        return node;
      }
    }
  }
  return NULL;
}

iree_hal_streaming_graph_instantiate_flags_t
iree_hal_streaming_graph_exec_state_flags(
    const iree_hal_streaming_graph_exec_state_guard_t* guard) {
  IREE_ASSERT_ARGUMENT(guard);
  IREE_ASSERT_ARGUMENT(guard->exec);
  return (iree_hal_streaming_graph_instantiate_flags_t)guard->exec->flags;
}

bool iree_hal_streaming_graph_exec_state_node_is_enabled(
    const iree_hal_streaming_graph_exec_state_guard_t* guard,
    const iree_hal_streaming_graph_node_t* node) {
  return guard && guard->exec && node &&
         (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED) == 0;
}

bool iree_hal_streaming_graph_exec_node_is_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  if (!exec || !node) return false;
  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  if (!iree_status_is_ok(
          iree_hal_streaming_graph_exec_state_begin(exec, &guard))) {
    return false;
  }
  iree_hal_streaming_graph_node_t* resolved_node =
      iree_hal_streaming_graph_exec_state_resolve_node(&guard, node);
  const bool is_enabled =
      resolved_node && iree_hal_streaming_graph_exec_state_node_is_enabled(
                           &guard, resolved_node);
  iree_hal_streaming_graph_exec_state_end(&guard);
  return is_enabled;
}

iree_status_t iree_hal_streaming_graph_exec_state_set_node_enabled(
    iree_hal_streaming_graph_exec_state_guard_t* guard,
    iree_hal_streaming_graph_node_t* node, bool enabled) {
  if (!guard || !guard->exec || !guard->deferred_cleanup || !node) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  if (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  const uint32_t old_flags = node->flags;
  if (enabled) {
    node->flags &= ~IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED;
  } else {
    node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED;
  }
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_from_template_locked(guard);
  if (!iree_status_is_ok(status)) {
    node->flags = old_flags;
    iree_hal_streaming_graph_normalize_compound_enabled_state(
        guard->exec->graph);
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_set_node_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node, bool enabled) {
  if (!exec) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_context_operation_begin(exec->context));
  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  iree_status_t status = iree_hal_streaming_graph_exec_rebuild_state_begin(
      exec, NULL, NULL, &guard);
  iree_hal_streaming_graph_node_t* resolved_node = NULL;
  if (iree_status_is_ok(status)) {
    resolved_node =
        iree_hal_streaming_graph_exec_state_resolve_node(&guard, node);
    if (!resolved_node) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_state_set_node_enabled(
        &guard, resolved_node, enabled);
  }
  iree_hal_streaming_graph_exec_state_end(&guard);
  iree_hal_streaming_context_operation_end(exec->context);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_context_operation_begin(exec->context));
  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  iree_status_t status = iree_hal_streaming_graph_exec_rebuild_state_begin(
      exec, NULL, NULL, &guard);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_state_rebuild(&guard);
  }
  iree_hal_streaming_graph_exec_state_end(&guard);
  iree_hal_streaming_context_operation_end(exec->context);
  return status;
}

static iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template_locked(
    iree_hal_streaming_graph_exec_state_guard_t* guard) {
  if (!guard || !guard->exec || !guard->deferred_cleanup) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION);
  }
  iree_hal_streaming_graph_exec_t* exec = guard->exec;
  iree_hal_streaming_graph_normalize_compound_enabled_state(exec->graph);
  iree_hal_streaming_graph_exec_deferred_cleanup_t* cleanup =
      (iree_hal_streaming_graph_exec_deferred_cleanup_t*)
          guard->deferred_cleanup;
  if (cleanup->has_compiled_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable state already rebuilt by this guard");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_graph_validate_kernel_modules(exec->graph));
  iree_hal_streaming_graph_t* graph_memory_owner =
      exec->graph->graph_memory_owner_graph
          ? exec->graph->graph_memory_owner_graph
          : exec->graph;
  iree_slim_mutex_lock(&graph_memory_owner->graph_memory_state_mutex);
  const bool has_transferred_unfreed_allocation =
      graph_memory_owner->has_transferred_unfreed_allocation;
  iree_slim_mutex_unlock(&graph_memory_owner->graph_memory_state_mutex);
  if (has_transferred_unfreed_allocation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "an executable with a transferred graph allocation cannot be rebuilt");
  }

  iree_hal_streaming_graph_exec_t candidate;
  memset(&candidate, 0, sizeof(candidate));
  candidate.host_allocator = exec->host_allocator;
  candidate.context = exec->context;
  candidate.graph = exec->graph;
  candidate.flags = exec->flags;
  iree_hal_streaming_graph_exec_initialize_compiled_state(&candidate);
  iree_hal_streaming_graph_t* retired_compiled_graph =
      cleanup->compiled_state.graph ? cleanup->compiled_state.graph
                                    : exec->graph;

  iree_status_t status = iree_hal_resource_set_allocate(
      &candidate.context->device_entry->block_pool, &candidate.resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_build_contributions(&candidate);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_retain_exec(&candidate);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_instantiate_from_template(
        &candidate, exec->graph->node_blocks, exec->graph->node_count);
  }
  if (iree_status_is_ok(status)) {
    cleanup->compiled_state.graph = retired_compiled_graph;
    iree_hal_streaming_graph_exec_move_compiled_state(&cleanup->compiled_state,
                                                      exec);
    cleanup->has_compiled_state = true;
    iree_hal_streaming_graph_exec_move_compiled_state(exec, &candidate);
  } else {
    cleanup->compiled_state.graph = candidate.graph;
    iree_hal_streaming_graph_exec_move_compiled_state(&cleanup->compiled_state,
                                                      &candidate);
    cleanup->has_compiled_state = true;
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_state_rebuild(
    iree_hal_streaming_graph_exec_state_guard_t* guard) {
  return iree_hal_streaming_graph_exec_rebuild_from_template_locked(guard);
}

iree_status_t iree_hal_streaming_graph_exec_set_batch_mem_op_node_params(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size) {
  if (!exec || !node || (params_size > 0 && !params) ||
      (param_array_size > 0 && !param_array)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  iree_status_t status = iree_hal_streaming_graph_exec_rebuild_state_begin(
      exec, NULL, NULL, &guard);
  iree_hal_streaming_graph_node_t* resolved_node = NULL;
  if (iree_status_is_ok(status)) {
    resolved_node =
        iree_hal_streaming_graph_exec_state_resolve_node(&guard, node);
    if (!resolved_node) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_state_set_batch_mem_op_node_params(
        &guard, resolved_node, params, params_size, param_array,
        param_array_size);
  }
  iree_hal_streaming_graph_exec_state_end(&guard);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_state_set_batch_mem_op_node_params(
    iree_hal_streaming_graph_exec_state_guard_t* guard,
    iree_hal_streaming_graph_node_t* node, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size) {
  if (!guard || !guard->exec || !guard->deferred_cleanup || !node ||
      node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP ||
      (params_size > 0 && !params) || (param_array_size > 0 && !param_array)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_graph_exec_t* exec = guard->exec;

  iree_status_t status =
      iree_hal_streaming_graph_validate_kernel_modules(exec->graph);
  iree_hal_streaming_graph_t* graph_memory_owner =
      exec->graph->graph_memory_owner_graph
          ? exec->graph->graph_memory_owner_graph
          : exec->graph;
  iree_slim_mutex_lock(&graph_memory_owner->graph_memory_state_mutex);
  const bool has_transferred_unfreed_allocation =
      graph_memory_owner->has_transferred_unfreed_allocation;
  iree_slim_mutex_unlock(&graph_memory_owner->graph_memory_state_mutex);
  if (iree_status_is_ok(status) && has_transferred_unfreed_allocation) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "an executable with a transferred graph allocation cannot be rebuilt");
  }

  iree_hal_streaming_graph_batch_mem_op_node_attrs_t old_attrs =
      node->attrs.batch_mem_op;
  const iree_arena_checkpoint_t arena_checkpoint =
      iree_arena_checkpoint_save(&node->graph->arena);
  void* old_params = NULL;
  void* old_param_array = NULL;
  if (iree_status_is_ok(status) && old_attrs.params_size > 0) {
    status = iree_allocator_malloc(exec->host_allocator, old_attrs.params_size,
                                   &old_params);
    if (iree_status_is_ok(status)) {
      memcpy(old_params, old_attrs.params, old_attrs.params_size);
    }
  }
  if (iree_status_is_ok(status) && old_attrs.param_array_size > 0) {
    status = iree_allocator_malloc(
        exec->host_allocator, old_attrs.param_array_size, &old_param_array);
    if (iree_status_is_ok(status)) {
      memcpy(old_param_array, old_attrs.param_array,
             old_attrs.param_array_size);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_set_batch_mem_op_node_params(
        node, params, params_size, param_array, param_array_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_rebuild_from_template_locked(guard);
  }

  if (!iree_status_is_ok(status)) {
    node->attrs.batch_mem_op = old_attrs;
    if (old_params) {
      memcpy(old_attrs.params, old_params, old_attrs.params_size);
    }
    if (old_param_array) {
      memcpy(old_attrs.param_array, old_param_array,
             old_attrs.param_array_size);
    }
    iree_arena_checkpoint_restore(&arena_checkpoint);
  }
  iree_allocator_free(exec->host_allocator, old_param_array);
  iree_allocator_free(exec->host_allocator, old_params);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_set_event_node_event(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_node_type_t type,
    iree_hal_streaming_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!exec || !node || !event ||
      (type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD &&
       type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  // The same rule iree_hal_streaming_graph_add_event_node holds the template
  // to, applied to the executable an event node is retargeted in. On a record
  // node it is what lets a launch answer for every record block by comparing
  // one pair of contexts, and an event of another context would be refused at
  // the record itself anyway. A wait node is held to it because that function
  // holds both node types to it: retargeting is another way of naming a node's
  // event, and it must not seat one the template would have refused.
  if (event->context != exec->context) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event must belong to the graph context");
  }

  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  iree_status_t status = iree_hal_streaming_graph_exec_rebuild_state_begin(
      exec, NULL, NULL, &guard);
  iree_hal_streaming_graph_node_t* resolved_node = NULL;
  if (iree_status_is_ok(status)) {
    resolved_node =
        iree_hal_streaming_graph_exec_state_resolve_node(&guard, node);
    if (!resolved_node) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }
  iree_hal_streaming_event_t* old_event = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_state_set_event_node_event(
        &guard, resolved_node, type, event, &old_event);
  }
  iree_hal_streaming_graph_exec_state_end(&guard);
  iree_hal_streaming_event_release(old_event);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_state_set_event_node_event(
    iree_hal_streaming_graph_exec_state_guard_t* guard,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_node_type_t type,
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_event_t** out_old_event) {
  IREE_ASSERT_ARGUMENT(out_old_event);
  *out_old_event = NULL;
  if (!guard || !guard->exec || !guard->deferred_cleanup || !node || !event ||
      (type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD &&
       type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_graph_exec_t* exec = guard->exec;
  if (event->context != exec->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event must belong to the graph context");
  }
  if (node->type != type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_event_t* old_event = node->attrs.event.event;
  iree_hal_streaming_event_retain(event);
  node->attrs.event.event = event;
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_from_template_locked(guard);
  if (iree_status_is_ok(status)) {
    *out_old_event = old_event;
  } else {
    node->attrs.event.event = old_event;
    *out_old_event = event;
  }
  return status;
}

// Calculate the size needed for a block with variable-length arrays.
static iree_host_size_t iree_hal_streaming_graph_block_calculate_size(
    uint16_t wait_semaphore_count, uint16_t signal_semaphore_count) {
  iree_host_size_t size = sizeof(iree_hal_streaming_graph_block_t);
  size = iree_host_align(size, iree_alignof(uint16_t));
  size += wait_semaphore_count * sizeof(uint16_t);  // wait_semaphore_indices
  size = iree_host_align(size, iree_alignof(uint32_t));
  size += wait_semaphore_count * sizeof(uint32_t);  // wait_payload_deltas
  size = iree_host_align(size, iree_alignof(uint16_t));
  size +=
      signal_semaphore_count * sizeof(uint16_t);  // signal_semaphore_indices
  size = iree_host_align(size, iree_alignof(uint32_t));
  size += signal_semaphore_count * sizeof(uint32_t);  // signal_payload_deltas
  size = iree_host_align(size,
                         iree_alignof(iree_hal_streaming_graph_block_attrs_t));
  size += sizeof(iree_hal_streaming_graph_block_attrs_t);  // type-specific data
  return size;
}

static inline uint8_t* iree_hal_streaming_graph_block_align_ptr(
    uint8_t* ptr, iree_host_size_t alignment) {
  return (uint8_t*)(uintptr_t)iree_host_align((uintptr_t)ptr, alignment);
}

// Get pointers to all variable-length arrays in a block.
static inline void iree_hal_streaming_graph_block_get_ptrs(
    iree_hal_streaming_graph_block_t* block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  uint8_t* ptr = (uint8_t*)block + sizeof(*block);

  ptr = iree_hal_streaming_graph_block_align_ptr(ptr, iree_alignof(uint16_t));
  out_ptrs->wait_semaphore_indices = (uint16_t*)ptr;
  ptr +=
      block->wait_semaphore_count * sizeof(*out_ptrs->wait_semaphore_indices);

  ptr = iree_hal_streaming_graph_block_align_ptr(ptr, iree_alignof(uint32_t));
  out_ptrs->wait_payload_deltas = (uint32_t*)ptr;
  ptr += block->wait_semaphore_count * sizeof(*out_ptrs->wait_payload_deltas);

  ptr = iree_hal_streaming_graph_block_align_ptr(ptr, iree_alignof(uint16_t));
  out_ptrs->signal_semaphore_indices = (uint16_t*)ptr;
  ptr += block->signal_semaphore_count *
         sizeof(*out_ptrs->signal_semaphore_indices);

  ptr = iree_hal_streaming_graph_block_align_ptr(ptr, iree_alignof(uint32_t));
  out_ptrs->signal_payload_deltas = (uint32_t*)ptr;
  ptr +=
      block->signal_semaphore_count * sizeof(*out_ptrs->signal_payload_deltas);

  ptr = iree_hal_streaming_graph_block_align_ptr(
      ptr, iree_alignof(iree_hal_streaming_graph_block_attrs_t));
  out_ptrs->attrs = (iree_hal_streaming_graph_block_attrs_t*)ptr;
}

// Allocates a block with variable-length arrays.
static iree_status_t iree_hal_streaming_graph_block_allocate(
    iree_arena_allocator_t* arena_allocator,
    iree_hal_streaming_graph_block_type_t type, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  const iree_host_size_t total_size =
      iree_hal_streaming_graph_block_calculate_size(wait_semaphore_count,
                                                    signal_semaphore_count);
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena_allocator, total_size, (void**)&block));

  block->type = type;
  block->node_start_index = node_start_index;
  block->node_count = node_count;
  block->wait_semaphore_count = wait_semaphore_count;
  block->signal_semaphore_count = signal_semaphore_count;

  iree_hal_streaming_graph_block_get_ptrs(block, out_ptrs);
  *out_block = block;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_barrier_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_queue_barrier_flags_t flags,
    iree_hal_streaming_graph_node_t* source_mem_alloc_node,
    bool transfer_mem_alloc_on_accept,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set barrier attributes.
  iree_hal_streaming_graph_barrier_block_attrs_t* attrs =
      &out_ptrs->attrs->barrier;
  attrs->flags = flags;
  attrs->source_mem_alloc_node = source_mem_alloc_node;
  attrs->transfer_mem_alloc_on_accept = transfer_mem_alloc_on_accept;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_fill_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags, iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL, node_start_index,
              node_count, wait_semaphore_count, signal_semaphore_count, &block,
              out_ptrs));

  // Set fill attributes.
  iree_hal_streaming_graph_fill_block_attrs_t* attrs = &out_ptrs->attrs->fill;
  attrs->target_buffer = target_buffer;
  attrs->target_offset = target_offset;
  attrs->length = length;
  attrs->flags = flags;

  // Copy pattern data if provided.
  if (pattern_length > 0) {
    IREE_ASSERT(pattern_length < sizeof(attrs->pattern));
    memcpy(&attrs->pattern, pattern, pattern_length);
    attrs->pattern_length = pattern_length;
  }

  // Add buffer to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set, 1, &target_buffer));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_copy_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_copy_flags_t flags, iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY, node_start_index,
              node_count, wait_semaphore_count, signal_semaphore_count, &block,
              out_ptrs));

  // Set copy attributes.
  iree_hal_streaming_graph_copy_block_attrs_t* attrs = &out_ptrs->attrs->copy;
  attrs->source_buffer = source_buffer;
  attrs->source_offset = source_offset;
  attrs->target_buffer = target_buffer;
  attrs->target_offset = target_offset;
  attrs->length = length;
  attrs->flags = flags;

  // Add buffers to resource set.
  void* resources[2] = {source_buffer, target_buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set,
                                       IREE_ARRAYSIZE(resources), resources));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_host_call_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, void (*fn)(void* user_data),
    iree_hal_streaming_graph_deferred_host_call_fn_t deferred_fn,
    void* user_data, const iree_hal_streaming_graph_node_t* source_node,
    iree_hal_host_call_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set host call attributes.
  iree_hal_streaming_graph_host_call_block_attrs_t* attrs =
      &out_ptrs->attrs->host_call;
  attrs->fn = fn;
  attrs->deferred_fn = deferred_fn;
  attrs->user_data = user_data;
  if (source_node && source_node->attrs.host.user_data_size > 0 && user_data) {
    void* copied_data = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator,
                                source_node->attrs.host.user_data_size,
                                (void**)&copied_data));
    memcpy(copied_data, user_data, source_node->attrs.host.user_data_size);
    attrs->user_data = copied_data;
  }
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_event_block(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_block_type_t type, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count,
    iree_hal_streaming_graph_node_t* source_node,
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD &&
      type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid graph event block type");
  }

  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator, type, node_start_index, node_count,
              wait_semaphore_count, signal_semaphore_count, &block, out_ptrs));

  out_ptrs->attrs->event.source_node = source_node;
  out_ptrs->attrs->event.event = event;
  iree_hal_streaming_event_retain(event);
  // The one place a record block is built, and so where |exec| learns it holds
  // one. A launch reads this to answer for every record at once.
  if (type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD) {
    exec->records_events = true;
  }

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_dispatch_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_executable_t* executable,
    iree_host_size_t entry_point, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set dispatch attributes.
  iree_hal_streaming_graph_dispatch_block_attrs_t* attrs =
      &out_ptrs->attrs->dispatch;
  attrs->executable = executable;
  attrs->entry_point = entry_point;
  attrs->config = config;
  attrs->flags = flags;

  // Copy constants if provided.
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, constants.data_length,
                                &constants_copy));
    memcpy(constants_copy, constants.data, constants.data_length);
    attrs->constants =
        iree_make_const_byte_span(constants_copy, constants.data_length);
  } else {
    attrs->constants = iree_const_byte_span_empty();
  }

  // Copy bindings if provided.
  if (bindings.count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    const iree_host_size_t bindings_size =
        bindings.count * sizeof(*bindings_copy);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, bindings_size,
                                (void**)&bindings_copy));
    memcpy(bindings_copy, bindings.values, bindings_size);
    attrs->bindings.count = bindings.count;
    attrs->bindings.values = bindings_copy;

    // Add buffers to resource set.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert_strided(
                exec->resource_set, bindings.count, bindings.values,
                offsetof(iree_hal_buffer_ref_t, buffer),
                sizeof(iree_hal_buffer_ref_t)));
  } else {
    attrs->bindings = iree_hal_buffer_ref_list_empty();
  }

  // Add executable to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set, 1, &executable));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_execute_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_queue_execute_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  iree_hal_streaming_graph_execute_block_attrs_t* attrs =
      &out_ptrs->attrs->execute;
  attrs->flags = flags;

  // Create command buffer. HIP pointer lifetime is enforced by explicit API
  // ordering, not by scanning graph or kernarg contents: synchronous
  // destruction waits for active streams, and stream-ordered destruction queues
  // the release after prior work.
  //
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_command_buffer_create(
          exec->context->device, iree_hal_queue_family(exec->context->queue),
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
          IREE_HAL_COMMAND_CATEGORY_TRANSFER |
              IREE_HAL_COMMAND_CATEGORY_DISPATCH,
          /*binding_capacity=*/0, &attrs->command_buffer));

  // Add to resource set for cleanup.
  iree_status_t status = iree_hal_resource_set_insert(exec->resource_set, 1,
                                                      &attrs->command_buffer);

  // We don't technically retain a reference to it past here on the stack, just
  // in the resource set associated with the exec.
  iree_hal_command_buffer_release(attrs->command_buffer);

  if (iree_status_is_ok(status)) {
    *out_block = block;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_create_child_graph_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_streaming_graph_t* child_graph,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* child_exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_instantiate(
              child_graph,
              (iree_hal_streaming_graph_instantiate_flags_t)exec->flags,
              &child_exec));

  iree_hal_streaming_graph_block_t* block = NULL;
  iree_status_t status = iree_hal_streaming_graph_block_allocate(
      &exec->arena_allocator, IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
      node_start_index, node_count, wait_semaphore_count,
      signal_semaphore_count, &block, out_ptrs);
  if (iree_status_is_ok(status)) {
    out_ptrs->attrs->child_graph.exec = child_exec;
    // A launch of |exec| walks the child's blocks too, so the records they hold
    // are records of this launch. The child instantiated above already carries
    // its own children's, which makes the property transitive.
    exec->records_events |= child_exec->records_events;
    *out_block = block;
  } else {
    iree_hal_streaming_graph_exec_release(child_exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_streaming_node_index_set_t {
  uint32_t values[8];
  uint32_t count : 31;
  uint32_t invalid : 1;
} iree_hal_streaming_node_index_set_t;

// Resets the set to empty.
static inline void iree_hal_streaming_node_index_set_reset(
    iree_hal_streaming_node_index_set_t* set) {
  set->count = 0;
  set->invalid = 0;
}

// Returns true if the |set| is invalid or |value| is present.
static bool iree_hal_streaming_node_index_set_test_hazard(
    const iree_hal_streaming_node_index_set_t* set, uint32_t value) {
  if (set->invalid) return true;
  for (uint32_t i = 0; i < set->count; ++i) {
    if (set->values[i] == value) {
      return true;
    }
  }
  return false;
}

// Inserts |value| into the |set|.
// If the set has reached capacity it is set to invalid and all future tests
// will return a hazard.
static void iree_hal_streaming_node_index_set_insert(
    iree_hal_streaming_node_index_set_t* set, uint32_t value) {
  if (set->count >= IREE_ARRAYSIZE(set->values)) {
    set->invalid = 1;
    return;
  }
  set->values[set->count++] = value;
}

static bool iree_hal_streaming_graph_node_has_recorded_dependency_hazard(
    const iree_hal_streaming_graph_node_t* node,
    const iree_hal_streaming_graph_edge_t* additional_edges,
    const uint32_t* node_index_map,
    const iree_hal_streaming_node_index_set_t* barrier_index_set) {
  for (uint32_t j = 0; j < node->dependency_count; ++j) {
    const uint32_t dependency_sort_index =
        node_index_map[node->dependencies[j]->node_index];
    if (dependency_sort_index == UINT32_MAX) {
      continue;
    }
    if (iree_hal_streaming_node_index_set_test_hazard(barrier_index_set,
                                                      dependency_sort_index)) {
      return true;
    }
  }

  for (const iree_hal_streaming_graph_edge_t* edge = additional_edges; edge;
       edge = edge->next) {
    if (edge->to != node) continue;
    const uint32_t dependency_sort_index =
        node_index_map[edge->from->node_index];
    if (dependency_sort_index == UINT32_MAX) {
      continue;
    }
    if (iree_hal_streaming_node_index_set_test_hazard(barrier_index_set,
                                                      dependency_sort_index)) {
      return true;
    }
  }

  return false;
}

static iree_status_t iree_hal_streaming_graph_record_dependency_barrier(
    iree_hal_command_buffer_t* command_buffer) {
  // A dependency orders both command execution and memory visibility. Graph
  // nodes can alternate between dispatches and transfers, so make writes from
  // either operation class available to reads by either class.
  static const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  return iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
}

static iree_status_t iree_hal_streaming_graph_append_pointer_validation(
    iree_hal_streaming_graph_exec_t* exec,
    const iree_hal_streaming_graph_kernel_node_attrs_t* attrs) {
  if (!attrs->pointer_validator ||
      attrs->symbol->parameters.binding_count == 0) {
    return iree_ok_status();
  }
  if (exec->pointer_validation_count >= exec->pointer_validation_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph pointer validation table overflow");
  }

  iree_hal_streaming_graph_pointer_validation_t* validation =
      &exec->pointer_validations[exec->pointer_validation_count];
  *validation = (iree_hal_streaming_graph_pointer_validation_t){
      .parameters = attrs->symbol->parameters,
      .validator = attrs->pointer_validator,
      .user_data = attrs->pointer_validator_user_data,
  };
  if (attrs->constants.data_length > 0) {
    void* constants = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &exec->arena_allocator, attrs->constants.data_length, &constants));
    memcpy(constants, attrs->constants.data, attrs->constants.data_length);
    validation->constants =
        iree_make_const_byte_span(constants, attrs->constants.data_length);
  }
  ++exec->pointer_validation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_resolve_memory_capability(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_memory_access_t required_access, uint64_t capability_id,
    iree_hal_streaming_buffer_ref_t fallback_ref,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(out_ref);
  *out_ref = fallback_ref;
  if (capability_id == 0) return iree_ok_status();
  if (!context || device_ptr == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VMM graph node has no context or device address");
  }
  if (exec->memory_validation_count >= exec->memory_validation_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph memory validation table overflow");
  }

  const iree_device_size_t lookup_size = size > 0 ? size : 1;
  uint64_t resolved_capability_id = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_lookup_range_with_access(
      context, device_ptr, lookup_size, required_access, out_ref,
      &resolved_capability_id));
  if (resolved_capability_id != capability_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VMM graph node access capability has been revoked or replaced");
  }
  exec->memory_validations[exec->memory_validation_count++] =
      (iree_hal_streaming_graph_memory_validation_t){
          .context = context,
          .device_ptr = device_ptr,
          .size = lookup_size,
          .required_access = required_access,
          .capability_id = capability_id,
      };
  return iree_ok_status();
}

// Helper to record nodes from a partition into a command buffer.
static iree_status_t iree_hal_streaming_graph_record_partition(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_sort_node_t* sorted_nodes,
    uint32_t node_start_index, uint32_t node_count,
    const uint32_t* node_index_map, uint8_t stream_id,
    iree_hal_streaming_graph_edge_t* additional_edges,
    bool preserve_sorted_order, iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Begin recording command buffer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin(command_buffer));

  // Scope the partition into a debug group.
  // TODO: propagate graph information (name, origin, etc).
  const iree_string_view_t label_name = iree_make_cstring_view("tbd_partition");
  const iree_hal_label_location_t* location = NULL;
  const iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin_debug_group(command_buffer, label_name,
                                                    label_color, location));

  // Record nodes assigned to this stream.
  //
  // HIP graph dependency edges define the minimum ordering. Graph memory nodes
  // are currently represented by graph-template allocations rather than
  // per-launch stream-ordered allocations, so executables containing them keep
  // sorted command order as a conservative lifetime boundary.
  //
  // We use a small linear scan set to make the test for hazards faster: we have
  // the original unsorted node indices of dependencies but not the sorted ones
  // we'd need to index into the sorted_nodes list and this avoids needing to
  // do that mapping.
  iree_status_t status = iree_ok_status();
  uint32_t in_stream_count = 0;
  iree_hal_streaming_node_index_set_t barrier_index_set;
  iree_hal_streaming_node_index_set_reset(&barrier_index_set);
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    iree_hal_streaming_graph_sort_node_t* sort_node =
        &sorted_nodes[node_start_index + i];
    // Ignore nodes from other streams.
    if (sort_node->stream_id != stream_id) continue;
    iree_hal_streaming_graph_node_t* node = sort_node->node;
    if (in_stream_count > 0) {
      const bool has_dependency_hazard =
          iree_hal_streaming_graph_node_has_recorded_dependency_hazard(
              node, additional_edges, node_index_map, &barrier_index_set);
      if (preserve_sorted_order || has_dependency_hazard) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_streaming_graph_record_dependency_barrier(command_buffer));
        iree_hal_streaming_node_index_set_reset(&barrier_index_set);
      }
    }
    ++in_stream_count;
    switch (node->type) {
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL: {
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            &node->attrs.kernel;
        iree_hal_streaming_symbol_t* symbol = attrs->symbol;
        const iree_hal_dispatch_config_t config = {
            .workgroup_size =
                {
                    attrs->block_dim[0],
                    attrs->block_dim[1],
                    attrs->block_dim[2],
                },
            .workgroup_count =
                {
                    attrs->grid_dim[0],
                    attrs->grid_dim[1],
                    attrs->grid_dim[2],
                },
            .dynamic_workgroup_local_memory = attrs->shared_memory_bytes,
        };
        const iree_hal_dispatch_flags_t flags =
            attrs->bindings.count
                ? IREE_HAL_DISPATCH_FLAG_NONE
                : IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
        status = iree_hal_command_buffer_dispatch(
            command_buffer, symbol->executable,
            iree_hal_executable_function_from_index(symbol->export_ordinal),
            config, attrs->constants, attrs->bindings, flags);
        if (iree_status_is_ok(status)) {
          status =
              iree_hal_streaming_graph_append_pointer_validation(exec, attrs);
        }
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY: {
        iree_hal_streaming_graph_memcpy_node_attrs_t attrs = node->attrs.memcpy;
        status = iree_hal_streaming_graph_resolve_memory_capability(
            exec, exec->context, attrs.dst_capability_ptr,
            attrs.dst_capability_size, IREE_HAL_MEMORY_ACCESS_WRITE,
            attrs.dst_capability_id, attrs.dst_ref, &attrs.dst_ref);
        if (iree_status_is_ok(status)) {
          status = iree_hal_streaming_graph_resolve_memory_capability(
              exec, exec->context, attrs.src_capability_ptr,
              attrs.src_capability_size, IREE_HAL_MEMORY_ACCESS_READ,
              attrs.src_capability_id, attrs.src_ref, &attrs.src_ref);
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_streaming_graph_record_memcpy_node(command_buffer,
                                                               &attrs);
        }
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET: {
        iree_hal_streaming_graph_memset_node_attrs_t attrs = node->attrs.memset;
        iree_device_size_t fill_length = 0;
        if (IREE_UNLIKELY(!iree_device_size_checked_mul(
                attrs.pattern_size, attrs.count, &fill_length))) {
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "memset node size overflows device size");
          break;
        }
        status = iree_hal_streaming_graph_resolve_memory_capability(
            exec, exec->context, attrs.dst_capability_ptr,
            attrs.dst_capability_size, IREE_HAL_MEMORY_ACCESS_WRITE,
            attrs.dst_capability_id, attrs.dst_ref, &attrs.dst_ref);
        if (iree_status_is_ok(status)) {
          status = iree_hal_command_buffer_fill_buffer(
              command_buffer,
              iree_hal_streaming_convert_range_buffer_ref(attrs.dst_ref,
                                                          fill_length),
              &attrs.pattern, attrs.pattern_size, attrs.flags);
        }
        break;
      }
      default: {
        // Non-recordable nodes shouldn't be here.
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "non-recordable node type %d in recordable partition",
            (int)node->type);
        break;
      }
    }
    iree_hal_streaming_node_index_set_insert(&barrier_index_set,
                                             node_index_map[node->node_index]);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end_debug_group(command_buffer);
  }

  // End recording.
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_streaming_graph_exec_allocate_launch_frontier_state(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec->block_count == 0) return iree_ok_status();

  iree_host_size_t completion_semaphore_size = 0;
  iree_host_size_t completion_value_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              exec->block_count, sizeof(*exec->block_completion_semaphores),
              &completion_semaphore_size) ||
          !iree_host_size_checked_mul(
              exec->block_count, sizeof(*exec->block_completion_base_values),
              &completion_value_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph block completion table overflows");
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&exec->arena_allocator, completion_semaphore_size,
                          (void**)&exec->block_completion_semaphores));
  memset(exec->block_completion_semaphores, 0, completion_semaphore_size);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&exec->arena_allocator, completion_value_size,
                          (void**)&exec->block_completion_base_values));
  memset(exec->block_completion_base_values, 0, completion_value_size);

  for (uint32_t i = 0; i < exec->block_count; ++i) {
    iree_hal_semaphore_t* semaphore = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        exec->context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0ull,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
    exec->block_completion_semaphores[i] = semaphore;
    iree_status_t status =
        iree_hal_resource_set_insert(exec->resource_set, 1, &semaphore);
    iree_hal_semaphore_release(semaphore);
    IREE_RETURN_IF_ERROR(status);
  }

  iree_host_size_t frontier_capacity = exec->block_count;
  for (uint32_t i = 0; i < exec->block_count; ++i) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[i];
    if (!block ||
        block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      continue;
    }
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            frontier_capacity,
            ptrs.attrs->child_graph.exec->launch_frontier_capacity,
            &frontier_capacity))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "recursive graph launch frontier overflows");
    }
  }
  iree_host_size_t frontier_semaphore_size = 0;
  iree_host_size_t frontier_value_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(frontier_capacity,
                                      sizeof(*exec->launch_frontier_semaphores),
                                      &frontier_semaphore_size) ||
          !iree_host_size_checked_mul(frontier_capacity,
                                      sizeof(*exec->launch_frontier_values),
                                      &frontier_value_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "recursive graph launch frontier table overflows");
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&exec->arena_allocator, frontier_semaphore_size,
                          (void**)&exec->launch_frontier_semaphores));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&exec->arena_allocator, frontier_value_size,
                          (void**)&exec->launch_frontier_values));
  exec->launch_frontier_capacity = frontier_capacity;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_instantiate_from_template(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_TRACE_ZONE_BEGIN(z0);
  exec->instantiated_node_count = node_count;
  exec->instantiated_visible_node_count =
      iree_hal_streaming_graph_visible_node_count(exec->graph);

  if (node_count > 0) {
    iree_host_size_t validation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            node_count, sizeof(*exec->pointer_validations),
            &validation_size))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph pointer validation table overflow");
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, validation_size,
                                (void**)&exec->pointer_validations));
    memset(exec->pointer_validations, 0, validation_size);
    exec->pointer_validation_capacity = node_count;

    iree_host_size_t memory_validation_capacity = 0;
    iree_host_size_t memory_validation_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(node_count, 2,
                                        &memory_validation_capacity) ||
            !iree_host_size_checked_mul(memory_validation_capacity,
                                        sizeof(*exec->memory_validations),
                                        &memory_validation_size))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph memory validation table overflow");
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, memory_validation_size,
                                (void**)&exec->memory_validations));
    memset(exec->memory_validations, 0, memory_validation_size);
    exec->memory_validation_capacity = memory_validation_capacity;
  }

  // Use the new scheduler to analyze and partition the graph.
  iree_hal_streaming_graph_schedule_t schedule;
  iree_hal_streaming_graph_edge_t* additional_edges =
      exec->graph ? exec->graph->additional_edges : NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_schedule_nodes(
              node_blocks, node_count, /*disabled_nodes=*/NULL,
              /*disabled_node_count=*/0, additional_edges,
              &exec->arena_allocator, &schedule));

  // Allocate block array.
  exec->block_count = schedule.block_count;
  if (schedule.partition_count == 0) {
    exec->block_count = 0;
    exec->blocks = NULL;
    exec->semaphore_count = 0;
    exec->semaphores = NULL;
    exec->semaphore_base_values = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&exec->arena_allocator,
                              exec->block_count * sizeof(*exec->blocks),
                              (void**)&exec->blocks));
  memset(exec->blocks, 0, exec->block_count * sizeof(*exec->blocks));

  // Calculate semaphore count needed.
  // We need semaphores at partition boundaries for synchronization.
  // Multi-stream partitions need join semaphores.
  //
  // This currently allocates semaphores per block boundary. A future
  // optimization can reduce this to the maximum layer size by advancing
  // timeline values between partitions.
  uint32_t semaphore_count = 0;
  if (schedule.partition_count > 1) {
    for (iree_host_size_t i = 0; i < schedule.partition_count - 1; i++) {
      if (schedule.partitions[i].stream_count > 1) {
        // Multi-stream partition needs one semaphore per stream for join.
        semaphore_count += schedule.partitions[i].stream_count;
      } else {
        // Single stream needs one semaphore.
        semaphore_count += 1;
      }
    }
  }
  exec->semaphore_count = semaphore_count;

  if (exec->semaphore_count > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&exec->arena_allocator,
                            exec->semaphore_count * sizeof(*exec->semaphores),
                            (void**)&exec->semaphores));
    memset(exec->semaphores, 0,
           exec->semaphore_count * sizeof(*exec->semaphores));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(
                &exec->arena_allocator,
                exec->semaphore_count * sizeof(*exec->semaphore_base_values),
                (void**)&exec->semaphore_base_values));
    memset(exec->semaphore_base_values, 0,
           exec->semaphore_count * sizeof(*exec->semaphore_base_values));

    // Create the internal semaphores that carry values between partitions. The
    // resource set is their sole owner; exec->semaphores are borrowed pointers
    // and anything outliving the executable takes its own reference.
    iree_status_t status = iree_ok_status();
    for (uint32_t i = 0; i < exec->semaphore_count && iree_status_is_ok(status);
         i++) {
      iree_hal_semaphore_t* semaphore = NULL;
      status = iree_hal_semaphore_create(
          exec->context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0ull,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
      if (iree_status_is_ok(status)) {
        status =
            iree_hal_resource_set_insert(exec->resource_set, 1, &semaphore);
        iree_hal_semaphore_release(semaphore);
      }
      if (iree_status_is_ok(status)) {
        exec->semaphores[i] = semaphore;
      }
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  // Create blocks from partitions.
  uint32_t block_index = 0;
  uint32_t semaphore_index = 0;
  for (iree_host_size_t p = 0; p < schedule.partition_count; p++) {
    const iree_hal_streaming_graph_partition_t* partition =
        &schedule.partitions[p];

    // Determine wait semaphores for chaining FROM the previous partition.
    // The first partition waits on the original submission stream timeline
    // semaphores. Subsequent partitions wait on the previous partitions
    // semaphores, of which there may be several for a join operation.
    // Note that we don't allocate space for the initial semaphores as those are
    // part of the submission, not the exec object.
    uint16_t wait_semaphore_count = 0;
    if (p > 0) {
      iree_hal_streaming_graph_partition_t* prev_partition =
          &schedule.partitions[p - 1];
      wait_semaphore_count =
          prev_partition->stream_count > 1 ? prev_partition->stream_count : 1;
    }

    // Determine signal semaphores for chaining TO the next partition.
    // Each partition gets at least one signal semaphore while multi-stream
    // partitions get one per stream to allow the subsequent partition to join
    // them. The final partition signals the original submission stream timeline
    // semaphores. Note that we don't allocate space for the final semaphores as
    // those are part of the submission, not the exec object.
    uint16_t signal_semaphore_count = 0;
    if (p < schedule.partition_count - 1) {
      signal_semaphore_count =
          partition->stream_count > 1 ? partition->stream_count : 1;
    }

    if (partition->type == IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE) {
      // If only one node is in the partition and it's recordable, we
      // may be able to route it to a dedicated partition type after this
      // function is split into smaller helpers.
      const uint8_t stream_count = partition->stream_count;
      const uint32_t partition_wait_semaphore_start =
          semaphore_index - wait_semaphore_count;
      const uint32_t partition_signal_semaphore_start = semaphore_index;
      for (uint8_t s = 0; s < stream_count; s++) {
        iree_hal_streaming_graph_block_t* block = NULL;

        // All streams in partition wait on same semaphores from previous.
        // Each stream in multi-stream partition signals its own semaphore.
        // Single stream signals all semaphores for the partition.
        // But if this is the last partition (signal_semaphore_count=0), don't
        // signal.
        uint16_t block_signal_count = 0;
        if (signal_semaphore_count > 0) {
          block_signal_count = (stream_count > 1) ? 1 : signal_semaphore_count;
        }
        iree_hal_streaming_graph_block_ptrs_t ptrs;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_execute_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, block_signal_count,
                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE, &block, &ptrs));

        // Record nodes for this stream into the command buffer.
        // For single stream (s=0), records all nodes.
        // For multi-stream, records nodes filtered by stream_id.
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_record_partition(
                    exec, schedule.sorted_nodes, partition->start_index,
                    partition->count, schedule.node_index_map, s,
                    additional_edges, exec->uses_graph_memory_nodes,
                    ptrs.attrs->execute.command_buffer));

        // Set up semaphore indices.
        if (wait_semaphore_count > 0) {
          for (uint16_t w = 0; w < wait_semaphore_count; w++) {
            ptrs.wait_semaphore_indices[w] = partition_wait_semaphore_start + w;
            ptrs.wait_payload_deltas[w] = 1;
          }
        }
        if (block_signal_count > 0) {
          if (stream_count > 1) {
            // Multi-stream: each stream signals its own semaphore.
            ptrs.signal_semaphore_indices[0] =
                partition_signal_semaphore_start + s;
            ptrs.signal_payload_deltas[0] = 1;
          } else {
            // Single stream: signal all semaphores.
            for (uint16_t i = 0; i < block_signal_count; i++) {
              ptrs.signal_semaphore_indices[i] =
                  partition_signal_semaphore_start + i;
              ptrs.signal_payload_deltas[i] = 1;
            }
          }
        }

        exec->blocks[block_index++] = block;
      }

      // Advance semaphore index by the number of signal semaphores.
      semaphore_index += signal_semaphore_count;
    } else {
      // Set up semaphore indices.
      iree_hal_streaming_graph_block_t* block = NULL;
      iree_hal_streaming_graph_block_ptrs_t ptrs;
      if (partition->type ==
          IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL) {
        // Host call gets its own block.
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_host_call_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    node->attrs.host.fn, node->attrs.host.deferred_fn,
                    node->attrs.host.user_data, node,
                    IREE_HAL_HOST_CALL_FLAG_NONE, &block, &ptrs));
        for (uint8_t j = 0; j < node->attrs.host.memory_validation_count; ++j) {
          const iree_hal_streaming_graph_host_memory_validation_t* validation =
              &node->attrs.host.memory_validations[j];
          iree_hal_streaming_buffer_ref_t resolved_ref = {0};
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, iree_hal_streaming_graph_resolve_memory_capability(
                      exec, validation->context, validation->device_ptr,
                      validation->size, validation->required_access,
                      validation->capability_id,
                      (iree_hal_streaming_buffer_ref_t){0}, &resolved_ref));
        }
      } else if (partition->type ==
                 IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_DISPATCH) {
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            &node->attrs.kernel;
        const iree_hal_dispatch_config_t config = {
            .workgroup_size =
                {
                    attrs->block_dim[0],
                    attrs->block_dim[1],
                    attrs->block_dim[2],
                },
            .workgroup_count =
                {
                    attrs->grid_dim[0],
                    attrs->grid_dim[1],
                    attrs->grid_dim[2],
                },
            .dynamic_workgroup_local_memory = attrs->shared_memory_bytes,
        };
        iree_hal_dispatch_flags_t flags =
            attrs->cooperative ? IREE_HAL_DISPATCH_FLAG_COOPERATIVE
                               : IREE_HAL_DISPATCH_FLAG_NONE;
        if (attrs->bindings.count == 0) {
          flags |= IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
        }
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_dispatch_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    attrs->symbol->executable, attrs->symbol->export_ordinal,
                    config, attrs->constants, attrs->bindings, flags, &block,
                    &ptrs));
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_streaming_graph_append_pointer_validation(exec, attrs));
      } else if (partition->type ==
                 IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_GRAPH) {
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_child_graph_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    node->attrs.child_graph.graph, &block, &ptrs));
      } else {
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD ||
            node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT) {
          const iree_hal_streaming_graph_block_type_t block_type =
              node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD
                  ? IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD
                  : IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_streaming_graph_create_event_block(
                  exec, block_type, partition->start_index, partition->count,
                  wait_semaphore_count, signal_semaphore_count, node,
                  node->attrs.event.event, &block, &ptrs));
        } else {
          // Empty/barrier partition.
          iree_hal_streaming_graph_node_t* source_mem_alloc_node = NULL;
          bool transfer_mem_alloc_on_accept = false;
          for (uint32_t j = 0; j < partition->count; ++j) {
            iree_hal_streaming_graph_node_t* partition_node =
                schedule.sorted_nodes[partition->start_index + j].node;
            if (partition_node->type !=
                IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC) {
              continue;
            }
            IREE_ASSERT(partition->count == 1,
                        "graph allocation must compile to an exact one-node "
                        "acceptance barrier");
            source_mem_alloc_node = partition_node;
            const bool has_matching_free =
                iree_hal_streaming_graph_has_free_node_for_pointer(
                    exec->graph, partition_node->attrs.mem_alloc.dptr);
            const bool auto_free_on_launch = iree_all_bits_set(
                exec->flags,
                IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            transfer_mem_alloc_on_accept =
                !has_matching_free && !auto_free_on_launch;
          }
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, iree_hal_streaming_graph_create_barrier_block(
                      exec, partition->start_index, partition->count,
                      wait_semaphore_count, signal_semaphore_count,
                      IREE_HAL_QUEUE_BARRIER_FLAG_NONE, source_mem_alloc_node,
                      transfer_mem_alloc_on_accept, &block, &ptrs));
        }
      }
      if (wait_semaphore_count > 0) {
        for (uint16_t w = 0; w < wait_semaphore_count; w++) {
          ptrs.wait_semaphore_indices[w] =
              semaphore_index - wait_semaphore_count + w;
          ptrs.wait_payload_deltas[w] = 1;
        }
      }
      if (signal_semaphore_count > 0) {
        for (uint16_t i = 0; i < signal_semaphore_count; i++) {
          ptrs.signal_semaphore_indices[i] = semaphore_index + i;
          ptrs.signal_payload_deltas[i] = 1;
        }
      }
      // Advance semaphore index by all signal semaphores.
      semaphore_index += signal_semaphore_count;
      exec->blocks[block_index++] = block;
    }
  }
  exec->block_count = block_index;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_exec_allocate_launch_frontier_state(exec));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Cells one chunk of the deferred launch-release list holds. The first chunk
// lives in the launching frame, so this is also how many event references a
// launch can replace without reaching the heap.
#define IREE_HAL_STREAMING_LAUNCH_RELEASE_CHUNK_CAPACITY 16

// References an event wait or record stops needing during a graph launch.
typedef struct iree_hal_streaming_launch_release_t {
  // Owning event point to release after the executable and stream unlock.
  iree_hal_streaming_recorded_point_t recorded_point;
  // Owned capture graph reference to release after those locks, or NULL.
  iree_hal_streaming_graph_t* graph;
} iree_hal_streaming_launch_release_t;

// Event references a launch must release after dropping its locks. Releasing a
// recorded point can retire a timestamp slot or destroy a semaphore, and
// releasing the last reference to a captured graph frees the allocations it
// owns, synchronizes every context, and relocks the launch stream.
//
// Storage is chunked and never moves, so a cell handed out stays valid for the
// life of the list.
typedef struct iree_hal_streaming_launch_release_chunk_t {
  // Next chunk, NULL at the tail.
  struct iree_hal_streaming_launch_release_chunk_t* next;
  // Number of cells in |releases| that are part of the list.
  iree_host_size_t count;
  // Owned references, zeroed for a cell whose event operation did not run.
  iree_hal_streaming_launch_release_t
      releases[IREE_HAL_STREAMING_LAUNCH_RELEASE_CHUNK_CAPACITY];
} iree_hal_streaming_launch_release_chunk_t;

typedef struct iree_hal_streaming_launch_release_list_t {
  // Allocator the heap chunks come from.
  iree_allocator_t host_allocator;
  // Chunk cells are added to, never NULL; |first| until the list grows.
  iree_hal_streaming_launch_release_chunk_t* tail;
  // Storage covering launches dropping no more references than it holds,
  // living in the launching frame.
  iree_hal_streaming_launch_release_chunk_t first;
} iree_hal_streaming_launch_release_list_t;

// Prepares |out_list| to collect references, growing from |host_allocator|.
static void iree_hal_streaming_launch_release_list_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_launch_release_list_t* out_list) {
  out_list->host_allocator = host_allocator;
  out_list->tail = &out_list->first;
  out_list->first.next = NULL;
  out_list->first.count = 0;
}

// Adds an empty cell to |list| and returns it. The cell counts as part of the
// list from here: deinitializing releases whatever it holds, and a zeroed cell
// releases nothing. Growing is the only step that can fail and it runs before
// the event operation acquires or displaces a reference, so a failure drops
// nothing. The cell stays valid for the life of the list.
IREE_MUST_USE_RESULT static iree_status_t
iree_hal_streaming_launch_release_list_push_empty(
    iree_hal_streaming_launch_release_list_t* list,
    iree_hal_streaming_launch_release_t** out_cell) {
  *out_cell = NULL;
  iree_hal_streaming_launch_release_chunk_t* tail = list->tail;
  if (tail->count == IREE_ARRAYSIZE(tail->releases)) {
    iree_hal_streaming_launch_release_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(list->host_allocator,
                                               sizeof(*chunk), (void**)&chunk));
    *chunk = (iree_hal_streaming_launch_release_chunk_t){0};
    tail->next = chunk;
    list->tail = chunk;
    tail = chunk;
  }
  iree_hal_streaming_launch_release_t* cell = &tail->releases[tail->count++];
  *cell = (iree_hal_streaming_launch_release_t){0};
  *out_cell = cell;
  return iree_ok_status();
}

// Releases every reference |list| collected and frees the chunks it grew into.
// The list is dead afterwards: the launch that initialized it is the only
// owner and it deinitializes once, on its way out.
static void iree_hal_streaming_launch_release_list_deinitialize(
    iree_hal_streaming_launch_release_list_t* list) {
  iree_hal_streaming_launch_release_chunk_t* chunk = &list->first;
  while (chunk) {
    for (iree_host_size_t i = 0; i < chunk->count; ++i) {
      iree_hal_streaming_event_release_recorded_point(
          &chunk->releases[i].recorded_point);
      iree_hal_streaming_graph_release(chunk->releases[i].graph);
    }
    iree_hal_streaming_launch_release_chunk_t* next = chunk->next;
    if (chunk != &list->first) {
      iree_allocator_free(list->host_allocator, chunk);
    }
    chunk = next;
  }
}

static iree_status_t iree_hal_streaming_graph_exec_validate_pointers_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_context_t* context) {
  for (iree_host_size_t i = 0; i < exec->memory_validation_count; ++i) {
    const iree_hal_streaming_graph_memory_validation_t* validation =
        &exec->memory_validations[i];
    iree_hal_streaming_context_t* validation_context =
        validation->context ? validation->context : context;
    iree_hal_streaming_buffer_ref_t resolved_ref;
    uint64_t resolved_capability_id = 0;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_lookup_range_with_access(
        validation_context, validation->device_ptr, validation->size,
        validation->required_access, &resolved_ref, &resolved_capability_id));
    if (resolved_capability_id != validation->capability_id) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VMM graph memory capability has been revoked or replaced");
    }
  }
  if (exec->pointer_validation_count > 0 && exec->context != context) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "captured kernel pointers require the graph execution context");
  }
  for (iree_host_size_t i = 0; i < exec->pointer_validation_count; ++i) {
    const iree_hal_streaming_graph_pointer_validation_t* validation =
        &exec->pointer_validations[i];
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_validate_native_kernel_argument_pointers(
            context, &validation->parameters, validation->constants,
            validation->validator, validation->user_data));
  }
  for (uint32_t i = 0; i < exec->block_count; ++i) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[i];
    if (!block ||
        block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      continue;
    }
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_validate_pointers_locked(
        ptrs.attrs->child_graph.exec, context));
  }
  return iree_ok_status();
}

typedef struct iree_hal_streaming_graph_launch_frontier_t {
  iree_hal_semaphore_t** semaphores;
  uint64_t* values;
  iree_host_size_t capacity;
  iree_host_size_t count;
} iree_hal_streaming_graph_launch_frontier_t;

static iree_status_t iree_hal_streaming_graph_exec_submit_blocks_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_launch_release_list_t* deferred_releases,
    iree_hal_streaming_graph_launch_frontier_t* launch_frontier,
    bool* out_external_signals_reachable);

static iree_status_t iree_hal_streaming_graph_host_callback(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_host_callback_t call_fn =
      (iree_hal_streaming_host_callback_t)args[0];
  void* call_user_data = (void*)args[1];
  iree_hal_streaming_graph_deferred_host_call_fn_t deferred_fn =
      (iree_hal_streaming_graph_deferred_host_call_fn_t)args[2];
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[0]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[1]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[2]);
  if (deferred_fn) {
    iree_status_t status = deferred_fn(call_user_data, context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  call_fn(call_user_data);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Submits |block| on |stream| behind |wait_semaphores|, signaling
// |signal_semaphores| when it completes. |launch_stream_tail_value| and
// |deferred_releases| is meaningful only to a child graph block, which carries
// it down into the child's own walk, and |record_point| only to an event record
// block, which receives it describing the point the record signals and leaves
// it owning what it names.
static iree_status_t iree_hal_streaming_graph_submit_block(
    iree_hal_streaming_graph_block_t* block,
    const iree_hal_streaming_graph_block_ptrs_t* ptrs,
    iree_hal_streaming_stream_t* stream, uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* record_point,
    iree_hal_streaming_launch_release_list_t* deferred_releases,
    iree_hal_streaming_graph_launch_frontier_t* launch_frontier,
    bool* out_external_signals_reachable) {
  *out_external_signals_reachable = false;
#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
  if (block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH &&
      iree_hal_streaming_graph_test_queue_submission_observer) {
    iree_hal_streaming_graph_test_queue_submission_observer(
        iree_hal_streaming_graph_test_queue_submission_observer_user_data);
  }
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION
  switch (block->type) {
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      return iree_hal_streaming_event_enqueue_record(
          ptrs->attrs->event.event, stream->context, stream->queue,
          wait_semaphores, signal_semaphores, record_point);
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
      const iree_hal_queue_barrier_flags_t flags =
          block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
              ? ptrs->attrs->barrier.flags
              : IREE_HAL_QUEUE_BARRIER_FLAG_NONE;
      return iree_hal_queue_barrier(stream->queue, wait_semaphores,
                                    signal_semaphores, flags);
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL: {
      return iree_hal_queue_fill(
          stream->queue, wait_semaphores, signal_semaphores,
          ptrs->attrs->fill.target_buffer, ptrs->attrs->fill.target_offset,
          ptrs->attrs->fill.length, &ptrs->attrs->fill.pattern,
          ptrs->attrs->fill.pattern_length, ptrs->attrs->fill.flags);
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY: {
      return iree_hal_queue_copy(
          stream->queue, wait_semaphores, signal_semaphores,
          ptrs->attrs->copy.source_buffer, ptrs->attrs->copy.source_offset,
          ptrs->attrs->copy.target_buffer, ptrs->attrs->copy.target_offset,
          ptrs->attrs->copy.length, ptrs->attrs->copy.flags);
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
      iree_hal_queue_t* dispatch_queue = stream->queue;
      if (iree_any_bit_set(ptrs->attrs->dispatch.flags,
                           IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
        IREE_RETURN_IF_ERROR(
            iree_hal_streaming_stream_select_cooperative_queue_locked(
                stream, &dispatch_queue));
      }
      iree_hal_buffer_ref_list_t bindings_list = {
          .count = ptrs->attrs->dispatch.bindings.count,
          .values = ptrs->attrs->dispatch.bindings.values,
      };
      return iree_hal_queue_dispatch(
          dispatch_queue, wait_semaphores, signal_semaphores,
          ptrs->attrs->dispatch.executable,
          iree_hal_executable_function_from_index(
              (uint32_t)ptrs->attrs->dispatch.entry_point),
          ptrs->attrs->dispatch.config, ptrs->attrs->dispatch.constants,
          bindings_list, ptrs->attrs->dispatch.flags);
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE: {
      return iree_hal_queue_execute(stream->queue, wait_semaphores,
                                    signal_semaphores,
                                    ptrs->attrs->execute.command_buffer,
                                    iree_hal_buffer_binding_table_empty(),
                                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
      ptrs->attrs->host_call.args[0] = (uint64_t)ptrs->attrs->host_call.fn;
      ptrs->attrs->host_call.args[1] =
          (uint64_t)ptrs->attrs->host_call.user_data;
      ptrs->attrs->host_call.args[2] =
          (uint64_t)ptrs->attrs->host_call.deferred_fn;
      ptrs->attrs->host_call.args[3] = 0;
      return iree_hal_queue_host_call(
          stream->queue, wait_semaphores, signal_semaphores,
          iree_hal_make_host_call(iree_hal_streaming_graph_host_callback, NULL),
          ptrs->attrs->host_call.args, ptrs->attrs->host_call.flags);
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH: {
      iree_hal_streaming_graph_exec_t* child_exec =
          ptrs->attrs->child_graph.exec;
      // Only the child's block 0 carries this block's waits, which are
      // themselves behind the launch tail; a record inside the child sits in a
      // single-block partition that either is block 0 or chains back to it, so
      // the tail carries down unchanged.
      return iree_hal_streaming_graph_exec_submit_blocks_locked(
          child_exec, stream, launch_stream_tail_value, wait_semaphores,
          signal_semaphores, deferred_releases, launch_frontier,
          out_external_signals_reachable);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported block type %u", block->type);
  }
}

// |launch_stream_tail_value| is the value on |stream|'s timeline that the whole
// launch waits behind, or 0 when the stream had never submitted. Only block 0
// carries the launch's external waits, and the extra workstream blocks of the
// first partition wait on nothing, so a block is behind the tail only when it
// chains back to block 0. An event record node is never recordable and so gets
// a partition of its own holding a single block, which either is block 0 or
// waits on every signal of the partition ahead of it, back to block 0.
static iree_status_t iree_hal_streaming_graph_exec_submit_blocks_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_launch_release_list_t* deferred_releases,
    iree_hal_streaming_graph_launch_frontier_t* launch_frontier,
    bool* out_external_signals_reachable) {
  enum {
    IREE_HAL_STREAMING_GRAPH_STACK_BASE_VALUE_COUNT = 64,
    IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT = 16,
  };
  *out_external_signals_reachable = false;
  const iree_host_size_t frontier_start = launch_frontier->count;

  for (uint32_t i = 0; i < exec->semaphore_count; i++) {
    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_query(exec->semaphores[i], &current_value));
    exec->semaphore_base_values[i] =
        iree_max(exec->semaphore_base_values[i], current_value);
  }

  if (exec->block_count == 0) {
    if (external_wait_semaphores.count == 0 &&
        external_signal_semaphores.count == 0) {
      *out_external_signals_reachable = true;
      return iree_ok_status();
    }
    iree_status_t status = iree_hal_queue_barrier(
        stream->queue, external_wait_semaphores, external_signal_semaphores,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    *out_external_signals_reachable = iree_status_is_ok(status);
    return status;
  }

  iree_status_t status = iree_ok_status();
  uint64_t stack_base_values[IREE_HAL_STREAMING_GRAPH_STACK_BASE_VALUE_COUNT];
  uint64_t* new_base_values = NULL;
  bool free_base_values = false;
  if (exec->semaphore_count > 0) {
    iree_host_size_t base_values_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            exec->semaphore_count, sizeof(uint64_t), &base_values_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph semaphore base value size overflow");
    }
    if (exec->semaphore_count <=
        IREE_HAL_STREAMING_GRAPH_STACK_BASE_VALUE_COUNT) {
      new_base_values = stack_base_values;
    } else {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          exec->host_allocator, base_values_size, (void**)&new_base_values));
      free_base_values = true;
    }
    memcpy(new_base_values, exec->semaphore_base_values, base_values_size);
  }

  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < exec->block_count;
       block_index++) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[block_index];

    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);

    iree_host_size_t required_frontier_capacity = 1;
    if (block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              required_frontier_capacity,
              ptrs.attrs->child_graph.exec->launch_frontier_capacity,
              &required_frontier_capacity))) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "child graph launch frontier overflows");
        break;
      }
    }
    if (IREE_UNLIKELY(launch_frontier->count > launch_frontier->capacity ||
                      required_frontier_capacity >
                          launch_frontier->capacity - launch_frontier->count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "graph launch frontier capacity exceeded");
      break;
    }

    uint64_t completion_current_value = 0;
    status =
        iree_hal_semaphore_query(exec->block_completion_semaphores[block_index],
                                 &completion_current_value);
    if (!iree_status_is_ok(status)) break;
    const uint64_t completion_base_value =
        iree_max(exec->block_completion_base_values[block_index],
                 completion_current_value);
    if (IREE_UNLIKELY(completion_base_value == UINT64_MAX)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "graph block completion timeline overflow");
      break;
    }
    const uint64_t completion_value = completion_base_value + 1;

    const bool block_waits_event =
        block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
    const bool block_records_event =
        block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD;
    const iree_host_size_t total_wait_count =
        block->wait_semaphore_count +
        (block_index == 0 ? external_wait_semaphores.count : 0) +
        (block_waits_event ? 1 : 0);
    const iree_host_size_t total_signal_count =
        (iree_host_size_t)block->signal_semaphore_count + 1;
    iree_host_size_t total_semaphores = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            total_wait_count, total_signal_count, &total_semaphores))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "graph launch semaphore count overflow");
      break;
    }

    iree_hal_semaphore_t*
        stack_semaphore_array[IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT];
    uint64_t stack_value_array[IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT];
    iree_hal_semaphore_t** semaphore_array = NULL;
    uint64_t* value_array = NULL;
    bool free_semaphore_array = false;
    bool free_value_array = false;
    if (total_semaphores > 0) {
      if (total_semaphores <= IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT) {
        semaphore_array = stack_semaphore_array;
        value_array = stack_value_array;
      } else {
        iree_host_size_t semaphore_array_size = 0;
        iree_host_size_t value_array_size = 0;
        if (IREE_UNLIKELY(
                !iree_host_size_checked_mul(total_semaphores,
                                            sizeof(iree_hal_semaphore_t*),
                                            &semaphore_array_size) ||
                !iree_host_size_checked_mul(total_semaphores, sizeof(uint64_t),
                                            &value_array_size))) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "graph launch semaphore list allocation size overflow");
          break;
        }
        status =
            iree_allocator_malloc(exec->host_allocator, semaphore_array_size,
                                  (void**)&semaphore_array);
        if (iree_status_is_ok(status)) {
          free_semaphore_array = true;
          status = iree_allocator_malloc(exec->host_allocator, value_array_size,
                                         (void**)&value_array);
          free_value_array = iree_status_is_ok(status);
        }
        if (!iree_status_is_ok(status)) {
          iree_allocator_free(exec->host_allocator, semaphore_array);
          break;
        }
      }
    }

    iree_hal_semaphore_t** wait_sems = semaphore_array;
    uint64_t* wait_vals = value_array;
    iree_hal_semaphore_t** signal_sems =
        semaphore_array ? semaphore_array + total_wait_count : NULL;
    uint64_t* signal_vals = value_array ? value_array + total_wait_count : NULL;

    iree_host_size_t wait_count = 0;
    if (block_index == 0) {
      for (iree_host_size_t i = 0; i < external_wait_semaphores.count; ++i) {
        wait_sems[wait_count] = external_wait_semaphores.semaphores[i];
        wait_vals[wait_count] = external_wait_semaphores.payload_values[i];
        ++wait_count;
      }
    }
    for (uint16_t i = 0; i < block->wait_semaphore_count; i++) {
      const uint16_t semaphore_index = ptrs.wait_semaphore_indices[i];
      const uint32_t delta = ptrs.wait_payload_deltas[i];
      wait_sems[wait_count] = exec->semaphores[semaphore_index];
      wait_vals[wait_count] =
          exec->semaphore_base_values[semaphore_index] + delta;
      ++wait_count;
    }
    // Claim stable cleanup storage before taking an event reference or
    // submitting a record that displaces one. Allocation failure therefore
    // leaves event state unchanged and owns nothing new.
    iree_hal_streaming_launch_release_t* event_release = NULL;
    if (block_waits_event || block_records_event) {
      status = iree_hal_streaming_launch_release_list_push_empty(
          deferred_releases, &event_release);
    }
    // Retained across the submission so a concurrent record of the same event
    // cannot drop the last reference to the timeline this block names. Unlike
    // iree_hal_streaming_stream_wait_event, the point's stream ordering is not
    // filed in |stream|'s memory reuse ledger; a missing entry only withholds
    // allocations from reuse, so the omission costs reuse and can never grant
    // it. The reference is released after the exec and stream locks.
    if (block_waits_event && iree_status_is_ok(status)) {
      iree_hal_streaming_event_acquire_recorded_point(
          ptrs.attrs->event.event, &event_release->recorded_point);
      // A wait on an event with no submitted record has nothing to wait for.
      if (event_release->recorded_point.semaphore) {
        wait_sems[wait_count] = event_release->recorded_point.semaphore;
        wait_vals[wait_count] = event_release->recorded_point.value;
        ++wait_count;
      }
    }

    iree_host_size_t signal_count = 0;
    for (uint16_t i = 0; i < block->signal_semaphore_count; i++) {
      const uint16_t semaphore_index = ptrs.signal_semaphore_indices[i];
      const uint32_t delta = ptrs.signal_payload_deltas[i];
      signal_sems[signal_count] = exec->semaphores[semaphore_index];
      signal_vals[signal_count] =
          exec->semaphore_base_values[semaphore_index] + delta;
      new_base_values[semaphore_index] = signal_vals[signal_count];
      ++signal_count;
    }
    signal_sems[signal_count] = exec->block_completion_semaphores[block_index];
    signal_vals[signal_count] = completion_value;
    ++signal_count;

    // A record node marks the point its own block reaches, which is the block's
    // first signal value: the partitioner gives a record node a partition of
    // its own, so that value is signaled exactly when the node's dependencies
    // complete. Every block signals its own completion timeline, so a record
    // block always has a value to name.
    if (block_records_event && IREE_UNLIKELY(signal_count == 0)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "event record block signals no timeline value");
    }

    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_count,
        .semaphores = wait_sems,
        .payload_values = wait_vals,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = signal_count,
        .semaphores = signal_sems,
        .payload_values = signal_vals,
    };
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_recorded_point_t record_point = {0};
      if (block_records_event) {
        // The block's first signal is the point the record names. When that
        // signal is the launching stream's own timeline the point is exactly a
        // stream point; otherwise it is internal to the launch and all that is
        // known about the stream is that the launch waited behind its tail.
        const bool signals_launch_stream_timeline =
            signal_sems[0] == stream->timeline_semaphore;
        record_point = (iree_hal_streaming_recorded_point_t){
            .semaphore = signal_sems[0],
            .value = signal_vals[0],
            .ordered_after_stream_id = stream->stream_id,
            .ordered_after_stream_value = signals_launch_stream_timeline
                                              ? signal_vals[0]
                                              : launch_stream_tail_value,
        };
      }
      if (iree_status_is_ok(status)) {
        bool child_external_signals_reachable = false;
        iree_status_t block_status = iree_hal_streaming_graph_submit_block(
            block, &ptrs, stream, launch_stream_tail_value, wait_semaphores,
            signal_semaphores, &record_point, deferred_releases,
            launch_frontier, &child_external_signals_reachable);
        const bool block_completion_reachable =
            block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
                ? child_external_signals_reachable
                : iree_status_is_ok(block_status);
        if (block_completion_reachable) {
          exec->block_completion_base_values[block_index] = completion_value;
          launch_frontier->semaphores[launch_frontier->count] =
              exec->block_completion_semaphores[block_index];
          launch_frontier->values[launch_frontier->count] = completion_value;
          ++launch_frontier->count;
        }
        if (block_completion_reachable &&
            block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER &&
            ptrs.attrs->barrier.transfer_mem_alloc_on_accept) {
          iree_hal_streaming_graph_mem_alloc_transfer_to_context(
              ptrs.attrs->barrier.source_mem_alloc_node);
          ptrs.attrs->barrier.transfer_mem_alloc_on_accept = false;
        }
        // A rejected block signals nothing, so the event keeps its old point.
        // The commit stays inside this iteration because later blocks in the
        // same launch wait on the committed point. The recording stream is
        // left alone: it carries capture state and a launch is not a capture.
        if (block_records_event && block_completion_reachable) {
          event_release->graph =
              iree_hal_streaming_event_commit_recorded_point_deferred(
                  ptrs.attrs->event.event, record_point,
                  &event_release->recorded_point);
        }
        status = block_status;
      }
    }
    if (free_value_array) {
      iree_allocator_free(exec->host_allocator, value_array);
    }
    if (free_semaphore_array) {
      iree_allocator_free(exec->host_allocator, semaphore_array);
    }
  }

  // The base values advance even when a block failed to submit: blocks that did
  // submit have already signaled their new values, and a value must name
  // exactly one submission. Nothing rejects a duplicate signal, so rewinding
  // the bases would make the next launch re-signal those values silently, and
  // its blocks would find their waits already satisfied and run ahead of the
  // work they were ordered behind.
  if (exec->semaphore_count > 0) {
    memcpy(exec->semaphore_base_values, new_base_values,
           exec->semaphore_count * sizeof(uint64_t));
  }
  if (free_base_values) {
    iree_allocator_free(exec->host_allocator, new_base_values);
  }

  const iree_host_size_t accepted_frontier_count =
      launch_frontier->count - frontier_start;
  if (accepted_frontier_count > 0) {
    iree_hal_semaphore_list_t accepted_frontier = {
        .count = accepted_frontier_count,
        .semaphores = launch_frontier->semaphores + frontier_start,
        .payload_values = launch_frontier->values + frontier_start,
    };
    iree_status_t closure_status = iree_hal_queue_barrier(
        stream->queue, accepted_frontier, external_signal_semaphores,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    if (iree_status_is_ok(closure_status)) {
      *out_external_signals_reachable = true;
    } else if (iree_status_is_ok(status)) {
      status = closure_status;
    } else {
      iree_status_ignore(closure_status);
    }
  }
  return status;
}

static bool iree_hal_streaming_graph_exec_frontier_drain_finished(void* arg) {
  iree_hal_streaming_graph_exec_t* exec = (iree_hal_streaming_graph_exec_t*)arg;
  return iree_atomic_load(&exec->active_frontier_draining,
                          iree_memory_order_acquire) == 0;
}

// Classifies one infinite-wait result without conflating an execution failure
// with failure to perform the wait. A persistent semaphore query failure is a
// terminal timeline result; a healthy query means the exact point is still
// live and must remain published for a later retry.
static bool iree_hal_streaming_graph_exec_classify_wait_result(
    iree_hal_semaphore_t* semaphore, iree_status_t* inout_status) {
  if (iree_status_is_ok(*inout_status)) return true;
  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(semaphore, &current_value);
  if (iree_status_is_ok(query_status)) return false;
  iree_status_ignore(*inout_status);
  *inout_status = query_status;
  return true;
}

// Waits for and clears the launch named by |exec|. A reachable stream closure
// is represented by |active_launch_stream|. When closure enqueue failed after
// accepting blocks, exactly one waiter drains their explicit completion
// points while other operations await its notification. The caller holds
// |exec->mutex| on entry and again on every return path. No user callback or
// final stream release runs while the executable mutex is held.
static iree_status_t
iree_hal_streaming_graph_exec_wait_for_active_launch_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_active_launch_wait_callback_t wait_callback,
    void* wait_callback_user_data, iree_status_t* out_execution_status,
    bool fail_if_destroyed) {
  IREE_ASSERT_ARGUMENT(out_execution_status);
  *out_execution_status = iree_ok_status();
  iree_status_t accumulated_status = iree_ok_status();
  while (exec->active_launch_stream || exec->active_frontier_count > 0) {
    if (exec->active_launch_stream) {
      iree_status_t wait_status = wait_callback
                                      ? wait_callback(wait_callback_user_data)
                                      : iree_ok_status();
      iree_hal_streaming_stream_t* active_stream = exec->active_launch_stream;
      const uint64_t active_value = exec->active_launch_value;
      iree_hal_streaming_stream_retain(active_stream);
      iree_slim_mutex_unlock(&exec->mutex);

      if (iree_status_is_ok(wait_status)) {
        wait_status = iree_hal_semaphore_wait(
            active_stream->timeline_semaphore, active_value,
            iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
      }
      const bool wait_is_terminal =
          iree_hal_streaming_graph_exec_classify_wait_result(
              active_stream->timeline_semaphore, &wait_status);

      iree_slim_mutex_lock(&exec->mutex);
      bool release_active_stream_ownership = false;
      if (wait_is_terminal && exec->active_launch_stream == active_stream &&
          exec->active_launch_value == active_value) {
        exec->active_launch_stream = NULL;
        exec->active_launch_value = 0;
        // A successful closure proves every accepted block completion point
        // preceding it is terminal. A failed closure proves no such thing: in
        // that case keep the explicit frontier and drain it below before any
        // snapshot resource can be replaced or released.
        if (iree_status_is_ok(wait_status)) {
          exec->active_frontier_count = 0;
        }
        release_active_stream_ownership = true;
      }
      iree_slim_mutex_unlock(&exec->mutex);
      if (release_active_stream_ownership) {
        iree_hal_streaming_stream_release(active_stream);
      }
      iree_hal_streaming_stream_release(active_stream);
      iree_slim_mutex_lock(&exec->mutex);

      if (!wait_is_terminal) {
        iree_status_ignore(accumulated_status);
        return wait_status;
      }
      if (!iree_status_is_ok(wait_status)) {
        if (iree_status_is_ok(accumulated_status)) {
          accumulated_status = wait_status;
        } else {
          iree_status_ignore(wait_status);
        }
      }
    } else if (iree_atomic_load(&exec->active_frontier_draining,
                                iree_memory_order_relaxed) == 0) {
      iree_status_t wait_status = wait_callback
                                      ? wait_callback(wait_callback_user_data)
                                      : iree_ok_status();
      iree_atomic_store(&exec->active_frontier_draining, 1,
                        iree_memory_order_release);
      const iree_host_size_t frontier_count = exec->active_frontier_count;
      iree_slim_mutex_unlock(&exec->mutex);

      bool frontier_is_terminal = iree_status_is_ok(wait_status);
      for (iree_host_size_t i = 0; frontier_is_terminal && i < frontier_count;
           ++i) {
        iree_status_t point_status = iree_hal_semaphore_wait(
            exec->launch_frontier_semaphores[i],
            exec->launch_frontier_values[i], iree_infinite_timeout(),
            IREE_ASYNC_WAIT_FLAG_NONE);
        const bool point_is_terminal =
            iree_hal_streaming_graph_exec_classify_wait_result(
                exec->launch_frontier_semaphores[i], &point_status);
        if (!point_is_terminal) {
          iree_status_ignore(wait_status);
          wait_status = point_status;
          frontier_is_terminal = false;
          break;
        }
        if (!iree_status_is_ok(point_status)) {
          if (iree_status_is_ok(wait_status)) {
            wait_status = point_status;
          } else {
            iree_status_ignore(point_status);
          }
        }
      }

      iree_slim_mutex_lock(&exec->mutex);
      if (frontier_is_terminal) exec->active_frontier_count = 0;
      iree_atomic_store(&exec->active_frontier_draining, 0,
                        iree_memory_order_release);
      iree_slim_mutex_unlock(&exec->mutex);
      iree_notification_post(&exec->active_frontier_notification,
                             IREE_ALL_WAITERS);
      iree_slim_mutex_lock(&exec->mutex);
      if (!frontier_is_terminal) {
        iree_status_ignore(accumulated_status);
        return wait_status;
      }
      if (!iree_status_is_ok(wait_status)) {
        if (iree_status_is_ok(accumulated_status)) {
          accumulated_status = wait_status;
        } else {
          iree_status_ignore(wait_status);
        }
      }
    } else {
      iree_slim_mutex_unlock(&exec->mutex);
      iree_notification_await(
          &exec->active_frontier_notification,
          iree_hal_streaming_graph_exec_frontier_drain_finished, exec,
          iree_infinite_timeout());
      iree_slim_mutex_lock(&exec->mutex);
    }
    if (fail_if_destroyed && (exec->is_destroyed || exec->destroy_pending)) {
      iree_status_ignore(accumulated_status);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }
  *out_execution_status = accumulated_status;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_exec_launch_impl(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Mutex needed for launch as multiple threads can submit at once.
  // It also protects the executable semaphore base values below, which
  // are reused and advanced across launches of the same executable graph.
  iree_slim_mutex_lock(&exec->mutex);
  // Every launch retains executable snapshot resources until its exact stream
  // closure completes. This also serializes graph-memory launches that share
  // one allocation and deferred host calls that borrow snapshot-owned data.
  iree_status_t execution_status = iree_ok_status();
  iree_status_t active_launch_status =
      iree_hal_streaming_graph_exec_wait_for_active_launch_locked(
          exec, NULL, NULL, &execution_status,
          /*fail_if_destroyed=*/true);
  if (iree_status_is_ok(active_launch_status)) {
    active_launch_status = execution_status;
  } else {
    iree_status_ignore(execution_status);
  }
  if (!iree_status_is_ok(active_launch_status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return active_launch_status;
  }

  // Every mutable-state decision must be made after the final reacquire above:
  // update and destroy are permitted to run while an active launch waits.
  if (exec->is_destroyed || exec->destroy_pending) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_status_t state_status =
      iree_hal_streaming_graph_validate_kernel_modules(exec->graph);
  if (!iree_status_is_ok(state_status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return state_status;
  }
  // Handle empty graph - nothing to do.
  if (exec->block_count == 0) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  // Every record this executable holds names an event of |exec->context|, so on
  // a stream of any other context iree_hal_streaming_event_enqueue_record would
  // refuse each of them in turn. Deciding the same question here refuses the
  // launch before it submits any of the graph, where the walk below breaks on
  // the first refusal and leaves the blocks ahead of the record in flight with
  // nothing signaling the launching stream's timeline.
  if (exec->records_events && stream->context != exec->context) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "an event can only be recorded on a stream of the context that "
        "created it");
  }
  if (exec->has_unfreed_graph_alloc_nodes && exec->launch_count > 0 &&
      !iree_all_bits_set(
          exec->flags,
          IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "graph contains live allocation nodes from a previous launch");
  }

  // Flush while holding executable state so the lock order remains lifecycle
  // admission -> executable -> stream on every launch path.
  iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
  if (!iree_status_is_ok(flush_status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return flush_status;
  }

  // Required lock order: binding lifecycle gate -> executable mutex -> stream.
  // Mutation and launch paths that need both lifecycle admission and this
  // mutex acquire them in that order. Pointer validation follows graph-memory
  // serialization and precedes queue acceptance of every block.
  iree_status_t pointer_status =
      iree_hal_streaming_graph_exec_validate_pointers_locked(exec,
                                                             stream->context);
  if (!iree_status_is_ok(pointer_status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return pointer_status;
  }

  // Event waits and records retain or replace event references while the
  // launch locks are held. They collect the references here and release them
  // once those locks have been dropped.
  iree_hal_streaming_launch_release_list_t deferred_releases;
  iree_hal_streaming_launch_release_list_initialize(exec->host_allocator,
                                                    &deferred_releases);

  iree_slim_mutex_lock(&stream->mutex);

  // Reserve the next stream timeline value while holding the stream lock so
  // concurrent host threads cannot submit same-stream work with the same wait
  // or signal value. The graph waits on the current stream tail, not the last
  // completion observed by the host; HIP stream ordering requires repeated
  // graph launches on the same stream to execute FIFO even when the caller does
  // not synchronize between launches.
  uint64_t stream_wait_value = 0;
  uint64_t stream_signal_value = 0;
  iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &stream_wait_value, &stream_signal_value);

  iree_hal_semaphore_t* wait_semaphore = stream->timeline_semaphore;
  uint64_t wait_payload_value = stream_wait_value;
  iree_hal_semaphore_t* signal_semaphore = stream->timeline_semaphore;
  uint64_t signal_payload_value = stream_signal_value;
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = stream_wait_value > 0 ? 1 : 0,
      .semaphores = &wait_semaphore,
      .payload_values = &wait_payload_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &signal_semaphore,
      .payload_values = &signal_payload_value,
  };
  iree_hal_streaming_graph_launch_frontier_t launch_frontier = {
      .semaphores = exec->launch_frontier_semaphores,
      .values = exec->launch_frontier_values,
      .capacity = exec->launch_frontier_capacity,
      .count = 0,
  };
  bool stream_signal_reachable = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_submit_blocks_locked(
        exec, stream, stream_wait_value, wait_semaphores, signal_semaphores,
        &deferred_releases, &launch_frontier, &stream_signal_reachable);
  }

  iree_hal_streaming_stream_t* retired_active_stream = NULL;
  if (stream_signal_reachable) {
    stream->pending_value = stream_signal_value;
    retired_active_stream = exec->active_launch_stream;
    iree_hal_streaming_stream_retain(stream);
    exec->active_launch_stream = stream;
    exec->active_launch_value = stream_signal_value;
    // Retain the exact accepted frontier as a fallback until the stream
    // closure is observed successfully. Queue failures may fail a closure
    // timeline before its waits become terminal; a failed closure alone is
    // therefore not proof that snapshot-owned resources are no longer in use.
    exec->active_frontier_count = launch_frontier.count;
    ++exec->launch_count;
  } else if (launch_frontier.count > 0) {
    if (iree_status_is_ok(status)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "graph launch closure rejected after accepting blocks");
    }
    // No later stream operation may pass the unsignaled reservation while the
    // accepted prefix is still active. Permanently fail the reserved timeline
    // and publish it as the stream tail; executable mutation separately drains
    // every accepted block completion before releasing snapshot resources.
    iree_hal_semaphore_list_fail(signal_semaphores, iree_status_clone(status));
    stream->pending_value = stream_signal_value;
    exec->active_frontier_count = launch_frontier.count;
    ++exec->launch_count;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&exec->mutex);
  iree_hal_streaming_stream_release(retired_active_stream);
  // Final event-point and captured-graph releases can retire pooled resources,
  // destroy semaphores, synchronize contexts, and relock this stream. Records
  // that displaced references remain committed even when a later block failed,
  // so deferred cleanup runs on both paths.
  iree_hal_streaming_launch_release_list_deinitialize(&deferred_releases);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_context_operation_begin(stream->context));
  iree_status_t status =
      iree_hal_streaming_graph_exec_launch_impl(exec, stream);
  iree_hal_streaming_context_operation_end(stream->context);
  return status;
}

static bool iree_hal_streaming_graph_node_is_visible(
    const iree_hal_streaming_graph_node_t* node) {
  return node && (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) == 0;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_node_public_identity(
    iree_hal_streaming_graph_node_t* node) {
  return node && node->executable_source_node
             ? (iree_hal_streaming_graph_node_t*)node->executable_source_node
             : node;
}

static iree_host_size_t iree_hal_streaming_graph_visible_node_count(
    const iree_hal_streaming_graph_t* graph) {
  iree_host_size_t visible_count = 0;
  if (!graph) return 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      if (iree_hal_streaming_graph_node_is_visible(block->nodes[i])) {
        ++visible_count;
      }
    }
  }
  return visible_count;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_visible_node_at_index(
    const iree_hal_streaming_graph_t* graph, iree_host_size_t visible_index) {
  iree_host_size_t current_visible_index = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (!iree_hal_streaming_graph_node_is_visible(node)) continue;
      if (current_visible_index == visible_index) return node;
      ++current_visible_index;
    }
  }
  return NULL;
}

static bool iree_hal_streaming_graph_visible_index_of_node(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node,
    iree_host_size_t* out_visible_index) {
  iree_host_size_t current_visible_index = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* candidate = block->nodes[i];
      if (!iree_hal_streaming_graph_node_is_visible(candidate)) continue;
      if (candidate == node) {
        *out_visible_index = current_visible_index;
        return true;
      }
      ++current_visible_index;
    }
  }
  return false;
}

static iree_host_size_t iree_hal_streaming_graph_visible_dependency_count(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node) {
  iree_host_size_t dependency_count = 0;
  for (uint32_t i = 0; i < node->dependency_count; ++i) {
    if (iree_hal_streaming_graph_node_is_visible(node->dependencies[i])) {
      ++dependency_count;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->to == node &&
        iree_hal_streaming_graph_node_is_visible(edge->from)) {
      ++dependency_count;
    }
  }
  return dependency_count;
}

static bool iree_hal_streaming_graph_has_visible_dependency(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node,
    iree_host_size_t dependency_visible_index) {
  for (uint32_t i = 0; i < node->dependency_count; ++i) {
    iree_host_size_t visible_index = 0;
    if (iree_hal_streaming_graph_visible_index_of_node(
            graph, node->dependencies[i], &visible_index) &&
        visible_index == dependency_visible_index) {
      return true;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    iree_host_size_t visible_index = 0;
    if (edge->to == node &&
        iree_hal_streaming_graph_visible_index_of_node(graph, edge->from,
                                                       &visible_index) &&
        visible_index == dependency_visible_index) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_streaming_graph_visible_dependencies_match(
    const iree_hal_streaming_graph_t* old_graph,
    const iree_hal_streaming_graph_node_t* old_node,
    const iree_hal_streaming_graph_t* new_graph,
    const iree_hal_streaming_graph_node_t* new_node) {
  if (iree_hal_streaming_graph_visible_dependency_count(old_graph, old_node) !=
      iree_hal_streaming_graph_visible_dependency_count(new_graph, new_node)) {
    return false;
  }
  for (uint32_t i = 0; i < old_node->dependency_count; ++i) {
    iree_host_size_t old_dependency_visible_index = 0;
    if (!iree_hal_streaming_graph_visible_index_of_node(
            old_graph, old_node->dependencies[i],
            &old_dependency_visible_index)) {
      continue;
    }
    if (!iree_hal_streaming_graph_has_visible_dependency(
            new_graph, new_node, old_dependency_visible_index)) {
      return false;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = old_graph->additional_edges;
       edge; edge = edge->next) {
    if (edge->to != old_node) continue;
    iree_host_size_t old_dependency_visible_index = 0;
    if (!iree_hal_streaming_graph_visible_index_of_node(
            old_graph, edge->from, &old_dependency_visible_index)) {
      continue;
    }
    if (!iree_hal_streaming_graph_has_visible_dependency(
            new_graph, new_node, old_dependency_visible_index)) {
      return false;
    }
  }
  return true;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_first_visible_kernel_node(
    const iree_hal_streaming_graph_t* graph) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hal_streaming_graph_node_is_visible(node) &&
          node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        return node;
      }
    }
  }
  return iree_hal_streaming_graph_visible_node_at_index(graph, 0);
}

static bool iree_hal_streaming_graph_memcpy_update_is_compatible(
    const iree_hal_streaming_graph_memcpy_node_attrs_t* old_attrs,
    const iree_hal_streaming_graph_memcpy_node_attrs_t* new_attrs) {
  return old_attrs->hip_kind == new_attrs->hip_kind &&
         old_attrs->size == new_attrs->size &&
         old_attrs->execution_extent_width ==
             new_attrs->execution_extent_width &&
         old_attrs->execution_extent_height ==
             new_attrs->execution_extent_height &&
         old_attrs->execution_extent_depth ==
             new_attrs->execution_extent_depth &&
         old_attrs->execution_dst_pitch == new_attrs->execution_dst_pitch &&
         old_attrs->execution_src_pitch == new_attrs->execution_src_pitch &&
         old_attrs->execution_dst_ysize == new_attrs->execution_dst_ysize &&
         old_attrs->execution_src_ysize == new_attrs->execution_src_ysize &&
         old_attrs->hip_extent_width == new_attrs->hip_extent_width &&
         old_attrs->hip_extent_height == new_attrs->hip_extent_height &&
         old_attrs->hip_extent_depth == new_attrs->hip_extent_depth &&
         old_attrs->hip_dst_pitch == new_attrs->hip_dst_pitch &&
         old_attrs->hip_src_pitch == new_attrs->hip_src_pitch &&
         old_attrs->hip_dst_xsize == new_attrs->hip_dst_xsize &&
         old_attrs->hip_src_xsize == new_attrs->hip_src_xsize &&
         old_attrs->hip_dst_ysize == new_attrs->hip_dst_ysize &&
         old_attrs->hip_src_ysize == new_attrs->hip_src_ysize;
}

static bool iree_hal_streaming_graph_memset_update_is_compatible(
    const iree_hal_streaming_graph_memset_node_attrs_t* old_attrs,
    const iree_hal_streaming_graph_memset_node_attrs_t* new_attrs) {
  return old_attrs->pattern_size == new_attrs->pattern_size &&
         old_attrs->count == new_attrs->count &&
         old_attrs->hip_width == new_attrs->hip_width &&
         old_attrs->hip_height == new_attrs->hip_height &&
         old_attrs->hip_pitch == new_attrs->hip_pitch;
}

static bool iree_hal_streaming_graph_update_is_compatible(
    const iree_hal_streaming_graph_t* old_graph,
    iree_host_size_t old_visible_node_count,
    const iree_hal_streaming_graph_t* new_graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  if (old_graph->context != new_graph->context) {
    *out_error_node = iree_hal_streaming_graph_node_public_identity(
        iree_hal_streaming_graph_first_visible_kernel_node(new_graph));
    *out_result =
        IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_UNSUPPORTED_FUNCTION_CHANGE;
    return false;
  }
  const iree_host_size_t new_visible_node_count =
      iree_hal_streaming_graph_visible_node_count(new_graph);
  if (old_visible_node_count != new_visible_node_count) {
    *out_error_node = NULL;
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
    return false;
  }

  for (iree_host_size_t i = 0; i < old_visible_node_count; ++i) {
    iree_hal_streaming_graph_node_t* old_node =
        iree_hal_streaming_graph_visible_node_at_index(old_graph, i);
    iree_hal_streaming_graph_node_t* new_node =
        iree_hal_streaming_graph_visible_node_at_index(new_graph, i);
    if (!old_node || !new_node) {
      *out_error_node = iree_hal_streaming_graph_node_public_identity(new_node);
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
      return false;
    }
    if (old_node->type != new_node->type) {
      *out_error_node = iree_hal_streaming_graph_node_public_identity(new_node);
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED;
      return false;
    }
    if (!iree_hal_streaming_graph_visible_dependencies_match(
            old_graph, old_node, new_graph, new_node)) {
      *out_error_node = iree_hal_streaming_graph_node_public_identity(new_node);
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
      return false;
    }

    switch (old_node->type) {
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
        if (!iree_hal_streaming_graph_memcpy_update_is_compatible(
                &old_node->attrs.memcpy, &new_node->attrs.memcpy)) {
          *out_error_node =
              iree_hal_streaming_graph_node_public_identity(new_node);
          *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED;
          return false;
        }
        break;
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
        if (!iree_hal_streaming_graph_memset_update_is_compatible(
                &old_node->attrs.memset, &new_node->attrs.memset)) {
          *out_error_node =
              iree_hal_streaming_graph_node_public_identity(new_node);
          *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED;
          return false;
        }
        break;
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
        if (!old_node->attrs.child_graph.graph ||
            !new_node->attrs.child_graph.graph ||
            !iree_hal_streaming_graph_update_is_compatible(
                old_node->attrs.child_graph.graph,
                iree_hal_streaming_graph_visible_node_count(
                    old_node->attrs.child_graph.graph),
                new_node->attrs.child_graph.graph, out_error_node,
                out_result)) {
          *out_error_node =
              iree_hal_streaming_graph_node_public_identity(new_node);
          return false;
        }
        break;
      default:
        break;
    }
  }
  return true;
}

iree_status_t iree_hal_streaming_graph_exec_state_set_child_graph(
    iree_hal_streaming_graph_exec_state_guard_t* guard,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_t* child_snapshot,
    iree_hal_streaming_graph_t** out_graph_to_release) {
  IREE_ASSERT_ARGUMENT(out_graph_to_release);
  *out_graph_to_release = NULL;
  if (!guard || !guard->exec || !guard->deferred_cleanup || !node ||
      node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH ||
      !node->attrs.child_graph.graph || !child_snapshot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_graph_node_t* error_node = NULL;
  iree_hal_streaming_graph_exec_update_result_t update_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  iree_hal_streaming_graph_t* old_child = node->attrs.child_graph.graph;
  if (!iree_hal_streaming_graph_update_is_compatible(
          old_child, iree_hal_streaming_graph_visible_node_count(old_child),
          child_snapshot, &error_node, &update_result)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "child graph topology is incompatible");
  }
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_validate_child_graph(
      node->graph, child_snapshot));

  const bool old_parent_edge_counted =
      node->attrs.child_graph.parent_edge_counted;
  iree_hal_streaming_graph_retain(child_snapshot);
  node->attrs.child_graph.graph = child_snapshot;
  node->attrs.child_graph.parent_edge_counted = false;
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_from_template_locked(guard);
  if (iree_status_is_ok(status)) {
    if (old_parent_edge_counted) {
      iree_slim_mutex_lock(&old_child->graph_memory_state_mutex);
      IREE_ASSERT(old_child->child_parent_edge_count > 0);
      --old_child->child_parent_edge_count;
      iree_slim_mutex_unlock(&old_child->graph_memory_state_mutex);
    }
    *out_graph_to_release = old_child;
  } else {
    node->attrs.child_graph.graph = old_child;
    node->attrs.child_graph.parent_edge_counted = old_parent_edge_counted;
    *out_graph_to_release = child_snapshot;
  }
  return status;
}

static void iree_hal_streaming_graph_exec_set_graph_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph) {
  exec->graph = graph;
  exec->uses_graph_memory_nodes = graph && graph->has_graph_memory_nodes;
}

static iree_status_t iree_hal_streaming_graph_exec_update_impl(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_error_node);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Snapshot preparation performs every allocation and user-object retain
  // before executable state is locked. The prepared graph remains private
  // unless the locked compatibility/rebuild transaction succeeds.
  iree_hal_streaming_graph_t* snapshot = NULL;
  iree_status_t status = iree_hal_streaming_graph_snapshot(graph, &snapshot);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_hal_streaming_graph_exec_state_guard_t guard = {0};
  status = iree_hal_streaming_graph_exec_rebuild_state_begin(exec, NULL, NULL,
                                                             &guard);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_release(snapshot);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  status = iree_hal_streaming_graph_exec_state_update(
      &guard, snapshot, out_error_node, out_result);
  iree_hal_streaming_graph_exec_state_end(&guard);
  iree_hal_streaming_graph_release(snapshot);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_state_update(
    iree_hal_streaming_graph_exec_state_guard_t* guard,
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(guard);
  IREE_ASSERT_ARGUMENT(guard->exec);
  IREE_ASSERT_ARGUMENT(guard->deferred_cleanup);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_error_node);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_error_node = NULL;
  *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  iree_hal_streaming_graph_exec_t* exec = guard->exec;
  iree_hal_streaming_graph_exec_deferred_cleanup_t* cleanup =
      (iree_hal_streaming_graph_exec_deferred_cleanup_t*)
          guard->deferred_cleanup;

  iree_status_t status =
      iree_hal_streaming_graph_validate_kernel_modules(exec->graph);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_validate_kernel_modules(graph);
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_hal_streaming_graph_t* old_graph_memory_owner =
      exec->graph->graph_memory_owner_graph
          ? exec->graph->graph_memory_owner_graph
          : exec->graph;
  iree_slim_mutex_lock(&old_graph_memory_owner->graph_memory_state_mutex);
  const bool old_graph_has_transferred_unfreed_allocation =
      old_graph_memory_owner->has_transferred_unfreed_allocation;
  iree_slim_mutex_unlock(&old_graph_memory_owner->graph_memory_state_mutex);
  if (old_graph_has_transferred_unfreed_allocation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "an executable with a transferred graph allocation cannot be updated");
  }
  if (!iree_hal_streaming_graph_update_is_compatible(
          exec->graph, exec->instantiated_visible_node_count, graph,
          out_error_node, out_result)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "graph update is not compatible");
  }

  // Enabled state is keyed by corresponding public visible topology. Hidden
  // implementation nodes never receive public enable slots and therefore
  // cannot shift the state when staged-copy orientation changes.
  for (iree_host_size_t i = 0; i < exec->instantiated_visible_node_count; ++i) {
    iree_hal_streaming_graph_node_t* old_node =
        iree_hal_streaming_graph_visible_node_at_index(exec->graph, i);
    iree_hal_streaming_graph_node_t* new_node =
        iree_hal_streaming_graph_visible_node_at_index(graph, i);
    IREE_ASSERT(old_node && new_node);
    if (old_node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED) {
      new_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED;
    } else {
      new_node->flags &= ~IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED;
    }
  }

  iree_hal_streaming_graph_t* old_graph = exec->graph;
  const bool old_uses_graph_memory_nodes = exec->uses_graph_memory_nodes;
  bool acquired_new_graph_memory_slot = false;
  if (graph != old_graph) {
    cleanup->compiled_state.graph = old_graph;
    iree_status_t slot_status =
        iree_hal_streaming_graph_memory_acquire_exec_slot(graph);
    if (!iree_status_is_ok(slot_status)) {
      return slot_status;
    }
    acquired_new_graph_memory_slot = graph->has_graph_memory_nodes;
    iree_hal_streaming_graph_retain(graph);
    iree_hal_streaming_graph_exec_set_graph_locked(exec, graph);
  }

  status = iree_hal_streaming_graph_exec_rebuild_from_template_locked(guard);
  if (iree_status_is_ok(status)) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS;
    if (graph != old_graph) {
      cleanup->graph = old_graph;
      cleanup->release_graph_memory_slot = old_uses_graph_memory_nodes;
    }
  } else if (graph != old_graph) {
    iree_hal_streaming_graph_exec_set_graph_locked(exec, old_graph);
    exec->uses_graph_memory_nodes = old_uses_graph_memory_nodes;
    cleanup->graph = graph;
    cleanup->release_graph_memory_slot = acquired_new_graph_memory_slot;
  }

  return status;
}

iree_status_t iree_hal_streaming_graph_exec_update(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_context_operation_begin(exec->context));
  iree_status_t status = iree_hal_streaming_graph_exec_update_impl(
      exec, graph, out_error_node, out_result);
  iree_hal_streaming_context_operation_end(exec->context);
  return status;
}
