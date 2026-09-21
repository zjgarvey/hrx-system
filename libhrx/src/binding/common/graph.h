// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
#define IREE_EXPERIMENTAL_STREAMING_GRAPH_H_

#include "common/internal.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Graph partitioning and instantiation
//===----------------------------------------------------------------------===//

// Chained block for growing arrays without reallocation.
typedef struct iree_hal_streaming_node_block_t {
  struct iree_hal_streaming_node_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_streaming_graph_node_t* nodes[];
} iree_hal_streaming_node_block_t;

// Edge structure for additional dependencies added after node creation.
typedef struct iree_hal_streaming_graph_edge_t {
  struct iree_hal_streaming_graph_edge_t* next;
  iree_hal_streaming_graph_node_t* from;  // Dependency (must complete first)
  iree_hal_streaming_graph_node_t* to;    // Dependent (waits for 'from')
} iree_hal_streaming_graph_edge_t;

// Graph-owned host allocation that backs staged host/device copy nodes.
typedef struct iree_hal_streaming_graph_owned_host_allocation_t {
  // Next allocation in the graph-owned singly-linked list.
  struct iree_hal_streaming_graph_owned_host_allocation_t* next;
  // Streaming buffer wrapper for this host-visible allocation.
  iree_hal_streaming_buffer_t* buffer;
  // Never-reused wrapper identity paired with the graph-owned wrapper pin.
  uint64_t buffer_id;
  // Host pointer returned by the streaming host allocation.
  void* host_ptr;
  // Device pointer associated with the host-visible allocation.
  iree_hal_streaming_deviceptr_t device_ptr;
  // Allocation size in bytes.
  iree_device_size_t size;
} iree_hal_streaming_graph_owned_host_allocation_t;

typedef iree_status_t (*iree_hal_streaming_graph_user_object_retain_fn_t)(
    void* object, uint64_t count);
typedef void (*iree_hal_streaming_graph_user_object_release_fn_t)(
    void* object, uint64_t count);

typedef struct iree_hal_streaming_graph_user_object_ref_t {
  // Next retained user object in the graph-owned singly-linked list.
  struct iree_hal_streaming_graph_user_object_ref_t* next;
  // Opaque API object retained by this graph template.
  void* object;
  // Number of references currently held by this graph template.
  uint64_t count;
  // Callback used when cloning this graph template.
  iree_hal_streaming_graph_user_object_retain_fn_t retain;
  // Callback used when destroying this graph template or releasing references.
  iree_hal_streaming_graph_user_object_release_fn_t release;
} iree_hal_streaming_graph_user_object_ref_t;

// Stable shared ownership for an eager graph-memory allocation. Public graph
// nodes and immutable executable snapshots retain this record independently;
// the exact backing pin is transferred or finalized only once.
struct iree_hal_streaming_graph_mem_allocation_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_streaming_buffer_t* buffer;
  uint64_t buffer_id;
  bool owns_device_allocation;
};

// Graph structure (template).
typedef struct iree_hal_streaming_graph_t {
  iree_atomic_ref_count_t ref_count;

  // Arena allocator for all graph allocations.
  iree_arena_allocator_t arena;
  iree_allocator_t arena_allocator;

  // Graph nodes stored in chained blocks.
  iree_hal_streaming_node_block_t* node_blocks;
  iree_hal_streaming_node_block_t* current_node_block;
  iree_host_size_t node_count;
  // Number of direct child graph nodes in this graph template.
  iree_host_size_t child_graph_node_count;
  // Process-unique identifier used for graph debug output and clone provenance.
  uint64_t debug_id;
  // Debug identifier of the graph this template was cloned from, or zero.
  uint64_t clone_source_graph_debug_id;
  // Next stable source ID assigned to graph nodes created in this template.
  uint32_t next_clone_source_node_index;

  // Root nodes stored in chained blocks.
  iree_hal_streaming_node_block_t* root_blocks;
  iree_hal_streaming_node_block_t* current_root_block;
  iree_host_size_t root_count;

  // Additional edges added after node creation via hipGraphAddDependencies.
  iree_hal_streaming_graph_edge_t* additional_edges;
  iree_host_size_t additional_edge_count;

  // Host allocations owned by this graph template.
  iree_hal_streaming_graph_owned_host_allocation_t* owned_host_allocations;
  // Opaque user objects retained by this graph template.
  iree_hal_streaming_graph_user_object_ref_t* user_object_refs;

  // Public graph retained by an immutable executable snapshot. This preserves
  // raw public node identity storage without making snapshot launches or
  // rebuilds consult mutable public topology. NULL for public templates and
  // public clones.
  struct iree_hal_streaming_graph_t* executable_source_graph;
  // Public graph whose memory-node slot and transfer state govern this
  // snapshot. Aliases |executable_source_graph| and owns no additional ref.
  struct iree_hal_streaming_graph_t* graph_memory_owner_graph;

  // True when the graph contains HIP memory allocation or free nodes.
  bool has_graph_memory_nodes;
  // Serializes the graph-memory executable slot and one-way transfer state.
  iree_slim_mutex_t graph_memory_state_mutex;
  // Number of live executable graphs instantiated from this memory-node graph.
  uint32_t active_graph_memory_exec_count;
  // Number of committed child-graph nodes in other templates that retain this
  // graph. Nonzero forbids subsequently adding graph-memory nodes.
  uint32_t child_parent_edge_count;
  // True after an accepted unmatched allocation transferred backing ownership
  // to the context registry. Reinstantiation is rejected because this eager
  // allocation implementation cannot recreate a fresh allocation epoch.
  bool has_transferred_unfreed_allocation;

  // Graph creation flags.
  uint32_t flags;
  // Streaming context that owns graph resources.
  iree_hal_streaming_context_t* context;

  // Host allocator used for graph object allocation.
  iree_allocator_t host_allocator;
} iree_hal_streaming_graph_t;

// Transfers an unmatched allocation node's independent wrapper pin to the
// still-published context allocation registry after the node's exact launch
// barrier has been accepted. Idempotent for already-transferred nodes.
void iree_hal_streaming_graph_mem_alloc_transfer_to_context(
    iree_hal_streaming_graph_node_t* node);

void iree_hal_streaming_graph_mem_allocation_initialize(
    iree_hal_streaming_graph_mem_allocation_t* allocation,
    iree_hal_streaming_buffer_t* buffer, uint64_t buffer_id);
void iree_hal_streaming_graph_mem_allocation_retain(
    iree_hal_streaming_graph_mem_allocation_t* allocation);
void iree_hal_streaming_graph_mem_allocation_release(
    iree_hal_streaming_graph_mem_allocation_t* allocation);

// Creates an immutable executable template with exact public-node identity
// mappings. Unlike the public clone API this supports graph-memory nodes.
iree_status_t iree_hal_streaming_graph_snapshot(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_graph_t** out_graph);

// Type of partition - determines how nodes are executed.
enum iree_hal_streaming_graph_partition_type_e {
  // Can go in command buffer (count 1 may also be optimizable into a queue op).
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE = 0,
  // Must be separate host call.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL,
  // Must be launched as a nested executable graph.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_GRAPH,
  // Must be submitted as a direct queue dispatch operation.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_DISPATCH,
  // Barrier node.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY,
};
typedef uint8_t iree_hal_streaming_graph_partition_type_t;

// Describes a partition of nodes that can be executed together.
typedef struct iree_hal_streaming_graph_partition_t {
  // Index into sorted_nodes array.
  uint32_t start_index;
  uint32_t count;
  iree_hal_streaming_graph_partition_type_t type;
  // Number of independent workstreams (~1-4).
  uint8_t stream_count;
} iree_hal_streaming_graph_partition_t;

// Chained block for growing partition arrays without reallocation.
typedef struct iree_hal_streaming_graph_partition_block_t {
  struct iree_hal_streaming_graph_partition_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_streaming_graph_partition_t partitions[];
} iree_hal_streaming_graph_partition_block_t;

iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec);

iree_status_t iree_hal_streaming_graph_exec_instantiate_from_template(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count);

iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template(
    iree_hal_streaming_graph_exec_t* exec);

#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
// Synchronous test observer invoked immediately before each compiled graph
// block attempts a HAL queue operation. Installation and graph launches must
// be externally serialized.
typedef void (*iree_hal_streaming_graph_test_queue_submission_observer_t)(
    void* user_data);
void iree_hal_streaming_graph_test_set_queue_submission_observer(
    iree_hal_streaming_graph_test_queue_submission_observer_t observer,
    void* user_data);
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

// Augmented node for sorting and partitioning.
typedef struct iree_hal_streaming_graph_sort_node_t {
  // Pointer to original node.
  iree_hal_streaming_graph_node_t* node;
  // Index in original linked list.
  uint32_t original_index;
  // Position in topological order.
  uint32_t sorted_index;
  // Maximum sorted index of all dependencies.
  uint32_t max_dependency_index;
  // Assigned partition ID.
  uint32_t partition_id;
  // For topological sort.
  uint16_t in_degree;
  // Cached from node->type.
  uint8_t type;
  // Workstream within partition (~4).
  uint8_t stream_id;
} iree_hal_streaming_graph_sort_node_t;
static_assert(sizeof(iree_hal_streaming_graph_sort_node_t) <= 32,
              "really want 2 per cache line");

// A produced schedule.
// References are into the arena used during scheduling and only valid as long
// as it is.
typedef struct iree_hal_streaming_graph_schedule_t {
  // Sorted nodes with some additional information from analysis.
  iree_hal_streaming_graph_sort_node_t* sorted_nodes;
  // A map of original unsorted graph node_index to sorted_nodes index.
  uint32_t* node_index_map;
  // Partition descriptors in execution order.
  iree_hal_streaming_graph_partition_t* partitions;
  // Total number of partitions.
  iree_host_size_t partition_count;
  // Total number of blocks across all partitions.
  iree_host_size_t block_count;
} iree_hal_streaming_graph_schedule_t;

// Unified scheduler that performs topological sorting, partitioning, and
// workstream detection in an efficient three-phase algorithm.
//
// Phase 1: Linearize nodes and detect if already sorted
// Phase 2: Topological sort if needed (with fast path)
// Phase 3: Partition into executable blocks with workstream detection
//
// Returns sorted nodes array and partition descriptors with workstream info
// allocated from the provided |arena|.
// |out_total_block_count| is the total number of graph blocks required
// calculated as the number of non-recordable partitions + the total number of
// streams in all recordable partitions.
// |additional_edges| is an optional linked list of extra dependencies added
// after node creation (can be NULL if none).
iree_status_t iree_hal_streaming_graph_schedule_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    const uint8_t* disabled_nodes, iree_host_size_t disabled_node_count,
    iree_hal_streaming_graph_edge_t* additional_edges,
    iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_schedule_t* out_schedule);

// Checks whether |child_graph| may be the child graph of a node in
// |parent_graph|: the two must be distinct, must belong to the same context,
// and |child_graph| must not reach |parent_graph| through its own child graph
// nodes. Instantiating a node's child graph instantiates that graph's own child
// graph nodes in turn, so containment that leads back to |parent_graph| would
// recurse until the stack ran out.
iree_status_t iree_hal_streaming_graph_validate_child_graph(
    iree_hal_streaming_graph_t* parent_graph,
    iree_hal_streaming_graph_t* child_graph);

// Rejects adding a graph-memory node while |graph| is retained as a child by
// any committed parent node.
iree_status_t iree_hal_streaming_graph_validate_memory_node_addition(
    iree_hal_streaming_graph_t* graph);

// Adds dependencies between nodes in the graph.
// For each index i in [0, count), adds an edge from from_nodes[i] to
// to_nodes[i], meaning to_nodes[i] will wait for from_nodes[i] to complete.
iree_status_t iree_hal_streaming_graph_add_dependencies(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** from_nodes,
    iree_hal_streaming_graph_node_t** to_nodes, iree_host_size_t count);

// Allocates a host-visible staging buffer owned by |graph|.
iree_status_t iree_hal_streaming_graph_allocate_host_staging(
    iree_hal_streaming_graph_t* graph, iree_device_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

// Allocates host staging without publishing it into |graph|. Callers must
// commit or abort the returned ownership record exactly once.
iree_status_t iree_hal_streaming_graph_prepare_host_staging(
    iree_hal_streaming_graph_t* graph, iree_device_size_t size,
    iree_hal_streaming_graph_owned_host_allocation_t** out_allocation,
    iree_hal_streaming_buffer_t** out_buffer);
void iree_hal_streaming_graph_commit_host_staging(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_owned_host_allocation_t* allocation);
void iree_hal_streaming_graph_abort_host_staging(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_owned_host_allocation_t* allocation);

// Finds a committed staging allocation by its buffer identity.
iree_hal_streaming_graph_owned_host_allocation_t*
iree_hal_streaming_graph_find_host_staging(
    iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_buffer_t* buffer);

// Removes a committed staging allocation from |graph| without releasing it.
// Returns true when detached. The caller must abort the detached allocation
// exactly once, and may do so after dropping an outer serialization lock.
bool iree_hal_streaming_graph_detach_host_staging(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_owned_host_allocation_t* allocation);

// Detaches and releases a committed staging allocation.
void iree_hal_streaming_graph_release_host_staging(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_owned_host_allocation_t* allocation);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
