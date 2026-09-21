// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#include "common/graph.h"
#include "common/internal.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if !defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
#error "graph_exec_test requires the instrumented common provider"
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

namespace {

// Marks the flag |user_data| names as reached. Used as the body of host calls
// whose only purpose is to say whether the stream got that far.
void SetFlag(void* user_data) {
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
}

void IncrementSubmissionCount(void* user_data) {
  ++*static_cast<int*>(user_data);
}

// A host call that exposes exact accepted/running and release transitions.
// Tests wait only on predicates and never use a deadline.
struct HostCallLatch {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return entered; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cv.notify_all();
  }
};

void BlockingHostCall(void* user_data) {
  auto* latch = static_cast<HostCallLatch*>(user_data);
  std::unique_lock<std::mutex> lock(latch->mutex);
  latch->entered = true;
  latch->cv.notify_all();
  latch->cv.wait(lock, [&] { return latch->released; });
}

// Reports that executable mutation reached its active-launch wait. The
// callback runs under executable serialization and therefore must not reenter
// the executable or binding API.
struct ActiveLaunchWaitLatch {
  std::mutex mutex;
  std::condition_variable cv;
  bool reached = false;

  void WaitUntilReached() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return reached; });
  }
};

iree_status_t NotifyActiveLaunchWait(void* user_data) {
  auto* latch = static_cast<ActiveLaunchWaitLatch*>(user_data);
  std::lock_guard<std::mutex> lock(latch->mutex);
  latch->reached = true;
  latch->cv.notify_all();
  return iree_ok_status();
}

struct FailActiveLaunchWaitOnce {
  bool called = false;
};

iree_status_t InjectActiveLaunchWaitFailure(void* user_data) {
  auto* state = static_cast<FailActiveLaunchWaitOnce*>(user_data);
  EXPECT_FALSE(state->called);
  state->called = true;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "injected active launch wait failure");
}

iree_status_t FailDeferredHostCall(void* user_data,
                                   iree_hal_host_call_context_t* context) {
  (void)context;
  auto* call_count = static_cast<std::atomic<int>*>(user_data);
  call_count->fetch_add(1, std::memory_order_acq_rel);
  return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                          "injected terminal graph execution failure");
}

// Runs |cleanup| when it leaves scope. A test body builds its handles across a
// run of fatal assertions and a fatal assertion returns from the body, so the
// releases have to sit somewhere that return cannot skip.
template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  // Called once, when this object is destroyed.
  Cleanup cleanup_;
};
// Required despite matching the implicit guide: clang builds this file with
// -Wctad-maybe-unsupported under -Werror, and that warning fires wherever a
// template's arguments are deduced and the template declares no guide of its
// own. A guard deduces because its cleanup is a lambda, whose type no
// declaration can spell.
template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

struct FailNthAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  std::atomic<int> fail_on_allocation{0};
  std::atomic<int> allocation_attempt_count{0};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailNthAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      const int allocation_attempt =
          allocator->allocation_attempt_count.fetch_add(
              1, std::memory_order_acq_rel) +
          1;
      if (allocation_attempt ==
          allocator->fail_on_allocation.load(std::memory_order_acquire)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &FailNthAllocator::Control};
  }
};

void ExpectArenaMatchesCheckpoint(const iree_arena_allocator_t& arena,
                                  const iree_arena_checkpoint_t& checkpoint) {
  EXPECT_EQ(checkpoint.allocation_head, arena.allocation_head);
  EXPECT_EQ(checkpoint.block_head, arena.block_head);
  EXPECT_EQ(checkpoint.block_tail, arena.block_tail);
  EXPECT_EQ(checkpoint.total_allocation_size, arena.total_allocation_size);
  EXPECT_EQ(checkpoint.used_allocation_size, arena.used_allocation_size);
  EXPECT_EQ(checkpoint.block_bytes_remaining, arena.block_bytes_remaining);
}

// Runs streaming graph launches against the host CPU device. Launches take the
// same block submit path they take on an accelerator; the event records they
// enqueue resolve to queue barriers because the device advertises no timestamp
// domain for the records to write into.
class GraphExecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

    // Stands in for the registry entry global initialization builds around an
    // enumerated accelerator: contexts take their HAL device from the entry and
    // graphs carve their node storage out of its block pool.
    memset(&device_entry_, 0, sizeof(device_entry_));
    device_entry_.hrx_device = hrx_device;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     device_allocator_.AsAllocator(),
                                     &device_entry_.block_pool);

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, iree_allocator_system(), &context_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
        /*priority=*/0, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  // Adds |count| event record nodes recording |events| to |graph|, each
  // depending on the node |tail| names. Chaining them pins the schedule's node
  // order and puts each node in a partition, and so a block, of its own.
  // Leaves |tail| naming the last node added.
  void AppendEventRecordChain(iree_hal_streaming_graph_t* graph,
                              iree_hal_streaming_event_t* const* events,
                              iree_host_size_t count,
                              iree_hal_streaming_graph_node_t** tail) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
          graph, *tail ? tail : nullptr, *tail ? 1 : 0,
          IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD, events[i], &node));
      ASSERT_NE(node, nullptr);
      *tail = node;
    }
  }

  // Host allocator backing the fixture's device block pool.
  FailNthAllocator device_allocator_;
  // Registry entry backing |context_|; outlives every context created from it.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context owning the graphs, events, and streams each test builds.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Stream every launch submits on.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Set by the host-call node a graph carries. A fixture member rather than a
  // local because no semaphore edge a test can name joins a block an aborted
  // launch left in flight; hrx_cpu_shutdown() above is what drains the workers,
  // so the flag has to outlive the test body it is read in.
  std::atomic<bool> graph_host_node_ran_{false};
};

TEST_F(GraphExecTest, RootNodeBlockAllocationFailureIsAtomic) {
  enum class NodeKind { kKernel, kMemcpy, kEvent, kChild };
  for (NodeKind kind : {NodeKind::kKernel, NodeKind::kMemcpy, NodeKind::kEvent,
                        NodeKind::kChild}) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    // Make the node, node-list block, and root-list block independent backing
    // allocations, then reject the third. This targets the only insertion
    // allocation that occurs after the node attrs have taken their references.
    FailNthAllocator allocator;
    allocator.fail_on_allocation = 3;
    iree_arena_block_pool_t fault_pool;
    iree_arena_block_pool_initialize(/*total_block_size=*/64,
                                     allocator.AsAllocator(), &fault_pool);
    iree_arena_deinitialize(&graph->arena);
    iree_arena_initialize(&fault_pool, &graph->arena);
    graph->arena_allocator = iree_arena_allocator(&graph->arena);

    iree_hal_streaming_module_t module = {};
    iree_atomic_ref_count_init(&module.ref_count);
    iree_atomic_store(&module.public_live, 1, iree_memory_order_release);
    iree_hal_streaming_symbol_t symbol = {};
    symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
    symbol.module = &module;
    iree_hal_streaming_dispatch_params_t dispatch_params = {};

    iree_hal_streaming_buffer_t dst_buffer = {};
    iree_hal_streaming_buffer_t src_buffer = {};
    iree_atomic_ref_count_init(&dst_buffer.ref_count);
    iree_atomic_ref_count_init(&src_buffer.ref_count);
    dst_buffer.size = 1;
    src_buffer.size = 1;
    const iree_hal_streaming_buffer_ref_t dst_ref = {
        .buffer = &dst_buffer,
        .offset = 0,
    };
    const iree_hal_streaming_buffer_ref_t src_ref = {
        .buffer = &src_buffer,
        .offset = 0,
    };

    iree_hal_streaming_event_t* event = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
        iree_allocator_system(), &event));
    iree_hal_streaming_graph_t* child_graph = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &child_graph));

    const int32_t module_refs = iree_atomic_ref_count_load(&module.ref_count);
    const int32_t dst_refs = iree_atomic_ref_count_load(&dst_buffer.ref_count);
    const int32_t src_refs = iree_atomic_ref_count_load(&src_buffer.ref_count);
    const int32_t event_refs = iree_atomic_ref_count_load(&event->ref_count);
    const int32_t child_refs =
        iree_atomic_ref_count_load(&child_graph->ref_count);

    iree_hal_streaming_graph_node_t* node = nullptr;
    iree_status_t status = iree_ok_status();
    switch (kind) {
      case NodeKind::kKernel:
        status = iree_hal_streaming_graph_add_kernel_node(
            graph, nullptr, 0, &symbol, &dispatch_params, &node);
        break;
      case NodeKind::kMemcpy:
        status = iree_hal_streaming_graph_add_copy_buffer_node(
            graph, nullptr, 0, dst_ref, src_ref, 1, &node);
        break;
      case NodeKind::kEvent:
        status = iree_hal_streaming_graph_add_event_node(
            graph, nullptr, 0, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
            event, &node);
        break;
      case NodeKind::kChild:
        status = iree_hal_streaming_graph_add_child_graph_node(
            graph, nullptr, 0, child_graph, &node);
        break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(nullptr, node);
    EXPECT_EQ(0u, graph->node_count);
    EXPECT_EQ(0u, graph->root_count);
    EXPECT_EQ(nullptr, graph->node_blocks);
    EXPECT_EQ(nullptr, graph->root_blocks);
    EXPECT_EQ(0u, graph->next_clone_source_node_index);
    EXPECT_EQ(module_refs, iree_atomic_ref_count_load(&module.ref_count));
    EXPECT_EQ(dst_refs, iree_atomic_ref_count_load(&dst_buffer.ref_count));
    EXPECT_EQ(src_refs, iree_atomic_ref_count_load(&src_buffer.ref_count));
    EXPECT_EQ(event_refs, iree_atomic_ref_count_load(&event->ref_count));
    EXPECT_EQ(child_refs, iree_atomic_ref_count_load(&child_graph->ref_count));

    // The unused arena allocations do not poison the graph. A later insertion
    // and instantiation must observe one ordinary root and no zombie node.
    iree_hal_streaming_graph_node_t* replacement = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(graph, nullptr, 0,
                                                           &replacement));
    ASSERT_NE(nullptr, replacement);
    EXPECT_EQ(1u, graph->node_count);
    EXPECT_EQ(1u, graph->root_count);
    iree_hal_streaming_graph_exec_t* exec = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
    iree_hal_streaming_graph_exec_release(exec);

    iree_hal_streaming_event_release(event);
    iree_hal_streaming_graph_release(child_graph);
    iree_hal_streaming_graph_release(graph);
    iree_arena_block_pool_deinitialize(&fault_pool);
  }
}

TEST_F(GraphExecTest,
       BatchParameterRebuildFailuresRestoreBytesAndKeepArenaUsageFlat) {
  static constexpr iree_host_size_t kEmptyNodeCount = 2048;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
  });
  ScopeExit reset_allocator([&] {
    device_allocator_.fail_on_allocation.store(0, std::memory_order_release);
  });

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  std::array<uint8_t, 4> initial_params = {1, 2, 3, 4};
  std::array<uint8_t, 4> initial_param_array = {5, 6, 7, 8};
  iree_hal_streaming_graph_node_t* batch_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_batch_mem_op_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      initial_params.data(), initial_params.size(), initial_param_array.data(),
      initial_param_array.size(), &batch_node));
  for (iree_host_size_t i = 0; i < kEmptyNodeCount; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
        graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
        /*out_node=*/nullptr));
  }
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  iree_hal_streaming_graph_t* snapshot_graph = nullptr;
  iree_arena_checkpoint_t checkpoint = {};
  iree_hal_streaming_graph_batch_mem_op_node_attrs_t original_attrs = {};
  std::array<uint8_t, 4> original_params;
  std::array<uint8_t, 4> original_param_array;
  {
    iree_hal_streaming_graph_exec_state_guard_t guard = {};
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_begin(exec, &guard));
    ScopeExit end_state(
        [&] { iree_hal_streaming_graph_exec_state_end(&guard); });
    iree_hal_streaming_graph_node_t* snapshot_node =
        iree_hal_streaming_graph_exec_state_resolve_node(&guard, batch_node);
    ASSERT_NE(nullptr, snapshot_node);
    snapshot_graph = snapshot_node->graph;
    if (snapshot_graph->arena.block_bytes_remaining < 256) {
      void* padding = nullptr;
      IREE_ASSERT_OK(iree_arena_allocate(
          &snapshot_graph->arena,
          snapshot_graph->arena.block_bytes_remaining + 1, &padding));
    }
    ASSERT_GE(snapshot_graph->arena.block_bytes_remaining, 256u);
    checkpoint = iree_arena_checkpoint_save(&snapshot_graph->arena);
    original_attrs = snapshot_node->attrs.batch_mem_op;
    memcpy(original_params.data(), original_attrs.params,
           original_params.size());
    memcpy(original_param_array.data(), original_attrs.param_array,
           original_param_array.size());
  }

  std::array<uint8_t, 64> new_params;
  std::array<uint8_t, 64> new_param_array;
  new_params.fill(0xA5);
  new_param_array.fill(0x5A);
  for (int iteration = 0; iteration < 4; ++iteration) {
    device_allocator_.allocation_attempt_count.store(0,
                                                     std::memory_order_release);
    device_allocator_.fail_on_allocation.store(1, std::memory_order_release);
    iree_status_t status =
        iree_hal_streaming_graph_exec_set_batch_mem_op_node_params(
            exec, batch_node, new_params.data(), new_params.size(),
            new_param_array.data(), new_param_array.size());
    device_allocator_.fail_on_allocation.store(0, std::memory_order_release);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(1, device_allocator_.allocation_attempt_count.load(
                     std::memory_order_acquire));

    iree_hal_streaming_graph_exec_state_guard_t guard = {};
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_begin(exec, &guard));
    ScopeExit end_state(
        [&] { iree_hal_streaming_graph_exec_state_end(&guard); });
    iree_hal_streaming_graph_node_t* snapshot_node =
        iree_hal_streaming_graph_exec_state_resolve_node(&guard, batch_node);
    ASSERT_NE(nullptr, snapshot_node);
    const iree_hal_streaming_graph_batch_mem_op_node_attrs_t& attrs =
        snapshot_node->attrs.batch_mem_op;
    EXPECT_EQ(original_attrs.params, attrs.params);
    EXPECT_EQ(original_attrs.params_size, attrs.params_size);
    EXPECT_EQ(original_attrs.params_capacity, attrs.params_capacity);
    EXPECT_EQ(0, memcmp(original_params.data(), attrs.params,
                        original_params.size()));
    EXPECT_EQ(original_attrs.param_array, attrs.param_array);
    EXPECT_EQ(original_attrs.param_array_size, attrs.param_array_size);
    EXPECT_EQ(original_attrs.param_array_capacity, attrs.param_array_capacity);
    EXPECT_EQ(0, memcmp(original_param_array.data(), attrs.param_array,
                        original_param_array.size()));
    ExpectArenaMatchesCheckpoint(snapshot_graph->arena, checkpoint);
  }

  device_allocator_.allocation_attempt_count.store(0,
                                                   std::memory_order_release);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_set_batch_mem_op_node_params(
      exec, batch_node, new_params.data(), new_params.size(),
      new_param_array.data(), new_param_array.size()));
  {
    iree_hal_streaming_graph_exec_state_guard_t guard = {};
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_begin(exec, &guard));
    ScopeExit end_state(
        [&] { iree_hal_streaming_graph_exec_state_end(&guard); });
    iree_hal_streaming_graph_node_t* snapshot_node =
        iree_hal_streaming_graph_exec_state_resolve_node(&guard, batch_node);
    ASSERT_NE(nullptr, snapshot_node);
    const iree_hal_streaming_graph_batch_mem_op_node_attrs_t& committed_attrs =
        snapshot_node->attrs.batch_mem_op;
    EXPECT_NE(original_attrs.params, committed_attrs.params);
    EXPECT_EQ(new_params.size(), committed_attrs.params_size);
    EXPECT_EQ(new_params.size(), committed_attrs.params_capacity);
    EXPECT_EQ(0, memcmp(new_params.data(), committed_attrs.params,
                        new_params.size()));
    EXPECT_NE(original_attrs.param_array, committed_attrs.param_array);
    EXPECT_EQ(new_param_array.size(), committed_attrs.param_array_size);
    EXPECT_EQ(new_param_array.size(), committed_attrs.param_array_capacity);
    EXPECT_EQ(0, memcmp(new_param_array.data(), committed_attrs.param_array,
                        new_param_array.size()));
    EXPECT_GT(snapshot_graph->arena.used_allocation_size,
              checkpoint.used_allocation_size);
  }
}

// A replayed event record ends its event's association with the graph a
// capture-time record left on it, and the launch releases every reference it
// takes over exactly once.
//
// The records are split across a parent graph and a child graph so the child's
// walk claims room in the same storage the parent's does, and there are more of
// them than the sixteen cells one chunk of that storage holds, so the walk has
// to grow it and carry across the growth what it already collected. The test
// keeps no reference to the capture graph, so the launch drops the last one and
// the release of the context the graph retained counts the destruction.
TEST_F(GraphExecTest, ReplayedEventRecordsDropEveryCapturedGraphReference) {
  static constexpr iree_host_size_t kChildEventCount = 7;
  static constexpr iree_host_size_t kParentEventCount = 10;
  std::array<iree_hal_streaming_event_t*, kChildEventCount + kParentEventCount>
      events = {};
  iree_hal_streaming_graph_t* child_graph = nullptr;
  iree_hal_streaming_graph_t* parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands these handles back, on the assertion failure paths as much as on the
  // last line. They start null and a release takes null, so a body cut short
  // hands back only what it reached. The capture graph is not among them: the
  // test drops its reference mid-body on purpose, to leave the events holding
  // the last ones.
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(parent_graph);
    iree_hal_streaming_graph_release(child_graph);
    for (iree_hal_streaming_event_t* event : events) {
      iree_hal_streaming_event_release(event);
    }
  });

  for (iree_hal_streaming_event_t*& event : events) {
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
        &event));
  }

  // Associates every event with one capture graph.
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  for (iree_hal_streaming_event_t* event : events) {
    IREE_ASSERT_OK(iree_hal_streaming_event_record(event, stream_));
  }
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_end_capture(stream_, &captured_graph));
  ASSERT_NE(captured_graph, nullptr);
  for (iree_hal_streaming_event_t* event : events) {
    ASSERT_EQ(event->capture_graph, captured_graph);
  }

  // Leaves the events holding the only references to the capture graph.
  iree_hal_streaming_graph_release(captured_graph);

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &child_graph));
  iree_hal_streaming_graph_node_t* child_tail = nullptr;
  AppendEventRecordChain(child_graph, events.data(), kChildEventCount,
                         &child_tail);
  ASSERT_FALSE(HasFatalFailure());

  // The child graph node leads, so the parent's own record nodes are walked
  // after the child's and on top of the room the child claimed.
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &parent_graph));
  iree_hal_streaming_graph_node_t* parent_tail = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      child_graph, &parent_tail));
  AppendEventRecordChain(parent_graph, events.data() + kChildEventCount,
                         kParentEventCount, &parent_tail);
  ASSERT_FALSE(HasFatalFailure());

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  // A graph holds a reference to the context that created it and drops it when
  // it is destroyed, so one release across the launch is the capture graph and
  // nothing else.
  const int32_t context_references_before =
      iree_atomic_ref_count_load(&context_->ref_count);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream_));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));

  for (iree_hal_streaming_event_t* event : events) {
    EXPECT_EQ(event->capture_graph, nullptr);
  }
  EXPECT_EQ(context_references_before - 1,
            iree_atomic_ref_count_load(&context_->ref_count))
      << "the launch destroyed the capture graph a number of times other than "
         "once";
}

// An instantiated executable takes an event node's event only from the context
// that created the graph, which is the rule the template setters hold every
// other way of naming an event node's event to. A launch relies on it for a
// record node: it answers for every record it holds by comparing the launching
// stream's context with its own, and a record block naming an event of some
// third context would make those two questions different ones. A wait node is
// held to the same rule because the template setter holds both node types to
// it, so retargeting cannot seat an event the template would have refused.
TEST_F(GraphExecTest, ExecEventNodeTakesOnlyItsOwnContextsEvent) {
  iree_hal_streaming_context_t* other_context = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_event_t* replacement = nullptr;
  iree_hal_streaming_event_t* other_context_event = nullptr;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands back whatever was built, on the assertion failure paths as much as
  // on the last line.
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_event_release(other_context_event);
    iree_hal_streaming_event_release(replacement);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_context_release(other_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_, context_flags, iree_allocator_system(), &other_context));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &replacement));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      other_context, IREE_HAL_STREAMING_EVENT_FLAG_NONE,
      iree_allocator_system(), &other_context_event));

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* record_node = nullptr;
  AppendEventRecordChain(graph, &event, /*count=*/1, &record_node);
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_node_t* wait_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
      graph, &record_node, /*dependency_count=*/1,
      IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT, event, &wait_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  iree_status_t status = iree_hal_streaming_graph_exec_set_event_node_event(
      exec, record_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
      other_context_event);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(status))
      << "an executable took a record node event from another context, which a "
         "launch would only find out at the record itself, with the blocks "
         "ahead of it already submitted";
  iree_status_free(status);

  status = iree_hal_streaming_graph_exec_set_event_node_event(
      exec, wait_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
      other_context_event);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(status))
      << "an executable took a wait node event from another context, which the "
         "template setter refuses for a wait node just as it does for a record "
         "one";
  iree_status_free(status);

  // The refusals are the context rule and not a blanket one.
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_set_event_node_event(
      exec, record_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
      replacement));
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_set_event_node_event(
      exec, wait_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
      replacement));
}

TEST_F(GraphExecTest, StateGuardSerializesUpdatesAndRejectsRetiredNodes) {
  iree_hal_streaming_graph_t* first_graph = nullptr;
  iree_hal_streaming_graph_t* second_graph = nullptr;
  iree_hal_streaming_graph_t* third_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(third_graph);
    iree_hal_streaming_graph_release(second_graph);
    iree_hal_streaming_graph_release(first_graph);
  });

  iree_hal_streaming_graph_node_t* first_node = nullptr;
  iree_hal_streaming_graph_node_t* second_node = nullptr;
  iree_hal_streaming_graph_node_t* third_node = nullptr;
  for (auto [graph, node] : {std::pair{&first_graph, &first_node},
                             std::pair{&second_graph, &second_node},
                             std::pair{&third_graph, &third_node}}) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        graph));
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
        *graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &SetFlag,
        &graph_host_node_ran_, node));
  }
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      first_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  iree_hal_streaming_graph_exec_state_guard_t first_guard = {};
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_begin(exec, &first_guard));
  iree_hal_streaming_graph_exec_state_guard_t contending_guard = {};
  EXPECT_FALSE(
      iree_hal_streaming_graph_exec_state_try_begin(exec, &contending_guard));
  EXPECT_NE(iree_hal_streaming_graph_exec_state_resolve_node(&first_guard,
                                                             first_node),
            nullptr);
  iree_hal_streaming_graph_exec_state_end(&first_guard);

  iree_hal_streaming_graph_node_t* error_node = nullptr;
  iree_hal_streaming_graph_exec_update_result_t update_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_update(
      exec, second_graph, &error_node, &update_result));
  EXPECT_EQ(update_result, IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS);

  iree_hal_streaming_graph_exec_state_guard_t second_guard = {};
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_state_begin(exec, &second_guard));
  EXPECT_EQ(iree_hal_streaming_graph_exec_state_resolve_node(&second_guard,
                                                             first_node),
            nullptr);
  EXPECT_NE(iree_hal_streaming_graph_exec_state_resolve_node(&second_guard,
                                                             second_node),
            nullptr);
  iree_hal_streaming_graph_exec_state_end(&second_guard);

  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_update(
      exec, third_graph, &error_node, &update_result));
  EXPECT_EQ(update_result, IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS);
  iree_hal_streaming_graph_exec_state_guard_t third_guard = {};
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_begin(exec, &third_guard));
  EXPECT_EQ(iree_hal_streaming_graph_exec_state_resolve_node(&third_guard,
                                                             second_node),
            nullptr);
  EXPECT_NE(iree_hal_streaming_graph_exec_state_resolve_node(&third_guard,
                                                             third_node),
            nullptr);
  iree_hal_streaming_graph_exec_state_end(&third_guard);

  iree_hal_streaming_graph_exec_state_guard_t disable_guard = {};
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_rebuild_state_begin(
      exec, nullptr, nullptr, &disable_guard));
  iree_hal_streaming_graph_node_t* resolved_third_node =
      iree_hal_streaming_graph_exec_state_resolve_node(&disable_guard,
                                                       third_node);
  ASSERT_NE(resolved_third_node, nullptr);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_set_node_enabled(
      &disable_guard, resolved_third_node, false));
  EXPECT_FALSE(iree_hal_streaming_graph_exec_state_node_is_enabled(
      &disable_guard, resolved_third_node));
  iree_hal_streaming_graph_exec_state_end(&disable_guard);

  iree_hal_streaming_graph_exec_state_guard_t enable_guard = {};
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_rebuild_state_begin(
      exec, nullptr, nullptr, &enable_guard));
  resolved_third_node = iree_hal_streaming_graph_exec_state_resolve_node(
      &enable_guard, third_node);
  ASSERT_NE(resolved_third_node, nullptr);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_state_set_node_enabled(
      &enable_guard, resolved_third_node, true));
  EXPECT_TRUE(iree_hal_streaming_graph_exec_state_node_is_enabled(
      &enable_guard, resolved_third_node));
  iree_hal_streaming_graph_exec_state_end(&enable_guard);
}

TEST_F(GraphExecTest, PartialLaunchFailurePublishesReachablePrefixClosure) {
  static constexpr iree_host_size_t kEventCount = 17;
  std::array<iree_hal_streaming_event_t*, kEventCount> events = {};
  HostCallLatch host_latch;
  ActiveLaunchWaitLatch wait_latch;
  FailNthAllocator exec_allocator;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  ScopeExit release_handles([&] {
    host_latch.Release();
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
    for (iree_hal_streaming_event_t* event : events) {
      iree_hal_streaming_event_release(event);
    }
  });

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* tail = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &BlockingHostCall, &host_latch, &tail));
  for (iree_hal_streaming_event_t*& event : events) {
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
        iree_allocator_system(), &event));
  }
  AppendEventRecordChain(graph, events.data(), events.size(), &tail);
  ASSERT_FALSE(HasFatalFailure());

  // Executables capture the graph allocator at creation. Leave injection
  // disabled while instantiating and restore the public graph immediately;
  // the executable snapshot's private wrapper remains alive through release.
  const iree_allocator_t old_graph_allocator = graph->host_allocator;
  graph->host_allocator = exec_allocator.AsAllocator();
  iree_status_t instantiate_status = iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec);
  graph->host_allocator = old_graph_allocator;
  IREE_ASSERT_OK(instantiate_status);

  // Sixteen event cleanup cells are inline. The first launch allocation is
  // therefore the 17th cell, after the host block and sixteen event blocks
  // have been accepted. Its failure must still enqueue the exact prefix
  // closure on the stream timeline.
  exec_allocator.allocation_attempt_count = 0;
  exec_allocator.fail_on_allocation = 1;
  iree_status_t launch_status =
      iree_hal_streaming_graph_exec_launch(exec, stream_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, launch_status);
  host_latch.WaitUntilEntered();

  std::atomic<bool> rebuild_finished{false};
  iree_status_t rebuild_status = iree_ok_status();
  std::thread rebuild_thread([&] {
    iree_hal_streaming_graph_exec_state_guard_t guard = {};
    rebuild_status = iree_hal_streaming_graph_exec_rebuild_state_begin(
        exec, &NotifyActiveLaunchWait, &wait_latch, &guard);
    if (iree_status_is_ok(rebuild_status)) {
      iree_hal_streaming_graph_exec_state_end(&guard);
    }
    rebuild_finished.store(true, std::memory_order_release);
  });
  wait_latch.WaitUntilReached();
  EXPECT_FALSE(rebuild_finished.load(std::memory_order_acquire));

  host_latch.Release();
  rebuild_thread.join();
  IREE_EXPECT_OK(rebuild_status);
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));

  // The failed attempt did not poison the executable. Its exact closure was
  // drained, so a later launch can safely reuse compiled state and timelines.
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream_));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
}

TEST_F(GraphExecTest, RejectedPrefixClosureDrainsExactAcceptedFrontier) {
  HostCallLatch host_latch;
  ActiveLaunchWaitLatch wait_latch;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  iree_hal_streaming_stream_t* recovery_stream = nullptr;
  ScopeExit release_handles([&] {
    host_latch.Release();
    iree_hal_streaming_stream_release(recovery_stream);
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
  });

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &BlockingHostCall, &host_latch, &host_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  // The block has no initial stream wait and can be accepted independently.
  // A prefailed stream timeline rejects only the root closure signal.
  iree_hal_semaphore_fail(
      stream_->timeline_semaphore,
      iree_make_status(IREE_STATUS_ABORTED, "injected closure rejection"));
  iree_status_t launch_status =
      iree_hal_streaming_graph_exec_launch(exec, stream_);
  IREE_EXPECT_OK(launch_status);
  host_latch.WaitUntilEntered();

  std::atomic<bool> rebuild_finished{false};
  iree_status_t rebuild_status = iree_ok_status();
  std::thread rebuild_thread([&] {
    iree_hal_streaming_graph_exec_state_guard_t guard = {};
    rebuild_status = iree_hal_streaming_graph_exec_rebuild_state_begin(
        exec, &NotifyActiveLaunchWait, &wait_latch, &guard);
    if (iree_status_is_ok(rebuild_status)) {
      iree_hal_streaming_graph_exec_state_end(&guard);
    }
    rebuild_finished.store(true, std::memory_order_release);
  });
  wait_latch.WaitUntilReached();
  EXPECT_FALSE(rebuild_finished.load(std::memory_order_acquire));

  host_latch.Release();
  rebuild_thread.join();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, rebuild_status);
  iree_status_t original_stream_status =
      iree_hal_streaming_stream_synchronize(stream_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, original_stream_status);

  // The failed launch poisons only the stream whose reserved closure could not
  // be submitted. Explicit frontier drain leaves the executable reusable.
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &recovery_stream));
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, recovery_stream));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(recovery_stream));
}

TEST_F(GraphExecTest,
       DirectDestroyWaitFailurePreservesFrontierAndRemainsRetryable) {
  HostCallLatch host_latch;
  ActiveLaunchWaitLatch retry_wait_latch;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  ScopeExit release_handles([&] {
    host_latch.Release();
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
  });

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &BlockingHostCall, &host_latch, &host_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  // Keep one test pin after successful destruction consumes the public edge.
  iree_hal_streaming_graph_exec_retain(exec);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream_));
  host_latch.WaitUntilEntered();

  // A failed wait with a healthy timeline has not proved the accepted launch
  // terminal. Destruction must preserve both its public edge and exact
  // frontier so the same handle can be retried.
  FailActiveLaunchWaitOnce failure;
  iree_status_t first_destroy_status =
      iree_hal_streaming_graph_exec_destroy_handle_with_wait_callback(
          exec, &InjectActiveLaunchWaitFailure, &failure);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, first_destroy_status);
  EXPECT_TRUE(failure.called);

  std::atomic<bool> retry_finished{false};
  iree_status_t retry_status = iree_ok_status();
  std::thread retry_thread([&] {
    retry_status =
        iree_hal_streaming_graph_exec_destroy_handle_with_wait_callback(
            exec, &NotifyActiveLaunchWait, &retry_wait_latch);
    retry_finished.store(true, std::memory_order_release);
  });
  retry_wait_latch.WaitUntilReached();
  EXPECT_FALSE(retry_finished.load(std::memory_order_acquire));

  host_latch.Release();
  retry_thread.join();
  IREE_EXPECT_OK(retry_status);
}

TEST_F(GraphExecTest, DirectDestroyConsumesVerifiedTerminalFrontierInOneCall) {
  std::atomic<int> call_count{0};
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
  });

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_deferred_host_call_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &FailDeferredHostCall, &call_count, &host_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  // Keep one test pin after successful destruction consumes the public edge.
  iree_hal_streaming_graph_exec_retain(exec);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream_));

  // The host block fails both its completion point and the reachable stream
  // closure. Those persistent terminal payloads are execution results, not a
  // reason to leave a fully quiesced public executable handle live for retry.
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
  EXPECT_EQ(call_count.load(std::memory_order_acquire), 1);
  iree_status_t stream_status = iree_hal_streaming_stream_synchronize(stream_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, stream_status);
}

// A child executable's event-record property is transitive to its parent. A
// cross-context launch must reject from the parent preflight, before any block
// reaches a HAL queue operation. The synchronous observer sits at that exact
// boundary, so a zero count distinguishes preflight rejection from a later
// rejection inside the child's block walk.
TEST_F(GraphExecTest, ChildGraphRecordRefusesALaunchOnAnotherContextsStream) {
  iree_hal_streaming_context_t* other_context = nullptr;
  iree_hal_streaming_stream_t* other_stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_graph_t* child_graph = nullptr;
  iree_hal_streaming_graph_t* parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands back whatever was built, on the assertion failure paths as much as
  // on the last line. Releasing the stream and the context is what drains and
  // unregisters them, and the fixture shuts the device down either way.
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(parent_graph);
    iree_hal_streaming_graph_release(child_graph);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(other_stream);
    iree_hal_streaming_context_release(other_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_, context_flags, iree_allocator_system(), &other_context));
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      other_context, other_context->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &other_stream));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &event));

  // The executable's only record node sits in the child graph.
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &child_graph));
  iree_hal_streaming_graph_node_t* child_tail = nullptr;
  AppendEventRecordChain(child_graph, &event, /*count=*/1, &child_tail);
  ASSERT_FALSE(HasFatalFailure());

  // The host node depends on nothing and the child graph node depends on it, so
  // the host call is the block a launch submits first and the child's record
  // the block that would refuse it. The synchronous observer below sees any
  // parent block that reaches the queue-submission boundary.
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &parent_graph));
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &SetFlag,
      &graph_host_node_ran_, &host_node));
  iree_hal_streaming_graph_node_t* child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      parent_graph, &host_node, /*dependency_count=*/1, child_graph,
      &child_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  int queue_submission_count = 0;
  iree_hal_streaming_graph_test_set_queue_submission_observer(
      &IncrementSubmissionCount, &queue_submission_count);
  ScopeExit clear_submission_observer([] {
    iree_hal_streaming_graph_test_set_queue_submission_observer(nullptr,
                                                                nullptr);
  });
  iree_status_t status =
      iree_hal_streaming_graph_exec_launch(exec, other_stream);
  EXPECT_EQ(IREE_STATUS_INCOMPATIBLE, iree_status_code(status))
      << "a launch on another context's stream was accepted for an executable "
         "whose only record sits in a child graph";
  iree_status_free(status);
  EXPECT_EQ(0, queue_submission_count)
      << "cross-context rejection occurred after a graph block reached the "
         "queue-submission boundary";
}

}  // namespace
