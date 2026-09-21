// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::Status;
using ::iree::StatusCode;
using ::iree::testing::status::StatusIs;

void InitializeLiveMockModule(iree_hal_streaming_module_t* module) {
  iree_atomic_ref_count_init(&module->ref_count);
  iree_atomic_store(&module->public_live, 1, iree_memory_order_relaxed);
}

// Owns a dependency-free graph node using the same variable-sized allocation
// shape as production graph construction.
class GraphNodeStorage {
 public:
  GraphNodeStorage() {
    IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(), sizeof(*node_),
                                        (void**)&node_));
    memset(node_, 0, sizeof(*node_));
  }

  ~GraphNodeStorage() { iree_allocator_free(iree_allocator_system(), node_); }

  GraphNodeStorage(const GraphNodeStorage&) = delete;
  GraphNodeStorage& operator=(const GraphNodeStorage&) = delete;

  iree_hal_streaming_graph_node_t* get() const { return node_; }

 private:
  // Allocated graph node header with no trailing dependency pointers.
  iree_hal_streaming_graph_node_t* node_ = nullptr;
};

TEST(GraphTest, KernelParameterUpdateIsFailureAtomic) {
  iree_hal_streaming_module_t module = {};
  InitializeLiveMockModule(&module);
  constexpr size_t kArgumentCount = 3;
  std::array<iree_hal_streaming_parameter_op_t, kArgumentCount> operations = {};
  for (uint16_t i = 0; i < kArgumentCount; ++i) {
    operations[i].copy = {
        /*.size=*/sizeof(uint32_t),
        /*.native_abi_destination_offset=*/
        static_cast<uint16_t>(i * sizeof(uint32_t)),
        /*.source_offset=*/static_cast<uint16_t>(i * sizeof(uint32_t)),
        /*.source_ordinal=*/static_cast<uint16_t>(i),
        /*.constant_destination_offset=*/
        static_cast<uint16_t>(i * sizeof(uint32_t)),
    };
  }

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.module = &module;
  symbol.parameters.buffer_size = kArgumentCount * sizeof(uint32_t);
  symbol.parameters.constant_bytes = kArgumentCount * sizeof(uint32_t);
  symbol.parameters.direct_arg_bytes = kArgumentCount * sizeof(uint32_t);
  symbol.parameters.copy_count = kArgumentCount;
  symbol.parameters.ops = operations.data();

  for (size_t missing_ordinal = 0; missing_ordinal < kArgumentCount;
       ++missing_ordinal) {
    iree_hal_streaming_graph_t graph = {};
    graph.host_allocator = iree_allocator_system();

    std::array<uint8_t, kArgumentCount * sizeof(uint32_t)> constants = {};
    memset(constants.data(), 0xA5, constants.size());
    const std::array<uint8_t, kArgumentCount * sizeof(uint32_t)>
        original_constants = constants;
    iree_hal_streaming_symbol_t previous_symbol = {};
    previous_symbol.module = &module;
    GraphNodeStorage node_storage;
    iree_hal_streaming_graph_node_t& node = *node_storage.get();
    node.graph = &graph;
    node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
    node.attrs.kernel.symbol = &previous_symbol;
    node.attrs.kernel.module = &module;
    node.attrs.kernel.grid_dim[0] = 7;
    node.attrs.kernel.grid_dim[1] = 5;
    node.attrs.kernel.grid_dim[2] = 3;
    node.attrs.kernel.block_dim[0] = 11;
    node.attrs.kernel.block_dim[1] = 13;
    node.attrs.kernel.block_dim[2] = 17;
    node.attrs.kernel.shared_memory_bytes = 19;
    node.attrs.kernel.constants =
        iree_make_const_byte_span(constants.data(), constants.size());
    node.attrs.kernel.constants_capacity = constants.size();
    std::array<iree_hal_buffer_ref_t, 1> binding_storage = {
        iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/23, /*offset=*/29,
                                          /*length=*/31),
    };
    node.attrs.kernel.bindings = {
        /*.count=*/binding_storage.size(),
        /*.values=*/binding_storage.data(),
    };
    node.attrs.kernel.binding_capacity = binding_storage.size();

    std::array<uint32_t, kArgumentCount> values = {1, 2, 3};
    std::array<void*, kArgumentCount> arguments = {
        &values[0],
        &values[1],
        &values[2],
    };
    arguments[missing_ordinal] = nullptr;
    const iree_hal_streaming_dispatch_params_t params = {
        /*.grid_dim=*/{23, 29, 31},
        /*.block_dim=*/{37, 41, 43},
        /*.shared_memory_bytes=*/47,
        /*.buffer=*/arguments.data(),
        /*.buffer_size=*/0,
        /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
    };

    EXPECT_THAT(Status(iree_hal_streaming_graph_set_kernel_node_params(
                    &node, &symbol, &params)),
                StatusIs(StatusCode::kInvalidArgument));
    EXPECT_EQ(original_constants, constants);
    EXPECT_EQ(&previous_symbol, node.attrs.kernel.symbol);
    EXPECT_EQ(7u, node.attrs.kernel.grid_dim[0]);
    EXPECT_EQ(5u, node.attrs.kernel.grid_dim[1]);
    EXPECT_EQ(3u, node.attrs.kernel.grid_dim[2]);
    EXPECT_EQ(11u, node.attrs.kernel.block_dim[0]);
    EXPECT_EQ(13u, node.attrs.kernel.block_dim[1]);
    EXPECT_EQ(17u, node.attrs.kernel.block_dim[2]);
    EXPECT_EQ(19u, node.attrs.kernel.shared_memory_bytes);
    EXPECT_EQ(constants.size(), node.attrs.kernel.constants.data_length);
    EXPECT_EQ(binding_storage.size(), node.attrs.kernel.bindings.count);
    EXPECT_EQ(binding_storage.data(), node.attrs.kernel.bindings.values);
    EXPECT_EQ(23u, binding_storage[0].buffer_slot);
    EXPECT_EQ(29u, binding_storage[0].offset);
    EXPECT_EQ(31u, binding_storage[0].length);
  }
}

TEST(GraphTest, KernelParameterUpdateRejectsShortPrepackedSpan) {
  iree_hal_streaming_module_t module = {};
  InitializeLiveMockModule(&module);
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 16> constants = {};
  constants.fill(0x5A);
  const std::array<uint8_t, 16> original_constants = constants;
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  iree_hal_streaming_symbol_t previous_symbol = {};
  previous_symbol.module = &module;
  node.attrs.kernel.symbol = &previous_symbol;
  node.attrs.kernel.module = &module;
  node.attrs.kernel.grid_dim[0] = 7;
  node.attrs.kernel.block_dim[0] = 11;
  node.attrs.kernel.shared_memory_bytes = 19;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  std::array<iree_hal_buffer_ref_t, 1> binding_storage = {};
  node.attrs.kernel.bindings = {
      /*.count=*/binding_storage.size(),
      /*.values=*/binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.module = &module;
  symbol.parameters.constant_bytes = constants.size();
  symbol.parameters.direct_arg_bytes = constants.size();

  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/reinterpret_cast<void*>(uintptr_t{1}),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  EXPECT_EQ(original_constants, constants);
  EXPECT_EQ(&previous_symbol, node.attrs.kernel.symbol);
  EXPECT_EQ(7u, node.attrs.kernel.grid_dim[0]);
  EXPECT_EQ(11u, node.attrs.kernel.block_dim[0]);
  EXPECT_EQ(19u, node.attrs.kernel.shared_memory_bytes);
  EXPECT_EQ(constants.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(binding_storage.size(), node.attrs.kernel.bindings.count);
  EXPECT_EQ(binding_storage.data(), node.attrs.kernel.bindings.values);
}

TEST(GraphTest, KernelParameterUpdateCapturesPrepackedArgumentSpans) {
  iree_hal_streaming_module_t module = {};
  InitializeLiveMockModule(&module);
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 24> constants = {};
  constants.fill(0xA5);
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node.attrs.kernel.module = &module;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.module = &module;
  symbol.parameters.constant_bytes = 16;
  symbol.parameters.direct_arg_bytes = 16;

  std::array<uint8_t, 16> exact_arguments = {};
  for (uint8_t i = 0; i < exact_arguments.size(); ++i) {
    exact_arguments[i] = i;
  }
  const iree_hal_streaming_dispatch_params_t exact_params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/exact_arguments.data(),
      /*.buffer_size=*/exact_arguments.size(),
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &symbol, &exact_params));
  EXPECT_EQ(exact_arguments.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(0, memcmp(exact_arguments.data(), constants.data(),
                      exact_arguments.size()));
  for (size_t i = exact_arguments.size(); i < constants.size(); ++i) {
    EXPECT_EQ(0u, constants[i]);
  }
  exact_arguments.fill(0xFF);
  EXPECT_NE(0, memcmp(exact_arguments.data(), constants.data(),
                      exact_arguments.size()));

  std::array<uint8_t, 24> padded_arguments = {};
  for (uint8_t i = 0; i < padded_arguments.size(); ++i) {
    padded_arguments[i] = static_cast<uint8_t>(0x80u + i);
  }
  const iree_hal_streaming_dispatch_params_t padded_params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/padded_arguments.data(),
      /*.buffer_size=*/padded_arguments.size(),
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &symbol, &padded_params));
  EXPECT_EQ(padded_arguments.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(0, memcmp(padded_arguments.data(), constants.data(),
                      padded_arguments.size()));
  padded_arguments.fill(0xFF);
  EXPECT_NE(0, memcmp(padded_arguments.data(), constants.data(),
                      padded_arguments.size()));

  iree_hal_streaming_symbol_t empty_symbol = {};
  empty_symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  empty_symbol.module = &module;
  const iree_hal_streaming_dispatch_params_t empty_params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/nullptr,
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &empty_symbol, &empty_params));
  EXPECT_EQ(0u, node.attrs.kernel.constants.data_length);
}

TEST(GraphTest, ArgsArrayPackingProducesCompleteNativeAbiImage) {
  iree_hal_streaming_module_t module = {};
  InitializeLiveMockModule(&module);
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  constexpr iree_host_size_t kNativeArgumentSize = 52;
  std::array<uint8_t, kNativeArgumentSize> constants;
  constants.fill(0xA5);
  std::array<iree_hal_buffer_ref_t, 2> binding_storage = {};
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node.attrs.kernel.module = &module;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  node.attrs.kernel.bindings = {
      /*.count=*/binding_storage.size(),
      /*.values=*/binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  std::array<iree_hal_streaming_parameter_op_t, 4> operations = {};
  operations[0].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/4,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  operations[1].copy = {
      /*.size=*/sizeof(uint16_t),
      /*.native_abi_destination_offset=*/28,
      /*.source_offset=*/12,
      /*.source_ordinal=*/2,
      /*.constant_destination_offset=*/4,
  };
  operations[2].resolve = {
      /*.native_abi_destination_offset=*/16,
      /*.reserved=*/0,
      /*.source_offset=*/4,
      /*.source_ordinal=*/1,
      /*.destination_ordinal=*/1,
  };
  operations[3].resolve = {
      /*.native_abi_destination_offset=*/40,
      /*.reserved=*/0,
      /*.source_offset=*/14,
      /*.source_ordinal=*/3,
      /*.destination_ordinal=*/0,
  };
  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.module = &module;
  symbol.parameters.buffer_size = 22;
  symbol.parameters.constant_bytes = 6;
  symbol.parameters.direct_arg_bytes = kNativeArgumentSize;
  symbol.parameters.binding_count = 2;
  symbol.parameters.copy_count = 2;
  symbol.parameters.ops = operations.data();

  uint32_t scalar0 = 0x11223344u;
  void* pointer1 = reinterpret_cast<void*>(uintptr_t{0x0102030405060708ull});
  uint16_t scalar2 = 0x5566u;
  void* pointer3 = reinterpret_cast<void*>(uintptr_t{0x1112131415161718ull});
  std::array<void*, 4> arguments = {
      &scalar0,
      &pointer1,
      &scalar2,
      &pointer3,
  };
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{1, 1, 1},
      /*.block_dim=*/{1, 1, 1},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_ASSERT_OK(
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  std::array<uint8_t, kNativeArgumentSize> expected = {};
  memcpy(expected.data() + 4, &scalar0, sizeof(scalar0));
  const iree_hal_streaming_deviceptr_t device_pointer1 =
      static_cast<iree_hal_streaming_deviceptr_t>(
          reinterpret_cast<uintptr_t>(pointer1));
  memcpy(expected.data() + 16, &device_pointer1, sizeof(device_pointer1));
  memcpy(expected.data() + 28, &scalar2, sizeof(scalar2));
  const iree_hal_streaming_deviceptr_t device_pointer3 =
      static_cast<iree_hal_streaming_deviceptr_t>(
          reinterpret_cast<uintptr_t>(pointer3));
  memcpy(expected.data() + 40, &device_pointer3, sizeof(device_pointer3));

  EXPECT_EQ(expected.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(expected, constants);
  EXPECT_EQ(0u, node.attrs.kernel.bindings.count);
}

TEST(GraphTest, ArgsArrayPackingRejectsDuplicateSourceWithoutMutation) {
  iree_hal_streaming_module_t module = {};
  InitializeLiveMockModule(&module);
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 16> constants;
  constants.fill(0xA5);
  const std::array<uint8_t, 16> original_constants = constants;
  std::array<iree_hal_buffer_ref_t, 1> binding_storage = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/23, /*offset=*/29,
                                        /*length=*/31),
  };
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  iree_hal_streaming_symbol_t previous_symbol = {};
  previous_symbol.module = &module;
  node.attrs.kernel.symbol = &previous_symbol;
  node.attrs.kernel.module = &module;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  node.attrs.kernel.bindings = {
      /*.count=*/binding_storage.size(),
      /*.values=*/binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  std::array<iree_hal_streaming_parameter_op_t, 2> operations = {};
  operations[0].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/0,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  operations[1].resolve = {
      /*.native_abi_destination_offset=*/8,
      /*.reserved=*/0,
      /*.source_offset=*/4,
      /*.source_ordinal=*/0,
      /*.destination_ordinal=*/0,
  };
  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.module = &module;
  symbol.parameters.buffer_size = 12;
  symbol.parameters.constant_bytes = 4;
  symbol.parameters.direct_arg_bytes = constants.size();
  symbol.parameters.binding_count = 1;
  symbol.parameters.copy_count = 1;
  symbol.parameters.ops = operations.data();

  uint32_t value = 7;
  std::array<void*, 1> arguments = {&value};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  EXPECT_EQ(original_constants, constants);
  EXPECT_EQ(&previous_symbol, node.attrs.kernel.symbol);
  EXPECT_EQ(1u, node.attrs.kernel.bindings.count);
  EXPECT_EQ(23u, binding_storage[0].buffer_slot);
  EXPECT_EQ(29u, binding_storage[0].offset);
  EXPECT_EQ(31u, binding_storage[0].length);
}

struct ProbedHostAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  bool fail_allocations = false;
  int fail_on_allocation = 0;
  int allocation_attempt_count = 0;
  int successful_allocation_count = 0;
  int free_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ProbedHostAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      ++allocator->allocation_attempt_count;
      if (allocator->fail_allocations || allocator->allocation_attempt_count ==
                                             allocator->fail_on_allocation) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
      ++allocator->successful_allocation_count;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      ++allocator->free_count;
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &ProbedHostAllocator::Control};
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

TEST(GraphTest, DependencyBatchSecondAllocationFailureIsAtomicAndRetryable) {
  ProbedHostAllocator allocator;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(sizeof(iree_arena_block_t),
                                   allocator.AsAllocator(), &block_pool);
  iree_hal_streaming_graph_t graph = {};
  iree_arena_initialize(&block_pool, &graph.arena);
  graph.arena_allocator = iree_arena_allocator(&graph.arena);

  GraphNodeStorage first_node_storage;
  GraphNodeStorage second_node_storage;
  GraphNodeStorage third_node_storage;
  iree_hal_streaming_graph_node_t* first_node = first_node_storage.get();
  iree_hal_streaming_graph_node_t* second_node = second_node_storage.get();
  iree_hal_streaming_graph_node_t* third_node = third_node_storage.get();
  first_node->graph = &graph;
  second_node->graph = &graph;
  third_node->graph = &graph;

  constexpr iree_host_size_t kNodeCount = 3;
  const iree_host_size_t node_block_size =
      sizeof(iree_hal_streaming_node_block_t) +
      kNodeCount * sizeof(iree_hal_streaming_graph_node_t*);
  iree_hal_streaming_node_block_t* node_block = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(), node_block_size,
                                       reinterpret_cast<void**>(&node_block)));
  node_block->next = nullptr;
  node_block->capacity = kNodeCount;
  node_block->count = kNodeCount;
  node_block->nodes[0] = first_node;
  node_block->nodes[1] = second_node;
  node_block->nodes[2] = third_node;
  graph.node_blocks = node_block;

  std::array<iree_hal_streaming_graph_node_t*, 2> from_nodes = {
      first_node,
      second_node,
  };
  std::array<iree_hal_streaming_graph_node_t*, 2> to_nodes = {
      second_node,
      third_node,
  };
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(&graph.arena);

  allocator.fail_on_allocation = 2;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_graph_add_dependencies(
          &graph, from_nodes.data(), to_nodes.data(), from_nodes.size()));
  EXPECT_EQ(2, allocator.allocation_attempt_count);
  EXPECT_EQ(nullptr, graph.additional_edges);
  EXPECT_EQ(0u, graph.additional_edge_count);
  ExpectArenaMatchesCheckpoint(graph.arena, checkpoint);

  allocator.allocation_attempt_count = 0;
  allocator.fail_on_allocation = 0;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
      &graph, from_nodes.data(), to_nodes.data(), from_nodes.size()));
  ASSERT_EQ(2u, graph.additional_edge_count);
  ASSERT_NE(nullptr, graph.additional_edges);
  EXPECT_EQ(second_node, graph.additional_edges->from);
  EXPECT_EQ(third_node, graph.additional_edges->to);
  ASSERT_NE(nullptr, graph.additional_edges->next);
  EXPECT_EQ(first_node, graph.additional_edges->next->from);
  EXPECT_EQ(second_node, graph.additional_edges->next->to);
  EXPECT_EQ(nullptr, graph.additional_edges->next->next);

  graph.node_blocks = nullptr;
  iree_allocator_free(iree_allocator_system(), node_block);
  iree_arena_deinitialize(&graph.arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(GraphTest, BatchParameterSecondAllocationFailureIsAtomicAndRewindsArena) {
  ProbedHostAllocator allocator;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/64,
                                   allocator.AsAllocator(), &block_pool);
  iree_hal_streaming_graph_t graph = {};
  iree_arena_initialize(&block_pool, &graph.arena);
  graph.arena_allocator = iree_arena_allocator(&graph.arena);

  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP;
  std::array<uint8_t, 8> old_params;
  std::array<uint8_t, 8> old_param_array;
  old_params.fill(0xA5);
  old_param_array.fill(0x5A);
  const std::array<uint8_t, 8> original_params = old_params;
  const std::array<uint8_t, 8> original_param_array = old_param_array;
  node.attrs.batch_mem_op = {
      /*.params=*/old_params.data(),
      /*.params_size=*/4,
      /*.params_capacity=*/old_params.size(),
      /*.param_array=*/old_param_array.data(),
      /*.param_array_size=*/6,
      /*.param_array_capacity=*/old_param_array.size(),
  };

  std::array<uint8_t, 128> new_params;
  std::array<uint8_t, 128> new_param_array;
  new_params.fill(0x3C);
  new_param_array.fill(0xC3);
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(&graph.arena);

  for (int iteration = 0; iteration < 4; ++iteration) {
    allocator.allocation_attempt_count = 0;
    allocator.fail_on_allocation = 2;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_hal_streaming_graph_set_batch_mem_op_node_params(
                              &node, new_params.data(), new_params.size(),
                              new_param_array.data(), new_param_array.size()));

    EXPECT_EQ(2, allocator.allocation_attempt_count);
    EXPECT_EQ(old_params.data(), node.attrs.batch_mem_op.params);
    EXPECT_EQ(4u, node.attrs.batch_mem_op.params_size);
    EXPECT_EQ(old_params.size(), node.attrs.batch_mem_op.params_capacity);
    EXPECT_EQ(old_param_array.data(), node.attrs.batch_mem_op.param_array);
    EXPECT_EQ(6u, node.attrs.batch_mem_op.param_array_size);
    EXPECT_EQ(old_param_array.size(),
              node.attrs.batch_mem_op.param_array_capacity);
    EXPECT_EQ(original_params, old_params);
    EXPECT_EQ(original_param_array, old_param_array);
    ExpectArenaMatchesCheckpoint(graph.arena, checkpoint);
  }

  allocator.allocation_attempt_count = 0;
  allocator.fail_on_allocation = 0;
  IREE_ASSERT_OK(iree_hal_streaming_graph_set_batch_mem_op_node_params(
      &node, new_params.data(), new_params.size(), new_param_array.data(),
      new_param_array.size()));
  EXPECT_NE(old_params.data(), node.attrs.batch_mem_op.params);
  EXPECT_EQ(new_params.size(), node.attrs.batch_mem_op.params_size);
  EXPECT_EQ(new_params.size(), node.attrs.batch_mem_op.params_capacity);
  EXPECT_EQ(0, memcmp(new_params.data(), node.attrs.batch_mem_op.params,
                      new_params.size()));
  EXPECT_NE(old_param_array.data(), node.attrs.batch_mem_op.param_array);
  EXPECT_EQ(new_param_array.size(), node.attrs.batch_mem_op.param_array_size);
  EXPECT_EQ(new_param_array.size(),
            node.attrs.batch_mem_op.param_array_capacity);
  EXPECT_EQ(0,
            memcmp(new_param_array.data(), node.attrs.batch_mem_op.param_array,
                   new_param_array.size()));

  iree_arena_deinitialize(&graph.arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

void InitializeSingleCopySymbol(uint16_t direct_arg_bytes,
                                uint16_t destination_offset,
                                iree_hal_streaming_parameter_op_t* operation,
                                iree_hal_streaming_symbol_t* out_symbol) {
  operation->copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/destination_offset,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  out_symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  out_symbol->parameters.buffer_size = sizeof(uint32_t);
  out_symbol->parameters.constant_bytes = sizeof(uint32_t);
  out_symbol->parameters.direct_arg_bytes = direct_arg_bytes;
  out_symbol->parameters.copy_count = 1;
  out_symbol->parameters.ops = operation;
}

TEST(GraphTest, LaunchUsesInlineArgumentStorageForSmallMetadata) {
  ProbedHostAllocator allocator;
  iree_hal_streaming_stream_t stream = {};
  stream.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_parameter_op_t operation = {};
  iree_hal_streaming_symbol_t symbol = {};
  InitializeSingleCopySymbol(/*direct_arg_bytes=*/128,
                             /*destination_offset=*/64, &operation, &symbol);
  std::array<void*, 1> arguments = {nullptr};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_launch_kernel(&symbol, &params, &stream));
  EXPECT_EQ(0, allocator.allocation_attempt_count);
  EXPECT_EQ(0, allocator.free_count);
}

TEST(GraphTest, LaunchFreesHeapArgumentStorageAfterPackingFailure) {
  ProbedHostAllocator allocator;
  iree_hal_streaming_stream_t stream = {};
  stream.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_parameter_op_t operation = {};
  iree_hal_streaming_symbol_t symbol = {};
  InitializeSingleCopySymbol(/*direct_arg_bytes=*/512,
                             /*destination_offset=*/256, &operation, &symbol);
  std::array<void*, 1> arguments = {nullptr};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_launch_kernel(&symbol, &params, &stream));
  EXPECT_EQ(1, allocator.allocation_attempt_count);
  EXPECT_EQ(1, allocator.successful_allocation_count);
  EXPECT_EQ(1, allocator.free_count);
}

TEST(GraphTest, LaunchReportsHeapArgumentStorageAllocationFailure) {
  ProbedHostAllocator allocator;
  allocator.fail_allocations = true;
  iree_hal_streaming_stream_t stream = {};
  stream.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_parameter_op_t operation = {};
  iree_hal_streaming_symbol_t symbol = {};
  InitializeSingleCopySymbol(/*direct_arg_bytes=*/512,
                             /*destination_offset=*/256, &operation, &symbol);
  uint32_t value = 7;
  std::array<void*, 1> arguments = {&value};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_launch_kernel(&symbol, &params, &stream));
  EXPECT_EQ(1, allocator.allocation_attempt_count);
  EXPECT_EQ(0, allocator.successful_allocation_count);
  EXPECT_EQ(0, allocator.free_count);
}

}  // namespace
