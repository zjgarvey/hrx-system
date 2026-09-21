// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <set>

#include "iree/hal/allocator.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct LegacyVmmAllocator {
  // HAL resource header carrying the fake allocator vtable.
  iree_hal_resource_t resource;
  // Number of calls dispatched through the legacy reservation slot.
  int reserve_call_count = 0;
  // Queue-family affinity received by the legacy reservation slot.
  iree_hal_queue_family_affinity_t last_queue_family_affinity = 0;
  // Reservation size received by the legacy reservation slot.
  iree_device_size_t last_size = 0;
  // Sentinel buffer returned by the legacy reservation slot.
  iree_hal_buffer_t* reserve_result = nullptr;
};

static LegacyVmmAllocator* LegacyVmmAllocatorCast(
    iree_hal_allocator_t* allocator) {
  return reinterpret_cast<LegacyVmmAllocator*>(allocator);
}

static void LegacyVmmAllocatorDestroy(iree_hal_allocator_t* allocator) {
  (void)allocator;
}

static iree_status_t LegacyVmmAllocatorReserve(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer) {
  LegacyVmmAllocator* allocator = LegacyVmmAllocatorCast(base_allocator);
  ++allocator->reserve_call_count;
  allocator->last_queue_family_affinity = queue_family_affinity;
  allocator->last_size = size;
  *out_virtual_buffer = allocator->reserve_result;
  return iree_ok_status();
}

static const iree_hal_allocator_vtable_t kLegacyVmmAllocatorVTable = {
    /*.destroy=*/LegacyVmmAllocatorDestroy,
    /*.host_allocator=*/nullptr,
    /*.trim=*/nullptr,
    /*.query_statistics=*/nullptr,
    /*.query_memory_heaps=*/nullptr,
    /*.query_buffer_compatibility=*/nullptr,
    /*.allocate_buffer=*/nullptr,
    /*.deallocate_buffer=*/nullptr,
    /*.import_buffer=*/nullptr,
    /*.export_buffer=*/nullptr,
    /*.supports_virtual_memory=*/nullptr,
    /*.virtual_memory_query_granularity=*/nullptr,
    /*.virtual_memory_reserve=*/LegacyVmmAllocatorReserve,
    /*.virtual_memory_release=*/nullptr,
    /*.physical_memory_allocate=*/nullptr,
    /*.physical_memory_free=*/nullptr,
    /*.virtual_memory_map=*/nullptr,
    /*.virtual_memory_unmap=*/nullptr,
    /*.virtual_memory_protect=*/nullptr,
    /*.virtual_memory_advise=*/nullptr,
    /*.virtual_memory_reserve_at=*/nullptr,
};

static iree_hal_allocator_t* LegacyVmmAllocatorAsBase(
    LegacyVmmAllocator* allocator) {
  iree_hal_resource_initialize(&kLegacyVmmAllocatorVTable,
                               &allocator->resource);
  return reinterpret_cast<iree_hal_allocator_t*>(allocator);
}

TEST(AllocatorTest, ReserveAtWithoutHintsUsesLegacyVtableSlot) {
  LegacyVmmAllocator allocator;
  iree_hal_buffer_t sentinel_buffer = {};
  allocator.reserve_result = &sentinel_buffer;
  iree_hal_allocator_t* base_allocator = LegacyVmmAllocatorAsBase(&allocator);
  const iree_hal_queue_family_affinity_t queue_family_affinity =
      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  const iree_device_size_t size = 4096;

  iree_hal_buffer_t* virtual_buffer = nullptr;
  IREE_EXPECT_OK(iree_hal_allocator_virtual_memory_reserve_at(
      base_allocator, queue_family_affinity, size, /*minimum_alignment=*/0,
      /*requested_address=*/0, &virtual_buffer));

  EXPECT_EQ(1, allocator.reserve_call_count);
  EXPECT_EQ(queue_family_affinity, allocator.last_queue_family_affinity);
  EXPECT_EQ(size, allocator.last_size);
  EXPECT_EQ(&sentinel_buffer, virtual_buffer);
  iree_hal_allocator_release(base_allocator);
}

TEST(AllocatorTest, ReserveAtRejectsHintsWhenOptionalVtableSlotIsNull) {
  LegacyVmmAllocator allocator;
  iree_hal_buffer_t sentinel_buffer = {};
  allocator.reserve_result = &sentinel_buffer;
  iree_hal_allocator_t* base_allocator = LegacyVmmAllocatorAsBase(&allocator);
  ASSERT_EQ(nullptr, kLegacyVmmAllocatorVTable.virtual_memory_reserve_at);

  iree_hal_buffer_t* virtual_buffer = &sentinel_buffer;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_allocator_virtual_memory_reserve_at(
          base_allocator, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 4096,
          /*minimum_alignment=*/4096, /*requested_address=*/0,
          &virtual_buffer));
  EXPECT_EQ(nullptr, virtual_buffer);

  virtual_buffer = &sentinel_buffer;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_allocator_virtual_memory_reserve_at(
          base_allocator, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 4096,
          /*minimum_alignment=*/0, /*requested_address=*/0x10000,
          &virtual_buffer));
  EXPECT_EQ(nullptr, virtual_buffer);
  EXPECT_EQ(0, allocator.reserve_call_count);
  iree_hal_allocator_release(base_allocator);
}

TEST(HeapAllocatorTest, ProvidesCoherentUnifiedMemory) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_allocator_memory_heap_t heap;
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(
      iree_hal_allocator_query_memory_heaps(allocator, 1, &heap, &heap_count));
  ASSERT_EQ(heap_count, 1);
  EXPECT_TRUE(
      iree_all_bits_set(heap.type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                       IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                                       IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  const iree_hal_atomic_operation_capabilities_t expected_atomic_operations =
      iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(heap.atomic_operations.device_scope_32,
            expected_atomic_operations.device_scope_32);
  EXPECT_EQ(heap.atomic_operations.device_scope_64,
            expected_atomic_operations.device_scope_64);
  EXPECT_EQ(heap.atomic_operations.system_scope_32,
            expected_atomic_operations.system_scope_32);
  EXPECT_EQ(heap.atomic_operations.system_scope_64,
            expected_atomic_operations.system_scope_64);

  const iree_hal_buffer_params_t params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                    IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));

  iree_hal_buffer_release(buffer);
  iree_hal_allocator_release(allocator);
}

// Records the pointers handed out by TrackingAllocatorCtl so that every free
// can be matched against the allocation that produced it.
struct TrackingAllocatorState {
  // Block bases this allocator returned that have not been freed yet. An
  // aligned allocation is served as one underlying block, so the entry is that
  // block's base and not the interior pointer the caller of
  // iree_allocator_malloc_aligned receives.
  std::set<void*> live_allocations;
  // Number of frees of pointers this allocator never returned.
  iree_host_size_t unowned_free_count = 0;
};

static iree_status_t TrackingAllocatorCtl(void* self,
                                          iree_allocator_command_t command,
                                          const void* params,
                                          void** inout_ptr) {
  auto* state = static_cast<TrackingAllocatorState*>(self);
  const iree_allocator_t system_allocator = iree_allocator_system();
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      iree_status_t status = system_allocator.ctl(system_allocator.self,
                                                  command, params, inout_ptr);
      if (iree_status_is_ok(status)) {
        state->live_allocations.insert(*inout_ptr);
      }
      return status;
    }
    case IREE_ALLOCATOR_COMMAND_FREE: {
      auto it = state->live_allocations.find(*inout_ptr);
      if (it == state->live_allocations.end()) {
        // Forwarding this would hand the system allocator a pointer it never
        // returned; the free is recorded and dropped so that the mismatch is
        // reported as a failed expectation.
        ++state->unowned_free_count;
        return iree_ok_status();
      }
      state->live_allocations.erase(it);
      return system_allocator.ctl(system_allocator.self, command, params,
                                  inout_ptr);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported tracking allocator command %u",
                              (unsigned)command);
  }
}

static iree_allocator_t TrackingAllocator(TrackingAllocatorState* state) {
  return iree_allocator_t{
      /*.self=*/state,
      /*.ctl=*/TrackingAllocatorCtl,
  };
}

// Buffers from an allocator with distinct data and host allocators keep their
// storage in an allocation separate from their metadata; this pins that the
// storage block is handed back to its data allocator as the pointer that
// allocator returned, which is the block base and not the interior pointer
// carved out of it.
TEST(HeapAllocatorTest, FreesSplitStorageAsItsDataAllocatorReturnedIt) {
  TrackingAllocatorState data_state;
  TrackingAllocatorState host_state;
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_create_heap(
      IREE_SV("test"), TrackingAllocator(&data_state),
      TrackingAllocator(&host_state), &allocator));

  // The allocator object is itself a host allocation; the buffer metadata
  // shows up as growth of the host live set across the allocation below.
  const size_t host_allocation_count_before_buffer =
      host_state.live_allocations.size();

  const iree_hal_buffer_params_t params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  // The data allocator differs from the host allocator, so the buffer storage
  // comes from the data allocator and the buffer metadata from the host
  // allocator; the checks below are vacuous unless that split path ran.
  ASSERT_EQ(data_state.live_allocations.size(), 1u);
  ASSERT_EQ(host_state.live_allocations.size(),
            host_allocation_count_before_buffer + 1);

  iree_hal_buffer_release(buffer);
  iree_hal_allocator_release(allocator);

  EXPECT_EQ(data_state.unowned_free_count, 0u);
  EXPECT_TRUE(data_state.live_allocations.empty());

  // The host allocator served the allocator object as well as the buffer
  // metadata, so it is only drained once both have been released.
  EXPECT_EQ(host_state.unowned_free_count, 0u);
  EXPECT_TRUE(host_state.live_allocations.empty());
}

}  // namespace
