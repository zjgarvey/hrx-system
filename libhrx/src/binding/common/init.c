// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "common/internal.h"
#include "iree/base/threading/call_once.h"
//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Global device registry.
static iree_hal_streaming_device_registry_t*
    iree_hal_streaming_global_registry = NULL;
static iree_atomic_uint64_t iree_hal_streaming_next_runtime_generation =
    IREE_ATOMIC_VAR_INIT(1);

typedef enum iree_hal_streaming_runtime_gate_state_e {
  IREE_HAL_STREAMING_RUNTIME_GATE_STATE_OPEN = 0,
  IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING = 1,
  IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLOSED = 2,
} iree_hal_streaming_runtime_gate_state_t;

// Process-lifetime gate preventing a new CUDA context/TLS ownership edge from
// appearing after cleanup has taken its ownership snapshot. HIP has a broader
// binding lifecycle writer; this common gate covers bindings that call common
// cleanup directly and remains valid across runtime generations.
static iree_once_flag iree_hal_streaming_runtime_gate_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hal_streaming_runtime_gate_mutex;
static iree_hal_streaming_runtime_gate_state_t
    iree_hal_streaming_runtime_gate_state =
        IREE_HAL_STREAMING_RUNTIME_GATE_STATE_OPEN;
static int32_t iree_hal_streaming_runtime_publication_count = 0;
static bool iree_hal_streaming_runtime_initialization_pending = false;

static void iree_hal_streaming_runtime_gate_initialize(void) {
  iree_slim_mutex_initialize(&iree_hal_streaming_runtime_gate_mutex);
}

iree_status_t iree_hal_streaming_context_publication_begin(void) {
  iree_call_once(&iree_hal_streaming_runtime_gate_once,
                 iree_hal_streaming_runtime_gate_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_streaming_runtime_gate_state !=
          IREE_HAL_STREAMING_RUNTIME_GATE_STATE_OPEN ||
      iree_hal_streaming_runtime_initialization_pending) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "streaming runtime context publication is closed for cleanup");
  } else if (iree_hal_streaming_runtime_publication_count == INT32_MAX) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "context publication count overflow");
  } else {
    ++iree_hal_streaming_runtime_publication_count;
  }
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
  return status;
}

void iree_hal_streaming_context_publication_end(void) {
  iree_call_once(&iree_hal_streaming_runtime_gate_once,
                 iree_hal_streaming_runtime_gate_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  IREE_ASSERT(iree_hal_streaming_runtime_publication_count > 0);
  --iree_hal_streaming_runtime_publication_count;
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
}

static iree_status_t iree_hal_streaming_runtime_initialization_begin(void) {
  iree_call_once(&iree_hal_streaming_runtime_gate_once,
                 iree_hal_streaming_runtime_gate_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_streaming_runtime_initialization_pending ||
      iree_hal_streaming_runtime_gate_state ==
          IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "streaming runtime initialization or cleanup is already in progress");
  } else {
    iree_hal_streaming_runtime_initialization_pending = true;
  }
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
  return status;
}

static void iree_hal_streaming_runtime_initialization_end(bool succeeded) {
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  IREE_ASSERT(iree_hal_streaming_runtime_initialization_pending);
  iree_hal_streaming_runtime_initialization_pending = false;
  (void)succeeded;
  IREE_ASSERT(iree_hal_streaming_runtime_gate_state !=
              IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING);
  // Starting the next initialization generation reopens publication. A failed
  // initialization publishes no registry, so leaving the gate open is safe
  // and lets its cleanup rollback acquire the writer normally.
  iree_hal_streaming_runtime_gate_state =
      IREE_HAL_STREAMING_RUNTIME_GATE_STATE_OPEN;
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
}

static iree_status_t iree_hal_streaming_runtime_cleanup_begin(
    bool* out_already_closed) {
  IREE_ASSERT_ARGUMENT(out_already_closed);
  *out_already_closed = false;
  iree_call_once(&iree_hal_streaming_runtime_gate_once,
                 iree_hal_streaming_runtime_gate_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_streaming_runtime_gate_state ==
      IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLOSED) {
    *out_already_closed = true;
  } else if (iree_hal_streaming_runtime_gate_state ==
                 IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING ||
             iree_hal_streaming_runtime_initialization_pending) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "streaming runtime initialization or cleanup is already in progress");
  } else if (iree_hal_streaming_runtime_publication_count != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot clean up streaming runtime during context publication");
  } else {
    iree_hal_streaming_runtime_gate_state =
        IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING;
  }
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
  return status;
}

static void iree_hal_streaming_runtime_cleanup_end(bool succeeded) {
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  IREE_ASSERT(iree_hal_streaming_runtime_gate_state ==
              IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING);
  iree_hal_streaming_runtime_gate_state =
      succeeded ? IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLOSED
                : IREE_HAL_STREAMING_RUNTIME_GATE_STATE_OPEN;
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
}

// Accessor function for the global device registry.
iree_hal_streaming_device_registry_t* iree_hal_streaming_device_registry(void) {
  return iree_hal_streaming_global_registry;
}

#if defined(IREE_HAL_STREAMING_TEST_INSTRUMENTATION)
void iree_hal_streaming_test_install_device_registry(
    iree_hal_streaming_device_registry_t* registry,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT(iree_hal_streaming_global_registry == NULL);
  iree_call_once(&iree_hal_streaming_runtime_gate_once,
                 iree_hal_streaming_runtime_gate_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_runtime_gate_mutex);
  IREE_ASSERT(iree_hal_streaming_runtime_gate_state !=
              IREE_HAL_STREAMING_RUNTIME_GATE_STATE_CLEANING);
  IREE_ASSERT(!iree_hal_streaming_runtime_initialization_pending);
  IREE_ASSERT(iree_hal_streaming_runtime_publication_count == 0);
  iree_hal_streaming_runtime_gate_state =
      IREE_HAL_STREAMING_RUNTIME_GATE_STATE_OPEN;
  iree_slim_mutex_unlock(&iree_hal_streaming_runtime_gate_mutex);
  memset(registry, 0, sizeof(*registry));
  registry->host_allocator = host_allocator;
  registry->runtime_generation = 1;
  iree_slim_mutex_initialize(&registry->mutex);
  iree_slim_mutex_initialize(&registry->context_list.mutex);
  iree_hal_streaming_global_registry = registry;
}

void iree_hal_streaming_test_remove_device_registry(
    iree_hal_streaming_device_registry_t* registry) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT(iree_hal_streaming_global_registry == registry);
  IREE_ASSERT(registry->context_list.head == NULL);
  IREE_ASSERT(registry->context_list.tail == NULL);
  iree_hal_streaming_global_registry = NULL;
  iree_slim_mutex_deinitialize(&registry->context_list.mutex);
  iree_slim_mutex_deinitialize(&registry->mutex);
}
#endif  // IREE_HAL_STREAMING_TEST_INSTRUMENTATION

uint64_t iree_hal_streaming_runtime_generation(void) {
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_global_registry;
  return registry ? registry->runtime_generation : 0;
}

void iree_hal_streaming_set_lifecycle_hooks(
    iree_hal_streaming_lifecycle_begin_fn_t lifecycle_begin,
    iree_hal_streaming_lifecycle_end_fn_t lifecycle_end,
    iree_hal_streaming_pointer_resolver_fn_t pointer_resolver,
    void* user_data) {
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_global_registry;
  IREE_ASSERT(registry);
  iree_slim_mutex_lock(&registry->mutex);
  registry->lifecycle_begin = lifecycle_begin;
  registry->lifecycle_end = lifecycle_end;
  registry->pointer_resolver = pointer_resolver;
  registry->lifecycle_user_data = user_data;
  iree_slim_mutex_unlock(&registry->mutex);
}

//===----------------------------------------------------------------------===//
// Device enumeration and management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_deinitialize_device(
    iree_hal_streaming_device_t* device);

iree_hal_streaming_device_t* iree_hal_streaming_device_entry(
    iree_hal_streaming_device_ordinal_t ordinal) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  iree_hal_streaming_device_t* device = NULL;
  if (!device_registry || ordinal >= device_registry->device_count) {
    device = NULL;
  } else {
    device = &device_registry->devices[ordinal];
  }
  return device;
}

static uint32_t iree_hal_streaming_u32_or_default(uint64_t value,
                                                  uint32_t default_value) {
  if (value == 0) return default_value;
  if (value > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)value;
}

// Queries device info and populates device properties.
static iree_status_t iree_hal_streaming_query_device_info(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device->hal_device);

  // Query compute capability from the device architecture string.
  // AMD GCN/CDNA/RDNA architectures map to HIP compute capability:
  //   gfx900 -> 9.0, gfx906 -> 9.0, gfx908 -> 9.0, gfx90a -> 9.0
  //   gfx942 -> 9.4, gfx950 -> 9.5
  //   gfx1030 -> 10.3, gfx1100 -> 11.0
  char arch_name[64] = {0};
  iree_status_t arch_status = hrx_to_iree_status(hrx_device_get_property(
      device->hrx_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch_name,
      sizeof(arch_name)));
  if (iree_status_is_ok(arch_status) && arch_name[0] != '\0') {
    // Parse "gfxNNNN" to extract major.minor.
    // gfx9xx -> major=9, minor=x (e.g., gfx942 -> 9.4)
    // gfx10xx -> major=10, minor=x
    // gfx11xx -> major=11, minor=x
    int gfx_num = 0;
    if (sscanf(arch_name, "gfx%d", &gfx_num) == 1) {
      if (gfx_num >= 1000) {
        device->compute_capability_major = gfx_num / 100;
        device->compute_capability_minor = (gfx_num / 10) % 10;
      } else if (gfx_num >= 900) {
        device->compute_capability_major = gfx_num / 100;
        device->compute_capability_minor = (gfx_num / 10) % 10;
      } else {
        device->compute_capability_major = 7;
        device->compute_capability_minor = 5;
      }
    } else {
      device->compute_capability_major = 7;
      device->compute_capability_minor = 5;
    }
    // Store the architecture name for hipGetDeviceProperties.
    size_t name_len = strlen(arch_name);
    if (name_len >= sizeof(device->gcn_arch_name)) {
      name_len = sizeof(device->gcn_arch_name) - 1;
    }
    memcpy(device->gcn_arch_name, arch_name, name_len);
    device->gcn_arch_name[name_len] = '\0';
  } else {
    iree_status_ignore(arch_status);
    device->compute_capability_major = 9;
    device->compute_capability_minor = 4;
    memcpy(device->gcn_arch_name, "gfx942", 7);
  }

  // Query total memory from the immutable HAL device spec.
  uint64_t total_memory = 0;
  iree_status_t status = HRX_CALL(hrx_device_get_property(
      device->hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total_memory,
      sizeof(total_memory)));
  if (!iree_status_is_ok(status)) return status;
  if (total_memory > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HRX device total memory exceeds the representable "
                            "iree_device_size_t range");
  }
  device->total_memory = (iree_device_size_t)total_memory;
  iree_atomic_store(&device->free_memory, device->total_memory,
                    iree_memory_order_relaxed);

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(device->hal_device);
  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  const iree_hal_device_launch_spec_t* launch =
      dispatch ? &dispatch->launch : NULL;
  const iree_hal_device_subgroup_spec_t* subgroup =
      dispatch ? &dispatch->subgroup : NULL;
  const iree_hal_device_execution_spec_t* execution =
      dispatch ? &dispatch->execution : NULL;
  const bool is_gfx1100 = strncmp(device->gcn_arch_name, "gfx1100", 7) == 0;
  const bool is_gfx942 = strncmp(device->gcn_arch_name, "gfx942", 6) == 0;

  // Cooperative launch is a property of the queue family selected for HIP
  // streams, not of a marketing architecture number. A provisioned
  // cooperative queue can be used directly; otherwise the family must support
  // both the feature and dynamic acquisition of the specialized realization.
  iree_hal_queue_t* primary_queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_device_select_primary_queue(device, &primary_queue));
  const iree_hal_queue_family_spec_t* primary_queue_family_spec =
      iree_hal_queue_family_spec(iree_hal_queue_family(primary_queue));
  device->supports_cooperative_launch =
      iree_all_bits_set(iree_hal_queue_features(primary_queue),
                        IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH) ||
      (iree_all_bits_set(primary_queue_family_spec->supported_queue_features,
                         IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH) &&
       iree_all_bits_set(primary_queue_family_spec->flags,
                         IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION));

  device->max_threads_per_block = iree_hal_streaming_u32_or_default(
      launch ? launch->maximum_workgroup_invocations : 0, 1024);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(device->max_block_dim); ++i) {
    device->max_block_dim[i] = iree_hal_streaming_u32_or_default(
        launch ? launch->maximum_workgroup_size[i] : 0, i == 2 ? 64 : 1024);
    device->max_grid_dim[i] = iree_hal_streaming_u32_or_default(
        launch ? launch->maximum_workgroup_count[i] : 0,
        i == 0 ? 2147483647u : 65535u);
  }

  device->warp_size = iree_hal_streaming_u32_or_default(
      subgroup ? subgroup->default_size : 0, 64);
  if (is_gfx1100) {
    // HIP reports wave32 as the warp size for RDNA3 devices. IREE/HSA may
    // expose the hardware wavefront width instead, which in turn prevents the
    // CU-to-WGP compatibility adjustment below.
    device->warp_size = 32;
  }

  uint32_t multiprocessor_count = iree_hal_streaming_u32_or_default(
      execution ? execution->unit_count : 0, 80);
  // IREE/HSA reports raw compute units, while HIP reports RDNA devices in
  // WGP-like units. Keep this HIP-compatible because rocBLAS/hipBLASLt query
  // the physical multiprocessor count when selecting GEMM solutions.
  if (device->compute_capability_major >= 10 && device->warp_size == 32 &&
      multiprocessor_count > 1 && (multiprocessor_count % 2) == 0) {
    multiprocessor_count /= 2;
  }
  device->multiprocessor_count = multiprocessor_count;

  device->max_threads_per_multiprocessor = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_resident_invocation_count : 0, 2048);
  device->max_blocks_per_multiprocessor = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_resident_workgroup_count : 0,
      is_gfx942 ? 2 : 32);
  uint64_t maximum_register_count =
      execution ? execution->maximum_register_count : 0;
  if (maximum_register_count == 0 && execution) {
    maximum_register_count = execution->maximum_workgroup_register_count;
  }
  device->max_registers_per_multiprocessor =
      iree_hal_streaming_u32_or_default(maximum_register_count, 65536);
  uint64_t maximum_local_memory_size =
      execution ? execution->maximum_local_memory_size : 0;
  if (maximum_local_memory_size == 0 && execution) {
    maximum_local_memory_size = execution->maximum_workgroup_local_memory_size;
  }
  device->max_shared_memory_per_multiprocessor =
      iree_hal_streaming_u32_or_default(maximum_local_memory_size,
                                        is_gfx942 ? 19922944u : 49152u);
  device->max_registers_per_block = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_workgroup_register_count : 0, 65536);
  device->max_shared_memory_per_block = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_workgroup_local_memory_size : 0,
      (is_gfx942 || is_gfx1100) ? 65536u : 49152u);
  device->max_shared_memory_per_block_optin = iree_hal_streaming_u32_or_default(
      execution ? execution->maximum_workgroup_local_memory_size_optin : 0,
      device->max_shared_memory_per_block);

  return iree_ok_status();
}

// Initializes a single device from a pyre device handle.
static iree_status_t iree_hal_streaming_initialize_device(
    iree_hal_streaming_device_registry_t* registry, hrx_device_t hrx_dev,
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_hal_streaming_device_t* out_device) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(hrx_dev);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_device, 0, sizeof(*out_device));
  // The entry was just zeroed, so the ordinal MUST be (re)assigned here.
  // Per-device pools, peer lookups, and context->device_ordinal all key off
  // it; if it stays 0 every device aliases device 0.
  out_device->ordinal = ordinal;
  out_device->runtime_generation = registry->runtime_generation;
  iree_atomic_store(&out_device->reset_epoch, 0, iree_memory_order_relaxed);

  // Store pyre device and extract HAL device for direct HAL usage.
  out_device->hrx_device = hrx_dev;
  out_device->hal_device = hrx_device_hal(hrx_dev);

  // Get device name from pyre.
  char name_buf[128] = {0};
  iree_status_t status = HRX_CALL(hrx_device_get_property(
      hrx_dev, HRX_DEVICE_PROPERTY_NAME, name_buf, sizeof(name_buf)));
  if (iree_status_is_ok(status) && name_buf[0] != '\0') {
    size_t len = strlen(name_buf);
    char* name_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(registry->host_allocator, len + 1,
                                  (void**)&name_copy));
    memcpy(name_copy, name_buf, len + 1);
    out_device->info.name = iree_make_string_view(name_copy, len);
  } else {
    iree_status_ignore(status);
    out_device->info.name = iree_string_view_empty();
  }

  // Use architecture as path.
  char arch_buf[64] = {0};
  status = HRX_CALL(hrx_device_get_property(
      hrx_dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch_buf, sizeof(arch_buf)));
  if (iree_status_is_ok(status) && arch_buf[0] != '\0') {
    size_t len = strlen(arch_buf);
    char* path_copy = NULL;
    iree_status_t path_status = iree_allocator_malloc(
        registry->host_allocator, len + 1, (void**)&path_copy);
    if (iree_status_is_ok(path_status)) {
      memcpy(path_copy, arch_buf, len + 1);
      out_device->info.path = iree_make_string_view(path_copy, len);
    } else {
      iree_status_ignore(path_status);
      out_device->info.path = iree_string_view_empty();
    }
  } else {
    iree_status_ignore(status);
    out_device->info.path = iree_string_view_empty();
  }

  // Query and initialize all device properties.
  status = iree_hal_streaming_query_device_info(out_device);

  // Initialize primary context flags with defaults.
  out_device->primary_context_flags.scheduling_mode =
      IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  out_device->primary_context_flags.map_host_memory = true;
  out_device->primary_context_flags.resize_local_mem_to_max = false;

  // Initialize primary context mutex for lazy initialization.
  iree_slim_mutex_initialize(&out_device->primary_context_mutex);

  // Initialize primary context reference count to 0.
  out_device->primary_context_ref_count = 0;
  out_device->retired_primary_context_ref_count = 0;

  // Initialize the arena block pool for graph allocations.
  // Use 64KB blocks as a good balance.
  if (iree_status_is_ok(status)) {
    const iree_host_size_t block_size = 64 * 1024;  // 64KB blocks
    iree_arena_block_pool_initialize(block_size, registry->host_allocator,
                                     &out_device->block_pool);
    status = iree_arena_block_pool_preallocate(&out_device->block_pool, 16);
  }

  // Primary context is NOT created here - it will be created lazily on first
  // access. This matches CUDA/HIP behavior where the primary context is not
  // active after init.
  out_device->primary_context = NULL;

  // Memory pools will be created when the primary context is created.
  out_device->default_mem_pool = NULL;
  out_device->current_mem_pool = NULL;

  iree_slim_mutex_initialize(&out_device->graph_memory_mutex);
  out_device->graph_memory_used_current = 0;
  out_device->graph_memory_used_high = 0;
  out_device->graph_memory_reserved_current = 0;
  out_device->graph_memory_reserved_high = 0;
  out_device->graph_memory_reusable_size_entries = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_execution_resource_table_initialize(
        out_device->hal_device, registry->host_allocator,
        &out_device->execution_resource_table);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_deinitialize_device(out_device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Deinitializes a device, releasing all its resources.
static void iree_hal_streaming_deinitialize_device(
    iree_hal_streaming_device_t* device) {
  if (!device) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get allocator from global registry for freeing string copies.
  iree_hal_streaming_device_registry_t* registry =
      iree_hal_streaming_device_registry();
  iree_allocator_t host_allocator =
      registry ? registry->host_allocator : iree_allocator_system();

  // Free the device name and path strings that were allocated during
  // initialization.
  if (device->info.path.size > 0 && device->info.path.data) {
    iree_allocator_free(host_allocator, (void*)device->info.path.data);
  }
  if (device->info.name.size > 0 && device->info.name.data) {
    iree_allocator_free(host_allocator, (void*)device->info.name.data);
  }
  device->info.path = iree_string_view_empty();
  device->info.name = iree_string_view_empty();

  // Release memory pools.
  hrx_mem_pool_release(device->current_mem_pool);
  device->current_mem_pool = NULL;
  hrx_mem_pool_release(device->default_mem_pool);
  device->default_mem_pool = NULL;

  // Release primary context (may not exist if never accessed).
  IREE_ASSERT(device->primary_context_ref_count == 0);
  IREE_ASSERT(device->retired_primary_context_ref_count == 0);
  iree_hal_streaming_context_release(device->primary_context);
  device->primary_context = NULL;

  iree_hal_streaming_execution_resource_table_deinitialize(
      &device->execution_resource_table);

  iree_hal_streaming_graph_memory_size_entry_t* graph_memory_entry =
      device->graph_memory_reusable_size_entries;
  device->graph_memory_reusable_size_entries = NULL;
  while (graph_memory_entry) {
    iree_hal_streaming_graph_memory_size_entry_t* next_entry =
        graph_memory_entry->next;
    iree_allocator_free(host_allocator, graph_memory_entry);
    graph_memory_entry = next_entry;
  }
  iree_slim_mutex_deinitialize(&device->graph_memory_mutex);

  // Deinitialize primary context mutex.
  iree_slim_mutex_deinitialize(&device->primary_context_mutex);

  // Deinitialize the arena block pool.
  iree_arena_block_pool_deinitialize(&device->block_pool);

  // HAL device and driver are owned by pyre — don't release here.
  // hrx_gpu_shutdown() handles cleanup.
  device->hal_device = NULL;
  device->hrx_device = NULL;

  IREE_TRACE_ZONE_END(z0);
}

// Queries device P2P capabilities and populates topology.
static iree_status_t iree_hal_streaming_query_p2p_capabilities(
    iree_hal_streaming_device_registry_t* registry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate P2P topology array.
  registry->p2p_link_count = registry->device_count * registry->device_count;
  const iree_host_size_t topology_size =
      registry->p2p_link_count * sizeof(iree_hal_streaming_p2p_link_t);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(registry->host_allocator, topology_size,
                                (void**)&registry->p2p_topology));
  memset(registry->p2p_topology, 0, topology_size);

  // Populate P2P links for all device pairs.
  iree_host_size_t link_index = 0;
  for (iree_host_size_t i = 0; i < registry->device_count; ++i) {
    for (iree_host_size_t j = 0; j < registry->device_count; ++j) {
      iree_hal_streaming_p2p_link_t* link =
          &registry->p2p_topology[link_index++];
      link->src_device = i;
      link->dst_device = j;
      if (i == j) {
        // Device can always access itself with best performance.
        link->access_supported = true;
        link->native_atomic_supported = true;
        link->cuda_array_access_supported = true;
        link->performance_rank = 100;   // Highest rank for same device.
        link->bandwidth_mbps = 900000;  // 900 GB/s typical for device memory.
        link->latency_ns = 10;          // Very low latency.
      } else {
        // TODO: Query actual P2P capabilities from pyre/HSA.
        link->access_supported = false;
        link->native_atomic_supported = false;
        link->cuda_array_access_supported = false;
        link->performance_rank = -1;  // Not supported.
        link->bandwidth_mbps = 0;
        link->latency_ns = 0;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Context registration
//===----------------------------------------------------------------------===//

void iree_hal_streaming_register_context(
    iree_hal_streaming_context_t* context) {
  if (!context) return;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device_registry->context_list.mutex);

  // Add to tail of list.
  context->context_list_entry.prev = device_registry->context_list.tail;
  context->context_list_entry.next = NULL;

  if (device_registry->context_list.tail) {
    device_registry->context_list.tail->context_list_entry.next = context;
  } else {
    // First context in list.
    device_registry->context_list.head = context;
  }
  device_registry->context_list.tail = context;

  // Retain for the global list.
  iree_hal_streaming_context_retain(context);

  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_unregister_context(
    iree_hal_streaming_context_t* context) {
  if (!context) return;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device_registry->context_list.mutex);

  // Check if the context is actually in the list.
  // A context might not be in the list if it failed during initialization
  // before it could be registered, or if this is called multiple times.
  // A context is in the list if it's either the head/tail or has neighbors.
  const bool was_in_list = context == device_registry->context_list.head ||
                           context == device_registry->context_list.tail ||
                           context->context_list_entry.prev ||
                           context->context_list_entry.next;
  if (was_in_list) {
    // Remove from list.
    if (context->context_list_entry.prev) {
      context->context_list_entry.prev->context_list_entry.next =
          context->context_list_entry.next;
    } else if (context == device_registry->context_list.head) {
      // Was head of list.
      device_registry->context_list.head = context->context_list_entry.next;
    }

    if (context->context_list_entry.next) {
      context->context_list_entry.next->context_list_entry.prev =
          context->context_list_entry.prev;
    } else if (context == device_registry->context_list.tail) {
      // Was tail of list.
      device_registry->context_list.tail = context->context_list_entry.prev;
    }

    // Clear list pointers.
    context->context_list_entry.next = NULL;
    context->context_list_entry.prev = NULL;
  }

  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  // Only release the global list reference if the context was actually in the
  // list.
  if (was_in_list) {
    iree_hal_streaming_context_release(context);
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_streaming_context_t* iree_hal_streaming_context_lookup_retain(
    const iree_hal_streaming_context_t* context) {
  if (!context) return NULL;
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return NULL;

  iree_hal_streaming_context_t* retained_context = NULL;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* candidate =
           device_registry->context_list.head;
       candidate; candidate = candidate->context_list_entry.next) {
    if (candidate == context) {
      iree_hal_streaming_context_retain(candidate);
      retained_context = candidate;
      break;
    }
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return retained_context;
}

bool iree_hal_streaming_context_is_registered(
    const iree_hal_streaming_context_t* context) {
  iree_hal_streaming_context_t* retained_context =
      iree_hal_streaming_context_lookup_retain(context);
  const bool found = retained_context != NULL;
  iree_hal_streaming_context_release(retained_context);
  return found;
}

void iree_hal_streaming_context_retire_all(void) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return;

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    iree_atomic_store(&context->accepting_work, 0, iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
}

void iree_hal_streaming_context_mark_all_teardown_quiesced(void) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return;

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    iree_hal_streaming_context_mark_teardown_quiesced(context);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
}

//===----------------------------------------------------------------------===//
// Global initialization via pyre
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_init_global(
    const iree_hal_device_create_params_extension_t* device_extensions,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_runtime_initialization_begin();
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  status = iree_hal_streaming_context_tls_initialize();
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_runtime_initialization_end(false);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (iree_hal_streaming_global_registry &&
      iree_hal_streaming_global_registry->initialized) {
    if (IREE_UNLIKELY(iree_hal_streaming_global_registry->device_extensions !=
                      device_extensions)) {
      iree_hal_streaming_runtime_initialization_end(false);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "streaming runtime is already initialized with a different HAL "
          "device extension configuration");
    }
    iree_hal_streaming_runtime_initialization_end(true);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Initialize pyre GPU subsystem (idempotent — handles HSA init,
  // driver registration, device enumeration).
  status = HRX_CALL(hrx_gpu_initialize_with_device_extensions(
      /*flags=*/0, device_extensions));
  if (!iree_status_is_ok(status)) {
    iree_status_t tls_status = iree_hal_streaming_context_tls_deinitialize();
    status = iree_status_join(status, tls_status);
    iree_hal_streaming_runtime_initialization_end(false);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Create global registry.
  status = iree_allocator_malloc(host_allocator,
                                 sizeof(iree_hal_streaming_device_registry_t),
                                 (void**)&iree_hal_streaming_global_registry);
  if (!iree_status_is_ok(status)) {
    iree_status_t shutdown_status = HRX_CALL(hrx_gpu_shutdown());
    status = iree_status_join(status, shutdown_status);
    iree_status_t tls_status = iree_hal_streaming_context_tls_deinitialize();
    status = iree_status_join(status, tls_status);
    iree_hal_streaming_runtime_initialization_end(false);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  memset(device_registry, 0, sizeof(*device_registry));
  device_registry->host_allocator = host_allocator;
  device_registry->device_extensions = device_extensions;
  if (IREE_UNLIKELY(!iree_hal_streaming_atomic_allocate_id(
          &iree_hal_streaming_next_runtime_generation, UINT64_MAX,
          &device_registry->runtime_generation))) {
    iree_allocator_free(host_allocator, device_registry);
    iree_hal_streaming_global_registry = NULL;
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "streaming runtime generation exhausted");
    iree_status_t shutdown_status = HRX_CALL(hrx_gpu_shutdown());
    status = iree_status_join(status, shutdown_status);
    iree_status_t tls_status = iree_hal_streaming_context_tls_deinitialize();
    status = iree_status_join(status, tls_status);
    iree_hal_streaming_runtime_initialization_end(false);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_slim_mutex_initialize(&device_registry->mutex);

  // Initialize context list.
  iree_slim_mutex_initialize(&device_registry->context_list.mutex);
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;

  // Enumerate GPU devices from pyre.
  int gpu_count = 0;
  status = HRX_CALL(hrx_gpu_device_count(&gpu_count));

  if (iree_status_is_ok(status)) {
    memset(device_registry->devices, 0, sizeof(device_registry->devices));
    device_registry->device_count = 0;

    for (int i = 0; i < gpu_count && i < IREE_HAL_STREAMING_MAX_DEVICES; ++i) {
      hrx_device_t hrx_dev = NULL;
      iree_status_t dev_status = HRX_CALL(hrx_gpu_device_get(i, &hrx_dev));
      if (!iree_status_is_ok(dev_status)) {
        iree_status_ignore(dev_status);
        continue;
      }

      iree_hal_streaming_device_t* device =
          &device_registry->devices[device_registry->device_count];

      dev_status = iree_hal_streaming_initialize_device(
          device_registry, hrx_dev, device_registry->device_count, device);
      if (!iree_status_is_ok(dev_status)) {
        iree_status_ignore(dev_status);
        continue;
      }

      device_registry->device_count++;
    }
  }

  // Must have at least one device.
  if (iree_status_is_ok(status) && device_registry->device_count == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no GPU devices found via pyre");
  }

  // Query P2P capabilities.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_query_p2p_capabilities(device_registry);
  }

  if (iree_status_is_ok(status)) {
    device_registry->initialized = true;
    iree_hal_streaming_runtime_initialization_end(true);
  } else {
    // Rollback uses the ordinary cleanup writer; release the initialization
    // exclusion first while preserving |status| as the primary error.
    iree_hal_streaming_runtime_initialization_end(false);
    iree_status_t cleanup_status = iree_hal_streaming_cleanup_global();
    status = iree_status_join(status, cleanup_status);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_cleanup_global(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool already_closed = false;
  iree_status_t status =
      iree_hal_streaming_runtime_cleanup_begin(&already_closed);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (already_closed) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_count = 0;
  if (device_registry) {
    status = iree_hal_streaming_context_snapshot_all(&contexts, &context_count);
  }

  // The retained list snapshot is also an exact ownership audit. Any edge
  // beyond the list, snapshot, device-primary publication, and caller TLS is a
  // live public/resource owner that could destroy the context after its inline
  // device registry has been freed.
  for (iree_host_size_t i = 0; i < context_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_streaming_context_t* context = contexts[i];
    iree_host_size_t expected_reference_count =
        2 + iree_hal_streaming_context_current_thread_tls_reference_count_for(
                context);
    iree_hal_streaming_device_t* device = context->device_entry;
    if (context->is_primary && device) {
      iree_slim_mutex_lock(&device->primary_context_mutex);
      if (device->primary_context == context) {
        ++expected_reference_count;
      }
      iree_slim_mutex_unlock(&device->primary_context_mutex);
    }
    const int32_t actual_reference_count =
        iree_atomic_ref_count_load(&context->ref_count);
    if (actual_reference_count < 0 ||
        (iree_host_size_t)actual_reference_count != expected_reference_count) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot clean up streaming runtime while a context has live "
          "ownership (%d actual, %" PRIhsz " cleanup-owned)",
          actual_reference_count, expected_reference_count);
    }
  }

  // A reset destroys its context generation but deliberately preserves each
  // logical retain until a matching release. With no object reference left to
  // expose that ownership in the context snapshot, audit the detached ledger
  // directly before any teardown mutation.
  for (iree_host_size_t i = 0;
       device_registry && i < device_registry->device_count &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_streaming_device_t* device = &device_registry->devices[i];
    iree_slim_mutex_lock(&device->primary_context_mutex);
    const int32_t retired_ref_count = device->retired_primary_context_ref_count;
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    if (retired_ref_count != 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot clean up streaming runtime with %d unreleased reset "
          "primary-context retains",
          retired_ref_count);
    }
  }

  // CUDA calls common cleanup directly, without HIP's teardown preparation.
  // Complete every fallible wait while the runtime remains fully published.
  // Execution failures still prove a terminal frontier and are deliberately
  // consumed; structural flush/wait failures keep cleanup retryable.
  for (iree_host_size_t i = 0; i < context_count && iree_status_is_ok(status);
       ++i) {
    if (iree_hal_streaming_context_is_teardown_certified(contexts[i])) {
      continue;
    }
    iree_status_t execution_status = iree_ok_status();
    status = iree_hal_streaming_context_quiesce_for_teardown(
        contexts[i], NULL, NULL, &execution_status);
    iree_status_ignore(execution_status);
  }

  const iree_host_size_t global_tls_reference_count =
      iree_hal_streaming_context_tls_reference_count();
  const iree_host_size_t local_tls_reference_count =
      iree_hal_streaming_context_current_thread_tls_reference_count();
  if (iree_status_is_ok(status) &&
      global_tls_reference_count != local_tls_reference_count) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot clean up streaming runtime while another execution context "
        "owns TLS contexts (%" PRIhsz " global, %" PRIhsz " local)",
        global_tls_reference_count, local_tls_reference_count);
  }

  // Clear all caller-owned references only after the foreign-owner preflight.
  // A marker-clear failure is precommit and leaves registry/key ownership and
  // the complete caller TLS state unchanged for retry.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_clear_current_thread();
  }
  if (iree_status_is_ok(status) &&
      iree_hal_streaming_context_tls_reference_count() != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "TLS context ownership was reestablished during global cleanup");
  }

  // Unregister the native destructor callback before destroying any registry
  // state. A failed key deletion keeps both the key and registry published so
  // cleanup can be retried and no callback can outlive its containing module.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_tls_deinitialize();
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release_snapshot_all(contexts, context_count);
    iree_hal_streaming_runtime_cleanup_end(false);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Key deletion is the final fallible precommit. Retire and detach the exact
  // snapshot only after it succeeds; from here cleanup is allocation-free and
  // no-fail. Contexts already certified by HIP need no second detach.
  for (iree_host_size_t i = 0; i < context_count; ++i) {
    if (iree_hal_streaming_context_is_teardown_certified(contexts[i])) {
      continue;
    }
    iree_hal_streaming_context_retire(contexts[i]);
    iree_hal_streaming_context_mark_teardown_quiesced(contexts[i]);
    iree_hal_streaming_context_detach_streams_quiesced(contexts[i],
                                                       /*abort_captures=*/true);
    IREE_ASSERT(iree_hal_streaming_context_is_teardown_certified(contexts[i]));
  }
  iree_hal_streaming_context_release_snapshot_all(contexts, context_count);

  if (!device_registry) {
    iree_hal_streaming_runtime_cleanup_end(true);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Drop each device publication before detaching the global list. A primary
  // preserved after failed native cleanup may now have only these two
  // teardown-ledger references. Releasing the device edge first guarantees
  // its eventual destructor can still consult the live context-list mutex and
  // the inline device entry when the list edge is released below.
  for (iree_host_size_t i = 0; i < device_registry->device_count; ++i) {
    iree_hal_streaming_device_t* device = &device_registry->devices[i];
    iree_slim_mutex_lock(&device->primary_context_mutex);
    iree_hal_streaming_context_t* primary_context = device->primary_context;
    device->primary_context = NULL;
    IREE_ASSERT(device->primary_context_ref_count == 0 &&
                    device->retired_primary_context_ref_count == 0,
                "global cleanup requires every public primary retain to be "
                "consumed");
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    iree_hal_streaming_context_release(primary_context);
  }

  // Force destroy all remaining contexts from the global list.
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  iree_hal_streaming_context_t* context_head =
      device_registry->context_list.head;
  device_registry->context_list.head = NULL;
  device_registry->context_list.tail = NULL;
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  while (context_head) {
    iree_hal_streaming_context_t* context = context_head;
    context_head = context->context_list_entry.next;
    context->context_list_entry.next = NULL;
    context->context_list_entry.prev = NULL;
    iree_hal_streaming_context_release(context);
  }

  iree_slim_mutex_deinitialize(&device_registry->context_list.mutex);
  iree_slim_mutex_lock(&device_registry->mutex);

  // Release all device resources.
  for (iree_host_size_t i = 0; i < device_registry->device_count; ++i) {
    iree_hal_streaming_deinitialize_device(&device_registry->devices[i]);
  }

  // Free P2P topology.
  iree_allocator_free(device_registry->host_allocator,
                      device_registry->p2p_topology);

  // Shutdown pyre GPU subsystem.
  hrx_status_ignore(hrx_gpu_shutdown());

  iree_slim_mutex_unlock(&device_registry->mutex);
  iree_slim_mutex_deinitialize(&device_registry->mutex);

  iree_hal_streaming_global_registry = NULL;
  iree_allocator_free(device_registry->host_allocator, device_registry);
  iree_hal_streaming_runtime_cleanup_end(true);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
