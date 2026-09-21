// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <string.h>

#include "hrx_internal.h"

#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
#include "iree/hal/drivers/amdgpu/api.h"
#endif

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
static iree_atomic_int32_t
    hrx_allocator_test_fail_virtual_reserve_after_hal_native =
        IREE_ATOMIC_VAR_INIT(0);
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

static void hrx_allocator_destroy_virtual_memory_wrapper(
    hrx_buffer_t virtual_buffer) {
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT(virtual_buffer->hal_buffer == NULL);
  hrx_device_release(virtual_buffer->device);
  iree_allocator_free(iree_allocator_system(), virtual_buffer);
}

hrx_allocator_t hrx_device_allocator(hrx_device_t device) {
  return &device->allocator;
}

void hrx_allocator_retain(hrx_allocator_t allocator) {
  if (!allocator) return;
  iree_hal_allocator_retain(allocator->hal_allocator);
  hrx_device_retain(allocator->device);
}

void hrx_allocator_release(hrx_allocator_t allocator) {
  if (!allocator) return;
  iree_hal_allocator_release(allocator->hal_allocator);
  hrx_device_release(allocator->device);
}

hrx_status_t hrx_allocator_allocate_buffer(hrx_allocator_t allocator,
                                           hrx_buffer_params_t params,
                                           size_t size, hrx_buffer_t* buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_allocate_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!allocator || !buffer) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "allocator or buffer is NULL"));
  }

  iree_hal_queue_family_affinity_t queue_family_affinity = 0;
  iree_status_t status = hrx_hal_queue_affinity_to_family_affinity(
      allocator->device->hal_device, params.queue_affinity,
      &queue_family_affinity);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }
  iree_hal_buffer_params_t hal_params = {
      .usage = (iree_hal_buffer_usage_t)params.usage,
      .access = (iree_hal_memory_access_t)params.access,
      .type = (iree_hal_memory_type_t)params.type,
      .queue_family_affinity = queue_family_affinity,
  };

  iree_hal_buffer_t* hal_buffer = NULL;
  status =
      iree_hal_allocator_allocate_buffer(allocator->hal_allocator, hal_params,
                                         (iree_device_size_t)size, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_buffer_t buf = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(hrx_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_status)) {
    iree_hal_buffer_release(hal_buffer);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(alloc_status));
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->device = allocator->device;
  hrx_device_retain(buf->device);
  buf->mem_type = params.type;
  buf->size = size;
  *buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_allocator_import_buffer(hrx_allocator_t allocator,
                                         hrx_buffer_params_t params,
                                         void* host_ptr, size_t size,
                                         hrx_buffer_t* buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_import_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!allocator || !host_ptr || !buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "allocator, host_ptr, or buffer is NULL"));
  }

  iree_hal_queue_family_affinity_t queue_family_affinity = 0;
  iree_status_t status = hrx_hal_queue_affinity_to_family_affinity(
      allocator->device->hal_device, params.queue_affinity,
      &queue_family_affinity);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }
  iree_hal_buffer_params_t hal_params = {
      .usage = (iree_hal_buffer_usage_t)params.usage,
      .access = (iree_hal_memory_access_t)params.access,
      .type = (iree_hal_memory_type_t)params.type,
      .queue_family_affinity = queue_family_affinity,
  };

  iree_hal_external_buffer_t ext = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = 0,
      .size = (iree_device_size_t)size,
      .handle.host_allocation.ptr = host_ptr,
  };

  iree_hal_buffer_t* hal_buffer = NULL;
  status = iree_hal_allocator_import_buffer(
      allocator->hal_allocator, hal_params, &ext,
      iree_hal_buffer_release_callback_null(), &hal_buffer);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_buffer_t buf = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(hrx_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_status)) {
    iree_hal_buffer_release(hal_buffer);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(alloc_status));
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->device = allocator->device;
  hrx_device_retain(buf->device);
  buf->mem_type = params.type;
  buf->size = size;
  *buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

//===----------------------------------------------------------------------===//
// Virtual memory
//===----------------------------------------------------------------------===//

hrx_status_t hrx_allocator_query_virtual_memory(hrx_allocator_t allocator,
                                                hrx_memory_type_t mem_type,
                                                bool* supported,
                                                size_t* min_page_size,
                                                size_t* recommended_page_size) {
  if (!allocator || !supported) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "allocator or supported is NULL");
  }

  if (!iree_hal_allocator_supports_virtual_memory(allocator->hal_allocator)) {
    *supported = false;
    if (min_page_size) *min_page_size = 0;
    if (recommended_page_size) *recommended_page_size = 0;
    return hrx_ok_status();
  }

  iree_hal_buffer_params_t hal_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = (iree_hal_memory_type_t)mem_type,
  };
  iree_device_size_t hal_minimum_page_size = 0;
  iree_device_size_t hal_recommended_page_size = 0;
  iree_status_t status = iree_hal_allocator_virtual_memory_query_granularity(
      allocator->hal_allocator, hal_params, &hal_minimum_page_size,
      &hal_recommended_page_size);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  *supported = true;
  if (min_page_size) *min_page_size = (size_t)hal_minimum_page_size;
  if (recommended_page_size) {
    *recommended_page_size = (size_t)hal_recommended_page_size;
  }
  return hrx_ok_status();
}

static hrx_status_t hrx_allocator_virtual_memory_reserve_impl(
    hrx_allocator_t allocator, hrx_queue_affinity_t affinity, size_t size,
    size_t minimum_alignment, uintptr_t requested_address,
    bool use_placement_hints, hrx_buffer_t* virtual_buffer) {
  if (!allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *virtual_buffer = NULL;

  iree_hal_queue_family_affinity_t queue_family_affinity = 0;
  iree_status_t status = hrx_hal_queue_affinity_to_family_affinity(
      allocator->device->hal_device, affinity, &queue_family_affinity);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  hrx_buffer_t buffer = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(hrx_buffer_s),
                                 (void**)&buffer);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  memset(buffer, 0, sizeof(*buffer));
  iree_atomic_ref_count_init(&buffer->ref_count);
  buffer->device = allocator->device;
  hrx_device_retain(buffer->device);
  buffer->mem_type = HRX_MEMORY_TYPE_DEVICE_LOCAL;
  buffer->size = size;

  // The HRX wrapper and its device retain are fully owned before native
  // acquisition, so publication never needs a post-acquisition allocation.
  iree_hal_buffer_t* hal_buffer = NULL;
  if (use_placement_hints) {
    status = iree_hal_allocator_virtual_memory_reserve_at(
        allocator->hal_allocator, queue_family_affinity,
        (iree_device_size_t)size, (iree_device_size_t)minimum_alignment,
        (iree_device_size_t)requested_address, &hal_buffer);
  } else {
    status = iree_hal_allocator_virtual_memory_reserve(
        allocator->hal_allocator, queue_family_affinity,
        (iree_device_size_t)size, &hal_buffer);
  }
  if (!iree_status_is_ok(status)) {
    hrx_allocator_destroy_virtual_memory_wrapper(buffer);
    return hrx_status_from_iree(status);
  }
  buffer->hal_buffer = hal_buffer;

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
  if (iree_atomic_exchange(
          &hrx_allocator_test_fail_virtual_reserve_after_hal_native, 0,
          iree_memory_order_acq_rel) != 0) {
    iree_status_t primary_status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "injected HRX failure after HAL virtual reservation");
    iree_status_t cleanup_status =
        iree_hal_amdgpu_allocator_virtual_memory_release_or_quarantine(
            allocator->hal_allocator, buffer->hal_buffer);
    buffer->hal_buffer = NULL;
    hrx_allocator_destroy_virtual_memory_wrapper(buffer);
    return hrx_status_from_iree(
        iree_status_join(primary_status, cleanup_status));
  }
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION

  *virtual_buffer = buffer;
  return hrx_ok_status();
}

hrx_status_t hrx_allocator_virtual_memory_reserve(
    hrx_allocator_t allocator, hrx_queue_affinity_t affinity, size_t size,
    hrx_buffer_t* virtual_buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_virtual_memory_reserve");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  hrx_status_t status = hrx_allocator_virtual_memory_reserve_impl(
      allocator, affinity, size, /*minimum_alignment=*/0,
      /*requested_address=*/0, /*use_placement_hints=*/false, virtual_buffer);
  HRX_RETURN_AND_END_ZONE(z0, status);
}

hrx_status_t hrx_allocator_virtual_memory_reserve_at(
    hrx_allocator_t allocator, hrx_queue_affinity_t affinity, size_t size,
    size_t minimum_alignment, uintptr_t requested_address,
    hrx_buffer_t* virtual_buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_virtual_memory_reserve_at");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  hrx_status_t status = hrx_allocator_virtual_memory_reserve_impl(
      allocator, affinity, size, minimum_alignment, requested_address,
      /*use_placement_hints=*/true, virtual_buffer);
  HRX_RETURN_AND_END_ZONE(z0, status);
}

hrx_status_t hrx_allocator_virtual_memory_release(hrx_allocator_t allocator,
                                                  hrx_buffer_t virtual_buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_virtual_memory_release");
  if (virtual_buffer) {
    HRX_TRACE_ZONE_APPEND_BYTES(z0, virtual_buffer->size);
  }
  if (!allocator || !virtual_buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument"));
  }
  iree_status_t status = iree_hal_allocator_virtual_memory_release(
      allocator->hal_allocator, virtual_buffer->hal_buffer);
  if (iree_status_is_ok(status)) {
    // Release consumes the HAL buffer only on success. Keep the wrapper usable
    // when active mappings prevent release so the caller can unmap and retry.
    virtual_buffer->hal_buffer = NULL;
    hrx_device_release(virtual_buffer->device);
    iree_allocator_free(iree_allocator_system(), virtual_buffer);
  }
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_allocator_virtual_memory_alias(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, hrx_memory_access_t allowed_access,
    hrx_buffer_t* out_alias_buffer) {
  if (!allocator || !virtual_buffer || !out_alias_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_alias_buffer = NULL;
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_buffer_t* hal_alias_buffer = NULL;
  iree_status_t status = iree_hal_amdgpu_allocator_virtual_memory_alias(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size,
      (iree_hal_memory_access_t)allowed_access, &hal_alias_buffer);
  if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);

  hrx_buffer_t alias_buffer = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(*alias_buffer),
                                 (void**)&alias_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(hal_alias_buffer);
    return hrx_status_from_iree(status);
  }
  memset(alias_buffer, 0, sizeof(*alias_buffer));
  iree_atomic_ref_count_init(&alias_buffer->ref_count);
  alias_buffer->hal_buffer = hal_alias_buffer;
  alias_buffer->device = allocator->device;
  hrx_device_retain(alias_buffer->device);
  alias_buffer->mem_type = HRX_MEMORY_TYPE_DEVICE_LOCAL;
  alias_buffer->size = size;
  *out_alias_buffer = alias_buffer;
  return hrx_ok_status();
#else
  (void)virtual_offset;
  (void)size;
  (void)allowed_access;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "AMDGPU virtual memory is not compiled in");
#endif
}

hrx_status_t hrx_allocator_physical_memory_allocate(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, size_t size,
    hrx_physical_memory_t* physical) {
  if (!allocator || !physical) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  iree_hal_buffer_params_t hal_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = (iree_hal_memory_type_t)mem_type,
  };
  return hrx_status_from_iree(iree_hal_allocator_physical_memory_allocate(
      allocator->hal_allocator, hal_params, (iree_device_size_t)size,
      iree_allocator_system(), (iree_hal_physical_memory_t**)physical));
}

hrx_status_t hrx_allocator_physical_memory_free(
    hrx_allocator_t allocator, hrx_physical_memory_t physical) {
  if (!allocator || !physical) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return hrx_status_from_iree(iree_hal_allocator_physical_memory_free(
      allocator->hal_allocator, (iree_hal_physical_memory_t*)physical));
}

hrx_status_t hrx_allocator_virtual_memory_map(hrx_allocator_t allocator,
                                              hrx_buffer_t virtual_buffer,
                                              size_t virtual_offset,
                                              hrx_physical_memory_t physical,
                                              size_t physical_offset,
                                              size_t size) {
  if (!allocator || !virtual_buffer || !physical) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return hrx_status_from_iree(iree_hal_allocator_virtual_memory_map(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_hal_physical_memory_t*)physical,
      (iree_device_size_t)physical_offset, (iree_device_size_t)size));
}

hrx_status_t hrx_allocator_virtual_memory_unmap(hrx_allocator_t allocator,
                                                hrx_buffer_t virtual_buffer,
                                                size_t virtual_offset,
                                                size_t size) {
  if (!allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return hrx_status_from_iree(iree_hal_allocator_virtual_memory_unmap(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size));
}

static iree_status_t hrx_virtual_memory_access_scope_to_hal(
    hrx_virtual_memory_access_scope_t access_scope,
    iree_hal_virtual_memory_access_scope_t* out_access_scope) {
  switch (access_scope) {
    case HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL:
      *out_access_scope = IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL;
      return iree_ok_status();
    case HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST:
      *out_access_scope = IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST;
      return iree_ok_status();
    case HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE:
      *out_access_scope = IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid virtual-memory access scope");
  }
}

hrx_status_t hrx_allocator_virtual_memory_protect(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size,
    hrx_virtual_memory_access_scope_t access_scope,
    hrx_memory_protection_t protection) {
  if (!allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  iree_hal_virtual_memory_access_scope_t hal_access_scope;
  iree_status_t status =
      hrx_virtual_memory_access_scope_to_hal(access_scope, &hal_access_scope);
  if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);
  return hrx_status_from_iree(iree_hal_allocator_virtual_memory_protect(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size,
      /*queue_family_affinity=*/IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      hal_access_scope, (iree_hal_memory_protection_t)protection));
}

hrx_status_t hrx_allocator_virtual_memory_protect_peer(
    hrx_allocator_t reservation_allocator, hrx_allocator_t access_allocator,
    hrx_buffer_t virtual_buffer, size_t virtual_offset, size_t size,
    hrx_memory_protection_t protection) {
  if (!reservation_allocator || !access_allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  return hrx_status_from_iree(
      iree_hal_amdgpu_allocator_virtual_memory_protect_peer(
          reservation_allocator->hal_allocator, access_allocator->hal_allocator,
          virtual_buffer->hal_buffer, (iree_device_size_t)virtual_offset,
          (iree_device_size_t)size, (iree_hal_memory_protection_t)protection));
#else
  (void)virtual_offset;
  (void)size;
  (void)protection;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "peer virtual-memory protection requires the AMDGPU "
                         "driver");
#endif
}

hrx_status_t hrx_allocator_vmm_native_operation_prepare_access(
    hrx_allocator_t reservation_allocator,
    hrx_allocator_t access_allocator_or_null, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, hrx_queue_affinity_t affinity,
    hrx_virtual_memory_access_scope_t access_scope,
    hrx_memory_protection_t protection, size_t physical_memory_count,
    hrx_physical_memory_t const* physical_memories,
    hrx_vmm_native_operation_t* out_operation) {
  if (!reservation_allocator || !virtual_buffer || !out_operation ||
      physical_memory_count == 0 || !physical_memories) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_operation = NULL;
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_queue_family_affinity_t queue_family_affinity = 0;
  hrx_device_t affinity_device = access_allocator_or_null
                                     ? access_allocator_or_null->device
                                     : reservation_allocator->device;
  iree_status_t status = hrx_hal_queue_affinity_to_family_affinity(
      affinity_device->hal_device, affinity, &queue_family_affinity);
  if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);

  iree_hal_virtual_memory_access_scope_t hal_access_scope;
  status =
      hrx_virtual_memory_access_scope_to_hal(access_scope, &hal_access_scope);
  if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  status = iree_hal_amdgpu_allocator_vmm_native_operation_prepare_access(
      reservation_allocator->hal_allocator,
      access_allocator_or_null ? access_allocator_or_null->hal_allocator : NULL,
      virtual_buffer->hal_buffer, (iree_device_size_t)virtual_offset,
      (iree_device_size_t)size, queue_family_affinity, hal_access_scope,
      (iree_hal_memory_protection_t)protection,
      (iree_host_size_t)physical_memory_count,
      (iree_hal_physical_memory_t* const*)physical_memories,
      iree_allocator_system(), &operation);
  if (iree_status_is_ok(status)) {
    *out_operation = (hrx_vmm_native_operation_t)operation;
  }
  return hrx_status_from_iree(status);
#else
  (void)access_allocator_or_null;
  (void)virtual_offset;
  (void)size;
  (void)affinity;
  (void)access_scope;
  (void)protection;
  (void)physical_memory_count;
  (void)physical_memories;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "prepared VMM operations require AMDGPU");
#endif
}

hrx_status_t hrx_allocator_vmm_native_operation_prepare_unmap(
    hrx_allocator_t reservation_allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size,
    hrx_vmm_native_operation_t* out_operation) {
  if (!reservation_allocator || !virtual_buffer || !out_operation) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_operation = NULL;
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  iree_status_t status =
      iree_hal_amdgpu_allocator_vmm_native_operation_prepare_unmap(
          reservation_allocator->hal_allocator, virtual_buffer->hal_buffer,
          (iree_device_size_t)virtual_offset, (iree_device_size_t)size,
          iree_allocator_system(), &operation);
  if (iree_status_is_ok(status)) {
    *out_operation = (hrx_vmm_native_operation_t)operation;
  }
  return hrx_status_from_iree(status);
#else
  (void)virtual_offset;
  (void)size;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "prepared VMM operations require AMDGPU");
#endif
}

hrx_status_t hrx_allocator_vmm_native_operation_prepare_release_reservation(
    hrx_allocator_t reservation_allocator, hrx_buffer_t virtual_buffer,
    hrx_vmm_native_operation_t* out_operation) {
  if (!reservation_allocator || !virtual_buffer || !out_operation) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_operation = NULL;
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  iree_status_t status =
      iree_hal_amdgpu_allocator_vmm_native_operation_prepare_release_reservation(
          reservation_allocator->hal_allocator, virtual_buffer->hal_buffer,
          iree_allocator_system(), &operation);
  if (iree_status_is_ok(status)) {
    *out_operation = (hrx_vmm_native_operation_t)operation;
  }
  return hrx_status_from_iree(status);
#else
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "prepared VMM operations require AMDGPU");
#endif
}

hrx_status_t hrx_allocator_vmm_native_operation_prepare_free_physical(
    hrx_allocator_t creator_allocator, hrx_physical_memory_t physical_memory,
    hrx_vmm_native_operation_t* out_operation) {
  if (!creator_allocator || !physical_memory || !out_operation) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_operation = NULL;
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_amdgpu_vmm_native_operation_t* operation = NULL;
  iree_status_t status =
      iree_hal_amdgpu_allocator_vmm_native_operation_prepare_free_physical(
          creator_allocator->hal_allocator,
          (iree_hal_physical_memory_t*)physical_memory, iree_allocator_system(),
          &operation);
  if (iree_status_is_ok(status)) {
    *out_operation = (hrx_vmm_native_operation_t)operation;
  }
  return hrx_status_from_iree(status);
#else
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "prepared VMM operations require AMDGPU");
#endif
}

hrx_vmm_native_status_t hrx_vmm_native_operation_apply(
    hrx_vmm_native_operation_t operation) {
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  return (hrx_vmm_native_status_t)iree_hal_amdgpu_vmm_native_operation_apply(
      (iree_hal_amdgpu_vmm_native_operation_t*)operation);
#else
  (void)operation;
  return UINT32_MAX;
#endif
}

bool hrx_vmm_native_status_is_success(hrx_vmm_native_status_t status) {
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  return status == (hrx_vmm_native_status_t)HSA_STATUS_SUCCESS;
#else
  (void)status;
  return false;
#endif
}

bool hrx_vmm_native_operation_may_invoke_callbacks(
    hrx_vmm_native_operation_t operation) {
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  return iree_hal_amdgpu_vmm_native_operation_may_invoke_callbacks(
      (iree_hal_amdgpu_vmm_native_operation_t*)operation);
#else
  (void)operation;
  return false;
#endif
}

void hrx_vmm_native_operation_destroy(hrx_vmm_native_operation_t operation) {
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_amdgpu_vmm_native_operation_destroy(
      (iree_hal_amdgpu_vmm_native_operation_t*)operation);
#else
  (void)operation;
#endif
}

void hrx_allocator_virtual_memory_dispose_consumed_reservation(
    hrx_allocator_t reservation_allocator, hrx_buffer_t virtual_buffer) {
  IREE_ASSERT_ARGUMENT(reservation_allocator);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_amdgpu_allocator_virtual_memory_dispose_consumed_reservation(
      reservation_allocator->hal_allocator, virtual_buffer->hal_buffer);
  virtual_buffer->hal_buffer = NULL;
  hrx_device_release(virtual_buffer->device);
  iree_allocator_free(iree_allocator_system(), virtual_buffer);
#else
  IREE_ASSERT_UNREACHABLE("consumed VMM reservation requires AMDGPU");
#endif
}

void hrx_allocator_physical_memory_dispose_consumed(
    hrx_allocator_t creator_allocator, hrx_physical_memory_t physical_memory) {
  IREE_ASSERT_ARGUMENT(creator_allocator);
  IREE_ASSERT_ARGUMENT(physical_memory);
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_hal_amdgpu_allocator_physical_memory_dispose_consumed(
      creator_allocator->hal_allocator,
      (iree_hal_physical_memory_t*)physical_memory);
#else
  IREE_ASSERT_UNREACHABLE("consumed VMM physical memory requires AMDGPU");
#endif
}

hrx_status_t hrx_allocator_virtual_memory_release_or_quarantine(
    hrx_allocator_t reservation_allocator, hrx_buffer_t virtual_buffer) {
  if (!reservation_allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  iree_status_t status =
      iree_hal_amdgpu_allocator_virtual_memory_release_or_quarantine(
          reservation_allocator->hal_allocator, virtual_buffer->hal_buffer);
  // The lower call either consumed the native owner or transferred it into its
  // embedded retry node. The HRX wrapper owns no resource after that transfer.
  virtual_buffer->hal_buffer = NULL;
  hrx_allocator_destroy_virtual_memory_wrapper(virtual_buffer);
  return hrx_status_from_iree(status);
#else
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "VMM cleanup quarantine requires AMDGPU");
#endif
}

hrx_status_t hrx_allocator_physical_memory_free_or_quarantine(
    hrx_allocator_t creator_allocator, hrx_physical_memory_t physical_memory) {
  if (!creator_allocator || !physical_memory) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  return hrx_status_from_iree(
      iree_hal_amdgpu_allocator_physical_memory_free_or_quarantine(
          creator_allocator->hal_allocator,
          (iree_hal_physical_memory_t*)physical_memory));
#else
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "VMM cleanup quarantine requires AMDGPU");
#endif
}

hrx_status_t hrx_allocator_vmm_quarantine_drain(hrx_allocator_t allocator) {
  if (!allocator) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
#if defined(HRX_HAS_IREE_AMDGPU_DRIVER)
  return hrx_status_from_iree(
      iree_hal_amdgpu_allocator_vmm_quarantine_drain(allocator->hal_allocator));
#else
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "VMM cleanup quarantine requires AMDGPU");
#endif
}

#if defined(IREE_HAL_AMDGPU_TEST_INSTRUMENTATION)
void hrx_allocator_test_fail_virtual_reserve_after_hal_native_once(void) {
  iree_atomic_store(&hrx_allocator_test_fail_virtual_reserve_after_hal_native,
                    1, iree_memory_order_release);
}

void hrx_allocator_test_reset_vmm_quarantine_observability(void) {
  iree_atomic_store(&hrx_allocator_test_fail_virtual_reserve_after_hal_native,
                    0, iree_memory_order_release);
  iree_hal_amdgpu_vmm_test_reset_quarantine_observability();
}

void hrx_allocator_test_fail_lower_virtual_reserve_after_native_once(void) {
  iree_hal_amdgpu_vmm_test_fail_virtual_reserve_after_native_once();
}

uint64_t hrx_allocator_test_last_virtual_reserve_address(void) {
  return iree_hal_amdgpu_vmm_test_last_virtual_reserve_address();
}

uint64_t hrx_allocator_test_last_virtual_reserve_alignment(void) {
  return iree_hal_amdgpu_vmm_test_last_virtual_reserve_alignment();
}

void hrx_allocator_test_fail_lower_cleanup_count(bool physical_memory,
                                                 int failure_count) {
  iree_hal_amdgpu_vmm_test_fail_cleanup_count(physical_memory, failure_count);
}

uint64_t hrx_allocator_test_vmm_quarantine_count(void) {
  return iree_hal_amdgpu_vmm_test_quarantine_count();
}

uint64_t hrx_allocator_test_vmm_cleanup_attempt_count(bool physical_memory) {
  return iree_hal_amdgpu_vmm_test_cleanup_attempt_count(physical_memory);
}

uint32_t hrx_allocator_test_vmm_last_cleanup_status(bool physical_memory) {
  return iree_hal_amdgpu_vmm_test_last_cleanup_status(physical_memory);
}

void hrx_allocator_test_arm_vmm_quarantine_drain_pause(void) {
  iree_hal_amdgpu_vmm_test_arm_quarantine_drain_pause();
}

void hrx_allocator_test_wait_vmm_quarantine_drain_paused(void) {
  iree_hal_amdgpu_vmm_test_wait_quarantine_drain_paused();
}

void hrx_allocator_test_release_vmm_quarantine_drain_pause(void) {
  iree_hal_amdgpu_vmm_test_release_quarantine_drain_pause();
}

int hrx_allocator_test_vmm_quarantine_drain_attempt_kind(int attempt_ordinal) {
  return iree_hal_amdgpu_vmm_test_drain_attempt_kind(attempt_ordinal);
}
#endif  // IREE_HAL_AMDGPU_TEST_INSTRUMENTATION
