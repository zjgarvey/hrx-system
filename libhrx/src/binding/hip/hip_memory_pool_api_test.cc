// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

const char* CandidateLibPath() {
#if defined(IREE_HIP_POOL_ADMISSION_TEST)
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64_PHASE");
      env && *env != '\0') {
    return env;
  }
#endif  // IREE_HIP_POOL_ADMISSION_TEST
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return "libamdhip64.so";
#endif
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipDeviceGetDefaultMemPoolFn = hipError_t (*)(hipMemPool_t* pool,
                                                    int device);
using HipDeviceSetMemPoolFn = hipError_t (*)(int device, hipMemPool_t pool);
using HipMemPoolCreateFn = hipError_t (*)(hipMemPool_t* pool,
                                          const hipMemPoolProps* properties);
using HipMemPoolDestroyFn = hipError_t (*)(hipMemPool_t pool);
using HipMemPoolSetAttributeFn = hipError_t (*)(hipMemPool_t pool,
                                                hipMemPool_attribute attribute,
                                                void* value);
using HipMemPoolGetAttributeFn = hipError_t (*)(hipMemPool_t pool,
                                                hipMemPool_attribute attribute,
                                                void* value);
using HipMemPoolSetAccessFn = hipError_t (*)(hipMemPool_t pool,
                                             const hipMemAccessDesc* map,
                                             size_t count);
using HipMemPoolGetAccessFn = hipError_t (*)(hipMemAccessFlags* flags,
                                             hipMemPool_t pool,
                                             hipMemLocation* location);
using HipMemPoolTrimToFn = hipError_t (*)(hipMemPool_t pool,
                                          size_t minimum_bytes_to_keep);
using HipMemPoolExportToShareableHandleFn =
    hipError_t (*)(void* handle, hipMemPool_t pool,
                   hipMemAllocationHandleType handle_type, unsigned int flags);
using HipMemPoolImportFromShareableHandleFn =
    hipError_t (*)(hipMemPool_t* pool, void* handle,
                   hipMemAllocationHandleType handle_type, unsigned int flags);
using HipMemPoolExportPointerFn =
    hipError_t (*)(hipMemPoolPtrExportData* share_data, void* pointer);
using HipMemPoolImportPointerFn = hipError_t (*)(
    void** pointer, hipMemPool_t pool, hipMemPoolPtrExportData* share_data);
using HipDeviceGetMemPoolFn = hipError_t (*)(hipMemPool_t* pool, int device);
using HipMemGetMemPoolFn = hipError_t (*)(hipMemPool_t* pool,
                                          hipMemLocation* location,
                                          hipMemAllocationType type);
using HipMemSetMemPoolFn = hipError_t (*)(hipMemLocation* location,
                                          hipMemAllocationType type,
                                          hipMemPool_t pool);
using HipMallocAsyncFn = hipError_t (*)(void** pointer, size_t size,
                                        hipStream_t stream);
using HipMallocFromPoolAsyncFn = hipError_t (*)(void** pointer, size_t size,
                                                hipMemPool_t pool,
                                                hipStream_t stream);
using HipFreeAsyncFn = hipError_t (*)(void* pointer, hipStream_t stream);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddMemAllocNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, void* allocation_parameters);
#if defined(IREE_HIP_POOL_ADMISSION_TEST)
using PoolPhaseObserverFn = void (*)(int phase, void* object, void* user_data);
using SetPoolPhaseObserverFn = hipError_t (*)(PoolPhaseObserverFn observer,
                                              void* user_data);
using QueryReaderDepthFn = uint32_t (*)(void);
using QueryReaderCountFn = uint64_t (*)(void);
using NativeCallbackFn = void (*)(void* user_data);
using RunNativeCallbackWindowFn = void (*)(NativeCallbackFn callback,
                                           void* user_data);
#endif  // IREE_HIP_POOL_ADMISSION_TEST

// Owns an RTLD_LOCAL HIP runtime instance and the entry points exercised by
// this test. All calls use the loaded library instead of a link-time runtime.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Returns the current device ordinal.
  HipGetDeviceFn get_device = nullptr;
  // Creates a stream for asynchronous allocation.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys the stream used by asynchronous allocation.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Returns a device's default memory pool.
  HipDeviceGetDefaultMemPoolFn device_get_default_mem_pool = nullptr;
  // Selects a device's current memory pool.
  HipDeviceSetMemPoolFn device_set_mem_pool = nullptr;
  // Creates a user-owned memory pool.
  HipMemPoolCreateFn mem_pool_create = nullptr;
  // Destroys a user-owned memory pool.
  HipMemPoolDestroyFn mem_pool_destroy = nullptr;
  // Sets a memory-pool attribute.
  HipMemPoolSetAttributeFn mem_pool_set_attribute = nullptr;
  // Reads a memory-pool attribute.
  HipMemPoolGetAttributeFn mem_pool_get_attribute = nullptr;
  // Sets a memory pool's device access.
  HipMemPoolSetAccessFn mem_pool_set_access = nullptr;
  // Reads a memory pool's device access.
  HipMemPoolGetAccessFn mem_pool_get_access = nullptr;
  // Releases unused memory retained by a pool.
  HipMemPoolTrimToFn mem_pool_trim_to = nullptr;
  // Exports a memory pool through the unsupported IPC surface.
  HipMemPoolExportToShareableHandleFn mem_pool_export_to_shareable_handle =
      nullptr;
  // Imports a memory pool through the unsupported IPC surface.
  HipMemPoolImportFromShareableHandleFn mem_pool_import_from_shareable_handle =
      nullptr;
  // Exports a pointer through the unsupported pool IPC surface.
  HipMemPoolExportPointerFn mem_pool_export_pointer = nullptr;
  // Imports a pointer through the unsupported pool IPC surface.
  HipMemPoolImportPointerFn mem_pool_import_pointer = nullptr;
  // Returns a device's currently selected memory pool.
  HipDeviceGetMemPoolFn device_get_mem_pool = nullptr;
  // Returns the selected pool for a memory location and allocation type.
  HipMemGetMemPoolFn mem_get_mem_pool = nullptr;
  // Selects the pool for a memory location and allocation type.
  HipMemSetMemPoolFn mem_set_mem_pool = nullptr;
  // Allocates stream-ordered memory from the selected default pool.
  HipMallocAsyncFn malloc_async = nullptr;
  // Allocates stream-ordered memory from an explicit pool.
  HipMallocFromPoolAsyncFn malloc_from_pool_async = nullptr;
  // Frees stream-ordered memory.
  HipFreeAsyncFn free_async = nullptr;
  // Creates a graph template.
  HipGraphCreateFn graph_create = nullptr;
  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;
  // Adds a memory-allocation node to a graph template.
  HipGraphAddMemAllocNodeFn graph_add_mem_alloc_node = nullptr;
#if defined(IREE_HIP_POOL_ADMISSION_TEST)
  // Installs or clears the instrumented lifecycle phase observer.
  SetPoolPhaseObserverFn set_phase_observer = nullptr;
  // Returns this thread's lifecycle-reader nesting depth.
  QueryReaderDepthFn reader_depth = nullptr;
  // Returns the process-wide number of outer lifecycle readers.
  QueryReaderCountFn reader_count = nullptr;
  // Runs a callback with the native teardown reentry guard active.
  RunNativeCallbackWindowFn run_native_callback_window = nullptr;
#endif  // IREE_HIP_POOL_ADMISSION_TEST
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipMemoryPoolApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      api_.library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
      if (!api_.library) {
        GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                     << dlerror();
      }

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.get_device =
          ResolveHipSymbol<HipGetDeviceFn>(api_.library, "hipGetDevice");
      api_.stream_create =
          ResolveHipSymbol<HipStreamCreateFn>(api_.library, "hipStreamCreate");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.device_get_default_mem_pool =
          ResolveHipSymbol<HipDeviceGetDefaultMemPoolFn>(
              api_.library, "hipDeviceGetDefaultMemPool");
      api_.device_set_mem_pool = ResolveHipSymbol<HipDeviceSetMemPoolFn>(
          api_.library, "hipDeviceSetMemPool");
      api_.mem_pool_create = ResolveHipSymbol<HipMemPoolCreateFn>(
          api_.library, "hipMemPoolCreate");
      api_.mem_pool_destroy = ResolveHipSymbol<HipMemPoolDestroyFn>(
          api_.library, "hipMemPoolDestroy");
      api_.mem_pool_set_attribute = ResolveHipSymbol<HipMemPoolSetAttributeFn>(
          api_.library, "hipMemPoolSetAttribute");
      api_.mem_pool_get_attribute = ResolveHipSymbol<HipMemPoolGetAttributeFn>(
          api_.library, "hipMemPoolGetAttribute");
      api_.mem_pool_set_access = ResolveHipSymbol<HipMemPoolSetAccessFn>(
          api_.library, "hipMemPoolSetAccess");
      api_.mem_pool_get_access = ResolveHipSymbol<HipMemPoolGetAccessFn>(
          api_.library, "hipMemPoolGetAccess");
      api_.mem_pool_trim_to = ResolveHipSymbol<HipMemPoolTrimToFn>(
          api_.library, "hipMemPoolTrimTo");
      api_.mem_pool_export_to_shareable_handle =
          ResolveHipSymbol<HipMemPoolExportToShareableHandleFn>(
              api_.library, "hipMemPoolExportToShareableHandle");
      api_.mem_pool_import_from_shareable_handle =
          ResolveHipSymbol<HipMemPoolImportFromShareableHandleFn>(
              api_.library, "hipMemPoolImportFromShareableHandle");
      api_.mem_pool_export_pointer =
          ResolveHipSymbol<HipMemPoolExportPointerFn>(
              api_.library, "hipMemPoolExportPointer");
      api_.mem_pool_import_pointer =
          ResolveHipSymbol<HipMemPoolImportPointerFn>(
              api_.library, "hipMemPoolImportPointer");
      api_.device_get_mem_pool = ResolveHipSymbol<HipDeviceGetMemPoolFn>(
          api_.library, "hipDeviceGetMemPool");
      api_.mem_get_mem_pool = ResolveHipSymbol<HipMemGetMemPoolFn>(
          api_.library, "hipMemGetMemPool");
      api_.mem_set_mem_pool = ResolveHipSymbol<HipMemSetMemPoolFn>(
          api_.library, "hipMemSetMemPool");
      api_.malloc_async =
          ResolveHipSymbol<HipMallocAsyncFn>(api_.library, "hipMallocAsync");
      api_.malloc_from_pool_async = ResolveHipSymbol<HipMallocFromPoolAsyncFn>(
          api_.library, "hipMallocFromPoolAsync");
      api_.free_async =
          ResolveHipSymbol<HipFreeAsyncFn>(api_.library, "hipFreeAsync");
      api_.graph_create =
          ResolveHipSymbol<HipGraphCreateFn>(api_.library, "hipGraphCreate");
      api_.graph_destroy =
          ResolveHipSymbol<HipGraphDestroyFn>(api_.library, "hipGraphDestroy");
      api_.graph_add_mem_alloc_node =
          ResolveHipSymbol<HipGraphAddMemAllocNodeFn>(
              api_.library, "hipGraphAddMemAllocNode");
#if defined(IREE_HIP_POOL_ADMISSION_TEST)
      api_.set_phase_observer = ResolveHipSymbol<SetPoolPhaseObserverFn>(
          api_.library, "iree_hip_vmm_test_set_phase_observer");
      api_.reader_depth = ResolveHipSymbol<QueryReaderDepthFn>(
          api_.library, "iree_hip_vmm_test_reader_depth");
      api_.reader_count = ResolveHipSymbol<QueryReaderCountFn>(
          api_.library, "iree_hip_vmm_test_reader_count");
      api_.run_native_callback_window =
          ResolveHipSymbol<RunNativeCallbackWindowFn>(
              api_.library, "iree_hip_vmm_test_run_native_callback_window");
#endif  // IREE_HIP_POOL_ADMISSION_TEST
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.device_get_default_mem_pool);
    ASSERT_NE(nullptr, api_.device_set_mem_pool);
    ASSERT_NE(nullptr, api_.mem_pool_create);
    ASSERT_NE(nullptr, api_.mem_pool_destroy);
    ASSERT_NE(nullptr, api_.mem_pool_set_attribute);
    ASSERT_NE(nullptr, api_.mem_pool_get_attribute);
    ASSERT_NE(nullptr, api_.mem_pool_set_access);
    ASSERT_NE(nullptr, api_.mem_pool_get_access);
    ASSERT_NE(nullptr, api_.mem_pool_trim_to);
    ASSERT_NE(nullptr, api_.mem_pool_export_to_shareable_handle);
    ASSERT_NE(nullptr, api_.mem_pool_import_from_shareable_handle);
    ASSERT_NE(nullptr, api_.mem_pool_export_pointer);
    ASSERT_NE(nullptr, api_.mem_pool_import_pointer);
    ASSERT_NE(nullptr, api_.device_get_mem_pool);
    ASSERT_NE(nullptr, api_.mem_get_mem_pool);
    ASSERT_NE(nullptr, api_.mem_set_mem_pool);
    ASSERT_NE(nullptr, api_.malloc_async);
    ASSERT_NE(nullptr, api_.malloc_from_pool_async);
    ASSERT_NE(nullptr, api_.free_async);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_destroy);
    ASSERT_NE(nullptr, api_.graph_add_mem_alloc_node);
#if defined(IREE_HIP_POOL_ADMISSION_TEST)
    ASSERT_NE(nullptr, api_.set_phase_observer);
    ASSERT_NE(nullptr, api_.reader_depth);
    ASSERT_NE(nullptr, api_.reader_count);
    ASSERT_NE(nullptr, api_.run_native_callback_window);
#endif  // IREE_HIP_POOL_ADMISSION_TEST

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    const hipError_t get_device_result = api_.get_device(&device_);
    if (get_device_result != hipSuccess) {
      GTEST_SKIP() << "hipGetDevice failed: " << get_device_result;
    }
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
      stream_ = nullptr;
    }
    // The runtime owns process-scoped driver services that cannot be
    // reinitialized after the final dlclose in the same process.
  }

  // Runtime entry points loaded once from the HIP shared object under test.
  static HipRuntimeApi api_;
  // Stream supplied to asynchronous allocation entry points.
  hipStream_t stream_ = nullptr;
  // Device ordinal associated with the test stream and pools.
  int device_ = -1;
};

HipRuntimeApi HipMemoryPoolApiTest::api_;

TEST_F(HipMemoryPoolApiTest, ZeroByteAllocationReturnsNull) {
  hipMemPool_t pool = nullptr;
  ASSERT_EQ(hipSuccess, api_.device_get_default_mem_pool(&pool, device_));

  void* pointer = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipSuccess,
            api_.malloc_from_pool_async(&pointer, /*size=*/0, pool, stream_));
  EXPECT_EQ(nullptr, pointer);
}

TEST_F(HipMemoryPoolApiTest, GraphAllocationReleasesSelectedPool) {
  hipMemPool_t default_pool = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_get_default_mem_pool(&default_pool, device_));

  hipMemPoolProps properties = {};
  properties.allocType = hipMemAllocationTypePinned;
  properties.location.type = hipMemLocationTypeDevice;
  properties.location.id = device_;

  hipMemPool_t pool = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_pool_create(&pool, &properties));

  const hipError_t set_pool_result = api_.device_set_mem_pool(device_, pool);
  EXPECT_EQ(hipSuccess, set_pool_result);
  if (set_pool_result != hipSuccess) {
    EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool));
    return;
  }

  hipGraph_t graph = nullptr;
  const hipError_t graph_create_result = api_.graph_create(&graph, /*flags=*/0);
  EXPECT_EQ(hipSuccess, graph_create_result);
  if (graph_create_result == hipSuccess) {
    hipMemAllocNodeParams parameters = {};
    parameters.poolProps = properties;
    parameters.bytesize = 4096;
    hipGraphNode_t node = nullptr;
    EXPECT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                              &node, graph, /*dependencies=*/nullptr,
                              /*dependency_count=*/0, &parameters));
    EXPECT_NE(nullptr, parameters.dptr);
    EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
    EXPECT_EQ(hipErrorInvalidValue, api_.free_async(parameters.dptr, stream_))
        << "graph destruction must release its backing allocation";
  }

  EXPECT_EQ(hipSuccess, api_.device_set_mem_pool(device_, default_pool));
  EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool));
}

#if defined(IREE_HIP_POOL_ADMISSION_TEST)
constexpr int kPoolApiAdmittedPhase = 21;

struct PoolApiRow {
  // Public API name reported when this row fails.
  const char* name;
  // Result required from the selected invocation.
  hipError_t expected_result;
  // Invocation configured for this row.
  std::function<hipError_t()> invoke;
};

struct PoolAdmissionObservation {
  // Instrumented runtime used to query lifecycle state from the callback.
  HipRuntimeApi* api;
  // Number of pool-admission notifications observed for the current row.
  size_t callback_count = 0;
  // Calling thread's reader depth at the notification boundary.
  uint32_t reader_depth = 0;
  // Process-wide outer reader count at the notification boundary.
  uint64_t reader_count = 0;
};

void ObservePoolAdmission(int phase, void* object, void* user_data) {
  (void)object;
  if (phase != kPoolApiAdmittedPhase) return;
  auto* observation = static_cast<PoolAdmissionObservation*>(user_data);
  ++observation->callback_count;
  observation->reader_depth = observation->api->reader_depth();
  observation->reader_count = observation->api->reader_count();
}

TEST_F(HipMemoryPoolApiTest, EveryPoolApiHoldsOneOuterLifecycleReader) {
  hipMemPoolProps properties = {};
  properties.allocType = hipMemAllocationTypePinned;
  properties.location.type = hipMemLocationTypeDevice;
  properties.location.id = device_;

  hipMemPool_t default_pool = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_get_default_mem_pool(&default_pool, device_));
  hipMemPool_t pool_to_destroy = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_pool_create(&pool_to_destroy, &properties));

  hipMemPool_t created_pool = nullptr;
  uint64_t release_threshold = 0;
  uint64_t queried_release_threshold = UINT64_MAX;
  hipMemLocation location = {};
  location.type = hipMemLocationTypeDevice;
  location.id = device_;
  hipMemAccessDesc access = {};
  access.location = location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  hipMemAccessFlags queried_access = hipMemAccessFlagsProtNone;
  int shareable_handle = 0;
  hipMemPool_t imported_pool = nullptr;
  hipMemPoolPtrExportData export_data = {};
  void* imported_pointer = nullptr;
  hipMemPool_t current_pool = nullptr;
  hipMemPool_t queried_default_pool = nullptr;
  hipMemPool_t selected_pool = nullptr;
  void* async_pointer = reinterpret_cast<void*>(uintptr_t{1});
  void* explicit_async_pointer = reinterpret_cast<void*>(uintptr_t{1});

  std::array<PoolApiRow, 19> rows = {{
      {"hipMemPoolCreate", hipSuccess,
       [&] { return api_.mem_pool_create(&created_pool, &properties); }},
      {"hipMemPoolDestroy", hipSuccess,
       [&] {
         hipError_t result = api_.mem_pool_destroy(pool_to_destroy);
         if (result == hipSuccess) pool_to_destroy = nullptr;
         return result;
       }},
      {"hipMemPoolSetAttribute", hipSuccess,
       [&] {
         return api_.mem_pool_set_attribute(
             default_pool, hipMemPoolAttrReleaseThreshold, &release_threshold);
       }},
      {"hipMemPoolGetAttribute", hipSuccess,
       [&] {
         return api_.mem_pool_get_attribute(default_pool,
                                            hipMemPoolAttrReleaseThreshold,
                                            &queried_release_threshold);
       }},
      {"hipMemPoolSetAccess", hipSuccess,
       [&] { return api_.mem_pool_set_access(default_pool, &access, 1); }},
      {"hipMemPoolGetAccess", hipSuccess,
       [&] {
         return api_.mem_pool_get_access(&queried_access, default_pool,
                                         &location);
       }},
      {"hipMemPoolTrimTo", hipSuccess,
       [&] { return api_.mem_pool_trim_to(default_pool, 0); }},
      {"hipMemPoolExportToShareableHandle", hipErrorNotSupported,
       [&] {
         return api_.mem_pool_export_to_shareable_handle(
             &shareable_handle, default_pool, hipMemHandleTypeNone, 0);
       }},
      {"hipMemPoolImportFromShareableHandle", hipErrorNotSupported,
       [&] {
         return api_.mem_pool_import_from_shareable_handle(
             &imported_pool, &shareable_handle, hipMemHandleTypeNone, 0);
       }},
      {"hipMemPoolExportPointer", hipErrorNotSupported,
       [&] { return api_.mem_pool_export_pointer(&export_data, nullptr); }},
      {"hipMemPoolImportPointer", hipErrorNotSupported,
       [&] {
         return api_.mem_pool_import_pointer(&imported_pointer, default_pool,
                                             &export_data);
       }},
      {"hipDeviceSetMemPool", hipSuccess,
       [&] { return api_.device_set_mem_pool(device_, default_pool); }},
      {"hipDeviceGetMemPool", hipSuccess,
       [&] { return api_.device_get_mem_pool(&current_pool, device_); }},
      {"hipDeviceGetDefaultMemPool", hipSuccess,
       [&] {
         return api_.device_get_default_mem_pool(&queried_default_pool,
                                                 device_);
       }},
      {"hipMemGetMemPool", hipSuccess,
       [&] {
         return api_.mem_get_mem_pool(&selected_pool, &location,
                                      hipMemAllocationTypePinned);
       }},
      {"hipMemSetMemPool", hipSuccess,
       [&] {
         return api_.mem_set_mem_pool(&location, hipMemAllocationTypePinned,
                                      default_pool);
       }},
      {"hipMallocAsync", hipSuccess,
       [&] { return api_.malloc_async(&async_pointer, 0, nullptr); }},
      {"hipMallocFromPoolAsync", hipSuccess,
       [&] {
         return api_.malloc_from_pool_async(&explicit_async_pointer, 0,
                                            default_pool, nullptr);
       }},
      {"hipFreeAsync", hipSuccess,
       [&] { return api_.free_async(nullptr, nullptr); }},
  }};

  PoolAdmissionObservation observation = {.api = &api_};
  ASSERT_EQ(hipSuccess,
            api_.set_phase_observer(ObservePoolAdmission, &observation));
  for (const PoolApiRow& row : rows) {
    SCOPED_TRACE(row.name);
    observation.callback_count = 0;
    observation.reader_depth = 0;
    observation.reader_count = 0;
    EXPECT_EQ(row.expected_result, row.invoke());
    EXPECT_EQ(1u, observation.callback_count);
    EXPECT_EQ(1u, observation.reader_depth);
    EXPECT_EQ(1u, observation.reader_count);
    EXPECT_EQ(0u, api_.reader_depth());
    EXPECT_EQ(0u, api_.reader_count());
  }
  EXPECT_EQ(hipSuccess, api_.set_phase_observer(nullptr, nullptr));

  if (created_pool) {
    EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(created_pool));
  }
  if (pool_to_destroy) {
    EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool_to_destroy));
  }
}

struct PoolReentryInvocation {
  // Reentry row table invoked while the callback guard is active.
  const std::array<PoolApiRow, 19>* rows;
  // Result slot paired by index with each row.
  std::array<hipError_t, 19>* results;
};

void InvokePoolApisFromNativeCallback(void* user_data) {
  auto* invocation = static_cast<PoolReentryInvocation*>(user_data);
  for (size_t i = 0; i < invocation->rows->size(); ++i) {
    (*invocation->results)[i] = (*invocation->rows)[i].invoke();
  }
}

TEST_F(HipMemoryPoolApiTest, EveryPoolApiRejectsNativeCallbackBeforeArguments) {
  hipMemPoolProps properties = {};
  hipMemLocation location = {};
  hipMemAccessDesc access = {};
  uint64_t attribute_value = 0;
  hipMemAccessFlags access_flags = hipMemAccessFlagsProtNone;
  int shareable_handle = 0;
  hipMemPoolPtrExportData export_data = {};
  hipMemPool_t pool_output = reinterpret_cast<hipMemPool_t>(uintptr_t{1});
  void* pointer_output = reinterpret_cast<void*>(uintptr_t{1});
  hipMemPool_t invalid_pool = reinterpret_cast<hipMemPool_t>(uintptr_t{1});
  hipStream_t invalid_stream = reinterpret_cast<hipStream_t>(uintptr_t{1});

  std::array<PoolApiRow, 19> rows = {{
      {"hipMemPoolCreate", hipErrorNotInitialized,
       [&] { return api_.mem_pool_create(nullptr, &properties); }},
      {"hipMemPoolDestroy", hipErrorNotInitialized,
       [&] { return api_.mem_pool_destroy(nullptr); }},
      {"hipMemPoolSetAttribute", hipErrorNotInitialized,
       [&] {
         return api_.mem_pool_set_attribute(
             nullptr, hipMemPoolAttrReleaseThreshold, &attribute_value);
       }},
      {"hipMemPoolGetAttribute", hipErrorNotInitialized,
       [&] {
         return api_.mem_pool_get_attribute(
             nullptr, hipMemPoolAttrReleaseThreshold, &attribute_value);
       }},
      {"hipMemPoolSetAccess", hipErrorNotInitialized,
       [&] { return api_.mem_pool_set_access(nullptr, &access, 1); }},
      {"hipMemPoolGetAccess", hipErrorNotInitialized,
       [&] {
         return api_.mem_pool_get_access(&access_flags, nullptr, &location);
       }},
      {"hipMemPoolTrimTo", hipErrorNotInitialized,
       [&] { return api_.mem_pool_trim_to(nullptr, 0); }},
      {"hipMemPoolExportToShareableHandle", hipErrorNotInitialized,
       [&] {
         return api_.mem_pool_export_to_shareable_handle(
             &shareable_handle, invalid_pool, hipMemHandleTypeNone, 0);
       }},
      {"hipMemPoolImportFromShareableHandle", hipErrorNotInitialized,
       [&] {
         return api_.mem_pool_import_from_shareable_handle(
             &pool_output, &shareable_handle, hipMemHandleTypeNone, 0);
       }},
      {"hipMemPoolExportPointer", hipErrorNotInitialized,
       [&] { return api_.mem_pool_export_pointer(&export_data, nullptr); }},
      {"hipMemPoolImportPointer", hipErrorNotInitialized,
       [&] {
         return api_.mem_pool_import_pointer(&pointer_output, invalid_pool,
                                             &export_data);
       }},
      {"hipDeviceSetMemPool", hipErrorNotInitialized,
       [&] { return api_.device_set_mem_pool(-1, nullptr); }},
      {"hipDeviceGetMemPool", hipErrorNotInitialized,
       [&] { return api_.device_get_mem_pool(nullptr, -1); }},
      {"hipDeviceGetDefaultMemPool", hipErrorNotInitialized,
       [&] { return api_.device_get_default_mem_pool(nullptr, -1); }},
      {"hipMemGetMemPool", hipErrorNotInitialized,
       [&] {
         return api_.mem_get_mem_pool(nullptr, &location,
                                      hipMemAllocationTypePinned);
       }},
      {"hipMemSetMemPool", hipErrorNotInitialized,
       [&] {
         return api_.mem_set_mem_pool(nullptr, hipMemAllocationTypePinned,
                                      invalid_pool);
       }},
      {"hipMallocAsync", hipErrorNotInitialized,
       [&] { return api_.malloc_async(nullptr, 0, invalid_stream); }},
      {"hipMallocFromPoolAsync", hipErrorNotInitialized,
       [&] {
         return api_.malloc_from_pool_async(nullptr, 0, invalid_pool,
                                            invalid_stream);
       }},
      {"hipFreeAsync", hipErrorNotInitialized,
       [&] { return api_.free_async(nullptr, invalid_stream); }},
  }};

  std::array<hipError_t, 19> results = {};
  PoolReentryInvocation invocation = {
      .rows = &rows,
      .results = &results,
  };
  api_.run_native_callback_window(InvokePoolApisFromNativeCallback,
                                  &invocation);
  for (size_t i = 0; i < rows.size(); ++i) {
    SCOPED_TRACE(rows[i].name);
    EXPECT_EQ(rows[i].expected_result, results[i]);
  }
  EXPECT_EQ(0u, api_.reader_depth());
  EXPECT_EQ(0u, api_.reader_count());
}
#endif  // IREE_HIP_POOL_ADMISSION_TEST

}  // namespace
