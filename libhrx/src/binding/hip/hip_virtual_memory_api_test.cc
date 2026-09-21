// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

const char* CandidateLibPath() {
  if (const char* path = std::getenv("HRX_TEST_LIBAMDHIP64");
      path && *path != '\0') {
    return path;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

using HipInitFn = hipError_t (*)(unsigned int);
using HipHALDeinitFn = hipError_t (*)(void);
using HipGetProcAddressFn = hipError_t (*)(const char*, void**, int, uint64_t,
                                           void*);
using HipGetDeviceFn = hipError_t (*)(int*);
using HipSetDeviceFn = hipError_t (*)(int);
using HipGetDeviceCountFn = hipError_t (*)(int*);
using HipGetDevicePropertiesFn = hipError_t (*)(hipDeviceProp_t*, int);
using HipDeviceGetAttributeFn = hipError_t (*)(int*, hipDeviceAttribute_t, int);
using HipDeviceResetFn = hipError_t (*)(void);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipDevicePrimaryCtxRetainFn = hipError_t (*)(hipCtx_t*, hipDevice_t);
using HipDevicePrimaryCtxReleaseFn = hipError_t (*)(hipDevice_t);
using HipDevicePrimaryCtxResetFn = hipError_t (*)(hipDevice_t);
using HipCtxCreateFn = hipError_t (*)(hipCtx_t*, unsigned int, hipDevice_t);
using HipCtxDestroyFn = hipError_t (*)(hipCtx_t);
using HipCtxSetCurrentFn = hipError_t (*)(hipCtx_t);
using HipCtxGetDeviceFn = hipError_t (*)(hipDevice_t*);
using HipMemGetAllocationGranularityFn = hipError_t (*)(
    size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags);
using HipMemAddressReserveFn = hipError_t (*)(void**, size_t, size_t, void*,
                                              unsigned long long);
using HipMemAddressFreeFn = hipError_t (*)(void*, size_t);
using HipMemCreateFn = hipError_t (*)(hipMemGenericAllocationHandle_t*, size_t,
                                      const hipMemAllocationProp*,
                                      unsigned long long);
using HipMemReleaseFn = hipError_t (*)(hipMemGenericAllocationHandle_t);
using HipMemMapFn = hipError_t (*)(void*, size_t, size_t,
                                   hipMemGenericAllocationHandle_t,
                                   unsigned long long);
using HipMemUnmapFn = hipError_t (*)(void*, size_t);
using HipMemSetAccessFn = hipError_t (*)(void*, size_t, const hipMemAccessDesc*,
                                         size_t);
using HipMemGetAccessFn = hipError_t (*)(unsigned long long*,
                                         const hipMemLocation*, void*);
using HipMemGetAddressRangeFn = hipError_t (*)(hipDeviceptr_t*, size_t*,
                                               hipDeviceptr_t);
using HipMemPtrGetInfoFn = hipError_t (*)(void*, size_t*);
using HipMemRetainAllocationHandleFn =
    hipError_t (*)(hipMemGenericAllocationHandle_t*, void*);
using HipMemGetAllocationPropertiesFn =
    hipError_t (*)(hipMemAllocationProp*, hipMemGenericAllocationHandle_t);
using HipPointerGetAttributeFn = hipError_t (*)(void*, hipPointer_attribute_t,
                                                hipDeviceptr_t);
using HipDrvPointerGetAttributesFn = hipError_t (*)(unsigned int,
                                                    hipPointer_attribute_t*,
                                                    void**, const void*);
using HipPointerGetAttributesFn = hipError_t (*)(hipPointerAttribute_t*,
                                                 const void*);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t*, const void*);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t*, hipModule_t,
                                              const char*);
using HipModuleLaunchKernelFn = hipError_t (*)(hipFunction_t, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               hipStream_t, void**, void**);
using HipMemcpyFn = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind);
using HipMemsetFn = hipError_t (*)(void*, int, size_t);
using HipMemsetAsyncFn = hipError_t (*)(void*, int, size_t, hipStream_t);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t);
using HipMallocFn = hipError_t (*)(void**, size_t);
using HipFreeFn = hipError_t (*)(void*);
using HipMemGetInfoFn = hipError_t (*)(size_t*, size_t*);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t*, unsigned int);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t);
using HipGraphDestroyNodeFn = hipError_t (*)(hipGraphNode_t);
using HipGraphAddKernelNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                               const hipGraphNode_t*, size_t,
                                               const void*);
using HipGraphAddChildGraphNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                                   const hipGraphNode_t*,
                                                   size_t, hipGraph_t);
using HipGraphAddMemcpyNode1DFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                                 const hipGraphNode_t*, size_t,
                                                 void*, const void*, size_t,
                                                 hipMemcpyKind);
using HipGraphAddHostNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                             const hipGraphNode_t*, size_t,
                                             const void*);
using HipGraphAddEmptyNodeFn = hipError_t (*)(hipGraphNode_t*, hipGraph_t,
                                              const hipGraphNode_t*, size_t);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t*, hipGraph_t,
                                             hipGraphNode_t*, char*, size_t);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t, hipStream_t);
using HipGraphExecUpdateFn = hipError_t (*)(hipGraphExec_t, hipGraph_t,
                                            hipGraphNode_t*,
                                            hipGraphExecUpdateResult*);
using HipGraphExecHostNodeSetParamsFn = hipError_t (*)(hipGraphExec_t,
                                                       hipGraphNode_t,
                                                       const void*);
using HipGraphExecChildGraphNodeSetParamsFn = hipError_t (*)(hipGraphExec_t,
                                                             hipGraphNode_t,
                                                             hipGraph_t);
using HipGraphNodeGetEnabledFn = hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                                unsigned int*);
using HipGraphNodeSetEnabledFn = hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                                unsigned int);

void IncrementAtomicCounter(void* user_data) {
  static_cast<std::atomic<int>*>(user_data)->fetch_add(
      1, std::memory_order_release);
}

template <typename T>
class ExactSizeOutput {
 public:
  ExactSizeOutput() { storage_.fill(kCanary); }

  void* output() { return storage_.data() + kGuardSize; }

  T value() const {
    T value = {};
    std::memcpy(&value, storage_.data() + kGuardSize, sizeof(value));
    return value;
  }

  void ExpectGuardsIntact() const {
    for (size_t i = 0; i < kGuardSize; ++i) {
      EXPECT_EQ(kCanary, storage_[i]);
    }
    for (size_t i = kGuardSize + sizeof(T); i < storage_.size(); ++i) {
      EXPECT_EQ(kCanary, storage_[i]);
    }
  }

  void ExpectUntouched() const {
    for (uint8_t byte : storage_) EXPECT_EQ(kCanary, byte);
  }

 private:
  static constexpr uint8_t kCanary = 0xA5;
  static constexpr size_t kGuardSize = alignof(std::max_align_t);
  alignas(std::max_align_t)
      std::array<uint8_t, kGuardSize + sizeof(T) + kGuardSize> storage_;
};

template <typename T>
T QueryExactPointerAttribute(HipPointerGetAttributeFn query,
                             hipPointer_attribute_t attribute,
                             hipDeviceptr_t pointer) {
  ExactSizeOutput<T> output;
  EXPECT_EQ(hipSuccess, query(output.output(), attribute, pointer));
  output.ExpectGuardsIntact();
  return output.value();
}

struct HipApi {
  void* library = nullptr;
  HipInitFn init = nullptr;
  HipHALDeinitFn hal_deinit = nullptr;
  HipGetProcAddressFn get_proc_address = nullptr;
  HipGetDeviceFn get_device = nullptr;
  HipSetDeviceFn set_device = nullptr;
  HipGetDeviceCountFn get_device_count = nullptr;
  HipGetDevicePropertiesFn get_device_properties = nullptr;
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  HipDeviceResetFn device_reset = nullptr;
  HipDeviceSynchronizeFn device_synchronize = nullptr;
  HipDevicePrimaryCtxRetainFn device_primary_ctx_retain = nullptr;
  HipDevicePrimaryCtxReleaseFn device_primary_ctx_release = nullptr;
  HipDevicePrimaryCtxResetFn device_primary_ctx_reset = nullptr;
  HipCtxCreateFn ctx_create = nullptr;
  HipCtxDestroyFn ctx_destroy = nullptr;
  HipCtxSetCurrentFn ctx_set_current = nullptr;
  HipCtxGetDeviceFn ctx_get_device = nullptr;
  HipMemGetAllocationGranularityFn mem_get_allocation_granularity = nullptr;
  HipMemAddressReserveFn mem_address_reserve = nullptr;
  HipMemAddressFreeFn mem_address_free = nullptr;
  HipMemCreateFn mem_create = nullptr;
  HipMemReleaseFn mem_release = nullptr;
  HipMemMapFn mem_map = nullptr;
  HipMemUnmapFn mem_unmap = nullptr;
  HipMemSetAccessFn mem_set_access = nullptr;
  HipMemGetAccessFn mem_get_access = nullptr;
  HipMemGetAddressRangeFn mem_get_address_range = nullptr;
  HipMemPtrGetInfoFn mem_ptr_get_info = nullptr;
  HipMemRetainAllocationHandleFn mem_retain_allocation_handle = nullptr;
  HipMemGetAllocationPropertiesFn mem_get_allocation_properties = nullptr;
  HipPointerGetAttributeFn pointer_get_attribute = nullptr;
  HipDrvPointerGetAttributesFn drv_pointer_get_attributes = nullptr;
  HipPointerGetAttributesFn pointer_get_attributes = nullptr;
  HipGetLastErrorFn get_last_error = nullptr;
  HipPeekAtLastErrorFn peek_at_last_error = nullptr;
  HipModuleLoadDataFn module_load_data = nullptr;
  HipModuleUnloadFn module_unload = nullptr;
  HipModuleGetFunctionFn module_get_function = nullptr;
  HipModuleLaunchKernelFn module_launch_kernel = nullptr;
  HipMemcpyFn memcpy = nullptr;
  HipMemsetFn memset = nullptr;
  HipMemsetAsyncFn memset_async = nullptr;
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  HipMallocFn malloc = nullptr;
  HipFreeFn free = nullptr;
  HipMemGetInfoFn mem_get_info = nullptr;
  HipGraphCreateFn graph_create = nullptr;
  HipGraphDestroyFn graph_destroy = nullptr;
  HipGraphDestroyNodeFn graph_destroy_node = nullptr;
  HipGraphAddKernelNodeFn graph_add_kernel_node = nullptr;
  HipGraphAddChildGraphNodeFn graph_add_child_graph_node = nullptr;
  HipGraphAddMemcpyNode1DFn graph_add_memcpy_node_1d = nullptr;
  HipGraphAddHostNodeFn graph_add_host_node = nullptr;
  HipGraphAddEmptyNodeFn graph_add_empty_node = nullptr;
  HipGraphInstantiateFn graph_instantiate = nullptr;
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  HipGraphLaunchFn graph_launch = nullptr;
  HipGraphExecUpdateFn graph_exec_update = nullptr;
  HipGraphExecHostNodeSetParamsFn graph_exec_host_node_set_params = nullptr;
  HipGraphExecChildGraphNodeSetParamsFn graph_exec_child_graph_node_set_params =
      nullptr;
  HipGraphNodeGetEnabledFn graph_node_get_enabled = nullptr;
  HipGraphNodeSetEnabledFn graph_node_set_enabled = nullptr;
};

class HipVirtualMemoryApiTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const char* path = CandidateLibPath();
    ASSERT_NE(path, nullptr)
        << "the build must provide the exact libamdhip64 artifact under test";
    api_.library = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    ASSERT_NE(api_.library, nullptr)
        << "cannot dlopen " << path << ": " << dlerror();
#define RESOLVE(name, field) \
  api_.field = ResolveHipSymbol<decltype(api_.field)>(api_.library, name)
    RESOLVE("hipInit", init);
    RESOLVE("hipHALDeinit", hal_deinit);
    RESOLVE("hipGetProcAddress", get_proc_address);
    RESOLVE("hipGetDevice", get_device);
    RESOLVE("hipSetDevice", set_device);
    RESOLVE("hipGetDeviceCount", get_device_count);
    RESOLVE("hipGetDeviceProperties", get_device_properties);
    RESOLVE("hipDeviceGetAttribute", device_get_attribute);
    RESOLVE("hipDeviceReset", device_reset);
    RESOLVE("hipDeviceSynchronize", device_synchronize);
    RESOLVE("hipDevicePrimaryCtxRetain", device_primary_ctx_retain);
    RESOLVE("hipDevicePrimaryCtxRelease", device_primary_ctx_release);
    RESOLVE("hipDevicePrimaryCtxReset", device_primary_ctx_reset);
    RESOLVE("hipCtxCreate", ctx_create);
    RESOLVE("hipCtxDestroy", ctx_destroy);
    RESOLVE("hipCtxSetCurrent", ctx_set_current);
    RESOLVE("hipCtxGetDevice", ctx_get_device);
    RESOLVE("hipMemGetAllocationGranularity", mem_get_allocation_granularity);
    RESOLVE("hipMemAddressReserve", mem_address_reserve);
    RESOLVE("hipMemAddressFree", mem_address_free);
    RESOLVE("hipMemCreate", mem_create);
    RESOLVE("hipMemRelease", mem_release);
    RESOLVE("hipMemMap", mem_map);
    RESOLVE("hipMemUnmap", mem_unmap);
    RESOLVE("hipMemSetAccess", mem_set_access);
    RESOLVE("hipMemGetAccess", mem_get_access);
    RESOLVE("hipMemGetAddressRange", mem_get_address_range);
    RESOLVE("hipMemPtrGetInfo", mem_ptr_get_info);
    RESOLVE("hipMemRetainAllocationHandle", mem_retain_allocation_handle);
    RESOLVE("hipMemGetAllocationPropertiesFromHandle",
            mem_get_allocation_properties);
    RESOLVE("hipPointerGetAttribute", pointer_get_attribute);
    RESOLVE("hipDrvPointerGetAttributes", drv_pointer_get_attributes);
    RESOLVE("hipPointerGetAttributes", pointer_get_attributes);
    RESOLVE("hipGetLastError", get_last_error);
    RESOLVE("hipPeekAtLastError", peek_at_last_error);
    RESOLVE("hipModuleLoadData", module_load_data);
    RESOLVE("hipModuleUnload", module_unload);
    RESOLVE("hipModuleGetFunction", module_get_function);
    RESOLVE("hipModuleLaunchKernel", module_launch_kernel);
    RESOLVE("hipMemcpy", memcpy);
    RESOLVE("hipMemset", memset);
    RESOLVE("hipMemsetAsync", memset_async);
    RESOLVE("hipStreamSynchronize", stream_synchronize);
    RESOLVE("hipMalloc", malloc);
    RESOLVE("hipFree", free);
    RESOLVE("hipMemGetInfo", mem_get_info);
    RESOLVE("hipGraphCreate", graph_create);
    RESOLVE("hipGraphDestroy", graph_destroy);
    RESOLVE("hipGraphDestroyNode", graph_destroy_node);
    RESOLVE("hipGraphAddKernelNode", graph_add_kernel_node);
    RESOLVE("hipGraphAddChildGraphNode", graph_add_child_graph_node);
    RESOLVE("hipGraphAddMemcpyNode1D", graph_add_memcpy_node_1d);
    RESOLVE("hipGraphAddHostNode", graph_add_host_node);
    RESOLVE("hipGraphAddEmptyNode", graph_add_empty_node);
    RESOLVE("hipGraphInstantiate", graph_instantiate);
    RESOLVE("hipGraphExecDestroy", graph_exec_destroy);
    RESOLVE("hipGraphLaunch", graph_launch);
    RESOLVE("hipGraphExecUpdate", graph_exec_update);
    RESOLVE("hipGraphExecHostNodeSetParams", graph_exec_host_node_set_params);
    RESOLVE("hipGraphExecChildGraphNodeSetParams",
            graph_exec_child_graph_node_set_params);
    RESOLVE("hipGraphNodeGetEnabled", graph_node_get_enabled);
    RESOLVE("hipGraphNodeSetEnabled", graph_node_set_enabled);
#undef RESOLVE
    ASSERT_NE(api_.init, nullptr);
    ASSERT_NE(api_.hal_deinit, nullptr);
    ASSERT_NE(api_.get_proc_address, nullptr);
    ASSERT_NE(api_.get_device, nullptr);
    ASSERT_NE(api_.set_device, nullptr);
    ASSERT_NE(api_.get_device_count, nullptr);
    ASSERT_NE(api_.get_device_properties, nullptr);
    ASSERT_NE(api_.device_get_attribute, nullptr);
    ASSERT_NE(api_.device_reset, nullptr);
    ASSERT_NE(api_.device_synchronize, nullptr);
    ASSERT_NE(api_.device_primary_ctx_retain, nullptr);
    ASSERT_NE(api_.device_primary_ctx_release, nullptr);
    ASSERT_NE(api_.device_primary_ctx_reset, nullptr);
    ASSERT_NE(api_.ctx_create, nullptr);
    ASSERT_NE(api_.ctx_destroy, nullptr);
    ASSERT_NE(api_.ctx_set_current, nullptr);
    ASSERT_NE(api_.ctx_get_device, nullptr);
    ASSERT_NE(api_.mem_get_allocation_granularity, nullptr);
    ASSERT_NE(api_.mem_address_reserve, nullptr);
    ASSERT_NE(api_.mem_address_free, nullptr);
    ASSERT_NE(api_.mem_create, nullptr);
    ASSERT_NE(api_.mem_release, nullptr);
    ASSERT_NE(api_.mem_map, nullptr);
    ASSERT_NE(api_.mem_unmap, nullptr);
    ASSERT_NE(api_.mem_set_access, nullptr);
    ASSERT_NE(api_.mem_get_access, nullptr);
    ASSERT_NE(api_.mem_get_address_range, nullptr);
    ASSERT_NE(api_.mem_ptr_get_info, nullptr);
    ASSERT_NE(api_.mem_retain_allocation_handle, nullptr);
    ASSERT_NE(api_.mem_get_allocation_properties, nullptr);
    ASSERT_NE(api_.pointer_get_attribute, nullptr);
    ASSERT_NE(api_.drv_pointer_get_attributes, nullptr);
    ASSERT_NE(api_.pointer_get_attributes, nullptr);
    ASSERT_NE(api_.get_last_error, nullptr);
    ASSERT_NE(api_.peek_at_last_error, nullptr);
    ASSERT_NE(api_.module_load_data, nullptr);
    ASSERT_NE(api_.module_unload, nullptr);
    ASSERT_NE(api_.module_get_function, nullptr);
    ASSERT_NE(api_.module_launch_kernel, nullptr);
    ASSERT_NE(api_.memcpy, nullptr);
    ASSERT_NE(api_.memset, nullptr);
    ASSERT_NE(api_.memset_async, nullptr);
    ASSERT_NE(api_.stream_synchronize, nullptr);
    ASSERT_NE(api_.malloc, nullptr);
    ASSERT_NE(api_.free, nullptr);
    ASSERT_NE(api_.mem_get_info, nullptr);
    ASSERT_NE(api_.graph_create, nullptr);
    ASSERT_NE(api_.graph_destroy, nullptr);
    ASSERT_NE(api_.graph_destroy_node, nullptr);
    ASSERT_NE(api_.graph_add_kernel_node, nullptr);
    ASSERT_NE(api_.graph_add_child_graph_node, nullptr);
    ASSERT_NE(api_.graph_add_memcpy_node_1d, nullptr);
    ASSERT_NE(api_.graph_add_host_node, nullptr);
    ASSERT_NE(api_.graph_add_empty_node, nullptr);
    ASSERT_NE(api_.graph_instantiate, nullptr);
    ASSERT_NE(api_.graph_exec_destroy, nullptr);
    ASSERT_NE(api_.graph_launch, nullptr);
    ASSERT_NE(api_.graph_exec_update, nullptr);
    ASSERT_NE(api_.graph_exec_host_node_set_params, nullptr);
    ASSERT_NE(api_.graph_exec_child_graph_node_set_params, nullptr);
    ASSERT_NE(api_.graph_node_get_enabled, nullptr);
    ASSERT_NE(api_.graph_node_set_enabled, nullptr);
  }

  hipError_t QueryVmmSupport(int device, bool* out_supported) {
    hipError_t result = api_.set_device(device);
    if (result != hipSuccess) return result;
    int supported = 0;
    result = api_.device_get_attribute(
        &supported, hipDeviceAttributeVirtualMemoryManagementSupported, device);
    if (result == hipSuccess) *out_supported = supported != 0;
    return result;
  }

  hipError_t QueryMinimumGranularity(int device, size_t* out_granularity) {
    hipMemAllocationProp properties = {};
    properties.type = hipMemAllocationTypePinned;
    properties.requestedHandleType = hipMemHandleTypeNone;
    properties.location.type = hipMemLocationTypeDevice;
    properties.location.id = device;
    size_t granularity = 0;
    const hipError_t result = api_.mem_get_allocation_granularity(
        &granularity, &properties, hipMemAllocationGranularityMinimum);
    if (result == hipSuccess) *out_granularity = granularity;
    return result;
  }

  hipMemAllocationProp DeviceProperties(int device) {
    hipMemAllocationProp properties = {};
    properties.type = hipMemAllocationTypePinned;
    properties.requestedHandleType = hipMemHandleTypeNone;
    properties.location.type = hipMemLocationTypeDevice;
    properties.location.id = device;
    return properties;
  }

  void ExpectAccessEveryByte(void* base, size_t size,
                             const hipMemLocation& location,
                             hipMemAccessFlags expected) {
    for (size_t offset = 0; offset < size; ++offset) {
      unsigned long long access = ~0ull;
      ASSERT_EQ(hipSuccess,
                api_.mem_get_access(&access, &location,
                                    static_cast<uint8_t*>(base) + offset))
          << "offset " << offset;
      EXPECT_EQ(static_cast<unsigned long long>(expected), access)
          << "offset " << offset;
    }
  }

  struct StoreKernelGraph {
    hipModule_t module = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphNode_t node = nullptr;
    hipGraphExec_t exec = nullptr;
  };

  hipError_t CreateStoreKernelGraph(const void* image, void* output,
                                    uint32_t value,
                                    StoreKernelGraph* out_graph) {
    hipError_t result = api_.module_load_data(&out_graph->module, image);
    if (result != hipSuccess) return result;
    hipFunction_t function = nullptr;
    result = api_.module_get_function(&function, out_graph->module,
                                      "hrx_store_output");
    if (result != hipSuccess) return result;
    hipDeviceptr_t device_output = static_cast<hipDeviceptr_t>(output);
    void* arguments[] = {&device_output, &value};
    hipKernelNodeParams node_params = {};
    node_params.blockDim = {1, 1, 1};
    node_params.func = function;
    node_params.gridDim = {1, 1, 1};
    node_params.kernelParams = arguments;
    result = api_.graph_create(&out_graph->graph, 0);
    if (result != hipSuccess) return result;
    result = api_.graph_add_kernel_node(&out_graph->node, out_graph->graph,
                                        nullptr, 0, &node_params);
    if (result != hipSuccess) return result;
    return api_.graph_instantiate(&out_graph->exec, out_graph->graph, nullptr,
                                  nullptr, 0);
  }

  static HipApi api_;
};

HipApi HipVirtualMemoryApiTest::api_;

TEST_F(HipVirtualMemoryApiTest, LoadsExactProductionDso) {
  Dl_info symbol_info = {};
  ASSERT_NE(0, dladdr(reinterpret_cast<void*>(api_.mem_map), &symbol_info));
  ASSERT_NE(symbol_info.dli_fname, nullptr);

  std::array<char, PATH_MAX> expected_path = {};
  std::array<char, PATH_MAX> loaded_path = {};
  ASSERT_NE(nullptr, realpath(CandidateLibPath(), expected_path.data()));
  ASSERT_NE(nullptr, realpath(symbol_info.dli_fname, loaded_path.data()));
  EXPECT_STREQ(expected_path.data(), loaded_path.data());

  void* resolved = nullptr;
  int symbol_status = -1;
  ASSERT_EQ(hipSuccess, api_.get_proc_address("hipMemSetAccess", &resolved,
                                              60000000, 0, &symbol_status));
  EXPECT_EQ(reinterpret_cast<void*>(api_.mem_set_access), resolved);
  EXPECT_EQ(0, symbol_status);
}

TEST_F(HipVirtualMemoryApiTest,
       AddressReserveAcceptsAlignedNonNullHintWithNativeFallback) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(device, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(device, &granularity));
  ASSERT_GT(granularity, 0u);
  ASSERT_EQ(0u, granularity & (granularity - 1));

  void* occupied = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&occupied, granularity, granularity,
                                     nullptr, /*flags=*/0));
  ASSERT_NE(nullptr, occupied);
  ASSERT_EQ(0u, reinterpret_cast<uintptr_t>(occupied) % granularity);

  void* fallback = nullptr;
  const hipError_t reserve_result = api_.mem_address_reserve(
      &fallback, granularity, granularity, occupied, /*flags=*/0);
  EXPECT_EQ(hipSuccess, reserve_result);
  if (reserve_result == hipSuccess) {
    EXPECT_NE(nullptr, fallback);
    EXPECT_NE(occupied, fallback);
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(fallback) % granularity);
    EXPECT_EQ(hipSuccess, api_.mem_address_free(fallback, granularity));
  }
  EXPECT_EQ(hipSuccess, api_.mem_address_free(occupied, granularity));
}

TEST_F(HipVirtualMemoryApiTest,
       PointerAttributesUseExactPublicTypesForOrdinaryAndVmmPointers) {
  static_assert(sizeof(hipMemoryType) == sizeof(uint32_t));
  static_assert(sizeof(bool) == 1);

  ASSERT_EQ(hipSuccess, api_.init(0));
  ASSERT_EQ(hipSuccess, api_.set_device(0));

  void* ordinary = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&ordinary, 64));
  const hipDeviceptr_t ordinary_pointer = static_cast<hipDeviceptr_t>(ordinary);
  hipCtx_t ordinary_context = QueryExactPointerAttribute<hipCtx_t>(
      api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_CONTEXT,
      ordinary_pointer);
  EXPECT_NE(nullptr, ordinary_context);
  EXPECT_EQ(static_cast<uint32_t>(hipMemoryTypeDevice),
            QueryExactPointerAttribute<uint32_t>(
                api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                ordinary_pointer));
  EXPECT_EQ(ordinary_pointer,
            QueryExactPointerAttribute<hipDeviceptr_t>(
                api_.pointer_get_attribute,
                HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ordinary_pointer));
  EXPECT_EQ(0, QueryExactPointerAttribute<int>(
                   api_.pointer_get_attribute,
                   HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ordinary_pointer));
  EXPECT_FALSE(QueryExactPointerAttribute<bool>(
      api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
      ordinary_pointer));
  EXPECT_EQ(ordinary_pointer,
            QueryExactPointerAttribute<hipDeviceptr_t>(
                api_.pointer_get_attribute,
                HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR, ordinary_pointer));
  EXPECT_EQ(64u, QueryExactPointerAttribute<uint32_t>(
                     api_.pointer_get_attribute,
                     HIP_POINTER_ATTRIBUTE_RANGE_SIZE, ordinary_pointer));
  EXPECT_TRUE(QueryExactPointerAttribute<bool>(api_.pointer_get_attribute,
                                               HIP_POINTER_ATTRIBUTE_MAPPED,
                                               ordinary_pointer));
  EXPECT_TRUE(QueryExactPointerAttribute<bool>(
      api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
      ordinary_pointer));
  EXPECT_NE(0u, QueryExactPointerAttribute<uint32_t>(
                    api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                    ordinary_pointer));
  ExactSizeOutput<void*> ordinary_host_pointer;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.pointer_get_attribute(ordinary_host_pointer.output(),
                                       HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                       ordinary_pointer));
  ordinary_host_pointer.ExpectUntouched();

  ExactSizeOutput<hipCtx_t> bulk_context;
  ExactSizeOutput<uint32_t> bulk_memory_type;
  ExactSizeOutput<hipDeviceptr_t> bulk_device_pointer;
  ExactSizeOutput<int> bulk_device_ordinal;
  ExactSizeOutput<bool> bulk_is_managed;
  ExactSizeOutput<hipDeviceptr_t> bulk_range_start;
  ExactSizeOutput<uint32_t> bulk_range_size;
  ExactSizeOutput<bool> bulk_mapped;
  ExactSizeOutput<bool> bulk_sync_memops;
  ExactSizeOutput<uint32_t> bulk_buffer_id;
  std::array<hipPointer_attribute_t, 10> bulk_attributes = {
      HIP_POINTER_ATTRIBUTE_CONTEXT,
      HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
      HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
      HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
      HIP_POINTER_ATTRIBUTE_IS_MANAGED,
      HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
      HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
      HIP_POINTER_ATTRIBUTE_MAPPED,
      HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
      HIP_POINTER_ATTRIBUTE_BUFFER_ID,
  };
  std::array<void*, 10> bulk_outputs = {
      bulk_context.output(),        bulk_memory_type.output(),
      bulk_device_pointer.output(), bulk_device_ordinal.output(),
      bulk_is_managed.output(),     bulk_range_start.output(),
      bulk_range_size.output(),     bulk_mapped.output(),
      bulk_sync_memops.output(),    bulk_buffer_id.output(),
  };
  ASSERT_EQ(hipSuccess, api_.drv_pointer_get_attributes(
                            bulk_attributes.size(), bulk_attributes.data(),
                            bulk_outputs.data(), ordinary));
  bulk_context.ExpectGuardsIntact();
  bulk_memory_type.ExpectGuardsIntact();
  bulk_device_pointer.ExpectGuardsIntact();
  bulk_device_ordinal.ExpectGuardsIntact();
  bulk_is_managed.ExpectGuardsIntact();
  bulk_range_start.ExpectGuardsIntact();
  bulk_range_size.ExpectGuardsIntact();
  bulk_mapped.ExpectGuardsIntact();
  bulk_sync_memops.ExpectGuardsIntact();
  bulk_buffer_id.ExpectGuardsIntact();
  EXPECT_EQ(ordinary_context, bulk_context.value());
  EXPECT_EQ(static_cast<uint32_t>(hipMemoryTypeDevice),
            bulk_memory_type.value());
  EXPECT_EQ(ordinary_pointer, bulk_device_pointer.value());
  EXPECT_EQ(0, bulk_device_ordinal.value());
  EXPECT_FALSE(bulk_is_managed.value());
  EXPECT_EQ(ordinary_pointer, bulk_range_start.value());
  EXPECT_EQ(64u, bulk_range_size.value());
  EXPECT_TRUE(bulk_mapped.value());
  EXPECT_TRUE(bulk_sync_memops.value());
  EXPECT_NE(0u, bulk_buffer_id.value());
  ASSERT_EQ(hipSuccess, api_.free(ordinary));

  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));
  const hipDeviceptr_t vmm_pointer = static_cast<hipDeviceptr_t>(reservation);
  EXPECT_NE(nullptr, QueryExactPointerAttribute<hipCtx_t>(
                         api_.pointer_get_attribute,
                         HIP_POINTER_ATTRIBUTE_CONTEXT, vmm_pointer));
  EXPECT_EQ(static_cast<uint32_t>(hipMemoryTypeDevice),
            QueryExactPointerAttribute<uint32_t>(
                api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                vmm_pointer));
  EXPECT_EQ(0, QueryExactPointerAttribute<int>(
                   api_.pointer_get_attribute,
                   HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, vmm_pointer));
  EXPECT_FALSE(QueryExactPointerAttribute<bool>(
      api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
      vmm_pointer));
  EXPECT_EQ(vmm_pointer,
            QueryExactPointerAttribute<hipDeviceptr_t>(
                api_.pointer_get_attribute,
                HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR, vmm_pointer));
  EXPECT_EQ(granularity, QueryExactPointerAttribute<uint32_t>(
                             api_.pointer_get_attribute,
                             HIP_POINTER_ATTRIBUTE_RANGE_SIZE, vmm_pointer));
  EXPECT_TRUE(QueryExactPointerAttribute<bool>(
      api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_MAPPED, vmm_pointer));
  EXPECT_TRUE(QueryExactPointerAttribute<bool>(
      api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
      vmm_pointer));
  EXPECT_NE(0u, QueryExactPointerAttribute<uint32_t>(
                    api_.pointer_get_attribute, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                    vmm_pointer));
  ExactSizeOutput<void*> vmm_host_pointer;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.pointer_get_attribute(vmm_host_pointer.output(),
                                       HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                       vmm_pointer));
  vmm_host_pointer.ExpectUntouched();
  ExactSizeOutput<hipDeviceptr_t> inaccessible_device_pointer;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.pointer_get_attribute(inaccessible_device_pointer.output(),
                                       HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                       vmm_pointer));
  inaccessible_device_pointer.ExpectUntouched();

  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, 0};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  EXPECT_EQ(vmm_pointer,
            QueryExactPointerAttribute<hipDeviceptr_t>(
                api_.pointer_get_attribute,
                HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, vmm_pointer));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
}

TEST_F(HipVirtualMemoryApiTest,
       BulkPointerAttributesIgnoreFieldErrorsWithoutChangingLastError) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  ASSERT_EQ(hipSuccess, api_.set_device(0));
  (void)api_.get_last_error();
  ASSERT_EQ(hipSuccess, api_.peek_at_last_error());

  void* ordinary = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&ordinary, 64));
  (void)api_.get_last_error();

  hipPointer_attribute_t memory_type_attribute =
      HIP_POINTER_ATTRIBUTE_MEMORY_TYPE;
  ExactSizeOutput<hipMemoryType> null_pointer_output;
  void* null_pointer_data[] = {null_pointer_output.output()};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.drv_pointer_get_attributes(1, &memory_type_attribute,
                                            null_pointer_data, nullptr));
  null_pointer_output.ExpectUntouched();
  EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());

  ExactSizeOutput<uint64_t> unsupported_single_output;
  EXPECT_EQ(hipErrorNotSupported,
            api_.pointer_get_attribute(unsupported_single_output.output(),
                                       HIP_POINTER_ATTRIBUTE_P2P_TOKENS,
                                       static_cast<hipDeviceptr_t>(ordinary)));
  unsupported_single_output.ExpectUntouched();
  EXPECT_EQ(hipErrorNotSupported, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorNotSupported, api_.get_last_error());
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());

  ExactSizeOutput<hipMemoryType> supported_bulk_output;
  ExactSizeOutput<uint64_t> unsupported_bulk_output;
  std::array<hipPointer_attribute_t, 2> attributes = {
      HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
      HIP_POINTER_ATTRIBUTE_P2P_TOKENS,
  };
  std::array<void*, 2> outputs = {
      supported_bulk_output.output(),
      unsupported_bulk_output.output(),
  };
  EXPECT_EQ(hipSuccess, api_.drv_pointer_get_attributes(
                            attributes.size(), attributes.data(),
                            outputs.data(), ordinary));
  supported_bulk_output.ExpectGuardsIntact();
  EXPECT_EQ(hipMemoryTypeDevice, supported_bulk_output.value());
  unsupported_bulk_output.ExpectUntouched();
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());

  EXPECT_EQ(hipSuccess, api_.free(ordinary));
}

TEST_F(HipVirtualMemoryApiTest,
       PointerRangeSizeOverflowDoesNotPartiallyWritePublicOutput) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const size_t max_u32 = std::numeric_limits<uint32_t>::max();
  const size_t oversized_range = ((max_u32 / granularity) + 1) * granularity;
  ASSERT_GT(oversized_range, max_u32);

  size_t free_memory = 0;
  size_t total_memory = 0;
  ASSERT_EQ(hipSuccess, api_.mem_get_info(&free_memory, &total_memory));
  if (free_memory < oversized_range + granularity) {
    GTEST_SKIP() << "requires a mapped VMM extent larger than UINT32_MAX";
  }

  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&reservation, oversized_range,
                                                 0, nullptr, 0));
  const hipMemAllocationProp properties = DeviceProperties(0);
  hipMemGenericAllocationHandle_t handle = nullptr;
  const hipError_t create_result =
      api_.mem_create(&handle, oversized_range, &properties, 0);
  if (create_result != hipSuccess) {
    EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, oversized_range));
    GTEST_SKIP() << "backend cannot create the oversized VMM handle";
  }
  const hipError_t map_result =
      api_.mem_map(reservation, oversized_range, 0, handle, 0);
  if (map_result != hipSuccess) {
    EXPECT_EQ(hipSuccess, api_.mem_release(handle));
    EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, oversized_range));
    GTEST_SKIP() << "backend cannot map the oversized VMM extent";
  }

  ExactSizeOutput<uint32_t> range_size;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.pointer_get_attribute(
                range_size.output(), HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                static_cast<hipDeviceptr_t>(reservation)));
  range_size.ExpectUntouched();

  EXPECT_EQ(hipSuccess, api_.mem_unmap(reservation, oversized_range));
  EXPECT_EQ(hipSuccess, api_.mem_release(handle));
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, oversized_range));
}

TEST_F(HipVirtualMemoryApiTest, MappedExtentsAreSeparateFromAccess) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = 0;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(device, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(device, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(device);

  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&reservation, 3 * granularity,
                                                 0, nullptr, 0));
  hipMemGenericAllocationHandle_t whole_handle = nullptr;
  hipMemGenericAllocationHandle_t replacement_handle = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&whole_handle, 3 * granularity, &properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&replacement_handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_map(reservation, 3 * granularity, 0, whole_handle, 0));
  EXPECT_NE(hipSuccess,
            api_.mem_map(reservation, granularity, 0, replacement_handle, 0));
  EXPECT_NE(hipSuccess, api_.mem_address_free(reservation, 3 * granularity));

  hipDeviceptr_t range_base = nullptr;
  size_t range_size = 0;
  ASSERT_EQ(hipSuccess, api_.mem_get_address_range(
                            &range_base, &range_size,
                            static_cast<uint8_t*>(reservation) + granularity));
  EXPECT_EQ(reservation, range_base);
  EXPECT_EQ(3 * granularity, range_size);
  size_t pointer_size = 0;
  EXPECT_EQ(hipSuccess, api_.mem_ptr_get_info(
                            static_cast<uint8_t*>(reservation) + granularity,
                            &pointer_size));
  EXPECT_EQ(3 * granularity, pointer_size);
  bool mapped = false;
  EXPECT_EQ(hipSuccess, api_.pointer_get_attribute(
                            &mapped, HIP_POINTER_ATTRIBUTE_MAPPED,
                            static_cast<hipDeviceptr_t>(reservation)));
  EXPECT_TRUE(mapped);
  uint32_t original_buffer_id = 0;
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attribute(
                &original_buffer_id, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                static_cast<hipDeviceptr_t>(reservation)));
  EXPECT_NE(0u, original_buffer_id);
  hipPointerAttribute_t aggregate_attributes = {};
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attributes(&aggregate_attributes, reservation));
  EXPECT_EQ(hipMemoryTypeDevice, aggregate_attributes.type);
  EXPECT_EQ(device, aggregate_attributes.device);
  EXPECT_EQ(nullptr, aggregate_attributes.devicePointer);

  const uint32_t value = 0x4A17B33Fu;
  uint32_t actual = 0;
  EXPECT_NE(hipSuccess, api_.memcpy(reservation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  EXPECT_NE(hipSuccess, api_.memset(reservation, 0, sizeof(value)));

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, device));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_load_data(&module, test_image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&function, module, "hrx_store_output"));
  hipDeviceptr_t output = static_cast<hipDeviceptr_t>(reservation);
  uint32_t kernel_value = 0x1397ACEDu;
  void* arguments[] = {&output, &kernel_value};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                      arguments, nullptr));

  hipMemLocation location = {hipMemLocationTypeDevice, device};
  unsigned long long access = ~0ull;
  ASSERT_EQ(hipSuccess, api_.mem_get_access(&access, &location, reservation));
  EXPECT_EQ(static_cast<unsigned long long>(hipMemAccessFlagsProtNone), access);

  uint8_t* middle = static_cast<uint8_t*>(reservation) + granularity;
  uint8_t* right = middle + granularity;
  hipMemAccessDesc descriptor = {};
  descriptor.location = location;
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(middle, granularity, &descriptor, 1));
  EXPECT_NE(hipSuccess, api_.memcpy(reservation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(middle, &value, sizeof(value), hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, middle, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(value, actual);
  EXPECT_NE(hipSuccess,
            api_.memcpy(right, &value, sizeof(value), hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.mem_get_address_range(&range_base, &range_size, middle));
  EXPECT_EQ(reservation, range_base);
  EXPECT_EQ(3 * granularity, range_size);

  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, 3 * granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.memcpy(reservation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(value, actual);
  ASSERT_EQ(hipSuccess, api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                                  nullptr, arguments, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(kernel_value, actual);

  descriptor.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, 3 * granularity, &descriptor, 1));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                      arguments, nullptr));
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, 3 * granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.module_unload(module));

  ASSERT_EQ(hipSuccess, api_.mem_unmap(middle, granularity));
  EXPECT_NE(hipSuccess,
            api_.mem_get_address_range(&range_base, &range_size, middle));
  EXPECT_NE(hipSuccess, api_.mem_ptr_get_info(middle, &pointer_size));
  EXPECT_NE(hipSuccess, api_.mem_unmap(middle, granularity));

  ASSERT_EQ(hipSuccess,
            api_.mem_get_address_range(&range_base, &range_size, reservation));
  EXPECT_EQ(reservation, range_base);
  EXPECT_EQ(granularity, range_size);
  ASSERT_EQ(hipSuccess,
            api_.mem_get_address_range(&range_base, &range_size, right));
  EXPECT_EQ(right, range_base);
  EXPECT_EQ(granularity, range_size);
  uint32_t right_buffer_id = 0;
  ASSERT_EQ(hipSuccess, api_.pointer_get_attribute(
                            &right_buffer_id, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                            static_cast<hipDeviceptr_t>(right)));
  EXPECT_EQ(original_buffer_id, right_buffer_id);

  ASSERT_EQ(hipSuccess,
            api_.mem_map(middle, granularity, 0, replacement_handle, 0));
  uint32_t replacement_buffer_id = 0;
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attribute(&replacement_buffer_id,
                                       HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                                       static_cast<hipDeviceptr_t>(middle)));
  EXPECT_NE(original_buffer_id, replacement_buffer_id);
  EXPECT_NE(hipSuccess,
            api_.memcpy(middle, &value, sizeof(value), hipMemcpyHostToDevice));
  EXPECT_EQ(hipSuccess,
            api_.memcpy(right, &value, sizeof(value), hipMemcpyHostToDevice));

  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(middle, granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(middle, &value, sizeof(value), hipMemcpyHostToDevice));

  ASSERT_EQ(hipSuccess, api_.mem_release(whole_handle));
  EXPECT_NE(hipSuccess, api_.mem_release(whole_handle));
  hipMemGenericAllocationHandle_t retained = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_retain_allocation_handle(&retained, reservation));
  EXPECT_EQ(whole_handle, retained);
  ASSERT_EQ(hipSuccess, api_.mem_release(replacement_handle));
  EXPECT_NE(hipSuccess, api_.mem_release(replacement_handle));

  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(middle, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(right, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(retained));
  EXPECT_EQ(hipSuccess, api_.mem_address_free(reservation, 3 * granularity));
}

TEST_F(HipVirtualMemoryApiTest,
       EqualAccessUpdatesNormalizeRunsAndRetireStaleCapabilities) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = 0;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(device, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(device, &granularity));
  ASSERT_GT(granularity, 0u);
  ASSERT_GE(granularity, 4u);
  const size_t reservation_size = 3 * granularity;
  const hipMemAllocationProp properties = DeviceProperties(device);

  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&reservation, reservation_size,
                                                 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&handle, reservation_size, &properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_map(reservation, reservation_size, 0, handle, 0));

  auto* left = static_cast<uint8_t*>(reservation);
  uint8_t* middle = left + granularity;
  uint8_t* right = middle + granularity;
  hipMemLocation location = {hipMemLocationTypeDevice, device};
  hipMemAccessDesc descriptor = {};
  descriptor.location = location;
  descriptor.flags = hipMemAccessFlagsProtReadWrite;

  // Separate adjacent grants must become one operational capability so a
  // legal transfer can span their boundary.
  ASSERT_EQ(hipSuccess, api_.mem_set_access(left, granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(middle, granularity, &descriptor, 1));
  ExpectAccessEveryByte(left, 2 * granularity, location,
                        hipMemAccessFlagsProtReadWrite);

  auto expect_round_trip = [&](uint8_t* address,
                               const std::array<uint8_t, 8>& expected) {
    std::array<uint8_t, 8> actual = {};
    ASSERT_EQ(hipSuccess, api_.memcpy(address, expected.data(), expected.size(),
                                      hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.memcpy(actual.data(), address, actual.size(),
                                      hipMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
  };
  const std::array<uint8_t, 8> adjacent_value = {0x10, 0x21, 0x32, 0x43,
                                                 0x54, 0x65, 0x76, 0x87};
  expect_round_trip(middle - 4, adjacent_value);

  // Leave two equal-permission neighbors separated by a gap, then bridge
  // them. The complete maximal run must receive one fresh capability.
  descriptor.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(middle, granularity, &descriptor, 1));
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(right, granularity, &descriptor, 1));
  ExpectAccessEveryByte(left, granularity, location,
                        hipMemAccessFlagsProtReadWrite);
  ExpectAccessEveryByte(middle, granularity, location,
                        hipMemAccessFlagsProtNone);
  ExpectAccessEveryByte(right, granularity, location,
                        hipMemAccessFlagsProtReadWrite);
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(middle, granularity, &descriptor, 1));
  ExpectAccessEveryByte(left, reservation_size, location,
                        hipMemAccessFlagsProtReadWrite);

  const std::array<uint8_t, 8> left_bridge_value = {0x91, 0x82, 0x73, 0x64,
                                                    0x55, 0x46, 0x37, 0x28};
  const std::array<uint8_t, 8> right_bridge_value = {0x19, 0x2A, 0x3B, 0x4C,
                                                     0x5D, 0x6E, 0x7F, 0x80};
  expect_round_trip(middle - 4, left_bridge_value);
  expect_round_trip(right - 4, right_bridge_value);

  // Capture a spanning graph operation under the current capability. A
  // redundant equal-permission subrange update must mint a replacement, so
  // replay through this old executable is rejected even though access remains
  // RW at every byte.
  const std::array<uint8_t, 8> graph_value = {0xA1, 0xB2, 0xC3, 0xD4,
                                              0xE5, 0xF6, 0x07, 0x18};
  hipGraph_t old_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&old_graph, 0));
  hipGraphNode_t old_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_memcpy_node_1d(
                &old_node, old_graph, nullptr, 0, middle - 4,
                graph_value.data(), graph_value.size(), hipMemcpyHostToDevice));
  hipGraphExec_t old_exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&old_exec, old_graph, nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(old_exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  std::array<uint8_t, 8> graph_actual = {};
  ASSERT_EQ(hipSuccess,
            api_.memcpy(graph_actual.data(), middle - 4, graph_actual.size(),
                        hipMemcpyDeviceToHost));
  EXPECT_EQ(graph_value, graph_actual);

  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(middle, granularity, &descriptor, 1));
  ExpectAccessEveryByte(left, reservation_size, location,
                        hipMemAccessFlagsProtReadWrite);
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(old_exec, nullptr));
  expect_round_trip(middle - 4, adjacent_value);
  expect_round_trip(right - 4, right_bridge_value);

  hipGraph_t fresh_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&fresh_graph, 0));
  hipGraphNode_t fresh_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_memcpy_node_1d(
                &fresh_node, fresh_graph, nullptr, 0, middle - 4,
                graph_value.data(), graph_value.size(), hipMemcpyHostToDevice));
  hipGraphExec_t fresh_exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&fresh_exec, fresh_graph,
                                               nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(fresh_exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  graph_actual.fill(0);
  ASSERT_EQ(hipSuccess,
            api_.memcpy(graph_actual.data(), middle - 4, graph_actual.size(),
                        hipMemcpyDeviceToHost));
  EXPECT_EQ(graph_value, graph_actual);

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(fresh_exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(fresh_graph));
  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(old_exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(old_graph));
  descriptor.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, reservation_size, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, reservation_size));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, reservation_size));
}

TEST_F(HipVirtualMemoryApiTest, PeerDeviceOwnsPhysicalAndAccessIdentity) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2) GTEST_SKIP() << "requires two devices";
  // hipDeviceCanAccessPeer does not yet surface backend topology. Exercise
  // the native cross-device mapping directly; set-access is the authoritative
  // capability check for this VMM contract.
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device 0 does not support HIP VMM";
  ASSERT_EQ(hipSuccess, QueryVmmSupport(1, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device 1 does not support HIP VMM";
  ASSERT_EQ(hipSuccess, api_.set_device(0));
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));

  const hipMemAllocationProp properties = DeviceProperties(1);
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));

  hipMemAccessDesc descriptors[2] = {};
  descriptors[0].location = {hipMemLocationTypeDevice, 0};
  descriptors[0].flags = hipMemAccessFlagsProtReadWrite;
  descriptors[1].location = {hipMemLocationTypeDevice, 1};
  descriptors[1].flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, descriptors, 2));

  int owner_device = -1;
  ASSERT_EQ(hipSuccess, api_.pointer_get_attribute(
                            &owner_device, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            static_cast<hipDeviceptr_t>(reservation)));
  EXPECT_EQ(1, owner_device);
  hipCtx_t owner_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.pointer_get_attribute(
                            &owner_context, HIP_POINTER_ATTRIBUTE_CONTEXT,
                            static_cast<hipDeviceptr_t>(reservation)));
  ASSERT_NE(nullptr, owner_context);
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(owner_context));
  hipDevice_t owner_context_device = -1;
  ASSERT_EQ(hipSuccess, api_.ctx_get_device(&owner_context_device));
  EXPECT_EQ(owner_device, owner_context_device);

  ASSERT_EQ(hipSuccess, api_.set_device(0));
  const uint32_t value = 0x9913A5C7u;
  ASSERT_EQ(hipSuccess, api_.memcpy(reservation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  hipDeviceptr_t device_pointer = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attribute(
                &device_pointer, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                static_cast<hipDeviceptr_t>(reservation)));
  EXPECT_EQ(reservation, device_pointer);

  hipMemAccessDesc revoke_owner = {};
  revoke_owner.location = {hipMemLocationTypeDevice, 0};
  revoke_owner.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &revoke_owner, 1));
  EXPECT_NE(hipSuccess, api_.memcpy(reservation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  device_pointer = nullptr;
  EXPECT_NE(hipSuccess,
            api_.pointer_get_attribute(
                &device_pointer, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                static_cast<hipDeviceptr_t>(reservation)));
  hipDeviceptr_t range_base = nullptr;
  size_t range_size = 0;
  EXPECT_EQ(hipSuccess,
            api_.mem_get_address_range(&range_base, &range_size, reservation));
  EXPECT_EQ(reservation, range_base);
  EXPECT_EQ(granularity, range_size);
  hipPointerAttribute_t aggregate_attributes = {};
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attributes(&aggregate_attributes, reservation));
  EXPECT_EQ(hipMemoryTypeDevice, aggregate_attributes.type);
  EXPECT_EQ(1, aggregate_attributes.device);
  EXPECT_EQ(nullptr, aggregate_attributes.devicePointer);

  hipDeviceProp_t device0_properties = {};
  ASSERT_EQ(hipSuccess,
            api_.get_device_properties(&device0_properties, /*device=*/0));
  const hrx_cts::AmdgpuExecutableTestImage device0_test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device0_properties.gcnArchName);
  ASSERT_NE(nullptr, device0_test_image.file)
      << "no embedded HSACO for " << device0_properties.gcnArchName;
  hipModule_t device0_module = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_load_data(&device0_module,
                                              device0_test_image.file->data));
  hipFunction_t device0_function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&device0_function, device0_module,
                                     "hrx_store_output"));
  hipDeviceptr_t device0_output = static_cast<hipDeviceptr_t>(reservation);
  uint32_t device0_kernel_value = 0xBAD0ACCEu;
  void* device0_arguments[] = {&device0_output, &device0_kernel_value};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.module_launch_kernel(device0_function, 1, 1, 1, 1, 1, 1, 0,
                                      nullptr, device0_arguments, nullptr));
  ASSERT_EQ(hipSuccess, api_.module_unload(device0_module));

  ASSERT_EQ(hipSuccess, api_.set_device(1));
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attribute(
                &device_pointer, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                static_cast<hipDeviceptr_t>(reservation)));
  EXPECT_EQ(reservation, device_pointer);
  aggregate_attributes = {};
  ASSERT_EQ(hipSuccess,
            api_.pointer_get_attributes(&aggregate_attributes, reservation));
  EXPECT_EQ(1, aggregate_attributes.device);
  EXPECT_EQ(reservation, aggregate_attributes.devicePointer);
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(reservation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(value, actual);

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, 1));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_load_data(&module, test_image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&function, module, "hrx_store_output"));
  hipDeviceptr_t output = static_cast<hipDeviceptr_t>(reservation);
  uint32_t kernel_value = 0xC001D00Du;
  void* arguments[] = {&output, &kernel_value};
  ASSERT_EQ(hipSuccess, api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                                  nullptr, arguments, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(kernel_value, actual);
  ASSERT_EQ(hipSuccess, api_.module_unload(module));

  ASSERT_EQ(hipSuccess, api_.set_device(0));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));

  const hipMemAllocationProp replacement_properties = DeviceProperties(0);
  hipMemGenericAllocationHandle_t replacement_handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&replacement_handle, granularity,
                                        &replacement_properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_map(reservation, granularity, 0, replacement_handle, 0));
  owner_device = -1;
  owner_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.pointer_get_attribute(
                            &owner_device, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            static_cast<hipDeviceptr_t>(reservation)));
  ASSERT_EQ(hipSuccess, api_.pointer_get_attribute(
                            &owner_context, HIP_POINTER_ATTRIBUTE_CONTEXT,
                            static_cast<hipDeviceptr_t>(reservation)));
  EXPECT_EQ(0, owner_device);
  ASSERT_NE(nullptr, owner_context);
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(owner_context));
  owner_context_device = -1;
  ASSERT_EQ(hipSuccess, api_.ctx_get_device(&owner_context_device));
  EXPECT_EQ(owner_device, owner_context_device);
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(replacement_handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
}

TEST_F(HipVirtualMemoryApiTest, NonVmmPointerKeepsKernelLaunchBehavior) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  ASSERT_EQ(hipSuccess, api_.set_device(0));

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, 0));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_load_data(&module, test_image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&function, module, "hrx_store_output"));

  void* output = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&output, sizeof(uint32_t)));
  uint32_t kernel_value = 0xE77E4A1u;
  void* arguments[] = {&output, &kernel_value};
  ASSERT_EQ(hipSuccess, api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                                  nullptr, arguments, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, output, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(kernel_value, actual);

  ASSERT_EQ(hipSuccess, api_.free(output));
  ASSERT_EQ(hipSuccess, api_.module_unload(module));
}

TEST_F(HipVirtualMemoryApiTest,
       PrimaryResetPreservesExplicitContextResourcesAndRetainAccounting) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  ASSERT_EQ(hipSuccess, api_.set_device(0));

  void* stale_primary_allocation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.malloc(&stale_primary_allocation, sizeof(uint32_t)));

  hipCtx_t explicit_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&explicit_context, 0, 0));
  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&allocation, sizeof(uint32_t)));
  const uint32_t value = 0xC017E57u;
  ASSERT_EQ(hipSuccess, api_.memcpy(allocation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));

  hipCtx_t retained_primary_a = nullptr;
  hipCtx_t retained_primary_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.device_primary_ctx_retain(&retained_primary_a, 0));
  ASSERT_EQ(hipSuccess, api_.device_primary_ctx_retain(&retained_primary_b, 0));
  ASSERT_EQ(retained_primary_a, retained_primary_b);
  EXPECT_EQ(hipErrorInvalidContext, api_.ctx_destroy(retained_primary_a));

  ASSERT_EQ(hipSuccess, api_.device_primary_ctx_reset(0));
  hipDevice_t current_device = -1;
  ASSERT_EQ(hipSuccess, api_.ctx_get_device(&current_device));
  EXPECT_EQ(0, current_device);
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, allocation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(value, actual);
  EXPECT_NE(hipSuccess, api_.ctx_set_current(retained_primary_a));
  uint32_t stale_primary_value = 0;
  EXPECT_NE(hipSuccess,
            api_.memcpy(&stale_primary_value, stale_primary_allocation,
                        sizeof(stale_primary_value), hipMemcpyDeviceToHost));

  void* second_allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&second_allocation, sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess, api_.free(second_allocation));
  hipCtx_t fresh_primary = nullptr;
  ASSERT_EQ(hipSuccess, api_.device_primary_ctx_retain(&fresh_primary, 0));
  ASSERT_NE(nullptr, fresh_primary);
  ASSERT_EQ(hipSuccess, api_.device_primary_ctx_release(0));

  ASSERT_EQ(hipSuccess, api_.ctx_set_current(explicit_context));
  ASSERT_EQ(hipSuccess, api_.free(allocation));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(explicit_context));
}

TEST_F(HipVirtualMemoryApiTest,
       WholeDeviceResetRetiresExplicitContextAndClearsCurrentTls) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  ASSERT_EQ(hipSuccess, api_.set_device(0));

  hipCtx_t explicit_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&explicit_context, 0, 0));
  void* stale_allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&stale_allocation, sizeof(uint32_t)));
  const uint32_t value = 0xD3A1CEu;
  ASSERT_EQ(hipSuccess, api_.memcpy(stale_allocation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));

  ASSERT_EQ(hipSuccess, api_.device_reset());
  EXPECT_NE(hipSuccess, api_.ctx_set_current(explicit_context));
  uint32_t actual = 0;
  EXPECT_NE(hipSuccess, api_.memcpy(&actual, stale_allocation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(explicit_context));

  void* fresh_allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&fresh_allocation, sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess, api_.memcpy(fresh_allocation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, fresh_allocation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(value, actual);
  ASSERT_EQ(hipSuccess, api_.free(fresh_allocation));
}

TEST_F(HipVirtualMemoryApiTest,
       DestroyedExplicitOwnerKeepsProcessVmmHandlesReleasable) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";

  hipCtx_t owner_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&owner_context, 0, 0));
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));
  hipMemAccessDesc access = {};
  access.location = {hipMemLocationTypeDevice, 0};
  access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &access, 1));

  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(owner_context));
  // hipCtxGetDevice follows the runtime API convention of lazily selecting a
  // primary context when this thread has no current context. Validate that
  // fallback independently from the destroyed explicit handle below.
  hipDevice_t destroyed_device = -1;
  EXPECT_EQ(hipSuccess, api_.ctx_get_device(&destroyed_device));
  EXPECT_EQ(0, destroyed_device);
  EXPECT_NE(hipSuccess, api_.ctx_set_current(owner_context));

  // Reservation/allocation registry records intentionally retain their owner
  // metadata after the public explicit context and its access alias are gone.
  hipMemAllocationProp queried_properties = {};
  ASSERT_EQ(hipSuccess,
            api_.mem_get_allocation_properties(&queried_properties, handle));
  EXPECT_EQ(0, queried_properties.location.id);
  hipDeviceptr_t range_base = nullptr;
  size_t range_size = 0;
  ASSERT_EQ(hipSuccess,
            api_.mem_get_address_range(&range_base, &range_size, reservation));
  EXPECT_EQ(reservation, range_base);
  EXPECT_EQ(granularity, range_size);

  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  ASSERT_EQ(hipSuccess, api_.init(0));
}

TEST_F(HipVirtualMemoryApiTest,
       GraphLaunchRevalidatesCapturedUpdatedAndDisabledNodes) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);

  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, 0));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_load_data(&module, test_image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&function, module, "hrx_store_output"));

  hipDeviceptr_t output = static_cast<hipDeviceptr_t>(reservation);
  uint32_t kernel_value = 0x61A9C3E5u;
  void* arguments[] = {&output, &kernel_value};
  hipKernelNodeParams node_params = {};
  node_params.blockDim = {1, 1, 1};
  node_params.func = function;
  node_params.gridDim = {1, 1, 1};
  node_params.kernelParams = arguments;

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, 0));
  hipGraphNode_t kernel_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_kernel_node(&kernel_node, graph, nullptr,
                                                   0, &node_params));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));

  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, 0};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(kernel_value, actual);

  descriptor.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));

  ASSERT_EQ(hipSuccess, api_.graph_node_set_enabled(exec, kernel_node, 0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  ASSERT_EQ(hipSuccess, api_.graph_node_set_enabled(exec, kernel_node, 1));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));

  void* ordinary_output = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&ordinary_output, sizeof(uint32_t)));
  uint32_t updated_value = 0x7D24B80Fu;
  void* updated_arguments[] = {&ordinary_output, &updated_value};
  hipKernelNodeParams updated_params = node_params;
  updated_params.kernelParams = updated_arguments;
  hipGraph_t updated_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&updated_graph, 0));
  hipGraphNode_t updated_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&updated_node, updated_graph, nullptr, 0,
                                       &updated_params));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess, api_.graph_exec_update(exec, updated_graph, &error_node,
                                               &update_result));
  EXPECT_EQ(nullptr, error_node);
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, ordinary_output, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(updated_value, actual);

  error_node = nullptr;
  update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess,
            api_.graph_exec_update(exec, graph, &error_node, &update_result));
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(updated_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  ASSERT_EQ(hipSuccess, api_.free(ordinary_output));
  ASSERT_EQ(hipSuccess, api_.module_unload(module));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
}

TEST_F(HipVirtualMemoryApiTest, GraphExecSnapshotRejectsDenseIndexReplacement) {
  ASSERT_EQ(hipSuccess, api_.init(0));

  std::atomic<int> original_count{0};
  std::atomic<int> replacement_count{0};
  std::atomic<int> setter_count{0};
  const hipHostNodeParams original_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &original_count,
  };
  const hipHostNodeParams replacement_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &replacement_count,
  };
  const hipHostNodeParams setter_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &setter_count,
  };

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, 0));
  hipGraphNode_t original_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&original_node, graph, nullptr,
                                                 0, &original_params));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

  ASSERT_EQ(hipSuccess, api_.graph_destroy_node(original_node));
  hipGraphNode_t replacement_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_host_node(&replacement_node, graph, nullptr, 0,
                                     &replacement_params));

  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_host_node_set_params(
                                      exec, replacement_node, &setter_params));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_host_node_set_params(
                                      exec, original_node, &setter_params));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(1, original_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, replacement_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, setter_count.load(std::memory_order_acquire));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipVirtualMemoryApiTest,
       GraphExecSetterUsesSnapshotAfterSourceTopologyMutationAndDestroy) {
  ASSERT_EQ(hipSuccess, api_.init(0));

  std::atomic<int> initial_count{0};
  std::atomic<int> updated_count{0};
  std::atomic<int> added_count{0};
  const hipHostNodeParams initial_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &initial_count,
  };
  const hipHostNodeParams updated_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &updated_count,
  };
  const hipHostNodeParams added_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &added_count,
  };

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, 0));
  hipGraphNode_t target_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&target_node, graph, nullptr,
                                                 0, &initial_params));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, graph, nullptr, nullptr, 0));

  hipGraphNode_t added_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&added_node, graph, nullptr, 0,
                                                 &added_params));
  ASSERT_EQ(hipSuccess, api_.graph_exec_host_node_set_params(exec, target_node,
                                                             &updated_params));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(0, initial_count.load(std::memory_order_acquire));
  EXPECT_EQ(1, updated_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, added_count.load(std::memory_order_acquire));

  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(2, updated_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, added_count.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_host_node_set_params(
                                      exec, target_node, &initial_params));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
}

TEST_F(HipVirtualMemoryApiTest,
       GraphExecUpdatePublishesOnlySuccessfulSnapshotIdentity) {
  ASSERT_EQ(hipSuccess, api_.init(0));

  std::atomic<int> old_count{0};
  std::atomic<int> new_count{0};
  std::atomic<int> updated_count{0};
  std::atomic<int> incompatible_count{0};
  const hipHostNodeParams old_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &old_count,
  };
  const hipHostNodeParams new_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &new_count,
  };
  const hipHostNodeParams updated_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &updated_count,
  };
  const hipHostNodeParams incompatible_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &incompatible_count,
  };

  hipGraph_t old_graph = nullptr;
  hipGraph_t new_graph = nullptr;
  hipGraph_t incompatible_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&old_graph, 0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&new_graph, 0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&incompatible_graph, 0));
  hipGraphNode_t old_node = nullptr;
  hipGraphNode_t new_node = nullptr;
  hipGraphNode_t incompatible_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&old_node, old_graph, nullptr,
                                                 0, &old_params));
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(&new_node, new_graph, nullptr,
                                                 0, &new_params));
  ASSERT_EQ(hipSuccess,
            api_.graph_add_host_node(&incompatible_node, incompatible_graph,
                                     nullptr, 0, &incompatible_params));
  hipGraphNode_t incompatible_tail = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(
                            &incompatible_tail, incompatible_graph,
                            &incompatible_node, 1, &incompatible_params));

  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, old_graph, nullptr, nullptr, 0));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess, api_.graph_exec_update(exec, new_graph, &error_node,
                                               &update_result));
  EXPECT_EQ(nullptr, error_node);
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_host_node_set_params(
                                      exec, old_node, &updated_params));
  ASSERT_EQ(hipSuccess, api_.graph_exec_host_node_set_params(exec, new_node,
                                                             &updated_params));

  error_node = nullptr;
  update_result = hipGraphExecUpdateError;
  EXPECT_EQ(hipErrorGraphExecUpdateFailure,
            api_.graph_exec_update(exec, incompatible_graph, &error_node,
                                   &update_result));
  EXPECT_EQ(hipGraphExecUpdateErrorTopologyChanged, update_result);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_exec_host_node_set_params(exec, incompatible_node,
                                                 &incompatible_params));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(0, old_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, new_count.load(std::memory_order_acquire));
  EXPECT_EQ(1, updated_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, incompatible_count.load(std::memory_order_acquire));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(incompatible_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(new_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(old_graph));
}

TEST_F(HipVirtualMemoryApiTest,
       ChildGraphExecSetterRequiresRecursiveTopologyCompatibility) {
  ASSERT_EQ(hipSuccess, api_.init(0));

  std::atomic<int> old_count{0};
  std::atomic<int> rejected_count{0};
  std::atomic<int> new_count{0};
  const hipHostNodeParams old_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &old_count,
  };
  const hipHostNodeParams rejected_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &rejected_count,
  };
  const hipHostNodeParams new_params = {
      .fn = &IncrementAtomicCounter,
      .userData = &new_count,
  };

  auto add_host_chain = [&](hipGraph_t graph, const hipHostNodeParams* params) {
    hipGraphNode_t first = nullptr;
    EXPECT_EQ(hipSuccess,
              api_.graph_add_host_node(&first, graph, nullptr, 0, params));
    hipGraphNode_t second = nullptr;
    EXPECT_EQ(hipSuccess,
              api_.graph_add_host_node(&second, graph, &first, 1, params));
  };

  hipGraph_t old_child = nullptr;
  hipGraph_t edge_mismatch_child = nullptr;
  hipGraph_t type_mismatch_child = nullptr;
  hipGraph_t new_child = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&old_child, 0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&edge_mismatch_child, 0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&type_mismatch_child, 0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&new_child, 0));
  add_host_chain(old_child, &old_params);
  ASSERT_FALSE(HasFailure());
  hipGraphNode_t edge_first = nullptr;
  hipGraphNode_t edge_second = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_host_node(&edge_first, edge_mismatch_child, nullptr,
                                     0, &rejected_params));
  ASSERT_EQ(hipSuccess,
            api_.graph_add_host_node(&edge_second, edge_mismatch_child, nullptr,
                                     0, &rejected_params));
  hipGraphNode_t empty_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(
                            &empty_node, type_mismatch_child, nullptr, 0));
  hipGraphNode_t typed_tail = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_host_node(&typed_tail, type_mismatch_child,
                                     &empty_node, 1, &rejected_params));
  add_host_chain(new_child, &new_params);
  ASSERT_FALSE(HasFailure());

  hipGraph_t parent_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&parent_graph, 0));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_child_graph_node(
                            &child_node, parent_graph, nullptr, 0, old_child));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, parent_graph, nullptr, nullptr, 0));

  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_child_graph_node_set_params(
                                      exec, child_node, edge_mismatch_child));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(2, old_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, rejected_count.load(std::memory_order_acquire));

  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_child_graph_node_set_params(
                                      exec, child_node, type_mismatch_child));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(4, old_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, rejected_count.load(std::memory_order_acquire));

  ASSERT_EQ(hipSuccess, api_.graph_exec_child_graph_node_set_params(
                            exec, child_node, new_child));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(4, old_count.load(std::memory_order_acquire));
  EXPECT_EQ(2, new_count.load(std::memory_order_acquire));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(parent_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(new_child));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(type_mismatch_child));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(edge_mismatch_child));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(old_child));
}

TEST_F(HipVirtualMemoryApiTest,
       DisabledMemcpyStateSurvivesStagedOrientationGrowthAndShrink) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));

  hipCtx_t local_context = nullptr;
  hipCtx_t remote_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&local_context, 0, device));
  void* local_source = nullptr;
  void* local_destination = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&local_source, sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess, api_.malloc(&local_destination, sizeof(uint32_t)));
  const uint32_t local_source_value = 0x1537A9CBu;
  const uint32_t local_destination_initial = 0x2648BDF0u;
  ASSERT_EQ(hipSuccess,
            api_.memcpy(local_source, &local_source_value,
                        sizeof(local_source_value), hipMemcpyHostToDevice));
  ASSERT_EQ(
      hipSuccess,
      api_.memcpy(local_destination, &local_destination_initial,
                  sizeof(local_destination_initial), hipMemcpyHostToDevice));

  ASSERT_EQ(hipSuccess, api_.ctx_create(&remote_context, 0, device));
  void* remote_buffer = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&remote_buffer, sizeof(uint32_t)));
  const uint32_t remote_initial = 0xF0DB8642u;
  ASSERT_EQ(hipSuccess,
            api_.memcpy(remote_buffer, &remote_initial, sizeof(remote_initial),
                        hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(local_context));

  auto add_copy = [&](hipGraph_t* out_graph, hipGraphNode_t* out_node,
                      void* dst, const void* src) {
    ASSERT_EQ(hipSuccess, api_.graph_create(out_graph, 0));
    ASSERT_EQ(hipSuccess, api_.graph_add_memcpy_node_1d(
                              out_node, *out_graph, nullptr, 0, dst, src,
                              sizeof(uint32_t), hipMemcpyDeviceToDevice));
  };

  hipGraph_t local_graph = nullptr;
  hipGraph_t pre_staged_graph = nullptr;
  hipGraph_t post_staged_graph = nullptr;
  hipGraph_t incompatible_graph = nullptr;
  hipGraphNode_t local_node = nullptr;
  hipGraphNode_t pre_staged_node = nullptr;
  hipGraphNode_t post_staged_node = nullptr;
  hipGraphNode_t incompatible_node = nullptr;
  add_copy(&local_graph, &local_node, local_destination, local_source);
  add_copy(&pre_staged_graph, &pre_staged_node, local_destination,
           remote_buffer);
  add_copy(&post_staged_graph, &post_staged_node, remote_buffer, local_source);
  add_copy(&incompatible_graph, &incompatible_node, local_destination,
           remote_buffer);
  hipGraphNode_t extra_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&extra_node, incompatible_graph,
                                      &incompatible_node, 1));

  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, local_graph, nullptr, nullptr, 0));
  ASSERT_EQ(hipSuccess, api_.graph_node_set_enabled(exec, local_node, 0));
  unsigned int is_enabled = 1;
  ASSERT_EQ(hipSuccess,
            api_.graph_node_get_enabled(exec, local_node, &is_enabled));
  EXPECT_EQ(0u, is_enabled);

  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess, api_.graph_exec_update(exec, pre_staged_graph,
                                               &error_node, &update_result));
  EXPECT_EQ(nullptr, error_node);
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  is_enabled = 1;
  ASSERT_EQ(hipSuccess,
            api_.graph_node_get_enabled(exec, pre_staged_node, &is_enabled));
  EXPECT_EQ(0u, is_enabled);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_node_get_enabled(exec, local_node, &is_enabled));

  error_node = nullptr;
  update_result = hipGraphExecUpdateError;
  EXPECT_EQ(hipErrorGraphExecUpdateFailure,
            api_.graph_exec_update(exec, incompatible_graph, &error_node,
                                   &update_result));
  EXPECT_EQ(hipGraphExecUpdateErrorTopologyChanged, update_result);
  is_enabled = 1;
  ASSERT_EQ(hipSuccess,
            api_.graph_node_get_enabled(exec, pre_staged_node, &is_enabled));
  EXPECT_EQ(0u, is_enabled);
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, local_destination, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(local_destination_initial, actual);

  error_node = nullptr;
  update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess, api_.graph_exec_update(exec, post_staged_graph,
                                               &error_node, &update_result));
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  is_enabled = 1;
  ASSERT_EQ(hipSuccess,
            api_.graph_node_get_enabled(exec, post_staged_node, &is_enabled));
  EXPECT_EQ(0u, is_enabled);
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(remote_context));
  actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, remote_buffer, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(remote_initial, actual);

  ASSERT_EQ(hipSuccess, api_.ctx_set_current(local_context));
  error_node = nullptr;
  update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess, api_.graph_exec_update(exec, local_graph, &error_node,
                                               &update_result));
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  is_enabled = 1;
  ASSERT_EQ(hipSuccess,
            api_.graph_node_get_enabled(exec, local_node, &is_enabled));
  EXPECT_EQ(0u, is_enabled);
  ASSERT_EQ(hipSuccess, api_.graph_node_set_enabled(exec, local_node, 1));
  ASSERT_EQ(hipSuccess,
            api_.graph_node_get_enabled(exec, local_node, &is_enabled));
  EXPECT_EQ(1u, is_enabled);
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, local_destination, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(local_source_value, actual);

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(incompatible_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(post_staged_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(pre_staged_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(local_graph));
  ASSERT_EQ(hipSuccess, api_.free(local_destination));
  ASSERT_EQ(hipSuccess, api_.free(local_source));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(remote_context));
  ASSERT_EQ(hipSuccess, api_.free(remote_buffer));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(remote_context));
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(local_context));
}

TEST_F(HipVirtualMemoryApiTest,
       ChildGraphValidationTracksAccessAndParentEnableState) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);

  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, 0));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_load_data(&module, test_image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&function, module, "hrx_store_output"));

  hipDeviceptr_t output = static_cast<hipDeviceptr_t>(reservation);
  uint32_t kernel_value = 0x351FCA9Du;
  void* arguments[] = {&output, &kernel_value};
  hipKernelNodeParams node_params = {};
  node_params.blockDim = {1, 1, 1};
  node_params.func = function;
  node_params.gridDim = {1, 1, 1};
  node_params.kernelParams = arguments;

  hipGraph_t child_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&child_graph, 0));
  hipGraphNode_t child_kernel_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&child_kernel_node, child_graph, nullptr,
                                       0, &node_params));
  hipGraph_t parent_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&parent_graph, 0));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_child_graph_node(&child_node, parent_graph, nullptr,
                                            0, child_graph));
  hipGraphExec_t exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&exec, parent_graph, nullptr, nullptr, 0));

  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));

  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, 0};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.graph_launch(exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  uint32_t actual = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&actual, reservation, sizeof(actual),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(kernel_value, actual);

  descriptor.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(exec, nullptr));

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(parent_graph));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(child_graph));
  ASSERT_EQ(hipSuccess, api_.module_unload(module));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
}

TEST_F(HipVirtualMemoryApiTest, GraphLaunchSerializesWithAccessRevocation) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));
  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, 0};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, 0));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  StoreKernelGraph graph;
  ASSERT_EQ(hipSuccess,
            CreateStoreKernelGraph(test_image.file->data, reservation,
                                   0xC04C7A11u, &graph));
  ASSERT_EQ(hipSuccess, api_.graph_launch(graph.exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());

  std::atomic<bool> start{false};
  std::atomic<int> launch_result{hipErrorUnknown};
  std::thread launch_thread([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    launch_result.store(api_.graph_launch(graph.exec, nullptr),
                        std::memory_order_release);
  });
  start.store(true, std::memory_order_release);
  descriptor.flags = hipMemAccessFlagsProtNone;
  const hipError_t revoke_result =
      api_.mem_set_access(reservation, granularity, &descriptor, 1);
  launch_thread.join();
  ASSERT_EQ(hipSuccess, revoke_result);
  const hipError_t raced_result =
      static_cast<hipError_t>(launch_result.load(std::memory_order_acquire));
  EXPECT_TRUE(raced_result == hipSuccess ||
              raced_result == hipErrorInvalidValue);
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(graph.exec, nullptr));

  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.graph_launch(graph.exec, nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());

  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(graph.exec));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph.graph));
  ASSERT_EQ(hipSuccess, api_.module_unload(graph.module));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(reservation, granularity));
}

TEST_F(HipVirtualMemoryApiTest, GraphLaunchSerializesWithDeviceReset) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);
  void* reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&reservation, granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(reservation, granularity, 0, handle, 0));
  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, 0};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(reservation, granularity, &descriptor, 1));

  hipDeviceProp_t device_properties = {};
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&device_properties, 0));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(device_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << device_properties.gcnArchName;
  StoreKernelGraph graph;
  ASSERT_EQ(hipSuccess,
            CreateStoreKernelGraph(test_image.file->data, reservation,
                                   0x8E5E7A11u, &graph));

  std::atomic<bool> start{false};
  std::atomic<int> launch_result{hipErrorUnknown};
  std::thread launch_thread([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    launch_result.store(api_.graph_launch(graph.exec, nullptr),
                        std::memory_order_release);
  });
  start.store(true, std::memory_order_release);
  const hipError_t reset_result = api_.device_reset();
  launch_thread.join();

  ASSERT_EQ(hipSuccess, reset_result);
  const hipError_t raced_result =
      static_cast<hipError_t>(launch_result.load(std::memory_order_acquire));
  EXPECT_TRUE(raced_result == hipSuccess ||
              raced_result == hipErrorInvalidValue);
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_launch(graph.exec, nullptr));

  // Device reset owns every resource layered on the target context. Their
  // exact handles were consumed during the reset commit and must not expose a
  // second public destruction edge.
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_exec_destroy(graph.exec));
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_destroy(graph.graph));
  EXPECT_EQ(hipErrorInvalidResourceHandle, api_.module_unload(graph.module));
}

TEST_F(HipVirtualMemoryApiTest, DeviceResetInvalidatesOnlyInvolvedObjects) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2) GTEST_SKIP() << "requires two devices";
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device 0 does not support HIP VMM";
  ASSERT_EQ(hipSuccess, QueryVmmSupport(1, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device 1 does not support HIP VMM";

  ASSERT_EQ(hipSuccess, api_.set_device(1));
  size_t unaffected_granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(1, &unaffected_granularity));
  ASSERT_GT(unaffected_granularity, 0u);
  const hipMemAllocationProp unaffected_properties = DeviceProperties(1);
  void* unaffected_reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&unaffected_reservation,
                                     unaffected_granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t unaffected_handle = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&unaffected_handle, unaffected_granularity,
                            &unaffected_properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_map(unaffected_reservation, unaffected_granularity, 0,
                         unaffected_handle, 0));
  hipMemAccessDesc unaffected_access = {};
  unaffected_access.location = {hipMemLocationTypeDevice, 1};
  unaffected_access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(unaffected_reservation, unaffected_granularity,
                                &unaffected_access, 1));
  const uint32_t unaffected_value = 0xA17EC7EDu;
  ASSERT_EQ(hipSuccess,
            api_.memcpy(unaffected_reservation, &unaffected_value,
                        sizeof(unaffected_value), hipMemcpyHostToDevice));

  ASSERT_EQ(hipSuccess, api_.set_device(0));
  size_t affected_granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &affected_granularity));
  ASSERT_GT(affected_granularity, 0u);
  const hipMemAllocationProp affected_properties = DeviceProperties(0);
  void* affected_reservation = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_address_reserve(&affected_reservation,
                                     affected_granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t affected_handle = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_create(&affected_handle, affected_granularity,
                                        &affected_properties, 0));
  ASSERT_EQ(hipSuccess, api_.mem_map(affected_reservation, affected_granularity,
                                     0, affected_handle, 0));
  hipMemAccessDesc affected_access = {};
  affected_access.location = {hipMemLocationTypeDevice, 0};
  affected_access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess,
            api_.mem_set_access(affected_reservation, affected_granularity,
                                &affected_access, 1));

  ASSERT_EQ(hipSuccess, api_.device_reset());

  hipMemAllocationProp queried_properties = {};
  EXPECT_EQ(hipErrorInvalidValue, api_.mem_get_allocation_properties(
                                      &queried_properties, affected_handle));
  EXPECT_EQ(hipErrorInvalidValue, api_.mem_release(affected_handle));
  hipDeviceptr_t range_base = nullptr;
  size_t range_size = 0;
  EXPECT_NE(hipSuccess, api_.mem_get_address_range(&range_base, &range_size,
                                                   affected_reservation));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.mem_unmap(affected_reservation, affected_granularity));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.mem_address_free(affected_reservation, affected_granularity));

  ASSERT_EQ(hipSuccess, api_.set_device(1));
  EXPECT_EQ(hipSuccess, api_.mem_get_allocation_properties(&queried_properties,
                                                           unaffected_handle));
  EXPECT_EQ(1, queried_properties.location.id);
  ASSERT_EQ(hipSuccess, api_.mem_get_address_range(&range_base, &range_size,
                                                   unaffected_reservation));
  EXPECT_EQ(unaffected_reservation, range_base);
  EXPECT_EQ(unaffected_granularity, range_size);
  uint32_t unaffected_actual = 0;
  ASSERT_EQ(hipSuccess,
            api_.memcpy(&unaffected_actual, unaffected_reservation,
                        sizeof(unaffected_actual), hipMemcpyDeviceToHost));
  EXPECT_EQ(unaffected_value, unaffected_actual);

  ASSERT_EQ(hipSuccess,
            api_.mem_unmap(unaffected_reservation, unaffected_granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(unaffected_handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(unaffected_reservation,
                                              unaffected_granularity));
}

TEST_F(HipVirtualMemoryApiTest,
       DeinitRejectsLiveExplicitContextAndAllocationBeforeCommit) {
  ASSERT_EQ(hipSuccess, api_.init(0));

  hipCtx_t explicit_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_create(&explicit_context, 0, 0));
  ASSERT_EQ(hipSuccess, api_.ctx_set_current(nullptr));
  EXPECT_EQ(hipErrorInvalidContext, api_.hal_deinit());
  int device = -1;
  EXPECT_EQ(hipSuccess, api_.get_device(&device));
  EXPECT_EQ(0, device);
  ASSERT_EQ(hipSuccess, api_.ctx_destroy(explicit_context));

  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&allocation, 64));
  EXPECT_EQ(hipErrorInvalidContext, api_.hal_deinit());
  const uint32_t value = 0xD31A117Eu;
  ASSERT_EQ(hipSuccess, api_.memcpy(allocation, &value, sizeof(value),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.free(allocation));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  ASSERT_EQ(hipSuccess, api_.init(0));
}

TEST_F(HipVirtualMemoryApiTest,
       DeinitRejectsRemotePerThreadStreamAndSucceedsAfterThreadExit) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&allocation, 64));

  std::mutex worker_mutex;
  std::condition_variable worker_cv;
  bool worker_ready = false;
  bool worker_exit = false;
  std::atomic<int> memset_result{hipErrorUnknown};
  std::atomic<int> synchronize_result{hipErrorUnknown};
  std::atomic<int> clear_context_result{hipErrorUnknown};
  std::thread worker([&] {
    memset_result.store(
        api_.memset_async(allocation, 0x5A, 64, hipStreamPerThread),
        std::memory_order_release);
    synchronize_result.store(api_.stream_synchronize(hipStreamPerThread),
                             std::memory_order_release);
    clear_context_result.store(api_.ctx_set_current(nullptr),
                               std::memory_order_release);
    std::unique_lock<std::mutex> lock(worker_mutex);
    worker_ready = true;
    worker_cv.notify_all();
    worker_cv.wait(lock, [&] { return worker_exit; });
  });

  {
    std::unique_lock<std::mutex> lock(worker_mutex);
    worker_cv.wait(lock, [&] { return worker_ready; });
  }
  EXPECT_EQ(hipSuccess, static_cast<hipError_t>(
                            memset_result.load(std::memory_order_acquire)));
  EXPECT_EQ(hipSuccess, static_cast<hipError_t>(synchronize_result.load(
                            std::memory_order_acquire)));
  EXPECT_EQ(hipSuccess, static_cast<hipError_t>(clear_context_result.load(
                            std::memory_order_acquire)));
  EXPECT_EQ(hipSuccess, api_.free(allocation));

  EXPECT_EQ(hipErrorInvalidContext, api_.hal_deinit());
  int device = -1;
  EXPECT_EQ(hipSuccess, api_.get_device(&device));
  EXPECT_EQ(0, device);

  {
    std::lock_guard<std::mutex> lock(worker_mutex);
    worker_exit = true;
  }
  worker_cv.notify_all();
  worker.join();
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  ASSERT_EQ(hipSuccess, api_.init(0));
}

TEST_F(HipVirtualMemoryApiTest, GlobalDeinitInvalidatesOldGeneration) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);

  void* stale_reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&stale_reservation,
                                                 granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t stale_handle = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&stale_handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_map(stale_reservation, granularity, 0, stale_handle, 0));
  hipMemAccessDesc descriptor = {};
  descriptor.location = {hipMemLocationTypeDevice, 0};
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess, api_.mem_set_access(stale_reservation, granularity,
                                            &descriptor, 1));

  ASSERT_EQ(hipSuccess, api_.hal_deinit());
  hipMemAllocationProp queried_properties = {};
  EXPECT_EQ(hipErrorNotInitialized, api_.mem_get_allocation_properties(
                                        &queried_properties, stale_handle));

  ASSERT_EQ(hipSuccess, api_.init(0));
  EXPECT_EQ(hipErrorInvalidValue, api_.mem_get_allocation_properties(
                                      &queried_properties, stale_handle));
  EXPECT_EQ(hipErrorInvalidValue, api_.mem_release(stale_handle));
  hipDeviceptr_t range_base = nullptr;
  size_t range_size = 0;
  EXPECT_NE(hipSuccess, api_.mem_get_address_range(&range_base, &range_size,
                                                   stale_reservation));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.mem_unmap(stale_reservation, granularity));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.mem_address_free(stale_reservation, granularity));

  void* fresh_reservation = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_address_reserve(&fresh_reservation,
                                                 granularity, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t fresh_handle = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&fresh_handle, granularity, &properties, 0));
  ASSERT_EQ(hipSuccess,
            api_.mem_map(fresh_reservation, granularity, 0, fresh_handle, 0));
  ASSERT_EQ(hipSuccess, api_.mem_set_access(fresh_reservation, granularity,
                                            &descriptor, 1));
  ASSERT_EQ(hipSuccess, api_.mem_unmap(fresh_reservation, granularity));
  ASSERT_EQ(hipSuccess, api_.mem_release(fresh_handle));
  ASSERT_EQ(hipSuccess, api_.mem_address_free(fresh_reservation, granularity));
}

TEST_F(HipVirtualMemoryApiTest, ConcurrentCallsAreSerializedAgainstDeinit) {
  ASSERT_EQ(hipSuccess, api_.init(0));
  bool vmm_supported = false;
  ASSERT_EQ(hipSuccess, QueryVmmSupport(0, &vmm_supported));
  if (!vmm_supported) GTEST_SKIP() << "device does not support HIP VMM";
  size_t granularity = 0;
  ASSERT_EQ(hipSuccess, QueryMinimumGranularity(0, &granularity));
  ASSERT_GT(granularity, 0u);
  const hipMemAllocationProp properties = DeviceProperties(0);
  hipMemGenericAllocationHandle_t stale_handle = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.mem_create(&stale_handle, granularity, &properties, 0));

  std::atomic<bool> stop{false};
  int successful_calls = 0;
  int teardown_rejections = 0;
  int unexpected_results = 0;
  std::mutex handshake_mutex;
  std::condition_variable handshake_cv;
  bool start = false;
  std::vector<std::thread> workers;
  workers.reserve(4);
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&] {
      {
        std::unique_lock<std::mutex> lock(handshake_mutex);
        handshake_cv.wait(lock, [&] { return start; });
      }
      while (!stop.load(std::memory_order_acquire)) {
        hipMemAllocationProp queried_properties = {};
        const hipError_t result = api_.mem_get_allocation_properties(
            &queried_properties, stale_handle);
        {
          std::lock_guard<std::mutex> lock(handshake_mutex);
          if (result == hipSuccess) {
            ++successful_calls;
          } else if (result == hipErrorNotInitialized) {
            ++teardown_rejections;
          } else {
            ++unexpected_results;
          }
        }
        handshake_cv.notify_all();
      }
    });
  }

  {
    std::lock_guard<std::mutex> lock(handshake_mutex);
    start = true;
  }
  handshake_cv.notify_all();
  {
    std::unique_lock<std::mutex> lock(handshake_mutex);
    handshake_cv.wait(lock, [&] { return successful_calls >= 4; });
  }
  const hipError_t deinit_result = api_.hal_deinit();
  {
    std::unique_lock<std::mutex> lock(handshake_mutex);
    handshake_cv.wait(lock, [&] { return teardown_rejections > 0; });
  }
  stop.store(true, std::memory_order_release);
  for (std::thread& worker : workers) worker.join();

  EXPECT_EQ(hipSuccess, deinit_result);
  EXPECT_GT(successful_calls, 0);
  EXPECT_GT(teardown_rejections, 0);
  EXPECT_EQ(0, unexpected_results);

  ASSERT_EQ(hipSuccess, api_.init(0));
  hipMemAllocationProp queried_properties = {};
  EXPECT_EQ(hipErrorInvalidValue, api_.mem_get_allocation_properties(
                                      &queried_properties, stale_handle));
  ASSERT_EQ(hipSuccess, api_.hal_deinit());
}

}  // namespace
