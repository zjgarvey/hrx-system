// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipGetDevicePropertiesFn = hipError_t (*)(hipDeviceProp_t* properties,
                                                int device);
using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamGetIdFn = hipError_t (*)(hipStream_t stream,
                                        unsigned long long* stream_id);
using HipLaunchKernelFn = hipError_t (*)(const void* function, dim3 grid_dim,
                                         dim3 block_dim, void** arguments,
                                         size_t shared_memory_bytes,
                                         hipStream_t stream);
using HipExtLaunchKernelFn = hipError_t (*)(const void* function, dim3 grid_dim,
                                            dim3 block_dim, void** arguments,
                                            size_t shared_memory_bytes,
                                            hipStream_t stream,
                                            hipEvent_t start_event,
                                            hipEvent_t stop_event, int flags);
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t* module,
                                           const void* image);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipFuncGetAttributeFn = hipError_t (*)(int* value,
                                             hipFuncAttribute_t attribute,
                                             hipFunction_t function);
using HipFuncSetAttributeFn = hipError_t (*)(hipFunction_t function,
                                             hipFuncAttribute_t attribute,
                                             int value);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddKernelNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* params);
using HipGraphKernelNodeGetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                     void* params);
using HipGraphKernelNodeSetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                     const void* params);

// Owns an RTLD_LOCAL HIP runtime instance and the entry points exercised by
// this test. All calls use the loaded library instead of a link-time runtime.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Queries the thread's current device ordinal.
  HipGetDeviceFn get_device = nullptr;
  // Queries the architecture used to select embedded executable data.
  HipGetDevicePropertiesFn get_device_properties = nullptr;
  // Queries hardware limits used to derive launch boundaries.
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  // Creates the stream used by immediate launch entry points.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys the stream used by immediate launch entry points.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Queries a stream without dereferencing a stale public handle.
  HipStreamGetIdFn stream_get_id = nullptr;
  // Launches a registered runtime kernel.
  HipLaunchKernelFn launch_kernel = nullptr;
  // Launches a registered runtime kernel with extended launch arguments.
  HipExtLaunchKernelFn ext_launch_kernel = nullptr;
  // Launches a module kernel with prepacked or pointer-array arguments.
  HipModuleLaunchKernelFn module_launch_kernel = nullptr;
  // Loads an in-memory module through the public driver ABI.
  HipModuleLoadDataFn module_load_data = nullptr;
  // Resolves a function owned by a loaded public module.
  HipModuleGetFunctionFn module_get_function = nullptr;
  // Unloads a public module after every dependent assertion completes.
  HipModuleUnloadFn module_unload = nullptr;
  // Queries a cached function compatibility attribute.
  HipFuncGetAttributeFn function_get_attribute = nullptr;
  // Updates a mutable function compatibility attribute.
  HipFuncSetAttributeFn function_set_attribute = nullptr;
  // Creates a graph template.
  HipGraphCreateFn graph_create = nullptr;
  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;
  // Adds a kernel node to a graph template.
  HipGraphAddKernelNodeFn graph_add_kernel_node = nullptr;
  // Reads the parameters retained by a graph kernel node.
  HipGraphKernelNodeGetParamsFn graph_kernel_node_get_params = nullptr;
  // Replaces the parameters retained by a graph kernel node.
  HipGraphKernelNodeSetParamsFn graph_kernel_node_set_params = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipLaunchValidationApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      const char* library_path = CandidateLibPath();
      ASSERT_NE(library_path, nullptr)
          << "the build must provide the libamdhip64 artifact under test";
      api_.library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
      ASSERT_NE(api_.library, nullptr)
          << "cannot dlopen " << library_path << ": " << dlerror();

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.get_device =
          ResolveHipSymbol<HipGetDeviceFn>(api_.library, "hipGetDevice");
      api_.get_device_properties = ResolveHipSymbol<HipGetDevicePropertiesFn>(
          api_.library, "hipGetDeviceProperties");
      api_.device_get_attribute = ResolveHipSymbol<HipDeviceGetAttributeFn>(
          api_.library, "hipDeviceGetAttribute");
      api_.stream_create =
          ResolveHipSymbol<HipStreamCreateFn>(api_.library, "hipStreamCreate");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.stream_get_id =
          ResolveHipSymbol<HipStreamGetIdFn>(api_.library, "hipStreamGetId");
      api_.launch_kernel =
          ResolveHipSymbol<HipLaunchKernelFn>(api_.library, "hipLaunchKernel");
      api_.ext_launch_kernel = ResolveHipSymbol<HipExtLaunchKernelFn>(
          api_.library, "hipExtLaunchKernel");
      api_.module_launch_kernel = ResolveHipSymbol<HipModuleLaunchKernelFn>(
          api_.library, "hipModuleLaunchKernel");
      api_.module_load_data = ResolveHipSymbol<HipModuleLoadDataFn>(
          api_.library, "hipModuleLoadData");
      api_.module_get_function = ResolveHipSymbol<HipModuleGetFunctionFn>(
          api_.library, "hipModuleGetFunction");
      api_.module_unload =
          ResolveHipSymbol<HipModuleUnloadFn>(api_.library, "hipModuleUnload");
      api_.function_get_attribute = ResolveHipSymbol<HipFuncGetAttributeFn>(
          api_.library, "hipFuncGetAttribute");
      api_.function_set_attribute = ResolveHipSymbol<HipFuncSetAttributeFn>(
          api_.library, "hipFuncSetAttribute");
      api_.graph_create =
          ResolveHipSymbol<HipGraphCreateFn>(api_.library, "hipGraphCreate");
      api_.graph_destroy =
          ResolveHipSymbol<HipGraphDestroyFn>(api_.library, "hipGraphDestroy");
      api_.graph_add_kernel_node = ResolveHipSymbol<HipGraphAddKernelNodeFn>(
          api_.library, "hipGraphAddKernelNode");
      api_.graph_kernel_node_get_params =
          ResolveHipSymbol<HipGraphKernelNodeGetParamsFn>(
              api_.library, "hipGraphKernelNodeGetParams");
      api_.graph_kernel_node_set_params =
          ResolveHipSymbol<HipGraphKernelNodeSetParamsFn>(
              api_.library, "hipGraphKernelNodeSetParams");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.get_device_properties);
    ASSERT_NE(nullptr, api_.device_get_attribute);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.stream_get_id);
    ASSERT_NE(nullptr, api_.launch_kernel);
    ASSERT_NE(nullptr, api_.ext_launch_kernel);
    ASSERT_NE(nullptr, api_.module_launch_kernel);
    ASSERT_NE(nullptr, api_.module_load_data);
    ASSERT_NE(nullptr, api_.module_get_function);
    ASSERT_NE(nullptr, api_.module_unload);
    ASSERT_NE(nullptr, api_.function_get_attribute);
    ASSERT_NE(nullptr, api_.function_set_attribute);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_destroy);
    ASSERT_NE(nullptr, api_.graph_add_kernel_node);
    ASSERT_NE(nullptr, api_.graph_kernel_node_get_params);
    ASSERT_NE(nullptr, api_.graph_kernel_node_set_params);

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
    ASSERT_EQ(hipSuccess, api_.get_device(&device_));
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess, api_.get_device_properties(&properties, device_));
    const hrx_cts::AmdgpuExecutableTestImage test_image =
        hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
    ASSERT_NE(nullptr, test_image.file)
        << "no embedded HSACO for " << properties.gcnArchName;
    ASSERT_EQ(hipSuccess,
              api_.module_load_data(&module_, test_image.file->data));
    ASSERT_EQ(hipSuccess,
              api_.module_get_function(&noop_function_, module_, "hrx_noop"));
    ASSERT_EQ(hipSuccess,
              api_.module_get_function(&store_output_function_, module_,
                                       "hrx_store_output"));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
      stream_ = nullptr;
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, api_.module_unload(module_));
      module_ = nullptr;
      noop_function_ = nullptr;
      store_output_function_ = nullptr;
    }
    // Keep the process-global runtime instance loaded across test cases. The
    // driver services it owns outlive an individual stream and are not
    // reinitializable after the final dlclose within the same process.
  }

  // Runtime entry points loaded once from the HIP shared object under test.
  static HipRuntimeApi api_;
  // Stream supplied to immediate launch entry points.
  hipStream_t stream_ = nullptr;
  // Device whose architecture selected the embedded module image.
  int device_ = -1;
  // Module kept live while its function handles are under test.
  hipModule_t module_ = nullptr;
  // Zero-argument function used by configuration-only assertions.
  hipFunction_t noop_function_ = nullptr;
  // Two-argument function used to validate short prepacked spans.
  hipFunction_t store_output_function_ = nullptr;
};

HipRuntimeApi HipLaunchValidationApiTest::api_;

TEST_F(HipLaunchValidationApiTest,
       FunctionDynamicSharedMemoryAttributeHonorsReportedCeiling) {
  int default_capacity = 0;
  int optin_capacity = 0;
  int fixed_size = 0;
  int configured_size = 0;
  ASSERT_EQ(hipSuccess,
            api_.device_get_attribute(&default_capacity,
                                      hipDeviceAttributeMaxSharedMemoryPerBlock,
                                      device_));
  ASSERT_EQ(hipSuccess, api_.device_get_attribute(
                            &optin_capacity,
                            hipDeviceAttributeSharedMemPerBlockOptin, device_));
  ASSERT_EQ(hipSuccess,
            api_.function_get_attribute(
                &fixed_size, hipFuncAttributeSharedSizeBytes, noop_function_));
  ASSERT_EQ(hipSuccess,
            api_.function_get_attribute(
                &configured_size, hipFuncAttributeMaxDynamicSharedSizeBytes,
                noop_function_));
  const int configurable_capacity =
      optin_capacity != 0 ? optin_capacity : default_capacity;
  ASSERT_GE(configurable_capacity, fixed_size);
  ASSERT_GE(default_capacity, 0);
  const int maximum_configurable_size = configurable_capacity - fixed_size;
  const int expected_configured_size =
      default_capacity > fixed_size ? default_capacity - fixed_size : 0;
  EXPECT_EQ(expected_configured_size, configured_size);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.function_set_attribute(
                noop_function_, hipFuncAttributeMaxDynamicSharedSizeBytes, -1));
  if (maximum_configurable_size < std::numeric_limits<int>::max()) {
    EXPECT_EQ(hipErrorInvalidValue,
              api_.function_set_attribute(
                  noop_function_, hipFuncAttributeMaxDynamicSharedSizeBytes,
                  maximum_configurable_size + 1));
  }
  EXPECT_EQ(hipSuccess,
            api_.function_set_attribute(
                noop_function_, hipFuncAttributeMaxDynamicSharedSizeBytes,
                maximum_configurable_size));
  EXPECT_EQ(hipSuccess,
            api_.function_get_attribute(
                &configured_size, hipFuncAttributeMaxDynamicSharedSizeBytes,
                noop_function_));
  EXPECT_EQ(maximum_configurable_size, configured_size);
}

TEST_F(HipLaunchValidationApiTest,
       LaunchEntryPointsRejectInvalidConfiguration) {
  const void* function = reinterpret_cast<const void*>(noop_function_);
  const dim3 invalid_grid = {0, 1, 1};
  const dim3 valid_dimension = {1, 1, 1};

  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel(function, invalid_grid, valid_dimension,
                               /*arguments=*/nullptr,
                               /*shared_memory_bytes=*/0, stream_));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.ext_launch_kernel(function, invalid_grid, valid_dimension,
                                   /*arguments=*/nullptr,
                                   /*shared_memory_bytes=*/0, stream_, nullptr,
                                   nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.module_launch_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/0, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0, stream_,
                /*arguments=*/nullptr, /*extra=*/nullptr));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipKernelNodeParams valid_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/nullptr,
      /*.func=*/const_cast<void*>(function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &valid_params));

  hipKernelNodeParams invalid_params = valid_params;
  invalid_params.gridDim = invalid_grid;
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.graph_kernel_node_set_params(node, &invalid_params));

  hipKernelNodeParams retained_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_kernel_node_get_params(node, &retained_params));
  EXPECT_EQ(valid_params.func, retained_params.func);
  EXPECT_EQ(valid_params.gridDim.x, retained_params.gridDim.x);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.graph_add_kernel_node(&rejected_node, graph,
                                 /*dependencies=*/nullptr,
                                 /*dependency_count=*/0, &invalid_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipLaunchValidationApiTest, LaunchEntryPointsRejectDestroyedStreams) {
  const void* function = reinterpret_cast<const void*>(noop_function_);
  const dim3 valid_dimension = {1, 1, 1};
  hipStream_t stale_stream = stream_;
  ASSERT_EQ(hipSuccess, api_.stream_destroy(stream_));
  stream_ = nullptr;

  EXPECT_EQ(hipErrorInvalidResourceHandle,
            api_.launch_kernel(function, valid_dimension, valid_dimension,
                               /*arguments=*/nullptr,
                               /*shared_memory_bytes=*/0, stale_stream));
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            api_.ext_launch_kernel(function, valid_dimension, valid_dimension,
                                   /*arguments=*/nullptr,
                                   /*shared_memory_bytes=*/0, stale_stream,
                                   nullptr, nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            api_.module_launch_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0, stale_stream,
                /*arguments=*/nullptr, /*extra=*/nullptr));
}

TEST_F(HipLaunchValidationApiTest,
       ConcurrentQueriesObserveStreamDestructionWithoutDereferencingIt) {
  hipStream_t stream = stream_;
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> observed_zero_id{false};
  std::atomic<hipError_t> final_result{hipSuccess};
  std::thread query_thread([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!stop.load(std::memory_order_acquire)) {
      unsigned long long stream_id = 0;
      const hipError_t result = api_.stream_get_id(stream, &stream_id);
      if (result != hipSuccess) {
        final_result.store(result, std::memory_order_release);
        return;
      }
      if (stream_id == 0) {
        observed_zero_id.store(true, std::memory_order_release);
      }
    }
  });

  start.store(true, std::memory_order_release);
  const hipError_t destroy_result = api_.stream_destroy(stream_);
  if (destroy_result == hipSuccess) {
    stream_ = nullptr;
  } else {
    stop.store(true, std::memory_order_release);
  }
  query_thread.join();
  EXPECT_EQ(hipSuccess, destroy_result);
  EXPECT_FALSE(observed_zero_id.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            final_result.load(std::memory_order_acquire));
}

TEST_F(HipLaunchValidationApiTest,
       LaunchEntryPointsRejectOutOfRangeSharedMemory) {
  if (sizeof(size_t) <= sizeof(uint32_t)) {
    GTEST_SKIP() << "size_t cannot represent a value above uint32_t";
  }

  const void* function = reinterpret_cast<const void*>(noop_function_);
  const dim3 valid_dimension = {1, 1, 1};
  const size_t largest_dispatch_shared_memory = UINT32_MAX;
  const size_t oversized_shared_memory = (size_t)UINT32_MAX + 1;

  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel(function, valid_dimension, valid_dimension,
                               /*arguments=*/nullptr,
                               largest_dispatch_shared_memory, stream_));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.ext_launch_kernel(function, valid_dimension, valid_dimension,
                                   /*arguments=*/nullptr,
                                   largest_dispatch_shared_memory, stream_,
                                   nullptr, nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel(function, valid_dimension, valid_dimension,
                               /*arguments=*/nullptr, oversized_shared_memory,
                               stream_));
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.ext_launch_kernel(function, valid_dimension, valid_dimension,
                             /*arguments=*/nullptr, oversized_shared_memory,
                             stream_, nullptr, nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.module_launch_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, UINT32_MAX, stream_, /*arguments=*/nullptr,
                /*extra=*/nullptr));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipKernelNodeParams valid_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/nullptr,
      /*.func=*/const_cast<void*>(function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &valid_params));

  hipKernelNodeParams rejected_params = valid_params;
  rejected_params.sharedMemBytes = largest_dispatch_shared_memory;
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.graph_kernel_node_set_params(node, &rejected_params));
  hipKernelNodeParams retained_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_kernel_node_get_params(node, &retained_params));
  EXPECT_EQ(0u, retained_params.sharedMemBytes);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.graph_add_kernel_node(&rejected_node, graph,
                                 /*dependencies=*/nullptr,
                                 /*dependency_count=*/0, &rejected_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);

  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipLaunchValidationApiTest,
       PrepackedGraphArgumentsRejectShortSpansWithoutMutatingTheNode) {
  const void* empty_function = reinterpret_cast<const void*>(noop_function_);
  const void* prepacked_function =
      reinterpret_cast<const void*>(store_output_function_);
  const dim3 valid_dimension = {1, 1, 1};

  uint8_t argument_storage[16] = {};
  size_t short_argument_size = 0;
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      argument_storage,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &short_argument_size,
      HIP_LAUNCH_PARAM_END,
  };
  hipKernelNodeParams empty_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/nullptr,
      /*.func=*/const_cast<void*>(empty_function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipKernelNodeParams short_prepacked_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/extra,
      /*.func=*/const_cast<void*>(prepacked_function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };

  EXPECT_EQ(
      hipErrorInvalidValue,
      api_.module_launch_kernel(
          (hipFunction_t)prepacked_function, /*grid_dim_x=*/1,
          /*grid_dim_y=*/1, /*grid_dim_z=*/1, /*block_dim_x=*/1,
          /*block_dim_y=*/1, /*block_dim_z=*/1,
          /*shared_memory_bytes=*/0, stream_, /*arguments=*/nullptr, extra));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &empty_params));

  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_kernel_node_set_params(node, &short_prepacked_params));
  hipKernelNodeParams retained_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_kernel_node_get_params(node, &retained_params));
  EXPECT_EQ(empty_params.func, retained_params.func);
  EXPECT_EQ(empty_params.extra, retained_params.extra);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_add_kernel_node(&rejected_node, graph,
                                       /*dependencies=*/nullptr,
                                       /*dependency_count=*/0,
                                       &short_prepacked_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

}  // namespace
