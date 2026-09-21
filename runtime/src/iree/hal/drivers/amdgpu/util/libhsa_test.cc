// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#include <algorithm>
#include <cstdint>
#include <thread>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
#if defined(IREE_HAL_AMDGPU_LIBHSA_TEST_INSTRUMENTATION)
struct AddressFreeCallbackState {
  iree_hal_amdgpu_libhsa_t libhsa = {};
  int callback_depth = 0;
  int max_callback_depth = 0;
  int enter_count = 0;
  int leave_count = 0;
  int raw_call_count = 0;
  bool query_was_active = true;
  bool nest_once = false;
  bool reject_clear_once = false;
  iree_status_code_t clear_inside_status = IREE_STATUS_OK;
  bool enabled = false;
};

static AddressFreeCallbackState address_free_callback_state;

class ScopedLibHSADeinitialize {
 public:
  explicit ScopedLibHSADeinitialize(iree_hal_amdgpu_libhsa_t* libhsa)
      : libhsa_(libhsa) {}
  ScopedLibHSADeinitialize(const ScopedLibHSADeinitialize&) = delete;
  ScopedLibHSADeinitialize& operator=(const ScopedLibHSADeinitialize&) = delete;
  ~ScopedLibHSADeinitialize() { iree_hal_amdgpu_libhsa_deinitialize(libhsa_); }

 private:
  iree_hal_amdgpu_libhsa_t* libhsa_;
};

class ScopedAddressFreeObserver {
 public:
  ScopedAddressFreeObserver(
      iree_hal_amdgpu_libhsa_vmem_address_free_observer_t observer,
      void* user_data)
      : observer_(observer), user_data_(user_data) {
    iree_status_t status =
        iree_hal_amdgpu_libhsa_set_vmem_address_free_observer(observer_,
                                                              user_data_);
    IREE_ASSERT(iree_status_is_ok(status));
    iree_status_ignore(status);
  }
  ScopedAddressFreeObserver(const ScopedAddressFreeObserver&) = delete;
  ScopedAddressFreeObserver& operator=(const ScopedAddressFreeObserver&) =
      delete;
  ~ScopedAddressFreeObserver() {
    iree_status_t status =
        iree_hal_amdgpu_libhsa_clear_vmem_address_free_observer(observer_,
                                                                user_data_);
    IREE_ASSERT(iree_status_is_ok(status));
    iree_status_ignore(status);
  }

 private:
  iree_hal_amdgpu_libhsa_vmem_address_free_observer_t observer_;
  void* user_data_;
};

static hsa_status_t HSA_API FakeAddressFree(void* address, size_t size) {
  (void)address;
  (void)size;
  ++address_free_callback_state.raw_call_count;
  address_free_callback_state.query_was_active &=
      iree_hal_amdgpu_libhsa_vmem_address_free_callback_window_is_active();
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

static void ObserveAddressFree(bool entering, void* address, size_t size,
                               void* user_data) {
  auto& state = *static_cast<AddressFreeCallbackState*>(user_data);
  if (!state.enabled) return;
  state.query_was_active &=
      iree_hal_amdgpu_libhsa_vmem_address_free_callback_window_is_active();
  if (entering) {
    ++state.enter_count;
    ++state.callback_depth;
    if (state.reject_clear_once) {
      state.reject_clear_once = false;
      iree_status_t status =
          iree_hal_amdgpu_libhsa_clear_vmem_address_free_observer(
              ObserveAddressFree, &state);
      state.clear_inside_status = iree_status_code(status);
      iree_status_free(status);
    }
    state.max_callback_depth =
        std::max(state.max_callback_depth, state.callback_depth);
    if (state.nest_once) {
      state.nest_once = false;
      EXPECT_EQ(
          HSA_STATUS_ERROR_INVALID_ARGUMENT,
          iree_hsa_amd_vmem_address_free_raw(&state.libhsa, address, size));
    }
  } else {
    ++state.leave_count;
    EXPECT_GT(state.callback_depth, 0);
    --state.callback_depth;
  }
}

TEST(LibHSATest,
     ProcessAddressFreeObserverObservesWorkerAndBalancesNestedFailure) {
  iree_hal_amdgpu_libhsa_t initialized_libhsa;
  iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
      iree_allocator_system(), &initialized_libhsa);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    GTEST_SKIP() << "HSA not available, skipping tests";
  }
  ScopedLibHSADeinitialize initialized_libhsa_scope(&initialized_libhsa);

  AddressFreeCallbackState& state = address_free_callback_state;
  state = AddressFreeCallbackState{};
  state.libhsa.hsa_amd_vmem_address_free = FakeAddressFree;
  state.enabled = true;
  state.nest_once = true;
  state.reject_clear_once = true;
  {
    ScopedAddressFreeObserver observer(ObserveAddressFree, &state);
    hsa_status_t worker_status = HSA_STATUS_SUCCESS;
    std::thread worker([&]() {
      worker_status = iree_hsa_amd_vmem_address_free_raw(
          &state.libhsa, reinterpret_cast<void*>(uintptr_t{0x10000}),
          /*size=*/4096);
    });
    worker.join();
    EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, worker_status);

    EXPECT_EQ(2, state.raw_call_count);
    EXPECT_EQ(2, state.enter_count);
    EXPECT_EQ(2, state.leave_count);
    EXPECT_EQ(2, state.max_callback_depth);
    EXPECT_EQ(0, state.callback_depth);
    EXPECT_TRUE(state.query_was_active);
    EXPECT_EQ(IREE_STATUS_FAILED_PRECONDITION, state.clear_inside_status);
    EXPECT_FALSE(
        iree_hal_amdgpu_libhsa_vmem_address_free_callback_window_is_active());
  }
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT,
            iree_hsa_amd_vmem_address_free_raw(
                &state.libhsa, reinterpret_cast<void*>(uintptr_t{0x20000}),
                /*size=*/4096));
  EXPECT_EQ(3, state.raw_call_count);
  EXPECT_EQ(2, state.enter_count);
  EXPECT_EQ(2, state.leave_count);
  state.enabled = false;
}
#endif  // IREE_HAL_AMDGPU_LIBHSA_TEST_INSTRUMENTATION

static bool callback_window_query_was_active = false;

static hsa_status_t HSA_API QueryAddressFreeCallbackWindow(void* address,
                                                           size_t size) {
  (void)address;
  (void)size;
  callback_window_query_was_active =
      iree_hal_amdgpu_libhsa_vmem_address_free_callback_window_is_active();
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

TEST(LibHSATest, AddressFreeCallbackWindowIsAvailableInProduction) {
  iree_hal_amdgpu_libhsa_t libhsa = {};
  libhsa.hsa_amd_vmem_address_free = QueryAddressFreeCallbackWindow;
  callback_window_query_was_active = false;
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT,
            iree_hsa_amd_vmem_address_free_raw(
                &libhsa, reinterpret_cast<void*>(uintptr_t{0x10000}),
                /*size=*/4096));
  EXPECT_TRUE(callback_window_query_was_active);
  EXPECT_FALSE(
      iree_hal_amdgpu_libhsa_vmem_address_free_callback_window_is_active());
}
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

// Tests that we can find, load, and unload HSA.
// In ASAN builds it tests that we don't leak the library (though ROCR itself
// leaks a bunch). If the library cannot be found then we skip the test so that
// it doesn't fail on machines without HSA installed.
TEST(LibHSATest, Load) {
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
      iree_allocator_system(), &libhsa);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    GTEST_SKIP() << "HSA not available, skipping tests";
  }

  // Ensure resolved symbols are callable without perturbing the HSA runtime
  // lifetime beyond the one owned by libhsa.
  uint16_t version_major = 0;
  IREE_ASSERT_OK(iree_hsa_system_get_info(
      IREE_LIBHSA(&libhsa), HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major));
  EXPECT_NE(version_major, 0u);

  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
}

}  // namespace
}  // namespace iree::hal::amdgpu
