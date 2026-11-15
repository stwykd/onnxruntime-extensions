// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fails one chosen allocation inside a C API call, to check that running out of memory comes back as an
// error code and never as an exception or std::terminate. This replaces the global allocation functions,
// so it's its own test executable; the library is linked statically, so its allocations come here too.

#include <cstdlib>
#include <new>
#include <string>

#include "gtest/gtest.h"
#include "ortx_processor.h"
#include "shared/api/c_api_utils.hpp"

namespace {
thread_local long allocations_until_failure = -1;  // negative: never fail
thread_local bool failed_allocation = false;

// Fails allocation number `index` (0 based) of the next calls on this thread.
void FailAllocation(long index) {
  failed_allocation = false;
  allocations_until_failure = index;
}

// Stops failing allocations; returns whether one was failed.
bool StopFailing() {
  allocations_until_failure = -1;
  return failed_allocation;
}
}  // namespace

void* operator new(std::size_t size) {
  if (allocations_until_failure >= 0 && allocations_until_failure-- == 0) {
    failed_allocation = true;
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

TEST(ImagePixelsAllocTest, OutOfMemoryIsReturnedAsAnError) {
  const uint8_t pixel[] = {1, 2, 3};
  const void* data[] = {pixel};
  const uint32_t side = 1;
  const size_t stride = 3;
  for (const size_t size : {size_t{3}, size_t{2}}) {  // a valid pixel, then a buffer one byte short
    const extError_t expected = size == 3 ? kOrtxOK : kOrtxErrorInvalidArgument;
    // Fail each allocation the call makes in turn, until a call completes without hitting one.
    for (long index = 0;; ++index) {
      ASSERT_LT(index, 100);
      OrtxRawImages* images = reinterpret_cast<OrtxRawImages*>(1);
      FailAllocation(index);
      const extError_t status =
          OrtxCreateImagesFromPixels(&images, data, &size, &side, &side, &stride, OrtxPixelFormat_RGB8, 1);
      const bool failed = StopFailing();
      if (!failed) {
        EXPECT_GT(index, 0);  // the call allocates, so earlier rounds did fail one
        EXPECT_EQ(status, expected);
        EXPECT_EQ(images != nullptr, expected == kOrtxOK);
        if (images != nullptr) OrtxDisposeOnly(images);
        break;
      }
      // An invalid image may already have its error when the allocation fails; otherwise it's out of memory.
      EXPECT_TRUE(status == kOrtxErrorOutOfMemory || status == expected) << "allocation " << index;
      EXPECT_NE(status, kOrtxOK) << "allocation " << index;
      EXPECT_EQ(images, nullptr) << "allocation " << index;
      EXPECT_NE(OrtxGetLastErrorMessage()[0], '\0') << "allocation " << index;
    }
  }
}

TEST(ImagePixelsAllocTest, ErrorMessageFallsBackWhenOutOfMemory) {
  // Start from an empty buffer, so publishing a long message has to allocate.
  std::string().swap(ort_extensions::ReturnableStatus::last_error_message_);
  {
    ort_extensions::ReturnableStatus status(OrtxStatus(kOrtxErrorInvalidArgument, std::string(100, 'x')));
    FailAllocation(0);
  }  // the destructor publishes the message
  EXPECT_TRUE(StopFailing());
  EXPECT_STREQ(OrtxGetLastErrorMessage(), "Out of memory");
}
