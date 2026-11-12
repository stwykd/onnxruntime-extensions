// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_processor.h"

using namespace ort_extensions;

namespace {
void CheckResult(extError_t status) {
  if (status != kOrtxOK) throw std::runtime_error(OrtxGetLastErrorMessage());
}

const char* kDecode = R"({"processor":{"transforms":[{"operation":{"name":"decode","type":"DecodeImage"}}]}})";

std::vector<uint8_t> Process(OrtxRawImages* images, const char* config = kDecode) {
  OrtxObjectPtr<OrtxProcessor> processor;
  CheckResult(OrtxCreateProcessor(processor.ToBeAssigned(), config));
  OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor.get(), images, result.ToBeAssigned()));
  OrtxObjectPtr<OrtxTensor> tensor;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned()));
  const void* data{};
  const int64_t* shape{};
  size_t rank{};
  CheckResult(OrtxGetTensorData(tensor.get(), &data, &shape, &rank));
  size_t count = 1;
  for (size_t i = 0; i < rank; ++i) count *= shape[i];
  const auto* pixels = static_cast<const uint8_t*>(data);
  return {pixels, pixels + count};
}
}  // namespace

TEST(RawPixelsTest, LayoutStrideOwnershipAndAlpha) {
  const std::vector<uint8_t> expected{1, 2, 3, 10, 20, 30, 40, 50, 60, 100, 150, 200};
  for (auto format : {kOrtxPixelFormatRGB8, kOrtxPixelFormatRGBA8, kOrtxPixelFormatBGRA8}) {
    const size_t channels = format == kOrtxPixelFormatRGB8 ? 3 : 4;
    const size_t stride = 2 * channels + 5;
    std::vector<uint8_t> bytes(stride + 2 * channels, 222);
    for (size_t i = 0; i < 4; ++i) {
      auto* pixel = bytes.data() + (i / 2) * stride + (i % 2) * channels;
      for (size_t c = 0; c < 3; ++c)
        pixel[format == kOrtxPixelFormatBGRA8 ? 2 - c : c] = expected[i * 3 + c];
      if (channels == 4) pixel[3] = static_cast<uint8_t>(i * 85);
    }
    OrtxRawImage descriptor{bytes.data(), bytes.size(), 2, 2, stride, format, kOrtxAlphaStraight};
    OrtxObjectPtr<OrtxRawImages> images;
    ASSERT_EQ(OrtxCreateImagesFromRawPixels(images.ToBeAssigned(), &descriptor, 1), kOrtxOK);
    std::fill(bytes.begin(), bytes.end(), 0);
    EXPECT_EQ(Process(images.get()), expected);
    EXPECT_EQ(Process(images.get()), expected);
    auto bgr = expected;
    for (size_t i = 0; i < bgr.size(); i += 3) std::swap(bgr[i], bgr[i + 2]);
    EXPECT_EQ(Process(images.get(), R"({"processor":{"transforms":[{"operation":{
      "name":"decode","type":"DecodeImage","attrs":{"color_space":"BGR"}}}]}})"), bgr);
  }
}

TEST(RawPixelsTest, PremultipliedAndOpaqueAlpha) {
  const std::array<uint8_t, 16> rgba{17, 18, 19, 0, 64, 32, 0, 128, 1, 0, 1, 1, 255, 128, 1, 255};
  OrtxRawImage descriptor{rgba.data(), rgba.size(), 4, 1, rgba.size(), kOrtxPixelFormatRGBA8, kOrtxAlphaPremultiplied};
  OrtxObjectPtr<OrtxRawImages> images;
  ASSERT_EQ(OrtxCreateImagesFromRawPixels(images.ToBeAssigned(), &descriptor, 1), kOrtxOK);
  EXPECT_EQ(Process(images.get()), (std::vector<uint8_t>{0, 0, 0, 128, 64, 0, 255, 0, 255, 255, 128, 1}));
  descriptor.alpha_mode = kOrtxAlphaOpaque;
  OrtxObjectPtr<OrtxRawImages> opaque;
  ASSERT_EQ(OrtxCreateImagesFromRawPixels(opaque.ToBeAssigned(), &descriptor, 1), kOrtxOK);
  EXPECT_EQ(Process(opaque.get()), (std::vector<uint8_t>{17, 18, 19, 64, 32, 0, 1, 0, 1, 255, 128, 1}));
}

TEST(RawPixelsTest, BatchAndPipelineValidation) {
  const std::array<uint8_t, 3> pixel{1, 2, 3};
  const OrtxRawImage descriptor{pixel.data(), 3, 1, 1, 3, kOrtxPixelFormatRGB8, kOrtxAlphaOpaque};
  const std::array descriptors{descriptor, descriptor};
  OrtxObjectPtr<OrtxRawImages> images;
  ASSERT_EQ(OrtxCreateImagesFromRawPixels(images.ToBeAssigned(), descriptors.data(), 2), kOrtxOK);
  EXPECT_EQ(Process(images.get()), (std::vector<uint8_t>{1, 2, 3, 1, 2, 3}));
  OrtxObjectPtr<OrtxProcessor> processor;
  ASSERT_EQ(OrtxCreateProcessor(processor.ToBeAssigned(),
    R"({"processor":{"transforms":[{"operation":{"name":"rgb","type":"ConvertRGB"}}]}})"), kOrtxOK);
  OrtxTensorResult* result{};
  EXPECT_EQ(OrtxImagePreProcess(processor.get(), images.get(), &result), kOrtxErrorInvalidArgument);
  EXPECT_EQ(result, nullptr);
}

TEST(RawPixelsTest, InvalidDescriptorsFailBeforeReadingPixels) {
  const std::array<uint8_t, 3> pixel{1, 2, 3};
  const OrtxRawImage valid{pixel.data(), 3, 1, 1, 3, kOrtxPixelFormatRGB8, kOrtxAlphaOpaque};
  OrtxRawImages* images{};
  EXPECT_EQ(OrtxCreateImagesFromRawPixels(nullptr, &valid, 1), kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxCreateImagesFromRawPixels(&images, nullptr, 1), kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxCreateImagesFromRawPixels(&images, &valid, 0), kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxCreateImagesFromRawPixels(&images, &valid, std::numeric_limits<size_t>::max()), kOrtxErrorInvalidArgument);
  std::vector<OrtxRawImage> invalid(10, valid);
  invalid[0].data = nullptr;
  invalid[1].width = 0;
  invalid[2].height = 16385;
  invalid[3].width = invalid[3].height = 10001;
  invalid[4].row_stride = 2;
  invalid[5].data_size = 2;
  invalid[6].format = static_cast<OrtxPixelFormat>(99);
  invalid[7].alpha_mode = static_cast<OrtxAlphaMode>(99);
  invalid[8].alpha_mode = kOrtxAlphaPremultiplied;
  invalid[9].height = 3;
  invalid[9].row_stride = std::numeric_limits<size_t>::max();
  for (const auto& descriptor : invalid) {
    images = reinterpret_cast<OrtxRawImages*>(1);
    EXPECT_EQ(OrtxCreateImagesFromRawPixels(&images, &descriptor, 1), kOrtxErrorInvalidArgument);
    EXPECT_EQ(images, nullptr);
  }
  auto large = valid;
  large.width = large.height = 10000;
  large.row_stride = 30000;
  large.data_size = 300000000;
  const std::array batch{large, large};
  EXPECT_EQ(OrtxCreateImagesFromRawPixels(&images, batch.data(), 2), kOrtxErrorInvalidArgument);
  EXPECT_EQ(images, nullptr);
}
