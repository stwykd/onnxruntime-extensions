// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_processor.h"
#include "c_only_test.h"

using namespace ort_extensions;

namespace {
void CheckResult(extError_t status) {
  if (status != kOrtxOK) throw std::runtime_error(OrtxGetLastErrorMessage());
}

const char* kDecode = R"({"processor":{"transforms":[{"operation":{"name":"decode","type":"DecodeImage"}}]}})";

struct Output {
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  bool operator==(const Output& other) const { return shape == other.shape && bytes == other.bytes; }
};

// Every output tensor of the processor, as shape and raw bytes.
std::vector<Output> Preprocess(OrtxRawImages* images, const char* config) {
  OrtxObjectPtr<OrtxProcessor> processor;
  CheckResult(OrtxCreateProcessor(processor.ToBeAssigned(), config));
  OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor.get(), images, result.ToBeAssigned()));
  std::vector<Output> outputs;
  for (size_t i = 0;; ++i) {
    OrtxObjectPtr<OrtxTensor> tensor;
    if (OrtxTensorResultGetAt(result.get(), i, tensor.ToBeAssigned()) != kOrtxOK) break;
    const void* data{};
    const int64_t* shape{};
    size_t rank{};
    size_t element_size{};
    CheckResult(OrtxGetTensorData(tensor.get(), &data, &shape, &rank));
    CheckResult(OrtxGetTensorSizeOfElement(tensor.get(), &element_size));
    Output output{{shape, shape + rank}, {}};
    size_t count = element_size;
    for (auto dim : output.shape) count *= static_cast<size_t>(dim);
    output.bytes.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + count);
    outputs.push_back(std::move(output));
  }
  return outputs;
}

std::vector<uint8_t> Decode(OrtxRawImages* images, const char* config = kDecode) {
  return Preprocess(images, config).at(0).bytes;
}

const char* const kFormats[] = {"RGB8", "RGBA8", "BGRA8"};

size_t Channels(std::string_view format) { return format == "RGB8" ? 3 : 4; }

// Writes RGB pixels in the given format into rows of `stride` bytes; alpha varies, including 0.
std::vector<uint8_t> Pack(const std::vector<uint8_t>& rgb, size_t width, size_t height, std::string_view format,
                          size_t stride) {
  const size_t channels = Channels(format);
  std::vector<uint8_t> packed((height - 1) * stride + width * channels, 0xEE);
  for (size_t i = 0; i < width * height; ++i) {
    auto* pixel = packed.data() + (i / width) * stride + (i % width) * channels;
    for (size_t c = 0; c < 3; ++c) pixel[format == "BGRA8" ? 2 - c : c] = rgb[i * 3 + c];
    if (channels == 4) pixel[3] = static_cast<uint8_t>(i * 37);
  }
  return packed;
}

OrtxObjectPtr<OrtxRawImages> FromPixels(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height,
                                        size_t stride, const char* format) {
  const void* data[] = {pixels.data()};
  const size_t size = pixels.size();
  OrtxObjectPtr<OrtxRawImages> images;
  CheckResult(OrtxCreateImagesFromPixels(images.ToBeAssigned(), data, &size, &width, &height, &stride, format, 1));
  return images;
}
}  // namespace

TEST(ImagePixelsTest, FormatsStrideAlphaAndOwnership) {
  const std::vector<uint8_t> rgb{1, 2, 3, 10, 20, 30, 40, 50, 60, 100, 150, 200, 7, 8, 9, 250, 251, 252};
  auto bgr = rgb;
  for (size_t i = 0; i < bgr.size(); i += 3) std::swap(bgr[i], bgr[i + 2]);
  for (const char* format : kFormats) {
    const size_t stride = 3 * Channels(format) + 5;
    auto pixels = Pack(rgb, 3, 2, format, stride);
    auto images = FromPixels(pixels, 3, 2, stride, format);
    std::fill(pixels.begin(), pixels.end(), 0);  // the batch owns a copy
    EXPECT_EQ(Decode(images.get()), rgb);
    EXPECT_EQ(Decode(images.get()), rgb);  // processing doesn't consume or modify it
    EXPECT_EQ(Decode(images.get(), R"({"processor":{"transforms":[{"operation":{
      "name":"decode","type":"DecodeImage","attrs":{"color_space":"BGR"}}}]}})"),
              bgr);
  }

  // The header compiles as C.
  auto pixels = Pack(rgb, 3, 2, "BGRA8", 12);
  OrtxObjectPtr<OrtxRawImages> images;
  ASSERT_EQ(create_images_from_bgra(pixels.data(), pixels.size(), 3, 2, 12, images.ToBeAssigned()), kOrtxOK);
  EXPECT_EQ(Decode(images.get()), rgb);
}

TEST(ImagePixelsTest, MatchesEncodedImagesThroughProcessors) {
  const char* paths[] = {"data/processor/exceltable.png", "data/processor/australia.jpg"};
  for (const char* path : paths) {
    OrtxObjectPtr<OrtxRawImages> encoded;
    CheckResult(OrtxLoadImages(encoded.ToBeAssigned(), &path, 1, nullptr));
    const auto decoded = Preprocess(encoded.get(), kDecode).at(0);
    ASSERT_EQ(decoded.shape.size(), 4u);
    const auto height = static_cast<uint32_t>(decoded.shape[1]);
    const auto width = static_cast<uint32_t>(decoded.shape[2]);
    for (const char* config : {"data/processor/clip_image.json", "data/processor/pad_to_square.json"}) {
      const auto expected = Preprocess(encoded.get(), config);
      for (const char* format : kFormats) {
        const size_t stride = width * Channels(format) + 3;
        auto images = FromPixels(Pack(decoded.bytes, width, height, format, stride), width, height, stride, format);
        EXPECT_EQ(Preprocess(images.get(), config), expected) << path << " " << config << " format " << format;
      }
    }
  }
}

TEST(ImagePixelsTest, BatchOrderAndWorkflowWithoutDecode) {
  const std::vector<uint8_t> first{1, 2, 3, 4, 5, 6};
  const std::vector<uint8_t> second{7, 8, 9, 0, 10, 11, 12};
  const void* data[] = {first.data(), second.data()};
  const size_t sizes[] = {first.size(), second.size()};
  const uint32_t widths[] = {2, 2};
  const uint32_t heights[] = {1, 1};
  const size_t strides[] = {6, 6};
  OrtxObjectPtr<OrtxRawImages> images;
  ASSERT_EQ(OrtxCreateImagesFromPixels(images.ToBeAssigned(), data, sizes, widths, heights, strides, "RGB8", 2),
            kOrtxOK);
  const auto outputs = Preprocess(images.get(), kDecode);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{2, 1, 2, 3}));
  EXPECT_EQ(outputs[0].bytes, (std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11}));

  OrtxObjectPtr<OrtxProcessor> processor;
  ASSERT_EQ(OrtxCreateProcessor(processor.ToBeAssigned(),
                                R"({"processor":{"transforms":[{"operation":{"name":"rgb","type":"ConvertRGB"}}]}})"),
            kOrtxOK);
  OrtxTensorResult* result{};
  EXPECT_EQ(OrtxImagePreProcess(processor.get(), images.get(), &result), kOrtxErrorInvalidArgument);
  EXPECT_EQ(result, nullptr);
}

TEST(ImagePixelsTest, InvalidInputFailsWithoutImages) {
  const std::array<uint8_t, 4> pixel{1, 2, 3, 4};
  const void* data[] = {pixel.data()};
  size_t size = 3;
  uint32_t width = 1;
  uint32_t height = 1;
  size_t stride = 3;
  auto call = [&](const char* format = "RGB8", size_t count = 1) {
    OrtxRawImages* images = reinterpret_cast<OrtxRawImages*>(1);
    const auto status = OrtxCreateImagesFromPixels(&images, data, &size, &width, &height, &stride, format, count);
    if (status == kOrtxOK) {
      OrtxDisposeOnly(images);
    } else {
      EXPECT_EQ(images, nullptr);
    }
    return status;
  };
  ASSERT_EQ(call(), kOrtxOK);  // the baseline is valid; each case below breaks one thing
  OrtxRawImages* images{};
  EXPECT_EQ(OrtxCreateImagesFromPixels(nullptr, data, &size, &width, &height, &stride, "RGB8", 1),
            kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxCreateImagesFromPixels(&images, nullptr, &size, &width, &height, &stride, "RGB8", 1),
            kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxCreateImagesFromPixels(&images, data, &size, &width, nullptr, &stride, "RGB8", 1),
            kOrtxErrorInvalidArgument);
  EXPECT_EQ(call("RGB8", 0), kOrtxErrorInvalidArgument);
  EXPECT_EQ(call("RGB8", std::numeric_limits<size_t>::max()), kOrtxErrorInvalidArgument);
  for (const char* format : {static_cast<const char*>(nullptr), "", "BGR8", "rgb8", "RGB8 ", "RGBA"}) {
    EXPECT_EQ(call(format), kOrtxErrorInvalidArgument) << (format ? format : "null");
    EXPECT_NE(std::strstr(OrtxGetLastErrorMessage(), "pixel_format"), nullptr);
  }
  EXPECT_EQ(call("RGBA8"), kOrtxErrorInvalidArgument);  // 3 bytes can't hold an RGBA pixel
  EXPECT_NE(std::strstr(OrtxGetLastErrorMessage(), "row stride"), nullptr);

  data[0] = nullptr;
  EXPECT_EQ(call(), kOrtxErrorInvalidArgument);
  EXPECT_NE(std::strstr(OrtxGetLastErrorMessage(), "null"), nullptr);
  data[0] = pixel.data();

  const std::array<std::pair<uint32_t, uint32_t>, 5> bad_sizes{
      {{0, 1}, {1, 0}, {16385, 1}, {1, 16385}, {10001, 10000}}};
  for (auto [w, h] : bad_sizes) {
    width = w;
    height = h;
    stride = size_t{w} * 3;
    size = std::numeric_limits<size_t>::max();
    EXPECT_EQ(call(), kOrtxErrorInvalidArgument) << w << "x" << h;
  }
  width = height = 1;
  size = stride = 3;

  stride = 2;
  EXPECT_EQ(call(), kOrtxErrorInvalidArgument);
  stride = 3;
  size = 2;
  EXPECT_EQ(call(), kOrtxErrorInvalidArgument);
  height = 3;
  size = std::numeric_limits<size_t>::max();
  stride = std::numeric_limits<size_t>::max() / 2;  // (height - 1) * stride + 3 overflows
  EXPECT_EQ(call(), kOrtxErrorInvalidArgument);
  height = 1;
  size = stride = 3;

  // Each image is within the limits, together they exceed 512 MiB of RGB.
  const void* batch_data[] = {pixel.data(), pixel.data()};
  const size_t batch_sizes[] = {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
  const uint32_t batch_sides[] = {10000, 10000};
  const size_t batch_strides[] = {30000, 30000};
  images = reinterpret_cast<OrtxRawImages*>(1);
  EXPECT_EQ(
      OrtxCreateImagesFromPixels(&images, batch_data, batch_sizes, batch_sides, batch_sides, batch_strides, "RGB8", 2),
      kOrtxErrorInvalidArgument);
  EXPECT_EQ(images, nullptr);
}
