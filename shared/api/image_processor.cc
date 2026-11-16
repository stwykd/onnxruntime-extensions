// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <string>
#include <string_view>

#include "nlohmann/json.hpp"
#include "file_sys.h"

#include "vision/decode_image.hpp"
#include "image_processor.h"
#include "c_api_utils.hpp"

#include "image_transforms.hpp"
#include "image_transforms_phi_3.hpp"
#include "image_transforms_mllama.hpp"
#include "image_transforms_phi_4.hpp"
#include "image_transforms_qwen2_5.hpp"
#include "image_transforms_gemma4.hpp"
#include "image_transforms_pixtral.hpp"

namespace ort_extensions {
std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(
    const std::initializer_list<const char*>& image_paths) {
  return ort_extensions::LoadRawData<const char* const*, ImageRawData>(image_paths.begin(), image_paths.end());
}

std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(const char* image_paths[], size_t num_images) {
  return ort_extensions::LoadRawData<const char* const*, ImageRawData>(image_paths, image_paths + num_images);
}

std::optional<PixelFormat> ParsePixelFormat(const char* name) noexcept {
  if (name == nullptr) {
    return std::nullopt;
  }
  const std::string_view format = name;
  if (format == "RGB8") {
    return PixelFormat::kRGB8;
  }
  if (format == "RGBA8") {
    return PixelFormat::kRGBA8;
  }
  if (format == "BGRA8") {
    return PixelFormat::kBGRA8;
  }
  return std::nullopt;
}

OrtxStatus LoadImagesFromPixels(const void* const data[], const size_t data_sizes[], const uint32_t widths[],
                                const uint32_t heights[], const size_t row_strides[], PixelFormat format,
                                size_t num_images, std::unique_ptr<ImageRawData[]>& images,
                                std::vector<ImageShape>& shapes) {
  constexpr size_t kMaxBatchBytes = size_t{512} * 1024 * 1024;
  if (!data || !data_sizes || !widths || !heights || !row_strides || num_images == 0) {
    return {kOrtxErrorInvalidArgument, "Images from pixels need non-null arrays and at least one image."};
  }
  if (num_images > kMaxBatchBytes / 3) {
    return {kOrtxErrorInvalidArgument, "Images from pixels exceed the 512 MiB batch limit."};
  }
  const size_t channels = format == PixelFormat::kRGB8 ? 3 : 4;

  size_t total_bytes = 0;
  for (size_t i = 0; i < num_images; ++i) {
    const auto error = [i](const char* message) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "Image " + std::to_string(i) + ": " + message);
    };
    const uint64_t width = widths[i];
    const uint64_t height = heights[i];
    if (!data[i]) {
      return error("pixel data is null.");
    }
    if (width == 0 || height == 0 || width > internal::kMaxImageDimension || height > internal::kMaxImageDimension ||
        width * height > internal::kMaxPixelCount) {
      return error("width and height must be 1 to 16384, with at most 100 million pixels.");
    }
    const size_t row_bytes = static_cast<size_t>(width) * channels;
    if (row_strides[i] < row_bytes) {
      return error("row stride is less than width * channels.");
    }
    if (height > 1 && row_strides[i] > (std::numeric_limits<size_t>::max() - row_bytes) / (height - 1)) {
      return error("row stride * height overflows.");
    }
    if (data_sizes[i] < static_cast<size_t>(height - 1) * row_strides[i] + row_bytes) {
      return error("data size is less than (height - 1) * row stride + width * channels.");
    }
    const size_t rgb_bytes = static_cast<size_t>(width * height * 3);
    if (rgb_bytes > kMaxBatchBytes - total_bytes) {
      return error("the batch exceeds 512 MiB of RGB pixels.");
    }
    total_bytes += rgb_bytes;
  }

  auto rgb_images = std::make_unique<ImageRawData[]>(num_images);
  std::vector<ImageShape> rgb_shapes;
  rgb_shapes.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    const size_t width = widths[i];
    const size_t height = heights[i];
    auto& rgb = rgb_images[i];
    rgb.resize(width * height * 3);
    rgb_shapes.push_back({static_cast<int64_t>(height), static_cast<int64_t>(width), 3});
    for (size_t y = 0; y < height; ++y) {
      const uint8_t* src = static_cast<const uint8_t*>(data[i]) + y * row_strides[i];
      uint8_t* dst = rgb.data() + y * width * 3;
      if (format == PixelFormat::kRGB8) {
        std::copy_n(src, width * 3, dst);
      } else if (format == PixelFormat::kRGBA8) {
        for (size_t x = 0; x < width; ++x, src += 4, dst += 3) {
          dst[0] = src[0];
          dst[1] = src[1];
          dst[2] = src[2];
        }
      } else {
        for (size_t x = 0; x < width; ++x, src += 4, dst += 3) {
          dst[0] = src[2];
          dst[1] = src[1];
          dst[2] = src[0];
        }
      }
    }
  }
  images = std::move(rgb_images);
  shapes = std::move(rgb_shapes);
  return {};
}
}  // namespace ort_extensions

using namespace ort_extensions;
using json = nlohmann::json;

namespace {
// The processor's decoded input is already owned, contiguous RGB. Keep this
// passthrough private to the processor; the ONNX DecodeImage op still takes bytes.
struct DecodeImageOrPixels : ort_extensions::DecodeImage {
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
    if (input.Shape().size() != 3) {
      return DecodeImage::Compute(input, output);
    }
    auto* dst = output.Allocate(input.Shape());
    std::copy_n(input.Data(), input.NumberOfElement(), dst);
    if (IsBgr()) {
      for (int64_t i = 0; i < input.NumberOfElement(); i += 3) {
        std::swap(dst[i], dst[i + 2]);
      }
    }
    return {};
  }
};
}  // namespace

Operation::KernelRegistry ImageProcessor::kernel_registry_ = {
    {"DecodeImage", []() { return CreateKernelInstance(&DecodeImageOrPixels::Compute); }},
    {"Resize", []() { return CreateKernelInstance(&Resize::Compute); }},
    {"Rescale", []() { return CreateKernelInstance(&Rescale::Compute); }},
    {"Normalize", []() { return CreateKernelInstance(&Normalize::Compute); }},
    {"CenterCrop", []() { return CreateKernelInstance(&CenterCrop::Compute); }},
    {"PadToSquare", []() { return CreateKernelInstance(&PadToSquare::Compute); }},
    {"ConvertRGB", []() { return CreateKernelInstance(convert_to_rgb); }},
    {"Permute3D", []() { return CreateKernelInstance(&Permute3D::Compute); }},
    {"Phi3ImageTransform", []() { return CreateKernelInstance(phi3_hd_transform); }},
    {"Llama3ImageTransform", []() { return CreateKernelInstance(&Llama3ImageTransform::Compute); }},
    {"Phi4VisionDynamicPreprocess", []() { return CreateKernelInstance(&Phi4VisionDynamicPreprocess::Compute); }},
    {"Phi4VisionProcessor", []() { return CreateKernelInstance(&Phi4VisionProcessor::Compute); }},
    {"PatchImage", []() { return CreateKernelInstance(&PatchImage::Compute); }},
    {"Gemma4ImageTransform", []() { return CreateKernelInstance(&Gemma4ImageTransform::Compute); }},
    {"PixtralImageSizes", []() { return CreateKernelInstance(&PixtralImageSizes::Compute); }}};  // NOLINT

OrtxStatus ImageProcessor::Init(std::string_view processor_def) {
  std::string processor_def_str;
  if (processor_def.size() >= 5 && processor_def.substr(processor_def.size() - 5) == ".json") {
    std::ifstream ifs = path({processor_def.data(), processor_def.size()}).open();
    if (!ifs.is_open()) {
      return {kOrtxErrorInvalidArgument, std::string("[ImageProcessor]: failed to open ") + std::string(processor_def)};
    }

    processor_def_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    processor_def = processor_def_str.c_str();
  }
  // pase the processor_def by json
  auto proc_json = json::parse(processor_def, nullptr, false);
  if (proc_json.is_discarded()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: failed to parse processor json configuration."};
  }

  auto processor_root = proc_json.at("processor");
  if (!processor_root.is_object()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: processor field is missing."};
  }

  if (auto iter = processor_root.find("name"); iter != processor_root.end()) {
    name_ = iter->get<std::string>();
  }

  if (auto iter = processor_root.find("output_aligner"); iter != processor_root.end()) {
    output_aligner_ = iter->get<std::string>();
  }

  auto transforms = processor_root.at("transforms");
  if (!transforms.is_array() || transforms.empty()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: transforms field is missing."};
  }

  starts_with_decode_ = transforms[0].at("operation").at("type") == "DecodeImage";
  return op_plan_.Init(transforms, kernel_registry_);
}

ImageProcessor::ImageProcessor() : OrtxObjectImpl(kOrtxKindProcessor) {}

template <typename T>
static ortc::Tensor<T>* StackTensor(const std::vector<TensorArgs>& arg_lists, int axis, ortc::IAllocator* allocator) {
  using TT = ortc::Tensor<T>;
  auto output = std::make_unique<TT>(allocator);

  if (arg_lists.empty()) {
    return nullptr;
  }

  size_t batch_size = arg_lists.size();

  std::vector<TT*> ts_ptrs;
  ts_ptrs.reserve(arg_lists.size());
  std::vector<int64_t> shape = arg_lists[0][axis]->Shape();
  for (auto& ts : arg_lists) {
    if (shape != ts[axis]->Shape()) {
      return nullptr;
    }
    ts_ptrs.push_back(static_cast<TT*>(ts[axis]));
  }

  std::vector<int64_t> output_shape = shape;
  output_shape.insert(output_shape.begin(), batch_size);

  char* buff = reinterpret_cast<char*>(output->Allocate(output_shape));
  for (size_t i = 0; i < batch_size; ++i) {
    auto ts = ts_ptrs[i];
    const char* ts_buff = reinterpret_cast<const char*>(ts->DataRaw());
    auto ts_size = ts->SizeInBytes();
    std::memcpy(buff + i * ts_size, ts_buff, ts_size);
  }

  return output.release();
}

std::tuple<OrtxStatus, ProcessorResult> ImageProcessor::PreProcess(ort_extensions::span<ImageRawData> image_data,
                                                                   ortc::Tensor<float>** pixel_values,
                                                                   ortc::Tensor<int64_t>** image_sizes,
                                                                   ortc::Tensor<int64_t>** num_img_takens) const {
  ProcessorResult r;

  std::vector<TensorArgs> inputs(image_data.size());
  std::vector<TensorPtr> input_tensor_objects(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    input_tensor_objects[i] = std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data());
    ts_input.push_back(input_tensor_objects[i].get());
  }

  std::vector<TensorArgs> outputs;
  OrtxRunner runner(op_plan_);
  auto status = runner.Run(inputs, outputs);
  if (!status.IsOk()) {
    return {status, r};
  }

  input_tensor_objects.clear();
  *pixel_values = r.pixel_values = StackTensor<float>(outputs, 0, runner.GetAllocator());
  *image_sizes = r.image_sizes = StackTensor<int64_t>(outputs, 1, runner.GetAllocator());
  *num_img_takens = r.num_img_tokens = StackTensor<int64_t>(outputs, 2, runner.GetAllocator());

  return {status, std::move(r)};
}

OrtxStatus ImageProcessor::PreProcess(ort_extensions::span<ImageRawData> image_data, TensorResult& r) const {
  return PreProcess(image_data, r, {nullptr, size_t{0}});
}

OrtxStatus ImageProcessor::PreProcess(ort_extensions::span<ImageRawData> image_data, TensorResult& r,
                                      ort_extensions::span<ImageShape> shapes) const {
  if (shapes.size() != 0 && (!starts_with_decode_ || shapes.size() != image_data.size())) {
    return {kOrtxErrorInvalidArgument, "Images from pixels need a workflow that starts with DecodeImage."};
  }
  std::vector<TensorArgs> inputs(image_data.size());
  std::vector<TensorPtr> input_tensor_objects(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    if (shapes.size() != 0) {
      const auto& s = shapes[i];
      if (s[0] <= 0 || s[1] <= 0 || s[2] != 3 || static_cast<uint64_t>(s[0]) * s[1] * 3 != image.size()) {
        return {kOrtxErrorInvalidArgument, "Invalid decoded RGB image shape."};
      }
      shape.assign(s.begin(), s.end());
    }
    input_tensor_objects[i] = std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data());
    ts_input.push_back(input_tensor_objects[i].get());
  }

  std::vector<TensorArgs> outputs;
  OrtxRunner runner(op_plan_);
  auto status = runner.Run(inputs, outputs);
  if (!status.IsOk()) {
    return status;
  }

  // clear the input tensors
  input_tensor_objects.clear();

  std::vector<TensorPtr> img_result = op_plan_.AllocateOutputs(runner.GetAllocator());
  ORTX_RETURN_IF_ERROR(OrtxRunner::StackTensors(outputs, img_result, runner.GetAllocator()));

  // Free up intermediate output tensors
  for (auto& tensor_arg : outputs) {
    for (auto& t : tensor_arg) {
      std::unique_ptr<ortc::TensorBase>(t).reset();
    }
  }

  if (output_aligner_ == "phi4-vision-aligner") {
    status = Phi4VisionProcessor::AlignOutputs(img_result);
  }

  if (status.IsOk()) {
    r.SetTensors(std::move(img_result));
  }

  return status;
}

void ImageProcessor::ClearOutputs(ProcessorResult* r) {
  if (r->pixel_values) {
    std::unique_ptr<ortc::TensorBase>(r->pixel_values).reset();
    r->pixel_values = nullptr;
  }

  if (r->image_sizes) {
    std::unique_ptr<ortc::TensorBase>(r->image_sizes).reset();
    r->image_sizes = nullptr;
  }

  if (r->num_img_tokens) {
    std::unique_ptr<ortc::TensorBase>(r->num_img_tokens).reset();
    r->num_img_tokens = nullptr;
  }
}
