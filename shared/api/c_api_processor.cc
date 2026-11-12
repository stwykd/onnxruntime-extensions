// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_processor.h"
#include "image_processor.h"

#include "c_api_utils.hpp"
#include <limits>

using namespace ort_extensions;

extError_t ORTX_API_CALL OrtxCreateProcessor(OrtxProcessor** processor, const char* def) {
  if (processor == nullptr || def == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto proc_ptr = std::make_unique<ImageProcessor>();
  ReturnableStatus status = proc_ptr->Init(def);
  if (status.IsOk()) {
    *processor = static_cast<OrtxProcessor*>(proc_ptr.release());
  } else {
    *processor = nullptr;
  }

  return status.Code();
}

struct RawImagesObject : public OrtxObjectImpl {
 public:
  RawImagesObject() : OrtxObjectImpl(kOrtxKindRawImages) {}
  std::unique_ptr<ort_extensions::ImageRawData[]> images;
  size_t num_images{};
  std::vector<std::array<int64_t, 3>> shapes;
};

extError_t ORTX_API_CALL OrtxCreateImagesFromRawPixels(OrtxRawImages** images, const OrtxRawImage* data,
                                                     size_t num_images) {
  if (images) *images = nullptr;
  auto invalid = [](const char* message) {
    ReturnableStatus::last_error_message_ = message;
    return kOrtxErrorInvalidArgument;
  };
  if (!images || !data || num_images == 0) {
    return invalid("Raw pixels require non-null arguments and at least one image.");
  }
  constexpr size_t max_bytes = 512 * 1024 * 1024;
  if (num_images > max_bytes / 3) {
    return invalid("Raw pixel batch exceeds the 512 MiB limit.");
  }
  size_t total = 0;
  for (size_t i = 0; i < num_images; ++i) {
    const auto& image = data[i];
    if (!image.data || image.width == 0 || image.height == 0 ||
        image.width > 16384 || image.height > 16384 ||
        uint64_t{image.width} * image.height > 100'000'000) {
      return invalid("Raw image dimensions must be nonzero, at most 16384, and at most 100 million pixels.");
    }
    if (image.format != kOrtxPixelFormatRGB8 && image.format != kOrtxPixelFormatRGBA8 &&
        image.format != kOrtxPixelFormatBGRA8) {
      return invalid("Unsupported raw image pixel format.");
    }
    if (image.alpha_mode != kOrtxAlphaOpaque && image.alpha_mode != kOrtxAlphaStraight &&
        image.alpha_mode != kOrtxAlphaPremultiplied) {
      return invalid("Unsupported raw image alpha mode.");
    }
    if (image.format == kOrtxPixelFormatRGB8 && image.alpha_mode == kOrtxAlphaPremultiplied) {
      return invalid("RGB8 cannot have premultiplied alpha.");
    }
    const size_t channels = image.format == kOrtxPixelFormatRGB8 ? 3 : 4;
    const size_t row_bytes = image.width * channels;
    if (image.row_stride < row_bytes ||
        (image.height > 1 && image.row_stride > (std::numeric_limits<size_t>::max() - row_bytes) / (image.height - 1)) ||
        image.data_size < (image.height - 1) * image.row_stride + row_bytes) {
      return invalid("Raw image row stride or buffer size is invalid.");
    }
    const size_t bytes = size_t{image.width} * image.height * 3;
    if (bytes > max_bytes - total) {
      return invalid("Raw pixel batch exceeds the 512 MiB limit.");
    }
    total += bytes;
  }
  try {
    auto object = std::make_unique<RawImagesObject>();
    object->images = std::make_unique<ImageRawData[]>(num_images);
    object->num_images = num_images;
    object->shapes.reserve(num_images);
    for (size_t i = 0; i < num_images; ++i) {
      const auto& image = data[i];
      auto& rgb = object->images[i];
      rgb.resize(size_t{image.width} * image.height * 3);
      object->shapes.push_back({image.height, image.width, 3});
      const size_t channels = image.format == kOrtxPixelFormatRGB8 ? 3 : 4;
      const bool bgr = image.format == kOrtxPixelFormatBGRA8;
      for (size_t y = 0; y < image.height; ++y) {
        const auto* src = static_cast<const uint8_t*>(image.data) + y * image.row_stride;
        auto* dst = rgb.data() + y * image.width * 3;
        if (channels == 3) {
          std::copy_n(src, image.width * 3, dst);
          continue;
        }
        for (size_t x = 0; x < image.width; ++x) {
          for (size_t c = 0; c < 3; ++c) {
            unsigned value = src[x * channels + (bgr ? 2 - c : c)];
            if (image.alpha_mode == kOrtxAlphaPremultiplied) {
              const unsigned alpha = src[x * channels + 3];
              value = alpha ? std::min(255u, (value * 255 + alpha / 2) / alpha) : 0;
            }
            dst[x * 3 + c] = static_cast<uint8_t>(value);
          }
        }
      }
    }
    *images = static_cast<OrtxRawImages*>(object.release());
    return kOrtxOK;
  } catch (const std::exception& e) {
    ReturnableStatus::last_error_message_ = e.what();
    return kOrtxErrorInternal;
  }
}

extError_t ORTX_API_CALL OrtxCreateRawImages(OrtxRawImages** images, const void* data[], const int64_t sizes[],
                                             size_t num_images) {
  if (images == nullptr || data == nullptr || sizes == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto images_obj = std::make_unique<RawImagesObject>();
  images_obj->images = std::make_unique<ImageRawData[]>(num_images);
  images_obj->num_images = num_images;
  for (size_t i = 0; i < num_images; ++i) {
    images_obj->images[i].resize(static_cast<size_t>(sizes[i]));
    std::copy_n(static_cast<const uint8_t*>(data[i]), sizes[i], images_obj->images[i].data());
  }

  *images = static_cast<OrtxRawImages*>(images_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxLoadImages(OrtxRawImages** images, const char** image_paths, size_t num_images,
                                        size_t* num_images_loaded) {
  if (images == nullptr || image_paths == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto images_obj = std::make_unique<RawImagesObject>();
  auto [img, num] = LoadRawData<char const**, ImageRawData>(image_paths, image_paths + num_images);
  images_obj->images = std::move(img);
  images_obj->num_images = num;
  if (num_images_loaded != nullptr) {
    *num_images_loaded = num;
  }

  *images = static_cast<OrtxRawImages*>(images_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxImagePreProcess(OrtxProcessor* processor, OrtxRawImages* images,
                                             OrtxTensorResult** result) {
  if (processor == nullptr || images == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto processor_ptr = static_cast<ImageProcessor*>(processor);
  ReturnableStatus status(processor_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindProcessor));
  if (!status.IsOk()) {
    return status.Code();
  }

  auto images_ptr = static_cast<RawImagesObject*>(images);
  status = images_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindRawImages);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result_ptr = std::make_unique<TensorResult>();
  status =
      processor_ptr->PreProcess(ort_extensions::span(images_ptr->images.get(), images_ptr->num_images), *result_ptr,
                               ort_extensions::span(images_ptr->shapes.data(), images_ptr->shapes.size()));
  if (status.IsOk()) {
    *result = static_cast<OrtxTensorResult*>(result_ptr.release());
  } else {
    *result = nullptr;
  }

  return status.Code();
}
