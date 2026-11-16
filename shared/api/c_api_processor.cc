// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_processor.h"
#include "image_processor.h"

#include "c_api_utils.hpp"

#include <new>

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
  std::vector<ImageShape> shapes;  // empty for encoded images
};

// Nothing throws out of this function. Running out of memory returns kOrtxErrorOutOfMemory, and an error
// message that can't be copied is published as "Out of memory".
extError_t ORTX_API_CALL OrtxCreateImagesFromPixels(OrtxRawImages** images, const void* data[],
                                                    const size_t data_sizes[], const uint32_t widths[],
                                                    const uint32_t heights[], const size_t row_strides[],
                                                    const char* pixel_format, size_t num_images) {
  if (images == nullptr) {
    ReturnableStatus::SetLastErrorMessage("Invalid argument");
    return kOrtxErrorInvalidArgument;
  }
  *images = nullptr;

  const auto format = ParsePixelFormat(pixel_format);
  if (!format) {
    ReturnableStatus::SetLastErrorMessage("pixel_format must be \"RGB8\", \"RGBA8\" or \"BGRA8\".");
    return kOrtxErrorInvalidArgument;
  }

  try {
    auto images_obj = std::make_unique<RawImagesObject>();
    const OrtxStatus status = LoadImagesFromPixels(data, data_sizes, widths, heights, row_strides, *format, num_images,
                                                   images_obj->images, images_obj->shapes);
    if (!status.IsOk()) {
      ReturnableStatus::SetLastErrorMessage(status.Message());
      return status.Code();
    }
    images_obj->num_images = num_images;
    *images = static_cast<OrtxRawImages*>(images_obj.release());
    return kOrtxOK;
  } catch (const std::bad_alloc&) {
    ReturnableStatus::SetLastErrorMessage("Out of memory loading image pixels.");
    return kOrtxErrorOutOfMemory;
  } catch (...) {
    ReturnableStatus::SetLastErrorMessage("Unexpected error loading image pixels.");
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
