// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

// typedefs to create/dispose function flood, and to make the API more C++ friendly with less casting
typedef OrtxObject OrtxProcessor;
typedef OrtxObject OrtxRawImages;

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a processor object with the specified processor definition
 *
 * \param processor Pointer to store the created processor object
 * \param processor_def The processor definition, either a path to the processor directory or a JSON string, and is
 * utf-8 encoded. \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxCreateProcessor(OrtxProcessor** processor, const char* processor_def);

/**
 * @brief Loads a set of images from the specified image paths.
 *
 * This function loads a set of images from the given image paths and returns a pointer to the loaded images.
 * The number of images loaded is also returned through the `num_images_loaded` parameter.
 *
 * @param[out] images A pointer to a pointer that will be set to the loaded images.
 * @param[in] image_paths An array of image paths.
 * @param[in] num_images The number of images to load.
 * @param[out] num_images_loaded A pointer to a variable that will be set to the number of images loaded.
 *
 * @return An error code indicating the status of the operation.
 */
extError_t ORTX_API_CALL OrtxLoadImages(OrtxRawImages** images, const char** image_paths, size_t num_images,
                                        size_t* num_images_loaded);


/**
 * @brief Copies encoded PNG/JPEG bytes into an image batch (not decoded pixels).
 *
 * This function creates raw images from the provided data. The raw images are stored in the `images` parameter.
 *
 * @param images Pointer to a pointer to the `OrtxRawImages` structure that will hold the created raw images.
 * @param data Array of pointers to the data for each image.
 * @param sizes Array of pointers to the sizes of each image.
 * @param num_images Number of images to create.
 * @return An `extError_t` value indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxCreateRawImages(OrtxRawImages** images, const void* data[], const int64_t sizes[], size_t num_images);

typedef enum {
  kOrtxPixelFormatRGB8 = 0,
  kOrtxPixelFormatRGBA8 = 1,
  kOrtxPixelFormatBGRA8 = 2,
} OrtxPixelFormat;

typedef enum {
  kOrtxAlphaOpaque = 0,
  kOrtxAlphaStraight = 1,
  kOrtxAlphaPremultiplied = 2,
} OrtxAlphaMode;

typedef struct {
  const void* data;
  size_t data_size;
  uint32_t width;
  uint32_t height;
  size_t row_stride;
  OrtxPixelFormat format;
  OrtxAlphaMode alpha_mode;
} OrtxRawImage;

/**
 * @brief Copies top-down, interleaved uint8 pixels into an owned RGB image batch.
 *
 * Unlike OrtxCreateRawImages (encoded bytes), this accepts decoded pixels. Each
 * row_stride must be at least width * channels; data_size must cover the last
 * pixel, not necessarily the final row's padding. Dimensions must be nonzero,
 * at most 16384 each, and at most 100 million pixels per image. The batch is
 * limited to 512 MiB of owned RGB pixels.
 *
 * Opaque and straight alpha are discarded without compositing. Premultiplied
 * RGBA/BGRA is unpremultiplied with min(255, (channel * 255 + alpha / 2) / alpha);
 * zero alpha produces black. RGB8 cannot be premultiplied. No color management,
 * orientation or resizing is performed. Callers may release the input on return.
 *
 * Preprocessing requires a leading DecodeImage transform; only decoding is
 * bypassed, including respecting its RGB/BGR color_space attribute.
 * On failure, *images is null.
 */
extError_t ORTX_API_CALL OrtxCreateImagesFromRawPixels(OrtxRawImages** images, const OrtxRawImage* data, size_t num_images);

/**
 * @brief Pre-processes the given raw images using the specified processor.
 *
 * This function applies preprocessing operations on the raw images using the provided processor.
 * The result of the preprocessing is stored in the `OrtxImageProcessorResult` object.
 *
 * @param processor A pointer to the `OrtxProcessor` object used for preprocessing.
 * @param images A pointer to the `OrtxRawImages` object containing the raw images to be processed.
 * @param result A pointer to the `OrtxImageProcessorResult` object to store the preprocessing result.
 * @return An `extError_t` value indicating the success or failure of the preprocessing operation.
 */
extError_t ORTX_API_CALL OrtxImagePreProcess(OrtxProcessor* processor, OrtxRawImages* images,
                                             OrtxTensorResult** result);

#ifdef __cplusplus
}
#endif
