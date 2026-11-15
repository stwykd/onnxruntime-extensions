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

/** Byte order of one 8-bit pixel in OrtxCreateImagesFromPixels. Alpha, if present, is ignored. */
typedef enum OrtxPixelFormat {
  OrtxPixelFormat_RGB8 = 0,   // R, G, B
  OrtxPixelFormat_RGBA8 = 1,  // R, G, B, A
  OrtxPixelFormat_BGRA8 = 2,  // B, G, R, A (Skia's kBGRA_8888, Windows' 32 bit DIBs)
} OrtxPixelFormat;

/**
 * @brief Copies decoded 8-bit pixels into an image batch, so PNG/JPEG encoding can be skipped.
 *
 * Unlike OrtxCreateRawImages, which takes encoded bytes, each image here is a top-down buffer of
 * interleaved pixels in the given byte order, `widths[i]` pixels per row and `heights[i]` rows.
 * `row_strides[i]` is the distance between rows in bytes, at least `widths[i] * channels`, so row
 * padding is allowed. `data_sizes[i]` must cover `(heights[i] - 1) * row_strides[i] + widths[i] * channels`;
 * the last row's padding isn't read. Widths and heights must be nonzero and at most 16384, with at
 * most 100 million pixels per image (the codecs' limits), and the batch can hold at most 512 MiB of RGB.
 *
 * Alpha is dropped without compositing, as the libpng decoder does, so pixels must be straight
 * (unpremultiplied) or opaque. Callers convert premultiplied pixels first. No color management,
 * orientation or resizing happens here.
 *
 * The pixels are copied, so callers may release them on return. Invalid input is rejected before any
 * pixels are copied; on any failure, including kOrtxErrorOutOfMemory, `*images` is null and the call
 * doesn't throw. The processor's workflow must start with DecodeImage: for these images it skips the
 * codec but keeps its `color_space`, and every later transform is unchanged.
 *
 * @param[out] images The created image batch, destroyed with OrtxDisposeOnly.
 * @param[in] data Array of num_images pixel buffers.
 * @param[in] data_sizes Array of num_images buffer sizes in bytes.
 * @param[in] widths Array of num_images widths in pixels.
 * @param[in] heights Array of num_images heights in pixels.
 * @param[in] row_strides Array of num_images row strides in bytes.
 * @param[in] format Pixel byte order of every image in the batch.
 * @param[in] num_images Number of images, at least one.
 * @return An `extError_t` value indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxCreateImagesFromPixels(OrtxRawImages** images, const void* data[],
                                                    const size_t data_sizes[], const uint32_t widths[],
                                                    const uint32_t heights[], const size_t row_strides[],
                                                    OrtxPixelFormat format, size_t num_images);

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
