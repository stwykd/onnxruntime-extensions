// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_tokenizer.h"
// make sure the C only compiler compatibility only.
#include "ortx_processor.h"
#include "ortx_extractor.h"


#ifdef __cplusplus
extern "C"
#endif  // __cplusplus
extError_t tokenize_text(OrtxTokenizer* tokenizer, const char* text, char** decoded_text);

#ifdef __cplusplus
extern "C"
#endif
    extError_t
    get_word_piece_info(const OrtxTokenizer* tokenizer, extTokenId_t token_id, OrtxWordPieceInfo* info);

#ifdef __cplusplus
extern "C"
#endif
    extError_t
    create_images_from_bgra(const void* pixels, size_t data_size, uint32_t width, uint32_t height, size_t row_stride,
                            OrtxRawImages** images);
