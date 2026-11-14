// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <opus.h>

#include "common/assert.h"
#include "core/hle/service/audio/errors.h"

namespace AudioCore::OpusDecoder {

constexpr Result ResultCodeFromLibOpusErrorCode(u64 error_code) {
    using namespace Service::Audio;

    s32 error{static_cast<s32>(error_code)};
    ASSERT(error <= OPUS_OK);
    switch (error) {
    case OPUS_ALLOC_FAIL:
        R_THROW(ResultLibOpusAllocFail);
    case OPUS_INVALID_STATE:
        R_THROW(ResultLibOpusInvalidState);
    case OPUS_UNIMPLEMENTED:
        R_THROW(ResultLibOpusUnimplemented);
    case OPUS_INVALID_PACKET:
        R_THROW(ResultLibOpusInvalidPacket);
    case OPUS_INTERNAL_ERROR:
        R_THROW(ResultLibOpusInternalError);
    case OPUS_BUFFER_TOO_SMALL:
        R_THROW(ResultBufferTooSmall);
    case OPUS_BAD_ARG:
        R_THROW(ResultLibOpusBadArg);
    case OPUS_OK:
        R_RETURN(ResultSuccess);
    }
    UNREACHABLE();
}

} // namespace AudioCore::OpusDecoder
