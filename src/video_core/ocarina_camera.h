// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

#include "common/common_types.h"

namespace VideoCommon {

// The last field used is the previous view-projection matrix at 0x180, ending at 0x1c0.
inline constexpr size_t OcarinaCameraBytes = 0x1c0;
using OcarinaCameraBuffer = std::array<f32, OcarinaCameraBytes / sizeof(f32)>;

enum class OcarinaCameraIssue { None, ShortBuffer, Extent, Projection };

struct OcarinaCamera {
    OcarinaCameraIssue issue{OcarinaCameraIssue::ShortBuffer};
    u32 width{}, height{};
    std::array<f32, 16> current{}, previous{};
};

inline OcarinaCamera DecodeOcarinaCamera(std::span<const f32> values, u32 target_width,
                                        u32 target_height, bool allow_padding) {
    OcarinaCamera camera;
    if (values.size_bytes() < OcarinaCameraBytes) {
        return camera;
    }
    const f32 width = values[0x120 / sizeof(f32)];
    const f32 height = values[0x124 / sizeof(f32)];
    camera.issue = OcarinaCameraIssue::Extent;
    if (!std::isfinite(width) || !std::isfinite(height) || width < 640.0f || height < 360.0f ||
        static_cast<f64>(width) > target_width || static_cast<f64>(height) > target_height) {
        return camera;
    }
    camera.width = static_cast<u32>(width);
    camera.height = static_cast<u32>(height);
    // DLSS Balanced draws 2276x1280 into 2288x1280 (QuincyForbes' f7d0938b5b).
    // Only DLSS accepts that padding, with the existing limit of 15 pixels per dimension.
    if (static_cast<f32>(camera.width) != width || static_cast<f32>(camera.height) != height ||
        target_width - camera.width >= 16 || target_height - camera.height >= 16 ||
        (!allow_padding && (camera.width != target_width || camera.height != target_height))) {
        return camera;
    }
    camera.issue = OcarinaCameraIssue::Projection;
    if (values[0xa8 / sizeof(f32)] != -1.0f || values[0xac / sizeof(f32)] != 0.0f) {
        return camera;
    }
    std::copy_n(values.begin() + 0x30 / sizeof(f32), 16, camera.current.begin());
    std::copy_n(values.begin() + 0x180 / sizeof(f32), 16, camera.previous.begin());
    camera.issue = OcarinaCameraIssue::None;
    return camera;
}

/// The FSR jitter for a phase (from 1) as a Halton (2, 3) offset in render pixels, applied to the
/// scene's viewport. AMD's own drivers (PAL) clip each viewport to an integer scissor that drops
/// the fraction of its right and bottom edges, so a negative offset loses the last column and row
/// of every jittered draw there; on those drivers the offsets stay in [0, 1) instead of
/// [-0.5, 0.5).
inline std::array<f32, 2> OcarinaJitter(u32 index, bool truncated_viewport_edges) {
    const auto halton = [](u32 value, u32 base) {
        f32 fraction = 1.0f;
        f32 result = 0.0f;
        while (value > 0) {
            fraction /= static_cast<f32>(base);
            result += fraction * static_cast<f32>(value % base);
            value /= base;
        }
        return result;
    };
    const f32 center = truncated_viewport_edges ? 0.0f : 0.5f;
    return {halton(index, 2) - center, halton(index, 3) - center};
}

} // namespace VideoCommon
