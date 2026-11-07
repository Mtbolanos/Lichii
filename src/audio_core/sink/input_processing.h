// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>
#include "common/common_types.h"

namespace AudioCore::Sink {

/// Apply microphone gain with saturation to the signed 16-bit PCM range.
inline s16 ScaleInputSample(s16 sample, float gain) {
    return static_cast<s16>(std::clamp(static_cast<float>(sample) * gain, -32768.0f, 32767.0f));
}

/// Measure the largest absolute sample after gain, including the full negative PCM range.
inline u32 InputPeak(std::span<const s16> samples, float gain) {
    u32 peak{};
    for (const auto sample : samples) {
        peak = std::max(
            peak, static_cast<u32>(std::abs(static_cast<s32>(ScaleInputSample(sample, gain)))));
    }
    return peak;
}

/// Convert capture to the guest layout and gain, padding missing samples with silence.
inline std::vector<s16> ConvertInputSamples(std::span<const s16> captured, u64 num_samples,
                                            u32 device_channels, u32 system_channels, float gain) {
    std::vector<s16> samples(num_samples, 0);
    if (device_channels == 0 || system_channels == 0) {
        return samples;
    }
    const auto frames = std::min(num_samples / system_channels, captured.size() / device_channels);
    for (size_t frame = 0; frame < frames; ++frame) {
        if (system_channels == 1 && device_channels > 1) {
            s32 sum{};
            for (u32 channel = 0; channel < device_channels; ++channel) {
                sum += captured[frame * device_channels + channel];
            }
            samples[frame] = ScaleInputSample(static_cast<s16>(sum / static_cast<s32>(device_channels)), gain);
            continue;
        }
        for (u32 channel = 0; channel < std::min(system_channels, device_channels); ++channel) {
            samples[frame * system_channels + channel] =
                ScaleInputSample(captured[frame * device_channels + channel], gain);
        }
    }
    return samples;
}

} // namespace AudioCore::Sink
