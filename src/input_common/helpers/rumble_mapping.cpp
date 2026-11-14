// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>

#include "input_common/helpers/rumble_mapping.h"

namespace InputCommon {

namespace {
f32 ApplyCurve(f32 amplitude, f32 curve_factor) {
    return (amplitude + std::pow(amplitude, curve_factor)) * 0.5f;
}

f32 FrequencyScale(f32 frequency, f32 start, f32 width) {
    if (frequency <= start) {
        return 1.0f;
    }
    return (std::max)(1.0f - (frequency - start) / width, 0.3f);
}

f32 HighMotorShare(f32 frequency) {
    constexpr f32 heavy_limit = 120.0f;
    constexpr f32 light_limit = 480.0f;
    if (!(frequency > heavy_limit)) {
        return 0.0f;
    }
    return std::clamp(std::log2(frequency / heavy_limit) / std::log2(light_limit / heavy_limit),
                      0.0f, 1.0f);
}
} // Anonymous namespace

MotorStrengths MapRumbleByBand(const Common::Input::VibrationStatus& vibration, f32 curve_factor) {
    return {
        .low = ApplyCurve(vibration.low_amplitude, curve_factor) *
               FrequencyScale(vibration.low_frequency, 140.0f, 400.0f),
        .high = ApplyCurve(vibration.high_amplitude, curve_factor) *
                FrequencyScale(vibration.high_frequency, 200.0f, 700.0f),
    };
}

MotorStrengths MapRumbleByFrequency(const Common::Input::VibrationStatus& vibration,
                                    f32 curve_factor) {
    const f32 low_share = HighMotorShare(vibration.low_frequency);
    const f32 high_share = HighMotorShare(vibration.high_frequency);
    const f32 low = vibration.low_amplitude * (1.0f - low_share) +
                    vibration.high_amplitude * (1.0f - high_share);
    const f32 high = vibration.low_amplitude * low_share + vibration.high_amplitude * high_share;
    return {
        .low = ApplyCurve((std::min)(low, 1.0f), curve_factor),
        .high = ApplyCurve((std::min)(high, 1.0f), curve_factor),
    };
}

} // namespace InputCommon
