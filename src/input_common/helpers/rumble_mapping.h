// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/input.h"

namespace InputCommon {

// Strengths in [0, 1] for the low frequency (heavy) and high frequency (light) motors of a
// controller with plain two motor rumble.
struct MotorStrengths {
    f32 low{};
    f32 high{};
};

// Each band drives its own motor, weakened as its frequency rises so some of the frequency can
// still be felt. curve_factor shapes the amplitude, 1 being linear.
MotorStrengths MapRumbleByBand(const Common::Input::VibrationStatus& vibration, f32 curve_factor);

// Each band's energy goes to the motor that matches its frequency: the heavy motor at or below
// 120 Hz, the light one at or above 480 Hz, and split on a log scale in between.
MotorStrengths MapRumbleByFrequency(const Common::Input::VibrationStatus& vibration,
                                    f32 curve_factor);

} // namespace InputCommon
