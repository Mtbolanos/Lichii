// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "input_common/helpers/rumble_mapping.h"

namespace {
Common::Input::VibrationStatus Bands(f32 low_amplitude, f32 low_frequency, f32 high_amplitude,
                                     f32 high_frequency) {
    return {
        .low_amplitude = low_amplitude,
        .low_frequency = low_frequency,
        .high_amplitude = high_amplitude,
        .high_frequency = high_frequency,
        .type = Common::Input::VibrationAmplificationType::Linear,
    };
}
} // Anonymous namespace

TEST_CASE("Frequency-aware rumble sends each band to the motor matching its frequency",
          "[input_common]") {
    using Catch::Approx;
    using InputCommon::MapRumbleByFrequency;

    const auto thud = MapRumbleByFrequency(Bands(0.0f, 160.0f, 0.8f, 80.0f), 1.0f);
    REQUIRE(thud.low == Approx(0.8f));
    REQUIRE(thud.high == Approx(0.0f));

    const auto buzz = MapRumbleByFrequency(Bands(0.6f, 600.0f, 0.0f, 320.0f), 1.0f);
    REQUIRE(buzz.low == Approx(0.0f));
    REQUIRE(buzz.high == Approx(0.6f));

    const auto middle = MapRumbleByFrequency(Bands(0.5f, 240.0f, 0.0f, 320.0f), 1.0f);
    REQUIRE(middle.low == Approx(0.25f));
    REQUIRE(middle.high == Approx(0.25f));

    const auto loud = MapRumbleByFrequency(Bands(1.0f, 100.0f, 1.0f, 100.0f), 1.0f);
    REQUIRE(loud.low == Approx(1.0f));
}

TEST_CASE("Rumble by band keeps the band to motor mapping", "[input_common]") {
    using Catch::Approx;

    const auto strengths = InputCommon::MapRumbleByBand(Bands(0.5f, 160.0f, 0.4f, 900.0f), 1.0f);
    REQUIRE(strengths.low == Approx(0.5f * 0.95f));
    REQUIRE(strengths.high == Approx(0.4f * 0.3f));
}
