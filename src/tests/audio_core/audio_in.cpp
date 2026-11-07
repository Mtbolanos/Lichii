// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "audio_core/in/audio_in_system.h"
#include "core/core.h"
#include "tests/core/system_test_context.h"

TEST_CASE("Audio input rejects unsupported configurations before opening a session",
          "[audio][input]") {
    using AudioCore::AudioIn::System;
    const AudioCore::AudioIn::AudioInParameter automatic{};
    REQUIRE(System::IsConfigValid("", automatic).IsSuccess());
    REQUIRE(System::IsConfigValid("BuiltInHeadset", automatic).IsSuccess());
    REQUIRE(System::IsConfigValid("Uac", automatic).IsSuccess());
    REQUIRE(System::IsConfigValid("MissingDevice", automatic) == Service::Audio::ResultNotFound);

    AudioCore::AudioIn::AudioInParameter parameter{};
    parameter.sample_rate = 48000;
    parameter.channel_count = 2;
    REQUIRE(System::IsConfigValid("Uac", parameter).IsSuccess());
    parameter.sample_rate = 44100;
    REQUIRE(System::IsConfigValid("Uac", parameter) == Service::Audio::ResultInvalidSampleRate);
}

TEST_CASE("Audio input cannot start after rejected initialization", "[audio][input]") {
    ScopedSystemTestContext environment;
    Core::System system;
    AudioCore::AudioIn::System input{system, nullptr, 0};
    REQUIRE(input.Initialize("MissingDevice", {}, nullptr, 0) == Service::Audio::ResultNotFound);
    REQUIRE(input.Start() == Service::Audio::ResultInvalidHandle);
    REQUIRE(input.GetState() == AudioCore::AudioIn::State::Stopped);
}
