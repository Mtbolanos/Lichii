// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/ocarina_upscale.h"
#include "video_core/fsr.h"

TEST_CASE("OoT temporal backend selection is opt-in and vendor neutral", "[video_core][ocarina]") {
    using enum VideoCommon::OcarinaUpscaleBackend;
    using VideoCommon::SelectOcarinaUpscaler;
    CHECK(SelectOcarinaUpscaler(false, false, false, "") == Native);
    CHECK(SelectOcarinaUpscaler(false, false, true, "") == Native);
    CHECK(SelectOcarinaUpscaler(true, false, false, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(true, false, true, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(false, true, false, "") == Native);
    CHECK(SelectOcarinaUpscaler(false, true, true, "") == Dlss);
    CHECK(SelectOcarinaUpscaler(true, true, false, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(true, true, true, "") == Dlss);
    CHECK(SelectOcarinaUpscaler(true, true, true, "native") == Native);
    CHECK(SelectOcarinaUpscaler(false, true, true, "fsr") == Fsr);
    CHECK(SelectOcarinaUpscaler(false, false, false, "fsr") == Fsr);
    CHECK(SelectOcarinaUpscaler(false, false, false, "dlss") == Native);
    CHECK(SelectOcarinaUpscaler(false, false, true, "dlss") == Dlss);
    CHECK(SelectOcarinaUpscaler(false, true, true, "dlss") == Dlss);
    CHECK(SelectOcarinaUpscaler(false, false, false, "garbage") == Native);
}

TEST_CASE("OoT anti-aliasing choice picks the upscaler; DLSS keeps its setting",
          "[video_core][ocarina]") {
    using enum VideoCommon::OcarinaUpscaleBackend;
    using VideoCommon::SelectOcarinaUpscaler;
    using Settings::OotAntiAliasing;
    // Game and No sharpening: DLSS when its setting is on and it's available, as before.
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Game, false, true, "") == Native);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Game, true, true, "") == Dlss);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Game, true, false, "") == Native);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::NoSharpening, true, true, "") == Dlss);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::NoSharpening, false, false, "") == Native);
    // An FSR choice wins over the DLSS setting.
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Fsr3, true, true, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Fsr4, true, true, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Fsr3, false, false, "") == Fsr);
    // The environment override still decides.
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Fsr3, false, false, "native") == Native);
    CHECK(SelectOcarinaUpscaler(OotAntiAliasing::Fsr4, false, true, "dlss") == Dlss);
}

TEST_CASE("OoT FSR texture LOD bias follows the upscaler, the console mode and failure",
          "[video_core][ocarina]") {
    using enum VideoCommon::OcarinaUpscaleBackend;
    using VideoCommon::KeepOcarinaFsrLodBias;
    using VideoCommon::OcarinaFsrLodBias;
    CHECK(OcarinaFsrLodBias(Fsr) == -1.58f);
    CHECK(OcarinaFsrLodBias(Dlss) == 0.0f);
    CHECK(OcarinaFsrLodBias(Native) == 0.0f);
    CHECK(KeepOcarinaFsrLodBias(true, false, false));
    CHECK_FALSE(KeepOcarinaFsrLodBias(false, false, false));
    CHECK_FALSE(KeepOcarinaFsrLodBias(true, false, true));
    CHECK_FALSE(KeepOcarinaFsrLodBias(true, true, false));
    CHECK_FALSE(KeepOcarinaFsrLodBias(true, true, true));
}

TEST_CASE("FSR temporal sharpness uses the existing stops slider", "[video_core][fsr][ocarina]") {
    CHECK(FSR::RcasSharpnessFromStops(0.0f) == 1.0f);
    CHECK(FSR::RcasSharpnessFromStops(0.81f) == 0.595f);
    CHECK(FSR::RcasSharpnessFromStops(1.0f) == 0.5f);
    CHECK(FSR::RcasSharpnessFromStops(2.0f) == 0.0f);
    CHECK(FSR::RcasSharpnessFromStops(3.0f) == 0.0f);
    CHECK(FSR::RcasSharpnessFromStops(-1.0f) == 1.0f);
}
