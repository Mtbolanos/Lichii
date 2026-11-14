// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"
#include "video_core/dlss_mode.h"

namespace Mode = VideoCommon::DlssMode;
using Quality = Settings::NvidiaDlssQuality;

TEST_CASE("DLSS settings preserve old profiles and per-game overrides", "[dlss_mode]") {
    REQUIRE(Settings::values.nvidia_dlss_preset.GetDefault() == Settings::NvidiaDlssPreset::M);
    REQUIRE(Settings::values.nvidia_dlss_quality.GetDefault() == Quality::Automatic);
    Settings::Linkage linkage;
    Settings::SwitchableSetting<Settings::NvidiaDlssPreset> preset{
        linkage, Settings::NvidiaDlssPreset::M, "preset", Settings::Category::Renderer};
    preset.LoadString("");
    REQUIRE(preset.GetValue() == Settings::NvidiaDlssPreset::M);
    preset.LoadString("0");
    REQUIRE(preset.GetValue() == Settings::NvidiaDlssPreset::M);
    preset.LoadString("invalid");
    REQUIRE(preset.GetValue() == Settings::NvidiaDlssPreset::M);
    for (const auto& [name, value] :
         Settings::EnumMetadata<Settings::NvidiaDlssPreset>::Canonicalizations()) {
        preset.SetValue(value);
        const auto saved = preset.ToString();
        preset.LoadString("");
        preset.LoadString(saved);
        REQUIRE(preset.GetValue() == value);
    }
    preset.SetValue(Settings::NvidiaDlssPreset::K);
    preset.SetGlobal(false);
    preset.LoadString("0");
    REQUIRE(preset.GetValue() == Settings::NvidiaDlssPreset::M);
    preset.SetGlobal(true);
    REQUIRE(preset.GetValue() == Settings::NvidiaDlssPreset::K);
    Settings::SwitchableSetting<Quality> quality{linkage, Quality::Automatic, "quality",
                                                 Settings::Category::Renderer};
    quality.LoadString("1");
    REQUIRE(quality.GetValue() == Quality::Native);
}

TEST_CASE("DLSS validates both dimensions and inclusive NGX size bounds", "[dlss_mode]") {
    constexpr Mode::Extent output{3840, 2160};
    constexpr Mode::Range range{{1280, 720}, {2560, 1440}};
    REQUIRE(Mode::Fits(range.min, output, Quality::Quality, range));
    REQUIRE(Mode::Fits(range.max, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({1280, 719}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({2561, 1440}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({1920, 1441}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({}, output, Quality::Quality, {}));
    REQUIRE_FALSE(Mode::Fits(range.max, output, Quality::Native, range));
    REQUIRE(Mode::Fits(range.max, range.max, Quality::Native, range));
}

TEST_CASE("DLSS requested modes fall back to a validated automatic mode", "[dlss_mode]") {
    const auto query = [](Quality mode) -> std::optional<Mode::Range> {
        if (mode == Quality::Quality)
            return Mode::Range{{2400, 1350}, {2700, 1520}};
        if (mode == Quality::Performance)
            return Mode::Range{{1800, 1000}, {2000, 1200}};
        return std::nullopt;
    };
    const auto valid = Mode::Select(Quality::Quality, {2560, 1440}, {3840, 2160}, query);
    REQUIRE(valid.mode == Quality::Quality);
    REQUIRE_FALSE(valid.fallback);
    const auto fallback = Mode::Select(Quality::Quality, {1920, 1080}, {3840, 2160}, query);
    REQUIRE(fallback.mode == Quality::Performance);
    REQUIRE(fallback.fallback);
    REQUIRE(fallback.requested_range->min.width == 2400);
    const auto unavailable = Mode::Select(Quality::Balanced, {1920, 1080}, {3840, 2160}, query);
    REQUIRE(unavailable.mode == Quality::Performance);
    REQUIRE(unavailable.fallback);
    REQUIRE_FALSE(unavailable.requested_range);
    const auto automatic = Mode::Select(Quality::Automatic, {2560, 1200}, {3840, 2160}, query);
    REQUIRE_FALSE(automatic.mode);
    REQUIRE_FALSE(automatic.fallback);
}

TEST_CASE("Automatic DLSS checks other modes when its ratio heuristic is invalid", "[dlss_mode]") {
    const auto result = Mode::Select(Quality::Automatic, {2560, 1440}, {3840, 2160},
                                     [](Quality mode) -> std::optional<Mode::Range> {
                                         if (mode == Quality::Balanced)
                                             return Mode::Range{{2000, 1000}, {3000, 1500}};
                                         return std::nullopt;
                                     });
    REQUIRE(result.mode == Quality::Balanced);
    REQUIRE_FALSE(result.fallback);
}
