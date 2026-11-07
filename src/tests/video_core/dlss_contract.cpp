// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/dlss_contract.h"
#include "video_core/dlss_evaluation.h"

namespace DC = VideoCommon::DlssContract;

TEST_CASE("Spatial cutscene replacement decodes sRGB into float without temporal history",
          "[dlss_contract]") {
    using VideoCore::Surface::PixelFormat;
    REQUIRE(DC::ValidColorFormats(PixelFormat::A8B8G8R8_SRGB,
                                  PixelFormat::R16G16B16A16_FLOAT, false));
    REQUIRE_FALSE(DC::ValidColorFormats(PixelFormat::A8B8G8R8_SRGB,
                                        PixelFormat::R16G16B16A16_FLOAT, true));
    for (const bool history : {false, true}) {
        REQUIRE(DC::ValidColorFormats(PixelFormat::B10G11R11_FLOAT,
                                      PixelFormat::B10G11R11_FLOAT, history));
        REQUIRE_FALSE(DC::ValidColorFormats(PixelFormat::A8B8G8R8_UNORM,
                                            PixelFormat::R16G16B16A16_FLOAT, history));
        REQUIRE_FALSE(DC::ValidColorFormats(PixelFormat::A8B8G8R8_SRGB,
                                            PixelFormat::A8B8G8R8_SRGB, history));
        REQUIRE_FALSE(DC::ValidColorFormats(PixelFormat::R16G16B16A16_FLOAT,
                                            PixelFormat::A8B8G8R8_SRGB, history));
    }
}

TEST_CASE("DLSS recording restores inputs before each evaluation failure returns", "[dlss_contract]") {
    using EvaluationResult = VideoCommon::DlssEvaluationResult;
    for (const auto failure : {EvaluationResult::Succeeded, EvaluationResult::OptionsFailed,
                               EvaluationResult::TokenFailed, EvaluationResult::ConstantsFailed,
                               EvaluationResult::EvaluateFailed}) {
        std::vector<int> events;
        bool sampled = false;
        bool partial_output = false;
        const auto result = VideoCommon::RecordDlssEvaluation(
            [&] { events.push_back(1); return failure != EvaluationResult::OptionsFailed; },
            [&] { events.push_back(2); return failure != EvaluationResult::TokenFailed; },
            [&] { events.push_back(3); sampled = true; },
            [&] { events.push_back(4); return failure != EvaluationResult::ConstantsFailed; },
            [&] { events.push_back(5); partial_output = true; return failure != EvaluationResult::EvaluateFailed; },
            [&] { events.push_back(6); sampled = false; });
        REQUIRE(result == failure);
        REQUIRE_FALSE(sampled);
        if (failure == EvaluationResult::OptionsFailed) {
            REQUIRE(events == std::vector<int>{1});
        } else if (failure == EvaluationResult::TokenFailed) {
            REQUIRE(events == std::vector<int>{1, 2});
        } else if (failure == EvaluationResult::ConstantsFailed) {
            REQUIRE(events == std::vector<int>{1, 2, 3, 4, 6});
        } else {
            REQUIRE(events == std::vector<int>{1, 2, 3, 4, 5, 6});
            REQUIRE(partial_output);
        }
    }
}

TEST_CASE("Fast Fusion contract tracing selects exact captured shaders", "[dlss_contract]") {
    REQUIRE(DC::FastFusionProgramId == 0x040041E0212AE000ULL);
    REQUIRE_FALSE(DC::IsFastFusionCandidate(0x040057C02159A000ULL, DC::FastFusionCandidates[0]));
    for (const auto key : DC::FastFusionCandidates) {
        REQUIRE(DC::IsFastFusionCandidate(DC::FastFusionProgramId, key));
        for (unsigned bit = 0; bit < 64; ++bit) {
            REQUIRE_FALSE(DC::IsFastFusionCandidate(DC::FastFusionProgramId ^ (1ULL << bit), key));
            REQUIRE_FALSE(DC::IsFastFusionCandidate(DC::FastFusionProgramId, key ^ (1ULL << bit)));
        }
    }
    REQUIRE_FALSE(DC::IsFastFusionCandidate(0x04007EF01F8AE000ULL, DC::FastFusionCandidates[0]));
}

TEST_CASE("Temporal replacement rejects invalid extents and nonfinite constants", "[dlss_contract]") {
    REQUIRE(DC::FitsExtent({1280, 720}, {1280, 720}));
    REQUIRE(DC::FitsExtent({1280, 720}, {2560, 1440}));
    REQUIRE_FALSE(DC::FitsExtent({1280, 720}, {1279, 720}));
    REQUIRE_FALSE(DC::FitsExtent({1280, 720}, {1280, 719}));
    REQUIRE_FALSE(DC::FitsExtent({2560, 1440}, {1280, 720}));
    REQUIRE(DC::ValidGeometry({1280, 720}, {2560, 1440}, {0.25f, -0.25f}, {1.0f, 1.0f}));
    REQUIRE(DC::ValidGeometry({1280, 720}, {1280, 720}, {0.0f, 0.0f}, {1.0f, 1.0f}));
    for (const auto render : {std::array<unsigned, 2>{0, 720}, {1280, 0}, {1, 720}, {1280, 1}}) {
        REQUIRE_FALSE(DC::ValidGeometry(render, {2560, 1440}, {}, {1.0f, 1.0f}));
    }
    REQUIRE_FALSE(DC::ValidGeometry({1280, 720}, {1279, 1440}, {}, {1.0f, 1.0f}));
    REQUIRE_FALSE(DC::ValidGeometry({1280, 720}, {2560, 719}, {}, {1.0f, 1.0f}));
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}) {
        for (unsigned component = 0; component < 2; ++component) {
            std::array<float, 2> value{};
            value[component] = invalid;
            REQUIRE_FALSE(DC::ValidGeometry({1280, 720}, {2560, 1440}, value, {1.0f, 1.0f}));
            REQUIRE_FALSE(DC::ValidGeometry({1280, 720}, {2560, 1440}, {}, value));
        }
    }
}
