// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/present/hdr_convert.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

using Settings::HdrSwapchain;
using Vulkan::ChooseHdrSurfaceFormat;

namespace {

constexpr VkSurfaceFormatKHR Sdr{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
constexpr VkSurfaceFormatKHR Hdr10{VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                   VK_COLOR_SPACE_HDR10_ST2084_EXT};
constexpr VkSurfaceFormatKHR Scrgb{VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};

bool Same(const std::optional<VkSurfaceFormatKHR>& a, const VkSurfaceFormatKHR& b) {
    return a && a->format == b.format && a->colorSpace == b.colorSpace;
}

} // Anonymous namespace

TEST_CASE("HDR output takes the preferred pair the surface lists", "[hdr_output]") {
    const std::array both{Sdr, Scrgb, Hdr10};
    REQUIRE(Same(ChooseHdrSurfaceFormat(both, HdrSwapchain::Hdr10), Hdr10));
    REQUIRE(Same(ChooseHdrSurfaceFormat(both, HdrSwapchain::Scrgb), Scrgb));
}

TEST_CASE("HDR output falls back to the other listed pair", "[hdr_output]") {
    const std::array scrgb_only{Sdr, Scrgb};
    REQUIRE(Same(ChooseHdrSurfaceFormat(scrgb_only, HdrSwapchain::Hdr10), Scrgb));
    const std::array hdr10_only{Sdr, Hdr10};
    REQUIRE(Same(ChooseHdrSurfaceFormat(hdr10_only, HdrSwapchain::Scrgb), Hdr10));
}

TEST_CASE("HDR output needs the exact format and colour space pair", "[hdr_output]") {
    // A 10-bit SDR format or an HDR10 colour space on another format is no HDR10 pair.
    const std::array mismatched{
        Sdr,
        VkSurfaceFormatKHR{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
        VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_HDR10_ST2084_EXT},
    };
    REQUIRE_FALSE(ChooseHdrSurfaceFormat(mismatched, HdrSwapchain::Hdr10));
    REQUIRE_FALSE(ChooseHdrSurfaceFormat(std::array{Sdr}, HdrSwapchain::Scrgb));
}

TEST_CASE("Layers are converted only where their encoding differs from the frame's",
          "[hdr_output]") {
    using Vulkan::HdrEncoding;
    using Vulkan::HdrPresent;
    Tegra::FramebufferConfig layer{};
    const HdrPresent sdr{};
    const HdrPresent hdr10{.color_space = VK_COLOR_SPACE_HDR10_ST2084_EXT};
    const HdrPresent scrgb{.color_space = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};

    layer.color_space = Tegra::ColorSpace::RGB709;
    REQUIRE_FALSE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, sdr)));
    const auto sdr_on_hdr10 = Vulkan::MakeHdrConvertParams(layer, hdr10);
    REQUIRE(Vulkan::NeedsHdrConvert(sdr_on_hdr10));
    REQUIRE(sdr_on_hdr10.input == HdrEncoding::Srgb);
    REQUIRE(sdr_on_hdr10.output == HdrEncoding::Pq);

    layer.color_space = Tegra::ColorSpace::RGB2020_PQ;
    REQUIRE_FALSE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, hdr10)));
    REQUIRE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, sdr)));
    REQUIRE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, scrgb)));

    layer.color_space = Tegra::ColorSpace::RGB709_Linear;
    layer.linear_unit_luminance = 0; // unknown: the scRGB 80 nits
    REQUIRE_FALSE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, scrgb)));
    layer.linear_unit_luminance = 100;
    const auto rescaled = Vulkan::MakeHdrConvertParams(layer, scrgb);
    REQUIRE(Vulkan::NeedsHdrConvert(rescaled));
    REQUIRE(rescaled.nits_per_unit == 100.0f);
}
