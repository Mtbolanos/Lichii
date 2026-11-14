// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/unbacked_fallback.h"

namespace {

constexpr u64 Texels(u64 width, u64 height, u64 depth = 1, u64 layers = 1) {
    return width * height * depth * layers;
}

constexpr u64 Rgba8Bytes(u64 texels) {
    return texels * 4;
}

} // namespace

TEST_CASE("A texture cache image that does not fit fails instead of going unbacked",
          "[video_core][vulkan][unbacked_fallback]") {
    // The 1280x720 render target an 8 GB RX 6650 XT ran out of memory for in Ocarina of Time.
    REQUIRE_FALSE(
        Vulkan::UsesUnbackedFallback(Texels(1280, 720), Rgba8Bytes(Texels(1280, 720)), true));
    REQUIRE_FALSE(
        Vulkan::UsesUnbackedFallback(Texels(1024, 1024), Rgba8Bytes(Texels(1024, 1024)), true));
    REQUIRE_FALSE(Vulkan::UsesUnbackedFallback(Vulkan::MaxBackedImageTexels,
                                               Rgba8Bytes(Vulkan::MaxBackedImageTexels), true));
}

TEST_CASE("An image too large to back stays unbacked", "[video_core][vulkan][unbacked_fallback]") {
    // Title 1's sparse virtual texture array: 8192x8192, 68 layers, re-compressed to BC3.
    REQUIRE(
        Vulkan::UsesUnbackedFallback(Texels(8192, 8192, 1, 68), Texels(8192, 8192, 1, 68), true));
    REQUIRE(Vulkan::UsesUnbackedFallback(Vulkan::MaxBackedImageTexels + 1,
                                         Rgba8Bytes(Vulkan::MaxBackedImageTexels + 1), true));
}

TEST_CASE("A compressed volume over the texel limit is backed",
          "[video_core][vulkan][unbacked_fallback]") {
    // Metroid Prime 4's light volumes: 1408x256x1008 BC1 (8 bytes a 4x4 block) and BC6H (16).
    constexpr u64 texels = Texels(1408, 256, 1008);
    constexpr u64 bc1_bytes = (1408 / 4) * (256 / 4) * 1008 * 8;
    REQUIRE(texels > Vulkan::MaxBackedImageTexels);
    REQUIRE_FALSE(Vulkan::IsTooLargeToBack(texels, bc1_bytes));
    REQUIRE_FALSE(Vulkan::IsTooLargeToBack(texels, bc1_bytes * 2));
    REQUIRE_FALSE(Vulkan::UsesUnbackedFallback(texels, bc1_bytes, true));
}

TEST_CASE("A caller that cannot fail still gets an unbacked image",
          "[video_core][vulkan][unbacked_fallback]") {
    REQUIRE(Vulkan::UsesUnbackedFallback(Texels(1280, 720), Rgba8Bytes(Texels(1280, 720)), false));
    REQUIRE(Vulkan::UsesUnbackedFallback(Texels(8192, 8192, 1, 68), Texels(8192, 8192, 1, 68),
                                         false));
}
