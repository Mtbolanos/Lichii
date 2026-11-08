// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/unbacked_fallback.h"

namespace {

constexpr u64 Texels(u64 width, u64 height, u64 depth = 1, u64 layers = 1) {
    return width * height * depth * layers;
}

} // namespace

TEST_CASE("A texture cache image that does not fit fails instead of going unbacked",
          "[video_core][vulkan][unbacked_fallback]") {
    // The 1280x720 render target an 8 GB RX 6650 XT ran out of memory for in Ocarina of Time.
    REQUIRE_FALSE(Vulkan::UsesUnbackedFallback(Texels(1280, 720), true));
    REQUIRE_FALSE(Vulkan::UsesUnbackedFallback(Texels(1024, 1024), true));
    REQUIRE_FALSE(Vulkan::UsesUnbackedFallback(Vulkan::MaxBackedImageTexels, true));
}

TEST_CASE("An image too large to back stays unbacked", "[video_core][vulkan][unbacked_fallback]") {
    // Title 1's sparse virtual texture array: 8192x8192, 68 layers.
    REQUIRE(Vulkan::UsesUnbackedFallback(Texels(8192, 8192, 1, 68), true));
    REQUIRE(Vulkan::UsesUnbackedFallback(Vulkan::MaxBackedImageTexels + 1, true));
}

TEST_CASE("A caller that cannot fail still gets an unbacked image",
          "[video_core][vulkan][unbacked_fallback]") {
    REQUIRE(Vulkan::UsesUnbackedFallback(Texels(1280, 720), false));
    REQUIRE(Vulkan::UsesUnbackedFallback(Texels(8192, 8192, 1, 68), false));
}
