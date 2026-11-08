// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/texture_cache/host_mip_limits.h"
using VideoCommon::AppendHostMipEntries;
using VideoCommon::HostMipLevelCount;
using VideoCommon::IsHostMipLevel;
using VideoCommon::LimitHostMipRange;
TEST_CASE("Host mip count cannot exceed the physical chain", "[video_core][mips]") {
    REQUIRE(HostMipLevelCount(48, 96, 1, 15) == 7);
    REQUIRE(HostMipLevelCount(128, 128, 1, 15) == 8);
    REQUIRE(HostMipLevelCount(1, 1, 1, 15) == 1);
    REQUIRE(HostMipLevelCount(1, 1, 256, 15) == 9);
    REQUIRE(HostMipLevelCount(512, 512, 1, 10) == 10);
    REQUIRE(HostMipLevelCount(384, 288, 1, 9) == 9);
    REQUIRE(HostMipLevelCount(48, 96, 1, 4) == 4);
    REQUIRE(HostMipLevelCount(0x80000000u, 1, 1, 32) == 32);
}
TEST_CASE("Host mip view ranges remain inside the retained chain", "[video_core][mips]") {
    for (const auto& row :
         std::array<std::array<int, 4>, 5>{{{{0, 15, 0, 7}},
                                            {{3, 12, 3, 4}},
                                            {{14, 1, 6, 1}},
                                            {{-1, 16, 0, 7}},
                                            {{6, std::numeric_limits<int>::max(), 6, 1}}}}) {
        const auto result = LimitHostMipRange(row[0], row[1], 7);
        REQUIRE(result.base == row[2]);
        REQUIRE(result.count == row[3]);
    }
}
TEST_CASE("Filtering host mips preserves guest staging offsets and array strides",
          "[video_core][mips]") {
    struct Copy {
        int level;
        unsigned layer;
        uint64_t offset;
    };
    std::vector<Copy> guest, host;
    constexpr uint64_t stride = 0x10000;
    for (unsigned layer = 0; layer < 3; ++layer)
        for (int level = 0; level < 15; ++level)
            guest.push_back({level, layer, layer * stride + static_cast<unsigned>(level) * 256});
    AppendHostMipEntries(guest, host, HostMipLevelCount(48, 96, 1, 15),
                         [](const Copy& copy) { return copy.level; });
    REQUIRE(guest.size() == 45);
    REQUIRE(host.size() == 21);
    for (size_t n = 0; n < host.size(); ++n) {
        const auto& original = guest[(n / 7) * 15 + n % 7];
        REQUIRE(host[n].level == original.level);
        REQUIRE(host[n].layer == original.layer);
        REQUIRE(host[n].offset == original.offset);
    }
    REQUIRE_FALSE(IsHostMipLevel(-1, 7));
    REQUIRE_FALSE(IsHostMipLevel(7, 7));
    REQUIRE(IsHostMipLevel(6, 7));
}
TEST_CASE("Downscaled images can require a shorter host mip chain", "[video_core][mips]") {
    const auto original = HostMipLevelCount(64, 64, 1, 7);
    REQUIRE(original == 7);
    REQUIRE(HostMipLevelCount(32, 32, 1, original) < original);
    REQUIRE(HostMipLevelCount(128, 128, 1, original) == original);
}
