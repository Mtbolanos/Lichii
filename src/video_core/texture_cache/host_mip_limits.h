// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>
namespace VideoCommon {
// Host limits only. Do not change guest mip offsets or array-layer stride.
constexpr uint32_t HostMipLevelCount(uint32_t width, uint32_t height, uint32_t depth,
                                     uint32_t requested) {
    const uint32_t full = std::bit_width((std::max)({width, height, depth, uint32_t{1}}));
    return std::clamp(requested, uint32_t{1}, full);
}
constexpr bool IsHostMipLevel(int32_t level, uint32_t levels) {
    return level >= 0 && static_cast<uint32_t>(level) < levels;
}
template <class Range, class Output, class Level>
void AppendHostMipEntries(const Range& entries, Output& output, uint32_t levels, Level level) {
    for (const auto& entry : entries)
        if (IsHostMipLevel(level(entry), levels))
            output.push_back(entry);
}
struct HostMipRange {
    int32_t base{}, count{};
};
constexpr HostMipRange LimitHostMipRange(int32_t base, int32_t count, uint32_t levels) {
    const int64_t limit = (std::max)(levels, uint32_t{1});
    const int64_t first = std::clamp(int64_t{base}, int64_t{0}, limit - 1);
    const int64_t last = std::clamp(int64_t{base} + count, first + 1, limit);
    return {static_cast<int32_t>(first), static_cast<int32_t>(last - first)};
}
} // namespace VideoCommon
