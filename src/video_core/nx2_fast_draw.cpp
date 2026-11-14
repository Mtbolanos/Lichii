// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"

namespace VideoCommon::Nx2FastDraw {

void DeviceSpans::Update(const Tegra::MemoryManager& memory_, GPUVAddr address_, u64 size_) {
    const u64 serial = memory_.MapSerial();
    if (memory == &memory_ && address == address_ && size == size_ && map_serial == serial) {
        return;
    }
    memory = &memory_;
    address = address_;
    size = size_;
    map_serial = serial;
    spans.clear();
    const auto submapped = memory_.GetSubmappedDeviceRange(address_, size_);
    // Segments come in GPU order, joined where device memory continues.
    linear = submapped.size() == 1 && submapped[0].second == size_;
    linear_base = linear ? submapped[0].first : 0;
    for (const auto& [device, bytes] : submapped) {
        spans.emplace_back(device, device + bytes);
    }
    std::ranges::sort(spans);
    size_t merged = 0;
    for (const auto& span : spans) {
        if (merged != 0 && span.first <= spans[merged - 1].second) {
            spans[merged - 1].second = (std::max)(spans[merged - 1].second, span.second);
        } else {
            spans[merged++] = span;
        }
    }
    spans.resize(merged);
}

bool DeviceSpans::Intersects(DAddr begin, DAddr end) const {
    const auto span = std::ranges::upper_bound(spans, begin, {}, &std::pair<DAddr, DAddr>::second);
    return span != spans.end() && span->first < end;
}

} // namespace VideoCommon::Nx2FastDraw
