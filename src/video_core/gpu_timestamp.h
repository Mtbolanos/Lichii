// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace Tegra {

/// How much faster Boost and Overclock make Switch 2 GPU work look.
[[nodiscard]] constexpr u64 AmpereClockDivisor(Settings::GpuClock clock) {
    switch (clock) {
    case Settings::GpuClock::Boost:
        return 4;
    case Settings::GpuClock::Overclock:
        return 16;
    default:
        return 1;
    }
}

/// The last present's CoreTiming time and the GPU clock latched with it, packed into one word so
/// both change together: a GPU Clocks change takes effect at the next present, never mid-frame.
struct AmpereAnchor {
    u64 bits{};

    [[nodiscard]] static constexpr AmpereAnchor At(u64 present_ns, Settings::GpuClock clock) {
        return {present_ns << 2 | static_cast<u64>(clock)};
    }
    [[nodiscard]] constexpr u64 PresentNs() const {
        return bits >> 2;
    }
    [[nodiscard]] constexpr Settings::GpuClock Clock() const {
        return static_cast<Settings::GpuClock>(bits & 3);
    }
};

/// Switch 2 titles read report timestamps as nanoseconds and calibrate them against CPU time, so
/// Boost and Overclock shrink only the time since the last present: GPU durations within a frame
/// look shorter, while a timestamp never runs ahead of the CPU clock and trails it by less than
/// the time since that present. Before the first present the clock is exact. Maxwell scales its
/// ticks by the current clock.
template <typename Timing>
[[nodiscard]] u64 ReadGpuTimestamp(const Timing& timing, bool ampere, Settings::GpuClock clock,
                                   AmpereAnchor anchor) {
    if (ampere) {
        const u64 now = static_cast<u64>(timing.GetGlobalTimeNs().count());
        const u64 present_ns = anchor.PresentNs();
        if (present_ns == 0 || now <= present_ns) {
            return now;
        }
        return present_ns + (now - present_ns) / AmpereClockDivisor(anchor.Clock());
    }
    const u64 ticks = timing.GetGPUTicks();
    switch (clock) {
    case Settings::GpuClock::Boost:
        return ticks / 256;
    case Settings::GpuClock::Overclock:
        return ticks / 512;
    default:
        return ticks;
    }
}

} // namespace Tegra
