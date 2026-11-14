// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "common/common_types.h"

namespace Core::Timing {

/// How long before a one-shot event the timer thread stops sleeping in the OS and waits the rest
/// in a hardware wait (Common::Event::SpinFor). OS sleeps end late by the host's timer tick (0.5 ms
/// on Windows) plus scheduling, so the margin follows the 90th percentile of how late the last
/// Samples sleeps ended. Hardware waits are paid from a budget of a fifth of the thread's time:
/// with events denser than the margin, waits fall back to OS sleeps rather than a busy core.
class HostWaitMargin {
public:
    static constexpr size_t Samples = 128;
    static constexpr size_t UpdateEvery = 16;
    static constexpr s64 MaxMarginNs = 2'000'000;
    static constexpr s64 BudgetDivisor = 5;
    static constexpr s64 BurstNs = 2'000'000;

    /// The margin for a wait that starts at `now`, at most what the budget has left.
    s64 Get(s64 now) {
        if (last_now) {
            budget_ns = (std::min)(BurstNs, budget_ns + (now - *last_now) / BudgetDivisor);
        }
        last_now = now;
        return std::clamp<s64>(budget_ns, 0, estimate_ns);
    }

    /// A timed OS sleep ended `late_ns` after the wake it asked for.
    void OnOversleep(s64 late_ns) {
        samples[next] = std::clamp<s64>(late_ns, 0, MaxMarginNs);
        next = (next + 1) % Samples;
        count = (std::min)(count + 1, Samples);
        if (++since_update < UpdateEvery) {
            return;
        }
        since_update = 0;
        std::array<s64, Samples> sorted = samples;
        const auto p90 = sorted.begin() + count * 9 / 10;
        std::nth_element(sorted.begin(), p90, sorted.begin() + count);
        estimate_ns = *p90;
    }

    /// The thread spent `ns` in a hardware wait.
    void OnPreciseWait(s64 ns) {
        budget_ns -= ns;
    }

    /// The margin before the budget applies.
    s64 Estimate() const {
        return estimate_ns;
    }

private:
    std::array<s64, Samples> samples{};
    size_t next{};
    size_t count{};
    size_t since_update{};
    s64 estimate_ns{};
    s64 budget_ns{BurstNs};
    std::optional<s64> last_now;
};

} // namespace Core::Timing
