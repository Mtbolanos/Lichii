// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/host_wait_margin.h"

using Core::Timing::HostWaitMargin;

TEST_CASE("HostWaitMargin follows the 90th percentile of OS oversleeps", "[core][core_timing]") {
    HostWaitMargin margin;
    CHECK(margin.Get(0) == 0);

    // Windows at 0.5 ms timer resolution: wakes 150-650 us late.
    for (s64 i = 0; i < 128; ++i) {
        margin.OnOversleep(150'000 + (i * 37 % 128) * 500'000 / 128);
    }
    CHECK(margin.Estimate() >= 590'000);
    CHECK(margin.Estimate() <= 610'000);

    // Linux: wakes 50-60 us late. The old samples age out.
    for (s64 i = 0; i < 128; ++i) {
        margin.OnOversleep(50'000 + i % 11 * 1'000);
    }
    CHECK(margin.Estimate() >= 58'000);
    CHECK(margin.Estimate() <= 60'000);

    // A sleep the thread was descheduled through does not stretch the margin past its cap.
    for (s64 i = 0; i < 128; ++i) {
        margin.OnOversleep(50'000'000);
    }
    CHECK(margin.Estimate() == HostWaitMargin::MaxMarginNs);
}

TEST_CASE("HostWaitMargin spends at most a fifth of the time in hardware waits",
          "[core][core_timing]") {
    HostWaitMargin margin;
    for (size_t i = 0; i < HostWaitMargin::Samples; ++i) {
        margin.OnOversleep(1'000'000);
    }
    REQUIRE(margin.Estimate() == 1'000'000);

    // Events every 100 us, each waited for in full: the margin shrinks to what the budget refills.
    s64 now = 0;
    s64 precise = 0;
    for (int i = 0; i < 100'000; ++i) {
        const s64 wait = (std::min<s64>)(margin.Get(now), 100'000);
        margin.OnPreciseWait(wait);
        precise += wait;
        now += 100'000;
    }
    CHECK(precise <= now / HostWaitMargin::BudgetDivisor + HostWaitMargin::BurstNs);
    CHECK(precise >= now / HostWaitMargin::BudgetDivisor - 100'000);

    // Idle time refills the budget up to one burst.
    now += 1'000'000'000;
    CHECK(margin.Get(now) == 1'000'000);
    margin.OnPreciseWait(HostWaitMargin::BurstNs);
    CHECK(margin.Get(now) == 0);
    CHECK(margin.Get(now + 1'000'000) == 200'000);
}
