// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_movie_clock.h"

namespace {
constexpr s64 Frame = 16667;
constexpr s64 Start = 5'000'000;
} // namespace

TEST_CASE("NX2 movie clock ignores frames acquired ahead of it", "[nx2_movie_clock]") {
    // A player with two output images that takes each frame as soon as it is written, a frame
    // ahead of when it shows it.
    Service::Nx2MovieClock clock;
    clock.Acquired(Start, 0, 0);
    for (s64 frame = 1; frame < 600; ++frame) {
        CAPTURE(frame);
        const s64 now = Start + (frame - 1) * Frame + 500;
        clock.Acquired(now, static_cast<u64>(frame * Frame), 0);
        REQUIRE(clock.Due(now, 0) == static_cast<u64>(now - Start));
    }
}

TEST_CASE("NX2 movie clock runs on while a slow client falls behind", "[nx2_movie_clock]") {
    // Ocarina of Time: one 60 fps movie frame per frame drawn, at 48 fps.
    Service::Nx2MovieClock clock;
    for (s64 frame = 0; frame < 120; ++frame) {
        const s64 now = Start + frame * 20833;
        clock.Acquired(now, static_cast<u64>(frame * Frame), 0);
        REQUIRE(clock.Due(now, 0) == static_cast<u64>(now - Start));
    }
}

TEST_CASE("NX2 movie clock restarts after a pause, a loop or a long lag", "[nx2_movie_clock]") {
    using Clock = Service::Nx2MovieClock;
    Clock clock;
    REQUIRE_FALSE(clock.Due(Start, 0));
    clock.Acquired(Start, 0, 0);
    clock.Acquired(Start + Frame, Frame, 0);

    // No acquire for over a second: no due pts until the next acquire restarts the clock.
    s64 now = Start + Frame + Clock::PauseGap + 1;
    REQUIRE_FALSE(clock.Due(now, 0));
    clock.Acquired(now, 2 * Frame, 0);
    REQUIRE(clock.Due(now + 1000, 0) == 2 * Frame + 1000);

    // Emulation paused and resumed since the last acquire.
    const s64 resumed = now + 2000;
    REQUIRE_FALSE(clock.Due(now + 3000, resumed));
    now += 4000;
    clock.Acquired(now, 3 * Frame, resumed);
    REQUIRE(clock.Due(now, resumed) == 3 * Frame);

    // The movie loops back to its start.
    now += Frame;
    clock.Acquired(now, 0, resumed);
    REQUIRE(clock.Due(now, resumed) == 0);

    // The client takes a frame every half second: once it lags by more than MaxCatchUp, the
    // clock restarts from its frame.
    const s64 loop = now;
    for (s64 step = 1; step <= 7; ++step) {
        now = loop + step * Clock::PauseGap / 2;
        clock.Acquired(now, static_cast<u64>(step * Frame), resumed);
        const s64 lag = (now - loop) - step * Frame;
        CAPTURE(step, lag);
        REQUIRE(clock.Due(now, resumed) ==
                static_cast<u64>(lag > Clock::MaxCatchUp ? step * Frame : now - loop));
    }

    clock.Reset();
    REQUIRE_FALSE(clock.Due(now, resumed));
}
