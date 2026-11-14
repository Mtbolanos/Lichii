// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_back_idle_trace.h"

using namespace VideoCommon::BackIdleTrace;

TEST_CASE("BackIdleTrace: an idle period splits by the front state in effect", "[video_core]") {
    const auto ns = [](const StateNs& split, FrontState state) {
        return split[static_cast<size_t>(state)];
    };
    // Parse from 50, drain from 120, queue empty from 150, parse again from 400 (after the end).
    const std::vector<u64> changes{Pack(50, FrontState::Parse), Pack(120, FrontState::Drain),
                                   Pack(150, FrontState::QueueEmpty), Pack(400, FrontState::Parse)};
    StateNs split{};
    SplitIdle(changes, 100, 300, split);
    REQUIRE(ns(split, FrontState::Parse) == 20);
    REQUIRE(ns(split, FrontState::Drain) == 30);
    REQUIRE(ns(split, FrontState::QueueEmpty) == 150);
    REQUIRE(ns(split, FrontState::Unknown) == 0);

    // The ring no longer holds the change in effect at the start: that part is unknown.
    StateNs wrapped{};
    SplitIdle(std::span{changes}.subspan(2), 100, 300, wrapped);
    REQUIRE(ns(wrapped, FrontState::Unknown) == 50);
    REQUIRE(ns(wrapped, FrontState::QueueEmpty) == 150);
}
