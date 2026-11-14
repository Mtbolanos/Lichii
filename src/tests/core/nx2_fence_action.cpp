// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_fence_action.h"

namespace {
using namespace std::chrono_literals;
constexpr u32 FenceId = 1;
using Action = Service::Nx2ScopedGuestFenceAction;
} // namespace

TEST_CASE("Scoped guest fence handles already completed and later completion", "[nx2_fence_action]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    unsigned callbacks = 0;
    SECTION("Fence is already complete") {
        syncpoints.IncrementGuest(FenceId);
        Action action{syncpoints, FenceId, 1, [&] { ++callbacks; }};
        REQUIRE(callbacks == 1);
        syncpoints.IncrementGuest(FenceId);
        REQUIRE(callbacks == 1);
    }
    SECTION("Fence completes during the wait") {
        Action action{syncpoints, FenceId, 2, [&] { ++callbacks; }};
        syncpoints.IncrementGuest(FenceId);
        REQUIRE(callbacks == 0);
        syncpoints.IncrementGuest(FenceId);
        REQUIRE(callbacks == 1);
        syncpoints.IncrementGuest(FenceId);
        REQUIRE(callbacks == 1);
    }
    REQUIRE(callbacks == 1);
}

TEST_CASE("Leaving a guest fence wait cancels only its own callback", "[nx2_fence_action]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    unsigned cancelled = 0;
    unsigned surviving = 0;
    Action other{syncpoints, FenceId, 2, [&] { ++surviving; }};
    {
        Action action{syncpoints, FenceId, 2, [&] { ++cancelled; }};
        syncpoints.IncrementGuest(FenceId);
        REQUIRE(cancelled == 0);
    }
    syncpoints.IncrementGuest(FenceId);
    REQUIRE(cancelled == 0);
    REQUIRE(surviving == 1);
}

TEST_CASE("Cancelled guest fence releases captured state before a later increment",
          "[nx2_fence_action]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    auto state = std::make_shared<unsigned>(0);
    std::weak_ptr<unsigned> weak = state;
    {
        Action action{syncpoints, FenceId, 1, [state] { ++*state; }};
        state.reset();
        REQUIRE_FALSE(weak.expired());
    }
    REQUIRE(weak.expired());
    syncpoints.IncrementGuest(FenceId);
    REQUIRE(weak.expired());
}

TEST_CASE("Leaving a guest fence wait joins a callback already running", "[nx2_fence_action]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    std::promise<void> entered_promise;
    auto entered = entered_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();
    std::atomic<bool> callback_finished = false;
    auto action = std::make_unique<Action>(syncpoints, FenceId, 1, [&] {
        entered_promise.set_value();
        release.wait();
        callback_finished = true;
    });
    auto producer = std::async(std::launch::async, [&] { syncpoints.IncrementGuest(FenceId); });
    const bool callback_entered = entered.wait_for(2s) == std::future_status::ready;
    auto cancel = std::async(std::launch::async, [&] {
        action.reset();
        return callback_finished.load();
    });
    const bool cancellation_waited = cancel.wait_for(10ms) == std::future_status::timeout;
    // Always unblock and join workers before any fatal assertion can unwind the test.
    release_promise.set_value();
    producer.get();
    const bool finished_before_cancel_returned = cancel.get();
    REQUIRE(callback_entered);
    REQUIRE(cancellation_waited);
    REQUIRE(finished_before_cancel_returned);
    syncpoints.IncrementGuest(FenceId);
    REQUIRE(callback_finished.load());
}
