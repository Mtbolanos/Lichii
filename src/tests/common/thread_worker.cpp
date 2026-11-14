// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <future>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "common/thread_worker.h"

namespace {
class CleanupGate {
public:
    CleanupGate(std::promise<void>& entered_, std::shared_future<void> release_)
        : entered{entered_}, release{std::move(release_)} {}

    ~CleanupGate() {
        entered.set_value();
        release.wait();
    }

private:
    std::promise<void>& entered;
    std::shared_future<void> release;
};
} // namespace

TEST_CASE("Thread worker drain includes task capture destruction", "[common][thread_worker]") {
    using namespace std::chrono_literals;

    std::promise<void> cleanup_entered;
    auto cleaning = cleanup_entered.get_future();
    std::promise<void> release_cleanup;
    Common::ThreadWorker workers{1, "CleanupTest"};
    auto resource =
        std::make_unique<CleanupGate>(cleanup_entered, release_cleanup.get_future().share());
    workers.QueueWork([resource = std::move(resource)] { static_cast<void>(resource); });
    cleaning.wait();

    std::promise<void> waiter_started;
    auto waiting = waiter_started.get_future();
    auto drained = std::async(std::launch::async, [&] {
        waiter_started.set_value();
        workers.WaitForRequests();
    });
    waiting.wait();
    const bool returned_during_cleanup = drained.wait_for(100ms) == std::future_status::ready;
    const bool admitted_during_cleanup = workers.TryQueueWork([] {});

    // Release the gate before asserting so a failure cannot strand either thread.
    release_cleanup.set_value();
    drained.get();
    workers.WaitForRequests();
    REQUIRE_FALSE(admitted_during_cleanup);
    REQUIRE_FALSE(returned_during_cleanup);
}
