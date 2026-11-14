// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <future>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/dkb_cpu_trace.h"

namespace {
using namespace Common::DkbCpuTrace;
using namespace std::chrono_literals;

std::optional<u64> BurnCpu() {
    const auto start = Common::CurrentThreadCpuTimeNs();
    if (!start) {
        return std::nullopt;
    }
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    auto end = start;
    while (end && *end - *start < 1'000'000 && std::chrono::steady_clock::now() < deadline) {
        end = Common::CurrentThreadCpuTimeNs();
    }
    return end ? std::optional<u64>{*end - *start} : std::nullopt;
}
}

TEST_CASE("Thread CPU clocks report monotonic execution counters", "[common][dkb_cpu_trace]") {
    Common::ThreadCpuClock disabled{false};
    REQUIRE_FALSE(disabled.Read());
    const auto start = Common::CurrentThreadCpuTimeNs();
#if defined(_WIN32) || defined(__linux__)
    REQUIRE(start);
    Common::ThreadCpuClock captured;
    const auto captured_time = captured.Read();
    REQUIRE(captured_time);
    REQUIRE(*captured_time >= *start);
    const auto busy = BurnCpu();
    REQUIRE(busy);
    REQUIRE(*busy > 0);
#else
    REQUIRE_FALSE(start);
    Common::ThreadCpuClock unsupported;
    REQUIRE_FALSE(unsupported.Read());
#endif
}

TEST_CASE("Thread CPU deltas reject missing or replaced counters", "[common][dkb_cpu_trace]") {
    const ThreadSample before{.generation = 3, .active = true, .cpu_ns = 100};
    ThreadSample after{.generation = 3, .active = true, .cpu_ns = 145};
    REQUIRE(CpuTimeDelta(before, after) == 45U);
    REQUIRE(CpuTimeDelta(before, before) == 0U);
    after.generation = 4;
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    after = before;
    after.cpu_ns = 99;
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    after.cpu_ns.reset();
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    REQUIRE_FALSE(CpuTimeDelta(after, before));
    after = before;
    after.active = false;
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    REQUIRE_FALSE(CpuTimeDelta(after, before));
}

TEST_CASE("Thread CPU registrations track lifetime", "[common][dkb_cpu_trace]") {
    constexpr auto role = Role::GuestSingleCore;
    constexpr auto index = static_cast<size_t>(role);
    ThreadSample first;
    {
        Registration registration{role};
        first = ReadSnapshot().threads[index];
        REQUIRE(first.active);
        {
            Registration duplicate{role};
            REQUIRE(ReadSnapshot().threads[index].generation == first.generation);
        }
        REQUIRE(ReadSnapshot().threads[index].active);
    }
    REQUIRE_FALSE(ReadSnapshot().threads[index].active);
    REQUIRE_FALSE(ReadSnapshot().threads[index].cpu_ns);
    REQUIRE_FALSE(CpuTimeDelta(first, ReadSnapshot().threads[index]));
    {
        Registration replacement{role};
        const auto second = ReadSnapshot().threads[index];
        REQUIRE(second.generation > first.generation);
        REQUIRE_FALSE(CpuTimeDelta(first, second));
    }
}

TEST_CASE("Thread CPU snapshots read the registered thread not the caller", "[common][dkb_cpu_trace]") {
    constexpr auto index = static_cast<size_t>(Role::GpuCommand);
    std::promise<std::optional<u64>> ready;
    std::promise<std::optional<u64>> ended;
    std::promise<void> stop;
    auto stop_future = stop.get_future();
    std::jthread worker{[&] {
        Registration registration{Role::GpuCommand};
        ready.set_value(Common::CurrentThreadCpuTimeNs());
        stop_future.wait();
        ended.set_value(Common::CurrentThreadCpuTimeNs());
    }};
    const auto owner_before = ready.get_future().get();
    const auto before = ReadSnapshot();
    const auto busy = BurnCpu();
    const auto after = ReadSnapshot();
    stop.set_value();
    worker.join();
    const auto owner_after = ended.get_future().get();
    REQUIRE(after.wall_time >= before.wall_time);
    REQUIRE_FALSE(ReadSnapshot().threads[index].active);
#if defined(_WIN32) || defined(__linux__)
    const auto delta = CpuTimeDelta(before.threads[index], after.threads[index]);
    REQUIRE(busy);
    REQUIRE(*busy > 0);
    REQUIRE(delta);
    REQUIRE(owner_before);
    REQUIRE(owner_after);
    REQUIRE(*before.threads[index].cpu_ns >= *owner_before);
    REQUIRE(*after.threads[index].cpu_ns <= *owner_after);
#else
    REQUIRE_FALSE(busy);
    REQUIRE_FALSE(owner_before);
    REQUIRE_FALSE(owner_after);
    REQUIRE_FALSE(after.threads[index].cpu_ns);
#endif
}

TEST_CASE("Thread CPU snapshots serialize against unregister", "[common][dkb_cpu_trace]") {
    std::atomic<u32> unavailable{};
    std::promise<void> ready;
    std::jthread reader{[&](std::stop_token stop) {
        ready.set_value();
        while (!stop.stop_requested()) {
            const auto sample = ReadSnapshot().threads[static_cast<size_t>(Role::GpuBack)];
            if (sample.active && !sample.cpu_ns) {
                unavailable.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }};
    ready.get_future().wait();
    for (size_t i = 0; i < 100; ++i) {
        Registration registration{Role::GpuBack};
        std::this_thread::yield();
    }
    reader.request_stop();
    reader.join();
#if defined(_WIN32) || defined(__linux__)
    REQUIRE(unavailable.load() == 0);
#endif
    REQUIRE_FALSE(ReadSnapshot().threads[static_cast<size_t>(Role::GpuBack)].active);
}
