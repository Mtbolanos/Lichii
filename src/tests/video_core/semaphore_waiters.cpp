// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/puller.h"
#include "video_core/nx2.h"

using Tegra::Nx2SemaphoreWaiters;

TEST_CASE("Semaphore waiters latch on the writes that meet them", "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    const int equal{}, gequal{}, mask{};
    REQUIRE_FALSE(waiters.Latched(&equal));
    REQUIRE(waiters.Add(&equal, 0x1000, 5, Nx2SemaphoreWaiters::AcquireEqual));
    REQUIRE(waiters.Add(&gequal, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal));
    REQUIRE(waiters.Add(&mask, 0x2000, 0x4, Nx2SemaphoreWaiters::AcquireMask));

    waiters.OnWrite(0x1000, 4);
    waiters.OnWrite(0x2000, 0x3);
    waiters.OnWrite(0x3000, 0xffffffff);
    REQUIRE_FALSE(waiters.Latched(&equal));
    REQUIRE_FALSE(waiters.Latched(&gequal));
    REQUIRE_FALSE(waiters.Latched(&mask));

    waiters.OnWrite(0x1000, 6);
    REQUIRE_FALSE(waiters.Latched(&equal));
    REQUIRE(waiters.Latched(&gequal));
    waiters.OnWrite(0x1000, 5);
    REQUIRE(waiters.Latched(&equal));
    // A later write that no longer meets it keeps the latch.
    waiters.OnWrite(0x1000, 0);
    REQUIRE(waiters.Latched(&equal));
    waiters.OnWrite(0x2000, 0x6);
    REQUIRE(waiters.Latched(&mask));

    // Adding again starts unlatched, on the new word.
    REQUIRE(waiters.Add(&equal, 0x1004, 7, Nx2SemaphoreWaiters::AcquireEqual));
    REQUIRE_FALSE(waiters.Latched(&equal));
    waiters.OnWrite(0x1000, 7);
    REQUIRE_FALSE(waiters.Latched(&equal));
    waiters.OnWrite(0x1004, 7);
    REQUIRE(waiters.Latched(&equal));

    waiters.Remove(&gequal);
    REQUIRE_FALSE(waiters.Latched(&gequal));
    waiters.Remove(&equal);
    waiters.Remove(&mask);
    waiters.Remove(&mask);
    REQUIRE_FALSE(waiters.Latched(&equal));
    REQUIRE_FALSE(waiters.Latched(&mask));
}

TEST_CASE("Semaphore waiters refuse owners past the table and reuse freed slots",
          "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    std::array<int, Nx2SemaphoreWaiters::MaxWaiters + 1> owners{};
    for (size_t i = 0; i < Nx2SemaphoreWaiters::MaxWaiters; ++i) {
        REQUIRE(waiters.Add(&owners[i], 0x1000 + i * 4, 1, Nx2SemaphoreWaiters::AcquireGequal));
    }
    const int* const extra = &owners.back();
    REQUIRE_FALSE(waiters.Add(extra, 0x8000, 1, Nx2SemaphoreWaiters::AcquireGequal));
    waiters.OnWrite(0x8000, 1);
    REQUIRE_FALSE(waiters.Latched(extra));
    // An owner already in the table moves to its new word.
    REQUIRE(waiters.Add(&owners[3], 0x9000, 2, Nx2SemaphoreWaiters::AcquireGequal));
    waiters.OnWrite(0x100c, 1);
    REQUIRE_FALSE(waiters.Latched(&owners[3]));
    waiters.OnWrite(0x9000, 2);
    REQUIRE(waiters.Latched(&owners[3]));

    waiters.Remove(&owners[5]);
    REQUIRE(waiters.Add(extra, 0x8000, 1, Nx2SemaphoreWaiters::AcquireGequal));
    waiters.OnWrite(0x8000, 1);
    REQUIRE(waiters.Latched(extra));
    REQUIRE_FALSE(waiters.Latched(&owners[5]));
}

TEST_CASE("Semaphore waiters see what the writer wrote before the latch",
          "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    constexpr u32 Rounds = 2000;
    std::array<u32, 4> words{};
    std::atomic<u32> round{};
    std::jthread writer([&] {
        for (u32 value = 1; value <= Rounds; ++value) {
            while (round.load(std::memory_order_acquire) < value) {
                std::this_thread::yield();
            }
            for (u32 word = 0; word < words.size(); ++word) {
                words[word] = value;
                waiters.OnWrite(0x1000 + word * 4, value);
            }
        }
    });
    std::array<int, 4> owners{};
    for (u32 value = 1; value <= Rounds; ++value) {
        for (u32 word = 0; word < words.size(); ++word) {
            REQUIRE(waiters.Add(&owners[word], 0x1000 + word * 4, value,
                                Nx2SemaphoreWaiters::AcquireGequal));
        }
        round.store(value, std::memory_order_release);
        for (u32 word = 0; word < words.size(); ++word) {
            while (!waiters.Latched(&owners[word])) {
                std::this_thread::yield();
            }
            REQUIRE(words[word] == value);
            waiters.Remove(&owners[word]);
        }
    }
}

TEST_CASE("Pending wakes end hardware and OS waits", "[video_core][nx2_waiters]") {
    using namespace std::chrono_literals;
    Tegra::Nx2PendingWake wake;
    wake.Arm();
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now() + 1ms));
    CHECK_FALSE(wake.Sleep(1ms));

    // A wake between Arm and the wait is not lost.
    wake.Notify();
    CHECK(wake.Poll(std::chrono::steady_clock::now()));
    CHECK(wake.Sleep(5s));
    wake.Arm();
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));

    for (int i = 0; i < 20; ++i) {
        const bool sleep = i % 2 != 0;
        wake.Arm();
        std::jthread notifier([&] {
            std::this_thread::sleep_for(1ms);
            wake.Notify();
        });
        const auto start = std::chrono::steady_clock::now();
        CHECK((sleep ? wake.Sleep(5s) : wake.Poll(start + 5s)));
        CHECK(std::chrono::steady_clock::now() - start < 1s);
    }
}

TEST_CASE("Semaphore waiters see stores to the words they wait on", "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    std::array<u32, 2> words{3, 3};
    const int watched{}, unwatched{};
    REQUIRE(waiters.Add(&watched, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal, &words[0]));
    REQUIRE(waiters.Add(&unwatched, 0x1004, 5, Nx2SemaphoreWaiters::AcquireGequal));
    CHECK_FALSE(waiters.WordStored());

    // Any store that changes the word, met or not, and once per change.
    std::jthread([&] { std::atomic_ref<u32>{words[0]}.store(4); }).join();
    CHECK(waiters.WordStored());
    CHECK_FALSE(waiters.WordStored());
    words[1] = 9;
    CHECK_FALSE(waiters.WordStored());
    words[0] = 5;
    CHECK(waiters.WordStored());

    // A stored word is no longer watched once its waiter leaves or moves to another word.
    REQUIRE(waiters.Add(&watched, 0x1004, 9, Nx2SemaphoreWaiters::AcquireGequal, &words[1]));
    words[0] = 6;
    CHECK_FALSE(waiters.WordStored());
    waiters.Remove(&watched);
    words[1] = 10;
    CHECK_FALSE(waiters.WordStored());
    waiters.Remove(&unwatched);
}

TEST_CASE("The GPU tick a list copy waits for wakes the GPU thread", "[video_core][nx2_waiters]") {
    Tegra::Nx2PendingWake wake;
    wake.Arm();
    wake.OnGpuTick(100);
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));

    wake.WaitForGpuTick(12);
    wake.WaitForGpuTick(10);
    wake.OnGpuTick(9);
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));
    wake.OnGpuTick(10);
    CHECK(wake.Poll(std::chrono::steady_clock::now()));

    // Woken once: the waiter asks again for a tick it still waits for.
    wake.Arm();
    wake.OnGpuTick(12);
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));
    wake.WaitForGpuTick(12);
    wake.OnGpuTick(12);
    CHECK(wake.Poll(std::chrono::steady_clock::now()));
}

TEST_CASE("Latching writes wake the GPU thread", "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    auto& wake = Tegra::Nx2PendingWake::Get();
    const int owner{};
    REQUIRE(waiters.Add(&owner, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal));
    wake.Arm();
    waiters.OnWrite(0x1000, 4);
    waiters.OnWrite(0x2000, 9);
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));
    waiters.OnWrite(0x1000, 5);
    CHECK(wake.Poll(std::chrono::steady_clock::now()));
    waiters.Remove(&owner);
}

TEST_CASE("A puller torn down while it waits leaves the semaphore waiters", "[video_core][nx2_waiters]") {
    // The waiters outlive an emulation session: a channel stopped mid-wait must not leave its
    // slot, whose word points into the session's freed guest memory, to the next GPU thread.
    auto& waiters = Nx2SemaphoreWaiters::Get();
    u32 word = 3;
    {
        Tegra::Engines::Puller puller;
        REQUIRE(waiters.Add(&puller, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal, &word));
        word = 4;
        CHECK(waiters.WordStored());
    }
    word = 5;
    CHECK_FALSE(waiters.WordStored());
}
