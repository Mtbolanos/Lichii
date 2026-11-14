// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/kernel/nx2_thread_stats.h"

namespace Stats = Kernel::Nx2ThreadStats;

TEST_CASE("Guest thread stats count the gaps of a counter-polling loop", "[nx2][kernel]") {
    constexpr u64 Spin = 52;
    Stats::RunTally run{};
    Stats::CountRead(run, 1000, Spin);
    Stats::CountRead(run, 1005, Spin);
    Stats::CountRead(run, 1040, Spin);
    Stats::CountRead(run, 2000, Spin);
    CHECK(run.reads == 4);
    CHECK(run.poll_reads == 2);
    CHECK(run.poll_ticks == 40);
    CHECK(run.tight_poll_ticks == 5);
}

TEST_CASE("Guest thread stats count hints by type and the time between close ones",
          "[nx2][kernel]") {
    constexpr u64 Spin = 52;
    Stats::RunTally run{};
    Stats::CountHint(run, 4, 500, Spin);
    Stats::CountHint(run, 4, 510, Spin);
    Stats::CountHint(run, 1, 900, Spin);
    CHECK(run.hints[4] == 2);
    CHECK(run.hints[1] == 1);
    CHECK(run.hint_ticks == 10);
}

TEST_CASE("Guest thread stats split read-backs from other cached memory calls", "[nx2][kernel]") {
    Stats::RunTally run{};
    Stats::CountMemory(run, 7, false);
    Stats::CountMemory(run, 300, true);
    CHECK(run.memory_calls == 2);
    CHECK(run.memory_ticks == 307);
    CHECK(run.read_backs == 1);
    CHECK(run.read_back_ticks == 300);
}

TEST_CASE("Guest thread stats name the wait of each blocking SVC", "[nx2][kernel]") {
    static_assert(Stats::WaitOf(0x21) == Stats::Wait::Ipc);
    static_assert(Stats::WaitOf(0x18) == Stats::Wait::SyncObjects);
    static_assert(Stats::WaitOf(0x34) == Stats::Wait::Arbiter);
    static_assert(Stats::WaitOf(0x1C) == Stats::Wait::CondVar);
    static_assert(Stats::WaitOf(0x1A) == Stats::Wait::Lock);
    static_assert(Stats::WaitOf(0x0B) == Stats::Wait::Sleep);
    CHECK(Stats::WaitOf(0x43) == Stats::Wait::Other);
}
