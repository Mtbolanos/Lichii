// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include "common/common_types.h"

namespace Kernel {
class KThread;
}

// EDEN_NX2_GUEST_THREAD_STATS=1: every 10 s, per frame, the guest threads with the most time on a
// core and where it went (game code in the JIT, counter polling, hint spins, GPU-cached memory,
// SVCs), how long each was ready without a core and what it waited for; per core, why the JIT
// returned and what its dispatcher, counter and hints did. Times are guest counter ticks
// (CoreTiming::GetClockTicks) until the report converts them.
namespace Kernel::Nx2ThreadStats {

inline const bool g_enabled = std::getenv("EDEN_NX2_GUEST_THREAD_STATS") != nullptr;

enum class Wait : u8 { Ipc, SyncObjects, Arbiter, CondVar, Lock, Sleep, Other, Count };
inline constexpr size_t Waits = static_cast<size_t>(Wait::Count);

/// Why the JIT returned to the kernel.
enum class Exit : u8 { Svc, Interrupt, CacheInvalidation, Other, Count };
inline constexpr size_t Exits = static_cast<size_t>(Exit::Count);

/// WFI, WFE, SEV, SEVL and YIELD, in Dynarmic::A64::Exception's order.
inline constexpr size_t Hints = 5;

/// The wait a thread that blocks in this SVC makes.
constexpr Wait WaitOf(u32 svc) {
    switch (svc) {
    case 0x0B:
        return Wait::Sleep;
    case 0x18:
        return Wait::SyncObjects;
    case 0x1A:
        return Wait::Lock;
    case 0x1C:
        return Wait::CondVar;
    case 0x21:
    case 0x22:
    case 0x23:
        return Wait::Ipc;
    case 0x34:
        return Wait::Arbiter;
    default:
        return Wait::Other;
    }
}

/// What the JIT's callbacks counted in the current Run() of this host thread. BeginRun clears
/// it, EndRun adds it to the thread's and the core's totals.
struct RunTally {
    u64 last_read;        ///< The previous counter read in this Run(), 0 before the first
    u64 last_hint;        ///< The previous hint, 0 before the first
    u64 reads;            ///< Counter reads
    u64 poll_reads;       ///< Of those: read within the spin window of the previous one
    u64 poll_ticks;       ///< The time between such reads: a loop polling the counter
    u64 tight_poll_ticks; ///< ...between reads under a seventh of the window (0.24 us) apart
    std::array<u64, Hints> hints;
    u64 hint_ticks;      ///< The time between hints within the spin window of each other
    u64 memory_ticks;    ///< In reads and writes of GPU-cached guest memory
    u64 memory_calls;    ///< ...and how many
    u64 read_back_ticks; ///< Of those: the calls that waited for a GPU download
    u64 read_backs;
};

inline thread_local RunTally t_run{};

constexpr void CountRead(RunTally& run, u64 ticks, u64 spin_ticks) {
    ++run.reads;
    const u64 gap = ticks - run.last_read;
    if (run.last_read != 0 && gap < spin_ticks) {
        ++run.poll_reads;
        run.poll_ticks += gap;
        if (gap < spin_ticks / 7) {
            run.tight_poll_ticks += gap;
        }
    }
    run.last_read = ticks;
}

constexpr void CountHint(RunTally& run, size_t hint, u64 ticks, u64 spin_ticks) {
    ++run.hints[hint];
    const u64 gap = ticks - run.last_hint;
    if (run.last_hint != 0 && gap < spin_ticks) {
        run.hint_ticks += gap;
    }
    run.last_hint = ticks;
}

constexpr void CountMemory(RunTally& run, u64 ticks, bool read_back) {
    run.memory_ticks += ticks;
    ++run.memory_calls;
    if (read_back) {
        run.read_back_ticks += ticks;
        ++run.read_backs;
    }
}

inline void BeginRun() {
    t_run = {};
}

/// The Run() of `thread` on `core` from `start` to `end` returned for `exit` (`svc` when it
/// is an SVC). The core's JIT totals so far: C++ block lookups, blocks compiled, full code cache
/// clears.
void EndRun(KThread* thread, size_t core, u64 start, u64 end, Exit exit, u32 svc,
            u64 block_lookups, u64 blocks_compiled, u64 cache_clears);

/// KScheduler::SwitchThread: `from` ran `from_ticks` on its core until `now`.
void OnSwitch(KThread* from, KThread* to, s64 now, s64 from_ticks);
/// KScheduler::OnThreadStateChanged, under the scheduler lock.
void OnStateChanged(KThread* thread, s64 now, bool was_waiting, bool now_runnable,
                    bool now_waiting);
/// The client thread of a PresentClientBuffer request.
void OnPresent(KThread* client);
/// HoldHleReply kept the reply to `client` back for `ns`.
void AddHold(KThread* client, s64 ns);

} // namespace Kernel::Nx2ThreadStats
