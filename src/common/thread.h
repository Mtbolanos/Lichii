// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include "common/common_types.h"
#include "common/polyfill_thread.h"

namespace Common {

class Event {
public:
#ifdef _WIN32
    ~Event();
#endif

    void Set() {
        std::scoped_lock lk{mutex};
        if (!is_set) {
            is_set = true;
#ifdef _WIN32
            if (sleepers != 0) {
                WakeSleeper();
            }
#endif
            condvar.notify_one();
        }
    }

    void Wait() {
        std::unique_lock lk{mutex};
        condvar.wait(lk, [&] { return is_set.load(); });
        is_set = false;
    }

    bool WaitFor(const std::chrono::nanoseconds time);

    /// Like WaitFor, but always blocks in the OS. On Windows, WaitFor keeps the thread running
    /// in a hardware wait; this sleeps on a high-resolution waitable timer instead.
    bool SleepFor(const std::chrono::nanoseconds time);

    /// Like WaitFor, but never blocks in the OS: waits in WaitForFlag, so a wake ends within
    /// microseconds of the deadline. For short waits; the thread counts as running throughout.
    bool SpinFor(const std::chrono::nanoseconds time);

    template<class Clock, class Duration>
    bool WaitUntil(const std::chrono::time_point<Clock, Duration> time) {
        std::unique_lock lk{mutex};
        if (!condvar.wait_until(lk, time, [this] { return is_set.load(); }))
            return false;
        is_set = false;
        return true;
    }

    void Reset() {
        std::unique_lock lk{mutex};
        // no other action required, since wait loops on the predicate and any lingering signal will
        // get cleared on the first iteration
        is_set = false;
    }

    [[nodiscard]] bool IsSet() const {
        return is_set;
    }

private:
#ifdef _WIN32
    void WakeSleeper();
#endif

    alignas(64) std::atomic<bool> is_set{false};
    std::condition_variable condvar;
    std::mutex mutex;
#ifdef _WIN32
    void* sleep_wake{};
    u32 sleepers{};
#endif
};

class Barrier {
public:
    explicit Barrier(std::size_t count_) : count(count_) {}

    /// Blocks until all "count" threads have called Sync()
    bool Sync(std::stop_token token = {}) {
        std::unique_lock lk{mutex};
        const std::size_t current_generation = generation;

        if (++waiting == count) {
            generation++;
            waiting = 0;
            condvar.notify_all();
            return true;
        } else {
            condvar.wait(lk, token,
                        [this, current_generation] { return current_generation != generation; });
            return !token.stop_requested();
        }
    }

private:
    std::condition_variable_any condvar;
    std::mutex mutex;
    std::size_t count;
    std::size_t waiting = 0;
    std::size_t generation = 0; // Incremented once each time the barrier is used
};

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

enum class ThreadPlacement : u32 {
    Default = 0,
    Background = 1,
    Efficiency = 2,
};

void SetCurrentThreadPriority(ThreadPriority new_priority);
void SetCurrentThreadName(const char* name);
void SetCurrentThreadToPerformanceCores();
void SetCurrentThreadToEfficiencyCores();
void SetCurrentThreadToBackgroundWork();
void SetCurrentThreadToAllCores();

enum class PinnedRole : u32 { GpuFront, GpuBack, GuestCore };

/// The experimental pin_gpu_threads setting (Windows and Linux): the split GPU thread's front and
/// back each get a physical performance core of their own and the guest cores keep off those two.
/// Skipped, with one log line, on CPUs with too few cores, with EDEN_HYBRID_CORES=0 (Windows), and
/// on other hosts.
void PinCurrentThread(PinnedRole role);

void RefreshThreadPolicies();
void PollThreadPolicies();

/// How long to poll before blocking next time, after a wait of `waited` that polled `poll`, with
/// `limit` the longest poll. This is the cpuidle-haltpoll governor's rule (Linux,
/// drivers/cpuidle/governors/haltpoll.c): a wait a longer poll would have caught doubles the poll,
/// and one longer than the limit, which no poll catches, halves it. The poll never drops below an
/// eighth of the limit, so the quickest wake-ups stay cheap.
constexpr std::chrono::nanoseconds NextPollWindow(std::chrono::nanoseconds poll,
                                                  std::chrono::nanoseconds waited,
                                                  std::chrono::nanoseconds limit) {
    if (waited <= poll) {
        return poll;
    }
    if (waited <= limit) {
        return std::min(poll * 2, limit);
    }
    return std::max(poll / 2, limit / 8);
}

/// Waits until flag is true or the deadline passes, and returns the flag. Where the CPU has a
/// user-mode wait on an address (MWAITX on AMD, UMWAIT on Intel) the thread waits in it: a store to
/// the flag's cache line ends the wait within a few hundred nanoseconds, and the core neither
/// issues instructions nor enters the OS meanwhile. Other CPUs yield until then.
bool WaitForFlag(const std::atomic<bool>& flag, std::chrono::steady_clock::time_point deadline);

} // namespace Common
