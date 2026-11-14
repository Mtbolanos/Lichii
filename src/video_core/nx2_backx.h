// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "common/common_types.h"
#include "common/logging.h"

/// The GPU back thread's work outside draw preparation: locks and passes it skips where a check
/// without the lock shows the locked work would change nothing (ported from jpkto/Lichii
/// 7a5e5357cb and f5edeabdbb). Guest CPU write traps hold the buffer cache's lock across page
/// protection calls, so every acquisition the back makes can meet one and spin.
namespace VideoCommon::BackX {

/// EDEN_NX2_BACKX_VERIFY=1: every skip also runs the locked check it replaces, logs
/// "BackX verify mismatch" where that check would have done something, and then does it. The
/// GPU dirty memory managers count their collects (Core::GPUDirtyMemoryManager::count_collects).
inline bool g_verify{};

/// Called by VideoCore::CreateGPU, before the caches exist.
inline void ReadSwitch() {
    const char* const value = std::getenv("EDEN_NX2_BACKX_VERIFY");
    g_verify = value != nullptr && value[0] == '1';
    if (g_verify) {
        LOG_INFO(HW_GPU, "BackX verify on");
    }
}

/// Verify: a skip the locked check contradicts (logged 64 times at most).
inline void Mismatch(const char* what, u64 address, u64 size) {
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_ERROR(HW_GPU, "BackX verify mismatch: {} at {:#x}+{:#x}", what, address, size);
    }
}

/// Verify of a gather InvalidateGPUCache skipped: `found` when the gather found writes,
/// `collected` when a manager collected one after the skip's check (a race the next call covers
/// either way). Writes found with nothing collected since are a miss. Logs the counts every 10 s.
inline void SkippedGather(bool found, bool collected, std::pair<u64, u64> first_write) {
    static std::mutex mutex;
    static u64 skips{};
    static u64 races{};
    static u64 misses{};
    static auto last_log = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++skips;
    if (found && collected) {
        ++races;
    } else if (found) {
        ++misses;
        Mismatch("collected CPU write while no manager showed one", first_write.first,
                 first_write.second);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log < std::chrono::seconds{10}) {
        return;
    }
    LOG_INFO(HW_GPU,
             "BackX gather checks in 10 s: {} skipped gathers, {} found writes collected after "
             "the check, {} found writes no manager showed (must stay 0)",
             skips, races, misses);
    skips = races = misses = 0;
    last_log = now;
}

} // namespace VideoCommon::BackX
