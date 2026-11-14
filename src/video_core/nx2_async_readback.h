// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"

// EDEN_NX2_EXP49, async read-back (port of zortos293/Lichii#1028's core, 0b83798bf6): a guest CPU
// read of GPU-written buffer memory (GPU::OnCPURead) has the GPU thread record and submit the
// downloads and go on. The reading thread waits for that submission and writes the bytes to guest
// memory itself (BufferCache::FinishReadDownloads). Before, the GPU thread waited for the host
// GPU (Scheduler::Finish) with the buffer cache's lock held, and the reader waited for it.
//
// Not ported: #1028's early downloads (pages read after each GPU write downloaded right behind the
// write), its EDEN_NX2_READ_CHECK verify mode and the macro DMA read changes.
namespace VideoCore::AsyncReadback {

/// Set on a guest thread while GPU::OnCPURead writes its downloads to guest memory.
inline thread_local bool t_reading{};

[[nodiscard]] inline u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

enum class Read : u8 {
    Sync,     ///< EXP49 off: the GPU thread downloaded and waited for the host GPU
    Async,    ///< The GPU thread submitted the downloads, the reading thread waited for them
    InFlight, ///< An earlier read's downloads covered it: no flush, only the wait
    Count,
};

/// Totals for the 10-second line, from every thread.
struct Counters {
    std::array<std::atomic<u64>, static_cast<size_t>(Read::Count)> reads{};
    /// From the read's request to its bytes in guest memory
    std::array<std::atomic<u64>, static_cast<size_t>(Read::Count)> read_ns{};
    /// The GPU thread's time in the reads' flushes (both modes)
    std::atomic<u64> flush_ns{};
    /// Reading threads whose download the host GPU had not run yet
    std::atomic<u64> reader_waits{};
    std::atomic<u64> reader_wait_ns{};
    std::atomic<u64> reader_max_wait_ns{};
    /// Writes from the GPU's side that waited for the host GPU to run a read download under them
    std::atomic<u64> gpu_side_waits{};
    std::atomic<u64> gpu_side_wait_ns{};
    /// Landings BufferCache::ClearDownload forced before a write guest memory doesn't hold yet (a
    /// DMA copy or clear, a discarded or inline write), and those that waited for the host GPU
    std::atomic<u64> clear_landings{};
    std::atomic<u64> clear_waits{};
    std::atomic<u64> clear_wait_ns{};
    std::atomic<u64> window_start{};
};

inline Counters g_counters;

inline void Add(std::atomic<u64>& counter, u64 amount) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

inline void CountRead(Read kind, u64 ns) {
    Add(g_counters.reads[static_cast<size_t>(kind)], 1);
    Add(g_counters.read_ns[static_cast<size_t>(kind)], ns);
    Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::ReadBack, ns);
}

/// BufferCache::FinishReadDownloads waited ns for the host GPU.
inline void CountHostWait(u64 ns) {
    if (!t_reading) {
        Add(g_counters.gpu_side_waits, 1);
        Add(g_counters.gpu_side_wait_ns, ns);
        return;
    }
    Add(g_counters.reader_waits, 1);
    Add(g_counters.reader_wait_ns, ns);
    u64 max = g_counters.reader_max_wait_ns.load(std::memory_order_relaxed);
    while (ns > max && !g_counters.reader_max_wait_ns.compare_exchange_weak(
                           max, ns, std::memory_order_relaxed)) {
    }
}

/// BufferCache::ClearDownload landed read downloads, waiting ns for the host GPU (0: their copies
/// had run).
inline void CountClearLanding(u64 ns) {
    Add(g_counters.clear_landings, 1);
    Add(g_counters.clear_waits, ns != 0 ? 1 : 0);
    Add(g_counters.clear_wait_ns, ns);
}

/// Logs the totals every 10 s and resets them, on whichever reading thread comes first.
inline void MaybeReport() {
    auto& c = g_counters;
    const u64 now = Now();
    u64 start = c.window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        c.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
        return;
    }
    if (now - start < 10'000'000'000ULL ||
        !c.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
        return;
    }
    const double window = static_cast<double>(now - start);
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const auto per_second = [window](u64 value) {
        return static_cast<double>(value) * 1e9 / window;
    };
    const auto ms_per_second = [window](u64 ns) { return static_cast<double>(ns) * 1e3 / window; };
    std::array<u64, static_cast<size_t>(Read::Count)> reads{};
    std::array<u64, static_cast<size_t>(Read::Count)> read_ns{};
    for (size_t i = 0; i < reads.size(); ++i) {
        reads[i] = take(c.reads[i]);
        read_ns[i] = take(c.read_ns[i]);
    }
    const u64 flush_ns = take(c.flush_ns);
    const u64 reader_waits = take(c.reader_waits);
    const u64 reader_wait_ns = take(c.reader_wait_ns);
    const u64 reader_max_wait_ns = take(c.reader_max_wait_ns);
    const u64 gpu_side_waits = take(c.gpu_side_waits);
    const u64 gpu_side_wait_ns = take(c.gpu_side_wait_ns);
    const u64 clear_landings = take(c.clear_landings);
    const u64 clear_waits = take(c.clear_waits);
    const u64 clear_wait_ns = take(c.clear_wait_ns);
    constexpr auto sync = static_cast<size_t>(Read::Sync);
    constexpr auto async = static_cast<size_t>(Read::Async);
    constexpr auto in_flight = static_cast<size_t>(Read::InFlight);
    LOG_INFO(HW_GPU,
             "Read-back (EXP49) per second: reads {:.0f} sync ({:.1f} ms), {:.0f} async ({:.1f} "
             "ms), {:.0f} only waited for a download on its way ({:.1f} ms) | GPU thread in read "
             "flushes {:.1f} ms | readers waited for the host GPU {:.0f} times, {:.1f} ms "
             "(longest {:.2f} ms) | GPU-side writes waited for a read download {:.0f} times, "
             "{:.1f} ms | GPU thread landed read downloads before a write {:.0f} times, {:.0f} "
             "waited ({:.1f} ms)",
             per_second(reads[sync]), ms_per_second(read_ns[sync]), per_second(reads[async]),
             ms_per_second(read_ns[async]), per_second(reads[in_flight]),
             ms_per_second(read_ns[in_flight]), ms_per_second(flush_ns), per_second(reader_waits),
             ms_per_second(reader_wait_ns), static_cast<double>(reader_max_wait_ns) / 1e6,
             per_second(gpu_side_waits), ms_per_second(gpu_side_wait_ns),
             per_second(clear_landings), per_second(clear_waits), ms_per_second(clear_wait_ns));
}

} // namespace VideoCore::AsyncReadback
