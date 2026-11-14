// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"
#include "common/logging.h"

// Async read-back: a guest CPU read of GPU-written buffer memory (GPU::OnCPURead) has the GPU
// thread submit the downloads and go on. The reading thread waits for that submission and writes
// the bytes to guest memory (BufferCache::FinishReadDownloads). Pages the guest reads after each
// GPU write get their download right behind the write. A macro DMA read on the split's front
// waits only for the queued work that can write its words (Maxwell3D::FrontDmaRead,
// GpuSplit::WaitForWrites); macros whose code reads memory into their FIFO run on the back
// instead (Maxwell3D::FrontRunOnBack).
//
// Diagnostics, off unless set:
// - EDEN_NX2_READ_CHECK=n: 1 in n read downloads (1 checks them all) is compared with a
//   synchronous flush's copy taken at the same point of the GPU's work
//   (BufferCache::CheckReadDownload).
// - EDEN_NX2_DMA_READ_VERIFY: every macro DMA read made without draining the split drains it
//   too and is compared (CheckDmaRead).
namespace VideoCore::AsyncReadback {

/// EDEN_NX2_READ_CHECK's value: unset, 0 or not a number checks none.
[[nodiscard]] inline u64 ParseReadCheck(const char* value) {
    return value == nullptr ? 0 : static_cast<u64>(std::strtoull(value, nullptr, 10));
}

inline std::atomic<u64> g_read_check_every{ParseReadCheck(std::getenv("EDEN_NX2_READ_CHECK"))};
inline std::atomic<bool> g_dma_read_verify{std::getenv("EDEN_NX2_DMA_READ_VERIFY") != nullptr};

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_read_check_every.store(ParseReadCheck(std::getenv("EDEN_NX2_READ_CHECK")),
                             std::memory_order_relaxed);
    g_dma_read_verify.store(std::getenv("EDEN_NX2_DMA_READ_VERIFY") != nullptr,
                            std::memory_order_relaxed);
}

/// EDEN_NX2_READ_CHECK: check 1 in this many read downloads, 0 for none.
[[nodiscard]] inline u64 ReadCheckEvery() {
    return g_read_check_every.load(std::memory_order_relaxed);
}

[[nodiscard]] inline bool DmaReadVerify() {
    return g_dma_read_verify.load(std::memory_order_relaxed);
}

/// Who wrote a read download to guest memory, named by the read check's mismatch lines.
enum class Landing : u8 {
    Reader,       ///< The guest thread whose read waited for it (GPU::OnCPURead)
    GpuSideWrite, ///< A write to guest memory from the GPU's side over its bytes
    Completed,    ///< Whichever thread found the host GPU had run it, before a read waited
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Landing::Count)> landing_names{
    "the reading thread",
    "a GPU-side write over it",
    "a thread that found it run by the host GPU",
};

/// Set on a guest thread while GPU::OnCPURead writes its downloads to guest memory.
inline thread_local bool t_reading{};
/// The reads this thread waited for (EDEN_NX2_GUEST_THREAD_STATS tells read-backs apart by it).
inline thread_local u64 t_reads{};

[[nodiscard]] inline u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

enum class Read : u8 {
    Async,    ///< The GPU thread submitted the downloads, the reading thread waited for them
    InFlight, ///< An earlier read's or an early download covered it: no flush, only the wait
    Count,
};

/// Totals for the per-second line (EDEN_NX2_GPU_PROFILE), from every thread.
struct Counters {
    std::array<std::atomic<u64>, static_cast<size_t>(Read::Count)> reads{};
    /// From the read's request to its bytes in guest memory
    std::array<std::atomic<u64>, static_cast<size_t>(Read::Count)> read_ns{};
    /// The GPU thread's time in the reads' flushes
    std::atomic<u64> flush_ns{};
    /// Reading threads whose download the host GPU had not run yet
    std::atomic<u64> reader_waits{};
    std::atomic<u64> reader_wait_ns{};
    std::atomic<u64> reader_max_wait_ns{};
    /// Writes from the GPU's side that waited for the host GPU to run a read download under them
    std::atomic<u64> gpu_side_waits{};
    std::atomic<u64> gpu_side_wait_ns{};
    /// Macro DMA reads on the split's front (Maxwell3D::FrontDmaRead), those that waited for its
    /// back and those that drained it, and the time they waited
    std::atomic<u64> dma_reads{};
    std::atomic<u64> dma_waited{};
    std::atomic<u64> dma_drained{};
    std::atomic<u64> dma_wait_ns{};
    /// Landings BufferCache::ClearDownload forced on the GPU thread before a write guest memory
    /// doesn't hold yet (a DMA copy, a discarded write), those that waited for the host GPU (also
    /// among gpu_side_waits) and the time they waited
    std::atomic<u64> clear_landings{};
    std::atomic<u64> clear_waits{};
    std::atomic<u64> clear_wait_ns{};
    /// Macros that read memory, run on the split's back in its front's place
    /// (Maxwell3D::FrontRunOnBack): the calls and the back's time in them, their DMA reads, those
    /// that found GPU-modified words to flush and the time the reads spent writing the caches out
    /// (part of the calls' time)
    std::atomic<u64> back_macros{};
    std::atomic<u64> back_macro_ns{};
    std::atomic<u64> back_dma_reads{};
    std::atomic<u64> back_dma_flushes{};
    std::atomic<u64> back_dma_flush_ns{};
    /// The front's waits for such macros to run: for registers they wrote, for shadow RAM entries
    /// they set, and for those sent before the macro code changes
    std::atomic<u64> back_waits{};
    std::atomic<u64> back_wait_ns{};
    std::atomic<u64> window_start{};
};

inline Counters g_counters;

inline void Add(std::atomic<u64>& counter, u64 amount) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

inline void CountRead(Read kind, u64 ns) {
    ++t_reads;
    Add(g_counters.reads[static_cast<size_t>(kind)], 1);
    Add(g_counters.read_ns[static_cast<size_t>(kind)], ns);
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

inline void CountDmaRead(bool waited, bool drained, u64 ns) {
    Add(g_counters.dma_reads, 1);
    Add(g_counters.dma_waited, waited ? 1 : 0);
    Add(g_counters.dma_drained, drained ? 1 : 0);
    Add(g_counters.dma_wait_ns, ns);
}

/// BufferCache::ClearDownload landed read downloads, waiting ns for the host GPU (0: their copies
/// had run).
inline void CountClearLanding(u64 ns) {
    Add(g_counters.clear_landings, 1);
    Add(g_counters.clear_waits, ns != 0 ? 1 : 0);
    Add(g_counters.clear_wait_ns, ns);
}

/// The split's back ran a macro in its front's place for ns.
inline void CountBackMacro(u64 ns) {
    Add(g_counters.back_macros, 1);
    Add(g_counters.back_macro_ns, ns);
}

/// A DMA read of such a macro spent ns writing the caches out, flushing GPU-modified words.
inline void CountBackDmaRead(bool flushed, u64 ns) {
    Add(g_counters.back_dma_reads, 1);
    Add(g_counters.back_dma_flushes, flushed ? 1 : 0);
    Add(g_counters.back_dma_flush_ns, ns);
}

/// The front waited ns for such a macro to run.
inline void CountBackWait(u64 ns) {
    Add(g_counters.back_waits, 1);
    Add(g_counters.back_wait_ns, ns);
}

/// Logs the totals every 10 s and resets them, on whichever thread counting comes first.
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
    const u64 dma_reads = take(c.dma_reads);
    const u64 dma_waited = take(c.dma_waited);
    const u64 dma_drained = take(c.dma_drained);
    const u64 dma_wait_ns = take(c.dma_wait_ns);
    const u64 clear_landings = take(c.clear_landings);
    const u64 clear_waits = take(c.clear_waits);
    const u64 clear_wait_ns = take(c.clear_wait_ns);
    const u64 back_macros = take(c.back_macros);
    const u64 back_macro_ns = take(c.back_macro_ns);
    const u64 back_dma_reads = take(c.back_dma_reads);
    const u64 back_dma_flushes = take(c.back_dma_flushes);
    const u64 back_dma_flush_ns = take(c.back_dma_flush_ns);
    const u64 back_waits = take(c.back_waits);
    const u64 back_wait_ns = take(c.back_wait_ns);
    constexpr auto async = static_cast<size_t>(Read::Async);
    constexpr auto in_flight = static_cast<size_t>(Read::InFlight);
    LOG_INFO(HW_GPU,
             "Async read-back per second: reads {:.0f} async ({:.1f} ms), {:.0f} only waited for a "
             "download on its way ({:.1f} ms) | GPU thread in read flushes {:.1f} ms | readers "
             "waited on their fence "
             "{:.0f} times, {:.1f} ms (longest {:.2f} ms) | GPU-side writes fell back to a host "
             "GPU wait {:.0f} times, {:.1f} ms | macro DMA reads {:.0f}, {:.0f} waited for the "
             "split's back ({:.1f} ms), {:.0f} drained it | GPU thread landed read downloads before "
             "writes guest memory doesn't hold {:.0f} times, {:.0f} waited for the host GPU "
             "({:.1f} ms) | split's back ran {:.0f} macros that read memory ({:.1f} ms), their DMA "
             "reads {:.0f}, {:.0f} flushed GPU-modified words ({:.1f} ms writing caches out) | "
             "front waited for them {:.0f} times ({:.1f} ms)",
             per_second(reads[async]), ms_per_second(read_ns[async]),
             per_second(reads[in_flight]), ms_per_second(read_ns[in_flight]),
             ms_per_second(flush_ns), per_second(reader_waits), ms_per_second(reader_wait_ns),
             static_cast<double>(reader_max_wait_ns) / 1e6, per_second(gpu_side_waits),
             ms_per_second(gpu_side_wait_ns), per_second(dma_reads), per_second(dma_waited),
             ms_per_second(dma_wait_ns), per_second(dma_drained), per_second(clear_landings),
             per_second(clear_waits), ms_per_second(clear_wait_ns), per_second(back_macros),
             ms_per_second(back_macro_ns), per_second(back_dma_reads), per_second(back_dma_flushes),
             ms_per_second(back_dma_flush_ns), per_second(back_waits), ms_per_second(back_wait_ns));
}

} // namespace VideoCore::AsyncReadback
