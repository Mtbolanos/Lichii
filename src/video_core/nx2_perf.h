// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"

// EXPERIMENT (not for nx2): GPU back-thread optimizations for Ocarina of Time.
// split_overlap is quarantined. Frizti build: the options reviewed as exact rewrites are on by
// default; those that change timing or reuse ideas reverted with zortos293/Lichii#161 (its
// slab of wrong geometry in Kokiri Forest) are off until EDEN_NX2_PERF_ON=<name>,... turns them
// on: frames_ahead, clean_words, index_rebind, submit_budget, fence_flush, fragment_barrier,
// front_sleep, async_pipelines, and bounds (this tree keeps its own range cache).
// EDEN_NX2_PERF_OPT=0 turns every one off for the A/B; EDEN_NX2_PERF_OFF=<name>,... turns off
// only those named (Opt below). EDEN_NX2_PERF_VERIFY=1 computes each optimized answer the old
// way too, logs any difference and uses the old answer.
namespace VideoCommon::Nx2Perf {

enum class Opt : unsigned {
    Bounds,            ///< MaxContinuousRange / GetMemoryLayoutSize from map-time runs
    Method,            ///< Maxwell3D fast method dispatch on the back
    CbBatch,           ///< Consecutive constant buffer words uploaded in one write
    CleanWords,        ///< SynchronizeBuffer skips clean tracker words
    IndexRebind,       ///< Repeated vkCmdBindIndexBuffer skipped
    UsageTracker,      ///< UsageTracker::Track skips pages known to be used
    BankCheck,         ///< Native 64 KiB banks: per-frame set checked first
    SplitOverlap,      ///< Quarantined: pending writes are not a proof of safe early MME reads
    FrontSleep,        ///< The split front sleeps instead of yielding while it waits
    SubmitBudget,      ///< 4096-draw native submission budget outside MoltenVK
    FragmentBarrier,   ///< FragmentBarrier deferred to the next draw that reads the pass
    FenceFlush,        ///< 5 ms DeferFlush, command list copies submitted at once
    FramesAhead,       ///< Ocarina presents a frame ahead
    PipelineRetention, ///< Sixteen sessions of used pipelines, saved every 15 s
    WriteIntersections,
    SortedPageBatches,
    AsyncPipelines,    ///< Native graphics SPIR-V emitted by the pipeline worker, not the back
    UnswizzleWorkers,
    Count,
};

inline constexpr std::array<std::string_view, static_cast<size_t>(Opt::Count)> opt_names{
    "bounds",         "method",          "cb_batch",    "clean_words",   "index_rebind",
    "usage_tracker",  "bank_check",      "split_overlap", "front_sleep", "submit_budget",
    "fragment_barrier", "fence_flush",   "frames_ahead", "pipeline_retention",
    "write_intersections", "sorted_page_batches", "async_pipelines", "unswizzle_workers",
};

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_OPT");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

inline bool On(Opt opt) {
    static const unsigned off = [] {
        const auto bit = [](Opt option) { return 1U << static_cast<unsigned>(option); };
        const auto names = [](const char* variable) {
            unsigned bits = 0;
            std::string_view list = std::getenv(variable) ? std::getenv(variable) : "";
            while (!list.empty()) {
                const size_t comma = list.find(',');
                const std::string_view name = list.substr(0, comma);
                bool known = false;
                for (size_t i = 0; i < opt_names.size(); ++i) {
                    if (name == opt_names[i]) {
                        bits |= 1U << i;
                        known = true;
                    }
                }
                if (!known && !name.empty()) {
                    LOG_WARNING(HW_GPU, "Nx2Perf: unknown name {} in {}", name, variable);
                }
                list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
            }
            return bits;
        };
        // FramesAhead is on: with Ocarina's CPU readbacks preflushed it is worth ~1 fps (48.3 ->
        // 49.5 on the Kokiri porch) and it waits for this frame too whenever the GPU falls behind.
        const unsigned default_off = bit(Opt::CleanWords) |
                                     bit(Opt::IndexRebind) | bit(Opt::SubmitBudget) |
                                     bit(Opt::FenceFlush) | bit(Opt::FragmentBarrier) |
                                     bit(Opt::FrontSleep) | bit(Opt::AsyncPipelines) |
                                     bit(Opt::Bounds);
        unsigned mask = (default_off & ~names("EDEN_NX2_PERF_ON")) | names("EDEN_NX2_PERF_OFF");
        mask |= bit(Opt::SplitOverlap);
        if (const char* value = std::getenv("EDEN_NX2_PERF_FRAMES_AHEAD"); value && value[0] == '0') {
            mask |= bit(Opt::FramesAhead);
        }
        std::string state;
        for (size_t i = 0; i < opt_names.size(); ++i) {
            state += fmt::format(" {}={}", opt_names[i], Enabled() && (mask >> i & 1) == 0 ? "on" : "off");
        }
        LOG_INFO(HW_GPU, "Nx2Perf:{}", state);
        return mask;
    }();
    return Enabled() && (off >> static_cast<unsigned>(opt) & 1) == 0;
}

inline bool Verify() {
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// The frame period the game asks for (nx2:vi2 present word 12 at the speed limit), 0 while
/// unknown or unpaced. The host frame pacer (vk_host_pacer.h) never holds frames longer.
inline std::atomic<long long> g_guest_frame_interval_ns{0};

/// The display's measured refresh period while the host frame pacer runs, else 0. The game's
/// present pacing (nx2_gsv.h) times its frames to whole multiples of it (genlock).
inline std::atomic<long long> g_host_refresh_ns{0};

/// The display's refresh period once the host frame pacer has measured it, paced or not, else
/// 0. Ocarina of Time's high refresh rate cap (nx2_gsv.h) stays at or below it.
inline std::atomic<long long> g_display_refresh_ns{0};

/// Whether the Vulkan device runs the guest's tensor operations on native cooperative matrices
/// (set at device creation). Ocarina's movie upscaler setting picks by it on Automatic.
inline std::atomic<bool> g_native_tensor_ops{false};

/// At 30 FPS and below the host frame pacer presents frames as soon as they are rendered and
/// publishes how long before the display's next refresh the last one was ready (ns), or -1
/// while it isn't pacing that way. The game's present pacing (nx2_gsv.h) moves its frames until
/// that is ~6 ms: each lands on a refresh with little wait (input lag) and evenly spaced.
inline std::atomic<long long> g_host_phase_ns{-1};

/// Host frame times: the interval between frames handed to the swapchain, counted in 0.25 ms
/// buckets (the last holds 63.75 ms and up). The debug server's status returns the counts; the
/// difference between two calls gives that window's percentiles (1% and 0.1% lows).
inline constexpr size_t FrameTimeBuckets = 256;
inline constexpr long long FrameTimeBucketNs = 250'000;
inline std::array<std::atomic<unsigned long long>, FrameTimeBuckets> g_frame_times{};

inline void NoteFrameTime(long long interval_ns) {
    // Longer gaps are loading screens or a paused game, not frames.
    if (interval_ns <= 0 || interval_ns > 1'000'000'000) {
        return;
    }
    const auto bucket =
        (std::min)(static_cast<size_t>(interval_ns / FrameTimeBucketNs), FrameTimeBuckets - 1);
    g_frame_times[bucket].fetch_add(1, std::memory_order_relaxed);
}

} // namespace VideoCommon::Nx2Perf
