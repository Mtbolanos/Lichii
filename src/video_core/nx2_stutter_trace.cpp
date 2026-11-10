// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <fstream>
#endif

#include "common/logging.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_stutter_trace.h"

namespace VideoCommon::StutterTrace {
namespace {

constexpr double StutterMs = 33.0;
constexpr u64 SummaryNs = 10'000'000'000ULL;
/// 1 ms bins; the last one holds every longer frame.
constexpr size_t HistogramBins = 300;
constexpr auto GraphicsMissIndex = static_cast<size_t>(Category::GraphicsMiss);
constexpr auto ComputeMissIndex = static_cast<size_t>(Category::ComputeMiss);
constexpr auto IdleIndex = static_cast<size_t>(Category::Idle);
constexpr std::array<std::string_view, ThreadCount> thread_names{"front", "back", "worker"};
constexpr size_t ZoneCount = static_cast<size_t>(Nx2Profile::Zone::Count);
/// Site runs at least this long are listed one by one in the frame's detail line.
constexpr double LongSiteMs = 2.0;
constexpr size_t MaxLongSites = 256;
constexpr size_t DetailZones = 8;
constexpr std::array<std::string_view, LookupCount> lookup_names{
    "gfx_prebuilt",     "gfx_background", "gfx_recorded", "gfx_new",
    "compute_prebuilt", "compute_new",    "program_hit",  "program_miss",
    "program_stale",
};
constexpr std::array<std::string_view, DriverBuildCount> driver_build_names{
    "binary", "compiled", "rejected",
};


/// Windows: commit (PagefileUsage) and its peak; Linux: resident set and its peak.
std::string MemoryFields() {
    constexpr double MiB = 1024.0 * 1024.0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return fmt::format("commit_mib={:.1f} peak_commit_mib={:.1f}",
                           static_cast<double>(counters.PagefileUsage) / MiB,
                           static_cast<double>(counters.PeakPagefileUsage) / MiB);
    }
#elif defined(__linux__)
    std::ifstream status{"/proc/self/status"};
    std::string key;
    u64 rss_kib{};
    u64 peak_kib{};
    while (status >> key) {
        if (key == "VmRSS:") {
            status >> rss_kib;
        } else if (key == "VmHWM:") {
            status >> peak_kib;
        }
    }
    if (rss_kib != 0) {
        return fmt::format("rss_mib={:.1f} peak_rss_mib={:.1f}",
                           static_cast<double>(rss_kib) * 1024.0 / MiB,
                           static_cast<double>(peak_kib) * 1024.0 / MiB);
    }
#endif
    return "memory=unavailable";
}

using ZoneTicks = std::array<std::array<u64, ZoneSlots>, ThreadCount>;

struct LongSite {
    Site site;
    std::string_view thread;
    u64 start;
    u64 end;
};

std::mutex g_long_sites_mutex;
std::vector<LongSite> g_long_sites;
/// Ticks per millisecond, measured by the frames (0 until the second frame).
std::atomic<double> g_ticks_per_ms{0.0};

struct FrameState {
    u64 frames{};
    double detail_ms{50.0};
    ZoneTicks zone_ticks{};
    ZoneTicks zone_counts{};
    std::array<u64, SiteCount> site_ticks{};
    std::array<u64, SiteCount> site_counts{};
    std::array<double, SiteCount> window_site_ms{};
    std::array<u64, SiteCount> window_site_counts{};
    u64 last_ns{};
    u64 last_ticks{};
    ThreadTicks ticks{};
    ThreadTicks counts{};
    u64 window_start_ns{};
    u64 window_frames{};
    u64 window_over_33{};
    u64 window_over_100{};
    std::array<u32, HistogramBins> histogram{};
    std::array<double, CategoryCount> window_ms{};
    u64 window_graphics_builds{};
    u64 window_compute_builds{};
    std::array<u64, LookupCount> lookups{};
    std::array<u64, DriverBuildCount> driver_builds{};
};

FrameState g_frame;
std::mutex g_cache_file_mutex;
std::string g_cache_file;

std::string CacheFileBytes() {
    std::scoped_lock lock{g_cache_file_mutex};
    std::error_code error;
    const auto size = g_cache_file.empty() ? 0 : std::filesystem::file_size(g_cache_file, error);
    return g_cache_file.empty() || error ? std::string{"na"} : std::to_string(size);
}

void Summarize(FrameState& state, u64 now) {
    std::string histogram;
    for (size_t bin = 0; bin < state.histogram.size(); ++bin) {
        if (state.histogram[bin] != 0) {
            histogram += fmt::format("{}{}:{}", histogram.empty() ? "" : ",", bin,
                                     state.histogram[bin]);
        }
    }
    std::string categories;
    for (size_t index = 0; index < IdleIndex; ++index) {
        categories += fmt::format("{}{}:{:.1f}", index == 0 ? "" : ",", category_names[index],
                                  state.window_ms[index]);
    }
    std::string sites;
    for (size_t index = 0; index < SiteCount; ++index) {
        if (state.window_site_counts[index] != 0 || state.window_site_ms[index] != 0.0) {
            sites += fmt::format("{}{}:{}x{:.1f}", sites.empty() ? "" : ",", site_names[index],
                                 state.window_site_counts[index], state.window_site_ms[index]);
        }
    }
    std::string lookups;
    for (size_t index = 0; index < LookupCount; ++index) {
        const u64 total = g_totals.lookups[index].load(std::memory_order_relaxed);
        lookups += fmt::format("{}{}:{}", index == 0 ? "" : ",", lookup_names[index],
                               total - state.lookups[index]);
        state.lookups[index] = total;
    }
    std::string driver_builds;
    for (size_t index = 0; index < DriverBuildCount; ++index) {
        const u64 total = g_totals.driver_builds[index].load(std::memory_order_relaxed);
        driver_builds += fmt::format("{}{}:{}", index == 0 ? "" : ",", driver_build_names[index],
                                     total - state.driver_builds[index]);
        state.driver_builds[index] = total;
    }
    LOG_INFO(HW_GPU,
             "Nx2StutterSummary window_s={:.2f} frames={} over33={} over100={} hist={} cat_ms={} "
             "builds=gfx:{},compute:{} lookups={} driver_builds={} cache_bytes={} sites={}",
             static_cast<double>(now - state.window_start_ns) / 1e9, state.window_frames,
             state.window_over_33, state.window_over_100, histogram.empty() ? "none" : histogram,
             categories, state.window_graphics_builds, state.window_compute_builds, lookups,
             driver_builds, CacheFileBytes(), sites.empty() ? "none" : sites);
    state.window_start_ns = now;
    state.window_frames = 0;
    state.window_over_33 = 0;
    state.window_over_100 = 0;
    state.histogram = {};
    state.window_ms = {};
    state.window_graphics_builds = 0;
    state.window_compute_builds = 0;
    state.window_site_ms = {};
    state.window_site_counts = {};
}

/// Each thread's zone ticks at now, its open zone's time so far included, and zone closes.
std::pair<ZoneTicks, ZoneTicks> SnapshotZones(u64 now) {
    ZoneTicks ticks;
    ZoneTicks counts;
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        auto& sequence = g_totals.sequence[thread];
        for (;;) {
            const u32 begin = sequence.load(std::memory_order_acquire);
            if (begin % 2 != 0) {
                std::this_thread::yield();
                continue;
            }
            for (size_t zone = 0; zone < ZoneSlots; ++zone) {
                ticks[thread][zone] = g_totals.zone_ticks[thread][zone].load(std::memory_order_relaxed);
                counts[thread][zone] =
                    g_totals.zone_counts[thread][zone].load(std::memory_order_relaxed);
            }
            const size_t open = g_totals.open_zone[thread].load(std::memory_order_relaxed);
            const u64 since = g_totals.open_since[thread].load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sequence.load(std::memory_order_relaxed) != begin) {
                continue;
            }
            if (open < ZoneSlots && now > since) {
                ticks[thread][open] += now - since;
            }
            break;
        }
    }
    return {ticks, counts};
}

ZoneTicks Difference(const ZoneTicks& now, const ZoneTicks& last) {
    ZoneTicks frame;
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        for (size_t zone = 0; zone < ZoneSlots; ++zone) {
            frame[thread][zone] = now[thread][zone] - (std::min)(now[thread][zone], last[thread][zone]);
        }
    }
    return frame;
}

/// Each thread's largest zones in the frame with their ms and calls, the time waiting for work or
/// the other thread left out: "back:update buffers 41.2/3120,draw 9.8/3120;front:...".
std::string DetailZonesText(const ZoneTicks& frame, const ZoneTicks& counts,
                            const std::array<bool, ThreadCount>& running, double ms_per_tick) {
    std::string text;
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        if (!running[thread]) {
            continue;
        }
        std::array<size_t, ZoneCount> order;
        for (size_t zone = 0; zone < ZoneCount; ++zone) {
            order[zone] = zone;
        }
        const auto busy = [&frame, thread](size_t zone) {
            return Nx2Profile::stutter_categories[zone] == Category::Idle ? u64{0}
                                                                          : frame[thread][zone];
        };
        std::partial_sort(order.begin(), order.begin() + DetailZones, order.end(),
                          [&busy](size_t a, size_t b) { return busy(a) > busy(b); });
        std::string zones;
        for (size_t rank = 0; rank < DetailZones; ++rank) {
            const double ms = static_cast<double>(busy(order[rank])) * ms_per_tick;
            if (ms < 0.5) {
                break;
            }
            zones += fmt::format("{}{} {:.1f}/{}", zones.empty() ? "" : ",",
                                 Nx2Profile::zone_names[order[rank]], ms,
                                 counts[thread][order[rank]]);
        }
        if (!zones.empty()) {
            text += fmt::format("{}{}:{}", text.empty() ? "" : ";", thread_names[thread], zones);
        }
    }
    return text.empty() ? std::string{"none"} : text;
}

} // Anonymous namespace

ThreadTicks Snapshot(u64 now) {
    ThreadTicks ticks;
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        auto& sequence = g_totals.sequence[thread];
        for (;;) {
            const u32 begin = sequence.load(std::memory_order_acquire);
            if (begin % 2 != 0) {
                std::this_thread::yield();
                continue;
            }
            auto& totals = g_totals.ticks[thread];
            for (size_t index = 0; index < CategoryCount; ++index) {
                ticks[thread][index] = totals[index].load(std::memory_order_relaxed);
            }
            const size_t open = g_totals.open[thread].load(std::memory_order_relaxed);
            const u64 since = g_totals.open_since[thread].load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sequence.load(std::memory_order_relaxed) != begin) {
                continue;
            }
            if (open < CategoryCount && now > since) {
                ticks[thread][open] += now - since;
            }
            break;
        }
    }
    return ticks;
}

Attribution Attribute(const ThreadTicks& ticks, const std::array<bool, ThreadCount>& running,
                      double interval_ms, double ms_per_tick) {
    Attribution result{.cause = "other"};
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        double categorized = 0.0;
        double idle = 0.0;
        for (size_t index = 0; index < CategoryCount; ++index) {
            const double ms = static_cast<double>(ticks[thread][index]) * ms_per_tick;
            if (index == IdleIndex) {
                idle = ms;
            } else {
                result.category_ms[index] += ms;
                categorized += ms;
            }
        }
        if (running[thread]) {
            const double busy = (std::max)(interval_ms - idle, 0.0);
            result.busy_ms[thread] = busy;
            result.other_ms = (std::max)(result.other_ms, busy - categorized);
        }
    }
    double top = result.other_ms;
    for (size_t index = 0; index < IdleIndex; ++index) {
        if (result.category_ms[index] > top) {
            top = result.category_ms[index];
            result.cause = category_names[index];
        }
    }
    return result;
}

void OnPresent() {
    auto& state = g_frame;
    const u64 now = Nx2Profile::Now();
    const u64 now_ticks = Nx2Profile::Ticks();
    const ThreadTicks ticks = Snapshot(now_ticks);
    ThreadTicks counts;
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        for (size_t index = 0; index < CategoryCount; ++index) {
            counts[thread][index] = g_totals.counts[thread][index].load(std::memory_order_relaxed);
        }
    }
    const ThreadTicks last_ticks = std::exchange(state.ticks, ticks);
    const ThreadTicks last_counts = std::exchange(state.counts, counts);
    auto [zone_ticks, zone_counts] = SnapshotZones(now_ticks);
    const ZoneTicks last_zones = std::exchange(state.zone_ticks, zone_ticks);
    const ZoneTicks last_zone_counts = std::exchange(state.zone_counts, zone_counts);
    std::array<u64, SiteCount> site_ticks;
    std::array<u64, SiteCount> site_counts;
    for (size_t index = 0; index < SiteCount; ++index) {
        site_ticks[index] = g_totals.site_ticks[index].load(std::memory_order_relaxed);
        site_counts[index] = g_totals.site_counts[index].load(std::memory_order_relaxed);
    }
    const auto last_site_ticks = std::exchange(state.site_ticks, site_ticks);
    const auto last_site_counts = std::exchange(state.site_counts, site_counts);
    const u64 frame_start_ticks = state.last_ticks;
    const u64 interval_ns = now - state.last_ns;
    const u64 interval_ticks = now_ticks - state.last_ticks;
    state.last_ns = now;
    state.last_ticks = now_ticks;
    if (++state.frames == 1) {
        BootMarker("first_frame");
        state.window_start_ns = now;
        for (size_t index = 0; index < LookupCount; ++index) {
            state.lookups[index] = g_totals.lookups[index].load(std::memory_order_relaxed);
        }
        for (size_t index = 0; index < DriverBuildCount; ++index) {
            state.driver_builds[index] =
                g_totals.driver_builds[index].load(std::memory_order_relaxed);
        }
        return;
    }
    // The zones count in Ticks, which run at a constant rate: convert with this frame's ratio.
    const double ms_per_tick =
        interval_ticks != 0 ? static_cast<double>(interval_ns) / 1e6 / interval_ticks : 0.0;
    if (ms_per_tick > 0.0) {
        g_ticks_per_ms.store(1.0 / ms_per_tick, std::memory_order_relaxed);
    }
    const double interval_ms = static_cast<double>(interval_ns) / 1e6;
    ThreadTicks frame_ticks;
    std::array<bool, ThreadCount> running;
    u64 graphics_builds = 0;
    u64 compute_builds = 0;
    for (size_t thread = 0; thread < ThreadCount; ++thread) {
        for (size_t index = 0; index < CategoryCount; ++index) {
            // The tick counters of different cores can differ slightly.
            const u64 total = ticks[thread][index];
            frame_ticks[thread][index] = total - (std::min)(total, last_ticks[thread][index]);
        }
        running[thread] = g_totals.running[thread].load(std::memory_order_relaxed);
        graphics_builds += counts[thread][GraphicsMissIndex] - last_counts[thread][GraphicsMissIndex];
        compute_builds += counts[thread][ComputeMissIndex] - last_counts[thread][ComputeMissIndex];
    }
    const Attribution frame = Attribute(frame_ticks, running, interval_ms, ms_per_tick);

    ++state.window_frames;
    ++state.histogram[(std::min)(static_cast<size_t>(interval_ms), HistogramBins - 1)];
    state.window_over_33 += interval_ms > StutterMs ? 1 : 0;
    state.window_over_100 += interval_ms > 100.0 ? 1 : 0;
    for (size_t index = 0; index < IdleIndex; ++index) {
        state.window_ms[index] += frame.category_ms[index];
    }
    state.window_graphics_builds += graphics_builds;
    state.window_compute_builds += compute_builds;
    std::string sites;
    for (size_t index = 0; index < SiteCount; ++index) {
        // Ticks and count are separate atomics: a run can land in one frame's ticks and the next
        // frame's count, so each adds up on its own.
        const u64 count = site_counts[index] - (std::min)(site_counts[index], last_site_counts[index]);
        const u64 delta = site_ticks[index] - (std::min)(site_ticks[index], last_site_ticks[index]);
        if (count == 0 && delta == 0) {
            continue;
        }
        const double ms = static_cast<double>(delta) * ms_per_tick;
        state.window_site_counts[index] += count;
        state.window_site_ms[index] += ms;
        sites += fmt::format("{}{}:{}x{:.1f}", sites.empty() ? "" : ",", site_names[index], count, ms);
    }
    std::vector<LongSite> long_sites;
    {
        std::scoped_lock lock{g_long_sites_mutex};
        long_sites.swap(g_long_sites);
    }

    if (interval_ms > StutterMs) {
        std::string fields;
        for (size_t index = 0; index < IdleIndex; ++index) {
            fields += fmt::format(" {}={:.1f}", category_names[index], frame.category_ms[index]);
        }
        std::string busy;
        for (size_t thread = 0; thread < ThreadCount; ++thread) {
            if (frame.busy_ms[thread]) {
                busy += fmt::format("{}{}:{:.1f}", busy.empty() ? "" : ",", thread_names[thread],
                                    *frame.busy_ms[thread]);
            }
        }
        LOG_INFO(HW_GPU, "Nx2Stutter frame={} ms={:.1f} cause={}{} other={:.1f} busy={} "
                         "builds=gfx:{},compute:{}",
                 state.frames, interval_ms, frame.cause, fields, frame.other_ms,
                 busy.empty() ? "none" : busy, graphics_builds, compute_builds);
    }
    if (interval_ms > state.detail_ms) {
        std::string longest;
        for (const LongSite& site : long_sites) {
            if (site.end <= frame_start_ticks) {
                continue;
            }
            longest += fmt::format("{}{}:{} {:.1f}", longest.empty() ? "" : ",", site.thread,
                                   site_names[static_cast<size_t>(site.site)],
                                   static_cast<double>(site.end - site.start) * ms_per_tick);
        }
        LOG_INFO(HW_GPU, "Nx2StutterDetail frame={} ms={:.1f} zones={} sites={} long={}",
                 state.frames, interval_ms,
                 DetailZonesText(Difference(state.zone_ticks, last_zones),
                                 Difference(state.zone_counts, last_zone_counts), running,
                                 ms_per_tick),
                 sites.empty() ? "none" : sites, longest.empty() ? "none" : longest);
    }
    if (now - state.window_start_ns >= SummaryNs) {
        Summarize(state, now);
    }
}

u64 SiteScope::Now() {
    return Nx2Profile::Ticks();
}

void AddSite(Site site, u64 start, u64 end) {
    const auto index = static_cast<size_t>(site);
    const u64 ticks = end > start ? end - start : 0;
    g_totals.site_ticks[index].fetch_add(ticks, std::memory_order_relaxed);
    g_totals.site_counts[index].fetch_add(1, std::memory_order_relaxed);
    const double ticks_per_ms = g_ticks_per_ms.load(std::memory_order_relaxed);
    if (ticks_per_ms == 0.0 || static_cast<double>(ticks) < LongSiteMs * ticks_per_ms) {
        return;
    }
    const std::string_view thread = t_thread < ThreadCount ? thread_names[t_thread] : "other";
    std::scoped_lock lock{g_long_sites_mutex};
    if (g_long_sites.size() < MaxLongSites) {
        g_long_sites.push_back({site, thread, start, end});
    }
}

void BootMarker(std::string_view event, std::string_view details) {
    if (!ReadEnv()) {
        return;
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    LOG_INFO(HW_GPU, "Nx2Boot event={} mono_s={:.3f} {}{}{}", event, seconds, MemoryFields(),
             details.empty() ? "" : " ", details);
}

void SetCacheFile(std::string path) {
    std::scoped_lock lock{g_cache_file_mutex};
    g_cache_file = std::move(path);
}

void StartSession() {
    g_frame = {};
    if (const char* const value = std::getenv("EDEN_NX2_STUTTER_DETAIL_MS")) {
        g_frame.detail_ms = std::strtod(value, nullptr);
    }
    {
        std::scoped_lock lock{g_long_sites_mutex};
        g_long_sites.clear();
    }
    SetCacheFile({});
}

} // namespace VideoCommon::StutterTrace
