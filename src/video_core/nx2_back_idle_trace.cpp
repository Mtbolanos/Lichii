// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <string>

#include "common/logging.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::BackIdleTrace {

namespace {

constexpr size_t RingSize = 4096;
std::array<std::atomic<u64>, RingSize> ring{};
std::atomic<u64> head{}; ///< State changes written
FrontState front_state = FrontState::Parse;

constexpr u64 TimeOf(u64 change) {
    return change >> 3;
}

} // Anonymous namespace

void SplitIdle(std::span<const u64> changes, u64 begin, u64 end, StateNs& out) {
    FrontState state = FrontState::Unknown;
    u64 from = begin;
    for (const u64 change : changes) {
        const u64 at = TimeOf(change);
        if (at >= end) {
            break;
        }
        if (at > from) {
            out[static_cast<size_t>(state)] += at - from;
            from = at;
        }
        state = static_cast<FrontState>(change & 7);
    }
    if (end > from) {
        out[static_cast<size_t>(state)] += end - from;
    }
}

u64 Now() {
    return Nx2Profile::Now();
}

FrontState SwapFrontState(FrontState state) {
    if (!GpuSplit::OnGpuThread()) {
        return FrontState::Count;
    }
    const FrontState previous = front_state;
    if (state != previous) {
        front_state = state;
        const u64 index = head.load(std::memory_order_relaxed);
        ring[index % RingSize].store(Pack(Now(), state), std::memory_order_relaxed);
        head.store(index + 1, std::memory_order_release);
    }
    return previous;
}

void NoteBackIdle(u64 begin, u64 end) {
    static std::array<u64, RingSize> changes;
    static StateNs idle_ns{};
    static std::array<u64, 4> periods{};
    static u64 window_start{};
    static u64 window_presents{};
    const u64 presents = Core::g_nx2_present_count.load(std::memory_order_relaxed);
    // A new window on the first period, and when a new session restarted the present count.
    if (window_start == 0 || presents < window_presents) {
        window_start = begin;
        window_presents = presents;
        idle_ns = {};
        periods = {};
    }
    // Newest first, back to the change in effect at `begin`, into the end of `changes`.
    const u64 last = head.load(std::memory_order_acquire);
    size_t count = 0;
    for (u64 index = last; index > 0 && count < RingSize; --index) {
        const u64 change = ring[(index - 1) % RingSize].load(std::memory_order_acquire);
        changes[RingSize - 1 - count++] = change;
        if (TimeOf(change) <= begin) {
            break;
        }
    }
    // The front may have written over the oldest ones meanwhile.
    const u64 now_head = head.load(std::memory_order_acquire);
    if (now_head >= RingSize) {
        const u64 oldest = now_head - RingSize + 1;
        count = static_cast<size_t>(std::min<u64>(count, last > oldest ? last - oldest : 0));
    }
    SplitIdle(std::span{changes}.last(count), begin, end, idle_ns);
    const u64 length = end - begin;
    ++periods[length < 100'000 ? 0 : length < 500'000 ? 1 : length < 2'000'000 ? 2 : 3];
    if (end - window_start < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(end - window_start) / 1e9;
    const u64 frames = presents - window_presents;
    const auto ms = [](u64 ns) { return static_cast<double>(ns) / 1e6; };
    const auto per_frame = [&](u64 ns) {
        return frames == 0 ? 0.0 : ms(ns) / static_cast<double>(frames);
    };
    static constexpr std::array<const char*, static_cast<size_t>(FrontState::Count)> names{
        "parse", "drain", "list writers", "semaphore", "list copy", "queue empty", "unknown"};
    u64 total = 0;
    std::string states;
    for (size_t i = 0; i < names.size(); ++i) {
        total += idle_ns[i];
        states += fmt::format("{} {} {:.2f} {:.3f}", i == 0 ? "" : ",", names[i],
                              ms(idle_ns[i]) / seconds, per_frame(idle_ns[i]));
    }
    LOG_INFO(HW_GPU,
             "Back idle in 10 s: {:.2f} ms/s, {:.3f} ms/frame over {} frames; by front state "
             "(ms/s, ms/frame):{}; idle periods: {} under 0.1 ms, {} 0.1-0.5 ms, {} 0.5-2 ms, {} "
             "over 2 ms",
             ms(total) / seconds, per_frame(total), frames, states, periods[0], periods[1],
             periods[2], periods[3]);
    idle_ns = {};
    periods = {};
    window_start = end;
    window_presents = presents;
}

} // namespace VideoCommon::BackIdleTrace
