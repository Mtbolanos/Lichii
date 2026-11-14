// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/assert.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/settings.h"
#include "core/core.h"
#include <optional>

#include "video_core/nx2_profile.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_qmd_prediction.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

namespace {

constexpr u32 Gf100BindClassMask = 0xffff;
constexpr u32 Gf100BindValidMask =  0x1f0000 | Gf100BindClassMask;

// Reads a semaphore word, flushing a GPU-written copy out of the buffer cache first.
u32 ReadSemaphoreWord(DmaPusher& dma_pusher, GPUVAddr address) {
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
        device && dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                          VideoCommon::CacheType::BufferCache)) {
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = dma_pusher.rasterizer, device = *device] {
                rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::SemaphoreFlush);
    }
    return dma_pusher.memory_manager.Read<u32>(address);
}

// With the GPU thread split (gpu_split.h) the rasterizer belongs to the back thread: work that
// only feeds it runs there in order; reads of what it wrote first wait for it (DrainBack).
template <typename Func>
void OnBack(const Func& func) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(func);
        split.NoteMemoryWrite();
    } else {
        func();
    }
}

void DrainBack() {
    VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::Acquire);
}

// EDEN_NX2_ACQUIRE_PRECHECK=1: a GEQUAL acquire whose word already holds the value continues
// without draining the split, if nothing queued for the back may write the word (a release that
// lowers it, a fence action) and no GPU work ever wrote its page (then no cache holds a newer word
// than guest memory). It counts as a host sync for later GPU-written command lists, as the drain
// did. Equal and mask acquires always drain. Ported from zortos293/Lichii#166 (e56b37fc0a), with
// the queued-writer check added.
// A release still queued for the split's back meets the acquire: the back runs it before the
// acquire's channel goes on, so the acquire needs no drain (GpuSplit::QueuedReleasePayload).
bool MetByQueuedRelease(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                        u32* miss_out = nullptr) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        return false;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device) {
        return false;
    }
    u32 miss = 0;
    const auto payload = split.QueuedReleasePayload(*device, acquire.address, &miss);
    if (miss_out) {
        *miss_out = payload ? 4 : miss;
    }
    if (!payload || !acquire.Matches(*payload)) {
        if (VideoCommon::Nx2Profile::Enabled()) {
            using Counter = VideoCommon::Nx2Profile::Counter;
            VideoCommon::Nx2Profile::Add(miss == 1   ? Counter::AcquireNoRelease
                                         : miss == 2 ? Counter::AcquireReleaseRan
                                         : miss == 3 ? Counter::AcquireLaterWrite
                                                     : Counter::AcquirePayloadMiss,
                                         1);
        }
        return false;
    }
    split.NoteSync();
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresQueued, 1);
    return true;
}

// What a noted queued writer of a semaphore word is (GpuSplit::QueuedGuestWriters), for the log.
const char* NotedWriterName(const VideoCommon::GpuSplit::GuestWriters& writers) {
    using Cause = VideoCommon::GuestWriterCause;
    using Kind = VideoCommon::GpuSplit::WriterKind;
    switch (writers.noted_cause) {
    case Cause::Pending:
        return "noted write (puller semaphore, 3D report or upload)";
    case Cause::WriterRange:
        return writers.range_kind == Kind::Release ? "compute 0x6c3 release"
               : writers.range_kind == Kind::Copy  ? "copy or copy semaphore"
               : writers.range_kind == Kind::Upload ? "compute inline upload"
                                                    : "compute or copy range";
    case Cause::Ranged:
        return "3D ranged write";
    default:
        return "other";
    }
}

// fable_answer_6 6i (b): ACQUIRE_PRECHECK=2 answers "met" from guest memory alone. NVN2 resets
// its counters (a plain release of a low value) at scene and pass boundaries; a reset still
// queued for the back leaves the old, higher value in guest memory, and the channel then runs
// before the producer the reset preceded (a suspect for the washed-out scene). Asked only where
// level 2 would answer met, with what the front knows of that word:
// - over a queued lower release: the newest plain release recorded for the word
//   (GpuSplit::QueuedReleaseTo: puller WriteLong and SEM_EXECUTE releases, 3D and compute 0x6c3
//   payload releases) is still queued and writes less than the word read;
// - over a queued writer: anything queued the front noted to the word: that release whatever its
//   payload, a noted write over it (QueuedGuestWriters' noted writers: puller semaphores, 3D
//   reports, compute 0x6c3 releases and reductions, copies and copy semaphores, uploads), or,
//   with EXP13 on, a release the QMD prediction recorded for it (QMD completion releases; with
//   EXP13=0 the front can't tell which word a queued QMD releases). The lower ones are among
//   them;
// - readable below: SemaphoreReadableNow read the word current (guest memory with the queued
//   writes the front knows the words of overlaid, GpuSplit::OverlayQueuedWrites, or after
//   waiting for the noted ones, WaitForWritesTo) and below the value, yet guest memory meets it
//   now. Both lookups above ask what is queued NOW, from the same tables the readable read used
//   (pending_writes, queued_releases), so a write the overlay applied that is still queued shows
//   up there too. What they can't see: the readable read came earlier (at back position
//   ReadableRead::done), and work sent before the acquire may have run in between. The writes
//   SemaphoreReadableNow neither waits for nor overlays are the ones not in pending_writes:
//   compute 0x6c3 releases and reductions and copy semaphores (writer_ranges), and QMD
//   completion releases (dispatches, no per-word note with EXP13=0). One of those landing
//   between the two reads raises the word, and has run, so it is queued no more. Or the back
//   ran a known write and then one of those between the overlay's position load and its guest
//   memory read: the overlay then puts the known (older, lower) word back over the newer one.
//   Either way the write the overlay applied ran before level 2 looked. The first 8 log
//   the overlay's newest write since the read (kind, ring, value) and the writers sent after
//   the read, to tell such a raise from a reset.
// The first 8 of each are logged, and one line every 5 s (HW_GPU, Info). Front thread only, as
// the GpuSplit tables it reads.
// EDEN_NX2_EXP34 (live, default 1, nx2_debug_control.h): 1 = an acquire over a queued writer
// (lower releases included) or readable below the value takes the normal path instead (the
// drain or the readable fence release, as when guest memory is below the value), 2 = only one
// over a queued lower release does, 0 = count only. It only turns met answers into the normal
// path, never the other way.

// What SemaphoreReadableNow read for an acquire: the word, and the back's position loaded before
// guest memory was read for it.
struct ReadableRead {
    u32 word;
    u64 done;
};

class PrecheckAudit {
public:
    static PrecheckAudit& Get() {
        static PrecheckAudit audit;
        return audit;
    }

    /// Level 2 would answer met: guest memory holds `word` (>= `value`) at GPU `address`
    /// (device `device`); `readable`: what SemaphoreReadableNow read, when it read the word
    /// current. Returns whether the answer stands (EXP34).
    bool Allow(GPUVAddr address, u64 device, u32 word, u32 value,
               std::optional<ReadableRead> readable) {
        auto& split = VideoCommon::GpuSplit::Get();
        const u64 done = split.ConsumedNow();
        u64 release_position = 0;
        const std::optional<u32> release = split.QueuedReleaseTo(device, done, release_position);
        const bool lower = release && *release < word;
        const auto writers = split.QueuedGuestWriters(done, address, device, sizeof(u32));
        const bool noted = writers.noted > done;
        VideoCommon::QmdPrediction::Queued records{};
        if (VideoCommon::QmdPrediction::Enabled()) {
            records = VideoCommon::QmdPrediction::QueuedPayloads(device, done);
        }
        const bool writer = release || noted || records.count != 0;
        ++met;
        if (lower) {
            ++over_lower;
            if (logged_lower < 8) {
                ++logged_lower;
                LOG_INFO(HW_GPU,
                         "Acquire precheck (level 2) met over a queued lower release: {:#x} "
                         "(device {:#x}) word {:#x} >= {:#x}, queued release of {:#x} at ring {} "
                         "(back at {}, sent {})",
                         address, device, word, value, *release, release_position, done,
                         split.SentPosition());
            }
        }
        if (writer) {
            ++over_writer;
            if (logged_writer < 8) {
                ++logged_writer;
                LOG_INFO(HW_GPU,
                         "Acquire precheck (level 2) met over a queued writer: {:#x} (device "
                         "{:#x}) word {:#x} >= {:#x}; release {} (payload {:#x}, ring {}), noted "
                         "{} (ring {}), QMD prediction records {} (first {:#x}); back at {}, sent "
                         "{}",
                         address, device, word, value, release ? "queued" : "none",
                         release.value_or(0), release_position,
                         noted ? NotedWriterName(writers) : "none", noted ? writers.noted : 0,
                         records.count, records.count != 0 ? records.payloads[0] : 0, done,
                         split.SentPosition());
            }
        }
        const bool below = readable && readable->word < value;
        if (below) {
            ++readable_below;
            if (logged_below < 8) {
                ++logged_below;
                LogReadableBelow(split, address, device, word, value, *readable, done);
            }
        }
        // EXP34: 1 = over a queued writer (the lower releases among them) or readable below the
        // value the normal path answers, 2 = only over a queued lower release; anything else
        // counts only.
        const int guard = VideoCore::Nx2Debug::Experiment(34);
        const bool refuse = (guard == 1 && (writer || below)) || (guard == 2 && lower);
        refused += refuse ? 1 : 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report >= std::chrono::seconds{5}) {
            const double window = std::chrono::duration<double>(now - last_report).count();
            last_report = now;
            LOG_INFO(HW_GPU,
                     "Acquire precheck (level 2): {} met, {} over a queued lower release, {} over "
                     "a queued writer, {} readable below the value; {} sent the normal way "
                     "(EXP34={}) ({:.1f} s)",
                     met, over_lower, over_writer, readable_below, refused, guard, window);
            met = over_lower = over_writer = readable_below = refused = 0;
        }
        return !refuse;
    }

private:
    // A readable-below case: what was written to the word, or sent that may write it, between
    // the readable read (back at `read.done`) and now (back at `done`).
    static void LogReadableBelow(VideoCommon::GpuSplit& split, GPUVAddr address, u64 device,
                                 u32 word, u32 value, const ReadableRead& read, u64 done) {
        // The overlay's source: the newest noted write over the word sent after the read.
        const auto overlay = split.NewestNotedWriteTo(read.done, device);
        // Releases and noted writers sent after the read, run since or still queued.
        u64 release_position = 0;
        const std::optional<u32> release =
            split.QueuedReleaseTo(device, read.done, release_position);
        const auto writers = split.QueuedGuestWriters(read.done, address, device, sizeof(u32));
        const bool noted = writers.noted > read.done;
        // Compute and copies (their 0x6c3, copy and QMD completion releases), unranged writers
        // and queued functions: what the readable read can't account for.
        const u64 untracked = split.UntrackedWriterPosition(device);
        LOG_INFO(HW_GPU,
                 "Acquire precheck (level 2) readable below the value: {:#x} (device {:#x}) "
                 "readable {:#x} < {:#x} <= word {:#x}; read at back {}, back now {}, sent {}; "
                 "overlay write {} at ring {} ({}, {} known words, writes {:#x}); release since "
                 "the read {} "
                 "(payload {:#x}, ring {}); noted since the read {} (ring {}); untracked writer "
                 "at ring {} ({})",
                 address, device, read.word, value, word, read.done, done, split.SentPosition(),
                 overlay.position != 0 ? (overlay.position > done ? "queued" : "ran")
                                       : "none",
                 overlay.position, overlay.what != nullptr ? overlay.what : "-", overlay.words,
                 overlay.words != 0 ? overlay.value : 0, release ? "sent" : "none",
                 release.value_or(0), release_position,
                 noted ? NotedWriterName(writers) : "none", noted ? writers.noted : 0, untracked,
                 untracked > done        ? "queued"
                 : untracked > read.done ? "ran since the read"
                                         : "before the read");
    }

    u64 met{};
    u64 over_lower{};
    u64 over_writer{};
    u64 readable_below{};
    u64 refused{};
    u32 logged_lower{};
    u32 logged_writer{};
    u32 logged_below{};
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

// EDEN_NX2_ACQUIRE_PRECHECK=2: also while work is queued and on GPU-written pages. The word only
// grows (GEQUAL semaphores count up), so guest memory, if older than a queued release or a cache,
// holds a lower value: one that meets the acquire already, meets it then too. Only a write that
// lowers the word (a reset) could make this wrong (PrecheckAudit counts the met answers given over
// a queued write the front knows of). Ocarina drained ~2,200 acquires a second that were met at
// once. `readable`: what SemaphoreReadableNow read, when it read the word current.
bool GequalMetWithoutDrain(DmaPusher& dma_pusher, GPUVAddr address, u32 value,
                           std::optional<ReadableRead> readable = std::nullopt) {
    static const int level = [] {
        const char* setting = std::getenv("EDEN_NX2_ACQUIRE_PRECHECK");
        return setting == nullptr ? 0 : setting[0] == '2' ? 2 : setting[0] != '0' ? 1 : 0;
    }();
    auto& split = VideoCommon::GpuSplit::Get();
    if (level == 0 || !split.Active()) {
        return false;
    }
    auto& memory = dma_pusher.memory_manager;
    if (memory.OverlapsNativeSyncpointShim(address, sizeof(u32)) ||
        (level == 1 && split.HasQueuedMemoryWriters(address, address + sizeof(u32)))) {
        return false;
    }
    const auto device = memory.GpuToCpuAddress(address);
    if (!device || (level == 1 && Nx2IsGpuWritten(*device))) {
        return false;
    }
    const u32 word = memory.Read<u32>(address);
    if (word < value) {
        return false;
    }
    if (level == 2 && !PrecheckAudit::Get().Allow(address, *device, word, value, readable)) {
        return false;
    }
    split.NoteSync();
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresFast, 1);
    return true;
}

void ReportLearning(double window);

// EDEN_NX2_ACQUIRE_TRACE=1: per semaphore address, the acquires that drained the split, the time
// the drains took, why the queued-release check missed and how each ended. Logged every 5 s.
class AcquireTrace {
public:
    enum Outcome : u32 { MetAfterDrain, MetAfterFences, Suspended, NumOutcomes };

    static bool Enabled() {
        static const bool enabled = std::getenv("EDEN_NX2_ACQUIRE_TRACE") != nullptr;
        return enabled;
    }

    void Record(GPUVAddr address, u32 op, u32 channel, u32 miss, u32 word_before, u32 value,
                u64 drain_ns, Outcome outcome, u32 writer, u32 writer_method) {
        std::scoped_lock lock{mutex};
        Entry& entry = entries[address];
        ++entry.writers[(std::min)(writer, 4U)];
        if (writer == 3) {
            entry.writer_method = writer_method;
        }
        ++entry.count;
        entry.drain_ns += drain_ns;
        ++entry.miss[(std::min)(miss, 4U)];
        ++entry.outcomes[outcome];
        entry.op = op;
        entry.channel = channel;
        entry.behind += word_before < value ? 1 : 0;
        entry.last_value = value;
        entry.last_word = word_before;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report < std::chrono::seconds{5}) {
            return;
        }
        const double window = std::chrono::duration<double>(now - last_report).count();
        last_report = now;
        std::vector<std::pair<GPUVAddr, Entry>> sorted(entries.begin(), entries.end());
        std::ranges::sort(sorted, [](const auto& a, const auto& b) {
            return a.second.drain_ns > b.second.drain_ns;
        });
        u64 total_ns = 0;
        u32 total = 0;
        for (const auto& [_, e] : sorted) {
            total_ns += e.drain_ns;
            total += e.count;
        }
        LOG_INFO(HW_GPU, "Acquire trace: {:.0f} drained acquires/s, {:.1f} ms/s draining, {} addresses",
                 total / window, static_cast<double>(total_ns) / 1e6 / window, sorted.size());
        ReportLearning(window);
        for (size_t i = 0; i < (std::min)(sorted.size(), size_t{12}); ++i) {
            const auto& [entry_address, e] = sorted[i];
            LOG_INFO(HW_GPU,
                     "  {:#x} op {} ch {}: {:.0f}/s, {:.2f} ms/s, word behind {}x (last {:#x} < {:#x}); "
                     "miss none/ran/later/payload-miss {}/{}/{}/{}; met after drain {}, after "
                     "fences {}, suspended {}; queued writer none/compute/sync/unranged/ranged "
                     "{}/{}/{}/{}/{} (unranged method {:#x})",
                     entry_address, e.op, e.channel, e.count / window,
                     static_cast<double>(e.drain_ns) / 1e6 / window, e.behind, e.last_word,
                     e.last_value, e.miss[1], e.miss[2], e.miss[3], e.miss[4],
                     e.outcomes[MetAfterDrain], e.outcomes[MetAfterFences], e.outcomes[Suspended],
                     e.writers[0], e.writers[1], e.writers[2], e.writers[3], e.writers[4],
                     e.writer_method);
        }
        entries.clear();
    }

    static AcquireTrace& Get() {
        static AcquireTrace trace;
        return trace;
    }

private:
    struct Entry {
        u32 count{};
        u64 drain_ns{};
        std::array<u32, 5> miss{};
        std::array<u32, NumOutcomes> outcomes{};
        std::array<u32, 5> writers{};
        u32 writer_method{};
        u32 op{};
        u32 channel{};
        u32 behind{};
        u32 last_value{};
        u32 last_word{};
    };
    std::mutex mutex;
    std::unordered_map<GPUVAddr, Entry> entries;
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

// EDEN_NX2_ACQUIRE_EQUAL=1: an equal acquire whose word holds the value continues without a
// drain, unless a release of another payload to the word is still queued. NVN2's queue-sync slots
// get one release per frame number, so the word changes again only after the acquiring queue
// moved on. Ocarina drained ~30 ms/s on such acquires that were met before the drain.
bool EqualMetWithoutDrain(DmaPusher& dma_pusher, GPUVAddr address, u32 value) {
    static const bool enabled = [] {
        const char* setting = std::getenv("EDEN_NX2_ACQUIRE_EQUAL");
        return setting != nullptr && setting[0] == '1';
    }();
    auto& split = VideoCommon::GpuSplit::Get();
    if (!enabled || !split.Active()) {
        return false;
    }
    auto& memory = dma_pusher.memory_manager;
    if (memory.OverlapsNativeSyncpointShim(address, sizeof(u32)) || memory.Read<u32>(address) != value) {
        return false;
    }
    const auto device = memory.GpuToCpuAddress(address);
    if (!device) {
        return false;
    }
    u32 miss = 0;
    if (const auto payload = split.QueuedReleasePayload(*device, address, &miss);
        (payload && *payload != value) || miss == 3) {
        return false;
    }
    if (u32 method = 0; split.QueuedWriterKind(address, address + sizeof(u32), method) == 4) {
        return false; // An upload to the word is queued
    }
    split.NoteSync();
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresFast, 1);
    return true;
}

// EDEN_NX2_ACQUIRE_LEARN=1 (experiment): acquires drained per address. An address whose last 32
// drained acquires were all met at once after the drain (releases of queued work: QMD
// completions) is taken as met without one; the back checks the word when it gets there and
// counts misses (a miss turns this off). One whose last 32 all suspended suspends at once:
// that is always safe, the drain told it nothing.
class AcquireLearning {
public:
    static constexpr u32 Streak = 32;

    static bool Enabled() {
        static const bool enabled = [] {
            const char* setting = std::getenv("EDEN_NX2_ACQUIRE_LEARN");
            return setting != nullptr && setting[0] == '1';
        }();
        return enabled;
    }

    struct Stats {
        u32 met{};
        u32 suspended{};
    };

    Stats& Get(GPUVAddr address) {
        return stats[address];
    }

    void NoteMet(GPUVAddr address) {
        Stats& entry = stats[address];
        entry.met = (std::min)(entry.met + 1, Streak);
        entry.suspended = 0;
    }

    void NoteSuspended(GPUVAddr address) {
        Stats& entry = stats[address];
        entry.suspended = (std::min)(entry.suspended + 1, Streak);
        entry.met = 0;
    }

    static inline std::atomic<u32> misses{};
    static inline std::atomic<u64> speculated{};
    static inline std::atomic<u64> early_suspends{};

    static AcquireLearning& Instance() {
        static AcquireLearning learning;
        return learning;
    }

private:
    std::unordered_map<GPUVAddr, Stats> stats;
};

void ReportLearning(double window) {
    static u64 last_speculated = 0, last_early = 0;
    const u64 speculated = AcquireLearning::speculated.load(std::memory_order_relaxed);
    const u64 early = AcquireLearning::early_suspends.load(std::memory_order_relaxed);
    LOG_INFO(HW_GPU, "Acquire learning: {:.0f} taken as met/s, {:.0f} suspended without a drain/s, "
             "{} misses", (speculated - last_speculated) / window, (early - last_early) / window,
             AcquireLearning::misses.load(std::memory_order_relaxed));
    last_speculated = speculated;
    last_early = early;
}

// EDEN_NX2_EXP3: FenceRelease drains skipped and taken, logged every 5 s.
struct FenceReleaseSkips {
    using FenceQuiet = VideoCommon::GpuSplit::FenceQuiet;
    std::array<u64, 4> taken{}; ///< By GpuSplit::FenceQuiet; Quiet = fence work added since
    u64 skipped{};
    std::chrono::steady_clock::time_point since{};

    void Count(FenceQuiet state, bool skip) {
        if (skip) {
            ++skipped;
        } else {
            ++taken[static_cast<size_t>(state)];
        }
        const auto now = std::chrono::steady_clock::now();
        if (since == std::chrono::steady_clock::time_point{}) {
            since = now;
            return;
        }
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
        LOG_INFO(HW_GPU,
                 "Fence release skip (EXP3={}): {:.0f} drains skipped/s, {:.0f} taken/s: {:.0f} "
                 "back behind, {:.0f} fence work pending, {:.0f} fence work added since, {:.0f} "
                 "unchecked",
                 VideoCore::Nx2Debug::Experiment(3), rate(skipped),
                 rate(taken[0] + taken[1] + taken[2] + taken[3]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Behind)]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Busy)]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Quiet)]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Unchecked)]));
        *this = {};
        since = now;
    }
};

// The fence manager records into the scheduler and downloads the CPU's writes over GPU data:
// the back's state, which a drained back still uses from its idle hook (it submits).
// EDEN_NX2_EXP3 (1, or 2 to allow pending fences without delayed fences): no drain when the
// back, having run everything sent, found the release would only invalidate the GPU cache
// (FenceManager::FencesQuiet) and no fence work was added since; the invalidation still runs
// on the back, in order. Fences already made complete without a release.
void ReleaseFences(DmaPusher& dma_pusher) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (VideoCore::Nx2Debug::Experiment(3) != 0 && split.Active()) {
        static FenceReleaseSkips skips;
        u64 epoch{};
        const auto state = split.FencesQuietNow(epoch);
        const bool skip = state == VideoCommon::GpuSplit::FenceQuiet::Quiet &&
                          dma_pusher.rasterizer->FenceWorkEpoch() == epoch;
        skips.Count(state, skip);
        if (skip) {
            OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); });
            return;
        }
    }
    split.Run([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
              VideoCommon::DrainSite::FenceRelease);
}

// EDEN_NX2_EXP12: the fence releases of acquires whose word the front read current but unmet
// (ReleaseFencesForReadable), logged every 5 s. Front thread only.
struct FenceWriterStats {
    using Clock = std::chrono::steady_clock;

    u64 acquires{};   ///< Readable acquires unmet before their fence release
    u64 delayed{};    ///< With EXP12 on, not applied: fences are delayed (GPU accuracy High)
    u64 waits{};      ///< Waits for the back to run the writers the read can't account for
    u64 wait_ns{};
    u64 met{};        ///< Met once the back ran those writers
    u64 full{};       ///< Full drains (EXP12 0 or 3, 1 when still unmet)
    u64 full_ns{};    ///< Their time (from the target on, when there was one)
    u64 full_met{};   ///< Met after the full drain
    u64 suspended{};  ///< EXP12=2: unmet, suspended without the full drain
    u64 saved_ns{};   ///< EXP12=3: the full drains' time past the target, of acquires met there
    u64 resumed{};    ///< Readable acquires suspended after their fence release, resumed since
    u64 resume_ns{};  ///< Their time from the acquire to the resume
    Clock::time_point since{};

    static FenceWriterStats& Get() {
        static FenceWriterStats stats;
        return stats;
    }

    static u64 Nanoseconds(Clock::time_point begin, Clock::time_point end) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    }

    void Tick() {
        const auto now = Clock::now();
        if (since == Clock::time_point{}) {
            since = now;
            return;
        }
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
        const auto ms = [window](u64 ns) { return static_cast<double>(ns) / 1e6 / window; };
        LOG_INFO(HW_GPU,
                 "Fence writer waits (EXP12={}): {:.0f} readable acquires/s unmet before their "
                 "fence release ({:.0f}/s not applied, delayed fences); writer waits {:.0f}/s "
                 "{:.2f} ms/s, met there {:.0f}/s; full drains {:.0f}/s {:.2f} ms/s, met after "
                 "them {:.0f}/s; suspended without one {:.0f}/s; full drain past the writers of "
                 "acquires met there (EXP12=3) {:.2f} ms/s; resumed {:.0f}/s, {:.0f} us from "
                 "acquire to resume",
                 VideoCore::Nx2Debug::Experiment(12), rate(acquires), rate(delayed), rate(waits),
                 ms(wait_ns), rate(met), rate(full), ms(full_ns), rate(full_met), rate(suspended),
                 ms(saved_ns), rate(resumed),
                 resumed != 0 ? static_cast<double>(resume_ns) / 1e3 / resumed : 0.0);
        *this = {};
        since = now;
    }
};

// EXP13 (nx2_qmd_prediction.h): whether a QMD or copy release the front predicted, still queued,
// would meet an unmet readable acquire. Returns its device and the back's release count, for
// QmdPredictionMetBy once the fences ran.
struct QmdPredictionCheck {
    u64 device;
    u64 sequence;
    /// EXP13=2: the reduction lower bound computed for it (fable_answer_6 6c).
    std::optional<VideoCommon::QmdPrediction::Bound> bound;
};
std::optional<QmdPredictionCheck> CheckQmdPrediction(DmaPusher& dma_pusher,
                                                     const Puller::PendingAcquire& acquire) {
    namespace Prediction = VideoCommon::QmdPrediction;
    auto& split = VideoCommon::GpuSplit::Get();
    if (!Prediction::Enabled() || !split.Active()) {
        return std::nullopt;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device) {
        return std::nullopt;
    }
    if (Prediction::MirrorMode()) {
        // The word first, then the back's position: a record past that position had not run
        // when the word was read, so it is not in it (6c section 3, steps 2-3).
        const u32 word = dma_pusher.memory_manager.Read<u32>(acquire.address);
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 done = split.ConsumedNow();
        const auto bound = Prediction::CheckAcquire({
            .device = *device,
            .address = acquire.address,
            .operation = acquire.operation,
            .value = acquire.value,
            .word = word,
            .done = done,
            .gpu_dirty = Nx2IsGpuWritten(*device),
            .pointer = dma_pusher.memory_manager.GetPointer(acquire.address),
        });
        return QmdPredictionCheck{*device, Prediction::BackSequence(), bound};
    }
    const auto queued = Prediction::QueuedPayloads(*device, split.ConsumedNow());
    const bool meets = std::any_of(queued.payloads.begin(),
                                   queued.payloads.begin() + queued.count,
                                   [&](u32 payload) { return acquire.Matches(payload); });
    Prediction::NoteAcquire(meets              ? Prediction::Coverage::WouldMeet
                            : queued.count != 0 ? Prediction::Coverage::WrongValue
                            : queued.any        ? Prediction::Coverage::AlreadyRan
                                                : Prediction::Coverage::None);
    return QmdPredictionCheck{*device, Prediction::BackSequence(), std::nullopt};
}

void QmdPredictionMetBy(DmaPusher& dma_pusher, const QmdPredictionCheck& check,
                        const Puller::PendingAcquire& acquire, bool met) {
    namespace Prediction = VideoCommon::QmdPrediction;
    using MetBy = Prediction::MetBy;
    if (!met) {
        Prediction::NoteMetBy(MetBy::Unmet);
        return;
    }
    if (check.bound) {
        // Safety: once met, the word must be at least the bound (counted, nothing else).
        Prediction::CheckBound(*check.bound,
                               dma_pusher.memory_manager.Read<u32>(acquire.address));
    }
    auto release = Prediction::BackReleaseSince(check.device, check.sequence);
    // A release the back made before the check that landed after it (a reduction, a fence
    // release) met it too: the newest one the back made for the word.
    bool before_check = false;
    if (!release) {
        release = Prediction::BackReleaseLatest(check.device);
        before_check = release.has_value();
    }
    MetBy met_by = MetBy::Other;
    if (release && release->reduction) {
        met_by = MetBy::QmdReduction;
    } else if (release && acquire.Matches(release->payload)) {
        met_by = release->writer == Prediction::Writer::Qmd ? MetBy::QmdRelease : MetBy::DmaRelease;
    }
    Prediction::NoteMetBy(met_by);
    if (before_check && met_by != MetBy::Other) {
        Prediction::NoteMetBy(MetBy::BeforeCheck);
    }
}

// EXP13=2: a plain release the puller sent, recorded so the reduction bound sees it queued or
// run but not landed (it sets the word).
void RecordPullerRelease(DmaPusher& dma_pusher, GPUVAddr address, u32 payload) {
    namespace Prediction = VideoCommon::QmdPrediction;
    auto& split = VideoCommon::GpuSplit::Get();
    if (!Prediction::MirrorMode() || !split.Active()) {
        return;
    }
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address)) {
        Prediction::Note(*device,
                         {.writer = Prediction::Writer::Puller,
                          .kind = Prediction::Kind::Plain,
                          .payload = payload,
                          .op = 0,
                          .is_signed = false},
                         dma_pusher.memory_manager.Read<u32>(address), split.SentPosition(), 0);
    }
}

// EXP13=3 (nx2_qmd_prediction.h): an acquire a queued release or the reduction bound meets goes
// on as one a queued release meets does (MetByQueuedRelease): the back runs the channel's later
// work behind the releases, and it counts as a host-level wait (NoteSync).
bool MetByQmdPrediction(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                        bool resume) {
    namespace Prediction = VideoCommon::QmdPrediction;
    auto& split = VideoCommon::GpuSplit::Get();
    if (!Prediction::FeedRequested() || !split.Active()) {
        return false;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device) {
        return false;
    }
    // The word first, then the back's position, then the marks (Feed drains them).
    const u32 word = dma_pusher.memory_manager.Read<u32>(acquire.address);
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 done = split.ConsumedNow();
    if (!Prediction::Feed(
            {
                .device = *device,
                .address = acquire.address,
                .operation = acquire.operation,
                .value = acquire.value,
                .word = word,
                .done = done,
                .gpu_dirty = Nx2IsGpuWritten(*device),
                .pointer = dma_pusher.memory_manager.GetPointer(acquire.address),
            },
            resume)) {
        return false;
    }
    split.NoteSync();
    return true;
}

} // Anonymous namespace

// `read_done` (optional): the back's position loaded before guest memory was read for `word`.
static bool SemaphoreReadableNow(DmaPusher& dma_pusher, GPUVAddr address, u32& word,
                                 u64* read_done = nullptr);

// The fence release of an acquire whose word SemaphoreReadableNow read current but unmet: the
// drain only matters for writes that read can't account for. Returns whether the acquire is met.
// EDEN_NX2_EXP12 (live, 0 = off): the back first runs only the queued work that may write the
// word without the front knowing its words (GpuSplit::UntrackedWriterPosition: compute and
// copies, with their QMD and report releases, unranged writers, a release recorded for the word,
// queued functions), the fence release queued behind everything as before. 1 = met there: go
// on, the fences released on the back in order; still unmet: the full drain as before.
// 2 = unmet there too suspends without the full drain (the channel polls as usual). 3 = the full
// drain always, timed past that point (measures what 1 saves). Not with delayed fences, whose
// releases land when the host GPU passed them, not when the back runs them.
static bool ReleaseFencesForReadable(DmaPusher& dma_pusher, const void* owner,
                                     const Puller::PendingAcquire& acquire) {
    using Clock = FenceWriterStats::Clock;
    auto& stats = FenceWriterStats::Get();
    const auto met = [&] {
        u32 word{};
        return Nx2SemaphoreWaiters::Get().Latched(owner) ||
               (SemaphoreReadableNow(dma_pusher, acquire.address, word) && acquire.Matches(word));
    };
    ++stats.acquires;
    const int mode = VideoCore::Nx2Debug::Experiment(12);
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    auto& split = VideoCommon::GpuSplit::Get();
    if (mode <= 0 || mode > 3 || !device || !split.Active() ||
        VideoCore::Nx2Debug::DelayedFences()) {
        stats.delayed += mode != 0 ? 1 : 0;
        const auto begin = Clock::now();
        ReleaseFences(dma_pusher);
        const bool result = met();
        ++stats.full;
        const u64 full_ns = FenceWriterStats::Nanoseconds(begin, Clock::now());
        stats.full_ns += full_ns;
        Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::FullDrain, full_ns);
        stats.full_met += result ? 1 : 0;
        stats.Tick();
        return result;
    }
    const u64 target = split.UntrackedWriterPosition(*device);
    split.Defer([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
    split.NoteMemoryWrite();
    const auto begin = Clock::now();
    if (split.WaitUntil(target, VideoCommon::DrainSite::FenceWriters)) {
        ++stats.waits;
    }
    const auto reached = Clock::now();
    stats.wait_ns += FenceWriterStats::Nanoseconds(begin, reached);
    const bool met_there = met();
    stats.met += met_there ? 1 : 0;
    if (mode == 3 || (mode == 1 && !met_there)) {
        split.Drain(VideoCommon::DrainSite::FenceRelease);
        const u64 ns = FenceWriterStats::Nanoseconds(reached, Clock::now());
        const bool result = met();
        ++stats.full;
        stats.full_ns += ns;
        Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::FullDrain, ns);
        stats.full_met += result ? 1 : 0;
        stats.saved_ns += mode == 3 && met_there ? ns : 0;
        stats.Tick();
        return result;
    }
    // A host-level wait for the commands after it, as the drain was; the fences go out at once.
    split.NoteSync();
    split.Publish();
    stats.suspended += met_there ? 0 : 1;
    stats.Tick();
    return met_there;
}

bool Puller::PendingAcquire::Matches(u32 current) const {
    switch (static_cast<GpuSemaphoreOperation>(operation)) {
    case GpuSemaphoreOperation::AcquireEqual:
        return current == value;
    case GpuSemaphoreOperation::AcquireGequal:
        return current >= value;
    case GpuSemaphoreOperation::AcquireMask:
        return (current & value) != 0;
    default:
        return true;
    }
}

bool Puller::CanResumeAcquire(DmaPusher& dma_pusher) {
    if (!pending_acquire) {
        return true;
    }
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePolls, 1);
    // Fast path (EDEN_NX2_SEM_FAST), as for the first check in ProcessSemaphoreTriggerMethod:
    // when the word is current without draining the back, answer from it. Ocarina of Time polls
    // each waiting acquire about 12 times before it is met, and every poll drained the back
    // (about 380 ms/s of the front's time). The drained path below runs now and then for an
    // unmet word, and reports an acquire not met for 2 s.
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        // The back ran a release that met the value and latched this waiter: resume at once.
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
        return true;
    }
    // EXP13=3: asked once per poll, before the unmet-poll throttle most polls end at.
    bool predicted = false;
    std::optional<ReadableRead> readable;
    u64 read_done = 0;
    if (u32 word{}; SemaphoreReadableNow(dma_pusher, pending_acquire->address, word, &read_done)) {
        if (pending_acquire->Matches(word)) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
            return true;
        }
        readable = ReadableRead{.word = word, .done = read_done};
        // An unmet word is answered without a drain for at most 16 polls or 1 ms since the back
        // was last drained for this acquire. A release the back holds as deferred work lands only
        // when the back runs, so a channel polled without ever draining again waited forever
        // (Mario Kart World hung after 63 s once the 2 s report had been made).
        predicted = true;
        if (MetByQmdPrediction(dma_pusher, *pending_acquire, true)) {
            return true;
        }
        const auto now = std::chrono::steady_clock::now();
        if (++pending_acquire->unmet_polls % 16 != 0 &&
            now - pending_acquire->last_drain < std::chrono::milliseconds{1}) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
            return false;
        }
    }
    if (MetByQueuedRelease(dma_pusher, *pending_acquire)) {
        return true;
    }
    if (!predicted && MetByQmdPrediction(dma_pusher, *pending_acquire, true)) {
        return true;
    }
    if (static_cast<GpuSemaphoreOperation>(pending_acquire->operation) ==
            GpuSemaphoreOperation::AcquireGequal &&
        GequalMetWithoutDrain(dma_pusher, pending_acquire->address, pending_acquire->value,
                              readable)) {
        return true;
    }
    // EDEN_NX2_ACQUIRE_LEARN: resume when a release met the waiter (GPU releases notify
    // Nx2SemaphoreWaiters) or the word in memory does (CPU releases); drain only once the channel
    // has waited 2 ms. Every command list pushed retried this with a drain (~5 ms a frame).
    if (AcquireLearning::Enabled() &&
        std::chrono::steady_clock::now() - pending_acquire->since < std::chrono::milliseconds{2}) {
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            pending_acquire->Matches(dma_pusher.memory_manager.Read<u32>(pending_acquire->address))) {
            VideoCommon::GpuSplit::Get().NoteSync();
            return true;
        }
        return false;
    }
    // The release may still be queued for the split GPU thread's back half (and the caches
    // ReadSemaphoreWord asks are its state).
    DrainBack();
    pending_acquire->last_drain = std::chrono::steady_clock::now();
    const u32 current = ReadSemaphoreWord(dma_pusher, pending_acquire->address);
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        if (static std::atomic<u32> rescued{};
            !pending_acquire->Matches(current) && rescued.fetch_add(1) < 64) {
            LOG_INFO(HW_GPU, "Channel {} met semaphore {:#x} value {:#x} at its release (now {:#x})",
                     dma_pusher.channel_state.bind_id, pending_acquire->address,
                     pending_acquire->value, current);
        }
        return true;
    }
    if (pending_acquire->Matches(current)) {
        return true;
    }
    // With delayed fence operations (Balanced or Accurate fences, accurate GPU), a release
    // queued after the acquire released the fences waits in the caches for the next fence. With
    // every channel waiting none comes: release them again now and then while waiting.
    const bool delayed = Settings::IsGPUFenceBehaviorDefault()
                             ? Settings::IsGPULevelHigh()
                             : Settings::IsGPUFenceBehaviorBalanced() ||
                                   Settings::IsGPUFenceBehaviorAccurate();
    if (const auto now = std::chrono::steady_clock::now();
        delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
        pending_acquire->last_fences = now;
        ReleaseFences(dma_pusher);
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            pending_acquire->Matches(ReadSemaphoreWord(dma_pusher, pending_acquire->address))) {
            return true;
        }
    }
    if (!pending_acquire->reported &&
        std::chrono::steady_clock::now() - pending_acquire->since > std::chrono::seconds{2}) {
        pending_acquire->reported = true;
        const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
        LOG_WARNING(HW_GPU, "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s; "
                    "device {:#x}, buffer cache holds GPU data: {}, words {:08x} {:08x} {:08x} {:08x}",
                    dma_pusher.channel_state.bind_id, pending_acquire->address,
                    pending_acquire->operation, pending_acquire->value, current, device.value_or(0),
                    device && dma_pusher.rasterizer->MustFlushRegion(*device, 16, VideoCommon::CacheType::All),
                    dma_pusher.memory_manager.Read<u32>(pending_acquire->address & ~GPUVAddr{0xf}),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 4),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 8),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 12));
    }
    return false;
}

bool Puller::ResumeAcquire(DmaPusher& dma_pusher) {
    if (!CanResumeAcquire(dma_pusher)) {
        return false;
    }
    // Only a resume from a suspension counts: DmaPusher::Step calls this before every entry, and
    // counting those made the count odd ("suspended") at random, so submit-time snapshots were
    // thrown away for about half of all entries (CommandList::snapshot_suspensions).
    if (pending_acquire) {
        if (pending_acquire->readable_release) {
            auto& stats = FenceWriterStats::Get();
            ++stats.resumed;
            stats.resume_ns += FenceWriterStats::Nanoseconds(
                pending_acquire->since, FenceWriterStats::Clock::now());
        }
        pending_acquire.reset();
        dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
        Nx2SemaphoreWaiters::Get().Remove(this);
    }
    return true;
}

void Puller::ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_DEBUG(HW_GPU, "Binding subchannel {} to engine {:#x}", method_call.subchannel, method_call.argument);
    u32 engine = method_call.argument;
    if ((engine & ~Gf100BindClassMask) != 0 && (engine & ~Gf100BindValidMask) == 0) {
        engine &= Gf100BindClassMask;
    }
    // Switch 2 binds Ampere classes; run them on the Maxwell engines (same method layout for
    // most state). Class 0 unbinds the subchannel.
    switch (engine & 0xFFFF) {
    case 0xC797: // AMPERE_B
        engine = static_cast<u32>(EngineID::MAXWELL_B);
        break;
    case 0xC7C0: // AMPERE_COMPUTE_B
        engine = static_cast<u32>(EngineID::KEPLER_COMPUTE_B);
        break;
    case 0xC7B5: // AMPERE_DMA_COPY_B
        engine = static_cast<u32>(EngineID::MAXWELL_DMA_COPY_A);
        break;
    case 0x0:
        if (g_nx2_ampere) {
            LOG_DEBUG(HW_GPU, "Switch 2 unbinds subchannel {}", method_call.subchannel);
            return;
        }
        break;
    default:
        break;
    }

    const auto engine_id = static_cast<EngineID>(engine);
    bound_engines[method_call.subchannel] = engine_id;
    switch (engine_id) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->fermi_2d, method_call.subchannel, EngineTypes::Fermi2D);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_3d, method_call.subchannel, EngineTypes::Maxwell3D);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_compute, method_call.subchannel, EngineTypes::KeplerCompute);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_dma, method_call.subchannel, EngineTypes::MaxwellDMA);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_memory, method_call.subchannel, EngineTypes::KeplerMemory);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->nv01_timer, method_call.subchannel, EngineTypes::Nv01Timer);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine {:04X}", engine_id);
        break;
    }
}

void Puller::ProcessFenceActionMethod(DmaPusher& dma_pusher) {
    switch (regs.fence_action.op) {
    case Puller::FenceOperation::Acquire:
        // UNIMPLEMENTED_MSG("Channel Scheduling pending.");
        // WaitFence(regs.fence_action.syncpoint_id, regs.fence_value);
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case Puller::FenceOperation::Increment:
        OnBack([rasterizer = dma_pusher.rasterizer, id = static_cast<u32>(regs.fence_action.syncpoint_id.Value())] {
            rasterizer->SignalSyncPoint(id);
        });
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented operation {}", regs.fence_action.op.Value());
        break;
    }
}

static bool SemaphoreReadableNow(DmaPusher& dma_pusher, GPUVAddr address, u32& word,
                                 u64* read_done);

void Puller::ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher) {
    const auto semaphoreOperationMask = 0xF;
    const auto op = GpuSemaphoreOperation(regs.semaphore_trigger & semaphoreOperationMask);
    // EDEN_NX2_SEM_WATCH=<gpu address>: log every release/acquire on the 16 bytes there.
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (watch != 0 && (regs.semaphore_address.SemaphoreAddress() & ~GPUVAddr{0xf}) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} as {} device {:#x} op {} {:#x} value {:#x} (now {:#x})",
                 dma_pusher.channel_state.bind_id, dma_pusher.memory_manager.GetID(),
                 dma_pusher.memory_manager.GpuToCpuAddress(regs.semaphore_address.SemaphoreAddress()).value_or(0),
                 static_cast<u32>(op),
                 regs.semaphore_address.SemaphoreAddress(), regs.semaphore_sequence,
                 dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress()));
    }
    if (op == GpuSemaphoreOperation::WriteLong) {
        const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
        const u32 payload = regs.semaphore_sequence;
        // RELEASE_SIZE (bit 24) set means a 4-byte release without the timestamp; writing 16
        // bytes there clobbers the words next to it (Switch 2 packs semaphores tightly).
        const auto properties = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0
                                    ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout;
        OnBack([rasterizer = dma_pusher.rasterizer, sequence_address, properties, payload] {
            rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload, properties, payload, 0);
        });
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(sequence_address)) {
                split.NoteRelease(*device, sequence_address, payload);
            }
        }
        RecordPullerRelease(dma_pusher, sequence_address, payload);
    } else if (g_nx2_ampere) {
        // Switch 2 waits on semaphores the CPU (or another channel) releases, e.g. once it has
        // written the commands this channel runs next: suspend the channel until the value is
        // met instead of reading ahead (approach from changes.patch).
        PendingAcquire acquire{
            .address = regs.semaphore_address.SemaphoreAddress(),
            .value = regs.semaphore_sequence,
            .operation = static_cast<u32>(op),
            .since = std::chrono::steady_clock::now(),
            .last_fences = std::chrono::steady_clock::now(),
            .reported = false,
        };
        if (op != GpuSemaphoreOperation::AcquireEqual && op != GpuSemaphoreOperation::AcquireGequal &&
            op != GpuSemaphoreOperation::AcquireMask) {
            LOG_ERROR(HW_GPU, "Invalid semaphore operation {}", static_cast<u32>(op));
            return;
        }
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireChecks, 1);
        u32 word{};
        u64 read_done = 0;
        const bool readable = SemaphoreReadableNow(dma_pusher, acquire.address, word, &read_done);
        if (readable && acquire.Matches(word)) {
            return;
        }
        u32 miss = 0;
        if (MetByQueuedRelease(dma_pusher, acquire, &miss)) {
            return;
        }
        if (MetByQmdPrediction(dma_pusher, acquire, false)) {
            return;
        }
        if (op == GpuSemaphoreOperation::AcquireGequal &&
            GequalMetWithoutDrain(dma_pusher, acquire.address, acquire.value,
                                  readable ? std::optional<ReadableRead>{ReadableRead{
                                                 .word = word, .done = read_done}}
                                           : std::nullopt)) {
            return;
        }
        if (op == GpuSemaphoreOperation::AcquireEqual &&
            EqualMetWithoutDrain(dma_pusher, acquire.address, acquire.value)) {
            return;
        }
        const bool trace = AcquireTrace::Enabled();
        const u32 word_before = trace ? dma_pusher.memory_manager.Read<u32>(acquire.address) : 0;
        u32 writer_method = 0;
        const u32 writer =
            trace ? VideoCommon::GpuSplit::Get().QueuedWriterKind(
                        acquire.address, acquire.address + sizeof(u32), writer_method)
                  : 0;
        const auto drain_begin = trace ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        const auto record = [&](AcquireTrace::Outcome outcome) {
            if (trace) {
                AcquireTrace::Get().Record(
                    acquire.address, acquire.operation, dma_pusher.channel_state.bind_id, miss,
                    word_before, acquire.value,
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - drain_begin)
                                         .count()),
                    outcome, writer, writer_method);
            }
        };
        const bool learn = AcquireLearning::Enabled();
        if (learn) {
            auto& learning = AcquireLearning::Instance();
            const auto stats = learning.Get(acquire.address);
            if (stats.met >= AcquireLearning::Streak &&
                AcquireLearning::misses.load(std::memory_order_relaxed) == 0) {
                AcquireLearning::speculated.fetch_add(1, std::memory_order_relaxed);
                VideoCommon::GpuSplit::Get().NoteSync();
                OnBack([memory = &dma_pusher.memory_manager, acquire] {
                    if (!acquire.Matches(memory->Read<u32>(acquire.address)) &&
                        AcquireLearning::misses.fetch_add(1) < 8) {
                        LOG_WARNING(HW_GPU, "Learned acquire {:#x} (op {}, {:#x}) not met on the "
                                    "back (word {:#x}): learning off",
                                    acquire.address, acquire.operation, acquire.value,
                                    memory->Read<u32>(acquire.address));
                    }
                });
                return;
            }
            if (stats.suspended >= AcquireLearning::Streak &&
                !acquire.Matches(dma_pusher.memory_manager.Read<u32>(acquire.address))) {
                AcquireLearning::early_suspends.fetch_add(1, std::memory_order_relaxed);
                if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address)) {
                    Nx2SemaphoreWaiters::Get().Add(this, *device, acquire.value, acquire.operation);
                }
                pending_acquire = acquire;
                dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
                return;
            }
        }
        if (readable) {
            // The word is current (no queued or GPU write the front knows of can change it):
            // the value comes from the CPU or another channel later, and the polls in
            // CanResumeAcquire pick it up without a drain (from zortos293/Lichii#231).
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireNoDrain, 1);
        } else {
            DrainBack();
            if (acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireMetAfterDrain, 1);
                record(AcquireTrace::MetAfterDrain);
                if (learn) {
                    AcquireLearning::Instance().NoteMet(acquire.address);
                }
                return;
            }
        }
        // Registered before the fences run, so a release among them that meets the value counts
        // even if a later one overwrites it.
        auto& waiters = Nx2SemaphoreWaiters::Get();
        if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address)) {
            waiters.Add(this, *device, acquire.value, acquire.operation);
        }
        bool met_now;
        if (readable) {
            const auto qmd_check = CheckQmdPrediction(dma_pusher, acquire);
            met_now = ReleaseFencesForReadable(dma_pusher, this, acquire);
            if (qmd_check) {
                QmdPredictionMetBy(dma_pusher, *qmd_check, acquire,
                                   met_now || waiters.Latched(this));
            }
        } else {
            ReleaseFences(dma_pusher);
            met_now = acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address));
        }
        if (!waiters.Latched(this) && !met_now) {
            pending_acquire = acquire;
            pending_acquire->last_drain = std::chrono::steady_clock::now();
            pending_acquire->readable_release = readable;
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
            record(AcquireTrace::Suspended);
            if (learn) {
                AcquireLearning::Instance().NoteSuspended(acquire.address);
            }
        } else {
            waiters.Remove(this);
            record(AcquireTrace::MetAfterFences);
            if (learn) {
                AcquireLearning::Instance().NoteMet(acquire.address);
            }
        }
    } else {
        DrainBack();
        do {
            const u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
            regs.acquire_source = true;
            regs.acquire_value = regs.semaphore_sequence;
            if (op == GpuSemaphoreOperation::AcquireEqual) {
                regs.acquire_active = true;
                regs.acquire_mode = false;
                if (word != regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireGequal) {
                regs.acquire_active = true;
                regs.acquire_mode = true;
                if (word < regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (word && regs.semaphore_sequence == 0) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else {
                LOG_ERROR(HW_GPU, "Invalid semaphore operation");
            }
        } while (false);
    }
}

void Puller::ProcessSemaphoreRelease(DmaPusher& dma_pusher) {
    const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
    const u32 payload = regs.semaphore_release;
    OnBack([rasterizer = dma_pusher.rasterizer, sequence_address, payload] {
        rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload,
                          VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
    });
    RecordPullerRelease(dma_pusher, sequence_address, payload);
}

void Puller::ProcessSemaphoreAcquire(DmaPusher& dma_pusher) {
    DrainBack();
    u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
    const auto value = regs.semaphore_acquire;
    while (word != value) {
        regs.acquire_active = true;
        regs.acquire_value = value;
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
        // TODO(kemathe73) figure out how to do the acquire_timeout
        regs.acquire_mode = false;
        regs.acquire_source = false;
    }
}

// Ampere SEM_EXECUTE (method 0x1B): operation in bits 2:0 on the semaphore at
// SEM_ADDR_HI:SEM_ADDR_LO (methods 0x18/0x17) with payload SEM_PAYLOAD_LO (0x19).
void Puller::ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument) {
    const GPUVAddr address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                             (regs.reg_array[0x17] & ~3U);
    const u32 payload = regs.reg_array[0x19];
    const u32 operation = argument & 7;
    if (operation == 1) { // RELEASE
        OnBack([rasterizer = dma_pusher.rasterizer, address, payload] {
            rasterizer->Query(address, VideoCommon::QueryType::Payload,
                              VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
        });
        // An acquire it meets while queued needs no drain (from zortos293/Lichii#179).
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address)) {
                split.NoteRelease(*device, address, payload);
            }
        }
        RecordPullerRelease(dma_pusher, address, payload);
        return;
    }
    // Acquires: the result only feeds the warning below, so with the split the pending fences are
    // released in order on the back instead of after a drain (from zortos293/Lichii#179). Later
    // GPU-written list fetches and macro reads still wait for the work before it (NoteSync).
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
        split.NoteSync();
        return;
    }
    // Acquires: check once after flushing pending fences; never block the GPU thread on them.
    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_sem_fast, "EDEN_NX2_SEM_FAST") != 0 &&
        VideoCommon::GpuSplit::Get().Active()) {
        // Only the warning below reads the result: release the fences in order instead.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
        return;
    }
    DrainBack();
    ReleaseFences(dma_pusher);
    const u32 word = dma_pusher.memory_manager.Read<u32>(address);
    const bool ready = operation == 0   ? word == payload
                       : operation == 2 ? word >= payload
                       : operation == 3 ? static_cast<s32>(word - payload) >= 0
                       : operation == 4 ? (word & payload) != 0
                       : operation == 5 ? (word & payload) == 0
                                        : true;
    if (!ready) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(HW_GPU, "SEM_EXECUTE op {} at {:#x}: {:#x} vs {:#x} not met, continuing",
                        operation, address, word, payload);
        }
    }
}

// A semaphore release queued for the split's back: the front's macro DMA reads of those bytes
// wait for it (EDEN_NX2_DMA_WAIT_WRITERS).
// A fence release (`fence_payload`) is written when the back runs it, so reads overlay it.
static void NoteSemaphoreWrite(DmaPusher& dma_pusher, GPUVAddr address,
                               std::optional<u32> fence_payload) {
    auto& split = VideoCommon::GpuSplit::Get();
    split.NoteMemoryWriter();
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return; // Queries write nothing at an unmapped address.
    } else if (fence_payload) {
        split.NoteMemoryWriteData(*device, &*fence_payload, 1);
    } else {
        // Written when its fence is signalled: reads wait until the back ran it and it was.
        split.NoteAsyncWrite(*device, 16);
        split.NoteMemoryWrite(*device, 16, "semaphore (timed)");
    }
}

// EDEN_NX2_SEM_FAST: whether the word at `address` can be read without draining the back: not
// GPU-written, after waiting only for the queued writes noted to it.
// =2 also overlays the queued fence releases the front knows, instead of waiting for them.
static bool SemaphoreReadableNow(DmaPusher& dma_pusher, GPUVAddr address, u32& word,
                                 u64* read_done) {
    auto& split = VideoCommon::GpuSplit::Get();
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_sem_fast, "EDEN_NX2_SEM_FAST");
    using VideoCommon::Nx2Profile::Counter;
    if (!split.Active() || mode == 0) {
        VideoCommon::Nx2Profile::Add(Counter::SemNoFastPath, 1);
        return false;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device || Nx2IsGpuWritten(*device)) {
        VideoCommon::Nx2Profile::Add(Counter::SemGpuWrittenPage, 1);
        return false;
    }
    if (const u64 async_position = split.HasAsyncWrite(*device, 4);
        async_position != 0 &&
        (async_position > split.ConsumedNow() || !Nx2AsyncReportsSettled(*device, 4))) {
        VideoCommon::Nx2Profile::Add(Counter::SemAsyncPending, 1);
        return false;
    }
    if (mode >= 2) {
        const u64 since = split.ConsumedNow();
        word = dma_pusher.memory_manager.Read<u32>(address);
        if (split.OverlayQueuedWrites(since, *device, &word, 1)) {
            if (read_done) {
                *read_done = since;
            }
            return true;
        }
    }
    if (!split.WaitForWritesTo(*device, 16)) {
        VideoCommon::Nx2Profile::Add(Counter::SemWritersWait, 1);
        return false;
    }
    if (read_done) {
        *read_done = split.ConsumedNow();
    }
    word = dma_pusher.memory_manager.Read<u32>(address);
    return true;
}

/// Calls a GPU puller method.
void Puller::CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    static const bool nx2_trace = std::getenv("EDEN_NX2_PULLER_TRACE") != nullptr;
    if (nx2_trace) {
        LOG_INFO(HW_GPU, "nx2 puller sub={} method={:#x} arg={:#x}", method_call.subchannel,
                 method_call.method, method_call.argument);
    }
    regs.reg_array[method_call.method] = method_call.argument;
    const auto method = static_cast<BufferMethods>(method_call.method);

    switch (method) {
    case BufferMethods::BindObject: {
        ProcessBindMethod(dma_pusher, method_call);
        break;
    }
    case BufferMethods::Nop:
    case BufferMethods::SemaphoreAddressHigh:
    case BufferMethods::SemaphoreAddressLow:
    case BufferMethods::SemaphoreSequencePayload:
    case BufferMethods::SyncpointPayload:
    case BufferMethods::WrcacheFlush:
        break;
    case BufferMethods::RefCnt:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->SignalReference(); });
        break;
    case BufferMethods::SyncpointOperation:
        ProcessFenceActionMethod(dma_pusher);
        VideoCommon::GpuSplit::Get().NoteMemoryWriter();
        break;
    case BufferMethods::WaitForIdle:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->WaitForIdle(); });
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case BufferMethods::SemaphoreOperation: {
        ProcessSemaphoreTriggerMethod(dma_pusher);
        if (static_cast<GpuSemaphoreOperation>(regs.semaphore_trigger & 0xF) ==
            GpuSemaphoreOperation::WriteLong) {
            const bool fence = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0;
            NoteSemaphoreWrite(dma_pusher, regs.semaphore_address.SemaphoreAddress(),
                               fence ? std::optional<u32>{regs.semaphore_sequence} : std::nullopt);
        }
        break;
    }
    case BufferMethods::NonStallInterrupt: {
        LOG_ERROR(HW_GPU, "Special puller engine method NonStallInterrupt not implemented");
        break;
    }
    case BufferMethods::MemOpA: {
        LOG_TRACE(HW_GPU, "Memory Operation A");
        break;
    }
    case BufferMethods::MemOpB: {
        // Implement this better.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); });
        break;
    }
    case BufferMethods::MemOpC:
    case BufferMethods::MemOpD: {
        LOG_TRACE(HW_GPU, "Memory Operation C,D");
        break;
    }
    case BufferMethods::SemaphoreAcquire: {
        // On Ampere host classes 0x17..0x1A are SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI.
        if (!g_nx2_ampere) {
            ProcessSemaphoreAcquire(dma_pusher);
        }
        break;
    }
    case BufferMethods::SemaphoreRelease: {
        if (g_nx2_ampere) {
            ProcessSemaphoreExecute(dma_pusher, method_call.argument);
        } else {
            ProcessSemaphoreRelease(dma_pusher);
        }
        if (g_nx2_ampere && (method_call.argument & 7) == 1) {
            NoteSemaphoreWrite(dma_pusher,
                               (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                                   (regs.reg_array[0x17] & ~3U),
                               regs.reg_array[0x19]);
        } else if (!g_nx2_ampere) {
            NoteSemaphoreWrite(dma_pusher, regs.semaphore_address.SemaphoreAddress(),
                               std::nullopt);
        }
        break;
    }
    case BufferMethods::Yield: {
        // TODO(Kmather73): Research and implement this method.
        LOG_ERROR(HW_GPU, "Special puller engine method Yield not implemented");
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Special puller engine method {:X} not implemented", method);
        if (static bool dumped = false; g_nx2_ampere && !dumped) {
            dumped = true;
            dma_pusher.DumpCurrentEntry("unknown host method");
        }
        break;
    }
}

/// Calls a GPU engine method.
static bool Nx2MethodTrace() {
    static const bool trace = std::getenv("EDEN_NX2_METHOD_TRACE") != nullptr;
    return trace;
}

void Puller::CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    const EngineID engine = bound_engines[method_call.subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 method sub={} engine={:04X} method={:#x} arg={:#x}",
                 method_call.subchannel, static_cast<u32>(engine), method_call.method,
                 method_call.argument);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU engine multivalue method.
void Puller::CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    const EngineID engine = bound_engines[subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 multimethod sub={} engine={:04X} method={:#x} amount={} first={:#x}",
                 subchannel, static_cast<u32>(engine), method, amount, base_start[0]);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU method.
void Puller::CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method_call.method, method_call.subchannel);
    ASSERT(method_call.subchannel < bound_engines.size());

    if (ExecuteMethodOnEngine(dma_pusher, method_call.method)) {
        CallEngineMethod(dma_pusher, method_call);
    } else {
        CallPullerMethod(dma_pusher, method_call);
    }
}

/// Calls a GPU multivalue method.
void Puller::CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method, subchannel);
    ASSERT(subchannel < bound_engines.size());
    if (ExecuteMethodOnEngine(dma_pusher, method)) {
        CallEngineMultiMethod(dma_pusher, method, subchannel, base_start, amount, methods_pending);
    } else {
        for (u32 i = 0; i < amount; i++) {
            CallPullerMethod(dma_pusher, MethodCall{
                method,
                base_start[i],
                subchannel,
                methods_pending - i,
            });
        }
    }
}

/// Determines where the method should be executed.
[[nodiscard]] bool Puller::ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method) {
    const auto buffer_method = BufferMethods(method);
    return buffer_method >= BufferMethods::NonPullerMethods;
}

} // namespace Tegra::Engines
