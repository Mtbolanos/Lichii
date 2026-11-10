// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#ifdef _MSC_VER
#include <intrin.h>
#elif defined(ARCHITECTURE_x86_64)
#include <x86intrin.h>
#endif
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "common/logging.h"
#include "common/cpu_features.h"
#include "common/thread_cpu_clock.h"
#include "common/nx2_timeline.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h)
    SplitWait,    ///< Split GPU thread: the back waits for work
    WriterWait,
    ListSubmitWait,
    ListGpuWait,
    MemoryQuery, ///< Device::QueryDeviceMemoryUsage, the driver's memory budget query (EXP29)
    QueryDmaWait, ///< Macro DMA reads of reports waiting for query results (FlushQueryResults)
    ReadFlush, ///< Guest reads of GPU-written memory run by TickWork (FlushRegion, Finish)
    // Of compute: the steps of a dispatch (RasterizerVulkan::DispatchCompute, Configure).
    ComputeLookup,   ///< CurrentComputePipeline
    ComputePointers, ///< Storage buffer pointers read (BindComputeStorageBuffer)
    ComputeViews,    ///< Texture and image handles read, image views found (FillImageViews)
    ComputeResolve,  ///< Bindings resolved to buffers (UpdateComputeBuffers)
    ComputeSync,     ///< Buffers synchronized and bound (BindHostComputeBuffers)
    // TEST: fine zones, EDEN_NX2_GPU_PROFILE=2 only (FineScope): they split the zones above.
    CbBind,          ///< Of method: constant buffer binds (Maxwell3D::ProcessCBBind)
    CbUpload,        ///< Of method: constant buffer words written to memory
    QueryReport,     ///< Of method: report semaphores (Maxwell3D::ProcessQueryGet)
    CounterReset,    ///< Of method: counter resets (Maxwell3D::ProcessCounterReset)
    RenderEnable,    ///< Of method: render enable conditions (Maxwell3D::ProcessQueryCondition)
    Sink,            ///< State writes held for the next executable method (ConsumeSinkImpl)
    MacroEnd,        ///< A front macro's end on the back (Maxwell3D::FinishFrontMacro)
    DrawSetup,       ///< Of draw: after the pipeline lookup, before Configure (cache locks)
    DrawQueries,     ///< Of draw: query and transform feedback state before recording
    TickWork,        ///< Of draw and compute: the sync operations after them (GPU::TickWork)
    UniformBuffers,  ///< Of bind stage buffers: the uniform buffers
    GeometryBuffers, ///< Of update buffers: index and vertex buffer binds
    StorageBindings, ///< Of read handles: storage buffer addresses from constant buffers
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "split drain", "split wait",
    "writer wait", "list submit wait", "list GPU wait", "memory query",
    "query DMA wait", "read flush",
    "compute lookup", "compute pointers", "compute views", "compute resolve",
    "compute synchronize",
    "cb bind", "cb upload", "query report", "counter reset", "render enable", "sink",
    "macro end", "draw setup", "draw queries", "tick work", "uniform buffers",
    "geometry buffers", "storage bindings",
};
static_assert(zone_names.back() != nullptr, "a zone without a name");

// Read once, when the GPU thread starts (Initialize): Enabled() runs several times per method
// write (no guard to check). Not at static initialization: nx2_switches.txt is only applied in
// main(). Off while EDEN_NX2_SAMPLE_PROFILE samples (common/nx2_sample_profile.h): the zone timers
// would show up in the samples.
inline bool g_enabled = false;
/// TEST: EDEN_NX2_GPU_PROFILE=2 adds the fine zones (FineScope) and counters (FineCounter), at
/// the cost of more time stamp reads: =1 keeps the zones runs compare with.
inline bool g_fine = false;

inline void Initialize() {
    g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr &&
                std::getenv("EDEN_NX2_SAMPLE_PROFILE") == nullptr;
    const char* value = std::getenv("EDEN_NX2_GPU_PROFILE");
    g_fine = g_enabled && value != nullptr && value[0] == '2';
}

inline bool Enabled() {
    return g_enabled;
}

inline bool Fine() {
    return g_fine;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

/// What a front macro memory read (MME DMA) had to wait for: the newest unexecuted writer.
enum class DmaWaitCause : u32 {
    ComputeCopy,       ///< Compute or copy work
    HostSync,          ///< The work before a host-level wait
    ConstBufferUpload, ///< A 3D constant buffer upload (const_buffer.buffer)
    InlineUpload,      ///< 3D inline data
    Macro3D,           ///< A 3D macro register
    Report,            ///< A 3D report (report_semaphore.query)
    DmaLaunch,         ///< 3D launch_dma
    SyncInfo,          ///< 3D sync_info
    WaitForIdle,       ///< 3D wait_for_idle
    Other3D,           ///< Another memory-writing 3D method
    OtherEngine,       ///< A compute, DMA, inline-to-memory or 2D engine method
    Deferred,          ///< Puller semaphore releases and maps
    AvoidedByRange,    ///< Not waited: the pending writes' ranges miss the read (RangeLastWrite)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DmaWaitCause::Count)>
    dma_wait_cause_names{"compute/copy",  "host sync",    "cbuf upload", "inline upload",
                         "3D macro",      "report",       "dma launch",  "sync info",
                         "wait for idle", "other 3D",     "other engine", "deferred",
                         "avoided (range)"};
// What the zones and drain waits measure in: the time stamp counter without the fences the wall
// clock reads it with (those fences cost ~10% of the GPU back thread with ~45,000 draws a
// second). Report converts it with the wall clock.
inline u64 Ticks() {
#ifdef ARCHITECTURE_x86_64
    return __rdtsc();
#else
    return Now();
#endif
}

/// TEST: amounts counted with EDEN_NX2_GPU_PROFILE=2, logged per second in a line of their own.
enum class FineCounter : u32 {
    UniformRebinds,  ///< Small uniform buffers bound again from the same stream allocation
    UniformStreamed, ///< Small uniform buffers copied into the stream buffer
    UniformCached,   ///< Uniform buffers bound from their host buffer
    UniformOther,    ///< Uniform buffers bound otherwise (root tables, OpenGL, fallbacks)
    UniformBytes,    ///< Bytes of the uniform buffers bound (all paths)
    StorageBinds,    ///< Graphics storage buffers bound
    TextureHandles,  ///< Graphics texture handles read
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(FineCounter::Count)>
    fine_counter_names{
        "uniform rebinds", "uniform streamed", "uniform cached",  "uniform other",
        "uniform bytes",   "storage binds",    "texture handles",
    };
static_assert(fine_counter_names.back() != nullptr, "a fine counter without a name");

struct ThreadData {
    std::array<u64, static_cast<size_t>(FineCounter::Count)> fine_counters{}; ///< TEST
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 64> counters{}; ///< See Counter (static_assert below)
    std::array<u64, static_cast<size_t>(DmaWaitCause::Count)> dma_wait_causes{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
    u64 window_start_ticks{};
    std::optional<u64> cpu_start_ns;
    u32 depth{};          ///< Zones open (Scope, timed SampledScope)
    u32 samples_left{1};  ///< SampledScope: entries until the next timed one
    u32 sample_weight{1}; ///< SampledScope: the entries the next timed one stands for
    u32 sample_seed{0x2545F491};
};

inline thread_local ThreadData t_data;

/// The zones a timeline records without detail: waits, whole draws, dispatches and macros.
[[nodiscard]] constexpr bool IsCoarse(Zone zone) {
    switch (zone) {
    case Zone::Idle:
    case Zone::PendingSleep:
    case Zone::ResumePending:
    case Zone::Push:
    case Zone::Macro:
    case Zone::Draw:
    case Zone::Clear:
    case Zone::Compute:
    case Zone::FlushWork:
    case Zone::TextureUpload:
    case Zone::Fence:
    case Zone::SplitDrain:
    case Zone::SplitWait:
    case Zone::WriterWait:
    case Zone::ListSubmitWait:
    case Zone::ListGpuWait:
    case Zone::QueryDmaWait:
    case Zone::ReadFlush:
        return true;
    default:
        return false;
    }
}

class Scope {
public:
    explicit Scope(Zone zone_) : Scope{zone_, Enabled(), true} {}

    /// A zone timed only if `on` (a profiled run's choice, such as Fine()); on the timeline only
    /// if `timeline_on` too.
    Scope(Zone zone_, bool on, bool timeline_on)
        : zone{zone_}, enabled{on},
          timeline{timeline_on && Common::Nx2Timeline::Active() &&
                   (Common::Nx2Timeline::Detail() || IsCoarse(zone_))} {
        if (timeline) {
            timeline_start = Common::Nx2Timeline::Now();
        }
        if (!enabled) {
            return;
        }
        auto& data = t_data;
        ++data.depth;
        parent_child_ns = data.child_ns;
        data.child_ns = 0;
        start = Ticks();
    }

    ~Scope() {
        if (timeline) {
            Common::Nx2Timeline::Record(zone_names[static_cast<size_t>(zone)], timeline_start,
                                        Common::Nx2Timeline::Now());
        }
        if (!enabled) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        data.child_ns = parent_child_ns + elapsed;
        --data.depth;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    bool enabled;
    bool timeline;
    u64 start{};
    u64 parent_child_ns{};
    u64 timeline_start{};
};

/// TEST: a fine zone (EDEN_NX2_GPU_PROFILE=2), on the timeline only then too.
class FineScope : public Scope {
public:
    explicit FineScope(Zone zone_) : Scope{zone_, Fine(), Fine()} {}
};

/// The ticks two time stamp reads in a row take: what a timed entry measures on top of its own
/// time (the least of a few tries).
inline u64 TimingOverheadTicks() {
    static const u64 overhead = [] {
        u64 least = ~u64{0};
        for (int i = 0; i < 64; ++i) {
            const u64 first = Ticks();
            const u64 second = Ticks();
            least = (std::min)(least, second - first);
        }
        return least;
    }();
    return overhead;
}

/// A zone entered millions of times a second, such as the split's back's 3D register writes
/// (Maxwell3D::CallMethod): timing every entry took two time stamp reads each, which cost more
/// than most entries and slowed the very runs it measured. Entered outside every other zone, one
/// entry in about SampleEvery is timed and stands for the entries since the last timed one; all
/// are counted. Entered inside another zone it does nothing: the entry is that zone's time.
/// Not recorded on the Nx2Timeline (a span per sampled method would say nothing there).
class SampledScope {
public:
    static constexpr u32 SampleEvery = 32;
    /// About 25 us at 4 GHz (time stamp ticks): far longer than these entries take.
    static constexpr u64 LongTicks = 100'000;

    /// `entries`: what this entry counts as (a run of constant buffer words counts each word).
    explicit SampledScope(Zone zone, u32 entries = 1) {
        if (!Enabled()) [[likely]] {
            return;
        }
        auto& data = t_data;
        if (data.depth != 0) {
            return;
        }
        index = static_cast<size_t>(zone);
        data.count[index] += entries;
        if (--data.samples_left != 0) {
            return;
        }
        weight = data.sample_weight;
        // The next interval: 1 to 2 * SampleEvery - 1 entries, so that no fixed period lines up
        // with a repeating pattern of entries (xorshift32).
        u32 seed = data.sample_seed;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        data.sample_seed = seed;
        data.sample_weight = 1 + seed % (2 * SampleEvery - 1);
        data.samples_left = data.sample_weight;
        ++data.depth;
        parent_child_ns = data.child_ns;
        data.child_ns = 0;
        start = Ticks();
    }

    ~SampledScope() {
        if (weight == 0) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        // Less the reads' own cost, which the untimed entries did not pay.
        const u64 overhead = TimingOverheadTicks();
        const u64 own = elapsed - data.child_ns;
        const u64 measured = own > overhead ? own - overhead : 0;
        // An entry this long was most likely interrupted or preempted: it counts once, not for
        // the untimed entries too.
        data.exclusive_ns[index] += measured > LongTicks ? measured : measured * weight;
        data.child_ns = parent_child_ns + elapsed;
        --data.depth;
    }

    SampledScope(const SampledScope&) = delete;
    SampledScope& operator=(const SampledScope&) = delete;

private:
    size_t index{};
    u64 weight{}; ///< 0: not timed
    u64 start{};
    u64 parent_child_ns{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    AcquiresFast,   ///< EDEN_NX2_ACQUIRE_PRECHECK: acquires met without draining the split
    AcquiresQueued, ///< Acquires met by a release still queued for the split's back
    PtrCacheHits,   ///< EDEN_NX2_HANDLE_PTR_CACHE: handle/descriptor reads, page pointer cached
    PtrCacheMisses, ///< EDEN_NX2_HANDLE_PTR_CACHE: handle/descriptor reads, page looked up
    UboInPlaceSame, ///< EDEN_NX2_UBO_INPLACE: small uniform buffers unchanged, nothing copied
    UboInPlaceChanged, ///< EDEN_NX2_UBO_INPLACE: small uniform buffers changed, copied once
    AcquireNoRelease,  ///< Drained acquires: no release of the word was sent this session
    AcquireReleaseRan, ///< Drained acquires: the last release of the word already ran on the back
    AcquireLaterWrite, ///< Drained acquires: work sent after its release may write the word
    AcquirePayloadMiss, ///< Drained acquires: the queued release's payload does not meet it
    AcquireMetAfterDrain, ///< Drained acquires met at once after the drain
    DmaReadsForwarded, ///< MME DMA reads served from a queued constant buffer upload
    DmaReadsEarly,  ///< MME DMA reads without a drain (Maxwell3D::FrontDmaReadEarly)
    DmaReadsWaitUnknown, ///< Of those, waited for a write the front can't place
    DmaReadsWaitGpu,     ///< waited for all queued GPU work with unknown destinations
    DmaReadsWaitWrite,   ///< waited for a noted write to the words' pages
    AcquirePolls,   ///< Polls of a waiting acquire (Puller::CanResumeAcquire)
    AcquirePollsFast, ///< Of AcquirePolls: answered from the word without draining the back
    SemNoFastPath,    ///< SemaphoreReadableNow false: split off or EDEN_NX2_SEM_FAST=0
    SemGpuWrittenPage, ///< SemaphoreReadableNow false: the word's page is marked GPU-written
    SemAsyncPending,  ///< SemaphoreReadableNow false: a counter report or timed release is pending
    SemWritersWait,   ///< SemaphoreReadableNow false: WaitForWritesTo gave up
    AcquireNoDrain,   ///< Acquires of a current word not met yet that registered without a drain
    // The compute bindings line (Report): what a dispatch's storage buffers cost.
    ComputeStorageBindings,  ///< Compute storage buffers bound (BindHostComputeStorageBuffers)
    ComputeStorageBytes,     ///< Their bound sizes
    ComputeSyncUploads,      ///< Their SynchronizeBuffer calls that uploaded something
    ComputeSyncBytes,        ///< Bytes those uploaded
    ComputeSyncDirtyWalks,   ///< Their synchronizes under EDEN_NX2_EXP59 & 4 or & 8 (dirty-word walk)
    ComputeSyncDirtySkips,   ///< Of those, all tracker words clean: the binding skipped entirely
    RangeCacheHits,          ///< Compute VirtualRangeCache::Query answered from the cache
    RangeCacheMisses,        ///< Compute VirtualRangeCache::Query that walked the page table
    StorageBindingHits,      ///< Compute StorageBufferBinding answered from its 4-way cache
    StorageBindingMisses,    ///< Compute StorageBufferBinding worked out again
    Count,
};
static_assert(static_cast<size_t>(Counter::Count) <= 64, "Counter outgrew the counters array");
static_assert(static_cast<size_t>(Counter::Count) <= std::tuple_size_v<decltype(ThreadData::counters)>);

/// Method calls on this thread by method (3D registers on the back, macros on the front), and
/// same-value writes among them; MME DMA reads by the macro that made them. The most frequent
/// are logged with the zones.
inline thread_local std::array<u32, 0x2000> t_method_count{};
inline thread_local std::array<u32, 0x2000> t_method_same{};
inline thread_local std::array<u32, 0x2000> t_dma_reads{};

inline void CountMethod(u32 method, bool same) {
    if (Enabled() && method < t_method_count.size()) {
        ++t_method_count[method];
        t_method_same[method] += same ? 1 : 0;
    }
}

/// A 3D register write (Maxwell3D::CallMethod below the macro range), `same` if the register
/// held the value already: the write counters and the per-method counts.
inline void CountRegisterWrite(u32 method, bool same) {
    if (!Enabled()) {
        return;
    }
    auto& counters = t_data.counters;
    ++counters[static_cast<size_t>(Counter::MethodWrites)];
    counters[static_cast<size_t>(Counter::SameValueWrites)] += same ? 1 : 0;
    CountMethod(method, same);
}

inline void CountDmaRead(u32 macro) {
    if (Enabled() && macro < t_dma_reads.size()) {
        ++t_dma_reads[macro];
    }
}

/// Compute dispatches on this thread by program (shader hash); the most frequent are logged
/// with the zones.
inline thread_local std::unordered_map<u64, u64> t_compute_programs;

inline void CountComputeProgram(u64 shader_hash) {
    if (Enabled()) {
        ++t_compute_programs[shader_hash];
    }
}

inline std::string TopMethods(const std::array<u32, 0x2000>& counts,
                              const std::array<u32, 0x2000>* same, u64 window) {
    std::array<u32, 0x2000> order;
    for (u32 i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    constexpr size_t top = 24;
    std::partial_sort(order.begin(), order.begin() + top, order.end(),
                      [&counts](u32 a, u32 b) { return counts[a] > counts[b]; });
    std::string text;
    for (size_t i = 0; i < top && counts[order[i]] != 0; ++i) {
        const u32 method = order[i];
        text += fmt::format(" {:#x} {:.0f}/s", method,
                            static_cast<double>(counts[method]) * 1e9 / window);
        if (same != nullptr && (*same)[method] != 0) {
            text += fmt::format(" ({:.0f} same)", static_cast<double>((*same)[method]) * 1e9 / window);
        }
        text += ',';
    }
    if (!text.empty()) {
        text.pop_back();
    }
    return text;
}

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// TEST: EDEN_NX2_GPU_PROFILE=2 only.
inline void AddFine(FineCounter counter, u64 amount) {
    if (Fine()) [[unlikely]] {
        t_data.fine_counters[static_cast<size_t>(counter)] += amount;
    }
}

/// The front waited for the back since `start` (Ticks) for `site`.
inline void AddDrain(DrainSite site, u64 start) {
    if (Enabled()) {
        const auto index = static_cast<size_t>(site);
        t_data.drain_ns[index] += Ticks() - start;
        ++t_data.drain_count[index];
    }
}

inline void AddDmaWaitCause(DmaWaitCause cause) {
    if (Enabled()) {
        ++t_data.dma_wait_causes[static_cast<size_t>(cause)];
    }
}

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.window_start_ticks = Ticks();
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    const auto cpu_now = Common::CurrentThreadCpuTimeNs();
    const u64 now_ticks = Ticks();
    const u64 window_ticks = (std::max<u64>)(now_ticks - data.window_start_ticks, 1);
    // Time in no zone at all is the wall time minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window_ticks > zoned ? window_ticks - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window_ticks;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    if (const u64 queued = data.counters[static_cast<size_t>(Counter::AcquiresQueued)]; queued != 0) {
        line += fmt::format(" | acquires met by a queued release {:.0f}",
                            static_cast<double>(queued) * 1e9 / window);
    }
    {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        if (data.counters[static_cast<size_t>(Counter::AcquireChecks)] != 0) {
            line += fmt::format(" | drained acquires: no release {:.0f}, release ran {:.0f}, "
                                "later write {:.0f}, payload miss {:.0f}, met after drain {:.0f}",
                                rate(Counter::AcquireNoRelease), rate(Counter::AcquireReleaseRan),
                                rate(Counter::AcquireLaterWrite),
                                rate(Counter::AcquirePayloadMiss),
                                rate(Counter::AcquireMetAfterDrain));
        }
    }
    if (const u64 forwarded = data.counters[static_cast<size_t>(Counter::DmaReadsForwarded)];
        forwarded != 0) {
        line += fmt::format(" | MME DMA reads from queued uploads {:.0f}",
                            static_cast<double>(forwarded) * 1e9 / window);
    }
    if (const u64 fast = data.counters[static_cast<size_t>(Counter::AcquiresFast)]; fast != 0) {
        line += fmt::format(" | acquires without a drain {:.0f}",
                            static_cast<double>(fast) * 1e9 / window);
    }
    if (const u64 hits = data.counters[static_cast<size_t>(Counter::PtrCacheHits)],
        misses = data.counters[static_cast<size_t>(Counter::PtrCacheMisses)];
        hits != 0 || misses != 0) {
        line += fmt::format(" | handle page pointers {:.0f} cached, {:.0f} looked up",
                            static_cast<double>(hits) * 1e9 / window,
                            static_cast<double>(misses) * 1e9 / window);
    }
    if (const u64 same = data.counters[static_cast<size_t>(Counter::UboInPlaceSame)],
        changed = data.counters[static_cast<size_t>(Counter::UboInPlaceChanged)];
        same != 0 || changed != 0) {
        line += fmt::format(" | uniform buffers compared in place {:.0f} unchanged, {:.0f} changed",
                            static_cast<double>(same) * 1e9 / window,
                            static_cast<double>(changed) * 1e9 / window);
    }
    if (const u64 early = data.counters[static_cast<size_t>(Counter::DmaReadsEarly)]; early != 0) {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        line += fmt::format(" | MME DMA reads without a drain {:.0f} (waited for: unknown writes "
                            "{:.0f}, GPU work {:.0f}, writes to the words {:.0f})",
                            static_cast<double>(early) * 1e9 / window,
                            rate(Counter::DmaReadsWaitUnknown), rate(Counter::DmaReadsWaitGpu),
                            rate(Counter::DmaReadsWaitWrite));
    }
    {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        if (data.counters[static_cast<size_t>(Counter::AcquirePolls)] != 0 ||
            data.counters[static_cast<size_t>(Counter::AcquireNoDrain)] != 0) {
            line += fmt::format(" | acquire polls {:.0f} ({:.0f} without a drain), acquires "
                                "waiting without a drain {:.0f}, semaphore word not readable: no "
                                "fast path {:.0f}, GPU-written page {:.0f}, report pending {:.0f}, "
                                "writers {:.0f}",
                                rate(Counter::AcquirePolls), rate(Counter::AcquirePollsFast),
                                rate(Counter::AcquireNoDrain), rate(Counter::SemNoFastPath),
                                rate(Counter::SemGpuWrittenPage), rate(Counter::SemAsyncPending),
                                rate(Counter::SemWritersWait));
        }
    }
    std::string drains;
    for (size_t i = 0; i < data.drain_count.size(); ++i) {
        if (data.drain_count[i] != 0) {
            drains += fmt::format(" {} {:.1f} ms ({:.0f}/s),", drain_site_names[i],
                                  static_cast<double>(data.drain_ns[i]) * 1000.0 / window_ticks,
                                  static_cast<double>(data.drain_count[i]) * 1e9 / window);
        }
    }
    if (!drains.empty()) {
        drains.pop_back();
        line += " | waits for the back by site:" + drains;
    }
    std::string causes;
    for (size_t i = 0; i < data.dma_wait_causes.size(); ++i) {
        if (data.dma_wait_causes[i] != 0) {
            causes += fmt::format(" {} {:.0f}/s,", dma_wait_cause_names[i],
                                  static_cast<double>(data.dma_wait_causes[i]) * 1e9 / window);
        }
    }
    if (!causes.empty()) {
        causes.pop_back();
        line += " | macro dma reads waited for:" + causes;
    }
    if (cpu_now && data.cpu_start_ns && *cpu_now >= *data.cpu_start_ns) {
        line += fmt::format(" | host_cpu_busy_ms_per_s={:.3f} window_wall_ms={:.3f}",
                            static_cast<double>(*cpu_now - *data.cpu_start_ns) * 1000.0 / window,
                            static_cast<double>(window) / 1e6);
    } else {
        line += fmt::format(" | host_cpu_busy_ms_per_s=unavailable window_wall_ms={:.3f}",
                            static_cast<double>(window) / 1e6);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    if (Fine()) {
        // TEST: the fine counters.
        std::string fine;
        for (size_t i = 0; i < data.fine_counters.size(); ++i) {
            if (data.fine_counters[i] != 0) {
                fine += fmt::format(" {} {:.0f},", fine_counter_names[i],
                                    static_cast<double>(data.fine_counters[i]) * 1e9 / window);
            }
        }
        if (!fine.empty()) {
            fine.pop_back();
            LOG_INFO(HW_GPU, "{} fine counters per second:{}", thread_name, fine);
        }
    }
    if (const std::string methods = TopMethods(t_method_count, &t_method_same, window); !methods.empty()) {
        LOG_INFO(HW_GPU, "{} methods per second:{}", thread_name, methods);
    }
    if (const std::string reads = TopMethods(t_dma_reads, nullptr, window); !reads.empty()) {
        LOG_INFO(HW_GPU, "{} MME DMA reads per second by macro:{}", thread_name, reads);
    }
    {
        // What a dispatch's storage buffers cost: bound against uploaded bytes, and how often
        // the binding caches answered (see the compute zones for the time).
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        const u64 dispatches = data.count[static_cast<size_t>(Zone::Compute)];
        if (dispatches != 0 || data.counters[static_cast<size_t>(Counter::ComputeStorageBindings)] != 0) {
            constexpr double MiB = 1024.0 * 1024.0;
            // EDEN_NX2_EXP59 & 4 / & 8: synchronizes that found every tracker word clean.
            const std::string dirty_walk =
                data.counters[static_cast<size_t>(Counter::ComputeSyncDirtyWalks)] == 0
                    ? std::string{}
                    : fmt::format(", dirty-word walk (EXP59) skipped {:.0f} of {:.0f}",
                                  rate(Counter::ComputeSyncDirtySkips),
                                  rate(Counter::ComputeSyncDirtyWalks));
            LOG_INFO(HW_GPU,
                     "{} compute bindings per second: {:.0f} dispatches, storage bound {:.1f} MiB "
                     "in {:.0f} bindings, synchronized/uploaded {:.2f} MiB in {:.0f} uploads, "
                     "range cache {:.0f} hits / {:.0f} misses, pointer cache {:.0f} hits / "
                     "{:.0f} misses{}",
                     thread_name, static_cast<double>(dispatches) * 1e9 / window,
                     rate(Counter::ComputeStorageBytes) / MiB,
                     rate(Counter::ComputeStorageBindings), rate(Counter::ComputeSyncBytes) / MiB,
                     rate(Counter::ComputeSyncUploads), rate(Counter::RangeCacheHits),
                     rate(Counter::RangeCacheMisses), rate(Counter::StorageBindingHits),
                     rate(Counter::StorageBindingMisses), dirty_walk);
        }
    }
    if (!t_compute_programs.empty()) {
        std::vector<std::pair<u64, u64>> programs; // count, shader hash
        programs.reserve(t_compute_programs.size());
        for (const auto& [hash, count] : t_compute_programs) {
            programs.emplace_back(count, hash);
        }
        const size_t top = (std::min<size_t>)(programs.size(), 8);
        std::partial_sort(programs.begin(), programs.begin() + top, programs.end(),
                          [](const auto& a, const auto& b) { return a > b; });
        std::string text;
        for (size_t i = 0; i < top; ++i) {
            text += fmt::format(" {:016x} {:.0f},", programs[i].second,
                                static_cast<double>(programs[i].first) * 1e9 / window);
        }
        text.pop_back();
        LOG_INFO(HW_GPU, "{} compute programs per second ({} programs, top {}):{}", thread_name,
                 programs.size(), top, text);
        t_compute_programs.clear();
    }
    t_method_count = {};
    t_method_same = {};
    t_dma_reads = {};
    // The zones open around this call (Report runs inside the split's back loop) stay open.
    const u32 depth = data.depth;
    data = {};
    data.depth = depth;
    data.window_start = now;
    data.window_start_ticks = now_ticks;
    data.cpu_start_ns = cpu_now;
}

} // namespace VideoCommon::Nx2Profile
