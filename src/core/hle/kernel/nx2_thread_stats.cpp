// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "common/cpu_features.h"
#include "common/logging.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/nx2_thread_stats.h"

namespace Kernel::Nx2ThreadStats {

namespace {

constexpr u64 ReportSeconds = 10;
constexpr size_t ReportThreads = 8;

// A thread's totals: guest counter ticks, except the counts and HeldNs.
enum ThreadField : size_t {
    OnCore,
    Jit,
    Poll,
    TightPoll,
    HintSpin,
    Memory,
    MemoryCalls,
    ReadBack,
    ReadBacks,
    Svcs,
    Ready,
    HeldNs,
    Holds,
    Presents,
    WaitTicks,
    WaitCounts = WaitTicks + Waits,
    ThreadFields = WaitCounts + Waits,
};

// A core's totals, written only by the host thread that runs the core.
enum CoreField : size_t {
    CoreJit,
    CorePoll,
    CoreHintSpin,
    Reads,
    PollReads,
    BlockLookups,
    BlocksCompiled,
    CacheClears,
    ExitCounts,
    HintCounts = ExitCounts + Exits,
    CoreFields = HintCounts + Hints,
};

template <size_t N>
using Totals = std::array<std::atomic<u64>, N>;
template <size_t N>
using Snapshot = std::array<u64, N>;

constexpr std::array<const char*, Waits> WaitNames{
    "IPC", "sync objects", "address", "condvar", "lock", "sleep", "other",
};

} // Anonymous namespace

struct Counters {
    u64 thread_id{};
    std::string process;
    u64 affinity{};
    std::atomic<s32> priority{};
    std::atomic<s64> ready_since{}; ///< 0 unless runnable and off its core
    std::atomic<s64> wait_since{};  ///< 0 unless waiting
    std::atomic<u8> wait_kind{};
    std::atomic<u32> svc{}; ///< The thread's last SVC: what it waits in
    Totals<ThreadFields> totals{};
};

namespace {

struct Entry {
    std::unique_ptr<Counters> counters;
    Snapshot<ThreadFields> previous{};
};

std::mutex g_registry_mutex;
std::vector<Entry> g_registry;

std::array<Totals<CoreFields>, Core::Hardware::NUM_CPU_CORES> g_cores{};
// PresentClientBuffer calls; unlike Core::g_nx2_present_count, a new session doesn't reset it.
std::atomic<u64> g_presents{};

// The report's state, under g_report_mutex.
std::mutex g_report_mutex;
std::atomic<u64> g_next_report{};
u64 g_window_ticks{};
s64 g_window_ns{};
u64 g_window_presents{};
std::array<Snapshot<CoreFields>, Core::Hardware::NUM_CPU_CORES> g_cores_previous{};

// For totals only one thread writes at a time: no locked instruction.
void Add(std::atomic<u64>& total, u64 amount) {
    total.store(total.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

void AddShared(std::atomic<u64>& total, u64 amount) {
    total.fetch_add(amount, std::memory_order_relaxed);
}

template <size_t N>
Snapshot<N> Delta(const Totals<N>& totals, Snapshot<N>& previous) {
    Snapshot<N> delta;
    for (size_t i = 0; i < N; ++i) {
        const u64 current = totals[i].load(std::memory_order_relaxed);
        delta[i] = current - previous[i];
        previous[i] = current;
    }
    return delta;
}

s64 SteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// The thread's counters, made on first use; nullptr for idle and host (dummy) threads.
Counters* Get(KThread* thread) {
    if (thread == nullptr || thread->IsDummyThread() || thread->GetOwnerProcess() == nullptr) {
        return nullptr;
    }
    auto& slot = thread->Nx2Stats();
    if (Counters* counters = slot.load(std::memory_order_acquire)) {
        return counters;
    }
    auto made = std::make_unique<Counters>();
    made->thread_id = thread->GetThreadId();
    made->process = thread->GetOwnerProcess()->GetName();
    made->affinity = thread->GetAffinityMask().GetAffinityMask();
    made->priority = thread->GetPriority();
    Counters* expected = nullptr;
    if (!slot.compare_exchange_strong(expected, made.get(), std::memory_order_acq_rel)) {
        return expected;
    }
    std::scoped_lock lk{g_registry_mutex};
    return g_registry.emplace_back(Entry{std::move(made)}).counters.get();
}

void Report(u64 now) {
    const s64 now_ns = SteadyNs();
    const u64 presents = g_presents.load(std::memory_order_relaxed) - g_window_presents;
    const double frames = static_cast<double>(std::max<u64>(presents, 1));
    const double seconds = static_cast<double>(now_ns - g_window_ns) / 1e9;
    const double ns_per_tick = static_cast<double>(now_ns - g_window_ns) /
                               static_cast<double>(std::max<u64>(now - g_window_ticks, 1));
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) * ns_per_tick / 1e6 / frames;
    };
    const auto ms_ns = [&](u64 ns) { return static_cast<double>(ns) / 1e6 / frames; };
    const auto per_frame = [&](u64 count) { return static_cast<double>(count) / frames; };
    const auto per_second = [&](u64 count) {
        return static_cast<u64>(static_cast<double>(count) / seconds);
    };

    struct Row {
        const Counters* counters;
        Snapshot<ThreadFields> delta;
    };
    std::vector<Row> rows;
    {
        std::scoped_lock lk{g_registry_mutex};
        for (auto& entry : g_registry) {
            const auto delta = Delta(entry.counters->totals, entry.previous);
            if (delta[OnCore] != 0 || delta[Jit] != 0) {
                rows.push_back({entry.counters.get(), delta});
            }
        }
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.delta[OnCore] > b.delta[OnCore]; });

    LOG_INFO(Kernel, "Guest threads over {:.1f} s, {} presents, {} threads ran; ms per frame:",
             seconds, presents, rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& d = rows[i].delta;
        if (i >= ReportThreads && d[Presents] == 0) {
            continue;
        }
        const Counters& c = *rows[i].counters;
        const u64 accounted = d[Poll] + d[HintSpin] + d[Memory];
        const u64 game = d[Jit] > accounted ? d[Jit] - accounted : 0;
        const u64 outside = d[OnCore] > d[Jit] ? d[OnCore] - d[Jit] : 0;
        u64 waited = 0;
        for (size_t kind = 0; kind < Waits; ++kind) {
            waited += d[WaitTicks + kind];
        }
        std::string waits = fmt::format("{} {:.2f} in {:.1f} with replies held {:.2f} in {:.1f}",
                                        WaitNames[0], ms(d[WaitTicks]), per_frame(d[WaitCounts]),
                                        ms_ns(d[HeldNs]), per_frame(d[Holds]));
        for (size_t kind = 1; kind < Waits; ++kind) {
            waits += fmt::format(", {} {:.2f} in {:.1f}", WaitNames[kind],
                                 ms(d[WaitTicks + kind]), per_frame(d[WaitCounts + kind]));
        }
        const std::string presenter =
            d[Presents] != 0 ? fmt::format(" (presents {:.2f} a frame)", per_frame(d[Presents]))
                             : std::string{};
        LOG_INFO(Kernel,
                 "  t{}{} {} prio {} mask {:#x}: on core {:.2f} (JIT {:.2f}: game code {:.2f}, "
                 "counter polling {:.2f} of it {:.2f} tight, hint spin {:.2f}, cached memory {:.2f} "
                 "in {:.1f} calls with read-back {:.2f} in {:.1f}; outside the JIT {:.2f} with "
                 "{:.1f} SVCs), ready {:.2f}, waiting {:.2f} ({})",
                 c.thread_id, presenter, c.process, c.priority.load(std::memory_order_relaxed),
                 c.affinity, ms(d[OnCore]), ms(d[Jit]), ms(game), ms(d[Poll]), ms(d[TightPoll]),
                 ms(d[HintSpin]), ms(d[Memory]), per_frame(d[MemoryCalls]), ms(d[ReadBack]),
                 per_frame(d[ReadBacks]), ms(outside), per_frame(d[Svcs]), ms(d[Ready]),
                 ms(waited), waits);
    }

    for (size_t core = 0; core < g_cores.size(); ++core) {
        const auto d = Delta(g_cores[core], g_cores_previous[core]);
        u64 exits = 0;
        for (size_t exit = 0; exit < Exits; ++exit) {
            exits += d[ExitCounts + exit];
        }
        if (exits == 0) {
            continue;
        }
        const auto exits_by = [&](Exit exit) {
            return per_second(d[ExitCounts + static_cast<size_t>(exit)]);
        };
        LOG_INFO(Kernel,
                 "Guest core {} over {:.1f} s: JIT {:.2f} ms/frame (counter polling {:.2f}, hint "
                 "spin {:.2f}); JIT exits/s {} (SVC {}, interrupt {}, cache invalidation {}, other "
                 "{}); {} C++ block lookups/s, {} blocks compiled/s, {} full code cache clears; {} "
                 "counter reads/s ({} within 1.7 us of the last); hints/s WFI {}, WFE {}, SEV {}, "
                 "SEVL {}, YIELD {}",
                 core, seconds, ms(d[CoreJit]), ms(d[CorePoll]), ms(d[CoreHintSpin]),
                 per_second(exits), exits_by(Exit::Svc), exits_by(Exit::Interrupt),
                 exits_by(Exit::CacheInvalidation), exits_by(Exit::Other),
                 per_second(d[BlockLookups]), per_second(d[BlocksCompiled]), d[CacheClears],
                 per_second(d[Reads]), per_second(d[PollReads]), per_second(d[HintCounts]),
                 per_second(d[HintCounts + 1]), per_second(d[HintCounts + 2]),
                 per_second(d[HintCounts + 3]), per_second(d[HintCounts + 4]));
    }
}

void MaybeReport(u64 now) {
    std::unique_lock lk{g_report_mutex, std::try_to_lock};
    if (!lk.owns_lock() || now < g_next_report.load(std::memory_order_relaxed)) {
        return;
    }
    if (g_next_report.load(std::memory_order_relaxed) != 0) {
        Report(now);
    }
    g_window_ticks = now;
    g_window_ns = SteadyNs();
    g_window_presents = g_presents.load(std::memory_order_relaxed);
    g_next_report.store(now + ReportSeconds * Common::g_wall_clock.GetGuestCNTFRQ(),
                        std::memory_order_relaxed);
}

} // Anonymous namespace

void EndRun(KThread* thread, size_t core, u64 start, u64 end, Exit exit, u32 svc,
            u64 block_lookups, u64 blocks_compiled, u64 cache_clears) {
    const RunTally& run = t_run;
    const u64 ticks = end - start;
    if (core < g_cores.size()) {
        auto& c = g_cores[core];
        Add(c[CoreJit], ticks);
        Add(c[CorePoll], run.poll_ticks);
        Add(c[CoreHintSpin], run.hint_ticks);
        Add(c[Reads], run.reads);
        Add(c[PollReads], run.poll_reads);
        Add(c[ExitCounts + static_cast<size_t>(exit)], 1);
        for (size_t hint = 0; hint < Hints; ++hint) {
            Add(c[HintCounts + hint], run.hints[hint]);
        }
        c[BlockLookups].store(block_lookups, std::memory_order_relaxed);
        c[BlocksCompiled].store(blocks_compiled, std::memory_order_relaxed);
        c[CacheClears].store(cache_clears, std::memory_order_relaxed);
    }
    // Only the core that runs a thread writes these.
    if (Counters* counters = Get(thread)) {
        auto& t = counters->totals;
        Add(t[Jit], ticks);
        Add(t[Poll], run.poll_ticks);
        Add(t[TightPoll], run.tight_poll_ticks);
        Add(t[HintSpin], run.hint_ticks);
        Add(t[Memory], run.memory_ticks);
        Add(t[MemoryCalls], run.memory_calls);
        Add(t[ReadBack], run.read_back_ticks);
        Add(t[ReadBacks], run.read_backs);
        if (exit == Exit::Svc) {
            Add(t[Svcs], 1);
            counters->svc.store(svc, std::memory_order_relaxed);
        }
    }
    if (end >= g_next_report.load(std::memory_order_relaxed)) {
        MaybeReport(end);
    }
}

void OnSwitch(KThread* from, KThread* to, s64 now, s64 from_ticks) {
    if (Counters* c = Get(from)) {
        if (from_ticks > 0) {
            AddShared(c->totals[OnCore], static_cast<u64>(from_ticks));
        }
        if (from->GetState() == ThreadState::Runnable) {
            // Preempted or yielded: ready again from now.
            c->ready_since.store(now, std::memory_order_relaxed);
        }
    }
    if (Counters* c = Get(to)) {
        if (const s64 since = c->ready_since.exchange(0, std::memory_order_relaxed);
            since != 0 && now > since) {
            AddShared(c->totals[Ready], static_cast<u64>(now - since));
        }
        c->priority.store(to->GetPriority(), std::memory_order_relaxed);
    }
}

void OnStateChanged(KThread* thread, s64 now, bool was_waiting, bool now_runnable,
                    bool now_waiting) {
    Counters* const c = Get(thread);
    if (c == nullptr) {
        return;
    }
    if (const s64 since = c->wait_since.exchange(0, std::memory_order_relaxed);
        was_waiting && since != 0) {
        const size_t kind = c->wait_kind.load(std::memory_order_relaxed);
        if (now > since) {
            AddShared(c->totals[WaitTicks + kind], static_cast<u64>(now - since));
        }
        AddShared(c->totals[WaitCounts + kind], 1);
    }
    c->ready_since.store(now_runnable ? now : 0, std::memory_order_relaxed);
    if (now_waiting) {
        c->wait_kind.store(static_cast<u8>(WaitOf(c->svc.load(std::memory_order_relaxed))),
                           std::memory_order_relaxed);
        c->wait_since.store(now, std::memory_order_relaxed);
    }
}

void OnPresent(KThread* client) {
    g_presents.fetch_add(1, std::memory_order_relaxed);
    if (Counters* c = Get(client)) {
        AddShared(c->totals[Presents], 1);
    }
}

void AddHold(KThread* client, s64 ns) {
    if (Counters* c = Get(client); c != nullptr && ns > 0) {
        AddShared(c->totals[HeldNs], static_cast<u64>(ns));
        AddShared(c->totals[Holds], 1);
    }
}

} // namespace Kernel::Nx2ThreadStats
