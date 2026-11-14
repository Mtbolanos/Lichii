// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <thread>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __GLIBC__
#include <execinfo.h>
#endif
#include <cstring>
#include <string>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};

BackIdleTrace::FrontState WaitState(DrainSite site) {
    return site == DrainSite::ListWriters || site == DrainSite::ListSync ||
                   site == DrainSite::ListFlush
               ? BackIdleTrace::FrontState::Writers
               : BackIdleTrace::FrontState::Drain;
}

// A wait of the front for its back, counted by site while g_acquire_stats counts.
class CountedWait {
public:
    explicit CountedWait(DrainSite site_) : site{static_cast<size_t>(site_)} {
        if (g_acquire_counting) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~CountedWait() {
        if (!start) {
            return;
        }
        Count(g_acquire_stats.site_drains[site]);
        Count(g_acquire_stats.site_drain_ns[site],
              static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - *start)
                                   .count()));
    }

private:
    size_t site;
    std::optional<std::chrono::steady_clock::time_point> start;
};

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (!instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        stack += fmt::format(" {}+{:#x}", base_name ? base_name + 1 : name,
                             reinterpret_cast<uintptr_t>(frames[i]) -
                                 reinterpret_cast<uintptr_t>(module));
    }
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on thread {} while the back runs; stack:{}",
              where, thread, stack);
#else
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (on_back_thread || !instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
#ifdef __GLIBC__
    void* frames[40];
    const int count = backtrace(frames, 40);
    if (char** const symbols = backtrace_symbols(frames, count)) {
        for (int i = 1; i < count; ++i) {
            stack += fmt::format("\n  {}", symbols[i]);
        }
        std::free(symbols);
    }
#endif
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on a thread other than the back; stack:{}", where,
              stack);
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    const char* const verify = std::getenv("EDEN_NX2_ACQUIRE_VERIFY");
    g_acquire_verify = verify != nullptr && verify[0] == '1';
    g_acquire_counting = g_acquire_verify || Nx2Profile::Enabled() ||
                         std::getenv("EDEN_NX2_ACQUIRE_STATS") != nullptr ||
                         std::getenv("EDEN_NX2_GUEST_THREAD_STATS") != nullptr;
    ring = std::make_unique<u64[]>(RingWords);
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
    ++starts;
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}",
             front_macros ? ", macros on the first" : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    last_sync_position = 0;
    synced_writer_position = 0;
    semaphore_writes.clear();
    semaphore_writes_limit = 1024;
    last_any_write = 0;
    last_noted_write = 0;
    write_positions.fill(0);
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    state_engine = nullptr;
    state_count = 0;
    state_breaks = false;
    state_anywhere = false;
    state_flags.fill(0);
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_idle_at.store(~u64{0}, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const CountedWait counted{DrainSite::RingFull};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            Nx2Profile::AddDrain(DrainSite::RingFull, start);
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::Publish() {
    FlushState();
    if (published_local == write_position) {
        return;
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void GpuSplit::Drain(DrainSite site) {
    FlushState();
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position && BackQuiet()) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const BackIdleTrace::FrontWait idle_trace{WaitState(site)};
    const CountedWait counted{site};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    // Everything consumed is not enough: the back submits deferred work (Scheduler::DeferFlush)
    // from its idle hook afterwards, and the caller may use the scheduler once this returns.
    bool hook_asked = false;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            if (BackQuiet()) {
                publish_every = 8;
                Nx2Profile::AddDrain(site, start);
                return;
            }
            if (!hook_asked) {
                RequestIdleHook();
                hook_asked = true;
            }
            // Backoff sleeps only while entries are unconsumed: spin until the hook is done.
            if (spins < 16384) {
                Pause();
            } else {
                std::this_thread::yield();
            }
            continue;
        }
        Backoff(spins, write_position);
    }
}

bool GpuSplit::WaitForWriters() {
    return WaitFor(std::min(last_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitForSync() {
    return WaitFor(std::min(last_sync_position, write_position), DrainSite::ListSync);
}

void GpuSplit::NoteSemaphoreWrite(u64 device_address, u32 payload, bool known) {
    if (semaphore_writes.size() >= semaphore_writes_limit) {
        // The words of releases the back ran hold their values in memory.
        const u64 done = consumed.load(std::memory_order_acquire);
        boost::unordered::erase_if(
            semaphore_writes, [done](const auto& entry) { return entry.second.position <= done; });
        semaphore_writes_limit = std::max<size_t>(1024, semaphore_writes.size() * 2);
    }
    semaphore_writes.insert_or_assign(device_address & ~u64{3},
                                      SemaphoreWrite{write_position, payload, known});
}

std::optional<GpuSplit::QueuedSemaphoreWrite> GpuSplit::QueuedSemaphoreWriteAt(
    u64 device_address) const {
    const auto it = semaphore_writes.find(device_address & ~u64{3});
    if (it == semaphore_writes.end() ||
        it->second.position <= consumed.load(std::memory_order_acquire)) {
        return std::nullopt;
    }
    return QueuedSemaphoreWrite{it->second.payload, it->second.known};
}

namespace {

/// Calls `func(device_address, size)` for each mapped piece of [address, address + size) within
/// one 4 KiB GPU page; nothing is read or written where no device memory is mapped.
template <typename Func>
void ForEachDeviceRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                        Func&& func) {
    constexpr GPUVAddr page_size = 0x1000;
    const GPUVAddr end = address + size;
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = std::min((begin & ~(page_size - 1)) + page_size, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            func(*device, page_end - begin);
        }
        begin = page_end;
    }
}

} // Anonymous namespace

void GpuSplit::NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    if (size > MaxNotedBytes || address + size < address) {
        NoteAnyWrite();
        return;
    }
    ForEachDeviceRange(memory, address, size,
                       [this](DAddr device, u64 bytes) { NoteDeviceWrite(device, bytes); });
}

void GpuSplit::NoteDeviceWrite(DAddr address, u64 size) {
    const u64 last = (address + size - 1) >> WriteGranuleBits;
    for (u64 granule = address >> WriteGranuleBits; granule <= last; ++granule) {
        write_positions[WriteSlot(granule)] = write_position;
    }
    last_noted_write = write_position;
}

u64 GpuSplit::WritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) const {
    if (size > MaxNotedBytes || address + size < address) {
        return std::max(last_any_write, last_noted_write);
    }
    u64 target = last_any_write;
    ForEachDeviceRange(memory, address, size, [this, &target](DAddr device, u64 bytes) {
        target = std::max(target, DeviceWritesTarget(device, bytes));
    });
    return target;
}

u64 GpuSplit::DeviceWritesTarget(DAddr address, u64 size) const {
    u64 target = 0;
    const u64 last = (address + size - 1) >> WriteGranuleBits;
    for (u64 granule = address >> WriteGranuleBits; granule <= last; ++granule) {
        target = std::max(target, write_positions[WriteSlot(granule)]);
    }
    return target;
}

bool GpuSplit::WaitForWrites(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    FlushState(); // A block that may issue a deferred draw notes its write anywhere.
    return WaitFor(std::min(WritesTarget(memory, address, size), write_position),
                   DrainSite::MacroDmaWrites);
}

bool GpuSplit::WaitForSyncedWriters() {
    return WaitFor(std::min(synced_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    FlushState();
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    const BackIdleTrace::FrontWait idle_trace{WaitState(site)};
    const CountedWait counted{site};
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            Nx2Profile::AddDrain(site, start);
            return true;
        }
        Backoff(spins, target);
    }
}

void GpuSplit::Backoff(u32 spins, u64 target) {
    if (spins < 16384) {
        Pause();
        return;
    }
    if (!back_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        return;
    }
    // A long wait sleeps until the back stores a new position (port of #161's front_wake): the
    // back checks front_waiting after each store, the front checks the position after setting it.
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) < target) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::FlushStateSlow() {
    // Cleared first: the entry's own SelectEngine, Reserve and Commit flush again.
    const u32 count = std::exchange(state_count, 0);
    SelectEngine(state_engine);
    const size_t size = 1 + StateFlagWords + count;
    u64* const entry = Reserve(size);
    entry[0] = static_cast<u64>(Kind::State) | (state_breaks ? HasBreak : 0) |
               (u64{state_break & 0xffff} << 8) | (u64{count} << 32);
    std::memcpy(entry + 1, state_flags.data(), sizeof(state_flags));
    std::memcpy(entry + 1 + StateFlagWords, state_writes.data(), size_t{count} * sizeof(u64));
    Commit(size);
    if (state_anywhere) {
        NoteAnyWrite();
    }
    state_breaks = false;
    state_anywhere = false;
    state_flags.fill(0);
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment,
                           Writes writes) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
    if (writes == Writes::Anywhere) {
        NoteAnyWrite();
    }
}

void GpuSplit::DeferFunction(std::function<void()>&& func) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer([heap] {
        (*heap)();
        delete heap;
    });
}

size_t GpuSplit::Execute(const u64* entry, size_t words) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if (engine->fast_maxwell3d) {
            // Nx2Perf: as below without the profile zones (off with fast_maxwell3d) and the
            // virtual call.
            using Tegra::Engines::Maxwell3D;
            auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
            constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
            if (method - cb_data < 16 && (head & HasSegment) == 0) {
                // Single constant buffer words in a row (macros write them one by one) go up in
                // one write. They are executable: each entry would only consume the sink first.
                std::array<u32, 64> methods;
                std::array<u32, 64> arguments;
                methods[0] = method;
                arguments[0] = argument;
                u32 count = 1;
                while (count < methods.size() && count < words) {
                    const u64 next = entry[count];
                    const u32 next_method = static_cast<u32>((next >> 8) & 0xffff);
                    if (static_cast<Kind>(next & 0xf) != Kind::Method || (next & HasSegment) != 0 ||
                        next_method - cb_data >= 16) {
                        break;
                    }
                    methods[count] = next_method;
                    arguments[count] = static_cast<u32>(next >> 32);
                    ++count;
                }
                maxwell3d->ConsumeSink(*system);
                if (count == 1) {
                    maxwell3d->CallMethod(*system, method, argument, true);
                } else {
                    maxwell3d->CallConstBufferWords(*system, methods.data(), arguments.data(),
                                                    count);
                }
                return count;
            }
            if ((head & Direct) != 0) {
                maxwell3d->ConsumeSink(*system);
                maxwell3d->CallMethod(*system, method, argument, true);
            } else if (!maxwell3d->execution_mask[method]) {
                maxwell3d->method_sink.emplace_back(method, argument);
            } else {
                maxwell3d->ConsumeSink(*system);
                if ((head & HasSegment) != 0) {
                    maxwell3d->current_dma_segment = entry[1];
                }
                maxwell3d->CallMethod(*system, method, argument, (head & IsLast) != 0);
            }
            return (head & HasSegment) != 0 ? 2 : 1;
        }
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        if ((head & Direct) != 0) {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::State: {
        const u32 count = static_cast<u32>(head >> 32);
        const u32 last_break = (head & HasBreak) != 0 ? static_cast<u32>((head >> 8) & 0xffff)
                                                      : Tegra::Engines::Maxwell3D::NoBreak;
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->WriteStateRegisters(*system, std::span<const u64, StateFlagWords>{entry + 1, StateFlagWords},
                                  std::span<const u64>{entry + 1 + StateFlagWords, count}, last_break);
        return 1 + StateFlagWords + count;
    }
    case Kind::Call: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return 2 + payload;
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    const StutterTrace::ThreadScope stutter_thread{StutterTrace::Thread::Back};
    Common::SetCurrentThreadName("GPUBack");
    on_back_thread = true;
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (Settings::values.pin_gpu_threads.GetValue()) {
        Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    }
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    const bool idle_trace = BackIdleTrace::Enabled();
    u64 idle_since = 0;
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread", true);
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            if (idle_trace && idle_since == 0) {
                idle_since = BackIdleTrace::Now();
            }
            // The front publishes what it holds after a few more words from now on.
            back_idle_at.store(read_position, std::memory_order_relaxed);
            const auto spin_start = g_acquire_counting ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position &&
                                !hook_requested.load(std::memory_order_relaxed);
                 ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (g_acquire_counting) {
                Count(g_acquire_stats.back_waits);
                Count(g_acquire_stats.back_spin_ns,
                      static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - spin_start)
                                           .count()));
            }
            if (available == read_position) {
                hook_requested.store(false, std::memory_order_seq_cst);
                // The front may already see everything consumed: Drain() waits for back_quiet
                // too, so the deferred submit can't overlap whatever the front does next.
                if (!back_quiet.load(std::memory_order_relaxed)) {
                    if (idle_hook != nullptr) {
                        idle_hook(idle_argument);
                    }
                    back_quiet.store(true, std::memory_order_release);
                }
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                // A request made after the hook ran (RequestIdleHook) runs it before sleeping.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst)) {
                    Count(g_acquire_stats.back_sleeps);
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        if (idle_since != 0) {
            BackIdleTrace::NoteBackIdle(idle_since, BackIdleTrace::Now());
            idle_since = 0;
        }
        u64 last_store = read_position;
        // Before any entry runs, so a front that sees these entries consumed also sees this.
        back_quiet.store(false, std::memory_order_seq_cst);
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            const size_t size =
                Execute(&ring[index], std::min<u64>(available - read_position, RingWords - index));
            read_position += size != 0 ? size : RingWords - index;
            if (hook_requested.load(std::memory_order_relaxed) &&
                idle_hook != nullptr) [[unlikely]] {
                hook_requested.store(false, std::memory_order_seq_cst);
                idle_hook(idle_argument);
            }
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                consumed.store(read_position, std::memory_order_seq_cst);
                last_store = read_position;
                NotifyFront();
            }
        }
        consumed.store(read_position, std::memory_order_seq_cst);
        NotifyFront();
    }
    back_running.store(false, std::memory_order_seq_cst);
    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

void GpuSplit::ReportAcquire() {
    static u32 calls{};
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    if (++calls % 256 != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds{10};
    auto& stats = g_acquire_stats;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 passed = take(stats.passed);
    const u64 by_queued = take(stats.by_queued);
    const u64 latched = take(stats.latched);
    const u64 drained = take(stats.drained);
    const u64 drain_operation = take(stats.drain_operation);
    const u64 drain_settings = take(stats.drain_settings);
    const u64 drain_word = take(stats.drain_word);
    const u64 drain_queued = take(stats.drain_queued);
    const u64 queued_unknown = take(stats.queued_unknown);
    const u64 queued_unknown_met = take(stats.queued_unknown_met);
    const u64 trigger_not_met = take(stats.trigger_not_met);
    const u64 trigger_not_met_twice = take(stats.trigger_not_met_twice);
    const u64 valve_drains = take(stats.valve_drains);
    const u64 polls = take(stats.polls);
    const u64 acquire_polls = take(stats.acquire_polls);
    const u64 list_polls = take(stats.list_polls);
    const u64 front_polls = take(stats.front_polls);
    const u64 front_sleeps = take(stats.front_sleeps);
    const u64 front_woken = take(stats.front_woken);
    const u64 front_stored = take(stats.front_stored);
    const u64 front_found_polling = take(stats.front_found_polling);
    const u64 front_found_late = take(stats.front_found_late);
    const u64 early_publishes = take(stats.early_publishes);
    const u64 settles = take(stats.settles);
    const u64 back_waits = take(stats.back_waits);
    const u64 back_sleeps = take(stats.back_sleeps);
    const double spin_ms = static_cast<double>(take(stats.back_spin_ns)) / 1e6;
    LOG_INFO(HW_GPU,
             "Acquires: {} passed without a drain ({} by a queued release), {} resumed by a "
             "latch, {} drained ({} by the 1 ms valve; {} equal or mask, {} by the settings, {} "
             "unmapped or GPU-written word, {} queued release unknown or short), {} polls waited "
             "without a drain in 10 s; polls: {} of acquires, {} of GPU-written list copies; "
             "back: {} waits for work, {} slept, {:.1f} ms spinning; {} early publishes, {} idle "
             "hooks asked by polls",
             passed, by_queued, latched, drained, valve_drains, drain_operation, drain_settings,
             drain_word, drain_queued, polls, acquire_polls, list_polls, back_waits, back_sleeps,
             spin_ms, early_publishes, settles);
    LOG_INFO(HW_GPU,
             "GPU thread waits for suspended channels in 10 s: {} in hardware, {} in the OS, {} "
             "ended by a wake, {} by a store to a semaphore word; channels that moved without "
             "either: {} found while polling, {} found late in the OS wait",
             front_polls, front_sleeps, front_woken, front_stored, front_found_polling,
             front_found_late);
    std::string sites;
    for (size_t site = 0; site < drain_site_names.size(); ++site) {
        const u64 count = take(stats.site_drains[site]);
        const u64 ns = take(stats.site_drain_ns[site]);
        if (count != 0) {
            sites += fmt::format("{}{} {} ({:.1f} ms)", sites.empty() ? "" : ", ",
                                 drain_site_names[site], count, static_cast<double>(ns) / 1e6);
        }
    }
    LOG_INFO(HW_GPU,
             "Front waits for the back in 10 s by site: {}; acquire methods not met without a "
             "drain: {} ({} still not met after it, which drain again to release the fences); "
             "drains for a queued release of unknown value (a reduction or a counter) to a word memory "
             "already meets: {} of {}",
             sites.empty() ? "none" : sites, trigger_not_met, trigger_not_met_twice,
             queued_unknown_met, queued_unknown);
    const u64 mismatches = take(stats.mismatches);
    const u64 unlanded = take(stats.unlanded);
    const u64 late = take(stats.late);
    if (mismatches != 0 || unlanded != 0 || late != 0) {
        LOG_WARNING(HW_GPU,
                    "Acquire verify: {} passed but not met after a drain, {} met by a "
                    "queued release not in memory after it, {} polls waited though a drain meets "
                    "them, in 10 s",
                    mismatches, unlanded, late);
    }
}

} // namespace VideoCommon
