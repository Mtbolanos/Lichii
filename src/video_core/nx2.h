// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/bit_util.h"
#include "common/common_types.h"
#include "common/thread.h"

namespace Tegra {

// Set once a Switch 2 (Ampere, NVN2) driver is talking to the GPU. Its command streams use
// Ampere classes and the Turing+ macro ISA, which Eden's Maxwell engines don't implement.
inline std::atomic<bool> g_nx2_ampere{false};

// The frame interval nx2:vi2 paces the game's presents to, in ns (0: not paced there, or not
// known). Frame generation plans against it (FrameGenPacer).
inline std::atomic<u64> g_nx2_guest_frame_interval_ns{0};

// GPU VA that nx2:gpu GetVaRange and GetUserPartition report to NVN. The in-process
// allocator hands out addresses in [Nx2GpuVaStart, Nx2GpuVaEnd). NvRm's AllocAsEx range
// can be smaller; the GMMU page tables still have to cover this one.
inline constexpr u64 Nx2GpuVaStart = 0x8000000ULL;
inline constexpr u64 Nx2GpuVaEnd = 0xC000000000ULL;

/// Bytes a Switch 2 GMMU covers when it is sized from Nx2GpuVaEnd. That end is not a
/// power of two, so the page tables round up (Log2Ceil). NvRm's va_range_end can only
/// make this larger. A range past it is left mapped, and its backing must stay.
inline constexpr u64 Nx2GpuAddressSpaceSize = 1ULL << Common::Log2Ceil(Nx2GpuVaEnd);
static_assert(Nx2GpuAddressSpaceSize == (1ULL << 40));

// Device pages the GPU has written through the buffer cache (shader stores, copies), ever. Title 1
// runs command lists the GPU writes each frame (20-byte entries its culling fills in); read
// without a flush they are stale and the draws behind them (trees) go missing. The bits never
// clear: the dispatch that rewrites a page may still be queued on the split's back when its front
// reads the command list. One bit per 4 KiB of the 36-bit device address space (2 MiB).
inline constexpr u64 Nx2WrittenPageBits = 12;
inline std::array<std::atomic<u64>, (u64{1} << (36 - Nx2WrittenPageBits)) / 64> g_nx2_gpu_written_pages{};

inline void Nx2MarkGpuWritten(u64 device_address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 last = std::min<u64>((device_address + size - 1) >> Nx2WrittenPageBits,
                                   g_nx2_gpu_written_pages.size() * 64 - 1);
    // A word at a time: storage buffers of several MiB are marked on every draw that binds them.
    for (u64 page = device_address >> Nx2WrittenPageBits; page <= last;) {
        const u64 first_bit = page % 64;
        const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
        const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
        auto& word = g_nx2_gpu_written_pages[page / 64];
        if ((word.load(std::memory_order_relaxed) & mask) != mask) {
            word.fetch_or(mask, std::memory_order_relaxed);
        }
        page += bits;
    }
}

// GPU-written device pages the front has read command lists from (same layout as above). The
// buffer cache copies these pages into host memory right after the compute or copy work that
// writes them and submits, so the front reads the copy once the GPU is done instead of draining
// the split and flushing the caches (BufferCache::Nx2ReadListCopy).
inline std::array<std::atomic<u64>, (u64{1} << (36 - Nx2WrittenPageBits)) / 64> g_nx2_command_list_pages{};
inline std::atomic<bool> g_nx2_has_command_list_pages{false};

inline void Nx2MarkCommandListPage(u64 device_address) {
    const u64 page = device_address >> Nx2WrittenPageBits;
    if (page >= g_nx2_command_list_pages.size() * 64) {
        return;
    }
    auto& word = g_nx2_command_list_pages[page / 64];
    const u64 bit = u64{1} << (page % 64);
    if ((word.load(std::memory_order_relaxed) & bit) == 0) {
        word.fetch_or(bit, std::memory_order_relaxed);
        g_nx2_has_command_list_pages.store(true, std::memory_order_relaxed);
    }
}

[[nodiscard]] inline bool Nx2IsCommandListPage(u64 device_address) {
    const u64 page = device_address >> Nx2WrittenPageBits;
    if (page >= g_nx2_command_list_pages.size() * 64) {
        return false;
    }
    return (g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1;
}

/// A new emulation session starts: no page has been written by its GPU yet.
inline void Nx2ClearGpuWrittenPages() {
    for (auto& word : g_nx2_gpu_written_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    for (auto& word : g_nx2_command_list_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    g_nx2_has_command_list_pages.store(false, std::memory_order_relaxed);
}

[[nodiscard]] inline bool Nx2IsGpuWritten(u64 device_address) {
    const u64 page = device_address >> Nx2WrittenPageBits;
    if (page >= g_nx2_gpu_written_pages.size() * 64) {
        return false;
    }
    return (g_nx2_gpu_written_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1;
}

/// False only when no page of [device_address, device_address + size) has been GPU-written.
/// Every range the buffer cache records as GPU-modified also marks its pages here, under its
/// lock, so this answers MustFlushRegion without taking it. Ranges past the bitmap answer true.
[[nodiscard]] inline bool Nx2MayBeGpuWritten(u64 device_address, u64 size) {
    if (size == 0) {
        return false;
    }
    const u64 first = device_address >> Nx2WrittenPageBits;
    const u64 last = (device_address + size - 1) >> Nx2WrittenPageBits;
    if (last < first || last >= g_nx2_gpu_written_pages.size() * 64) {
        return true;
    }
    for (u64 page = first; page <= last; ++page) {
        if ((g_nx2_gpu_written_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1) {
            return true;
        }
    }
    return false;
}

// NVN2 queues release their progress values into the second page of a two-page GPU-accessible
// region and GPU acquires read the same offsets in the first page (the CPU polls the second).
// Something on the console copies one into the other; until that is known, the puller mirrors
// releases that land in a second page. Every queue has such a region, and games make well over
// 32 of them. Region starts are page aligned, so a lookup is one hash probe.
inline std::shared_mutex g_nx2_semaphore_regions_mutex;
inline std::unordered_set<u64> g_nx2_semaphore_regions; // starts

inline void Nx2AddSemaphoreRegion(u64 start) {
    std::unique_lock lock{g_nx2_semaphore_regions_mutex};
    g_nx2_semaphore_regions.insert(start);
}

/// The region's memory was destroyed: stop mirroring into it.
inline void Nx2RemoveSemaphoreRegion(u64 start) {
    std::unique_lock lock{g_nx2_semaphore_regions_mutex};
    g_nx2_semaphore_regions.erase(start);
}

[[nodiscard]] inline bool Nx2IsSemaphoreRegion(u64 start) {
    std::shared_lock lock{g_nx2_semaphore_regions_mutex};
    return g_nx2_semaphore_regions.contains(start);
}

inline bool Nx2InSemaphoreFirstPage(u64 address) {
    return Nx2IsSemaphoreRegion(address & ~u64{0xfff});
}

/// Some first-page words also get direct releases of their own counter (a queue's fence at
/// offset 0). A mirrored release into such a word would show the guest CPU another counter's
/// progress, so its waits end before the work they wait for has run. Mirrored releases into
/// those words go to this table instead, and only GPU acquires read it. Words only the mirror
/// writes keep getting guest memory writes. Keyed by device word address. Ported from
/// jpasqui/Eglantine f8941169650 and ca73069b2e.
class Nx2MirrorShadow {
public:
    static Nx2MirrorShadow& Get() {
        static Nx2MirrorShadow instance;
        return instance;
    }

    /// A release that is not the mirror's landed at gpu_address (device_address).
    void NoteDirectRelease(u64 device_address, u64 gpu_address) {
        if (!Nx2InSemaphoreFirstPage(gpu_address)) {
            return;
        }
        const u64 word = device_address & ~u64{3};
        std::scoped_lock lock{mutex};
        if (direct_words.size() < MaxDirectWords) {
            direct_words.insert(word);
            has_direct.store(true, std::memory_order_release);
        }
        words.erase(word);
    }

    /// A direct release that combines with the word it lands on (a QMD max or reduction) took guest
    /// memory from guest_before to guest_after. While the word holds a mirrored value, GPU acquires
    /// see combine(mirrored), what the word would hold without the shadow. Returns what they see.
    template <typename Combine>
    u32 NoteCombinedRelease(u64 device_address, u64 gpu_address, u32 guest_before, u32 guest_after,
                            Combine&& combine) {
        const auto mirrored = Read(device_address, guest_before);
        NoteDirectRelease(device_address, gpu_address);
        if (!mirrored) {
            return guest_after;
        }
        const u32 seen = combine(*mirrored);
        Set(device_address, seen, guest_after);
        return seen;
    }

    /// Whether a mirrored release into this word goes to the table instead of guest memory.
    bool Diverts(u64 device_address) const {
        if (!has_direct.load(std::memory_order_acquire)) {
            return false;
        }
        std::scoped_lock lock{mutex};
        return direct_words.contains(device_address & ~u64{3});
    }

    /// guest: what guest memory holds at the word now.
    void Set(u64 device_address, u32 value, u32 guest) {
        std::scoped_lock lock{mutex};
        words.insert_or_assign(device_address & ~u64{3}, Word{value, guest});
        has_words.store(true, std::memory_order_release);
    }

    /// The mirrored value of a word whose guest memory holds guest, or none. Once the guest word
    /// changed (a direct release or a CPU write), it is newer than the mirrored value.
    std::optional<u32> Read(u64 device_address, u32 guest) {
        if (!has_words.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        std::scoped_lock lock{mutex};
        const auto it = words.find(device_address & ~u64{3});
        if (it == words.end()) {
            return std::nullopt;
        }
        if (it->second.guest != guest) {
            words.erase(it);
            return std::nullopt;
        }
        return it->second.value;
    }

    void Clear() {
        std::scoped_lock lock{mutex};
        direct_words.clear();
        words.clear();
        has_direct.store(false, std::memory_order_release);
        has_words.store(false, std::memory_order_release);
    }

private:
    struct Word {
        u32 value;
        u32 guest;
    };
    static constexpr size_t MaxDirectWords = 1024;

    mutable std::mutex mutex;
    std::atomic<bool> has_direct{};
    std::atomic<bool> has_words{};
    std::unordered_set<u64> direct_words;
    std::unordered_map<u64, Word> words;
};

/// A new emulation session starts: no queue has a region yet.
inline void Nx2ClearSemaphoreRegions() {
    {
        std::unique_lock lock{g_nx2_semaphore_regions_mutex};
        g_nx2_semaphore_regions.clear();
    }
    Nx2MirrorShadow::Get().Clear();
}

// Switch 2 sparse regions (nx2:gpu 214, word 7 bit 0x200: title 1's terrain texture arrays and
// layer pools, 170 MB-5.8 GB each, of which the game fills a little) get memory per block when the
// GPU first writes there, not in full. nx2:gpu installs the handler and the span its regions cover;
// the GPU memory manager calls Nx2BackSparseWrite before each write, with the bytes it writes
// (blocks nobody wrote that only get zeros need no memory: they read zero already).
using Nx2SparseBackingHandler = std::function<void(u64 gpu_address, u64 size, const u8* data)>;
inline std::recursive_mutex g_nx2_sparse_backing_mutex;
inline Nx2SparseBackingHandler g_nx2_sparse_backing;
inline std::atomic<u64> g_nx2_sparse_low{~u64{0}};
inline std::atomic<u64> g_nx2_sparse_high{0};

inline void Nx2SetSparseBacking(Nx2SparseBackingHandler handler) {
    std::scoped_lock lk{g_nx2_sparse_backing_mutex};
    g_nx2_sparse_backing = std::move(handler);
    g_nx2_sparse_low.store(~u64{0});
    g_nx2_sparse_high.store(0);
}

/// Widens the span that writes are checked against (a new lazily backed region).
inline void Nx2AddSparseSpan(u64 gpu_address, u64 size) {
    u64 low = g_nx2_sparse_low.load();
    while (gpu_address < low && !g_nx2_sparse_low.compare_exchange_weak(low, gpu_address)) {
    }
    u64 high = g_nx2_sparse_high.load();
    while (gpu_address + size > high &&
           !g_nx2_sparse_high.compare_exchange_weak(high, gpu_address + size)) {
    }
}

inline void Nx2BackSparseWrite(u64 gpu_address, u64 size, const void* data) {
    if (gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return;
    }
    std::scoped_lock lk{g_nx2_sparse_backing_mutex};
    if (g_nx2_sparse_backing) {
        g_nx2_sparse_backing(gpu_address, size, static_cast<const u8*>(data));
    }
}

// Which blocks of each sparse region have been written (got memory), for the texture cache: an
// image over blocks nobody wrote is all zeros and needs no device memory. Its own lock, taken last:
// the texture cache asks with its lock held, and nx2:gpu updates it holding only its region mutex.
struct Nx2SparseRegionState {
    u64 size;
    u64 block_size;
    std::vector<bool> written;
    /// Once the game maps its own memory into the region (nx2:gpu 216): which tiles it maps.
    /// Only those hold data then (writes elsewhere are dropped, as on the console).
    std::vector<bool> mapped;
};

/// The granularity of the game's sparse tiles (nx2:gpu 216 maps 64 KiB pages).
inline constexpr u64 Nx2SparseTileSize = 64ULL << 10;
inline std::mutex g_nx2_sparse_state_mutex;
inline std::map<u64, Nx2SparseRegionState> g_nx2_sparse_state;

inline void Nx2SparseRegionCreated(u64 gpu_address, u64 size, u64 block_size) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    g_nx2_sparse_state[gpu_address] = {size, block_size,
                                       std::vector<bool>((size + block_size - 1) / block_size)};
}

inline void Nx2SparseRegionDestroyed(u64 gpu_address) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    g_nx2_sparse_state.erase(gpu_address);
}

/// A new emulation session: the previous game's regions are gone (nvdrv's Module clears them).
inline void Nx2ClearSparseRegions() {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    g_nx2_sparse_state.clear();
}

inline void Nx2SparseBlockWritten(u64 region_address, u64 block) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    if (const auto it = g_nx2_sparse_state.find(region_address); it != g_nx2_sparse_state.end()) {
        it->second.written[block] = true;
    }
}

/// True unless the block of the region at region_address is known to hold nothing (no write from
/// the CPU or the GPU yet).
inline bool Nx2SparseBlockIsWritten(u64 region_address, u64 block) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    const auto it = g_nx2_sparse_state.find(region_address);
    return it == g_nx2_sparse_state.end() || block >= it->second.written.size() ||
           it->second.written[block];
}

/// The game maps its own memory over [offset, +size) of the region at region_address (mapped), or
/// unmaps it there.
inline void Nx2SparseTilesMapped(u64 region_address, u64 offset, u64 size, bool mapped) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    const auto it = g_nx2_sparse_state.find(region_address);
    if (it == g_nx2_sparse_state.end() || size == 0) {
        return;
    }
    auto& region = it->second;
    if (region.mapped.empty()) {
        region.mapped.resize((region.size + Nx2SparseTileSize - 1) / Nx2SparseTileSize);
    }
    const u64 last = (std::min)((offset + size - 1) / Nx2SparseTileSize, u64{region.mapped.size() - 1});
    for (u64 tile = offset / Nx2SparseTileSize; tile <= last; ++tile) {
        region.mapped[tile] = mapped;
    }
}

/// The parts of [gpu_address, +size), which lies in one sparse region, that can hold data: the
/// tiles the game maps there if it maps any, otherwise the blocks written. As (offset from
/// gpu_address, size) runs.
inline std::vector<std::pair<u64, u64>> Nx2SparseDataRanges(u64 gpu_address, u64 size) {
    std::vector<std::pair<u64, u64>> ranges;
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    auto it = g_nx2_sparse_state.upper_bound(gpu_address);
    if (it == g_nx2_sparse_state.begin()) {
        return ranges;
    }
    --it;
    const auto& [start, region] = *it;
    if (size == 0 || gpu_address + size > start + region.size) {
        return ranges;
    }
    const bool by_tile = !region.mapped.empty();
    const u64 unit = by_tile ? Nx2SparseTileSize : region.block_size;
    const std::vector<bool>& present = by_tile ? region.mapped : region.written;
    const u64 begin = gpu_address - start;
    const u64 end = begin + size;
    for (u64 index = begin / unit; index * unit < end && index < present.size(); ++index) {
        if (!present[index]) {
            continue;
        }
        const u64 run_begin = (std::max)(index * unit, begin) - begin;
        const u64 run_end = (std::min)((index + 1) * unit, end) - begin;
        if (!ranges.empty() && ranges.back().first + ranges.back().second == run_begin) {
            ranges.back().second += run_end - run_begin;
        } else {
            ranges.emplace_back(run_begin, run_end - run_begin);
        }
    }
    return ranges;
}

/// True when [gpu_address, +size) lies in one sparse region.
inline bool Nx2InSparseRegion(u64 gpu_address, u64 size) {
    if (size == 0 || gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return false;
    }
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    auto it = g_nx2_sparse_state.upper_bound(gpu_address);
    if (it == g_nx2_sparse_state.begin()) {
        return false;
    }
    --it;
    return gpu_address + size <= it->first + it->second.size;
}

/// The GPU renders into [gpu_address, +size): count its blocks as written, so images made there
/// later get memory (their contents live in the host image, not in guest memory).
inline void Nx2SparseMarkWritten(u64 gpu_address, u64 size) {
    if (size == 0 || gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return;
    }
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    for (auto& [start, region] : g_nx2_sparse_state) {
        const u64 end = start + region.size;
        if (gpu_address >= end || gpu_address + size <= start) {
            continue;
        }
        const u64 first = ((std::max)(gpu_address, start) - start) / region.block_size;
        const u64 last = ((std::min)(gpu_address + size, end) - 1 - start) / region.block_size;
        for (u64 block = first; block <= last; ++block) {
            region.written[block] = true;
        }
    }
}

/// True when [gpu_address, +size) lies in one sparse region and none of it has been written.
inline bool Nx2SparseUnwritten(u64 gpu_address, u64 size) {
    if (size == 0 || gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return false;
    }
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    auto it = g_nx2_sparse_state.upper_bound(gpu_address);
    if (it == g_nx2_sparse_state.begin()) {
        return false;
    }
    --it;
    const auto& [start, region] = *it;
    if (gpu_address + size > start + region.size) {
        return false;
    }
    const u64 last = (gpu_address + size - 1 - start) / region.block_size;
    for (u64 block = (gpu_address - start) / region.block_size; block <= last; ++block) {
        if (region.written[block]) {
            return false;
        }
    }
    return true;
}

/// GPU VA in the first page that mirrors a release at `address`, or 0.
inline u64 Nx2SemaphoreMirror(u64 address) {
    return Nx2IsSemaphoreRegion((address & ~u64{0xfff}) - 0x1000) ? address - 0x1000 : 0;
}

/// Wakes the GPU thread while it waits for suspended channels (gpu_thread.cpp) on what lets a
/// channel go on: a new command (ThreadManager::PushCommand), a GPU write that latches a waiter
/// (Nx2SemaphoreWaiters::OnWrite) and the host GPU passing the tick of a command list copy a
/// channel waits for (OnGpuTick). A guest CPU store to a semaphore goes straight to memory through
/// fastmem and sends nothing: the waiting thread watches the words instead
/// (Nx2SemaphoreWaiters::WordStored).
class Nx2PendingWake {
public:
    static Nx2PendingWake& Get() {
        static Nx2PendingWake wake;
        return wake;
    }

    void Notify() {
        if (!woken.exchange(true, std::memory_order_seq_cst) &&
            sleeping.load(std::memory_order_seq_cst)) {
            event.Set();
        }
    }

    /// GPU thread: a channel waits for the host GPU to pass `tick` (DmaPusher::pending_list_tick).
    void WaitForGpuTick(u64 tick) {
        u64 wanted = wanted_tick.load(std::memory_order_seq_cst);
        while (tick < wanted &&
               !wanted_tick.compare_exchange_weak(wanted, tick, std::memory_order_seq_cst)) {
        }
    }

    /// The host GPU passed `tick` (Vulkan::MasterSemaphore): wakes a channel waiting for it.
    void OnGpuTick(u64 tick) {
        // Ordered after the store that made the tick known, which the waiter checks after
        // WaitForGpuTick: one of the two sees the other.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        u64 wanted = wanted_tick.load(std::memory_order_seq_cst);
        while (tick >= wanted) {
            if (wanted_tick.compare_exchange_weak(wanted, NoTick, std::memory_order_seq_cst)) {
                Notify();
                return;
            }
        }
    }

    /// Called before the waiter checks its channels: a Notify after it ends the next wait.
    void Arm() {
        woken.exchange(false, std::memory_order_acq_rel);
    }

    /// Waits in hardware (Common::WaitForFlag) until a Notify or the deadline; true if notified.
    bool Poll(std::chrono::steady_clock::time_point deadline) {
        return Common::WaitForFlag(woken, deadline);
    }

    /// Blocks in the OS until a Notify, for at most `time`; true if notified.
    bool Sleep(std::chrono::nanoseconds time) {
        sleeping.store(true, std::memory_order_seq_cst);
        if (!woken.load(std::memory_order_seq_cst)) {
            event.SleepFor(time);
        }
        sleeping.store(false, std::memory_order_relaxed);
        return woken.load(std::memory_order_acquire);
    }

private:
    static constexpr u64 NoTick = ~u64{0};

    alignas(64) std::atomic<bool> woken{};
    std::atomic<bool> sleeping{};
    Common::Event event;
    alignas(64) std::atomic<u64> wanted_tick{NoTick}; ///< Earliest tick a channel waits for
};

/// Channels suspended on a semaphore acquire. A console channel polls and sees every release;
/// Eden writes releases in bursts when fences signal and resumes waiters only between command
/// lists, so an acquire-equal could miss its value. Writes latch the waiters they satisfy.
/// The channels' acquires (Add, Remove, Latched) all run on the GPU thread (the split's front);
/// writes (OnWrite) come from the thread that runs the engines. Latched and WordStored take no
/// lock: they read slots only the GPU thread fills and empties.
class Nx2SemaphoreWaiters {
public:
    // Puller GpuSemaphoreOperation values.
    static constexpr u32 AcquireEqual = 0x1;
    static constexpr u32 AcquireGequal = 0x4;
    static constexpr u32 AcquireMask = 0x8;
    /// One per channel suspended at once.
    static constexpr size_t MaxWaiters = 64;

    static Nx2SemaphoreWaiters& Get() {
        static Nx2SemaphoreWaiters waiters;
        return waiters;
    }

    /// False, and nothing is latched for owner, when MaxWaiters others wait already. `word`, if
    /// set, is the semaphore in host memory (4-byte aligned), for WordStored.
    bool Add(const void* owner, u64 device_address, u32 value, u32 operation,
             u32* word = nullptr) {
        std::scoped_lock lk{mutex};
        Slot* free_slot = nullptr;
        for (Slot& slot : slots) {
            const void* const current = slot.owner.load(std::memory_order_relaxed);
            if (current == owner) {
                slot.owner.store(nullptr, std::memory_order_relaxed);
                count.fetch_sub(1, std::memory_order_relaxed);
            }
            if ((current == nullptr || current == owner) && free_slot == nullptr) {
                free_slot = &slot;
            }
        }
        if (free_slot == nullptr) {
            return false;
        }
        free_slot->address = device_address;
        free_slot->value = value;
        free_slot->operation = operation;
        free_slot->word = word;
        free_slot->seen =
            word != nullptr ? std::atomic_ref<u32>{*word}.load(std::memory_order_relaxed) : 0;
        free_slot->latched.store(false, std::memory_order_relaxed);
        free_slot->owner.store(owner, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_release);
        return true;
    }

    void Remove(const void* owner) {
        if (count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lk{mutex};
        for (Slot& slot : slots) {
            if (slot.owner.load(std::memory_order_relaxed) == owner) {
                slot.owner.store(nullptr, std::memory_order_relaxed);
                count.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    [[nodiscard]] bool Latched(const void* owner) const {
        if (count.load(std::memory_order_acquire) == 0) {
            return false;
        }
        for (const Slot& slot : slots) {
            if (slot.owner.load(std::memory_order_relaxed) == owner) {
                // Acquire: what the engine wrote before the latch is visible after it.
                return slot.latched.load(std::memory_order_acquire);
            }
        }
        return false;
    }

    /// GPU thread: whether a watched word changed in memory since it was added or since the last
    /// call. Guest CPU stores to a semaphore notify nothing; the thread checks this while it waits.
    [[nodiscard]] bool WordStored() {
        if (count.load(std::memory_order_acquire) == 0) {
            return false;
        }
        bool stored = false;
        for (Slot& slot : slots) {
            if (slot.word == nullptr || slot.owner.load(std::memory_order_relaxed) == nullptr) {
                continue;
            }
            const u32 current = std::atomic_ref<u32>{*slot.word}.load(std::memory_order_relaxed);
            if (current != slot.seen) {
                slot.seen = current;
                stored = true;
            }
        }
        return stored;
    }

    /// A GPU write of value to the u32 at device_address.
    void OnWrite(u64 device_address, u32 value) {
        if (count.load(std::memory_order_acquire) == 0) {
            return;
        }
        bool latched = false;
        {
            std::scoped_lock lk{mutex};
            for (Slot& slot : slots) {
                if (slot.owner.load(std::memory_order_relaxed) != nullptr &&
                    slot.address == device_address && Matches(slot, value)) {
                    slot.latched.store(true, std::memory_order_release);
                    latched = true;
                }
            }
        }
        if (latched) {
            Nx2PendingWake::Get().Notify();
        }
    }

private:
    struct Slot {
        std::atomic<const void*> owner{};
        u64 address{};
        u32 value{};
        u32 operation{};
        u32* word{}; ///< The semaphore in host memory, or null
        u32 seen{};  ///< *word when last looked at (WordStored)
        std::atomic<bool> latched{};
    };

    static bool Matches(const Slot& slot, u32 current) {
        switch (slot.operation) {
        case AcquireEqual:
            return current == slot.value;
        case AcquireGequal:
            return current >= slot.value;
        case AcquireMask:
            return (current & slot.value) != 0;
        default:
            return false;
        }
    }

    std::mutex mutex; ///< Held to fill, empty and latch slots
    std::array<Slot, MaxWaiters> slots{};
    std::atomic<size_t> count{};
};

} // namespace Tegra
