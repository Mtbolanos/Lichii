// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <compare>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "core/hle/service/nvdrv/devices/nvdevice.h"

namespace Service::Nvidia::Devices {

// Set once a Switch 2 NvRm has identified itself (0xD0-byte GetCharacteristics). Devices then
// answer ioctls they don't know with a logged, input-echoing success instead of NotImplemented.
inline std::atomic<bool> g_nx2_driver{false};

inline NvResult Nx2UnknownIoctl(const char* device, Ioctl command, std::span<const u8> input,
                                std::span<u8> output) {
    LOG_WARNING(Service_NVDRV, "(STUBBED) Switch 2 {} ioctl={:08X} group={:01X} command={:02X} in={:#x}",
                device, command.raw, command.group, command.cmd, input.size());
    std::fill(output.begin(), output.end(), u8{0});
    std::copy_n(input.begin(), std::min(input.size(), output.size()), output.begin());
    return NvResult::Success;
}

// Whether [start, +size) lies inside [span_start, +span_size), without overflowing near the top.
constexpr bool Nx2RangeInside(u64 start, u64 size, u64 span_start, u64 span_size) {
    return start >= span_start && start - span_start <= span_size &&
           size <= span_size - (start - span_start);
}

// Erases the device ranges of `ranges` (keyed by CPU start and size) that lie inside
// [cpu_address, +size) and passes each to `release`. A game maps parts of one memory at offsets
// (211), so the key a range was made under is rarely the span released later; ranges that reach
// past the span are kept.
template <typename Ranges, typename Release>
void Nx2EraseContainedRanges(Ranges& ranges, u64 cpu_address, u64 size, Release&& release) {
    for (auto it = ranges.lower_bound({cpu_address, 0});
         it != ranges.end() && it->first.first - cpu_address < size;) {
        const auto [range_cpu, range_size] = it->first;
        if (!Nx2RangeInside(range_cpu, range_size, cpu_address, size)) {
            ++it;
            continue;
        }
        const auto range = it->second;
        it = ranges.erase(it);
        release(range_cpu, range_size, range);
    }
}

// GPU mappings a Switch 2 process made through nx2:gpu: CreateGpuAccessibleMemory (sub 101) maps
// the memory at GPU address == CPU address, and NormalMapGpuMemory4K (211) maps a memory handle at
// a chosen GPU address (also in sparse regions). Every nvhost-as-gpu address space applies each
// mapping, including ones made before the address space existed.
//
// Listeners are called outside `mutex`: a listener can call back into the emulator
// (GPU::MapInOrder waits for the GPU thread, and the GPU thread's back reaches the registry
// from BackSparseWrite through BackSparse), so holding `mutex` across the call deadlocks
// (guest: MapAt under `mutex` waits for the GPU thread that waits for `mutex`).
//
// Order: each change takes a ticket under `mutex`, together with the listeners registered at
// that moment, and the calls run after the unlock strictly in ticket order. A listener sees
// the changes in the order they were made (a delayed map can't follow a later unmap of the
// same range), and a new listener's replay covers exactly the changes before it.
// BackSparse is the exception: the GPU thread's back calls it and must never wait for a
// guest thread's turn (that guest may be waiting for the GPU thread).
class Nx2GpuMemoryRegistry {
public:
    using Listener = std::function<void(u64 gpu_address, u64 cpu_address, u64 size)>;
    using UnmapListener = std::function<void(u64 gpu_address, u64 size)>;
    using ReleaseListener = std::function<void(u64 cpu_address, u64 size)>;
    /// A sparse region is created (true) or destroyed (false).
    using SparseListener = std::function<void(u64 gpu_address, u64 size, bool create)>;
    /// A block of a sparse region gets memory [cpu_address, +size).
    using BackListener = std::function<void(u64 gpu_address, u64 cpu_address, u64 size)>;
    /// A tile of a sparse region shows process memory [cpu_address, +size), or reads zero again
    /// (cpu_address 0).
    using TileListener = std::function<void(u64 gpu_address, u64 cpu_address, u64 size)>;

    static Nx2GpuMemoryRegistry& Get() {
        static Nx2GpuMemoryRegistry registry;
        return registry;
    }

    void Register(u64 address, u64 size) {
        MapAt(address, address, size);
    }

    void MapAt(u64 gpu_address, u64 cpu_address, u64 size) {
        MapAt(0, 0, gpu_address, cpu_address, size);
    }

    void MapAt(u32 handle, u64 offset, u64 gpu_address, u64 cpu_address, u64 size) {
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            TrimMappings(gpu_address, size);
            mappings.push_back({gpu_address, cpu_address, size, handle, offset});
            Collect(notification, &Listeners::map, gpu_address, cpu_address, size);
            Seal(notification);
        }
        Deliver(std::move(notification));
    }

    // NormalUnmapGpuMemory4K, or destroying the memory: the GPU range stops reaching it.
    void Unmap(u64 gpu_address, u64 size) {
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            TrimMappings(gpu_address, size);
            Collect(notification, &Listeners::unmap, gpu_address, size);
            Seal(notification);
        }
        Deliver(std::move(notification));
    }

    // Memory [cpu_address, +size) is destroyed (sub 102): every GPU range that still reaches it
    // stops reaching it, wherever it was mapped, because the game reuses the memory right after.
    // Returns the ranges other than the one at the memory's own address, as (GPU address, size).
    std::vector<std::pair<u64, u64>> UnmapMemory(u64 cpu_address, u64 size) {
        struct Unmapped {
            u64 gpu_address;
            u64 size;
            bool own; ///< The mapping at the memory's own address
        };
        std::vector<Unmapped> unmapped; ///< Every range that stopped reaching the memory
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            const u64 end = cpu_address + size;
            for (const auto& m : mappings) {
                const u64 first = (std::max)(m.cpu_address, cpu_address);
                const u64 last = (std::min)(m.cpu_address + m.size, end);
                if (first < last) {
                    unmapped.push_back({m.gpu_address + (first - m.cpu_address), last - first,
                                        m.gpu_address == m.cpu_address});
                }
            }
            for (const auto& range : unmapped) {
                TrimMappings(range.gpu_address, range.size);
                Collect(notification, &Listeners::unmap, range.gpu_address, range.size);
            }
            Seal(notification);
        }
        Deliver(std::move(notification));
        // The ranges other than the one at the memory's own address, for the caller.
        std::vector<std::pair<u64, u64>> leftovers;
        leftovers.reserve(unmapped.size());
        for (const auto& range : unmapped) {
            if (!range.own) {
                leftovers.emplace_back(range.gpu_address, range.size);
            }
        }
        return leftovers;
    }

    // Backing memory [cpu_address, +size) is about to be freed (its GPU mappings are gone): each
    // address space drops its SMMU mappings inside it, in GPU-thread order.
    void ReleaseBacking(u64 cpu_address, u64 size) {
        {
            std::scoped_lock lk{device_mutex};
            for (auto it = device_ranges.lower_bound(cpu_address);
                 it != device_ranges.end() && it->first - cpu_address < size;) {
                it = Nx2RangeInside(it->first, it->second.first, cpu_address, size)
                         ? device_ranges.erase(it)
                         : std::next(it);
            }
        }
        NotifyListeners(&Listeners::release, cpu_address, size);
    }

    /// Holds back the listener calls this thread makes (and the work it hands to Defer) until
    /// the outermost scope ends, then makes them in order. For callers that change the registry
    /// under a lock the GPU thread can wait for (nx2:gpu's region mutex): declared before that
    /// lock, the calls, which can wait for the GPU thread, run after it is released. Must not
    /// be open around RemoveListener, which waits for these calls.
    class [[nodiscard]] DeferScope {
    public:
        DeferScope() {
            ++ThisThread().depth;
        }
        ~DeferScope() {
            if (--ThisThread().depth == 0) {
                FlushThisThread();
            }
        }
        DeferScope(const DeferScope&) = delete;
        DeferScope& operator=(const DeferScope&) = delete;
    };

    /// Runs fn after the listener calls this thread holds back (DeferScope), or now if it holds
    /// none back. For GPU-thread work that has to be queued behind them.
    static void Defer(std::function<void()> fn) {
        auto& local = ThisThread();
        if (local.depth == 0) {
            fn();
            return;
        }
        Notification notification;
        notification.calls.push_back(std::move(fn));
        local.pending.push_back(std::move(notification));
    }

    // Sparse regions (nx2:gpu 214 with the sparse flag): the GPU range reaches device memory of its
    // own from the start, and each block gets memory when the GPU first writes it (BackSparse).
    // The GPU mapping never changes, so the texture cache doesn't rebuild the images over it as
    // blocks get memory; unbacked blocks read zero.
    void CreateSparse(u64 gpu_address, u64 size) {
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            sparse_regions[gpu_address] = size;
            Collect(notification, &Listeners::sparse, gpu_address, size, true);
            Seal(notification);
        }
        Deliver(std::move(notification));
    }

    void DestroySparse(u64 gpu_address, u64 size) {
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            sparse_regions.erase(gpu_address);
            Collect(notification, &Listeners::sparse, gpu_address, size, false);
            Seal(notification);
        }
        Deliver(std::move(notification));
    }

    // Called by the GPU thread's back (BackSparseWrite), before the write: runs right away, out of
    // ticket order and never held back, so the back never waits for a guest thread's turn. The
    // region it backs was created (and its creation delivered) before the guest could submit the
    // work that writes into it.
    // A block waiting for memory keeps what is written to it until BackSparse gives it some
    // (the back listener with cpu_address 0).
    void KeepSparseWrites(u64 gpu_address, u64 size) {
        BackSparse(gpu_address, 0, size);
    }

    void BackSparse(u64 gpu_address, u64 cpu_address, u64 size) {
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            Collect(notification, &Listeners::back, gpu_address, cpu_address, size);
            Count(notification);
        }
        Run(notification, false);
    }

    // nx2:gpu 216 (sparse map requests): the tile [gpu_address, +size) of a sparse region now
    // shows the process memory at cpu_address, the game's own memory (sub 101), or with
    // cpu_address 0 it is unmapped and reads zero. The GPU mapping stays as it is; only the
    // region's device pages change, in GPU-thread order.
    void MapSparseTile(u64 gpu_address, u64 cpu_address, u64 size) {
        NotifyListeners(&Listeners::tile, gpu_address, cpu_address, size);
    }

    void AddListener(const void* owner, Listener listener, UnmapListener unmap_listener,
                     ReleaseListener release_listener = {}, SparseListener sparse_listener = {},
                     BackListener back_listener = {}, TileListener tile_listener = {}) {
        // The replay of existing mappings runs outside `mutex`, like every other listener call:
        // the listener (an address space's Nx2Map) queues blocking GPU-thread work, and the GPU
        // thread's back reaches the registry from BackSparseWrite; under the mutex the replay
        // would close the same cycle MapAt used to. Its ticket is taken with the registration:
        // changes before it reach the new listener only through the replay, later ones only
        // after it.
        Notification replay;
        {
            std::scoped_lock lk{mutex};
            listeners.insert_or_assign(owner, Listeners{std::move(listener), std::move(unmap_listener),
                                                        std::move(release_listener),
                                                        std::move(sparse_listener),
                                                        std::move(back_listener),
                                                        std::move(tile_listener)});
            // Registered before the replay runs, so RemoveListener (which erases under the same
            // mutex and waits for in-flight calls) also protects the new owner during it.
            const auto& registered = listeners.at(owner);
            if (registered.map) {
                for (const auto& m : mappings) {
                    replay.calls.push_back([fn = registered.map, m] {
                        fn(m.gpu_address, m.cpu_address, m.size);
                    });
                }
            }
            if (registered.sparse) {
                for (const auto& [gpu_address, size] : sparse_regions) {
                    replay.calls.push_back([fn = registered.sparse, gpu_address, size] {
                        fn(gpu_address, size, true);
                    });
                }
            }
            Seal(replay);
        }
        Deliver(std::move(replay));
    }

    void RemoveListener(const void* owner) {
        {
            std::scoped_lock lk{mutex};
            listeners.erase(owner);
        }
        // The notifications run outside `mutex`: wait for any copied listener call of this owner
        // (or another) still in flight or waiting for its turn before returning, so the owner's
        // destructor, which calls this, cannot destroy the object a callback captures.
        std::unique_lock lk{notify_mutex};
        notify_cv.wait(lk, [this] { return active_notifications == 0; });
    }

    // A new emulation session starts: forget the previous game's mappings and device ranges (its
    // address spaces removed their listeners when they closed).
    void Reset() {
        {
            std::scoped_lock lk{device_mutex};
            device_ranges.clear();
            shared_ranges.clear();
        }
        std::scoped_lock lk{mutex};
        mappings.clear();
        sparse_regions.clear();
    }

    // Device (SMMU) range of a mapping outside the heap, shared by every address space of one
    // process: the caches key memory by device address, so a range per address space made separate
    // copies of the same memory, and presentation could read one the game never drew into. The
    // first address space to map the range allocates it with `allocate` (0: out of space); every
    // call takes a reference.
    u64 AcquireDeviceRange(size_t asid, u64 cpu_address, u64 size,
                           const std::function<u64()>& allocate) {
        std::scoped_lock lk{device_mutex};
        const SharedRangeKey key{asid, cpu_address, size};
        if (const auto it = shared_ranges.find(key); it != shared_ranges.end()) {
            ++it->second.references;
            return it->second.device_address;
        }
        const u64 device_address = allocate();
        if (device_address != 0) {
            shared_ranges.emplace(key, SharedRange{device_address, 1});
        }
        return device_address;
    }

    // Drops a reference. True for the last one: the caller frees the range.
    bool ReleaseDeviceRange(size_t asid, u64 cpu_address, u64 size) {
        std::scoped_lock lk{device_mutex};
        const auto it = shared_ranges.find(SharedRangeKey{asid, cpu_address, size});
        if (it == shared_ranges.end() || --it->second.references != 0) {
            return false;
        }
        shared_ranges.erase(it);
        return true;
    }

    // The process's session closed and its ASID can be reused: stop sharing its ranges.
    void ForgetDeviceRanges(size_t asid) {
        std::scoped_lock lk{device_mutex};
        std::erase_if(shared_ranges, [asid](const auto& entry) { return entry.first.asid == asid; });
    }

    // Device (SMMU) address of a CPU range an address space mapped; used to present images.
    void RecordDeviceRange(u64 cpu_address, u64 size, u64 device_address) {
        std::scoped_lock lk{device_mutex};
        device_ranges[cpu_address] = {size, device_address};
    }

    std::optional<u64> DeviceAddressOf(u64 cpu_address) {
        std::scoped_lock lk{device_mutex};
        auto it = device_ranges.upper_bound(cpu_address);
        if (it == device_ranges.begin()) {
            return std::nullopt;
        }
        --it;
        const auto& [start, range] = *it;
        if (cpu_address - start >= range.first) {
            return std::nullopt;
        }
        return range.second + (cpu_address - start);
    }

    std::optional<u64> GpuAddressOf(u32 handle, u64 offset) {
        std::scoped_lock lk{mutex};
        for (auto it = mappings.rbegin(); it != mappings.rend(); ++it) {
            if (handle != 0 && it->handle == handle && offset >= it->offset &&
                offset - it->offset < it->size) {
                return it->gpu_address + (offset - it->offset);
            }
        }
        return std::nullopt;
    }

    std::optional<u64> DeviceAddressOfGpu(u64 gpu_address) {
        std::optional<u64> cpu_address;
        {
            std::scoped_lock lk{mutex};
            for (const auto& mapping : mappings) {
                if (gpu_address >= mapping.gpu_address &&
                    gpu_address - mapping.gpu_address < mapping.size) {
                    cpu_address = mapping.cpu_address + (gpu_address - mapping.gpu_address);
                    break;
                }
            }
        }
        return cpu_address ? DeviceAddressOf(*cpu_address) : std::nullopt;
    }

private:
    // Separate lock: ranges are recorded from listeners that run under `mutex`.
    std::mutex device_mutex;
    std::map<u64, std::pair<u64, u64>> device_ranges; // cpu start -> (size, device address)

    struct SharedRangeKey {
        size_t asid;
        u64 cpu_address;
        u64 size;
        auto operator<=>(const SharedRangeKey&) const = default;
    };
    struct SharedRange {
        u64 device_address;
        u32 references;
    };
    std::map<SharedRangeKey, SharedRange> shared_ranges;

    struct Mapping {
        u64 gpu_address;
        u64 cpu_address;
        u64 size;
        u32 handle;
        u64 offset;
    };

    struct Listeners {
        Listener map;
        UnmapListener unmap;
        ReleaseListener release;
        SparseListener sparse;
        BackListener back;
        TileListener tile;
    };

    // Drops [gpu_address, +size) from the recorded mappings, keeping the parts outside it.
    void TrimMappings(u64 gpu_address, u64 size) {
        const u64 end = gpu_address + size;
        std::vector<Mapping> kept;
        kept.reserve(mappings.size() + 1);
        for (const auto& m : mappings) {
            const u64 m_end = m.gpu_address + m.size;
            if (m_end <= gpu_address || m.gpu_address >= end) {
                kept.push_back(m);
                continue;
            }
            if (m.gpu_address < gpu_address) {
                kept.push_back({m.gpu_address, m.cpu_address, gpu_address - m.gpu_address,
                                m.handle, m.offset});
            }
            if (m_end > end) {
                kept.push_back({end, m.cpu_address + (end - m.gpu_address), m_end - end,
                                m.handle, m.offset + (end - m.gpu_address)});
            }
        }
        mappings = std::move(kept);
    }

    std::mutex mutex;
    std::vector<Mapping> mappings;
    std::map<u64, u64> sparse_regions; ///< GPU address -> size
    std::map<const void*, Listeners> listeners;

    /// Listener calls copied under `mutex` for one change, made after it is released.
    struct Notification {
        Nx2GpuMemoryRegistry* registry{}; ///< Null: Defer work, ordered only within its thread
        u64 ticket{};
        std::vector<std::function<void()>> calls;
    };

    /// Listener calls a thread holds back (DeferScope), or makes while it is already making
    /// some: a listener that changes the registry again is called after the current call
    /// instead of waiting for a turn its own call holds.
    struct ThreadState {
        int depth = 0;
        std::deque<Notification> pending;
    };
    static ThreadState& ThisThread() {
        thread_local ThreadState state;
        return state;
    }

    // Needs `mutex`. Copies the call of every registered listener that has `member` (a null
    // member is skipped, as before), in owner order.
    template <typename Member, typename... Args>
    void Collect(Notification& notification, Member member, Args... args) {
        for (auto& [owner, listener] : listeners) {
            if (auto& fn = listener.*member) {
                notification.calls.push_back([fn, args...]() { fn(args...); });
            }
        }
    }

    // Needs `mutex`. Counts the calls as in flight: raised while `mutex` is held, so
    // RemoveListener (which erases its owner under the same mutex and then waits for the count)
    // can never observe zero between the copy and this increment.
    void Count(Notification& notification) {
        notification.registry = this;
        std::scoped_lock notify_lk{notify_mutex};
        active_notifications += notification.calls.size();
    }

    // Needs `mutex`. Counts the calls and orders them after every change made before this one.
    void Seal(Notification& notification) {
        if (notification.calls.empty()) {
            return; // nobody to tell, nothing to order
        }
        notification.ticket = next_ticket++;
        Count(notification);
    }

    // Outside `mutex`: makes the calls now, in turn, or once this thread's DeferScope ends.
    static void Deliver(Notification&& notification) {
        if (notification.calls.empty()) {
            return;
        }
        auto& local = ThisThread();
        local.pending.push_back(std::move(notification));
        if (local.depth == 0) {
            FlushThisThread();
        }
    }

    static void FlushThisThread() {
        auto& local = ThisThread();
        ++local.depth; // listener calls made from these calls queue behind them
        while (!local.pending.empty()) {
            Notification notification = std::move(local.pending.front());
            local.pending.pop_front();
            if (notification.registry != nullptr) {
                notification.registry->Run(notification, true);
            } else {
                for (auto& call : notification.calls) {
                    call();
                }
            }
        }
        --local.depth;
    }

    // Makes the calls, after every earlier ticket's when `ordered`, then gives up the turn and
    // the in-flight count even if a call throws.
    void Run(Notification& notification, bool ordered) {
        if (ordered) {
            std::unique_lock lk{notify_mutex};
            notify_cv.wait(lk, [&] { return delivered_ticket == notification.ticket; });
        }
        struct Done {
            Nx2GpuMemoryRegistry& self;
            size_t count;
            bool advance;
            ~Done() {
                {
                    std::scoped_lock lk{self.notify_mutex};
                    self.active_notifications -= count;
                    if (advance) {
                        ++self.delivered_ticket;
                    }
                }
                self.notify_cv.notify_all();
            }
        } done{*this, notification.calls.size(), ordered};
        for (auto& call : notification.calls) {
            call();
        }
    }

    // For changes with no registry state of their own.
    template <typename Member, typename... Args>
    void NotifyListeners(Member member, Args... args) {
        Notification notification;
        {
            std::scoped_lock lk{mutex};
            Collect(notification, member, args...);
            Seal(notification);
        }
        Deliver(std::move(notification));
    }

    u64 next_ticket = 0; ///< Under `mutex`

    std::mutex notify_mutex;
    std::condition_variable notify_cv;
    /// Copied listener calls in flight or waiting for their turn; see RemoveListener.
    size_t active_notifications = 0;
    u64 delivered_ticket = 0; ///< Tickets before it are delivered
};

// Device node that Switch 2 nvgpu drivers open but Switch 1 never had (e.g. /dev/nvhost-tsg-gpu).
// Every ioctl succeeds with a zeroed reply and is logged, so the driver's use of it shows up.
class nx2_stub_device final : public nvdevice {
public:
    explicit nx2_stub_device(Core::System& system_, std::string name_)
        : nvdevice{system_}, name{std::move(name_)} {}

    NvResult Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                    std::span<u8> output) override {
        return Stub(fd, command, input, output, {});
    }
    NvResult Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                    std::span<const u8> inline_input, std::span<u8> output) override {
        return Stub(fd, command, input, output, {});
    }
    NvResult Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output,
                    std::span<u8> inline_output) override {
        return Stub(fd, command, input, output, inline_output);
    }

    void OnOpen(NvCore::SessionId session_id, DeviceFD fd) override {
        LOG_WARNING(Service_NVDRV, "(STUBBED) {} opened as fd {}", name, fd);
    }
    void OnClose(DeviceFD fd) override {}

private:
    NvResult Stub(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output,
                  std::span<u8> inline_output) {
        LOG_WARNING(Service_NVDRV, "(STUBBED) {} fd={} ioctl={:08X} group={:01X} command={:02X} in={:#x}",
                    name, fd, command.raw, command.group, command.cmd, input.size());
        // Echo the input (in/out ioctls keep their fields), zero the rest.
        std::fill(output.begin(), output.end(), u8{0});
        std::copy_n(input.begin(), std::min(input.size(), output.size()), output.begin());
        std::fill(inline_output.begin(), inline_output.end(), u8{0});
        return NvResult::Success;
    }

    std::string name;
};

} // namespace Service::Nvidia::Devices
