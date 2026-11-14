// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <source_location>
#include <string_view>
#include <thread>
#include <type_traits>
#include <algorithm>
#include <vector>

#include <boost/container/deque.hpp>

#include "common/address_space.h"
#include "common/address_space.inc"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/memory_detect.h"
#include "common/paged_scalar.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/device_memory.h"
#include "core/device_memory_manager.h"
#include "core/memory.h"
#include "video_core/nx2_perf.h"

namespace Core {

namespace {

class MultiAddressContainer {
public:
    MultiAddressContainer() = default;
    ~MultiAddressContainer() = default;

    void GatherValues(u32 start_entry, Common::ScratchBuffer<u32>& buffer) {
        buffer.resize(8);
        const auto add_value = [&buffer](u32 value, size_t index) {
            if (buffer.size() < index + 1)
                buffer.resize(index + 8);
            buffer[index++] = value;
            return index;
        };

        size_t index = 0;
        u32 iter_entry = start_entry;
        Entry* current = &storage[iter_entry - 1];
        index = add_value(current->value, index);
        while (current->next_entry != 0) {
            iter_entry = current->next_entry;
            current = &storage[iter_entry - 1];
            index = add_value(current->value, index);
        }
        buffer.resize(index);
    }

    u32 Register(u32 value) {
        return RegisterImplementation(value);
    }

    void Register(u32 value, u32 start_entry) {
        auto entry_id = RegisterImplementation(value);
        u32 iter_entry = start_entry;
        Entry* current = &storage[iter_entry - 1];
        while (current->next_entry != 0) {
            iter_entry = current->next_entry;
            current = &storage[iter_entry - 1];
        }
        current->next_entry = entry_id;
    }

    std::pair<bool, u32> Unregister(u32 value, u32 start_entry) {
        u32 iter_entry = start_entry;
        Entry* previous{};
        Entry* current = &storage[iter_entry - 1];
        Entry* next{};
        bool more_than_one_remaining = false;
        u32 result_start{start_entry};
        size_t count = 0;
        while (current->value != value) {
            count++;
            previous = current;
            iter_entry = current->next_entry;
            current = &storage[iter_entry - 1];
        }
        // Find next
        u32 next_entry = current->next_entry;
        if (next_entry != 0) {
            next = &storage[next_entry - 1];
            more_than_one_remaining = next->next_entry != 0 || previous != nullptr;
        }
        if (previous) {
            previous->next_entry = next_entry;
        } else {
            result_start = next_entry;
        }
        free_entries.emplace_back(iter_entry);
        return std::make_pair(more_than_one_remaining || count > 1, result_start);
    }

    u32 ReleaseEntry(u32 start_entry) {
        Entry* current = &storage[start_entry - 1];
        free_entries.emplace_back(start_entry);
        return current->value;
    }

private:
    u32 RegisterImplementation(u32 value) {
        auto entry_id = GetNewEntry();
        auto& entry = storage[entry_id - 1];
        entry.next_entry = 0;
        entry.value = value;
        return entry_id;
    }
    u32 GetNewEntry() {
        if (!free_entries.empty()) {
            u32 result = free_entries.front();
            free_entries.pop_front();
            return result;
        }
        storage.emplace_back();
        u32 new_entry = static_cast<u32>(storage.size());
        return new_entry;
    }

    struct Entry {
        u32 next_entry{};
        u32 value{};
    };

    std::vector<Entry> storage;
    boost::container::deque<u32> free_entries;
};

struct EmptyAllocator {
    EmptyAllocator([[maybe_unused]] DAddr address) {}
};

} // namespace

// An unmapped read returns zeros. The first 16 are logged with their caller, every one with
// EDEN_NX2_READBLOCK_TRACE=1.
inline void ReportUnmappedRead(DAddr first_unmapped, DAddr address, size_t size,
                               size_t unmapped_bytes, const std::source_location& caller) {
    static const bool trace = [] {
        const char* value = std::getenv("EDEN_NX2_READBLOCK_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    static std::atomic<u32> reported{};
    if (!trace && reported.fetch_add(1, std::memory_order_relaxed) >= 16) {
        return;
    }
    std::string_view file{caller.file_name()};
    if (const size_t slash = file.find_last_of("/\\"); slash != std::string_view::npos && slash > 0) {
        const size_t parent = file.find_last_of("/\\", slash - 1);
        file.remove_prefix(parent == std::string_view::npos ? 0 : parent + 1);
    }
    LOG_ERROR(HW_Memory,
              "Unmapped Device ReadBlock @ {:#016x} (start address = {:#016x}, size = {}, {} "
              "bytes unmapped) from {}:{}",
              first_unmapped, address, size, unmapped_bytes, file, caller.line());
}

template <typename DTraits>
struct DeviceMemoryManagerAllocator {
    static constexpr size_t device_virtual_bits = DTraits::device_virtual_bits;
    static constexpr DAddr first_address = 1ULL << Memory::LICHII_PAGEBITS;
    static constexpr DAddr max_device_area = 1ULL << device_virtual_bits;

    DeviceMemoryManagerAllocator() : main_allocator(first_address) {}

    Common::FlatAllocator<DAddr, 0, device_virtual_bits> main_allocator;
    MultiAddressContainer multi_dev_address;

    /// Returns true when vaddr -> vaddr+size is fully contained in the buffer
    template <bool pin_area>
    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= 0 && addr + size <= max_device_area;
    }

    // The main allocator only hands out fresh space until it reaches the end of the address space,
    // so a range freed and mapped again every frame would climb through it, and caches keyed by
    // device address would see new memory each time. Freed blocks are kept for the next allocation
    // of the same size instead, up to this many bytes.
    static constexpr size_t max_recycled_bytes = 1ULL << 30;

    std::mutex recycle_mutex;
    std::vector<std::pair<DAddr, size_t>> recycled;
    size_t recycled_bytes = 0;

    DAddr Allocate(size_t size) {
        std::scoped_lock lock(recycle_mutex);
        const auto it = std::find_if(recycled.rbegin(), recycled.rend(),
                                     [size](const auto& block) { return block.second == size; });
        if (it != recycled.rend()) {
            const DAddr address = it->first;
            recycled.erase(std::next(it).base());
            recycled_bytes -= size;
            return address;
        }
        if (const DAddr address = main_allocator.Allocate(size); address != 0) {
            return address;
        }
        // Out of space: give the kept blocks back so the allocator can merge them.
        for (const auto& [address, block_size] : recycled) {
            main_allocator.Free(address, block_size);
        }
        recycled.clear();
        recycled_bytes = 0;
        return main_allocator.Allocate(size);
    }

    void AllocateFixed(DAddr b_address, size_t b_size) {
        main_allocator.AllocateFixed(b_address, b_size);
    }

    void Free(DAddr b_address, size_t b_size) {
        std::scoped_lock lock(recycle_mutex);
        recycled.emplace_back(b_address, b_size);
        recycled_bytes += b_size;
        while (recycled_bytes > max_recycled_bytes) {
            const auto [address, size] = recycled.front();
            recycled.erase(recycled.begin());
            recycled_bytes -= size;
            main_allocator.Free(address, size);
        }
    }
};

template <typename Traits>
DeviceMemoryManager<Traits>::DeviceMemoryManager(const DeviceMemory& device_memory_)
    : physical_base{uintptr_t(device_memory_.buffer.BackingBasePointer())}
    , device_memory_ptr{&device_memory_}
    , device_inter{nullptr}
    , compressed_device_addr(1ULL << ((Settings::values.memory_layout_mode.GetValue() == Settings::MemoryLayout::Memory_4Gb ? physical_min_bits : physical_max_bits) - Memory::LICHII_PAGEBITS))
    , tracked_entries(device_as_size >> Memory::LICHII_PAGEBITS)
{
    impl = std::make_unique<DeviceMemoryManagerAllocator<Traits>>();
    cached_pages = std::make_unique<CachedPages>();
}

template <typename Traits>
DeviceMemoryManager<Traits>::~DeviceMemoryManager() = default;

template <typename Traits>
void DeviceMemoryManager<Traits>::HintCold(DAddr address, size_t size) {
    // The hint takes the pages out of the working set at once (VirtualUnlock), whether or not
    // the host needs the RAM, and an image made again over them faults them all back in during
    // its upload (Ocarina of Time: 1% lows). Only when RAM is short.
    if (!Common::IsHostLowOnMemory()) {
        return;
    }
    struct Run {
        u64 physical_offset;
        VAddr virtual_address; ///< 0: no CPU mapping known
        u64 size;
    };
    // Images at GPU addresses without memory get placeholder addresses past the device address
    // space, which has no pages to look up.
    constexpr DAddr address_space_end = DAddr{1} << AS_BITS;
    if (address >= address_space_end) {
        return;
    }
    size = static_cast<size_t>((std::min)(DAddr{size}, address_space_end - address));
    std::vector<Run> runs;
    const size_t page_end = Common::DivCeil(address + size, Memory::LICHII_PAGESIZE);
    for (size_t page = address >> Memory::LICHII_PAGEBITS; page < page_end; ++page) {
        const u32 compressed = tracked_entries[page].compressed_physical_ptr;
        if (compressed == 0) {
            continue;
        }
        const u64 physical = u64{compressed - 1} << Memory::LICHII_PAGEBITS;
        const VAddr vaddr = ExtractCPUBacking(page).second;
        if (!runs.empty()) {
            Run& last = runs.back();
            const bool contiguous_view =
                last.virtual_address == 0 ? vaddr == 0 : last.virtual_address + last.size == vaddr;
            if (last.physical_offset + last.size == physical && contiguous_view) {
                last.size += Memory::LICHII_PAGESIZE;
                continue;
            }
        }
        runs.push_back({physical, vaddr, Memory::LICHII_PAGESIZE});
    }
    if (runs.empty()) {
        return;
    }
    if (!cold_hint_worker) {
        cold_hint_worker = std::make_unique<Common::ThreadWorker>(1, "DeviceMemory:ColdHint");
    }
    cold_hint_worker->QueueWork([this, runs = std::move(runs)] {
        for (const Run& run : runs) {
            device_memory_ptr->buffer.HintCold(
                run.physical_offset, run.size,
                run.virtual_address != 0 ? std::optional<size_t>{run.virtual_address}
                                         : std::nullopt);
        }
    });
}

template <typename Traits>
void DeviceMemoryManager<Traits>::BindInterface(DeviceInterface* device_inter_) {
    device_inter = device_inter_;
}

template <typename Traits>
DAddr DeviceMemoryManager<Traits>::Allocate(size_t size) {
    return impl->Allocate(size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::AllocateFixed(DAddr start, size_t size) {
    return impl->AllocateFixed(start, size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::Free(DAddr start, size_t size) {
    impl->Free(start, size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::Map(DAddr address, VAddr virtual_address, size_t size, Asid asid,
                                      bool track) {
    Core::Memory::Memory* process_memory = registered_processes[asid.id];
    size_t start_page_d = address >> Memory::LICHII_PAGEBITS;
    size_t num_pages = Common::AlignUp(size, Memory::LICHII_PAGESIZE) >> Memory::LICHII_PAGEBITS;
    std::scoped_lock lk(kept_mutex, mapping_guard);

    tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages);
    CutContinuityAt(start_page_d);
    for (size_t i = 0; i < num_pages; i++) {
        const VAddr new_vaddress = virtual_address + i * Memory::LICHII_PAGESIZE;
        auto* ptr = process_memory->GetPointerSilent(Common::ProcessAddress(new_vaddress));
        if (const auto kept = kept_pages.find(start_page_d + i);
            kept != kept_pages.end() && ptr != nullptr) [[unlikely]] {
            // The page's memory gets what was written meanwhile before anyone can see it.
            if (kept->second) {
                std::memcpy(ptr, kept->second->data(), sizeof(KeptPage));
                std::atomic_thread_fence(std::memory_order_release);
            }
            kept_pages.erase(kept);
            kept_count.store(kept_pages.size(), std::memory_order_release);
        }
        // One page per run unless tracked below: a run left from an earlier mapping of these
        // device pages would make WalkBlock copy the host memory that followed that mapping.
        tracked_entries.GetUnchecked(start_page_d + i).continuity_tracker = 0;
        if (ptr == nullptr) [[unlikely]] {
            tracked_entries.GetUnchecked(start_page_d + i).compressed_physical_ptr = 0;
            continue;
        }
        auto phys_addr = static_cast<u32>(GetRawPhysicalAddr(ptr) >> Memory::LICHII_PAGEBITS) + 1U;
        tracked_entries.GetUnchecked(start_page_d + i).compressed_physical_ptr = phys_addr;
        InsertCPUBacking(start_page_d + i, new_vaddress, asid);
        const u32 base_dev = compressed_device_addr[phys_addr - 1U];
        const u32 new_dev = static_cast<u32>(start_page_d + i);
        if (base_dev == 0) [[likely]] {
            compressed_device_addr.GetAndFault(phys_addr - 1U) = new_dev;
            continue;
        }
        u32 start_id = base_dev & MULTI_MASK;
        if ((base_dev >> MULTI_FLAG_BITS) == 0) {
            start_id = impl->multi_dev_address.Register(base_dev);
            compressed_device_addr.GetAndFault(phys_addr - 1U) = MULTI_FLAG | start_id;
        }
        impl->multi_dev_address.Register(new_dev, start_id);
    }
    translation_generation.fetch_add(1, std::memory_order_release);
    if (track) {
        TrackContinuityImpl(address, virtual_address, size, asid);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::Unmap(DAddr address, size_t size) {
    size_t start_page_d = address >> Memory::LICHII_PAGEBITS;
    size_t num_pages = Common::AlignUp(size, Memory::LICHII_PAGESIZE) >> Memory::LICHII_PAGEBITS;
    device_inter->InvalidateRegion(address, size);
    while (deferred_cached_updates.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
    std::scoped_lock lk(kept_mutex, mapping_guard);
    if (!kept_pages.empty()) {
        for (size_t i = 0; i < num_pages; i++) {
            kept_pages.erase(start_page_d + i);
        }
        kept_count.store(kept_pages.size(), std::memory_order_release);
    }

    tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages); // should already be committed, but just in case
    CutContinuityAt(start_page_d);
    for (size_t i = 0; i < num_pages; i++) {
        auto& entry = tracked_entries.GetUnchecked(start_page_d + i);
        auto phys_addr = entry.compressed_physical_ptr;
        entry.compressed_physical_ptr = 0;
        entry.cpu_backing_address = 0;
        entry.continuity_tracker = 0;
        if (phys_addr != 0) [[likely]] {
            u32& base_dev = compressed_device_addr.GetAndFault(phys_addr - 1U);
            if ((base_dev >> MULTI_FLAG_BITS) == 0) [[likely]] {
                base_dev = 0;
                continue;
            }
            const auto [more_entries, new_start] = impl->multi_dev_address.Unregister(
                static_cast<u32>(start_page_d + i), base_dev & MULTI_MASK);
            if (!more_entries) {
                base_dev = impl->multi_dev_address.ReleaseEntry(new_start);
                continue;
            }
            base_dev = new_start | MULTI_FLAG;
        }
    }
    translation_generation.fetch_add(1, std::memory_order_release);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::CutContinuityAt(size_t page) {
    // Runs counted from the pages before `page` end there: the pages from `page` on are being
    // mapped again or unmapped. A run is counted down page by page (TrackContinuityImpl), so the
    // first page whose run already ends before `page` ends the walk.
    for (size_t index = page; index-- > 0;) {
        const size_t last_count = page - 1 - index;
        if (tracked_entries[index].continuity_tracker <= last_count) {
            break;
        }
        tracked_entries.GetUnchecked(index).continuity_tracker = static_cast<u32>(last_count);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::TrackContinuityImpl(DAddr address, VAddr virtual_address,
                                                      size_t size, Asid asid) {
    Core::Memory::Memory* process_memory = registered_processes[asid.id];
    size_t start_page_d = address >> Memory::LICHII_PAGEBITS;
    size_t num_pages = Common::AlignUp(size, Memory::LICHII_PAGESIZE) >> Memory::LICHII_PAGEBITS;
    uintptr_t last_ptr = 0;
    size_t page_count = 1;

    tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages);
    for (size_t i = num_pages; i > 0; i--) {
        size_t index = i - 1;
        const VAddr new_vaddress = virtual_address + index * Memory::LICHII_PAGESIZE;
        const uintptr_t new_ptr = reinterpret_cast<uintptr_t>(
            process_memory->GetPointerSilent(Common::ProcessAddress(new_vaddress)));
        if (new_ptr + page_size == last_ptr) {
            page_count++;
        } else {
            page_count = 1;
        }
        last_ptr = new_ptr;
        tracked_entries.GetUnchecked(start_page_d + index).continuity_tracker = static_cast<u32>(page_count) - 1;
    }
}
template <typename Traits>
u8* DeviceMemoryManager<Traits>::GetSpan(const DAddr src_addr, const std::size_t size) {
    size_t page_index = src_addr >> page_bits;
    size_t subbits = src_addr & page_mask;
    if ((static_cast<size_t>(tracked_entries[page_index].continuity_tracker+1) << page_bits) >= size + subbits) {
        return GetPointer<u8>(src_addr);
    }
    return nullptr;
}

template <typename Traits>
const u8* DeviceMemoryManager<Traits>::GetSpan(const DAddr src_addr, const std::size_t size) const {
    size_t page_index = src_addr >> page_bits;
    size_t subbits = src_addr & page_mask;
    if ((static_cast<size_t>(tracked_entries[page_index].continuity_tracker+1) << page_bits) >= size + subbits) {
        return GetPointer<u8>(src_addr);
    }
    return nullptr;
}

template <typename Traits>
void DeviceMemoryManager<Traits>::InnerGatherDeviceAddresses(Common::ScratchBuffer<u32>& buffer,
                                                             PAddr address) {
    size_t phys_addr = address >> page_bits;
    std::scoped_lock lk(mapping_guard);
    u32 backing = compressed_device_addr[phys_addr];
    if ((backing >> MULTI_FLAG_BITS) != 0) {
        impl->multi_dev_address.GatherValues(backing & MULTI_MASK, buffer);
        return;
    }
    buffer.resize(1);
    buffer[0] = backing;
}

template <typename Traits>
template <typename T>
T* DeviceMemoryManager<Traits>::GetPointer(DAddr address) {
    const size_t index = address >> Memory::LICHII_PAGEBITS;
    const size_t offset = address & Memory::LICHII_PAGEMASK;
    auto phys_addr = tracked_entries[index].compressed_physical_ptr;
    if (phys_addr == 0) [[unlikely]]
        return nullptr;
    return GetPointerFromRaw<T>((PAddr(phys_addr - 1) << Memory::LICHII_PAGEBITS) + offset);
}

template <typename Traits>
template <typename T>
const T* DeviceMemoryManager<Traits>::GetPointer(DAddr address) const {
    const size_t index = address >> Memory::LICHII_PAGEBITS;
    const size_t offset = address & Memory::LICHII_PAGEMASK;
    auto phys_addr = tracked_entries[index].compressed_physical_ptr;
    if (phys_addr == 0)
        return nullptr;
    return GetPointerFromRaw<T>((PAddr(phys_addr - 1) << Memory::LICHII_PAGEBITS) + offset);
}

template <typename Traits>
template <typename T>
void DeviceMemoryManager<Traits>::Write(DAddr address, T value) {
    Common::WritePagedScalar(address, value, [&](DAddr page_address) {
        return page_address < device_as_size ? GetPointer<u8>(page_address) : nullptr;
    });
}

template <typename Traits>
template <typename T>
T DeviceMemoryManager<Traits>::Read(DAddr address) const {
    return Common::ReadPagedScalar<T>(address, [&](DAddr page_address) {
        return page_address < device_as_size ? GetPointer<u8>(page_address) : nullptr;
    });
}

template <typename Traits>
void DeviceMemoryManager<Traits>::KeepWrites(DAddr address, size_t size) {
    const size_t first = address >> Memory::LICHII_PAGEBITS;
    const size_t last = (address + size - 1) >> Memory::LICHII_PAGEBITS;
    std::scoped_lock lk(kept_mutex);
    for (size_t page = first; page <= last; ++page) {
        if (tracked_entries[page].compressed_physical_ptr == 0) {
            kept_pages.try_emplace(page);
        }
    }
    kept_count.store(kept_pages.size(), std::memory_order_release);
}

template <typename Traits>
bool DeviceMemoryManager<Traits>::ReadKept(DAddr address, void* dest_pointer, size_t size) {
    if (kept_count.load(std::memory_order_acquire) == 0) [[likely]] {
        return false;
    }
    std::scoped_lock lk(kept_mutex);
    bool handled = false;
    auto* dest = static_cast<u8*>(dest_pointer);
    while (size != 0) {
        const size_t page = address >> Memory::LICHII_PAGEBITS;
        const size_t offset = address & Memory::LICHII_PAGEMASK;
        const size_t amount = (std::min<size_t>)(Memory::LICHII_PAGESIZE - offset, size);
        if (const auto phys = tracked_entries[page].compressed_physical_ptr; phys != 0) {
            handled = true; // mapped since the caller looked
            std::memcpy(dest, GetPointerFromRaw<u8>((PAddr(phys - 1) << Memory::LICHII_PAGEBITS) + offset),
                        amount);
        } else if (const auto kept = kept_pages.find(page); kept != kept_pages.end()) {
            handled = true;
            if (kept->second) {
                std::memcpy(dest, kept->second->data() + offset, amount);
            } else {
                std::memset(dest, 0, amount);
            }
        } else {
            std::memset(dest, 0, amount);
        }
        address += amount;
        dest += amount;
        size -= amount;
    }
    return handled;
}

template <typename Traits>
bool DeviceMemoryManager<Traits>::WriteKept(DAddr address, const void* src_pointer, size_t size) {
    if (kept_count.load(std::memory_order_acquire) == 0) [[likely]] {
        return false;
    }
    std::scoped_lock lk(kept_mutex);
    bool handled = false;
    const auto* src = static_cast<const u8*>(src_pointer);
    while (size != 0) {
        const size_t page = address >> Memory::LICHII_PAGEBITS;
        const size_t offset = address & Memory::LICHII_PAGEMASK;
        const size_t amount = (std::min<size_t>)(Memory::LICHII_PAGESIZE - offset, size);
        if (const auto phys = tracked_entries[page].compressed_physical_ptr; phys != 0) {
            handled = true;
            std::memcpy(GetPointerFromRaw<u8>((PAddr(phys - 1) << Memory::LICHII_PAGEBITS) + offset), src,
                        amount);
        } else if (const auto kept = kept_pages.find(page); kept != kept_pages.end()) {
            handled = true;
            if (!kept->second) {
                kept->second = std::make_unique<KeptPage>();
            }
            std::memcpy(kept->second->data() + offset, src, amount);
        }
        address += amount;
        src += amount;
        size -= amount;
    }
    return handled;
}

template <typename Traits>
void DeviceMemoryManager<Traits>::WalkBlock(DAddr addr, std::size_t size, auto on_unmapped, auto on_memory, auto increment) {
    std::size_t remaining_size = size;
    std::size_t page_index = addr >> Memory::LICHII_PAGEBITS;
    std::size_t page_offset = addr & Memory::LICHII_PAGEMASK;
    while (remaining_size) {
        const auto phys_addr = tracked_entries[page_index].compressed_physical_ptr;
        // An unmapped page says nothing about the pages after it: its run cannot be trusted, so
        // only this page is handed to on_unmapped before the walk moves on.
        const size_t next_pages =
            phys_addr == 0 ? 1 : std::size_t(tracked_entries[page_index].continuity_tracker + 1);
        const std::size_t copy_amount = (std::min)((next_pages << Memory::LICHII_PAGEBITS) - page_offset, remaining_size);
        const auto current_vaddr = u64((page_index << Memory::LICHII_PAGEBITS) + page_offset);
        SCOPE_EXIT{
            page_index += next_pages;
            page_offset = 0;
            increment(copy_amount);
            remaining_size -= copy_amount;
        };

        if (phys_addr == 0) {
            on_unmapped(copy_amount, current_vaddr);
            continue;
        }
        auto* mem_ptr = GetPointerFromRaw<u8>((PAddr(phys_addr - 1) << Memory::LICHII_PAGEBITS) + page_offset);
        on_memory(copy_amount, mem_ptr);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReadBlock(DAddr address, void* dest_pointer, size_t size,
                                            std::source_location caller) {
    device_inter->FlushRegion(address, size);
    const std::size_t page_offset = address & Memory::LICHII_PAGEMASK;
    if (size <= Memory::LICHII_PAGESIZE - page_offset) {
        const DAddr guest_page = address & ~static_cast<DAddr>(Memory::LICHII_PAGEMASK);
        TranslationCache& cache = Translations();
        for (const TranslationEntry& entry : cache.slots) {
            if (entry.guest_page == guest_page && entry.host_ptr != nullptr) {
                std::memcpy(dest_pointer, entry.host_ptr + page_offset, size);
                return;
            }
        }

        const std::size_t page_index = address >> Memory::LICHII_PAGEBITS;
        const auto phys_addr = tracked_entries[page_index].compressed_physical_ptr;
        if (phys_addr != 0) {
            auto* const mem_ptr = GetPointerFromRaw<u8>((PAddr(phys_addr - 1) << Memory::LICHII_PAGEBITS));
            cache.slots[cache.cursor] = TranslationEntry{.guest_page = guest_page, .host_ptr = mem_ptr};
            cache.cursor = (cache.cursor + 1) & 3U;
            std::memcpy(dest_pointer, mem_ptr + page_offset, size);
            return;
        }
    }
    DAddr first_unmapped = 0;
    size_t unmapped_bytes = 0;
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            if (ReadKept(current_vaddr, dest_pointer, copy_amount)) {
                return;
            }
            if (unmapped_bytes == 0) {
                first_unmapped = current_vaddr;
            }
            unmapped_bytes += copy_amount;
            std::memset(dest_pointer, 0, copy_amount);
        },
        [&](size_t copy_amount, const u8* const src_ptr) {
            std::memcpy(dest_pointer, src_ptr, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            dest_pointer = static_cast<u8*>(dest_pointer) + copy_amount;
        });
    if (unmapped_bytes != 0) {
        ReportUnmappedRead(first_unmapped, address, size, unmapped_bytes, caller);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::WriteBlock(DAddr address, const void* src_pointer, size_t size) {
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            if (WriteKept(current_vaddr, src_pointer, copy_amount)) {
                return;
            }
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(
                    HW_Memory,
                    "Unmapped Device WriteBlock @ {:#016x} (start address = {:#016x}, size = {})",
                    current_vaddr, address, size);
            }
        },
        [&](size_t copy_amount, u8* const dst_ptr) {
            std::memcpy(dst_ptr, src_pointer, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            src_pointer = static_cast<const u8*>(src_pointer) + copy_amount;
        });
    device_inter->InvalidateRegion(address, size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReadBlockUnsafe(DAddr address, void* dest_pointer, size_t size,
                                                  std::source_location caller) {
    const std::size_t page_offset = address & Memory::LICHII_PAGEMASK;
    if (size <= Memory::LICHII_PAGESIZE - page_offset) {
        const DAddr guest_page = address & ~static_cast<DAddr>(Memory::LICHII_PAGEMASK);
        TranslationCache& cache = Translations();
        for (const TranslationEntry& entry : cache.slots) {
            if (entry.guest_page == guest_page && entry.host_ptr != nullptr) {
                std::memcpy(dest_pointer, entry.host_ptr + page_offset, size);
                return;
            }
        }

        const std::size_t page_index = address >> Memory::LICHII_PAGEBITS;
        const auto phys_addr = tracked_entries[page_index].compressed_physical_ptr;
        if (phys_addr != 0) {
            auto* const mem_ptr = GetPointerFromRaw<u8>((PAddr(phys_addr - 1) << Memory::LICHII_PAGEBITS));
            cache.slots[cache.cursor] = TranslationEntry{.guest_page = guest_page, .host_ptr = mem_ptr};
            cache.cursor = (cache.cursor + 1) & 3U;
            std::memcpy(dest_pointer, mem_ptr + page_offset, size);
            return;
        }
    }
    DAddr first_unmapped = 0;
    size_t unmapped_bytes = 0;
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            if (ReadKept(current_vaddr, dest_pointer, copy_amount)) {
                return;
            }
            if (unmapped_bytes == 0) {
                first_unmapped = current_vaddr;
            }
            unmapped_bytes += copy_amount;
            std::memset(dest_pointer, 0, copy_amount);
        },
        [&](size_t copy_amount, const u8* const src_ptr) {
            std::memcpy(dest_pointer, src_ptr, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            dest_pointer = static_cast<u8*>(dest_pointer) + copy_amount;
        });
    if (unmapped_bytes != 0) {
        ReportUnmappedRead(first_unmapped, address, size, unmapped_bytes, caller);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::WriteBlockUnsafe(DAddr address, const void* src_pointer,
                                                   size_t size) {
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            if (WriteKept(current_vaddr, src_pointer, copy_amount)) {
                return;
            }
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(
                    HW_Memory,
                    "Unmapped Device WriteBlock @ {:#016x} (start address = {:#016x}, size = {})",
                    current_vaddr, address, size);
            }
        },
        [&](size_t copy_amount, u8* const dst_ptr) {
            std::memcpy(dst_ptr, src_pointer, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            src_pointer = static_cast<const u8*>(src_pointer) + copy_amount;
        });
}

template <typename Traits>
Asid DeviceMemoryManager<Traits>::RegisterProcess(Memory::Memory* memory_device_inter) {
    size_t new_id{};
    if (!id_pool.empty()) {
        new_id = id_pool.front();
        id_pool.pop_front();
        registered_processes[new_id] = memory_device_inter;
    } else {
        registered_processes.emplace_back(memory_device_inter);
        new_id = registered_processes.size() - 1U;
    }
    return Asid{new_id};
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UnregisterProcess(Asid asid) {
    registered_processes[asid.id] = nullptr;
    id_pool.push_front(asid.id);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UpdatePagesCachedCountNoLock(DAddr addr, size_t size, s32 delta) {
    u64 uncache_begin = 0;
    u64 cache_begin = 0;
    u64 uncache_bytes = 0;
    u64 cache_bytes = 0;
    const auto MarkRegionCaching = &DeviceMemoryManager<Traits>::DeviceMethods::MarkRegionCaching;

    std::atomic_thread_fence(std::memory_order_acquire);
    const size_t page_end = Common::DivCeil(addr + size, Memory::LICHII_PAGESIZE);
    size_t page = addr >> Memory::LICHII_PAGEBITS;
    auto [asid, base_vaddress] = ExtractCPUBacking(page);
    auto* memory_device_inter =
        asid.id < registered_processes.size() ? registered_processes[asid.id] : nullptr;
    const auto release_pending = [&] {
        if (uncache_bytes > 0) {
            if (memory_device_inter != nullptr) {
                MarkRegionCaching(memory_device_inter, uncache_begin << Memory::LICHII_PAGEBITS,
                                  uncache_bytes, false);
            }
            uncache_bytes = 0;
        }
        if (cache_bytes > 0) {
            if (memory_device_inter != nullptr) {
                MarkRegionCaching(memory_device_inter, cache_begin << Memory::LICHII_PAGEBITS,
                                  cache_bytes, true);
            }
            cache_bytes = 0;
        }
    };
    size_t old_vpage = (base_vaddress >> Memory::LICHII_PAGEBITS) - 1;
    for (; page != page_end; ++page) {
        CounterAtomicType& count = cached_pages->at(page >> subentries_shift).Count(page);
        auto [asid_2, vpage] = ExtractCPUBacking(page);
        vpage >>= Memory::LICHII_PAGEBITS;

        if (vpage == 0) [[unlikely]] {
            release_pending();
            continue;
        }

        if (asid.id != asid_2.id) [[unlikely]] {
            release_pending();
            asid = asid_2;
            memory_device_inter =
                asid_2.id < registered_processes.size() ? registered_processes[asid_2.id] : nullptr;
        }

        if (vpage != old_vpage + 1) [[unlikely]] {
            release_pending();
        }

        old_vpage = vpage;

        // Adds or subtracts 1, as count is a unsigned 8-bit value
        count.fetch_add(static_cast<CounterType>(delta), std::memory_order_release);

        // Assume delta is either -1 or 1
        if (count.load(std::memory_order::relaxed) == 0) {
            if (uncache_bytes == 0) {
                uncache_begin = vpage;
            }
            uncache_bytes += Memory::LICHII_PAGESIZE;
        } else if (uncache_bytes > 0) {
            if (memory_device_inter != nullptr) {
                MarkRegionCaching(memory_device_inter, uncache_begin << Memory::LICHII_PAGEBITS,
                                  uncache_bytes, false);
            }
            uncache_bytes = 0;
        }
        if (count.load(std::memory_order::relaxed) == 1 && delta > 0) {
            if (cache_bytes == 0) {
                cache_begin = vpage;
            }
            cache_bytes += Memory::LICHII_PAGESIZE;
        } else if (cache_bytes > 0) {
            if (memory_device_inter != nullptr) {
                MarkRegionCaching(memory_device_inter, cache_begin << Memory::LICHII_PAGEBITS,
                                  cache_bytes, true);
            }
            cache_bytes = 0;
        }
    }
    release_pending();
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UpdatePagesCachedCount(DAddr addr, size_t size, s32 delta) {
    Common::ScopedRangeLock lk(counter_guard, addr, size);
    UpdatePagesCachedCountNoLock(addr, size, delta);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta) {
    if (ranges.empty()) {
        return;
    }
    // Runs already sorted with gaps between them (the buffer tracker's) are what the sort and
    // merge below would produce: use them as they are instead of two heap copies.
    const bool separated =
        std::ranges::adjacent_find(ranges, [](const auto& a, const auto& b) {
            return a.first > b.first || a.first + a.second >= b.first;
        }) == ranges.end();
    if (separated && !VideoCommon::Nx2Perf::Verify()) {
        const DAddr lock_begin = ranges.front().first;
        const DAddr lock_end = ranges.back().first + ranges.back().second;
        Common::ScopedRangeLock lk(counter_guard, lock_begin,
                                   static_cast<size_t>(lock_end - lock_begin));
        for (const auto& [addr, size] : ranges) {
            UpdatePagesCachedCountNoLock(addr, size, delta);
        }
        return;
    }
    // Make a local copy and sort by address
    std::vector<std::pair<DAddr, size_t>> tmp(ranges.begin(), ranges.end());
    std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Coalesce adjacent/overlapping ranges
    std::vector<std::pair<DAddr, size_t>> coalesced;
    DAddr cur_addr = tmp[0].first;
    size_t cur_size = tmp[0].second;
    for (size_t i = 1; i < tmp.size(); ++i) {
        DAddr next_addr = tmp[i].first;
        size_t next_size = tmp[i].second;
        if (cur_addr + cur_size >= next_addr) {
            // overlapping or contiguous
            const DAddr end = std::max(cur_addr + cur_size, next_addr + next_size);
            cur_size = end - cur_addr;
        } else {
            coalesced.emplace_back(cur_addr, cur_size);
            cur_addr = next_addr;
            cur_size = next_size;
        }
    }
    coalesced.emplace_back(cur_addr, cur_size);
    if (separated) {
        static std::atomic<u32> logged{};
        if (!std::ranges::equal(ranges, coalesced) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: cached page runs from {:#x} merged differently",
                      coalesced.front().first);
        }
    }

    const DAddr lock_begin = coalesced.front().first;
    const DAddr lock_end = coalesced.back().first + coalesced.back().second;
    Common::ScopedRangeLock lk(counter_guard, lock_begin, static_cast<size_t>(lock_end - lock_begin));

    for (const auto& [addr, size] : coalesced) {
        UpdatePagesCachedCountNoLock(addr, size, delta);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ArmReadTraps(DAddr addr, size_t size) {
    const auto MarkRegionReadTrap = &DeviceMemoryManager<Traits>::DeviceMethods::MarkRegionReadTrap;
    Memory::Memory* run_inter = nullptr;
    u64 run_begin = 0;
    u64 run_pages = 0;
    const auto release_run = [&] {
        if (run_pages > 0 && run_inter != nullptr) {
            MarkRegionReadTrap(run_inter, run_begin << Memory::LICHII_PAGEBITS,
                               run_pages << Memory::LICHII_PAGEBITS);
        }
        run_pages = 0;
    };
    const size_t page_end = Common::DivCeil(addr + size, Memory::LICHII_PAGESIZE);
    for (size_t page = addr >> Memory::LICHII_PAGEBITS; page != page_end; ++page) {
        auto [asid, vaddr] = ExtractCPUBacking(page);
        const u64 vpage = vaddr >> Memory::LICHII_PAGEBITS;
        if (vpage == 0) {
            release_run();
            continue;
        }
        auto* const inter = registered_processes[asid.id];
        if (inter != run_inter || vpage != run_begin + run_pages) {
            release_run();
            run_inter = inter;
            run_begin = vpage;
        }
        ++run_pages;
    }
    release_run();
}

} // namespace Core
