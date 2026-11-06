// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon {

struct VirtualSegment {
    GPUVAddr gpu_addr;
    DAddr device_addr;
    u32 size;
};

using VirtualSegments = boost::container::small_vector<VirtualSegment, 8>;

class VirtualRangeCache {
public:
    static constexpr size_t MAX_ENTRIES = 8192;
    static constexpr size_t MAX_DEFERRED = 4096;

    const VirtualSegments* Query(Tegra::MemoryManager& memory, GPUVAddr gpu_addr, u32 size) {
        if (has_deferred.load(std::memory_order_acquire)) {
            ApplyDeferred();
        }
        if (entries.size() > MAX_ENTRIES) {
            entries.clear();
        }
        const size_t as_id = memory.GetID();
        const u64 key = MakeKey(as_id, gpu_addr);
        Nx2Profile::Add(Nx2Profile::Counter::RangeQueries, 1);
        const auto it = entries.find(key);
        if (it != entries.end() && it->second.as_id == as_id && it->second.gpu_addr == gpu_addr) {
            if (it->second.size == size) {
                return &it->second.segments;
            }
            // Bindings of one address come with different sizes (a storage buffer bound to the
            // end of its mapping, then a smaller view of it): a contiguous answer for a larger
            // size holds for this one cut to `size`, without walking the page table again.
            if (it->second.size > size && !it->second.segments.empty()) {
                trimmed.clear();
                u64 remaining = size;
                for (const VirtualSegment& segment : it->second.segments) {
                    const u32 piece = static_cast<u32>((std::min<u64>)(segment.size, remaining));
                    trimmed.push_back(VirtualSegment{
                        .gpu_addr = segment.gpu_addr,
                        .device_addr = segment.device_addr,
                        .size = piece,
                    });
                    remaining -= piece;
                    if (remaining == 0) {
                        break;
                    }
                }
                return &trimmed;
            }
            if (it->second.size < size) {
                Nx2Profile::Add(Nx2Profile::Counter::RangeSizeMiss, 1);
            }
        }
        Nx2Profile::Add(Nx2Profile::Counter::RangeWalks, 1);
        Entry entry{};
        entry.as_id = as_id;
        entry.gpu_addr = gpu_addr;
        entry.size = size;
        const auto ranges = memory.GetSubmappedRange(gpu_addr, size);
        GPUVAddr expected = gpu_addr;
        bool contiguous = true;
        for (const auto& [range_addr, range_size] : ranges) {
            if (range_addr != expected || range_size == 0) {
                contiguous = false;
                break;
            }
            const std::optional<DAddr> device_addr = memory.GpuToCpuAddress(range_addr);
            if (!device_addr || *device_addr == 0) {
                contiguous = false;
                break;
            }
            if (range_size > static_cast<size_t>((std::numeric_limits<u32>::max)())) {
                contiguous = false;
                break;
            }
            entry.segments.push_back(VirtualSegment{
                .gpu_addr = range_addr,
                .device_addr = *device_addr,
                .size = static_cast<u32>(range_size),
            });
            expected += range_size;
        }
        if (!contiguous || expected != gpu_addr + size) {
            entry.segments.clear();
        }
        // Keep the answer for the larger size cached: smaller sizes of a split mapping would
        // otherwise replace it and the two sizes would walk the page table in turn.
        if (it != entries.end() && it->second.as_id == as_id && it->second.gpu_addr == gpu_addr &&
            it->second.size > size) {
            trimmed = std::move(entry.segments);
            return &trimmed;
        }
        const auto result = entries.insert_or_assign(key, std::move(entry));
        return &result.first->second.segments;
    }

    void Unmap(size_t as_id, GPUVAddr gpu_addr, u64 size) {
        if (size == 0) {
            return;
        }
        {
            std::scoped_lock lock{deferred_mutex};
            if (!deferred.empty()) {
                DeferredUnmap& last = deferred.back();
                if (last.as_id == as_id && last.gpu_addr + last.size == gpu_addr) {
                    last.size += size;
                    has_deferred.store(true, std::memory_order_release);
                    return;
                }
            }
            if (deferred.size() >= MAX_DEFERRED) {
                deferred.clear();
                deferred_overflow = true;
            } else {
                deferred.push_back(DeferredUnmap{
                    .as_id = as_id,
                    .gpu_addr = gpu_addr,
                    .size = size,
                });
            }
        }
        has_deferred.store(true, std::memory_order_release);
    }

private:
    struct Entry {
        VirtualSegments segments;
        size_t as_id{};
        GPUVAddr gpu_addr{};
        u32 size{};
    };

    struct DeferredUnmap {
        size_t as_id;
        GPUVAddr gpu_addr;
        u64 size;
    };

    static u64 MakeKey(size_t as_id, GPUVAddr gpu_addr) {
        return (static_cast<u64>(as_id) << 48) ^ gpu_addr;
    }

    void ApplyDeferred() {
        std::vector<DeferredUnmap> pending;
        bool overflow = false;
        {
            std::scoped_lock lock{deferred_mutex};
            has_deferred.store(false, std::memory_order_release);
            pending.swap(deferred);
            overflow = deferred_overflow;
            deferred_overflow = false;
        }
        if (overflow) {
            entries.clear();
            return;
        }
        if (pending.empty() || entries.empty()) {
            return;
        }
        for (auto it = entries.begin(); it != entries.end();) {
            const Entry& entry = it->second;
            const GPUVAddr entry_end = entry.gpu_addr + entry.size;
            bool overlaps = false;
            for (const DeferredUnmap& unmap : pending) {
                if (unmap.as_id != entry.as_id) {
                    continue;
                }
                if (entry.gpu_addr < unmap.gpu_addr + unmap.size && unmap.gpu_addr < entry_end) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }

    ::Common::unordered_map<u64, Entry> entries;
    /// Query's answer for a smaller size than the cached one; valid until the next Query.
    VirtualSegments trimmed;
    std::vector<DeferredUnmap> deferred;
    std::mutex deferred_mutex;
    std::atomic<bool> has_deferred{false};
    bool deferred_overflow{};
};

} // namespace VideoCommon
