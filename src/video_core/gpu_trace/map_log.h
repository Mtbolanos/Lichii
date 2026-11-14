// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <iterator>
#include <map>
#include <mutex>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::GpuTrace {

/// What a GPU address space maps, by range, kept so a recording can start at any time: the page
/// tables answer per address and can't list their mappings.
class MapLog {
public:
    struct Entry {
        u64 gpu_address;
        u64 end;
        u64 device_address;
        u32 kind;
        bool big_pages;
        bool sparse;
    };

    void Map(const Entry& entry) {
        std::scoped_lock lock{mutex};
        EraseLocked(entry.gpu_address, entry.end);
        maps.emplace(entry.gpu_address, entry);
    }

    void Unmap(u64 begin, u64 end) {
        std::scoped_lock lock{mutex};
        EraseLocked(begin, end);
    }

    [[nodiscard]] std::vector<Entry> Entries() const {
        std::scoped_lock lock{mutex};
        std::vector<Entry> entries;
        entries.reserve(maps.size());
        for (const auto& [begin, entry] : maps) {
            entries.push_back(entry);
        }
        return entries;
    }

private:
    void EraseLocked(u64 begin, u64 end) {
        auto it = maps.lower_bound(begin);
        if (it != maps.begin() && std::prev(it)->second.end > begin) {
            --it;
        }
        while (it != maps.end() && it->first < end) {
            const Entry entry = it->second;
            it = maps.erase(it);
            if (entry.gpu_address < begin) {
                Entry left = entry;
                left.end = begin;
                maps.emplace(left.gpu_address, left);
            }
            if (entry.end > end) {
                Entry right = entry;
                right.gpu_address = end;
                if (!entry.sparse) {
                    right.device_address += end - entry.gpu_address;
                }
                it = maps.emplace(right.gpu_address, right).first;
                ++it;
            }
        }
    }

    mutable std::mutex mutex;
    std::map<u64, Entry> maps;
};

} // namespace VideoCommon::GpuTrace
