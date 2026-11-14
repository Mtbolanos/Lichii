// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <deque>
#include <optional>
#include <unordered_map>
#include <utility>

#include "common/common_types.h"

namespace VideoCommon {

/// Values by key up to a total size, the oldest added dropped first past it
template <typename Key, typename Value>
class BoundedStore {
public:
    explicit BoundedStore(u64 max_bytes_) : max_bytes{max_bytes_} {}

    /// Keeps `value`, which holds `bytes`, in place of any value of its key
    void Add(const Key& key, Value value, u64 bytes) {
        Erase(key);
        if (bytes > max_bytes) {
            return;
        }
        while (total_bytes + bytes > max_bytes) {
            auto [old_key, serial] = std::move(order.front());
            order.pop_front();
            const auto it = entries.find(old_key);
            if (it != entries.end() && it->second.serial == serial) {
                total_bytes -= it->second.bytes;
                entries.erase(it);
            }
        }
        // Taken and replaced values leave their age records behind; at twice the live ones they go.
        if (order.size() >= 2 * entries.size() + 64) {
            std::erase_if(order, [this](const std::pair<Key, u64>& record) {
                const auto it = entries.find(record.first);
                return it == entries.end() || it->second.serial != record.second;
            });
        }
        total_bytes += bytes;
        order.emplace_back(key, ++last_serial);
        entries.insert_or_assign(key, Entry{std::move(value), bytes, last_serial});
    }

    /// The value of `key`, forgotten once taken
    [[nodiscard]] std::optional<Value> Take(const Key& key) {
        auto node = entries.extract(key);
        if (node.empty()) {
            return std::nullopt;
        }
        total_bytes -= node.mapped().bytes;
        return std::move(node.mapped().value);
    }

    [[nodiscard]] size_t Size() const noexcept {
        return entries.size();
    }

    [[nodiscard]] u64 Bytes() const noexcept {
        return total_bytes;
    }

    /// Age records held, live and stale
    [[nodiscard]] size_t AgeRecords() const noexcept {
        return order.size();
    }

private:
    struct Entry {
        Value value;
        u64 bytes;
        u64 serial;
    };

    void Erase(const Key& key) {
        if (const auto it = entries.find(key); it != entries.end()) {
            total_bytes -= it->second.bytes;
            entries.erase(it);
        }
    }

    u64 max_bytes;
    u64 total_bytes{};
    u64 last_serial{};
    std::unordered_map<Key, Entry> entries;
    std::deque<std::pair<Key, u64>> order; ///< (key, serial) by age, stale ones skipped
};

} // namespace VideoCommon
