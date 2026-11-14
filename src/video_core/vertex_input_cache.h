// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// The registers RasterizerVulkan::UpdateVertexInput builds a dynamic vertex input state from:
/// every attribute's format word, every binding's stride, divisor and instancing, the bindings the
/// buffer cache repacks, and how many of each the device takes. The build reads nothing else, so
/// equal keys build equal states. From jpkto/Lichii e7581fa9f6.
struct VertexInputKey {
    static constexpr size_t MaxEntries = 32;

    std::array<u32, MaxEntries> attributes{};
    std::array<u32, MaxEntries> strides{};
    std::array<u32, MaxEntries> divisors{};
    u32 instanced{};
    u32 repacked{};
    u32 num_attributes{};
    u32 num_bindings{};

    [[nodiscard]] bool operator==(const VertexInputKey&) const = default;

    [[nodiscard]] u64 Hash() const noexcept {
        u64 hash = 0x9E3779B97F4A7C15ULL ^ ((u64{num_attributes} << 32) | num_bindings);
        const auto mix = [&hash](u64 value) {
            hash = (hash ^ value) * 0xFF51AFD7ED558CCDULL;
            hash ^= hash >> 32;
        };
        for (size_t index = 0; index < MaxEntries; index += 2) {
            mix(u64{attributes[index]} | (u64{attributes[index + 1]} << 32));
            mix(u64{strides[index]} | (u64{strides[index + 1]} << 32));
            mix(u64{divisors[index]} | (u64{divisors[index + 1]} << 32));
        }
        mix(u64{instanced} | (u64{repacked} << 32));
        return hash;
    }
};

/// Built vertex input states by their keys, direct-mapped. A state is immutable once built and
/// shared with the recorded commands that set it, so replacing an entry never frees a state a
/// pending command still reads.
template <typename State>
class VertexInputCache {
public:
    static constexpr size_t NumEntries = 256;

    [[nodiscard]] std::shared_ptr<const State> Find(const VertexInputKey& key, u64 hash) const {
        if (entries.empty()) {
            return nullptr;
        }
        const Entry& entry = entries[hash % NumEntries];
        if (entry.state == nullptr || entry.hash != hash || !(entry.key == key)) {
            return nullptr;
        }
        return entry.state;
    }

    void Insert(const VertexInputKey& key, u64 hash, std::shared_ptr<const State> state) {
        if (entries.empty()) {
            entries.resize(NumEntries);
        }
        Entry& entry = entries[hash % NumEntries];
        entry.hash = hash;
        entry.key = key;
        entry.state = std::move(state);
    }

private:
    struct Entry {
        u64 hash{};
        VertexInputKey key{};
        std::shared_ptr<const State> state;
    };
    std::vector<Entry> entries;
};

} // namespace VideoCommon
