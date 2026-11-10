// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// The constant buffer slots each program of an evicted graphics pipeline bound as storage buffer
/// pointers, by pipeline key hash. Translating the pipeline again classifies a slot by what it
/// holds at that moment, and a slot that is null or unmapped then (around an area load) loses its
/// buffer: the new translation reads zeros through it for the rest of the session. Kept for the
/// next translation, the pipeline can't bind fewer pointers than the one it replaces.
class EvictedPointerSlots {
public:
    static constexpr size_t NumPrograms = 6;
    using Slots = std::array<std::vector<u64>, NumPrograms>;

    /// Keeps the slots of a pipeline that bound any. When full, an arbitrary older entry goes.
    void Remember(u64 key_hash, const Slots& slots) {
        bool any = false;
        for (const auto& keys : slots) {
            any |= !keys.empty();
        }
        if (!any) {
            return;
        }
        if (!entries.contains(key_hash) && entries.size() >= MaxEntries) {
            entries.erase(entries.begin());
        }
        entries.insert_or_assign(key_hash, slots);
    }

    /// The slots kept for this pipeline, forgotten once taken
    [[nodiscard]] std::optional<Slots> Take(u64 key_hash) {
        auto node = entries.extract(key_hash);
        if (node.empty()) {
            return std::nullopt;
        }
        return std::move(node.mapped());
    }

    [[nodiscard]] size_t Size() const noexcept {
        return entries.size();
    }

    static constexpr size_t MaxEntries = 8192;

private:
    std::unordered_map<u64, Slots> entries;
};

} // namespace VideoCommon
