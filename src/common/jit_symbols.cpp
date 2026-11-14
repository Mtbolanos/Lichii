// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>

#include "common/jit_symbols.h"

namespace Common::JitSymbols {
namespace {

struct Range {
    std::uintptr_t end;
    std::string name;
};

std::mutex mutex;
std::map<std::uintptr_t, Range> ranges;

} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_JIT_SYMBOLS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void Register(const void* start, const void* end, std::string_view name) {
    if (!Enabled() || start == end) {
        return;
    }
    const auto first = reinterpret_cast<std::uintptr_t>(start);
    std::scoped_lock lock{mutex};
    ranges.insert_or_assign(first, Range{reinterpret_cast<std::uintptr_t>(end), std::string{name}});
}

void Clear(const void* begin, const void* end) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    ranges.erase(ranges.lower_bound(reinterpret_cast<std::uintptr_t>(begin)),
                 ranges.lower_bound(reinterpret_cast<std::uintptr_t>(end)));
}

std::optional<std::string> Lookup(const void* address) {
    if (!Enabled()) {
        return std::nullopt;
    }
    const auto value = reinterpret_cast<std::uintptr_t>(address);
    std::scoped_lock lock{mutex};
    auto it = ranges.upper_bound(value);
    if (it == ranges.begin()) {
        return std::nullopt;
    }
    --it;
    if (value >= it->second.end) {
        return std::nullopt;
    }
    return it->second.name;
}

} // namespace Common::JitSymbols
