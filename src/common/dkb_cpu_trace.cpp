// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <mutex>

#include "common/dkb_cpu_trace.h"

namespace Common::DkbCpuTrace {
namespace {

struct Registry {
    std::mutex mutex;
    std::array<const ThreadCpuClock*, RoleCount> clocks{};
    std::array<u64, RoleCount> generations{};
};

Registry& GetRegistry() {
    static Registry registry;
    return registry;
}

} // namespace

Registration::Registration(Role role_) : role{role_} {
    auto& registry = GetRegistry();
    const auto index = static_cast<size_t>(role);
    std::scoped_lock lock{registry.mutex};
    if (registry.clocks[index]) {
        return;
    }
    ++registry.generations[index];
    registry.clocks[index] = &clock;
    registered = true;
}

Registration::~Registration() {
    if (!registered) {
        return;
    }
    auto& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    registry.clocks[static_cast<size_t>(role)] = nullptr;
}

Snapshot ReadSnapshot() {
    auto& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    Snapshot result{.wall_time = std::chrono::steady_clock::now(), .threads = {}};
    for (size_t i = 0; i < RoleCount; ++i) {
        result.threads[i] = {
            .generation = registry.generations[i],
            .active = registry.clocks[i] != nullptr,
            .cpu_ns = registry.clocks[i] ? registry.clocks[i]->Read() : std::nullopt,
        };
    }
    return result;
}

} // namespace Common::DkbCpuTrace
