// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace Common {

/// Counts the steps of an emulation teardown that can take long (pipeline caches written,
/// thousands of pipelines destroyed), so a frontend waiting for its emulation thread can tell a
/// slow teardown that still moves from one that hangs. Terminating the thread mid-teardown left a
/// lock it held (the heap's, the driver's) taken for good, and the whole app froze at 0% CPU.
inline std::atomic<u64> shutdown_progress{0};

inline void NoteShutdownProgress() noexcept {
    shutdown_progress.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline u64 ShutdownProgress() noexcept {
    return shutdown_progress.load(std::memory_order_relaxed);
}

} // namespace Common
