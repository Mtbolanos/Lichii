// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <cstring>
#include <span>

#include "common/common_types.h"

// EDEN_NX2_BACK_IDLE_TRACE=1: what the GPU thread (the split's front) does while the back has
// nothing to execute. The front logs its state changes into a ring; when the back gets work again
// it splits the time it waited by the state in effect.
namespace VideoCommon::BackIdleTrace {

enum class FrontState : u8 {
    Parse,            ///< Handling a command, parsing or resuming channels
    Drain,            ///< Waiting for the back (GpuSplit::Drain, WaitFor)
    Writers,          ///< Waiting for the back to run the writers of a GPU-written command list
    PendingSemaphore, ///< Channels suspended on semaphore acquires
    PendingCopy,      ///< ...one of them on the host GPU's copy of a command list
    QueueEmpty,       ///< Waiting for the guest to submit
    Unknown,          ///< Older than the oldest state change in the ring
    Count,
};

using StateNs = std::array<u64, static_cast<size_t>(FrontState::Count)>;

/// A state change: (nanoseconds << 3) | state.
constexpr u64 Pack(u64 ns, FrontState state) {
    return (ns << 3) | static_cast<u64>(state);
}

/// Adds [begin, end) to `out` by the state in effect, each change holding until the next.
/// `changes` are oldest first; time before the first is Unknown.
void SplitIdle(std::span<const u64> changes, u64 begin, u64 end, StateNs& out);

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BACK_IDLE_TRACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

u64 Now();

/// GPU thread: sets the front's state and returns the one before (Count on other threads).
FrontState SwapFrontState(FrontState state);

inline void SetFrontState(FrontState state) {
    if (Enabled()) {
        SwapFrontState(state);
    }
}

/// The front is in `state` while this lives.
class FrontWait {
public:
    explicit FrontWait(FrontState state)
        : previous{Enabled() ? SwapFrontState(state) : FrontState::Count} {}
    ~FrontWait() {
        if (previous != FrontState::Count) {
            SwapFrontState(previous);
        }
    }
    FrontWait(const FrontWait&) = delete;
    FrontWait& operator=(const FrontWait&) = delete;

private:
    FrontState previous;
};

/// Back thread: it had nothing to execute from `begin` to `end` (Now). Logs every 10 s.
void NoteBackIdle(u64 begin, u64 end);

} // namespace VideoCommon::BackIdleTrace
