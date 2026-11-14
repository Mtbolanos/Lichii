// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace VideoCommon::Nx2Perf {

[[nodiscard]] inline bool VerifySwitchSet() {
    const char* value = std::getenv("EDEN_NX2_PERF_VERIFY");
    return value != nullptr && value[0] == '1';
}

/// EDEN_NX2_PERF_VERIFY=1. A plain flag: the buffer cache, the memory tracker and the 3D engine
/// check it on hot paths, and a function's static is checked for its initialization on every
/// call (MSVC reads a thread-local for it).
inline bool g_verify = VerifySwitchSet();

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_verify = VerifySwitchSet();
}

inline bool Verify() {
    return g_verify;
}

} // namespace VideoCommon::Nx2Perf
