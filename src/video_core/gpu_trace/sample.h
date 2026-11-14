// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>

namespace VideoCommon::GpuTrace {

/// Writes a small synthetic trace that needs no game: one Switch 2 channel whose frames are
/// CPU-written images and 3D engine clears (whole and scissored), each presented.
/// Returns false if the file can't be written.
bool WriteSampleTrace(const std::filesystem::path& path);

} // namespace VideoCommon::GpuTrace
