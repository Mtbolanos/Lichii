// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>

namespace Common::Windows {

/// On an unhandled exception, abort() or a CRT invalid-parameter failure, logs one Critical "Host
/// crash" line (code, faulting address and stack as module+offset, thread), flushes the log and
/// writes <dump_directory>/eden_crash.dmp. Call after Common::Log::Initialize.
void InstallCrashHandler(const std::filesystem::path& dump_directory);

} // namespace Common::Windows
