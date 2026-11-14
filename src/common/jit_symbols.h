// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>

// Host code ranges of JIT-compiled guest blocks, so the Windows sampler (nx2_sampler) can name
// the guest code a guest CPU thread runs instead of "(jit)". Dynarmic's perf map feeds it on
// Windows (on Linux the same names go to perf's /tmp/perf-<pid>.map). It records only with
// EDEN_NX2_JIT_SYMBOLS=1: naming takes a process-wide lock per compiled block.
namespace Common::JitSymbols {

bool Enabled();

void Register(const void* start, const void* end, std::string_view name);

/// Forgets the ranges that start in [begin, end): that JIT cleared its code cache. Each core has
/// its own JIT and code buffer, so the other cores' blocks keep their names.
void Clear(const void* begin, const void* end);

/// The name of the block holding the host address, if one is registered.
std::optional<std::string> Lookup(const void* address);

} // namespace Common::JitSymbols
