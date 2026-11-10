// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <limits>

#include "common/common_types.h"

namespace VideoCommon {

/// Commit the texture cache leaves to the rest of the process and the system.
constexpr u64 HostCommitReserve = u64{2} << 30;

/// The cache is not shrunk below this for host memory: a scene needs about this much, and under
/// it the collector only deletes and decodes the same textures again.
constexpr u64 MinHostCriticalMemory = u64{3} << 30;

/// The most device memory the texture cache may hold before it collects in earnest, as host
/// memory allows it. Windows charges device memory to the system's commit limit (RAM plus the
/// page files) like any other memory, since each allocation has a backing store there. A 16 GB
/// card's budget alone let the cache fill 9.7 GiB on a PC with 16 GiB of RAM and a 16 GiB page
/// file, where the guest's memory already takes 14 GiB of commit: the driver then refused a
/// 2.6 MiB image in every heap, host memory included, and the session ended (Ocarina of Time,
/// RX 7600 XT). `device_usage` is the device memory in use now and `commit_room` the commit the
/// system can still hand out: their sum is what the cache would hold with all of it.
constexpr u64 HostCriticalMemory(u64 device_usage, u64 commit_room) {
    const u64 limit = commit_room > (std::numeric_limits<u64>::max)() - device_usage
                          ? (std::numeric_limits<u64>::max)()
                          : device_usage + commit_room;
    return (std::max)(limit - (std::min)(limit, HostCommitReserve), MinHostCriticalMemory);
}

} // namespace VideoCommon
