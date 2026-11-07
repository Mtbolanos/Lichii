// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <span>
#include <thread>

#include "common/common_types.h"

namespace Vulkan {

/// EDEN_NX2_PIPELINE_BUILD_TRACE=<file>: appends a line before and after every driver pipeline
/// build, flushed at once, so the last BEGIN without an END names the shaders a driver crashed on.
/// END lines carry the build's wall time (ms=), which includes time spent waiting on other builds.
inline void TracePipelineBuild(const char* phase, const char* kind, u64 key_hash,
                               std::span<const u64> stage_hashes, bool unoptimized,
                               double elapsed_ms = -1.0) {
    static std::FILE* const file = [] {
        const char* const path = std::getenv("EDEN_NX2_PIPELINE_BUILD_TRACE");
        return path != nullptr && *path != '\0' ? std::fopen(path, "a") : nullptr;
    }();
    if (file == nullptr) {
        return;
    }
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    std::fprintf(file, "%s %s pipeline=%016llx thread=%zu%s", phase, kind,
                 static_cast<unsigned long long>(key_hash),
                 std::hash<std::thread::id>{}(std::this_thread::get_id()),
                 unoptimized ? " unoptimized" : "");
    for (size_t stage = 0; stage < stage_hashes.size(); ++stage) {
        if (stage_hashes[stage] != 0) {
            std::fprintf(file, " s%zu:%016llx", stage,
                         static_cast<unsigned long long>(stage_hashes[stage]));
        }
    }
    if (elapsed_ms >= 0.0) {
        std::fprintf(file, " ms=%.1f", elapsed_ms);
    }
    std::fputc('\n', file);
    std::fflush(file);
}

} // namespace Vulkan
