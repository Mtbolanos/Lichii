// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <unordered_map>

#include "common/common_types.h"
#include "video_core/shader_cache_v2/cache_writer.h"
#include "video_core/shader_cache_v2/records.h"

namespace VideoCommon {
struct LoweredNativeCode;
}

namespace VideoCommon::ShaderCacheV2 {

struct LogRecord;

/// The log of a title, in its shader cache directory
inline constexpr std::string_view FileName{"lichii-cache.v2.bin"};
/// What the driver compiled for that title's pipelines, beside it (driver_blobs.h)
inline constexpr std::string_view DriverFileName{"lichii-cache.v2.driver.bin"};

/// What the log held for a program the shader cache captured
enum class ProgramLookup : u8 {
    Hit,   ///< A record of this build's lowering: its code is used without lowering
    Miss,  ///< No record of it
    Stale, ///< A record of another build's lowering
    Count,
};

struct FoundProgram {
    ProgramLookup lookup{ProgramLookup::Miss};
    std::shared_ptr<const LoweredNativeCode> code; ///< On a hit
};

/// A title's v2 log: the thread writing it, and the programs it holds.
class Cache {
public:
    explicit Cache(std::filesystem::path file, WriterOptions options = {});
    ~Cache();

    [[nodiscard]] CacheWriter& Writer() noexcept {
        return writer;
    }

    /// Notes a Program record the log holds (see ProgramStamp), so that it isn't written again
    void AddProgram(u64 key, u64 lowering_stamp);

    /// Takes the Program records of the log read at boot: the ones of this build's lowering are
    /// looked up from then on (FindProgram), and all are noted (AddProgram). Called once, while
    /// FindProgram may already run on another thread.
    void LoadPrograms(std::span<const LogRecord> live);

    /// The record of the program these inputs describe, if LoadPrograms took one of this build's
    /// lowering. Without locks or allocations on a hit; safe beside the writer thread.
    [[nodiscard]] FoundProgram FindProgram(const ProgramKeyInputs& inputs);

    /// Counts a lookup (FindProgram) for the stutter trace and the summary logged on destruction:
    /// the time to capture and look the program up, and to lower it after a hit was missed.
    void CountProgram(ProgramLookup lookup, u64 lookup_ns, u64 lowering_ns);

    /// Queues the record of a program unless the log holds it from this build's lowering, or
    /// from another build's after KeepStalePrograms
    void RecordProgram(const ProgramRecord& record);

    /// Leaves the records of another build's lowering in place for this session, when the boot
    /// load didn't lower them again: the pipeline records built from them move to this build's
    /// code only when a later boot load does
    void KeepStalePrograms();

private:
    /// A record of this build's lowering, its code moved out into `code`
    struct LoadedProgram {
        ProgramRecord record;
        std::shared_ptr<const LoweredNativeCode> code;
    };

    struct ProgramCounts {
        std::atomic<u64> lookups{};
        std::atomic<u64> lookup_ns{};
        std::atomic<u64> lowering_ns{};
    };

    std::mutex mutex;
    std::unordered_map<u64, u64> programs; ///< Record key -> lowering stamp
    bool keep_stale{};                     ///< Under mutex
    /// Written by LoadPrograms only before `loaded_ready` is set, then never changed
    std::unordered_map<u64, LoadedProgram> loaded;
    std::atomic<bool> loaded_ready{};
    std::array<ProgramCounts, static_cast<size_t>(ProgramLookup::Count)> counts;
    CacheWriter writer;
};

} // namespace VideoCommon::ShaderCacheV2
