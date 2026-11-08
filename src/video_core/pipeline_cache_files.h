// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>

#include <fmt/format.h>

#include "common/common_types.h"

namespace VideoCommon {

inline constexpr u32 OpenGLPipelineCacheVersion = 48;
inline constexpr u32 VulkanPipelineCacheVersion = 52;

// Official Eglantine (cache version 18 at c0cb3325) and this fork (a different version) both
// used <shader dir>/<title>/vulkan.bin and vulkan_pipelines.bin. Each build deletes a file
// with another version at boot, so alternating builds made every run start with no pipelines
// (MKW log: "Deleting old pipeline cache", "Total Pipeline Count: 0", then hundreds of
// pipelines built on the GPU thread). Version-suffixed names let the builds coexist.
[[nodiscard]] inline std::filesystem::path VersionedPipelineCachePath(
    const std::filesystem::path& directory, std::string_view stem, u32 version) {
    return directory / fmt::format("{}.fork-v{}.bin", stem, version);
}

/// True when the file starts with this magic number followed by this cache version.
[[nodiscard]] inline bool PipelineCacheHeaderMatches(const std::filesystem::path& file,
                                                     const std::array<char, 8>& magic,
                                                     u32 version) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }
    std::array<char, 8> file_magic{};
    u32 file_version{};
    stream.read(file_magic.data(), file_magic.size());
    stream.read(reinterpret_cast<char*>(&file_version), sizeof(file_version));
    return stream.good() && file_magic == magic && file_version == version;
}

/// Moves a legacy-named cache that this exact cache version wrote to its versioned name, so an
/// existing warm cache survives the rename. A file of any other version (another build's) is
/// never touched; neither is an existing versioned file.
inline bool AdoptLegacyPipelineCache(const std::filesystem::path& legacy,
                                     const std::filesystem::path& versioned,
                                     const std::array<char, 8>& magic, u32 version) {
    std::error_code error;
    if (std::filesystem::exists(versioned, error) || error) {
        return false;
    }
    if (!PipelineCacheHeaderMatches(legacy, magic, version)) {
        return false;
    }
    std::filesystem::rename(legacy, versioned, error);
    return !error;
}

/// Copies a cache whose records this build reads, written with another cache version, to
/// `destination` with this build's version in its header (from zortos293/Lichii#194). The
/// source is never changed; an existing destination is kept.
inline bool CopyCompatiblePipelineCache(const std::filesystem::path& source,
                                        const std::filesystem::path& destination,
                                        const std::array<char, 8>& magic, u32 source_version,
                                        u32 destination_version) {
    std::error_code error;
    if (std::filesystem::exists(destination, error) || error ||
        !PipelineCacheHeaderMatches(source, magic, source_version)) {
        return false;
    }
    auto temporary = destination;
    temporary += ".migrating";
    if (!std::filesystem::copy_file(source, temporary,
                                    std::filesystem::copy_options::overwrite_existing, error)) {
        return false;
    }
    {
        std::fstream stream(temporary, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(static_cast<std::streamoff>(magic.size()));
        stream.write(reinterpret_cast<const char*>(&destination_version),
                     sizeof(destination_version));
        stream.close();
        if (stream.fail()) {
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

/// Reads one pipeline record from `file` (positioned at its start, with exceptions enabled,
/// `end` its size) and returns the bytes that identify its pipeline, or nullopt when the record
/// is malformed. The bytes must compare equal for records of the same pipeline.
using PipelineRecordKeyReader =
    std::function<std::optional<std::string>(std::ifstream& file, std::streamoff end)>;

using PipelineKeySet = std::unordered_set<std::string>;

struct PipelineCacheMergeResult {
    enum class Status {
        Merged,         ///< `added_records` appended (possibly none)
        Missing,        ///< No source file
        WrongHeader,    ///< The source is not a cache of the expected magic and version
        BadSource,      ///< A source record does not parse; nothing was written
        BadDestination, ///< The destination does not parse; it was left as it is
        NoSpace,        ///< Not enough free disk space for the merged copy
        WriteFailed,    ///< Writing or renaming the merged copy failed; the destination is unchanged
    };
    Status status{};
    size_t added_pipelines{};
    size_t added_records{};
    size_t known_pipelines{}; ///< Source pipelines the destination already had
    std::string detail;
};

/// Appends to `destination` the records of `source` (a cache with `source_version` in its
/// header, otherwise identical records) whose pipeline key the destination does not hold, every
/// record of a new key in source order. The result is written next to `destination` and renamed
/// over it; the source is never changed. A destination that does not exist is created with
/// `destination_version`. `known_keys` holds the destination's keys: filled by scanning it when
/// empty, and extended by a successful merge, so several sources scan the destination once.
PipelineCacheMergeResult MergePipelineCache(const std::filesystem::path& source,
                                            const std::filesystem::path& destination,
                                            const std::array<char, 8>& magic, u32 source_version,
                                            u32 destination_version,
                                            const PipelineRecordKeyReader& read_key,
                                            std::optional<PipelineKeySet>& known_keys);

/// Adds to the used-pipelines list `destination` (magic, count, then u64 hash + u8 age per
/// entry) the hashes of `source` it lacks, keeping the lower age of a hash both hold, through a
/// temporary file. Returns how many were added, or nullopt when `source` is truncated or either
/// file has another magic (then nothing is written).
std::optional<size_t> MergeUsedPipelineLists(const std::filesystem::path& source,
                                             const std::filesystem::path& destination, u32 magic);

/// Size and modification time of a file, to tell whether it changed since it was merged.
struct PipelineCacheFileStamp {
    u64 size{};
    s64 modified{};
    bool operator==(const PipelineCacheFileStamp&) const = default;
};

[[nodiscard]] std::optional<PipelineCacheFileStamp> StampPipelineCacheFile(
    const std::filesystem::path& file);

/// The sidecar recording the stamp of each source file (by file name) when it was last merged.
using PipelineMergeLog = std::map<std::string, PipelineCacheFileStamp>;

[[nodiscard]] PipelineMergeLog ReadPipelineMergeLog(const std::filesystem::path& file);
bool WritePipelineMergeLog(const std::filesystem::path& file, const PipelineMergeLog& log);

} // namespace VideoCommon
