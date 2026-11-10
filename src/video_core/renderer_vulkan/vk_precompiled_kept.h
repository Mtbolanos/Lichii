// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The shader infos and SPIR-V the precompiled preparation keeps for each pipeline it builds
// (zortos293/Lichii#896, re-implemented for Fritchii), so a draw of a precompiled pipeline can
// be created from them instead of translating its programs on the GPU thread
// (EDEN_NX2_PRECOMPILED_FAST, PipelineCache::CreatePrecompiledGraphicsPipeline).
//
// The file (precompiled_infos.fork-v<N>.bin beside the precompiled file) is written as the
// preparation goes, as an append-only log, so an interrupted preparation resumes it like it
// resumes its checkpoint. Only its index is held in memory at run time (Ocarina of Time: about
// 10 MB for 132k pipelines); every info, SPIR-V and recorded environment is read from disk with
// a positional read when a pipeline needs it, never the whole file.
//
// Layout: "NX2KEPT4", u32 stamp size, the stamp, then entries: u8 1 (info) or 2 (SPIR-V, one
// zstd frame of its words), u64 content hash, u32 size, the bytes; u8 3 (record), u32 record
// index in the precompiled file, then a Record (72 bytes); u8 4 ends the entries. Then the
// index: u32 graphics count, u32 compute count, u32 blob count, the Records of each kind sorted
// by key hash, per blob u64 offset, u32 size, u32 0, then per Record (graphics, then compute)
// the u64 offset of its record in the precompiled file. Last, u64 index offset and "NX2KEPTE".
// Blob ids number the info and SPIR-V blobs together, in the order they were written.
//
// Validity: the stamp holds what is cheap and decisive (the prepared stamp with the release
// version, driver and GPU, the cache version, the SPIR-V settings, graphics pipeline library
// support, the OoT visual workaround, canonical keys). The build itself is not in it: at boot a
// sample of the kept pipelines is translated again from their recorded environments and compared
// word for word (PipelineCache::CheckKeptSample), so a new build whose translation is the same
// keeps the file, and one whose translation changed prepares it again.

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace Shader {
struct Info;
}

namespace VideoCommon {
class FileEnvironment;
}

namespace Vulkan::PrecompiledKept {

constexpr u32 NoBlob = ~0U;

/// Record flags.
constexpr u8 FlagLibraries = 1; ///< Emitted for graphics pipeline libraries (one set per stage)

struct Record {
    u64 key_hash;
    /// The precompiled file's environments, in stage order (env_count of them)
    std::array<u32, 5> envs;
    /// Per pipeline stage (vertex B to fragment; a compute pipeline's is stage 0), NoBlob for none
    std::array<u32, 5> infos;
    std::array<u32, 5> spirv;
    u8 env_count;
    u8 kind; ///< 0 graphics, 1 compute
    u8 flags;
    u8 reserved;
};
static_assert(sizeof(Record) == 72 && std::is_trivially_copyable_v<Record> &&
              std::has_unique_object_representations_v<Record>);

/// Bytes read from kept files and recorded environments at run time (for the 10 s line).
std::atomic<u64>& BytesRead();

/// The seed of the boot check's sample: the running exe's size and modification time, hashed.
/// The same build checks the same pipelines on every boot; each new build checks others, so a
/// translation change the sample of one build misses can be found by the next.
u64 SampleSeed();

/// `wanted` positions out of `count` (all of them when there are fewer), one in each of `wanted`
/// equal stretches of the index, picked in it from `seed`.
std::vector<size_t> SamplePositions(size_t count, size_t wanted, u64 seed);

/// A file read with positional reads, safe from several threads at once; opened so the file can
/// still be renamed or deleted while it is open.
class PositionalFile {
public:
    explicit PositionalFile(const std::filesystem::path& path);
    ~PositionalFile();
    PositionalFile(const PositionalFile&) = delete;
    PositionalFile& operator=(const PositionalFile&) = delete;

    [[nodiscard]] bool IsOpen() const;
    /// Reads out.size() bytes at offset; false when the file ends first or the read fails.
    [[nodiscard]] bool ReadAt(u64 offset, std::span<u8> out) const;

private:
#ifdef _WIN32
    void* handle;
#else
    int fd{-1};
#endif
};

/// The run time index of a kept file. Reads are thread-safe (the GPU thread checks and creates,
/// pipeline workers build from the SPIR-V); a pipeline's build holds the index it was made from.
class Index {
public:
    /// The kept file `kept` written for `stamp` and the environment index of `precompiled`
    /// (cache version `cache_version`); null with `why` set when either does not match.
    static std::shared_ptr<const Index> Load(const std::filesystem::path& kept,
                                             const std::filesystem::path& precompiled,
                                             const std::string& stamp, u32 cache_version,
                                             std::string& why);

    /// The records of a key (a compute key has one per storage pointer state it was prepared in).
    [[nodiscard]] std::span<const Record> Graphics(u64 key_hash) const;
    [[nodiscard]] std::span<const Record> Compute(u64 key_hash) const;

    [[nodiscard]] bool ReadInfoBytes(u32 blob, std::vector<u8>& bytes) const;
    [[nodiscard]] bool ReadInfo(u32 blob, Shader::Info& info) const;
    [[nodiscard]] bool ReadSpirv(u32 blob, std::vector<u32>& words) const;
    [[nodiscard]] bool ReadEnvironment(u32 env, VideoCommon::FileEnvironment& out) const;
    /// Bytes of the precompiled file (a record's environment ids and key, for the boot check).
    [[nodiscard]] bool ReadPrecompiled(u64 offset, std::span<u8> out) const;

    /// Every record of a kind, sorted by key hash (the boot check samples them).
    [[nodiscard]] std::span<const Record> AllGraphics() const noexcept {
        return graphics;
    }
    [[nodiscard]] std::span<const Record> AllCompute() const noexcept {
        return compute;
    }
    /// Where the precompiled file's record of AllGraphics()[position] (compute: of
    /// AllCompute()[position]) starts.
    [[nodiscard]] bool RecordOffset(bool compute_record, size_t position, u64& offset) const;

    [[nodiscard]] size_t GraphicsCount() const noexcept {
        return graphics.size();
    }
    [[nodiscard]] size_t ComputeCount() const noexcept {
        return compute.size();
    }
    /// What the index holds in memory.
    [[nodiscard]] size_t MemoryBytes() const noexcept;

    Index(const std::filesystem::path& kept, const std::filesystem::path& precompiled);

private:
    friend class Writer;

    [[nodiscard]] bool ReadBlob(u32 blob, std::vector<u8>& bytes) const;

    std::vector<Record> graphics; ///< Sorted by key hash
    std::vector<Record> compute;  ///< Sorted by key hash
    std::vector<std::pair<u64, u32>> blobs;        ///< Offset, size
    std::vector<std::pair<u64, u64>> environments; ///< The precompiled file's: offset, size
    /// Where the precompiled record offsets start in the kept file (read per sampled record), or
    /// for an unfinished file (Writer::PartIndex) the offsets themselves, graphics then compute.
    u64 record_offsets_at{};
    std::vector<u64> record_offsets;
    PositionalFile kept_file;
    PositionalFile precompiled_file;
};

/// Writes a kept file while the preparation runs (several workers at once).
class Writer {
public:
    /// Starts `part` for `stamp`, or with `resume` continues it when it was started for the same
    /// stamp (the entries up to the last complete one are kept). `total`: the precompiled
    /// file's record count. False when the file cannot be written.
    bool Open(const std::filesystem::path& part, const std::string& stamp, bool resume,
              size_t total);
    /// Records kept from an earlier, interrupted preparation.
    [[nodiscard]] size_t Resumed() const noexcept {
        return resumed;
    }

    /// Keeps one built pipeline: per stage its serialized info and its SPIR-V (empty for none).
    void Keep(size_t index, u8 kind, u8 flags, u64 key_hash, std::span<const u32> envs,
              const std::array<std::vector<u8>, 5>& infos,
              const std::array<std::vector<u32>, 5>& spirv);

    /// The records kept so far (an earlier, interrupted preparation resumed: maybe by another
    /// build), as an index over `part` for the boot check. `environments` and `record_offsets`
    /// are the precompiled file's environment table and where each of its records starts.
    [[nodiscard]] std::shared_ptr<const Index> PartIndex(
        const std::filesystem::path& part, const std::filesystem::path& precompiled,
        std::span<const std::pair<u64, u64>> environments, std::span<const u64> record_offsets);

    /// Flushes the log (before a checkpoint is written).
    void Flush();
    /// Writes the index and closes the file; false when it could not be written.
    /// `record_offsets`: where each record of the precompiled file starts.
    bool Finish(std::span<const u64> record_offsets);
    /// Closes the file as it is (an interrupted preparation), to be resumed.
    void Close();

    struct Totals {
        size_t records{};
        size_t infos{};
        size_t spirv{};
        u64 spirv_bytes{}; ///< Compressed
        u64 file_bytes{};
    };
    [[nodiscard]] Totals GetTotals();

private:
    bool Scan(const std::filesystem::path& part, const std::string& stamp, u64& valid_end);
    /// The kept records of each kind sorted by key hash, and their precompiled record offsets
    /// (graphics, then compute). Under the mutex.
    void SortedRecords(std::span<const u64> record_offsets, std::vector<Record>& graphics,
                       std::vector<Record>& compute, std::vector<u64>& offsets) const;
    u32 KeepBlob(u8 type, u64 hash, std::span<const u8> bytes);

    std::mutex mutex;
    std::ofstream file;
    u64 offset{};
    std::unordered_map<u64, u32> infos_by_hash;
    std::unordered_map<u64, u32> spirv_by_hash;
    std::vector<std::pair<u64, u32>> blobs;
    std::vector<Record> records; ///< By record index; key_hash 0 and env_count 0 when not kept
    std::vector<u8> present;
    size_t kept{};
    size_t resumed{};
    u64 spirv_bytes{};
    bool failed{};
};

} // namespace Vulkan::PrecompiledKept
