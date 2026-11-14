// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/fs/file.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/cache_writer.h"

/// The driver layer of shader cache v2: what the driver compiled, per pipeline, in a log of its own
/// beside the title's v2 log (DriverFileName), framed and written the same way (cache_format.h).
/// The file holds the blobs of one device and driver, named by its DeviceIdentity record (key 0);
/// a file of another identity is discarded whole when it is opened. The v2 log is never touched.
namespace VideoCommon::ShaderCacheV2 {

/// What a driver blob is valid for
struct DeviceIdentity {
    u32 vendor_id{};
    u32 device_id{};
    u32 driver_id{};
    u32 driver_version{};
    std::array<u8, 16> pipeline_cache_uuid{};

    bool operator==(const DeviceIdentity&) const = default;
};

[[nodiscard]] std::vector<u8> EncodeIdentity(const DeviceIdentity& identity);
[[nodiscard]] std::optional<DeviceIdentity> DecodeIdentity(std::span<const u8> payload);

/// The driver's binaries of one pipeline (VK_KHR_pipeline_binary): the payload of a
/// GraphicsBinary or ComputeBinary record keyed by the pipeline's v2 key
struct PipelineBinaries {
    struct Binary {
        std::vector<u8> key;
        std::vector<u8> data;

        bool operator==(const Binary&) const = default;
    };

    /// The driver's key of the create info they were made from. The SPIR-V of a v2 key changes
    /// with the translator, so a create info with another key doesn't take them.
    std::vector<u8> pipeline_key;
    std::vector<Binary> binaries;

    bool operator==(const PipelineBinaries&) const = default;
};

/// More binaries for one pipeline are taken for a damaged record
inline constexpr u32 MaxPipelineBinaries = 64;

[[nodiscard]] std::vector<u8> EncodeBinaries(const PipelineBinaries& binaries);
[[nodiscard]] std::optional<PipelineBinaries> DecodeBinaries(std::span<const u8> payload);

struct DriverBlobOptions {
    /// Closing compacts the file when its live records take more than this...
    u64 max_bytes{512ULL << 20};
    /// ...down to this, keeping the blobs the session used or wrote first, then the newest
    u64 target_bytes{384ULL << 20};
    /// It also compacts when dead records take this much and a quarter of the file
    u64 min_dead_bytes{64ULL << 20};
    /// No blob is stored in a session once the file would take more than this, dead records
    /// included (Full); closing then brings it back under max_bytes
    u64 ceiling_bytes{1ULL << 30};
    /// When above max_bytes, closing keeps every blob the session used or wrote up to this many
    /// bytes, past max_bytes and target_bytes if they need it: the limit follows what the game uses
    u64 session_cap_bytes{0};
    WriterOptions writer{};
};

/// Counts since the store was opened
struct DriverBlobCounts {
    u64 loaded{};       ///< Blobs the file held when it was opened
    u64 loaded_bytes{}; ///< Its size then
    u64 used{};         ///< Pipelines built from stored blobs
    u64 rejected{};     ///< Stored blobs not taken: damaged, stale, or refused by the driver
    u64 written{};      ///< Blobs queued for the file
    u64 written_bytes{};
};

/// Why the store started without the blobs the file held
enum class DriverBlobDiscard {
    None,
    OtherIdentity, ///< They were made on another device or driver
    NoIdentity,    ///< The file names no identity, or it is another format version's
    Failed,        ///< Such a file couldn't be removed: the store stays off for the session
};

/// A driver blob file. Any thread looks blobs up, rejects and stores them; the writer thread alone
/// writes the file. Only an index stays in memory: Find reads the blob from the file.
class DriverBlobStore {
public:
    explicit DriverBlobStore(std::filesystem::path file, const DeviceIdentity& identity,
                             DriverBlobOptions options = {});
    ~DriverBlobStore();

    DriverBlobStore(const DriverBlobStore&) = delete;
    DriverBlobStore& operator=(const DriverBlobStore&) = delete;

    /// False after DriverBlobDiscard::Failed: nothing is read or written.
    [[nodiscard]] bool Active() const noexcept {
        return writer.has_value();
    }

    [[nodiscard]] DriverBlobDiscard Discarded() const noexcept {
        return discarded;
    }

    /// The stored payload of this pipeline, read from the file. Empty when there is none, or it is
    /// damaged (which counts as rejected).
    [[nodiscard]] std::optional<std::vector<u8>> Find(RecordKind kind, u64 key);

    /// A pipeline was built from the payload Find gave
    void Used(RecordKind kind, u64 key);

    /// The payload Find gave can't be used: it isn't given again, and Store replaces it.
    void Reject(RecordKind kind, u64 key);

    /// Queues the payload of a pipeline for the file, replacing a stored one. A pipeline stored in
    /// this session isn't stored again.
    void Store(RecordKind kind, u64 key, std::span<const u8> payload);

    /// The file has reached ceiling_bytes: nothing more is stored this session.
    [[nodiscard]] bool Full() const;

    [[nodiscard]] DriverBlobCounts Counts() const;

    /// Writes the queued blobs, and compacts the file when it is due (see DriverBlobOptions). Find
    /// gives nothing afterwards. Destruction calls it.
    void Close();

private:
    enum class State : u8 {
        Stored,   ///< In the file at `offset`
        Rejected, ///< In the file, but not to be used
        Written,  ///< Queued in this session
    };

    struct Entry {
        u64 offset{};
        u32 payload_size{};
        State state{};
        bool used{}; ///< Used or written in this session
    };

    struct EntryKey {
        RecordKind kind;
        u64 key;
        bool operator==(const EntryKey&) const = default;
    };

    struct EntryKeyHash {
        size_t operator()(const EntryKey& value) const noexcept {
            return static_cast<size_t>(value.key ^
                                       (u64{static_cast<u16>(value.kind)} * 0x9e3779b97f4a7c15ULL));
        }
    };

    const std::filesystem::path file;
    const DriverBlobOptions options;

    mutable std::mutex mutex;
    std::unordered_map<EntryKey, Entry, EntryKeyHash> index;
    Common::FS::IOFile reader;
    u64 file_bytes{}; ///< The file's size with the blobs queued for it
    u64 dead_bytes{}; ///< Of those, the bytes compaction would remove
    DriverBlobDiscard discarded{};
    bool closed{};

    u64 loaded{};
    u64 loaded_bytes{};
    std::atomic<u64> used{};
    std::atomic<u64> rejected{};
    std::atomic<u64> written{};
    std::atomic<u64> written_bytes{};

    std::optional<CacheWriter> writer;
};

} // namespace VideoCommon::ShaderCacheV2
