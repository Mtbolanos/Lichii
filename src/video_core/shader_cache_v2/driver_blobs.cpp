// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "common/fs/file_lock.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/driver_blobs.h"
#include "video_core/shader_cache_v2/payload_io.h"

namespace VideoCommon::ShaderCacheV2 {
namespace {

/// VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR
constexpr u32 MaxKeySize = 32;
/// How long closing waits for the last blobs and the compaction
constexpr std::chrono::seconds CloseDrain{60};

bool IsBlobKind(RecordKind kind) {
    return kind == RecordKind::GraphicsBinary || kind == RecordKind::ComputeBinary;
}

void PutBytes(PayloadWriter& out, std::span<const u8> bytes) {
    out.Put(static_cast<u32>(bytes.size()));
    out.PutSpan(bytes);
}

bool GetKey(PayloadReader& in, std::vector<u8>& key) {
    u32 size{};
    return in.Get(size) && size != 0 && size <= MaxKeySize && in.GetVector(key, size);
}

} // Anonymous namespace

std::vector<u8> EncodeIdentity(const DeviceIdentity& identity) {
    PayloadWriter out;
    out.Put(identity.vendor_id);
    out.Put(identity.device_id);
    out.Put(identity.driver_id);
    out.Put(identity.driver_version);
    out.Put(identity.pipeline_cache_uuid);
    return out.Take();
}

std::optional<DeviceIdentity> DecodeIdentity(std::span<const u8> payload) {
    PayloadReader in{payload};
    DeviceIdentity identity;
    if (!in.Get(identity.vendor_id) || !in.Get(identity.device_id) || !in.Get(identity.driver_id) ||
        !in.Get(identity.driver_version) || !in.Get(identity.pipeline_cache_uuid) || !in.AtEnd()) {
        return std::nullopt;
    }
    return identity;
}

std::vector<u8> EncodeBinaries(const PipelineBinaries& binaries) {
    PayloadWriter out;
    PutBytes(out, binaries.pipeline_key);
    out.Put(static_cast<u32>(binaries.binaries.size()));
    for (const PipelineBinaries::Binary& binary : binaries.binaries) {
        PutBytes(out, binary.key);
        out.Put(static_cast<u64>(binary.data.size()));
        out.PutSpan(std::span<const u8>{binary.data});
    }
    return out.Take();
}

std::optional<PipelineBinaries> DecodeBinaries(std::span<const u8> payload) {
    PayloadReader in{payload};
    PipelineBinaries binaries;
    u32 count{};
    if (!GetKey(in, binaries.pipeline_key) || !in.Get(count) || count == 0 ||
        count > MaxPipelineBinaries) {
        return std::nullopt;
    }
    binaries.binaries.resize(count);
    for (PipelineBinaries::Binary& binary : binaries.binaries) {
        u64 size{};
        if (!GetKey(in, binary.key) || !in.Get(size) || size == 0 ||
            !in.GetVector(binary.data, size)) {
            return std::nullopt;
        }
    }
    if (!in.AtEnd()) {
        return std::nullopt;
    }
    return binaries;
}

DriverBlobStore::DriverBlobStore(std::filesystem::path file_, const DeviceIdentity& identity,
                                 DriverBlobOptions options_)
    : file{std::move(file_)}, options{options_} {
    {
        // Read without the lock, as the v2 log is: appends are complete records or a cut-off tail.
        const LogFile log = ReadLogFile(file);
        bool same_identity{};
        bool has_identity{};
        for (const LogRecord& record : log.parsed.live) {
            if (record.kind == RecordKind::DeviceIdentity && record.key == 0) {
                has_identity = true;
                same_identity = DecodeIdentity(record.payload) == identity;
            }
        }
        if (log.parsed.header != HeaderStatus::Empty && !same_identity) {
            discarded =
                has_identity ? DriverBlobDiscard::OtherIdentity : DriverBlobDiscard::NoIdentity;
            const Common::FS::FileLock lock{PipelineCacheLockPath(file)};
            std::error_code error;
            if (!lock.Busy()) {
                std::filesystem::remove(file, error);
            }
            if (lock.Busy() || error) {
                LOG_WARNING(Render_Vulkan,
                            "Driver blobs: can't remove {}, made for another device or driver: "
                            "{}",
                            Common::FS::PathToUTF8String(file),
                            lock.Busy() ? "another emulator holds it" : error.message());
                discarded = DriverBlobDiscard::Failed;
                return;
            }
        } else if (same_identity) {
            for (const LogRecord& record : log.parsed.live) {
                if (!IsBlobKind(record.kind)) {
                    continue;
                }
                index.insert_or_assign(
                    EntryKey{record.kind, record.key},
                    Entry{.offset = record.offset,
                          .payload_size = static_cast<u32>(record.payload.size()),
                          .state = State::Stored});
            }
            loaded = index.size();
            loaded_bytes = log.parsed.stats.file_size;
            file_bytes = log.parsed.stats.file_size;
            dead_bytes = log.parsed.stats.dead_bytes;
            reader.Open(file, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile,
                        Common::FS::FileShareFlag::ShareReadWrite);
        }
    }
    writer.emplace(file, options.writer);
    if (discarded != DriverBlobDiscard::None || loaded_bytes == 0) {
        const std::vector<u8> payload = EncodeIdentity(identity);
        writer->Append(RecordKind::DeviceIdentity, 0, payload);
        file_bytes = FileHeaderSize + RecordSize(payload.size());
    }
}

DriverBlobStore::~DriverBlobStore() {
    Close();
}

std::optional<std::vector<u8>> DriverBlobStore::Find(RecordKind kind, u64 key) {
    std::scoped_lock lock{mutex};
    const auto it = index.find({kind, key});
    if (closed || it == index.end() || it->second.state != State::Stored) {
        return std::nullopt;
    }
    Entry& entry = it->second;
    std::vector<u8> bytes(RecordHeaderSize + entry.payload_size + RecordHashSize);
    std::optional<LogRecord> record;
    if (reader.Seek(static_cast<s64>(entry.offset)) &&
        reader.ReadSpan(std::span<u8>{bytes}) == bytes.size()) {
        record = ParseRecord(bytes, entry.offset);
    }
    if (!record || record->kind != kind || record->key != key) {
        entry.state = State::Rejected;
        rejected.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
    return std::vector<u8>(record->payload.begin(), record->payload.end());
}

void DriverBlobStore::Used(RecordKind kind, u64 key) {
    used.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (const auto it = index.find({kind, key}); it != index.end()) {
        it->second.used = true;
    }
}

void DriverBlobStore::Reject(RecordKind kind, u64 key) {
    rejected.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (const auto it = index.find({kind, key}); it != index.end()) {
        it->second.state = State::Rejected;
    }
}

void DriverBlobStore::Store(RecordKind kind, u64 key, std::span<const u8> payload) {
    std::scoped_lock lock{mutex};
    if (!writer || closed || file_bytes + RecordSize(payload.size()) > options.ceiling_bytes) {
        return;
    }
    const auto [it, inserted] = index.try_emplace({kind, key});
    if (!inserted) {
        if (it->second.state == State::Written) {
            return;
        }
        dead_bytes += RecordSize(it->second.payload_size);
    }
    it->second = {
        .payload_size = static_cast<u32>(payload.size()), .state = State::Written, .used = true};
    file_bytes += RecordSize(payload.size());
    written.fetch_add(1, std::memory_order_relaxed);
    written_bytes.fetch_add(RecordSize(payload.size()), std::memory_order_relaxed);
    // Under the lock, so that Close never ends the writer under it
    writer->Append(kind, key, payload);
}

bool DriverBlobStore::Full() const {
    std::scoped_lock lock{mutex};
    return file_bytes >= options.ceiling_bytes;
}

DriverBlobCounts DriverBlobStore::Counts() const {
    return {
        .loaded = loaded,
        .loaded_bytes = loaded_bytes,
        .used = used.load(std::memory_order_relaxed),
        .rejected = rejected.load(std::memory_order_relaxed),
        .written = written.load(std::memory_order_relaxed),
        .written_bytes = written_bytes.load(std::memory_order_relaxed),
    };
}

void DriverBlobStore::Close() {
    std::unique_lock lock{mutex};
    if (closed || !writer) {
        closed = true;
        return;
    }
    closed = true;
    // Before the compaction: Windows doesn't rename over a file that is open.
    reader.Close();
    u64 max_bytes = options.max_bytes;
    u64 target_bytes = options.target_bytes;
    if (options.session_cap_bytes > max_bytes) {
        // The device identity record is kept first
        u64 session_bytes = RecordSize(EncodeIdentity(DeviceIdentity{}).size());
        for (const auto& [entry_key, entry] : index) {
            if (entry.used) {
                session_bytes += RecordSize(entry.payload_size);
            }
        }
        target_bytes = std::max(target_bytes, std::min(session_bytes, options.session_cap_bytes));
        max_bytes = std::max(max_bytes, target_bytes);
    }
    if (file_bytes > max_bytes ||
        (dead_bytes >= options.min_dead_bytes && dead_bytes * 4 >= file_bytes)) {
        std::unordered_set<u64> first;
        for (const auto& [entry_key, entry] : index) {
            if (entry.used) {
                first.insert(EntryKeyHash{}(entry_key));
            }
        }
        LOG_INFO(Render_Vulkan, "Driver blobs: compacting {} ({} bytes, {} of them dead)",
                 Common::FS::PathToUTF8String(file), file_bytes, dead_bytes);
        writer->Compact({
            .max_bytes = max_bytes,
            .target_bytes = target_bytes,
            .rank = [first = std::move(first)](const LogRecord& record) -> u32 {
                if (record.kind == RecordKind::DeviceIdentity) {
                    return 0;
                }
                return first.contains(EntryKeyHash{}({record.kind, record.key})) ? 1 : 2;
            },
        });
    }
    lock.unlock();
    if (!writer->Drain(CloseDrain)) {
        LOG_WARNING(Render_Vulkan, "Driver blobs: {} isn't written yet",
                    Common::FS::PathToUTF8String(file));
    }
    writer.reset();
}

} // namespace VideoCommon::ShaderCacheV2
