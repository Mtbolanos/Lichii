// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/fs/file_lock.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/driver_blobs.h"

namespace {

namespace V2 = VideoCommon::ShaderCacheV2;
using V2::RecordKind;
using namespace std::chrono_literals;

constexpr V2::WriterOptions FastWriter{
    .idle = 1ms, .interval = 5ms, .lock_wait = 5ms, .retry = 5ms};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-shader-cache-v2-driver-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::vector<u8> ReadAll(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void WriteAll(const std::filesystem::path& file, std::span<const u8> bytes) {
    std::ofstream stream(file, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

V2::DeviceIdentity TestIdentity() {
    V2::DeviceIdentity identity{
        .vendor_id = 0x10de,
        .device_id = 0x2206,
        .driver_id = 4,
        .driver_version = 0x9b448000,
    };
    for (size_t i = 0; i < identity.pipeline_cache_uuid.size(); ++i) {
        identity.pipeline_cache_uuid[i] = static_cast<u8>(0xa0 + i);
    }
    return identity;
}

/// The binaries a driver might give for pipeline `index`, of a size that differs between them
V2::PipelineBinaries TestBinaries(u64 index, size_t size = 0) {
    V2::PipelineBinaries binaries;
    binaries.pipeline_key.assign(32, static_cast<u8>(index));
    const size_t count = 1 + index % 3;
    for (size_t i = 0; i < count; ++i) {
        V2::PipelineBinaries::Binary binary;
        binary.key.assign(20, static_cast<u8>(index * 7 + i));
        binary.data.resize(size != 0 ? size : 16 + (index * 53 + i * 11) % 700);
        for (size_t j = 0; j < binary.data.size(); ++j) {
            binary.data[j] = static_cast<u8>(index * 131 + i * 17 + j);
        }
        binaries.binaries.push_back(std::move(binary));
    }
    return binaries;
}

std::vector<u8> TestPayload(u64 index, size_t size = 0) {
    return V2::EncodeBinaries(TestBinaries(index, size));
}

RecordKind TestKind(u64 index) {
    return index % 3 == 0 ? RecordKind::ComputeBinary : RecordKind::GraphicsBinary;
}

/// Stores blobs 0..count-1 in a new file, as a session compiling them would
void StoreBlobs(const std::filesystem::path& file, u64 count) {
    V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
    REQUIRE(store.Active());
    for (u64 i = 0; i < count; ++i) {
        store.Store(TestKind(i), 0x100 + i, TestPayload(i));
    }
}

/// The offsets in the file where the record of each blob begins and ends
std::vector<std::pair<u64, u64>> BlobRanges(const std::filesystem::path& file) {
    const V2::LogFile log = V2::ReadLogFile(file);
    std::vector<std::pair<u64, u64>> ranges;
    for (const V2::LogRecord& record : log.parsed.live) {
        if (record.kind != RecordKind::DeviceIdentity) {
            ranges.emplace_back(record.offset,
                                record.offset + V2::RecordSize(record.payload.size()));
        }
    }
    return ranges;
}

} // Anonymous namespace

TEST_CASE("Shader cache v2 driver blobs: payloads read back as written",
          "[video_core][shader_cache_v2]") {
    const V2::DeviceIdentity identity = TestIdentity();
    REQUIRE(V2::DecodeIdentity(V2::EncodeIdentity(identity)) == identity);
    for (u64 i = 0; i < 6; ++i) {
        const V2::PipelineBinaries binaries = TestBinaries(i);
        REQUIRE(V2::DecodeBinaries(V2::EncodeBinaries(binaries)) == binaries);
    }
    // Every shorter or longer payload is refused
    const std::vector<u8> identity_payload = V2::EncodeIdentity(identity);
    const std::vector<u8> payload = TestPayload(4);
    for (size_t size = 0; size < payload.size(); ++size) {
        REQUIRE_FALSE(V2::DecodeBinaries(std::span{payload}.first(size)).has_value());
    }
    for (size_t size = 0; size < identity_payload.size(); ++size) {
        REQUIRE_FALSE(V2::DecodeIdentity(std::span{identity_payload}.first(size)).has_value());
    }
    std::vector<u8> longer = payload;
    longer.push_back(0);
    REQUIRE_FALSE(V2::DecodeBinaries(longer).has_value());
    // Neither decoder takes the other's payload
    REQUIRE_FALSE(V2::DecodeBinaries(identity_payload).has_value());
    REQUIRE_FALSE(V2::DecodeIdentity(payload).has_value());
    // Keys longer than VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR, and no binaries at all
    V2::PipelineBinaries long_key = TestBinaries(1);
    long_key.pipeline_key.resize(33);
    REQUIRE_FALSE(V2::DecodeBinaries(V2::EncodeBinaries(long_key)).has_value());
    V2::PipelineBinaries empty = TestBinaries(1);
    empty.binaries.clear();
    REQUIRE_FALSE(V2::DecodeBinaries(V2::EncodeBinaries(empty)).has_value());
}

TEST_CASE("Shader cache v2 driver blobs: a file round trip", "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    {
        V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
        REQUIRE(store.Active());
        REQUIRE(store.Discarded() == V2::DriverBlobDiscard::None);
        REQUIRE(store.Counts().loaded == 0);
        for (u64 i = 0; i < 12; ++i) {
            store.Store(TestKind(i), 0x100 + i, TestPayload(i));
        }
        // Written in this session: not stored again, and not read back before the next one
        store.Store(TestKind(0), 0x100, TestPayload(99));
        REQUIRE_FALSE(store.Find(TestKind(0), 0x100).has_value());
        REQUIRE(store.Counts().written == 12);
    }
    V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
    REQUIRE(store.Discarded() == V2::DriverBlobDiscard::None);
    const V2::DriverBlobCounts counts = store.Counts();
    REQUIRE(counts.loaded == 12);
    REQUIRE(counts.loaded_bytes == std::filesystem::file_size(file));
    for (u64 i = 0; i < 12; ++i) {
        INFO("blob " << i);
        const auto payload = store.Find(TestKind(i), 0x100 + i);
        REQUIRE(payload.has_value());
        REQUIRE(*payload == TestPayload(i));
        store.Used(TestKind(i), 0x100 + i);
    }
    // The kind is part of the key
    REQUIRE_FALSE(store.Find(RecordKind::GraphicsBinary, 0x100).has_value());
    REQUIRE_FALSE(store.Find(RecordKind::ComputeBinary, 0x999).has_value());
    REQUIRE(store.Counts().used == 12);
    REQUIRE(store.Counts().rejected == 0);
}

TEST_CASE("Shader cache v2 driver blobs: another device or driver discards the file whole",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    // The v2 log beside it is never touched.
    const auto log = temp.path / V2::FileName;
    std::vector<u8> log_bytes;
    V2::EncodeRecord(log_bytes, RecordKind::Graphics, 7, std::vector<u8>{1, 2, 3});
    REQUIRE(V2::AppendToLog(log, log_bytes) == V2::WriteResult::Written);
    const std::vector<u8> log_before = ReadAll(log);

    const auto changes = std::vector<void (*)(V2::DeviceIdentity&)>{
        [](V2::DeviceIdentity& id) { ++id.vendor_id; },
        [](V2::DeviceIdentity& id) { ++id.device_id; },
        [](V2::DeviceIdentity& id) { ++id.driver_id; },
        [](V2::DeviceIdentity& id) { ++id.driver_version; },
        [](V2::DeviceIdentity& id) { id.pipeline_cache_uuid[15] ^= 1; },
    };
    for (size_t change = 0; change < changes.size(); ++change) {
        INFO("change " << change);
        std::filesystem::remove(file);
        StoreBlobs(file, 5);
        V2::DeviceIdentity other = TestIdentity();
        changes[change](other);
        {
            V2::DriverBlobStore store{file, other, {.writer = FastWriter}};
            REQUIRE(store.Active());
            REQUIRE(store.Discarded() == V2::DriverBlobDiscard::OtherIdentity);
            REQUIRE(store.Counts().loaded == 0);
            for (u64 i = 0; i < 5; ++i) {
                REQUIRE_FALSE(store.Find(TestKind(i), 0x100 + i).has_value());
            }
            store.Store(RecordKind::GraphicsBinary, 0x500, TestPayload(50));
        }
        // The new file holds the new identity and its blob alone
        const V2::LogFile read = V2::ReadLogFile(file);
        REQUIRE(read.parsed.live.size() == 2);
        REQUIRE(read.parsed.live[0].kind == RecordKind::DeviceIdentity);
        REQUIRE(V2::DecodeIdentity(read.parsed.live[0].payload) == other);
        REQUIRE(read.parsed.live[1].key == 0x500);
        // The old identity now discards it in turn
        V2::DriverBlobStore back{file, TestIdentity(), {.writer = FastWriter}};
        REQUIRE(back.Discarded() == V2::DriverBlobDiscard::OtherIdentity);
    }
    REQUIRE(ReadAll(log) == log_before);

    // A file that names no identity, or another format version's
    std::vector<u8> no_identity;
    const auto header = V2::EncodeFileHeader(1);
    no_identity.assign(header.begin(), header.end());
    V2::EncodeRecord(no_identity, RecordKind::GraphicsBinary, 0x100, TestPayload(0));
    WriteAll(file, no_identity);
    {
        V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
        REQUIRE(store.Discarded() == V2::DriverBlobDiscard::NoIdentity);
        REQUIRE_FALSE(store.Find(RecordKind::GraphicsBinary, 0x100).has_value());
    }
    REQUIRE(V2::ReadLogFile(file).parsed.live.size() == 1);
    REQUIRE(ReadAll(log) == log_before);
}

TEST_CASE("Shader cache v2 driver blobs: a file another emulator holds stays as it is",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    StoreBlobs(file, 3);
    const std::vector<u8> before = ReadAll(file);
    V2::DeviceIdentity other = TestIdentity();
    ++other.driver_version;
    const Common::FS::FileLock lock{VideoCommon::PipelineCacheLockPath(file)};
    REQUIRE(lock.Held());
    {
        V2::DriverBlobStore store{file, other, {.writer = FastWriter}};
        REQUIRE_FALSE(store.Active());
        REQUIRE(store.Discarded() == V2::DriverBlobDiscard::Failed);
        REQUIRE_FALSE(store.Find(TestKind(0), 0x100).has_value());
        store.Store(RecordKind::GraphicsBinary, 0x500, TestPayload(50));
    }
    REQUIRE(ReadAll(file) == before);
}

TEST_CASE("Shader cache v2 driver blobs: a file cut off anywhere loses only the last blob",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto source = temp.path / "source.bin";
    StoreBlobs(source, 6);
    const std::vector<u8> bytes = ReadAll(source);
    const auto ranges = BlobRanges(source);
    REQUIRE(ranges.size() == 6);
    const auto [begin, end] = ranges.back();
    for (u64 size = begin; size < end; size += size + 64 < end ? 13 : 1) {
        INFO("size " << size);
        const auto file = temp.path / ("cut-" + std::to_string(size) + ".bin");
        WriteAll(file, std::span{bytes}.first(size));
        {
            V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
            REQUIRE(store.Discarded() == V2::DriverBlobDiscard::None);
            // Only padding missing still leaves the record whole
            const u64 framed = V2::RecordHeaderSize + TestPayload(5).size() + V2::RecordHashSize;
            REQUIRE(store.Counts().loaded == (size - begin >= framed ? 6 : 5));
            for (u64 i = 0; i < 5; ++i) {
                REQUIRE(store.Find(TestKind(i), 0x100 + i) == TestPayload(i));
            }
            // The next session's blobs follow the cut-off one
            store.Store(TestKind(5), 0x105, TestPayload(5));
            store.Store(RecordKind::GraphicsBinary, 0x200, TestPayload(20));
        }
        V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
        REQUIRE(store.Counts().loaded == 7);
        REQUIRE(store.Find(TestKind(5), 0x105) == TestPayload(5));
        REQUIRE(store.Find(RecordKind::GraphicsBinary, 0x200) == TestPayload(20));
    }
}

TEST_CASE("Shader cache v2 driver blobs: a damaged blob is dropped alone",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    StoreBlobs(file, 6);
    const auto ranges = BlobRanges(file);
    std::vector<u8> bytes = ReadAll(file);

    // Damaged before the store opens: the reader drops it
    std::vector<u8> damaged = bytes;
    damaged[ranges[2].first + V2::RecordHeaderSize + 9] ^= 0x10;
    WriteAll(file, damaged);
    {
        V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
        REQUIRE(store.Counts().loaded == 5);
        REQUIRE_FALSE(store.Find(TestKind(2), 0x102).has_value());
        REQUIRE(store.Find(TestKind(3), 0x103) == TestPayload(3));
    }

    // Damaged while it is open: Find checks the blob and rejects it
    WriteAll(file, bytes);
    V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
    REQUIRE(store.Counts().loaded == 6);
    {
        std::fstream stream(file, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(static_cast<std::streamoff>(ranges[4].first + V2::RecordHeaderSize + 30));
        const char flipped =
            static_cast<char>(bytes[ranges[4].first + V2::RecordHeaderSize + 30] ^ 1);
        stream.write(&flipped, 1);
    }
    REQUIRE_FALSE(store.Find(TestKind(4), 0x104).has_value());
    REQUIRE(store.Counts().rejected == 1);
    // Not given again; the next compile replaces it
    REQUIRE_FALSE(store.Find(TestKind(4), 0x104).has_value());
    REQUIRE(store.Find(TestKind(5), 0x105) == TestPayload(5));
}

TEST_CASE("Shader cache v2 driver blobs: a rejected blob is replaced",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    StoreBlobs(file, 4);
    {
        V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
        REQUIRE(store.Find(TestKind(1), 0x101) == TestPayload(1));
        store.Reject(TestKind(1), 0x101);
        REQUIRE_FALSE(store.Find(TestKind(1), 0x101).has_value());
        store.Store(TestKind(1), 0x101, TestPayload(77));
        REQUIRE(store.Counts().rejected == 1);
        REQUIRE(store.Counts().written == 1);
    }
    V2::DriverBlobStore store{file, TestIdentity(), {.writer = FastWriter}};
    REQUIRE(store.Counts().loaded == 4);
    REQUIRE(store.Find(TestKind(1), 0x101) == TestPayload(77));
    REQUIRE(store.Find(TestKind(2), 0x102) == TestPayload(2));
}

TEST_CASE("Shader cache v2 driver blobs: the file stays under its size cap",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    constexpr size_t BlobSize = 4000;
    // One binary each, so that every record takes the same size
    const auto payload = [](u64 index) {
        V2::PipelineBinaries binaries = TestBinaries(index, BlobSize);
        binaries.binaries.resize(1);
        return V2::EncodeBinaries(binaries);
    };
    const u64 record = V2::RecordSize(payload(0).size());
    const V2::DriverBlobOptions options{.max_bytes = record * 40,
                                        .target_bytes = record * 25,
                                        .min_dead_bytes = ~u64{0},
                                        .writer = FastWriter};
    {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        for (u64 i = 0; i < 30; ++i) {
            store.Store(RecordKind::GraphicsBinary, i, payload(i));
        }
    }
    // Under the cap: kept whole
    REQUIRE(BlobRanges(file).size() == 30);
    {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        // The session uses a few of the oldest and writes 20 more, past the cap
        for (u64 i = 0; i < 4; ++i) {
            REQUIRE(store.Find(RecordKind::GraphicsBinary, i).has_value());
            store.Used(RecordKind::GraphicsBinary, i);
        }
        for (u64 i = 30; i < 50; ++i) {
            store.Store(RecordKind::GraphicsBinary, i, payload(i));
        }
    }
    REQUIRE(std::filesystem::file_size(file) <= options.target_bytes + 4096);
    V2::DriverBlobStore store{file, TestIdentity(), options};
    // Still this device's file
    REQUIRE(store.Discarded() == V2::DriverBlobDiscard::None);
    REQUIRE(store.Counts().loaded == 24);
    // The session's (used and written) first, then the newest of the others
    for (u64 i = 0; i < 4; ++i) {
        REQUIRE(store.Find(RecordKind::GraphicsBinary, i) == payload(i));
    }
    for (u64 i = 30; i < 50; ++i) {
        REQUIRE(store.Find(RecordKind::GraphicsBinary, i).has_value());
    }
    for (u64 i = 4; i < 30; ++i) {
        REQUIRE_FALSE(store.Find(RecordKind::GraphicsBinary, i).has_value());
    }
    REQUIRE(V2::ReadLogFile(file).parsed.stats.dead_bytes == 0);
}

TEST_CASE("Shader cache v2 driver blobs: a session cap keeps what the session used past the cap",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    const auto payload = [](u64 index) {
        V2::PipelineBinaries binaries = TestBinaries(index, 4000);
        binaries.binaries.resize(1);
        return V2::EncodeBinaries(binaries);
    };
    const u64 record = V2::RecordSize(payload(0).size());
    V2::DriverBlobOptions options{.max_bytes = record * 20,
                                  .target_bytes = record * 15,
                                  .min_dead_bytes = ~u64{0},
                                  .ceiling_bytes = record * 100,
                                  .session_cap_bytes = record * 60,
                                  .writer = FastWriter};
    {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        for (u64 i = 0; i < 10; ++i) {
            store.Store(RecordKind::GraphicsBinary, i, payload(i));
        }
    }
    {
        // A precache-like session: uses nothing it had and writes 40 more, twice max_bytes
        V2::DriverBlobStore store{file, TestIdentity(), options};
        for (u64 i = 10; i < 50; ++i) {
            store.Store(RecordKind::GraphicsBinary, i, payload(i));
        }
    }
    {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        // Every blob the session wrote is kept; the unused older ones go
        REQUIRE(store.Counts().loaded == 40);
        for (u64 i = 10; i < 50; ++i) {
            REQUIRE(store.Find(RecordKind::GraphicsBinary, i).has_value());
            // Find doesn't mark a blob used; without this the store compacts to its target here
            store.Used(RecordKind::GraphicsBinary, i);
        }
    }
    // Past the session cap, the cap bounds it
    options.session_cap_bytes = record * 30;
    {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        REQUIRE(store.Counts().loaded == 40);
        for (u64 i = 10; i < 50; ++i) {
            store.Used(RecordKind::GraphicsBinary, i);
        }
    }
    REQUIRE(std::filesystem::file_size(file) <= record * 30 + 4096);
}

TEST_CASE("Shader cache v2 driver blobs: replaced blobs are compacted away",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    const V2::DriverBlobOptions options{.min_dead_bytes = 1, .writer = FastWriter};
    StoreBlobs(file, 8);
    const u64 before = std::filesystem::file_size(file);
    for (int session = 0; session < 3; ++session) {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        for (u64 i = 0; i < 8; ++i) {
            store.Reject(TestKind(i), 0x100 + i);
            store.Store(TestKind(i), 0x100 + i, TestPayload(i + 100 * (session + 1)));
        }
    }
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.stats.dead_bytes == 0);
    REQUIRE(log.parsed.live.size() == 9);
    REQUIRE(std::filesystem::file_size(file) < before * 2);
    V2::DriverBlobStore store{file, TestIdentity(), options};
    for (u64 i = 0; i < 8; ++i) {
        REQUIRE(store.Find(TestKind(i), 0x100 + i) == TestPayload(i + 300));
    }
}

TEST_CASE("Shader cache v2: a compaction limit keeps the lowest ranks, then the newest",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    std::vector<u8> records;
    const std::vector<u8> payload(100, 7);
    for (u64 key = 0; key < 10; ++key) {
        V2::EncodeRecord(records, RecordKind::Compute, key, payload);
    }
    REQUIRE(V2::AppendToLog(file, records) == V2::WriteResult::Written);
    const u64 record = V2::RecordSize(payload.size());
    // Under the limit nothing goes
    REQUIRE(V2::CompactLog(file, {.max_bytes = record * 10, .target_bytes = record}) ==
            V2::WriteResult::Written);
    REQUIRE(V2::ReadLogFile(file).parsed.live.size() == 10);
    // Past it: key 1 and 3 first, then the newest, in the file's order
    REQUIRE(V2::CompactLog(file, {.max_bytes = record * 9,
                                  .target_bytes = record * 5,
                                  .rank = [](const V2::LogRecord& r) -> u32 {
                                      return r.key == 1 || r.key == 3 ? 0 : 1;
                                  }}) == V2::WriteResult::Written);
    std::vector<u64> keys;
    for (const V2::LogRecord& r : V2::ReadLogFile(file).parsed.live) {
        keys.push_back(r.key);
    }
    REQUIRE(keys == std::vector<u64>{1, 3, 7, 8, 9});
}

TEST_CASE("Shader cache v2 driver blobs: a session stores nothing past the ceiling",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / V2::DriverFileName;
    const auto payload = [](u64 index) {
        V2::PipelineBinaries binaries = TestBinaries(index, 4000);
        binaries.binaries.resize(1);
        return V2::EncodeBinaries(binaries);
    };
    const u64 record = V2::RecordSize(payload(0).size());
    const u64 identity =
        V2::FileHeaderSize + V2::RecordSize(V2::EncodeIdentity(TestIdentity()).size());
    const V2::DriverBlobOptions options{.max_bytes = record * 4,
                                        .target_bytes = record * 3,
                                        .min_dead_bytes = ~u64{0},
                                        .ceiling_bytes = identity + record * 5,
                                        .writer = FastWriter};
    {
        V2::DriverBlobStore store{file, TestIdentity(), options};
        for (u64 i = 0; i < 4; ++i) {
            store.Store(RecordKind::GraphicsBinary, i, payload(i));
        }
        REQUIRE_FALSE(store.Full());
        for (u64 i = 4; i < 10; ++i) {
            store.Store(RecordKind::GraphicsBinary, i, payload(i));
        }
        REQUIRE(store.Full());
        REQUIRE(store.Counts().written == 5);
    }
    // Closing compacts it back under the cap
    REQUIRE(std::filesystem::file_size(file) <= options.target_bytes + 4096);
    V2::DriverBlobStore store{file, TestIdentity(), options};
    REQUIRE_FALSE(store.Full());
    REQUIRE(store.Counts().loaded > 0);
    REQUIRE_FALSE(store.Find(RecordKind::GraphicsBinary, 5).has_value());
}
