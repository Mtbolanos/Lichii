// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {

constexpr std::array<char, 8> Magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
constexpr std::array<char, 8> OtherMagic{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-pipeline-cache-files-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void WriteCache(const std::filesystem::path& file, const std::array<char, 8>& magic, u32 version,
                std::string_view body) {
    std::ofstream stream(file, std::ios::binary);
    stream.write(magic.data(), magic.size());
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    stream.write(body.data(), static_cast<std::streamsize>(body.size()));
}

std::string ReadAll(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string Header(u32 version) {
    return std::string(Magic.data(), Magic.size()) +
           std::string(reinterpret_cast<const char*>(&version), sizeof(version));
}

std::string Header(u32 version) {
    return std::string(Magic.data(), Magic.size()) +
           std::string(reinterpret_cast<const char*>(&version), sizeof(version));
}

// Merge tests use records of a key, a size and that many bytes.
std::string Record(u32 key, std::string_view body) {
    std::string record(reinterpret_cast<const char*>(&key), sizeof(key));
    const u32 size = static_cast<u32>(body.size());
    record.append(reinterpret_cast<const char*>(&size), sizeof(size));
    record.append(body);
    return record;
}

std::optional<std::string> ReadRecordKey(std::ifstream& file, std::streamoff end) {
    u32 key{};
    u32 size{};
    file.read(reinterpret_cast<char*>(&key), sizeof(key));
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (size > end) {
        return std::nullopt;
    }
    std::string body(size, '\0');
    file.read(body.data(), size);
    return std::string(reinterpret_cast<const char*>(&key), sizeof(key));
}

VideoCommon::PipelineCacheMergeResult Merge(const std::filesystem::path& source,
                                            const std::filesystem::path& destination,
                                            std::optional<VideoCommon::PipelineKeySet>& keys) {
    return VideoCommon::MergePipelineCache(source, destination, Magic, 54, 52, ReadRecordKey,
                                           keys);
}

void WriteUsed(const std::filesystem::path& file, u32 magic,
               const std::vector<std::pair<u64, u8>>& entries, u32 count) {
    std::ofstream stream(file, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [hash, age] : entries) {
        stream.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        stream.write(reinterpret_cast<const char*>(&age), sizeof(age));
    }
}

std::unordered_map<u64, u8> ReadUsed(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    u32 magic{};
    u32 count{};
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::unordered_map<u64, u8> used;
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 age{};
        stream.read(reinterpret_cast<char*>(&hash), sizeof(hash));
        stream.read(reinterpret_cast<char*>(&age), sizeof(age));
        used.emplace(hash, age);
    }
    return used;
}

constexpr u32 UsedMagic = 0x31505545;

} // Anonymous namespace

TEST_CASE("A newer cache's pipelines are appended once, its source untouched",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto own = temp.path / "vulkan-frizti.fork-v52.bin";
    const auto zortos = temp.path / "vulkan.fork-v54.bin";
    const std::string own_body = Record(1, "a") + Record(2, "b");
    WriteCache(own, Magic, 52, own_body);
    // Key 3 was stored twice (a rebuilt translation): both records come along.
    const std::string new_records = Record(3, "c") + Record(3, "cc") + Record(4, "d");
    WriteCache(zortos, Magic, 54, Record(2, "other b") + new_records);
    const std::string source_before = ReadAll(zortos);

    std::optional<VideoCommon::PipelineKeySet> keys;
    const auto result = Merge(zortos, own, keys);
    REQUIRE(result.status == VideoCommon::PipelineCacheMergeResult::Status::Merged);
    CHECK(result.added_pipelines == 2);
    CHECK(result.added_records == 3);
    CHECK(result.known_pipelines == 1);
    CHECK(ReadAll(own) == Header(52) + own_body + new_records);
    CHECK(ReadAll(zortos) == source_before);
    CHECK_FALSE(std::filesystem::exists(temp.path / "vulkan-frizti.fork-v52.bin.merging"));

    // Merging again adds nothing, with the keys kept or scanned anew.
    const std::string merged = ReadAll(own);
    CHECK(Merge(zortos, own, keys).added_records == 0);
    std::optional<VideoCommon::PipelineKeySet> fresh;
    CHECK(Merge(zortos, own, fresh).added_records == 0);
    CHECK(ReadAll(own) == merged);
}

TEST_CASE("A cache is created from a newer one when there is none", "[pipeline_cache_files]") {
    TempDir temp;
    const auto own = temp.path / "vulkan-frizti.fork-v52.bin";
    const auto zortos = temp.path / "vulkan.fork-v54.bin";
    WriteCache(zortos, Magic, 54, Record(7, "x") + Record(8, "y"));
    std::optional<VideoCommon::PipelineKeySet> keys;
    const auto result = Merge(zortos, own, keys);
    REQUIRE(result.status == VideoCommon::PipelineCacheMergeResult::Status::Merged);
    CHECK(result.added_pipelines == 2);
    CHECK(ReadAll(own) == Header(52) + Record(7, "x") + Record(8, "y"));
}

TEST_CASE("A broken or foreign cache is never merged", "[pipeline_cache_files]") {
    using Status = VideoCommon::PipelineCacheMergeResult::Status;
    TempDir temp;
    const auto own = temp.path / "vulkan-frizti.fork-v52.bin";
    const auto zortos = temp.path / "vulkan.fork-v54.bin";
    WriteCache(own, Magic, 52, Record(1, "a"));
    const std::string own_before = ReadAll(own);

    // The last record is cut short: nothing at all is taken, not even the first new one.
    WriteCache(zortos, Magic, 54, Record(5, "new") + Record(6, "truncated").substr(0, 10));
    std::optional<VideoCommon::PipelineKeySet> keys;
    CHECK(Merge(zortos, own, keys).status == Status::BadSource);
    CHECK(ReadAll(own) == own_before);

    WriteCache(zortos, Magic, 53, Record(5, "new"));
    CHECK(Merge(zortos, own, keys).status == Status::WrongHeader);
    WriteCache(zortos, OtherMagic, 54, Record(5, "new"));
    CHECK(Merge(zortos, own, keys).status == Status::WrongHeader);
    CHECK(Merge(temp.path / "missing.bin", own, keys).status == Status::Missing);
    CHECK(ReadAll(own) == own_before);

    // Our own cache does not parse: it is left for the loader as it is.
    WriteCache(zortos, Magic, 54, Record(5, "new"));
    WriteCache(own, Magic, 52, Record(1, "a").substr(0, 6));
    const std::string broken_own = ReadAll(own);
    std::optional<VideoCommon::PipelineKeySet> fresh;
    CHECK(Merge(zortos, own, fresh).status == Status::BadDestination);
    CHECK(ReadAll(own) == broken_own);
    CHECK_FALSE(std::filesystem::exists(temp.path / "vulkan-frizti.fork-v52.bin.merging"));
}

TEST_CASE("Used pipeline lists are merged with the lowest age", "[pipeline_cache_files]") {
    TempDir temp;
    const auto own = temp.path / "own_used.bin";
    const auto zortos = temp.path / "zortos_used.bin";
    WriteUsed(own, UsedMagic, {{1, 0}, {2, 3}}, 2);
    WriteUsed(zortos, UsedMagic, {{2, 1}, {3, 0}, {4, 5}}, 3);
    REQUIRE(VideoCommon::MergeUsedPipelineLists(zortos, own, UsedMagic) == 2);
    const auto used = ReadUsed(own);
    CHECK(used == std::unordered_map<u64, u8>{{1, 0}, {2, 1}, {3, 0}, {4, 5}});

    // A truncated list or one of another format changes nothing.
    const std::string before = ReadAll(own);
    WriteUsed(zortos, UsedMagic, {{9, 0}}, 2);
    CHECK_FALSE(VideoCommon::MergeUsedPipelineLists(zortos, own, UsedMagic));
    WriteUsed(zortos, UsedMagic + 1, {{9, 0}}, 1);
    CHECK_FALSE(VideoCommon::MergeUsedPipelineLists(zortos, own, UsedMagic));
    CHECK(ReadAll(own) == before);

    // With no list of our own, the other one is taken whole.
    const auto missing = temp.path / "missing_used.bin";
    WriteUsed(zortos, UsedMagic, {{9, 0}}, 1);
    CHECK(VideoCommon::MergeUsedPipelineLists(zortos, missing, UsedMagic) == 1);
    CHECK(ReadUsed(missing) == std::unordered_map<u64, u8>{{9, 0}});
}

TEST_CASE("The merge log records each source's size and time", "[pipeline_cache_files]") {
    TempDir temp;
    const auto source = temp.path / "vulkan.fork-v54.bin";
    const auto log_file = temp.path / "vulkan-frizti.fork-v52.bin.merged";
    WriteCache(source, Magic, 54, Record(1, "a"));
    const auto stamp = VideoCommon::StampPipelineCacheFile(source);
    REQUIRE(stamp);
    CHECK(stamp->size == ReadAll(source).size());
    CHECK_FALSE(VideoCommon::StampPipelineCacheFile(temp.path / "missing.bin"));

    VideoCommon::PipelineMergeLog log;
    log.emplace("vulkan.fork-v54.bin", *stamp);
    log.emplace("vulkan_used.fork-v54.bin", VideoCommon::PipelineCacheFileStamp{9, -3});
    REQUIRE(VideoCommon::WritePipelineMergeLog(log_file, log));
    CHECK(VideoCommon::ReadPipelineMergeLog(log_file) == log);
    CHECK(VideoCommon::ReadPipelineMergeLog(temp.path / "missing.merged").empty());

    WriteCache(source, Magic, 54, Record(1, "a") + Record(2, "b"));
    CHECK(VideoCommon::StampPipelineCacheFile(source) != stamp);
}

TEST_CASE("Pipeline cache names are version-suffixed per build", "[pipeline_cache_files]") {
    const std::filesystem::path dir{"shader"};
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 39).filename() ==
          "vulkan.fork-v39.bin");
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan_pipelines", 39).filename() ==
          "vulkan_pipelines.fork-v39.bin");
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 18) !=
          VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 39));
}

TEST_CASE("A legacy cache of this version is adopted with its contents",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 39, "pipelines");
    const std::string before = ReadAll(legacy);
    REQUIRE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK_FALSE(std::filesystem::exists(legacy));
    CHECK(ReadAll(versioned) == before);
    // Idempotent: nothing left to adopt.
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
}

TEST_CASE("Another build's legacy cache is never touched", "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 18, "official");
    const std::string before = ReadAll(legacy);
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(ReadAll(legacy) == before);
    CHECK_FALSE(std::filesystem::exists(versioned));

    WriteCache(legacy, OtherMagic, 39, "wrong magic");
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(std::filesystem::exists(legacy));

    // Truncated header.
    {
        std::ofstream stream(legacy, std::ios::binary | std::ios::trunc);
        stream.write(Magic.data(), 4);
    }
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(std::filesystem::exists(legacy));
}

TEST_CASE("An existing versioned cache is never overwritten by a legacy one",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 39, "older");
    WriteCache(versioned, Magic, 39, "current");
    const std::string current = ReadAll(versioned);
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(ReadAll(versioned) == current);
    CHECK(std::filesystem::exists(legacy));
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(temp.path / "missing.bin", versioned,
                                                      Magic, 39));
}

TEST_CASE("Cached compute failures retain their cause and the current dispatch address",
          "[pipeline_cache][diagnostics]") {
    const Vulkan::ComputePipelineCacheKey key{
        .unique_hash = 0x1234,
        .shared_memory_size = 1024,
        .workgroup_size = {32, 1, 1},
    };
    std::unordered_map<Vulkan::ComputePipelineCacheKey, Vulkan::ComputePipelineCacheEntry> cache;
    const std::string cause =
        "SPIR-V emission failed at IR opcode UndefU8: SPIR-V Instruction is not implemented";
    cache[key].failure = cause;
    for (const GPUVAddr address : {0x211ddd9000ULL, 0x21ebdfa000ULL}) {
        const auto [entry, inserted] = cache.try_emplace(key);
        REQUIRE_FALSE(inserted);
        CHECK(entry->second.Get(false, address, key) == nullptr);
        try {
            static_cast<void>(entry->second.Get(true, address, key));
            FAIL("Native compute failure was not propagated");
        } catch (const Shader::RuntimeError& exception) {
            const std::string message{exception.what()};
            CHECK(message.find(cause) != std::string::npos);
            CHECK(message.find(fmt::format("address={:#x}", address)) != std::string::npos);
            CHECK(message.find(fmt::format("shader={:#016x}", key.unique_hash)) != std::string::npos);
            CHECK(message.find(fmt::format("pipeline={:#016x}", key.Hash())) != std::string::npos);
        }
        CHECK(entry->second.failure == cause);
        CHECK_FALSE(entry->second.pipeline);
    }
    auto other = key;
    other.workgroup_size[0] = 64;
    const auto [entry, inserted] = cache.try_emplace(other);
    REQUIRE(inserted);
    CHECK(entry->second.failure.empty());
    CHECK(entry->second.Get(false, 0x2120633c00, other) == nullptr);
    REQUIRE_THROWS_AS(entry->second.Get(true, 0x2120633c00, other), Shader::RuntimeError);
}
