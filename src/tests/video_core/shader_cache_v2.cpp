// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <boost/crc.hpp>
#include <catch2/catch_test_macros.hpp>

#ifndef _WIN32
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "common/cityhash.h"
#include "common/fs/file_lock.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/program_header.h"
#include "video_core/native_lowering.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_records.h"
#include "video_core/renderer_vulkan/vk_shader_cache_v2.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/cache_writer.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_jobs.h"
#include "video_core/shader_warm/warm_records.h"

namespace {

namespace V2 = VideoCommon::ShaderCacheV2;
using V2::RecordKind;
using Shader::Ampere::Instruction;
using namespace std::chrono_literals;

constexpr u32 ReadProperties{0x15};
constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, 0}};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-shader-cache-v2-" + std::to_string(stamp));
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

struct Written {
    RecordKind kind;
    u64 key;
    std::vector<u8> payload;

    bool operator==(const Written&) const = default;
};

std::vector<Written> Live(const V2::ParsedLog& log) {
    std::vector<Written> live;
    for (const V2::LogRecord& record : log.live) {
        live.push_back({record.kind, record.key, {record.payload.begin(), record.payload.end()}});
    }
    return live;
}

/// A payload that tells its index and differs in size between records
std::vector<u8> TestPayload(u64 index) {
    std::vector<u8> payload(static_cast<size_t>(1 + (index * 37) % 211));
    for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<u8>(index * 131 + i * 7);
    }
    return payload;
}

/// A log of records with distinct keys, and where each record begins and ends
struct TestLog {
    std::vector<u8> bytes;
    std::vector<Written> records;
    std::vector<std::pair<u64, u64>> ranges;
};

TestLog MakeLog(u64 count) {
    TestLog log;
    const auto header = V2::EncodeFileHeader(1234);
    log.bytes.assign(header.begin(), header.end());
    for (u64 i = 0; i < count; ++i) {
        const RecordKind kind = i % 3 == 0 ? RecordKind::Graphics : RecordKind::Compute;
        log.records.push_back({kind, 0x1000 + i, TestPayload(i)});
        const u64 begin = log.bytes.size();
        V2::EncodeRecord(log.bytes, kind, 0x1000 + i, log.records.back().payload);
        log.ranges.emplace_back(begin, log.bytes.size());
    }
    return log;
}

class TestEnvironment final : public VideoCommon::GenericEnvironment {
public:
    explicit TestEnvironment(Shader::Stage shader_stage, u64 seed = 0) {
        stage = shader_stage;
        code = {0x123456789abcdef0 ^ seed, 0xfedcba9876543210};
        cached_lowest = 0;
        cached_highest = sizeof(u64);
        texture_bound = 2;
        cbuf_values.emplace((u64{1} << 32) | 16, 0x1357);
        texture_types.emplace(3, Shader::TextureType::Color2D);
        if (shader_stage == Shader::Stage::Compute) {
            workgroup_size = {8, 4, 1};
            shared_memory_size = 0x400;
        }
    }

    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return {};
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
};

/// An environment of a lowered native program's code
class NativeTestEnvironment final : public VideoCommon::GenericEnvironment {
public:
    NativeTestEnvironment(Shader::Stage shader_stage, std::span<const u64> lowered_code,
                          u32 texture_bound_) {
        stage = shader_stage;
        code.assign(lowered_code.begin(), lowered_code.end());
        cached_lowest = 0;
        cached_highest = static_cast<u32>((code.size() - 1) * sizeof(u64));
        texture_bound = texture_bound_;
        cbuf_values.emplace((u64{1} << 32) | 16, 0x1357);
        if (shader_stage == Shader::Stage::Compute) {
            workgroup_size = {8, 4, 1};
            shared_memory_size = 0x400;
        }
    }

    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return {};
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
};

Vulkan::GraphicsPipelineCacheKey TestGraphicsKey(u64 shader, u32 driver_version) {
    Vulkan::GraphicsPipelineCacheKey key{};
    key.unique_hashes[4] = shader;
    key.state.driver_id = 4;
    key.state.driver_version = driver_version;
    key.state.alpha_test_ref = 0x3f800000;
    return key;
}

Vulkan::ComputePipelineCacheKey TestComputeKey(u64 shader) {
    Vulkan::ComputePipelineCacheKey key{};
    key.unique_hash = shader;
    key.shared_memory_size = 0x400;
    key.workgroup_size = {8, 4, 1};
    return key;
}

template <typename Key>
std::span<const char> KeyBytes(const Key& key) {
    return {reinterpret_cast<const char*>(&key), sizeof(key)};
}

/// The keys of the pipelines a payload holds, as the pipeline cache reads them
struct ReadPipeline {
    bool compute{};
    std::vector<Shader::Stage> stages;
    std::vector<u8> key;
};

ReadPipeline ReadPayload(std::span<const u8> payload) {
    ReadPipeline read;
    Common::UniqueFunction<void, std::istream&, VideoCommon::FileEnvironment> compute{
        [&](std::istream& file, VideoCommon::FileEnvironment env) {
            read.compute = true;
            read.stages.push_back(env.ShaderStage());
            read.key.resize(sizeof(Vulkan::ComputePipelineCacheKey));
            file.read(reinterpret_cast<char*>(read.key.data()),
                      static_cast<std::streamsize>(read.key.size()));
        }};
    Common::UniqueFunction<void, std::istream&, std::vector<VideoCommon::FileEnvironment>>
        graphics{[&](std::istream& file, std::vector<VideoCommon::FileEnvironment> envs) {
            for (const auto& env : envs) {
                read.stages.push_back(env.ShaderStage());
            }
            read.key.resize(sizeof(Vulkan::GraphicsPipelineCacheKey));
            file.read(reinterpret_cast<char*>(read.key.data()),
                      static_cast<std::streamsize>(read.key.size()));
        }};
    VideoCommon::ReadPipelinePayload(payload, compute, graphics);
    return read;
}

V2::ProgramRecord LowerTestProgram(std::span<const Instruction> native, bool geometry,
                                   u64 address) {
    const VideoCommon::NativeLoweringInputs inputs{.geometry = geometry,
                                                   .native_code_address = address};
    auto lowered = VideoCommon::LowerNativeProgram(native, inputs);
    REQUIRE(static_cast<bool>(lowered));
    Shader::ProgramHeader sph{};
    std::memset(&sph, 0x5a, sizeof(sph));
    const auto code = VideoCommon::MakeLoweredNativeCode(
        native, sph, 128, VideoCommon::NativeTrapOffsets(native), lowered);
    return V2::MakeProgramRecord(geometry ? Shader::Stage::Geometry : Shader::Stage::VertexB, 128,
                                 sph, native, inputs, *code);
}

constexpr std::array<Instruction, 3> PlainProgram{{
    {{0x06007947, 0x10 | 3, 0x0b800000, 0}}, // BRA.CONV to the second EXIT
    EXIT,
    EXIT,
}};

constexpr std::array<Instruction, 3> ProgramCounterProgram{{
    {{0x00007918, 0, 0, 0}}, // NOP
    {{0x0014734e, 0, 0, 0}}, // LEPC R20
    EXIT,
}};

} // Anonymous namespace

TEST_CASE("Shader cache v2: every kind of record reads back as written", "[video_core][shader_cache_v2]") {
    TestEnvironment fragment{Shader::Stage::Fragment};
    TestEnvironment compute{Shader::Stage::Compute};
    const std::array<const VideoCommon::GenericEnvironment*, 1> graphics_envs{&fragment};
    const std::array<const VideoCommon::GenericEnvironment*, 1> compute_envs{&compute};
    const auto graphics_key = TestGraphicsKey(0x1111, 0);
    const auto compute_key = TestComputeKey(0x2222);
    const auto graphics_payload =
        VideoCommon::SerializePipelinePayload(KeyBytes(graphics_key), graphics_envs, ReadProperties);
    const auto compute_payload =
        VideoCommon::SerializePipelinePayload(KeyBytes(compute_key), compute_envs, ReadProperties);
    REQUIRE_FALSE(graphics_payload.empty());
    REQUIRE_FALSE(compute_payload.empty());

    const V2::ProgramRecord program = LowerTestProgram(PlainProgram, false, 0x10000);
    const V2::UsedRecord used{.session_time = 77, .pipelines = {1, 2, 3}};
    const V2::MigratedRecord migrated{.source = "vulkan.fork-v54.bin",
                                      .source_size = 99,
                                      .source_mtime = -5,
                                      .source_version = 54,
                                      .graphics = 1,
                                      .compute = 2,
                                      .used = 3};
    const std::vector<Written> written{
        {RecordKind::Program, V2::ProgramKey(program), V2::EncodeProgram(program)},
        {RecordKind::Graphics, Vulkan::GraphicsRecordKey(graphics_key), graphics_payload},
        {RecordKind::Compute, Vulkan::ComputeRecordKey(compute_payload), compute_payload},
        {RecordKind::Used, 5, V2::EncodeUsed(used)},
        {RecordKind::Migrated, 6, V2::EncodeMigrated(migrated)},
        {RecordKind::Tombstone, 0x999, V2::TombstonePayload(RecordKind::Graphics)},
        {RecordKind::Graphics, 0x40, {}},
    };

    TempDir temp;
    const auto file = temp.path / "cache.bin";
    {
        V2::CacheWriter writer{file};
        for (const Written& record : written) {
            writer.Append(record.kind, record.key, record.payload);
        }
        REQUIRE(writer.Drain(5s));
        REQUIRE(writer.Stats().written_records == written.size());
    }
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.header == V2::HeaderStatus::Valid);
    REQUIRE(log.parsed.stats.records == written.size());
    REQUIRE(log.parsed.stats.tombstones == 1);
    REQUIRE(log.parsed.stats.bad_headers == 0);
    REQUIRE(log.parsed.stats.bad_payloads == 0);
    std::vector<Written> expected = written;
    std::erase_if(expected, [](const Written& record) { return record.kind == RecordKind::Tombstone; });
    REQUIRE(Live(log.parsed) == expected);
    REQUIRE(log.parsed.stats.dead_bytes == V2::RecordSize(sizeof(u16)));

    const auto decoded = V2::DecodeProgram(log.parsed.live[0].payload);
    REQUIRE(decoded.has_value());
    REQUIRE(V2::EncodeProgram(*decoded) == V2::EncodeProgram(program));
    REQUIRE(V2::ProgramStamp(log.parsed.live[0].payload) == V2::LoweringStamp());
    REQUIRE(V2::DecodeUsed(log.parsed.live[3].payload) == used);
    REQUIRE(V2::DecodeMigrated(log.parsed.live[4].payload) == migrated);

    const ReadPipeline graphics = ReadPayload(log.parsed.live[1].payload);
    REQUIRE_FALSE(graphics.compute);
    REQUIRE(graphics.stages == std::vector{Shader::Stage::Fragment});
    REQUIRE(std::memcmp(graphics.key.data(), &graphics_key, sizeof(graphics_key)) == 0);
    const ReadPipeline computed = ReadPayload(log.parsed.live[2].payload);
    REQUIRE(computed.compute);
    REQUIRE(std::memcmp(computed.key.data(), &compute_key, sizeof(compute_key)) == 0);

    // A payload cut short, or with bytes after its key, is damaged.
    auto cut = graphics_payload;
    cut.pop_back();
    REQUIRE_THROWS_AS(ReadPayload(cut), std::ios_base::failure);
    auto longer = graphics_payload;
    longer.push_back(0);
    REQUIRE_THROWS_AS(ReadPayload(longer), std::ios_base::failure);
    // Decoders refuse what isn't theirs.
    REQUIRE_FALSE(V2::DecodeProgram(graphics_payload).has_value());
    REQUIRE_FALSE(V2::DecodeUsed(std::vector<u8>(15)).has_value());
}

TEST_CASE("Shader cache v2: a record cut off anywhere at the end is ignored alone",
          "[video_core][shader_cache_v2]") {
    const TestLog log = MakeLog(9);
    const auto [begin, end] = log.ranges.back();
    const std::vector<Written> before(log.records.begin(), log.records.end() - 1);
    for (u64 size = begin; size < end; ++size) {
        const V2::ParsedLog parsed = V2::ParseLog(std::span{log.bytes}.first(size));
        INFO("size " << size);
        REQUIRE(parsed.header == V2::HeaderStatus::Valid);
        REQUIRE(parsed.stats.bad_headers == 0);
        REQUIRE(parsed.stats.bad_payloads == 0);
        const u64 framed = V2::RecordHeaderSize + log.records.back().payload.size() + V2::RecordHashSize;
        if (size - begin >= framed) {
            // Only padding is missing
            REQUIRE(Live(parsed) == log.records);
        } else {
            REQUIRE(Live(parsed) == before);
            REQUIRE(parsed.stats.torn_tail_bytes == size - begin);
        }
    }
    REQUIRE(Live(V2::ParseLog(log.bytes)) == log.records);
    // A file cut off within its header holds nothing.
    for (u64 size = 1; size < V2::FileHeaderSize; ++size) {
        REQUIRE(V2::ParseLog(std::span{log.bytes}.first(size)).live.empty());
    }
}

TEST_CASE("Shader cache v2: a flipped byte loses only the record it is in",
          "[video_core][shader_cache_v2]") {
    const TestLog log = MakeLog(24);
    std::mt19937_64 random{0x5eed};
    for (int round = 0; round < 3000; ++round) {
        std::vector<u8> bytes = log.bytes;
        const int flips = round < 2000 ? 1 : 1 + static_cast<int>(random() % 4);
        std::set<size_t> touched;
        for (int flip = 0; flip < flips; ++flip) {
            const size_t offset = random() % bytes.size();
            bytes[offset] ^= static_cast<u8>(1 + random() % 255);
            for (size_t i = 0; i < log.ranges.size(); ++i) {
                if (offset >= log.ranges[i].first && offset < log.ranges[i].second) {
                    touched.insert(i);
                }
            }
        }
        const std::vector<Written> live = Live(V2::ParseLog(bytes));
        INFO("round " << round);
        for (const Written& record : live) {
            // Never a record that wasn't written
            REQUIRE(std::ranges::find(log.records, record) != log.records.end());
        }
        for (size_t i = 0; i < log.records.size(); ++i) {
            if (!touched.contains(i)) {
                REQUIRE(std::ranges::find(live, log.records[i]) != live.end());
            }
        }
    }
}

TEST_CASE("Shader cache v2: reading resumes at the record after a damaged header",
          "[video_core][shader_cache_v2]") {
    const TestLog log = MakeLog(6);
    for (const u64 field : {0u, 5u, 9u, 13u, 21u}) {
        std::vector<u8> bytes = log.bytes;
        bytes[log.ranges[2].first + field] ^= 0x40;
        const V2::ParsedLog parsed = V2::ParseLog(bytes);
        std::vector<Written> expected = log.records;
        expected.erase(expected.begin() + 2);
        REQUIRE(Live(parsed) == expected);
        REQUIRE(parsed.stats.bad_headers == 1);
        REQUIRE(parsed.stats.skipped_bytes == log.ranges[2].second - log.ranges[2].first);
    }
    // A damaged payload drops its record alone.
    std::vector<u8> bytes = log.bytes;
    bytes[log.ranges[4].first + V2::RecordHeaderSize] ^= 1;
    const V2::ParsedLog parsed = V2::ParseLog(bytes);
    REQUIRE(parsed.stats.bad_payloads == 1);
    REQUIRE(parsed.live.size() == log.records.size() - 1);
    // A damaged file header loses no record.
    bytes = log.bytes;
    bytes[17] ^= 1;
    const V2::ParsedLog damaged_header = V2::ParseLog(bytes);
    REQUIRE(damaged_header.header == V2::HeaderStatus::Damaged);
    REQUIRE(Live(damaged_header) == log.records);
}

TEST_CASE("Shader cache v2: the last record of a kind and key wins, tombstones remove",
          "[video_core][shader_cache_v2]") {
    const auto header = V2::EncodeFileHeader(1);
    std::vector<u8> bytes(header.begin(), header.end());
    const auto add = [&](RecordKind kind, u64 key, std::vector<u8> payload) {
        V2::EncodeRecord(bytes, kind, key, payload);
    };
    add(RecordKind::Graphics, 1, {1});
    add(RecordKind::Compute, 1, {2});
    add(RecordKind::Graphics, 1, {3});
    add(static_cast<RecordKind>(99), 1, {4}); // A kind of a later build
    add(RecordKind::Tombstone, 1, V2::TombstonePayload(RecordKind::Compute));
    add(RecordKind::Tombstone, 7, V2::TombstonePayload(RecordKind::Compute)); // Nothing to remove
    add(RecordKind::Used, 2, {5});
    add(RecordKind::Tombstone, 2, V2::TombstonePayload(RecordKind::Used));
    add(RecordKind::Used, 2, {6});
    add(static_cast<RecordKind>(99), 1, {7});
    const V2::ParsedLog parsed = V2::ParseLog(bytes);
    REQUIRE(Live(parsed) == std::vector<Written>{
                                {RecordKind::Graphics, 1, {3}},
                                {RecordKind::Used, 2, {6}},
                                {static_cast<RecordKind>(99), 1, {7}},
                            });
    REQUIRE(parsed.stats.superseded == 4);
    REQUIRE(parsed.stats.tombstones == 3);
}

TEST_CASE("Shader cache v2: compaction keeps the live records and only them",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    std::vector<Written> expected;
    {
        V2::CacheWriter writer{file};
        for (u64 i = 0; i < 200; ++i) {
            writer.Append(RecordKind::Graphics, i % 50, TestPayload(i));
            if (i % 7 == 0) {
                writer.Append(RecordKind::Tombstone, (i + 3) % 50,
                              V2::TombstonePayload(RecordKind::Graphics));
            }
        }
        REQUIRE(writer.Drain(5s));
    }
    const V2::LogFile before = V2::ReadLogFile(file);
    REQUIRE(before.parsed.stats.dead_bytes > before.bytes.size() / 2);
    expected = Live(before.parsed);

    {
        // Never while another emulator holds the lock
        const Common::FS::FileLock other{VideoCommon::PipelineCacheLockPath(file)};
        REQUIRE(other.Held());
        REQUIRE(V2::CompactLog(file) == V2::WriteResult::Busy);
        REQUIRE(ReadAll(file) == before.bytes);
    }
    REQUIRE(V2::CompactLog(file) == V2::WriteResult::Written);
    const V2::LogFile after = V2::ReadLogFile(file);
    REQUIRE(Live(after.parsed) == expected);
    REQUIRE(after.parsed.stats.dead_bytes == 0);
    REQUIRE(after.parsed.created_unix == before.parsed.created_unix);
    REQUIRE(after.bytes.size() < before.bytes.size());
    REQUIRE_FALSE(std::filesystem::exists(temp.path / "cache.bin.tmp"));

    // Through the writer, after the records queued before it
    {
        V2::CacheWriter writer{file};
        writer.Append(RecordKind::Graphics, 0, std::vector<u8>{9});
        writer.Compact();
        REQUIRE(writer.Drain(5s));
        for (int i = 0; i < 500 && writer.Stats().compactions == 0; ++i) {
            std::this_thread::sleep_for(10ms);
        }
        REQUIRE(writer.Stats().compactions == 1);
    }
    const V2::LogFile compacted = V2::ReadLogFile(file);
    REQUIRE(compacted.parsed.stats.dead_bytes == 0);
    REQUIRE(compacted.parsed.live.back().key == 0);
    REQUIRE(compacted.parsed.live.size() == expected.size());
}

TEST_CASE("Shader cache v2: a record reads again at its offset until a compaction moves it",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    {
        V2::CacheWriter writer{file};
        for (u64 i = 0; i < 20; ++i) {
            writer.Append(RecordKind::Graphics, i, TestPayload(i));
        }
        writer.Append(RecordKind::Tombstone, 3, V2::TombstonePayload(RecordKind::Graphics));
        REQUIRE(writer.Drain(5s));
    }
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.live.size() == 19);
    for (const V2::LogRecord& record : log.parsed.live) {
        const auto payload = V2::ReadRecordPayload(file, record.kind, record.key, record.offset,
                                                   record.payload.size());
        REQUIRE(payload);
        REQUIRE(std::ranges::equal(*payload, record.payload));
    }
    const V2::LogRecord& last = log.parsed.live.back();
    const u64 size = last.payload.size();
    REQUIRE_FALSE(V2::ReadRecordPayload(file, RecordKind::Compute, last.key, last.offset, size));
    REQUIRE_FALSE(V2::ReadRecordPayload(file, last.kind, last.key + 1, last.offset, size));
    REQUIRE_FALSE(V2::ReadRecordPayload(file, last.kind, last.key, last.offset, size - 1));
    REQUIRE_FALSE(V2::ReadRecordPayload(file, last.kind, last.key, last.offset + 8, size));
    REQUIRE_FALSE(V2::ReadRecordPayload(file, last.kind, last.key, log.bytes.size(), size));

    REQUIRE(V2::CompactLog(file) == V2::WriteResult::Written);
    REQUIRE_FALSE(V2::ReadRecordPayload(file, last.kind, last.key, last.offset, size));
}

TEST_CASE("Shader cache v2: two writers on one file lose no record",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    constexpr u64 Threads = 4;
    constexpr u64 PerThread = 400;
    const V2::WriterOptions options{.idle = 1ms, .interval = 5ms, .lock_wait = 2ms, .retry = 3ms};
    V2::WriterStats first_stats;
    {
        V2::CacheWriter first{file, options};
        V2::CacheWriter second{file, options};
        std::vector<std::jthread> producers;
        // Another emulator holding the lock for a while: the records wait.
        std::optional<Common::FS::FileLock> other{VideoCommon::PipelineCacheLockPath(file)};
        REQUIRE(other->Held());
        for (u64 thread = 0; thread < Threads * 2; ++thread) {
            producers.emplace_back([&, thread] {
                V2::CacheWriter& writer = thread % 2 == 0 ? first : second;
                for (u64 i = 0; i < PerThread; ++i) {
                    const u64 key = thread * PerThread + i;
                    writer.Append(RecordKind::Compute, key, TestPayload(key));
                    if (i % 64 == 0) {
                        std::this_thread::sleep_for(1ms);
                    }
                }
            });
        }
        std::this_thread::sleep_for(100ms);
        REQUIRE(first.Stats().written_records == 0);
        other.reset();
        producers.clear();
        REQUIRE(first.Drain(10s));
        REQUIRE(second.Drain(10s));
        first_stats = first.Stats();
    }
    REQUIRE(first_stats.busy > 0);
    REQUIRE(first_stats.dropped == 0);
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.stats.bad_headers == 0);
    REQUIRE(log.parsed.stats.bad_payloads == 0);
    REQUIRE(log.parsed.stats.torn_tail_bytes == 0);
    REQUIRE(log.parsed.live.size() == Threads * 2 * PerThread);
    std::vector<u64> last(Threads * 2, 0);
    for (const V2::LogRecord& record : log.parsed.live) {
        const auto payload = TestPayload(record.key);
        REQUIRE(std::ranges::equal(record.payload, payload));
        // In each producer's order
        const u64 thread = record.key / PerThread;
        REQUIRE(record.key >= last[thread]);
        last[thread] = record.key;
    }
}

TEST_CASE("Shader cache v2: appends follow a record cut off by a crash",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const TestLog log = MakeLog(5);
    // Cut off in the middle of the last record's payload, off an 8-byte boundary
    const u64 cut = log.ranges.back().first + V2::RecordHeaderSize + 3;
    WriteAll(file, std::span{log.bytes}.first(cut));
    std::vector<u8> records;
    V2::EncodeRecord(records, RecordKind::Used, 0x77, TestPayload(1));
    V2::EncodeRecord(records, RecordKind::Used, 0x78, TestPayload(2));
    REQUIRE(V2::AppendToLog(file, records) == V2::WriteResult::Written);
    const V2::LogFile read = V2::ReadLogFile(file);
    std::vector<Written> expected(log.records.begin(), log.records.end() - 1);
    expected.push_back({RecordKind::Used, 0x77, TestPayload(1)});
    expected.push_back({RecordKind::Used, 0x78, TestPayload(2)});
    REQUIRE(Live(read.parsed) == expected);

    // A file of another format version is never written to.
    auto other = V2::EncodeFileHeader(5);
    other[8] = 3;
    REQUIRE(V2::ParseLog(other).header == V2::HeaderStatus::Damaged);
    boost::crc_32_type crc;
    crc.process_bytes(other.data(), 24);
    const u32 check = crc.checksum();
    std::memcpy(other.data() + 24, &check, sizeof(check));
    REQUIRE(V2::ParseLog(other).header == V2::HeaderStatus::OtherVersion);
    const auto other_file = temp.path / "other.bin";
    WriteAll(other_file, other);
    REQUIRE(V2::AppendToLog(other_file, records) == V2::WriteResult::OtherVersion);
    REQUIRE(ReadAll(other_file) == std::vector<u8>(other.begin(), other.end()));
}

TEST_CASE("Shader cache v2: a log is built record by record", "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const std::array written{Written{RecordKind::Graphics, 0x11, TestPayload(1)},
                             Written{RecordKind::Used, 0x22, TestPayload(2)}};

    V2::LogBuilder builder;
    REQUIRE(builder.Start(file) == V2::WriteResult::Written);
    for (const Written& record : written) {
        REQUIRE(builder.Append(record.kind, record.key, record.payload));
    }
    // Nothing is there before the commit: a reader never sees half a log
    REQUIRE_FALSE(std::filesystem::exists(file));
    REQUIRE(builder.Commit() == V2::WriteResult::Written);
    REQUIRE(Live(V2::ReadLogFile(file).parsed) ==
            std::vector<Written>(written.begin(), written.end()));
    REQUIRE_FALSE(std::filesystem::exists(temp.path / "cache.bin.migrating"));

    // A log that is there is never written over
    const auto bytes = ReadAll(file);
    V2::LogBuilder second;
    REQUIRE(second.Start(file) == V2::WriteResult::Exists);
    REQUIRE(ReadAll(file) == bytes);

    // One that is given up on leaves the log as it was
    const auto other = temp.path / "other.bin";
    V2::LogBuilder discarded;
    REQUIRE(discarded.Start(other) == V2::WriteResult::Written);
    REQUIRE(discarded.Append(RecordKind::Used, 0x33, TestPayload(3)));
    discarded.Discard();
    REQUIRE_FALSE(std::filesystem::exists(other));
    REQUIRE_FALSE(std::filesystem::exists(temp.path / "other.bin.migrating"));

    // Enough records that the buffer is written out more than once: the whole point of the
    // builder, since a source of this size used to be held in memory before it was written
    const auto large_file = temp.path / "large.bin";
    V2::LogBuilder large;
    REQUIRE(large.Start(large_file) == V2::WriteResult::Written);
    std::vector<Written> many;
    for (u64 i = 0; i < 40; ++i) {
        std::vector<u8> payload(256 * 1024, static_cast<u8>(i));
        REQUIRE(large.Append(RecordKind::Used, 0x100 + i, payload));
        many.push_back({RecordKind::Used, 0x100 + i, std::move(payload)});
    }
    REQUIRE(large.Commit() == V2::WriteResult::Written);
    const V2::LogFile large_log = V2::ReadLogFile(large_file);
    REQUIRE(Live(large_log.parsed) == many);
    REQUIRE(large_log.parsed.stats.dead_bytes == 0);
}

#ifndef _WIN32
TEST_CASE("Shader cache v2: a writer killed at any point leaves complete records",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    std::mt19937 random{1234};
    size_t most_records = 0;
    for (int run = 0; run < 24; ++run) {
        const auto file = temp.path / ("killed-" + std::to_string(run) + ".bin");
        const pid_t child = fork();
        REQUIRE(child >= 0);
        if (child == 0) {
            {
                V2::CacheWriter writer{file, {.idle = 0ms, .interval = 0ms, .lock_wait = 1ms,
                                              .retry = 1ms}};
                for (u64 i = 0;; ++i) {
                    writer.Append(RecordKind::Compute, i, TestPayload(i));
                    if (i % 16 == 0) {
                        writer.Drain(1s);
                    }
                }
            }
            _exit(0);
        }
        std::this_thread::sleep_for(std::chrono::microseconds{2000 + random() % 40000});
        kill(child, SIGKILL);
        int status{};
        waitpid(child, &status, 0);
        const V2::LogFile log = V2::ReadLogFile(file);
        INFO("run " << run << ", " << log.bytes.size() << " bytes");
        REQUIRE(log.parsed.stats.bad_headers == 0);
        REQUIRE(log.parsed.stats.bad_payloads == 0);
        for (u64 i = 0; i < log.parsed.live.size(); ++i) {
            REQUIRE(log.parsed.live[i].key == i);
            REQUIRE(std::ranges::equal(log.parsed.live[i].payload, TestPayload(i)));
        }
        most_records = std::max(most_records, log.parsed.live.size());
    }
    REQUIRE(most_records > 100);
}
#endif

TEST_CASE("Shader cache v2: a program record lowers again to the code it holds",
          "[video_core][shader_cache_v2]") {
    const V2::ProgramRecord plain = LowerTestProgram(PlainProgram, false, 0x2000);
    REQUIRE(plain.native_code_address == 0); // Not part of the code
    REQUIRE(plain.lowering_stamp == V2::LoweringStamp());
    REQUIRE(plain.unique_hash == VideoCommon::NativeShaderHash(plain.lowered_code));
    const auto relowered = V2::RelowerProgram(plain);
    REQUIRE(relowered != nullptr);
    REQUIRE(relowered->code == plain.lowered_code);
    REQUIRE(relowered->texture_bound == plain.texture_bound);
    REQUIRE(relowered->source_size == plain.source_size);
    // The address of the code doesn't change the key of such a program...
    REQUIRE(V2::ProgramKey(LowerTestProgram(PlainProgram, false, 0x9000)) == V2::ProgramKey(plain));

    // ...but it is part of the code and the key of one loading the program counter.
    REQUIRE(VideoCommon::LoadsProgramCounter(ProgramCounterProgram, true));
    REQUIRE_FALSE(VideoCommon::LoadsProgramCounter(ProgramCounterProgram, false));
    const V2::ProgramRecord loads = LowerTestProgram(ProgramCounterProgram, true, 0x12fffffff0);
    REQUIRE(loads.native_code_address == 0x12fffffff0);
    const auto relowered_loads = V2::RelowerProgram(loads);
    REQUIRE(relowered_loads != nullptr);
    REQUIRE(relowered_loads->code == loads.lowered_code);
    const V2::ProgramRecord elsewhere = LowerTestProgram(ProgramCounterProgram, true, 0x8000);
    REQUIRE(elsewhere.lowered_code != loads.lowered_code);
    REQUIRE(V2::ProgramKey(elsewhere) != V2::ProgramKey(loads));

    // A program that doesn't lower again gives nothing.
    V2::ProgramRecord broken = plain;
    broken.native = {Instruction{{0xffffffff, 0, 0, 0}}};
    REQUIRE(V2::RelowerProgram(broken) == nullptr);
}

TEST_CASE("Shader cache v2: a program is recorded once per lowering", "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const V2::ProgramRecord program = LowerTestProgram(PlainProgram, false, 0);
    {
        V2::Cache cache{file};
        cache.RecordProgram(program);
        cache.RecordProgram(program);
        V2::ProgramRecord relowered = program;
        relowered.lowering_stamp ^= 1;
        cache.RecordProgram(relowered);
        cache.RecordProgram(relowered);
        // One the log held already
        const V2::ProgramRecord other = LowerTestProgram(ProgramCounterProgram, true, 0x40);
        cache.AddProgram(V2::ProgramKey(other), other.lowering_stamp);
        cache.RecordProgram(other);
        REQUIRE(cache.Writer().Drain(5s));
        REQUIRE(cache.Writer().Stats().written_records == 2);
    }
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.live.size() == 1);
    REQUIRE(V2::ProgramStamp(log.parsed.live[0].payload) == (program.lowering_stamp ^ 1));
}

TEST_CASE("Shader cache v2: the precache lowers as the emulator does", "[video_core][shader_cache_v2]") {
    // The layout the precache wrote before it called the emulator's lowering
    const auto lowered = Shader::Ampere::LowerToMaxwell(PlainProgram, false, true, {},
                                                        Shader::Ampere::TessellationStage::None);
    REQUIRE(static_cast<bool>(lowered));
    std::vector<u64> words(10, 0);
    for (const u64 instruction : lowered.instructions) {
        if (words.size() % 4 == 0) {
            words.push_back(0);
        }
        words.push_back(instruction);
    }
    words.push_back(0);
    for (const u32 i : lowered.source_instruction_offsets) {
        words.push_back(0x50 + u64{i} * 8 + ((u64{i} + 1) / 3) * 8);
    }
    words.push_back(lowered.source_instruction_offsets.size());
    words.push_back(0x50414d4e4544454EULL);

    VideoCommon::ShaderWarm::LoweredCode out;
    const std::span<const u8> program{reinterpret_cast<const u8*>(PlainProgram.data()),
                                      sizeof(PlainProgram)};
    REQUIRE_FALSE(VideoCommon::ShaderWarm::LowerProgram(program, out).has_value());
    REQUIRE(out.code.size() == words.size() * sizeof(u64));
    REQUIRE(std::memcmp(out.code.data(), words.data(), out.code.size()) == 0);
    REQUIRE(out.texture_buffer == lowered.required_texture_buffer);
    REQUIRE(VideoCommon::ShaderWarm::StageHash(out.code) ==
            (Common::CityHash64(reinterpret_cast<const char*>(words.data()),
                                words.size() * sizeof(u64)) ^
             0x4f554e4345000002ULL));
}

TEST_CASE("Shader cache v2: a log parsed in parallel reads as one parsed in order",
          "[video_core][shader_cache_v2]") {
    TestLog log = MakeLog(1500);
    log.bytes[log.ranges[10].second - 9] ^= 0x40;   // A payload byte
    log.bytes[log.ranges[700].first + 13] ^= 0x01;  // A header's key
    log.bytes[log.ranges[1200].first + 30] ^= 0x80; // Another payload
    V2::EncodeRecord(log.bytes, RecordKind::Compute, 0x1000 + 1, TestPayload(7));
    V2::EncodeRecord(log.bytes, RecordKind::Tombstone, 0x1000 + 4,
                     V2::TombstonePayload(RecordKind::Compute));
    std::vector<u8> torn;
    V2::EncodeRecord(torn, RecordKind::Graphics, 0x9999, TestPayload(3));
    log.bytes.insert(log.bytes.end(), torn.begin(), torn.begin() + torn.size() / 2);

    const V2::ParsedLog in_order = V2::ParseLog(log.bytes);
    VideoCommon::ShaderJobScheduler jobs{{.cpu_threads = 4, .driver_threads = 1}};
    std::atomic<size_t> tasks{};
    const V2::ParsedLog parallel =
        V2::ParseLog(log.bytes, [&](size_t count, const std::function<void(size_t)>& task) {
            tasks += count;
            VideoCommon::ParallelFor(jobs, count, task);
        });
    // A task left out is checked by the walk itself.
    const V2::ParsedLog partial =
        V2::ParseLog(log.bytes, [](size_t count, const std::function<void(size_t)>& task) {
            for (size_t index = 0; index < count; index += 2) {
                task(index);
            }
        });

    REQUIRE(tasks > 1);
    REQUIRE(in_order.stats.bad_payloads == 2);
    REQUIRE(in_order.stats.bad_headers >= 1);
    REQUIRE(in_order.stats.superseded >= 2);
    REQUIRE(in_order.stats.torn_tail_bytes != 0);
    for (const V2::ParsedLog* other : {&parallel, &partial}) {
        REQUIRE(Live(*other) == Live(in_order));
        REQUIRE(other->header == in_order.header);
        const auto& a = other->stats;
        const auto& b = in_order.stats;
        REQUIRE(std::tie(a.file_size, a.records, a.live, a.superseded, a.tombstones, a.bad_headers,
                         a.bad_payloads, a.skipped_bytes, a.torn_tail_bytes, a.dead_bytes) ==
                std::tie(b.file_size, b.records, b.live, b.superseded, b.tombstones, b.bad_headers,
                         b.bad_payloads, b.skipped_bytes, b.torn_tail_bytes, b.dead_bytes));
    }
}

TEST_CASE("Shader cache v2: a program of this build's lowering keeps its code",
          "[video_core][shader_cache_v2]") {
    V2::ProgramRecord program = LowerTestProgram(PlainProgram, false, 0x2000);
    program.lowering_stamp ^= 1;
    const auto relowered = V2::RelowerProgramRecord(program);
    REQUIRE(relowered.has_value());
    REQUIRE(relowered->lowering_stamp == V2::LoweringStamp());
    REQUIRE(relowered->unique_hash == program.unique_hash);
    REQUIRE(relowered->lowered_code == program.lowered_code);

    // A migrated record without the native program stays as it is.
    V2::ProgramRecord migrated = program;
    migrated.native.clear();
    REQUIRE_FALSE(V2::RelowerProgramRecord(migrated).has_value());
}

TEST_CASE("Shader cache v2: pipeline records move to the code a program lowers to now",
          "[video_core][shader_cache_v2]") {
    const V2::ProgramRecord old_program = LowerTestProgram(PlainProgram, false, 0);
    V2::ProgramRecord new_program = LowerTestProgram(ProgramCounterProgram, true, 0x40);
    REQUIRE(new_program.unique_hash != old_program.unique_hash);
    new_program.texture_bound = 6;
    const V2::ProgramRecord unchanged = LowerTestProgram(ProgramCounterProgram, true, 0x80);

    const NativeTestEnvironment old_fragment{Shader::Stage::Fragment, old_program.lowered_code, 5};
    const NativeTestEnvironment new_fragment{Shader::Stage::Fragment, new_program.lowered_code, 6};
    const NativeTestEnvironment old_compute{Shader::Stage::Compute, old_program.lowered_code, 5};
    const NativeTestEnvironment new_compute{Shader::Stage::Compute, new_program.lowered_code, 6};
    const NativeTestEnvironment other_compute{Shader::Stage::Compute, unchanged.lowered_code, 5};
    const auto payload = [](const auto& key, const VideoCommon::GenericEnvironment& env) {
        const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&env};
        return VideoCommon::SerializePipelinePayload(KeyBytes(key), envs, ReadProperties);
    };
    auto graphics_key = TestGraphicsKey(old_program.unique_hash, 0);
    graphics_key.unique_hashes[4] ^= 0x77; // As the mesh state mixed into a hash
    const auto graphics = payload(graphics_key, old_fragment);
    const auto compute = payload(TestComputeKey(old_program.unique_hash), old_compute);
    const auto other = payload(TestComputeKey(unchanged.unique_hash), other_compute);

    std::vector<u8> bytes;
    const auto header = V2::EncodeFileHeader(1);
    bytes.assign(header.begin(), header.end());
    const u64 graphics_record = Vulkan::GraphicsRecordKey(graphics_key);
    const u64 compute_record = Vulkan::ComputeRecordKey(compute);
    V2::EncodeRecord(bytes, RecordKind::Graphics, graphics_record, graphics);
    V2::EncodeRecord(bytes, RecordKind::Compute, compute_record, compute);
    V2::EncodeRecord(bytes, RecordKind::Compute, Vulkan::ComputeRecordKey(other), other);
    std::vector<u8> damaged = compute;
    damaged.resize(damaged.size() / 2);
    damaged.insert(damaged.end(), compute.end() - sizeof(Vulkan::ComputePipelineCacheKey),
                   compute.end());
    V2::EncodeRecord(bytes, RecordKind::Compute, 0x5555, damaged);
    const V2::ParsedLog log = V2::ParseLog(bytes);
    REQUIRE(log.live.size() == 4);

    Vulkan::RelinkedPrograms programs;
    programs.emplace(old_program.unique_hash,
                     Vulkan::RelinkedProgram{.old_unique_hash = old_program.unique_hash,
                                             .old_texture_bound = 5,
                                             .program = new_program});
    const auto appends = Vulkan::RelinkPipelineRecords(log.live, programs);

    auto new_graphics_key = TestGraphicsKey(new_program.unique_hash, 0);
    new_graphics_key.unique_hashes[4] ^= 0x77;
    const auto new_graphics = payload(new_graphics_key, new_fragment);
    const auto new_compute_payload = payload(TestComputeKey(new_program.unique_hash), new_compute);
    REQUIRE(appends.size() == 4);
    REQUIRE(appends[0] == V2::LogAppend{RecordKind::Graphics,
                                        Vulkan::GraphicsRecordKey(new_graphics_key), new_graphics});
    REQUIRE(appends[1] == V2::LogAppend{RecordKind::Tombstone, graphics_record,
                                        V2::TombstonePayload(RecordKind::Graphics)});
    REQUIRE(appends[2] == V2::LogAppend{RecordKind::Compute,
                                        Vulkan::ComputeRecordKey(new_compute_payload),
                                        new_compute_payload});
    REQUIRE(appends[3] == V2::LogAppend{RecordKind::Tombstone, compute_record,
                                        V2::TombstonePayload(RecordKind::Compute)});
    REQUIRE(ReadPayload(appends[0].payload).stages ==
            std::vector<Shader::Stage>{Shader::Stage::Fragment});

    // A texture bound buffer the engine gave the old code can't be told for the new code.
    programs.begin()->second.program.texture_bound = ~0U;
    REQUIRE(Vulkan::RelinkPipelineRecords(log.live, programs).empty());
    // Stopped before the first record
    programs.begin()->second.program.texture_bound = 6;
    REQUIRE(Vulkan::RelinkPipelineRecords(log.live, programs, [] { return true; }).empty());
}


TEST_CASE("Shader cache v2: a program whose texture bound buffer alone moved is relinked",
          "[video_core][shader_cache_v2]") {
    V2::ProgramRecord current = LowerTestProgram(PlainProgram, false, 0x2000);
    V2::ProgramRecord unchanged = current;
    unchanged.lowering_stamp ^= 1;
    V2::ProgramRecord moved = current;
    moved.lowering_stamp ^= 1;
    moved.texture_bound = current.texture_bound == 5 ? 4 : 5;
    V2::ProgramRecord migrated = unchanged;
    migrated.native.clear();

    const Vulkan::RevalidatedPrograms revalidated =
        Vulkan::RevalidateProgramRecords({unchanged, moved, migrated});
    REQUIRE(revalidated.unchanged.size() == 1);
    REQUIRE(revalidated.unchanged[0].lowering_stamp == V2::LoweringStamp());
    REQUIRE(revalidated.relinked.size() == 1);
    const Vulkan::RelinkedProgram& relinked = revalidated.relinked.at(moved.unique_hash);
    REQUIRE(relinked.program.unique_hash == moved.unique_hash);
    REQUIRE(relinked.old_texture_bound == moved.texture_bound);
    REQUIRE(relinked.program.texture_bound == current.texture_bound);

    // Same code, so the same pipeline keys. A compute record's key covers its payload.
    V2::ProgramRecord old_program = current;
    old_program.texture_bound = 5;
    V2::ProgramRecord new_program = current;
    new_program.texture_bound = 6;
    const NativeTestEnvironment old_fragment{Shader::Stage::Fragment, old_program.lowered_code, 5};
    const NativeTestEnvironment new_fragment{Shader::Stage::Fragment, new_program.lowered_code, 6};
    const NativeTestEnvironment old_compute{Shader::Stage::Compute, old_program.lowered_code, 5};
    const NativeTestEnvironment new_compute{Shader::Stage::Compute, new_program.lowered_code, 6};
    const auto payload = [](const auto& key, const VideoCommon::GenericEnvironment& env) {
        const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&env};
        return VideoCommon::SerializePipelinePayload(KeyBytes(key), envs, ReadProperties);
    };
    const auto graphics_key = TestGraphicsKey(current.unique_hash, 0);
    const auto compute_key = TestComputeKey(current.unique_hash);
    const auto graphics = payload(graphics_key, old_fragment);
    const auto compute = payload(compute_key, old_compute);
    std::vector<u8> bytes;
    const auto header = V2::EncodeFileHeader(1);
    bytes.assign(header.begin(), header.end());
    const u64 graphics_record = Vulkan::GraphicsRecordKey(graphics_key);
    const u64 compute_record = Vulkan::ComputeRecordKey(compute);
    V2::EncodeRecord(bytes, RecordKind::Graphics, graphics_record, graphics);
    V2::EncodeRecord(bytes, RecordKind::Compute, compute_record, compute);
    const V2::ParsedLog log = V2::ParseLog(bytes);
    REQUIRE(log.live.size() == 2);

    Vulkan::RelinkedPrograms programs;
    programs.emplace(current.unique_hash,
                     Vulkan::RelinkedProgram{.old_unique_hash = current.unique_hash,
                                             .old_texture_bound = 5,
                                             .program = new_program});
    const auto appends = Vulkan::RelinkPipelineRecords(log.live, programs);
    const auto new_compute_payload = payload(compute_key, new_compute);
    REQUIRE(appends.size() == 3);
    REQUIRE(appends[0] == V2::LogAppend{RecordKind::Graphics, graphics_record,
                                        payload(graphics_key, new_fragment)});
    REQUIRE(appends[1] == V2::LogAppend{RecordKind::Compute,
                                        Vulkan::ComputeRecordKey(new_compute_payload),
                                        new_compute_payload});
    REQUIRE(appends[2] == V2::LogAppend{RecordKind::Tombstone, compute_record,
                                        V2::TombstonePayload(RecordKind::Compute)});
    REQUIRE(Vulkan::PipelineRecordKey<Vulkan::ComputePipelineCacheKey>(appends[1].payload)
                ->unique_hash == current.unique_hash);
}
TEST_CASE("Shader cache v2: an older Vulkan cache and its used list are migrated",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto legacy = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 53);
    const auto log_file = temp.path / std::string{V2::FileName};
    TestEnvironment fragment{Shader::Stage::Fragment};
    TestEnvironment other_fragment{Shader::Stage::Fragment, 1};
    TestEnvironment compute{Shader::Stage::Compute};
    const std::array<const VideoCommon::GenericEnvironment*, 1> fragment_envs{&fragment};
    const std::array<const VideoCommon::GenericEnvironment*, 1> other_envs{&other_fragment};
    const std::array<const VideoCommon::GenericEnvironment*, 1> compute_envs{&compute};
    // The same pipeline under two drivers, another pipeline, and a compute pipeline twice
    const auto old_driver = TestGraphicsKey(0x1111, 100);
    const auto new_driver = TestGraphicsKey(0x1111, 101);
    const auto second = TestGraphicsKey(0x3333, 101);
    VideoCommon::SerializePipeline(old_driver, fragment_envs, legacy, 53, ReadProperties);
    VideoCommon::SerializePipeline(TestComputeKey(0x2222), compute_envs, legacy, 53, ReadProperties);
    VideoCommon::SerializePipeline(new_driver, other_envs, legacy, 53, ReadProperties);
    VideoCommon::SerializePipeline(second, fragment_envs, legacy, 53, ReadProperties);
    VideoCommon::SerializePipeline(TestComputeKey(0x2222), compute_envs, legacy, 53, ReadProperties);
    {
        // The used list: the new-driver pipeline last session, the second one three ago, and one
        // not in the cache
        std::ofstream used(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan_used", 53),
                           std::ios::binary);
        const u32 magic = Vulkan::UsedPipelinesMagic;
        const u32 count = 3;
        used.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        used.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [hash, age] : std::array<std::pair<u64, u8>, 3>{
                 {{new_driver.Hash(), 0}, {second.Hash(), 3}, {0xdead, 1}}}) {
            used.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
            used.write(reinterpret_cast<const char*>(&age), sizeof(age));
        }
    }
    const auto legacy_bytes = ReadAll(legacy);

    const auto migration = Vulkan::MigrateLegacyPipelineCache(temp.path, log_file, 16);
    REQUIRE(migration.version == 53u);
    REQUIRE(migration.graphics == 2);
    REQUIRE(migration.compute == 1);
    REQUIRE(migration.collapsed == 2);
    REQUIRE(migration.used == 2);
    REQUIRE(ReadAll(legacy) == legacy_bytes);

    const V2::LogFile log = V2::ReadLogFile(log_file);
    REQUIRE(log.parsed.stats.dead_bytes == 0);
    std::map<RecordKind, std::vector<const V2::LogRecord*>> kinds;
    for (const V2::LogRecord& record : log.parsed.live) {
        kinds[record.kind].push_back(&record);
    }
    REQUIRE(kinds[RecordKind::Graphics].size() == 2);
    REQUIRE(kinds[RecordKind::Compute].size() == 1);
    REQUIRE(kinds[RecordKind::Migrated].size() == 1);
    REQUIRE(kinds[RecordKind::Used].size() == 2);

    // Keys without the driver; the collapsed pipeline holds the later record
    const ReadPipeline first = ReadPayload(kinds[RecordKind::Graphics][0]->payload);
    Vulkan::GraphicsPipelineCacheKey key;
    std::memcpy(&key, first.key.data(), sizeof(key));
    REQUIRE(key.state.driver_id == 0);
    REQUIRE(key.state.driver_version == 0);
    REQUIRE(kinds[RecordKind::Graphics][0]->key == key.Hash());
    REQUIRE(kinds[RecordKind::Graphics][0]->key == Vulkan::GraphicsRecordKey(new_driver));
    const auto other_payload = VideoCommon::SerializePipelinePayload(
        KeyBytes(new_driver), other_envs, ReadProperties);
    REQUIRE(std::ranges::equal(std::span{kinds[RecordKind::Graphics][0]->payload}.first(
                                   other_payload.size() - sizeof(key)),
                               std::span{other_payload}.first(other_payload.size() - sizeof(key))));
    REQUIRE(ReadPayload(kinds[RecordKind::Compute][0]->payload).compute);

    std::vector<u64> expired;
    const auto used = Vulkan::UsedPipelinesFromLog(log.parsed.live, 16, expired);
    REQUIRE(expired.empty());
    // Ages count the sessions that used a pipeline: the migrated list kept two.
    REQUIRE(used == std::unordered_map<u64, u8>{{Vulkan::GraphicsRecordKey(new_driver), 0},
                                               {Vulkan::GraphicsRecordKey(second), 1}});
    const auto migrated = V2::DecodeMigrated(kinds[RecordKind::Migrated][0]->payload);
    REQUIRE(migrated.has_value());
    REQUIRE(migrated->source == "vulkan.fork-v53.bin");
    REQUIRE(migrated->source_size == legacy_bytes.size());
    REQUIRE(migrated->source_version == 53);

    // Once only, and never over a log
    const auto log_bytes = ReadAll(log_file);
    REQUIRE_FALSE(Vulkan::MigrateLegacyPipelineCache(temp.path, log_file, 16).version.has_value());
    REQUIRE(ReadAll(log_file) == log_bytes);

    // The newest version first
    const auto newer = VideoCommon::VersionedPipelineCachePath(
        temp.path, "vulkan", VideoCommon::VulkanPipelineCacheVersion);
    VideoCommon::SerializePipeline(second, fragment_envs, newer,
                                   VideoCommon::VulkanPipelineCacheVersion, ReadProperties);
    const auto newer_log = temp.path / "newer.bin";
    const auto from_newer = Vulkan::MigrateLegacyPipelineCache(temp.path, newer_log, 16);
    REQUIRE(from_newer.version == VideoCommon::VulkanPipelineCacheVersion);
    REQUIRE(from_newer.graphics == 1);
    REQUIRE(from_newer.used == 0);
}

TEST_CASE("Shader cache v2: used pipelines age by the sessions that used them",
          "[video_core][shader_cache_v2]") {
    const auto header = V2::EncodeFileHeader(1);
    std::vector<u8> bytes(header.begin(), header.end());
    for (u64 session = 0; session < 5; ++session) {
        V2::EncodeRecord(bytes, RecordKind::Used, 100 + session,
                         V2::EncodeUsed({.session_time = 1000 + session * 10,
                                         .pipelines = {session, 50}}));
    }
    // An older session written later (another emulator), then a session's later save
    V2::EncodeRecord(bytes, RecordKind::Used, 200,
                     V2::EncodeUsed({.session_time = 5, .pipelines = {7}}));
    V2::EncodeRecord(bytes, RecordKind::Used, 104,
                     V2::EncodeUsed({.session_time = 1040, .pipelines = {4, 8}}));
    const V2::ParsedLog parsed = V2::ParseLog(bytes);
    std::vector<u64> expired;
    const auto used = Vulkan::UsedPipelinesFromLog(parsed.live, 3, expired);
    REQUIRE(used == std::unordered_map<u64, u8>{{4, 0}, {8, 0}, {3, 1}, {50, 1}, {2, 2}});
    std::ranges::sort(expired);
    REQUIRE(expired == std::vector<u64>{100, 101, 200});
}

TEST_CASE("Shader cache v2: used pipelines follow the graphics records a relink moves",
          "[video_core][shader_cache_v2]") {
    const auto header = V2::EncodeFileHeader(1);
    std::vector<u8> bytes(header.begin(), header.end());
    V2::EncodeRecord(bytes, RecordKind::Used, 100,
                     V2::EncodeUsed({.session_time = 10, .pipelines = {1}}));
    V2::EncodeRecord(bytes, RecordKind::Used, 101,
                     V2::EncodeUsed({.session_time = 20, .pipelines = {3}}));
    V2::EncodeRecord(bytes, RecordKind::Used, 102,
                     V2::EncodeUsed({.session_time = 30, .pipelines = {1, 2, 9}}));
    const V2::ParsedLog parsed = V2::ParseLog(bytes);
    std::vector<u64> expired;
    auto used = Vulkan::UsedPipelinesFromLog(parsed.live, 2, expired);
    REQUIRE(used == std::unordered_map<u64, u8>{{1, 0}, {2, 0}, {9, 0}, {3, 1}});

    // 1 moves to 5 and 3 to 9; 4 keeps its key, and the compute record of key 2 moved
    const std::vector<V2::LogAppend> appends{
        {RecordKind::Graphics, 5, {}},
        {RecordKind::Tombstone, 1, V2::TombstonePayload(RecordKind::Graphics)},
        {RecordKind::Graphics, 4, {}},
        {RecordKind::Compute, 7, {}},
        {RecordKind::Tombstone, 2, V2::TombstonePayload(RecordKind::Compute)},
        {RecordKind::Graphics, 9, {}},
        {RecordKind::Tombstone, 3, V2::TombstonePayload(RecordKind::Graphics)},
    };
    const auto rewritten = Vulkan::RelinkUsedPipelines(parsed.live, appends, 2, used);
    REQUIRE(used == std::unordered_map<u64, u8>{{5, 0}, {2, 0}, {9, 0}});
    // The expired session stays as it is
    REQUIRE(
        rewritten ==
        std::vector<V2::LogAppend>{
            {RecordKind::Used, 102, V2::EncodeUsed({.session_time = 30, .pipelines = {2, 5, 9}})},
            {RecordKind::Used, 101, V2::EncodeUsed({.session_time = 20, .pipelines = {9}})}});
    REQUIRE(Vulkan::RelinkUsedPipelines(parsed.live, {}, 2, used).empty());
}
