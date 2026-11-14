// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <istream>
#include <map>
#include <unordered_map>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/unique_function.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_records.h"
#include "video_core/renderer_vulkan/vk_shader_cache_v2.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"

namespace Vulkan {

namespace ShaderCacheV2 = VideoCommon::ShaderCacheV2;
using ShaderCacheV2::RecordKind;

void ClearDriverFields(GraphicsPipelineCacheKey& key) {
    key.state.driver_id = 0;
    key.state.driver_version = 0;
}

u64 ComputeRecordKey(std::span<const u8> payload) {
    return ShaderCacheV2::Hash64(payload);
}

u64 GraphicsRecordKey(const GraphicsPipelineCacheKey& key) {
    GraphicsPipelineCacheKey cleared{key};
    ClearDriverFields(cleared);
    return cleared.Hash();
}

u64 MigratedUsedSessionKey(u8 age) {
    return 0x4c58555345440000ULL | age; // "LXUSED"
}

u64 MigratedUsedSessionTime(u8 age) {
    // Before every session of the log, which are timed in Unix nanoseconds
    return 0x100 - u64{age};
}

LegacyCacheMigration MigrateLegacyPipelineCache(const std::filesystem::path& directory,
                                                const std::filesystem::path& log,
                                                u8 max_used_age) {
    LegacyCacheMigration result;
    std::error_code error;
    if (std::filesystem::exists(log, error) || error) {
        return result;
    }
    std::array<u32, 1 + VideoCommon::CompatibleVulkanPipelineCacheVersions.size()> versions{
        VideoCommon::VulkanPipelineCacheVersion};
    std::ranges::copy(VideoCommon::CompatibleVulkanPipelineCacheVersions, versions.begin() + 1);
    for (const u32 version : versions) {
        const auto source = VideoCommon::VersionedPipelineCachePath(directory, "vulkan", version);
        if (!std::filesystem::is_regular_file(source, error)) {
            continue;
        }
        ShaderCacheV2::LogBuilder builder;
        switch (builder.Start(log)) {
        case ShaderCacheV2::WriteResult::Exists:
            return result; // Another emulator migrated it first
        case ShaderCacheV2::WriteResult::Written:
            break;
        default:
            return {.failed = true};
        }
        // The source is read twice. The first pass keeps only where each record lies and which key
        // it has, so that a record of one key replaces the ones before it; the second writes the
        // kept records out as the bytes they are. The source, and a copy of it, never sit in
        // memory at once, which a shared cache of tens of thousands of pipelines did not fit in.
        struct LegacyRecord {
            RecordKind kind;
            u64 key;
            u64 begin;
            u64 end;
        };
        std::vector<LegacyRecord> records;
        std::unordered_map<u64, size_t> last_of_key; ///< The record each key was last written by
        std::unordered_map<u64, u64> rekeyed;        ///< Key hash with the driver -> without
        const auto source_size = std::filesystem::file_size(source, error);
        const auto source_mtime = std::filesystem::last_write_time(source, error);
        u64 record_begin{VideoCommon::PipelineCacheMagic.size() + sizeof(u32)};
        // The bytes of the record the callback has read, from its start to the end of its key
        const auto record_bytes = [&](std::istream& file, u64& begin) {
            const u64 end{static_cast<u64>(std::streamoff{file.tellg()})};
            begin = record_begin;
            std::vector<u8> bytes(static_cast<size_t>(end - record_begin));
            file.seekg(static_cast<std::streamoff>(record_begin));
            file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            record_begin = end;
            return bytes;
        };
        const auto add = [&](RecordKind kind, u64 key, u64 begin) {
            last_of_key.insert_or_assign(key, records.size());
            records.push_back({kind, key, begin, record_begin});
        };
        const bool read = VideoCommon::ReadPipelinesUnchanged(
            source, version,
            [&](std::istream& file, VideoCommon::FileEnvironment env) {
                ComputePipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                VideoCommon::CheckComputeRecord(env, key.shared_memory_size, key.workgroup_size);
                u64 begin{};
                const std::vector<u8> bytes = record_bytes(file, begin);
                add(RecordKind::Compute, ComputeRecordKey(bytes), begin);
            },
            [&](std::istream& file, std::vector<VideoCommon::FileEnvironment> envs) {
                GraphicsPipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                VideoCommon::CheckGraphicsRecord(key.unique_hashes, envs.size());
                u64 begin{};
                const std::vector<u8> bytes = record_bytes(file, begin);
                const u64 hash = key.Hash();
                ClearDriverFields(key);
                rekeyed.insert_or_assign(hash, key.Hash());
                add(RecordKind::Graphics, key.Hash(), begin);
            });
        if (!read) {
            builder.Discard();
            continue; // Another file under that name
        }
        result.collapsed = static_cast<u32>(records.size() - last_of_key.size());
        for (size_t i = 0; i < records.size(); ++i) {
            if (last_of_key[records[i].key] != i) {
                continue;
            }
            ++(records[i].kind == RecordKind::Graphics ? result.graphics : result.compute);
        }

        // The second pass copies the records the first kept. A source another process wrote in
        // between no longer matches the offsets. The migration fails then, as when the log is
        // busy, rather than falling back to an older source that would shadow this one.
        std::ifstream kept_source(source, std::ios::binary);
        if (!kept_source.is_open() ||
            std::filesystem::file_size(source, error) != source_size ||
            std::filesystem::last_write_time(source, error) != source_mtime) {
            builder.Discard();
            return {.failed = true};
        }
        std::vector<u8> bytes;
        bool appended{true};
        for (size_t i = 0; i < records.size() && appended; ++i) {
            const LegacyRecord& record = records[i];
            if (last_of_key[record.key] != i) {
                continue;
            }
            bytes.resize(static_cast<size_t>(record.end - record.begin));
            kept_source.seekg(static_cast<std::streamoff>(record.begin));
            kept_source.read(reinterpret_cast<char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
            if (!kept_source.good()) {
                appended = false;
                break;
            }
            if (record.kind == RecordKind::Graphics) {
                // The key the record ends with is kept without the driver that built it
                GraphicsPipelineCacheKey key;
                std::memcpy(&key, bytes.data() + bytes.size() - sizeof(key), sizeof(key));
                ClearDriverFields(key);
                std::memcpy(bytes.data() + bytes.size() - sizeof(key), &key, sizeof(key));
            }
            appended = builder.Append(record.kind, record.key, bytes);
        }
        if (!appended) {
            builder.Discard();
            return {.failed = true};
        }

        // The used list saved beside the cache, by sessions since the last use
        const auto used = LoadUsedPipelines(
            VideoCommon::VersionedPipelineCachePath(directory, "vulkan_used", version),
            max_used_age);
        std::map<u8, std::vector<u64>> sessions;
        for (const auto& [hash, age] : used) {
            if (const auto it = rekeyed.find(hash); it != rekeyed.end()) {
                sessions[age].push_back(it->second);
                ++result.used;
            }
        }
        for (auto& [age, pipelines] : sessions) {
            std::ranges::sort(pipelines);
            const std::vector<u8> payload = ShaderCacheV2::EncodeUsed({
                .session_time = MigratedUsedSessionTime(age),
                .pipelines = std::move(pipelines),
            });
            appended = builder.Append(RecordKind::Used, MigratedUsedSessionKey(age), payload) &&
                       appended;
        }

        const ShaderCacheV2::MigratedRecord migrated{
            .source = Common::FS::PathToUTF8String(source.filename()),
            .source_size = static_cast<u64>(std::filesystem::file_size(source, error)),
            .source_mtime = static_cast<s64>(
                std::filesystem::last_write_time(source, error).time_since_epoch().count()),
            .source_version = version,
            .graphics = result.graphics,
            .compute = result.compute,
            .used = result.used,
        };
        const std::vector<u8> name(migrated.source.begin(), migrated.source.end());
        const std::vector<u8> migrated_payload = ShaderCacheV2::EncodeMigrated(migrated);
        appended = builder.Append(RecordKind::Migrated, ShaderCacheV2::Hash64(name),
                                  migrated_payload) &&
                   appended;
        if (!appended) {
            builder.Discard();
            return {.failed = true};
        }

        switch (builder.Commit()) {
        case ShaderCacheV2::WriteResult::Written:
            result.version = version;
            break;
        case ShaderCacheV2::WriteResult::Exists:
            result = {}; // Another emulator migrated it first
            break;
        default:
            result = {.failed = true};
            break;
        }
        return result;
    }
    return result;
}

std::optional<RekeyedRecord> RekeyPipelineRecord(RecordKind kind, std::span<const u8> payload,
                                                 const RelinkedPrograms& programs) {
    bool replaced_any{};
    auto replaced = VideoCommon::ReplacePipelineCode(
        payload,
        [&](u64 shader_hash, u32 texture_bound) -> std::optional<VideoCommon::NativeCodeReplacement> {
            const auto it = programs.find(shader_hash);
            if (it == programs.end()) {
                return std::nullopt;
            }
            const auto& program = it->second.program;
            if (program.texture_bound != ~0U) {
                replaced_any = true;
                return VideoCommon::NativeCodeReplacement{program.lowered_code, program.texture_bound};
            }
            if (it->second.old_texture_bound != ~0U) {
                return std::nullopt; // The engine's value was never recorded
            }
            replaced_any = true;
            return VideoCommon::NativeCodeReplacement{program.lowered_code, texture_bound};
        });
    if (!replaced_any) {
        return std::nullopt;
    }
    RekeyedRecord record{.key = 0, .payload = std::move(replaced.payload)};
    const auto store = [&record]<typename Key>(const Key& key) {
        std::memcpy(record.payload.data() + record.payload.size() - sizeof(Key), &key, sizeof(Key));
    };
    if (kind == RecordKind::Compute) {
        auto key = PipelineRecordKey<ComputePipelineCacheKey>(record.payload);
        if (!key || replaced.shader_hashes.size() != 1 ||
            key->unique_hash != replaced.shader_hashes[0].first) {
            throw std::ios_base::failure("Compute record key doesn't match its program");
        }
        key->unique_hash = replaced.shader_hashes[0].second;
        store(*key);
        record.key = ComputeRecordKey(record.payload);
        return record;
    }
    auto key = PipelineRecordKey<GraphicsPipelineCacheKey>(record.payload);
    if (!key) {
        throw std::ios_base::failure("Graphics record without a key");
    }
    VideoCommon::CheckGraphicsRecord(key->unique_hashes, replaced.shader_hashes.size());
    size_t env{};
    for (u64& unique_hash : key->unique_hashes) {
        if (unique_hash == 0) {
            continue;
        }
        // Task and mesh stages hold their program's hash with the mesh state's mixed in.
        const auto [old_hash, new_hash] = replaced.shader_hashes[env++];
        unique_hash ^= old_hash ^ new_hash;
    }
    store(*key);
    record.key = GraphicsRecordKey(*key);
    return record;
}

RevalidatedPrograms RevalidateProgramRecords(std::vector<ShaderCacheV2::ProgramRecord> programs,
                                             const ShaderCacheV2::ForEach& for_each) {
    std::vector<std::optional<ShaderCacheV2::ProgramRecord>> relowered(programs.size());
    const auto relower = [&](size_t index) {
        try {
            relowered[index] = ShaderCacheV2::RelowerProgramRecord(programs[index]);
        } catch (const std::exception& e) {
            LOG_DEBUG(Render_Vulkan, "Shader cache v2: program {:016x}: {}",
                      programs[index].unique_hash, e.what());
        }
    };
    if (for_each) {
        for_each(programs.size(), relower);
    } else {
        for (size_t index = 0; index < programs.size(); ++index) {
            relower(index);
        }
    }
    RevalidatedPrograms result;
    for (size_t index = 0; index < programs.size(); ++index) {
        if (!relowered[index]) {
            continue;
        }
        const ShaderCacheV2::ProgramRecord& program = programs[index];
        if (relowered[index]->unique_hash == program.unique_hash &&
            relowered[index]->texture_bound == program.texture_bound) {
            result.unchanged.push_back(std::move(*relowered[index]));
            continue;
        }
        result.relinked.emplace(program.unique_hash,
                                RelinkedProgram{.old_unique_hash = program.unique_hash,
                                                .old_texture_bound = program.texture_bound,
                                                .program = std::move(*relowered[index])});
    }
    return result;
}

std::vector<ShaderCacheV2::LogAppend> RelinkPipelineRecords(
    std::span<const ShaderCacheV2::LogRecord> live, const RelinkedPrograms& programs,
    const std::function<bool()>& stop) {
    std::vector<ShaderCacheV2::LogAppend> appends;
    if (programs.empty()) {
        return appends;
    }
    for (const ShaderCacheV2::LogRecord& record : live) {
        if (stop && stop()) {
            break;
        }
        if (record.kind != RecordKind::Graphics && record.kind != RecordKind::Compute) {
            continue;
        }
        std::optional<RekeyedRecord> rekeyed;
        try {
            rekeyed = RekeyPipelineRecord(record.kind, record.payload, programs);
        } catch (const std::ios_base::failure&) {
            continue; // The boot load removes it
        }
        if (!rekeyed) {
            continue;
        }
        appends.push_back({record.kind, rekeyed->key, std::move(rekeyed->payload)});
        if (rekeyed->key != record.key) {
            appends.push_back({RecordKind::Tombstone, record.key,
                               ShaderCacheV2::TombstonePayload(record.kind)});
        }
    }
    return appends;
}

namespace {
struct UsedSession {
    u64 key;
    ShaderCacheV2::UsedRecord used;
};

/// The log's Used records, newest first
std::vector<UsedSession> UsedSessions(std::span<const ShaderCacheV2::LogRecord> live) {
    std::vector<UsedSession> sessions;
    for (const ShaderCacheV2::LogRecord& record : live) {
        if (record.kind != RecordKind::Used) {
            continue;
        }
        if (auto used = ShaderCacheV2::DecodeUsed(record.payload)) {
            sessions.push_back({record.key, std::move(*used)});
        }
    }
    std::ranges::stable_sort(sessions, std::ranges::greater{},
                             [](const UsedSession& session) { return session.used.session_time; });
    return sessions;
}
} // Anonymous namespace

std::unordered_map<u64, u8> UsedPipelinesFromLog(std::span<const ShaderCacheV2::LogRecord> live,
                                                 u8 max_age, std::vector<u64>& expired) {
    const std::vector<UsedSession> sessions = UsedSessions(live);
    std::unordered_map<u64, u8> pipelines;
    for (size_t age = 0; age < sessions.size(); ++age) {
        if (age >= max_age) {
            expired.push_back(sessions[age].key);
            continue;
        }
        for (const u64 hash : sessions[age].used.pipelines) {
            pipelines.try_emplace(hash, static_cast<u8>(age));
        }
    }
    return pipelines;
}

std::vector<ShaderCacheV2::LogAppend> RelinkUsedPipelines(
    std::span<const ShaderCacheV2::LogRecord> live,
    std::span<const ShaderCacheV2::LogAppend> appends, u8 max_age,
    std::unordered_map<u64, u8>& used) {
    // A graphics record whose key moved is followed by the old key's tombstone. Used lists hold
    // key hashes, which are the record keys.
    std::unordered_map<u64, u64> moved;
    for (size_t index = 0; index + 1 < appends.size(); ++index) {
        if (appends[index].kind == RecordKind::Graphics &&
            appends[index + 1].kind == RecordKind::Tombstone) {
            moved.emplace(appends[index + 1].key, appends[index].key);
        }
    }
    std::vector<ShaderCacheV2::LogAppend> result;
    if (moved.empty()) {
        return result;
    }
    const auto new_key = [&moved](u64 key) {
        const auto it = moved.find(key);
        return it == moved.end() ? key : it->second;
    };
    std::unordered_map<u64, u8> relinked_used;
    for (const auto& [key, age] : used) {
        const auto it = relinked_used.try_emplace(new_key(key), age).first;
        it->second = std::min(it->second, age);
    }
    used = std::move(relinked_used);

    std::vector<UsedSession> sessions = UsedSessions(live);
    sessions.resize(std::min<size_t>(sessions.size(), max_age));
    for (UsedSession& session : sessions) {
        std::vector<u64>& pipelines = session.used.pipelines;
        if (std::ranges::none_of(pipelines, [&moved](u64 key) { return moved.contains(key); })) {
            continue;
        }
        std::ranges::transform(pipelines, pipelines.begin(), new_key);
        std::ranges::sort(pipelines);
        pipelines.erase(std::ranges::unique(pipelines).begin(), pipelines.end());
        result.push_back({RecordKind::Used, session.key, ShaderCacheV2::EncodeUsed(session.used)});
    }
    return result;
}

} // namespace Vulkan
