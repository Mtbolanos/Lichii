// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/native_lowering.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/records.h"

namespace VideoCommon::ShaderCacheV2 {
namespace {

/// Whether a record holds these inputs: a key that matches by chance never gives another
/// program's code
bool HoldsInputs(const ProgramRecord& record, const ProgramKeyInputs& inputs) {
    return record.source == inputs.source && record.stage == inputs.stage &&
           record.header_size == inputs.header_size && record.geometry == inputs.geometry &&
           record.tessellation_stage == inputs.tessellation_stage &&
           record.native_code_address == inputs.native_code_address &&
           std::ranges::equal(record.sph, inputs.sph) &&
           std::ranges::equal(record.verified_traps, inputs.verified_traps) &&
           record.native.size() == inputs.native.size() &&
           std::memcmp(record.native.data(), inputs.native.data(), inputs.native.size_bytes()) ==
               0;
}

} // Anonymous namespace

Cache::Cache(std::filesystem::path file, WriterOptions options)
    : writer{std::move(file), options} {}

Cache::~Cache() {
    const auto load = [this](ProgramLookup lookup, std::atomic<u64> ProgramCounts::*field) {
        return (counts[static_cast<size_t>(lookup)].*field).load(std::memory_order_relaxed);
    };
    const auto us_each = [](u64 ns, u64 lookups) {
        return lookups == 0 ? 0.0 : static_cast<double>(ns) / 1000.0 / static_cast<double>(lookups);
    };
    const u64 hits = load(ProgramLookup::Hit, &ProgramCounts::lookups);
    const u64 misses = load(ProgramLookup::Miss, &ProgramCounts::lookups);
    const u64 stale = load(ProgramLookup::Stale, &ProgramCounts::lookups);
    const u64 lowered = misses + stale;
    if (hits + lowered == 0) {
        return;
    }
    const u64 lowered_lookup_ns = load(ProgramLookup::Miss, &ProgramCounts::lookup_ns) +
                                  load(ProgramLookup::Stale, &ProgramCounts::lookup_ns);
    const u64 lowering_ns = load(ProgramLookup::Miss, &ProgramCounts::lowering_ns) +
                            load(ProgramLookup::Stale, &ProgramCounts::lowering_ns);
    LOG_INFO(Shader,
             "Shader cache v2 programs: {} from records ({:.1f} us each to capture and look up), "
             "{} lowered ({} without a record, {} with a stale one; {:.1f} us each to capture and "
             "look up, then {:.1f} us to lower)",
             hits, us_each(load(ProgramLookup::Hit, &ProgramCounts::lookup_ns), hits), lowered,
             misses, stale, us_each(lowered_lookup_ns, lowered), us_each(lowering_ns, lowered));
}

void Cache::AddProgram(u64 key, u64 lowering_stamp) {
    std::scoped_lock lock{mutex};
    programs.insert_or_assign(key, lowering_stamp);
}

void Cache::LoadPrograms(std::span<const LogRecord> live) {
    ASSERT(!loaded_ready.load(std::memory_order_relaxed));
    std::unordered_map<u64, LoadedProgram> table;
    for (const LogRecord& entry : live) {
        if (entry.kind != RecordKind::Program) {
            continue;
        }
        const std::optional<u64> stamp = ProgramStamp(entry.payload);
        if (!stamp) {
            continue;
        }
        AddProgram(entry.key, *stamp);
        if (*stamp != LoweringStamp()) {
            continue;
        }
        std::optional<ProgramRecord> record = DecodeProgram(entry.payload);
        // A record under another key than its inputs make would never be found
        if (!record || ProgramKey(*record) != entry.key) {
            continue;
        }
        std::shared_ptr<const LoweredNativeCode> code = TakeProgramCode(*record);
        table.insert_or_assign(entry.key, LoadedProgram{std::move(*record), std::move(code)});
    }
    loaded = std::move(table);
    loaded_ready.store(true, std::memory_order_release);
}

FoundProgram Cache::FindProgram(const ProgramKeyInputs& inputs) {
    const u64 key = ProgramKey(inputs);
    if (loaded_ready.load(std::memory_order_acquire)) {
        if (const auto it = loaded.find(key);
            it != loaded.end() && HoldsInputs(it->second.record, inputs)) {
            return {ProgramLookup::Hit, it->second.code};
        }
    }
    std::scoped_lock lock{mutex};
    const auto it = programs.find(key);
    return {it != programs.end() && it->second != LoweringStamp() ? ProgramLookup::Stale
                                                                  : ProgramLookup::Miss,
            nullptr};
}

void Cache::CountProgram(ProgramLookup lookup, u64 lookup_ns, u64 lowering_ns) {
    ProgramCounts& count = counts[static_cast<size_t>(lookup)];
    count.lookups.fetch_add(1, std::memory_order_relaxed);
    count.lookup_ns.fetch_add(lookup_ns, std::memory_order_relaxed);
    count.lowering_ns.fetch_add(lowering_ns, std::memory_order_relaxed);
    namespace StutterTrace = VideoCommon::StutterTrace;
    StutterTrace::CountLookup(lookup == ProgramLookup::Hit    ? StutterTrace::Lookup::ProgramHit
                              : lookup == ProgramLookup::Miss ? StutterTrace::Lookup::ProgramMiss
                                                              : StutterTrace::Lookup::ProgramStale);
}

void Cache::RecordProgram(const ProgramRecord& record) {
    const u64 key = ProgramKey(record);
    {
        std::scoped_lock lock{mutex};
        const auto [it, inserted] = programs.try_emplace(key, record.lowering_stamp);
        if (!inserted) {
            if (it->second == record.lowering_stamp || keep_stale) {
                return;
            }
            it->second = record.lowering_stamp;
        }
    }
    writer.Append(RecordKind::Program, key, EncodeProgram(record));
}

void Cache::KeepStalePrograms() {
    std::scoped_lock lock{mutex};
    keep_stale = true;
}

} // namespace VideoCommon::ShaderCacheV2
