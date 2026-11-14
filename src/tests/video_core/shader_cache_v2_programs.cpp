// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shader cache v2: native programs taken from Program records instead of lowered again.

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/page_table.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/ampere/trap_guards.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/stage.h"
#include "tests/core/system_test_context.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/gpu.h"
#include "video_core/macro.h"
#include "video_core/memory_manager.h"
#include "video_core/native_lowering.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"

namespace {

namespace V2 = VideoCommon::ShaderCacheV2;
using Shader::Ampere::Instruction;
using Shader::Ampere::TessellationStage;
using namespace std::chrono_literals;

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-shader-cache-v2-programs-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

/// A native program and what its lowering takes, as the environments capture them
struct TestProgram {
    const char* name;
    Shader::Stage stage;
    std::vector<Instruction> native;
    VideoCommon::NativeLoweringInputs inputs;

    u32 HeaderSize() const {
        return stage == Shader::Stage::Compute ? 0 : 128;
    }
    Shader::ProgramHeader Header() const {
        Shader::ProgramHeader sph{};
        if (stage != Shader::Stage::Compute) {
            std::memset(&sph, 0x5a, sizeof(sph));
        }
        return sph;
    }
    std::vector<u32> Traps() const {
        return VideoCommon::NativeTrapOffsets(native);
    }
};

const Instruction ControlLoad{{0xff057321, 0x000018ff, 0x00009000, 0x000e6200}};
const Instruction EvalLoad{{0xff057321, 0x000018ff, 0x00001000, 0x000e6200}};

/// Guarded by bank 0 offset 0 being below 0x80000000 (ampere_trap_guards.cpp)
const std::vector<Instruction> TrapProgram{{
    {{0xff037624, 0, 0x078e00ff, 0}},
    {{0x0300720c, 255, 0x03f06270, 0}},
    {{0x0947, 16, 0x03800000, 0}},
    {{0x795c, 4, 0x00300000, 0}},
    Exit,
}};

/// Fire Emblem's byte fill, two trap guards (ampere_trap_guards.cpp)
const std::vector<Instruction> ByteFill{{
    {{0xff017624, 0x0a00, 0x078e00ff, 0}},  {{0xff007a0c, 0x5f00, 0x03f05270, 0}},
    {{0xff037624, 0x5b00, 0x078e00ff, 0}},  {{0xff007807, 1, 0x04000000, 0}},
    {{0xff007a0c, 0x5f00, 0x03f05270, 0}},  {{0x03027a12, 0x5a00, 0x078efcff, 0}},
    {{0xff007388, 0, 0, 0}},                {{0x0200780c, 0xffffffff, 0x04704670, 0}},
    {{0x0947, 0x10, 0x03800000, 0}},        {{0x795c, 4, 0x00300000, 0}},
    {{0x03007a10, 0x5d00, 0x07ffe0ff, 0}},  {{0xff027624, 0x5c00, 0x078e00ff, 0}},
    {{0x7a0c, 0x5900, 0x03f04270, 0}},      {{0x02007a10, 0x5a00, 0x07ffe0ff, 0}},
    {{0x7a0c, 0x5800, 0x04703270, 0}},      {{0xff007a0c, 0x5f00, 0x00702670, 0}},
    {{0x0947, 0x10, 0x03800000, 0}},        {{0x795c, 4, 0x00300000, 0}},
    Exit,
}};

/// Requires constant buffer 2 for its texture handles (ampere_fast_fusion.cpp)
const Instruction Txd{{0x10047b6c, 0x20800c04, 0x041e0707, 0x00014800}};

const std::vector<Instruction> PlainProgram{{
    {{0x06007947, 0x10 | 3, 0x0b800000, 0}}, // BRA.CONV to the second EXIT
    Exit,
    Exit,
}};

const std::vector<Instruction> ProgramCounterProgram{{
    {{0x00007918, 0, 0, 0}}, // NOP
    {{0x0014734e, 0, 0, 0}}, // LEPC R20
    Exit,
}};

std::vector<TestProgram> TestPrograms() {
    using Shader::Stage;
    return {
        {"plain", Stage::VertexB, PlainProgram, {.native_code_address = 0x2000}},
        {"plain fragment", Stage::Fragment, PlainProgram, {.native_code_address = 0x2000}},
        {"program counter", Stage::Geometry, ProgramCounterProgram,
         {.geometry = true, .native_code_address = 0x12fffffff0}},
        {"program counter elsewhere", Stage::Geometry, ProgramCounterProgram,
         {.geometry = true, .native_code_address = 0x8000}},
        {"tessellation control", Stage::TessellationControl, {ControlLoad, Exit},
         {.tessellation_stage = TessellationStage::Control, .native_code_address = 0x3000}},
        {"tessellation evaluation", Stage::TessellationEval, {EvalLoad, Exit},
         {.tessellation_stage = TessellationStage::Evaluation, .native_code_address = 0x3000}},
        {"compute traps", Stage::Compute, TrapProgram, {.native_code_address = 0x5000}},
        {"compute byte fill", Stage::Compute, ByteFill, {.native_code_address = 0x6000}},
        {"compute texture buffer", Stage::Compute, {Txd, Exit}, {.native_code_address = 0x7000}},
    };
}

/// What the environment makes of a program without the cache: LowerNativeCode's lowering
std::shared_ptr<VideoCommon::LoweredNativeCode> LowerFresh(const TestProgram& program) {
    auto lowered = VideoCommon::LowerNativeProgram(program.native, program.inputs);
    INFO(program.name << ": " << lowered.error);
    REQUIRE(static_cast<bool>(lowered));
    return VideoCommon::MakeLoweredNativeCode(program.native, program.Header(),
                                              program.HeaderSize(), program.Traps(), lowered);
}

V2::ProgramRecord RecordOf(const TestProgram& program,
                           const VideoCommon::LoweredNativeCode& code) {
    return V2::MakeProgramRecord(program.stage, program.HeaderSize(), program.Header(),
                                 program.native, program.inputs, code);
}

/// Every field the environments and the shader cache read
void RequireSameCode(const VideoCommon::LoweredNativeCode& actual,
                     const VideoCommon::LoweredNativeCode& expected) {
    REQUIRE(actual.native_program_key == expected.native_program_key);
    REQUIRE(actual.code == expected.code);
    REQUIRE(actual.source_size == expected.source_size);
    REQUIRE(actual.texture_bound == expected.texture_bound);
    REQUIRE(actual.trap_prefix.size() == expected.trap_prefix.size());
    REQUIRE(std::memcmp(actual.trap_prefix.data(), expected.trap_prefix.data(),
                        expected.trap_prefix.size() * sizeof(Instruction)) == 0);
    REQUIRE(VideoCommon::NativeShaderHash(actual.code) ==
            VideoCommon::NativeShaderHash(expected.code));
}

/// The key as the first v2 build computed it: the hash of the fields laid out in a buffer
u64 BufferedProgramKey(const V2::ProgramRecord& record) {
    std::vector<u8> bytes;
    const auto put = [&bytes](const void* data, size_t size) {
        const auto* begin = static_cast<const u8*>(data);
        bytes.insert(bytes.end(), begin, begin + size);
    };
    const u64 trap_count = record.verified_traps.size();
    const u64 native_count = record.native.size();
    put(&record.source, sizeof(record.source));
    put(&record.stage, sizeof(record.stage));
    put(&record.header_size, sizeof(record.header_size));
    put(record.sph.data(), record.sph.size());
    put(&record.geometry, sizeof(record.geometry));
    put(&record.tessellation_stage, sizeof(record.tessellation_stage));
    put(&record.native_code_address, sizeof(record.native_code_address));
    put(&trap_count, sizeof(trap_count));
    put(record.verified_traps.data(), record.verified_traps.size() * sizeof(u32));
    put(&native_count, sizeof(native_count));
    put(record.native.data(), record.native.size() * sizeof(Instruction));
    return V2::Hash64(bytes);
}

V2::ProgramKeyInputs KeyInputs(const TestProgram& program, const Shader::ProgramHeader& sph,
                               const std::vector<u32>& traps) {
    return V2::MakeProgramKeyInputs(program.stage, program.HeaderSize(), sph, program.native,
                                    program.inputs, traps);
}

/// A log holding these records, written by a cache's thread
void WriteLog(const std::filesystem::path& file, std::span<const V2::ProgramRecord> records) {
    V2::Cache cache{file};
    for (const V2::ProgramRecord& record : records) {
        cache.RecordProgram(record);
    }
    REQUIRE(cache.Writer().Drain(5s));
}

std::vector<u8> LogBytes(std::span<const std::pair<u64, std::vector<u8>>> programs) {
    const auto header = V2::EncodeFileHeader(1);
    std::vector<u8> bytes(header.begin(), header.end());
    for (const auto& [key, payload] : programs) {
        V2::EncodeRecord(bytes, V2::RecordKind::Program, key, payload);
    }
    return bytes;
}

} // Anonymous namespace

TEST_CASE("Shader cache v2: a program from its record is the code a lowering makes",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const std::vector<TestProgram> programs = TestPrograms();
    std::vector<std::shared_ptr<VideoCommon::LoweredNativeCode>> fresh;
    std::vector<V2::ProgramRecord> records;
    for (const TestProgram& program : programs) {
        fresh.push_back(LowerFresh(program));
        records.push_back(RecordOf(program, *fresh.back()));
        INFO(program.name);
        // The keys of logs written before the key was streamed stay the same
        REQUIRE(V2::ProgramKey(records.back()) == BufferedProgramKey(records.back()));
        const auto sph = program.Header();
        const auto traps = program.Traps();
        REQUIRE(V2::ProgramKey(KeyInputs(program, sph, traps)) == V2::ProgramKey(records.back()));
        // What a log holds, decoded
        auto decoded = V2::DecodeProgram(V2::EncodeProgram(records.back()));
        REQUIRE(decoded.has_value());
        RequireSameCode(*V2::TakeProgramCode(*decoded), *fresh.back());
    }
    // The fixtures cover what the record carries besides the code
    REQUIRE(fresh[2]->code != fresh[3]->code);
    REQUIRE(!fresh[6]->trap_prefix.empty());
    REQUIRE(!fresh[7]->trap_prefix.empty());
    REQUIRE(fresh[8]->texture_bound == 2);

    WriteLog(file, records);
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.live.size() == programs.size());
    V2::Cache cache{file};
    cache.LoadPrograms(log.parsed.live);
    for (size_t i = 0; i < programs.size(); ++i) {
        INFO(programs[i].name);
        const auto sph = programs[i].Header();
        const auto traps = programs[i].Traps();
        const V2::FoundProgram found = cache.FindProgram(KeyInputs(programs[i], sph, traps));
        REQUIRE(found.lookup == V2::ProgramLookup::Hit);
        REQUIRE(found.code != nullptr);
        RequireSameCode(*found.code, *fresh[i]);
    }

    // The code address only tells programs loading the program counter apart: one of those at
    // a third address finds nothing, a plain program at another address finds its record.
    TestProgram moved = programs[2];
    moved.inputs.native_code_address = 0x9000;
    TestProgram plain_moved = programs[0];
    plain_moved.inputs.native_code_address = 0xabc000;
    for (const auto& [program, lookup] :
         {std::pair{moved, V2::ProgramLookup::Miss}, {plain_moved, V2::ProgramLookup::Hit}}) {
        const auto sph = program.Header();
        const auto traps = program.Traps();
        REQUIRE(cache.FindProgram(KeyInputs(program, sph, traps)).lookup == lookup);
    }
    // Another header or stage is another program
    TestProgram vertex_a = programs[0];
    vertex_a.stage = Shader::Stage::VertexA;
    auto other_header = programs[0].Header();
    std::memset(&other_header, 0x11, 4);
    const auto traps = programs[0].Traps();
    REQUIRE(cache.FindProgram(KeyInputs(vertex_a, vertex_a.Header(), traps)).lookup ==
            V2::ProgramLookup::Miss);
    REQUIRE(cache.FindProgram(KeyInputs(programs[0], other_header, traps)).lookup ==
            V2::ProgramLookup::Miss);
}

TEST_CASE("Shader cache v2: a record of another lowering is stale and never used",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const TestProgram program = TestPrograms()[0];
    V2::ProgramRecord stale = RecordOf(program, *LowerFresh(program));
    stale.lowering_stamp ^= 1;
    WriteLog(file, std::span{&stale, 1});
    const V2::LogFile log = V2::ReadLogFile(file);
    REQUIRE(log.parsed.live.size() == 1);
    V2::Cache cache{file};
    cache.LoadPrograms(log.parsed.live);
    const auto sph = program.Header();
    const auto traps = program.Traps();
    const V2::FoundProgram found = cache.FindProgram(KeyInputs(program, sph, traps));
    REQUIRE(found.lookup == V2::ProgramLookup::Stale);
    REQUIRE(found.code == nullptr);
    // Lowered again, its record of this lowering is written
    cache.RecordProgram(RecordOf(program, *LowerFresh(program)));
    REQUIRE(cache.Writer().Drain(5s));
    REQUIRE(cache.Writer().Stats().written_records == 1);

    const TestProgram unknown = TestPrograms()[4];
    const auto unknown_sph = unknown.Header();
    const auto unknown_traps = unknown.Traps();
    REQUIRE(cache.FindProgram(KeyInputs(unknown, unknown_sph, unknown_traps)).lookup ==
            V2::ProgramLookup::Miss);
}

TEST_CASE("Shader cache v2: a stale record stays after KeepStalePrograms",
          "[video_core][shader_cache_v2]") {
    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const std::vector<TestProgram> programs = TestPrograms();
    const TestProgram& program = programs[0];
    V2::ProgramRecord stale = RecordOf(program, *LowerFresh(program));
    stale.lowering_stamp ^= 1;
    WriteLog(file, std::span{&stale, 1});
    const V2::LogFile log = V2::ReadLogFile(file);
    V2::Cache cache{file};
    cache.KeepStalePrograms();
    cache.LoadPrograms(log.parsed.live);
    // Lowered again, its record of this lowering isn't written, so the next boot load still
    // lowers it again and moves its pipelines' records
    cache.RecordProgram(RecordOf(program, *LowerFresh(program)));
    const auto sph = program.Header();
    const auto traps = program.Traps();
    REQUIRE(cache.FindProgram(KeyInputs(program, sph, traps)).lookup == V2::ProgramLookup::Stale);
    // A program the log has no record of is still recorded
    const TestProgram& unknown = programs[4];
    cache.RecordProgram(RecordOf(unknown, *LowerFresh(unknown)));
    REQUIRE(cache.Writer().Drain(5s));
    REQUIRE(cache.Writer().Stats().written_records == 1);
}

TEST_CASE("Shader cache v2: a damaged program record never gives code",
          "[video_core][shader_cache_v2]") {
    const std::vector<TestProgram> programs = TestPrograms();
    const TestProgram& program = programs[6];
    const V2::ProgramRecord record = RecordOf(program, *LowerFresh(program));
    const u64 key = V2::ProgramKey(record);
    const std::vector<u8> payload = V2::EncodeProgram(record);
    const auto sph = program.Header();
    const auto traps = program.Traps();
    TempDir temp;
    const auto find = [&](std::span<const u8> bytes) {
        const V2::ParsedLog parsed = V2::ParseLog(bytes);
        V2::Cache cache{temp.path / "unused.bin"};
        cache.LoadPrograms(parsed.live);
        return std::pair{parsed.stats, cache.FindProgram(KeyInputs(program, sph, traps))};
    };

    const std::pair<u64, std::vector<u8>> intact{key, payload};
    REQUIRE(find(LogBytes(std::span{&intact, 1})).second.lookup == V2::ProgramLookup::Hit);

    // A byte of the lowered code changed on the disk: the payload hash drops the record.
    std::vector<u8> bytes = LogBytes(std::span{&intact, 1});
    bytes[V2::FileHeaderSize + V2::RecordHeaderSize + payload.size() - 3] ^= 0x10;
    const auto [stats, damaged] = find(bytes);
    REQUIRE(stats.bad_payloads == 1);
    REQUIRE(damaged.lookup == V2::ProgramLookup::Miss);
    REQUIRE(damaged.code == nullptr);

    // An intact record under the key of another program, or one that doesn't decode
    const V2::ProgramRecord other = RecordOf(programs[7], *LowerFresh(programs[7]));
    const std::pair<u64, std::vector<u8>> misfiled{key, V2::EncodeProgram(other)};
    REQUIRE(find(LogBytes(std::span{&misfiled, 1})).second.code == nullptr);
    std::vector<u8> cut = payload;
    cut.pop_back();
    const std::pair<u64, std::vector<u8>> truncated{key, cut};
    REQUIRE(find(LogBytes(std::span{&truncated, 1})).second.code == nullptr);
}

namespace {

/// Guest memory holding native programs at GPU address 0x100000, and a native compute engine
struct NativeComputeFixture {
    static constexpr GPUVAddr Base = 0x100000;
    static constexpr u64 ConstantsOffset = 0xf000;

    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    std::optional<Tegra::MemoryManager> memory;
    std::optional<Tegra::Engines::KeplerCompute> compute;
    Tegra::MacroEngine macros{true};
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    u8* bytes{};

    NativeComputeFixture() {
        system.Initialize();
        process = std::make_unique<Kernel::KProcess>(system.Kernel());
        auto& table = process->GetPageTable().GetImpl();
        table.Resize(24, 12);
        bytes = physical.GetPointerFromRaw<u8>(0x10000);
        std::memset(bytes, 0, 0x10000);
        for (u64 page = 0; page < 16; ++page) {
            table.entries.GetAndFault(0x10 + page)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(bytes) - 0x10000);
        }
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device.RegisterProcess(&process->GetMemory());
        device.BindInterface(&rasterizer);
        device.Map(0x10000, 0x10000, 0x10000, asid);
        memory.emplace(system, device, 32, 1ULL << 31, 16);
        memory->BindRasterizer(&rasterizer);
        memory->Map(Base, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
        compute.emplace(*memory);
        compute->SetNativeMacroMode(&macros, {});
        auto& qmd = compute->launch_description;
        qmd.const_buffer_enable_mask.Assign(1);
        qmd.const_buffer_config[0].address_low = static_cast<u32>(Base + ConstantsOffset);
        qmd.const_buffer_config[0].size.Assign(0x1000);
    }

    ~NativeComputeFixture() {
        compute.reset();
        memory.reset();
        device.Unmap(0x10000, 0x10000);
        device.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(process.get());
    }

    /// Places a program at this offset into the mapped range and returns its address
    GPUVAddr Place(u64 offset, std::span<const Instruction> native) {
        std::memcpy(bytes + offset, native.data(), native.size_bytes());
        return Base + offset;
    }

    void SetConstant(u32 offset, u32 value) {
        std::memcpy(bytes + ConstantsOffset + offset, &value, sizeof(value));
    }

    VideoCommon::ComputeEnvironment Environment(GPUVAddr address, V2::Cache* cache) {
        return VideoCommon::ComputeEnvironment{*compute, *memory, address, 0, {}, cache};
    }
};

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

} // Anonymous namespace

TEST_CASE("Shader cache v2: compute environments take recorded programs with every check",
          "[video_core][shader_cache_v2]") {
    std::vector<TestProgram> programs = TestPrograms();
    TestProgram& traps = programs[6];
    TestProgram& texture = programs[8];
    TestProgram rejected{"rejected", Shader::Stage::Compute, {{{0x00007918, 0, 0, 0}}, Exit}, {}};
    // EDEN_NX2_REJECT_HASH is read at the first lowering in the process: no other test lowers
    // in an environment.
    SetEnvironment("EDEN_NX2_REJECT_HASH",
                   fmt::format("{:016x}", Shader::Ampere::NativeProgramKey(rejected.native)));

    NativeComputeFixture fixture;
    traps.inputs.native_code_address = fixture.Place(0x1000, traps.native);
    texture.inputs.native_code_address = fixture.Place(0x2000, texture.native);
    rejected.inputs.native_code_address = fixture.Place(0x3000, rejected.native);
    fixture.SetConstant(0, 5); // The trap program's guard holds.

    // A session without records: the environment lowers.
    auto lowered_traps = fixture.Environment(traps.inputs.native_code_address, nullptr);
    auto lowered_texture = fixture.Environment(texture.inputs.native_code_address, nullptr);
    REQUIRE_THROWS_AS(fixture.Environment(rejected.inputs.native_code_address, nullptr),
                      Shader::RuntimeError);
    RequireSameCode(*lowered_traps.LoweredNative(), *LowerFresh(traps));
    RequireSameCode(*lowered_texture.LoweredNative(), *LowerFresh(texture));

    TempDir temp;
    const auto file = temp.path / "cache.bin";
    const std::array records{RecordOf(traps, *lowered_traps.LoweredNative()),
                             RecordOf(texture, *lowered_texture.LoweredNative()),
                             RecordOf(rejected, *LowerFresh(rejected))};
    WriteLog(file, records);
    const V2::LogFile log = V2::ReadLogFile(file);
    V2::Cache cache{file};
    cache.LoadPrograms(log.parsed.live);

    // The next session: the environments take the recorded code, the same code they lowered.
    const auto loaded = [&](const TestProgram& program) {
        const auto sph = program.Header();
        const auto program_traps = program.Traps();
        return cache.FindProgram(KeyInputs(program, sph, program_traps)).code;
    };
    auto recorded_traps = fixture.Environment(traps.inputs.native_code_address, &cache);
    auto recorded_texture = fixture.Environment(texture.inputs.native_code_address, &cache);
    REQUIRE(recorded_traps.LoweredNative() == loaded(traps));
    REQUIRE(recorded_texture.LoweredNative() == loaded(texture));
    RequireSameCode(*recorded_traps.LoweredNative(), *lowered_traps.LoweredNative());
    RequireSameCode(*recorded_texture.LoweredNative(), *lowered_texture.LoweredNative());
    REQUIRE(recorded_traps.Analyze() == lowered_traps.Analyze());
    REQUIRE(recorded_texture.Analyze() == lowered_texture.Analyze());
    REQUIRE(recorded_texture.TextureBoundBuffer() == 2);
    REQUIRE(recorded_traps.SourceSizeBytes() == lowered_traps.SourceSizeBytes());
    REQUIRE(recorded_texture.TakeProgramRecord() == nullptr);

    // Trap guards are checked against the constants of this dispatch.
    fixture.SetConstant(0, 0x80000000);
    REQUIRE_THROWS_AS(fixture.Environment(traps.inputs.native_code_address, &cache),
                      Shader::Ampere::TrapGuardError);
    fixture.SetConstant(0, 5);

    // Rejections by native key and by lowered hash still apply.
    REQUIRE(loaded(rejected) != nullptr);
    REQUIRE_THROWS_AS(fixture.Environment(rejected.inputs.native_code_address, &cache),
                      Shader::RuntimeError);
    const auto texture_code = loaded(texture);
    for (const u64 hash : {texture_code->native_program_key,
                           VideoCommon::NativeShaderHash(texture_code->code)}) {
        const std::array hashes{hash};
        (void)VideoCore::Nx2Debug::SetRejected(hashes, true);
        REQUIRE_THROWS_AS(fixture.Environment(texture.inputs.native_code_address, &cache),
                          Shader::RuntimeError);
        (void)VideoCore::Nx2Debug::SetRejected(hashes, false);
    }
    REQUIRE_FALSE(VideoCore::Nx2Debug::AnyRejected());
    REQUIRE(fixture.Environment(texture.inputs.native_code_address, &cache).LoweredNative() ==
            texture_code);
}
