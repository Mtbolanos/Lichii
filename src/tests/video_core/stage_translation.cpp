// SPDX-License-Identifier: GPL-3.0-or-later

// Shader cache v2, run-time misses: stages translated at once on shader job workers, and compute
// SPIR-V emitted by a job, give what the GPU thread gives. Built on its own so it can run under
// ThreadSanitizer (see CMakeLists.txt).

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/shader_jobs.h"
#include "video_core/shader_stage_translation.h"

namespace {

using VideoCommon::ShaderPools;
using VideoCommon::StageTranslation;

constexpr u64 MakeCbufKey(u32 index, u32 offset) {
    return (static_cast<u64>(index) << 32) | offset;
}

constexpr u64 MakeIndirectCbufKey(u32 index, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (static_cast<u64>(indirect & 0x3fffff) << 40) |
           MakeCbufKey(index, offset);
}

/// A captured environment of tools/shader-sweep/fixtures. It answers as FileEnvironment answers
/// from the record it was taken from (recorded values only), without the rest of video_core.
class FixtureEnvironment final : public Shader::Environment {
public:
    explicit FixtureEnvironment(const nlohmann::json& env) {
        const auto& parameters = env.at("parameters");
        local_memory_size = parameters.at(0);
        texture_bound = parameters.at(1);
        start_address = parameters.at(2);
        read_lowest = parameters.at(3);
        read_highest = parameters.at(4);
        viewport_transform_state = parameters.at(5);
        stage = static_cast<Shader::Stage>(parameters.at(6).get<u32>());
        for (const auto& word : env.at("code")) {
            code.push_back(std::stoull(word.get<std::string>(), nullptr, 16));
        }
        for (const auto& entry : env.at("texture_types")) {
            texture_types.emplace(entry.at(0), static_cast<Shader::TextureType>(entry.at(1)));
        }
        for (const auto& entry : env.at("texture_formats")) {
            texture_formats.emplace(entry.at(0),
                                    static_cast<Shader::TexturePixelFormat>(entry.at(1)));
        }
        for (const auto& entry : env.at("cbuf_values")) {
            cbuf_values.emplace(entry.at(0), entry.at(1));
        }
        cbuf_values.erase(MakeCbufKey(~0U, 0)); // The record's read properties
        for (const auto& entry : env.at("storage_pointers")) {
            storage_pointers.emplace(entry.at(0), entry.at(1));
        }
        for (const auto& entry : env.at("replacements")) {
            replacements.emplace(entry.at(0), static_cast<Shader::ReplaceConstant>(entry.at(1)));
        }
        const std::string tail_hex = env.at("tail");
        std::vector<u8> tail;
        for (size_t i = 0; i + 1 < tail_hex.size(); i += 2) {
            tail.push_back(static_cast<u8>(std::stoul(tail_hex.substr(i, 2), nullptr, 16)));
        }
        size_t offset = 0;
        const auto read = [&](auto& value) {
            REQUIRE(offset + sizeof(value) <= tail.size());
            std::memcpy(&value, tail.data() + offset, sizeof(value));
            offset += sizeof(value);
        };
        if (stage == Shader::Stage::Compute) {
            read(workgroup_size);
            read(shared_memory_size);
        } else {
            read(sph);
            if (stage == Shader::Stage::Geometry) {
                read(gp_passthrough_mask);
            }
        }
        is_proprietary_driver = texture_bound == 2 || IsNativeProgram();
    }

    /// Where PipelineCache starts the control flow graph of this stage
    u32 CfgOffset() const {
        return start_address +
               (stage == Shader::Stage::Compute ? 0 : u32{sizeof(Shader::ProgramHeader)});
    }

    u64 ReadInstruction(u32 address) override {
        if (address < read_lowest || address > read_highest) {
            throw Shader::LogicError("Out of bounds address {}", address);
        }
        return code[(address - read_lowest) / sizeof(u64)];
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) {
            return *translated;
        }
        if (Shader::Ampere::HasNativeAddressMap(code)) {
            throw Shader::RuntimeError("Invalid cached native indirect branch address {:#x}",
                                       address);
        }
        return address;
    }
    bool IsNativeProgram() const override {
        return Shader::Ampere::HasNativeAddressMap(code);
    }
    bool HasNativeInstructions() const override {
        return IsNativeProgram();
    }
    bool IsCbufStorageBufferPointer(u32 index, u32 offset) override {
        return Lookup(storage_pointers, MakeCbufKey(index, offset)) != 0;
    }
    bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) override {
        return Lookup(storage_pointers, MakeIndirectCbufKey(index, offset, indirect)) != 0;
    }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        return Lookup(cbuf_values, MakeCbufKey(index, offset));
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return Lookup(texture_types, handle);
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return Lookup(texture_formats, handle);
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        using F = Shader::TexturePixelFormat;
        switch (ReadTexturePixelFormat(handle)) {
        case F::A8B8G8R8_SINT:
        case F::A8B8G8R8_UINT:
        case F::A2B10G10R10_UINT:
        case F::R8_SINT:
        case F::R8_UINT:
        case F::R16G16B16A16_SINT:
        case F::R16G16B16A16_UINT:
        case F::R32G32B32A32_UINT:
        case F::R32G32B32A32_SINT:
        case F::R32G32_SINT:
        case F::R16_UINT:
        case F::R16_SINT:
        case F::R16G16_UINT:
        case F::R16G16_SINT:
        case F::R8G8_SINT:
        case F::R8G8_UINT:
        case F::R32G32_UINT:
        case F::R32_UINT:
        case F::R32_SINT:
            return true;
        default:
            return false;
        }
    }
    u32 ReadViewportTransformState() override {
        return viewport_transform_state;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return local_memory_size;
    }
    u32 SharedMemorySize() const override {
        return shared_memory_size;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return workgroup_size;
    }
    bool HasHLEMacroState() const override {
        return !replacements.empty();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        const auto it = replacements.find(MakeCbufKey(bank, offset));
        return it == replacements.end() ? std::nullopt : std::optional{it->second};
    }
    void Dump(u64, u64) override {}

private:
    template <typename Map>
    static typename Map::mapped_type Lookup(const Map& map, typename Map::key_type key) {
        const auto it = map.find(key);
        if (it == map.end()) {
            throw Shader::LogicError("Unrecorded environment value {:#x}", u64{key});
        }
        return it->second;
    }

    std::vector<u64> code;
    std::unordered_map<u32, Shader::TextureType> texture_types;
    std::unordered_map<u32, Shader::TexturePixelFormat> texture_formats;
    std::unordered_map<u64, u32> cbuf_values;
    std::unordered_map<u64, u8> storage_pointers;
    std::unordered_map<u64, Shader::ReplaceConstant> replacements;
    u32 local_memory_size{};
    u32 texture_bound{};
    u32 read_lowest{};
    u32 read_highest{};
    u32 viewport_transform_state{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};
};

struct Fixture {
    std::string name;
    nlohmann::json environment;
};

std::vector<Fixture> LoadFixtures() {
    std::vector<Fixture> fixtures;
    for (const auto& entry : std::filesystem::directory_iterator{SHADER_SWEEP_FIXTURES}) {
        if (entry.path().extension() != ".json") {
            continue;
        }
        std::ifstream file{entry.path()};
        fixtures.push_back(
            {entry.path().stem().string(), nlohmann::json::parse(file).at("environment")});
    }
    std::ranges::sort(fixtures, {}, &Fixture::name);
    REQUIRE(fixtures.size() >= 4);
    return fixtures;
}

/// The offline host and profile of tools/shader-sweep
Shader::HostTranslateInfo Host() {
    Shader::HostTranslateInfo host{};
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_conditional_barrier = true;
    host.support_viewport_index_layer = true;
    host.support_geometry_shader_passthrough = true;
    host.ApplyDescriptorLimitPolicy();
    return host;
}

Shader::Profile Profile(const Shader::HostTranslateInfo& host) {
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010400;
    profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = host.min_ssbo_alignment;
    profile.support_int64 = true;
    profile.support_int16 = true;
    profile.support_int8 = true;
    profile.support_vote = true;
    profile.support_shuffle_relative = true;
    profile.support_geometry_streams = true;
    profile.support_geometry_shader_passthrough = true;
    profile.support_viewport_index_layer_non_geometry = true;
    profile.support_multi_viewport = true;
    profile.max_user_clip_distances = 8;
    profile.max_subgroup_size = 32;
    profile.support_typeless_image_loads = true;
    return profile;
}

/// What translating and emitting a stage gave: its SPIR-V and Info, or what it threw
struct Result {
    std::vector<u32> spirv;
    Shader::Info info;
    std::string error;

    bool operator==(const Result&) const = default;
};

Result Emit(const Shader::Profile& profile, Shader::IR::Program program) {
    Result result;
    try {
        Shader::Backend::Bindings bindings;
        result.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, {}, program, bindings);
        result.info = program.info;
    } catch (const std::exception& exception) {
        result.error = exception.what();
    }
    return result;
}

/// As PipelineCache::CreateGraphicsPipeline translates a stage on the GPU thread
Result TranslateHere(const Fixture& fixture, const Shader::HostTranslateInfo& host,
                     const Shader::Profile& profile) {
    FixtureEnvironment env{fixture.environment};
    ShaderPools pools;
    try {
        Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.CfgOffset()};
        return Emit(profile,
                    Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg, host));
    } catch (const std::exception& exception) {
        return {.error = exception.what()};
    }
}

VideoCommon::ShaderJobConfig JobConfig(size_t cpu_threads) {
    return {.cpu_threads = cpu_threads, .driver_threads = 1, .name = "StageTest"};
}

} // Anonymous namespace

TEST_CASE("Stages translated at once match the GPU thread's translation", "[stage_translation]") {
    const auto fixtures = LoadFixtures();
    const auto host = Host();
    const auto profile = Profile(host);
    std::vector<Result> expected;
    for (const Fixture& fixture : fixtures) {
        expected.push_back(TranslateHere(fixture, host, profile));
        INFO(fixture.name << ": " << expected.back().error);
        CHECK(expected.back().error.empty());
        CHECK(!expected.back().spirv.empty());
    }

    // Each fixture several times over, so copies of one program translate at once as well
    constexpr size_t Copies = 6;
    VideoCommon::ShaderJobScheduler jobs{JobConfig(4)};
    for (int round = 0; round < 3; ++round) {
        std::vector<std::unique_ptr<FixtureEnvironment>> envs;
        std::vector<std::unique_ptr<ShaderPools>> pools;
        std::vector<StageTranslation> stages;
        for (size_t copy = 0; copy < Copies; ++copy) {
            for (const Fixture& fixture : fixtures) {
                auto& env =
                    envs.emplace_back(std::make_unique<FixtureEnvironment>(fixture.environment));
                stages.push_back({
                    .env = env.get(),
                    .pools = pools.emplace_back(std::make_unique<ShaderPools>()).get(),
                    .cfg_offset = env->CfgOffset(),
                });
            }
        }
        VideoCommon::TranslateStages(jobs, stages, host);
        for (size_t index = 0; index < stages.size(); ++index) {
            const Fixture& fixture = fixtures[index % fixtures.size()];
            INFO(fixture.name << " copy " << index / fixtures.size() << " round " << round);
            CHECK(stages[index].time.count() > 0);
            CHECK(Emit(profile, stages[index].Take()) == expected[index % fixtures.size()]);
        }
    }
}

TEST_CASE("A stage that fails to translate rethrows on the calling thread", "[stage_translation]") {
    const auto fixtures = LoadFixtures();
    const auto host = Host();
    VideoCommon::ShaderJobScheduler jobs{JobConfig(2)};
    // A record without its code: the first instruction read is out of bounds
    nlohmann::json empty = fixtures.front().environment;
    empty["parameters"][4] = 0;
    empty["code"] = nlohmann::json::array({"0"});
    FixtureEnvironment broken{empty};
    FixtureEnvironment good{fixtures.front().environment};
    ShaderPools broken_pools, good_pools;
    std::array<StageTranslation, 2> stages{{
        {.env = &broken, .pools = &broken_pools, .cfg_offset = broken.CfgOffset()},
        {.env = &good, .pools = &good_pools, .cfg_offset = good.CfgOffset()},
    }};
    VideoCommon::TranslateStages(jobs, stages, host);
    CHECK_THROWS_AS(stages[0].Take(), Shader::LogicError);
    CHECK(!stages[1].Take().blocks.empty());
}

TEST_CASE("Stages a shut down scheduler drops fail instead of hanging", "[stage_translation]") {
    const auto fixtures = LoadFixtures();
    const auto host = Host();
    VideoCommon::ShaderJobScheduler jobs{JobConfig(2)};
    jobs.Shutdown();
    std::array<std::unique_ptr<FixtureEnvironment>, 2> envs{
        std::make_unique<FixtureEnvironment>(fixtures.front().environment),
        std::make_unique<FixtureEnvironment>(fixtures.front().environment)};
    std::array<ShaderPools, 2> pools;
    std::array<StageTranslation, 2> stages{{
        {.env = envs[0].get(), .pools = &pools[0], .cfg_offset = envs[0]->CfgOffset()},
        {.env = envs[1].get(), .pools = &pools[1], .cfg_offset = envs[1]->CfgOffset()},
    }};
    VideoCommon::TranslateStages(jobs, stages, host);
    for (StageTranslation& stage : stages) {
        CHECK_THROWS_AS(stage.Take(), Shader::RuntimeError);
    }
}

TEST_CASE("Compute SPIR-V emitted by a job matches the GPU thread's", "[stage_translation]") {
    const auto fixtures = LoadFixtures();
    const auto host = Host();
    const auto profile = Profile(host);
    VideoCommon::ShaderJobScheduler jobs{JobConfig(4)};
    size_t compute_fixtures = 0;
    for (const Fixture& fixture : fixtures) {
        if (FixtureEnvironment{fixture.environment}.ShaderStage() != Shader::Stage::Compute) {
            continue;
        }
        ++compute_fixtures;
        INFO(fixture.name);
        const auto translate = [&](ShaderPools& pools) {
            FixtureEnvironment env{fixture.environment};
            Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};
            return Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg, host);
        };
        // Without shader cache v2 the GPU thread emits it
        ShaderPools here_pools;
        auto here = translate(here_pools);
        const std::vector<u32> expected = Shader::Backend::SPIRV::EmitSPIRV(profile, here);

        // With it the program and its pools move into the CPU stage of the pipeline's job; with
        // EDEN_NX2_PERF_VERIFY the GPU thread's SPIR-V goes along as the reference
        for (const bool verify : {false, true}) {
            struct Deferred {
                std::shared_ptr<ShaderPools> pools;
                Shader::IR::Program program;
                std::vector<u32> reference;
            };
            auto deferred = std::make_shared<Deferred>();
            deferred->pools = std::make_shared<ShaderPools>();
            deferred->program = translate(*deferred->pools);
            if (verify) {
                deferred->reference = Vulkan::EmitComputeShader(profile, deferred->program, 1);
            }
            std::vector<u32> emitted;
            const VideoCommon::ShaderJobHandle job = jobs.Submit({
                .priority = VideoCommon::ShaderJobPriority::Soon,
                .cpu = [&emitted, &profile, deferred](VideoCommon::ShaderJobContext&) mutable
                    -> VideoCommon::ShaderDriverStage {
                    emitted = Vulkan::EmitComputeShader(profile, deferred->program, 1,
                                                        deferred->reference);
                    deferred.reset();
                    return {};
                },
            });
            deferred.reset();
            job.Wait();
            REQUIRE(job.Status() == VideoCommon::ShaderJobStatus::Done);
            CHECK(emitted == expected);
        }

        // A job's SPIR-V that differs from the reference is replaced by it
        ShaderPools pools;
        auto program = translate(pools);
        const std::vector<u32> reference{1, 2, 3};
        CHECK(Vulkan::EmitComputeShader(profile, program, 1, reference) == reference);
    }
    CHECK(compute_fixtures > 0);
}

TEST_CASE("Texture descriptor pools up to 4 MiB are flushed for translation on workers",
          "[stage_translation]") {
    using VideoCommon::CanFlushTexturePool;
    using VideoCommon::MaxFlushedTexturePoolBytes;
    using VideoCommon::TexturePoolBytes;

    // Pools under the old 1 MiB cap still flush
    CHECK(TexturePoolBytes(0) == 32);
    CHECK(CanFlushTexturePool(0));
    CHECK(TexturePoolBytes(32767) == (u64{1} << 20));
    CHECK(CanFlushTexturePool(32767));

    // Ocarina of Time's pool of 33,068 descriptors, just over 1 MiB
    CHECK(TexturePoolBytes(33067) == 1'058'176);
    CHECK(CanFlushTexturePool(33067));

    // The cap is 4 MiB, it is inclusive, and the largest limit does not overflow
    CHECK(MaxFlushedTexturePoolBytes == (u64{4} << 20));
    constexpr u32 last_index = static_cast<u32>(MaxFlushedTexturePoolBytes / 32 - 1);
    CHECK(TexturePoolBytes(last_index) == MaxFlushedTexturePoolBytes);
    CHECK(CanFlushTexturePool(last_index));
    CHECK_FALSE(CanFlushTexturePool(last_index + 1));
    CHECK(TexturePoolBytes(0xFFFFFFFF) == (u64{1} << 37));
    CHECK_FALSE(CanFlushTexturePool(0xFFFFFFFF));
}

TEST_CASE("A texture descriptor pool's last byte is found without wrapping",
          "[stage_translation]") {
    using VideoCommon::TexturePoolLastAddress;

    // A 2 MiB pool starting 1 MiB before the end of a 40-bit address space ends past it
    constexpr u64 space_end = u64{1} << 40;
    constexpr u32 two_mib_limit = (u32{2} << 20) / 32 - 1;
    CHECK(TexturePoolLastAddress(space_end - (u64{1} << 20), two_mib_limit) ==
          space_end + (u64{1} << 20) - 1);
    CHECK(TexturePoolLastAddress(space_end - (u64{2} << 20), two_mib_limit) == space_end - 1);

    // One descriptor at address 0 ends at byte 31
    CHECK(TexturePoolLastAddress(0, 0) == 31);

    // The last byte of the 64-bit range is reachable, one past it wraps
    CHECK(TexturePoolLastAddress(~u64{0} - 31, 0) == ~u64{0});
    CHECK_FALSE(TexturePoolLastAddress(~u64{0} - 30, 0).has_value());
}
