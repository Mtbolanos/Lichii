// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/cityhash.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/ampere/trap_guards.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#ifdef SHADER_SWEEP_V2
#include "common/unique_function.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_cache_v2.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"
#endif

namespace {
using Json = nlohmann::json;
using Shader::Stage;
using Shader::Ampere::Instruction;

struct MissingContext : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Serialized environment layouts this tool reads. 54 has 53's layout (see
// CompatibleVulkanPipelineCacheVersions in video_core/pipeline_cache_files.h).
constexpr u32 OldestCacheVersion = 46;
constexpr u32 NewestCacheVersion = 54;

constexpr std::array StageNames{"vertex",  "tess-control", "tess-eval", "geometry", "fragment",
                                "compute", "vertex-a",     "task",      "mesh"};

Stage ParseStage(std::string_view name) {
    for (size_t i = 0; i < StageNames.size(); ++i) {
        if (name == StageNames[i])
            return static_cast<Stage>(i);
    }
    throw std::runtime_error("Unknown stage: " + std::string{name});
}

std::vector<u8> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path.string());
    const auto size = file.tellg();
    if (size < 0 || size > 64 * 1024 * 1024) {
        throw std::runtime_error("Input exceeds the 64 MiB limit");
    }
    std::vector<u8> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return bytes;
}

// The same path, or another link to the same existing file (a hard link passes the path check).
bool SameFile(const std::filesystem::path& a, const std::filesystem::path& b) {
    if (std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b))
        return true;
    std::error_code error;
    return std::filesystem::equivalent(a, b, error) && !error;
}

// Range-checked before narrowing, so a value wider than 32 bits cannot wrap into a valid one.
u32 ParseOptionValue(const std::string& option, const std::string& value, int base, u64 max) {
    const u64 parsed = std::stoull(value, nullptr, base);
    if (parsed > max)
        throw std::runtime_error("Invalid " + option);
    return static_cast<u32>(parsed);
}

class Reader {
public:
    explicit Reader(std::span<const u8> bytes_) : bytes{bytes_} {}

    template <typename T>
    T Read() {
        T value;
        Copy(&value, sizeof(value));
        return value;
    }

    void Copy(void* destination, size_t size) {
        if (size > bytes.size())
            throw std::runtime_error("Truncated environment record");
        std::memcpy(destination, bytes.data(), size);
        bytes = bytes.subspan(size);
    }

    size_t Remaining() const {
        return bytes.size();
    }

private:
    std::span<const u8> bytes;
};

template <typename K, typename V>
V Lookup(const std::map<K, V>& values, K key, std::string_view name) {
    const auto it = values.find(key);
    if (it == values.end()) {
        throw MissingContext(fmt::format("Uncaptured {} at {:#x}", name, key));
    }
    return it->second;
}

u64 CbufKey(u32 bank, u32 offset) {
    return (u64{bank} << 32) | offset;
}

u64 IndirectCbufKey(u32 bank, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (u64{indirect & 0x3fffff} << 40) | CbufKey(bank, offset);
}

// --guess-missing: values used for reads the environment has no record of. The defaults
// equal the emulator's warm-up loader guesses.
struct Guesses {
    bool enabled{};
    u32 cbuf0{0x10000000};
    u32 cbuf_other{0};
    u32 texture_type{3};
    u32 pixel_format{16};
    bool storage_pointer{true};       // bank 0
    bool storage_pointer_other{false}; // banks 1..17 (GuessMissing::storage_pointer_other)
};

class OfflineEnvironment final : public Shader::Environment {
public:
    Guesses guesses;
    u32 guessed_reads{};

    void Load(Reader& reader, u32 version) {
        const u64 size = reader.Read<u64>();
        const u64 type_count = reader.Read<u64>();
        const u64 format_count = reader.Read<u64>();
        const u64 cbuf_count = reader.Read<u64>();
        const u64 pointer_count = version >= 47 ? reader.Read<u64>() : 0;
        const u64 replacement_count = reader.Read<u64>();
        if (size % 8 || !size || size > reader.Remaining()) {
            throw std::runtime_error("Invalid environment code size");
        }
        for (const u64 count :
             {type_count, format_count, cbuf_count, pointer_count, replacement_count}) {
            if (count > reader.Remaining() / 8)
                throw std::runtime_error("Invalid environment entry count");
        }
        local_memory = reader.Read<u32>();
        texture_bound = reader.Read<u32>();
        start_address = reader.Read<u32>();
        low = reader.Read<u32>();
        high = reader.Read<u32>();
        viewport = reader.Read<u32>();
        const u32 stage_value = reader.Read<u32>();
        if (stage_value >= StageNames.size())
            throw std::runtime_error("Invalid cached stage");
        stage = static_cast<Stage>(stage_value);
        code.resize(size / 8);
        reader.Copy(code.data(), size);
        if (low % 8 || high < low || u64{high} - low + 8 != size) {
            throw std::runtime_error("Invalid cached instruction bounds");
        }
        for (u64 i = 0; i < type_count; ++i) {
            const u32 key = reader.Read<u32>();
            const u32 type = reader.Read<u32>();
            if (type >= Shader::NUM_TEXTURE_TYPES)
                throw std::runtime_error("Invalid texture type");
            texture_types.emplace(key, static_cast<Shader::TextureType>(type));
        }
        for (u64 i = 0; i < format_count; ++i) {
            const u32 key = reader.Read<u32>();
            const u32 value = reader.Read<u32>();
            if (value > std::numeric_limits<u8>::max()) {
                throw std::runtime_error("Invalid texture pixel format");
            }
            texture_formats.emplace(key, static_cast<Shader::TexturePixelFormat>(value));
        }
        for (u64 i = 0; i < cbuf_count; ++i) {
            const u64 key = reader.Read<u64>();
            cbuf_values.emplace(key, reader.Read<u32>());
        }
        for (u64 i = 0; i < pointer_count; ++i) {
            const u64 key = reader.Read<u64>();
            const u8 value = reader.Read<u8>();
            if (value > 1)
                throw std::runtime_error("Invalid storage pointer classification");
            pointers.emplace(key, value);
        }
        for (u64 i = 0; i < replacement_count; ++i) {
            const u64 key = reader.Read<u64>();
            const u32 value = reader.Read<u32>();
            if (value > 2)
                throw std::runtime_error("Invalid constant replacement");
            replacements.emplace(key, static_cast<Shader::ReplaceConstant>(value));
        }
        if (stage == Stage::Compute) {
            workgroup = reader.Read<std::array<u32, 3>>();
            shared_memory = reader.Read<u32>();
            if (!workgroup[0] || !workgroup[1] || !workgroup[2]) {
                throw std::runtime_error("Invalid cached workgroup dimensions");
            }
        } else {
            sph = reader.Read<Shader::ProgramHeader>();
            if (stage == Stage::Geometry)
                gp_passthrough_mask = reader.Read<std::array<u32, 8>>();
            if (stage == Stage::Task || stage == Stage::Mesh) {
                if (version < 53)
                    throw std::runtime_error("Task/mesh environments require cache version 53");
                mesh_registers = reader.Read<std::array<u32, 3>>();
                mesh_primitive_header = reader.Read<Shader::ProgramHeader>();
            }
        }
        is_proprietary_driver = texture_bound == 2 || IsNativeProgram();
        captured = true;
    }

    void SetStage(Stage value) {
        stage = value;
    }

    void SetHeader(std::span<const u8> bytes) {
        std::memcpy(&sph, bytes.data(), sizeof(sph));
        const u64 size = sph.LocalMemorySize() + sph.common3.shader_local_memory_crs_size;
        if (size > std::numeric_limits<u32>::max())
            throw std::runtime_error("Local memory overflow");
        local_memory = static_cast<u32>(size);
    }

    void SetCode(const Shader::Ampere::LoweredProgram& lowered) {
        code.resize(sizeof(sph) / sizeof(u64));
        std::memcpy(code.data(), &sph, sizeof(sph));
        for (const u64 instruction : lowered.instructions) {
            if (code.size() % 4 == 0)
                code.push_back(0);
            code.push_back(instruction);
        }
        code.push_back(0);
        Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
        low = 0;
        high = static_cast<u32>(code.size() * 8 - 8);
        start_address = stage == Stage::Compute ? sizeof(sph) : 0;
        if (lowered.required_texture_buffer != ~0U)
            texture_bound = lowered.required_texture_buffer;
        is_proprietary_driver = true;
    }

    u32 EntryPoint() const {
        return start_address + (stage == Stage::Compute ? 0 : sizeof(sph));
    }

    u64 ReadInstruction(u32 address) override {
        if (address < low || address > high || (address - low) % 8) {
            throw std::runtime_error(
                fmt::format("Instruction address outside capture: {:#x}", address));
        }
        return code.at((address - low) / 8);
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (!IsNativeProgram())
            return address;
        const auto result = Shader::Ampere::NativeProgramAddress(code, address);
        if (!result)
            throw std::runtime_error("Invalid native indirect branch target");
        return *result;
    }
    bool IsNativeProgram() const override {
        return Shader::Ampere::HasNativeAddressMap(code);
    }
    bool HasNativeInstructions() const override {
        return IsNativeProgram();
    }
    /// Recorded storage pointer classification, or the bank's guess when guessing is on.
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        return Find(pointers, CbufKey(bank, offset), "storage pointer classification",
                    u8{bank == 0 ? guesses.storage_pointer : guesses.storage_pointer_other}) != 0;
    }
    /// Recorded storage pointer table classification, or the bank's guess when guessing is on.
    bool IsCbufIndirectStorageBufferPointer(u32 bank, u32 offset, u32 indirect) override {
        return Find(pointers, IndirectCbufKey(bank, offset, indirect),
                    "storage pointer table classification",
                    u8{bank == 0 ? guesses.storage_pointer : guesses.storage_pointer_other}) != 0;
    }
    /// Recorded constant buffer value, or the bank's guess when guessing is on.
    u32 ReadCbufValue(u32 bank, u32 offset) override {
        return Find(cbuf_values, CbufKey(bank, offset), "constant buffer value",
                    bank == 0 ? guesses.cbuf0 : guesses.cbuf_other);
    }
    /// Recorded constant buffer value only, never a guess: throws MissingContext when absent.
    u32 RecordedCbufValue(u32 bank, u32 offset) const {
        return Lookup(cbuf_values, CbufKey(bank, offset), "constant buffer value");
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return Find(texture_types, handle, "texture type",
                    static_cast<Shader::TextureType>(guesses.texture_type));
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return Find(texture_formats, handle, "texture pixel format",
                    static_cast<Shader::TexturePixelFormat>(guesses.pixel_format));
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
        return viewport;
    }
    u32 TextureBoundBuffer() const override {
        if (texture_bound == ~0U)
            throw MissingContext("Uncaptured texture-bound constant buffer");
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return local_memory;
    }
    u32 SharedMemorySize() const override {
        return shared_memory;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return workgroup;
    }
    bool HasHLEMacroState() const override {
        return !replacements.empty();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        const auto it = replacements.find(CbufKey(bank, offset));
        return it == replacements.end() ? std::nullopt : std::optional{it->second};
    }
    void Dump(u64, u64) override {}

    bool captured{};
    std::vector<u64> code;

private:
    /// The recorded value for `key`. With guessing on, a missing value counts as a guessed read and
    /// returns `guess`; otherwise it throws.
    template <typename K, typename V>
    V Find(const std::map<K, V>& values, K key, std::string_view name, V guess) {
        if (guesses.enabled && !values.contains(key)) {
            ++guessed_reads;
            return guess;
        }
        return Lookup(values, key, name);
    }

    u32 low{}, high{}, local_memory{}, shared_memory{}, viewport{1}, texture_bound{~0U};
    std::array<u32, 3> workgroup{1, 1, 1};
    std::map<u32, Shader::TextureType> texture_types;
    std::map<u32, Shader::TexturePixelFormat> texture_formats;
    std::map<u64, u32> cbuf_values;
    std::map<u64, u8> pointers;
    std::map<u64, Shader::ReplaceConstant> replacements;
};

template <typename T>
void WriteFile(const std::filesystem::path& path, const std::vector<T>& words) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file ||
        !file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(T))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    file.close();
    if (!file)
        throw std::runtime_error("Cannot close " + path.string());
}

u32 KeyWord(std::span<const u8> key, size_t offset) {
    u32 value;
    std::memcpy(&value, key.data() + offset, sizeof(value));
    return value;
}

Shader::InputTopology InputTopology(u32 topology) {
    switch (topology) {
    case 0:
        return Shader::InputTopology::Points;
    case 1:
    case 2:
    case 3:
        return Shader::InputTopology::Lines;
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 14:
        return Shader::InputTopology::Triangles;
    case 10:
    case 11:
        return Shader::InputTopology::LinesAdjacency;
    case 12:
    case 13:
        return Shader::InputTopology::TrianglesAdjacency;
    default:
        throw std::runtime_error("Invalid pipeline input topology");
    }
}

void SetTessellation(u32 raw1, Shader::RuntimeInfo& info) {
    const u32 primitive = (raw1 >> 10) & 3, spacing = (raw1 >> 12) & 3;
    if (primitive > 2 || spacing > 2)
        throw std::runtime_error("Invalid tessellation state");
    info.tess_primitive = static_cast<Shader::TessPrimitive>(primitive);
    info.tess_spacing = static_cast<Shader::TessSpacing>(spacing);
    info.tess_clockwise = (raw1 >> 14) & 1;
}

void SetVertexInputTypes(std::span<const u8> key, Shader::RuntimeInfo& info) {
    const u32 raw1 = KeyWord(key, 48);
    for (size_t index = 0; index < info.generic_input_types.size(); ++index) {
        auto& type = info.generic_input_types[index];
        if (raw1 & (1U << 5)) {
            constexpr std::array types{
                Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                Shader::AttributeType::SignedInt, Shader::AttributeType::UnsignedInt};
            type = types[(KeyWord(key, 112 + (index / 16) * 4) >> ((index % 16) * 2)) & 3];
            continue;
        }
        const u32 attribute = KeyWord(key, 160 + index * 4);
        if (!(attribute & 1)) {
            type = Shader::AttributeType::Disabled;
            continue;
        }
        switch ((attribute >> 20) & 7) {
        case 1:
        case 2:
        case 7:
            type = Shader::AttributeType::Float;
            break;
        case 3:
            type = Shader::AttributeType::SignedInt;
            break;
        case 4:
            type = Shader::AttributeType::UnsignedInt;
            break;
        case 5:
            type = Shader::AttributeType::UnsignedScaled;
            break;
        case 6:
            type = Shader::AttributeType::SignedScaled;
            break;
        default:
            throw std::runtime_error("Invalid pipeline vertex attribute type");
        }
    }
}

Shader::RuntimeInfo Runtime(std::span<const u8> key, Stage stage, Json& assumptions) {
    Shader::RuntimeInfo info{};
    info.previous_stage_stores.mask.set();
    assumptions.push_back(
        "Sibling stage linkage is unavailable; all previous-stage varyings are enabled");
    if (key.empty()) {
        assumptions.push_back("No pipeline key: point input topology, isolines/equal tessellation, "
                              "float attributes, no alpha test or transform feedback");
        return info;
    }
    if (key.size() != 1056 || stage == Stage::Compute) {
        throw std::runtime_error("Expected a 1056-byte graphics pipeline key");
    }
    const auto word = [&](size_t offset) { return KeyWord(key, offset); };
    const u32 raw1 = word(48), raw2 = word(52);
    if (raw1 & (1U << 6))
        throw MissingContext("Transform feedback pipeline state is not supported offline");
    const u32 topology = (raw1 >> 24) & 15;
    info.input_topology = InputTopology(topology);
    if (stage == Stage::TessellationEval)
        SetTessellation(raw1, info);
    info.force_early_z = (raw2 >> 4) & 1;
    info.y_negate = (raw2 >> 11) & 1;
    if (stage == Stage::Geometry || (stage == Stage::VertexB && word(32) == 0 && word(36) == 0)) {
        info.convert_depth_mode = (raw1 >> 7) & 1;
        if (topology == 0)
            info.fixed_state_point_size = std::bit_cast<float>(word(76));
    }
    if (stage == Stage::Fragment) {
        info.alpha_test_func = static_cast<Shader::CompareFunction>((raw2 >> 1) & 7);
        info.alpha_test_reference = std::bit_cast<float>(word(72));
        info.dual_source_blend = (raw1 >> 21) & 1;
        for (size_t index = 0; index < info.color_output_types.size(); ++index) {
            switch (key[56 + index]) {
            case 0xc1:
            case 0xc4:
            case 0xc8:
            case 0xcc:
            case 0xd8:
            case 0xdc:
            case 0xe3:
            case 0xec:
            case 0xf0:
            case 0xf5:
                info.color_output_types[index] = Shader::AttributeType::SignedInt;
                break;
            case 0xc2:
            case 0xc5:
            case 0xc9:
            case 0xcd:
            case 0xd2:
            case 0xd9:
            case 0xdd:
            case 0xe4:
            case 0xed:
            case 0xf1:
            case 0xf6:
                info.color_output_types[index] = Shader::AttributeType::UnsignedInt;
                break;
            case 0:
            case 0xc0:
            case 0xc3:
            case 0xc6:
            case 0xc7:
            case 0xca:
            case 0xcb:
            case 0xce:
            case 0xcf:
            case 0xd0:
            case 0xd1:
            case 0xd5:
            case 0xd6:
            case 0xd7:
            case 0xda:
            case 0xdb:
            case 0xde:
            case 0xdf:
            case 0xe0:
            case 0xe5:
            case 0xe6:
            case 0xe7:
            case 0xe8:
            case 0xe9:
            case 0xea:
            case 0xeb:
            case 0xee:
            case 0xef:
            case 0xf2:
            case 0xf3:
            case 0xf4:
            case 0xf8:
            case 0xf9:
            case 0xfa:
                info.color_output_types[index] = Shader::AttributeType::Float;
                break;
            default:
                throw std::runtime_error("Invalid pipeline color attachment format");
            }
        }
    }
    if (stage == Stage::VertexB)
        SetVertexInputTypes(key, info);
    assumptions.push_back("Pipeline key supplies vertex/fragment interface types and fixed state; "
                          "integer fragment attachments use typed outputs on every offline host");
    return info;
}

// --device-profile: the HostTranslateInfo ("host") and Profile ("profile") fields that
// PipelineCache's constructor takes from a device, in place of the offline defaults.
Json device_profile = Json::object();

template <typename Struct>
void ApplyDeviceFields(std::string_view section, Struct& target, auto&& fields) {
    const auto overrides = device_profile.find(section);
    if (overrides == device_profile.end())
        return;
    size_t applied{};
    fields([&](std::string_view name, auto Struct::*field) {
        if (const auto value = overrides->find(name); value != overrides->end()) {
            value->get_to(target.*field);
            ++applied;
        }
    });
    if (applied != overrides->size())
        throw std::runtime_error(fmt::format("Unknown {} field in the device profile", section));
}

Shader::HostTranslateInfo OfflineHost() {
    using H = Shader::HostTranslateInfo;
    Shader::HostTranslateInfo host{};
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_conditional_barrier = true;
    host.support_viewport_index_layer = true;
    host.support_geometry_shader_passthrough = true;
    host.support_handle_table_arrays = true;
    ApplyDeviceFields("host", host, [](auto field) {
        field("min_ssbo_alignment", &H::min_ssbo_alignment);
        field("max_per_stage_descriptor_sampled_images",
              &H::max_per_stage_descriptor_sampled_images);
        field("max_per_stage_resources", &H::max_per_stage_resources);
        field("max_per_stage_descriptor_storage_buffers",
              &H::max_per_stage_descriptor_storage_buffers);
        field("max_per_stage_descriptor_samplers", &H::max_per_stage_descriptor_samplers);
        field("max_descriptor_set_samplers", &H::max_descriptor_set_samplers);
        field("max_descriptor_set_uniform_buffers", &H::max_descriptor_set_uniform_buffers);
        field("max_descriptor_set_uniform_buffers_dynamic",
              &H::max_descriptor_set_uniform_buffers_dynamic);
        field("max_descriptor_set_storage_buffers", &H::max_descriptor_set_storage_buffers);
        field("max_descriptor_set_storage_buffers_dynamic",
              &H::max_descriptor_set_storage_buffers_dynamic);
        field("max_descriptor_set_sampled_images", &H::max_descriptor_set_sampled_images);
        field("max_descriptor_set_storage_images", &H::max_descriptor_set_storage_images);
        field("max_descriptor_set_input_attachements", &H::max_descriptor_set_input_attachements);
        field("support_float64", &H::support_float64);
        field("support_float16", &H::support_float16);
        field("support_int64", &H::support_int64);
        field("needs_demote_reorder", &H::needs_demote_reorder);
        field("support_snorm_render_buffer", &H::support_snorm_render_buffer);
        field("support_viewport_index_layer", &H::support_viewport_index_layer);
        field("support_viewport_mask", &H::support_viewport_mask);
        field("support_geometry_shader_passthrough", &H::support_geometry_shader_passthrough);
        field("support_conditional_barrier", &H::support_conditional_barrier);
        field("single_lane_geometry_subgroups", &H::single_lane_geometry_subgroups);
        field("support_byte_addressed_r16", &H::support_byte_addressed_r16);
        field("support_handle_table_arrays", &H::support_handle_table_arrays);
    });
    host.ApplyDescriptorLimitPolicy();
    return host;
}

Shader::Profile OfflineProfile(const Shader::HostTranslateInfo& host) {
    using P = Shader::Profile;
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
    // Handle table arrays are indexed per invocation, as with Vulkan.
    profile.support_sampled_image_array_nonuniform_indexing = true;
    ApplyDeviceFields("profile", profile, [](auto field) {
        field("supported_spirv", &P::supported_spirv);
        field("support_descriptor_aliasing", &P::support_descriptor_aliasing);
        field("support_int8", &P::support_int8);
        field("support_uniform_and_storage_buffer_8bit",
              &P::support_uniform_and_storage_buffer_8bit);
        field("support_storage_buffer_8bit", &P::support_storage_buffer_8bit);
        field("support_int16", &P::support_int16);
        field("support_uniform_and_storage_buffer_16bit",
              &P::support_uniform_and_storage_buffer_16bit);
        field("support_storage_buffer_16bit", &P::support_storage_buffer_16bit);
        field("support_int64", &P::support_int64);
        field("support_float_controls", &P::support_float_controls);
        field("support_separate_denorm_behavior", &P::support_separate_denorm_behavior);
        field("support_separate_rounding_mode", &P::support_separate_rounding_mode);
        field("support_fp16_denorm_preserve", &P::support_fp16_denorm_preserve);
        field("support_fp32_denorm_preserve", &P::support_fp32_denorm_preserve);
        field("support_fp16_denorm_flush", &P::support_fp16_denorm_flush);
        field("support_fp32_denorm_flush", &P::support_fp32_denorm_flush);
        field("support_fp16_signed_zero_nan_preserve", &P::support_fp16_signed_zero_nan_preserve);
        field("support_fp32_signed_zero_nan_preserve", &P::support_fp32_signed_zero_nan_preserve);
        field("support_fp64_signed_zero_nan_preserve", &P::support_fp64_signed_zero_nan_preserve);
        field("support_explicit_workgroup_layout", &P::support_explicit_workgroup_layout);
        field("support_workgroup_layout_8bit_access", &P::support_workgroup_layout_8bit_access);
        field("support_workgroup_layout_16bit_access", &P::support_workgroup_layout_16bit_access);
        field("support_shader_quad_control", &P::support_shader_quad_control);
        field("support_quad_shuffles", &P::support_quad_shuffles);
        field("support_vote", &P::support_vote);
        field("support_shuffle_relative", &P::support_shuffle_relative);
        field("supported_subgroup_stages", &P::supported_subgroup_stages);
        field("support_viewport_index_layer_non_geometry",
              &P::support_viewport_index_layer_non_geometry);
        field("support_viewport_mask", &P::support_viewport_mask);
        field("support_typeless_image_loads", &P::support_typeless_image_loads);
        field("support_demote_to_helper_invocation", &P::support_demote_to_helper_invocation);
        field("support_int64_atomics", &P::support_int64_atomics);
        field("support_shared_int64_atomics", &P::support_shared_int64_atomics);
        field("support_derivative_control", &P::support_derivative_control);
        field("support_geometry_shader_passthrough", &P::support_geometry_shader_passthrough);
        field("support_native_ndc", &P::support_native_ndc);
        field("support_scaled_attributes", &P::support_scaled_attributes);
        field("support_multi_viewport", &P::support_multi_viewport);
        field("support_geometry_streams", &P::support_geometry_streams);
        field("support_sampled_image_array_nonuniform_indexing",
              &P::support_sampled_image_array_nonuniform_indexing);
        field("support_storage_image_array_nonuniform_indexing",
              &P::support_storage_image_array_nonuniform_indexing);
        field("support_uniform_texel_buffer_array_nonuniform_indexing",
              &P::support_uniform_texel_buffer_array_nonuniform_indexing);
        field("support_storage_texel_buffer_array_nonuniform_indexing",
              &P::support_storage_texel_buffer_array_nonuniform_indexing);
        field("select_global_memory_buffer", &P::select_global_memory_buffer);
        field("warp_size_potentially_larger_than_guest",
              &P::warp_size_potentially_larger_than_guest);
        field("need_gather_subpixel_offset", &P::need_gather_subpixel_offset);
        field("has_broken_spirv_fragment_shuffle", &P::has_broken_spirv_fragment_shuffle);
        field("max_compute_shared_memory_size", &P::max_compute_shared_memory_size);
        field("max_subgroup_size", &P::max_subgroup_size);
        field("max_user_clip_distances", &P::max_user_clip_distances);
    });
    return profile;
}

// Byte count GraphicsPipelineCacheKey::Hash() and operator== cover: the six unique hashes and
// FixedPipelineState::Size(). The offsets are those of the 1056-byte serialized key.
size_t GraphicsKeySize(std::span<const u8> key) {
    const u32 raw1 = KeyWord(key, 48);
    const bool extended_dynamic_state = raw1 & 1, extended_dynamic_state_3_blend = raw1 & (1U << 3),
               dynamic_vertex_input = raw1 & (1U << 5), xfb_enabled = raw1 & (1U << 6);
    if (xfb_enabled)
        return key.size();
    if (dynamic_vertex_input)
        return extended_dynamic_state_3_blend ? 128 : 160; // attachments, attributes
    return extended_dynamic_state ? 416 : 480;             // vertex_strides, xfb_state
}

// MakeRuntimeInfo in renderer_vulkan/vk_pipeline_cache.cpp for a device with the offline profile:
// geometry, viewport-layer and mesh shaders, not MoltenVK. Keep the two in step.
Shader::RuntimeInfo PipelineRuntime(std::span<const u8> key, std::span<const u64, 6> hashes,
                                    std::span<const Shader::IR::Program, 6> programs,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous) {
    Shader::RuntimeInfo info;
    if (previous) {
        info.previous_stage_stores = previous->info.stores;
        info.previous_stage_legacy_stores_mapping = previous->info.legacy_stores_mapping;
        if (previous->is_geometry_passthrough)
            info.previous_stage_stores.mask |= previous->info.passthrough.mask;
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous && previous->stage == Stage::Mesh) {
        for (const u32 line : previous->mesh.primitive_lines) {
            const auto attribute = static_cast<Shader::IR::Attribute>(line);
            if (Shader::IR::IsGeneric(attribute))
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
        }
    }
    const u32 raw1 = KeyWord(key, 48), raw2 = KeyWord(key, 52);
    const u32 topology = (raw1 >> 24) & 15;
    const bool xfb_enabled = (raw1 >> 6) & 1, gl_ndc = (raw1 >> 7) & 1;
    const bool has_geometry = hashes[4] != 0 && !programs[4].is_geometry_passthrough;
    const float point_size = std::bit_cast<float>(KeyWord(key, 76));
    if (xfb_enabled &&
        (program.stage == Stage::Geometry || (program.stage == Stage::VertexB && !has_geometry))) {
        throw MissingContext("Transform feedback pipeline state is not supported offline");
    }
    switch (program.stage) {
    case Stage::VertexB:
        if (!has_geometry) {
            if (topology == 0)
                info.fixed_state_point_size = point_size;
            info.convert_depth_mode = gl_ndc;
        }
        SetVertexInputTypes(key, info);
        break;
    case Stage::TessellationEval:
        SetTessellation(raw1, info);
        break;
    case Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0)
            info.fixed_state_point_size = point_size;
        break;
    case Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList)
            info.fixed_state_point_size = point_size;
        info.convert_depth_mode = gl_ndc;
        break;
    case Stage::Fragment:
        info.alpha_test_func = static_cast<Shader::CompareFunction>((raw2 >> 1) & 7);
        info.alpha_test_reference = std::bit_cast<float>(KeyWord(key, 72));
        info.dual_source_blend = (raw1 >> 21) & 1;
        break;
    default:
        break;
    }
    info.input_topology = InputTopology(topology);
    info.force_early_z = (raw2 >> 4) & 1;
    info.y_negate = (raw2 >> 11) & 1;
    const bool geometry_passthrough = hashes[4] != 0 && programs[4].is_geometry_passthrough;
    if (hashes[5] != 0 && !geometry_passthrough && !xfb_enabled &&
        !programs[5].info.loads_indexed_attributes) {
        const bool has_tess = hashes[3] != 0;
        const Stage stage = program.stage;
        const bool last_pre_raster = (stage == Stage::VertexB && !has_tess && !has_geometry) ||
                                     (stage == Stage::TessellationEval && !has_geometry) ||
                                     (stage == Stage::Geometry && has_geometry);
        if (stage == Stage::Fragment) {
            info.scaled_noperspective_inputs =
                previous &&
                (previous->stage == Stage::VertexB || previous->stage == Stage::TessellationEval ||
                 previous->stage == Stage::Geometry);
        }
        if (last_pre_raster) {
            const auto& fragment_info = programs[5].info;
            for (size_t index = 0; index < 32; ++index) {
                if (fragment_info.interpolation[index] == Shader::Interpolation::NoPerspective)
                    info.noperspective_outputs.set(index);
            }
        }
    }
    return info;
}

struct PipelineStage {
    u32 slot;
    Stage stage;
    std::vector<u32> spirv;
};

// The legacy path: translation and emission as PipelineCache::CreateGraphicsPipeline and
// CreateComputePipeline do them, one module per stage. Vertex A merges into vertex B and has no
// module of its own.
std::vector<PipelineStage> CompileLegacyPipeline(std::span<OfflineEnvironment> envs,
                                                 std::span<const u8> key, Json& diagnostic) {
    const Shader::HostTranslateInfo host = OfflineHost();
    const Shader::Profile profile = OfflineProfile(host);
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    std::vector<PipelineStage> stages;
    if (envs.front().ShaderStage() == Stage::Compute) {
        OfflineEnvironment& env = envs.front();
        diagnostic["phase"] = "translate";
        diagnostic["failed_slot"] = 0;
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{env.EntryPoint()}};
        auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
        diagnostic["phase"] = "spirv";
        stages.push_back({0, Stage::Compute, Shader::Backend::SPIRV::EmitSPIRV(profile, program)});
        diagnostic.erase("failed_slot");
        return stages;
    }
    std::array<u64, 6> hashes;
    std::memcpy(hashes.data(), key.data(), sizeof(hashes));
    std::array<Shader::IR::Program, 6> programs;
    const bool uses_vertex_a = hashes[0] != 0, uses_vertex_b = hashes[1] != 0;
    diagnostic["phase"] = "translate";
    size_t env_index = 0;
    for (u32 slot = 0; slot < hashes.size(); ++slot) {
        if (hashes[slot] == 0)
            continue;
        diagnostic["failed_slot"] = slot;
        OfflineEnvironment& env = envs[env_index++];
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{env.EntryPoint()},
                                       slot == 0};
        if (!uses_vertex_a || slot != 1) {
            programs[slot] =
                Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
        } else {
            auto vertex_b =
                Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
            programs[slot] =
                Shader::Maxwell::MergeDualVertexPrograms(programs[0], vertex_b, env, host);
        }
        stages.push_back({slot, env.ShaderStage(), {}});
    }
    // The runtime fits the arrays of every stage the pipeline emits to one descriptor set.
    std::array<Shader::IR::Program*, 6> emitted_programs{};
    size_t num_programs{};
    for (const PipelineStage& stage : stages) {
        if (stage.slot != 0 || !uses_vertex_b) {
            emitted_programs[num_programs++] = &programs[stage.slot];
        }
    }
    Shader::Optimization::LimitPipelineSampledTextureArrays(
        std::span{emitted_programs.data(), num_programs}, host);
    diagnostic["phase"] = "spirv";
    Shader::Backend::Bindings bindings;
    const Shader::IR::Program* previous = nullptr;
    for (PipelineStage& stage : stages) {
        if (stage.slot == 0 && uses_vertex_b)
            continue;
        diagnostic["failed_slot"] = stage.slot;
        if (stage.slot == 0)
            throw std::runtime_error("Vertex A program without a vertex B program");
        Shader::IR::Program& program = programs[stage.slot];
        auto runtime = PipelineRuntime(key, hashes, programs, program, previous);
        Shader::Maxwell::ConvertLegacyToGeneric(program, runtime);
        stage.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
        previous = &program;
    }
    diagnostic.erase("failed_slot");
    return stages;
}

struct ParsedRecord {
    std::vector<OfflineEnvironment> envs;
    std::vector<size_t> env_offsets; ///< From the start of the record
    std::vector<u8> key;
    std::array<u64, 6> hashes{};
    bool compute{};
};

// One pipeline record as the cache stores it: environment count, environments, pipeline key.
ParsedRecord ParseRecord(std::span<const u8> bytes, u32 version, size_t& size) {
    Reader reader{bytes};
    const auto offset = [&] { return bytes.size() - reader.Remaining(); };
    const u32 count = reader.Read<u32>();
    if (count == 0 || count > 6)
        throw std::runtime_error("Invalid number of shader environments");
    ParsedRecord record;
    record.envs = std::vector<OfflineEnvironment>(count);
    for (OfflineEnvironment& env : record.envs) {
        record.env_offsets.push_back(offset());
        env.Load(reader, version);
    }
    record.compute = record.envs.front().ShaderStage() == Stage::Compute;
    if (record.compute ? count != 1 : std::ranges::any_of(record.envs, [](const auto& env) {
            return env.ShaderStage() == Stage::Compute;
        })) {
        throw std::runtime_error("Invalid shader stages in the pipeline record");
    }
    record.key.resize(record.compute ? 24 : 1056);
    reader.Copy(record.key.data(), record.key.size());
    if (record.compute) {
        u32 shared_memory;
        std::array<u32, 3> workgroup;
        std::memcpy(&shared_memory, record.key.data() + 8, sizeof(shared_memory));
        std::memcpy(workgroup.data(), record.key.data() + 12, sizeof(workgroup));
        if (record.envs.front().SharedMemorySize() != shared_memory ||
            record.envs.front().WorkgroupSize() != workgroup) {
            throw std::runtime_error("Pipeline key and shader environment disagree");
        }
    } else {
        std::memcpy(record.hashes.data(), record.key.data(), sizeof(record.hashes));
        if (static_cast<size_t>(
                std::ranges::count_if(record.hashes, [](u64 h) { return h != 0; })) != count) {
            throw std::runtime_error("Pipeline key and shader environments disagree");
        }
    }
    size = offset();
    return record;
}

ParsedRecord ParseWholeRecord(std::span<const u8> bytes, u32 version) {
    size_t size{};
    ParsedRecord record = ParseRecord(bytes, version, size);
    if (size != bytes.size())
        throw std::runtime_error("Trailing bytes after pipeline record");
    return record;
}

// The key a v2 log files a record under (ComputeRecordKey, GraphicsRecordKey): it leaves the
// driver out, so it pairs a legacy record with its migrated copy. Nullopt without the v2 cache.
std::optional<u64> RecordKey(std::span<const u8> bytes, const ParsedRecord& record) {
#ifdef SHADER_SWEEP_V2
    if (record.compute)
        return Vulkan::ComputeRecordKey(bytes);
    Vulkan::GraphicsPipelineCacheKey key;
    static_assert(sizeof(key) == 1056);
    std::memcpy(&key, record.key.data(), sizeof(key));
    return Vulkan::GraphicsRecordKey(key);
#else
    (void)bytes;
    (void)record;
    return std::nullopt;
#endif
}

Json ListEntry(const std::string& file, u64 offset, std::span<const u8> bytes, u32 version) {
    const ParsedRecord record = ParseWholeRecord(bytes, version);
    const auto record_key = RecordKey(bytes, record);
    Json stages = Json::array();
    for (size_t index = 0; index < record.envs.size(); ++index) {
        stages.push_back({{"stage", StageNames[static_cast<u32>(record.envs[index].ShaderStage())]},
                          {"offset", offset + record.env_offsets[index]}});
    }
    return {{"file", file},
            {"offset", offset},
            {"kind", record.compute ? "compute" : "graphics"},
            {"record_key", record_key ? Json(fmt::format("{:016x}", *record_key)) : Json()},
            {"stages", stages}};
}

std::vector<u8> ReadWholeFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path.string());
    std::vector<u8> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Cannot read " + path.string());
    return bytes;
}

u32 CacheFileVersion(std::span<const u8> header) {
    static constexpr std::array<u8, 8> Magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
    u32 version{};
    if (header.size() >= 12 && std::ranges::equal(header.first(8), Magic))
        std::memcpy(&version, header.data() + 8, sizeof(version));
    if (version < OldestCacheVersion || version > NewestCacheVersion) {
        throw std::runtime_error(fmt::format("Not a Vulkan pipeline cache of versions {} to {}",
                                             OldestCacheVersion, NewestCacheVersion));
    }
    return version;
}

// The legacy path's records: those of the cache file as they are, up to the first damaged one,
// where the runtime stops reading.
void ListLegacy(const std::filesystem::path& cache, const std::filesystem::path& output_dir,
                Json& diagnostic) {
    const auto bytes = ReadWholeFile(cache);
    const u32 version = CacheFileVersion(bytes);
    diagnostic["cache_version"] = version;
    diagnostic["record_version"] = version;
    Json records = Json::array();
    std::span<const u8> rest = std::span{bytes}.subspan(12);
    while (!rest.empty()) {
        const u64 offset = bytes.size() - rest.size();
        size_t size{};
        try {
            ParseRecord(rest, version, size);
        } catch (const std::exception& error) {
            diagnostic["damaged"] = {{"offset", offset}, {"message", error.what()}};
            break;
        }
        const auto record = rest.first(size);
        const auto name = fmt::format("{:06d}.record", records.size());
        WriteFile(output_dir / name, std::vector<u8>(record.begin(), record.end()));
        records.push_back(ListEntry(name, offset, record, version));
        rest = rest.subspan(size);
    }
    diagnostic["records"] = records;
}

#ifdef SHADER_SWEEP_V2
// The v2 path's records: the cache migrated by the production migration into a new log, read
// back by the production reader. With a v2 log of Program records, native programs take the
// code of this build's records (FindProgram, TakeProgramCode), and records of another build's
// lowering are lowered again and their pipeline records moved to the new code, as the boot does.
void ListV2(const std::filesystem::path& cache, const std::filesystem::path& programs_log,
            const std::filesystem::path& output_dir, Json& diagnostic) {
    namespace V2 = VideoCommon::ShaderCacheV2;
    std::array<u8, 12> header{};
    {
        std::ifstream file(cache, std::ios::binary);
        if (!file || !file.read(reinterpret_cast<char*>(header.data()), header.size()))
            throw std::runtime_error("Cannot read " + cache.string());
    }
    const u32 version = CacheFileVersion(header);
    diagnostic["cache_version"] = version;
    // Migrated records keep their bytes, so they are read in the source version's layout
    diagnostic["record_version"] = version;
    const auto work = output_dir / "migration";
    std::filesystem::create_directories(work);
    std::filesystem::copy_file(cache,
                               VideoCommon::VersionedPipelineCachePath(work, "vulkan", version),
                               std::filesystem::copy_options::overwrite_existing);
    const auto log = work / V2::FileName;
    std::filesystem::remove(log);
    const auto migration = Vulkan::MigrateLegacyPipelineCache(work, log, 16);
    if (!migration.version) {
        throw std::runtime_error(
            migration.failed
                ? "The migration couldn't write the v2 log"
                : fmt::format("The v2 cache doesn't migrate version {} caches", version));
    }
    diagnostic["migration"] = {{"source_version", *migration.version},
                               {"graphics", migration.graphics},
                               {"compute", migration.compute},
                               {"collapsed", migration.collapsed}};
    V2::LogFile file = V2::ReadLogFile(log);
    const V2::LogStats& stats = file.parsed.stats;
    diagnostic["log"] = {{"records", stats.records},
                         {"live", stats.live},
                         {"bad_headers", stats.bad_headers},
                         {"bad_payloads", stats.bad_payloads},
                         {"torn_tail_bytes", stats.torn_tail_bytes}};

    // As the boot does: programs of this build's lowering are found (LoadPrograms, FindProgram),
    // those of another build's are lowered again (RelowerProgramRecord). Every program found or
    // lowered to other code replaces the code of the pipeline records using it.
    Vulkan::RelinkedPrograms programs;
    u64 program_records{}, hits{}, relowered_count{}, unusable{};
    if (!programs_log.empty()) {
        const V2::LogFile program_file = V2::ReadLogFile(programs_log);
        V2::Cache program_cache{work / "programs.v2.bin"};
        program_cache.LoadPrograms(program_file.parsed.live);
        for (const V2::LogRecord& entry : program_file.parsed.live) {
            if (entry.kind != V2::RecordKind::Program)
                continue;
            ++program_records;
            const std::optional<V2::ProgramRecord> record = V2::DecodeProgram(entry.payload);
            if (!record) {
                ++unusable;
                continue;
            }
            const V2::FoundProgram found =
                program_cache.FindProgram(V2::ProgramKeyInputsOf(*record));
            if (found.lookup == V2::ProgramLookup::Hit) {
                ++hits;
                V2::ProgramRecord taken = *record;
                taken.lowered_code = found.code->code;
                taken.texture_bound = found.code->texture_bound;
                programs.insert_or_assign(record->unique_hash,
                                          Vulkan::RelinkedProgram{record->unique_hash,
                                                                  record->texture_bound,
                                                                  std::move(taken)});
                continue;
            }
            std::optional<V2::ProgramRecord> relowered;
            if (record->lowering_stamp != V2::LoweringStamp()) {
                try {
                    relowered = V2::RelowerProgramRecord(*record);
                } catch (const std::exception&) {
                }
            }
            if (!relowered) {
                ++unusable;
                continue;
            }
            ++relowered_count;
            if (relowered->unique_hash != record->unique_hash) {
                programs.insert_or_assign(record->unique_hash,
                                          Vulkan::RelinkedProgram{record->unique_hash,
                                                                  record->texture_bound,
                                                                  std::move(*relowered)});
            }
        }
    }

    Json records = Json::array();
    u64 relinked{};
    for (const V2::LogRecord& entry : file.parsed.live) {
        if (entry.kind != V2::RecordKind::Graphics && entry.kind != V2::RecordKind::Compute)
            continue;
        std::vector<u8> payload(entry.payload.begin(), entry.payload.end());
        bool rekeyed_record{};
        if (!programs.empty()) {
            try {
                if (auto rekeyed =
                        Vulkan::RekeyPipelineRecord(entry.kind, entry.payload, programs)) {
                    payload = std::move(rekeyed->payload);
                    rekeyed_record = true;
                    ++relinked;
                }
            } catch (const std::ios_base::failure&) {
                // Compiling it reports the damage
            }
        }
        const auto name = fmt::format("{:06d}.record", records.size());
        WriteFile(output_dir / name, payload);
        Json listed;
        try {
            listed = ListEntry(name, entry.offset + V2::RecordHeaderSize, payload, version);
        } catch (const std::exception& error) {
            listed = {{"file", name},
                      {"offset", entry.offset},
                      {"stages", Json::array()},
                      {"error", error.what()}};
        }
        listed["kind"] = entry.kind == V2::RecordKind::Compute ? "compute" : "graphics";
        // The key the migration filed it under, which the legacy record has too
        listed["record_key"] = fmt::format("{:016x}", entry.key);
        listed["relinked"] = rekeyed_record;
        records.push_back(std::move(listed));
    }
    diagnostic["programs"] = {{"records", program_records},
                              {"hits", hits},
                              {"relowered", relowered_count},
                              {"unusable", unusable},
                              {"relinked_records", relinked}};
    diagnostic["records"] = records;
}

// What the v2 boot does with a record before building it: the production reader's checks.
void CheckV2Payload(std::span<const u8> payload) {
    Common::UniqueFunction<void, std::istream&, VideoCommon::FileEnvironment> read_compute{
        [](std::istream& file, VideoCommon::FileEnvironment env) {
            Vulkan::ComputePipelineCacheKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            VideoCommon::CheckComputeRecord(env, key.shared_memory_size, key.workgroup_size);
        }};
    Common::UniqueFunction<void, std::istream&, std::vector<VideoCommon::FileEnvironment>>
        read_graphics{[](std::istream& file, std::vector<VideoCommon::FileEnvironment> envs) {
            Vulkan::GraphicsPipelineCacheKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            VideoCommon::CheckGraphicsRecord(key.unique_hashes, envs.size());
        }};
    try {
        VideoCommon::ReadPipelinePayload(payload, read_compute, read_graphics);
    } catch (const std::ios_base::failure& error) {
        throw std::runtime_error(fmt::format("The v2 reader rejects the record: {}", error.what()));
    }
}
#endif

int RunList(const std::filesystem::path& cache, const std::string& path_name,
            const std::filesystem::path& programs_log, const std::filesystem::path& output_dir,
            Json& diagnostic) {
    diagnostic["input_format"] = "cache";
    diagnostic["path"] = path_name;
    std::filesystem::create_directories(output_dir);
    if (path_name == "legacy") {
        ListLegacy(cache, output_dir, diagnostic);
    } else {
#ifdef SHADER_SWEEP_V2
        ListV2(cache, programs_log, output_dir, diagnostic);
#endif
    }
    diagnostic["status"] = "ok";
    diagnostic["phase"] = "output";
    std::cout << diagnostic.dump() << '\n';
    return 0;
}

int RunPipeline(const std::string& record_path, u32 version, const std::string& path_name,
                const std::filesystem::path& output_dir, Json& diagnostic) {
    diagnostic["input_format"] = "pipeline";
    diagnostic["path"] = path_name;
    const auto bytes = ReadFile(record_path);
#ifdef SHADER_SWEEP_V2
    if (path_name == "v2")
        CheckV2Payload(bytes);
#endif
    ParsedRecord record = ParseWholeRecord(bytes, version);
    const auto& key = record.key;
    const bool compute = record.compute;
    // The key the legacy cache looks the pipeline up by; every stage module belongs to it.
    const u64 pipeline_hash = Common::CityHash64(reinterpret_cast<const char*>(key.data()),
                                                 compute ? key.size() : GraphicsKeySize(key));
    const auto record_key = RecordKey(bytes, record);
    // The key each path files the stage modules under
    const u64 path_key = path_name == "v2" ? record_key.value() : pipeline_hash;
    diagnostic["kind"] = compute ? "compute" : "graphics";
    diagnostic["pipeline_hash"] = fmt::format("{:016x}", pipeline_hash);
    diagnostic["record_key"] = record_key ? Json(fmt::format("{:016x}", *record_key)) : Json();
    Json stages = Json::array();
    size_t env_index = 0;
    for (u32 slot = 0; slot < (compute ? 1U : 6U); ++slot) {
        if (!compute && record.hashes[slot] == 0)
            continue;
        const auto& env = record.envs[env_index++];
        u64 unique_hash = record.hashes[slot];
        if (compute)
            std::memcpy(&unique_hash, key.data(), sizeof(unique_hash));
        stages.push_back({{"slot", slot},
                          {"unique_hash", fmt::format("{:016x}", unique_hash)},
                          {"stage", StageNames[static_cast<u32>(env.ShaderStage())]},
                          {"key", compute ? fmt::format("{:016x}", path_key)
                                          : fmt::format("{:016x}/{}", path_key, slot)},
                          {"spirv", nullptr},
                          {"spirv_words", 0}});
    }
    diagnostic["stages"] = stages;
    diagnostic["assumptions"].push_back(
        "Pipeline replay mirrors the runtime's per-pipeline translation on the offline SPIR-V 1.4 "
        "profile: geometry, viewport-layer and mesh shader support, not MoltenVK, no transform "
        "feedback");
    auto compiled = CompileLegacyPipeline(record.envs, key, diagnostic);
    diagnostic["phase"] = "output";
    std::filesystem::create_directories(output_dir);
    for (size_t index = 0; index < compiled.size(); ++index) {
        auto& entry = diagnostic["stages"][index];
        const PipelineStage& stage = compiled[index];
        if (stage.spirv.empty())
            continue;
        const auto name =
            fmt::format("{}-{}.spv", stage.slot, StageNames[static_cast<u32>(stage.stage)]);
        WriteFile(output_dir / name, stage.spirv);
        entry["spirv"] = name;
        entry["spirv_words"] = stage.spirv.size();
    }
    diagnostic["status"] = "ok";
    std::cout << diagnostic.dump() << '\n';
    return 0;
}
} // namespace

/// Parses the command line, reads the program (and its environment and pipeline key when given),
/// translates it and writes the SPIR-V or lowered code. Prints a JSON report; the exit code is 2
/// for bad input and 1 for a failed translation.
int main(int argc, char** argv) {
    Json diagnostic{{"schema_version", 1}, {"status", "error"},           {"phase", "input"},
                    {"stage", nullptr},    {"input_format", nullptr},     {"source_bytes", 0},
                    {"spirv_words", 0},    {"assumptions", Json::array()}};
    try {
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error("Only little-endian hosts are supported");
        }
        std::string input, output, environment, pipeline_key, override_out, pipeline, output_dir,
            list, programs_log;
        std::string format{"instructions"}, path{"legacy"};
        std::optional<Stage> requested_stage;
        u32 version{};
        bool lower_only{};
        bool packed_code{};
        bool capture_extent{};
        Guesses guesses;
        bool storage_pointer_other_set{};
        for (int i = 1; i < argc; ++i) {
            const std::string arg{argv[i]};
            if (arg == "--help") {
                std::cout << "shader-sweep [INPUT] [--stage "
                             "vertex|tess-control|tess-eval|geometry|fragment|compute|task|mesh]\n"
                             "  [--format instructions|native-header] [--environment FILE "
                             "--cache-version 46..54]\n"
                             "  [--guess-missing [--guess-cbuf0 HEX] [--guess-cbuf-other HEX]\n"
                             "   [--guess-texture-type N] [--guess-pixel-format N] "
                             "[--guess-storage-pointer 0|1]\n"
                             "   [--guess-storage-pointer-other 0|1]]\n"
                             "  [--pipeline-key FILE] [--capture-extent] [--override-out FILE] "
                             "(--output SPV | --lower-only WORDS | --packed-code WORDS)\n"
                             "shader-sweep --pipeline RECORD --cache-version 46..54 "
                             "--output-dir DIR [--path legacy|v2]\n"
                             "  [--device-profile JSON] (any mode but --list)\n"
                             "shader-sweep --list CACHE --output-dir DIR [--path legacy|v2] "
                             "[--programs-log V2LOG]\n";
                return 0;
            }
            if (!arg.starts_with("--")) {
                if (!input.empty())
                    throw std::runtime_error("Multiple input files");
                input = arg;
                continue;
            }
            if (arg == "--capture-extent") {
                capture_extent = true;
                continue;
            }
            if (arg == "--guess-missing") {
                guesses.enabled = true;
                continue;
            }
            if (++i == argc)
                throw std::runtime_error("Missing value for " + arg);
            const std::string value{argv[i]};
            if (arg == "--stage")
                requested_stage = ParseStage(value);
            else if (arg == "--format")
                format = value;
            else if (arg == "--environment")
                environment = value;
            else if (arg == "--override-out")
                override_out = value;
            else if (arg == "--pipeline-key")
                pipeline_key = value;
            else if (arg == "--guess-cbuf0")
                guesses.cbuf0 = ParseOptionValue(arg, value, 16, std::numeric_limits<u32>::max());
            else if (arg == "--guess-cbuf-other")
                guesses.cbuf_other =
                    ParseOptionValue(arg, value, 16, std::numeric_limits<u32>::max());
            else if (arg == "--guess-texture-type")
                guesses.texture_type =
                    ParseOptionValue(arg, value, 0, Shader::NUM_TEXTURE_TYPES - 1);
            else if (arg == "--guess-pixel-format")
                guesses.pixel_format =
                    ParseOptionValue(arg, value, 0, std::numeric_limits<u8>::max());
            else if (arg == "--guess-storage-pointer") {
                if (value != "0" && value != "1")
                    throw std::runtime_error("Expected --guess-storage-pointer 0 or 1");
                guesses.storage_pointer = value == "1";
                if (!storage_pointer_other_set)
                    guesses.storage_pointer_other = guesses.storage_pointer;
            } else if (arg == "--guess-storage-pointer-other") {
                if (value != "0" && value != "1")
                    throw std::runtime_error("Expected --guess-storage-pointer-other 0 or 1");
                guesses.storage_pointer_other = value == "1";
                storage_pointer_other_set = true;
            } else if (arg == "--pipeline")
                pipeline = value;
            else if (arg == "--output-dir")
                output_dir = value;
            else if (arg == "--list")
                list = value;
            else if (arg == "--programs-log")
                programs_log = value;
            else if (arg == "--device-profile") {
                std::ifstream file{value};
                if (!file)
                    throw std::runtime_error("Cannot open " + value);
                device_profile = Json::parse(file);
                OfflineProfile(OfflineHost());
            }
            else if (arg == "--path") {
#ifdef SHADER_SWEEP_V2
                if (value == "v2") {
                    path = value;
                    continue;
                }
#else
                if (value == "v2")
                    throw std::runtime_error("This build has no v2 cache (build it in the tree)");
#endif
                if (value != "legacy")
                    throw std::runtime_error("Unknown compilation path " + value);
                path = value;
            } else if (arg == "--cache-version") {
                version = value.size() == 2 && std::isdigit(static_cast<unsigned char>(value[0])) &&
                                  std::isdigit(static_cast<unsigned char>(value[1]))
                              ? static_cast<u32>(std::stoul(value))
                              : 0;
                if (version < OldestCacheVersion || version > NewestCacheVersion) {
                    throw std::runtime_error(fmt::format("Expected cache version {} to {}",
                                                         OldestCacheVersion, NewestCacheVersion));
                }
            } else if (arg == "--output" || arg == "--lower-only" || arg == "--packed-code") {
                if (!output.empty())
                    throw std::runtime_error("Only one output mode is permitted");
                output = value;
                lower_only = arg != "--output";
                packed_code = arg == "--packed-code";
            } else
                throw std::runtime_error("Unknown option " + arg);
        }
        if (!list.empty()) {
            if (!input.empty() || !environment.empty() || !pipeline_key.empty() ||
                !override_out.empty() || !output.empty() || !pipeline.empty() || requested_stage ||
                capture_extent || version != 0 || output_dir.empty() ||
                (!programs_log.empty() && path != "v2")) {
                throw std::runtime_error("Invalid arguments; see --help");
            }
            return RunList(list, path, programs_log, output_dir, diagnostic);
        }
        if (!pipeline.empty()) {
            if (!input.empty() || !environment.empty() || !pipeline_key.empty() ||
                !override_out.empty() || !output.empty() || requested_stage || capture_extent ||
                version == 0 || output_dir.empty() || !programs_log.empty()) {
                throw std::runtime_error("Invalid arguments; see --help");
            }
            return RunPipeline(pipeline, version, path, output_dir, diagnostic);
        }
        if (output.empty() || !output_dir.empty() || !programs_log.empty() || path != "legacy" ||
            (input.empty() && environment.empty()) || (environment.empty() != (version == 0)) ||
            (input.empty() && (lower_only || capture_extent || !override_out.empty()))) {
            throw std::runtime_error("Invalid arguments; see --help");
        }
        if (format != "instructions" && format != "native-header")
            throw std::runtime_error("Invalid input format");
        if (!override_out.empty() && SameFile(output, override_out))
            throw std::runtime_error("--override-out must not name the output file");
        auto override_partial = override_out;
        if (!override_out.empty())
            override_partial += ".partial";
        for (const auto& target : {output, override_out, override_partial}) {
            if (target.empty())
                continue;
            for (const auto& path : {input, environment, pipeline_key}) {
                if (!path.empty() && SameFile(path, target))
                    throw std::runtime_error("Output must not overwrite an input");
            }
        }
        OfflineEnvironment env;
        env.guesses = guesses;
        if (!environment.empty()) {
            const auto bytes = ReadFile(environment);
            Reader reader{bytes};
            env.Load(reader, version);
            if (reader.Remaining())
                throw std::runtime_error("Trailing bytes after environment record");
        }
        if (requested_stage) {
            if (env.captured && env.ShaderStage() != *requested_stage)
                throw std::runtime_error("Stage disagrees with environment");
            env.SetStage(*requested_stage);
        }
        diagnostic["input_format"] = input.empty() ? "environment" : format;
        if (!input.empty()) {
            const auto bytes = ReadFile(input);
            const size_t header_size = format == "native-header" ? 128 : 0;
            if (bytes.size() <= header_size || (bytes.size() - header_size) % sizeof(Instruction) ||
                (bytes.size() - header_size) / sizeof(Instruction) > 65536) {
                throw std::runtime_error(
                    "Native input must contain 1..65536 complete 16-byte instructions");
            }
            if (header_size) {
                Shader::ProgramHeader header;
                std::memcpy(&header, bytes.data(), sizeof(header));
                const u32 type = header.common0.shader_type;
                if (type < 1 || type > 5)
                    throw std::runtime_error("Invalid native graphics header stage");
                const Stage header_stage = static_cast<Stage>(type - 1);
                const Stage stage = requested_stage.value_or(
                    env.captured ? env.ShaderStage() : header_stage);
                const bool mesh_envelope =
                    (stage == Stage::Task && type == 1) ||
                    (stage == Stage::Mesh && (type == 1 || type == 3));
                if ((stage != header_stage && !mesh_envelope) ||
                    (env.captured && env.ShaderStage() != stage)) {
                    throw std::runtime_error("Stage disagrees with native header");
                }
                env.SetStage(stage);
                env.SetHeader(bytes);
            } else if (!requested_stage && !env.captured && !lower_only) {
                throw std::runtime_error(
                    "Instruction-only input requires --stage or a captured environment");
            }
            if (requested_stage || env.captured || header_size) {
                diagnostic["stage"] = StageNames[static_cast<u32>(env.ShaderStage())];
            }
            diagnostic["source_bytes"] = bytes.size() - header_size;
            std::vector<Instruction> native((bytes.size() - header_size) / sizeof(Instruction));
            std::memcpy(native.data(), bytes.data() + header_size,
                        native.size() * sizeof(Instruction));
            if (capture_extent) {
                diagnostic["phase"] = "capture";
                diagnostic["allocation_bytes"] = native.size() * sizeof(Instruction);
                Shader::Ampere::ProgramCaptureState capture;
                bool complete = false;
                for (u32 index = 0; index < native.size(); ++index) {
                    const auto result = capture.Observe(native[index], index);
                    if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
                        throw MissingContext("Native allocation has an out-of-range branch target");
                    }
                    if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) {
                        native.resize(index + 1);
                        complete = true;
                        break;
                    }
                }
                if (!complete) {
                    throw MissingContext("Native allocation lacks a complete runtime capture");
                }
                diagnostic["source_bytes"] = native.size() * sizeof(Instruction);
            }
            diagnostic["phase"] = "lower";
            std::vector<u32> traps;
            for (u32 i = 0; i < native.size(); ++i) {
                if (native[i].Bits(0, 12) == 0x95c)
                    traps.push_back(i * sizeof(Instruction));
            }
            if (!traps.empty()) {
                if (!env.captured)
                    throw MissingContext("Trap guards require captured constant buffers");
                const bool verified = env.ShaderStage() == Stage::Compute &&
                                      Shader::Ampere::VerifyTrapGuards(
                                          std::span{native}.first(traps.back() / 16 + 1),
                                          [&](u32 bank, u32 offset) -> std::optional<u32> {
                                              // Guards are checked against captured values only,
                                              // even with --guess-missing.
                                              return env.RecordedCbufValue(bank, offset);
                                          });
                if (!verified)
                    throw MissingContext("Native trap guards could not be verified");
            }
            const auto tessellation = env.ShaderStage() == Stage::TessellationControl
                                          ? Shader::Ampere::TessellationStage::Control
                                      : env.ShaderStage() == Stage::TessellationEval
                                          ? Shader::Ampere::TessellationStage::Evaluation
                                          : Shader::Ampere::TessellationStage::None;
            const auto lowered = Shader::Ampere::LowerToMaxwell(
                native, env.ShaderStage() == Stage::Geometry, true, traps, tessellation);
            if (!lowered) {
                diagnostic["error"] = {{"code", "lowering_failed"},
                                       {"message", lowered.error},
                                       {"offset", lowered.error_offset}};
                std::cout << diagnostic.dump() << '\n';
                return 1;
            }
            diagnostic["lowered_words"] = lowered.instructions.size();
            // The emulator's key for this program (Shader::Ampere::NativeProgramKey over the
            // captured source), and the lowering as a program override record
            // (the tools/nx2/make_*_warm.py scripts read it).
            diagnostic["native_key"] =
                fmt::format("{:016x}", Shader::Ampere::NativeProgramKey(native));
            if (!override_out.empty()) {
                const auto record = Shader::Ampere::EncodeProgramOverride(native, lowered);
                if (record.empty())
                    throw std::runtime_error("Cannot encode the lowering as an override record");
                // Larger records still serve the warm-file scripts; the emulator's loader
                // (shader_environment.cpp) would refuse them as an override.
                diagnostic["override_within_emulator_limits"] =
                    Shader::Ampere::ProgramOverrideSourceSize(record) != 0 &&
                    record.size() <= size_t{16} * 1024 * 1024;
                // Renamed into place once complete, so a reader never sees a partial record.
                auto partial = override_out;
                partial += ".partial";
                WriteFile(partial, record);
                std::filesystem::rename(partial, override_out);
            }
            if (lower_only) {
                diagnostic["phase"] = "output";
                if (packed_code) {
                    env.SetCode(lowered);
                    WriteFile(output, env.code);
                } else {
                    WriteFile(output, lowered.instructions);
                }
                diagnostic["status"] = "ok";
                std::cout << diagnostic.dump() << '\n';
                return 0;
            }
            if (!env.captured && !header_size && env.ShaderStage() != Stage::Compute) {
                throw MissingContext(
                    "Graphics compilation requires a native header or captured environment");
            }
            env.SetCode(lowered);
            if (!env.captured) {
                diagnostic["assumptions"].push_back(
                    "No environment: workgroup 1x1x1, shared memory 0, viewport enabled, no HLE "
                    "replacements");
                if (!header_size)
                    diagnostic["assumptions"].push_back(
                        "Compute local memory defaults to zero without a captured environment");
            }
        }
        diagnostic["stage"] = StageNames[static_cast<u32>(env.ShaderStage())];
        if (env.ShaderStage() == Stage::Task || env.ShaderStage() == Stage::Mesh ||
            env.ShaderStage() == Stage::VertexA) {
            throw MissingContext(
                "Task, mesh and vertex-A require additional pipeline stage context");
        }
        const auto key = pipeline_key.empty() ? std::vector<u8>{} : ReadFile(pipeline_key);
        auto runtime = Runtime(key, env.ShaderStage(), diagnostic["assumptions"]);
        diagnostic["assumptions"].push_back(
            device_profile.empty()
                ? "Offline SPIR-V 1.4 profile; no device feature probing, default shader settings, "
                  "no resolution scaling"
                : "Device profile from --device-profile; default shader settings, no resolution "
                  "scaling");
        diagnostic["phase"] = "translate";
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{env.EntryPoint()}};
        Shader::ObjectPool<Shader::IR::Inst> inst_pool;
        Shader::ObjectPool<Shader::IR::Block> block_pool;
        const Shader::HostTranslateInfo host = OfflineHost();
        auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
        Shader::Maxwell::ConvertLegacyToGeneric(program, runtime);
        diagnostic["phase"] = "spirv";
        const Shader::Profile profile = OfflineProfile(host);
        Shader::Backend::Bindings bindings{};
        const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
        diagnostic["spirv_words"] = spirv.size();
        if (guesses.enabled)
            diagnostic["guessed_reads"] = env.guessed_reads;
        diagnostic["phase"] = "output";
        WriteFile(output, spirv);
        diagnostic["status"] = "ok";
        std::cout << diagnostic.dump() << '\n';
        return 0;
    } catch (const MissingContext& error) {
        diagnostic["phase"] = "context";
        diagnostic["error"] = {{"code", "missing_context"}, {"message", error.what()}};
    } catch (const std::exception& error) {
        diagnostic["error"] = {
            {"code", diagnostic["phase"] == "input" ? "invalid_input" : "compilation_failed"},
            {"message", error.what()}};
    }
    std::cout << diagnostic.dump() << '\n';
    return diagnostic["phase"] == "input" ? 2 : 1;
}
