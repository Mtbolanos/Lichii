// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <cstring>
#include <string>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include "common/assert.h"
#include "common/cityhash.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/fs/file.h"
#include "common/fs/file_lock.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include <ranges>
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/trap_guards.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"
#include "video_core/native_lowering.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

constexpr std::array<char, 8> MAGIC_NUMBER{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};

constexpr size_t INST_SIZE = sizeof(u64);

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

// Constant buffer slots that held a null pointer when the program was translated and were
// therefore not bound as storage buffers. A later non-null value makes the translation stale.
static std::vector<std::pair<u32, u32>> NullStoragePointerSlots(
    const ::Common::unordered_map<u64, u32>& cbuf_values,
    const ::Common::unordered_map<u64, u8>& storage_pointer_types) {
    std::vector<std::pair<u32, u32>> slots;
    for (const auto& [key, type] : storage_pointer_types) {
        if (type != 0 || (key >> 63) != 0) {
            continue;
        }
        const auto low = cbuf_values.find(key);
        const auto high = cbuf_values.find(key + 4);
        if (low != cbuf_values.end() && high != cbuf_values.end() && low->second == 0 &&
            high->second == 0) {
            slots.emplace_back(static_cast<u32>(key >> 32), static_cast<u32>(key));
        }
    }
    return slots;
}

static std::vector<u64> MappedStoragePointerKeys(
    const ::Common::unordered_map<u64, u8>& storage_pointer_types) {
    std::vector<u64> keys;
    for (const auto& [key, type] : storage_pointer_types) {
        if (type != 0) {
            keys.push_back(key);
        }
    }
    std::ranges::sort(keys);
    return keys;
}

static constexpr u64 MakeCbufKey(u32 index, u32 offset) {
    return (static_cast<u64>(index) << 32) | offset;
}

// A constant buffer value translation never reads, as no constant buffer has this index: in each
// environment of a record, the EnvironmentReadProperties of the host that made it. Older builds
// load it as one more value.
constexpr u64 READ_PROPERTIES_KEY{MakeCbufKey(~0U, 0)};

// Another such value: present in records whose texture types were read as ReadTextureInfo reads
// an empty descriptor now (a 2D texture). Records without it can hold a 1D type for the null
// texture at handle 0 (issue #846).
constexpr u64 NULL_TEXTURE_2D_KEY{MakeCbufKey(~0U, 1)};

// The most values of a kind an environment of a record holds. Translation reads a few hundred at
// most; a damaged count can't make a load allocate more than a few MB for them.
constexpr u64 MAX_RECORDED_VALUES{u64{1} << 16};

// Pointer-table entries share the classification map with plain constant buffer pointers.
static u64 MakeIndirectCbufKey(u32 index, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (static_cast<u64>(indirect & 0x3fffff) << 40) |
           MakeCbufKey(index, offset);
}

// Whether a value read from a cache record names a stage
static bool IsShaderStage(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::VertexB:
    case Shader::Stage::TessellationControl:
    case Shader::Stage::TessellationEval:
    case Shader::Stage::Geometry:
    case Shader::Stage::Fragment:
    case Shader::Stage::Compute:
    case Shader::Stage::VertexA:
    case Shader::Stage::Task:
    case Shader::Stage::Mesh:
        return true;
    }
    return false;
}

static Shader::TextureType ConvertTextureType(const Tegra::Texture::TICEntry& entry) {
    switch (entry.texture_type) {
    case Tegra::Texture::TextureType::Texture1D:
        return Shader::TextureType::Color1D;
    case Tegra::Texture::TextureType::Texture2D:
    case Tegra::Texture::TextureType::Texture2DNoMipmap:
        return entry.normalized_coords ? Shader::TextureType::Color2D
                                       : Shader::TextureType::Color2DRect;
    case Tegra::Texture::TextureType::Texture3D:
        return Shader::TextureType::Color3D;
    case Tegra::Texture::TextureType::TextureCubemap:
        return Shader::TextureType::ColorCube;
    case Tegra::Texture::TextureType::Texture1DArray:
        return Shader::TextureType::ColorArray1D;
    case Tegra::Texture::TextureType::Texture2DArray:
        return Shader::TextureType::ColorArray2D;
    case Tegra::Texture::TextureType::Texture1DBuffer:
        return Shader::TextureType::Buffer;
    case Tegra::Texture::TextureType::TextureCubeArray:
        return Shader::TextureType::ColorArrayCube;
    default:
        LOG_ERROR(Shader, "Invalid texture_type={}. Falling back to texture_type={}", static_cast<int>(entry.texture_type.Value()), Shader::TextureType::Color2D);
        return Shader::TextureType::Color2D;
    }
}

static Shader::TexturePixelFormat ConvertTexturePixelFormat(const Tegra::Texture::TICEntry& entry) {
    return static_cast<Shader::TexturePixelFormat>(
        PixelFormatFromTextureInfo(entry.format, entry.r_type, entry.g_type, entry.b_type,
                                   entry.a_type, entry.srgb_conversion));
}

static std::string_view StageToPrefix(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::VertexB:
        return "vs";
    case Shader::Stage::TessellationControl:
        return "tc";
    case Shader::Stage::TessellationEval:
        return "te";
    case Shader::Stage::Geometry:
        return "gs";
    case Shader::Stage::Fragment:
        return "fs";
    case Shader::Stage::Compute:
        return "cs";
    case Shader::Stage::VertexA:
        return "va";
    default:
        return "uk";
    }
}

static void DumpImpl(u64 /*pipeline_hash*/, u64 shader_hash, std::span<const u64> code,
                     [[maybe_unused]] u32 read_highest, [[maybe_unused]] u32 read_lowest,
                     u32 initial_offset, Shader::Stage stage) {
    const auto dump_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::DumpDir)};
    if (!Common::FS::CreateDir(dump_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create dump directory");
        return;
    }
    const auto prefix = StageToPrefix(stage);
    const auto name{dump_dir /
                    fmt::format("{:016x}_{:016x}_{}.ash",
                                Settings::GetCurrentProgramID(), shader_hash, prefix)};
    std::fstream shader_file(name, std::ios::out | std::ios::binary);
    ASSERT(initial_offset % sizeof(u64) == 0);
    const size_t jump_index = initial_offset / sizeof(u64);
    const size_t code_size = code.size_bytes() - initial_offset;
    shader_file.write(reinterpret_cast<const char*>(&code[jump_index]), code_size);

    // + 1 instruction, due to the fact that we skip the final self branch instruction in the code,
    // but we need to consider it for padding, otherwise nvdisasm rages.
    const size_t padding_needed = (32 - ((code_size + INST_SIZE) % 32)) % 32;
    for (size_t i = 0; i < INST_SIZE + padding_needed; i++) {
        shader_file.put(0);
    }
}

GenericEnvironment::GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                       u32 start_address_)
    : gpu_memory{&gpu_memory_}, program_base{program_base_} {
    start_address = start_address_;
}

GenericEnvironment::~GenericEnvironment() = default;

u32 GenericEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

u32 GenericEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 GenericEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

std::array<u32, 3> GenericEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

u64 GenericEnvironment::ReadInstruction(u32 address) {
    read_lowest = (std::min)(read_lowest, address);
    read_highest = (std::max)(read_highest, address);

    if (address >= cached_lowest && address < cached_highest) {
        return code[(address - cached_lowest) / INST_SIZE];
    }
    if (native_source_size) {
        throw Shader::RuntimeError("Lowered native instruction address out of range: {:#x}", address);
    }
    has_unbound_instructions = true;
    return gpu_memory->Read<u64>(program_base + address);
}

u32 GenericEnvironment::TranslateProgramAddress(u32 address) const {
    if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) return *translated;
    if (native_source_size || Shader::Ampere::HasNativeAddressMap(code))
        throw Shader::RuntimeError("Invalid native indirect branch address {:#x}", address);
    return address;
}

bool GenericEnvironment::IsNativeProgram() const {
    return native_source_size != 0 || Shader::Ampere::HasNativeAddressMap(code);
}

bool GenericEnvironment::IsCbufStorageBufferPointer(u32 index, u32 offset) {
    const u64 key = MakeCbufKey(index, offset);
    if (const auto it = storage_pointer_types.find(key); it != storage_pointer_types.end()) {
        return it->second != 0;
    }
    const u64 address = u64{ReadCbufValue(index, offset)} |
                        (u64{ReadCbufValue(index, offset + 4)} << 32);
    const bool mapped = address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
                        gpu_memory->IsFullyMappedRange(address, 1);
    storage_pointer_types.emplace(key, static_cast<u8>(mapped));
    return mapped;
}

bool GenericEnvironment::IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) {
    const u64 key = MakeIndirectCbufKey(index, offset, indirect);
    if (const auto it = storage_pointer_types.find(key); it != storage_pointer_types.end()) {
        return it->second != 0;
    }
    bool mapped = false;
    if (IsCbufStorageBufferPointer(index, offset)) {
        const u64 holder =
            u64{ReadCbufValue(index, offset)} | (u64{ReadCbufValue(index, offset + 4)} << 32);
        if (gpu_memory->IsFullyMappedRange(holder + indirect, sizeof(u64))) {
            const u64 address = gpu_memory->Read<u64>(holder + indirect);
            mapped = address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
                     gpu_memory->IsFullyMappedRange(address, 1);
        }
    }
    storage_pointer_types.emplace(key, static_cast<u8>(mapped));
    return mapped;
}

std::vector<std::pair<u32, u32>> GenericEnvironment::NullStoragePointerSlots() const {
    return VideoCommon::NullStoragePointerSlots(cbuf_values, storage_pointer_types);
}

std::vector<u64> GenericEnvironment::MappedStoragePointerKeys() const {
    return VideoCommon::MappedStoragePointerKeys(storage_pointer_types);
}

void GenericEnvironment::KeepStoragePointers(std::span<const u64> keys) {
    for (const u64 key : keys) {
        storage_pointer_types.insert_or_assign(key, u8{1});
    }
}

std::optional<u64> GenericEnvironment::Analyze() {
    if (native_source_size) {
        // Include the converted header and a version tag to isolate this experimental ABI.
        return NativeShaderHash(code);
    }
    const std::optional<u64> size{TryFindSize()};
    if (!size) {
        return std::nullopt;
    }
    cached_lowest = start_address;
    cached_highest = start_address + static_cast<u32>(*size);
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()), *size);
}

void GenericEnvironment::SetCachedSize(size_t size_bytes) {
    if (native_source_size) {
        if (size_bytes != CachedSizeBytes()) {
            throw Shader::RuntimeError("Native shader changed while constructing pipeline");
        }
        return;
    }
    cached_lowest = start_address;
    cached_highest = start_address + static_cast<u32>(size_bytes);
    code.resize(CachedSizeWords());
    gpu_memory->ReadBlock(program_base + cached_lowest, code.data(), code.size() * sizeof(u64));
}

size_t GenericEnvironment::CachedSizeWords() const noexcept {
    return CachedSizeBytes() / INST_SIZE;
}

size_t GenericEnvironment::CachedSizeBytes() const noexcept {
    return static_cast<size_t>(cached_highest) - cached_lowest + INST_SIZE;
}

size_t GenericEnvironment::ReadSizeBytes() const noexcept {
    return read_highest - read_lowest + INST_SIZE;
}

bool GenericEnvironment::CanBeSerialized() const noexcept {
    // The read properties and the null texture marker are two more constant buffer values.
    return !has_unbound_instructions &&
           std::max({texture_types.size(), texture_pixel_formats.size(), cbuf_values.size() + 2,
                     storage_pointer_types.size(), cbuf_replacements.size()}) <=
               MAX_RECORDED_VALUES;
}

bool GenericEnvironment::HasNativeInstructions() const {
    return native_source_size || Shader::Ampere::HasNativeAddressMap(code);
}

bool FileEnvironment::HasNativeInstructions() const {
    return Shader::Ampere::HasNativeAddressMap(code);
}

u64 GenericEnvironment::CalculateHash() const {
    const size_t size{ReadSizeBytes()};
    const auto data{std::make_unique<char[]>(size)};
    gpu_memory->ReadBlock(program_base + read_lowest, data.get(), size);
    return Common::CityHash64(data.get(), size);
}

void GenericEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    DumpImpl(pipeline_hash, shader_hash, code, read_highest, read_lowest, initial_offset, stage);
}

void GenericEnvironment::Serialize(std::ostream& file, u32 read_properties) const {
    const u64 code_size{static_cast<u64>(CachedSizeBytes())};
    const u64 num_texture_types{static_cast<u64>(texture_types.size())};
    const u64 num_texture_pixel_formats{static_cast<u64>(texture_pixel_formats.size())};
    const u64 num_cbuf_values{static_cast<u64>(cbuf_values.size()) + 2};
    const u64 num_storage_pointer_types{static_cast<u64>(storage_pointer_types.size())};
    const u64 num_cbuf_replacement_values{static_cast<u64>(cbuf_replacements.size())};

    file.write(reinterpret_cast<const char*>(&code_size), sizeof(code_size))
        .write(reinterpret_cast<const char*>(&num_texture_types), sizeof(num_texture_types))
        .write(reinterpret_cast<const char*>(&num_texture_pixel_formats),
               sizeof(num_texture_pixel_formats))
        .write(reinterpret_cast<const char*>(&num_cbuf_values), sizeof(num_cbuf_values))
        .write(reinterpret_cast<const char*>(&num_storage_pointer_types),
               sizeof(num_storage_pointer_types))
        .write(reinterpret_cast<const char*>(&num_cbuf_replacement_values),
               sizeof(num_cbuf_replacement_values))
        .write(reinterpret_cast<const char*>(&local_memory_size), sizeof(local_memory_size))
        .write(reinterpret_cast<const char*>(&texture_bound), sizeof(texture_bound))
        .write(reinterpret_cast<const char*>(&start_address), sizeof(start_address))
        .write(reinterpret_cast<const char*>(&cached_lowest), sizeof(cached_lowest))
        .write(reinterpret_cast<const char*>(&cached_highest), sizeof(cached_highest))
        .write(reinterpret_cast<const char*>(&viewport_transform_state),
               sizeof(viewport_transform_state))
        .write(reinterpret_cast<const char*>(&stage), sizeof(stage))
        .write(reinterpret_cast<const char*>(code.data()), code_size);
    for (const auto& [key, type] : texture_types) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    for (const auto& [key, format] : texture_pixel_formats) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&format), sizeof(format));
    }
    for (const auto& [key, type] : cbuf_values) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    constexpr u32 null_texture_2d{1};
    file.write(reinterpret_cast<const char*>(&READ_PROPERTIES_KEY), sizeof(READ_PROPERTIES_KEY))
        .write(reinterpret_cast<const char*>(&read_properties), sizeof(read_properties))
        .write(reinterpret_cast<const char*>(&NULL_TEXTURE_2D_KEY), sizeof(NULL_TEXTURE_2D_KEY))
        .write(reinterpret_cast<const char*>(&null_texture_2d), sizeof(null_texture_2d));
    for (const auto& [key, type] : storage_pointer_types) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    for (const auto& [key, type] : cbuf_replacements) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    if (stage == Shader::Stage::Compute) {
        file.write(reinterpret_cast<const char*>(&workgroup_size), sizeof(workgroup_size))
            .write(reinterpret_cast<const char*>(&shared_memory_size), sizeof(shared_memory_size));
    } else {
        file.write(reinterpret_cast<const char*>(&sph), sizeof(sph));
        if (stage == Shader::Stage::Geometry) {
            file.write(reinterpret_cast<const char*>(&gp_passthrough_mask),
                       sizeof(gp_passthrough_mask));
        }
        if (stage == Shader::Stage::Task || stage == Shader::Stage::Mesh) {
            file.write(reinterpret_cast<const char*>(&mesh_registers), sizeof(mesh_registers))
                .write(reinterpret_cast<const char*>(&mesh_primitive_header),
                       sizeof(mesh_primitive_header));
        }
    }
}

std::optional<u64> GenericEnvironment::TryFindSize() {
    static constexpr size_t BLOCK_SIZE = 0x1000;
    static constexpr size_t MAXIMUM_SIZE = 0x100000;

    static constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
    static constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;

    static constexpr u64 EXIT_VALUE = 0xE30000000007000FULL;

    code.resize(MAXIMUM_SIZE / INST_SIZE);

    GPUVAddr guest_addr{program_base + start_address};
    size_t offset{0};
    size_t size{BLOCK_SIZE};
    while (size <= MAXIMUM_SIZE) {
        u64* const data = code.data() + offset / INST_SIZE;
        gpu_memory->ReadBlock(guest_addr, data, BLOCK_SIZE);
        for (size_t index = 0; index < BLOCK_SIZE; index += INST_SIZE) {
            const u64 inst = data[index / INST_SIZE];
            if (inst == SELF_BRANCH_A || inst == SELF_BRANCH_B) {
                return offset + index;
            }
            if (!is_proprietary_driver && inst == EXIT_VALUE) {
                return offset + index + INST_SIZE;
            }
        }
        guest_addr += BLOCK_SIZE;
        size += BLOCK_SIZE;
        offset += BLOCK_SIZE;
    }
    return std::nullopt;
}

Tegra::Texture::TICEntry GenericEnvironment::ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                                             bool via_header_index, u32 raw) {
    const auto handle{Tegra::Texture::TexturePair(raw, via_header_index)};
    const u32 index{Tegra::Texture::TextureHeaderIndex(handle.first, tic_limit,
                                                       gpu_memory->UsesNativeTiling())};
    Tegra::Texture::TICEntry entry;
    const auto null_2d_texture = [&entry] {
        entry.raw.fill(0);
        entry.format.Assign(Tegra::Texture::TextureFormat::A8B8G8R8);
        entry.r_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.g_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.b_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.a_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.texture_type.Assign(Tegra::Texture::TextureType::Texture2D);
        entry.normalized_coords.Assign(1);
    };
    if (index > tic_limit) {
        // Past the pool, the bytes are not a descriptor: reading them gave the shader a random
        // texture type and format (garbage-format asserts, then crashes and device losses on AMD
        // in Ocarina of Time). At draw time the texture cache binds the null view for such an
        // index (VisitImageView), so compile for what that view is: a plain 2D RGBA8 texture.
        static std::atomic<u32> warnings{};
        if (warnings.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(HW_GPU, "Texture handle {:#x} indexes past the descriptor pool "
                                "(index {} > limit {}); using a null 2D texture",
                        raw, index, tic_limit);
        }
        null_2d_texture();
        return entry;
    }
    const GPUVAddr descriptor_addr{tic_addr + index * sizeof(Tegra::Texture::TICEntry)};
    if (texture_pool_flushed) {
        gpu_memory->ReadBlockUnsafe(descriptor_addr, &entry, sizeof(entry));
    } else {
        gpu_memory->ReadBlock(descriptor_addr, &entry, sizeof(entry));
    }
    if (entry.raw == decltype(entry.raw){}) {
        // An empty descriptor (the null texture, handle 0 in NVN games): a game can draw with it
        // before it assigns the real texture, and the shader built for that draw is kept for the
        // pipeline. Read as it is, the zeroes say "1D texture", so a size query of a texture that
        // turns out 2D (it has no type of its own) got a null 1D view for good: Pokemon Legends
        // Z-A's tonemap then samples one texel for the whole screen (issue #846). Compile for a
        // plain 2D RGBA8 texture instead, as for an index past the pool.
        null_2d_texture();
        return entry;
    }
    if (entry.format.Value() == Tegra::Texture::TextureFormat{} &&
        std::getenv("EDEN_NX2_RENDER_TRACE")) {
        static std::atomic<u32> traces{};
        if (traces.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_WARNING(HW_GPU,
                        "Zero-format shader TIC: program={:#x} stage={} handle={:#x} "
                        "index={} limit={} descriptor={:#x} mapped={} image={:#x} "
                        "size={}x{}x{} header={} type={} raw={:016x}/{:016x}/{:016x}/{:016x}",
                        program_base, static_cast<u32>(stage), raw, index, tic_limit,
                        descriptor_addr, gpu_memory->IsFullyMappedRange(descriptor_addr, sizeof(entry)),
                        entry.Address(), entry.Width(), entry.Height(), entry.Depth(),
                        static_cast<u32>(entry.header_version.Value()),
                        static_cast<u32>(entry.texture_type.Value()),
                        entry.raw[0], entry.raw[1], entry.raw[2], entry.raw[3]);
        }
    }
    return entry;
}

namespace {

// Reads a native program a 4 KiB page at a time instead of one 16-byte instruction at a time.
// Each page is read with ReadBlock, so GPU-written code is flushed first, once per page instead
// of once per instruction. An instruction that is not inside a fully mapped page is left to the
// caller.
class NativeInstructionReader {
public:
    using Instruction = Shader::Ampere::Instruction;
    static constexpr GPUVAddr PageSize = 0x1000;

    explicit NativeInstructionReader(Tegra::MemoryManager& gpu_memory_)
        : gpu_memory{gpu_memory_} {}

    /// Copies the instruction at `address` into `out`, loading its page first if needed.
    /// Returns false when the instruction is not inside a fully mapped page.
    bool TryRead(GPUVAddr address, Instruction& out) {
        if (!Holds(address)) {
            LoadFrom(address);
        }
        if (!Holds(address)) {
            return false;
        }
        out = page[(address - page_begin) / sizeof(Instruction)];
        return true;
    }

private:
    bool Holds(GPUVAddr address) const {
        return address >= page_begin && address + sizeof(Instruction) <= page_end;
    }

    // Loads from `address` up to the next page boundary; loads nothing if that range is not
    // fully mapped.
    void LoadFrom(GPUVAddr address) {
        const GPUVAddr next_page = (address | (PageSize - 1)) + 1;
        const GPUVAddr length = next_page - address;
        page_begin = address;
        page_end = address;
        if (length >= sizeof(Instruction) && gpu_memory.IsFullyMappedRange(address, length)) {
            gpu_memory.ReadBlock(address, page.data(), length);
            page_end = next_page;
        }
    }

    Tegra::MemoryManager& gpu_memory;
    std::array<Instruction, PageSize / sizeof(Instruction)> page;
    GPUVAddr page_begin{};
    GPUVAddr page_end{};
};

} // Anonymous namespace

// A native program is captured and lowered once, when the shader cache first meets it. The
// environments built later for its pipelines receive that result instead of lowering again.
void GenericEnvironment::InitializeNativeCode(std::shared_ptr<const LoweredNativeCode> lowered,
                                              u32 source_header_size, bool geometry_stage,
                                              ShaderCacheV2::Cache* cache_v2) {
    if (!lowered) {
        lowered = LowerNativeCode(source_header_size, geometry_stage, cache_v2);
    }
    if (!Shader::Ampere::VerifyTrapGuards(lowered->trap_prefix, [&](u32 bank, u32 offset) {
            return ReadTrapGuardConstant(bank, offset);
        }))
        throw Shader::Ampere::TrapGuardError{};
    if (lowered->texture_bound != ~0U) {
        // Native TEX carries its cbuf index explicitly; TEXS obtains it from the environment.
        texture_bound = lowered->texture_bound;
    }
    native_source_size = lowered->source_size;
    code = lowered->code;
    cached_lowest = 0;
    cached_highest = static_cast<u32>((code.size() - 1) * sizeof(u64));
    lowered_native = std::move(lowered);
}

std::shared_ptr<const LoweredNativeCode> GenericEnvironment::LowerNativeCode(
    u32 source_header_size, bool geometry_stage, ShaderCacheV2::Cache* cache_v2) {
    using Shader::Ampere::Instruction;
    using Clock = std::chrono::steady_clock;
    const Clock::time_point capture_begin = cache_v2 ? Clock::now() : Clock::time_point{};
    const u32 native_header_size = source_header_size;
    constexpr u32 maximum_instructions = Shader::Ampere::ProgramCaptureState::MaximumInstructions;
    std::vector<Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    NativeInstructionReader reader{*gpu_memory};
    for (u32 index = 0; index < maximum_instructions; ++index) {
        const GPUVAddr address = program_base + native_header_size + index * sizeof(Instruction);
        Instruction instruction;
        if (!reader.TryRead(address, instruction)) {
            // Not inside a fully mapped page: check and read this one instruction on its own.
            if (!gpu_memory->IsFullyMappedRange(address, sizeof(Instruction))) {
                // The program may end before a branch target past its last EXIT (the capture
                // can't tell); stop at the mapping's end and let the lowering judge what was
                // captured.
                if (!native.empty()) break;
                throw Shader::RuntimeError("Unmapped native shader instruction at {:#x}", address);
            }
            gpu_memory->ReadBlock(address, &instruction, sizeof(instruction));
        }
        native.push_back(instruction);
        const auto status = capture.Observe(instruction, index);
        if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            if (const char* dir = std::getenv("EDEN_NX2_SHADER_DUMP")) {
                const auto path = fmt::format("{}/rejected-{:x}.bin", dir, program_base + native_header_size);
                if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
                    std::fwrite(native.data(), sizeof(Instruction), std::min<size_t>(native.size(), 4096), file);
                    std::fclose(file);
                }
            }
            throw Shader::RuntimeError("Native shader branch outside capture bounds at {:#x} +{:#x} [{:08x} {:08x} {:08x} {:08x}]",
                                       program_base + native_header_size, index * 16,
                                       instruction.words[0], instruction.words[1],
                                       instruction.words[2], instruction.words[3]);
        }
        if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) break;
    }
    const std::vector<u32> verified_traps = NativeTrapOffsets(native);
    const size_t trap_count = verified_traps.empty() ? 0 : verified_traps.back() / sizeof(Instruction) + 1;
    if (trap_count && (stage != Shader::Stage::Compute ||
                      !Shader::Ampere::VerifyTrapGuards(std::span{native}.first(trap_count),
                                                       [&](u32 bank, u32 offset) {
                                                           return ReadTrapGuardConstant(bank, offset);
                                                       })))
        throw Shader::Ampere::TrapGuardError{};
    const auto tessellation_stage = stage == Shader::Stage::TessellationControl
                                       ? Shader::Ampere::TessellationStage::Control
                                   : stage == Shader::Stage::TessellationEval
                                       ? Shader::Ampere::TessellationStage::Evaluation
                                       : Shader::Ampere::TessellationStage::None;
    const NativeLoweringInputs inputs{.geometry = geometry_stage,
                                      .tessellation_stage = tessellation_stage,
                                      .native_code_address = program_base + native_header_size};
    // A program the v2 cache holds a record of from this build's lowering is not lowered again:
    // the lowering depends on nothing else. Every check below still runs on its code.
    ShaderCacheV2::FoundProgram found;
    if (cache_v2) {
        found = cache_v2->FindProgram(ShaderCacheV2::MakeProgramKeyInputs(
            stage, native_header_size, sph, native, inputs, verified_traps));
    }
    const Clock::time_point lookup_end = cache_v2 ? Clock::now() : Clock::time_point{};
    Shader::Ampere::LoweredProgram lowered;
    if (!found.code) {
        lowered = LowerNativeProgram(native, inputs);
    }
    if (cache_v2) {
        const auto nanoseconds = [](Clock::duration duration) {
            return static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
        };
        cache_v2->CountProgram(found.lookup, nanoseconds(lookup_end - capture_begin),
                               found.code ? 0 : nanoseconds(Clock::now() - lookup_end));
    }
    bool overridden = false;
    // EDEN_NX2_REJECT=<hex address>,...: treat these programs as rejected (to find which newly
    // lowered program changes a game's behaviour without rebuilding).
    if (static const std::string reject_list = [] {
            const char* text = std::getenv("EDEN_NX2_REJECT");
            return text ? "," + std::string{text} + "," : std::string{};
        }();
        !reject_list.empty() &&
        reject_list.find(fmt::format(",{:x},", program_base)) != std::string::npos) {
        throw Shader::RuntimeError("Native shader at {:#x} rejected by EDEN_NX2_REJECT", program_base);
    }
    // EDEN_NX2_SHADER_DUMP=<dir>: keep every captured native program (named by its key) so the
    // lowering can be tested offline against all of a game's shaders.
    if (static const char* const dump_directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        dump_directory != nullptr) {
        const auto folder = std::filesystem::path{dump_directory};
        const auto path = folder / fmt::format("{:016x}-{}-{:x}.bin", Shader::Ampere::NativeProgramKey(native),
                                               source_header_size == 0 ? "cs" : "gfx", program_base);
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(folder);
            std::ofstream dump(path, std::ios::binary);
            dump.write(reinterpret_cast<const char*>(native.data()),
                       native.size() * sizeof(Shader::Ampere::Instruction));
        }
        if (native_header_size != 0) {
            auto header_path = path;
            header_path.replace_extension(".header.bin");
            if (!std::filesystem::exists(header_path) &&
                gpu_memory->IsFullyMappedRange(program_base, native_header_size)) {
                std::vector<u8> header(native_header_size);
                gpu_memory->ReadBlock(program_base, header.data(), header.size());
                std::ofstream dump(header_path, std::ios::binary);
                dump.write(reinterpret_cast<const char*>(header.data()), header.size());
                dump.close();
                if (dump) {
                    LOG_INFO(Shader, "Captured native shader header at {:#x}: {} bytes, path={}",
                             program_base, header.size(), header_path.string());
                } else {
                    LOG_ERROR(Shader, "Cannot write native shader header {}", header_path.string());
                }
            }
        }
    }
    if (!found.code && !lowered) {
        if (const char* directory = std::getenv("EDEN_NATIVE_SHADER_OVERRIDE_DIR")) {
            const auto key = fmt::format("{:016x}", Shader::Ampere::NativeProgramKey(native));
            const auto folder = std::filesystem::path{directory};
            std::filesystem::create_directories(folder);
            const auto source_path = folder / (key + ".source.bin");
            const auto override_path = folder / (key + ".override.bin");
            {
                std::ofstream dump(source_path, std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native.data()), native.size() * sizeof(Shader::Ampere::Instruction));
            }
            {
                // Retain mapped bytes beyond the first EXIT for investigating
                // native CALL targets in compiler library subroutines.
                std::ofstream mapped(folder / (key + ".mapped.bin"), std::ios::binary);
                std::array<u8, 0x1000> block{};
                for (u32 offset = 0; offset < 0x10000; offset += 0x1000) {
                    if (!gpu_memory->IsFullyMappedRange(program_base + offset, block.size())) break;
                    gpu_memory->ReadBlock(program_base + offset, block.data(), block.size());
                    mapped.write(reinterpret_cast<const char*>(block.data()), block.size());
                }
            }
            LOG_WARNING(Shader, "Native shader paused: key={}, offset={:#x}, reason={}, source={}",
                        key, lowered.error_offset, lowered.error, source_path.string());
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes{30};
            while (!gpu_memory->IsShuttingDown() && std::chrono::steady_clock::now() < deadline) {
                if (std::filesystem::exists(override_path)) {
                    std::ifstream file(override_path, std::ios::binary | std::ios::ate);
                    const auto length = file.tellg();
                    if (length <= 0 || length > 16 * 1024 * 1024) {
                        throw Shader::RuntimeError("Invalid native shader override size");
                    }
                    std::vector<u8> bytes(static_cast<size_t>(length));
                    file.seekg(0);
                    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
                    if (!file) throw Shader::RuntimeError("Native shader override read failed");
                    const u32 source_bytes = Shader::Ampere::ProgramOverrideSourceSize(bytes);
                    if (source_bytes > native.size() * sizeof(Shader::Ampere::Instruction)) {
                        if (!gpu_memory->IsFullyMappedRange(program_base + native_header_size, source_bytes))
                            throw Shader::RuntimeError("Unmapped extended native shader source");
                        native.resize(source_bytes / sizeof(Shader::Ampere::Instruction));
                        gpu_memory->ReadBlock(program_base + native_header_size, native.data(), source_bytes);
                    }
                    lowered = Shader::Ampere::DecodeProgramOverride(bytes, native);
                    if (!lowered) throw Shader::RuntimeError("Native shader override rejected: {}", lowered.error);
                    overridden = true;
                    LOG_WARNING(Shader, "Native shader resumed: key={}, instructions={}", key, lowered.instructions.size());
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{100});
            }
        }
    }
    if (!found.code && !lowered) {
        const size_t failed = lowered.error_offset / sizeof(Shader::Ampere::Instruction);
        const auto words = failed < native.size() ? native[failed].words : std::array<u32, 4>{};
        throw Shader::RuntimeError(
            "Native shader at {:#x}, instruction {:#x}: {} [op {:03x} {:08x} {:08x} {:08x} {:08x}]",
            program_base, lowered.error_offset, lowered.error, words[0] & 0xfff, words[0],
            words[1], words[2], words[3]);
    }
    std::shared_ptr<const LoweredNativeCode> result = found.code;
    if (!result) {
        result = MakeLoweredNativeCode(native, sph, native_header_size, verified_traps, lowered);
    }
    const auto& lowered_code = result->code;
    // EDEN_NX2_REJECT_HASH=<hash>,...: like EDEN_NX2_REJECT, but by native program key (shader
    // dump file names) or shader hash (frame dump "shaders"), which stay the same between runs.
    if (static const std::string reject_hashes = [] {
            const char* text = std::getenv("EDEN_NX2_REJECT_HASH");
            std::string list = text ? "," + std::string{text} + "," : std::string{};
            std::ranges::transform(list, list.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return list;
        }();
        !reject_hashes.empty()) {
        for (const u64 hash : {result->native_program_key, NativeShaderHash(lowered_code)}) {
            if (reject_hashes.find(fmt::format(",{:016x},", hash)) != std::string::npos) {
                throw Shader::RuntimeError("Native shader at {:#x} ({:016x}) rejected by EDEN_NX2_REJECT_HASH",
                                           program_base, hash);
            }
        }
    }
    // The same list, changed at runtime by the debug server (docs/nx2/debug-mcp.md).
    if (VideoCore::Nx2Debug::AnyRejected()) {
        for (const u64 hash : {result->native_program_key, NativeShaderHash(lowered_code)}) {
            if (VideoCore::Nx2Debug::IsRejected(hash)) {
                throw Shader::RuntimeError("Native shader at {:#x} ({:016x}) rejected by the debug server",
                                           program_base, hash);
            }
        }
    }
    if (found.code) {
        LOG_DEBUG(Shader, "Native shader at {:#x} from its v2 record: {} source bytes",
                  program_base, result->source_size);
        return result;
    }
    // A program from an override file depends on more than the lowering: the v2 cache leaves it out.
    if (cache_v2 && !overridden) {
        program_record = std::make_shared<const ShaderCacheV2::ProgramRecord>(
            ShaderCacheV2::MakeProgramRecord(stage, native_header_size, sph, native, inputs,
                                             *result));
    }
    LOG_DEBUG(Shader, "Lowered native shader at {:#x}: {} source bytes, {} instructions",
              program_base, result->source_size, lowered.instructions.size());
    return result;
}

GraphicsEnvironment::GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                         Tegra::MemoryManager& gpu_memory_,
                                         Maxwell::ShaderType program, GPUVAddr program_base_,
                                         u32 start_address_,
                                         std::shared_ptr<const LoweredNativeCode> lowered_native_,
                                         ShaderCacheV2::Cache* cache_v2)
    : GenericEnvironment{gpu_memory_, program_base_, start_address_}, maxwell3d{&maxwell3d_} {
    gpu_memory->ReadBlock(program_base + start_address, &sph, sizeof(sph));
    initial_offset = sizeof(sph);
    texture_bound = maxwell3d->regs.bindless_texture_const_buffer_slot;
    const auto& regs{maxwell3d->regs};
    const bool native{maxwell3d->GetNativeMacroEngine().native_turing};
    static const bool geometry_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_EXPERIMENTAL_GEOMETRY");
        return !value || std::string_view{value} != "0";
    }();
    const bool geometry{native && program == Maxwell::ShaderType::Geometry && geometry_enabled};
    // Task and mesh programs: the task in the vertex slot, its mesh program in the tessellation
    // slot (without a task, the mesh program is in the vertex slot).
    const bool mesh_slot{native && regs.IsMeshEnabled() &&
                         (program == Maxwell::ShaderType::VertexB ||
                          program == Maxwell::ShaderType::Tessellation)};
    if (native) {
        const bool tessellation =
            (program == Maxwell::ShaderType::TessellationInit ||
             program == Maxwell::ShaderType::Tessellation) &&
            sph.common0.shader_type == static_cast<u32>(program);
        if (program != Maxwell::ShaderType::VertexB && program != Maxwell::ShaderType::Pixel &&
            !geometry && !mesh_slot && !tessellation) {
            if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP"); directory &&
                gpu_memory->IsFullyMappedRange(program_base, 128)) {
                constexpr u32 MaxCaptureInstructions = 4096;
                std::array<u8, 128> header{};
                gpu_memory->ReadBlock(program_base, header.data(), header.size());
                std::vector<Shader::Ampere::Instruction> instructions;
                Shader::Ampere::ProgramCaptureState capture;
                auto status = Shader::Ampere::ProgramCaptureState::Result::Continue;
                NativeInstructionReader reader{*gpu_memory};
                for (u32 index = 0; index < MaxCaptureInstructions; ++index) {
                    const GPUVAddr address = program_base + header.size() + index * 16;
                    Shader::Ampere::Instruction instruction;
                    if (!reader.TryRead(address, instruction)) {
                        if (!gpu_memory->IsFullyMappedRange(address, sizeof(instruction))) {
                            break;
                        }
                        gpu_memory->ReadBlock(address, &instruction, sizeof(instruction));
                    }
                    instructions.push_back(instruction);
                    status = capture.Observe(instruction, index);
                    if (status != Shader::Ampere::ProgramCaptureState::Result::Continue) {
                        break;
                    }
                }
                const auto folder = std::filesystem::path{directory};
                std::error_code error;
                std::filesystem::create_directories(folder, error);
                if (error) {
                    LOG_ERROR(Shader, "Cannot create rejected shader dump directory: {}",
                              error.message());
                } else {
                    const auto path = folder / fmt::format("rejected-stage-{}-{:x}.bin",
                                                          static_cast<u32>(program), program_base);
                    std::ofstream dump(path, std::ios::binary);
                    dump.write(reinterpret_cast<const char*>(header.data()), header.size());
                    dump.write(reinterpret_cast<const char*>(instructions.data()),
                               instructions.size() * sizeof(Shader::Ampere::Instruction));
                    dump.close();
                    if (dump) {
                        LOG_INFO(Shader,
                                 "Captured rejected native stage {} at {:#x}: {} header bytes, "
                                 "{} instructions, complete={}, invalid_target={}, path={}",
                                 static_cast<u32>(program), program_base, header.size(),
                                 instructions.size(),
                                 status == Shader::Ampere::ProgramCaptureState::Result::Complete,
                                 status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget,
                                 path.string());
                    } else {
                        LOG_ERROR(Shader, "Cannot write rejected shader dump {}", path.string());
                    }
                }
            }
            throw Shader::RuntimeError("Unsupported native graphics shader stage {}", static_cast<u32>(program));
        }
        if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS")) {
            const u32 native_stage = program == Maxwell::ShaderType::Pixel ? 4 :
                                     program == Maxwell::ShaderType::Geometry ? 3 :
                                     program == Maxwell::ShaderType::Tessellation ? 2 :
                                     program == Maxwell::ShaderType::TessellationInit ? 1 : 0;
            const auto& table = maxwell3d->state.shader_stages[native_stage].const_buffers[1];
            if (table.enabled && table.size && table.size <= Tegra::Engines::Maxwell3D::Regs::MaxConstBufferSize) {
                std::vector<u8> bytes(table.size);
                gpu_memory->ReadBlock(table.address, bytes.data(), bytes.size());
                std::ofstream dump(fmt::format("native-graphics-cbuf-{}-1.bin", static_cast<u32>(program)), std::ios::binary);
                dump.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            }
        }
    }
    gp_passthrough_mask = maxwell3d->regs.post_vtg_shader_attrib_skip_mask;
    switch (program) {
    case Maxwell::ShaderType::VertexA:
        stage = Shader::Stage::VertexA;
        stage_index = 0;
        break;
    case Maxwell::ShaderType::VertexB:
        stage = Shader::Stage::VertexB;
        stage_index = 0;
        break;
    case Maxwell::ShaderType::TessellationInit:
        stage = Shader::Stage::TessellationControl;
        stage_index = 1;
        break;
    case Maxwell::ShaderType::Tessellation:
        stage = Shader::Stage::TessellationEval;
        stage_index = 2;
        break;
    case Maxwell::ShaderType::Geometry:
        stage = Shader::Stage::Geometry;
        stage_index = 3;
        break;
    case Maxwell::ShaderType::Pixel:
        stage = Shader::Stage::Fragment;
        stage_index = 4;
        break;
    default:
        ASSERT_MSG(false, "Invalid program={}", program);
        break;
    }
    if (mesh_slot) {
        const bool has_task{regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation)};
        stage = program == Maxwell::ShaderType::VertexB && has_task ? Shader::Stage::Task
                                                                    : Shader::Stage::Mesh;
        mesh_registers = {regs.reg_array[Maxwell::MeshShaderA], regs.reg_array[Maxwell::MeshShaderB],
                          regs.reg_array[Maxwell::MeshInitShader]};
        if (stage == Shader::Stage::Mesh && regs.IsShaderConfigEnabled(Maxwell::ShaderType::Geometry)) {
            const auto& geometry_pipeline{
                regs.pipelines[static_cast<size_t>(Maxwell::ShaderType::Geometry)]};
            const GPUVAddr address{(static_cast<u64>(geometry_pipeline.reserved[0] & 0xff) << 32) |
                                   geometry_pipeline.reserved[1]};
            gpu_memory->ReadBlock(address, &mesh_primitive_header, sizeof(mesh_primitive_header));
        }
    }
    if (native) {
        if (!lowered_native_ && std::getenv("EDEN_NX2_SHADER_DUMP") &&
            (stage == Shader::Stage::TessellationControl || stage == Shader::Stage::TessellationEval)) {
            LOG_INFO(Shader,
                     "Native tessellation context at {:#x}: stage={} domain={} spacing={} output={} patch_vertices={}",
                     program_base, static_cast<u32>(stage),
                     static_cast<u32>(regs.tessellation.params.domain_type.Value()),
                     static_cast<u32>(regs.tessellation.params.spacing.Value()),
                     static_cast<u32>(regs.tessellation.params.output_primitives.Value()),
                     regs.patch_vertices);
        }
        InitializeNativeCode(std::move(lowered_native_), 128, geometry, cache_v2);
    }
    const u64 local_size{sph.LocalMemorySize()};
    ASSERT(local_size <= (std::numeric_limits<u32>::max)());
    local_memory_size = static_cast<u32>(local_size) + sph.common3.shader_local_memory_crs_size;
    is_proprietary_driver = maxwell3d->GetNativeMacroEngine().native_turing || texture_bound == 2;
    has_hle_engine_state =
        maxwell3d->engine_state == Tegra::Engines::Maxwell3D::EngineHint::OnHLEMacro;
}

u32 GraphicsEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto& cbuf{maxwell3d->state.shader_stages[stage_index].const_buffers[cbuf_index]};
    ASSERT(cbuf.enabled);
    u32 value{};
    if (cbuf_offset < cbuf.size) {
        value = gpu_memory->Read<u32>(cbuf.address + cbuf_offset);
    }
    cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
    return value;
}

std::optional<Shader::ReplaceConstant> GraphicsEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                                  u32 offset) {
    if (!has_hle_engine_state) {
        return std::nullopt;
    }
    const u64 key = (static_cast<u64>(bank) << 32) | static_cast<u64>(offset);
    auto it = maxwell3d->replace_table.find(key);
    if (it == maxwell3d->replace_table.end()) {
        return std::nullopt;
    }
    const auto converted_value = [](Tegra::Engines::Maxwell3D::HLEReplacementAttributeType name) {
        switch (name) {
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::BaseVertex:
            return Shader::ReplaceConstant::BaseVertex;
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::BaseInstance:
            return Shader::ReplaceConstant::BaseInstance;
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::DrawID:
            return Shader::ReplaceConstant::DrawID;
        default:
            UNREACHABLE();
        }
    }(it->second);
    cbuf_replacements.emplace(key, converted_value);
    return converted_value;
}

Shader::TextureType GraphicsEnvironment::ReadTextureType(u32 handle) {
    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    auto entry =
        ReadTextureInfo(regs.tex_header.Address(), regs.tex_header.limit, via_header_index, handle);
    const Shader::TextureType result{ConvertTextureType(entry)};
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat GraphicsEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    auto entry =
        ReadTextureInfo(regs.tex_header.Address(), regs.tex_header.limit, via_header_index, handle);
    const Shader::TexturePixelFormat result(ConvertTexturePixelFormat(entry));
    texture_pixel_formats.emplace(handle, result);
    return result;
}

bool GraphicsEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 GraphicsEnvironment::ReadViewportTransformState() {
    const auto& regs{maxwell3d->regs};
    viewport_transform_state = regs.viewport_scale_offset_enabled;
    return viewport_transform_state;
}

ComputeEnvironment::ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                       Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                       u32 start_address_,
                                       std::shared_ptr<const LoweredNativeCode> lowered_native_,
                                       ShaderCacheV2::Cache* cache_v2)
    : GenericEnvironment{gpu_memory_, program_base_, start_address_}, kepler_compute{
                                                                          &kepler_compute_} {
    const auto& qmd{kepler_compute->launch_description};
    stage = Shader::Stage::Compute;
    local_memory_size = qmd.local_pos_alloc + qmd.local_crs_alloc;
    texture_bound = kepler_compute->regs.tex_cb_index;
    is_proprietary_driver = texture_bound == 2;
    shared_memory_size = qmd.shared_alloc;
    workgroup_size = {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z};
    if (kepler_compute->IsNativeMode()) {
        initial_offset = sizeof(sph);
        is_proprietary_driver = true;
        InitializeNativeCode(std::move(lowered_native_), 0, false, cache_v2);
        start_address = sizeof(sph);
    }
}

std::optional<u32> ReadNativeGuardConstant(Tegra::Engines::KeplerCompute& compute,
                                         Tegra::MemoryManager& memory, u32 bank, u32 offset) {
    const auto& qmd = compute.launch_description;
    if (bank >= qmd.const_buffer_config.size() ||
        !(qmd.const_buffer_enable_mask.Value() & (1U << bank)))
        return std::nullopt;
    const auto& buffer = qmd.const_buffer_config[bank];
    const GPUVAddr address = buffer.Address() + offset;
    if (offset % 4 || offset > buffer.size || buffer.size - offset < sizeof(u32) ||
        address < buffer.Address() || !memory.IsFullyMappedRange(address, sizeof(u32)))
        return std::nullopt;
    u32 value{};
    memory.ReadBlock(address, &value, sizeof(value));
    return value;
}

std::optional<u32> ComputeEnvironment::ReadTrapGuardConstant(u32 bank, u32 offset) {
    return ReadNativeGuardConstant(*kepler_compute, *gpu_memory, bank, offset);
}

u32 ComputeEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto& qmd{kepler_compute->launch_description};
    ASSERT(((qmd.const_buffer_enable_mask.Value() >> cbuf_index) & 1) != 0);
    const auto& cbuf{qmd.const_buffer_config[cbuf_index]};
    u32 value{};
    if (cbuf_offset < cbuf.size) {
        value = gpu_memory->Read<u32>(cbuf.Address() + cbuf_offset);
    }
    cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
    return value;
}

Shader::TextureType ComputeEnvironment::ReadTextureType(u32 handle) {
    const auto& regs{kepler_compute->regs};
    const auto& qmd{kepler_compute->launch_description};
    auto entry = ReadTextureInfo(regs.tic.Address(), regs.tic.limit, qmd.linked_tsc != 0, handle);
    const Shader::TextureType result{ConvertTextureType(entry)};
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat ComputeEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto& regs{kepler_compute->regs};
    const auto& qmd{kepler_compute->launch_description};
    auto entry = ReadTextureInfo(regs.tic.Address(), regs.tic.limit, qmd.linked_tsc != 0, handle);
    const Shader::TexturePixelFormat result(ConvertTexturePixelFormat(entry));
    const bool inserted{texture_pixel_formats.emplace(handle, result).second};
    if (inserted && result == Shader::TexturePixelFormat::R16_FLOAT &&
        kepler_compute->IsNativeMode() && std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS")) {
        const auto pair{Tegra::Texture::TexturePair(handle, qmd.linked_tsc != 0)};
        const u32 index{Tegra::Texture::TextureHeaderIndex(pair.first, regs.tic.limit,
                                                        gpu_memory->UsesNativeTiling())};
        const GPUVAddr tic_gpu{regs.tic.Address() + index * sizeof(entry)};
        const auto tic_device{gpu_memory->GpuToCpuAddress(tic_gpu)};
        const auto image_device{gpu_memory->GpuToCpuAddress(entry.Address())};
        LOG_INFO(HW_GPU,
                 "Native compute R16_FLOAT descriptor: program_base={:#x} raw_handle={:#010x} "
                 "tic_pool={:#x} tic_limit={:#x} tic_low20={:#x} tic_candidate={:#x} tic_index={:#x} "
                 "via_header_index={} native_tiling={} tic_gpu={:#x} tic_mapped={} "
                 "tic_device={:#x} image_gpu={:#x} image_mapped={} image_device={:#x} "
                 "tic64={:016x} {:016x} {:016x} {:016x} "
                 "pixel_format={} texture_type={} tic_format={:#x} tic_type={} size={}x{}x{}",
                 program_base, handle, regs.tic.Address(), regs.tic.limit, handle & 0xfffff,
                 pair.first, index, qmd.linked_tsc != 0, gpu_memory->UsesNativeTiling(), tic_gpu,
                 tic_device.has_value(), tic_device.value_or(0), entry.Address(),
                 image_device.has_value(), image_device.value_or(0), entry.raw[0], entry.raw[1],
                 entry.raw[2], entry.raw[3], result, ConvertTextureType(entry),
                 static_cast<u32>(entry.format.Value()),
                 static_cast<u32>(entry.texture_type.Value()), entry.Width(), entry.Height(),
                 entry.Depth());
        for (const auto& [key, value] : cbuf_values) {
            if (value != handle) {
                continue;
            }
            const u32 bank{static_cast<u32>(key >> 32)};
            const u32 offset{static_cast<u32>(key)};
            const auto& cbuf{qmd.const_buffer_config[bank]};
            LOG_INFO(HW_GPU,
                     "Native compute R16_FLOAT cached cbuf candidate: program_base={:#x} "
                     "raw_handle={:#010x} bank={} offset={:#x} base={:#x} size={:#x} enabled={}",
                     program_base, handle, bank, offset, cbuf.Address(), cbuf.size.Value(),
                     ((qmd.const_buffer_enable_mask.Value() >> bank) & 1) != 0);
        }
    }
    return result;
}

bool ComputeEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 ComputeEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

void FileEnvironment::Deserialize(std::istream& file, std::streampos end) {
    u64 code_size{};
    u64 num_texture_types{};
    u64 num_texture_pixel_formats{};
    u64 num_cbuf_values{};
    u64 num_storage_pointer_types{};
    u64 num_cbuf_replacement_values{};
    file.read(reinterpret_cast<char*>(&code_size), sizeof(code_size))
        .read(reinterpret_cast<char*>(&num_texture_types), sizeof(num_texture_types))
        .read(reinterpret_cast<char*>(&num_texture_pixel_formats),
              sizeof(num_texture_pixel_formats))
        .read(reinterpret_cast<char*>(&num_cbuf_values), sizeof(num_cbuf_values))
        .read(reinterpret_cast<char*>(&num_storage_pointer_types),
              sizeof(num_storage_pointer_types))
        .read(reinterpret_cast<char*>(&num_cbuf_replacement_values),
              sizeof(num_cbuf_replacement_values))
        .read(reinterpret_cast<char*>(&local_memory_size), sizeof(local_memory_size))
        .read(reinterpret_cast<char*>(&texture_bound), sizeof(texture_bound))
        .read(reinterpret_cast<char*>(&start_address), sizeof(start_address))
        .read(reinterpret_cast<char*>(&read_lowest), sizeof(read_lowest))
        .read(reinterpret_cast<char*>(&read_highest), sizeof(read_highest))
        .read(reinterpret_cast<char*>(&viewport_transform_state), sizeof(viewport_transform_state))
        .read(reinterpret_cast<char*>(&stage), sizeof(stage));
    if (!IsShaderStage(stage)) {
        throw std::ios_base::failure("Invalid shader stage in the pipeline cache");
    }
    // A damaged record can hold any count. Each environment holds the code of its address range
    // and no more than MAX_RECORDED_VALUES values of a kind, and what it claims must fit in the
    // file: all are checked before anything is allocated or read.
    if (read_highest < read_lowest || code_size != u64{read_highest} - read_lowest + INST_SIZE) {
        throw std::ios_base::failure("Shader environment code doesn't match its address range");
    }
    for (const u64 count : {num_texture_types, num_texture_pixel_formats, num_cbuf_values,
                            num_storage_pointer_types, num_cbuf_replacement_values}) {
        if (count > MAX_RECORDED_VALUES) {
            throw std::ios_base::failure("Too many values in a shader environment");
        }
    }
    u64 left{static_cast<u64>(end - file.tellg())};
    const auto claim{[&left](u64 count, u64 entry_size) {
        if (count > left / entry_size) {
            throw std::ios_base::failure("Shader environment past the end of the pipeline cache");
        }
        left -= count * entry_size;
    }};
    claim(code_size, 1);
    claim(num_texture_types, sizeof(u32) + sizeof(Shader::TextureType));
    claim(num_texture_pixel_formats, sizeof(u32) + sizeof(Shader::TexturePixelFormat));
    claim(num_cbuf_values, sizeof(u64) + sizeof(u32));
    claim(num_storage_pointer_types, sizeof(u64) + sizeof(u8));
    claim(num_cbuf_replacement_values, sizeof(u64) + sizeof(Shader::ReplaceConstant));
    code.resize(Common::DivCeil(code_size, sizeof(u64)));
    file.read(reinterpret_cast<char*>(code.data()), code_size);
    for (size_t i = 0; i < num_texture_types; ++i) {
        u32 key;
        Shader::TextureType type;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&type), sizeof(type));
        texture_types.emplace(key, type);
    }
    for (size_t i = 0; i < num_texture_pixel_formats; ++i) {
        u32 key;
        Shader::TexturePixelFormat format;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&format), sizeof(format));
        texture_pixel_formats.emplace(key, format);
    }
    for (size_t i = 0; i < num_cbuf_values; ++i) {
        u64 key;
        u32 value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        cbuf_values.emplace(key, value);
    }
    if (const auto it{cbuf_values.find(READ_PROPERTIES_KEY)}; it != cbuf_values.end()) {
        recorded_read_properties = it->second;
        cbuf_values.erase(it);
    }
    if (const auto it{cbuf_values.find(NULL_TEXTURE_2D_KEY)}; it != cbuf_values.end()) {
        cbuf_values.erase(it);
    } else {
        // An older build read every empty descriptor as 1D (issue #846), whatever its index, or
        // the game binds a real 1D texture there: the record can't tell. Drop every 1D type, so
        // that the pipeline is built again from the game (LacksRecordedValue).
        for (auto it{texture_types.begin()}; it != texture_types.end();) {
            if (it->second == Shader::TextureType::Color1D) {
                it = texture_types.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (size_t i = 0; i < num_storage_pointer_types; ++i) {
        u64 key;
        u8 value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        storage_pointer_types.emplace(key, value);
    }
    for (size_t i = 0; i < num_cbuf_replacement_values; ++i) {
        u64 key;
        Shader::ReplaceConstant value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        cbuf_replacements.emplace(key, value);
    }
    if (stage == Shader::Stage::Compute) {
        file.read(reinterpret_cast<char*>(&workgroup_size), sizeof(workgroup_size))
            .read(reinterpret_cast<char*>(&shared_memory_size), sizeof(shared_memory_size));
        initial_offset = Shader::Ampere::HasNativeAddressMap(code) ? sizeof(sph) : 0;
    } else {
        file.read(reinterpret_cast<char*>(&sph), sizeof(sph));
        initial_offset = sizeof(sph);
        if (stage == Shader::Stage::Geometry) {
            file.read(reinterpret_cast<char*>(&gp_passthrough_mask), sizeof(gp_passthrough_mask));
        }
        if (stage == Shader::Stage::Task || stage == Shader::Stage::Mesh) {
            file.read(reinterpret_cast<char*>(&mesh_registers), sizeof(mesh_registers))
                .read(reinterpret_cast<char*>(&mesh_primitive_header),
                      sizeof(mesh_primitive_header));
        }
    }
    // Native programs fold bank 1 reads as they did when they were built.
    is_proprietary_driver = texture_bound == 2 || Shader::Ampere::HasNativeAddressMap(code);
}

size_t FileEnvironment::MemorySize() const {
    const auto map_bytes = [](const auto& map) {
        return (map.bucket_count() + 1) *
               (sizeof(typename std::decay_t<decltype(map)>::value_type) + 2);
    };
    return sizeof(*this) + code.capacity() * sizeof(u64) + map_bytes(texture_types) +
           map_bytes(texture_pixel_formats) + map_bytes(cbuf_values) +
           map_bytes(storage_pointer_types) + map_bytes(cbuf_replacements);
}

void FileEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    DumpImpl(pipeline_hash, shader_hash, code, read_highest, read_lowest, initial_offset, stage);
}

u32 FileEnvironment::TranslateProgramAddress(u32 address) const {
    if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) return *translated;
    if (Shader::Ampere::HasNativeAddressMap(code))
        throw Shader::RuntimeError("Invalid cached native indirect branch address {:#x}", address);
    return address;
}

bool FileEnvironment::IsNativeProgram() const {
    return Shader::Ampere::HasNativeAddressMap(code);
}

std::vector<std::pair<u32, u32>> FileEnvironment::NullStoragePointerSlots() const {
    return VideoCommon::NullStoragePointerSlots(cbuf_values, storage_pointer_types);
}

std::vector<u64> FileEnvironment::MappedStoragePointerKeys() const {
    return VideoCommon::MappedStoragePointerKeys(storage_pointer_types);
}

/// Recorded classification of the constant buffer word at `index`/`offset` as a storage buffer
/// pointer. Without a record, the guess when one is set, else throws.
bool FileEnvironment::IsCbufStorageBufferPointer(u32 index, u32 offset) {
    const auto it = storage_pointer_types.find(MakeCbufKey(index, offset));
    if (it == storage_pointer_types.end()) {
        if (guess) {
            return index == 0 ? guess->storage_pointer_bank0 : guess->storage_pointer_other;
        }
        lacks_recorded_value = true;
        throw Shader::LogicError("Uncached storage pointer classification");
    }
    return it->second != 0;
}

/// Same as IsCbufStorageBufferPointer for a pointer read through an indirect table.
bool FileEnvironment::IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) {
    const auto it = storage_pointer_types.find(MakeIndirectCbufKey(index, offset, indirect));
    if (it == storage_pointer_types.end()) {
        if (guess) {
            return index == 0 ? guess->storage_pointer_bank0 : guess->storage_pointer_other;
        }
        lacks_recorded_value = true;
        throw Shader::LogicError("Uncached storage pointer table classification");
    }
    return it->second != 0;
}

u64 FileEnvironment::ReadInstruction(u32 address) {
    if (address < read_lowest || address > read_highest) {
        throw Shader::LogicError("Out of bounds address {}", address);
    }
    return code[(address - read_lowest) / sizeof(u64)];
}

/// Recorded constant buffer value at `cbuf_index`/`cbuf_offset`. Without a record, the guess when
/// one is set, else throws.
u32 FileEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto it{cbuf_values.find(MakeCbufKey(cbuf_index, cbuf_offset))};
    if (it == cbuf_values.end()) {
        if (guess) {
            return cbuf_index == 0 ? guess->cbuf_bank0 : guess->cbuf_other;
        }
        lacks_recorded_value = true;
        throw Shader::LogicError("Uncached read texture type");
    }
    return it->second;
}

/// Recorded texture type of `handle`. Without a record, the guess when one is set, else throws.
Shader::TextureType FileEnvironment::ReadTextureType(u32 handle) {
    const auto it{texture_types.find(handle)};
    if (it == texture_types.end()) {
        if (guess) {
            return guess->texture_type;
        }
        lacks_recorded_value = true;
        throw Shader::LogicError("Uncached read texture type");
    }
    return it->second;
}

/// Recorded texture pixel format of `handle`. Without a record, the guess when one is set, else
/// throws.
Shader::TexturePixelFormat FileEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto it{texture_pixel_formats.find(handle)};
    if (it == texture_pixel_formats.end()) {
        if (guess) {
            return guess->pixel_format;
        }
        lacks_recorded_value = true;
        throw Shader::LogicError("Uncached read texture pixel format");
    }
    return it->second;
}

bool FileEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 FileEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

u32 FileEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 FileEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

u32 FileEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

std::array<u32, 3> FileEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

std::optional<Shader::ReplaceConstant> FileEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                              u32 offset) {
    const u64 key = (static_cast<u64>(bank) << 32) | static_cast<u64>(offset);
    auto it = cbuf_replacements.find(key);
    if (it == cbuf_replacements.end()) {
        return std::nullopt;
    }
    return it->second;
}

// Beside a cache file: the rewrite being written, and the size the cache had when rewriting it
// last failed
static std::filesystem::path RewritePath(const std::filesystem::path& filename) {
    auto path = filename;
    path += ".tmp";
    return path;
}

static std::filesystem::path FailedRewritePath(const std::filesystem::path& filename) {
    auto path = filename;
    path += ".rewrite-failed";
    return path;
}

void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version,
                       u32 read_properties) {
    // Another emulator loading the cache, appending to it or rewriting it goes first. One that
    // keeps it through the whole wait (see PipelineCacheLockWait) may change the file at any time:
    // the pipeline is recorded when a later boot builds it again.
    const Common::FS::FileLock lock{PipelineCacheLockPath(filename), PipelineCacheLockWait};
    if (lock.Busy()) {
        return;
    }
    std::ofstream file(filename, std::ios::binary | std::ios::ate | std::ios::app);
    const std::streamoff size{file.tellp()};
    if (!file.is_open() || size < 0) {
        LOG_ERROR(Common_Filesystem, "Failed to open pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    try {
        file.exceptions(std::ifstream::failbit);
        if (size == 0) {
            // Write header
            file.write(MAGIC_NUMBER.data(), MAGIC_NUMBER.size())
                .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));
        }
        if (!std::ranges::all_of(envs, &GenericEnvironment::CanBeSerialized)) {
            return;
        }
        const u32 num_envs{static_cast<u32>(envs.size())};
        file.write(reinterpret_cast<const char*>(&num_envs), sizeof(num_envs));
        for (const GenericEnvironment* const env : envs) {
            env->Serialize(file, read_properties);
        }
        file.write(key.data(), key.size_bytes());
        file.close();
    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        // Cut off the part of the record that was written (a full disk), so that the next one
        // follows complete records.
        file.exceptions(std::ios::goodbit);
        file.close();
        std::error_code error;
        std::filesystem::resize_file(filename, static_cast<std::uintmax_t>(size), error);
        if (error) {
            LOG_ERROR(Common_Filesystem,
                      "Failed to cut pipeline cache file {} back to {} bytes: {}",
                      Common::FS::PathToUTF8String(filename), size, error.message());
        }
    }
}

// Reads the environments of the record at the stream's position and passes them to the callback of
// their kind, which reads the key
template <typename Stream, typename LoadCompute, typename LoadGraphics>
static void ReadPipelineEnvironments(Stream& file, std::streampos end, LoadCompute& load_compute,
                                     LoadGraphics& load_graphics) {
    const auto is_compute{[](const FileEnvironment& environment) {
        return environment.ShaderStage() == Shader::Stage::Compute;
    }};
    u32 num_envs{};
    file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
    if (num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
        throw std::ios_base::failure("Invalid number of shader environments");
    }
    std::vector<FileEnvironment> envs(num_envs);
    for (FileEnvironment& env : envs) {
        env.Deserialize(file, end);
    }
    // A compute record holds one environment, a graphics record no compute one.
    const bool compute{is_compute(envs.front())};
    if (compute ? num_envs != 1 : std::ranges::any_of(envs, is_compute)) {
        throw std::ios_base::failure("Invalid shader stages in the pipeline cache");
    }
    if (compute) {
        load_compute(file, std::move(envs.front()));
    } else {
        load_graphics(file, std::move(envs));
    }
}

// LoadPipelines, holding the cache's lock when `locked`
static void ReadPipelines(
    std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    bool locked, Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics) try {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return;
    }
    file.exceptions(std::ifstream::failbit);
    const auto end{file.tellg()};
    file.seekg(0, std::ios::beg);

    std::array<char, 8> magic_number;
    u32 cache_version;
    file.read(magic_number.data(), magic_number.size())
        .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
    if (magic_number != MAGIC_NUMBER || cache_version != expected_cache_version) {
        file.close();
        if (Common::FS::RemoveFile(filename)) {
            if (magic_number != MAGIC_NUMBER) {
                LOG_ERROR(Common_Filesystem, "Invalid pipeline cache file");
            }
            if (cache_version != expected_cache_version) {
                LOG_INFO(Common_Filesystem, "Deleting old pipeline cache");
            }
        } else {
            LOG_ERROR(Common_Filesystem,
                      "Invalid pipeline cache file and failed to delete it in \"{}\"",
                      Common::FS::PathToUTF8String(filename));
        }
        return;
    }
    // Where the record being read begins: the end of the complete records
    std::streampos record_begin{file.tellg()};
    try {
        for (; record_begin != end; record_begin = file.tellg()) {
            if (stop_loading.stop_requested()) {
                return;
            }
            ReadPipelineEnvironments(file, end, load_compute, load_graphics);
        }
    } catch (const std::bad_alloc& e) {
        // Memory is short, but the file is fine: it stays as it is.
        LOG_ERROR(Common_Filesystem, "Stopped loading pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), e.what());
    } catch (const std::ios_base::failure& e) {
        // Keep the records before the damaged one (one cut short by a crash or a full disk): the
        // next record is appended after them.
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        file.close();
        if (!locked) {
            // Another emulator may be appending it: a load that holds the lock cuts it off.
            return;
        }
        const std::streamoff complete{record_begin};
        std::error_code error;
        std::filesystem::resize_file(filename, static_cast<std::uintmax_t>(complete), error);
        if (error) {
            // The file stays: the records before it still load, and the next load tries again.
            LOG_ERROR(Common_Filesystem,
                      "Failed to cut pipeline cache file {} back to {} bytes: {}",
                      Common::FS::PathToUTF8String(filename), complete, error.message());
            return;
        }
        LOG_WARNING(
            Common_Filesystem,
            "Dropped the damaged pipeline cache record at byte {} and the {} bytes after it",
            complete, std::streamoff{end} - complete);
    }

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

void LoadPipelines(
    std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics) {
    std::error_code error;
    if (!std::filesystem::exists(filename, error)) {
        return;
    }
    // Another emulator appending to the cache or rewriting it finishes first, so that what it is
    // writing is never taken for a damaged record.
    const Common::FS::FileLock lock{PipelineCacheLockPath(filename), PipelineCacheLockWait,
                                    stop_loading};
    if (stop_loading.stop_requested()) {
        return;
    }
    if (lock.Held()) {
        std::filesystem::remove(RewritePath(filename), error); // A rewrite cut off
    }
    ReadPipelines(stop_loading, filename, expected_cache_version, lock.Held(),
                  std::move(load_compute), std::move(load_graphics));
}

namespace {

// A stream over bytes in memory
class SpanBuffer final : public std::streambuf {
public:
    explicit SpanBuffer(std::span<const u8> bytes) {
        char* const begin = const_cast<char*>(reinterpret_cast<const char*>(bytes.data()));
        setg(begin, begin, begin + bytes.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                     std::ios_base::openmode mode) override {
        const off_type base = direction == std::ios_base::beg   ? 0
                              : direction == std::ios_base::cur ? gptr() - eback()
                                                                : egptr() - eback();
        return seekpos(pos_type(base + offset), mode);
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode) override {
        const off_type offset{position};
        if (offset < 0 || offset > egptr() - eback()) {
            return pos_type(off_type(-1));
        }
        setg(eback(), eback() + offset, egptr());
        return position;
    }
};

} // Anonymous namespace

std::vector<u8> SerializePipelinePayload(std::span<const char> key,
                                         std::span<const GenericEnvironment* const> envs,
                                         u32 read_properties) {
    if (!std::ranges::all_of(envs, &GenericEnvironment::CanBeSerialized)) {
        return {};
    }
    std::ostringstream stream(std::ios::binary);
    const u32 num_envs{static_cast<u32>(envs.size())};
    stream.write(reinterpret_cast<const char*>(&num_envs), sizeof(num_envs));
    for (const GenericEnvironment* const env : envs) {
        env->Serialize(stream, read_properties);
    }
    stream.write(key.data(), static_cast<std::streamsize>(key.size_bytes()));
    const std::string bytes{std::move(stream).str()};
    return {bytes.begin(), bytes.end()};
}

void ReadPipelinePayload(
    std::span<const u8> payload,
    Common::UniqueFunction<void, std::istream&, FileEnvironment>& load_compute,
    Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>>& load_graphics) {
    SpanBuffer buffer{payload};
    std::istream file{&buffer};
    file.exceptions(std::ios::failbit);
    const std::streampos end{static_cast<std::streamoff>(payload.size())};
    ReadPipelineEnvironments(file, end, load_compute, load_graphics);
    if (file.tellg() != end) {
        throw std::ios_base::failure("Bytes after the key of a pipeline cache record");
    }
}

ReplacedPipelineCode ReplacePipelineCode(std::span<const u8> payload,
                                         const NativeCodeReplacer& replace) {
    // Where Serialize writes the texture bound buffer, the read range and the code
    constexpr size_t TextureBoundOffset = 6 * sizeof(u64) + sizeof(u32);
    constexpr size_t ReadHighestOffset = TextureBoundOffset + 3 * sizeof(u32);
    constexpr size_t CodeOffset = 6 * sizeof(u64) + 7 * sizeof(u32);
    SpanBuffer buffer{payload};
    std::istream file{&buffer};
    file.exceptions(std::ios::failbit);
    const std::streampos end{static_cast<std::streamoff>(payload.size())};
    u32 num_envs{};
    file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
    if (num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
        throw std::ios_base::failure("Invalid number of shader environments");
    }
    const auto load = [&payload]<typename T>(T& value, size_t offset) {
        std::memcpy(&value, payload.data() + offset, sizeof(T));
    };
    ReplacedPipelineCode result;
    size_t copied{};
    for (u32 index = 0; index < num_envs; ++index) {
        const auto begin = static_cast<size_t>(std::streamoff{file.tellg()});
        FileEnvironment env;
        env.Deserialize(file, end);
        const auto env_end = static_cast<size_t>(std::streamoff{file.tellg()});
        u64 code_size{};
        u32 texture_bound{};
        u32 read_lowest{};
        load(code_size, begin);
        load(texture_bound, begin + TextureBoundOffset);
        load(read_lowest, begin + ReadHighestOffset - sizeof(u32));
        std::vector<u64> code(static_cast<size_t>(code_size / sizeof(u64)));
        std::memcpy(code.data(), payload.data() + begin + CodeOffset, code.size() * sizeof(u64));
        const u64 hash{NativeShaderHash(code)};
        const auto replacement =
            env.IsNativeProgram() ? replace(hash, texture_bound) : std::nullopt;
        if (!replacement || replacement->code.empty()) {
            result.shader_hashes.emplace_back(hash, hash);
            continue;
        }
        const auto store = [&result]<typename T>(const T& value, size_t offset) {
            std::memcpy(result.payload.data() + offset, &value, sizeof(T));
        };
        result.payload.insert(result.payload.end(), payload.begin() + copied,
                              payload.begin() + begin + CodeOffset);
        const size_t header{result.payload.size() - CodeOffset};
        const u64 new_size{replacement->code.size_bytes()};
        store(new_size, header);
        store(replacement->texture_bound, header + TextureBoundOffset);
        store(static_cast<u32>(read_lowest + new_size - sizeof(u64)), header + ReadHighestOffset);
        const auto* const new_code = reinterpret_cast<const u8*>(replacement->code.data());
        result.payload.insert(result.payload.end(), new_code, new_code + new_size);
        copied = begin + CodeOffset + static_cast<size_t>(code_size);
        result.payload.insert(result.payload.end(), payload.begin() + copied,
                              payload.begin() + env_end);
        copied = env_end;
        result.shader_hashes.emplace_back(hash, NativeShaderHash(replacement->code));
    }
    if (copied == 0) {
        result.payload.assign(payload.begin(), payload.end());
    } else {
        result.payload.insert(result.payload.end(), payload.begin() + copied, payload.end());
    }
    return result;
}

bool ReadPipelinesUnchanged(
    const std::filesystem::path& filename, u32 cache_version,
    Common::UniqueFunction<void, std::istream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>> load_graphics) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }
    const std::streampos end{file.tellg()};
    file.seekg(0, std::ios::beg);
    std::array<char, 8> magic_number{};
    u32 version{};
    file.read(magic_number.data(), magic_number.size())
        .read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file || magic_number != MAGIC_NUMBER || version != cache_version) {
        return false;
    }
    file.exceptions(std::ifstream::failbit);
    try {
        while (file.tellg() != end) {
            ReadPipelineEnvironments(file, end, load_compute, load_graphics);
        }
    } catch (const std::bad_alloc& e) {
        LOG_ERROR(Common_Filesystem, "Stopped reading pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), e.what());
    } catch (const std::ios_base::failure& e) {
        LOG_WARNING(Common_Filesystem, "Stopped reading pipeline cache file {} at a damaged record: {}",
                    Common::FS::PathToUTF8String(filename), e.what());
    }
    return true;
}

PipelineRecord ReadPipelineRecord(std::istream& file, u64 begin) {
    const u64 end{static_cast<u64>(std::streamoff{file.tellg()})};
    std::vector<char> bytes(static_cast<size_t>(end - begin));
    file.seekg(static_cast<std::streamoff>(begin));
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return {begin, end, Common::CityHash64(bytes.data(), bytes.size())};
}

void CheckGraphicsRecord(std::span<const u64> unique_hashes, size_t num_envs) {
    const auto shaders{std::ranges::count_if(unique_hashes, [](u64 hash) { return hash != 0; })};
    if (static_cast<size_t>(shaders) != num_envs) {
        throw std::ios_base::failure("Pipeline cache key and shader environments disagree");
    }
}

void CheckComputeRecord(const FileEnvironment& env, u32 shared_memory_size,
                        const std::array<u32, 3>& workgroup_size) {
    if (env.SharedMemorySize() != shared_memory_size || env.WorkgroupSize() != workgroup_size) {
        throw std::ios_base::failure("Pipeline cache key and shader environment disagree");
    }
}

// Whether the file still holds these records (sorted, apart) as the load read them, and each
// copy's original
static bool HoldsRecords(std::ifstream& source, u64 size, std::span<const PipelineRecord> records) {
    std::vector<char> bytes;
    std::vector<char> original;
    const auto read_range{[&source](std::vector<char>& buffer, u64 offset, size_t length) {
        buffer.resize(length);
        source.seekg(static_cast<std::streamoff>(offset));
        return static_cast<bool>(source.read(buffer.data(), static_cast<std::streamsize>(length)));
    }};
    u64 position{MAGIC_NUMBER.size() + sizeof(u32)};
    for (const PipelineRecord& record : records) {
        if (record.begin < position || record.end <= record.begin || record.end > size) {
            return false;
        }
        position = record.end;
        const size_t length{static_cast<size_t>(record.end - record.begin)};
        if (!read_range(bytes, record.begin, length) ||
            Common::CityHash64(bytes.data(), length) != record.hash) {
            return false;
        }
        if (record.copy_of != 0 &&
            (record.copy_of > size - length || !read_range(original, record.copy_of, length) ||
             original != bytes)) {
            return false;
        }
    }
    return true;
}

// Copies the file to `target` without these records (sorted, inside it), unless stopped
static bool CopyWithout(std::ifstream& source, const Common::FS::IOFile& target, u64 size,
                        std::span<const PipelineRecord> records, const std::stop_token& stop) {
    std::vector<char> buffer(1 << 20);
    u64 position{};
    source.seekg(0, std::ios::beg);
    const auto copy_until{[&](u64 until) {
        while (position < until && !stop.stop_requested()) {
            const size_t count{static_cast<size_t>(std::min<u64>(until - position, buffer.size()))};
            if (!source.read(buffer.data(), static_cast<std::streamsize>(count)) ||
                target.WriteSpan(std::span<const char>{buffer.data(), count}) != count) {
                return false;
            }
            position += count;
        }
        return position == until;
    }};
    for (const PipelineRecord& record : records) {
        if (!copy_until(record.begin)) {
            return false;
        }
        source.seekg(static_cast<std::streamoff>(record.end));
        position = record.end;
    }
    return copy_until(size);
}

bool RemovePipelines(std::stop_token stop, const std::filesystem::path& filename, u32 cache_version,
                     u64 loaded_size, std::vector<PipelineRecord> records) {
    if (records.empty() || stop.stop_requested()) {
        return false;
    }
    std::ranges::sort(records, {}, &PipelineRecord::begin);
    records.erase(std::ranges::unique(records).begin(), records.end());
    const std::string name{Common::FS::PathToUTF8String(filename)};
    // Another emulator loading the cache, appending to it or rewriting it: a later boot does it.
    const Common::FS::FileLock lock{PipelineCacheLockPath(filename)};
    if (!lock.Held()) {
        LOG_INFO(Common_Filesystem, "Not rewriting {} now: it is in use or can't be locked", name);
        return false;
    }
    std::ifstream source(filename, std::ios::binary | std::ios::ate);
    if (!source.is_open()) {
        return false;
    }
    const u64 size{static_cast<u64>(std::streamoff{source.tellg()})};
    const auto failed{FailedRewritePath(filename)};
    if (std::ifstream marker{failed, std::ios::binary}) {
        u64 failed_size{};
        if (marker.read(reinterpret_cast<char*>(&failed_size), sizeof(failed_size)) &&
            failed_size == size) {
            LOG_INFO(Common_Filesystem, "Not rewriting {} until it changes: it failed last time",
                     name);
            return false;
        }
    }
    source.seekg(0, std::ios::beg);
    std::array<char, 8> magic_number{};
    u32 version{};
    source.read(magic_number.data(), magic_number.size())
        .read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!source || magic_number != MAGIC_NUMBER || version != cache_version || size < loaded_size ||
        !HoldsRecords(source, size, records)) {
        LOG_INFO(Common_Filesystem, "Not rewriting {}: it changed since it was read", name);
        return false;
    }
    u64 removed{};
    for (const PipelineRecord& record : records) {
        removed += record.end - record.begin;
    }
    // Room for the new file, and never a disk's last space. Where the free space can't be told, a
    // full disk fails the copy instead.
    constexpr u64 SpareSpace{u64{64} << 20};
    std::error_code space_error;
    const auto disk{std::filesystem::space(filename.parent_path(), space_error)};
    if (!space_error && disk.available < size - removed + SpareSpace) {
        LOG_WARNING(Common_Filesystem, "Not rewriting {}: {} bytes of disk space are free", name,
                    disk.available);
        return false;
    }
    const auto temporary{RewritePath(filename)};
    bool written{};
    {
        const Common::FS::IOFile target{temporary, Common::FS::FileAccessMode::Write};
        // The rename can reach the disk before the data: commit the data first.
        written =
            target.IsOpen() && CopyWithout(source, target, size, records, stop) && target.Commit();
    }
    source.close(); // Windows replaces no file that is open
    std::error_code rename_error;
    std::error_code remove_error;
    if (written && !stop.stop_requested()) {
        std::filesystem::rename(temporary, filename, rename_error);
        if (!rename_error) {
            std::filesystem::remove(failed, remove_error);
            LOG_INFO(Common_Filesystem,
                     "Removed {} pipelines that cannot be chosen ({} bytes) from {}",
                     records.size(), removed, name);
            return true;
        }
    }
    std::filesystem::remove(temporary, remove_error);
    if (!stop.stop_requested()) {
        LOG_ERROR(Common_Filesystem, "Failed to rewrite {}: {}", name,
                  rename_error ? rename_error.message() : "the new file can't be written");
        // Not again at every boot: once the file has changed.
        std::ofstream marker{failed, std::ios::binary | std::ios::trunc};
        marker.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    return false;
}

} // namespace VideoCommon
