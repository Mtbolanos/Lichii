// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/unique_function.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/native_lowering.h"

namespace Tegra {
class Memorymanager;
}

namespace VideoCommon::ShaderCacheV2 {
class Cache;
struct ProgramRecord;
} // namespace VideoCommon::ShaderCacheV2

namespace VideoCommon {

class GenericEnvironment : public Shader::Environment {
public:
    explicit GenericEnvironment() = default;
    explicit GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_);

    ~GenericEnvironment() override;

    [[nodiscard]] u32 TextureBoundBuffer() const final;

    [[nodiscard]] u32 LocalMemorySize() const final;

    [[nodiscard]] u32 SharedMemorySize() const final;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const final;

    [[nodiscard]] u64 ReadInstruction(u32 address) final;

    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;

    [[nodiscard]] bool IsNativeProgram() const override;

    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;

    [[nodiscard]] bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                          u32 indirect) override;

    [[nodiscard]] std::vector<std::pair<u32, u32>> NullStoragePointerSlots() const override;

    [[nodiscard]] std::vector<u64> MappedStoragePointerKeys() const override;

    /// Answers the classification of these keys (see MappedStoragePointerKeys) as storage buffer
    /// pointers, whatever the slots hold now; called before the program is translated
    void KeepStoragePointers(std::span<const u64> keys);

    [[nodiscard]] std::optional<u64> Analyze();

    void SetCachedSize(size_t size_bytes);

    [[nodiscard]] size_t CachedSizeWords() const noexcept;

    [[nodiscard]] size_t CachedSizeBytes() const noexcept;

    [[nodiscard]] size_t ReadSizeBytes() const noexcept;

    [[nodiscard]] size_t SourceSizeBytes() const noexcept {
        return native_source_size;
    }

    [[nodiscard]] const std::shared_ptr<const LoweredNativeCode>& LoweredNative() const noexcept {
        return lowered_native;
    }

    /// The v2 cache record of the native program this environment lowered, once
    [[nodiscard]] std::shared_ptr<const ShaderCacheV2::ProgramRecord> TakeProgramRecord() noexcept {
        return std::move(program_record);
    }

    /// False for a program with instructions outside what is cached, or with more values of a
    /// kind than a cache record holds
    [[nodiscard]] bool CanBeSerialized() const noexcept;

    /// The texture descriptor pool was flushed on the GPU thread before this environment's
    /// translation moved to another thread, which may not flush: descriptors are read as they
    /// are in memory. Nothing can make the pool dirty again while the GPU thread waits for it.
    void SetTexturePoolFlushed() noexcept {
        texture_pool_flushed = true;
    }

    [[nodiscard]] u64 CalculateHash() const;

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    /// Writes the environment, with the host's Shader::HostTranslateInfo::EnvironmentReadProperties
    void Serialize(std::ostream& file, u32 read_properties) const;

    bool HasHLEMacroState() const override {
        return has_hle_engine_state;
    }
    bool HasNativeInstructions() const override;

protected:
    std::optional<u64> TryFindSize();
    virtual std::optional<u32> ReadTrapGuardConstant(u32, u32) { return std::nullopt; }

    /// Adopts an already lowered native program, or captures and lowers it when there is none.
    /// With a v2 cache, a captured program it holds a record of isn't lowered again.
    void InitializeNativeCode(std::shared_ptr<const LoweredNativeCode> lowered,
                              u32 source_header_size = 128, bool geometry_stage = false,
                              ShaderCacheV2::Cache* cache_v2 = nullptr);
    std::shared_ptr<const LoweredNativeCode> LowerNativeCode(u32 source_header_size,
                                                          bool geometry_stage,
                                                          ShaderCacheV2::Cache* cache_v2);
    size_t native_source_size{};
    std::shared_ptr<const LoweredNativeCode> lowered_native;
    std::shared_ptr<const ShaderCacheV2::ProgramRecord> program_record;

    Tegra::Texture::TICEntry ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                             bool via_header_index, u32 raw);

    Tegra::MemoryManager* gpu_memory{};
    GPUVAddr program_base{};

    std::vector<u64> code;
    ::Common::unordered_map<u32, Shader::TextureType> texture_types;
    ::Common::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    ::Common::unordered_map<u64, u32> cbuf_values;
    ::Common::unordered_map<u64, u8> storage_pointer_types; ///< IsCbufStorageBufferPointer answers
    ::Common::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;

    u32 local_memory_size{};
    u32 texture_bound{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    u32 read_lowest = (std::numeric_limits<u32>::max)();
    u32 read_highest = 0;

    u32 cached_lowest = (std::numeric_limits<u32>::max)();
    u32 cached_highest = 0;
    u32 initial_offset = 0;

    u32 viewport_transform_state = 1;

    bool has_unbound_instructions = false;
    bool has_hle_engine_state = false;
    bool texture_pool_flushed = false;
};

class GraphicsEnvironment final : public GenericEnvironment {
public:
    explicit GraphicsEnvironment() = default;
    explicit GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                 Tegra::MemoryManager& gpu_memory_,
                                 Tegra::Engines::Maxwell3D::Regs::ShaderType program,
                                 GPUVAddr program_base_, u32 start_address_,
                                 std::shared_ptr<const LoweredNativeCode> lowered_native_ = {},
                                 ShaderCacheV2::Cache* cache_v2 = nullptr);

    ~GraphicsEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;

private:
    Tegra::Engines::Maxwell3D* maxwell3d{};
    size_t stage_index{};
};

class ComputeEnvironment final : public GenericEnvironment {
public:
    explicit ComputeEnvironment() = default;
    explicit ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_,
                                std::shared_ptr<const LoweredNativeCode> lowered_native_ = {},
                                ShaderCacheV2::Cache* cache_v2 = nullptr);

    ~ComputeEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(
        [[maybe_unused]] u32 bank, [[maybe_unused]] u32 offset) override {
        return std::nullopt;
    }

private:
    std::optional<u32> ReadTrapGuardConstant(u32 bank, u32 offset) override;
    Tegra::Engines::KeplerCompute* kepler_compute{};
};

std::optional<u32> ReadNativeGuardConstant(Tegra::Engines::KeplerCompute& compute,
                                         Tegra::MemoryManager& memory, u32 bank, u32 offset);

/// Answers for reads a FileEnvironment has no record of (shader precache: programs from the game
/// files, never run, converted only to fill the driver's cache). Without it a missing read is an
/// error.
struct GuessMissing {
    u32 cbuf_bank0{0x10000000};  ///< Constant buffer 0 (driver constants, descriptor handles)
    u32 cbuf_other{0};           ///< Other constant buffers (game constants)
    Shader::TextureType texture_type{Shader::TextureType::ColorArray2D};
    Shader::TexturePixelFormat pixel_format{static_cast<Shader::TexturePixelFormat>(16)};
    bool storage_pointer_bank0{true};  ///< Bank 0 always held storage pointers in the test title's records
    bool storage_pointer_other{false}; ///< Banks 1-4 never did (95% identical SPIR-V offline)
};

class FileEnvironment final : public Shader::Environment {
public:
    FileEnvironment() = default;

    /// Missing reads get these answers instead of an error; the guess must outlive the
    /// environment.
    void SetGuessMissing(const GuessMissing* guess_) noexcept {
        guess = guess_;
    }
    ~FileEnvironment() override = default;

    FileEnvironment& operator=(FileEnvironment&&) noexcept = default;
    FileEnvironment(FileEnvironment&&) noexcept = default;

    FileEnvironment& operator=(const FileEnvironment&) = delete;
    FileEnvironment(const FileEnvironment&) = delete;

    /// Reads an environment that must end before `end`, the end of the file
    void Deserialize(std::istream& file, std::streampos end);

    [[nodiscard]] size_t MemorySize() const;

    [[nodiscard]] u64 ReadInstruction(u32 address) override;

    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;

    [[nodiscard]] bool IsNativeProgram() const override;

    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;

    [[nodiscard]] bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                          u32 indirect) override;

    [[nodiscard]] std::vector<std::pair<u32, u32>> NullStoragePointerSlots() const override;

    [[nodiscard]] std::vector<u64> MappedStoragePointerKeys() const override;

    [[nodiscard]] u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    [[nodiscard]] Shader::TextureType ReadTextureType(u32 handle) override;

    [[nodiscard]] Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    [[nodiscard]] bool IsTexturePixelFormatInteger(u32 handle) override;

    [[nodiscard]] u32 ReadViewportTransformState() override;

    [[nodiscard]] u32 LocalMemorySize() const override;

    [[nodiscard]] u32 SharedMemorySize() const override;

    [[nodiscard]] u32 TextureBoundBuffer() const override;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const override;

    [[nodiscard]] std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                               u32 offset) override;

    [[nodiscard]] bool HasHLEMacroState() const override {
        return cbuf_replacements.size() != 0;
    }
    bool HasNativeInstructions() const override;

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    /// Whether the translator asked for a constant buffer value, texture or pointer classification
    /// the record lacks, while the record was made where translation reads the same values (with
    /// these `read_properties`, see Shader::HostTranslateInfo::EnvironmentReadProperties) or by a
    /// build that didn't record where: then it was made before the translator read the value. A
    /// record made where they differ can lack it and hold all that translation reads there.
    [[nodiscard]] bool LacksRecordedValue(u32 read_properties) const noexcept {
        return lacks_recorded_value &&
               (!recorded_read_properties || *recorded_read_properties == read_properties);
    }

private:
    std::vector<u64> code;
    ::Common::unordered_map<u32, Shader::TextureType> texture_types;
    ::Common::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    ::Common::unordered_map<u64, u32> cbuf_values;
    ::Common::unordered_map<u64, u8> storage_pointer_types;
    ::Common::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;
    std::array<u32, 3> workgroup_size{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    u32 texture_bound{};
    u32 read_lowest{};
    u32 read_highest{};
    u32 initial_offset{};
    u32 viewport_transform_state = 1;
    const GuessMissing* guess{};
    /// The EnvironmentReadProperties of the host that made the record, when its build wrote them
    std::optional<u32> recorded_read_properties;
    bool lacks_recorded_value{};
};

/// Where a pipeline is in a cache file, from its environment count to the end of its key, and a
/// hash of those bytes: a rewrite removes it only while the file still holds them.
struct PipelineRecord {
    u64 begin;
    u64 end;
    u64 hash;
    /// For a record removed as a copy, where the earlier record with the same bytes begins (else 0)
    u64 copy_of{};

    bool operator==(const PipelineRecord&) const = default;
};

/// Appends a record with the host's Shader::HostTranslateInfo::EnvironmentReadProperties, holding
/// the cache's lock (see PipelineCacheLockPath).
void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version,
                       u32 read_properties);

template <typename Key, typename Envs>
    requires std::is_trivially_copyable_v<Key> && std::has_unique_object_representations_v<Key>
void SerializePipeline(const Key& key, const Envs& envs, const std::filesystem::path& filename,
                       u32 cache_version, u32 read_properties) {
    SerializePipeline(std::span(reinterpret_cast<const char*>(&key), sizeof(key)),
                      std::span(envs.data(), envs.size()), filename, cache_version,
                      read_properties);
}

/// Reads the pipelines of a cache file. The callbacks read the key that ends each record first, so
/// that a record cut short throws before it is built. A damaged record is dropped with the records
/// after it, and the file is cut back to the complete records before it; running out of memory
/// only stops the load. The load holds the cache's lock throughout (waiting for it as
/// PipelineCacheLockWait says), so that a record another process is appending is not taken for a
/// damaged one.
void LoadPipelines(std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics);

/// The record a LoadPipelines callback has read: from `begin`, where its environment count is, to
/// where `file` is now, after its key.
[[nodiscard]] PipelineRecord ReadPipelineRecord(std::istream& file, u64 begin);

/// The bytes SerializePipeline appends for a record, for a shader cache v2 record. Empty when an
/// environment can't be serialized (see GenericEnvironment::CanBeSerialized).
[[nodiscard]] std::vector<u8> SerializePipelinePayload(
    std::span<const char> key, std::span<const GenericEnvironment* const> envs,
    u32 read_properties);

/// Reads one record held in memory (SerializePipelinePayload) as LoadPipelines reads one in a
/// file: the callback reads the key. Throws std::ios_base::failure when the record is damaged or
/// has bytes after its key.
void ReadPipelinePayload(
    std::span<const u8> payload,
    Common::UniqueFunction<void, std::istream&, FileEnvironment>& load_compute,
    Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>>& load_graphics);

/// New code for the native program of an environment in a cache record (ReplacePipelineCode)
struct NativeCodeReplacement {
    std::span<const u64> code;
    u32 texture_bound; ///< The environment's texture bound buffer with this code
};

/// Answers, for the shader hash (NativeShaderHash) of a native environment's code and its texture
/// bound buffer, the code to replace it with, or nothing to keep it
using NativeCodeReplacer =
    std::function<std::optional<NativeCodeReplacement>(u64 shader_hash, u32 texture_bound)>;

struct ReplacedPipelineCode {
    std::vector<u8> payload;
    /// For each environment, the shader hash of its code before and after (equal when kept)
    std::vector<std::pair<u64, u64>> shader_hashes;
};

/// A record (SerializePipelinePayload) with the code of its native environments replaced as
/// `replace` answers, and every other byte kept, the key included. Throws std::ios_base::failure
/// when the record is damaged, as ReadPipelinePayload does.
[[nodiscard]] ReplacedPipelineCode ReplacePipelineCode(std::span<const u8> payload,
                                                       const NativeCodeReplacer& replace);

/// Reads the records of a cache file in order, up to the first damaged one, without its lock and
/// without ever changing or removing the file. False when it isn't a cache of this version.
bool ReadPipelinesUnchanged(
    const std::filesystem::path& filename, u32 cache_version,
    Common::UniqueFunction<void, std::istream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>> load_graphics);

/// Throws std::ios_base::failure, as for a damaged record, unless a graphics record's key has as
/// many nonzero shader hashes as the record has environments: one is built from each.
void CheckGraphicsRecord(std::span<const u64> unique_hashes, size_t num_envs);

/// Throws std::ios_base::failure, as for a damaged record, unless a compute record's key has the
/// shared memory and workgroup sizes of its environment: both are read from the same launch.
void CheckComputeRecord(const FileEnvironment& env, u32 shared_memory_size,
                        const std::array<u32, 3>& workgroup_size);

/// Rewrites a cache file without these records (complete records LoadPipelines read from it when
/// it was `loaded_size` bytes long), keeping every other byte, records appended since included. The
/// new file is written beside the cache and renamed over it, so the cache is never left partly
/// written. All of it happens under the cache's lock, and only while the file still holds every
/// record as the load read it (and a copy's original); else, or when another process holds the
/// lock, disk space is short, it is stopped, or the last rewrite of this file failed and it hasn't
/// changed since, it returns false and leaves the file as it was.
bool RemovePipelines(std::stop_token stop, const std::filesystem::path& filename, u32 cache_version,
                     u64 loaded_size, std::vector<PipelineRecord> records);

} // namespace VideoCommon
