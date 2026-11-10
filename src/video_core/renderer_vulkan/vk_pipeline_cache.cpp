// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <bit>
#include <numeric>
#include "common/cityhash.h"
#include "common/scope_exit.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_cooperative_hmma_test.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_binaries.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_records.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_cache_v2.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GuessMissing;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = VideoCommon::VulkanPipelineCacheVersion;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr auto& VULKAN_CACHE_MAGIC_NUMBER = VideoCommon::VulkanDriverPipelineCacheMagic;

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    // Transform feedback would capture the scaled values, a passthrough geometry shader doesn't
    // scale the outputs it writes itself, and indexed attribute reads in the fragment stage don't
    // undo the scaling, so all three keep native noperspective interpolation.
    const bool geometry_passthrough{key.unique_hashes[4] != 0 &&
                                    programs[4].is_geometry_passthrough};
    if (key.unique_hashes[5] != 0 && !geometry_passthrough && key.state.xfb_enabled == 0 &&
        !programs[5].info.loads_indexed_attributes) {
        const bool has_tess{key.unique_hashes[3] != 0};
        const bool last_pre_raster{
            (stage == Shader::Stage::VertexB && !has_tess && !has_geometry) ||
            (stage == Shader::Stage::TessellationEval && !has_geometry) ||
            (stage == Shader::Stage::Geometry && has_geometry)};
        if (stage == Shader::Stage::Fragment) {
            info.scaled_noperspective_inputs =
                previous_program && (previous_program->stage == Shader::Stage::VertexB ||
                                     previous_program->stage == Shader::Stage::TessellationEval ||
                                     previous_program->stage == Shader::Stage::Geometry);
        }
        if (last_pre_raster) {
            const auto& fragment_info{programs[5].info};
            for (size_t index = 0; index < 32; ++index) {
                if (fragment_info.interpolation[index] == Shader::Interpolation::NoPerspective) {
                    info.noperspective_outputs.set(index);
                }
            }
        }
    }
    return info;
}

/// SPIR-V emission of run-time pipelines: graphics on the GPU thread, graphics on workers and
/// compute on workers (shader cache v2)
std::array<std::atomic<std::chrono::nanoseconds::rep>, 3> runtime_spirv_ns{};

void AddRuntimeSpirvTime(size_t thread, std::chrono::steady_clock::time_point begin) {
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    runtime_spirv_ns[thread].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order::relaxed);
}

vk::ShaderModule EmitShaderModule(const Device& device, const Shader::Profile& profile,
                                 const Shader::RuntimeInfo& runtime_info,
                                 Shader::IR::Program& program, Shader::Backend::Bindings& binding,
                                 u64 shader_hash, size_t stage_index,
                                 std::span<const u32> reference = {}) {
    const auto code =
        EmitPipelineShader(profile, runtime_info, program, binding, shader_hash, reference);
    device.SaveShader(code);
    vk::ShaderModule module{BuildShader(device, code)};

    // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
    // is independent and gated only by gpu_log_shader_dumps.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry",
                                               "fragment"};
        const std::string shader_name =
            fmt::format("shader_{:016x}_{}", shader_hash, stage_names[stage_index]);
        if (should_log) {
            const std::string shader_info = fmt::format(
                "SPIR-V size: {} bytes, hash: {:016x}", code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                         std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const std::string name{fmt::format("Shader {:016x}", shader_hash)};
        module.SetObjectNameEXT(name.c_str());
    }
    return module;
}

vk::ShaderModule EmitComputeModule(const Device& device, const Shader::Profile& profile,
                                   Shader::IR::Program& program, u64 shader_hash,
                                   std::span<const u32> reference = {}) {
    const std::vector<u32> code{EmitComputeShader(profile, program, shader_hash, reference)};
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", shader_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", shader_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    return spv_module;
}

/// A compute program whose SPIR-V the CPU stage of its pipeline's job emits (shader cache v2)
struct DeferredCompute {
    std::unique_ptr<QueuedIrBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    Shader::IR::Program program;
    std::vector<u32> reference_spirv;
};

struct DeferredStages {
    std::unique_ptr<QueuedIrBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    /// Pools of the stages translated on workers (ParallelTranslation)
    std::vector<std::shared_ptr<ShaderPools>> stage_pools;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
};

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    if (unique_hashes != rhs.unique_hashes) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + sizeof(unique_hashes),
                       reinterpret_cast<const char*>(&rhs) + sizeof(unique_hashes),
                       state.Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, render_pass_cache{render_pass_cache_},
      buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue() &&
                                !device_.DriverCachesPipelines()},
      serialization_thread(1, "VkPipelineSerialization", {}, Common::ThreadPlacement::Background) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .max_compute_shared_memory_size = device.GetMaxComputeSharedMemorySize(),
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_spirv_fragment_shuffle = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };
    profile.support_cooperative_hmma =
        device.IsCooperativeHmmaSupported() && TestCooperativeHmma(device);
    profile.support_cooperative_hmma_16x16x16 =
        !profile.support_cooperative_hmma && device.IsCooperativeHmma16x16x16Supported();
    if (profile.support_cooperative_hmma_16x16x16) {
        LOG_INFO(Render_Vulkan, "Guest tensor operations run on 16x16x16 cooperative matrices");
    }

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = device.GetMaxPerStageDescriptorStorageBuffers(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_byte_addressed_r16 = true,
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));

    dynamic_features = {};
    // Keys without the driver, so that a driver update keeps the recorded pipelines
    dynamic_features.driver_id = 0;
    dynamic_features.driver_version = 0;

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();
    dynamic_features.has_static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();

    const size_t cpu_threads = GetTotalPipelineWorkers();
    const size_t driver_threads =
        device.HasBrokenParallelShaderCompiling()
            ? 1
            : std::min(cpu_threads, std::max<size_t>(std::thread::hardware_concurrency() / 2, 2));
    shader_jobs.emplace(VideoCommon::ShaderJobConfig{
        .cpu_threads = cpu_threads,
        .driver_threads = driver_threads,
        .name = "VkShader",
    });
}

namespace {
// Pipelines the last two sessions used are built at boot. Each holds ~0.45 MB of driver memory:
// title 1 built 4,800 from eight sessions, 1.5 GB more than building them on first use (but that
// stalled frames 4x as often), while one session of its title screen uses ~1,500.
// MoltenVK keeps the previous eight sessions: a pipeline built while playing goes through
// SPIR-V -> MSL -> Metal, far more than the 2-3 ms the shorter window assumes, so each one
// built at boot is a stutter avoided.
// Ocarina still built pipelines for 1-2 s of every 10 s early in play with two sessions kept:
// keep sixteen (#161).
constexpr u8 UsedPipelinesMaxAge = 16;
constexpr u8 UsedPipelinesMaxAgeMoltenVK = 8;

u8 MaxUsedPipelineAge(const Device& device) {
    if (device.IsMoltenVK()) {
        return UsedPipelinesMaxAgeMoltenVK;
    }
    return UsedPipelinesMaxAge;
}

// Without a list of the pipelines recent sessions used (the first boot with this cache, or a session
// that ended before saving it), this many cached graphics pipelines at most are built at boot and the
// rest when first needed. A cache that grew over many builds held 48k pipelines, ~21 GB of driver
// memory built before the game started; the list is only saved at exit or after two minutes, so a
// boot that ran out of memory never saved one and the next boot did the same.
constexpr size_t MaxPipelinesWithoutUsedList = 3000;

// Captured environments the shader precache holds for skipped graphics pipelines; the rest wait
// for a later precache.
constexpr size_t PrecacheCaptureLimit = size_t{1} << 30;

// Shader cache v2: the records of the background prebuild the queue may hold, by payload bytes
constexpr u64 PrebuildCaptureLimit = u64{256} << 20;

// Shader cache v2: bytes a job's stage is taken to hold for the scheduler's budget. EGLA-80
// measured ~100 MiB of commit per driver compile in flight (24 at once held ~2.5 GB more than
// idle). A translation's IR pools grow with the program; first guesses until measured.
constexpr u64 DriverCompileBytes = u64{96} << 20;

u64 TranslationBytes(u64 record_bytes) {
    return std::clamp<u64>(record_bytes * 512, u64{8} << 20, u64{512} << 20);
}

u64 IrBytes(std::span<const Shader::IR::Program> programs) {
    u64 bytes{};
    for (const Shader::IR::Program& program : programs) {
        for (const Shader::IR::Block* const block : program.blocks) {
            bytes += sizeof(Shader::IR::Block) + block->Instructions().size() * sizeof(Shader::IR::Inst);
        }
    }
    return bytes;
}

// EDEN_NX2_PRELOAD_ALL_PIPELINES=1 builds every cached pipeline at boot, as before.
bool PreloadAllPipelines() {
    static const bool all = [] {
        const char* value = std::getenv("EDEN_NX2_PRELOAD_ALL_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return all;
}
} // Anonymous namespace

PipelineCache::~PipelineCache() {
    serialization_stop.request_stop();
    StopBackgroundJobs();
    // The pipelines' builds end before the driver pipeline cache they build into goes.
    shader_jobs->Shutdown();
    serialization_thread.WaitForRequests();
    SaveUsedPipelines();
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
    // After the records the serialization thread queued: its writer drains on destruction
    shader_cache_v2 = nullptr;
    cache_v2.reset();
    graphics_cache.clear();
    evicted_pipelines.clear();
    if (pipeline_binaries) {
        pipeline_binaries->LogCounts("at the end of the session");
        pipeline_binaries.reset();
    }
}

void PipelineCache::SaveUsedPipelines() {
    used_pipelines_saved = std::chrono::steady_clock::now();
    // A session too short to say what the game uses (a boot to the menu) keeps the old list.
    constexpr size_t MinimumUsed = 64;
    if (!cache_v2 || used_this_session.size() + used_and_evicted.size() < MinimumUsed) {
        return;
    }
    // This session's record, written by the cache's thread; older sessions keep theirs.
    VideoCommon::ShaderCacheV2::UsedRecord record{.session_time = used_session_time};
    for (const auto& [key, pipeline] : graphics_cache) {
        if (used_this_session.contains(pipeline.get())) {
            record.pipelines.push_back(key.Hash());
        }
    }
    record.pipelines.insert(record.pipelines.end(), used_and_evicted.begin(),
                            used_and_evicted.end());
    std::ranges::sort(record.pipelines);
    record.pipelines.erase(std::ranges::unique(record.pipelines).begin(), record.pipelines.end());
    cache_v2->Writer().Append(VideoCommon::ShaderCacheV2::RecordKind::Used, used_session_key,
                              VideoCommon::ShaderCacheV2::EncodeUsed(record));
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* ComputePipelineCacheEntry::Get(bool native, GPUVAddr address,
                                                const ComputePipelineCacheKey& key) const {
    if (!pipeline && native) {
        throw Shader::RuntimeError(
            "Native compute pipeline unavailable: address={:#x} shader={:#016x} pipeline={:#016x}: {}",
            address, key.unique_hash, key.Hash(),
            failure.empty() ? "No compiler diagnostic was recorded" : failure);
    }
    return pipeline.get();
}

constexpr u32 MaxComputeRebuilds = 4;

namespace {
/// Which pointer slots a native program's arguments fill depends on the area (Ocarina of Time's
/// foliage culling fills different list slots in Hyrule Field and Castle Town), so a translation
/// rebuilt in one area may lack pointers the first cached one binds. Try first the translations
/// that bind every pointer the first one binds, those binding the most first, then the others in
/// cache order.
void OrderComputeTranslations(std::vector<CachedComputeTranslation>& candidates) {
    if (candidates.size() < 2) {
        return;
    }
    std::vector<std::vector<u64>> keys;
    keys.reserve(candidates.size());
    for (const CachedComputeTranslation& candidate : candidates) {
        keys.push_back(candidate.env.MappedStoragePointerKeys());
    }
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const auto keeps_first = [&](size_t index) {
        return std::ranges::includes(keys[index], keys.front());
    };
    std::ranges::stable_sort(order, [&](size_t lhs, size_t rhs) {
        const bool lhs_keeps{keeps_first(lhs)};
        if (lhs_keeps != keeps_first(rhs)) {
            return lhs_keeps;
        }
        return lhs_keeps && keys[lhs].size() > keys[rhs].size();
    });
    std::vector<CachedComputeTranslation> ordered;
    ordered.reserve(candidates.size());
    for (const size_t index : order) {
        ordered.push_back(std::move(candidates[index]));
    }
    candidates = std::move(ordered);
}
} // Anonymous namespace

bool PipelineCache::StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const {
    const auto& qmd{kepler_compute->launch_description};
    for (const auto& [index, offset] : entry.null_pointer_slots) {
        if (index >= qmd.const_buffer_config.size() ||
            ((qmd.const_buffer_enable_mask >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        if (offset + sizeof(u64) > cbuf.size) {
            continue; // The translator reads past the bound size as zero, too
        }
        const GPUVAddr slot{cbuf.Address() + offset};
        if (!gpu_memory->IsFullyMappedRange(slot, sizeof(u64))) {
            continue;
        }
        const GPUVAddr address{gpu_memory->Read<u64>(slot)};
        if (address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
            gpu_memory->IsFullyMappedRange(address, 1)) {
            LOG_INFO(Render_Vulkan,
                     "Rebuilding compute program {:#x}: c{}[{:#x}] was null at translation and "
                     "now points to {:#x}",
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, index, offset, address);
            return true;
        }
    }
    return false;
}

ComputePipeline* PipelineCache::CurrentComputePipeline(u64& native_key) {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    native_key = shader->lowered_native ? shader->lowered_native->native_program_key : 0;
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& entry{pair->second};
    if (is_new) {
        entry = CreateComputePipeline(key, shader);
    } else if (!entry.null_pointer_slots.empty() && entry.rebuilds < MaxComputeRebuilds &&
               StaleNullPointerSlot(entry)) {
        // The translation dropped accesses through a pointer that was null back then. Which slots
        // the game fills depends on the area, so the new one keeps binding every pointer this one
        // binds, null now or not: a null pointer binds nothing.
        auto rebuilt{CreateComputePipeline(key, shader, entry.mapped_pointers)};
        if (rebuilt.pipeline) {
            if (entry.pipeline) {
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(entry.pipeline));
            }
            rebuilt.rebuilds = entry.rebuilds + 1;
            rebuilt.used = entry.used;
            entry = std::move(rebuilt);
        } else {
            // Keep the working translation when the new one cannot be built (for example a
            // narrow access that now stays global across two buffers), and stop retrying.
            LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed: {}",
                        key.unique_hash, rebuilt.failure);
            entry.rebuilds = MaxComputeRebuilds;
        }
    }
    if (VideoCommon::StutterTrace::Enabled() && !std::exchange(entry.used, true)) {
        VideoCommon::StutterTrace::CountLookup(
            is_new ? VideoCommon::StutterTrace::Lookup::ComputeNew
                   : VideoCommon::StutterTrace::Lookup::ComputePrebuilt);
    }
    return entry.Get(kepler_compute->IsNativeMode(),
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, key);
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        StopBackgroundJobs();
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
        shader_cache_v2 = nullptr;
        cache_v2.reset();
        pipeline_binaries.reset();
    }

    namespace StutterTrace = VideoCommon::StutterTrace;
    const auto load_begin = std::chrono::steady_clock::now();
    StutterTrace::BootMarker("shader_load_begin");
    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    namespace CacheV2 = VideoCommon::ShaderCacheV2;
    pipeline_cache_filename = base_dir / CacheV2::FileName;
    const auto migration =
        MigrateLegacyPipelineCache(base_dir, pipeline_cache_filename, MaxUsedPipelineAge(device));
    if (migration.version) {
        LOG_INFO(Render_Vulkan,
                 "Shader cache v2 started from the version {} cache: {} graphics and {} "
                 "compute pipelines ({} records collapsed), {} used-list entries",
                 *migration.version, migration.graphics, migration.compute, migration.collapsed,
                 migration.used);
    } else if (migration.failed) {
        LOG_WARNING(Render_Vulkan, "Couldn't start shader cache v2 from the older cache");
    }
    cache_v2 = std::make_unique<CacheV2::Cache>(pipeline_cache_filename);
    shader_cache_v2 = cache_v2.get();
    pipeline_binaries = PipelineBinaries::Open(device, base_dir);
    // Without the lock: another emulator's appends are complete records or a cut-off tail.
    // The payload hashes are checked on the shader job workers.
    auto log_v2 = std::make_shared<CacheV2::LogFile>(CacheV2::ReadLogFile(
        pipeline_cache_filename, [this](size_t count, const std::function<void(size_t)>& task) {
            VideoCommon::ParallelFor(*shader_jobs, count, task);
        }));
    cache_v2->LoadPrograms(log_v2->parsed.live);
    const CacheV2::LogStats& stats = log_v2->parsed.stats;
    LOG_INFO(Render_Vulkan,
             "Shader cache v2: {} bytes, {} live records, {} superseded, {} tombstones, {} "
             "damaged headers, {} damaged records, {} bytes skipped, {} bytes cut off",
             stats.file_size, stats.live, stats.superseded, stats.tombstones, stats.bad_headers,
             stats.bad_payloads, stats.skipped_bytes, stats.torn_tail_bytes);
    std::vector<u64> expired_sessions;
    used_pipelines =
        UsedPipelinesFromLog(log_v2->parsed.live, MaxUsedPipelineAge(device), expired_sessions);
    for (const u64 session : expired_sessions) {
        cache_v2->Writer().Append(CacheV2::RecordKind::Tombstone, session,
                                  CacheV2::TombstonePayload(CacheV2::RecordKind::Used));
    }
    used_session_time = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count());
    used_session_key = used_session_time ^ std::random_device{}();
    used_pipelines_saved = std::chrono::steady_clock::now();

    // With the shader precache this boot, the cached graphics pipelines skipped below are built
    // on the loading screen too, unless an earlier precache built them for this driver.
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const bool precache = ShaderWarm::PrecacheBeforeStart(
        Settings::values.precache_shaders.GetValue(), ShaderWarm::TakeWarmBeforeStart());
    std::unordered_set<u64> precached_pipelines;
    if (precache) {
        precached_pipelines = ShaderWarm::ReadPrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION});
    }
    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_pipelines", CACHE_VERSION);
        if (VideoCommon::CarryOverPipelineCache(
                base_dir, "vulkan_pipelines", VULKAN_CACHE_MAGIC_NUMBER,
                VideoCommon::CompatibleVulkanPipelineCacheVersions, CACHE_VERSION)
                .failed) {
            LOG_WARNING(Render_Vulkan,
                        "Couldn't copy the older driver pipeline cache; this one starts empty");
        }
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }
    LoadPipelinesV2(title_id, std::move(log_v2), stop_loading, callback, load_begin, precache,
                    std::move(precached_pipelines));
}

bool PipelineCache::KeyFitsDevice(const GraphicsPipelineCacheKey& key) const {
    if ((key.state.extended_dynamic_state != 0) != dynamic_features.has_extended_dynamic_state ||
        (key.state.extended_dynamic_state_2 != 0) !=
            dynamic_features.has_extended_dynamic_state_2 ||
        (key.state.extended_dynamic_state_2_logic_op != 0) !=
            dynamic_features.has_extended_dynamic_state_2_logic_op ||
        (key.state.extended_dynamic_state_3_blend != 0) !=
            dynamic_features.has_extended_dynamic_state_3_blend ||
        (key.state.extended_dynamic_state_3_enables != 0) !=
            dynamic_features.has_extended_dynamic_state_3_enables ||
        (key.state.color_write_enable_dynamic != 0) != dynamic_features.has_color_write_enable ||
        (key.state.dynamic_vertex_input != 0) != dynamic_features.has_dynamic_vertex_input) {
        return false;
    }
    const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
    if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
        return false;
    }
    const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
    return !key_uses_transform_feedback || !key_requests_provoking_last ||
           dynamic_features.has_provoking_vertex_tf_preserve;
}

PipelineBuildQueue PipelineCache::BuildQueue(bool build_in_parallel, u64 cpu_bytes) {
    if (!build_in_parallel) {
        return {.owner_builds = true};
    }
    return {.jobs = &*shader_jobs, .cpu_bytes = cpu_bytes, .driver_bytes = DriverCompileBytes};
}

std::shared_ptr<ShaderPools> PipelineCache::TakeSparePools() {
    {
        std::scoped_lock lock{spare_pools_mutex};
        if (!spare_pools.empty()) {
            auto pools = std::move(spare_pools.back());
            spare_pools.pop_back();
            return pools;
        }
    }
    return std::make_shared<ShaderPools>();
}

void PipelineCache::ReturnPools(std::shared_ptr<ShaderPools> pools) {
    pools->ReleaseContents();
    std::scoped_lock lock{spare_pools_mutex};
    if (spare_pools.size() < 4) {
        spare_pools.push_back(std::move(pools));
    }
}

void PipelineCache::StopBackgroundJobs() {
    scheduled_prebuild.reset();
}

std::unique_ptr<GraphicsPipeline> PipelineCache::TranslateCachedGraphics(
    std::span<const u8> payload, const GraphicsPipelineCacheKey& key,
    PipelineStatistics* statistics, bool& incomplete) {
    std::vector<FileEnvironment> envs;
    Common::UniqueFunction<void, std::istream&, FileEnvironment> read_compute{
        [](std::istream&, FileEnvironment) {
            throw std::ios_base::failure("Compute pipeline in another kind of record");
        }};
    Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>> read_graphics{
        [&](std::istream& file, std::vector<FileEnvironment> read) {
            GraphicsPipelineCacheKey read_key;
            file.read(reinterpret_cast<char*>(&read_key), sizeof(read_key));
            VideoCommon::CheckGraphicsRecord(read_key.unique_hashes, read.size());
            envs = std::move(read);
        }};
    VideoCommon::ReadPipelinePayload(payload, read_compute, read_graphics);
    ShaderPools pools;
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
    for (auto& env : envs) {
        env_ptrs.push_back(&env);
    }
    auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), statistics, false)};
    incomplete = !pipeline && std::ranges::any_of(envs, [this](const FileEnvironment& env) {
        return env.LacksRecordedValue(host_info.EnvironmentReadProperties());
    });
    return pipeline;
}

void PipelineCache::LoadPipelinesV2(u64 title_id,
                                    std::shared_ptr<const VideoCommon::ShaderCacheV2::LogFile> log,
                                    std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback,
                                    std::chrono::steady_clock::time_point load_begin,
                                    bool precache, std::unordered_set<u64> precached_pipelines) {
    namespace CacheV2 = VideoCommon::ShaderCacheV2;
    namespace StutterTrace = VideoCommon::StutterTrace;
    using CacheV2::RecordKind;
    using VideoCommon::ShaderJobPriority;
    auto& jobs = *shader_jobs;
    const u32 read_properties = host_info.EnvironmentReadProperties();

    // The boot builds (Soon), which Loading Shaders waits for, and the background prebuild. A
    // stopped load cancels both; the workers stay for the game.
    const VideoCommon::ShaderJobBatch boot = jobs.CreateBatch();
    scheduled_prebuild.emplace(jobs, PrebuildCaptureLimit);
    const std::stop_callback cancel_load{
        stop_loading, [&jobs, boot, prebuild = scheduled_prebuild->Batch()] {
            jobs.Cancel(boot);
            jobs.Cancel(prebuild);
        }};

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        /// Records the load never chooses, as in the old cache; begin is the index in records
        std::vector<VideoCommon::PipelineRecord> incomplete;
        std::vector<std::pair<u64, VideoCommon::PipelineRecord>> incomplete_graphics;
        std::vector<VideoCommon::PipelineRecord> copies;
    } state;
    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    // Under state.mutex
    const auto built_one = [&state, &callback] {
        ++state.built;
        if (state.has_loaded) {
            callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
        }
    };
    // The graphics and compute records read, by the index a PipelineRecord's begin stands for
    std::vector<const CacheV2::LogRecord*> records;
    const auto pipeline_record = [&records](size_t index) {
        return VideoCommon::PipelineRecord{.begin = index, .end = index + 1, .hash = records[index]->key};
    };
    // Intact, but not a record this build reads: it goes.
    const auto unreadable = [this](const CacheV2::LogRecord& record, const char* reason) {
        LOG_ERROR(Common_Filesystem, "Shader cache v2 record {:016x}: {}", record.key, reason);
        cache_v2->Writer().Append(RecordKind::Tombstone, record.key,
                                  CacheV2::TombstonePayload(record.kind));
    };

    // One pass over the records without reading them, which the jobs do in parallel: all
    // translations of a compute key go to one job, graphics records by their key's hash.
    ::Common::unordered_map<ComputePipelineCacheKey, std::vector<size_t>> compute_keys;
    std::vector<std::pair<size_t, GraphicsPipelineCacheKey>> boot_graphics;
    struct PrebuildRecord {
        GraphicsPipelineCacheKey key;
        std::vector<u8> payload;
    };
    std::vector<PrebuildRecord> prebuild;
    u64 prebuild_bytes = 0;
    std::vector<PrecacheJob> precache_graphics;
    u64 precache_capture_bytes = 0;
    std::vector<CacheV2::ProgramRecord> stale_programs;
    for (const CacheV2::LogRecord& record : log->parsed.live) {
        if (record.kind != RecordKind::Program) {
            continue;
        }
        const auto stamp = CacheV2::ProgramStamp(record.payload);
        if (!stamp) {
            continue;
        }
        cache_v2->AddProgram(record.key, *stamp);
        if (*stamp != CacheV2::LoweringStamp()) {
            if (auto program = CacheV2::DecodeProgram(record.payload)) {
                stale_programs.push_back(std::move(*program));
            }
        }
    }
    // Programs another build recorded are lowered again before any of their pipelines is built:
    // the records of those whose code or texture bound buffer changed move to this build's code
    // (a pipeline key holds the code's hash alone, so one built from the old record would be hit
    // with the new texture bound buffer). Their own records follow last, so that a load cut short
    // does this again at the next boot.
    std::vector<CacheV2::LogAppend> relinked_appends;
    std::vector<CacheV2::LogRecord> relinked_records;
    std::unordered_set<u64> replaced_records;
    if (!stale_programs.empty() && !stop_loading.stop_requested()) {
        const size_t stale_count = stale_programs.size();
        CacheV2::ForEach for_each = [&jobs](size_t count, const std::function<void(size_t)>& task) {
            VideoCommon::ParallelFor(jobs, count, task);
        };
        RevalidatedPrograms revalidated =
            RevalidateProgramRecords(std::move(stale_programs), for_each);
        relinked_appends =
            RelinkPipelineRecords(log->parsed.live, revalidated.relinked,
                                  [&stop_loading] { return stop_loading.stop_requested(); });
        if (!stop_loading.stop_requested()) {
            for (const CacheV2::LogAppend& append : relinked_appends) {
                cache_v2->Writer().Append(append.kind, append.key, append.payload);
                replaced_records.insert(append.key);
                if (append.kind == RecordKind::Tombstone) {
                    continue;
                }
                relinked_records.push_back({.kind = append.kind,
                                            .flags = 0,
                                            .key = append.key,
                                            .offset = 0,
                                            .payload = append.payload});
            }
            for (const CacheV2::LogAppend& append :
                 RelinkUsedPipelines(log->parsed.live, relinked_appends, MaxUsedPipelineAge(device),
                                     used_pipelines)) {
                cache_v2->Writer().Append(append.kind, append.key, append.payload);
            }
            for (const CacheV2::ProgramRecord& program : revalidated.unchanged) {
                cache_v2->RecordProgram(program);
            }
            for (const auto& [hash, relinked] : revalidated.relinked) {
                cache_v2->RecordProgram(relinked.program);
            }
            LOG_INFO(Render_Vulkan,
                     "Shader cache v2: {} programs of another build lowered again, {} to other "
                     "code or texture bound buffer, {} pipeline records moved to it",
                     stale_count, revalidated.relinked.size(), relinked_records.size());
        }
    }
    std::vector<const CacheV2::LogRecord*> plan;
    plan.reserve(log->parsed.live.size() + relinked_records.size());
    for (const CacheV2::LogRecord& record : log->parsed.live) {
        if (record.kind == RecordKind::Program ||
            ((record.kind == RecordKind::Graphics || record.kind == RecordKind::Compute) &&
             replaced_records.contains(record.key))) {
            continue;
        }
        plan.push_back(&record);
    }
    for (const CacheV2::LogRecord& record : relinked_records) {
        plan.push_back(&record);
    }

    size_t graphics_records = 0;
    size_t compute_records = 0;
    size_t skipped_pipelines = 0;
    size_t queued_graphics = 0;
    for (const CacheV2::LogRecord* const planned : plan) {
        const CacheV2::LogRecord& record = *planned;
        if (stop_loading.stop_requested()) {
            break;
        }
        if (record.kind == RecordKind::Compute) {
            ++compute_records;
            const auto key = PipelineRecordKey<ComputePipelineCacheKey>(record.payload);
            if (!key) {
                unreadable(record, "Compute pipeline record without a key");
                continue;
            }
            compute_keys[*key].push_back(records.size());
            records.push_back(&record);
            continue;
        }
        if (record.kind != RecordKind::Graphics) {
            continue;
        }
        ++graphics_records;
        const auto key = PipelineRecordKey<GraphicsPipelineCacheKey>(record.payload);
        if (!key) {
            unreadable(record, "Graphics pipeline record without a key");
            continue;
        }
        const size_t index = records.size();
        records.push_back(&record);
        if (!KeyFitsDevice(*key)) {
            continue;
        }
        const u64 hash = key->Hash();
        if (!disk_pipelines.insert(hash).second) {
            continue;
        }
        if (PreloadAllPipelines() || (used_pipelines.empty()
                                          ? queued_graphics < MaxPipelinesWithoutUsedList
                                          : used_pipelines.contains(hash))) {
            ++queued_graphics;
            boot_graphics.emplace_back(index, *key);
            continue;
        }
        ++skipped_pipelines;
        const u64 bytes = record.payload.size();
        std::vector<u8> payload(record.payload.begin(), record.payload.end());
        if (precache && !precached_pipelines.contains(hash) &&
            bytes <= PrecacheCaptureLimit - precache_capture_bytes) {
            precache_capture_bytes += bytes;
            precache_graphics.push_back({
                .graphics_hash = hash,
                .translate = [this, key = *key, payload = std::move(payload)]()
                    -> Common::UniqueFunction<bool> {
                    bool incomplete{};
                    auto pipeline = TranslateCachedGraphics(payload, key, nullptr, incomplete);
                    if (!pipeline) {
                        return {};
                    }
                    return [pipeline = std::move(pipeline)] {
                        pipeline->Build();
                        return !pipeline->Failed();
                    };
                },
                .cpu_bytes = TranslationBytes(bytes),
            });
        } else if (bytes <= PrebuildCaptureLimit - prebuild_bytes) {
            prebuild_bytes += bytes;
            prebuild.push_back({*key, std::move(payload)});
        }
    }

    for (auto& [key, indices] : compute_keys) {
        ++state.total;
        u64 bytes = 0;
        for (const size_t index : indices) {
            bytes = std::max<u64>(bytes, records[index]->payload.size());
        }
        jobs.Submit(
            {
                .priority = ShaderJobPriority::Soon,
                .cpu_bytes = TranslationBytes(bytes),
                .driver_bytes = DriverCompileBytes,
                .cpu = [this, &state, &built_one, &records, &pipeline_record, &unreadable, key,
                        indices = std::move(indices),
                        read_properties](VideoCommon::ShaderJobContext& context)
                    -> VideoCommon::ShaderDriverStage {
                    std::vector<CachedComputeTranslation> candidates;
                    for (const size_t index : indices) {
                        const CacheV2::LogRecord& record = *records[index];
                        Common::UniqueFunction<void, std::istream&, FileEnvironment> read_compute{
                            [&](std::istream& file, FileEnvironment env) {
                                ComputePipelineCacheKey read_key;
                                file.read(reinterpret_cast<char*>(&read_key), sizeof(read_key));
                                VideoCommon::CheckComputeRecord(env, read_key.shared_memory_size,
                                                                read_key.workgroup_size);
                                candidates.push_back({std::move(env), pipeline_record(index)});
                            }};
                        Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>>
                            read_graphics{[](std::istream&, std::vector<FileEnvironment>) {
                                throw std::ios_base::failure(
                                    "Graphics pipeline in another kind of record");
                            }};
                        try {
                            VideoCommon::ReadPipelinePayload(record.payload, read_compute,
                                                             read_graphics);
                        } catch (const std::ios_base::failure& e) {
                            unreadable(record, e.what());
                        }
                    }
                    OrderComputeTranslations(candidates);
                    ShaderPools pools;
                    ComputePipelineCacheEntry entry;
                    size_t chosen{};
                    for (; chosen < candidates.size() && !context.StopRequested(); ++chosen) {
                        pools.ReleaseContents();
                        entry = CreateComputePipeline(pools, key, candidates[chosen].env,
                                                      state.statistics.get(), false);
                        if (entry.pipeline) {
                            break;
                        }
                    }
                    std::scoped_lock lock{state.mutex};
                    AddUnchosenComputeTranslations(candidates, chosen, read_properties,
                                                   state.incomplete, state.copies);
                    if (!entry.pipeline) {
                        built_one();
                        return {};
                    }
                    return [this, &state, &built_one, key, entry = std::move(entry)]() mutable {
                        entry.pipeline->Build();
                        std::scoped_lock driver_lock{state.mutex};
                        compute_cache.emplace(key, std::move(entry));
                        built_one();
                    };
                },
            },
            &boot);
    }
    compute_keys.clear();
    for (const auto& [index, key] : boot_graphics) {
        ++state.total;
        const CacheV2::LogRecord& record = *records[index];
        jobs.Submit(
            {
                .priority = ShaderJobPriority::Soon,
                .cpu_bytes = TranslationBytes(record.payload.size()),
                .driver_bytes = DriverCompileBytes,
                .cpu = [this, &state, &built_one, &pipeline_record, &unreadable, &record, index,
                        key](VideoCommon::ShaderJobContext& context)
                    -> VideoCommon::ShaderDriverStage {
                    std::unique_ptr<GraphicsPipeline> pipeline;
                    bool incomplete{};
                    if (!context.StopRequested()) {
                        try {
                            pipeline = TranslateCachedGraphics(record.payload, key,
                                                               state.statistics.get(), incomplete);
                        } catch (const std::ios_base::failure& e) {
                            unreadable(record, e.what());
                        }
                    }
                    if (!pipeline) {
                        std::scoped_lock lock{state.mutex};
                        if (incomplete) {
                            state.incomplete_graphics.emplace_back(key.Hash(),
                                                                   pipeline_record(index));
                        }
                        built_one();
                        return {};
                    }
                    return [this, &state, &built_one, key,
                            pipeline = std::move(pipeline)]() mutable {
                        pipeline->Build();
                        std::scoped_lock lock{state.mutex};
                        graphics_cache.emplace(key, std::move(pipeline));
                        built_one();
                    };
                },
            },
            &boot);
    }
    boot_graphics.clear();

    // With the precache, the prebuild starts after it.
    size_t deferred_pipelines = 0;
    const auto start_prebuild = [&] {
        for (PrebuildRecord& record : prebuild) {
            const u64 bytes = record.payload.size();
            const bool added = scheduled_prebuild->Add(
                record.key, bytes, TranslationBytes(bytes), DriverCompileBytes,
                [this, key = record.key, payload = std::move(record.payload)] {
                    bool incomplete{};
                    try {
                        return TranslateCachedGraphics(payload, key, nullptr, incomplete);
                    } catch (const std::ios_base::failure&) {
                        return std::unique_ptr<GraphicsPipeline>{};
                    }
                });
            deferred_pipelines += added ? 1 : 0;
        }
        prebuild.clear();
    };
    if (!precache) {
        start_prebuild();
    }

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (skipped_pipelines != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipelines skipped at boot, {} {} for background prebuild",
                 skipped_pipelines, precache ? prebuild.size() : deferred_pipelines,
                 precache ? "held" : "admitted");
    }
    {
        std::scoped_lock lock{state.mutex};
        callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
        state.has_loaded = true;
    }
    boot.Wait();

    std::vector<DuplicateGraphicsRecord> no_duplicates;
    AddIncompleteGraphics(state.incomplete_graphics, no_duplicates, disk_pipelines,
                          state.incomplete);
    // The records the load never chooses go by tombstones, and the log is compacted once a
    // quarter of it is records no load reads.
    if (!stop_loading.stop_requested()) {
        for (const auto* removed : {&state.incomplete, &state.copies}) {
            for (const VideoCommon::PipelineRecord& record : *removed) {
                const CacheV2::LogRecord& read = *records[record.begin];
                cache_v2->Writer().Append(RecordKind::Tombstone, read.key,
                                          CacheV2::TombstonePayload(read.kind));
            }
        }
        const CacheV2::LogStats& stats = log->parsed.stats;
        if (stats.dead_bytes != 0 && stats.dead_bytes >= stats.file_size / 4) {
            cache_v2->Writer().Compact();
        }
    }
    records.clear();
    log.reset();

    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        last_flush = std::chrono::steady_clock::now();
    }
    if (state.statistics) {
        state.statistics->Report();
    }
    if (StutterTrace::Enabled()) {
        const auto bytes = [](const std::filesystem::path& path) {
            std::error_code error;
            const auto size = std::filesystem::file_size(path, error);
            return error ? std::uintmax_t{0} : size;
        };
        StutterTrace::SetCacheFile(Common::FS::PathToUTF8String(pipeline_cache_filename));
        StutterTrace::BootMarker(
            "shader_load_end",
            fmt::format("seconds={:.3f} graphics_records={} compute_records={} queued={} skipped={} "
                        "deferred={} cache_bytes={} driver_cache_bytes={}",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin)
                            .count(),
                        graphics_records, compute_records, state.total, skipped_pipelines,
                        precache ? prebuild.size() : deferred_pipelines,
                        bytes(pipeline_cache_filename), bytes(vulkan_pipeline_cache_filename)));
    }
    if (precache && !stop_loading.stop_requested()) {
        PrecacheShaders(title_id, std::move(precache_graphics), std::move(precached_pipelines),
                        stop_loading, callback);
    }
    if (stop_loading.stop_requested()) {
        scheduled_prebuild.reset();
    } else if (precache) {
        start_prebuild();
        LOG_INFO(Render_Vulkan, "{} cached graphics pipelines admitted for background prebuild",
                 deferred_pipelines);
    }
}

void PipelineCache::PrecacheShaders(u64 title_id, std::vector<PrecacheJob> jobs,
                                    std::unordered_set<u64> precached, std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const size_t graphics_jobs = jobs.size();
    const std::filesystem::path warm_file = ShaderWarm::ComputeWarmPath(title_id);
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    const auto stamp = [&](u64 entries) {
        return ShaderWarm::CurrentWarmStamp(warm_file, entries, device.GetDriverVersion(),
                                            device.GetDeviceID(), CACHE_VERSION);
    };
    // Compute programs of the precache file, unless a matching marker says they were built.
    size_t file_entries = 0;
    bool compute_pending = false;
    std::error_code ec;
    const auto current = stamp(0);
    const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
    if (!std::filesystem::exists(warm_file, ec)) {
        LOG_INFO(Render_Vulkan, "Shader precache: no precache file at {}", path);
    } else if (current && marker && marker->Matches(*current)) {
        LOG_INFO(Render_Vulkan, "Shader precache: compute programs already done for this driver");
    } else {
        // Read the whole file here (its own reader: the disk cache loader deletes files it rejects).
        try {
            std::ifstream file(warm_file, std::ios::binary | std::ios::ate);
            file.exceptions(std::ifstream::failbit);
            const auto end{file.tellg()};
            file.seekg(0, std::ios::beg);
            std::array<char, 8> magic;
            u32 version{};
            file.read(magic.data(), magic.size())
                .read(reinterpret_cast<char*>(&version), sizeof(version));
            constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
            if (magic != expected_magic || version != CACHE_VERSION) {
                LOG_WARNING(Render_Vulkan, "Shader precache: {} is not a version {} file", path,
                            CACHE_VERSION);
                throw std::runtime_error("wrong version");
            }
            while (file.tellg() != end) {
                u32 num_envs{};
                file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                FileEnvironment env;
                env.Deserialize(file, end);
                if (num_envs != 1 || env.ShaderStage() != Shader::Stage::Compute) {
                    LOG_WARNING(Render_Vulkan, "Shader precache: unexpected entry in {}", path);
                    throw std::runtime_error("unexpected entry");
                }
                ComputePipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                ++file_entries;
                // Programs already in the disk cache were built above; leave them out.
                if (compute_cache.contains(key)) {
                    continue;
                }
                const u64 cpu_bytes = TranslationBytes(env.MemorySize());
                jobs.push_back({
                    .translate = [this, key,
                                  env = std::move(env)]() mutable -> Common::UniqueFunction<bool> {
                        static const GuessMissing guess{};
                        env.SetGuessMissing(&guess);
                        ShaderPools pools;
                        auto pipeline =
                            CreateComputePipeline(pools, key, env, nullptr, false).pipeline;
                        if (!pipeline) {
                            return {};
                        }
                        return [pipeline = std::move(pipeline)] {
                            pipeline->Build();
                            return true;
                        };
                    },
                    .cpu_bytes = cpu_bytes,
                });
            }
            compute_pending = true;
        } catch (const std::exception& e) {
            LOG_WARNING(Render_Vulkan, "Shader precache: reading {} failed: {}", path, e.what());
            jobs.resize(graphics_jobs);
            file_entries = 0;
        }
    }
    if (jobs.empty() && !compute_pending) {
        LOG_INFO(Render_Vulkan, "Shader precache: nothing to precache");
        return;
    }

    const auto begin = std::chrono::steady_clock::now();
    const size_t num_threads = GetTotalPipelineWorkers();
    std::vector<u8> built(jobs.size());
    std::atomic<size_t> done{};
    bool skipped = false;
    LOG_INFO(Render_Vulkan,
             "Shader precache: {} cached graphics pipelines and {} of {} compute programs to "
             "build on {} threads",
             graphics_jobs, jobs.size() - graphics_jobs, file_entries, num_threads);
    VideoCommon::StutterTrace::BootMarker("precache_begin", fmt::format("jobs={}", jobs.size()));
    if (!jobs.empty()) {
        // Background jobs of the shader workers, each built and destroyed: only the driver's
        // cache keeps it.
        const VideoCommon::ShaderJobBatch batch = shader_jobs->CreateBatch();
        for (size_t index = 0; index < jobs.size(); ++index) {
            shader_jobs->Submit(
                {
                    .priority = VideoCommon::ShaderJobPriority::Background,
                    .cpu_bytes = jobs[index].cpu_bytes,
                    .driver_bytes = DriverCompileBytes,
                    .cpu = [&built, &done, index, translate = std::move(jobs[index].translate)](
                               VideoCommon::ShaderJobContext& context) mutable
                    -> VideoCommon::ShaderDriverStage {
                        Common::UniqueFunction<bool> build;
                        if (!context.StopRequested()) {
                            try {
                                build = translate();
                            } catch (const std::exception& e) {
                                LOG_DEBUG(Render_Vulkan, "Shader precache: {}", e.what());
                            }
                        }
                        translate = {};
                        if (!build) {
                            done.fetch_add(1);
                            return {};
                        }
                        return [&built, &done, index, build = std::move(build)]() mutable {
                            try {
                                built[index] = build();
                            } catch (const std::exception& e) {
                                LOG_DEBUG(Render_Vulkan, "Shader precache: {}", e.what());
                            }
                            done.fetch_add(1);
                        };
                    },
                },
                &batch);
        }
        while (!batch.Idle()) {
            if (stop_loading.stop_requested() || ShaderWarm::TakeSkipWarm()) {
                // The builds not started are dropped; the running ones end first.
                shader_jobs->Cancel(batch);
                skipped = true;
                break;
            }
            if (callback) {
                callback(VideoCore::LoadCallbackStage::Warm, done, jobs.size());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        batch.Wait();
    }
    const size_t total_built = static_cast<size_t>(std::count(built.begin(), built.end(), u8{1}));
    if (compute_pending && !skipped) {
        if (const auto stamp_done = stamp(file_entries)) {
            ShaderWarm::WriteWarmMarker(warm_file, *stamp_done);
        }
    }
    const size_t precached_before = precached.size();
    for (size_t index = 0; index < graphics_jobs; ++index) {
        if (built[index]) {
            precached.insert(jobs[index].graphics_hash);
        }
    }
    if (precached.size() != precached_before) {
        ShaderWarm::WritePrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION}, precached);
    }
    if (!stop_loading.stop_requested() && total_built != 0) {
        // Without this, the builds reach the file only at the next boot, game change or clean
        // close, and are lost if the app ends otherwise.
        QueueVulkanPipelineCacheSave();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    LOG_INFO(Render_Vulkan, "Shader precache: {}, {} built, {} failed, {} not reached, {:.1f} s",
             stop_loading.stop_requested() ? "stopped" : (skipped ? "skipped" : "finished"),
             total_built, done - total_built, jobs.size() - std::min(jobs.size(), done.load()),
             seconds);
    VideoCommon::StutterTrace::BootMarker(
        "precache_end", fmt::format("built={} seconds={:.3f}", total_built, seconds));
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        flush_in_flight.store(false, std::memory_order_release);
    });
}

void PipelineCache::QueueVulkanPipelineCacheSave() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    // Not skipped when a timed flush is in flight: that one may have read the cache before the
    // stage's last builds. The serialization thread runs the two one after the other.
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "Driver pipeline cache saved after the shader precache, {} bytes",
                 size);
    });
}

namespace {
// EDEN_NX2_PIPELINE_DIFF=1: for a new graphics pipeline whose shaders an earlier one uses too,
// log which parts of the fixed state differ from that one (finds state that splits the same
// draws into many pipelines, each translated on the GPU thread).
void ReportNewPipelineKey(const GraphicsPipelineCacheKey& key) {
    static const bool enabled = std::getenv("EDEN_NX2_PIPELINE_DIFF") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static std::unordered_map<u64, GraphicsPipelineCacheKey> last_by_shaders;
    static u32 logged{};
    const u64 shaders = Common::CityHash64(reinterpret_cast<const char*>(key.unique_hashes.data()),
                                           sizeof(key.unique_hashes));
    const auto [it, first] = last_by_shaders.try_emplace(shaders, key);
    if (first) {
        return;
    }
    const FixedPipelineState& now = key.state;
    const FixedPipelineState& before = it->second.state;
    std::string diffs;
    if (now.raw1 != before.raw1) diffs += fmt::format(" raw1^{:#x}", now.raw1 ^ before.raw1);
    if (now.raw2 != before.raw2) diffs += fmt::format(" raw2^{:#x}", now.raw2 ^ before.raw2);
    if (now.color_formats != before.color_formats) {
        diffs += fmt::format(" color_formats {:02x} -> {:02x}", fmt::join(before.color_formats, ""),
                             fmt::join(now.color_formats, ""));
    }
    if (now.alpha_test_ref != before.alpha_test_ref) {
        diffs += fmt::format(" alpha_test_ref {:#x} -> {:#x} (func {})", before.alpha_test_ref,
                             now.alpha_test_ref, now.alpha_test_func.Value());
    }
    if (now.point_size != before.point_size) {
        diffs += fmt::format(" point_size {:#x} -> {:#x}", before.point_size, now.point_size);
    }
    if (now.viewport_swizzles != before.viewport_swizzles) diffs += " viewport_swizzles";
    if (now.attribute_types != before.attribute_types) {
        diffs += fmt::format(" attribute_types^{:#x}", now.attribute_types ^ before.attribute_types);
    }
    if (std::memcmp(&now.dynamic_state, &before.dynamic_state, sizeof(now.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    if (now.driver_id != before.driver_id || now.driver_version != before.driver_version) {
        diffs += " driver";
    }
    const size_t size = now.Size();
    if (diffs.empty() && size > offsetof(FixedPipelineState, attachments)) {
        diffs = fmt::format(" past attachments (key size {})", size);
    }
    it->second = key;
    if (logged++ < 400) {
        LOG_INFO(Render_Vulkan, "New pipeline for known shaders {:016x} differs in:{}", shaders,
                 diffs.empty() ? std::string{" nothing in the key prefix"} : diffs);
    }
}

// Without VK_EXT_mesh_shader: logs, once per shader set, what a skipped task/mesh pipeline would
// need from an emulation (task stage, per-primitive outputs, topology, output sizes).
template <typename Programs>
void ReportUnsupportedMeshPipeline(const GraphicsPipelineCacheKey& key, const Programs& programs) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    static u32 with_task{}, per_primitive{}, non_triangle{}, total{};
    u64 shaders{};
    for (const u64 hash : key.unique_hashes) {
        shaders = shaders * 31 + hash;
    }
    std::scoped_lock lock{mutex};
    if (!seen.insert(shaders).second) {
        return;
    }
    const Shader::IR::Program* task{};
    const Shader::IR::Program* mesh{};
    bool has_pixel{};
    for (size_t index = 0; index < programs.size(); ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const auto& program{programs[index]};
        if (program.stage == Shader::Stage::Task) {
            task = &program;
        } else if (program.stage == Shader::Stage::Mesh) {
            mesh = &program;
        } else if (program.stage == Shader::Stage::Fragment) {
            has_pixel = true;
        }
    }
    ++total;
    if (mesh == nullptr) {
        LOG_WARNING(Render_Vulkan, "Mesh emulation survey: pipeline {:016x} without a mesh stage",
                    shaders);
        return;
    }
    const auto& info{mesh->mesh};
    u32 position{}, generic{}, other{};
    std::string others;
    for (const u32 line : info.vertex_lines) {
        const auto attribute{static_cast<Shader::IR::Attribute>(line)};
        if (attribute >= Shader::IR::Attribute::PositionX &&
            attribute <= Shader::IR::Attribute::PositionW) {
            ++position;
        } else if (Shader::IR::IsGeneric(attribute)) {
            ++generic;
        } else {
            ++other;
            others += fmt::format(" {}", line);
        }
    }
    with_task += task != nullptr;
    per_primitive += !info.primitive_lines.empty();
    non_triangle += info.topology != 4;
    LOG_WARNING(Render_Vulkan,
                "Mesh emulation survey #{}: task={} (payload {} words, wg {}x{}x{}) mesh wg "
                "{}x{}x{} max_vertices={} max_primitives={} topology={} vertex_lines={} "
                "(position {} generic {} other {}:{}) primitive_lines={} pixel={} | totals: "
                "task {} per-primitive {} non-triangle {}",
                total, task != nullptr, info.payload_words, task ? task->workgroup_size[0] : 0,
                task ? task->workgroup_size[1] : 0, task ? task->workgroup_size[2] : 0,
                mesh->workgroup_size[0], mesh->workgroup_size[1], mesh->workgroup_size[2],
                info.max_vertices, info.max_primitives, info.topology, info.vertex_lines.size(),
                position, generic, other, others, info.primitive_lines.size(), has_pixel,
                with_task, per_primitive, non_triangle);
}
} // Anonymous namespace

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    namespace StutterTrace = VideoCommon::StutterTrace;
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        ReportNewPipelineKey(graphics_key);
        if (scheduled_prebuild) {
            pipeline = scheduled_prebuild->Take(graphics_key);
        }
        if (StutterTrace::Enabled()) {
            StutterTrace::CountLookup(pipeline ? StutterTrace::Lookup::GraphicsBackground
                                      : disk_pipelines.contains(graphics_key.Hash())
                                          ? StutterTrace::Lookup::GraphicsRecorded
                                          : StutterTrace::Lookup::GraphicsNew);
        }
        if (!pipeline) {
            const auto keep = evicted_pointer_slots.Take(graphics_key.Hash());
            pipeline = CreateGraphicsPipeline(keep ? &*keep : nullptr);
        }
    } else if (StutterTrace::Enabled() && pipeline && !pipeline->WasUsed()) {
        StutterTrace::CountLookup(StutterTrace::Lookup::GraphicsPrebuilt);
    }
    if (!pipeline) {
        return nullptr;
    }
    constexpr auto save_interval = std::chrono::seconds{15};
    if (used_this_session.insert(pipeline.get()).second &&
        std::chrono::steady_clock::now() - used_pipelines_saved > save_interval) {
        SaveUsedPipelines();
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    current_pipeline->MarkUsed(frame_count);
    return BuiltPipeline(current_pipeline);
}

void PipelineCache::TickFrame() {
    ++frame_count;
    native_wait_budget_us =
        static_cast<s64>(Settings::values.native_pipeline_wait_ms.GetValue()) * 1000;
    while (!evicted_pipelines.empty() && scheduler.IsFree(evicted_pipelines.front().first)) {
        evicted_pipelines.pop_front();
    }
    while (!retired_compute_pipelines.empty() &&
           scheduler.IsFree(retired_compute_pipelines.front().first) &&
           retired_compute_pipelines.front().second->IsBuildTaskDone()) {
        retired_compute_pipelines.pop_front();
    }
    constexpr u64 EvictCheckFrames = 600;
    if (frame_count % EvictCheckFrames == 0) {
        EvictUnusedPipelines();
    }
}

void PipelineCache::EvictUnusedPipelines() {
    // Building a pipeline again on MoltenVK means a Metal compile mid-game, far dearer than the
    // rebuild cost below, so none are evicted there.
    if (device.IsMoltenVK()) {
        return;
    }
    // EDEN_NX2_PRELOAD_ALL_PIPELINES asked for every cached pipeline built: evicting the ones not
    // used yet threw the preload away after 9000 frames (Ocarina: 22,510 of 24,285, then rebuilt
    // one by one in new places).
    if (PreloadAllPipelines()) {
        return;
    }
    // Each built pipeline holds ~0.45 MB of driver memory and none was ever destroyed, so it grew
    // with every place a session went (title 1: 1,500 on its title screen alone). One unused for
    // this long is destroyed; rebuilding it later requires a full GPU-thread shader translation
    // (~8.5 ms each) plus a driver pipeline cache hit.
    constexpr u64 EvictAfterFrames = 9000;
    if (frame_count <= EvictAfterFrames) {
        return;
    }
    const u64 cutoff = frame_count - EvictAfterFrames;
    const u64 tick = scheduler.CurrentTick();
    // ~0.9 GB of driver memory at ~0.45 MB each.
    constexpr size_t MaxUnusedPrebuiltPipelines = 2000;
    size_t kept_prebuilt = 0;
    size_t evicted = 0;
    for (auto it = graphics_cache.begin(); it != graphics_cache.end();) {
        GraphicsPipeline* const pipeline = it->second.get();
        if (pipeline == nullptr || pipeline == current_pipeline || !pipeline->IsBuilt() ||
            pipeline->LastUse() >= cutoff) {
            ++it;
            continue;
        }
        // A pipeline built from the disk cache at boot that no draw has used yet stays, up to a
        // budget: evicting it throws the prebuild away, and the game translates it again on the
        // GPU thread when it reaches that content (OoT: 1174 of 1346 evicted on the title screen,
        // 684 rebuilt in 10 s on entering the field). Past the budget they go like the rest, so a
        // large cache can't hold their driver memory for the whole session.
        if (!pipeline->WasUsed() && kept_prebuilt < MaxUnusedPrebuiltPipelines) {
            ++kept_prebuilt;
            ++it;
            continue;
        }
        if (used_this_session.erase(pipeline) != 0) {
            used_and_evicted.insert(it->first.Hash());
        }
        evicted_pointer_slots.Remember(it->first.Hash(), pipeline->MappedPointers());
        evicted_pipelines.emplace_back(tick, std::move(it->second));
        it = graphics_cache.erase(it);
        ++evicted;
    }
    if (evicted == 0) {
        return;
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    LOG_INFO(Render_Vulkan, "Evicted {} graphics pipelines unused for {} frames, {} left", evicted,
             EvictAfterFrames, graphics_cache.size());
}

/// Returns `pipeline` when it is built or the current draw must wait for it, and nullptr when the
/// draw may be skipped while it builds (asynchronous shaders; native draws only with
/// EDEN_NX2_ASYNC_DISCARD=1). Each nullptr returned for an unbuilt pipeline is counted for the log.
GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    // EDEN_NX2_ASYNC_DISCARD=1: skip native draws whose pipeline is still building instead of
    // waiting for it (experimental; the draw may be missing for a frame or more).
    /// Reads EDEN_NX2_ASYNC_DISCARD once and logs when it is on.
    static const bool discard_native = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_DISCARD");
        const bool enabled = value != nullptr && value[0] == '1';
        if (enabled) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_DISCARD: native draws skip unbuilt pipelines");
        }
        return enabled;
    }();
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    // The environment variable wins over the setting; the setting is read per call.
    NativePendingMode mode = NativePendingMode::Wait;
    if (native) {
        switch (Settings::values.native_async_draws.GetValue()) {
        case Settings::NativeAsyncDraws::Bounded:
            mode = NativePendingMode::Bounded;
            break;
        case Settings::NativeAsyncDraws::Skip:
            mode = NativePendingMode::Skip;
            break;
        default:
            break;
        }
        if (discard_native) {
            mode = NativePendingMode::Skip;
        }
    }
    // Native draws are judged by the heuristic only when a mode other than Wait is chosen.
    const bool can_discard = CanDiscardPendingDraw(
        native, maxwell3d->regs.zeta_enable != 0, draw_state.draw_element_count,
        draw_state.draw_element_count, native && mode != NativePendingMode::Wait);
    bool discard = can_discard;
    if (native) {
        switch (DecideNativePending(mode, can_discard, native_wait_budget_us)) {
        case NativeDrawAction::Wait:
            discard = false;
            break;
        case NativeDrawAction::Skip:
            discard = true;
            break;
        case NativeDrawAction::WaitTimed: {
            const auto begin = std::chrono::steady_clock::now();
            const bool built =
                pipeline->WaitBuilt(std::chrono::microseconds{native_wait_budget_us});
            const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            native_wait_budget_us -= spent;
            native_wait_used_us += spent;
            static auto window = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() - window > std::chrono::seconds{10}) {
                LOG_INFO(Render_Vulkan,
                         "Bounded pipeline wait: {:.1f} ms spent in the last 10 s (budget {} ms "
                         "per frame)",
                         static_cast<double>(native_wait_used_us) / 1000.0,
                         Settings::values.native_pipeline_wait_ms.GetValue());
                native_wait_used_us = 0;
                window = std::chrono::steady_clock::now();
            }
            if (built) {
                return pipeline;
            }
            discard = true;
            break;
        }
        }
    }
    if (discard) {
        static std::atomic<u64> skipped{};
        if ((skipped.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
            // Counts lookups, not draws: a draw state can look its pipeline up more than once.
            LOG_INFO(Render_Vulkan,
                     "Async shaders: {} pipeline lookups skipped for unbuilt pipelines",
                     skipped.load(std::memory_order_relaxed));
        }
    }
    return discard ? nullptr : pipeline;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools,
    ParallelTranslation* parallel) try {
    auto hash = key.Hash();
    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);
    // Declared before the programs, whose IR they hold
    std::vector<std::shared_ptr<ShaderPools>> stage_pools;
    SCOPE_EXIT {
        for (std::shared_ptr<ShaderPools>& stage : stage_pools) {
            ReturnPools(std::move(stage));
        }
    };
    std::vector<VideoCommon::StageTranslation> translations;
    std::array<VideoCommon::StageTranslation*, Maxwell::MaxShaderProgram> translated{};
    if (parallel) {
        translations.reserve(envs.size());
        size_t stage_env{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::Environment& env{*envs[stage_env++]};
            if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
                !device.IsGeometryShaderSupported() && env.IsNativeProgram()) {
                continue; // The pipeline is skipped below before this stage is needed
            }
            translations.push_back({
                .env = &env,
                .pools = stage_pools.emplace_back(TakeSparePools()).get(),
                .cfg_offset = static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader)),
                .exits_to_dispatcher = index == 0,
            });
            translated[index] = &translations.back();
        }
        const auto begin = std::chrono::steady_clock::now();
        VideoCommon::TranslateStages(*shader_jobs, translations, host_info);
        parallel->wait += std::chrono::steady_clock::now() - begin;
        for (const VideoCommon::StageTranslation& translation : translations) {
            parallel->stages += translation.time;
        }
        parallel->count = translations.size();
    }
    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};
    bool mesh_unsupported{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;
        if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
            !device.IsGeometryShaderSupported() && env.IsNativeProgram()) {
            // A native (Switch 2) geometry program on a device without geometry shaders: the
            // pipeline cannot run it, so its draws are skipped.
            static std::atomic<bool> logged{};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "Skipping native geometry shader pipelines: the device has no geometry shaders");
            }
            return nullptr;
        }
        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline; the draw is
        // skipped (see the pipeline check in the rasterizer). The stages are still translated
        // so the skipped pipeline's needs can be reported below.
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "{} has no VK_EXT_mesh_shader: pipelines with task/mesh shaders are "
                            "skipped, so the geometry they draw (such as foliage and props) will "
                            "be missing",
                            device.GetModelName());
            }
            mesh_unsupported = true;
        }

        const auto translate = [&] {
            if (translated[index]) {
                return translated[index]->Take();
            }
            const u32 cfg_offset{
                static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            return TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        };
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = translate();
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{translate()};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env, host_info);
        }

        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return nullptr;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    // Mesh emulation without VK_EXT_mesh_shader (prototype): a mesh program, after an optional
    // task program, drawing triangles without per-primitive outputs, before a pixel program only.
    std::optional<MeshEmulationSetup> mesh_emulation;
    size_t mesh_index{};
    if (mesh_unsupported) {
        ReportUnsupportedMeshPipeline(key, programs);
        const bool has_task{key.unique_hashes[1] != 0 &&
                            programs[1].stage == Shader::Stage::Task};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0 && programs[index].stage == Shader::Stage::Mesh) {
                mesh_index = index;
                break;
            }
        }
        bool eligible{mesh_index != 0 && (mesh_index == 1 || has_task) &&
                      layer_source_program == nullptr};
        const Shader::IR::Program& mesh{programs[mesh_index]};
        eligible = eligible && mesh.mesh.topology == 4 && mesh.mesh.primitive_lines.empty() &&
                   !mesh.mesh.vertex_lines.empty() && mesh.mesh.max_vertices != 0 &&
                   mesh.mesh.max_primitives != 0;
        for (size_t index = 1; index < Maxwell::MaxShaderProgram && eligible; ++index) {
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            eligible = emulated || key.unique_hashes[index] == 0 ||
                       programs[index].stage == Shader::Stage::Fragment;
        }
        if (!eligible) {
            return nullptr;
        }
        // The output buffer's binding follows every program's (unified bindings, one per
        // descriptor element, as the SPIR-V backend numbers them). The task and mesh programs
        // share the compute stage, so their descriptors and the emulation's own buffers count
        // against one set of per-stage limits.
        u32 output_binding{};
        u32 compute_storage_buffers{has_task ? 4U : 1U};
        u32 compute_sampled_images{};
        u32 compute_resources{compute_storage_buffers};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            const Shader::Info& info{programs[index].info};
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            const auto add = [&](const auto& descriptors, u32* category) {
                const u32 count{Shader::NumDescriptors(descriptors)};
                output_binding += count;
                if (emulated) {
                    compute_resources += count;
                    if (category) {
                        *category += count;
                    }
                }
            };
            add(info.constant_buffer_descriptors, nullptr);
            add(info.storage_buffers_descriptors, &compute_storage_buffers);
            add(info.texture_buffer_descriptors, &compute_sampled_images);
            add(info.image_buffer_descriptors, nullptr);
            add(info.texture_descriptors, &compute_sampled_images);
            add(info.image_descriptors, nullptr);
        }
        if (compute_storage_buffers > device.GetMaxPerStageDescriptorStorageBuffers() ||
            compute_sampled_images > device.GetMaxPerStageDescriptorSampledImages() ||
            compute_resources > device.GetMaxPerStageResources()) {
            LOG_WARNING(Render_Vulkan,
                        "Mesh emulation skipped: {} storage buffers, {} sampled images and {} "
                        "resources exceed the compute stage limits",
                        compute_storage_buffers, compute_sampled_images, compute_resources);
            return nullptr;
        }
        mesh_emulation.emplace();
        mesh_emulation->output_binding = output_binding;
        if (has_task) {
            // After the output: the task payload, offset and argument buffers.
            mesh_emulation->has_task = true;
            mesh_emulation->mesh_stage = static_cast<u32>(mesh_index - 1);
            mesh_emulation->payload_stride = (std::max)(programs[1].mesh.payload_words, 2U);
            mesh_emulation->payload_binding = output_binding + 1;
            mesh_emulation->offsets_binding = output_binding + 2;
            mesh_emulation->args_binding = output_binding + 3;
            mesh_emulation->distribution_spirv =
                Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile);
        }
        mesh_emulation->block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh);
        mesh_emulation->max_primitives = mesh.mesh.max_primitives;
        mesh_emulation->max_vertices = mesh.mesh.max_vertices;
        mesh_emulation->vertex_lines = static_cast<u32>(mesh.mesh.vertex_lines.size());
        for (u32 line = 0; line < mesh.mesh.vertex_lines.size(); ++line) {
            const u32 attribute{mesh.mesh.vertex_lines[line]};
            const u32 position_x{static_cast<u32>(Shader::IR::Attribute::PositionX)};
            if (attribute >= position_x && attribute < position_x + 4) {
                mesh_emulation->position_lines[attribute - position_x] = line;
            }
        }
        static std::atomic<u32> emulated{};
        if (const u32 count{++emulated}; count <= 64) {
            // The input region addresses the mesh program reads (0 is its workgroup index).
            std::string reads;
            for (const Shader::IR::Block* block : mesh.blocks) {
                for (const Shader::IR::Inst& inst : block->Instructions()) {
                    if (inst.GetOpcode() != Shader::IR::Opcode::IsbeRead) {
                        continue;
                    }
                    const Shader::IR::Value address{inst.Arg(0)};
                    reads += address.IsImmediate() ? fmt::format(" {:#x}", address.U32())
                                                   : std::string{" dynamic"};
                }
            }
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation #{}: pipeline {:016x}, {} vertices x {} lines, {} triangles "
                     "per workgroup, output binding {}, task {} (payload {} words), input "
                     "reads:{}",
                     count, hash, mesh.mesh.max_vertices, mesh.mesh.vertex_lines.size(),
                     mesh.mesh.max_primitives, output_binding, has_task,
                     mesh_emulation->payload_stride, reads.empty() ? " none" : reads);
        }
    }
    // A tessellation pipeline the driver would refuse is skipped like a missing mesh pipeline:
    // the draw is dropped (see the pipeline check in the rasterizer) instead of failing the build.
    {
        const Shader::IR::Program* control{};
        const Shader::IR::Program* evaluation{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            if (programs[index].stage == Shader::Stage::TessellationControl) {
                control = &programs[index];
            } else if (programs[index].stage == Shader::Stage::TessellationEval) {
                evaluation = &programs[index];
            }
        }
        if (control != nullptr || evaluation != nullptr) {
            const u32 max_patch_size = device.GetMaxTessellationPatchSize();
            const char* reason = nullptr;
            if (!device.IsTessellationSupported()) {
                reason = "the device has no tessellation shaders";
            } else if (control == nullptr || evaluation == nullptr) {
                reason = "it has only one of the control and evaluation stages";
            } else if (control->invocations == 0 || control->invocations > max_patch_size) {
                reason = "the control shader's output vertex count is out of range";
            }
            if (reason != nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping tessellation pipeline: {} (tcs={} tes={}, tcs output "
                            "vertices {}, patch control points {}, device max patch size {})",
                            reason, control != nullptr, evaluation != nullptr,
                            control != nullptr ? control->invocations : 0,
                            key.state.patch_control_points_minus_one.Value() + 1, max_patch_size);
                return nullptr;
            }
        }
    }
    // Each stage stays within the limit on its own; together they share one descriptor set.
    if (const u32 set_limit = host_info.max_descriptor_set_storage_buffers; set_limit != 0) {
        size_t storage_buffers{};
        for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
             index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                storage_buffers +=
                    Shader::NumDescriptors(programs[index].info.storage_buffers_descriptors);
            }
        }
        // The fallback binds its output buffer (and, for task draws, payload,
        // offsets and indirect arguments) into the same descriptor set.
        const size_t auxiliary_buffers{
            mesh_emulation ? (mesh_emulation->has_task ? 4U : 1U) : 0U};
        if (!FitsStorageBufferSetLimit(storage_buffers, auxiliary_buffers, set_limit)) {
            throw Shader::NotImplementedException(
                "Binding {} storage buffers ({} for mesh fallback) to a descriptor set limited to {}",
                storage_buffers + auxiliary_buffers, auxiliary_buffers, set_limit);
        }
    }
    u64 deferred_ir_bytes = 0;
    std::unique_ptr<QueuedIrBudget::Reservation> ir_reservation;
    if (emit_pools) {
        deferred_ir_bytes = IrBytes(programs);
        ir_reservation = queued_ir.TryReserve(deferred_ir_bytes);
    }
    const bool defer_emission = ir_reservation != nullptr;
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> deferred_runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        if (mesh_emulation && (program.stage == Shader::Stage::Mesh ||
                               program.stage == Shader::Stage::Task)) {
            runtime_info.mesh_as_compute = true;
            runtime_info.mesh_output_binding = mesh_emulation->output_binding;
            if (mesh_emulation->has_task) {
                runtime_info.mesh_task_payload_binding = mesh_emulation->payload_binding;
                runtime_info.mesh_task_offsets_binding = mesh_emulation->offsets_binding;
                runtime_info.mesh_task_args_binding = mesh_emulation->args_binding;
                runtime_info.mesh_task_payload_stride = mesh_emulation->payload_stride;
            }
            if (program.stage == Shader::Stage::Mesh) {
                mesh_emulation->vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
                    profile, program.mesh, mesh_emulation->output_binding,
                    mesh_emulation->has_task ? mesh_emulation->args_binding : ~0U,
                    runtime_info.convert_depth_mode && !profile.support_native_ndc);
            }
        }
        ConvertLegacyToGeneric(program, runtime_info);
        // EDEN_NX2_MESH_EMULATION_DUMP=<dir>: write the emulation's vertex shader for spirv-val.
        // (Emitting the mesh program a second time here to dump its compute shader crashed the
        // build that followed, so only the vertex shader is written.)
        if (mesh_emulation && program.stage == Shader::Stage::Mesh) {
            if (const char* const dump{std::getenv("EDEN_NX2_MESH_EMULATION_DUMP")}) {
                const auto& code{mesh_emulation->vertex_spirv};
                std::ofstream file{fmt::format("{}/{:016x}.vert.spv", dump, hash),
                                   std::ios::binary};
                file.write(reinterpret_cast<const char*>(code.data()),
                           static_cast<std::streamsize>(code.size() * sizeof(u32)));
            }
        }
        if (defer_emission) {
            deferred_runtime_infos[stage_index] = runtime_info;
            if (VideoCommon::Nx2Perf::Verify()) {
                reference_spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
            }
        } else {
            const auto begin = std::chrono::steady_clock::now();
            modules[stage_index] = EmitShaderModule(device, profile, runtime_info, program, binding,
                                                   key.unique_hashes[index], stage_index);
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
        }
        previous_stage = &program;
    }
    std::function<std::array<vk::ShaderModule, Maxwell::MaxShaderStage>()> emit_stages;
    if (defer_emission) {
        auto job = std::make_shared<DeferredStages>();
        job->reservation = std::move(ir_reservation);
        job->pools = emit_pools;
        job->stage_pools = std::move(stage_pools);
        stage_pools.clear();
        job->programs = std::move(programs);
        job->runtime_infos = std::move(deferred_runtime_infos);
        job->reference_spirv = std::move(reference_spirv);
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                infos[stage] = &job->programs[stage + 1].info;
            }
        }
        emit_stages = [this, job = std::move(job), hashes = key.unique_hashes]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> stage_modules;
            Shader::Backend::Bindings stage_binding;
            for (size_t stage = 0; stage < stage_modules.size(); ++stage) {
                if (job->runtime_infos[stage]) {
                    stage_modules[stage] =
                        EmitShaderModule(device, profile, *job->runtime_infos[stage],
                                         job->programs[stage + 1], stage_binding,
                                         hashes[stage + 1], stage, job->reference_spirv[stage]);
                }
            }
            auto completed_pools = std::move(job->pools);
            auto completed_stages = std::move(job->stage_pools);
            job.reset();
            ReturnPools(std::move(completed_pools));
            for (std::shared_ptr<ShaderPools>& stage : completed_stages) {
                ReturnPools(std::move(stage));
            }
            AddRuntimeSpirvTime(1, begin);
            return stage_modules;
        };
    }
    static_assert(VideoCommon::EvictedPointerSlots::NumPrograms == Maxwell::MaxShaderProgram);
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    for (size_t index = 0, env_index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] != 0) {
            mapped_pointers[index] = envs[env_index++]->MappedStoragePointerKeys();
        }
    }
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, pipeline_binaries.get(),
        &shader_notify, device, descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
        BuildQueue(build_in_parallel, defer_emission ? deferred_ir_bytes : 0), statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(emit_stages),
        std::move(mesh_emulation));
    pipeline->SetMappedPointers(std::move(mapped_pointers));
    return pipeline;

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        try {
            // Walks the program for the dump; the one that failed may fail here again, and an
            // exception leaving this handler would end the process.
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        } catch (const Shader::Exception&) {
        }
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

namespace {
// Pipelines built on the GPU thread at run time (shader translation, and the SPIR-V unless a worker
// emits it; the driver compile is on workers): the first 400 over 2 ms and every one over 100 ms
// are logged, and the totals every 10 s, to tell how much of the stutter on new content is
// translation.
void ReportRuntimePipeline(const char* kind, std::chrono::steady_clock::time_point begin,
                           const std::string& shaders,
                           std::chrono::nanoseconds translation_wait = {},
                           std::chrono::nanoseconds stage_translation = {}) {
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static std::chrono::steady_clock::duration total{};
    static std::chrono::nanoseconds total_translation_wait{}, total_stage_translation{};
    static u32 count{}, logged{};
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - begin;
    total += elapsed;
    total_translation_wait += translation_wait;
    total_stage_translation += stage_translation;
    ++count;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    // Slow builds are always logged with their shaders, to find the programs that are slow to
    // translate.
    if (elapsed > std::chrono::milliseconds{100} ||
        (elapsed > std::chrono::milliseconds{2} && logged < 400)) {
        ++logged;
        LOG_INFO(Render_Vulkan, "{} pipeline built on the GPU thread in {:.1f} ms: {}", kind,
                 ms(elapsed), shaders);
    }
    if (now - window > std::chrono::seconds{10}) {
        const auto take = [](auto& counter) {
            return std::chrono::nanoseconds{counter.exchange(0, std::memory_order::relaxed)};
        };
        // Shader cache v2: the GPU thread's wait for graphics stages translated at once on
        // workers, the translation time they took together, and compute SPIR-V on workers
        const std::string v2 = fmt::format(
            ", {:.1f} ms on the GPU thread waiting for {:.1f} ms of graphics "
            "stage translation on workers, {:.1f} ms compute SPIR-V on workers",
            ms(total_translation_wait), ms(total_stage_translation), ms(take(runtime_spirv_ns[2])));
        LOG_INFO(Render_Vulkan,
                 "Runtime pipelines in the last 10 s: {} built, {:.1f} ms on the GPU thread "
                 "({:.1f} ms of it graphics SPIR-V), {:.1f} ms graphics SPIR-V on workers, "
                 "{:.1f} ms of draws held for unbuilt pipelines{}",
                 count, ms(total), ms(take(runtime_spirv_ns[0])), ms(take(runtime_spirv_ns[1])),
                 ms(GraphicsPipeline::TakeHoldTime()), v2);
        total = {};
        total_translation_wait = {};
        total_stage_translation = {};
        count = 0;
        window = now;
    }
}
} // Anonymous namespace

void PipelineCache::WarnKeptPointer(size_t program, u64 slot) const {
    if ((slot >> 63) != 0) {
        return;
    }
    const u32 index = static_cast<u32>(slot >> 32);
    const u32 offset = static_cast<u32>(slot);
    const size_t stage = program == 0 ? 0 : program - 1;
    const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
    GPUVAddr value = 0;
    const char* reason = nullptr;
    if (index >= cbufs.size() || !cbufs[index].enabled ||
        offset + sizeof(u64) > cbufs[index].size ||
        !gpu_memory->IsFullyMappedRange(cbufs[index].address + offset, sizeof(u64))) {
        reason = "constant buffer not bound";
    } else {
        value = gpu_memory->Read<u64>(cbufs[index].address + offset);
        if (value == 0) {
            reason = "null";
        } else if (!gpu_memory->IsWithinGPUAddressRange(value)) {
            reason = "outside the GPU address range";
        } else if (!gpu_memory->IsFullyMappedRange(value, 1)) {
            reason = "unmapped";
        }
    }
    if (!reason) {
        return;
    }
    static std::atomic<u64> kept{};
    const u64 count = kept.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 32 || count % 1000 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "Graphics pipeline {:016x} translated again after eviction: s{}:c{}[{:#x}] "
                    "holds {:#x} ({}), kept as the storage buffer pointer it was ({} so far)",
                    graphics_key.Hash(), program, index, offset, value, reason, count);
    }
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    const VideoCommon::EvictedPointerSlots::Slots* keep_pointers) {
    const VideoCommon::Nx2Profile::Scope profile_scope{
        VideoCommon::Nx2Profile::Zone::GraphicsPipelineMiss};
    const auto begin = std::chrono::steady_clock::now();
    std::optional<ParallelTranslation> parallel;
    SCOPE_EXIT {
        std::string shaders = fmt::format("pipeline={:016x}", graphics_key.Hash());
        for (size_t stage = 0; stage < graphics_key.unique_hashes.size(); ++stage) {
            if (graphics_key.unique_hashes[stage] != 0) {
                shaders += fmt::format(" s{}:{:016x}", stage, graphics_key.unique_hashes[stage]);
            }
        }
        if (!parallel) {
            ReportRuntimePipeline("Graphics", begin, shaders);
            return;
        }
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        shaders += fmt::format(" translate_wait_ms={:.1f} stage_translate_ms={:.1f} stages={}",
                               ms(parallel->wait), ms(parallel->stages), parallel->count);
        ReportRuntimePipeline("Graphics", begin, shaders, parallel->wait, parallel->stages);
    };
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    if (keep_pointers) {
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (graphics_key.unique_hashes[index] == 0) {
                continue;
            }
            environments.envs[index].KeepStoragePointers((*keep_pointers)[index]);
            for (const u64 slot : (*keep_pointers)[index]) {
                WarnKeptPointer(index, slot);
            }
        }
    }

    // Shader cache v2: two or more stages translate at once on the shader jobs' CPU workers while
    // this thread waits. Only this thread may flush guest memory, as reading a texture descriptor
    // does, so the descriptor pool is flushed here first; with this thread waiting nothing can
    // make it dirty again. A pool too large to flush on every miss translates here as before.
    if (std::ranges::count_if(graphics_key.unique_hashes, [](u64 hash) { return hash != 0; }) >=
        2) {
        constexpr u64 MaxFlushedTexturePoolBytes = u64{1} << 20;
        const auto& tex_header = maxwell3d->regs.tex_header;
        const u64 pool_bytes = (u64{tex_header.limit} + 1) * sizeof(Tegra::Texture::TICEntry);
        if (pool_bytes <= MaxFlushedTexturePoolBytes) {
            gpu_memory->FlushRegion(tex_header.Address(), pool_bytes);
            for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
                if (graphics_key.unique_hashes[index] != 0) {
                    environments.envs[index].SetTexturePoolFlushed();
                }
            }
            parallel.emplace();
        } else {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_INFO(Render_Vulkan,
                         "Texture descriptor pool of {} bytes: pipelines that use it translate "
                         "on the GPU thread",
                         pool_bytes);
            }
        }
    }

    // Native programs emit their SPIR-V in the pipeline's job instead of here
    std::shared_ptr<ShaderPools> emit_pools;
    if (maxwell3d->GetNativeMacroEngine().native_turing) {
        emit_pools = TakeSparePools();
    } else {
        main_pools.ReleaseContents();
    }
    ShaderPools& pools = emit_pools ? *emit_pools : main_pools;
    auto pipeline{CreateGraphicsPipeline(pools, graphics_key, environments.Span(), nullptr, true,
                                         std::move(emit_pools), parallel ? &*parallel : nullptr)};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    if (disk_pipelines.contains(graphics_key.Hash())) {
        // In the disk cache already, only not built at boot.
        QueueVulkanPipelineCacheFlush();
        return pipeline;
    }
    disk_pipelines.insert(graphics_key.Hash());
    serialization_thread.QueueWork([this, key = graphics_key, envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        const std::vector<u8> payload{VideoCommon::SerializePipelinePayload(
            std::span(reinterpret_cast<const char*>(&key), sizeof(key)), env_ptrs,
            host_info.EnvironmentReadProperties())};
        if (!payload.empty()) {
            cache_v2->Writer().Append(VideoCommon::ShaderCacheV2::RecordKind::Graphics,
                                      GraphicsRecordKey(key), payload);
        }
    });
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                               const ShaderInfo* shader,
                                                               std::span<const u64> keep_pointers) {
    const VideoCommon::Nx2Profile::Scope profile_scope{
        VideoCommon::Nx2Profile::Zone::ComputePipelineMiss};
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Compute", begin,
                              fmt::format("pipeline={:016x} s6:{:016x} address={:#x}", key.Hash(),
                                          key.unique_hash,
                                          kepler_compute->regs.code_loc.Address() +
                                              kepler_compute->launch_description.program_start));
    };
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    env.KeepStoragePointers(keep_pointers);

    // The SPIR-V is emitted by the pipeline's job instead of here
    std::shared_ptr<ShaderPools> emit_pools = TakeSparePools();
    ShaderPools& pools = *emit_pools;
    auto entry{CreateComputePipeline(pools, key, env, nullptr, true, std::move(emit_pools))};
    if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        directory && !entry.pipeline && env.CanBeSerialized()) {
        const auto path = std::filesystem::path{directory} /
                          fmt::format("failed-compute-{:016x}-v{}.env", key.unique_hash, CACHE_VERSION);
        std::ofstream file{path, std::ios::binary};
        if (file) {
            env.Serialize(file, host_info.EnvironmentReadProperties());
            LOG_INFO(Render_Vulkan, "Captured failed compute environment: {} written={}",
                     path.string(), file.good());
        }
    }
    if (!entry.pipeline || pipeline_cache_filename.empty()) {
        return entry;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        const std::array<const GenericEnvironment*, 1> envs{&env_};
        const std::vector<u8> payload{VideoCommon::SerializePipelinePayload(
            std::span(reinterpret_cast<const char*>(&key), sizeof(key)), envs,
            host_info.EnvironmentReadProperties())};
        if (!payload.empty()) {
            cache_v2->Writer().Append(VideoCommon::ShaderCacheV2::RecordKind::Compute,
                                      ComputeRecordKey(payload), payload);
        }
    });
    QueueVulkanPipelineCacheFlush();
    return entry;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel,
    std::shared_ptr<ShaderPools> emit_pools) try {
    SCOPE_EXIT {
        if (emit_pools) {
            ReturnPools(std::move(emit_pools));
        }
    };
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return {.failure = "Device reports broken compute support"};
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY || driver_id == VK_DRIVER_ID_MOLTENVK;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    std::unique_ptr<ComputePipeline> pipeline;
    const u64 ir_bytes = emit_pools ? IrBytes(std::span(&program, 1)) : 0;
    if (auto ir_reservation = emit_pools ? queued_ir.TryReserve(ir_bytes) : nullptr) {
        auto job = std::make_shared<DeferredCompute>();
        if (VideoCommon::Nx2Perf::Verify()) {
            job->reference_spirv = EmitComputeShader(profile, program, key.unique_hash);
        }
        job->reservation = std::move(ir_reservation);
        job->pools = std::move(emit_pools);
        job->program = std::move(program);
        // Copied by the pipeline before its job can run
        const Shader::Info& info = job->program.info;
        auto emit = [this, job, shader_hash = key.unique_hash]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            vk::ShaderModule module{EmitComputeModule(device, profile, job->program, shader_hash,
                                                      job->reference_spirv)};
            auto completed_pools = std::move(job->pools);
            job.reset();
            ReturnPools(std::move(completed_pools));
            AddRuntimeSpirvTime(2, begin);
            return module;
        };
        pipeline = std::make_unique<ComputePipeline>(
            device, scheduler, vulkan_pipeline_cache, pipeline_binaries.get(), hash,
            descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
            BuildQueue(build_in_parallel, ir_bytes), statistics,
            &shader_notify, info, vk::ShaderModule{}, key.unique_hash, std::move(emit));
    } else {
        vk::ShaderModule module{EmitComputeModule(device, profile, program, key.unique_hash)};
        const Shader::Info info{std::move(program.info)};
        if (!build_in_parallel) {
            // The driver compiles on this thread. Precache and boot workers would otherwise hold their
            // IR through the compile, up to hundreds of MiB for one large shader.
            program = {};
            pools.block.Free();
            pools.inst.Free();
        }
        pipeline = std::make_unique<ComputePipeline>(
            device, scheduler, vulkan_pipeline_cache, pipeline_binaries.get(), hash,
            descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
            BuildQueue(build_in_parallel), statistics, &shader_notify, info, std::move(module),
            key.unique_hash);
    }
    return {.pipeline = std::move(pipeline),
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Compute pipeline compilation failed shader={:#016x} pipeline={:#016x}: {}",
              key.unique_hash, key.Hash(), exception.what());
    return {.failure = exception.what()};
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) {
    // Written beside the cache and renamed over it: the emulator stopped mid-write left the cache
    // empty, and every later boot failed to read it.
    auto temporary = filename;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
        return;
    }
    try {
        file.exceptions(std::ifstream::failbit);
        file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

        size_t cache_size = 0;
        std::vector<char> cache_data;
        if (pipeline_cache) {
            pipeline_cache.Read(&cache_size, nullptr);
            cache_data.resize(cache_size);
            pipeline_cache.Read(&cache_size, cache_data.data());
        }
        file.write(cache_data.data(), cache_size);
        file.close();
    } catch (const std::ios_base::failure& e) {
        // The cache stays as it was: only the file being written goes.
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        file.exceptions(std::ios::goodbit);
        file.close();
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return;
    }
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), error.message());
        std::filesystem::remove(temporary, error);
        return;
    }

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        // A file cut short (the emulator stopped while writing it) would throw from the reads
        // below, and macOS's libc++ throws an ios_base::failure the handler doesn't match.
        if (static_cast<size_t>(end) < magic_number.size() + sizeof(cache_version)) {
            file.close();
            LOG_ERROR(Common_Filesystem, "Truncated Vulkan driver pipeline cache file, deleting it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: {}", Common::FS::PathToUTF8String(filename));

        try {
            return create_pipeline_cache(cache_size, cache_data.data());
        } catch (const vk::Exception& e) {
            // The blob came from another driver or driver version (e.g. switching between
            // MoltenVK and KosmicKrisp on macOS): the driver rejects it. Start empty instead of
            // aborting, and drop the stale file so the next run doesn't retry it.
            LOG_ERROR(Render_Vulkan,
                      "Driver rejected the Vulkan pipeline cache file ({}), starting with an empty "
                      "cache and deleting it",
                      e.what());
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
