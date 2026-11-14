// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_build_failure.h"
#include "video_core/renderer_vulkan/pipeline_build_trace.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"

#include "common/bit_field.h"
#include "common/scope_exit.h"
#include "common/zstd_compression.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_binaries.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_cache_v2.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_handle_program.h"
#include "video_core/shader_notify.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"

#if defined(__clang__) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE __attribute__((always_inline))
#elif defined(_MSC_VER) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE [[msvc::forceinline]]
#else
#define LAMBDA_FORCEINLINE
#endif

namespace Vulkan {
namespace {
using boost::container::small_vector;
using boost::container::static_vector;
using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;

constexpr size_t NUM_STAGES = Maxwell::MaxShaderStage;

// The stage of a program: task and mesh programs sit in the vertex and tessellation slots.
VkShaderStageFlagBits StageFlag(const Shader::Info& info, VkShaderStageFlagBits slot_stage) {
    if (info.is_task) {
        return VK_SHADER_STAGE_TASK_BIT_EXT;
    }
    if (info.is_mesh) {
        return VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    return slot_stage;
}

DescriptorLayoutBuilder MakeBuilder(const Device& device, std::span<const Shader::Info> infos,
                                    const std::optional<MeshEmulationSetup>& mesh_emulation = {}) {
    DescriptorLayoutBuilder builder{device};
    if (mesh_emulation) {
        builder.EnableMeshEmulation();
    }
    for (size_t index = 0; index < infos.size(); ++index) {
        static constexpr std::array stages{
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        // Mesh emulation: the mesh program's resources are read by its compute shader.
        const bool emulated{infos[index].is_mesh || infos[index].is_task};
        const VkShaderStageFlags stage{
            mesh_emulation && emulated
                ? static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_COMPUTE_BIT)
                : static_cast<VkShaderStageFlags>(StageFlag(infos[index], stages.at(index)))};
        builder.Add(infos[index], stage);
    }
    if (mesh_emulation) {
        builder.AddMeshEmulationOutput(mesh_emulation->output_binding);
        if (mesh_emulation->has_task) {
            builder.AddMeshEmulationOutput(mesh_emulation->payload_binding);
            builder.AddMeshEmulationOutput(mesh_emulation->offsets_binding);
            builder.AddMeshEmulationOutput(mesh_emulation->args_binding);
        }
    }
    return builder;
}

template <class StencilFace>
VkStencilOpState GetStencilFaceState(const StencilFace& face) {
    return {
        .failOp = MaxwellToVK::StencilOp(face.ActionStencilFail()),
        .passOp = MaxwellToVK::StencilOp(face.ActionDepthPass()),
        .depthFailOp = MaxwellToVK::StencilOp(face.ActionDepthFail()),
        .compareOp = MaxwellToVK::ComparisonOp(face.TestFunc()),
        .compareMask = 0,
        .writeMask = 0,
        .reference = 0,
    };
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    static constexpr std::array unsupported_topologies{
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
        // VK_PRIMITIVE_TOPOLOGY_QUAD_LIST_EXT,
    };
    return std::ranges::find(unsupported_topologies, topology) == unsupported_topologies.end();
}

bool IsLine(VkPrimitiveTopology topology) {
    static constexpr std::array line_topologies{
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        // VK_PRIMITIVE_TOPOLOGY_LINE_LOOP_EXT,
    };
    return std::ranges::find(line_topologies, topology) != line_topologies.end();
}

VkViewportSwizzleNV UnpackViewportSwizzle(u16 swizzle) {
    union Swizzle {
        u32 raw;
        BitField<0, 3, Maxwell::ViewportSwizzle> x;
        BitField<4, 3, Maxwell::ViewportSwizzle> y;
        BitField<8, 3, Maxwell::ViewportSwizzle> z;
        BitField<12, 3, Maxwell::ViewportSwizzle> w;
    };
    const Swizzle unpacked{swizzle};
    return VkViewportSwizzleNV{
        .x = MaxwellToVK::ViewportSwizzle(unpacked.x),
        .y = MaxwellToVK::ViewportSwizzle(unpacked.y),
        .z = MaxwellToVK::ViewportSwizzle(unpacked.z),
        .w = MaxwellToVK::ViewportSwizzle(unpacked.w),
    };
}

PixelFormat DecodeFormat(u8 encoded_format) {
    const auto format{static_cast<Tegra::RenderTargetFormat>(encoded_format)};
    if (format == Tegra::RenderTargetFormat::NONE) {
        return PixelFormat::Invalid;
    }
    return PixelFormatFromRenderTargetFormat(format);
}

RenderPassKey MakeRenderPassKey(const FixedPipelineState& state, const Device& device) {
    RenderPassKey key{};
    std::ranges::transform(state.color_formats, key.color_formats.begin(), DecodeFormat);
    if (state.depth_enabled != 0) {
        const auto depth_format{static_cast<Tegra::DepthFormat>(state.depth_format.Value())};
        key.depth_format = PixelFormatFromDepthFormat(depth_format);
    } else {
        key.depth_format = PixelFormat::Invalid;
    }
    key.samples = MaxwellToVK::MsaaMode(state.msaa_mode);
    const bool has_color = std::ranges::any_of(key.color_formats, [](PixelFormat format) {
        return format != PixelFormat::Invalid;
    });
    key.resolve_color =
        key.samples != VK_SAMPLE_COUNT_1_BIT && has_color && device.IsTiler();
    return key;
}

size_t NumAttachments(const FixedPipelineState& state) {
    size_t num{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; ++index) {
        const auto format{static_cast<Tegra::RenderTargetFormat>(state.color_formats[index])};
        if (format != Tegra::RenderTargetFormat::NONE) {
            num = index + 1;
        }
    }
    return num;
}

template <typename Spec>
bool Passes(const std::array<bool, NUM_STAGES>& stages,
            const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!Spec::enabled_stages[stage] && stages[stage]) {
            return false;
        }
        const auto& info{stage_infos[stage]};
        if constexpr (!Spec::has_storage_buffers) {
            if (!info.storage_buffers_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_texture_buffers) {
            if (!info.texture_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_image_buffers) {
            if (!info.image_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_images) {
            if (!info.image_descriptors.empty()) {
                return false;
            }
        }
    }
    return true;
}

using ConfigureFuncPtr = bool (*)(GraphicsPipeline*, bool);

template <typename Spec, typename... Specs>
ConfigureFuncPtr FindSpec(const std::array<bool, NUM_STAGES>& stages,
                          const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    if constexpr (sizeof...(Specs) > 0) {
        if (!Passes<Spec>(stages, stage_infos)) {
            return FindSpec<Specs...>(stages, stage_infos);
        }
    }
    return GraphicsPipeline::MakeConfigureSpecFunc<Spec>();
}

struct SimpleVertexFragmentSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleVertexSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, false};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleStorageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleImageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = true;
};

struct DefaultSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, true, true, true, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = true;
    static constexpr bool has_image_buffers = true;
    static constexpr bool has_images = true;
};

ConfigureFuncPtr ConfigureFunc(const std::array<bool, NUM_STAGES>& stages,
                               const std::array<Shader::Info, NUM_STAGES>& infos) {
    return FindSpec<SimpleVertexSpec, SimpleVertexFragmentSpec, SimpleStorageSpec, SimpleImageSpec,
                    DefaultSpec>(stages, infos);
}

std::atomic<std::chrono::nanoseconds::rep> hold_time_ns{};
// EDEN_NX2_TEXTURE_MEMO_VERIFY=1, read once: draws resolved from a memo are also resolved in full
// and compared.
bool TextureMemoVerify() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_TEXTURE_MEMO_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

// EDEN_NX2_DELTA_DESCRIPTORS_VERIFY=1: delta draws resolve their image descriptors as before and
// compare them with the kept ones.
const bool delta_descriptors_verify = [] {
    const char* const value = std::getenv("EDEN_NX2_DELTA_DESCRIPTORS_VERIFY");
    return value != nullptr && value[0] == '1';
}();

// Delta draws on the drawing thread, logged every 10 s with EDEN_NX2_GPU_PROFILE or the verify
// env.
struct DeltaDescriptorStats {
    u64 replayed{};
    u64 pushed{};
    u64 verified{};
    u64 mismatches{};
    std::chrono::steady_clock::time_point next_report{};
} delta_descriptor_stats;

void ReportDeltaDescriptors() {
    DeltaDescriptorStats& stats = delta_descriptor_stats;
    if (((stats.replayed + stats.pushed) & 1023) != 0 ||
        (!VideoCommon::Nx2Profile::Enabled() && !delta_descriptors_verify)) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < stats.next_report) {
        return;
    }
    if (stats.next_report != std::chrono::steady_clock::time_point{}) {
        LOG_INFO(Render_Vulkan,
                 "Delta descriptors in 10 s: {} replayed, {} pushed, {} verified, {} verify "
                 "mismatches",
                 stats.replayed, stats.pushed, stats.verified, stats.mismatches);
    }
    stats = {.next_report = now + std::chrono::seconds{10}};
}

// Texture memo counts, logged every 10 s with EDEN_NX2_GPU_PROFILE or the verify env.
std::atomic<u64> g_memo_hits{};
std::atomic<u64> g_memo_misses{};
std::atomic<u64> g_memo_mismatches{};

// Counters the drawing thread writes: a plain load and store, not a locked add.
void Count(std::atomic<u64>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void ReportTextureMemo() {
    if (!VideoCommon::Nx2Profile::Enabled() && !TextureMemoVerify()) {
        return;
    }
    using Clock = std::chrono::steady_clock;
    static Clock::time_point next_report{};
    const Clock::time_point now = Clock::now();
    if (now < next_report) {
        return;
    }
    const bool first = next_report == Clock::time_point{};
    next_report = now + std::chrono::seconds{10};
    const u64 hits = g_memo_hits.exchange(0, std::memory_order_relaxed);
    const u64 misses = g_memo_misses.exchange(0, std::memory_order_relaxed);
    if (!first && (hits != 0 || misses != 0)) {
        LOG_INFO(Render_Vulkan,
                 "Texture memo: {} draws resolved from the memo, {} in full in 10 s; {} verify "
                 "mismatches in total",
                 hits, misses, g_memo_mismatches.load(std::memory_order_relaxed));
    }
}
} // Anonymous namespace

std::chrono::nanoseconds GraphicsPipeline::TakeHoldTime() noexcept {
    return std::chrono::nanoseconds{hold_time_ns.exchange(0, std::memory_order::relaxed)};
}

std::pair<u64, u64> GraphicsPipeline::TakeTextureMemoCounts() noexcept {
    return {g_memo_hits.exchange(0, std::memory_order_relaxed),
            g_memo_misses.exchange(0, std::memory_order_relaxed)};
}

// TODO(crueter): This is the worst-formatted code I have EVER seen
GraphicsPipeline::GraphicsPipeline(
    Scheduler& scheduler_, BufferCache& buffer_cache_, TextureCache& texture_cache_,
    vk::PipelineCache& pipeline_cache_, PipelineBinaries* binaries_,
    VideoCore::ShaderNotify* shader_notify,
    const Device& device_, DescriptorPool& descriptor_pool,
    GuestDescriptorQueue& guest_descriptor_queue_, DescriptorBufferRing& descriptor_buffer_ring_,
    const PipelineBuildQueue& build_queue,
    PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
    const GraphicsPipelineCacheKey& key_, std::array<std::vector<u32>, NUM_STAGES> stages,
    const std::array<const Shader::Info*, NUM_STAGES>& infos,
    std::function<std::array<std::vector<u32>, NUM_STAGES>()> emit_stages,
    std::optional<MeshEmulationSetup> mesh_emulation_)
    : key{key_}, device{device_}, texture_cache{texture_cache_}, buffer_cache{buffer_cache_},
      pipeline_cache(pipeline_cache_), binaries{binaries_}, scheduler{scheduler_},
      guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, spirv{std::move(stages)},
      worker_only_build{build_queue.jobs && static_cast<bool>(emit_stages)},
      mesh_emulation{std::move(mesh_emulation_)} {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info* const info{infos[stage]};
        if (!info) {
            continue;
        }
        stage_infos[stage] = *info;
        enabled_uniform_buffer_masks[stage] = info->constant_buffer_mask;
        std::ranges::copy(info->constant_buffer_used_sizes, uniform_buffer_sizes[stage].begin());
        num_image_elements += Shader::NumDescriptors(info->texture_buffer_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_buffer_descriptors);
        num_textures += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_descriptors);
        num_descriptor_entries += NumDescriptorEntries(*info);
    }
    fragment_has_color0_output = stage_infos[NUM_STAGES - 1].stores_frag_color[0];
    // The guest leaves render target components the fragment shader doesn't output untouched,
    // while Vulkan writes undefined values to them (AMD writes garbage), so they are masked out.
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        color_output_mask |= u32{stage_infos[NUM_STAGES - 1].frag_color_components[rt]}
                             << (rt * 4);
    }
    const auto any_written = [](const auto& descriptors) {
        return std::ranges::any_of(descriptors, [](const auto& desc) { return desc.is_written; });
    };
    writes_storage = std::ranges::any_of(stage_infos, [&](const Shader::Info& info) {
        return info.stores_global_memory || any_written(info.storage_buffers_descriptors) ||
               any_written(info.image_descriptors) || any_written(info.image_buffer_descriptors);
    });
    texture_memo_eligible = std::ranges::all_of(stage_infos, [](const Shader::Info& info) {
        return info.texture_buffer_descriptors.empty() && info.image_buffer_descriptors.empty() &&
               info.image_descriptors.empty() &&
               std::ranges::all_of(info.texture_descriptors, [](const auto& desc) {
                   return desc.handle_program == ~0U && !desc.has_secondary;
               });
    });
    writes_images = std::ranges::any_of(stage_infos, [&](const Shader::Info& info) {
        return any_written(info.image_descriptors) || any_written(info.image_buffer_descriptors);
    });
    const auto block = [&](DeltaBlocker blocker) {
        if (delta_blocker == DeltaBlocker::None) {
            delta_blocker = blocker;
        }
    };
    if (writes_images) {
        block(DeltaBlocker::WritesImages);
    }
    if (IsMeshPipeline()) {
        block(DeltaBlocker::Mesh);
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info& info = stage_infos[stage];
        u32& handles = handle_constant_buffers[stage];
        auto& ranges = handle_ranges[stage];
        std::array<boost::container::small_vector<std::pair<u32, u32>, 8>,
                   Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers>
            words;
        const auto add_range = [&](u32 slot, u32 begin, u32 end) {
            handles |= 1U << slot;
            auto& range = ranges[slot];
            range = range.first == range.second
                        ? std::pair{begin, end}
                        : std::pair{(std::min)(range.first, begin), (std::max)(range.second, end)};
            words[slot].emplace_back(begin, end);
        };
        // The words read_handle reads. Handles computed at draw time neither block delta draws
        // nor take part: ComputedHandlesChanged evaluates them again for each delta draw, from
        // the constant buffers bound and the data in memory then.
        const auto add = [&](const auto& desc) {
            if (desc.handle_program != ~0U) {
                return;
            }
            // Each element's word on its own: strided arrays leave words between the handles.
            for (u32 index = 0; index < desc.count; ++index) {
                // Read as a storage buffer address: the texture is the null one (ConfigureImpl).
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(desc)>,
                                             Shader::TextureDescriptor>) {
                    if (VideoCommon::HandleIsStoragePointer(info, desc, index)) {
                        continue;
                    }
                }
                const u32 offset = desc.cbuf_offset + (index << desc.size_shift);
                add_range(desc.cbuf_index, offset, offset + 4);
            }
            if constexpr (requires { desc.has_secondary; }) {
                if (desc.has_secondary) {
                    add_range(desc.secondary_cbuf_index, desc.secondary_cbuf_offset,
                              desc.secondary_cbuf_offset + 4);
                }
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add(desc);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add(desc);
        }
        for (const auto& desc : info.texture_descriptors) {
            add(desc);
        }
        for (const auto& desc : info.image_descriptors) {
            add(desc);
        }
        for (u32 slot = 0; slot < words.size(); ++slot) {
            auto& list = words[slot];
            std::ranges::sort(list);
            const size_t first = handle_words.size();
            for (const auto& range : list) {
                if (handle_words.size() != first && range.first <= handle_words.back().second) {
                    handle_words.back().second = (std::max)(handle_words.back().second, range.second);
                } else {
                    handle_words.push_back(range);
                }
            }
            handle_word_index[stage][slot] = {static_cast<u16>(first),
                                              static_cast<u16>(handle_words.size() - first)};
        }
    }

    DescriptorLayoutBuilder builder{MakeBuilder(device, stage_infos, mesh_emulation)};
    // Mesh emulation binds one descriptor set to both the compute dispatch and the draw.
    uses_push_descriptor = !mesh_emulation && builder.CanUsePushDescriptor();
    uses_descriptor_buffer = !mesh_emulation && builder.CanUseDescriptorBuffer() &&
                             descriptor_buffer_ring.IsValid();
    // After every stage's descriptors: the output buffer, then with a task program its payload,
    // offset and map buffers.
    const u32 emulation_buffers{mesh_emulation ? (mesh_emulation->has_task ? 4U : 1U) : 0U};
    num_descriptor_entries += emulation_buffers;
    descriptor_set_layout =
        builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
    if (uses_descriptor_buffer) {
        descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
        if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
            LOG_WARNING(Render_Vulkan,
                        "Graphics pipeline {:016X} needs {} descriptor bytes per draw, falling back "
                        "to sets",
                        key.Hash(), descriptor_buffer_layout.size);
            uses_descriptor_buffer = false;
            descriptor_buffer_layout = {};
            descriptor_set_layout = builder.CreateDescriptorSetLayout(uses_push_descriptor);
        }
    }

    const VkDescriptorSetLayout set_layout{*descriptor_set_layout};
    pipeline_layout = builder.CreatePipelineLayout(set_layout);
    if (!uses_descriptor_buffer) {
        descriptor_update_template =
            builder.CreateTemplate(set_layout, *pipeline_layout, uses_push_descriptor);
        if (!uses_push_descriptor && mesh_emulation) {
            // The set also holds the emulation's buffers.
            std::array<Shader::Info, NUM_STAGES + 1> bank_infos{};
            std::ranges::copy(stage_infos, bank_infos.begin());
            bank_infos.back().storage_buffers_descriptors.push_back({
                .cbuf_index = 0,
                .cbuf_offset = 0,
                .count = emulation_buffers,
                .is_written = true,
            });
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, set_layout, bank_infos);
            if (mesh_emulation->has_task) {
                // The distribution pass: payload, offsets, map and arguments, plus its push
                // constants (Shader::Backend::SPIRV::MeshEmulationDistributionConstants).
                std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
                for (u32 binding = 0; binding < bindings.size(); ++binding) {
                    bindings[binding] = {
                        .binding = binding,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                        .pImmutableSamplers = nullptr,
                    };
                }
                distribution_set_layout = device.GetLogical().CreateDescriptorSetLayout({
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .bindingCount = static_cast<u32>(bindings.size()),
                    .pBindings = bindings.data(),
                });
                const VkDescriptorSetLayout distribution_set{*distribution_set_layout};
                const VkPushConstantRange range{
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .offset = 0,
                    .size = sizeof(Shader::Backend::SPIRV::MeshEmulationDistributionConstants),
                };
                distribution_layout = device.GetLogical().CreatePipelineLayout({
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .setLayoutCount = 1,
                    .pSetLayouts = &distribution_set,
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &range,
                });
                distribution_allocator = descriptor_pool.Allocator(
                    device, scheduler, distribution_set, DescriptorBankInfo{
                                                             .storage_buffers = 4,
                                                             .score = 4,
                                                         });
            }
        } else if (!uses_push_descriptor) {
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, set_layout, stage_infos);
        }
    }

    // Shader cache v2: the emission is the CPU stage of the build's job.
    std::function<void()> emit;
    if (build_queue.jobs && emit_stages) {
        emit = [this, shader_notify, emit_stages = std::exchange(emit_stages, {})] {
            try {
                spirv = emit_stages();
            } catch (const Shader::Exception& exception) {
                LOG_ERROR(Render_Vulkan, "Graphics pipeline {:016x}: {}", key.Hash(),
                          exception.what());
                if (shader_notify) {
                    shader_notify->MarkShaderComplete();
                }
                throw;
            } catch (...) {
                if (shader_notify) {
                    shader_notify->MarkShaderComplete();
                }
                throw;
            }
        };
    }
    auto func{[this, shader_notify, &render_pass_cache, pipeline_statistics,
                emit_stages = std::move(emit_stages)] {
        const auto begin = std::chrono::steady_clock::now();
        SCOPE_EXIT {
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
        };
        try {
            if (emit_stages) {
                spirv = emit_stages();
            }
            for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
                if (spirv[stage].empty()) {
                    continue;
                }
                spv_modules[stage] = BuildShader(device, spirv[stage]);
                if (device.HasDebuggingToolAttached()) {
                    const std::string name{
                        fmt::format("Shader {:016x}", key.unique_hashes[stage + 1])};
                    spv_modules[stage].SetObjectNameEXT(name.c_str());
                }
            }
            const VkRenderPass render_pass{render_pass_cache.Get(MakeRenderPassKey(key.state, device))};
            Validate();
            MakePipeline(render_pass);
        } catch (const Shader::Exception& exception) {
            LOG_ERROR(Render_Vulkan, "Graphics pipeline {:016x}: {}", key.Hash(), exception.what());
            throw;
        } catch (const vk::Exception& exception) {
            LOG_CRITICAL(Render_Vulkan,
                         "Graphics pipeline build failed: {} (stages: tcs={} tes={} geom={}, "
                         "patch control points {}, device max patch size {})",
                         exception.what(), spv_modules[1] ? 1 : 0, spv_modules[2] ? 1 : 0,
                         spv_modules[3] ? 1 : 0, key.state.patch_control_points_minus_one.Value() + 1,
                         device.GetMaxTessellationPatchSize());
            ReportPipelineBuildFailure(exception.GetResult());
            throw;
        }
        if (pipeline_statistics) {
            pipeline_statistics->Collect(device, *pipeline);
        }
        // Nothing reads the shader modules once the pipeline exists, and the driver keeps a copy
        // of each one's SPIR-V while it lives (title 1: ~1,500 pipelines, 2-5 modules each).
        // Its SPIR-V is kept compressed (about a tenth of it), so that an evicted pipeline is
        // built again without translating its shaders.
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            spv_modules[stage].reset();
            if (!spirv[stage].empty()) {
                kept_spirv[stage] = Common::Compression::CompressDataZSTD(
                    reinterpret_cast<const u8*>(spirv[stage].data()),
                    spirv[stage].size() * sizeof(u32), 1);
                kept_spirv[stage].shrink_to_fit();
                spirv[stage] = std::vector<u32>{};
            }
        }
        if (worker_only_build) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            static std::atomic<u32> logged{};
            if (elapsed > std::chrono::milliseconds{100} ||
                (elapsed > std::chrono::milliseconds{2} && logged.fetch_add(1) < 400)) {
                LOG_INFO(Render_Vulkan,
                         "Nx2Perf async_pipelines: {:016x} worker emission and driver build {:.1f} ms",
                         key.Hash(), std::chrono::duration<double, std::milli>(elapsed).count());
            }
        }
    }};
    std::array<bool, NUM_STAGES> present_stages{};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        present_stages[stage] = infos[stage] != nullptr;
    }
    configure_func = ConfigureFunc(present_stages, stage_infos);
    build.emplace(build_queue, std::move(func), std::move(emit));
}

u64 KeptGraphicsStages::Bytes() const noexcept {
    u64 bytes = sizeof(*this);
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        bytes += spirv[stage].size() + (infos[stage] ? sizeof(Shader::Info) : 0);
    }
    if (mesh_emulation) {
        bytes += (mesh_emulation->vertex_spirv.size() + mesh_emulation->distribution_spirv.size()) *
                 sizeof(u32);
    }
    return bytes;
}

KeptGraphicsStages GraphicsPipeline::TakeKeptStages() {
    KeptGraphicsStages kept{
        .spirv = std::move(kept_spirv),
        .mesh_emulation = mesh_emulation,
        .mapped_pointers = mapped_pointers,
    };
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!kept.spirv[stage].empty()) {
            kept.infos[stage] = std::make_unique<Shader::Info>(stage_infos[stage]);
        }
    }
    return kept;
}

void GraphicsPipeline::AddTransition(GraphicsPipeline* transition) {
    transition_keys.push_back(transition->key);
    transitions.push_back(transition);
}

template <typename Spec>
bool GraphicsPipeline::ConfigureImpl(bool is_indexed) {
    boost::container::small_vector<VideoCommon::ImageViewInOut, 160> views;
    boost::container::small_vector<VideoCommon::SamplerId, 96> samplers;
    boost::container::small_vector<ComputedHandle, 16> computed_handles;
    boost::container::small_vector<u32, 64> computed_values;
    views.reserve(num_image_elements);
    samplers.reserve(num_textures);

    namespace Profile = VideoCommon::Nx2Profile;
    {
        Profile::Scope scope{Profile::Zone::SyncDescriptors};
        texture_cache.SynchronizeDescriptors(false);
    }

    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);

    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};

    const bool use_memo{texture_memo_eligible && !RecordingSampledTextures()};
    const bool verify_memo{use_memo && TextureMemoVerify()};
    boost::container::small_vector<GPUVAddr, 32> handle_addresses;
    boost::container::small_vector<VideoCommon::TextureHandleRead, 32> handle_reads;
    boost::container::small_vector<VideoCommon::ImageViewInOut, 32> memo_views;
    boost::container::small_vector<VideoCommon::SamplerId, 32> memo_samplers;
    std::vector<VideoCommon::ImageId> memo_prepared;
    std::vector<VideoCommon::ImageId> full_prepared;
    bool memo_hit{};
    if (use_memo) {
        const auto collect_addresses{[&](size_t stage) LAMBDA_FORCEINLINE {
            const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
            for (const auto& desc : stage_infos[stage].texture_descriptors) {
                ASSERT(cbufs[desc.cbuf_index].enabled);
                for (u32 index = 0; index < desc.count; ++index) {
                    handle_addresses.push_back(cbufs[desc.cbuf_index].address + desc.cbuf_offset +
                                               (index << desc.size_shift));
                }
            }
        }};
        if constexpr (Spec::enabled_stages[0]) {
            collect_addresses(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            collect_addresses(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            collect_addresses(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            collect_addresses(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            collect_addresses(4);
        }
        views.resize(num_image_elements);
        samplers.resize(num_textures);
        if (verify_memo) {
            // The full resolve prepares the images: the memo only lists those it would prepare.
            texture_cache.LogPrepares(&memo_prepared, true);
        }
        memo_hit = texture_cache.ResolveMemoizedTextures(
            texture_memo, std::span(handle_addresses.data(), handle_addresses.size()),
            std::span(views.data(), views.size()), std::span(samplers.data(), samplers.size()));
        if (verify_memo) {
            texture_cache.LogPrepares(nullptr, false);
        }
        if (memo_hit && verify_memo) {
            memo_views.assign(views.begin(), views.end());
            memo_samplers.assign(samplers.begin(), samplers.end());
        }
        if (!memo_hit || verify_memo) {
            views.clear();
            samplers.clear();
        }
    }
    const bool resolve_in_full{!memo_hit || verify_memo};

    const auto config_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        const Shader::Info& info{stage_infos[stage]};
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        if constexpr (Spec::has_storage_buffers) {
            size_t ssbo_index{};
            for (const auto& desc : info.storage_buffers_descriptors) {
                ASSERT(desc.count == 1);
                buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                       desc.cbuf_offset, desc.is_written,
                                                       desc.indirect_offset);
                ++ssbo_index;
            }
        }
        const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
        const auto cbuf_address{[&cbufs](u32 cbuf) -> GPUVAddr {
            return cbuf < cbufs.size() && cbufs[cbuf].enabled ? cbufs[cbuf].address : GPUVAddr{};
        }};
        // Records each handle computed at draw time (ComputedHandlesChanged evaluates them again):
        // one entry per descriptor, and its elements' values in order.
        const auto note_computed{[&](const auto& desc, u32 element, u32 value) {
            using Desc = std::remove_cvref_t<decltype(desc)>;
            const auto [kind, first] = [&]() -> std::pair<ComputedHandleKind, const Desc*> {
                if constexpr (std::is_same_v<Desc, Shader::TextureBufferDescriptor>) {
                    return {TextureBuffer, info.texture_buffer_descriptors.data()};
                } else if constexpr (std::is_same_v<Desc, Shader::ImageBufferDescriptor>) {
                    return {ImageBuffer, info.image_buffer_descriptors.data()};
                } else if constexpr (std::is_same_v<Desc, Shader::TextureDescriptor>) {
                    return {Texture, info.texture_descriptors.data()};
                } else {
                    return {Image, info.image_descriptors.data()};
                }
            }();
            if (element == 0) {
                computed_handles.push_back({static_cast<u8>(stage), kind,
                                            static_cast<u16>(&desc - first), 0,
                                            static_cast<u32>(computed_values.size())});
            }
            ++computed_handles.back().count;
            computed_values.push_back(value);
        }};
        const auto read_handle{[&](const auto& desc, u32 index) {
            if (desc.handle_program != ~0U) {
                // Read by the shader from a storage buffer: computed for this draw.
                const u32 value{VideoCommon::EvaluateTextureHandle(
                    info, desc.handle_program, *gpu_memory, cbuf_address,
                    device.GetStorageBufferAlignment(), index)};
                note_computed(desc, index, value);
                return TexturePair(value, via_header_index);
            }
            ASSERT(cbufs[desc.cbuf_index].enabled);
            const u32 index_offset{index << desc.size_shift};
            const u32 offset{desc.cbuf_offset + index_offset};
            const GPUVAddr addr{cbufs[desc.cbuf_index].address + offset};
            if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                          std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
                if (desc.has_secondary) {
                    ASSERT(cbufs[desc.secondary_cbuf_index].enabled);
                    const u32 second_offset{desc.secondary_cbuf_offset};
                    const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].address +
                                                 second_offset};
                    const u32 lhs_raw{gpu_memory->Read<u32>(addr) << desc.shift_left};
                    const u32 rhs_raw{gpu_memory->Read<u32>(separate_addr)
                                      << desc.secondary_shift_left};
                    const u32 raw{lhs_raw | rhs_raw};
                    return TexturePair(raw, via_header_index);
                }
            }
            return TexturePair(gpu_memory->Read<u32>(addr), via_header_index);
        }};
        const auto add_image{[&](const auto& desc, bool blacklist) LAMBDA_FORCEINLINE {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{read_handle(desc, index)};
                views.push_back({
                    .index = handle.first,
                    .blacklist = blacklist,
                    .id = {}
                });
            }
        }};
        if constexpr (Spec::has_texture_buffers) {
            for (const auto& desc : info.texture_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        if constexpr (Spec::has_image_buffers) {
            for (const auto& desc : info.image_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        if (resolve_in_full) {
            u32 texture{};
            for (const auto& desc : info.texture_descriptors) {
                if (desc.count > 1 && desc.handle_program != ~0U) {
                    // A table of handles the shader indexes (one shadow map per light, say), read
                    // in one go; elements repeating a sampler share its lookup. A pipeline with
                    // handle programs never takes the texture memo (texture_memo_eligible).
                    const size_t first{views.size()};
                    u32 last_sampler_index{~0U};
                    VideoCommon::SamplerId sampler{};
                    VideoCommon::EvaluateTextureHandles(
                        info, desc.handle_program, *gpu_memory, cbuf_address,
                        device.GetStorageBufferAlignment(), desc.count, [&](u32 element, u32 raw) {
                            note_computed(desc, element, raw);
                            const auto handle{TexturePair(raw, via_header_index)};
                            views.push_back({handle.first});
                            if (handle.second != last_sampler_index) {
                                sampler = texture_cache.GetSamplerId(handle.second, false);
                                last_sampler_index = handle.second;
                            }
                            samplers.push_back(sampler);
                        });
                    if (VideoCommon::HandleTableNotes::Wanted()) {
                        std::vector<u32> tics(desc.count);
                        for (u32 index = 0; index < desc.count; ++index) {
                            tics[index] = views[first + index].index;
                        }
                        VideoCommon::HandleTableNotes::Note(key.unique_hashes[stage + 1], texture,
                                                            tics);
                    }
                    ++texture;
                    continue;
                }
                for (u32 index = 0; index < desc.count; ++index) {
                    const auto handle{VideoCommon::HandleIsStoragePointer(info, desc, index)
                                          ? VideoCommon::NULL_TEXTURE_HANDLE
                                          : read_handle(desc, index)};
                    views.push_back({handle.first});

                    VideoCommon::SamplerId sampler{texture_cache.GetSamplerId(handle.second, false)};
                    samplers.push_back(sampler);
                    if (use_memo) {
                        handle_reads.push_back(
                            {handle_addresses[handle_reads.size()],
                             via_header_index ? handle.first : handle.first | (handle.second << 20)});
                    }
                }
                ++texture;
            }
        }
        if constexpr (Spec::has_images) {
            for (const auto& desc : info.image_descriptors) {
                add_image(desc, desc.is_written);
            }
        }

        return true;
    }};
    {
        Profile::Scope scope{Profile::Zone::ReadHandles};
        if constexpr (Spec::enabled_stages[0]) {
            config_stage(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            config_stage(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            config_stage(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            config_stage(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            config_stage(4);
        }
    }
    ASSERT(views.size() == num_image_elements);
    ASSERT(samplers.size() == num_textures);
    if (resolve_in_full) {
        Profile::Scope scope{Profile::Zone::FillImageViews};
        if (memo_hit && verify_memo) {
            texture_cache.LogPrepares(&full_prepared, false);
        }
        texture_cache.FillImageViews(std::span(views.data(), views.size()), false,
                                     Spec::has_images);
        if (memo_hit && verify_memo) {
            texture_cache.LogPrepares(nullptr, false);
            // The dry memo pass skipped PrepareImage, and guest memory can change while the
            // full resolve runs: drop the memo hit when preparing deleted images or the inputs
            // moved since ResolveMemoizedTextures.
            if (!texture_cache.TextureMemoStillHolds(
                    texture_memo, std::span(handle_addresses.data(), handle_addresses.size()))) {
                texture_memo.valid = false;
                memo_hit = false;
            }
        }
    }
    if (use_memo) {
        if (!memo_hit) {
            Count(g_memo_misses);
            // A pipeline whose draws keep changing textures stops paying for a memo nobody uses.
            if (++texture_memo.misses <= 8 || (texture_memo.misses & 15) == 0) {
                texture_cache.MemoizeTextures(texture_memo,
                                              std::span(handle_reads.data(), handle_reads.size()),
                                              std::span(views.data(), views.size()),
                                              std::span(samplers.data(), samplers.size()));
            }
        } else {
            texture_memo.misses = 0;
            Count(g_memo_hits);
        }
        if (memo_hit && verify_memo) {
            // Bindings alone miss what the paths do to the images (#904 prepared the null one).
            for (auto* const prepared : {&memo_prepared, &full_prepared}) {
                std::ranges::sort(*prepared);
                prepared->erase(std::unique(prepared->begin(), prepared->end()), prepared->end());
            }
            const bool same =
                std::ranges::equal(views, memo_views,
                                   [](const auto& lhs, const auto& rhs) {
                                       return lhs.index == rhs.index && lhs.id == rhs.id;
                                   }) &&
                std::ranges::equal(samplers, memo_samplers) && memo_prepared == full_prepared;
            if (!same) {
                Count(g_memo_mismatches);
                LOG_ERROR(Render_Vulkan,
                          "Texture memo: pipeline {:016x} resolved other views, samplers or "
                          "prepared images than its memo",
                          key.Hash());
            }
        }
        ReportTextureMemo();
    }
    if (RecordingSampledTextures() || DrawReuseProbeEnabled()) {
        sampled_images.clear();
        size_t view_index{};
        size_t sampler_index{};
        for (size_t stage = 0; stage < Spec::enabled_stages.size(); ++stage) {
            if (!Spec::enabled_stages[stage]) {
                continue;
            }
            const Shader::Info& info{stage_infos[stage]};
            if constexpr (Spec::has_texture_buffers) {
                view_index += Shader::NumDescriptors(info.texture_buffer_descriptors);
            }
            if constexpr (Spec::has_image_buffers) {
                view_index += Shader::NumDescriptors(info.image_buffer_descriptors);
            }
            const u32 textures = Shader::NumDescriptors(info.texture_descriptors);
            for (u32 texture = 0; texture < textures; ++texture, ++view_index, ++sampler_index) {
                sampled_images.push_back({static_cast<u32>(stage), views[view_index].index,
                                          views[view_index].id, samplers[sampler_index]});
            }
            if constexpr (Spec::has_images) {
                view_index += Shader::NumDescriptors(info.image_descriptors);
            }
        }
    }
    const bool captured_delta =
        std::exchange(capture_delta, false) && delta_blocker == DeltaBlocker::None;
    if (captured_delta) {
        const auto same_view = [](const VideoCommon::ImageViewInOut& lhs,
                                  const VideoCommon::ImageViewInOut& rhs) {
            return lhs.index == rhs.index && lhs.id == rhs.id;
        };
        if (std::exchange(verify_delta, false)) {
            delta_mismatch = !delta_views_valid ||
                             !std::ranges::equal(views, delta_views, same_view) ||
                             !std::ranges::equal(samplers, delta_samplers);
        }
        delta_views.assign(views.begin(), views.end());
        delta_samplers.assign(samplers.begin(), samplers.end());
        delta_computed_handles.assign(computed_handles.begin(), computed_handles.end());
        delta_computed_values.assign(computed_values.begin(), computed_values.end());
        configured_stages = Spec::enabled_stages;
        delta_views_valid = true;
        kept_images_valid = false;
    }
    if (captured_delta) {
        kept_image_entries.clear();
    }

    if constexpr (Spec::has_texture_buffers || Spec::has_image_buffers) {
        VideoCommon::ImageViewInOut* texture_buffer_it{views.data()};
        const auto bind_stage_info{[&](size_t stage) LAMBDA_FORCEINLINE {
            size_t index{};
            const auto add_buffer{[&](const auto& desc) {
                constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
                for (u32 i = 0; i < desc.count; ++i) {
                    bool is_written{false};
                    if constexpr (is_image) {
                        is_written = desc.is_written;
                    }
                    ImageView& image_view{texture_cache.GetImageView(texture_buffer_it->id)};
                    PixelFormat format{image_view.format};
                    if constexpr (is_image) {
                        if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                            format = *explicit_format;
                        }
                    }
                    buffer_cache.BindGraphicsTextureBuffer(stage, index, image_view.GpuAddr(),
                                                           image_view.BufferSize(), format,
                                                           is_written, is_image);
                    ++index;
                    ++texture_buffer_it;
                }
            }};
            buffer_cache.UnbindGraphicsTextureBuffers(stage);

            const Shader::Info& info{stage_infos[stage]};
            if constexpr (Spec::has_texture_buffers) {
                for (const auto& desc : info.texture_buffer_descriptors) {
                    add_buffer(desc);
                }
            }
            if constexpr (Spec::has_image_buffers) {
                for (const auto& desc : info.image_buffer_descriptors) {
                    add_buffer(desc);
                }
            }
            texture_buffer_it += Shader::NumDescriptors(info.texture_descriptors);
            if constexpr (Spec::has_images) {
                texture_buffer_it += Shader::NumDescriptors(info.image_descriptors);
            }
        }};
        Profile::Scope scope{Profile::Zone::TextureBuffers};
        if constexpr (Spec::enabled_stages[0]) {
            bind_stage_info(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            bind_stage_info(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            bind_stage_info(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            bind_stage_info(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            bind_stage_info(4);
        }
    } else {
        if constexpr (Spec::enabled_stages[0]) {
            buffer_cache.UnbindGraphicsTextureBuffers(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            buffer_cache.UnbindGraphicsTextureBuffers(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            buffer_cache.UnbindGraphicsTextureBuffers(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            buffer_cache.UnbindGraphicsTextureBuffers(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            buffer_cache.UnbindGraphicsTextureBuffers(4);
        }
    }

    if (regs.transform_feedback_enabled != 0) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }

    {
        Profile::Scope scope{Profile::Zone::UpdateBuffers};
        buffer_cache.UpdateGraphicsBuffers(is_indexed);
        buffer_cache.BindHostGeometryBuffers(is_indexed);
    }

    {
        Profile::Scope scope{Profile::Zone::DescriptorAcquire};
        guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);
    }

    RescalingPushConstant rescaling;
    RenderAreaPushConstant render_area;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    const auto prepare_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        {
            Profile::Scope scope{Profile::Zone::BindStageBuffers};
            buffer_cache.BindHostStageBuffers(stage);
        }
        {
            Profile::Scope scope{Profile::Zone::PushDescriptors};
            const size_t first_entry = guest_descriptor_queue.UpdateCount();
            PushImageDescriptors(texture_cache, guest_descriptor_queue, stage_infos[stage],
                                 rescaling, samplers_it, views_it);
            if (captured_delta) {
                KeepImageDescriptors(stage, first_entry);
            }
        }
        const auto& info{stage_infos[stage]};
        if (info.uses_render_area) {
            render_area.uses_render_area = true;
            // Switch 2 titles may never program the surface clip; use colour target 0 then.
            const u32 clip_width = regs.surface_clip.width, clip_height = regs.surface_clip.height;
            const bool clip_set = clip_width != 0 && clip_height != 0;
            render_area.words = {static_cast<float>(clip_set ? clip_width : u32{regs.rt[0].width}),
                                 static_cast<float>(clip_set ? clip_height : u32{regs.rt[0].height})};
        }
    }};
    if constexpr (Spec::enabled_stages[0]) {
        prepare_stage(0);
    }
    if constexpr (Spec::enabled_stages[1]) {
        prepare_stage(1);
    }
    if constexpr (Spec::enabled_stages[2]) {
        prepare_stage(2);
    }
    if constexpr (Spec::enabled_stages[3]) {
        prepare_stage(3);
    }
    if constexpr (Spec::enabled_stages[4]) {
        prepare_stage(4);
    }
    if (captured_delta) {
        kept_rescaling = rescaling.Data();
        kept_scale_epoch = texture_cache.ScaleEpoch();
        kept_images_valid = true;
    }
    if (mesh_emulation) {
        if (!mesh_output_buffer) {
            return false;
        }
        guest_descriptor_queue.AddBuffer(mesh_output_buffer, mesh_output_offset, mesh_output_size);
        if (mesh_emulation->has_task) {
            for (size_t index = 0; index < 3; ++index) { // payload, offsets, arguments
                const MeshEmulationBuffer& task_buffer{mesh_task_buffers[index]};
                if (!task_buffer.buffer) {
                    return false;
                }
                guest_descriptor_queue.AddBuffer(task_buffer.buffer, task_buffer.offset,
                                                 task_buffer.size);
            }
        }
    }
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    {
        Profile::Scope scope{Profile::Zone::RenderTargets};
        texture_cache.UpdateRenderTargets(false);
        const std::span<const VideoCommon::ImageViewInOut> view_span{views.data(), views.size()};
        texture_cache.CheckFeedbackLoop(view_span);
        if (scheduler.HasFragmentBarrier() &&
            (scheduler.RenderPassHasStores() || texture_cache.SamplesRenderTarget(view_span))) {
            scheduler.RequestOutsideRenderPassOperationContext();
        }
    }
    if (IsBuilt() && (!pipeline || build->Failed())) {
        return false;
    }
    Profile::Scope scope{Profile::Zone::ConfigureDraw};
    return ConfigureDraw(rescaling, render_area);
}

bool GraphicsPipeline::DeltaImagesChange() const noexcept {
    return texture_cache.SampledViewsChange(delta_views);
}

bool GraphicsPipeline::ComputedHandlesChanged() const {
    for (const ComputedHandle& handle : delta_computed_handles) {
        const Shader::Info& info{stage_infos[handle.stage]};
        const auto& cbufs{maxwell3d->state.shader_stages[handle.stage].const_buffers};
        u32 program{};
        switch (handle.kind) {
        case TextureBuffer:
            program = info.texture_buffer_descriptors[handle.descriptor].handle_program;
            break;
        case ImageBuffer:
            program = info.image_buffer_descriptors[handle.descriptor].handle_program;
            break;
        case Texture:
            program = info.texture_descriptors[handle.descriptor].handle_program;
            break;
        default:
            program = info.image_descriptors[handle.descriptor].handle_program;
            break;
        }
        // Each element as ConfigureImpl computes it (EvaluateTextureHandles gives the values
        // EvaluateTextureHandle does, reading the words the elements share once).
        bool changed{};
        VideoCommon::EvaluateTextureHandles(
            info, program, *gpu_memory,
            [&](u32 cbuf) -> GPUVAddr {
                return cbuf < cbufs.size() && cbufs[cbuf].enabled ? cbufs[cbuf].address
                                                                  : GPUVAddr{};
            },
            device.GetStorageBufferAlignment(), handle.count, [&](u32 element, u32 value) {
                changed |= value != delta_computed_values[handle.first_value + element];
            });
        if (changed) {
            return true;
        }
    }
    return false;
}

bool GraphicsPipeline::ConfigureDelta(bool is_indexed) {
    namespace Profile = VideoCommon::Nx2Profile;
    // ConfigureImpl without reading handles and resolving image views, samplers and texture
    // buffers: the buffers are resolved, synchronized and bound as it does.
    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);
    {
        Profile::Scope scope{Profile::Zone::ReadHandles};
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            if (!configured_stages[stage]) {
                continue;
            }
            buffer_cache.UnbindGraphicsStorageBuffers(stage);
            size_t ssbo_index{};
            for (const auto& desc : stage_infos[stage].storage_buffers_descriptors) {
                ASSERT(desc.count == 1);
                buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                       desc.cbuf_offset, desc.is_written,
                                                       desc.indirect_offset);
                ++ssbo_index;
            }
        }
    }
    {
        Profile::Scope scope{Profile::Zone::UpdateBuffers};
        buffer_cache.UpdateGraphicsBuffers(is_indexed);
        buffer_cache.BindHostGeometryBuffers(is_indexed);
    }
    {
        Profile::Scope scope{Profile::Zone::DescriptorAcquire};
        guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);
    }
    const auto& regs{maxwell3d->regs};
    RescalingPushConstant rescaling;
    RenderAreaPushConstant render_area;
    const VideoCommon::SamplerId* samplers_it{delta_samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{delta_views.data()};
    // Push the entries kept for delta_views instead of resolving every view and sampler again;
    // with EDEN_NX2_DELTA_DESCRIPTORS_VERIFY resolve them and compare.
    const bool kept_usable = kept_images_valid && kept_scale_epoch == texture_cache.ScaleEpoch();
    const bool replay_images = kept_usable && !delta_descriptors_verify;
    struct Kept {
        std::vector<DescriptorUpdateEntry> entries;
        std::array<std::pair<u32, u32>, NUM_STAGES> ranges;
        std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> rescaling;
    };
    std::optional<Kept> verify_kept;
    if (!replay_images) {
        if (kept_usable) {
            verify_kept.emplace(std::move(kept_image_entries), kept_image_ranges, kept_rescaling);
        }
        kept_image_entries.clear();
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!configured_stages[stage]) {
            continue;
        }
        {
            Profile::Scope scope{Profile::Zone::BindStageBuffers};
            buffer_cache.BindHostStageBuffers(stage);
        }
        if (replay_images) {
            Profile::Scope scope{Profile::Zone::PushDescriptors};
            const auto [first, count] = kept_image_ranges[stage];
            guest_descriptor_queue.AddEntries({kept_image_entries.data() + first, count});
        } else {
            Profile::Scope scope{Profile::Zone::PushDescriptors};
            const size_t first_entry = guest_descriptor_queue.UpdateCount();
            PushImageDescriptors(texture_cache, guest_descriptor_queue, stage_infos[stage],
                                 rescaling, samplers_it, views_it);
            KeepImageDescriptors(stage, first_entry);
        }
        if (stage_infos[stage].uses_render_area) {
            render_area.uses_render_area = true;
            const u32 clip_width = regs.surface_clip.width, clip_height = regs.surface_clip.height;
            const bool clip_set = clip_width != 0 && clip_height != 0;
            render_area.words = {static_cast<float>(clip_set ? clip_width : u32{regs.rt[0].width}),
                                 static_cast<float>(clip_set ? clip_height : u32{regs.rt[0].height})};
        }
    }
    if (replay_images) {
        rescaling.Assign(kept_rescaling);
        ++delta_descriptor_stats.replayed;
    } else {
        kept_rescaling = rescaling.Data();
        kept_scale_epoch = texture_cache.ScaleEpoch();
        kept_images_valid = true;
        ++delta_descriptor_stats.pushed;
        if (verify_kept) {
            ++delta_descriptor_stats.verified;
            const auto same_image = [](const DescriptorUpdateEntry& lhs,
                                       const DescriptorUpdateEntry& rhs) {
                return lhs.image.sampler == rhs.image.sampler &&
                       lhs.image.imageView == rhs.image.imageView &&
                       lhs.image.imageLayout == rhs.image.imageLayout;
            };
            if ((verify_kept->ranges != kept_image_ranges ||
                 verify_kept->rescaling != kept_rescaling ||
                 !std::ranges::equal(verify_kept->entries, kept_image_entries, same_image)) &&
                delta_descriptor_stats.mismatches++ < 16) {
                LOG_ERROR(Render_Vulkan,
                          "Delta descriptors verify: pipeline {:016x} resolved other image "
                          "descriptors or rescaling words than the previous draw",
                          key.Hash());
            }
        }
    }
    ReportDeltaDescriptors();
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    {
        Profile::Scope scope{Profile::Zone::RenderTargets};
        texture_cache.UpdateRenderTargets(false);
        const std::span<const VideoCommon::ImageViewInOut> view_span{delta_views};
        texture_cache.CheckFeedbackLoop(view_span);
        if (scheduler.HasFragmentBarrier() &&
            (scheduler.RenderPassHasStores() || texture_cache.SamplesRenderTarget(view_span))) {
            scheduler.RequestOutsideRenderPassOperationContext();
        }
    }
    if (IsBuilt() && (!pipeline || build->Failed())) {
        return false;
    }
    Profile::Scope scope{Profile::Zone::ConfigureDraw};
    return ConfigureDraw(rescaling, render_area);
}

void GraphicsPipeline::KeepImageDescriptors(size_t stage, size_t first_entry) {
    const DescriptorUpdateEntry* const entries = guest_descriptor_queue.UpdateData();
    const size_t end_entry = guest_descriptor_queue.UpdateCount();
    kept_image_ranges[stage] = {static_cast<u32>(kept_image_entries.size()),
                                static_cast<u32>(end_entry - first_entry)};
    kept_image_entries.insert(kept_image_entries.end(), entries + first_entry,
                              entries + end_entry);
}

/// Writes the descriptors the stages queued and the push constants, then records the draw's state
/// and bindings. Returns false, and the draw is skipped, when the queued descriptor count does not
/// match the pipeline or no descriptor memory can be reserved.
bool GraphicsPipeline::ConfigureDraw(const RescalingPushConstant& rescaling,
                                     const RenderAreaPushConstant& render_area) {
    if (guest_descriptor_queue.UpdateCount() != num_descriptor_entries) {
        LOG_CRITICAL(Render_Vulkan, "Graphics descriptor count mismatch: expected {}, got {}",
                     num_descriptor_entries, guest_descriptor_queue.UpdateCount());
        return false;
    }
    const void* const descriptor_data{guest_descriptor_queue.UpdateData()};

    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (descriptor_set_layout && uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        const bool reuse_allocation =
            last_descriptor_buffer_generation == descriptor_buffer_ring.CurrentGeneration() &&
            last_descriptor_payload.size() == num_descriptor_entries &&
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) == 0;
        if (reuse_allocation) {
            descriptor_buffer_offset = last_descriptor_buffer_offset;
            descriptor_buffer_chunk = last_descriptor_buffer_chunk;
            descriptor_buffer_ring.TouchFrame(scheduler);
        } else {
            const DescriptorBufferRing::Allocation alloc{
                descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
            if (!alloc.host) {
                LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping draw");
                return false;
            }
            WriteDescriptorBuffer(device, descriptor_buffer_layout, entries, alloc.host);
            descriptor_buffer_offset = alloc.offset;
            descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_offset = alloc.offset;
            last_descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_generation = alloc.generation;
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }

    if (mesh_emulation) {
        // The mesh program runs as a compute dispatch outside the render pass and writes the
        // triangles the draw then reads; both use the descriptor set committed here.
        scheduler.RequestOutsideRenderPassOperationContext();
        if (!IsBuilt()) {
            scheduler.Record([this](vk::CommandBuffer) {
                const VideoCommon::Nx2Profile::Scope scope{
                    VideoCommon::Nx2Profile::Zone::PipelineWait};
                build->Wait(!worker_only_build);
            });
        }
        scheduler.Record([this, descriptor_data, groups = mesh_groups, capacity = mesh_capacity,
                          task_buffers = mesh_task_buffers,
                          rescaling_data = rescaling.Data(), is_rescaling = texture_cache.IsRescaling(),
                          uses_render_area = render_area.uses_render_area,
                          render_area_data = render_area.words](vk::CommandBuffer cmdbuf) {
            mesh_descriptor_set = VK_NULL_HANDLE;
            const bool has_task{mesh_emulation->has_task};
            if (!pipeline || !mesh_compute_pipeline || build->Failed() || groups == 0 ||
                (has_task && (!task_compute_pipeline || !distribution_pipeline))) {
                return;
            }
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            device.GetLogical().UpdateDescriptorSet(descriptor_set, *descriptor_update_template,
                                                    descriptor_data);
            mesh_descriptor_set = descriptor_set;
            // The task and mesh programs read the draw's push constants and descriptors.
            const auto bind_program_state = [&] {
                constexpr VkShaderStageFlags push_stages{VK_SHADER_STAGE_ALL_GRAPHICS |
                                                         VK_SHADER_STAGE_COMPUTE_BIT};
                cmdbuf.PushConstants(*pipeline_layout, push_stages, RESCALING_LAYOUT_WORDS_OFFSET,
                                     sizeof(rescaling_data), rescaling_data.data());
                const f32 config_down_factor{Settings::values.resolution_info.down_factor};
                const f32 down_factor{is_rescaling ? config_down_factor : 1.0f};
                cmdbuf.PushConstants(*pipeline_layout, push_stages,
                                     RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof(down_factor),
                                     &down_factor);
                if (uses_render_area) {
                    cmdbuf.PushConstants(*pipeline_layout, push_stages, RENDERAREA_LAYOUT_OFFSET,
                                         sizeof(render_area_data), &render_area_data);
                }
                cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0,
                                          descriptor_set, nullptr);
            };
            const auto memory_barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access,
                                            VkPipelineStageFlags src_stages,
                                            VkPipelineStageFlags dst_stages) {
                const VkMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                };
                cmdbuf.PipelineBarrier(src_stages, dst_stages, 0, barrier);
            };
            // The scheduler orders earlier compute writes only before the next render pass
            // (COMPUTE -> ALL_GRAPHICS); these dispatches run before it, and the programs read
            // what the game's compute passes wrote (foliage instance lists, for one).
            memory_barrier(VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                           vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            if (!has_task) {
                cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *mesh_compute_pipeline);
                bind_program_state();
                cmdbuf.Dispatch(groups, 1, 1);
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
                return;
            }
            // 1. One task workgroup per draw group writes its payload (mesh workgroup count at
            //    word 1).
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *task_compute_pipeline);
            bind_program_state();
            cmdbuf.Dispatch(groups, 1, 1);
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // 2. Distribution: each task's first mesh workgroup, the draw's place in the output
            //    arena, and the mesh dispatch and draw arguments.
            const VkDescriptorSet distribution_set{distribution_allocator.Commit()};
            std::array<VkDescriptorBufferInfo, 4> buffer_infos{};
            std::array<VkWriteDescriptorSet, 4> writes{};
            for (u32 binding = 0; binding < 4; ++binding) {
                buffer_infos[binding] = {
                    .buffer = task_buffers[binding].buffer,
                    .offset = task_buffers[binding].offset,
                    .range = task_buffers[binding].size,
                };
                writes[binding] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = distribution_set,
                    .dstBinding = binding,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &buffer_infos[binding],
                    .pTexelBufferView = nullptr,
                };
            }
            device.GetLogical().UpdateDescriptorSets(writes, {});
            const Shader::Backend::SPIRV::MeshEmulationDistributionConstants constants{
                .tasks = groups,
                .payload_words = mesh_emulation->payload_stride,
                .capacity_words = capacity,
                .block_words = mesh_emulation->block_words,
                .vertices_per_group = MeshEmulationVerticesPerGroup(),
                .max_groups = device.GetMaxComputeWorkGroupCount()[0],
            };
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *distribution_pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *distribution_layout, 0,
                                      distribution_set, nullptr);
            cmdbuf.PushConstants(*distribution_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(constants), &constants);
            cmdbuf.Dispatch(1, 1, 1);
            // The host reads the arena counter once the submission finishes.
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                               VK_ACCESS_HOST_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT);
            // 3. The mesh workgroups, as many as the tasks asked for (up to the arena's end).
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *mesh_compute_pipeline);
            bind_program_state();
            cmdbuf.DispatchIndirect(task_buffers[2].buffer, task_buffers[2].offset);
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
        });
    }
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    if (writes_storage) {
        scheduler.NoteRenderPassStores();
    }
    if (!IsBuilt()) {
        // Wait for the pipeline to be built
        scheduler.Record([this](vk::CommandBuffer) {
            const VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::PipelineWait};
            const auto begin = std::chrono::steady_clock::now();
            build->Wait(!worker_only_build);
            hold_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - begin)
                                       .count(),
                                   std::memory_order::relaxed);
        });
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const bool update_rescaling{scheduler.UpdateRescaling(is_rescaling, IsMeshPipeline())};
    const bool bind_pipeline{scheduler.UpdateGraphicsPipeline(this)};
    const bool bind_descriptor_buffer{
        descriptor_set_layout && uses_descriptor_buffer &&
        scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    // Log graphics pipeline binding
    if (bind_pipeline && GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string pipeline_info = fmt::format("hash={:#016x}", key.Hash());
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(false, pipeline_info);
    }

    bool update_descriptors = true;
    if (descriptor_set_layout && !uses_push_descriptor && !uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        update_descriptors =
            bind_pipeline || last_descriptor_payload.size() != num_descriptor_entries ||
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) != 0;
        if (update_descriptors) {
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }
    scheduler.Record([this, descriptor_data, bind_pipeline, update_descriptors,
                      descriptor_buffer_offset, descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data(), is_rescaling, update_rescaling,
                      uses_render_area = render_area.uses_render_area,
                      render_area_data = render_area.words](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (bind_pipeline) {
            scheduler.SetGraphicsPipelineMissing(!pipeline || build->Failed());
            if (!pipeline || build->Failed()) {
                return;
            }
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
        }
        const VkShaderStageFlags push_stages =
            mesh_emulation ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT
            : IsMeshPipeline()
                ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |
                      VK_SHADER_STAGE_MESH_BIT_EXT
                : VK_SHADER_STAGE_ALL_GRAPHICS;
        cmdbuf.PushConstants(*pipeline_layout, push_stages, RESCALING_LAYOUT_WORDS_OFFSET,
                             sizeof(rescaling_data), rescaling_data.data());
        if (update_rescaling || mesh_emulation) {
            const f32 config_down_factor{Settings::values.resolution_info.down_factor};
            const f32 scale_down_factor{is_rescaling ? config_down_factor : 1.0f};
            cmdbuf.PushConstants(*pipeline_layout, push_stages,
                                 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof(scale_down_factor),
                                 &scale_down_factor);
        }
        if (uses_render_area) {
            cmdbuf.PushConstants(*pipeline_layout, push_stages, RENDERAREA_LAYOUT_OFFSET,
                                 sizeof(render_area_data), &render_area_data);
        }
        if (!descriptor_set_layout) {
            return;
        }
        if (mesh_emulation) {
            // The set the emulation's dispatch committed (see above).
            if (mesh_descriptor_set) {
                cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                          mesh_descriptor_set, nullptr);
            }
            return;
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else if (update_descriptors) {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

/// Creates the Vulkan pipeline for `render_pass`. A mesh pipeline without VK_EXT_mesh_shader is
/// built from the mesh emulation's vertex shader as a point, line or triangle list.
void GraphicsPipeline::MakePipeline(VkRenderPass render_pass) {
    if (IsMeshPipeline() && !device.IsExtMeshShaderSupported() && !mesh_emulation) {
        return;
    }
    if (mesh_emulation) {
        // The mesh program (stage 0, or mesh_stage after a task program in stage 0) is built as
        // a compute shader, and so is the task program; the draw uses the vertex shader that
        // reads the mesh program's output.
        // Votes, ballots and shuffles span the guest's 32-thread warp, which a driver may split
        // in compute shaders too (RDNA runs them wave64).
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT compute_subgroup_size_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .pNext = nullptr,
            .requiredSubgroupSize = GuestWarpSize,
        };
        // Their binaries are stored under the graphics key, one key per program
        const auto make_compute = [&](VkShaderModule module, VkPipelineLayout layout,
                                      const Shader::Info* info, u64 program) {
            const bool uses_warp_ops{info && (info->uses_subgroup_vote ||
                                              info->uses_subgroup_invocation_id ||
                                              info->uses_subgroup_shuffles ||
                                              info->uses_subgroup_mask ||
                                              info->uses_quad_shuffles || info->uses_fswzadd)};
            const bool force_warp_size{uses_warp_ops && device.IsExtSubgroupSizeControlSupported() &&
                                       device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT)};
            const VkComputePipelineCreateInfo compute_ci{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = force_warp_size ? &compute_subgroup_size_ci : nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = module,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
                },
                .layout = layout,
                .basePipelineHandle = nullptr,
                .basePipelineIndex = 0,
            };
            return binaries ? binaries->CreateComputePipeline(
                                  compute_ci, *pipeline_cache,
                                  GraphicsRecordKey(key) ^ (program * 0x9e3779b97f4a7c15ULL))
                            : device.GetLogical().CreateComputePipeline(compute_ci,
                                                                        *pipeline_cache);
        };
        mesh_vertex_module = BuildShader(device, mesh_emulation->vertex_spirv);
        mesh_compute_pipeline = make_compute(*spv_modules[mesh_emulation->mesh_stage],
                                             *pipeline_layout,
                                             &stage_infos[mesh_emulation->mesh_stage], 1);
        if (mesh_emulation->has_task) {
            task_compute_pipeline =
                make_compute(*spv_modules[0], *pipeline_layout, &stage_infos[0], 2);
            distribution_module = BuildShader(device, mesh_emulation->distribution_spirv);
            distribution_pipeline =
                make_compute(*distribution_module, *distribution_layout, nullptr, 3);
        }
    }
    FixedPipelineState::DynamicState dynamic{};
    if (!key.state.extended_dynamic_state) {
        dynamic = key.state.dynamic_state;
    } else {
        dynamic.raw1 = key.state.dynamic_state.raw1;
    }
    static_vector<VkVertexInputBindingDescription, 32> vertex_bindings;
    static_vector<VkVertexInputBindingDivisorDescriptionEXT, 32> vertex_binding_divisors;
    static_vector<VkVertexInputAttributeDescription, 32> vertex_attributes;
    if (!key.state.dynamic_vertex_input) {
        const size_t num_vertex_arrays = (std::min)(
            Maxwell::NumVertexArrays, static_cast<size_t>(device.GetMaxVertexInputBindings()));
        for (size_t index = 0; index < num_vertex_arrays; ++index) {
            const bool instanced = key.state.binding_divisors[index] != 0;
            const auto rate =
                instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
            vertex_bindings.push_back({
                .binding = static_cast<u32>(index),
                .stride = key.state.vertex_strides[index],
                .inputRate = rate,
            });
            if (instanced) {
                vertex_binding_divisors.push_back({
                    .binding = static_cast<u32>(index),
                    .divisor = key.state.binding_divisors[index],
                });
            }
        }
        for (size_t index = 0; index < key.state.attributes.size(); ++index) {
            const auto& attribute = key.state.attributes[index];
            if (!attribute.enabled || !stage_infos[0].loads.Generic(index)) {
                continue;
            }
            vertex_attributes.push_back({
                .location = static_cast<u32>(index),
                .binding = attribute.buffer,
                .format = MaxwellToVK::VertexFormat(device, attribute.Type(), attribute.Size()),
                .offset = attribute.offset,
            });
        }
    }
    ASSERT(vertex_attributes.size() <= device.GetMaxVertexInputAttributes());

    VkPipelineVertexInputStateCreateInfo vertex_input_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attributes.size()),
        .pVertexAttributeDescriptions = vertex_attributes.data(),
    };
    const VkPipelineVertexInputDivisorStateCreateInfoEXT input_divisor_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .vertexBindingDivisorCount = static_cast<u32>(vertex_binding_divisors.size()),
        .pVertexBindingDivisors = vertex_binding_divisors.data(),
    };
    if (!vertex_binding_divisors.empty()) {
        vertex_input_ci.pNext = &input_divisor_ci;
    }
    const bool has_tess_stages = spv_modules[1] || spv_modules[2];
    auto input_assembly_topology = MaxwellToVK::PrimitiveTopology(device, key.state.topology);
    if (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        if (!has_tess_stages) {
            LOG_WARNING(Render_Vulkan, "Patch topology used without tessellation, using points");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        }
    } else {
        if (has_tess_stages) {
            // The Vulkan spec requires patch list IA topology be used with tessellation
            // shader stages. Forcing it fixes a crash on some drivers
            LOG_WARNING(Render_Vulkan,
                        "Patch topology not used with tessellation, using patch list");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        }
    }
    const VkPipelineInputAssemblyStateCreateInfo input_assembly_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = input_assembly_topology,
        .primitiveRestartEnable =
        // MoltenVK/Metal always has primitive restart enabled and cannot disable it
        device.IsMoltenVK() ? VK_TRUE :
        (dynamic.primitive_restart_enable != 0 &&
                ((input_assembly_topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsTopologyListPrimitiveRestartSupported()) ||
                 SupportsPrimitiveRestart(input_assembly_topology) ||
                 (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsPatchListPrimitiveRestartSupported()))
            ? VK_TRUE
            : VK_FALSE),
    };
    const VkPipelineTessellationStateCreateInfo tessellation_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .patchControlPoints = std::min(key.state.patch_control_points_minus_one.Value() + 1,
                                       device.GetMaxTessellationPatchSize()),
    };
    std::array<VkViewportSwizzleNV, Maxwell::NumViewports> swizzles;
    std::ranges::transform(key.state.viewport_swizzles, swizzles.begin(), UnpackViewportSwizzle);
    VkPipelineViewportSwizzleStateCreateInfoNV swizzle_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = Maxwell::NumViewports,
        .pViewportSwizzles = swizzles.data(),
    };
    VkPipelineViewportDepthClipControlCreateInfoEXT ndc_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = key.state.ndc_minus_one_to_one.Value() != 0 ? VK_TRUE : VK_FALSE,
    };
    const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
    VkPipelineViewportStateCreateInfo viewport_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = num_viewports,
        .pViewports = nullptr,
        .scissorCount = num_viewports,
        .pScissors = nullptr,
    };
    if (device.IsNvViewportSwizzleSupported()) {
        swizzle_ci.pNext = std::exchange(viewport_ci.pNext, &swizzle_ci);
    }
    if (device.IsExtDepthClipControlSupported()) {
        ndc_info.pNext = std::exchange(viewport_ci.pNext, &ndc_info);
    }
    VkPipelineRasterizationStateCreateInfo rasterization_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable =
        static_cast<VkBool32>(dynamic.depth_clamp_disabled == 0 ? VK_TRUE : VK_FALSE),
        .rasterizerDiscardEnable =
        static_cast<VkBool32>(dynamic.rasterize_enable == 0 ? VK_TRUE : VK_FALSE),
        .polygonMode =
        MaxwellToVK::PolygonMode(FixedPipelineState::UnpackPolygonMode(key.state.polygon_mode)),
        .cullMode = static_cast<VkCullModeFlags>(
            dynamic.cull_enable ? MaxwellToVK::CullFace(dynamic.CullFace()) : VK_CULL_MODE_NONE),
        .frontFace = MaxwellToVK::FrontFace(dynamic.FrontFace()),
        .depthBiasEnable = (dynamic.depth_bias_enable != 0 ? VK_TRUE : VK_FALSE),
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
        // TODO(alekpop): Transfer from regs
    };
    const bool smooth_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsSmoothLines();
    const bool stippled_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsStippledRectangularLines();
    VkPipelineRasterizationLineStateCreateInfoEXT line_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .lineRasterizationMode = key.state.smooth_lines != 0 && smooth_lines_supported
                                     ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                     : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
        .stippledLineEnable =
            (dynamic.line_stipple_enable && stippled_lines_supported) ? VK_TRUE : VK_FALSE,
        .lineStippleFactor = key.state.line_stipple_factor,
        .lineStipplePattern = static_cast<uint16_t>(key.state.line_stipple_pattern),
    };
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .conservativeRasterizationMode = key.state.conservative_raster_enable != 0
                                             ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                             : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
        .extraPrimitiveOverestimationSize = 0.0f,
    };
    const bool supports_provoking_first_mode = device.SupportsProvokingVertexFirstMode();
    const bool supports_provoking_last_mode = device.SupportsProvokingVertexLastMode();
    const bool requested_provoking_last_mode = key.state.provoking_vertex_last != 0;
    const VkProvokingVertexModeEXT provoking_vertex_mode =
        requested_provoking_last_mode
            ? (supports_provoking_last_mode ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                            : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT)
            : (supports_provoking_first_mode ? VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
                                             : VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .provokingVertexMode = provoking_vertex_mode,
    };

    if (IsLine(input_assembly_topology) && device.IsExtLineRasterizationSupported()) {
        line_state.pNext = std::exchange(rasterization_ci.pNext, &line_state);
    }
    if (device.IsExtConservativeRasterizationSupported()) {
        conservative_raster.pNext = std::exchange(rasterization_ci.pNext, &conservative_raster);
    }
    if (device.IsExtProvokingVertexSupported()) {
        provoking_vertex.pNext = std::exchange(rasterization_ci.pNext, &provoking_vertex);
    }

    const bool supports_alpha_output = fragment_has_color0_output;
    const bool alpha_to_one_supported = device.SupportsAlphaToOne();
    const VkPipelineMultisampleStateCreateInfo multisample_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = MaxwellToVK::MsaaMode(key.state.msaa_mode),
        .sampleShadingEnable = Settings::values.sample_shading.GetValue() > 0 ? VK_TRUE : VK_FALSE,
        .minSampleShading = f32(Settings::values.sample_shading.GetValue()) / 100.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable =
            supports_alpha_output && key.state.alpha_to_coverage_enabled != 0 ? VK_TRUE : VK_FALSE,
        .alphaToOneEnable = supports_alpha_output && alpha_to_one_supported &&
                           key.state.alpha_to_one_enabled != 0 ? VK_TRUE : VK_FALSE,
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = dynamic.depth_test_enable,
        .depthWriteEnable = dynamic.depth_write_enable,
        .depthCompareOp = dynamic.depth_test_enable
                              ? MaxwellToVK::ComparisonOp(dynamic.DepthTestFunc())
                              : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = dynamic.depth_bounds_enable && device.IsDepthBoundsSupported(),
        .stencilTestEnable = dynamic.stencil_enable,
        .front = GetStencilFaceState(dynamic.front),
        .back = GetStencilFaceState(dynamic.back),
        .minDepthBounds = f32(key.state.depth_bounds_min),
        .maxDepthBounds = f32(key.state.depth_bounds_max),
    };
    if (dynamic.depth_bounds_enable && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
    }
    static_vector<VkPipelineColorBlendAttachmentState, Maxwell::NumRenderTargets> cb_attachments;
    const size_t num_attachments{NumAttachments(key.state)};
    for (size_t index = 0; index < num_attachments; ++index) {
        static constexpr std::array mask_table{
            VK_COLOR_COMPONENT_R_BIT,
            VK_COLOR_COMPONENT_G_BIT,
            VK_COLOR_COMPONENT_B_BIT,
            VK_COLOR_COMPONENT_A_BIT,
        };
        const auto& blend{key.state.attachments[index]};
        const std::array mask{blend.Mask()};
        VkColorComponentFlags write_mask{};
        for (size_t i = 0; i < mask_table.size(); ++i) {
            write_mask |= mask[i] ? mask_table[i] : 0;
        }
        write_mask &= ColorOutputMask(index);
        cb_attachments.push_back({
            .blendEnable = blend.enable != 0,
            .srcColorBlendFactor = MaxwellToVK::BlendFactor(blend.SourceRGBFactor()),
            .dstColorBlendFactor = MaxwellToVK::BlendFactor(blend.DestRGBFactor()),
            .colorBlendOp = MaxwellToVK::BlendEquation(blend.EquationRGB()),
            .srcAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.SourceAlphaFactor()),
            .dstAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.DestAlphaFactor()),
            .alphaBlendOp = MaxwellToVK::BlendEquation(blend.EquationAlpha()),
            .colorWriteMask = write_mask,
        });
    }
    const VkPipelineColorBlendStateCreateInfo color_blend_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = dynamic.logic_op_enable != 0,
        .logicOp = static_cast<VkLogicOp>(dynamic.logic_op.Value()),
        .attachmentCount = static_cast<u32>(cb_attachments.size()),
        .pAttachments = cb_attachments.data(),
        .blendConstants = {}
    };
    static_vector<VkDynamicState, 34> dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,         VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,       VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };
    if (key.state.extended_dynamic_state) {
        static constexpr std::array extended{
            VK_DYNAMIC_STATE_CULL_MODE_EXT,
            VK_DYNAMIC_STATE_FRONT_FACE_EXT,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_OP_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());

        // VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT
        if (!key.state.dynamic_vertex_input) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT);
        }
    }

    // VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
    if (key.state.dynamic_vertex_input) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    }

    // EDS2 - Core (3 states)
    if (key.state.extended_dynamic_state_2) {
        static constexpr std::array extended2{
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended2.begin(), extended2.end());
    }

    // EDS2 - LogicOp (granular)
    if (key.state.extended_dynamic_state_2_logic_op) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_EXT);
    }

    // EDS3 - Blending (composite: 3 states)
    if (key.state.extended_dynamic_state_3_blend) {
        static constexpr std::array extended3{
            VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
            VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
            VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended3.begin(), extended3.end());
    }

    // VK_EXT_color_write_enable fallback for fully on/off render targets when EDS3 blending is not available.
    if (!key.state.extended_dynamic_state_3_blend && key.state.color_write_enable_dynamic) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT);
    }

    // EDS3 - Enables (composite: per-feature)
    if (key.state.extended_dynamic_state_3_enables) {
        if (device.SupportsDynamicState3DepthClampEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LogicOpEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LineRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3ConservativeRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3LineStippleEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToCoverageEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToOneEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT);
        }
    }

    const bool is_mesh{IsMeshPipeline()};
    if (is_mesh) {
        // Mesh pipelines have no vertex input or input assembly.
        dynamic_states.erase(std::remove_if(dynamic_states.begin(), dynamic_states.end(),
                                            [](VkDynamicState state) {
                                                return state == VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT ||
                                                       state == VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT ||
                                                       state == VK_DYNAMIC_STATE_VERTEX_INPUT_EXT ||
                                                       state == VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT ||
                                                       state == VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT;
                                            }),
                             dynamic_states.end());
    }
    const VkPipelineDynamicStateCreateInfo dynamic_state_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .requiredSubgroupSize = GuestWarpSize,
    };
    static_vector<VkPipelineShaderStageCreateInfo, 5> shader_stages;
    for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        if (!spv_modules[stage]) {
            continue;
        }
        if (mesh_emulation && mesh_emulation->has_task && stage == mesh_emulation->mesh_stage) {
            continue; // the mesh program runs as a compute shader
        }
        const Shader::Info& info{stage_infos[stage]};
        // Mesh emulation: stage 0 is the vertex shader reading the compute pass's output.
        const bool emulated_vertex{mesh_emulation && stage == 0};
        const VkShaderStageFlagBits stage_flag{
            emulated_vertex
                ? VK_SHADER_STAGE_VERTEX_BIT
                : StageFlag(info, MaxwellToVK::ShaderStage(Shader::StageFromIndex(stage)))};
        // Votes, ballots and shuffles span the guest's 32-thread warp. A driver free to pick a
        // narrower subgroup (Intel runs mesh shaders SIMD16) splits the warp and breaks them.
        const bool uses_warp_ops = info.demoted_conditional_barrier || info.uses_subgroup_vote ||
                                   info.uses_subgroup_invocation_id ||
                                   info.uses_subgroup_shuffles || info.uses_subgroup_mask;
        const bool force_warp_size = !emulated_vertex && uses_warp_ops && device.IsExtSubgroupSizeControlSupported() &&
                                     device.IsGuestWarpSizeSupported(stage_flag);
        shader_stages.emplace_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = force_warp_size ? &subgroup_size_ci : nullptr,
            .flags = 0,
            .stage = stage_flag,
            .module = emulated_vertex ? *mesh_vertex_module : *spv_modules[stage],
            .pName = "main",
            .pSpecializationInfo = nullptr,
        });
    }
    // Mesh emulation draws a plain point, line or triangle list without vertex attributes.
    const VkPipelineVertexInputStateCreateInfo emulation_vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    const u32 emulation_corners{mesh_emulation ? mesh_emulation->primitive_vertices : 3U};
    const VkPipelineInputAssemblyStateCreateInfo emulation_input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = emulation_corners == 1   ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                    : emulation_corners == 2 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                             : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    VkPipelineCreateFlags flags{};
    if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
        flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    if (uses_descriptor_buffer) {
        flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    const bool unoptimized = (flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT) != 0;
    const auto trace_begin = std::chrono::steady_clock::now();
    TracePipelineBuild("BEGIN", "graphics", key.Hash(), key.unique_hashes, unoptimized);
    const VkGraphicsPipelineCreateInfo pipeline_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = mesh_emulation ? &emulation_vertex_input
                             : is_mesh      ? nullptr
                                            : &vertex_input_ci,
        .pInputAssemblyState = mesh_emulation ? &emulation_input_assembly
                               : is_mesh      ? nullptr
                                              : &input_assembly_ci,
        .pTessellationState = is_mesh ? nullptr : &tessellation_ci,
        .pViewportState = &viewport_ci,
        .pRasterizationState = &rasterization_ci,
        .pMultisampleState = &multisample_ci,
        .pDepthStencilState = &depth_stencil_ci,
        .pColorBlendState = &color_blend_ci,
        .pDynamicState = &dynamic_state_ci,
        .layout = *pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    };
    pipeline = binaries ? binaries->CreateGraphicsPipeline(pipeline_ci, *pipeline_cache,
                                                           GraphicsRecordKey(key))
                        : device.GetLogical().CreateGraphicsPipeline(pipeline_ci, *pipeline_cache);
    TracePipelineBuild("END", "graphics", key.Hash(), key.unique_hashes, unoptimized,
                       std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - trace_begin)
                           .count());

    // Log graphics pipeline creation
    if (GPU::Logging::IsActive()) {
        const std::string pipeline_info = fmt::format(
            "GraphicsPipeline created: stages={}, attachments={}",
            shader_stages.size(),
            color_blend_ci.attachmentCount
        );
        GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(pipeline_info);
    }
}

void GraphicsPipeline::Validate() {
    size_t num_images{};
    for (const auto& info : stage_infos) {
        num_images += Shader::NumDescriptors(info.texture_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.image_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.texture_descriptors);
        num_images += Shader::NumDescriptors(info.image_descriptors);
    }
    ASSERT(num_images == num_image_elements);
}

} // namespace Vulkan
