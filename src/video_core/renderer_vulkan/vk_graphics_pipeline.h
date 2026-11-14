// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/evicted_pointer_slots.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineBinaries;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// EDEN_NX2_DRAW_REUSE_PROBE=1: how many draws repeat the previous frame's slowly changing state,
/// and what preparing them costs (RasterizerVulkan::NoteDrawReuse).
inline bool DrawReuseProbeEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_DRAW_REUSE_PROBE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Mesh emulation on a device without VK_EXT_mesh_shader: the mesh program (stage 0) is built as
/// a compute shader that writes its output to a storage buffer, and this vertex shader draws it
/// (see Shader::Backend::SPIRV::EmitMeshEmulationVertex).
struct MeshEmulationSetup {
    std::vector<u32> vertex_spirv;
    u32 output_binding{};
    u32 block_words{};          ///< output words per mesh workgroup
    u32 max_primitives{};       ///< primitives per mesh workgroup
    u32 primitive_vertices{3};  ///< vertices per primitive: 1 points, 2 lines, 3 triangles
    // With a task program (stage 0, the mesh program at mesh_stage): its payload buffer and the
    // distribution pass's offsets and arguments (Shader::RuntimeInfo), and that pass's shader.
    bool has_task{};
    u32 mesh_stage{};
    u32 payload_stride{};
    u32 payload_binding{};
    u32 offsets_binding{};
    u32 args_binding{};
    std::vector<u32> distribution_spirv;
    // Output layout, for the readback diagnostics.
    u32 max_vertices{};
    u32 vertex_lines{};
    std::array<u32, 4> position_lines{~0U, ~0U, ~0U, ~0U}; ///< line of PositionX..W
};

/// What an evicted graphics pipeline needs to be built again without translating its shaders
struct KeptGraphicsStages {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

    std::array<std::vector<u8>, NUM_STAGES> spirv; ///< zstd, empty for absent stages
    std::array<std::unique_ptr<Shader::Info>, NUM_STAGES> infos;
    std::optional<MeshEmulationSetup> mesh_emulation;
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;

    /// Memory it holds, roughly
    [[nodiscard]] u64 Bytes() const noexcept;
};

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, PipelineBinaries* binaries,
        VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, const PipelineBuildQueue& build_queue,
        PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<std::vector<u32>, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos,
        std::function<std::array<std::vector<u32>, NUM_STAGES>()> emit_stages = {},
        std::optional<MeshEmulationSetup> mesh_emulation = std::nullopt);

    static std::chrono::nanoseconds TakeHoldTime() noexcept;
    /// Draws resolved from the texture memo and in full since the last call.
    static std::pair<u64, u64> TakeTextureMemoCounts() noexcept;

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    u64 DynamicAttributeTypes() const noexcept { return key.state.attribute_types; }
    const std::array<u64, 6>& ShaderHashes() const noexcept {
        return key.unique_hashes;
    }
    const GraphicsPipelineCacheKey& Key() const noexcept {
        return key;
    }
    /// Each stage's SPIR-V (zstd), kept once the pipeline is built; empty for absent stages.
    const std::array<std::vector<u8>, NUM_STAGES>& KeptSpirv() const noexcept {
        return kept_spirv;
    }

    struct SampledImage {
        u32 stage;
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The sampled textures of the last configured draw, kept while a sampled frame dump or the
    /// draw reuse probe runs.
    std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    /// Color components of a render target the fragment shader writes, as VkColorComponentFlags
    VkColorComponentFlags ColorOutputMask(size_t render_target) const noexcept {
        return (color_output_mask >> (render_target * 4)) & 0xF;
    }

    /// Color components the fragment shader writes, 4 bits per render target
    u32 ColorOutputMask() const noexcept {
        return color_output_mask;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }

    /// Switch 2 task/mesh pipeline: drawn with vkCmdDrawMeshTasksEXT.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return stage_infos[0].is_task || stage_infos[0].is_mesh;
    }

    /// Mesh pipeline drawn through the compute emulation (MeshEmulationSetup).
    [[nodiscard]] bool IsEmulatedMesh() const noexcept {
        return mesh_emulation.has_value();
    }

    /// Output bytes the emulation needs for one mesh workgroup.
    [[nodiscard]] u64 MeshEmulationBlockBytes() const noexcept {
        return mesh_emulation ? u64{mesh_emulation->block_words} * 4 : 0;
    }

    /// Vertices of the emulation's point, line or triangle list for one mesh workgroup.
    [[nodiscard]] u32 MeshEmulationVerticesPerGroup() const noexcept {
        return mesh_emulation ? mesh_emulation->max_primitives * mesh_emulation->primitive_vertices
                              : 0;
    }

    /// The emulation's setup (output layout), or null.
    [[nodiscard]] const MeshEmulationSetup* MeshEmulation() const noexcept {
        return mesh_emulation ? &*mesh_emulation : nullptr;
    }

    /// The emulation's output buffer and workgroup count for the next Configure.
    void SetMeshEmulationOutput(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                u32 groups) noexcept {
        mesh_output_buffer = buffer;
        mesh_output_offset = offset;
        mesh_output_size = size;
        mesh_groups = groups;
    }

    struct MeshEmulationBuffer {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        VkDeviceSize size{};
    };
    /// With a task program: the task payload, offset and argument buffers, the arena counter,
    /// and the words of the output arena (the draw's group count is the task's).
    void SetMeshEmulationTaskBuffers(const MeshEmulationBuffer& payload,
                                     const MeshEmulationBuffer& offsets,
                                     const MeshEmulationBuffer& args,
                                     const MeshEmulationBuffer& counter,
                                     u32 capacity_words) noexcept {
        mesh_task_buffers = {payload, offsets, args, counter};
        mesh_capacity = capacity_words;
    }

    [[nodiscard]] bool MeshEmulationHasTask() const noexcept {
        return mesh_emulation && mesh_emulation->has_task;
    }

    [[nodiscard]] u32 MeshEmulationPayloadStride() const noexcept {
        return mesh_emulation ? mesh_emulation->payload_stride : 0;
    }
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    bool Configure(bool is_indexed) {
        return configure_func(this, is_indexed);
    }

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it{std::find(transition_keys.begin(), transition_keys.end(), current_key)};
        return it != transition_keys.end() ? transitions[std::distance(transition_keys.begin(), it)]
                                           : nullptr;
    }

    /// The frame the pipeline cache last handed this pipeline out (evicts long unused ones).
    void MarkUsed(u64 frame) noexcept {
        last_use_frame = frame;
        used = true;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use_frame;
    }

    /// Whether the cache has handed this pipeline out at all (frame 0 included), as opposed to a
    /// pipeline prebuilt from the disk cache that no draw has used yet.
    [[nodiscard]] bool WasUsed() const noexcept {
        return used;
    }

    /// The constant buffer slots each program binds as storage buffer pointers
    [[nodiscard]] const VideoCommon::EvictedPointerSlots::Slots& MappedPointers() const noexcept {
        return mapped_pointers;
    }

    void SetMappedPointers(VideoCommon::EvictedPointerSlots::Slots slots) noexcept {
        mapped_pointers = std::move(slots);
    }

    /// Moves out the SPIR-V kept after the build, with the stage infos: for a built pipeline being
    /// evicted
    [[nodiscard]] KeptGraphicsStages TakeKeptStages();

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transition_keys.clear();
        transitions.clear();
    }

    [[nodiscard]] bool WritesStorage() const noexcept {
        return writes_storage;
    }

    /// Writes storage images or image buffers (their images are marked modified when resolved).
    [[nodiscard]] bool WritesImages() const noexcept {
        return writes_images;
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return build->IsBuilt();
    }

    /// Builds it on this thread, for a pipeline made with PipelineBuildQueue::owner_builds
    void Build() {
        build->Build();
    }

    /// Waits up to `timeout` for the build; true when built.
    [[nodiscard]] bool WaitBuilt(std::chrono::microseconds timeout) const {
        return build->WaitFor(timeout);
    }

    [[nodiscard]] bool Failed() const noexcept {
        return build->Failed();
    }

    /// Fast draw (nx2_fast_draw.h): whether draws can keep the image views and samplers the last
    /// captured Configure resolved. Handles computed at draw time are evaluated again for each
    /// delta draw (ComputedHandlesChanged).
    [[nodiscard]] bool TakesDeltaDraws() const noexcept {
        return delta_blocker == DeltaBlocker::None && delta_views_valid;
    }

    /// Whether a handle the last captured Configure computed at draw time (a handle program, which
    /// reads constant and storage buffers) evaluates to another value now: a delta draw would keep
    /// views and samplers resolved from the old one.
    [[nodiscard]] bool ComputedHandlesChanged() const;

    /// Why TakesDeltaDraws is false (diagnostics).
    enum class DeltaBlocker : u8 {
        None,
        WritesImages,
        Mesh,
        NotConfigured,
        Count,
    };
    [[nodiscard]] DeltaBlocker DeltaDrawBlocker() const noexcept {
        return delta_blocker != DeltaBlocker::None ? delta_blocker
               : delta_views_valid                 ? DeltaBlocker::None
                                                   : DeltaBlocker::NotConfigured;
    }

    /// Constant buffer slots per stage the pipeline reads texture and image handles from.
    [[nodiscard]] const std::array<u32, NUM_STAGES>& HandleConstantBuffers() const noexcept {
        return handle_constant_buffers;
    }

    /// The bytes [first, second) of a HandleConstantBuffers slot that handles are read from.
    [[nodiscard]] std::pair<u32, u32> HandleRange(size_t stage, u32 slot) const noexcept {
        return handle_ranges[stage][slot];
    }

    /// The same bytes word by word: sorted, disjoint ranges within HandleRange.
    [[nodiscard]] std::span<const std::pair<u32, u32>> HandleWords(size_t stage,
                                                                   u32 slot) const noexcept {
        const auto [first, count] = handle_word_index[stage][slot];
        return std::span{handle_words}.subspan(first, count);
    }

    /// Resolves and binds every buffer as Configure does (storage buffer addresses are read from
    /// the constant buffers again) and records the draw's descriptors with the image views,
    /// samplers and texture buffers the last captured Configure resolved. False when the draw
    /// can't be recorded, as for Configure.
    bool ConfigureDelta(bool is_indexed);

    /// The next Configure keeps the views and samplers it resolves for delta draws. With
    /// `verify`, it first compares them with the ones kept before; DeltaMismatch reports it.
    void CaptureNextDelta(bool verify) noexcept {
        capture_delta = true;
        verify_delta = verify;
    }
    [[nodiscard]] bool DeltaMismatch() const noexcept {
        return delta_mismatch;
    }

    /// Whether an image the kept views sample has an upload pending or a newer alias to copy in:
    /// ConfigureDelta prepares none, Configure does.
    [[nodiscard]] bool DeltaImagesChange() const noexcept;

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) { return pl->ConfigureImpl<Spec>(is_indexed); };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    template <typename Spec>
    bool ConfigureImpl(bool is_indexed);

    void KeepImageDescriptors(size_t stage, size_t first_entry);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    void MakePipeline(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    PipelineBinaries* const binaries; ///< Shader cache v2's driver layer, else null
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    bool (*configure_func)(GraphicsPipeline*, bool){};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    std::vector<GraphicsPipeline*> transitions;
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    u64 last_use_frame{};
    bool used{};

    std::array<std::vector<u32>, NUM_STAGES> spirv; ///< Until the build makes its modules
    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;
    std::array<std::vector<u8>, NUM_STAGES> kept_spirv; ///< zstd, after the build

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    std::vector<SampledImage> sampled_images;
    u32 num_textures{};
    bool fragment_has_color0_output{};
    u32 color_output_mask{};
    bool writes_storage{};
    /// Every texture is sampled with a handle read from a constant buffer: ConfigureImpl can
    /// resolve them from texture_memo.
    bool texture_memo_eligible{};
    VideoCommon::SampledTextureMemo texture_memo;
    bool writes_images{};

    DeltaBlocker delta_blocker{};
    bool delta_views_valid{};
    bool capture_delta{};
    bool verify_delta{};
    bool delta_mismatch{};
    std::array<bool, NUM_STAGES> configured_stages{};
    std::array<u32, NUM_STAGES> handle_constant_buffers{};
    std::array<std::array<std::pair<u32, u32>, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers>,
               NUM_STAGES>
        handle_ranges{};
    std::vector<std::pair<u32, u32>> handle_words;
    std::array<std::array<std::pair<u16, u16>, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers>,
               NUM_STAGES>
        handle_word_index{};
    std::vector<VideoCommon::ImageViewInOut> delta_views;
    std::vector<VideoCommon::SamplerId> delta_samplers;
    /// A descriptor whose handles the captured Configure computed at draw time: the descriptor
    /// (stage, kind as ComputedHandleKind, index in its list) and its elements' values, count of
    /// them from delta_computed_values[first_value].
    struct ComputedHandle {
        u8 stage;
        u8 kind;
        u16 descriptor;
        u32 count;
        u32 first_value;
    };
    enum ComputedHandleKind : u8 { TextureBuffer, ImageBuffer, Texture, Image };
    std::vector<ComputedHandle> delta_computed_handles;
    std::vector<u32> delta_computed_values;
    /// Delta draws: the image descriptor entries (first entry and count per
    /// stage) and rescaling words pushed for delta_views, which delta draws push again while no
    /// image changed its scale (TextureCache::ScaleEpoch).
    std::vector<DescriptorUpdateEntry> kept_image_entries;
    std::array<std::pair<u32, u32>, NUM_STAGES> kept_image_ranges{};
    std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> kept_rescaling{};
    u64 kept_scale_epoch{};
    bool kept_images_valid{};

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    const bool worker_only_build;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};

    // Mesh emulation (MeshEmulationSetup).
    std::optional<MeshEmulationSetup> mesh_emulation;
    vk::ShaderModule mesh_vertex_module;
    vk::Pipeline mesh_compute_pipeline;
    VkDescriptorSet mesh_descriptor_set{}; ///< shared by the dispatch and the draw it feeds
    VkBuffer mesh_output_buffer{};
    VkDeviceSize mesh_output_offset{};
    VkDeviceSize mesh_output_size{};
    u32 mesh_groups{};
    // With a task program: its compute pipeline, the distribution pass between it and the mesh
    // dispatch, and the per-draw buffers (payload, offsets, arguments, arena counter).
    vk::Pipeline task_compute_pipeline;
    vk::ShaderModule distribution_module;
    vk::DescriptorSetLayout distribution_set_layout;
    vk::PipelineLayout distribution_layout;
    vk::Pipeline distribution_pipeline;
    DescriptorAllocator distribution_allocator;
    std::array<MeshEmulationBuffer, 4> mesh_task_buffers{};
    u32 mesh_capacity{};

    std::optional<PipelineBuild> build;
};

} // namespace Vulkan
