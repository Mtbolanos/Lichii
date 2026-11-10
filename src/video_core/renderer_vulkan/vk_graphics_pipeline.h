// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <type_traits>
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
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
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
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// Graphics pipeline libraries (EDEN_NX2_GPL): non-null `library_cache` builds the pipeline
/// from shared per-stage libraries; `libraries_only` builds only the libraries (warm-up).
struct GraphicsPipelineLibraryParams {
    GraphicsPipelineLibraryCache* library_cache{};
    std::array<u128, Tegra::Engines::Maxwell3D::Regs::MaxShaderStage> spirv_hashes{};
    bool libraries_only{};
};

/// Mesh emulation on a device without VK_EXT_mesh_shader: the mesh program (stage 0) is built as
/// a compute shader that writes its output to a storage buffer, and this vertex shader draws it
/// (see Shader::Backend::SPIRV::EmitMeshEmulationVertex).
struct MeshEmulationSetup {
    std::vector<u32> vertex_spirv;
    u32 output_binding{};
    u32 block_words{};    ///< output words per mesh workgroup
    u32 max_primitives{}; ///< triangles per mesh workgroup
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

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    /// Emits the stages' SPIR-V on a worker; with libraries it also returns each stage's SPIR-V
    /// hash (zero for absent stages).
    using EmitStagesFunc = std::function<std::array<vk::ShaderModule, NUM_STAGES>(
        std::array<u128, NUM_STAGES>& spirv_hashes)>;

    /// Graphics pipeline libraries (EDEN_NX2_GPL): non-null `library_cache` builds the pipeline
    /// from per-stage libraries; the stages were emitted with one descriptor set per stage.
    /// `libraries_only` builds the shader libraries and nothing else (warm-up, no draw).
    // Declared outside the class: GCC rejects a default argument `= {}` of a nested struct
    // with member initializers inside its enclosing class.
    using LibraryParams = GraphicsPipelineLibraryParams;

    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, Common::ThreadWorker* worker_thread,
        PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos,
        EmitStagesFunc emit_stages = {}, LibraryParams library_params = {},
        std::optional<MeshEmulationSetup> mesh_emulation = std::nullopt);

    static std::chrono::nanoseconds TakeHoldTime() noexcept;
    /// Draws held for unbuilt pipelines this session (never reset; the worst frame line takes
    /// differences of it).
    static std::chrono::nanoseconds HoldTimeTotal() noexcept;

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    const std::array<u64, 6>& ShaderHashes() const noexcept {
        return key.unique_hashes;
    }

    struct SampledImage {
        u32 stage;
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The sampled textures of the last configured draw, kept while a sampled frame dump runs.
    std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    /// Texture handles of the last configured draw whose handle program gave 0.
    u32 ZeroTextureHandles() const noexcept {
        return zero_texture_handles;
    }

    bool FragmentCanDiscard() const noexcept {
        return stage_infos[NUM_STAGES - 1].uses_demote_to_helper_invocation;
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

    /// Vertices of the emulation's triangle list for one mesh workgroup.
    [[nodiscard]] u32 MeshEmulationVerticesPerGroup() const noexcept {
        return mesh_emulation ? mesh_emulation->max_primitives * 3 : 0;
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

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transition_keys.clear();
        transitions.clear();
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return build->IsBuilt();
    }

    [[nodiscard]] bool Failed() const noexcept {
        return build->Failed();
    }

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) { return pl->ConfigureImpl<Spec>(is_indexed); };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    /// EDEN_NX2_EXP32 (texture binding memo; zortos293/Lichii#717's BINDMEMO reduced to "the same
    /// textures as this pipeline's previous draw"): that draw's texture bindings as resolved and
    /// pushed, for pipelines whose image bindings are all sampled textures (no texture or image
    /// buffers, no storage images). A draw reading the same handles with the same texture cache
    /// state (TextureCache::TextureMemoState), the same guest TIC/TSC entries and clean images
    /// takes the view and sampler ids and the descriptor entries from it instead of resolving them
    /// (GetSamplerId, FillImageViews) and building them (PushImageDescriptors).
    struct TextureMemo {
        bool valid{};
        u64 generation{};  ///< the switch generation it was recorded in (a switch change drops it)
        u64 touch_frame{}; ///< the frame its images were last touched in the LRU
        TextureCache::TextureMemoState state{};
        std::vector<u32> handles;         ///< per texture: the TIC handle read
        std::vector<u32> sampler_handles; ///< per texture: the TSC handle read
        std::vector<TextureCache::TextureMemoEntry> entries;
        std::vector<VideoCommon::ImageViewId> views;
        std::vector<VideoCommon::SamplerId> samplers;
        std::vector<DescriptorUpdateEntry> descriptors; ///< per texture, in push order
        std::vector<u8> rescaled; ///< per texture descriptor (array), in push order
    };

    /// Whether the memo holds this draw's textures (counts why not).
    [[nodiscard]] bool TextureMemoHit(const TextureCache::TextureMemoState& state,
                                      std::span<const VideoCommon::ImageViewInOut> views,
                                      std::span<const u32> sampler_handles);

    /// Remembers the textures this draw resolved and pushed, when nothing in the texture cache
    /// changed since `state` was taken and the tables still hold them as resolved.
    void RecordTextureMemo(const TextureCache::TextureMemoState& state,
                           std::span<const VideoCommon::ImageViewInOut> views,
                           std::span<const u32> sampler_handles,
                           std::span<const VideoCommon::SamplerId> samplers,
                           std::span<const DescriptorUpdateEntry> descriptors,
                           std::span<const u8> rescaled);

    /// EXP32 verify: a memo hit resolved and pushed the full way; compares the memo with it,
    /// logs and counts a difference, and turns the memo off when the switch asks (bit 4).
    bool VerifyTextureMemo(std::span<const VideoCommon::ImageViewInOut> views,
                           std::span<const VideoCommon::SamplerId> samplers,
                           std::span<const DescriptorUpdateEntry> descriptors,
                           std::span<const u8> rescaled, int switch_value);

    template <typename Spec>
    bool ConfigureImpl(bool is_indexed);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    struct PipelineStates;

    /// The fixed-function state of the pipeline (shared by the monolithic and library builds).
    void FillStates(PipelineStates& states) const;

    void MakePipeline(VkRenderPass render_pass);

    /// Library build (EDEN_NX2_GPL): gets or builds the two shader libraries, then (unless
    /// libraries_only) links them with this pipeline's vertex input and fragment output parts.
    /// False when a library failed; the caller then builds the pipeline the monolithic way.
    bool MakeLibraryPipeline(VkRenderPass render_pass);

    GplLibrary MakeShaderLibrary(GplPart part, const PipelineStates& states,
                                 VkRenderPass render_pass) const;

    std::string ShaderLibraryKey(GplPart part, const PipelineStates& states,
                                 VkRenderPass render_pass) const;

    vk::Pipeline LinkLibraries(VkRenderPass render_pass, bool optimized);

    /// The library route failed: per-stage sets kept, with a layout a monolithic pipeline takes.
    void FallBackToMonolithic(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    bool (*configure_func)(GraphicsPipeline*, bool){};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    std::vector<GraphicsPipeline*> transitions;
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    u64 last_use_frame{};
    bool used{};

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    std::vector<SampledImage> sampled_images;
    u32 num_textures{};
    bool fragment_has_color0_output{};
    u32 zero_texture_handles{};
    u32 color_output_mask{};
    bool writes_storage{};
    TextureMemo texture_memo;

    /// Graphics pipeline libraries (EDEN_NX2_GPL): one descriptor set per stage (set index =
    /// stage index), the shader libraries, and this pipeline's own vertex input and fragment
    /// output libraries.
    GraphicsPipelineLibraryCache* const library_cache;
    const bool libraries_only;
    std::array<u128, NUM_STAGES> spirv_hashes{};
    std::array<vk::DescriptorSetLayout, NUM_STAGES> stage_set_layouts;
    std::array<std::string, NUM_STAGES> stage_set_signatures;
    std::array<DescriptorAllocator, NUM_STAGES> stage_allocators;
    std::array<vk::DescriptorUpdateTemplate, NUM_STAGES> stage_templates;
    std::array<size_t, NUM_STAGES> stage_descriptor_offsets{}; ///< Bytes into the update data
    bool has_stage_sets{};
    std::array<std::shared_ptr<const GplLibrary>, 2> shader_libraries;
    vk::Pipeline vertex_input_library;
    vk::Pipeline fragment_output_library;
    vk::DescriptorSetLayout fallback_empty_set_layout;
    bool linked{};

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
