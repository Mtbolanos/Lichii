// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <span>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
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

class Device;
class PipelineBinaries;
class PipelineStatistics;
class Scheduler;

class ComputePipeline {
public:
    /// `binary_key` is the pipeline's key for shader cache v2's driver layer, `binaries` (or null)
    explicit ComputePipeline(const Device& device, Scheduler& scheduler, vk::PipelineCache& pipeline_cache,
                             PipelineBinaries* binaries, u64 binary_key,
                             DescriptorPool& descriptor_pool,
                             GuestDescriptorQueue& guest_descriptor_queue,
                             DescriptorBufferRing& descriptor_buffer_ring,
                             const PipelineBuildQueue& build_queue,
                             PipelineStatistics* pipeline_statistics,
                             VideoCore::ShaderNotify* shader_notify, const Shader::Info& info,
                             vk::ShaderModule spv_module, u64 shader_hash,
                             std::vector<u8> kept_spirv,
                             std::function<vk::ShaderModule(std::vector<u8>&)> emit = {});

    ~ComputePipeline();

    ComputePipeline& operator=(ComputePipeline&&) noexcept = delete;
    ComputePipeline(ComputePipeline&&) noexcept = delete;

    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(const ComputePipeline&) = delete;

    using BindCommand = std::function<void(vk::CommandBuffer)>;

    [[nodiscard]] bool Configure(Tegra::Engines::KeplerCompute& kepler_compute,
                                 Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                 BufferCache& buffer_cache, TextureCache& texture_cache,
                                 TextureCacheRuntime& texture_runtime,
                                 std::optional<ByteAddressedShadow>& shadow,
                                 BindCommand* deferred_bind = nullptr);

    static constexpr u64 Ocarina1440pShader = 0x6bd913e1f34e83ebULL;
    /// The same upscale to 2560x1440 from a scene the DLSS render scale shrank (ProfessorTee's
    /// #903).
    static constexpr u64 Ocarina1440pScaledShader = 0x4c1ae2fe58f30a97ULL;      // menus
    static constexpr u64 Ocarina1440pScaledFieldShader = 0x9302bc0d6d1d9bc1ULL; // gameplay
    /// The game's 3/4 upscale, which it runs to 2560x1440 at the DLSS Performance render scale.
    static constexpr u64 OcarinaThreeQuarterShader = 0xe40f5efb5480e692ULL;
    /// The same output pass at 1:1, run at 1920x1080 in handheld with the Max Resolution add-on.
    static constexpr u64 OcarinaFullSizeShader = 0xc033c3ddc9f0897bULL;
    [[nodiscard]] bool CanReplaceOcarinaUpscale() const;

    bool IsBound() const noexcept {
        return static_cast<bool>(pipeline);
    }

    [[nodiscard]] u64 ShaderHash() const noexcept {
        return shader_hash;
    }

    /// Whether the driver build has finished (or failed: then IsBound is false), so KeptSpirv() is
    /// no longer written
    [[nodiscard]] bool IsBuilt() const noexcept {
        return is_built.load(std::memory_order_acquire);
    }

    /// The SPIR-V (zstd) the pipeline was built from; read it only once IsBuilt().
    [[nodiscard]] const std::vector<u8>& KeptSpirv() const noexcept {
        return kept_spirv;
    }

    /// Storage images written by the last configured dispatch
    [[nodiscard]] std::span<const VideoCommon::ImageViewId> WrittenImages() const noexcept {
        return written_images;
    }

    struct SampledImage {
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The textures sampled by the last configured dispatch, kept while a sampled frame dump runs.
    [[nodiscard]] std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    /// Advances the frame window of EDEN_NX2_UPSCALER_TRACE
    static void TickUpscalerTrace();

    /// Whether no build of this pipeline is left queued on a worker thread, which holds it
    [[nodiscard]] bool IsBuildTaskDone() const noexcept {
        return !jobs || job.Ready();
    }

    /// Builds it on this thread, for a pipeline made with PipelineBuildQueue::owner_builds
    void Build() {
        RunBuild();
    }

private:
    /// Builds the pipeline once, on whichever thread gets here first.
    void RunBuild();

    const Device& device;
    vk::PipelineCache& pipeline_cache;
    PipelineBinaries* const binaries;
    const u64 binary_key;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    Shader::Info info;
    u64 shader_hash{};
    u32 num_descriptor_entries{};
    boost::container::small_vector<VideoCommon::ImageViewId, 8> written_images;
    std::vector<SampledImage> sampled_images;

    VideoCommon::ComputeUniformBufferSizes uniform_buffer_sizes{};

    vk::ShaderModule spv_module;
    std::vector<u8> kept_spirv;
    vk::DescriptorSetLayout descriptor_set_layout;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
    DescriptorBufferLayout descriptor_buffer_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    std::function<void()> build_task;
    std::atomic_bool build_claimed{false};
    std::condition_variable build_condvar;
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
    /// Shader cache v2: the job that builds it (PipelineBuildQueue::jobs)
    VideoCommon::ShaderJobScheduler* jobs{};
    VideoCommon::ShaderJobHandle job;
};

} // namespace Vulkan
