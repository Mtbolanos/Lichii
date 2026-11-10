// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stop_token>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;
class TranslateQueue;
struct TranslationJob;
struct TranslationRecord;

namespace PrecompiledKept {
class Index;
struct Record;
} // namespace PrecompiledKept

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct ComputePipelineCacheEntry {
    std::unique_ptr<ComputePipeline> pipeline;
    std::string failure;
    /// Constant buffer slots that were null pointers at translation (see NullStoragePointerSlots)
    std::vector<std::pair<u32, u32>> null_pointer_slots;
    /// Constant buffer slots bound as storage buffer pointers (see MappedStoragePointerKeys)
    std::vector<u64> mapped_pointers;
    u32 rebuilds{};
    /// A rebuild still compiling: dispatches keep this entry's pipeline until it is ready
    /// (zortos293/Lichii#969).
    std::unique_ptr<ComputePipelineCacheEntry> pending;
    /// GPU thread: frame_count at the last dispatch through this entry (EXP51 eviction)
    u64 last_use_frame{};
    /// Whether a dispatch went through it this session (a boot build may never be dispatched)
    bool dispatched{};

    [[nodiscard]] ComputePipeline* Get(bool native, GPUVAddr address,
                                       const ComputePipelineCacheKey& key) const;
};

/// EDEN_NX2_ASYNC_DISCARD, read on first use: what a native draw does while its pipeline is
/// missing. 0 waits for every pipeline, 1 (the default) skips by the legacy heuristic alone, 2
/// skips only draws into render targets the frame clears and redraws (fable_answer_18 1b).
/// Non-native draws keep the legacy heuristic in every mode.
[[nodiscard]] int AsyncDiscardMode();

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    /// Whether the last CurrentGraphicsPipeline returned null because its draw was skipped for an
    /// unbuilt pipeline (asynchronous shaders), not because there is no pipeline to build.
    [[nodiscard]] bool LastLookupSkipped() const noexcept {
        return discard_decision.serial == lookup_serial && discard_decision.skip;
    }

    void TickFrame();

    /// What the "Memory owners" log line counts here (GPU thread).
    struct MemoryOwnerCounts {
        size_t graphics_built{};       ///< Built graphics pipelines (each holds driver memory)
        size_t graphics_entries{};     ///< graphics_cache entries, unbuilt ones included
        size_t compute{};              ///< Compute pipelines, pending rebuilds included
        size_t awaiting_destruction{}; ///< Evicted or retired, destroyed once the GPU is done
        u64 cache_data_bytes{};        ///< vkGetPipelineCacheData's size, at most a minute old
    };
    [[nodiscard]] MemoryOwnerCounts MemoryOwners();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// What a graphics warm-up entry came to: built, not buildable (a guess the translation
    /// rejects, which another warm-up would reject too), or refused by the driver.
    enum class WarmResult { Built, Unbuildable, DriverFailed };

    /// Graphics pipeline libraries (EDEN_NX2_GPL): builds the pre-rasterization and fragment
    /// shader libraries of the pipeline `key` would make, on the calling thread, with no draw and
    /// no pipeline kept (for a warm-up). `pools` must be the calling thread's own. Unbuildable
    /// when the device has no libraries or the pipeline takes the monolithic route, DriverFailed
    /// when a library build failed.
    /// The key may hold a single stage (no partner): a vertex-only key builds just that vertex
    /// program's pre-rasterization library (vertex SPIR-V does not depend on the fragment
    /// stage); a fragment-only key builds just the fragment library, emitted as if the partner
    /// writes every varying (the common variant). The key's state (attribute types, render
    /// target formats, depth and raster state) is what the libraries are keyed and built with.
    WarmResult WarmGraphicsLibraries(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                     std::span<Shader::Environment* const> envs);

private:
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    /// FixedPipelineState::Refresh of graphics_key, keeping only the vertex attribute types the
    /// bound programs read (see VideoCommon::KeyedAttributeTypes).
    void RefreshGraphicsKey();

    /// EDEN_NX2_PIPELINE_CLEAN / EDEN_NX2_EXP19: CurrentGraphicsPipeline without
    /// FixedPipelineState::Refresh nor the key comparison while nothing Refresh reads changed
    /// since the last lookup. `patch`: EXP19=2. With EXP50 the blend registers are compared by
    /// what they make (FixedPipelineState::BlendingMatches), not word for word. With EXP52 a
    /// change of programs alone keys only the attribute types again and looks the key up.
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineClean(bool patch);
    /// The registers Refresh reads with these features, as (first word, words) ranges.
    /// `blend_by_value` (EXP50): the blend registers go to their own ranges.
    void BuildCleanWatch(bool patch, bool blend_by_value, bool patch_vertex);
    [[nodiscard]] bool CleanInputsUnchanged() const;
    void TakeCleanSnapshot();
    [[nodiscard]] bool CleanBlendUnchanged() const;
    void TakeCleanBlendSnapshot();

    struct CleanLookup {
        struct Range {
            u32 first;
            u32 count;
        };
        bool built{};
        bool patch{};                  ///< Built for CurrentGraphicsPipelineClean(true)
        bool blend_by_value{};         ///< Built for EXP50: blend registers in blend_ranges
        bool patch_vertex{};           ///< Built for EXP55: vertex attribute formats left out
        std::vector<Range> ranges;     ///< Read by every Refresh
        std::vector<Range> xfb_ranges; ///< Read while transform feedback is enabled
        /// With blend_by_value: the registers the attachments and the dual-source bit come from.
        std::vector<Range> blend_ranges;
        std::vector<u32> snapshot;     ///< ranges' words at the last Refresh
        std::vector<u32> xfb_snapshot;
        /// blend_ranges' words the key's attachments and dual-source bit were last made from (or
        /// proved equal to, by value).
        std::vector<u32> blend_snapshot;
        std::array<u64, 6> hashes{};   ///< graphics_key.unique_hashes at the last Refresh
        const Tegra::Engines::Maxwell3D* maxwell3d{};
        Maxwell::PrimitiveTopology topology{};
        Tegra::Engines::Maxwell3D::EngineHint engine_state{};
        /// current_pipeline when its key was graphics_key after the last Refresh, else null.
        GraphicsPipeline* pipeline{};
        u64 hits{};
        u64 patches{};
        u64 misses{};
        /// Hits the old check would have refreshed: blend registers or Dirty::Blending moved
        /// while the state they make stayed the same (EXP50).
        u64 blend_saved{};
        /// Hits whose programs changed while every other Refresh input stood: only the attribute
        /// types were keyed again, then Next looked the new key up (EXP52).
        u64 programs_only{};
        /// EXP52=2: programs-only keys checked against a full Refresh, and those that differed.
        u64 programs_verified{};
        u64 programs_mismatches{};
        u32 programs_mismatches_logged{}; ///< For the whole session: the first 16 are logged
        /// Why lookups refreshed: pipeline changed or unknown, programs, Dirty::Blending,
        /// other flags / topology / engine hint / dual-source bit, watched registers. With EXP52
        /// on, programs count only when nothing else is in the way, which never refreshes.
        std::array<u64, 5> miss_reasons{};
        /// miss_reasons[3] ("other state") by its first differing part: vertex input, viewport
        /// swizzles, topology, engine state, dual-source blend.
        std::array<u64, 5> other_parts{};
        std::chrono::steady_clock::time_point last_log{};
    } clean;

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    /// Whether the current draw may be skipped while its pipeline is not ready (native draws
    /// unless EDEN_NX2_ASYNC_DISCARD=0, and the legacy heuristic; full-screen passes never; with
    /// EDEN_NX2_ASYNC_DISCARD=2 only into recurring render targets). The one test BuiltPipeline
    /// and the translate queue both ask, answered once per lookup.
    [[nodiscard]] bool DrawMayBeSkipped() const;
    /// Counts CurrentGraphicsPipeline calls, so DrawMayBeSkipped classifies the targets once per
    /// lookup however often a lookup asks.
    u64 lookup_serial{};
    struct DiscardDecision {
        u64 serial{~u64{0}};
        bool skip{};
    };
    mutable DiscardDecision discard_decision;
    /// Counts one decision for the 10 s line, and lists a persistent target the first time a
    /// draw into it misses its pipeline. GPU thread.
    void NoteDiscardDecision(bool skip, const VideoCommon::DrawTargets& targets) const;
    /// TickFrame: the EDEN_NX2_ASYNC_DISCARD counts of the last 10 s.
    void ReportAsyncDiscard();
    /// EDEN_NX2_ASYNC_DISCARD counts (fable_answer_18 1c), GPU thread only. Lookups by class of
    /// their targets (index: VideoCommon::DrawTargetClass), a lookup that repeats the one before
    /// (same pipeline key and target, same frame: the rasterizer's second lookups) counted once.
    struct DiscardStats {
        std::array<u32, 3> skipped{};
        std::array<u32, 3> waited{};
        /// GPU thread time in the translate queue's capped Wait for a draw that waits
        std::chrono::steady_clock::duration queue_wait{};
        std::chrono::nanoseconds held_at_window{}; ///< GraphicsPipeline::HoldTimeTotal then
        std::chrono::steady_clock::time_point window{};
        // The last lookup counted.
        GraphicsPipelineCacheKey last_key{};
        u64 last_frame{~u64{0}};
        GPUVAddr last_target{};
        VideoCommon::DrawTargetClass last_kind{};
        bool last_skip{};
    };
    mutable DiscardStats discard_stats;
    /// Persistent targets listed so far (32 at most per session).
    mutable std::unordered_set<GPUVAddr> listed_persistent_targets;

    /// With `keep_pointers`, the slots an evicted pipeline of this key bound stay storage buffer
    /// pointers whatever they hold now (from zortos293/Lichii#1143).
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        const VideoCommon::EvictedPointerSlots::Slots* keep_pointers = nullptr);

    /// The pointer slots kept for the current key (evicted_pointer_slots), null when none
    [[nodiscard]] const VideoCommon::EvictedPointerSlots::Slots* KeptPointers() const;
    /// Answers the kept slots of each program as storage buffer pointers (KeepStoragePointers)
    static void KeepPointers(std::span<VideoCommon::GraphicsEnvironment> envs,
                             const GraphicsPipelineCacheKey& key,
                             const VideoCommon::EvictedPointerSlots::Slots* keep_pointers);
    /// Logs a kept pointer slot (an index of keep_pointers) that this translation would otherwise
    /// have classified as no pointer: null, unmapped, or out of range right now.
    void WarnKeptPointer(size_t program, u64 slot) const;

    /// Stores a graphics pipeline built at run time in the disk cache (unless it is there and
    /// loads), with the environments it was translated from. GPU thread.
    void StoreRuntimeGraphicsPipeline(
        const GraphicsPipelineCacheKey& key,
        const std::array<VideoCommon::GraphicsEnvironment, Maxwell::MaxShaderProgram>& envs);

    /// EDEN_NX2_ASYNC_TRANSLATE=1/2 (translate_queue set). Queues the current key's translation
    /// from snapshot environments when the draw may be skipped; false when it is to be built on
    /// the GPU thread instead. `keep_pointers` as CreateGraphicsPipeline's.
    bool EnqueueTranslation(const VideoCommon::EvictedPointerSlots::Slots* keep_pointers = nullptr);
    /// A later lookup of a key whose entry is empty: its translated pipeline when ready; `pending`
    /// when the draw is to be skipped meanwhile. Waits (capped) or builds here for a draw that
    /// must not be skipped.
    std::unique_ptr<GraphicsPipeline> AdoptTranslation(bool& pending);
    /// What a finished job gives: its pipeline once its texture reads check out on the GPU
    /// thread. `retry` when it was not usable (to be translated again, or built here).
    std::unique_ptr<GraphicsPipeline> AdoptJob(std::unique_ptr<TranslationJob> job, bool& retry);
    /// The translation itself, on a translation thread.
    void TranslateOnWorker(TranslationJob& job);
    /// Between frames: adopts finished jobs whose draws did not come back, sweeps wedged ones
    /// and logs the counts every 10 s.
    void TickTranslations();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools = nullptr,
        std::unique_ptr<EmissionBudget::Reservation> reservation = nullptr,
        bool libraries_only = false, TranslationRecord* record = nullptr);

    /// EDEN_NX2_PRECOMPILED_FAST (zortos293/Lichii#896): the current key's pipeline from the
    /// shader infos and SPIR-V the precompiled preparation kept, when the key is precompiled and
    /// the game's state answers the recorded environments' questions as they were answered
    /// (FileEnvironment::AnswersHoldFor): created on the GPU thread without translating, built on
    /// a worker from the kept SPIR-V, stored in the disk cache like a translated one. Null when
    /// the key is not precompiled or a check differs: the draw takes the normal path then.
    /// `keep_pointers` (an evicted key's) are answered as pointers before the check, as a
    /// translation would see them.
    std::unique_ptr<GraphicsPipeline> CreatePrecompiledGraphicsPipeline(
        const VideoCommon::EvictedPointerSlots::Slots* keep_pointers = nullptr);
    /// The compute counterpart: the first of the key's records (one per storage pointer state it
    /// was prepared in) whose answers hold for `env`. An entry without a pipeline otherwise.
    ComputePipelineCacheEntry CreatePrecompiledComputePipeline(const ComputePipelineCacheKey& key,
                                                               Shader::Environment& env);
    /// EDEN_NX2_PRECOMPILED_FAST=2: the stages translated from the game's state as the normal
    /// path would, on the GPU thread, compared with the kept SPIR-V and infos (true when they
    /// are the same). The programs are translated and emitted once, into audit_pools.
    bool AuditPrecompiled(const PrecompiledKept::Index& index, const PrecompiledKept::Record& record,
                          std::span<Shader::Environment* const> envs, bool compute,
                          const ComputePipelineCacheKey* compute_key);
    /// TickFrame: precompiled graphics pipelines whose build failed are dropped (their keys
    /// rejected), so their next draw translates them.
    void CheckPrecompiledBuilds();
    /// TickFrame: the fast path's counts of the last 10 s.
    void ReportPrecompiled();
    /// Publishes the kept file's index for the GPU thread (`kept`, or a finished `part` the
    /// preparation could not rename over it); false when there is none for `stamp`. With
    /// `check` (a file another build may have written) it is published only when
    /// CheckKeptSample finds it equal, and deleted otherwise.
    bool LoadKept(const std::filesystem::path& kept, const std::filesystem::path& part,
                  const std::filesystem::path& precompiled, const std::string& stamp, bool check);
    /// The boot check of a kept file: a sample of its pipelines (64 graphics, up to 16 compute,
    /// spread over the index) translated again from their recorded environments with this build
    /// and its settings, and compared with the kept SPIR-V and infos. True when all are equal.
    bool CheckKeptSample(const PrecompiledKept::Index& index, std::string_view file_name);
    [[nodiscard]] std::shared_ptr<const PrecompiledKept::Index> KeptIndex() const;
    /// What a kept file must have been written for, beyond the prepared stamp (which holds the
    /// release, driver and GPU): the cache version, the settings SPIR-V depends on, the library
    /// split, the OoT visual workaround and canonical keys. Not the build: CheckKeptSample.
    [[nodiscard]] std::string KeptStamp(const std::string& prepared_stamp) const;

    /// Set by LoadKept; read by the GPU thread when a new key is looked up.
    mutable std::mutex kept_mutex;
    std::shared_ptr<const PrecompiledKept::Index> kept_index;
    std::atomic<bool> kept_available{};
    /// GPU thread: keys whose precompiled build failed or whose audit differed.
    std::unordered_set<u64> precompiled_rejected;
    /// GPU thread: precompiled graphics pipelines not built yet (checked by TickFrame).
    std::vector<GraphicsPipelineCacheKey> precompiled_pending;
    /// EDEN_NX2_PRECOMPILED_FAST=2: the audit's own pools (nothing else holds IR in them).
    std::unique_ptr<ShaderPools> audit_pools;
    /// GPU thread counts for the 10 s line (the worker builds count in an atomic).
    struct PrecompiledStats {
        u32 graphics{};
        u32 compute{};
        u32 failed{};
        u32 not_precompiled{};
        u32 audited{};
        u32 audit_differ{};
        std::chrono::steady_clock::duration gpu_time{};
        std::chrono::steady_clock::duration gpu_worst{};
        std::vector<std::pair<const char*, u32>> misses; ///< By reason (string literals)
        u64 bytes_at_window{};
        std::chrono::steady_clock::time_point window{};
    } precompiled_stats;
    void NotePrecompiledMiss(const char* reason);

    /// Builds a disk-cached graphics pipeline from the first of its records (newest first) that
    /// loads. When none does, the GPU thread's build of it is stored again.
    std::unique_ptr<GraphicsPipeline> CreateCachedGraphicsPipeline(
        const GraphicsPipelineCacheKey& key,
        std::vector<std::vector<VideoCommon::FileEnvironment>>& records,
        PipelineStatistics* statistics);

    [[nodiscard]] bool StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const;

    /// Translates the current compute program, binding the pointers in keep_pointers whatever
    /// their slots hold now (see GenericEnvironment::KeepStoragePointers)
    /// `store` false: the program is in the disk cache already (one EXP51 evicted), so no record
    /// is appended for it.
    ComputePipelineCacheEntry CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                    const ShaderInfo* shader,
                                                    std::span<const u64> keep_pointers = {},
                                                    bool store = true);

    /// A compute translation's SPIR-V and serialized info (the precompiled preparation keeps
    /// them; its audit compares them). With translate_only no pipeline is made.
    struct ComputeKeep {
        std::vector<u32> spirv;
        std::vector<u8> info;
        bool translate_only{};
        /// The program's storage descriptors, and how many have a proven read extent
        u32 storage_descriptors{};
        u32 proven_read_extents{};
    };

    ComputePipelineCacheEntry CreateComputePipeline(ShaderPools& pools,
                                                     const ComputePipelineCacheKey& key,
                                                     Shader::Environment& env,
                                                     PipelineStatistics* statistics,
                                                     bool build_in_parallel, bool warm = false,
                                                     ComputeKeep* keep = nullptr);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();
    /// Writes the driver pipeline cache on the serialization thread, unless a write is queued
    /// already (`again_if_busy`: then one more follows it, for a save that must hold the latest
    /// builds). Callable from any thread.
    void QueueVulkanPipelineCacheSave(bool again_if_busy);
    /// True (and `last` set to now) when a write of the driver cache is due since `last`.
    [[nodiscard]] bool VulkanPipelineCacheSaveDue(
        std::chrono::steady_clock::time_point& last) const;

    /// The title's warm-compute.bin (ShaderWarm::ComputeWarmPath, EDEN_NX2_COMPUTE_WARM=<file>
    /// overrides): compute programs prepared from the game files are converted (missing reads
    /// guessed) and built in the background, then thrown away. Nothing uses them; they only fill
    /// the driver's shader cache, so a dispatch that later converts to the same SPIR-V finds the
    /// driver build ready instead of waiting seconds for it.
    void StartComputeWarm(u64 title_id);
    std::vector<std::jthread> compute_warm_threads;
    std::atomic<bool> compute_warm_done{true};

    /// The title's warm-graphics.bin (ShaderWarm::GraphicsWarmPath, EDEN_NX2_GRAPHICS_WARM=<file>
    /// overrides; needs graphics pipeline libraries): vertex and fragment programs prepared from
    /// the game files are built as single-stage libraries in the background after the compute
    /// warm-up, then released; the driver's disk cache keeps them, so a later pipeline with those
    /// stages costs a link.
    void StartGraphicsWarm(u64 title_id);
    std::vector<std::jthread> graphics_warm_threads;

    /// Set as soon as emulation is asked to stop (the loading stop token): the warm-ups end
    /// after their current build instead of compiling on until the cache is destroyed, which
    /// kept shutdown waiting on driver compiles.
    std::atomic<bool> warm_cancel{false};
    std::unique_ptr<std::stop_callback<std::function<void()>>> warm_stop_callback;

    /// Waits until the warm-up may build (ShaderNotify::WarmMayBuild). False when it is to stop.
    bool WaitForWarmTurn(std::stop_token stop);

    /// EDEN_NX2_PRECOMPILE (zortos293/Lichii#895): finds or assembles the title's precompiled
    /// file (VideoCommon::Nx2Precompile) and prepares it on precompile_thread, at idle priority
    /// while the game runs. Prepare Shaders Now (ShaderNotify::RequestPrecompile) prepares it
    /// again. Started at the end of LoadDiskResources.
    void StartPrecompiled(u64 title_id, const std::filesystem::path& base_dir);
    std::jthread precompile_thread;

    /// Builds and drops every pipeline of the game's precompiled file (written offline by
    /// tools/nx2-precompile, or assembled from a recipe), so the driver's own cache holds them
    /// before they are first drawn. Draws never use these pipelines: they build their own from
    /// the guest's shaders and find the driver's compiled code. Runs once per precompiled file,
    /// driver and release (or again when `force`). Fritchii runs it on precompile_thread,
    /// so `callback` (the loading screen in #895) is empty and progress goes to ShaderNotify.
    void PreparePrecompiled(const std::filesystem::path& base_dir, std::stop_token stop_loading,
                            const VideoCore::DiskResourceLoadCallback& callback,
                            const std::filesystem::path& file = {}, bool force = false);
    /// A graphics key read from the precompiled file, as draws key it on this device: its driver
    /// fields rewritten, canonical with EDEN_NX2_CANONICAL_KEYS=1.
    void RecordedKeyForThisDevice(GraphicsPipelineCacheKey& key) const noexcept;

    /// Native pipelines: the key without state that cannot change the pipeline (see
    /// FixedPipelineState::CanonicalizeNative), with #895's attribute mask from the header. Not
    /// called: EDEN_NX2_CANONICAL_KEYS=1 canonicalizes draws, the loader and the precompiled
    /// preparation with CanonicalizeKeyState (vk_pipeline_cache.cpp), which leaves the attribute
    /// types to VideoCommon::KeyedAttributeTypes, the masking our keys had before.
    void CanonicalizeNativeKey(GraphicsPipelineCacheKey& key,
                               std::span<const Shader::Environment* const> envs) const noexcept;

    /// Native vertex programs found to load attributes their header does not list: their keys
    /// keep every attribute type (#895).
    bool IsUnmaskedVertexProgram(u64 hash) const {
        if (!any_unmasked_vertex_program.load(std::memory_order_acquire)) {
            return false;
        }
        std::scoped_lock lock{unmasked_mutex};
        return unmasked_vertex_programs.contains(hash);
    }
    mutable std::mutex unmasked_mutex;
    std::unordered_set<u64> unmasked_vertex_programs;
    std::atomic<bool> any_unmasked_vertex_program{};
    /// disk_pipelines as boot left it: the preparation skips those keys (read off the GPU
    /// thread, which keeps adding to disk_pipelines).
    std::shared_ptr<const std::unordered_set<u64>> precompile_disk_keys;

    /// The pipeline cache warm-up builds go through: none by default, so the tens of thousands
    /// of warm-up programs stay in the driver's disk cache and out of vulkan_pipelines.bin (which
    /// is loaded into memory at every boot). EDEN_NX2_WARM_APP_CACHE=1 uses ours.
    /// The cache a pipeline built in play goes to (EDEN_NX2_AMD_APP_CACHE rules).
    [[nodiscard]] vk::PipelineCache& PlayPipelineCache();
    [[nodiscard]] vk::PipelineCache& WarmPipelineCache();
    vk::PipelineCache no_pipeline_cache;

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};
    bool oot_visual_workaround{};
    u64 visual_workaround_title{};
    const size_t num_pipeline_workers;
    EmissionBudget emission_budget;

    /// EDEN_NX2_CANONICAL_KEYS=1, read by LoadDiskResources (draws look up canonical keys).
    bool canonical_keys{};

    GraphicsPipelineCacheKey graphics_key{};
    u64 vertex_attribute_types{}; ///< graphics_key's attribute types before the programs' mask
    GraphicsPipeline* current_pipeline{};

    /// EDEN_NX2_GPL: shader libraries shared by the graphics pipelines. Declared before the
    /// pipelines, so it outlives them.
    GraphicsPipelineLibraryCache graphics_libraries;

    /// Destroys graphics pipelines unused for a while, once the GPU is done with them.
    void EvictUnusedPipelines();
    u64 frame_count{};
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> evicted_pipelines; ///< (tick, pipeline)
    /// Pipelines the translate queue did not use (discarded, stale, abandoned, rejected by the
    /// verify mode), handed over on the GPU thread: destroyed by TickFrame once the GPU and
    /// their queued build are done with them, never on a translation thread (fable_answer_18
    /// 2b). Beside evicted_pipelines rather than in it: one whose build waits behind the boot
    /// prebuild would hold the evictions behind it there, as that deque is freed from the front.
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> retired_translations; ///< (tick, pipeline)
    void RetireTranslatedPipeline(std::unique_ptr<GraphicsPipeline> pipeline);
    std::unordered_set<u64> used_and_evicted; ///< Key hashes of pipelines used this session, evicted since
    /// The pointer slots of evicted graphics pipelines, for their next translation (from
    /// zortos293/Lichii#1143). Forgotten once a pipeline of the key is made again.
    VideoCommon::EvictedPointerSlots evicted_pointer_slots;

    ::Common::unordered_map<ComputePipelineCacheKey, ComputePipelineCacheEntry> compute_cache;
    /// Replaced compute pipelines, destroyed once the GPU and their queued build are done with
    /// them (tick, pipeline)
    std::deque<std::pair<u64, std::unique_ptr<ComputePipeline>>> retired_compute_pipelines;
    /// EXP51: destroys compute pipelines unused for a while (through retired_compute_pipelines).
    void EvictUnusedComputePipelines(int mode);
    /// What an evicted compute entry learned, kept so its next dispatch translates it as it was
    /// (the pointers it bound, the rebuilds it used) without a new disk cache record.
    struct EvictedCompute {
        std::vector<u64> mapped_pointers;
        u32 rebuilds{};
    };
    ::Common::unordered_map<ComputePipelineCacheKey, EvictedCompute> evicted_compute;
    size_t compute_evicted_total{};       ///< EXP51 evictions this session
    size_t compute_rebuilt_after_evict{}; ///< Evicted compute programs dispatched again
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;
    /// EDEN_NX2_ASYNC_TRANSLATE=2: the GPU thread's reference translation at enqueue uses these
    /// rather than releasing main_pools under whatever the GPU thread's own builds hold
    /// (fable_answer_18 2c, the fallback). Made on first use, only in that mode.
    std::unique_ptr<ShaderPools> verify_pools;
    std::mutex spare_pools_mutex;
    std::vector<std::shared_ptr<ShaderPools>> spare_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    /// Key hashes of the graphics pipelines in the disk cache. Only those a recent session used
    /// (used_pipelines) are built at boot; the rest are built when first needed, and not stored
    /// again. Each built pipeline holds driver memory (title 1: 6.5k cached, a quarter used).
    std::unordered_set<u64> disk_pipelines;
    /// Key hashes of disk-cached graphics pipelines none of whose records loaded (one written
    /// before a translator change can ask about slots it never classified): stored again when the
    /// GPU thread builds them.
    std::unordered_set<u64> failed_disk_pipelines;
    std::mutex failed_disk_pipelines_mutex;
    /// Disk cache records the translation rejected since the last report: the first few are
    /// logged, the rest counted (ReportDiskRecordFailures, after the boot and background builds).
    std::atomic<size_t> disk_record_failures{};
    void NoteDiskRecordFailure(std::string_view kind, u64 hash, const char* reason);
    void ReportDiskRecordFailures();
    std::unordered_map<u64, u8> used_pipelines; ///< Key hash -> sessions since it was last used
    std::unordered_set<const GraphicsPipeline*> used_this_session;
    std::filesystem::path used_pipelines_filename;
    std::chrono::steady_clock::time_point used_pipelines_saved{};
    void SaveUsedPipelines();

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    /// When MemoryOwners last asked the driver for the cache data size.
    std::chrono::steady_clock::time_point last_cache_size_query{};
    std::atomic_bool flush_in_flight{};
    std::atomic_bool save_again{};

    Common::ThreadWorker workers;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;

    /// Cached graphics pipelines boot skipped (not in used_pipelines) are built on background
    /// threads once the game runs, so a draw that needs one finds it built instead of translating
    /// it on the GPU thread. EDEN_NX2_BACKGROUND_PIPELINES=0 turns it off.
    std::unique_ptr<GraphicsPipeline> TakeBackgroundPipeline(const GraphicsPipelineCacheKey& key);
    void StopBackgroundPipelines();
    std::mutex background_mutex;
    /// Built and not yet asked for by a draw.
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> background_ready;
    /// Keys the GPU thread built itself meanwhile: their background build is skipped or dropped.
    std::unordered_set<GraphicsPipelineCacheKey> background_claimed;
    std::atomic_bool background_active{};
    std::atomic_bool background_stop{};
    std::atomic<size_t> background_left{};
    /// Numbers the background threads for the EDEN_NX2_EXP5 thread limit.
    std::atomic<size_t> background_slots{};
    size_t background_taken{}; ///< Under background_mutex
    std::chrono::steady_clock::time_point background_start{};
    std::chrono::steady_clock::time_point background_last_save{}; ///< Under background_mutex
    /// Last: destroyed (its threads joined) before the state above.
    std::unique_ptr<Common::ThreadWorker> background_workers;

    /// EDEN_NX2_ASYNC_TRANSLATE=1/2: graphics pipelines translated off the GPU thread. Null with
    /// the switch off. Reset first in the destructor (its threads build with the members above).
    std::unique_ptr<TranslateQueue> translate_queue;
    /// Keys whose translations kept failing their checks or were swept: built on the GPU thread.
    std::unordered_set<GraphicsPipelineCacheKey> translate_sync_only;
    /// Unusable translations per key (a stale texture descriptor, a guarded read that gave up).
    std::unordered_map<GraphicsPipelineCacheKey, u32> translate_failures;
    /// Counts for the 10 s line, GPU thread only.
    struct TranslateStats {
        u32 queued{};
        u32 full{};        ///< Not queued: too many in flight
        u32 adopted{};
        u32 adopted_between_frames{};
        u32 no_pipeline{};
        u32 stale{};       ///< A texture descriptor changed before adoption
        u32 gave_up{};     ///< A guarded read gave up
        u32 differs{};     ///< Verify mode: not what the GPU thread made
        u32 requeued{};
        u32 waits{};
        u32 fallbacks{};   ///< Built on the GPU thread after all
        u32 swept{};
        u64 skipped{};     ///< Lookups skipped while translating
        size_t snapshot_bytes{};
        std::chrono::steady_clock::duration enqueue_time{};
        std::chrono::steady_clock::duration wait_time{};
        std::chrono::steady_clock::duration translation_time{};
        std::chrono::steady_clock::duration ready_total{};
        std::chrono::steady_clock::duration ready_max{};
        std::chrono::steady_clock::time_point window{};
    } translate_stats;
};

} // namespace Vulkan
