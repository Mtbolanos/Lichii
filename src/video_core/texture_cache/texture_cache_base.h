// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
// TODO: find out which don't require stable iters
#include <unordered_map>
#include "video_core/page_filter.h"
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <queue>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/hash.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include <ranges>
#include "common/scratch_buffer.h"
#include "common/slot_vector.h"
#include "video_core/texture_cache/pending_download.h"
#include "common/thread_worker.h"
#include "video_core/compatible_formats.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/nx2.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/descriptor_table.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/render_targets.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"
#include "video_core/textures/unswizzle_worker.h"

namespace Tegra {
namespace Control {
struct ChannelState;
}
} // namespace Tegra

namespace VideoCommon {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

struct ImageViewInOut {
    u32 index{};
    bool blacklist{};
    ImageViewId id{};
};

struct AsyncDecodeContext {
    ImageId image_id;
    Common::ScratchBuffer<u8> decoded_data;
    boost::container::small_vector<BufferImageCopy, 16> copies;
    std::mutex mutex;
    std::atomic_bool complete;
    bool cancelled{}; ///< The image was deleted meanwhile: drop the result (GPU thread only).
};

using TextureCacheGPUMap = ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>;

/// The current draw's render targets as the asynchronous shaders' skip rule sees them
/// (TextureCache::ClassifyDrawTargets): the class of the draw, and the target that gave it (the
/// first of the draw's least safe class; address 0 for a draw without targets).
struct DrawTargets {
    DrawTargetClass kind{DrawTargetClass::Recurring};
    GPUVAddr gpu_addr{};
    u32 width{};
    u32 height{};
    VideoCore::Surface::PixelFormat format{};
};

class TextureCacheChannelInfo : public ChannelInfo {
public:
    TextureCacheChannelInfo() = delete;
    TextureCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept;
    TextureCacheChannelInfo(const TextureCacheChannelInfo& state) = delete;
    TextureCacheChannelInfo& operator=(const TextureCacheChannelInfo&) = delete;

    DescriptorTable<TICEntry, ImageViewId> graphics_image_table;
    DescriptorTable<TSCEntry, SamplerId> graphics_sampler_table;
    DescriptorTable<TICEntry, ImageViewId> compute_image_table;
    DescriptorTable<TSCEntry, SamplerId> compute_sampler_table;

    // TODO: still relies on bad iterators :(
    std::unordered_map<TICEntry, ImageViewId> image_views;
    std::unordered_map<TSCEntry, SamplerId> samplers;

    TextureCacheGPUMap* gpu_page_table = nullptr;
    TextureCacheGPUMap* sparse_page_table = nullptr;
};

template <class P>
class TextureCache : public VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo> {
    /// Address shift for caching images into a hash table
    static constexpr u64 YUZU_PAGEBITS = 20;

    /// Enables debugging features to the texture cache
    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    /// Implement blits as copies between framebuffers
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;
    /// True when some copies have to be emulated
    static constexpr bool HAS_EMULATED_COPIES = P::HAS_EMULATED_COPIES;
    /// True when the API can provide info about the memory of the device.
    static constexpr bool HAS_DEVICE_MEMORY_INFO = P::HAS_DEVICE_MEMORY_INFO;
    /// True when the API can do asynchronous texture downloads.
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = P::IMPLEMENTS_ASYNC_DOWNLOADS;

    static constexpr size_t UNSET_CHANNEL{(std::numeric_limits<size_t>::max)()};

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    // Periodic statistics (TickFrame).
    u64 gc_deletes{};
    u64 gc_costly_deletes{};
    u64 gc_download_deletes{}; ///< Evictions that downloaded the image first (a full GPU wait)
    u64 gc_downloads_deferred{}; ///< EXP57: downloads left for a later frame (one wait a frame)
    u64 gc_waited_frame{~0ULL};  ///< EXP57: the frame of the collector's last download wait
    u64 async_decodes_queued{};
    u64 images_created{};
    u64 images_uploaded{};   ///< Image contents uploaded from guest memory (last 600 frames)
    u64 uploaded_bytes{};    ///< Guest bytes those uploads read
    u64 largest_upload{};    ///< Largest of them
    u64 layer_uploads{};     ///< Of them, uploads of only the written layers
    bool native_thresholds_applied{};
    u64 native_critical_memory{};
    u64 native_expected_memory{};
    u64 budget_check_frames{};
    u32 seen_pressure_events{};   ///< Allocator pressure events already answered
    u32 seen_failed_images{};     ///< Images the allocator could not make, already collected for
    u64 pressure_margin{};        ///< Extra room under the budget after images did not fit
    u64 pressure_quiet_frames{};  ///< Frames since the last such image
    u64 slack_margin{};           ///< Extra room for allocator block space the driver counts as used
    u64 astc_opaque_images{};
    u64 astc_alpha_images{};
    std::vector<u8> astc_scan_buffer;
    u64 images_deleted{};
    std::map<u64, std::pair<u32, GPUVAddr>> created_shapes;
    bool debug_stats{true}; ///< Collect what only the debug statistics lines use
    /// EDEN_NX2_MEM_TRACE: when the last memory trace lines were logged.
    std::chrono::steady_clock::time_point mem_trace_time{};

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 1_GiB + 125_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB + 625_MiB;
    /// The most device memory the collector lets the cache fill before it collects in earnest on
    /// cards up to 8 GiB; larger ones 5/8 of the driver's budget (ApplyThresholdOverrides).
    static constexpr u64 MaxCriticalMemory = 5_GiB;
    static constexpr size_t GC_EMERGENCY_COUNTS = 2;

    using Runtime = typename P::Runtime;
    using Image = typename P::Image;
    using ImageAlloc = typename P::ImageAlloc;
    using ImageView = typename P::ImageView;
    using Sampler = typename P::Sampler;
    using Framebuffer = typename P::Framebuffer;
    using AsyncBuffer = typename P::AsyncBuffer;
    using BufferType = typename P::BufferType;

    static constexpr u32 LayerUploadMinSize = 64U << 20;
    static constexpr size_t LargeScratchCapacity = 64U << 20;

    /// Splits large 2D image uploads into smaller chunks, avoiding multi-hundred MiB staging buffers.
    bool UploadInChunks(Image& image);

    /// Large layered images whose refresh can decode only the layers written since the last one.
    [[nodiscard]] bool UploadsByLayer(const ImageBase& image) const noexcept;

    /// Records the layers a write of [cpu_addr, +size) touches.
    void MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: notes a written range, uploaded with the next refresh.
    void MarkRangeWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: binds and uploads the parts written since the last upload.
    void RefreshSparseTiles(Image& image);

    /// Image::BindTiles, counting what it binds in total_used_memory.
    bool BindSparseTiles(Image& image, std::span<const LevelRect> rects);

    /// Image::BackTilesWhole, counting what it binds in total_used_memory.
    bool BackSparseTilesWhole(Image& image);

    /// Uploads the image's dirty layers. False when it needs a full upload instead.
    bool UploadWrittenLayers(Image& image);

    /// Gives the written layers of a SparseLayers image memory and uploads them.
    void RefreshSparseLayers(Image& image);

    /// Image::BindLayers, with the memory it binds counted in total_used_memory.
    std::vector<u32> BindSparseLayers(Image& image, std::span<const u32> layers);

    /// Uploads `layers` of the image from guest memory.
    void UploadLayers(Image& image, std::span<const u32> layers);

    struct PendingUnswizzle {
        ImageId image_id;
        VideoCommon::ImageInfo info;
        size_t current_offset = 0;
        size_t total_size = 0;
        AsyncBuffer staging_buffer;
        size_t last_submitted_offset = 0;
        size_t bytes_per_slice;
        bool initialized = false;
    };

    struct BlitImages {
        ImageId dst_id;
        ImageId src_id;
        PixelFormat dst_format;
        PixelFormat src_format;
    };

public:
    /// Device pages this cache ever registered objects on (InnerInvalidation skips the rest).
    VideoCommon::PageFilter page_filter;

    explicit TextureCache(Runtime&, Tegra::MaxwellDeviceMemoryManager&);

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Return a constant reference to the given image view id
    [[nodiscard]] const ImageView& GetImageView(ImageViewId id) const noexcept;

    /// Return a reference to the given image view id
    [[nodiscard]] ImageView& GetImageView(ImageViewId id) noexcept;

    /// Get the imageview from the graphics descriptor table in the specified index
    [[nodiscard]] ImageView& GetImageView(u32 index) noexcept;

    /// Mark an image as modified from the GPU
    void MarkModification(ImageId id) noexcept;

    /// Fill image_view_ids with the graphics/compute images in indices
    void FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist = true);

    /// Handle feedback loops during draws.
    void CheckFeedbackLoop(std::span<const ImageViewInOut> views);

    [[nodiscard]] bool SamplesRenderTarget(std::span<const ImageViewInOut> views);

    /// Get the sampler from the graphics/compute descriptor table in the specified index
    Sampler* GetSampler(u32 index, bool compute);

    /// Get the sampler id from the graphics/compute descriptor table in the specified index
    SamplerId GetSamplerId(u32 index, bool compute);

    /// Return a constant reference to the given sampler id
    [[nodiscard]] const Sampler& GetSampler(SamplerId id) const noexcept;

    /// Return a reference to the given sampler id
    [[nodiscard]] Sampler& GetSampler(SamplerId id) noexcept;

    /// Refresh the state for graphics/compute image view and sampler descriptors
    void SynchronizeDescriptors(bool compute);

    /// Updates the Render Targets if they can be rescaled
    /// @retval True if the Render Targets have been rescaled.
    bool RescaleRenderTargets();

    /// Update bound render targets and upload memory if necessary
    /// @param is_clear True when the render targets are being used for clears
    void UpdateRenderTargets(bool is_clear);

    /// Find a framebuffer with the currently bound render targets
    /// UpdateRenderTargets should be called before this
    Framebuffer* GetFramebuffer();

    [[nodiscard]] const RenderTargets& GetRenderTargets() const noexcept {
        return render_targets;
    }

    /// EDEN_NX2_ASYNC_DISCARD: classifies the render targets the current registers bind (the
    /// bound ones when they are current, otherwise the registered images at the registers'
    /// addresses; nothing is created). Takes the cache's lock. Meant for a draw whose pipeline is
    /// missing, so it runs per miss, not per draw.
    [[nodiscard]] DrawTargets ClassifyDrawTargets();

    /// Mark images in a range as modified from the CPU
    void WriteMemory(DAddr cpu_addr, size_t size);

    /// Download contents of host images to guest memory in a region
    void DownloadMemory(DAddr cpu_addr, size_t size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr cpu_addr, u64 size);

    /// Remove images in a region
    void UnmapMemory(DAddr cpu_addr, size_t size);

    /// Remove images in a region
    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// Blit an image with the given parameters
    bool BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                   const Tegra::Engines::Fermi2D::Surface& src,
                   const Tegra::Engines::Fermi2D::Config& copy);

    /// Stand-in for a DLSS evaluation: temporal upscale of src into dst (see the runtime's
    /// TemporalUpscale). False when unsupported; the caller retains its native dispatch.
    bool TemporalUpscale(const Tegra::Engines::Fermi2D::Surface& dst,
                         const Tegra::Engines::Fermi2D::Surface& src,
                         const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                         const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
                         std::array<f32, 2> motion_scale, bool use_history);

    /// Health check of a game's DLSS network output (dst) against its input (src); see the
    /// runtime's ProbeGameDlssOutput. Returns an earlier check's result when one is ready.
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src);

    /// Try to find a cached image view in the given CPU address
    [[nodiscard]] std::pair<ImageView*, bool> TryFindFramebufferImageView(
        const Tegra::FramebufferConfig& config, DAddr cpu_addr);

    /// Return true when there are uncommitted images to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    /// Whether CommitAsyncFlushes would commit anything (downloads or their buffers).
    [[nodiscard]] bool HasAsyncFlushWork() const noexcept;

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();

    [[nodiscard]] ImageId DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload);

    [[nodiscard]] std::pair<Image*, BufferImageCopy> DmaBufferImageCopy(
        const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
        const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image);

    void DownloadImageIntoBuffer(Image* image, BufferType buffer, size_t buffer_offset,
                                 std::span<const VideoCommon::BufferImageCopy> copies,
                                 GPUVAddr address = 0, size_t size = 0);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    [[nodiscard]] bool IsRescaling() const noexcept;

    [[nodiscard]] bool IsRescaling(const ImageViewBase& image_view) const noexcept;

    /// Create channel state.
    void CreateChannel(Tegra::Control::ChannelState& channel) final override;

    /// Prepare an image to be used
    void PrepareImage(ImageId image_id, bool is_modification, bool invalidate);

    /// EDEN_NX2_EXP32 (texture binding memo, after zortos293/Lichii#717's BINDMEMO): what the
    /// graphics texture bindings of a draw are resolved against. With the same state, the same
    /// handles and the same guest TIC/TSC entries, VisitImageView and GetSamplerId give the same
    /// view and sampler ids: a view leaves a descriptor only through RemoveImageViewReferences
    /// (image deleted, joined or rescaled), which moves `epoch`, as do a new channel and the
    /// collector's out of memory table reset (null views made again); null views of unmapped
    /// addresses follow the mappings (`map_serial`). Samplers are never deleted.
    struct TextureMemoState {
        const void* channel{};
        const void* memory{};
        GPUVAddr tic_pool{};
        GPUVAddr tsc_pool{};
        u32 tic_limit{};
        u32 tsc_limit{};
        u64 map_serial{};
        u64 epoch{};

        [[nodiscard]] bool operator==(const TextureMemoState&) const noexcept = default;
    };

    /// One texture binding's guest entries as the draw that resolved it read them.
    struct TextureMemoEntry {
        u32 tic_index{}; ///< ~0U: the handle is out of the pool (null view whatever memory holds)
        u32 tsc_index{}; ///< ~0U: out of the pool (null sampler)
        TICEntry tic{};
        TSCEntry tsc{};
    };

    [[nodiscard]] TextureMemoState GraphicsTextureMemoState() const noexcept;

    /// The entries the graphics tables resolved `tic_handle` and `tsc_handle` from in this draw;
    /// false when the tables no longer hold them as resolved to `view` and `sampler`.
    [[nodiscard]] bool TakeTextureMemoEntry(u32 tic_handle, u32 tsc_handle, ImageViewId view,
                                            SamplerId sampler, TextureMemoEntry& entry) const;

    /// Whether guest memory still holds the entry's TIC and TSC entries (read as
    /// DescriptorTable::Read reads them).
    [[nodiscard]] bool TextureMemoEntryCurrent(const TextureMemoState& state,
                                               const TextureMemoEntry& entry) const;

    /// Whether preparing the view for sampling (PrepareImageView(view, false, false)) would do
    /// nothing but touch its image in the LRU: no CPU modification to upload, no aliases.
    [[nodiscard]] bool TextureMemoViewClean(ImageViewId view) const noexcept;

    /// What preparing the view does once its contents are clean: the image's LRU touch.
    void TouchTextureMemoView(ImageViewId view) noexcept;

    [[nodiscard]] u64 FrameTick() const noexcept {
        return frame_tick;
    }

    /// What a clear's render target lookups did (RasterizerVulkan::Clear's "Slow clear" line):
    /// ResetRenderTargetStats before UpdateRenderTargets, TakeRenderTargetStats after
    /// GetFramebuffer. Counts and times of JoinImages, RemoveFramebuffers and GetFramebufferId.
    struct RenderTargetStats {
        u64 images_created{};
        u64 images_deleted{};
        u64 uploaded_bytes{};
        u32 joins{};
        u32 overlaps_walked{};      ///< Images region_check looked at
        u32 bad_overlaps{};
        u32 framebuffers_removed{};
        u32 framebuffers_scanned{}; ///< Framebuffer map entries RemoveFramebuffers looked at
        u32 framebuffers_created{};
        u32 pooled_images{};        ///< New images that took a pooled VkImage (EXP42)
        u32 relocated_images{};     ///< Images moved instead of made (EXP43=2)
        u32 retired_images{};       ///< Predecessors deleted at once (EXP43=1)
        u64 walk_ns{};              ///< JoinImages' overlap walk
        u64 create_ns{};            ///< Making the new image (vkCreateImage, memory)
        u64 refresh_ns{};           ///< Its upload (RefreshContents) or layout transition
        u64 delete_ns{};            ///< Deleting the images it replaced
        u64 framebuffer_ns{};       ///< New framebuffers
        u64 framebuffer_map_size{}; ///< At TakeRenderTargetStats
        u64 image_count{};
    };
    void ResetRenderTargetStats() noexcept {
        rt_stats = {};
        rt_stats_base = {images_created, images_deleted, uploaded_bytes};
    }
    [[nodiscard]] RenderTargetStats TakeRenderTargetStats() const noexcept {
        RenderTargetStats stats = rt_stats;
        const auto delta = [](u64 now, u64 base) { return now >= base ? now - base : now; };
        stats.images_created = delta(images_created, rt_stats_base[0]);
        stats.images_deleted = delta(images_deleted, rt_stats_base[1]);
        stats.uploaded_bytes = delta(uploaded_bytes, rt_stats_base[2]);
        stats.framebuffer_map_size = framebuffers.size();
        stats.image_count = slot_images.size();
        return stats;
    }

    Common::SpinRecursiveMutex mutex;

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachCPUPage(DAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    template <typename Func>
    static void ForEachGPUPage(GPUVAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    void OnGPUASRegister(size_t map_id) final override;

    /// Runs the Garbage Collector.
    void RunGarbageCollector(bool out_of_memory);

    /// Find or create an image view in the guest descriptor table
    ImageViewId VisitImageView(u32 index, bool compute);

    /// Find or create a framebuffer with the given render target parameters
    FramebufferId GetFramebufferId(const RenderTargets& key);

    /// Refresh the contents (pixel data) of an image
    [[nodiscard]] bool IsDownloadable(const ImageBase& image) const noexcept;

    void RefreshContents(Image& image, ImageId image_id);

    /// Upload data from guest to an image
    template <typename StagingBuffer>
    void UploadImageContents(Image& image, StagingBuffer& staging_buffer);

    /// Create a new image view from a guest descriptor
    [[nodiscard]] ImageViewId CreateImageView(const TICEntry& config);

    /// Find or create an image from the given parameters
    [[nodiscard]] ImageId FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                            RelaxedOptions options = RelaxedOptions{});

    /// Find an image from the given parameters
    [[nodiscard]] ImageId FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                    RelaxedOptions options);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                      RelaxedOptions options);

    /// Create a new image and join perfectly matching existing images
    /// Remove joined images from the cache
    [[nodiscard]] ImageId JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr);

    [[nodiscard]] ImageId FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Return a blit image pair from the given guest blit parameters
    [[nodiscard]] std::optional<BlitImages> GetBlitImages(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Config& copy);

    /// Find or create a sampler from a guest descriptor sampler
    [[nodiscard]] SamplerId FindSampler(const TSCEntry& config, bool compute);

    /// Find or create an image view for the given color buffer index
    [[nodiscard]] ImageViewId FindColorBuffer(size_t index);

    /// Find or create an image view for the depth buffer
    [[nodiscard]] ImageViewId FindDepthBuffer();

    /// Find or create a view for a render target with the given image parameters
    [[nodiscard]] ImageViewId FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachSparseSegment(ImageBase& image, Func&& func);

    /// Find or create an image view in the given image with the passed parameters
    [[nodiscard]] ImageViewId FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Track CPU reads and writes for image
    void TrackImage(ImageBase& image, ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageBase& image, ImageId image_id);

    /// Delete image from the cache
    void DeleteImage(ImageId image, bool immediate_delete = false);

    /// Remove image views references from the cache
    void RemoveImageViewReferences(std::span<const ImageViewId> removed_views);

    /// Per image view, the (channel, descriptor) entries of the channels' image_views maps that
    /// point to it. Deleting an image erased them by scanning every map (title 1: ~5,000 entries,
    /// 50-130 us for each deleted image, dozens a frame while the collector runs).
    std::unordered_map<ImageViewId, boost::container::small_vector<std::pair<size_t, TICEntry>, 2>>
        image_view_descriptors;

    /// Remove framebuffers using the given image views from the cache
    void RemoveFramebuffers(std::span<const ImageViewId> removed_views);

    /// Render target relocation (EXP43, fable_answer_12 2A). Per shape (format, size, samples,
    /// levels, layers), sticky for the session: a shape whose GPU-written contents were read in a
    /// later frame (sampled, rendered to without a clear) or read by the CPU never relocates or
    /// retires early, since its predecessor's contents may still be wanted.
    enum RelocationGuard : u8 {
        GuardSampled = 1,  ///< Sampled or copied from in a frame after the GPU wrote it
        GuardLoaded = 2,   ///< Rendered to without a clear in a frame after the GPU wrote it
        GuardCpuRead = 4,  ///< Read by the CPU (GetFlushArea, DownloadMemory)
    };
    struct RelocationShape {
        u8 guard{};
        bool candidate_logged{};
        bool relocated_logged{};
        u32 same_frame_refused{}; ///< Since the last report (render-frame clock line)
    };
    ::Common::unordered_map<u64, RelocationShape> relocation_shapes;
    /// The guard by address too: OoT draws every dialogue box's text into a render target at one
    /// address, sized to the box. Per shape, each new box size was relocated once before its
    /// guard tripped, and that box's text showed other images' contents (2026-11-09, the owner).
    ::Common::unordered_map<GPUVAddr, u8> relocation_guarded_addresses;
    /// Since the last report (TickFrame, every 600 frames once a candidate was seen).
    u64 relocation_candidates{};
    u64 relocation_guarded{};
    u64 relocation_relocated{};
    u64 relocation_retired{};
    u64 relocation_crowded{}; ///< Cleared targets over two or more images of their shape
    u64 relocation_same_frame{}; ///< Candidates whose only predecessors were written this frame
    bool relocation_seen{};

    /// The frame last_write_frame, CheckCrossFrameUse and RelocationCandidate's same-frame rule
    /// count in. frame_tick advances at the composite, a sync request the back serves wherever it
    /// is, so one host tick holds the end of one guest frame and the start of the next
    /// (fable_answer_16 1a). With EDEN_NX2_RENDER_CLOCK=1 it follows Tegra::g_nx2_render_frame
    /// instead (bumped where the frame's last GPU work is), falling back to frame_tick while that
    /// does not advance (no word learned yet, other titles). Monotonic across the switches; the
    /// LRU, the collector and the logs keep frame_tick.
    [[nodiscard]] u64 RenderFrame() const noexcept {
        const u64 source = render_clock_live
                               ? Tegra::g_nx2_render_frame.load(std::memory_order_relaxed)
                               : frame_tick;
        return render_clock_base + (source - render_clock_source_base);
    }
    /// TickFrame: follows the counter, or falls back to frame_tick after two host frames
    /// without a count; logs the rate next to the relocation line.
    void TickRenderClock();
    bool render_clock_live{};
    bool render_clock_enabled{}; ///< Last tick: the clock wanted (Resume on the way back on)
    bool render_clock_logged{};
    u64 render_clock_base{};
    u64 render_clock_source_base{};
    u64 render_clock_last_counter{};
    u64 render_clock_window_counter{}; ///< The counter at the last report
    u32 render_clock_idle{};           ///< Host frames since the counter last advanced
    u32 render_clock_live_frames{};    ///< Since the last report
    u32 render_clock_switches{};

    /// EXP43=4 (fable_answer_16 1c): relocation as 3, under the render-frame clock, and each
    /// relocated image's old range kept for 2 render frames. A read that lands on it (an image
    /// made at the old address and then sampled or rendered to without a clear, or a CPU read of
    /// the old range while the image there was GPU-written) is what relocation would have gotten
    /// wrong: counted and its first few logged. Must stay 0.
    struct RelocationTombstone {
        GPUVAddr gpu_addr;
        DAddr cpu_addr;
        u64 size;
        u64 shape_key;
        u64 render_frame;
        GPUVAddr moved_to;
        bool gpu_modified;
    };
    enum TombstoneRead : u8 { TombstoneSampled, TombstoneLoaded, TombstoneCpuRead, TombstoneReads };
    void AddRelocationTombstone(const ImageBase& old_image, GPUVAddr moved_to);
    /// JoinImages: a new image at a tombstoned address waits for its first PrepareImage.
    void CheckTombstoneCreation(ImageId image_id, const ImageBase& image, bool overwritten);
    void CheckTombstonePrepare(ImageId image_id, const ImageBase& image, bool is_modification,
                               bool invalidate);
    void CheckTombstoneCpuRead(DAddr cpu_addr, u64 size);
    void NoteTombstoneHit(const RelocationTombstone& tombstone, TombstoneRead reason,
                          GPUVAddr gpu_addr, const ImageInfo* info);
    /// TickFrame: drops tombstones older than 2 render frames; logs every 600 frames.
    void TickRelocationTombstones();
    std::vector<RelocationTombstone> relocation_tombstones;
    DAddr tombstone_cpu_begin{~DAddr{0}}; ///< Bounds of all tombstones (quick reject)
    DAddr tombstone_cpu_end{};
    struct TombstoneBorn {
        ImageId image_id;
        GPUVAddr gpu_addr;
        RelocationTombstone tombstone;
    };
    boost::container::small_vector<TombstoneBorn, 4> tombstone_born;
    std::array<u64, TombstoneReads> tombstone_hits{}; ///< Since the last report
    u64 tombstones_made{};
    u64 tombstone_overlap_reads{}; ///< New images read over one, not at its address
    u32 tombstone_hits_logged{};

    /// Cold sweep (EDEN_NX2_TC_COLD_FRAMES=K, 0 = off; fable_answer_16 3.2): a relocation-shaped
    /// image with BadOverlap (a member of a drifted chain) untouched for K host frames, with no
    /// download queued and not bound, is deleted in TickFrame (at most 16 a frame), so its
    /// VkImage goes to the EXP42 pool instead of waiting for the collector, which only starts
    /// near the device budget. Never a GPU-written image that may be read again: one that was
    /// itself read in a later frame or by the CPU, or of a shape guarded as read by the CPU, and
    /// unless EDEN_NX2_TC_COLD_GUARDED=1 of a shape guarded as sampled or rendered to without a
    /// clear in a later frame. JoinImages puts the chain members it sees on the watch list.
    void WatchColdImage(ImageId image_id);
    void SweepColdImages();
    std::vector<std::pair<ImageId, u64>> cold_watch; ///< Image, frame_tick it was put on
    ::Common::unordered_set<ImageId> cold_watch_set;
    u64 cold_deleted{};       ///< Since the last report
    /// Images made since the last cold sweep report, per (format, width), with their heights
    /// (fable52_answer_13 section 6: in the Stalchild fight new render target sizes one pixel
    /// apart, 528x513 to 528x527, kept making images). Logged with the cold sweep line.
    struct MadeShapes {
        u64 made{};
        std::vector<u32> heights; ///< Distinct, at most 64 kept
    };
    std::map<std::pair<u32, u32>, MadeShapes> made_shapes;
    void NoteMadeShape(const ImageInfo& info);
    u64 cold_kept_guarded{};  ///< Dropped from the watch: GPU-written and guarded
    u64 cold_superseded{};    ///< EXP58: guarded but superseded and old, deleted
    /// EXP58: whether a newer render target of another shape covers this image's memory.
    [[nodiscard]] bool ColdSuperseded(ImageId image_id, const ImageBase& image);
    u64 cold_kept_read{};     ///< Dropped from the watch: GPU-written and read later or by the CPU

    [[nodiscard]] static bool RelocationShaped(const ImageInfo& info) noexcept {
        return info.type == ImageType::e2D && info.resources.levels == 1 &&
               info.resources.layers == 1 && info.num_samples == 1;
    }
    [[nodiscard]] static u64 RelocationShapeKey(const ImageInfo& info) noexcept {
        return (u64{static_cast<u32>(info.format) & 0xFF} << 56) |
               (u64{info.num_samples & 0xF} << 52) |
               (u64{static_cast<u32>(info.resources.levels) & 0xF} << 48) |
               (u64{static_cast<u32>(info.resources.layers) & 0xFFF} << 36) |
               (u64{info.size.width & 0x3FFFF} << 18) | (info.size.height & 0x3FFFF);
    }
    /// JoinImages' only bad overlap, when the new render target could take its place: the same
    /// single-level, single-layer 2D shape and layout, a drift of less than its size, nothing
    /// unusual about it (sparse, converted, aliased, remapped, rescaled, CPU-read, a download
    /// queued, bound as another target of this clear, registered in another channel). Else none.
    [[nodiscard]] ImageId RelocationCandidate(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                              DAddr cpu_addr);
    /// EXP43=2: moves the image to gpu_addr / cpu_addr in place of a new one (fable_answer_12 2A
    /// steps 1-6); its VkImage, memory, views, framebuffers and scale state stay.
    void RelocateImage(ImageId image_id, GPUVAddr gpu_addr, DAddr cpu_addr);
    /// Drops the image's overlapping_images links from the other images (DeleteImage, relocation).
    void DropOverlapLinks(ImageId image_id);
    /// Sets `reason` on the image's shape (once per image and reason; logs the shape's first).
    void NoteRelocationGuard(ImageBase& image, RelocationGuard reason);
    /// The cross-frame read checks of PrepareImage and the texture memo's touch.
    void CheckCrossFrameUse(ImageBase& image, bool is_modification, bool invalidate) {
        if (image.last_write_frame == ~u64{0} || invalidate ||
            image.last_write_frame == RenderFrame()) {
            return;
        }
        const RelocationGuard reason = is_modification ? GuardLoaded : GuardSampled;
        if ((image.relocation_guard_noted & reason) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, reason);
        }
    }

    /// Drops a framebuffer from framebuffers_by_view and framebuffer_keys.
    void UnindexFramebuffer(FramebufferId framebuffer_id, const RenderTargets& key);

    /// The framebuffers each image view is in, and each framebuffer's key: RemoveFramebuffers
    /// takes a deleted image's framebuffers from here instead of scanning the whole map, which
    /// grows by one entry per drifted render target a frame (fable_answer_12 2E).
    ::Common::unordered_map<ImageViewId, boost::container::small_vector<FramebufferId, 4>>
        framebuffers_by_view;
    ::Common::unordered_map<FramebufferId, RenderTargets> framebuffer_keys;

    /// Mark an image as modified from the GPU
    void MarkModification(ImageBase& image) noexcept;

    /// Synchronize image aliases, copying data if needed
    void SynchronizeAliases(ImageId image_id);

    /// Prepare an image view to be used
    void PrepareImageView(ImageViewId image_view_id, bool is_modification, bool invalidate);

    /// Execute copies from one image to the other, even if they are incompatible
    void CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies);

    /// Bind an image view as render target, downloading resources preemtively if needed
    void BindRenderTarget(ImageViewId* old_id, ImageViewId new_id);

    /// Create a render target from a given image and image view parameters
    [[nodiscard]] std::pair<FramebufferId, ImageViewId> RenderTargetFromImage(
        ImageId, const ImageViewInfo& view_info);

    /// Returns true if the current clear parameters clear the whole image of a given image view
    [[nodiscard]] bool IsFullClear(ImageViewId id);

    /// UpdateRenderTargets(true): records a full clear on the targets the clear writes (the
    /// colour target it names, the depth target when depth is cleared), for ClassifyDrawTargets.
    void NoteFullClears();

    [[nodiscard]] std::pair<u32, u32> PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                      bool mark_as_modified);

    bool ImageCanRescale(ImageBase& image);
    void InvalidateScale(Image& image);
    bool ScaleUp(Image& image);
    bool ScaleDown(Image& image);
    u64 GetScaledImageSizeBytes(const ImageBase& image);

    /// The device memory an image counts for in total_used_memory between two driver readings:
    /// none for an unbacked image (too large to back, or over unwritten sparse blocks), what is
    /// bound to its layers for a SparseLayers one.
    u64 ImageMemoryBytes(const Image& image) const;

    void QueueAsyncDecode(Image& image, ImageId image_id);
    void TickAsyncDecode();

    void QueueAsyncUnswizzle(Image& image, ImageId image_id);
    void TickAsyncUnswizzle();

    Runtime& runtime;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    std::deque<TextureCacheGPUMap> gpu_page_table_storage;

    /// EDEN_NX2_EXP31 (RT_FAST, zortos293/Lichii#717): whether UpdateRenderTargets' clean path
    /// would change nothing: the same targets were prepared for a draw, no image was modified,
    /// created or deleted since, in the same frame, and none needs an upload, alias sync or
    /// sparse backing.
    [[nodiscard]] bool RenderTargetsStillPrepared() const;
    void NoteRenderTargetsPrepared();
    /// EXP31 bit 2: the clean path ran although RenderTargetsStillPrepared; checks it changed
    /// nothing but the targets' own modification ticks. False (and logs) on a difference.
    bool VerifyRenderTargetsPrepared();

    RenderTargets render_targets;
    u64 render_targets_serial = 0;
    struct RenderTargetsPrepared {
        bool valid = false;
        u64 serial = 0;
        u64 modification_tick = 0;
        u64 frame_tick = 0;
        u64 bindings_epoch = 0;
        std::array<ImageViewId, NUM_RT + 1> views{};
    } rt_prepared;
    /// EXP31 counts since the last report (TickFrame, every 600 frames while it is on).
    u64 rt_fast_skips = 0;
    u64 rt_fast_clean_paths = 0;
    u64 rt_fast_verified = 0;
    u64 rt_fast_mismatches = 0;
    /// Changes whenever a prepared render target could need preparing again: images or views
    /// created, deleted, rescaled or CPU modified, descriptor pools moved, and every frame
    /// (zortos293/Lichii#717's bindings_epoch).
    u64 bindings_epoch = 1;
    /// See TextureMemoState::epoch (EXP32). 0 never matches.
    u64 texture_memo_epoch = 1;
    u32 rt_active_mask = 0;
    std::array<ImageId, 8> rt_image_id{};
    ImageId rt_depth_image_id{};
    u64 texture_bindings_serial = 0;
    u64 last_feedback_loop_serial = 0;
    u64 last_feedback_texture_serial = 0;
    bool last_feedback_loop_result = false;
    bool feedback_depth_dirty = true;
    FramebufferId last_framebuffer_id{};
    u64 last_framebuffer_serial = 0;

    ::Common::unordered_map<RenderTargets, FramebufferId> framebuffers;
    ::Common::unordered_map<u64, std::vector<ImageMapId>, Common::IdentityHash<u64>> page_table;
    ::Common::unordered_map<ImageId, boost::container::small_vector<ImageViewId, 16>> sparse_views;

    DAddr virtual_invalid_space{};

    /// Applies EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB and logs the thresholds.
    void ApplyThresholdOverrides(bool log = true);

    /// A GPU-written image dropped without being written back leaves its
    /// guest memory stale (the GPU's data is gone). Its whole pages are marked here and kept
    /// tracked, so CPU writes, unmaps and write-backs clear them. An image made entirely on marked
    /// pages starts without an upload: the upload would only copy stale bytes the GPU overwrites.
    void MarkStale(const ImageBase& image);
    void ClearStale(DAddr addr, size_t size);
    void ClearStaleImage(const ImageBase& image);
    /// Clears every device range [gpu_addr, gpu_addr + size) maps to.
    void ClearStaleGpu(GPUVAddr gpu_addr, size_t size);
    [[nodiscard]] bool IsAllStale(DAddr addr, size_t size) const;
    void UpdateStaleTracking(u64 run_begin, u64 run_pages, s32 delta);
    // Off here unless EDEN_NX2_TC_SKIP_STALE=1: no gain at an Ocarina of Time Hyrule Field save
    // (fresh launches, walking 1% low 40.2 off vs 36.5 on); #463 measured +20% in Castle Town.
    bool tc_skip_stale{[] {
        const char* value = std::getenv("EDEN_NX2_TC_SKIP_STALE");
        return value != nullptr && value[0] == '1';
    }()};
    /// Device memory pages (Core::Memory::YUZU_PAGEBITS), not this cache's 1 MiB lookup pages.
    static constexpr u64 STALE_PAGE_BITS = 12;
    std::vector<u64> stale_bits; ///< One bit per device page
    u64 stale_page_count{};
    u64 tc_stale_skips{};        ///< Uploads skipped (last 600 frames)
    u64 tc_stale_skip_bytes{};
    u64 tc_stale_gpu_written{};  ///< Stale but buffer-written pages: uploaded after all

    /// The CPU address an image at gpu_addr is keyed by: its backing, or a fixed stand-in.
    [[nodiscard]] std::optional<DAddr> ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const;

    bool has_deleted_images = false;
    /// True while a clear binds its render targets (UpdateRenderTargets(true)), so JoinImages can
    /// skip the upload of a new target the clear overwrites whole.
    bool clearing_render_targets = false;
    bool is_rescaling = false;
    u64 total_used_memory = 0;
    u64 minimum_memory;
    u64 expected_memory;
    u64 critical_memory;
    size_t gpu_unswizzle_maxsize = 0;
    size_t swizzle_chunk_size = 0;
    u32 swizzle_slices_per_batch = 0;

    struct BufferDownload {
        GPUVAddr address;
        size_t size;
    };

    using PendingDownload = PendingImageDownload<Image>;

    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageMapView> slot_map_views;
    Common::SlotVector<ImageView> slot_image_views;
    Common::SlotVector<ImageAlloc> slot_image_allocs;
    Common::SlotVector<Sampler> slot_samplers;
    Common::SlotVector<Framebuffer> slot_framebuffers;
    Common::SlotVector<BufferDownload> slot_buffer_downloads;

    // TODO: This data structure is not optimal and it should be reworked

    std::vector<PendingDownload> uncommitted_downloads;
    std::deque<std::vector<PendingDownload>> committed_downloads;
    std::vector<AsyncBuffer> uncommitted_async_buffers;
    std::deque<std::vector<AsyncBuffer>> async_buffers;
    std::deque<AsyncBuffer> async_buffers_death_ring;

    struct LRUItemParams {
        using ObjectType = ImageId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;

 #ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
 #else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Image, TICKS_TO_DESTROY> sentenced_images;
    /// Device memory of the images in each slot of sentenced_images (same index): deleted, but
    /// counted in the driver's usage until the ring destroys them.
    std::array<u64, TICKS_TO_DESTROY> sentenced_image_bytes{};
    size_t sentenced_image_index = 0;
    /// Deleted images on runtimes that report GPU progress (CurrentTick, IsFree): destroyed, with
    /// their device memory, as soon as the GPU is past them rather than TICKS_TO_DESTROY frames
    /// later. While the collector deletes dozens a frame, eight frames of them held 1-1.5 GB of
    /// VRAM above its threshold (title 1 on an 8 GB card: 6.6-7.0 GB in use against 5 GiB).
    struct GpuSentencedImage {
        u64 tick;
        u64 bytes;
        Image image;
    };
    std::deque<GpuSentencedImage> gpu_sentenced_images;
    u64 gpu_sentenced_bytes{};

    /// Bytes of images per device memory block, for the collector (refreshed every 60 frames).
    std::unordered_map<u64, u64> block_image_bytes;
    u64 block_image_bytes_frame{~0ULL};
    DelayedDestructionRing<ImageView, TICKS_TO_DESTROY> sentenced_image_view;
    DelayedDestructionRing<Framebuffer, TICKS_TO_DESTROY> sentenced_framebuffers;

    ::Common::unordered_map<GPUVAddr, ImageAllocId> image_allocs_table;

    Common::ScratchBuffer<u8> swizzle_data_buffer;
    Common::ScratchBuffer<u8> unswizzle_data_buffer;
    Tegra::Texture::UnswizzleWorker unswizzle_worker;

    u64 modification_tick = 0;
    u64 frame_tick = 0;

    Common::ThreadWorker texture_decode_worker{1, "TextureDecoder", {},
                                               Common::ThreadPlacement::Efficiency};
    std::vector<std::unique_ptr<AsyncDecodeContext>> async_decodes;
    /// Guest memory images were uploaded from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_ranges;

    std::deque<PendingUnswizzle> unswizzle_queue;
    u8 current_unswizzle_frame;

    // Join caching
    boost::container::small_vector<ImageId, 4> join_overlap_ids;
    ::Common::unordered_set<ImageId> join_overlaps_found;
    boost::container::small_vector<ImageId, 4> join_left_aliased_ids;
    boost::container::small_vector<ImageId, 4> join_right_aliased_ids;
    ::Common::unordered_set<ImageId> join_ignore_textures;
    boost::container::small_vector<ImageId, 4> join_bad_overlap_ids;
    struct JoinCopy {
        bool is_alias;
        ImageId id;
    };
    boost::container::small_vector<JoinCopy, 4> join_copies_to_do;
    ::Common::unordered_map<ImageId, size_t> join_alias_indices;

    RenderTargetStats rt_stats;
    std::array<u64, 3> rt_stats_base{}; ///< images_created, images_deleted, uploaded_bytes
};

} // namespace VideoCommon
