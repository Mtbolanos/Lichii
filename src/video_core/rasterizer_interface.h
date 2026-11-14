// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/settings.h"
#include "video_core/cache_types.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/gpu.h"
#include "video_core/query_cache/types.h"
#include "video_core/rasterizer_download_area.h"

namespace Tegra {
class MemoryManager;
namespace Engines {
class AccelerateDMAInterface;
}
namespace Control {
struct ChannelState;
}
namespace Texture {
struct TICEntry;
}
} // namespace Tegra

namespace VideoCommon::OotNetwork {
struct BypassState;
}

namespace VideoCore {

enum class LoadCallbackStage {
    Prepare,
    Build,
    Complete,
    /// The shader precache running before the game starts (programs built / total).
    /// Last, so the values of the others stay as frontends that know only those expect.
    Warm,
};
using DiskResourceLoadCallback = std::function<void(LoadCallbackStage, std::size_t, std::size_t)>;

/// Set while the query cache signals the fence that writes a fence-flagged report, so the fence
/// manager makes that fence even when it is empty (FenceManager::SignalFence).
inline thread_local bool g_signaling_query_report_fence = false;

class RasterizerInterface {
public:
    virtual ~RasterizerInterface() = default;

    /// Dispatches a draw invocation
    virtual void Draw(bool is_indexed, u32 instance_count) = 0;

    /// Dispatches an indirect draw invocation
    virtual void DrawIndirect() {}

    /// Dispatches an draw texture invocation
    virtual void DrawTexture() = 0;

    /// Clear the current framebuffer
    virtual void Clear(u32 layer_count) = 0;

    /// Dispatches a compute shader invocation
    virtual void DispatchCompute() = 0;

    /// Resets the counter of a query
    virtual void ResetCounter(VideoCommon::QueryType type) = 0;

    /// Records a GPU query and caches it
    virtual void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                       VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) = 0;

    /// Signal an uniform buffer binding
    virtual void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                           u32 size) = 0;

    /// Signal disabling of a uniform buffer
    virtual void DisableGraphicsUniformBuffer(size_t stage, u32 index) = 0;

    /// Signal a GPU based semaphore as a fence
    virtual void SignalFence(std::function<void()>&& func) = 0;

    /// Send an operation to be done after a certain amount of flushes.
    virtual void SyncOperation(std::function<void()>&& func) = 0;

    /// Signal a GPU based syncpoint as a fence
    virtual void SignalSyncPoint(u32 value) = 0;

    /// Signal a GPU based reference as point
    virtual void SignalReference() = 0;

    /// Release all pending fences.
    virtual void ReleaseFences(bool force = true) = 0;

    /// Writes the region's query reports to guest memory, waiting for pending host results.
    virtual void FlushQueryResults(DAddr addr, u64 size) {
        FlushRegion(addr, size, VideoCommon::CacheType::QueryCache);
    }

    /// Notify rasterizer that all caches should be flushed to Switch memory
    virtual void FlushAll() = 0;

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    virtual void FlushRegion(DAddr addr, u64 size,
                             VideoCommon::CacheType which = VideoCommon::CacheType::All) = 0;

    /// FlushRegion for a guest CPU read of the region (GPU::OnCPURead), on the GPU thread: a
    /// cache may submit its downloads without waiting for the host GPU, for FinishRead to write.
    virtual void FlushRegionForRead(DAddr addr, u64 size) {
        FlushRegion(addr, size);
    }

    /// Writes to guest memory what FlushRegionForRead submitted over the region (and anything
    /// submitted before it), waiting for the host GPU if it must. Any thread: the reading one
    /// before it reads, and the GPU's side before it writes guest memory there.
    virtual void FinishRead([[maybe_unused]] DAddr addr, [[maybe_unused]] u64 size) {}

    /// A query result the query cache writes to guest memory later, at a fence or sync point.
    /// When this returns true, FinishReport once it wrote or dropped it.
    virtual bool NoteReportPending([[maybe_unused]] DAddr addr, [[maybe_unused]] u64 size) {
        return false;
    }
    virtual void FinishReport([[maybe_unused]] DAddr addr, [[maybe_unused]] u64 size) {}

    /// Switch 2, from the GPU thread's front: where a GPU-written command list's words are.
    enum class GpuWrittenList {
        Unknown, ///< Neither: the caller drains, flushes the page and notes it synced
        Copied,  ///< `out` (within one 4 KiB page) is filled from the copy made after its writer
        Synced,  ///< Guest memory holds the page: no GPU work wrote it since it was flushed
        /// Draws write the page: ask again with `past_sync` once the split's back ran the work
        /// before the last host-level wait (GpuSplit::WaitForSync)
        NeedsSync,
        /// Only with `pending_tick`: the copy is not done yet, `*pending_tick` is the host GPU
        /// tick that completes it. Ask again once IsGpuTickDone(*pending_tick).
        Pending,
    };
    /// With `pending_tick`, a copy the GPU is still making returns Pending instead of waiting.
    virtual GpuWrittenList ReadGpuWrittenCommandList([[maybe_unused]] DAddr addr,
                                                     [[maybe_unused]] std::span<u8> out,
                                                     [[maybe_unused]] bool past_sync,
                                                     [[maybe_unused]] u64* pending_tick = nullptr) {
        return GpuWrittenList::Unknown;
    }

    /// Whether the host GPU passed `tick`, without waiting. Asks for deferred work to be
    /// submitted when the tick is not submitted yet, so it completes.
    virtual bool IsGpuTickDone([[maybe_unused]] u64 tick) {
        return true;
    }

    /// Switch 2: the front flushed this command list page with the split drained.
    virtual void NoteGpuWrittenCommandListSynced([[maybe_unused]] DAddr page) {}

    virtual void FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
        for (const DAddr page : pages) {
            FlushRegion(page, 0x1000, VideoCommon::CacheType::BufferCache);
        }
    }

    /// Check if the the specified memory area requires flushing to CPU Memory.
    virtual bool MustFlushRegion(DAddr addr, u64 size,
                                 VideoCommon::CacheType which = VideoCommon::CacheType::All) = 0;

    virtual RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) = 0;

    /// Whether reports in the range wait for host query results not yet in guest memory.
    [[nodiscard]] virtual bool HasPendingQueries([[maybe_unused]] DAddr addr,
                                                 [[maybe_unused]] u64 size) {
        return false;
    }

    /// Any thread: whether the query cache holds reports that can still land in the range, as
    /// FlushQueryResults writes them out or once their host results come in. Without a query
    /// cache that can tell, assume so.
    [[nodiscard]] virtual bool HoldsQueries([[maybe_unused]] DAddr addr,
                                            [[maybe_unused]] u64 size) {
        return true;
    }

    /// Notify rasterizer that any caches of the specified region should be invalidated
    virtual void InvalidateRegion(DAddr addr, u64 size,
                                  VideoCommon::CacheType which = VideoCommon::CacheType::All) = 0;

    /// Called before guest memory in the region is overwritten, ahead of InvalidateRegion
    virtual void PrepareRegionWrite([[maybe_unused]] DAddr addr, [[maybe_unused]] u64 size,
                                    [[maybe_unused]] VideoCommon::CacheType which) {}

    virtual void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
        // The skip keeps the texture cache: software engine writes reach images only here.
        const auto which = Settings::values.skip_cpu_inner_invalidation.GetValue()
                               ? VideoCommon::CacheType::TextureCache
                               : VideoCommon::CacheType::All;
        for (const auto& [cpu_addr, size] : sequences) {
            InvalidateRegion(cpu_addr, size, which);
        }
    }

    /// InnerInvalidation of constant buffer data (MemoryManager::WriteConstantBufferData).
    virtual void InvalidateConstantBufferData(
        std::span<const std::pair<DAddr, std::size_t>> sequences) {
        InnerInvalidation(sequences);
    }

    /// Notify rasterizer that any caches of the specified region are desync with guest
    virtual void OnCacheInvalidation(PAddr addr, u64 size) = 0;

    /// OnCacheInvalidation for each of the CPU writes InvalidateGPUCache collected.
    virtual void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
        for (const auto& [addr, size] : ranges) {
            OnCacheInvalidation(addr, size);
        }
    }

    virtual bool TryCacheInvalidation(DAddr addr, u64 size) {
        if (MustFlushRegion(addr, size, VideoCommon::CacheType::BufferCache)) return false;
        OnCacheInvalidation(addr, size);
        return true;
    }

    virtual bool OnCPUWrite(PAddr addr, u64 size) = 0;

    /// Sync memory between guest and host.
    virtual void InvalidateGPUCache() = 0;

    /// Unmap memory range
    virtual void UnmapMemory(DAddr addr, u64 size) = 0;

    /// Remap GPU memory range. This means underneath backing memory changed
    virtual void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) = 0;

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    /// and invalidated
    virtual void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) = 0;

    /// Notify the host renderer to wait for previous primitive and compute operations.
    virtual void WaitForIdle() = 0;

    /// Notify the host renderer to wait for reads and writes to render targets and flush caches.
    virtual void FragmentBarrier() = 0;

    /// Notify the host renderer to make available previous render target writes.
    virtual void TiledCacheBarrier() = 0;

    /// Notify the rasterizer to send all written commands to the host GPU.
    virtual void FlushCommands() = 0;

    /// Notify rasterizer that a frame is about to finish
    virtual void TickFrame() = 0;

    virtual bool AccelerateConditionalRendering() {
        return false;
    }

    /// Attempt to use a faster method to perform a surface copy
    [[nodiscard]] virtual bool AccelerateSurfaceCopy(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        const Tegra::Engines::Fermi2D::Config& copy_config) {
        return false;
    }

    /// Stand-in for a DLSS evaluation: temporal upscale of src into dst along motion_vectors
    /// (jitter in src pixels; motion_scale turns a motion vector into a UV offset to the previous
    /// frame), with depth for NVIDIA DLSS or FSR 3.1. With use_history=false, only de-jitter src.
    /// False when unsupported; the caller retains its native dispatch.
    [[nodiscard]] virtual bool AccelerateTemporalUpscale(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        const Tegra::Engines::Fermi2D::Surface& motion_vectors,
        const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
        std::array<f32, 2> motion_scale, bool use_history = true,
        std::optional<std::array<u32, 2>> render_region = std::nullopt) {
        return false;
    }

    /// Whether a game's own DLSS network can run here: its matrix multiplies are emulated with
    /// 32-wide shuffles, and its output must be checkable (ProbeGameDlssOutput).
    [[nodiscard]] virtual bool SupportsGameDlssNetwork() const {
        return false;
    }

    /// Whether NVIDIA DLSS (the "Use NVIDIA DLSS" setting) would draw a game's DLSS evaluation.
    [[nodiscard]] virtual bool UsesNvidiaDlss() const {
        return false;
    }

    /// Whether ProbeGameDlssOutput checks anything on this backend.
    [[nodiscard]] virtual bool CanProbeGameDlssOutput() const {
        return false;
    }

    /// Records a health check of a game's DLSS network output (dst) against its input (src) and
    /// returns the result of an earlier check once the GPU has finished it.
    /// `src_region` is the part of src the game rendered, from its top-left corner (a dynamic
    /// resolution network's render region); without it, the whole of src.
    [[nodiscard]] virtual std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        std::optional<std::array<u32, 2>> src_region = std::nullopt) {
        return std::nullopt;
    }

    /// State of the Ocarina of Time anti-aliasing network bypass (oot_network.h), or null when
    /// this backend cannot bypass it or NVIDIA DLSS replaces the same output pass.
    [[nodiscard]] virtual VideoCommon::OotNetwork::BypassState* OotNetworkBypass() {
        return nullptr;
    }

    [[nodiscard]] virtual Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() = 0;

    virtual void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                          std::span<const u8> memory) = 0;

    /// Initialize disk cached resources for the game being emulated
    virtual void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                   const DiskResourceLoadCallback& callback) {}

    virtual void InitializeChannel(Tegra::Control::ChannelState& channel) {}

    virtual void BindChannel(Tegra::Control::ChannelState& channel) {}

    virtual void ReleaseChannel(s32 channel_id) {}

    /// Register the address as a Transform Feedback Object
    virtual void RegisterTransformFeedback(GPUVAddr tfb_object_addr) {}

    /// Returns true when the rasterizer has Draw Transform Feedback capabilities
    virtual bool HasDrawTransformFeedback() {
        return false;
    }
};
} // namespace VideoCore
