// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <array>
#include <bit>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <source_location>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/read_trap_tracker.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_download_trace.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 18;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Native shaders can reference more than 16 storage buffers. Dropping a binding
// shifts all following descriptor payloads into the wrong descriptor types.
constexpr u32 NUM_STORAGE_BUFFERS = Shader::Info::MAX_SSBOS;
static_assert(NUM_STORAGE_BUFFERS <= 64, "storage binding masks must fit u64");
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u64, NUM_STAGES> enabled_storage_buffers{};
    std::array<u64, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;
    // Bound by a native (Switch 2) compute program, whose pointers carry no size.
    u64 native_compute_storage_buffers = 0;
    // Bytes from each compute storage binding's start that a dispatch marks GPU-written.
    std::array<u32, NUM_STORAGE_BUFFERS> compute_storage_write_marks{};

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    friend struct BufferCacheTestAccess;
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef LICHII_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef LICHII_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        /// Large buffers that only touch the range: deleted, not joined
        boost::container::small_vector<BufferId, 2> cut;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool,
                               u64 write_mark_limit = ~u64{0});

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    void RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// `keep_page_gpu_data`: download GPU-written bytes sharing the written pages first (false
    /// when the memory is going away). `engine_write`: the GPU's own engines wrote it (inline
    /// constant buffer data, MemoryManager::FlushCaching), not the guest CPU.
    void WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data = true,
                     bool engine_write = false);

    void CachedWriteMemory(DAddr device_addr, u64 size);

    /// Whether constant buffer data written to [device_addr, device_addr + size) must reach
    /// WriteMemory. Pages no buffer was ever uploaded from and the GPU never wrote are
    /// CPU-modified still, so WriteMemory would change nothing there that the streamed uniform
    /// buffers reading them need: those read guest memory, and their keeps learn of the write
    /// from the draw's fast draw delta. Drawing thread, without the lock.
    [[nodiscard]] bool ConstantDataReachesBuffers(DAddr device_addr, u64 size) const;

    /// Cancels pending downloads over the range (fences' and reads'), so none writes them over
    /// newer bytes. Those bytes reach guest memory later if at all (a GPU copy, a clear, an
    /// inline or a discarded write), so read downloads already taken land first: their reads get
    /// what a synchronous flush would have written.
    void ClearDownload(DAddr base_addr, u64 size);

    /// ClearDownload where guest memory gets the newer bytes before a read can find the range: a
    /// guest CPU write, a synchronous download landing under `mutex`, or a GPU-side write whose
    /// read downloads MemoryManager landed first. Read downloads leave the range rather than land,
    /// so this never waits for the host GPU.
    void ClearReplacedDownload(DAddr device_addr, u64 size);

    /// CPU writes collected but not yet applied (SetPendingCpuWrites).
    [[nodiscard]] Common::RangeSet<DAddr> PendingCpuWriteRanges();

    /// Writes downloaded bytes to guest memory, except over `cpu_written`. Callers take
    /// `cpu_written` inside a VideoCore::DownloadLanding over the bytes and write in it: the CPU
    /// writes collected after it store their bytes once the landing has ended.
    void WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written, DAddr device_addr,
                               const u8* data, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    /// DownloadMemory for a guest CPU read: submits the copies without waiting for the host GPU,
    /// for FinishReadDownloads to write to guest memory. Their ranges stay GPU-modified until
    /// then, and another read of them waits for these copies (GetFlushArea: in_flight).
    void DownloadMemoryForRead(DAddr device_addr, u64 size);

    /// Writes the DownloadMemoryForRead copies over [device_addr, device_addr + size), and those
    /// submitted before them, to guest memory, waiting for the host GPU if it must. Any thread:
    /// the reading one before it reads, the GPU's side before it writes guest memory there. Takes
    /// `mutex` itself (a caller holding it keeps it held while the host GPU finishes). Returns how
    /// long it waited for the host GPU (at least 1 ns if it did), or nothing when no download was
    /// over the range.
    std::optional<u64> FinishReadDownloads(DAddr device_addr, u64 size);

    /// A GPU write into the buffers over [device_addr, device_addr + size), after the read
    /// downloads taken so far: they still write their older bytes to guest memory, as synchronous
    /// ones would have, but a later read takes the range again. MarkWrittenBuffer does this for
    /// the writes it records; the query cache's copies of its results call it themselves.
    void OutdateReadDownloads(DAddr device_addr, u64 size);

    /// A query result over [device_addr, device_addr + size) that the query cache writes to guest
    /// memory later, at a fence or sync point (QueryCacheBase::CounterReport). On a page that gets
    /// early downloads, a GPU write over it after the report stays GPU-modified when a download
    /// lands it: the older result lands over it, and a later read downloads it again. True when
    /// it keeps the result, which FinishReport then ends.
    [[nodiscard]] bool NoteReportPending(DAddr device_addr, u64 size);

    /// The query cache wrote a result NoteReportPending kept to guest memory, or dropped it.
    void FinishReport(DAddr device_addr, u64 size);

    /// Without `mutex`: false when the page under device_addr can't be one the guest reads back
    /// (NoteReportPending would return false). Seeing the bounds before a page joins them is
    /// the same as looking before that read took the lock.
    [[nodiscard]] bool MayBeReadBackPage(DAddr device_addr) const noexcept {
        const DAddr page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        return page >= read_back_first.load(std::memory_order_acquire) &&
               page <= read_back_last.load(std::memory_order_acquire);
    }

    /// The GPU thread recorded a draw or dispatch (`mutex` held). After a dispatch, the downloads
    /// of the pages written since that the guest reads after each write go right behind it
    /// (RecordEarlyDownloads). Those the host GPU has run land in guest memory.
    void NoteWorkRecorded(bool dispatch);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    /// The core consumed the read traps of these pages for a read the texture cache answered:
    /// the next GPU write here must arm them again (GetFlushArea forgets them otherwise).
    void ForgetReadTraps(DAddr device_addr, u64 size) {
        read_trapped_ranges.Subtract(
            Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE),
            Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE) -
                Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE));
    }

    /// Without downloads at every fence, a guest CPU read is the only way GPU-written buffer
    /// data reaches guest memory, so reads must ask the buffer cache (GetFlushArea).
    bool FlushesOnCpuRead() const {
        return !AsyncBufferDownloads();
    }

    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    void UpdateComputeBuffers();

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage);

    void BindHostComputeBuffers();

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 indirect_offset = ~0U);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    /// write_extent: how far the program's stores reach past the pointer (native programs).
    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 indirect_offset = ~0U,
                                  const Shader::StorageWriteExtent& write_extent = {});

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);
    void FlushCachedWrites();

    /// Where the CPU writes into GPU-written memory that were collected but not applied yet
    /// (WriteMemory) come from: downloads must not write the GPU's older copy over them.
    using PendingCpuWrites = std::function<void(const std::function<void(DAddr, size_t)>&)>;
    void SetPendingCpuWrites(PendingCpuWrites source) {
        pending_cpu_writes = std::move(source);
    }

    /// Takes CPU writes about to be applied one by one out of the GPU-written ranges first, so
    /// applying one doesn't download the GPU's copy over another (WriteMemory keeps the GPU data
    /// on the pages it writes).
    template <typename Ranges>
    void ClaimCpuWrites(const Ranges& ranges);

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Switch 2 GPU-written command lists (nx2.h, g_nx2_command_list_pages). While collecting (a
    /// compute dispatch or a DMA copy being recorded), writes to command list pages are noted;
    /// Nx2CopyListPages then records copies of those pages into host memory after the work.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    bool Nx2CopyListPages(u64 tick);
    enum class Nx2ListRead { Unknown, Copied, Synced, NeedsSync, Pending };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    /// Pages draws write are only Synced or Unknown, and NeedsSync until `past_sync`: the
    /// split's back ran the work before the last host-level wait (GpuSplit::WaitForSync).
    /// With `pending_tick`, a copy the GPU has not finished returns Pending and sets the tick
    /// instead of waiting for it.
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync,
                                u64* pending_tick = nullptr);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);

    void Nx2FlushListPages(std::span<const DAddr> pages);
    /// CPU and inline writes: they reach guest memory, so a synced page stays synced, but its copy
    /// no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    /// Visits the uniform and storage buffers bound for the last draw or dispatch with the cached
    /// host buffer holding each range, if one does, for diagnostics. Nothing is
    /// created or synchronized. Stage 5 is compute.
    template <typename Func>
    void ForEachBoundBuffer(bool compute, Func&& func) {
        const auto visit = [&](u32 stage, bool storage, u32 index, bool written,
                               const Binding& binding, u32 size) {
            if (binding.device_addr == 0 || size == 0) {
                return;
            }
            const Buffer* host = nullptr;
            u32 offset = 0;
            if (const BufferId id = page_table[binding.device_addr >> CACHING_PAGEBITS]) {
                const Buffer& buffer = slot_buffers[id];
                if (buffer.IsInBounds(binding.device_addr, size)) {
                    host = &buffer;
                    offset = buffer.Offset(binding.device_addr);
                }
            }
            func(stage, storage, index, written, binding, size, host, offset);
        };
        if (compute) {
            const auto* const sizes = channel_state->compute_uniform_buffer_sizes;
            ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
                const Binding& binding = channel_state->compute_uniform_buffers[index];
                visit(5, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
                const Binding& binding = channel_state->compute_storage_buffers[index];
                visit(5, true, index, ((channel_state->written_compute_storage_buffers >> index) & 1) != 0,
                      binding, binding.size);
            });
            return;
        }
        const auto* const sizes = channel_state->uniform_buffer_sizes;
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                const Binding& binding = channel_state->uniform_buffers[stage][index];
                visit(stage, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[stage][index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                visit(stage, true, index,
                      ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0, binding,
                      binding.size);
            });
        }
    }

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    /// Fast draw verify (nx2_fast_draw.h): a hash of the bindings a delta draw keeps.
    [[nodiscard]] u64 KeptGraphicsBindingsHash(bool vertex, bool storage, bool texture) const;

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    /// Called before each graphics draw binds its buffers (uniform keeps). `delta`: the constant
    /// buffer binds and data since the previous call, or nullptr when they aren't known (then
    /// nothing is kept).
    void BeginGraphicsUniforms(const Nx2FastDraw::Delta* delta);

    Common::SpinRecursiveMutex mutex;
    Runtime& runtime;
    bool any_buffer_uploaded = false;
    /// Indirectly read banks of native programs already uploaded this frame.
    std::unordered_set<DAddr> native_indirect_uniforms_this_frame;

private:
    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector(bool aggressive_gc);

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    void BindHostGraphicsStorageBuffers(size_t stage);

    void BindHostGraphicsTextureBuffers(size_t stage);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers();

    void BindHostComputeTextureBuffers();

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers();

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers();

    void UpdateComputeTextureBuffers();

    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false);

    /// Counts a GPU write to the command list pages in the range (invalidating their copies) and,
    /// while collecting, asks for new copies.
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible,
                                      std::source_location caller = std::source_location::current());

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size, bool sparse_compatible,
                                        std::source_location caller = std::source_location::current());

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size);

    /// SynchronizeBuffer for vertex buffer slots 0-31 and the index buffer (slot 32).
    void SynchronizeGeometryBuffer(u32 slot, BufferId buffer_id, DAddr device_addr, u32 size);

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    /// Downloads the GPU-written data of several buffers with one wait for the GPU.
    void DownloadBuffersMemory(std::span<const BufferId> buffer_ids);

    void DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges);

    /// ClearDownload, for the downloads taken at fences only.
    void ClearFenceDownloads(DAddr device_addr, u64 size);

    /// Whether every GPU-modified byte in the range is in a read download's clean_ranges.
    [[nodiscard]] bool ReadDownloadsCover(DAddr device_addr, u64 size) const;

    /// Writes the read downloads up to `tick` to guest memory, oldest first (`mutex` held, the
    /// host GPU past `tick`). `landed_by`: who does, for the read check's lines.
    void WriteReadDownloads(
        u64 tick,
        VideoCore::AsyncReadback::Landing landed_by = VideoCore::AsyncReadback::Landing::Completed);

    /// WriteReadDownloads for those the host GPU has completed: nothing waits.
    void WriteCompletedReadDownloads();

    /// WriteReadDownloads up to the newest read download over the range, if the host GPU ran it.
    /// Returns whether it did.
    bool WriteFinishedReadDownloads(DAddr device_addr, u64 size);

    /// Records a read download of the GPU-modified bytes over `ranges` that no read download
    /// brings yet. `own`: the bytes the read itself asked for (the read check compares them).
    /// A read's download is submitted at once; an early one goes with the GPU thread's next
    /// submission.
    void RecordReadDownload(std::span<const std::pair<DAddr, u64>> ranges,
                            std::span<const std::pair<DAddr, u64>> own, bool early);

    /// A guest read of the pages [begin, end): learns which pages the guest reads after the GPU
    /// writes them. `gpu_modified`: the read found GPU writes not yet in guest memory.
    void NoteReadBack(DAddr begin, DAddr end, bool gpu_modified);

    /// A GPU write: the pages under it that the guest reads after each write get an early
    /// download once the writing work is recorded.
    void NoteReadBackWrite(DAddr device_addr, u64 size);

    /// Records the early downloads of the pages written since their last one (`mutex` held).
    void RecordEarlyDownloads();

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                               bool native_pointer = false,
                                               bool table_pointer = false,
                                               bool compute = false) const;

    void TraceTablePointer(bool compute, size_t stage, size_t ssbo_index, u32 cbuf_index,
                           u32 cbuf_offset, u32 indirect_offset, GPUVAddr holder,
                           const Binding& binding);

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    /// Flags the vertex buffers overlapping a written range so host copies derived from them are
    /// rebuilt.
    void MarkVertexBuffersStale(DAddr device_addr, u64 size);

    /// Re-dirties repacked vertex buffers whose source data was written since they were bound.
    void RefreshRepackedVertexBuffers();

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef LICHII_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    u32 stale_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    /// The range each geometry slot last scanned for uploads (SynchronizeGeometryBuffer).
    struct GeometrySync {
        BufferId buffer_id{};
        DAddr device_addr{};
        u32 size{};
        u64 deletions{};
    };
    std::array<GeometrySync, NUM_VERTEX_BUFFERS + 1> geometry_syncs{};
    /// Counts DeleteBuffer calls: a buffer id older than this count may have been freed or reused.
    u64 buffer_deletions = 0;
    /// EDEN_NX2_BUFFER_TRACE: buffers created (with the buffers they joined) and collected, logged
    /// every 10 s, and each creation of 64 MiB or more with the line that asked for it.
    struct CreateTrace {
        bool enabled = std::getenv("EDEN_NX2_BUFFER_TRACE") != nullptr;
        u64 creates{};
        u64 created_bytes{};
        u64 joins{};
        u64 joined_bytes{};
        u64 leaps{};
        u64 collected{};
        u64 collected_bytes{};
        u32 large_logged{};
        std::chrono::steady_clock::time_point last{};
    } create_trace;
    struct GeometryStats {
        u64 syncs{};
        u64 syncs_skipped{};
        u64 verify_mismatches{};
    } geometry_stats;

    /// Uniform keeps: a draw binds a small uniform buffer from the stream allocation an earlier
    /// draw of the command buffer made for its slot and range, without reading guest memory, when
    /// nothing ordered between them wrote the range: no inline constant data, no rebind, no CPU
    /// write the caches were told about, no GPU write, no buffer deletion and no memory map
    /// change. Nor anything the guest CPU could have synchronised with (Nx2FastDraw::SyncSerial):
    /// it may write pages the caches don't watch after one.
    [[nodiscard]] bool UniformRangeKept(size_t stage, u32 index, DAddr device_addr,
                                        u32 size) const noexcept;
    void NoteUniformWrite(DAddr device_addr, u64 size) {
        UniformKeep& keep = uniform_keep;
        if (keep.memory_writes.size() == keep.memory_writes.capacity()) {
            keep.memory_writes_lost = true;
            return;
        }
        keep.memory_writes.emplace_back(device_addr, device_addr + size);
    }
    /// EDEN_NX2_UBO_KEEP_VERIFY=1: binds what a keep would skip the original way and logs where
    /// guest memory changed.
    const bool ubo_keep_verify = [] {
        const char* const value = std::getenv("EDEN_NX2_UBO_KEEP_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    struct UniformKeep {
        /// Slots whose last stream allocation still holds what guest memory has at `ranges`.
        std::array<u32, NUM_STAGES> clean{};
        std::array<std::array<std::pair<DAddr, u32>, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
            ranges{};
        /// CPU writes and GPU-modified marks since the previous draw began.
        boost::container::static_vector<std::pair<DAddr, DAddr>, 8> memory_writes;
        bool memory_writes_lost{}; ///< More writes than memory_writes holds
        u64 draw_deletions{}; ///< buffer_deletions when the previous draw began
        u64 sync_serial{};    ///< Nx2FastDraw::SyncSerial then
        u64 host_serial{};    ///< MemoryManager::HostPointerSerial then
        /// gpu_memory then: another one's HostPointerSerial can be the same number
        const Tegra::MemoryManager* gpu_memory{};
        u64 kept{};
        u64 streamed_binds{};
        u64 verified{};
        u64 mismatches{};
    } uniform_keep;

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;

    /// Whether GPU-written buffer ranges are copied back to guest memory at the next fence;
    /// otherwise only when something flushes them (as upstream yuzu below High GPU accuracy).
    /// Not for Switch 2 titles: title 1 had 250-330 MB/s copied back (a fence per compute
    /// dispatch, the fencing thread holding the buffer cache while it wrote them), while its CPU
    /// read GPU-written memory at most 64 times in 200 s (counted with reactive flushing) and
    /// the native MME's DMA reads never touched those pages. Without them a fence costs ~1 us
    /// instead of ~5 and draw setup ~7% less. EDEN_NX2_BUFFER_DOWNLOADS=0/1 forces it.
    bool AsyncBufferDownloads() const {
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_BUFFER_DOWNLOADS");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        if (forced >= 0) {
            return forced == 1;
        }
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }

    /// Whether a GPU write to this range is copied back at the next fence. Ranges the guest CPU
    /// has read (Preflushable, set by the first reactive flush) always are: later reads skip the
    /// flush and rely on these downloads, e.g. collision meshes skinned by compute shaders.
    bool DownloadsAtFence(DAddr device_addr, u64 size) noexcept {
        return AsyncBufferDownloads() || memory_tracker.IsRegionPreflushable(device_addr, size);
    }

    /// Whether the range holds GPU-written data that no fence downloads (not Preflushable).
    bool HasGpuWritesNotDownloadedAtFence(DAddr device_addr, u64 size) {
        if (AsyncBufferDownloads()) {
            return false;
        }
        bool found = false;
        memory_tracker.ForEachDownloadRange(
            device_addr, size, false, [&](u64 start, u64 range_size) {
                if (found) {
                    return;
                }
                u64 covered = 0;
                memory_tracker.ForEachPreflushableRange(
                    start, range_size, [&](u64 lower, u64 length) {
                        const u64 begin = (std::max)(lower, start);
                        const u64 end = (std::min)(lower + length, start + range_size);
                        covered += end > begin ? end - begin : 0;
                    });
                found = covered < range_size;
            });
        return found;
    }

    /// Queues a GPU write for download at the next fence where DownloadsAtFence holds. With only
    /// Preflushable marks, just the marked pages are queued: a write that covers one marked page
    /// of a large buffer would otherwise copy the whole buffer back at every fence.
    void QueueFenceDownload(DAddr device_addr, u64 size) {
        static const bool exact = [] {
            const char* value = std::getenv("EDEN_NX2_EXACT_PREFLUSH");
            return value == nullptr || value[0] != '0';
        }();
        if (AsyncBufferDownloads()) {
            uncommitted_gpu_modified_ranges.Add(device_addr, size);
            return;
        }
        if (!exact) {
            if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
                uncommitted_gpu_modified_ranges.Add(device_addr, size);
            } else {
                ArmReadTraps(device_addr, size);
            }
            return;
        }
        const DAddr end = device_addr + size;
        u64 queued = 0;
        memory_tracker.ForEachPreflushableRange(device_addr, size, [&](DAddr begin, u64 length) {
            const DAddr lower = (std::max)(begin, device_addr);
            const DAddr upper = (std::min)(begin + length, end);
            if (lower < upper) {
                uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
                queued += upper - lower;
            }
        });
        if (queued < size) {
            ArmReadTraps(device_addr, size);
        }
        if (Nx2DownloadTrace::g_enabled && queued != 0) [[unlikely]] {
            Nx2DownloadTrace::Queued(device_addr, size, queued);
        }
    }

    /// GPU writes that are not downloaded at fences: trap the guest CPU's next read of their
    /// pages, which flushes them on demand (or marks them Preflushable with preflushing enabled).
    void ArmReadTraps(DAddr device_addr, u64 size) {
        const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        // Only the parts not armed yet: the core visits each page it is given, and a buffer the
        // GPU rewrites every frame is mostly still armed (the guest read a few of its pages).
        read_trapped_ranges.Arm(begin, end - begin, [this](DAddr part, u64 part_size) {
            device_memory.ArmReadTraps(part, part_size);
        });
    }

    ReadTrapTracker read_trapped_ranges;
    /// Ranges the guest CPU read after GPU writes. A read that must flush downloads the GPU
    /// writes in all of them, so the reads that usually follow it find nothing left to flush.
    Common::RangeSet<DAddr> cpu_read_ranges;
    bool download_cpu_read_ranges{};
    Common::RangeSet<DAddr> gpu_modified_ranges;
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    /// For each entry of pending_downloads: the ranges it still writes to guest memory when its
    /// fence pops. A write to a range after a download was taken removes it from that download
    /// (and every other pending one): its bytes are older than the write.
    std::deque<Common::RangeSet<DAddr>> pending_download_ranges;
    PendingCpuWrites pending_cpu_writes;
    std::optional<Async_Buffer> current_buffer;

    /// EDEN_NX2_READ_CHECK (1 in n read downloads): what a synchronous flush of
    /// the read would have copied, copied again in the same submission, to compare guest memory
    /// with once the download is written.
    struct ReadCheck {
        /// The bytes the read asked for, or the pages of an early download
        DAddr read_begin{};
        DAddr read_end{};
        bool early{};
        Async_Buffer staging{};
        /// src_offset: the device address copied; dst_offset: where it is in `staging`.
        boost::container::small_vector<BufferCopy, 4> copies;
        /// Replaced in guest memory since (ClearDownload): not compared.
        Common::RangeSet<DAddr> changed;
        /// Not in this download: an earlier one brings them, as the synchronous flush of its own
        /// read did. Guest memory may get newer bytes there after it lands, which the
        /// synchronous flushes would have kept too.
        Common::RangeSet<DAddr> kept;
        /// What guest memory held over `kept` right after those downloads landed (`landed`), at
        /// the offsets the copies have in `staging`.
        std::vector<u8> landed_bytes;
        Common::RangeSet<DAddr> landed;
        /// A check alone, its read's bytes all brought by other downloads: the bytes it compares,
        /// which its read waits for as for a download of its own.
        Common::RangeSet<DAddr> own;
    };
    struct ReadCheckTotals {
        u64 every{};      ///< Tests: check 1 in `every` read downloads, over EDEN_NX2_READ_CHECK
        u64 next{};       ///< Read downloads seen, for the sampling
        u64 checked{};    ///< Read downloads compared
        u64 compared{};   ///< Bytes compared
        u64 mismatched{}; ///< Bytes that differ from the synchronous copy
        /// Of mismatched, by who wrote the download to guest memory
        std::array<u64, static_cast<size_t>(VideoCore::AsyncReadback::Landing::Count)>
            mismatched_by_landing{};
        u64 uncovered{};  ///< Bytes a synchronous flush copies that no download brings
        u64 rewritten{};  ///< Kept bytes guest memory got anew after their download landed
        u64 logged{};     ///< Error lines so far (at most 16)
        u64 logged_rewrites{}; ///< Lines about rewritten bytes so far (at most 16)
        u64 reads_recorded{};  ///< Checks recorded with a read's download
        u64 alone_recorded{};  ///< Checks recorded alone (other downloads bring the read's bytes)
        u64 early_recorded{};  ///< Checks recorded with an early download
        u64 last_log{};
    };
    ReadCheckTotals read_check{};
    static u64 ReadCheckEvery() {
        return VideoCore::AsyncReadback::ReadCheckEvery();
    }
    void CheckReadDownload(const ReadCheck& check, const Common::RangeSet<DAddr>& cpu_written,
                           VideoCore::AsyncReadback::Landing landing);
    [[nodiscard]] bool ReadChecks() const {
        return read_check.every != 0 || ReadCheckEvery() != 0;
    }
    /// A download landed `written`, but for the guest CPU writes collected then (`cpu_written`):
    /// what guest memory holds now over a later check's `kept`.
    void NoteLandedBytes(ReadCheck& check, const Common::RangeSet<DAddr>& written,
                         const Common::RangeSet<DAddr>& cpu_written);

    /// A DownloadMemoryForRead: its copies, and what is left of them to do once the host GPU has
    /// run them.
    struct ReadDownload {
        u64 tick{}; ///< The submission with the copies
        Async_Buffer staging{};
        /// src_offset: the device address copied; dst_offset: where it is in `staging`.
        boost::container::small_vector<BufferCopy, 4> copies;
        /// Still to write to guest memory: a newer value there (ClearDownload) takes its bytes
        /// out. A newer GPU write doesn't: until a download of its own lands, guest memory is to
        /// hold what a synchronous flush would have left.
        Common::RangeSet<DAddr> write_ranges;
        /// No longer GPU-modified once written: a newer GPU write (MarkWrittenBuffer) or value in
        /// guest memory takes its bytes out.
        Common::RangeSet<DAddr> clean_ranges;
        std::unique_ptr<ReadCheck> check; ///< EDEN_NX2_READ_CHECK
        /// Recorded right after a GPU write, before any read (RecordEarlyDownloads)
        bool early{};
    };
    /// Oldest first, and written to guest memory in that order: none lands over newer bytes.
    std::deque<ReadDownload> read_downloads;
    /// All the read downloads' clean_ranges (they don't overlap): GPU-modified bytes already on
    /// their way, which later reads wait for rather than download again.
    Common::RangeSet<DAddr> read_download_ranges;

    /// EDEN_NX2_READ_CHECK: the bytes early downloads brought to a page's guest memory, and
    /// what guest memory held there as they landed, until newer downloaded bytes or values replace
    /// them. The read they serve compares them with the GPU's bytes then: a GPU write in between
    /// that took nothing out of them shows.
    struct EarlyLanded {
        Common::RangeSet<DAddr> bytes;
        std::vector<u8> as_landed; ///< The page, DEVICE_PAGESIZE bytes
    };
    std::unordered_map<DAddr, EarlyLanded> early_landed;
    /// EDEN_NX2_READ_CHECK: an upload sends guest memory's bytes to the buffer. Where they
    /// are no longer what an early download landed, the guest wrote there since, and landed bytes
    /// are no longer GPU-modified, so that write may never have reached WriteMemory. The buffer
    /// then holds it while guest memory may hold the landed value again: no read compares those
    /// bytes, as a synchronous flush would not download them.
    void ForgetUploadedEarlyLanded(DAddr device_addr, u64 size) {
        if (!early_landed.empty()) {
            ForgetRewrittenEarlyLanded(device_addr, size);
        }
    }
    void ForgetRewrittenEarlyLanded(DAddr device_addr, u64 size);
    /// Whether an early download serves this read: in check mode it flushes (and is compared).
    [[nodiscard]] bool EarlyDownloadServes(DAddr device_addr, u64 size) const;
    void NoteEarlyLanded(const ReadDownload& download, const Common::RangeSet<DAddr>& cpu_written);
    /// Calls func(begin, end, the bytes as they landed) for what early downloads brought to
    /// [device_addr, device_addr + size) that no GPU write changed since, which a read takes:
    /// a later read compares only what lands after it.
    template <typename Func>
    void TakeEarlyLanded(DAddr device_addr, u64 size, Func&& func);
    void ForgetEarlyLanded(DAddr device_addr, u64 size);

    /// A page the guest CPU read GPU writes from. Switch 2 titles read compute results back every
    /// frame, milliseconds after the dispatch that wrote them: a download recorded at the read
    /// waits for all the GPU work recorded in between. Pages read after each of their writes get
    /// their download recorded right behind the write instead (RecordEarlyDownloads), which the
    /// read finds landed or on its way (GetFlushArea: in_flight).
    struct ReadBackPage {
        DAddr page{};
        u64 last_read{};     ///< read_back_reads at its last read of a write (eviction)
        u16 reads{};         ///< Writes in a row the guest read, up to EarlyAfterReads
        u16 unread_writes{}; ///< GPU writes since its last read
        bool written{};      ///< A GPU write since its last read
    };
    /// Reads of a page's writes in a row before its writes get early downloads.
    static constexpr u16 EarlyAfterReads = 2;
    /// Writes in a row without a read that stop them.
    static constexpr u16 MaxUnreadWrites = 4;
    static constexpr size_t MaxReadBackPages = 64;
    /// Early download bytes a frame, and early downloads not yet in guest memory: past them, reads
    /// flush on demand.
    static constexpr u64 EarlyBytesPerFrame = 1_MiB;
    static constexpr u32 MaxPendingEarlyDownloads = 32;
    std::vector<ReadBackPage> read_back_pages; ///< Sorted by page
    /// The first and last page of read_back_pages (MayBeReadBackPage), set under `mutex`.
    std::atomic<DAddr> read_back_first{~DAddr{0}};
    std::atomic<DAddr> read_back_last{0};
    /// NoteReportPending's results not yet written, oldest first.
    struct PendingReport {
        DAddr addr;
        u64 size;
        bool rewritten; ///< A GPU write over it since the report
    };
    std::vector<PendingReport> pending_reports;
    /// Pages to download once the work writing them is recorded.
    boost::container::small_vector<DAddr, 8> early_pages;
    u64 read_back_reads{};
    u64 early_bytes_this_frame{};
    u32 pending_early_downloads{};
    u32 early_landing_checks{};

    std::deque<Async_Buffer> async_buffers_death_ring;

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;
    std::vector<u8> guest_uint8_indices;
    /// Guest memory large uploads came from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_buffer_ranges;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    std::vector<MultiRangeSegment> graphics_segments;
    std::vector<MultiRangeSegment> compute_segments;
    u64 frame_tick = 0;
    /// Buffers created since the last TickFrame; ResolveOverlaps does not cut these
    std::vector<BufferId> frame_created_buffers;
    /// EDEN_NX2_TABLE_POINTER_TRACE: last table entry and size bound per table pointer slot.
    std::unordered_map<u64, std::pair<GPUVAddr, u32>> traced_table_pointers;
    u64 total_used_memory = 0;
    u64 buffer_bytes = 0; ///< The buffers' allocated size (total_used_memory may be the device's)
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef LICHII_LEGACY
    bool immediately_free = false;
#endif

    // Covers the device address space (MaxwellDeviceTraits::device_virtual_bits).
    std::array<BufferId, ((1ULL << 36) >> CACHING_PAGEBITS)> page_table;
    Common::ScratchBuffer<u8> tmp_buffer;

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    struct Nx2ListPage {
        u64 writes{};      ///< GPU writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{};  ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent which draws write the page, so it reads
        /// no copy. A draw's writes reach a pushbuffer fetch only after a host-level wait, so
        /// the page's lists read guest memory once the back ran the work before the last such
        /// wait and no write came since the front flushed the page (Nx2ReadListCopy).
        bool draw_written{};
        std::optional<Nx2Staging> copy;
    };
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
