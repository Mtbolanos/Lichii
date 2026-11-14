// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/page_table.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/indirect_buffer.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {

struct Runtime;

struct Buffer : VideoCommon::BufferBase {
    Buffer(Runtime&, VideoCommon::NullBufferParams params) : BufferBase{params} {}
    Buffer(Runtime&, DAddr address, u32 size, bool mapped)
        : BufferBase{address, size}, bytes(size), host_mapped{mapped} {}

    void MarkUsage(u64 offset, u64 size) {
        REQUIRE(offset + size <= bytes.size());
    }
    std::span<u8> Mapped() {
        return host_mapped ? std::span{bytes} : std::span<u8>{};
    }
    u64 LastUsageTick() const {
        return tick;
    }

    std::vector<u8> bytes;
    bool host_mapped{};
    u64 tick{};
};

struct Staging {
    Buffer* buffer;
    u64 offset;
    std::span<u8> mapped_span;
};

struct Runtime {
    bool CanReportMemoryUsage() const {
        return false;
    }
    u64 GetBufferAllocationTail() const {
        return 0;
    }
    u64 GetDeviceLocalMemory() const {
        return 0;
    }
    u64 GetMaxStorageBufferRange() const {
        return 1ULL << 30;
    }
    u32 GetStorageBufferAlignment() const {
        return alignment;
    }
    u32 GetUniformBufferAlignment() const {
        return 1;
    }
    bool IsFree(u64 tick) const {
        return tick <= completed_tick;
    }
    u64 CurrentTick() const {
        return submitted_tick + 1;
    }
    void WaitSubmittedTick(u64) {}
    void ClearBuffer(Buffer& buffer, u32 offset, size_t size, u32 value) {
        std::memset(buffer.bytes.data() + offset, static_cast<int>(value), size);
    }
    bool CanReorderUpload(const Buffer&, std::span<VideoCommon::BufferCopy>) const {
        return false;
    }
    Staging DownloadStagingBuffer(u64 size, bool = false) {
        // Kept until the end: read downloads hold theirs past the next request.
        stagings.push_back(std::make_unique<Buffer>(*this, 0, size + 128, false));
        ++allocations;
        Buffer* const staging = stagings.back().get();
        return {staging, 128, std::span{staging->bytes}.subspan(128)};
    }
    Staging UploadStagingBuffer(u64 size) {
        return DownloadStagingBuffer(size);
    }
    void FreeDeferredStagingBuffer(Staging&) {}
    void CopyBuffer(Buffer* dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst->bytes.size());
            pending.push_back([dst, &src, copy] {
                std::memcpy(dst->bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                            copy.size);
            });
            ++downloads;
        }
    }
    void CopyBuffer(Buffer& dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        CopyBuffer(dst, &src, copies, true, true);
    }
    void CopyBuffer(Buffer& dst, Buffer* src, std::span<const VideoCommon::BufferCopy> copies, bool,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src->bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src->bytes.data() + copy.src_offset,
                        copy.size);
            ++uploads;
            upload_bytes += copy.size;
        }
    }
    void Finish() {
        if (before_completion) {
            before_completion();
        }
        Complete(submitted_tick);
        for (auto& copy : pending) {
            copy();
        }
        pending.clear();
        ++finishes;
    }
    u64 SubmitForHostRead() {
        ++host_read_barriers;
        submitted.emplace_back(++submitted_tick, std::move(pending));
        pending.clear();
        return submitted_tick;
    }
    u64 QueueForHostRead() {
        ++queued_host_reads;
        submitted.emplace_back(++submitted_tick, std::move(pending));
        pending.clear();
        return submitted_tick;
    }
    bool IsFreeNow(u64 tick) const {
        return IsFree(tick);
    }
    void WaitForHostRead(u64 tick) {
        if (before_completion) {
            before_completion();
        }
        Complete(tick);
        ++host_read_waits;
    }
    /// The host GPU runs what was submitted up to `tick`.
    void Complete(u64 tick) {
        while (!submitted.empty() && submitted.front().first <= tick) {
            for (auto& copy : submitted.front().second) {
                copy();
            }
            submitted.pop_front();
        }
        completed_tick = (std::max)(completed_tick, tick);
    }
    void Wait(u64 tick) {
        if (before_completion) {
            before_completion();
        }
        waited_tick = tick;
        ++waits;
    }
    void FinishForHostRead() {
        ++host_read_barriers;
        Finish();
    }
    void PreCopyBarrier() {}
    void PostCopyBarrier() {}
    void BindUniformBuffer(Buffer&, u32, u32) {}
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        stream.resize(size);
        return stream;
    }

    u32 alignment{256};
    u32 allocations{}, downloads{}, uploads{}, finishes{}, waits{};
    u32 host_read_barriers{}, host_read_waits{}, queued_host_reads{};
    u64 upload_bytes{}, waited_tick{};
    u64 submitted_tick{}, completed_tick{};
    std::vector<std::unique_ptr<Buffer>> stagings;
    std::vector<u8> stream;
    std::vector<std::function<void()>> pending;
    std::deque<std::pair<u64, std::vector<std::function<void()>>>> submitted;
    std::function<void()> before_completion;
};

struct Tracker : VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager> {
    using MemoryTrackerBase::MemoryTrackerBase;
    void MarkRegionAsCpuModified(DAddr address, u64 size) {
        ++cpu_marks;
        MemoryTrackerBase::MarkRegionAsCpuModified(address, size);
    }
    u32 cpu_marks{};
};

struct Policy {
    using Runtime = ::Runtime;
    using Buffer = ::Buffer;
    using Async_Buffer = Staging;
    using MemoryTracker = Tracker;
    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using Cache = VideoCommon::BufferCache<Policy>;

/// Streams small uniform buffers as the Vulkan runtime does (StreamUniformReusable).
struct KeepRuntime : Runtime {
    struct StreamUniformRef {
        size_t allocation;
    };
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        allocations.emplace_back(size);
        last = bound = StreamUniformRef{allocations.size() - 1};
        return allocations.back();
    }
    StreamUniformRef LastStreamUniformRef() const noexcept {
        return last;
    }
    void RebindStreamUniform(StreamUniformRef ref, u32 size) {
        REQUIRE(allocations.at(ref.allocation).size() == size);
        bound = ref;
    }
    u64 CurrentTick() const noexcept {
        return tick;
    }
    std::vector<std::vector<u8>> allocations;
    StreamUniformRef last{}, bound{};
    u64 tick{1};
};

struct KeepPolicy : Policy {
    using Runtime = ::KeepRuntime;
};

using KeepCache = VideoCommon::BufferCache<KeepPolicy>;

} // namespace

namespace VideoCommon {

struct BufferCacheTestAccess {
    static Binding Bind(Cache& cache, Tegra::MemoryManager& memory, GPUVAddr descriptor,
                        bool written = false, bool native = false, bool table = false,
                        bool compute = false) {
        cache.gpu_memory = &memory;
        return cache.StorageBufferBinding(descriptor, 0, written, native, table, compute);
    }
    template <class C>
    static BufferId Add(C& cache, DAddr address, u32 size, bool mapped = false) {
        const auto id = cache.slot_buffers.insert(cache.runtime, address, size, mapped);
        cache.Register(id);
        cache.memory_tracker.ForEachUploadRange(address, size, [](auto, auto) {});
        return id;
    }
    template <class C>
    static Buffer& Get(C& cache, BufferId id) {
        return cache.slot_buffers[id];
    }
    /// Add without the upload: guest memory not yet read into the buffer.
    static BufferId Register(Cache& cache, DAddr address, u32 size) {
        const auto id = cache.slot_buffers.insert(cache.runtime, address, size, false);
        cache.Register(id);
        return id;
    }
    static void Dirty(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsGpuModified(address, size);
        cache.gpu_modified_ranges.Add(address, size);
        cache.uncommitted_gpu_modified_ranges.Add(address, size);
        cache.committed_gpu_modified_ranges.emplace_back().Add(address, size);
        cache.pending_download_ranges.emplace_back().Add(address, size);
    }
    static bool Pending(Cache& cache, DAddr address, u64 size) {
        if (cache.gpu_modified_ranges.Intersects(address, size) ||
            cache.uncommitted_gpu_modified_ranges.Intersects(address, size)) {
            return true;
        }
        for (const auto& ranges : cache.committed_gpu_modified_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        for (const auto& ranges : cache.pending_download_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        return false;
    }
    static void DrawWrite(Cache& cache, DAddr page) {
        std::scoped_lock lock{cache.nx2_list_mutex};
        auto& entry = cache.nx2_list_pages[page];
        entry.draw_written = true;
        ++entry.writes;
    }
    static void Download(Cache& cache, std::span<const std::pair<DAddr, u64>> ranges) {
        cache.DownloadRangesMemory(ranges);
    }
    /// A draw or dispatch binds [address, address + size) of buffer `id`: its CPU-modified bytes
    /// are uploaded.
    static void Upload(Cache& cache, BufferId id, DAddr address, u32 size) {
        (void)cache.SynchronizeBuffer(cache.slot_buffers[id], address, size);
    }
    /// GPU addresses resolve through `memory` (ObtainBuffer).
    static void UseGpuMemory(Cache& cache, Tegra::MemoryManager& memory) {
        cache.gpu_memory = &memory;
    }
    /// A guest write the buffer cache doesn't hear of, on a page the tracker counts as written.
    static void MarkCpuModified(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsCpuModified(address, size);
    }
    static void Uniform(Cache& cache, BufferCacheChannelInfo& channel, BufferId id, DAddr address,
                        u32 size) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
        channel.uniform_buffers[0][0] = {.device_addr = address, .size = size, .buffer_id = id};
        cache.BindHostGraphicsUniformBuffer(0, 0, 0, true);
    }
    static void NextFrame(Cache& cache) {
        cache.native_indirect_uniforms_this_frame.clear();
    }
    static u32 CpuMarks(Cache& cache) {
        return cache.memory_tracker.cpu_marks;
    }
    /// A draw that binds no uniform buffer.
    static void KeepDrawWithout(KeepCache& cache, const Nx2FastDraw::Delta* delta) {
        cache.BeginGraphicsUniforms(delta);
    }
    static void KeepDraw(KeepCache& cache, BufferCacheChannelInfo& channel,
                         const Nx2FastDraw::Delta* delta, BufferId id, DAddr address,
                         u32 size) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
        cache.BeginGraphicsUniforms(delta);
        channel.uniform_buffers[0][0] = {.device_addr = address, .size = size, .buffer_id = id};
        cache.BindHostGraphicsUniformBuffer(0, 0, 0, true);
    }
    struct KeepCounts {
        u64 kept, streamed, verified, mismatches;
    };
    static KeepCounts Counts(const KeepCache& cache) {
        const auto& keep = cache.uniform_keep;
        return {keep.kept, keep.streamed_binds, keep.verified, keep.mismatches};
    }
    static void GpuWrite(KeepCache& cache, BufferId id, DAddr address, u32 size) {
        cache.MarkWrittenBuffer(id, address, size);
    }
    static void MarkWritten(Cache& cache, BufferId id, DAddr address, u32 size) {
        cache.MarkWrittenBuffer(id, address, size);
    }
    static size_t ReadDownloads(Cache& cache) {
        return cache.read_downloads.size();
    }
    /// Every read download compared with a synchronous flush (EDEN_NX2_READ_CHECK=1).
    static void CheckEveryRead(Cache& cache) {
        cache.read_check.every = 1;
    }
    static const auto& ReadChecks(Cache& cache) {
        return cache.read_check;
    }
    static BufferCacheChannelInfo& Channel(Cache& cache, s32 bind_id) {
        return cache.channel_storage[cache.channel_map.find(bind_id)->second];
    }
    static void Delete(Cache& cache, BufferId id) {
        cache.DeleteBuffer(id);
    }
    static void UpdateUniforms(Cache& cache, size_t stage) {
        cache.UpdateUniformBuffers(stage);
    }
};

} // namespace VideoCommon

namespace {

using Access = VideoCommon::BufferCacheTestAccess;

/// EDEN_NX2_READ_CHECK for a test, restored after it.
struct ReadCheckScope {
    explicit ReadCheckScope(u64 every)
        : previous{VideoCore::AsyncReadback::g_read_check_every.exchange(every)} {}
    ~ReadCheckScope() {
        VideoCore::AsyncReadback::g_read_check_every = previous;
    }
    u64 previous;
};

template <class R, class C>
struct BasicFixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    std::optional<Tegra::MemoryManager> memory;
    R runtime;
    std::unique_ptr<C> cache;
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    u8* bytes{};

    BasicFixture() {
        system.Initialize();
        process = std::make_unique<Kernel::KProcess>(system.Kernel());
        auto& table = process->GetPageTable().GetImpl();
        table.Resize(24, 12);
        bytes = physical.GetPointerFromRaw<u8>(0x10000);
        std::memset(bytes, 0, 0x40000);
        for (u64 page = 0; page < 64; ++page) {
            table.entries.GetAndFault(0x10 + page)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(bytes) - 0x10000);
        }
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device.RegisterProcess(&process->GetMemory());
        device.BindInterface(&rasterizer);
        device.Map(0x10000, 0x10000, 0x40000, asid);
        ResetMemory();
        cache = std::make_unique<C>(device, runtime);
    }
    ~BasicFixture() {
        cache.reset();
        memory.reset();
        device.Unmap(0x10000, 0x40000);
        device.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(process.get());
    }
    void ResetMemory() {
        memory.reset();
        memory.emplace(system, device, 32, 1ULL << 31, 16);
        memory->BindRasterizer(&rasterizer);
        memory->Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    }
    void Descriptor(GPUVAddr address, u64 size) {
        std::memcpy(bytes, &address, sizeof(address));
        std::memcpy(bytes + 8, &size, sizeof(size));
    }
    VideoCommon::Binding Bind(bool written = false, bool native = false, bool table = false,
                              bool compute = false) {
        return Access::Bind(*cache, *memory, 0x100000, written, native, table, compute);
    }
};

using Fixture = BasicFixture<Runtime, Cache>;

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct EnvGate {
    explicit EnvGate(const char* name_) : name{name_} {
        if (name != nullptr) {
            SetEnv(name, "1");
        }
    }
    ~EnvGate() {
        if (name != nullptr) {
            SetEnv(name, nullptr);
        }
    }
    const char* name;
};

/// A buffer cache built with the switch `gate` set (none if null), drawing with one 0x100-byte
/// uniform buffer at stage 0 slot 0, device 0x10800 (GPU 0x100800).
struct KeepFixture : EnvGate, BasicFixture<KeepRuntime, KeepCache> {
    static constexpr DAddr Ubo = 0x10800;
    static constexpr GPUVAddr UboGpu = 0x100800;
    static constexpr u32 UboSize = 0x100;

    explicit KeepFixture(const char* gate) : EnvGate{gate} {
        state.memory_manager = &*memory;
        state.payload.emplace(system, *memory, state);
        channel.emplace(state);
        sizes[0][0] = UboSize;
        channel->uniform_buffer_sizes = &sizes;
        id = Access::Add(*cache, 0x10000, 0x10000);
    }
    u8* UboBytes() {
        return bytes + (Ubo - 0x10000);
    }
    /// Draws and checks that the bound uniform data is what guest memory holds.
    void Draw(const VideoCommon::Nx2FastDraw::Delta* delta) {
        Access::KeepDraw(*cache, *channel, delta, id, Ubo, UboSize);
        const auto& data = runtime.allocations.at(runtime.bound.allocation);
        REQUIRE(data.size() == UboSize);
        REQUIRE(std::memcmp(data.data(), UboBytes(), UboSize) == 0);
    }
    Access::KeepCounts Counts() const {
        return Access::Counts(*cache);
    }

    Tegra::Control::ChannelState state{0};
    std::optional<VideoCommon::BufferCacheChannelInfo> channel;
    VideoCommon::UniformBufferSizes sizes{};
    VideoCommon::BufferId id;
};

/// Inline constant data over device [begin, end), through GPU address begin + UboGpu - Ubo.
VideoCommon::Nx2FastDraw::Delta Written(DAddr begin, DAddr end) {
    VideoCommon::Nx2FastDraw::Delta delta{};
    delta.write_begin[0] = begin + (KeepFixture::UboGpu - KeepFixture::Ubo);
    delta.write_end[0] = end + (KeepFixture::UboGpu - KeepFixture::Ubo);
    delta.num_writes = 1;
    delta.device_begin[0] = begin;
    delta.device_end[0] = end;
    delta.num_device_writes = 1;
    return delta;
}

/// Hands FinishRead to a buffer cache, as RasterizerVulkan does.
class ReadRasterizer final : public VideoCore::RasterizerInterface {
public:
    explicit ReadRasterizer(Cache& cache_) : cache{cache_} {}

    void FinishRead(DAddr addr, u64 size) override {
        cache.FinishReadDownloads(addr, size);
    }
    void Draw(bool, u32) override {}
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&&) override {}
    void SyncOperation(std::function<void()>&&) override {}
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override {
        return false;
    }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, true};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(DAddr, u64) override {}
    bool OnCPUWrite(DAddr, u64) override {
        return false;
    }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override {
        std::terminate();
    }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    Cache& cache;
};

} // namespace

TEST_CASE("Storage binding cache tracks map changes and descriptor dependencies",
          "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x200000;
    f.memory->Map(base, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base + 0x123, 0x456);
    for (int i = 0; i < 3; ++i) {
        const auto binding = f.Bind(true);
        REQUIRE(binding.device_addr == 0x20100);
        REQUIRE(binding.gpu_addr == base + 0x100);
        REQUIRE(binding.size == 0x479);
        REQUIRE(binding.buffer_id == VideoCommon::BufferId{});
        REQUIRE(binding.segment_count == 0);
    }
    REQUIRE(f.Bind().size == 0xf00);
    f.Descriptor(base + 0x123, 0x1456);
    REQUIRE(f.Bind(true).size == 0x1479);
    f.runtime.alignment = 4096;
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    REQUIRE(f.Bind(true).size == 0x1579);
    REQUIRE(f.Bind(true, true).size == 0x10000);
    f.memory->Map(base, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
    f.memory->MapSparse(base, 0x10000, true);
    REQUIRE(f.Bind().size == 0);
    f.memory->Unmap(base, 0x10000);
    REQUIRE(f.Bind().size == 0);
    f.memory->Map(base, 0x20000, 0x2000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x2000);
    f.memory->Map(base + 0x1000, 0x40000, 0x1000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x1000);
    const auto old_owner = f.memory->GetID();
    f.ResetMemory();
    f.memory->Map(base, 0x40000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.memory->GetID() != old_owner);
    REQUIRE(f.Bind(true).device_addr == 0x40000);
    f.Descriptor(0, 0x123);
    REQUIRE(f.Bind().size == 0);
}

TEST_CASE("Storage binding cache preserves native and table size policies", "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x4000000;
    f.memory->Map(base, 0x10000000, 80 << 20, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base, 256);
    REQUIRE(f.Bind().size == 4096);
    REQUIRE(f.Bind(false, true).size == 8 << 20);
    REQUIRE(f.Bind(false, true, false, true).size == 8 << 20);
    REQUIRE(f.Bind(true, true, false, true).size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 80 << 20);
    f.Descriptor(base, 40 << 20);
    REQUIRE(f.Bind(false, true, false, true).size == 40 << 20);
    REQUIRE(f.Bind(true, true, false, true).size == 8 << 20);
    REQUIRE(f.Bind(false, true).size == 8 << 20);
    f.Descriptor(base, 100 << 20);
    REQUIRE(f.Bind(false, true, false, true).size == 8 << 20);
    f.Descriptor(base, 256);
    REQUIRE(f.Bind(true, false, true).size == 256);
    f.Descriptor(base, (1ULL << 32) | 256);
    REQUIRE(f.Bind().size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 80 << 20);
    f.Descriptor(base + (79 << 20), 0);
    REQUIRE(f.Bind(false, false, true).size == 1 << 20);
}

TEST_CASE("Storage binding cache does not reuse a destroyed address space", "[buffer_ports]") {
    Fixture f;
    f.memory->Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(0x200000, 0x1000);
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    const auto serial = f.memory->MapSerial();
    const auto owner = f.memory->GetID();
    const auto* address = &*f.memory;
    f.ResetMemory();
    f.memory->Map(0x200000, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(&*f.memory == address);
    REQUIRE(f.memory->MapSerial() == serial);
    REQUIRE(f.memory->GetID() != owner);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
}

TEST_CASE("Batch downloads clamp buffers, deduplicate pages and finish before publication",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, first).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, second).bytes, 0x34);
    Access::Dirty(*f.cache, 0x2f000, 0x2000);
    Access::Dirty(*f.cache, 0x32000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.bytes[0x1f000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
    };
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x2f008, 0x1000},
        {0x2f000, 0x2000},
        {0x20000, 0},
    }};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 1);
    REQUIRE(f.runtime.downloads == 2);
    REQUIRE(f.bytes[0x1f000] == 0x12);
    REQUIRE(f.bytes[0x1ffff] == 0x12);
    REQUIRE(f.bytes[0x20000] == 0x34);
    REQUIRE(f.bytes[0x20fff] == 0x34);
    REQUIRE(f.bytes[0x22000] == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x2f000, 0x2000));
    REQUIRE(Access::Pending(*f.cache, 0x32000, 0x1000));
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Mapped and mixed batches make device writes host visible in one completion",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000, true);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000, true);
    const auto clean = Access::Add(*f.cache, 0x40000, 0x10000, true);
    auto& a = Access::Get(*f.cache, first);
    auto& b = Access::Get(*f.cache, second);
    a.tick = 8;
    b.tick = 19;
    Access::Get(*f.cache, clean).tick = 99;
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x20000, 0x1000},
        {0x30000, 0x1000},
        {0x40000, 0x1000},
    }};
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
        std::ranges::fill(a.bytes, 0x56);
        std::ranges::fill(b.bytes, 0x78);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.host_read_barriers == 1);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0x56);
    REQUIRE(f.bytes[0x20000] == 0x78);
    b.host_mapped = false;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 2);
        REQUIRE(f.bytes[0x10000] == 0x56);
        REQUIRE(f.bytes[0x20000] == 0x78);
        std::ranges::fill(a.bytes, 0x9a);
        std::ranges::fill(b.bytes, 0xbc);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.finishes == 2);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.bytes[0x10000] == 0x9a);
    REQUIRE(f.bytes[0x20000] == 0xbc);
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.host_read_barriers == 2);
    REQUIRE(f.runtime.finishes == 2);
}

TEST_CASE("Single mapped download completes host visibility before guest publication",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, true);
    auto& buffer = Access::Get(*f.cache, id);
    buffer.tick = 7;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        std::ranges::fill(buffer.bytes, 0xcd);
    };
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0xcd);
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Command page batch publishes only completed matching write generations",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x3000);
    Access::DrawWrite(*f.cache, 0x20000);
    Access::DrawWrite(*f.cache, 0x21000);
    Access::DrawWrite(*f.cache, 0x22000);
    std::array<u8, 4> out{};
    using Read = Cache::Nx2ListRead;
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Unknown);
        Access::DrawWrite(*f.cache, 0x22000);
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == (mapped ? 0U : 3U));
    REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x22000, out, true) == Read::Unknown);
    REQUIRE(f.bytes[0x12000] == 0x42);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, false) == Read::NeedsSync);
    Access::DrawWrite(*f.cache, 0x21000);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Unknown);
}

TEST_CASE("Command page readers remain ordered during batch completion", "[buffer_ports]") {
    struct {
        Core::DeviceMemory physical;
        Tegra::MaxwellDeviceMemoryManager device{physical};
        Runtime runtime;
        std::unique_ptr<Cache> cache = std::make_unique<Cache>(device, runtime);
    } f;
    Access::Add(*f.cache, 0x20000, 0x10000);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::DrawWrite(*f.cache, 0x20000);
    std::barrier rendezvous{2};
    Cache::Nx2ListRead before{}, after{};
    std::jthread reader{[&] {
        std::array<u8, 4> out{};
        rendezvous.arrive_and_wait();
        before = f.cache->Nx2ReadListCopy(0x20000, out, true);
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
        after = f.cache->Nx2ReadListCopy(0x20000, out, true);
    }};
    f.runtime.before_completion = [&] {
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    rendezvous.arrive_and_wait();
    reader.join();
    REQUIRE(before == Cache::Nx2ListRead::Unknown);
    REQUIRE(after == Cache::Nx2ListRead::Synced);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == 1);
}

TEST_CASE("Native uniform banks upload directly without overwriting GPU-owned bytes",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::memset(f.bytes + 0x10000, 0x5a, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x10000);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    REQUIRE(Access::Get(*f.cache, id).bytes[0xffff] == 0x5a);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    Access::NextFrame(*f.cache);
    Access::Dirty(*f.cache, 0x21000, 0x1000);
    std::memset(f.bytes + 0x10000, 0xa5, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(Access::Get(*f.cache, id).bytes[0x1000] == 0x5a);
}

TEST_CASE("Native uniform uploads stop at holes and keep unaligned bank boundaries",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x20000);
    std::memset(f.bytes + 0x10000, 0x6b, 0x20000);
    f.device.Unmap(0x22000, 0x1000);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x1edd);
    const auto& data = Access::Get(*f.cache, id).bytes;
    REQUIRE(data[0x122] == 0);
    REQUIRE(data[0x123] == 0x6b);
    REQUIRE(data[0x1fff] == 0x6b);
    REQUIRE(data[0x2000] == 0);
    REQUIRE(data[0x3000] == 0);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    f.device.Map(0x22000, 0x22000, 0x1000, f.asid);
    Access::NextFrame(*f.cache);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x11edd);
    REQUIRE(data[0x10122] == 0x6b);
    REQUIRE(data[0x10123] == 0);
}

TEST_CASE("Downloads keep CPU writes collected and not yet applied", "[buffer_ports]") {
    Fixture f;
    const auto staged = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto mapped = Access::Add(*f.cache, 0x30000, 0x10000, true);
    std::ranges::fill(Access::Get(*f.cache, staged).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, mapped).bytes, 0x34);
    f.cache->SetPendingCpuWrites([](const std::function<void(DAddr, size_t)>& callback) {
        callback(0x20100, 0x10);
        callback(0x30ff8, 0x10);
    });
    std::memset(f.bytes + 0x10100, 0xee, 0x10);
    std::memset(f.bytes + 0x20ff8, 0xee, 0x10);
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    Access::Dirty(*f.cache, 0x30000, 0x2000);
    const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x2000}}};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10100, f.bytes + 0x10110, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x10110] == 0x12);
    REQUIRE(f.bytes[0x11fff] == 0x12);
    f.cache->DownloadMemory(0x30000, 0x2000);
    REQUIRE(f.bytes[0x20ff7] == 0x34);
    REQUIRE(std::all_of(f.bytes + 0x20ff8, f.bytes + 0x21008, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x21008] == 0x34);
}

TEST_CASE("Uniform keep binds unwritten uniform buffers of delta draws without reading them",
          "[buffer_ports]") {
    KeepFixture f{nullptr};
    using Delta = VideoCommon::Nx2FastDraw::Delta;
    const Delta unchanged{};
    std::memset(f.UboBytes(), 1, KeepFixture::UboSize);
    f.Draw(nullptr);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    REQUIRE(f.runtime.allocations.size() == 1);

    // Inline constant data over the range, then a rebind of the slot.
    f.UboBytes()[0x10] = 2;
    const Delta written = Written(KeepFixture::Ubo + 0x10, KeepFixture::Ubo + 0x14);
    f.Draw(&written);
    REQUIRE(f.Counts().kept == 1);
    REQUIRE(f.runtime.allocations.size() == 2);
    Delta rebound{};
    rebound.binds[0] = 1;
    f.Draw(&rebound);
    REQUIRE(f.Counts().kept == 1);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 2);

    // Data written next to the range, and GPU writes elsewhere, leave it kept.
    const Delta beside = Written(KeepFixture::Ubo + 0x100, KeepFixture::Ubo + 0x140);
    f.Draw(&beside);
    REQUIRE(f.Counts().kept == 3);
    Access::GpuWrite(*f.cache, f.id, 0x18000, 0x100);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 4);

    // A GPU write over the range: the original path binds it from the buffer from now on.
    Access::GpuWrite(*f.cache, f.id, KeepFixture::Ubo + 0x80, 4);
    const size_t allocations = f.runtime.allocations.size();
    Access::KeepDraw(*f.cache, *f.channel, &unchanged, f.id, KeepFixture::Ubo, KeepFixture::UboSize);
    Access::KeepDraw(*f.cache, *f.channel, &unchanged, f.id, KeepFixture::Ubo, KeepFixture::UboSize);
    REQUIRE(f.Counts().kept == 4);
    REQUIRE(f.runtime.allocations.size() == allocations);
}

TEST_CASE("Uniform keep ends at CPU writes the caches are told about", "[buffer_ports]") {
    KeepFixture f{nullptr};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    f.Draw(nullptr);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    f.UboBytes()[0x20] = 7;
    f.cache->WriteMemory(KeepFixture::Ubo + 0x20, 1);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    f.cache->WriteMemory(0x18000, 0x10);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 2);
}

TEST_CASE("Uniform keep never keeps across command buffers", "[buffer_ports]") {
    KeepFixture f{nullptr};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    f.Draw(nullptr);
    ++f.runtime.tick;
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 0);
    REQUIRE(f.runtime.allocations.size() == 2);
    f.Draw(nullptr);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
}

TEST_CASE("Uniform keep verify counts guest writes the delta did not report",
          "[buffer_ports]") {
    KeepFixture f{"EDEN_NX2_UBO_KEEP_VERIFY"};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    f.Draw(nullptr);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().verified == 1);
    REQUIRE(f.Counts().mismatches == 0);
    f.UboBytes()[3] = 9;
    f.Draw(&unchanged);
    REQUIRE(f.Counts().verified == 2);
    REQUIRE(f.Counts().mismatches == 1);
    REQUIRE(f.Counts().kept == 0);
}

TEST_CASE("Uniform keep verify checks only the ranges a keep would bind", "[buffer_ports]") {
    KeepFixture f{"EDEN_NX2_UBO_KEEP_VERIFY"};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    f.Draw(nullptr);
    // A new command buffer: the draw streams the range again instead of keeping it.
    ++f.runtime.tick;
    f.UboBytes()[3] = 9;
    f.Draw(&unchanged);
    REQUIRE(f.Counts().verified == 0);
    REQUIRE(f.Counts().mismatches == 0);
}

TEST_CASE("Uniform keep binds unwritten uniform buffers streamed by earlier draws",
          "[buffer_ports]") {
    KeepFixture f{nullptr};
    using Delta = VideoCommon::Nx2FastDraw::Delta;
    const Delta unchanged{};
    std::memset(f.UboBytes(), 3, KeepFixture::UboSize);
    f.Draw(nullptr);
    // Draws without the slot, and inline data elsewhere, leave its allocation to the next bind.
    Access::KeepDrawWithout(*f.cache, &unchanged);
    const Delta elsewhere = Written(KeepFixture::Ubo + 0x200, KeepFixture::Ubo + 0x210);
    Access::KeepDrawWithout(*f.cache, &elsewhere);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    REQUIRE(f.runtime.allocations.size() == 1);

    // Inline data over it in a draw without the slot: the next bind reads it again.
    f.UboBytes()[0x40] = 4;
    const Delta over = Written(KeepFixture::Ubo + 0x40, KeepFixture::Ubo + 0x44);
    Access::KeepDrawWithout(*f.cache, &over);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    REQUIRE(f.runtime.allocations.size() == 2);

    // Writes the draws did not all hear of: a full delta, or a draw that never bound its uniforms.
    Delta overflowed{};
    overflowed.no_uniform_keeps = true;
    f.Draw(&overflowed);
    f.Draw(nullptr);
    REQUIRE(f.Counts().kept == 1);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 2);
}

TEST_CASE("Uniform keep ends where the CPU may have synchronised and at memory map changes",
          "[buffer_ports]") {
    KeepFixture f{nullptr};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    f.Draw(nullptr);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);

    // A fence: the guest may have rewritten the buffer since without the caches seeing it.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    f.UboBytes()[0x10] = 5;
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 2);
    // State changes between draws are no such point.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Register);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 3);

    f.memory->Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.UboBytes()[0x20] = 6;
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 3);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 4);
}

TEST_CASE("Uniform keep ends at DMA copies, DMA clears and inline uploads over the range",
          "[buffer_ports]") {
    KeepFixture f{nullptr};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    f.Draw(nullptr);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);

    // From memory no GPU wrote: the copy lands in guest memory only.
    std::memset(f.bytes + 0x8000, 8, 0x10);
    REQUIRE(f.cache->DMACopy(0x108000, KeepFixture::UboGpu + 0x20, 0x10));
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 1);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 2);

    // The engine writes guest memory after the cache cleared the buffer.
    REQUIRE(f.cache->DMAClear(KeepFixture::UboGpu + 0x40, 4, 0x09090909));
    std::memset(f.UboBytes() + 0x40, 9, 0x10);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 2);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 3);

    // An inline upload in place, allowed by GPU data on the next page.
    Access::GpuWrite(*f.cache, f.id, 0x11000, 4);
    constexpr DAddr inline_begin = KeepFixture::Ubo + 0xf0;
    std::vector<u8> data(0x11010 - inline_begin, 10);
    REQUIRE(f.cache->InlineMemory(inline_begin, data.size(), data));
    std::memcpy(f.UboBytes() + 0xf0, data.data(), 0x10);
    f.Draw(&unchanged);
    REQUIRE(f.Counts().kept == 3);
}

TEST_CASE("Uniform keep ends when the draw's GPU memory is another one", "[buffer_ports]") {
    KeepFixture f{nullptr};
    const VideoCommon::Nx2FastDraw::Delta unchanged{};
    Tegra::MemoryManager other{f.system, f.device, 32, 1ULL << 31, 16};
    other.BindRasterizer(&f.rasterizer);
    other.Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(other.HostPointerSerial() == f.memory->HostPointerSerial());
    Tegra::Control::ChannelState state{1};
    state.memory_manager = &other;
    state.payload.emplace(f.system, other, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    channel.uniform_buffer_sizes = &f.sizes;

    f.Draw(nullptr);
    // Its serial being the same number says nothing about the device memory behind the range.
    f.UboBytes()[0x30] = 11;
    Access::KeepDraw(*f.cache, channel, &unchanged, f.id, KeepFixture::Ubo, KeepFixture::UboSize);
    REQUIRE(f.Counts().kept == 0);
    const auto& data = f.runtime.allocations.at(f.runtime.bound.allocation);
    REQUIRE(std::memcmp(data.data(), f.UboBytes(), KeepFixture::UboSize) == 0);
}

TEST_CASE("Constant buffer data skips WriteMemory only on pages no other buffer use reached",
          "[buffer_ports]") {
    Fixture f;
    // A buffer that only streamed uniform buffers read: its pages are still CPU-modified.
    const auto id = Access::Register(*f.cache, 0x20000, 0x10000);
    REQUIRE_FALSE(f.cache->ConstantDataReachesBuffers(0x20100, 0x40));
    REQUIRE_FALSE(f.cache->ConstantDataReachesBuffers(0x22000, 0x1000));

    // A vertex, index or storage buffer (or the cached uniform path) uploads a range of it: those
    // pages need the CPU-modified mark again.
    Access::Upload(*f.cache, id, 0x20f00, 0x200);
    REQUIRE(f.cache->ConstantDataReachesBuffers(0x20000, 4));
    REQUIRE(f.cache->ConstantDataReachesBuffers(0x21ffc, 4));
    REQUIRE_FALSE(f.cache->ConstantDataReachesBuffers(0x22000, 0x1000));
    REQUIRE(f.cache->ConstantDataReachesBuffers(0x21f00, 0x200));

    // GPU-written pages keep their downloads and GPU-modified ranges right.
    Access::MarkWritten(*f.cache, id, 0x24010, 0x10);
    REQUIRE(f.cache->ConstantDataReachesBuffers(0x24ff0, 4));
    REQUIRE_FALSE(f.cache->ConstantDataReachesBuffers(0x25000, 4));
}

TEST_CASE("Read downloads stay GPU-modified until the reading thread writes them",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 0);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.runtime.host_read_barriers == 1);
    REQUIRE(VideoCore::g_read_downloads == 1);
    REQUIRE(f.bytes[0x10000] == 0);
    REQUIRE(Access::Pending(*f.cache, 0x20000, 0x1000));
    // Another read of the range waits for the same copy, and its flush copies nothing.
    const auto area = f.cache->GetFlushArea(0x20800, 4);
    REQUIRE(area);
    REQUIRE_FALSE(area->preemtive);
    REQUIRE(area->in_flight);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.runtime.submitted_tick == 1);
    f.cache->FinishReadDownloads(0x20800, 4);
    REQUIRE(f.runtime.host_read_waits == 1);
    REQUIRE(VideoCore::g_read_downloads == 0);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE(f.bytes[0x11000] == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
    REQUIRE(Access::Pending(*f.cache, 0x21000, 0x1000));
    REQUIRE(f.cache->GetFlushArea(0x20800, 4)->preemtive);
    f.cache->FinishReadDownloads(0x20000, 0x2000);
    REQUIRE(f.runtime.host_read_waits == 1);
}

TEST_CASE("Read downloads keep CPU writes made while they were on their way", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // After the submission: one write still collected, one already applied.
    std::memset(f.bytes + 0x10100, 0xee, 0x10);
    std::memset(f.bytes + 0x10200, 0xdd, 0x10);
    f.cache->SetPendingCpuWrites(
        [](const std::function<void(DAddr, size_t)>& callback) { callback(0x20100, 0x10); });
    const std::array<std::pair<DAddr, u64>, 1> applied{{{0x20200, 0x10}}};
    f.cache->ClaimCpuWrites(applied);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10100, f.bytes + 0x10110, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x10110] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10200, f.bytes + 0x10210, [](u8 b) { return b == 0xdd; }));
    REQUIRE(f.bytes[0x10210] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Indirect dispatch arguments remain GPU modified across a neighboring CPU upload",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x10000, 0x10000);
    auto &original = Access::Get(*f.cache, id);
    std::fill_n(original.bytes.begin() + 0x40, 12, u8{0x73});
    Access::Dirty(*f.cache, 0x10040, 12);
    Access::UseGpuMemory(*f.cache, *f.memory);
    const auto [arguments, offset] = VideoCommon::ObtainIndirectBuffer(*f.cache, 0x100040);
    REQUIRE(Access::Pending(*f.cache, 0x10040, 12));
    REQUIRE(arguments->bytes[offset] == 0x73);
    // A CPU write elsewhere on this tracked page must not re-upload the stale argument bytes.
    f.bytes[0x80] = 0x42;
    Access::MarkCpuModified(*f.cache, 0x10000, 0x1000);
    Access::Upload(*f.cache, id, 0x10000, 0x1000);
    REQUIRE(arguments->bytes[offset] == 0x73);
    REQUIRE(arguments->bytes[arguments->Offset(0x10080)] == 0x42);
    f.cache->DownloadMemory(0x10040, 12);
    REQUIRE(f.bytes[0x40] == 0x73);
}

TEST_CASE("Newer bytes that guest memory doesn't hold yet land the read downloads under them first",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    auto& counters = VideoCore::AsyncReadback::g_counters;
    const u64 landings = counters.clear_landings.load();
    const u64 waits = counters.clear_waits.load();
    SECTION("an overwritten range discards its old GPU contents") {
        (void)f.cache->ObtainCPUBuffer(0x20040, 12,
                                       VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                                       VideoCommon::ObtainBufferOperation::DiscardWrite);
    }
    SECTION("a DMA copy's destination (ClearDownload)") {
        f.cache->ClearDownload(0x20040, 12);
    }
    // The read that took the download finds the GPU's bytes, as its synchronous flush wrote them.
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x10040] == 0x12);
    REQUIRE(f.bytes[0x1004b] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE(f.runtime.host_read_waits == 1);
    REQUIRE(VideoCore::g_read_downloads == 0);
    // The landing went on the GPU thread and waited for the host GPU (the profile line counts it).
    REQUIRE(counters.clear_landings.load() - landings == 1);
    REQUIRE(counters.clear_waits.load() - waits == 1);
}

TEST_CASE("Read downloads land oldest first and leave newer GPU writes to download",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // The GPU thread's next work writes part of the range: on the host GPU, after the copy.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x100, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x100);
    const auto area = f.cache->GetFlushArea(0x20800, 4);
    REQUIRE_FALSE(area->preemtive);
    REQUIRE_FALSE(area->in_flight);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(f.runtime.downloads == 2);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    REQUIRE(f.cache->GetFlushArea(0x20800, 4)->in_flight);
    // The first copy holds what a synchronous flush would have left for the whole range.
    f.cache->FinishReadDownloads(0x20000, 4);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x12);
    REQUIRE(Access::Pending(*f.cache, 0x20800, 0x100));
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x800));
    f.cache->FinishReadDownloads(0x20800, 4);
    REQUIRE(f.bytes[0x107ff] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE(f.bytes[0x108ff] == 0x34);
    REQUIRE(f.bytes[0x10900] == 0x12);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
    // Both written in order even when the newer one's reader comes first.
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x100, u8{0x56}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x100);
    f.cache->DownloadMemoryForRead(0x20800, 0x100);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    f.cache->FinishReadDownloads(0x20800, 4);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10800] == 0x56);
    REQUIRE(f.bytes[0x10900] == 0x12);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Read downloads land before a later fence's downloads", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A later GPU write the next fence downloads, and the host GPU runs all of it before anyone
    // waits for the read's copy.
    f.runtime.pending.push_back([&buffer] { std::fill_n(buffer.bytes.begin(), 0x100, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->CommitAsyncFlushes();
    f.runtime.Finish();
    f.cache->PopAsyncFlushes();
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x10000] == 0x34);
    REQUIRE(f.bytes[0x100ff] == 0x34);
    REQUIRE(f.bytes[0x10100] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Preflushed ranges wait for the read download still on its way", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    // The first read marks the range for fence downloads, then flushes it.
    const auto first = f.cache->GetFlushArea(0x20000, 4);
    REQUIRE_FALSE(first->preemtive);
    REQUIRE_FALSE(first->in_flight);
    f.cache->DownloadMemoryForRead(first->start_address, first->end_address - first->start_address);
    const auto second = f.cache->GetFlushArea(0x20010, 4);
    REQUIRE_FALSE(second->preemtive);
    REQUIRE(second->in_flight);
    f.cache->FinishReadDownloads(second->start_address,
                                 second->end_address - second->start_address);
    REQUIRE(f.bytes[0x10010] == 0x12);
    REQUIRE(f.cache->GetFlushArea(0x20010, 4)->preemtive);
}

TEST_CASE("GPU writes to guest memory land the read downloads under them first", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A semaphore release, then an inline upload, after the copy was taken.
    gpu_memory.Write<u32>(0x200100, 0xdeadbeef);
    REQUIRE(VideoCore::g_read_downloads == 0);
    const u32 words = 0x55555555;
    gpu_memory.WriteBlockUnsafe(0x200200, &words, sizeof(words));
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10100, sizeof(value));
    REQUIRE(value == 0xdeadbeef);
    std::memcpy(&value, f.bytes + 0x10200, sizeof(value));
    REQUIRE(value == 0x55555555);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10104] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
}

TEST_CASE("Read downloads take again what a GPU write outside the cache's records changed",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // After the read's copy: a query result copy into the buffer (SyncValues), then a GPU write
    // elsewhere on the page, which the next read downloads.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    f.cache->OutdateReadDownloads(0x20040, 0x10);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x10, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x1003f] == 0x12);
    REQUIRE(f.bytes[0x10040] == 0x56);
    REQUIRE(f.bytes[0x1004f] == 0x56);
    REQUIRE(f.bytes[0x10050] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Switch 2 writes to guest memory land read downloads only on GPU-written pages",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    struct Nx2Title {
        Nx2Title() {
            Tegra::g_nx2_ampere = true;
        }
        ~Nx2Title() {
            Tegra::g_nx2_ampere = false;
            Tegra::Nx2ClearGpuWrittenPages();
        }
    } nx2_title;
    Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A page no GPU write reached: no download can be on its way there.
    gpu_memory.Write<u32>(0x201000, 0xdeadbeef);
    REQUIRE(VideoCore::g_read_downloads == 1);
    REQUIRE(f.runtime.host_read_waits == 0);
    gpu_memory.Write<u32>(0x200100, 0xcafef00d);
    REQUIRE(VideoCore::g_read_downloads == 0);
    REQUIRE(f.runtime.host_read_waits == 1);
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10100, sizeof(value));
    REQUIRE(value == 0xcafef00d);
    std::memcpy(&value, f.bytes + 0x11000, sizeof(value));
    REQUIRE(value == 0xdeadbeef);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
}

TEST_CASE("Read download check leaves out newer bytes over an earlier read's download",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::CheckEveryRead(*f.cache);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    // The first read takes the page's GPU-written bytes, the second only those written since.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x10, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    // Constant buffer data over bytes the first read brings: that read lands first, and the
    // buffer cache hears of the write at the next FlushCaching. Synchronous flushes leave it too.
    std::array<u8, 0x30> words;
    words.fill(0xb6);
    gpu_memory.WriteBlockCached(0x200060, words.data(), words.size());
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x1005f] == 0x12);
    REQUIRE(f.bytes[0x10060] == 0xb6);
    REQUIRE(f.bytes[0x1008f] == 0xb6);
    REQUIRE(f.bytes[0x10090] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).rewritten == 0x30);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check finds an earlier read's bytes the GPU changed unrecorded",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::CheckEveryRead(*f.cache);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A GPU write into the first read's bytes that nothing takes out of its download.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x10, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x10040] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0x10);
    REQUIRE(Access::ReadChecks(*f.cache).rewritten == 0);
}

namespace {
struct Nx2TitleScope {
    Nx2TitleScope() {
        Tegra::g_nx2_ampere = true;
    }
    ~Nx2TitleScope() {
        Tegra::g_nx2_ampere = false;
        Tegra::Nx2ClearGpuWrittenPages();
    }
};

/// A dispatch that writes `value` over [address, address + size) of the buffer at 0x20000.
void WriteByDispatch(Fixture& f, VideoCommon::BufferId id, DAddr address, u32 size, u8 value) {
    auto& buffer = Access::Get(*f.cache, id);
    f.runtime.pending.push_back([&buffer, address, size, value] {
        std::fill_n(buffer.bytes.begin() + static_cast<std::ptrdiff_t>(address - 0x20000), size,
                    value);
    });
    Access::MarkWritten(*f.cache, id, address, size);
    f.cache->NoteWorkRecorded(true);
}

/// A guest read of a word at `address`, as GPU::OnCPURead serves it.
VideoCore::RasterizerDownloadArea ReadWord(Fixture& f, DAddr address) {
    const auto area = f.cache->GetFlushArea(address, 4);
    REQUIRE(area);
    const u64 size = area->end_address - area->start_address;
    if (!area->preemtive && !area->in_flight) {
        f.cache->DownloadMemoryForRead(area->start_address, size);
    }
    if (!area->preemtive) {
        f.cache->FinishReadDownloads(area->start_address, size);
    }
    return *area;
}
} // namespace

TEST_CASE("Pages read after each GPU write get their downloads right behind the writes",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    // Learned from two writes the guest read; those reads still flush.
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        REQUIRE(f.runtime.queued_host_reads == 0);
        const auto area = ReadWord(f, 0x20010);
        REQUIRE_FALSE(area.preemtive);
        REQUIRE_FALSE(area.in_flight);
        REQUIRE(f.bytes[0x10010] == frame);
    }
    REQUIRE(f.runtime.host_read_waits == 2);
    // The next write's download goes right behind it, before any read.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    REQUIRE(f.runtime.queued_host_reads == 1);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    REQUIRE(f.bytes[0x10010] == 2);
    // A read before the host GPU ran it waits for that copy alone.
    const auto waiting = f.cache->GetFlushArea(0x20010, 4);
    REQUIRE(waiting->in_flight);
    f.cache->FinishReadDownloads(waiting->start_address,
                                 waiting->end_address - waiting->start_address);
    REQUIRE(f.runtime.host_read_waits == 3);
    REQUIRE(f.runtime.host_read_barriers == 2);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(f.bytes[0x100ff] == 3);
    // Once the host GPU ran it, the read finds the bytes in guest memory and waits for nothing.
    WriteByDispatch(f, id, 0x20000, 0x100, 4);
    REQUIRE(f.runtime.queued_host_reads == 2);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.runtime.host_read_waits == 3);
    REQUIRE(f.bytes[0x10010] == 4);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(VideoCore::g_read_downloads == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Early downloads leave what a newer GPU write changed to a download of its own",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    // Another dispatch writes part of it again before the read: it gets an early download too.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    WriteByDispatch(f, id, 0x20080, 0x40, 4);
    REQUIRE(f.runtime.queued_host_reads == 2);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x1007f] == 3);
    REQUIRE(f.bytes[0x10080] == 4);
    REQUIRE(f.bytes[0x100bf] == 4);
    REQUIRE(f.bytes[0x100c0] == 3);
    // A draw writes part of it again: the read downloads that part after the early download.
    WriteByDispatch(f, id, 0x20000, 0x100, 5);
    auto& buffer = Access::Get(*f.cache, id);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{6}); });
    Access::MarkWritten(*f.cache, id, 0x20040, 0x10);
    const auto area = ReadWord(f, 0x20010);
    REQUIRE_FALSE(area.preemtive);
    REQUIRE_FALSE(area.in_flight);
    REQUIRE(f.bytes[0x1003f] == 5);
    REQUIRE(f.bytes[0x10040] == 6);
    REQUIRE(f.bytes[0x1004f] == 6);
    REQUIRE(f.bytes[0x10050] == 5);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Early downloads keep CPU writes, and stop for pages no longer read", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    // The guest writes part of the page before the early download lands: its values stay.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    std::memset(f.bytes + 0x10020, 0xee, 0x10);
    const std::array<std::pair<DAddr, u64>, 1> applied{{{0x20020, 0x10}}};
    f.cache->ClaimCpuWrites(applied);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x1001f] == 3);
    REQUIRE(f.bytes[0x10020] == 0xee);
    REQUIRE(f.bytes[0x1002f] == 0xee);
    REQUIRE(f.bytes[0x10030] == 3);
    // Not read any more: a few more writes get early downloads, then none.
    const u32 queued = f.runtime.queued_host_reads;
    for (u8 frame = 4; frame < 12; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        f.runtime.Complete(f.runtime.submitted_tick);
    }
    REQUIRE(f.runtime.queued_host_reads - queued == 4);
    const auto area = ReadWord(f, 0x20010);
    REQUIRE_FALSE(area.preemtive);
    REQUIRE(f.bytes[0x10010] == 11);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("GPU writes to guest memory land the early downloads under them first",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    REQUIRE(VideoCore::g_read_downloads == 1);
    // A semaphore release into the copied bytes before the host GPU ran the copy.
    gpu_memory.Write<u32>(0x200040, 0xcafef00d);
    REQUIRE(VideoCore::g_read_downloads == 0);
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10040, sizeof(value));
    REQUIRE(value == 0xcafef00d);
    REQUIRE(f.bytes[0x1003f] == 3);
    REQUIRE(f.bytes[0x10044] == 3);
    REQUIRE(ReadWord(f, 0x20040).preemtive);
}

namespace {
/// A page read after each dispatch's write, which gets its next write's early download, and a
/// query result to 0x20040 that the query cache writes at the report's fence.
void ReportOnReadBackPage(Fixture& f, VideoCommon::BufferId id) {
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    REQUIRE(f.cache->NoteReportPending(0x20040, 4));
}

/// The report's fence callback (QueryCacheBase::CounterReport) writes its result, and ends it if
/// the buffer cache kept it (`noted`).
void WriteReportResult(Fixture& f, u32 result, bool noted) {
    f.cache->FinishReadDownloads(0x20040, 4);
    std::memcpy(f.bytes + 0x10040, &result, sizeof(result));
    if (noted) {
        f.cache->FinishReport(0x20040, 4);
    }
}
} // namespace

TEST_CASE("A query result older than a GPU write it lands over leaves the write to be read again",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    ReportOnReadBackPage(f, id);
    // A dispatch writes the result's words after the report, before its fence.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    SECTION("its early download landed by the result's own wait") {}
    SECTION("its early download landed before, once the host GPU ran it") {
        f.runtime.Complete(f.runtime.submitted_tick);
        f.cache->NoteWorkRecorded(true);
        REQUIRE(Access::ReadDownloads(*f.cache) == 0);
        REQUIRE(f.bytes[0x10040] == 3);
    }
    WriteReportResult(f, 0xcafef00d, true);
    // The dispatch's words are newer: the next read downloads them again.
    REQUIRE_FALSE(ReadWord(f, 0x20040).preemtive);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10043] == 3);
}

TEST_CASE("A query result newer than the GPU writes under it stays in guest memory",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    // The dispatch writes the words before the report.
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    const bool noted = f.cache->NoteReportPending(0x20040, 4);
    REQUIRE(noted);
    WriteReportResult(f, 0xcafef00d, noted);
    REQUIRE(ReadWord(f, 0x20040).preemtive);
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10040, sizeof(value));
    REQUIRE(value == 0xcafef00d);
}

TEST_CASE("A result the buffer cache didn't keep ends no other result's tracking",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    // R1 at 0x20040, reported before the page gets early downloads: not kept.
    const bool r1 = f.cache->NoteReportPending(0x20040, 4);
    REQUIRE_FALSE(r1);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    // R2 at the same words, then R1's callback.
    const bool r2 = f.cache->NoteReportPending(0x20040, 4);
    REQUIRE(r2);
    WriteReportResult(f, 0x11111111, r1);
    // A dispatch writes the words after R2's report, then R2's callback.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    WriteReportResult(f, 0x22222222, r2);
    REQUIRE_FALSE(ReadWord(f, 0x20040).preemtive);
    REQUIRE(f.bytes[0x10040] == 3);
}

TEST_CASE("Reports skip the lock only outside the read-back pages' bounds", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    REQUIRE_FALSE(f.cache->MayBeReadBackPage(0x20040));
    ReportOnReadBackPage(f, id);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x24000, 0x100, frame);
        ReadWord(f, 0x24010);
    }
    REQUIRE(f.cache->MayBeReadBackPage(0x20040));
    REQUIRE(f.cache->MayBeReadBackPage(0x24040));
    // Between the read-back pages: inside the bounds, so the locked check answers.
    REQUIRE(f.cache->MayBeReadBackPage(0x22040));
    REQUIRE_FALSE(f.cache->NoteReportPending(0x22040, 4));
    // Outside them: neither would keep the report.
    for (const DAddr addr : {DAddr{0x1f040}, DAddr{0x25040}}) {
        REQUIRE_FALSE(f.cache->MayBeReadBackPage(addr));
        REQUIRE_FALSE(f.cache->NoteReportPending(addr, 4));
    }
    f.cache->FinishReport(0x20040, 4);
}

TEST_CASE("Finished early downloads land a few draws later, or at the next dispatch",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    for (u32 draw = 1; draw < 32; ++draw) {
        f.cache->NoteWorkRecorded(false);
    }
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    f.cache->NoteWorkRecorded(false);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10010] == 3);
    // A draw writes it: the early download goes behind the next dispatch.
    auto& buffer = Access::Get(*f.cache, id);
    f.runtime.pending.push_back([&buffer] { std::fill_n(buffer.bytes.begin(), 0x100, u8{4}); });
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->NoteWorkRecorded(false);
    REQUIRE(f.runtime.queued_host_reads == 1);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(f.runtime.queued_host_reads == 2);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10010] == 4);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
}

TEST_CASE("Early downloads stop at 32 on their way", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x30000);
    // 33 pages, each read after two writes.
    for (u8 frame = 1; frame <= 2; ++frame) {
        for (DAddr page = 0x20000; page < 0x41000; page += 0x1000) {
            WriteByDispatch(f, id, page, 0x100, frame);
            ReadWord(f, page);
        }
    }
    // Their next writes before the host GPU ran any copy: the 33rd page's read flushes.
    for (DAddr page = 0x20000; page < 0x41000; page += 0x1000) {
        WriteByDispatch(f, id, page, 0x100, 3);
    }
    REQUIRE(f.runtime.queued_host_reads == 32);
    REQUIRE(Access::ReadDownloads(*f.cache) == 32);
    const auto last = ReadWord(f, 0x40010);
    REQUIRE_FALSE(last.preemtive);
    REQUIRE_FALSE(last.in_flight);
    REQUIRE(f.bytes[0x30010] == 3);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x21000));
}

TEST_CASE("Early downloads stop at 1 MiB a frame", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x1000, frame);
        ReadWord(f, 0x20010);
    }
    // A whole page each, read once the host GPU ran it: 256 make 1 MiB.
    for (u32 write = 0; write < 256; ++write) {
        WriteByDispatch(f, id, 0x20000, 0x1000, static_cast<u8>(write));
        f.runtime.Complete(f.runtime.submitted_tick);
        REQUIRE(ReadWord(f, 0x20010).preemtive);
    }
    REQUIRE(f.runtime.queued_host_reads == 256);
    // The frame's later writes leave their reads to flush.
    WriteByDispatch(f, id, 0x20000, 0x1000, 0xab);
    REQUIRE(f.runtime.queued_host_reads == 256);
    const auto area = ReadWord(f, 0x20010);
    REQUIRE_FALSE(area.preemtive);
    REQUIRE_FALSE(area.in_flight);
    REQUIRE(f.bytes[0x10010] == 0xab);
    REQUIRE(f.bytes[0x10fff] == 0xab);
}

TEST_CASE("Read download check compares what early downloads brought with a synchronous flush",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    const auto& checks = Access::ReadChecks(*f.cache);
    const u64 compared = checks.compared;
    // In guest memory before the read: the read flushes, and compares them as they landed.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    // The early download's own check and the read's.
    REQUIRE(checks.compared - compared == 0x200);
    // On its way at the read: compared as it lands.
    WriteByDispatch(f, id, 0x20000, 0x100, 4);
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x10010] == 4);
    REQUIRE(checks.compared - compared == 0x400);
    REQUIRE(checks.mismatched == 0);
    REQUIRE(checks.uncovered == 0);
    REQUIRE(checks.rewritten == 0);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Read download check finds early downloaded bytes the GPU changed unrecorded",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    // A GPU write into the landed bytes that nothing records: a synchronous flush at the read
    // would bring it.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    // The guest writes others: newer values, not compared.
    std::memset(f.bytes + 0x10080, 0xee, 0x8);
    ReadWord(f, 0x20010);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0x10);
    REQUIRE(Access::ReadChecks(*f.cache).rewritten == 0x8);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
    // Compared once: a later read finds nothing an early download brought since.
    const u64 compared = Access::ReadChecks(*f.cache).compared;
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(Access::ReadChecks(*f.cache).compared == compared);
}

namespace {
/// Last frame's values of 16 bytes the guest CPU writes each frame (the GPU's copy), and this
/// frame's: three bytes of each word differ, the last word's top two don't.
constexpr std::array<u8, 16> OldGuestValue{0x68, 0x34, 0x12, 0x3f, 0x8d, 0x11, 0x47, 0xbe,
                                           0x20, 0x55, 0x20, 0x41, 0x73, 0x0c, 0x40, 0x3f};
constexpr std::array<u8, 16> NewGuestValue{0xcd, 0x9a, 0x15, 0x3f, 0x95, 0x02, 0x49, 0xbe,
                                           0x4b, 0x13, 0x22, 0x41, 0x54, 0xe6, 0x40, 0x3f};

/// Guest CPU writes into GPU-written memory, as Core::Memory's HandleRasterizerWrite makes them
/// on any core: collected (GPUDirtyMemoryManager::Collect, in 64-byte blocks), then stored once
/// no download landing under them looked at the collected writes before.
class GuestCpuWrites {
public:
    explicit GuestCpuWrites(Cache& cache_) : cache{cache_} {
        cache.SetPendingCpuWrites([this](const std::function<void(DAddr, size_t)>& callback) {
            {
                std::scoped_lock lock{mutex};
                for (const auto& [address, size] : collected) {
                    callback(address, size);
                }
            }
            if (on_look) {
                on_look(++looks);
            }
        });
    }
    ~GuestCpuWrites() {
        cache.SetPendingCpuWrites({});
    }

    void Write(u8* guest, DAddr address, std::span<const u8> value) {
        const DAddr first = Common::AlignDown(address, 64);
        {
            std::scoped_lock lock{mutex};
            collected.emplace_back(first, Common::AlignUp(address + value.size(), 64) - first);
        }
        VideoCore::DownloadLanding::Await(address, value.size());
        std::memcpy(guest, value.data(), value.size());
    }

    /// GPU::InvalidateGPUCache: the collected writes are applied.
    void Apply() {
        std::vector<std::pair<DAddr, u64>> applied;
        {
            std::scoped_lock lock{mutex};
            applied.swap(collected);
        }
        cache.ClaimCpuWrites(applied);
    }

    /// Called after the buffer cache's n-th look at the collected writes.
    std::function<void(int)> on_look;
    int looks{};

private:
    Cache& cache;
    std::mutex mutex;
    std::vector<std::pair<DAddr, u64>> collected;
};

/// Another core writes NewGuestValue at `address` right after the buffer cache's `at`-th look at
/// the collected writes: the look returns once the write is stored or waits for a landing.
class RacingGuestWrite {
public:
    RacingGuestWrite(Fixture& f, GuestCpuWrites& writes_, DAddr address_, int at)
        : writes{writes_}, guest{f.bytes + (address_ - 0x10000)}, address{address_} {
        writes.looks = 0;
        writes.on_look = [this, at](int look) {
            if (look != at) {
                return;
            }
            const u64 waits = VideoCore::DownloadLanding::Waits();
            go.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!stored.load() && VideoCore::DownloadLanding::Waits() == waits &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
        };
        thread = std::thread([this] {
            while (!go.load() && !quit.load()) {
                std::this_thread::yield();
            }
            if (go.load()) {
                writes.Write(guest, address, NewGuestValue);
                stored.store(true);
            }
        });
    }
    ~RacingGuestWrite() {
        quit.store(true);
        thread.join();
        writes.on_look = {};
    }

    /// Whether the write was stored, waiting up to 5 s for it.
    bool Stored() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (go.load() && !stored.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        return stored.load();
    }

private:
    GuestCpuWrites& writes;
    u8* guest;
    DAddr address;
    std::atomic<bool> go{};
    std::atomic<bool> stored{};
    std::atomic<bool> quit{};
    std::thread thread;
};

/// The buffer at 0x20000, its first page written by a dispatch (the whole storage binding counts)
/// and read back, with OldGuestValue at 0x20a20 in the GPU's copy and in guest memory.
VideoCommon::BufferId GuestWrittenPage(Fixture& f) {
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    std::ranges::copy(OldGuestValue, buffer.bytes.begin() + 0xa20);
    std::ranges::copy(OldGuestValue, f.bytes + 0x10a20);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
    return id;
}

bool GuestHolds(const Fixture& f, DAddr address, std::span<const u8> value) {
    return std::equal(value.begin(), value.end(), f.bytes + (address - 0x10000));
}
} // namespace

TEST_CASE("A guest write stored while a read download lands keeps its bytes", "[buffer_ports]") {
    Fixture f;
    GuestWrittenPage(f);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    const u64 waits = VideoCore::DownloadLanding::Waits();
    {
        // Collected right after the landing looked at the collected writes.
        RacingGuestWrite race{f, writes, 0x20a20, 1};
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        REQUIRE(race.Stored());
    }
    REQUIRE(GuestHolds(f, 0x20a20, NewGuestValue));
    REQUIRE(f.bytes[0x10a1f] == 0x12);
    REQUIRE(f.bytes[0x10a30] == 0x12);
    // It waited for the landing to end.
    REQUIRE(VideoCore::DownloadLanding::Waits() > waits);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 1);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
}

TEST_CASE("Read download check leaves out a guest write stored while it compares",
          "[buffer_ports]") {
    Fixture f;
    GuestWrittenPage(f);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    {
        // The download is in guest memory: collected right after the check looked at the
        // collected writes, before it reads guest memory.
        RacingGuestWrite race{f, writes, 0x20a20, 2};
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        REQUIRE(race.Stored());
    }
    REQUIRE(GuestHolds(f, 0x20a20, NewGuestValue));
    REQUIRE(Access::ReadChecks(*f.cache).checked == 1);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
}

TEST_CASE("An early download landed at a draw keeps a guest write stored meanwhile",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x1000, frame);
        ReadWord(f, 0x20010);
    }
    // The next write gets its early download. The host GPU runs it, and the GPU thread lands it
    // at a draw while another core writes into the page.
    WriteByDispatch(f, id, 0x20000, 0x1000, 3);
    std::ranges::copy(OldGuestValue, buffer.bytes.begin() + 0xa20);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    {
        RacingGuestWrite race{f, writes, 0x20a20, 1};
        for (u32 draw = 0; draw < 32; ++draw) {
            f.cache->NoteWorkRecorded(false);
        }
        REQUIRE(Access::ReadDownloads(*f.cache) == 0);
        REQUIRE(race.Stored());
    }
    REQUIRE(GuestHolds(f, 0x20a20, NewGuestValue));
    REQUIRE(f.bytes[0x10a1f] == 3);
    REQUIRE(f.bytes[0x10a30] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
}

TEST_CASE("Synchronous downloads keep guest writes collected while they wait for the host GPU",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    GuestCpuWrites writes{*f.cache};
    // Collected and stored while the GPU thread waits for the host GPU's copies.
    f.runtime.before_completion = [&] {
        f.runtime.before_completion = {};
        writes.Write(f.bytes + 0x10100, 0x20100, NewGuestValue);
    };
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x2000}}};
    Access::Download(*f.cache, ranges);
    REQUIRE(GuestHolds(f, 0x20100, NewGuestValue));
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(f.bytes[0x10140] == 0x12);
    // The same through DownloadMemory, one buffer at a time.
    writes.Apply();
    f.runtime.before_completion = [&] {
        f.runtime.before_completion = {};
        writes.Write(f.bytes + 0x10200, 0x20200, NewGuestValue);
    };
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    f.cache->DownloadMemory(0x20000, 0x2000);
    REQUIRE(GuestHolds(f, 0x20200, NewGuestValue));
    REQUIRE(f.bytes[0x101ff] == 0x12);
    REQUIRE(f.bytes[0x10240] == 0x12);
}

TEST_CASE("Guest writes racing read downloads keep their bytes", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    u32 seed = 1234;
    const auto next_delay = [&seed] {
        seed = seed * 1103515245 + 12345;
        return (seed >> 8) % 4000;
    };
    const auto spin = [](u32 iterations) {
        for (volatile u32 i = 0; i < iterations; i = i + 1) {
        }
    };
    int lost = 0;
    for (int iteration = 0; iteration < 2000; ++iteration) {
        // The guest's earlier writes are applied, then a dispatch writes the page: 16 of its bytes
        // hold the GPU's value until the guest writes this round's from another core.
        writes.Apply();
        std::array<u8, 16> gpu_value;
        gpu_value.fill(static_cast<u8>(iteration * 2 + 1));
        std::array<u8, 16> cpu_value;
        cpu_value.fill(static_cast<u8>(iteration * 2 + 2));
        std::ranges::copy(gpu_value, buffer.bytes.begin() + 0xa20);
        Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
        f.cache->DownloadMemoryForRead(0x20000, 0x1000);
        const u32 writer_delay = next_delay();
        const u32 reader_delay = next_delay();
        std::thread writer{[&] {
            spin(writer_delay);
            writes.Write(f.bytes + 0x10a20, 0x20a20, cpu_value);
        }};
        spin(reader_delay);
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        writer.join();
        lost += GuestHolds(f, 0x20a20, cpu_value) ? 0 : 1;
    }
    REQUIRE(lost == 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out early downloaded bytes the guest wrote since",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    // The guest writes 16 of the landed bytes, as GPU::InvalidateGPUCache applies it, and a draw
    // uploads them. Then it writes the landed value back, which no upload has taken yet.
    const std::array<std::pair<DAddr, u64>, 1> written{{{0x20040, 0x10}}};
    const auto guest_write = [&](u8 value) {
        std::memset(f.bytes + 0x10040, value, 0x10);
        f.cache->ClaimCpuWrites(written);
        f.cache->WriteMemory(0x20040, 0x10);
    };
    guest_write(0xee);
    Access::Upload(*f.cache, id, 0x20000, 0x1000);
    REQUIRE(buffer.bytes[0x40] == 0xee);
    guest_write(3);
    // A synchronous flush at the read leaves them to the guest, as the other landed bytes are
    // what it would bring.
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).compared != 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out a guest write an earlier read's landing kept",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    // A read's download, then a read whose bytes it brings: that read's check copy comes alone,
    // in the next submission.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    REQUIRE(Access::ReadChecks(*f.cache).alone_recorded == 1);
    // The guest writes into those bytes. The first download lands at a fence and keeps the
    // write, collected and not yet applied; then the write is applied, before the second lands.
    writes.Write(f.bytes + 0x10040, 0x20040, NewGuestValue);
    f.runtime.Complete(f.runtime.submitted_tick - 1);
    f.cache->PopAsyncFlushes();
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    writes.Apply();
    f.cache->WriteMemory(0x20040, 0x40);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(GuestHolds(f, 0x20040, NewGuestValue));
    REQUIRE(f.bytes[0x1003f] == 0x12);
    REQUIRE(f.bytes[0x10080] == 0x12);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out early downloaded bytes a query result copy replaced",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    // A query result copied into landed bytes on the GPU (SyncValues), which the buffer cache
    // hears of only as outdating downloads. The query cache writes guest memory after this read,
    // and a synchronous flush at the read leaves guest memory as it is.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 8, u8{0x56}); });
    f.cache->OutdateReadDownloads(0x20040, 8);
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(buffer.bytes[0x40] == 0x56);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10048] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).compared != 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out early downloaded bytes an upload replaced",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    // The guest writes 16 of the landed bytes where the buffer cache doesn't hear of it, and a
    // draw uploads them. Then it writes the landed value back, which no upload has taken yet.
    std::memset(f.bytes + 0x10040, 0xee, 0x10);
    Access::MarkCpuModified(*f.cache, 0x20040, 0x10);
    Access::Upload(*f.cache, id, 0x20000, 0x1000);
    REQUIRE(buffer.bytes[0x40] == 0xee);
    std::memset(f.bytes + 0x10040, 3, 0x10);
    // A synchronous flush at the read leaves them to the guest: they are not GPU-modified.
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).compared != 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("EDEN_NX2_READ_CHECK values select how many read downloads are checked",
          "[buffer_ports]") {
    using VideoCore::AsyncReadback::ParseReadCheck;
    REQUIRE(ParseReadCheck(nullptr) == 0);
    REQUIRE(ParseReadCheck("") == 0);
    REQUIRE(ParseReadCheck("0") == 0);
    REQUIRE(ParseReadCheck("1") == 1);
    REQUIRE(ParseReadCheck("16") == 16);
}

TEST_CASE("EDEN_NX2_READ_CHECK=1 compares every read download", "[buffer_ports]") {
    Fixture f;
    const ReadCheckScope check{1};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    for (u32 read = 1; read <= 3; ++read) {
        Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
        f.cache->DownloadMemoryForRead(0x20000, 0x1000);
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        REQUIRE(Access::ReadChecks(*f.cache).checked == read);
    }
    REQUIRE(Access::ReadChecks(*f.cache).compared == 0x300);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

namespace {
/// Two read downloads, and a GPU write nothing records into the first one's bytes: the second
/// download's check finds 0x10 bytes outdated once both land.
VideoCommon::BufferId OutdatedReadDownloads(Fixture& f) {
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    return id;
}

/// The guest thread whose read waits for its downloads (GPU::OnCPURead).
struct ReadingThreadScope {
    ReadingThreadScope() {
        VideoCore::AsyncReadback::t_reading = true;
    }
    ~ReadingThreadScope() {
        VideoCore::AsyncReadback::t_reading = false;
    }
};

u64 MismatchedBy(Fixture& f, VideoCore::AsyncReadback::Landing landing) {
    return Access::ReadChecks(*f.cache).mismatched_by_landing[static_cast<size_t>(landing)];
}
} // namespace

TEST_CASE("The read check names who wrote a differing download to guest memory",
          "[buffer_ports]") {
    using VideoCore::AsyncReadback::Landing;
    Fixture f;
    const ReadCheckScope check{1};
    OutdatedReadDownloads(f);
    SECTION("the reading thread") {
        const ReadingThreadScope reading;
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        REQUIRE(MismatchedBy(f, Landing::Reader) == 0x10);
    }
    SECTION("a write to guest memory from the GPU's side") {
        f.cache->FinishReadDownloads(0x20800, 4);
        REQUIRE(MismatchedBy(f, Landing::GpuSideWrite) == 0x10);
    }
    SECTION("the next read download, once the host GPU ran them") {
        f.runtime.Complete(f.runtime.submitted_tick);
        f.cache->DownloadMemoryForRead(0x30000, 0x1000);
        REQUIRE(MismatchedBy(f, Landing::Completed) == 0x10);
    }
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0x10);
}

TEST_CASE("Async read-back counts the reading thread's and the GPU side's waits apart",
          "[buffer_ports]") {
    auto& counters = VideoCore::AsyncReadback::g_counters;
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    const u64 reader_waits = counters.reader_waits.load();
    const u64 gpu_side_waits = counters.gpu_side_waits.load();
    // The reading thread waits for a download the host GPU hasn't run.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    {
        const ReadingThreadScope reading;
        f.cache->FinishReadDownloads(0x20000, 0x1000);
    }
    REQUIRE(counters.reader_waits.load() - reader_waits == 1);
    REQUIRE(counters.gpu_side_waits.load() == gpu_side_waits);
    // A semaphore release over one falls back to a host GPU wait on the GPU's side.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->FinishReadDownloads(0x20040, 4);
    REQUIRE(counters.gpu_side_waits.load() - gpu_side_waits == 1);
    // Run by the host GPU before the read: nothing waits.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.runtime.Complete(f.runtime.submitted_tick);
    {
        const ReadingThreadScope reading;
        f.cache->FinishReadDownloads(0x20000, 0x1000);
    }
    REQUIRE(counters.reader_waits.load() - reader_waits == 1);
    REQUIRE(counters.gpu_side_waits.load() - gpu_side_waits == 1);
    REQUIRE(f.runtime.host_read_waits == 2);
}

TEST_CASE("Deleting a buffer clears its id from every channel's bindings", "[buffer_ports]") {
    struct ChannelsFixture : Fixture {
        ChannelsFixture() {
            for (auto* state : {&first, &second}) {
                state->memory_manager = &*memory;
                state->payload.emplace(system, *memory, *state);
                cache->CreateChannel(*state);
            }
        }
        ~ChannelsFixture() {
            cache.reset();
        }
        Tegra::Control::ChannelState first{0};
        Tegra::Control::ChannelState second{1};
    } f;
    namespace Dirty = VideoCommon::Dirty;
    const auto id = Access::Add(*f.cache, 0x10000, 0x10000);
    const VideoCommon::Binding bound{.device_addr = 0x10800, .size = 0x100, .buffer_id = id};
    auto& current = Access::Channel(*f.cache, 0);
    auto& other = Access::Channel(*f.cache, 1);
    current.uniform_buffers[0][0] = bound;
    other.index_buffer = bound;
    other.vertex_buffers[2] = bound;
    other.uniform_buffers[0][0] = bound;
    other.storage_buffers[4][1] = bound;
    other.transform_feedback_buffers[3] = bound;
    other.indirect_buffer_binding = bound;
    other.compute_uniform_buffers[5] = bound;
    other.enabled_uniform_buffer_masks[0] = 1;
    auto& flags = other.maxwell3d.dirty.flags;
    flags.reset();

    // Deleted while channel 0 is current; the next buffer takes its slot.
    f.cache->BindToChannel(0);
    Access::Delete(*f.cache, id);
    REQUIRE(Access::Add(*f.cache, 0x30000, 0x10000) == id);

    REQUIRE(!current.uniform_buffers[0][0].buffer_id);
    for (const auto* binding :
         {&other.index_buffer, &other.vertex_buffers[2], &other.uniform_buffers[0][0],
          &other.storage_buffers[4][1], &other.transform_feedback_buffers[3],
          &other.indirect_buffer_binding, &other.compute_uniform_buffers[5]}) {
        REQUIRE(!binding->buffer_id);
        REQUIRE(binding->device_addr == 0x10800);
    }
    REQUIRE(flags[Dirty::IndexBuffer]);
    REQUIRE(flags[Dirty::VertexBuffers]);
    REQUIRE(flags[Dirty::VertexBuffer0 + 2]);
    REQUIRE(!flags[Dirty::VertexBuffer0 + 1]);

    // Channel 1's next draw resolves its uniform buffer again, to a buffer that holds the range.
    f.cache->BindToChannel(1);
    Access::UpdateUniforms(*f.cache, 0);
    const auto resolved = other.uniform_buffers[0][0].buffer_id;
    REQUIRE(resolved);
    REQUIRE(resolved != VideoCommon::NULL_BUFFER_ID);
    REQUIRE(resolved != id);
    REQUIRE(Access::Get(*f.cache, resolved).IsInBounds(0x10800, 0x100));
}

TEST_CASE("Claimed CPU writes keep their bytes when an earlier one downloads their page",
          "[buffer_ports]") {
    Fixture f;
    // The download branch of WriteMemory: only the scheduler's owner downloads, and Switch 2
    // titles upload inline instead.
    const bool ampere = Tegra::g_nx2_ampere.exchange(false);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    std::memset(f.bytes + 0x10100, 0xaa, 0x10);
    std::memset(f.bytes + 0x10800, 0xbb, 0x10);
    const std::array<std::pair<DAddr, u64>, 2> ranges{{{0x20100, 0x10}, {0x20800, 0x10}}};
    // A thread of its own, so marking it the GPU thread doesn't carry over to other tests.
    std::thread{[&] {
        VideoCommon::GpuSplit::MarkGpuThread();
        f.cache->ClaimCpuWrites(ranges);
        for (const auto& [address, size] : ranges) {
            f.cache->WriteMemory(address, size);
        }
    }}.join();
    Tegra::g_nx2_ampere.store(ampere);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10100] == 0xaa);
    REQUIRE(f.bytes[0x10800] == 0xbb);
    REQUIRE(f.bytes[0x1080f] == 0xbb);
    REQUIRE(f.bytes[0x10810] == 0x12);
}

namespace {
/// Command list page 0x18000 (GPU 0x108000) of KeepFixture's buffer, read from its copies as the
/// split's front reads them (DmaPusher::ReadGpuWrittenCommands).
struct ListPageFixture : KeepFixture {
    static constexpr DAddr Page = 0x18000;
    static constexpr GPUVAddr PageGpu = 0x108000;

    ListPageFixture() : KeepFixture{nullptr} {
        Draw(nullptr); // Binds the channel DMACopy and DMAClear use.
        Tegra::Nx2MarkCommandListPage(Page);
    }
    /// Collects the page writes of a dispatch or DMA operation and has the host GPU make the
    /// copies taken after it, as the rasterizer does.
    template <typename Func>
    void Work(Func&& func) {
        cache->Nx2CollectListCopies(true);
        func();
        if (cache->Nx2CopyListPages(runtime.CurrentTick())) {
            runtime.Complete(runtime.SubmitForHostRead());
        }
        cache->Nx2CollectListCopies(false);
    }
    /// A dispatch that writes `value` over [address, address + size).
    void Dispatch(DAddr address, u32 size, u8 value) {
        Work([&] {
            auto& buffer = Access::Get(*cache, id);
            runtime.pending.push_back([&buffer, address, size, value] {
                std::fill_n(buffer.bytes.begin() + static_cast<std::ptrdiff_t>(address - 0x10000),
                            size, value);
            });
            Access::GpuWrite(*cache, id, address, size);
        });
    }
    /// Whether the front reads [address, address + 0x20) from a copy that holds `value`.
    bool CopyHolds(DAddr address, u8 value) {
        std::array<u8, 0x20> out{};
        if (cache->Nx2ReadListCopy(address, out, true) != KeepCache::Nx2ListRead::Copied) {
            return false;
        }
        return std::ranges::all_of(out, [value](u8 byte) { return byte == value; });
    }

    const Nx2TitleScope nx2_title;
};
} // namespace

TEST_CASE("A command list page is copied again after a DMA copy of data no GPU work wrote",
          "[buffer_ports]") {
    ListPageFixture f;
    f.Dispatch(ListPageFixture::Page, 0x20, 3);
    REQUIRE(f.CopyHolds(ListPageFixture::Page, 3));

    // The source holds only what the CPU wrote, so none of the copied bytes are GPU-modified.
    std::memset(f.bytes + 0x4000, 7, 0x20);
    f.cache->WriteMemory(0x14000, 0x20);
    f.Work([&] { REQUIRE(f.cache->DMACopy(0x104000, ListPageFixture::PageGpu, 0x20)); });
    REQUIRE(f.bytes[0x8000] == 7);
    REQUIRE(f.CopyHolds(ListPageFixture::Page, 7));
}

TEST_CASE("A command list page is copied again after a DMA clear", "[buffer_ports]") {
    ListPageFixture f;
    f.Dispatch(ListPageFixture::Page, 0x20, 3);
    REQUIRE(f.CopyHolds(ListPageFixture::Page, 3));

    // MaxwellDMA writes guest memory once the cache cleared the buffer.
    f.Work([&] { REQUIRE(f.cache->DMAClear(ListPageFixture::PageGpu, 8, 0x09090909)); });
    std::memset(f.bytes + 0x8000, 9, 0x20);
    REQUIRE(f.CopyHolds(ListPageFixture::Page, 9));
}

TEST_CASE("A command list page's copy holds the CPU's bytes the buffer had not taken yet",
          "[buffer_ports]") {
    ListPageFixture f;
    f.Dispatch(ListPageFixture::Page, 0x20, 3);
    f.cache->DownloadMemory(ListPageFixture::Page, 0x1000);
    REQUIRE(f.bytes[0x8000] == 3);

    // Later CPU writes to the page reach no cache: only the first is reported.
    std::memset(f.bytes + 0x8100, 5, 0x20);
    f.cache->WriteMemory(ListPageFixture::Page + 0x100, 0x20);
    REQUIRE_FALSE(f.CopyHolds(ListPageFixture::Page, 3));

    f.Work([&] { REQUIRE(f.cache->DMAClear(ListPageFixture::PageGpu, 8, 0x09090909)); });
    std::memset(f.bytes + 0x8000, 9, 0x20);
    REQUIRE(f.CopyHolds(ListPageFixture::Page, 9));
    REQUIRE(f.CopyHolds(ListPageFixture::Page + 0x100, 5));
}
