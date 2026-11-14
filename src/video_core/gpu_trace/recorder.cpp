// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_trace/recorder.h"
#include "video_core/gpu_trace/state.h"
#include "video_core/gpu_trace/trace_file.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GpuTrace {
namespace {
using Clock = std::chrono::steady_clock;

struct PendingPresent {
    std::vector<PresentLayer> layers;
    std::vector<PresentFence> fences;
};

struct Recorder {
    /// Guards everything below but the dirty pages, and the writer.
    std::mutex mutex;
    RecordOptions options;
    bool stop = false;
    Core::System* system = nullptr;
    std::unique_ptr<Writer> writer;
    Clock::time_point start;
    u32 presents = 0;
    std::vector<PendingPresent> pending_presents;
    std::unordered_set<size_t> address_spaces;
    std::unordered_set<s32> channels;
    /// Device pages whose SMMU mapping and content are in the trace.
    std::unordered_set<u64> device_pages;
    /// Physical pages whose content is in the trace.
    std::vector<bool> physical_pages;
    /// Device ranges the recording holds a cached-page reference on, so that every CPU write to
    /// them reaches NoteCpuWrite.
    std::vector<std::pair<DAddr, u64>> cached;
    std::string last_trace;
    std::string last_error;
    Clock::time_point last_file_check;

    std::mutex dirty_mutex;
    const u8* backing = nullptr;
    size_t backing_size = 0;
    /// Physical pages in the trace: only their CPU writes are recorded.
    std::vector<u8> tracked;
    /// Per physical page, the written byte range [low, high) as low | high << 16; 0 for none.
    std::vector<u32> dirty;
    std::vector<u64> dirty_pages;
};

u32 MergeExtent(u32 extent, u32 low, u32 high) {
    if (extent == 0) {
        return low | (high << 16);
    }
    return std::min(extent & 0xffff, low) | (std::max(extent >> 16, high) << 16);
}

Recorder& Get() {
    static Recorder recorder;
    return recorder;
}

std::span<const u8> Bytes(const auto& value) {
    return {reinterpret_cast<const u8*>(&value), sizeof(value)};
}

template <typename T>
std::span<const u8> ArrayBytes(std::span<const T> values) {
    return {reinterpret_cast<const u8*>(values.data()), values.size_bytes()};
}

PresentLayer ToLayer(const Tegra::FramebufferConfig& config) {
    PresentLayer layer{};
    layer.address = config.address;
    layer.offset = config.offset;
    layer.width = config.width;
    layer.height = config.height;
    layer.stride = config.stride;
    layer.pixel_format = static_cast<u32>(config.pixel_format);
    layer.transform_flags = static_cast<u32>(config.transform_flags);
    layer.crop_left = config.crop_rect.left;
    layer.crop_top = config.crop_rect.top;
    layer.crop_right = config.crop_rect.right;
    layer.crop_bottom = config.crop_rect.bottom;
    layer.blending = static_cast<u32>(config.blending);
    layer.native_tiling = config.native_tiling ? 1 : 0;
    layer.layer_stack_mask = config.layer_stack_mask;
    layer.color_space = static_cast<u32>(config.color_space);
    layer.linear_unit_luminance = config.linear_unit_luminance;
    return layer;
}

bool IsZero(std::span<const u8> bytes) {
    return std::ranges::all_of(bytes, [](u8 byte) { return byte == 0; });
}

/// The SMMU runs behind [device, device + size) not in the trace yet.
std::vector<SmmuMapRecord> NewDeviceRuns(Recorder& r, DAddr device, u64 size) {
    auto& smmu = r.system->Host1x().MemoryManager();
    std::vector<SmmuMapRecord> runs;
    const u64 end = device + size;
    for (u64 page = device & ~(PageSize - 1); page < end; page += PageSize) {
        if (!r.device_pages.insert(page >> PageBits).second) {
            continue;
        }
        const u64 physical = smmu.GetPhysicalRawAddressFromDAddr(page);
        if (physical == 0) {
            continue;
        }
        if (!runs.empty() && runs.back().device_address + runs.back().size == page &&
            runs.back().physical_address + runs.back().size == physical) {
            runs.back().size += PageSize;
        } else {
            runs.push_back({page, physical, PageSize});
        }
    }
    return runs;
}

/// Holds cached-page references on the runs, then writes their maps and memory. A CPU write
/// that lands during the copy is recorded again in the next delta.
void CaptureRuns(Recorder& r, std::span<const SmmuMapRecord> runs) {
    auto& smmu = r.system->Host1x().MemoryManager();
    for (const SmmuMapRecord& run : runs) {
        smmu.UpdatePagesCachedCount(run.device_address, run.size, 1);
        r.cached.emplace_back(run.device_address, run.size);
    }
    for (const SmmuMapRecord& run : runs) {
        r.writer->WritePod(RecordType::SmmuMap, run);
        for (u64 offset = 0; offset < run.size; offset += PageSize) {
            const u64 physical = run.physical_address + offset;
            const u64 page = physical >> PageBits;
            if (physical + PageSize > r.backing_size || r.physical_pages[page]) {
                continue;
            }
            r.physical_pages[page] = true;
            {
                std::scoped_lock lock{r.dirty_mutex};
                r.tracked[page] = 1;
            }
            const std::span<const u8> bytes{r.backing + physical, PageSize};
            if (!IsZero(bytes)) {
                r.writer->WriteMemory(physical, bytes);
            }
        }
    }
}

void WriteGmmuMap(Recorder& r, const Tegra::MemoryManager& manager, const MapLog::Entry& entry) {
    r.writer->WritePod(RecordType::GmmuMap,
                       GmmuMapRecord{static_cast<u32>(manager.GetID()), entry.kind,
                                     entry.gpu_address, entry.device_address,
                                     entry.end - entry.gpu_address,
                                     static_cast<u8>(entry.big_pages ? 1 : 0),
                                     static_cast<u8>(entry.sparse ? 1 : 0)});
}

/// The CPU writes since the last command, as they are now: only the bytes written, so GPU
/// results elsewhere on the page stay the replay's own.
void FlushDirty(Recorder& r) {
    std::vector<std::pair<u64, u32>> ranges;
    {
        std::scoped_lock lock{r.dirty_mutex};
        ranges.reserve(r.dirty_pages.size());
        for (const u64 page : r.dirty_pages) {
            ranges.emplace_back(page, r.dirty[page]);
            r.dirty[page] = 0;
        }
        r.dirty_pages.clear();
    }
    std::ranges::sort(ranges);
    for (const auto& [page, extent] : ranges) {
        const u64 physical = (page << PageBits) + (extent & 0xffff);
        r.writer->WriteMemory(physical, {r.backing + physical, (extent >> 16) - (extent & 0xffff)});
    }
}

void WriteSubmit(const Tegra::CommandList& list, s32 channel, Recorder& r) {
    if (list.copy_job != 0) {
        Tegra::DmaPusher::WaitForSnapshotCopies();
    }
    std::vector<u64> entries;
    entries.reserve(list.command_lists.size());
    for (const auto& entry : list.command_lists) {
        entries.push_back(entry.raw);
    }
    std::vector<u32> prefetch;
    prefetch.reserve(list.prefetch_command_list.size());
    for (const auto& word : list.prefetch_command_list) {
        prefetch.push_back(word.argument);
    }
    const std::span<const u32> words{reinterpret_cast<const u32*>(list.submitted_words.data()),
                                     list.submitted_words.size()};
    const SubmitRecord header{channel,
                              static_cast<u32>(entries.size()),
                              static_cast<u32>(prefetch.size()),
                              static_cast<u32>(list.submitted_offsets.size()),
                              words.size(),
                              static_cast<u32>(list.deferred_entries.size())};
    const std::array<std::span<const u8>, 6> parts{
        Bytes(header),
        ArrayBytes(std::span<const u64>{entries}),
        ArrayBytes(std::span<const u32>{prefetch}),
        ArrayBytes(std::span<const u32>{list.submitted_offsets}),
        ArrayBytes(words),
        ArrayBytes(std::span<const u32>{list.deferred_entries})};
    r.writer->Write(RecordType::Submit, parts);
}

void WritePresents(Recorder& r) {
    for (const PendingPresent& present : r.pending_presents) {
        const PresentRecord header{
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                                  r.start)
                                 .count()),
            static_cast<u32>(present.layers.size()), static_cast<u32>(present.fences.size())};
        const std::array<std::span<const u8>, 3> parts{
            Bytes(header), ArrayBytes(std::span<const PresentLayer>{present.layers}),
            ArrayBytes(std::span<const PresentFence>{present.fences})};
        r.writer->Write(RecordType::Present, parts);
        ++r.presents;
    }
    r.pending_presents.clear();
}

void Release(Recorder& r) {
    g_recording = false;
    if (r.system != nullptr) {
        auto& smmu = r.system->Host1x().MemoryManager();
        for (const auto& [address, size] : r.cached) {
            smmu.UpdatePagesCachedCount(address, size, -1);
        }
    }
    r.cached.clear();
    {
        std::scoped_lock lock{r.dirty_mutex};
        r.tracked = {};
        r.dirty = {};
        r.dirty_pages = {};
        r.backing = nullptr;
        r.backing_size = 0;
    }
    r.writer.reset();
    r.pending_presents.clear();
    r.address_spaces.clear();
    r.channels.clear();
    r.device_pages.clear();
    r.physical_pages = {};
    r.presents = 0;
    r.stop = false;
    g_requested = false;
}

void Finish(Recorder& r) {
    const u64 bytes = r.writer->BytesIn();
    if (r.writer->Finish()) {
        r.last_trace = r.options.path.string();
        LOG_INFO(HW_GPU, "GPU trace: wrote {} ({} presents, {} MiB before compression)",
                 r.last_trace, r.presents, bytes >> 20);
    } else {
        r.last_error = "writing " + r.options.path.string() + " failed";
        LOG_ERROR(HW_GPU, "GPU trace: {}", r.last_error);
    }
    Release(r);
}

/// On the GPU thread at a command boundary with no suspended channel.
bool Snapshot(Recorder& r, Core::System& system) {
    if (Settings::IsNceEnabled()) {
        r.last_error = "GPU traces need the JIT CPU backend: NCE notes CPU writes before their store";
        return false;
    }
    auto& gpu = system.GPU();
    const auto channels = gpu.Channels();
    auto& split = GpuSplit::Get();
    if (split.Active()) {
        split.Drain();
    }
    for (const auto& channel : channels) {
        if (channel->initialized) {
            StateAccess::Settle(*channel);
        }
    }
    r.writer = std::make_unique<Writer>(r.options.path);
    if (!r.writer->IsOpen()) {
        r.last_error = "can't create " + r.options.path.string();
        return false;
    }
    r.system = &system;
    r.start = Clock::now();

    std::vector<Tegra::MemoryManager*> managers;
    u64 program_id = 0;
    for (const auto& channel : channels) {
        if (channel->initialized && channel->memory_manager != nullptr &&
            r.address_spaces.insert(channel->memory_manager->GetID()).second) {
            managers.push_back(channel->memory_manager);
        }
        if (channel->initialized) {
            program_id = channel->program_id;
        }
    }
    const nlohmann::json config{
        {"build", std::string{Common::g_build_fullname}},
        {"program_id", fmt::format("{:016X}", program_id)},
        {"switch2", Tegra::g_nx2_ampere.load()},
        {"memory_layout", static_cast<u32>(Settings::values.memory_layout_mode.GetValue())},
        {"width", 1920},
        {"height", 1080},
        {"presents", r.options.presents},
    };
    const std::string text = config.dump();
    r.writer->Write(RecordType::Config, {reinterpret_cast<const u8*>(text.data()), text.size()});

    std::vector<SmmuMapRecord> runs;
    for (Tegra::MemoryManager* manager : managers) {
        r.writer->WritePod(RecordType::AddressSpace, StateAccess::AddressSpace(*manager));
        for (const MapLog::Entry& entry : StateAccess::Maps(*manager)) {
            WriteGmmuMap(r, *manager, entry);
            if (!entry.sparse) {
                auto more = NewDeviceRuns(r, entry.device_address, entry.end - entry.gpu_address);
                runs.insert(runs.end(), more.begin(), more.end());
            }
        }
    }
    // GPU-written data the caches hold goes to guest memory first, so the copy below has it.
    split.Run([&] {
        auto* const rasterizer = gpu.Renderer().ReadRasterizer();
        for (const SmmuMapRecord& run : runs) {
            rasterizer->FlushRegion(run.device_address, run.size);
        }
    });
    {
        std::scoped_lock lock{r.dirty_mutex};
        auto& memory = system.DeviceMemory().buffer;
        r.backing = memory.BackingBasePointer();
        r.backing_size = memory.BackingSize();
        r.tracked.assign(r.backing_size >> PageBits, 0);
        r.dirty.assign(r.backing_size >> PageBits, 0);
    }
    r.physical_pages.assign(r.backing_size >> PageBits, false);
    g_recording = true;
    CaptureRuns(r, runs);

    Tegra::Control::ChannelState* any_channel = nullptr;
    for (const auto& channel : channels) {
        if (!channel->initialized || channel->memory_manager == nullptr) {
            continue;
        }
        r.writer->WritePod(RecordType::Channel,
                           ChannelRecord{channel->bind_id,
                                         static_cast<u32>(channel->memory_manager->GetID()),
                                         channel->program_id});
        r.channels.insert(channel->bind_id);
        StateAccess::CaptureChannel(*r.writer, *channel);
        any_channel = channel.get();
    }
    StateAccess::CaptureGlobals(*r.writer, any_channel, system.Host1x().GetSyncpointManager());
    r.writer->Write(RecordType::SnapshotEnd, std::span<const u8>{});
    LOG_INFO(HW_GPU, "GPU trace: recording {} presents to {}", r.options.presents,
             r.options.path.string());
    return true;
}

void CheckFileTrigger(Recorder& r) {
    const auto now = Clock::now();
    if (now - r.last_file_check < std::chrono::seconds{1}) {
        return;
    }
    r.last_file_check = now;
    const std::filesystem::path directory{std::getenv("EDEN_NX2_GPU_TRACE")};
    std::error_code error;
    if (!std::filesystem::remove(directory / "record.now", error)) {
        return;
    }
    RecordOptions options;
    options.path = directory / fmt::format("trace-{}.lxtrace",
                                           std::chrono::duration_cast<std::chrono::seconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count());
    if (const char* presents = std::getenv("EDEN_NX2_GPU_TRACE_PRESENTS")) {
        options.presents = std::max(1u, static_cast<u32>(std::strtoul(presents, nullptr, 10)));
    }
    r.options = options;
    r.stop = false;
    r.last_error.clear();
    g_requested = true;
}
} // Anonymous namespace

bool RequestRecording(const RecordOptions& options, std::string& error) {
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (g_requested) {
        error = "a GPU trace is already being recorded";
        return false;
    }
    if (options.path.empty() || options.presents == 0) {
        error = "a GPU trace needs a path and at least one present";
        return false;
    }
    r.options = options;
    r.stop = false;
    r.last_error.clear();
    g_requested = true;
    return true;
}

void StopRecording() {
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (g_requested) {
        r.stop = true;
    }
}

RecorderStatus GetRecorderStatus() {
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    return {
        .requested = g_requested,
        .recording = g_recording,
        .presents = r.presents,
        .presents_wanted = r.options.presents,
        .bytes = r.writer ? r.writer->BytesIn() : 0,
        .path = r.options.path.string(),
        .last_trace = r.last_trace,
        .last_error = r.last_error,
    };
}

void NoteCpuWrite(const u8* pointer, size_t size) {
    if (!g_recording.load(std::memory_order_relaxed) || size == 0) {
        return;
    }
    auto& r = Get();
    std::scoped_lock lock{r.dirty_mutex};
    if (pointer < r.backing || pointer >= r.backing + r.backing_size) {
        return;
    }
    const u64 begin = static_cast<u64>(pointer - r.backing);
    const u64 end = std::min<u64>(begin + size, r.backing_size);
    for (u64 page = begin >> PageBits; (page << PageBits) < end; ++page) {
        if (r.tracked[page] == 0) {
            continue;
        }
        const u32 low = static_cast<u32>(std::max(begin, page << PageBits) - (page << PageBits));
        const u32 high = static_cast<u32>(std::min(end, (page + 1) << PageBits) - (page << PageBits));
        if (r.dirty[page] == 0) {
            r.dirty_pages.push_back(page);
        }
        r.dirty[page] = MergeExtent(r.dirty[page], low, high);
    }
}

void NoteDeviceWrite(DAddr address, u64 size) {
    if (!g_recording.load(std::memory_order_relaxed) || size == 0) {
        return;
    }
    auto& r = Get();
    Core::System* const system = r.system;
    if (system == nullptr) {
        return;
    }
    auto& smmu = system->Host1x().MemoryManager();
    const u64 end = address + size;
    for (u64 page = address & ~(PageSize - 1); page < end; page += PageSize) {
        const u64 physical = smmu.GetPhysicalRawAddressFromDAddr(page);
        if (physical == 0) {
            continue;
        }
        const u64 begin = std::max<u64>(address, page);
        const u64 stop = std::min(end, page + PageSize);
        NoteCpuWrite(r.backing + physical + (begin - page), stop - begin);
    }
}

void OnGpuShutdown() {
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (r.writer) {
        LOG_INFO(HW_GPU, "GPU trace: the GPU stopped, ending the recording");
        Finish(r);
    } else if (g_requested) {
        Release(r);
    }
}

void NoteGmmuMap(const Tegra::MemoryManager& memory_manager, const MapLog::Entry& entry) {
    if (!g_recording.load(std::memory_order_relaxed)) {
        return;
    }
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (!r.writer || !r.address_spaces.contains(memory_manager.GetID())) {
        return;
    }
    if (!entry.sparse) {
        const auto runs = NewDeviceRuns(r, entry.device_address, entry.end - entry.gpu_address);
        CaptureRuns(r, runs);
    }
    WriteGmmuMap(r, memory_manager, entry);
}

void NoteGmmuUnmap(const Tegra::MemoryManager& memory_manager, GPUVAddr gpu_address, u64 size) {
    if (!g_recording.load(std::memory_order_relaxed)) {
        return;
    }
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (!r.writer || !r.address_spaces.contains(memory_manager.GetID())) {
        return;
    }
    r.writer->WritePod(RecordType::GmmuUnmap,
                       GmmuUnmapRecord{static_cast<u32>(memory_manager.GetID()), 0, gpu_address,
                                       size});
}

void NotePresent(std::span<const Tegra::FramebufferConfig> layers,
                 std::span<const Service::Nvidia::NvFence> fences) {
    if (!g_recording.load(std::memory_order_relaxed)) {
        return;
    }
    PendingPresent present;
    for (const auto& layer : layers) {
        present.layers.push_back(ToLayer(layer));
    }
    for (const auto& fence : fences) {
        present.fences.push_back({fence.id, fence.value});
    }
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (r.writer) {
        r.pending_presents.push_back(std::move(present));
    }
}

void OnCommand(Core::System& system, bool channels_pending, const Tegra::CommandList* submit,
               s32 channel, bool tick) {
    auto& r = Get();
    std::scoped_lock lock{r.mutex};
    if (!r.writer) {
        if (g_file_trigger && tick && !g_requested) {
            CheckFileTrigger(r);
        }
        if (!g_requested) {
            return;
        }
        if (r.stop) {
            Release(r);
            return;
        }
        if (channels_pending) {
            return;
        }
        if (!Snapshot(r, system)) {
            LOG_ERROR(HW_GPU, "GPU trace: {}", r.last_error);
            Release(r);
            return;
        }
    }
    if (submit != nullptr && !r.channels.contains(channel)) {
        LOG_WARNING(HW_GPU, "GPU trace: channel {} began after the snapshot, ending the recording",
                    channel);
        WritePresents(r);
        Finish(r);
        return;
    }
    FlushDirty(r);
    if (submit != nullptr) {
        WriteSubmit(*submit, channel, r);
    }
    if (tick) {
        WritePresents(r);
    }
    if (r.stop || r.presents >= r.options.presents || !r.writer->IsOpen()) {
        Finish(r);
    }
}

} // namespace VideoCommon::GpuTrace
