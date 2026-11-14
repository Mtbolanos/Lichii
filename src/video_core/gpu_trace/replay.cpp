// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <thread>
#include <memory>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging.h"
#include "common/settings.h"
#include "common/stb.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/nvdrv/devices/nx2_stub_device.h"
#include "video_core/control/channel_state.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/gpu_trace/recorder.h"
#include "video_core/gpu_trace/replay.h"
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

class HeadlessContext final : public Core::Frontend::GraphicsContext {};

class HeadlessWindow final : public Core::Frontend::EmuWindow {
public:
    HeadlessWindow(u32 width, u32 height) {
        window_info.type = Core::Frontend::WindowSystemType::Headless;
        UpdateCurrentFramebufferLayout(width, height);
    }

    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<HeadlessContext>();
    }

    bool IsShown() const override {
        return true;
    }
};

struct HashKey {
    size_t operator()(const Hash& hash) const noexcept {
        return static_cast<size_t>(hash[0] ^ (hash[1] * 0x9E3779B97F4A7C15ULL));
    }
};

std::string HashString(const Hash& hash) {
    return fmt::format("{:016x}{:016x}", hash[1], hash[0]);
}

template <typename T>
std::span<const T> Array(const Record& record, size_t& offset, size_t count) {
    if (record.payload.size() < offset + count * sizeof(T)) {
        return {};
    }
    const std::span<const T> result{reinterpret_cast<const T*>(record.payload.data() + offset),
                                    count};
    offset += count * sizeof(T);
    return result;
}

Tegra::FramebufferConfig ToConfig(const PresentLayer& layer) {
    Tegra::FramebufferConfig config{};
    config.address = layer.address;
    config.offset = layer.offset;
    config.width = layer.width;
    config.height = layer.height;
    config.stride = layer.stride;
    config.pixel_format = static_cast<Service::android::PixelFormat>(layer.pixel_format);
    config.transform_flags =
        static_cast<Service::android::BufferTransformFlags>(layer.transform_flags);
    config.crop_rect = {layer.crop_left, layer.crop_top, layer.crop_right, layer.crop_bottom};
    config.blending = static_cast<Tegra::BlendMode>(layer.blending);
    config.native_tiling = layer.native_tiling != 0;
    config.layer_stack_mask = layer.layer_stack_mask;
    config.color_space = static_cast<Tegra::ColorSpace>(layer.color_space);
    config.linear_unit_luminance = static_cast<u16>(layer.linear_unit_luminance);
    return config;
}

/// One replay: the trace's address spaces, channels and memory, brought up on a fresh System.
class Replayer {
public:
    Replayer(Core::System& system_, const ReplayOptions& options_)
        : system{system_}, options{options_}, reader{options_.trace} {}

    ~Replayer() {
        Shutdown();
    }

    bool Run(std::string& error);

private:
    bool ReadSnapshot(std::string& error);
    bool BringUp(std::string& error);
    bool Stream(std::string& error);
    void Shutdown();

    bool ApplyBlob(const Record& record, std::string& error);
    const std::vector<u8>* FindBlob(const Hash& hash) const;
    void WriteMemory(const MemoryRecord& record, const std::vector<u8>& bytes);
    void InvalidateDevice(u64 physical, u64 size);
    bool Submit(const Record& record, std::string& error);
    bool Present(const Record& record, std::string& error);
    Tegra::MemoryManager* AddressSpace(u32 id);
    void MapGmmu(const GmmuMapRecord& map);
    void MapSmmu(const SmmuMapRecord& map);

    Core::System& system;
    const ReplayOptions& options;
    Reader reader;

    nlohmann::json config;
    u64 program_id = 0;
    std::vector<AddressSpaceRecord> spaces;
    std::vector<GmmuMapRecord> gmmu_maps;
    std::vector<SmmuMapRecord> smmu_maps;
    std::vector<ChannelRecord> channel_records;
    std::vector<Record> states;
    std::vector<MemoryRecord> start_memory;
    std::unordered_map<Hash, std::shared_ptr<const std::vector<u8>>, HashKey> blobs;

    std::unique_ptr<HeadlessWindow> window;
    bool powered = false;
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    std::unordered_map<u32, std::shared_ptr<Tegra::MemoryManager>> memory_managers;
    std::unordered_map<s32, std::shared_ptr<Tegra::Control::ChannelState>> channels;
    /// Physical page -> the device pages that show it.
    std::unordered_map<u64, std::vector<u64>> device_pages;

    nlohmann::json frames = nlohmann::json::array();
    Clock::time_point last_frame;
    u32 presents = 0;
    bool stop = false;
};

bool Replayer::Run(std::string& error) {
    if (!reader.Error().empty()) {
        error = reader.Error();
        return false;
    }
    if (!ReadSnapshot(error) || !BringUp(error) || !Stream(error)) {
        return false;
    }
    nlohmann::json summary;
    summary["trace"] = options.trace.filename().string();
    summary["program_id"] = fmt::format("{:016X}", program_id);
    summary["renderer"] = system.GPU().Renderer().GetDeviceVendor();
    summary["frames"] = frames;
    std::ofstream file{options.out / "frames.json"};
    file << summary.dump(2) << '\n';
    if (!file) {
        error = "can't write frames.json";
        return false;
    }
    return true;
}

bool Replayer::ApplyBlob(const Record& record, std::string& error) {
    if (!HasPayload(record, sizeof(BlobRecord))) {
        error = "short blob record";
        return false;
    }
    const Hash hash = record.As<BlobRecord>().hash;
    blobs.try_emplace(hash, std::make_shared<const std::vector<u8>>(
                                record.payload.begin() + sizeof(BlobRecord), record.payload.end()));
    return true;
}

const std::vector<u8>* Replayer::FindBlob(const Hash& hash) const {
    const auto it = blobs.find(hash);
    return it != blobs.end() ? it->second.get() : nullptr;
}

bool Replayer::ReadSnapshot(std::string& error) {
    while (auto record = reader.Next()) {
        switch (record->type) {
        case RecordType::Config:
            config = nlohmann::json::parse(record->payload.begin(), record->payload.end(), nullptr,
                                           false);
            if (config.is_discarded()) {
                error = "the trace's config is not JSON";
                return false;
            }
            program_id = std::stoull(config.value("program_id", std::string{"0"}), nullptr, 16);
            break;
        case RecordType::AddressSpace:
            if (!HasPayload(*record, sizeof(AddressSpaceRecord))) {
                error = "short address space record";
                return false;
            }
            spaces.push_back(record->As<AddressSpaceRecord>());
            break;
        case RecordType::GmmuMap:
            if (!HasPayload(*record, sizeof(GmmuMapRecord))) {
                error = "short GMMU map record";
                return false;
            }
            gmmu_maps.push_back(record->As<GmmuMapRecord>());
            break;
        case RecordType::SmmuMap:
            if (!HasPayload(*record, sizeof(SmmuMapRecord))) {
                error = "short SMMU map record";
                return false;
            }
            smmu_maps.push_back(record->As<SmmuMapRecord>());
            break;
        case RecordType::Channel:
            if (!HasPayload(*record, sizeof(ChannelRecord))) {
                error = "short channel record";
                return false;
            }
            channel_records.push_back(record->As<ChannelRecord>());
            break;
        case RecordType::State:
            states.push_back(std::move(*record));
            break;
        case RecordType::Blob:
            if (!ApplyBlob(*record, error)) {
                return false;
            }
            break;
        case RecordType::Memory:
            if (!HasPayload(*record, sizeof(MemoryRecord))) {
                error = "short memory record";
                return false;
            }
            start_memory.push_back(record->As<MemoryRecord>());
            break;
        case RecordType::SnapshotEnd:
            return true;
        default:
            error = fmt::format("record type {} before the snapshot's end",
                                static_cast<u32>(record->type));
            return false;
        }
    }
    error = reader.Error().empty() ? "the trace has no snapshot end" : reader.Error();
    return false;
}

bool Replayer::BringUp(std::string& error) {
    if (config.is_null()) {
        error = "the trace has no config record";
        return false;
    }
    const bool switch2 = config.value("switch2", true);
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
    Settings::values.use_asynchronous_gpu_emulation.SetValue(true);
    Settings::values.use_multi_core.SetValue(false);
    Settings::values.memory_layout_mode.SetValue(
        static_cast<Settings::MemoryLayout>(config.value("memory_layout", 0)));
    Settings::values.resolution_setup.SetValue(Settings::ResolutionSetup::Res1X);
    Settings::SetCurrentProgramID(program_id);
    Settings::SetCurrentProgramIsSwitch2(switch2);

    Service::Nvidia::Devices::Nx2GpuMemoryRegistry::Get().Reset();
    Tegra::Nx2ClearSemaphoreRegions();
    Tegra::Nx2ClearSparseRegions();
    Tegra::Nx2ClearGpuWrittenPages();
    Tegra::g_nx2_ampere = switch2;
    Service::Nvidia::Devices::g_nx2_driver = switch2;

    system.Initialize();
    window = std::make_unique<HeadlessWindow>(config.value("width", 1920u),
                                              config.value("height", 1080u));
    if (system.InitializeForGpuReplay(*window, program_id) != Core::SystemResultStatus::Success) {
        error = "the GPU could not start (is a Vulkan device available?)";
        return false;
    }
    powered = true;

    auto& backing = system.DeviceMemory().buffer;
    for (const MemoryRecord& memory : start_memory) {
        const auto* bytes = FindBlob(memory.hash);
        if (bytes == nullptr || bytes->size() != memory.size) {
            error = "a memory record names a missing blob";
            return false;
        }
        if (memory.physical_address + memory.size > backing.BackingSize()) {
            error = fmt::format("memory at {:#x} is outside guest memory", memory.physical_address);
            return false;
        }
        std::memcpy(backing.BackingBasePointer() + memory.physical_address, bytes->data(),
                    bytes->size());
    }
    start_memory.clear();

    // The SMMU translates through a process page table: one whose addresses are physical offsets.
    process = std::make_unique<Kernel::KProcess>(system.Kernel());
    auto& table = process->GetPageTable().GetImpl();
    table.Resize(36, PageBits);
    process->GetMemory().SetCurrentPageTable(*process);
    auto& smmu = system.Host1x().MemoryManager();
    asid = smmu.RegisterProcess(&process->GetMemory());
    for (const SmmuMapRecord& map : smmu_maps) {
        MapSmmu(map);
    }

    for (const AddressSpaceRecord& space : spaces) {
        auto manager = std::make_shared<Tegra::MemoryManager>(
            system, smmu, space.address_space_bits, space.split_address, space.big_page_bits,
            space.page_bits);
        system.GPU().InitAddressSpace(*manager);
        memory_managers.emplace(space.id, std::move(manager));
    }
    for (const GmmuMapRecord& map : gmmu_maps) {
        MapGmmu(map);
    }

    Tegra::Control::ChannelState* any_channel = nullptr;
    for (const ChannelRecord& record : channel_records) {
        auto* const manager = AddressSpace(record.address_space);
        if (manager == nullptr) {
            error = fmt::format("channel {} uses unknown address space {}", record.bind_id,
                                record.address_space);
            return false;
        }
        auto channel = system.GPU().AllocateChannel();
        channel->memory_manager = manager;
        system.GPU().InitChannel(*channel, program_id);
        any_channel = channel.get();
        channels.emplace(record.bind_id, std::move(channel));
    }
    auto& syncpoints = system.Host1x().GetSyncpointManager();
    for (const Record& state : states) {
        const auto& header = state.As<StateRecord>();
        const auto it = channels.find(header.channel);
        auto* const channel = it != channels.end() ? it->second.get() : nullptr;
        if (!StateAccess::Restore(state, channel, any_channel, syncpoints)) {
            error = fmt::format("state record (kind {}, channel {}) doesn't fit this build",
                                header.kind, header.channel);
            return false;
        }
    }

    system.GPU().Start();
    if (!states.empty()) {
        std::promise<void> rebound;
        system.GPU().RunInOrder([this, &rebound] {
            for (auto& [recorded_id, channel] : channels) {
                system.GPU().BindChannel(channel->bind_id);
                StateAccess::Rebind(*channel);
            }
            rebound.set_value();
        });
        rebound.get_future().wait();
    }
    states.clear();
    std::filesystem::create_directories(options.out);
    last_frame = Clock::now();
    return true;
}

Tegra::MemoryManager* Replayer::AddressSpace(u32 id) {
    const auto it = memory_managers.find(id);
    return it != memory_managers.end() ? it->second.get() : nullptr;
}

void Replayer::MapGmmu(const GmmuMapRecord& map) {
    auto* const manager = AddressSpace(map.address_space);
    if (manager == nullptr) {
        LOG_ERROR(HW_GPU, "GPU trace: map in unknown address space {}", map.address_space);
        return;
    }
    if (map.sparse != 0) {
        manager->MapSparse(map.gpu_address, map.size, map.big_pages != 0);
    } else {
        manager->Map(map.gpu_address, map.device_address, map.size,
                     static_cast<Tegra::PTEKind>(map.kind), map.big_pages != 0);
    }
}

void Replayer::MapSmmu(const SmmuMapRecord& map) {
    auto& table = process->GetPageTable().GetImpl();
    auto* const base = system.DeviceMemory().buffer.BackingBasePointer();
    for (u64 offset = 0; offset < map.size; offset += PageSize) {
        const u64 physical = map.physical_address + offset;
        table.entries.GetAndFault(physical >> PageBits)
            .Store(false, Common::PageType::Memory, 1, reinterpret_cast<uintptr_t>(base));
        auto& pages = device_pages[physical >> PageBits];
        const u64 device_page = (map.device_address + offset) >> PageBits;
        if (std::ranges::find(pages, device_page) == pages.end()) {
            pages.push_back(device_page);
        }
    }
    auto& smmu = system.Host1x().MemoryManager();
    smmu.AllocateFixed(map.device_address, map.size);
    smmu.Map(map.device_address, map.physical_address, map.size, asid);
}

void Replayer::InvalidateDevice(u64 physical, u64 size) {
    auto* const rasterizer = system.GPU().Renderer().ReadRasterizer();
    const u64 end = physical + size;
    for (u64 page = physical >> PageBits; (page << PageBits) < end; ++page) {
        const auto it = device_pages.find(page);
        if (it == device_pages.end()) {
            continue;
        }
        const u64 begin = std::max(physical, page << PageBits);
        const u64 stop_at = std::min(end, (page + 1) << PageBits);
        for (const u64 device_page : it->second) {
            rasterizer->InvalidateRegion((device_page << PageBits) + (begin & (PageSize - 1)),
                                         stop_at - begin);
        }
    }
}

void Replayer::WriteMemory(const MemoryRecord& record, const std::vector<u8>& bytes) {
    u8* const pointer =
        system.DeviceMemory().buffer.BackingBasePointer() + record.physical_address;
    std::memcpy(pointer, bytes.data(), bytes.size());
    // The replay writes for the guest CPU: a recording of the replay sees it as one.
    NoteCpuWrite(pointer, bytes.size());
    InvalidateDevice(record.physical_address, record.size);
}

bool Replayer::Submit(const Record& record, std::string& error) {
    if (!HasPayload(record, sizeof(SubmitRecord))) {
        error = "short submit record";
        return false;
    }
    const auto& header = record.As<SubmitRecord>();
    const auto it = channels.find(header.channel);
    if (it == channels.end()) {
        error = fmt::format("submit to unknown channel {}", header.channel);
        return false;
    }
    size_t offset = sizeof(SubmitRecord);
    const auto entries = Array<u64>(record, offset, header.entry_count);
    const auto prefetch = Array<u32>(record, offset, header.prefetch_count);
    const auto offsets = Array<u32>(record, offset, header.offset_count);
    const auto words = Array<u32>(record, offset, header.word_count);
    const auto deferred = Array<u32>(record, offset, header.deferred_count);
    if (entries.size() != header.entry_count || prefetch.size() != header.prefetch_count ||
        offsets.size() != header.offset_count || words.size() != header.word_count ||
        deferred.size() != header.deferred_count) {
        error = "short submit record";
        return false;
    }
    Tegra::CommandList list;
    for (const u64 entry : entries) {
        list.command_lists.push_back(Tegra::CommandListHeader{entry});
    }
    for (const u32 word : prefetch) {
        list.prefetch_command_list.push_back(Tegra::CommandHeader{word});
    }
    list.submitted_offsets.assign(offsets.begin(), offsets.end());
    list.submitted_words.reserve(words.size());
    for (const u32 word : words) {
        list.submitted_words.push_back(Tegra::CommandHeader{word});
    }
    list.deferred_entries.assign(deferred.begin(), deferred.end());
    system.GPU().PushGPUEntries(it->second->bind_id, std::move(list));
    return true;
}

bool Replayer::Present(const Record& record, std::string& error) {
    if (!HasPayload(record, sizeof(PresentRecord))) {
        error = "short present record";
        return false;
    }
    const auto& header = record.As<PresentRecord>();
    size_t offset = sizeof(PresentRecord);
    const auto layers = Array<PresentLayer>(record, offset, header.layer_count);
    const auto fences = Array<PresentFence>(record, offset, header.fence_count);
    if (layers.size() != header.layer_count || fences.size() != header.fence_count) {
        error = "short present record";
        return false;
    }
    std::vector<Tegra::FramebufferConfig> configs;
    u32 width = 0;
    u32 height = 0;
    for (const PresentLayer& layer : layers) {
        configs.push_back(ToConfig(layer));
        const u32 layer_width =
            layer.crop_right > layer.crop_left ? static_cast<u32>(layer.crop_right - layer.crop_left)
                                               : layer.width;
        const u32 layer_height =
            layer.crop_bottom > layer.crop_top ? static_cast<u32>(layer.crop_bottom - layer.crop_top)
                                               : layer.height;
        width = std::max(width, layer_width);
        height = std::max(height, layer_height);
    }
    std::vector<Service::Nvidia::NvFence> nv_fences;
    for (const PresentFence& fence : fences) {
        nv_fences.push_back({fence.id, fence.value});
    }
    if (width == 0 || height == 0) {
        system.GPU().RequestComposite(std::move(configs), std::move(nv_fences));
        return true;
    }

    if (!options.record.empty() && presents == options.record_from) {
        RecordOptions recording{options.record, options.record_presents != 0
                                                    ? options.record_presents
                                                    : std::numeric_limits<u32>::max()};
        if (!RequestRecording(recording, error)) {
            return false;
        }
    }
    // A composite runs at the GPU thread's next tick, which can come before commands still
    // queued: let those finish first, as they had when the present was recorded.
    std::promise<void> drained;
    system.GPU().RunInOrder([&drained] { drained.set_value(); });
    drained.get_future().wait();

    std::vector<u8> pixels(size_t{width} * height * 4);
    std::promise<bool> captured;
    auto done = captured.get_future();
    system.GPU().Renderer().RequestScreenshot(
        pixels.data(), [&captured](bool) { captured.set_value(true); },
        Layout::DefaultFrameLayout(width, height));
    system.GPU().RequestComposite(std::move(configs), std::move(nv_fences));
    if (done.wait_for(std::chrono::seconds{120}) != std::future_status::ready) {
        error = fmt::format("present {} was never composited", presents);
        return false;
    }
    const auto now = Clock::now();
    for (size_t i = 0; i < pixels.size(); i += 4) {
        std::swap(pixels[i], pixels[i + 2]);
    }
    const std::string name = fmt::format("frame-{:04}.png", presents);
    const auto path = options.out / name;
    std::vector<u8> png;
    const auto append = [](void* context, void* data, int size) {
        auto& out = *static_cast<std::vector<u8>*>(context);
        const auto* bytes = static_cast<const u8*>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    std::ofstream file{path, std::ios::binary};
    if (stbi_write_png_to_func(append, &png, static_cast<int>(width), static_cast<int>(height), 4,
                               pixels.data(), static_cast<int>(width * 4)) == 0 ||
        !file.write(reinterpret_cast<const char*>(png.data()),
                    static_cast<std::streamsize>(png.size()))) {
        error = "can't write " + path.string();
        return false;
    }
    frames.push_back({
        {"index", presents},
        {"file", name},
        {"width", width},
        {"height", height},
        {"hash", HashString(HashBytes(pixels))},
        {"ms", std::chrono::duration<double, std::milli>(now - last_frame).count()},
    });
    last_frame = now;
    ++presents;
    if (options.max_frames != 0 && presents >= options.max_frames) {
        stop = true;
    }
    return true;
}

bool Replayer::Stream(std::string& error) {
    auto& gpu = system.GPU();
    while (!stop) {
        auto record = reader.Next();
        if (!record) {
            error = reader.Error().empty() ? "the trace has no end record" : reader.Error();
            return false;
        }
        switch (record->type) {
        case RecordType::End:
            stop = true;
            break;
        case RecordType::Blob:
            if (!ApplyBlob(*record, error)) {
                return false;
            }
            break;
        case RecordType::Memory: {
            if (!HasPayload(*record, sizeof(MemoryRecord))) {
                error = "short memory record";
                return false;
            }
            const auto memory = record->As<MemoryRecord>();
            const auto it = blobs.find(memory.hash);
            if (it == blobs.end() || it->second->size() != memory.size ||
                memory.physical_address + memory.size >
                    system.DeviceMemory().buffer.BackingSize()) {
                error = "a memory record names a missing blob or lies outside guest memory";
                return false;
            }
            // Before later reads: the split's front may parse the next list from this memory.
            gpu.RunInOrder([this, memory, bytes = it->second] { WriteMemory(memory, *bytes); },
                           true);
            break;
        }
        case RecordType::GmmuMap: {
            if (!HasPayload(*record, sizeof(GmmuMapRecord))) {
                error = "short GMMU map record";
                return false;
            }
            const auto map = record->As<GmmuMapRecord>();
            gpu.MapInOrder(map.gpu_address, map.size, [this, map] { MapGmmu(map); });
            break;
        }
        case RecordType::GmmuUnmap: {
            if (!HasPayload(*record, sizeof(GmmuUnmapRecord))) {
                error = "short GMMU unmap record";
                return false;
            }
            const auto unmap = record->As<GmmuUnmapRecord>();
            gpu.MapInOrder(unmap.gpu_address, unmap.size, [this, unmap] {
                if (auto* const manager = AddressSpace(unmap.address_space)) {
                    manager->Unmap(unmap.gpu_address, unmap.size);
                }
            });
            break;
        }
        case RecordType::SmmuMap: {
            if (!HasPayload(*record, sizeof(SmmuMapRecord))) {
                error = "short SMMU map record";
                return false;
            }
            const auto map = record->As<SmmuMapRecord>();
            gpu.RunInOrder([this, map] { MapSmmu(map); }, true);
            break;
        }
        case RecordType::SmmuUnmap: {
            if (!HasPayload(*record, sizeof(SmmuUnmapRecord))) {
                error = "short SMMU unmap record";
                return false;
            }
            const auto unmap = record->As<SmmuUnmapRecord>();
            gpu.RunInOrder(
                [this, unmap] {
                    system.Host1x().MemoryManager().Unmap(unmap.device_address, unmap.size);
                },
                true);
            break;
        }
        case RecordType::Submit:
            if (!Submit(*record, error)) {
                return false;
            }
            break;
        case RecordType::Present:
            if (!Present(*record, error)) {
                return false;
            }
            break;
        case RecordType::Resume:
            break;
        default:
            LOG_WARNING(HW_GPU, "GPU trace: skipping record type {} in the stream",
                        static_cast<u32>(record->type));
            break;
        }
    }
    gpu.WaitForIdle();
    if (!options.record.empty()) {
        // The recorder ends at the GPU thread's next command.
        StopRecording();
        while (GetRecorderStatus().requested) {
            gpu.WaitForIdle();
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        if (const auto status = GetRecorderStatus(); !status.last_error.empty()) {
            error = "recording: " + status.last_error;
            return false;
        }
    }
    return true;
}

void Replayer::Shutdown() {
    if (!powered) {
        return;
    }
    powered = false;
    system.GPU().WaitForIdle();
    channels.clear();
    system.ShutdownMainProcess();
    memory_managers.clear();
    if (process) {
        system.Kernel().UnregisterKernelObject(process.get());
        process.reset();
    }
}
} // Anonymous namespace

bool Replay(Core::System& system, const ReplayOptions& options, std::string& error) {
    Replayer replayer{system, options};
    return replayer.Run(error);
}

} // namespace VideoCommon::GpuTrace
