// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include <fmt/ranges.h>

#include "common/string_util.h"
#include "audio_core/audio_core.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/memory.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/audio/nx2_sysmix.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Service::Audio {

struct Nx2SysmixStreams {
    explicit Nx2SysmixStreams(Core::System& system_) : system{system_} {
        timer = Core::Timing::CreateEvent("sysmix output", [this](s64, std::chrono::nanoseconds late) {
            std::unique_lock lock{mutex, std::try_to_lock};
            if (lock) {
                for (auto& [id, stream] : streams) {
                    stream->Pump();
                }
            }
            const auto interval = std::chrono::nanoseconds{std::chrono::milliseconds{5}};
            const auto missed = std::max(late, std::chrono::nanoseconds::zero()) / interval;
            return std::optional<std::chrono::nanoseconds>{interval * (missed + 1)};
        });
        system.CoreTiming().ScheduleLoopingEvent(std::chrono::milliseconds{5},
                                                 std::chrono::milliseconds{5}, timer);
    }

    ~Nx2SysmixStreams() {
        system.CoreTiming().UnscheduleEvent(timer);
    }

    Result SetEngaged(u32 id, bool engaged, const Kernel::KProcess* process) {
        std::shared_ptr<SysmixOutput> stream;
        {
            std::scoped_lock lock{mutex};
            const auto it = streams.find(id);
            R_UNLESS(it != streams.end(), ResultNotFound);
            stream = it->second;
        }
        R_UNLESS(stream->IsOwnedBy(process), ResultInvalidHandle);
        return stream->SetEngaged(engaged);
    }

    /// Engage an owned stream only when its direction matches the device functionality.
    Result Attach(u32 id, u32 functionality, const Kernel::KProcess* process) {
        std::shared_ptr<SysmixOutput> stream;
        {
            std::scoped_lock lock{mutex};
            const auto it = streams.find(id);
            R_UNLESS(it != streams.end(), ResultNotFound);
            stream = it->second;
        }
        R_UNLESS(stream->IsOwnedBy(process), ResultInvalidHandle);
        R_UNLESS(functionality == (stream->IsInput() ? 2U : 1U), ResultNotSupported);
        return stream->SetEngaged(true);
    }

    std::mutex mutex;
    std::map<u32, std::shared_ptr<SysmixOutput>> streams;
    u32 next_id{1};
    Core::System& system;
    std::shared_ptr<Core::Timing::EventType> timer;
};

// Switch 2 virtual output devices (nn::audio::ListVirtualOutputDevices and friends), reached through
// the object auddmg:u command 0 returns. Eden plays everything through one stereo 48 kHz output, so
// there is one virtual device. The nx2: name keeps the other commands auto-stubbed and logged.
// - 0(u32): u32 count, the device names in the out buffer (0x100 bytes each).
// - 100(u32): the default device's name in the out buffer.
// - 300(16-byte stream format, name buffer): whether the format is supported (result only).
// - 401(u32, name buffer): the device's 0x228-byte property block in the out buffer. The sdk hands
//   the game the name, the channel count (0x108) and the u64 at 0x218; with the zeroed placeholder
//   reply it saw 0 channels and Switch Sports failed to set up audio (2153-1604).
// - 500(16 bytes, name buffer): an event signalled when the device's state changes.
// - 700(u32 stream id, u32 device functionality, name buffer): attach a sysmix stream to the
//   device, which starts it (nn::audio::devmgr::AttachStream, behind OutputStream Start).
// - 800(u32 stream id): detach it again (Stop).
class INx2AudioVirtualDeviceManager final : public ServiceFramework<INx2AudioVirtualDeviceManager> {
public:
    explicit INx2AudioVirtualDeviceManager(Core::System& system_,
                                          std::shared_ptr<Nx2SysmixStreams> streams_)
        : ServiceFramework{system_, "nx2:auddmg:u:manager"},
          service_context{system_, "nx2:auddmg:u:manager"},
          state_event{service_context.CreateEvent("nx2:auddmg:u:StateEvent")},
          streams{std::move(streams_)} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &INx2AudioVirtualDeviceManager::ListVirtualOutputDevices, "ListVirtualOutputDevices"},
            {100, &INx2AudioVirtualDeviceManager::GetDefaultVirtualOutputDevice, "GetDefaultVirtualOutputDevice"},
            {300, &INx2AudioVirtualDeviceManager::IsStreamFormatSupported, "IsVirtualOutputDeviceStreamFormatSupported"},
            {401, &INx2AudioVirtualDeviceManager::GetVirtualOutputDeviceProperties, "GetVirtualOutputDeviceProperties"},
            {500, &INx2AudioVirtualDeviceManager::AcquireStateChangeEvent, "AcquireVirtualOutputDeviceStateEvent (guessed)"},
            {700, &INx2AudioVirtualDeviceManager::AttachStream, "AttachStream"},
            {800, &INx2AudioVirtualDeviceManager::DetachStream, "DetachStream"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

    ~INx2AudioVirtualDeviceManager() override {
        service_context.CloseEvent(state_event);
    }

private:
    static constexpr std::size_t NameSize = 0x100;
    static constexpr std::string_view DeviceName = "AudioVirtualOutput0";
    static constexpr std::string_view PhysicalDeviceName = "AudioTvOutput";
    static constexpr std::string_view InputDeviceName = "AudioVirtualInput0";
    static constexpr std::string_view PhysicalInputDeviceName = "AudioBuiltInMicrophone";

    static void CopyName(u8* out, std::string_view name) {
        std::memcpy(out, name.data(), std::min(name.size(), NameSize - 1));
    }

    /// List the virtual microphone for input functionality, or the virtual playback device.
    void ListVirtualOutputDevices(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto functionality = rp.Pop<u32>();
        std::array<u8, NameSize> names{};
        CopyName(names.data(), functionality == 2 ? InputDeviceName : DeviceName);
        const u32 count = ctx.GetWriteBufferSize() >= NameSize ? 1 : 0;
        if (count) {
            ctx.WriteBuffer(names);
        }
        LOG_INFO(Service_Audio, "list devices: functionality {}, count {}", functionality, count);
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push(count);
    }

    /// Return the default virtual device name for the requested stream direction.
    void GetDefaultVirtualOutputDevice(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto functionality = rp.Pop<u32>();
        std::array<u8, NameSize> name{};
        CopyName(name.data(), functionality == 2 ? InputDeviceName : DeviceName);
        ctx.WriteBuffer(name);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    /// Reject formats or functionalities that the corresponding host stream cannot handle.
    void IsStreamFormatSupported(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto type = rp.Pop<u32>();
        const auto channels = rp.Pop<u32>();
        const auto rate = rp.Pop<u32>();
        const auto functionality = rp.Pop<u32>();
        const auto supported =
            functionality == 2
                ? SysmixOutput::IsInputFormatSupported(type, channels, rate)
                : functionality == 1 && SysmixOutput::IsFormatSupported(type, channels, rate);
        const auto result = supported ? ResultSuccess : ResultNotSupported;
        LOG_INFO(Service_Audio,
                  "format support: sample type {}, {} channels, {} Hz, functionality {}, result {:#x}",
                  type, channels, rate, functionality, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    /**
     * Return the direction's properties: name (0x000), type (0x100, 3), u32 (0x104, 2),
     * channels (0x108), rate (0x10c), physical name (0x110), physical ID (0x210),
     * latency in ns (0x218), device ID (0x220). CreateStream requires a nonzero device ID
     * (Switch Sports failed with 2153-1604 on a zero there).
     */
    void GetVirtualOutputDeviceProperties(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto functionality = rp.Pop<u32>();
        const bool input = functionality == 2;
        std::array<u8, 0x228> block{};
        const auto name = ctx.ReadBuffer(0);
        std::memcpy(block.data(), name.data(), std::min(name.size(), NameSize - 1));
        if (block[0] == 0) {
            CopyName(block.data(), input ? InputDeviceName : DeviceName);
        }
        const auto put32 = [&](std::size_t offset, u32 value) {
            std::memcpy(block.data() + offset, &value, sizeof(value));
        };
        put32(0x100, 3);
        put32(0x104, 2);
        // Match the output channel count audren reports. Orbitals rejects a stereo device after
        // GetActiveChannelCount said six and retries forever.
        const bool surround =
            !input && system.AudioCore().GetOutputSink().GetSystemChannels() == 6;
        put32(0x108, surround ? 6 : 2);
        put32(0x10c, 48000); // Hz
        CopyName(block.data() + 0x110, input ? PhysicalInputDeviceName : PhysicalDeviceName);
        const u64 latency_ns = 10'000'000;
        std::memcpy(block.data() + 0x218, &latency_ns, sizeof(latency_ns));
        put32(0x220, input ? 2 : 1);
        ctx.WriteBuffer(block);
        LOG_INFO(Service_Audio, "properties: functionality {}, device {}", functionality,
                 Common::StringFromBuffer(name));
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void AcquireStateChangeEvent(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess);
        rb.PushCopyObjects(ctx, state_event->GetReadableEvent());
    }

    /// Validate stream ownership and direction before attaching it and starting host audio.
    void AttachStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto stream_id = rp.Pop<u32>();
        const auto functionality = rp.Pop<u32>();
        LOG_INFO(Service_Audio, "attach stream {} to {} (functionality {})", stream_id,
                 Common::StringFromBuffer(ctx.ReadBuffer(0)), functionality);
        const auto result =
            streams->Attach(stream_id, functionality, ctx.GetThread().GetOwnerProcess());
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    void DetachStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto stream_id = rp.Pop<u32>();
        LOG_INFO(Service_Audio, "detach stream {}", stream_id);
        const auto result = streams->SetEngaged(stream_id, false, ctx.GetThread().GetOwnerProcess());
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* state_event;
    std::shared_ptr<Nx2SysmixStreams> streams;
};

// Switch 2 "auddmg:u": command 0 (16 bytes and the pid) returns the virtual device manager.
class INx2AudioDeviceManagerService final : public ServiceFramework<INx2AudioDeviceManagerService> {
public:
    explicit INx2AudioDeviceManagerService(Core::System& system_,
                                          std::shared_ptr<Nx2SysmixStreams> streams_)
        : ServiceFramework{system_, "nx2:auddmg:u"}, streams{std::move(streams_)} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &INx2AudioDeviceManagerService::OpenVirtualDeviceManager, "OpenVirtualDeviceManager (guessed)"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    void OpenVirtualDeviceManager(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "called");
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<INx2AudioVirtualDeviceManager>(ctx, system, streams);
    }

    std::shared_ptr<Nx2SysmixStreams> streams;
};

// The stream object IUserSession command 1 returns. Commands (ids, request -> reply):
// - 0(u32 sample type, u32 channels, u32 sample rate, u32 frames, u32 prefill, u32 flags): open.
// - 1: close. 2: u32 state. 3: f32 gain. 4(f32 gain, s64 ramp ns): set the gain.
// - 5 / 6: acquire (copy handle) / release the state-change event.
// - 7(u32 mode, s32 threshold) / 8: acquire (copy handle) / release the data event.
// - 9(s64 ns): clock the stream. 10(s32 frames) -> u32: prime. 11(s32 frames) -> u32: purge.
// - 12(name buffer): rename.
class INx2SysmixStreamSession final : public ServiceFramework<INx2SysmixStreamSession> {
public:
    INx2SysmixStreamSession(Core::System& system_, std::shared_ptr<Nx2SysmixStreams> streams_,
                           std::shared_ptr<SysmixOutput> stream_, u32 id_)
        : ServiceFramework{system_, "nx2:audsmx:u:stream"}, streams{std::move(streams_)},
          stream{std::move(stream_)}, id{id_} {
        static const FunctionInfo functions[] = {
            {0, &INx2SysmixStreamSession::OpenStream, "OpenStream"},
            {1, &INx2SysmixStreamSession::CloseStream, "CloseStream"},
            {2, &INx2SysmixStreamSession::GetStreamState, "GetStreamState"},
            {3, &INx2SysmixStreamSession::GetStreamGain, "GetStreamGain"},
            {4, &INx2SysmixStreamSession::SetStreamGain, "SetStreamGain"},
            {5, &INx2SysmixStreamSession::AcquireStateChangeEvent, "AcquireStreamStateChangeEvent"},
            {6, &INx2SysmixStreamSession::Succeed, "ReleaseStreamStateChangeEvent"},
            {7, &INx2SysmixStreamSession::AcquireDataEvent, "AcquireStreamDataEvent"},
            {8, &INx2SysmixStreamSession::ReleaseDataEvent, "ReleaseStreamDataEvent"},
            {9, &INx2SysmixStreamSession::Unsupported, "ClockStream"},
            {10, &INx2SysmixStreamSession::Unsupported, "PrimeStream"},
            {11, &INx2SysmixStreamSession::PurgeStream, "PurgeStream"},
            {12, &INx2SysmixStreamSession::Succeed, "RenameStream"},
        };
        RegisterHandlers(functions);
    }

    ~INx2SysmixStreamSession() override {
        {
            std::scoped_lock lock{streams->mutex};
            streams->streams.erase(id);
        }
        stream->Close();
    }

private:
    void OpenStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto sample_type = rp.Pop<u32>();
        const auto channels = rp.Pop<u32>();
        const auto sample_rate = rp.Pop<u32>();
        const auto frames = rp.Pop<u32>();
        const auto prefill = rp.Pop<u32>();
        const auto flags = rp.Pop<u32>();
        const auto result = stream->Open(sample_type, channels, sample_rate, frames, prefill, flags);
        LOG_INFO(Service_Audio,
                 "stream {} open: sample type {}, {} channels, {} Hz, {} frames, prefill {}, flags {:#x}, result {:#x}",
                 id, sample_type, channels, sample_rate, frames, prefill, flags, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    void CloseStream(HLERequestContext& ctx) {
        stream->Close();
        Succeed(ctx);
    }

    void GetStreamState(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push(stream->GetState());
    }

    void GetStreamGain(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push(stream->GetGain());
    }

    void SetStreamGain(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto gain = rp.Pop<f32>();
        rp.Pop<u32>();
        const auto ramp = rp.Pop<s64>();
        const auto result = stream->SetGain(gain, ramp);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    void AcquireStateChangeEvent(HLERequestContext& ctx) {
        auto* event = stream->GetStateEvent();
        IPC::ResponseBuilder rb{ctx, 2, event ? 1u : 0u};
        rb.Push(event ? ResultSuccess : ResultOutOfSessions);
        if (event) {
            rb.PushCopyObjects(ctx, event);
        }
    }

    void AcquireDataEvent(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto mode = rp.Pop<u32>();
        const auto threshold = rp.Pop<s32>();
        const auto result = stream->SetDataNotification(mode, threshold);
        IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1u : 0u};
        rb.Push(result);
        if (result.IsSuccess()) {
            rb.PushCopyObjects(ctx, stream->GetDataEvent());
        }
    }

    void ReleaseDataEvent(HLERequestContext& ctx) {
        const auto result = stream->SetDataNotification(0, 0);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    /// Discard requested input frames and return only the count actually purged.
    void PurgeStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto frames = rp.Pop<u32>();
        u32 purged{};
        const auto result = stream->Purge(frames, purged);
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result);
        rb.Push(purged);
    }

    void Unsupported(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultNotSupported);
    }

    void Succeed(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    std::shared_ptr<Nx2SysmixStreams> streams;
    std::shared_ptr<SysmixOutput> stream;
    u32 id;
};

class INx2SysmixUserSession final : public ServiceFramework<INx2SysmixUserSession> {
public:
    INx2SysmixUserSession(Core::System& system_, std::shared_ptr<Nx2SysmixStreams> streams_,
                         Kernel::KProcess* process_)
        : ServiceFramework{system_, "nx2:audsmx:u:session"}, streams{std::move(streams_)},
          process{system.Kernel(), process_} {
        static const FunctionInfo functions[] = {
            {1, &INx2SysmixUserSession::CreateStream, "CreateStream"},
            {2, &INx2SysmixUserSession::EngageStreams, "EngageStreams"},
            {3, &INx2SysmixUserSession::DisengageStreams, "DisengageStreams"},
        };
        RegisterHandlers(functions);
    }

private:
    /// Create an owned input or output stream after validating its device and guest ring region.
    void CreateStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto device = rp.Pop<u32>();
        const auto flags = rp.Pop<u32>();
        [[maybe_unused]] const auto applet_resource_user_id = rp.Pop<u64>();
        const auto buffer = rp.Pop<u64>();
        const auto size = rp.Pop<u64>();
        Result result = ResultInvalidArgument;
        if ((device == 1 || device == 2) &&
            ctx.GetThread().GetOwnerProcess() == process.GetPointerUnsafe()) {
            result = SysmixOutput::ValidateRegion(*process, buffer, size, flags, device == 2);
        }
        if (result.IsFailure()) {
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(result);
            return;
        }
        u32 id{};
        {
            std::scoped_lock lock{streams->mutex};
            id = streams->next_id++;
        }
        auto stream = std::make_shared<SysmixOutput>(system, *process,
            device == 2 ? system.AudioCore().GetInputSink() : system.AudioCore().GetOutputSink(),
            id, buffer, size, flags, device == 2);
        {
            std::scoped_lock lock{streams->mutex};
            streams->streams.emplace(id, stream);
        }
        LOG_INFO(Service_Audio, "stream {} '{}': flags {:#x}, buffer {:#x} size {:#x}", id,
                 Common::StringFromBuffer(ctx.ReadBuffer(0)), flags, buffer, size);
        IPC::ResponseBuilder rb{ctx, 3, 0, 1};
        rb.Push(ResultSuccess);
        rb.Push(id);
        rb.PushIpcInterface<INx2SysmixStreamSession>(ctx, system, streams, stream, id);
    }

    void SetEngaged(HLERequestContext& ctx, bool engaged) {
        const auto buffer = ctx.ReadBuffer(0);
        Result result = ResultInvalidArgument;
        if (buffer.size() % sizeof(u32) == 0) {
            result = ResultSuccess;
            for (std::size_t offset = 0; offset < buffer.size(); offset += sizeof(u32)) {
                u32 id{};
                std::memcpy(&id, buffer.data() + offset, sizeof(id));
                result = streams->SetEngaged(id, engaged, ctx.GetThread().GetOwnerProcess());
                if (result.IsFailure()) {
                    break;
                }
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    void EngageStreams(HLERequestContext& ctx) {
        SetEngaged(ctx, true);
    }

    void DisengageStreams(HLERequestContext& ctx) {
        SetEngaged(ctx, false);
    }

    std::shared_ptr<Nx2SysmixStreams> streams;
    Kernel::KScopedAutoObject<Kernel::KProcess> process;
};

class INx2AudioSoundMixerService final : public ServiceFramework<INx2AudioSoundMixerService> {
public:
    INx2AudioSoundMixerService(Core::System& system_, std::shared_ptr<Nx2SysmixStreams> streams_)
        : ServiceFramework{system_, "nx2:audsmx:u"}, streams{std::move(streams_)} {
        static const FunctionInfo functions[] = {
            {0, &INx2AudioSoundMixerService::OpenUserSession, "OpenUserSession"},
        };
        RegisterHandlers(functions);
    }

private:
    void OpenUserSession(HLERequestContext& ctx) {
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(ctx.GetCopyHandle(0));
        if (process.IsNull() || process.GetPointerUnsafe() != ctx.GetThread().GetOwnerProcess()) {
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultInvalidHandle);
            return;
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<INx2SysmixUserSession>(ctx, system, streams, process.GetPointerUnsafe());
    }

    std::shared_ptr<Nx2SysmixStreams> streams;
};

} // namespace Service::Audio
