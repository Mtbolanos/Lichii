// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/audio_core.h"
#include "audio_core/audio_manager.h"
#include "audio_core/device/audio_buffer.h"
#include "audio_core/device/device_session.h"
#include "audio_core/sink/silent_input_stream.h"
#include "audio_core/sink/sink_stream.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/guest_memory.h"
#include "core/memory.h"

#include "core/hle/kernel/k_process.h"

namespace AudioCore {

using namespace std::literals;
constexpr auto INCREMENT_TIME{5ms};

DeviceSession::DeviceSession(Core::System& system_)
    : system{system_}, thread_event{Core::Timing::CreateEvent(
                           "AudioOutSampleTick",
                           [this](s64 time, std::chrono::nanoseconds) { return ThreadFunc(); })} {}

DeviceSession::~DeviceSession() {
    Finalize();
}

/// Acquire the host stream and process reference; capture falls back to silence.
Result DeviceSession::Initialize(std::string_view name_, SampleFormat sample_format_,
                                 u16 channel_count_, size_t session_id_, Kernel::KProcess* handle_,
                                 u64 applet_resource_user_id_, Sink::StreamType type_) {
    if (stream) {
        Finalize();
    }
    name = fmt::format("{}-{}", name_, session_id_);
    type = type_;
    sample_format = sample_format_;
    channel_count = channel_count_;
    session_id = session_id_;
    handle = handle_;
    handle->Open(system.Kernel());
    applet_resource_user_id = applet_resource_user_id_;

    if (type == Sink::StreamType::In) {
        sink = &system.AudioCore().GetInputSink();
    } else {
        sink = &system.AudioCore().GetOutputSink();
    }
    stream = sink->AcquireSinkStream(system, channel_count, name, type);
    if (type == Sink::StreamType::In && !stream->IsReady()) {
        UseSilentInput();
    }
    initialized = true;
    return ResultSuccess;
}

/// Release the host stream and process reference, including after a failed initialization.
void DeviceSession::Finalize() {
    if (initialized) {
        Stop();
        ReleaseStream();
        initialized = false;
    }

    if (handle) {
        handle->Close(system.Kernel());
        handle = nullptr;
    }
}

/// Start the host stream before scheduling audio ticks; capture falls back to silence.
Result DeviceSession::Start() {
    if (stream) {
        stream->Start();
        if (type == Sink::StreamType::In && (!stream->IsReady() || stream->IsPaused())) {
            UseSilentInput();
            stream->Start();
        }
        system.CoreTiming().ScheduleLoopingEvent(std::chrono::nanoseconds::zero(), INCREMENT_TIME,
                                                 thread_event);
        return ResultSuccess;
    }
    return Service::Audio::ResultOperationFailed;
}

void DeviceSession::Stop() {
    if (stream) {
        stream->Stop();
        system.CoreTiming().UnscheduleEvent(thread_event, {});
    }
}

void DeviceSession::ClearBuffers() {
    if (stream) {
        stream->ClearQueue();
    }
}

void DeviceSession::AppendBuffers(std::span<const AudioBuffer> buffers) {
    for (const auto& buffer : buffers) {
        Sink::SinkBuffer new_buffer{
            .frames = buffer.size / (channel_count * sizeof(s16)),
            .frames_played = 0,
            .tag = buffer.tag,
            .consumed = false,
        };

        tmp_samples.resize_destructive(buffer.size / sizeof(s16));
        if (type == Sink::StreamType::In) {
            stream->AppendBuffer(new_buffer, tmp_samples);
        } else {
            Core::Memory::CpuGuestMemory<s16, Core::Memory::GuestMemoryFlags::UnsafeRead> samples(
                handle->GetMemory(), buffer.samples, buffer.size / sizeof(s16));
            stream->AppendBuffer(new_buffer, samples);
        }
    }
}

/// Copy captured PCM to the guest, padding short or odd-sized buffers with silence.
void DeviceSession::ReleaseBuffer(const AudioBuffer& buffer) const {
    if (type == Sink::StreamType::In) {
        auto samples{stream->ReleaseBuffer(buffer.size / sizeof(s16))};
        samples.resize(buffer.size / sizeof(s16) + (buffer.size % sizeof(s16) != 0), 0);
        handle->GetMemory().WriteBlockUnsafe(buffer.samples, samples.data(), buffer.size);
    }
}

bool DeviceSession::IsBufferConsumed(const AudioBuffer& buffer) const {
    return played_sample_count >= buffer.end_timestamp;
}

void DeviceSession::SetVolume(f32 volume) const {
    if (stream) {
        stream->SetSystemVolume(volume);
    }
}

u64 DeviceSession::GetPlayedSampleCount() const {
    return played_sample_count;
}

std::optional<std::chrono::nanoseconds> DeviceSession::ThreadFunc() {
    played_sample_count = stream->GetExpectedPlayedSampleCount();
    if (type == Sink::StreamType::Out) {
        system.AudioCore().GetAudioManager().SetEvent(Event::Type::AudioOutManager, true);
    } else {
        system.AudioCore().GetAudioManager().SetEvent(Event::Type::AudioInManager, true);
    }
    return std::nullopt;
}

void DeviceSession::ReleaseStream() {
    if (stream == silent_input.get()) {
        silent_input.reset();
    } else {
        sink->CloseStream(stream);
    }
    stream = nullptr;
}

/// Replace a host microphone that failed to open or start with silence the guest can capture.
void DeviceSession::UseSilentInput() {
    LOG_WARNING(Service_Audio, "{}: host microphone unavailable, capturing silence", name);
    ReleaseStream();
    silent_input = std::make_unique<Sink::SilentInputStream>(system, channel_count);
    stream = silent_input.get();
}

void DeviceSession::SetRingSize(u32 ring_size) {
    stream->SetRingSize(ring_size);
}

} // namespace AudioCore
