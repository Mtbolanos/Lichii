// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"

namespace Core {
class System;
} // namespace Core

namespace AudioCore::Sink {
class NullSinkStreamImpl final : public SinkStream {
public:
    explicit NullSinkStreamImpl(Core::System& system_, StreamType type_)
        : SinkStream{system_, type_} {}
    ~NullSinkStreamImpl() override {}
    void AppendBuffer(SinkBuffer&, std::span<s16>) override {}
    bool TryAppendOutputBlock(SinkBuffer& buffer, std::span<s16> samples) override {
        return buffer.frames == TargetSampleCount && buffer.frames_played == 0 &&
               !buffer.consumed &&
               (samples.size() == TargetSampleCount * 2 || samples.size() == TargetSampleCount * 6);
    }
    /// Preserve guest PCM pacing without host callbacks: one block per 5 ms pump.
    u32 SysmixQueueLimit() const override {
        return 1;
    }
    bool IsReady() const override {
        return true;
    }
    void Start(bool = false) override {
        paused = false;
    }
    void Stop() override {
        SignalPause();
    }
    std::vector<s16> ReleaseBuffer(u64) override {
        return {};
    }
};

/**
 * A no-op sink for when no audio out is wanted.
 */
class NullSink final : public Sink {
public:
    explicit NullSink(std::string_view) {}
    ~NullSink() override = default;

    SinkStream* AcquireSinkStream(Core::System& system, u32, const std::string&,
                                  StreamType type) override {
        if (null_sink == nullptr) {
            null_sink = std::make_unique<NullSinkStreamImpl>(system, type);
            null_sink->SetDeviceVolume(device_volume);
        }
        return null_sink.get();
    }

    void CloseStream(SinkStream*) override {}
    void CloseStreams() override {}
    f32 GetDeviceVolume() const override {
        return device_volume;
    }
    void SetDeviceVolume(f32 volume) override {
        device_volume = volume;
        if (null_sink != nullptr) {
            null_sink->SetDeviceVolume(volume);
        }
    }
    void SetSystemVolume(f32 volume) override {}

private:
    SinkStreamPtr null_sink{};
};

} // namespace AudioCore::Sink
