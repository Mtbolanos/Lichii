// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2026 Three-Dola
// SPDX-License-Identifier: MIT

// Synthesis adapted from RenHDrum src/dualsense/haptics.rs: https://github.com/trzz1214/RenHDrum

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <string>

#include "common/logging.h"
#include "input_common/helpers/dualsense_haptics.h"

namespace InputCommon {

namespace {
constexpr int SampleRate = 48000;
constexpr int Channels = 4;
// Channels 1 and 2 are the speaker and headset, 3 and 4 the left and right actuators
constexpr std::size_t FirstActuatorChannel = 2;
constexpr std::size_t LeftActuator = 0;
constexpr std::size_t RightActuator = 1;

// Targets are picked up in fixed blocks so the feel doesn't depend on the host buffer size
constexpr int UpdateFrames = 256;

// Keeps sustained output away from the level where the coils heat up and distort
constexpr f32 OutputGain = 0.8f;

// One pole smoothing with a 4ms time constant on amplitude and frequency changes
const f32 SmoothingFactor = static_cast<f32>(1.0 - std::exp(-1000.0 / (SampleRate * 4.0)));

constexpr f32 MinFrequency = 20.0f;
constexpr f32 MaxFrequency = 1000.0f;
constexpr f32 DefaultLowFrequency = 160.0f;
constexpr f32 DefaultHighFrequency = 320.0f;

// Matches the duration SDL rumble requests are sent with
constexpr u64 VibrationTimeoutMs = 2000;

// Only one controller can own the audio device
std::atomic<bool> device_in_use{false};

f32 SanitizeAmplitude(f32 amplitude) {
    if (!std::isfinite(amplitude)) {
        return 0.0f;
    }
    // Square root curve so faint vibrations stay above what can be felt on the actuators
    return std::sqrt(std::clamp(amplitude, 0.0f, 1.0f));
}

f32 SanitizeFrequency(f32 frequency, f32 fallback) {
    if (!(frequency > 0.0f)) {
        return fallback;
    }
    return std::clamp(frequency, MinFrequency, MaxFrequency);
}

// The endpoint is named after the USB product ("DualSense Wireless Controller", "DualSense Edge
// Wireless Controller"), but Wine and some Linux setups only show "Wireless Controller"
bool IsDualSenseAudioDeviceName(const char* name) {
    return std::strstr(name, "DualSense") != nullptr ||
           std::strstr(name, "Wireless Controller") != nullptr;
}
} // Anonymous namespace

DualSenseHaptics::~DualSenseHaptics() {
    Close();
}

bool DualSenseHaptics::Open() {
    if (stream != nullptr) {
        return true;
    }
    if (device_in_use.exchange(true)) {
        return false;
    }

    if (!SDL_WasInit(SDL_INIT_AUDIO) && !SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        LOG_ERROR(Input, "Failed to initialize SDL audio for DualSense haptics: {}",
                  SDL_GetError());
        device_in_use = false;
        return false;
    }

    int device_count = 0;
    SDL_AudioDeviceID* devices = SDL_GetAudioPlaybackDevices(&device_count);
    SDL_AudioDeviceID haptics_device = 0;
    std::string haptics_device_name;
    int candidate_count = 0;
    for (int i = 0; devices != nullptr && i < device_count; ++i) {
        const char* name = SDL_GetAudioDeviceName(devices[i]);
        if (name == nullptr || !IsDualSenseAudioDeviceName(name)) {
            continue;
        }
        // The actuators need all four channels. This also skips the stereo audio device of a
        // DualShock 4, which is called "Wireless Controller" too.
        SDL_AudioSpec device_spec{};
        if (!SDL_GetAudioDeviceFormat(devices[i], &device_spec, nullptr) ||
            device_spec.channels < Channels) {
            continue;
        }
        haptics_device = devices[i];
        haptics_device_name = name;
        ++candidate_count;
    }
    SDL_free(devices);

    // SDL doesn't tell which audio device belongs to which gamepad. With more than one the
    // haptics could end up on the wrong controller, so use standard rumble instead.
    if (candidate_count != 1) {
        if (candidate_count > 1) {
            LOG_DEBUG(Input, "Found {} DualSense audio devices, using standard rumble",
                      candidate_count);
        }
        device_in_use = false;
        return false;
    }
    LOG_INFO(Input, "Using audio device \"{}\" for DualSense haptics", haptics_device_name);

    const SDL_AudioSpec spec{
        .format = SDL_AUDIO_F32,
        .channels = Channels,
        .freq = SampleRate,
    };
    stream = SDL_OpenAudioDeviceStream(haptics_device, &spec, AudioCallback, this);
    if (stream == nullptr) {
        LOG_ERROR(Input, "Failed to open DualSense haptics stream: {}", SDL_GetError());
        device_in_use = false;
        return false;
    }

    if (!SDL_ResumeAudioStreamDevice(stream)) {
        LOG_ERROR(Input, "Failed to start DualSense haptics stream: {}", SDL_GetError());
        Close();
        return false;
    }
    device = haptics_device;
    return true;
}

void DualSenseHaptics::Close() {
    if (stream == nullptr) {
        return;
    }
    SDL_DestroyAudioStream(stream);
    stream = nullptr;
    device = 0;
    device_in_use = false;
}

void DualSenseHaptics::SetVibration(const Common::Input::VibrationStatus& vibration) {
    f32 low_amplitude = SanitizeAmplitude(vibration.low_amplitude);
    f32 high_amplitude = SanitizeAmplitude(vibration.high_amplitude);

    // Both bands are mixed into the same signal, keep the sum within full scale
    const f32 total = low_amplitude + high_amplitude;
    if (total > 1.0f) {
        low_amplitude /= total;
        high_amplitude /= total;
    }

    const f32 low_frequency = SanitizeFrequency(vibration.low_frequency, DefaultLowFrequency);
    const f32 high_frequency = SanitizeFrequency(vibration.high_frequency, DefaultHighFrequency);
    const auto set_target = [&](Target& target) {
        target.low_amplitude = low_amplitude;
        target.low_frequency = low_frequency;
        target.high_amplitude = high_amplitude;
        target.high_frequency = high_frequency;
    };
    if (vibration.side != Common::Input::VibrationSide::Right) {
        set_target(targets[LeftActuator]);
    }
    if (vibration.side != Common::Input::VibrationSide::Left) {
        set_target(targets[RightActuator]);
    }
    last_update_ms = SDL_GetTicks();
}

void SDLCALL DualSenseHaptics::AudioCallback(void* userdata, SDL_AudioStream* stream,
                                             int additional_amount, int total_amount) {
    if (additional_amount <= 0) {
        return;
    }
    auto* haptics = static_cast<DualSenseHaptics*>(userdata);
    const int frames = additional_amount / static_cast<int>(sizeof(f32) * Channels);
    haptics->Render(frames);
    SDL_PutAudioStreamData(stream, haptics->buffer.data(),
                           static_cast<int>(haptics->buffer.size() * sizeof(f32)));
}

void DualSenseHaptics::Advance(Band& band, const Band& target) {
    band.amplitude += (target.amplitude - band.amplitude) * SmoothingFactor;
    band.frequency += (target.frequency - band.frequency) * SmoothingFactor;
    band.phase += static_cast<f64>(band.frequency) / SampleRate;
    if (band.phase >= 1.0) {
        band.phase -= 1.0;
    }
}

void DualSenseHaptics::UpdateTargets() {
    // Both sides fade out together once neither has been updated for a while
    const bool timed_out = SDL_GetTicks() - last_update_ms.load() > VibrationTimeoutMs;
    for (std::size_t i = 0; i < actuators.size(); ++i) {
        const Target& target = targets[i];
        Actuator& actuator = actuators[i];
        actuator.low_target.amplitude = timed_out ? 0.0f : target.low_amplitude.load();
        actuator.low_target.frequency = target.low_frequency.load();
        actuator.high_target.amplitude = timed_out ? 0.0f : target.high_amplitude.load();
        actuator.high_target.frequency = target.high_frequency.load();
    }
    frames_until_update = UpdateFrames;
}

void DualSenseHaptics::Render(int frames) {
    constexpr f64 TwoPi = 2.0 * std::numbers::pi;
    buffer.assign(static_cast<std::size_t>(frames) * Channels, 0.0f);

    for (int i = 0; i < frames; ++i) {
        if (frames_until_update == 0) {
            UpdateTargets();
        }
        --frames_until_update;

        f32* frame = &buffer[static_cast<std::size_t>(i) * Channels];
        for (std::size_t a = 0; a < actuators.size(); ++a) {
            Actuator& actuator = actuators[a];
            Advance(actuator.low, actuator.low_target);
            Advance(actuator.high, actuator.high_target);

            const f64 value = actuator.low.amplitude * std::sin(TwoPi * actuator.low.phase) +
                              actuator.high.amplitude * std::sin(TwoPi * actuator.high.phase);
            frame[FirstActuatorChannel + a] =
                static_cast<f32>(std::clamp(value, -1.0, 1.0)) * OutputGain;
        }
    }
}

} // namespace InputCommon
