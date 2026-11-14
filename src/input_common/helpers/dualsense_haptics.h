// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2026 Three-Dola
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <atomic>
#include <vector>

#include <SDL3/SDL.h>

#include "common/common_types.h"
#include "common/input.h"

namespace InputCommon {

// Drives the DualSense voice coil actuators through the controller's USB audio device. The
// actuators are exposed as channels 3 and 4 of a 4 channel playback device, so HD rumble can be
// reproduced by synthesizing the requested frequencies instead of collapsing them into motor
// strengths. The left side of the emulated controller plays on the left actuator and the right
// side on the right one.
class DualSenseHaptics {
public:
    DualSenseHaptics() = default;
    ~DualSenseHaptics();

    DualSenseHaptics(const DualSenseHaptics&) = delete;
    DualSenseHaptics& operator=(const DualSenseHaptics&) = delete;

    // Opens the DualSense audio device. Returns false if there is none, more than one, or it is
    // already used by another controller.
    bool Open();
    void Close();

    bool IsOpen() const {
        return stream != nullptr;
    }

    // True when the stream plays on the given physical audio device
    bool UsesDevice(SDL_AudioDeviceID device_id) const {
        return stream != nullptr && device == device_id;
    }

    // Amplitudes are in the range [0, 1], frequencies in Hz.
    void SetVibration(const Common::Input::VibrationStatus& vibration);

private:
    struct Band {
        f32 amplitude{};
        f32 frequency{};
        f64 phase{};
    };

    // Latest request for one actuator
    struct Target {
        std::atomic<f32> low_amplitude{};
        std::atomic<f32> low_frequency{160.0f};
        std::atomic<f32> high_amplitude{};
        std::atomic<f32> high_frequency{320.0f};
    };

    // Only touched from the audio callback
    struct Actuator {
        Band low{.frequency = 160.0f};
        Band high{.frequency = 320.0f};
        Band low_target{.frequency = 160.0f};
        Band high_target{.frequency = 320.0f};
    };

    static void SDLCALL AudioCallback(void* userdata, SDL_AudioStream* stream,
                                      int additional_amount, int total_amount);
    static void Advance(Band& band, const Band& target);
    void Render(int frames);
    void UpdateTargets();

    SDL_AudioStream* stream{};
    SDL_AudioDeviceID device{};

    std::array<Target, 2> targets;
    std::atomic<u64> last_update_ms{};

    std::array<Actuator, 2> actuators;
    int frames_until_update{};
    std::vector<f32> buffer;
};

} // namespace InputCommon
