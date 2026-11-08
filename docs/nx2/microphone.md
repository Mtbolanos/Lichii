# Microphone input

Configure -> Controls -> Microphone selects the host input device, adjusts input gain
(0–200%, 100% is unchanged PCM), and provides a start/stop level test. The test uses
the same audio backend and gain conversion as capture, works while emulation is
stopped or paused, and closes its recording stream when the tab is hidden, settings
are applied, or the dialog closes. Input-device and gain changes require emulation to stop.
The existing input-device setting is retained; per-game Audio overrides still apply.

Switch 2 clients can obtain `IAudioInManager` from `aud:u` command 0. This follows the
[documented Switch 2 audio interface](https://switchbrew.org/wiki/Audio_services#(S2)_aud:u).
Capture uses the selected host input rather than the playback device, and does not
inherit a surround playback channel count. Captured stereo is padded with silence
for any additional requested guest channels. Guest capture buffers are zero-filled on underrun.

When the host microphone cannot be opened or started, capture falls back to silence
and logs one warning (`host microphone unavailable, capturing silence`). The guest still
gets success and receives silent frames at 48 kHz. Console hardware always has a
microphone, so games do not handle a failed open: Ocarina read a null pointer 1 ms after
`OpenStream` returned an error, when cubeb could not open a Linux input device.

Ocarina uses the newer `nn::audio::InputStream` interface through `auddmg:u` and
`audsmx:u`, rather than the legacy `IAudioInManager`. The reproduced freeze was a
guest panic after the output-only format check rejected input functionality `2`.
The game then called `PurgeInputStream` on its failed, uninitialised stream.

This path now exposes an input virtual device and creates capture streams from the
selected host microphone. Input streams support S16/F32, mono/stereo, at 48 kHz.
The guest owns the consumer at ring offset `0x44`; capture publishes the producer
at `0x1004`, after writing PCM at `0x1040`. These offsets, input creation flags `2`,
mode-2 filled-frame notifications, and purge-zero semantics were verified against
the running SDK code. Tests cover fresh samples, mono downmix, wraparound, gain,
producer ownership, notifications, purge, allocation size, the IPC format query, and
the silent fallback when the host stream fails to open or start.
The user retested the revised build in Ocarina, confirmed the microphone test no
longer crashes, and confirmed working voice input. Other games and sample rates
have not been validated.

## Verification

- Select two different microphones and verify the meter responds to the selected
  device only. Test System default, Refresh devices, gain 0%, 100%, and 200%.
- Close the tab during a test and verify the operating system's recording indicator
  clears. Repeat with emulation paused, and test denied microphone permissions and
  disconnecting the active device.
- Apply the microphone choice, reopen Configure, and restart the emulator to verify
  the selection and gain persist. Check Cancel leaves saved settings unchanged.
- In Ocarina, enter the microphone test and check both input response and return from
  the screen. If it still fails, launch with `EDEN_NX2_IPC_TRACE=1` and save the log
  immediately after reproducing. Do not infer that the saved null-pointer panic was
  fixed solely because the missing `aud:u` interface is now implemented.

`src/tests/audio_core/microphone.cpp` covers PCM gain, clipping, meter level,
underrun padding, and stereo-to-multichannel expansion. `audio_in.cpp` covers invalid
device/sample-rate handling and starting an invalid session.

Validation completed on Windows with Clang and LLD:

- Qt frontend and test executable built successfully.
- 25 focused audio tests passed (3,785 assertions).
- Ocarina's in-game microphone test no longer panics; voice input was confirmed by
  the tester using a Logitech C920e microphone and the Cubeb backend.

Run the focused regression tests from a configured build with:

```sh
cmake --build build --target yuzu tests --parallel 12
build/bin/tests '*Microphone*,*Audio input*,[sysmix],[sysmix_sink]'
```

Input streams currently support 48 kHz S16/F32, mono/stereo. Other games and sample
rates have not been validated. If changing the Audio backend in the same Configure
dialog, apply that change and reopen Configure before selecting or testing its
microphone devices.
