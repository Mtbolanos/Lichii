# FidelityFX SDK (subset)

The FSR 3.1 upscaler's GPU code from AMD's FidelityFX SDK, used in place of a game's DLSS
evaluation (`src/video_core/renderer_vulkan/vk_fsr3_upscaler.cpp`).

- Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.
- Licence: MIT, `LICENSE.txt` (v1.1.4's `sdk/LICENSE.txt`). v2.4.0's `docs/license.md` lists every
  file taken from it below under the same MIT terms.
- `Kits/FidelityFX/`: from tag `v2.4.0` (commit `7539ae8b6568ad29ed78a753404351daf3b5cca7`),
  FSR 3.1.6 upscaler, at the SDK's own paths so its relative includes resolve:
  - `api/internal/gpu/`: the core headers (not the CPU and HLSL ones).
  - `upscalers/fsr3/include/gpu/`: `fsr3upscaler/` without the HLSL callbacks and the debug view,
    `spd/ffx_spd.h`, `fsr1/ffx_fsr1.h`.
- From tag `v1.1.4` (commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`): the GLSL code, which
  SDK 2.x no longer ships (its Vulkan backend compiles the HLSL passes).
  - `include/FidelityFX/gpu/`: `ffx_core_glsl.h` and `fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h`
    from `sdk/include/FidelityFX/gpu/`.
  - `shaders/fsr3upscaler/`: the Vulkan pass files from `sdk/src/backends/vk/shaders/fsr3upscaler/`
    (the eight upscaling passes; not the autogen-reactive or debug-view pass).

  Between 3.1.4 and 3.1.6 the HLSL callbacks only changed their register macros, and the core
  and upscaler headers kept the same functions, constant buffers and resources, so this glue
  matches the 3.1.6 headers.

The files are unmodified. They are compiled to SPIR-V by `src/video_core/host_shaders`; no
prebuilt binary or AMD tool from the SDK is used.
