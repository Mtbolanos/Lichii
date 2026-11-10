# AMD FidelityFX API headers

The FFX API headers from AMD's FidelityFX SDK 2.4.0, used by
`src/video_core/renderer_vulkan/vk_fsr4_upscaler.cpp` to call AMD's FSR 4 runtime.

- Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `v2.4.0`
  (commit `7539ae8b6568ad29ed78a753404351daf3b5cca7`).
- Files, unmodified: `Kits/FidelityFX/api/include/ffx_api.h`, `ffx_api_types.h`,
  `vk/ffx_api_vk.h` and `Kits/FidelityFX/upscalers/include/ffx_upscale.h`.
- Licence: MIT (the SDK's `Kits/FidelityFX/docs/license.md` lists these files under its MIT
  terms; the notice is in each header).

AMD's signed runtime DLLs are not part of this repository or of any Lichii package. Users
download them from the same SDK commit in the Graphics settings; see
`src/video_core/vulkan_common/ffx_files.h`.
