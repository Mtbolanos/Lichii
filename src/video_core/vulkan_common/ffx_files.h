// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD's FidelityFX runtime (FSR 4), as signed Windows DLLs from AMD's FidelityFX SDK release on
// GitHub. Nothing of AMD's ships with the emulator: the user downloads the files into the user
// folder's "fidelityfx" folder from the Graphics settings. Before loading, every file must match
// the pinned size and SHA-256 and carry AMD's Authenticode signature.

#pragma once

#include <array>
#include <filesystem>
#include <string_view>

#include "common/common_types.h"

namespace Vulkan::FfxRuntime {

/// The FidelityFX SDK release the files come from (FSR Upscaling 4.1.2).
inline constexpr std::string_view SDK_VERSION = "2.4.0";
inline constexpr std::string_view DOWNLOAD_HOST = "https://raw.githubusercontent.com";
/// The SDK repository at the release's commit (tag v2.4.0).
inline constexpr std::string_view DOWNLOAD_PATH =
    "/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/7539ae8b6568ad29ed78a753404351daf3b5cca7/";
inline constexpr std::string_view RELEASE_PAGE =
    "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.4.0";
/// AMD's licence for the signed DLLs, saved next to them.
inline constexpr std::string_view LICENSE_ENTRY = "Kits/FidelityFX/docs/license.md";
inline constexpr std::string_view LICENSE_FILE = "license.md";

struct File {
    std::string_view entry; ///< Path in the SDK repository.
    std::string_view name;
    u64 size;
    std::string_view sha256; ///< Lower-case hex.
};

/// The loader, which loads the upscaler next to it.
inline constexpr std::array FILES{
    File{"Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll", "amd_fidelityfx_loader_vk.dll",
         39184, "38a56a01fb3565c0757d9f5c1e4e69de4b65e5f2e5769b323b200dc1f142bab9"},
    File{"Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll",
         "amd_fidelityfx_upscaler_vk.dll", 36135696,
         "4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7"},
};

/// The user folder's "fidelityfx" folder.
[[nodiscard]] std::filesystem::path Folder();

/// Whether every file is in Folder() with its pinned size (not hashed).
[[nodiscard]] bool FilesPresent();

/// Whether a file carries a valid Authenticode signature by Advanced Micro Devices. Always false
/// outside Windows.
[[nodiscard]] bool IsSignedByAmd(const std::filesystem::path& path);

/// Whether a file has the pinned SHA-256. Always false outside Windows.
[[nodiscard]] bool HasSha256(const std::filesystem::path& path, std::string_view sha256);

} // namespace Vulkan::FfxRuntime
