// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

// The vendored FFX API headers declare their entry points with MSVC's __declspec; the runtime is
// only called through pointers.
#ifndef _WIN32
#define __declspec(x)
#endif
#include "api/include/ffx_api.h"
#ifndef _WIN32
#undef __declspec
#endif

#include "video_core/vulkan_common/ffx_files.h"

namespace Vulkan::FfxRuntime {

/// The FFX API entry points of AMD's loader.
struct Api {
    PfnFfxCreateContext create_context{};
    PfnFfxDestroyContext destroy_context{};
    PfnFfxQuery query{};
    PfnFfxDispatch dispatch{};
};

/// Loads the runtime after checking every file, or null when a file is missing, doesn't match
/// or the loader lacks its exports (the log says which), and always outside Windows. Files that
/// passed are not checked again this session; the runtime stays loaded while a returned pointer
/// lives.
[[nodiscard]] std::shared_ptr<const Api> Acquire();

} // namespace Vulkan::FfxRuntime
