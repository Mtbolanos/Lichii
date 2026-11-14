// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Streamline's API as loaded from the user's sl.interposer.dll. Only for the
// Streamline sources: it pulls in <windows.h> and Streamline's (MIT-licensed) headers.

#pragma once

#include <mutex>

#include <windows.h>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

#include <sl.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_helpers_vk.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

namespace Vulkan::Streamline {

struct Api {
    PFun_slInit* init{};
    PFun_slShutdown* shutdown{};
    PFun_slIsFeatureSupported* is_feature_supported{};
    PFun_slGetFeatureRequirements* get_feature_requirements{};
    PFun_slGetFeatureVersion* get_feature_version{};
    PFun_slGetFeatureFunction* get_feature_function{};
    PFun_slGetNewFrameToken* get_new_frame_token{};
    PFun_slSetConstants* set_constants{};
    PFun_slEvaluateFeature* evaluate_feature{};
    PFun_slFreeResources* free_resources{};
    PFun_slSetVulkanInfo* set_vulkan_info{};
    PFun_slSetTagForFrame* set_tag_for_frame{};
    PFun_slSetFeatureLoaded* set_feature_loaded{};

    // Frame generation, set only while FrameGenSession().
    PFun_slDLSSGSetOptions* dlssg_set_options{};
    PFun_slDLSSGGetState* dlssg_get_state{};
    PFun_slReflexSetOptions* reflex_set_options{};
    PFun_slReflexSleep* reflex_sleep{};
    PFun_slReflexGetState* reflex_get_state{};
    PFun_slPCLSetMarker* pcl_set_marker{};
};

/// The loaded functions; only valid while IsDlssAvailable().
[[nodiscard]] const Api& GetApi();

/// Serialises the Streamline calls of the renderer, the Vulkan worker and the present thread:
/// slSetFeatureLoaded and slEvaluateFeature aren't thread safe, and a frame token is a slot
/// Streamline hands to another frame index a few calls later, so hold this from getting a token
/// until done with it.
[[nodiscard]] std::unique_lock<std::mutex> LockStreamline();

/// Forgets the frame generation state of an earlier session.
void ResetFrameGen();

/// Reads numFramesToGenerateMax once DLSS-G is attached.
void QueryFrameGenMax();

/// For the upscaler while FrameGenSession(): whether it sets this frame's constants, true the
/// first time only (DLSS-G rejects a second set). Frames that get none reuse the last ones.
[[nodiscard]] bool ClaimFrameConstants(u32 index, const sl::Constants& constants);

/// slSetConstants with a frame's claimed constants; call with LockStreamline held.
[[nodiscard]] sl::Result SetFrameConstants(u32 index, const sl::FrameToken& token);

/// The upscaler tagged depth and motion vectors for DLSS-G, with real camera constants.
void NoteFrameGenInputs(u32 index, VkExtent2D extent, VkFormat depth_format,
                        VkFormat motion_format);

} // namespace Vulkan::Streamline
