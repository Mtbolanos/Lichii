// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS Frame Generation's per-frame work. Everything here does nothing unless
// FrameGenSession(). One frame index runs through all of it: the upscaler's constants and tags,
// the Reflex and PCL markers and the frame Composite presents.

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Streamline {

/// The guest frame being rendered.
[[nodiscard]] u32 FrameGenIndex();

/// Begins the current frame unless that was done: Reflex sleep, then the simulation and
/// render-submit start markers.
void FrameGenBeginFrame();

struct FrameGenFrame {
    u32 index{};
    bool inputs{}; ///< DLSS-G has depth, motion vectors and camera constants for it
};

/// Ends the frame Composite is about to present. A frame the upscaler gave no constants (menus,
/// movies and loading run without DLSS) gets the last ones with reset set, and a frame without
/// DLSS-G inputs gets its depth and motion tags cleared, so interpolation pauses.
[[nodiscard]] FrameGenFrame FrameGenEndFrame();

/// Moves on to the next guest frame and begins it.
void FrameGenNextFrame();

struct FrameGenPresent {
    FrameGenFrame frame;
    VkRect2D guest_rect;         ///< Where the guest image is in the frame
    VkExtent2D frame_extent;     ///< The frame, which is scaled to the swapchain image
    VkExtent2D swapchain_extent;
    VkFormat swapchain_format;
    u32 swapchain_images;
};

/// Present thread, right before the frame's vkQueuePresentKHR: turns DLSS-G on or off, tags
/// where the guest image is in the swapchain image and sets the present start marker.
void FrameGenBeforePresent(const FrameGenPresent& present);

/// Present thread, right after: the present end marker, and DLSS-G's state for the log.
void FrameGenAfterPresent(u32 index);

/// Present thread, once the frame is presented or dropped: sets the present markers
/// FrameGenBeforePresent and FrameGenAfterPresent did not, so every frame's token is closed.
void FrameGenClosePresent(u32 index);

/// DLSS-G presents SDR and HDR10 swapchains; NVIDIA's DLSS-G guide rules out FP16 scRGB.
[[nodiscard]] bool FrameGenSupportsSwapchain(VkFormat format);

/// numFramesToGenerateMax: frames this system can generate per rendered frame, 0 while unknown.
[[nodiscard]] u32 FrameGenMaxGenerated();

/// Frames DLSS-G generates per rendered frame: the frame generation multiplier less one, capped at
/// FrameGenMaxGenerated() (1 while that is unknown).
[[nodiscard]] u32 FrameGenGenerated();

/// Turns DLSS-G off before the swapchain is recreated or destroyed (present thread).
void FrameGenSuspend();

struct FrameGenRates {
    f64 real_fps{};    ///< Frames the emulator presented
    f64 shown_fps{};   ///< Frames on screen, generated ones included
    bool generating{}; ///< DLSS-G is on and added frames at the last present
};

/// Any thread: the present rates over about the last second.
[[nodiscard]] FrameGenRates FrameGenRecentRates();

} // namespace Vulkan::Streamline
