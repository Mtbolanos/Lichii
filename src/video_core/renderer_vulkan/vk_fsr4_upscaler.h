// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR 4 through AMD's FidelityFX runtime (vulkan_common/ffx_runtime.h), which the user
// downloads from AMD in the Graphics settings. Same inputs as the Ocarina FSR 3.1 path, which takes
// over when FSR 4 is off, missing or fails.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

namespace FfxRuntime {
struct Api;
}

class Fsr4Upscaler {
public:
    struct Image {
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The used level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
    };

    struct Inputs {
        Image color;          ///< Tonemapped (LDR) colour at the render size.
        Image depth;          ///< Standard (not reversed) device depth at the render size.
        Image motion_vectors; ///< Motion to the previous frame, in render pixels.
        std::optional<Image> reactive;
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        f32 frame_time_ms;
        bool reset;
        bool sharpen;
        f32 sharpness; ///< 0-1
        f32 vertical_fov;
        f32 camera_near = 0.1f;
        f32 camera_far = 25000.0f;
        std::array<f32, 2> jitter{}; ///< Offset the scene was drawn with, in render pixels.
    };

    /// Whether FSR 4 is turned on and the device runs on AMD's driver.
    [[nodiscard]] static bool IsWanted(const Device& device);

    /// Throws vk::Exception when FSR 4 can't start on this device: AMD's files are missing or
    /// don't check out, or AMD's runtime offers no FSR 4 for the GPU.
    explicit Fsr4Upscaler(const Device& device, Scheduler& scheduler,
                          MemoryAllocator& memory_allocator);
    ~Fsr4Upscaler();

    /// Records one evaluation into the output image (RGBA16F at the upscale size, GENERAL
    /// layout). False when FSR 4 can't run (the caller then uses FSR 3.1 from now on).
    bool Upscale(const Inputs& inputs);

    [[nodiscard]] bool Failed() const noexcept {
        return failed.load();
    }
    [[nodiscard]] VkImage OutputImage() const noexcept {
        return *output_image;
    }
    [[nodiscard]] VkImageView OutputView() const noexcept {
        return *output_view;
    }

private:
    bool CreateContext(VkExtent2D render_size, VkExtent2D upscale_size);
    void DestroyContext();
    void CreateOutput(VkExtent2D upscale_size);
    void CreateExposure();

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    std::shared_ptr<const FfxRuntime::Api> api;
    u64 provider_id{};
    std::string provider_name;
    void* context{}; ///< ffxContext
    VkExtent2D render_extent{};
    VkExtent2D upscale_extent{};
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Image exposure_image; ///< 1x1 R32F holding 1 (the input is tonemapped, unit exposure)
    std::atomic<bool> failed{};
};

} // namespace Vulkan
