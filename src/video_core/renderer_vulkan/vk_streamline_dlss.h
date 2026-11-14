// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS
// evaluation. NVIDIA's DLLs come from the user's PC; see vulkan_common/streamline.h.

#pragma once

#include <array>
#include <atomic>
#include <optional>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
struct DlssStatus;
}

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

class StreamlineDlssUpscaler {
public:
    struct Image {
        VkImageView view;
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The view's level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
    };

    struct Inputs {
        Image color;          ///< Jittered colour at the render size.
        Image depth;          ///< Device depth (reversed) at the render size.
        Image motion_vectors; ///< Motion to the previous frame at the render size.
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        /// This frame's sample offset in render pixels, as the game hands its own DLSS.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels (the render size for UV offsets).
        std::array<f32, 2> motion_vector_scale;
        bool reset;
        bool depth_inverted = true;
        bool hdr = true; ///< Linear HDR colour. False for tonemapped, perceptually encoded colour.
        /// The depth image's size when it is larger than the render size (the scene lies in its
        /// top left corner); zero for the render size.
        VkExtent2D depth_size{};
        /// The scene camera, for DLSS Frame Generation. Row-major matrices that take a column
        /// vector, unjittered, with OpenGL clip depth (-1 to 1) and clip y up: the
        /// view-projection and the motion vectors' clip-to-previous-clip. Without it the frame
        /// gets no frame generation.
        struct Camera {
            std::array<f32, 16> view_projection;
            std::array<f32, 16> clip_to_previous_clip;
        };
        std::optional<Camera> camera;
    };

    /// Throws vk::Exception when Streamline has no DLSS for this device.
    explicit StreamlineDlssUpscaler(const Device& device, Scheduler& scheduler,
                                    MemoryAllocator& memory_allocator);
    ~StreamlineDlssUpscaler();

    /// Records one evaluation into the output image (RGBA16F at the upscale size, GENERAL
    /// layout). Returns false when DLSS can't run, so the caller can fall back to FSR 3.1.
    bool Upscale(const Inputs& inputs);

    [[nodiscard]] bool Failed() const noexcept {
        return failed.load();
    }

    /// Evaluations that succeeded in this process, for the status bar.
    [[nodiscard]] static VideoCore::DlssStatus Status();

    [[nodiscard]] VkImage OutputImage() const noexcept {
        return *output_image;
    }
    [[nodiscard]] VkImageView OutputView() const noexcept {
        return *output_view;
    }
    /// Nearest-neighbour clamp sampler (normalized coordinates), for reading the output.
    [[nodiscard]] VkSampler PointSampler() const noexcept {
        return *point_sampler;
    }
    [[nodiscard]] VkSampler LinearSampler() const noexcept {
        return *linear_sampler;
    }

private:
    void CreateOutput(VkExtent2D upscale_size);

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    void* set_options{}; ///< slDLSSSetOptions
    void* get_state{};   ///< slDLSSGetState
    void* get_optimal_settings{};
    VkExtent2D render_extent{};
    VkExtent2D upscale_extent{};
    bool hdr_input = true;
    Settings::NvidiaDlssQuality quality_mode{};
    Settings::NvidiaDlssPreset model_preset{};
    Settings::NvidiaDlssQuality evaluated_mode{}; ///< Recorded on the scheduler's worker
    u32 frame_index{};
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Sampler point_sampler;
    vk::Sampler linear_sampler;
    std::atomic<bool> failed{};
    std::atomic<bool> vram_warned{};
};

} // namespace Vulkan
