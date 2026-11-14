// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "video_core/framebuffer_config.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// The colour encoding of a layer or a frame, as vulkan_hdr_convert.frag numbers them.
enum class HdrEncoding : u32 {
    Srgb = 0,  ///< SDR: sRGB BT.709
    Pq = 1,    ///< HDR10: PQ BT.2020
    Scrgb = 2, ///< linear BT.709, 80 nits per 1.0 on output
};

/// What a frame is presented as, and the luminances the game was told.
struct HdrPresent {
    VkColorSpaceKHR color_space{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    f32 reference_white{203.0f};
    f32 peak{1000.0f};

    [[nodiscard]] bool IsHdr() const {
        return color_space != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
};

struct HdrConvertParams {
    HdrEncoding input{};
    HdrEncoding output{};
    f32 reference_white{};
    f32 peak{};
    f32 nits_per_unit{};
    bool premultiplied{}; ///< colour is premultiplied by alpha (kept so across the conversion)
};

[[nodiscard]] HdrEncoding EncodingOf(Tegra::ColorSpace color_space);
[[nodiscard]] HdrEncoding EncodingOf(VkColorSpaceKHR color_space);

/// The conversion of a layer with framebuffer's encoding into a frame presented as output.
[[nodiscard]] HdrConvertParams MakeHdrConvertParams(const Tegra::FramebufferConfig& framebuffer,
                                                    const HdrPresent& output);

/// False when the layer's values already are what the frame needs.
[[nodiscard]] bool NeedsHdrConvert(const HdrConvertParams& params);

/// vulkan_hdr_convert.frag drawing into framebuffers of one format (Eglantine #557's passes, one
/// shader).
class HdrConvertPass {
public:
    explicit HdrConvertPass(const Device& device, size_t image_count, VkFormat output_format);
    ~HdrConvertPass();

    /// Every attachment is overwritten, so its old contents and layout are discarded.
    [[nodiscard]] VkRenderPass RenderPass() const {
        return *m_renderpass;
    }

    /// Draws input converted into framebuffer, an output-format attachment of output_image.
    void Record(const Device& device, Scheduler& scheduler, size_t image_index,
                const HdrConvertParams& params, VkImage input_image, VkImageView input_view,
                VkImage output_image, VkFramebuffer framebuffer, VkExtent2D extent);

private:
    vk::ShaderModule m_vertex_shader{};
    vk::ShaderModule m_fragment_shader{};
    vk::DescriptorPool m_descriptor_pool{};
    vk::DescriptorSetLayout m_descriptor_set_layout{};
    std::vector<vk::DescriptorSets> m_descriptor_sets{};
    vk::PipelineLayout m_pipeline_layout{};
    vk::RenderPass m_renderpass{};
    vk::Pipeline m_pipeline{};
    vk::Sampler m_sampler{};
};

/// Converts a layer between SDR, HDR10 and scRGB before it is drawn into the frame: HDR layers
/// on an SDR frame or a screenshot are tone mapped, SDR layers on an HDR frame are put at the
/// reference white.
class HdrConvert {
public:
    explicit HdrConvert(const Device& device, MemoryAllocator& allocator, size_t image_count,
                        VkExtent2D extent);
    ~HdrConvert();

    void Draw(const Device& device, Scheduler& scheduler, size_t image_index,
              const HdrConvertParams& params, VkImage* inout_image, VkImageView* inout_image_view);

    [[nodiscard]] VkExtent2D Extent() const {
        return m_extent;
    }

private:
    struct Image {
        vk::Image image{};
        vk::ImageView image_view{};
        vk::Framebuffer framebuffer{};
    };
    HdrConvertPass m_pass;
    std::vector<Image> m_images{};
    const VkExtent2D m_extent;
};

} // namespace Vulkan
