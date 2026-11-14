// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/common_types.h"

#include "video_core/host_shaders/fxaa_vert_spv.h"
#include "video_core/host_shaders/vulkan_hdr_convert_frag_spv.h"
#include "video_core/renderer_vulkan/present/hdr_convert.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

constexpr VkFormat IntermediateFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr f32 ScrgbNits = 80.0f;

struct PushConstants {
    u32 input_encoding;
    u32 output_encoding;
    f32 reference_white;
    f32 peak;
    f32 nits_per_unit;
    u32 premultiplied;
};

} // Anonymous namespace

HdrEncoding EncodingOf(Tegra::ColorSpace color_space) {
    switch (color_space) {
    case Tegra::ColorSpace::RGB2020_PQ:
        return HdrEncoding::Pq;
    case Tegra::ColorSpace::RGB709_Linear:
        return HdrEncoding::Scrgb;
    case Tegra::ColorSpace::RGB709:
        break;
    }
    return HdrEncoding::Srgb;
}

HdrEncoding EncodingOf(VkColorSpaceKHR color_space) {
    switch (color_space) {
    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
        return HdrEncoding::Pq;
    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
        return HdrEncoding::Scrgb;
    default:
        return HdrEncoding::Srgb;
    }
}

HdrConvertParams MakeHdrConvertParams(const Tegra::FramebufferConfig& framebuffer,
                                      const HdrPresent& output) {
    return {
        .input = EncodingOf(framebuffer.color_space),
        .output = EncodingOf(output.color_space),
        .reference_white = output.reference_white,
        .peak = output.peak,
        .nits_per_unit = framebuffer.linear_unit_luminance != 0
                             ? static_cast<f32>(framebuffer.linear_unit_luminance)
                             : ScrgbNits,
        .premultiplied = framebuffer.blending == Tegra::BlendMode::Premultiplied,
    };
}

bool NeedsHdrConvert(const HdrConvertParams& params) {
    if (params.input != params.output) {
        return true;
    }
    return params.input == HdrEncoding::Scrgb && params.nits_per_unit != ScrgbNits;
}

HdrConvertPass::HdrConvertPass(const Device& device, size_t image_count, VkFormat output_format) {
    m_renderpass = CreateWrappedRenderPass(device, output_format, VK_IMAGE_LAYOUT_UNDEFINED);
    m_sampler = CreateWrappedSampler(device);
    m_vertex_shader = CreateWrappedShaderModule(device, FXAA_VERT_SPV);
    m_fragment_shader = CreateWrappedShaderModule(device, VULKAN_HDR_CONVERT_FRAG_SPV);
    // fxaa.vert reads the texture size from binding 0, the fragment shader samples binding 1.
    m_descriptor_pool = CreateWrappedDescriptorPool(device, 2 * image_count, image_count);
    m_descriptor_set_layout = CreateWrappedDescriptorSetLayout(
        device,
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
    for (size_t i = 0; i < image_count; ++i) {
        m_descriptor_sets.push_back(
            CreateWrappedDescriptorSets(m_descriptor_pool, {*m_descriptor_set_layout}));
    }
    const VkPushConstantRange push_constant_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    m_pipeline_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = m_descriptor_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    });
    m_pipeline = CreateWrappedPipeline(device, m_renderpass, m_pipeline_layout,
                                       std::tie(m_vertex_shader, m_fragment_shader));
}

HdrConvertPass::~HdrConvertPass() = default;

void HdrConvertPass::Record(const Device& device, Scheduler& scheduler, size_t image_index,
                            const HdrConvertParams& params, VkImage input_image,
                            VkImageView input_view, VkImage output_image,
                            VkFramebuffer framebuffer, VkExtent2D extent) {
    const VkDescriptorSet descriptor_set{m_descriptor_sets[image_index][0]};
    std::vector<VkDescriptorImageInfo> image_infos;
    image_infos.reserve(2);
    const std::array updates{
        CreateWriteDescriptorSet(image_infos, *m_sampler, input_view, descriptor_set, 0),
        CreateWriteDescriptorSet(image_infos, *m_sampler, input_view, descriptor_set, 1),
    };
    device.GetLogical().UpdateDescriptorSets(updates, {});

    const PushConstants push_constants{
        .input_encoding = static_cast<u32>(params.input),
        .output_encoding = static_cast<u32>(params.output),
        .reference_white = params.reference_white,
        .peak = params.peak,
        .nits_per_unit = params.nits_per_unit,
        .premultiplied = params.premultiplied ? 1U : 0U,
    };
    const VkRenderPass renderpass{*m_renderpass};
    const VkPipeline pipeline{*m_pipeline};
    const VkPipelineLayout layout{*m_pipeline_layout};

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        TransitionImageLayout(cmdbuf, input_image, VK_IMAGE_LAYOUT_GENERAL);
        BeginRenderPass(cmdbuf, renderpass, framebuffer, extent);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, descriptor_set, {});
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, push_constants);
        cmdbuf.Draw(3, 1, 0, 0);
        cmdbuf.EndRenderPass();
        TransitionImageLayout(cmdbuf, output_image, VK_IMAGE_LAYOUT_GENERAL);
    });
}

HdrConvert::HdrConvert(const Device& device, MemoryAllocator& allocator, size_t image_count,
                       VkExtent2D extent)
    : m_pass(device, image_count, IntermediateFormat), m_extent(extent) {
    for (size_t i = 0; i < image_count; ++i) {
        Image& image = m_images.emplace_back();
        image.image = CreateWrappedImage(allocator, m_extent, IntermediateFormat);
        image.image_view = CreateWrappedImageView(device, image.image, IntermediateFormat);
        image.framebuffer = device.GetLogical().CreateFramebuffer(VkFramebufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = m_pass.RenderPass(),
            .attachmentCount = 1,
            .pAttachments = image.image_view.address(),
            .width = m_extent.width,
            .height = m_extent.height,
            .layers = 1,
        });
    }
}

HdrConvert::~HdrConvert() = default;

void HdrConvert::Draw(const Device& device, Scheduler& scheduler, size_t image_index,
                      const HdrConvertParams& params, VkImage* inout_image,
                      VkImageView* inout_image_view) {
    const Image& image{m_images[image_index]};
    m_pass.Record(device, scheduler, image_index, params, *inout_image, *inout_image_view,
                  *image.image, *image.framebuffer, m_extent);
    *inout_image = *image.image;
    *inout_image_view = *image.image_view;
}

} // namespace Vulkan
