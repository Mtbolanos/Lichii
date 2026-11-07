// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/frame_gen/lossless_dll.h"
#include "video_core/renderer_vulkan/present/lsfg_shaders.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

LsfgShaders::LsfgShaders(const Device& device) {
    const bool wanted = Settings::values.frame_gen.GetValue();
    if (!device.IsVulkanMemoryModelSupported() || !device.HasNullDescriptor() ||
        !device.IsHostFloat16Supported()) {
        if (wanted) {
            LOG_WARNING(Render_Vulkan,
                        "LSFG unavailable: memory model {}, null descriptor {}, float16 {}",
                        device.IsVulkanMemoryModelSupported(), device.HasNullDescriptor(),
                        device.IsHostFloat16Supported());
        }
        return;
    }
    device_supported = true;

    VideoCore::FrameGen::ShaderModules code;
    const auto status = VideoCore::FrameGen::LoadShaderModules(code);
    if (status != VideoCore::FrameGen::LosslessStatus::Ok) {
        if (wanted) {
            LOG_WARNING(Render_Vulkan, "LSFG unavailable: Lossless.dll status {}",
                        static_cast<u32>(status));
        }
        return;
    }
    LOG_INFO(Render_Vulkan, "LSFG shaders loaded: {} modules", code.size());

    for (const auto& [id, words] : code) {
        modules.emplace(id, CreateWrappedShaderModule(device, words));
    }
    valid = true;
}

VkShaderModule LsfgShaders::Get(u32 shader_id) const {
    const auto hit = modules.find(shader_id);
    return hit == modules.end() ? VK_NULL_HANDLE : *hit->second;
}

} // namespace Vulkan
