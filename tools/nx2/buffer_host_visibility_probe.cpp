// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_vulkan/vk_buffer_cache.cpp"

extern "C" __attribute__((visibility("default"))) void
BufferHostReadProbe(VkCommandBuffer command, PFN_vkCmdPipelineBarrier barrier) {
  Vulkan::vk::DeviceDispatch dispatch{};
  dispatch.vkCmdPipelineBarrier = barrier;
  Vulkan::HostReadBarrier(Vulkan::vk::CommandBuffer{command, dispatch});
}

extern "C" __attribute__((visibility("default"))) bool
CheckBufferHostReadProbe() {
  static bool legacy_ok;
  static bool sync2_ok;
  legacy_ok = false;
  sync2_ok = false;
  Vulkan::vk::DeviceDispatch dispatch{};
  dispatch.vkCmdPipelineBarrier =
      [](VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
         VkDependencyFlags flags, uint32_t count,
         const VkMemoryBarrier *barriers, uint32_t buffers,
         const VkBufferMemoryBarrier *, uint32_t images,
         const VkImageMemoryBarrier *) {
        legacy_ok = src == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT &&
                    dst == VK_PIPELINE_STAGE_HOST_BIT && flags == 0 &&
                    count == 1 && buffers == 0 && images == 0 &&
                    barriers[0].srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT &&
                    barriers[0].dstAccessMask == VK_ACCESS_HOST_READ_BIT;
      };
  Vulkan::HostReadBarrier(Vulkan::vk::CommandBuffer{VK_NULL_HANDLE, dispatch});
  dispatch.vkCmdPipelineBarrier2 = [](VkCommandBuffer,
                                      const VkDependencyInfo *info) {
    sync2_ok =
        info->dependencyFlags == 0 && info->memoryBarrierCount == 1 &&
        info->bufferMemoryBarrierCount == 0 &&
        info->imageMemoryBarrierCount == 0 &&
        info->pMemoryBarriers[0].srcStageMask ==
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
        info->pMemoryBarriers[0].dstStageMask == VK_PIPELINE_STAGE_2_HOST_BIT &&
        info->pMemoryBarriers[0].srcAccessMask ==
            VK_ACCESS_2_MEMORY_WRITE_BIT &&
        info->pMemoryBarriers[0].dstAccessMask == VK_ACCESS_2_HOST_READ_BIT;
  };
  Vulkan::HostReadBarrier(Vulkan::vk::CommandBuffer{VK_NULL_HANDLE, dispatch});
  return legacy_ok && sync2_ok;
}
