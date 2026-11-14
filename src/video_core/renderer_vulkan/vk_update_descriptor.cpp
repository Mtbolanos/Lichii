// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <stdexcept>
#include <variant>
#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/descriptor_payload_growth.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

UpdateDescriptorQueue::UpdateDescriptorQueue(const Device& device_, size_t frame_payload_size_,
                                             bool supports_descriptor_buffer_)
    : device{device_}, frame_payload_size{frame_payload_size_},
      maximum_frame_payload_size{frame_payload_size_ * 4},
      supports_descriptor_buffer{supports_descriptor_buffer_},
      payload(std::make_unique_for_overwrite<DescriptorUpdateEntry[]>(frame_payload_size_ * FRAMES_IN_FLIGHT))
{
    payload_start = payload.get();
    payload_cursor = payload.get();
}

UpdateDescriptorQueue::~UpdateDescriptorQueue() = default;

void UpdateDescriptorQueue::TickFrame() {
    if (++frame_index >= FRAMES_IN_FLIGHT) {
        frame_index = 0;
    }
    payload_start = payload.get() + frame_index * frame_payload_size;
    payload_cursor = payload_start;
}

/// Reserve one batch, growing drained host storage when possible, otherwise reusing it safely.
void UpdateDescriptorQueue::Acquire(Scheduler& scheduler, size_t required_entries,
                                    bool use_descriptor_buffer_) {
    use_descriptor_buffer = supports_descriptor_buffer && use_descriptor_buffer_;
    static constexpr size_t DEFAULT_REQUIRED_ENTRIES = 0x400;
    const size_t reserve = required_entries > 0 ? required_entries : DEFAULT_REQUIRED_ENTRIES;
    if (reserve > maximum_frame_payload_size) {
        throw std::length_error("Descriptor reservation exceeds the bounded staging capacity");
    }
    const size_t used = static_cast<size_t>(std::distance(payload_start, payload_cursor));
    if (reserve > frame_payload_size || used > frame_payload_size - reserve) {
        const size_t old_capacity = frame_payload_size;
        auto result = PayloadGrowthResult::AtLimit;
        if (!growth_disabled) {
            result = GrowDescriptorPayload(
                payload, frame_payload_size, maximum_frame_payload_size, FRAMES_IN_FLIGHT,
                (std::min)(used + reserve, maximum_frame_payload_size),
                [&] { scheduler.WaitWorker(); },
                [](size_t count) {
                    return std::make_unique_for_overwrite<DescriptorUpdateEntry[]>(count);
                });
        }
        if (result == PayloadGrowthResult::Grown) {
            LOG_INFO(Render_Vulkan,
                     "Descriptor payload grew: {} -> {} entries/frame, host storage {} MiB, used={}, reserve={}",
                     old_capacity, frame_payload_size,
                     frame_payload_size * FRAMES_IN_FLIGHT * sizeof(DescriptorUpdateEntry) / 1048576,
                     used, reserve);
        } else {
            growth_disabled |= result == PayloadGrowthResult::AllocationFailed;
            if (reserve > frame_payload_size) {
                throw std::bad_alloc{};
            }
            if (!fallback_reported) {
                LOG_WARNING(Render_Vulkan,
                            "Descriptor payload growth {} at {} entries/frame; using worker drain fallback",
                            growth_disabled ? "allocation failed" : "limit reached", frame_payload_size);
                fallback_reported = true;
            }
            scheduler.WaitWorker();
        }
        payload_start = payload.get() + frame_index * frame_payload_size;
        payload_cursor = payload_start;
    }
    upload_start = payload_cursor;
}

} // namespace Vulkan
