// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <optional>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/present/frame_gen_pacer.h"
#include "video_core/renderer_vulkan/present/lsfg_chain.h"
#include "video_core/renderer_vulkan/present/lsfg_shaders.h"
#include "video_core/renderer_vulkan/present/fsr3_frame_gen.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

class Device;
class Scheduler;
struct Frame;

class FrameGen {
public:
    explicit FrameGen(MemoryAllocator& memory_allocator, Scheduler& scheduler);
    ~FrameGen();

    void Process(const Device& device, Frame* frame, VkFormat format, VkExtent2D guest_extent);

    /// blocked: the time the renderer waited for the present thread since the last call.
    [[nodiscard]] size_t WantedGenerations(size_t capacity,
                                           std::chrono::steady_clock::duration blocked = {});

    [[nodiscard]] size_t GeneratedFrameCount() const;

    /// The measured time between real frames in seconds, or 0 while it is unknown.
    [[nodiscard]] f32 FrameInterval() const {
        return pacer.SmoothedInterval();
    }

    void GenerateInto(const Device& device, Frame* destination, size_t generation);

    [[nodiscard]] VideoCore::FrameGenStatus CurrentStatus() const {
        return {.state = state.load(std::memory_order_relaxed),
                .generated_frames = generated_frames.load(std::memory_order_relaxed)};
    }

private:
    void Rebuild(const Device& device, VkExtent2D extent, VkFormat format, f32 flow_scale,
                 Settings::FrameGenType type);
    /// Updates the status bar state and logs each change with its reason.
    void SetState(VideoCore::FrameGenState new_state, VkFormat format);
    /// Logs every 10 s how many frames were generated, when that changes.
    void LogActivity();
    void DumpDebugImages(u64 count);

    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    std::optional<LsfgShaders> shaders;
    std::optional<LsfgChain> chain;
    std::optional<Fsr3FrameGen> fsr3_chain;
    FrameGenPacer pacer;
    FrameGenPlan plan{};
    VkExtent2D peak_guest_extent{};
    VkExtent2D built_extent{};
    VkFormat built_format{VK_FORMAT_UNDEFINED};
    Settings::FrameGenType built_type{};
    f32 built_flow_scale{};
    u64 frame_count{};
    u64 last_count{};
    size_t last_generations{};
    u32 warm_streak{};
    bool generated{};
    bool unavailable{};
    bool dumped{};
    std::atomic<VideoCore::FrameGenState> state{VideoCore::FrameGenState::Pending};
    std::atomic<u64> generated_frames{};
    struct {
        std::chrono::steady_clock::time_point since{};
        u64 frames{};
        u64 generated{};
        f32 logged_per_frame{-1.0f};
    } activity;
};

} // namespace Vulkan
