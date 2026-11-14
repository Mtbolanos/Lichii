// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <boost/container/deque.hpp>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline_frame_gen.h"
#endif

namespace Core::Frontend {
class EmuWindow;
} // namespace Core::Frontend

namespace Vulkan {

class Device;
class Scheduler;
class Swapchain;

struct Frame {
    u32 width;
    u32 height;
    u32 index;
    // Requested presentation view format and underlying image format may differ for SDR.
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkFormat image_format{VK_FORMAT_UNDEFINED};
    VkColorSpaceKHR color_space{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    vk::Image image;
    vk::ImageView image_view;
    vk::ImageView storage_view;
    vk::Framebuffer framebuffer;
    vk::CommandBuffer cmdbuf;
    vk::Semaphore render_ready;
    vk::Fence present_done;
    bool storage_capable{};
    /// Time the present thread leaves before presenting the next frame. Frame generation sets it
    /// on generated frames, so they share the screen time evenly with the real ones.
    std::chrono::nanoseconds present_spacing{};
#ifdef EGLANTINE_STREAMLINE
    // DLSS Frame Generation: the guest frame presented and where the guest image is in the frame.
    bool frame_gen{};
    Streamline::FrameGenFrame frame_gen_frame{};
    VkRect2D guest_rect{};
#endif
};

class PresentManager {
public:
    PresentManager(const vk::Instance& instance,
                   Core::Frontend::EmuWindow& render_window,
                   const Device& device,
                   MemoryAllocator& memory_allocator,
                   Scheduler& scheduler,
                   Swapchain& swapchain,
                   vk::SurfaceKHR& surface);
    ~PresentManager();

    /// Returns the last used presentation frame
    Frame* GetRenderFrame();

    /// The time GetRenderFrame calls waited for a frame since the last call of this (the
    /// present thread holding the renderer back).
    [[nodiscard]] std::chrono::steady_clock::duration TakeBlockedTime() {
        return std::chrono::steady_clock::duration{
            blocked_time.exchange(0, std::memory_order_relaxed)};
    }

    /// Pushes a frame for presentation
    void Present(Frame* frame);

    /// Recreates the present frame to match the provided parameters
    void RecreateFrame(Frame* frame, u32 width, u32 height, VkFormat image_view_format,
                       VkRenderPass rd, bool storage);

    [[nodiscard]] bool NeedsStorage(const Frame* frame, bool required) const;

    /// Selects the backing image format, preserving the cached SDR format or explicit HDR format.
    [[nodiscard]] VkFormat GetFrameImageFormat(VkFormat image_view_format) const;

    /// Waits for the present thread to finish presenting all queued frames.
    void WaitPresent();

    /// How many additional frames can be queued without stalling the render thread
    [[nodiscard]] size_t MaxExtraFrames() const;

    [[nodiscard]] std::size_t SwapchainImageCount() const { return swapchain_image_count; }
    [[nodiscard]] VkFormat SwapchainImageFormat() const { return swapchain_image_format; }

private:
    enum class TransferMode { Unsupported, Copy, Blit };

    void StartPresentThread();

    void PresentThread(std::stop_token token);

    void CopyToSwapchain(Frame* frame);

    void CopyToSwapchainImpl(Frame* frame);

    // Returns false after discarding the frame if creation or transfer cannot proceed.
    bool RecreateSwapchain(Frame* frame);

    /// Caches a copy, blit, or unsupported transfer decision for this frame and active swapchain.
    void ConfigureTransfer(const Frame* frame);

    void DiscardFrame(Frame* frame);

    /// Sends the HDR10 metadata of the monitor's current state to an HDR10 swapchain.
    void UpdateHdrMetadata();

    void SetImageCount();

    /// With EDEN_DKB_FRAME_TRACE=1 and frame generation on, logs the presents of every 10 s.
    void LogPacing(const Frame* frame, std::chrono::steady_clock::duration took,
                   std::chrono::steady_clock::time_point now);

private:
    const vk::Instance& instance;
    Core::Frontend::EmuWindow& render_window;
    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    Swapchain& swapchain;
    vk::SurfaceKHR& surface;
    vk::CommandPool cmdpool;
    std::vector<Frame> frames;
    boost::container::deque<Frame*> present_queue;
    boost::container::deque<Frame*> free_queue;
    std::condition_variable_any frame_cv;
    std::condition_variable free_cv;
    std::mutex swapchain_mutex;
    std::mutex queue_mutex;
    std::mutex free_mutex;
    std::jthread present_thread;
    VkFormat transfer_source_format{VK_FORMAT_UNDEFINED};
    TransferMode transfer_mode{TransferMode::Unsupported};
    u64 hdr_metadata_generation{}; ///< DisplayHdrMonitor generation of the metadata sent
    bool storage_supported;
    bool use_present_thread;
    std::atomic<std::size_t> image_count{};
    std::atomic<std::size_t> swapchain_image_count{};
    std::atomic<VkFormat> swapchain_image_format{};
    /// The present thread presents no frame before this (Frame::present_spacing).
    std::chrono::steady_clock::time_point next_present{};
    struct {
        std::chrono::steady_clock::time_point since{};
        u64 real{};
        u64 generated{};
        std::chrono::nanoseconds spacing{};
        std::chrono::steady_clock::duration present_time{};
    } pacing_log;
    /// See TakeBlockedTime. Atomic: a composite can also run from a syncpoint action on
    /// another thread (GPU::RequestComposite).
    std::atomic<std::chrono::steady_clock::rep> blocked_time{};
};

} // namespace Vulkan
