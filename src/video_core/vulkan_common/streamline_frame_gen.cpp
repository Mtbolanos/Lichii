// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#include "video_core/vulkan_common/streamline_frame_gen.h"
#include "video_core/vulkan_common/vk_enum_string_helper.h"

#include <sl_helpers.h>

namespace Vulkan::Streamline {
namespace {

// Frames in a row with or without DLSS-G inputs before DLSS-G turns on or off.
constexpr u32 FRAMES_TO_TURN_ON = 3;
constexpr u32 FRAMES_TO_TURN_OFF = 3;
// After DLSS-G reported a failure, how long it stays off before it is tried again.
constexpr u32 RETRY_FRAMES = 600;
constexpr auto REPORT_INTERVAL = std::chrono::seconds{5};
constexpr auto RATES_INTERVAL = std::chrono::seconds{1};
constexpr u32 NO_FRAME = std::numeric_limits<u32>::max();
// Frames rendered ahead of the one presented whose constants are kept for its present.
constexpr size_t KEPT_CONSTANTS = 16;

const sl::ViewportHandle VIEWPORT{0u};

struct FrameState {
    std::atomic<u32> index{};
    std::atomic<u32> begun{NO_FRAME};
    std::atomic<u32> constants_index{NO_FRAME};
    std::atomic<u32> inputs_index{NO_FRAME};
    std::atomic<u32> input_width{};
    std::atomic<u32> input_height{};
    std::atomic<VkFormat> depth_format{VK_FORMAT_UNDEFINED};
    std::atomic<VkFormat> motion_format{VK_FORMAT_UNDEFINED};
    std::mutex token_mutex;
    std::mutex constants_mutex;
    std::optional<sl::Constants> last_constants;
    struct FrameConstants {
        u32 index = NO_FRAME;
        sl::Constants constants;
    };
    std::array<FrameConstants, KEPT_CONSTANTS> frame_constants;
};

// Only the present thread uses this (the renderer's thread without one), and the present
// manager's destructor once that thread is gone.
struct PresentState {
    bool on{};
    u32 with_inputs{};
    u32 without_inputs{};
    u32 retry_in{};
    std::optional<sl::DLSSGStatus> status;
    bool state_failed{};
    sl::DLSSGOptions options{};
    bool clamp_logged{};
    u32 open_present{NO_FRAME};   ///< PresentStart set, PresentEnd not yet
    u32 closed_present{NO_FRAME}; ///< Last frame whose PresentEnd was set
    u64 vram{};
    u64 real{};
    u64 shown{};
    u32 last_shown{};
    std::chrono::steady_clock::time_point report_from;
    u32 rates_real{};
    u32 rates_shown{};
    std::optional<std::chrono::steady_clock::time_point> rates_from;
};

// Written by the present thread, read by the frontend.
struct Rates {
    std::atomic<f64> real_fps{};
    std::atomic<f64> shown_fps{};
    std::atomic<bool> generating{};
    std::atomic<u32> max_generated{}; ///< numFramesToGenerateMax, 0 while unknown
};

std::atomic<int> g_api_error{};

FrameState& Frames() {
    static FrameState state;
    return state;
}

PresentState& Presents() {
    static PresentState state;
    return state;
}

Rates& PresentRates() {
    static Rates rates;
    return rates;
}

std::string ResultName(sl::Result result) {
    return fmt::format("{} ({})", sl::getResultAsStr(result), static_cast<int>(result));
}

std::string StatusName(sl::DLSSGStatus status) {
    if (status == sl::DLSSGStatus::eOk) {
        return "ok";
    }
    std::string text;
    const auto add = [&](sl::DLSSGStatus bit, const char* name) {
        if ((static_cast<u32>(status) & static_cast<u32>(bit)) != 0) {
            text += fmt::format("{}{}", text.empty() ? "" : "|", name);
        }
    };
    add(sl::DLSSGStatus::eFailResolutionTooLow, "resolution too low");
    add(sl::DLSSGStatus::eFailReflexNotDetectedAtRuntime, "Reflex not detected");
    add(sl::DLSSGStatus::eFailHDRFormatNotSupported, "HDR format not supported");
    add(sl::DLSSGStatus::eFailCommonConstantsInvalid, "common constants invalid");
    add(sl::DLSSGStatus::eFailGetCurrentBackBufferIndexNotCalled, "back buffer index not queried");
    return fmt::format("{} ({:#x})", text.empty() ? "unknown" : text, static_cast<u32>(status));
}

void OnApiError(const sl::APIError& error) {
    g_api_error.store(static_cast<int>(error.vkRes), std::memory_order_relaxed);
}

sl::FrameToken* TokenFor(u32 index) {
    sl::FrameToken* token{};
    if (GetApi().get_new_frame_token(token, &index) != sl::Result::eOk) {
        return nullptr;
    }
    return token;
}

void SetMarker(sl::PCLMarker marker, const sl::FrameToken& token) {
    if (const sl::Result result = GetApi().pcl_set_marker(marker, token);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: PCL marker {} failed for frame {}, result {}",
                    sl::getPCLMarkerAsStr(marker), static_cast<u32>(token),
                    ResultName(result));
    }
}

sl::Constants DefaultConstants() {
    sl::float4x4 identity;
    identity.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    identity.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    identity.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    identity.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    sl::Constants constants{};
    constants.cameraViewToClip = identity;
    constants.clipToCameraView = identity;
    constants.clipToLensClip = identity;
    constants.clipToPrevClip = identity;
    constants.prevClipToClip = identity;
    constants.jitterOffset = {0.0f, 0.0f};
    constants.mvecScale = {1.0f, 1.0f};
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.cameraPos = {0.0f, 0.0f, 0.0f};
    constants.cameraUp = {0.0f, 1.0f, 0.0f};
    constants.cameraRight = {1.0f, 0.0f, 0.0f};
    constants.cameraFwd = {0.0f, 0.0f, 1.0f};
    constants.cameraNear = 0.1f;
    constants.cameraFar = 1000.0f;
    constants.cameraFOV = 1.0f;
    constants.cameraAspectRatio = 16.0f / 9.0f;
    constants.depthInverted = sl::eFalse;
    constants.cameraMotionIncluded = sl::eTrue;
    constants.motionVectors3D = sl::eFalse;
    constants.motionVectorsJittered = sl::eFalse;
    return constants;
}

// Streamline compares a frame's constants set twice byte for byte, padding included: they are
// only ever copied whole.
static_assert(std::is_trivially_copyable_v<sl::Constants>);

/// Call with constants_mutex held.
void KeepConstants(FrameState& frames, u32 index, const sl::Constants& constants) {
    frames.last_constants = constants;
    auto& kept = frames.frame_constants[index % KEPT_CONSTANTS];
    kept.index = index;
    std::memcpy(&kept.constants, &constants, sizeof(constants));
}

bool SetMode(PresentState& state, bool on, const FrameGenPresent& present) {
    sl::DLSSGOptions options{};
    options.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    options.numFramesToGenerate = on ? FrameGenGenerated() : 1;
    // Menus and movies turn it off and on again: keep its memory meanwhile.
    options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    options.numBackBuffers = present.swapchain_images;
    const FrameState& frames = Frames();
    options.mvecDepthWidth = frames.input_width.load();
    options.mvecDepthHeight = frames.input_height.load();
    options.colorWidth = present.swapchain_extent.width;
    options.colorHeight = present.swapchain_extent.height;
    options.colorBufferFormat = static_cast<u32>(present.swapchain_format);
    options.mvecBufferFormat = static_cast<u32>(frames.motion_format.load());
    options.depthBufferFormat = static_cast<u32>(frames.depth_format.load());
    options.onErrorCallback = OnApiError;
    const Api& api = GetApi();
    if (const sl::Result result = api.dlssg_set_options(VIEWPORT, options);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: slDLSSGSetOptions({}) failed, result {}",
                    on ? "on" : "off", ResultName(result));
        return false;
    }
    state.on = on;
    state.options = options;
    if (!on) {
        return true;
    }
    sl::DLSSGOptions estimate = options;
    estimate.flags = estimate.flags | sl::DLSSGFlags::eRequestVRAMEstimate;
    sl::DLSSGState dlssg{};
    if (api.dlssg_get_state(VIEWPORT, dlssg, &estimate) == sl::Result::eOk) {
        state.vram = dlssg.estimatedVRAMUsageInBytes;
        PresentRates().max_generated.store(dlssg.numFramesToGenerateMax);
    }
    const u32 wanted = Settings::FrameGenMultiplier() - 1;
    if (wanted > options.numFramesToGenerate && !state.clamp_logged) {
        state.clamp_logged = true;
        LOG_INFO(Render_Vulkan, "DLSS-G: {}x asked, this system does up to {}x "
                 "(numFramesToGenerateMax {})",
                 wanted + 1, options.numFramesToGenerate + 1, FrameGenMaxGenerated());
    }
    state.real = 0;
    state.shown = 0;
    state.report_from = std::chrono::steady_clock::now();
    const VkRect2D& guest = present.guest_rect;
    LOG_INFO(Render_Vulkan,
             "DLSS-G: on, {} generated frames per frame (up to {} on this system); swapchain "
             "{}x{} {} x{}, guest image {}x{} at {},{} of a {}x{} frame; depth/motion {}x{}; "
             "estimated VRAM {} MiB",
             options.numFramesToGenerate, FrameGenMaxGenerated(), options.colorWidth, options.colorHeight,
             string_VkFormat(present.swapchain_format), options.numBackBuffers,
             guest.extent.width, guest.extent.height, guest.offset.x, guest.offset.y,
             present.frame_extent.width, present.frame_extent.height, options.mvecDepthWidth,
             options.mvecDepthHeight, state.vram >> 20);
    return true;
}

bool OptionsChanged(const PresentState& state, const FrameGenPresent& present) {
    const FrameState& frames = Frames();
    return state.options.colorWidth != present.swapchain_extent.width ||
           state.options.colorHeight != present.swapchain_extent.height ||
           state.options.colorBufferFormat != static_cast<u32>(present.swapchain_format) ||
           state.options.numBackBuffers != present.swapchain_images ||
           state.options.numFramesToGenerate != FrameGenGenerated() ||
           state.options.mvecDepthWidth != frames.input_width.load() ||
           state.options.mvecDepthHeight != frames.input_height.load();
}

void UpdateMode(PresentState& state, const FrameGenPresent& present) {
    if (state.retry_in != 0) {
        --state.retry_in;
    }
    if (!FrameGenActive()) {
        state.with_inputs = 0;
        if (state.on && SetMode(state, false, present)) {
            LOG_INFO(Render_Vulkan, "DLSS-G: off, frame generation was turned off");
        }
        return;
    }
    if (present.frame.inputs) {
        ++state.with_inputs;
        state.without_inputs = 0;
    } else {
        ++state.without_inputs;
        state.with_inputs = 0;
    }
    if (state.on) {
        if (state.without_inputs >= FRAMES_TO_TURN_OFF) {
            if (SetMode(state, false, present)) {
                LOG_INFO(Render_Vulkan,
                         "DLSS-G: off, no DLSS inputs for {} frames (menu, movie or loading)",
                         state.without_inputs);
            }
        } else if (OptionsChanged(state, present)) {
            void(SetMode(state, true, present));
        }
        return;
    }
    if (state.with_inputs >= FRAMES_TO_TURN_ON && state.retry_in == 0 &&
        !SetMode(state, true, present)) {
        state.retry_in = RETRY_FRAMES;
    }
}

void Report(PresentState& state, const sl::DLSSGState& dlssg) {
    const auto now = std::chrono::steady_clock::now();
    if (!state.on || now - state.report_from < REPORT_INTERVAL) {
        return;
    }
    const f64 seconds = std::chrono::duration<f64>(now - state.report_from).count();
    LOG_INFO(Render_Vulkan,
             "DLSS-G: on, status {}, presented {} frames for 1 (last {:.0f} s: {} real, {} "
             "shown, {:.1f} fps shown), up to {} generated, estimated VRAM {} MiB",
             StatusName(dlssg.status), state.last_shown, seconds, state.real, state.shown,
             static_cast<f64>(state.shown) / seconds, FrameGenMaxGenerated(), state.vram >> 20);
    state.real = 0;
    state.shown = 0;
    state.report_from = now;
}

void CountPresent(PresentState& state, u32 shown, bool generating) {
    Rates& rates = PresentRates();
    rates.generating.store(generating, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    if (!state.rates_from) {
        state.rates_from = now;
        return;
    }
    ++state.rates_real;
    state.rates_shown += shown;
    if (now - *state.rates_from < RATES_INTERVAL) {
        return;
    }
    const f64 seconds = std::chrono::duration<f64>(now - *state.rates_from).count();
    rates.real_fps.store(static_cast<f64>(state.rates_real) / seconds, std::memory_order_relaxed);
    rates.shown_fps.store(static_cast<f64>(state.rates_shown) / seconds,
                          std::memory_order_relaxed);
    state.rates_real = 0;
    state.rates_shown = 0;
    state.rates_from = now;
}

} // Anonymous namespace

std::unique_lock<std::mutex> LockStreamline() {
    return std::unique_lock{Frames().token_mutex};
}

void ResetFrameGen() {
    FrameState& frames = Frames();
    frames.index.store(0);
    frames.begun.store(NO_FRAME);
    frames.constants_index.store(NO_FRAME);
    frames.inputs_index.store(NO_FRAME);
    frames.input_width.store(0);
    frames.input_height.store(0);
    frames.depth_format.store(VK_FORMAT_UNDEFINED);
    frames.motion_format.store(VK_FORMAT_UNDEFINED);
    {
        std::scoped_lock lock{frames.constants_mutex};
        frames.last_constants.reset();
        frames.frame_constants = {};
    }
    Presents() = {};
    Rates& rates = PresentRates();
    rates.real_fps.store(0.0);
    rates.shown_fps.store(0.0);
    rates.generating.store(false);
    rates.max_generated.store(0);
    g_api_error.store(0);
}

void QueryFrameGenMax() {
    sl::DLSSGState dlssg{};
    if (GetApi().dlssg_get_state(VIEWPORT, dlssg, nullptr) == sl::Result::eOk) {
        PresentRates().max_generated.store(dlssg.numFramesToGenerateMax);
    }
}

bool ClaimFrameConstants(u32 index, const sl::Constants& constants) {
    FrameState& frames = Frames();
    const bool claimed = frames.constants_index.exchange(index) != index;
    if (claimed) {
        std::scoped_lock lock{frames.constants_mutex};
        KeepConstants(frames, index, constants);
    }
    return claimed;
}

sl::Result SetFrameConstants(u32 index, const sl::FrameToken& token) {
    FrameState& frames = Frames();
    sl::Constants constants;
    {
        std::scoped_lock lock{frames.constants_mutex};
        const auto& kept = frames.frame_constants[index % KEPT_CONSTANTS];
        if (kept.index != index) {
            return sl::Result::eErrorMissingConstants;
        }
        std::memcpy(&constants, &kept.constants, sizeof(constants));
    }
    return GetApi().set_constants(constants, token, VIEWPORT);
}

void NoteFrameGenInputs(u32 index, VkExtent2D extent, VkFormat depth_format,
                        VkFormat motion_format) {
    FrameState& frames = Frames();
    frames.input_width.store(extent.width);
    frames.input_height.store(extent.height);
    frames.depth_format.store(depth_format);
    frames.motion_format.store(motion_format);
    frames.inputs_index.store(index);
}

u32 FrameGenIndex() {
    return Frames().index.load();
}

void FrameGenBeginFrame() {
    if (!FrameGenSession()) {
        return;
    }
    FrameState& frames = Frames();
    const u32 index = frames.index.load();
    if (frames.begun.exchange(index) == index) {
        return;
    }
    const Api& api = GetApi();
    sl::FrameToken* token{};
    {
        const auto lock = LockStreamline();
        token = TokenFor(index);
    }
    if (!token) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: no frame token for frame {}", index);
        return;
    }
    // Not under the token lock: the present thread sets its markers while this waits. Reflex's
    // sleep doesn't read the frame index.
    if (const sl::Result result = api.reflex_sleep(*token); result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: slReflexSleep failed for frame {}, result {}", index,
                    ResultName(result));
    }
    const auto lock = LockStreamline();
    token = TokenFor(index);
    if (!token) {
        return;
    }
    // The guest's simulation runs on other threads: this thread's frame is all render
    // submission.
    SetMarker(sl::PCLMarker::eSimulationStart, *token);
    SetMarker(sl::PCLMarker::eSimulationEnd, *token);
    SetMarker(sl::PCLMarker::eRenderSubmitStart, *token);
}

FrameGenFrame FrameGenEndFrame() {
    if (!FrameGenSession()) {
        return {};
    }
    FrameState& frames = Frames();
    const u32 index = frames.index.load();
    const bool inputs = frames.inputs_index.load() == index;
    const bool needs_constants = frames.constants_index.exchange(index) != index;
    const Api& api = GetApi();
    const auto lock = LockStreamline();
    sl::FrameToken* const token = TokenFor(index);
    if (!token) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: no frame token for frame {}", index);
        return {index, false};
    }
    if (needs_constants) {
        {
            std::scoped_lock constants_lock{frames.constants_mutex};
            sl::Constants constants = frames.last_constants.value_or(DefaultConstants());
            constants.reset = sl::eTrue;
            KeepConstants(frames, index, constants);
        }
        if (const sl::Result result = SetFrameConstants(index, *token);
            result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan, "DLSS-G: slSetConstants failed for frame {}, result {}",
                        index, ResultName(result));
        }
    }
    if (!inputs) {
        const sl::ResourceTag tags[]{
            {nullptr, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent},
            {nullptr, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent},
        };
        if (const sl::Result result =
                api.set_tag_for_frame(*token, VIEWPORT, tags, std::size(tags), nullptr);
            result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan, "DLSS-G: clearing the tags of frame {} failed, result {}",
                        index, ResultName(result));
        }
    }
    SetMarker(sl::PCLMarker::eRenderSubmitEnd, *token);
    return {index, inputs};
}

void FrameGenNextFrame() {
    if (!FrameGenSession()) {
        return;
    }
    Frames().index.fetch_add(1);
    FrameGenBeginFrame();
}

void FrameGenBeforePresent(const FrameGenPresent& present) {
    const auto lock = LockStreamline();
    if (!FrameGenSession()) {
        return;
    }
    PresentState& state = Presents();
    UpdateMode(state, present);

    const f64 scale_x = static_cast<f64>(present.swapchain_extent.width) /
                        static_cast<f64>(std::max(present.frame_extent.width, 1u));
    const f64 scale_y = static_cast<f64>(present.swapchain_extent.height) /
                        static_cast<f64>(std::max(present.frame_extent.height, 1u));
    const VkRect2D& guest = present.guest_rect;
    const auto scaled = [](s64 value, f64 scale, u32 limit) {
        return static_cast<u32>(std::clamp<f64>(std::round(static_cast<f64>(value) * scale), 0.0,
                                                static_cast<f64>(limit)));
    };
    sl::Extent subrect{};
    subrect.left = scaled(guest.offset.x, scale_x, present.swapchain_extent.width);
    subrect.top = scaled(guest.offset.y, scale_y, present.swapchain_extent.height);
    subrect.width = std::min(scaled(guest.extent.width, scale_x, present.swapchain_extent.width),
                             present.swapchain_extent.width - subrect.left);
    subrect.height =
        std::min(scaled(guest.extent.height, scale_y, present.swapchain_extent.height),
                 present.swapchain_extent.height - subrect.top);

    const Api& api = GetApi();
    sl::FrameToken* const token = TokenFor(present.frame.index);
    if (!token) {
        return;
    }
    // Streamline knows the back buffer being presented: this only says where in it DLSS-G runs.
    const sl::ResourceTag backbuffer{nullptr, sl::kBufferTypeBackbuffer,
                                     sl::ResourceLifecycle::eValidUntilPresent, &subrect};
    if (const sl::Result result = api.set_tag_for_frame(*token, VIEWPORT, &backbuffer, 1, nullptr);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: back buffer tag failed for frame {}, result {}",
                    present.frame.index, ResultName(result));
    }
    // A retried present of the same frame keeps its first PresentStart.
    if (state.open_present != present.frame.index) {
        SetMarker(sl::PCLMarker::ePresentStart, *token);
        state.open_present = present.frame.index;
    }
}

void FrameGenAfterPresent(u32 index) {
    const auto lock = LockStreamline();
    if (!FrameGenSession()) {
        return;
    }
    const Api& api = GetApi();
    PresentState& state = Presents();
    if (sl::FrameToken* const token = TokenFor(index)) {
        SetMarker(sl::PCLMarker::ePresentEnd, *token);
    }
    state.open_present = NO_FRAME;
    state.closed_present = index;
    if (const int error = g_api_error.exchange(0, std::memory_order_relaxed); error != 0) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: vkQueuePresentKHR or vkAcquireNextImageKHR returned {}",
                    error);
    }
    sl::DLSSGState dlssg{};
    const sl::Result result = api.dlssg_get_state(VIEWPORT, dlssg, nullptr);
    const bool generating =
        result == sl::Result::eOk && state.on && dlssg.numFramesActuallyPresented > 1;
    CountPresent(state, generating ? dlssg.numFramesActuallyPresented : 1, generating);
    if (result != sl::Result::eOk) {
        if (!state.state_failed) {
            LOG_WARNING(Render_Vulkan, "DLSS-G: slDLSSGGetState failed, result {}",
                        ResultName(result));
        }
        state.state_failed = true;
        return;
    }
    state.state_failed = false;
    if (state.status != dlssg.status) {
        if (dlssg.status == sl::DLSSGStatus::eOk) {
            LOG_INFO(Render_Vulkan, "DLSS-G: status ok");
        } else {
            LOG_WARNING(Render_Vulkan, "DLSS-G: status {}", StatusName(dlssg.status));
        }
        state.status = dlssg.status;
    }
    if (!state.on) {
        return;
    }
    state.last_shown = dlssg.numFramesActuallyPresented;
    ++state.real;
    state.shown += dlssg.numFramesActuallyPresented;
    const bool too_small =
        dlssg.minWidthOrHeight != 0 &&
        std::min(state.options.colorWidth, state.options.colorHeight) < dlssg.minWidthOrHeight;
    if (dlssg.status != sl::DLSSGStatus::eOk || too_small) {
        FrameGenPresent off{};
        off.swapchain_extent = {state.options.colorWidth, state.options.colorHeight};
        off.swapchain_images = state.options.numBackBuffers;
        off.swapchain_format = static_cast<VkFormat>(state.options.colorBufferFormat);
        if (SetMode(state, false, off)) {
            LOG_WARNING(Render_Vulkan,
                        "DLSS-G: off ({}); trying again in {} frames",
                        too_small ? fmt::format("swapchain under {} pixels", dlssg.minWidthOrHeight)
                                  : StatusName(dlssg.status),
                        RETRY_FRAMES);
        }
        state.retry_in = RETRY_FRAMES;
        return;
    }
    Report(state, dlssg);
}

void FrameGenClosePresent(u32 index) {
    const auto lock = LockStreamline();
    if (!FrameGenSession()) {
        return;
    }
    PresentState& state = Presents();
    if (state.closed_present == index) {
        return;
    }
    if (sl::FrameToken* const token = TokenFor(index)) {
        if (state.open_present != index) {
            SetMarker(sl::PCLMarker::ePresentStart, *token);
        }
        SetMarker(sl::PCLMarker::ePresentEnd, *token);
    }
    state.open_present = NO_FRAME;
    state.closed_present = index;
}

bool FrameGenSupportsSwapchain(VkFormat format) {
    return format != VK_FORMAT_R16G16B16A16_SFLOAT;
}

u32 FrameGenMaxGenerated() {
    return PresentRates().max_generated.load(std::memory_order_relaxed);
}

u32 FrameGenGenerated() {
    const u32 wanted = Settings::FrameGenMultiplier() - 1;
    const u32 max = FrameGenMaxGenerated();
    return max == 0 ? 1 : std::min(wanted, max);
}

void FrameGenSuspend() {
    const auto lock = LockStreamline();
    if (!FrameGenSession()) {
        return;
    }
    PresentState& state = Presents();
    state.with_inputs = 0;
    if (!state.on) {
        return;
    }
    FrameGenPresent present{};
    present.swapchain_extent = {state.options.colorWidth, state.options.colorHeight};
    present.swapchain_images = state.options.numBackBuffers;
    present.swapchain_format = static_cast<VkFormat>(state.options.colorBufferFormat);
    if (SetMode(state, false, present)) {
        LOG_INFO(Render_Vulkan, "DLSS-G: off while the swapchain is recreated or destroyed");
    }
}

FrameGenRates FrameGenRecentRates() {
    const Rates& rates = PresentRates();
    return {
        .real_fps = rates.real_fps.load(std::memory_order_relaxed),
        .shown_fps = rates.shown_fps.load(std::memory_order_relaxed),
        .generating = rates.generating.load(std::memory_order_relaxed),
    };
}

} // namespace Vulkan::Streamline
