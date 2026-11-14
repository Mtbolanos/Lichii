// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS evaluation.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string_view>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/dlss_evaluation.h"
#include "video_core/dlss_mode.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#include "video_core/vulkan_common/streamline_frame_gen.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
// Streamline clears the output before DLSS writes it.
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;

std::atomic<u64> evaluations{};
std::atomic<u32> last_evaluation{}; ///< Mode, requested mode and preset, a byte each

// slEvaluateFeature returns eWarnOutOfVRAM after it has evaluated DLSS, when the VRAM budget is
// used up (commonInterface.cpp, slEvaluateFeatureInternal). The output is written; it is a warning.
bool Evaluated(sl::Result result) {
    return result == sl::Result::eOk || result == sl::Result::eWarnOutOfVRAM;
}

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

sl::DLSSMode ModeFor(Settings::NvidiaDlssQuality mode) {
    using Quality = Settings::NvidiaDlssQuality;
    switch (mode) {
    case Quality::Native:
        return sl::DLSSMode::eDLAA;
    case Quality::Quality:
        return sl::DLSSMode::eMaxQuality;
    case Quality::Balanced:
        return sl::DLSSMode::eBalanced;
    case Quality::Performance:
        return sl::DLSSMode::eMaxPerformance;
    case Quality::UltraPerformance:
        return sl::DLSSMode::eUltraPerformance;
    case Quality::Automatic:
        return sl::DLSSMode::eOff;
    }
    return sl::DLSSMode::eOff;
}

sl::DLSSPreset PresetFor(Settings::NvidiaDlssPreset preset) {
    using Preset = Settings::NvidiaDlssPreset;
    switch (preset) {
    case Preset::NvidiaDefault:
        return sl::DLSSPreset::eDefault;
    case Preset::E:
        return sl::DLSSPreset::ePresetE;
    case Preset::F:
        return sl::DLSSPreset::ePresetF;
    case Preset::J:
        return sl::DLSSPreset::ePresetJ;
    case Preset::K:
        return sl::DLSSPreset::ePresetK;
    case Preset::L:
        return sl::DLSSPreset::ePresetL;
    case Preset::M:
        return sl::DLSSPreset::ePresetM;
    }
    return sl::DLSSPreset::ePresetM;
}

sl::float4x4 Identity() {
    sl::float4x4 matrix;
    matrix.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    matrix.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    matrix.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    matrix.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return matrix;
}

VkImageMemoryBarrier Barrier(const StreamlineDlssUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

using Matrix4 = std::array<std::array<f64, 4>, 4>;
using Vector3 = std::array<f64, 3>;

Matrix4 ToMatrix(const std::array<f32, 16>& values) {
    Matrix4 matrix{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            matrix[row][col] = values[row * 4 + col];
        }
    }
    return matrix;
}

Matrix4 Multiply(const Matrix4& a, const Matrix4& b) {
    Matrix4 result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            for (size_t i = 0; i < 4; ++i) {
                result[row][col] += a[row][i] * b[i][col];
            }
        }
    }
    return result;
}

std::optional<Matrix4> Invert(const Matrix4& matrix) {
    std::array<std::array<f64, 8>, 4> rows{};
    for (size_t i = 0; i < 4; ++i) {
        std::copy_n(matrix[i].begin(), 4, rows[i].begin());
        rows[i][i + 4] = 1.0;
    }
    for (size_t col = 0; col < 4; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1; row < 4; ++row) {
            if (std::abs(rows[row][col]) > std::abs(rows[pivot][col])) {
                pivot = row;
            }
        }
        std::swap(rows[col], rows[pivot]);
        const f64 divisor = rows[col][col];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-12) {
            return std::nullopt;
        }
        for (f64& value : rows[col]) {
            value /= divisor;
        }
        for (size_t row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const f64 factor = rows[row][col];
            for (size_t i = 0; i < 8; ++i) {
                rows[row][i] -= factor * rows[col][i];
            }
        }
    }
    Matrix4 result{};
    for (size_t row = 0; row < 4; ++row) {
        std::copy_n(rows[row].begin() + 4, 4, result[row].begin());
    }
    return result;
}

f64 Dot(const Vector3& a, const Vector3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

f64 Determinant(const std::array<Vector3, 3>& rows) {
    return rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) -
           rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0]) +
           rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
}

std::optional<Vector3> Normalize(const Vector3& vector) {
    const f64 length = std::sqrt(Dot(vector, vector));
    if (!std::isfinite(length) || length < 1e-9) {
        return std::nullopt;
    }
    return Vector3{vector[0] / length, vector[1] / length, vector[2] / length};
}

/// Takes out of a vector its part along a unit direction.
Vector3 Without(Vector3 vector, const Vector3& direction) {
    const f64 amount = Dot(vector, direction);
    for (size_t i = 0; i < 3; ++i) {
        vector[i] -= amount * direction[i];
    }
    return vector;
}

/// Streamline's matrices take a row vector: they are the transpose of ours.
sl::float4x4 ToStreamline(const Matrix4& matrix) {
    sl::float4x4 result;
    for (u32 row = 0; row < 4; ++row) {
        result.setRow(row, {static_cast<f32>(matrix[0][row]), static_cast<f32>(matrix[1][row]),
                            static_cast<f32>(matrix[2][row]), static_cast<f32>(matrix[3][row])});
    }
    return result;
}

sl::float3 ToStreamline(const Vector3& vector) {
    return {static_cast<f32>(vector[0]), static_cast<f32>(vector[1]),
            static_cast<f32>(vector[2])};
}

/// Fills the camera constants DLSS-G checks from the scene's view-projection: the camera's
/// position and axes come out of it, and the projection is what is left once the view is taken
/// out. Streamline's clip depth goes from 0 to 1 where ours goes from -1 to 1.
bool SetCamera(sl::Constants& constants, const StreamlineDlssUpscaler::Inputs::Camera& camera) {
    constexpr Matrix4 TO_ZERO_ONE{{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.5, 0.5},
        {0.0, 0.0, 0.0, 1.0},
    }};
    constexpr Matrix4 FROM_ZERO_ONE{{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 2.0, -1.0},
        {0.0, 0.0, 0.0, 1.0},
    }};
    Matrix4 view_projection = ToMatrix(camera.view_projection);
    const f64 scale =
        std::hypot(view_projection[3][0], view_projection[3][1], view_projection[3][2]);
    if (!std::isfinite(scale) || scale < 1e-12) {
        return false;
    }
    // Clip w is then the distance along the view direction.
    for (auto& row : view_projection) {
        for (f64& value : row) {
            value /= scale;
        }
    }
    const auto axis = [&](size_t row) {
        return Vector3{view_projection[row][0], view_projection[row][1], view_projection[row][2]};
    };
    const Vector3 forward = axis(3);
    const Vector3 clip_x = axis(0);
    const Vector3 clip_y = axis(1);
    // The eye is where clip x, y and w are all zero (Cramer's rule).
    const std::array<Vector3, 3> system{clip_x, clip_y, forward};
    const Vector3 rhs{-view_projection[0][3], -view_projection[1][3], -view_projection[3][3]};
    const f64 det = Determinant(system);
    if (!std::isfinite(det) || std::abs(det) < 1e-12) {
        return false;
    }
    Vector3 position{};
    for (size_t column = 0; column < 3; ++column) {
        std::array<Vector3, 3> replaced = system;
        for (size_t i = 0; i < 3; ++i) {
            replaced[i][column] = rhs[i];
        }
        position[column] = Determinant(replaced) / det;
    }
    const std::optional<Vector3> right = Normalize(Without(clip_x, forward));
    if (!right) {
        return false;
    }
    const std::optional<Vector3> up = Normalize(Without(Without(clip_y, forward), *right));
    if (!up) {
        return false;
    }
    const Matrix4 view_to_world{{
        {(*right)[0], (*up)[0], forward[0], position[0]},
        {(*right)[1], (*up)[1], forward[1], position[1]},
        {(*right)[2], (*up)[2], forward[2], position[2]},
        {0.0, 0.0, 0.0, 1.0},
    }};
    const Matrix4 projection = Multiply(TO_ZERO_ONE, Multiply(view_projection, view_to_world));
    const std::optional<Matrix4> clip_to_view = Invert(projection);
    const Matrix4 clip_to_previous = Multiply(
        TO_ZERO_ONE, Multiply(ToMatrix(camera.clip_to_previous_clip), FROM_ZERO_ONE));
    const std::optional<Matrix4> previous_to_clip = Invert(clip_to_previous);
    if (!clip_to_view || !previous_to_clip || projection[0][0] <= 0.0 ||
        projection[1][1] <= 0.0) {
        return false;
    }
    // Depth is (a * z + b) / z: 0 at the near plane, 1 at the far one.
    const f64 a = projection[2][2];
    const f64 b = projection[2][3];
    const f64 near_plane = -b / a;
    f64 far_plane = b / (1.0 - a);
    if (!std::isfinite(near_plane) || near_plane <= 0.0) {
        return false;
    }
    if (!std::isfinite(far_plane) || far_plane <= near_plane) {
        far_plane = near_plane * 100000.0; // an infinite far plane
    }
    constants.cameraViewToClip = ToStreamline(projection);
    constants.clipToCameraView = ToStreamline(*clip_to_view);
    constants.clipToPrevClip = ToStreamline(clip_to_previous);
    constants.prevClipToClip = ToStreamline(*previous_to_clip);
    constants.cameraPos = ToStreamline(position);
    constants.cameraRight = ToStreamline(*right);
    constants.cameraUp = ToStreamline(*up);
    constants.cameraFwd = ToStreamline(forward);
    constants.cameraNear = static_cast<f32>(near_plane);
    constants.cameraFar = static_cast<f32>(far_plane);
    constants.cameraFOV = static_cast<f32>(2.0 * std::atan(1.0 / projection[1][1]));
    constants.cameraAspectRatio = static_cast<f32>(projection[1][1] / projection[0][0]);
    return true;
}

/// DLSS-G reads depth and motion at present, after the game reused them: Streamline copies them
/// here, on the evaluation's command buffer.
void TagFrameGenInputs(const sl::FrameToken& frame, const sl::ViewportHandle& viewport,
                       const sl::Resource& depth, const sl::Resource& motion,
                       const sl::Extent& area, VkCommandBuffer cmdbuf) {
    constexpr VkImageUsageFlags USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sl::Resource depth_copy = depth;
    depth_copy.usage = USAGE;
    sl::Resource motion_copy = motion;
    motion_copy.usage = USAGE;
    const sl::ResourceTag tags[]{
        {&depth_copy, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &area},
        {&motion_copy, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &area},
    };
    static std::atomic<bool> warned{};
    if (const sl::Result result =
            Streamline::GetApi().set_tag_for_frame(frame, viewport, tags, std::size(tags),
                                                   reinterpret_cast<sl::CommandBuffer*>(cmdbuf));
        result != sl::Result::eOk && !warned.exchange(true)) {
        LOG_WARNING(Render_Vulkan, "DLSS-G: tagging depth and motion vectors failed, result {}",
                    static_cast<int>(result));
    }
}

sl::Resource Resource(VkImage image, VkImageView view, VkFormat format, VkImageLayout layout,
                      VkExtent2D size, VkImageUsageFlags usage) {
    sl::Resource resource{sl::ResourceType::eTex2d, image, nullptr, view,
                          static_cast<uint32_t>(layout)};
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<uint32_t>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = usage;
    return resource;
}

} // Anonymous namespace

StreamlineDlssUpscaler::StreamlineDlssUpscaler(const Device& device_, Scheduler& scheduler_,
                                               MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Streamline::IsDlssAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    const sl::Result result = Streamline::GetApi().get_feature_function(
        sl::kFeatureDLSS, "slDLSSSetOptions", set_options);
    if (result != sl::Result::eOk || !set_options) {
        LOG_ERROR(Render_Vulkan, "Streamline's DLSS has no slDLSSSetOptions (result {})",
                  static_cast<int>(result));
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    Streamline::GetApi().get_feature_function(sl::kFeatureDLSS, "slDLSSGetState", get_state);
    Streamline::GetApi().get_feature_function(sl::kFeatureDLSS, "slDLSSGetOptimalSettings",
                                              get_optimal_settings);
    if (!get_optimal_settings) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    VkSamplerCreateInfo sampler_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    point_sampler = device.GetLogical().CreateSampler(sampler_info);
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    linear_sampler = device.GetLogical().CreateSampler(sampler_info);
}

StreamlineDlssUpscaler::~StreamlineDlssUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
    if (Streamline::IsDlssAvailable()) {
        Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{0u});
    }
}

void StreamlineDlssUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

bool StreamlineDlssUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    const bool extent_changed = !output_image || in.render_size.width != render_extent.width ||
                                in.render_size.height != render_extent.height ||
                                in.upscale_size.width != upscale_extent.width ||
                                in.upscale_size.height != upscale_extent.height;
    const auto settings = Settings::GetNvidiaDlssRuntimeSettings();
    const bool options_changed = extent_changed || in.hdr != hdr_input ||
                                 quality_mode != settings.quality ||
                                 model_preset != settings.preset;
    const bool reset = in.reset || options_changed;
    if (options_changed && !extent_changed) {
        scheduler.Finish();
    }
    if (extent_changed) {
        CreateOutput(in.upscale_size);
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
    }
    hdr_input = in.hdr;
    quality_mode = settings.quality;
    model_preset = settings.preset;

    // Keep the existing HDR controls for linear input such as Fortune's Weave. Tonemapped
    // input such as Ocarina's already has its exposure applied.
    sl::DLSSOptions options{};
    options.outputWidth = in.upscale_size.width;
    options.outputHeight = in.upscale_size.height;
    options.colorBuffersHDR =
        in.hdr && EnvInt("EDEN_NX2_DLSS_HDR", 1) != 0 ? sl::eTrue : sl::eFalse;
    options.useAutoExposure =
        in.hdr && EnvInt("EDEN_NX2_DLSS_AUTO_EXPOSURE", 1) != 0 ? sl::eTrue : sl::eFalse;
    const sl::DLSSPreset preset = PresetFor(settings.preset);
    options.dlaaPreset = preset;
    options.qualityPreset = preset;
    options.balancedPreset = preset;
    options.performancePreset = preset;
    options.ultraPerformancePreset = preset;
    options.ultraQualityPreset = preset;

    // DLSS super resolution only reads the jitter, motion and depth conventions; the camera
    // fields are for frame generation.
    const f32 rw = static_cast<f32>(in.render_size.width);
    const f32 rh = static_cast<f32>(in.render_size.height);
    sl::Constants constants{};
    constants.cameraViewToClip = Identity();
    constants.clipToCameraView = Identity();
    constants.clipToLensClip = Identity();
    constants.clipToPrevClip = Identity();
    constants.prevClipToClip = Identity();
    constants.jitterOffset = {in.jitter[0], in.jitter[1]};
    // Streamline wants motion in [-1,1] screen units; ours turns into render pixels.
    constants.mvecScale = {in.motion_vector_scale[0] / rw, in.motion_vector_scale[1] / rh};
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.cameraPos = {0.0f, 0.0f, 0.0f};
    constants.cameraUp = {0.0f, 1.0f, 0.0f};
    constants.cameraRight = {1.0f, 0.0f, 0.0f};
    constants.cameraFwd = {0.0f, 0.0f, 1.0f};
    constants.cameraNear = 0.1f;
    constants.cameraFar = 1000.0f;
    constants.cameraFOV = 1.0f;
    constants.cameraAspectRatio = rw / rh;
    constants.depthInverted = in.depth_inverted ? sl::eTrue : sl::eFalse;
    constants.cameraMotionIncluded = sl::eTrue;
    constants.motionVectors3D = sl::eFalse;
    constants.reset = reset ? sl::eTrue : sl::eFalse;
    constants.motionVectorsJittered = sl::eFalse;

    // With frame generation the frame index is the presented frame's, and only the first
    // evaluation of a frame sets its constants.
    const bool frame_gen = Streamline::FrameGenSession();
    bool frame_gen_inputs = false;
    if (frame_gen) {
        frame_gen_inputs = in.camera && SetCamera(constants, *in.camera);
    }
    const u32 index = frame_gen ? Streamline::FrameGenIndex() : frame_index++;
    const bool set_constants = !frame_gen || Streamline::ClaimFrameConstants(index, constants);
    if (frame_gen_inputs) {
        Streamline::NoteFrameGenInputs(index, in.render_size, in.depth.format,
                                       in.motion_vectors.format);
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, options, options_changed, requested = settings.quality,
                      preset_setting = settings.preset, index, constants, set_constants,
                      frame_gen, frame_gen_inputs, output_image_handle = *output_image,
                      output_view_handle = *output_view](vk::CommandBuffer cmdbuf) {
        const Streamline::Api& api = Streamline::GetApi();
        const sl::ViewportHandle viewport{0u};
        // The caller draws the output this frame even when DLSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };
        sl::FrameToken* frame{};

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // DLSS reads them as sampled images.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };

        sl::Resource color = Resource(in.color.image, in.color.view, in.color.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource depth = Resource(in.depth.image, in.depth.view, in.depth.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      in.depth_size.width != 0 ? in.depth_size : in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource motion = Resource(in.motion_vectors.image, in.motion_vectors.view,
                                       in.motion_vectors.format,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                       VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource output = Resource(output_image_handle, output_view_handle, OUTPUT_FORMAT,
                                       VK_IMAGE_LAYOUT_GENERAL, in.upscale_size, OUTPUT_USAGE);
        const sl::Extent render_area{0, 0, in.render_size.width, in.render_size.height};
        const sl::Extent output_area{0, 0, in.upscale_size.width, in.upscale_size.height};
        // Local tags: everything stays put until the evaluation is recorded.
        sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag output_tag{&output, sl::kBufferTypeScalingOutputColor,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &output_area};
        const sl::BaseStructure* inputs[] = {&viewport, &color_tag, &depth_tag, &motion_tag,
                                             &output_tag};
        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        sl::Result api_result = sl::Result::eOk;
        // Also without DLSS-G: DisableFrameGen may unload it on the present thread meanwhile.
        const auto lock = Streamline::LockStreamline();
        const auto result = VideoCommon::RecordDlssEvaluation(
            [&] {
                if (!options_changed) {
                    return true;
                }
                namespace Mode = VideoCommon::DlssMode;
                auto validated = options;
                const auto selection = Mode::Select(
                    requested, {in.render_size.width, in.render_size.height},
                    {in.upscale_size.width, in.upscale_size.height},
                    [&](Settings::NvidiaDlssQuality mode) -> std::optional<Mode::Range> {
                        validated.mode = ModeFor(mode);
                        sl::DLSSOptimalSettings optimal{};
                        const auto query =
                            reinterpret_cast<PFun_slDLSSGetOptimalSettings*>(get_optimal_settings);
                        api_result = query(validated, optimal);
                        if (api_result != sl::Result::eOk) {
                            return std::nullopt;
                        }
                        return Mode::Range{{optimal.renderWidthMin, optimal.renderHeightMin},
                                           {optimal.renderWidthMax, optimal.renderHeightMax}};
                    });
                if (selection.fallback) {
                    const Mode::Range range = selection.requested_range.value_or(Mode::Range{});
                    LOG_WARNING(
                        Render_Vulkan,
                        "NVIDIA DLSS requested {} but using {} for {}x{} -> {}x{}; "
                        "requested range {}x{} to {}x{} (query {})",
                        Settings::CanonicalizeEnum(requested),
                        selection.mode ? Settings::CanonicalizeEnum(*selection.mode) : "FSR 3.1",
                        in.render_size.width, in.render_size.height, in.upscale_size.width,
                        in.upscale_size.height, range.min.width, range.min.height, range.max.width,
                        range.max.height, selection.requested_range ? "supported" : "unavailable");
                }
                if (!selection.mode) {
                    LOG_WARNING(Render_Vulkan, "No NVIDIA DLSS mode accepts {}x{} -> {}x{}",
                                in.render_size.width, in.render_size.height, in.upscale_size.width,
                                in.upscale_size.height);
                    return false;
                }
                validated.mode = ModeFor(*selection.mode);
                evaluated_mode = *selection.mode;
                LOG_INFO(Render_Vulkan,
                         "NVIDIA DLSS (Streamline): {}x{} -> {}x{}, mode {}, preset {}, {}",
                         in.render_size.width, in.render_size.height, in.upscale_size.width,
                         in.upscale_size.height, Settings::CanonicalizeEnum(*selection.mode),
                         static_cast<int>(validated.dlaaPreset),
                         validated.colorBuffersHDR == sl::eTrue ? "HDR" : "LDR");
                const auto set = reinterpret_cast<PFun_slDLSSSetOptions*>(set_options);
                api_result = set(viewport, validated);
                return api_result == sl::Result::eOk;
            },
            [&] {
                api_result = api.get_new_frame_token(frame, &index);
                return api_result == sl::Result::eOk && frame;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);
            },
            [&] {
                if (!set_constants) {
                    return true;
                }
                api_result = frame_gen ? Streamline::SetFrameConstants(index, *frame)
                                       : api.set_constants(constants, *frame, viewport);
                return api_result == sl::Result::eOk;
            },
            [&] {
                api_result = api.evaluate_feature(sl::kFeatureDLSS, *frame, inputs, std::size(inputs),
                                                  reinterpret_cast<sl::CommandBuffer*>(*cmdbuf));
                if (api_result == sl::Result::eWarnOutOfVRAM && !vram_warned.exchange(true)) {
                    LOG_WARNING(Render_Vulkan, "NVIDIA DLSS: Streamline reports the VRAM budget is "
                                               "used up; DLSS stays on");
                }
                if (!Evaluated(api_result)) {
                    return false;
                }
                if (frame_gen_inputs) {
                    TagFrameGenInputs(*frame, viewport, depth, motion, render_area, *cmdbuf);
                }
                return true;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                                       to_general);
            });
        if (result != VideoCommon::DlssEvaluationResult::Succeeded && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan,
                      "NVIDIA DLSS recording failed at stage {} (Streamline result {}); using FSR 3.1",
                      static_cast<int>(result), static_cast<int>(api_result));
        }
        if (result != VideoCommon::DlssEvaluationResult::Succeeded) {
            clear_output();
            return;
        }
        last_evaluation.store(static_cast<u32>(evaluated_mode) |
                                  (static_cast<u32>(requested) << 8) |
                                  (static_cast<u32>(preset_setting) << 16),
                              std::memory_order_relaxed);
        evaluations.fetch_add(1, std::memory_order_release);
        // NGX doesn't report the model it picked, but its memory tells them apart: at 3840x2160
        // output DLSS 310.9 allocates about 198 MB for presets E and F and 307 MB for J and K.
        sl::DLSSState state{};
        if (options_changed && get_state &&
            reinterpret_cast<PFun_slDLSSGetState*>(get_state)(viewport, state) == sl::Result::eOk) {
            LOG_INFO(Render_Vulkan, "NVIDIA DLSS (Streamline): preset {} allocated {:.1f} MB",
                     static_cast<int>(options.dlaaPreset),
                     static_cast<double>(state.estimatedVRAMUsageInBytes) / 1e6);
        }
    });
    // DLSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

VideoCore::DlssStatus StreamlineDlssUpscaler::Status() {
    const u64 count = evaluations.load(std::memory_order_acquire);
    const u32 last = last_evaluation.load(std::memory_order_relaxed);
    return {
        .evaluations = count,
        .mode = static_cast<Settings::NvidiaDlssQuality>(last & 0xff),
        .requested = static_cast<Settings::NvidiaDlssQuality>((last >> 8) & 0xff),
        .preset = static_cast<Settings::NvidiaDlssPreset>((last >> 16) & 0xff),
    };
}

} // namespace Vulkan
