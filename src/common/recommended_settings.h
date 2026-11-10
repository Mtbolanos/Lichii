// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <type_traits>
#include <vector>
#include "common/settings.h"

namespace Settings {

/// The Lichii developer's recommended settings. This table is the only place that lists them:
/// change a value here to change what "Use recommended settings" writes. Personal and hardware
/// choices (language, region, resolution, upscaler, backend, device) stay out of it.
template <typename Func>
void ForEachRecommendedSetting(Values& settings, Func&& func) {
    const auto entry = [&func]<typename Type, bool ranged>(SwitchableSetting<Type, ranged>& setting,
                                                           std::type_identity_t<Type> value) {
        func(setting, value);
    };

    // Graphics > Advanced
    entry(settings.gpu_accuracy, GpuAccuracy::Low);
    entry(settings.dma_accuracy, DmaAccuracy::Default);
    entry(settings.gpu_fence_behavior, GpuFenceBehavior::Default);
    entry(settings.native_compute_write_mark, NativeComputeWriteMark::Auto);
    entry(settings.vram_usage_mode, VramUsageMode::Conservative);
    entry(settings.nvdec_emulation, NvdecEmulation::Gpu);
    entry(settings.max_anisotropy, AnisotropyMode::Automatic);
    entry(settings.accelerate_astc, AstcDecodeMode::Gpu);
    entry(settings.frame_pacing_mode, FramePacingMode::Target_Auto);
    entry(settings.astc_recompression, AstcRecompression::Bc3);
    entry(settings.sync_memory_operations, false);
    entry(settings.renderer_force_max_clock, false);
    entry(settings.use_disk_shader_cache, true);
    entry(settings.use_vulkan_driver_pipeline_cache, true);
    entry(settings.native_async_draws, NativeAsyncDraws::Bounded);
    entry(settings.native_pipeline_wait_ms, 8);
    entry(settings.precache_shaders, PrecacheShaders::Off);
    entry(settings.use_video_framerate, false);
    entry(settings.mkw_30fps, false);
    entry(settings.mkw_dynamic_fps, false);
    entry(settings.use_reactive_flushing, true);
    entry(settings.barrier_feedback_loops, true);
    entry(settings.enable_buffer_history, false);

    // Graphics > Extras
    entry(settings.skip_cpu_inner_invalidation, false);
    entry(settings.async_presentation, true);
    entry(settings.fix_bloom_effects, false);
    entry(settings.use_asynchronous_shaders, true);
    entry(settings.gpu_unswizzle_texture_size, GpuUnswizzleSize::Large);
    entry(settings.gpu_unswizzle_stream_size, GpuUnswizzle::Medium);
    entry(settings.gpu_unswizzle_chunk_size, GpuUnswizzleChunk::Medium);
    entry(settings.gpu_unswizzle_enabled, false);
#if defined(_WIN32) || defined(__linux__)
    // Other hosts skip the pinning.
    entry(settings.pin_gpu_threads, true);
#endif
#ifndef __APPLE__
    // Extended dynamic state stays disabled on macOS: MoltenVK renders black screens with it.
    entry(settings.dyna_state, ExtendedDynamicState::EDS2);
#endif
    entry(settings.sample_shading, 0);
    entry(settings.vertex_input_dynamic_state, true);

    // System
    entry(settings.cpu_clock, CpuClock::Normal);
    entry(settings.gpu_clock, GpuClock::Boost);
    entry(settings.use_multi_core, true);
    entry(settings.memory_layout_mode, MemoryLayout::Memory_4Gb);
    entry(settings.use_speed_limit, true);
    entry(settings.speed_limit, 100);
    entry(settings.slow_speed_limit, 50);
    entry(settings.turbo_speed_limit, 200);
    entry(settings.sync_core_speed, false);

    // CPU
    entry(settings.cpu_accuracy, CpuAccuracy::Auto);
    entry(settings.use_custom_cpu_ticks, false);
}

/// A global setting that changed, with its values as BasicSetting::ToString gives them.
struct RecommendedSettingChange {
    const BasicSetting* setting;
    std::string previous;
    std::string current;
};

/// Writes the recommended values to the global settings and returns the ones that changed.
/// Per-game values, and whether a setting uses its per-game value, stay as they are. When
/// something changed, recommended_settings_backup takes the previous value of every recommended
/// setting it does not hold yet. A value it holds stays until it is restored, so applying twice
/// does not lose the values from before the first time.
std::vector<RecommendedSettingChange> ApplyRecommendedSettings(Values& settings);

/// Puts back the global values recommended_settings_backup holds, clears it and returns the
/// settings that changed. Only settings in the recommended table are restored.
std::vector<RecommendedSettingChange> RestorePreviousSettings(Values& settings);

} // namespace Settings
