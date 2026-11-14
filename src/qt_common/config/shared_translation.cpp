// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 Torzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "shared_translation.h"

#include <map>
#include <memory>
#include <utility>
#include <QCoreApplication>
#include "common/settings.h"
#include "common/settings_enums.h"
#include "common/settings_setting.h"
#include "common/time_zone.h"
#include "qt_common/config/uisettings.h"

namespace ConfigurationShared {

/**
 * Builds the label and tooltip every settings widget shows, keyed by setting id.
 *
 * @param parent Object used for translation lookups.
 * @return The translation map.
 */
std::unique_ptr<TranslationMap> InitializeTranslations(QObject* parent) {
    std::unique_ptr<TranslationMap> translations = std::make_unique<TranslationMap>();
    const auto& tr = [](const char* text) -> QString { return QCoreApplication::translate("ConfigurationShared", text); };

#define INSERT(SETTINGS, ID, NAME, TOOLTIP)                                                        \
    translations->insert(std::pair{SETTINGS::values.ID.Id(), std::pair{(NAME), (TOOLTIP)}})

    // A setting can be ignored by giving it a blank name

    // Applets
    INSERT(Settings, cabinet_applet_mode, tr("Amiibo editor"), QString());
    INSERT(Settings, controller_applet_mode, tr("Controller configuration"), QString());
    INSERT(Settings, data_erase_applet_mode, tr("Data erase"), QString());
    INSERT(Settings, error_applet_mode, tr("Error"), QString());
    INSERT(Settings, net_connect_applet_mode, tr("Net connect"), QString());
    INSERT(Settings, player_select_applet_mode, tr("Player select"), QString());
    INSERT(Settings, swkbd_applet_mode, tr("Software keyboard"), QString());
    INSERT(Settings, mii_edit_applet_mode, tr("Mii Edit"), QString());
    INSERT(Settings, web_applet_mode, tr("Online web"), QString());
    INSERT(Settings, shop_applet_mode, tr("Shop"), QString());
    INSERT(Settings, photo_viewer_applet_mode, tr("Photo viewer"), QString());
    INSERT(Settings, offline_web_applet_mode, tr("Offline web"), QString());
    INSERT(Settings, login_share_applet_mode, tr("Login share"), QString());
    INSERT(Settings, wifi_web_auth_applet_mode, tr("Wifi web auth"), QString());
    INSERT(Settings, my_page_applet_mode, tr("My page"), QString());
    INSERT(Settings, enable_overlay, tr("Enable Overlay Applet"),
           tr("Enables Horizon\'s built-in overlay applet. Press and hold the home button for 1 "
              "second to show it."));

    // Audio
    INSERT(Settings, sink_id, tr("Output Engine:"), QString());
    INSERT(Settings, audio_output_device_id, tr("Output Device:"), QString());
    INSERT(Settings, audio_input_device_id, tr("Input Device:"), QString());
    INSERT(Settings, audio_muted, tr("Mute audio"), QString());
    INSERT(Settings, volume, tr("Volume:"), QString());
    INSERT(Settings, dump_audio_commands, QString(), QString());
    INSERT(UISettings, mute_when_in_background, tr("Mute audio when in background"), QString());

    // Core
    INSERT(Settings, use_multi_core, tr("Multicore CPU Emulation"),
           tr("This option increases CPU emulation thread use from 1 to the maximum of 4.\n"
              "This is mainly a debug option and shouldn't be disabled."));
    INSERT(Settings, memory_layout_mode, tr("Memory Layout"),
           tr("Increases the amount of emulated RAM.\nDoesn't affect performance/stability but may "
              "allow HD texture "
              "mods to load."));
    INSERT(Settings, use_speed_limit, QString(), QString());
    INSERT(Settings, current_speed_mode, QString(), QString());
    INSERT(Settings, speed_limit, tr("Limit Speed Percent"),
           tr("Controls the game's maximum rendering speed, but it's up to each game if it runs "
              "faster or not.\n200% for a 30 FPS game is 60 FPS, and for a "
              "60 FPS game it will be 120 FPS.\nDisabling it means unlocking the framerate to the "
              "maximum your PC can reach."));

    INSERT(Settings, turbo_speed_limit, tr("Turbo Speed"),
           tr("When the Turbo Speed hotkey is pressed, the speed will be limited to this "
              "percentage."));
    INSERT(Settings, slow_speed_limit, tr("Slow Speed"),
           tr("When the Slow Speed hotkey is pressed, the speed will be limited to this "
              "percentage."));

    INSERT(Settings, sync_core_speed, tr("Synchronize Core Speed"),
           tr("Synchronizes CPU core speed with the game's maximum rendering speed to boost FPS "
              "without affecting game speed (animations, physics, etc.).\n"
              "Can help reduce stuttering at lower framerates."));

    // Cpu
    INSERT(Settings, cpu_accuracy, tr("Accuracy:"),
           tr("Change the accuracy of the emulated CPU (for debugging only)."));
    INSERT(Settings, cpu_backend, tr("Backend:"), QString());

    INSERT(Settings, cpu_clock, tr("CPU Clocks"),
           tr("Raises the clock the emulated CPU reports, which removes some FPS limiters.\n"
              "Weaker CPUs may see reduced performance, and certain games may behave improperly."));

    INSERT(Settings, use_custom_cpu_ticks, QString(), QString());
    INSERT(Settings, cpu_ticks, tr("Custom CPU Ticks"),
           tr("Set a custom value of CPU ticks. Higher values can increase performance, but may "
              "cause deadlocks. A range of 77-21000 is recommended."));
    INSERT(Settings, cpu_backend, tr("Backend:"), QString());

    // Cpu Debug

    // Cpu Unsafe
    INSERT(
        Settings, cpuopt_unsafe_host_mmu, tr("Enable Host MMU Emulation (fastmem)"),
        tr("This optimization speeds up memory accesses by the guest program.\nEnabling it causes "
           "guest memory reads/writes to be done directly into memory and make use of Host's "
           "MMU.\nDisabling this forces all memory accesses to use Software MMU Emulation."));
    INSERT(
        Settings, cpuopt_unsafe_unfuse_fma,
        tr("Unfuse FMA (improve performance on CPUs without FMA)"),
        tr("This option improves speed by reducing accuracy of fused-multiply-add instructions on "
           "CPUs without native FMA support."));
    INSERT(
        Settings, cpuopt_unsafe_reduce_fp_error, tr("Faster FRSQRTE and FRECPE"),
        tr("This option improves the speed of some approximate floating-point functions by using "
           "less accurate native approximations."));
    INSERT(Settings, cpuopt_unsafe_ignore_standard_fpcr,
           tr("Faster ASIMD instructions (32 bits only)"),
           tr("This option improves the speed of 32 bits ASIMD floating-point functions by running "
              "with incorrect rounding modes."));
    INSERT(Settings, cpuopt_unsafe_inaccurate_nan, tr("Inaccurate NaN handling"),
           tr("This option improves speed by removing NaN checking.\nPlease note this also reduces "
              "accuracy of certain floating-point instructions."));
    INSERT(Settings, cpuopt_unsafe_fastmem_check, tr("Disable address space checks"),
           tr("This option improves speed by eliminating a safety check before every memory "
              "operation.\nDisabling it may allow arbitrary code execution."));
    INSERT(
        Settings, cpuopt_unsafe_ignore_global_monitor, tr("Ignore global monitor"),
        tr("This option improves speed by relying only on the semantics of cmpxchg to ensure "
           "safety of exclusive access instructions.\nPlease note this may result in deadlocks and "
           "other race conditions."));

    // Renderer
    INSERT(Settings, hdr_mode, tr("HDR output:"),
           tr("HDR output for games with an HDR mode. Requires Vulkan.\n"
              "Off (default): always SDR.\n"
              "Auto: HDR while Windows HDR (\"Use HDR\") is on for the monitor that shows the "
              "game, SDR otherwise and on hosts that cannot tell.\n"
              "On: HDR whenever the display driver offers an HDR format.\n"
              "The game is told the display is HDR only while Lichii really presents HDR. SDR "
              "games and menus are shown at the reference white.\n"
              "While HDR is active, anti-aliasing, post-processing and FSR/SGSR upscaling are off "
              "and the window is scaled bilinearly. Frame generation keeps working.\n"
              "Screenshots are tone mapped to SDR."));
    INSERT(Settings, hdr_swapchain, tr("HDR format:"),
           tr("HDR10: 10-bit PQ, what HDR displays take.\n"
              "scRGB: 16-bit float. Try it if HDR shows banding, which some NVIDIA drivers do "
              "with 10-bit in fullscreen."));
    INSERT(Settings, hdr_reference_white, tr("HDR Reference White"),
           tr("Reference white in nits reported to the game, the brightness of its HUD and "
              "menus. 0 (default): Windows' SDR content brightness for the monitor, so the game "
              "matches the desktop; elsewhere the Switch 2 calibration, HDR Peak Luminance "
              "divided by 1.5, rounded up and clamped to 50-300 nits.\n"
              "SDR games and menus are shown at this brightness while HDR is active, and HDR "
              "screenshots map it to SDR white."));
    INSERT(Settings, hdr_peak_luminance, tr("HDR Peak Luminance"),
           tr("Peak luminance in nits reported to the game, which tone maps to it. 0 (default): "
              "the monitor's peak from its EDID as Windows reports it; elsewhere 300 nits, the "
              "Switch 2 HDR calibration's starting value.\n"
              "Also the HDR10 metadata peak, and what an HDR image rolls off to on an SDR "
              "screen."));
    INSERT(Settings, renderer_backend, tr("API:"),
           tr("Changes the output graphics API.\nVulkan is recommended."));
    INSERT(Settings, vulkan_device, tr("Device:"),
           tr("This setting selects the GPU to use (Vulkan only)."));
    INSERT(Settings, resolution_setup, tr("Resolution:"),
           tr("Forces to render at a different resolution.\n"
              "Higher resolutions require more VRAM and bandwidth.\n"
              "Options lower than 1X can cause artifacts."));
    INSERT(Settings, scaling_filter, tr("Window Adapting Filter:"),
           tr("Bilinear while HDR output is active."));
    INSERT(Settings, game_dlss_network, tr("Use the game's own DLSS"),
           tr("Runs a Switch 2 game's own DLSS (its upscaling network, translated for this GPU) "
              "instead of AMD FSR 3.1 in its place.\nAMD FSR 3.1 still draws while the network is "
              "checked, when it can't run on this GPU, with resolution scaling, and for the rest "
              "of the session if its output fails Lichii's checks.\nTakes effect when a game "
              "starts."));
    INSERT(Settings, frame_rate_limit, tr("Frame rate limit (Switch 2):"),
           tr("Rate of the Switch 2 VSync event that games start their frames on (default 60 Hz)."
              "\nAbove 60 Hz most games run too fast; use it with a game patch that follows real "
              "time, such as \"FPS Unlock (built-in)\" for Ocarina of Time.\nTakes effect when a "
              "game starts."));
    INSERT(Settings, fsr_sharpening_slider, tr("RCAS Sharpness:"),
           tr("Strength of RCAS (AMD's contrast-adaptive sharpening) on the upscaler used in place "
              "of a game's DLSS (AMD FSR 3.1); 0% turns it off.\nAlso sets the sharpening "
              "of the AMD FSR 1 and SGSR window adapting filters."));
    INSERT(Settings, ocarina_fsr, tr("Use AMD FSR 3.1 upscaling in Ocarina of Time"),
           tr("Replaces Ocarina of Time's native upscaler using scene depth and camera motion. "
              "Works on any supported Vulkan GPU. NVIDIA DLSS takes precedence when enabled and "
              "available. RCAS Sharpness controls sharpening. Takes effect when a game starts."));
#if defined(__linux__) && !defined(__ANDROID__)
    INSERT(Settings, amd_fsr4, tr("Use FSR 4 (fsr4-vk community Vulkan port) in place of FSR 3.1"),
           tr("Uses fsr4-vk, a community Vulkan port of AMD's ML-based FSR 4.1.1 that Lichii's "
              "x86-64 Linux packages include, where FSR replaces a game's upscaler (Ocarina of Time with "
              "the setting above). It is not AMD's official FSR 4 and is not affiliated with or "
              "endorsed by AMD. Radeon RX 6000 or newer, or an RDNA 3 Ryzen GPU, on Mesa's RADV "
              "driver; RDNA 4 runs the FP8 network, older GPUs the INT8 one.\nIf FSR 4 can't run, "
              "FSR 3.1 is used and the log says why. Takes effect when a game starts."));
#else
    INSERT(Settings, amd_fsr4, tr("Use AMD FSR 4 in place of FSR 3.1"),
           tr("Uses AMD's ML-based FSR 4 where FSR replaces a game's upscaler (Ocarina of Time "
              "with the setting above). Radeon RX 6000 or newer, or a Ryzen AI processor, with a "
              "current AMD driver.\nNeeds AMD's FidelityFX runtime, which Lichii downloads from "
              "AMD when you ask. The files are AMD's, under AMD's licence. Without them, or if FSR 4 "
              "can't run, FSR 3.1 is used. Takes effect when a game starts."));
#endif
    INSERT(Settings, nvidia_dlss, tr("Use NVIDIA DLSS in place of a game's DLSS (experimental)"),
           tr("Uses NVIDIA DLSS on GeForce RTX cards in place of a Switch 2 game's own DLSS, "
              "instead of AMD FSR 3.1.\nNeeds NVIDIA's Streamline loader, and NVIDIA DLSS unless "
              "the NVIDIA app has downloaded it; Lichii does not ship NVIDIA's files. Use the "
              "download button below.\nFire Emblem: Fortune's Weave also needs "
              "EDEN_NX2_DLSS_REPLACEMENT=dlss. Without that override, its spatial workaround "
              "remains active.\nIn The Legend of Zelda: Ocarina of Time it replaces the game's "
              "upscaler at 1x resolution scale.\nThe window adapting filter (the FSR 1, Bilinear "
              "and similar button in the status bar) is separate and runs after DLSS. While DLSS "
              "runs, the status bar shows DLSS, its mode and its preset. If DLSS fails, it stays "
              "off until the game restarts and the log says why.\nTakes effect when a game "
              "starts."));
    INSERT(Settings, nvidia_dlss_quality, tr("NVIDIA DLSS quality:"),
           tr("Applies only when a game uses its own DLSS.\nAutomatic selects a mode for the "
              "game's render/output sizes. Native (DLAA) "
              "applies anti-aliasing at the game's render resolution, then scales to its output; "
              "Ocarina's 1440p mode stays at 1440p instead of reconstructing to 4K.\n"
              "In The Legend of Zelda: Ocarina of Time (docked, with the Max Resolution add-on), "
              "Balanced, Performance and Ultra Performance lower the game's render resolution "
              "from 2560x1440 to 2276x1280, 1920x1080 and 1280x720 before DLSS reconstructs 4K. "
              "That resolution changes when the game starts.\n"
              "In other games, explicit modes only apply when the game's render size is within "
              "NVIDIA's supported range; otherwise Automatic is used and the fallback is logged. "
              "There they don't change the game's render resolution. Apply/OK updates the mode "
              "while playing."));
    INSERT(
        Settings, nvidia_dlss_preset, tr("NVIDIA DLSS model preset:"),
        tr("Applies only when a game uses its own DLSS.\nM (DLSS 4.5) is Lichii's default for "
           "every quality mode. NVIDIA recommends K for "
           "DLAA/Quality/Balanced, M for Performance, and L for Ultra Performance.\n"
           "L/M cost more on RTX 20/30 GPUs; consider K. J may reduce ghosting but flicker more. "
           "E/F are deprecated CNN models. NVIDIA default lets the runtime choose.\n"
           "L/M require DLSS 310.5 or newer; the download button provides 310.9.16. "
           "If Fire Emblem shows horizontal lines with L/M at 2x, try K. "
           "NVIDIA app overrides may supersede this choice. Apply/OK updates it while playing."));
    INSERT(Settings, anti_aliasing, tr("Anti-Aliasing Method:"),
           tr("The anti-aliasing method to use.\nSMAA offers the best quality.\nFXAA "
              "can produce a more stable picture in lower resolutions.\nOff while HDR output "
              "is active."));
    INSERT(Settings, oot_anti_aliasing, tr("Ocarina of Time anti-aliasing:"),
           tr("The Legend of Zelda: Ocarina of Time only.\nGame: the game's own anti-aliasing "
              "network, which sharpens heavily.\nNo sharpening: the image as the game renders it, "
              "without the network (faster; pairs well with a higher resolution). Has no effect while "
              "NVIDIA DLSS is on.\nTakes effect when the game starts."));
    INSERT(Settings, post_shader_chain, QString(), QString());
    INSERT(Settings, post_shader_preset, QString(), QString());
    INSERT(Settings, post_shader_enabled, tr("Enable post-processing effects"),
           tr("Applies post-processing effects to the final image."));
    INSERT(Settings, fullscreen_mode, tr("Fullscreen Mode:"),
           tr("The method used to render the window in fullscreen.\nBorderless offers the best "
              "compatibility with the on-screen keyboard that some games request for "
              "input.\nExclusive "
              "fullscreen may offer better performance and better Freesync/Gsync support."));
    INSERT(Settings, aspect_ratio, tr("Aspect Ratio:"),
           tr("Stretches the renderer to fit the specified aspect ratio.\nMost games only support "
              "16:9, so modifications are required to get other ratios.\nAlso controls the "
              "aspect ratio of captured screenshots."));
    INSERT(Settings, use_disk_shader_cache, tr("Use persistent pipeline cache"),
           tr("Allows saving shaders to storage for faster loading on following game "
              "boots.\nDisabling it is only intended for debugging."));
    INSERT(
        Settings, use_asynchronous_gpu_emulation, tr("Use asynchronous GPU emulation"),
        tr("Uses an extra CPU thread for rendering.\nThis option should always remain enabled."));
    INSERT(Settings, nvdec_emulation, tr("NVDEC emulation:"),
           tr("Specifies how videos should be decoded.\nIt can either use the CPU or the GPU for "
              "decoding, or perform no decoding at all (black screen on videos).\n"
              "In most cases, GPU decoding provides the best performance."));
    INSERT(Settings, accelerate_astc, tr("ASTC Decoding Method:"),
           tr("This option controls how ASTC textures should be decoded.\n"
              "CPU: Use the CPU for decoding.\n"
              "GPU: Use the GPU's compute shaders to decode ASTC textures (recommended).\n"
              "CPU Asynchronously: Use the CPU to decode ASTC textures on demand. Eliminates"
              "ASTC decoding\nstuttering but may present artifacts."));
    INSERT(Settings, astc_recompression, tr("ASTC Recompression Method:"),
           tr("Most GPUs lack support for ASTC textures and must decompress to an"
              "intermediate format: RGBA8.\n"
              "BC1/BC3: The intermediate format will be recompressed to BC1 or BC3 format,\n"
              " saving VRAM but degrading image quality."));
    INSERT(Settings, frame_pacing_mode, tr("Frame Pacing Mode (Vulkan only)"),
           tr("Controls how the emulator manages frame pacing to reduce stuttering and make the "
              "frame rate smoother and more consistent."));
    INSERT(Settings, vram_usage_mode, tr("VRAM Usage Mode:"),
           tr("Selects whether the emulator should prefer to conserve memory or make maximum usage "
              "of available video memory for performance.\nAggressive mode may impact performance "
              "of other applications such as recording software."));
    INSERT(Settings, skip_cpu_inner_invalidation, tr("Skip CPU Inner Invalidation"),
           tr("Skips certain cache invalidations during memory updates, reducing CPU usage and "
              "improving latency. This may cause soft-crashes."));
    INSERT(Settings, fps_limit, tr("FPS Limit"),
           tr("Caps how often the game presents, so the game itself runs at this rate. 0 leaves "
              "the frame rate to the game and the display."));
    INSERT(Settings, vsync_mode, tr("VSync Mode:"),
           tr("FIFO (VSync) does not drop frames or exhibit tearing but is limited by the screen "
              "refresh rate.\nFIFO Relaxed allows tearing as it recovers from a slow down.\n"
              "Mailbox can have lower latency than FIFO and does not tear but may drop "
              "frames.\nImmediate (no synchronization) presents whatever is available and can "
              "exhibit tearing."));
    INSERT(Settings, bg_red, QString(), QString());
    INSERT(Settings, bg_green, QString(), QString());
    INSERT(Settings, bg_blue, QString(), QString());

    INSERT(Settings, frame_gen, tr("Enable Frame Generation"),
           tr("Paused while HDR output is active."));
    INSERT(Settings, frame_gen_type, tr("Frame Generation Type"),
           tr("Optical Flow Interpolation is built in and runs on any GPU.\nNVIDIA DLSS Frame "
              "Generation (Windows only) needs a GeForce RTX 40 series or newer GPU (3x and 4x need "
              "an RTX 50 series), Hardware-accelerated GPU scheduling on in Windows' graphics "
              "settings, \"Use NVIDIA DLSS\" on and NVIDIA's files from the NVIDIA DLSS panel. "
              "It takes the game's depth and motion vectors from the DLSS evaluation, so it runs "
              "in The Legend of Zelda: Ocarina of Time only, docked, with DLSS as its upscaler. "
              "Other games, and scRGB HDR output, use Optical Flow Interpolation instead.\nVSync "
              "is off while it runs. Choosing it takes effect when a game starts."));
    INSERT(Settings, frame_gen_multiplier, tr("Frame Generation Multiplier"), QString());
    INSERT(Settings, frame_gen_target_rate, tr("Target Frame Rate"), QString());
    INSERT(Settings, frame_gen_flow_scale_auto, tr("Auto Flow Scale"), QString());
    INSERT(Settings, frame_gen_flow_scale, tr("Flow Scale"), QString());
    INSERT(Settings, frame_gen_queue_target, tr("Queue Target"), QString());
    INSERT(Settings, frame_gen_dump_flow, tr("Dump Optical Flow"), QString());

    // Renderer (Advanced Graphics)
    INSERT(Settings, use_asynchronous_gpu_emulation, QString(), QString());

    INSERT(Settings, sync_memory_operations, tr("Sync Memory Operations"),
           tr("Ensures data consistency between compute and memory operations.\nThis option fixes "
              "issues in games, but may degrade performance.\nUnreal Engine 4 games often see the "
              "most significant changes thereof."));
    INSERT(Settings, async_presentation, tr("Enable asynchronous presentation (Vulkan only)"),
           tr("Slightly improves performance by moving presentation to a separate CPU thread."));
    INSERT(
        Settings, renderer_force_max_clock, tr("Force maximum clocks (Vulkan only)"),
        tr("Runs work in the background while waiting for graphics commands to keep the GPU from "
           "lowering its clock speed."));
    INSERT(Settings, max_anisotropy, tr("Anisotropic Filtering:"),
           tr("Controls the quality of texture rendering at oblique angles.\nSafe to set at 16x on "
              "most GPUs."));
    INSERT(Settings, gpu_accuracy, tr("GPU Mode:"),
           tr("Controls the GPU emulation mode.\nMost games render fine with Fast, but Accurate is still "
              "required for some.\nParticles tend to only render correctly with Accurate mode."));
    INSERT(Settings, dma_accuracy, tr("DMA Accuracy:"),
           tr("Controls the DMA read mode.\nUnsafe is faster, while Safe is more stable and can fix issues in some games.\nDefault follows the GPU Accuracy setting."));
    INSERT(Settings, gpu_fence_behavior, tr("GPU Fence Behavior:"),
           tr("Controls the GPU fence synchronization behavior.\nImmediate is the fastest option, but can introduce some issues.\nBalanced offers better compatibility and may fix issues in some games.\nAccurate further improves compatibility at the cost of some performance.\nDefault follows the GPU Mode setting."));
    INSERT(Settings, native_compute_write_mark, tr("Native compute write mark:"),
           tr("How much of a Switch 2 compute shader's storage buffer is treated as written by "
              "the GPU after each dispatch.\nAuto marks what the shader's stores can reach and "
              "falls back to Full when that cannot be proven.\nFull marks the whole binding (up to "
              "8 MiB), the original behaviour: it can delay the game's own writes to nearby memory "
              "(flickering effects) and copy far more data back from the GPU."));
    INSERT(Settings, use_asynchronous_shaders, tr("Enable asynchronous shader compilation"),
           tr("May reduce shader stutter."));
    INSERT(Settings, gpu_clock, tr("GPU Clocks"),
           tr("Makes the game believe GPU work finishes faster than it does, so it stops lowering "
              "resolution and render distance to fit the Switch's clocks."));
    INSERT(Settings, gpu_unswizzle_enabled, tr("GPU Unswizzle"),
           tr("Accelerates BCn 3D texture decoding using GPU compute.\n"
              "Disable if experiencing crashes or graphical glitches."));
    INSERT(Settings, gpu_unswizzle_texture_size, tr("GPU Unswizzle Max Texture Size"),
           tr("Sets the maximum size (MiB) for GPU-based texture unswizzling.\n"
              "While the GPU is faster for medium and large textures, the CPU may be more "
              "efficient for very small ones.\n"
              "Adjust this to find the balance between GPU acceleration and CPU overhead."));
    INSERT(Settings, gpu_unswizzle_stream_size, tr("GPU Unswizzle Stream Size"),
           tr("Sets the maximum amount of texture data (in MiB) processed per frame.\n"
              "Higher values can reduce stutter during texture loading but may impact frame "
              "consistency."));
    INSERT(Settings, gpu_unswizzle_chunk_size, tr("GPU Unswizzle Chunk Size"),
           tr("Determines the number of depth slices processed in a single dispatch.\n"
              "Increasing this can improve throughput on high-end GPUs but may cause TDR or driver "
              "timeouts on weaker hardware."));
    INSERT(Settings, pin_gpu_threads, tr("Pin GPU threads to their own cores"),
           tr("Pins the emulator's GPU threads to dedicated physical CPU cores and keeps the "
              "emulated CPU cores off them.\n"
              "Can hurt performance on CPUs with few cores. Skipped on CPUs with fewer than 6 "
              "physical performance cores.\n"
              "Windows and Linux only. Takes effect when a game starts."));

    INSERT(Settings, use_vulkan_driver_pipeline_cache, tr("Use Vulkan pipeline cache"),
           tr("Enables GPU vendor-specific pipeline cache.\nThis option can improve shader loading "
              "time significantly in cases where the Vulkan driver does not store pipeline cache "
              "files internally."));
    INSERT(Settings, precache_shaders, tr("Precache shaders before game start:"),
           tr("Builds shaders on the loading screen, on every CPU thread, so they don't stutter "
              "the first time the game uses them: the compute shaders found in the game files. "
              "The first launch reads the game files and takes a few minutes; later launches "
              "only build what is new, until the GPU driver changes. The loading screen has a "
              "Skip button.\nAsk asks at launch while the game has never been precached."));
    INSERT(Settings, build_all_cached_shaders, tr("Build all cached shaders before game start"),
           tr("Builds every shader saved by earlier play sessions on the loading screen and keeps "
              "them built, so places you visited long ago don't stutter. Without it, only the "
              "shaders of recent sessions are built there and the rest in the background or when "
              "the game first needs them.\nLoading takes longer and uses more memory, about 0.3 "
              "MB per shader, up to a quarter of the RAM. Takes effect when a game starts."));
    INSERT(Settings, native_async_draws, tr("Switch 2 draws with unbuilt pipelines:"),
           tr("Only applies when asynchronous shader building is on. Decides what a Switch 2 draw "
              "does when its graphics pipeline is still building.\n"
              "Wait stalls the GPU until the pipeline is ready (no missing effects, but long "
              "stalls on new content). Skip drops the draw, which can leave an effect missing for "
              "a few frames. Bounded waits up to the time below per frame, then skips. The "
              "EDEN_NX2_ASYNC_DISCARD=1 environment variable still forces Skip."));
    INSERT(Settings, native_pipeline_wait_ms, tr("Pipeline wait per frame (ms):"),
           tr("Used by the Bounded mode: the most time per frame that Switch 2 draws may wait in "
              "total for pipelines that are still building. Once spent, the remaining unbuilt "
              "draws of that frame are skipped."));
    INSERT(Settings, enable_compute_pipelines, tr("Enable Compute Pipelines (Intel Vulkan Only)"),
           tr("Required by some games.\nThis setting only exists for Intel "
              "proprietary drivers and may crash if enabled.\nCompute pipelines are always enabled "
              "on all other drivers."));
    INSERT(
        Settings, use_reactive_flushing, tr("Enable Reactive Flushing"),
        tr("Uses reactive flushing instead of predictive flushing, allowing more accurate memory "
           "syncing."));
    INSERT(Settings, use_video_framerate, tr("Sync to framerate of video playback"),
           tr("Run the game at normal speed during video playback, even when the framerate is "
              "unlocked."));
    INSERT(Settings, mkw_30fps, tr("Mario Kart World: 30 FPS mode"),
           tr("Forces the game to use its native 30 FPS mode at normal game speed. "
              "Takes effect after restarting the game."));
    INSERT(Settings, mkw_60fps, tr("Mario Kart World: 60 FPS mode"),
           tr("Forces the game's 60 FPS mode everywhere, not only in cutscenes. The game "
              "advances one 60 FPS step per frame, so it never runs fast. Takes effect after "
              "restarting the game and takes precedence over the 30 FPS mode."));
    INSERT(Settings, mkw_60fps_fallback, tr("Mario Kart World: drop to 30 FPS when slow"),
           tr("Only applies with the 60 FPS mode. When your PC can't hold 60 FPS, the game "
              "switches to its 30 FPS mode and keeps normal speed, then tries 60 again every few "
              "seconds. Turn it off to stay at 60 FPS, where slow frames play in slow motion. "
              "Takes effect after restarting the game."));
    INSERT(Settings, mkw_dynamic_fps, tr("Mario Kart World: dynamic FPS (experimental)"),
           tr("Lets the game adjust between 30 and 60 FPS using whole-frame timing to maintain "
              "normal game speed. Takes effect after restarting the game and takes precedence over "
              "the 30 FPS mode."));
    INSERT(Settings, hyrule_dynamic_fps,
           tr("Hyrule Warriors: Age of Imprisonment: dynamic FPS (experimental)"),
           tr("Drops to 30 FPS with the game's split-screen time step while the game runs below "
              "about 50 FPS, so heavy slowdowns don't slow down the game, and tries 60 FPS again "
              "every few seconds."));
    INSERT(Settings, hyrule_30fps, tr("Hyrule Warriors: Age of Imprisonment: 30 FPS mode"),
           tr("Always runs at 30 FPS with the game's split-screen time step, at normal game speed. "
              "Takes precedence over dynamic FPS."));
    INSERT(Settings, barrier_feedback_loops, tr("Barrier feedback loops"),
           tr("Improves rendering of transparency effects in specific games."));
    INSERT(Settings, enable_buffer_history, tr("Enable buffer history"),
           tr("Enables access to previous buffer states.\nThis option may improve rendering "
              "quality and performance consistency in some games."));
    INSERT(Settings, fix_bloom_effects, tr("Fix bloom effects"), tr("Removes bloom in Burnout."));

    // Renderer (Extensions)
    INSERT(Settings, dyna_state, tr("Extended Dynamic State"),
           tr("Controls the number of features that can be used in Extended Dynamic State.\n"
              "Higher states allow for more features and can increase performance, but may cause "
              "additional graphical issues."));

    INSERT(Settings, vertex_input_dynamic_state, tr("Vertex Input Dynamic State"),
           tr("Enables vertex input dynamic state feature for better quality and performance."));

    INSERT(
        Settings, sample_shading, tr("Sample Shading"),
        tr("Allows the fragment shader to execute per sample in a multi-sampled fragment "
           "instead of once per fragment. Improves graphics quality at the cost of performance.\n"
           "Higher values improve quality but degrade performance."));

    // Renderer (Debug)

    // System
    INSERT(Settings, rng_seed, tr("RNG Seed"),
           tr("Controls the seed of the random number generator.\nMainly used for speedrunning."));
    INSERT(Settings, rng_seed_enabled, QString(), QString());
    INSERT(Settings, device_name, tr("Device Name"), tr("The name of the console."));
    INSERT(Settings, program_args, tr("Homebrew Args"),
           tr("Command-line arguments passed to homebrew at launch (e.g. -noglsl)."));
    INSERT(Settings, custom_rtc, tr("Custom RTC Date:"),
           tr("This option allows to change the clock of the console.\n"
              "Can be used to manipulate time in games."));
    INSERT(Settings, custom_rtc_enabled, QString(), QString());
    INSERT(Settings, custom_rtc_offset, QStringLiteral(" "),
           tr("The number of seconds from the current unix time"));
    INSERT(Settings, language_index, tr("Language:"),
           tr("This option can be overridden when region setting is auto-select"));
    INSERT(Settings, region_index, tr("Region:"), tr("The region of the console."));
    INSERT(Settings, time_zone_index, tr("Time Zone:"), tr("The time zone of the console."));
    INSERT(Settings, sound_index, tr("Sound Output Mode:"), QString());
    INSERT(Settings, use_docked_mode, tr("Console Mode:"),
           tr("Selects if the console is in Docked or Handheld mode.\nGames will change "
              "their resolution, details and supported controllers and depending on this setting.\n"
              "Setting to Handheld can help improve performance for low end systems."));
    INSERT(Settings, current_user, QString(), QString());

    // Controls

    // Data Storage

    // Debugging

    // Debugging Graphics

    // Network

    // Web Service

    // Ui

    // Ui General
    INSERT(UISettings, select_user_on_boot, tr("Prompt for user profile on boot"),
           tr("Useful if multiple people use the same PC."));
    INSERT(UISettings, pause_when_in_background, tr("Pause when not in focus"),
           tr("Pauses emulation when focusing on other windows."));
    INSERT(UISettings, confirm_before_stopping, tr("Confirm before stopping emulation"),
           tr("Overrides prompts asking to confirm stopping the emulation.\nEnabling "
              "it bypasses such prompts and directly exits the emulation."));
    INSERT(UISettings, hide_mouse, tr("Hide mouse on inactivity"),
           tr("Hides the mouse after 2.5s of inactivity."));
    INSERT(UISettings, check_for_updates, tr("Check for updates"),
           tr("Whether or not to check for updates upon startup."));

    // Linux
    INSERT(UISettings, enable_gamemode, tr("Enable Gamemode"), QString());
#ifdef __unix__
    INSERT(UISettings, gui_force_x11, tr("Force X11 as Graphics Backend"), QString());
    INSERT(UISettings, gui_hide_backend_warning, QString(), QString());
#endif

    // Ui Debugging

    // Ui Multiplayer

    // Ui Games list

#undef INSERT

    return translations;
}

std::unique_ptr<ComboboxTranslationMap> ComboboxEnumeration(QObject* parent) {
    std::unique_ptr<ComboboxTranslationMap> translations =
        std::make_unique<ComboboxTranslationMap>();
    const auto& tr = [](const char* text, const char* context = "") {
        return QCoreApplication::translate("ConfigurationShared", text, context);
    };

#define PAIR(ENUM, VALUE, TRANSLATION) {static_cast<u32>(Settings::ENUM::VALUE), (TRANSLATION)}

    // Intentionally skipping VSyncMode to let the UI fill that one out
    translations->insert({Settings::EnumMetadata<Settings::AppletMode>::Index(),
                          {
                              PAIR(AppletMode, HLE, tr("Custom frontend")),
                              PAIR(AppletMode, LLE, tr("Real applet")),
                              PAIR(AppletMode, Disabled, tr("Disabled")),
                          }});

    translations->insert({Settings::EnumMetadata<Settings::SpirvOptimizeMode>::Index(),
                          {
                              PAIR(SpirvOptimizeMode, Never, tr("Never")),
                              PAIR(SpirvOptimizeMode, OnLoad, tr("On Load")),
                              PAIR(SpirvOptimizeMode, Always, tr("Always")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::NvidiaDlssQuality>::Index(),
                          {PAIR(NvidiaDlssQuality, Automatic, tr("Automatic (game resolution)")),
                           PAIR(NvidiaDlssQuality, Native, tr("Native (DLAA)")),
                           PAIR(NvidiaDlssQuality, Quality, tr("Quality")),
                           PAIR(NvidiaDlssQuality, Balanced, tr("Balanced")),
                           PAIR(NvidiaDlssQuality, Performance, tr("Performance")),
                           PAIR(NvidiaDlssQuality, UltraPerformance, tr("Ultra Performance"))}});
    translations->insert(
        {Settings::EnumMetadata<Settings::NvidiaDlssPreset>::Index(),
         {PAIR(NvidiaDlssPreset, M, tr("M (DLSS 4.5, default)")),
          PAIR(NvidiaDlssPreset, NvidiaDefault, tr("NVIDIA default")),
          PAIR(NvidiaDlssPreset, E, tr("E (legacy CNN)")),
          PAIR(NvidiaDlssPreset, F, tr("F (legacy CNN)")),
          PAIR(NvidiaDlssPreset, J, tr("J (DLSS 4)")), PAIR(NvidiaDlssPreset, K, tr("K (DLSS 4)")),
          PAIR(NvidiaDlssPreset, L, tr("L (DLSS 4.5)"))}});
    translations->insert({Settings::EnumMetadata<Settings::AstcDecodeMode>::Index(),
                          {
                              PAIR(AstcDecodeMode, Cpu, tr("CPU")),
                              PAIR(AstcDecodeMode, Gpu, tr("GPU")),
                              PAIR(AstcDecodeMode, CpuAsynchronous, tr("CPU Asynchronous")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::AstcRecompression>::Index(),
         {
             PAIR(AstcRecompression, Uncompressed, tr("Uncompressed (Best quality)")),
             PAIR(AstcRecompression, Bc1, tr("BC1 (Low quality)")),
             PAIR(AstcRecompression, Bc3, tr("BC3 (Medium quality)")),
         }});
    translations->insert({Settings::EnumMetadata<Settings::FramePacingMode>::Index(),
                          {
                              PAIR(FramePacingMode, Target_Auto, tr("Auto")),
                              PAIR(FramePacingMode, Target_30, tr("30 FPS")),
                              PAIR(FramePacingMode, Target_60, tr("60 FPS")),
                              PAIR(FramePacingMode, Target_90, tr("90 FPS")),
                              PAIR(FramePacingMode, Target_120, tr("120 FPS")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::VramUsageMode>::Index(),
                          {
                              PAIR(VramUsageMode, Conservative, tr("Conservative")),
                              PAIR(VramUsageMode, Aggressive, tr("Aggressive")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::RendererBackend>::Index(),
         {PAIR(RendererBackend, Vulkan, tr("Vulkan")),
#ifdef HAS_OPENGL
          PAIR(RendererBackend, OpenGL_GLSL, tr("OpenGL GLSL")),
          PAIR(RendererBackend, OpenGL_GLASM, tr("OpenGL GLASM (Assembly Shaders, NVIDIA Only)")),
          PAIR(RendererBackend, OpenGL_SPIRV, tr("OpenGL SPIR-V (Experimental, AMD/Mesa Only)")),
#endif
          PAIR(RendererBackend, Null, tr("Null"))}});
    translations->insert({Settings::EnumMetadata<Settings::GpuAccuracy>::Index(),
                          {
                              PAIR(GpuAccuracy, Low, tr("Fast")),
                              PAIR(GpuAccuracy, High, tr("Accurate")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::DmaAccuracy>::Index(),
                          {
                              PAIR(DmaAccuracy, Default, tr("Default")),
                              PAIR(DmaAccuracy, Unsafe, tr("Unsafe (fast)")),
                              PAIR(DmaAccuracy, Safe, tr("Safe (stable)")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::NativeComputeWriteMark>::Index(),
                          {
                              PAIR(NativeComputeWriteMark, Auto, tr("Auto (shader extent)")),
                              PAIR(NativeComputeWriteMark, Full, tr("Full binding")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::GpuFenceBehavior>::Index(),
                          {
                              PAIR(GpuFenceBehavior, Default, tr("Default")),
                              PAIR(GpuFenceBehavior, Immediate, tr("Immediate")),
                              PAIR(GpuFenceBehavior, Balanced, tr("Balanced")),
                              PAIR(GpuFenceBehavior, Accurate, tr("Accurate")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::CpuAccuracy>::Index(),
         {
             PAIR(CpuAccuracy, Auto, tr("Auto")),
             PAIR(CpuAccuracy, Accurate, tr("Accurate")),
             PAIR(CpuAccuracy, Unsafe, tr("Unsafe")),
             PAIR(CpuAccuracy, Paranoid, tr("Paranoid (disables most optimizations)")),
             PAIR(CpuAccuracy, Debugging, tr("Debugging")),
         }});
    translations->insert({Settings::EnumMetadata<Settings::CpuBackend>::Index(),
                          {
                              PAIR(CpuBackend, Dynarmic, tr("Dynarmic")),
                              PAIR(CpuBackend, Nce, tr("NCE")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::FullscreenMode>::Index(),
                          {
                              PAIR(FullscreenMode, Borderless, tr("Borderless Windowed")),
                              PAIR(FullscreenMode, Exclusive, tr("Exclusive Fullscreen")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::NvdecEmulation>::Index(),
                          {
                              PAIR(NvdecEmulation, Off, tr("No Video Output")),
                              PAIR(NvdecEmulation, Cpu, tr("CPU Video Decoding")),
                              PAIR(NvdecEmulation, Gpu, tr("GPU Video Decoding (Default)")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::ResolutionSetup>::Index(),
         {
             PAIR(ResolutionSetup, Res1_4X, tr("0.25X (180p/270p) [EXPERIMENTAL]")),
             PAIR(ResolutionSetup, Res1_2X, tr("0.5X (360p/540p) [EXPERIMENTAL]")),
             PAIR(ResolutionSetup, Res3_4X, tr("0.75X (540p/810p) [EXPERIMENTAL]")),
             PAIR(ResolutionSetup, Res1X, tr("1X (720p/1080p)")),
             PAIR(ResolutionSetup, Res5_4X, tr("1.25X (900p/1350p) [EXPERIMENTAL]")),
             PAIR(ResolutionSetup, Res3_2X, tr("1.5X (1080p/1620p) [EXPERIMENTAL]")),
             PAIR(ResolutionSetup, Res2X, tr("2X (1440p/2160p)")),
             PAIR(ResolutionSetup, Res3X, tr("3X (2160p/3240p)")),
             PAIR(ResolutionSetup, Res4X, tr("4X (2880p/4320p)")),
             PAIR(ResolutionSetup, Res5X, tr("5X (3600p/5400p)")),
             PAIR(ResolutionSetup, Res6X, tr("6X (4320p/6480p)")),
             PAIR(ResolutionSetup, Res7X, tr("7X (5040p/7560p)")),
             PAIR(ResolutionSetup, Res8X, tr("8X (5760p/8640p)")),
         }});
    translations->insert({Settings::EnumMetadata<Settings::FrameGenType>::Index(),
                          {
                              PAIR(FrameGenType, Lsfg, tr("Lossless Scaling (LSFG)")),
                              PAIR(FrameGenType, Fsr3, tr("Optical Flow Interpolation (built-in)")),
                              PAIR(FrameGenType, NvidiaDlssG, tr("NVIDIA DLSS Frame Generation")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::ScalingFilter>::Index(),
                          {
                              PAIR(ScalingFilter, NearestNeighbor, tr("Nearest Neighbor")),
                              PAIR(ScalingFilter, Bilinear, tr("Bilinear")),
                              PAIR(ScalingFilter, Bicubic, tr("Bicubic")),
                              PAIR(ScalingFilter, Gaussian, tr("Gaussian")),
                              PAIR(ScalingFilter, Lanczos, tr("Lanczos")),
                              PAIR(ScalingFilter, ScaleForce, tr("ScaleForce")),
                              PAIR(ScalingFilter, Fsr, tr("AMD FidelityFX Super Resolution 1 (FSR 1)")),
                              PAIR(ScalingFilter, Area, tr("Area")),
                              PAIR(ScalingFilter, Mmpx, tr("MMPX")),
                              PAIR(ScalingFilter, ZeroTangent, tr("Zero-Tangent")),
                              PAIR(ScalingFilter, BSpline, tr("B-Spline")),
                              PAIR(ScalingFilter, Mitchell, tr("Mitchell")),
                              PAIR(ScalingFilter, Spline1, tr("Spline-1")),
                              PAIR(ScalingFilter, Sgsr, tr("Snapdragon Game Super Resolution")),
                              PAIR(ScalingFilter, SgsrEdge, tr("Snapdragon Game Super Resolution EdgeDir")),
                              PAIR(ScalingFilter, Fsr3, tr("AMD FidelityFX Super Resolution 1, softer sharpening")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::AntiAliasing>::Index(),
                          {
                              PAIR(AntiAliasing, None, tr("None")),
                              PAIR(AntiAliasing, Fxaa, tr("FXAA")),
                              PAIR(AntiAliasing, Smaa, tr("SMAA")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::OotAntiAliasing>::Index(),
                          {
                              PAIR(OotAntiAliasing, Game, tr("Game")),
                              PAIR(OotAntiAliasing, NoSharpening, tr("No sharpening")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::HdrMode>::Index(),
                          {
                              PAIR(HdrMode, Off, tr("Off")),
                              PAIR(HdrMode, Auto, tr("Auto")),
                              PAIR(HdrMode, On, tr("On")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::HdrSwapchain>::Index(),
                          {
                              PAIR(HdrSwapchain, Hdr10, tr("HDR10 (10-bit)")),
                              PAIR(HdrSwapchain, Scrgb, tr("scRGB (16-bit float)")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::AspectRatio>::Index(),
                          {
                              PAIR(AspectRatio, R16_9, tr("Default (16:9)")),
                              PAIR(AspectRatio, R4_3, tr("Force 4:3")),
                              PAIR(AspectRatio, R21_9, tr("Force 21:9")),
                              PAIR(AspectRatio, R16_10, tr("Force 16:10")),
                              PAIR(AspectRatio, Stretch, tr("Stretch to Window")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::AnisotropyMode>::Index(),
                          {
                              PAIR(AnisotropyMode, Automatic, tr("Automatic")),
                              PAIR(AnisotropyMode, Default, tr("Default")),
                              PAIR(AnisotropyMode, X2, tr("2x")),
                              PAIR(AnisotropyMode, X4, tr("4x")),
                              PAIR(AnisotropyMode, X8, tr("8x")),
                              PAIR(AnisotropyMode, X16, tr("16x")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::Language>::Index(),
         {
             PAIR(Language, Japanese, tr("Japanese (日本語)")),
             PAIR(Language, EnglishAmerican, tr("American English")),
             PAIR(Language, French, tr("French (français)")),
             PAIR(Language, German, tr("German (Deutsch)")),
             PAIR(Language, Italian, tr("Italian (italiano)")),
             PAIR(Language, Spanish, tr("Spanish (español)")),
             PAIR(Language, Chinese, tr("Chinese")),
             PAIR(Language, Korean, tr("Korean (한국어)")),
             PAIR(Language, Dutch, tr("Dutch (Nederlands)")),
             PAIR(Language, Portuguese, tr("Portuguese (português)")),
             PAIR(Language, Russian, tr("Russian (Русский)")),
             PAIR(Language, Taiwanese, tr("Taiwanese")),
             PAIR(Language, EnglishBritish, tr("British English")),
             PAIR(Language, FrenchCanadian, tr("Canadian French")),
             PAIR(Language, SpanishLatin, tr("Latin American Spanish")),
             PAIR(Language, ChineseSimplified, tr("Simplified Chinese")),
             PAIR(Language, ChineseTraditional, tr("Traditional Chinese (正體中文)")),
             PAIR(Language, PortugueseBrazilian, tr("Brazilian Portuguese (português do Brasil)")),
             PAIR(Language, Polish, tr("Polish (polska)")),
             PAIR(Language, Thai, tr("Thai (แบบไทย)")),
         }});
    translations->insert({Settings::EnumMetadata<Settings::Region>::Index(),
                          {
                              PAIR(Region, Japan, tr("Japan")),
                              PAIR(Region, Usa, tr("USA")),
                              PAIR(Region, Europe, tr("Europe")),
                              PAIR(Region, Australia, tr("Australia")),
                              PAIR(Region, China, tr("China")),
                              PAIR(Region, Korea, tr("Korea")),
                              PAIR(Region, Taiwan, tr("Taiwan")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::TimeZone>::Index(),
         {
             {static_cast<u32>(Settings::TimeZone::Auto),
              tr("Auto (%1)", "Auto select time zone")
                  .arg(QString::fromStdString(
                      Settings::GetTimeZoneString(Settings::TimeZone::Auto)))},
             {static_cast<u32>(Settings::TimeZone::Default),
              tr("Default (%1)", "Default time zone")
                  .arg(QString::fromStdString(Common::TimeZone::GetDefaultTimeZone()))},
             PAIR(TimeZone, Cet, tr("CET")),
             PAIR(TimeZone, Cst6Cdt, tr("CST6CDT")),
             PAIR(TimeZone, Cuba, tr("Cuba")),
             PAIR(TimeZone, Eet, tr("EET")),
             PAIR(TimeZone, Egypt, tr("Egypt")),
             PAIR(TimeZone, Eire, tr("Eire")),
             PAIR(TimeZone, Est, tr("EST")),
             PAIR(TimeZone, Est5Edt, tr("EST5EDT")),
             PAIR(TimeZone, Gb, tr("GB")),
             PAIR(TimeZone, GbEire, tr("GB-Eire")),
             PAIR(TimeZone, Gmt, tr("GMT")),
             PAIR(TimeZone, GmtPlusZero, tr("GMT+0")),
             PAIR(TimeZone, GmtMinusZero, tr("GMT-0")),
             PAIR(TimeZone, GmtZero, tr("GMT0")),
             PAIR(TimeZone, Greenwich, tr("Greenwich")),
             PAIR(TimeZone, Hongkong, tr("Hongkong")),
             PAIR(TimeZone, Hst, tr("HST")),
             PAIR(TimeZone, Iceland, tr("Iceland")),
             PAIR(TimeZone, Iran, tr("Iran")),
             PAIR(TimeZone, Israel, tr("Israel")),
             PAIR(TimeZone, Jamaica, tr("Jamaica")),
             PAIR(TimeZone, Japan, tr("Japan")),
             PAIR(TimeZone, Kwajalein, tr("Kwajalein")),
             PAIR(TimeZone, Libya, tr("Libya")),
             PAIR(TimeZone, Met, tr("MET")),
             PAIR(TimeZone, Mst, tr("MST")),
             PAIR(TimeZone, Mst7Mdt, tr("MST7MDT")),
             PAIR(TimeZone, Navajo, tr("Navajo")),
             PAIR(TimeZone, Nz, tr("NZ")),
             PAIR(TimeZone, NzChat, tr("NZ-CHAT")),
             PAIR(TimeZone, Poland, tr("Poland")),
             PAIR(TimeZone, Portugal, tr("Portugal")),
             PAIR(TimeZone, Prc, tr("PRC")),
             PAIR(TimeZone, Pst8Pdt, tr("PST8PDT")),
             PAIR(TimeZone, Roc, tr("ROC")),
             PAIR(TimeZone, Rok, tr("ROK")),
             PAIR(TimeZone, Singapore, tr("Singapore")),
             PAIR(TimeZone, Turkey, tr("Turkey")),
             PAIR(TimeZone, Uct, tr("UCT")),
             PAIR(TimeZone, Universal, tr("Universal")),
             PAIR(TimeZone, Utc, tr("UTC")),
             PAIR(TimeZone, WSu, tr("W-SU")),
             PAIR(TimeZone, Wet, tr("WET")),
             PAIR(TimeZone, Zulu, tr("Zulu")),
         }});
    translations->insert({Settings::EnumMetadata<Settings::AudioMode>::Index(),
                          {
                              PAIR(AudioMode, Mono, tr("Mono")),
                              PAIR(AudioMode, Stereo, tr("Stereo")),
                              PAIR(AudioMode, Surround, tr("Surround")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::MemoryLayout>::Index(),
                          {
                              PAIR(MemoryLayout, Memory_4Gb, tr("4GB DRAM (Default)")),
                              PAIR(MemoryLayout, Memory_6Gb, tr("6GB DRAM (Unsafe)")),
                              PAIR(MemoryLayout, Memory_8Gb, tr("8GB DRAM")),
                              PAIR(MemoryLayout, Memory_10Gb, tr("10GB DRAM (Unsafe)")),
                              PAIR(MemoryLayout, Memory_12Gb, tr("12GB DRAM (Unsafe)")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::ConsoleMode>::Index(),
                          {
                              PAIR(ConsoleMode, Docked, tr("Docked")),
                              PAIR(ConsoleMode, Handheld, tr("Handheld")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::PrecacheShaders>::Index(),
                          {
                              PAIR(PrecacheShaders, Off, tr("Off")),
                              PAIR(PrecacheShaders, Ask, tr("Ask")),
                              PAIR(PrecacheShaders, Always, tr("Always")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::NativeAsyncDraws>::Index(),
                          {
                              PAIR(NativeAsyncDraws, Wait, tr("Wait")),
                              PAIR(NativeAsyncDraws, Bounded, tr("Bounded wait")),
                              PAIR(NativeAsyncDraws, Skip, tr("Skip")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::CpuClock>::Index(),
                          {
                              PAIR(CpuClock, Normal, tr("Normal")),
                              PAIR(CpuClock, Boost, tr("Boost")),
                              PAIR(CpuClock, Overclock, tr("Overclock")),
                          }});
    translations->insert(
        {Settings::EnumMetadata<Settings::ConfirmStop>::Index(),
         {
             PAIR(ConfirmStop, Ask_Always, tr("Always ask (Default)")),
             PAIR(ConfirmStop, Ask_Based_On_Game, tr("Only if game specifies not to stop")),
             PAIR(ConfirmStop, Ask_Never, tr("Never ask")),
         }});
    translations->insert({Settings::EnumMetadata<Settings::GpuClock>::Index(),
                          {
                              PAIR(GpuClock, Normal, tr("Normal")),
                              PAIR(GpuClock, Boost, tr("Boost")),
                              PAIR(GpuClock, Overclock, tr("Overclock")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::GpuUnswizzleSize>::Index(),
                          {
                              PAIR(GpuUnswizzleSize, VerySmall, tr("Very Small (16 MB)")),
                              PAIR(GpuUnswizzleSize, Small, tr("Small (32 MB)")),
                              PAIR(GpuUnswizzleSize, Normal, tr("Normal (128 MB)")),
                              PAIR(GpuUnswizzleSize, Large, tr("Large (256 MB)")),
                              PAIR(GpuUnswizzleSize, VeryLarge, tr("Very Large (512 MB)")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::GpuUnswizzle>::Index(),
                          {
                              PAIR(GpuUnswizzle, VeryLow, tr("Very Low (4 MB)")),
                              PAIR(GpuUnswizzle, Low, tr("Low (8 MB)")),
                              PAIR(GpuUnswizzle, Normal, tr("Normal (16 MB)")),
                              PAIR(GpuUnswizzle, Medium, tr("Medium (32 MB)")),
                              PAIR(GpuUnswizzle, High, tr("High (64 MB)")),
                          }});
    translations->insert({Settings::EnumMetadata<Settings::GpuUnswizzleChunk>::Index(),
                          {
                              PAIR(GpuUnswizzleChunk, VeryLow, tr("Very Low (32)")),
                              PAIR(GpuUnswizzleChunk, Low, tr("Low (64)")),
                              PAIR(GpuUnswizzleChunk, Normal, tr("Normal (128)")),
                              PAIR(GpuUnswizzleChunk, Medium, tr("Medium (256)")),
                              PAIR(GpuUnswizzleChunk, High, tr("High (512)")),
                          }});

    translations->insert({Settings::EnumMetadata<Settings::ExtendedDynamicState>::Index(),
                          {
                              PAIR(ExtendedDynamicState, Disabled, tr("Disabled")),
                              PAIR(ExtendedDynamicState, EDS1, tr("ExtendedDynamicState 1")),
                              PAIR(ExtendedDynamicState, EDS2, tr("ExtendedDynamicState 2")),
                              PAIR(ExtendedDynamicState, EDS3, tr("ExtendedDynamicState 3")),
                          }});

    translations->insert({Settings::EnumMetadata<Settings::GameListMode>::Index(),
                          {
                              PAIR(GameListMode, TreeView, tr("Tree View")),
                              PAIR(GameListMode, GridView, tr("Grid View")),
                          }});

#undef PAIR
#undef CTX_PAIR

    return translations;
}
} // namespace ConfigurationShared
