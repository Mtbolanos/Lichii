// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <vector>
#include <optional>
#include <bit>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/memory_detect.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/conversion.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/texture_cache/util.h"
#include "video_core/nx2.h"
#include "video_core/churn_counters.h"
#include "video_core/nx2_profile.h"
#include "video_core/render_trace.h"
#include "video_core/textures/decoders.h"

namespace VideoCommon {

/// EDEN_NATIVE_DRAW_DIAGNOSTICS is read once: getenv scans the whole environment block on Windows.
[[nodiscard]] inline bool NativeDrawDiagnostics() {
    static const bool enabled = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
    return enabled;
}

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceType;
using namespace Common::Literals;

/// Nanoseconds since `start` (the render target statistics of the slow clear log).
[[nodiscard]] inline u64 NsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

template <class P>
TextureCache<P>::TextureCache(Runtime& runtime_, Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : runtime{runtime_}, device_memory{device_memory_} {
    // Configure null sampler
    TSCEntry sampler_descriptor{};
    sampler_descriptor.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    sampler_descriptor.cubemap_anisotropy.Assign(1);

    // These values were chosen based on typical peak swizzle data sizes seen in some titles
    constexpr size_t SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 8_MiB;
    constexpr size_t UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 1_MiB;
    swizzle_data_buffer.resize_destructive(SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);
    unswizzle_data_buffer.resize_destructive(UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);

    // Make sure the first index is reserved for the null resources
    // This way the null resource becomes a compile time constant
    void(slot_images.insert(NullImageParams{}));
    void(slot_image_views.insert(runtime, NullImageViewParams{}));
    void(slot_samplers.insert(runtime, sampler_descriptor));

    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
        const s64 min_spacing_expected = device_local_memory - 1_GiB;
        const s64 min_spacing_critical = device_local_memory - 512_MiB;
        const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        expected_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                     DEFAULT_EXPECTED_MEMORY));
        critical_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                     DEFAULT_CRITICAL_MEMORY));
        minimum_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    } else {
        expected_memory = DEFAULT_EXPECTED_MEMORY + 512_MiB;
        critical_memory = DEFAULT_CRITICAL_MEMORY + 1_GiB;
        minimum_memory = 0;
    }
    ApplyThresholdOverrides();

    const bool gpu_unswizzle_enabled = Settings::values.gpu_unswizzle_enabled.GetValue();

    if (gpu_unswizzle_enabled) {
        switch (Settings::values.gpu_unswizzle_texture_size.GetValue()) {
            case Settings::GpuUnswizzleSize::VerySmall:    gpu_unswizzle_maxsize = 16_MiB; break;
            case Settings::GpuUnswizzleSize::Small:        gpu_unswizzle_maxsize = 32_MiB; break;
            case Settings::GpuUnswizzleSize::Normal:       gpu_unswizzle_maxsize = 128_MiB; break;
            case Settings::GpuUnswizzleSize::Large:        gpu_unswizzle_maxsize = 256_MiB; break;
            case Settings::GpuUnswizzleSize::VeryLarge:    gpu_unswizzle_maxsize = 512_MiB; break;
            default:                                       gpu_unswizzle_maxsize = 128_MiB; break;
        }

        switch (Settings::values.gpu_unswizzle_stream_size.GetValue()) {
            case Settings::GpuUnswizzle::VeryLow: swizzle_chunk_size = 4_MiB; break;
            case Settings::GpuUnswizzle::Low:     swizzle_chunk_size = 8_MiB; break;
            case Settings::GpuUnswizzle::Normal:  swizzle_chunk_size = 16_MiB; break;
            case Settings::GpuUnswizzle::Medium:  swizzle_chunk_size = 32_MiB; break;
            case Settings::GpuUnswizzle::High:    swizzle_chunk_size = 64_MiB; break;
            default:                              swizzle_chunk_size = 16_MiB;
        }

        switch (Settings::values.gpu_unswizzle_chunk_size.GetValue()) {
            case Settings::GpuUnswizzleChunk::VeryLow: swizzle_slices_per_batch = 32; break;
            case Settings::GpuUnswizzleChunk::Low:     swizzle_slices_per_batch = 64; break;
            case Settings::GpuUnswizzleChunk::Normal:  swizzle_slices_per_batch = 128; break;
            case Settings::GpuUnswizzleChunk::Medium:  swizzle_slices_per_batch = 256; break;
            case Settings::GpuUnswizzleChunk::High:    swizzle_slices_per_batch = 512; break;
            default:                                   swizzle_slices_per_batch = 128;
        }
    } else {
        gpu_unswizzle_maxsize = 0;
        swizzle_chunk_size = 0;
        swizzle_slices_per_batch = 0;
    }
}

template <class P>
void TextureCache<P>::RunGarbageCollector(bool out_of_memory) {
    bool high_priority_mode = false;
    bool aggressive_mode = false;
    bool evict_old_costly = false;
    u64 ticks_to_destroy = 0;
    size_t num_iterations = 0;
    const auto Configure = [&](bool allow_aggressive) {
        high_priority_mode = out_of_memory || total_used_memory >= expected_memory;
        aggressive_mode =
            allow_aggressive && (out_of_memory || total_used_memory >= critical_memory);
        ticks_to_destroy = aggressive_mode ? 10ULL : high_priority_mode ? 25ULL : 50ULL;
        // EDEN_NX2_TC_MIN_AGE=<frames>: never evict images used more recently (thrash tests).
        static const u64 min_age = [] {
            const char* value = std::getenv("EDEN_NX2_TC_MIN_AGE");
            return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
        }();
        ticks_to_destroy = (std::max)(ticks_to_destroy, min_age);
        num_iterations = aggressive_mode ? 40 : (high_priority_mode ? 20 : 10);
    };
    const auto Cleanup = [this, &num_iterations, &high_priority_mode, &aggressive_mode,
                          &evict_old_costly, out_of_memory](ImageId image_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        auto& image = slot_images[image_id];
        if (True(image.flags & ImageFlagBits::IsDecoding)) {
            return false;
        }
        const bool must_download = IsDownloadable(image) && False(image.flags & ImageFlagBits::BadOverlap);
        if ((!aggressive_mode && !evict_old_costly && True(image.flags & ImageFlagBits::CostlyLoad)) ||
            (!high_priority_mode && must_download)) {
            return false;
        }
        // EDEN_NX2_EXP57 (live, 0 = off): at most one download with a full wait for the GPU per
        // frame unless an allocation failed. A player's first high-priority pass downloaded
        // image after image and froze for 3 s (fable52_answer_3 s7); the rest wait for later
        // frames.
        if (must_download && !out_of_memory && VideoCore::Nx2Debug::Experiment(57) != 0 &&
            gc_waited_frame == frame_tick) {
            ++gc_downloads_deferred;
            return false;
        }
        if (must_download) {
            gc_waited_frame = frame_tick;
            auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
            const auto copies = FullDownloadCopies(image.info);
            image.DownloadMemory(map, copies);
            runtime.Finish();
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span, swizzle_data_buffer);
            ++gc_download_deletes;
            // Guest memory holds the GPU's data now (DeleteImage must not mark it stale).
            image.flags &= ~ImageFlagBits::GpuModified;
            ClearStaleImage(image);
        } else {
            MarkStale(image); // before UntrackImage (see MarkStale)
        }
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        ++(True(image.flags & ImageFlagBits::CostlyLoad) ? gc_costly_deletes : gc_deletes);
        UnregisterImage(image_id);
        DeleteImage(image_id, image.scale_tick > frame_tick + 5);
        if (out_of_memory) {
            return false;
        }
        if (aggressive_mode && total_used_memory < critical_memory) {
            num_iterations >>= 2;
            aggressive_mode = false;
        } else if (high_priority_mode && total_used_memory < expected_memory) {
            num_iterations >>= 1;
            high_priority_mode = false;
        }
        return false;
    };
    // Deleting by age alone left every device memory block partly used: the collector deletes
    // thousands of images a minute and new ones land in the holes, so blocks seldom empty (title
    // 1: ~0.8 GB of holes in ~190 32 MiB blocks after ten minutes, which the driver counts as
    // used). Among the oldest candidates (8 per image this pass may delete), those in the
    // emptiest blocks go first, so blocks drain and go back to the driver; placement already
    // prefers the fullest blocks (best fit). Images with memory of their own sit in the middle.
    const auto Collect = [&] {
        if constexpr (requires(const Image& image) { image.MemoryBlock(); }) {
            std::vector<ImageId> candidates;
            const size_t limit = num_iterations * 8;
            lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](ImageId image_id) {
                candidates.push_back(image_id);
                return candidates.size() >= limit;
            });
            if (candidates.empty()) {
                return;
            }
            if (block_image_bytes_frame == ~0ULL || frame_tick - block_image_bytes_frame >= 60) {
                block_image_bytes_frame = frame_tick;
                block_image_bytes.clear();
                for (const auto [image_id, image] : slot_images) {
                    const auto block = image->MemoryBlock();
                    if (block.block != 0) {
                        block_image_bytes[block.block] += block.size;
                    }
                }
            }
            std::vector<std::pair<double, ImageId>> ordered;
            ordered.reserve(candidates.size());
            for (const ImageId image_id : candidates) {
                const auto block = slot_images[image_id].MemoryBlock();
                double fill = 0.5;
                if (block.block != 0 && block.block_size != 0) {
                    const auto it = block_image_bytes.find(block.block);
                    fill = it != block_image_bytes.end()
                               ? static_cast<double>(it->second) / static_cast<double>(block.block_size)
                               : 0.0;
                }
                ordered.emplace_back(fill, image_id);
            }
            std::stable_sort(ordered.begin(), ordered.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [fill, image_id] : ordered) {
                if (Cleanup(image_id)) {
                    break;
                }
            }
        } else {
            lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, Cleanup);
        }
    };
    Configure(false);
    static const u64 old_age = [] {
        const char* value = std::getenv("EDEN_NX2_TC_OLD_AGE");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 1800ULL;
    }();
    if (high_priority_mode && old_age != 0 && frame_tick > old_age) {
        evict_old_costly = true;
        lru_cache.ForEachItemBelow(frame_tick - old_age, Cleanup);
        evict_old_costly = false;
    }
    Collect();
    if (out_of_memory || total_used_memory >= critical_memory) {
        Configure(true);
        Collect();
    }
}

template <class P>
void TextureCache<P>::UpdateStaleTracking(u64 run_begin, u64 run_pages, s32 delta) {
    if (run_pages != 0) {
        device_memory.UpdatePagesCachedCount(run_begin << STALE_PAGE_BITS,
                                             run_pages << STALE_PAGE_BITS, delta);
    }
}

template <class P>
void TextureCache<P>::MarkStale(const ImageBase& image) {
    if (!tc_skip_stale || False(image.flags & ImageFlagBits::GpuModified) ||
        True(image.flags & (ImageFlagBits::Sparse | ImageFlagBits::Unbacked)) ||
        image.guest_size_bytes == 0) {
        return;
    }
    // Whole pages only: a page the image shares with other data keeps that data's state.
    constexpr u64 page_size = u64{1} << STALE_PAGE_BITS;
    const u64 first = (image.cpu_addr + page_size - 1) >> STALE_PAGE_BITS;
    const u64 end = (image.cpu_addr + image.guest_size_bytes) >> STALE_PAGE_BITS;
    if (end <= first || end > (u64{1} << 28)) {
        return;
    }
    if (stale_bits.size() * 64 < end) {
        stale_bits.resize((end + 63) / 64);
    }
    // Tracked from here on so CPU writes reach WriteMemory. Marking before the image's own
    // tracking goes keeps the pages' write protection from switching off and on again.
    u64 run_begin = 0;
    u64 run_pages = 0;
    for (u64 page = first; page < end; ++page) {
        u64& word = stale_bits[page / 64];
        const u64 bit = u64{1} << (page % 64);
        if ((word & bit) != 0) {
            UpdateStaleTracking(run_begin, run_pages, 1);
            run_pages = 0;
            continue;
        }
        word |= bit;
        ++stale_page_count;
        if (run_pages == 0) {
            run_begin = page;
        }
        ++run_pages;
    }
    UpdateStaleTracking(run_begin, run_pages, 1);
}

template <class P>
void TextureCache<P>::ClearStale(DAddr addr, size_t size) {
    if (stale_page_count == 0 || size == 0) {
        return;
    }
    constexpr u64 page_size = u64{1} << STALE_PAGE_BITS;
    const u64 first = addr >> STALE_PAGE_BITS;
    const u64 end = (std::min<u64>)((addr + size + page_size - 1) >> STALE_PAGE_BITS,
                                    stale_bits.size() * 64);
    u64 run_begin = 0;
    u64 run_pages = 0;
    for (u64 page = first; page < end;) {
        u64& word = stale_bits[page / 64];
        if (word == 0 && page % 64 == 0) {
            UpdateStaleTracking(run_begin, run_pages, -1);
            run_pages = 0;
            page += 64;
            continue;
        }
        const u64 bit = u64{1} << (page % 64);
        if ((word & bit) != 0) {
            word &= ~bit;
            --stale_page_count;
            if (run_pages == 0) {
                run_begin = page;
            }
            ++run_pages;
        } else {
            UpdateStaleTracking(run_begin, run_pages, -1);
            run_pages = 0;
        }
        ++page;
    }
    UpdateStaleTracking(run_begin, run_pages, -1);
}

template <class P>
void TextureCache<P>::ClearStaleImage(const ImageBase& image) {
    if (stale_page_count == 0) {
        return;
    }
    if (False(image.flags & ImageFlagBits::Sparse)) {
        ClearStale(image.cpu_addr, image.guest_size_bytes);
        return;
    }
    ClearStaleGpu(image.gpu_addr, image.guest_size_bytes);
}

template <class P>
void TextureCache<P>::ClearStaleGpu(GPUVAddr gpu_addr, size_t size) {
    if (stale_page_count == 0) {
        return;
    }
    // Each piece is contiguous in device memory (GetSubmappedRange splits where it is not).
    for (const auto& [piece_addr, piece_size] : gpu_memory->GetSubmappedRange(gpu_addr, size)) {
        if (const auto device_addr = gpu_memory->GpuToCpuAddress(piece_addr)) {
            ClearStale(*device_addr, piece_size);
        }
    }
}

template <class P>
bool TextureCache<P>::IsAllStale(DAddr addr, size_t size) const {
    if (stale_page_count == 0 || size == 0) {
        return false;
    }
    constexpr u64 page_size = u64{1} << STALE_PAGE_BITS;
    const u64 first = addr >> STALE_PAGE_BITS;
    const u64 end = (addr + size + page_size - 1) >> STALE_PAGE_BITS;
    if (end > stale_bits.size() * 64) {
        return false;
    }
    for (u64 page = first; page < end;) {
        const u64 word = stale_bits[page / 64];
        if (page % 64 == 0 && page + 64 <= end) {
            if (word != ~u64{0}) {
                return false;
            }
            page += 64;
            continue;
        }
        if (((word >> (page % 64)) & 1) == 0) {
            return false;
        }
        ++page;
    }
    return true;
}

template <class P>
void TextureCache<P>::ApplyThresholdOverrides(bool log) {
    // Past what a frame needs, more device memory only keeps the textures of earlier scenes
    // (title 1 filled 6.7 GB of an 8 GB card and draws its races in ~5 GB), so collect from
    // MaxCriticalMemory on rather than near the card's budget. Larger cards get 5/8 of the
    // driver's budget (GetDeviceLocalMemory is capped near 7 GiB): at 5 GiB title 1's races
    // recreate and upload ~8 MiB of render targets a frame (16 GB card: 53 -> 59 fps at 10 GiB).
    // 5/8 of the budget only passes 5 GiB above an 8 GiB budget, so cards up to 8 GB (and
    // integrated GPUs, whose budget is capped at 4-6 GiB) keep the 5 GiB cap.
    u64 max_critical = MaxCriticalMemory;
    if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
        const u64 budget = runtime.GetDeviceMemoryBudget();
        max_critical = (std::max)(max_critical, budget / 8 * 5);
        // Cards with 10 GiB or more of budget: 5/8 left ~7.3 GiB on a 12 GB RX 6700 XT, under what
        // Ocarina's Kokiri Forest keeps alive, so every aggressive pass evicted its CostlyLoad ASTC
        // textures (a 6144x6144 5x5 one, 31 MiB) and the game re-created them ~128 frames later:
        // a 24 ms texture stall every ~3 s. Keep 2.5 GiB of the budget for everything else instead.
        if (budget >= 10_GiB) {
            max_critical = (std::max)(max_critical, budget - 2560_MiB);
        }
    }
    critical_memory = (std::min)(critical_memory, max_critical);
    expected_memory =
        (std::min)(expected_memory, critical_memory - (std::min)(critical_memory, u64{768_MiB}));
    minimum_memory =
        (std::min)(minimum_memory, expected_memory - (std::min)(expected_memory, u64{1_GiB}));
    // EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB override the collector's thresholds.
    for (const auto& [name, target] : {std::pair{"EDEN_NX2_TC_EXPECTED_MIB", &expected_memory},
                                       std::pair{"EDEN_NX2_TC_CRITICAL_MIB", &critical_memory}}) {
        if (const char* value = std::getenv(name)) {
            *target = std::strtoull(value, nullptr, 10) << 20;
        }
    }
    if (log) {
        LOG_INFO(HW_GPU, "Texture cache thresholds: expected {} MiB, critical {} MiB, minimum {} MiB",
                 expected_memory >> 20, critical_memory >> 20, minimum_memory >> 20);
    }
}

template <class P>
void TextureCache<P>::TickFrame() {
    // Upload buffers grown for one huge image (title 1's 341 MiB terrain layer pools) are freed
    // rather than kept for the rest of the session.
    for (auto* const scratch : {&swizzle_data_buffer, &unswizzle_data_buffer}) {
        if (scratch->capacity() > LargeScratchCapacity) {
            scratch->release();
        }
    }
    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        // Switch 2 titles (native tiling) stream far more texture data than Eden's thresholds
        // were tuned for: title 1's attract loop alone keeps ~5.3 GB resident, so the collector
        // sat in aggressive mode evicting decoded ASTC (and decoding it again). Total use is
        // measured from the driver, so collect only near the device budget instead.
        if (!native_thresholds_applied && gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
            native_thresholds_applied = true;
            const u64 device_local = runtime.GetDeviceLocalMemory();
            critical_memory = (std::max)(critical_memory, device_local - 256_MiB);
            expected_memory = (std::max)(expected_memory, device_local - 1280_MiB);
            ApplyThresholdOverrides();
            native_critical_memory = critical_memory;
            native_expected_memory = expected_memory;
        }
        // Follow the driver's budget (re-read now and then): it shrinks when other programs use
        // VRAM (another game: 7.0 -> 6.4 GB), and collecting only past it means the driver pages
        // textures out. It is also the room to use: Eden's device memory figure is capped near
        // 7 GB, which on a 12 GB card (11.2 GB budget) kept title 1 evicting textures with 3 GB
        // unused. Both thresholds stay under it with a margin (1/8 of it, at most 1 GiB): past it
        // allocations fail and images end up unbacked (garbage textures), and Windows shrinks the
        // budget as the card fills (16 GB card: 13.5 GB left mid-race), hence the re-read every 30
        // frames. EDEN_NX2_TC_*_MIB overrides still win.
        if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
            if constexpr (requires { runtime.MemoryPressureEvents(); }) {
                // Images that did not fit (made in host memory or with nothing bound) mean the
                // thresholds sit too high for this card and driver: leave more room until the
                // pressure has been gone for a while.
                if (native_thresholds_applied) {
                    const u32 events = runtime.MemoryPressureEvents();
                    if (events != seen_pressure_events) {
                        seen_pressure_events = events;
                        pressure_margin = (std::min)(pressure_margin + 512_MiB, 3_GiB);
                        pressure_quiet_frames = 0;
                        budget_check_frames = 0; // thresholds are recomputed below
                        LOG_WARNING(HW_GPU, "Out of device memory: keeping {} MiB more free",
                                    pressure_margin >> 20);
                    } else if (pressure_margin != 0 && ++pressure_quiet_frames % 600 == 0) {
                        pressure_margin -= (std::min)(pressure_margin, 128_MiB);
                        budget_check_frames = 0;
                    }
                }
            }
            if (native_thresholds_applied && budget_check_frames++ % 30 == 0) {
                const u64 budget = runtime.GetDeviceMemoryBudget();
                if (budget > 2_GiB) {
                    const u64 old_critical = critical_memory;
                    const u64 margin = (std::min)(budget / 8, 1_GiB);
                    if constexpr (requires { runtime.GetDeviceMemorySlack(); }) {
                        // The usage counted here leaves out the unused part of the allocator's
                        // blocks, which the driver counts and a full heap cannot hand to a new
                        // image (RX 7800 XT: 1.06 GiB of it, the whole margin, with the heap at
                        // its budget for a minute before images stopped fitting). Reserve it
                        // too: at once when it grows, slowly when it shrinks, so evicting images
                        // (which leaves partly used blocks) does not chase the threshold down.
                        const u64 slack = (std::min)(runtime.GetDeviceMemorySlack(), 1536_MiB);
                        slack_margin = slack >= slack_margin
                                           ? slack
                                           : slack_margin - (std::min)(slack_margin - slack, 16_MiB);
                    }
                    critical_memory =
                        budget - (std::min)(margin + pressure_margin + slack_margin, budget / 2);
                    expected_memory = critical_memory - (std::min)(margin, critical_memory / 2);
                    ApplyThresholdOverrides(false);
                    // Log the first values and real changes (the budget moves by a few MiB all
                    // the time). Compared after the overrides: the value before them is above
                    // the 5/8-of-budget cap on large cards, so every check looked like a change
                    // and logged the same thresholds every 30 frames.
                    const u64 change = critical_memory > old_critical
                                           ? critical_memory - old_critical
                                           : old_critical - critical_memory;
                    const bool log = budget_check_frames == 1 || change >= 256_MiB;
                    if (log) {
                        LOG_INFO(HW_GPU,
                                 "Texture cache thresholds: expected {} MiB, critical {} MiB, "
                                 "minimum {} MiB",
                                 expected_memory >> 20, critical_memory >> 20,
                                 minimum_memory >> 20);
                        LOG_INFO(HW_GPU,
                                 "Memory margins: base {} MiB, out of memory {} MiB, block slack {} MiB",
                                 margin >> 20, pressure_margin >> 20, slack_margin >> 20);
                    }
                }
            }
        }
    }
    if constexpr (requires { runtime.CurrentTick(); }) {
        while (!gpu_sentenced_images.empty() &&
               runtime.IsFree(gpu_sentenced_images.front().tick)) {
            // EXP42: the GPU is done with it, so its VkImage may go to the runtime's pool for
            // the next image of its shape (fable_answer_12 2B).
            if constexpr (requires(Image& image) { image.ReleaseToPool(); }) {
                gpu_sentenced_images.front().image.ReleaseToPool();
            }
            gpu_sentenced_bytes -= gpu_sentenced_images.front().bytes;
            gpu_sentenced_images.pop_front();
        }
    }
    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
        // Images the collector deleted keep their memory for TICKS_TO_DESTROY frames: counting
        // it kept the collector in aggressive mode, deleting 40+ images a frame for frames after
        // it had freed enough (title 1: frames over 100 ms for seconds at a time).
        for (const u64 bytes : sentenced_image_bytes) {
            total_used_memory -= (std::min)(total_used_memory, bytes);
        }
        total_used_memory -= (std::min)(total_used_memory, gpu_sentenced_bytes);
        // EXP42's pooled images are deleted images too: the pool caps and ages them itself.
        if constexpr (requires { runtime.PooledImageBytes(); }) {
            total_used_memory -= (std::min)(total_used_memory, runtime.PooledImageBytes());
        }
    }
    // EDEN_NX2_MEM_TRACE=1 (from zortos293/Lichii#951): every 2 s, what the cache holds next to
    // what the process and the driver report (memory climbs).
    static const bool mem_trace = std::getenv("EDEN_NX2_MEM_TRACE") != nullptr;
    if (const auto now = std::chrono::steady_clock::now();
        mem_trace && now - mem_trace_time >= std::chrono::seconds{2}) {
        mem_trace_time = now;
        u64 image_bytes{};
        u64 gpu_written_bytes{};
        size_t images{};
        size_t unbacked{};
        for (const auto [image_id, image] : slot_images) {
            ++images;
            if (True(image->flags & ImageFlagBits::Unbacked)) {
                ++unbacked;
                continue;
            }
            const u64 bytes = GetScaledImageSizeBytes(*image);
            image_bytes += bytes;
            if (True(image->flags & ImageFlagBits::GpuModified)) {
                gpu_written_bytes += bytes;
            }
        }
        u64 decode_bytes{};
        for (const auto& decode : async_decodes) {
            // The worker sizes the buffer before it publishes complete.
            if (decode->complete.load(std::memory_order_acquire)) {
                decode_bytes += decode->decoded_data.capacity();
            }
        }
        const u64 sentenced = std::accumulate(sentenced_image_bytes.begin(),
                                              sentenced_image_bytes.end(), gpu_sentenced_bytes);
        LOG_INFO(HW_GPU,
                 "Memory trace: texture cache {} images {} MiB ({} MiB GPU-written, {} unbacked), "
                 "device usage {} MiB (expected {} MiB, critical {} MiB), sentenced {} MiB, {} "
                 "async decodes {} MiB, scratch {} MiB",
                 images, image_bytes >> 20, gpu_written_bytes >> 20, unbacked,
                 total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20,
                 sentenced >> 20, async_decodes.size(), decode_bytes >> 20,
                 (swizzle_data_buffer.capacity() + unswizzle_data_buffer.capacity()) >> 20);
        if constexpr (requires { runtime.LogMemoryTrace(); }) {
            runtime.LogMemoryTrace();
        }
    }
    // Every 600 frames: device memory in use against the collector's thresholds.
    // Skip expensive heap allocation, map construction, and sorting unless debug logging is explicitly enabled.
    static const bool log_tc = std::getenv("EDEN_NX2_TC_LOG") != nullptr;
    static u64 frames{};
    if (log_tc && ++frames % 600 == 0) {
        u64 image_bytes{};
        size_t images{};
        std::vector<std::pair<u64, const Image*>> largest;
        std::map<u32, std::pair<u64, u32>> by_format;
        for (const auto [image_id, image] : slot_images) {
            const u64 bytes = GetScaledImageSizeBytes(*image);
            image_bytes += bytes;
            ++images;
            if (debug_stats) {
                largest.emplace_back(bytes, image);
                auto& format = by_format[static_cast<u32>(image->info.format)];
                format.first += bytes;
                ++format.second;
            }
        }
        LOG_INFO(HW_GPU, "Texture cache: device memory {} MiB (expected {} MiB, critical {} MiB), {} images {} MiB",
                 total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20, images,
                 image_bytes >> 20);
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(12, largest.size()),
                          largest.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min<size_t>(12, largest.size()); ++i) {
            const Image& image = *largest[i].second;
            LOG_DEBUG(HW_GPU, "  {} MiB: {:#x} {}x{}x{} layers {} levels {} format {} type {} flags {:#x}",
                     largest[i].first >> 20, image.gpu_addr, image.info.size.width, image.info.size.height,
                     image.info.size.depth, image.info.resources.layers, image.info.resources.levels,
                     static_cast<u32>(image.info.format), static_cast<u32>(image.info.type),
                     static_cast<u32>(image.flags));
        }
        std::vector<std::pair<u64, u32>> formats;
        for (const auto& [format, total] : by_format) formats.emplace_back(total.first, format);
        std::sort(formats.rbegin(), formats.rend());
        for (size_t i = 0; i < std::min<size_t>(8, formats.size()); ++i) {
            LOG_DEBUG(HW_GPU, "  format {}: {} MiB in {} images", formats[i].second, formats[i].first >> 20,
                     by_format[formats[i].second].second);
        }
        if constexpr (requires { runtime.LogMemoryStatistics(); }) {
            runtime.LogMemoryStatistics();
        }
        LOG_INFO(HW_GPU, "Texture cache last 600 frames: created {} images, deleted {} (collector {}, {} costly, {} downloaded, {} downloads deferred), {} async decodes, ASTC {} opaque / {} alpha",
                 images_created, images_deleted, gc_deletes + gc_costly_deletes, gc_costly_deletes,
                 gc_download_deletes, gc_downloads_deferred,
                 async_decodes_queued, astc_opaque_images, astc_alpha_images);
        LOG_INFO(HW_GPU, "Texture cache last 600 frames: {} uploads from guest memory, {} MiB (largest {} MiB, {} by layer)",
                 images_uploaded, uploaded_bytes >> 20, largest_upload >> 20, layer_uploads);
        LOG_INFO(HW_GPU,
                 "Texture cache last 600 frames: stale uploads skipped {} ({} MiB), {} uploaded "
                 "for buffer-written pages; {} MiB of pages marked stale",
                 tc_stale_skips, tc_stale_skip_bytes >> 20, tc_stale_gpu_written,
                 (stale_page_count << STALE_PAGE_BITS) >> 20);
        tc_stale_skips = tc_stale_skip_bytes = tc_stale_gpu_written = 0;
        images_uploaded = uploaded_bytes = largest_upload = layer_uploads = 0;
        astc_opaque_images = astc_alpha_images = 0;
        std::vector<std::pair<u32, u64>> shapes;
        for (const auto& [key, value] : created_shapes) shapes.emplace_back(value.first, key);
        std::sort(shapes.rbegin(), shapes.rend());
        for (size_t i = 0; i < std::min<size_t>(10, shapes.size()); ++i) {
            const u64 key = shapes[i].second;
            LOG_DEBUG(HW_GPU, "  created {}x: format {} type {} levels {} layers {} {}x{} (last at {:#x})",
                     shapes[i].first, key >> 56, (key >> 52) & 0xF, (key >> 48) & 0xF,
                     (key >> 36) & 0xFFF, (key >> 18) & 0x3FFFF, key & 0x3FFFF,
                     created_shapes[key].second);
        }
        created_shapes.clear();
        // The per-image and per-shape tallies only feed debug lines: skip them while those are off.
        debug_stats = Common::Log::IsLevelEnabled(Common::Log::Class::HW_GPU, Common::Log::Level::Debug);
        gc_deletes = gc_costly_deletes = gc_download_deletes = gc_downloads_deferred =
            async_decodes_queued = 0;
        images_created = images_deleted = 0;
    }
    SweepColdImages();
    // An image that could not be made at all: the driver is out of memory whatever the counts
    // say, so collect as at the critical threshold.
    bool out_of_memory = false;
    if constexpr (requires { runtime.FailedImages(); }) {
        const u32 failed = runtime.FailedImages();
        out_of_memory = failed != seen_failed_images;
        seen_failed_images = failed;
    }
    if (out_of_memory || total_used_memory > minimum_memory) {
        RunGarbageCollector(out_of_memory);
    }
    if (out_of_memory) {
        // Textures left out for lack of memory are made again on their next read, once a frame
        // while images keep failing.
        for (size_t c : active_channel_ids) {
            channel_storage[c].graphics_image_table.Invalidate();
            channel_storage[c].compute_image_table.Invalidate();
        }
        // Texture memos of their null views would skip that read (EXP32, zortos293/Lichii#717).
        ++texture_memo_epoch;
    }
    sentenced_images.Tick();
    sentenced_image_index = (sentenced_image_index + 1) % TICKS_TO_DESTROY;
    sentenced_image_bytes[sentenced_image_index] = 0;
    sentenced_framebuffers.Tick();
    sentenced_image_view.Tick();
    TickAsyncDecode();
    TickAsyncUnswizzle();
    if (frame_tick % 600 == 0 && !uploaded_ranges.empty()) {
        // EDEN_NX2_EXP8 bit 8: none. An image created again over cold memory (re-created after
        // eviction, or a render target at a new size) faults all of it back in while uploading.
        if ((VideoCore::Nx2Debug::Experiment(8) & 8) == 0) {
            for (const auto& [address, size] : uploaded_ranges) {
                device_memory.HintCold(address, size);
            }
        }
        uploaded_ranges.clear();
    }

    runtime.TickFrame();
    ++frame_tick;
    // Prepared render targets skip the LRU touch of their images: keep them to one frame.
    ++bindings_epoch;
    if (VideoCore::Nx2Debug::Experiment(31) != 0 && frame_tick % 600 == 0) {
        LOG_INFO(HW_GPU,
                 "RT_FAST (EXP31) last 600 frames: {} of {} clean render target updates skipped; "
                 "verified {}, mismatches {}",
                 rt_fast_skips, rt_fast_clean_paths, rt_fast_verified, rt_fast_mismatches);
        rt_fast_skips = rt_fast_clean_paths = rt_fast_verified = rt_fast_mismatches = 0;
    }
    if (relocation_seen && frame_tick % 600 == 0) {
        u32 guarded_shapes = 0;
        for (const auto& [key, shape] : relocation_shapes) {
            guarded_shapes += shape.guard != 0 ? 1 : 0;
        }
        LOG_INFO(HW_GPU,
                 "Render target relocation (EXP43={}) last 600 frames: {} candidates, {} kept by "
                 "the guard, {} relocated, {} retired early, {} over 2+ images of their shape; "
                 "{} of {} shapes guarded",
                 VideoCore::Nx2Debug::Experiment(43), relocation_candidates, relocation_guarded,
                 relocation_relocated, relocation_retired, relocation_crowded, guarded_shapes,
                 relocation_shapes.size());
        relocation_candidates = relocation_guarded = relocation_relocated = relocation_retired =
            relocation_crowded = 0;
    }
    TickRenderClock();
    TickRelocationTombstones();
    // What the caches made and destroyed (fable_answer_16 4; counted always, logged on request).
    static const bool churn_log = [] {
        const char* value = std::getenv("EDEN_NX2_CHURN_LOG");
        return value != nullptr && value[0] == '1';
    }();
    if (churn_log && frame_tick % 600 == 0) {
        const auto churn = Churn::Take();
        LOG_INFO(HW_GPU,
                 "Churn (600 frames): images created {}, deleted {}, relocated {}, pooled {}; "
                 "image views created {}, destroyed {}; buffers created {}, deleted {}, joined {} "
                 "({} KiB copied; {} skipped as CPU-modified, {} KiB not copied); staging "
                 "buffers allocated {} ({} MiB); device memory "
                 "allocations {}, frees {}; descriptor pools created {}; {} images now",
                 churn[Churn::ImagesCreated], churn[Churn::ImagesDeleted],
                 churn[Churn::ImagesRelocated], churn[Churn::ImagesPooled],
                 churn[Churn::ImageViewsCreated], churn[Churn::ImageViewsDestroyed],
                 churn[Churn::BuffersCreated], churn[Churn::BuffersDeleted],
                 churn[Churn::BuffersJoined], churn[Churn::BufferJoinBytes] >> 10,
                 churn[Churn::BufferJoinsSkipped], churn[Churn::BufferJoinSkippedBytes] >> 10,
                 churn[Churn::StagingBuffers], churn[Churn::StagingBytes] >> 20,
                 churn[Churn::DeviceAllocations], churn[Churn::DeviceFrees],
                 churn[Churn::DescriptorPools], slot_images.size());
    }

    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        for (auto& buffer : async_buffers_death_ring) {
            runtime.FreeDeferredStagingBuffer(buffer);
        }
        async_buffers_death_ring.clear();
    }
}

/// EDEN_NX2_TC_COLD_FRAMES (the cold sweep's K, 0 = off).
inline u64 TcColdFrames() {
    static const u64 frames = [] {
        // 4 by default since 2026-11-08 (device memory 8.8 -> ~6.4 GB at 2 min in Castle Town,
        // 11 loads 0 washed out); EDEN_NX2_TC_COLD_FRAMES=0 turns it off.
        const char* value = std::getenv("EDEN_NX2_TC_COLD_FRAMES");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 4ULL;
    }();
    return frames;
}

template <class P>
void TextureCache<P>::WatchColdImage(ImageId image_id) {
    if (TcColdFrames() == 0 || !RelocationShaped(slot_images[image_id].info)) {
        return;
    }
    if (cold_watch_set.insert(image_id).second) {
        cold_watch.emplace_back(image_id, frame_tick);
    }
}

template <class P>
bool TextureCache<P>::ColdSuperseded(ImageId image_id, const ImageBase& image) {
    const u64 shape = RelocationShapeKey(image.info);
    const u64 touched = lru_cache.Tick(image.lru_index);
    bool superseded = false;
    ForEachImageInRegion(image.cpu_addr, image.guest_size_bytes,
                         [&](ImageId other_id, Image& other) {
                             if (other_id == image_id ||
                                 False(other.flags & ImageFlagBits::Registered) ||
                                 RelocationShapeKey(other.info) == shape ||
                                 lru_cache.Tick(other.lru_index) <= touched) {
                                 return false;
                             }
                             superseded = true;
                             return true;
                         });
    return superseded;
}

template <class P>
void TextureCache<P>::SweepColdImages() {
    const u64 cold_frames = TcColdFrames();
    if (cold_frames == 0) {
        return;
    }
    static const bool sweep_guarded = [] {
        const char* value = std::getenv("EDEN_NX2_TC_COLD_GUARDED");
        return value != nullptr && value[0] == '1';
    }();
    constexpr u32 MaxPerFrame = 16;
    constexpr u64 MaxWatchFrames = 600; // then a watched image that never got BadOverlap leaves
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::Converted |
        ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding | ImageFlagBits::Remapped |
        ImageFlagBits::CostlyLoad;
    const auto queued = [](ImageId image_id, const auto& downloads) {
        return std::ranges::any_of(downloads, [image_id](const auto& download) {
            return download.is_swizzle && download.object_id == image_id;
        });
    };
    const auto bound = [this](const ImageBase& image) {
        return std::ranges::any_of(image.image_view_ids, [this](ImageViewId view) {
            return render_targets.depth_buffer_id == view ||
                   std::ranges::find(render_targets.color_buffer_ids, view) !=
                       render_targets.color_buffer_ids.end();
        });
    };
    u32 deleted = 0;
    size_t index = 0;
    while (index < cold_watch.size()) {
        const ImageId image_id = cold_watch[index].first;
        const u64 watched_at = cold_watch[index].second;
        const auto drop = [&] {
            cold_watch_set.erase(image_id);
            cold_watch[index] = cold_watch.back();
            cold_watch.pop_back();
        };
        if (!slot_images.contains(image_id)) {
            drop();
            continue;
        }
        Image& image = slot_images[image_id];
        if (!RelocationShaped(image.info) || False(image.flags & ImageFlagBits::Registered) ||
            True(image.flags & excluded)) {
            drop();
            continue;
        }
        if (False(image.flags & ImageFlagBits::BadOverlap)) {
            // The newest image of a chain has one overlap; the next drift makes it bad.
            if (frame_tick - watched_at > MaxWatchFrames) {
                drop();
            } else {
                ++index;
            }
            continue;
        }
        // Touched (prepared, bound, sampled) in the last K host frames.
        if (frame_tick - lru_cache.Tick(image.lru_index) < cold_frames) {
            ++index;
            continue;
        }
        bool superseded = false; // EXP58 chose it
        constexpr u64 SupersededFrames = 60;
        // EDEN_NX2_EXP58=2 (live): images the CPU read once go the same way (below). Owner's
        // Castle Town market + shooting gallery session: 150-170k "kept as read later" a window,
        // ~6,000 images and device memory at the collector's line with EXP58=1.
        const int sweep_mode = VideoCore::Nx2Debug::Experiment(58);
        if (True(image.flags & ImageFlagBits::GpuModified)) {
            // Its contents exist only on the GPU and the collector would not download them
            // either (BadOverlap): only where nothing says they may be read again.
            if (image.relocation_guard_noted != 0 || image.info.forced_flushed ||
                image.info.dma_downloaded) {
                const u64 age = frame_tick - lru_cache.Tick(image.lru_index);
                if (sweep_mode >= 2 && age < SupersededFrames &&
                    frame_tick - watched_at <= MaxWatchFrames) {
                    ++index; // looked at again when old enough
                    continue;
                }
                // A newer target of another shape owns that memory now: a CPU read there goes
                // to it, not to this image.
                if (sweep_mode < 2 || age < SupersededFrames || !ColdSuperseded(image_id, image)) {
                    ++cold_kept_read;
                    drop();
                    continue;
                }
                superseded = true;
            }
            const auto shape = relocation_shapes.find(RelocationShapeKey(image.info));
            const u8 guard = shape != relocation_shapes.end() ? shape->second.guard : 0;
            if (!superseded &&
                ((guard & GuardCpuRead) != 0 ||
                 (!sweep_guarded && (guard & (GuardSampled | GuardLoaded)) != 0))) {
                // EDEN_NX2_EXP58 (live, 0 = off; fable52_answer_2 3.3 step 4): a guarded image
                // still dies once it has been untouched for SupersededFrames and a render target
                // of another shape, touched since, covers its memory: the game reused that memory
                // for another layout, so the old contents have no reader left. Deleted without a
                // download, as the collector would. The guard was made for relocation (same
                // shape, live contents); these piled up (350k "kept" in 600 frames) until the
                // collector's bursts. Shapes the CPU read only with EXP58=2.
                const bool superseded_sweep =
                    sweep_mode != 0 && ((guard & GuardCpuRead) == 0 || sweep_mode >= 2);
                const u64 age = frame_tick - lru_cache.Tick(image.lru_index);
                if (superseded_sweep && age < SupersededFrames &&
                    frame_tick - watched_at <= MaxWatchFrames) {
                    ++index; // looked at again when old enough
                    continue;
                }
                if (!superseded_sweep || age < SupersededFrames ||
                    !ColdSuperseded(image_id, image)) {
                    ++cold_kept_guarded;
                    drop();
                    continue;
                }
                superseded = true;
            }
        }
        if (queued(image_id, uncommitted_downloads) ||
            std::ranges::any_of(committed_downloads,
                                [&](const auto& downloads) { return queued(image_id, downloads); }) ||
            bound(image)) {
            ++index;
            continue;
        }
        if (deleted >= MaxPerFrame) {
            break;
        }
        // As the collector deletes an image it does not download.
        MarkStale(image); // before UntrackImage (see MarkStale)
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id);
        ++deleted;
        cold_superseded += superseded ? 1 : 0;
        drop();
    }
    cold_deleted += deleted;
    if (frame_tick % 600 == 0) {
        LOG_INFO(HW_GPU,
                 "Cold sweep (EDEN_NX2_TC_COLD_FRAMES={}{}) last 600 frames: {} chain images "
                 "deleted ({} of them guarded but superseded, EXP58), {} kept as GPU-written and "
                 "guarded, {} kept as read later; {} watched; {} images, device memory {} MiB "
                 "(expected {} MiB)",
                 cold_frames, sweep_guarded ? ", guarded too" : "", cold_deleted, cold_superseded,
                 cold_kept_guarded, cold_kept_read, cold_watch.size(), slot_images.size(),
                 total_used_memory >> 20, expected_memory >> 20);
        cold_deleted = cold_kept_guarded = cold_kept_read = cold_superseded = 0;
    }
}

template <class P>
void TextureCache<P>::TickRenderClock() {
    // EXP43=4 relocates under this clock (with tombstones).
    const bool enabled =
        Tegra::Nx2RenderClockEnv() != 0 || VideoCore::Nx2Debug::Experiment(43) == 4;
    if (enabled && !render_clock_enabled) {
        Tegra::g_nx2_frame_fence.Resume(); // nothing was noted while it was off
    }
    render_clock_enabled = enabled;
    Tegra::g_nx2_render_clock_wanted.store(enabled, std::memory_order_relaxed);
    const u64 counter = Tegra::g_nx2_render_frame.load(std::memory_order_relaxed);
    render_clock_idle = counter != render_clock_last_counter ? 0 : render_clock_idle + 1;
    render_clock_last_counter = counter;
    // Live while the counter advanced in this host frame or the one before (a count may land in
    // the next tick); two host frames without one and frame_tick takes over again.
    const bool live = enabled && counter != 0 && render_clock_idle < 2;
    if (live != render_clock_live) {
        // Continue from the current value, so the clock never goes back across a switch.
        render_clock_base = RenderFrame() + 1;
        render_clock_source_base = live ? counter : frame_tick;
        render_clock_live = live;
        ++render_clock_switches;
        if (live && !render_clock_logged) {
            render_clock_logged = true;
            LOG_INFO(HW_GPU,
                     "Render frame clock (EDEN_NX2_RENDER_CLOCK): live at host frame {} ({}); "
                     "last_write_frame, cross-frame reads and the same-frame rule follow it",
                     frame_tick,
                     Tegra::Nx2RenderClockSyncpt() >= 0
                         ? fmt::format("syncpoint {}", Tegra::Nx2RenderClockSyncpt())
                     : Tegra::Nx2RenderClockWordSource()
                         ? std::string{"the present poll's learned word"}
                         : std::string{"the flip's frame fence"});
        }
    }
    render_clock_live_frames += render_clock_live ? 1 : 0;
    if (frame_tick % 600 != 0 || (!enabled && !relocation_seen)) {
        return;
    }
    // The shapes refused most often as written this frame.
    std::vector<std::pair<u32, u64>> refused;
    for (auto& [key, shape] : relocation_shapes) {
        if (shape.same_frame_refused != 0) {
            refused.emplace_back(shape.same_frame_refused, key);
            shape.same_frame_refused = 0;
        }
    }
    const size_t top = (std::min)(refused.size(), size_t{3});
    std::partial_sort(refused.begin(), refused.begin() + top, refused.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string refused_text;
    for (size_t i = 0; i < top; ++i) {
        const u64 key = refused[i].second;
        refused_text += fmt::format("{}format {} {}x{} {}", i == 0 ? "" : ", ", key >> 56,
                                    (key >> 18) & 0x3FFFF, key & 0x3FFFF, refused[i].first);
    }
    // Syncpoints signalled in these frames, most first: one at about 1 per host frame may end
    // each frame (EDEN_NX2_RENDER_CLOCK_SYNCPT).
    std::string syncpt_text;
    if (enabled) {
        std::vector<std::pair<u32, u32>> signals;
        for (u32 id = 0; id < Tegra::Nx2SyncpointSignalSlots; ++id) {
            const u32 count =
                Tegra::g_nx2_syncpoint_signals[id].exchange(0, std::memory_order_relaxed);
            if (count != 0) {
                signals.emplace_back(count, id);
            }
        }
        std::sort(signals.rbegin(), signals.rend());
        for (size_t i = 0; i < (std::min)(signals.size(), size_t{4}); ++i) {
            syncpt_text += fmt::format(" {}:{:.2f}", signals[i].second, signals[i].first / 600.0);
        }
    }
    // The flip-fence source's confirmations (video_core/nx2_frame_fence.h): equal and ahead are
    // exact boundaries, late ones came at the flip (as frame_tick), early must stay 0.
    std::string fence_text;
    if (enabled && Tegra::Nx2RenderClockFlipSource()) {
        const auto fence = Tegra::g_nx2_frame_fence.TakeWindow();
        const auto per_frame = [](u32 count) { return count / 600.0; };
        fence_text = fmt::format(
            "; frame fence {} {}: {} flips ({:.2f} per host frame), {} equal, {} ahead, {} late "
            "({} merged), {} early{}; bumps {} armed, {} flip, {} late; arms refused {} old, {} "
            "not largest, {} reached, {} dropped, {} held, {} relearns",
            fence.syncpoint == ~0U ? std::string{"none"}
                                   : fmt::format("syncpoint {}", fence.syncpoint),
            fence.slot_key == ~0ULL ? std::string{"(no slot armed it)"}
                                    : fmt::format("(session {} slot {:#x})", fence.slot_key >> 32,
                                                  static_cast<u32>(fence.slot_key)),
            fence.flips, per_frame(fence.flips), fence.equal, fence.ahead, fence.late,
            fence.merged, fence.early, fence.fault ? ", armed source off" : "",
            fence.bumps_armed, fence.bumps_flip, fence.bumps_late, fence.refused_old,
            fence.refused_not_largest, fence.refused_reached, fence.dropped, fence.held,
            fence.relearns);
    }
    const u64 counted = counter - render_clock_window_counter;
    render_clock_window_counter = counter;
    LOG_INFO(HW_GPU,
             "Render frame clock (EDEN_NX2_RENDER_CLOCK={}) last 600 frames: {} render frames "
             "({:.2f} per host frame), live {} of 600 frames, {} switches; {} relocation "
             "candidates refused as written this frame{}{}; syncpoint signals per host frame:{}{}",
             enabled ? 1 : 0, counted, counted / 600.0, render_clock_live_frames,
             render_clock_switches, relocation_same_frame, top != 0 ? " (" : "",
             top != 0 ? refused_text + ")" : std::string{},
             syncpt_text.empty() ? std::string{" none"} : syncpt_text, fence_text);
    render_clock_live_frames = 0;
    render_clock_switches = 0;
    relocation_same_frame = 0;
}

template <class P>
const typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) const noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(u32 index) noexcept {
    // Not compute!
    const auto image_view_id = VisitImageView(index, false);
    return slot_image_views[image_view_id];
}

template <class P>
void TextureCache<P>::MarkModification(ImageId id) noexcept {
    MarkModification(slot_images[id]);
}

template <class P>
void TextureCache<P>::FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist) {
    bool has_blacklisted = false;
    do {
        has_deleted_images = false;
        if (blacklist) {
            has_blacklisted = false;
        }
        for (ImageViewInOut& view : views) {
            view.id = VisitImageView(view.index, compute);
            if (blacklist) {
                if (view.blacklist && view.id != NULL_IMAGE_VIEW_ID) {
                    const ImageViewBase& image_view = slot_image_views[view.id];
                    if (image_view.image_id == NULL_IMAGE_ID) {
                        continue;
                    }
                    auto& image = slot_images[image_view.image_id];
                    has_blacklisted |= ScaleDown(image);
                    image.scale_rating = 0;
                }
            }
        }
    } while (has_deleted_images || (blacklist && has_blacklisted));
}

template <class P>
void TextureCache<P>::CheckFeedbackLoop(std::span<const ImageViewInOut> views) {
    if (!Settings::values.barrier_feedback_loops.GetValue()) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    const bool writes_depth_stencil = regs.depth_write_enabled != 0 || regs.stencil_enable != 0;
    const auto barrier = [&] {
        if (feedback_depth_dirty) {
            runtime.BarrierFeedbackLoop();
            feedback_depth_dirty = false;
        }
    };
    SCOPE_EXIT {
        if (writes_depth_stencil) {
            feedback_depth_dirty = true;
        }
    };

    if (render_targets_serial == last_feedback_loop_serial &&
        texture_bindings_serial == last_feedback_texture_serial) {
        if (last_feedback_loop_result) {
            barrier();
        }
        return;
    }
    if (render_targets_serial != last_feedback_loop_serial) {
        feedback_depth_dirty = true;
    }

    if (rt_active_mask == 0) {
        last_feedback_loop_serial = render_targets_serial;
        last_feedback_texture_serial = texture_bindings_serial;
        last_feedback_loop_result = false;
        return;
    }
    const u32 depth_bit = 1u << NUM_RT;
    const bool depth_active = (rt_active_mask & depth_bit) != 0;

    const bool requires_barrier = [&] {
        for (const auto& view : views) {
            if (!view.id) {
                continue;
            }

            {
                bool is_continue = false;
                for (size_t i = 0; i < 8; ++i)
                    is_continue |= (rt_active_mask & (1u << i)) && view.id == render_targets.color_buffer_ids[i];
                if (is_continue)
                    continue;
            }

            if (depth_active && view.id == render_targets.depth_buffer_id)
                continue;

            const ImageId view_image_id = slot_image_views[view.id].image_id;
            {
                bool is_continue = false;
                for (size_t i = 0; i < 8; ++i)
                    is_continue |= (rt_active_mask & (1u << i)) && view_image_id == rt_image_id[i];
                if (is_continue)
                    continue;
            }
            if (depth_active && view_image_id == rt_depth_image_id) {
                return true;
            }
        }

        return false;
    }();

    last_feedback_loop_serial = render_targets_serial;
    last_feedback_texture_serial = texture_bindings_serial;
    last_feedback_loop_result = requires_barrier;
    if (requires_barrier) {
        barrier();
    }
}

template <class P>
bool TextureCache<P>::SamplesRenderTarget(std::span<const ImageViewInOut> views) {
    if (rt_active_mask == 0) {
        return false;
    }
    const bool depth_active = (rt_active_mask & (1u << NUM_RT)) != 0;
    for (const auto& view : views) {
        if (!view.id) {
            continue;
        }
        const ImageId image_id = slot_image_views[view.id].image_id;
        if (depth_active && image_id == rt_depth_image_id) {
            return true;
        }
        for (size_t i = 0; i < NUM_RT; ++i) {
            if ((rt_active_mask & (1u << i)) != 0 && image_id == rt_image_id[i]) {
                return true;
            }
        }
    }
    return false;
}

template <class P>
typename P::Sampler* TextureCache<P>::GetSampler(u32 index, bool compute) {
    return &slot_samplers[GetSamplerId(index, compute)];
}

template <class P>
SamplerId TextureCache<P>::GetSamplerId(u32 index, bool compute) {
    auto& table = compute ? channel_state->compute_sampler_table : channel_state->graphics_sampler_table;
    if (index > table.current_limit) {
        LOG_TRACE(HW_GPU, "Invalid sampler index={}", index);
        return NULL_SAMPLER_ID;
    }
    auto const [descriptor, is_new] = table.Read(*gpu_memory, index);
    if (is_new) {
        auto const id = FindSampler(descriptor, compute);
        table.bound_ids[index] = id;
        return id;
    }
    return table.bound_ids[index];
}

template <class P>
typename TextureCache<P>::TextureMemoState TextureCache<P>::GraphicsTextureMemoState()
    const noexcept {
    const auto& image_table = channel_state->graphics_image_table;
    const auto& sampler_table = channel_state->graphics_sampler_table;
    return TextureMemoState{
        .channel = channel_state,
        .memory = gpu_memory,
        .tic_pool = image_table.current_gpu_addr,
        .tsc_pool = sampler_table.current_gpu_addr,
        .tic_limit = image_table.current_limit,
        .tsc_limit = sampler_table.current_limit,
        .map_serial = gpu_memory->MapSerial(),
        .epoch = texture_memo_epoch,
    };
}

template <class P>
bool TextureCache<P>::TakeTextureMemoEntry(u32 tic_handle, u32 tsc_handle, ImageViewId view,
                                           SamplerId sampler, TextureMemoEntry& entry) const {
    // The entry must be in the table as this draw read it, and still give the resolved id
    // (as zortos293/Lichii#717's RecordBinding checks).
    const auto read_entry = [](const auto& table, u32 index, auto id) {
        return !table.invalidated && index / 64 < table.read_descriptors.size() &&
               index < table.descriptors.size() &&
               (table.read_descriptors[index / 64] & (1ULL << (index % 64))) != 0 &&
               table.bound_ids[index] == id;
    };
    const auto& image_table = channel_state->graphics_image_table;
    const auto& sampler_table = channel_state->graphics_sampler_table;
    const u32 header_index = Tegra::Texture::TextureHeaderIndex(
        tic_handle, image_table.current_limit, gpu_memory->UsesNativeTiling());
    if (header_index <= image_table.current_limit) {
        if (!read_entry(image_table, header_index, view)) {
            return false;
        }
        entry.tic_index = header_index;
        entry.tic = image_table.descriptors[header_index];
    } else if (view != NULL_IMAGE_VIEW_ID) {
        return false;
    } else {
        entry.tic_index = ~0U;
        entry.tic = TICEntry{};
    }
    if (tsc_handle <= sampler_table.current_limit) {
        if (!read_entry(sampler_table, tsc_handle, sampler)) {
            return false;
        }
        entry.tsc_index = tsc_handle;
        entry.tsc = sampler_table.descriptors[tsc_handle];
    } else if (sampler != NULL_SAMPLER_ID) {
        return false;
    } else {
        entry.tsc_index = ~0U;
        entry.tsc = TSCEntry{};
    }
    return true;
}

template <class P>
bool TextureCache<P>::TextureMemoEntryCurrent(const TextureMemoState& state,
                                              const TextureMemoEntry& entry) const {
    const auto same = [this](GPUVAddr gpu_addr, const auto& expected) {
        using T = std::remove_cvref_t<decltype(expected)>;
        T guest{};
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory->GetPointerForDescriptor(gpu_addr)
                                            : nullptr) {
            std::memcpy(&guest, host, sizeof(T));
        } else {
            gpu_memory->ReadBlockUnsafe(gpu_addr, &guest, sizeof(T));
        }
        return std::memcmp(&guest, &expected, sizeof(T)) == 0;
    };
    if (entry.tic_index != ~0U &&
        !same(state.tic_pool + u64{entry.tic_index} * sizeof(TICEntry), entry.tic)) {
        return false;
    }
    if (entry.tsc_index != ~0U &&
        !same(state.tsc_pool + u64{entry.tsc_index} * sizeof(TSCEntry), entry.tsc)) {
        return false;
    }
    return true;
}

template <class P>
bool TextureCache<P>::TextureMemoViewClean(ImageViewId view) const noexcept {
    if (!view || view == NULL_IMAGE_VIEW_ID) {
        // VisitImageView prepares nothing for the null view.
        return true;
    }
    const ImageViewBase& image_view = slot_image_views[view];
    if (image_view.IsBuffer()) {
        return true;
    }
    // What PrepareImage(id, false, false) acts on besides the LRU: RefreshContents and
    // SynchronizeAliases.
    const ImageBase& image = slot_images[image_view.image_id];
    return False(image.flags & ImageFlagBits::CpuModified) && image.aliased_images.empty();
}

template <class P>
void TextureCache<P>::TouchTextureMemoView(ImageViewId view) noexcept {
    if (!view || view == NULL_IMAGE_VIEW_ID) {
        return;
    }
    const ImageViewBase& image_view = slot_image_views[view];
    if (image_view.IsBuffer()) {
        return;
    }
    ImageBase& image = slot_images[image_view.image_id];
    CheckCrossFrameUse(image, false, false); // as PrepareImage(id, false, false) would
    lru_cache.Touch(image.lru_index, frame_tick);
}

template <class P>
const typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) const noexcept {
    return slot_samplers[id];
}

template <class P>
typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) noexcept {
    return slot_samplers[id];
}

template <class P>
void TextureCache<P>::SynchronizeDescriptors(bool compute) {
    if (compute) {
        const bool linked_tsc = kepler_compute->launch_description.linked_tsc;
        const u32 tic_limit = kepler_compute->regs.tic.limit;
        const u32 tsc_limit = linked_tsc ? tic_limit : kepler_compute->regs.tsc.limit;
        bool bindings_changed = false;
        if (channel_state->compute_sampler_table.Synchronize(kepler_compute->regs.tsc.Address(), tsc_limit))
            bindings_changed = true;
        if (channel_state->compute_image_table.Synchronize(kepler_compute->regs.tic.Address(), tic_limit))
            bindings_changed = true;
        if (bindings_changed) {
            ++texture_bindings_serial;
            ++bindings_epoch;
        }
    } else {
        const bool linked_tsc = maxwell3d->regs.sampler_binding == Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
        const u32 tic_limit = maxwell3d->regs.tex_header.limit;
        const u32 tsc_limit = linked_tsc ? tic_limit : maxwell3d->regs.tex_sampler.limit;
        bool bindings_changed = false;
        if (channel_state->graphics_sampler_table.Synchronize(maxwell3d->regs.tex_sampler.Address(), tsc_limit))
            bindings_changed = true;
        if (channel_state->graphics_image_table.Synchronize(maxwell3d->regs.tex_header.Address(), tic_limit))
            bindings_changed = true;
        if (bindings_changed) {
            ++texture_bindings_serial;
            ++bindings_epoch;
        }
    }
}

template <class P>
bool TextureCache<P>::RescaleRenderTargets() {
    auto& flags = maxwell3d->dirty.flags;
    u32 scale_rating = 0;
    bool rescaled = false;
    std::array<ImageId, NUM_RT> tmp_color_images{};
    ImageId tmp_depth_image{};
    do {
        flags[Dirty::RenderTargets] = false;

        has_deleted_images = false;
        // Render target control is used on all render targets, so force look ups when this one is
        // up
        const bool force = flags[Dirty::RenderTargetControl];
        flags[Dirty::RenderTargetControl] = false;

        scale_rating = 0;
        bool any_rescaled = false;
        bool can_rescale = true;
        const auto check_rescale = [&](ImageViewId view_id, ImageId& id_save) {
            if (view_id != NULL_IMAGE_VIEW_ID && view_id != ImageViewId{}) {
                const auto& view = slot_image_views[view_id];
                const auto image_id = view.image_id;
                id_save = image_id;
                auto& image = slot_images[image_id];
                can_rescale &= ImageCanRescale(image);
                any_rescaled |= True(image.flags & ImageFlagBits::Rescaled) ||
                                GetFormatType(image.info.format) != SurfaceType::ColorTexture;
                scale_rating = std::max<u32>(scale_rating, image.scale_tick <= frame_tick
                                                               ? image.scale_rating + 1U
                                                               : image.scale_rating);
            } else {
                id_save = CORRUPT_ID;
            }
        };
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            if (flags[Dirty::ColorBuffer0 + index] || force) {
                flags[Dirty::ColorBuffer0 + index] = false;
                BindRenderTarget(&color_buffer_id, FindColorBuffer(index));
            }
            check_rescale(color_buffer_id, tmp_color_images[index]);
        }
        if (flags[Dirty::ZetaBuffer] || force) {
            flags[Dirty::ZetaBuffer] = false;
            BindRenderTarget(&render_targets.depth_buffer_id, FindDepthBuffer());
        }
        check_rescale(render_targets.depth_buffer_id, tmp_depth_image);

        if (can_rescale) {
            rescaled = any_rescaled || scale_rating >= 2;
            const auto scale_up = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleUp(image);
                }
            };
            if (rescaled) {
                for (size_t index = 0; index < NUM_RT; ++index) {
                    scale_up(tmp_color_images[index]);
                }
                scale_up(tmp_depth_image);
                scale_rating = 2;
            }
        } else {
            rescaled = false;
            const auto scale_down = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleDown(image);
                }
            };
            for (size_t index = 0; index < NUM_RT; ++index) {
                scale_down(tmp_color_images[index]);
            }
            scale_down(tmp_depth_image);
            scale_rating = 1;
        }
    } while (has_deleted_images);
    const auto set_rating = [this, scale_rating](ImageId image_id) {
        if (image_id != CORRUPT_ID) {
            Image& image = slot_images[image_id];
            image.scale_rating = scale_rating;
            if (image.scale_tick <= frame_tick) {
                image.scale_tick = frame_tick + 1;
            }
        }
    };
    for (size_t index = 0; index < NUM_RT; ++index) {
        set_rating(tmp_color_images[index]);
    }
    set_rating(tmp_depth_image);

    return rescaled;
}

template <class P>
void TextureCache<P>::UpdateRenderTargets(bool is_clear) {
    using namespace VideoCommon::Dirty;
    struct ClearingScope {
        bool& flag;
        ~ClearingScope() {
            flag = false;
        }
    } clearing_scope{clearing_render_targets};
    clearing_render_targets = is_clear;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::RenderTargets]) {
        // EDEN_NX2_EXP31 (RT_FAST, zortos293/Lichii#717): bit 1 = skip the loop below while it
        // would change nothing, bit 2 = run it anyway and check that it changed nothing.
        const int rt_fast = VideoCore::Nx2Debug::Experiment(31);
        bool verify = false;
        if (rt_fast != 0 && !is_clear) {
            ++rt_fast_clean_paths;
            if ((rt_fast & 1) != 0 && RenderTargetsStillPrepared()) {
                if ((rt_fast & 2) == 0) {
                    ++rt_fast_skips;
                    return;
                }
                verify = true;
            }
        }
        if (verify) {
            if (!VerifyRenderTargetsPrepared()) {
                rt_prepared.valid = false;
                return;
            }
            NoteRenderTargetsPrepared();
            return;
        }
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
        }
        const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;
        PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));
        if (is_clear) {
            NoteFullClears();
            rt_prepared.valid = false;
        } else if ((rt_fast & 1) != 0) {
            NoteRenderTargetsPrepared();
        }
        return;
    }
    rt_prepared.valid = false;

    const VideoCommon::RenderTargets previous_render_targets = render_targets;
    const bool rescaled = RescaleRenderTargets();
    if (is_rescaling != rescaled) {
        flags[Dirty::RescaleViewports] = true;
        flags[Dirty::RescaleScissors] = true;
        is_rescaling = rescaled;
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
        PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
    }
    const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;

    PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));
    if (is_clear) {
        NoteFullClears();
    }

    rt_active_mask = 0;
    rt_image_id = {};
    for (size_t i = 0; i < rt_image_id.size(); ++i) {
        if (ImageViewId const view = render_targets.color_buffer_ids[i]; view) {
            rt_active_mask |= 1u << i;
            rt_image_id[i] = slot_image_views[view].image_id;
        }
    }
    if (depth_buffer_id) {
        rt_active_mask |= (1u << NUM_RT);
        rt_depth_image_id = slot_image_views[depth_buffer_id].image_id;
    } else {
        rt_depth_image_id = ImageId{};
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        render_targets.draw_buffers[index] = static_cast<u8>(maxwell3d->regs.rt_control.Map(index));
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (is_rescaling) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    render_targets.size = Extent2D{
        (maxwell3d->regs.surface_clip.width * up_scale) >> down_shift,
        (maxwell3d->regs.surface_clip.height * up_scale) >> down_shift,
    };
    render_targets.is_rescaled = is_rescaling;

    if (render_targets != previous_render_targets) {
        ++render_targets_serial;
    }

    flags[Dirty::DepthBiasGlobal] = true;
}

template <class P>
bool TextureCache<P>::RenderTargetsStillPrepared() const {
    const auto& prepared = rt_prepared;
    if (!prepared.valid || prepared.serial != render_targets_serial ||
        prepared.modification_tick != modification_tick || prepared.frame_tick != frame_tick ||
        prepared.bindings_epoch != bindings_epoch) {
        return false;
    }
    for (size_t index = 0; index <= NUM_RT; ++index) {
        const ImageViewId id = index < NUM_RT ? render_targets.color_buffer_ids[index]
                                              : render_targets.depth_buffer_id;
        if (id != prepared.views[index]) {
            return false;
        }
        if (!id) {
            continue;
        }
        const ImageViewBase& image_view = slot_image_views[id];
        if (image_view.IsBuffer()) {
            continue;
        }
        // What PrepareImage(is_modification, !invalidate) would act on.
        const ImageBase& image = slot_images[image_view.image_id];
        if (False(image.flags & ImageFlagBits::GpuModified) ||
            True(image.flags & (ImageFlagBits::CpuModified | ImageFlagBits::SparseTiles |
                                ImageFlagBits::SparseLayers | ImageFlagBits::UnwrittenSparse)) ||
            !image.aliased_images.empty()) {
            return false;
        }
    }
    // MarkModification would give them the newest ticks again, in the same order: they have them,
    // nothing was marked since. LRU touches within a frame are no-ops.
    return true;
}

template <class P>
void TextureCache<P>::NoteRenderTargetsPrepared() {
    auto& prepared = rt_prepared;
    prepared.valid = true;
    prepared.serial = render_targets_serial;
    prepared.modification_tick = modification_tick;
    prepared.frame_tick = frame_tick;
    prepared.bindings_epoch = bindings_epoch;
    for (size_t index = 0; index < NUM_RT; ++index) {
        prepared.views[index] = render_targets.color_buffer_ids[index];
    }
    prepared.views[NUM_RT] = render_targets.depth_buffer_id;
}

template <class P>
bool TextureCache<P>::VerifyRenderTargetsPrepared() {
    // The images the clean path prepares, with their flags before it runs.
    std::array<ImageId, NUM_RT + 1> images{};
    std::array<ImageFlagBits, NUM_RT + 1> flags_before{};
    u64 marked = 0;
    for (size_t index = 0; index <= NUM_RT; ++index) {
        const ImageViewId id = index < NUM_RT ? render_targets.color_buffer_ids[index]
                                              : render_targets.depth_buffer_id;
        if (!id || slot_image_views[id].IsBuffer()) {
            continue;
        }
        images[index] = slot_image_views[id].image_id;
        flags_before[index] = slot_images[images[index]].flags;
        ++marked;
    }
    const u64 tick_before = modification_tick;
    const u64 epoch_before = bindings_epoch;
    for (size_t index = 0; index < NUM_RT; ++index) {
        PrepareImageView(render_targets.color_buffer_ids[index], true, false);
    }
    PrepareImageView(render_targets.depth_buffer_id, true, false);
    ++rt_fast_verified;
    // Skipping it would have left the ticks alone: MarkModification once per target is all it
    // may have done.
    bool same = modification_tick - tick_before == marked && bindings_epoch == epoch_before;
    size_t first_difference = NUM_RT + 1;
    for (size_t index = 0; index <= NUM_RT && same; ++index) {
        if (images[index] && slot_images[images[index]].flags != flags_before[index]) {
            same = false;
            first_difference = index;
        }
    }
    if (!same) {
        ++rt_fast_mismatches;
        static u32 logged = 0;
        if (logged < 32) {
            ++logged;
            const ImageId image =
                first_difference <= NUM_RT ? images[first_difference] : ImageId{};
            LOG_WARNING(HW_GPU,
                        "RT_FAST (EXP31) mismatch: the clean path changed what the skip assumed: "
                        "modification tick +{} for {} targets, epoch {} -> {}, target {} flags "
                        "{:#x} -> {:#x}",
                        modification_tick - tick_before, marked, epoch_before, bindings_epoch,
                        first_difference,
                        image ? static_cast<u32>(flags_before[first_difference]) : 0U,
                        image ? static_cast<u32>(slot_images[image].flags) : 0U);
        }
    }
    return same;
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer() {
    if (last_framebuffer_id && last_framebuffer_serial == render_targets_serial) {
        return &slot_framebuffers[last_framebuffer_id];
    }
    const FramebufferId framebuffer_id = GetFramebufferId(render_targets);
    last_framebuffer_id = framebuffer_id;
    last_framebuffer_serial = render_targets_serial;
    return &slot_framebuffers[framebuffer_id];
}

template <class P>
ImageViewId TextureCache<P>::VisitImageView(u32 index, bool compute) {
    auto& table = compute ? channel_state->compute_image_table : channel_state->graphics_image_table;
    index = Tegra::Texture::TextureHeaderIndex(index, table.current_limit,
                                               gpu_memory->UsesNativeTiling());
    if (index > table.current_limit) {
        LOG_TRACE(HW_GPU, "Invalid image view index={}", index);
        return NULL_IMAGE_VIEW_ID;
    }
    auto const [descriptor, is_new] = table.Read(*gpu_memory, index);
    if (is_new) {
        if (IsValidEntry(*gpu_memory, descriptor)) {
            const auto [pair, is_new_tc] = channel_state->image_views.try_emplace(descriptor);
            if (is_new_tc) {
                const auto failed_images = [&]() -> u32 {
                    if constexpr (requires { runtime.FailedImages(); }) {
                        return runtime.FailedImages();
                    } else {
                        return 0;
                    }
                };
                const u32 failed_before = failed_images();
                pair->second = CreateImageView(descriptor);
                if (pair->second == NULL_IMAGE_VIEW_ID && failed_images() != failed_before) {
                    // No memory for its image: TickFrame invalidates the table after collecting,
                    // and the descriptor's next read makes the image again.
                    channel_state->image_views.erase(pair);
                    table.bound_ids[index] = NULL_IMAGE_VIEW_ID;
                    return NULL_IMAGE_VIEW_ID;
                }
                if (pair->second != NULL_IMAGE_VIEW_ID) {
                    image_view_descriptors[pair->second].emplace_back(current_channel_id,
                                                                      descriptor);
                }
            }
            PrepareImageView(pair->second, false, false);
            table.bound_ids[index] = pair->second;
            return pair->second;
        }
        table.bound_ids[index] = NULL_IMAGE_VIEW_ID;
        return NULL_IMAGE_VIEW_ID;
    }
    const auto id = table.bound_ids[index];
    if (id != NULL_IMAGE_VIEW_ID)
        PrepareImageView(id, false, false);
    return id;
}

template <class P>
FramebufferId TextureCache<P>::GetFramebufferId(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;
    if (!is_new) {
        return framebuffer_id;
    }
    std::array<ImageView*, NUM_RT> color_buffers;
    std::ranges::transform(key.color_buffer_ids, color_buffers.begin(), [this](ImageViewId id) {
        return id ? &slot_image_views[id] : nullptr;
    });
    ImageView* const depth_buffer = key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;
    const auto framebuffer_start = std::chrono::steady_clock::now();
    runtime.FlushDeferredClear();
    framebuffer_id = slot_framebuffers.insert(runtime, color_buffers, depth_buffer, key);
    framebuffer_keys.emplace(framebuffer_id, key);
    const auto index_view = [this, id = framebuffer_id](ImageViewId view) {
        if (!view) {
            return;
        }
        auto& ids = framebuffers_by_view[view];
        if (std::ranges::find(ids, id) == ids.end()) {
            ids.push_back(id);
        }
    };
    std::ranges::for_each(key.color_buffer_ids, index_view);
    index_view(key.depth_buffer_id);
    ++rt_stats.framebuffers_created;
    rt_stats.framebuffer_ns += NsSince(framebuffer_start);
    return framebuffer_id;
}

template <class P>
void TextureCache<P>::WriteMemory(DAddr cpu_addr, size_t size) {
    ClearStale(cpu_addr, size);
    ForEachImageInRegion(cpu_addr, size, [this, cpu_addr, size](ImageId image_id, Image& image) {
        ++bindings_epoch;
        if (True(image.flags & ImageFlagBits::UnwrittenSparse)) {
            // Written at last: a backed image replaces it where it is used next.
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
            return;
        }
        if (True(image.flags & ImageFlagBits::SparseTiles)) {
            MarkRangeWritten(image, cpu_addr, size);
            return;
        }
        if (UploadsByLayer(image)) {
            MarkLayersWritten(image, cpu_addr, size);
            return;
        }
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            return;
        }
        image.flags |= ImageFlagBits::CpuModified;
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
    });
}

template <class P>
bool TextureCache<P>::IsDownloadable(const ImageBase& image) const noexcept {
    if (!image.IsSafeGpuCopy() || True(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels))) {
        return false;
    }
    if (image.info.num_samples == 1) {
        return true;
    }
    if constexpr (P::HAS_MSAA_DOWNLOADS) {
        return runtime.CanDownloadMsaa(image.info);
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::DownloadMemory(DAddr cpu_addr, size_t size) {
    if (!relocation_tombstones.empty()) {
        CheckTombstoneCpuRead(cpu_addr, size); // EXP43=4
    }
    boost::container::small_vector<ImageId, 16> images;
    ForEachImageInRegion(cpu_addr, size, [this, &images](ImageId image_id, ImageBase& image) {
        if (!IsDownloadable(image)) {
            return;
        }
        if (True(image.flags & ImageFlagBits::GpuModified) &&
            (image.relocation_guard_noted & GuardCpuRead) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, GuardCpuRead);
        }
        image.flags &= ~ImageFlagBits::GpuModified;
        images.push_back(image_id);
    });
    if (images.empty()) {
        return;
    }
    std::ranges::sort(images, [this](ImageId lhs, ImageId rhs) {
        return slot_images[lhs].modification_tick < slot_images[rhs].modification_tick;
    });
    // Record the copies of a batch, wait for the GPU once, then swizzle in modification order
    // (the copies only read, so they do not depend on each other). A batch is capped so the
    // staging memory held at once stays bounded.
    constexpr size_t MaxBatchBytes = 64_MiB;
    using Map = decltype(runtime.DownloadStagingBuffer(size_t{}));
    using Copies = decltype(FullDownloadCopies(std::declval<const ImageInfo&>()));
    struct Pending {
        ImageId image_id;
        Map map;
        Copies copies;
    };
    boost::container::small_vector<Pending, 16> pending;
    size_t pending_bytes = 0;
    const auto flush_pending = [&] {
        if (pending.empty()) {
            return;
        }
        runtime.Finish();
        for (Pending& entry : pending) {
            Image& image = slot_images[entry.image_id];
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, entry.copies,
                         entry.map.mapped_span, swizzle_data_buffer);
            ClearStaleImage(image);
        }
        pending.clear();
        pending_bytes = 0;
    };
    for (const ImageId image_id : images) {
        Image& image = slot_images[image_id];
        if (!pending.empty() && pending_bytes + image.unswizzled_size_bytes > MaxBatchBytes) {
            flush_pending();
        }
        auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
        auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, copies);
        pending_bytes += image.unswizzled_size_bytes;
        pending.push_back(Pending{image_id, map, std::move(copies)});
    }
    flush_pending();
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> TextureCache<P>::GetFlushArea(DAddr cpu_addr,
                                                                               u64 size) {
    if (!relocation_tombstones.empty()) {
        CheckTombstoneCpuRead(cpu_addr, size); // EXP43=4
    }
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    ForEachImageInRegion(cpu_addr, size, [&](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        if (!area) {
            area.emplace();
            area->start_address = cpu_addr;
            area->end_address = cpu_addr + size;
            area->preemtive = true;
        }
        area->start_address = (std::min)(area->start_address, image.cpu_addr);
        area->end_address = (std::max)(area->end_address, image.cpu_addr_end);
        for (auto image_view_id : image.image_view_ids) {
            auto& image_view = slot_image_views[image_view_id];
            image_view.flags |= ImageViewFlagBits::PreemtiveDownload;
        }
        area->preemtive &= image.info.forced_flushed;
        image.info.forced_flushed = true;
        if ((image.relocation_guard_noted & GuardCpuRead) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, GuardCpuRead);
        }
    });
    return area;
}

template <class P>
void TextureCache<P>::UnmapMemory(DAddr cpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, id);
        }
        UnregisterImage(id);
        DeleteImage(id);
    }
    ClearStale(cpu_addr, size);
}

template <class P>
void TextureCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegionGPU(as_id, gpu_addr, size,
                            [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        ++bindings_epoch;
        if (False(image.flags & ImageFlagBits::CpuModified)) {
            image.flags |= ImageFlagBits::CpuModified;
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, id);
            }
        }
        image.dirty_layers.clear(); // all of it
        if (True(image.flags & ImageFlagBits::Remapped)) {
            continue;
        }
        image.flags |= ImageFlagBits::Remapped;
    }
}

template <class P>
bool TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                                const Tegra::Engines::Fermi2D::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    // A deferred clear parked before this copy has to land before it: realizing it afterwards
    // starts its clear pass after the copy and wipes what the copy wrote. That is what made
    // Ocarina of Time draw its terrain layer over the scene (#101): the game clears the
    // occlusion depth, blits the scene depth over it, and reads it back during the layer draw.
    // GetFramebufferId only flushes when it creates a framebuffer, so the bug starts on the
    // second frame. Texture cache copies bypass render passes and keep stream order here.
    runtime.FlushDeferredClear();
    const auto result = GetBlitImages(dst, src, copy);
    if (!result) {
        return false;
    }
    const BlitImages images = *result;
    const ImageId dst_id = images.dst_id;
    const ImageId src_id = images.src_id;

    PrepareImage(src_id, false, false);
    PrepareImage(dst_id, true, false);

    Image& dst_image = slot_images[dst_id];
    Image& src_image = slot_images[src_id];
    if (True(dst_image.flags & ImageFlagBits::Unbacked)) {
        static u32 logged{};
        if (logged++ < 64) {
            LOG_INFO(HW_GPU, "2D blit into unbacked image {:#x}: dst {:#x} {}x{} from {:#x} {}x{}",
                     dst_image.gpu_addr, dst.Address(), dst.width, dst.height, src.Address(),
                     src.width, src.height);
        }
    }
    bool is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
    bool is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);

    const bool is_resolve = src_image.info.num_samples != 1 && dst_image.info.num_samples == 1;
    if (is_src_rescaled != is_dst_rescaled) {
        if (ImageCanRescale(src_image)) {
            ScaleUp(src_image);
            is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
            if (is_resolve) {
                dst_image.info.rescaleable = true;
                for (const auto& alias : dst_image.aliased_images) {
                    Image& other_image = slot_images[alias.id];
                    other_image.info.rescaleable = true;
                }
            }
        }
        if (ImageCanRescale(dst_image)) {
            ScaleUp(dst_image);
            is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
        }
    }
    if (is_resolve && (is_src_rescaled != is_dst_rescaled)) {
        // A resolve requires both images to be the same dimensions. Resize down if needed.
        ScaleDown(src_image);
        ScaleDown(dst_image);
        is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
        is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
    }
    const auto& resolution = Settings::values.resolution_info;
    const auto scale_region = [&](Region2D& region) {
        region.start.x = resolution.ScaleUp(region.start.x);
        region.start.y = resolution.ScaleUp(region.start.y);
        region.end.x = resolution.ScaleUp(region.end.x);
        region.end.y = resolution.ScaleUp(region.end.y);
    };

    // TODO: Deduplicate
    const std::optional src_base = src_image.TryFindBase(src.Address());
    const SubresourceRange src_range{.base = src_base.value(), .extent = {1, 1}};
    const ImageViewInfo src_view_info(ImageViewType::e2D, images.src_format, src_range);
    const auto [src_framebuffer_id, src_view_id] = RenderTargetFromImage(src_id, src_view_info);
    const auto [src_samples_x, src_samples_y] = SamplesLog2(src_image.info.num_samples);
    Region2D src_region{
        Offset2D{.x = copy.src_x0 >> src_samples_x, .y = copy.src_y0 >> src_samples_y},
        Offset2D{.x = copy.src_x1 >> src_samples_x, .y = copy.src_y1 >> src_samples_y},
    };
    if (is_src_rescaled) {
        scale_region(src_region);
    }

    const std::optional dst_base = dst_image.TryFindBase(dst.Address());
    const SubresourceRange dst_range{.base = dst_base.value(), .extent = {1, 1}};
    const ImageViewInfo dst_view_info(ImageViewType::e2D, images.dst_format, dst_range);
    const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
    const auto [dst_samples_x, dst_samples_y] = SamplesLog2(dst_image.info.num_samples);
    Region2D dst_region{
        Offset2D{.x = copy.dst_x0 >> dst_samples_x, .y = copy.dst_y0 >> dst_samples_y},
        Offset2D{.x = copy.dst_x1 >> dst_samples_x, .y = copy.dst_y1 >> dst_samples_y},
    };
    if (is_dst_rescaled) {
        scale_region(dst_region);
    }

    // Always call this after src_framebuffer_id was queried, as the address might be invalidated.
    Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
    if constexpr (FRAMEBUFFER_BLITS) {
        // OpenGL blits from framebuffers, not images
        Framebuffer* const src_framebuffer = &slot_framebuffers[src_framebuffer_id];
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, dst_region, src_region,
                                copy.filter, copy.operation);
    } else {
        // Vulkan can blit images, but it lacks format reinterpretations
        // Provide a framebuffer in case it's necessary
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        runtime.BlitImage(dst_framebuffer, dst_view, src_view, dst_region, src_region, copy.filter,
                          copy.operation);
    }
    return true;
}

template <class P>
std::optional<VideoCore::GameDlssProbe> TextureCache<P>::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src) {
    if constexpr (FRAMEBUFFER_BLITS) {
        return std::nullopt;
    } else {
        const Tegra::Engines::Fermi2D::Config copy{
            .operation = Tegra::Engines::Fermi2D::Operation::SrcCopy,
            .filter = Tegra::Engines::Fermi2D::Filter::Point,
            .must_accelerate = true,
            .dst_x0 = 0,
            .dst_y0 = 0,
            .dst_x1 = static_cast<s32>(dst.width),
            .dst_y1 = static_cast<s32>(dst.height),
            .src_x0 = 0,
            .src_y0 = 0,
            .src_x1 = static_cast<s32>(src.width),
            .src_y1 = static_cast<s32>(src.height),
        };
        const auto result = GetBlitImages(dst, src, copy);
        if (!result) {
            return std::nullopt;
        }
        const ImageId dst_id = result->dst_id;
        const ImageId src_id = result->src_id;
        PrepareImage(src_id, false, false);
        PrepareImage(dst_id, false, false);
        for (const ImageId id : {src_id, dst_id}) {
            // The network runs only at the game's own resolution.
            if (slot_images[id].info.num_samples != 1 ||
                True(slot_images[id].flags & ImageFlagBits::Rescaled)) {
                return std::nullopt;
            }
        }
        const auto view = [&](ImageId id, const Tegra::Engines::Fermi2D::Surface& surface,
                              PixelFormat format) -> std::optional<ImageViewId> {
            const std::optional base = slot_images[id].TryFindBase(surface.Address());
            if (!base) {
                return std::nullopt;
            }
            const SubresourceRange range{.base = *base, .extent = {1, 1}};
            return FindOrEmplaceImageView(id, ImageViewInfo(ImageViewType::e2D, format, range));
        };
        const auto src_view = view(src_id, src, result->src_format);
        const auto dst_view = view(dst_id, dst, result->dst_format);
        if (!src_view || !dst_view) {
            return std::nullopt;
        }
        return runtime.ProbeGameDlssOutput(slot_image_views[*src_view],
                                           slot_image_views[*dst_view]);
    }
}

template <class P>
bool TextureCache<P>::TemporalUpscale(const Tegra::Engines::Fermi2D::Surface& dst,
                                      const Tegra::Engines::Fermi2D::Surface& src,
                                      const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                      const Tegra::Texture::TICEntry* depth,
                                      std::array<f32, 2> jitter, std::array<f32, 2> motion_scale,
                                      bool use_history) {
    if constexpr (FRAMEBUFFER_BLITS) {
        return false;
    } else {
        const Tegra::Engines::Fermi2D::Config copy{
            .operation = Tegra::Engines::Fermi2D::Operation::SrcCopy,
            .filter = Tegra::Engines::Fermi2D::Filter::Bilinear,
            .must_accelerate = true,
            .dst_x0 = 0,
            .dst_y0 = 0,
            .dst_x1 = static_cast<s32>(dst.width),
            .dst_y1 = static_cast<s32>(dst.height),
            .src_x0 = 0,
            .src_y0 = 0,
            .src_x1 = static_cast<s32>(src.width),
            .src_y1 = static_cast<s32>(src.height),
        };
        const auto result = GetBlitImages(dst, src, copy);
        if (!result) {
            return false;
        }
        const auto& motion_surface = use_history ? motion_vectors : src;
        const ImageInfo motion_info(motion_surface);
        has_deleted_images = false;
        const ImageId motion_id =
            FindOrInsertImage(motion_info, motion_surface.Address(), RelaxedOptions{});
        // The depth only feeds NVIDIA DLSS and FSR 3.1 (looked up only when they can run); without
        // it the stand-in runs.
        ImageId depth_id{};
        if (use_history && depth && motion_id && runtime.WantsTemporalUpscaleDepth()) {
            depth_id = FindOrInsertImage(ImageInfo(*depth), depth->Address(), RelaxedOptions{});
            if (!depth_id) {
                return false;
            }
        }
        if (has_deleted_images || !motion_id) {
            return false;
        }
        const ImageId dst_id = result->dst_id;
        const ImageId src_id = result->src_id;
        PrepareImage(src_id, false, false);
        PrepareImage(motion_id, false, false);
        PrepareImage(dst_id, true, false);
        if (depth_id) {
            PrepareImage(depth_id, false, false);
            if (slot_images[depth_id].info.num_samples != 1 ||
                slot_images[depth_id].info.format != PixelFormat::D32_FLOAT) {
                return false;
            }
        }
        for (const ImageId id : {src_id, motion_id, dst_id}) {
            if (slot_images[id].info.num_samples != 1) {
                return false;
            }
        }
        // With resolution scaling, bring the images that can follow to the same scale, as BlitImage
        // does; the stand-in reads each image at its own size, so any left behind still work.
        const auto rescaled = [&](ImageId id) {
            return True(slot_images[id].flags & ImageFlagBits::Rescaled);
        };
        if (rescaled(src_id) || rescaled(motion_id) || rescaled(dst_id) ||
            (depth_id && rescaled(depth_id))) {
            for (const ImageId id : {src_id, motion_id, dst_id, depth_id}) {
                if (id && !rescaled(id) && ImageCanRescale(slot_images[id])) {
                    ScaleUp(slot_images[id]);
                }
            }
        }
        // FSR 3.1 reads colour, motion and depth texel for texel: all at the colour's scale.
        if (depth_id && (rescaled(depth_id) != rescaled(src_id) ||
                         rescaled(motion_id) != rescaled(src_id))) {
            return false;
        }
        const Extent3D src_size = slot_images[src_id].info.size;
        if (src_size.width != src.width || src_size.height != src.height) {
            return false;
        }
        const std::array<u32, 2> render_size =
            rescaled(src_id)
                ? std::array{Settings::values.resolution_info.ScaleUp(src_size.width),
                             Settings::values.resolution_info.ScaleUp(src_size.height)}
                : std::array{src_size.width, src_size.height};
        // The jitter is in the input's guest pixels.
        if (rescaled(src_id)) {
            const f32 factor = Settings::values.resolution_info.up_factor;
            jitter = {jitter[0] * factor, jitter[1] * factor};
        }
        const auto view = [&](ImageId id, const Tegra::Engines::Fermi2D::Surface& surface,
                              PixelFormat format) -> std::optional<std::pair<FramebufferId, ImageViewId>> {
            const std::optional base = slot_images[id].TryFindBase(surface.Address());
            if (!base) {
                return std::nullopt;
            }
            const SubresourceRange range{.base = *base, .extent = {1, 1}};
            return RenderTargetFromImage(id, ImageViewInfo(ImageViewType::e2D, format, range));
        };
        const auto src_view = view(src_id, src, result->src_format);
        const auto motion_view = view(motion_id, motion_surface, motion_info.format);
        const auto dst_view = view(dst_id, dst, result->dst_format);
        if (!src_view || !motion_view || !dst_view) {
            return false;
        }
        ImageViewId depth_view{};
        if (depth_id) {
            if (const std::optional base = slot_images[depth_id].TryFindBase(depth->Address())) {
                const SubresourceRange range{.base = *base, .extent = {1, 1}};
                depth_view = FindOrEmplaceImageView(
                    depth_id, ImageViewInfo(ImageViewType::e2D, PixelFormat::D32_FLOAT, range));
            }
            if (!depth_view) {
                return false;
            }
        }
        // After the views: creating them may reallocate the framebuffer slots.
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_view->first];
        return runtime.TemporalUpscale(
            dst_framebuffer, slot_image_views[dst_view->second], slot_image_views[src_view->second],
            slot_image_views[motion_view->second],
            depth_view ? &slot_image_views[depth_view] : nullptr, render_size, jitter,
            motion_scale, use_history);
    }
}

template <class P>
std::pair<typename P::ImageView*, bool> TextureCache<P>::TryFindFramebufferImageView(
    const Tegra::FramebufferConfig& config, DAddr cpu_addr) {
    // TODO: Properly implement this
    const auto it = page_table.find(cpu_addr >> YUZU_PAGEBITS);
    if (it == page_table.end()) {
        return {};
    }
    const auto view_format = [&]() {
        switch (config.pixel_format) {
        case Service::android::PixelFormat::Rgb565:
            return PixelFormat::R5G6B5_UNORM;
        case Service::android::PixelFormat::Bgra8888:
            return PixelFormat::B8G8R8A8_UNORM;
        default:
            return PixelFormat::A8B8G8R8_UNORM;
        }
    }();

    const auto& image_map_ids = it->second;
    boost::container::small_vector<ImageId, 4> valid_image_ids;
    for (const ImageMapId map_id : image_map_ids) {
        const ImageMapView& map = slot_map_views[map_id];
        const ImageBase& image = slot_images[map.image_id];
        if (image.cpu_addr != cpu_addr) {
            continue;
        }
        if (image.image_view_ids.empty()) {
            continue;
        }
        if (!VideoCore::Surface::IsViewCompatible(image.info.format, view_format, false, true)) {
            static u32 rejected_alias_count{};
            if (NativeDrawDiagnostics() && rejected_alias_count++ < 8) {
                LOG_WARNING(HW_GPU, "Ignoring incompatible framebuffer alias: address={:#x}, image={}, view={}",
                            cpu_addr, image.info.format, view_format);
            }
            continue;
        }
        valid_image_ids.push_back(map.image_id);
    }

    const auto GetImageViewForFramebuffer = [&](ImageId image_id) {
        ImageViewInfo info{ImageViewType::e2D, view_format};
        if (config.blending == Tegra::BlendMode::Opaque) {
            info.x_source = static_cast<u8>(SwizzleSource::R);
            info.y_source = static_cast<u8>(SwizzleSource::G);
            info.z_source = static_cast<u8>(SwizzleSource::B);
            info.w_source = static_cast<u8>(SwizzleSource::OneFloat);
        }
        return std::make_pair(&slot_image_views[FindOrEmplaceImageView(image_id, info)],
                              slot_images[image_id].IsRescaled());
    };

    if (valid_image_ids.size() == 1) [[likely]] {
        return GetImageViewForFramebuffer(valid_image_ids.front());
    }

    if (valid_image_ids.size() > 0) [[unlikely]] {
        auto most_recent = std::ranges::max_element(valid_image_ids, [&](auto a, auto b) {
            return slot_images[a].modification_tick < slot_images[b].modification_tick;
        });
        return GetImageViewForFramebuffer(*most_recent);
    }

    return {};
}

template <class P>
bool TextureCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_downloads.empty();
}

template <class P>
bool TextureCache<P>::HasAsyncFlushWork() const noexcept {
    return !uncommitted_downloads.empty() || !uncommitted_async_buffers.empty();
}

template <class P>
bool TextureCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return !committed_downloads.empty() && !committed_downloads.front().empty();
}

template <class P>
void TextureCache<P>::CommitAsyncFlushes() {
    // This is intentionally passing the value by copy
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        auto& download_ids = uncommitted_downloads;
        const size_t redundant_images = DeduplicatePendingImageDownloads(download_ids, slot_images);
        if (download_ids.empty()) {
            committed_downloads.emplace_back(std::move(uncommitted_downloads));
            uncommitted_downloads.clear();
            async_buffers.emplace_back(std::move(uncommitted_async_buffers));
            uncommitted_async_buffers.clear();
            return;
        }
        size_t total_size_bytes = 0;
        size_t last_async_buffer_id = uncommitted_async_buffers.size();
        bool any_none_dma = false;
        for (PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes +=
                    Common::AlignUp(download_info.GetImage(slot_images).unswizzled_size_bytes, 64);
                any_none_dma = true;
                download_info.async_buffer_id = last_async_buffer_id;
            }
        }

        if (any_none_dma) {
            if (NativeDrawDiagnostics() && total_size_bytes >= 64_MiB) {
                LOG_WARNING(HW_GPU, "Large image readback batch: bytes={}, entries={}, duplicates_removed={}, submitted_batches={}, retained_buffers={}",
                            total_size_bytes, download_ids.size(), redundant_images,
                            committed_downloads.size(), async_buffers_death_ring.size());
                size_t traced_images = 0;
                for (const auto& download : download_ids) {
                    if (!download.is_swizzle || traced_images++ >= 8) continue;
                    const auto& image = download.GetImage(slot_images);
                    LOG_WARNING(HW_GPU, "Readback image: gpu={:#x}, cpu={:#x}, format={}, size={}x{}x{}, levels={}, layers={}, layer_stride={}, guest_bytes={}, download_bytes={}",
                                image.gpu_addr, image.cpu_addr, image.info.format,
                                image.info.size.width, image.info.size.height, image.info.size.depth,
                                image.info.resources.levels, image.info.resources.layers,
                                image.info.layer_stride, image.guest_size_bytes, image.unswizzled_size_bytes);
                }
            }
            auto download_map = runtime.DownloadStagingBuffer(total_size_bytes, true);
            for (const PendingDownload& download_info : download_ids) {
                if (download_info.is_swizzle) {
                    Image& image = download_info.GetImage(slot_images);
                    const auto copies = FullDownloadCopies(image.info);
                    image.DownloadMemory(download_map, copies);
                    download_map.offset += Common::AlignUp(image.unswizzled_size_bytes, 64);
                }
            }
            uncommitted_async_buffers.emplace_back(download_map);
        }

        async_buffers.emplace_back(std::move(uncommitted_async_buffers));
        uncommitted_async_buffers.clear();
    }
    committed_downloads.emplace_back(std::move(uncommitted_downloads));
    uncommitted_downloads.clear();
}

template <class P>
void TextureCache<P>::PopAsyncFlushes() {
    if (committed_downloads.empty()) {
        return;
    }
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            async_buffers.pop_front();
            return;
        }
        auto download_map = std::move(async_buffers.front());
        for (size_t i = download_ids.size(); i > 0; i--) {
            auto& download_info = download_ids[i - 1];
            auto& download_buffer = download_map[download_info.async_buffer_id];
            if (download_info.is_swizzle) {
                const ImageBase& image = download_info.GetImage(slot_images);
                const auto copies = FullDownloadCopies(image.info);
                download_buffer.offset -= Common::AlignUp(image.unswizzled_size_bytes, 64);
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span, swizzle_data_buffer);
                ClearStaleImage(image);
            } else {
                const BufferDownload& buffer_info = slot_buffer_downloads[download_info.object_id];
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                gpu_memory->WriteBlockUnsafe(buffer_info.address, download_span.data(),
                                             buffer_info.size);
                ClearStaleGpu(buffer_info.address, buffer_info.size);
                slot_buffer_downloads.erase(download_info.object_id);
            }
        }
        for (auto& download_buffer : download_map) {
            async_buffers_death_ring.emplace_back(download_buffer);
        }
        committed_downloads.pop_front();
        async_buffers.pop_front();
    } else {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            return;
        }
        size_t total_size_bytes = 0;
        for (const PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes += download_info.GetImage(slot_images).unswizzled_size_bytes;
            }
        }
        auto download_map = runtime.DownloadStagingBuffer(total_size_bytes);
        const size_t original_offset = download_map.offset;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            Image& image = download_info.GetImage(slot_images);
            const auto copies = FullDownloadCopies(image.info);
            image.DownloadMemory(download_map, copies);
            download_map.offset += image.unswizzled_size_bytes;
        }
        // Wait for downloads to finish
        runtime.Finish();
        download_map.offset = original_offset;
        std::span<u8> download_span = download_map.mapped_span;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            const ImageBase& image = download_info.GetImage(slot_images);
            const auto copies = FullDownloadCopies(image.info);
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span, swizzle_data_buffer);
            ClearStaleImage(image);
            download_map.offset += image.unswizzled_size_bytes;
            download_span = download_span.subspan(image.unswizzled_size_bytes);
        }
        committed_downloads.pop_front();
    }
}

template <class P>
ImageId TextureCache<P>::DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload) {
    const ImageInfo dst_info(operand);
    const ImageId dst_id = FindDMAImage(dst_info, operand.address);
    if (!dst_id) {
        return NULL_IMAGE_ID;
    }
    auto& image = slot_images[dst_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        // No need to waste time on an image that's synced with guest
        return NULL_IMAGE_ID;
    }
    if (image.info.type == ImageType::e3D) {
        // Don't accelerate 3D images.
        return NULL_IMAGE_ID;
    }
    if (!is_upload && !image.info.dma_downloaded) {
        // Force a full sync.
        image.info.dma_downloaded = true;
        return NULL_IMAGE_ID;
    }
    const auto base = image.TryFindBase(operand.address);
    if (!base) {
        return NULL_IMAGE_ID;
    }
    return dst_id;
}

template <class P>
bool TextureCache<P>::IsRescaling() const noexcept {
    return is_rescaling;
}

template <class P>
bool TextureCache<P>::IsRescaling(const ImageViewBase& image_view) const noexcept {
    if (image_view.type == ImageViewType::Buffer) {
        return false;
    }
    const ImageBase& image = slot_images[image_view.image_id];
    return True(image.flags & ImageFlagBits::Rescaled);
}

template <class P>
bool TextureCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    bool is_modified = false;
    ForEachImageInRegion(addr, size, [&is_modified](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return false;
        }
        is_modified = true;
        return true;
    });
    return is_modified;
}

template <class P>
std::pair<typename TextureCache<P>::Image*, BufferImageCopy> TextureCache<P>::DmaBufferImageCopy(
    const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
    const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image) {
    const auto [level, base] = PrepareDmaImage(image_id, image_operand.address, modifies_image);
    auto* image = &slot_images[image_id];
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    const u32 bpp = VideoCore::Surface::BytesPerBlock(image->info.format);
    const auto convert = [old_bpp = image_operand.bytes_per_pixel, bpp](u32 value) {
        return (old_bpp * value) / bpp;
    };
    const u32 base_x = convert(image_operand.params.origin.x.Value());
    const u32 base_y = image_operand.params.origin.y.Value();
    const u32 length_x = convert(copy_info.length_x);
    const u32 length_y = copy_info.length_y;

    const BufferImageCopy copy{
        .buffer_offset = 0,
        .buffer_size = buffer_size,
        .buffer_row_length = convert(buffer_operand.pitch),
        .buffer_image_height = buffer_operand.height,
        .image_subresource =
            {
                .base_level = static_cast<s32>(level),
                .base_layer = static_cast<s32>(base),
                .num_layers = 1,
            },
        .image_offset =
            {
                .x = static_cast<s32>(base_x),
                .y = static_cast<s32>(base_y),
                .z = 0,
            },
        .image_extent =
            {
                .width = length_x,
                .height = length_y,
                .depth = 1,
            },
    };
    return {image, copy};
}

template <class P>
void TextureCache<P>::DownloadImageIntoBuffer(typename TextureCache<P>::Image* image,
                                              typename TextureCache<P>::BufferType buffer,
                                              size_t buffer_offset,
                                              std::span<const VideoCommon::BufferImageCopy> copies,
                                              GPUVAddr address, size_t size) {
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const BufferDownload new_buffer_download{address, size};
        auto slot = slot_buffer_downloads.insert(new_buffer_download);
        const PendingDownload new_download{false, uncommitted_async_buffers.size(), slot};
        uncommitted_downloads.emplace_back(new_download);
        auto download_map = runtime.DownloadStagingBuffer(size, true);
        uncommitted_async_buffers.emplace_back(download_map);
        std::array buffers{
            buffer,
            download_map.buffer,
        };
        std::array<size_t, 2> buffer_offsets{
            buffer_offset,
            download_map.offset,
        };
        image->DownloadMemory(buffers, buffer_offsets, copies);
    } else {
        image->DownloadMemory(buffer, buffer_offset, copies);
    }
}

template <class P>
void TextureCache<P>::RefreshContents(Image& image, ImageId image_id) {
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        // Only upload modified images
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::TextureUpload};

    image.flags &= ~ImageFlagBits::CpuModified;

    // Images uploaded by layer stay tracked while some of their layers are dirty.
    if (False(image.flags & ImageFlagBits::Tracked)) {
        TrackImage(image, image_id);
    }

    // Made where a dropped GPU-written image left stale bytes (MarkStale): nothing to upload.
    // Pages the buffer cache wrote on the GPU may hold data written back since: upload those.
    if (tc_skip_stale && image.dirty_layers.empty() &&
        False(image.flags & (ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                             ImageFlagBits::SparseTiles | ImageFlagBits::Unbacked)) &&
        IsAllStale(image.cpu_addr, image.guest_size_bytes)) {
        if (!Tegra::Nx2MayBeGpuWritten(image.cpu_addr, image.guest_size_bytes)) {
            ++tc_stale_skips;
            tc_stale_skip_bytes += image.guest_size_bytes;
            runtime.TransitionImageLayout(image);
            return;
        }
        ++tc_stale_gpu_written;
    }

    if (True(image.flags & ImageFlagBits::SparseLayers)) {
        RefreshSparseLayers(image);
        return;
    }
    if (True(image.flags & ImageFlagBits::SparseTiles)) {
        RefreshSparseTiles(image);
        return;
    }
    if (!image.dirty_layers.empty() && UploadWrittenLayers(image)) {
        runtime.InsertUploadMemoryBarrier();
        return;
    }
    image.dirty_layers.clear();

    if (True(image.flags & ImageFlagBits::Unbacked)) {
        // Nothing to upload into (and staging it could take tens of GB).
        runtime.TransitionImageLayout(image);
        return;
    }

    if (image.info.num_samples > 1 && !runtime.CanUploadMSAA()) {
        LOG_WARNING(HW_GPU, "MSAA image uploads are not implemented");
        runtime.TransitionImageLayout(image);
        return;
    }
    if (True(image.flags & ImageFlagBits::AsynchronousDecode)) {
        QueueAsyncDecode(image, image_id);
        return;
    }

    const bool gpu_unswizzle_enabled = Settings::values.gpu_unswizzle_enabled.GetValue();

    if (gpu_unswizzle_enabled &&
        IsPixelFormatBCn(image.info.format) &&
        image.info.type == ImageType::e3D &&
        image.info.resources.levels == 1 &&
        image.info.resources.layers == 1 &&
        MapSizeBytes(image) >= gpu_unswizzle_maxsize &&
        False(image.flags & ImageFlagBits::GpuModified)) {

        QueueAsyncUnswizzle(image, image_id);
        return;
    }
    ++images_uploaded;
    uploaded_bytes += image.guest_size_bytes;
    largest_upload = (std::max<u64>)(largest_upload, image.guest_size_bytes);
    // EDEN_NX2_EXP7: log the uploads that hold a draw back (over 2 ms).
    const bool upload_trace = VideoCore::Nx2Debug::Experiment(7) != 0;
    const auto upload_begin = upload_trace ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    auto staging_end = upload_begin;
    // Page faults of the whole process meanwhile: ~256 per MiB when the guest memory was paged
    // out (cold-hinted) or the staging memory is new.
    const u64 faults_begin = upload_trace ? Common::GetProcessPageFaults() : 0;
    if (!UploadInChunks(image)) {
        auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
        if (upload_trace) [[unlikely]] {
            staging_end = std::chrono::steady_clock::now();
        }
        UploadImageContents(image, staging);
        runtime.InsertUploadMemoryBarrier();
    }
    if (upload_trace) [[unlikely]] {
        using Ms = std::chrono::duration<double, std::milli>;
        const double ms = Ms(std::chrono::steady_clock::now() - upload_begin).count();
        static std::atomic<u32> logged{};
        if (ms > 2.0 && logged.fetch_add(1, std::memory_order_relaxed) < 300) {
            LOG_WARNING(HW_GPU,
                        "Slow upload {:.2f} ms (staging {:.2f} ms, {} page faults): format {} "
                        "type {} {}x{}x{} levels {} layers {}, {} KiB guest, {} KiB host, "
                        "flags {:#x}",
                        ms, Ms(staging_end - upload_begin).count(),
                        Common::GetProcessPageFaults() - faults_begin,
                        static_cast<u32>(image.info.format), static_cast<u32>(image.info.type),
                        image.info.size.width, image.info.size.height, image.info.size.depth,
                        image.info.resources.levels, image.info.resources.layers,
                        image.guest_size_bytes >> 10, MapSizeBytes(image) >> 10,
                        static_cast<u32>(image.flags));
        }
    }
    // The host GPU has its own copy now and the guest rarely reads this memory again: TickFrame
    // marks it cold, so the OS pages it out before anything else when it needs RAM (title 1
    // streams several GB of textures through it).
    constexpr u64 ColdHintMinBytes = 64_KiB;
    if (image.guest_size_bytes >= ColdHintMinBytes &&
        False(image.flags & ImageFlagBits::GpuModified)) {
        if (True(image.flags & ImageFlagBits::Sparse)) {
            // Non-contiguous: each mapped piece (title 1 maps its own memory into these).
            ForEachSparseSegment(image, [&](GPUVAddr, DAddr address, size_t size) {
                uploaded_ranges.emplace_back(address, size);
            });
        } else {
            uploaded_ranges.emplace_back(image.cpu_addr, image.guest_size_bytes);
        }
    }
}

template <class P>
bool TextureCache<P>::UploadInChunks(Image& image) {
    // A whole upload of a very large image needs a staging buffer of its full size (rounded up to
    // a power of two: 1 GiB for 600 MiB of data), which can fail when memory is tight.
    constexpr u64 min_size = 64_MiB;
    constexpr u64 chunk_size = 32_MiB;
    constexpr ImageFlagBits excluded = ImageFlagBits::Sparse | ImageFlagBits::Converted |
                                       ImageFlagBits::AcceleratedUpload | ImageFlagBits::Unbacked |
                                       ImageFlagBits::TrimmedLevels;
    if (True(image.flags & excluded) || image.info.type != ImageType::e2D ||
        MapSizeBytes(image) < min_size) {
        return false;
    }
    for (u64 offset = 0; offset < image.guest_size_bytes; offset += chunk_size) {
        const std::array range{
            std::pair<u64, u64>{offset, std::min<u64>(offset + chunk_size, image.guest_size_bytes)}};
        const auto plan = PlanUnswizzleImageRows(image.gpu_addr, image.info, range);
        if (!plan.layout_valid) {
            if (offset == 0) {
                return false; // layout not handled: one staging buffer after all
            }
            continue;
        }
        if (plan.copies.empty()) {
            continue; // padding between levels or layers
        }
        auto staging = runtime.UploadStagingBuffer(plan.total_host_bytes);
        ExecuteUnswizzlePlan(*gpu_memory, plan, staging.mapped_span);
        image.UploadMemory(staging, plan.copies);
    }
    runtime.InsertUploadMemoryBarrier();
    return true;
}

template <class P>
bool TextureCache<P>::UploadsByLayer(const ImageBase& image) const noexcept {
    // Title 1's terrain arrays (8192x8192x68 ASTC, 1.45 GB each) get a few layers streamed in at a
    // time; re-decoding all of them for each took a 2 GiB staging buffer and stalled the GPU.
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        if (True(image.flags & ImageFlagBits::SparseLayers)) {
            return true;
        }
        return True(image.flags & ImageFlagBits::AcceleratedUpload) &&
               image.info.type == ImageType::e2D && image.info.resources.layers > 1 &&
               image.guest_size_bytes >= LayerUploadMinSize &&
               False(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::Sparse |
                                    ImageFlagBits::Remapped));
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size) {
    if (True(image.flags & ImageFlagBits::CpuModified) && image.dirty_layers.empty()) {
        return; // all of it is dirty already
    }
    const u32 layers = static_cast<u32>(image.info.resources.layers);
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        image.flags |= ImageFlagBits::CpuModified;
        image.dirty_layers.assign(Common::DivCeil(layers, 64U), 0);
    }
    const DAddr begin = (std::max<DAddr>)(cpu_addr, image.cpu_addr) - image.cpu_addr;
    const DAddr end = (std::min<DAddr>)(cpu_addr + size, image.cpu_addr + image.guest_size_bytes) -
                      image.cpu_addr;
    const u32 stride = image.info.layer_stride;
    if (end <= begin || stride == 0) {
        return;
    }
    const u32 last = (std::min)(static_cast<u32>((end - 1) / stride), layers - 1);
    for (u32 layer = static_cast<u32>(begin / stride); layer <= last; ++layer) {
        image.dirty_layers[layer / 64] |= u64{1} << (layer % 64);
    }
}

template <class P>
void TextureCache<P>::MarkRangeWritten(Image& image, DAddr cpu_addr, size_t size) {
    if (True(image.flags & ImageFlagBits::CpuModified) && image.dirty_ranges.empty()) {
        return; // all of it is dirty already
    }
    const DAddr begin = (std::max<DAddr>)(cpu_addr, image.cpu_addr);
    const DAddr end = (std::min<DAddr>)(cpu_addr + size, image.cpu_addr + image.guest_size_bytes);
    if (end <= begin) {
        return;
    }
    image.flags |= ImageFlagBits::CpuModified;
    image.dirty_ranges.emplace_back(begin - image.cpu_addr, end - begin);
}

template <class P>
void TextureCache<P>::RefreshSparseTiles(Image& image) {
    if constexpr (requires { &Image::BindTiles; }) {
        // The ranges written since the last upload, or, for a new image, all of it that can hold
        // data (the tiles the game maps there, or the blocks written).
        std::vector<std::pair<u64, u64>> ranges = std::move(image.dirty_ranges);
        image.dirty_ranges.clear();
        if (ranges.empty()) {
            ranges = Tegra::Nx2SparseDataRanges(image.gpu_addr, image.guest_size_bytes);
        }
        const std::vector<LevelRect> rects =
            SparseTileRects(image.info, ranges, image.SparseTileWidth(), image.SparseTileHeight(),
                            image.SparseTailLevel());
        if (rects.empty() || !BindSparseTiles(image, rects)) {
            runtime.TransitionImageLayout(image);
            return;
        }
        const size_t bytes = UnswizzledRectsBytes(image.info, rects);
        auto staging = runtime.UploadStagingBuffer(bytes);
        const std::vector<BufferImageCopy> copies =
            UnswizzleImageRects(*gpu_memory, image.gpu_addr, image.info, rects, staging.mapped_span);
        image.UploadMemory(staging, copies);
        runtime.InsertUploadMemoryBarrier();
        ++images_uploaded;
        uploaded_bytes += bytes;
    }
}

template <class P>
bool TextureCache<P>::BindSparseTiles(Image& image, std::span<const LevelRect> rects) {
    if constexpr (requires { &Image::BindTiles; }) {
        const u64 bound_bytes = ImageMemoryBytes(image);
        const bool bound = image.BindTiles(rects);
        // RegisterImage counts what an image has bound by then; later binds are added here.
        if (True(image.flags & ImageFlagBits::Registered)) {
            total_used_memory += ImageMemoryBytes(image) - bound_bytes;
        }
        return bound;
    } else {
        return false;
    }
}

template <class P>
bool TextureCache<P>::BackSparseTilesWhole(Image& image) {
    if constexpr (requires { &Image::BackTilesWhole; }) {
        const u64 bound_bytes = ImageMemoryBytes(image);
        const bool backed = image.BackTilesWhole();
        if (True(image.flags & ImageFlagBits::Registered)) {
            total_used_memory += ImageMemoryBytes(image) - bound_bytes;
        }
        return backed;
    } else {
        return false;
    }
}

template <class P>
bool TextureCache<P>::UploadWrittenLayers(Image& image) {
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        const u32 num_layers = static_cast<u32>(image.info.resources.layers);
        std::vector<u32> layers;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            if ((image.dirty_layers[layer / 64] >> (layer % 64)) & 1) {
                layers.push_back(layer);
            }
        }
        image.dirty_layers.clear();
        if (layers.empty() || layers.size() == num_layers) {
            return false;
        }
        UploadLayers(image, layers);
        return true;
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::RefreshSparseLayers(Image& image) {
    if constexpr (requires { &Image::BindLayers; }) {
        // The layers written since the last upload, or, for a new image, every layer any of whose
        // blocks has been written.
        const u32 num_layers = static_cast<u32>(image.info.resources.layers);
        const u64 stride = image.info.layer_stride;
        std::vector<u32> layers;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            const bool written =
                image.dirty_layers.empty()
                    ? !Tegra::Nx2SparseUnwritten(image.gpu_addr + layer * stride, stride)
                    : ((image.dirty_layers[layer / 64] >> (layer % 64)) & 1) != 0;
            if (written) {
                layers.push_back(layer);
            }
        }
        image.dirty_layers.clear();
        const std::vector<u32> bound =
            layers.empty() ? std::vector<u32>{} : BindSparseLayers(image, layers);
        if (bound.empty()) {
            runtime.TransitionImageLayout(image);
            return;
        }
        UploadLayers(image, bound);
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
std::vector<u32> TextureCache<P>::BindSparseLayers(Image& image, std::span<const u32> layers) {
    if constexpr (requires { &Image::BindLayers; }) {
        const u64 bound_bytes = ImageMemoryBytes(image);
        std::vector<u32> bound = image.BindLayers(layers);
        // RegisterImage counts what an image has bound by then; later binds are added here.
        if (True(image.flags & ImageFlagBits::Registered)) {
            total_used_memory += ImageMemoryBytes(image) - bound_bytes;
        }
        return bound;
    } else {
        return {};
    }
}

template <class P>
void TextureCache<P>::UploadLayers(Image& image, std::span<const u32> layers) {
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        if (False(image.flags & ImageFlagBits::AcceleratedUpload)) {
            // Plain formats: each layer unswizzled on its own into its layer of the image.
            ImageInfo layer_info = image.info;
            layer_info.resources.layers = 1;
            const size_t layer_size = CalculateUnswizzledSizeBytes(layer_info);
            auto staging = runtime.UploadStagingBuffer(layer_size * layers.size());
            std::vector<BufferImageCopy> copies;
            for (size_t i = 0; i < layers.size(); ++i) {
                const GPUVAddr layer_addr = image.gpu_addr + u64{layers[i]} * image.info.layer_stride;
                Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead>
                    swizzle_data(*gpu_memory, layer_addr, image.info.layer_stride,
                                 &swizzle_data_buffer);
                auto layer_copies =
                    UnswizzleImage(*gpu_memory, layer_addr, layer_info, swizzle_data,
                                   staging.mapped_span.subspan(i * layer_size, layer_size));
                for (BufferImageCopy copy : layer_copies) {
                    copy.buffer_offset += i * layer_size;
                    copy.image_subresource.base_layer = static_cast<s32>(layers[i]);
                    copies.push_back(copy);
                }
            }
            image.UploadMemory(staging, copies);
            ++images_uploaded;
            uploaded_bytes += u64{image.info.layer_stride} * layers.size();
            ++layer_uploads;
            return;
        }
        // Each layer's mip chain from the first level the host keeps on (as UploadImageContents).
        auto uploads = FullUploadSwizzles(image.info);
        const size_t level_offset = uploads[image.host_level_offset].buffer_offset;
        const size_t tail = image.info.layer_stride - level_offset;
        auto staging = runtime.UploadStagingBuffer(tail * layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            gpu_memory->ReadBlock(image.gpu_addr + u64{layers[i]} * image.info.layer_stride +
                                      level_offset,
                                  staging.mapped_span.data() + i * tail, tail,
                                  VideoCommon::CacheType::NoTextureCache);
        }
        uploads.erase(uploads.begin(), uploads.begin() + image.host_level_offset);
        for (auto& upload : uploads) {
            upload.buffer_offset -= level_offset;
        }
        runtime.AccelerateImageUploadLayers(image, staging, uploads, layers);
        ++images_uploaded;
        uploaded_bytes += tail * layers.size();
        ++layer_uploads;
    }
}

template <class P>
template <typename StagingBuffer>
void TextureCache<P>::UploadImageContents(Image& image, StagingBuffer& staging) {
    const std::span<u8> mapped_span = staging.mapped_span;
    const GPUVAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & ImageFlagBits::AcceleratedUpload) && image.host_level_offset != 0) {
        // Only levels >= host_level_offset exist on the host: read each layer's tail of the mip
        // chain into a compact buffer (layer stride = that tail's size) and decode those levels.
        auto uploads = FullUploadSwizzles(image.info);
        const size_t level_offset = uploads[image.host_level_offset].buffer_offset;
        const size_t tail = image.info.layer_stride - level_offset;
        for (s32 layer = 0; layer < image.info.resources.layers; ++layer) {
            gpu_memory->ReadBlock(gpu_addr + layer * image.info.layer_stride + level_offset,
                                  mapped_span.data() + layer * tail, tail,
                                  VideoCommon::CacheType::NoTextureCache);
        }
        uploads.erase(uploads.begin(), uploads.begin() + image.host_level_offset);
        for (auto& upload : uploads) {
            upload.buffer_offset -= level_offset;
        }
        runtime.AccelerateImageUpload(image, staging, uploads, 0, 0);
        return;
    }
    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        gpu_memory->ReadBlock(gpu_addr, mapped_span.data(), mapped_span.size_bytes(),
                              VideoCommon::CacheType::NoTextureCache);
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, staging, uploads, 0, 0);
        return;
    }

    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(
        *gpu_memory, gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);
    if (True(image.flags & ImageFlagBits::Converted)) {
        unswizzle_data_buffer.resize_destructive(image.unswizzled_size_bytes);
        auto copies = UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data,
                                    unswizzle_data_buffer, &unswizzle_worker);
        ConvertImage(unswizzle_data_buffer, image.info, mapped_span, copies);
        image.UploadMemory(staging, copies);
    } else {
        const auto copies = UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data,
                                          mapped_span, &unswizzle_worker);
        image.UploadMemory(staging, copies);
    }
}

template <class P>
ImageViewId TextureCache<P>::CreateImageView(const TICEntry& raw_config) {
    const bool native = gpu_memory->UsesNativeTiling();
    if (native && raw_config.format.Value() == Tegra::Texture::TextureFormat{} &&
        RenderTrace::Enabled()) {
        static u32 traces{};
        if (traces++ < 64) {
            LOG_WARNING(HW_GPU,
                        "Zero-format image TIC: image={:#x} size={}x{}x{} header={} type={} "
                        "raw={:016x}/{:016x}/{:016x}/{:016x}",
                        raw_config.Address(), raw_config.Width(), raw_config.Height(),
                        raw_config.Depth(), static_cast<u32>(raw_config.header_version.Value()),
                        static_cast<u32>(raw_config.texture_type.Value()), raw_config.raw[0],
                        raw_config.raw[1], raw_config.raw[2], raw_config.raw[3]);
        }
    }
    if (native && raw_config.BaseLayer() && NativeDrawDiagnostics()) {
        static u32 texture_view_traces{};
        if (texture_view_traces++ < 8) {
            LOG_WARNING(HW_GPU, "Native texture view header: address={:#x}, dimensions={}x{}x{}, type={}, legacy_layer={}, raw={:016x}/{:016x}/{:016x}/{:016x}",
                        raw_config.Address(), raw_config.Width(), raw_config.Height(), raw_config.Depth(),
                        static_cast<u32>(raw_config.texture_type.Value()), raw_config.BaseLayer(),
                        raw_config.raw[0], raw_config.raw[1], raw_config.raw[2], raw_config.raw[3]);
        }
    }
    const TICEntry config = TextureViewHeader(raw_config, native);
    const ImageInfo info(config);
    ++bindings_epoch;
    if (info.type == ImageType::Buffer) {
        const ImageViewInfo view_info(config, 0);
        Churn::Add(Churn::ImageViewsCreated);
        return slot_image_views.insert(runtime, info, view_info, config.Address());
    }
    const u32 layer_offset = config.BaseLayer() * info.layer_stride;
    const GPUVAddr image_gpu_addr = config.Address() - layer_offset;
    const ImageId image_id = FindOrInsertImage(info, image_gpu_addr);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    ImageBase& image = slot_images[image_id];
    const SubresourceBase base = image.TryFindBase(config.Address()).value();
    ASSERT(base.level == 0);
    const ImageViewInfo view_info(config, base.layer);
    const ImageViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);
    ImageViewBase& image_view = slot_image_views[image_view_id];
    image_view.flags |= ImageViewFlagBits::Strong;
    image.flags |= ImageFlagBits::Strong;
    return image_view_id;
}

template <class P>
ImageId TextureCache<P>::FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                           RelaxedOptions options) {
    if (const ImageId image_id = FindImage(info, gpu_addr, options); image_id) {
        return image_id;
    }
    return InsertImage(info, gpu_addr, options);
}

template <class P>
ImageId TextureCache<P>::FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                   RelaxedOptions options) {
    const std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        return ImageId{};
    }
    const bool broken_views =
        runtime.HasBrokenTextureViewFormats() || True(options & RelaxedOptions::ForceBrokenViews);
    const bool native_bgr = runtime.HasNativeBgr();
    const bool flexible_formats = True(options & RelaxedOptions::Format);
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, broken_views, native_bgr)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return !flexible_formats && existing.format == info.format;
            }
        } else if (IsSubresource(info, existing_image, gpu_addr, options, broken_views,
                                 native_bgr)) {
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return !flexible_formats && existing_image.info.format == info.format;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
bool TextureCache<P>::ImageCanRescale(ImageBase& image) {
    if (!image.info.rescaleable) {
        return false;
    }
    if (Settings::values.resolution_info.downscale && !image.info.downscaleable) {
        return false;
    }
    if (True(image.flags & (ImageFlagBits::Rescaled | ImageFlagBits::CheckingRescalable))) {
        return true;
    }
    if (True(image.flags & ImageFlagBits::IsRescalable)) {
        return true;
    }
    image.flags |= ImageFlagBits::CheckingRescalable;
    for (const auto& alias : image.aliased_images) {
        Image& other_image = slot_images[alias.id];
        if (!ImageCanRescale(other_image)) {
            image.flags &= ~ImageFlagBits::CheckingRescalable;
            return false;
        }
    }
    image.flags &= ~ImageFlagBits::CheckingRescalable;
    image.flags |= ImageFlagBits::IsRescalable;
    return true;
}

template <class P>
void TextureCache<P>::InvalidateScale(Image& image) {
    ++bindings_epoch;
    if (image.scale_tick <= frame_tick) {
        image.scale_tick = frame_tick + 1;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);
    for (const ImageViewId image_view_id : image_view_ids) {
        sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        slot_image_views.erase(image_view_id);
        Churn::Add(Churn::ImageViewsDestroyed);
    }
    image.image_view_ids.clear();
    image.image_view_infos.clear();
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];

#ifndef NDEBUG
        if constexpr (ENABLE_VALIDATION) {
            std::ranges::fill(channel_info.graphics_image_table.LiveBoundIds(), CORRUPT_ID);
            std::ranges::fill(channel_info.compute_image_table.LiveBoundIds(), CORRUPT_ID);
        }
#endif
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
u64 TextureCache<P>::GetScaledImageSizeBytes(const ImageBase& image) {
    const u64 scale_up = static_cast<u64>(Settings::values.resolution_info.up_scale *
                                          Settings::values.resolution_info.up_scale);
    const u64 down_shift = static_cast<u64>(Settings::values.resolution_info.down_shift +
                                            Settings::values.resolution_info.down_shift);
    const u64 image_size_bytes =
        static_cast<u64>((std::max)(image.guest_size_bytes, image.unswizzled_size_bytes));
    const u64 tentative_size = (image_size_bytes * scale_up) >> down_shift;
    const u64 fitted_size = Common::AlignUp(tentative_size, 1024);
    return fitted_size;
}

template <class P>
u64 TextureCache<P>::ImageMemoryBytes(const Image& image) const {
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        // No device memory but what BindLayers gave a SparseLayers image: at its full size (GBs
        // for title 1's sparse terrain images), deleting it would look to the collector like
        // freeing that much.
        if constexpr (requires { image.SparseBoundBytes(); }) {
            return image.SparseBoundBytes();
        }
        return 0;
    }
    u64 tentative_size = (std::max)(image.guest_size_bytes, image.unswizzled_size_bytes);
    if ((IsPixelFormatASTC(image.info.format) &&
         True(image.flags & ImageFlagBits::AcceleratedUpload)) ||
        True(image.flags & ImageFlagBits::Converted)) {
        tentative_size = TranscodedAstcSize(tentative_size, image.info.format);
    }
    return Common::AlignUp(tentative_size, 1024);
}

template <class P>
void TextureCache<P>::QueueAsyncDecode(Image& image, ImageId image_id) {
    UNIMPLEMENTED_IF(False(image.flags & ImageFlagBits::Converted));
    LOG_DEBUG(HW_GPU, "Queuing async texture decode {:#x}", image.gpu_addr);

    image.flags |= ImageFlagBits::IsDecoding;
    ++async_decodes_queued;
    auto decode = std::make_unique<AsyncDecodeContext>();
    auto* decode_ptr = decode.get();
    decode->image_id = image_id;
    async_decodes.push_back(std::move(decode));

    std::vector<u8> local_unswizzle_data_buffer(image.unswizzled_size_bytes, 0);
    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(*gpu_memory, image.gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);
    auto copies = UnswizzleImage(*gpu_memory, image.gpu_addr, image.info, swizzle_data, local_unswizzle_data_buffer);
    const size_t out_size = MapSizeBytes(image);
    if (False(image.flags & ImageFlagBits::ZeroFilled)) {
        // Until the decode lands (a second or more while title 1 streams ~40 textures a second)
        // a new image showed whatever its memory last held, often other textures. Zero blocks
        // read as opaque black (BC1) or transparent (BC3) instead.
        image.flags |= ImageFlagBits::ZeroFilled;
        auto zero_copies = copies;
        ConvertedLayout(image.info, zero_copies);
        auto staging = runtime.UploadStagingBuffer(out_size);
        std::memset(staging.mapped_span.data(), 0, out_size);
        image.UploadMemory(staging, zero_copies);
        runtime.InsertUploadMemoryBarrier();
    }

    auto func = [out_size, copies, info = image.info,
                 input = std::move(local_unswizzle_data_buffer),
                 async_decode = decode_ptr]() mutable {
        async_decode->decoded_data.resize_destructive(out_size);
        std::span copies_span{copies.data(), copies.size()};
        ConvertImage(input, info, async_decode->decoded_data, copies_span);

        // TODO: Do we need this lock?
        std::unique_lock lock{async_decode->mutex};
        async_decode->copies = std::move(copies);
        async_decode->complete = true;
    };
    texture_decode_worker.QueueWork(std::move(func));
}

template <class P>
void TextureCache<P>::QueueAsyncUnswizzle(Image& image, ImageId image_id) {
    if (True(image.flags & ImageFlagBits::IsDecoding)) {
        return;
    }

    image.flags |= ImageFlagBits::IsDecoding;

    unswizzle_queue.push_back({
        .image_id = image_id,
        .info = image.info
    });
}

template <class P>
void TextureCache<P>::TickAsyncDecode() {
    bool has_uploads{};
    auto i = async_decodes.begin();
    while (i != async_decodes.end()) {
        auto* async_decode = i->get();
        {
            // Release the lock before the context is erased below: unlocking it afterwards wrote
            // into freed memory (heap corruption, crashes minutes into title 1).
            std::unique_lock lock{async_decode->mutex};
            if (!async_decode->complete) {
                ++i;
                continue;
            }
        }
        if (async_decode->cancelled) {
            i = async_decodes.erase(i);
            continue;
        }
        Image& image = slot_images[async_decode->image_id];
        if (async_decode->decoded_data.size() > MapSizeBytes(image)) {
            LOG_ERROR(HW_GPU, "Async decode of {} bytes for a {}-byte image at {:#x}, dropped",
                      async_decode->decoded_data.size(), MapSizeBytes(image), image.gpu_addr);
            i = async_decodes.erase(i);
            continue;
        }
        auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
        std::memcpy(staging.mapped_span.data(), async_decode->decoded_data.data(),
                    async_decode->decoded_data.size());
        image.UploadMemory(staging, async_decode->copies);
        image.flags &= ~ImageFlagBits::IsDecoding;
        has_uploads = true;
        i = async_decodes.erase(i);
    }
    if (has_uploads) {
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
void TextureCache<P>::TickAsyncUnswizzle() {
    if (unswizzle_queue.empty()) {
        return;
    }

    if(current_unswizzle_frame > 0) {
        current_unswizzle_frame--;
        return;
    }

    PendingUnswizzle& task = unswizzle_queue.front();
    Image& image = slot_images[task.image_id];

    if (!task.initialized) {
        task.total_size = MapSizeBytes(image);
        task.staging_buffer = runtime.UploadStagingBuffer(task.total_size, true);

        const auto& info = image.info;
        const u32 bytes_per_block = BytesPerBlock(info.format);
        const u32 width_blocks = Common::DivCeil(info.size.width, 4u);
        const u32 height_blocks = Common::DivCeil(info.size.height, 4u);

        const u32 stride = width_blocks * bytes_per_block;
        const u32 aligned_height = height_blocks;
        task.bytes_per_slice = static_cast<size_t>(stride) * aligned_height;
        task.last_submitted_offset = 0;
        task.initialized = true;
    }

    // Read data
    if (task.current_offset < task.total_size) {
        const size_t remaining = task.total_size - task.current_offset;

        size_t copy_amount = (std::min)(swizzle_chunk_size, remaining);

        if (remaining > swizzle_chunk_size) {
            copy_amount = (copy_amount / task.bytes_per_slice) * task.bytes_per_slice;
            if (copy_amount == 0) copy_amount = task.bytes_per_slice;
        }

        gpu_memory->ReadBlock(image.gpu_addr + task.current_offset,
                              task.staging_buffer.mapped_span.data() + task.current_offset,
                              copy_amount);
        task.current_offset += copy_amount;
    }

    const bool is_final_batch = task.current_offset >= task.total_size;
    const size_t bytes_ready = task.current_offset - task.last_submitted_offset;
    const u32 complete_slices = static_cast<u32>(bytes_ready / task.bytes_per_slice);

    if (complete_slices >= swizzle_slices_per_batch || (is_final_batch && complete_slices > 0)) {
        const u32 z_start = static_cast<u32>(task.last_submitted_offset / task.bytes_per_slice);
        const u32 slices_to_process = (std::min)(complete_slices, swizzle_slices_per_batch);
        const u32 z_count = (std::min)(slices_to_process, image.info.size.depth - z_start);

        if (z_count > 0) {
            const auto uploads = FullUploadSwizzles(task.info);
            runtime.AccelerateImageUpload(image, task.staging_buffer, uploads, z_start, z_count);
            task.last_submitted_offset += (static_cast<size_t>(z_count) * task.bytes_per_slice);
        }
    }

    // Check if complete
    const u32 slices_submitted = static_cast<u32>(task.last_submitted_offset / task.bytes_per_slice);
    const bool all_slices_submitted = slices_submitted >= image.info.size.depth;

    if (is_final_batch && all_slices_submitted) {
        runtime.FreeDeferredStagingBuffer(task.staging_buffer);
        image.flags &= ~ImageFlagBits::IsDecoding;
        unswizzle_queue.pop_front();

        // Wait 4 frames to process the next entry
        current_unswizzle_frame = 4u;
    }
}

template <class P>
bool TextureCache<P>::ScaleUp(Image& image) {
    const bool has_copy = image.HasScaled();
    const bool rescaled = image.ScaleUp();
    if (!has_copy && image.HasScaled()) {
        total_used_memory += GetScaledImageSizeBytes(image);
    }
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
bool TextureCache<P>::ScaleDown(Image& image) {
    const bool rescaled = image.ScaleDown();
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
std::optional<DAddr> TextureCache<P>::ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const {
    if (const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr)) {
        return cpu_addr;
    }
    if (const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr, size)) {
        return cpu_addr;
    }
    // No CPU backing (title 1 renders a bloom chain at GPU 0xf0000-0x15b600): a stand-in address
    // fixed by the GPU address, at or above ~(1 << 40) so TrackImage skips it. A fresh address per
    // lookup made every use create another image (thousands a minute, never found again).
    constexpr GPUVAddr StandInRange = 1ULL << 40;
    if (gpu_addr >= StandInRange || size > StandInRange - gpu_addr - 1) {
        return std::nullopt;
    }
    return ~(1ULL << 40ULL) + gpu_addr;
}

template <class P>
ImageId TextureCache<P>::InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                     RelaxedOptions options) {
    std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        const DAddr fake_addr = ~(1ULL << 40ULL) + virtual_invalid_space;
        virtual_invalid_space += Common::AlignUp(CalculateGuestSizeInBytes(info), 32);
        cpu_addr = std::optional<DAddr>(fake_addr);
    }
    ASSERT_MSG(cpu_addr, "Tried to insert an image to an invalid gpu_addr={:#x}", gpu_addr);
    ImageInfo storage_info = info;
    storage_info.native_tiling = gpu_memory->UsesNativeTiling();
    if (storage_info.native_tiling && info.type == ImageType::e3D) {
        // Switch 2 3D images (lighting/fog volumes) are rare: log each one's shape.
        LOG_DEBUG(HW_GPU, "3D image at {:#x}: {}x{}x{} levels {} format {} block {}x{}x{}",
                 gpu_addr, info.size.width, info.size.height, info.size.depth, info.resources.levels,
                 static_cast<u32>(info.format), info.block.width, info.block.height, info.block.depth);
    }
    const ImageId image_id = JoinImages(storage_info, gpu_addr, *cpu_addr);
    if (!image_id) {
        return image_id;
    }
    const Image& image = slot_images[image_id];
    // Using "image.gpu_addr" instead of "gpu_addr" is important because it might be different
    const auto [it, is_new] = image_allocs_table.try_emplace(image.gpu_addr);
    if (is_new) {
        it->second = slot_image_allocs.insert();
    }
    slot_image_allocs[it->second].images.push_back(image_id);
    return image_id;
}

template <class P>
ImageId TextureCache<P>::JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr) {
    ImageInfo new_info = info;
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    const bool broken_views = runtime.HasBrokenTextureViewFormats();
    const bool native_bgr = runtime.HasNativeBgr();
    join_overlap_ids.clear();
    join_overlaps_found.clear();
    join_left_aliased_ids.clear();
    join_right_aliased_ids.clear();
    join_ignore_textures.clear();
    join_bad_overlap_ids.clear();
    join_copies_to_do.clear();
    join_alias_indices.clear();
    ++rt_stats.joins;
    const auto walk_start = std::chrono::steady_clock::now();
    const bool this_is_linear = info.type == ImageType::Linear;
    const auto region_check = [&](ImageId overlap_id, ImageBase& overlap) {
        ++rt_stats.overlaps_walked;
        if (True(overlap.flags & ImageFlagBits::Remapped)) {
            join_ignore_textures.insert(overlap_id);
            return;
        }
        const bool overlap_is_linear = overlap.info.type == ImageType::Linear;
        if (this_is_linear != overlap_is_linear) {
            return;
        }
        if (this_is_linear && overlap_is_linear) {
            if (info.pitch == overlap.info.pitch && gpu_addr == overlap.gpu_addr) {
                // Alias linear images with the same pitch
                join_left_aliased_ids.push_back(overlap_id);
            }
            return;
        }
        join_overlaps_found.insert(overlap_id);
        // A drift: a single-level, single-layer 2D target placed a little off a previous one
        // (Ocarina of Time's bump-allocated per-frame targets move 40 KB - 2 MB a frame). The
        // offset is under one layer, and not 0, so it is no mip or layer base of either image:
        // ResolveOverlap and both IsSubresource checks below would fail. Skips them.
        if (new_info.type == ImageType::e2D && overlap.info.type == ImageType::e2D &&
            new_info.resources.levels == 1 && overlap.info.resources.levels == 1 &&
            new_info.resources.layers == 1 && overlap.info.resources.layers == 1 &&
            new_info.layer_stride == overlap.info.layer_stride && gpu_addr != overlap.gpu_addr) {
            const u64 distance = gpu_addr > overlap.gpu_addr ? gpu_addr - overlap.gpu_addr
                                                             : overlap.gpu_addr - gpu_addr;
            const u64 limit = new_info.layer_stride != 0 ? new_info.layer_stride : size_bytes;
            if (distance < limit && distance < size_bytes &&
                distance < overlap.guest_size_bytes) {
                join_bad_overlap_ids.push_back(overlap_id);
                return;
            }
        }
        constexpr bool strict_size = true;
        const std::optional<OverlapResult> solution = ResolveOverlap(
            new_info, gpu_addr, cpu_addr, overlap, strict_size, broken_views, native_bgr);
        if (solution) {
            gpu_addr = solution->gpu_addr;
            cpu_addr = solution->cpu_addr;
            new_info.resources = solution->resources;
            join_overlap_ids.push_back(overlap_id);
            join_copies_to_do.emplace_back(JoinCopy{false, overlap_id});
            return;
        }
        constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        const ImageBase new_image_base(new_info, gpu_addr, cpu_addr);
        if (IsSubresource(new_info, overlap, gpu_addr, options, broken_views, native_bgr)) {
            join_left_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else if (IsSubresource(overlap.info, new_image_base, overlap.gpu_addr, options,
                                 broken_views, native_bgr)) {
            join_right_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else {
            join_bad_overlap_ids.push_back(overlap_id);
        }
    };
    ForEachImageInRegion(cpu_addr, size_bytes, region_check);
    const auto region_check_gpu = [&](ImageId overlap_id, ImageBase& overlap) {
        if (!join_overlaps_found.contains(overlap_id)) {
            if (True(overlap.flags & ImageFlagBits::Remapped)) {
                join_ignore_textures.insert(overlap_id);
            }
            if (overlap.gpu_addr == gpu_addr && overlap.guest_size_bytes == size_bytes) {
                join_ignore_textures.insert(overlap_id);
            }
        }
    };
    ForEachSparseImageInRegion(channel_state->gpu_memory.GetID(), gpu_addr, size_bytes,
                               region_check_gpu);
    rt_stats.walk_ns += NsSince(walk_start);
    rt_stats.bad_overlaps += static_cast<u32>(join_bad_overlap_ids.size());

    bool can_rescale = info.rescaleable;
    bool any_rescaled = false;
    for (const auto& copy : join_copies_to_do) {
        if (!can_rescale) {
            break;
        }
        Image& sibling = slot_images[copy.id];
        can_rescale &= ImageCanRescale(sibling);
        any_rescaled |= True(sibling.flags & ImageFlagBits::Rescaled);
    }

    can_rescale &= any_rescaled;

    if (can_rescale) {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleUp(sibling);
        }
    } else {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleDown(sibling);
        }
    }

    if constexpr (requires { runtime.UseOpaqueAstcFormat(); }) {
        // Opaque ASTC goes to BC1 rather than BC3 when it is re-compressed (the Vulkan runtime
        // decides). The guest data is already there when an image is first used.
        constexpr size_t MaxScanBytes = 64_MiB;
        if (IsPixelFormatASTC(new_info.format) && runtime.UseOpaqueAstcFormat()) {
            const size_t scan_bytes = CalculateGuestSizeInBytes(new_info);
            if (scan_bytes <= MaxScanBytes) {
                astc_scan_buffer.resize(scan_bytes);
                gpu_memory->ReadBlockUnsafe(gpu_addr, astc_scan_buffer.data(), scan_bytes);
                new_info.astc_opaque = AstcDataLooksOpaque(astc_scan_buffer);
                ++(new_info.astc_opaque ? astc_opaque_images : astc_alpha_images);
            }
        }
    }
    // A new render target that the clear binding it overwrites whole (every channel, the whole
    // image) needs no upload from guest memory first: Ocarina of Time creates ~4 such targets a
    // frame outdoors, and uploading them was ~15% of the GPU back thread.
    // Which test refused the last call (EXP7's new-image line).
    const char* clear_refusal = "";
    const auto overwritten_by_clear = [&] {
        if (!clearing_render_targets) {
            return (clear_refusal = "not clearing", false);
        }
        if (!join_copies_to_do.empty()) {
            return (clear_refusal = "join copies pending", false);
        }
        if (new_info.type != ImageType::e2D || new_info.resources.levels != 1 ||
            new_info.resources.layers != 1 || new_info.num_samples != 1) {
            return (clear_refusal = "not a plain 2D target", false);
        }
        const auto& regs = maxwell3d->regs;
        const auto& surface = regs.clear_surface;
        if (surface.layer != 0) {
            return (clear_refusal = "clear layer", false);
        }
        const u32 width = new_info.size.width;
        const u32 height = new_info.size.height;
        if (regs.clear_control.use_scissor != 0) {
            const auto& scissor = regs.scissor_test[0];
            if (scissor.min_x != 0 || scissor.min_y != 0 || scissor.max_x < width ||
                scissor.max_y < height) {
                // EXP7: the numbers, for the per-object depth targets whose width is padded to
                // 16 (fable52_answer_4: is the clear only a few pixels of padding short?).
                static std::atomic<u32> scissor_logged{};
                if (VideoCore::Nx2Debug::Experiment(7) != 0 &&
                    scissor_logged.fetch_add(1, std::memory_order_relaxed) < 40) {
                    LOG_WARNING(HW_GPU,
                                "Clear rule: scissor ({},{})-({},{}) vs new image format {} "
                                "{}x{} at {:#x}",
                                scissor.min_x.Value(), scissor.min_y.Value(),
                                scissor.max_x.Value(), scissor.max_y.Value(),
                                static_cast<u32>(new_info.format), width, height, gpu_addr);
                }
                return (clear_refusal = "scissor", false);
            }
        }
        // The clear covers the framebuffer's render area, the smallest bound target.
        // EDEN_NX2_EXP56 (live, 0 = off): not for the clear's own depth target, which follows
        // IsFullClear as an existing one does (its contents are dropped, the colour targets'
        // sizes aside). Ocarina binds a 1920x1024 depth buffer under 1280x720 colour and this
        // rule uploaded its 7.5 MiB from guest memory every frame (fable52_answer_3 s7).
        const bool depth_follows_full_clear = VideoCore::Nx2Debug::Experiment(56) != 0 &&
                                              regs.zeta_enable && regs.zeta.Address() == gpu_addr;
        for (u32 index = 0; index < regs.rt_control.count && index < NUM_RT; ++index) {
            const auto& rt = regs.rt[index];
            if (!depth_follows_full_clear && rt.Address() != 0 &&
                rt.format != Tegra::RenderTargetFormat::NONE &&
                (rt.width < width || rt.height < height)) {
                return (clear_refusal = "a smaller colour target", false);
            }
        }
        if (regs.zeta_enable && regs.zeta.Address() != 0 &&
            (regs.zeta_size.width < width || regs.zeta_size.height < height)) {
            return (clear_refusal = "a smaller depth target", false);
        }
        const bool is_depth_target = regs.zeta_enable && regs.zeta.Address() == gpu_addr;
        const bool full_stencil = surface.S && regs.stencil_front_mask == 0xFF;
        switch (VideoCore::Surface::GetFormatType(new_info.format)) {
        case VideoCore::Surface::SurfaceType::ColorTexture: {
            const u32 index = surface.RT;
            const bool covered = surface.R && surface.G && surface.B && surface.A &&
                                 index < NUM_RT && index < regs.rt_control.count &&
                                 regs.rt[index].Address() == gpu_addr;
            clear_refusal = covered ? "" : "colour channels or target";
            return covered;
        }
        case VideoCore::Surface::SurfaceType::Depth:
            clear_refusal = (is_depth_target && surface.Z) ? "" : "depth: not the bound target or no Z";
            return is_depth_target && surface.Z;
        case VideoCore::Surface::SurfaceType::Stencil:
            clear_refusal = (is_depth_target && full_stencil) ? "" : "stencil: not the bound target or mask";
            return is_depth_target && full_stencil;
        case VideoCore::Surface::SurfaceType::DepthStencil:
            clear_refusal = (is_depth_target && surface.Z && full_stencil) ? "" : "depth-stencil: not bound, no Z or stencil mask";
            return is_depth_target && surface.Z && full_stencil;
        default:
            return false;
        }
    };
    // EXP43 (fable_answer_12 2A): a cleared render target that drifted off exactly one image of
    // its shape takes that image's place (2: the image moves here) or retires it at once (1).
    if (clearing_render_targets && !join_bad_overlap_ids.empty() && join_copies_to_do.empty() &&
        join_left_aliased_ids.empty() && join_right_aliased_ids.empty() &&
        join_ignore_textures.empty()) {
        if (const ImageId old_id = RelocationCandidate(new_info, gpu_addr, cpu_addr);
            old_id && overwritten_by_clear()) {
            const GPUVAddr old_gpu_addr = slot_images[old_id].gpu_addr;
            const s64 drift = static_cast<s64>(gpu_addr) - static_cast<s64>(old_gpu_addr);
            RelocationShape& shape = relocation_shapes[RelocationShapeKey(new_info)];
            const int mode = VideoCore::Nx2Debug::Experiment(43);
            relocation_seen = true;
            ++relocation_candidates;
            if (shape.guard != 0 || relocation_guarded_addresses.contains(gpu_addr) ||
                relocation_guarded_addresses.contains(old_gpu_addr)) {
                ++relocation_guarded;
            } else {
                if (!shape.candidate_logged) {
                    shape.candidate_logged = true;
                    LOG_INFO(HW_GPU,
                             "Render target relocation candidate (EXP43={}): format {} {}x{} at "
                             "{:#x}, drifted {:+} bytes from the image at {:#x}",
                             mode, static_cast<u32>(new_info.format), new_info.size.width,
                             new_info.size.height, gpu_addr, drift, old_gpu_addr);
                }
                const bool relocate = mode == 2 || mode == 3 || mode == 4;
                if ((relocate || mode == 1) && !shape.relocated_logged) {
                    shape.relocated_logged = true;
                    LOG_INFO(HW_GPU,
                             "Render target {} (EXP43={}): format {} {}x{}, {:#x} -> {:#x}",
                             relocate ? "relocated" : "retired early", mode,
                             static_cast<u32>(new_info.format), new_info.size.width,
                             new_info.size.height, old_gpu_addr, gpu_addr);
                }
                if (relocate) {
                    ++relocation_relocated;
                    ++rt_stats.relocated_images;
                    Churn::Add(Churn::ImagesRelocated);
                    if (mode == 4) {
                        AddRelocationTombstone(slot_images[old_id], gpu_addr);
                    }
                    RelocateImage(old_id, gpu_addr, cpu_addr);
                    // The other bad overlaps are linked to it as to a new image (below).
                    Image& moved = slot_images[old_id];
                    for (const ImageId other_id : join_bad_overlap_ids) {
                        if (other_id == old_id) {
                            continue;
                        }
                        ImageBase& other = slot_images[other_id];
                        other.overlapping_images.push_back(old_id);
                        moved.overlapping_images.push_back(other_id);
                        if (other.info.resources.levels == 1 && other.info.block.depth == 0 &&
                            other.overlapping_images.size() > 1) {
                            other.flags |= ImageFlagBits::BadOverlap;
                        }
                        if (moved.info.resources.levels == 1 && moved.info.block.depth == 0 &&
                            moved.overlapping_images.size() > 1) {
                            moved.flags |= ImageFlagBits::BadOverlap;
                        }
                        WatchColdImage(other_id);
                    }
                    WatchColdImage(old_id);
                    return old_id;
                }
                if (mode == 1) {
                    // 2C: the predecessor goes now instead of 50 frames later; a new image is
                    // made below as without the switch.
                    ++relocation_retired;
                    ++rt_stats.retired_images;
                    Image& old = slot_images[old_id];
                    MarkStale(old); // before UntrackImage (see MarkStale)
                    if (True(old.flags & ImageFlagBits::Tracked)) {
                        UntrackImage(old, old_id);
                    }
                    UnregisterImage(old_id);
                    DeleteImage(old_id);
                    join_bad_overlap_ids.erase(std::remove(join_bad_overlap_ids.begin(),
                                                           join_bad_overlap_ids.end(), old_id),
                                               join_bad_overlap_ids.end());
                }
            }
        }
    }
    const auto create_start = std::chrono::steady_clock::now();
    const ImageId new_image_id = slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    rt_stats.create_ns += NsSince(create_start);
    if constexpr (requires(const Image& image) { image.FromPool(); }) {
        rt_stats.pooled_images += slot_images[new_image_id].FromPool() ? 1 : 0;
        Churn::Add(Churn::ImagesPooled, slot_images[new_image_id].FromPool() ? 1 : 0);
    }
    if constexpr (requires(const Image& image) { image.OutOfMemory(); }) {
        if (slot_images[new_image_id].OutOfMemory()) {
            // Left out of the cache: lookups see no image (a null view) and make it again once
            // the collector has freed memory.
            slot_images.erase(new_image_id);
            static u32 logged{};
            if (logged++ < 32) {
                LOG_ERROR(HW_GPU, "No memory for the {}x{}x{} image at {:#x} (format {}); skipping it",
                          new_info.size.width, new_info.size.height, new_info.size.depth, gpu_addr,
                          static_cast<u32>(new_info.format));
            }
            return ImageId{};
        }
    }
    Image& new_image = slot_images[new_image_id];
    ++bindings_epoch;

    new_image.allocation_tick = frame_tick;
    {
        // Which shapes are being created (periodic log): format, type, levels, layers, size.
        const u64 key = (u64{static_cast<u32>(new_info.format)} << 56) |
                        (u64{static_cast<u32>(new_info.type)} << 52) |
                        (u64{static_cast<u32>(new_info.resources.levels) & 0xF} << 48) |
                        (u64{static_cast<u32>(new_info.resources.layers) & 0xFFF} << 36) |
                        (u64{new_info.size.width & 0x3FFFF} << 18) | (new_info.size.height & 0x3FFFF);
        if (debug_stats) {
            auto& shape = created_shapes[key];
            ++shape.first;
            shape.second = gpu_addr;
        }
        ++images_created;
        Churn::Add(Churn::ImagesCreated);
    }

    if (!gpu_memory->IsContinuousRange(new_image.gpu_addr, new_image.guest_size_bytes) &&
        new_info.is_sparse) {
        new_image.flags |= ImageFlagBits::Sparse;
    }

    const auto ignored_delete_start = std::chrono::steady_clock::now();
    for (const ImageId overlap_id : join_ignore_textures) {
        Image& overlap = slot_images[overlap_id];
        // Remapped images no longer describe the current guest memory. Discard their GPU
        // contents rather than downloading them through the replacement mapping.
        if (True(overlap.flags & ImageFlagBits::GpuModified) &&
            False(overlap.flags & ImageFlagBits::Remapped)) {
            UNIMPLEMENTED();
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, overlap_id);
        }
        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }
    rt_stats.delete_ns += NsSince(ignored_delete_start);

    // Whether one of the GPU copies below writes all of a single-level 2D image, so an upload
    // before it is never seen. Ocarina of Time creates a D32 image at each new size of a
    // dynamic-resolution target over the same memory, uploading 6-14 MiB of it each time.
    const auto overwritten_by_copies = [&] {
        const ImageInfo& image_info = new_image.info;
        constexpr ImageFlagBits unusual = ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                                          ImageFlagBits::SparseTiles |
                                          ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked;
        if (image_info.type != ImageType::e2D || image_info.resources.levels != 1 ||
            image_info.resources.layers != 1 || image_info.num_samples != 1 ||
            True(new_image.flags & (unusual | ImageFlagBits::Converted |
                                    ImageFlagBits::AsynchronousDecode))) {
            return false;
        }
        const auto blocks = [](const ImageInfo& of, s32 level) {
            const Extent3D size = MipSize(of.size, static_cast<u32>(level));
            return Extent2D{
                .width = Common::DivCeil(size.width, DefaultBlockWidth(of.format)),
                .height = Common::DivCeil(size.height, DefaultBlockHeight(of.format)),
            };
        };
        const Extent2D new_blocks = blocks(image_info, 0);
        const auto covers = [&](const ImageInfo& overlap_info, s32 level) {
            const Extent2D overlap_blocks = blocks(overlap_info, level);
            return overlap_blocks.width >= new_blocks.width &&
                   overlap_blocks.height >= new_blocks.height;
        };
        for (const JoinCopy& copy : join_copies_to_do) {
            const ImageBase& overlap = slot_images[copy.id];
            if (overlap.info.type != ImageType::e2D || overlap.info.num_samples != 1 ||
                True(overlap.flags & unusual) ||
                GetFormatType(overlap.info.format) != GetFormatType(image_info.format) ||
                BytesPerBlock(overlap.info.format) != BytesPerBlock(image_info.format) ||
                overlap.info.astc_opaque != image_info.astc_opaque) {
                continue;
            }
            if (!copy.is_alias) {
                // Copied whole (MakeShrinkImageCopies) when the GPU wrote it.
                const std::optional<SubresourceBase> base = new_image.TryFindBase(overlap.gpu_addr);
                if (True(overlap.flags & ImageFlagBits::GpuModified) && base &&
                    base->level == 0 && base->layer == 0) {
                    return true;
                }
                continue;
            }
            if (!overlap.IsSafeGpuCopy()) {
                continue;
            }
            // The extent AddImageAlias gives the copy: the smaller of the two at the base found.
            constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
            if (std::ranges::find(join_left_aliased_ids, copy.id) != join_left_aliased_ids.end()) {
                const std::optional<SubresourceBase> base =
                    FindSubresource(image_info, overlap, new_image.gpu_addr, options, false, true);
                if (base && covers(overlap.info, base->level)) {
                    return true;
                }
            } else {
                const std::optional<SubresourceBase> base =
                    FindSubresource(overlap.info, new_image, overlap.gpu_addr, options, false, true);
                if (base && base->level == 0 && base->layer == 0 && covers(overlap.info, 0)) {
                    return true;
                }
            }
        }
        return false;
    };
    const bool clear_skip = overwritten_by_clear();
    // EDEN_NX2_EXP8 bit 2: skip the upload the copies overwrite.
    const bool copy_skip =
        !clear_skip && (VideoCore::Nx2Debug::Experiment(8) & 2) != 0 && overwritten_by_copies();
    if (VideoCore::Nx2Debug::Experiment(7) != 0 && new_image.guest_size_bytes >= 1_MiB)
        [[unlikely]] {
        // Why a large new image was (or was not) uploaded from guest memory.
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 300) {
            u32 aliases = 0;
            u32 gpu_written = 0;
            for (const JoinCopy& copy : join_copies_to_do) {
                aliases += copy.is_alias ? 1 : 0;
                const bool gpu = True(slot_images[copy.id].flags & ImageFlagBits::GpuModified);
                gpu_written += gpu ? 1 : 0;
            }
            LOG_WARNING(HW_GPU,
                        "New image format {} {}x{}x{} levels {} layers {} at {:#x}: {} copies "
                        "({} aliases, {} GPU-written), {} bad overlaps, clearing {}, covered {}, "
                        "upload {}{}{}",
                        static_cast<u32>(new_info.format), new_info.size.width,
                        new_info.size.height, new_info.size.depth, new_info.resources.levels,
                        new_info.resources.layers, new_image.gpu_addr, join_copies_to_do.size(),
                        aliases, gpu_written, join_bad_overlap_ids.size(),
                        clearing_render_targets, overwritten_by_copies(),
                        clear_skip  ? "skipped (clear)"
                        : copy_skip ? "skipped (copies)"
                                    : "done",
                        clear_skip || clear_refusal[0] == '\0' ? "" : ", clear refused: ",
                        clear_skip ? "" : clear_refusal);
        }
    }
    const auto refresh_start = std::chrono::steady_clock::now();
    if (clear_skip) {
        // Stays CpuModified: UpdateRenderTargets' PrepareImage drops the upload only when
        // IsFullClear agrees the clear covers the whole image, and uploads it otherwise.
        runtime.TransitionImageLayout(new_image);
    } else if (copy_skip) {
        // As RefreshContents leaves it, with the contents the copies write.
        new_image.flags &= ~ImageFlagBits::CpuModified;
        TrackImage(new_image, new_image_id);
        runtime.TransitionImageLayout(new_image);
    } else {
        // TODO: Only upload what we need
        RefreshContents(new_image, new_image_id);
    }
    rt_stats.refresh_ns += NsSince(refresh_start);
    if (!relocation_tombstones.empty()) {
        CheckTombstoneCreation(new_image_id, new_image, clear_skip); // EXP43=4
    }

    if (can_rescale) {
        ScaleUp(new_image);
    } else {
        ScaleDown(new_image);
    }

    std::ranges::sort(join_copies_to_do, [this](const JoinCopy& lhs, const JoinCopy& rhs) {
        const ImageBase& lhs_image = slot_images[lhs.id];
        const ImageBase& rhs_image = slot_images[rhs.id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });

    ImageBase& new_image_base = new_image;
    for (const ImageId aliased_id : join_right_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(new_image_base, aliased, new_image_id, aliased_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_left_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(aliased, new_image_base, aliased_id, new_image_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_bad_overlap_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        aliased.overlapping_images.push_back(new_image_id);
        new_image.overlapping_images.push_back(aliased_id);
        if (aliased.info.resources.levels == 1 && aliased.info.block.depth == 0 &&
            aliased.overlapping_images.size() > 1) {
            aliased.flags |= ImageFlagBits::BadOverlap;
        }
        if (new_image.info.resources.levels == 1 && new_image.info.block.depth == 0 &&
            new_image.overlapping_images.size() > 1) {
            new_image.flags |= ImageFlagBits::BadOverlap;
        }
        WatchColdImage(aliased_id);
    }
    if (!join_bad_overlap_ids.empty()) {
        WatchColdImage(new_image_id);
    }

    const auto joined_delete_start = std::chrono::steady_clock::now();
    for (const auto& copy_object : join_copies_to_do) {
        Image& overlap = slot_images[copy_object.id];
        if (copy_object.is_alias) {
            if (!overlap.IsSafeGpuCopy()) {
                continue;
            }
            if (overlap.info.num_samples != new_image.info.num_samples) {
                continue;
            }
            const auto alias_pointer = join_alias_indices.find(copy_object.id);
            if (alias_pointer == join_alias_indices.end()) {
                continue;
            }
            const AliasedImage& aliased = new_image.aliased_images[alias_pointer->second];
            CopyImage(new_image_id, aliased.id, aliased.copies);
            new_image.modification_tick = overlap.modification_tick;
            continue;
        }
        if (True(overlap.flags & ImageFlagBits::GpuModified)) {
            new_image.flags |= ImageFlagBits::GpuModified;
            const auto& resolution = Settings::values.resolution_info;
            const SubresourceBase base = new_image.TryFindBase(overlap.gpu_addr).value();
            const u32 up_scale = can_rescale ? resolution.up_scale : 1;
            const u32 down_shift = can_rescale ? resolution.down_shift : 0;
            auto copies = MakeShrinkImageCopies(new_info, overlap.info, base, up_scale, down_shift);
            if (overlap.info.num_samples != new_image.info.num_samples) {
                runtime.CopyImageMSAA(new_image, overlap, copies);
            } else {
                runtime.CopyImage(new_image, overlap, copies);
            }
            new_image.modification_tick = overlap.modification_tick;
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, copy_object.id);
        }
        UnregisterImage(copy_object.id);
        DeleteImage(copy_object.id);
    }
    rt_stats.delete_ns += NsSince(joined_delete_start);

    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
std::optional<typename TextureCache<P>::BlitImages> TextureCache<P>::GetBlitImages(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
    const Tegra::Engines::Fermi2D::Config& copy) {

    constexpr auto FIND_OPTIONS = RelaxedOptions::Samples;
    const GPUVAddr dst_addr = dst.Address();
    const GPUVAddr src_addr = src.Address();
    ImageInfo dst_info(dst);
    ImageInfo src_info(src);
    const bool can_be_depth_blit =
        dst_info.format == src_info.format && copy.filter == Tegra::Engines::Fermi2D::Filter::Point;
    ImageId dst_id;
    ImageId src_id;
    RelaxedOptions try_options = FIND_OPTIONS;
    if (can_be_depth_blit) {
        try_options |= RelaxedOptions::Format;
    }
    do {
        has_deleted_images = false;
        src_id = FindImage(src_info, src_addr, try_options);
        dst_id = FindImage(dst_info, dst_addr, try_options);
        if (!copy.must_accelerate) {
            do {
                if (!src_id && !dst_id) {
                    return std::nullopt;
                }
                if (src_id && True(slot_images[src_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                if (dst_id && True(slot_images[dst_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                return std::nullopt;
            } while (false);
        }
        const ImageBase* const src_image = src_id ? &slot_images[src_id] : nullptr;
        if (src_image && src_image->info.num_samples > 1) {
            RelaxedOptions find_options{FIND_OPTIONS | RelaxedOptions::ForceBrokenViews};
            src_id = FindOrInsertImage(src_info, src_addr, find_options);
            dst_id = FindOrInsertImage(dst_info, dst_addr, find_options);
            if (has_deleted_images) {
                continue;
            }
            break;
        }
        if (can_be_depth_blit) {
            const ImageBase* const dst_image = dst_id ? &slot_images[dst_id] : nullptr;
            DeduceBlitImages(dst_info, src_info, dst_image, src_image);
            if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
                continue;
            }
        }
        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions{});
        }
        if (!dst_id) {
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions{});
        }
    } while (has_deleted_images);
    if (!src_id || !dst_id) {
        return std::nullopt;
    }
    const ImageBase& src_image = slot_images[src_id];
    const ImageBase& dst_image = slot_images[dst_id];
    const bool native_bgr = runtime.HasNativeBgr();
    if (GetFormatType(dst_info.format) != GetFormatType(dst_image.info.format) ||
        GetFormatType(src_info.format) != GetFormatType(src_image.info.format) ||
        !VideoCore::Surface::IsViewCompatible(dst_info.format, dst_image.info.format, false,
                                              native_bgr) ||
        !VideoCore::Surface::IsViewCompatible(src_info.format, src_image.info.format, false,
                                              native_bgr)) {
        // Make sure the images match the expected format.
        do {
            has_deleted_images = false;
            src_id = FindOrInsertImage(src_info, src_addr, RelaxedOptions{});
            dst_id = FindOrInsertImage(dst_info, dst_addr, RelaxedOptions{});
        } while (has_deleted_images);
        if (!src_id || !dst_id) {
            return std::nullopt;
        }
    }
    return {BlitImages{
        .dst_id = dst_id,
        .src_id = src_id,
        .dst_format = dst_info.format,
        .src_format = src_info.format,
    }};
}

template <class P>
ImageId TextureCache<P>::FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr) {
    const std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        return ImageId{};
    }
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, false, true)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return true;
            }
        } else if (IsSubCopy(info, existing_image, gpu_addr)) {
            if (True(existing_image.flags & ImageFlagBits::Unbacked)) {
                static u32 logged{};
                if (logged++ < 64) {
                    LOG_INFO(HW_GPU, "DMA copy at {:#x} ({}x{}) into unbacked image {:#x}", gpu_addr,
                             info.size.width, info.size.height, existing_image.gpu_addr);
                }
            }
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return true;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
std::pair<u32, u32> TextureCache<P>::PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                     bool mark_as_modified) {
    // As in BlitImage: land a parked deferred clear before the DMA write.
    runtime.FlushDeferredClear();
    const auto& image = slot_images[dst_id];
    const auto base = image.TryFindBase(base_addr);
    PrepareImage(dst_id, mark_as_modified, false);
    const auto& new_image = slot_images[dst_id];
    lru_cache.Touch(new_image.lru_index, frame_tick);
    return std::make_pair(base->level, base->layer);
}

template <class P>
SamplerId TextureCache<P>::FindSampler(const TSCEntry& config, bool compute) {
    if (std::ranges::all_of(config.raw, [](u64 value) { return value == 0; })) {
        return NULL_SAMPLER_ID;
    }
    const auto [pair, is_new] = channel_state->samplers.try_emplace(config);
    if (is_new) {
        pair->second = slot_samplers.insert(runtime, config);
    }
    return pair->second;
}

template <class P>
ImageViewId TextureCache<P>::FindColorBuffer(size_t index) {
    const auto& regs = maxwell3d->regs;
    if (index >= regs.rt_control.count) {
        return ImageViewId{};
    }
    const auto& rt = regs.rt[index];
    const GPUVAddr gpu_addr = rt.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    if (rt.format == Tegra::RenderTargetFormat::NONE) {
        return ImageViewId{};
    }
    const ImageInfo info(regs.rt[index], regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindDepthBuffer() {
    const auto& regs = maxwell3d->regs;
    if (!regs.zeta_enable) {
        return ImageViewId{};
    }
    const GPUVAddr gpu_addr = regs.zeta.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    const ImageInfo info(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr) {
    static constexpr u32 MaxRenderTargetSize = 32768;
    static constexpr s32 MaxRenderTargetLayers = 2048;
    if (info.size.width == 0 || info.size.height == 0 || info.size.width > MaxRenderTargetSize ||
        info.size.height > MaxRenderTargetSize || info.resources.layers > MaxRenderTargetLayers) {
        LOG_ERROR(HW_GPU, "Ignoring {}x{} render target with {} layers at {:#x}", info.size.width,
                  info.size.height, info.resources.layers, gpu_addr);
        return ImageViewId{};
    }
    ImageId image_id{};
    bool delete_state = has_deleted_images;
    do {
        has_deleted_images = false;
        image_id = FindOrInsertImage(info, gpu_addr);
        delete_state |= has_deleted_images;
    } while (has_deleted_images);
    has_deleted_images = delete_state;
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    Image& image = slot_images[image_id];
    const ImageViewType view_type = RenderTargetImageViewType(info);
    SubresourceBase base;
    if (image.info.type == ImageType::Linear) {
        base = SubresourceBase{.level = 0, .layer = 0};
    } else {
        base = image.TryFindBase(gpu_addr).value();
    }
    const s32 layers = image.info.type == ImageType::e3D ? info.size.depth : info.resources.layers;
    if (RenderTrace::SampleFrame(frame_tick)) {
        LOG_DEBUG(HW_GPU,
                  "Render trace target match: docked={} requested={:#x} {}x{} layers={} format={} "
                  "stride={} block={}/{}/{} selected={:#x} {}x{} layers={} format={} stride={} "
                  "block={}/{}/{} base_level={} base_layer={} aliases={} overlaps={}",
                  Settings::IsDockedMode(), gpu_addr, info.size.width, info.size.height, layers,
                  info.format, info.layer_stride, info.block.width, info.block.height,
                  info.block.depth, image.gpu_addr, image.info.size.width, image.info.size.height,
                  image.info.resources.layers, image.info.format, image.info.layer_stride,
                  image.info.block.width, image.info.block.height, image.info.block.depth,
                  base.level, base.layer, image.aliased_images.size(), image.overlapping_images.size());
    }
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        static u32 logged{};
        if (logged++ < 64) {
            LOG_INFO(HW_GPU, "Render target in unbacked image {:#x} at {:#x}: {}x{} level {} layer {}",
                     image.gpu_addr, gpu_addr, info.size.width, info.size.height, base.level, base.layer);
        }
    }
    const SubresourceRange range{
        .base = base,
        .extent = {.levels = 1, .layers = layers},
    };
    return FindOrEmplaceImageView(image_id, ImageViewInfo(view_type, info.format, range));
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 32> images;
    boost::container::small_vector<ImageMapId, 32> maps;
    ForEachCPUPage(cpu_addr, size, [this, &images, &maps, cpu_addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const ImageMapId map_id : it->second) {
            ImageMapView& map = slot_map_views[map_id];
            if (map.picked) {
                continue;
            }
            if (!map.Overlaps(cpu_addr, size)) {
                continue;
            }
            map.picked = true;
            maps.push_back(map_id);
            Image& image = slot_images[map.image_id];
            if (True(image.flags & ImageFlagBits::Picked)) {
                continue;
            }
            image.flags |= ImageFlagBits::Picked;
            images.push_back(map.image_id);
            if constexpr (BOOL_BREAK) {
                if (func(map.image_id, image)) {
                    return true;
                }
            } else {
                func(map.image_id, image);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
    for (const ImageMapId map_id : maps) {
        slot_map_views[map_id].picked = false;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                              Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& gpu_page_table = gpu_page_table_storage[*storage_id * 2];
    ForEachGPUPage(gpu_addr, size,
                   [this, &gpu_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = gpu_page_table.find(page);
                       if (it == gpu_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                                 Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& sparse_page_table = gpu_page_table_storage[*storage_id * 2 + 1];
    ForEachGPUPage(gpu_addr, size,
                   [this, &sparse_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = sparse_page_table.find(page);
                       if (it == sparse_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseSegment(ImageBase& image, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, GPUVAddr, DAddr, size_t>::type;
    constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
    const auto segments = gpu_memory->GetSubmappedRange(image.gpu_addr, image.guest_size_bytes);
    for (const auto& [gpu_addr, size] : segments) {
        std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        ASSERT(cpu_addr);
        if constexpr (RETURNS_BOOL) {
            if (func(gpu_addr, *cpu_addr, size)) {
                break;
            }
        } else {
            func(gpu_addr, *cpu_addr, size);
        }
    }
}

template <class P>
ImageViewId TextureCache<P>::FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info) {
    Image& image = slot_images[image_id];
    if (const ImageViewId image_view_id = image.FindView(info); image_view_id) {
        return image_view_id;
    }
    if (True(image.flags & ImageFlagBits::Unbacked) &&
        False(image.flags & ImageFlagBits::SparseLayers)) {
        // How the game uses an image too large to back (e.g. a sparse virtual texture).
        LOG_INFO(HW_GPU, "View of unbacked image {:#x}: type {} format {} levels {}+{} layers {}+{}",
                 image.gpu_addr, static_cast<u32>(info.type), static_cast<u32>(info.format),
                 info.range.base.level, info.range.extent.levels, info.range.base.layer,
                 info.range.extent.layers);
    }
    const ImageViewId image_view_id = slot_image_views.insert(runtime, info, image_id, image, slot_images);
    Churn::Add(Churn::ImageViewsCreated);
    image.InsertView(info, image_view_id);
    ++bindings_epoch;
    return image_view_id;
}

template <class P>
void TextureCache<P>::RegisterImage(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += ImageMemoryBytes(slot_images[image_id]);
    image.lru_index = lru_cache.Insert(image_id, frame_tick);

    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->gpu_page_table)[page].push_back(image_id);
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        auto map_id =
            slot_map_views.insert(image.gpu_addr, image.cpu_addr, image.guest_size_bytes, image_id);
        page_filter.Mark(image.cpu_addr, image.guest_size_bytes);
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes,
                       [this, map_id](u64 page) { page_table[page].push_back(map_id); });
        image.map_view_id = map_id;
        return;
    }
    boost::container::small_vector<ImageViewId, 16> sparse_maps;
    ForEachSparseSegment(
        image, [this, image_id, &sparse_maps](GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
            auto map_id = slot_map_views.insert(gpu_addr, cpu_addr, size, image_id);
            page_filter.Mark(cpu_addr, size);
            ForEachCPUPage(cpu_addr, size,
                           [this, map_id](u64 page) { page_table[page].push_back(map_id); });
            sparse_maps.push_back(map_id);
        });
    sparse_views.emplace(image_id, std::move(sparse_maps));
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->sparse_page_table)[page].push_back(image_id);
    });
}

template <class P>
void TextureCache<P>::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already registered image");
    image.flags &= ~ImageFlagBits::Registered;
    image.flags &= ~ImageFlagBits::BadOverlap;
    lru_cache.Free(image.lru_index);
    const auto& clear_page_table =
        [image_id](u64 page, ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>& selected_page_table) {
            const auto page_it = selected_page_table.find(page);
            if (page_it == selected_page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageId>& image_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_ids, image_id);
            if (vector_it == image_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page={:#x}",
                           page << YUZU_PAGEBITS);
                return;
            }
            image_ids.erase(vector_it);
        };
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, &clear_page_table](u64 page) {
        clear_page_table(page, (*channel_state->gpu_page_table));
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        const auto map_id = image.map_view_id;
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes, [this, map_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_map_ids, map_id);
            if (vector_it == image_map_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page={:#x}",
                           page << YUZU_PAGEBITS);
                return;
            }
            image_map_ids.erase(vector_it);
        });
        slot_map_views.erase(map_id);
        return;
    }
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, &clear_page_table](u64 page) {
        clear_page_table(page, (*channel_state->sparse_page_table));
    });
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map_range = slot_map_views[map_view_id];
        const DAddr cpu_addr = map_range.cpu_addr;
        const std::size_t size = map_range.size;
        ForEachCPUPage(cpu_addr, size, [this, image_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            auto vector_it = image_map_ids.begin();
            while (vector_it != image_map_ids.end()) {
                ImageMapView& map = slot_map_views[*vector_it];
                if (map.image_id != image_id) {
                    vector_it++;
                    continue;
                }
                if (!map.picked) {
                    map.picked = true;
                }
                vector_it = image_map_ids.erase(vector_it);
            }
        });
        slot_map_views.erase(map_view_id);
    }
    sparse_views.erase(it);
}

template <class P>
void TextureCache<P>::TrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(False(image.flags & ImageFlagBits::Tracked));
    image.flags |= ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
        }
        return;
    }
    if (True(image.flags & ImageFlagBits::Registered)) {
        auto it = sparse_views.find(image_id);
        ASSERT(it != sparse_views.end());
        auto& sparse_maps = it->second;
        for (auto& map_view_id : sparse_maps) {
            const auto& map = slot_map_views[map_view_id];
            const DAddr cpu_addr = map.cpu_addr;
            const std::size_t size = map.size;
            device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
        }
        return;
    }
    ForEachSparseSegment(image,
                         [this]([[maybe_unused]] GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
                             device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
                         });
}

template <class P>
void TextureCache<P>::UntrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
        }
        return;
    }
    ASSERT(True(image.flags & ImageFlagBits::Registered));
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map = slot_map_views[map_view_id];
        const DAddr cpu_addr = map.cpu_addr;
        const std::size_t size = map.size;
        device_memory.UpdatePagesCachedCount(cpu_addr, size, -1);
    }
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id, bool immediate_delete) {
    ImageBase& image = slot_images[image_id];
    ++images_deleted;
    Churn::Add(Churn::ImagesDeleted);
    ++bindings_epoch;
    MarkStale(image);
    // Drop every decode still queued for it, whatever IsDecoding says: an image re-decoded while
    // a decode was in flight has two, and the first to finish clears the flag. A decode left
    // behind was uploaded into the next image to take this slot (another texture's data: garbage
    // textures, and a staging overrun when that image was smaller).
    for (auto& async_decode : async_decodes) {
        if (async_decode->image_id == image_id) {
            async_decode->cancelled = true;
        }
    }
    if (image.HasScaled()) {
        total_used_memory -= GetScaledImageSizeBytes(image);
    }
    // TickFrame sets the total from the driver, which can be under the sum of these estimates.
    total_used_memory -= (std::min)(total_used_memory, ImageMemoryBytes(slot_images[image_id]));
    const u64 freed_bytes = (image.HasScaled() ? GetScaledImageSizeBytes(image) : 0) +
                            ImageMemoryBytes(slot_images[image_id]);
    const GPUVAddr gpu_addr = image.gpu_addr;
    const auto alloc_it = image_allocs_table.find(gpu_addr);
    if (alloc_it == image_allocs_table.end()) {
        ASSERT_MSG(false, "Trying to delete an image alloc that does not exist in address {:#x}",
                   gpu_addr);
        return;
    }
    const ImageAllocId alloc_id = alloc_it->second;
    std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        ASSERT_MSG(false, "Trying to delete an image that does not exist");
        return;
    }
    ASSERT_MSG(False(image.flags & ImageFlagBits::Tracked), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Mark render targets as dirty
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    for (const AliasedImage& alias : image.aliased_images) {
        ImageBase& other_image = slot_images[alias.id];
        [[maybe_unused]] const size_t num_removed_aliases =
            std::erase_if(other_image.aliased_images, [image_id](const AliasedImage& other_alias) {
                return other_alias.id == image_id;
            });
        other_image.CheckAliasState();
        ASSERT_MSG(num_removed_aliases == 1, "Invalid number of removed aliases: {}",
                   num_removed_aliases);
    }
    DropOverlapLinks(image_id);
    for (const ImageViewId image_view_id : image_view_ids) {
        if (!immediate_delete) {
            sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        }
        slot_image_views.erase(image_view_id);
        Churn::Add(Churn::ImageViewsDestroyed);
    }
    const bool retained = RetainPendingImageDownloads(image_id, slot_images[image_id],
                                                       uncommitted_downloads, committed_downloads);
    if (retained) {
        LOG_INFO(Render_Vulkan, "Retained image for queued download: id={}", image_id.Value());
    } else if (!immediate_delete) {
        if constexpr (requires { runtime.CurrentTick(); }) {
            gpu_sentenced_images.push_back(
                {runtime.CurrentTick(), freed_bytes, std::move(slot_images[image_id])});
            gpu_sentenced_bytes += freed_bytes;
        } else {
            sentenced_images.Push(std::move(slot_images[image_id]));
            sentenced_image_bytes[sentenced_image_index] += freed_bytes;
        }
    }
    slot_images.erase(image_id);

    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_allocs_table.erase(alloc_it);
        slot_image_allocs.erase(alloc_id);
    }
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
#ifndef NDEBUG
        if constexpr (ENABLE_VALIDATION) {
            std::ranges::fill(channel_info.graphics_image_table.LiveBoundIds(), CORRUPT_ID);
            std::ranges::fill(channel_info.compute_image_table.LiveBoundIds(), CORRUPT_ID);
        }
#endif
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::DropOverlapLinks(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
    for (const ImageId overlap_id : image.overlapping_images) {
        ImageBase& other_image = slot_images[overlap_id];
        [[maybe_unused]] const size_t num_removed_overlaps = std::erase_if(
            other_image.overlapping_images,
            [image_id](const ImageId other_overlap_id) { return other_overlap_id == image_id; });
        other_image.CheckBadOverlapState();
        ASSERT_MSG(num_removed_overlaps == 1, "Invalid number of removed overlapps: {}",
                   num_removed_overlaps);
    }
}

template <class P>
void TextureCache<P>::RelocateImage(ImageId image_id, GPUVAddr gpu_addr, DAddr cpu_addr) {
    Image& image = slot_images[image_id];
    // 1. The GPU's data at the old address is lost, as when the collector deletes the image.
    MarkStale(image); // before UntrackImage (see MarkStale)
    if (True(image.flags & ImageFlagBits::Tracked)) {
        UntrackImage(image, image_id);
    }
    UnregisterImage(image_id);
    // RegisterImage below counts it again.
    total_used_memory -= (std::min)(total_used_memory, ImageMemoryBytes(image));
    // 2. No overlap links: the chain of drifted images does not form.
    DropOverlapLinks(image_id);
    image.overlapping_images.clear();
    image.CheckBadOverlapState();
    // 3. Out of the alloc entry of the old address (InsertImage files it under the new one).
    if (const auto alloc_it = image_allocs_table.find(image.gpu_addr);
        alloc_it != image_allocs_table.end()) {
        const ImageAllocId alloc_id = alloc_it->second;
        std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
        std::erase(alloc_images, image_id);
        if (alloc_images.empty()) {
            image_allocs_table.erase(alloc_it);
            slot_image_allocs.erase(alloc_id);
        }
    } else {
        ASSERT_MSG(false, "Relocating an image without an alloc at {:#x}", image.gpu_addr);
    }
    // Lookups by address (descriptors, bound render targets) must not keep finding its views
    // under the old address; the views themselves, and the framebuffers keyed by them, stay.
    for (const ImageViewId image_view_id : image.image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image.image_view_ids);
    // 4. The new address. Contents count as the guest's (the clear overwrites them whole, and
    // PrepareImage's invalidate path drops the upload, as for a new image).
    image.gpu_addr = gpu_addr;
    image.cpu_addr = cpu_addr;
    image.cpu_addr_end = cpu_addr + image.guest_size_bytes;
    image.flags &= ~ImageFlagBits::GpuModified;
    image.flags |= ImageFlagBits::CpuModified;
    image.last_write_frame = ~u64{0};
    image.relocation_guard_noted = 0;
    // A new target to the asynchronous shaders' skip rule too: its draws wait until it was
    // cleared at this address in two render frames (ClassifyDrawTargets).
    image.last_clear_frame = ~u64{0};
    image.previous_clear_frame = ~u64{0};
    image.drawn_since_clear = false;
    // 5. Page tables, map view, page filter, LRU.
    RegisterImage(image_id);
    // 6. As DeleteImage: render targets and descriptor tables are looked up again.
    ++bindings_epoch;
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::RemoveImageViewReferences(std::span<const ImageViewId> removed_views) {
    ++bindings_epoch;
    // Their descriptors resolve to other views now: every texture memo goes (EXP32).
    ++texture_memo_epoch;
    for (const ImageViewId removed : removed_views) {
        const auto refs = image_view_descriptors.find(removed);
        if (refs == image_view_descriptors.end()) {
            continue;
        }
        for (const auto& [channel, descriptor] : refs->second) {
            // The channel may be gone, or its entry point to another view since.
            if (std::ranges::find(active_channel_ids, channel) == active_channel_ids.end()) {
                continue;
            }
            auto& image_views = channel_storage[channel].image_views;
            if (const auto it = image_views.find(descriptor);
                it != image_views.end() && it->second == removed) {
                image_views.erase(it);
            }
        }
        image_view_descriptors.erase(refs);
    }
}

template <class P>
void TextureCache<P>::RemoveFramebuffers(std::span<const ImageViewId> removed_views) {
    runtime.FlushDeferredClear();
    const auto destroy = [this](FramebufferId framebuffer_id) {
        ASSERT(framebuffer_id);
        ++rt_stats.framebuffers_removed;
        if (framebuffer_id == last_framebuffer_id) {
            last_framebuffer_id = {};
            last_framebuffer_serial = 0;
        }
        sentenced_framebuffers.Push(std::move(slot_framebuffers[framebuffer_id]));
        slot_framebuffers.erase(framebuffer_id);
    };
    // A null id matches every unused slot of a key: only the full scan finds those.
    if (std::ranges::any_of(removed_views, [](ImageViewId view) { return !view; })) {
        rt_stats.framebuffers_scanned += static_cast<u32>(framebuffers.size());
        auto it = framebuffers.begin();
        while (it != framebuffers.end()) {
            if (it->first.Contains(removed_views)) {
                const FramebufferId framebuffer_id = it->second;
                UnindexFramebuffer(framebuffer_id, it->first);
                destroy(framebuffer_id);
                it = framebuffers.erase(it);
            } else {
                ++it;
            }
        }
        return;
    }
    // The same framebuffers the scan finds (every key that holds one of the views), by index.
    boost::container::small_vector<FramebufferId, 16> removed;
    for (const ImageViewId view : removed_views) {
        const auto it = framebuffers_by_view.find(view);
        if (it == framebuffers_by_view.end()) {
            continue;
        }
        for (const FramebufferId framebuffer_id : it->second) {
            if (std::ranges::find(removed, framebuffer_id) == removed.end()) {
                removed.push_back(framebuffer_id);
            }
        }
    }
    rt_stats.framebuffers_scanned += static_cast<u32>(removed.size());
    for (const FramebufferId framebuffer_id : removed) {
        const auto key_it = framebuffer_keys.find(framebuffer_id);
        ASSERT(key_it != framebuffer_keys.end());
        const RenderTargets key = key_it->second;
        UnindexFramebuffer(framebuffer_id, key);
        [[maybe_unused]] const size_t erased = framebuffers.erase(key);
        ASSERT(erased == 1);
        destroy(framebuffer_id);
    }
}

template <class P>
void TextureCache<P>::UnindexFramebuffer(FramebufferId framebuffer_id, const RenderTargets& key) {
    const auto unindex_view = [this, framebuffer_id](ImageViewId view) {
        if (!view) {
            return;
        }
        const auto it = framebuffers_by_view.find(view);
        if (it == framebuffers_by_view.end()) {
            return;
        }
        auto& ids = it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), framebuffer_id), ids.end());
        if (ids.empty()) {
            framebuffers_by_view.erase(it);
        }
    };
    std::ranges::for_each(key.color_buffer_ids, unindex_view);
    unindex_view(key.depth_buffer_id);
    framebuffer_keys.erase(framebuffer_id);
}

template <class P>
void TextureCache<P>::MarkModification(ImageBase& image) noexcept {
    image.flags |= ImageFlagBits::GpuModified;
    image.modification_tick = ++modification_tick;
    image.last_write_frame = RenderFrame();
    if (!clearing_render_targets) {
        image.drawn_since_clear = true;
    }
}

template <class P>
ImageId TextureCache<P>::RelocationCandidate(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                             DAddr cpu_addr) {
    if (!RelocationShaped(new_info)) {
        return ImageId{};
    }
    // Exactly one bad overlap of the same shape and layout; others (a neighbouring target of
    // another shape the drift moved into) stay bad overlaps, as for a new image.
    const auto same_shape = [&new_info](const ImageInfo& info) {
        return RelocationShaped(info) && info.format == new_info.format &&
               info.size.width == new_info.size.width &&
               info.size.height == new_info.size.height &&
               info.size.depth == new_info.size.depth && info.block == new_info.block &&
               info.layer_stride == new_info.layer_stride &&
               info.maybe_unaligned_layer_stride == new_info.maybe_unaligned_layer_stride &&
               info.tile_width_spacing == new_info.tile_width_spacing &&
               info.native_tiling == new_info.native_tiling &&
               info.rescaleable == new_info.rescaleable &&
               info.downscaleable == new_info.downscaleable &&
               info.is_sparse == new_info.is_sparse && info.astc_opaque == new_info.astc_opaque;
    };
    // Predecessors: images of the shape the GPU did not write in this frame (one it did may
    // still be read in this frame; it is never taken, and not counted).
    ImageId old_id{};
    u32 predecessors = 0;
    u32 same_frame = 0;
    const u64 render_frame = RenderFrame();
    for (const ImageId overlap_id : join_bad_overlap_ids) {
        const ImageBase& overlap = slot_images[overlap_id];
        if (!same_shape(overlap.info)) {
            continue;
        }
        if (overlap.last_write_frame == render_frame) {
            ++same_frame;
            continue;
        }
        ++predecessors;
        // EXP43=3 takes the most recently modified of several.
        if (!old_id || overlap.modification_tick > slot_images[old_id].modification_tick) {
            old_id = overlap_id;
        }
    }
    if (predecessors == 0) {
        if (same_frame != 0) {
            // Refused as written this frame (the render-frame clock line counts them by shape).
            ++relocation_same_frame;
            ++relocation_shapes[RelocationShapeKey(new_info)].same_frame_refused;
            relocation_seen = true;
        }
        return ImageId{};
    }
    if (predecessors > 1) {
        // A chain from before the switch (or two targets of the shape in reach): EXP43=1 and 2
        // need exactly one.
        ++relocation_crowded;
        relocation_seen = true;
        if (VideoCore::Nx2Debug::Experiment(43) < 3) {
            return ImageId{};
        }
    }
    const ImageBase& old = slot_images[old_id];
    // Read by the CPU.
    if (old.info.forced_flushed || old.info.dma_downloaded) {
        return ImageId{};
    }
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::Converted |
        ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding | ImageFlagBits::Remapped |
        ImageFlagBits::Alias | ImageFlagBits::CostlyLoad | ImageFlagBits::Rescaled |
        ImageFlagBits::TrimmedLevels;
    if (True(old.flags & excluded) || False(old.flags & ImageFlagBits::Registered) ||
        old.HasScaled() || !old.aliased_images.empty()) {
        return ImageId{};
    }
    // A drift, not a reuse of distant memory; real CPU addresses on both sides (no stand-ins).
    const u64 distance =
        gpu_addr > old.gpu_addr ? gpu_addr - old.gpu_addr : old.gpu_addr - gpu_addr;
    constexpr DAddr StandInBase = ~(1ULL << 40);
    if (distance == 0 || distance >= old.guest_size_bytes || cpu_addr >= StandInBase ||
        old.cpu_addr >= StandInBase) {
        return ImageId{};
    }
    if (new_info.is_sparse && !gpu_memory->IsContinuousRange(gpu_addr, old.guest_size_bytes)) {
        return ImageId{};
    }
    // Registered in this channel's GPU page table (UnregisterImage clears that one).
    const auto& gpu_pages = *channel_state->gpu_page_table;
    const auto page = gpu_pages.find(old.gpu_addr >> YUZU_PAGEBITS);
    if (page == gpu_pages.end() || std::ranges::find(page->second, old_id) == page->second.end()) {
        return ImageId{};
    }
    // No download queued for it.
    const auto queued = [old_id](const auto& downloads) {
        return std::ranges::any_of(downloads, [old_id](const auto& download) {
            return download.is_swizzle && download.object_id == old_id;
        });
    };
    if (queued(uncommitted_downloads) || std::ranges::any_of(committed_downloads, queued)) {
        return ImageId{};
    }
    // Not bound as another target of this clear.
    const auto& regs = maxwell3d->regs;
    for (u32 index = 0; index < regs.rt_control.count && index < NUM_RT; ++index) {
        const auto& rt = regs.rt[index];
        const GPUVAddr address = rt.Address();
        if (address != 0 && address != gpu_addr && rt.format != Tegra::RenderTargetFormat::NONE &&
            old.OverlapsGPU(address, 1)) {
            return ImageId{};
        }
    }
    if (regs.zeta_enable && regs.zeta.Address() != 0 && regs.zeta.Address() != gpu_addr &&
        old.OverlapsGPU(regs.zeta.Address(), 1)) {
        return ImageId{};
    }
    return old_id;
}

template <class P>
void TextureCache<P>::NoteRelocationGuard(ImageBase& image, RelocationGuard reason) {
    image.relocation_guard_noted |= reason;
    relocation_guarded_addresses[image.gpu_addr] |= reason;
    RelocationShape& shape = relocation_shapes[RelocationShapeKey(image.info)];
    if ((shape.guard & reason) != 0) {
        return;
    }
    shape.guard |= reason;
    // Each shape's first trip of each reason (a handful per session).
    LOG_INFO(HW_GPU,
             "Render target relocation guard (EXP43): format {} {}x{} now never relocates: {} "
             "(image at {:#x}, written in frame {}, frame {})",
             static_cast<u32>(image.info.format), image.info.size.width, image.info.size.height,
             reason == GuardSampled  ? "sampled in a later frame"
             : reason == GuardLoaded ? "rendered to without a clear in a later frame"
                                     : "read by the CPU",
             image.gpu_addr,
             image.last_write_frame == ~u64{0} ? s64{-1} : static_cast<s64>(image.last_write_frame),
             RenderFrame());
}

template <class P>
void TextureCache<P>::AddRelocationTombstone(const ImageBase& old_image, GPUVAddr moved_to) {
    relocation_tombstones.push_back(RelocationTombstone{
        .gpu_addr = old_image.gpu_addr,
        .cpu_addr = old_image.cpu_addr,
        .size = old_image.guest_size_bytes,
        .shape_key = RelocationShapeKey(old_image.info),
        .render_frame = RenderFrame(),
        .moved_to = moved_to,
        .gpu_modified = True(old_image.flags & ImageFlagBits::GpuModified),
    });
    tombstone_cpu_begin = (std::min)(tombstone_cpu_begin, old_image.cpu_addr);
    tombstone_cpu_end = (std::max)(tombstone_cpu_end, old_image.cpu_addr + old_image.guest_size_bytes);
    ++tombstones_made;
}

template <class P>
void TextureCache<P>::CheckTombstoneCreation(ImageId image_id, const ImageBase& image,
                                             bool overwritten) {
    for (const RelocationTombstone& tombstone : relocation_tombstones) {
        if (image.gpu_addr == tombstone.gpu_addr) {
            // A lookup that, without the relocation, would have found the old image. A clear
            // that overwrites it whole reads nothing; anything else is settled by its first use.
            if (!overwritten) {
                tombstone_born.push_back(TombstoneBorn{
                    .image_id = image_id, .gpu_addr = image.gpu_addr, .tombstone = tombstone});
            }
            return;
        }
    }
    if (!overwritten && image.cpu_addr < tombstone_cpu_end &&
        tombstone_cpu_begin < image.cpu_addr_end) {
        for (const RelocationTombstone& tombstone : relocation_tombstones) {
            if (image.cpu_addr < tombstone.cpu_addr + tombstone.size &&
                tombstone.cpu_addr < image.cpu_addr_end) {
                // Not at the old address: it read guest memory with or without the relocation
                // (a bad overlap gets no copies). Counted only.
                ++tombstone_overlap_reads;
                return;
            }
        }
    }
}

template <class P>
void TextureCache<P>::CheckTombstonePrepare(ImageId image_id, const ImageBase& image,
                                            bool is_modification, bool invalidate) {
    const auto it = std::ranges::find_if(tombstone_born, [&](const TombstoneBorn& born) {
        return born.image_id == image_id && born.gpu_addr == image.gpu_addr;
    });
    if (it == tombstone_born.end()) {
        return;
    }
    const TombstoneBorn born = *it;
    tombstone_born.erase(it);
    if (invalidate) {
        return; // overwritten whole: nothing read
    }
    NoteTombstoneHit(born.tombstone, is_modification ? TombstoneLoaded : TombstoneSampled,
                     image.gpu_addr, &image.info);
}

template <class P>
void TextureCache<P>::CheckTombstoneCpuRead(DAddr cpu_addr, u64 size) {
    if (cpu_addr >= tombstone_cpu_end || cpu_addr + size <= tombstone_cpu_begin) {
        return;
    }
    for (const RelocationTombstone& tombstone : relocation_tombstones) {
        if (tombstone.gpu_modified && cpu_addr < tombstone.cpu_addr + tombstone.size &&
            tombstone.cpu_addr < cpu_addr + size) {
            NoteTombstoneHit(tombstone, TombstoneCpuRead, cpu_addr, nullptr);
            return;
        }
    }
}

template <class P>
void TextureCache<P>::NoteTombstoneHit(const RelocationTombstone& tombstone, TombstoneRead reason,
                                       GPUVAddr address, const ImageInfo* info) {
    ++tombstone_hits[reason];
    if (tombstone_hits_logged >= 8) {
        return;
    }
    ++tombstone_hits_logged;
    const u64 key = tombstone.shape_key;
    LOG_WARNING(HW_GPU,
                "Relocation tombstone hit (EXP43=4): {} at {:#x}{}; the image at {:#x} (format {} "
                "{}x{}, {}GPU-written) moved to {:#x} {} render frame(s) ago",
                reason == TombstoneSampled  ? "sampled"
                : reason == TombstoneLoaded ? "rendered to without a clear"
                                            : "read by the CPU",
                address,
                info != nullptr ? fmt::format(" (format {} {}x{})", static_cast<u32>(info->format),
                                              info->size.width, info->size.height)
                                : std::string{},
                tombstone.gpu_addr, key >> 56, (key >> 18) & 0x3FFFF, key & 0x3FFFF,
                tombstone.gpu_modified ? "" : "not ", tombstone.moved_to,
                RenderFrame() - tombstone.render_frame);
}

template <class P>
void TextureCache<P>::TickRelocationTombstones() {
    if (!relocation_tombstones.empty()) {
        const u64 render_frame = RenderFrame();
        const auto expired = [render_frame](const RelocationTombstone& tombstone) {
            return render_frame > tombstone.render_frame + 2;
        };
        std::erase_if(relocation_tombstones, expired);
        tombstone_born.erase(std::remove_if(tombstone_born.begin(), tombstone_born.end(),
                                            [&](const TombstoneBorn& born) {
                                                return expired(born.tombstone);
                                            }),
                             tombstone_born.end());
        tombstone_cpu_begin = ~DAddr{0};
        tombstone_cpu_end = 0;
        for (const RelocationTombstone& tombstone : relocation_tombstones) {
            tombstone_cpu_begin = (std::min)(tombstone_cpu_begin, tombstone.cpu_addr);
            tombstone_cpu_end = (std::max)(tombstone_cpu_end, tombstone.cpu_addr + tombstone.size);
        }
    }
    if (frame_tick % 600 != 0 || (tombstones_made == 0 && VideoCore::Nx2Debug::Experiment(43) != 4)) {
        return;
    }
    const u64 hits = tombstone_hits[TombstoneSampled] + tombstone_hits[TombstoneLoaded] +
                     tombstone_hits[TombstoneCpuRead];
    LOG_INFO(HW_GPU,
             "Relocation tombstones (EXP43=4) last 600 frames: {} hits ({} sampled, {} rendered to "
             "without a clear, {} CPU read); {} tombstones made, {} live; {} new images read over "
             "one, not at its address",
             hits, tombstone_hits[TombstoneSampled], tombstone_hits[TombstoneLoaded],
             tombstone_hits[TombstoneCpuRead], tombstones_made, relocation_tombstones.size(),
             tombstone_overlap_reads);
    tombstone_hits = {};
    tombstones_made = 0;
    tombstone_overlap_reads = 0;
}

template <class P>
void TextureCache<P>::SynchronizeAliases(ImageId image_id) {
    boost::container::small_vector<const AliasedImage*, 8> aliased_images;
    Image& image = slot_images[image_id];
    bool any_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    bool any_modified = True(image.flags & ImageFlagBits::GpuModified);
    u64 most_recent_tick = image.modification_tick;
    for (const AliasedImage& aliased : image.aliased_images) {
        ImageBase& aliased_image = slot_images[aliased.id];
        if (image.modification_tick < aliased_image.modification_tick) {
            most_recent_tick = (std::max)(most_recent_tick, aliased_image.modification_tick);
            aliased_images.push_back(&aliased);
            any_rescaled |= True(aliased_image.flags & ImageFlagBits::Rescaled);
            any_modified |= True(aliased_image.flags & ImageFlagBits::GpuModified);
        }
    }
    if (aliased_images.empty()) {
        return;
    }
    const bool can_rescale = ImageCanRescale(image);
    if (any_rescaled) {
        if (can_rescale) {
            ScaleUp(image);
        } else {
            ScaleDown(image);
        }
    }
    image.modification_tick = most_recent_tick;
    if (any_modified) {
        image.flags |= ImageFlagBits::GpuModified;
    }
    std::ranges::sort(aliased_images, [this](const AliasedImage* lhs, const AliasedImage* rhs) {
        const ImageBase& lhs_image = slot_images[lhs->id];
        const ImageBase& rhs_image = slot_images[rhs->id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });
    const auto& resolution = Settings::values.resolution_info;
    for (const AliasedImage* const aliased : aliased_images) {
        if (!resolution.active || !any_rescaled) {
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        Image& aliased_image = slot_images[aliased->id];
        if (!can_rescale) {
            ScaleDown(aliased_image);
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        ScaleUp(aliased_image);
        CopyImage(image_id, aliased->id, aliased->copies);
    }
}

template <class P>
void TextureCache<P>::PrepareImage(ImageId image_id, bool is_modification, bool invalidate) {
    Image& image = slot_images[image_id];
    CheckCrossFrameUse(image, is_modification, invalidate);
    if (!tombstone_born.empty()) {
        CheckTombstonePrepare(image_id, image, is_modification, invalidate); // EXP43=4
    }
    if (invalidate) {
        image.flags &= ~(ImageFlagBits::CpuModified | ImageFlagBits::GpuModified);
        if (False(image.flags & ImageFlagBits::Tracked)) {
            TrackImage(image, image_id);
        }
    } else {
        RefreshContents(image, image_id);
        if (!image.aliased_images.empty()) {
            SynchronizeAliases(image_id);
        }
    }
    if (is_modification) {
        if constexpr (requires { &Image::BackTilesWhole; }) {
            if (True(image.flags & ImageFlagBits::SparseTiles) && image.SparseTilesShareZeroBlock()) {
                // The render would write the zeroed block all empty blocks share: each gets memory
                // of its own first, and the parts that hold data are loaded again.
                LOG_INFO(HW_GPU, "GPU write into sparse tiled image {:#x}; backing it whole",
                         image.gpu_addr);
                if (BackSparseTilesWhole(image)) {
                    image.dirty_ranges.clear();
                    RefreshSparseTiles(image);
                } else {
                    LOG_ERROR(HW_GPU,
                              "Sparse tiled image {:#x}: out of memory backing it whole, the GPU "
                              "write lands in the shared zeroed block",
                              image.gpu_addr);
                }
            }
        }
        if constexpr (requires { &Image::BindLayers; }) {
            if (True(image.flags & ImageFlagBits::SparseLayers)) {
                std::vector<u32> layers(static_cast<size_t>(image.info.resources.layers));
                std::iota(layers.begin(), layers.end(), 0U);
                // Layers over the image's memory cap stay unbacked (logged once).
                BindSparseLayers(image, layers);
            }
        }
        if (True(image.flags & ImageFlagBits::UnwrittenSparse) &&
            False(image.flags & ImageFlagBits::Remapped)) {
            // The GPU renders into it: these writes are lost, later uses get a backed image.
            LOG_WARNING(HW_GPU, "GPU write into unwritten sparse image {:#x}; replacing it",
                        image.gpu_addr);
            Tegra::Nx2SparseMarkWritten(image.gpu_addr, image.guest_size_bytes);
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
            ++bindings_epoch;
        }
        MarkModification(image);
    }
    lru_cache.Touch(image.lru_index, frame_tick);
}

template <class P>
void TextureCache<P>::PrepareImageView(ImageViewId image_view_id, bool is_modification,
                                       bool invalidate) {
    if (!image_view_id) {
        return;
    }
    const ImageViewBase& image_view = slot_image_views[image_view_id];
    if (image_view.IsBuffer()) {
        return;
    }
    PrepareImage(image_view.image_id, is_modification, invalidate);
}

template <class P>
void TextureCache<P>::CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies) {
    // As in BlitImage: land a parked deferred clear before the copy overwrites the image.
    runtime.FlushDeferredClear();
    Image& dst = slot_images[dst_id];
    Image& src = slot_images[src_id];
    if (True(dst.flags & ImageFlagBits::UnwrittenSparse) ||
        True(src.flags & ImageFlagBits::UnwrittenSparse)) {
        return;
    }
    if (dst.info.astc_opaque != src.info.astc_opaque) {
        // BC1 and BC3 blocks differ in size, so the host can't copy between them; both hold the
        // same guest data, which is uploaded from guest memory instead.
        return;
    }
    const bool is_rescaled = True(src.flags & ImageFlagBits::Rescaled);
    if (is_rescaled) {
        ASSERT(True(dst.flags & ImageFlagBits::Rescaled));
        const bool both_2d{src.info.type == ImageType::e2D && dst.info.type == ImageType::e2D};
        const auto& resolution = Settings::values.resolution_info;
        for (auto& copy : copies) {
            copy.src_offset.x = resolution.ScaleUp(copy.src_offset.x);
            copy.dst_offset.x = resolution.ScaleUp(copy.dst_offset.x);
            copy.extent.width = resolution.ScaleUp(copy.extent.width);
            if (both_2d) {
                copy.src_offset.y = resolution.ScaleUp(copy.src_offset.y);
                copy.dst_offset.y = resolution.ScaleUp(copy.dst_offset.y);
                copy.extent.height = resolution.ScaleUp(copy.extent.height);
            }
        }
    }
    const auto dst_format_type = GetFormatType(dst.info.format);
    const auto src_format_type = GetFormatType(src.info.format);
    if (src_format_type == dst_format_type) {
        if constexpr (HAS_EMULATED_COPIES) {
            if (!runtime.CanImageBeCopied(dst, src)) {
                return runtime.EmulateCopyImage(dst, src, copies);
            }
        }
        return runtime.CopyImage(dst, src, copies);
    }
    UNIMPLEMENTED_IF(dst.info.type != ImageType::e2D);
    UNIMPLEMENTED_IF(src.info.type != ImageType::e2D);
    if (runtime.ShouldReinterpret(dst, src)) {
        return runtime.ReinterpretImage(dst, src, copies);
    }
    for (const ImageCopy& copy : MakeImageCopyLayers(copies)) {
        UNIMPLEMENTED_IF(copy.dst_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_offset != Offset3D{});
        UNIMPLEMENTED_IF(copy.dst_offset != Offset3D{});

        const SubresourceBase dst_base{
            .level = copy.dst_subresource.base_level,
            .layer = copy.dst_subresource.base_layer,
        };
        const SubresourceBase src_base{
            .level = copy.src_subresource.base_level,
            .layer = copy.src_subresource.base_layer,
        };
        const SubresourceExtent dst_extent{.levels = 1, .layers = 1};
        const SubresourceExtent src_extent{.levels = 1, .layers = 1};
        const SubresourceRange dst_range{.base = dst_base, .extent = dst_extent};
        const SubresourceRange src_range{.base = src_base, .extent = src_extent};
        PixelFormat dst_format = GetAliasConversionFormat(dst.info.format, src.info.format);
        const PixelFormat src_format = GetAliasConversionFormat(src.info.format, dst.info.format);
        if (GetFormatType(src.info.format) == SurfaceType::DepthStencil &&
            GetFormatType(dst_format) == SurfaceType::ColorTexture &&
            BytesPerBlock(dst_format) == 4) {
            dst_format = PixelFormat::A8B8G8R8_UNORM;
        }
        const ImageViewInfo dst_view_info(ImageViewType::e2D, dst_format, dst_range);
        const ImageViewInfo src_view_info(ImageViewType::e2D, src_format, src_range);
        const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
        const ImageViewId src_view_id = FindOrEmplaceImageView(src_id, src_view_info);
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        [[maybe_unused]] const Extent3D expected_size{
            .width = (std::min)(dst_view.size.width, src_view.size.width),
            .height = (std::min)(dst_view.size.height, src_view.size.height),
            .depth = (std::min)(dst_view.size.depth, src_view.size.depth),
        };
        const Extent3D scaled_extent = [is_rescaled, expected_size]() {
            if (!is_rescaled) {
                return expected_size;
            }
            const auto& resolution = Settings::values.resolution_info;
            return Extent3D{
                .width = resolution.ScaleUp(expected_size.width),
                .height = resolution.ScaleUp(expected_size.height),
                .depth = expected_size.depth,
            };
        }();
        UNIMPLEMENTED_IF(copy.extent != scaled_extent);

        runtime.ConvertImage(dst_framebuffer, dst_view, src_view);
    }
}

template <class P>
void TextureCache<P>::BindRenderTarget(ImageViewId* old_id, ImageViewId new_id) {
    if (*old_id == new_id) {
        return;
    }
    if (new_id) {
        const ImageViewBase& old_view = slot_image_views[new_id];
        if (True(old_view.flags & ImageViewFlagBits::PreemtiveDownload) &&
            False(slot_images[old_view.image_id].flags & (ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels))) {
            const PendingDownload new_download{true, 0, old_view.image_id};
            uncommitted_downloads.emplace_back(new_download);
        }
    }
    *old_id = new_id;
}

template <class P>
std::pair<FramebufferId, ImageViewId> TextureCache<P>::RenderTargetFromImage(
    ImageId image_id, const ImageViewInfo& view_info) {
    const ImageViewId view_id = FindOrEmplaceImageView(image_id, view_info);
    const ImageBase& image = slot_images[image_id];
    const bool is_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    const bool is_color = GetFormatType(image.info.format) == SurfaceType::ColorTexture;
    const ImageViewId color_view_id = is_color ? view_id : ImageViewId{};
    const ImageViewId depth_view_id = is_color ? ImageViewId{} : view_id;
    Extent3D extent = MipSize(image.info.size, view_info.range.base.level);
    if (is_rescaled) {
        const auto& resolution = Settings::values.resolution_info;
        extent.width = resolution.ScaleUp(extent.width);
        if (image.info.type == ImageType::e2D) {
            extent.height = resolution.ScaleUp(extent.height);
        }
    }
    const u32 num_samples = image.info.num_samples;
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    const FramebufferId framebuffer_id = GetFramebufferId(RenderTargets{
        .color_buffer_ids = {color_view_id},
        .depth_buffer_id = depth_view_id,
        .size = {extent.width >> samples_x, extent.height >> samples_y},
        .is_rescaled = is_rescaled,
    });
    return {framebuffer_id, view_id};
}

template <class P>
bool TextureCache<P>::IsFullClear(ImageViewId id) {
    if (!id) {
        return true;
    }
    const ImageViewBase& image_view = slot_image_views[id];
    const ImageBase& image = slot_images[image_view.image_id];
    const Extent3D size = image_view.size;
    const auto& regs = maxwell3d->regs;
    const auto& scissor = regs.scissor_test[0];
    if (image.info.resources.levels > 1 || image.info.resources.layers > 1) {
        // Images with multiple resources can't be cleared in a single call
        return false;
    }
    if (regs.clear_control.use_scissor == 0) {
        // If scissor testing is disabled, the clear is always full
        return true;
    }
    // Make sure the clear covers all texels in the subresource
    return scissor.min_x == 0 && scissor.min_y == 0 && scissor.max_x >= size.width &&
           scissor.max_y >= size.height;
}

template <class P>
void TextureCache<P>::NoteFullClears() {
    // Not PrepareImage's invalidate path itself: UpdateRenderTargets(true) invalidates every bound
    // target, also those the clear leaves alone (the other colour targets, the depth target of a
    // colour clear), and those must not count as cleared here.
    const auto& clear = maxwell3d->regs.clear_surface;
    const bool color = clear.R || clear.G || clear.B || clear.A;
    const u64 render_frame = RenderFrame();
    const auto note = [&](ImageViewId view_id) {
        if (!view_id || view_id == NULL_IMAGE_VIEW_ID) {
            return;
        }
        const ImageViewBase& view = slot_image_views[view_id];
        if (view.IsBuffer() || !IsFullClear(view_id)) {
            return;
        }
        ImageBase& image = slot_images[view.image_id];
        if (image.last_clear_frame != render_frame) {
            // Only a clear after a write starts a new one: two clears with nothing between are
            // the same clear, wherever the frame's tick fell.
            if (image.drawn_since_clear) {
                image.previous_clear_frame = image.last_clear_frame;
            }
            image.last_clear_frame = render_frame;
        }
        image.drawn_since_clear = false;
    };
    if (color && clear.RT < NUM_RT) {
        note(render_targets.color_buffer_ids[clear.RT]);
    }
    // A stencil-only clear leaves the depth contents.
    if (clear.Z) {
        note(render_targets.depth_buffer_id);
    }
}

template <class P>
DrawTargets TextureCache<P>::ClassifyDrawTargets() {
    std::scoped_lock lock{mutex};
    const auto& regs = maxwell3d->regs;
    // The bound targets are the registers' while their dirty flag is clear (UpdateRenderTargets
    // finds them again when it is set); otherwise the images at the registers' addresses.
    const bool bound_current = !maxwell3d->dirty.flags[Dirty::RenderTargets];
    const u64 render_frame = RenderFrame();
    DrawTargets result{};
    bool any_target{};
    const auto take = [&](ImageViewId bound, const ImageInfo& info, GPUVAddr gpu_addr,
                          VideoCore::Surface::PixelFormat format) {
        any_target = true;
        const ImageBase* image{};
        if (bound_current) {
            if (bound && bound != NULL_IMAGE_VIEW_ID && !slot_image_views[bound].IsBuffer()) {
                image = &slot_images[slot_image_views[bound].image_id];
            }
        } else if (const ImageId image_id = FindImage(info, gpu_addr, RelaxedOptions{});
                   image_id) {
            image = &slot_images[image_id];
        }
        // Not registered yet: the draw is the first use of the target.
        const DrawTargetClass kind =
            image ? ClassifyClearedTarget(image->last_clear_frame, image->previous_clear_frame,
                                          render_frame)
                  : DrawTargetClass::FirstTime;
        if (kind > result.kind) {
            result = DrawTargets{
                .kind = kind,
                .gpu_addr = gpu_addr,
                .width = image ? image->info.size.width : info.size.width,
                .height = image ? image->info.size.height : info.size.height,
                .format = image ? image->info.format : format,
            };
        }
    };
    const u32 color_count = (std::min)(static_cast<u32>(regs.rt_control.count),
                                       static_cast<u32>(NUM_RT));
    for (u32 index = 0; index < color_count; ++index) {
        const auto& rt = regs.rt[index];
        const GPUVAddr gpu_addr = rt.Address();
        if (gpu_addr == 0 || rt.format == Tegra::RenderTargetFormat::NONE) {
            continue;
        }
        const ImageInfo info(rt, regs.anti_alias_samples_mode);
        take(render_targets.color_buffer_ids[index], info, gpu_addr, info.format);
    }
    if (regs.zeta_enable && regs.zeta.Address() != 0) {
        const ImageInfo info(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
        take(render_targets.depth_buffer_id, info, regs.zeta.Address(), info.format);
    }
    if (!any_target) {
        // No colour or depth target: what the draw does stays in storage it writes.
        result.kind = DrawTargetClass::Persistent;
    }
    return result;
}

template <class P>
void TextureCache<P>::CreateChannel(struct Tegra::Control::ChannelState& channel) {
    VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo>::CreateChannel(channel);
    ++texture_memo_epoch;
    ++bindings_epoch;
    const auto it = channel_map.find(channel.bind_id);
    auto* this_state = &channel_storage[it->second];
    const auto& this_as_ref = address_spaces[channel.memory_manager->GetID()];
    this_state->gpu_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2];
    this_state->sparse_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2 + 1];
}

/// Bind a channel for execution.
template <class P>
void TextureCache<P>::OnGPUASRegister([[maybe_unused]] size_t map_id) {
    gpu_page_table_storage.emplace_back();
    gpu_page_table_storage.emplace_back();
}

} // namespace VideoCommon
