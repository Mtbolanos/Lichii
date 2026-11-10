# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    common/addon_config.cpp
    video_core/vulkan_overlay_filter.cpp
    video_core/buffer_cache_ports.cpp
    video_core/unswizzle_workers.cpp
    video_core/aftermath_dump.cpp
    video_core/frame_dump_schedule.cpp
    video_core/nx2_stutter_trace.cpp
    video_core/host_mip_limits.cpp
    video_core/gpu_timestamp.cpp
    video_core/semaphore_waiters.cpp
    core/nx2_streamplay.cpp
    core/nx2_video_ready.cpp
    core/nx2_debug_server.cpp
    core/nx2_present_pacer.cpp
    common/dkb_cpu_trace.cpp
    common/dkb_frame_trace.cpp
    audio_core/nx2_sysmix.cpp
    audio_core/nx2_input_service.cpp
    audio_core/audio_in.cpp
    audio_core/sysmix_sink.cpp
    audio_core/sysmix_test_context.h
    video_core/ampere_starfox_test10.cpp
    video_core/ampere_yoshi_title.cpp
    video_core/ampere_yoshi_compute.cpp
    video_core/texture_rect_descriptor.cpp
    video_core/join_whole_copy.cpp
    audio_core/batch_decoder.cpp
    audio_core/microphone.cpp
    audio_core/hardware_audio_codec.cpp
    core/gpu_dirty_memory.cpp
    core/memory_acquire.cpp
    core/builtin_patches.cpp
    core/builtin_font_glyphs.cpp
    core/system_test_context.h
    video_core/ampere_branch_modifiers.cpp
    video_core/ampere_fast_fusion_logged.cpp
    video_core/fast_fusion_surface_store.cpp
    video_core/fast_fusion_byte_store.cpp
    video_core/vk_byte_addressed_shadow.cpp
    video_core/global_memory_coherence.cpp
    video_core/storage_buffer_pointer_tracking.cpp
    video_core/storage_write_extent.cpp
    video_core/storage_buffer_control_flow.cpp
    core/nup_validation.cpp
    core/nx_archive.cpp
    frontend_common/nx_archive.cpp
    core/vfs_vector.cpp
    core/file_sys_parsers.cpp
    input_common/calibration_lifetime.cpp
    video_core/ampere_logged_numeric.cpp
    video_core/ampere_logged_textures.cpp
    video_core/native_texture_header_index.cpp
    video_core/texture_alias_ticks.cpp
    video_core/sampled_views.cpp
    video_core/texture_handles.cpp
    video_core/ampere_selevet_modes.cpp
    video_core/ampere_selevet_spills.cpp
    video_core/blit_dither.cpp
    video_core/maxwell_numeric.cpp
    video_core/maxwell_permute_lod.cpp
    video_core/submission_policy.cpp
    video_core/native_pending_policy.cpp
    video_core/framebuffer_clear_layers.cpp
    video_core/framebuffer_limits.cpp
    video_core/unbacked_fallback.cpp
    video_core/fire_emblem_upscale.cpp
    video_core/dlss_contract.cpp
    video_core/ampere_fast_fusion.cpp
    video_core/ampere_async_copy.cpp
    video_core/ampere_trap_guards.cpp
    video_core/async_draw_policy.cpp
    video_core/async_shader_emission.cpp
    video_core/oot_network.cpp
    video_core/page_filter.cpp
    video_core/pipeline_optimization.cpp
    video_core/spirv_atomic_retry.cpp
    video_core/spirv_storage_alignment.cpp
    video_core/spirv_fragment_shuffle.cpp
    video_core/mesh_fallback.cpp
    video_core/multi_range_refresh.cpp
    video_core/command_fetch_order.cpp
    video_core/gpu_written_pages.cpp
    video_core/nx2_semaphore_mirror.cpp
    video_core/command_writer.cpp
    video_core/split_fence_wait.cpp
    video_core/nx2_written_pages.cpp
    video_core/command_snapshot_suspensions.cpp
    video_core/pending_maps.cpp
    video_core/maxwell_fast_method.cpp
    hid_core/sensor_info.cpp
    common/paged_scalar.cpp
    common/queue_lifetime.cpp
    common/queue_timed_wait.cpp
    common/range_set_move.cpp
    common/range_set_subtract.cpp
    video_core/uniform_cache_history.cpp
    video_core/x8d24_conversion.cpp
    video_core/ampere_ported_half_compare.cpp
    video_core/ampere_geometry_capture.cpp
    video_core/ampere_geometry_exit.cpp
    video_core/ampere_geometry_lowering.cpp
    video_core/ampere_ported_conversions.cpp
    video_core/ampere_ported_uniform_vote.cpp
    video_core/ampere_ported_textures.cpp
    video_core/ampere_tennis_i2f.cpp
    video_core/ampere_tennis_tessellation.cpp
    video_core/pipeline_cache_files.cpp
    video_core/shader_cache_v2.cpp
    video_core/evicted_pointer_slots.cpp
    video_core/shader_cache_v2_programs.cpp
    video_core/shader_cache_v2_driver.cpp
    video_core/vulkan_pipeline_binaries.cpp
    video_core/shader_precache.cpp
    video_core/shader_object_pool.cpp
    video_core/storage_binding_capacity.cpp
    video_core/native_indirect_reset.cpp
    video_core/native_qmd.cpp
    video_core/fast_draw.cpp
    video_core/gpu_command_safety.cpp
    video_core/syncpoint_actions.cpp
    video_core/nx2_video_av1.cpp
    video_core/nx2_video_render.cpp
    video_core/ampere_matrix.cpp
    video_core/spirv_software_matrix.cpp
    video_core/spirv_builtin_capabilities.cpp
    video_core/spirv_min_max_denorms.cpp
    video_core/spirv_nan_safe_min_max.cpp
    video_core/virtual_range_cache.cpp
    video_core/ocarina_upscale.cpp
    video_core/fsr4_support.cpp
    common/bit_field.cpp
    common/bit_util.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/core_pinning.cpp
    common/event.cpp
    common/fibers.cpp
    common/fs_iterate.cpp
    common/game_presets.cpp
    common/host_memory.cpp
    common/host_memory_failure.cpp
    common/host_memory_test_access.h
    core/host_memory_failure.cpp
    common/multi_level_page_table.cpp
    common/page_table.cpp
    common/param_package.cpp
    common/range_map.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/slot_vector.cpp
    common/sparse_large_vector.cpp
    common/undefined_fix.cpp
    common/unique_function.cpp
    common/zstd_compression.cpp
    core/core_timing.cpp
    core/thread_startup_policy.cpp
    core/nup_writer.cpp
    core/nup_metadata.cpp
    core/nup_read_bench.cpp
    core/nup_zstd_read.cpp
    core/nx2_image_layout.cpp
    core/nx2_movie_clock.cpp
    core/nx2_video_input_queue.cpp
    core/font_merge.cpp
    core/ldn.cpp
    core/internal_network/socket_proxy.cpp
    core/priority_queue.cpp
    core/internal_network/network.cpp
    video_core/memory_tracker.cpp
    video_core/invalidation_accumulator.cpp
    video_core/memory_manager_runs.cpp
    video_core/nx2_perf_draw_binds.cpp
    video_core/nx2_perf_texture_writes.cpp
    video_core/h264_packet.cpp
    video_core/nx2_gpu_mapping.cpp
    video_core/vertex_repack.cpp
    video_core/ampere_lowering.cpp
    video_core/ampere_geometry.cpp
    video_core/ampere_i2f.cpp
    video_core/ampere_returns.cpp
    video_core/ampere_guarded_ulea.cpp
    video_core/ampere_compute_lowering.cpp
    video_core/ampere_uniform_immediate_shift.cpp
    video_core/ampere_bindless_surfaces.cpp
    video_core/native_macro_fifo.cpp
    video_core/turing_macro_effects.cpp
    video_core/turing_macro_ram.cpp
    video_core/uniform_stream_reuse.cpp
    video_core/usage_tracker.cpp
    video_core/shader_invalidation.cpp
    video_core/descriptor_table.cpp
    video_core/descriptor_payload_growth.cpp
    video_core/global_address_tracking.cpp
    video_core/global_atomic_routing.cpp
    video_core/nvn_storage_descriptor.cpp
    video_core/ampere_f2i.cpp
    video_core/ampere_ldc.cpp
    video_core/ampere_vote.cpp
    video_core/shader_attributes.cpp
    video_core/indirect_reset.cpp
    video_core/stale_pages.cpp
    input_common/calibration_configuration_job.cpp
)

create_target_directory_groups(tests)

target_link_libraries(tests PRIVATE common core audio_core input_common video_core frontend_common)
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)

if (TARGET dynarmic::dynarmic)
    target_sources(tests PRIVATE core/window_trace.cpp)
    target_link_libraries(tests PRIVATE dynarmic::dynarmic)
endif()

target_compile_definitions(tests PRIVATE
    LICHII_NX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/core/nx_fixtures")

add_test(NAME tests COMMAND tests)

if (TARGET SDL3::SDL3)
    add_executable(sdl_window_state_tests yuzu_cmd/sdl_window_state.cpp)
    target_link_libraries(sdl_window_state_tests PRIVATE SDL3::SDL3 Catch2::Catch2WithMain)
    add_test(NAME sdl_window_state_tests COMMAND sdl_window_state_tests)
endif()

if (WIN32 AND ARCHITECTURE_x86_64)
    add_executable(event_wait_probe ../../tools/nx2/event_wait_probe.cpp)
    target_link_libraries(event_wait_probe PRIVATE common)
endif()

add_executable(update_checker_tests
    common/update_checker.cpp
    common/update_installer.cpp
    ../frontend_common/update_checker.cpp
    ../frontend_common/update_installer.cpp)
target_link_libraries(update_checker_tests PRIVATE common Catch2::Catch2WithMain)
add_test(NAME update_checker_tests COMMAND update_checker_tests)

# A libFuzzer target for the shader cache v2 reader, where the compiler has libFuzzer. Built on
# request: ninja shader_cache_v2_fuzzer, then shader_cache_v2_fuzzer <corpus directory>
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS -fsanitize=fuzzer)
check_cxx_source_compiles("
    #include <cstddef>
    #include <cstdint>
    extern \"C\" int LLVMFuzzerTestOneInput(const std::uint8_t*, std::size_t) { return 0; }"
    HAVE_LIBFUZZER)
unset(CMAKE_REQUIRED_FLAGS)
if (HAVE_LIBFUZZER)
    add_executable(shader_cache_v2_fuzzer EXCLUDE_FROM_ALL
        fuzz/shader_cache_v2_reader.cpp
        ../video_core/shader_cache_v2/cache_format.cpp)
    target_compile_options(shader_cache_v2_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(shader_cache_v2_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_libraries(shader_cache_v2_fuzzer PRIVATE common xxhash)
    if (BOOST_NO_HEADERS)
        target_link_libraries(shader_cache_v2_fuzzer PRIVATE Boost::crc)
    endif()
endif()

# The shader job scheduler on its own, so it can be built with a sanitizer: configure with
# -DSHADER_JOBS_SANITIZER=thread (or address,undefined), then build and run shader_jobs_tests.
set(SHADER_JOBS_SANITIZER "" CACHE STRING "Build shader_jobs_tests with -fsanitize=<value>")
add_executable(shader_jobs_tests
    video_core/shader_jobs.cpp
    ../video_core/shader_jobs.cpp)
target_link_libraries(shader_jobs_tests PRIVATE common Catch2::Catch2WithMain Threads::Threads)
if (SHADER_JOBS_SANITIZER)
    target_compile_options(shader_jobs_tests PRIVATE -fsanitize=${SHADER_JOBS_SANITIZER}
        -fno-omit-frame-pointer -fno-sanitize-recover=all -g)
    target_link_options(shader_jobs_tests PRIVATE -fsanitize=${SHADER_JOBS_SANITIZER})
endif()
add_test(NAME shader_jobs_tests COMMAND shader_jobs_tests)

# Stage translation on shader job workers over the tools/shader-sweep fixtures. For
# ThreadSanitizer the shader recompiler needs instrumenting too: configure a separate build with
# -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_CXX_FLAGS=-fsanitize=thread, then build and run
# shader_translation_tests.
add_executable(shader_translation_tests
    video_core/stage_translation.cpp
    ../video_core/shader_jobs.cpp
    ../video_core/shader_stage_translation.cpp)
target_link_libraries(shader_translation_tests PRIVATE common shader_recompiler
    Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(shader_translation_tests PRIVATE
    SHADER_SWEEP_FIXTURES="${PROJECT_SOURCE_DIR}/tools/shader-sweep/fixtures")
add_test(NAME shader_translation_tests COMMAND shader_translation_tests)

if (NOT ANDROID)
    target_sources(tests PRIVATE input_common/joycon_subcommand_response.cpp)
    target_link_libraries(tests PRIVATE SDL3::SDL3)

    add_executable(config_tests
        common/first_run_config.cpp
        ../frontend_common/config.cpp)
    target_link_libraries(config_tests PRIVATE common SimpleIni::SimpleIni Catch2::Catch2WithMain
        ${PLATFORM_LIBRARIES})
    add_test(NAME config_tests COMMAND config_tests)
endif()

# needed for vma
if (NOT MSVC)
    target_compile_options(tests PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()
