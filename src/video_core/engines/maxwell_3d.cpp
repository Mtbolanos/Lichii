// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_qmd_prediction.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

// EDEN_NX2_3D_METHODS: log the first write to each 3D register (finds methods Eden doesn't model).
static const bool g_trace_first_method_writes = std::getenv("EDEN_NX2_3D_METHODS") != nullptr;

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    memory_write_mask.reset();
    for (u32 i = 0; i < 16; ++i) {
        memory_write_mask[MAXWELL3D_REG_INDEX(const_buffer.buffer) + i] = true;
    }
    for (const auto method : {MAXWELL3D_REG_INDEX(launch_dma), MAXWELL3D_REG_INDEX(inline_data),
                             MAXWELL3D_REG_INDEX(report_semaphore.query),
                             MAXWELL3D_REG_INDEX(sync_info), MAXWELL3D_REG_INDEX(wait_for_idle)}) {
        memory_write_mask[method] = true;
    }
    for (size_t i = MacroRegistersStart; i < memory_write_mask.size(); ++i) {
        memory_write_mask[i] = true;
    }
    // Profiled runs take the same path: CallMethod counts and times its calls itself.
    fast_maxwell3d = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::Method) && !record_write_history &&
                     !g_trace_first_method_writes;
    const char* skip_same = std::getenv("EDEN_NX2_SKIP_SAME");
    skip_same_value_writes = skip_same == nullptr || skip_same[0] != '0';
    perf_verify = VideoCommon::Nx2Perf::Verify();
}

Maxwell3D::~Maxwell3D() {
    // Its held-back draw batch (EXP22, FrontFlushPending) goes with it.
    VideoCommon::GpuSplit::Get().DropPendingFlush(this);
}

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    auto const c = shadow_state.shadow_ram_control;
    if (c == Regs::ShadowRamControl::Track || c == Regs::ShadowRamControl::TrackWithFilter)
        return shadow_state.reg_array[method] = argument;
    else if (c == Regs::ShadowRamControl::Replay)
        return shadow_state.reg_array[method];
    return argument;
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::Sink};
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ApplyPairs(Core::System& system, const u64* pairs, u32 count) {
    // EDEN_NX2_RING_PAIRS: state writes the front would have sunk (never in execution_mask),
    // applied as ConsumeSinkImpl applies the sink. The shadow RAM mode cannot change inside a
    // block: shadow_ram_control is executable, and any other entry ends the block.
    const auto control = shadow_state.shadow_ram_control;
    const bool track = control == Regs::ShadowRamControl::Track ||
                       control == Regs::ShadowRamControl::TrackWithFilter;
    const bool replay = control == Regs::ShadowRamControl::Replay;
    const bool skip_same = SkipSameValueWrites();
    for (u32 i = 0; i < count; ++i) {
        const u32 method = static_cast<u32>(pairs[i] & 0xffff);
        u32 value = static_cast<u32>(pairs[i] >> 32);
        if (track) {
            shadow_state.reg_array[method] = value;
        } else if (replay) {
            value = shadow_state.reg_array[method];
        }
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    }
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    // The instances after the first are drawn together once their loop ends, and they must see
    // the state they were begun with: NVN's draw macros reset the base instance (and its copy in
    // c[0][0x644]) right after the loop, before the macro returns.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    regs.reg_array[method] = argument;
    for (auto const& table : dirty.tables)
        dirty.flags[table[method]] = true;
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        return ProfiledCallMethod(system, method, method_argument, is_last_call);
    }
    DispatchMethod(system, method, method_argument, is_last_call);
}

void Maxwell3D::DispatchMethod(Core::System& system, u32 method, u32 method_argument,
                               bool is_last_call) {
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (perf_verify) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

// EDEN_NX2_GPU_PROFILE: the path unprofiled runs take, counted. GpuSplit's back calls the 3D engine
// outside every zone (fast_maxwell3d), millions of times a second: a sample of those calls is
// timed (SampledScope). Called inside another zone (a back macro, a run of words, the unsplit GPU
// thread's method zone), the call is that zone's time, as before.
void Maxwell3D::ProfiledCallMethod(Core::System& system, u32 method, u32 method_argument,
                                   bool is_last_call) {
    const VideoCommon::Nx2Profile::SampledScope scope{VideoCommon::Nx2Profile::Zone::Method};
    if (method < MacroRegistersStart) {
        VideoCommon::Nx2Profile::CountRegisterWrite(method,
                                                    regs.reg_array[method] == method_argument);
    }
    DispatchMethod(system, method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    const u32 argument = ProcessShadowRam(method, method_argument);
    if (!execution_mask[method]) {
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (perf_verify) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // Counted and timed as CallMethod counts and times each word (EDEN_NX2_GPU_PROFILE).
    const VideoCommon::Nx2Profile::SampledScope scope{VideoCommon::Nx2Profile::Zone::Method, count};
    const bool profile = VideoCommon::Nx2Profile::Enabled();
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        if (profile) [[unlikely]] {
            VideoCommon::Nx2Profile::CountRegisterWrite(methods[i],
                                                        regs.reg_array[methods[i]] == arguments[i]);
        }
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    const VideoCommon::Nx2Profile::FineScope upload_scope{VideoCommon::Nx2Profile::Zone::CbUpload};
    memory_manager.WriteWordsCached(address + offset, values.data(), count);
    regs.const_buffer.offset += static_cast<u32>(bytes);
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (executing_macro != 0 && method != executing_macro + 1) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (g_trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    // The profile's write counters: ProfiledCallMethod, for this path and CallMethodFast.
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (Common::Nx2GpuWrite::g_listener.load(std::memory_order_relaxed) != nullptr) {
        Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::Stamp3D,
                                    memory_manager.GpuToCpuAddress(sequence_address).value_or(0),
                                    sequence_address, memory_manager.GetPointer(sequence_address),
                                    static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::QueryReport};
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::RenderEnable};
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    if (VideoCore::Nx2Debug::Experiment(53) != 0 &&
        regs.render_enable.mode >= Regs::RenderEnable::Mode::Conditional) {
        // EXP53: the front left the Compare words to this thread. Report words wait in the query
        // cache until a fence, and the caches belong to the thread running this (the split's
        // back): write them out before they are compared.
        memory_manager.FlushRegion(condition_address, sizeof(Regs::ReportSemaphore::Compare),
                                   VideoCommon::CacheType::QueryCache);
    }
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::CounterReset};
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::CbBind};
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::CbUpload};
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        memory_manager.WriteBlockCached(address, start_base, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself.
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    /// EXP53: per macro, 0 unknown, 1 a render-enable predicate, 2 something else.
    std::vector<u8> predicate_macros;
    u64 predicate_generation{~u64{0}};
    std::vector<u32> dma_words; ///< FrontDmaReadEarly's words
    u64 calls{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// The inline upload being sent, when its words land linearly (EDEN_NX2_DMA_WAIT_WRITERS).
    bool upload_known{};
    /// The back's draw manager is in an instance loop (DrawMode::Instance): draw.end does nothing.
    bool instance_loop{};
    /// EDEN_NX2_EXP2, read in Front(): same-value index_buffer.first writes are not sent.
    int same_experiment{};
    int early_same{}; ///< EXP14, read once per front call like same_experiment
    /// EDEN_NX2_EXP22 (zortos293/Lichii#641's front state, draw batches and same-value draw
    /// registers), read once per front call like early_same.
    int exp22{};
    /// StateDelta blocks would skip the back's write history and first-write trace
    /// (EDEN_NX2_3D_HISTORY, EDEN_NX2_3D_METHODS): off with them.
    bool delta_allowed{};
    /// EXP22 bit 2: the last complete instance-loop group (draw.begin Subsequent and the loop
    /// writes after it) held back, how many times in a row it came, and the next group while it
    /// is captured. Held only until the next ring entry, publish or drain (FrontFlushPending).
    struct BatchWrite {
        u32 method;
        u32 argument;
        bool is_last;
        bool from_macro;
    };
    static constexpr u32 MaxBatchWrites = 8;
    std::array<BatchWrite, MaxBatchWrites> batch{};
    u32 batch_size{};
    u32 batch_repeats{};
    std::array<BatchWrite, MaxBatchWrites> capture{};
    u32 capture_size{};
    /// An inline or small index draw (which binds an index buffer not taken from the registers)
    /// was sent since the last index_buffer.count write. Starts set: nothing is known.
    bool other_index_draw{true};
    /// EDEN_NX2_EXP2 counters, logged every ~5 s: writes the experiment dropped, writes the
    /// existing same-value filters dropped, and writes sent.
    u64 exp_skipped{};
    u64 exp_same{};
    u64 exp_sent{};
    u64 exp_ticks{};
    std::chrono::steady_clock::time_point exp_report{};
    u64 upload_device{};
    u32 upload_size{};
    u32 upload_offset{};
    /// EXP13=2-4: the upload's write-stamp tag (Nx2PatchQmdCopy looks past its own stamps).
    u64 upload_tag{};
    /// Macro DMA read snapshots (EDEN_NX2_DMA_SNAPSHOT, from EvilOOT's dma-snapshot fix): device
    /// pages a drained read flushed whole, and the split position right after. Cleared past 8192
    /// pages.
    struct DmaSnapshot {
        u64 position;
        /// EXP10: the noted compute and copy ranges and recorded dispatch writes already found
        /// to miss the page (GpuSplit::WritersMayWriteSince, DispatchesMayWriteSince).
        u64 writer_ranges;
        u64 dispatch_writes;
    };
    std::unordered_map<u64, DmaSnapshot> dma_snapshots;
};

namespace {

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
bool FrontVerifyEnabled() {
    static const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    return verify;
}

// Methods whose execution on the back writes guest memory (inline uploads, reports).
bool WritesGuestMemory(u32 method) {
    return method == MAXWELL3D_REG_INDEX(launch_dma) ||
           method == MAXWELL3D_REG_INDEX(inline_data) ||
           method == MAXWELL3D_REG_INDEX(report_semaphore.query);
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

// What an executable method may write on the back, besides a deferred draw it may issue
// (GpuSplit::Effect). Constant buffer data and reports write ranges the front notes.
VideoCommon::GpuSplit::Effect FrontMethodEffect(u32 method) {
    using Effect = VideoCommon::GpuSplit::Effect;
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return Effect::None;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return Effect::None;
    default:
        return FrontHandledOutsideDrawManager(method) ? Effect::Unknown : Effect::Gpu;
    }
}

// Where the next constant buffer data word goes (ProcessCBMultiData), from the front's registers.
GPUVAddr ConstBufferWriteAddress(std::span<const u32> regs) {
    return ((GPUVAddr{regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)]) +
           regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
}

// A report and what it may write: payload and timestamp.
GPUVAddr ReportAddress(std::span<const u32> regs) {
    return (GPUVAddr{regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
           regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
}
constexpr u64 ReportBytes = 16;

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        front->delta_allowed = !record_write_history && !g_trace_first_method_writes;
        front->executing_macro = executing_macro;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
    }
    front->same_experiment = VideoCore::Nx2Debug::Experiment(2);
    front->early_same = VideoCore::Nx2Debug::Experiment(14);
    front->exp22 = VideoCore::Nx2Debug::Experiment(22);
    return *front;
}

namespace {

// Indexed draws that bind an index buffer other than the registers' (inline indices, small index
// draws): the back sets and consumes Dirty::IndexBuffer for them itself.
bool IsOtherIndexDraw(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

void Maxwell3D::FrontTrackIndexDraws(FrontState& fs, u32 method) {
    if (method == MAXWELL3D_REG_INDEX(index_buffer.count)) {
        fs.other_index_draw = false;
    } else if (IsOtherIndexDraw(method)) {
        fs.other_index_draw = true;
    }
}

bool Maxwell3D::FrontSkipsIndexFirst(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    // EDEN_NX2_EXP2: index_buffer.first written with its value. On the back it only stores the
    // register and sets Dirty::IndexBuffer (an instance loop method: no deferred draw flush, and
    // the draw manager reads the register when it draws). The flag matters only when the index
    // buffer binding is stale for the next register-indexed draw; that draw needs an
    // index_buffer.count write (draw_indexed), which always sets the flag, unless the count was
    // written before an inline or small index draw that rebound and consumed the flag (then the
    // write is sent).
    if (method != MAXWELL3D_REG_INDEX(index_buffer.first) || !fs.skip_same || !shadow_kept ||
        fs.regs[method] != value || fs.other_index_draw) {
        return false;
    }
    ++fs.exp_skipped;
    return true;
}

void Maxwell3D::FrontReportSameWrites(FrontState& fs) {
    if ((++fs.exp_ticks & 0xffff) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (fs.exp_report == std::chrono::steady_clock::time_point{}) {
        fs.exp_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - fs.exp_report).count();
    if (seconds < 5.0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "EDEN_NX2_EXP2: 3D writes per second: {:.0f} sent, {:.0f} same-value index_buffer.first "
             "not sent, {:.0f} other same-value writes not sent",
             fs.exp_sent / seconds, fs.exp_skipped / seconds, fs.exp_same / seconds);
    fs.exp_skipped = fs.exp_same = fs.exp_sent = 0;
    fs.exp_report = now;
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    // As ProcessShadowRam does on the back.
    u32 value = argument;
    bool shadow_kept = true;
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    // EXP14: an unchanged state register (6.6M of Ocarina's 11.4M macro writes a second) is
    // dropped before the checks below, as the same-value skip further down would.
    if (!execution_mask[method] && fs.regs[method] == value && shadow_kept && fs.skip_same &&
        fs.same_experiment == 0 && fs.early_same != 0) [[likely]] {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::MethodWrites, 1);
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    // EXP22 bit 1 (zortos293/Lichii#641 EDEN_NX2_FRONT_STATE): a write to a state register that
    // writes no memory, which the rest of this function would only compare, store and send as a
    // Direct MacroWrite (a macro's, EDEN_NX2_MACRO_SINK off) or a PairWrite (EDEN_NX2_RING_PAIRS:
    // the command list's, and the macros' with MACRO_SINK), goes in the open StateDelta block at
    // that same place, with the effect that path notes. Sunk Method entries (MACRO_SINK without
    // RING_PAIRS) apply later than a block would and stay as they are.
    if ((fs.exp22 & 1) != 0 && !execution_mask[method] && !memory_write_mask[method] &&
        fs.delta_allowed && fs.same_experiment == 0) {
        const bool direct = from_macro && !VideoCommon::GpuSplit::MacroSink();
        if (direct || VideoCommon::GpuSplit::RingPairs()) {
            if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
                if (Profile::Enabled()) [[unlikely]] {
                    Profile::Add(Profile::Counter::SameValueWrites, 1);
                }
                return;
            }
            fs.regs[method] = value;
            using Effect = VideoCommon::GpuSplit::Effect;
            split.DeltaWrite(this, method, argument,
                             direct && fs.draw_pending ? Effect::Gpu : Effect::None);
            if (from_macro && !direct) {
                split.CountBatched(VideoCommon::GpuSplit::Batched::MacroSunk);
            }
            return;
        }
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    // From zortos293/Lichii#178 (tibzejoker): instance loop writes the back would ignore.
    if (execution_mask[method] && FrontSkipsLoopWrite(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        if (fs.same_experiment != 0) [[unlikely]] {
            ++fs.exp_same;
            FrontReportSameWrites(fs);
        }
        return;
    }
    // EXP2, and EXP22 bit 4 (#641 EDEN_NX2_SKIP_SAME_DRAWREG: its same-value vertex_buffer
    // first and count are the loop writes skipped above).
    if ((fs.same_experiment != 0 || (fs.exp22 & 4) != 0) && execution_mask[method] &&
        FrontSkipsIndexFirst(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        if (fs.same_experiment != 0) [[unlikely]] {
            FrontReportSameWrites(fs);
        }
        split.CountBatched(VideoCommon::GpuSplit::Batched::SameIndexFirst);
        return;
    }
    // What a constant buffer upload word or a report writes, for the front's macro memory reads.
    const auto [write_begin, write_end] = FrontWriteRange(fs, method, 1);
    using Effect = VideoCommon::GpuSplit::Effect;
    // What the entry may write on the back (FrontDmaRead): any method may issue a deferred
    // instanced draw first.
    Effect effect = fs.draw_pending ? Effect::Gpu : Effect::None;
    std::optional<GPUVAddr> cb_write;
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            if (fs.same_experiment != 0) [[unlikely]] {
                ++fs.exp_same;
                FrontReportSameWrites(fs);
            }
            return;
        }
        fs.regs[method] = value;
    } else {
        fs.regs[method] = value;
        FrontTrackIndexDraws(fs, method);
        effect = VideoCommon::SplitWriteTracker::Combine(effect, FrontMethodEffect(method));
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                cb_write = ConstBufferWriteAddress(fs.regs);
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    if (fs.same_experiment != 0) [[unlikely]] {
        ++fs.exp_sent;
        FrontReportSameWrites(fs);
    }
    // EXP22 bit 2: instance loop methods, which nothing after the send below concerns.
    if ((fs.exp22 & 2) != 0 && execution_mask[method] &&
        FrontBatchWrite(split, fs, method, argument, value, is_last_call, from_macro)) {
        return;
    }
    if (from_macro && (execution_mask[method] || !VideoCommon::GpuSplit::MacroSink())) {
        split.MacroWrite(this, method, argument, effect);
    } else {
        // EDEN_NX2_MACRO_SINK: a macro's state write goes to the back's method sink as the
        // command list's own do; the back applies the sink in order (shadow RAM and same-value
        // skip as CallMethod) before the next executable or Direct write and in
        // FinishFrontMacro, so the order against executable writes and deferred draws is kept.
        // EDEN_NX2_RING_PAIRS: the same writes as pairs, applied the same way (ApplyPairs) when
        // the back reaches them, which is never later than the sink would.
        if (!execution_mask[method] && VideoCommon::GpuSplit::RingPairs()) {
            split.PairWrite(this, method, argument);
        } else {
            split.Method(this, method, argument, is_last_call, nullptr, effect);
        }
        if (from_macro) {
            split.CountBatched(VideoCommon::GpuSplit::Batched::MacroSunk);
        }
    }
    if (cb_write) {
        split.NoteWrite(memory_manager, *cb_write, sizeof(u32));
    } else if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        split.NoteWrite(memory_manager, ReportAddress(fs.regs), ReportBytes);
    }
    if (write_end != 0) {
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        const bool upload = method >= cb_data && method < cb_data + 16;
        split.RangeLastWrite(write_begin, write_end, upload ? &value : nullptr,
                             memory_manager.GetID() + 1);
        if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
            FrontNoteRelease(split, fs, value);
        }
    }
    if (WritesGuestMemory(method)) {
        FrontNoteMemoryWrite(split, fs, method, &value, 1);
    }
}

void Maxwell3D::FrontNoteRelease(VideoCommon::GpuSplit& split, const FrontState& fs,
                                 u32 query) const {
    // A semaphore release of the payload register (no reduction) writes a value the front knows.
    Regs::ReportSemaphore report{};
    report.query.raw = query;
    if (report.query.operation != Regs::ReportSemaphore::Operation::Release ||
        report.query.report != Regs::ReportSemaphore::Report::Payload ||
        report.query.reduction_enable != 0) {
        return;
    }
    const GPUVAddr address =
        (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
        fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
    if (const auto device = memory_manager.GpuToCpuAddress(address)) {
        const u32 payload = fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.payload)];
        split.NoteRelease(*device, address, payload);
        if (VideoCommon::QmdPrediction::MirrorMode()) {
            // EXP13=2: recorded so the reduction bound sees it queued or run but not landed.
            namespace Prediction = VideoCommon::QmdPrediction;
            Prediction::Note(*device,
                             {.writer = Prediction::Writer::Report,
                              .kind = Prediction::Kind::Plain,
                              .payload = payload,
                              .op = 0,
                              .is_signed = false},
                             memory_manager.Read<u32>(address), split.SentPosition(), 0);
        }
    }
}

std::pair<GPUVAddr, GPUVAddr> Maxwell3D::FrontWriteRange(const FrontState& fs, u32 method,
                                                         u32 amount) const {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        // Uploaded words go to the buffer's address plus the running offset.
        const GPUVAddr address =
            (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)];
        const GPUVAddr begin = address + fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
        return {begin, begin + GPUVAddr{amount} * sizeof(u32)};
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount == 1) {
        // A report writes its payload or a 16-byte counter and timestamp at its address.
        const GPUVAddr begin =
            (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
            fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
        return {begin, begin + 16};
    }
    return {0, 0};
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // The instance loop state the back ends in (Replay: it acts on the shadow copy). The back
    // runs draw.begin once per word, so each word can switch the loop; other methods leave it
    // as their last word does.
    const bool replay = fs.control == Regs::ShadowRamControl::Replay;
    if (method == MAXWELL3D_REG_INDEX(draw.begin)) {
        for (u32 i = 0; i < amount; ++i) {
            FrontTrackInstanceLoop(fs, method, replay ? fs.shadow[method] : base_start[i]);
        }
    } else if (amount != 0) {
        FrontTrackInstanceLoop(fs, method, replay ? fs.shadow[method] : base_start[amount - 1]);
    }
    if (amount != 0) {
        FrontTrackIndexDraws(fs, method);
    }
    if (fs.same_experiment != 0) [[unlikely]] {
        fs.exp_sent += amount;
        FrontReportSameWrites(fs);
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words.
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto [write_begin, write_end] = FrontWriteRange(fs, method, amount);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    using Effect = VideoCommon::GpuSplit::Effect;
    Effect effect = fs.draw_pending ? Effect::Gpu : Effect::None;
    std::optional<GPUVAddr> cb_write;
    if (method >= cb_data && method < cb_data + 16) {
        cb_write = ConstBufferWriteAddress(fs.regs);
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
        effect = Effect::Unknown;
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
        effect = VideoCommon::SplitWriteTracker::Combine(effect, Effect::Gpu);
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            if (!execution_mask[method]) {
                continue;
            }
            effect = VideoCommon::SplitWriteTracker::Combine(effect, FrontMethodEffect(method));
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr, effect);
    if (write_end != 0) {
        // Uploaded words, unless the shadow RAM replays other values for them on the back.
        const bool upload = method >= cb_data && method < cb_data + 16 &&
                            fs.control != Regs::ShadowRamControl::Replay;
        split.RangeLastWrite(write_begin, write_end, upload ? base_start : nullptr,
                             memory_manager.GetID() + 1);
        if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount == 1) {
            FrontNoteRelease(split, fs, base_start[0]);
        }
    }
    if (cb_write) {
        split.NoteWrite(memory_manager, *cb_write, u64{amount} * sizeof(u32));
    } else if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount != 0) {
        split.NoteWrite(memory_manager, ReportAddress(fs.regs), ReportBytes);
    }
    if (WritesGuestMemory(method) && amount != 0) {
        FrontNoteMemoryWrite(split, fs, method, base_start, amount);
    }
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    // EDEN_NX2_MACRO_TRACE=<method>: the first calls of that macro, parameters and writes.
    static const u32 traced = [] {
        const char* value = std::getenv("EDEN_NX2_MACRO_TRACE");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 16)) : 0U;
    }();
    static u32 traced_calls{};
    const bool trace = traced != 0 && method == traced &&
                       std::ranges::any_of(fs.parameters, [](u32 p) { return p != 0; }) &&
                       traced_calls++ < 8;
    std::string trace_text;
    if (trace) [[unlikely]] {
        trace_text = fmt::format("3D macro {:#x} start {:#x} params", traced, start);
        for (const u32 p : fs.parameters) {
            trace_text += fmt::format(" {:x}", p);
        }
        trace_text += " writes";
    }
    // EDEN_NX2_EXP53 (live, 0 = off; port of Eglantine d3612f82b1 by kraibse): NVN's conditional
    // render macro reads the six Compare words at an address and sets render_enable.mode from
    // them, which drained the split on every call (our largest front wait, "macro dma read").
    // For such a macro the front skips the read, feeds six zero words and writes the address
    // with mode IfNotEqual: the back evaluates the same condition in order.
    const bool predicate = VideoCore::Nx2Debug::Experiment(53) != 0 &&
                           fs.parameters.size() == 3 && fs.parameters[0] == 0 &&
                           IsPredicateMacro(fs, method);
    {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        engine.native_interpreter.active_dram = &native_data_ram;
        const auto parameter_count = fs.parameters.size();
        const auto read = [&fs](u32 address) {
            return address < Regs::NUM_REGS ? fs.regs[address] : 0U;
        };
        const auto emit = [&](u32 address, u32 value) {
            if (trace) [[unlikely]] {
                trace_text += fmt::format(" {:x}={:x}", address, value);
            }
            if (address >= MacroRegistersStart) {
                return false;
            }
            if (predicate && address == MAXWELL3D_REG_INDEX(render_enable.mode)) {
                FrontWrite(split, fs, MAXWELL3D_REG_INDEX(render_enable.address_high),
                           fs.regs[0x154] & 0x1ffffff, true, true);
                FrontWrite(split, fs, MAXWELL3D_REG_INDEX(render_enable.address_low),
                           fs.regs[0x155], true, true);
                FrontWrite(split, fs, address,
                           static_cast<u32>(Regs::RenderEnable::Mode::IfNotEqual), true, true);
                return true;
            }
            FrontWrite(split, fs, address, value, true, true);
            // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
            if (address == 0x157 || address == 0x158) {
                if (predicate && address == 0x158) {
                    // The back reads the words when it evaluates the condition: the macro only
                    // has to consume six.
                    fs.parameters.insert(fs.parameters.end(), value, 0U);
                    static std::atomic<u64> skipped{};
                    if (const u64 n = skipped.fetch_add(1, std::memory_order_relaxed);
                        n == 0 || (n + 1) % 100000 == 0) {
                        LOG_INFO(HW_GPU, "Conditional render macro (EXP53): {} reads left to the back",
                                 n + 1);
                    }
                } else {
                    FrontDmaRead(split, fs, method, address, value);
                }
            }
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        if (!engine.native_jit) {
            engine.native_jit = std::make_shared<TuringMacroJit>();
        }
        engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
        const auto result = engine.native_jit->Execute(
            start, fs.parameters, std::span<u32>{native_data_ram}, read, emit);
#else
        const auto result =
            engine.native_interpreter.Execute(engine.native_code, start, fs.parameters, read, emit);
#endif
        if (!result.success) {
            engine.native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                      start, result.consumed, parameter_count, result.error);
        }
    }
    if (trace) [[unlikely]] {
        LOG_INFO(HW_GPU, "{}", trace_text);
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                    VideoCommon::GpuSplit::Effect::Gpu);
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

bool Maxwell3D::IsPredicateMacro(FrontState& fs, u32 method) {
#ifdef ARCHITECTURE_x86_64
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    if (positions.empty() || engine.native_execution_failed) {
        return false;
    }
    const size_t index = ((method - MacroRegistersStart) >> 1) % positions.size();
    if (fs.predicate_generation != engine.native_code_generation ||
        fs.predicate_macros.size() != positions.size()) {
        fs.predicate_generation = engine.native_code_generation;
        fs.predicate_macros.assign(positions.size(), 0);
    }
    if (fs.predicate_macros[index] != 0) {
        return fs.predicate_macros[index] == 1;
    }
    // Dry run against a copy of the data RAM, feeding test words when the macro asks for six. It
    // must write exactly the DMA address, the read of six words and render_enable.mode, and the
    // mode must be what IfNotEqual makes of the Compare words for every test vector.
    constexpr u32 address_high = 0x21, address_low = 0x13d60da0;
    constexpr std::array<std::array<u32, 6>, 8> vectors{{
        {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {5, 0, 0, 0, 0, 0}, {2, 3, 7, 9, 2, 3},
    }};
    if (!engine.native_jit) {
        engine.native_jit = std::make_shared<TuringMacroJit>();
    }
    engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
    bool matches = true;
    std::array<u32, vectors.size()> modes{};
    for (size_t i = 0; i < vectors.size() && matches; ++i) {
        std::vector<u32> parameters{0, address_high, address_low};
        std::vector<u32> scratch_ram(native_data_ram.begin(), native_data_ram.end());
        std::vector<std::pair<u32, u32>> writes;
        const auto result = engine.native_jit->Execute(
            positions[index], parameters, std::span<u32>{scratch_ram},
            [&fs](u32 address) { return address < Regs::NUM_REGS ? fs.regs[address] : 0U; },
            [&](u32 address, u32 value) {
                if (address >= MacroRegistersStart || writes.size() >= 16) {
                    return false;
                }
                writes.emplace_back(address, value);
                if (address == 0x158 && value == 6) {
                    parameters.insert(parameters.end(), vectors[i].begin(), vectors[i].end());
                }
                return true;
            });
        matches = result.success && writes.size() == 4 && writes[0].first == 0x154 &&
                  writes[0].second == address_high && writes[1].first == 0x155 &&
                  writes[1].second == address_low && writes[2] == std::pair<u32, u32>{0x158, 6} &&
                  writes[3].first == MAXWELL3D_REG_INDEX(render_enable.mode);
        if (matches) {
            const auto& w = vectors[i];
            modes[i] = writes[3].second;
            matches = modes[i] == ((w[0] != w[4] || w[1] != w[5]) ? 1U : 0U);
        }
    }
    fs.predicate_macros[index] = matches ? 1 : 2;
    LOG_INFO(HW_GPU, "Native macro {:#x}: {} (EXP53; modes for the test vectors: {})", method,
             matches ? "render-enable predicate, the back evaluates IfNotEqual"
                     : "not a render-enable predicate (or not IfNotEqual), runs as before",
             fmt::join(modes, " "));
    return matches;
#else
    static_cast<void>(fs);
    static_cast<void>(method);
    return false;
#endif
}

bool Maxwell3D::FrontMacroCountsDraw(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, u32 count) {
#ifdef ARCHITECTURE_x86_64
    FrontState& fs = Front(system, split);
    if (count != 2 || fs.executing_macro != 0 || !fs.parameters.empty()) {
        return false;
    }
    auto& engine = GetNativeMacroEngine();
    if (engine.native_execution_failed) {
        return false;
    }
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    // Cached per macro start and code generation: the dry runs only repeat when the code changes.
    struct Verdict {
        u32 start;
        u64 generation;
        bool counts;
    };
    static std::array<Verdict, 8> verdicts{};
    for (const Verdict& verdict : verdicts) {
        if (verdict.start == start && verdict.generation == engine.native_code_generation + 1) {
            return verdict.counts;
        }
    }
    if (!engine.native_jit) {
        engine.native_jit = std::make_shared<TuringMacroJit>();
    }
    engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
    const auto run = [&](u32 first, u32 second, std::vector<std::pair<u32, u32>>& writes,
                         std::array<u32, 0xc00>& dram) {
        dram = native_data_ram;
        std::vector<u32> parameters{first, second};
        bool reads_memory = false;
        const auto read = [&fs](u32 address) {
            return address < Regs::NUM_REGS ? fs.regs[address] : 0U;
        };
        const auto emit = [&](u32 address, u32 value) {
            if (address >= MacroRegistersStart) {
                return false;
            }
            reads_memory |= address == 0x157 || address == 0x158;
            writes.emplace_back(address, value);
            return true;
        };
        const auto result =
            engine.native_jit->Execute(start, parameters, std::span<u32>{dram}, read, emit);
        return result.success && !reads_memory && parameters.size() == 2;
    };
    constexpr u32 vertex_count = MAXWELL3D_REG_INDEX(vertex_buffer.count);
    constexpr u32 first_vertex = MAXWELL3D_REG_INDEX(vertex_buffer.first);
    std::vector<std::pair<u32, u32>> a, b;
    std::array<u32, 0xc00> dram_a{}, dram_b{};
    bool counts = run(0x13572468u, 0x0BADF00Du, a, dram_a) && run(0x2468ACE0u, 0x1337C0DEu, b, dram_b) &&
                  a.size() == b.size() && dram_a == dram_b;
    bool copied = false;
    for (size_t i = 0; counts && i < a.size(); ++i) {
        if (a[i].first != b[i].first) {
            counts = false;
        } else if (a[i].second != b[i].second) {
            // The first parameter, unchanged, into the vertex count; the second into the first
            // vertex (0 in the GPU's words: the back draws from vertex 0 and checks in debug).
            const bool count_write = a[i].first == vertex_count && a[i].second == 0x13572468u &&
                                     b[i].second == 0x2468ACE0u;
            const bool first_write = a[i].first == first_vertex && a[i].second == 0x0BADF00Du &&
                                     b[i].second == 0x1337C0DEu;
            counts = count_write || first_write;
            copied |= count_write;
        }
    }
    counts = counts && copied;
    static u32 next_verdict{};
    verdicts[next_verdict++ % verdicts.size()] = {start, engine.native_code_generation + 1, counts};
    LOG_INFO(HW_GPU, "GPU-counted draws: 3D macro {:#x} (start {:#x}) {}", method, start,
             counts ? "copies its count into the draw" : "uses its parameters otherwise: read as before");
    return counts;
#else
    return false;
#endif
}

// Not inlined on GCC (zortos293/Lichii#491): GCC 13 -O3 unswitched a per-word loop on this
// switch there and dropped the macro upload cases from it (an empty macro RAM on the front).
#if defined(__GNUC__) && !defined(__clang__)
[[gnu::noinline]]
#endif
void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}

bool Maxwell3D::FrontSkipsLoopWrite(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    FrontTrackInstanceLoop(fs, method, value);
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        break;
    default:
        return false;
    }
    if (!fs.skip_same || !shadow_kept || fs.regs[method] != value) {
        return false;
    }
    // Instance loop methods flush no deferred draw, and the draw manager ignores these: DrawEnd
    // draws nothing in an instance loop, and it only reads the vertex buffer first and count
    // (index_buffer writes also mark the index buffer dirty, so they are always sent).
    return method != MAXWELL3D_REG_INDEX(draw.end) || fs.instance_loop;
}

void Maxwell3D::FrontNoteMemoryWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                     const u32* words, u32 count) {
    split.NoteMemoryWriter();
    const auto unknown = [&split](const char* what) { split.NoteMemoryWrite(0, ~u64{0}, what); };
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        constexpr u32 base = MAXWELL3D_REG_INDEX(report_semaphore);
        const GPUVAddr address = (GPUVAddr{fs.regs[base]} << 32) | fs.regs[base + 1];
        Regs::ReportSemaphore query_regs{};
        query_regs.query.raw = words[count - 1];
        const auto device = memory_manager.GpuToCpuAddress(address);
        if (!device) {
            return; // CounterReport writes nothing at an unmapped address.
        }
        // A short payload release is a fence: written when the back runs it (fast fences).
        if (query_regs.query.operation == Regs::ReportSemaphore::Operation::Release &&
            query_regs.query.short_query != 0 &&
            query_regs.query.report == Regs::ReportSemaphore::Report::Payload) {
            const u32 payload = fs.regs[base + 2];
            split.NoteMemoryWriteData(*device, &payload, 1);
        } else {
            // Counters and timed releases are written when the host query completes, not when
            // the back runs them: reads of these bytes wait until the back ran the report and
            // the host query completed.
            split.NoteAsyncWrite(*device, 16);
            split.NoteMemoryWrite(*device, 16, "3D report");
        }
        return;
    }
    constexpr u32 base = MAXWELL3D_REG_INDEX(upload);
    if (method == MAXWELL3D_REG_INDEX(launch_dma)) {
        const u64 line_length = fs.regs[base];
        const u64 lines = (std::max)(fs.regs[base + 1], 1U);
        const u64 pitch = (std::max<u64>)(fs.regs[base + 4], line_length);
        const GPUVAddr address = (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3];
        const u64 size = pitch * (lines - 1) + line_length;
        const auto device = memory_manager.GpuToCpuAddress(address);
        // LAUNCH_DMA: bit 0 pitch (linear) layout, bits 5:4 completion (2 releases a semaphore).
        const u32 launch = words[count - 1];
        const bool linear = (launch & 1) != 0;
        if (((launch >> 4) & 3) == 2) {
            unknown("upload release");
        }
        fs.upload_known = device && linear && lines == 1 && (*device & 3) == 0 &&
                          memory_manager.GpuToCpuAddress(address + size - 1) ==
                              *device + size - 1;
        fs.upload_device = device.value_or(0);
        fs.upload_size = static_cast<u32>((std::min<u64>)(size, ~0U));
        fs.upload_offset = 0;
        fs.upload_tag = split.NewWriteStampTag();
        if (!fs.upload_known) {
            if (!device || size == 0 || size > (u64{64} << 20)) {
                return unknown("upload unmapped or huge");
            }
            split.NoteMemoryWrite(*device, size, "upload (not linear)");
        }
        // Its bytes are noted (above, or by its inline_data words); an upload release is an
        // unknown write of its own.
        split.RangedMethodWrite();
        return;
    }
    // inline_data words of the current upload.
    if (!fs.upload_known) {
        const auto device = memory_manager.GpuToCpuAddress(
            (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3]);
        if (!device) {
            return unknown("upload data unmapped");
        }
        split.NoteMemoryWrite(*device, fs.regs[base] * u64{(std::max)(fs.regs[base + 1], 1U)} +
                                           u64{fs.regs[base + 4]} * fs.regs[base + 1],
                              "upload data");
        split.RangedMethodWrite();
        return;
    }
    const u32 remaining_bytes =
        fs.upload_offset < fs.upload_size ? fs.upload_size - fs.upload_offset : 0;
    const u32 whole = (std::min<u32>)(count, remaining_bytes / 4);
    // Tagged, so the QMD copy's patch below looks past this upload's own stamps.
    split.SetWriteStampTag(fs.upload_tag);
    split.NoteMemoryWriteData(fs.upload_device + fs.upload_offset, words, whole);
    if (whole < count && remaining_bytes % 4 != 0) {
        split.NoteMemoryWrite(fs.upload_device + fs.upload_offset + whole * 4, remaining_bytes % 4);
    }
    split.SetWriteStampTag(0);
    if (VideoCommon::QmdPrediction::MirrorMode()) {
        // EXP13=2-4: a QMD patched by a 3D inline upload (6h): the copy takes the bytes too.
        const GPUVAddr destination = (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3];
        Nx2PatchQmdCopy(split, memory_manager, destination + fs.upload_offset,
                        reinterpret_cast<const u8*>(words),
                        (std::min<u64>)(u64{count} * sizeof(u32), remaining_bytes),
                        split.SentPosition(), fs.upload_tag);
    }
    split.RangedMethodWrite();
    fs.upload_offset += (std::min<u32>)(count * 4, remaining_bytes);
}

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    // The words may be written by methods still queued for the back (indirect draw arguments
    // from a culling dispatch, constant buffer and inline uploads, reports), and the caches that
    // may hold newer ones belong to it. Draws write memory the 3D engine reads only after a
    // host-level wait, so the work after the last one need not have run. EDEN_NX2_MACRO_DMA_DRAIN=1
    // waits for everything queued, as before. With EDEN_NX2_CMDPAGE_DIAG the read is repeated
    // after a full drain and the words that differ are counted.
    const GPUVAddr source = (GPUVAddr{fs.regs[0x154] & 0x1ffffff} << 32) | fs.regs[0x155];
    const u32 ram = fs.regs[0x156];
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    static const bool full_drain = std::getenv("EDEN_NX2_MACRO_DMA_DRAIN") != nullptr;
    const auto flush_if_dirty = [&] {
        if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, read_bytes] {
                    memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
    };
    // Words a constant buffer upload still queued for the back writes come from the upload
    // itself (9,600 waits a second in Ocarina's forest).
    if (!full_drain && !Tegra::Nx2CmdPageDiag::g_enabled &&
        !VideoCommon::QueryDma::TraceEnabled()) {
        thread_local std::vector<u32> forwarded;
        forwarded.resize(words);
        if (split.ForwardRead(memory_manager.GetID() + 1, source, words, forwarded.data())) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadsForwarded, 1);
            for (u32 i = 0; i < words; ++i) {
                if (method == 0x158) {
                    fs.parameters.push_back(forwarded[i]);
                } else if (ram + i < native_data_ram.size()) {
                    native_data_ram[ram + i] = forwarded[i];
                }
            }
            return;
        }
    }
    // EDEN_NX2_DMA_SNAPSHOT (EvilOOT's dma-snapshot fix): Ocarina reads an occlusion report for
    // each object (macro 0xe2c, MME_DMA_READ_FIFOED), ~35,000 times a second, and its pages are
    // GPU-written, so each read drained the back: the front and the back took turns. A drained
    // read now flushes its whole device pages and remembers the split position; later reads of
    // those pages skip the drain while nothing sent since may write them
    // (GpuSplit::MayBeWrittenSince) and the query cache lists no report there. 10 also drains,
    // reads again and logs differences.
    const int snapshot_mode =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_dma_snapshot, "EDEN_NX2_DMA_SNAPSHOT",
                                         9);
    // Waits until pending reports in the range have landed (#215), as the drained path below.
    const auto flush_queries = [&](GPUVAddr begin, size_t bytes) {
        split.Run(
            [this, begin, bytes] {
                VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, begin, bytes);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    };
    if (snapshot_mode >= 9 && !full_drain && !Tegra::Nx2CmdPageDiag::g_enabled &&
        method == 0x158 && words != 0) {
        const auto device = memory_manager.GpuToCpuAddress(source);
        if (device &&
            memory_manager.GpuToCpuAddress(source + read_bytes - 1) == *device + read_bytes - 1) {
            static u64 hits{};
            static u64 misses{};
            static u64 differences{};
            static u64 report_reads{};
            static std::array<u64, 4> miss_reasons{};
            // EXP10: compute and copies sent since count only where their writes may reach the
            // pages read: the ranges the fronts noted (GpuSplit::WritersMayWriteSince), the
            // writes the back recorded for the dispatches (DispatchesMayWriteSince, once it ran
            // them: 1 waits for that, 2 drains instead). Copies and other work whose ranges
            // nobody can tell still count for every read.
            const int exp10 = VideoCore::Nx2Debug::Experiment(10);
            const bool ranged_writers = exp10 != 0;
            using DispatchCheck = VideoCommon::GpuSplit::DispatchCheck;
            static u64 writer_unbounded{};
            static u64 writer_overlaps{};
            static std::array<u64, static_cast<size_t>(DispatchCheck::Count)> dispatch_checks{};
            static u64 dispatch_waits{};
            static u64 range_skips{};
            static u64 range_differences{};
            bool range_skip = false;
            const u64 first_page = *device >> 12;
            const u64 last_page = (*device + read_bytes - 1) >> 12;
            bool hit = true;
            for (u64 page = first_page; page <= last_page && hit; ++page) {
                const auto it = fs.dma_snapshots.find(page);
                if (it == fs.dma_snapshots.end()) {
                    hit = false;
                    ++miss_reasons[0];
                    break;
                }
                // Compute/copy sent since, unranged writer sent since, noted write.
                auto& snapshot = it->second;
                const u64 since = snapshot.position;
                const bool writer_since = split.LastWriterPosition() > since;
                u32 reason = 4;
                if (!ranged_writers) {
                    reason = writer_since                             ? 1
                             : split.LastUnrangedPosition() > since ? 2
                             : split.NotedWriteSince(since, *device, read_bytes) ? 3
                                                                                 : 4;
                } else if (split.WritersMayWriteSince(since, page << 12, 0x1000,
                                                      &snapshot.writer_ranges)) {
                    // Whole pages: the ranges found to miss one need no second look.
                    reason = 1;
                    ++(split.LastUnboundedWriterPosition() > since ? writer_unbounded
                                                                   : writer_overlaps);
                } else if (split.LastUnrangedPosition() > since) {
                    reason = 2;
                } else if (split.NotedWriteSince(since, *device, read_bytes)) {
                    reason = 3;
                } else {
                    // Last: the only check that may wait.
                    bool waited = false;
                    const DispatchCheck check = split.DispatchesMayWriteSince(
                        since, page << 12, 0x1000, (exp10 & 3) != 2, snapshot.dispatch_writes,
                        waited);
                    dispatch_waits += waited ? 1 : 0;
                    if (check != DispatchCheck::Miss) {
                        reason = 1;
                        ++dispatch_checks[static_cast<size_t>(check)];
                    } else {
                        range_skip |= writer_since;
                    }
                }
                if (reason != 4) {
                    hit = false;
                    ++miss_reasons[reason];
                }
            }
            if (hit && range_skip) {
                ++range_skips;
            }
            // A report's value is up to 8 bytes, listed by its first 4: look 12 bytes back. A
            // listed report may not be in guest memory yet: take the drained path.
            if (!hit || !rasterizer->HasCachedReports(*device - 12, u64{read_bytes} + 12)) {
                thread_local std::vector<u32> snap_words;
                snap_words.resize(words);
                if (hit) {
                    ++hits;
                    memory_manager.ReadBlockUnsafe(source, snap_words.data(), read_bytes);
                    if (snapshot_mode == 10) {
                        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
                        flush_if_dirty();
                        flush_queries(source, read_bytes);
                        std::vector<u32> exact(words);
                        memory_manager.ReadBlockUnsafe(source, exact.data(), read_bytes);
                        if (exact != snap_words) {
                            range_differences += range_skip ? 1 : 0;
                            if (differences++ < 32) {
                                LOG_WARNING(HW_GPU,
                                            "DMA snapshot check: macro {:#x} read of {} words at "
                                            "{:#x} differs from the drained read ({:#x} vs {:#x}){}",
                                            macro, words, source, snap_words[0], exact[0],
                                            range_skip ? " (skipped by EXP10 ranges)" : "");
                            }
                            snap_words = std::move(exact);
                        }
                    }
                } else {
                    ++misses;
                    // The GPU data of the whole pages (a flush of only the bytes read left stale
                    // neighbours), the pending reports of the bytes read.
                    const GPUVAddr flush_begin = source & ~GPUVAddr{0xfff};
                    const GPUVAddr flush_end = (source + read_bytes + 0xfff) & ~GPUVAddr{0xfff};
                    const size_t flush_bytes = static_cast<size_t>(flush_end - flush_begin);
                    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
                    if (memory_manager.IsMemoryDirty(flush_begin, flush_bytes,
                                                     VideoCommon::CacheType::BufferCache)) {
                        split.Run(
                            [memory = &memory_manager, flush_begin, flush_bytes] {
                                memory->FlushRegion(flush_begin, flush_bytes,
                                                    VideoCommon::CacheType::BufferCache);
                            },
                            VideoCommon::DrainSite::MacroDmaFlush);
                    }
                    flush_queries(source, read_bytes);
                    if (fs.dma_snapshots.size() > 8192) {
                        fs.dma_snapshots.clear();
                    }
                    for (u64 page = first_page; page <= last_page; ++page) {
                        fs.dma_snapshots[page] = {split.SentPosition(), split.WriterRangeCount(),
                                                  split.DispatchWriteCount()};
                    }
                    memory_manager.ReadBlockUnsafe(source, snap_words.data(), read_bytes);
                }
                static auto last_log = std::chrono::steady_clock::now();
                if (const auto now = std::chrono::steady_clock::now();
                    now - last_log >= std::chrono::seconds{5}) {
                    last_log = now;
                    LOG_INFO(HW_GPU,
                             "DMA snapshot: {} reads skipped the drain, {} drained, {} differed, "
                             "{} reads of listed reports (drained path); misses: {} new page, {} "
                             "compute/copy, {} unranged, {} noted write",
                             hits, misses, differences, report_reads, miss_reasons[0],
                             miss_reasons[1], miss_reasons[2], miss_reasons[3]);
                    if (ranged_writers) {
                        using Unbounded = VideoCommon::GpuSplit::DispatchUnbounded;
                        const auto& sent = split.UnboundedWriters();
                        const auto dispatches = split.GetDispatchStats();
                        const auto checks = [](DispatchCheck check) {
                            return dispatch_checks[static_cast<size_t>(check)];
                        };
                        const auto unbounded = [&dispatches](Unbounded reason) {
                            return dispatches.unbounded[static_cast<size_t>(reason)];
                        };
                        u64 dispatch_drains = 0;
                        for (const u64 count : dispatch_checks) {
                            dispatch_drains += count;
                        }
                        u64 dispatches_unbounded = 0;
                        for (const u64 count : dispatches.unbounded) {
                            dispatches_unbounded += count;
                        }
                        using Release = VideoCommon::GpuSplit::ReleaseNote;
                        const auto& releases = split.ReleaseNotes();
                        const auto release = [&releases](Release note) {
                            return releases[static_cast<size_t>(note)];
                        };
                        LOG_INFO(HW_GPU,
                                 "DMA snapshot EXP10: compute/copy: {} drained ({} unbounded "
                                 "writer, {} overlapping range; dispatches: {} queued, {} "
                                 "overlapping, {} unbounded, {} past the scan), {} skipped by "
                                 "ranges ({} differed), {} waited for dispatches only; dispatch "
                                 "launches sent: {}, dispatches run: {} bounded, {} unbounded ({} "
                                 "unconfirmed, {} replaced, {} image); unbounded writers sent: "
                                 "{} upload, {} release, {} compute, {} copy, {} other; compute "
                                 "releases: {} ranged, {} unmapped (write nothing), {} write "
                                 "nothing, unbounded: {} pending map, {} sparse, {} unmapped "
                                 "reduction, {} split",
                                 writer_unbounded + writer_overlaps + dispatch_drains,
                                 writer_unbounded, writer_overlaps, checks(DispatchCheck::Queued),
                                 checks(DispatchCheck::Overlap), checks(DispatchCheck::Unbounded),
                                 checks(DispatchCheck::ScanLimit), range_skips, range_differences,
                                 dispatch_waits, dispatches.sent, dispatches.bounded,
                                 dispatches_unbounded, unbounded(Unbounded::Unconfirmed),
                                 unbounded(Unbounded::Replaced), unbounded(Unbounded::Image),
                                 sent[0], sent[1], sent[2], sent[3], sent[4],
                                 release(Release::Ranged), release(Release::Unmapped),
                                 release(Release::NoWrite), release(Release::PendingMap),
                                 release(Release::Sparse), release(Release::LateUnmapped),
                                 release(Release::Split));
                    }
                }
                for (u32 i = 0; i < words; ++i) {
                    fs.parameters.push_back(snap_words[i]);
                }
                return;
            }
            ++report_reads;
        }
    }
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    if (full_drain) {
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    } else {
        split.WaitForMemoryWriters(VideoCommon::DrainSite::MacroDmaRead, source,
                                   source + read_bytes);
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    // EDEN_NX2_QUERY_DMA_TRACE: reads of reports whose host query result is still pending.
    if (VideoCommon::QueryDma::TraceEnabled()) {
        split.Run(
            [this, macro, method, source, read_bytes] {
                if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer, source,
                                                             read_bytes)) {
                    VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                 source, read_bytes);
                }
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    flush_if_dirty();
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
        split.Drain(VideoCommon::DrainSite::MacroDmaCheck);
        flush_if_dirty();
    }
    // Write out the reports held in the query cache, or the read returns stale occlusion counts.
    split.Run(
        [this, source, read_bytes] {
            VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source, read_bytes);
        },
        VideoCommon::DrainSite::MacroDmaFlush);
    u32 changed{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

bool Maxwell3D::FrontDmaReadEarly(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                                  u32 method, GPUVAddr source, u32 ram, u32 words) {
    namespace Profile = VideoCommon::Nx2Profile;
    const u64 bytes = u64{words} * sizeof(u32);
    const bool sparse = source < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
                        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < source + bytes;
    if (words == 0 || sparse) {
        return false;
    }
    const u64 consumed = split.SettleWrites();
    std::array<u64, 5> pages;
    size_t page_count = 0;
    u64 target = 0;
    u64 write_target = 0;
    for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + bytes; page += 0x1000) {
        const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page);
        if (page_count == pages.size() || !device || Tegra::Nx2IsGpuWritten(*device)) {
            return false;
        }
        const u64 device_page = *device & ~u64{0xfff};
        pages[page_count++] = device_page;
        target = (std::max)(target, split.ReadTarget(device_page, device_page + 0x1000));
        write_target = (std::max)(write_target,
                                  split.Writes().LastWriteTo(device_page, device_page + 0x1000));
    }
    const auto wait_reason = target == split.Writes().LastUnknown()
                                 ? Profile::Counter::DmaReadsWaitUnknown
                             : target == write_target ? Profile::Counter::DmaReadsWaitWrite
                                                      : Profile::Counter::DmaReadsWaitGpu;
    split.WaitUntil(target, VideoCommon::DrainSite::MacroDmaWrites);
    for (size_t i = 0; i < page_count; ++i) {
        if (Tegra::Nx2IsGpuWritten(pages[i])) {
            return false;
        }
    }
    if (Profile::Enabled()) {
        Profile::Add(Profile::Counter::DmaReadBytes, bytes);
        Profile::Add(Profile::Counter::DmaReadsEarly, 1);
        if (target > consumed) {
            Profile::Add(wait_reason, 1);
        }
    }
    auto& read = fs.dma_words;
    read.resize(words);
    for (u32 i = 0; i < words; ++i) {
        read[i] = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
    }
    if (perf_verify) {
        // The old path: drain, flush, read again. Its words are the ones used.
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
        if (memory_manager.IsMemoryDirty(source, bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, bytes] {
                    memory->FlushRegion(source, bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
        static u32 logged{};
        for (u32 i = 0; i < words; ++i) {
            const u32 drained = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
            if (drained != read[i] && logged < 64) {
                ++logged;
                LOG_ERROR(HW_GPU,
                          "Nx2Perf split: macro {:#x} DMA read of {} words at {:#x}: word {} is "
                          "{:#x} without the drain, {:#x} after it",
                          macro, words, source, i, read[i], drained);
            }
            read[i] = drained;
        }
    }
    for (u32 i = 0; i < words; ++i) {
        if (method == 0x158) {
            fs.parameters.push_back(read[i]);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = read[i];
        }
    }
    return true;
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    // EDEN_NX2_MACRO_SINK sinks macro writes too: in a tracking mode their shadow RAM copies are
    // written only when the sink is applied, so they count as done for the shadow check as well
    // (else the check would "resynchronize" the front's shadow to older values).
    std::array<u32, Regs::NUM_REGS> expected_shadow;
    std::memcpy(expected_shadow.data(), shadow_state.reg_array.data(), sizeof(expected_shadow));
    if (VideoCommon::GpuSplit::MacroSink() &&
        (shadow_state.shadow_ram_control == Regs::ShadowRamControl::Track ||
         shadow_state.shadow_ram_control == Regs::ShadowRamControl::TrackWithFilter)) {
        for (const auto& [method, value] : method_sink) {
            expected_shadow[method] = value;
        }
    }
    static u64 control_differences{};
    if ((fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected_shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.regs[i] != expected[i]) {
            if (differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
        }
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != expected_shadow[i]) {
            if (control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], expected_shadow[i]);
            }
            fs.shadow[i] = expected_shadow[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::MacroEnd};
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

namespace {

// EXP22 bit 2: the instance loop writes that may follow a draw.begin (Subsequent) in a draw
// batch. Written again with the same values they store what the registers hold and raise dirty
// flags that are raised (no deferred draw goes before an instance loop method), set draw_indexed
// again (index_buffer.count) or draw nothing (draw.end in DrawMode::Instance).
bool IsBatchMiddle(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw.end):
        return true;
    default:
        return false;
    }
}

// A held instance loop write (EXP22), sent as the entry it would have been. Draw manager
// methods: FrontMethodEffect's Gpu.
template <typename Write>
void SendHeldWrite(VideoCommon::GpuSplit& split, EngineInterface* engine, const Write& write) {
    constexpr auto effect = VideoCommon::GpuSplit::Effect::Gpu;
    if (write.from_macro) {
        split.MacroWrite(engine, write.method, write.argument, effect);
    } else {
        split.Method(engine, write.method, write.argument, write.is_last, nullptr, effect);
    }
}

// ApplyFrontDelta past a deferred draw, with the same-value skip and a shadow RAM that tracks
// (Track) or passes through: each write's register and its two dirty flags, raised together at
// the end (nothing reads them in between), as ProcessDirtyRegisters raises them one by one.
template <bool Track>
void ApplyDeltaWrites(Maxwell3D& maxwell3d, const u64* writes, u32 count) {
    static_assert(std::tuple_size<Maxwell3D::DirtyState::Tables>::value == 2);
    const auto& table0 = maxwell3d.dirty.tables[0];
    const auto& table1 = maxwell3d.dirty.tables[1];
    u32* const reg_array = maxwell3d.regs.reg_array.data();
    u32* const shadow_array = maxwell3d.shadow_state.reg_array.data();
    Maxwell3D::DirtyState::Flags raised;
    for (u32 i = 0; i < count; ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        const u32 value = static_cast<u32>(writes[i] >> 32);
        if constexpr (Track) {
            shadow_array[method] = value;
        }
        if (reg_array[method] == value) {
            continue;
        }
        reg_array[method] = value;
        raised[table0[method]] = true;
        raised[table1[method]] = true;
    }
    maxwell3d.dirty.flags |= raised;
}

// EXP22 bit 8 (back thread): every 65536th check, how many were made and how many differed.
struct Exp22Checks {
    u64 deltas;
    u64 delta_differences;
    u64 batches;
    u64 batch_differences;
};
Exp22Checks g_exp22_checks{};

void ReportExp22Checks() {
    const Exp22Checks& c = g_exp22_checks;
    if (((c.deltas + c.batches) & 0xffff) == 1) {
        LOG_INFO(HW_GPU,
                 "EXP22 check: {} state deltas checked, {} differed; {} draw batches checked, "
                 "{} differed",
                 c.deltas, c.delta_differences, c.batches, c.batch_differences);
    }
}

} // Anonymous namespace

void Maxwell3D::ApplyFrontDelta(Core::System& system, const u64* writes, u32 count) {
    // The Direct MacroWrite entries or the Pairs block the block stands for, each write as
    // CallMethod and ApplyPairs apply a state register write: the sink first, then the shadow
    // RAM, the same-value skip and ProcessDirtyRegisters. The shadow RAM mode can't change within
    // a block (its register is executable, and any other entry ends the block).
    ConsumeSink(system);
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, count);
    }
    const auto control = shadow_state.shadow_ram_control;
    const bool track = control == Regs::ShadowRamControl::Track ||
                       control == Regs::ShadowRamControl::TrackWithFilter;
    const bool fast = SkipSameValueWrites() &&
                      (track || control == Regs::ShadowRamControl::Passthrough);
    // One by one while a deferred instanced draw waits (the first changed write issues it), and
    // with a replaying shadow RAM or without the same-value skip.
    u32 i = 0;
    for (; i < count && (!fast || draw_manager.HasDeferredDraw()); ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        const u32 value = ProcessShadowRam(method, static_cast<u32>(writes[i] >> 32));
        if (!SkipSameValueWrites() || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    }
    if (i == count) {
        return;
    }
    if ((VideoCore::Nx2Debug::Experiment(22) & 8) != 0) [[unlikely]] {
        CheckFrontDelta(writes + i, count - i);
    } else if (track) {
        ApplyDeltaWrites<true>(*this, writes + i, count - i);
    } else {
        ApplyDeltaWrites<false>(*this, writes + i, count - i);
    }
}

void Maxwell3D::CheckFrontDelta(const u64* writes, u32 count) {
    // ApplyDeltaWrites, undone, then the single writes as CallMethodFast applies them (no
    // deferred draw waits here); the two results must match, and the second one stays.
    struct Saved {
        u32 reg;
        u32 shadow;
        u32 fast_reg;
        u32 fast_shadow;
    };
    thread_local std::vector<Saved> saved;
    saved.resize(count);
    const auto method_of = [writes](u32 k) { return static_cast<u32>(writes[k]) & 0xffff; };
    for (u32 k = 0; k < count; ++k) {
        saved[k].reg = regs.reg_array[method_of(k)];
        saved[k].shadow = shadow_state.reg_array[method_of(k)];
    }
    const DirtyState::Flags flags = dirty.flags;
    const auto control = shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track ||
        control == Regs::ShadowRamControl::TrackWithFilter) {
        ApplyDeltaWrites<true>(*this, writes, count);
    } else {
        ApplyDeltaWrites<false>(*this, writes, count);
    }
    const DirtyState::Flags fast_flags = dirty.flags;
    for (u32 k = 0; k < count; ++k) {
        saved[k].fast_reg = regs.reg_array[method_of(k)];
        saved[k].fast_shadow = shadow_state.reg_array[method_of(k)];
    }
    // Newest first: a register written twice gets its value from before the block back.
    for (u32 k = count; k-- > 0;) {
        regs.reg_array[method_of(k)] = saved[k].reg;
        shadow_state.reg_array[method_of(k)] = saved[k].shadow;
    }
    dirty.flags = flags;
    for (u32 k = 0; k < count; ++k) {
        const u32 value = ProcessShadowRam(method_of(k), static_cast<u32>(writes[k] >> 32));
        if (!SkipSameValueWrites() || regs.reg_array[method_of(k)] != value) {
            ProcessDirtyRegisters(method_of(k), value);
        }
    }
    u32 first = count;
    for (u32 k = 0; k < count && first == count; ++k) {
        if (regs.reg_array[method_of(k)] != saved[k].fast_reg ||
            shadow_state.reg_array[method_of(k)] != saved[k].fast_shadow) {
            first = k;
        }
    }
    const bool differs = first != count || fast_flags != dirty.flags;
    ++g_exp22_checks.deltas;
    if (differs && g_exp22_checks.delta_differences++ < 32) {
        LOG_ERROR(HW_GPU,
                  "EXP22 check: state delta of {} writes differs: register {:#x}, dirty flags {}",
                  count, first != count ? method_of(first) : 0U,
                  fast_flags != dirty.flags ? "differ" : "match");
    }
    ReportExp22Checks();
}

void Maxwell3D::RunDrawBatch(Core::System& system, const u64* writes, u32 count, u32 repeats) {
    // The first group as its single entries ran: the sink, then CallMethod (is_last_call only
    // matters to inline data).
    const auto run_group = [&] {
        for (u32 i = 0; i < count; ++i) {
            ConsumeSink(system);
            CallMethod(system, static_cast<u32>(writes[i]) & 0xffff,
                       static_cast<u32>(writes[i] >> 32), true);
        }
    };
    run_group();
    if (repeats <= 1) {
        return;
    }
    auto& state = draw_manager.draw_state;
    if ((VideoCore::Nx2Debug::Experiment(22) & 8) == 0) [[likely]] {
        // The repeats write what the registers and shadow RAM hold, raise flags that are raised,
        // set draw_indexed again and draw nothing (IsBatchMiddle): each counts an instance in
        // its draw.begin (DrawManager::DrawBegin, Subsequent).
        state.instance_count += repeats - 1;
        return;
    }
    // EXP22 bit 8: the repeats as their single entries, against that.
    constexpr u32 MaxWrites = FrontState::MaxBatchWrites;
    const u32 checked = (std::min)(count, MaxWrites);
    const u32 expected = state.instance_count + (repeats - 1);
    const DirtyState::Flags flags = dirty.flags;
    const auto mode = state.draw_mode;
    const bool indexed = state.draw_indexed;
    const auto topology = state.topology;
    std::array<u32, MaxWrites> group_regs{};
    std::array<u32, MaxWrites> group_shadow{};
    for (u32 i = 0; i < checked; ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        group_regs[i] = regs.reg_array[method];
        group_shadow[i] = shadow_state.reg_array[method];
    }
    for (u32 repeat = 1; repeat < repeats; ++repeat) {
        run_group();
    }
    bool differs = state.instance_count != expected || dirty.flags != flags ||
                   state.draw_mode != mode || state.draw_indexed != indexed ||
                   state.topology != topology;
    for (u32 i = 0; i < checked; ++i) {
        const u32 method = static_cast<u32>(writes[i]) & 0xffff;
        differs |= regs.reg_array[method] != group_regs[i] ||
                   shadow_state.reg_array[method] != group_shadow[i];
    }
    ++g_exp22_checks.batches;
    if (differs && g_exp22_checks.batch_differences++ < 32) {
        LOG_ERROR(HW_GPU,
                  "EXP22 check: draw batch of {} writes x {} differs: {} instances, expected {}",
                  count, repeats, state.instance_count, expected);
    }
    ReportExp22Checks();
}

bool Maxwell3D::FrontBatchWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                u32 argument, u32 value, bool is_last_call, bool from_macro) {
    const FrontState::BatchWrite write{method, argument, is_last_call, from_macro};
    if (method == MAXWELL3D_REG_INDEX(draw.begin)) {
        // Not with a replaying shadow RAM: the back's instance id is then the argument's.
        constexpr u32 subsequent = static_cast<u32>(Regs::Draw::InstanceId::Subsequent);
        if (fs.control == Regs::ShadowRamControl::Replay || ((value >> 26) & 3) != subsequent) {
            return false;
        }
        if (fs.capture_size != 0) {
            FrontEndCapture(split, fs);
        }
        split.SetPendingFlush(&FrontFlushPending, this);
        fs.capture[0] = write;
        fs.capture_size = 1;
        return true;
    }
    if (fs.capture_size == 0 || fs.capture_size == FrontState::MaxBatchWrites ||
        !IsBatchMiddle(method)) {
        // Sent as before, after what is held (GpuSplit::SelectEngine sends that first).
        return false;
    }
    fs.capture[fs.capture_size++] = write;
    return true;
}

bool Maxwell3D::FrontCaptureRepeats(const FrontState& fs) {
    if (fs.batch_size != fs.capture_size) {
        return false;
    }
    for (u32 i = 0; i < fs.capture_size; ++i) {
        if (fs.batch[i].method != fs.capture[i].method ||
            fs.batch[i].argument != fs.capture[i].argument) {
            return false;
        }
    }
    return true;
}

void Maxwell3D::FrontEndCapture(VideoCommon::GpuSplit& split, FrontState& fs) {
    // The captured writes are one more of the held batch, or the next batch to hold.
    if (FrontCaptureRepeats(fs)) {
        ++fs.batch_repeats;
        fs.capture_size = 0;
        return;
    }
    split.DropPendingFlush(this); // The held batch goes now, by itself
    FrontEmitBatch(split, fs);
    fs.batch = fs.capture;
    fs.batch_size = fs.capture_size;
    fs.batch_repeats = 1;
    fs.capture_size = 0;
    split.SetPendingFlush(&FrontFlushPending, this);
}

void Maxwell3D::FrontEmitBatch(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 size = fs.batch_size;
    const u32 repeats = fs.batch_repeats;
    fs.batch_size = 0;
    fs.batch_repeats = 0;
    if (size == 0) {
        return;
    }
    if (repeats > 1) {
        std::array<u64, FrontState::MaxBatchWrites> words;
        for (u32 i = 0; i < size; ++i) {
            words[i] = u64{fs.batch[i].method} | (u64{fs.batch[i].argument} << 32);
        }
        split.DrawBatch(this, words.data(), size, repeats, VideoCommon::GpuSplit::Effect::Gpu);
        return;
    }
    for (u32 i = 0; i < size; ++i) {
        SendHeldWrite(split, this, fs.batch[i]);
    }
}

void Maxwell3D::FrontEmitCapture(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 size = fs.capture_size;
    fs.capture_size = 0;
    for (u32 i = 0; i < size; ++i) {
        SendHeldWrite(split, this, fs.capture[i]);
    }
}

void Maxwell3D::FrontFlushPending(void* maxwell3d) {
    // GpuSplit cleared the hook before the call: what is sent here does not come back to it.
    auto* const self = static_cast<Maxwell3D*>(maxwell3d);
    FrontState& fs = *self->front;
    auto& split = VideoCommon::GpuSplit::Get();
    if (fs.capture_size != 0 && FrontCaptureRepeats(fs)) {
        ++fs.batch_repeats;
        fs.capture_size = 0;
    }
    self->FrontEmitBatch(split, fs);
    self->FrontEmitCapture(split, fs);
}

} // namespace Tegra::Engines
