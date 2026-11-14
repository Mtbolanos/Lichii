// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <mutex>
#include <vector>

#include "common/logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu_trace/state.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/macro.h"
#include "video_core/memory_manager.h"
#include "video_core/native_qmd.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon::GpuTrace {
namespace {
using Tegra::Control::ChannelState;
using Tegra::Engines::EngineTypes;
using Tegra::Engines::Maxwell3D;

/// Appends trivially copyable values to a byte buffer, and reads them back in the same order.
class Bytes {
public:
    template <typename T>
    void Put(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* begin = reinterpret_cast<const u8*>(&value);
        data.insert(data.end(), begin, begin + sizeof(T));
    }

    std::vector<u8> data;
};

class Cursor {
public:
    explicit Cursor(std::span<const u8> bytes_) : bytes{bytes_} {}

    template <typename T>
    bool Get(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (bytes.size() - offset < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }

    [[nodiscard]] bool Done() const {
        return offset == bytes.size();
    }

    [[nodiscard]] size_t Remaining() const {
        return bytes.size() - offset;
    }

private:
    std::span<const u8> bytes;
    size_t offset = 0;
};

void Emit(Writer& writer, StateKind kind, s32 channel, const Bytes& bytes) {
    const StateRecord header{static_cast<u32>(kind), channel};
    const std::array<std::span<const u8>, 2> parts{
        std::span<const u8>{reinterpret_cast<const u8*>(&header), sizeof(header)}, bytes.data};
    writer.Write(RecordType::State, parts);
}

struct MacroEngineState {
    std::array<u32, 0x8000> code;
    u32 write_offset;
    u32 execution_failed;
};

void PutMacroEngine(Bytes& bytes, const Tegra::MacroEngine& engine) {
    MacroEngineState state{};
    state.code = engine.native_code;
    state.write_offset = engine.native_write_offset;
    state.execution_failed = engine.native_execution_failed ? 1 : 0;
    bytes.Put(state);
}

bool GetMacroEngine(Cursor& cursor, Tegra::MacroEngine& engine) {
    MacroEngineState state{};
    if (!cursor.Get(state)) {
        return false;
    }
    engine.native_code = state.code;
    engine.native_write_offset = state.write_offset;
    engine.native_execution_failed = state.execution_failed != 0;
    // Compiled, memoized and JIT code key on the generation.
    ++engine.native_code_generation;
    return true;
}

Tegra::Engines::EngineInterface* EngineOfType(ChannelState::Payload& payload, EngineTypes type) {
    switch (type) {
    case EngineTypes::Nv01Timer:
        return &payload.nv01_timer;
    case EngineTypes::KeplerCompute:
        return &payload.kepler_compute;
    case EngineTypes::Maxwell3D:
        return &payload.maxwell_3d;
    case EngineTypes::Fermi2D:
        return &payload.fermi_2d;
    case EngineTypes::MaxwellDMA:
        return &payload.maxwell_dma;
    case EngineTypes::KeplerMemory:
        return &payload.kepler_memory;
    }
    return nullptr;
}

template <typename Regs>
void PutRegs(Writer& writer, StateKind kind, s32 channel, const Regs& regs) {
    Bytes bytes;
    bytes.Put(regs);
    Emit(writer, kind, channel, bytes);
}

template <typename Regs>
bool GetRegs(Cursor& cursor, Regs& regs) {
    return cursor.Get(regs) && cursor.Done();
}
} // Anonymous namespace

AddressSpaceRecord StateAccess::AddressSpace(const Tegra::MemoryManager& memory_manager) {
    return {static_cast<u32>(memory_manager.GetID()),
            static_cast<u32>(memory_manager.address_space_bits), memory_manager.split_address,
            static_cast<u32>(memory_manager.big_page_bits),
            static_cast<u32>(memory_manager.page_bits)};
}

std::vector<MapLog::Entry> StateAccess::Maps(const Tegra::MemoryManager& memory_manager) {
    return memory_manager.trace_maps.Entries();
}

void StateAccess::Settle(ChannelState& channel) {
    auto& payload = *channel.payload;
    if (payload.maxwell_3d.front_macros) {
        payload.maxwell_3d.ReleaseFront();
    }
    if (payload.kepler_compute.front_macros) {
        payload.kepler_compute.ReleaseFront();
    }
}

void StateAccess::CaptureChannel(Writer& writer, ChannelState& channel) {
    auto& payload = *channel.payload;
    const s32 id = channel.bind_id;
    {
        auto& pusher = payload.dma_pusher;
        Bytes bytes;
        bytes.Put(pusher.dma_state);
        bytes.Put(static_cast<u32>(pusher.dma_increment_once ? 1 : 0));
        std::array<u32, 8> types{};
        for (size_t i = 0; i < types.size(); ++i) {
            types[i] = pusher.subchannels[i] != nullptr
                           ? static_cast<u32>(pusher.subchannel_type[i]) + 1
                           : 0;
        }
        bytes.Put(types);
        bytes.Put(pusher.dma_semaphore);
        bytes.Put(pusher.puller.regs);
        bytes.Put(pusher.puller.bound_engines);
        Emit(writer, StateKind::Pusher, id, bytes);
    }
    {
        auto& maxwell = payload.maxwell_3d;
        Bytes bytes;
        bytes.Put(maxwell.regs);
        bytes.Put(maxwell.shadow_state);
        bytes.Put(maxwell.state);
        bytes.Put(maxwell.root_tables);
        bytes.Put(maxwell.native_data_ram);
        bytes.Put(static_cast<u32>(maxwell.execute_on ? 1 : 0));
        Emit(writer, StateKind::Maxwell3D, id, bytes);
    }
    {
        auto& compute = payload.kepler_compute;
        Bytes bytes;
        bytes.Put(compute.regs);
        bytes.Put(compute.launch_description);
        bytes.Put(compute.native_registers);
        bytes.Put(static_cast<u32>(compute.native_compute_uploaded ? 1 : 0));
        bytes.Put(compute.native_bind_pointer);
        Emit(writer, StateKind::KeplerCompute, id, bytes);
    }
    if (!payload.maxwell_3d.native_macro_context) {
        const auto& maxwell = payload.maxwell_3d;
        Bytes bytes;
        bytes.Put(maxwell.macro_positions);
        for (const auto& [method, code] : maxwell.macro_engine.uploaded_macro_code) {
            bytes.Put(method);
            bytes.Put(static_cast<u32>(code.size()));
            for (const u32 word : code) {
                bytes.Put(word);
            }
        }
        Emit(writer, StateKind::Macros, id, bytes);
    }
    PutRegs(writer, StateKind::Fermi2D, id, payload.fermi_2d.regs);
    PutRegs(writer, StateKind::MaxwellDMA, id, payload.maxwell_dma.regs);
    PutRegs(writer, StateKind::KeplerMemory, id, payload.kepler_memory.regs);
}

void StateAccess::CaptureGlobals(Writer& writer, ChannelState* any_channel,
                                 Tegra::Host1x::SyncpointManager& syncpoints) {
    if (any_channel != nullptr) {
        if (const auto& context = any_channel->payload->maxwell_3d.native_macro_context) {
            Bytes bytes;
            PutMacroEngine(bytes, context->engine);
            bytes.Put(context->positions);
            PutMacroEngine(bytes, context->compute_engine);
            bytes.Put(context->compute_positions);
            Emit(writer, StateKind::MacroContext, -1, bytes);
        }
    }
    {
        Bytes bytes;
        for (const auto& [address, state] : Tegra::NativeQmdScheduler::states) {
            bytes.Put(address);
            bytes.Put(state);
        }
        Emit(writer, StateKind::QmdStates, -1, bytes);
    }
    {
        Bytes bytes;
        for (u32 i = 0; i < Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS; ++i) {
            bytes.Put(syncpoints.GetGuestSyncpointValue(i));
            bytes.Put(syncpoints.GetHostSyncpointValue(i));
        }
        Emit(writer, StateKind::Syncpoints, -1, bytes);
    }
    {
        Bytes bytes;
        std::shared_lock lock{Tegra::g_nx2_semaphore_regions_mutex};
        for (const u64 start : Tegra::g_nx2_semaphore_regions) {
            bytes.Put(start);
        }
        Emit(writer, StateKind::SemaphoreRegions, -1, bytes);
    }
}

bool StateAccess::Restore(const Record& record, ChannelState* channel, ChannelState* any_channel,
                          Tegra::Host1x::SyncpointManager& syncpoints) {
    if (!HasPayload(record, sizeof(StateRecord))) {
        return false;
    }
    const auto& header = record.As<StateRecord>();
    Cursor cursor{std::span<const u8>{record.payload}.subspan(sizeof(StateRecord))};
    const auto kind = static_cast<StateKind>(header.kind);
    const bool per_channel = (kind >= StateKind::Pusher && kind <= StateKind::KeplerMemory) ||
                             kind == StateKind::Macros;
    if (per_channel && (channel == nullptr || !channel->payload)) {
        LOG_ERROR(HW_GPU, "GPU trace: state for unknown channel {}", header.channel);
        return false;
    }
    switch (kind) {
    case StateKind::Pusher: {
        auto& payload = *channel->payload;
        auto& pusher = payload.dma_pusher;
        u32 increment_once{};
        std::array<u32, 8> types{};
        if (!cursor.Get(pusher.dma_state) || !cursor.Get(increment_once) || !cursor.Get(types) ||
            !cursor.Get(pusher.dma_semaphore) || !cursor.Get(pusher.puller.regs) ||
            !cursor.Get(pusher.puller.bound_engines) || !cursor.Done()) {
            return false;
        }
        pusher.dma_increment_once = increment_once != 0;
        for (u32 i = 0; i < types.size(); ++i) {
            if (types[i] == 0) {
                continue;
            }
            const auto type = static_cast<EngineTypes>(types[i] - 1);
            if (auto* engine = EngineOfType(payload, type)) {
                pusher.BindSubchannel(engine, i, type);
            }
        }
        return true;
    }
    case StateKind::Maxwell3D: {
        auto& maxwell = channel->payload->maxwell_3d;
        u32 execute_on{};
        if (!cursor.Get(maxwell.regs) || !cursor.Get(maxwell.shadow_state) ||
            !cursor.Get(maxwell.state) || !cursor.Get(maxwell.root_tables) ||
            !cursor.Get(maxwell.native_data_ram) || !cursor.Get(execute_on) || !cursor.Done()) {
            return false;
        }
        maxwell.execute_on = execute_on != 0;
        maxwell.dirty.flags.set();
        return true;
    }
    case StateKind::KeplerCompute: {
        auto& compute = channel->payload->kepler_compute;
        u32 uploaded{};
        if (!cursor.Get(compute.regs) || !cursor.Get(compute.launch_description) ||
            !cursor.Get(compute.native_registers) || !cursor.Get(uploaded) ||
            !cursor.Get(compute.native_bind_pointer) || !cursor.Done()) {
            return false;
        }
        compute.native_compute_uploaded = uploaded != 0;
        if (auto* context = compute.native_context) {
            compute.native_macro =
                compute.native_compute_uploaded ? &context->compute_engine : &context->engine;
            compute.native_macro_positions = compute.native_compute_uploaded
                                                 ? std::span<u32>{context->compute_positions}
                                                 : std::span<u32>{context->positions};
        }
        return true;
    }
    case StateKind::Macros: {
        auto& maxwell = channel->payload->maxwell_3d;
        if (!cursor.Get(maxwell.macro_positions)) {
            return false;
        }
        auto& engine = maxwell.macro_engine;
        engine.macro_cache.clear();
        engine.uploaded_macro_code.clear();
        while (!cursor.Done()) {
            u32 method{};
            u32 size{};
            if (!cursor.Get(method) || !cursor.Get(size) ||
                size > cursor.Remaining() / sizeof(u32)) {
                return false;
            }
            auto& code = engine.uploaded_macro_code[method];
            code.resize(size);
            for (u32& word : code) {
                if (!cursor.Get(word)) {
                    return false;
                }
            }
        }
        return true;
    }
    case StateKind::Fermi2D:
        return GetRegs(cursor, channel->payload->fermi_2d.regs);
    case StateKind::MaxwellDMA:
        return GetRegs(cursor, channel->payload->maxwell_dma.regs);
    case StateKind::KeplerMemory:
        return GetRegs(cursor, channel->payload->kepler_memory.regs);
    case StateKind::MacroContext: {
        if (any_channel == nullptr || !any_channel->payload) {
            return false;
        }
        const auto& context = any_channel->payload->maxwell_3d.native_macro_context;
        if (!context) {
            LOG_ERROR(HW_GPU, "GPU trace: the trace has Switch 2 macros, the replay's channels don't");
            return false;
        }
        return GetMacroEngine(cursor, context->engine) && cursor.Get(context->positions) &&
               GetMacroEngine(cursor, context->compute_engine) &&
               cursor.Get(context->compute_positions) && cursor.Done();
    }
    case StateKind::QmdStates: {
        auto& states = Tegra::NativeQmdScheduler::states;
        states.clear();
        while (!cursor.Done()) {
            Tegra::NativeQmdScheduler::Address address{};
            Tegra::NativeQmdScheduler::State state{};
            if (!cursor.Get(address) || !cursor.Get(state)) {
                return false;
            }
            states.insert_or_assign(address, state);
        }
        return true;
    }
    case StateKind::Syncpoints:
        for (u32 i = 0; i < Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS; ++i) {
            u32 guest{};
            u32 host{};
            if (!cursor.Get(guest) || !cursor.Get(host)) {
                return false;
            }
            syncpoints.syncpoints_guest[i].store(guest, std::memory_order_release);
            syncpoints.syncpoints_host[i].store(host, std::memory_order_release);
        }
        return cursor.Done();
    case StateKind::SemaphoreRegions:
        Tegra::Nx2ClearSemaphoreRegions();
        while (!cursor.Done()) {
            u64 start{};
            if (!cursor.Get(start)) {
                return false;
            }
            Tegra::Nx2AddSemaphoreRegion(start);
        }
        return true;
    }
    LOG_WARNING(HW_GPU, "GPU trace: skipping unknown state kind {}", header.kind);
    return true;
}

void StateAccess::Rebind(ChannelState& channel) {
    auto& maxwell = channel.payload->maxwell_3d;
    auto* const rasterizer = maxwell.rasterizer;
    for (size_t stage = 0; stage < maxwell.state.shader_stages.size(); ++stage) {
        const auto& buffers = maxwell.state.shader_stages[stage].const_buffers;
        for (u32 slot = 0; slot < buffers.size(); ++slot) {
            const auto& buffer = buffers[slot];
            if (buffer.enabled) {
                rasterizer->BindGraphicsUniformBuffer(stage, slot, buffer.address, buffer.size);
            } else {
                rasterizer->DisableGraphicsUniformBuffer(stage, slot);
            }
        }
    }
    maxwell.dirty.flags.set();
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::TickWork);
}

} // namespace VideoCommon::GpuTrace
