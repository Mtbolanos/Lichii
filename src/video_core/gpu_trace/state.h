// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"
#include <vector>

#include "video_core/gpu_trace/map_log.h"
#include "video_core/gpu_trace/trace_file.h"

namespace Tegra {
class GPU;
class MemoryManager;
namespace Control {
struct ChannelState;
}
namespace Host1x {
class SyncpointManager;
}
} // namespace Tegra

namespace VideoCommon::GpuTrace {

/// What a State record holds.
enum class StateKind : u32 {
    /// Per channel: the DmaPusher's method state and subchannel bindings, and the Puller's.
    Pusher = 1,
    Maxwell3D = 2,
    KeplerCompute = 3,
    Fermi2D = 4,
    MaxwellDMA = 5,
    KeplerMemory = 6,
    /// The Switch 2 macro RAM all channels share (both macro engines and their bind tables).
    MacroContext = 7,
    /// NativeQmdScheduler: per QMD address its descriptor and dependency counters.
    QmdStates = 8,
    /// Host1x syncpoint values, guest and host.
    Syncpoints = 9,
    /// Switch 2 queue semaphore regions (their GPU address starts).
    SemaphoreRegions = 10,
    /// Per channel, Switch 1 only: the 3D engine's uploaded macro code and its bind table.
    Macros = 11,
};

/// Reads and writes the GPU state a replay needs that isn't in memory. Every engine it reads
/// names it a friend.
struct StateAccess {
    /// Writes the State records of one channel. Call on the thread that owns the engines, with
    /// the channel at a command boundary (no macro call or upload in flight, the split's front
    /// handed back).
    static void CaptureChannel(Writer& writer, Tegra::Control::ChannelState& channel);
    /// Writes the State records shared by every channel.
    static void CaptureGlobals(Writer& writer, Tegra::Control::ChannelState* any_channel,
                               Tegra::Host1x::SyncpointManager& syncpoints);

    /// Applies a State record. `channel` is the record's channel, initialized (GPU::InitChannel),
    /// or null for a global one. Returns false if the record doesn't fit this build.
    static bool Restore(const Record& record, Tegra::Control::ChannelState* channel,
                        Tegra::Control::ChannelState* any_channel,
                        Tegra::Host1x::SyncpointManager& syncpoints);

    /// After every State record: points the rasterizer at the restored constant buffers. Runs on
    /// the GPU thread with the channel bound.
    static void Rebind(Tegra::Control::ChannelState& channel);

    /// Hands the split's front state back to the engines and applies batched writes, so the
    /// engine objects hold the channel's whole state (before CaptureChannel).
    static void Settle(Tegra::Control::ChannelState& channel);

    /// A GPU address space's page geometry.
    static AddressSpaceRecord AddressSpace(const Tegra::MemoryManager& memory_manager);
    /// What it maps.
    static std::vector<MapLog::Entry> Maps(const Tegra::MemoryManager& memory_manager);
};

} // namespace VideoCommon::GpuTrace
