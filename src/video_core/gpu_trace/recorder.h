// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>

#include "common/common_types.h"
#include "video_core/gpu_trace/map_log.h"

namespace Core {
class System;
}
namespace Service::Nvidia {
struct NvFence;
}
namespace Tegra {
class MemoryManager;
struct CommandList;
struct FramebufferConfig;
} // namespace Tegra

namespace VideoCommon::GpuTrace {

/// From a request until the recording ends: the GPU thread calls OnCommand.
inline std::atomic<bool> g_requested{false};
/// While a recording runs (from its snapshot): the Note* hooks record.
inline std::atomic<bool> g_recording{false};
/// EDEN_NX2_GPU_TRACE=<dir>: a recording starts when <dir>/record.now appears.
inline const bool g_file_trigger = std::getenv("EDEN_NX2_GPU_TRACE") != nullptr;

struct RecordOptions {
    std::filesystem::path path;
    /// The recording ends after this many presents (180: 3 s at 60 fps).
    u32 presents = 180;
};

/// Asks the GPU thread to start recording at its next command with no suspended channel. Any
/// thread. False with `error` set while a recording is pending or running.
bool RequestRecording(const RecordOptions& options, std::string& error);

/// Ends the recording at the GPU thread's next command; a pending one is dropped.
void StopRecording();

struct RecorderStatus {
    bool requested{};
    bool recording{};
    u32 presents{};
    u32 presents_wanted{};
    u64 bytes{};
    std::string path;
    std::string last_trace;
    std::string last_error;
};
[[nodiscard]] RecorderStatus GetRecorderStatus();

/// The guest CPU wrote [pointer, pointer + size) of guest physical memory. Call after the store.
void NoteCpuWrite(const u8* pointer, size_t size);
/// The same, by device address, after a service wrote the range (GPU::InvalidateRegion).
void NoteDeviceWrite(DAddr address, u64 size);

/// The GPU thread stopped: writes what was recorded and releases the recorder.
void OnGpuShutdown();

void NoteGmmuMap(const Tegra::MemoryManager& memory_manager, const MapLog::Entry& entry);
void NoteGmmuUnmap(const Tegra::MemoryManager& memory_manager, GPUVAddr gpu_address, u64 size);

/// GPU::RequestComposite was asked for this present: it is recorded at the GPU tick after it.
void NotePresent(std::span<const Tegra::FramebufferConfig> layers,
                 std::span<const Service::Nvidia::NvFence> fences);

/// The GPU thread popped a command and is about to run it: `submit` for a command list,
/// `tick` for a GPU tick (which runs presents). Takes the snapshot when a recording is due and
/// records the memory the CPU wrote since the last command, then the command.
void OnCommand(Core::System& system, bool channels_pending, const Tegra::CommandList* submit,
               s32 channel, bool tick);

} // namespace VideoCommon::GpuTrace
