// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

/// A draw that follows a draw of the same pipeline with only light state changes in between takes
/// the delta path. It keeps the previous draw's pipeline, image views, samplers and texture buffer
/// bindings, and resolves and binds every buffer (index, vertex, uniform, storage, texture) again
/// as a full resolve does, so guest memory written without a notification is streamed as in a
/// full resolve.
/// Light changes are draw parameters, index buffer and vertex stream registers, constant buffer
/// binds and inline constant buffer data (collected in g_delta: the rasterizer resolves in full
/// when they reach the pipeline's texture handles at their GPU addresses or through another
/// mapping of their device memory). Anything else that could change what a resolve produces bumps
/// a serial: a 3D register that changes, any 3D method with side effects (uploads, reports,
/// clears, barriers, ...), constant buffer data reaching the texture or sampler headers (the same
/// way), any other engine, deferred calls on the split's back (except the ones marked
/// GpuSplit::FastDrawNeutral), GPU::TickWork requests, texture cache writes, unmaps and
/// deletions, shader removals, fence releases, HLE macros and frames. The rasterizer takes the
/// delta path only while the serial, the GPU memory map, the command buffer, the render pass and
/// the bound pipeline are the ones the previous draw left.
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the pipeline, views,
/// samplers or kept buffer bindings differ.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    Map,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "fence release", "indirect reset", "frame",           "texture header data",
    "memory map",
};

inline std::atomic<bool> g_enabled{true};

/// Whether pipelines take their vertex input as dynamic state (VK_EXT_vertex_input_dynamic_state):
/// then the vertex attribute formats are draw parameters, as only each attribute's type class is
/// in the pipeline key. Set by the pipeline cache before the GPU threads run, read on the back.
inline std::atomic<bool> g_dynamic_vertex_input{};

/// Any thread: whether fast draws run (always, unless a test turned them off). The rasterizer
/// latches it per draw (on the thread that draws) and starts with a full resolve when it turns on.
[[nodiscard]] inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Tests only.
inline void SetEnabledForTest(bool on) {
    g_enabled.store(on, std::memory_order_relaxed);
}

/// EDEN_NX2_FAST_DRAW_VERIFY=1, read once.
[[nodiscard]] inline bool VerifyEnabled() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// On its own cache line, away from what every thread reads (g_enabled).
alignas(64) inline std::atomic<u64> g_serial{1};
/// The last break's reason and, for Register and Method, its 3D method (diagnostics). Stored by
/// every break, also those that skip the add below, so a register block's one break names what
/// its writes one by one would.
inline std::atomic<u32> g_last_break{};

/// The thread that draws reads the serial before each draw (Serial). Once one of its breaks
/// bumped the serial since that read, more change nothing it compares, so they skip the locked
/// add (every changed 3D register breaks). Threads that never read it (the front's puller, guest
/// CPU write traps, unmaps) always bump. From lucastaonline's 865afaa4f3 (#1161).
inline thread_local bool t_reads_serial{};
inline thread_local bool t_serial_bumped{};

/// Bumped where the guest CPU may have synchronised with the GPU: fences, semaphores, syncpoints,
/// reports, frames, other engines and requests from other threads. After one the guest may have
/// written memory the caches were not told about (pages that aren't write-protected), so uniform
/// keeps read their buffers again.
inline std::atomic<u64> g_sync_serial{1};

inline void NoteSync() {
    g_sync_serial.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline u64 SyncSerial() {
    return g_sync_serial.load(std::memory_order_acquire);
}

inline void Break(Reason reason, u32 method = 0) {
    // 3D methods that let the CPU synchronise (reports, syncpoints) note it where they reach the
    // rasterizer. Invalidations end image and shader lifetimes, not guest memory writes.
    if (reason != Reason::Register && reason != Reason::Method && reason != Reason::ConstantData &&
        reason != Reason::TextureHeaders && reason != Reason::Invalidation) {
        NoteSync();
    }
    if (!Enabled()) {
        return;
    }
    g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    if (!(t_reads_serial && t_serial_bumped)) {
        g_serial.fetch_add(1, std::memory_order_relaxed);
        t_serial_bumped = t_reads_serial;
    }
}

/// Unsplit counterparts of commands whose split entries already end draw reuse.
inline void BreakUnsplit(Reason reason) {
    Break(reason);
}

/// The thread that draws: what it compares fast draws with (and from then on its next break
/// bumps the serial again).
inline u64 Serial() {
    t_reads_serial = true;
    t_serial_bumped = false;
    return g_serial.load(std::memory_order_relaxed);
}

/// The device memory a GPU range maps to. A write through another GPU address that maps the
/// same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless that range of memory was translated last
    /// and memory's mappings haven't changed since.
    void Update(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const;

    /// The GPU address the range starts at.
    [[nodiscard]] GPUVAddr Address() const noexcept {
        return address;
    }

    /// Whether the range maps to one run of device memory in GPU order, starting at LinearBase.
    [[nodiscard]] bool IsLinear() const noexcept {
        return linear;
    }
    [[nodiscard]] DAddr LinearBase() const noexcept {
        return linear_base;
    }

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    bool linear{};
    DAddr linear_base{};
    std::vector<std::pair<DAddr, DAddr>> spans; ///< Sorted, disjoint [begin, end) pairs
};

/// Light changes since the last draw that can reach texture handles. Written and read on the
/// thread that runs the 3D engine and the rasterizer (the back, or sole GPU thread).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    /// Constant buffer data written, by GPU address (unmapped addresses included)...
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    /// ...and by the device address it landed on.
    std::array<DAddr, MaxWrites> device_begin{};
    std::array<DAddr, MaxWrites> device_end{};
    u32 num_device_writes{};
    /// Set when the ranges were widened (AppendRange) or reached a buffer of texture handles
    /// between its handles: the draw keeps no uniforms, as when such writes ended draw reuse.
    bool no_uniform_keeps{};

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    /// Whether data written reaches [address, address + size), whose device memory is spans: at
    /// one of its GPU addresses, or through any GPU address that maps its device memory.
    [[nodiscard]] bool Reaches(GPUVAddr address, u64 size, const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (spans.Intersects(device_begin[write], device_end[write])) {
                return true;
            }
        }
        return false;
    }

    /// Reaches for only the bytes [base + first, base + second) of each of words, all inside the
    /// range spans was updated with. Device writes are compared word by word when that range maps
    /// linearly, else with the whole range.
    [[nodiscard]] bool ReachesWords(GPUVAddr base, std::span<const std::pair<u32, u32>> words,
                                    const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            for (const auto& [first, second] : words) {
                if (write_begin[write] < base + second && base + first < write_end[write]) {
                    return true;
                }
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (!spans.Intersects(device_begin[write], device_end[write])) {
                continue;
            }
            if (!spans.IsLinear()) {
                return true;
            }
            for (const auto& [first, second] : words) {
                const DAddr begin = spans.LinearBase() + (base + first - spans.Address());
                const DAddr end = begin + (second - first);
                if (device_begin[write] < end && begin < device_end[write]) {
                    return true;
                }
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// Moves g_delta into `out` and clears it, copying only the ranges in use (Delta is ~0.5 KiB and
/// taken for every draw).
inline void TakeDelta(Delta& out) {
    Delta& delta = g_delta;
    out.binds = delta.binds;
    out.num_writes = delta.num_writes;
    std::copy_n(delta.write_begin.begin(), delta.num_writes, out.write_begin.begin());
    std::copy_n(delta.write_end.begin(), delta.num_writes, out.write_end.begin());
    out.num_device_writes = delta.num_device_writes;
    std::copy_n(delta.device_begin.begin(), delta.num_device_writes, out.device_begin.begin());
    std::copy_n(delta.device_end.begin(), delta.num_device_writes, out.device_end.begin());
    out.no_uniform_keeps = delta.no_uniform_keeps;
    delta.binds = {};
    delta.num_writes = 0;
    delta.num_device_writes = 0;
    delta.no_uniform_keeps = false;
}

/// Adds [begin, end) to a delta's range list: extends a range it touches or overlaps, else
/// appends it. A full list widens the range that grows least instead (true then), which only
/// makes Reaches more conservative.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (begin <= ends[i] && begins[i] <= end) {
            begins[i] = (std::min)(begins[i], begin);
            ends[i] = (std::max)(ends[i], end);
            return false;
        }
    }
    if (count < Delta::MaxWrites) {
        begins[count] = begin;
        ends[count] = end;
        ++count;
        return false;
    }
    u32 best = 0;
    u64 best_growth = ~u64{0};
    for (u32 i = 0; i < count; ++i) {
        const u64 growth = ((std::max)(ends[i], end) - (std::min)(begins[i], begin)) -
                           (ends[i] - begins[i]);
        if (growth < best_growth) {
            best = i;
            best_growth = growth;
        }
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    return true;
}

inline void NoteBind(size_t stage, u32 slot) {
    if (Enabled()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// Inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (Enabled()) {
        Delta& delta = g_delta;
        delta.no_uniform_keeps |=
            AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end);
    }
}

/// The device addresses [begin, end) that inline constant buffer data landed on.
inline void NoteDeviceData(DAddr begin, DAddr end) {
    if (Enabled()) {
        Delta& delta = g_delta;
        delta.no_uniform_keeps |= AppendRange(delta.device_begin, delta.device_end,
                                              delta.num_device_writes, begin, end);
    }
}

} // namespace VideoCommon::Nx2FastDraw
