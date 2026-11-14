// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <type_traits>

#include "common/common_types.h"

namespace VideoCommon {

/// What a buffer cache runtime offers for stream uniform reuse.
///  - StreamUniformRef: a small trivially copyable description of one stream allocation (the
///    host buffer, its offset and whatever else the runtime needs to add it to a descriptor
///    payload again, e.g. the device address of the descriptor buffer path).
///  - LastStreamUniformRef(): the allocation made by the latest BindMappedUniformBuffer call.
///  - RebindStreamUniform(ref, size): adds `ref` to the descriptor payload exactly as
///    BindMappedUniformBuffer added the allocation the first time.
template <class R>
concept StreamUniformReusable = requires(R& runtime, u32 size) {
    typename R::StreamUniformRef;
    requires std::is_trivially_copyable_v<typename R::StreamUniformRef>;
    { runtime.LastStreamUniformRef() } -> std::same_as<typename R::StreamUniformRef>;
    runtime.RebindStreamUniform(runtime.LastStreamUniformRef(), size);
    { runtime.CurrentTick() } -> std::convertible_to<u64>;
};

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBOREUSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

template <class Ref>
class UniformStreamReuse {
public:
    static constexpr u32 MaxBytes = 4096; // BufferCache's DEFAULT_SKIP_CACHE_SIZE
    static constexpr size_t NumStages = 5;
    static constexpr size_t NumSlots = 18; // Maxwell constant buffers per stage
    /// After this many binds in a row with other bytes than the slot's last, the slot's next
    /// DirectBinds binds skip the compare (per-draw matrices change every bind).
    static constexpr u16 MissStreak = 8;
    static constexpr u16 DirectBinds = 64;

    struct Slot {
        Ref ref{};
        u64 tick{};
        u64 device_addr{}; ///< Where `bytes` were last read
        u32 size{};        // 0: nothing streamed yet
        u16 misses{};      ///< Binds in a row whose bytes differed from the slot's last
        u16 direct{};      ///< Binds left that copy without comparing
        std::array<u8, MaxBytes> bytes{};
    };

    void Reset() noexcept {
        for (auto& stage : slots) {
            for (Slot& slot : stage) {
                slot.size = 0;
                slot.tick = 0;
                slot.misses = 0;
                slot.direct = 0;
            }
        }
    }

    std::array<std::array<Slot, NumSlots>, NumStages> slots{};
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    static UniformStreamReuse<Ref> state;
    return state;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, reusing the slot's previous allocation when the bytes and the command buffer are the
/// same. Returns false (nothing bound) when the buffer does not qualify, so the caller binds it
/// the classic way.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool BindSmallUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                           size_t stage, u32 index, u32 binding_index,
                                           u64 device_addr, u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    slot.device_addr = device_addr;
    if (slot.direct != 0) {
        // A slot whose bytes change every bind: read straight into its bytes, no compare (from
        // skolvolt 9267908a3d). The bytes stay the last bind's, for keeps and their verify.
        --slot.direct;
        device_memory.ReadBlockUnsafe(device_addr, slot.bytes.data(), size);
        slot.size = size;
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), slot.bytes.data(), size);
        slot.ref = runtime.LastStreamUniformRef();
        slot.tick = runtime.CurrentTick();
        return true;
    }
    u8* const scratch = reuse.scratch.data();
    device_memory.ReadBlockUnsafe(device_addr, scratch, size);

    const bool same_bytes =
        slot.size == size && std::memcmp(slot.bytes.data(), scratch, size) == 0;
    if (same_bytes && slot.tick == runtime.CurrentTick()) {
        slot.misses = 0;
        runtime.RebindStreamUniform(slot.ref, size);
        return true;
    }
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    std::memcpy(span.data(), scratch, size);
    slot.ref = runtime.LastStreamUniformRef();
    // After the allocation: the tick the stream region is tagged with is the one current now.
    slot.tick = runtime.CurrentTick();
    if (same_bytes) {
        slot.misses = 0;
    } else {
        slot.size = size;
        std::memcpy(slot.bytes.data(), scratch, size);
        if (++slot.misses >= Reuse::MissStreak) {
            slot.misses = 0;
            slot.direct = Reuse::DirectBinds;
        }
    }
    return true;
}

/// Whether RebindKeptUniform binds the slot's last allocation again for this range: the slot's
/// last bind read the same range in the command buffer that is current now.
template <class Runtime>
[[nodiscard]] bool KeptUniformMatches(Runtime& runtime, size_t stage, u32 index, u64 device_addr,
                                      u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    const typename Reuse::Slot& slot =
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().slots[stage][index];
    return slot.size == size && slot.device_addr == device_addr &&
           slot.tick == runtime.CurrentTick();
}

/// Uniform keeps: binds the slot's last allocation again without reading guest memory,
/// when that bind read the same range in this command buffer. The caller guarantees that nothing
/// ordered before this draw wrote the range since that bind (the previous draw's). Returns false
/// (nothing bound) otherwise.
template <class Runtime>
[[nodiscard]] bool RebindKeptUniform(Runtime& runtime, size_t stage, u32 index, u64 device_addr,
                                     u32 size) {
    if (!KeptUniformMatches(runtime, stage, index, device_addr, size)) {
        return false;
    }
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    const typename Reuse::Slot& slot =
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().slots[stage][index];
    runtime.RebindStreamUniform(slot.ref, size);
    return true;
}

/// EDEN_NX2_UBO_KEEP_VERIFY: whether guest memory still holds the bytes the slot's last
/// bind read from the range RebindKeptUniform would reuse.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool KeptUniformHolds(DeviceMemory& device_memory, size_t stage, u32 index,
                                    u64 device_addr, u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    const typename Reuse::Slot& slot = reuse.slots[stage][index];
    if (slot.size != size || slot.device_addr != device_addr) {
        return false;
    }
    device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
    return std::memcmp(slot.bytes.data(), reuse.scratch.data(), size) == 0;
}

} // namespace VideoCommon
