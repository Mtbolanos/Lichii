// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <exception>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/object_pool.h"

namespace Shader {
class Environment;
struct HostTranslateInfo;
} // namespace Shader

namespace VideoCommon {

class ShaderJobScheduler;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

/// One graphics stage of a pipeline translated on a CPU worker of the shader jobs, so the stages
/// of a pipeline translate at once (shader cache v2). Each stage has its own environment and IR
/// pools: an environment memoizes what it reads, and pools are not thread-safe.
struct StageTranslation {
    Shader::Environment* env{};
    ShaderPools* pools{};
    u32 cfg_offset{};
    bool exits_to_dispatcher{};

    std::optional<Shader::IR::Program> program;
    std::exception_ptr exception;
    /// Time the translation took on its thread
    std::chrono::nanoseconds time{};

    /// The translated program; rethrows what the translation threw
    Shader::IR::Program Take();
};

/// Bytes of one texture descriptor (Tegra::Texture::TICEntry)
inline constexpr u64 TextureDescriptorBytes = 32;

/// The largest texture descriptor pool the GPU thread flushes before a pipeline's stages
/// translate on workers; past it they translate on the GPU thread. The flush copies nothing: the
/// caches write back what the host GPU wrote over the range, one call per mapped big page (64 or
/// 128 KiB), so a pool this size costs at most 64 calls per miss. Ocarina of Time's pool is
/// 1,058,176 bytes, just over the 1 MiB this was before.
inline constexpr u64 MaxFlushedTexturePoolBytes = u64{4} << 20;

/// Bytes of a texture descriptor pool whose last index is `tic_limit`
[[nodiscard]] constexpr u64 TexturePoolBytes(u32 tic_limit) noexcept {
    return (u64{tic_limit} + 1) * TextureDescriptorBytes;
}

/// True when the pool of `tic_limit` is small enough to flush for a translation on workers
[[nodiscard]] constexpr bool CanFlushTexturePool(u32 tic_limit) noexcept {
    return TexturePoolBytes(tic_limit) <= MaxFlushedTexturePoolBytes;
}

/// Address of the last byte of the pool at `pool_addr` whose last index is `tic_limit`, or none
/// when it wraps past 2^64. The pool flushes whole only when this is inside the GPU address
/// space: MemoryManager::FlushRegion does nothing for a range that runs past its end.
[[nodiscard]] constexpr std::optional<u64> TexturePoolLastAddress(u64 pool_addr,
                                                                  u32 tic_limit) noexcept {
    const u64 last_offset = TexturePoolBytes(tic_limit) - 1;
    if (pool_addr > ~u64{0} - last_offset) {
        return std::nullopt;
    }
    return pool_addr + last_offset;
}

/// Translates every stage as a NeedNow job of `jobs`, the calling thread running those no worker
/// took, and returns once all are done. The environments must not be used elsewhere meanwhile.
void TranslateStages(ShaderJobScheduler& jobs, std::span<StageTranslation> stages,
                     const Shader::HostTranslateInfo& host_info);

} // namespace VideoCommon
