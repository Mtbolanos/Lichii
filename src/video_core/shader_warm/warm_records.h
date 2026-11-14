// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/pipeline_cache_files.h"

namespace VideoCommon::ShaderWarm {

/// Precache file records: compute entries in the disk pipeline cache format, with no recorded
/// reads (the emulator guesses them, see GuessMissing).

/// Precache files use the pipeline cache version: the emulator loads one only when it matches,
/// so the two must move together.
constexpr u32 WarmCacheVersion = VideoCommon::VulkanPipelineCacheVersion;

/// What the builder finds in the game files, recorded in the precache's count file. A file built by
/// an older builder is built again at launch (WarmFilesStale). 2: Mario Kart World's .esetb
/// effects. 3: .ptcl particle effects (Fire Emblem).
constexpr u32 WarmBuilderVersion = 3;

using ComputeKey = std::array<u8, 24>;

/// A program lowered and laid out as LowerNativeCode (video_core/shader_environment.cpp) does,
/// with the same functions (video_core/native_lowering.h).
struct LoweredCode {
    std::vector<u8> code;
    /// Constant buffer the program's bound textures read, ~0U when it names none.
    u32 texture_buffer{~0U};
};

/// Lowers one extracted compute program like `shader-sweep <program> --capture-extent
/// --lower-only --override-out` and builds the runtime code layout: 10 zero header words,
/// instructions with a zero scheduling word before every 4th word, one zero word, the address
/// map, its count and the magic. Returns the error text when the program cannot be lowered.
std::optional<std::string> LowerProgram(std::span<const u8> program, LoweredCode& out);

/// NativeShaderHash of the code
u64 StageHash(std::span<const u8> code);

struct ComputeControl {
    std::array<u32, 3> workgroup{};
    u32 shared_memory{};
    u32 local_memory{};
};

/// Workgroup (control bytes 88, 92, 96), shared memory (byte 100 rounded up to 256) and local
/// memory (byte 104). Empty when the control is too short.
std::optional<ComputeControl> ReadComputeControl(std::span<const u8> control);

/// One compute entry (env count, env, code, compute trailer, key); `key` gets the 24-byte key.
std::vector<u8> ComputeEntry(const LoweredCode& lowered, const ComputeControl& control,
                             ComputeKey& key);

} // namespace VideoCommon::ShaderWarm
