// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>

#include "common/common_types.h"
#include "common/settings.h"
#include "video_core/engines/maxwell_3d.h"

// The Legend of Zelda: Ocarina of Time's anti-aliasing. After tonemapping, the game runs a
// single-frame tensor network on the tonemapped image and an output pass that sharpens the result
// hard. The No sharpening setting replaces the output pass with a copy of the tonemapped image
// and can also skip the update's network passes (docs/nx2/oot-network-bypass.md).
//
// The update and the base game use different native output programs.
namespace VideoCommon::OotNetwork {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;

// Native program keys of the network's compute passes.
constexpr u64 PrepassProgramKey = 0x8475d671e6967428ULL;
constexpr u64 NetworkProgramKey = 0x42593194bb0b2b32ULL;
constexpr u64 OutputProgramKey = 0xda6df3e54c716df2ULL;

// Base v65536 output passes sample cb0+0xc0 and store to cb0+0xd0. Match them
// before translation because the translator may reject their per-tap gathers.
struct BaseOutputProgram {
    u32 first_word;
    u32 size;
    u64 key;
};

constexpr std::array BaseOutputPrograms{
    BaseOutputProgram{0x006b7919, 0x57a0, 0x1e49acb491d9fb59ULL},
    BaseOutputProgram{0x00397919, 0x6430, 0x2142429c43740683ULL},
    BaseOutputProgram{0x003c7919, 0x6690, 0xd01939a775921a6cULL},
};

/// Constant buffer 0 offset of the output pass's texture handle for the B10G11R11 image it writes
/// (the scene image the rest of the frame uses).
constexpr u32 OutputImageHandleOffset = 0xd0;

/// Fragment shaders of the game's tonemap passes; render target 0 of their draws is the
/// network's input.
constexpr bool IsTonemapShader(u64 hash) {
    return hash == 0xbded5ce917183214ULL || // outdoors
           hash == 0x6cb06edf14be1d79ULL || // indoors
           hash == 0x3ac011bd2d772994ULL;   // outdoors at another time of day
}

/// Owned by the rasterizer: the tonemap pass draws on the 3D engine, and the network's passes do
/// not all run on the same compute engine.
struct BypassState {
    std::mutex mutex;
    /// Render target 0 of the latest tonemap draw (address 0 until one is seen), under mutex.
    Tegra::Engines::Maxwell3D::Regs::RenderTargetConfig tonemapped{};
    std::atomic<bool> output_replaced{}; ///< the latest output pass was replaced
};

inline bool BypassEnabled(u64 program_id) {
    return program_id == ProgramId &&
           Settings::values.oot_anti_aliasing.GetValue() == Settings::OotAntiAliasing::NoSharpening;
}

} // namespace VideoCommon::OotNetwork
