// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>

#include "common/assert.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "common/settings.h"
#ifndef _WIN32
#include "common/string_util.h"
#endif

#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/ips_layer.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_cached.h"
#include "core/file_sys/vfs/vfs_layered.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/ns/language.h"
#include "core/hle/service/set/settings_server.h"
#include "core/loader/loader.h"
#include "core/loader/nso.h"
#include "core/memory/cheat_engine.h"

namespace FileSys {
namespace {

constexpr u32 SINGLE_BYTE_MODULUS = 0x100;

constexpr std::array<const char*, 14> EXEFS_FILE_NAMES{
    "main",    "main.npdm", "rtld",    "sdk",     "subsdk0", "subsdk1", "subsdk2",
    "subsdk3", "subsdk4",   "subsdk5", "subsdk6", "subsdk7", "subsdk8", "subsdk9",
};

struct BuiltinPatchRecord {
    size_t module_offset;
    std::span<const u8> bytes;
    std::span<const u8> original;
    bool executable_padding = false;
};

struct BuiltinPatch {
    std::string_view name;
    u64 title_id;
    std::string_view module;
    std::string_view build_id;
    std::span<const BuiltinPatchRecord> records;
    size_t state_offset = 0;
    size_t state_size = 0;
    // Another built-in this one contains; it is skipped once this one is applied.
    std::string_view replaces{};
    // Applies only with NVIDIA DLSS on at this quality in docked mode, and is not listed in
    // Add-Ons.
    std::optional<Settings::NvidiaDlssQuality> dlss_quality{};
};

constexpr std::string_view MAX_RESOLUTION_NAME = "Max Resolution (built-in)";

constexpr std::array<u8, 4> MAX_RESOLUTION_ORIGINAL{0xE8, 0x41, 0x00, 0x36};
constexpr std::array<u8, 4> MAX_RESOLUTION_BRANCH{0x0F, 0x02, 0x00, 0x14};
constexpr std::array MAX_RESOLUTION_RECORDS{
    BuiltinPatchRecord{0x805BD0, MAX_RESOLUTION_BRANCH, MAX_RESOLUTION_ORIGINAL},
};

// ProfessorTee's render scales (#903): Max Resolution's fixed-scale path returns 1.0 of 2560x1440
// (main+0x806420: mov w9, #0x3f800000), which DLSS takes to 4K as Quality. The lower scales give
// the other modes' render sizes.
constexpr std::array<u8, 4> RENDER_SCALE_ONE{0x09, 0xF0, 0xA7, 0x52};         // 1.0
constexpr std::array<u8, 4> RENDER_SCALE_BALANCED{0xE9, 0xEB, 0xA7, 0x52};    // 0.871, 2276x1280
constexpr std::array<u8, 4> RENDER_SCALE_PERFORMANCE{0x09, 0xE8, 0xA7, 0x52}; // 0.75, 1920x1080
constexpr std::array<u8, 4> RENDER_SCALE_ULTRA{0x09, 0xE0, 0xA7, 0x52};       // 0.5, 1280x720
constexpr std::array RENDER_BALANCED_RECORDS{
    BuiltinPatchRecord{0x806420, RENDER_SCALE_BALANCED, RENDER_SCALE_ONE},
};
constexpr std::array RENDER_PERFORMANCE_RECORDS{
    BuiltinPatchRecord{0x806420, RENDER_SCALE_PERFORMANCE, RENDER_SCALE_ONE},
};
constexpr std::array RENDER_ULTRA_RECORDS{
    BuiltinPatchRecord{0x806420, RENDER_SCALE_ULTRA, RENDER_SCALE_ONE},
};

// RsMan-Dev's texture pool patch: stp x8, x9, [sp, #0x10] stores the graphics allocator's
// initial pool size (x8); the call doubles it first (lsl x8, x8, #1 ; stp ; ret).
constexpr std::array<u8, 4> TEXTURE_POOL_ORIGINAL{0xE8, 0x27, 0x01, 0xA9};
constexpr std::array<u8, 4> TEXTURE_POOL_CALL{0xD5, 0x6F, 0xC0, 0x94};
constexpr std::array<u8, 12> TEXTURE_POOL_ROUTINE{
    0x08, 0xF9, 0x7F, 0xD3, 0xE8, 0x27, 0x01, 0xA9, 0xC0, 0x03, 0x5F, 0xD6,
};
constexpr std::array TEXTURE_POOL_RECORDS{
    BuiltinPatchRecord{0xC2A36C, TEXTURE_POOL_CALL, TEXTURE_POOL_ORIGINAL},
    BuiltinPatchRecord{0x3C462C0, TEXTURE_POOL_ROUTINE, {}, true},
};

constexpr std::array<u8, 4> DYNAMIC_FPS_ORIGINAL{0x28, 0x18, 0x20, 0x1E};
constexpr std::array<u8, 4> DYNAMIC_FPS_CALL{0x4F, 0x7D, 0xD5, 0x94};
// Dark's routine up to the step computation; the two built-ins differ only in the tail.
#define DYNAMIC_FPS_ROUTINE_HEAD                                                                      0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91, 0x2B, 0xE0, 0x3B,             0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9, 0xCC, 0x02, 0x00, 0xB4, 0x6B, 0x01,         0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5, 0x4E, 0x01, 0x80, 0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F,         0x01, 0x0E, 0xEB, 0xCB, 0x81, 0x8B, 0x9A, 0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5,         0xEC, 0x03, 0x0B, 0xAA, 0x6E, 0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B, 0x4C, 0x05, 0x00,         0xF9, 0x82, 0x01, 0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E, 0x42, 0x18, 0x23, 0x1E, 0x42, 0x08
// The original routine caps the measured delta at CNTFRQ / 20 (50 ms); the head uses /10
// (100 ms) so Ocarina keeps real speed down to 10 fps.
constexpr std::array<u8, 120> DYNAMIC_FPS_ROUTINE{
    DYNAMIC_FPS_ROUTINE_HEAD,
    // Dark's tail: the smoothed step if it is above original * 1.03, else the original step.
    0x21, 0x1E, 0xC3, 0x00, 0x00, 0x1C, 0x03, 0x09, 0x23, 0x1E, 0x40, 0x20, 0x23, 0x1E, 0x4D,
    0x00, 0x00, 0x54, 0x48, 0x40, 0x20, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A, 0xD7, 0x83, 0x3F,
};
constexpr std::array<u8, 128> FPS_UNLOCK_ROUTINE{
    DYNAMIC_FPS_ROUTINE_HEAD,
    // Unlock tail: the smoothed real-time step in 1/60-s units is clamped into [0.02, 6.0] and
    // always returned, so logic follows real time above 60 fps too (0.5 units/frame at 120 fps)
    // and down to 10 fps (6.0 matches the 100 ms delta cap). At 60 fps it equals Dark's tail.
    // 0x5C: ldr s3, +0x78 (0.02f) ; fmaxnm s2, s2, s3 ; 0x64: ldr s3, +0x7c (6.0f) ;
    // fminnm s2, s2, s3 ; fmov s8, s2 ; ret ; nop ; 0.02f ; 6.0f
    0x21, 0x1E, 0xE3, 0x00, 0x00, 0x1C, 0x42, 0x68, 0x23, 0x1E, 0xC3, 0x00, 0x00, 0x1C, 0x42,
    0x78, 0x23, 0x1E, 0x48, 0x40, 0x20, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x1F, 0x20, 0x03, 0xD5,
    0x0A, 0xD7, 0xA3, 0x3C, 0x00, 0x00, 0xC0, 0x40,
};
#undef DYNAMIC_FPS_ROUTINE_HEAD
// The water ripple simulation runs iterations * (int)step passes per frame, so any step below 1.0
// runs none: the ripples freeze and the grid shifts that follow the player are lost. With these
// the pass count follows the step through an accumulator, with at least one pass while time moves.
constexpr std::array<u8, 4> WATER_STEPS_ORIGINAL{0x09, 0x00, 0x38, 0x1E};
constexpr std::array<u8, 4> WATER_STEPS_CALL{0x42, 0xFB, 0xEF, 0x94}; // bl main+0x3C46300
constexpr std::array<u8, 4> WATER_MUL_ORIGINAL{0x08, 0x7D, 0x09, 0x1B};
constexpr std::array<u8, 4> WATER_MUL_NOP{0x1F, 0x20, 0x03, 0xD5};
// In: w8 iterations, s0 step, x21 the water object. Out: w8 passes. Each water object keeps its own
// remainder: 8 slots of {u64 object, f32 remainder, pad} at state + 0x10 and the next slot to reuse
// at state + 0x90, taken in turn when no slot has the object. Uses x9-x12 and s0-s2, which the game
// doesn't read again before writing them.
constexpr std::array<u8, 148> WATER_STEPS_ROUTINE{
    0x01, 0x01, 0x22, 0x1E, 0x00, 0x08, 0x21, 0x1E, 0x08, 0x20, 0x20, 0x1E, 0x0D, 0x04, 0x00,
    0x54, 0xC9, 0x92, 0x00, 0xB0, 0x29, 0x41, 0x38, 0x91, 0x2A, 0x01, 0x02, 0x91, 0xEB, 0x03,
    0x09, 0xAA, 0x6C, 0x01, 0x40, 0xF9, 0x9F, 0x01, 0x15, 0xEB, 0x60, 0x01, 0x00, 0x54, 0x6B,
    0x41, 0x00, 0x91, 0x7F, 0x01, 0x0A, 0xEB, 0x61, 0xFF, 0xFF, 0x54, 0x4C, 0x01, 0x40, 0xB9,
    0x2B, 0x51, 0x2C, 0x8B, 0x75, 0x01, 0x00, 0xF9, 0x7F, 0x09, 0x00, 0xB9, 0x8C, 0x05, 0x00,
    0x11, 0x8C, 0x09, 0x00, 0x12, 0x4C, 0x01, 0x00, 0xB9, 0x61, 0x09, 0x40, 0xBD, 0x00, 0x28,
    0x21, 0x1E, 0x01, 0x10, 0x2C, 0x1E, 0x02, 0x28, 0x21, 0x1E, 0x48, 0x00, 0x38, 0x1E, 0x1F,
    0x05, 0x00, 0x71, 0x4A, 0x00, 0x00, 0x54, 0x28, 0x00, 0x80, 0x52, 0x01, 0x01, 0x22, 0x1E,
    0x00, 0x38, 0x21, 0x1E, 0x01, 0x10, 0x3C, 0x1E, 0x00, 0x48, 0x21, 0x1E, 0x60, 0x09, 0x00,
    0xBD, 0xC0, 0x03, 0x5F, 0xD6, 0xE8, 0x03, 0x1F, 0x2A, 0xC0, 0x03, 0x5F, 0xD6,
};
// The game's sound engine assumes at least one whole 60 fps step per update: its fade and delay
// timers and parameter ramps count updates from the step it is given each frame. Below a whole
// step (FPS Unlock above 60 fps) some sounds are never released when an area unloads; they pile
// up until the game's voice budget is full, and then most new sounds and the music stop until a
// restart (#716, #1092). So above 60 fps FPS Unlock runs the sound manager job (main+0x415300,
// phase in w2; phase 0 reads the step and updates the engine, phase 1 follows) at 60 Hz: its entry
// goes to SOUND_GATE, which in phase 0 adds the step to an accumulator (timer state + 0xA0) and
// runs the job only once a whole step has built up, else returns; the other phases follow that
// frame's decision (flag at state + 0xA4). At 60 fps and below it runs every frame as before. The
// job's step load (main+0x415334) goes to SOUND_FEED, max(step, 1.0), so the engine always gets
// whole steps and sound still follows real time.
constexpr std::array<u8, 4> SOUND_ENTRY_ORIGINAL{0x7F, 0x23, 0x03, 0xD5}; // pacibsp
constexpr std::array<u8, 4> SOUND_ENTRY_BRANCH{0x40, 0xC5, 0xE0, 0x14};   // b main+0x3C46800
constexpr std::array<u8, 4> SOUND_STEP_ORIGINAL{0x00, 0x11, 0x40, 0xBD};  // ldr s0, [x8, #0x10]
constexpr std::array<u8, 4> SOUND_STEP_CALL{0xD3, 0xC3, 0xE0, 0x94};      // bl main+0x3C46280
// adrp x16, state ; add x16, x16, #0xea0 ; cbnz w2, other ; adrp x9, time ; ldr x9, [x9, #0x890] ;
// ldr x9, [x9] ; ldr s16, [x9, #0x10] ; fmov s17, #1.0 ; fcmp s16, s17 ; b.ge every_frame ;
// ldr s18, [x16] ; fadd s18, s18, s16 ; fcmp s18, s17 ; b.lt wait ; fsub s18, s18, s17 ;
// str s18, [x16] ; mov w9, #1 ; str w9, [x16, #4] ; b run ;
// wait: str s18, [x16] ; str wzr, [x16, #4] ; ret ;
// every_frame: nop (a remainder built up below a whole step carries over) ; mov w9, #1 ;
// str w9, [x16, #4] ; b run ;
// other: ldr w9, [x16, #4] ; cbz w9, skip ; run: pacibsp ; b main+0x415304 ; skip: ret
// x9, x16 and s16-s18 are free at the job's entry.
constexpr std::array<u8, 124> SOUND_GATE{
    0xD0, 0x92, 0x00, 0xB0, 0x10, 0x82, 0x3A, 0x91, 0x02, 0x03, 0x00, 0x35, 0xC9, 0x81, 0x00, 0x90,
    0x29, 0x49, 0x44, 0xF9, 0x29, 0x01, 0x40, 0xF9, 0x30, 0x11, 0x40, 0xBD, 0x11, 0x10, 0x2E, 0x1E,
    0x00, 0x22, 0x31, 0x1E, 0xAA, 0x01, 0x00, 0x54, 0x12, 0x02, 0x40, 0xBD, 0x52, 0x2A, 0x30, 0x1E,
    0x40, 0x22, 0x31, 0x1E, 0xCB, 0x00, 0x00, 0x54, 0x52, 0x3A, 0x31, 0x1E, 0x12, 0x02, 0x00, 0xBD,
    0x29, 0x00, 0x80, 0x52, 0x09, 0x06, 0x00, 0xB9, 0x0A, 0x00, 0x00, 0x14, 0x12, 0x02, 0x00, 0xBD,
    0x1F, 0x06, 0x00, 0xB9, 0xC0, 0x03, 0x5F, 0xD6, 0x1F, 0x20, 0x03, 0xD5, 0x29, 0x00, 0x80, 0x52,
    0x09, 0x06, 0x00, 0xB9, 0x03, 0x00, 0x00, 0x14, 0x09, 0x06, 0x40, 0xB9, 0x69, 0x00, 0x00, 0x34,
    0x7F, 0x23, 0x03, 0xD5, 0xA4, 0x3A, 0x1F, 0x17, 0xC0, 0x03, 0x5F, 0xD6,
};
// ldr s0, [x8, #0x10] ; fmov s16, #1.0 ; fmaxnm s0, s0, s16 ; ret
constexpr std::array<u8, 16> SOUND_FEED{
    0x00, 0x11, 0x40, 0xBD, 0x10, 0x10, 0x2E, 0x1E, 0x00, 0x68, 0x30, 0x1E, 0xC0, 0x03, 0x5F, 0xD6,
};
constexpr std::array DYNAMIC_FPS_RECORDS{
    BuiltinPatchRecord{0x6E6CC4, DYNAMIC_FPS_CALL, DYNAMIC_FPS_ORIGINAL},
    BuiltinPatchRecord{0x3C46200, DYNAMIC_FPS_ROUTINE, {}, true},
    BuiltinPatchRecord{0x475F8, WATER_STEPS_CALL, WATER_STEPS_ORIGINAL},
    BuiltinPatchRecord{0x4760C, WATER_MUL_NOP, WATER_MUL_ORIGINAL},
    BuiltinPatchRecord{0x3C46300, WATER_STEPS_ROUTINE, {}, true},
};

// Truncated steps stall the pickup window count-up and ocarina playback tick above 60 fps.
// The step rate manager counts down by max(round(step), 1) each frame and runs fast;
// the ambient wave phase counts frames. Accumulate fractional steps per counter instead.
// Helper: x13 = counter address, s0 = step; w13 = ticks. Sixteen slots keep one remainder
// per counter; new or evicted slots return max(trunc(step), 1). Uses x13-x16 and NZCV,
// preserves q0/q1; stubs save x13-x16 and LR. Hitstop replaces s0, ambient reloads it.
constexpr std::array<u8, 4> STEP_HUD_ORIGINAL{
    0x0C, 0x00, 0x38, 0x1E,
};
constexpr std::array<u8, 4> STEP_HUD_HOOK{
    0x80, 0x79, 0xC7, 0x14,
};
// add x13, x19, #0x360 ; bl helper ; mov w12, w13 ; b 0xA68304
constexpr std::array<u8, 40> STEP_HUD_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x6D, 0x82,
    0x0D, 0x91, 0x1C, 0x00, 0x00, 0x94, 0xEC, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0x78, 0x86, 0x38, 0x17,
};
constexpr std::array<u8, 4> STEP_REPLAY_ORIGINAL{
    0x08, 0x00, 0x39, 0x1E,
};
constexpr std::array<u8, 4> STEP_REPLAY_HOOK{
    0x46, 0xCD, 0x26, 0x14,
};
// add x13, x19, x20 ; bl helper ; mov w8, w13 ; b 0x3293414
constexpr std::array<u8, 40> STEP_REPLAY_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x6D, 0x02,
    0x14, 0x8B, 0x12, 0x00, 0x00, 0x94, 0xE8, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0xB2, 0x32, 0xD9, 0x17,
};
constexpr std::array<u8, 44> STEP_HITSTOP_ORIGINAL{
    0x00, 0xB9, 0xA1, 0x5E, 0x09, 0x01, 0x38, 0x1E, 0x00, 0xD8, 0x21, 0x5E, 0x00, 0x39,
    0x20, 0x1E, 0x00, 0x20, 0x30, 0x1E, 0xAA, 0x17, 0x00, 0x54, 0x00, 0x20, 0x31, 0x1E,
    0x09, 0x18, 0x00, 0x54, 0x0A, 0x9B, 0x40, 0xB9, 0x3F, 0x05, 0x00, 0x71, 0x29, 0xC5,
    0x9F, 0x1A,
};
constexpr std::array<u8, 44> STEP_HITSTOP_HOOK{
    0xBA, 0xCD, 0x33, 0x14, 0x09, 0x01, 0x38, 0x1E, 0x00, 0xD8, 0x21, 0x5E, 0x00, 0x39,
    0x20, 0x1E, 0x00, 0x20, 0x30, 0x1E, 0xAA, 0x17, 0x00, 0x54, 0x00, 0x20, 0x31, 0x1E,
    0x09, 0x18, 0x00, 0x54, 0x0A, 0x9B, 0x40, 0xB9, 0x3F, 0x05, 0x00, 0x71, 0x29, 0xC5,
    0x9F, 0x1A,
};
// add x13, x24, #0x98 ; fmov s0, s8 ; bl helper ; mov w9, w13 ; ldr w10, [x24, #0x98] ; b 0x2F53294
constexpr std::array<u8, 48> STEP_HITSTOP_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x0D, 0x63,
    0x02, 0x91, 0x00, 0x41, 0x20, 0x1E, 0x07, 0x00, 0x00, 0x94, 0xE9, 0x03, 0x0D, 0x2A,
    0xFE, 0x13, 0x40, 0xF9, 0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0x0A, 0x9B,
    0x40, 0xB9, 0x46, 0x32, 0xCC, 0x17,
};
constexpr std::array<u8, 128> STEP_SHARED_ROUTINE{
    0xE0, 0x07, 0xBF, 0xAD, 0xCE, 0x92, 0x00, 0xB0, 0xCE, 0x01, 0x3B, 0x91, 0x0F, 0x02,
    0x80, 0x52, 0xD0, 0x01, 0x40, 0xF9, 0x1F, 0x02, 0x0D, 0xEB, 0x40, 0x02, 0x00, 0x54,
    0x70, 0x01, 0x00, 0xB4, 0xCE, 0x41, 0x00, 0x91, 0xEF, 0x05, 0x00, 0x71, 0x41, 0xFF,
    0xFF, 0x54, 0xCE, 0x41, 0x04, 0xD1, 0xD0, 0x01, 0x40, 0xB9, 0x0F, 0x06, 0x00, 0x11,
    0xEF, 0x0D, 0x00, 0x12, 0xCF, 0x01, 0x00, 0xB9, 0xCE, 0x41, 0x00, 0x91, 0xCE, 0x11,
    0x10, 0x8B, 0xCD, 0x01, 0x00, 0xF9, 0xDF, 0x09, 0x00, 0xB9, 0x0D, 0x00, 0x38, 0x1E,
    0xBF, 0x05, 0x00, 0x71, 0xAD, 0xA5, 0x9F, 0x1A, 0x07, 0x00, 0x00, 0x14, 0xC1, 0x09,
    0x40, 0xBD, 0x00, 0x28, 0x21, 0x1E, 0x01, 0xC0, 0x25, 0x1E, 0x00, 0x38, 0x21, 0x1E,
    0xC0, 0x09, 0x00, 0xBD, 0x2D, 0x00, 0x38, 0x1E, 0xE0, 0x07, 0xC1, 0xAC, 0xC0, 0x03,
    0x5F, 0xD6,
};
constexpr std::array<u8, 4> STEP_AMBIENT_ORIGINAL{
    0x7A, 0x07, 0x00, 0x11,
};
constexpr std::array<u8, 4> STEP_AMBIENT_HOOK{
    0x3B, 0x1C, 0xDC, 0x14,
};
// add x13, x28, x1, lsl #2 ; load raw step ; bl helper ; add w26, w27, w13 ; ldr s0, [x0, #0x18] ; b 0x53F918
constexpr std::array<u8, 60> STEP_AMBIENT_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x8D, 0x0B,
    0x01, 0x8B, 0xCE, 0x81, 0x00, 0x90, 0xCE, 0x49, 0x44, 0xF9, 0xCE, 0x01, 0x40, 0xF9,
    0xC0, 0x11, 0x40, 0xBD, 0xD8, 0xFF, 0xFF, 0x97, 0x7A, 0x03, 0x0D, 0x0B, 0x00, 0x18,
    0x40, 0xBD, 0xFE, 0x13, 0x40, 0xF9, 0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8,
    0xB8, 0xE3, 0x23, 0x17,
};

// The character controller takes a tenth of its ground's velocity once the frame rate is over
// 100 (b.gt after fcmp with 100.0, in three places), so a moving platform leaves the player behind.
constexpr std::array<u8, 4> GROUND_VELOCITY_ORIGINAL_A{
    0x6C, 0x19, 0x00, 0x54,
};
constexpr std::array<u8, 4> GROUND_VELOCITY_ORIGINAL_B{
    0x2C, 0x1E, 0x00, 0x54,
};
constexpr std::array<u8, 4> GROUND_VELOCITY_ORIGINAL_C{
    0xCC, 0x17, 0x00, 0x54,
};
constexpr std::array<u8, 4> GROUND_VELOCITY_NOP{
    0x1F, 0x20, 0x03, 0xD5,
};
// A turn action (main+0x261E8E0, ExecuteASApproachStickX; Gohma steers with it) ends by clamping
// its turn speed, kept per 60 Hz frame, to the change it applied times the frame step. The change
// already carries the step, so below a step of 1 the speed decays by step^2 every frame. The routine
// divides by the step there instead and runs the game's two instructions at a step of 1 and above.
// The routine replaces both fmul (0x261EB18 and 0x261EB1C) and returns to 0x261EB20, so the hook
// guards both; the second instruction stays in place, unreached.
constexpr std::array<u8, 8> TURN_SPEED_ORIGINAL{0x01, 0x08, 0x21, 0x1E, 0x00, 0x08, 0x28, 0x1E};
constexpr std::array<u8, 8> TURN_SPEED_HOOK{0xCA, 0x9F, 0x58, 0x14, 0x00, 0x08, 0x28, 0x1E};
// fmov s1, #1.0 ; fcmp s8, s1 ; b.ge +0x10 ; fdiv s0, s0, s8 ; fneg s1, s0 ; b 0x261EB20 ;
// ldr s1, [sp, #0xc] ; fmul s1, s0, s1 ; fmul s0, s0, s8 ; b 0x261EB20
constexpr std::array<u8, 40> TURN_SPEED_ROUTINE{
    0x01, 0x10, 0x2E, 0x1E, 0x00, 0x21, 0x21, 0x1E, 0x8A, 0x00, 0x00, 0x54, 0x00, 0x18,
    0x28, 0x1E, 0x01, 0x40, 0x21, 0x1E, 0x33, 0x60, 0xA7, 0x17, 0xE1, 0x0F, 0x40, 0xBD,
    0x01, 0x08, 0x21, 0x1E, 0x00, 0x08, 0x28, 0x1E, 0x2F, 0x60, 0xA7, 0x17,
};
// A movie player counts its updates with a picture shown (+0x2F8, one more each), and a cutscene
// that plays a movie takes that count as its timeline frame: its voices, overlays and its end are
// placed by it. The picture and the sound follow the clock, so at any other frame rate than 60 the
// cues drift from them (sages' lines over each other at 30 or 120 fps, #1474, #1572). The count
// takes whole ticks of the step through the shared helper instead, keyed by the counter's address.
// Same site as Hookshot's movie frame count (issue 1742).
constexpr std::array<u8, 4> MOVIE_FRAMES_ORIGINAL{0x08, 0x05, 0x00, 0x11}; // add w8, w8, #1
constexpr std::array<u8, 4> MOVIE_FRAMES_HOOK{0xAA, 0xA8, 0x03, 0x14};     // b main+0x3C46A70
// add x13, x20, #0x2f8 ; load raw step ; bl helper ; add w8, w8, w13 ; b 0x3B5C7CC
constexpr std::array<u8, 56> MOVIE_FRAMES_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x8D, 0xE2,
    0x0B, 0x91, 0xCE, 0x81, 0x00, 0x90, 0xCE, 0x49, 0x44, 0xF9, 0xCE, 0x01, 0x40, 0xF9,
    0xC0, 0x11, 0x40, 0xBD, 0xBC, 0xFF, 0xFF, 0x97, 0x08, 0x01, 0x0D, 0x0B, 0xFE, 0x13,
    0x40, 0xF9, 0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0x4A, 0x57, 0xFC, 0x17,
};
// Scarecrow's Song recording keeps each note's and gap's length in the elapsed counter that
// playback (0x3293410 above) advances, and adds trunc(step) to it in two more places: while a note
// or gap is held (0x3296B08) and when the last note is timed (0x32926C8). With FPS Unlock the step
// sits around 1.0 at 60 fps and below it above 60, so the cut is often 0: lengths come out short or
// zero, and the song stops after its first note (#1730). Both take whole ticks for that counter
// from the shared helper instead. Hookshot found the two sites (f_832b6840 and f_832b2690).
constexpr std::array<u8, 4> SCARECROW_CUT_ORIGINAL{0x08, 0x00, 0x39, 0x1E}; // fcvtzu w8, s0
constexpr std::array<u8, 4> SCARECROW_RECORD_HOOK{0xEA, 0xBF, 0x26, 0x14};  // b main+0x3C46AB0
constexpr std::array<u8, 4> SCARECROW_LAST_HOOK{0x04, 0xD1, 0x26, 0x14};    // b main+0x3C46AD8
// mov x13, x9 ; bl helper ; mov w8, w13 ; b 0x3296B0C
constexpr std::array<u8, 40> SCARECROW_RECORD_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0xED, 0x03,
    0x09, 0xAA, 0xB0, 0xFF, 0xFF, 0x97, 0xE8, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0x0E, 0x40, 0xD9, 0x17,
};
// add x13, x20, #0x5fc ; bl helper ; mov w8, w13 ; b 0x32926CC
constexpr std::array<u8, 40> SCARECROW_LAST_ROUTINE{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x8D, 0xF2,
    0x17, 0x91, 0xA6, 0xFF, 0xFF, 0x97, 0xE8, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0xF4, 0x2E, 0xD9, 0x17,
};
constexpr std::array FPS_UNLOCK_RECORDS{
    BuiltinPatchRecord{0x6E6CC4, DYNAMIC_FPS_CALL, DYNAMIC_FPS_ORIGINAL},
    BuiltinPatchRecord{0x3C46200, FPS_UNLOCK_ROUTINE, {}, true},
    BuiltinPatchRecord{0x475F8, WATER_STEPS_CALL, WATER_STEPS_ORIGINAL},
    BuiltinPatchRecord{0x4760C, WATER_MUL_NOP, WATER_MUL_ORIGINAL},
    BuiltinPatchRecord{0x3C46300, WATER_STEPS_ROUTINE, {}, true},
    BuiltinPatchRecord{0x415300, SOUND_ENTRY_BRANCH, SOUND_ENTRY_ORIGINAL},
    BuiltinPatchRecord{0x415334, SOUND_STEP_CALL, SOUND_STEP_ORIGINAL},
    BuiltinPatchRecord{0x3C46800, SOUND_GATE, {}, true},
    BuiltinPatchRecord{0x3C46280, SOUND_FEED, {}, true},
    BuiltinPatchRecord{0xA68300, STEP_HUD_HOOK, STEP_HUD_ORIGINAL},
    BuiltinPatchRecord{0x3C46900, STEP_HUD_ROUTINE, {}, true},
    BuiltinPatchRecord{0x3293410, STEP_REPLAY_HOOK, STEP_REPLAY_ORIGINAL},
    BuiltinPatchRecord{0x3C46928, STEP_REPLAY_ROUTINE, {}, true},
    BuiltinPatchRecord{0x2F53268, STEP_HITSTOP_HOOK, STEP_HITSTOP_ORIGINAL},
    BuiltinPatchRecord{0x3C46950, STEP_HITSTOP_ROUTINE, {}, true},
    BuiltinPatchRecord{0x53F914, STEP_AMBIENT_HOOK, STEP_AMBIENT_ORIGINAL},
    BuiltinPatchRecord{0x3C46A00, STEP_AMBIENT_ROUTINE, {}, true},
    BuiltinPatchRecord{0x3C46980, STEP_SHARED_ROUTINE, {}, true},
    BuiltinPatchRecord{0x13A070, GROUND_VELOCITY_NOP, GROUND_VELOCITY_ORIGINAL_A},
    BuiltinPatchRecord{0x13D710, GROUND_VELOCITY_NOP, GROUND_VELOCITY_ORIGINAL_B},
    BuiltinPatchRecord{0x389AC2C, GROUND_VELOCITY_NOP, GROUND_VELOCITY_ORIGINAL_C},
    BuiltinPatchRecord{0x261EB18, TURN_SPEED_HOOK, TURN_SPEED_ORIGINAL},
    BuiltinPatchRecord{0x3C46A40, TURN_SPEED_ROUTINE, {}, true},
    BuiltinPatchRecord{0x3B5C7C8, MOVIE_FRAMES_HOOK, MOVIE_FRAMES_ORIGINAL},
    BuiltinPatchRecord{0x3C46A70, MOVIE_FRAMES_ROUTINE, {}, true},
    BuiltinPatchRecord{0x3296B08, SCARECROW_RECORD_HOOK, SCARECROW_CUT_ORIGINAL},
    BuiltinPatchRecord{0x3C46AB0, SCARECROW_RECORD_ROUTINE, {}, true},
    BuiltinPatchRecord{0x32926C8, SCARECROW_LAST_HOOK, SCARECROW_CUT_ORIGINAL},
    BuiltinPatchRecord{0x3C46AD8, SCARECROW_LAST_ROUTINE, {}, true},
};

constexpr std::array BUILTIN_PATCHES{
    // Ocarina of Time takes its "dynamic resolution disabled" path, so the 3D renders at full scale
    // instead of following the GPU-time controller down to its 480p floor.
    BuiltinPatch{
        .name = MAX_RESOLUTION_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = MAX_RESOLUTION_RECORDS,
    },
    // With Max Resolution the guard house outgrows the initial pool and dereferences a null
    // allocation.
    BuiltinPatch{
        .name = "Texture Pool x2 (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = TEXTURE_POOL_RECORDS,
    },
    // Dark's routine with a tail that also lowers the step above 60 fps, so the game keeps real
    // speed at any frame rate limit (Settings frame_rate_limit).
    BuiltinPatch{
        .name = "FPS Unlock (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = FPS_UNLOCK_RECORDS,
        .state_offset = 0x4E9FE00,
        // Timer state at 0, the water slots to 0x98, the sound gate at 0xA0, the step counters
        // from 0xB0.
        .state_size = 0x1C0,
        .replaces = "Dynamic FPS by Dark (built-in)",
    },
    BuiltinPatch{
        .name = "Dynamic FPS by Dark (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = DYNAMIC_FPS_RECORDS,
        .state_offset = 0x4E9FE00,
        .state_size = 0x98,
    },
    // The NVIDIA DLSS quality lowers Ocarina's render resolution under Max Resolution, since DLSS
    // replaces the game's own 1440p -> 4K upscale and takes whatever the game renders.
    BuiltinPatch{
        .name = "DLSS Balanced render scale (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_BALANCED_RECORDS,
        .dlss_quality = Settings::NvidiaDlssQuality::Balanced,
    },
    BuiltinPatch{
        .name = "DLSS Performance render scale (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_PERFORMANCE_RECORDS,
        .dlss_quality = Settings::NvidiaDlssQuality::Performance,
    },
    BuiltinPatch{
        .name = "DLSS Ultra Performance render scale (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_ULTRA_RECORDS,
        .dlss_quality = Settings::NvidiaDlssQuality::UltraPerformance,
    },
};

enum class TitleVersionFormat : u8 {
    ThreeElements, ///< vX.Y.Z
    FourElements,  ///< vX.Y.Z.W
};

std::string FormatTitleVersion(u32 version,
                               TitleVersionFormat format = TitleVersionFormat::ThreeElements) {
    std::array<u8, sizeof(u32)> bytes{};
    bytes[0] = static_cast<u8>(version % SINGLE_BYTE_MODULUS);
    for (std::size_t i = 1; i < bytes.size(); ++i) {
        version /= SINGLE_BYTE_MODULUS;
        bytes[i] = static_cast<u8>(version % SINGLE_BYTE_MODULUS);
    }

    if (format == TitleVersionFormat::FourElements) {
        return fmt::format("v{}.{}.{}.{}", bytes[3], bytes[2], bytes[1], bytes[0]);
    }
    return fmt::format("v{}.{}.{}", bytes[3], bytes[2], bytes[1]);
}

// Returns a directory with name matching name case-insensitive. Returns nullptr if directory
// doesn't have a directory with name.
VirtualDir FindSubdirectoryCaseless(const VirtualDir dir, std::string_view name) {
#ifdef _WIN32
    return dir->GetSubdirectory(name);
#else
    const auto subdirs = dir->GetSubdirectories();
    for (const auto& subdir : subdirs) {
        std::string dir_name = Common::ToLower(subdir->GetName());
        if (dir_name == name) {
            return subdir;
        }
    }

    return nullptr;
#endif
}

std::optional<std::vector<Core::Memory::CheatEntry>> ReadCheatFileFromFolder(
    u64 title_id, const PatchManager::BuildID& build_id_, const VirtualDir& base_path, bool upper) {
    const auto build_id_raw = Common::HexToString(build_id_, upper);
    const auto build_id = build_id_raw.substr(0, sizeof(u64) * 2);
    const auto file = base_path->GetFile(fmt::format("{}.txt", build_id));

    if (file == nullptr) {
        LOG_INFO(Common_Filesystem, "No cheats file found for title_id={:016X}, build_id={}",
                 title_id, build_id);
        return std::nullopt;
    }

    std::vector<u8> data(file->GetSize());
    if (file->Read(data.data(), data.size()) != data.size()) {
        LOG_INFO(Common_Filesystem, "Failed to read cheats file for title_id={:016X}, build_id={}",
                 title_id, build_id);
        return std::nullopt;
    }

    const Core::Memory::TextCheatParser parser;
    return parser.Parse(std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
}

void AppendCommaIfNotEmpty(std::string& to, std::string_view with) {
    if (to.empty()) {
        to += with;
    } else {
        to += ", ";
        to += with;
    }
}

bool IsDirValidAndNonEmpty(const VirtualDir& dir) {
    return dir != nullptr && (!dir->GetFiles().empty() || !dir->GetSubdirectories().empty());
}

bool IsVersionedExternalUpdateDisabled(const std::vector<std::string>& disabled, u32 version) {
    const std::string disabled_key = fmt::format("Update@{}", version);
    return std::find(disabled.cbegin(), disabled.cend(), disabled_key) != disabled.cend() ||
           std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();
}

std::string GetUpdateVersionStringFromSlot(const ContentProvider* provider, u64 update_tid) {
    if (provider == nullptr) {
        return {};
    }

    auto control_nca = provider->GetEntry(update_tid, ContentRecordType::Control);
    if (control_nca == nullptr ||
        control_nca->GetStatus() != Loader::ResultStatus::Success) {
        return {};
    }

    const auto romfs = control_nca->GetRomFS();
    if (romfs == nullptr) {
        return {};
    }

    const auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return {};
    }

    auto nacp_file = extracted->GetFile("control.nacp");
    if (nacp_file == nullptr) {
        nacp_file = extracted->GetFile("Control.nacp");
    }
    if (nacp_file == nullptr) {
        return {};
    }

    NACP nacp{nacp_file};
    return nacp.GetVersionString();
}
LICHII_NO_INLINE std::optional<u64> GetParentApplicationId(const ContentProvider* provider, u64 title_id) {
    return provider == nullptr ? std::nullopt : provider->GetParentApplicationId(title_id);
}
} // Anonymous namespace

PatchManager::PatchManager(u64 title_id_,
                           const Service::FileSystem::FileSystemController& fs_controller_,
                           const ContentProvider& content_provider_)
    : title_id{title_id_}, parent_title_id{GetParentApplicationId(std::addressof(content_provider_), title_id_)}, fs_controller{fs_controller_}, content_provider{content_provider_} {}

PatchManager::~PatchManager() = default;

u64 PatchManager::GetTitleID() const {
    return title_id;
}

VirtualDir PatchManager::GetModificationLoadRoot(bool sdmc) const {
    const auto get_root = [&](u64 id) {
        return sdmc ? fs_controller.GetSDMCModificationLoadRoot(id) : fs_controller.GetModificationLoadRoot(id);
    };
    auto root = get_root(title_id);
    if (!parent_title_id) {
        return root;
    }
    std::vector<VirtualDir> roots{std::move(root), get_root(*parent_title_id)};
    std::erase(roots, nullptr);
    return LayeredVfsDirectory::MakeLayeredDirectory(std::move(roots));
}

std::vector<std::string> PatchManager::GetDisabledAddons() const {
    auto disabled = Settings::values.disabled_addons[title_id];
    if (parent_title_id) {
        const auto& shared = Settings::values.disabled_addons[*parent_title_id];
        disabled.insert(disabled.end(), shared.begin(), shared.end());
    }
    return disabled;
}

VirtualDir PatchManager::PatchExeFS(VirtualDir exefs, bool apply_update) const {
    LOG_INFO(Loader, "Patching ExeFS for title_id={:016X}", title_id);

    if (exefs == nullptr)
        return exefs;

    const auto disabled = GetDisabledAddons();

    bool update_disabled = true;
    std::optional<u32> enabled_version;
    bool checked_external = false;
    bool checked_manual = false;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);
    const auto update_tid = GetUpdateTitleID(title_id);

    if (content_union) {
        // First, check ExternalContentProvider
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            if (!update_versions.empty()) {
                checked_external = true;
                for (const auto& update_entry : update_versions) {
                    if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                        update_disabled = false;
                        enabled_version = update_entry.version;
                        break;
                    }
                }
            }
        }

        // Also check ManualContentProvider (for Android)
        if (!checked_external) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

                if (!manual_update_versions.empty()) {
                    checked_manual = true;
                    for (const auto& update_entry : manual_update_versions) {
                        if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                            update_disabled = false;
                            enabled_version = update_entry.version;
                            break;
                        }
                    }
                }
            }
        }
    }

    // check for original NAND style
    // Check NAND if: no external updates exist, OR all external updates are disabled
    if (!checked_external && !checked_manual) {
        // Only enable NAND update if it exists AND is not disabled
        // We need to check if an update actually exists in the content provider
        const bool has_nand_update = content_provider.HasEntry(update_tid, ContentRecordType::Program);

        if (has_nand_update) {
            const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
            const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();
            const bool generic_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();

            if (!nand_disabled && !sdmc_disabled && !generic_disabled) {
                update_disabled = false;
            }
        }
    } else if (update_disabled && content_union) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();

        if (!nand_disabled || !sdmc_disabled) {
            const auto nand_sdmc_entries = content_union->ListEntriesFilterOrigin(
                std::nullopt, TitleType::Update, ContentRecordType::Program, update_tid);

            for (const auto& [slot, entry] : nand_sdmc_entries) {
                if (slot == ContentProviderUnionSlot::UserNAND ||
                    slot == ContentProviderUnionSlot::SysNAND) {
                    if (!nand_disabled) {
                        update_disabled = false;
                        break;
                    }
                } else if (slot == ContentProviderUnionSlot::SDMC) {
                    if (!sdmc_disabled) {
                        update_disabled = false;
                        break;
                    }
                }
            }
        }
    }

    // Game Updates
    std::unique_ptr<NCA> update = nullptr;

    // If we have a specific enabled version from external provider, use it
    if (enabled_version.has_value() && content_union) {
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            auto file = external_provider->GetEntryForVersion(update_tid, ContentRecordType::Program, *enabled_version);
            if (file != nullptr) {
                update = std::make_unique<NCA>(file);
            }
        }

        // Also try ManualContentProvider
        if (update == nullptr) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                auto file = manual_provider->GetEntryForVersion(update_tid, ContentRecordType::Program, *enabled_version);
                if (file != nullptr) {
                    update = std::make_unique<NCA>(file);
                }
            }
        }
    }

    // Fallback to regular content provider if no external update was loaded
    if (update == nullptr && !update_disabled) {
        update = content_provider.GetEntry(update_tid, ContentRecordType::Program);
    }

    if (!update_disabled && update != nullptr && update->GetExeFS() != nullptr) {
        const auto version =
            FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0));
        if (apply_update) {
            LOG_INFO(Loader, "    ExeFS: Update ({}) applied successfully", version);
            exefs = update->GetExeFS();
        } else {
            LOG_WARNING(Loader,
                        "    ExeFS: Ignoring installed update ({}) for {:016X}: the container "
                        "carries its own program and RomFS",
                        version, title_id);
        }
    }

    // LayeredExeFS
    const auto load_dir = GetModificationLoadRoot();
    const auto sdmc_load_dir = GetModificationLoadRoot(true);

    std::vector<VirtualDir> patch_dirs;
    if (sdmc_load_dir != nullptr) {
        patch_dirs.push_back(sdmc_load_dir);
    }
    if (load_dir != nullptr) {
        const auto load_patch_dirs = load_dir->GetSubdirectories();
        patch_dirs.insert(patch_dirs.end(), load_patch_dirs.begin(), load_patch_dirs.end());
    }

    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    std::vector<VirtualDir> layers;
    layers.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.begin(), disabled.end(), subdir->GetName()) != disabled.end())
            continue;

        auto exefs_dir = FindSubdirectoryCaseless(subdir, "exefs");
        if (exefs_dir != nullptr) {
            LOG_INFO(Loader, "    ExeFS: LayeredExeFS mod {} (exefs)", subdir->GetName());
            layers.push_back(std::move(exefs_dir));
        }
    }
    layers.push_back(exefs);

    auto layered = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers));
    if (layered != nullptr) {
        LOG_INFO(Loader, "    ExeFS: LayeredExeFS patches applied successfully");
        exefs = std::move(layered);
    }

    if (Settings::values.dump_exefs) {
        LOG_INFO(Loader, "Dumping ExeFS for title_id={:016X}", title_id);
        const auto dump_dir = fs_controller.GetModificationDumpRoot(title_id);
        if (dump_dir != nullptr) {
            const auto exefs_dir = GetOrCreateDirectoryRelative(dump_dir, "/exefs");
            VfsRawCopyD(exefs, exefs_dir);
        }
    }

    return exefs;
}

std::vector<VirtualFile> PatchManager::CollectPatches(const std::vector<VirtualDir>& patch_dirs, const std::string& build_id) const {
    const auto disabled = GetDisabledAddons();
    const auto nso_build_id = fmt::format("{:0<64}", build_id);

    std::vector<VirtualFile> out;
    out.reserve(patch_dirs.size());
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend())
            continue;

        auto exefs_dir = FindSubdirectoryCaseless(subdir, "exefs");
        if (exefs_dir != nullptr) {
            for (const auto& file : exefs_dir->GetFiles()) {
                if (file->GetExtension() == "ips") {
                    auto name = file->GetName();
                    const auto this_build_id = fmt::format("{:0<64}", name.substr(0, name.find('.')));
                    if (nso_build_id == this_build_id)
                        out.push_back(file);
                } else if (file->GetExtension() == "pchtxt") {
                    IPSwitchCompiler compiler{file};
                    const auto this_build_id = Common::HexToString(compiler.GetBuildID());
                    if (nso_build_id == this_build_id)
                        out.push_back(file);
                }
            }
        }
    }
    return out;
}

std::vector<u8> PatchManager::PatchNSO(const std::vector<u8>& nso, const std::string& name) const {
    if (name == "main") {
        Settings::SetAppliedOcarinaRenderScale(std::nullopt);
    }
    if (nso.size() < sizeof(Loader::NSOHeader)) {
        return nso;
    }

    Loader::NSOHeader header;
    std::memcpy(&header, nso.data(), sizeof(header));

    if (header.magic != Common::MakeMagic('N', 'S', 'O', '0')) {
        return nso;
    }

    const auto build_id_raw = Common::HexToString(header.build_id);
    const auto build_id = build_id_raw.substr(0, build_id_raw.find_last_not_of('0') + 1);

    if (Settings::values.dump_nso) {
        LOG_INFO(Loader, "Dumping NSO for name={}, build_id={}, title_id={:016X}", name, build_id,
                 title_id);
        const auto dump_dir = fs_controller.GetModificationDumpRoot(title_id);
        if (dump_dir != nullptr) {
            const auto nso_dir = GetOrCreateDirectoryRelative(dump_dir, "/nso");
            const auto file = nso_dir->CreateFile(fmt::format("{}-{}.nso", name, build_id));

            file->Resize(nso.size());
            file->WriteBytes(nso);
        }
    }

    LOG_INFO(Loader, "Patching NSO for name={}, build_id={}", name, build_id);

    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return nso;
    }

    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });
    const auto patches = CollectPatches(patch_dirs, build_id);

    auto out = nso;
    for (const auto& patch_file : patches) {
        if (patch_file->GetExtension() == "ips") {
            LOG_INFO(Loader, "    - Applying IPS patch from mod \"{}\"",
                     patch_file->GetContainingDirectory()->GetParentDirectory()->GetName());
            const auto patched = PatchIPS(std::make_shared<VectorVfsFile>(out), patch_file);
            if (patched != nullptr)
                out = patched->ReadAllBytes();
        } else if (patch_file->GetExtension() == "pchtxt") {
            LOG_INFO(Loader, "    - Applying IPSwitch patch from mod \"{}\"",
                     patch_file->GetContainingDirectory()->GetParentDirectory()->GetName());
            const IPSwitchCompiler compiler{patch_file};
            const auto patched = compiler.Apply(std::make_shared<VectorVfsFile>(out));
            if (patched != nullptr)
                out = patched->ReadAllBytes();
        }
    }

    const auto disabled = GetDisabledAddons();
    std::vector<std::string_view> replaced;
    for (const auto& patch : BUILTIN_PATCHES) {
        if (patch.title_id != title_id || patch.module != name) {
            continue;
        }
        if (std::find(disabled.cbegin(), disabled.cend(), patch.name) != disabled.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": disabled in Add-Ons", patch.name);
            continue;
        }
        if (std::find(replaced.cbegin(), replaced.cend(), patch.name) != replaced.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": included in an applied built-in patch",
                     patch.name);
            continue;
        }
        if (patch.dlss_quality) {
#ifdef EGLANTINE_STREAMLINE
            const bool wanted = Settings::OcarinaDlssRenderScale() == patch.dlss_quality &&
                                std::find(disabled.cbegin(), disabled.cend(),
                                          MAX_RESOLUTION_NAME) == disabled.cend();
#else
            constexpr bool wanted = false;
#endif
            if (!wanted) {
                continue;
            }
        }
        if (patch.build_id != build_id) {
            LOG_INFO(Loader, "    - Skipping \"{}\": build id {} is not {}", patch.name, build_id,
                     patch.build_id);
            continue;
        }
        const auto fits = [](const std::vector<u8>& data, size_t offset, size_t size) {
            return data.size() >= sizeof(Loader::NSOHeader) &&
                   offset <= data.size() - sizeof(Loader::NSOHeader) &&
                   size <= data.size() - sizeof(Loader::NSOHeader) - offset;
        };
        if (patch.state_size != 0) {
            const auto& data = header.segments[2];
            const u64 bss_start = u64{data.location} + data.size;
            const u64 bss_end = bss_start + data.bss_size;
            if (patch.state_offset < bss_start || patch.state_offset > bss_end ||
                patch.state_size > bss_end - patch.state_offset ||
                !fits(nso, patch.state_offset, patch.state_size) ||
                !fits(out, patch.state_offset, patch.state_size)) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": timer state is outside BSS",
                            patch.name);
                continue;
            }
            const size_t offset = sizeof(Loader::NSOHeader) + patch.state_offset;
            if (!std::all_of(nso.begin() + offset, nso.begin() + offset + patch.state_size,
                             [](u8 byte) { return byte == 0; }) ||
                !std::equal(nso.begin() + offset, nso.begin() + offset + patch.state_size,
                            out.begin() + offset)) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": timer state is not zero-filled",
                            patch.name);
                continue;
            }
        }
        bool apply = true;
        for (const auto& record : patch.records) {
            if (!fits(nso, record.module_offset, record.bytes.size()) ||
                !fits(out, record.module_offset, record.bytes.size()) ||
                !fits(nso, record.module_offset, record.original.size())) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": {}+{:#x} is past the module's end",
                            patch.name, name, record.module_offset);
                apply = false;
                break;
            }
            const size_t offset = sizeof(Loader::NSOHeader) + record.module_offset;
            if (record.executable_padding) {
                const auto& text = header.segments[0];
                const u64 text_end = u64{text.location} + text.size;
                const u64 executable_end = (text_end + 0xFFF) & ~u64{0xFFF};
                if (record.module_offset < text_end ||
                    record.module_offset + record.bytes.size() > executable_end ||
                    record.module_offset + record.bytes.size() > header.segments[1].location ||
                    !std::all_of(nso.begin() + offset, nso.begin() + offset + record.bytes.size(),
                                 [](u8 byte) { return byte == 0; })) {
                    LOG_WARNING(Loader,
                                "    - Skipping \"{}\": code cave is not executable padding",
                                patch.name);
                    apply = false;
                    break;
                }
            }
            if (!std::equal(record.original.begin(), record.original.end(), nso.begin() + offset)) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": unexpected original bytes at {}+{:#x}",
                            patch.name, name, record.module_offset);
                apply = false;
                break;
            }
            if (!std::equal(nso.begin() + offset, nso.begin() + offset + record.bytes.size(),
                            out.begin() + offset)) {
                LOG_INFO(Loader, "    - Skipping \"{}\": a user mod already patched {}+{:#x}",
                         patch.name, name, record.module_offset);
                apply = false;
                break;
            }
        }
        if (!apply) {
            continue;
        }
        for (const auto& record : patch.records) {
            std::memcpy(out.data() + sizeof(Loader::NSOHeader) + record.module_offset,
                        record.bytes.data(), record.bytes.size());
        }
        LOG_INFO(Loader, "    - Applying built-in patch \"{}\" ({} records) to {}", patch.name,
                 patch.records.size(), name);
        if (!patch.replaces.empty()) {
            replaced.push_back(patch.replaces);
        }
        if (patch.dlss_quality) {
            Settings::SetAppliedOcarinaRenderScale(patch.dlss_quality);
        }
    }

    if (out.size() < sizeof(Loader::NSOHeader)) {
        return nso;
    }

    std::memcpy(out.data(), &header, sizeof(header));
    return out;
}

bool PatchManager::HasNSOPatch(const BuildID& build_id_, std::string_view name) const {
    const auto build_id_raw = Common::HexToString(build_id_);
    const auto build_id = build_id_raw.substr(0, build_id_raw.find_last_not_of('0') + 1);

    LOG_INFO(Loader, "Querying NSO patch existence for build_id={}, name={}", build_id, name);

    if (std::ranges::any_of(BUILTIN_PATCHES, [&](const BuiltinPatch& patch) {
            return patch.title_id == title_id && patch.module == name;
        })) {
        return true;
    }

    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return false;
    }

    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    return !CollectPatches(patch_dirs, build_id).empty();
}

std::vector<Core::Memory::CheatEntry> PatchManager::CreateCheatList(const BuildID& build_id_) const {
    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return {};
    }

    const auto disabled = GetDisabledAddons();
    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(), [](auto const& l, auto const& r) { return l->GetName() < r->GetName(); });

    // <mod dir> / <folder> / cheats / <build id>.txt
    std::vector<Core::Memory::CheatEntry> out;
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) == disabled.cend()) {
            if (auto cheats_dir = FindSubdirectoryCaseless(subdir, "cheats"); cheats_dir != nullptr) {
                if (auto const res = ReadCheatFileFromFolder(title_id, build_id_, cheats_dir, true))
                    std::copy(res->begin(), res->end(), std::back_inserter(out));
                if (auto const res = ReadCheatFileFromFolder(title_id, build_id_, cheats_dir, false))
                    std::copy(res->begin(), res->end(), std::back_inserter(out));
            }
        }
    }
    // Uncareless user-friendly loading of patches (must start with 'cheat_')
    // <mod dir> / <cheat file>.txt
    for (auto const& f : load_dir->GetFiles()) {
        auto const name = f->GetName();
        if (name.starts_with("cheat_") && std::find(disabled.cbegin(), disabled.cend(), name) == disabled.cend()) {
            std::vector<u8> data(f->GetSize());
            if (f->Read(data.data(), data.size()) == data.size()) {
                const Core::Memory::TextCheatParser parser;
                auto const res = parser.Parse(std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
                std::copy(res.begin(), res.end(), std::back_inserter(out));
            } else {
                LOG_INFO(Common_Filesystem, "Failed to read cheats file for title_id={:016X}", title_id);
            }
        }
    }
    return out;
}

void PatchManager::ApplyLayeredFS(VirtualFile& romfs, ContentRecordType type) const {
    const auto load_dir = GetModificationLoadRoot();
    const auto sdmc_load_dir = GetModificationLoadRoot(true);
    if ((type != ContentRecordType::Program && type != ContentRecordType::Data &&
         type != ContentRecordType::HtmlDocument) ||
        (load_dir == nullptr && sdmc_load_dir == nullptr)) {
        return;
    }

    const auto disabled = GetDisabledAddons();
    std::vector<VirtualDir> patch_dirs;
    if (load_dir != nullptr) {
        patch_dirs = load_dir->GetSubdirectories();
    }
    if (std::find(disabled.cbegin(), disabled.cend(), "SDMC") == disabled.cend() &&
        sdmc_load_dir != nullptr) {
        patch_dirs.push_back(sdmc_load_dir);
    }
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    std::vector<VirtualDir> layers;
    std::vector<VirtualDir> layers_ext;
    std::vector<std::string> mods;
    layers.reserve(patch_dirs.size() + 1);
    layers_ext.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend()) {
            continue;
        }

        const auto layer = [&](std::vector<VirtualDir>& out, std::string_view kind) {
            auto dir = FindSubdirectoryCaseless(subdir, kind);
            if (dir == nullptr) {
                return;
            }
            out.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(dir)));
            mods.push_back(fmt::format("{} ({})", subdir->GetName(), kind));
        };
        layer(layers, "romfs");
        // Support for romfslite introduced in Atmosphere 1.9.5
        layer(layers, "romfslite");
        layer(layers_ext, "romfs_ext");
        if (type == ContentRecordType::HtmlDocument) {
            layer(layers, "manual_html");
        }
    }

    // When there are no layers to apply, return early as there is no need to rebuild the RomFS
    if (layers.empty() && layers_ext.empty()) {
        return;
    }

    auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return;
    }

    layers.emplace_back(std::move(extracted));

    auto layered = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers));
    auto layered_ext = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers_ext));
    if (layered == nullptr && layered_ext == nullptr) {
        return;
    }

    auto packed = CreateRomFS(std::move(layered), std::move(layered_ext));
    if (packed == nullptr) {
        return;
    }

    LOG_INFO(Loader, "    RomFS: LayeredFS patches applied successfully");
    for (const auto& mod : mods) {
        LOG_INFO(Loader, "    RomFS: LayeredFS mod {}", mod);
    }
    romfs = std::move(packed);
}

VirtualFile PatchManager::PatchRomFS(const NCA* base_nca, VirtualFile base_romfs,
                                     ContentRecordType type, VirtualFile packed_update_raw,
                                     bool apply_layeredfs) const {
    const auto log_string = fmt::format("Patching RomFS for title_id={:016X}, type={:02X}",
                                        title_id, static_cast<u8>(type));
    if (type == ContentRecordType::Program || type == ContentRecordType::Data) {
        LOG_INFO(Loader, "{}", log_string);
    } else {
        LOG_DEBUG(Loader, "{}", log_string);
    }

    auto romfs = base_romfs;

    // Game Updates
    const auto update_tid = GetUpdateTitleID(title_id);
    const auto disabled = GetDisabledAddons();

    bool update_disabled = true;
    std::optional<u32> enabled_version;
    VirtualFile update_raw = nullptr;
    bool checked_external = false;
    bool checked_manual = false;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);
    if (content_union) {
        // First, check ExternalContentProvider
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            if (!update_versions.empty()) {
                checked_external = true;
                for (const auto& update_entry : update_versions) {
                    if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                        update_disabled = false;
                        enabled_version = update_entry.version;
                        update_raw = external_provider->GetEntryForVersion(update_tid, type, update_entry.version);
                        break;
                    }
                }
            }
        }

        if (!checked_external) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

                if (!manual_update_versions.empty()) {
                    checked_manual = true;
                    for (const auto& update_entry : manual_update_versions) {
                        if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                            update_disabled = false;
                            enabled_version = update_entry.version;
                            update_raw = manual_provider->GetEntryForVersion(update_tid, type, update_entry.version);
                            break;
                        }
                    }
                }
            }
        }
    }

    if (!checked_external && !checked_manual) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();
        const bool generic_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();

        if (!nand_disabled && !sdmc_disabled && !generic_disabled) {
            update_disabled = false;
        }
        if (!update_disabled) {
            update_raw = content_provider.GetEntryRaw(update_tid, type);
        }
    } else if (update_disabled && content_union) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();

        if (!nand_disabled || !sdmc_disabled) {
            const auto nand_sdmc_entries = content_union->ListEntriesFilterOrigin(
                std::nullopt, TitleType::Update, type, update_tid);

            for (const auto& [slot, entry] : nand_sdmc_entries) {
                if (slot == ContentProviderUnionSlot::UserNAND ||
                    slot == ContentProviderUnionSlot::SysNAND) {
                    if (!nand_disabled) {
                        update_disabled = false;
                        update_raw = content_provider.GetEntryRaw(update_tid, type);
                        break;
                    }
                } else if (slot == ContentProviderUnionSlot::SDMC) {
                    if (!sdmc_disabled) {
                        update_disabled = false;
                        update_raw = content_provider.GetEntryRaw(update_tid, type);
                        break;
                    }
                }
            }
        }
    }

    if (!update_disabled && update_raw != nullptr && base_nca != nullptr) {
        const auto new_nca = std::make_shared<NCA>(update_raw, base_nca);
        if (new_nca->GetStatus() == Loader::ResultStatus::Success &&
            new_nca->GetRomFS() != nullptr) {
            LOG_INFO(Loader, "    RomFS: Update ({}) applied successfully",
                     enabled_version.has_value() ? FormatTitleVersion(*enabled_version) :
                     FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0)));
            romfs = new_nca->GetRomFS();
        }
    } else if (!update_disabled && packed_update_raw != nullptr && base_nca != nullptr) {
        const auto new_nca = std::make_shared<NCA>(packed_update_raw, base_nca);
        if (new_nca->GetStatus() == Loader::ResultStatus::Success &&
            new_nca->GetRomFS() != nullptr) {
            LOG_INFO(Loader, "    RomFS: Update (PACKED) applied successfully");
            romfs = new_nca->GetRomFS();
        }
    }

    // LayeredFS
    if (apply_layeredfs) {
        ApplyLayeredFS(romfs, type);
    }

    return romfs;
}

std::vector<Patch> PatchManager::GetPatches(VirtualFile update_raw) const {
    if (title_id == 0) {
        return {};
    }

    std::vector<Patch> out;
    const auto& disabled = Settings::values.disabled_addons[title_id];

    // Game Updates
    const auto update_tid = GetUpdateTitleID(title_id);

    std::vector<Patch> external_update_patches;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);

    if (content_union) {
        // First, check ExternalContentProvider for updates
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            for (const auto& update_entry : update_versions) {
                std::string version_str = update_entry.version_string;
                if (version_str.empty()) {
                    version_str = FormatTitleVersion(update_entry.version);
                }

                const auto update_disabled =
                    IsVersionedExternalUpdateDisabled(disabled, update_entry.version);

                Patch update_patch = {.enabled = !update_disabled,
                                      .name = "Update",
                                      .version = version_str,
                                      .type = PatchType::Update,
                                      .program_id = title_id,
                                      .title_id = update_tid,
                                      .source = PatchSource::External,
                                      .numeric_version = update_entry.version};

                external_update_patches.push_back(update_patch);
            }
        }

        const auto* manual_provider = static_cast<const ManualContentProvider*>(
            content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
        if (manual_provider && external_update_patches.empty()) {
            const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

            for (const auto& update_entry : manual_update_versions) {
                std::string version_str = update_entry.version_string;
                if (version_str.empty()) {
                    version_str = FormatTitleVersion(update_entry.version);
                }

                const auto update_disabled =
                    IsVersionedExternalUpdateDisabled(disabled, update_entry.version);


                Patch update_patch = {.enabled = !update_disabled,
                                      .name = "Update",
                                      .version = version_str,
                                      .type = PatchType::Update,
                                      .program_id = title_id,
                                      .title_id = update_tid,
                                      .source = PatchSource::External,
                                      .numeric_version = update_entry.version};

                external_update_patches.push_back(update_patch);
            }
        }

        if (external_update_patches.size() > 1) {
            bool found_enabled = false;
            for (auto& patch : external_update_patches) {
                if (patch.enabled) {
                    if (found_enabled) {
                        patch.enabled = false;
                    } else {
                        found_enabled = true;
                    }
                }
            }
        }

        for (auto& patch : external_update_patches) {
            out.push_back(std::move(patch));
        }

        const auto all_updates = content_union->ListEntriesFilterOrigin(
            std::nullopt, std::nullopt, ContentRecordType::Program, update_tid);

        for (const auto& [slot, entry] : all_updates) {
            (void)entry;
            if (slot == ContentProviderUnionSlot::External ||
                slot == ContentProviderUnionSlot::FrontendManual) {
                continue;
            }

            PatchSource source_type = PatchSource::Unknown;
            std::string source_suffix;

            switch (slot) {
                case ContentProviderUnionSlot::UserNAND:
                case ContentProviderUnionSlot::SysNAND:
                    source_type = PatchSource::NAND;
                    source_suffix = " (NAND)";
                    break;
                case ContentProviderUnionSlot::SDMC:
                    source_type = PatchSource::SDMC;
                    source_suffix = " (SDMC)";
                    break;
                default:
                    break;
            }

            std::string version_str;
            u32 numeric_ver = 0;
            const auto* slot_provider = content_union->GetSlotProvider(slot);
            version_str = GetUpdateVersionStringFromSlot(slot_provider, update_tid);

            if (slot_provider != nullptr) {
                const auto slot_ver = slot_provider->GetEntryVersion(update_tid);
                if (slot_ver.has_value()) {
                    numeric_ver = *slot_ver;
                    if (version_str.empty() && numeric_ver != 0) {
                        version_str = FormatTitleVersion(numeric_ver);
                    }
                }
            }

            std::string patch_name = "Update" + source_suffix;

            bool update_disabled =
                std::find(disabled.cbegin(), disabled.cend(), patch_name) != disabled.cend();

            Patch update_patch = {.enabled = !update_disabled,
                                  .name = patch_name,
                                  .version = version_str,
                                  .type = PatchType::Update,
                                  .program_id = title_id,
                                  .title_id = update_tid,
                                  .source = source_type,
                                  .numeric_version = numeric_ver};

            out.push_back(update_patch);
        }
    } else {
        PatchManager update{update_tid, fs_controller, content_provider};
        const auto metadata = update.GetControlMetadata();
        const auto& nacp = metadata.first;

        bool update_disabled =
            std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();
        Patch update_patch = {.enabled = !update_disabled,
                              .name = "Update",
                              .version = "",
                              .type = PatchType::Update,
                              .program_id = title_id,
                              .title_id = title_id,
                              .source = PatchSource::Unknown,
                              .numeric_version = 0};

        if (nacp != nullptr) {
            update_patch.version = nacp->GetVersionString();
            out.push_back(update_patch);
        } else {
            if (content_provider.HasEntry(update_tid, ContentRecordType::Program)) {
                const auto meta_ver = content_provider.GetEntryVersion(update_tid);
                if (meta_ver.value_or(0) == 0) {
                    out.push_back(update_patch);
                } else {
                    update_patch.version = FormatTitleVersion(*meta_ver);
                    update_patch.numeric_version = *meta_ver;
                    out.push_back(update_patch);
                }
            } else if (update_raw != nullptr) {
                update_patch.version = "PACKED";
                update_patch.source = PatchSource::Packed;
                out.push_back(update_patch);
            }
        }
    }

    // General Mods (LayeredFS and IPS)
    const auto mod_dir = fs_controller.GetModificationLoadRoot(title_id);
    if (mod_dir != nullptr) {
        for (auto const& f : mod_dir->GetFiles())
            if (auto const name = f->GetName(); name.starts_with("cheat_")) {
                auto const mod_disabled = std::find(disabled.begin(), disabled.end(), name) != disabled.end();
                out.push_back({
                    .enabled = !mod_disabled,
                    .name = name,
                    .version = "Cheats",
                    .type = PatchType::Mod,
                    .program_id = title_id,
                    .title_id = title_id,
                    .source = PatchSource::Unknown,
                    .location = f->GetFullPath(),
                });
            }

        for (const auto& mod : mod_dir->GetSubdirectories()) {
            std::string types;

            const auto exefs_dir = FindSubdirectoryCaseless(mod, "exefs");
            if (IsDirValidAndNonEmpty(exefs_dir)) {
                bool ips = false;
                bool ipswitch = false;
                bool layeredfs = false;

                for (const auto& file : exefs_dir->GetFiles()) {
                    if (file->GetExtension() == "ips") {
                        ips = true;
                    } else if (file->GetExtension() == "pchtxt") {
                        ipswitch = true;
                    } else if (std::find(EXEFS_FILE_NAMES.begin(), EXEFS_FILE_NAMES.end(),
                                         file->GetName()) != EXEFS_FILE_NAMES.end()) {
                        layeredfs = true;
                                         }
                }

                if (ips)
                    AppendCommaIfNotEmpty(types, "IPS");
                if (ipswitch)
                    AppendCommaIfNotEmpty(types, "IPSwitch");
                if (layeredfs)
                    AppendCommaIfNotEmpty(types, "LayeredExeFS");
            }
            if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfs")) ||
                IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfslite")))
                AppendCommaIfNotEmpty(types, "LayeredFS");
            if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfs_ext")))
                AppendCommaIfNotEmpty(types, "ExtLayeredFS");
            if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "cheats")))
                AppendCommaIfNotEmpty(types, "Cheats");

            if (types.empty())
                continue;

            const auto mod_disabled = std::find(disabled.begin(), disabled.end(), mod->GetName()) != disabled.end();
            out.push_back({.enabled = !mod_disabled,
                           .name = mod->GetName(),
                           .version = types,
                           .type = PatchType::Mod,
                           .program_id = title_id,
                           .title_id = title_id,
                           .source = PatchSource::Unknown,
                           .location = mod->GetFullPath()});
        }
    }

    // SDMC mod directory (RomFS LayeredFS)
    const auto sdmc_mod_dir = fs_controller.GetSDMCModificationLoadRoot(title_id);
    if (sdmc_mod_dir != nullptr) {
        std::string types;
        if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "exefs")))
            AppendCommaIfNotEmpty(types, "LayeredExeFS");
        if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfs")) ||
            IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfslite")))
            AppendCommaIfNotEmpty(types, "LayeredFS");
        if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfs_ext")))
            AppendCommaIfNotEmpty(types, "ExtLayeredFS");

        if (!types.empty()) {
            const auto mod_disabled =
                std::find(disabled.begin(), disabled.end(), "SDMC") != disabled.end();
            out.push_back({.enabled = !mod_disabled,
                           .name = "SDMC",
                           .version = types,
                           .type = PatchType::Mod,
                           .program_id = title_id,
                           .title_id = title_id,
                           .source = PatchSource::Unknown});
        }
    }

    for (const auto& patch : BUILTIN_PATCHES) {
        if (patch.title_id != title_id || patch.dlss_quality) {
            continue;
        }
        const auto is_disabled = [&](std::string_view name) {
            return std::find(disabled.begin(), disabled.end(), name) != disabled.end();
        };
        // Name an enabled built-in that contains this one, so unchecking only this one is not
        // mistaken for turning the change off. Whether it applies is only known in PatchNSO (a
        // user mod can make it skip), hence "when it applies".
        const auto replacing = std::ranges::find_if(BUILTIN_PATCHES, [&](const BuiltinPatch& other) {
            return other.title_id == title_id && other.replaces == patch.name &&
                   !is_disabled(other.name);
        });
        out.push_back({.enabled = !is_disabled(patch.name),
                       .name = std::string{patch.name},
                       .version = replacing != BUILTIN_PATCHES.end()
                                      ? fmt::format("IPS, replaced by {} when it applies",
                                                    replacing->name)
                                      : std::string{"IPS"},
                       .type = PatchType::Mod,
                       .program_id = title_id,
                       .title_id = title_id,
                       .source = PatchSource::Unknown});
    }

    // DLC
    std::vector<ContentProviderEntry> dlc_match;
    bool has_external_dlc = false;
    bool has_nand_dlc = false;
    bool has_sdmc_dlc = false;
    bool has_other_dlc = false;
    const auto dlc_entries_with_origin =
        content_union->ListEntriesFilterOrigin(std::nullopt, TitleType::AOC, ContentRecordType::Data);

    dlc_match.reserve(dlc_entries_with_origin.size());
    for (const auto& [slot, entry] : dlc_entries_with_origin) {
        const auto base_tid = GetBaseTitleID(entry.title_id);
        const bool matches_base = base_tid == title_id;
        if (!matches_base) {
            LOG_DEBUG(Loader, "DLC {:016X} base {:016X} doesn't match title {:016X}",
                      entry.title_id, base_tid, title_id);
            continue;
        }

        const auto* slot_provider = content_union->GetSlotProvider(slot);
        if (slot_provider == nullptr) {
            continue;
        }

        auto nca = slot_provider->GetEntry(entry);
        if (!nca) {
            LOG_DEBUG(Loader, "Failed to get NCA for DLC {:016X}", entry.title_id);
            continue;
        }

        const auto status = nca->GetStatus();
        if (status != Loader::ResultStatus::Success) {
            LOG_DEBUG(Loader, "DLC {:016X} NCA has status {}", entry.title_id,
                      static_cast<int>(status));
            continue;
        }

        switch (slot) {
            case ContentProviderUnionSlot::External:
            case ContentProviderUnionSlot::FrontendManual:
                has_external_dlc = true;
                break;
            case ContentProviderUnionSlot::UserNAND:
            case ContentProviderUnionSlot::SysNAND:
                has_nand_dlc = true;
                break;
            case ContentProviderUnionSlot::SDMC:
                has_sdmc_dlc = true;
                break;
            default:
                has_other_dlc = true;
                break;
        }
        dlc_match.push_back(entry);
    }

    if (!dlc_match.empty()) {
        // Ensure sorted so DLC IDs show in order.
        std::sort(dlc_match.begin(), dlc_match.end());

        std::string list;
        for (size_t i = 0; i < dlc_match.size() - 1; ++i)
            list += fmt::format("{}, ", dlc_match[i].title_id & 0x7FF);

        list += fmt::format("{}", dlc_match.back().title_id & 0x7FF);

        const auto dlc_disabled =
            std::find(disabled.begin(), disabled.end(), "DLC") != disabled.end();
        PatchSource dlc_source = PatchSource::Unknown;
        if (has_external_dlc && !has_nand_dlc && !has_sdmc_dlc && !has_other_dlc) {
            dlc_source = PatchSource::External;
        } else if (has_nand_dlc && !has_external_dlc && !has_sdmc_dlc && !has_other_dlc) {
            dlc_source = PatchSource::NAND;
        } else if (has_sdmc_dlc && !has_external_dlc && !has_nand_dlc && !has_other_dlc) {
            dlc_source = PatchSource::SDMC;
        }

        out.push_back({.enabled = !dlc_disabled,
                       .name = "DLC",
                       .version = std::move(list),
                       .type = PatchType::DLC,
                       .program_id = title_id,
                       .title_id = dlc_match.back().title_id,
                       .source = dlc_source});
    }

    return out;
}

std::optional<u32> PatchManager::GetGameVersion() const {
    const auto update_tid = GetUpdateTitleID(title_id);
    if (content_provider.HasEntry(update_tid, ContentRecordType::Program)) {
        const auto version = content_provider.GetEntryVersion(update_tid);
        return version || !parent_title_id ? version : content_provider.GetEntryVersion(GetUpdateTitleID(*parent_title_id));
    }

    const auto version = content_provider.GetEntryVersion(title_id);
    return version || !parent_title_id ? version : content_provider.GetEntryVersion(*parent_title_id);
}

PatchManager::Metadata PatchManager::GetControlMetadata() const {
    const auto base_control_nca = content_provider.GetEntry(title_id, ContentRecordType::Control);
    if (base_control_nca == nullptr) {
        if (parent_title_id) {
            const auto control_id = content_provider.HasEntry(*parent_title_id, ContentRecordType::Control) ? *parent_title_id : GetUpdateTitleID(*parent_title_id);
            const PatchManager parent{control_id, fs_controller, content_provider};
            return parent.GetControlMetadata();
        }
        return {};
    }

    return ParseControlNCA(*base_control_nca);
}

PatchManager::Metadata PatchManager::ParseControlNCA(const NCA& nca) const {
    const auto base_romfs = nca.GetRomFS();
    if (base_romfs == nullptr) {
        return {};
    }

    return ParseControlRomFS(PatchRomFS(&nca, base_romfs, ContentRecordType::Control));
}

PatchManager::Metadata PatchManager::ParseControlRomFS(const VirtualFile& romfs) {
    if (romfs == nullptr) {
        return {};
    }

    const auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return {};
    }

    auto nacp_file = extracted->GetFile("control.nacp");
    if (nacp_file == nullptr) {
        nacp_file = extracted->GetFile("Control.nacp");
    }

    auto nacp = nacp_file == nullptr ? nullptr : std::make_unique<NACP>(nacp_file);

    // Get language code from settings
    const auto language_code = Service::Set::GetLanguageCodeFromIndex(
        static_cast<u32>(Settings::values.language_index.GetValue()));

    // Convert to application language and get priority list
    const auto application_language =
        Service::NS::ConvertToApplicationLanguage(language_code)
            .value_or(Service::NS::ApplicationLanguage::AmericanEnglish);
    const auto language_priority_list =
        Service::NS::GetApplicationLanguagePriorityList(application_language);

    // Convert to language names
    auto priority_language_names = FileSys::LANGUAGE_NAMES; // Copy
    if (language_priority_list) {
        for (size_t i = 0; i < priority_language_names.size(); ++i) {
            // Relies on FileSys::LANGUAGE_NAMES being in the same order as
            // Service::NS::ApplicationLanguage
            const auto language_index = static_cast<u8>(language_priority_list->at(i));

            if (language_index < FileSys::LANGUAGE_NAMES.size()) {
                priority_language_names[i] = FileSys::LANGUAGE_NAMES[language_index];
            } else {
                // Not a catastrophe, unlikely to happen
                LOG_WARNING(Loader, "Invalid language index {}", language_index);
            }
        }
    }

    // Get first matching icon
    VirtualFile icon_file;
    for (const auto& language : priority_language_names) {
        icon_file = extracted->GetFile(std::string("icon_").append(language).append(".dat"));
        if (icon_file != nullptr) {
            break;
        }
    }

    return {std::move(nacp), icon_file};
}

[[nodiscard]] PatchManager::Metadata PatchManager::GetMetadataFromBaseOrUpdate(Core::System& system, u64 application_id) noexcept {
    const FileSys::PatchManager pm{application_id, system.GetFileSystemController(), system.GetContentProvider()};
    auto metadata = pm.GetControlMetadata();
    if (metadata.first != nullptr)
        return metadata;
    const FileSys::PatchManager pm_update{FileSys::GetUpdateTitleID(application_id), system.GetFileSystemController(), system.GetContentProvider()};
    return pm_update.GetControlMetadata();
}

} // namespace FileSys
