// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <span>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "common/assert.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "common/settings.h"
#ifndef _WIN32
#include "common/string_util.h"
#endif

#include "core/arm/debug.h"
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
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
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

/// Bytes a built-in patch writes at an offset into the module's code (after the NSO header; an
/// exefs IPS patch's offset is 0x100 more). With `checked`, the module must hold `original` there.
struct BuiltinPatchRecord {
    size_t module_offset;
    std::span<const u8> bytes;
    bool checked;
    u32 original;
};

struct BuiltinPatch {
    std::string_view name;
    u64 title_id;
    std::string_view module;
    std::string_view build_id;
    std::span<const BuiltinPatchRecord> records;
    /// When set, the patch applies only while this environment variable is 1 (A/B before it
    /// becomes a default).
    const char* enable_env = nullptr;
    /// When set, this environment variable at 1 skips the patch and at 0 applies it even when
    /// Add-Ons has it off (A/B runs without touching the player's Add-Ons).
    const char* disable_env = nullptr;
    /// When set, the patch applies (and is listed in Add-Ons) only while this returns true.
    bool (*setting_enabled)() = nullptr;
    /// Another built-in this one contains: skipped once this one is applied.
    std::string_view replaces{};
    /// When set, guest state the patch keeps in the module (offset, size): it must lie in BSS.
    size_t state_offset = 0;
    size_t state_size = 0;
};

// Ocarina of Time takes its "dynamic resolution disabled" path, so the 3D renders at full scale
// instead of following the GPU-time controller down to its 480p floor.
constexpr std::array<u8, 4> MAX_RESOLUTION_BRANCH{0x0F, 0x02, 0x00, 0x14}; // b 0x80640c
constexpr std::array MAX_RESOLUTION_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x805BD0,
                       .bytes = MAX_RESOLUTION_BRANCH,
                       .checked = true,
                       .original = 0x360041E8}, // tbz w8, #0, 0x80640c
};

// Dynamic FPS by Dark (Eglantine Dev), bundled with his permission: the game's frame step follows
// the time measured with the system counter instead of assuming 30 fps, so it runs at full speed
// at any frame rate. A call (bl 0x3c46200) to a routine in unused space after the code.
constexpr std::array<u8, 4> DYNAMIC_FPS_CALL{0x4F, 0x7D, 0xD5, 0x94};
constexpr std::array<u8, 120> DYNAMIC_FPS_ROUTINE{
    0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91,
    0x2B, 0xE0, 0x3B, 0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9,
    0xCC, 0x02, 0x00, 0xB4, 0x6B, 0x01, 0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5,
    0x8E, 0x02, 0x80, 0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F, 0x01, 0x0E, 0xEB,
    0xCB, 0x81, 0x8B, 0x9A, 0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5,
    0xEC, 0x03, 0x0B, 0xAA, 0x6E, 0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B,
    0x4C, 0x05, 0x00, 0xF9, 0x82, 0x01, 0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E,
    0x42, 0x18, 0x23, 0x1E, 0x42, 0x08, 0x21, 0x1E, 0xC3, 0x00, 0x00, 0x1C,
    0x03, 0x09, 0x23, 0x1E, 0x40, 0x20, 0x23, 0x1E, 0x4D, 0x00, 0x00, 0x54,
    0x48, 0x40, 0x20, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A, 0xD7, 0x83, 0x3F,
};
constexpr std::array DYNAMIC_FPS_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x6E6CC4, .bytes = DYNAMIC_FPS_CALL},
    BuiltinPatchRecord{.module_offset = 0x3C46200, .bytes = DYNAMIC_FPS_ROUTINE},
};

// High refresh rate (experimental, Settings oot_high_refresh): Dark's routine with the tail of
// zortos293/Lichii#655 ("FPS Unlock", by atk3001): the smoothed real-time step is always
// returned, also below 1.0, so the game keeps real speed above 60 fps (0.5 a frame at 120).
// Through the fmul at +0x58 it is Dark's routine byte for byte (the 50 ms delta cap included;
// #655 raised it to 100 ms). The tail clamps the step to [0.02, 3.0] (3.0 = that cap; #655: 6.0) and multiplies it
// by a speed scale word that the present pacing sets from the speed limit (Turbo, Slow; 1.0 at
// 100%), since pacing alone no longer changes the game's speed. The first call (no earlier
// tick) still returns the game's own step from +0x70 (the head's cbz target).
//   +0x5C ldr s3, +0x80 ; fmaxnm s2, s2, s3 ; ldr s3, +0x84 ; fminnm s2, s2, s3 ; b +0x74
//   +0x70 ret
//   +0x74 ldr s3, +0x88 ; fmul s8, s2, s3 ; ret
//   +0x80 0.02f ; 3.0f ; speed scale (1.0f)
constexpr std::array<u8, 140> HIGH_REFRESH_ROUTINE{
    0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91,
    0x2B, 0xE0, 0x3B, 0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9,
    0xCC, 0x02, 0x00, 0xB4, 0x6B, 0x01, 0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5,
    0x8E, 0x02, 0x80, 0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F, 0x01, 0x0E, 0xEB,
    0xCB, 0x81, 0x8B, 0x9A, 0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5,
    0xEC, 0x03, 0x0B, 0xAA, 0x6E, 0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B,
    0x4C, 0x05, 0x00, 0xF9, 0x82, 0x01, 0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E,
    0x42, 0x18, 0x23, 0x1E, 0x42, 0x08, 0x21, 0x1E, 0x23, 0x01, 0x00, 0x1C,
    0x42, 0x68, 0x23, 0x1E, 0x03, 0x01, 0x00, 0x1C, 0x42, 0x78, 0x23, 0x1E,
    0x02, 0x00, 0x00, 0x14, 0xC0, 0x03, 0x5F, 0xD6, 0xA3, 0x00, 0x00, 0x1C,
    0x48, 0x08, 0x23, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A, 0xD7, 0xA3, 0x3C,
    0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x3F,
};
constexpr size_t HIGH_REFRESH_ROUTINE_OFFSET = 0x3C46200;
constexpr size_t HIGH_REFRESH_SCALE = 0x88; // the speed scale word, in the routine

// Water ripples above 60 fps (Port of zortos293/Lichii#1152, by atk3001): the water renderer
// (main+0x472E0) runs its ripple simulation trunc(step) * substeps times a frame (fcvtzs w9, s0 at
// main+0x475F8), so below a step of 1 the rings Link makes in water freeze. That fcvtzs branches
// to an accumulator that keeps the fractional step per water object (x21) in 8 slots in BSS
// after Dark's timer state (main+0x4E9FE00, 16 bytes): a u32 round-robin index at +0x10, then
// (u64 owner, f32 rest) slots with a 16-byte stride at +0x20, replaced round-robin when full.
// It returns w9 = trunc(step + rest) and keeps the rest. Upstream puts it at main+0x3C46280,
// inside our routine (+0x80..+0x8C), so it sits at +0x90 here: the same page, so its adrp is
// upstream's word; only the two branches between the hook and it are re-encoded.
constexpr size_t WATER_HOOK_OFFSET = 0x475F8;
constexpr u32 WATER_FCVTZS = 0x1E380009;                        // fcvtzs w9, s0
constexpr std::array<u8, 4> WATER_HOOK{0x26, 0xFB, 0xEF, 0x14}; // b 0x3c46290
constexpr size_t WATER_ACCUMULATOR_OFFSET = HIGH_REFRESH_ROUTINE_OFFSET + 0x90;
constexpr std::array<u8, 128> WATER_ACCUMULATOR{
    0xC9, 0x92, 0x00, 0xB0, 0x29, 0x81, 0x38, 0x91, 0x0B, 0x01, 0x80, 0x52, 0x2A, 0x01,
    0x40, 0xF9, 0x5F, 0x01, 0x15, 0xEB, 0x80, 0x02, 0x00, 0x54, 0x29, 0x41, 0x00, 0x91,
    0x6B, 0x05, 0x00, 0x71, 0x61, 0xFF, 0xFF, 0x54, 0x29, 0x01, 0x02, 0xD1, 0x0B, 0x01,
    0x80, 0x52, 0x2A, 0x01, 0x40, 0xF9, 0x6A, 0x01, 0x00, 0xB4, 0x29, 0x41, 0x00, 0x91,
    0x6B, 0x05, 0x00, 0x71, 0x81, 0xFF, 0xFF, 0x54, 0x29, 0x41, 0x02, 0xD1, 0x2A, 0x01,
    0x40, 0xB9, 0x4B, 0x05, 0x00, 0x11, 0x6B, 0x09, 0x00, 0x12, 0x2B, 0x01, 0x00, 0xB9,
    0x29, 0x41, 0x00, 0x91, 0x29, 0x11, 0x0A, 0x8B, 0x35, 0x01, 0x00, 0xF9, 0x3F, 0x09,
    0x00, 0xB9, 0x21, 0x09, 0x40, 0xBD, 0x00, 0x28, 0x21, 0x1E, 0x01, 0xC0, 0x25, 0x1E,
    0x00, 0x38, 0x21, 0x1E, 0x20, 0x09, 0x00, 0xBD, 0x29, 0x00, 0x38, 0x1E, 0xBC, 0x04,
    0x10, 0x17, // +0x7C b 0x475fc (upstream, from 0x3c46280: 0x171004c0)
};
// Dark's timer state and the ripple slots (ldr/str through adrp 0x4e9f000 + 0xe00 / + 0xe20).
constexpr size_t HIGH_REFRESH_STATE_OFFSET = 0x4E9FE00;
constexpr size_t HIGH_REFRESH_STATE_SIZE = 0xA0;

// The code cave after the end of the code (main+0x3C46200; #1152 checks it is executable padding,
// the text page runs on to main+0x3C47000) as the high refresh rate mode leaves it: its routine, a zero word, the water
// ripple accumulator. Past Dark's 120 bytes every word must be free (zero) in the module.
constexpr size_t OOT_CAVE_SIZE = 0x110;
constexpr std::array<u8, OOT_CAVE_SIZE> HIGH_REFRESH_CAVE = [] {
    std::array<u8, OOT_CAVE_SIZE> cave{};
    std::ranges::copy(HIGH_REFRESH_ROUTINE, cave.begin());
    std::ranges::copy(WATER_ACCUMULATOR,
                      cave.begin() + (WATER_ACCUMULATOR_OFFSET - HIGH_REFRESH_ROUTINE_OFFSET));
    return cave;
}();
static_assert(WATER_ACCUMULATOR_OFFSET - HIGH_REFRESH_ROUTINE_OFFSET + WATER_ACCUMULATOR.size() ==
              OOT_CAVE_SIZE);
// The same cave with Dark's routine (the default, and the high refresh mode switched off live).
constexpr std::array<u8, OOT_CAVE_SIZE> DYNAMIC_FPS_CAVE = [] {
    std::array<u8, OOT_CAVE_SIZE> cave{};
    std::ranges::copy(DYNAMIC_FPS_ROUTINE, cave.begin());
    return cave;
}();
constexpr std::array<u8, 4> WATER_ORIGINAL{0x09, 0x00, 0x38, 0x1E}; // fcvtzs w9, s0
static_assert(OotHighRefreshCode::CallOffset == 0x6E6CC4 &&
              OotHighRefreshCode::WaterOffset == WATER_HOOK_OFFSET &&
              OotHighRefreshCode::CaveOffset == HIGH_REFRESH_ROUTINE_OFFSET &&
              OotHighRefreshCode::CaveSize == OOT_CAVE_SIZE &&
              OotHighRefreshCode::ScaleOffset == HIGH_REFRESH_ROUTINE_OFFSET + HIGH_REFRESH_SCALE);

// The call, Dark's 120 bytes, the water hook (over its fcvtzs), then the rest of the cave one word
// at a time, each checked to be zero in the module.
constexpr auto HIGH_REFRESH_RECORDS = [] {
    constexpr size_t dark = DYNAMIC_FPS_ROUTINE.size();
    std::array<BuiltinPatchRecord, 3 + (OOT_CAVE_SIZE - dark) / 4> records{};
    records[0] = BuiltinPatchRecord{.module_offset = 0x6E6CC4, .bytes = DYNAMIC_FPS_CALL};
    records[1] = BuiltinPatchRecord{.module_offset = HIGH_REFRESH_ROUTINE_OFFSET,
                                    .bytes = std::span<const u8>{HIGH_REFRESH_CAVE}.first(dark)};
    records[2] = BuiltinPatchRecord{.module_offset = WATER_HOOK_OFFSET,
                                    .bytes = WATER_HOOK,
                                    .checked = true,
                                    .original = WATER_FCVTZS};
    for (size_t i = 3, offset = dark; offset < OOT_CAVE_SIZE; ++i, offset += 4) {
        records[i] = BuiltinPatchRecord{.module_offset = HIGH_REFRESH_ROUTINE_OFFSET + offset,
                                        .bytes = std::span<const u8>{HIGH_REFRESH_CAVE}.subspan(offset, 4),
                                        .checked = true,
                                        .original = 0};
    }
    return records;
}();
constexpr std::string_view DYNAMIC_FPS_NAME = "Dynamic FPS by Dark (built-in)";
constexpr std::string_view HIGH_REFRESH_NAME = "High refresh rate (built-in, experimental)";

bool HighRefreshSettingOn() {
    // Tests boot either way without touching the player's config: EDEN_NX2_OOT_HIGH_REFRESH=0/1.
    if (const char* const value = std::getenv("EDEN_NX2_OOT_HIGH_REFRESH"); value && *value) {
        return *value != '0';
    }
    return Settings::values.oot_high_refresh.GetValue();
}

// Occlusion Off (cLohan's community patch, from zortos293/Lichii#641): the game creates no
// z-prepass occlusion queries and no samples-passed queries (ldrb w8, [x0] -> mov w8, wzr), so it
// issues no occlusion reports and no conditional renders; every candidate object is drawn.
constexpr std::array<u8, 4> MOV_W8_WZR{0xE8, 0x03, 0x1F, 0x2A};
constexpr u32 LDRB_W8_X0 = 0x39400008;
constexpr std::array OCCLUSION_OFF_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x11ACE68, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B828C, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B82A4, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11ACEC8, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B82DC, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B82F4, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
};

// Resource poll backoff (zortos293/Lichii#641): the resource request thread (main+0xF7A0)
// rescans its pending requests at once when one is not ready (`tbz w8, #0, loop` at main+0x14ABC,
// w8 = "all done"). That branch now goes through the zero padding at main+0x17088: `svc #0xf3`
// (a short sleep while EXP25 is on, core/hle/kernel/svc/svc_nx2_oot_poll.cpp) and `b loop`.
constexpr std::array<u8, 4> POLL_RESCAN_BRANCH{0x68, 0x2E, 0x01, 0x36};  // tbz w8, #0, 0x17088
constexpr std::array<u8, 4> POLL_BACKOFF_SVC{0x61, 0x1E, 0x00, 0xD4};    // svc #0xf3
constexpr std::array<u8, 4> POLL_BACKOFF_RETURN{0x0D, 0xE2, 0xFF, 0x17}; // b 0xf8c0
constexpr std::array POLL_BACKOFF_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x14ABC,
                       .bytes = POLL_RESCAN_BRANCH,
                       .checked = true,
                       .original = 0x36057028}, // tbz w8, #0, 0xf8c0
    BuiltinPatchRecord{.module_offset = 0x17088, .bytes = POLL_BACKOFF_SVC, .checked = true, .original = 0},
    BuiltinPatchRecord{.module_offset = 0x1708C, .bytes = POLL_BACKOFF_RETURN, .checked = true, .original = 0},
};

constexpr std::array BUILTIN_PATCHES{
    BuiltinPatch{
        .name = "Max Resolution (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = MAX_RESOLUTION_RECORDS,
        .disable_env = "EDEN_NX2_NO_MAX_RES",
    },
    // Before Dark's: it contains his routine and replaces it while applied. Off by default; only
    // listed in Add-Ons while the setting is on.
    BuiltinPatch{
        .name = HIGH_REFRESH_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = HIGH_REFRESH_RECORDS,
        .setting_enabled = &HighRefreshSettingOn,
        .replaces = DYNAMIC_FPS_NAME,
        .state_offset = HIGH_REFRESH_STATE_OFFSET,
        .state_size = HIGH_REFRESH_STATE_SIZE,
    },
    BuiltinPatch{
        .name = DYNAMIC_FPS_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = DYNAMIC_FPS_RECORDS,
    },
    BuiltinPatch{
        .name = "Occlusion Off (built-in, EDEN_NX2_OCCLUSION_OFF=1)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = OCCLUSION_OFF_RECORDS,
        .enable_env = "EDEN_NX2_OCCLUSION_OFF",
    },
    BuiltinPatch{
        .name = "Resource poll backoff (built-in, EDEN_NX2_OOT_POLL_BACKOFF=1)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = POLL_BACKOFF_RECORDS,
        .enable_env = "EDEN_NX2_OOT_POLL_BACKOFF",
    },
};

// Whether [offset, offset + size) of a module (offsets after its NSO header) lies in its BSS.
bool InModuleBss(const Loader::NSOHeader& header, u64 offset, u64 size) {
    const auto& data = header.segments[2];
    const u64 bss_start = u64{data.location} + data.size;
    const u64 bss_end = bss_start + data.bss_size;
    return offset >= bss_start && offset <= bss_end && size <= bss_end - offset;
}

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
YUZU_NO_INLINE std::optional<u64> GetParentApplicationId(const ContentProvider* provider, u64 title_id) {
    return provider == nullptr ? std::nullopt : provider->GetParentApplicationId(title_id);
}

// The IPS and IPSwitch patches in a mod's exefs for the NSO with this build id (hex, padded with
// zeros to 64 digits).
std::vector<VirtualFile> ModPatchFiles(const VirtualDir& mod, const std::string& nso_build_id) {
    std::vector<VirtualFile> out;
    const auto exefs_dir = FindSubdirectoryCaseless(mod, "exefs");
    if (exefs_dir == nullptr) {
        return out;
    }
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
    return out;
}

// Whether a mod is Dark's standalone Dynamic FPS, which Fritchii has built in: one of its patches
// for Ocarina's main writes into the built-in's records (the call or the routine). Such a mod is
// left out (not applied, not listed in Add-Ons) so the built-in, which the high refresh rate mode
// switches live, is applied in its place. Mods overlapping the other built-ins still win.
bool IsBuiltinDynamicFpsMod(u64 title_id, const VirtualDir& mod) {
    const auto it = std::ranges::find(BUILTIN_PATCHES, DYNAMIC_FPS_NAME, &BuiltinPatch::name);
    if (it == BUILTIN_PATCHES.end() || it->title_id != title_id || mod == nullptr) {
        return false;
    }
    // A player who turned the built-in off in Add-Ons keeps their mod (as before), or they would
    // get no Dynamic FPS at all.
    if (const auto disabled = Settings::values.disabled_addons.find(title_id);
        disabled != Settings::values.disabled_addons.end() &&
        std::ranges::find(disabled->second, std::string{DYNAMIC_FPS_NAME}) != disabled->second.end()) {
        return false;
    }
    const auto overlaps = [&](const std::pair<u64, u64>& range) {
        return std::ranges::any_of(it->records, [&](const BuiltinPatchRecord& record) {
            const u64 begin = sizeof(Loader::NSOHeader) + record.module_offset;
            return range.first < begin + record.bytes.size() && begin < range.second;
        });
    };
    for (const auto& file : ModPatchFiles(mod, fmt::format("{:0<64}", it->build_id))) {
        const auto ranges = file->GetExtension() == "ips" ? GetIPSWriteRanges(file)
                                                          : IPSwitchCompiler{file}.GetWriteRanges();
        if (std::ranges::any_of(ranges, overlaps)) {
            return true;
        }
    }
    return false;
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

VirtualDir PatchManager::PatchExeFS(VirtualDir exefs) const {
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
        LOG_INFO(Loader, "    ExeFS: Update ({}) applied successfully",
                 FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0)));
        exefs = update->GetExeFS();
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
        if (exefs_dir != nullptr)
            layers.push_back(std::move(exefs_dir));
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

        const auto files = ModPatchFiles(subdir, nso_build_id);
        out.insert(out.end(), files.begin(), files.end());
    }
    return out;
}

std::vector<u8> PatchManager::PatchNSO(const std::vector<u8>& nso, const std::string& name) const {
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
    // Dark's standalone Dynamic FPS: the built-in is applied instead (the player's files stay).
    std::erase_if(patch_dirs, [&](const VirtualDir& mod) {
        if (!IsBuiltinDynamicFpsMod(title_id, mod)) {
            return false;
        }
        if (name == "main") {
            LOG_INFO(Loader,
                     "    - Skipping mod \"{}\": it is Dark's Dynamic FPS, which Fritchii has "
                     "built in",
                     mod->GetName());
        }
        return true;
    });
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
        if (patch.enable_env != nullptr) {
            const char* const value = std::getenv(patch.enable_env);
            if (value == nullptr || std::string_view{value} != "1") {
                continue;
            }
        }
        if (patch.setting_enabled != nullptr && !patch.setting_enabled()) {
            continue;
        }
        if (std::find(replaced.cbegin(), replaced.cend(), patch.name) != replaced.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": included in an applied built-in patch",
                     patch.name);
            continue;
        }
        // disable_env=1 skips the patch, =0 applies it even when Add-Ons has it off (A/B runs).
        const char* const ab_value = patch.disable_env ? std::getenv(patch.disable_env) : nullptr;
        const std::string_view ab{ab_value ? ab_value : ""};
        if (ab == "1") {
            LOG_INFO(Loader, "    - Skipping \"{}\": {}=1", patch.name, patch.disable_env);
            continue;
        }
        if (ab != "0" &&
            std::find(disabled.cbegin(), disabled.cend(), patch.name) != disabled.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": disabled in Add-Ons", patch.name);
            continue;
        }
        if (patch.build_id != build_id) {
            LOG_INFO(Loader, "    - Skipping \"{}\": build id {} is not {}", patch.name, build_id,
                     patch.build_id);
            continue;
        }
        if (patch.state_size != 0 && !InModuleBss(header, patch.state_offset, patch.state_size)) {
            LOG_WARNING(Loader, "    - Skipping \"{}\": its state at {}+{:#x} is outside BSS",
                        patch.name, name, patch.state_offset);
            continue;
        }
        // Every record must fit and hold its original bytes, and none may have been patched by a
        // user mod already (the mod's patch is used then, for all records).
        bool apply = true;
        for (const BuiltinPatchRecord& record : patch.records) {
            const size_t offset = sizeof(Loader::NSOHeader) + record.module_offset;
            if (nso.size() < offset + record.bytes.size() ||
                out.size() < offset + record.bytes.size()) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": {}+{:#x} is past the module's end",
                            patch.name, name, record.module_offset);
                apply = false;
                break;
            }
            if (record.checked) {
                u32 original;
                std::memcpy(&original, nso.data() + offset, sizeof(original));
                if (original != record.original) {
                    LOG_WARNING(Loader,
                                "    - Skipping \"{}\": {}+{:#x} holds {:#010x}, expected {:#010x}",
                                patch.name, name, record.module_offset, original, record.original);
                    apply = false;
                    break;
                }
            }
            if (!std::equal(nso.begin() + offset, nso.begin() + offset + record.bytes.size(),
                            out.begin() + offset)) {
                LOG_INFO(Loader,
                         "    - Skipping \"{}\": a user mod already patched {}+{:#x}, the mod's "
                         "patch is used",
                         patch.name, name, record.module_offset);
                apply = false;
                break;
            }
        }
        if (!apply) {
            continue;
        }
        for (const BuiltinPatchRecord& record : patch.records) {
            std::memcpy(out.data() + sizeof(Loader::NSOHeader) + record.module_offset,
                        record.bytes.data(), record.bytes.size());
        }
        LOG_INFO(Loader, "    - Applying built-in patch \"{}\" ({} records) to {}", patch.name,
                 patch.records.size(), name);
        if (!patch.replaces.empty()) {
            replaced.push_back(patch.replaces);
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
    layers.reserve(patch_dirs.size() + 1);
    layers_ext.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend()) {
            continue;
        }

        auto romfs_dir = FindSubdirectoryCaseless(subdir, "romfs");
        if (romfs_dir != nullptr)
            layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(romfs_dir)));

        // Support for romfslite introduced in Atmosphere 1.9.5
        auto romfslite_dir = FindSubdirectoryCaseless(subdir, "romfslite");
        if (romfslite_dir != nullptr)
            layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(romfslite_dir)));

        auto ext_dir = FindSubdirectoryCaseless(subdir, "romfs_ext");
        if (ext_dir != nullptr)
            layers_ext.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(ext_dir)));

        if (type == ContentRecordType::HtmlDocument) {
            auto manual_dir = FindSubdirectoryCaseless(subdir, "manual_html");
            if (manual_dir != nullptr)
                layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(manual_dir)));
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
            // Not listed: Fritchii's built-in Dynamic FPS (listed below) is used instead.
            if (IsBuiltinDynamicFpsMod(title_id, mod)) {
                continue;
            }
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
        if (patch.title_id != title_id ||
            (patch.enable_env != nullptr &&
             std::string_view{std::getenv(patch.enable_env) ? std::getenv(patch.enable_env) : ""} !=
                 "1") ||
            (patch.setting_enabled != nullptr && !patch.setting_enabled())) {
            continue;
        }
        const auto patch_disabled =
            std::find(disabled.begin(), disabled.end(), patch.name) != disabled.end();
        out.push_back({.enabled = !patch_disabled,
                       .name = std::string{patch.name},
                       .version = "IPS",
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

    const auto romfs = PatchRomFS(&nca, base_romfs, ContentRecordType::Control);
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

std::optional<size_t> OotHighRefreshScaleOffset(std::span<const u8> code) {
    const auto parts = ReadOotHighRefreshCode(code);
    if (!parts || GetOotHighRefreshLayout(*parts) != OotHighRefreshLayout::On) {
        return std::nullopt;
    }
    return OotHighRefreshCode::ScaleOffset;
}

std::optional<OotHighRefreshCode> ReadOotHighRefreshCode(std::span<const u8> code) {
    using Code = OotHighRefreshCode;
    static_assert(Code::WaterOffset < Code::CallOffset && Code::CallOffset < Code::CaveOffset);
    if (code.size() < Code::CaveOffset + Code::CaveSize) {
        return std::nullopt;
    }
    Code parts;
    std::ranges::copy(code.subspan(Code::CallOffset, parts.call.size()), parts.call.begin());
    std::ranges::copy(code.subspan(Code::WaterOffset, parts.water.size()), parts.water.begin());
    std::ranges::copy(code.subspan(Code::CaveOffset, parts.cave.size()), parts.cave.begin());
    return parts;
}

OotHighRefreshLayout GetOotHighRefreshLayout(const OotHighRefreshCode& code) {
    if (code.call != DYNAMIC_FPS_CALL) {
        return OotHighRefreshLayout::Unknown;
    }
    if (code.water == WATER_ORIGINAL && code.cave == DYNAMIC_FPS_CAVE) {
        return OotHighRefreshLayout::Off;
    }
    // The cave but the speed scale word, which the present pacing sets (Turbo, Slow).
    const auto cave = std::span<const u8>{code.cave};
    const auto ours = std::span<const u8>{HIGH_REFRESH_CAVE};
    constexpr size_t after_scale = HIGH_REFRESH_SCALE + 4;
    if (code.water == WATER_HOOK &&
        std::ranges::equal(cave.first(HIGH_REFRESH_SCALE), ours.first(HIGH_REFRESH_SCALE)) &&
        std::ranges::equal(cave.subspan(after_scale), ours.subspan(after_scale))) {
        return OotHighRefreshLayout::On;
    }
    return OotHighRefreshLayout::Unknown;
}

std::optional<OotHighRefreshCode> SwitchOotHighRefreshCode(const OotHighRefreshCode& code,
                                                           bool on) {
    const OotHighRefreshLayout layout = GetOotHighRefreshLayout(code);
    if (layout == OotHighRefreshLayout::Unknown) {
        return std::nullopt;
    }
    if ((layout == OotHighRefreshLayout::On) == on) {
        return code;
    }
    OotHighRefreshCode switched = code;
    switched.water = on ? WATER_HOOK : WATER_ORIGINAL;
    switched.cave = on ? HIGH_REFRESH_CAVE : DYNAMIC_FPS_CAVE;
    return switched;
}

bool OotHighRefreshStateInBss(u64 bss_start, u64 bss_end) {
    return HIGH_REFRESH_STATE_OFFSET >= bss_start && HIGH_REFRESH_STATE_OFFSET <= bss_end &&
           HIGH_REFRESH_STATE_SIZE <= bss_end - HIGH_REFRESH_STATE_OFFSET;
}

namespace {

// One attempt, with the application stalled: no guest thread of it runs on any core, and each
// one's saved context is its real state.
OotHighRefreshSwitch SwitchOotHighRefreshStalled(Kernel::KProcess& process, u64 code_address,
                                                 bool on) {
    using Code = OotHighRefreshCode;
    const u64 cave_address = code_address + Code::CaveOffset;
    // No thread may stand inside the cave, where the bytes under it change (it could have been
    // stopped at a block boundary in it). The call and the water hook are single words: either
    // word leads into complete code.
    {
        Kernel::KScopedLightLock lk{process.GetListLock()};
        for (auto& thread : process.GetThreadList()) {
            const u64 pc = thread.GetContext().pc;
            if (pc >= cave_address && pc < cave_address + Code::CaveSize) {
                return OotHighRefreshSwitch::Busy;
            }
        }
    }

    // The present pacing writes the speed scale word under this lock too (it runs on a service
    // thread, which the stall does not stop).
    std::scoped_lock lk{process.GetOotHighRefreshLock()};
    auto& memory = process.GetMemory();
    Code current;
    if (!memory.ReadBlock(code_address + Code::CallOffset, current.call.data(), current.call.size()) ||
        !memory.ReadBlock(code_address + Code::WaterOffset, current.water.data(),
                          current.water.size()) ||
        !memory.ReadBlock(cave_address, current.cave.data(), current.cave.size())) {
        return OotHighRefreshSwitch::UnknownCode;
    }
    const auto target = SwitchOotHighRefreshCode(current, on);
    if (!target) {
        return OotHighRefreshSwitch::UnknownCode;
    }
    if (!on) {
        process.SetOotHighRefreshScaleAddress(0);
        Settings::SetOotHighRefreshPatched(false);
    }
    if (target->cave != current.cave) {
        memory.WriteBlock(cave_address, target->cave.data(), target->cave.size());
        Core::InvalidateInstructionCacheRange(&process, cave_address, target->cave.size());
    }
    if (target->water != current.water) {
        memory.WriteBlock(code_address + Code::WaterOffset, target->water.data(),
                          target->water.size());
        Core::InvalidateInstructionCacheRange(&process, code_address + Code::WaterOffset,
                                              target->water.size());
    }
    if (on) {
        process.SetOotHighRefreshScaleAddress(code_address + Code::ScaleOffset);
        Settings::SetOotHighRefreshPatched(true);
    }
    return OotHighRefreshSwitch::Switched;
}

} // Anonymous namespace

OotHighRefreshSwitch SwitchOotHighRefreshLive(Core::System& system, bool on) {
    constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
    Kernel::KProcess* const process = system.ApplicationProcess();
    if (!system.IsPoweredOn() || system.IsShuttingDown() || process == nullptr ||
        process->GetProgramId() != OcarinaOfTime ||
        process->GetOotMainCodeAddress() == 0) {
        return OotHighRefreshSwitch::NotRunning;
    }
    if (on) {
        const auto it = Settings::values.disabled_addons.find(OcarinaOfTime);
        if (it != Settings::values.disabled_addons.end() &&
            std::ranges::find(it->second, HIGH_REFRESH_NAME) != it->second.end()) {
            return OotHighRefreshSwitch::DisabledInAddOns;
        }
    }
    // A thread stopped inside the cave is all but impossible (a few dozen instructions once a
    // frame); try a few times anyway, letting the game run a moment in between.
    OotHighRefreshSwitch result = OotHighRefreshSwitch::Busy;
    for (int attempt = 0; attempt < 8 && result == OotHighRefreshSwitch::Busy; ++attempt) {
        if (attempt != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{2});
        }
        auto stall = system.StallApplication();
        result = SwitchOotHighRefreshStalled(*process, process->GetOotMainCodeAddress(), on);
        system.UnstallApplication();
    }
    LOG_INFO(Loader, "Ocarina of Time: high refresh rate switched live {}: {}", on ? "on" : "off",
             result == OotHighRefreshSwitch::Switched ? "done"
             : result == OotHighRefreshSwitch::Busy   ? "a thread stayed in the cave"
                                                      : "its code holds neither mode");
    return result;
}

} // namespace FileSys
