// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include "common/common_funcs.h"
#include "common/fs/path_util.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/nso.h"

namespace {

constexpr u64 OcarinaTitleId = 0x0400486024DEC000;
constexpr u64 OtherTitleId = 0x0100000000010000;
constexpr std::string_view MaxResolution = "Max Resolution (built-in)";
constexpr std::string_view OcarinaBuildIdHex = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348";
constexpr FileSys::PatchManager::BuildID OcarinaBuildId{
    0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0,
    0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48,
};
constexpr size_t PatchOffset = sizeof(Loader::NSOHeader) + 0x805BD0;
constexpr u32 TbzInstruction = 0x360041E8;
constexpr u32 BranchInstruction = 0x1400020F;
constexpr u32 NopInstruction = 0xD503201F;

constexpr std::array<std::pair<Common::FS::EdenPath, const char*>, 6> TestDirectories{{
    {Common::FS::EdenPath::NANDDir, "nand"},
    {Common::FS::EdenPath::SDMCDir, "sdmc"},
    {Common::FS::EdenPath::LoadDir, "load"},
    {Common::FS::EdenPath::DumpDir, "dump"},
    {Common::FS::EdenPath::KeysDir, "keys"},
    {Common::FS::EdenPath::ConfigDir, "config"},
}};

void WriteIps(const std::filesystem::path& path, u32 word) {
    std::vector<u8> ips{'P', 'A', 'T', 'C', 'H'};
    ips.push_back(static_cast<u8>(PatchOffset >> 16));
    ips.push_back(static_cast<u8>(PatchOffset >> 8));
    ips.push_back(static_cast<u8>(PatchOffset));
    ips.push_back(0);
    ips.push_back(sizeof(word));
    for (size_t i = 0; i < sizeof(word); ++i) {
        ips.push_back(static_cast<u8>(word >> (i * 8)));
    }
    ips.insert(ips.end(), {'E', 'O', 'F'});
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<const char*>(ips.data()), static_cast<std::streamsize>(ips.size()));
}

// An IPS (24-bit offsets) or IPS32 file with one record: `bytes` at `offset` of the NSO file.
std::vector<u8> MakeIps(bool ips32, u32 offset, std::span<const u8> bytes) {
    std::vector<u8> ips;
    if (ips32) {
        ips = {'I', 'P', 'S', '3', '2', static_cast<u8>(offset >> 24)};
    } else {
        ips = {'P', 'A', 'T', 'C', 'H'};
    }
    ips.insert(ips.end(), {static_cast<u8>(offset >> 16), static_cast<u8>(offset >> 8),
                           static_cast<u8>(offset), static_cast<u8>(bytes.size() >> 8),
                           static_cast<u8>(bytes.size())});
    ips.insert(ips.end(), bytes.begin(), bytes.end());
    if (ips32) {
        ips.push_back('E');
    }
    ips.insert(ips.end(), {'E', 'O', 'F'});
    return ips;
}

class PatchManagerTestContext {
public:
    explicit PatchManagerTestContext(std::optional<u32> user_mod_word = std::nullopt) {
        if (const auto it = Settings::values.disabled_addons.find(OcarinaTitleId);
            it != Settings::values.disabled_addons.end()) {
            previous_disabled = it->second;
        }
        for (size_t i = 0; i < TestDirectories.size(); ++i) {
            const auto& [path, name] = TestDirectories[i];
            previous[i] = Common::FS::GetEdenPath(path);
            std::filesystem::create_directories(root / name);
            Common::FS::SetEdenPath(path, root / name);
        }
        const auto title_dir = root / "load" / fmt::format("{:016X}", OcarinaTitleId);
        std::filesystem::create_directories(title_dir);
        if (user_mod_word) {
            const auto exefs = title_dir / "UserMod" / "exefs";
            std::filesystem::create_directories(exefs);
            WriteIps(exefs / fmt::format("{}.ips", OcarinaBuildIdHex), *user_mod_word);
        }
        system.emplace();
        system->SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
        system->SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
        system->GetFileSystemController().CreateFactories(*system->GetFilesystem());
    }

    ~PatchManagerTestContext() {
        system.reset();
        for (size_t i = 0; i < TestDirectories.size(); ++i) {
            Common::FS::SetEdenPath(TestDirectories[i].first, previous[i]);
        }
        if (previous_disabled) {
            Settings::values.disabled_addons[OcarinaTitleId] = std::move(*previous_disabled);
        } else {
            Settings::values.disabled_addons.erase(OcarinaTitleId);
        }
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    PatchManagerTestContext(const PatchManagerTestContext&) = delete;
    PatchManagerTestContext& operator=(const PatchManagerTestContext&) = delete;

    FileSys::PatchManager Manager(u64 title_id = OcarinaTitleId) {
        return FileSys::PatchManager{title_id, system->GetFileSystemController(),
                                     system->GetContentProvider()};
    }

    void AddUserBytePatch(u32 offset, u8 value) {
        const auto exefs = root / "load" / fmt::format("{:016X}", OcarinaTitleId) / "UserMod" / "exefs";
        std::filesystem::create_directories(exefs);
        const std::array<u8, 16> ips{
            'I', 'P', 'S', '3', '2',
            static_cast<u8>(offset >> 24), static_cast<u8>(offset >> 16),
            static_cast<u8>(offset >> 8), static_cast<u8>(offset),
            0, 1, value, 'E', 'E', 'O', 'F',
        };
        std::ofstream file{exefs / fmt::format("{}.ips", OcarinaBuildIdHex), std::ios::binary};
        file.write(reinterpret_cast<const char*>(ips.data()), static_cast<std::streamsize>(ips.size()));
        file.flush();
        REQUIRE(file.good());
    }

    // Writes `contents` as <load>/<Ocarina>/<mod>/exefs/<file_name>; returns its path.
    std::filesystem::path AddModFile(std::string_view mod, std::string_view file_name,
                                     std::span<const u8> contents) {
        const auto exefs =
            root / "load" / fmt::format("{:016X}", OcarinaTitleId) / std::string{mod} / "exefs";
        std::filesystem::create_directories(exefs);
        const auto path = exefs / std::string{file_name};
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<const char*>(contents.data()),
                   static_cast<std::streamsize>(contents.size()));
        file.flush();
        REQUIRE(file.good());
        return path;
    }

private:
    std::array<std::filesystem::path, TestDirectories.size()> previous;
    std::optional<std::vector<std::string>> previous_disabled;
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("builtin-patch-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::optional<Core::System> system;
};

std::vector<u8> MakeMain(const FileSys::PatchManager::BuildID& build_id, u32 word) {
    std::vector<u8> nso(PatchOffset + 0x100);
    Loader::NSOHeader header{};
    header.magic = Common::MakeMagic('N', 'S', 'O', '0');
    header.build_id = build_id;
    std::memcpy(nso.data(), &header, sizeof(header));
    std::memcpy(nso.data() + PatchOffset, &word, sizeof(word));
    return nso;
}

u32 WordAt(const std::vector<u8>& nso) {
    REQUIRE(nso.size() >= PatchOffset + sizeof(u32));
    u32 word;
    std::memcpy(&word, nso.data() + PatchOffset, sizeof(word));
    return word;
}

u32 PatchMain(PatchManagerTestContext& context, const FileSys::PatchManager::BuildID& build_id,
              u32 word, const std::string& name = "main", u64 title_id = OcarinaTitleId) {
    return WordAt(context.Manager(title_id).PatchNSO(MakeMain(build_id, word), name));
}

constexpr size_t OcarinaDark = 0x3C46200; // Dark's routine
constexpr size_t OcarinaCave = 0x3C46E00; // ours, at the free end of that page
constexpr size_t OcarinaWaterHook = 0x475F8;
constexpr u32 WaterFcvtzs = 0x1E380009; // fcvtzs w9, s0

// Ocarina's main as the high refresh rate tests need it: the water renderer's fcvtzs (the ripple
// hook's original), a BSS that holds the routine's state, and zero bytes from Dark's routine to
// the end of its page.
std::vector<u8> MakeOcarinaMain() {
    constexpr size_t Code = sizeof(Loader::NSOHeader);
    auto nso = MakeMain(OcarinaBuildId, NopInstruction);
    nso.resize(Code + OcarinaCave + 0x200);
    std::memcpy(nso.data() + Code + OcarinaWaterHook, &WaterFcvtzs, sizeof(WaterFcvtzs));
    Loader::NSOHeader header;
    std::memcpy(&header, nso.data(), sizeof(header));
    header.segments[2].location = 0x4000000;
    header.segments[2].size = 0x100000;
    header.segments[2].bss_size = 0x1000000;
    std::memcpy(nso.data(), &header, sizeof(header));
    return nso;
}

// Writes `parts` into `code` where they belong (what the live switch writes to guest memory).
void WriteOotHighRefreshCode(std::span<u8> code, const FileSys::OotHighRefreshCode& parts) {
    using Parts = FileSys::OotHighRefreshCode;
    std::ranges::copy(parts.call, code.begin() + Parts::CallOffset);
    std::ranges::copy(parts.water, code.begin() + Parts::WaterOffset);
    std::ranges::copy(parts.dark, code.begin() + Parts::DarkOffset);
    std::ranges::copy(parts.cave, code.begin() + Parts::CaveOffset);
}

} // Anonymous namespace

TEST_CASE("Built-in Max Resolution patch is gated on build id and original word",
          "[patch_manager]") {
    PatchManagerTestContext context;

    SECTION("Matching build id and original word applies") {
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == BranchInstruction);
    }

    SECTION("Wrong original word skips") {
        REQUIRE(PatchMain(context, OcarinaBuildId, NopInstruction) == NopInstruction);
    }

    SECTION("Other build id skips") {
        auto other = OcarinaBuildId;
        other[0] ^= 1;
        REQUIRE(PatchMain(context, other, TbzInstruction) == TbzInstruction);
    }

    SECTION("Other module skips") {
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction, "sdk") == TbzInstruction);
    }

    SECTION("Other title skips") {
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction, "main", OtherTitleId) ==
                TbzInstruction);
    }

    SECTION("Disabled in Add-Ons skips") {
        Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == TbzInstruction);
    }
}

TEST_CASE("Built-in Max Resolution patch defers to a user mod at the same offset",
          "[patch_manager]") {
    SECTION("A different word from the mod is kept") {
        PatchManagerTestContext context{NopInstruction};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == NopInstruction);
    }

    SECTION("The same word from the mod is applied once") {
        PatchManagerTestContext context{BranchInstruction};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == BranchInstruction);
    }

    SECTION("Disabling the built-in patch keeps the mod") {
        PatchManagerTestContext context{BranchInstruction};
        Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
        REQUIRE(PatchMain(context, OcarinaBuildId, TbzInstruction) == BranchInstruction);
    }
}

TEST_CASE("Built-in Max Resolution patch is listed and loaded for Ocarina's main",
          "[patch_manager]") {
    PatchManagerTestContext context;
    const auto manager = context.Manager();

    REQUIRE(manager.HasNSOPatch(OcarinaBuildId, "main"));
    REQUIRE_FALSE(manager.HasNSOPatch(OcarinaBuildId, "sdk"));
    REQUIRE_FALSE(context.Manager(OtherTitleId).HasNSOPatch(OcarinaBuildId, "main"));

    const auto builtin_enabled = [&] {
        const auto patches = manager.GetPatches();
        const auto it = std::ranges::find(patches, MaxResolution, &FileSys::Patch::name);
        REQUIRE(it != patches.end());
        REQUIRE(it->type == FileSys::PatchType::Mod);
        return it->enabled;
    };
    REQUIRE(builtin_enabled());

    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    REQUIRE_FALSE(builtin_enabled());
}

TEST_CASE("Built-in patches preserve user overrides on already-patched input", "[patch_manager]") {
    PatchManagerTestContext context;
    constexpr size_t CallOffset = sizeof(Loader::NSOHeader) + 0x6E6CC4;
    constexpr u32 RoutineLastByte = sizeof(Loader::NSOHeader) + 0x3C46200 + 119;
    constexpr std::array<u8, 4> DynamicCall{0x4F, 0x7D, 0xD5, 0x94};

    auto nso = MakeMain(OcarinaBuildId, TbzInstruction);
    nso.resize(RoutineLastByte + 1);
    nso = context.Manager().PatchNSO(nso, "main");
    REQUIRE(nso.size() == RoutineLastByte + 1);
    REQUIRE(std::equal(DynamicCall.begin(), DynamicCall.end(), nso.begin() + CallOffset));
    REQUIRE(nso[RoutineLastByte] == 0x3F);
    REQUIRE(WordAt(nso) == BranchInstruction);

    // Over Max Resolution's branch (a mod writing into Dark's routine is his standalone Dynamic
    // FPS, which the built-in replaces: see below).
    constexpr u8 UserByte = 0x5A;
    context.AddUserBytePatch(PatchOffset, UserByte);
    auto expected = nso;
    expected[PatchOffset] = UserByte;
    const auto patched = context.Manager().PatchNSO(nso, "main");
    REQUIRE(std::ranges::equal(patched, expected));
}

TEST_CASE("High refresh rate built-in calls its own routine only while its setting is on",
          "[patch_manager]") {
    constexpr std::string_view HighRefresh = "High refresh rate (built-in, experimental)";
    constexpr std::string_view DynamicFps = "Dynamic FPS by Dark (built-in)";
    constexpr size_t Code = sizeof(Loader::NSOHeader);
    constexpr size_t Dark = Code + OcarinaDark;
    constexpr size_t Routine = Code + OcarinaCave;
    constexpr size_t CallOffset = Code + 0x6E6CC4;
    constexpr u32 DynamicCall = 0x94D57D4F;     // bl 0x3c46200 (Dark's routine)
    constexpr u32 HighRefreshCall = 0x94D5804F; // bl 0x3c46e00 (ours)
    constexpr u32 DarkThreshold = 0x3F83D70A;   // Dark's routine +0x74: 1.03f
    const auto word_at = [](const std::vector<u8>& nso, size_t offset) {
        u32 word;
        std::memcpy(&word, nso.data() + offset, sizeof(word));
        return word;
    };
    const auto listed = [](const FileSys::PatchManager& manager, std::string_view name) {
        const auto patches = manager.GetPatches();
        return std::ranges::find(patches, name, &FileSys::Patch::name) != patches.end();
    };
    // Dark's Dynamic FPS: his call and routine, not the high refresh mode (the water ripple
    // built-in may or may not be there next to it).
    const auto dark_only = [&](const std::vector<u8>& patched) {
        return word_at(patched, CallOffset) == DynamicCall &&
               word_at(patched, Dark + 0x74) == DarkThreshold &&
               !FileSys::OotHighRefreshScaleOffset(std::span{patched}.subspan(Code));
    };
    struct RestoreSetting {
        ~RestoreSetting() {
            Settings::values.oot_high_refresh.SetValue(false);
        }
    } restore;

    PatchManagerTestContext context;
    auto nso = MakeOcarinaMain();

    SECTION("Off: Dark's routine, with the water ripple built-in next to it, high refresh not listed") {
        Settings::values.oot_high_refresh.SetValue(false);
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(dark_only(patched));
        for (size_t offset = Dark + 0x78; offset < Routine; offset += 4) {
            REQUIRE(word_at(patched, offset) == 0);
        }
        // v5.8: the ripple accumulator and its hook with Dark's mode too (zortos293/Lichii#1180).
        REQUIRE(word_at(patched, Code + OcarinaWaterHook) == 0x14EFFE26); // b 0x3c46e90
        REQUIRE(word_at(patched, Routine + 0x90) == 0xB00092C9);          // adrp x9, 0x4e9f000
        REQUIRE(word_at(patched, Routine + 0x10C) == 0x171001BC);         // b 0x475fc
        for (size_t offset = 0x110; offset < 0x200; offset += 4) {
            REQUIRE(word_at(patched, Routine + offset) == 0);
        }
        REQUIRE_FALSE(listed(context.Manager(), HighRefresh));
        REQUIRE(listed(context.Manager(), DynamicFps));
        REQUIRE(listed(context.Manager(), "Water ripples at any frame rate (built-in)"));
    }

    SECTION("Off, with the water ripple built-in off in Add-Ons: Dark's routine alone, as before") {
        Settings::values.oot_high_refresh.SetValue(false);
        Settings::values.disabled_addons[OcarinaTitleId] = {
            std::string{"Water ripples at any frame rate (built-in)"}};
        const auto patched = context.Manager().PatchNSO(nso, "main");
        Settings::values.disabled_addons.erase(OcarinaTitleId);
        REQUIRE(dark_only(patched));
        REQUIRE(word_at(patched, Code + OcarinaWaterHook) == WaterFcvtzs);
        for (size_t offset = Dark + 0x78; offset < Routine + 0x200; offset += 4) {
            REQUIRE(word_at(patched, offset) == 0);
        }
    }

    SECTION("On: the call to the high refresh routine, with its speed scale word") {
        Settings::values.oot_high_refresh.SetValue(true);
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(word_at(patched, CallOffset) == HighRefreshCall);
        // Dark's routine stays, nothing after it (another mod's spot).
        REQUIRE(word_at(patched, Dark + 0x74) == DarkThreshold);
        for (size_t offset = Dark + 0x78; offset < Routine; offset += 4) {
            REQUIRE(word_at(patched, offset) == 0);
        }
        REQUIRE(word_at(patched, Routine + 0x18) == 0xB40002CC); // cbz x12, +0x70 (Dark's)
        REQUIRE(word_at(patched, Routine + 0x6C) == 0x14000002); // b +0x74
        REQUIRE(word_at(patched, Routine + 0x70) == 0xD65F03C0); // ret
        REQUIRE(word_at(patched, Routine + 0x78) == 0x1E230848); // fmul s8, s2, s3
        REQUIRE(word_at(patched, Routine + 0x84) == 0x40400000); // 3.0f
        REQUIRE(word_at(patched, Routine + 0x88) == 0x3F800000); // 1.0f
        REQUIRE(word_at(patched, Routine + 0x8C) == 0);
        // The water ripple accumulator (zortos293/Lichii#1152) right after it, and its hook.
        REQUIRE(word_at(patched, Code + OcarinaWaterHook) == 0x14EFFE26); // b 0x3c46e90
        REQUIRE(word_at(patched, Routine + 0x90) == 0xB00092C9);  // adrp x9, 0x4e9f000
        REQUIRE(word_at(patched, Routine + 0x94) == 0x91388129);  // add x9, x9, #0xe20
        REQUIRE(word_at(patched, Routine + 0x108) == 0x1E380029); // fcvtzs w9, s1
        REQUIRE(word_at(patched, Routine + 0x10C) == 0x171001BC); // b 0x475fc
        for (size_t offset = 0x110; offset < 0x200; offset += 4) {
            REQUIRE(word_at(patched, Routine + offset) == 0);
        }
        REQUIRE(FileSys::OotHighRefreshScaleOffset(std::span{patched}.subspan(Code)) ==
                std::optional<size_t>{0x3C46E88});
        REQUIRE(listed(context.Manager(), HighRefresh));
    }

    SECTION("On, with another mod's code right after Dark's routine: still on") {
        // The Hyrule Cheat Menu writes main+0x3C46280..0x3C46398 (2026-11-10).
        Settings::values.oot_high_refresh.SetValue(true);
        nso[Dark + 0x80] = 1;
        nso[Dark + 0x190] = 1;
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(word_at(patched, CallOffset) == HighRefreshCall);
        REQUIRE(patched[Dark + 0x80] == 1);
        REQUIRE(patched[Dark + 0x190] == 1);
        REQUIRE(FileSys::OotHighRefreshScaleOffset(std::span{patched}.subspan(Code)) ==
                std::optional<size_t>{0x3C46E88});
    }

    SECTION("On, with an accumulator word in use: Dark's routine") {
        Settings::values.oot_high_refresh.SetValue(true);
        nso[Routine + 0x10C] = 1;
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(word_at(patched, CallOffset) == DynamicCall);
        REQUIRE(word_at(patched, Code + OcarinaWaterHook) == WaterFcvtzs);
        REQUIRE(patched[Routine + 0x10C] == 1);
        REQUIRE_FALSE(FileSys::OotHighRefreshScaleOffset(std::span{patched}.subspan(Code)));
    }

    SECTION("On, with another word at the water hook: Dark's routine, no ripple built-in") {
        Settings::values.oot_high_refresh.SetValue(true);
        std::memcpy(nso.data() + Code + OcarinaWaterHook, &NopInstruction, sizeof(u32));
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(dark_only(patched));
        REQUIRE(word_at(patched, Code + OcarinaWaterHook) == NopInstruction);
        REQUIRE(word_at(patched, Routine + 0x90) == 0);
    }

    SECTION("On, with the state past the end of BSS: Dark's routine") {
        Settings::values.oot_high_refresh.SetValue(true);
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[2].bss_size = 0x4E9FE00 + 0x9F - 0x4100000; // one byte short
        std::memcpy(nso.data(), &header, sizeof(header));
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(dark_only(patched));
        REQUIRE(word_at(patched, Code + OcarinaWaterHook) == WaterFcvtzs);
    }

    SECTION("On, but off in Add-Ons: Dark's routine") {
        Settings::values.oot_high_refresh.SetValue(true);
        Settings::values.disabled_addons[OcarinaTitleId] = {std::string{HighRefresh}};
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(dark_only(patched));
    }
}

TEST_CASE("High refresh rate mode switches live between the two boot layouts",
          "[patch_manager]") {
    using FileSys::OotHighRefreshLayout;
    constexpr size_t Code = sizeof(Loader::NSOHeader);
    constexpr size_t Scale = 0x88; // the speed scale word, in the cave
    struct RestoreSetting {
        ~RestoreSetting() {
            Settings::values.oot_high_refresh.SetValue(false);
        }
    } restore;

    // What boots with the setting off and on.
    PatchManagerTestContext context;
    const auto nso = MakeOcarinaMain();
    Settings::values.oot_high_refresh.SetValue(false);
    const auto off = context.Manager().PatchNSO(nso, "main");
    Settings::values.oot_high_refresh.SetValue(true);
    const auto on = context.Manager().PatchNSO(nso, "main");
    REQUIRE(off != on);
    const auto off_parts = FileSys::ReadOotHighRefreshCode(std::span<const u8>{off}.subspan(Code));
    const auto on_parts = FileSys::ReadOotHighRefreshCode(std::span<const u8>{on}.subspan(Code));
    REQUIRE(off_parts);
    REQUIRE(on_parts);
    REQUIRE(FileSys::GetOotHighRefreshLayout(*off_parts) == OotHighRefreshLayout::Off);
    REQUIRE(FileSys::GetOotHighRefreshLayout(*on_parts) == OotHighRefreshLayout::On);

    SECTION("Switching writes exactly what boots in the other mode, both ways") {
        auto module = off;
        const auto switched_on = FileSys::SwitchOotHighRefreshCode(*off_parts, true);
        REQUIRE(switched_on);
        WriteOotHighRefreshCode(std::span{module}.subspan(Code), *switched_on);
        REQUIRE(module == on);
        REQUIRE(FileSys::OotHighRefreshScaleOffset(std::span<const u8>{module}.subspan(Code)) ==
                std::optional<size_t>{0x3C46E88});

        const auto switched_off = FileSys::SwitchOotHighRefreshCode(*switched_on, false);
        REQUIRE(switched_off);
        WriteOotHighRefreshCode(std::span{module}.subspan(Code), *switched_off);
        REQUIRE(module == off);
        REQUIRE_FALSE(FileSys::OotHighRefreshScaleOffset(std::span<const u8>{module}.subspan(Code)));
    }

    SECTION("Switching to the mode it is in changes nothing, the speed scale word kept") {
        auto turbo = *on_parts;
        constexpr std::array<u8, 4> TwoPointZero{0x00, 0x00, 0x00, 0x40}; // 2.0f
        std::ranges::copy(TwoPointZero, turbo.cave.begin() + Scale);
        REQUIRE(FileSys::GetOotHighRefreshLayout(turbo) == OotHighRefreshLayout::On);
        const auto still_on = FileSys::SwitchOotHighRefreshCode(turbo, true);
        REQUIRE(still_on);
        REQUIRE(still_on->cave == turbo.cave);
        REQUIRE(still_on->water == turbo.water);

        const auto still_off = FileSys::SwitchOotHighRefreshCode(*off_parts, false);
        REQUIRE(still_off);
        REQUIRE(still_off->cave == off_parts->cave);
        REQUIRE(still_off->water == off_parts->water);

        // Off from any speed scale: Dark's cave, the scale word zero again.
        const auto from_turbo = FileSys::SwitchOotHighRefreshCode(turbo, false);
        REQUIRE(from_turbo);
        REQUIRE(from_turbo->cave == off_parts->cave);
        REQUIRE(from_turbo->water == off_parts->water);
    }

    SECTION("Code in neither mode is never switched") {
        const auto refused = [](const FileSys::OotHighRefreshCode& parts) {
            return FileSys::GetOotHighRefreshLayout(parts) == OotHighRefreshLayout::Unknown &&
                   !FileSys::SwitchOotHighRefreshCode(parts, true) &&
                   !FileSys::SwitchOotHighRefreshCode(parts, false);
        };
        auto used_cave = *off_parts; // a word of our cave in use (another mod's code)
        used_cave.cave[0x100] = 1;
        REQUIRE(refused(used_cave));
        auto changed_dark = *on_parts; // Dark's routine changed: switching off would call it
        changed_dark.dark[0x40] ^= 1;
        REQUIRE(refused(changed_dark));
        auto stray_scale = *off_parts; // Off holds no speed scale word
        stray_scale.cave[Scale + 3] = 0x3F;
        REQUIRE(refused(stray_scale));
        auto no_call = *off_parts; // Dark's routine off in Add-Ons: the game's own fdiv
        no_call.call = {0x28, 0x18, 0x20, 0x1E};
        REQUIRE(refused(no_call));
        auto modded_water = *on_parts; // a mod's word at the water hook
        modded_water.water = {0x1F, 0x20, 0x03, 0xD5};
        REQUIRE(refused(modded_water));
        auto off_water_hooked = *off_parts; // Dark's cave with the hook: half switched
        off_water_hooked.water = on_parts->water;
        REQUIRE(refused(off_water_hooked));
        auto half_written = *off_parts; // the first half of the high refresh cave
        std::copy_n(on_parts->cave.begin(), 0x80, half_written.cave.begin());
        REQUIRE(refused(half_written));
        auto changed_routine = *on_parts;
        changed_routine.cave[0x5C] ^= 1;
        REQUIRE(refused(changed_routine));
        auto changed_accumulator = *on_parts;
        changed_accumulator.cave[0x10C] ^= 1;
        REQUIRE(refused(changed_accumulator));
    }

    SECTION("A module too small for the cave holds neither") {
        const auto cut = std::span<const u8>{off}.subspan(Code, OcarinaCave + 0x10F);
        REQUIRE_FALSE(FileSys::ReadOotHighRefreshCode(cut));
        REQUIRE_FALSE(FileSys::OotHighRefreshScaleOffset(cut));
    }

    SECTION("Its state (Dark's timer, the ripple slots, 0x4E9FE00..0x4E9FEA0) must lie in BSS") {
        REQUIRE(FileSys::OotHighRefreshStateInBss(0x4100000, 0x5100000));
        REQUIRE(FileSys::OotHighRefreshStateInBss(0x4E9FE00, 0x4E9FEA0));
        REQUIRE_FALSE(FileSys::OotHighRefreshStateInBss(0x4E9FE01, 0x5100000));
        REQUIRE_FALSE(FileSys::OotHighRefreshStateInBss(0x4100000, 0x4E9FE9F));
        REQUIRE_FALSE(FileSys::OotHighRefreshStateInBss(0, 0));
    }
}

TEST_CASE("Dark's standalone Dynamic FPS mod is skipped and hidden for the built-in",
          "[patch_manager]") {
    constexpr std::string_view DynamicFps = "Dynamic FPS by Dark (built-in)";
    constexpr size_t Code = sizeof(Loader::NSOHeader);
    constexpr u32 CallOffset = Code + 0x6E6CC4;
    constexpr std::array<u8, 4> DynamicCall{0x4F, 0x7D, 0xD5, 0x94};
    constexpr std::array<u8, 4> Nop{0x1F, 0x20, 0x03, 0xD5};
    const auto ips_name = fmt::format("{}.ips", OcarinaBuildIdHex);
    const auto listed = [](const FileSys::PatchManager& manager, std::string_view name) {
        const auto patches = manager.GetPatches();
        return std::ranges::find(patches, name, &FileSys::Patch::name) != patches.end();
    };

    PatchManagerTestContext context;
    auto nso = MakeOcarinaMain();
    std::memcpy(nso.data() + PatchOffset, &TbzInstruction, sizeof(TbzInstruction));
    // What boots without mods: Dark's call and Max Resolution's branch.
    const auto builtin = context.Manager().PatchNSO(nso, "main");
    REQUIRE(std::equal(DynamicCall.begin(), DynamicCall.end(), builtin.begin() + CallOffset));
    REQUIRE(WordAt(builtin) == BranchInstruction);

    SECTION("An IPS writing Dark's call: skipped, not listed, the built-in applied") {
        const auto ips = MakeIps(false, CallOffset, Nop);
        const auto path = context.AddModFile("Dynamic FPS", ips_name, ips);
        REQUIRE(context.Manager().PatchNSO(nso, "main") == builtin);
        REQUIRE_FALSE(listed(context.Manager(), "Dynamic FPS"));
        REQUIRE(listed(context.Manager(), DynamicFps));
        // The player's files stay as they were.
        REQUIRE(std::filesystem::file_size(path) == ips.size());
    }

    SECTION("An IPSwitch patch writing into Dark's routine: the same") {
        constexpr std::string_view Pchtxt = "@nsobid-E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348\n"
                                            "@flag offset_shift 0x100\n"
                                            "@enabled\n"
                                            "03C46274 00000000\n"
                                            "@stop\n";
        context.AddModFile("Dynamic FPS 60", "dynamic_fps.pchtxt",
                           {reinterpret_cast<const u8*>(Pchtxt.data()), Pchtxt.size()});
        REQUIRE(context.Manager().PatchNSO(nso, "main") == builtin);
        REQUIRE_FALSE(listed(context.Manager(), "Dynamic FPS 60"));
        REQUIRE(listed(context.Manager(), DynamicFps));
    }

    SECTION("A mod patching an unrelated offset still applies and is listed") {
        constexpr u32 Unrelated = Code + 0x1000;
        context.AddModFile("Other mod", ips_name, MakeIps(true, Unrelated, Nop));
        auto expected = builtin;
        std::ranges::copy(Nop, expected.begin() + Unrelated);
        REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
        REQUIRE(listed(context.Manager(), "Other mod"));
    }

    SECTION("Next to an unrelated mod, only the Dynamic FPS mod is left out") {
        constexpr u32 Unrelated = Code + 0x1000;
        context.AddModFile("Dynamic FPS", ips_name, MakeIps(false, CallOffset, Nop));
        context.AddModFile("Other mod", ips_name, MakeIps(true, Unrelated, Nop));
        auto expected = builtin;
        std::ranges::copy(Nop, expected.begin() + Unrelated);
        REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
        REQUIRE(listed(context.Manager(), "Other mod"));
        REQUIRE_FALSE(listed(context.Manager(), "Dynamic FPS"));
    }

    SECTION("A mod patching Max Resolution's word still wins over that built-in") {
        context.AddModFile("Other mod", ips_name, MakeIps(true, PatchOffset, Nop));
        const auto patched = context.Manager().PatchNSO(nso, "main");
        REQUIRE(WordAt(patched) == NopInstruction);
        REQUIRE(std::equal(DynamicCall.begin(), DynamicCall.end(), patched.begin() + CallOffset));
        REQUIRE(listed(context.Manager(), "Other mod"));
    }
}
