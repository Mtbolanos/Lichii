// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include "common/common_funcs.h"
#include "common/fs/path_util.h"
#include "common/hex_util.h"
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
constexpr std::string_view DynamicFps = "Dynamic FPS by Dark (built-in)";
constexpr std::string_view FpsUnlock = "FPS Unlock (built-in)";
constexpr std::string_view OcarinaBuildIdHex = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348";
constexpr FileSys::PatchManager::BuildID OcarinaBuildId{
    0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0,
    0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48,
};
constexpr size_t PatchOffset = sizeof(Loader::NSOHeader) + 0x805BD0;
constexpr u32 TbzInstruction = 0x360041E8;
constexpr u32 BranchInstruction = 0x1400020F;
constexpr u32 NopInstruction = 0xD503201F;
constexpr size_t DynamicCallOffset = sizeof(Loader::NSOHeader) + 0x6E6CC4;
constexpr size_t DynamicRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46200;
constexpr size_t DynamicStateOffset = sizeof(Loader::NSOHeader) + 0x4E9FE00;
constexpr size_t DynamicRoutineSize = 120;
constexpr size_t UnlockRoutineSize = 128;
constexpr size_t DynamicStateSize = 0x98;
constexpr size_t WaterHookOffset = sizeof(Loader::NSOHeader) + 0x475F8;
constexpr size_t WaterMulOffset = sizeof(Loader::NSOHeader) + 0x4760C;
constexpr size_t WaterRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46300;
constexpr size_t WaterRoutineSize = 148;
constexpr u32 WaterOriginalInstruction = 0x1E380009;
constexpr u32 WaterMulOriginalInstruction = 0x1B097D08;
constexpr u32 WaterCallInstruction = 0x94EFFB42;
constexpr std::string_view WaterRoutine =
    "0101221E0008211E0820201E0D040054C99200B0294138912A010291EB0309AA6C0140F9"
    "9F0115EB600100546B4100917F010AEB61FFFF544C0140B92B512C8B750100F97F0900B9"
    "8C0500118C0900124C0100B9610940BD0028211E01102C1E0228211E4800381E1F050071"
    "4A000054280080520101221E0038211E01103C1E0048211E600900BDC0035FD6E8031F2A"
    "C0035FD6";
constexpr size_t SoundEntryOffset = sizeof(Loader::NSOHeader) + 0x415300;
constexpr size_t SoundStepOffset = sizeof(Loader::NSOHeader) + 0x415334;
constexpr size_t SoundGateOffset = sizeof(Loader::NSOHeader) + 0x3C46800;
constexpr size_t SoundFeedOffset = sizeof(Loader::NSOHeader) + 0x3C46280;
constexpr u32 SoundEntryOriginalInstruction = 0xD503237F; // pacibsp
constexpr u32 SoundEntryBranchInstruction = 0x14E0C540;   // b main+0x3C46800
constexpr u32 SoundStepOriginalInstruction = 0xBD401100;  // ldr s0, [x8, #0x10]
constexpr u32 SoundStepCallInstruction = 0x94E0C3D3;      // bl main+0x3C46280
constexpr std::string_view SoundGate =
    "D09200B010823A9102030035C9810090294944F9290140F9301140BD11102E1E0022311EAA010054"
    "120240BD522A301E4022311ECB000054523A311E120200BD29008052090600B90A000014120200BD"
    "1F0600B9C0035FD61F2003D529008052090600B903000014090640B9690000347F2303D5A43A1F17"
    "C0035FD6";
constexpr std::string_view SoundFeed = "001140BD10102E1E0068301EC0035FD6";
constexpr u32 DynamicCallInstruction = 0x94D57D4F;
constexpr u32 DynamicOriginalInstruction = 0x1E201828;
constexpr size_t TurnSpeedHookOffset = sizeof(Loader::NSOHeader) + 0x261EB18;
constexpr u32 TurnSpeedOriginalInstruction = 0x1E210801;
constexpr u32 TurnSpeedHookInstruction = 0x14589FCA;
// The second instruction the routine replaces; the hook guards it and leaves it in place.
constexpr u32 TurnSpeedSecondInstruction = 0x1E280800;
constexpr size_t TurnSpeedRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46A40;
constexpr size_t TurnSpeedRoutineSize = 40;
constexpr size_t MovieFramesHookOffset = sizeof(Loader::NSOHeader) + 0x3B5C7C8;
constexpr u32 MovieFramesOriginalInstruction = 0x11000508; // add w8, w8, #1
constexpr u32 MovieFramesHookInstruction = 0x1403A8AA;     // b main+0x3C46A70
constexpr size_t MovieFramesRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46A70;
constexpr size_t MovieFramesRoutineSize = 56;
constexpr u32 ScarecrowCutOriginalInstruction = 0x1E390008; // fcvtzu w8, s0
constexpr size_t ScarecrowRecordHookOffset = sizeof(Loader::NSOHeader) + 0x3296B08;
constexpr u32 ScarecrowRecordHookInstruction = 0x1426BFEA; // b main+0x3C46AB0
constexpr size_t ScarecrowRecordRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46AB0;
constexpr size_t ScarecrowLastHookOffset = sizeof(Loader::NSOHeader) + 0x32926C8;
constexpr u32 ScarecrowLastHookInstruction = 0x1426D104; // b main+0x3C46AD8
constexpr size_t ScarecrowLastRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46AD8;
constexpr size_t ScarecrowRoutineSize = 40;

constexpr size_t StepRoutineOffset = sizeof(Loader::NSOHeader) + 0x3C46900;
constexpr size_t StepStateSize = 0x1C0;
constexpr size_t HudHookOffset = sizeof(Loader::NSOHeader) + 0xA68300;
constexpr u32 HudOriginalInstruction = 0x1E38000C;
constexpr u32 HudHookInstruction = 0x14C77980;
constexpr size_t ReplayHookOffset = sizeof(Loader::NSOHeader) + 0x3293410;
constexpr u32 ReplayOriginalInstruction = 0x1E390008;
constexpr u32 ReplayHookInstruction = 0x1426CD46;
constexpr size_t HitstopHookOffset = sizeof(Loader::NSOHeader) + 0x2F53268;
constexpr u32 HitstopOriginalInstruction = 0x5EA1B900;
constexpr u32 HitstopHookInstruction = 0x1433CDBA;
constexpr size_t AmbientHookOffset = sizeof(Loader::NSOHeader) + 0x53F914;
constexpr u32 AmbientOriginalInstruction = 0x1100077A;
constexpr u32 AmbientHookInstruction = 0x14DC1C3B;

constexpr std::array<std::pair<Common::FS::EdenPath, const char*>, 4> TestDirectories{{
    {Common::FS::EdenPath::NANDDir, "nand"},
    {Common::FS::EdenPath::SDMCDir, "sdmc"},
    {Common::FS::EdenPath::LoadDir, "load"},
    {Common::FS::EdenPath::DumpDir, "dump"},
}};

void WriteIps(const std::filesystem::path& path, u32 word, size_t offset = PatchOffset) {
    const bool ips32 = offset > 0xFFFFFF;
    std::vector<u8> ips =
        ips32 ? std::vector<u8>{'I', 'P', 'S', '3', '2'} : std::vector<u8>{'P', 'A', 'T', 'C', 'H'};
    if (ips32) {
        ips.push_back(static_cast<u8>(offset >> 24));
    }
    ips.push_back(static_cast<u8>(offset >> 16));
    ips.push_back(static_cast<u8>(offset >> 8));
    ips.push_back(static_cast<u8>(offset));
    ips.push_back(0);
    ips.push_back(sizeof(word));
    for (size_t i = 0; i < sizeof(word); ++i) {
        ips.push_back(static_cast<u8>(word >> (i * 8)));
    }
    if (ips32) {
        ips.push_back('E');
    }
    ips.insert(ips.end(), {'E', 'O', 'F'});
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<const char*>(ips.data()), static_cast<std::streamsize>(ips.size()));
}

class PatchManagerTestContext {
public:
    explicit PatchManagerTestContext(std::optional<u32> user_mod_word = std::nullopt,
                                     size_t user_mod_offset = PatchOffset) {
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
            WriteIps(exefs / fmt::format("{}.ips", OcarinaBuildIdHex), *user_mod_word,
                     user_mod_offset);
        }
        system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
        system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
        system.GetFileSystemController().CreateFactories(*system.GetFilesystem());
    }

    ~PatchManagerTestContext() {
        for (size_t i = 0; i < TestDirectories.size(); ++i) {
            Common::FS::SetEdenPath(TestDirectories[i].first, previous[i]);
        }
        Settings::values.disabled_addons.erase(OcarinaTitleId);
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    PatchManagerTestContext(const PatchManagerTestContext&) = delete;
    PatchManagerTestContext& operator=(const PatchManagerTestContext&) = delete;

    FileSys::PatchManager Manager(u64 title_id = OcarinaTitleId) {
        return FileSys::PatchManager{title_id, system.GetFileSystemController(),
                                     system.GetContentProvider()};
    }

private:
    std::array<std::filesystem::path, TestDirectories.size()> previous;
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("builtin-patch-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Core::System system;
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

u32 WordAt(const std::vector<u8>& nso, size_t offset = PatchOffset) {
    REQUIRE(nso.size() >= offset + sizeof(u32));
    u32 word;
    std::memcpy(&word, nso.data() + offset, sizeof(word));
    return word;
}

u32 PatchMain(PatchManagerTestContext& context, const FileSys::PatchManager::BuildID& build_id,
              u32 word, const std::string& name = "main", u64 title_id = OcarinaTitleId) {
    return WordAt(context.Manager(title_id).PatchNSO(MakeMain(build_id, word), name));
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

namespace {

// The three "frame rate over 100" branches FPS Unlock removes, with their original words.
constexpr std::array<std::pair<size_t, u32>, 3> GroundVelocityBranches{{
    {sizeof(Loader::NSOHeader) + 0x13A070, 0x5400196C},
    {sizeof(Loader::NSOHeader) + 0x13D710, 0x54001E2C},
    {sizeof(Loader::NSOHeader) + 0x389AC2C, 0x540017CC},
}};

std::vector<u8> MakeDynamicMain() {
    auto nso = MakeMain(OcarinaBuildId, TbzInstruction);
    nso.resize(DynamicStateOffset + StepStateSize);
    Loader::NSOHeader header;
    std::memcpy(&header, nso.data(), sizeof(header));
    header.segments[0].size = 63201744;
    header.segments[1].location = 63205376;
    header.segments[1].size = 12603520;
    header.segments[2].location = 75812864;
    header.segments[2].size = 5217152;
    header.segments[2].bss_size = 1414272;
    std::memcpy(nso.data(), &header, sizeof(header));
    std::memcpy(nso.data() + DynamicCallOffset, &DynamicOriginalInstruction,
                sizeof(DynamicOriginalInstruction));
    std::memcpy(nso.data() + WaterHookOffset, &WaterOriginalInstruction,
                sizeof(WaterOriginalInstruction));
    std::memcpy(nso.data() + WaterMulOffset, &WaterMulOriginalInstruction,
                sizeof(WaterMulOriginalInstruction));
    std::memcpy(nso.data() + SoundEntryOffset, &SoundEntryOriginalInstruction,
                sizeof(SoundEntryOriginalInstruction));
    std::memcpy(nso.data() + SoundStepOffset, &SoundStepOriginalInstruction,
                sizeof(SoundStepOriginalInstruction));
    std::memcpy(nso.data() + HudHookOffset, &HudOriginalInstruction, sizeof(u32));
    std::memcpy(nso.data() + ReplayHookOffset, &ReplayOriginalInstruction, sizeof(u32));
    std::memcpy(nso.data() + AmbientHookOffset, &AmbientOriginalInstruction, sizeof(u32));
    constexpr std::array<u8, 44> hitstop_original{
        0x00, 0xB9, 0xA1, 0x5E, 0x09, 0x01, 0x38, 0x1E, 0x00, 0xD8, 0x21, 0x5E,
        0x00, 0x39, 0x20, 0x1E, 0x00, 0x20, 0x30, 0x1E, 0xAA, 0x17, 0x00, 0x54,
        0x00, 0x20, 0x31, 0x1E, 0x09, 0x18, 0x00, 0x54, 0x0A, 0x9B, 0x40, 0xB9,
        0x3F, 0x05, 0x00, 0x71, 0x29, 0xC5, 0x9F, 0x1A,
    };
    std::copy(hitstop_original.begin(), hitstop_original.end(), nso.begin() + HitstopHookOffset);
    for (const auto& [offset, original] : GroundVelocityBranches) {
        std::memcpy(nso.data() + offset, &original, sizeof(original));
    }
    std::memcpy(nso.data() + TurnSpeedHookOffset, &TurnSpeedOriginalInstruction, sizeof(u32));
    std::memcpy(nso.data() + TurnSpeedHookOffset + 4, &TurnSpeedSecondInstruction, sizeof(u32));
    std::memcpy(nso.data() + MovieFramesHookOffset, &MovieFramesOriginalInstruction, sizeof(u32));
    std::memcpy(nso.data() + ScarecrowRecordHookOffset, &ScarecrowCutOriginalInstruction,
                sizeof(u32));
    std::memcpy(nso.data() + ScarecrowLastHookOffset, &ScarecrowCutOriginalInstruction,
                sizeof(u32));
    return nso;
}

} // Anonymous namespace

TEST_CASE("Built-in FPS Unlock defaults on and replaces Dynamic FPS", "[patch_manager]") {
    PatchManagerTestContext context;
    const auto manager = context.Manager();
    const auto listed = [&](std::string_view name) {
        const auto patches = manager.GetPatches();
        const auto it = std::ranges::find(patches, name, &FileSys::Patch::name);
        REQUIRE(it != patches.end());
        REQUIRE(it->type == FileSys::PatchType::Mod);
        return *it;
    };
    const auto routine = [](const std::vector<u8>& nso, size_t size) {
        return Common::HexToString(std::span{nso}.subspan(DynamicRoutineOffset, size));
    };
    constexpr std::string_view RoutineHead =
        "2818201ECA9200B04A0138912BE03BD54C0140F94B0100F9CC0200B46B010CCB0DE03BD5"
        "4E0180D2AE09CE9A7F010EEBCB818B9A4C0540F94C0000B5EC030BAA6E010CCB8C0D8E8B"
        "4C0500F98201239EA301239E4218231E4208211E";
    const auto water_patched = [](const std::vector<u8>& nso) {
        return WordAt(nso, WaterHookOffset) == WaterCallInstruction &&
               WordAt(nso, WaterMulOffset) == NopInstruction &&
               Common::HexToString(std::span{nso}.subspan(WaterRoutineOffset, WaterRoutineSize)) ==
                   WaterRoutine;
    };
    const auto water_untouched = [](const std::vector<u8>& nso) {
        return WordAt(nso, WaterHookOffset) == WaterOriginalInstruction &&
               WordAt(nso, WaterMulOffset) == WaterMulOriginalInstruction &&
               WordAt(nso, WaterRoutineOffset) == 0;
    };
    const auto hex_at = [](const std::vector<u8>& nso, size_t offset, size_t size) {
        return Common::HexToString(std::span{nso}.subspan(offset, size));
    };
    const auto sound_gate_patched = [&](const std::vector<u8>& nso) {
        return WordAt(nso, SoundEntryOffset) == SoundEntryBranchInstruction &&
               WordAt(nso, SoundStepOffset) == SoundStepCallInstruction &&
               hex_at(nso, SoundGateOffset, SoundGate.size() / 2) == SoundGate &&
               hex_at(nso, SoundFeedOffset, SoundFeed.size() / 2) == SoundFeed;
    };
    const auto sound_gate_untouched = [](const std::vector<u8>& nso) {
        return WordAt(nso, SoundEntryOffset) == SoundEntryOriginalInstruction &&
               WordAt(nso, SoundStepOffset) == SoundStepOriginalInstruction &&
               WordAt(nso, SoundGateOffset) == 0 && WordAt(nso, SoundFeedOffset) == 0;
    };
    const auto nso = MakeDynamicMain();

    // Default: FPS Unlock applies, Dynamic FPS is listed as replaced by it when it applies.
    REQUIRE(listed(FpsUnlock).enabled);
    REQUIRE(listed(DynamicFps).enabled);
    REQUIRE(listed(DynamicFps).version ==
            fmt::format("IPS, replaced by {} when it applies", FpsUnlock));
    auto patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(routine(patched, UnlockRoutineSize) ==
            std::string{RoutineHead} + "E300001C4268231EC300001C4278231E4840201EC0035FD6"
                                       "1F2003D50AD7A33C0000C040");
    REQUIRE(WordAt(patched, TurnSpeedHookOffset) == TurnSpeedHookInstruction);
    REQUIRE(WordAt(patched, TurnSpeedHookOffset + 4) == TurnSpeedSecondInstruction);
    REQUIRE(Common::HexToString(
                std::span{patched}.subspan(TurnSpeedRoutineOffset, TurnSpeedRoutineSize)) ==
            "01102E1E0021211E8A0000540018281E0140211E3360A717E10F40BD0108211E0008281E"
            "2F60A717");
    REQUIRE(WordAt(patched, MovieFramesHookOffset) == MovieFramesHookInstruction);
    REQUIRE(Common::HexToString(
                std::span{patched}.subspan(MovieFramesRoutineOffset, MovieFramesRoutineSize)) ==
            "ED3BBDA9EF4301A9FE1300F98DE20B91CE810090CE4944F9CE0140F9C01140BDBCFFFF97"
            "08010D0BFE1340F9EF4341A9ED3BC3A84A57FC17");
    REQUIRE(WordAt(patched, ScarecrowRecordHookOffset) == ScarecrowRecordHookInstruction);
    REQUIRE(Common::HexToString(std::span{patched}.subspan(ScarecrowRecordRoutineOffset,
                                                           ScarecrowRoutineSize)) ==
            "ED3BBDA9EF4301A9FE1300F9ED0309AAB0FFFF97E8030D2AFE1340F9EF4341A9ED3BC3A8"
            "0E40D917");
    REQUIRE(WordAt(patched, ScarecrowLastHookOffset) == ScarecrowLastHookInstruction);
    REQUIRE(Common::HexToString(
                std::span{patched}.subspan(ScarecrowLastRoutineOffset, ScarecrowRoutineSize)) ==
            "ED3BBDA9EF4301A9FE1300F98DF21791A6FFFF97E8030D2AFE1340F9EF4341A9ED3BC3A8"
            "F42ED917");
    REQUIRE(WordAt(patched) == BranchInstruction);
    REQUIRE(water_patched(patched));
    REQUIRE(sound_gate_patched(patched));

    // Unchecking only Dynamic FPS keeps FPS Unlock.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{DynamicFps}};
    REQUIRE_FALSE(listed(DynamicFps).enabled);
    REQUIRE(routine(manager.PatchNSO(nso, "main"), UnlockRoutineSize) ==
            routine(patched, UnlockRoutineSize));

    // Without FPS Unlock, Dark's routine applies unchanged.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{FpsUnlock}};
    REQUIRE_FALSE(listed(FpsUnlock).enabled);
    REQUIRE(listed(DynamicFps).version == "IPS");
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(routine(patched, DynamicRoutineSize) ==
            std::string{RoutineHead} + "C300001C0309231E4020231E4D0000544840201EC0035FD6"
                                       "0AD7833F");
    REQUIRE(WordAt(patched, DynamicRoutineOffset + DynamicRoutineSize) == 0);
    REQUIRE(water_patched(patched));
    // Dark's step stays at or above 1.0, so the sound manager keeps running every frame.
    REQUIRE(sound_gate_untouched(patched));
    // The turn speed fix belongs to FPS Unlock only.
    REQUIRE(WordAt(patched, TurnSpeedHookOffset) == TurnSpeedOriginalInstruction);
    REQUIRE(WordAt(patched, TurnSpeedRoutineOffset) == 0);
    // So does the movie frame count, since its helper is FPS Unlock's.
    REQUIRE(WordAt(patched, MovieFramesHookOffset) == MovieFramesOriginalInstruction);
    REQUIRE(WordAt(patched, MovieFramesRoutineOffset) == 0);
    // So do the Scarecrow's Song recording counters, since their helper is FPS Unlock's.
    REQUIRE(WordAt(patched, ScarecrowRecordHookOffset) == ScarecrowCutOriginalInstruction);
    REQUIRE(WordAt(patched, ScarecrowRecordRoutineOffset) == 0);
    REQUIRE(WordAt(patched, ScarecrowLastHookOffset) == ScarecrowCutOriginalInstruction);
    REQUIRE(WordAt(patched, ScarecrowLastRoutineOffset) == 0);

    // Both unchecked: hook and code cave stay untouched.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{FpsUnlock},
                                                        std::string{DynamicFps}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicOriginalInstruction);
    REQUIRE(WordAt(patched, DynamicRoutineOffset) == 0);
    REQUIRE(water_untouched(patched));
    REQUIRE(sound_gate_untouched(patched));

    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(WordAt(patched) == TbzInstruction);

    const auto unrelated = context.Manager(OtherTitleId).GetPatches();
    REQUIRE(std::ranges::find(unrelated, DynamicFps, &FileSys::Patch::name) == unrelated.end());
    REQUIRE(std::ranges::find(unrelated, FpsUnlock, &FileSys::Patch::name) == unrelated.end());
}

TEST_CASE("Built-in FPS Unlock skips the sound gate's sites only as a whole patch",
          "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    auto nso = MakeDynamicMain();
    SECTION("Wrong original job entry") {
        nso[SoundEntryOffset] ^= 1;
    }
    SECTION("Wrong original step load") {
        nso[SoundStepOffset] ^= 1;
    }
    SECTION("Nonzero gate cave") {
        nso[SoundGateOffset + 4] = 1;
    }
    SECTION("Nonzero sound state") {
        nso[DynamicStateOffset + 0xA4] = 1;
    }
    // FPS Unlock doesn't apply, so Dark's routine does, without the sound gate.
    const auto patched = context.Manager().PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(WordAt(patched, DynamicRoutineOffset + DynamicRoutineSize) == 0);
    REQUIRE(WordAt(patched, SoundStepOffset) != SoundStepCallInstruction);
    REQUIRE(WordAt(patched, SoundEntryOffset) != SoundEntryBranchInstruction);
}

TEST_CASE("Built-in FPS Unlock applies all step counters", "[patch_manager]") {
    PatchManagerTestContext context;
    const auto nso = MakeDynamicMain();
    const auto patched = context.Manager().PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, HudHookOffset) == HudHookInstruction);
    REQUIRE(WordAt(patched, ReplayHookOffset) == ReplayHookInstruction);
    REQUIRE(WordAt(patched, HitstopHookOffset) == HitstopHookInstruction);
    REQUIRE(WordAt(patched, AmbientHookOffset) == AmbientHookInstruction);
    REQUIRE(Common::HexToString(std::span{patched}.subspan(StepRoutineOffset, 40)) ==
            "ED3BBDA9EF4301A9FE1300F96D820D911C000094EC030D2AFE1340F9EF4341A9ED3BC3A8"
            "78863817");
    REQUIRE(Common::HexToString(std::span{patched}.subspan(StepRoutineOffset + 40, 40)) ==
            "ED3BBDA9EF4301A9FE1300F96D02148B12000094E8030D2AFE1340F9EF4341A9ED3BC3A8"
            "B232D917");
    REQUIRE(Common::HexToString(std::span{patched}.subspan(StepRoutineOffset + 80, 48)) ==
            "ED3BBDA9EF4301A9FE1300F90D6302910041201E07000094E9030D2AFE1340F9EF4341A9"
            "ED3BC3A80A9B40B94632CC17");
    REQUIRE(Common::HexToString(std::span{patched}.subspan(StepRoutineOffset + 128, 128)) ==
            "E007BFADCE9200B0CE013B910F028052D00140F91F020DEB40020054700100B4CE410091"
            "EF05007141FFFF54CE4104D1D00140B90F060011EF0D0012CF0100B9CE410091CE11108B"
            "CD0100F9DF0900B90D00381EBF050071ADA59F1A07000014C10940BD0028211E01C0251E"
            "0038211EC00900BD2D00381EE007C1ACC0035FD6");
    REQUIRE(Common::HexToString(std::span{patched}.subspan(StepRoutineOffset + 256, 60)) ==
            "ED3BBDA9EF4301A9FE1300F98D0B018BCE810090CE4944F9CE0140F9C01140BDD8FFFF97"
            "7A030D0B001840BDFE1340F9EF4341A9ED3BC3A8B8E32317");
    // The remaining instructions in the 44-byte hitstop record are guarded but unchanged.
    REQUIRE(std::equal(nso.begin() + HitstopHookOffset + sizeof(u32),
                       nso.begin() + HitstopHookOffset + 44,
                       patched.begin() + HitstopHookOffset + sizeof(u32)));
}

TEST_CASE("Built-in Dynamic FPS leaves all step counters untouched", "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{FpsUnlock}};
    const auto nso = MakeDynamicMain();
    const auto patched = context.Manager().PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    REQUIRE(WordAt(patched, HudHookOffset) == HudOriginalInstruction);
    REQUIRE(WordAt(patched, ReplayHookOffset) == ReplayOriginalInstruction);
    REQUIRE(WordAt(patched, HitstopHookOffset) == HitstopOriginalInstruction);
    REQUIRE(WordAt(patched, AmbientHookOffset) == AmbientOriginalInstruction);
    REQUIRE(std::equal(nso.begin() + HitstopHookOffset, nso.begin() + HitstopHookOffset + 44,
                       patched.begin() + HitstopHookOffset));
    REQUIRE(std::all_of(patched.begin() + StepRoutineOffset,
                        patched.begin() + StepRoutineOffset + 0x13C,
                        [](u8 byte) { return byte == 0; }));
}

TEST_CASE("Built-in FPS Unlock validates all step records before writing", "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                     std::string{DynamicFps}};
    auto nso = MakeDynamicMain();
    SECTION("Wrong HUD original") {
        nso[HudHookOffset] ^= 1;
    }
    SECTION("Wrong replay original") {
        nso[ReplayHookOffset] ^= 1;
    }
    SECTION("Wrong ambient original") {
        nso[AmbientHookOffset] ^= 1;
    }
    SECTION("Wrong hitstop original") {
        nso[HitstopHookOffset] ^= 1;
    }
    SECTION("Changed instruction inside skipped hitstop block") {
        nso[HitstopHookOffset + 40] ^= 1;
    }
    SECTION("Nonzero stub cave") {
        nso[StepRoutineOffset + 79] = 1;
    }
    SECTION("Nonzero ambient cave") {
        nso[StepRoutineOffset + 0x13B] = 1;
    }
    SECTION("Nonzero helper cave") {
        nso[StepRoutineOffset + 239] = 1;
    }
    SECTION("Nonzero step state") {
        nso[DynamicStateOffset + StepStateSize - 1] = 1;
    }
    SECTION("Truncated step state") {
        nso.resize(DynamicStateOffset + StepStateSize - 1);
    }
    SECTION("Step state outside BSS") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[2].bss_size =
            0x4E9FE00 + StepStateSize - 1 - header.segments[2].location - header.segments[2].size;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    REQUIRE(context.Manager().PatchNSO(nso, "main") == nso);
}

TEST_CASE("Built-in FPS Unlock defers transactionally to mods in step records", "[patch_manager]") {
    size_t offset = HudHookOffset;
    SECTION("HUD hook") {}
    SECTION("Replay hook") {
        offset = ReplayHookOffset;
    }
    SECTION("Hitstop hook") {
        offset = HitstopHookOffset;
    }
    SECTION("Ambient hook") {
        offset = AmbientHookOffset;
    }
    SECTION("HUD stub") {
        offset = StepRoutineOffset;
    }
    SECTION("Replay stub") {
        offset = StepRoutineOffset + 40;
    }
    SECTION("Hitstop stub") {
        offset = StepRoutineOffset + 80;
    }
    SECTION("Ambient stub") {
        offset = StepRoutineOffset + 256;
    }
    SECTION("Skipped hitstop instruction") {
        offset = HitstopHookOffset + 40;
    }
    SECTION("Shared helper") {
        offset = StepRoutineOffset + 128;
    }
    SECTION("Step state") {
        offset = DynamicStateOffset + StepStateSize - sizeof(u32);
    }
    constexpr u32 mod_word = 0x11223344;
    PatchManagerTestContext context{mod_word, offset};
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                     std::string{DynamicFps}};
    const auto nso = MakeDynamicMain();
    auto expected = nso;
    std::memcpy(expected.data() + offset, &mod_word, sizeof(mod_word));
    REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
}

TEST_CASE("Built-in FPS Unlock removes the ground velocity branches", "[patch_manager]") {
    PatchManagerTestContext context;
    const auto manager = context.Manager();
    auto nso = MakeDynamicMain();
    auto patched = manager.PatchNSO(nso, "main");
    for (const auto& [offset, original] : GroundVelocityBranches) {
        REQUIRE(WordAt(patched, offset) == NopInstruction);
    }

    // Dynamic FPS never exceeds 60 fps and keeps the game's branches.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{FpsUnlock}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicCallInstruction);
    for (const auto& [offset, original] : GroundVelocityBranches) {
        REQUIRE(WordAt(patched, offset) == original);
    }

    // Another branch at one of the sites skips all of FPS Unlock.
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{DynamicFps}};
    for (const auto& [changed, word] : GroundVelocityBranches) {
        auto other = nso;
        other[changed] ^= 1;
        patched = manager.PatchNSO(other, "main");
        REQUIRE(WordAt(patched, DynamicCallOffset) == DynamicOriginalInstruction);
        for (const auto& [offset, original] : GroundVelocityBranches) {
            REQUIRE(WordAt(patched, offset) == WordAt(other, offset));
        }
    }
}

TEST_CASE("Built-in Dynamic FPS rejects mismatched or truncated modules transactionally",
          "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    auto nso = MakeDynamicMain();
    std::string module = "main";
    u64 title = OcarinaTitleId;
    SECTION("Wrong build") {
        nso[offsetof(Loader::NSOHeader, build_id)] ^= 1;
    }
    SECTION("Wrong module") {
        module = "sdk";
    }
    SECTION("Wrong title") {
        title = OtherTitleId;
    }
    SECTION("Incomplete header") {
        nso.resize(sizeof(Loader::NSOHeader) - 1);
    }
    SECTION("Incomplete call") {
        nso.resize(DynamicCallOffset + 3);
    }
    SECTION("Incomplete routine") {
        nso.resize(DynamicRoutineOffset + DynamicRoutineSize - 1);
    }
    SECTION("Incomplete timer state") {
        nso.resize(DynamicStateOffset + DynamicStateSize - 1);
    }
    SECTION("Timer state outside BSS") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[2].bss_size = 15;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    SECTION("Wrong original hook") {
        nso[DynamicCallOffset] ^= 1;
    }
    SECTION("Nonzero code cave") {
        nso[DynamicRoutineOffset + DynamicRoutineSize - 1] = 1;
    }
    SECTION("Nonzero timer state") {
        nso[DynamicStateOffset + DynamicStateSize - 1] = 1;
    }
    SECTION("Wrong original water hook") {
        nso[WaterHookOffset] ^= 1;
    }
    SECTION("Wrong original water multiply") {
        nso[WaterMulOffset] ^= 1;
    }
    SECTION("Nonzero water code cave") {
        nso[WaterRoutineOffset + WaterRoutineSize - 1] = 1;
    }
    SECTION("Code cave outside executable padding") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[0].size = 0x3C46000;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    SECTION("Code cave overlaps read-only segment") {
        Loader::NSOHeader header;
        std::memcpy(&header, nso.data(), sizeof(header));
        header.segments[1].location = 0x3C46200;
        std::memcpy(nso.data(), &header, sizeof(header));
    }
    REQUIRE(context.Manager(title).PatchNSO(nso, module) == nso);
}

TEST_CASE("Built-in Dynamic FPS leaves both records alone when a user mod changes either",
          "[patch_manager]") {
    size_t offset = DynamicCallOffset;
    u32 mod_word = 0x11223344;
    SECTION("User call takes precedence") {}
    SECTION("Identical call still takes precedence for the whole patch") {
        mod_word = DynamicCallInstruction;
    }
    SECTION("User routine takes precedence") {
        offset = DynamicRoutineOffset;
    }
    SECTION("Last routine byte takes precedence") {
        offset = DynamicRoutineOffset + DynamicRoutineSize - sizeof(u32);
    }
    SECTION("User timer state takes precedence") {
        offset = DynamicStateOffset;
    }
    PatchManagerTestContext context{mod_word, offset};
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    const auto nso = MakeDynamicMain();
    auto expected = nso;
    std::memcpy(expected.data() + offset, &mod_word, sizeof(mod_word));
    REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
}

TEST_CASE("Built-in FPS Unlock validates the turn speed records before writing", "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                        std::string{DynamicFps}};
    auto nso = MakeDynamicMain();
    SECTION("Wrong turn speed original") {
        nso[TurnSpeedHookOffset] ^= 1;
    }
    SECTION("Wrong second turn speed instruction") {
        nso[TurnSpeedHookOffset + 4] ^= 1;
    }
    SECTION("Nonzero turn speed cave") {
        nso[TurnSpeedRoutineOffset + TurnSpeedRoutineSize - 1] = 1;
    }
    REQUIRE(context.Manager().PatchNSO(nso, "main") == nso);
}

TEST_CASE("Built-in FPS Unlock leaves all records alone when a user mod changes the turn speed fix",
          "[patch_manager]") {
    size_t offset = TurnSpeedHookOffset;
    SECTION("User turn speed hook takes precedence") {}
    SECTION("User second turn speed instruction takes precedence") {
        offset = TurnSpeedHookOffset + 4;
    }
    SECTION("User turn speed routine takes precedence") {
        offset = TurnSpeedRoutineOffset;
    }
    constexpr u32 mod_word = 0x11223344;
    PatchManagerTestContext context{mod_word, offset};
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                        std::string{DynamicFps}};
    const auto nso = MakeDynamicMain();
    auto expected = nso;
    std::memcpy(expected.data() + offset, &mod_word, sizeof(mod_word));
    REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
}

TEST_CASE("Built-in FPS Unlock validates the movie frame count records before writing",
          "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                        std::string{DynamicFps}};
    auto nso = MakeDynamicMain();
    SECTION("Wrong movie frame count original") {
        nso[MovieFramesHookOffset] ^= 1;
    }
    SECTION("Nonzero movie frame count cave") {
        nso[MovieFramesRoutineOffset + MovieFramesRoutineSize - 1] = 1;
    }
    REQUIRE(context.Manager().PatchNSO(nso, "main") == nso);
}

TEST_CASE("Built-in FPS Unlock leaves all records alone when a user mod changes the movie frame "
          "count",
          "[patch_manager]") {
    size_t offset = MovieFramesHookOffset;
    SECTION("User movie frame count hook takes precedence") {}
    SECTION("User movie frame count routine takes precedence") {
        offset = MovieFramesRoutineOffset;
    }
    constexpr u32 mod_word = 0x11223344;
    PatchManagerTestContext context{mod_word, offset};
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                        std::string{DynamicFps}};
    const auto nso = MakeDynamicMain();
    auto expected = nso;
    std::memcpy(expected.data() + offset, &mod_word, sizeof(mod_word));
    REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
}

TEST_CASE("Built-in FPS Unlock validates the Scarecrow's Song records before writing",
          "[patch_manager]") {
    PatchManagerTestContext context;
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                        std::string{DynamicFps}};
    auto nso = MakeDynamicMain();
    SECTION("Wrong recording original") {
        nso[ScarecrowRecordHookOffset] ^= 1;
    }
    SECTION("Wrong last note original") {
        nso[ScarecrowLastHookOffset] ^= 1;
    }
    SECTION("Nonzero recording cave") {
        nso[ScarecrowRecordRoutineOffset + ScarecrowRoutineSize - 1] = 1;
    }
    SECTION("Nonzero last note cave") {
        nso[ScarecrowLastRoutineOffset + ScarecrowRoutineSize - 1] = 1;
    }
    REQUIRE(context.Manager().PatchNSO(nso, "main") == nso);
}

TEST_CASE("Built-in FPS Unlock leaves all records alone when a user mod changes the Scarecrow's "
          "Song fix",
          "[patch_manager]") {
    size_t offset = ScarecrowRecordHookOffset;
    SECTION("User recording hook takes precedence") {}
    SECTION("User last note hook takes precedence") {
        offset = ScarecrowLastHookOffset;
    }
    SECTION("User recording routine takes precedence") {
        offset = ScarecrowRecordRoutineOffset;
    }
    SECTION("User last note routine takes precedence") {
        offset = ScarecrowLastRoutineOffset;
    }
    constexpr u32 mod_word = 0x11223344;
    PatchManagerTestContext context{mod_word, offset};
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution},
                                                        std::string{DynamicFps}};
    const auto nso = MakeDynamicMain();
    auto expected = nso;
    std::memcpy(expected.data() + offset, &mod_word, sizeof(mod_word));
    REQUIRE(context.Manager().PatchNSO(nso, "main") == expected);
}

TEST_CASE("Built-in Texture Pool x2 doubles the allocator's initial pool size",
          "[patch_manager]") {
    constexpr std::string_view TexturePool = "Texture Pool x2 (built-in)";
    constexpr size_t CallOffset = sizeof(Loader::NSOHeader) + 0xC2A36C;
    constexpr size_t RoutineOffset = sizeof(Loader::NSOHeader) + 0x3C462C0;
    constexpr u32 StpInstruction = 0xA90127E8;
    constexpr u32 CallInstruction = 0x94C06FD5;
    PatchManagerTestContext context;
    const auto manager = context.Manager();
    auto nso = MakeDynamicMain();
    std::memcpy(nso.data() + CallOffset, &StpInstruction, sizeof(StpInstruction));

    auto patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, CallOffset) == CallInstruction);
    REQUIRE(Common::HexToString(std::span{patched}.subspan(RoutineOffset, 12)) ==
            "08F97FD3E82701A9C0035FD6");
    REQUIRE(WordAt(patched, RoutineOffset + 12) == 0);

    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{TexturePool}};
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, CallOffset) == StpInstruction);
    REQUIRE(WordAt(patched, RoutineOffset) == 0);

    Settings::values.disabled_addons.erase(OcarinaTitleId);
    std::memcpy(nso.data() + CallOffset, &NopInstruction, sizeof(NopInstruction));
    patched = manager.PatchNSO(nso, "main");
    REQUIRE(WordAt(patched, CallOffset) == NopInstruction);
    REQUIRE(WordAt(patched, RoutineOffset) == 0);
}

TEST_CASE("Built-in DLSS render scale follows the NVIDIA DLSS quality", "[patch_manager]") {
#ifdef EGLANTINE_STREAMLINE
    constexpr bool DlssBuild = true;
#else
    constexpr bool DlssBuild = false;
#endif
    constexpr size_t ScaleOffset = sizeof(Loader::NSOHeader) + 0x806420;
    constexpr u32 ScaleOne = 0x52A7F009;
    struct SettingsGuard {
        ~SettingsGuard() {
            Settings::values.nvidia_dlss.SetValue(false);
            Settings::values.nvidia_dlss_quality.SetValue(Settings::NvidiaDlssQuality::Automatic);
            Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
        }
    } guard;
    PatchManagerTestContext context;
    auto nso = MakeMain(OcarinaBuildId, TbzInstruction);
    nso.resize(ScaleOffset + sizeof(u32));
    std::memcpy(nso.data() + ScaleOffset, &ScaleOne, sizeof(ScaleOne));
    const auto scale = [&] { return WordAt(context.Manager().PatchNSO(nso, "main"), ScaleOffset); };

    Settings::values.nvidia_dlss.SetValue(true);
    Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Docked);
    using Quality = Settings::NvidiaDlssQuality;
    for (const auto& [quality, word] : std::array<std::pair<Quality, u32>, 6>{{
             {Quality::Automatic, ScaleOne},
             {Quality::Native, ScaleOne},
             {Quality::Quality, ScaleOne},
             {Quality::Balanced, 0x52A7EBE9},
             {Quality::Performance, 0x52A7E809},
             {Quality::UltraPerformance, 0x52A7E009},
         }}) {
        Settings::values.nvidia_dlss_quality.SetValue(quality);
        CHECK(scale() == (DlssBuild ? word : ScaleOne));
        CHECK(Settings::AppliedOcarinaRenderScale() ==
              (DlssBuild && word != ScaleOne ? std::optional{quality} : std::nullopt));
    }

    Settings::values.nvidia_dlss_quality.SetValue(Quality::Performance);
    Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
    CHECK(scale() == ScaleOne);
    Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Docked);
    Settings::values.nvidia_dlss.SetValue(false);
    CHECK(scale() == ScaleOne);
    Settings::values.nvidia_dlss.SetValue(true);
    Settings::values.disabled_addons[OcarinaTitleId] = {std::string{MaxResolution}};
    CHECK(scale() == ScaleOne);
    CHECK_FALSE(Settings::AppliedOcarinaRenderScale());

    for (const auto& patch : context.Manager().GetPatches()) {
        CHECK(patch.name.find("render scale") == std::string::npos);
    }
}
