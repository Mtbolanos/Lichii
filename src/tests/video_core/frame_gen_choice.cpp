// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/frame_gen_choice.h"

namespace {

using Settings::FrameGenType;
using VideoCore::ActiveFrameGenType;
using VideoCore::BlockedDlssg;
using VideoCore::DlssgBlock;
using VideoCore::DlssgSession;

constexpr DlssgSession OCARINA{
    .streamline = true,
    .dlss = true,
    .program_id = VideoCore::OCARINA_PROGRAM_ID,
    .loaded = true,
};

} // Anonymous namespace

TEST_CASE("DLSS Frame Generation runs only for a game with inputs, else FSR3", "[video_core]") {
    REQUIRE(BlockedDlssg(OCARINA) == DlssgBlock::None);
    REQUIRE(ActiveFrameGenType(FrameGenType::NvidiaDlssG, DlssgBlock::None) ==
            FrameGenType::NvidiaDlssG);

    DlssgSession fire_emblem = OCARINA;
    fire_emblem.program_id = 0x04007EF01F8AE000ULL;
    fire_emblem.loaded = false;
    REQUIRE(BlockedDlssg(fire_emblem) == DlssgBlock::NoGameInputs);

    DlssgSession no_dlss = OCARINA;
    no_dlss.dlss = false;
    REQUIRE(BlockedDlssg(no_dlss) == DlssgBlock::DlssOff);

    DlssgSession not_loaded = OCARINA;
    not_loaded.loaded = false;
    REQUIRE(BlockedDlssg(not_loaded) == DlssgBlock::NotLoaded);

    DlssgSession linux_build = OCARINA;
    linux_build.streamline = false;
    REQUIRE(BlockedDlssg(linux_build) == DlssgBlock::NoStreamline);

    for (const DlssgBlock block : {DlssgBlock::NoStreamline, DlssgBlock::DlssOff,
                                   DlssgBlock::NoGameInputs, DlssgBlock::NotLoaded}) {
        REQUIRE(ActiveFrameGenType(FrameGenType::NvidiaDlssG, block) == FrameGenType::Fsr3);
    }
    REQUIRE(ActiveFrameGenType(FrameGenType::Lsfg, DlssgBlock::NotLoaded) == FrameGenType::Lsfg);
    REQUIRE(ActiveFrameGenType(FrameGenType::Fsr3, DlssgBlock::None) == FrameGenType::Fsr3);
}
