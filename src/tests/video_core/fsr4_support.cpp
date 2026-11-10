// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/fsr4_support.h"

TEST_CASE("FSR 4 is offered on RDNA 2 and newer Radeons on AMD's driver", "[video_core][fsr4]") {
    using VideoCommon::IsFsr4CapableGpu;
    constexpr u32 amd = 0x1002;
    for (const char* name :
         {"AMD Radeon RX 9070 XT", "AMD Radeon RX 9060 XT", "AMD Radeon RX 7900 XTX",
          "AMD Radeon RX 7600S", "AMD Radeon RX 6800 XT", "AMD Radeon RX 6700M",
          "AMD Radeon PRO W7900", "AMD Radeon(TM) 890M Graphics", "AMD Radeon(TM) 880M",
          "AMD Radeon(TM) 8060S Graphics", "AMD Radeon 780M Graphics"}) {
        INFO(name);
        CHECK(IsFsr4CapableGpu(amd, true, name));
    }
    for (const char* name :
         {"AMD Radeon RX 5700 XT", "AMD Radeon RX 580", "AMD Radeon RX Vega", "AMD Radeon(TM) 680M",
          "AMD Radeon(TM) Graphics", "AMD Radeon VII", "AMD Radeon RX 590 GME"}) {
        INFO(name);
        CHECK_FALSE(IsFsr4CapableGpu(amd, true, name));
    }
    CHECK_FALSE(IsFsr4CapableGpu(amd, false, "AMD Radeon RX 9070 XT (RADV GFX1201)"));
    CHECK_FALSE(IsFsr4CapableGpu(0x10DE, true, "NVIDIA GeForce RTX 3080"));
    CHECK_FALSE(IsFsr4CapableGpu(0x8086, true, "Intel(R) Arc(TM) A770 Graphics"));
}
