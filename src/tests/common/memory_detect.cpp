// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include "common/literals.h"
#include "common/memory_detect.h"

using namespace Common::Literals;
using Common::HostRam;
using Common::HostRamFromUdev;
using Common::InstalledRamFromUdev;
using Common::IsBelowSwitch2RamMinimum;

TEST_CASE("Switch 2 RAM minimum: installed RAM below 16 GiB", "[common]") {
    CHECK(IsBelowSwitch2RamMinimum(HostRam{8_GiB, true}));
    CHECK(IsBelowSwitch2RamMinimum(HostRam{12_GiB, true}));
    CHECK(IsBelowSwitch2RamMinimum(HostRam{16_GiB - 1_MiB, true}));
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{16_GiB, true}));
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{32_GiB, true}));
}

TEST_CASE("Switch 2 RAM minimum: managed memory below 12 GiB", "[common]") {
    CHECK(IsBelowSwitch2RamMinimum(HostRam{7_GiB + 800_MiB, false}));
    CHECK(IsBelowSwitch2RamMinimum(HostRam{11_GiB + 600_MiB, false}));
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{12_GiB, false}));
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{14_GiB + 500_MiB, false}));
}

TEST_CASE("Cold hints only when the host is short of RAM", "[common]") {
    using Common::IsLowOnMemory;
    CHECK_FALSE(IsLowOnMemory(0, 32_GiB)); // unknown
    CHECK_FALSE(IsLowOnMemory(18_GiB, 32_GiB));
    CHECK_FALSE(IsLowOnMemory(5_GiB, 32_GiB));
    CHECK(IsLowOnMemory(3_GiB, 32_GiB));
    CHECK_FALSE(IsLowOnMemory(5_GiB, 16_GiB));
    CHECK(IsLowOnMemory(3_GiB, 16_GiB));
    CHECK(IsLowOnMemory(1_GiB, 8_GiB));
    CHECK_FALSE(IsLowOnMemory(2_GiB + 1, 8_GiB));
}

TEST_CASE("Switch 2 RAM minimum: unknown RAM gives no warning", "[common]") {
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{0, true}));
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{0, false}));
}

TEST_CASE("Installed RAM sums udev's DMI memory devices", "[common]") {
    // A Steam Deck: four 4 GiB LPDDR5 devices, of which Linux manages about 11.5 GiB.
    constexpr std::string_view deck = "E:MEMORY_ARRAY_MAX_CAPACITY=68719476736\n"
                                      "E:MEMORY_DEVICE_0_SIZE=4294967296\n"
                                      "E:MEMORY_DEVICE_0_TYPE=LPDDR5\n"
                                      "E:MEMORY_DEVICE_1_SIZE=4294967296\n"
                                      "E:MEMORY_DEVICE_2_SIZE=4294967296\n"
                                      "E:MEMORY_DEVICE_3_SIZE=4294967296\n";
    CHECK(InstalledRamFromUdev(deck) == 16_GiB);
    CHECK_FALSE(IsBelowSwitch2RamMinimum(HostRam{InstalledRamFromUdev(deck), true}));
    CHECK(InstalledRamFromUdev("E:MEMORY_DEVICE_0_SIZE=8589934592") == 8_GiB);
}

TEST_CASE("Installed RAM from udev ignores other and malformed entries", "[common]") {
    CHECK(InstalledRamFromUdev("") == 0);
    CHECK(InstalledRamFromUdev("E:MEMORY_ARRAY_MAX_CAPACITY=68719476736\n") == 0);
    CHECK(InstalledRamFromUdev("E:MEMORY_DEVICE_0_SIZE=\n") == 0);
    CHECK(InstalledRamFromUdev("E:MEMORY_DEVICE_0_SIZE=4G\n") == 0);
    CHECK(InstalledRamFromUdev("E:MEMORY_DEVICE__SIZE=4294967296\n") == 0);
    CHECK(InstalledRamFromUdev("E:MEMORY_DEVICE_0_SPEED=6400\n") == 0);
}

TEST_CASE("Host RAM takes udev's total only when it covers managed memory", "[common]") {
    // A Steam Deck: 16 GiB installed, about 11.5 GiB managed.
    const HostRam deck = HostRamFromUdev("E:MEMORY_DEVICE_0_SIZE=8589934592\n"
                                         "E:MEMORY_DEVICE_1_SIZE=8589934592\n",
                                         11_GiB + 512_MiB);
    CHECK(deck.bytes == 16_GiB);
    CHECK(deck.installed);
    CHECK_FALSE(IsBelowSwitch2RamMinimum(deck));

    // Two 8 GiB devices, the second with a size the firmware reports as unknown: udev has no
    // SIZE entry for it, and the partial total must not pass for the installed RAM.
    const HostRam partial = HostRamFromUdev("E:MEMORY_DEVICE_0_SIZE=8589934592\n"
                                            "E:MEMORY_DEVICE_1_TYPE=DDR4\n"
                                            "E:MEMORY_ARRAY_NUM_DEVICES=2\n",
                                            15_GiB);
    CHECK(partial.bytes == 15_GiB);
    CHECK_FALSE(partial.installed);
    CHECK_FALSE(IsBelowSwitch2RamMinimum(partial));

    // No memory devices at all (a VM or container without SMBIOS data).
    const HostRam none = HostRamFromUdev("", 15_GiB);
    CHECK(none.bytes == 15_GiB);
    CHECK_FALSE(none.installed);
}
