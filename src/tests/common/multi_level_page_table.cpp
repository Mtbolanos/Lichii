// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include "common/multi_level_page_table.h"

TEST_CASE("MultiLevelPageTable reserves ranges that end at the top of the address space",
          "[multi_level_page_table]") {
    Common::MultiLevelPageTable<u32> table{32, 4, 12};
    constexpr u64 last_page = (1ULL << 20) - 1;
    table.ReserveRange((1ULL << 32) - 0x1000, 0x1000);
    table[last_page] = 7;
    REQUIRE(table[last_page] == 7);
    table.ReserveRange(0, 1ULL << 32);
    table[0] = 3;
    REQUIRE(table[0] == 3);
    REQUIRE(table[last_page] == 7);
}
