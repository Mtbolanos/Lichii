// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/macro.h"

TEST_CASE("Turing macro uploads reach entry points past instruction 0x1000",
          "[video_core][macro]") {
    Tegra::MacroEngine engine{false};
    engine.native_turing = true;
    // Ocarina of Time calls macros starting at instructions 0x1010 and 0x1018.
    for (const u32 start : {0x1010U, 0x1018U}) {
        CAPTURE(start);
        engine.ClearCode(start);
        for (u32 word = 0; word < 3; ++word) {
            engine.AddCode(0, 0xabc0 + word);
        }
        for (u32 word = 0; word < 3; ++word) {
            REQUIRE(engine.native_code[start * 3 + word] == 0xabc0 + word);
        }
    }
    // An instruction pointer past the RAM leaves every word alone, even when * 3 wraps a u32.
    for (const u32 start : {0x2aabU, 0x55555556U}) {
        CAPTURE(start);
        const auto before = engine.native_code;
        engine.ClearCode(start);
        engine.AddCode(0, 0xdead);
        REQUIRE(engine.native_code == before);
    }
}
