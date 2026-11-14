// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

TEST_CASE("Ampere surface reductions preserve 3D coordinates", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    // Star Fox: atomic unsigned add of R3 to the 3D image at c[0][0x88], xyz in R0-R2.
    Instruction sured{{0x0000759f, 0xa0002203, 0x0010e000, 0x0011e600}};
    for (const u32 dimension : {3U, 5U}) {
        sured.words[1] = (sured.words[1] & 0x1fffffffU) | (dimension << 29);
        const auto lowered = LowerToMaxwell(std::array{sured, exit});
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.required_texture_buffer == 0);
        REQUIRE(lowered.instructions.size() == 2);
        const u64 atomic = lowered.instructions[0];
        REQUIRE((atomic & 0xfff0000000000000ULL) == 0xeb40000000000000ULL);
        REQUIRE((atomic & 0xff) == 3); // data register
        REQUIRE(((atomic >> 8) & 0xff) == 0); // first coordinate
        REQUIRE(((atomic >> 16) & 15) == 7); // original guard
        REQUIRE(((atomic >> 20) & 7) == 0); // U32
        REQUIRE(((atomic >> 24) & 7) == 0); // ADD
        REQUIRE(((atomic >> 33) & 7) == dimension);
        REQUIRE(((atomic >> 36) & 0x1fff) == 0x22); // descriptor word offset
        REQUIRE(((atomic >> 51) & 1) == 1); // bound descriptor
    }
    sured.words[0] = 0xfd00759f; // R253-R255 would cross into the zero register in 3D
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{sured, exit})));
    sured.words[0] = 0x0a00859f; // xyz in R10-R12, guarded by !P0
    auto lowered = LowerToMaxwell(std::array{sured, exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(((lowered.instructions[0] >> 8) & 255) == 10);
    REQUIRE(((lowered.instructions[0] >> 16) & 15) == 8);
    sured = {{0x0000759f, 0xa0002403, 0x0310e000, 0x000fe200}};
    lowered = LowerToMaxwell(std::array{sured, exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(((lowered.instructions[0] >> 24) & 7) == 6); // OR
    REQUIRE(((lowered.instructions[0] >> 33) & 7) == 5); // still 3D
}

TEST_CASE("Ampere surface atomic add returns the old 3D texel value", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    Instruction atomic{{0x00087593, 0xa0002407, 0x001ee000, 0x000ea400}};
    const auto lowered = LowerToMaxwell(std::array{atomic, exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.required_texture_buffer == 0);
    REQUIRE(lowered.instructions.size() == 2);
    const u64 word = lowered.instructions[0];
    REQUIRE((word & 0xff80000000000000ULL) == 0xea00000000000000ULL);
    REQUIRE((word & 255) == 8); // old texel value returned to R8
    REQUIRE(((word >> 8) & 255) == 0); // xyz R0-R2
    REQUIRE(((word >> 20) & 255) == 7); // addend R7
    REQUIRE(((word >> 29) & 15) == 0); // ADD
    REQUIRE(((word >> 33) & 7) == 5); // 3D
    REQUIRE(((word >> 36) & 0x1fff) == 0x24); // c[0][0x90]
    REQUIRE(((word >> 51) & 7) == 0); // U32
    REQUIRE(((word >> 54) & 1) == 0); // bound descriptor
    atomic.words[2] &= ~(7U << 17); // writing a fault predicate is not yet supported
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{atomic, exit})));
}

TEST_CASE("Ampere gradient textures accept uniform array handles", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    // Star Fox's material shader: uniform handle UR4, coordinates R20, derivatives R16.
    Instruction txd{{0x14127f6c, 0xa0000410, 0x081e0f10, 0x001f4200}};
    auto lowered = LowerToMaxwell(std::array{txd, exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.required_texture_buffer == ~0U);
    REQUIRE(lowered.instructions.size() == 10);
    REQUIRE(lowered.instructions[0] == 0x5c9807800dc700f8ULL);
    const u64 texture = lowered.instructions[4];
    REQUIRE((texture & 0xffc0000000000000ULL) == 0xde40000000000000ULL);
    REQUIRE(((texture >> 8) & 0xff) == 248);
    REQUIRE(((texture >> 20) & 0xff) == 16);
    REQUIRE(((texture >> 28) & 7) == 3);
    REQUIRE(((texture >> 31) & 15) == 15);
    REQUIRE((lowered.instructions[5] & 0xff) == 18);
    REQUIRE((lowered.instructions[7] & 0xff) == 16);

    txd = {{0x18320f6c, 0xa000041c, 0x0c1e0f34, 0x00414200}};
    lowered = LowerToMaxwell(std::array{txd, exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(((lowered.instructions[4] >> 49) & 1) == 1);
    REQUIRE(((lowered.instructions[4] >> 16) & 15) == 0);

    txd.words[1] = 0xa0004020; // invalid uniform handle
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{txd, exit})));
}

TEST_CASE("Ampere array gathers preserve the layer and split coordinates", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    const std::array<Instruction, 2> code{{
        {{0x201c7b63, 0xb0000a2b, 0x041e0f16, 0x000f4200}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(((lowered.instructions[0] >> 20) & 0xff) == 32);
    REQUIRE(((lowered.instructions[1] >> 20) & 0xff) == 33);
    REQUIRE(((lowered.instructions[2] >> 20) & 0xff) == 43);
    REQUIRE(((lowered.instructions[3] >> 28) & 7) == 3);
}

TEST_CASE("Ampere uniform multiply accepts an immediate addend", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    const std::array<Instruction, 2> code{{
        {{0x040474a4, 0x00000001, 0x0f8e0205, 0x000fe200}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE((lowered.instructions[lowered.instructions.size() - 2] & 255) == 220);
}
