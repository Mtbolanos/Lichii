// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};
constexpr u64 MaxwellExit = 0xe30000000007000fULL;

u64 Vote(u32 target, u32 source_predicate, u32 temporary, u32 mode) {
    return 0x50d8000000070000ULL | target | (u64{source_predicate} << 39) |
           (u64{temporary} << 45) | (u64{mode} << 48);
}

// UR0-UR21 live in R216-R237, UR22-UR27 in R240-R245 and URZ is RZ.
u32 LowUniform(u32 reg) {
    return reg < 22 ? 216 + reg : reg < 28 ? 218 + reg : 255;
}
} // namespace

TEST_CASE("Ampere captured uniform votes lower into the mapped uniform register",
          "[video_core][ampere][uniform_vote]") {
    // VOTEU.ANY UR16/UR18, UPT, PT: rejected by the uniform vote check in the 040074A01BF12000 log.
    for (const Instruction captured : std::array{
             Instruction{{0x00107886, 0x00000000, 0x038e0100, 0x000fe400}},
             Instruction{{0x00127886, 0x00000000, 0x038e0100, 0x000fe400}}}) {
        const auto lowered = LowerToMaxwell(std::array{captured, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(lowered.instructions[0] == Vote(216 + captured.Bits(16, 8), 7, 7, 1));
        REQUIRE(lowered.instructions[1] == MaxwellExit);
    }
}

TEST_CASE("Ampere uniform votes use the complete low uniform register bank",
          "[video_core][ampere][uniform_vote]") {
    for (u32 target = 0; target <= 63; ++target) {
        if (target >= 28 && target != 63) continue;
        for (u32 output = 0; output < 8; ++output) {
            for (u32 mode = 0; mode < 3; ++mode) {
                const Instruction vote{{0x00007886U | (target << 16), 0,
                                        0x03800000U | (output << 17) | (mode << 8), 0x000fe400}};
                const auto lowered = LowerToMaxwell(std::array{vote, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                const u32 temporary = output == 7 ? 7 : 6 - output;
                const u64 mask = u64{1} << temporary;
                REQUIRE(lowered.instructions.size() == (output == 7 ? 2U : 5U));
                REQUIRE(lowered.instructions[output == 7 ? 0 : 1] ==
                        Vote(LowUniform(target), 7, temporary, mode));
                if (output != 7) {
                    // UPn lives in R239 bit 6-n: save P(6-n), vote into it, merge, restore.
                    REQUIRE(lowered.instructions[0] ==
                            (0x38e8000000070000ULL | 253 | (255ULL << 8) | (mask << 20)));
                    REQUIRE(lowered.instructions[2] ==
                            (0x38e8000000070000ULL | 239 | (239ULL << 8) | (mask << 20)));
                    REQUIRE(lowered.instructions[3] ==
                            (0x38f0000000070000ULL | (253ULL << 8) | (mask << 20)));
                }
                REQUIRE(lowered.instructions.back() == MaxwellExit);
            }
        }
    }
}

TEST_CASE("Ampere uniform votes reach UR28 to UR62 through the high uniform mirrors",
          "[video_core][ampere][uniform_vote]") {
    for (u32 target = 28; target < 63; ++target) {
        const Instruction vote{{0x00007886U | (target << 16), 0, 0x038e0100, 0x000fe400}};
        const auto lowered = LowerToMaxwell(std::array{vote, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        // A program whose highest uniform is URn mirrors (n - 26) & ~1 registers below R216.
        const u32 high_uniforms = (target - 26) & ~1U;
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(lowered.instructions[0] == Vote(216 - high_uniforms + target - 28, 7, 7, 1));
    }
    // Both ends of the high bank in one program share one mirror layout (UR28 at R180).
    const auto lowered = LowerToMaxwell(std::array{
        Instruction{{0x003e7886, 0, 0x038e0100, 0x000fe400}},
        Instruction{{0x001c7886, 0, 0x038e0100, 0x000fe400}}, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == 3);
    REQUIRE(lowered.instructions[0] == Vote(214, 7, 7, 1));
    REQUIRE(lowered.instructions[1] == Vote(180, 7, 7, 1));
    // The mirrors are reserved: a native register inside them (R214, while UR28 alone would sit
    // at R214) is shadowed. Shadowing needs an aligned base, so the program is lowered again with
    // the mirrors at R212: the vote reads UR28 there and the native move writes its shadow, R210.
    const auto shadowed = LowerToMaxwell(std::array{
        Instruction{{0x00d67802, 0x3f800000, 0x00000f00, 0x000fe400}},
        Instruction{{0x001c7886, 0, 0x038e0100, 0x000fe400}}, Exit});
    INFO(shadowed.error);
    REQUIRE(shadowed);
    REQUIRE(std::ranges::find(shadowed.instructions, Vote(212, 7, 7, 1)) !=
            shadowed.instructions.end());
    REQUIRE(std::ranges::find(shadowed.instructions, 0x0103f8000007f0d2ULL) !=
            shadowed.instructions.end());
}

TEST_CASE("Ampere uniform votes keep the source predicate and its negation",
          "[video_core][ampere][uniform_vote]") {
    for (const u32 source : {0U, 3U, 7U, 8U, 11U, 15U}) {
        const Instruction vote{{0x00107886, 0, 0x000e0100U | (source << 23), 0x000fe400}};
        const auto lowered = LowerToMaxwell(std::array{vote, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions[0] == Vote(232, source, 7, 1));
    }
}

TEST_CASE("Ampere uniform votes reject unmapped registers and unsupported fields",
          "[video_core][ampere][uniform_vote]") {
    const Instruction captured{{0x00107886, 0, 0x038e0100, 0x000fe400}};
    for (const u32 target : {64U, 100U, 215U}) {
        auto invalid = captured;
        invalid.words[0] = 0x00007886U | (target << 16);
        const auto lowered = LowerToMaxwell(std::array{invalid, Exit});
        REQUIRE_FALSE(lowered);
        REQUIRE(lowered.error == "native uniform vote mode");
    }
    for (const u32 target : {216U, 245U, 254U}) {
        auto invalid = captured;
        invalid.words[0] = 0x00007886U | (target << 16);
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
    // Execution predicate, Ra, word 1, Rc, mode 3 and the unassigned bits of word 2.
    for (const u32 bit : {12U, 24U, 31U, 32U, 63U, 64U, 71U, 73U, 74U, 75U, 76U, 80U, 84U, 85U,
                          86U, 91U, 95U}) {
        auto invalid = captured;
        invalid.words[bit / 32] ^= 1U << (bit % 32);
        const auto lowered = LowerToMaxwell(std::array{invalid, Exit});
        INFO("bit " << bit);
        REQUIRE_FALSE(lowered);
    }
}
