// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};

// TLD4.PTP R10/R8, R4 (x, y), R8 (offsets in R8, R9), 2D, RGBA: rejected as a texture gather
// mode in the 040057C02159A000 log.
constexpr Instruction CapturedPerTapGather{{0x040a7b63, 0x33800408, 0x041e2f08, 0x100fc200}};

u64 Copy(u64 predicate, u32 target, u32 source) {
    return 0x5c98000000000000ULL | predicate | target | (15ULL << 39) | (u64{source} << 20);
}

u64 MoveImmediate(u64 predicate, u32 target, u32 value) {
    return 0x0100000000000000ULL | predicate | target | (15ULL << 12) | (u64{value} << 20);
}

std::array<Instruction, 4> PerTapProgram(u32 packed = 0xff00ff00) {
    return {{
        {{0x00087802, packed, 0x00000f00, 0x000fe400}},
        {{0x00097802, 0x01000001, 0x00000f00, 0x000fe400}},
        CapturedPerTapGather,
        Exit,
    }};
}
} // namespace

TEST_CASE("Ampere per-tap gathers pass proven constant offsets to Maxwell TLD4.PTP",
          "[video_core][ampere][gather]") {
    for (const u32 guard : {1U, 7U, 9U}) {
        for (u32 component = 0; component < 4; ++component) {
            for (const u32 packed : {0xff00ff00U, 0x1ff0002U, 0x1fe01fe0U}) {
                auto code = PerTapProgram(packed);
                code[2].words[0] = (code[2].words[0] & ~0xf000U) | (guard << 12);
                code[2].words[2] = (code[2].words[2] & ~(3U << 23)) | (component << 23);
                const auto lowered = LowerToMaxwell(code);
                INFO(lowered.error);
                REQUIRE(lowered);
                REQUIRE(lowered.required_texture_buffer == 14);
                const u64 pred = u64{guard} << 16;
                const std::array<u64, 9> expected{
                    MoveImmediate(7ULL << 16, 248, packed),
                    MoveImmediate(7ULL << 16, 249, 0x01000001),
                    Copy(pred, 250, 4),
                    Copy(pred, 251, 5),
                    0xc800000000000000ULL | pred | 250 | (250ULL << 8) | (248ULL << 20) |
                        (2ULL << 28) | (15ULL << 31) | (4ULL << 36) | (1ULL << 49) |
                        (7ULL << 51) | (2ULL << 54) | (u64{component} << 56),
                    Copy(pred, 10, 250),
                    Copy(pred, 11, 251),
                    Copy(pred, 8, 252),
                    Copy(pred, 9, 253),
                };
                const auto begin = lowered.instructions.begin() + lowered.source_instruction_offsets[2];
                REQUIRE(std::equal(expected.begin(), expected.end(), begin));
                REQUIRE(begin + expected.size() + 1 == lowered.instructions.end());
            }
        }
    }
}

TEST_CASE("Ampere per-tap gathers prove constants across unrelated arithmetic and gathers",
          "[video_core][ampere][gather]") {
    const std::array<Instruction, 6> code{{
        {{0x00067802, 0x01ff0002, 0x00000f00, 0x000fe400}},
        {{0x00077802, 0x02020200, 0x00000f00, 0x000fe400}},
        {{0x00057421, 0xbf000000, 0x00010000, 0x002fc800}},
        {{0x04037b63, 0x33800405, 0x041e02ff, 0x000f4200}},
        {{0x04007b63, 0x33800406, 0x041e2f06, 0x000f4400}},
        Exit,
    }};
    const auto lowered = LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    auto clobber = code;
    clobber[2].words[0] = 0x00067421;
    REQUIRE(LowerToMaxwell(clobber).error == "native per-tap gather nonconstant offsets");
    clobber = code;
    clobber[3].words[0] = 0x04067b63;
    REQUIRE(LowerToMaxwell(clobber).error == "native per-tap gather offset clobber");
    clobber = code;
    clobber[3].words[2] = 0x041e0f06;
    REQUIRE(LowerToMaxwell(clobber).error == "native per-tap gather offset clobber");
    auto branch = code;
    branch[2].words = {0x00007947, 0, 0x03800000, 0};
    REQUIRE(LowerToMaxwell(branch).error == "native per-tap gather control-flow proof");
}

TEST_CASE("Ampere per-tap gathers reject unproven offsets and unsupported modes",
          "[video_core][ampere][gather]") {
    // The captured instruction alone has no visible offset definitions.
    const auto alone = LowerToMaxwell(std::array{CapturedPerTapGather, Exit});
    REQUIRE_FALSE(alone);
    REQUIRE(alone.error == "native per-tap gather missing offsets");
    // Offsets are signed 6-bit values, one per byte.
    for (const u32 value : {32U, 127U, 128U, 223U}) {
        for (u32 shift = 0; shift < 32; shift += 8) {
            const auto lowered =
                LowerToMaxwell(PerTapProgram((0xff00ff00U & ~(255U << shift)) | (value << shift)));
            REQUIRE(lowered.error == "native per-tap gather offset range");
        }
    }
    for (const u32 bit : {60U, 72U, 76U, 78U, 80U, 81U, 89U, 91U}) {
        auto invalid = PerTapProgram();
        invalid[2].words[bit / 32] ^= 1U << (bit % 32);
        INFO("bit " << bit);
        REQUIRE_FALSE(LowerToMaxwell(invalid));
    }
    auto invalid = PerTapProgram();
    invalid[0].words[0] = 0x00081802; // predicated offset definition
    REQUIRE(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
    invalid = PerTapProgram();
    invalid[1].words[0] = 0x000a7802; // R9 never defined
    REQUIRE(LowerToMaxwell(invalid).error == "native per-tap gather missing offsets");
    // Register pairs that would reach the lowering scratch registers.
    for (const u32 word : {0xf50a7b63U, 0x04f57b63U}) {
        invalid = PerTapProgram();
        invalid[2].words[0] = word;
        REQUIRE_FALSE(LowerToMaxwell(invalid));
    }
    invalid = PerTapProgram();
    invalid[2].words[1] = 0x338004f5;
    REQUIRE_FALSE(LowerToMaxwell(invalid));
    invalid = PerTapProgram();
    invalid[2].words[2] = 0x041e2ff5;
    REQUIRE_FALSE(LowerToMaxwell(invalid));
    // Uniform-handle gathers have no per-tap form here.
    invalid = PerTapProgram();
    invalid[2].words[0] = 0x040a7f63;
    REQUIRE_FALSE(LowerToMaxwell(invalid));
    const Instruction branch{{0x00007947, 0, 0x03800000, 0}};
    const auto program = PerTapProgram();
    REQUIRE_FALSE(LowerToMaxwell(std::array{branch, program[0], program[1], program[2], Exit}));
    REQUIRE_FALSE(LowerToMaxwell(std::array{program[0], program[1], program[2], branch, Exit}));
}

TEST_CASE("Ampere gathers without per-tap offsets keep their previous lowering",
          "[video_core][ampere][gather]") {
    // Plain and AOFFI gathers still read no R248/R249 offsets.
    for (const u32 offsets : {0U, 1U}) {
        const Instruction gather{{0x040a7b63, 0x33800408, 0x041e0f08U | (offsets << 12), 0x100fc200}};
        const auto lowered = LowerToMaxwell(std::array{gather, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        const u64 tld4 = lowered.instructions[offsets ? 3 : 2];
        REQUIRE((tld4 >> 56) == 0xc8);
        REQUIRE(((tld4 >> 20) & 255) == (offsets ? 246U : 255U));
        REQUIRE(((tld4 >> 54) & 3) == offsets);
    }
}

TEST_CASE("Ampere depth gathers preserve coordinates reference and result pairs",
          "[video_core][ampere][gather]") {
    for (const u32 guard : {1U, 7U, 9U}) {
        for (const u32 nodep : {0U, 1U}) {
            Instruction gather{{0x180c7b63, 0x30800e1c, 0x001e4f0e, 0x000f4200}};
            gather.words[0] = (gather.words[0] & ~0xf000U) | (guard << 12);
            gather.words[2] |= nodep << 26;
            const auto lowered = LowerToMaxwell(std::array{gather, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            REQUIRE(lowered.required_texture_buffer == 2);
            const u64 pred = u64{guard} << 16;
            const std::array<u64, 8> expected{
                Copy(pred, 250, 24), Copy(pred, 251, 25), Copy(pred, 246, 28),
                0xc800000000000000ULL | pred | 250 | (250ULL << 8) | (246ULL << 20) |
                    (2ULL << 28) | (15ULL << 31) | (14ULL << 36) | (u64{nodep} << 49) |
                    (1ULL << 50) | (7ULL << 51),
                Copy(pred, 12, 250), Copy(pred, 13, 251),
                Copy(pred, 14, 252), Copy(pred, 15, 253),
            };
            REQUIRE(std::equal(expected.begin(), expected.end(), lowered.instructions.begin()));
            REQUIRE(lowered.instructions.size() == expected.size() + 1);
        }
    }
}

TEST_CASE("Ampere half gathers prove literal IMAD offsets across outgoing branches",
          "[video_core][ampere][gather]") {
    const std::array code{
        Instruction{{0xff007424, 0x00010000, 0x078e00ff, 0}},
        Instruction{{0xff017424, 0x01010100, 0x078e00ff, 0}},
        Instruction{{0x00000947, 16, 0x03800000, 0}},
        Instruction{{0x0a0c7b63, 0x30003000, 0x001eaf0d, 0}},
        Exit,
    };
    const auto lowered = LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    const auto start = lowered.source_instruction_offsets[3];
    CHECK(lowered.instructions[start] == MoveImmediate(7ULL << 16, 248, 0x00010000));
    CHECK(lowered.instructions[start + 1] == MoveImmediate(7ULL << 16, 249, 0x01010100));
    auto invalid = code;
    invalid[2].words[1] = 0;
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather control-flow proof");
    invalid = code;
    invalid[0].words[0] = 0xff001424;
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
    invalid = code;
    invalid[2].words = {0x00007805, 0, 0x0001ff00, 0};
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
}

TEST_CASE("Ampere guarded IMAD offsets remain constant across unrelated predicates",
          "[video_core][ampere][gather]") {
    const std::array code{
        Instruction{{0xff109424, 0xfffeffff, 0x078e00ff, 0}},
        Instruction{{0xff119424, 0xfffcfffd, 0x078e00ff, 0}},
        Instruction{{0x1c1a7212, 0xff, 0x0784a003, 0}},
        Instruction{{0x0000281c, 0, 0x03f0f070, 0}},
        Instruction{{0x00002947, 16, 0x03800000, 0}},
        Instruction{{0x06189b63, 0x30007c10, 0x001e2f0e, 0}},
        Exit,
    };
    const auto lowered = LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    const auto start = lowered.source_instruction_offsets[5];
    CHECK(lowered.instructions[start] == MoveImmediate(7ULL << 16, 248, 0xfffeffff));
    CHECK(lowered.instructions[start + 1] == MoveImmediate(7ULL << 16, 249, 0xfffcfffd));
    CHECK(((lowered.instructions[start + 4] >> 54) & 3) == 2);
    for (const u32 definition : {0U, 1U}) {
        for (const u32 guard : {1U, 2U, 7U}) {
            auto changed = code;
            changed[definition].words[0] = (changed[definition].words[0] & ~0xf000U) | (guard << 12);
            if (guard == 7) {
                CHECK(LowerToMaxwell(changed));
            } else {
                CHECK(LowerToMaxwell(changed).error == "native per-tap gather nonconstant offsets");
            }
        }
    }
    for (const u32 instruction : {2U, 3U}) {
        auto invalid = code;
        invalid[instruction].words[2] = (invalid[instruction].words[2] & ~(7U << 17)) | (1U << 17);
        CHECK(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
    }
    auto invalid = code;
    invalid[3].words[2] = (invalid[3].words[2] & ~(7U << 20)) | (1U << 20);
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
    invalid = code;
    invalid[2].words = {0x041c7810, 4, 0x07f3e0ff, 0};
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
    invalid = code;
    invalid[4].words[1] = 0;
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather control-flow proof");
    invalid = code;
    invalid[2].words = {0x00107805, 0, 0x0001ff00, 0};
    CHECK(LowerToMaxwell(invalid).error == "native per-tap gather nonconstant offsets");
}

TEST_CASE("Ampere half texture results keep the first component in the low lane",
          "[video_core][ampere][half_texture]") {
    const Instruction sample{{0x06007b60, 0x33c0440f, 0x059e8702, 0x101fc200}};
    for (const bool split : {false, true}) {
        for (const bool rgba : {false, true}) {
            auto instruction = sample;
            instruction.words[2] =
                (instruction.words[2] & ~0xfffU) | (split ? 2U : 255U) | (rgba ? 0xf00U : 0x700U);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            // Evaluate the F2F/ISCADD packing with known halves for each sampled component.
            std::array<u32, 256> converted{};
            converted[250] = 0x3c00;
            converted[251] = 0x3800;
            converted[252] = 0x3400;
            converted[253] = 0x3a00;
            std::array<u32, 256> registers{};
            u32 packed_registers = 0;
            for (const u64 word : lowered.instructions) {
                const u32 dest = word & 255;
                if ((word >> 48) == 0x5ca8) {
                    registers[dest] = converted[(word >> 20) & 255];
                } else if ((word >> 48) == 0x5c18) {
                    registers[dest] =
                        (registers[(word >> 8) & 255] << ((word >> 39) & 31)) + registers[(word >> 20) & 255];
                    ++packed_registers;
                }
            }
            REQUIRE(packed_registers == 2);
            REQUIRE(registers[instruction.Bits(16, 8)] == 0x38003c00);
            const u32 second = split ? instruction.Bits(64, 8) : instruction.Bits(16, 8) + 1;
            REQUIRE(registers[second] == (rgba ? 0x3a003400U : 0x00003400U));
        }
    }
}

TEST_CASE("Ampere half texel fetches use Maxwell TLDS half precision",
          "[video_core][ampere][half_texture]") {
    // TLD.LL.F16 R4 (R5 for the third channel), R4 (x, y), R14 (level), 2D, RGB.
    for (const u32 guard : {1U, 7U, 9U}) {
        for (const bool half : {false, true}) {
            const Instruction fetch{{0x04040b66U | (guard << 12), 0x33c0440e,
                                     0x059e0705U | (u32{half} << 15), 0x002f4200}};
            const auto lowered = LowerToMaxwell(std::array{fetch, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            REQUIRE(lowered.required_texture_buffer == 15);
            REQUIRE(lowered.instructions.size() == 2);
            // Maxwell TLDS: bit 59 clear selects F16 results, encoding 5 is 2D with a level.
            const u64 expected = (half ? 0xd200000000000000ULL : 0xda00000000000000ULL) |
                                 (u64{guard} << 16) | 4 | (4ULL << 8) | (14ULL << 20) | (5ULL << 28) |
                                 (0x44ULL << 36) | (1ULL << 49) | (0ULL << 50) | (5ULL << 53);
            REQUIRE(lowered.instructions[0] == expected);
        }
    }
}

TEST_CASE("Ampere half texel fetches reject forms without a TLDS half encoding",
          "[video_core][ampere][half_texture]") {
    const Instruction fetch{{0x04047b66, 0x33c0440e, 0x059e8705, 0x002f4200}};
    for (const u32 bit : {59U, 77U, 78U, 80U, 81U}) {
        auto invalid = fetch;
        invalid.words[bit / 32] ^= 1U << (bit % 32);
        INFO("bit " << bit);
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
    // A second descriptor buffer goes through TLD.B, which has no half form.
    const Instruction other_binding{{0x00007b66, 0x3080000e, 0x009e03ff, 0x002f4200}};
    REQUIRE(LowerToMaxwell(std::array{other_binding, Exit}));
    auto level_zero = fetch;
    level_zero.words[2] = (level_zero.words[2] & ~(7U << 23)) | (1U << 23);
    auto single = level_zero;
    single.words[2] &= ~(1U << 15);
    REQUIRE(LowerToMaxwell(std::array{other_binding, single, Exit}));
    const auto conflict = LowerToMaxwell(std::array{other_binding, level_zero, Exit});
    REQUIRE(conflict.error == "native texture descriptor buffer conflict");
    // Array fetches with a level go through TLD and are not packed.
    auto array_level = fetch;
    array_level.words[1] = (array_level.words[1] & ~(7U << 29)) | (5U << 29);
    REQUIRE_FALSE(LowerToMaxwell(std::array{array_level, Exit}));
}

TEST_CASE("Ampere array depth comparisons support nonlegacy descriptor bindings",
          "[video_core][ampere]") {
    // TEX.DC R5, R8 (layer, x), R0 (y, reference in R1), ARRAY_2D, descriptor buffer 4.
    const Instruction sample{{0x08057b60, 0xb1012400, 0x041e41ff, 0x10114400}};
    for (const u32 binding : {2U, 4U, 17U}) {
        auto instruction = sample;
        instruction.words[1] = (instruction.words[1] & ~(0x1fU << 22)) | (binding << 22);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.required_texture_buffer == binding);
        REQUIRE(lowered.instructions.size() == 5);
        const std::array<u32, 3> sources{8, 9, 0};
        for (u32 i = 0; i < sources.size(); ++i)
            REQUIRE(lowered.instructions[i] == Copy(7ULL << 16, 250 + i, sources[i]));
        REQUIRE(lowered.instructions[3] ==
                (0xc000000000070000ULL | 5 | (250ULL << 8) | (1ULL << 20) | (3ULL << 28) |
                 (1ULL << 31) | (0x124ULL << 36) | (1ULL << 49) | (1ULL << 50) | (7ULL << 51)));
    }
    auto invalid = sample;
    invalid.words[1] = (invalid.words[1] & ~(0x1fU << 22)) | (18U << 22);
    REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    // One bound descriptor buffer per program.
    auto other = sample;
    other.words[1] = (other.words[1] & ~(0x1fU << 22)) | (3U << 22);
    const auto mixed = LowerToMaxwell(std::array{sample, other, Exit});
    INFO(mixed.error);
    REQUIRE(mixed);
    REQUIRE(mixed.required_texture_buffer == 4);
    REQUIRE(mixed.instructions[3] ==
            (0xc000000000070000ULL | 5 | (250ULL << 8) | (1ULL << 20) | (3ULL << 28) |
             (1ULL << 31) | (0x124ULL << 36) | (1ULL << 49) | (1ULL << 50) | (7ULL << 51)));
    const std::array<u64, 7> alternate{
        Copy(7ULL << 16, 250, 8),
        Copy(7ULL << 16, 251, 9),
        Copy(7ULL << 16, 252, 0),
        0x4c98000000070000ULL | 246 | (15ULL << 39) | (0x124ULL << 20) | (3ULL << 34),
        Copy(7ULL << 16, 247, 1),
        0xde80000000070000ULL | 250 | (250ULL << 8) | (246ULL << 20) | (3ULL << 28) |
            (1ULL << 31) | (1ULL << 49) | (1ULL << 50) | (7ULL << 51),
        Copy(7ULL << 16, 5, 250),
    };
    const auto begin = mixed.instructions.begin() + mixed.source_instruction_offsets[1];
    REQUIRE(std::equal(alternate.begin(), alternate.end(), begin));
    REQUIRE(begin + alternate.size() + 1 == mixed.instructions.end());
}

TEST_CASE("Ampere gathers from a second descriptor buffer lower to Maxwell TLD4.B",
          "[video_core][ampere][gather]") {
    // The depth gather binds descriptor buffer 2; the gathers after it read buffer 14, whose
    // handle at c[14][0x10] is loaded with MOV and passed as the first TLD4.B meta register.
    const Instruction depth{{0x180c7b63, 0x30800e1c, 0x001e4f0e, 0x000f4200}};
    const u64 pred = 7ULL << 16;
    const u64 handle = 0x4c98000000070000ULL | (15ULL << 39) | (4ULL << 20) | (14ULL << 34);
    const auto tld4b = [&](u32 meta, u32 offset_type) -> u64 {
        return 0xdec0000000000000ULL | pred | 250 | (250ULL << 8) | (u64{meta} << 20) |
               (2ULL << 28) | (15ULL << 31) | (u64{offset_type} << 36) | (1ULL << 49) |
               (7ULL << 51);
    };
    const std::array<u64, 4> results{Copy(pred, 10, 250), Copy(pred, 11, 251), Copy(pred, 8, 252),
                                     Copy(pred, 9, 253)};
    const auto check = [&](const auto& lowered, u32 index, const auto& expected) {
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.required_texture_buffer == 2);
        const auto begin = lowered.instructions.begin() + lowered.source_instruction_offsets[index];
        REQUIRE(std::equal(expected.begin(), expected.end(), begin));
        REQUIRE(std::equal(results.begin(), results.end(), begin + expected.size()));
    };
    // Plain and AOFFI: the handle in R246, the offset register after it.
    const Instruction plain{{0x040a7b63, 0x33800408, 0x041e0f08, 0x100fc200}};
    check(LowerToMaxwell(std::array{depth, plain, Exit}), 1,
          std::array<u64, 4>{Copy(pred, 250, 4), Copy(pred, 251, 8), handle | 246, tld4b(246, 0)});
    auto aoffi = plain;
    aoffi.words[2] |= 1U << 12;
    check(LowerToMaxwell(std::array{depth, aoffi, Exit}), 1,
          std::array<u64, 5>{Copy(pred, 250, 4), Copy(pred, 251, 5), handle | 246, Copy(pred, 247, 8),
                     tld4b(246, 1)});
    // PTP: the handle in R247, so the packed offsets stay in R248 and R249.
    const auto per_tap = PerTapProgram();
    check(LowerToMaxwell(std::array{depth, per_tap[0], per_tap[1], per_tap[2], Exit}), 3,
          std::array<u64, 6>{MoveImmediate(7ULL << 16, 248, 0xff00ff00),
                     MoveImmediate(7ULL << 16, 249, 0x01000001), Copy(pred, 250, 4),
                     Copy(pred, 251, 5), handle | 247, tld4b(247, 2)});
    // Gathers from the bound buffer keep the bound TLD4.
    const auto same = LowerToMaxwell(std::array{plain, plain, Exit});
    INFO(same.error);
    REQUIRE(same);
    REQUIRE((same.instructions[same.source_instruction_offsets[1] + 2] >> 56) == 0xc8);
}
