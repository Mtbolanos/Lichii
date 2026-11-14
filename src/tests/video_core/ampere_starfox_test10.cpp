// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <unordered_map>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace {
namespace IR = Shader::IR;
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction CapturedI2F{{0x00027d06, 0x00000004, 0x08101000, 0x000e6200}};
constexpr Instruction CapturedUniformLoad{{0x00047ab9, 0x01800800, 0x00000400, 0x000fe200}};
constexpr Instruction CapturedShift{{0xff247219, 0x00000025, 0x00011444, 0x000fc600}};
constexpr Instruction CapturedGather{{0x201c7b63, 0xb0000a2b, 0x041e0f16, 0x000f4200}};
constexpr Instruction CapturedLut{{0x00c8781c, 0x00000000, 0x0010f010, 0x000fc400}};
constexpr std::array<Instruction, 6> CapturedGradients{{
    {{0x10180f6c, 0xa0000a14, 0x0c1e0f1a, 0x00114200}},
    {{0x14127f6c, 0xa0000410, 0x081e0f10, 0x001f4200}},
    {{0x18300f6c, 0xa000041c, 0x0c1e0f32, 0x00814200}},
    {{0x08027f6c, 0xa0000400, 0x081e0f00, 0x001f4200}},
    {{0x14107f6c, 0xa0000410, 0x081e0f12, 0x001f4200}},
    {{0x18320f6c, 0xa000041c, 0x0c1e0f34, 0x00414200}},
}};

void SetBits(Instruction& instruction, u32 bit, u32 width, u32 value) {
    const u32 mask = ((1U << width) - 1) << (bit % 32);
    instruction.words[bit / 32] =
        (instruction.words[bit / 32] & ~mask) | ((value << (bit % 32)) & mask);
}

class ConversionEnvironment final : public Shader::Environment {
public:
    u64 ReadInstruction(u32) override {
        FAIL("Unexpected instruction read");
        return 0;
    }
    u32 ReadCbufValue(u32, u32) override {
        FAIL("Unexpected constant buffer read");
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    bool HasNativeInstructions() const override { return true; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

u32 ConvertUniformHalfword(u64 word, u32 source_register, u32 input) {
    REQUIRE(Shader::Maxwell::Decode(word) == Shader::Maxwell::Opcode::I2F_reg);
    ConversionEnvironment env;
    Shader::ObjectPool<IR::Inst> pool{32};
    IR::Block block{pool};
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.I2F_reg(word);

    std::unordered_map<const IR::Inst*, u32> values;
    std::optional<u32> output;
    const auto read = [&](IR::Value value) -> u32 {
        if (!value.IsImmediate()) {
            return values.at(value.Inst());
        }
        return value.Type() == IR::Type::F32 ? std::bit_cast<u32>(value.F32()) : value.U32();
    };
    for (const IR::Inst& inst : block) {
        const auto arg = [&](size_t index) { return read(inst.Arg(index)); };
        u32 result{};
        switch (inst.GetOpcode()) {
        case IR::Opcode::GetRegister:
            REQUIRE(static_cast<u32>(inst.Arg(0).Reg()) == source_register);
            result = input;
            break;
        case IR::Opcode::BitFieldUExtract:
            REQUIRE(arg(2) == 16);
            REQUIRE(arg(1) <= 16);
            result = (arg(0) >> arg(1)) & 0xffffU;
            break;
        case IR::Opcode::ConvertF32U32:
            CHECK(inst.Flags<IR::FpControl>().rounding == IR::FpRounding::RN);
            result = std::bit_cast<u32>(static_cast<float>(arg(0)));
            break;
        case IR::Opcode::BitCastU32F32:
            result = arg(0);
            break;
        case IR::Opcode::SetRegister:
            REQUIRE(inst.Arg(0).Reg() == IR::Reg::R2);
            REQUIRE_FALSE(output.has_value());
            output = arg(1);
            break;
        default:
            FAIL("Unexpected conversion opcode " << IR::NameOf(inst.GetOpcode()));
        }
        values.emplace(&inst, result);
    }
    REQUIRE(output.has_value());
    return *output;
}

struct Translation {
    using Vector = std::array<u32, 4>;
    std::array<u32, IR::NUM_REGS> registers{};
    std::array<bool, 8> predicates{false, false, false, false, false, false, false, true};
    Vector sample{0x3f800000, 0xbf000000, 0x3e800000, 0x3f400000};
    Vector coordinates{}, derivatives{};
    IR::TextureInstInfo texture_info{};
    u32 handle{}, textures{};
    std::unordered_map<u32, u32> constants;
    u32 constant_reads{};

    void Run(const Shader::Ampere::LoweredProgram& lowered) {
        ConversionEnvironment env;
        for (const u64 word : lowered.instructions) {
            const auto opcode = Shader::Maxwell::Decode(word);
            if (opcode == Shader::Maxwell::Opcode::EXIT) continue;
            const u32 guard = static_cast<u32>((word >> 16) & 15);
            if (predicates[guard & 7] == ((guard & 8) != 0)) continue;
            Shader::ObjectPool<IR::Inst> pool{64};
            IR::Block block{pool};
            Shader::Maxwell::TranslatorVisitor visitor{env, block};
            switch (opcode) {
            case Shader::Maxwell::Opcode::I2F_reg: visitor.I2F_reg(word); break;
            case Shader::Maxwell::Opcode::MOV_reg: visitor.MOV_reg(word); break;
            case Shader::Maxwell::Opcode::MOV32I: visitor.MOV32I(word); break;
            case Shader::Maxwell::Opcode::IMNMX_imm: visitor.IMNMX_imm(word); break;
            case Shader::Maxwell::Opcode::SHR_reg: visitor.SHR_reg(word); break;
            case Shader::Maxwell::Opcode::SEL_imm: visitor.SEL_imm(word); break;
            case Shader::Maxwell::Opcode::LOP3_reg: visitor.LOP3_reg(word); break;
            case Shader::Maxwell::Opcode::LOP32I: visitor.LOP32I(word); break;
            case Shader::Maxwell::Opcode::LOP_reg: visitor.LOP_reg(word); break;
            case Shader::Maxwell::Opcode::R2P_imm: visitor.R2P_imm(word); break;
            case Shader::Maxwell::Opcode::P2R_imm: visitor.P2R_imm(word); break;
            case Shader::Maxwell::Opcode::BFE_imm: visitor.BFE_imm(word); break;
            case Shader::Maxwell::Opcode::IADD_reg: visitor.IADD_reg(word); break;
            case Shader::Maxwell::Opcode::TLD4: visitor.TLD4(word); break;
            case Shader::Maxwell::Opcode::TXD_b: visitor.TXD_b(word); break;
            case Shader::Maxwell::Opcode::LDC: visitor.LDC(word); break;
            default: FAIL("Unexpected lowered instruction " << static_cast<u32>(opcode));
            }
            std::unordered_map<const IR::Inst*, Vector> values;
            const auto read = [&](IR::Value value) -> Vector {
                if (!value.IsImmediate()) return values.at(value.Inst());
                switch (value.Type()) {
                case IR::Type::U1: return {static_cast<u32>(value.U1())};
                case IR::Type::U32: return {value.U32()};
                case IR::Type::F32: return {std::bit_cast<u32>(value.F32())};
                default: FAIL("Unexpected immediate type"); return {};
                }
            };
            for (const IR::Inst& inst : block) {
                const auto arg = [&](size_t index) { return read(inst.Arg(index))[0]; };
                Vector result{};
                switch (inst.GetOpcode()) {
                case IR::Opcode::GetCbufU16:
                    REQUIRE(arg(0) == 6);
                    REQUIRE((arg(1) & 1) == 0);
                    result[0] = (constants.at(arg(1) & ~3U) >> ((arg(1) & 2) * 8)) & 0xffffU;
                    ++constant_reads;
                    break;
                case IR::Opcode::GetRegister:
                    result[0] = inst.Arg(0).Reg() == IR::Reg::RZ
                                    ? 0U : registers.at(static_cast<size_t>(inst.Arg(0).Reg()));
                    break;
                case IR::Opcode::SetRegister:
                    if (inst.Arg(0).Reg() != IR::Reg::RZ)
                        registers.at(static_cast<size_t>(inst.Arg(0).Reg())) = arg(1);
                    break;
                case IR::Opcode::GetPred:
                    result[0] = predicates.at(static_cast<size_t>(inst.Arg(0).Pred()));
                    break;
                case IR::Opcode::SetPred:
                    predicates.at(static_cast<size_t>(inst.Arg(0).Pred())) = arg(1) != 0;
                    break;
                case IR::Opcode::BitCastF32U32:
                case IR::Opcode::BitCastU32F32: result[0] = arg(0); break;
                case IR::Opcode::BitwiseAnd32: result[0] = arg(0) & arg(1); break;
                case IR::Opcode::BitwiseOr32: result[0] = arg(0) | arg(1); break;
                case IR::Opcode::BitwiseXor32: result[0] = arg(0) ^ arg(1); break;
                case IR::Opcode::BitwiseNot32: result[0] = ~arg(0); break;
                case IR::Opcode::LogicalNot: result[0] = !arg(0); break;
                case IR::Opcode::IAdd32: result[0] = arg(0) + arg(1); break;
                case IR::Opcode::INeg32: result[0] = 0U - arg(0); break;
                case IR::Opcode::UMin32: result[0] = (std::min)(arg(0), arg(1)); break;
                case IR::Opcode::UMax32: result[0] = (std::max)(arg(0), arg(1)); break;
                case IR::Opcode::IEqual: result[0] = arg(0) == arg(1); break;
                case IR::Opcode::UGreaterThanEqual: result[0] = arg(0) >= arg(1); break;
                case IR::Opcode::SelectU1:
                case IR::Opcode::SelectU32: result[0] = arg(0) ? arg(1) : arg(2); break;
                case IR::Opcode::GetZeroFromOp: result[0] = arg(0) == 0; break;
                case IR::Opcode::BitFieldUExtract:
                    REQUIRE(arg(2) < 32);
                    REQUIRE(arg(1) + arg(2) <= 32);
                    result[0] = (arg(0) >> arg(1)) & ((1U << arg(2)) - 1);
                    break;
                case IR::Opcode::ShiftRightArithmetic32:
                    REQUIRE(arg(1) < 32);
                    result[0] = static_cast<u32>(std::bit_cast<s32>(arg(0)) >> arg(1));
                    break;
                case IR::Opcode::ConvertF32U16:
                    result[0] = std::bit_cast<u32>(static_cast<float>(arg(0) & 0xffffU));
                    break;
                case IR::Opcode::ConvertF32U32:
                    result[0] = std::bit_cast<u32>(static_cast<float>(arg(0)));
                    break;
                case IR::Opcode::CompositeConstructF32x2:
                    result = {arg(0), arg(1)};
                    break;
                case IR::Opcode::CompositeConstructF32x3:
                    result = {arg(0), arg(1), arg(2)};
                    break;
                case IR::Opcode::CompositeConstructF32x4:
                    result = {arg(0), arg(1), arg(2), arg(3)};
                    break;
                case IR::Opcode::CompositeExtractF32x4:
                    result[0] = read(inst.Arg(0)).at(arg(1));
                    break;
                case IR::Opcode::BoundImageGather:
                case IR::Opcode::BindlessImageGradient:
                    handle = arg(0);
                    coordinates = read(inst.Arg(1));
                    texture_info = inst.Flags<IR::TextureInstInfo>();
                    if (inst.GetOpcode() == IR::Opcode::BindlessImageGradient) {
                        derivatives = read(inst.Arg(2));
                        REQUIRE(inst.Arg(3).IsEmpty());
                        REQUIRE(inst.Arg(4).IsEmpty());
                    } else {
                        REQUIRE(inst.Arg(2).IsEmpty());
                        REQUIRE(inst.Arg(3).IsEmpty());
                    }
                    ++textures;
                    result = sample;
                    break;
                default: FAIL("Unexpected IR opcode " << IR::NameOf(inst.GetOpcode()));
                }
                values.emplace(&inst, result);
            }
        }
    }
};
} // namespace

TEST_CASE("Star Fox captured ULDC loads an unsigned halfword through the real translator",
          "[video_core][ampere][starfox_uldc]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedUniformLoad, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == 2);
    CHECK(lowered.instructions.front() == 0xef9200600207ffdcULL);
    for (const u32 input : {0x12340000U, 0x12340001U, 0x12347fffU, 0x12348000U, 0x1234ffffU}) {
        Translation translated;
        translated.constants.emplace(0x20, input);
        translated.registers.fill(0xdeadbeefU);
        const auto predicates = translated.predicates;
        translated.Run(lowered);
        CHECK(translated.constant_reads == 1);
        CHECK(translated.predicates == predicates);
        for (u32 reg = 0; reg < IR::NUM_REGS; ++reg)
            CHECK(translated.registers[reg] == (reg == 220 ? input & 0xffffU : 0xdeadbeefU));
    }
}

TEST_CASE("Star Fox ULDC U16 preserves halfword offsets uniform destinations and predicates",
          "[video_core][ampere][starfox_uldc]") {
    for (const u32 offset : {0U, 2U, 0x20U, 0x22U, 0x7ffeU}) {
        for (const u32 dest : {0U, 4U, 21U, 22U, 27U}) {
            for (const u32 guard : {0U, 7U, 8U}) {
                auto instruction = CapturedUniformLoad;
                SetBits(instruction, 38, 16, offset);
                SetBits(instruction, 16, 8, dest);
                SetBits(instruction, 12, 4, guard);
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                for (const bool enabled : {false, true}) {
                    Translation translated;
                    translated.constants.emplace(offset & ~3U, 0x8001fedcU);
                    translated.registers.fill(0xdeadbeefU);
                    translated.registers[239] = enabled ? 0x40U : 0U;
                    const auto registers = translated.registers;
                    const auto predicates = translated.predicates;
                    translated.Run(lowered);
                    const bool taken = guard == 7 || enabled != (guard == 8);
                    const u32 target = dest < 22 ? dest + 216 : dest + 218;
                    const u32 expected = offset & 2 ? 0x8001U : 0xfedcU;
                    CHECK(translated.constant_reads == (taken ? 1U : 0U));
                    CHECK(translated.predicates == predicates);
                    for (u32 reg = 0; reg < 246; ++reg)
                        CHECK(translated.registers[reg] ==
                              (taken && reg == target ? expected : registers[reg]));
                }
            }
        }
    }
}

TEST_CASE("Star Fox ULDC U16 rejects unrepresentable offsets and unsupported modes",
          "[video_core][ampere][starfox_uldc]") {
    for (const auto change : std::array{
             std::array{38U, 16U, 1U}, std::array{38U, 16U, 0x7fffU},
             std::array{38U, 16U, 0x8000U}, std::array{38U, 16U, 0xfffeU},
             std::array{16U, 8U, 63U}, std::array{16U, 8U, 64U},
             std::array{54U, 5U, 18U}, std::array{62U, 2U, 1U},
             std::array{73U, 3U, 6U}, std::array{78U, 2U, 2U},
             std::array{91U, 1U, 1U}}) {
        auto instruction = CapturedUniformLoad;
        SetBits(instruction, change[0], change[1], change[2]);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
    auto indexed = CapturedUniformLoad;
    SetBits(indexed, 0, 12, 0xabb);
    SetBits(indexed, 91, 1, 1);
    // Ocarina of Time loads U8 and U16 values through an index register.
    const auto indexed_load = LowerToMaxwell(std::array{indexed, Exit});
    INFO(indexed_load.error);
    CHECK(indexed_load);
    SetBits(indexed, 73, 3, 6);
    CHECK_FALSE(LowerToMaxwell(std::array{indexed, Exit}));
}

TEST_CASE("Star Fox Test10 uniform U16 conversion reaches the real translator",
          "[video_core][ampere][starfox_test10]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedI2F, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == 2);
    CHECK(lowered.instructions.front() ==
          (0x5cb8000000070602ULL | (u64{220} << 20)));
    for (const u32 input : {0U, 1U, 0xffffU, 0xffff0000U, 0x12348000U}) {
        CHECK(ConvertUniformHalfword(lowered.instructions.front(), 220, input) ==
              std::bit_cast<u32>(static_cast<float>(input & 0xffffU)));
    }
}

TEST_CASE("Star Fox upstream U16 port keeps selectors uniform mapping and signed registers",
          "[video_core][ampere][starfox_test10]") {
    for (const u32 source : {0U, 4U, 21U, 22U, 27U}) {
        const u32 mapped = source < 22 ? source + 216 : source + 218;
        for (const u32 half : {0U, 1U}) {
            auto instruction = CapturedI2F;
            instruction.words[1] = source | (half << 28);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            REQUIRE(lowered.instructions.size() == 2);
            CHECK(ConvertUniformHalfword(lowered.instructions.front(), mapped, 0x8001ffffU) ==
                  std::bit_cast<u32>(static_cast<float>(half == 0 ? 65535 : 32769)));
        }
    }
    for (const u32 sign : {0U, 1U}) {
        for (const u32 half : {0U, 1U}) {
            for (const u32 guard : {0U, 7U, 9U}) {
                const Instruction instruction{{0x00020306U | (guard << 12),
                                                3U | (half << 28),
                                                0x00101000U | (sign << 10), 0}};
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                REQUIRE(lowered);
                REQUIRE(lowered.instructions.size() == 2);
                CHECK(lowered.instructions.front() ==
                      (0x5cb8000000000602ULL | (u64{guard} << 16) | (u64{3} << 20) |
                       (u64{sign} << 13) | (u64{half * 2} << 41)));
            }
        }
    }
}

TEST_CASE("Star Fox uniform U16 conversion rejects invalid uniform sources and other modes",
          "[video_core][ampere][starfox_test10]") {
    for (const u32 source : {64U, 128U, 216U, 255U, 0x10004U, 0x20000004U, 0x30000004U}) {
        auto instruction = CapturedI2F;
        instruction.words[1] = source;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
    for (const u32 mode : {0x00101000U, 0x08109000U, 0x08100800U}) {
        auto instruction = CapturedI2F;
        instruction.words[2] = mode;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
    for (const u32 bit : {24U, 96U}) {
        auto instruction = CapturedI2F;
        SetBits(instruction, bit, 1, 1);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
}

TEST_CASE("Star Fox uniform U16 conversion preserves false guards and the zero uniform source",
          "[video_core][ampere][starfox_test10]") {
    for (const u32 source : {4U, 63U}) {
        for (const u32 guard : {0U, 7U, 8U}) {
            auto instruction = CapturedI2F;
            SetBits(instruction, 32, 8, source);
            SetBits(instruction, 12, 4, guard);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(lowered);
            for (const bool enabled : {false, true}) {
                Translation translated;
                translated.registers.fill(0xdeadbeef);
                translated.registers[220] = 0x1234ffff;
                translated.predicates[0] = enabled;
                const bool taken = translated.predicates[guard & 7] != ((guard & 8) != 0);
                translated.Run(lowered);
                const u32 expected = std::bit_cast<u32>(source == 63 ? 0.0f : 65535.0f);
                CHECK(translated.registers[2] == (taken ? expected : 0xdeadbeefU));
            }
        }
    }
}

TEST_CASE("Star Fox signed register shifts clamp wrap and preserve aliased inputs",
          "[video_core][ampere][starfox_test10]") {
    for (const u32 input : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU}) {
        for (const u32 shift : {0U, 1U, 31U, 32U, 63U, 0xffffffffU}) {
            for (const bool wrap : {false, true}) {
                for (const u32 destination : {36U, 37U, 68U}) {
                    for (const u32 guard : {0U, 7U, 8U}) {
                        auto instruction = CapturedShift;
                        SetBits(instruction, 16, 8, destination);
                        SetBits(instruction, 12, 4, guard);
                        SetBits(instruction, 75, 1, wrap);
                        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                        INFO(lowered.error);
                        REQUIRE(lowered);
                        for (const bool enabled : {false, true}) {
                            Translation translated;
                            translated.predicates[0] = enabled;
                            translated.registers.fill(0xdeadbeef);
                            translated.registers[68] = input;
                            translated.registers[37] = shift;
                            const auto before = translated.registers;
                            const auto predicates = translated.predicates;
                            const bool taken = predicates[guard & 7] != ((guard & 8) != 0);
                            translated.Run(lowered);
                            const u32 count = wrap ? shift & 31 : (std::min)(shift, 31U);
                            const u32 expected = static_cast<u32>(std::bit_cast<s32>(input) >> count);
                            for (u32 reg = 0; reg < 246; ++reg)
                                CHECK(translated.registers[reg] ==
                                      (taken && reg == destination ? expected : before[reg]));
                            CHECK(translated.predicates == predicates);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Star Fox signed register shift keeps other funnel modes rejected",
          "[video_core][ampere][starfox_test10]") {
    for (const auto change : std::array{
             std::array{24U, 8U, 3U}, std::array{73U, 2U, 0U}, std::array{73U, 2U, 1U},
             std::array{76U, 1U, 0U}, std::array{80U, 1U, 0U}, std::array{72U, 1U, 1U},
             std::array{32U, 16U, 0x125U}, std::array{0U, 12U, 0xc19U},
             std::array{96U, 1U, 1U}}) {
        auto instruction = CapturedShift;
        SetBits(instruction, change[0], change[1], change[2]);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
}

TEST_CASE("Star Fox array gather preserves layer coordinates components and split destinations",
          "[video_core][ampere][starfox_test10]") {
    for (const bool alias : {false, true}) {
        for (const u32 component : {0U, 1U, 2U, 3U}) {
            for (const u32 guard : {0U, 7U, 8U}) {
                auto instruction = CapturedGather;
                SetBits(instruction, 87, 2, component);
                SetBits(instruction, 12, 4, guard);
                if (alias) {
                    SetBits(instruction, 16, 8, 32);
                    SetBits(instruction, 64, 8, 42);
                }
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                CHECK(lowered.required_texture_buffer == 0);
                for (const bool enabled : {false, true}) {
                    Translation translated;
                    translated.predicates[0] = enabled;
                    translated.registers.fill(0xdeadbeef);
                    translated.registers[32] = 0xface0003;
                    translated.registers[33] = 0x3e800000;
                    translated.registers[43] = 0x3f400000;
                    auto expected = translated.registers;
                    const bool taken = translated.predicates[guard & 7] != ((guard & 8) != 0);
                    if (taken) {
                        for (u32 i = 0; i < 4; ++i) {
                            const u32 reg = i < 2 ? instruction.Bits(16, 8) + i
                                                 : instruction.Bits(64, 8) + i - 2;
                            expected[reg] = translated.sample[i];
                        }
                    }
                    translated.Run(lowered);
                    CHECK(translated.textures == static_cast<u32>(taken));
                    if (taken) {
                        CHECK(translated.handle == 40);
                        CHECK(translated.coordinates == Translation::Vector{0x3e800000, 0x3f400000, 0x40400000});
                        CHECK(translated.texture_info.type == Shader::TextureType::ColorArray2D);
                        CHECK(translated.texture_info.gather_component == component);
                        CHECK(translated.texture_info.is_depth == 0);
                    }
                    for (u32 reg = 0; reg < 246; ++reg)
                        CHECK(translated.registers[reg] == expected[reg]);
                }
            }
        }
    }
}

TEST_CASE("Star Fox array gather rejects unsupported modes and scratch crossings",
          "[video_core][ampere][starfox_test10]") {
    for (const auto change : std::array{
             std::array{60U, 1U, 0U}, std::array{61U, 3U, 7U}, std::array{76U, 2U, 1U},
             std::array{76U, 2U, 2U}, std::array{78U, 1U, 1U}, std::array{79U, 1U, 1U},
             std::array{80U, 1U, 1U}, std::array{81U, 3U, 0U}, std::array{72U, 4U, 3U},
             std::array{24U, 8U, 245U}, std::array{32U, 8U, 255U}, std::array{64U, 8U, 245U},
             std::array{64U, 8U, 28U}, std::array{0U, 12U, 0xf63U}}) {
        auto instruction = CapturedGather;
        SetBits(instruction, change[0], change[1], change[2]);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
    auto instruction = CapturedGather;
    SetBits(instruction, 24, 8, 215);
    const auto lowered = LowerToMaxwell(std::array{CapturedI2F, instruction, Exit});
    CHECK_FALSE(lowered);
    CHECK(lowered.instructions.empty());
}

TEST_CASE("Star Fox captured uniform TXD instructions preserve handles gradients and output aliases",
          "[video_core][ampere][starfox_test10]") {
    for (const auto instruction : CapturedGradients) {
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        CHECK(lowered.required_texture_buffer == ~0U);
        for (const bool enabled : {false, true}) {
            Translation translated;
            translated.predicates[0] = enabled;
            translated.registers.fill(0xdeadbeef);
            const u32 source = instruction.Bits(24, 8), gradient = instruction.Bits(32, 8);
            translated.registers[216 + instruction.Bits(40, 6)] = 0x12345678;
            translated.registers[source] = 0x3e800000;
            translated.registers[source + 1] = 0x3f400000;
            translated.registers[source + 2] = 0xface0005;
            const Translation::Vector derivatives{0x3d800000, 0x3e000000, 0x3e800000, 0x3f000000};
            for (u32 i = 0; i < 4; ++i) translated.registers[gradient + i] = derivatives[i];
            auto expected = translated.registers;
            const u32 guard = instruction.Bits(12, 4);
            const bool taken = translated.predicates[guard & 7] != ((guard & 8) != 0);
            if (taken) {
                for (u32 i = 0; i < 4; ++i) {
                    const u32 reg = i < 2 ? instruction.Bits(16, 8) + i
                                         : instruction.Bits(64, 8) + i - 2;
                    expected[reg] = translated.sample[i];
                }
            }
            translated.Run(lowered);
            CHECK(translated.textures == static_cast<u32>(taken));
            if (taken) {
                CHECK(translated.handle == 0x12345678);
                CHECK(translated.coordinates == Translation::Vector{0x3e800000, 0x3f400000, 0x40a00000});
                CHECK(translated.derivatives == derivatives);
                CHECK(translated.texture_info.type == Shader::TextureType::ColorArray2D);
                CHECK(translated.texture_info.num_derivatives == 2);
                CHECK(translated.texture_info.has_lod_clamp == 0);
            }
            for (u32 reg = 0; reg < 246; ++reg)
                CHECK(translated.registers[reg] == expected[reg]);
        }
    }
}

TEST_CASE("Star Fox uniform TXD maps uniform banks and preserves partial masks",
          "[video_core][ampere][starfox_test10]") {
    for (const u32 uniform : {0U, 21U, 22U, 27U, 28U, 29U, 62U}) {
        for (const u32 mask : {1U, 3U, 5U, 7U, 15U}) {
            for (const u32 dimension : {1U, 5U}) {
                auto instruction = CapturedGradients[1];
                SetBits(instruction, 40, 6, uniform);
                SetBits(instruction, 61, 3, dimension);
                SetBits(instruction, 72, 4, mask);
                const u32 channels = static_cast<u32>(std::popcount(mask));
                SetBits(instruction, 64, 8, channels > 2 ? 16U : 255U);
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                Translation translated;
                translated.registers.fill(0xdeadbeef);
                const u32 mapped = uniform < 22 ? uniform + 216
                                   : uniform < 28 ? uniform + 218 : 214 + (uniform & 1);
                translated.registers[mapped] = 0x24681234;
                translated.registers[20] = 0x3e800000;
                translated.registers[21] = 0x3f400000;
                translated.registers[22] = 5;
                auto expected = translated.registers;
                u32 output = 0;
                for (u32 channel = 0; channel < 4; ++channel) {
                    if (!(mask & (1U << channel))) continue;
                    expected[output < 2 ? 18 + output : 16 + output - 2] = translated.sample[channel];
                    ++output;
                }
                translated.Run(lowered);
                CHECK(translated.textures == 1);
                CHECK(translated.handle == 0x24681234);
                CHECK(translated.coordinates ==
                      Translation::Vector{0x3e800000, 0x3f400000, dimension == 5 ? 0x40a00000U : 0U});
                CHECK(translated.texture_info.type == (dimension == 5 ? Shader::TextureType::ColorArray2D
                                                                      : Shader::TextureType::Color2D));
                for (u32 reg = 0; reg < 246; ++reg)
                    CHECK(translated.registers[reg] == expected[reg]);
            }
        }
    }
}

TEST_CASE("Star Fox uniform TXD rejects unsupported modes and reserved register spans",
          "[video_core][ampere][starfox_test10]") {
    for (const auto change : std::array{
             std::array{40U, 6U, 63U}, std::array{46U, 1U, 1U}, std::array{59U, 1U, 1U},
             std::array{60U, 1U, 1U}, std::array{61U, 3U, 7U}, std::array{76U, 1U, 1U},
             std::array{77U, 1U, 1U}, std::array{79U, 1U, 1U}, std::array{91U, 1U, 0U},
             std::array{24U, 8U, 214U}, std::array{32U, 8U, 213U}, std::array{16U, 8U, 215U},
             std::array{64U, 8U, 17U}, std::array{64U, 8U, 18U}, std::array{72U, 4U, 0U}}) {
        auto instruction = CapturedGradients[1];
        SetBits(instruction, change[0], change[1], change[2]);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
    auto instruction = CapturedGradients[1];
    SetBits(instruction, 40, 6, 28);
    SetBits(instruction, 24, 8, 212);
    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
    CHECK_FALSE(lowered);
    CHECK(lowered.instructions.empty());
}

TEST_CASE("Star Fox dual predicate LUT snapshots aliased inputs and execution predicates",
          "[video_core][ampere][starfox_test10]") {
    for (const bool captured_inputs : {false, true}) {
        for (const u32 primary_lut : {0x80U, 0x96U, 0xffU}) {
            for (const u32 second_lut : {0xc8U, 0xf8U, 0xffU}) {
                for (const u32 guard : {0U, 1U, 7U, 8U, 9U}) {
                    auto instruction = CapturedLut;
                    SetBits(instruction, 12, 4, guard);
                    SetBits(instruction, 16, 8, second_lut);
                    SetBits(instruction, 64, 3, primary_lut & 7);
                    SetBits(instruction, 72, 5, primary_lut >> 3);
                    if (!captured_inputs) SetBits(instruction, 77, 4, 2);
                    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                    INFO(lowered.error);
                    REQUIRE(lowered);
                    for (u32 bits = 0; bits < 8; ++bits) {
                        Translation translated;
                        for (u32 i = 0; i < 3; ++i) translated.predicates[i] = (bits >> i) & 1;
                        translated.predicates[4] = true;
                        const auto registers = translated.registers;
                        auto expected = translated.predicates;
                        const bool taken = expected[guard & 7] != ((guard & 8) != 0);
                        const u32 index = (u32{expected[0]} << 2) |
                                          (u32{captured_inputs || expected[2]} << 1) | u32{expected[1]};
                        if (taken) {
                            expected[0] = (primary_lut >> index) & 1;
                            expected[1] = (second_lut >> index) & 1;
                        }
                        translated.Run(lowered);
                        CHECK(translated.predicates == expected);
                        for (u32 reg = 0; reg < 246; ++reg)
                            CHECK(translated.registers[reg] == registers[reg]);
                    }
                }
            }
        }
    }
}

TEST_CASE("Star Fox dual LUT preserves uniform predicate inputs negations and discarded output",
          "[video_core][ampere][starfox_test10]") {
    for (const u32 operand : {0U, 1U, 7U, 8U, 9U, 15U}) {
        for (const u32 primary : {0U, 7U}) {
            for (const u32 negations : {0U, 1U, 2U, 3U}) {
                auto instruction = CapturedLut;
                SetBits(instruction, 67, 1, 1);
                SetBits(instruction, 68, 4, operand);
                SetBits(instruction, 81, 3, primary);
                SetBits(instruction, 87, 4, (negations & 1) ? 8 : 0);
                SetBits(instruction, 77, 4, (negations & 2) ? 10 : 2);
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                for (u32 bits = 0; bits < 8; ++bits) {
                    Translation translated;
                    translated.predicates[0] = bits & 1;
                    translated.predicates[2] = bits & 2;
                    translated.registers[239] = (bits & 4) ? 0x7fU : 0U;
                    auto expected = translated.predicates;
                    const bool a = expected[0] != ((negations & 1) != 0);
                    const bool b = expected[2] != ((negations & 2) != 0);
                    const bool c = ((operand & 7) == 7 || (bits & 4)) != ((operand & 8) != 0);
                    const u32 index = (u32{a} << 2) | (u32{b} << 1) | u32{c};
                    if (primary != 7) expected[primary] = (0x80U >> index) & 1;
                    expected[1] = (0xc8U >> index) & 1;
                    translated.Run(lowered);
                    CHECK(translated.predicates == expected);
                    CHECK(translated.registers[239] == ((bits & 4) ? 0x7fU : 0U));
                }
            }
        }
    }
}

TEST_CASE("Star Fox dual predicate LUT rejects ambiguous destinations and reserved fields",
          "[video_core][ampere][starfox_test10]") {
    for (const auto change : std::array{
             std::array{81U, 3U, 1U}, std::array{24U, 8U, 1U}, std::array{32U, 1U, 1U},
             std::array{91U, 1U, 1U}, std::array{96U, 1U, 1U}}) {
        auto instruction = CapturedLut;
        SetBits(instruction, change[0], change[1], change[2]);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
}
