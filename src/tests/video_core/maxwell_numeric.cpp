// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/glasm/emit_glasm.h"
#include "shader_recompiler/backend/glsl/emit_glsl.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
namespace IR = Shader::IR;
using IR::Opcode;

class NumericEnvironment final : public Shader::Environment {
public:
    u64 ReadInstruction(u32) override { throw std::logic_error("Unexpected instruction read"); }
    u32 ReadCbufValue(u32, u32) override { throw std::logic_error("Unexpected cbuf read"); }
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
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

f64 HalfValue(u16 bits) {
    const int exponent = (bits >> 10) & 31;
    const f64 fraction = bits & 1023;
    const f64 magnitude = exponent == 31
                              ? (fraction == 0 ? std::numeric_limits<f64>::infinity()
                                               : std::numeric_limits<f64>::quiet_NaN())
                              : std::ldexp(exponent == 0 ? fraction : fraction + 1024,
                                           exponent == 0 ? -24 : exponent - 25);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

struct NumericProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block block{pool};
    NumericEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    std::array<u32, IR::NUM_REGS> registers{};
    std::unordered_map<const IR::Inst*, u64> values;

    explicit NumericProgram(bool to_float, u32 format, u32 rounding, bool is_signed = false,
                            u32 source_format = 2, bool negate = false, bool absolute = false) {
        const u64 insn = u64{format} << 8 | u64{source_format} << 10 | u64{2} << 20 |
                         u64{rounding} << 39 |
                         u64{is_signed} << (to_float ? 13 : 12) |
                         u64{negate} << (to_float ? 45 : 49) |
                         u64{absolute} << (to_float ? 49 : 45);
        if (to_float) {
            visitor.I2F_reg(insn);
        } else {
            visitor.F2I_reg(insn);
        }
    }

    void Lower() {
        IR::Program program;
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        Shader::Optimization::LowerInt64ToInt32(program);
        REQUIRE(std::ranges::none_of(block.Instructions(), [](const auto& inst) {
            return inst.Type() == IR::Type::U64;
        }));
    }

    u64 Value(IR::Value value) const {
        if (!value.IsImmediate()) {
            return values.at(value.Inst());
        }
        switch (value.Type()) {
        case IR::Type::U1: return value.U1();
        case IR::Type::U32: return value.U32();
        case IR::Type::U64: return value.U64();
        case IR::Type::F32: return std::bit_cast<u32>(value.F32());
        case IR::Type::F64: return std::bit_cast<u64>(value.F64());
        default: throw std::logic_error("Unsupported immediate");
        }
    }

    f64 Float(IR::Value value) const {
        const u64 bits = Value(value);
        switch (value.Type()) {
        case IR::Type::F16: return HalfValue(static_cast<u16>(bits));
        case IR::Type::F32: return std::bit_cast<f32>(static_cast<u32>(bits));
        case IR::Type::F64: return std::bit_cast<f64>(bits);
        default: throw std::logic_error("Expected floating-point value");
        }
    }

    u64 Evaluate(const IR::Inst& inst) const {
        const auto a = [&](size_t index) { return Value(inst.Arg(index)); };
        const auto f = [&](size_t index) { return Float(inst.Arg(index)); };
        const auto fp = [&](f64 result) -> u64 {
            if (inst.Type() == IR::Type::F64) {
                return std::bit_cast<u64>(result);
            }
            if (inst.Type() == IR::Type::F16) {
                for (u32 bits = 0; bits < 65536; ++bits) {
                    if (HalfValue(static_cast<u16>(bits)) == result ||
                        (std::isnan(result) && std::isnan(HalfValue(static_cast<u16>(bits))))) {
                        return bits;
                    }
                }
                throw std::logic_error("Half result is not exact");
            }
            return std::bit_cast<u32>(static_cast<f32>(result));
        };
        const auto convert = [&](bool is_signed, unsigned width) -> u64 {
            const f64 value = f(0);
            const f64 bound = std::ldexp(1.0, static_cast<int>(width - is_signed));
            if (!std::isfinite(value) || value < (is_signed ? -bound : 0.0) || value >= bound) {
                throw std::domain_error("Out-of-range floating-point to integer IR conversion");
            }
            return is_signed ? static_cast<u64>(static_cast<s64>(value)) : static_cast<u64>(value);
        };
        switch (inst.GetOpcode()) {
        case Opcode::Void: return 0;
        case Opcode::Identity: return a(0);
        case Opcode::GetRegister: return registers.at(static_cast<size_t>(inst.Arg(0).Reg()));
        case Opcode::BitCastF32U32:
        case Opcode::BitCastU32F32:
        case Opcode::PackUint2x32:
        case Opcode::UnpackUint2x32:
        case Opcode::PackDouble2x32: return a(0);
        case Opcode::CompositeConstructU32x2: return u32(a(0)) | a(1) << 32;
        case Opcode::CompositeExtractU32x2:
        case Opcode::CompositeExtractF16x2: return u32(a(0) >> (a(1) * 32));
        case Opcode::UnpackFloat2x16: return (a(0) & 65535) | (a(0) >> 16) << 32;
        case Opcode::SelectU1:
        case Opcode::SelectU32:
        case Opcode::SelectU64:
        case Opcode::SelectF32:
        case Opcode::SelectF64: return a(0) != 0 ? a(1) : a(2);
        case Opcode::IAdd32: return u32(a(0) + a(1));
        case Opcode::ISub32: return u32(a(0) - a(1));
        case Opcode::UMax32: return std::max(u32(a(0)), u32(a(1)));
        case Opcode::BitwiseAnd32: return a(0) & a(1);
        case Opcode::BitwiseOr32: return a(0) | a(1);
        case Opcode::ShiftRightLogical32: return u32(a(0)) >> (a(1) & 31);
        case Opcode::ShiftLeftLogical32: return u32(a(0) << (a(1) & 31));
        case Opcode::ShiftRightLogical64: return a(0) >> (a(1) & 63);
        case Opcode::ShiftLeftLogical64: return a(0) << (a(1) & 63);
        case Opcode::BitFieldUExtract: {
            const u32 offset = std::min(u32(a(1)), 32U);
            const u32 count = std::min(u32(a(2)), 32 - offset);
            return count == 0 ? 0 : (a(0) >> offset) & ((u64{1} << count) - 1);
        }
        case Opcode::BitFieldInsert: {
            const u32 offset = std::min(u32(a(2)), 32U);
            const u32 count = std::min(u32(a(3)), 32 - offset);
            const u64 mask = ((u64{1} << count) - 1) << offset;
            return u32((a(0) & ~mask) | ((a(1) << offset) & mask));
        }
        case Opcode::FindUMsb32: return a(0) == 0 ? ~0U : 31U - std::countl_zero(u32(a(0)));
        case Opcode::IEqual: return a(0) == a(1);
        case Opcode::INotEqual: return a(0) != a(1);
        case Opcode::UGreaterThan: return u32(a(0)) > u32(a(1));
        case Opcode::SGreaterThanEqual: return s32(a(0)) >= s32(a(1));
        case Opcode::LogicalAnd: return a(0) != 0 && a(1) != 0;
        case Opcode::LogicalOr: return a(0) != 0 || a(1) != 0;
        case Opcode::LogicalNot: return a(0) == 0;
        case Opcode::FPAbs16:
        case Opcode::FPAbs32:
        case Opcode::FPAbs64: return fp(std::abs(f(0)));
        case Opcode::FPNeg16:
        case Opcode::FPNeg32:
        case Opcode::FPNeg64: return fp(-f(0));
        case Opcode::FPClamp16:
        case Opcode::FPClamp32:
        case Opcode::FPClamp64:
            if (std::isnan(f(0))) {
                throw std::domain_error("NaN-dependent floating-point clamp");
            }
            return fp(std::fmin(std::fmax(f(0), f(1)), f(2)));
        case Opcode::FPRoundEven16:
        case Opcode::FPRoundEven32:
        case Opcode::FPRoundEven64: {
            const f64 low = std::floor(f(0));
            const f64 fraction = f(0) - low;
            return fp(fraction < 0.5 || (fraction == 0.5 && std::fmod(low, 2.0) == 0)
                          ? low : low + 1);
        }
        case Opcode::FPFloor16:
        case Opcode::FPFloor32:
        case Opcode::FPFloor64: return fp(std::floor(f(0)));
        case Opcode::FPCeil16:
        case Opcode::FPCeil32:
        case Opcode::FPCeil64: return fp(std::ceil(f(0)));
        case Opcode::FPTrunc16:
        case Opcode::FPTrunc32:
        case Opcode::FPTrunc64: return fp(std::trunc(f(0)));
        case Opcode::FPIsNan16:
        case Opcode::FPIsNan32:
        case Opcode::FPIsNan64: return std::isnan(f(0));
        case Opcode::FPOrdGreaterThanEqual32: return f(0) >= f(1);
        case Opcode::FPMul32: return fp(f(0) * f(1));
        case Opcode::FPFma32: return fp(std::fma(f32(f(0)), f32(f(1)), f32(f(2))));
        case Opcode::ConvertF16F32:
        case Opcode::ConvertF32F16:
        case Opcode::ConvertF32F64: return fp(f(0));
        case Opcode::ConvertS32F16:
        case Opcode::ConvertS32F32:
        case Opcode::ConvertS32F64: return u32(convert(true, 32));
        case Opcode::ConvertU32F16:
        case Opcode::ConvertU32F32:
        case Opcode::ConvertU32F64: return u32(convert(false, 32));
        case Opcode::ConvertU64F32: return convert(false, 64);
        case Opcode::ConvertF32U32: return fp(static_cast<f32>(u32(a(0))));
        case Opcode::ConvertF32U64: return fp(static_cast<f32>(a(0)));
        case Opcode::ConvertF32S32: return fp(static_cast<f32>(s32(a(0))));
        default: throw std::logic_error(std::string{IR::NameOf(inst.GetOpcode())});
        }
    }

    u64 Run(u64 input) {
        registers.fill(0xdeadbeefU);
        registers[2] = u32(input);
        registers[3] = u32(input >> 32);
        values.clear();
        for (const auto& inst : block.Instructions()) {
            if (inst.GetOpcode() == Opcode::SetRegister) {
                registers.at(static_cast<size_t>(inst.Arg(0).Reg())) = u32(Value(inst.Arg(1)));
            } else {
                values.emplace(&inst, Evaluate(inst));
            }
        }
        REQUIRE(registers[2] == u32(input));
        REQUIRE(registers[3] == u32(input >> 32));
        REQUIRE(registers[4] == 0xdeadbeefU);
        return registers[0] | u64{registers[1]} << 32;
    }
};

u32 UnsignedFloatOracle(u64 value, u32 rounding) {
    if (value == 0) {
        return 0;
    }
    f32 lower = static_cast<f32>(value);
    if (lower >= 0x1p64f || static_cast<u64>(lower) > value) {
        lower = std::nextafter(lower, 0.0f);
    }
    if (static_cast<u64>(lower) == value || rounding == 1 || rounding == 3) {
        return std::bit_cast<u32>(lower);
    }
    const f32 upper = std::nextafter(lower, std::numeric_limits<f32>::infinity());
    if (rounding == 2) {
        return std::bit_cast<u32>(upper);
    }
    const u64 below = value - static_cast<u64>(lower);
    const u64 above = upper >= 0x1p64f ? ~value + 1 : static_cast<u64>(upper) - value;
    return std::bit_cast<u32>(below < above || (below == above &&
                                              (std::bit_cast<u32>(lower) & 1) == 0)
                                 ? lower : upper);
}

std::vector<u64> UnsignedInputs() {
    std::vector<u64> inputs{0, 1, 0xffffffffULL, 0x100000000ULL, ~u64{0}};
    for (unsigned exponent = 0; exponent < 64; ++exponent) {
        const u64 power = u64{1} << exponent;
        inputs.insert(inputs.end(), {power - 1, power, power + 1});
        if (exponent >= 24) {
            const u64 half_ulp = u64{1} << (exponent - 24);
            for (u64 multiple : {1, 3, 5}) {
                const u64 tie = power + multiple * half_ulp;
                inputs.insert(inputs.end(), {tie - 1, tie, tie + 1});
            }
        }
    }
    std::mt19937_64 random{0x64f32};
    for (unsigned i = 0; i < 512; ++i) {
        inputs.push_back(random());
    }
    return inputs;
}
} // namespace

TEST_CASE("Maxwell unsigned I2F has backend-independent rounding", "[shader][maxwell][numeric]") {
    const auto inputs = UnsignedInputs();
    for (bool lowered : {false, true}) {
        for (u32 rounding = 0; rounding < 4; ++rounding) {
            for (u32 modifiers = 0; modifiers < 4; ++modifiers) {
                const bool negate = (modifiers & 1) != 0;
                const bool absolute = (modifiers & 2) != 0;
                NumericProgram program{true, 2, rounding, false, 3, negate, absolute};
                if (lowered) {
                    program.Lower();
                }
                for (const u64 input : inputs) {
                    INFO("U64 " << input << " rounding " << rounding << " lowered " << lowered
                                << " modifiers " << modifiers);
                    const u32 expected = UnsignedFloatOracle(input, rounding) ^
                                         (negate ? 0x80000000U : 0);
                    REQUIRE(u32(program.Run(input)) == expected);
                }
            }
        }
    }
}

TEST_CASE("Maxwell U32 to F32 truncates without backend rounding support", "[shader][maxwell][numeric]") {
    const auto inputs = UnsignedInputs();
    for (u32 modifiers = 0; modifiers < 4; ++modifiers) {
        const bool negate = (modifiers & 1) != 0;
        const bool absolute = (modifiers & 2) != 0;
        NumericProgram program{true, 2, 3, false, 2, negate, absolute};
        for (const u64 input : inputs) {
            const u32 narrowed = u32(input);
            INFO("U32 " << narrowed << " modifiers " << modifiers);
            const u32 expected = UnsignedFloatOracle(narrowed, 3) ^ (negate ? 0x80000000U : 0);
            REQUIRE(u32(program.Run(narrowed)) == expected);
        }
    }
}

TEST_CASE("Maxwell I2F keeps signed and F64 conversion routes", "[shader][maxwell][numeric]") {
    for (const u32 rounding : {0U, 1U, 2U, 3U}) {
        for (const bool negate : {false, true}) {
            for (const bool absolute : {false, true}) {
                NumericProgram program{true, 2, rounding, true, 3, negate, absolute};
                size_t conversions = 0;
                for (const auto& inst : program.block.Instructions()) {
                    if (inst.GetOpcode() == Opcode::ConvertF32S64) {
                        ++conversions;
                        REQUIRE(inst.Flags<IR::FpControl>().rounding ==
                                static_cast<IR::FpRounding>(rounding + 1));
                    }
                }
                REQUIRE(conversions == 1);
            }
        }
        NumericProgram program{true, 3, rounding, false, 3};
        REQUIRE(std::ranges::count_if(program.block.Instructions(), [](const auto& inst) {
                    return inst.GetOpcode() == Opcode::ConvertF64U64;
                }) == 1);
    }
}

TEST_CASE("Maxwell F32 to U64 never converts an out-of-range operand", "[shader][maxwell][numeric]") {
    const std::array<u32, 20> inputs{
        0, 0x80000000U, 1, 0x80000001U, 0xbf000000U, 0xbf800000U,
        0x3f000000U, 0x3fc00000U, 0x40200000U, 0x4b800001U, 0x5f000000U,
        0x5f7ffffeU, 0x5f7fffffU, 0x5f800000U, 0x5f800001U,
        0x7f800000U, 0xff800000U, 0x7fc01234U, 0xffc01234U, 0x7f801234U};
    for (bool lowered : {false, true}) {
        for (u32 rounding = 0; rounding < 4; ++rounding) {
            for (u32 modifiers = 0; modifiers < 4; ++modifiers) {
                const bool negate = (modifiers & 1) != 0;
                const bool absolute = (modifiers & 2) != 0;
                NumericProgram program{false, 3, rounding, false, 2, negate, absolute};
                if (lowered) {
                    program.Lower();
                }
                for (const u32 bits : inputs) {
                    INFO("F32 bits " << bits << " rounding " << rounding << " lowered " << lowered
                                     << " modifiers " << modifiers);
                    f64 value = std::bit_cast<f32>(bits);
                    if (absolute) {
                        value = std::abs(value);
                    }
                    if (negate) {
                        value = -value;
                    }
                    u64 expected;
                    if (std::isnan(value)) {
                        expected = 0x8000000000000000ULL;
                    } else if (value >= 0x1p64) {
                        expected = ~u64{0};
                    } else if (value <= 0) {
                        expected = 0;
                    } else {
                        const f64 rounded = rounding == 0 ? std::nearbyint(value)
                                            : rounding == 1 ? std::floor(value)
                                            : rounding == 2 ? std::ceil(value) : std::trunc(value);
                        expected = static_cast<u64>(rounded);
                    }
                    REQUIRE(program.Run(bits) == expected);
                }
            }
        }
    }
}

TEST_CASE("Maxwell F32 to signed32 saturates before integer conversion",
          "[shader][maxwell][numeric]") {
    const std::array<u32, 24> inputs{
        0U,          0x80000000U, 1U,          0x80000001U, 0x007fffffU, 0x807fffffU,
        0x3f000000U, 0xbf000000U, 0x3fc00000U, 0xbfc00000U, 0x4effffffU, 0x4f000000U,
        0x4f000001U, 0xceffffffU, 0xcf000000U, 0xcf000001U, 0x7f7fffffU, 0xff7fffffU,
        0x7f800000U, 0xff800000U, 0x7fc00000U, 0xffc12345U, 0x7f800001U, 0xff800001U};
    for (u32 rounding = 0; rounding < 4; ++rounding) {
        for (u32 modifiers = 0; modifiers < 4; ++modifiers) {
            const bool negate = (modifiers & 1) != 0;
            const bool absolute = (modifiers & 2) != 0;
            NumericProgram program{false, 2, rounding, true, 2, negate, absolute};
            for (const u32 bits : inputs) {
                f64 value = std::bit_cast<f32>(bits);
                if (absolute) {
                    value = std::abs(value);
                }
                if (negate) {
                    value = -value;
                }
                const f64 rounded = rounding == 0   ? std::nearbyint(value)
                                    : rounding == 1 ? std::floor(value)
                                    : rounding == 2 ? std::ceil(value)
                                                    : std::trunc(value);
                const s32 expected =
                    std::isnan(value)
                        ? 0
                        : static_cast<s32>(std::clamp(rounded, -2147483648.0, 2147483647.0));
                CAPTURE(bits, rounding, modifiers);
                u64 result{};
                REQUIRE_NOTHROW(result = program.Run(bits));
                REQUIRE(u32(result) == std::bit_cast<u32>(expected));
                REQUIRE(u32(result >> 32) == 0xdeadbeefU);
            }
        }
    }
}

TEST_CASE("Maxwell F2I byte destinations saturate and extend", "[shader][maxwell][numeric]") {
    const std::array<u16, 19> half_inputs{
        0, 0x8000, 0xb800, 0x3800, 0x3e00, 0x4100, 0x57f0, 0x57f8, 0x5800,
        0xd7f0, 0xd800, 0xd808, 0x5bf4, 0x5bf8, 0x5bfc, 0x5c00, 0x7c00, 0xfc00, 0x7e01};
    for (u32 source_format : {1U, 2U, 3U}) {
        for (bool is_signed : {false, true}) {
            for (u32 rounding = 0; rounding < 4; ++rounding) {
                NumericProgram program{false, 0, rounding, is_signed, source_format};
                for (const u16 half : half_inputs) {
                    const f64 value = HalfValue(half);
                    const u64 input = source_format == 1 ? half
                                      : source_format == 2 ? std::bit_cast<u32>(f32(value))
                                                           : std::bit_cast<u64>(value);
                    const f64 rounded = rounding == 0 ? std::nearbyint(value)
                                        : rounding == 1 ? std::floor(value)
                                        : rounding == 2 ? std::ceil(value) : std::trunc(value);
                    const s32 expected = std::isnan(value) ? 0 : static_cast<s32>(
                        std::clamp(rounded, is_signed ? -128.0 : 0.0, is_signed ? 127.0 : 255.0));
                    INFO("source " << source_format << " signed " << is_signed << " round " << rounding);
                    const u64 result = program.Run(input);
                    REQUIRE(u32(result) == u32(expected));
                    REQUIRE(u32(result >> 32) == 0xdeadbeefU);
                }
            }
        }
    }
}

TEST_CASE("Maxwell saturating conversions emit with native and lowered float64",
          "[shader][maxwell][numeric_backend]") {
    for (u32 backend = 0; backend < 3; ++backend) {
        for (bool float64 : {false, true}) {
            for (const auto [destination, format] : std::array{
                     std::pair{0U, 1U}, std::pair{0U, 2U}, std::pair{0U, 3U}, std::pair{2U, 2U}}) {
                for (bool sign : {false, true}) {
                    if (destination == 2 && !sign) {
                        continue;
                    }
                    NumericProgram translation{false, destination, 3, sign, format};
                    IR::IREmitter start{translation.block, translation.block.begin()};
                    start.SetReg(IR::Reg::R2, start.LocalInvocationIdX());
                    start.SetReg(IR::Reg::R3, start.LocalInvocationIdY());
                    IR::IREmitter end{translation.block};
                    translation.block.AppendNewInst(Opcode::WriteStorage32,
                        {end.Imm32(0), end.Imm32(0), end.GetReg(IR::Reg::R0)});
                    IR::Program program;
                    program.stage = Shader::Stage::Compute;
                    program.workgroup_size = {1, 1, 1};
                    program.blocks = {&translation.block};
                    program.post_order_blocks = {&translation.block};
                    program.syntax_list = {
                        {{.block = &translation.block}, IR::AbstractSyntaxNode::Type::Block},
                        {{}, IR::AbstractSyntaxNode::Type::Return}};
                    Shader::Optimization::LowerFp16ToFp32(program);
                    if (!float64) {
                        Shader::Optimization::LowerFp64ToFp32(program);
                    }
                    Shader::Optimization::SsaRewritePass(program);
                    Shader::Optimization::DeadCodeEliminationPass(program);
                    Shader::Optimization::CollectShaderInfoPass(translation.env, program);
                    program.info.used_storage_buffer_types = IR::Type::U32;
                    program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
                    CAPTURE(backend, float64, destination, format, sign);
                    for (const auto& inst : translation.block.Instructions()) {
                        if (!float64) {
                            REQUIRE(inst.Type() != IR::Type::F64);
                        }
                        if (inst.GetOpcode() == Opcode::SelectF32 ||
                            inst.GetOpcode() == Opcode::SelectF64) {
                            REQUIRE(inst.Arg(1).Type() == inst.Type());
                            REQUIRE(inst.Arg(2).Type() == inst.Type());
                        }
                    }
                    Shader::Profile profile{};
                    profile.supported_spirv = 0x00010300;
                    profile.support_int64 = true;
                    if (backend == 0) {
                        REQUIRE_FALSE(Shader::Backend::SPIRV::EmitSPIRV(profile, program).empty());
                    } else if (backend == 1) {
                        REQUIRE_FALSE(Shader::Backend::GLSL::EmitGLSL(profile, program).empty());
                    } else {
                        Shader::RuntimeInfo runtime{};
                        runtime.glasm_use_storage_buffers = true;
                        REQUIRE_FALSE(Shader::Backend::GLASM::EmitGLASM(profile, runtime, program).empty());
                    }
                }
            }
        }
    }
}

TEST_CASE("Maxwell double byte conversions round before narrowing",
          "[shader][maxwell][numeric]") {
    const std::array<f64, 19> inputs{
        0.0, -0.0, std::numeric_limits<f64>::denorm_min(),
        -std::numeric_limits<f64>::denorm_min(),
        std::nextafter(127.5, 0.0), 127.5, std::nextafter(127.5, 128.0),
        std::nextafter(-128.5, -129.0), -128.5, std::nextafter(-128.5, -128.0),
        std::nextafter(255.5, 255.0), 255.5, std::nextafter(255.5, 256.0),
        std::numeric_limits<f64>::max(), -std::numeric_limits<f64>::max(),
        std::numeric_limits<f64>::infinity(), -std::numeric_limits<f64>::infinity(),
        std::numeric_limits<f64>::quiet_NaN(), -std::numeric_limits<f64>::quiet_NaN()};
    for (bool sign : {false, true}) {
        for (u32 rounding = 0; rounding < 4; ++rounding) {
            for (u32 modifiers = 0; modifiers < 4; ++modifiers) {
                NumericProgram program{false, 0, rounding, sign, 3,
                                       (modifiers & 1) != 0, (modifiers & 2) != 0};
                for (f64 input : inputs) {
                    f64 value = (modifiers & 2) != 0 ? std::abs(input) : input;
                    if ((modifiers & 1) != 0) {
                        value = -value;
                    }
                    const f64 rounded = rounding == 0 ? std::nearbyint(value)
                                        : rounding == 1 ? std::floor(value)
                                        : rounding == 2 ? std::ceil(value) : std::trunc(value);
                    const s32 expected = std::isnan(value) ? 0 : static_cast<s32>(
                        std::clamp(rounded, sign ? -128.0 : 0.0, sign ? 127.0 : 255.0));
                    CAPTURE(sign, rounding, modifiers, input);
                    REQUIRE(u32(program.Run(std::bit_cast<u64>(input))) == u32(expected));
                }
            }
        }
    }
}
