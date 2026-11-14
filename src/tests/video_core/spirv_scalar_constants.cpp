// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <map>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spirv/unified1/spirv.hpp11>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
namespace IR = Shader::IR;

class TestEnvironment final : public Shader::Environment {
public:
    TestEnvironment() { stage = Shader::Stage::Compute; }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
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

struct Module {
    /// Ids of 32-bit scalar OpConstants by (type id, bit pattern), with how often each is declared.
    std::map<std::pair<u32, u32>, std::vector<u32>> constants;
    std::map<u32, u32> bitcast_operands;
    u32 float_type{};
    u32 uint_type{};

    explicit Module(const std::vector<u32>& code) {
        for (size_t offset = 5; offset < code.size();) {
            const u32 count{code[offset] >> 16};
            REQUIRE(count > 0);
            REQUIRE(offset + count <= code.size());
            const auto op{static_cast<spv::Op>(code[offset] & 0xffff)};
            const u32* const words{&code[offset]};
            if (op == spv::Op::OpTypeFloat && words[2] == 32) {
                float_type = words[1];
            } else if (op == spv::Op::OpTypeInt && words[2] == 32 && words[3] == 0) {
                uint_type = words[1];
            } else if (op == spv::Op::OpConstant && count == 4) {
                constants[{words[1], words[3]}].push_back(words[2]);
            } else if (op == spv::Op::OpBitcast) {
                bitcast_operands[words[2]] = words[3];
            }
            offset += count;
        }
    }

    std::vector<u32> Ids(u32 type, u32 bits) const {
        const auto it{constants.find({type, bits})};
        return it != constants.end() ? it->second : std::vector<u32>{};
    }
};
} // namespace

TEST_CASE("Scalar constants keep one declaration per type and bit pattern",
          "[video_core][spirv][scalar_constants]") {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = Shader::Stage::Compute;
    program.workgroup_size = {1, 1, 1};
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    program.info.storage_buffers_descriptors = {{0, 0, 1, true}};

    // Zero and negative zero, two NaN payloads, and a float whose bits match an integer stored too.
    const std::vector<u32> float_bits{0x00000000, 0x80000000, 0x7fc00000, 0x7fc00001,
                                      0x3f800000, 0x80000000, 0x7fc00001, 0x00000000};
    const std::vector<u32> uint_values{5, 0x80000000, 5};
    IR::IREmitter ir{block};
    u32 offset{};
    for (const u32 bits : float_bits) {
        const IR::F32 value{ir.Imm32(std::bit_cast<f32>(bits))};
        block.AppendNewInst(IR::Opcode::WriteStorage32,
                            {ir.Imm32(0), ir.Imm32(offset += 4), ir.BitCast<IR::U32>(value)});
    }
    for (const u32 value : uint_values) {
        block.AppendNewInst(IR::Opcode::WriteStorage32,
                            {ir.Imm32(0), ir.Imm32(offset += 4), ir.Imm32(value)});
    }
    TestEnvironment env;
    Shader::Optimization::CollectShaderInfoPass(env, program);
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010300;

    const Module module{Shader::Backend::SPIRV::EmitSPIRV(profile, program)};
    REQUIRE(module.float_type != 0);
    REQUIRE(module.uint_type != 0);
    for (const auto& [key, ids] : module.constants) {
        CAPTURE(key.first, key.second);
        CHECK(ids.size() == 1);
    }
    for (const u32 value : {5U, 0x80000000U}) {
        CAPTURE(value);
        CHECK(module.Ids(module.uint_type, value).size() == 1);
    }
    CHECK(module.Ids(module.float_type, 0x00000000) != module.Ids(module.float_type, 0x80000000));
    CHECK(module.Ids(module.float_type, 0x80000000) != module.Ids(module.uint_type, 0x80000000));

    // Every float stored is bitcast from the constant with its exact bits.
    std::map<u32, u32> float_bits_by_id;
    for (const auto& [key, ids] : module.constants) {
        if (key.first == module.float_type) {
            float_bits_by_id[ids.front()] = key.second;
        }
    }
    std::vector<u32> bitcast_bits;
    for (const auto& [result, operand] : module.bitcast_operands) {
        if (const auto it{float_bits_by_id.find(operand)}; it != float_bits_by_id.end()) {
            bitcast_bits.push_back(it->second);
        }
    }
    std::ranges::sort(bitcast_bits);
    std::vector<u32> expected{float_bits};
    std::ranges::sort(expected);
    CHECK(bitcast_bits == expected);
}
