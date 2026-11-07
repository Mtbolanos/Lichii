// SPDX-License-Identifier: GPL-3.0-or-later

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

struct Program {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Shader::Profile profile{};

    Program() {
        program.stage = Shader::Stage::Compute;
        program.workgroup_size = {1, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
        profile.supported_spirv = 0x00010300;
        profile.support_float_controls = true;
        profile.support_fp32_denorm_flush = true;
        profile.support_fp32_denorm_preserve = true;
    }

    void Add(IR::Opcode opcode, IR::FmzMode mode, u32 offset = 0) {
        IR::IREmitter ir{block};
        // Bitcasts keep the operands live without introducing another floating-point mode.
        const auto a = ir.BitCast<IR::F32>(ir.LocalInvocationIdX());
        const auto b = ir.BitCast<IR::F32>(ir.LocalInvocationIdY());
        const IR::FpControl control{.fmz_mode = mode};
        IR::F32 result;
        if (opcode == IR::Opcode::FPMin32) {
            result = IR::F32{ir.FPMin(a, b, control)};
        } else if (opcode == IR::Opcode::FPMax32) {
            result = IR::F32{ir.FPMax(a, b, control)};
        } else {
            result = IR::F32{ir.FPAdd(a, b, control)};
        }
        block.AppendNewInst(IR::Opcode::WriteStorage32,
                            {ir.Imm32(0), ir.Imm32(offset), ir.BitCast<IR::U32>(result)});
    }

    void Collect() {
        TestEnvironment env;
        Shader::Optimization::CollectShaderInfoPass(env, program);
    }
};

bool HasMode(const std::vector<u32>& code, spv::ExecutionMode mode) {
    for (size_t offset = 5; offset < code.size();) {
        const u32 count = code[offset] >> 16;
        REQUIRE(count > 0);
        REQUIRE(offset + count <= code.size());
        if ((code[offset] & 0xffff) == static_cast<u32>(spv::Op::OpExecutionMode) && count == 4 &&
            code[offset + 2] == static_cast<u32>(mode) && code[offset + 3] == 32) {
            return true;
        }
        offset += count;
    }
    return false;
}

void CheckModes(Program& fixture, bool flush, bool preserve) {
    const auto code = Shader::Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program);
    CHECK(HasMode(code, spv::ExecutionMode::DenormFlushToZero) == flush);
    CHECK(HasMode(code, spv::ExecutionMode::DenormPreserve) == preserve);
}
} // namespace

TEST_CASE("FP32 min and max retain their denormal controls through SPIR-V emission",
          "[video_core][spirv][min_max_denorms]") {
    for (const auto opcode : {IR::Opcode::FPMin32, IR::Opcode::FPMax32}) {
        for (const auto mode : {IR::FmzMode::FTZ, IR::FmzMode::None, IR::FmzMode::DontCare}) {
            CAPTURE(opcode, mode);
            Program fixture;
            fixture.Add(opcode, mode);
            fixture.Collect();
            CHECK(fixture.program.info.uses_fp32_denorms_flush == (mode == IR::FmzMode::FTZ));
            CHECK(fixture.program.info.uses_fp32_denorms_preserve == (mode == IR::FmzMode::None));
            CheckModes(fixture, mode == IR::FmzMode::FTZ, mode == IR::FmzMode::None);
        }
    }
}

TEST_CASE("Min max controls obey existing mixed denormal policy",
          "[video_core][spirv][min_max_denorms]") {
    for (const auto other : {IR::Opcode::FPMax32, IR::Opcode::FPAdd32}) {
        Program fixture;
        fixture.Add(IR::Opcode::FPMin32, IR::FmzMode::FTZ);
        fixture.Add(other, IR::FmzMode::None, 4);
        fixture.Collect();
        CHECK(fixture.program.info.uses_fp32_denorms_flush);
        CHECK(fixture.program.info.uses_fp32_denorms_preserve);
        // The existing emitter declines to select a shader-wide mode when both are requested.
        CheckModes(fixture, false, false);
    }
}

TEST_CASE("Unsupported or broken float-control profiles suppress min max execution modes",
          "[video_core][spirv][min_max_denorms]") {
    for (const auto mode : {IR::FmzMode::FTZ, IR::FmzMode::None}) {
        for (const bool all_controls_disabled : {false, true}) {
            Program fixture;
            fixture.Add(IR::Opcode::FPMin32, mode);
            fixture.Collect();
            CHECK(fixture.program.info.uses_fp32_denorms_flush == (mode == IR::FmzMode::FTZ));
            CHECK(fixture.program.info.uses_fp32_denorms_preserve == (mode == IR::FmzMode::None));
            fixture.profile.support_float_controls = !all_controls_disabled;
            if (!all_controls_disabled) {
                fixture.profile.support_fp32_denorm_flush = false;
                fixture.profile.support_fp32_denorm_preserve = false;
            }
            CheckModes(fixture, false, false);
        }
    }
    Program fixture;
    fixture.Add(IR::Opcode::FPMax32, IR::FmzMode::FTZ);
    fixture.Collect();
    CHECK(fixture.program.info.uses_fp32_denorms_flush);
    fixture.profile.has_broken_fp32_denorm_flush = true;
    CheckModes(fixture, false, false);
}
