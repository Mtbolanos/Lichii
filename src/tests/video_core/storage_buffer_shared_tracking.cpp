// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
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
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

struct Fixture {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block block{pool};
    IR::IREmitter ir{block};
    IR::Program program;

    Fixture() {
        program.stage = Shader::Stage::Compute;
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
    }

    IR::U32 Word(u32 offset) {
        return ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset));
    }

    /// A low address the constant buffer word only reaches through a mask, so neither the
    /// paired-pointer nor the addend search finds it and the breadth-first searches run.
    IR::U32 Masked(u32 offset) {
        return ir.BitwiseAnd(Word(offset), ir.Imm32(~0U));
    }

    void Store(const IR::U32& low) {
        ir.WriteGlobal32(ir.PackUint2x32(ir.CompositeConstruct(low, ir.Imm32(0))), ir.Imm32(7));
    }

    /// Runs the pass; returns the bound constant buffer offsets and whether accesses stayed global.
    std::pair<std::vector<u32>, bool> Run() {
        TestEnvironment env;
        Shader::HostTranslateInfo host{};
        host.min_ssbo_alignment = 16;
        Shader::Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
        std::vector<u32> offsets;
        for (const auto& descriptor : program.info.storage_buffers_descriptors) {
            offsets.push_back(descriptor.cbuf_offset);
        }
        std::ranges::sort(offsets);
        const bool global{std::ranges::any_of(block.Instructions(), [](const IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::WriteGlobal32;
        })};
        return {offsets, global};
    }
};
} // namespace

TEST_CASE("Addresses built on a pointer an earlier search failed on still track their buffers",
          "[shader][storage_buffer_shared_tracking]") {
    Fixture fixture;
    auto& ir{fixture.ir};
    // c0[0x10] lies outside the NVN range, so the biased search fails and the unbiased one binds it.
    const IR::U32 shared{fixture.Masked(0x10)};
    fixture.Store(shared);
    // Biased, this search reaches only what the failed one walked; unbiased, it finds c0[0x10].
    fixture.Store(ir.BitwiseOr(shared, ir.LocalInvocationIdX()));
    // The biased search finds c0[0x120] next to the walked pointer.
    fixture.Store(ir.BitwiseOr(shared, fixture.Masked(0x120)));
    CHECK(fixture.Run() == std::pair{std::vector<u32>{0x10, 0x120}, false});
}
