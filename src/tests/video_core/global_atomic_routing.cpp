// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/profile.h"

namespace {
using namespace Shader;

class StubEnvironment final : public Environment {
public:
    StubEnvironment() {
        stage = Stage::Compute;
    }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    TextureType ReadTextureType(u32) override { return {}; }
    TexturePixelFormat ReadTexturePixelFormat(u32) override { return {}; }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 0; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

// SPIR-V instructions with opcode `opcode`, as their operand words.
std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t i = 5; i < code.size();) {
        const u32 count = code[i] >> 16;
        REQUIRE(count > 0);
        REQUIRE(i + count <= code.size());
        if ((code[i] & 0xffff) == opcode) {
            result.emplace_back(code.begin() + i + 1, code.begin() + i + count);
        }
        i += count;
    }
    return result;
}

// A compute program whose atomic address comes from global memory, so the storage buffer pass
// tracks the load but not the atomic.
std::vector<u32> EmitUntrackedAtomic(IR::Opcode opcode, bool select) {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    const auto low = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x110));
    const auto high = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x114));
    const auto loaded = ir.LoadGlobal64(ir.PackUint2x32(ir.CompositeConstruct(low, high)));
    const IR::U64 address{ir.PackUint2x32(
        ir.CompositeConstruct(ir.CompositeExtract(loaded, 0), ir.CompositeExtract(loaded, 1)))};
    const IR::Value value{opcode == IR::Opcode::GlobalAtomicAddF32 ? IR::Value{ir.Imm32(1.0f)}
                                                                   : IR::Value{ir.Imm32(1)}};
    const IR::Value previous{&*block.PrependNewInst(block.end(), opcode, {address, value})};
    const IR::U32 result{opcode == IR::Opcode::GlobalAtomicAddF32
                             ? IR::U32{ir.BitCast<IR::U32>(IR::F32{previous})}
                             : IR::U32{previous}};
    ir.WriteGlobal32(address, result);

    IR::Program program;
    program.stage = Stage::Compute;
    program.workgroup_size = {1, 1, 1};
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    StubEnvironment env;
    Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
    REQUIRE(std::ranges::any_of(block.Instructions(), [&](const IR::Inst& inst) {
        return inst.GetOpcode() == opcode;
    }));
    Optimization::CollectShaderInfoPass(env, program);
    program.info.native_global_bounds = true;
    program.info.constant_buffer_descriptors = {{0, 1}};

    Profile profile{};
    profile.supported_spirv = 0x00010300;
    profile.support_int64 = true;
    profile.support_descriptor_aliasing = true;
    profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = 16;
    profile.select_global_memory_buffer = select;
    return Backend::SPIRV::EmitSPIRV(profile, program);
}

// A compute program with two tracked pointers in constant buffer 0 and, through the pointer loaded
// from the first one, untracked accesses of every width, an atomic and a compare-exchange.
std::vector<u32> EmitUntrackedAccesses(bool native, u32 min_ssbo_alignment, bool select) {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    const auto pointer = [&](u32 offset) {
        return IR::U64{
            ir.PackUint2x32(ir.CompositeConstruct(ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset)),
                                                  ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset + 4))))};
    };
    const auto loaded = ir.LoadGlobal64(pointer(0x110));
    const IR::U32 value{ir.LoadGlobal32(pointer(0x120))};
    const IR::U64 address{ir.PackUint2x32(
        ir.CompositeConstruct(ir.CompositeExtract(loaded, 0), ir.CompositeExtract(loaded, 1)))};
    ir.WriteGlobal32(address, ir.LoadGlobal32(address));
    ir.WriteGlobal64(address, ir.LoadGlobal64(address));
    ir.WriteGlobal128(address, ir.LoadGlobal128(address));
    const IR::U32 sum{
        &*block.PrependNewInst(block.end(), IR::Opcode::GlobalAtomicIAdd32, {address, value})};
    const IR::U32 swapped{&*block.PrependNewInst(
        block.end(), IR::Opcode::GlobalAtomicCompareExchange32, {address, value, sum})};
    ir.WriteGlobal32(address, swapped);

    IR::Program program;
    program.stage = Stage::Compute;
    program.workgroup_size = {1, 1, 1};
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    HostTranslateInfo host{};
    host.min_ssbo_alignment = min_ssbo_alignment;
    StubEnvironment env;
    Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
    REQUIRE(std::ranges::any_of(block.Instructions(), [](const IR::Inst& inst) {
        return inst.GetOpcode() == IR::Opcode::LoadGlobal128;
    }));
    Optimization::CollectShaderInfoPass(env, program);
    program.info.native_global_bounds = native;
    program.info.nvn_buffer_base = 0x110;
    program.info.nvn_buffer_used.set(0);
    program.info.nvn_buffer_used.set(1);
    program.info.constant_buffer_descriptors = {{0, 1}};
    REQUIRE(program.info.storage_buffers_descriptors.size() >= 2);

    Profile profile{};
    profile.supported_spirv = 0x00010300;
    profile.support_int64 = true;
    profile.support_descriptor_aliasing = true;
    profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = min_ssbo_alignment;
    profile.select_global_memory_buffer = select;
    return Backend::SPIRV::EmitSPIRV(profile, program);
}
} // namespace

TEST_CASE("Global memory helpers branch per buffer and return only from their last block",
          "[shader_recompiler][global_memory]") {
    constexpr u32 OpEntryPoint = 15;
    constexpr u32 OpFunction = 54;
    constexpr u32 OpFunctionEnd = 56;
    constexpr u32 OpSwitch = 251;
    constexpr u32 OpReturn = 253;
    constexpr u32 OpReturnValue = 254;
    for (const bool native : {true, false}) {
        for (const u32 alignment : {16U, 4U}) {
            CAPTURE(native, alignment);
            std::vector<u32> code;
            REQUIRE_NOTHROW(code = EmitUntrackedAccesses(native, alignment, false));
            const auto entry_points = Instructions(code, OpEntryPoint);
            REQUIRE(entry_points.size() == 1);
            const u32 main_id = entry_points[0][1];

            size_t helpers{};
            u32 function_id{};
            u32 returns{};
            u32 previous{};
            for (size_t i = 5; i < code.size();) {
                const u32 count = code[i] >> 16;
                const u32 opcode = code[i] & 0xffff;
                REQUIRE(count > 0);
                if (opcode == OpFunction) {
                    function_id = code[i + 2];
                    returns = 0;
                } else if (opcode == OpReturn || opcode == OpReturnValue) {
                    ++returns;
                } else if (opcode == OpFunctionEnd && function_id != main_id) {
                    ++helpers;
                    CAPTURE(function_id);
                    CHECK(returns == 1);
                    CHECK((previous == OpReturn || previous == OpReturnValue));
                }
                previous = opcode;
                i += count;
            }
            CHECK(helpers >= 7);
            CHECK(Instructions(code, OpSwitch).empty());
        }
    }
}

TEST_CASE("Global memory helpers with variable pointers select the buffer and branch only for writes",
          "[shader_recompiler][global_memory]") {
    constexpr u32 OpCapability = 17;
    constexpr u32 OpTypePointer = 32;
    constexpr u32 OpEntryPoint = 15;
    constexpr u32 OpFunction = 54;
    constexpr u32 OpFunctionEnd = 56;
    constexpr u32 OpFunctionCall = 57;
    constexpr u32 OpStore = 62;
    constexpr u32 OpDecorate = 71;
    constexpr u32 OpSelect = 169;
    constexpr u32 OpAtomicExchange = 229;
    constexpr u32 OpAtomicXor = 242;
    constexpr u32 OpSelectionMerge = 247;
    constexpr u32 OpBranchConditional = 250;
    constexpr u32 OpSwitch = 251;
    constexpr u32 OpReturn = 253;
    constexpr u32 OpReturnValue = 254;
    constexpr u32 VariablePointersStorageBuffer = 4441;
    constexpr u32 ShaderNonUniform = 5301;
    constexpr u32 NonUniform = 5300;
    for (const bool native : {true, false}) {
        for (const u32 alignment : {16U, 4U}) {
            CAPTURE(native, alignment);
            std::vector<u32> code;
            REQUIRE_NOTHROW(code = EmitUntrackedAccesses(native, alignment, true));
            const auto has_capability = [&](u32 capability) {
                return std::ranges::any_of(Instructions(code, OpCapability),
                                           [&](const auto& ops) { return ops[0] == capability; });
            };
            CHECK(has_capability(VariablePointersStorageBuffer));
            CHECK(has_capability(ShaderNonUniform));
            CHECK(Instructions(code, OpSwitch).empty());

            std::vector<u32> pointer_types;
            for (const auto& ops : Instructions(code, OpTypePointer)) {
                pointer_types.push_back(ops[0]);
            }
            std::vector<u32> non_uniform;
            for (const auto& ops : Instructions(code, OpDecorate)) {
                if (ops[1] == NonUniform) {
                    non_uniform.push_back(ops[0]);
                }
            }
            const auto is_non_uniform = [&](u32 id) {
                return std::ranges::find(non_uniform, id) != non_uniform.end();
            };
            size_t selected_pointers{};
            for (const auto& ops : Instructions(code, OpSelect)) {
                if (std::ranges::find(pointer_types, ops[0]) != pointer_types.end()) {
                    ++selected_pointers;
                }
            }
            CHECK(selected_pointers > 0);

            const u32 main_id = Instructions(code, OpEntryPoint)[0][1];
            size_t loads{};
            size_t writes{};
            u32 function_id{};
            u32 branches{};
            u32 merges{};
            u32 returns{};
            u32 last_select{};
            bool writes_before_branch{};
            bool writes_memory{};
            for (size_t i = 5; i < code.size();) {
                const u32 count = code[i] >> 16;
                const u32 opcode = code[i] & 0xffff;
                REQUIRE(count > 0);
                if (opcode == OpFunction) {
                    function_id = code[i + 2];
                    branches = merges = returns = 0;
                    last_select = 0;
                    writes_before_branch = writes_memory = false;
                } else if (opcode == OpSelect &&
                           std::ranges::find(pointer_types, code[i + 1]) != pointer_types.end()) {
                    last_select = code[i + 2];
                } else if (opcode == OpBranchConditional) {
                    ++branches;
                } else if (opcode == OpSelectionMerge) {
                    ++merges;
                } else if (opcode == OpReturn || opcode == OpReturnValue) {
                    ++returns;
                } else if (opcode == OpStore || opcode == OpFunctionCall ||
                           (opcode >= OpAtomicExchange && opcode <= OpAtomicXor)) {
                    writes_memory = true;
                    writes_before_branch |= branches == 0;
                } else if (opcode == OpFunctionEnd && function_id != main_id) {
                    CAPTURE(function_id);
                    CHECK(returns == 1);
                    CHECK(merges == branches);
                    CHECK(last_select != 0);
                    CHECK(is_non_uniform(last_select));
                    if (writes_memory) {
                        ++writes;
                        CHECK(branches == 1);
                        CHECK_FALSE(writes_before_branch);
                    } else {
                        ++loads;
                        CHECK(branches == 0);
                    }
                }
                i += count;
            }
            CHECK(loads == 3);
            CHECK(writes >= 5);
        }
    }
}

TEST_CASE("Untracked 32-bit global atomics route to their storage buffer",
          "[shader_recompiler][global_memory]") {
    constexpr u32 OpFunctionCall = 57;
    constexpr u32 OpAtomicCompareExchange = 230;
    // OpAtomicExchange 229, IAdd 234, SMin 236, UMin 237, SMax 238, UMax 239, And 240, Or 241,
    // Xor 242. INC, DEC and F32 add go through the storage buffer CAS loops.
    const std::array<std::pair<IR::Opcode, u32>, 12> cases{{
        {IR::Opcode::GlobalAtomicIAdd32, 234},
        {IR::Opcode::GlobalAtomicSMin32, 236},
        {IR::Opcode::GlobalAtomicUMin32, 237},
        {IR::Opcode::GlobalAtomicSMax32, 238},
        {IR::Opcode::GlobalAtomicUMax32, 239},
        {IR::Opcode::GlobalAtomicAnd32, 240},
        {IR::Opcode::GlobalAtomicOr32, 241},
        {IR::Opcode::GlobalAtomicXor32, 242},
        {IR::Opcode::GlobalAtomicExchange32, 229},
        {IR::Opcode::GlobalAtomicInc32, OpAtomicCompareExchange},
        {IR::Opcode::GlobalAtomicDec32, OpAtomicCompareExchange},
        {IR::Opcode::GlobalAtomicAddF32, OpAtomicCompareExchange},
    }};
    for (const bool select : {false, true}) {
        for (const auto& [opcode, atomic] : cases) {
            CAPTURE(select, static_cast<u32>(opcode));
            std::vector<u32> code;
            REQUIRE_NOTHROW(code = EmitUntrackedAtomic(opcode, select));
            CHECK_FALSE(Instructions(code, atomic).empty());
            CHECK_FALSE(Instructions(code, OpFunctionCall).empty());
        }
    }
}
