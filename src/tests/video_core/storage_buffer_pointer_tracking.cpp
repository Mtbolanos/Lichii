// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "shader_recompiler/backend/nvn_storage_descriptor.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
namespace IR = Shader::IR;

class TestEnvironment final : public Shader::Environment {
public:
    bool native{};
    std::vector<u32> scalar_offsets;
    TestEnvironment() { stage = Shader::Stage::Compute; }
    bool IsNativeProgram() const override { return native; }
    bool IsCbufStorageBufferPointer(u32 index, u32 offset) override {
        return index == 0 && std::ranges::find(scalar_offsets, offset) == scalar_offsets.end();
    }
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

struct KernelProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block entry{pool};
    IR::Block loop{pool};
    IR::Program program;
    TestEnvironment env;
    bool int64{true};
    u32 storage_buffer_limit{};     ///< The stage's storage buffer limit, none when 0
    u32 set_storage_buffer_limit{}; ///< The descriptor set's storage buffer limit, none when 0

    KernelProgram() {
        program.stage = Shader::Stage::Compute;
        program.blocks = {&entry};
        program.post_order_blocks = {&entry};
    }

    std::vector<u32> TrackedOffsets() {
        Shader::HostTranslateInfo host{};
        host.min_ssbo_alignment = 16;
        host.support_int64 = int64;
        host.max_per_stage_descriptor_storage_buffers = storage_buffer_limit;
        host.max_descriptor_set_storage_buffers = set_storage_buffer_limit;
        Shader::Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
        std::vector<u32> offsets;
        for (const auto& descriptor : program.info.storage_buffers_descriptors) {
            offsets.push_back(descriptor.cbuf_offset);
        }
        return offsets;
    }
};

IR::U32 Word(IR::IREmitter& ir, u32 offset) {
    return ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset));
}

IR::U64 Address(IR::IREmitter& ir, const IR::U32& low, const IR::U32& high) {
    return ir.PackUint2x32(ir.CompositeConstruct(low, high));
}
} // Anonymous namespace

TEST_CASE("Kernel parameter pointers win over 16-byte aligned strides",
          "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.LocalInvocationIdX()};
    const IR::U32 row{ir.IAdd(ir.ShiftLeftLogical(lane, ir.Imm32(3)), Word(ir, 0x1c0))};
    const IR::U32 low{ir.IAdd(ir.IAdd(row, Word(ir, 0x178)), ir.ShiftLeftLogical(lane, ir.Imm32(4)))};
    ir.WriteGlobal32(Address(ir, low, Word(ir, 0x17c)), lane);
    CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x178});
}

TEST_CASE("Filter tap offset tables are not taken for the weight pointer",
          "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.LocalInvocationIdX()};
    const IR::Value weights{ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x168), 64, false)};
    const IR::U32 taps_low{ir.IAdd(ir.IAdd(Word(ir, 0x210), Word(ir, 0x218)), Word(ir, 0x220))};
    const IR::U32 taps_high{ir.IAdd(ir.IAdd(Word(ir, 0x214), Word(ir, 0x21c)), Word(ir, 0x224))};
    const IR::U32 low{ir.IAdd(ir.IAdd(IR::U32{ir.CompositeExtract(weights, 0)},
                                     ir.ShiftLeftLogical(lane, ir.Imm32(1))),
                              taps_low)};
    const IR::U32 high{ir.IAdd(IR::U32{ir.CompositeExtract(weights, 1)}, taps_high)};
    static_cast<void>(ir.LoadGlobal128(Address(ir, low, high)));
    CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x168});
}

TEST_CASE("Loop increments are not taken for the iterated pointer", "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    fixture.entry.AddBranch(&fixture.loop);
    fixture.loop.AddBranch(&fixture.loop);
    fixture.program.blocks = {&fixture.entry, &fixture.loop};
    fixture.program.post_order_blocks = {&fixture.loop, &fixture.entry};
    IR::IREmitter entry{fixture.entry};
    const IR::Value activations{entry.GetCbuf(entry.Imm32(0), entry.Imm32(0x160), 64, false)};
    const IR::Value step{entry.GetCbuf(entry.Imm32(0), entry.Imm32(0x200), 64, false)};
    const IR::U32 first{entry.IAdd(IR::U32{entry.CompositeExtract(activations, 0)},
                                   IR::U32{entry.CompositeExtract(step, 0)})};
    const IR::U32 first_high{entry.IAdd(IR::U32{entry.CompositeExtract(activations, 1)},
                                        IR::U32{entry.CompositeExtract(step, 1)})};
    IR::IREmitter body{fixture.loop};
    fixture.loop.AppendNewInst(IR::Opcode::Phi, {});
    IR::Inst* const low{&fixture.loop.back()};
    low->SetFlags(IR::Type::U32);
    fixture.loop.AppendNewInst(IR::Opcode::Phi, {});
    IR::Inst* const high{&fixture.loop.back()};
    high->SetFlags(IR::Type::U32);
    static_cast<void>(
        body.LoadGlobal128(Address(body, IR::U32{IR::Value{low}}, IR::U32{IR::Value{high}})));
    const IR::U32 next{body.IAdd(IR::U32{IR::Value{low}}, Word(body, 0x200))};
    const IR::U32 next_high{body.IAdd(IR::U32{IR::Value{high}}, Word(body, 0x204))};
    low->AddPhiOperand(&fixture.entry, first);
    low->AddPhiOperand(&fixture.loop, next);
    high->AddPhiOperand(&fixture.entry, first_high);
    high->AddPhiOperand(&fixture.loop, next_high);
    CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x160});
}

TEST_CASE("NVN storage buffer records keep their tracking", "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.LocalInvocationIdX()};
    const IR::U32 low{ir.IAdd(Word(ir, 0x310), ir.ShiftLeftLogical(lane, ir.Imm32(2)))};
    static_cast<void>(ir.LoadGlobal32(Address(ir, low, Word(ir, 0x314))));
    CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x310});
}

TEST_CASE("Native pointer classification rejects a scalar selected by word tracking",
          "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = true;
    fixture.env.scalar_offsets = {0x310};
    IR::IREmitter ir{fixture.entry};
    const auto low = ir.IAdd(Word(ir, 0x310), Word(ir, 0x168));
    ir.WriteGlobal32(Address(ir, low, Word(ir, 0x314)), ir.Imm32(1));
    CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x168});
    CHECK_FALSE(fixture.program.info.native_global_bounds);
    CHECK(std::ranges::count_if(fixture.entry.Instructions(), [](const auto& inst) {
        return inst.GetOpcode() == IR::Opcode::WriteStorage32;
    }) == 1);
    CHECK(fixture.program.info.storage_buffers_descriptors.front().is_written);
}

TEST_CASE("Native accesses without mapped pointers are not rebased on scalars",
          "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = true;
    fixture.env.scalar_offsets = {0x310};
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, Word(ir, 0x310), Word(ir, 0x314)), ir.Imm32(1));
    CHECK(fixture.TrackedOffsets().empty());
    CHECK(fixture.entry.back().GetOpcode() == IR::Opcode::WriteGlobal32);
}

TEST_CASE("Native append counters behind unmapped pointers become storage atomics",
          "[shader][storage_buffer][global_atomic]") {
    // Donkey Kong Bananza's compute programs add a ballot count to the counter whose pointer is
    // at c0[0x1d0], or at c0[0x1e0]. The counter was unmapped when the pipeline was first built.
    for (const u32 counter : {0x1d0U, 0x1e0U}) {
        CAPTURE(counter);
        KernelProgram fixture;
        fixture.env.native = true;
        fixture.env.scalar_offsets = {counter};
        IR::IREmitter ir{fixture.entry};
        const IR::Value pointer{ir.GetCbuf(ir.Imm32(0), ir.Imm32(counter), 64, false)};
        const IR::U32 low{ir.CompositeExtract(pointer, 0)};
        const IR::U32 high{ir.CompositeExtract(pointer, 1)};
        const IR::Value old{ir.GlobalAtomicIAdd(Address(ir, low, high), ir.Imm32(32))};
        ir.WriteGlobal32(Address(ir, Word(ir, 0x170), Word(ir, 0x174)), IR::U32{old});
        const IR::U32 record{ir.IAdd(low, ir.IAdd(IR::U32{old}, ir.Imm32(16)))};
        ir.WriteGlobal32(Address(ir, record, high), ir.Imm32(1));

        CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x170, counter});
        CHECK_FALSE(fixture.program.info.native_global_bounds);
        for (const auto& descriptor : fixture.program.info.storage_buffers_descriptors) {
            CHECK(descriptor.is_written);
        }
        const auto count = [&](IR::Opcode opcode) {
            return std::ranges::count_if(fixture.entry.Instructions(), [=](const IR::Inst& inst) {
                return inst.GetOpcode() == opcode;
            });
        };
        CHECK(count(IR::Opcode::StorageAtomicIAdd32) == 1);
        CHECK(count(IR::Opcode::GlobalAtomicIAdd32) == 0);
        CHECK(count(IR::Opcode::WriteStorage32) == 2);
        CHECK(count(IR::Opcode::WriteGlobal32) == 0);
    }
}

TEST_CASE("Native atomics offset from an unmapped pointer stay global",
          "[shader][storage_buffer][global_atomic]") {
    KernelProgram fixture;
    fixture.env.native = true;
    fixture.env.scalar_offsets = {0x1d0, 0x40};
    IR::IREmitter ir{fixture.entry};
    const IR::U32 low{ir.IAdd(Word(ir, 0x1d0), Word(ir, 0x40))};
    static_cast<void>(ir.GlobalAtomicIAdd(Address(ir, low, Word(ir, 0x1d4)), ir.Imm32(1)));
    CHECK(fixture.TrackedOffsets().empty());
    CHECK(fixture.entry.back().GetOpcode() == IR::Opcode::GlobalAtomicIAdd32);
}

TEST_CASE("Native atomics carried into an unmapped pointer's high word become storage atomics",
          "[shader][storage_buffer][global_atomic]") {
    // Kirby Air Riders compute 0x1175573eed7481a4 adds 64-bit offsets to the pointers at c0[0x40]
    // and c1[0x0] in 32-bit halves, carrying into the high word. Both were unmapped when the
    // pipeline was built, which left its atomics without a buffer.
    const bool dynamic_offset = GENERATE(false, true);
    CAPTURE(dynamic_offset);
    KernelProgram fixture;
    fixture.env.native = true;
    fixture.env.scalar_offsets = {0x40};
    IR::IREmitter ir{fixture.entry};
    const IR::Value pointer{ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x40), 64, false)};
    const IR::U32 pointer_low{ir.CompositeExtract(pointer, 0)};
    const IR::U32 pointer_high{ir.CompositeExtract(pointer, 1)};
    const IR::U32 lane{ir.LocalInvocationIdX()};
    IR::U32 low;
    IR::U1 carry;
    IR::U32 high{pointer_high};
    if (dynamic_offset) {
        low = ir.IAdd(ir.ShiftLeftLogical(lane, ir.Imm32(4)), pointer_low);
        carry = ir.GetCarryFromOp(low);
        high = ir.IAdd(high, ir.ShiftRightLogical(lane, ir.Imm32(28)));
    } else {
        low = ir.IAdd(pointer_low, ir.Imm32(0x1c));
        carry = ir.ILessThan(low, pointer_low, false);
    }
    high = ir.IAdd(high, IR::U32{ir.Select(carry, ir.Imm32(1), ir.Imm32(0))});
    const IR::U64 address{ir.IAdd(Address(ir, low, high), ir.Imm64(u64{8}))};
    const IR::Value old{ir.GlobalAtomicIAdd(address, ir.Imm32(1))};
    ir.WriteGlobal32(Address(ir, ir.IAdd(low, IR::U32{old}), high), ir.Imm32(1));

    CHECK(fixture.TrackedOffsets() == std::vector<u32>{0x40});
    CHECK_FALSE(fixture.program.info.native_global_bounds);
    CHECK(fixture.program.info.storage_buffers_descriptors.front().is_written);
    const auto count = [&](IR::Opcode opcode) {
        return std::ranges::count_if(fixture.entry.Instructions(), [=](const IR::Inst& inst) {
            return inst.GetOpcode() == opcode;
        });
    };
    CHECK(count(IR::Opcode::StorageAtomicIAdd32) == 1);
    CHECK(count(IR::Opcode::GlobalAtomicIAdd32) == 0);
    CHECK(count(IR::Opcode::WriteStorage32) == 1);
    CHECK(count(IR::Opcode::WriteGlobal32) == 0);
}

TEST_CASE("Native atomics with a carry that skips the pointer stay global",
          "[shader][storage_buffer][global_atomic]") {
    KernelProgram fixture;
    fixture.env.native = true;
    fixture.env.scalar_offsets = {0x1d0, 0x40};
    IR::IREmitter ir{fixture.entry};
    const IR::U32 low{ir.IAdd(Word(ir, 0x1d0), Word(ir, 0x40))};
    const IR::U1 carry{ir.ILessThan(Word(ir, 0x40), ir.Imm32(16), false)};
    const IR::U32 carried{ir.Select(carry, ir.Imm32(1), ir.Imm32(0))};
    const IR::U32 high{ir.IAdd(Word(ir, 0x1d4), carried)};
    static_cast<void>(ir.GlobalAtomicIAdd(Address(ir, low, high), ir.Imm32(1)));
    CHECK(fixture.TrackedOffsets().empty());
    CHECK(fixture.entry.back().GetOpcode() == IR::Opcode::GlobalAtomicIAdd32);
}

TEST_CASE("Loaded global pointers do not inherit the source buffer descriptor",
          "[shader][storage_buffer]") {
    const bool native = GENERATE(false, true);
    for (const bool add_offset : {false, true}) {
        KernelProgram fixture;
        fixture.env.native = native;
        IR::IREmitter ir{fixture.entry};
        const auto source = Address(ir, Word(ir, 0x110), Word(ir, 0x114));
        const auto loaded = ir.LoadGlobal64(source);
        const IR::U32 low{ir.CompositeExtract(loaded, 0)};
        const IR::U32 high{ir.CompositeExtract(loaded, 1)};
        ir.WriteGlobal32(Address(ir, add_offset ? IR::U32{ir.IAdd(low, ir.Imm32(16))} : low, high),
                         ir.Imm32(1));

        const std::vector<u32> offsets{fixture.TrackedOffsets()};
        const auto has = [&](IR::Opcode opcode) {
            return std::ranges::any_of(fixture.entry.Instructions(), [=](const IR::Inst& inst) {
                return inst.GetOpcode() == opcode;
            });
        };
        CHECK(has(IR::Opcode::LoadStorage64));
        CHECK(has(IR::Opcode::WriteGlobal32));
        CHECK_FALSE(has(IR::Opcode::WriteStorage32));
        const auto& descriptors{fixture.program.info.storage_buffers_descriptors};
        if (!native) {
            CHECK(offsets == std::vector<u32>{0x110});
            REQUIRE(descriptors.size() == 1);
            CHECK_FALSE(descriptors.front().is_written);
            continue;
        }
        // A native program also binds the buffer the loaded pointer points to, as its own
        // descriptor read through the source: the write stays global and the source unwritten.
        CHECK(offsets == std::vector<u32>{0x110, 0x110});
        REQUIRE(descriptors.size() == 2);
        const auto source_descriptor{std::ranges::find_if(
            descriptors, [](const auto& desc) { return desc.indirect_offset == ~0U; })};
        REQUIRE(source_descriptor != descriptors.end());
        CHECK(source_descriptor->cbuf_index == 0);
        CHECK(source_descriptor->cbuf_offset == 0x110);
        CHECK_FALSE(source_descriptor->is_written);
        CHECK(std::ranges::count_if(descriptors,
                                    [](const auto& desc) { return desc.is_written; }) == 1);
        const auto loaded_descriptor{std::ranges::find_if(
            descriptors, [](const auto& desc) { return desc.is_written; })};
        REQUIRE(loaded_descriptor != descriptors.end());
        CHECK(loaded_descriptor->cbuf_index == 0);
        CHECK(loaded_descriptor->cbuf_offset == 0x110);
        CHECK(loaded_descriptor->indirect_offset == 0);
    }
}

/// A native record address that is c0[0x30] or the pointer loaded from *c0[0x30], through a
/// select or a phi, plus a lane offset. Ocarina of Time's near-leaf pass 539ffb3f650a34ac reads
/// its foliage unit placement records this way, from c0[12] or from *c0[12].
struct DirectOrLoadedPointer {
    KernelProgram fixture;
    IR::Block taken{fixture.pool};
    IR::Block merge{fixture.pool};

    IR::U64 RecordAddress(bool phi) {
        fixture.env.native = true;
        IR::IREmitter ir{fixture.entry};
        const IR::Value loaded{ir.LoadGlobal64(Address(ir, Word(ir, 0x30), Word(ir, 0x34)))};
        const IR::U32 loaded_low{ir.CompositeExtract(loaded, 0)};
        const IR::U32 loaded_high{ir.CompositeExtract(loaded, 1)};
        IR::U32 low;
        IR::U32 high;
        IR::Block* access{&fixture.entry};
        if (phi) {
            fixture.entry.AddBranch(&taken);
            fixture.entry.AddBranch(&merge);
            taken.AddBranch(&merge);
            fixture.program.blocks = {&fixture.entry, &taken, &merge};
            fixture.program.post_order_blocks = {&merge, &taken, &fixture.entry};
            for (u32 index = 0; index < fixture.program.blocks.size(); ++index) {
                fixture.program.blocks[index]->SetOrder(index);
            }
            IR::IREmitter direct{taken};
            const IR::U32 direct_low{Word(direct, 0x30)};
            const IR::U32 direct_high{Word(direct, 0x34)};
            merge.AppendNewInst(IR::Opcode::Phi, {});
            IR::Inst* const low_phi{&merge.back()};
            low_phi->SetFlags(IR::Type::U32);
            merge.AppendNewInst(IR::Opcode::Phi, {});
            IR::Inst* const high_phi{&merge.back()};
            high_phi->SetFlags(IR::Type::U32);
            low_phi->AddPhiOperand(&fixture.entry, loaded_low);
            low_phi->AddPhiOperand(&taken, direct_low);
            high_phi->AddPhiOperand(&fixture.entry, loaded_high);
            high_phi->AddPhiOperand(&taken, direct_high);
            low = IR::U32{IR::Value{low_phi}};
            high = IR::U32{IR::Value{high_phi}};
            access = &merge;
        } else {
            const IR::U1 direct{ir.ILessThan(ir.LocalInvocationIdX(), ir.Imm32(16), false)};
            low = IR::U32{ir.Select(direct, Word(ir, 0x30), loaded_low)};
            high = IR::U32{ir.Select(direct, Word(ir, 0x34), loaded_high)};
        }
        record.emplace(*access);
        const IR::U32 offset{
            record->ShiftLeftLogical(record->LocalInvocationIdX(), record->Imm32(4))};
        return Address(*record, record->IAdd(low, offset), high);
    }

    std::ptrdiff_t Count(IR::Opcode opcode) const {
        std::ptrdiff_t total{};
        for (const IR::Block* const block : fixture.program.blocks) {
            total += std::ranges::count_if(block->Instructions(), [=](const IR::Inst& inst) {
                return inst.GetOpcode() == opcode;
            });
        }
        return total;
    }

    std::size_t LoadedDescriptors() const {
        return std::ranges::count_if(fixture.program.info.storage_buffers_descriptors,
                                     [](const auto& desc) { return desc.indirect_offset == 0; });
    }

    std::optional<IR::IREmitter> record;
};

TEST_CASE("Native accesses through a cbuf pointer or a pointer loaded from it bind both",
          "[shader][storage_buffer]") {
    // Pairing the address with c0[0x30] alone rebased the loaded path on c0[0x30]'s binding,
    // where its records read out of bounds as zeros.
    const bool phi = GENERATE(false, true);
    CAPTURE(phi);
    DirectOrLoadedPointer program;
    static_cast<void>(program.record->LoadGlobal128(program.RecordAddress(phi)));

    CHECK(program.fixture.TrackedOffsets() == std::vector<u32>{0x30, 0x30});
    CHECK(program.fixture.program.info.native_global_bounds);
    CHECK(program.LoadedDescriptors() == 1);
    CHECK(program.Count(IR::Opcode::LoadStorage64) == 1);
    CHECK(program.Count(IR::Opcode::LoadGlobal128) == 1);
    CHECK(program.Count(IR::Opcode::LoadStorage128) == 0);
}

TEST_CASE("Native accesses the backend cannot route keep their paired cbuf pointer",
          "[shader][storage_buffer]") {
    // The SPIR-V backend has no global subword accesses or global exchange atomics, and routes
    // no global access without 64-bit integers, so these accesses are rebased on the pair rather
    // than left global.
    const bool phi = GENERATE(false, true);
    const int access = GENERATE(0, 1, 2);
    const bool atomic = access == 1;
    CAPTURE(phi, access);
    DirectOrLoadedPointer program;
    program.fixture.int64 = access != 2;
    const IR::U64 address{program.RecordAddress(phi)};
    if (access == 2) {
        static_cast<void>(program.record->LoadGlobal128(address));
    } else if (atomic) {
        static_cast<void>(program.record->GlobalAtomicExchange(address, program.record->Imm32(1)));
    } else {
        static_cast<void>(program.record->LoadGlobalU16(address));
    }

    CHECK(program.fixture.TrackedOffsets() == std::vector<u32>{0x30});
    CHECK_FALSE(program.fixture.program.info.native_global_bounds);
    CHECK(program.LoadedDescriptors() == 0);
    const std::array storage{IR::Opcode::LoadStorageU16, IR::Opcode::StorageAtomicExchange32,
                             IR::Opcode::LoadStorage128};
    const std::array global{IR::Opcode::LoadGlobalU16, IR::Opcode::GlobalAtomicExchange32,
                            IR::Opcode::LoadGlobal128};
    CHECK(program.Count(storage[access]) == 1);
    CHECK(program.Count(global[access]) == 0);
}

TEST_CASE("Loaded pointers that compete with a pair do not take the last storage buffer",
          "[shader][storage_buffer]") {
    // 31 more pointers fill every binding with c0[0x30]; the loaded pointer would be the 33rd.
    DirectOrLoadedPointer program;
    static_cast<void>(program.record->LoadGlobal128(program.RecordAddress(false)));
    IR::IREmitter ir{program.fixture.entry};
    for (u32 slot = 0; slot < Shader::Info::MAX_SSBOS - 1; ++slot) {
        const u32 offset{0x100 + slot * 8};
        static_cast<void>(ir.LoadGlobal32(Address(ir, Word(ir, offset), Word(ir, offset + 4))));
    }

    CHECK(program.fixture.TrackedOffsets().size() == Shader::Info::MAX_SSBOS);
    CHECK_FALSE(program.fixture.program.info.native_global_bounds);
    CHECK(program.LoadedDescriptors() == 0);
    CHECK(program.Count(IR::Opcode::LoadStorage128) == 1);
    CHECK(program.Count(IR::Opcode::LoadGlobal128) == 0);
}

// The loop reads its records through the table entry at *c0[0x30] or *c0[0x38], loaded before
// it. That load binds c0[0x30] and c0[0x38] in the first pass, with `pointers` more pointers.
static void EmitTableLoop(KernelProgram& fixture, u32 pointers) {
    fixture.env.native = true;
    fixture.entry.AddBranch(&fixture.loop);
    fixture.loop.AddBranch(&fixture.loop);
    fixture.program.blocks = {&fixture.entry, &fixture.loop};
    fixture.program.post_order_blocks = {&fixture.loop, &fixture.entry};
    IR::IREmitter entry{fixture.entry};
    const IR::U32 lane{entry.LocalInvocationIdX()};
    const IR::U1 first{entry.ILessThan(lane, entry.Imm32(16), false)};
    const IR::U32 holder_low{entry.Select(first, Word(entry, 0x30), Word(entry, 0x38))};
    const IR::U32 holder_high{entry.Select(first, Word(entry, 0x34), Word(entry, 0x3c))};
    const IR::Value table{entry.LoadGlobal64(Address(entry, holder_low, holder_high))};
    for (u32 slot = 0; slot < pointers; ++slot) {
        const u32 offset{0x100 + slot * 8};
        static_cast<void>(
            entry.LoadGlobal32(Address(entry, Word(entry, offset), Word(entry, offset + 4))));
    }
    IR::IREmitter body{fixture.loop};
    const IR::U32 record{body.IAdd(IR::U32{body.CompositeExtract(table, 0)},
                                   body.ShiftLeftLogical(lane, body.Imm32(4)))};
    static_cast<void>(
        body.LoadGlobal128(Address(body, record, IR::U32{body.CompositeExtract(table, 1)})));
}

TEST_CASE("Pointer table entries are charged only for bindings the first pass left free",
          "[shader][storage_buffer]") {
    // The first pass leaves a binding for each entry. The loop is collected first: charged for its
    // holder as well, the first entry shut out the second, as Ocarina of Time's foliage culling
    // compute 331e607152ce78d5 bound 4 of its 8 per-type instance tables.
    KernelProgram fixture;
    EmitTableLoop(fixture, Shader::Info::MAX_SSBOS - 4);

    CHECK(fixture.TrackedOffsets().size() == Shader::Info::MAX_SSBOS);
    CHECK(std::ranges::count_if(fixture.program.info.storage_buffers_descriptors,
                                [](const auto& desc) { return desc.indirect_offset != ~0U; }) == 2);
}

TEST_CASE("Pointer table entries stay within the stage's storage buffer limit",
          "[shader][storage_buffer]") {
    // The stage binds 31 storage buffers: the first pass leaves one binding, for one entry.
    KernelProgram fixture;
    fixture.storage_buffer_limit = 31;
    EmitTableLoop(fixture, 28);

    CHECK(fixture.TrackedOffsets().size() == 31);
    CHECK(std::ranges::count_if(fixture.program.info.storage_buffers_descriptors,
                                [](const auto& desc) { return desc.indirect_offset != ~0U; }) == 1);
}

TEST_CASE("Programs that read more pointers than a stage binds are rejected",
          "[shader][storage_buffer]") {
    KernelProgram fixture;
    fixture.env.native = true;
    SECTION("Stage limit") {
        fixture.storage_buffer_limit = 4;
    }
    SECTION("Descriptor set limit") {
        fixture.set_storage_buffer_limit = 4;
    }
    IR::IREmitter ir{fixture.entry};
    for (u32 slot = 0; slot < 5; ++slot) {
        const u32 offset{0x100 + slot * 8};
        static_cast<void>(ir.LoadGlobal32(Address(ir, Word(ir, offset), Word(ir, offset + 4))));
    }

    CHECK_THROWS_AS(fixture.TrackedOffsets(), Shader::NotImplementedException);
}

TEST_CASE("Dual vertex halves that bind more storage buffers than the stage are rejected",
          "[shader][storage_buffer]") {
    // Vertex B's buffers join Vertex A's; one they share counts once.
    Shader::Info base;
    base.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0x110, .count = 1});
    Shader::Info source;
    source.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x110, .count = 1});
    source.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x120, .count = 1});
    SECTION("Within the limit") {
        Shader::Optimization::JoinStorageInfo(base, source, 2);
        CHECK(base.storage_buffers_descriptors.size() == 2);
    }
    SECTION("Past the limit") {
        source.storage_buffers_descriptors.push_back(
            {.cbuf_index = 0, .cbuf_offset = 0x130, .count = 1});
        CHECK_THROWS_AS(Shader::Optimization::JoinStorageInfo(base, source, 2),
                        Shader::NotImplementedException);
    }
}

TEST_CASE("Pointer table entries that would overflow the storage buffers are left out",
          "[shader][storage_buffer]") {
    // Records at c0[0x30] or in the table *c0[0x38], at an offset c0[0x40] the environment cannot
    // tell from a pointer. The table entry competes with the pair, so with it the access binds
    // every candidate, c0[0x40] too: 33 storage buffers with 29 more pointers. The program keeps
    // the 30 of the pass without table entries.
    KernelProgram fixture;
    fixture.env.native = true;
    IR::IREmitter ir{fixture.entry};
    const IR::Value table{
        ir.LoadGlobal64(ir.PackUint2x32(ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x38), 64, false)))};
    const IR::U1 direct{ir.ILessThan(ir.LocalInvocationIdX(), ir.Imm32(16), false)};
    const IR::U32 low{ir.Select(direct, Word(ir, 0x30), IR::U32{ir.CompositeExtract(table, 0)})};
    const IR::U32 high{ir.Select(direct, Word(ir, 0x34), IR::U32{ir.CompositeExtract(table, 1)})};
    static_cast<void>(ir.LoadGlobal128(Address(ir, ir.IAdd(low, Word(ir, 0x40)), high)));
    for (u32 slot = 0; slot < Shader::Info::MAX_SSBOS - 3; ++slot) {
        const u32 offset{0x100 + slot * 8};
        static_cast<void>(ir.LoadGlobal32(Address(ir, Word(ir, offset), Word(ir, offset + 4))));
    }

    CHECK(fixture.TrackedOffsets().size() == Shader::Info::MAX_SSBOS - 2);
    CHECK_FALSE(fixture.program.info.native_global_bounds);
    CHECK(std::ranges::none_of(fixture.program.info.storage_buffers_descriptors,
                               [](const auto& desc) { return desc.indirect_offset != ~0U; }));
}

TEST_CASE("NVN fallback descriptors use guest slots rather than compact binding indices",
          "[shader][storage_buffer]") {
    Shader::Info info{};
    info.nvn_buffer_base = 0x310;
    info.nvn_buffer_used.set(3);
    info.nvn_buffer_used.set(15);
    const std::vector<Shader::StorageBufferDescriptor> descriptors{
        {1, 0x340, 1, false},
        {0, 0x340, 1, false},
        {0, 0x400, 1, false},
        {0, 0x330, 1, false},
        {0, 0x341, 1, false},
        {0, 0x410, 1, false},
        {0, 0x300, 1, false},
    };
    const auto selected = [&](const auto& ordered) {
        std::vector<u32> offsets;
        for (const auto& descriptor : ordered) {
            if (Shader::Backend::UsesNvnStorageDescriptor(info, descriptor)) {
                offsets.push_back(descriptor.cbuf_offset);
            }
        }
        return offsets;
    };
    CHECK(selected(descriptors) == std::vector<u32>{0x340, 0x400});
    auto reversed = descriptors;
    std::ranges::reverse(reversed);
    CHECK(selected(reversed) == std::vector<u32>{0x400, 0x340});
}
