// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "video_core/shader_handle_program.h"

namespace {
namespace IR = Shader::IR;
using IR::Opcode;
using IR::Value;

class HandleEnvironment final : public Shader::Environment {
public:
    bool integer_texture{};
    u32 format_reads{};
    u32 expected_handle{0xabc};
    std::map<std::pair<u32, u32>, u32> cbuf_values{{{0, 0}, 0xabc}};

    u64 ReadInstruction(u32) override { throw std::logic_error("Unexpected instruction read"); }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        const auto it = cbuf_values.find({index, offset});
        REQUIRE(it != cbuf_values.end());
        return it->second;
    }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        REQUIRE(handle == expected_handle);
        ++format_reads;
        return integer_texture;
    }
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

struct HandleProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block block{pool};
    IR::Block left{pool};
    IR::Block right{pool};
    IR::Block merge{pool};
    IR::Block* insertion_block{&block};
    IR::Program program;
    HandleEnvironment env;

    HandleProgram() {
        program.stage = Shader::Stage::Fragment;
        program.blocks = {&block};
        program.post_order_blocks = {&block};
    }

    Value Emit(Opcode opcode, std::initializer_list<Value> args = {}) {
        insertion_block->AppendNewInst(opcode, args);
        return Value{&insertion_block->back()};
    }

    Value Storage(u32 offset) {
        return Emit(Opcode::LoadStorage32, {Value{0U}, Value{offset}});
    }

    Value Add(Value lhs, Value rhs) {
        return Emit(Opcode::IAdd32, {lhs, rhs});
    }

    Value Phi(Value first, Value second) {
        block.AddBranch(&left);
        block.AddBranch(&right);
        left.AddBranch(&merge);
        right.AddBranch(&merge);
        program.blocks = {&block, &left, &right, &merge};
        program.post_order_blocks = {&merge, &right, &left, &block};
        insertion_block = &merge;
        const Value value{Emit(Opcode::Phi)};
        value.Inst()->SetFlags(IR::Type::U32);
        value.Inst()->AddPhiOperand(&left, first);
        value.Inst()->AddPhiOperand(&right, second);
        return value;
    }

    IR::Inst& Sample(Value handle) {
        IR::IREmitter ir{*insertion_block};
        IR::TextureInstInfo flags{};
        flags.type.Assign(Shader::TextureType::Color2D);
        return *ir.ImageSampleExplicitLod(handle,
                                         ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f)),
                                         ir.Imm32(0.0f), {}, flags).Inst();
    }

    void Run() {
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        Shader::Optimization::TexturePass(env, program, host);
    }

    const Shader::TextureDescriptor& Descriptor(const IR::Inst& sample) const {
        REQUIRE(sample.GetOpcode() == Opcode::ImageSampleExplicitLod);
        const u32 index{sample.Flags<IR::TextureInstInfo>().descriptor_index};
        REQUIRE(index < program.info.texture_descriptors.size());
        return program.info.texture_descriptors[index];
    }

    u32 Evaluate(const IR::Inst& sample, u32 first = 0x1234, u32 second = 0x50) const {
        const u32 index{Descriptor(sample).handle_program};
        REQUIRE(index < program.info.handle_programs.size());
        const auto result{Shader::EvaluateHandleProgram(
            program.info.handle_programs[index],
            [](u32, u32) -> u32 { throw std::logic_error("Unexpected evaluator cbuf read"); },
            [=](u32 binding, u32 offset) {
                REQUIRE(binding == 0);
                REQUIRE((offset == 0 || offset == 4));
                return offset == 0 ? first : second;
            })};
        REQUIRE(result.has_value());
        return *result;
    }
};
}

TEST_CASE("Sampled descriptors distinguish static and computed handles",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    auto& first{shader.Sample(shader.Storage(0))};
    auto& second{shader.Sample(shader.Storage(4))};
    auto& repeated{shader.Sample(shader.Storage(0))};
    auto& fixed{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}}))};
    shader.Run();

    CHECK(shader.program.info.handle_programs.size() == 2);
    CHECK(shader.program.info.texture_descriptors.size() == 3);
    CHECK(shader.Descriptor(first).handle_program != shader.Descriptor(second).handle_program);
    CHECK(shader.Descriptor(first).handle_program == shader.Descriptor(repeated).handle_program);
    CHECK(shader.Descriptor(fixed).handle_program == ~0U);
    CHECK(first.Flags<IR::TextureInstInfo>().descriptor_index ==
          repeated.Flags<IR::TextureInstInfo>().descriptor_index);
    CHECK(first.Flags<IR::TextureInstInfo>().descriptor_index !=
          fixed.Flags<IR::TextureInstInfo>().descriptor_index);
    CHECK(shader.Evaluate(first) == 0x1234);
    CHECK(shader.Evaluate(second) == 0x50);
    CHECK(shader.Evaluate(repeated, 0x5678) == 0x5678);
}

TEST_CASE("Computed handles do not share the static texture format cache",
          "[video_core][texture_handles]") {
    for (const bool dynamic_first : {false, true}) {
        CAPTURE(dynamic_first);
        HandleProgram shader;
        shader.env.integer_texture = true;
        const Value dynamic{shader.Storage(0)};
        const Value fixed{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
        auto& first{shader.Sample(dynamic_first ? dynamic : fixed)};
        auto& second{shader.Sample(dynamic_first ? fixed : dynamic)};
        auto& third{shader.Sample(fixed)};
        shader.Run();

        const auto& static_sample{dynamic_first ? second : first};
        const auto& dynamic_sample{dynamic_first ? first : second};
        CHECK(shader.program.info.texture_descriptors.size() == 2);
        CHECK(static_sample.Flags<IR::TextureInstInfo>().is_integer == 1);
        CHECK(third.Flags<IR::TextureInstInfo>().is_integer == 1);
        CHECK(dynamic_sample.Flags<IR::TextureInstInfo>().is_integer == 0);
        CHECK(shader.Descriptor(static_sample).is_integer);
        CHECK_FALSE(shader.Descriptor(dynamic_sample).is_integer);
        CHECK(shader.env.format_reads == 1);
    }
}

TEST_CASE("A sampled handle keeps a storage-indexed cbuf texture with its sampler",
          "[video_core][texture_handles]") {
    for (const bool sampler_first : {false, true}) {
        HandleProgram shader;
        shader.env.cbuf_values = {{{6, 0}, 0x123}, {{2, 0x570}, 0x24000000}};
        shader.env.expected_handle = 0x24000123;
        const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
        const Value texture{shader.Emit(Opcode::GetCbufU32, {Value{6U}, offset})};
        const Value sampler{shader.Emit(Opcode::GetCbufU32, {Value{2U}, Value{0x570U}})};
        auto& sample{shader.Sample(shader.Emit(Opcode::BitwiseOr32,
                                              {sampler_first ? sampler : texture,
                                               sampler_first ? texture : sampler}))};
        shader.Run();
        const auto& descriptor = shader.Descriptor(sample);
        CHECK(descriptor.handle_program == ~0U);
        CHECK(descriptor.count > 1);
        CHECK(descriptor.size_shift == 4);
        CHECK(descriptor.cbuf_index == 6);
        CHECK(descriptor.cbuf_offset == 0);
        CHECK(descriptor.has_secondary);
        CHECK(descriptor.secondary_cbuf_index == 2);
        CHECK(descriptor.secondary_cbuf_offset == 0x570);
        CHECK(shader.program.info.handle_programs.empty());
        REQUIRE_FALSE(sample.Arg(0).IsEmpty());
        const auto* selection = sample.Arg(0).InstRecursive();
        REQUIRE(selection->GetOpcode() == Opcode::UMin32);
        const auto* index = selection->Arg(0).InstRecursive();
        REQUIRE(index->GetOpcode() == Opcode::ShiftRightLogical32);
        CHECK(index->Arg(0).Resolve() == offset.Resolve());
        CHECK(index->Arg(1).U32() == 4);
    }
}

TEST_CASE("Combining static cbuf handles keeps the static descriptor path",
          "[video_core][texture_handles]") {
    HandleProgram shader;
    const Value texture{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
    const Value sampler{shader.Emit(Opcode::GetCbufU32, {Value{0U}, Value{0U}})};
    auto& sample{shader.Sample(shader.Emit(Opcode::BitwiseOr32, {texture, sampler}))};
    shader.Run();
    CHECK(shader.Descriptor(sample).handle_program == ~0U);
    CHECK(shader.Descriptor(sample).has_secondary);
    CHECK(shader.program.info.handle_programs.empty());
}

TEST_CASE("Texture arrays reserve the final static and buffer resources",
          "[video_core][texture_handles]") {
    for (const u32 arrays : {1U, 2U}) {
        HandleProgram shader;
        const Value offset{shader.Emit(Opcode::ShiftLeftLogical32, {shader.Storage(0), Value{4U}})};
        auto& sample{shader.Sample(shader.Emit(Opcode::GetCbufU32, {Value{0U}, offset}))};
        shader.Run();
        auto& info = shader.program.info;
        const auto descriptor = info.texture_descriptors.front();
        for (u32 i = 1; i < arrays; ++i) {
            info.texture_descriptors.push_back(descriptor);
        }
        for (u32 i = 0; i < 3; ++i) {
            auto fixed = descriptor;
            fixed.count = 1;
            info.texture_descriptors.push_back(fixed);
        }
        for (u32 i = 0; i < 5; ++i) {
            info.constant_buffer_descriptors.push_back({.index = i, .count = 1});
        }
        for (u32 i = 0; i < 2; ++i) {
            info.storage_buffers_descriptors.push_back({.count = 1});
        }
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        host.max_per_stage_resources = 16;
        host.max_per_stage_descriptor_sampled_images = 16;
        host.max_descriptor_set_sampled_images = 16;
        Shader::Optimization::LimitSampledTextureArrays(shader.program, host);
        for (u32 i = 0; i < arrays; ++i) {
            CHECK(info.texture_descriptors[i].count == 6 / arrays);
        }
        CHECK(Shader::NumDescriptors(info.texture_descriptors) == 9);
        CHECK(sample.Arg(0).InstRecursive()->Arg(1).U32() == 6 / arrays - 1);
        if (arrays == 2) {
            info.texture_descriptors[0].count = 2;
            info.texture_descriptors[1].count = 12;
            host.max_per_stage_resources = 24;
            host.max_per_stage_descriptor_sampled_images = 24;
            host.max_descriptor_set_sampled_images = 24;
            Shader::Optimization::LimitSampledTextureArrays(shader.program, host);
            CHECK(info.texture_descriptors[0].count == 2);
            CHECK(info.texture_descriptors[1].count == 12);
        }
        host.max_per_stage_resources = 10;
        REQUIRE_THROWS(Shader::Optimization::LimitSampledTextureArrays(shader.program, host));
    }
}

TEST_CASE("Failed Phi handle subtrees discard their cached values",
          "[video_core][texture_handles]") {
    for (const bool nested : {false, true}) {
        for (const bool retained_slot : {false, true}) {
            CAPTURE(nested, retained_slot);
            HandleProgram shader;
            const Value shared{shader.Storage(0)};
            const Value retained{shader.Storage(4)};
            const Value unsupported{shader.Emit(Opcode::GetRegister, {Value{IR::Reg::R0}})};
            const Value subtree{nested ? shader.Add(shared, retained) : shared};
            const Value failed{shader.Add(subtree, unsupported)};
            Value fallback{shader.Add(subtree, Value{7U})};
            if (retained_slot) {
                fallback = shader.Add(fallback, retained);
            }
            Value handle{shader.Phi(failed, fallback)};
            if (retained_slot) {
                handle = shader.Add(retained, handle);
            }
            auto& sample{shader.Sample(handle)};
            shader.Run();

            REQUIRE(shader.program.info.handle_programs.size() == 1);
            CHECK(shader.Evaluate(sample) ==
                  0x1234U + 7U + (nested ? 0x50U : 0U) + (retained_slot ? 2U * 0x50U : 0U));
            CHECK(shader.Evaluate(sample, 0x5678, 0x90) ==
                  0x5678U + 7U + (nested ? 0x90U : 0U) + (retained_slot ? 2U * 0x90U : 0U));
        }
    }
}

namespace {
struct HandleTestMemory {
    std::map<GPUVAddr, u32> words;
    template <typename T>
    T Read(GPUVAddr address) const {
        const auto word = [&](GPUVAddr at) -> u64 {
            const auto it = words.find(at);
            return it != words.end() ? it->second : 0U;
        };
        if constexpr (sizeof(T) == 8) {
            return static_cast<T>(word(address) | (word(address + 4) << 32));
        } else {
            return static_cast<T>(word(address));
        }
    }
};
} // namespace

TEST_CASE("Storage handle words are read from the host-aligned buffer base",
          "[video_core][texture_handles]") {
    // The program's storage offsets are relative to the buffer address aligned down to the host
    // storage alignment (StorageOffset), where the buffer cache binds the buffer.
    constexpr GPUVAddr cbuf = 0x1000;
    HandleTestMemory memory;
    memory.words[cbuf + 0x10] = 0x2008; // the storage buffer address, 8 bytes past 16
    memory.words[cbuf + 0x14] = 0;
    memory.words[0x2008] = 0x11; // the handle the shader reads
    memory.words[0x2010] = 0x22;

    Shader::Info info;
    info.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    // Offset 8 from the base aligned to 16, as StorageOffset gives it for a read at 0x2008.
    info.handle_programs.push_back({static_cast<u32>(Shader::HandleOp::Immediate), 8U,
                                    static_cast<u32>(Shader::HandleOp::Storage), 0U});
    const auto cbuf_address = [](u32 index) -> GPUVAddr { return index == 0 ? cbuf : 0; };
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 16) == 0x11U);
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 4) == 0x22U);
    CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 0) == 0x22U);

    SECTION("Indirect buffers align the address read from the table, not the table") {
        // The constant buffer points at a table that is not aligned. The buffer address is read
        // from the table entry at its exact address, then aligned down.
        memory.words[cbuf + 0x10] = 0x3004;
        memory.words[0x3004 + 0x8] = 0x2008;
        memory.words[0x3004 + 0xc] = 0;
        info.storage_buffers_descriptors[0].indirect_offset = 0x8;
        CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 16) == 0x11U);
        CHECK(VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_address, 4) == 0x22U);
    }
}
