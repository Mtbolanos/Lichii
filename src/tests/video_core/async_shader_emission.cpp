// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/thread_worker.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"

namespace {
class VertexEnvironment final : public Shader::Environment {
public:
    VertexEnvironment() {
        stage = Shader::Stage::VertexB;
    }
    u64 ReadInstruction(u32) override {
        throw std::runtime_error("unexpected instruction read");
    }
    u32 ReadCbufValue(u32, u32) override {
        throw std::runtime_error("unexpected buffer read");
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

struct GraphicsPrograms {
    Shader::ObjectPool<Shader::IR::Inst> instructions;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    std::array<Shader::IR::Program, 2> programs;

    GraphicsPrograms() {
        for (size_t stage = 0; stage < programs.size(); ++stage) {
            auto& program = programs[stage];
            auto* block = blocks.Create(instructions);
            program.stage = stage == 0 ? Shader::Stage::VertexB : Shader::Stage::Fragment;
            program.blocks = {block};
            program.post_order_blocks = {block};
            program.syntax_list = {
                {{.block = block}, Shader::IR::AbstractSyntaxNode::Type::Block},
                {{}, Shader::IR::AbstractSyntaxNode::Type::Return},
            };
            program.info.constant_buffer_descriptors = {{0, 1}};
            program.info.storage_buffers_descriptors = {{0, 0x110, 1, true}};
            program.info.used_storage_buffer_types = Shader::IR::Type::U32;
            Shader::IR::IREmitter ir{*block};
            block->AppendNewInst(Shader::IR::Opcode::WriteStorage32,
                                 {ir.Imm32(0), ir.Imm32(0), ir.Imm32(0x12345678)});
        }
    }

    void AddControlFlowAndDualVertex() {
        namespace IR = Shader::IR;
        auto& vertex_b = programs[0];
        auto* entry = vertex_b.blocks.front();
        auto* body = blocks.Create(instructions);
        auto* merge = blocks.Create(instructions);
        entry->AddBranch(body);
        entry->AddBranch(merge);
        body->AddBranch(merge);
        IR::IREmitter start{*entry};
        const auto vertex_id = start.GetAttributeU32(IR::Attribute::VertexId);
        const auto condition = start.ConditionRef(start.IEqual(vertex_id, start.Imm32(0)));
        IR::IREmitter branch{*body};
        const auto incremented = branch.IAdd(vertex_id, branch.Imm32(1));
        merge->AppendNewInst(IR::Opcode::Phi, {});
        auto& phi = merge->back();
        phi.SetFlags(IR::Type::U32);
        phi.AddPhiOperand(entry, vertex_id);
        phi.AddPhiOperand(body, incremented);
        IR::IREmitter end{*merge};
        merge->AppendNewInst(IR::Opcode::WriteStorage32,
                             {end.Imm32(0), end.Imm32(4), IR::Value{&phi}});
        vertex_b.blocks = {entry, body, merge};
        vertex_b.post_order_blocks = {merge, body, entry};
        vertex_b.syntax_list = {
            {{.block = entry}, IR::AbstractSyntaxNode::Type::Block},
            {{.if_node{condition, body, merge}}, IR::AbstractSyntaxNode::Type::If},
            {{.block = body}, IR::AbstractSyntaxNode::Type::Block},
            {{.end_if{merge}}, IR::AbstractSyntaxNode::Type::EndIf},
            {{.block = merge}, IR::AbstractSyntaxNode::Type::Block},
            {{}, IR::AbstractSyntaxNode::Type::Return},
        };
        vertex_b.local_memory_size = 16;
        auto* first = blocks.Create(instructions);
        first->AddBranch(entry);
        IR::IREmitter first_ir{*first};
        first_ir.Prologue();
        first_ir.SetAttribute(IR::Attribute::PositionX, first_ir.Imm32(0.5f), first_ir.Imm32(0));
        first_ir.Epilogue();
        entry->PrependNewInst(entry->begin(), IR::Opcode::Prologue);
        end.Epilogue();
        Shader::IR::Program vertex_a;
        vertex_a.stage = Shader::Stage::VertexA;
        vertex_a.local_memory_size = 8;
        vertex_a.blocks = {first};
        vertex_a.post_order_blocks = {first};
        vertex_a.syntax_list = {
            {{.block = first}, IR::AbstractSyntaxNode::Type::Block},
            {{}, IR::AbstractSyntaxNode::Type::Return},
        };
        VertexEnvironment env;
        programs[0] = Shader::Maxwell::MergeDualVertexPrograms(vertex_a, vertex_b, env,
                                                               Shader::HostTranslateInfo{});
        for (size_t index = 0; index < programs[0].blocks.size(); ++index) {
            programs[0].blocks[index]->SetOrder(index);
        }
    }

    std::array<std::vector<u32>, 2> Emit(const std::array<std::vector<u32>, 2>& reference = {}) {
        Shader::Profile profile{};
        profile.supported_spirv = 0x00010300;
        profile.unified_descriptor_binding = true;
        profile.support_vertex_instance_id = false;
        Shader::Backend::Bindings bindings;
        std::array<std::vector<u32>, 2> result;
        for (size_t stage = 0; stage < programs.size(); ++stage) {
            result[stage] = Vulkan::EmitPipelineShader(profile, {}, programs[stage], bindings,
                                                       stage, reference[stage]);
        }
        return result;
    }
};
} // namespace

TEST_CASE("Graphics IR survives transfer to the shader worker and emits identical SPIR-V",
          "[video_core][async_draw][spirv]") {
    const auto producer_thread = std::this_thread::get_id();
    std::promise<void> release;
    auto released = release.get_future().share();
    Common::ThreadWorker workers{1, "TestPipelineBuilder"};
    auto job = std::make_shared<GraphicsPrograms>();
    const auto reference = job->Emit();
    std::weak_ptr<GraphicsPrograms> lifetime = job;
    std::promise<std::array<std::vector<u32>, 2>> result;
    auto emitted = result.get_future();
    std::thread::id build_thread;
    workers.QueueWork([job = std::move(job), released, &result, &build_thread] {
        released.wait();
        build_thread = std::this_thread::get_id();
        try {
            result.set_value(job->Emit());
        } catch (...) {
            result.set_exception(std::current_exception());
        }
    });
    const bool held_by_worker = !lifetime.expired();
    const bool still_pending =
        emitted.wait_for(std::chrono::seconds{0}) != std::future_status::ready;
    release.set_value();
    const auto actual = emitted.get();
    workers.WaitForRequests();
    REQUIRE(held_by_worker);
    REQUIRE(still_pending);
    REQUIRE(build_thread != producer_thread);
    REQUIRE(actual == reference);
    REQUIRE(lifetime.expired());
}

TEST_CASE("Production verification preserves merged vertex control flow and stage metadata",
          "[video_core][async_draw][spirv]") {
    auto job = std::make_shared<GraphicsPrograms>();
    job->AddControlFlowAndDualVertex();
    const auto vertex_info = job->programs[0].info;
    const auto fragment_info = job->programs[1].info;
    const auto reference = job->Emit();
    std::array<std::vector<u32>, 2> emitted, verified;
    Common::ThreadWorker workers{1, "PipelineTest"};
    Vulkan::PipelineBuild build{&workers, [&] {
                                    emitted = job->Emit();
                                    verified = job->Emit(reference);
                                }};
    build.Wait();
    REQUIRE_FALSE(build.Failed());
    REQUIRE(emitted == reference);
    REQUIRE(verified == reference);
    REQUIRE(job->programs[0].stage == Shader::Stage::VertexB);
    REQUIRE(job->programs[0].blocks.size() == 4);
    REQUIRE(job->programs[0].local_memory_size == 16);
    REQUIRE(vertex_info.loads.mask == job->programs[0].info.loads.mask);
    REQUIRE(vertex_info.stores.mask == job->programs[0].info.stores.mask);
    REQUIRE(vertex_info.storage_buffers_descriptors ==
            job->programs[0].info.storage_buffers_descriptors);
    REQUIRE(vertex_info.storage_buffers_descriptors.size() == 1);
    REQUIRE(vertex_info.storage_buffers_descriptors[0].is_written);
    REQUIRE(fragment_info.constant_buffer_descriptors ==
            job->programs[1].info.constant_buffer_descriptors);
    REQUIRE(fragment_info.storage_buffers_descriptors ==
            job->programs[1].info.storage_buffers_descriptors);
}

TEST_CASE("Production verification selects the reference when emitted words differ",
          "[video_core][async_draw][spirv]") {
    GraphicsPrograms job;
    const auto emitted = job.Emit();
    auto reference = emitted;
    reference[0][2] ^= 1;
    REQUIRE(job.Emit(reference) == reference);
    REQUIRE(job.Emit() == emitted);
}
