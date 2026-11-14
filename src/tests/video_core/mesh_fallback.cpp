// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"

namespace {
using namespace Shader;
namespace SPIRV = Shader::Backend::SPIRV;

constexpr u32 OUTPUT_BINDING = 4;
constexpr u32 PAYLOAD_BINDING = 5;
constexpr u32 OFFSETS_BINDING = 6;
constexpr u32 ARGS_BINDING = 7;

struct Instruction {
    spv::Op op;
    std::span<const u32> operands;
};

std::vector<Instruction> Instructions(const std::vector<u32>& code) {
    std::vector<Instruction> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 count{code[offset] >> 16};
        REQUIRE(count != 0);
        REQUIRE(offset + count <= code.size());
        result.push_back({static_cast<spv::Op>(code[offset] & 0xffff),
                          std::span<const u32>(code).subspan(offset + 1, count - 1)});
        offset += count;
    }
    return result;
}

spv::ExecutionModel Model(const std::vector<u32>& code) {
    for (const Instruction& inst : Instructions(code)) {
        if (inst.op == spv::Op::OpEntryPoint) {
            return static_cast<spv::ExecutionModel>(inst.operands[0]);
        }
    }
    FAIL("no entry point");
    return {};
}

bool HasCapability(const std::vector<u32>& code, spv::Capability capability) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpCapability &&
               inst.operands[0] == static_cast<u32>(capability);
    });
}

bool HasExtension(const std::vector<u32>& code, std::string_view name) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpExtension &&
               std::string_view{reinterpret_cast<const char*>(inst.operands.data())} == name;
    });
}

bool HasVariable(const std::vector<u32>& code, spv::StorageClass storage_class) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpVariable &&
               inst.operands[2] == static_cast<u32>(storage_class);
    });
}

bool UsesStorageClass(const std::vector<u32>& code, spv::StorageClass storage_class) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return (inst.op == spv::Op::OpTypePointer &&
                inst.operands[1] == static_cast<u32>(storage_class)) ||
               (inst.op == spv::Op::OpVariable &&
                inst.operands[2] == static_cast<u32>(storage_class));
    });
}

std::vector<u32> Bindings(const std::vector<u32>& code) {
    std::vector<u32> bindings;
    for (const Instruction& inst : Instructions(code)) {
        if (inst.op == spv::Op::OpDecorate &&
            inst.operands[1] == static_cast<u32>(spv::Decoration::Binding)) {
            bindings.push_back(inst.operands[2]);
        }
    }
    std::ranges::sort(bindings);
    return bindings;
}

// EDEN_MESH_FALLBACK_TEST_OUTPUT=<dir>: write the modules for spirv-val.
void Dump(const std::vector<u32>& code, std::string_view name) {
    const char* const directory{std::getenv("EDEN_MESH_FALLBACK_TEST_OUTPUT")};
    if (!directory) {
        return;
    }
    std::filesystem::create_directories(directory);
    std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                       std::ios::binary};
    file.write(reinterpret_cast<const char*>(code.data()),
               static_cast<std::streamsize>(code.size() * sizeof(u32)));
    REQUIRE(file.good());
}

MeshInfo TriangleMesh() {
    MeshInfo mesh;
    mesh.payload_words = 32;
    mesh.max_vertices = 3;
    mesh.max_primitives = 1;
    mesh.topology = 4;
    for (const IR::Attribute attribute : {IR::Attribute::PositionX, IR::Attribute::PositionY,
                                          IR::Attribute::PositionZ, IR::Attribute::PositionW}) {
        mesh.vertex_lines.push_back(static_cast<u32>(attribute));
    }
    return mesh;
}

Profile MakeProfile() {
    Profile profile;
    profile.supported_spirv = 0x00010500;
    return profile;
}

// A one-triangle mesh program (or the task program launching it), as the frontend lowers them.
// With lane_id the mesh program writes its lane as the vertex's first attribute word.
std::vector<u32> EmitProgram(Stage stage, bool as_compute, bool with_task,
                             Profile profile = MakeProfile(), bool lane_id = false,
                             bool culled = false) {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = stage;
    program.workgroup_size = {32, 1, 1};
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    program.mesh = TriangleMesh();
    program.info.is_task = stage == Stage::Task;
    program.info.is_mesh = stage == Stage::Mesh;
    program.info.uses_workgroup_id = true;
    program.info.uses_local_invocation_id = true;
    IR::IREmitter ir{block};
    ir.Prologue();
    if (stage == Stage::Mesh) {
        const IR::U32 index{lane_id ? ir.LaneId() : ir.IsbeRead(ir.Imm32(0))};
        program.info.uses_subgroup_invocation_id = lane_id;
        ir.IsbeWriteAttribute(ir.Imm32(0), index);
        ir.IsbeWriteMapU8(ir.Imm32(4), ir.Imm32(0));
        ir.IsbeWriteMapU32(ir.Imm32(3), ir.Imm32(1));
    } else if (!culled) {
        ir.IsbeWriteAttribute(ir.Imm32(4), ir.Imm32(1)); // one mesh workgroup
    }
    ir.Epilogue();

    RuntimeInfo runtime_info;
    runtime_info.mesh_as_compute = as_compute;
    runtime_info.mesh_output_binding = OUTPUT_BINDING;
    if (with_task) {
        runtime_info.mesh_task_payload_binding = PAYLOAD_BINDING;
        runtime_info.mesh_task_offsets_binding = OFFSETS_BINDING;
        runtime_info.mesh_task_args_binding = ARGS_BINDING;
        runtime_info.mesh_task_payload_stride = program.mesh.payload_words;
    }
    Backend::Bindings bindings;
    return SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

void RequireNoMeshShading(const std::vector<u32>& code) {
    REQUIRE(Model(code) == spv::ExecutionModel::GLCompute);
    REQUIRE_FALSE(HasCapability(code, spv::Capability::MeshShadingEXT));
    REQUIRE_FALSE(HasExtension(code, "SPV_EXT_mesh_shader"));
    REQUIRE_FALSE(UsesStorageClass(code, spv::StorageClass::TaskPayloadWorkgroupEXT));
    REQUIRE_FALSE(HasVariable(code, spv::StorageClass::Output));
}
} // Anonymous namespace

TEST_CASE("Mesh fallback leaves native task and mesh programs as mesh shaders",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> mesh{EmitProgram(Stage::Mesh, false, false)};
    Dump(mesh, "native-mesh");
    REQUIRE(Model(mesh) == spv::ExecutionModel::MeshEXT);
    REQUIRE(HasCapability(mesh, spv::Capability::MeshShadingEXT));
    REQUIRE(Bindings(mesh).empty());

    const std::vector<u32> task{EmitProgram(Stage::Task, false, false)};
    Dump(task, "native-task");
    REQUIRE(Model(task) == spv::ExecutionModel::TaskEXT);
    REQUIRE(UsesStorageClass(task, spv::StorageClass::TaskPayloadWorkgroupEXT));
}

TEST_CASE("Native task programs zero the mesh workgroup count before the guest runs",
          "[video_core][spirv][mesh_fallback]") {
    // A task that culls every thread never writes payload word 1, which OpEmitMeshTasksEXT reads.
    const std::vector<u32> task{EmitProgram(Stage::Task, false, false, MakeProfile(), false, true)};
    Dump(task, "native-task-culled");
    std::unordered_map<u32, u32> constants;
    std::unordered_set<u32> payload_variables;
    std::unordered_map<u32, std::pair<u32, u32>> umins;
    std::unordered_map<u32, u32> payload_indices;
    const auto value_of = [&](u32 id) -> std::optional<u32> {
        if (const auto it{constants.find(id)}; it != constants.end()) {
            return it->second;
        }
        if (const auto it{umins.find(id)}; it != umins.end()) {
            const auto a{constants.find(it->second.first)};
            const auto b{constants.find(it->second.second)};
            if (a != constants.end() && b != constants.end()) {
                return (std::min)(a->second, b->second);
            }
        }
        return std::nullopt;
    };
    bool zeroed{};
    for (const Instruction& inst : Instructions(task)) {
        if (inst.op == spv::Op::OpControlBarrier) {
            break;
        }
        switch (inst.op) {
        case spv::Op::OpConstant:
            constants[inst.operands[1]] = inst.operands[2];
            break;
        case spv::Op::OpVariable:
            if (inst.operands[2] == static_cast<u32>(spv::StorageClass::TaskPayloadWorkgroupEXT)) {
                payload_variables.insert(inst.operands[1]);
            }
            break;
        case spv::Op::OpExtInst:
            if (inst.operands[3] == 38) { // GLSL.std.450 UMin
                umins[inst.operands[1]] = {inst.operands[4], inst.operands[5]};
            }
            break;
        case spv::Op::OpAccessChain:
            if (inst.operands.size() == 4 && payload_variables.contains(inst.operands[2])) {
                payload_indices[inst.operands[1]] = inst.operands[3];
            }
            break;
        case spv::Op::OpStore:
            if (const auto index{payload_indices.find(inst.operands[0])};
                index != payload_indices.end() &&
                value_of(index->second) == std::optional<u32>{1} &&
                value_of(inst.operands[1]) == std::optional<u32>{0}) {
                zeroed = true;
            }
            break;
        default:
            break;
        }
    }
    REQUIRE(zeroed);
}

TEST_CASE("Mesh fallback emits mesh programs as compute shaders writing storage buffers",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> alone{EmitProgram(Stage::Mesh, true, false)};
    Dump(alone, "fallback-mesh");
    RequireNoMeshShading(alone);
    REQUIRE(Bindings(alone) == std::vector<u32>{OUTPUT_BINDING});

    const std::vector<u32> after_task{EmitProgram(Stage::Mesh, true, true)};
    Dump(after_task, "fallback-mesh-after-task");
    RequireNoMeshShading(after_task);
    REQUIRE(Bindings(after_task) ==
            std::vector<u32>{OUTPUT_BINDING, PAYLOAD_BINDING, OFFSETS_BINDING, ARGS_BINDING});
}

TEST_CASE("Mesh fallback programs use compute subgroup support",
          "[video_core][spirv][mesh_fallback]") {
    // A device without VK_EXT_mesh_shader reports subgroup support for compute, not mesh.
    Profile profile{MakeProfile()};
    profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
    const std::vector<u32> code{EmitProgram(Stage::Mesh, true, false, profile, true)};
    Dump(code, "fallback-mesh-lane-id");
    RequireNoMeshShading(code);
    REQUIRE(HasCapability(code, spv::Capability::GroupNonUniform));
}

TEST_CASE("Mesh fallback emits task programs as compute shaders writing their payload",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> task{EmitProgram(Stage::Task, true, true)};
    Dump(task, "fallback-task");
    RequireNoMeshShading(task);
    REQUIRE(Bindings(task) == std::vector<u32>{PAYLOAD_BINDING});
}

TEST_CASE("Mesh fallback vertex shader pulls the compute output",
          "[video_core][spirv][mesh_fallback]") {
    const MeshInfo mesh{TriangleMesh()};
    const std::vector<u32> alone{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false)};
    Dump(alone, "fallback-vertex");
    REQUIRE(Model(alone) == spv::ExecutionModel::Vertex);
    REQUIRE(Bindings(alone) == std::vector<u32>{OUTPUT_BINDING});

    const std::vector<u32> after_task{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ARGS_BINDING, true)};
    Dump(after_task, "fallback-vertex-after-task");
    REQUIRE(Model(after_task) == spv::ExecutionModel::Vertex);
    REQUIRE(Bindings(after_task) == std::vector<u32>{OUTPUT_BINDING, ARGS_BINDING});
}

namespace {
bool HasBuiltIn(const std::vector<u32>& code, spv::BuiltIn builtin) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpDecorate &&
               inst.operands[1] == static_cast<u32>(spv::Decoration::BuiltIn) &&
               inst.operands[2] == static_cast<u32>(builtin);
    });
}
} // Anonymous namespace

TEST_CASE("Mesh fallback blocks hold one index per point and two per line",
          "[video_core][spirv][mesh_fallback]") {
    MeshInfo mesh{TriangleMesh()};
    mesh.max_primitives = 8;
    const u32 vertex_words{mesh.max_vertices * static_cast<u32>(mesh.vertex_lines.size())};
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8 * 3);
    mesh.topology = 1;
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8 * 2);
    mesh.topology = 0;
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8);
}

TEST_CASE("Mesh fallback vertex shader gives points a size and culls unused points and lines",
          "[video_core][spirv][mesh_fallback]") {
    MeshInfo mesh{TriangleMesh()};
    REQUIRE_FALSE(HasBuiltIn(
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false),
        spv::BuiltIn::PointSize));
    mesh.topology = 1;
    const std::vector<u32> lines{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false)};
    Dump(lines, "fallback-vertex-lines");
    REQUIRE_FALSE(HasBuiltIn(lines, spv::BuiltIn::PointSize));
    REQUIRE(HasBuiltIn(lines, spv::BuiltIn::ClipDistance));
    mesh.topology = 0;
    const std::vector<u32> points{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ARGS_BINDING, false,
                                       2.0f)};
    Dump(points, "fallback-vertex-points");
    REQUIRE(Model(points) == spv::ExecutionModel::Vertex);
    REQUIRE(HasBuiltIn(points, spv::BuiltIn::PointSize));
    REQUIRE(HasBuiltIn(points, spv::BuiltIn::ClipDistance));
}

TEST_CASE("Mesh fallback distribution pass binds its buffers and constants",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> code{SPIRV::EmitMeshEmulationDistribution(MakeProfile())};
    Dump(code, "fallback-distribution");
    REQUIRE(Model(code) == spv::ExecutionModel::GLCompute);
    REQUIRE(Bindings(code) == std::vector<u32>{0, 1, 2, 3});
    REQUIRE(UsesStorageClass(code, spv::StorageClass::PushConstant));
    const auto instructions{Instructions(code)};
    const bool has_constants{std::ranges::any_of(instructions, [](const Instruction& inst) {
        return inst.op == spv::Op::OpTypeStruct &&
               inst.operands.size() == 1 + sizeof(SPIRV::MeshEmulationDistributionConstants) / 4;
    })};
    REQUIRE(has_constants);
}

TEST_CASE("Mesh fallback reserves auxiliary storage buffers in the descriptor set limit",
          "[video_core][mesh_fallback]") {
    using Vulkan::FitsStorageBufferSetLimit;

    // Ordinary graphics pipelines do not add emulation descriptors.
    REQUIRE(FitsStorageBufferSetLimit(8, 0, 8));
    REQUIRE_FALSE(FitsStorageBufferSetLimit(9, 0, 8));

    // Mesh only: one output buffer. Task+mesh: four auxiliary bindings.
    REQUIRE(FitsStorageBufferSetLimit(7, 1, 8));
    REQUIRE_FALSE(FitsStorageBufferSetLimit(8, 1, 8));
    REQUIRE(FitsStorageBufferSetLimit(4, 4, 8));
    REQUIRE_FALSE(FitsStorageBufferSetLimit(5, 4, 8));

    // No limit is represented by zero. Reject huge inputs without addition overflow.
    REQUIRE(FitsStorageBufferSetLimit(200, 4, 0));
    REQUIRE_FALSE(FitsStorageBufferSetLimit((std::numeric_limits<size_t>::max)(), 4, 8));
}
