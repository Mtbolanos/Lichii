// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace {
using namespace Shader;

struct TensorMode {
    std::string previous;
    bool existed{};

    explicit TensorMode(const char* mode) {
        if (const char* value = std::getenv("EDEN_NX2_TENSOR")) {
            previous = value;
            existed = true;
        }
        Set(mode);
    }

    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("EDEN_NX2_TENSOR", value ? value : "");
#else
        if (value) {
            setenv("EDEN_NX2_TENSOR", value, 1);
        } else {
            unsetenv("EDEN_NX2_TENSOR");
        }
#endif
    }

    ~TensorMode() {
        Set(existed ? previous.c_str() : nullptr);
    }
};

struct TensorProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile;

    explicit TensorProgram(u32 count = 1, u32 software_depth = 0) {
        program.stage = Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        program.info.uses_subgroup_shuffles = count != 0 && software_depth != 0;
        program.info.uses_subgroup_invocation_id = count != 0;
        profile.supported_spirv = 0x00010500;
        profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
        profile.max_subgroup_size = 32;
        IR::IREmitter ir{block};
        if (count == 0) {
            return;
        }
        const auto lane = ir.LaneId();
        std::array<IR::U32, 8> inputs;
        for (u32 i = 0; i < inputs.size(); ++i) {
            inputs[i] = software_depth ? ir.IAdd(ir.Imm32(0x3c003c00U + i * 0x01000100U),
                ir.ShiftLeftLogical(lane, ir.Imm32(3))) : ir.IAdd(ir.Imm32((i + 1) * 0x01000100U), lane);
        }
        for (u32 i = 0; i < count; ++i) {
            const auto result = ir.TensorMulAdd(
                                       ir.CompositeConstruct(inputs[0], inputs[1], inputs[2], inputs[3]),
                                       ir.CompositeConstruct(inputs[4], inputs[5]),
                                       ir.CompositeConstruct(inputs[6], inputs[7]), software_depth);
            inputs[6] = IR::U32{ir.CompositeExtract(result, 0)};
            inputs[7] = IR::U32{ir.CompositeExtract(result, 1)};
        }
        if (software_depth) {
            program.info.used_storage_buffer_types = IR::Type::U32;
            program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
            program.info.uses_local_invocation_id = true;
            const auto offset = ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(3));
            block.AppendNewInst(IR::Opcode::WriteStorage32, {ir.Imm32(0), offset, inputs[6]});
            block.AppendNewInst(IR::Opcode::WriteStorage32,
                {ir.Imm32(0), ir.IAdd(offset, ir.Imm32(4)), inputs[7]});
        }
    }

    std::vector<u32> Emit(std::string_view name = {}) {
        auto code = Backend::SPIRV::EmitSPIRV(profile, program);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
            directory && !name.empty()) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
            REQUIRE(file.good());
        }
        return code;
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        REQUIRE(offset + size <= code.size());
        if ((code[offset] & 0xffffU) == opcode) {
            result.emplace_back(code.begin() + offset + 1, code.begin() + offset + size);
        }
        offset += size;
    }
    return result;
}

bool HasCapability(const std::vector<u32>& code, u32 capability) {
    const auto capabilities = Instructions(code, 17);
    return std::ranges::any_of(capabilities, [=](const auto& words) {
        return words[0] == capability;
    });
}

namespace Op {
constexpr u32 ExtInst = 12, Constant = 43, Function = 54, FunctionParameter = 55, Load = 61,
              Store = 62, CompositeExtract = 81, Select = 169, IEqual = 170, ULessThan = 176,
              LoopMerge = 246, Label = 248, GroupNonUniformShuffle = 345;
constexpr u32 RoundEven = 2, Fma = 50;
} // namespace Op

/// The instructions of a module in order, with each result's defining instruction.
struct Module {
    struct Inst {
        u32 opcode;
        std::vector<u32> words;
    };
    std::vector<Inst> insts;
    std::map<u32, size_t> defs;

    explicit Module(const std::vector<u32>& code) {
        for (size_t offset = 5; offset < code.size();) {
            const u32 size = code[offset] >> 16;
            REQUIRE(size != 0);
            const u32 opcode = code[offset] & 0xffffU;
            insts.push_back({opcode, {code.begin() + offset + 1, code.begin() + offset + size}});
            offset += size;
        }
        // OpTypeVoid..OpTypeFunction declare types; an instruction whose first operand is a type
        // defines a typed result next. Debug, mode and decoration instructions (whose first
        // operand can equal a type id) are skipped.
        std::vector<u32> types;
        for (const auto& inst : insts) {
            if (inst.opcode >= 19 && inst.opcode <= 33) {
                types.push_back(inst.words[0]);
            }
        }
        for (size_t index = 0; index < insts.size(); ++index) {
            const auto& inst = insts[index];
            const bool annotation =
                (inst.opcode >= 2 && inst.opcode <= 33 && inst.opcode != Op::ExtInst) ||
                inst.opcode == 71 || inst.opcode == 72 || inst.opcode == 332 ||
                inst.opcode == 5632;
            if (inst.opcode == Op::Label) {
                defs.emplace(inst.words[0], index);
            } else if (!annotation && inst.words.size() >= 2 &&
                       std::ranges::find(types, inst.words[0]) != types.end()) {
                defs.emplace(inst.words[1], index);
            }
        }
    }

    const Inst* Def(u32 id) const {
        const auto it = defs.find(id);
        return it == defs.end() ? nullptr : &insts[it->second];
    }

    /// Ids an instruction reads (literal operands skipped where the opcode has them).
    static std::vector<u32> Operands(const Inst& inst) {
        switch (inst.opcode) {
        case Op::Constant:
        case Op::FunctionParameter:
        case 59: // OpVariable
            return {};
        case Op::CompositeExtract:
            return {inst.words[2]};
        case Op::ExtInst:
            return {inst.words.begin() + 4, inst.words.end()};
        default:
            return {inst.words.begin() + 2, inst.words.end()};
        }
    }

    bool IsExtInst(const Inst& inst, u32 number) const {
        return inst.opcode == Op::ExtInst && inst.words[3] == number;
    }

    /// Whether `id` is computed from an OpExtInst `number`, through values only (not memory).
    bool DependsOn(u32 id, u32 number) const {
        std::vector<u32> pending{id};
        std::vector<u32> seen;
        while (!pending.empty()) {
            const u32 current = pending.back();
            pending.pop_back();
            if (std::ranges::find(seen, current) != seen.end()) {
                continue;
            }
            seen.push_back(current);
            const Inst* const inst = Def(current);
            if (!inst || inst->opcode == Op::Load) {
                continue;
            }
            if (IsExtInst(*inst, number)) {
                return true;
            }
            for (const u32 operand : Operands(*inst)) {
                pending.push_back(operand);
            }
        }
        return false;
    }

    size_t IndexOf(const Inst& inst) const {
        return static_cast<size_t>(&inst - insts.data());
    }
};

/// Where a software matrix operand comes from: parameter (0 = A, 1 = B) and component.
std::pair<u32, u32> OperandSource(const Module& module, u32 id,
                                  const std::vector<u32>& parameters) {
    const auto* const extract = module.Def(id);
    REQUIRE(extract);
    REQUIRE(extract->opcode == Op::CompositeExtract);
    const auto parameter = std::ranges::find(parameters, extract->words[2]);
    REQUIRE(parameter != parameters.end());
    return {static_cast<u32>(parameter - parameters.begin()), extract->words[3]};
}

}

TEST_CASE("Software HMMA shares one helper per depth without tensor extensions",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram single{1, depth};
        const auto first = single.Emit(depth == 8 ? "software-k8" : "software-k16");
        REQUIRE_FALSE(HasCapability(first, 5357));
        REQUIRE(Instructions(first, 57).size() == 1); // OpFunctionCall
        TensorProgram repeated{64, depth};
        const auto many = repeated.Emit(depth == 8 ? "software-k8-many" : "software-k16-many");
        REQUIRE(Instructions(many, 57).size() == 64);
        // Three fragment-vector constructions, the call and two result extracts
        // take at most 40 words per site; the matrix body itself is never duplicated.
        REQUIRE(many.size() < first.size() + 64 * 40);
    }
}

TEST_CASE("Software HMMA rounds once per uniform K8 group", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        const auto code = fixture.Emit();
        std::map<u32, u32> constants;
        for (const auto& words : Instructions(code, 43)) { // OpConstant
            constants.emplace(words[1], words[2]);
        }
        // One K=8 group is emitted once; K=16 runs it twice from a uniform counter.
        const auto loops = Instructions(code, 246); // OpLoopMerge
        if (depth == 8) {
            REQUIRE(loops.empty());
        } else {
            REQUIRE(loops.size() == 1);
            const auto limits = Instructions(code, 176); // OpULessThan
            REQUIRE(limits.size() == 1);
            REQUIRE(constants.at(limits.front()[3]) == 2);
            const auto loads = Instructions(code, 61); // OpLoad
            const auto counter_load = std::ranges::find_if(
                loads, [&](const auto& words) { return words[1] == limits.front()[2]; });
            REQUIRE(counter_load != loads.end());
            const u32 counter = (*counter_load)[2];
            u32 counter_stores = 0;
            for (const auto& store : Instructions(code, 62)) { // OpStore
                if (store[0] != counter) {
                    continue;
                }
                ++counter_stores;
                if (constants.contains(store[1])) {
                    REQUIRE(constants.at(store[1]) == 0);
                }
            }
            REQUIRE(counter_stores == 2); // Zero, then the increment: no lane input.
        }
        u32 rounding_operations = 0;
        u32 fma_operations = 0;
        for (const auto& words : Instructions(code, 12)) { // OpExtInst
            rounding_operations += words[3] == 2;          // GLSL.std.450 RoundEven
            fma_operations += words[3] == 50;              // GLSL.std.450 Fma
        }
        REQUIRE(rounding_operations == 4); // One per accumulator, after each K=8 group.
        REQUIRE(fma_operations == 32);
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
        REQUIRE(Instructions(code, 247).empty());      // No per-pair rounding branch.
    }
}

TEST_CASE("Software HMMA stores each group's rounded accumulators and reads the right halves",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        CAPTURE(depth);
        TensorProgram fixture{1, depth};
        const Module module{fixture.Emit()};
        const auto& insts = module.insts;

        // The helper: the DontInline function with three parameters (A, B, C).
        const auto function = std::ranges::find_if(insts, [](const Module::Inst& inst) {
            return inst.opcode == Op::Function && (inst.words[2] & 0x2) != 0;
        });
        REQUIRE(function != insts.end());
        std::vector<u32> parameters;
        for (auto it = function + 1; it != insts.end() && it->opcode == Op::FunctionParameter;
             ++it) {
            parameters.push_back(it->words[1]);
        }
        REQUIRE(parameters.size() == 3);

        // The group body: the loop body for K=16, the whole helper for K=8.
        size_t body_begin = module.IndexOf(*function);
        size_t body_end = insts.size();
        const auto loop = std::ranges::find_if(
            insts, [](const Module::Inst& inst) { return inst.opcode == Op::LoopMerge; });
        u32 group = 0;
        if (depth == 16) {
            REQUIRE(loop != insts.end());
            const u32 continue_label = loop->words[1];
            body_begin = module.IndexOf(*loop);
            body_end = module.defs.at(continue_label);
            REQUIRE(body_begin < body_end);
            const auto limit = std::ranges::find_if(
                insts, [](const Module::Inst& inst) { return inst.opcode == Op::ULessThan; });
            REQUIRE(limit != insts.end());
            group = limit->words[2];
            REQUIRE(module.Def(group)->opcode == Op::Load);
        } else {
            REQUIRE(loop == insts.end());
        }
        const auto in_body = [&](size_t index) { return index > body_begin && index < body_end; };

        size_t last_fma = 0;
        u32 roundings = 0, fmas = 0, shuffles = 0;
        std::vector<std::pair<std::pair<u32, u32>, std::pair<u32, u32>>> halves;
        for (size_t index = 0; index < insts.size(); ++index) {
            const auto& inst = insts[index];
            if (module.IsExtInst(inst, Op::RoundEven)) {
                CHECK(in_body(index));
                ++roundings;
            } else if (module.IsExtInst(inst, Op::Fma)) {
                CHECK(in_body(index));
                last_fma = (std::max)(last_fma, index);
                ++fmas;
            } else if (inst.opcode == Op::GroupNonUniformShuffle) {
                CHECK(in_body(index));
                ++shuffles;
                const u32 value = inst.words[3];
                if (depth == 8) {
                    // One group: always the low halves.
                    const auto source = OperandSource(module, value, parameters);
                    halves.push_back({source, source});
                    continue;
                }
                // K=16: the group counter picks the high half in the second iteration.
                const auto* const select = module.Def(value);
                REQUIRE(select);
                REQUIRE(select->opcode == Op::Select);
                const auto* const condition = module.Def(select->words[2]);
                REQUIRE(condition);
                REQUIRE(condition->opcode == Op::IEqual);
                const u32 compared = condition->words[2] == group ? condition->words[3]
                                                                  : condition->words[2];
                CHECK((condition->words[2] == group || condition->words[3] == group));
                const auto* const one = module.Def(compared);
                REQUIRE(one);
                REQUIRE(one->opcode == Op::Constant);
                CHECK(one->words[2] == 1);
                halves.push_back({OperandSource(module, select->words[3], parameters),
                                  OperandSource(module, select->words[4], parameters)});
            }
        }
        CHECK(roundings == 4);
        CHECK(fmas == 32);
        CHECK(shuffles == 16);

        // A shuffles read A[row] then A[row + 2]; B shuffles read B[0] then B[1].
        using Source = std::pair<u32, u32>;
        const std::vector<std::pair<Source, Source>> allowed =
            depth == 8 ? std::vector<std::pair<Source, Source>>{{{0, 0}, {0, 0}},
                                                                {{0, 1}, {0, 1}},
                                                                {{1, 0}, {1, 0}}}
                       : std::vector<std::pair<Source, Source>>{{{0, 2}, {0, 0}},
                                                                {{0, 3}, {0, 1}},
                                                                {{1, 1}, {1, 0}}};
        for (const auto& pair : halves) {
            CHECK(std::ranges::find(allowed, pair) != allowed.end());
        }
        for (const auto& expected : allowed) {
            CHECK(std::ranges::count(halves, expected) == (expected.first.first == 0 ? 4 : 8));
        }

        // After the group's last FMA, every accumulator is stored rounded inside the body.
        std::vector<u32> rounded_targets;
        std::vector<u32> fma_targets;
        for (size_t index = body_begin + 1; index < body_end; ++index) {
            const auto& inst = insts[index];
            if (inst.opcode != Op::Store) {
                continue;
            }
            if (module.DependsOn(inst.words[1], Op::RoundEven)) {
                CHECK(index > last_fma);
                rounded_targets.push_back(inst.words[0]);
            } else if (module.DependsOn(inst.words[1], Op::Fma)) {
                fma_targets.push_back(inst.words[0]);
            }
        }
        std::ranges::sort(rounded_targets);
        CHECK(std::ranges::adjacent_find(rounded_targets) == rounded_targets.end());
        CHECK(rounded_targets.size() == 4);
        for (const u32 target : rounded_targets) {
            CHECK(std::ranges::find(fma_targets, target) != fma_targets.end());
        }
    }
}

TEST_CASE("Software matrix helpers also emit for 64-lane host subgroups",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        fixture.profile.max_subgroup_size = 64;
        const auto code = fixture.Emit(depth == 8 ? "software-k8-wave64" : "software-k16-wave64");
        REQUIRE_FALSE(HasCapability(code, 5357));
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
    }
}

namespace {

constexpr u32 OP_COMPOSITE_EXTRACT = 81;
constexpr u32 OP_COMPOSITE_INSERT = 82;
constexpr u32 OP_BITCAST = 124;
constexpr u32 OP_TYPE_COOPERATIVE_MATRIX = 4456;
constexpr u32 OP_COOPERATIVE_MATRIX_LOAD = 4457;
constexpr u32 OP_COOPERATIVE_MATRIX_STORE = 4458;
constexpr u32 OP_COOPERATIVE_MATRIX_MUL_ADD = 4459;
constexpr u32 CAPABILITY_COOPERATIVE_MATRIX = 6022;
constexpr u32 CAPABILITY_FLOAT16 = 9;
constexpr u32 CAPABILITY_VULKAN_MEMORY_MODEL = 5345;
constexpr u32 CAPABILITY_VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 5346;

std::map<u32, u32> Constants(const std::vector<u32>& code) {
    std::map<u32, u32> constants;
    for (const auto& words : Instructions(code, 43)) { // OpConstant
        constants.emplace(words[1], words[2]);
    }
    return constants;
}

void RequireHmmaMatrixTypes(const std::vector<u32>& code, u32 columns = 8) {
    const auto constants = Constants(code);
    const auto types = Instructions(code, OP_TYPE_COOPERATIVE_MATRIX);
    REQUIRE(types.size() == 3);
    std::array<u32, 3> columns_by_use{};
    for (const auto& words : types) {
        REQUIRE(words.size() == 6);
        REQUIRE(constants.at(words[2]) == 3); // Subgroup scope
        REQUIRE(constants.at(words[3]) == 16);
        columns_by_use.at(constants.at(words[5])) = constants.at(words[4]);
    }
    REQUIRE(columns_by_use == std::array<u32, 3>{16, columns, columns}); // A, B, accumulator
}

// SPV_KHR_cooperative_matrix requires the Vulkan memory model, which spirv-val does not check.
void RequireVulkanMemoryModel(const std::vector<u32>& code) {
    REQUIRE(HasCapability(code, CAPABILITY_VULKAN_MEMORY_MODEL));
    REQUIRE(HasCapability(code, CAPABILITY_VULKAN_MEMORY_MODEL_DEVICE_SCOPE));
    const auto models = Instructions(code, 14); // OpMemoryModel
    REQUIRE(models.size() == 1);
    REQUIRE(models[0][1] == 3); // Vulkan
    bool extension = false;
    for (const auto& words : Instructions(code, 10)) { // OpExtension
        extension |= std::string{reinterpret_cast<const char*>(words.data())} ==
                     "SPV_KHR_vulkan_memory_model";
    }
    REQUIRE(extension == (code[1] < 0x00010500));
}

struct WorkgroupMemoryOperands {
    u32 loads{};
    u32 stores{};
    u32 cooperative_loads{};
    u32 cooperative_stores{};
    u32 barriers{};
};

// Under the Vulkan memory model a plain access is private and can't carry data between
// invocations, so every workgroup memory access must be non-private and make itself available or
// visible to the workgroup, and every barrier must make memory available and visible. A GLSL450
// module carries none of these operands.
WorkgroupMemoryOperands RequireWorkgroupMemoryOperands(const std::vector<u32>& code,
                                                       bool vulkan_memory_model) {
    constexpr u32 STORAGE_CLASS_WORKGROUP = 4;
    constexpr u32 SCOPE_WORKGROUP = 2;
    constexpr u32 MAKE_POINTER_AVAILABLE = 0x8, MAKE_POINTER_VISIBLE = 0x10,
                  NON_PRIVATE_POINTER = 0x20;
    constexpr u32 MAKE_AVAILABLE = 0x2000, MAKE_VISIBLE = 0x4000;
    const auto constants = Constants(code);
    std::set<u32> pointer_types;
    for (const auto& words : Instructions(code, 32)) { // OpTypePointer
        if (words[1] == STORAGE_CLASS_WORKGROUP) {
            pointer_types.insert(words[0]);
        }
    }
    std::set<u32> pointers;
    for (const u32 opcode : {59U, 65U}) { // OpVariable, OpAccessChain
        for (const auto& words : Instructions(code, opcode)) {
            if (pointer_types.contains(words[0])) {
                pointers.insert(words[1]);
            }
        }
    }
    WorkgroupMemoryOperands result;
    struct Access {
        u32 opcode;
        size_t pointer;
        size_t operands;
        u32 availability;
        u32 WorkgroupMemoryOperands::*count;
    };
    for (const Access& access : {
             Access{Op::Load, 2, 3, MAKE_POINTER_VISIBLE, &WorkgroupMemoryOperands::loads},
             Access{Op::Store, 0, 2, MAKE_POINTER_AVAILABLE, &WorkgroupMemoryOperands::stores},
             Access{OP_COOPERATIVE_MATRIX_LOAD, 2, 5, MAKE_POINTER_VISIBLE,
                    &WorkgroupMemoryOperands::cooperative_loads},
             Access{OP_COOPERATIVE_MATRIX_STORE, 0, 4, MAKE_POINTER_AVAILABLE,
                    &WorkgroupMemoryOperands::cooperative_stores}}) {
        for (const auto& words : Instructions(code, access.opcode)) {
            if (!pointers.contains(words[access.pointer])) {
                continue;
            }
            ++(result.*access.count);
            if (!vulkan_memory_model) {
                REQUIRE(words.size() == access.operands);
                continue;
            }
            REQUIRE(words.size() == access.operands + 2);
            REQUIRE(words[access.operands] == (access.availability | NON_PRIVATE_POINTER));
            REQUIRE(constants.at(words[access.operands + 1]) == SCOPE_WORKGROUP);
        }
    }
    const u32 expected_semantics = vulkan_memory_model ? MAKE_AVAILABLE | MAKE_VISIBLE : 0;
    for (const auto& [opcode, semantics] : {std::pair{224U, 2U}, std::pair{225U, 1U}}) {
        for (const auto& words : Instructions(code, opcode)) { // OpControlBarrier, OpMemoryBarrier
            REQUIRE((constants.at(words[semantics]) & (MAKE_AVAILABLE | MAKE_VISIBLE)) ==
                    expected_semantics);
            ++result.barriers;
        }
    }
    return result;
}

} // namespace

TEST_CASE("Cooperative HMMA replaces the K16 software helper", "[video_core][tensor][ampere]") {
    TensorProgram fixture{3, 16};
    fixture.profile.support_cooperative_hmma = true;
    const auto code = fixture.Emit("cooperative-k16");
    REQUIRE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
    REQUIRE(HasCapability(code, CAPABILITY_FLOAT16));
    RequireVulkanMemoryModel(code);
    REQUIRE(fixture.program.info.uses_cooperative_matrix);
    RequireHmmaMatrixTypes(code);
    REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 3);
    REQUIRE(Instructions(code, 246).empty()); // OpLoopMerge
    REQUIRE(Instructions(code, 57).empty());  // OpFunctionCall
    REQUIRE(Instructions(code, 345).empty()); // OpGroupNonUniformShuffle
    for (const auto& words : Instructions(code, 169)) { // OpSelect
        for (const auto& type : Instructions(code, OP_TYPE_COOPERATIVE_MATRIX)) {
            REQUIRE(words[0] != type[0]);
        }
    }
}

TEST_CASE("Cooperative HMMA keeps barriers valid under the Vulkan memory model",
          "[video_core][tensor][ampere]") {
    for (const u32 version : {0x00010400U, 0x00010600U}) {
        TensorProgram fixture{1, 16};
        fixture.profile.supported_spirv = version;
        fixture.profile.support_cooperative_hmma = true;
        IR::IREmitter ir{fixture.block};
        ir.Barrier();
        ir.WorkgroupMemoryBarrier();
        ir.DeviceMemoryBarrier();
        const auto code = fixture.Emit("cooperative-k16-barriers-" + std::to_string(version));
        REQUIRE(code[1] == version);
        RequireVulkanMemoryModel(code);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 1);
    }
}

TEST_CASE("Cooperative HMMA keeps workgroup memory coherent under the Vulkan memory model",
          "[video_core][tensor][ampere]") {
    const auto emit = [](std::string_view variant, auto configure) {
        TensorProgram fixture{1, 16};
        fixture.profile.supported_spirv = 0x00010600;
        fixture.program.shared_memory_size = 64;
        configure(fixture.profile);
        IR::IREmitter ir{fixture.block};
        ir.WriteShared(32, ir.Imm32(4), ir.LaneId());
        ir.Barrier();
        static_cast<void>(ir.LoadShared(32, false, ir.Imm32(8)));
        ir.WorkgroupMemoryBarrier();
        ir.DeviceMemoryBarrier();
        return fixture.Emit("workgroup-memory-" + std::string{variant});
    };
    SECTION("16x8x16") {
        const auto code = emit("16x8x16", [](Profile& profile) {
            profile.support_cooperative_hmma = true;
        });
        RequireVulkanMemoryModel(code);
        const auto operands = RequireWorkgroupMemoryOperands(code, true);
        REQUIRE(operands.loads == 1);
        REQUIRE(operands.stores == 1);
        REQUIRE(operands.barriers == 3);
    }
    SECTION("16x16x16") {
        const auto code = emit("16x16x16", [](Profile& profile) {
            profile.support_cooperative_hmma_16x16x16 = true;
        });
        RequireVulkanMemoryModel(code);
        const auto operands = RequireWorkgroupMemoryOperands(code, true);
        REQUIRE(operands.loads == 3);   // The guest's load and the lane's two D words.
        REQUIRE(operands.stores == 11); // The guest's store and the lane's A, B and C words.
        REQUIRE(operands.cooperative_loads == 3);
        REQUIRE(operands.cooperative_stores == 1);
        REQUIRE(operands.barriers == 5);
    }
    SECTION("software HMMA on GLSL450") {
        const auto code = emit("software", [](Profile&) {});
        REQUIRE_FALSE(HasCapability(code, CAPABILITY_VULKAN_MEMORY_MODEL));
        const auto operands = RequireWorkgroupMemoryOperands(code, false);
        REQUIRE(operands.loads == 1);
        REQUIRE(operands.stores == 1);
        REQUIRE(operands.barriers == 3);
    }
}

TEST_CASE("Cooperative HMMA keeps the driver's order of A's registers",
          "[video_core][tensor][ampere]") {
    TensorProgram fixture{1, 16};
    fixture.profile.support_cooperative_hmma = true;
    const auto code = fixture.Emit();
    const auto extracts = Instructions(code, OP_COMPOSITE_EXTRACT);
    const auto bitcasts = Instructions(code, OP_BITCAST);
    const auto by_result = [](const auto& instructions, u32 result) {
        const auto it = std::ranges::find_if(
            instructions, [=](const auto& words) { return words[1] == result; });
        REQUIRE(it != instructions.end());
        return *it;
    };
    const auto types = Instructions(code, OP_TYPE_COOPERATIVE_MATRIX);
    const auto constants = Constants(code);
    const auto matrix_a = std::ranges::find_if(
        types, [&](const auto& words) { return constants.at(words[5]) == 0; });
    REQUIRE(matrix_a != types.end());

    std::array<u32, 8> register_of_element{};
    u32 inserts = 0;
    for (const auto& words : Instructions(code, OP_COMPOSITE_INSERT)) {
        if (words[0] != (*matrix_a)[0]) {
            continue;
        }
        const auto half = by_result(extracts, words[2]);
        const auto halves = by_result(bitcasts, half[2]);
        const auto guest_register = by_result(extracts, halves[2]);
        REQUIRE(half[3] == (words[4] & 1));
        register_of_element.at(words[4]) = guest_register[3];
        ++inserts;
    }
    REQUIRE(inserts == 8);
    REQUIRE(register_of_element == std::array<u32, 8>{0, 0, 2, 2, 1, 1, 3, 3});
}

TEST_CASE("Cooperative HMMA falls back to the software helper", "[video_core][tensor][ampere]") {
    const auto require_software = [](TensorProgram& fixture) {
        const auto code = fixture.Emit();
        REQUIRE_FALSE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        REQUIRE(Instructions(code, OP_TYPE_COOPERATIVE_MATRIX).empty());
        REQUIRE(Instructions(code, 57).size() == 1); // OpFunctionCall
        REQUIRE_FALSE(fixture.program.info.uses_cooperative_matrix);
    };
    SECTION("unsupported by the device") {
        TensorProgram fixture{1, 16};
        require_software(fixture);
    }
    SECTION("the K8 form") {
        TensorProgram fixture{1, 8};
        fixture.profile.support_cooperative_hmma = true;
        require_software(fixture);
    }
    SECTION("a workgroup that leaves a warp unfilled") {
        TensorProgram fixture{1, 16};
        fixture.profile.support_cooperative_hmma = true;
        fixture.program.workgroup_size = {48, 1, 1};
        require_software(fixture);
    }
    SECTION("a workgroup row that leaves a warp unfilled") {
        TensorProgram fixture{1, 16};
        fixture.profile.support_cooperative_hmma = true;
        fixture.program.workgroup_size = {16, 2, 1};
        require_software(fixture);
    }
    SECTION("coherent memory, which the Vulkan memory model has no decoration for") {
        TensorProgram fixture{1, 16};
        fixture.profile.support_cooperative_hmma = true;
        fixture.program.info.requires_coherent_global_memory = true;
        require_software(fixture);
    }
}

TEST_CASE("Cooperative HMMA self-test shader runs one lowering on one warp",
          "[video_core][tensor][ampere]") {
    for (const u32 version : {0x00010300U, 0x00010600U}) {
        const auto code = Backend::SPIRV::EmitCooperativeHmmaTest(version);
        REQUIRE(code[1] == version);
        REQUIRE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        RequireVulkanMemoryModel(code);
        RequireHmmaMatrixTypes(code);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 1);
        const auto modes = Instructions(code, 16); // OpExecutionMode
        REQUIRE(modes.size() == 1);
        REQUIRE(modes[0][1] == 17); // LocalSize
        REQUIRE(modes[0][2] == 32);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT")) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} /
                                   ("cooperative-self-test-" + std::to_string(version) + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
        }
    }
}

TEST_CASE("16x16x16 cooperative HMMA runs both depths through shared memory",
          "[video_core][tensor][ampere]") {
    for (const auto& [depth, version] : {std::pair{16U, 0x00010500U}, std::pair{16U, 0x00010600U},
                                        std::pair{8U, 0x00010600U}}) {
        TensorProgram fixture{3, depth};
        fixture.profile.supported_spirv = version;
        fixture.profile.support_cooperative_hmma_16x16x16 = true;
        const auto code = fixture.Emit("cooperative-16x16x16-k" + std::to_string(depth) + "-" +
                                       std::to_string(version));
        REQUIRE(code[1] == version);
        REQUIRE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        REQUIRE(HasCapability(code, CAPABILITY_FLOAT16));
        RequireVulkanMemoryModel(code);
        RequireHmmaMatrixTypes(code, 16);
        REQUIRE(fixture.program.info.uses_cooperative_matrix);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_LOAD).size() == 9);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 3);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_STORE).size() == 3);
        REQUIRE(Instructions(code, 202).empty()); // OpBitFieldUExtract
        REQUIRE(Instructions(code, 57).empty());  // OpFunctionCall
        REQUIRE(Instructions(code, 345).empty()); // OpGroupNonUniformShuffle
        REQUIRE(fixture.program.shared_memory_size == 2 * 384 * 4);
    }
}

TEST_CASE("16x16x16 cooperative HMMA falls back to the software helper",
          "[video_core][tensor][ampere]") {
    TensorProgram fixture{1, 16};
    fixture.profile.support_cooperative_hmma_16x16x16 = true;
    SECTION("a workgroup that leaves a warp unfilled") {
        fixture.program.workgroup_size = {48, 1, 1};
    }
    SECTION("a workgroup row that leaves a warp unfilled") {
        fixture.program.workgroup_size = {16, 2, 1};
    }
    SECTION("coherent memory, which the Vulkan memory model has no decoration for") {
        fixture.program.info.requires_coherent_image_memory = true;
    }
    SECTION("scratch past the shared memory limit") {
        fixture.profile.max_compute_shared_memory_size = 2 * 384 * 4 - 4;
    }
    const auto code = fixture.Emit();
    REQUIRE_FALSE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
    REQUIRE(Instructions(code, 57).size() == 1); // OpFunctionCall
    REQUIRE(fixture.program.shared_memory_size == 0);
    REQUIRE_FALSE(fixture.program.info.uses_cooperative_matrix);
}

TEST_CASE("Captured Bananza matrix shaders lower without an environment override",
          "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    for (const Ampere::Instruction instruction : {
             Ampere::Instruction{{0x102c723c, 0x30, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1020723c, 0x24, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1418723c, 0x1c, 0x8ff, 0x04aff000}}}) {
        const std::array code{instruction,
                              Ampere::Instruction{{0x794d, 0, 0x03800000, 0}}};
        const auto lowered = Ampere::LowerToMaxwell(code, false, true);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(Maxwell::Decode(lowered.instructions[0]) == Maxwell::Opcode::TENSOR_MUL_ADD);
    }
}

TEST_CASE("Native matrix lowering selects compact software helpers without legacy toggles", "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    std::array<Ampere::Instruction, 2> code{{
        {{0x0810723c, 12, 0x8ff, 0}}, {{0x0000794d, 0, 0x03800000, 0}}}};
    const auto supported = Ampere::LowerToMaxwell(code, false, false);
    REQUIRE(static_cast<bool>(supported));
    REQUIRE(supported.instructions.size() > 100);
    const auto compact = Ampere::LowerToMaxwell(code, false, true);
    REQUIRE(compact);
    REQUIRE(compact.instructions.size() == 2);
    REQUIRE(Maxwell::Decode(compact.instructions.front()) == Maxwell::Opcode::TENSOR_MUL_ADD);
    REQUIRE((compact.instructions.front() & (1ULL << 48)) != 0);
    for (const char* legacy : {"fast", "stub", "off"}) {
        TensorMode::Set(legacy);
        REQUIRE(Ampere::LowerToMaxwell(code, false, true).instructions == compact.instructions);
        REQUIRE(Ampere::LowerToMaxwell(code).instructions == supported.instructions);
    }
    code[0].words[0] &= ~0x7000U;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[0] |= 0x7000;
    code[0].words[3] = 1;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[3] = 0;
    code[0].words[0] += 1U << 24;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
}
