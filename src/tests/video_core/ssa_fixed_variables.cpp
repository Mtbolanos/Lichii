// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
namespace IR = Shader::IR;

using Definitions = std::vector<std::pair<const IR::Block*, IR::Value>>;

/// The definitions a read resolves to after the SSA pass: one per predecessor for a phi.
Definitions Resolved(const IR::Value& read, const IR::Block* block) {
    const IR::Value value{read.Resolve()};
    if (value.IsImmediate() || value.InstRecursive()->GetOpcode() != IR::Opcode::Phi) {
        return {{block, value}};
    }
    const IR::Inst* const phi{value.InstRecursive()};
    Definitions result;
    for (size_t arg = 0; arg < phi->NumArgs(); ++arg) {
        result.emplace_back(phi->PhiBlock(arg), phi->Arg(arg).Resolve());
    }
    return result;
}

IR::U1 Condition(IR::IREmitter& ir, u32 value) {
    return ir.ILessThan(ir.LocalInvocationIdX(), ir.Imm32(value), false);
}

/// Values of every fixed-count variable kind written by one block.
struct Writes {
    IR::U1 pred, zero, sign, carry, overflow, go_to;
    IR::U32 indirect;

    Writes(IR::IREmitter& ir, u32 seed)
        : pred{Condition(ir, seed)}, zero{Condition(ir, seed + 1)}, sign{Condition(ir, seed + 2)},
          carry{Condition(ir, seed + 3)}, overflow{Condition(ir, seed + 4)},
          go_to{Condition(ir, seed + 5)},
          indirect{ir.IAdd(ir.LocalInvocationIdX(), ir.Imm32(seed + 6))} {
        ir.SetPred(IR::Pred::P2, pred);
        ir.SetZFlag(zero);
        ir.SetSFlag(sign);
        ir.SetCFlag(carry);
        ir.SetOFlag(overflow);
        ir.SetGotoVariable(3, go_to);
        ir.SetIndirectBranchVariable(indirect);
    }
};

/// Reads of every fixed-count variable kind in one block.
struct Reads {
    IR::Value pred, zero, sign, carry, overflow, go_to, indirect;

    explicit Reads(IR::IREmitter& ir)
        : pred{ir.GetPred(IR::Pred::P2)}, zero{ir.GetZFlag()}, sign{ir.GetSFlag()},
          carry{ir.GetCFlag()}, overflow{ir.GetOFlag()}, go_to{ir.GetGotoVariable(3)},
          indirect{ir.GetIndirectBranchVariable()} {
        // The pass drops identities nothing uses, so each read gets a use.
        for (const IR::Value& value : {pred, zero, sign, carry, overflow, go_to}) {
            static_cast<void>(ir.LogicalNot(IR::U1{value}));
        }
        static_cast<void>(ir.BitwiseNot(IR::U32{indirect}));
    }
};

/// Runs the SSA pass over blocks given in program order, visited in reverse post order.
void Run(const IR::BlockList& blocks, IR::BlockList post_order) {
    for (u32 index = 0; index < blocks.size(); ++index) {
        blocks[index]->SetOrder(index);
    }
    IR::Program program;
    program.blocks = blocks;
    program.post_order_blocks = std::move(post_order);
    Shader::Optimization::SsaRewritePass(program);
}
} // namespace

TEST_CASE("SSA reads of predicates, flags and branch variables merge their definitions",
          "[video_core][shader][ssa]") {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block entry{pool}, left{pool}, right{pool}, merge{pool};
    entry.AddBranch(&left);
    entry.AddBranch(&right);
    left.AddBranch(&merge);
    right.AddBranch(&merge);

    IR::IREmitter entry_ir{entry};
    const Writes before{entry_ir, 0};
    IR::IREmitter left_ir{left};
    // Only the predicate, the zero flag and both branch variables change on the left.
    const IR::U1 left_pred{Condition(left_ir, 20)};
    const IR::U1 left_zero{Condition(left_ir, 21)};
    const IR::U1 left_goto{Condition(left_ir, 22)};
    const IR::U32 left_indirect{left_ir.IAdd(left_ir.LocalInvocationIdX(), left_ir.Imm32(23))};
    left_ir.SetPred(IR::Pred::P2, left_pred);
    left_ir.SetZFlag(left_zero);
    left_ir.SetGotoVariable(3, left_goto);
    left_ir.SetIndirectBranchVariable(left_indirect);
    IR::IREmitter right_ir{right};
    const Reads in_right{right_ir};
    IR::IREmitter merge_ir{merge};
    const Reads in_merge{merge_ir};

    Run({&entry, &left, &right, &merge}, {&merge, &right, &left, &entry});

    const auto merged = [&](const IR::Value& after_left, const IR::Value& after_entry) {
        return Definitions{{&left, after_left}, {&right, after_entry}};
    };
    CHECK(Resolved(in_right.pred, &right) == Definitions{{&right, before.pred}});
    CHECK(Resolved(in_right.indirect, &right) == Definitions{{&right, before.indirect}});
    CHECK(Resolved(in_merge.pred, &merge) == merged(left_pred, before.pred));
    CHECK(Resolved(in_merge.zero, &merge) == merged(left_zero, before.zero));
    CHECK(Resolved(in_merge.go_to, &merge) == merged(left_goto, before.go_to));
    CHECK(Resolved(in_merge.indirect, &merge) == merged(left_indirect, before.indirect));
    // Variables no branch redefines need no phi.
    CHECK(Resolved(in_merge.sign, &merge) == Definitions{{&merge, before.sign}});
    CHECK(Resolved(in_merge.carry, &merge) == Definitions{{&merge, before.carry}});
    CHECK(Resolved(in_merge.overflow, &merge) == Definitions{{&merge, before.overflow}});
}
