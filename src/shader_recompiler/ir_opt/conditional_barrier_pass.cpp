// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "common/alignment.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
using NodeType = IR::AbstractSyntaxNode::Type;

// Values that can differ between the invocations of a workgroup whatever their operands are.
bool IsDivergentSource(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LocalInvocationId:
    case IR::Opcode::InvocationId:
    case IR::Opcode::SampleId:
    case IR::Opcode::IsHelperInvocation:
    case IR::Opcode::GetAttribute:
    case IR::Opcode::GetAttributeU32:
    case IR::Opcode::GetAttributeIndexed:
    case IR::Opcode::IsbeRead:
    case IR::Opcode::LoadLocal:
    case IR::Opcode::LaneId:
    case IR::Opcode::SubgroupBallot:
    case IR::Opcode::SubgroupEqMask:
    case IR::Opcode::SubgroupLtMask:
    case IR::Opcode::SubgroupLeMask:
    case IR::Opcode::SubgroupGtMask:
    case IR::Opcode::SubgroupGeMask:
    case IR::Opcode::FSwizzleAdd:
    case IR::Opcode::DPdxFine:
    case IR::Opcode::DPdyFine:
    case IR::Opcode::DPdxCoarse:
    case IR::Opcode::DPdyCoarse:
    case IR::Opcode::GetInBoundsFromOp:
        return true;
    case IR::Opcode::ConditionRef:
        return false;
    default:
        // Atomics return a different value to each invocation.
        return inst.MayHaveSideEffects() && inst.Type() != IR::Type::Void;
    }
}

// A conservative divergence analysis over the structured program: a value is uniform when every
// invocation of the workgroup that computes it gets the same result, and control flow is uniform
// when every invocation that reaches a construct takes the same path through it.
class Divergence {
public:
    explicit Divergence(IR::Program& program_)
        : program{program_}, divergent_loops(program.syntax_list.size()) {
        while (Walk(Mode::Analyze)) {
        }
    }

    bool HasDivergentBarrier() {
        return Walk(Mode::Find);
    }

    void DemoteBarriers() {
        Walk(Mode::Demote);
    }

    // Keeps the execution barriers of a compute program while the whole workgroup takes the same
    // way at every branch that could split it around one. Each such branch first counts its
    // condition into a shared word; a barrier runs as an execution barrier while that count stays
    // a multiple of the workgroup size, and keeps only its memory ordering once a branch split the
    // workgroup. Returns false when a branch has no block to count its condition in, or when the
    // word does not fit in shared memory (a SPIR-V explicit layout rounds it up to 16 bytes).
    bool VoteBarriers(u32 max_shared_memory_size) {
        const u32 offset{Common::AlignUp(program.shared_memory_size, 4U)};
        if (max_shared_memory_size != 0 &&
            Common::AlignUp(offset + 4, 16U) > max_shared_memory_size) {
            return false;
        }
        FindVotes();
        const auto& syntax_list{program.syntax_list};
        for (size_t index = 0; index < syntax_list.size(); ++index) {
            if (votes[index] && (index == 0 || syntax_list[index - 1].type != NodeType::Block)) {
                return false;
            }
        }
        vote_offset = offset;
        program.shared_memory_size = vote_offset + 4;
        workgroup_size = program.workgroup_size[0] * program.workgroup_size[1] *
                         program.workgroup_size[2];
        Walk(Mode::Vote);

        IR::Block& entry{*syntax_list.front().data.block};
        const auto it{
            std::ranges::find(entry.Instructions(), IR::Opcode::Prologue, &IR::Inst::GetOpcode)};
        entry.PrependNewInst(it == entry.end() ? entry.begin() : std::next(it),
                             IR::Opcode::BarrierVoteInit, {IR::Value{vote_offset}});
        program.info.uses_barrier_vote = true;
        return true;
    }

private:
    enum class Mode { Analyze, Find, Demote, Vote };

    struct Construct {
        size_t node;
        bool divergent;
    };

    // The divergent branches whose way decides whether some invocations skip a barrier: an if
    // around a barrier, a return or a break out of a loop with a barrier, and the breaks and
    // repeats of such a loop. A vote waits for the workgroup as a barrier does, so the constructs
    // around one need votes too, until no new vote is found.
    void FindVotes() {
        const auto& syntax_list{program.syntax_list};
        std::vector<bool> has_barrier(syntax_list.size());
        votes.assign(syntax_list.size(), false);
        std::vector<size_t> stack;
        bool changed{true};
        const auto vote{[&](size_t index, const IR::U1& cond) {
            if (!votes[index] && IsDivergent(cond)) {
                votes[index] = true;
                changed = true;
            }
        }};
        const auto vote_ifs{[&] {
            for (const size_t construct : stack) {
                if (syntax_list[construct].type == NodeType::If) {
                    vote(construct, syntax_list[construct].data.if_node.cond);
                }
            }
        }};
        while (changed) {
            changed = false;
            stack.clear();
            for (size_t index = 0; index < syntax_list.size(); ++index) {
                const IR::AbstractSyntaxNode& node{syntax_list[index]};
                const bool waits{
                    votes[index] ||
                    (node.type == NodeType::Block &&
                     std::ranges::any_of(node.data.block->Instructions(), [](const IR::Inst& inst) {
                         return inst.GetOpcode() == IR::Opcode::Barrier;
                     }))};
                if (waits) {
                    for (const size_t construct : stack) {
                        has_barrier[construct] = true;
                    }
                }
                switch (node.type) {
                case NodeType::If:
                case NodeType::Loop:
                    stack.push_back(index);
                    break;
                case NodeType::EndIf:
                case NodeType::Repeat:
                    stack.pop_back();
                    break;
                default:
                    break;
                }
            }
            for (size_t index = 0; index < syntax_list.size(); ++index) {
                const IR::AbstractSyntaxNode& node{syntax_list[index]};
                switch (node.type) {
                case NodeType::If:
                    if (has_barrier[index]) {
                        vote(index, node.data.if_node.cond);
                    }
                    stack.push_back(index);
                    break;
                case NodeType::Loop:
                    stack.push_back(index);
                    break;
                case NodeType::EndIf:
                    stack.pop_back();
                    break;
                case NodeType::Repeat:
                    if (has_barrier[stack.back()]) {
                        vote(index, node.data.repeat.cond);
                    }
                    stack.pop_back();
                    break;
                case NodeType::Return:
                    vote_ifs();
                    break;
                case NodeType::Break: {
                    const auto loop{std::ranges::find_if(stack.rbegin(), stack.rend(), [&](size_t c) {
                        return syntax_list[c].type == NodeType::Loop;
                    })};
                    if (loop != stack.rend() && has_barrier[*loop]) {
                        vote(index, node.data.break_node.cond);
                        vote_ifs();
                    }
                    break;
                }
                default:
                    break;
                }
            }
        }
    }

    void Vote(size_t index, const IR::U1& cond) {
        if (!votes[index]) {
            return;
        }
        program.syntax_list[index - 1].data.block->AppendNewInst(
            IR::Opcode::BarrierVote,
            {cond, IR::Value{vote_offset}, IR::Value{workgroup_size}});
    }

    bool IsDivergent(const IR::Value& value) const {
        return !value.IsImmediate() && values.contains(value.InstRecursive());
    }

    bool Mark(const IR::Inst& inst, bool divergent_loop, bool divergent_merge) {
        if (values.contains(&inst)) {
            return false;
        }
        bool divergent{divergent_loop || IsDivergentSource(inst) ||
                       (divergent_merge && inst.GetOpcode() == IR::Opcode::Phi)};
        for (size_t arg = 0; !divergent && arg < inst.NumArgs(); ++arg) {
            divergent = IsDivergent(inst.Arg(arg));
        }
        return divergent && values.insert(&inst).second;
    }

    bool MarkLoop(size_t node) {
        if (divergent_loops[node]) {
            return false;
        }
        divergent_loops[node] = true;
        return true;
    }

    // One pass in program order. Returns true when it found new divergence, so loop back edges
    // and divergence found after a construct's start reach a fixed point.
    bool Walk(Mode mode) {
        std::vector<Construct> stack;
        bool exited{};
        bool changed{};
        auto& syntax_list{program.syntax_list};
        for (size_t index = 0; index < syntax_list.size(); ++index) {
            const IR::AbstractSyntaxNode& node{syntax_list[index]};
            // Once part of the workgroup has returned, the rest runs without it.
            const bool divergent_flow{
                exited || std::ranges::any_of(stack, &Construct::divergent)};
            switch (node.type) {
            case NodeType::Block: {
                // Values computed in a loop that invocations leave on different iterations differ
                // after it, even when every iteration computes them uniformly.
                const bool divergent_loop{std::ranges::any_of(stack, [&](const Construct& c) {
                    return syntax_list[c.node].type == NodeType::Loop && c.divergent;
                })};
                const bool divergent_merge{divergent_merges.contains(node.data.block)};
                IR::Block& block{*node.data.block};
                for (auto it = block.begin(); it != block.end(); ++it) {
                    IR::Inst& inst{*it};
                    if (mode != Mode::Analyze && divergent_flow &&
                        inst.GetOpcode() == IR::Opcode::Barrier) {
                        changed = true;
                        if (mode == Mode::Find) {
                            continue;
                        }
                        program.info.demoted_conditional_barrier = true;
                        if (mode == Mode::Vote) {
                            LOG_INFO(Shader, "Barrier in divergent control flow waits for the "
                                             "workgroup while its branches agree");
                            block.PrependNewInst(
                                it, IR::Opcode::VotedBarrier,
                                {IR::Value{vote_offset}, IR::Value{workgroup_size}});
                            inst.Invalidate();
                            continue;
                        }
                        LOG_WARNING(Shader, "Barrier in divergent control flow keeps only its "
                                            "memory ordering");
                        inst.ReplaceOpcode(IR::Opcode::WorkgroupMemoryBarrier);
                        continue;
                    }
                    if (mode == Mode::Analyze) {
                        changed |= Mark(inst, divergent_loop, divergent_merge);
                    }
                }
                break;
            }
            case NodeType::If: {
                const bool divergent{IsDivergent(node.data.if_node.cond)};
                if (mode == Mode::Vote) {
                    Vote(index, node.data.if_node.cond);
                }
                if (divergent) {
                    changed |= divergent_merges.insert(node.data.if_node.merge).second;
                }
                stack.push_back({index, divergent});
                break;
            }
            case NodeType::Loop:
                stack.push_back({index, divergent_loops[index]});
                break;
            case NodeType::Break: {
                if (mode == Mode::Vote) {
                    Vote(index, node.data.break_node.cond);
                }
                if (!divergent_flow && !IsDivergent(node.data.break_node.cond)) {
                    break;
                }
                const auto loop{std::ranges::find_if(stack.rbegin(), stack.rend(),
                                                     [&](const Construct& c) {
                                                         return syntax_list[c.node].type ==
                                                                NodeType::Loop;
                                                     })};
                if (loop != stack.rend()) {
                    changed |= MarkLoop(loop->node);
                }
                break;
            }
            case NodeType::EndIf:
                stack.pop_back();
                break;
            case NodeType::Repeat: {
                if (mode == Mode::Vote) {
                    Vote(index, node.data.repeat.cond);
                }
                const size_t loop{stack.back().node};
                if (IsDivergent(node.data.repeat.cond)) {
                    changed |= MarkLoop(loop);
                }
                if (divergent_loops[loop]) {
                    changed |= divergent_merges.insert(node.data.repeat.merge).second;
                }
                stack.pop_back();
                break;
            }
            case NodeType::Return:
                if (!divergent_flow) {
                    break;
                }
                // The invocations still in an enclosing loop run its next iterations without the
                // ones that returned.
                exited = true;
                for (const Construct& construct : stack) {
                    if (syntax_list[construct.node].type == NodeType::Loop) {
                        changed |= MarkLoop(construct.node);
                    }
                }
                break;
            case NodeType::Unreachable:
                break;
            }
        }
        ASSERT(stack.empty());
        return changed;
    }

    IR::Program& program;
    std::unordered_set<const IR::Inst*> values;
    std::unordered_set<const IR::Block*> divergent_merges;
    std::vector<bool> divergent_loops;
    std::vector<bool> votes;
    u32 vote_offset{};
    u32 workgroup_size{};
};
} // Anonymous namespace

// Some drivers hang when only part of a workgroup reaches a barrier, which the guest allows: the
// invocations that skip it or have exited count as arrived. Compute programs vote on the branches
// around such barriers and wait at them while the workgroup stays together; elsewhere barriers
// under divergent control flow keep their memory ordering without the execution barrier. The rest
// stay as they are.
void ConditionalBarrierPass(IR::Program& program, u32 max_shared_memory_size) {
    Divergence divergence{program};
    if (!divergence.HasDivergentBarrier()) {
        return;
    }
    if (program.stage == Stage::Compute && divergence.VoteBarriers(max_shared_memory_size)) {
        return;
    }
    divergence.DemoteBarriers();
}

} // namespace Shader::Optimization
