// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.Const(static_cast<u32>(scope)), ctx.BarrierSemantics(semantics));
}
} // Anonymous namespace

void EmitBarrier(EmitContext& ctx) {
    const Id workgroup{ctx.Const(static_cast<u32>(spv::Scope::Workgroup))};
    ctx.OpControlBarrier(workgroup, workgroup, ctx.GuestBarrierSemantics());
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Device);
}

void EmitBarrierVoteInit(EmitContext& ctx, u32 offset) {
    const Id workgroup{ctx.Const(static_cast<u32>(spv::Scope::Workgroup))};
    ctx.OpAtomicExchange(ctx.U32[1], ctx.BarrierVoteWord(ctx.Const(offset)), workgroup,
                         ctx.u32_zero_value, ctx.u32_zero_value);
    ctx.OpControlBarrier(workgroup, workgroup,
                         ctx.BarrierSemantics(spv::MemorySemanticsMask::AcquireRelease |
                                              spv::MemorySemanticsMask::WorkgroupMemory));
}

void EmitBarrierVote(EmitContext& ctx, Id cond, u32 offset, u32 size) {
    const Id value{ctx.OpSelect(ctx.U32[1], cond, ctx.Const(1U), ctx.u32_zero_value)};
    ctx.OpFunctionCall(ctx.void_id, ctx.barrier_vote_func, ctx.Const(offset), ctx.Const(size),
                       value);
}

void EmitVotedBarrier(EmitContext& ctx, u32 offset, u32 size) {
    ctx.OpFunctionCall(ctx.void_id, ctx.voted_barrier_func, ctx.Const(offset), ctx.Const(size));
}

} // namespace Shader::Backend::SPIRV
