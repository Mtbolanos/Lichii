// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/glsl/emit_glsl_instructions.h"
#include "shader_recompiler/backend/glsl/glsl_emit_context.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Backend::GLSL {
void EmitBarrier(EmitContext& ctx) {
    if (ctx.info.requires_coherent_global_memory || ctx.info.requires_coherent_image_memory) {
        ctx.Add("groupMemoryBarrier();");
    }
    ctx.Add("barrier();");
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    ctx.Add("groupMemoryBarrier();");
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    ctx.Add("memoryBarrier();");
}

void EmitBarrierVoteInit(EmitContext& ctx, u32 offset) {
    ctx.Add("atomicExchange(smem[{}],0u);barrier();", offset / 4);
}

void EmitBarrierVote(EmitContext& ctx, std::string_view cond, u32 offset, u32 size) {
    ctx.Add("if(atomicAdd(smem[{}],0u)%{}u==0u){{barrier();atomicAdd(smem[{}],{}?1u:0u);"
            "barrier();}}",
            offset / 4, size, offset / 4, cond);
}

void EmitVotedBarrier(EmitContext& ctx, u32 offset, u32 size) {
    ctx.Add("if(atomicAdd(smem[{}],0u)%{}u==0u){{barrier();}}else{{groupMemoryBarrier();}}",
            offset / 4, size);
}
} // namespace Shader::Backend::GLSL
