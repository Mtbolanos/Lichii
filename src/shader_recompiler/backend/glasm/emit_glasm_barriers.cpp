// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/glasm/emit_glasm_instructions.h"
#include "shader_recompiler/backend/glasm/glasm_emit_context.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Backend::GLASM {

void EmitBarrier(EmitContext& ctx) {
    if (ctx.info.requires_coherent_global_memory || ctx.info.requires_coherent_image_memory) {
        ctx.Add("MEMBAR.CTA;");
    }
    ctx.Add("BAR;");
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    ctx.Add("MEMBAR.CTA;");
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    ctx.Add("MEMBAR;");
}

} // namespace Shader::Backend::GLASM
