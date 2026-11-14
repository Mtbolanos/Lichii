// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/a64_emit_x64.h"

#include <algorithm>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include "common/assert.h"
#include "common/common_types.h"
#include "dynarmic/ir/terminal.h"
#include "dynarmic/mcl/integer_of_size.hpp"
#include <boost/container/static_vector.hpp>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

// TODO: Have ARM flags in host flags and not have them use up GPR registers unless necessary.
// TODO: Actually implement that proper instruction selector you've always wanted to sweetheart.

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

A64EmitContext::A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& shared_labels)
    : EmitContext(reg_alloc, block, shared_labels)
    , conf(conf)
{}

A64::LocationDescriptor A64EmitContext::Location() const {
    return A64::LocationDescriptor{block.Location()};
}

bool A64EmitContext::IsSingleStep() const {
    return Location().SingleStepping();
}

FP::FPCR A64EmitContext::FPCR(bool fpcr_controlled) const {
    return fpcr_controlled ? Location().FPCR() : Location().FPCR().ASIMDStandardValue();
}

A64EmitX64::A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface)
        : EmitX64(code), conf(conf), jit_interface{jit_interface} {
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        size_t bits = conf.fast_dispatch_table_bits != 0 ? conf.fast_dispatch_table_bits : default_fast_dispatch_table_bits;
        bits = std::clamp<size_t>(bits, 8, 24);
        fast_dispatch_set_size = conf.fast_dispatch_set_associative ? fast_dispatch_ways : 1;
        fast_dispatch_table.resize(size_t(1) << bits);
        fast_dispatch_table_mask = u32(((size_t(1) << bits) - 1) * sizeof(FastDispatchEntry)) & ~u32(fast_dispatch_set_size * sizeof(FastDispatchEntry) - 1);
    }
    code.elide_redundant_zext = conf.elide_redundant_zext;
    if (conf.far_code) {
        code.EnableFarCode(conf.code_cache_size / 4);
    }
    GenMemory128Accessors();
    GenFastmemFallbacks();
    GenTerminalHandlers();
    code.PreludeComplete();
    ClearFastDispatchTable();

    if (conf.fastmem_pointer.has_value()) {
        exception_handler.SetFastmemCallback([this](u64 rip_) {
            return FastmemCallback(rip_);
        });
    }
}

A64EmitX64::~A64EmitX64() = default;

A64EmitX64::BlockDescriptor A64EmitX64::Emit(IR::Block& block) noexcept {
    if (conf.very_verbose_debugging_output) [[unlikely]] {
        std::puts(IR::DumpBlock(block).c_str());
    }

    code.EnableWriting();
    this->reg_alloc.Reset([this] {
        std::bitset<32> gprs = any_gpr;
        if (conf.fastmem_pointer)
            gprs.reset(size_t(HostLoc::R13));
        if (conf.page_table)
            gprs.reset(size_t(HostLoc::R14));
        return gprs;
    }(), any_xmm, conf.spill_to_free_gpr, conf.scratch_copy_to_free_reg,
       HostCallReturn(conf) ? std::optional<s32>{s32(offsetof(A64JitState, spill))} : std::nullopt);

    A64EmitContext ctx{conf, reg_alloc, block, shared_labels};

    // Fold Add64(base, disp) into the [r13 + base + disp] address of the fastmem access that is its
    // only use (unchecked 64-bit fastmem only: the sum then wraps exactly like the guest's).
    code.folded_address_adds.clear();
    if (conf.fold_address_offsets && conf.fastmem_address_space_bits == 64) {
        for (auto& inst : block.instructions) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::A64ReadMemory8:
            case IR::Opcode::A64ReadMemory16:
            case IR::Opcode::A64ReadMemory32:
            case IR::Opcode::A64ReadMemory64:
            case IR::Opcode::A64WriteMemory8:
            case IR::Opcode::A64WriteMemory16:
            case IR::Opcode::A64WriteMemory32:
            case IR::Opcode::A64WriteMemory64:
                break;
            default:
                continue;
            }
            const IR::Value vaddr = inst.GetArg(1);
            if (vaddr.IsImmediate()) {
                continue;
            }
            IR::Inst* const add = vaddr.GetInst();
            if (add->GetOpcode() != IR::Opcode::Add64 || add->UseCount() != 1) {
                continue;
            }
            const IR::Value base = add->GetArg(0);
            const IR::Value offset = add->GetArg(1);
            const IR::Value carry = add->GetArg(2);
            if (base.IsImmediate() || !offset.IsImmediate() || !carry.IsImmediate() || carry.GetU1()) {
                continue;
            }
            const s64 disp = static_cast<s64>(offset.GetImmediateAsU64());
            if (disp < -0x7FFF'FFFF || disp > 0x7FFF'FFFF || !ShouldFastmem(ctx, &inst)) {
                continue;
            }
            code.folded_address_adds.emplace_back(add, static_cast<s32>(disp));
        }
    }

    // Start emitting.
    code.align();
    const auto* const entrypoint = code.getCurr();
    code.nzcv_flags_pos = nullptr;
    code.nzcv_live_pos = nullptr;
    check_bit_in_zf = false;
    call_return_target.reset();
    emitting_block = true;
    pending_exit_stubs.clear();
    if (!conf.omit_block_frame) {
        code.mov(code.qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, abi_base_pointer)], rbp);
        code.lea(rbp, code.ptr[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, abi_base_pointer) - 8]);
    }

    DEBUG_ASSERT(block.GetCondition() == IR::Cond::AL);
    typedef void (EmitX64::*EmitHandlerFn)(EmitContext& context, IR::Inst* inst);
    constexpr EmitHandlerFn opcode_handlers[] = {
#define OPCODE(name, type, ...) &EmitX64::Emit##name,
#define A32OPC(name, type, ...)
#define A64OPC(name, type, ...)
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
    };
    typedef void (A64EmitX64::*A64EmitHandlerFn)(A64EmitContext& context, IR::Inst* inst);
    constexpr A64EmitHandlerFn a64_handlers[] = {
#define OPCODE(...)
#define A32OPC(...)
#define A64OPC(name, type, ...) &A64EmitX64::EmitA64##name,
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
    };

    for (auto& inst : block.instructions) {
        auto const opcode = inst.GetOpcode();
        // Call the relevant Emit* member function.
        switch (opcode) {
#define OPCODE(name, type, ...) case IR::Opcode::name: goto opcode_branch;
#define A32OPC(name, type, ...)
#define A64OPC(name, type, ...) case IR::Opcode::A64##name: goto a64_branch;
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
        default:
            UNREACHABLE();
        }
opcode_branch:
        // PushRSB has an A64 version (the opcode table holds EmitX64's).
        if (opcode == IR::Opcode::PushRSB) {
            EmitPushRSB(ctx, &inst);
        } else {
            (this->*opcode_handlers[size_t(opcode)])(ctx, &inst);
        }
        goto finish_this_inst;
a64_branch:
        (this->*a64_handlers[size_t(opcode) - std::size(opcode_handlers)])(ctx, &inst);
finish_this_inst:
        ctx.reg_alloc.EndOfAllocScope();
#ifndef NDEBUG
        if (conf.very_verbose_debugging_output)
            EmitVerboseDebuggingOutput(reg_alloc);
#endif
    }

    reg_alloc.AssertNoMoreUses();
    if (conf.enable_cycle_counting) {
        EmitAddCycles(block.CycleCount());
        code.nzcv_live_pos = nullptr;
    }
    if (!conf.omit_block_frame) {
        // mov leaves the flags alone.
        const bool flags_live = code.nzcv_live_pos == code.getCurr();
        code.mov(rbp, code.qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, abi_base_pointer)]);
        if (flags_live)
            code.nzcv_live_pos = code.getCurr();
    }
    if (call_return_target && !ctx.IsSingleStep()) {
        EmitHostCall(*call_return_target);
    }
    call_return_target.reset();
    EmitTerminal(block.terminal, ctx.Location().SetSingleStepping(false), ctx.IsSingleStep());
    code.folded_address_adds.clear();
    code.nzcv_flags_pos = nullptr;
    code.nzcv_live_pos = nullptr;
    check_bit_in_zf = false;
    code.int3();
    const bool far_code = code.HasFarCode();
    if (far_code) {
        code.SwitchToFarCode();
    }
    FlushPendingExitStubs();
    emitting_block = false;
    for (auto& deferred_emit : ctx.far_deferred_emits)
        deferred_emit();
    if (far_code) {
        code.SwitchToNearCode();
    }
    for (auto& deferred_emit : ctx.deferred_emits)
        deferred_emit();
    code.int3();

    const size_t size = size_t(code.getCurr() - entrypoint);

    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};

    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);

    auto bdesc = RegisterBlock(descriptor, entrypoint, size);
    code.DisableWriting();
    shared_labels.clear();
    return bdesc;
}

void A64EmitX64::ClearCache() {
    EmitX64::ClearCache();
    exit_stubs.clear();
    pending_exit_stubs.clear();
    block_ranges.ClearCache();
    ClearFastDispatchTable();
    fastmem_patch_info.clear();
}

void A64EmitX64::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A64EmitX64::ClearFastDispatchTable() {
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        std::fill(fast_dispatch_table.begin(), fast_dispatch_table.end(), FastDispatchEntry{});
    }
}

void A64EmitX64::GenTerminalHandlers() {
    // PC ends up in rcx, location_descriptor ends up in rbx
    //static_assert(ABI_ALL_CALLEE_SAVE.test(size_t(HostLoc::R12)));
    const auto calculate_location_descriptor = [this] {
        // This calculation has to match up with A64::LocationDescriptor::UniqueHash
        // TODO: Optimization is available here based on known state of fpcr.
        code.mov(rdi, qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)]);
        code.mov(rcx, A64::LocationDescriptor::pc_mask);
        code.and_(rcx, rdi);
        code.mov(ebx, dword[code.ABI_JIT_PTR + offsetof(A64JitState, fpcr)]);
        code.and_(ebx, A64::LocationDescriptor::fpcr_mask);
        code.shl(rbx, A64::LocationDescriptor::fpcr_shift);
        code.or_(rbx, rcx);
    };

    // With halt_check_on_back_edges forward links skip the halt check, so indirect branches
    // (the PC is already stored) take it.
    const auto check_halt = [this] {
        if (conf.halt_check_on_back_edges && !conf.enable_cycle_counting) {
            code.cmp(dword[code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
            code.jne(code.GetForceReturnFromRunCodeAddress());
        }
    };

    Xbyak::Label fast_dispatch_cache_miss;
    Xbyak::Label rsb_cache_miss;

    code.align();
    terminal_handler_pop_rsb_hint = code.getCurr<const void*>();
    check_halt();
    calculate_location_descriptor();
    code.mov(eax, dword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)]);
    code.sub(eax, 1);
    code.and_(eax, u32(code.GetJitStateInfo().rsb_ptr_mask));
    code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_ptr)], eax);
    code.cmp(rbx, qword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jne(rsb_cache_miss, code.T_NEAR);
    } else {
        code.jne(code.GetReturnFromRunCodeAddress());
    }
    code.mov(rax, qword[code.ABI_JIT_PTR + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
    code.jmp(rax);
    PerfMapRegister(terminal_handler_pop_rsb_hint, code.getCurr(), "a64_terminal_handler_pop_rsb_hint");

    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.align();
        terminal_handler_fast_dispatch_hint = code.getCurr<const void*>();
        check_halt();
        calculate_location_descriptor();
        code.L(rsb_cache_miss);
        code.mov(r8, u64(fast_dispatch_table.data()));
        //code.mov(r12, qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)]);
        code.mov(r12, rbx);
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(r12, r8);
        }
        code.and_(r12d, fast_dispatch_table_mask);
        code.lea(r12, ptr[r8 + r12]);
        for (size_t way = 0; way < fast_dispatch_set_size; ++way) {
            const size_t entry = way * sizeof(FastDispatchEntry);
            Xbyak::Label next_way;
            code.cmp(rbx, qword[r12 + entry + offsetof(FastDispatchEntry, location_descriptor)]);
            code.jne(way + 1 < fast_dispatch_set_size ? next_way : fast_dispatch_cache_miss, code.T_NEAR);
            code.jmp(ptr[r12 + entry + offsetof(FastDispatchEntry, code_ptr)]);
            code.L(next_way);
        }
        code.L(fast_dispatch_cache_miss);
        // Insert at way 0: the older ways move down one and the last one is dropped.
        for (size_t way = fast_dispatch_set_size - 1; way > 0; --way) {
            const size_t to = way * sizeof(FastDispatchEntry);
            const size_t from = to - sizeof(FastDispatchEntry);
            code.mov(rax, qword[r12 + from + offsetof(FastDispatchEntry, location_descriptor)]);
            code.mov(rcx, qword[r12 + from + offsetof(FastDispatchEntry, code_ptr)]);
            code.mov(qword[r12 + to + offsetof(FastDispatchEntry, location_descriptor)], rax);
            code.mov(qword[r12 + to + offsetof(FastDispatchEntry, code_ptr)], rcx);
        }
        code.mov(qword[r12 + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        if (HostCallReturn(conf)) {
            // Compiling can clear the whole code cache, and the guest call frames hold return
            // addresses into it: drop them first (as the dispatcher does).
            code.mov(rsp, qword[code.ABI_JIT_PTR + offsetof(A64JitState, host_stack_base)]);
        }
        code.LookupBlock();
        code.mov(ptr[r12 + offsetof(FastDispatchEntry, code_ptr)], rax);
        code.jmp(rax);
        PerfMapRegister(terminal_handler_fast_dispatch_hint, code.getCurr(), "a64_terminal_handler_fast_dispatch_hint");

        code.align();
        fast_dispatch_table_lookup = code.getCurr<FastDispatchEntry& (*)(u64)>();
        code.mov(code.ABI_PARAM2, u64(fast_dispatch_table.data()));
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(code.ABI_PARAM1, code.ABI_PARAM2);
        }
        code.and_(code.ABI_PARAM1.cvt32(), fast_dispatch_table_mask);
        code.lea(code.ABI_RETURN, code.ptr[code.ABI_PARAM2 + code.ABI_PARAM1]);
        code.ret();
        PerfMapRegister(fast_dispatch_table_lookup, code.getCurr(), "a64_fast_dispatch_table_lookup");
    }
}

void A64EmitX64::EmitPushRSB(EmitContext& ctx, IR::Inst* inst) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        return;
    }
    if (HostCallReturn(conf)) {
        call_return_target = IR::LocationDescriptor{inst->GetArg(0).GetU64()};
        return;
    }

    EmitX64::EmitPushRSB(ctx, inst);
}

void A64EmitX64::EmitA64SetCheckBit(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg8 to_store = ctx.reg_alloc.UseGpr(code, args[0]).cvt8();
    // Last instruction of a CheckBit block: only flag-neutral code (the %rbp restore) follows
    // before the terminal, so the bit can travel in ZF instead of the stack slot.
    if (conf.reuse_host_flags && !conf.enable_cycle_counting && &*ctx.block.instructions.rbegin() == inst &&
        std::holds_alternative<IR::Term::CheckBit>(ctx.block.GetTerminal())) {
        code.test(to_store, to_store);
        check_bit_in_zf = true;
        return;
    }
    code.mov(CheckBitAddress(), to_store);
}

Xbyak::Address A64EmitX64::CheckBitAddress() const {
    if (HostCallReturn(conf)) {
        return code.byte[code.ABI_JIT_PTR + offsetof(A64JitState, check_bit)];
    }
    return code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)];
}

// Guest call frame (host_call_return), from the caller's rsp R down:
//   [R - 8]                   host return address: the landing code after the `call`
//   [R - 16]                  location descriptor of the guest return address
//   [R - 16 - shadow, R - 16) shadow space for the host calls the callee makes
// so the callee runs with rsp = R - HostCallFrameSize, 16-byte aligned like R.
void A64EmitX64::EmitHostCall(const IR::LocationDescriptor& return_target) {
    static_assert(HostCallFrameSize == 16 + ABI_SHADOW_SPACE);
    // Too deep (or frames left by returns that did not match): start again from no frames.
    code.cmp(rsp, qword[code.ABI_JIT_PTR + offsetof(A64JitState, host_stack_limit)]);
    code.cmovb(rsp, qword[code.ABI_JIT_PTR + offsetof(A64JitState, host_stack_base)]);
    Xbyak::Label frame;
    code.mov(rax, return_target.Value());
    code.call(frame);
    // Landing: a RET to the return address popped the frame, rsp is R again.
    AddPatchLocation(return_target, PatchKind::Jmp);
    if (const auto next_bb = GetBasicBlock(return_target)) {
        EmitPatchJmp(return_target, next_bb->entrypoint);
    } else {
        EmitPatchJmp(return_target);
    }
    code.L(frame);
    code.lea(rsp, ptr[rsp - (HostCallFrameSize - 8)]);
    code.mov(qword[rsp + ABI_SHADOW_SPACE], rax);
    // The terminal (the link to the callee) follows.
}

void A64EmitX64::EmitA64GetCFlag(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)]);
    code.shr(result, NZCV::x64_c_flag_bit);
    code.and_(result, 1);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.ScratchGpr(code).cvt32();

    code.mov(nzcv_raw, dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)]);

    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr(code).cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pext(nzcv_raw, nzcv_raw, tmp);
        code.shl(nzcv_raw, 28);
    } else {
        code.and_(nzcv_raw, NZCV::x64_mask);
        code.imul(nzcv_raw, nzcv_raw, NZCV::from_x64_multiplier);
        code.and_(nzcv_raw, NZCV::arm_mask);
    }

    ctx.reg_alloc.DefineValue(code, inst, nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.UseScratchGpr(code, args[0]).cvt32();

    code.shr(nzcv_raw, 28);
    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr(code).cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pdep(nzcv_raw, nzcv_raw, tmp);
    } else {
        code.imul(nzcv_raw, nzcv_raw, NZCV::to_x64_multiplier);
        code.and_(nzcv_raw, NZCV::x64_mask);
    }
    code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)], nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCV(A64EmitContext& ctx, IR::Inst* inst) {
    // Stored right after the Add/Sub that set it (nothing emitted in between): the host flags
    // stay equal to the stored NZCV until the next flag-changing instruction.
    const bool flags_live = conf.reuse_host_flags && code.nzcv_flags_pos != nullptr && code.nzcv_flags_pos == code.getCurr() &&
                            !inst->GetArg(0).IsImmediate() && code.nzcv_flags_inst == static_cast<const void*>(inst->GetArg(0).GetInst());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 to_store = ctx.reg_alloc.UseScratchGpr(code, args[0]).cvt32();
    code.mov(dword[code.ABI_JIT_PTR + offsetof(A64JitState, cpsr_nzcv)], to_store);
    code.nzcv_live_pos = flags_live ? code.getCurr() : nullptr;
}

void A64EmitX64::EmitA64GetW(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();

    code.mov(result, dword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetX(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);

    code.mov(result, qword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetS(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
    code.movd(result, addr);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetD(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
    code.movq(result, addr);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetQ(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
    code.movaps(result, addr);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetSP(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);
    code.mov(result, qword[code.ABI_JIT_PTR + offsetof(A64JitState, sp)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, dword[code.ABI_JIT_PTR + offsetof(A64JitState, fpcr)]);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

static u32 GetFPSRImpl(A64JitState* jit_state) {
    return jit_state->GetFpsr();
}

void A64EmitX64::EmitA64GetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, inst);
    code.mov(code.ABI_PARAM1, code.ABI_JIT_PTR);
    code.stmxcsr(code.dword[code.ABI_JIT_PTR + offsetof(A64JitState, guest_MXCSR)]);
    code.CallFunction(GetFPSRImpl);
}

void A64EmitX64::EmitA64SetW(A64EmitContext& ctx, IR::Inst* inst) {
    // Only movs below: live host flags stay live.
    const bool flags_live = code.nzcv_live_pos != nullptr && code.nzcv_live_pos == code.getCurr();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (conf.elide_redundant_zext && !args[1].IsImmediate() && IsZeroExtendedU32Producer(args[1].GetInst())) {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[1]);
        code.mov(addr, to_store);
    } else {
        // TODO: zext tracking, xmm variant
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseScratchGpr(code, args[1]);
        code.mov(to_store.cvt32(), to_store.cvt32());
        code.mov(addr, to_store);
    }
    if (flags_live)
        code.nzcv_live_pos = code.getCurr();
}

void A64EmitX64::EmitA64SetX(A64EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = code.nzcv_live_pos != nullptr && code.nzcv_live_pos == code.getCurr();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (args[1].IsInXmm(ctx.reg_alloc)) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[1]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[1]);
        code.mov(addr, to_store);
    }
    if (flags_live)
        code.nzcv_live_pos = code.getCurr();
}

void A64EmitX64::EmitA64SetS(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[1]);
    const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm(code);
    // TODO: Optimize
    code.pxor(tmp, tmp);
    code.movss(tmp, to_store);
    code.movaps(addr, tmp);
}

void A64EmitX64::EmitA64SetD(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseScratchXmm(code, args[1]);
    code.movq(to_store, to_store);  // TODO: Remove when able
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetQ(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[code.ABI_JIT_PTR + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[1]);
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetSP(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, sp)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm(ctx.reg_alloc)) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[0]);
        code.mov(addr, to_store);
    }
}

static void SetFPCRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpcr(value);
}

void A64EmitX64::EmitA64SetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.ABI_JIT_PTR);
    code.CallFunction(SetFPCRImpl);
    code.ldmxcsr(code.dword[code.ABI_JIT_PTR + offsetof(A64JitState, guest_MXCSR)]);
}

static void SetFPSRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpsr(value);
}

void A64EmitX64::EmitA64SetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.ABI_JIT_PTR);
    code.CallFunction(SetFPSRImpl);
    code.ldmxcsr(code.dword[code.ABI_JIT_PTR + offsetof(A64JitState, guest_MXCSR)]);
}

void A64EmitX64::EmitA64SetPC(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm(ctx.reg_alloc)) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(code, args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(code, args[0]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64CallSupervisor(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const u32 imm = args[0].GetImmediateU32();
    Devirtualize<&A64::UserCallbacks::CallSVC>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], imm);
    });
    // The kernel would have to execute ERET to get here, which would clear exclusive state.
    code.mov(code.byte[code.ABI_JIT_PTR + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64ExceptionRaised(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate() && args[1].IsImmediate());
    const u64 pc = args[0].GetImmediateU64();
    const u64 exception = args[1].GetImmediateU64();
    Devirtualize<&A64::UserCallbacks::ExceptionRaised>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], pc);
        code.mov(param[1], exception);
    });
}

void A64EmitX64::EmitA64DataCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[1], args[2]);
    Devirtualize<&A64::UserCallbacks::DataCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64InstructionCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(code, nullptr, {}, args[0], args[1]);
    Devirtualize<&A64::UserCallbacks::InstructionCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64DataSynchronizationBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
    code.lfence();
}

void A64EmitX64::EmitA64DataMemoryBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
}

void A64EmitX64::EmitA64InstructionSynchronizationBarrier(A64EmitContext& ctx, IR::Inst*) {
    if (!conf.hook_isb) {
        return;
    }

    ctx.reg_alloc.HostCall(code, nullptr);
    Devirtualize<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCNTFRQ(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, conf.cntfrq_el0);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetCNTPCT(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(code, inst);
    if (!conf.wall_clock_cntpct) {
        code.UpdateTicks();
    }
    Devirtualize<&A64::UserCallbacks::GetCNTPCT>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCTR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, conf.ctr_el0);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetDCZID(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr(code).cvt32();
    code.mov(result, conf.dczid_el0);
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);
    if (conf.tpidr_el0) {
        code.mov(result, u64(conf.tpidr_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64GetTPIDRRO(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code);
    if (conf.tpidrro_el0) {
        code.mov(result, u64(conf.tpidrro_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64SetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg64 value = ctx.reg_alloc.UseGpr(code, args[0]);
    const Xbyak::Reg64 addr = ctx.reg_alloc.ScratchGpr(code);
    if (conf.tpidr_el0) {
        code.mov(addr, u64(conf.tpidr_el0));
        code.mov(qword[addr], value);
    }
}

std::string A64EmitX64::LocationDescriptorToFriendlyName(const IR::LocationDescriptor& ir_descriptor) const {
    const A64::LocationDescriptor descriptor{ir_descriptor};
    return fmt::format("a64_{:016X}_fpcr{:08X}",
                       descriptor.PC(),
                       descriptor.FPCR().Value());
}

namespace {
bool EmitTerminalImpl(A64EmitX64& e, IR::Term::ReturnToDispatch, IR::LocationDescriptor, bool) {
    e.code.ReturnFromRunCode();
    return true;
}

// A link to a higher guest PC cannot close a loop on its own: every cycle of blocks contains a
// link to the same or a lower start PC, or an indirect branch, and those check for a halt.
bool IsForwardLink(const A64EmitX64& e, const IR::LocationDescriptor& next, const IR::LocationDescriptor& initial_location) {
    return e.conf.halt_check_on_back_edges && !e.conf.enable_cycle_counting &&
           A64::LocationDescriptor{next}.PC() > A64::LocationDescriptor{initial_location}.PC();
}

void EmitHaltCheckedLink(A64EmitX64& e, const IR::LocationDescriptor& next) {
    e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
    e.AddPatchLocation(next, EmitX64::PatchKind::Jz);
    if (const auto next_bb = e.GetBasicBlock(next)) {
        e.EmitPatchJz(next, next_bb->entrypoint);
    } else {
        e.EmitPatchJz(next);
    }
    e.EmitUnlinkedExit(next);
}

void EmitUncheckedLink(A64EmitX64& e, const IR::LocationDescriptor& next) {
    e.AddPatchLocation(next, EmitX64::PatchKind::Jmp);
    if (auto next_bb = e.GetBasicBlock(next)) {
        e.EmitPatchJmp(next, next_bb->entrypoint);
    } else {
        e.EmitPatchJmp(next);
    }
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    // Used for patches and linking
    if (e.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
        if (IsForwardLink(e, terminal.next, initial_location)) {
            EmitUncheckedLink(e, terminal.next);
            return true;
        }
        if (e.conf.enable_cycle_counting) {
            e.code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
            e.AddPatchLocation(terminal.next, EmitX64::PatchKind::Jg);
            if (const auto next_bb = e.GetBasicBlock(terminal.next)) {
                e.EmitPatchJg(terminal.next, next_bb->entrypoint);
            } else {
                e.EmitPatchJg(terminal.next);
            }
        } else {
            e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
            e.AddPatchLocation(terminal.next, EmitX64::PatchKind::Jz);
            if (const auto next_bb = e.GetBasicBlock(terminal.next)) {
                e.EmitPatchJz(terminal.next, next_bb->entrypoint);
            } else {
                e.EmitPatchJz(terminal.next);
            }
        }
        e.EmitUnlinkedExit(terminal.next);
    } else {
        e.code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        e.code.mov(qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        e.code.ReturnFromRunCode();
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (e.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
        if (e.conf.halt_check_on_back_edges && !e.conf.enable_cycle_counting && !IsForwardLink(e, terminal.next, initial_location)) {
            EmitHaltCheckedLink(e, terminal.next);
        } else {
            EmitUncheckedLink(e, terminal.next);
        }
    } else {
        e.code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        e.code.mov(qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        e.code.ReturnFromRunCode();
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::PopRSBHint, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (HostCallReturn(e.conf) && e.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) && !is_single_step) {
        // Host `ret` when the guest returns to the address the innermost frame's call pushed:
        // its location descriptor is the PC masked to pc_mask plus this block's FPCR bits (an
        // RET does not change the FPCR). Else the frame stays and the fast dispatcher runs.
        e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
        e.code.jne(e.code.GetForceReturnFromRunCodeAddress());
        e.code.mov(rax, qword[e.code.ABI_JIT_PTR + offsetof(A64JitState, pc)]);
        static_assert(A64::LocationDescriptor::pc_mask == (u64(1) << 56) - 1);
        e.code.shl(rax, 8);
        e.code.shr(rax, 8);
        const u64 fpcr_bits = A64::LocationDescriptor{initial_location}.SetPC(0).UniqueHash();
        if (fpcr_bits != 0) {
            e.code.mov(rcx, fpcr_bits);
            e.code.or_(rax, rcx);
        }
        e.code.cmp(rax, qword[rsp + ABI_SHADOW_SPACE]);
        if (e.conf.HasOptimization(OptimizationFlag::FastDispatch)) {
            e.code.jne(e.terminal_handler_fast_dispatch_hint);
        } else {
            e.code.jne(e.code.GetReturnFromRunCodeAddress());
        }
        e.code.lea(rsp, ptr[rsp + (HostCallFrameSize - 8)]);
        e.code.ret();
        return true;
    }
    if (e.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) && !is_single_step) {
        e.code.jmp(e.terminal_handler_pop_rsb_hint);
    } else {
        e.code.ReturnFromRunCode();
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::FastDispatchHint, IR::LocationDescriptor, bool is_single_step) {
    if (!e.conf.HasOptimization(OptimizationFlag::FastDispatch) || is_single_step) {
        e.code.ReturnFromRunCode();
    } else {
        e.code.jmp(e.terminal_handler_fast_dispatch_hint);
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    switch (terminal.if_) {
    case IR::Cond::AL:
    case IR::Cond::NV:
        e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
        break;
    default:
        Xbyak::Label pass = e.EmitCond(terminal.if_);
        e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
        e.code.L(pass);
        e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
        break;
    }
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    Xbyak::Label fail;
    if (e.check_bit_in_zf) {
        // EmitA64SetCheckBit left `test bit, bit`: ZF set means the bit is clear.
        e.check_bit_in_zf = false;
    } else {
        e.code.cmp(e.CheckBitAddress(), u8(0));
    }
    e.code.jz(fail);
    e.EmitLeafTerminal(terminal.then_, initial_location, is_single_step);
    e.code.L(fail);
    e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
    return true;
}

bool EmitTerminalImpl(A64EmitX64& e, IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    e.code.cmp(dword[e.code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], 0);
    e.code.jne(e.code.GetForceReturnFromRunCodeAddress());
    e.EmitLeafTerminal(terminal.else_, initial_location, is_single_step);
    return true;
}

}

bool A64EmitX64::EmitLeafTerminal(IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept {
    if (auto const x = std::get_if<IR::Term::ReturnToDispatch>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::LinkBlock>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::LinkBlockFast>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::PopRSBHint>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::FastDispatchHint>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    UNREACHABLE();
}

bool A64EmitX64::EmitTerminal(IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept {
    if (auto const x = std::get_if<IR::Term::LeafTerminal>(&terminal))
        return EmitLeafTerminal(*x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::If>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::CheckBit>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::CheckHalt>(&terminal))
        return EmitTerminalImpl(*this, *x, initial_location, is_single_step);
    UNREACHABLE();
}

Xbyak::Label* A64EmitX64::PendingExitStub(const IR::LocationDescriptor& target) {
    for (const auto& [value, label] : pending_exit_stubs) {
        if (value == target.Value()) {
            return label;
        }
    }
    Xbyak::Label* const label = &shared_labels.emplace_back();
    pending_exit_stubs.emplace_back(target.Value(), label);
    return label;
}

void A64EmitX64::EnsureExitStub(const IR::LocationDescriptor& target) {
    if (!exit_stubs.contains(target.Value())) {
        PendingExitStub(target);
    }
}

void A64EmitX64::FlushPendingExitStubs() {
    for (const auto& [value, label] : pending_exit_stubs) {
        code.L(*label);
        exit_stubs[value] = code.getCurr<CodePtr>();
        code.mov(rax, A64::LocationDescriptor{IR::LocationDescriptor{value}}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
    }
    pending_exit_stubs.clear();
}

void A64EmitX64::EmitUnlinkedExit(const IR::LocationDescriptor& target) {
    if (conf.compact_block_exits) {
        // The stub returns through the dispatcher, which also returns when a halt is pending.
        if (const auto it = exit_stubs.find(target.Value()); it != exit_stubs.end()) {
            code.jmp(it->second, code.T_NEAR);
        } else {
            ASSERT(emitting_block);
            code.jmp(*PendingExitStub(target), code.T_NEAR);
        }
        return;
    }
    code.mov(rax, A64::LocationDescriptor{target}.PC());
    code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
}

// Compact exits: a 5/6-byte rel32 jump to the target block, or to the target's shared exit stub
// while unlinked. Every site gets its stub at first emission, so Unpatch always finds one.
#define COMPACT_PATCH(JCC, SIZE)                                                                    \
    if (conf.compact_block_exits) {                                                                 \
        if (emitting_block) {                                                                       \
            EnsureExitStub(target_desc);                                                            \
        }                                                                                           \
        if (target_code_ptr) {                                                                      \
            code.JCC(target_code_ptr);                                                              \
        } else if (const auto it = exit_stubs.find(target_desc.Value()); it != exit_stubs.end()) { \
            code.JCC(it->second);                                                                   \
        } else {                                                                                    \
            ASSERT(emitting_block);                                                                 \
            code.JCC(*PendingExitStub(target_desc), code.T_NEAR);                                   \
        }                                                                                           \
        code.EnsurePatchLocationSize(patch_location, SIZE);                                         \
        return;                                                                                     \
    }

void A64EmitX64::EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    COMPACT_PATCH(jg, 6)
    if (target_code_ptr) {
        code.jg(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.jg(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    COMPACT_PATCH(jz, 6)
    if (target_code_ptr) {
        code.jz(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.jz(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    COMPACT_PATCH(jmp, 5)
    if (target_code_ptr) {
        code.jmp(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
        code.jmp(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 22);
}

#undef COMPACT_PATCH

void A64EmitX64::EmitPatchMovRcx(CodePtr target_code_ptr) {
    if (!target_code_ptr) {
        target_code_ptr = code.GetReturnFromRunCodeAddress();
    }
    const CodePtr patch_location = code.getCurr();
    code.mov(code.rcx, u64(target_code_ptr));
    code.EnsurePatchLocationSize(patch_location, 10);
}

void A64EmitX64::Unpatch(const IR::LocationDescriptor& location) {
    EmitX64::Unpatch(location);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.DisableWriting();
        FastDispatchEntry* const set = &(*fast_dispatch_table_lookup)(location.Value());
        for (size_t way = 0; way < fast_dispatch_set_size; ++way) {
            if (set[way].location_descriptor == location.Value()) {
                set[way] = {};
            }
        }
        code.EnableWriting();
    }
}

}  // namespace Dynarmic::Backend::X64
