// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// UserConfig::host_call_return: guest BL/BLR/RET as host call/ret. Every program runs with the
// option on and off and must end in the same state, including returns that do not go back to
// the caller (frames left behind), recursion deeper than HostCallMaxDepth, an FPCR change in a
// callee and blocks that spill.

#include <array>

#include <catch2/catch_test_macros.hpp>
#include <oaknut/oaknut.hpp>

#include "./testenv.h"
#include "dynarmic/interface/A64/a64.h"

using namespace Dynarmic;
using namespace oaknut::util;

namespace {

class HaltOnSvcEnv final : public A64TestEnv {
public:
    A64::Jit* jit = nullptr;
    void CallSVC(std::uint32_t) override { jit->HaltExecution(); }
};

struct Result {
    std::array<u64, 31> regs;
    u64 sp;
    u32 fpcr;
    bool operator==(const Result&) const = default;
};

template<typename Gen>
Result RunProgram(bool host_call_return, Gen&& gen) {
    HaltOnSvcEnv env;
    oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
    gen(code);

    A64::UserConfig config{};
    config.callbacks = &env;
    config.enable_cycle_counting = false;
    config.omit_block_frame = true;
    config.halt_check_on_back_edges = true;
    config.host_call_return = host_call_return;
    A64::Jit jit{config};
    env.jit = &jit;

    jit.SetPC(0);
    jit.SetSP(0x80000);
    // Run twice: the second run starts with every block compiled and linked.
    for (int i = 0; i < 2; ++i) {
        jit.SetPC(0);
        jit.Run();
    }
    Result r{};
    for (size_t i = 0; i < 31; ++i) {
        r.regs[i] = jit.GetRegister(i);
    }
    r.sp = jit.GetSP();
    r.fpcr = jit.GetFpcr();
    return r;
}

template<typename Gen>
Result RunBoth(Gen&& gen) {
    const Result off = RunProgram(false, gen);
    const Result on = RunProgram(true, gen);
    REQUIRE(on == off);
    return on;
}

}  // namespace

TEST_CASE("A64: host_call_return nested calls", "[a64][host_call_return]") {
    const Result r = RunBoth([](auto& code) {
        oaknut::Label loop, f1, f2, skip;
        code.MOV(X0, XZR);
        code.MOV(X1, XZR);
        code.MOVZ(X19, 1000);
        code.l(loop);
        code.BL(f1);
        code.SUBS(X19, X19, 1);
        code.B(NE, loop);
        code.SVC(0);
        code.l(f1);
        code.STP(X29, X30, SP, PRE_INDEXED, -16);
        code.ADD(X0, X0, 1);
        code.BL(f2);
        code.CBZ(X1, skip);  // CheckBit terminal
        code.ADD(X0, X0, 1000);
        code.l(skip);
        code.LDP(X29, X30, SP, POST_INDEXED, 16);
        code.RET();
        code.l(f2);
        code.ADD(X0, X0, 2);
        code.RET();
    });
    REQUIRE(r.regs[0] == 3000);
    REQUIRE(r.sp == 0x80000);
}

TEST_CASE("A64: host_call_return returns elsewhere", "[a64][host_call_return]") {
    // Each callee returns past the instruction after its BL (RET to another address) or with BR,
    // so frames stay on the host stack: 5000 of them, beyond HostCallMaxDepth.
    const Result r = RunBoth([](auto& code) {
        oaknut::Label loop, skip_ret, br_ret;
        code.MOV(X1, XZR);
        code.MOV(X2, XZR);
        code.MOV(X3, XZR);
        code.MOVZ(X19, 5000);
        code.l(loop);
        code.BL(skip_ret);
        code.ADD(X1, X1, 100);  // skipped
        code.ADD(X2, X2, 1);
        code.BL(br_ret);
        code.ADD(X3, X3, 1);
        code.SUBS(X19, X19, 1);
        code.B(NE, loop);
        code.SVC(0);
        code.l(skip_ret);
        code.ADD(X30, X30, 4);
        code.RET();
        code.l(br_ret);
        code.BR(X30);
    });
    REQUIRE(r.regs[1] == 0);
    REQUIRE(r.regs[2] == 5000);
    REQUIRE(r.regs[3] == 5000);
}

TEST_CASE("A64: host_call_return deep recursion", "[a64][host_call_return]") {
    const Result r = RunBoth([](auto& code) {
        oaknut::Label rec, base, loop;
        code.MOVZ(X19, 3);
        code.MOV(X0, XZR);
        code.l(loop);
        code.MOVZ(X1, 2000);
        code.BL(rec);
        code.SUBS(X19, X19, 1);
        code.B(NE, loop);
        code.SVC(0);
        code.l(rec);
        code.CBZ(X1, base);
        code.STP(X29, X30, SP, PRE_INDEXED, -16);
        code.SUB(X1, X1, 1);
        code.BL(rec);
        code.ADD(X0, X0, 1);
        code.LDP(X29, X30, SP, POST_INDEXED, 16);
        code.l(base);
        code.RET();
    });
    REQUIRE(r.regs[0] == 6000);
    REQUIRE(r.sp == 0x80000);
}

TEST_CASE("A64: host_call_return indirect calls and FPCR changes", "[a64][host_call_return]") {
    // BLR to two callees; one flips FPCR.FZ, so the RET's location (PC + FPCR) differs from the
    // one the call pushed every other time.
    const Result r = RunBoth([](auto& code) {
        oaknut::Label loop, a, b;
        code.MOV(X0, XZR);
        code.MOV(X7, XZR);
        code.MOVZ(X5, 0x0100'0000);  // FPCR.FZ
        code.MOVZ(X19, 300);
        code.ADR(X9, a);
        code.ADR(X10, b);
        code.l(loop);
        code.BLR(X9);
        code.BLR(X10);
        code.MRS(X6, oaknut::SystemReg::FPCR);
        code.ADD(X7, X7, X6, oaknut::AddSubShift::LSR, 24);
        code.SUBS(X19, X19, 1);
        code.B(NE, loop);
        code.SVC(0);
        code.l(a);
        code.ADD(X0, X0, 3);
        code.RET();
        code.l(b);
        code.MRS(X6, oaknut::SystemReg::FPCR);
        code.EOR(X6, X6, X5);
        code.MSR(oaknut::SystemReg::FPCR, X6);
        code.RET();
    });
    REQUIRE(r.regs[0] == 900);
    REQUIRE(r.regs[7] == 150);
}

TEST_CASE("A64: host_call_return spills", "[a64][host_call_return]") {
    // 28 live values across one block: the register allocator spills (into the JIT state when
    // host_call_return is on) inside a callee two frames deep.
    const Result r = RunBoth([](auto& code) {
        oaknut::Label outer, inner;
        code.BL(outer);
        code.SVC(0);
        code.l(outer);
        code.STP(X29, X30, SP, PRE_INDEXED, -16);
        code.BL(inner);
        code.LDP(X29, X30, SP, POST_INDEXED, 16);
        code.RET();
        code.l(inner);
        for (int i = 0; i < 28; ++i) {
            code.MOVZ(oaknut::XReg(i), u16(i * 7 + 1));
        }
        for (int round = 0; round < 3; ++round) {
            for (int i = 0; i < 28; ++i) {
                code.MADD(oaknut::XReg(i), oaknut::XReg(i), oaknut::XReg((i + 5) % 28), oaknut::XReg((i + 11) % 28));
            }
        }
        code.RET();
    });
    REQUIRE(r.sp == 0x80000);
}
