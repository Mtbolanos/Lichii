// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The x86-64 code generation options ArmDynarmic64 turns on (register allocation, zero-extends,
// address folding, block layout, host flags, host call/ret) must not change what guest code
// computes: every program runs with the options off and on and must end in the same state.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#if defined(__linux__) && defined(__x86_64__)
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
extern char** environ;
#endif

#include "common/common_types.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"

namespace {

using namespace Dynarmic;
using Vector = A64::Vector;

constexpr u32 Svc0 = 0xD4000001;

void EnableCodegenOptions(A64::UserConfig& config) {
    config.spill_to_free_gpr = true;
    config.scratch_copy_to_free_reg = true;
    config.elide_redundant_zext = true;
    config.fold_address_offsets = true;
    config.omit_block_frame = true;
    config.fast_dispatch_table_bits = 16;
    config.fast_dispatch_set_associative = true;
    config.rsb_size = 32;
    config.compact_block_exits = true;
    config.far_code = true;
    config.reuse_host_flags = true;
    config.halt_check_on_back_edges = true;
    config.host_call_return = true;
}

struct State {
    std::array<u64, 31> regs;
    std::array<Vector, 32> vecs;
    u64 sp;
    u64 pc;
    u32 nzcv;
    u32 fpsr;
    std::map<u64, u8> memory;
    bool operator==(const State&) const = default;
};

/// FPSR without the cumulative exception bits: host callbacks run under the guest's MXCSR and
/// can leave flags in it (a hash map insert sets IXC) with either setting.
State Capture(A64::Jit& jit, std::map<u64, u8> memory) {
    return State{jit.GetRegisters(), jit.GetVectors(), jit.GetSP(), jit.GetPC(), jit.GetPstate() & 0xF000'0000,
                 jit.GetFpsr() & ~u32{0x9F}, std::move(memory)};
}

/// Memory is a map with a pattern for unwritten bytes; code outside the block is SVC #0, which
/// stops the run (there is no cycle counter to do it).
class RandomBlockEnv final : public A64::UserCallbacks {
public:
    A64::Jit* jit = nullptr;
    std::vector<u32> code;
    u64 code_start = 0;
    std::map<u64, u8> memory;

    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        return vaddr - code_start < code.size() * 4 ? code[(vaddr - code_start) / 4] : Svc0;
    }
    u64 MemoryRead(u64 vaddr, size_t size) override {
        u64 value = 0;
        for (size_t i = 0; i < size; ++i) {
            const auto it = memory.find(vaddr + i);
            value |= u64(it != memory.end() ? it->second : u8(vaddr + i)) << (8 * i);
        }
        return value;
    }
    Vector MemoryRead128(u64 vaddr) override {
        return {MemoryRead(vaddr, 8), MemoryRead(vaddr + 8, 8)};
    }
    void MemoryWrite(u64 vaddr, u64 value, size_t size) override {
        for (size_t i = 0; i < size; ++i) {
            memory[vaddr + i] = u8(value >> (8 * i));
        }
    }
    void MemoryWrite128(u64 vaddr, Vector value) override {
        MemoryWrite(vaddr, value[0], 8);
        MemoryWrite(vaddr + 8, value[1], 8);
    }
    bool MemoryWriteExclusive8(u64 vaddr, u8 value, u8) override { return Write(vaddr, value, 1); }
    bool MemoryWriteExclusive16(u64 vaddr, u16 value, u16) override { return Write(vaddr, value, 2); }
    bool MemoryWriteExclusive32(u64 vaddr, u32 value, u32) override { return Write(vaddr, value, 4); }
    bool MemoryWriteExclusive64(u64 vaddr, u64 value, u64) override { return Write(vaddr, value, 8); }
    bool MemoryWriteExclusive128(u64 vaddr, Vector value, Vector) override {
        MemoryWrite128(vaddr, value);
        return true;
    }
    void CallSVC(std::uint32_t) override { jit->HaltExecution(); }
    void ExceptionRaised(u64, A64::Exception) override {}
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 0; }
    u64 GetCNTPCT() override { return 0; }

private:
    bool Write(u64 vaddr, u64 value, size_t size) {
        MemoryWrite(vaddr, value, size);
        return true;
    }
};

/// One A64 encoding class from the decoder table: fixed bits and the bits that may vary.
struct Encoding {
    u32 bits = 0;
    u32 mask = 0;
};

std::vector<Encoding> TestableEncodings() {
    const std::vector<std::tuple<std::string, const char*>> list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A64/decoder/a64.inc"
#undef INST
    };
    // System instructions read or write host-side state; the translator asserts on some
    // encodings of the complex FP ops; the SM4 S-box needs the table in the low 2 GiB with
    // either setting.
    const std::vector<std::string> excluded{
        "UnallocatedEncoding", "SVC", "BRK", "MSR_reg", "MSR_imm", "MRS", "SYS",
        "FCADD_vec", "FCMLA_vec", "FCMLA_elt", "SM4E", "SM4EKEY",
    };
    std::vector<Encoding> result;
    for (const auto& [fn, bitstring] : list) {
        if (std::find(excluded.begin(), excluded.end(), fn) != excluded.end()) {
            continue;
        }
        Encoding encoding;
        for (size_t i = 0; i < 32; ++i) {
            const u32 bit = 1u << (31 - i);
            if (bitstring[i] == '0' || bitstring[i] == '1') {
                encoding.mask |= bit;
                encoding.bits |= bitstring[i] == '1' ? bit : 0;
            }
        }
        result.push_back(encoding);
    }
    return result;
}

bool ShouldTest(u32 instruction, u64 pc, bool is_last) {
    const A64::LocationDescriptor location{pc, {}};
    IR::Block block{location};
    if (!A64::TranslateSingleInstruction(block, location, instruction) && !is_last) {
        return false;
    }
    return std::none_of(block.instructions.begin(), block.instructions.end(), [](const IR::Inst& inst) {
        switch (inst.GetOpcode()) {
        case IR::Opcode::A64ExceptionRaised:
        case IR::Opcode::A64CallSupervisor:
        case IR::Opcode::A64DataCacheOperationRaised:
        case IR::Opcode::A64InstructionCacheOperationRaised:
        case IR::Opcode::A64GetCNTPCT:
            return true;
        default:
            return false;
        }
    });
}

/// A branch back into the block would loop forever without a cycle counter.
bool BranchesIntoBlock(u32 inst, u64 pc, u64 start, u64 end) {
    const auto target = [&](u32 imm, int bits) {
        return pc + u64(s64(s32(imm << (32 - bits)) >> (32 - bits)) * 4);
    };
    u64 to;
    if ((inst & 0x7C00'0000) == 0x1400'0000) {
        to = target(inst & 0x03FF'FFFF, 26);
    } else if ((inst & 0xFF00'0010) == 0x5400'0000 || (inst & 0x7E00'0000) == 0x3400'0000) {
        to = target((inst >> 5) & 0x7FFFF, 19);
    } else if ((inst & 0x7E00'0000) == 0x3600'0000) {
        to = target((inst >> 5) & 0x3FFF, 14);
    } else {
        return false;
    }
    return to >= start && to <= end;
}

} // namespace

TEST_CASE("JIT codegen options: random blocks", "[jit_codegen_options]") {
    const std::vector<Encoding> encodings = TestableEncodings();
    std::mt19937_64 rng{150};
    const auto rand = [&rng](u64 min, u64 max) { return std::uniform_int_distribution<u64>{min, max}(rng); };

    std::array<RandomBlockEnv, 2> envs;
    std::array<std::optional<ExclusiveMonitor>, 2> monitors;
    std::array<std::optional<A64::Jit>, 2> jits;
    for (size_t i = 0; i < 2; ++i) {
        A64::UserConfig config{};
        config.callbacks = &envs[i];
        config.global_monitor = &monitors[i].emplace(1);
        config.enable_cycle_counting = false;
        if (i == 1) {
            EnableCodegenOptions(config);
        }
        jits[i].emplace(config);
        envs[i].jit = &*jits[i];
    }

    for (size_t iteration = 0; iteration < 1000; ++iteration) {
        std::array<u64, 31> regs;
        std::array<Vector, 32> vecs;
        std::generate(regs.begin(), regs.end(), [&] { return rand(0, ~u64{0}); });
        std::generate(vecs.begin(), vecs.end(), [&] { return Vector{rand(0, ~u64{0}), rand(0, ~u64{0})}; });
        const u64 start = rand(0, 0x10'0000'0000) * 4;
        const size_t count = rand(1, 40);
        std::vector<u32> code(count);
        for (size_t j = 0; j < count; ++j) {
            const u64 pc = start + j * 4;
            do {
                const Encoding& encoding = encodings[rand(0, encodings.size() - 1)];
                code[j] = encoding.bits | (u32(rand(0, ~u32{0})) & ~encoding.mask);
            } while (!ShouldTest(code[j], pc, j == count - 1) || BranchesIntoBlock(code[j], pc, start, start + count * 4));
        }
        const u64 sp = rand(0x30'0000'0000, 0x40'0000'0000) * 4;
        const u32 pstate = u32(rand(0, 0xF)) << 28;
        // AHP, DN, FZ, RMode and FZ16.
        const u32 fpcr = u32(rand(0, ~u32{0})) & 0x07C8'0000;

        std::array<State, 2> states;
        for (size_t i = 0; i < 2; ++i) {
            auto& jit = *jits[i];
            envs[i].code = code;
            envs[i].code_start = start;
            envs[i].memory.clear();
            jit.ClearCache();
            jit.SetRegisters(regs);
            jit.SetVectors(vecs);
            jit.SetSP(sp);
            jit.SetPC(start);
            jit.SetPstate(pstate);
            jit.SetFpcr(fpcr);
            jit.SetFpsr(0);
            jit.Run();
            states[i] = Capture(jit, envs[i].memory);
        }
        INFO("iteration " << iteration << ", " << count << " instructions at 0x" << std::hex << start);
        for (const u32 inst : code) {
            INFO("0x" << std::hex << inst);
        }
        REQUIRE(states[0] == states[1]);
    }
}

#if defined(__linux__) && defined(__x86_64__)

namespace {

constexpr const char* FaultTestName = "JIT codegen options: fastmem faults";
constexpr const char* FaultChildVariable = "LICHII_JIT_FAULT_TEST_CHILD";

constexpr u32 LdrX(u32 rt, u32 rn, u32 imm) { return 0xF9400000 | (imm / 8) << 10 | rn << 5 | rt; }
constexpr u32 LdrW(u32 rt, u32 rn, u32 imm) { return 0xB9400000 | (imm / 4) << 10 | rn << 5 | rt; }
constexpr u32 LdrhW(u32 rt, u32 rn, u32 imm) { return 0x79400000 | (imm / 2) << 10 | rn << 5 | rt; }
constexpr u32 LdrbW(u32 rt, u32 rn, u32 imm) { return 0x39400000 | imm << 10 | rn << 5 | rt; }
constexpr u32 LdurX(u32 rt, u32 rn, s32 imm) { return 0xF8400000 | (u32(imm) & 0x1FF) << 12 | rn << 5 | rt; }
constexpr u32 StrX(u32 rt, u32 rn, u32 imm) { return 0xF9000000 | (imm / 8) << 10 | rn << 5 | rt; }
constexpr u32 StrhW(u32 rt, u32 rn, u32 imm) { return 0x79000000 | (imm / 2) << 10 | rn << 5 | rt; }
constexpr u32 StrbW(u32 rt, u32 rn, u32 imm) { return 0x39000000 | imm << 10 | rn << 5 | rt; }
constexpr u32 AddImm(u32 rd, u32 rn, u32 imm) { return 0x91000000 | imm << 10 | rn << 5 | rd; }
constexpr u32 AddReg(u32 rd, u32 rn, u32 rm) { return 0x8B000000 | rm << 16 | rn << 5 | rd; }

/// Unchecked 64-bit fastmem over a small arena whose page at FaultPage is inaccessible: accesses
/// there fault and take the callbacks, which keep that page's bytes on the side.
class FastmemFaultEnv final : public A64::UserCallbacks {
public:
    static constexpr u64 ArenaSize = 0x10000;
    static constexpr u64 FaultPage = 0x3000;

    A64::Jit* jit = nullptr;
    std::vector<u32> code;
    u8* arena = nullptr;
    std::map<u64, u8> fault_page;

    FastmemFaultEnv() {
        void* const map = mmap(nullptr, ArenaSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        REQUIRE(map != MAP_FAILED);
        arena = static_cast<u8*>(map);
        REQUIRE(mprotect(arena + FaultPage, 0x1000, PROT_NONE) == 0);
        Reset();
    }
    ~FastmemFaultEnv() override {
        munmap(arena, ArenaSize);
    }

    void Reset() {
        for (u64 i = 0; i < ArenaSize; ++i) {
            if (i - FaultPage >= 0x1000) {
                arena[i] = u8(i * 13 + 5);
            }
        }
        fault_page.clear();
    }

    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        return vaddr / 4 < code.size() ? code[vaddr / 4] : Svc0;
    }
    u64 MemoryRead(u64 vaddr, size_t size) override {
        u64 value = 0;
        for (size_t i = 0; i < size; ++i) {
            value |= u64(Read8(vaddr + i)) << (8 * i);
        }
        return value;
    }
    Vector MemoryRead128(u64 vaddr) override {
        return {MemoryRead(vaddr, 8), MemoryRead(vaddr + 8, 8)};
    }
    void MemoryWrite(u64 vaddr, u64 value, size_t size) override {
        for (size_t i = 0; i < size; ++i) {
            Write8(vaddr + i, u8(value >> (8 * i)));
        }
    }
    void MemoryWrite128(u64 vaddr, Vector value) override {
        MemoryWrite(vaddr, value[0], 8);
        MemoryWrite(vaddr + 8, value[1], 8);
    }
    void CallSVC(std::uint32_t) override { jit->HaltExecution(); }
    void ExceptionRaised(u64, A64::Exception) override {}
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 0; }
    u64 GetCNTPCT() override { return 0; }

private:
    u8 Read8(u64 vaddr) {
        if (vaddr - FaultPage < 0x1000) {
            const auto it = fault_page.find(vaddr);
            return it != fault_page.end() ? it->second : u8(vaddr * 7 + 1);
        }
        REQUIRE(vaddr < ArenaSize);
        return arena[vaddr];
    }
    void Write8(u64 vaddr, u8 value) {
        if (vaddr - FaultPage < 0x1000) {
            fault_page[vaddr] = value;
            return;
        }
        REQUIRE(vaddr < ArenaSize);
        arena[vaddr] = value;
    }
};

std::vector<State> RunFastmemProgram(bool options) {
    FastmemFaultEnv env;
    // X0 = 0x2800 and X8 = 0x2900, set before the run so the addresses stay base + immediate:
    // accesses at +0x800 and beyond fault, and X0 is used again after accesses whose fault path
    // moves it by the displacement and back.
    env.code = {
        LdrX(1, 0, 0x10),   LdrX(2, 0, 0x800),  AddImm(11, 0, 3),   LdrbW(3, 0, 0x801),
        LdrhW(4, 0, 0x7FE), LdrW(5, 0, 0x900),  LdurX(10, 0, -8),   StrX(6, 0, 0x808),
        StrbW(7, 0, 0xF),   StrX(8, 8, 0x700),  StrhW(8, 8, 0x702), AddReg(12, 8, 11),
        LdrX(0, 0, 0x810),  Svc0,
    };

    A64::UserConfig config{};
    config.callbacks = &env;
    config.enable_cycle_counting = false;
    config.fastmem_pointer = reinterpret_cast<uintptr_t>(env.arena);
    config.fastmem_address_space_bits = 64;
    config.silently_mirror_fastmem = false;
    if (options) {
        EnableCodegenOptions(config);
    }
    A64::Jit jit{config};
    env.jit = &jit;

    // The first run faults at each access to the page; the second runs the recompiled block,
    // whose accesses there go to the callbacks directly.
    std::vector<State> states;
    for (int run = 0; run < 2; ++run) {
        env.Reset();
        for (size_t i = 0; i < 31; ++i) {
            jit.SetRegister(i, 0x1111'0000'0000'0000 * (i % 3) + i * 0x101);
        }
        jit.SetRegister(0, 0x2800);
        jit.SetRegister(8, 0x2900);
        jit.SetSP(0x8000);
        jit.SetPC(0);
        jit.Run();
        std::map<u64, u8> memory = env.fault_page;
        for (u64 i = 0x2000; i < 0x3000; ++i) {
            memory[i] = env.arena[i];
        }
        states.push_back(Capture(jit, std::move(memory)));
    }
    return states;
}

} // namespace

TEST_CASE(FaultTestName, "[jit_codegen_options]") {
    if (!std::getenv(FaultChildVariable)) {
        // dynarmic installs its fault handler once per process, and Catch replaces the process's
        // SIGSEGV handler around each test case: run this one in a fresh process.
        char* argv[] = {const_cast<char*>("/proc/self/exe"), const_cast<char*>(FaultTestName), nullptr};
        REQUIRE(setenv(FaultChildVariable, "1", 1) == 0);
        pid_t pid{};
        const int spawned = posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, argv, environ);
        unsetenv(FaultChildVariable);
        REQUIRE(spawned == 0);
        int status = 0;
        REQUIRE(waitpid(pid, &status, 0) == pid);
        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) == 0);
        return;
    }
    const auto off = RunFastmemProgram(false);
    const auto on = RunFastmemProgram(true);
    REQUIRE(off[0] == off[1]);
    REQUIRE(on[0] == off[0]);
    REQUIRE(on[1] == off[1]);
}

#endif
