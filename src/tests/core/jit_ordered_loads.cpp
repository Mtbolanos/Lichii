// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ordered (LDAR, LDAPR) and exclusive (LDAXR) loads through the JIT's fastmem path. On x86-64 a
// load is already ordered before later loads and stores, so these are plain loads: they must not
// write the page they read, which a locked read-modify-write did.
//
// The microbenchmark is hidden from normal runs; start it with
//   tests "[.jit_ordered_load_benchmark]"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "common/common_types.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace {

constexpr u32 AddressSpaceBits = 20;
constexpr size_t ArenaSize = size_t{1} << AddressSpaceBits;
constexpr size_t PageSize = 0x10000; // A multiple of every host's page size
constexpr u64 Writable = PageSize;
constexpr u64 ReadOnly = 2 * PageSize;
constexpr u64 CodeBase = 8 * PageSize;

constexpr u32 Svc = 0xD4000001;
constexpr u32 AddX1X1One = 0x91000421;  // ADD X1, X1, #1
constexpr u32 SubsX2X2One = 0xF1000442; // SUBS X2, X2, #1
constexpr u32 BranchNotEqualBack2 = 0x54FFFFC1; // B.NE to two instructions back

constexpr u32 Ldar(u32 size, u32 rn, u32 rt) {
    return 0x08DFFC00 | (size << 30) | (rn << 5) | rt;
}
constexpr u32 Ldapr(u32 size, u32 rn, u32 rt) {
    return 0x38BFC000 | (size << 30) | (rn << 5) | rt;
}
constexpr u32 Ldaxr(u32 size, u32 rn, u32 rt) {
    return 0x085FFC00 | (size << 30) | (rn << 5) | rt;
}
constexpr u32 Stlxr(u32 size, u32 rs, u32 rn, u32 rt) {
    return 0x0800FC00 | (size << 30) | (rs << 16) | (rn << 5) | rt;
}
/// LDADDAL: adds Rs to the memory at Rn and loads the old value into Rt.
constexpr u32 Ldaddal(u32 size, u32 rs, u32 rn, u32 rt) {
    return 0x38E00000 | (size << 30) | (rs << 16) | (rn << 5) | rt;
}

/// Host memory the guest addresses [0, ArenaSize) map onto.
class Arena {
public:
    Arena() {
#ifdef _WIN32
        base = static_cast<u8*>(
            VirtualAlloc(nullptr, ArenaSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        REQUIRE(base != nullptr);
#else
        void* const mapping = mmap(nullptr, ArenaSize, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        REQUIRE(mapping != MAP_FAILED);
        base = static_cast<u8*>(mapping);
#endif
    }
    ~Arena() {
#ifdef _WIN32
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, ArenaSize);
#endif
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void ProtectReadOnly(u64 address) {
#ifdef _WIN32
        DWORD old{};
        REQUIRE(VirtualProtect(base + address, PageSize, PAGE_READONLY, &old) != 0);
#else
        REQUIRE(mprotect(base + address, PageSize, PROT_READ) == 0);
#endif
    }

    u8* base{};
};

class Environment final : public Dynarmic::A64::UserCallbacks {
public:
    Dynarmic::A64::Jit* jit{};
    u8* memory{};
    std::vector<u32> code;
    /// Memory accesses the JIT made through a callback instead of inline.
    u64 callbacks{};

    std::optional<u32> MemoryReadCode(u64 address) override {
        const u64 index = (address - CodeBase) / 4;
        if (address < CodeBase || index >= code.size()) {
            return std::nullopt;
        }
        return code[index];
    }
    u64 MemoryRead(u64 address, size_t size) override {
        ++callbacks;
        u64 value{};
        std::memcpy(&value, memory + address, size);
        return value;
    }
    Dynarmic::A64::Vector MemoryRead128(u64) override {
        FAIL("Unexpected 128-bit read");
        return {};
    }
    void MemoryWrite(u64 address, u64 value, size_t size) override {
        ++callbacks;
        std::memcpy(memory + address, &value, size);
    }
    void MemoryWrite128(u64, Dynarmic::A64::Vector) override {
        FAIL("Unexpected 128-bit write");
    }
    bool MemoryWriteExclusive64(u64 address, u64 value, u64 expected) override {
        ++callbacks;
        return reinterpret_cast<std::atomic<u64>*>(memory + address)
            ->compare_exchange_strong(expected, value);
    }
    void CallSVC(u32) override {
        jit->HaltExecution();
    }
    void ExceptionRaised(u64, Dynarmic::A64::Exception) override {
        FAIL("Unexpected exception");
    }
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override {
        return 1000;
    }
    u64 GetCNTPCT() override {
        return 0;
    }
};

/// The emulator's configuration at the default CPU accuracy, over a small arena.
Dynarmic::A64::UserConfig Config(Environment& env, Dynarmic::ExclusiveMonitor& monitor,
                                 size_t processor) {
    Dynarmic::A64::UserConfig config;
    config.callbacks = &env;
    config.enable_cycle_counting = false;
    config.fastmem_pointer = reinterpret_cast<std::uintptr_t>(env.memory);
    config.fastmem_address_space_bits = AddressSpaceBits;
    config.silently_mirror_fastmem = false;
    config.fastmem_exclusive_access = true;
    config.global_monitor = &monitor;
    config.processor_id = static_cast<u8>(processor);
    config.unsafe_optimizations = true;
    config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
    return config;
}

void Store(Arena& arena, u64 address, u64 value) {
    std::memcpy(arena.base + address, &value, sizeof(value));
}
u64 Load(const Arena& arena, u64 address) {
    u64 value{};
    std::memcpy(&value, arena.base + address, sizeof(value));
    return value;
}

void Run(Environment& env, Dynarmic::A64::Jit& jit) {
    env.jit = &jit;
    jit.SetPC(CodeBase);
    jit.Run();
}

} // namespace

TEST_CASE("Ordered loads read a read-only page inline", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, ReadOnly, 0x8877665544332211ULL);
    arena.ProtectReadOnly(ReadOnly);

    Environment env;
    env.memory = arena.base;
    env.code = {Ldar(0, 0, 1),  Ldar(1, 0, 2),  Ldar(2, 0, 3),  Ldar(3, 0, 4), Ldapr(0, 0, 5),
                Ldapr(1, 0, 6), Ldapr(2, 0, 7), Ldapr(3, 0, 8), Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, ReadOnly);
    for (size_t reg = 1; reg <= 8; ++reg) {
        jit.SetRegister(reg, ~0ULL);
    }
    Run(env, jit);

    constexpr std::array<u64, 4> expected{0x11, 0x2211, 0x44332211, 0x8877665544332211ULL};
    for (size_t size = 0; size < expected.size(); ++size) {
        REQUIRE(jit.GetRegister(1 + size) == expected[size]);
        REQUIRE(jit.GetRegister(5 + size) == expected[size]);
    }
    REQUIRE(Load(arena, ReadOnly) == 0x8877665544332211ULL);
#ifdef ARCHITECTURE_x86_64
    REQUIRE(env.callbacks == 0);
#endif
}

TEST_CASE("Exclusive loads read a read-only page inline", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, ReadOnly, 0x0123456789ABCDEFULL);
    arena.ProtectReadOnly(ReadOnly);

    Environment env;
    env.memory = arena.base;
    env.code = {Ldaxr(3, 0, 1), Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, ReadOnly);
    Run(env, jit);

    REQUIRE(jit.GetRegister(1) == 0x0123456789ABCDEFULL);
#ifdef ARCHITECTURE_x86_64
    REQUIRE(env.callbacks == 0);
#endif
}

TEST_CASE("Exclusive and atomic updates still store through plain loads", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, Writable, 41);

    Environment env;
    env.memory = arena.base;
    // An exclusive increment, then an atomic add of X3 that loads the old value into X4.
    env.code = {Ldaxr(3, 0, 1), AddX1X1One, Stlxr(3, 2, 0, 1), Ldaddal(3, 3, 0, 4), Ldar(3, 0, 5),
                Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, Writable);
    jit.SetRegister(2, ~0ULL);
    jit.SetRegister(3, 100);
    Run(env, jit);

    REQUIRE(jit.GetRegister(1) == 42);
    REQUIRE(jit.GetRegister(2) == 0);
    REQUIRE(jit.GetRegister(4) == 42);
    REQUIRE(jit.GetRegister(5) == 142);
    REQUIRE(Load(arena, Writable) == 142);
}

TEST_CASE("Each processor keeps its own reservation", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, Writable, 7);

    // Inline exclusives on the last processor of the monitor.
    Environment env;
    env.memory = arena.base;
    env.code = {Ldaxr(3, 0, 1), AddX1X1One, Stlxr(3, 2, 0, 1), Svc};
    Dynarmic::ExclusiveMonitor monitor{4};
    Dynarmic::A64::Jit jit{Config(env, monitor, 3)};
    jit.SetRegister(0, Writable);
    jit.SetRegister(2, ~0ULL);
    Run(env, jit);
    REQUIRE(jit.GetRegister(2) == 0);
    REQUIRE(Load(arena, Writable) == 8);

    // The monitor's own interface, as the kernel uses it: a store through one processor's
    // reservation ends the others' on that address and leaves the rest.
    const auto mark = [&](size_t processor, u64 address) {
        return monitor.ReadAndMark<u64>(processor, address, [&] { return Load(arena, address); });
    };
    const auto store = [&](size_t processor, u64 address) {
        u64 saved{};
        const bool stored =
            monitor.DoExclusiveOperation<u64>(processor, address, [&](u64 expected) {
                saved = expected;
                return true;
            });
        return stored ? saved : ~0ULL;
    };
    Store(arena, Writable + 8, 21);
    REQUIRE(mark(0, Writable) == 8);
    REQUIRE(mark(1, Writable) == 8);
    REQUIRE(mark(2, Writable + 8) == 21);
    REQUIRE(store(1, Writable) == 8);
    REQUIRE(store(0, Writable) == ~0ULL);
    REQUIRE(store(2, Writable + 8) == 21);
    REQUIRE(store(2, Writable + 8) == ~0ULL);
}

TEST_CASE("Ordered load microbenchmark", "[.jit_ordered_load_benchmark]") {
    constexpr u64 Iterations = 20'000'000;
    struct Loop {
        const char* name;
        u32 instruction;
    };
    constexpr std::array loops{
        Loop{"LDAR", Ldar(3, 0, 1)},
        Loop{"LDAPR", Ldapr(3, 0, 1)},
        Loop{"LDADDAL", Ldaddal(3, 3, 0, 4)},
    };
    struct Case {
        size_t threads;
        bool shared;
    };
    constexpr std::array cases{Case{1, true}, Case{2, true}, Case{4, true}, Case{2, false},
                               Case{4, false}};
    for (const Loop& loop : loops) {
        // Sharing one word shows the instruction's own cost between cores; a word each, a page
        // apart, shows what the cores share besides guest memory.
        for (const Case& test : cases) {
            const size_t threads = test.threads;
            const bool shared = test.shared;
            Arena arena;
            Dynarmic::ExclusiveMonitor monitor{threads};
            std::atomic<bool> start{};
            std::vector<double> nanoseconds(threads);
            std::vector<std::thread> workers;
            for (size_t thread = 0; thread < threads; ++thread) {
                workers.emplace_back([&, thread] {
                    Environment env;
                    env.memory = arena.base;
                    env.code = {loop.instruction, SubsX2X2One, BranchNotEqualBack2, Svc};
                    Dynarmic::A64::Jit jit{Config(env, monitor, thread)};
                    jit.SetRegister(0, shared ? Writable : Writable + thread * 0x1000);
                    jit.SetRegister(2, Iterations);
                    jit.SetRegister(3, 1);
                    while (!start.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    const auto begin = std::chrono::steady_clock::now();
                    Run(env, jit);
                    const std::chrono::duration<double, std::nano> elapsed =
                        std::chrono::steady_clock::now() - begin;
                    nanoseconds[thread] = elapsed.count() / static_cast<double>(Iterations);
                });
            }
            start.store(true, std::memory_order_release);
            double total{};
            for (size_t thread = 0; thread < threads; ++thread) {
                workers[thread].join();
                total += nanoseconds[thread];
            }
            std::printf("%-8s %zu thread(s) on %s: %.2f ns per iteration\n", loop.name, threads,
                        shared ? "one word" : "a word each", total / static_cast<double>(threads));
        }
    }
}
