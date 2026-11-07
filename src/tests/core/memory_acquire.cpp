// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstring>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"

TEST_CASE("Memory acquire reads and exclusive publication use mapped backing", "[hwacod_memory]") {
    ScopedSystemTestContext context;
    Core::System system;
    Core::Memory::Memory unbound{system};
    u32 value = 99;
    REQUIRE_FALSE(unbound.Read32Acquire(0x4000, value));
    REQUIRE(value == 0);

    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT { system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* page = system.DeviceMemory().buffer.BackingBasePointer() + 0x10000;
    std::memset(page, 0, 0x1000);
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(page) - 0x4000);
    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);
    REQUIRE(memory.Read32Acquire(0x4000, value));
    REQUIRE(value == 0);
    REQUIRE(memory.WriteExclusive32(0x4000, 0x12345678, 0));
    REQUIRE_FALSE(memory.WriteExclusive32(0x4000, 99, 0));
    REQUIRE(memory.Read32Acquire(0x4000, value));
    REQUIRE(value == 0x12345678);
    REQUIRE(memory.WriteExclusive32(0x4ffc, 0xffffffff, 0));
    REQUIRE(memory.Read32Acquire(0x4ffc, value));
    REQUIRE(value == 0xffffffff);
    for (const u64 address : {0x4001ULL, 0x5000ULL, 0x100000ULL}) {
        value = 99;
        REQUIRE_FALSE(memory.Read32Acquire(address, value));
        REQUIRE(value == 0);
    }

    for (u32 iteration = 0; iteration < 32; ++iteration) {
        CAPTURE(iteration);
        const u32 zero{};
        REQUIRE(memory.WriteBlock(0x4000, &zero, sizeof(zero)));
        std::array<u32, 16> expected{};
        expected.fill(iteration + 1);
        bool written{};
        std::jthread writer{[&] {
            written = memory.WriteBlock(0x4080, expected.data(), sizeof(expected)) &&
                      memory.WriteExclusive32(0x4000, 1, 0);
        }};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        bool acquired{};
        do {
            if (!memory.Read32Acquire(0x4000, value)) {
                break;
            }
            acquired = value == 1;
            if (!acquired) {
                std::this_thread::yield();
            }
        } while (!acquired && std::chrono::steady_clock::now() < deadline);
        std::array<u32, 16> observed{};
        const bool read = acquired && memory.ReadBlock(0x4080, observed.data(), sizeof(observed));
        writer.join();
        REQUIRE(written);
        REQUIRE(acquired);
        REQUIRE(read);
        REQUIRE(observed == expected);
    }
}

TEST_CASE("Virtual address ranges that wrap past the address space end are invalid",
          "[hwacod_memory]") {
    ScopedSystemTestContext context;
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT { system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* page = system.DeviceMemory().buffer.BackingBasePointer() + 0x10000;
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(page) - 0x4000);
    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);
    REQUIRE(memory.IsValidVirtualAddressRange(0x4000, 0x1000));
    REQUIRE(memory.IsValidVirtualAddressRange(0x4ff8, 8));
    REQUIRE_FALSE(memory.IsValidVirtualAddressRange(0x4ffc, 8));
    REQUIRE_FALSE(memory.IsValidVirtualAddressRange(~u64{0} - 3, 8));
    REQUIRE_FALSE(memory.IsValidVirtualAddressRange(0x4000, ~u64{0}));
}
