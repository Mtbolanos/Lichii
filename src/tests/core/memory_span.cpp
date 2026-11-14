// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"

TEST_CASE("Memory spans need every page backed by one linear host range", "[core]") {
    ScopedSystemTestContext context;
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT { system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    u8* const backing = system.DeviceMemory().buffer.BackingBasePointer();
    const auto map = [&](u64 page, u16 block, u8* host) {
        table.entries.GetAndFault(page).Store(false, Common::PageType::Memory, block,
                                              reinterpret_cast<uintptr_t>(host) - (page << 12));
    };
    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);

    // Pages 4-5 are one mapping and page 6 another: a span ending on the boundary is one range.
    map(4, 1, backing + 0x10000);
    map(5, 1, backing + 0x11000);
    map(6, 2, backing + 0x40000);
    REQUIRE(memory.GetSpan(0x4000, 0x2000) == backing + 0x10000);
    REQUIRE(memory.GetSpan(0x4800, 0x1800) == backing + 0x10800);
    REQUIRE(memory.GetSpan(0x4000, 0x2001) == nullptr);

    // Pages 8 and 10 keep the block id of one mapping while page 9 was remapped elsewhere.
    map(8, 3, backing + 0x20000);
    map(9, 4, backing + 0x50000);
    map(10, 3, backing + 0x22000);
    REQUIRE(memory.GetSpan(0x8000, 0x2001) == nullptr);
    REQUIRE(memory.GetSpan(0x8000, 0x1000) == backing + 0x20000);

    // Ids are no proof either way: one linear range under different ids is still a span.
    map(12, Common::PageTable::PageEntryData::INVALID_BLOCK_ID, backing + 0x30000);
    map(13, 5, backing + 0x31000);
    REQUIRE(memory.GetSpan(0xC000, 0x2000) == backing + 0x30000);
}
