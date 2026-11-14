// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <functional>
#include <random>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/gpu_dirty_memory_manager.h"

// Ported from jpkto/Lichii 7a5e5357cb (src/tests/video_core/nx2_backx.cpp).

namespace {

/// The 64-byte blocks of [address, address + size).
void AddBlocks(std::set<u64>& blocks, u64 address, u64 size) {
    for (u64 block = address >> 6; block <= (address + size - 1) >> 6; ++block) {
        blocks.insert(block);
    }
}

} // Anonymous namespace

TEST_CASE("BackX gather: a manager shows a collected write until it is gathered",
          "[core][nx2_backx]") {
    Core::GPUDirtyMemoryManager manager;
    std::vector<std::pair<PAddr, size_t>> gathered;
    std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
        gathered.emplace_back(address, size);
    };
    REQUIRE_FALSE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE(gathered.empty());
    REQUIRE_FALSE(manager.MayHavePending());

    // In `current`.
    manager.Collect(0x10040, 0x10);
    REQUIRE(manager.MayHavePending());
    manager.Collect(0x10080, 0x8);
    REQUIRE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE_FALSE(gathered.empty());
    REQUIRE_FALSE(manager.MayHavePending());

    // In back_buffer and `current`.
    gathered.clear();
    manager.Collect(0x20000, 0x40);
    manager.Collect(0x30000, 0x40);
    manager.Collect(0x40000, 0x40);
    REQUIRE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE(gathered.size() == 3);
    REQUIRE_FALSE(manager.MayHavePending());
    gathered.clear();
    manager.Gather(callback);
    REQUIRE(gathered.empty());

    // A write already collected collects nothing new, and still shows until gathered.
    manager.Collect(0x50000, 0x4);
    manager.Collect(0x50000, 0x4);
    REQUIRE(manager.MayHavePending());
    manager.Gather(callback);
    REQUIRE(gathered.size() == 1);
    REQUIRE_FALSE(manager.MayHavePending());
}

TEST_CASE("BackX gather: gathering only when a manager shows a write loses none",
          "[core][nx2_backx]") {
    // One collecting thread per manager, as each guest core has its own; the GPU thread gathers
    // only when MayHavePending says so, at the same time.
    for (u64 seed = 1; seed <= 8; ++seed) {
        Core::GPUDirtyMemoryManager manager;
        std::set<u64> collected;
        std::set<u64> gathered;
        std::atomic<bool> done{false};
        u64 checks = 0;
        std::thread collector([&] {
            std::mt19937_64 rng{seed};
            for (int i = 0; i < 300000; ++i) {
                // A few 2 KiB pages, so both the compare-exchange and the push path run.
                const u64 page = rng() % (seed % 2 == 0 ? 3 : 64);
                const u64 offset = rng() % 2040;
                const u64 size = 1 + rng() % 8;
                const u64 address = 0x100000 + page * 2048 + offset;
                manager.Collect(address, size);
                AddBlocks(collected, address, size);
            }
            done.store(true, std::memory_order_release);
        });
        std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
            AddBlocks(gathered, address, size);
        };
        while (!done.load(std::memory_order_acquire)) {
            ++checks;
            if (manager.MayHavePending()) {
                manager.Gather(callback);
            }
        }
        collector.join();
        if (manager.MayHavePending()) {
            manager.Gather(callback);
        }
        REQUIRE_FALSE(manager.MayHavePending());
        const size_t before = gathered.size();
        manager.Gather(callback);
        REQUIRE(gathered.size() == before);
        REQUIRE(std::ranges::includes(gathered, collected));
        REQUIRE(checks > 0);
    }
}

TEST_CASE("BackX gather: a write crossing a 2 KiB page is collected whole and shows",
          "[core][nx2_backx]") {
    // A transform covers 2 KiB in 64-byte bits. A write past its page's 32 bits must still show
    // (MayHavePending) and be gathered to its last byte.
    Core::GPUDirtyMemoryManager manager;
    std::vector<std::pair<PAddr, size_t>> gathered;
    std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
        gathered.emplace_back(address, size);
    };
    for (const auto& [offset, size] :
         {std::pair<u64, u64>{600, 1960}, {0, 4096}, {1000, 3000}, {2047, 2}, {64, 2048}}) {
        CAPTURE(offset, size);
        manager.Collect(0x7000, 0x40); // Something already collected on another page
        manager.Collect(0x10000 + offset, size);
        REQUIRE(manager.MayHavePending());
        gathered.clear();
        manager.Gather(callback);
        std::set<u64> blocks;
        for (const auto& [address, length] : gathered) {
            AddBlocks(blocks, address, length);
        }
        for (u64 block = (0x10000 + offset) >> 6; block <= (0x10000 + offset + size - 1) >> 6;
             ++block) {
            REQUIRE(blocks.contains(block));
        }
        REQUIRE_FALSE(manager.MayHavePending());
    }
}

TEST_CASE("BackX gather: Collects counts the collects that add bits", "[core][nx2_backx]") {
    // The verify tells a race from a miss by this count, so it must move for every collect a
    // gather can find (a page split counting each page) and only when counting is on.
    Core::GPUDirtyMemoryManager manager;
    manager.Collect(0x10000, 0x40);
    REQUIRE(manager.Collects() == 0);
    Core::GPUDirtyMemoryManager::count_collects = true;
    manager.Collect(0x10000, 0x40);
    REQUIRE(manager.Collects() == 0);
    manager.Collect(0x10040, 0x40);
    REQUIRE(manager.Collects() == 1);
    manager.Collect(0x7000, 0x40);
    REQUIRE(manager.Collects() == 2);
    manager.Collect(0x107c0, 0x80);
    REQUIRE(manager.Collects() == 4);
    Core::GPUDirtyMemoryManager::count_collects = false;
}
