// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/renderer_vulkan/descriptor_payload_growth.h"

using Vulkan::GrowDescriptorPayload;
using Vulkan::PayloadGrowthResult;

TEST_CASE("Descriptor growth keeps queued payload alive until drained", "[descriptor_payload]") {
    size_t capacity = 4;
    auto storage = std::make_unique<int[]>(capacity * 8);
    storage[3] = 123;
    int* const queued = storage.get() + 3;
    bool drained = false;
    const auto result = GrowDescriptorPayload(
        storage, capacity, 16, 8, 5,
        [&] {
            REQUIRE(*queued == 123);
            REQUIRE(capacity == 4);
            drained = true;
        },
        [&](size_t count) {
            REQUIRE_FALSE(drained);
            REQUIRE(count == 64);
            return std::make_unique<int[]>(count);
        });
    REQUIRE(result == PayloadGrowthResult::Grown);
    REQUIRE(drained);
    REQUIRE(capacity == 8);
}

TEST_CASE("Descriptor growth satisfies large reservations within the ceiling",
          "[descriptor_payload]") {
    size_t capacity = 4;
    auto storage = std::make_unique<int[]>(32);
    unsigned waits = 0;
    const auto result = GrowDescriptorPayload(
        storage, capacity, 16, 8, 15, [&] { ++waits; },
        [](size_t count) { return std::make_unique<int[]>(count); });
    REQUIRE(result == PayloadGrowthResult::Grown);
    REQUIRE(capacity == 16);
    REQUIRE(waits == 1);
    storage[capacity * 8 - 1] = 42;
    REQUIRE(storage[127] == 42);
    REQUIRE(GrowDescriptorPayload(
                storage, capacity, 16, 8, 17, [&] { ++waits; },
                [](size_t count) { return std::make_unique<int[]>(count); }) ==
            PayloadGrowthResult::AtLimit);
    REQUIRE(waits == 1);
}

TEST_CASE("Descriptor allocation failure preserves reusable storage", "[descriptor_payload]") {
    size_t capacity = 4;
    auto storage = std::make_unique<int[]>(32);
    storage[0] = 99;
    int* const original = storage.get();
    unsigned waits = 0;
    REQUIRE(GrowDescriptorPayload(
                storage, capacity, 16, 8, 5, [&] { ++waits; },
                [](size_t) -> std::unique_ptr<int[]> { throw std::bad_alloc{}; }) ==
            PayloadGrowthResult::AllocationFailed);
    REQUIRE(capacity == 4);
    REQUIRE(storage.get() == original);
    REQUIRE(storage[0] == 99);
    REQUIRE(waits == 0);
}

TEST_CASE("Descriptor growth clips doubling to its configured ceiling", "[descriptor_payload]") {
    size_t capacity = 8;
    auto storage = std::make_unique<int[]>(64);
    REQUIRE(GrowDescriptorPayload(
                storage, capacity, 12, 8, 9, [] {},
                [](size_t count) {
                    REQUIRE(count == 96);
                    return std::make_unique<int[]>(count);
                }) == PayloadGrowthResult::Grown);
    REQUIRE(capacity == 12);
    REQUIRE(GrowDescriptorPayload(
                storage, capacity, 12, 8, 12, [] { FAIL("Must not drain at the ceiling"); },
                [](size_t) -> std::unique_ptr<int[]> {
                    FAIL("Must not allocate at ceiling");
                    return {};
                }) == PayloadGrowthResult::AtLimit);
}
