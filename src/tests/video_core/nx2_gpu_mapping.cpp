// SPDX-License-Identifier: GPL-3.0-or-later

#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nvdrv/devices/nx2_stub_device.h"

TEST_CASE("NX2 presentation resolves parent handles and image offsets", "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    registry.MapAt(7, 0x1000, 0x8000, 0x4000, 0x3000);
    registry.RecordDeviceRange(0x4000, 0x3000, 0x10000);
    REQUIRE(registry.GpuAddressOf(7, 0x1800) == 0x8800U);
    REQUIRE(registry.DeviceAddressOfGpu(0x8800) == 0x10800U);
    REQUIRE_FALSE(registry.GpuAddressOf(7, 0xfff));
    REQUIRE_FALSE(registry.GpuAddressOf(7, 0x4000));
    REQUIRE_FALSE(registry.GpuAddressOf(8, 0x1800));
    REQUIRE_FALSE(registry.DeviceAddressOfGpu(0xb000));
}

TEST_CASE("NX2 partial unmaps preserve remaining parent offsets", "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    registry.MapAt(7, 0x1000, 0x8000, 0x4000, 0x3000);
    registry.RecordDeviceRange(0x4000, 0x3000, 0x10000);
    registry.Unmap(0x9000, 0x1000);
    REQUIRE(registry.GpuAddressOf(7, 0x1000) == 0x8000U);
    REQUIRE_FALSE(registry.GpuAddressOf(7, 0x2000));
    REQUIRE_FALSE(registry.DeviceAddressOfGpu(0x9000));
    REQUIRE(registry.GpuAddressOf(7, 0x3000) == 0xa000U);
    REQUIRE(registry.DeviceAddressOfGpu(0xa000) == 0x12000U);
}

TEST_CASE("NX2 remaps and resets discard stale parent handles", "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    registry.MapAt(7, 0, 0x8000, 0x4000, 0x1000);
    registry.MapAt(8, 0x1000, 0x8000, 0x6000, 0x1000);
    registry.RecordDeviceRange(0x6000, 0x1000, 0x20000);
    REQUIRE_FALSE(registry.GpuAddressOf(7, 0));
    REQUIRE(registry.GpuAddressOf(8, 0x1000) == 0x8000U);
    REQUIRE(registry.DeviceAddressOfGpu(0x8000) == 0x20000U);
    registry.Reset();
    REQUIRE_FALSE(registry.GpuAddressOf(8, 0x1000));
    REQUIRE_FALSE(registry.DeviceAddressOfGpu(0x8000));
}

TEST_CASE("NX2 destroyed memory leaves no GPU range mapped into it", "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    std::vector<std::pair<u64, u64>> unmapped;
    registry.AddListener(
        &unmapped, [](u64, u64, u64) {},
        [&unmapped](u64 gpu_address, u64 size) { unmapped.emplace_back(gpu_address, size); });
    registry.Register(0x10000, 0x4000);
    registry.MapAt(5, 0x1000, 0x80000, 0x11000, 0x2000);
    registry.MapAt(6, 0, 0x90000, 0x20000, 0x1000);
    registry.MapAt(7, 0, 0xa0000, 0x13000, 0x3000);

    const auto leftovers = registry.UnmapMemory(0x10000, 0x4000);

    const std::vector<std::pair<u64, u64>> expected_leftovers{{0x80000, 0x2000}, {0xa0000, 0x1000}};
    REQUIRE(leftovers == expected_leftovers);
    const std::vector<std::pair<u64, u64>> expected_unmapped{
        {0x10000, 0x4000}, {0x80000, 0x2000}, {0xa0000, 0x1000}};
    REQUIRE(unmapped == expected_unmapped);
    REQUIRE_FALSE(registry.GpuAddressOf(5, 0x1000));
    REQUIRE(registry.GpuAddressOf(6, 0) == 0x90000U);
    // Only the part of a mapping that reached the destroyed memory goes.
    REQUIRE_FALSE(registry.GpuAddressOf(7, 0));
    REQUIRE(registry.GpuAddressOf(7, 0x1000) == 0xa1000U);
}
