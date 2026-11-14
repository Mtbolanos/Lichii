// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
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

TEST_CASE("NX2 destroyed memory gets a new device range when registered again",
          "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    // Like nvhost_as_gpu: device ranges cached by CPU range, shared through the registry.
    std::map<std::pair<u64, u64>, u64> ranges;
    u64 next_device = 0x100000;
    const auto allocate = [&next_device] {
        const u64 device_address = next_device;
        next_device += 0x20000;
        return device_address;
    };
    registry.AddListener(
        &ranges,
        [&](u64, u64 cpu_address, u64 size) {
            const std::pair key{cpu_address, size};
            if (!ranges.contains(key)) {
                ranges[key] = registry.AcquireDeviceRange(1, cpu_address, size, allocate);
            }
            registry.RecordDeviceRange(cpu_address, size, ranges[key]);
        },
        [](u64, u64) {},
        [&](u64 cpu_address, u64 size) {
            if (ranges.erase(std::pair{cpu_address, size}) != 0) {
                registry.ReleaseDeviceRange(1, cpu_address, size);
            }
        });
    registry.Register(0x10000, 0x4000);
    const u64 first = ranges.at({0x10000, 0x4000});
    REQUIRE(registry.DeviceAddressOf(0x10000) == first);

    // Sub 102, then the game registers new memory at the same address.
    registry.UnmapMemory(0x10000, 0x4000);
    registry.ReleaseBacking(0x10000, 0x4000);
    REQUIRE(ranges.empty());
    REQUIRE_FALSE(registry.DeviceAddressOf(0x10000));
    registry.Register(0x10000, 0x4000);

    const u64 second = ranges.at({0x10000, 0x4000});
    REQUIRE(second != first);
    REQUIRE(registry.DeviceAddressOf(0x10000) == second);
}
namespace {

// Records the calls one listener receives, in order, from any thread.
struct Recorder {
    std::mutex mutex;
    std::vector<std::string> events;

    void Add(std::string event) {
        std::scoped_lock lk{mutex};
        events.push_back(std::move(event));
    }
    std::vector<std::string> Get() {
        std::scoped_lock lk{mutex};
        return events;
    }
};

// Lets a test hold a listener call (a GPU-thread wait in the emulator) until it releases it.
struct Gate {
    std::atomic<bool> entered{false};
    std::atomic<bool> open{false};

    void Hold() {
        entered = true;
        while (!open) {
            std::this_thread::yield();
        }
    }
    void WaitEntered() const {
        while (!entered) {
            std::this_thread::yield();
        }
    }
};

} // namespace

TEST_CASE("NX2 listeners see a map and a later unmap of the same range in order",
          "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    Recorder recorder;
    Gate gate;
    registry.AddListener(
        &recorder,
        [&](u64 gpu_address, u64, u64) {
            gate.Hold();
            recorder.Add("map " + std::to_string(gpu_address));
        },
        [&](u64 gpu_address, u64) { recorder.Add("unmap " + std::to_string(gpu_address)); });

    // The map's listener call is held (as if waiting for the GPU thread) while another thread
    // unmaps the range: the unmap must not reach the listener first.
    std::thread mapper{[&] { registry.MapAt(0x8000, 0x4000, 0x1000); }};
    gate.WaitEntered();
    std::thread unmapper{[&] { registry.Unmap(0x8000, 0x1000); }};
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    REQUIRE(recorder.Get().empty());
    gate.open = true;
    mapper.join();
    unmapper.join();

    const std::vector<std::string> expected{"map 32768", "unmap 32768"};
    REQUIRE(recorder.Get() == expected);
    REQUIRE_FALSE(registry.DeviceAddressOfGpu(0x8000));
}

TEST_CASE("NX2 a new listener's replay comes before a racing unmap", "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    registry.MapAt(0x8000, 0x4000, 0x1000);
    Recorder recorder;
    Gate gate;

    // AddListener replays the existing mapping outside the registry lock; an unmap made while
    // the replay is held must reach the new listener after it, not before (which would leave
    // the replayed mapping behind).
    std::thread adder{[&] {
        registry.AddListener(
            &recorder,
            [&](u64 gpu_address, u64, u64) {
                gate.Hold();
                recorder.Add("map " + std::to_string(gpu_address));
            },
            [&](u64 gpu_address, u64) { recorder.Add("unmap " + std::to_string(gpu_address)); });
    }};
    gate.WaitEntered();
    std::thread unmapper{[&] { registry.Unmap(0x8000, 0x1000); }};
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    REQUIRE(recorder.Get().empty());
    gate.open = true;
    adder.join();
    unmapper.join();

    const std::vector<std::string> expected{"map 32768", "unmap 32768"};
    REQUIRE(recorder.Get() == expected);
}

TEST_CASE("NX2 listeners can call back into the registry", "[video_core][nx2_mapping]") {
    Service::Nvidia::Devices::Nx2GpuMemoryRegistry registry;
    Recorder recorder;
    registry.AddListener(
        &recorder,
        [&](u64 gpu_address, u64 cpu_address, u64 size) {
            // Lookups and further changes from a listener call neither deadlock on the registry
            // lock nor wait for the turn the call itself holds: the nested map follows it.
            REQUIRE(registry.GpuAddressOf(7, 0) == 0x8000U);
            recorder.Add("map " + std::to_string(gpu_address));
            if (gpu_address == 0x8000) {
                registry.MapAt(0x9000, cpu_address + size, size);
                recorder.Add("nested map queued");
            }
        },
        [](u64, u64) {});
    registry.MapAt(7, 0, 0x8000, 0x4000, 0x1000);

    const std::vector<std::string> expected{"map 32768", "nested map queued", "map 36864"};
    REQUIRE(recorder.Get() == expected);
}

TEST_CASE("NX2 changes made in a defer scope reach listeners after it, in order",
          "[video_core][nx2_mapping]") {
    using Registry = Service::Nvidia::Devices::Nx2GpuMemoryRegistry;
    Registry registry;
    Recorder recorder;
    registry.AddListener(
        &recorder,
        [&](u64 gpu_address, u64, u64) { recorder.Add("map " + std::to_string(gpu_address)); },
        [&](u64 gpu_address, u64) { recorder.Add("unmap " + std::to_string(gpu_address)); });
    {
        // Like nx2:gpu under its region mutex: the registry changes now, the listeners (which
        // wait for the GPU thread in the emulator) are called when the scope ends.
        const Registry::DeferScope defer;
        registry.MapAt(0x8000, 0x4000, 0x1000);
        registry.Unmap(0x8000, 0x1000);
        Registry::Defer([&] { recorder.Add("deferred work"); });
        REQUIRE_FALSE(registry.DeviceAddressOfGpu(0x8000));
        REQUIRE(recorder.Get().empty());
    }
    const std::vector<std::string> expected{"map 32768", "unmap 32768", "deferred work"};
    REQUIRE(recorder.Get() == expected);

    // Without a scope, Defer runs right away.
    Registry::Defer([&] { recorder.Add("now"); });
    REQUIRE(recorder.Get().back() == "now");
}

namespace {
using Service::Nvidia::Devices::Nx2EraseContainedRanges;
using Service::Nvidia::Devices::Nx2GpuMemoryRegistry;
using Key = std::pair<u64, u64>;

// Like nvhost_as_gpu: device ranges by CPU range, shared through the registry, freed by the last
// address space that maps them.
struct AddressSpace {
    Nx2GpuMemoryRegistry& registry;
    std::vector<Key>& freed;
    std::map<Key, u64> ranges;

    void Map(u64 cpu_address, u64 size) {
        ranges[{cpu_address, size}] =
            registry.AcquireDeviceRange(1, cpu_address, size, [cpu_address] { return cpu_address << 4; });
    }

    void Release(u64 cpu_address, u64 size) {
        Nx2EraseContainedRanges(ranges, cpu_address, size, [this](u64 range_cpu, u64 range_size, u64) {
            if (registry.ReleaseDeviceRange(1, range_cpu, range_size)) {
                freed.emplace_back(range_cpu, range_size);
            }
        });
    }
};
} // namespace

TEST_CASE("NX2 releasing memory frees every device range inside it",
          "[video_core][nx2_mapping][nx2_release]") {
    Nx2GpuMemoryRegistry registry;
    std::vector<Key> freed;
    AddressSpace as{registry, freed};
    as.Map(0x10000, 0x4000); // the memory itself
    as.Map(0x11000, 0x1000); // 211 maps of parts of it
    as.Map(0x12000, 0x2000);
    as.Map(0x10000, 0x1000);
    as.Map(0x13000, 0x2000); // reaches past its end
    as.Map(0xf000, 0x2000);  // starts before it
    as.Map(0xe000, 0x2000);  // neighbours
    as.Map(0x14000, 0x1000);

    as.Release(0x10000, 0x4000);

    const std::vector<Key> expected_freed{
        {0x10000, 0x1000}, {0x10000, 0x4000}, {0x11000, 0x1000}, {0x12000, 0x2000}};
    REQUIRE(freed == expected_freed);
    const std::vector<Key> expected_kept{{0xe000, 0x2000}, {0xf000, 0x2000}, {0x13000, 0x2000},
                                         {0x14000, 0x1000}};
    std::vector<Key> kept;
    for (const auto& [key, device_address] : as.ranges) {
        kept.push_back(key);
    }
    REQUIRE(kept == expected_kept);
}

TEST_CASE("NX2 releasing an exact range frees only that range",
          "[video_core][nx2_mapping][nx2_release]") {
    Nx2GpuMemoryRegistry registry;
    std::vector<Key> freed;
    AddressSpace as{registry, freed};
    as.Map(0x10000, 0x1000);
    as.Map(0x10000, 0x2000);
    as.Map(0x11000, 0x1000);

    as.Release(0x10000, 0x1000);

    REQUIRE(freed == std::vector<Key>{{0x10000, 0x1000}});
    REQUIRE(as.ranges.size() == 2);
}

TEST_CASE("NX2 a released range another address space maps stays until it releases it too",
          "[video_core][nx2_mapping][nx2_release]") {
    Nx2GpuMemoryRegistry registry;
    std::vector<Key> freed;
    AddressSpace first{registry, freed};
    AddressSpace second{registry, freed};
    first.Map(0x10000, 0x4000);
    first.Map(0x11000, 0x1000);
    second.Map(0x11000, 0x1000);
    REQUIRE(first.ranges.at({0x11000, 0x1000}) == second.ranges.at({0x11000, 0x1000}));

    first.Release(0x10000, 0x4000);
    REQUIRE(freed == std::vector<Key>{{0x10000, 0x4000}});
    REQUIRE(first.ranges.empty());

    second.Release(0x10000, 0x4000);
    REQUIRE(freed == std::vector<Key>{{0x10000, 0x4000}, {0x11000, 0x1000}});
}

TEST_CASE("NX2 releasing memory at the top of the address space does not wrap",
          "[video_core][nx2_mapping][nx2_release]") {
    Nx2GpuMemoryRegistry registry;
    std::vector<Key> freed;
    AddressSpace as{registry, freed};
    constexpr u64 top = 0xffff'ffff'ffff'0000ULL;
    as.Map(top + 0x8000, 0x8000);
    as.Map(top + 0x8000, 0x10000); // would wrap past the end of the address space
    as.Map(0x1000, 0x1000);

    as.Release(top, 0x10000);

    REQUIRE(freed == std::vector<Key>{{top + 0x8000, 0x8000}});
    REQUIRE(as.ranges.size() == 2);
}

TEST_CASE("NX2 released memory loses the device addresses of the ranges inside it",
          "[video_core][nx2_mapping][nx2_release]") {
    Nx2GpuMemoryRegistry registry;
    registry.RecordDeviceRange(0x10000, 0x1000, 0x100000);
    registry.RecordDeviceRange(0x11000, 0x1000, 0x200000);
    registry.RecordDeviceRange(0x13000, 0x2000, 0x300000); // reaches past the memory
    registry.RecordDeviceRange(0x14000, 0x1000, 0x400000); // neighbour

    registry.ReleaseBacking(0x10000, 0x4000);

    REQUIRE_FALSE(registry.DeviceAddressOf(0x10000));
    REQUIRE_FALSE(registry.DeviceAddressOf(0x11000));
    REQUIRE(registry.DeviceAddressOf(0x13800) == 0x300800U);
    REQUIRE(registry.DeviceAddressOf(0x14000) == 0x400000U);
}
