// SPDX-License-Identifier: GPL-3.0-or-later

#include <optional>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#ifdef _WIN32
#include <windows.h>
#endif

#include "common/host_memory.h"
#include "tests/common/host_memory_test_access.h"

namespace {
#if defined(__OPENORBIS__) || defined(__managarm__)
constexpr size_t FastmemAttemptsPerAllocation = 0;
#else
constexpr size_t FastmemAttemptsPerAllocation = 1;
#endif

#ifdef _WIN32
void CheckMemoryState(void* address, DWORD expected) {
    MEMORY_BASIC_INFORMATION info{};
    REQUIRE(VirtualQuery(address, &info, sizeof(info)) == sizeof(info));
    REQUIRE(info.State == expected);
}
#endif
} // namespace

TEST_CASE("Host memory allocation failure permits a successful retry", "[host_memory_failure]") {
    Common::HostMemoryTestAccess allocations;
    {
        Common::HostMemory memory{0x10000, 0x400000};
        CHECK(memory.BackingBasePointer() == nullptr);
        CHECK(memory.VirtualBasePointer() == nullptr);
        CHECK(allocations.FastmemAttempts() == FastmemAttemptsPerAllocation);
        CHECK(allocations.FallbackAttempts() == 1);
    }

    allocations.AllowFallback();
    {
        Common::HostMemory memory{0x10000, 0x400000};
        REQUIRE(memory.BackingBasePointer() != nullptr);
        CHECK(memory.VirtualBasePointer() == nullptr);
        memory.BackingBasePointer()[0] = 42;
        memory.BackingBasePointer()[0xffff] = 99;
        CHECK(memory.BackingBasePointer()[0] == 42);
        CHECK(memory.BackingBasePointer()[0xffff] == 99);
    }
    CHECK(allocations.FastmemAttempts() == 2 * FastmemAttemptsPerAllocation);
    CHECK(allocations.FallbackAttempts() == 2);
}

#ifdef _WIN32
TEST_CASE("Fallback backing memory is released on Windows", "[host_memory_failure]") {
    Common::HostMemoryTestAccess allocations{false};
    void* backing{};
    {
        Common::HostMemory memory{0x10000, 0x400000};
        backing = memory.BackingBasePointer();
        REQUIRE(backing != nullptr);
    }
    CheckMemoryState(backing, MEM_FREE);
}
#endif

TEST_CASE("Moving fallback memory transfers ownership", "[host_memory_failure]") {
    Common::HostMemoryTestAccess allocations{false};
    std::optional<Common::HostMemory> destination;
    u8* backing{};
    {
        Common::HostMemory source{0x10000, 0x400000};
        backing = source.BackingBasePointer();
        REQUIRE(backing != nullptr);
        backing[0] = 42;
        destination.emplace(std::move(source));
#ifdef _WIN32
        CHECK(source.BackingBasePointer() == nullptr);
#else
        REQUIRE(source.BackingBasePointer() == nullptr);
#endif
        REQUIRE(destination->BackingBasePointer() == backing);
    }
#ifdef _WIN32
    CheckMemoryState(backing, MEM_COMMIT);
#endif
    CHECK(destination->BackingBasePointer()[0] == 42);
    destination.reset();
#ifdef _WIN32
    CheckMemoryState(backing, MEM_FREE);
#endif
}

TEST_CASE("Move assignment releases replaced fallback memory", "[host_memory_failure]") {
    Common::HostMemoryTestAccess allocations{false};
    Common::HostMemory destination{0x10000, 0x400000};
    [[maybe_unused]] u8* replaced = destination.BackingBasePointer();
    REQUIRE(replaced != nullptr);
    u8* backing{};
    {
        Common::HostMemory source{0x10000, 0x400000};
        backing = source.BackingBasePointer();
        REQUIRE(backing != nullptr);
        backing[0] = 99;
        destination = std::move(source);
#ifdef _WIN32
        CHECK(source.BackingBasePointer() == nullptr);
#else
        REQUIRE(source.BackingBasePointer() == nullptr);
#endif
        REQUIRE(destination.BackingBasePointer() == backing);
#ifdef _WIN32
        CheckMemoryState(replaced, MEM_FREE);
#endif
    }
#ifdef _WIN32
    CheckMemoryState(backing, MEM_COMMIT);
#endif
    CHECK(destination.BackingBasePointer()[0] == 99);
    auto& self = destination;
    destination = std::move(self);
    REQUIRE(destination.BackingBasePointer() == backing);
    CHECK(destination.BackingBasePointer()[0] == 99);
}
