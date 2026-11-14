// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include "core/crypto/ncz_cache.h"
#ifdef _WIN32
#include <windows.h>
#endif

TEST_CASE("NCZ cache preserves random writes and backward reads", "[ncz-cache]") {
    Core::Crypto::NczCache cache;
    REQUIRE(cache.IsOpen());
    std::array<u8, 8> out{};
    REQUIRE(cache.Read(out, 0) == 0);
    const std::array<u8, 4> first{1, 2, 3, 4};
    const std::array<u8, 3> tail{8, 9, 10};
    REQUIRE(cache.Write(first, 0));
    REQUIRE(cache.Write(tail, 4));
    REQUIRE(cache.Read(out, 2) == 5);
    CHECK(out[0] == 3);
    CHECK(out[1] == 4);
    CHECK(out[2] == 8);
    CHECK(out[4] == 10);
    REQUIRE(cache.Read(out, 0) == 7);
    CHECK(out[0] == 1);
    CHECK(out[6] == 10);
}

TEST_CASE("NCZ cache instances never share their bytes", "[ncz-cache]") {
    Core::Crypto::NczCache first;
    Core::Crypto::NczCache second;
    REQUIRE(first.IsOpen());
    REQUIRE(second.IsOpen());
    const std::array<u8, 2> input{42, 43};
    REQUIRE(first.Write(input, 0));
    std::array<u8, 2> out{};
    CHECK(second.Read(out, 0) == 0);
    REQUIRE(first.Read(out, 0) == 2);
    CHECK(out == input);
}

TEST_CASE("NCZ cache rejects unrepresentable offsets", "[ncz-cache]") {
    Core::Crypto::NczCache cache;
    REQUIRE(cache.IsOpen());
    std::array<u8, 2> bytes{5, 6};
    CHECK_FALSE(cache.Write(bytes, std::numeric_limits<u64>::max()));
    CHECK(cache.Read(bytes, std::numeric_limits<u64>::max()) == 0);
    REQUIRE(cache.Write(bytes, 0));
    CHECK(cache.Read(bytes, 0) == 2);
}

#ifdef _WIN32
TEST_CASE("NCZ cache releases native handles on scope exit", "[ncz-cache]") {
    {
        Core::Crypto::NczCache warmup;
        REQUIRE(warmup.IsOpen());
    }
    DWORD before{};
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &before));
    {
        Core::Crypto::NczCache first;
        Core::Crypto::NczCache second;
        REQUIRE(first.IsOpen());
        REQUIRE(second.IsOpen());
        DWORD during{};
        REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &during));
        CHECK(during == before + 2);
    }
    DWORD after{};
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &after));
    CHECK(after == before);
}
#endif
