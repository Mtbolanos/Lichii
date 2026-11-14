// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/bounded_store.h"

using VideoCommon::BoundedStore;

TEST_CASE("BoundedStore drops the oldest values past its size", "[video_core][bounded_store]") {
    BoundedStore<int, int> store{100};
    store.Add(1, 10, 40);
    store.Add(2, 20, 40);
    store.Add(3, 30, 40);
    REQUIRE(store.Size() == 2);
    REQUIRE(store.Bytes() == 80);
    REQUIRE_FALSE(store.Take(1).has_value());
    REQUIRE(store.Take(2) == 20);
    REQUIRE(store.Take(2) == std::nullopt);
    REQUIRE(store.Bytes() == 40);
}

TEST_CASE("BoundedStore ages a value again when it is added again", "[video_core][bounded_store]") {
    BoundedStore<int, int> store{100};
    store.Add(1, 10, 40);
    store.Add(2, 20, 40);
    REQUIRE(store.Take(1) == 10);
    store.Add(1, 11, 40);
    store.Add(3, 30, 40);
    REQUIRE_FALSE(store.Take(2).has_value());
    REQUIRE(store.Take(1) == 11);
    REQUIRE(store.Take(3) == 30);
    REQUIRE(store.Bytes() == 0);
}

TEST_CASE("BoundedStore drops the age records of taken values", "[video_core][bounded_store]") {
    BoundedStore<int, int> store{100};
    for (int i = 0; i < 10'000; ++i) {
        store.Add(1, i, 10);
        REQUIRE(store.Take(1) == i);
    }
    REQUIRE(store.AgeRecords() <= 65);
}

TEST_CASE("BoundedStore keeps nothing larger than its size", "[video_core][bounded_store]") {
    BoundedStore<int, int> store{100};
    store.Add(1, 10, 50);
    store.Add(2, 20, 101);
    REQUIRE(store.Size() == 1);
    REQUIRE(store.Take(1) == 10);
}
