// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/object_pool.h"

namespace {
struct Counted {
    static inline int live = 0;

    explicit Counted(int value_) : value{value_} {
        ++live;
    }
    ~Counted() {
        --live;
    }

    int value;
};
} // Anonymous namespace

TEST_CASE("ObjectPool::Free destroys every object across chunks", "[shader]") {
    Counted::live = 0;
    {
        Shader::ObjectPool<Counted> pool{4};
        for (int index = 0; index < 10; ++index) {
            REQUIRE(pool.Create(index)->value == index);
        }
        REQUIRE(Counted::live == 10);
        pool.Free();
        REQUIRE(Counted::live == 0);
    }
    REQUIRE(Counted::live == 0);
}

TEST_CASE("ObjectPool stays usable after Free", "[shader]") {
    Counted::live = 0;
    {
        Shader::ObjectPool<Counted> pool{4};
        for (int index = 0; index < 6; ++index) {
            (void)pool.Create(index);
        }
        pool.Free();
        Counted* const first{pool.Create(100)};
        for (int index = 0; index < 8; ++index) {
            (void)pool.Create(index);
        }
        REQUIRE(first->value == 100);
        REQUIRE(Counted::live == 9);
        pool.ReleaseContents();
        REQUIRE(Counted::live == 0);
        (void)pool.Create(7);
        REQUIRE(Counted::live == 1);
    }
    REQUIRE(Counted::live == 0);
}
