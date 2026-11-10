// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2017 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <math.h>
#include <utility>
#include "common/logging.h"
#include "common/param_package.h"

namespace Common {

TEST_CASE("ParamPackage", "[common]") {
    ParamPackage original{
        {"abc", "xyz"},
        {"def", "42"},
        {"jkl", "$$:1:$2$,3"},
    };
    original.Set("ghi", 3.14f);
    ParamPackage copy(original.Serialize());
    REQUIRE(copy.Get("abc", "") == "xyz");
    REQUIRE(copy.Get("def", 0) == 42);
    REQUIRE(std::abs(copy.Get("ghi", 0.0f) - 3.14f) < 0.01f);
    REQUIRE(copy.Get("jkl", "") == "$$:1:$2$,3");
    REQUIRE(copy.Get("mno", "uvw") == "uvw");
    REQUIRE(copy.Get("abc", 42) == 42);
}

TEST_CASE("ParamPackage preserves empty values", "[common][param_package]") {
    const auto [serialized, key] =
        GENERATE(std::pair{"key:", "key"}, std::pair{"key:,other:value", "key"},
                 std::pair{"other:value,key:", "key"}, std::pair{"key:,other:", "key"},
                 std::pair{"$0$1$2:", ":,$"}, std::pair{":", ""});
    ParamPackage parsed(serialized);
    REQUIRE(parsed.Has(key));
    REQUIRE(parsed.Get(key, "fallback").empty());

    ParamPackage copy(parsed.Serialize());
    REQUIRE(copy.Has(key));
    REQUIRE(copy.Get(key, "fallback").empty());
    if (parsed.Has("other")) {
        REQUIRE(copy.Get("other", "fallback") == parsed.Get("other", "fallback"));
    }
}

TEST_CASE("ParamPackage round trips an explicitly empty value", "[common][param_package]") {
    ParamPackage original{{"key", ""}, {"other", "value"}};
    ParamPackage copy(original.Serialize());
    REQUIRE(copy.Has("key"));
    REQUIRE(copy.Get("key", "fallback").empty());
    REQUIRE(copy.Get("other", "fallback") == "value");
}

TEST_CASE("ParamPackage skips malformed pairs", "[common][param_package]") {
    ParamPackage parsed("missing_separator,too:many:separators,trailing:value:,valid:value");
    REQUIRE_FALSE(parsed.Has("missing_separator"));
    REQUIRE_FALSE(parsed.Has("too"));
    REQUIRE_FALSE(parsed.Has("trailing"));
    REQUIRE(parsed.Get("valid", "fallback") == "value");
}

TEST_CASE("ParamPackage preserves a cleared stick modifier", "[common][param_package]") {
    ParamPackage stick{{"engine", "analog_from_button"}};
    stick.Set("modifier", "");
    ParamPackage copy(stick.Serialize());
    const std::string null_engine = ParamPackage{{"engine", "null"}}.Serialize();
    REQUIRE(copy.Has("modifier"));
    REQUIRE(copy.Get("modifier", null_engine).empty());
    REQUIRE(copy.Get("missing", null_engine) == null_engine);
    ParamPackage modifier(copy.Get("modifier", null_engine));
    REQUIRE(modifier.Get("engine", "null") == "null");
}

TEST_CASE("ParamPackage empty numeric values still use defaults", "[common][param_package]") {
    ParamPackage parsed("integer:,float:");
    REQUIRE(parsed.Has("integer"));
    REQUIRE(parsed.Has("float"));
    REQUIRE(parsed.Get("integer", 42) == 42);
    REQUIRE(parsed.Get("float", 1.25f) == 1.25f);
}

} // namespace Common
