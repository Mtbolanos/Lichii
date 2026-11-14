// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/fs/path_util.h"

using namespace Common::FS;

TEST_CASE("Lichii keeps its data apart from Eden's", "[fs]") {
    const auto& data = GetEdenPath(EdenPath::EdenDir);
    if (data.filename() == "user") {
        SKIP("portable user directory in use");
    }
    CHECK(data.filename() == "fritchii");
    CHECK(GetLegacyPath(LegacyEdenDir).filename() == "eden");
    CHECK(GetLegacyPath(LegacyEdenDir).parent_path() == data.parent_path());
}
