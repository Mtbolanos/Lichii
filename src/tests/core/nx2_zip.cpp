// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/debugger/nx2_zip.h"

using Core::Nx2Debug::ZipWriter;

namespace {

std::vector<u8> Bytes(std::string_view text) {
    return {text.begin(), text.end()};
}

} // Anonymous namespace

TEST_CASE("Bug bundle zip sizes match what is written", "[core][nx2_zip]") {
    const auto path = std::filesystem::temp_directory_path() / "lichii-nx2-zip-test.zip";
    const std::vector<u8> repeated = Bytes(std::string(4096, 'a'));
    const std::vector<u8> short_text = Bytes("ab");

    const ZipWriter::Entry deflated = ZipWriter::Prepare("logs/eden_log.txt", repeated, true);
    REQUIRE(deflated.deflated);
    REQUIRE(deflated.data.size() < repeated.size());
    const ZipWriter::Entry stored = ZipWriter::Prepare("bundle.json", short_text, true);
    REQUIRE_FALSE(stored.deflated);

    u64 size{};
    {
        ZipWriter zip{path};
        zip.Add(deflated);
        REQUIRE(zip.Size() == 22 + deflated.ZipBytes());
        zip.Add(stored);
        size = zip.Size();
        zip.Finish();
    }
    std::ifstream file{path, std::ios::binary};
    const std::vector<char> written{std::istreambuf_iterator<char>{file}, {}};
    file.close();
    std::filesystem::remove(path);
    REQUIRE(written.size() == size);
    REQUIRE(std::string(written.data(), 4) == "PK\x03\x04");
    REQUIRE(std::string(written.data() + written.size() - 22, 4) == "PK\x05\x06");
    REQUIRE(written[written.size() - 12] == 2);
}
