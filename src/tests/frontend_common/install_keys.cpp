// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>
#include <random>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include "common/fs/path_util.h"
#include "frontend_common/firmware_manager.h"

namespace {

/// Points the keys folder at an empty temporary directory for one test.
class KeysDirectory {
public:
    KeysDirectory()
        : original{Common::FS::GetEdenPath(Common::FS::EdenPath::KeysDir)},
          root{std::filesystem::temp_directory_path() /
               fmt::format("lichii-keys-{}", std::random_device{}())},
          keys{root / "keys"} {
        REQUIRE(std::filesystem::create_directories(keys));
        Common::FS::SetEdenPath(Common::FS::EdenPath::KeysDir, keys);
    }

    ~KeysDirectory() {
        Common::FS::SetEdenPath(Common::FS::EdenPath::KeysDir, original);
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    KeysDirectory(const KeysDirectory&) = delete;
    KeysDirectory& operator=(const KeysDirectory&) = delete;

    const std::filesystem::path original;
    const std::filesystem::path root;
    const std::filesystem::path keys;
};

void WriteFile(const std::filesystem::path& path, const char* text) {
    std::ofstream file{path};
    file << text;
    REQUIRE(file.good());
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file{path};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

} // Anonymous namespace

TEST_CASE("Installing the keys already in the keys folder does not copy them onto themselves",
          "[frontend_common][install_keys]") {
    const KeysDirectory directory;
    const auto prod_keys = directory.keys / "prod.keys";
    WriteFile(prod_keys, "header_key = 00\n");
    WriteFile(directory.keys / "title.keys", "0100000000000000 = 00\n");

    // The dummy keys are not valid ones, so the result may be ErrorFailedInit; what matters is
    // that the copy neither throws (it aborted the emulator) nor reports a failed copy.
    FirmwareManager::KeyInstallResult result{};
    REQUIRE_NOTHROW(result = FirmwareManager::InstallKeys(prod_keys.string(), "keys"));
    CHECK(result != FirmwareManager::ErrorFailedCopy);
    CHECK(ReadFile(prod_keys) == "header_key = 00\n");
}

TEST_CASE("Installing keys from another folder copies them into the keys folder",
          "[frontend_common][install_keys]") {
    const KeysDirectory directory;
    const auto source = directory.root / "dump";
    REQUIRE(std::filesystem::create_directory(source));
    WriteFile(source / "prod.keys", "header_key = 11\n");
    WriteFile(source / "title.keys", "0100000000000000 = 11\n");
    WriteFile(directory.keys / "prod.keys", "header_key = 00\n");

    FirmwareManager::KeyInstallResult result{};
    REQUIRE_NOTHROW(result = FirmwareManager::InstallKeys((source / "prod.keys").string(), "keys"));
    CHECK(result != FirmwareManager::ErrorFailedCopy);
    CHECK(ReadFile(directory.keys / "prod.keys") == "header_key = 11\n");
    CHECK(ReadFile(directory.keys / "title.keys") == "0100000000000000 = 11\n");
}
