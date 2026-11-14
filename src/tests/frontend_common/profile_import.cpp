// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <catch2/catch_test_macros.hpp>
#ifndef SI_NO_CONVERSION
#define SI_NO_CONVERSION 1
#endif
#include <SimpleIni.h>
#include "common/fs/path_util.h"
#include "common/fs/profile_migration.h"
#include "frontend_common/profile_import.h"

namespace fs = std::filesystem;
namespace Migration = Common::FS::ProfileMigration;
using namespace FrontendCommon;

namespace {
struct Fixture {
    fs::path root = fs::temp_directory_path() /
                    ("lichii-import-config-" + std::to_string(std::random_device{}()));
    Fixture() {
        fs::create_directories(root);
    }
    ~Fixture() {
        std::error_code error;
        fs::remove_all(root, error);
    }
};
} // namespace

TEST_CASE("Imported paths are remapped by path component", "[migration]") {
    const auto old_root = fs::absolute("eden");
    const auto new_root = fs::absolute("lichii");
    CHECK(RemapImportedPath(old_root / "custom-nand", old_root, new_root) ==
          new_root / "custom-nand");
    CHECK(RemapImportedPath(old_root, old_root, new_root) == new_root);
    const auto external = old_root.parent_path() / "eden-other" / "save";
    CHECK(RemapImportedPath(external, old_root, new_root) == external);
    const auto escape = old_root / ".." / "external";
    CHECK(RemapImportedPath(escape, old_root, new_root) == escape);
    CHECK(RemapImportedPath({}, old_root, new_root).empty());
    CHECK(RemapImportedPath("relative", old_root, new_root) == "relative");
}

TEST_CASE("Move remaps every raw imported storage path before the source disappears",
          "[migration]") {
    Fixture fixture;
    const auto source = fixture.root / "eden";
    const auto destination = fixture.root / "lichii";
    const auto config = source / "config";
    fs::create_directories(config / "custom");
    constexpr std::array keys{"nand_directory", "save_directory", "sdmc_directory",
                              "load_directory", "tas_directory",  "dump_directory"};
    for (const auto name : {"qt-config.ini", "sdl2-config.ini", "custom/game.ini"}) {
        std::ofstream file{config / name};
        file << "[Data%20Storage]\n";
        for (const auto* key : keys) {
            fs::create_directories(source / key);
            file << key << "=\"" << (source / key).generic_string() << "\"\n";
        }
        fs::create_directories(source / "custom-screenshots");
        file << "[Screenshots]\nscreenshot_path=\""
             << (source / "custom-screenshots").generic_string() << "\"\n";
    }
    const auto state = fixture.root / "lichii.migration";
    Migration::RunImport(
        {{{source, destination}}, Migration::Strategy::Move, state, {}}, [&](const auto& staged) {
            RemapImportedConfig(staged.at(0).destination / "config", source, destination);
        });
    CHECK_FALSE(fs::exists(source));
    for (const auto name : {"qt-config.ini", "sdl2-config.ini", "custom/game.ini"}) {
        std::ifstream file{destination / "config" / name};
        const std::string contents{std::istreambuf_iterator<char>{file}, {}};
        CSimpleIniA ini;
        REQUIRE(ini.LoadData(contents) >= 0);
        for (const auto* key : keys) {
            const fs::path selected{ini.GetValue("Data%20Storage", key, "")};
            CHECK(selected == destination / key);
            CHECK(fs::is_directory(selected));
        }
        const fs::path screenshot{ini.GetValue("Screenshots", "screenshot_path", "")};
        CHECK(screenshot == destination / "custom-screenshots");
        CHECK(fs::is_directory(screenshot));
    }
    CHECK(Migration::GetState(state) == Migration::State::Complete);
}

#if !defined(_WIN32) && !defined(__ANDROID__)
TEST_CASE("CLI-first configuration is blocked and portable user stays untouched", "[migration]") {
    Fixture fixture;
    const auto cwd = fs::current_path();
    const auto read_env = [](const char* name) {
        const auto* value = std::getenv(name);
        return value ? std::optional<std::string>{value} : std::nullopt;
    };
    const auto data_env = read_env("XDG_DATA_HOME");
    const auto config_env = read_env("XDG_CONFIG_HOME");
    const auto cache_env = read_env("XDG_CACHE_HOME");
    struct Restore {
        fs::path cwd;
        std::array<std::optional<std::string>, 3> env;
        ~Restore() {
            fs::current_path(cwd);
            const std::array names{"XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"};
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (env[i]) {
                    setenv(names[i], env[i]->c_str(), 1);
                } else {
                    unsetenv(names[i]);
                }
            }
            Common::FS::SetAppDirectory("");
        }
    } restore{cwd, {data_env, config_env, cache_env}};
    fs::current_path(fixture.root);
    setenv("XDG_DATA_HOME", (fixture.root / "data").c_str(), 1);
    setenv("XDG_CONFIG_HOME", (fixture.root / "config").c_str(), 1);
    setenv("XDG_CACHE_HOME", (fixture.root / "cache").c_str(), 1);
    fs::create_directories(fixture.root / "data/eden");
    fs::create_directories(fixture.root / "config/lichii");
    Common::FS::SetAppDirectory("");
    CHECK(NeedsLegacyProfileImport());
    CHECK_THROWS(CheckLegacyProfileBeforeConfig());
    CHECK(fs::is_empty(fixture.root / "config/lichii"));
    Migration::MarkFresh(ProfileImportStateDirectory());
    CHECK_FALSE(NeedsLegacyProfileImport());
    CHECK_NOTHROW(CheckLegacyProfileBeforeConfig());

    fs::create_directories(fixture.root / "user");
    Common::FS::SetAppDirectory("");
    CHECK(Common::FS::IsPortableMode());
    CHECK(Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir) == fixture.root / "user");
    CHECK_FALSE(NeedsLegacyProfileImport());
    CHECK_NOTHROW(CheckLegacyProfileBeforeConfig());
    CHECK(fs::is_empty(fixture.root / "user"));
}
#endif
