// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/fs/profile_migration.h"
#include "common/fs/symlink.h"

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace Common::FS::ProfileMigration;

struct Fixture {
    fs::path base = fs::temp_directory_path() /
                    ("lichii-migration-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::path source = base / "eden";
    fs::path destination = base / "lichii";
    fs::path state = base / "lichii.migration";

    Fixture() {
        fs::create_directories(source);
    }

    ~Fixture() {
        std::error_code error;
        fs::remove_all(base, error);
    }

    Plan MakePlan(Strategy strategy = Strategy::Copy) const {
        return {{{source, destination}}, strategy, state, {}};
    }
};

void Write(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    std::ofstream file{path, std::ios::binary};
    file << value;
    REQUIRE(file.good());
}

std::string Read(const fs::path& path) {
    std::ifstream file{path, std::ios::binary};
    REQUIRE(file.good());
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

std::vector<fs::path> Backups(const fs::path& root) {
    std::vector<fs::path> paths;
    for (const auto& item : fs::directory_iterator(root.parent_path())) {
        const auto name = item.path().filename().string();
        if (name.starts_with(root.filename().string() + ".lichii-import-") &&
            name.ends_with(".backup")) {
            paths.push_back(item.path());
        }
    }
    return paths;
}

} // namespace

TEST_CASE("Migration decision is independent of config directory existence",
          "[profile_migration]") {
    Fixture f;
    fs::create_directories(f.destination / "config");
    CHECK(GetState(f.state) == State::Unresolved);
    CHECK(NeedsMigration(f.state));
    MarkFresh(f.state);
    CHECK(GetState(f.state) == State::Fresh);
    CHECK_FALSE(NeedsMigration(f.state));
    CHECK(Read(f.state / "fresh") == "1\n");
}

TEST_CASE("Copy verifies and stages every root before preparing or publishing",
          "[profile_migration]") {
    Fixture f;
    const auto cache = f.base / "eden-cache";
    const auto new_cache = f.base / "lichii-cache";
    Write(f.source / "config/settings.ini", "old paths");
    Write(cache / "shader/blob", std::string(140000, 'x'));
    fs::create_directories(f.source / "empty");
    auto plan = f.MakePlan();
    plan.roots.push_back({cache, new_cache});
    int prepared{};
    RunImport(plan, [&](const std::vector<Root>& staged) {
        ++prepared;
        REQUIRE(staged.size() == 2);
        CHECK(GetState(f.state) == State::Pending);
        CHECK_FALSE(fs::exists(f.destination));
        CHECK_FALSE(fs::exists(new_cache));
        CHECK(fs::equivalent(staged[0].source, f.source));
        CHECK(Read(staged[1].destination / "shader/blob") == std::string(140000, 'x'));
        Write(staged[0].destination / "config/settings.ini", "remapped paths");
        fs::remove_all(staged[1].destination / "shader");
    });
    CHECK(prepared == 1);
    CHECK(GetState(f.state) == State::Complete);
    CHECK_FALSE(NeedsMigration(f.state));
    CHECK(Read(f.destination / "config/settings.ini") == "remapped paths");
    CHECK(Read(f.source / "config/settings.ini") == "old paths");
    CHECK(fs::is_directory(f.destination / "empty"));
    CHECK_FALSE(fs::exists(new_cache / "shader"));
    CHECK(fs::exists(cache / "shader/blob"));
    RunImport(plan, [&](const auto&) { FAIL("Completed import ran twice"); });
}

TEST_CASE("Import keeps the entire old destination in a sibling backup", "[profile_migration]") {
    Fixture f;
    Write(f.source / "save", "legacy save");
    Write(f.destination / "save", "existing save");
    Write(f.destination / "only-existing", "keep this too");
    RunImport(f.MakePlan());
    REQUIRE(Backups(f.destination).size() == 1);
    CHECK(Read(Backups(f.destination)[0] / "save") == "existing save");
    CHECK(Read(Backups(f.destination)[0] / "only-existing") == "keep this too");
    CHECK(Read(f.destination / "save") == "legacy save");
}

TEST_CASE("Interrupted multi-root publication remains pending and retries safely",
          "[profile_migration]") {
    Fixture f;
    const auto cache = f.base / "eden-cache";
    const auto new_cache = f.base / "lichii-cache";
    Write(f.source / "save", "legacy");
    Write(f.destination / "save", "existing");
    Write(cache / "blob", "cache");
    auto plan = f.MakePlan(Strategy::Move);
    plan.roots.push_back({cache, new_cache});
    REQUIRE_THROWS(RunImport(
        plan, [&](const auto&) { Write(new_cache / "appeared-during-staging", "preserve"); }));
    CHECK(GetState(f.state) == State::Pending);
    CHECK(Read(f.destination / "save") == "legacy");
    CHECK(Read(f.source / "save") == "legacy");
    CHECK_FALSE(fs::exists(f.state / "complete"));
    CHECK_THROWS(MarkFresh(f.state));
    RunImport(plan);
    CHECK(GetState(f.state) == State::Complete);
    CHECK(Read(f.destination / "save") == "legacy");
    CHECK(Read(new_cache / "blob") == "cache");
    REQUIRE(Backups(f.destination).size() == 1);
    CHECK(Read(Backups(f.destination)[0] / "save") == "existing");
    REQUIRE(Backups(new_cache).size() == 1);
    CHECK(Read(Backups(new_cache)[0] / "appeared-during-staging") == "preserve");
    REQUIRE(Backups(f.source).size() == 1);
    CHECK(Read(Backups(f.source)[0] / "save") == "legacy");
}

TEST_CASE("Recovery restores moved sources when completion was not recorded",
          "[profile_migration]") {
    Fixture f;
    Write(f.source / "save", "legacy");
    Write(f.destination / "save", "existing");
    RunImport(f.MakePlan(Strategy::Move));
    CHECK_FALSE(fs::exists(f.source));
    fs::remove(f.state / "complete");
    Write(f.state / "pending", "1\n");
    CHECK(GetState(f.state) == State::Pending);
    Recover(f.state);
    CHECK(GetState(f.state) == State::Unresolved);
    CHECK(Read(f.source / "save") == "legacy");
    CHECK(Read(f.destination / "save") == "existing");
    Recover(f.state);
    CHECK(Read(f.destination / "save") == "existing");
    MarkFresh(f.state);
    CHECK(GetState(f.state) == State::Fresh);
}

#ifndef _WIN32
TEST_CASE("Concurrent processes cannot replace an active import journal", "[profile_migration]") {
    Fixture f;
    Write(f.source / "save", "legacy");
    Write(f.destination / "save", "existing");
    int ready[2]{};
    int release[2]{};
    REQUIRE(pipe(ready) == 0);
    REQUIRE(pipe(release) == 0);
    const auto child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        close(ready[0]);
        close(release[1]);
        try {
            RunImport(f.MakePlan(), [&](const auto&) {
                const char value = '1';
                if (write(ready[1], &value, 1) != 1) {
                    _exit(3);
                }
                char received{};
                if (read(release[0], &received, 1) != 1) {
                    _exit(4);
                }
            });
            _exit(0);
        } catch (...) {
            _exit(1);
        }
    }
    close(ready[1]);
    close(release[0]);
    char received{};
    REQUIRE(read(ready[0], &received, 1) == 1);
    const auto journal = Read(f.state / "journal");
    CHECK_THROWS(RunImport(f.MakePlan()));
    CHECK_THROWS(Recover(f.state));
    CHECK_THROWS(MarkFresh(f.state));
    CHECK(Read(f.state / "journal") == journal);
    CHECK(Read(f.destination / "save") == "existing");
    const char value = '1';
    CHECK(write(release[1], &value, 1) == 1);
    close(release[1]);
    close(ready[0]);
    int status{};
    REQUIRE(waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    CHECK(GetState(f.state) == State::Complete);
    CHECK(Read(f.destination / "save") == "legacy");
}

TEST_CASE("Every multi-root publication cutpoint can recover after process exit",
          "[profile_migration]") {
    for (int cut = 0; cut <= 8; ++cut) {
        CAPTURE(cut);
        Fixture f;
        const auto cache = f.base / "eden-cache";
        const auto new_cache = f.base / "lichii-cache";
        Write(f.source / "save", "legacy");
        Write(f.destination / "save", "existing");
        Write(cache / "blob", "old cache");
        Write(new_cache / "blob", "new cache");
        auto plan = f.MakePlan(Strategy::Move);
        plan.roots.push_back({cache, new_cache});
        const auto child = fork();
        REQUIRE(child >= 0);
        if (child == 0) {
            try {
                RunImport(plan, [](const auto&) { _exit(0); });
            } catch (...) {
                _exit(1);
            }
            _exit(2);
        }
        int status{};
        REQUIRE(waitpid(child, &status, 0) == child);
        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) == 0);
        REQUIRE(GetState(f.state) == State::Pending);
        std::vector<fs::path> stages;
        std::vector<fs::path> backups;
        std::vector<fs::path> sources;
        for (const auto& root : plan.roots) {
            for (const auto& item : fs::directory_iterator(f.base)) {
                const auto name = item.path().filename().string();
                if (name.starts_with(root.destination.filename().string() + ".lichii-import-") &&
                    name.ends_with(".stage")) {
                    stages.push_back(item.path());
                    auto backup = item.path();
                    backup.replace_extension(".backup");
                    backups.push_back(backup);
                    const auto suffix = backup.filename().string().substr(
                        root.destination.filename().string().size());
                    sources.push_back(fs::path{root.source.string() + suffix});
                }
            }
        }
        REQUIRE(stages.size() == 2);
        for (int step = 0; step < cut; ++step) {
            if (step < 6) {
                const auto i = static_cast<std::size_t>(step / 3);
                if (step % 3 == 0) {
                    Write(f.state / ("publishing-" + std::to_string(i)), "1\n");
                } else if (step % 3 == 1) {
                    fs::rename(plan.roots[i].destination, backups[i]);
                } else {
                    fs::rename(stages[i], plan.roots[i].destination);
                }
            } else {
                const auto i = static_cast<std::size_t>(step - 6);
                fs::rename(plan.roots[i].source, sources[i]);
            }
        }
        Recover(f.state);
        CHECK(GetState(f.state) == State::Unresolved);
        CHECK(Read(f.source / "save") == "legacy");
        CHECK(Read(f.destination / "save") == "existing");
        CHECK(Read(cache / "blob") == "old cache");
        CHECK(Read(new_cache / "blob") == "new cache");
        Recover(f.state);
        RunImport(plan);
        CHECK(GetState(f.state) == State::Complete);
    }
}
#endif

TEST_CASE("A failed second link removes only this attempt's staged links", "[profile_migration]") {
    Fixture f;
    const auto cache = f.base / "eden-cache";
    const auto new_cache = f.base / "lichii-cache";
    Write(f.source / "save", "legacy");
    Write(f.destination / "save", "existing");
    Write(cache / "blob", "cache");
    auto plan = f.MakePlan(Strategy::Link);
    plan.roots.push_back({cache, new_cache});
    std::vector<fs::path> attempted;
    plan.create_link = [&](const fs::path& source, const fs::path& link) {
        attempted.push_back(link);
        if (attempted.size() == 2) {
            return false;
        }
        return Common::FS::CreateSymlink(source, link);
    };
    REQUIRE_THROWS(RunImport(plan));
    REQUIRE(attempted.size() == 2);
    CHECK_FALSE(fs::exists(attempted[0]));
    CHECK_FALSE(fs::exists(new_cache));
    CHECK(Read(f.destination / "save") == "existing");
    CHECK(Read(f.source / "save") == "legacy");
    CHECK(Read(cache / "blob") == "cache");
    CHECK(GetState(f.state) == State::Pending);
    CHECK_FALSE(fs::exists(f.state / "complete"));
    RunImport(f.MakePlan(Strategy::Copy));
    CHECK(GetState(f.state) == State::Complete);
}

TEST_CASE("Link never invokes source-mutating preparation", "[profile_migration]") {
    Fixture f;
    Write(f.source / "shader/blob", "keep shader");
    Write(f.destination / "existing", "keep destination");
    RunImport(f.MakePlan(Strategy::Link),
              [&](const auto&) { FAIL("Link must not clear caches or rewrite linked INIs"); });
    CHECK(Common::FS::IsSymlink(f.destination));
    CHECK(Read(f.source / "shader/blob") == "keep shader");
    CHECK(Read(f.destination / "shader/blob") == "keep shader");
    REQUIRE(Backups(f.destination).size() == 1);
    CHECK(Read(Backups(f.destination)[0] / "existing") == "keep destination");
    CHECK(GetState(f.state) == State::Complete);
}

TEST_CASE("Failed preparation keeps both originals and offers recovery", "[profile_migration]") {
    Fixture f;
    Write(f.source / "save", "legacy");
    Write(f.destination / "save", "existing");
    REQUIRE_THROWS(RunImport(f.MakePlan(Strategy::Move), [&](const auto& staged) {
        CHECK(Read(staged[0].destination / "save") == "legacy");
        throw std::runtime_error("cannot remap config");
    }));
    CHECK(Read(f.source / "save") == "legacy");
    CHECK(Read(f.destination / "save") == "existing");
    CHECK(GetState(f.state) == State::Pending);
    Recover(f.state);
    CHECK(GetState(f.state) == State::Unresolved);
}

TEST_CASE("Migration rejects overlapping paths and symlink trees", "[profile_migration]") {
    Fixture f;
    Write(f.source / "save", "legacy");
    auto plan = f.MakePlan();
    SECTION("destination inside source") {
        plan.roots[0].destination = f.source / "nested";
    }
    SECTION("trailing slash on destination") {
        plan.roots[0].destination = fs::path{f.source.string() + "/"};
    }
    SECTION("relative destination aliases source") {
        plan.roots[0].destination = fs::relative(f.source);
    }
    SECTION("source inside destination") {
        plan.roots[0].destination = f.base;
    }
    SECTION("source alias as destination") {
        REQUIRE(Common::FS::CreateSymlink(f.source, f.destination));
    }
    SECTION("state inside source") {
        plan.state_dir = f.source / "state";
    }
    SECTION("state inside destination") {
        plan.state_dir = f.destination / "state";
    }
    SECTION("source link to external file") {
        Write(f.base / "outside", "do not touch");
        std::error_code error;
        fs::create_symlink(f.base / "outside", f.source / "link", error);
        if (error) {
            SKIP("This host cannot create file symlinks");
        }
    }
    SECTION("source link to external directory") {
        fs::create_directories(f.base / "outside");
        REQUIRE(Common::FS::CreateSymlink(f.base / "outside", f.source / "link"));
    }
    REQUIRE_THROWS(RunImport(plan));
    CHECK(Read(f.source / "save") == "legacy");
    CHECK_FALSE(fs::exists(f.state / "complete"));
}

#ifndef _WIN32
TEST_CASE("A real copy permission failure keeps both profiles and can retry",
          "[profile_migration]") {
    Fixture f;
    Write(f.source / "save", "legacy");
    Write(f.destination / "save", "existing");
    fs::permissions(f.source / "save", fs::perms::none);
    if (std::ifstream{f.source / "save"}.is_open()) {
        fs::permissions(f.source / "save", fs::perms::owner_read | fs::perms::owner_write);
        SKIP("This user can bypass file permissions");
    }
    CHECK_THROWS(RunImport(f.MakePlan()));
    CHECK(GetState(f.state) == State::Pending);
    CHECK(Read(f.destination / "save") == "existing");
    fs::permissions(f.source / "save", fs::perms::owner_read | fs::perms::owner_write);
    CHECK(Read(f.source / "save") == "legacy");
    CHECK_NOTHROW(RunImport(f.MakePlan()));
    CHECK(GetState(f.state) == State::Complete);
    CHECK(Read(f.destination / "save") == "legacy");
    CHECK(Read(f.source / "save") == "legacy");
}
#endif
