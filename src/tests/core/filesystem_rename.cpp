// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "core/file_sys/errors.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/service/filesystem/filesystem.h"

TEST_CASE("Renaming a missing guest file returns path not found across directories",
          "[core][filesystem_rename]") {
    auto root = std::make_shared<FileSys::VectorVfsDirectory>();
    auto source = std::make_shared<FileSys::VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{}, std::vector<FileSys::VirtualDir>{}, "source");
    auto target = std::make_shared<FileSys::VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{}, std::vector<FileSys::VirtualDir>{}, "target");
    root->AddDirectory(source);
    root->AddDirectory(target);
    Service::FileSystem::VfsDirectoryServiceWrapper filesystem{root};

    SECTION("missing root file") {
        CHECK(filesystem.RenameFile("/missing", "/target/new") == FileSys::ResultPathNotFound);
    }
    SECTION("missing nested file") {
        CHECK(filesystem.RenameFile("/source/missing", "/target/new") == FileSys::ResultPathNotFound);
    }
    SECTION("same-directory missing file") {
        CHECK(filesystem.RenameFile("/source/missing", "/source/new") == FileSys::ResultPathNotFound);
    }
    SECTION("existing file still renames") {
        auto present = std::make_shared<FileSys::VectorVfsFile>(std::vector<u8>{1, 2}, "present");
        source->AddFile(present);
        CHECK(filesystem.RenameFile("/source/present", "/source/renamed") == ResultSuccess);
        CHECK(source->GetFile("renamed") == present);
        CHECK(present->ReadAllBytes() == std::vector<u8>{1, 2});
    }
    CHECK(target->GetFiles().empty());
}
