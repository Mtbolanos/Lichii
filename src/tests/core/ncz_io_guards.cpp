// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "core/file_sys/fssystem/fssystem_aes_ctr_storage.h"
#include "core/file_sys/fssystem/fssystem_alignment_matching_storage_impl.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "frontend_common/content_copy.h"

namespace {
class ShortReadFile final : public FileSys::VectorVfsFile {
public:
    ShortReadFile() : FileSys::VectorVfsFile(std::vector<u8>(64, 0x5a)) {}
    std::size_t Read(u8* out, std::size_t length, std::size_t offset) const override {
        const auto actual = FileSys::VectorVfsFile::Read(out, length, offset);
        return actual == 0 ? 0 : actual - 1;
    }
};

class ShortWriteFile final : public FileSys::VectorVfsFile {
public:
    std::size_t Write(const u8* input, std::size_t length, std::size_t offset) override {
        const auto actual = FileSys::VectorVfsFile::Write(input, length, offset);
        return actual == 0 ? 0 : actual - 1;
    }
};
} // namespace

TEST_CASE("NCA CTR storage does not decrypt a short source read as success", "[ncz-io]") {
    auto source = std::make_shared<ShortReadFile>();
    std::array<u8, 16> key{};
    std::array<u8, 16> iv{};
    FileSys::AesCtrStorage storage(source, key.data(), key.size(), iv.data(), iv.size());
    std::array<u8, 32> out{};
    CHECK(storage.Read(out.data(), out.size(), 0) == 0);
}

TEST_CASE("NCA CTR storage preserves NIST AES CTR plaintext", "[ncz-io]") {
    const std::array<u8, 16> key{0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                                 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    const std::array<u8, 16> iv{0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
                                0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff};
    const std::vector<u8> encrypted{0x87, 0x4d, 0x61, 0x91, 0xb6, 0x20, 0xe3, 0x26,
                                    0x1b, 0xef, 0x68, 0x64, 0x99, 0x0d, 0xb6, 0xce};
    const std::array<u8, 16> expected{0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
                                      0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a};
    auto source = std::make_shared<FileSys::VectorVfsFile>(encrypted);
    FileSys::AesCtrStorage storage(source, key.data(), key.size(), iv.data(), iv.size());
    std::array<u8, 16> out{};
    REQUIRE(storage.Read(out.data(), out.size(), 0) == out.size());
    CHECK(out == expected);
}

TEST_CASE("NCA alignment storage propagates head core and tail short reads", "[ncz-io]") {
    auto source = std::make_shared<ShortReadFile>();
    alignas(16) std::array<u8, 64> out{};
    alignas(16) std::array<char, 16> work{};
    std::size_t offset{};
    std::size_t length{};
    SECTION("core") {
        offset = 0;
        length = 32;
    }
    SECTION("head") {
        offset = 1;
        length = 31;
    }
    SECTION("tail") {
        offset = 0;
        length = 1;
    }
    CHECK(FileSys::AlignmentMatchingStorageImpl::Read(source, work.data(), work.size(), 16, 16,
                                                      offset, out.data(), length) == 0);
}

TEST_CASE("NCA alignment storage preserves valid partial ranges", "[ncz-io]") {
    std::vector<u8> data(64);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<u8>(i);
    auto source = std::make_shared<FileSys::VectorVfsFile>(data);
    alignas(16) std::array<u8, 64> out{};
    alignas(16) std::array<char, 16> work{};
    for (std::size_t offset : {0, 1, 15, 16, 17}) {
        for (std::size_t length : {1, 15, 16, 31}) {
            REQUIRE(FileSys::AlignmentMatchingStorageImpl::Read(source, work.data(), work.size(),
                                                                16, 16, offset, out.data(),
                                                                length) == length);
            CHECK(std::equal(out.begin(), out.begin() + length, data.begin() + offset));
        }
    }
}

TEST_CASE("Installation copy rejects short IO and clears its incomplete destination", "[ncz-io]") {
    FileSys::VirtualFile source =
        std::make_shared<FileSys::VectorVfsFile>(std::vector<u8>{1, 2, 3, 4, 5});
    FileSys::VirtualFile dest = std::make_shared<FileSys::VectorVfsFile>();
    SECTION("read failure") {
        source = std::make_shared<ShortReadFile>();
    }
    SECTION("write failure") {
        dest = std::make_shared<ShortWriteFile>();
    }
    CHECK_FALSE(FrontendCommon::CopyInstallFile(source, dest, 4, {}));
    CHECK(dest->GetSize() == 0);
}

TEST_CASE("Installation copy preserves tail bytes and cancellation", "[ncz-io]") {
    auto source = std::make_shared<FileSys::VectorVfsFile>(std::vector<u8>{1, 2, 3, 4, 5});
    auto dest = std::make_shared<FileSys::VectorVfsFile>();
    SECTION("success") {
        REQUIRE(FrontendCommon::CopyInstallFile(source, dest, 4, {}));
        CHECK(dest->ReadAllBytes() == std::vector<u8>{1, 2, 3, 4, 5});
    }
    SECTION("cancel after prefix") {
        CHECK_FALSE(FrontendCommon::CopyInstallFile(
            source, dest, 4, [](std::size_t, std::size_t offset) { return offset >= 4; }));
        CHECK(dest->GetSize() == 0);
    }
    SECTION("zero block size") {
        CHECK_FALSE(FrontendCommon::CopyInstallFile(source, dest, 0, {}));
    }
}
