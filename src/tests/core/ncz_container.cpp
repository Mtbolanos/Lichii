// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <zstd.h>
#include "core/file_sys/submission_package.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/loader/loader.h"

TEST_CASE("NSP rejects an unsupported compressed entry before materializing its NCAs",
          "[ncz-container]") {
    constexpr std::size_t MetaSize = 0x4000;
    const std::string meta_name = std::string(32, 'a') + ".cnmt.nca";
    const std::string compressed_name = std::string(32, 'b') + ".ncz";
    const auto strings_size = meta_name.size() + compressed_name.size() + 2;
    const auto start = 16 + 48 + strings_size;
    std::vector<u8> container(start + MetaSize + 32);
    const auto put = [&container](std::size_t offset, u64 value, std::size_t length) {
        for (std::size_t i = 0; i < length; ++i)
            container[offset + i] = static_cast<u8>(value >> (8 * i));
    };
    const std::array<u8, 4> magic{'P', 'F', 'S', '0'};
    std::copy(magic.begin(), magic.end(), container.begin());
    put(4, 2, 4);
    put(8, strings_size, 4);
    put(24, MetaSize, 8);
    put(40, MetaSize, 8);
    put(48, 32, 8);
    put(56, meta_name.size() + 1, 4);
    std::copy(meta_name.begin(), meta_name.end(), container.begin() + 64);
    std::copy(compressed_name.begin(), compressed_name.end(),
              container.begin() + 65 + meta_name.size());
    class CountingContainer final : public FileSys::VectorVfsFile {
    public:
        CountingContainer(std::vector<u8> bytes, std::size_t begin_)
            : FileSys::VectorVfsFile(std::move(bytes), "synthetic.nsz"), begin(begin_) {}
        std::size_t Read(u8* out, std::size_t size, std::size_t offset) const override {
            if (offset < begin + MetaSize && offset + size > begin)
                ++metadata_reads;
            return FileSys::VectorVfsFile::Read(out, size, offset);
        }
        mutable std::size_t metadata_reads{};

    private:
        std::size_t begin;
    };
    auto source = std::make_shared<CountingContainer>(std::move(container), start);
    const FileSys::NSP nsp(source);
    CHECK(nsp.GetStatus() != Loader::ResultStatus::Success);
    CHECK(source->metadata_reads == 0);
}

TEST_CASE("NSP resolves a compressed content ID to the same cached reader", "[ncz-container]") {
    constexpr std::size_t Header = 0x4000;
    const std::string name = std::string(32, '0') + ".ncz";
    std::vector<u8> ncz(Header + 16 + 64);
    const auto put = [](std::vector<u8>& target, std::size_t offset, u64 value, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i)
            target[offset + i] = static_cast<u8>(value >> (i * 8));
    };
    const std::array<u8, 8> magic{'N', 'C', 'Z', 'S', 'E', 'C', 'T', 'N'};
    std::copy(magic.begin(), magic.end(), ncz.begin() + Header);
    put(ncz, Header + 8, 1, 8);
    put(ncz, Header + 16, Header, 8);
    put(ncz, Header + 24, 64, 8);
    put(ncz, Header + 32, 1, 8);
    std::array<u8, 64> expected{};
    expected.fill(0x35);
    std::vector<u8> encoded(ZSTD_compressBound(expected.size()));
    const auto compressed_size =
        ZSTD_compress(encoded.data(), encoded.size(), expected.data(), expected.size(), 3);
    REQUIRE_FALSE(ZSTD_isError(compressed_size));
    ncz.insert(ncz.end(), encoded.begin(), encoded.begin() + compressed_size);
    std::vector<u8> container(40 + name.size() + 1);
    const std::array<u8, 4> pfs_magic{'P', 'F', 'S', '0'};
    std::copy(pfs_magic.begin(), pfs_magic.end(), container.begin());
    put(container, 4, 1, 4);
    put(container, 8, name.size() + 1, 4);
    put(container, 24, ncz.size(), 8);
    std::copy(name.begin(), name.end(), container.begin() + 40);
    container.insert(container.end(), ncz.begin(), ncz.end());
    const FileSys::NSP nsp(
        std::make_shared<FileSys::VectorVfsFile>(std::move(container), "synthetic.nsz"));
    REQUIRE(nsp.GetStatus() == Loader::ResultStatus::Success);
    const std::array<u8, 16> id{};
    auto first = nsp.GetNCAFileById(id);
    auto second = nsp.GetNCAFileById(id);
    REQUIRE(first);
    REQUIRE(second);
    CHECK(first == second);
    std::array<u8, 64> out{};
    REQUIRE(first->Read(out.data(), out.size(), Header) == out.size());
    CHECK(out == expected);
    CHECK(nsp.GetNCAFileById(id) == first);
}
