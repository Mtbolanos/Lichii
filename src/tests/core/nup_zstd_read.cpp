// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <openssl/evp.h>

#include "common/zstd_compression.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace {

namespace fs = std::filesystem;
using FileSys::NupContainer;

struct TempDir {
    fs::path path = fs::temp_directory_path() /
                    ("nup-read-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void Put(std::vector<u8>& bytes, std::size_t offset, u64 value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
        bytes[offset + i] = static_cast<u8>(value >> (8 * i));
    }
}

// Compressible bytes: literals from a skewed distribution and copies of recent data.
std::vector<u8> Content(std::size_t size, u64 seed) {
    std::vector<u8> out(size);
    std::mt19937_64 rng{seed};
    std::size_t pos = 0;
    while (pos < size) {
        const u64 r = rng();
        const std::size_t length = std::min<std::size_t>(8 + (r & 127), size - pos);
        if (pos > 1024 && (r >> 8) % 3 != 0) {
            const std::size_t back = 1 + (r >> 16) % std::min<std::size_t>(pos, 256 * 1024);
            for (std::size_t i = 0; i < length; ++i) {
                out[pos + i] = out[pos - back + i];
            }
        } else {
            for (std::size_t i = 0; i < length; ++i) {
                const u64 x = rng();
                out[pos + i] = static_cast<u8>(x & (x >> 8));
            }
        }
        pos += length;
    }
    return out;
}

std::vector<u8> Zstd(std::span<const u8> data) {
    return Common::Compression::CompressDataZSTD(data.data(), data.size(), 3);
}

struct Member {
    std::vector<u8> extent;
    u64 size;
    u8 codec;
    std::string name{};
};

Member ZstdMember(const std::vector<u8>& content) {
    return {Zstd(content), content.size(), 1};
}

// A single-package container whose root directory holds the members as files 0..n-1.
std::vector<u8> BuildNup(const std::vector<Member>& members) {
    constexpr std::size_t TocOffset = 0x80;
    const std::size_t files_at = 0x30 + 0x40 + 0x18;
    const std::size_t children_at = files_at + members.size() * 0x50;
    const std::size_t pool_at = children_at + members.size() * 4;
    std::string pool;
    for (const auto& member : members) {
        pool += member.name;
    }
    const std::size_t toc_size = pool_at + pool.size();
    const std::size_t blob_offset = (TocOffset + toc_size + 15) & ~std::size_t{15};
    std::vector<u8> bytes(blob_offset);
    std::vector<u64> extent_offsets;
    for (const auto& member : members) {
        bytes.resize((bytes.size() + 15) & ~std::size_t{15});
        extent_offsets.push_back(bytes.size());
        bytes.insert(bytes.end(), member.extent.begin(), member.extent.end());
    }

    std::copy(NupContainer::HeaderMagic.begin(), NupContainer::HeaderMagic.end(), bytes.begin());
    Put(bytes, 0x04, 0x80, 2);
    Put(bytes, 0x08, TocOffset, 8);
    Put(bytes, 0x10, toc_size, 8);
    Put(bytes, 0x38, blob_offset, 8);
    Put(bytes, 0x40, bytes.size(), 8);
    Put(bytes, 0x48, 1, 4);
    const std::size_t toc = TocOffset;
    Put(bytes, toc + 0x00, 0x01434F54, 4);
    Put(bytes, toc + 0x04, 1, 4);
    Put(bytes, toc + 0x08, 1, 8);
    Put(bytes, toc + 0x10, members.size(), 8);
    Put(bytes, toc + 0x18, pool_at, 8);
    Put(bytes, toc + 0x20, pool.size(), 8);
    std::copy(pool.begin(), pool.end(), bytes.begin() + toc + pool_at);
    Put(bytes, toc + 0x30 + 0x0C, 0xFFFFFFFF, 4);
    Put(bytes, toc + 0x30 + 0x40 + 0x0C, members.size(), 4);
    std::size_t name_offset = 0;
    for (std::size_t i = 0; i < members.size(); ++i) {
        const std::size_t record = toc + files_at + i * 0x50;
        Put(bytes, record + 0x00, name_offset, 4);
        Put(bytes, record + 0x04, members[i].name.size(), 4);
        name_offset += members[i].name.size();
        Put(bytes, record + 0x08, members[i].size, 8);
        Put(bytes, record + 0x10, extent_offsets[i], 8);
        Put(bytes, record + 0x18, members[i].extent.size(), 8);
        bytes[record + 0x40] = members[i].codec;
        Put(bytes, toc + children_at + i * 4, 1 + i, 4);
    }
    unsigned int length{};
    EVP_Digest(bytes.data() + toc, toc_size, bytes.data() + 0x18, &length, EVP_sha256(), nullptr);
    return bytes;
}

FileSys::VirtualFile Backing(std::vector<u8> bytes) {
    return std::make_shared<FileSys::VectorVfsFile>(std::move(bytes));
}

bool ReadMatches(const FileSys::VirtualFile& file, const std::vector<u8>& content,
                 std::size_t offset, std::size_t length) {
    std::vector<u8> buffer(length);
    const std::size_t expected =
        std::min(length, content.size() - std::min(offset, content.size()));
    return file->Read(buffer.data(), length, offset) == expected &&
           std::equal(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(expected),
                      content.begin() + static_cast<std::ptrdiff_t>(offset));
}

std::size_t DirectoryBytes(const fs::path& dir) {
    std::size_t total = 0;
    // Windows directory entries can report a stale size for a file that is still open.
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::ifstream stream{entry.path(), std::ios::binary | std::ios::ate};
        total += static_cast<std::size_t>(stream.tellg());
    }
    return total;
}

} // namespace

TEST_CASE("NUP zstd random reads match the source in memory and spilled", "[nup][nup-read]") {
    const std::vector<std::size_t> sizes{1, 100, 64 * 1024, 300 * 1024, 3 * 1024 * 1024};
    std::vector<std::vector<u8>> contents;
    std::vector<Member> members;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        contents.push_back(Content(sizes[i], i + 1));
        members.push_back(ZstdMember(contents.back()));
    }
    const auto bytes = BuildNup(members);
    TempDir t;
    // 64 MiB keeps every member in memory; 128 KiB spills those above 64 KiB.
    for (const std::size_t cap : {std::size_t{64} << 20, std::size_t{128} << 10}) {
        NupContainer nup{Backing(bytes), cap, t.path};
        REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
        u8 byte{};
        REQUIRE(nup.OpenFile(4)->Read(&byte, 0, 10) == 0);
        std::mt19937_64 rng{cap};
        for (int i = 0; i < 400; ++i) {
            const std::size_t member = rng() % contents.size();
            const auto& content = contents[member];
            const std::size_t offset = rng() % (content.size() + 1);
            const std::size_t length = 1 + rng() % (200 * 1024);
            REQUIRE(ReadMatches(nup.OpenFile(member), content, offset, length));
        }
        for (std::size_t i = 0; i < contents.size(); ++i) {
            REQUIRE(nup.OpenFile(i)->ReadAllBytes() == contents[i]);
        }
    }
}

TEST_CASE("NUP zstd reads are safe across threads and evictions", "[nup][nup-read]") {
    std::vector<std::vector<u8>> contents;
    std::vector<Member> members;
    for (int i = 0; i < 6; ++i) {
        contents.push_back(Content(std::size_t{1} << 20, 100 + i));
        members.push_back(ZstdMember(contents.back()));
    }
    const auto bytes = BuildNup(members);
    TempDir t;
    // 4 MiB holds three 1 MiB members at a time; 1 MiB spills them all.
    for (const std::size_t cap : {std::size_t{4} << 20, std::size_t{1} << 20}) {
        NupContainer nup{Backing(bytes), cap, t.path};
        REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
        std::vector<FileSys::VirtualFile> files;
        for (std::size_t i = 0; i < contents.size(); ++i) {
            files.push_back(nup.OpenFile(i));
        }
        std::atomic<int> mismatches{};
        std::vector<std::thread> threads;
        for (unsigned worker = 0; worker < 8; ++worker) {
            threads.emplace_back([&, worker] {
                std::mt19937_64 rng{worker * 7919 + cap};
                if (worker % 2 == 0) {
                    // Streams that decode the member a little at a time, racing each other.
                    const std::size_t member = worker / 2;
                    for (std::size_t offset = 0; offset < contents[member].size();
                         offset += 8 * 1024) {
                        mismatches += !ReadMatches(files[member], contents[member], offset, 8192);
                    }
                    return;
                }
                for (int i = 0; i < 300; ++i) {
                    const std::size_t member = rng() % contents.size();
                    const std::size_t offset = rng() % contents[member].size();
                    const std::size_t length = 1 + rng() % (64 * 1024);
                    mismatches += !ReadMatches(files[member], contents[member], offset, length);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        REQUIRE(mismatches == 0);
    }
}

TEST_CASE("NUP zstd members of several frames decode in memory and spilled", "[nup][nup-read]") {
    const auto first = Content(200 * 1024, 7);
    const auto second = Content(300 * 1024, 8);
    std::vector<u8> content = first;
    content.insert(content.end(), second.begin(), second.end());
    Member member{Zstd(first), content.size(), 1};
    const auto tail = Zstd(second);
    member.extent.insert(member.extent.end(), tail.begin(), tail.end());
    const auto bytes = BuildNup({member});
    TempDir t;
    for (const std::size_t cap : {std::size_t{64} << 20, std::size_t{64} << 10}) {
        NupContainer nup{Backing(bytes), cap, t.path};
        REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
        REQUIRE(ReadMatches(nup.OpenFile(0), content, content.size() - 5000, 5000));
        REQUIRE(nup.OpenFile(0)->ReadAllBytes() == content);
    }
}

TEST_CASE("NUP zstd spill file grows only as far as reads need", "[nup][nup-read]") {
    const auto content = Content(8 << 20, 3);
    const auto bytes = BuildNup({ZstdMember(content)});
    TempDir t;
    NupContainer nup{Backing(bytes), 1 << 20, t.path};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    const auto file = nup.OpenFile(0);
    REQUIRE(ReadMatches(file, content, 0, 4096));
    REQUIRE(DirectoryBytes(t.path) < content.size() / 4);
    REQUIRE(ReadMatches(file, content, content.size() - 4096, 4096));
    REQUIRE(DirectoryBytes(t.path) == content.size());
}

TEST_CASE("NUP zstd reads fail cleanly on corrupt or truncated extents", "[nup][nup-read]") {
    const auto content = Content(2 << 20, 5);
    const auto good = ZstdMember(content);
    TempDir t;
    for (const std::size_t cap : {std::size_t{64} << 20, std::size_t{64} << 10}) {
        DYNAMIC_SECTION("truncated, cap " << cap) {
            Member truncated = good;
            truncated.extent.resize(truncated.extent.size() / 2);
            NupContainer nup{Backing(BuildNup({truncated})), cap, t.path};
            REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
            const auto file = nup.OpenFile(0);
            // The intact prefix still reads; anything past the cut fails every time.
            REQUIRE(ReadMatches(file, content, 1000, 4096));
            std::vector<u8> buffer(4096);
            for (int attempt = 0; attempt < 2; ++attempt) {
                REQUIRE(file->Read(buffer.data(), buffer.size(), content.size() - 4096) == 0);
            }
            REQUIRE(file->ReadAllBytes().empty());
            REQUIRE(nup.VerifyIntegrity({}) ==
                    Loader::ResultStatus::ErrorIntegrityVerificationFailed);
        }
        DYNAMIC_SECTION("bad frame header, cap " << cap) {
            Member corrupt = good;
            std::fill_n(corrupt.extent.begin(), 8, u8{0xA5});
            NupContainer nup{Backing(BuildNup({corrupt})), cap, t.path};
            REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
            std::vector<u8> buffer(4096);
            REQUIRE(nup.OpenFile(0)->Read(buffer.data(), buffer.size(), 0) == 0);
        }
        DYNAMIC_SECTION("recorded size smaller than the frame, cap " << cap) {
            Member oversized = good;
            oversized.size = content.size() - 1000;
            NupContainer nup{Backing(BuildNup({oversized})), cap, t.path};
            REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
            REQUIRE(nup.OpenFile(0)->ReadAllBytes().empty());
        }
        DYNAMIC_SECTION("recorded size ending on a decoder output block, cap " << cap) {
            // zstd hands spilled output over in 128 KiB blocks, so the recorded size is reached
            // exactly at a block end while the frame still holds more.
            Member oversized = good;
            oversized.size = content.size() - 128 * 1024;
            NupContainer nup{Backing(BuildNup({oversized})), cap, t.path};
            REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
            const auto file = nup.OpenFile(0);
            std::vector<u8> buffer(4096);
            REQUIRE(file->Read(buffer.data(), buffer.size(), oversized.size - 1) == 0);
            REQUIRE(file->ReadAllBytes().empty());
        }
        DYNAMIC_SECTION("recorded size larger than the frame, cap " << cap) {
            Member undersized = good;
            undersized.size = content.size() + 1000;
            NupContainer nup{Backing(BuildNup({undersized})), cap, t.path};
            REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
            REQUIRE(nup.OpenFile(0)->ReadAllBytes().empty());
        }
    }
}

TEST_CASE("NUP zstd spilled members resume after their decoder is freed", "[nup][nup-read]") {
    std::vector<std::vector<u8>> contents;
    std::vector<Member> members;
    for (int i = 0; i < 5; ++i) {
        contents.push_back(Content(std::size_t{2} << 20, 200 + i));
        members.push_back(ZstdMember(contents.back()));
    }
    TempDir t;
    NupContainer nup{Backing(BuildNup(members)), 1 << 20, t.path};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    // Starting every member leaves only the newest decoders alive; the older members decode
    // again from their first frame when read further.
    for (int round = 0; round < 3; ++round) {
        for (std::size_t i = 0; i < contents.size(); ++i) {
            const std::size_t offset = static_cast<std::size_t>(round) * 512 * 1024;
            REQUIRE(ReadMatches(nup.OpenFile(i), contents[i], offset, 4096));
        }
    }
    for (std::size_t i = 0; i < contents.size(); ++i) {
        REQUIRE(nup.OpenFile(i)->ReadAllBytes() == contents[i]);
    }
}

TEST_CASE("NUP serves a sound sibling in place of a .zs member with no zstd header",
          "[nup][nup-read]") {
    const auto zstd_file = [](u64 seed) { return Zstd(Content(200 * 1024, seed)); };
    auto damaged = zstd_file(1);
    std::fill_n(damaged.begin(), 12288, u8{0});
    const auto stand_in = zstd_file(2);
    const auto other_prefix = zstd_file(3);
    const auto other_kind = zstd_file(4);
    auto lonely = zstd_file(5);
    std::fill_n(lonely.begin(), 12288, u8{0});
    const std::vector<u8> zeros(4096, 0);
    Member unreadable = ZstdMember(zstd_file(6));
    std::fill_n(unreadable.extent.begin(), 8, u8{0xA5});

    const auto named = [](Member member, std::string name) {
        member.name = std::move(name);
        return member;
    };
    const std::vector<Member> members{
        named(ZstdMember(damaged), "Rock_WaterTemple_Gate_M_01_Nrm.bntx.zs"),
        named(ZstdMember(stand_in), "Rock_WaterTemple_Gate_P_01_Nrm.bntx.zs"),
        named({other_prefix, other_prefix.size(), 0}, "Rock_WaterTemple_Wall_A_01_Nrm.bntx.zs"),
        named(ZstdMember(other_kind), "Rock_WaterTemple_Gate_M_01_Alb.bntx.zs"),
        named(ZstdMember(lonely), "Lonely_Mask.bntx.zs"),
        named(unreadable, "Broken_A_01_Nrm.bntx.zs"),
        named(ZstdMember(zeros), "Model.bfres"),
    };
    TempDir t;
    NupContainer nup{Backing(BuildNup(members)), std::size_t{64} << 20, t.path};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    const auto dir = nup.GetRomFSDir(0);
    REQUIRE(dir != nullptr);
    REQUIRE(dir->GetFiles().size() == members.size());
    const auto read = [&](std::string_view name) {
        const auto file = dir->GetFile(name);
        REQUIRE(file != nullptr);
        REQUIRE(file->GetName() == name);
        return file->ReadAllBytes();
    };

    // Same last name token, then the longest common prefix.
    REQUIRE(read("Rock_WaterTemple_Gate_M_01_Nrm.bntx.zs") == stand_in);
    REQUIRE(read("Rock_WaterTemple_Gate_P_01_Nrm.bntx.zs") == stand_in);
    REQUIRE(read("Rock_WaterTemple_Wall_A_01_Nrm.bntx.zs") == other_prefix);
    REQUIRE(read("Rock_WaterTemple_Gate_M_01_Alb.bntx.zs") == other_kind);
    // Without a sibling of its kind a damaged member is served as stored.
    REQUIRE(read("Lonely_Mask.bntx.zs") == lonely);
    // A member that cannot be decoded is not known to be damaged, so it is not replaced.
    REQUIRE(read("Broken_A_01_Nrm.bntx.zs").empty());
    REQUIRE(read("Model.bfres") == zeros);

    // Extraction reads the package tree, which keeps the stored bytes.
    const auto package = nup.GetPackageDir(0);
    REQUIRE(package != nullptr);
    REQUIRE(package->GetFile("Rock_WaterTemple_Gate_M_01_Nrm.bntx.zs")->ReadAllBytes() == damaged);
}
