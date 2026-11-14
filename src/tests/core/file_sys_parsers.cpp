// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Malformed PFS0, RomFS and NCA inputs, built here from scratch, that crashed or hung the parsers
// before they checked their header fields. Each must now fail cleanly, and valid input of the
// same shape must still parse.

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/crypto/key_manager.h"
#include "core/file_sys/card_image.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/errors.h"
#include "core/file_sys/fssystem/fssystem_bucket_tree.h"
#include "core/file_sys/fssystem/fssystem_compression_configuration.h"
#include "core/file_sys/fssystem/fssystem_crypto_configuration.h"
#include "core/file_sys/fssystem/fssystem_indirect_storage.h"
#include "core/file_sys/fssystem/fssystem_integrity_verification_storage.h"
#include "core/file_sys/fssystem/fssystem_nca_file_system_driver.h"
#include "core/file_sys/fssystem/fssystem_nca_header.h"
#include "core/file_sys/partition_filesystem.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/loader/loader.h"

namespace {

using namespace FileSys;

template <typename T>
void Put(std::vector<u8>& out, size_t offset, const T& value) {
    if (out.size() < offset + sizeof(T)) {
        out.resize(offset + sizeof(T));
    }
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

VirtualFile MakeFile(std::vector<u8> bytes) {
    return std::make_shared<VectorVfsFile>(std::move(bytes), "test.bin");
}

// PFS0 with one 0x18-byte entry per name, the names packed into the string table.
std::vector<u8> MakePfs0(const std::vector<std::string>& names) {
    std::string strtab;
    std::vector<u32> name_offsets;
    for (const auto& name : names) {
        name_offsets.push_back(static_cast<u32>(strtab.size()));
        strtab += name;
        strtab += '\0';
    }

    const u32 num_entries = static_cast<u32>(names.size());
    std::vector<u8> out;
    Put(out, 0, Common::MakeMagic('P', 'F', 'S', '0'));
    Put(out, 4, num_entries);
    Put(out, 8, static_cast<u32>(strtab.size()));
    out.resize(0x10 + num_entries * 0x18);
    for (u32 i = 0; i < num_entries; ++i) {
        Put(out, 0x10 + i * 0x18 + 0x10, name_offsets[i]);
    }
    out.insert(out.end(), strtab.begin(), strtab.end());
    return out;
}

// PFS0 holding the given files, like an NSP.
std::vector<u8> MakePfs0WithFiles(const std::vector<std::pair<std::string, std::vector<u8>>>& files) {
    std::vector<std::string> names;
    for (const auto& file : files) {
        names.push_back(file.first);
    }
    auto out = MakePfs0(names);
    u64 offset = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        const u64 size = files[i].second.size();
        Put(out, 0x10 + i * 0x18, offset);
        Put(out, 0x10 + i * 0x18 + 8, size);
        offset += size;
    }
    for (const auto& file : files) {
        out.insert(out.end(), file.second.begin(), file.second.end());
    }
    return out;
}

TEST_CASE("PFS0 with valid entries parses", "[core][file_sys]") {
    PartitionFilesystem pfs{MakeFile(MakePfs0({"main", "main.npdm"}))};
    REQUIRE(pfs.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(pfs.GetFiles().size() == 2);
    REQUIRE(pfs.GetFile("main.npdm") != nullptr);
}

TEST_CASE("PFS0 rejects a name offset past the string table", "[core][file_sys]") {
    auto bytes = MakePfs0({"main"});
    Put(bytes, 0x10 + 0x10, u32{0x7FFFFFF0});
    PartitionFilesystem pfs{MakeFile(std::move(bytes))};
    REQUIRE(pfs.GetStatus() == Loader::ResultStatus::ErrorBadPFSHeader);
    REQUIRE(pfs.GetFiles().empty());
}

TEST_CASE("PFS0 name stops at the end of the string table", "[core][file_sys]") {
    auto bytes = MakePfs0({"main"});
    bytes.back() = 'x';
    PartitionFilesystem pfs{MakeFile(std::move(bytes))};
    REQUIRE(pfs.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(pfs.GetFile("mainx") != nullptr);
}

TEST_CASE("PFS0 rejects a table larger than the file without allocating it",
          "[core][file_sys]") {
    auto bytes = MakePfs0({"main"});
    Put(bytes, 4, u32{0xFFFFFFFF});
    Put(bytes, 8, u32{0xFFFFFFFF});
    PartitionFilesystem pfs{MakeFile(std::move(bytes))};
    REQUIRE(pfs.GetStatus() == Loader::ResultStatus::ErrorIncorrectPFSFileSize);
}

TEST_CASE("PFS0 with more than 65535 entries reads every entry", "[core][file_sys]") {
    PartitionFilesystem pfs{MakeFile(MakePfs0(std::vector<std::string>(70000, "a")))};
    REQUIRE(pfs.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(pfs.GetFiles().size() == 70000);
}

constexpr u32 Empty = 0xFFFFFFFF;
constexpr size_t Sibling = 1;
constexpr size_t ChildDir = 2;

// RomFS with the given directory and file meta tables and the data after them.
struct RomFSBuilder {
    std::vector<u8> dirs;
    std::vector<u8> files;
    std::vector<u8> data;

    u32 AddDirectory(u32 parent, u32 sibling, u32 child_dir, u32 child_file,
                     const std::string& name) {
        const auto offset = static_cast<u32>(dirs.size());
        const u32 fields[] = {parent, sibling, child_dir, child_file, 0,
                              static_cast<u32>(name.size())};
        for (size_t i = 0; i < std::size(fields); ++i) {
            Put(dirs, offset + i * 4, fields[i]);
        }
        dirs.insert(dirs.end(), name.begin(), name.end());
        dirs.resize((dirs.size() + 3) & ~size_t{3});
        return offset;
    }

    u32 AddFile(u32 parent, u32 sibling, const std::string& name, const std::string& contents) {
        const auto offset = static_cast<u32>(files.size());
        Put(files, offset, parent);
        Put(files, offset + 4, sibling);
        Put(files, offset + 8, static_cast<u64>(data.size()));
        Put(files, offset + 0x10, static_cast<u64>(contents.size()));
        Put(files, offset + 0x18, u32{0});
        Put(files, offset + 0x1C, static_cast<u32>(name.size()));
        files.insert(files.end(), name.begin(), name.end());
        files.resize((files.size() + 3) & ~size_t{3});
        data.insert(data.end(), contents.begin(), contents.end());
        return offset;
    }

    void SetDirectory(u32 offset, size_t field, u32 value) {
        Put(dirs, offset + field * 4, value);
    }

    std::vector<u8> Build() const {
        const u64 dirs_offset = 0x50;
        const u64 files_offset = dirs_offset + dirs.size();
        const u64 data_offset = files_offset + files.size();
        std::vector<u8> out;
        Put(out, 0x00, u64{0x50});
        Put(out, 0x18, dirs_offset);
        Put(out, 0x20, static_cast<u64>(dirs.size()));
        Put(out, 0x38, files_offset);
        Put(out, 0x40, static_cast<u64>(files.size()));
        Put(out, 0x48, data_offset);
        out.insert(out.end(), dirs.begin(), dirs.end());
        out.insert(out.end(), files.begin(), files.end());
        out.insert(out.end(), data.begin(), data.end());
        return out;
    }
};

// Root, then directory "a" at 0x18 holding file "f" with "hi".
RomFSBuilder MakeSmallRomFS() {
    RomFSBuilder romfs;
    romfs.AddDirectory(0, Empty, 0x18, Empty, "");
    romfs.AddDirectory(0, Empty, Empty, 0, "a");
    romfs.AddFile(0x18, Empty, "f", "hi");
    return romfs;
}

TEST_CASE("RomFS with a valid tree extracts", "[core][file_sys]") {
    const auto root = ExtractRomFS(MakeFile(MakeSmallRomFS().Build()));
    REQUIRE(root != nullptr);
    const auto file = root->GetFileRelative("a/f");
    REQUIRE(file != nullptr);
    REQUIRE(file->ReadAllBytes() == std::vector<u8>{'h', 'i'});
}

TEST_CASE("RomFS rejects a directory that is its own sibling", "[core][file_sys]") {
    auto romfs = MakeSmallRomFS();
    romfs.SetDirectory(0x18, Sibling, 0x18);
    REQUIRE(ExtractRomFS(MakeFile(romfs.Build())) == nullptr);
}

TEST_CASE("RomFS rejects a directory whose child is its parent", "[core][file_sys]") {
    auto romfs = MakeSmallRomFS();
    romfs.SetDirectory(0x18, ChildDir, 0);
    REQUIRE(ExtractRomFS(MakeFile(romfs.Build())) == nullptr);
}

TEST_CASE("RomFS rejects a file that is its own sibling", "[core][file_sys]") {
    auto romfs = MakeSmallRomFS();
    Put(romfs.files, 4, u32{0});
    REQUIRE(ExtractRomFS(MakeFile(romfs.Build())) == nullptr);
}

TEST_CASE("RomFS rejects an entry past the end of its table", "[core][file_sys]") {
    auto romfs = MakeSmallRomFS();
    romfs.SetDirectory(0x18, Sibling, 0x1000);
    REQUIRE(ExtractRomFS(MakeFile(romfs.Build())) == nullptr);
}

TEST_CASE("RomFS rejects a table larger than the file without allocating it",
          "[core][file_sys]") {
    auto bytes = MakeSmallRomFS().Build();
    Put(bytes, 0x20, u64{0x7FFF'FFFF'FFFF'0000});
    REQUIRE(ExtractRomFS(MakeFile(std::move(bytes))) == nullptr);
}

// A chain of directories the given number of levels below the root.
std::vector<u8> MakeDeepRomFS(size_t depth) {
    RomFSBuilder romfs;
    romfs.AddDirectory(0, Empty, Empty, Empty, "");
    u32 parent = 0;
    for (size_t i = 0; i < depth; ++i) {
        const u32 child = romfs.AddDirectory(parent, Empty, Empty, Empty, "d");
        romfs.SetDirectory(parent, ChildDir, child);
        parent = child;
    }
    return romfs.Build();
}

TEST_CASE("RomFS extracts a tree as deep as a Horizon path allows", "[core][file_sys]") {
    auto dir = ExtractRomFS(MakeFile(MakeDeepRomFS(0x300 / 2)));
    REQUIRE(dir != nullptr);
    size_t depth = 0;
    while (!dir->GetSubdirectories().empty()) {
        dir = dir->GetSubdirectories().front();
        ++depth;
    }
    REQUIRE(depth == 0x300 / 2);
}

TEST_CASE("RomFS rejects a tree deeper than a Horizon path allows", "[core][file_sys]") {
    REQUIRE(ExtractRomFS(MakeFile(MakeDeepRomFS(100000))) == nullptr);
}

// A plaintext NCA, which nx2 opens without a header key, with one RomFS section spanning
// [0xC00, 0x4C00) whose FS header and contents the test fills in.
struct PlaintextNca {
    NcaHeader header{};
    NcaFsHeader fs{};
    std::vector<u8> section;

    PlaintextNca() {
        header.magic = NcaHeader::Magic3;
        header.content_type = NcaHeader::ContentType::Program;
        header.sdk_addon_version = 0x000C0000;
        header.fs_info[0].start_sector = 6;
        header.fs_info[0].end_sector = 0x26;
        fs.version = 2;
        fs.fs_type = NcaFsHeader::FsType::RomFs;
        fs.encryption_type = NcaFsHeader::EncryptionType::None;
    }

    Result Open() const {
        std::vector<u8> bytes(std::max<size_t>(0x4C00, 0xC00 + section.size()));
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::memcpy(bytes.data() + sizeof(header), &fs, sizeof(fs));
        std::copy(section.begin(), section.end(), bytes.begin() + 0xC00);

        auto reader = std::make_shared<NcaReader>();
        R_TRY(reader->Initialize(MakeFile(std::move(bytes)), GetCryptoConfiguration(),
                                 GetNcaCompressionConfiguration()));
        NcaFileSystemDriver driver{reader};
        VirtualFile storage;
        NcaFsHeaderReader header_reader;
        R_TRY(driver.OpenStorage(&storage, &header_reader, 0));
        R_UNLESS(storage != nullptr, ResultInvalidNcaFsHeader);
        R_SUCCEED();
    }
};

PlaintextNca MakeSha256Nca() {
    PlaintextNca nca;
    nca.fs.hash_type = NcaFsHeader::HashType::HierarchicalSha256Hash;
    auto& sha = nca.fs.hash_data.hierarchical_sha256_data;
    sha.hash_block_size = 0x1000;
    sha.hash_layer_count = 2;
    sha.hash_layer_region[0].offset = 0;
    sha.hash_layer_region[0].size = 0x20;
    sha.hash_layer_region[1].offset = 0x1000;
    sha.hash_layer_region[1].size = 0x1000;
    return nca;
}

TEST_CASE("NCA with a valid SHA-256 section opens", "[core][file_sys]") {
    REQUIRE(MakeSha256Nca().Open() == ResultSuccess);
}

TEST_CASE("NCA with an AES-XTS section opens a storage", "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    nca.fs.encryption_type = NcaFsHeader::EncryptionType::AesXts;
    REQUIRE(nca.Open() == ResultSuccess);
}

TEST_CASE("NCA without a header key reports the missing key", "[core][file_sys]") {
    if (Core::Crypto::KeyManager::Instance().HasKey(Core::Crypto::S256KeyType::Header)) {
        SKIP("A header key is installed");
    }
    const NCA nca{MakeFile(std::vector<u8>(0xC00, 0xA5))};
    CHECK(nca.GetStatus() == Loader::ResultStatus::ErrorMissingHeaderKey);
}

TEST_CASE("NSP that cannot be decrypted reports the missing key", "[core][file_sys]") {
    if (Core::Crypto::KeyManager::Instance().HasKey(Core::Crypto::S256KeyType::Header)) {
        SKIP("A header key is installed");
    }
    const auto nsp = MakeFile(MakePfs0WithFiles(
        {{"0123456789abcdef0123456789abcdef.cnmt.nca", std::vector<u8>(0xC00, 0xA5)}}));
    CHECK_FALSE(Loader::IsBootableGameContainer(nsp, Loader::FileType::NSP));
    CHECK(Loader::GetContainerProgramStatus(nsp, Loader::FileType::NSP) ==
          Loader::ResultStatus::ErrorMissingHeaderKey);
}

TEST_CASE("NSP without programs reports no missing key", "[core][file_sys]") {
    const auto nsp = MakeFile(MakePfs0WithFiles({{"readme.txt", std::vector<u8>(0x10)}}));
    CHECK_FALSE(Loader::IsBootableGameContainer(nsp, Loader::FileType::NSP));
    CHECK(Loader::GetContainerProgramStatus(nsp, Loader::FileType::NSP) ==
          Loader::ResultStatus::ErrorNSPMissingProgramNCA);
}

TEST_CASE("NCA rejects a file shorter than its header", "[core][file_sys]") {
    auto reader = std::make_shared<NcaReader>();
    REQUIRE(reader->Initialize(MakeFile(std::vector<u8>(0x100)), GetCryptoConfiguration(),
                               GetNcaCompressionConfiguration()) == ResultInvalidNcaHeader);
}

TEST_CASE("NCA rejects a SHA-256 block smaller than a hash", "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    nca.fs.hash_data.hierarchical_sha256_data.hash_block_size = 8;
    REQUIRE(nca.Open() == ResultInvalidHierarchicalSha256BlockSize);
}

TEST_CASE("NCA with a SHA-256 block larger than the file opens without allocating it",
          "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    auto& sha = nca.fs.hash_data.hierarchical_sha256_data;
    sha.hash_block_size = 0x10000;
    REQUIRE(nca.Open() == ResultSuccess);
    sha.hash_block_size = 0x40000000;
    REQUIRE(nca.Open() == ResultSuccess);
}

TEST_CASE("NCA rejects a SHA-256 hash layer larger than one block", "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    nca.fs.hash_data.hierarchical_sha256_data.hash_layer_region[0].size = 0x2000;
    REQUIRE(nca.Open() == ResultInvalidHierarchicalSha256BlockSize);
}

TEST_CASE("NCA rejects a SHA-256 hash region past the section before allocating it",
          "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    nca.fs.hash_data.hierarchical_sha256_data.hash_layer_region[0].size = s64{1} << 60;
    REQUIRE(nca.Open() == ResultNcaBaseStorageOutOfRangeC);
}

PlaintextNca MakeIntegrityNca() {
    PlaintextNca nca;
    nca.fs.hash_type = NcaFsHeader::HashType::HierarchicalIntegrityHash;
    auto& levels = nca.fs.hash_data.integrity_meta_info.level_hash_info;
    levels.max_layers = 7;
    for (size_t i = 0; i < levels.info.size(); ++i) {
        levels.info[i].offset = static_cast<s64>(i) * 0x200;
        levels.info[i].size = 0x200;
        levels.info[i].block_order = 9;
    }
    return nca;
}

TEST_CASE("NCA with a valid integrity section opens", "[core][file_sys]") {
    REQUIRE(MakeIntegrityNca().Open() == ResultSuccess);
}

TEST_CASE("NCA rejects an integrity block order a shift cannot hold", "[core][file_sys]") {
    auto nca = MakeIntegrityNca();
    nca.fs.hash_data.integrity_meta_info.level_hash_info.info[2].block_order = 200;
    REQUIRE(nca.Open() == ResultInvalidNcaHierarchicalIntegrityVerificationLayerCount);
}

TEST_CASE("NCA rejects an integrity tree with fewer than three layers", "[core][file_sys]") {
    auto nca = MakeIntegrityNca();
    nca.fs.hash_data.integrity_meta_info.level_hash_info.max_layers = 2;
    REQUIRE(nca.Open() == ResultInvalidNcaHierarchicalIntegrityVerificationLayerCount);
}

TEST_CASE("NCA rejects a layer count outside the hash target table", "[core][file_sys]") {
    auto nca = MakeIntegrityNca();
    nca.fs.encryption_type = NcaFsHeader::EncryptionType::AesCtrSkipLayerHash;
    nca.fs.hash_data.integrity_meta_info.level_hash_info.max_layers = 0;
    REQUIRE(nca.Open() == ResultInvalidNcaFsHeader);
}

TEST_CASE("NCA rejects a negative indirect table size", "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    nca.fs.patch_info.indirect_offset = 0x200;
    nca.fs.patch_info.indirect_size = -0x100;
    REQUIRE(nca.Open() == ResultNcaBaseStorageOutOfRangeE);
}

TEST_CASE("NCA rejects an indirect table larger than the file behind a sparse layer",
          "[core][file_sys]") {
    auto nca = MakeSha256Nca();
    nca.header.fs_info[0].end_sector = 0xFFFFFF06;
    nca.fs.sparse_info.generation = 1;
    BucketTree::Header sparse_header;
    sparse_header.Format(0);
    std::memcpy(nca.fs.sparse_info.bucket.header.data(), &sparse_header, sizeof(sparse_header));
    nca.fs.patch_info.indirect_offset = 0;
    nca.fs.patch_info.indirect_size = s64{1} << 36;
    REQUIRE(nca.Open() == ResultNcaBaseStorageOutOfRangeE);
}

// A 0x10000-byte section: 0x8000 bytes of patch data, then an indirect table mapping the virtual
// range [0, end) onto them.
PlaintextNca MakeIndirectNca(s64 end) {
    auto nca = MakeSha256Nca();
    nca.header.fs_info[0].end_sector = 6 + 0x10000 / 0x200;
    nca.section.resize(0x10000);
    const BucketTree::NodeHeader node{0, 1, end};
    Put(nca.section, 0x8000, node);
    Put(nca.section, 0x8000 + sizeof(node), s64{0});
    Put(nca.section, 0xC000, node);
    IndirectStorage::Entry entry{};
    entry.storage_index = 1;
    Put(nca.section, 0xC000 + sizeof(node), entry);
    BucketTree::Header header;
    header.Format(1);
    nca.fs.patch_info.indirect_offset = 0x8000;
    nca.fs.patch_info.indirect_size = 0x8000;
    std::memcpy(nca.fs.patch_info.indirect_header.data(), &header, sizeof(header));
    return nca;
}

TEST_CASE("NCA with a valid indirect table opens", "[core][file_sys]") {
    REQUIRE(MakeIndirectNca(0x8000).Open() == ResultSuccess);
}

TEST_CASE("NCA rejects a hash region larger than the file behind an indirect layer",
          "[core][file_sys]") {
    auto nca = MakeIndirectNca(s64{1} << 40);
    auto& sha = nca.fs.hash_data.hierarchical_sha256_data;
    sha.hash_layer_region[0].size = s64{1} << 36;
    sha.hash_layer_region[1].offset = s64{1} << 37;
    REQUIRE(nca.Open() == ResultNcaBaseStorageOutOfRangeC);
}

// A decrypted NCA with its ExeFS at the start and its logo partition 2 MiB in, as in a program
// NCA. Reaching the logo of a solid NCZ means decoding everything before it.
TEST_CASE("NCA opens a section only when it is used", "[core][file_sys]") {
    constexpr u32 LogoSector = 0x1006;
    std::vector<u8> bytes(0xC00);
    NcaHeader header{};
    header.magic = NcaHeader::MagicDecrypted;
    header.content_type = NcaHeader::ContentType::Program;
    header.sdk_addon_version = 0x000C0000;
    header.fs_info[0].start_sector = 6;
    header.fs_info[0].end_sector = 0x26;
    header.fs_info[1].start_sector = LogoSector;
    header.fs_info[1].end_sector = LogoSector + 0x20;
    std::memcpy(bytes.data(), &header, sizeof(header));
    NcaFsHeader fs{};
    fs.version = 2;
    fs.fs_type = NcaFsHeader::FsType::PartitionFs;
    fs.hash_type = NcaFsHeader::HashType::None;
    fs.encryption_type = NcaFsHeader::EncryptionType::None;
    std::memcpy(bytes.data() + sizeof(header), &fs, sizeof(fs));
    std::memcpy(bytes.data() + sizeof(header) + sizeof(fs), &fs, sizeof(fs));

    const auto exefs = MakePfs0WithFiles({{"main", std::vector<u8>(0x10)}, {"main.npdm", {}}});
    const auto logo =
        MakePfs0WithFiles({{"NintendoLogo.png", std::vector<u8>(0x10)}, {"StartupMovie.gif", {}}});
    bytes.resize(size_t{LogoSector + 0x20} * 0x200);
    std::copy(exefs.begin(), exefs.end(), bytes.begin() + 0xC00);
    std::copy(logo.begin(), logo.end(), bytes.begin() + size_t{LogoSector} * 0x200);

    class FarthestRead final : public VectorVfsFile {
    public:
        explicit FarthestRead(std::vector<u8> data) : VectorVfsFile(std::move(data), "test.nca") {}
        std::size_t Read(u8* out, std::size_t size, std::size_t offset) const override {
            farthest = std::max(farthest, offset + size);
            return VectorVfsFile::Read(out, size, offset);
        }
        mutable std::size_t farthest{};
    };
    const auto file = std::make_shared<FarthestRead>(std::move(bytes));
    const NCA nca{file};
    REQUIRE(nca.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(nca.GetExeFS() != nullptr);
    CHECK(file->farthest <= size_t{LogoSector} * 0x200);

    REQUIRE(nca.GetLogoPartition() != nullptr);
    CHECK(file->farthest > size_t{LogoSector} * 0x200);
    CHECK(nca.GetSubdirectories().size() == 2);
}

// A card image whose root HFS0 holds empty partitions with the given names.
std::vector<u8> MakeXci(const std::vector<std::string>& partitions) {
    std::vector<u8> partition;
    Put(partition, 0, Common::MakeMagic('H', 'F', 'S', '0'));
    Put(partition, 8, u32{0});

    std::string strtab;
    std::vector<u8> root;
    Put(root, 0, Common::MakeMagic('H', 'F', 'S', '0'));
    Put(root, 4, static_cast<u32>(partitions.size()));
    root.resize(0x10 + partitions.size() * 0x40);
    for (size_t i = 0; i < partitions.size(); ++i) {
        Put(root, 0x10 + i * 0x40, static_cast<u64>(i * partition.size()));
        Put(root, 0x10 + i * 0x40 + 8, static_cast<u64>(partition.size()));
        Put(root, 0x10 + i * 0x40 + 0x10, static_cast<u32>(strtab.size()));
        strtab += partitions[i];
        strtab += '\0';
    }
    Put(root, 8, static_cast<u32>(strtab.size()));
    root.insert(root.end(), strtab.begin(), strtab.end());
    for (size_t i = 0; i < partitions.size(); ++i) {
        root.insert(root.end(), partition.begin(), partition.end());
    }

    std::vector<u8> out(0x200);
    Put(out, 0x100, Common::MakeMagic('H', 'E', 'A', 'D'));
    Put(out, 0x130, u64{0x200});
    Put(out, 0x138, static_cast<u64>(root.size()));
    out.insert(out.end(), root.begin(), root.end());
    return out;
}

TEST_CASE("XCI with a secure partition parses", "[core][file_sys]") {
    XCI xci{MakeFile(MakeXci({"update", "normal", "secure"}))};
    REQUIRE(xci.GetStatus() == Loader::ResultStatus::Success);
}

TEST_CASE("XCI rejects a card without a secure partition", "[core][file_sys]") {
    XCI xci{MakeFile(MakeXci({"update", "normal"}))};
    REQUIRE(xci.GetStatus() == Loader::ResultStatus::ErrorXCIMissingPartition);
}

TEST_CASE("XCI rejects an HFS0 offset past the end of the image", "[core][file_sys]") {
    auto bytes = MakeXci({"update", "normal", "secure"});
    Put(bytes, 0x130, u64{0x10'0000});
    XCI xci{MakeFile(std::move(bytes))};
    REQUIRE(xci.GetStatus() == Loader::ResultStatus::ErrorBadXCIHeader);
}

TEST_CASE("Integrity storage zeroes only the requested bytes past the data",
          "[core][file_sys]") {
    IntegrityVerificationStorage storage;
    storage.Initialize(MakeFile(std::vector<u8>(0x20)), MakeFile(std::vector<u8>(100, 0x11)),
                       0x200, IntegrityVerificationStorage::HashSize, true);

    std::vector<u8> buffer(48, 0xAA);
    REQUIRE(storage.Read(buffer.data(), 32, 80) == 20);
    REQUIRE(buffer[19] == 0x11);
    REQUIRE(buffer[20] == 0);
    REQUIRE(buffer[31] == 0);
    REQUIRE(buffer[32] == 0xAA);
    REQUIRE(buffer[47] == 0xAA);

    REQUIRE(storage.Read(buffer.data(), 1, 0x1000) == 0);
}

} // namespace
