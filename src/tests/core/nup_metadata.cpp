// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <openssl/evp.h>

#ifdef _WIN32
#include "common/fs/file.h"
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "common/settings.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/nup_writer.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/loader/nup.h"

namespace {

namespace fs = std::filesystem;
using FileSys::Nup::MetadataUpdate;
using FileSys::Nup::PackResult;
using FileSys::Nup::UpdateMetadata;
using FileSys::NupContainer;

constexpr std::size_t Toc = 0x80;
constexpr std::size_t PackageCount = 4;
constexpr std::size_t Directories = Toc + 0x30 + PackageCount * 0x40;
constexpr std::size_t Files = Directories + PackageCount * 0x18;
constexpr std::size_t Children = Files + PackageCount * 0x50;
constexpr std::size_t Pool = Children + PackageCount * 4;
constexpr u64 MetadataLimit = 64ULL * 1024 * 1024;
constexpr u64 IconLimit = 16ULL * 1024 * 1024;

struct TempDir {
    fs::path path;

    TempDir() {
        static std::atomic<u64> sequence{};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        do {
            path = fs::temp_directory_path() /
                   ("nup-metadata-test-" + std::to_string(stamp) + "-" +
                    std::to_string(sequence.fetch_add(1)));
        } while (!fs::create_directory(path));
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void Write(const fs::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    REQUIRE_FALSE(out.fail());
}

std::string Read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.is_open());
    return {std::istreambuf_iterator<char>(in), {}};
}

u64 Le(std::string_view bytes, std::size_t offset, std::size_t size) {
    REQUIRE(offset <= bytes.size());
    REQUIRE(size <= bytes.size() - offset);
    u64 value{};
    for (std::size_t i = 0; i < size; ++i) {
        value |= static_cast<u64>(static_cast<u8>(bytes[offset + i])) << (8 * i);
    }
    return value;
}

void SetLe(std::string& bytes, std::size_t offset, u64 value, std::size_t size) {
    REQUIRE(offset <= bytes.size());
    REQUIRE(size <= bytes.size() - offset);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[offset + i] = static_cast<char>(value >> (8 * i));
    }
}

std::array<u8, 32> Sha256(std::string_view bytes) {
    std::array<u8, 32> hash{};
    unsigned int length{};
    REQUIRE(EVP_Digest(bytes.data(), bytes.size(), hash.data(), &length, EVP_sha256(),
                       nullptr) == 1);
    REQUIRE(length == hash.size());
    return hash;
}

std::array<u8, 32> HashRange(const fs::path& path, u64 offset, u64 size) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.is_open());
    in.seekg(static_cast<std::streamoff>(offset));
    auto* context = EVP_MD_CTX_new();
    REQUIRE(context != nullptr);
    REQUIRE(EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1);
    std::array<char, 65536> buffer{};
    while (size != 0) {
        const auto count = static_cast<std::streamsize>(std::min<u64>(size, buffer.size()));
        in.read(buffer.data(), count);
        REQUIRE(in.gcount() == count);
        REQUIRE(EVP_DigestUpdate(context, buffer.data(), static_cast<std::size_t>(count)) == 1);
        size -= static_cast<u64>(count);
    }
    std::array<u8, 32> hash{};
    unsigned int length{};
    const auto result = EVP_DigestFinal_ex(context, hash.data(), &length);
    EVP_MD_CTX_free(context);
    REQUIRE(result == 1);
    REQUIRE(length == hash.size());
    return hash;
}

void RefreshTocHash(std::string& bytes) {
    const auto hash = Sha256(std::string_view(bytes).substr(Le(bytes, 8, 8), Le(bytes, 16, 8)));
    std::copy(hash.begin(), hash.end(), bytes.begin() + 0x18);
}

constexpr std::size_t Package(std::size_t index) {
    return Toc + 0x30 + index * 0x40;
}

std::string Tlv(u16 type, std::string_view value, u16 reserved = 0xA15E) {
    std::string bytes(8, '\0');
    SetLe(bytes, 0, type, 2);
    SetLe(bytes, 2, reserved, 2);
    SetLe(bytes, 4, value.size(), 4);
    bytes += value;
    return bytes;
}

std::string Localized(std::string_view value) {
    std::string bytes{"\x02nl"};
    bytes.resize(7);
    SetLe(bytes, 3, value.size(), 4);
    bytes += value;
    return bytes;
}

std::string Meta(std::string_view bytes, std::size_t index) {
    return std::string(bytes.substr(Le(bytes, Package(index) + 0x20, 8),
                                   Le(bytes, Package(index) + 0x28, 4)));
}

void AppendMeta(std::string& bytes, std::size_t index, std::string_view meta) {
    SetLe(bytes, Package(index) + 0x20, bytes.size(), 8);
    SetLe(bytes, Package(index) + 0x28, meta.size(), 4);
    bytes += meta;
    SetLe(bytes, 0x40, bytes.size(), 8);
    RefreshTocHash(bytes);
}

std::string RetainedTlvs(std::string_view meta, const MetadataUpdate& update) {
    std::string retained;
    for (std::size_t at = 0; at < meta.size();) {
        REQUIRE(meta.size() - at >= 8);
        const auto type = Le(meta, at, 2);
        const auto size = Le(meta, at + 4, 4);
        REQUIRE(size <= meta.size() - at - 8);
        if (!(type == 3 && update.icon) && !(type == 5 && update.title_name)) {
            retained += meta.substr(at, 8 + size);
        }
        at += 8 + size;
    }
    return retained;
}

std::string Fixture() {
    constexpr std::string_view pool = "payload.bin";
    const std::size_t blob = (Pool + pool.size() + 15) & ~std::size_t{15};
    std::string bytes(blob, '\0');
    bytes.replace(0, 4, "NUP\x01");
    SetLe(bytes, 4, 0x80, 2);
    SetLe(bytes, 6, 1, 2);
    SetLe(bytes, 8, Toc, 8);
    SetLe(bytes, 16, Pool + pool.size() - Toc, 8);
    SetLe(bytes, 0x38, blob, 8);
    SetLe(bytes, 0x48, PackageCount, 4);
    std::fill(bytes.begin() + 0x50, bytes.begin() + 0x80, '\x6D');
    bytes.replace(Toc, 4, "TOC\x01");
    SetLe(bytes, Toc + 4, PackageCount, 4);
    SetLe(bytes, Toc + 8, PackageCount, 8);
    SetLe(bytes, Toc + 16, PackageCount, 8);
    SetLe(bytes, Toc + 24, Pool - Toc, 8);
    SetLe(bytes, Toc + 32, pool.size(), 8);
    SetLe(bytes, Toc + 40, 0xABCDEF1234567890ULL, 8);
    bytes.replace(Pool, pool.size(), pool);
    for (std::size_t i = 0; i < PackageCount; ++i) {
        const auto package = Package(i);
        SetLe(bytes, package, 0x0400ABCD00010000ULL + i, 8);
        SetLe(bytes, package + 8, i, 4);
        SetLe(bytes, package + 12, 0xFFFFFFFF, 4);
        SetLe(bytes, package + 16, i, 4);
        SetLe(bytes, package + 20, i * 65536, 4);
        bytes[package + 24] = i == 0 ? 0 : (i == 3 ? 2 : 1);
        std::fill(bytes.begin() + package + 0x1A, bytes.begin() + package + 0x20, '\x72');
        std::fill(bytes.begin() + package + 0x2C, bytes.begin() + package + 0x40, '\x39');
        const auto directory = Directories + i * 0x18;
        SetLe(bytes, directory + 8, i, 4);
        SetLe(bytes, directory + 12, 1, 4);
        SetLe(bytes, directory + 16, 0x12345678, 8);
        SetLe(bytes, Children + i * 4, PackageCount + i, 4);
        const auto file = Files + i * 0x50;
        const std::string payload(32, static_cast<char>('a' + i));
        SetLe(bytes, file + 4, pool.size(), 4);
        SetLe(bytes, file + 8, payload.size(), 8);
        SetLe(bytes, file + 16, bytes.size(), 8);
        SetLe(bytes, file + 24, payload.size(), 8);
        const auto hash = Sha256(payload);
        std::copy(hash.begin(), hash.end(), bytes.begin() + file + 32);
        std::fill(bytes.begin() + file + 0x42, bytes.begin() + file + 0x50, '\x57');
        bytes += payload;
    }
    for (std::size_t i = 0; i < PackageCount; ++i) {
        const auto suffix = std::to_string(i);
        const auto meta = Tlv(1, "synthetic npdm " + suffix) + Tlv(5, "title " + suffix) +
                          Tlv(3, "icon " + suffix) + Tlv(6, "publisher " + suffix) +
                          Tlv(8, "source " + suffix) + Tlv(4, "version " + suffix) +
                          Tlv(9, Localized("titel " + suffix)) +
                          Tlv(10, Localized("pictogram " + suffix)) +
                          Tlv(0xBEEF, std::string("\0\xFF\x7F", 3)) + Tlv(0xBEEF, "second") +
                          Tlv(0xF00D, "");
        AppendMeta(bytes, i, meta);
    }
    bytes += "unreferenced trailing bytes must survive";
    SetLe(bytes, 0x40, bytes.size(), 8);
    return bytes;
}

std::string PreBlobFixture(std::size_t count) {
    auto bytes = Fixture();
    const auto toc_end = Le(bytes, 8, 8) + Le(bytes, 16, 8);
    const auto old_blob = Le(bytes, 0x38, 8);
    std::vector<std::string> metadata;
    std::size_t total = 0;
    for (std::size_t i = 0; i < count; ++i) {
        metadata.push_back(Meta(bytes, i));
        total += metadata.back().size();
    }
    const auto new_blob = (toc_end + total + 8 + 15) & ~u64{15};
    metadata.back() += Tlv(0xF00D, std::string(new_blob - toc_end - total - 8, '\0'));
    const auto shift = new_blob - old_blob;
    bytes.insert(old_blob, shift, '\0');
    u64 cursor = toc_end;
    for (std::size_t i = 0; i < PackageCount; ++i) {
        SetLe(bytes, Files + i * 0x50 + 16, Le(bytes, Files + i * 0x50 + 16, 8) + shift, 8);
        if (i < count) {
            bytes.replace(cursor, metadata[i].size(), metadata[i]);
            SetLe(bytes, Package(i) + 0x20, cursor, 8);
            SetLe(bytes, Package(i) + 0x28, metadata[i].size(), 4);
            cursor += metadata[i].size();
        } else {
            SetLe(bytes, Package(i) + 0x20, Le(bytes, Package(i) + 0x20, 8) + shift, 8);
        }
    }
    REQUIRE(cursor == new_blob);
    SetLe(bytes, 0x38, new_blob, 8);
    SetLe(bytes, 0x40, bytes.size(), 8);
    RefreshTocHash(bytes);
    return bytes;
}

void CheckPreserved(std::string_view before, std::string_view after, std::size_t target,
                    const MetadataUpdate& update) {
    REQUIRE(after.size() >= before.size());
    REQUIRE(Le(after, 0x40, 8) == after.size());
    const auto hash = Sha256(after.substr(Le(after, 8, 8), Le(after, 16, 8)));
    REQUIRE(std::equal(hash.begin(), hash.end(),
                       reinterpret_cast<const u8*>(after.data() + 0x18)));
    std::string restored(after.substr(0, before.size()));
    restored.replace(0x18, 32, before.substr(0x18, 32));
    restored.replace(0x40, 8, before.substr(0x40, 8));
    restored.replace(Package(target) + 0x20, 12, before.substr(Package(target) + 0x20, 12));
    REQUIRE(restored == before);
    for (std::size_t i = 0; i < PackageCount; ++i) {
        if (i != target) {
            REQUIRE(Meta(after, i) == Meta(before, i));
        }
    }
    REQUIRE(RetainedTlvs(Meta(after, target), update) ==
            RetainedTlvs(Meta(before, target), update));
}

void CheckContainer(const fs::path& path, std::string_view title, const std::vector<u8>& icon,
                    std::size_t target = 0) {
    FileSys::RealVfsFilesystem real;
    NupContainer nup{real.OpenFile(path.string(), FileSys::OpenMode::Read)};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(nup.VerifyIntegrity({}) == Loader::ResultStatus::Success);
    REQUIRE(nup.GetPackages().size() == PackageCount);
    REQUIRE(nup.GetFileCount() == PackageCount);
    for (std::size_t i = 0; i < PackageCount; ++i) {
        const auto suffix = std::to_string(i);
        const auto& package = nup.GetPackages()[i];
        REQUIRE(package.title_name == (i == target ? std::string(title) : "title " + suffix));
        const std::string original_icon = "icon " + suffix;
        REQUIRE(package.icon == (i == target ? icon : std::vector<u8>(original_icon.begin(),
                                                                      original_icon.end())));
        REQUIRE(package.publisher == "publisher " + suffix);
        REQUIRE(package.source_dump == "source " + suffix);
        REQUIRE(package.display_version == "version " + suffix);
        const auto npdm = "synthetic npdm " + suffix;
        REQUIRE(package.npdm == std::vector<u8>(npdm.begin(), npdm.end()));
        REQUIRE(package.localized_titles.size() == 1);
        REQUIRE(package.localized_titles[0].first == "nl");
        REQUIRE(package.localized_titles[0].second == "titel " + suffix);
        REQUIRE(package.localized_icons.size() == 1);
        REQUIRE(package.localized_icons[0].first == "nl");
        const auto localized_icon = "pictogram " + suffix;
        REQUIRE(package.localized_icons[0].second ==
                std::vector<u8>(localized_icon.begin(), localized_icon.end()));
        const auto dir = nup.GetRomFSDir(i);
        REQUIRE(dir != nullptr);
        const auto file = dir->GetFile("payload.bin");
        REQUIRE(file != nullptr);
        REQUIRE(file->ReadAllBytes() == std::vector<u8>(32, static_cast<u8>('a' + i)));
    }
}

void RejectUnchanged(const fs::path& archive, const MetadataUpdate& update) {
    const auto before = Read(archive);
    std::string error;
    const auto result = UpdateMetadata(archive, update, &error);
    INFO(error);
    REQUIRE(result == PackResult::InvalidSource);
    REQUIRE_FALSE(error.empty());
    REQUIRE(Read(archive) == before);
}

} // namespace

TEST_CASE("NUP metadata edits preserve every package and all original bytes", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    auto before = Fixture();
    Write(archive, before);
    const auto sentinel = fs::path(archive.string() + ".meta.part");
    Write(sentinel, "unrelated temporary file");
    const auto entries = std::distance(fs::directory_iterator(temp.path), fs::directory_iterator{});
    std::string title = "title 0";
    std::vector<u8> icon{'i', 'c', 'o', 'n', ' ', '0'};
    CheckContainer(archive, title, icon);
    const std::array<MetadataUpdate, 5> updates{{
        {.title_name = "Replacement title"},
        {.icon = std::vector<u8>{0, 0xFF, 0x80, 1}},
        {.title_name = "Second title", .icon = std::vector<u8>{3, 2, 1}},
        {.title_name = "", .icon = std::vector<u8>{}},
        {.title_name = "Restored title", .icon = std::vector<u8>{9, 8}},
    }};
    for (const auto& update : updates) {
        std::string error;
        const auto result = UpdateMetadata(archive, update, &error);
        INFO(error);
        REQUIRE(result == PackResult::Success);
        if (update.title_name) {
            title = *update.title_name;
        }
        if (update.icon) {
            icon = *update.icon;
        }
        const auto after = Read(archive);
        CheckPreserved(before, after, 0, update);
        REQUIRE(Le(after, Package(0) + 0x20, 8) >= before.size());
        CheckContainer(archive, title, icon);
        REQUIRE(Read(sentinel) == "unrelated temporary file");
        REQUIRE(std::distance(fs::directory_iterator(temp.path), fs::directory_iterator{}) == entries);
        before = after;
    }
    REQUIRE(UpdateMetadata(archive, {}) == PackResult::Success);
    REQUIRE(Read(archive) == before);
}

TEST_CASE("NUP metadata editors accept metadata between the TOC and blob", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    for (const std::size_t count : {std::size_t{1}, PackageCount}) {
        for (const bool dialog_editor : {false, true}) {
            DYNAMIC_SECTION(count << " pre-blob packages, dialog editor " << dialog_editor) {
                CAPTURE(count, dialog_editor);
                const auto before = PreBlobFixture(count);
                const auto blob = Le(before, 0x38, 8);
                Write(archive, before);
                const std::string original_icon = "icon 0";
                CheckContainer(archive, "title 0", {original_icon.begin(), original_icon.end()});
                const MetadataUpdate update{.title_name = "Custom title",
                                            .icon = std::vector<u8>{1, 2, 3}};
                std::string error;
                if (dialog_editor) {
                    const FileSys::Nup::MetadataEdit edit{.title_name = update.title_name,
                                                        .icon = update.icon};
                    const auto result = FileSys::Nup::EditNupMetadata(archive, edit, &error);
                    INFO(error);
                    REQUIRE(result == FileSys::Nup::EditResult::Success);
                    const auto after = Read(archive);
                    REQUIRE(after.substr(blob, before.size() - blob) == before.substr(blob));
                } else {
                    const auto result = UpdateMetadata(archive, update, &error);
                    INFO(error);
                    REQUIRE(result == PackResult::Success);
                    CheckPreserved(before, Read(archive), 0, update);
                }
                CheckContainer(archive, *update.title_name, *update.icon);
                REQUIRE(UpdateMetadata(archive, {.icon = std::vector<u8>{4, 5}}, &error) ==
                        PackResult::Success);
                CheckContainer(archive, *update.title_name, {4, 5});
            }
        }
    }
}

TEST_CASE("NUP metadata rejects ranges overlapping the TOC or crossing into the blob",
          "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    const auto original = PreBlobFixture(PackageCount);
    const auto toc_end = Le(original, 8, 8) + Le(original, 16, 8);
    for (const u64 offset : {u64{0x40}, Toc + u64{0x30}, toc_end - 1}) {
        auto bytes = original;
        SetLe(bytes, Package(0) + 0x20, offset, 8);
        RefreshTocHash(bytes);
        Write(archive, bytes);
        RejectUnchanged(archive, {.icon = std::vector<u8>{1}});
    }
    auto bytes = original;
    const auto last = Package(PackageCount - 1);
    SetLe(bytes, last + 0x28, Le(bytes, last + 0x28, 4) + 1, 4);
    RefreshTocHash(bytes);
    Write(archive, bytes);
    RejectUnchanged(archive, {.icon = std::vector<u8>{1}});
}

TEST_CASE("NUP metadata selects the latest update without a base", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    auto before = Fixture();
    before[Package(0) + 0x18] = 2;
    RefreshTocHash(before);
    Write(archive, before);
    REQUIRE(UpdateMetadata(archive, {.title_name = "Latest update"}) == PackResult::Success);
    CheckPreserved(before, Read(archive), 2, {.title_name = "Latest update"});
    CheckContainer(archive, "Latest update", {'i', 'c', 'o', 'n', ' ', '2'}, 2);
}

TEST_CASE("NUP metadata accepts an EOF boundary and absent package metadata", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    auto bytes = PreBlobFixture(PackageCount);
    SetLe(bytes, Package(1) + 0x20, 0, 8);
    SetLe(bytes, Package(1) + 0x28, 0, 4);
    AppendMeta(bytes, 0, Tlv(5, "At EOF"));
    Write(archive, bytes);
    const MetadataUpdate update{.icon = std::vector<u8>{9, 8, 7}};
    std::string error;
    const auto result = UpdateMetadata(archive, update, &error);
    INFO(error);
    REQUIRE(result == PackResult::Success);
    CheckPreserved(bytes, Read(archive), 0, update);
    FileSys::RealVfsFilesystem real;
    NupContainer nup{real.OpenFile(archive.string(), FileSys::OpenMode::Read)};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(nup.VerifyIntegrity({}) == Loader::ResultStatus::Success);
    REQUIRE(nup.GetPackages()[0].icon == *update.icon);
    REQUIRE(nup.GetPackages()[1].title_name.empty());
}

TEST_CASE("NUP edited display title overrides the selected language without changing control data",
          "[nup][metadata]") {
    TempDir temp;
    const auto root = temp.path / "Game";
    fs::create_directories(root / "Base/exefs");
    fs::create_directories(root / "Base/romfs");
    Write(root / "Base/exefs/main", "synthetic main");
    std::string npdm(0x100, '\0');
    npdm.replace(0x80, 4, "ACI0");
    SetLe(npdm, 0x90, 0x0400ABCD00010000ULL, 8);
    Write(root / "Base/exefs/main.npdm", npdm);
    Write(root / "Base/romfs/dummy", "synthetic data");
    const auto archive = temp.path / "Game.nup";
    std::string error;
    const auto packed = FileSys::Nup::PackDump(root, archive, {}, {}, nullptr, &error);
    INFO(error);
    REQUIRE(packed == PackResult::Success);

    auto bytes = Read(archive);
    const auto toc = static_cast<std::size_t>(Le(bytes, 8, 8));
    REQUIRE(Le(bytes, toc + 4, 4) == 1);
    const auto package = toc + 0x30;
    auto metadata = bytes.substr(Le(bytes, package + 0x20, 8), Le(bytes, package + 0x28, 4));
    const std::string dutch_title = "Oorspronkelijke Nederlandse titel";
    const auto localized = Tlv(9, Localized(dutch_title));
    metadata += localized;
    SetLe(bytes, package + 0x20, bytes.size(), 8);
    SetLe(bytes, package + 0x28, metadata.size(), 4);
    bytes += metadata;
    SetLe(bytes, 0x40, bytes.size(), 8);
    RefreshTocHash(bytes);
    Write(archive, bytes);

    struct RestoreLanguage {
        Settings::Language value = Settings::values.language_index.GetValue();

        ~RestoreLanguage() {
            Settings::values.language_index.SetValue(value);
        }
    } restore;
    Settings::values.language_index.SetValue(Settings::Language::Dutch);
    {
        FileSys::RealVfsFilesystem real;
        Loader::AppLoader_NUP loader{real.OpenFile(archive.string(), FileSys::OpenMode::Read)};
        FileSys::NACP control;
        REQUIRE(loader.ReadControlData(control) == Loader::ResultStatus::Success);
        REQUIRE(control.GetApplicationName() == dutch_title);
    }

    const std::string edited_title = "My edited display title";
    REQUIRE(UpdateMetadata(archive, {.title_name = edited_title}, &error) == PackResult::Success);
    FileSys::RealVfsFilesystem real;
    Loader::AppLoader_NUP loader{real.OpenFile(archive.string(), FileSys::OpenMode::Read)};
    std::string displayed_title;
    REQUIRE(loader.ReadTitle(displayed_title) == Loader::ResultStatus::Success);
    REQUIRE(displayed_title == edited_title);
    FileSys::NACP control;
    REQUIRE(loader.ReadControlData(control) == Loader::ResultStatus::Success);
    REQUIRE(control.GetApplicationName() == dutch_title);
    NupContainer nup{real.OpenFile(archive.string(), FileSys::OpenMode::Read)};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(nup.VerifyIntegrity({}) == Loader::ResultStatus::Success);
    REQUIRE(nup.GetPackages().size() == 1);
    REQUIRE(nup.GetPackages()[0].title_name == edited_title);
    REQUIRE(nup.GetPackages()[0].localized_titles.size() == 1);
    REQUIRE(nup.GetPackages()[0].localized_titles[0].first == "nl");
    REQUIRE(nup.GetPackages()[0].localized_titles[0].second == dutch_title);
    const auto after = Read(archive);
    const auto after_metadata =
        after.substr(Le(after, package + 0x20, 8), Le(after, package + 0x28, 4));
    REQUIRE(after_metadata.find(localized) != std::string::npos);
}

TEST_CASE("NUP metadata rejects hostile ranges and counts without modifying the archive",
          "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    constexpr auto max = std::numeric_limits<u64>::max();
    struct Mutation {
        const char* name;
        std::size_t offset;
        u64 value;
        std::size_t size;
        bool rehash;
    };
    const auto original = Fixture();
    const std::vector<Mutation> mutations{
        {"overflowing TOC offset", 8, max - 15, 8, false},
        {"overflowing TOC size", 16, max, 8, false},
        {"TOC outside archive", 8, original.size() + 16, 8, false},
        {"TOC overlaps superblock", 8, 0x40, 8, false},
        {"TOC overlaps blob", 16, original.size(), 8, false},
        {"short TOC header", 16, 0x20, 8, true},
        {"blob outside archive", 0x38, max, 8, false},
        {"declared size mismatch", 0x40, original.size() + 1, 8, false},
        {"zero packages", 0x48, 0, 4, false},
        {"package count mismatch", Toc + 4, 5, 4, true},
        {"overflowing directory count", Toc + 8, 1ULL << 61, 8, true},
        {"overflowing file count", Toc + 16, 1ULL << 60, 8, true},
        {"truncated directory records", Toc + 8, 1000, 8, true},
        {"truncated file records", Toc + 16, 1000, 8, true},
        {"overflowing pool offset", Toc + 24, max, 8, true},
        {"overflowing pool length", Toc + 32, max, 8, true},
        {"pool overlaps records", Toc + 24, 0x30, 8, true},
        {"overflowing extent offset", Files + 16, max - 15, 8, true},
        {"overflowing extent length", Files + 24, max, 8, true},
        {"extent before blob", Files + 16, 0x80, 8, true},
        {"extent beyond EOF", Files + 16, original.size() + 16, 8, true},
        {"extent straddles EOF", Files + 24, original.size(), 8, true},
        {"overflowing metadata offset", Package(0) + 0x20, max - 7, 8, true},
        {"metadata straddles EOF", Package(0) + 0x28, original.size(), 4, true},
        {"non-target metadata outside archive", Package(3) + 0x20, max - 7, 8, true},
        {"metadata without an offset", Package(0) + 0x20, 0, 8, true},
    };
    for (const auto& mutation : mutations) {
        INFO(mutation.name);
        auto bytes = original;
        SetLe(bytes, mutation.offset, mutation.value, mutation.size);
        if (mutation.rehash) {
            RefreshTocHash(bytes);
        }
        Write(archive, bytes);
        RejectUnchanged(archive, {.title_name = "rejected"});
    }
    auto bytes = original;
    SetLe(bytes, 0x48, 0xFFFFFFFF, 4);
    SetLe(bytes, Toc + 4, 0xFFFFFFFF, 4);
    RefreshTocHash(bytes);
    Write(archive, bytes);
    RejectUnchanged(archive, {.icon = std::vector<u8>{1}});
}

TEST_CASE("NUP metadata rejects truncated TLVs in every package", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    for (std::size_t package = 0; package < PackageCount; ++package) {
        for (std::size_t length = 1; length < 8; ++length) {
            INFO(package);
            INFO(length);
            auto bytes = Fixture();
            AppendMeta(bytes, package, Meta(bytes, package) + std::string(length, '\x42'));
            Write(archive, bytes);
            RejectUnchanged(archive, {.title_name = "rejected"});
        }
        auto bytes = Fixture();
        auto malformed = Tlv(0xCAFE, "x");
        SetLe(malformed, 4, 0xFFFFFFFF, 4);
        AppendMeta(bytes, package, malformed);
        Write(archive, bytes);
        RejectUnchanged(archive, {.icon = std::vector<u8>{1}});
    }
}

TEST_CASE("NUP metadata enforces title and icon update limits", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    Write(archive, Fixture());
    RejectUnchanged(archive, {.title_name = std::string(4097, 't')});
    RejectUnchanged(archive, {.icon = std::vector<u8>(IconLimit + 1, 0x42)});
    const MetadataUpdate limit{.title_name = std::string(4096, 't'),
                               .icon = std::vector<u8>(IconLimit, 0x42)};
    REQUIRE(UpdateMetadata(archive, limit) == PackResult::Success);
    CheckContainer(archive, *limit.title_name, *limit.icon);
}

TEST_CASE("NUP metadata rejects malformed localized payloads in every package", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    auto overflowing = Localized("x");
    SetLe(overflowing, 3, 0xFFFFFFFF, 4);
    const std::array<std::string, 5> malformed{{
        std::string("\x05nl", 3),
        std::string("\x02nl", 3),
        Localized("x").substr(0, 7),
        Localized("x") + std::string(1, '\xFF'),
        overflowing,
    }};
    for (std::size_t package = 0; package < PackageCount; ++package) {
        for (const u16 type : {u16{9}, u16{10}}) {
            for (std::size_t i = 0; i < malformed.size(); ++i) {
                INFO(package);
                INFO(type);
                INFO(i);
                auto bytes = Fixture();
                AppendMeta(bytes, package, Meta(bytes, package) + Tlv(type, malformed[i]));
                Write(archive, bytes);
                RejectUnchanged(archive, {.title_name = "rejected", .icon = std::vector<u8>{1}});
            }
        }
    }
}

#ifdef _WIN32
TEST_CASE("NUP metadata preserves the archive when Windows blocks replacement",
          "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    const auto before = Fixture();
    Write(archive, before);
    const auto sentinel = fs::path(archive.string() + ".meta.part");
    Write(sentinel, "unrelated temporary file");
    CheckContainer(archive, "title 0", {'i', 'c', 'o', 'n', ' ', '0'});
    Common::FS::IOFile held{archive, Common::FS::FileAccessMode::Read};
    REQUIRE(held.IsOpen());
    std::string error;
    const auto result = UpdateMetadata(archive, {.title_name = "blocked title"}, &error);
    INFO(error);
    REQUIRE(result == PackResult::IoError);
    REQUIRE(error.starts_with("cannot replace NUP archive:"));
    REQUIRE(Read(archive) == before);
    REQUIRE(Read(sentinel) == "unrelated temporary file");
    REQUIRE(std::distance(fs::directory_iterator(temp.path), fs::directory_iterator{}) == 2);
    held.Close();
    REQUIRE(UpdateMetadata(archive, {.title_name = "retry title"}, &error) == PackResult::Success);
    CheckContainer(archive, "retry title", {'i', 'c', 'o', 'n', ' ', '0'});
    REQUIRE(Read(sentinel) == "unrelated temporary file");
    REQUIRE(std::distance(fs::directory_iterator(temp.path), fs::directory_iterator{}) == 2);
}
#elif defined(__linux__)
TEST_CASE("NUP metadata preserves the archive when transaction creation is denied",
          "[nup][metadata]") {
    if (geteuid() == 0) {
        SKIP("Root can create transactions despite directory permissions");
    }
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    const auto before = Fixture();
    Write(archive, before);
    const auto sentinel = fs::path(archive.string() + ".meta.part");
    Write(sentinel, "unrelated temporary file");
    CheckContainer(archive, "title 0", {'i', 'c', 'o', 'n', ' ', '0'});
    const auto original_permissions = fs::status(temp.path).permissions();
    {
        struct RestorePermissions {
            fs::path path;
            fs::perms permissions;

            ~RestorePermissions() {
                std::error_code ec;
                fs::permissions(path, permissions, ec);
            }
        } restore{temp.path, original_permissions};
        fs::permissions(temp.path, fs::perms::owner_write | fs::perms::group_write |
                                       fs::perms::others_write,
                        fs::perm_options::remove);
        std::string error;
        const auto result = UpdateMetadata(archive, {.title_name = "blocked title"}, &error);
        INFO(error);
        REQUIRE(result == PackResult::IoError);
        REQUIRE(error == "cannot create metadata transaction directory");
        REQUIRE(Read(archive) == before);
        REQUIRE(Read(sentinel) == "unrelated temporary file");
        REQUIRE(std::distance(fs::directory_iterator(temp.path), fs::directory_iterator{}) == 2);
    }
    REQUIRE(fs::status(temp.path).permissions() == original_permissions);
    REQUIRE(UpdateMetadata(archive, {.title_name = "retry title"}) == PackResult::Success);
    CheckContainer(archive, "retry title", {'i', 'c', 'o', 'n', ' ', '0'});
    REQUIRE(Read(sentinel) == "unrelated temporary file");
    REQUIRE(std::distance(fs::directory_iterator(temp.path), fs::directory_iterator{}) == 2);
}
#endif

TEST_CASE("NUP metadata bounds aggregate metadata across packages", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    auto bytes = Fixture();
    const auto offset = bytes.size();
    const auto meta = Tlv(0xCAFE, std::string(MetadataLimit / PackageCount, 'x'));
    bytes += meta;
    for (std::size_t i = 0; i < PackageCount; ++i) {
        SetLe(bytes, Package(i) + 0x20, offset, 8);
        SetLe(bytes, Package(i) + 0x28, meta.size(), 4);
    }
    SetLe(bytes, 0x40, bytes.size(), 8);
    RefreshTocHash(bytes);
    Write(archive, bytes);
    RejectUnchanged(archive, {.title_name = "rejected"});
}

TEST_CASE("NUP metadata accepts the aggregate limit but rejects a larger replacement",
          "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "archive.nup";
    auto bytes = Fixture();
    const auto offset = bytes.size();
    const auto meta = Tlv(0xCAFE, std::string(MetadataLimit / PackageCount - 8, 'x'));
    bytes += meta;
    for (std::size_t i = 0; i < PackageCount; ++i) {
        SetLe(bytes, Package(i) + 0x20, offset, 8);
        SetLe(bytes, Package(i) + 0x28, meta.size(), 4);
    }
    SetLe(bytes, 0x40, bytes.size(), 8);
    RefreshTocHash(bytes);
    Write(archive, bytes);
    RejectUnchanged(archive, {.icon = std::vector<u8>{1}});
    const MetadataUpdate update{.icon = std::vector<u8>{}};
    std::string error;
    const auto result = UpdateMetadata(archive, update, &error);
    INFO(error);
    REQUIRE(result == PackResult::Success);
    CheckPreserved(bytes, Read(archive), 0, update);
}

TEST_CASE("NUP metadata rejects an oversized valid sparse TOC", "[nup][metadata]") {
    TempDir temp;
    const auto archive = temp.path / "sparse.nup";
    constexpr u64 toc_size = MetadataLimit + 16;
    constexpr u64 file_size = Toc + toc_size;
    std::string prefix(Toc + 0x30 + 0x40 + 0x18, '\0');
    prefix.replace(0, 4, "NUP\x01");
    SetLe(prefix, 4, 0x80, 2);
    SetLe(prefix, 6, 1, 2);
    SetLe(prefix, 8, Toc, 8);
    SetLe(prefix, 16, toc_size, 8);
    SetLe(prefix, 0x38, file_size, 8);
    SetLe(prefix, 0x40, file_size, 8);
    SetLe(prefix, 0x48, 1, 4);
    prefix.replace(Toc, 4, "TOC\x01");
    SetLe(prefix, Toc + 4, 1, 4);
    SetLe(prefix, Toc + 8, 1, 8);
    SetLe(prefix, Toc + 24, prefix.size() - Toc, 8);
    SetLe(prefix, Package(0) + 12, 0xFFFFFFFF, 4);
    Write(archive, prefix);
    fs::resize_file(archive, file_size);
    const auto toc_hash = HashRange(archive, Toc, toc_size);
    {
        std::fstream out(archive, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(out.is_open());
        out.seekp(0x18);
        out.write(reinterpret_cast<const char*>(toc_hash.data()), toc_hash.size());
        out.close();
        REQUIRE_FALSE(out.fail());
    }
    const auto before = HashRange(archive, 0, file_size);
    std::string error;
    const auto result = UpdateMetadata(archive, {.title_name = "rejected"}, &error);
    INFO(error);
    REQUIRE(result == PackResult::InvalidSource);
    REQUIRE_FALSE(error.empty());
    REQUIRE(fs::file_size(archive) == file_size);
    REQUIRE(HashRange(archive, 0, file_size) == before);
}

TEST_CASE("NUP metadata fails safely for invalid paths and incomplete archives", "[nup][metadata]") {
    TempDir temp;
    const auto missing = temp.path / "missing" / "archive.nup";
    std::string error;
    REQUIRE(UpdateMetadata(missing, {.title_name = "new title"}, &error) == PackResult::IoError);
    REQUIRE_FALSE(error.empty());
    REQUIRE_FALSE(fs::exists(missing.parent_path()));
    const auto archive = temp.path / "archive.nup";
    Write(archive, "NUP\x01");
    RejectUnchanged(archive, {.title_name = "new title"});
    auto bytes = Fixture();
    bytes[0] = 'X';
    Write(archive, bytes);
    RejectUnchanged(archive, {.title_name = "new title"});
    bytes = Fixture();
    bytes[Toc + 40] ^= 1;
    Write(archive, bytes);
    RejectUnchanged(archive, {.title_name = "new title"});
}
