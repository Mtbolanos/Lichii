// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/loader/loader.h"

namespace FileSys {

struct NupDecompressCache;

/**
 * NUP (NX Package) container: stores loose ExeFS/RomFS trees for a base program,
 * its updates and add-on content in one file. Layout:
 *
 *   Superblock (0x80 bytes, magic "NUP\x01")
 *   Table of Contents (packages, dir/file nodes, children array, string pool)
 *   Blob region (per-file extents, raw or zstd codec, 16-byte aligned)
 *
 * Per-package meta TLVs carry NPDM bytes, title names (default + localized),
 * icons and a display version. All integers little-endian.
 */
class NupContainer {
public:
    static constexpr std::array<u8, 4> HeaderMagic{'N', 'U', 'P', 0x01};
    /// Memory budget for decoded zstd members.
    static constexpr std::size_t DefaultCacheCap = 512 * 1024 * 1024;

    enum class PackageKind : u8 {
        Base = 0,
        Update = 1,
        Addon = 2,
    };


    struct PackageInfo {
        u64 title_id{};
        u32 version{};
        PackageKind kind{};
        bool standalone_update{};
        u32 root_dir{};
        u32 exefs_dir{};
        u32 romfs_dir{};
        u64 meta_offset{};
        u32 meta_size{};

        std::vector<u8> npdm;
        std::string title_name;
        std::string publisher;
        std::string source_dump;
        std::vector<std::pair<std::string, std::string>> localized_titles;
        std::vector<u8> icon;
        std::vector<std::pair<std::string, std::vector<u8>>> localized_icons;
        std::string display_version;
    };

    /// Quick magic check without a full parse.
    static bool IsNupFile(const VirtualFile& file);

    /// Recursively merges overlay into a copy of base; overlay paths win.
    static VirtualDir MergeShadowDirs(const VirtualDir& base, const VirtualDir& overlay);

    /// Zstd members larger than half of `memory_cache_cap` are decompressed to a temporary file
    /// in `spill_dir` (the cache directory by default) instead of being held in memory.
    explicit NupContainer(VirtualFile file_, std::size_t memory_cache_cap = DefaultCacheCap,
                          std::filesystem::path spill_dir_ = {});
    ~NupContainer();

    Loader::ResultStatus GetStatus() const {
        return status;
    }

    const std::vector<PackageInfo>& GetPackages() const {
        return packages;
    }

    std::optional<std::size_t> FindPackage(PackageKind kind,
                                           std::optional<u64> title_id = std::nullopt) const;
    std::optional<std::size_t> FindBasePackage() const;
    /// Highest version among Update packages, if any.
    std::optional<std::size_t> FindBestUpdatePackage() const;

    /// Opens a member file by node index; raw extents are zero-copy views into
    /// the container, zstd extents are decompressed only as far as reads reach,
    /// cached in memory or, when large, in a temporary file.
    VirtualFile OpenFile(std::size_t file_index);

    VirtualDir GetExeFSDir(std::size_t package_index);
    VirtualDir GetRomFSDir(std::size_t package_index);
    VirtualDir GetPackageDir(std::size_t package_index);

    /// Stream-hashes every member file against its stored SHA-256.
    Loader::ResultStatus VerifyIntegrity(
        std::function<bool(std::size_t, std::size_t)> progress_callback) const;

    std::size_t GetFileCount() const {
        return files.size();
    }

    bool IsZstdMember(std::size_t file_index) const {
        return file_index < files.size() && files[file_index].size != 0 &&
               files[file_index].codec != 0;
    }

private:
    struct DirNode {
        std::string name;
        u32 first_child{};
        u32 child_count{};
    };

    struct FileNode {
        std::string name;
        u64 size{};
        std::array<u8, 0x20> sha256{};
        u64 extent_offset{};
        u64 extent_size{};
        u8 codec{};
    };

    bool Parse();
    bool ParseMeta(PackageInfo& package);
    bool ValidateDirTree() const;
    /// With serve_stand_ins, a .zs member with no zstd header is served from a sound sibling.
    void BuildDirTree(u32 dir_index, VectorVfsDirectory& out, bool serve_stand_ins);
    VirtualFile OpenFileAs(std::size_t file_index, const std::string& name);
    /// Whether a member starts with the zstd frame magic; nothing when it cannot be read.
    std::optional<bool> HasZstdMagic(std::size_t file_index) const;
    /// A sound sibling to serve in place of a damaged .zs member, if any.
    std::optional<std::size_t> FindStandIn(std::size_t damaged,
                                           std::span<const std::size_t> siblings) const;
    VirtualDir BuildPackageDir(u32 dir_index, bool serve_stand_ins);
    std::vector<u8> ReadExtentRaw(const FileNode& node) const;

    VirtualFile file;
    Loader::ResultStatus status{Loader::ResultStatus::ErrorNullFile};

    u64 toc_offset{};
    u64 toc_size{};
    u64 blob_offset{};
    u64 file_size{};

    std::vector<DirNode> dirs;
    std::vector<FileNode> files;
    std::vector<u32> children;
    std::vector<PackageInfo> packages;

    std::vector<VirtualDir> exefs_dir_cache;
    std::vector<VirtualDir> romfs_dir_cache;

    std::shared_ptr<NupDecompressCache> cache;
};

} // namespace FileSys
