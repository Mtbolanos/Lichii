// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Core::Nx2Debug {

/// Writes a zip file (deflate or store, no zip64, so under 4 GiB) for the bug bundle. Entries are
/// compressed before they are written, so the caller can check what each one costs first.
class ZipWriter {
public:
    struct Entry {
        std::string name;
        std::vector<u8> data;
        u32 crc{};
        u32 size{};
        bool deflated{};

        /// Bytes it takes in the zip: its local header, name and data, and its central record.
        [[nodiscard]] u64 ZipBytes() const noexcept;
    };

    /// Throws std::runtime_error when the file cannot be created.
    explicit ZipWriter(const std::filesystem::path& path);

    /// Deflates the data, or stores it when deflating does not make it smaller.
    [[nodiscard]] static Entry Prepare(std::string name, std::span<const u8> data, bool compress);

    void Add(const Entry& entry);
    /// Writes the central directory. Throws std::runtime_error when a write failed.
    void Finish();

    /// The bytes the zip will take once finished with the entries added so far.
    [[nodiscard]] u64 Size() const noexcept;

private:
    struct Record {
        std::string name;
        u32 crc;
        u32 compressed;
        u32 size;
        u32 offset;
        bool deflated;
    };

    std::ofstream file;
    std::vector<Record> records;
    u64 offset{};
    u64 central_bytes{};
    u16 dos_time{};
    u16 dos_date{};
};

} // namespace Core::Nx2Debug
