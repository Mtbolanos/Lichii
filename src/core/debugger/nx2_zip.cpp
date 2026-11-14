// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ctime>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>
#include <zlib.h>

#include "core/debugger/nx2_zip.h"

namespace Core::Nx2Debug {
namespace {

constexpr u64 LOCAL_HEADER = 30;
constexpr u64 CENTRAL_HEADER = 46;
constexpr u64 END_RECORD = 22;
/// General purpose flag 11: the name is UTF-8.
constexpr u16 UTF8_NAME = 1 << 11;

void Put16(std::string& out, u16 value) {
    out += static_cast<char>(value & 0xff);
    out += static_cast<char>(value >> 8);
}

void Put32(std::string& out, u32 value) {
    Put16(out, static_cast<u16>(value & 0xffff));
    Put16(out, static_cast<u16>(value >> 16));
}

std::vector<u8> RawDeflate(std::span<const u8> data) {
    z_stream stream{};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error{"deflateInit2 failed"};
    }
    std::vector<u8> out(deflateBound(&stream, static_cast<uLong>(data.size())));
    stream.next_in = const_cast<Bytef*>(data.data());
    stream.avail_in = static_cast<uInt>(data.size());
    stream.next_out = out.data();
    stream.avail_out = static_cast<uInt>(out.size());
    const int result = deflate(&stream, Z_FINISH);
    out.resize(stream.total_out);
    deflateEnd(&stream);
    if (result != Z_STREAM_END) {
        throw std::runtime_error{"deflate failed"};
    }
    return out;
}

} // Anonymous namespace

u64 ZipWriter::Entry::ZipBytes() const noexcept {
    return LOCAL_HEADER + CENTRAL_HEADER + 2 * name.size() + data.size();
}

ZipWriter::ZipWriter(const std::filesystem::path& path) : file{path, std::ios::binary | std::ios::trunc} {
    if (!file) {
        throw std::runtime_error{fmt::format("cannot create {}", path.generic_string())};
    }
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    dos_time = static_cast<u16>((local.tm_hour << 11) | (local.tm_min << 5) | (local.tm_sec / 2));
    dos_date = static_cast<u16>(((std::max)(local.tm_year - 80, 0) << 9) |
                                ((local.tm_mon + 1) << 5) | local.tm_mday);
}

ZipWriter::Entry ZipWriter::Prepare(std::string name, std::span<const u8> data, bool compress) {
    if (data.size() > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error{fmt::format("{} is too large for a zip entry", name)};
    }
    Entry entry{
        .name = std::move(name),
        .crc = static_cast<u32>(crc32_z(crc32_z(0, nullptr, 0), data.data(), data.size())),
        .size = static_cast<u32>(data.size()),
    };
    if (compress && !data.empty()) {
        std::vector<u8> deflated = RawDeflate(data);
        if (deflated.size() < data.size()) {
            entry.data = std::move(deflated);
            entry.deflated = true;
            return entry;
        }
    }
    entry.data.assign(data.begin(), data.end());
    return entry;
}

void ZipWriter::Add(const Entry& entry) {
    if (offset + entry.ZipBytes() + END_RECORD > (std::numeric_limits<u32>::max)() ||
        records.size() == (std::numeric_limits<u16>::max)()) {
        throw std::runtime_error{"the zip would pass 4 GiB or 65535 entries"};
    }
    const u16 method = entry.deflated ? 8 : 0;
    std::string header;
    Put32(header, 0x04034b50);
    Put16(header, 20);
    Put16(header, UTF8_NAME);
    Put16(header, method);
    Put16(header, dos_time);
    Put16(header, dos_date);
    Put32(header, entry.crc);
    Put32(header, static_cast<u32>(entry.data.size()));
    Put32(header, entry.size);
    Put16(header, static_cast<u16>(entry.name.size()));
    Put16(header, 0);
    header += entry.name;
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(entry.data.data()),
               static_cast<std::streamsize>(entry.data.size()));
    records.push_back({.name = entry.name,
                       .crc = entry.crc,
                       .compressed = static_cast<u32>(entry.data.size()),
                       .size = entry.size,
                       .offset = static_cast<u32>(offset),
                       .deflated = entry.deflated});
    offset += header.size() + entry.data.size();
    central_bytes += CENTRAL_HEADER + entry.name.size();
}

void ZipWriter::Finish() {
    std::string central;
    for (const Record& record : records) {
        Put32(central, 0x02014b50);
        Put16(central, 20);
        Put16(central, 20);
        Put16(central, UTF8_NAME);
        Put16(central, record.deflated ? 8 : 0);
        Put16(central, dos_time);
        Put16(central, dos_date);
        Put32(central, record.crc);
        Put32(central, record.compressed);
        Put32(central, record.size);
        Put16(central, static_cast<u16>(record.name.size()));
        Put16(central, 0);
        Put16(central, 0);
        Put16(central, 0);
        Put16(central, 0);
        Put32(central, 0);
        Put32(central, record.offset);
        central += record.name;
    }
    Put32(central, 0x06054b50);
    Put16(central, 0);
    Put16(central, 0);
    Put16(central, static_cast<u16>(records.size()));
    Put16(central, static_cast<u16>(records.size()));
    Put32(central, static_cast<u32>(central_bytes));
    Put32(central, static_cast<u32>(offset));
    Put16(central, 0);
    file.write(central.data(), static_cast<std::streamsize>(central.size()));
    file.close();
    if (!file) {
        throw std::runtime_error{"writing the zip failed"};
    }
}

u64 ZipWriter::Size() const noexcept {
    return offset + central_bytes + END_RECORD;
}

} // namespace Core::Nx2Debug
