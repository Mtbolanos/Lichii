// SPDX-FileCopyrightText: Copyright 2026 the ScarletNX contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <zlib.h>

#include "common/swap.h"
#include "core/file_sys/system_archive/mii_model.h"
#include "core/file_sys/vfs/vfs.h"

namespace {

u32 Read32(std::span<const u8> bytes, std::size_t offset) {
    REQUIRE(offset + sizeof(u32_le) <= bytes.size());
    u32_le value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::vector<u8> InflatePart(std::span<const u8> file, std::size_t entry) {
    const auto offset = Read32(file, entry);
    const auto size = Read32(file, entry + 4);
    const auto compressed_size = Read32(file, entry + 8);
    REQUIRE(size > 0);
    REQUIRE(size <= 0x1000);
    REQUIRE(offset <= file.size());
    REQUIRE(compressed_size <= file.size() - offset);
    std::vector<u8> output(size);
    uLongf written = output.size();
    REQUIRE(uncompress(output.data(), &written, file.data() + offset, compressed_size) == Z_OK);
    REQUIRE(written == size);
    return output;
}

void CheckHeader(std::span<const u8> file, const char* magic, std::size_t kinds) {
    REQUIRE(file.size() >= 0xc + kinds * 8);
    REQUIRE(std::memcmp(file.data(), magic, 4) == 0);
    REQUIRE(Read32(file, 4) == 1);
    for (std::size_t kind = 0; kind < kinds; ++kind) {
        REQUIRE(Read32(file, 0xc + kind * 4) > 0);
        const auto alignment = Read32(file, 0xc + (kinds + kind) * 4);
        REQUIRE(alignment > 0);
        REQUIRE((alignment & (alignment - 1)) == 0);
    }
}

} // Anonymous namespace

TEST_CASE("MiiModel fallback provides every texture part without firmware", "[core][mii_model]") {
    const auto archive = FileSys::SystemArchive::MiiModel();
    REQUIRE(archive->GetFiles().size() == 6);
    // SDK part-kind ordering; this covers every lookup, including the save-slot face parts.
    constexpr std::array<u16, 11> counts{132, 62, 24, 2, 12, 12, 20, 2, 37, 6, 18};
    constexpr std::array<u8, 11> formats{4, 8, 8, 10, 10, 10, 1, 8, 8, 9, 1};
    for (const char* name : {"NXTextureLowLinear.dat", "NXTextureLowSRGB.dat",
                             "NXTextureMidLinear.dat", "NXTextureMidSRGB.dat"}) {
        INFO(name);
        const auto file = archive->GetFile(name);
        REQUIRE(file != nullptr);
        const auto data = file->ReadAllBytes();
        CheckHeader(data, "NFTR", counts.size());
        REQUIRE(data.size() < 0x10000);
        std::size_t entry = 0x64;
        for (std::size_t kind = 0; kind < counts.size(); ++kind) {
            for (u16 part = 0; part < counts[kind]; ++part, entry += 0x1c) {
                CAPTURE(kind, part);
                REQUIRE(Read32(data, entry) >= 0x64 + 327 * 0x1c);
                const auto pixels = InflatePart(data, entry);
                REQUIRE(std::ranges::all_of(pixels, [](u8 byte) { return byte == 0; }));
                REQUIRE(Read32(data, entry + 4) <= Read32(data, 0xc + kind * 4));
                REQUIRE(Read32(data, entry + 0x10) == 0x200);
                REQUIRE(Read32(data, entry + 0x14) == 0x00080008);
                REQUIRE(data.at(entry + 0x18) == 1);
                REQUIRE(data.at(entry + 0x19) == formats[kind]);
            }
        }
    }
}

TEST_CASE("MiiModel fallback shapes have valid degenerate geometry", "[core][mii_model]") {
    const auto archive = FileSys::SystemArchive::MiiModel();
    for (const char* name : {"ShapeMid.dat", "ShapeHigh.dat"}) {
        INFO(name);
        const auto file = archive->GetFile(name);
        REQUIRE(file != nullptr);
        const auto data = file->ReadAllBytes();
        CheckHeader(data, "NFSR", 12);
        REQUIRE(data.size() < 0x20000);
        for (std::size_t part = 0; part < 857; ++part) {
            CAPTURE(part);
            const auto entry = 0x6c + part * 0x58;
            REQUIRE(Read32(data, entry) >= 0x6c + 857 * 0x58 + 132 * 0x48 + 12 * 0x24);
            const auto mesh = InflatePart(data, entry);
            const auto vertex_offset = Read32(data, entry + 16);
            const auto vertex_bytes = Read32(data, entry + 36);
            REQUIRE(vertex_bytes == 8);
            REQUIRE(vertex_offset + vertex_bytes <= mesh.size());
            const auto index_offset = Read32(data, entry + 56);
            const auto index_bytes = Read32(data, entry + 60);
            REQUIRE(index_bytes == 6);
            REQUIRE(index_offset + index_bytes <= mesh.size());
            REQUIRE(std::ranges::all_of(mesh, [](u8 byte) { return byte == 0; }));
        }
    }
}
