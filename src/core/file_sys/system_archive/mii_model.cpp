// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 the ScarletNX contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <span>
#include <vector>

#include "common/swap.h"
#include "core/file_sys/system_archive/mii_model.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace FileSys::SystemArchive {
namespace {

// Adapted from ScarletNX #863 (05324b57629d0894506b31beb2c72869138b7ad1).
// Missing parts leave null textures/buffers in the guest's Mii renderer. Supply blank textures
// and degenerate triangles; installed firmware still supplies the real models before this fallback.
constexpr std::size_t TextureDataOffset = 0x2430;
constexpr u32 TextureBytes = 0x200;
constexpr std::size_t ShapeDataOffset = 0x14de0;
constexpr u32 ShapeBytes = 0x20;
constexpr u32 ZlibOverhead = 11;
constexpr std::size_t ShapeCount = 857;

struct TextureKind {
    u16 count;
    u8 format;
};
constexpr std::array<TextureKind, 11> TextureKinds{{
    {132, 4}, {62, 8}, {24, 8}, {2, 10}, {12, 10}, {12, 10},
    {20, 1}, {2, 8}, {37, 8}, {6, 9}, {18, 1},
}};
static_assert(0x64 + 327 * 0x1c <= TextureDataOffset);
static_assert(0x6c + ShapeCount * 0x58 + 132 * 0x48 + 12 * 0x24 <= ShapeDataOffset);

void Write32(std::span<u8> data, std::size_t offset, u32 value) {
    const u32_le encoded{value};
    std::memcpy(data.data() + offset, &encoded, sizeof(encoded));
}

void Write16(std::span<u8> data, std::size_t offset, u16 value) {
    const u16_le encoded{value};
    std::memcpy(data.data() + offset, &encoded, sizeof(encoded));
}

void WriteHeader(std::span<u8> data, const char* magic, std::size_t kinds) {
    std::memcpy(data.data(), magic, 4);
    Write32(data, 4, 1);
    for (std::size_t kind = 0; kind < kinds; ++kind) {
        Write32(data, 0xc + kind * 4, 0x1000);
        Write32(data, 0xc + (kinds + kind) * 4, 0x1000);
    }
}

// A single final stored DEFLATE block of zero bytes. Adler-32 is (size << 16) | 1.
void WriteBlankZlib(std::span<u8> data, std::size_t offset, u16 size) {
    data[offset] = 0x78;
    data[offset + 1] = 0x01;
    data[offset + 2] = 0x01;
    Write16(data, offset + 3, size);
    Write16(data, offset + 5, static_cast<u16>(~size));
    data[offset + 7 + size] = static_cast<u8>(size >> 8);
    data[offset + 8 + size] = static_cast<u8>(size);
    data[offset + 10 + size] = 1;
}

std::vector<u8> MakeTextures() {
    std::vector<u8> data(TextureDataOffset + TextureBytes + ZlibOverhead);
    WriteHeader(data, "NFTR", TextureKinds.size());
    std::size_t entry = 0x64;
    for (const auto& kind : TextureKinds) {
        for (u16 index = 0; index < kind.count; ++index, entry += 0x1c) {
            Write32(data, entry, TextureDataOffset);
            Write32(data, entry + 4, TextureBytes);
            Write32(data, entry + 8, TextureBytes + ZlibOverhead);
            Write32(data, entry + 0x10, 0x200);
            Write16(data, entry + 0x14, 8);
            Write16(data, entry + 0x16, 8);
            data[entry + 0x18] = 1;
            data[entry + 0x19] = kind.format;
        }
    }
    WriteBlankZlib(data, TextureDataOffset, TextureBytes);
    return data;
}

std::vector<u8> MakeShapes() {
    std::vector<u8> data(ShapeDataOffset + ShapeBytes + ZlibOverhead);
    WriteHeader(data, "NFSR", 12);
    for (std::size_t index = 0; index < ShapeCount; ++index) {
        const auto entry = 0x6c + index * 0x58;
        Write32(data, entry, ShapeDataOffset);
        Write32(data, entry + 4, ShapeBytes);
        Write32(data, entry + 8, ShapeBytes + ZlibOverhead);
        Write32(data, entry + 36, 8); // One position vertex (four half floats).
        Write32(data, entry + 56, 0x10);
        Write32(data, entry + 60, 6); // Three u16 indices, all selecting vertex zero.
    }
    WriteBlankZlib(data, ShapeDataOffset, ShapeBytes);
    return data;
}

} // Anonymous namespace

VirtualDir MiiModel() {
    auto out = std::make_shared<VectorVfsDirectory>(std::vector<VirtualFile>{},
                                                   std::vector<VirtualDir>{}, "data");
    const auto textures = MakeTextures();
    for (const char* name : {"NXTextureLowLinear.dat", "NXTextureLowSRGB.dat",
                             "NXTextureMidLinear.dat", "NXTextureMidSRGB.dat"}) {
        out->AddFile(std::make_shared<VectorVfsFile>(textures, name));
    }
    const auto shapes = MakeShapes();
    for (const char* name : {"ShapeHigh.dat", "ShapeMid.dat"}) {
        out->AddFile(std::make_shared<VectorVfsFile>(shapes, name));
    }
    return out;
}

} // namespace FileSys::SystemArchive
