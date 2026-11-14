// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "video_core/textures/decoders.h"

namespace {

using namespace Tegra::Texture;

struct Layout {
    u32 width;
    u32 height;
    u32 depth;
    u32 bpp;
    u32 block_height;
    u32 block_depth;
    u32 alignment;
};

// The block linear address of byte `x` of row `y` in slice `z`, written out bit by bit (GOBs of
// 64 bytes by 8 rows, stacked in blocks of 2^block_height GOBs high and 2^block_depth deep).
u64 BlockLinearOffset(const Layout& layout, u32 x, u32 y, u32 z, bool native) {
    const u32 stride = Common::AlignUpLog2(layout.width, layout.alignment) * layout.bpp;
    const u64 gobs_in_x = Common::DivCeil(stride, GOB_SIZE_X);
    const u64 block_size = gobs_in_x << (GOB_SIZE_SHIFT + layout.block_height + layout.block_depth);
    const u64 blocks_in_y = Common::DivCeil(layout.height, GOB_SIZE_Y << layout.block_height);
    const u64 slice_size = blocks_in_y * block_size;
    const u32 depth_mask = (1U << layout.block_depth) - 1;
    const u32 height_mask = (1U << layout.block_height) - 1;
    const u64 offset_z = (z >> layout.block_depth) * slice_size +
                         (u64{z & depth_mask} << (GOB_SIZE_SHIFT + layout.block_height));
    const u32 gob_y = y / GOB_SIZE_Y;
    const u64 offset_y = (gob_y >> layout.block_height) * block_size +
                         (u64{gob_y & height_mask} << GOB_SIZE_SHIFT);
    const u64 offset_x = u64{x / GOB_SIZE_X}
                         << (GOB_SIZE_SHIFT + layout.block_height + layout.block_depth);
    // Within a GOB: x bits 0-3 stay, x bit 4 goes to bit 5 and x bit 5 to bit 8; y bits 0, 1 and
    // 2 go to bits 4, 6 and 7.
    u32 within = (x & 0xf) | (((x >> 4) & 1) << 5) | (((x >> 5) & 1) << 8) | ((y & 1) << 4) |
                 (((y >> 1) & 1) << 6) | (((y >> 2) & 1) << 7);
    if (native) {
        const u32 swap = ((within >> 5) ^ (within >> 6)) & 1;
        within ^= (swap << 5) | (swap << 6);
    }
    return offset_z + offset_y + offset_x + within;
}

size_t GuestBytes(const Layout& layout) {
    // Whole blocks: the aligned stride in GOBs, the height and depth rounded up to blocks.
    const u32 stride = Common::AlignUpLog2(layout.width, layout.alignment) * layout.bpp;
    const size_t gobs_in_x = Common::DivCeil(stride, GOB_SIZE_X);
    const size_t blocks_in_y = Common::DivCeil(layout.height, GOB_SIZE_Y << layout.block_height);
    const size_t blocks_in_z = Common::DivCeil(layout.depth, 1U << layout.block_depth);
    return gobs_in_x * blocks_in_y * blocks_in_z *
           (size_t{GOB_SIZE} << (layout.block_height + layout.block_depth));
}

size_t LinearBytes(const Layout& layout) {
    return size_t{layout.width} * layout.height * layout.depth * layout.bpp;
}

std::vector<u8> RandomBytes(size_t size, u32 seed) {
    std::mt19937 engine{seed};
    std::vector<u8> bytes(size);
    for (u8& byte : bytes) {
        byte = static_cast<u8>(engine());
    }
    return bytes;
}

void CheckLayout(const Layout& layout, bool native) {
    INFO("width " << layout.width << " height " << layout.height << " depth " << layout.depth
                  << " bpp " << layout.bpp << " block height " << layout.block_height
                  << " block depth " << layout.block_depth << " alignment " << layout.alignment
                  << " native " << native);
    const size_t guest_bytes = GuestBytes(layout);
    const size_t linear_bytes = LinearBytes(layout);
    const u32 row_bytes = layout.width * layout.bpp;

    // Unswizzle: every linear byte comes from its block linear address.
    const std::vector<u8> guest = RandomBytes(guest_bytes, layout.width * 31 + layout.height);
    std::vector<u8> linear(linear_bytes, 0xcd);
    UnswizzleTexture(linear, guest, layout.bpp, layout.width, layout.height, layout.depth,
                     layout.block_height, layout.block_depth, layout.alignment, native);
    bool unswizzle_ok = true;
    for (u32 z = 0; z < layout.depth && unswizzle_ok; ++z) {
        for (u32 y = 0; y < layout.height && unswizzle_ok; ++y) {
            for (u32 x = 0; x < row_bytes; ++x) {
                const size_t linear_offset = (size_t{z} * layout.height + y) * row_bytes + x;
                if (linear[linear_offset] != guest[BlockLinearOffset(layout, x, y, z, native)]) {
                    unswizzle_ok = false;
                    break;
                }
            }
        }
    }
    CHECK(unswizzle_ok);

    // Swizzle: every block linear address of an image byte holds that byte, and the padding
    // between them keeps what was there.
    const std::vector<u8> source = RandomBytes(linear_bytes, layout.height * 17 + layout.bpp);
    std::vector<u8> swizzled(guest_bytes, 0xab);
    std::vector<u8> expected(guest_bytes, 0xab);
    for (u32 z = 0; z < layout.depth; ++z) {
        for (u32 y = 0; y < layout.height; ++y) {
            for (u32 x = 0; x < row_bytes; ++x) {
                expected[BlockLinearOffset(layout, x, y, z, native)] =
                    source[(size_t{z} * layout.height + y) * row_bytes + x];
            }
        }
    }
    SwizzleTexture(swizzled, source, layout.bpp, layout.width, layout.height, layout.depth,
                   layout.block_height, layout.block_depth, layout.alignment, native);
    CHECK(swizzled == expected);
}

struct Subrect {
    u32 origin_x;
    u32 origin_y;
    u32 extent_x;
    u32 lines;
    u32 pitch;
};

// SwizzleSubrect: the lines start at origin_y in each slice and continue into the next one.
void CheckSubrect(const Layout& layout, const Subrect& rect, bool native) {
    INFO("width " << layout.width << " height " << layout.height << " depth " << layout.depth
                  << " bpp " << layout.bpp << " block height " << layout.block_height
                  << " block depth " << layout.block_depth << " origin " << rect.origin_x << ","
                  << rect.origin_y << " extent " << rect.extent_x << " lines " << rect.lines
                  << " pitch " << rect.pitch << " native " << native);
    const size_t guest_bytes = GuestBytes(layout);
    const std::vector<u8> source =
        RandomBytes(size_t{rect.pitch} * layout.height * layout.depth, rect.extent_x * 7 + 3);
    std::vector<u8> swizzled(guest_bytes, 0x5a);
    std::vector<u8> expected(guest_bytes, 0x5a);
    u32 remaining = rect.lines;
    const u32 lines_per_slice = (std::min)(rect.lines, layout.height - rect.origin_y);
    for (u32 z = 0; z < layout.depth && remaining != 0; ++z) {
        const u32 lines = (std::min)(remaining, lines_per_slice);
        for (u32 line = 0; line < lines; ++line) {
            for (u32 x = 0; x < rect.extent_x * layout.bpp; ++x) {
                expected[BlockLinearOffset(layout, rect.origin_x * layout.bpp + x,
                                           rect.origin_y + line, z, native)] =
                    source[(size_t{z} * layout.height + line) * rect.pitch + x];
            }
        }
        remaining -= lines;
    }
    SwizzleSubrect(swizzled, source, layout.bpp, layout.width, layout.height, layout.depth,
                   rect.origin_x, rect.origin_y, rect.extent_x, rect.lines, layout.block_height,
                   layout.block_depth, rect.pitch, native);
    CHECK(swizzled == expected);
}

} // Anonymous namespace

TEST_CASE("Block linear subrect swizzle matches the GOB layout", "[block_linear_swizzle]") {
    // Origins and extents that start and end inside 16-byte units and GOBs.
    for (const u32 bpp : {1U, 2U, 4U, 8U, 16U}) {
        for (const u32 block_height : {0U, 2U, 4U}) {
            for (const bool native : {false, true}) {
                const Layout layout{100, 40, 1, bpp, block_height, 0, 0};
                for (const Subrect rect : {Subrect{0, 0, 100, 40, 100 * bpp},
                                           Subrect{3, 5, 17, 9, 17 * bpp},
                                           Subrect{13, 7, 61, 33, 64 * bpp},
                                           Subrect{99, 39, 1, 1, bpp},
                                           Subrect{31, 0, 2, 40, 5 * bpp}}) {
                    CheckSubrect(layout, rect, native);
                }
            }
        }
    }
    // Lines that continue into the next slices.
    for (const bool native : {false, true}) {
        CheckSubrect({24, 9, 3, 4, 1, 1, 0}, {5, 2, 11, 20, 11 * 4}, native);
        CheckSubrect({40, 16, 2, 16, 2, 0, 0}, {1, 0, 37, 32, 40 * 16}, native);
    }
}

TEST_CASE("Block linear swizzle matches the GOB layout", "[block_linear_swizzle]") {
    // Widths whose rows are and aren't a whole number of 16-byte units (UnswizzleTexture widens
    // to the largest power of two unit that divides a row, up to 16 bytes), partial GOBs at the
    // right and bottom, and tall and deep blocks.
    for (const u32 bpp : {1U, 2U, 4U, 8U, 16U}) {
        for (const u32 width : {1U, 3U, 4U, 16U, 17U, 33U, 64U, 100U}) {
            for (const u32 height : {1U, 7U, 8U, 9U, 40U}) {
                for (const u32 block_height : {0U, 1U, 3U, 5U}) {
                    for (const bool native : {false, true}) {
                        CheckLayout({width, height, 1, bpp, block_height, 0, 0}, native);
                    }
                }
            }
        }
    }
    for (const u32 depth : {2U, 3U, 5U}) {
        for (const u32 block_depth : {0U, 1U, 2U}) {
            for (const bool native : {false, true}) {
                CheckLayout({48, 20, depth, 16, 2, block_depth, 0}, native);
                CheckLayout({13, 9, depth, 4, 1, block_depth, 0}, native);
            }
        }
    }
    // A row stride aligned past the width.
    for (const u32 alignment : {1U, 4U, 6U}) {
        for (const bool native : {false, true}) {
            CheckLayout({37, 19, 1, 4, 2, 0, alignment}, native);
            CheckLayout({64, 24, 1, 16, 3, 0, alignment}, native);
        }
    }
}

TEST_CASE("Block linear subrect swizzle microbenchmark", "[.block_linear_swizzle_benchmark]") {
    // A DMA copy of a 1024x1024 RGBA8 region from pitch linear into block linear.
    const Layout layout{1024, 1024, 1, 4, 4, 0, 0};
    const std::vector<u8> source = RandomBytes(LinearBytes(layout), 2);
    std::vector<u8> guest(GuestBytes(layout));
    for (const bool native : {false, true}) {
        constexpr int iterations = 200;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            SwizzleSubrect(guest, source, layout.bpp, layout.width, layout.height, 1, 0, 0,
                           layout.width, layout.height, layout.block_height, 0,
                           layout.width * layout.bpp, native);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("1024x1024 RGBA8 subrect swizzle%s: %.3f ms, %.2f GB/s\n",
                    native ? " (native tiling)" : "", seconds * 1e3 / iterations,
                    LinearBytes(layout) * iterations / seconds / 1e9);
    }
}

TEST_CASE("Block linear unswizzle microbenchmark", "[.block_linear_swizzle_benchmark]") {
    const Layout layout{1024, 1024, 1, 4, 4, 0, 0};
    const std::vector<u8> guest = RandomBytes(GuestBytes(layout), 1);
    std::vector<u8> linear(LinearBytes(layout));
    for (const bool native : {false, true}) {
        constexpr int iterations = 200;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            UnswizzleTexture(linear, guest, layout.bpp, layout.width, layout.height, 1,
                             layout.block_height, 0, 0, native);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("1024x1024 RGBA8 unswizzle%s: %.3f ms, %.2f GB/s\n",
                    native ? " (native tiling)" : "", seconds * 1e3 / iterations,
                    LinearBytes(layout) * iterations / seconds / 1e9);
    }
}
