// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/util.h"

namespace {

using VideoCommon::CopiesWriteWholeImage;
using VideoCommon::Extent3D;
using VideoCommon::ImageBase;
using VideoCommon::ImageCopy;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::MarkSubresourcesWrittenWhole;
using VideoCore::Surface::PixelFormat;

ImageInfo Info(PixelFormat format, u32 width, u32 height, s32 levels = 1, s32 layers = 1) {
    ImageInfo info;
    info.format = format;
    info.type = ImageType::e2D;
    info.size = {width, height, 1};
    info.resources.levels = levels;
    info.resources.layers = layers;
    return info;
}

ImageCopy LevelCopy(s32 level, u32 width, u32 height, s32 layers = 1) {
    ImageCopy copy{};
    copy.src_subresource = {.base_level = level, .base_layer = 0, .num_layers = layers};
    copy.dst_subresource = copy.src_subresource;
    copy.extent = {width, height, 1};
    return copy;
}

constexpr PixelFormat RGBA = PixelFormat::A8B8G8R8_UNORM;

} // namespace

TEST_CASE("CopiesWriteWholeImage accepts a 1x copy of every level and layer",
          "[video_core][texture_cache]") {
    const ImageBase dst(Info(RGBA, 5, 4, 2, 3), 0, 0);
    const ImageBase src(Info(RGBA, 5, 4, 2, 3), 0, 0);
    const std::vector<ImageCopy> copies{LevelCopy(0, 5, 4, 3), LevelCopy(1, 2, 2, 3)};
    REQUIRE(CopiesWriteWholeImage(dst, src, copies));
}

TEST_CASE("CopiesWriteWholeImage rejects compressed formats", "[video_core][texture_cache]") {
    const ImageBase dst(Info(PixelFormat::BC1_RGBA_UNORM, 64, 30), 0, 0);
    const ImageBase src(Info(PixelFormat::BC1_RGBA_UNORM, 64, 30), 0, 0);
    // AddImageAlias counts the extent in blocks; the backends copy only 16x8 texels of it
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, std::vector{LevelCopy(0, 16, 8)}));
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, std::vector{LevelCopy(0, 64, 30)}));
}

TEST_CASE("CopiesWriteWholeImage rejects copies that leave texels unwritten",
          "[video_core][texture_cache]") {
    const ImageBase dst(Info(RGBA, 5, 4, 2, 3), 0, 0);
    const ImageBase src(Info(RGBA, 5, 4, 2, 3), 0, 0);
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, std::vector{LevelCopy(0, 5, 4, 3)}));
    REQUIRE_FALSE(CopiesWriteWholeImage(
        dst, src, std::vector{LevelCopy(0, 5, 4, 3), LevelCopy(1, 1, 2, 3)}));
    REQUIRE_FALSE(CopiesWriteWholeImage(
        dst, src, std::vector{LevelCopy(0, 5, 4, 2), LevelCopy(1, 2, 2, 2)}));
    std::vector copies{LevelCopy(0, 5, 4, 3), LevelCopy(1, 2, 2, 3)};
    copies[0].dst_offset.x = 1;
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, copies));
    copies[0].dst_offset.x = 0;
    copies[1].dst_subresource.base_layer = 1;
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, copies));
}

TEST_CASE("CopiesWriteWholeImage rejects other formats and types", "[video_core][texture_cache]") {
    const ImageBase dst(Info(RGBA, 8, 8), 0, 0);
    const std::vector copies{LevelCopy(0, 8, 8)};
    REQUIRE_FALSE(
        CopiesWriteWholeImage(dst, ImageBase(Info(PixelFormat::R32_FLOAT, 8, 8), 0, 0), copies));
    ImageInfo info_3d = Info(RGBA, 8, 8);
    info_3d.type = ImageType::e3D;
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, ImageBase(info_3d, 0, 0), copies));
    ImageInfo opaque = Info(RGBA, 8, 8);
    opaque.astc_opaque = true;
    REQUIRE_FALSE(CopiesWriteWholeImage(dst, ImageBase(opaque, 0, 0), copies));
}

TEST_CASE("CopiesWriteWholeImage rejects rescaled and partly backed images",
          "[video_core][texture_cache]") {
    const std::vector copies{LevelCopy(0, 8, 8)};
    for (const ImageFlagBits flag :
         {ImageFlagBits::Rescaled, ImageFlagBits::TrimmedLevels, ImageFlagBits::Unbacked,
          ImageFlagBits::UnwrittenSparse, ImageFlagBits::SparseLayers,
          ImageFlagBits::SparseTiles}) {
        ImageBase dst(Info(RGBA, 8, 8), 0, 0);
        ImageBase src(Info(RGBA, 8, 8), 0, 0);
        REQUIRE(CopiesWriteWholeImage(dst, src, copies));
        src.flags |= flag;
        REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, copies));
        dst.flags |= flag;
        REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, copies));
        src.flags &= ~flag;
        REQUIRE_FALSE(CopiesWriteWholeImage(dst, src, copies));
    }
}

TEST_CASE("MarkSubresourcesWrittenWhole adds up layers copied from several images",
          "[video_core][texture_cache]") {
    const ImageBase dst(Info(RGBA, 5, 4, 1, 3), 0, 0);
    const ImageBase src(Info(RGBA, 5, 4), 0, 0);
    const auto layer_copy = [](s32 layer) {
        ImageCopy copy = LevelCopy(0, 5, 4);
        copy.dst_subresource.base_layer = layer;
        return copy;
    };
    std::vector<u8> written(3);
    MarkSubresourcesWrittenWhole(dst, src, std::vector{layer_copy(0)}, written);
    MarkSubresourcesWrittenWhole(dst, src, std::vector{layer_copy(2)}, written);
    REQUIRE(written == std::vector<u8>{1, 0, 1});
    MarkSubresourcesWrittenWhole(dst, ImageBase(Info(PixelFormat::R32_FLOAT, 5, 4), 0, 0),
                                 std::vector{layer_copy(1)}, written);
    REQUIRE(written == std::vector<u8>{1, 0, 1});
    ImageCopy partial = layer_copy(1);
    partial.extent.width = 4;
    MarkSubresourcesWrittenWhole(dst, src, std::vector{partial}, written);
    REQUIRE(written == std::vector<u8>{1, 0, 1});
    MarkSubresourcesWrittenWhole(dst, src, std::vector{layer_copy(1)}, written);
    REQUIRE(written == std::vector<u8>{1, 1, 1});
    REQUIRE(CopiesWriteWholeImage(dst, src,
                                  std::vector{layer_copy(0), layer_copy(1), layer_copy(2)}));
}
