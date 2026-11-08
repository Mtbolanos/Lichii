// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "core/file_sys/system_archive/data/font_korean.h"
#include "core/file_sys/system_archive/data/font_nintendo_extended.h"
#include "core/file_sys/system_archive/data/font_standard.h"
#include "core/hle/service/ns/font_merge.h"

namespace {

using FileSys::SystemArchive::SharedFontData::FONT_KOREAN;
using FileSys::SystemArchive::SharedFontData::FONT_NINTENDO_EXTENDED;
using FileSys::SystemArchive::SharedFontData::FONT_STANDARD;

// The controller button icons the standard font of this repository has and FONT_KOREAN lacks.
constexpr std::array<std::uint32_t, 24> BUTTON_GLYPHS{
    0xE0A0, 0xE0A1, 0xE0A2, 0xE0A3, 0xE0A4, 0xE0A5, 0xE0A6, 0xE0A7, 0xE0A8, 0xE0A9, 0xE0B3, 0xE0B4,
    0xE0E0, 0xE0E1, 0xE0E2, 0xE0E3, 0xE0E4, 0xE0E5, 0xE0E6, 0xE0E7, 0xE0E8, 0xE0E9, 0xE0EF, 0xE0F0,
};

std::uint16_t Be16(std::span<const std::uint8_t> data, std::size_t offset) {
    return static_cast<std::uint16_t>((data[offset] << 8) | data[offset + 1]);
}

std::uint32_t Be32(std::span<const std::uint8_t> data, std::size_t offset) {
    return (std::uint32_t{Be16(data, offset)} << 16) | Be16(data, offset + 2);
}

/// The tables of a font by tag, for the checks below.
std::map<std::string, std::span<const std::uint8_t>> Tables(std::span<const std::uint8_t> font) {
    std::map<std::string, std::span<const std::uint8_t>> tables;
    const std::size_t count = Be16(font, 4);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t entry = 12 + i * 16;
        const std::string tag(reinterpret_cast<const char*>(font.data() + entry), 4);
        tables[tag] = font.subspan(Be32(font, entry + 8), Be32(font, entry + 12));
    }
    return tables;
}

/// Where the table directory of a font gives the length of a table.
std::size_t TableLengthAt(std::span<const std::uint8_t> font, const char* tag) {
    const std::size_t count = Be16(font, 4);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t entry = 12 + i * 16;
        if (std::equal(tag, tag + 4, font.begin() + entry)) {
            return entry + 12;
        }
    }
    return 0;
}

void Set16(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value) {
    data[offset] = static_cast<std::uint8_t>(value >> 8);
    data[offset + 1] = static_cast<std::uint8_t>(value);
}

std::uint16_t NumGlyphs(std::span<const std::uint8_t> font) {
    return Be16(Tables(font).at("maxp"), 4);
}

} // Anonymous namespace

TEST_CASE("MergeGlyphs adds the missing glyphs", "[core][font]") {
    const auto merged = Service::NS::MergeGlyphs(FONT_KOREAN, FONT_STANDARD, BUTTON_GLYPHS);
    REQUIRE_FALSE(merged.empty());

    // A TrueType font whose table checksums and whole-file checksum are valid.
    REQUIRE(Be32(merged, 0) == 0x00010000);
    std::uint32_t sum = 0;
    for (std::size_t i = 0; i + 4 <= merged.size(); i += 4) {
        sum += Be32(merged, i);
    }
    CHECK(sum == 0xb1b0afba);

    // One glyph per added icon, and a long glyph index (loca) now that the glyph table grew.
    CHECK(NumGlyphs(merged) == NumGlyphs(FONT_KOREAN) + BUTTON_GLYPHS.size());
    CHECK(Be16(Tables(merged).at("head"), 50) == 1);

    // The character map now has every icon, so there is nothing left to add.
    CHECK(Service::NS::MergeGlyphs(merged, FONT_STANDARD, BUTTON_GLYPHS).empty());
}

TEST_CASE("MergeGlyphs keeps every original glyph", "[core][font]") {
    const auto merged = Service::NS::MergeGlyphs(FONT_KOREAN, FONT_STANDARD, BUTTON_GLYPHS);
    REQUIRE_FALSE(merged.empty());
    const auto original = Tables(FONT_KOREAN);
    const auto result = Tables(merged);

    // Every outline of the original font is at the same place in the merged glyph table.
    const bool long_loca = Be16(original.at("head"), 50) != 0;
    const auto offset = [&](std::span<const std::uint8_t> loca, bool is_long, std::size_t glyph) {
        return is_long ? Be32(loca, glyph * 4) : std::size_t{Be16(loca, glyph * 2)} * 2;
    };
    const std::size_t glyphs = NumGlyphs(FONT_KOREAN);
    const auto old_glyf = original.at("glyf");
    const auto new_glyf = result.at("glyf");
    for (std::size_t glyph = 0; glyph < glyphs; ++glyph) {
        const auto old_begin = offset(original.at("loca"), long_loca, glyph);
        const auto old_end = offset(original.at("loca"), long_loca, glyph + 1);
        const auto new_begin = offset(result.at("loca"), true, glyph);
        const auto new_end = offset(result.at("loca"), true, glyph + 1);
        REQUIRE(new_end - new_begin == old_end - old_begin);
        REQUIRE(std::equal(old_glyf.begin() + old_begin, old_glyf.begin() + old_end,
                           new_glyf.begin() + new_begin));
    }
}

TEST_CASE("MergeGlyphs refuses what it cannot merge", "[core][font]") {
    // A font that already has the icons: nothing to add.
    CHECK(Service::NS::MergeGlyphs(FONT_STANDARD, FONT_STANDARD, BUTTON_GLYPHS).empty());

    // Different units per em (2048 against 1024): the glyphs would need scaling.
    CHECK(Service::NS::MergeGlyphs(FONT_KOREAN, FONT_NINTENDO_EXTENDED, BUTTON_GLYPHS).empty());

    // No codepoints, a codepoint the donor lacks, and data that is not a font.
    CHECK(Service::NS::MergeGlyphs(FONT_KOREAN, FONT_STANDARD, {}).empty());
    constexpr std::array<std::uint32_t, 1> missing{0xE0FF};
    CHECK(Service::NS::MergeGlyphs(FONT_KOREAN, FONT_STANDARD, missing).empty());
    const std::vector<std::uint8_t> garbage(64, 0x5a);
    CHECK(Service::NS::MergeGlyphs(garbage, FONT_STANDARD, BUTTON_GLYPHS).empty());
    CHECK(Service::NS::MergeGlyphs(FONT_KOREAN, garbage, BUTTON_GLYPHS).empty());
    CHECK(Service::NS::MergeGlyphs({}, {}, BUTTON_GLYPHS).empty());
}

TEST_CASE("MergeGlyphs refuses truncated head and cmap tables", "[core][font]") {
    // A donor head table too short to hold its units per em.
    std::vector<std::uint8_t> short_head(FONT_STANDARD.begin(), FONT_STANDARD.end());
    const std::size_t head_length = TableLengthAt(short_head, "head");
    REQUIRE(head_length != 0);
    Set16(short_head, head_length, 0);
    Set16(short_head, head_length + 2, 16);
    CHECK(Service::NS::MergeGlyphs(FONT_KOREAN, short_head, BUTTON_GLYPHS).empty());

    // A format 4 character map whose length leaves no room for its segment count.
    std::vector<std::uint8_t> short_cmap(FONT_STANDARD.begin(), FONT_STANDARD.end());
    const auto cmap = Tables(short_cmap).at("cmap");
    const std::size_t cmap_offset = static_cast<std::size_t>(cmap.data() - short_cmap.data());
    bool truncated = false;
    for (std::size_t i = 0; i < Be16(cmap, 2); ++i) {
        const std::size_t sub = Be32(cmap, 4 + i * 8 + 4);
        if (Be16(cmap, sub) == 4) {
            Set16(short_cmap, cmap_offset + sub + 2, 6);
            truncated = true;
        }
    }
    REQUIRE(truncated);
    CHECK(Service::NS::MergeGlyphs(FONT_KOREAN, short_cmap, BUTTON_GLYPHS).empty());
}
