// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "common/swap.h"

namespace Service::LDN {

// The LAN protocol of ldn_mitm (https://github.com/spacemeowx2/ldn_mitm): scans are UDP broadcasts
// and replies on LanPort, stations hold a TCP connection to the host on the same port.
constexpr u16 LanPort = 11452;
constexpr u32 FrameMagic = 0x11451400;
constexpr std::size_t FrameSizeMax = 2048;

enum class FrameType : u8 {
    Scan,
    ScanResponse,
    Connect,
    SyncNetwork,
};

struct FrameHeader {
    u32_le magic;
    FrameType type;
    u8 is_compressed;
    u16_le body_size;
    u16_le raw_size;
    std::array<u8, 2> reserved;
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader is an invalid size");

struct Frame {
    FrameType type;
    std::vector<u8> data;
};

enum class FrameResult {
    Incomplete,
    Malformed,
    Complete,
};

/// Each zero byte is followed by the count of zeros after it, up to 0xFF.
/// nullopt when the result would be larger than size_max.
std::optional<std::vector<u8>> CompressFrameBody(std::span<const u8> input, std::size_t size_max);
std::optional<std::vector<u8>> DecompressFrameBody(std::span<const u8> input, std::size_t size_max);

std::vector<u8> EncodeFrame(FrameType type, std::span<const u8> data);

/// Moves the first whole frame of buffer into frame. A malformed frame clears buffer.
FrameResult TakeFrame(std::vector<u8>& buffer, Frame& frame);

} // namespace Service::LDN
