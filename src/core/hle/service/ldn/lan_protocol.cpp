// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "core/hle/service/ldn/lan_protocol.h"

namespace Service::LDN {

std::optional<std::vector<u8>> CompressFrameBody(std::span<const u8> input, std::size_t size_max) {
    std::vector<u8> output;
    for (std::size_t i = 0; i < input.size();) {
        const u8 c = input[i++];
        output.push_back(c);
        if (c == 0) {
            u8 count = 0;
            while (i < input.size() && input[i] == 0 && count < 0xFF) {
                ++count;
                ++i;
            }
            output.push_back(count);
        }
        if (output.size() > size_max) {
            return std::nullopt;
        }
    }
    return output;
}

std::optional<std::vector<u8>> DecompressFrameBody(std::span<const u8> input,
                                                   std::size_t size_max) {
    std::vector<u8> output;
    for (std::size_t i = 0; i < input.size();) {
        const u8 c = input[i++];
        std::size_t count = 1;
        if (c == 0) {
            if (i == input.size()) {
                return std::nullopt;
            }
            count += input[i++];
        }
        if (output.size() + count > size_max) {
            return std::nullopt;
        }
        output.insert(output.end(), count, c);
    }
    return output;
}

std::vector<u8> EncodeFrame(FrameType type, std::span<const u8> data) {
    FrameHeader header{
        .magic = FrameMagic,
        .type = type,
        .is_compressed = 0,
        .body_size = static_cast<u16>(data.size()),
        .raw_size = 0,
        .reserved = {},
    };
    auto compressed = data.empty() ? std::nullopt : CompressFrameBody(data, data.size());
    if (compressed) {
        header.is_compressed = 1;
        header.raw_size = static_cast<u16>(data.size());
        header.body_size = static_cast<u16>(compressed->size());
        data = *compressed;
    }

    std::vector<u8> frame(sizeof(header) + data.size());
    std::memcpy(frame.data(), &header, sizeof(header));
    if (!data.empty()) {
        std::memcpy(frame.data() + sizeof(header), data.data(), data.size());
    }
    return frame;
}

FrameResult TakeFrame(std::vector<u8>& buffer, Frame& frame) {
    if (buffer.size() < sizeof(FrameHeader)) {
        return FrameResult::Incomplete;
    }
    FrameHeader header;
    std::memcpy(&header, buffer.data(), sizeof(header));
    const std::size_t total = sizeof(header) + header.body_size;
    if (header.magic != FrameMagic || total > FrameSizeMax) {
        buffer.clear();
        return FrameResult::Malformed;
    }
    if (buffer.size() < total) {
        return FrameResult::Incomplete;
    }

    frame.type = header.type;
    frame.data.assign(buffer.begin() + sizeof(header), buffer.begin() + total);
    buffer.erase(buffer.begin(), buffer.begin() + total);
    if (!header.is_compressed) {
        return FrameResult::Complete;
    }
    auto data = DecompressFrameBody(frame.data, FrameSizeMax);
    if (!data || data->size() != header.raw_size) {
        buffer.clear();
        return FrameResult::Malformed;
    }
    frame.data = std::move(*data);
    return FrameResult::Complete;
}

} // namespace Service::LDN
