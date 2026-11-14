// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/sockets/numeric_name_info.h"

#include <cstring>
#include <string>
#include <utility>

#include "core/hle/ipc.h"

namespace Service::Sockets {

NumericNameInfoResult ResolveNumericNameInfo(const Network::SockAddrIn& address, s32 flags,
                                             std::size_t host_capacity,
                                             std::size_t service_capacity) {
    // Guest NI_* values, not WinSock/Linux constants. No DNS or service database lookup.
    constexpr u32 NumericHost = 2;
    constexpr u32 NumericService = 8;
    constexpr u32 AllowedFlags = 1 | NumericHost | 4 | NumericService | 16;
    const auto bits = static_cast<u32>(flags);
    if ((bits & ~AllowedFlags) != 0) {
        return {GetAddrInfoError::BADFLAGS, {}, {}};
    }
    if (address.family != Network::Domain::INET) {
        return {GetAddrInfoError::FAMILY, {}, {}};
    }
    if ((host_capacity == 0 && service_capacity == 0) ||
        (host_capacity != 0 && (bits & NumericHost) == 0) ||
        (service_capacity != 0 && (bits & NumericService) == 0)) {
        return {GetAddrInfoError::NONAME, {}, {}};
    }

    std::string host;
    if (host_capacity != 0) {
        host = std::to_string(address.ip[0]) + "." + std::to_string(address.ip[1]) + "." +
               std::to_string(address.ip[2]) + "." + std::to_string(address.ip[3]);
    }
    const std::string service = service_capacity != 0 ? std::to_string(address.portno) : "";
    if ((host_capacity != 0 && host.size() + 1 > host_capacity) ||
        (service_capacity != 0 && service.size() + 1 > service_capacity)) {
        return {GetAddrInfoError::OVERFLOW_, {}, {}};
    }
    NumericNameInfoResult result{GetAddrInfoError::SUCCESS, {}, {}};
    if (host_capacity != 0) {
        result.host.assign(host.begin(), host.end());
        result.host.push_back(0);
    }
    if (service_capacity != 0) {
        result.service.assign(service.begin(), service.end());
        result.service.push_back(0);
    }
    return result;
}

NumericNameInfoResult ResolveNumericNameInfoRequest(std::span<const u8> parameters,
                                                    std::span<const u8> sockaddr,
                                                    std::span<const u8> options,
                                                    std::size_t host_capacity,
                                                    std::size_t service_capacity) {
    if (parameters.size() != 24 || sockaddr.size() < 16 || sockaddr.size() > 256 ||
        !options.empty()) {
        return {GetAddrInfoError::FAIL, {}, {}};
    }
    const auto read_u32 = [&parameters](std::size_t offset) {
        return u32{parameters[offset]} | (u32{parameters[offset + 1]} << 8) |
               (u32{parameters[offset + 2]} << 16) | (u32{parameters[offset + 3]} << 24);
    };
    // +4 is not options_count, and its exact legacy meaning is not required here.
    if (read_u32(8) != 0 || (sockaddr[0] != 0 && sockaddr[0] != 16)) {
        return {GetAddrInfoError::FAIL, {}, {}};
    }
    if (sockaddr[1] != static_cast<u8>(Domain::INET)) {
        return {GetAddrInfoError::FAMILY, {}, {}};
    }
    const Network::SockAddrIn address{
        .family = Network::Domain::INET,
        .ip = {sockaddr[4], sockaddr[5], sockaddr[6], sockaddr[7]},
        .portno = static_cast<u16>((u16{sockaddr[2]} << 8) | sockaddr[3]),
    };
    return ResolveNumericNameInfo(address, static_cast<s32>(read_u32(0)), host_capacity,
                                  service_capacity);
}

NumericNameInfoResponse MakeNumericNameInfoResponse(GetAddrInfoError error) {
    // Pure numeric validation does not produce host-database or BSD syscall errors.
    return {error, 0, Errno::SUCCESS};
}

std::span<const u8> GetNumericNameInfoParameters(std::span<const u32> command_buffer,
                                                 std::size_t data_payload_word_offset) {
    if (command_buffer.size() < 2 || data_payload_word_offset > command_buffer.size() - 2) {
        return {};
    }
    IPC::CommandHeader header{};
    std::memcpy(&header, command_buffer.data(), sizeof(header));
    if (header.IsTipc() || header.IsCloseCommand()) {
        return {};
    }
    std::size_t data_start = sizeof(header) / sizeof(u32);
    if (header.enable_handle_descriptor) {
        if (data_start >= command_buffer.size()) {
            return {};
        }
        IPC::HandleDescriptorHeader handles{};
        std::memcpy(&handles, command_buffer.data() + data_start, sizeof(handles));
        data_start += sizeof(handles) / sizeof(u32) + (handles.send_current_pid ? 2 : 0) +
                      handles.num_handles_to_copy + handles.num_handles_to_move;
    }
    data_start += header.num_buf_x_descriptors * (sizeof(IPC::BufferDescriptorX) / sizeof(u32)) +
                  (header.num_buf_a_descriptors + header.num_buf_b_descriptors +
                   header.num_buf_w_descriptors) *
                      (sizeof(IPC::BufferDescriptorABW) / sizeof(u32));
    const auto raw_start = data_payload_word_offset + 2; // Skip command ID.
    const auto raw_end = data_start + header.data_size;
    if (data_start > command_buffer.size() || raw_end > command_buffer.size() ||
        raw_start < data_start + 4 || raw_start > raw_end || raw_end - raw_start < 6) {
        return {};
    }
    return {reinterpret_cast<const u8*>(command_buffer.data() + raw_start), 24};
}

} // namespace Service::Sockets
