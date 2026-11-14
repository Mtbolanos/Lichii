// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "common/socket_types.h"
#include "core/hle/service/sockets/sockets.h"

namespace Service::Sockets {

struct NumericNameInfoResult {
    GetAddrInfoError error;
    std::vector<u8> host;
    std::vector<u8> service;
};

// Pure numeric formatting of a parsed address, independent of any IPC wire layout.
// Nonzero capacities request an output, including its trailing NUL. Errors return
// no partial outputs. Name lookup and service-name lookup are deliberately unsupported.
NumericNameInfoResult ResolveNumericNameInfo(const Network::SockAddrIn& address, s32 flags,
                                             std::size_t host_capacity,
                                             std::size_t service_capacity);

// Command 13 uses 24 raw parameter bytes; sockaddr is the guest BSD byte layout.
// The initial implementation deliberately accepts no resolver options.
NumericNameInfoResult ResolveNumericNameInfoRequest(std::span<const u8> parameters,
                                                    std::span<const u8> sockaddr,
                                                    std::span<const u8> options,
                                                    std::size_t host_capacity,
                                                    std::size_t service_capacity);

struct NumericNameInfoResponse {
    GetAddrInfoError gai_error;
    s32 netdb_error;
    Errno bsd_errno;
};
static_assert(sizeof(NumericNameInfoResponse) == 12);
static_assert(offsetof(NumericNameInfoResponse, gai_error) == 0);
static_assert(offsetof(NumericNameInfoResponse, netdb_error) == 4);
static_assert(offsetof(NumericNameInfoResponse, bsd_errno) == 8);

NumericNameInfoResponse MakeNumericNameInfoResponse(GetAddrInfoError error);

// Bounded command13 parameters, using HIPC's declared raw-data end rather than
// unused words left in the fixed command-buffer array. Offset points at command ID.
std::span<const u8> GetNumericNameInfoParameters(std::span<const u32> command_buffer,
                                                 std::size_t data_payload_word_offset);

} // namespace Service::Sockets
