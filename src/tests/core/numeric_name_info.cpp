// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/ipc.h"
#include "core/hle/service/sockets/numeric_name_info.h"

namespace {
using Service::Sockets::GetAddrInfoError;
using Service::Sockets::ResolveNumericNameInfo;

std::string Text(const std::vector<u8>& bytes) {
    if (bytes.empty() || bytes.back() != 0) {
        return "<missing NUL-terminated output>";
    }
    return {bytes.begin(), bytes.end() - 1};
}

constexpr Network::SockAddrIn Address{Network::Domain::INET, {192, 168, 1, 1}, 7777};
} // namespace

TEST_CASE("Numeric name info formats the supplied IPv4, not an empty success", "[nameinfo]") {
    const auto result = ResolveNumericNameInfo(Address, 2, 1025, 0);
    REQUIRE(result.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(result.host) == "192.168.1.1");
    CHECK(result.service.empty());

    const Network::SockAddrIn other{Network::Domain::INET, {10, 23, 45, 67}, 32109};
    const auto changed = ResolveNumericNameInfo(other, 2 | 8, 13, 6);
    REQUIRE(changed.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(changed.host) == "10.23.45.67");
    CHECK(Text(changed.service) == "32109");
}

TEST_CASE("Numeric name info includes terminators in capacity checks", "[nameinfo]") {
    const Network::SockAddrIn largest{Network::Domain::INET, {255, 255, 255, 255}, 65535};
    const auto exact = ResolveNumericNameInfo(largest, 2 | 8, 16, 6);
    REQUIRE(exact.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(exact.host) == "255.255.255.255");
    CHECK(Text(exact.service) == "65535");

    for (const auto& [host_size, service_size] :
         std::array<std::pair<std::size_t, std::size_t>, 3>{{{15, 6}, {16, 5}, {1, 6}}}) {
        const auto small = ResolveNumericNameInfo(largest, 2 | 8, host_size, service_size);
        CHECK(small.error == GetAddrInfoError::OVERFLOW_);
        CHECK(small.host.empty());
        CHECK(small.service.empty());
    }
    // Capacity is a bound, never an instruction to allocate capacity bytes.
    const auto huge =
        ResolveNumericNameInfo(largest, 2 | 8, std::numeric_limits<std::size_t>::max(), 6);
    REQUIRE(huge.error == GetAddrInfoError::SUCCESS);
    CHECK(huge.host.size() == 16);
}

TEST_CASE("Numeric name info supports omitted outputs and zero port", "[nameinfo]") {
    const Network::SockAddrIn zero{Network::Domain::INET, {0, 0, 0, 0}, 0};
    const auto service_only = ResolveNumericNameInfo(zero, 8, 0, 2);
    REQUIRE(service_only.error == GetAddrInfoError::SUCCESS);
    CHECK(service_only.host.empty());
    CHECK(Text(service_only.service) == "0");
    const auto host_only = ResolveNumericNameInfo(zero, 2, 8, 0);
    REQUIRE(host_only.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(host_only.host) == "0.0.0.0");
    CHECK(ResolveNumericNameInfo(zero, 2 | 8, 0, 0).error == GetAddrInfoError::NONAME);
}

TEST_CASE("Numeric name info rejects unsupported requests without outputs or DNS", "[nameinfo]") {
    const auto check_failure = [](const Network::SockAddrIn& address, s32 flags, std::size_t host,
                                  std::size_t service, GetAddrInfoError error) {
        const auto result = ResolveNumericNameInfo(address, flags, host, service);
        CHECK(result.error == error);
        CHECK(result.host.empty());
        CHECK(result.service.empty());
    };
    auto unsupported = Address;
    unsupported.family = Network::Domain::Unspecified;
    check_failure(unsupported, 2, 1025, 0, GetAddrInfoError::FAMILY);
    check_failure(Address, 0x20, 1025, 0, GetAddrInfoError::BADFLAGS);
    check_failure(Address, -1, 1025, 0, GetAddrInfoError::BADFLAGS);
    check_failure(Address, 0, 1025, 0, GetAddrInfoError::NONAME);
    check_failure(Address, 4, 1025, 0, GetAddrInfoError::NONAME);
    check_failure(Address, 2, 1025, 16, GetAddrInfoError::NONAME);
    // NOFQDN, NAMEREQD and DGRAM do not alter explicitly numeric output.
    const auto numeric = ResolveNumericNameInfo(Address, 1 | 2 | 4 | 8 | 16, 16, 6);
    REQUIRE(numeric.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(numeric.host) == "192.168.1.1");
    CHECK(Text(numeric.service) == "7777");
}

TEST_CASE("NameInfo13 parses the SDK request and BSD sockaddr without host endian constants",
          "[nameinfo][ipc]") {
    // flags2, inherited value deliberately nonzero, options_count0, PID placeholder0.
    const std::array<u8, 24> raw{2, 0, 0, 0, 0x78, 0x56, 0x34, 0x12};
    const std::array<u8, 16> address{16, 2, 0x1e, 0x61, 192, 168, 1, 1};
    const auto result = Service::Sockets::ResolveNumericNameInfoRequest(raw, address, {}, 1025, 0);
    REQUIRE(result.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(result.host) == "192.168.1.1");
    CHECK(result.service.empty());

    auto numeric_service = raw;
    numeric_service[0] = 2 | 8;
    const auto with_service =
        Service::Sockets::ResolveNumericNameInfoRequest(numeric_service, address, {}, 12, 5);
    REQUIRE(with_service.error == GetAddrInfoError::SUCCESS);
    CHECK(Text(with_service.service) == "7777");
}

TEST_CASE("NameInfo13 rejects incomplete data and unsupported options without partial output",
          "[nameinfo][ipc]") {
    const std::array<u8, 24> raw{2};
    const std::array<u8, 16> address{16, 2, 0x1e, 0x61, 192, 168, 1, 1};
    const auto failure = [](std::span<const u8> request, std::span<const u8> addr,
                            std::span<const u8> options, GetAddrInfoError expected) {
        const auto result =
            Service::Sockets::ResolveNumericNameInfoRequest(request, addr, options, 1025, 0);
        CHECK(result.error == expected);
        CHECK(result.host.empty());
        CHECK(result.service.empty());
    };
    for (std::size_t length = 0; length < raw.size(); ++length) {
        failure(std::span{raw}.first(length), address, {}, GetAddrInfoError::FAIL);
    }
    for (std::size_t length = 0; length < address.size(); ++length) {
        failure(raw, std::span{address}.first(length), {}, GetAddrInfoError::FAIL);
    }
    auto with_options = raw;
    with_options[8] = 1;
    failure(with_options, address, {}, GetAddrInfoError::FAIL);
    const std::array<u8, 20> option{};
    failure(with_options, address, option, GetAddrInfoError::FAIL);
    failure(raw, address, option, GetAddrInfoError::FAIL);
    auto other_family = address;
    other_family[1] = 28;
    failure(raw, other_family, {}, GetAddrInfoError::FAMILY);
    auto short_length = address;
    short_length[0] = 8;
    failure(raw, short_length, {}, GetAddrInfoError::FAIL);
    const std::array<u8, 257> huge{};
    failure(raw, huge, {}, GetAddrInfoError::FAIL);
}

TEST_CASE("NameInfo13 returns three fields and never hides resolver errors behind IPC success",
          "[nameinfo][ipc]") {
    const auto response =
        Service::Sockets::MakeNumericNameInfoResponse(GetAddrInfoError::OVERFLOW_);
    std::array<s32, 3> words{};
    std::memcpy(words.data(), &response, sizeof(response));
    CHECK(words[0] == 14);
    CHECK(words[1] == 0);
    CHECK(words[2] == 0);
    const auto success = Service::Sockets::MakeNumericNameInfoResponse(GetAddrInfoError::SUCCESS);
    CHECK(success.gai_error == GetAddrInfoError::SUCCESS);
    CHECK(success.netdb_error == 0);
    CHECK(success.bsd_errno == Service::Sockets::Errno::SUCCESS);
}

TEST_CASE("NameInfo13 bounds raw parameters by the HIPC data size", "[nameinfo][ipc]") {
    std::array<u32, IPC::COMMAND_BUFFER_LENGTH> words{};
    IPC::CommandHeader header{};
    header.type.Assign(IPC::CommandType::Request);
    header.data_size.Assign(12); // descriptors end2, raw end14, parameters start8.
    std::memcpy(words.data(), &header, sizeof(header));
    words[4] = 0x49434653;
    words[6] = 13;
    words[8] = 2;
    const auto raw = Service::Sockets::GetNumericNameInfoParameters(words, 6);
    REQUIRE(raw.size() == 24);
    CHECK(raw[0] == 2);
    CHECK(raw.data() == reinterpret_cast<const u8*>(words.data() + 8));

    header.data_size.Assign(11); // Last parameter word is not actually present.
    std::memcpy(words.data(), &header, sizeof(header));
    CHECK(Service::Sockets::GetNumericNameInfoParameters(words, 6).empty());
    header.data_size.Assign(1023);
    std::memcpy(words.data(), &header, sizeof(header));
    CHECK(Service::Sockets::GetNumericNameInfoParameters(words, 6).empty());
    CHECK(Service::Sockets::GetNumericNameInfoParameters(std::span{words}.first(1), 6).empty());
    CHECK(Service::Sockets::GetNumericNameInfoParameters(words, words.size()).empty());
    CHECK(Service::Sockets::GetNumericNameInfoParameters(words,
                                                         std::numeric_limits<std::size_t>::max())
              .empty());
    header.data_size.Assign(12);
    header.enable_handle_descriptor.Assign(1);
    std::memcpy(words.data(), &header, sizeof(header));
    CHECK(Service::Sockets::GetNumericNameInfoParameters(std::span{words}.first(2), 6).empty());
}
