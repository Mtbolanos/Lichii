// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>
#include <iterator>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "core/internal_network/local_redirect.h"
#include "core/internal_network/network.h"

namespace LocalRedirect = Network::LocalRedirect;

namespace {

constexpr Network::SockAddrIn LoopbackHttps{Network::Domain::INET, {127, 0, 0, 1}, 443};
constexpr Network::SockAddrIn Public{Network::Domain::INET, {192, 0, 2, 7}, 443};

// A self-signed P-256 test certificate (CN=Test CA), DER.
constexpr unsigned char TestCertificate[] = {
    0x30, 0x82, 0x01, 0x7A, 0x30, 0x82, 0x01, 0x21, 0xA0, 0x03, 0x02, 0x01,
    0x02, 0x02, 0x14, 0x7B, 0x3F, 0xAE, 0x7F, 0x05, 0x43, 0x05, 0x4F, 0x8A,
    0x01, 0x95, 0xB0, 0x54, 0x28, 0xCA, 0x3F, 0xEF, 0x34, 0x81, 0x13, 0x30,
    0x0A, 0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02, 0x30,
    0x12, 0x31, 0x10, 0x30, 0x0E, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0C, 0x07,
    0x54, 0x65, 0x73, 0x74, 0x20, 0x43, 0x41, 0x30, 0x20, 0x17, 0x0D, 0x32,
    0x36, 0x31, 0x31, 0x31, 0x32, 0x31, 0x36, 0x35, 0x36, 0x31, 0x34, 0x5A,
    0x18, 0x0F, 0x32, 0x31, 0x32, 0x36, 0x31, 0x30, 0x31, 0x39, 0x31, 0x36,
    0x35, 0x36, 0x31, 0x34, 0x5A, 0x30, 0x12, 0x31, 0x10, 0x30, 0x0E, 0x06,
    0x03, 0x55, 0x04, 0x03, 0x0C, 0x07, 0x54, 0x65, 0x73, 0x74, 0x20, 0x43,
    0x41, 0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D,
    0x02, 0x01, 0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07,
    0x03, 0x42, 0x00, 0x04, 0xDA, 0xF7, 0xEC, 0xB5, 0xB8, 0x4A, 0x59, 0xC1,
    0xF3, 0xCF, 0xE8, 0xE3, 0xC2, 0xA2, 0xF2, 0x15, 0x6A, 0xD9, 0x86, 0xA6,
    0x49, 0x4B, 0x08, 0x41, 0xD5, 0x4B, 0xCB, 0x0D, 0xAB, 0x90, 0xBB, 0xF4,
    0xA3, 0xDF, 0xBD, 0x54, 0x0B, 0xCF, 0xE3, 0x25, 0xD1, 0x46, 0xB9, 0x73,
    0x60, 0x74, 0x8D, 0xC7, 0xC0, 0xAE, 0xB4, 0x20, 0x27, 0x6F, 0x39, 0x5F,
    0x58, 0xE5, 0xD4, 0x89, 0xBE, 0xC3, 0x19, 0xA3, 0xA3, 0x53, 0x30, 0x51,
    0x30, 0x1D, 0x06, 0x03, 0x55, 0x1D, 0x0E, 0x04, 0x16, 0x04, 0x14, 0xAF,
    0xDD, 0x6F, 0x9D, 0xBC, 0x03, 0xC1, 0x8A, 0x65, 0x1B, 0xC8, 0xB2, 0x95,
    0x46, 0x4F, 0xD4, 0x6D, 0x21, 0x36, 0xBF, 0x30, 0x1F, 0x06, 0x03, 0x55,
    0x1D, 0x23, 0x04, 0x18, 0x30, 0x16, 0x80, 0x14, 0xAF, 0xDD, 0x6F, 0x9D,
    0xBC, 0x03, 0xC1, 0x8A, 0x65, 0x1B, 0xC8, 0xB2, 0x95, 0x46, 0x4F, 0xD4,
    0x6D, 0x21, 0x36, 0xBF, 0x30, 0x0F, 0x06, 0x03, 0x55, 0x1D, 0x13, 0x01,
    0x01, 0xFF, 0x04, 0x05, 0x30, 0x03, 0x01, 0x01, 0xFF, 0x30, 0x0A, 0x06,
    0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02, 0x03, 0x47, 0x00,
    0x30, 0x44, 0x02, 0x20, 0x4F, 0xA2, 0x0A, 0xDC, 0xEF, 0xB6, 0x31, 0x3B,
    0x5B, 0xA6, 0xA5, 0x32, 0xCF, 0x9D, 0x9C, 0xA9, 0x5C, 0x4B, 0xAE, 0xEF,
    0xE0, 0xF8, 0x1B, 0xF8, 0x8F, 0xC1, 0x6E, 0x53, 0xFE, 0xA5, 0xB8, 0xF0,
    0x02, 0x20, 0x16, 0x44, 0xB7, 0xFE, 0x7B, 0x3A, 0x1B, 0x10, 0x7D, 0x19,
    0x94, 0x82, 0x54, 0x1B, 0x7F, 0x55, 0x89, 0x65, 0xF4, 0x47, 0x40, 0xA2,
    0x1B, 0xC6, 0x17, 0xA0, 0xBF, 0xDE, 0xE8, 0xFB, 0x51, 0xB3
};

void WriteFile(const std::filesystem::path& path, const void* data, std::streamsize size) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file.write(static_cast<const char*>(data), size);
}

bool Same(const Network::SockAddrIn& a, const Network::SockAddrIn& b) {
    return a.family == b.family && a.ip == b.ip && a.portno == b.portno;
}

void ResetRedirectSettings() {
    Settings::values.online_service_redirect.SetValue(false);
    Settings::values.online_redirect_hosts.SetValue({});
    Settings::values.online_redirect_server.SetValue({});
    Settings::values.online_redirect_ca.SetValue({});
    Settings::values.online_redirect_loopback_only.SetValue(false);
    LocalRedirect::Configure();
}

} // Anonymous namespace

TEST_CASE("LocalRedirect: off by default changes nothing", "[core]") {
    REQUIRE_FALSE(Settings::values.online_service_redirect.GetDefault());
    REQUIRE_FALSE(Settings::values.online_redirect_loopback_only.GetDefault());

    ResetRedirectSettings();
    Settings::values.online_redirect_hosts = "a.example";
    Settings::values.online_redirect_server = "192.0.2.1:8443";
    Settings::values.online_redirect_loopback_only = true;
    LocalRedirect::Configure();

    REQUIRE_FALSE(LocalRedirect::Matches("a.example"));
    REQUIRE_FALSE(LocalRedirect::Resolve("a.example", "https").has_value());
    REQUIRE(Same(LocalRedirect::ConnectTarget(LoopbackHttps), LoopbackHttps));
    REQUIRE(LocalRedirect::Allowed(Public));
    REQUIRE_FALSE(LocalRedirect::LoopbackOnly());
    REQUIRE(LocalRedirect::TestCa().empty());

    ResetRedirectSettings();
}

TEST_CASE("LocalRedirect: host names map to the user's server", "[core]") {
    ResetRedirectSettings();
    Settings::values.online_service_redirect = true;
    Settings::values.online_redirect_hosts = " A.Example. ,, b.example";
    Settings::values.online_redirect_server = "192.0.2.1:8443";
    LocalRedirect::Configure();

    REQUIRE(LocalRedirect::Matches("a.example"));
    REQUIRE(LocalRedirect::Matches("API.a.example."));
    REQUIRE(LocalRedirect::Matches("b.example"));
    REQUIRE_FALSE(LocalRedirect::Matches("nota.example"));
    REQUIRE_FALSE(LocalRedirect::Matches("a.example.org"));
    REQUIRE_FALSE(LocalRedirect::Matches("example"));

    const auto resolved = LocalRedirect::Resolve("api.a.example", "https");
    REQUIRE(resolved.has_value());
    REQUIRE(resolved->size() == 1);
    REQUIRE(Same(resolved->front().addr, LoopbackHttps));
    REQUIRE(LocalRedirect::Resolve("a.example", "8080")->front().addr.portno == 8080);
    REQUIRE_FALSE(LocalRedirect::Resolve("c.example", "https").has_value());

    const Network::SockAddrIn server{Network::Domain::INET, {192, 0, 2, 1}, 8443};
    REQUIRE(Same(LocalRedirect::ConnectTarget(LoopbackHttps), server));
    const Network::SockAddrIn loopback_http{Network::Domain::INET, {127, 0, 0, 1}, 80};
    REQUIRE(Same(LocalRedirect::ConnectTarget(loopback_http), loopback_http));
    REQUIRE(Same(LocalRedirect::ConnectTarget(Public), Public));
    REQUIRE(LocalRedirect::Allowed(Public));

    Settings::values.online_redirect_loopback_only = true;
    LocalRedirect::Configure();
    REQUIRE_FALSE(LocalRedirect::Allowed(Public));
    REQUIRE(LocalRedirect::Allowed(LoopbackHttps));

    // Other names never reach the host resolver; numeric loopback and localhost still resolve.
    REQUIRE(std::holds_alternative<Network::GetAddrInfoError>(
        Network::GetAddressInfo("c.example", std::nullopt)));
    REQUIRE(std::holds_alternative<Network::GetAddrInfoError>(
        Network::GetAddressInfo("192.0.2.7", std::nullopt)));
    for (const char* name : {"localhost", "LOCALHOST.", "localhost..", "127.0.0.1"}) {
        const auto local = Network::GetAddressInfo(name, std::nullopt);
        REQUIRE(std::holds_alternative<std::vector<Network::AddrInfo>>(local));
        for (const auto& info : std::get<std::vector<Network::AddrInfo>>(local)) {
            REQUIRE(info.addr.ip == Network::IPv4Address{127, 0, 0, 1});
        }
    }

    ResetRedirectSettings();
    REQUIRE_FALSE(LocalRedirect::Matches("a.example"));
}

TEST_CASE("LocalRedirect: malformed server and CA certificates are ignored", "[core]") {
    ResetRedirectSettings();
    Settings::values.online_service_redirect = true;
    Settings::values.online_redirect_hosts = "a.example";
    Settings::values.online_redirect_server = "192.0.2.1";
    const auto ca_path = std::filesystem::temp_directory_path() / "lichii_redirect_test_ca.der";
    constexpr char Pem[] = "-----BEGIN CERTIFICATE-----";
    WriteFile(ca_path, Pem, sizeof(Pem) - 1);
    Settings::values.online_redirect_ca = ca_path.string();
    LocalRedirect::Configure();
    REQUIRE(Same(LocalRedirect::ConnectTarget(LoopbackHttps), LoopbackHttps));
    REQUIRE(LocalRedirect::TestCa().empty());

    // An ASN.1 sequence that isn't a certificate.
    constexpr unsigned char Sequence[] = {0x30, 0x03, 0x02, 0x01, 0x00};
    WriteFile(ca_path, Sequence, sizeof(Sequence));
    LocalRedirect::Configure();
    REQUIRE(LocalRedirect::TestCa().empty());

    // A certificate followed by other bytes.
    std::vector<unsigned char> trailing{std::begin(TestCertificate), std::end(TestCertificate)};
    trailing.push_back(0);
    WriteFile(ca_path, trailing.data(), static_cast<std::streamsize>(trailing.size()));
    LocalRedirect::Configure();
    REQUIRE(LocalRedirect::TestCa().empty());

    WriteFile(ca_path, TestCertificate, sizeof(TestCertificate));
    LocalRedirect::Configure();
    REQUIRE(LocalRedirect::TestCa().size() == sizeof(TestCertificate));

    std::filesystem::remove(ca_path);
    ResetRedirectSettings();
}
