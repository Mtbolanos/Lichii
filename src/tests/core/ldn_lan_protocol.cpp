// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/ldn/direct_lan.h"
#include "core/hle/service/ldn/lan_protocol.h"
#include "core/hle/service/ldn/ldn_types.h"
#include "core/internal_network/network.h"
#include "core/internal_network/sockets.h"

namespace {
using namespace Service::LDN;
using namespace std::chrono_literals;

std::vector<u8> Bytes(std::initializer_list<u8> bytes) {
    return bytes;
}

constexpr Network::IPv4Address Loopback{127, 0, 0, 1};

// A DirectLan host on loopback with a raw TCP station that has sent its Connect
struct LoopbackHost {
    LoopbackHost()
        : host{[this](const Network::LDNPacket& packet) {
              std::scoped_lock lock{mutex};
              packets.push_back(packet);
              received.notify_all();
          }} {}

    bool WaitFor(std::size_t count) {
        std::unique_lock lock{mutex};
        return received.wait_for(lock, 2s, [&] { return packets.size() >= count; });
    }

    Network::NetworkInstance network;
    std::mutex mutex;
    std::condition_variable received;
    std::vector<Network::LDNPacket> packets;
    DirectLan host;
    Network::Socket station;
};

void ConnectStation(LoopbackHost& lan, std::span<const u8> node_info) {
    if (!lan.host.Open(Loopback, {255, 0, 0, 0})) {
        SKIP("Port " << LanPort << " is in use");
    }
    REQUIRE(lan.station.Initialize(Network::Domain::INET, Network::Type::STREAM,
                                   Network::Protocol::TCP) == Network::Errno::SUCCESS);
    lan.station.SetRcvBuf(4096);
    lan.station.SetRcvTimeo(2000);
    REQUIRE(lan.station.Connect({Network::Domain::INET, Loopback, LanPort}) ==
            Network::Errno::SUCCESS);
    lan.station.Send(EncodeFrame(FrameType::Connect, node_info), 0);
    REQUIRE(lan.WaitFor(1));
    CHECK(lan.packets[0].type == Network::LDNPacketType::Connect);
    CHECK(lan.packets[0].local_ip == Loopback);
    CHECK(lan.packets[0].data == std::vector<u8>(node_info.begin(), node_info.end()));
}

const auto NodeInfoBytes = Bytes({0, 0, 0, 0, 2, 0, 127, 0, 0, 1, 1, 1});
} // namespace

TEST_CASE("LAN frame header matches ldn_mitm's layout", "[core][ldn][LAN]") {
    const auto frame = EncodeFrame(FrameType::Connect, Bytes({1, 2, 3}));
    REQUIRE(frame.size() == sizeof(FrameHeader) + 3);
    CHECK(Bytes({frame[0], frame[1], frame[2], frame[3]}) == Bytes({0x00, 0x14, 0x45, 0x11}));
    CHECK(frame[4] == static_cast<u8>(FrameType::Connect));
    CHECK(frame[5] == 1);
    CHECK(Bytes({frame[6], frame[7], frame[8], frame[9]}) == Bytes({3, 0, 3, 0}));
    CHECK(Bytes({frame[12], frame[13], frame[14]}) == Bytes({1, 2, 3}));

    const auto scan = EncodeFrame(FrameType::Scan, {});
    CHECK(scan.size() == sizeof(FrameHeader));
}

TEST_CASE("LAN zero-run compression round trips", "[core][ldn][LAN]") {
    std::vector<u8> input(600, 0);
    input[0] = 7;
    input[300] = 9;
    const auto compressed = CompressFrameBody(input, input.size());
    REQUIRE(compressed);
    CHECK(*compressed == Bytes({7, 0, 0xFF, 0, 0x2A, 9, 0, 0xFF, 0, 0x2A}));
    CHECK(DecompressFrameBody(*compressed, FrameSizeMax) == input);

    CHECK(!CompressFrameBody(Bytes({0, 1, 0, 1}), 4));
    CHECK(!DecompressFrameBody(Bytes({5, 0}), FrameSizeMax));
}

TEST_CASE("LAN scan response parses from a split stream", "[core][ldn][LAN]") {
    NetworkInfo info{};
    info.common.bssid.raw = {0x02, 0x00, 192, 168, 1, 20};
    info.ldn.node_count = 1;
    info.ldn.nodes[0].ipv4_address = {20, 1, 168, 192};
    std::vector<u8> data(sizeof(info));
    std::memcpy(data.data(), &info, sizeof(info));

    const auto response_frame = EncodeFrame(FrameType::ScanResponse, data);
    CHECK(response_frame[5] == 1);
    CHECK(response_frame.size() < sizeof(FrameHeader) + sizeof(info));

    Frame frame;
    std::vector<u8> stream(response_frame.begin(), response_frame.begin() + 20);
    CHECK(TakeFrame(stream, frame) == FrameResult::Incomplete);
    stream.insert(stream.end(), response_frame.begin() + 20, response_frame.end());
    const auto scan = EncodeFrame(FrameType::Scan, {});
    stream.insert(stream.end(), scan.begin(), scan.end());

    REQUIRE(TakeFrame(stream, frame) == FrameResult::Complete);
    CHECK(frame.type == FrameType::ScanResponse);
    REQUIRE(frame.data.size() == sizeof(NetworkInfo));
    NetworkInfo parsed;
    std::memcpy(&parsed, frame.data.data(), sizeof(parsed));
    CHECK(parsed.common.bssid == info.common.bssid);
    CHECK(parsed.ldn.nodes[0].ipv4_address == info.ldn.nodes[0].ipv4_address);

    REQUIRE(TakeFrame(stream, frame) == FrameResult::Complete);
    CHECK(frame.type == FrameType::Scan);
    CHECK(stream.empty());

    std::vector<u8> garbage(sizeof(FrameHeader), 0xAB);
    CHECK(TakeFrame(garbage, frame) == FrameResult::Malformed);
    CHECK(garbage.empty());
}

TEST_CASE("LAN host takes a station over loopback TCP", "[core][ldn][LAN]") {
    LoopbackHost lan;
    ConnectStation(lan, NodeInfoBytes);

    lan.host.Send(
        {Network::LDNPacketType::SyncNetwork, Loopback, Loopback, false, Bytes({4, 0, 5})});
    std::vector<u8> stream;
    Frame frame;
    std::array<u8, FrameSizeMax> buffer;
    while (TakeFrame(stream, frame) == FrameResult::Incomplete) {
        const auto [size, error] = lan.station.Recv(0, buffer);
        REQUIRE(size > 0);
        stream.insert(stream.end(), buffer.begin(), buffer.begin() + size);
    }
    CHECK(frame.type == FrameType::SyncNetwork);
    CHECK(frame.data == Bytes({4, 0, 5}));

    lan.station.Close();
    REQUIRE(lan.WaitFor(2));
    CHECK(lan.packets[1].type == Network::LDNPacketType::Disconnect);
    CHECK(lan.packets[1].data == NodeInfoBytes);
    lan.host.Close();
}

TEST_CASE("LAN host drops a station that stops reading", "[core][ldn][LAN]") {
    LoopbackHost lan;
    ConnectStation(lan, NodeInfoBytes);

    std::vector<u8> network_info(sizeof(NetworkInfo));
    for (std::size_t i = 0; i < network_info.size(); ++i) {
        network_info[i] = static_cast<u8>(i % 251 + 1);
    }
    bool dropped = false;
    for (int i = 0; i < 100000 && !dropped; ++i) {
        lan.host.Send(
            {Network::LDNPacketType::SyncNetwork, Loopback, Loopback, false, network_info});
        std::scoped_lock lock{lan.mutex};
        dropped = lan.packets.size() >= 2;
    }
    REQUIRE((dropped || lan.WaitFor(2)));
    CHECK(lan.packets[1].type == Network::LDNPacketType::Disconnect);
    CHECK(lan.packets[1].data == NodeInfoBytes);
    lan.host.Close();
}
