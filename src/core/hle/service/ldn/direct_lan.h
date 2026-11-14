// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "common/socket_types.h"
#include "network/room_member.h"

namespace Network {
class SocketBase;
}

namespace Service::LDN {

/// Carries LANDiscovery's packets over the host network with ldn_mitm's LAN protocol instead of a
/// room. Guest traffic between the nodes then uses the host sockets and the nodes' real addresses.
class DirectLan {
public:
    using PacketHandler = std::function<void(const Network::LDNPacket&)>;

    explicit DirectLan(PacketHandler handler_);
    ~DirectLan();

    bool Open(Network::IPv4Address ip, Network::IPv4Address subnet_mask);
    /// Stops the worker, which delivers no more packets once this returns.
    void Close();

    Network::IPv4Address GetLocalIp() const {
        return local_ip;
    }

    /// Joins the host's network with the station's NodeInfo. Waits on the host, so the caller must
    /// not hold a lock the packet handler takes.
    bool ConnectToHost(Network::IPv4Address host_ip, std::span<const u8> node_info);

    /// Never waits on a peer. A connection that can't take a whole frame at once is dropped.
    void Send(const Network::LDNPacket& packet);

private:
    struct Connection {
        std::unique_ptr<Network::SocketBase> socket;
        Network::IPv4Address peer;
        bool to_host{};
        /// Set under the mutex when a send failed; the worker closes it.
        bool dropped{};
        std::chrono::steady_clock::time_point opened_at;
        std::vector<u8> received;
        /// The station's Connect payload, for the Disconnect its closing stands for.
        std::vector<u8> node_info;
    };

    static bool SendFrame(Connection& connection, std::span<const u8> frame);
    void Run(std::stop_token stop_token);
    void ReceiveDatagrams(std::vector<Network::LDNPacket>& packets);
    void Accept();
    /// Whether the connection is still open after reading what it holds.
    bool ReceiveStream(Connection& connection, std::vector<Network::LDNPacket>& packets);
    void OnClosed(const std::shared_ptr<Connection>& connection,
                  std::vector<Network::LDNPacket>& packets);
    bool IsOnSubnet(Network::IPv4Address ip) const;

    PacketHandler handler;
    Network::IPv4Address local_ip{};
    Network::IPv4Address mask{};
    Network::IPv4Address broadcast_ip{};

    std::unique_ptr<Network::SocketBase> udp;
    std::unique_ptr<Network::SocketBase> listener;

    std::mutex mutex;
    std::shared_ptr<Connection> host;
    std::vector<std::shared_ptr<Connection>> stations;

    std::jthread worker;
};

} // namespace Service::LDN
