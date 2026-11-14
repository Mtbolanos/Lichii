// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>

#include <mstcpip.h>
#else
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

#include "common/logging.h"
#include "core/hle/service/ldn/direct_lan.h"
#include "core/hle/service/ldn/lan_protocol.h"
#include "core/hle/service/ldn/ldn_types.h"
#include "core/internal_network/network.h"
#include "core/internal_network/sockets.h"

namespace Service::LDN {

namespace {

constexpr s32 PollIntervalMs = 50;
constexpr s32 ConnectTimeoutMs = 2000;
constexpr s32 ListenBacklog = 8;
constexpr std::size_t DatagramsPerWake = 16;
constexpr auto ConnectDeadline = std::chrono::seconds(5);
constexpr int KeepAliveIdleSeconds = 5;
constexpr int KeepAliveIntervalSeconds = 1;
constexpr int KeepAliveProbes = 3;

Network::SockAddrIn LanAddress(Network::IPv4Address ip) {
    return {Network::Domain::INET, ip, LanPort};
}

bool Succeeded(Network::Errno result) {
    return result == Network::Errno::SUCCESS;
}

void SetOption(Network::SocketBase& socket, int level, int option, int value) {
    setsockopt(socket.GetFD(), level, option, reinterpret_cast<const char*>(&value), sizeof(value));
}

// TCP keepalive notices a peer that vanished without closing, below the protocol, which has no
// keepalive of its own.
void ConfigureStream(Network::SocketBase& socket) {
    socket.SetNonBlock(true);
    socket.SetNoDelay(true);
    socket.SetKeepAlive(true);
#if defined(TCP_KEEPIDLE)
    SetOption(socket, IPPROTO_TCP, TCP_KEEPIDLE, KeepAliveIdleSeconds);
#elif defined(TCP_KEEPALIVE)
    SetOption(socket, IPPROTO_TCP, TCP_KEEPALIVE, KeepAliveIdleSeconds);
#endif
#ifdef TCP_KEEPINTVL
    SetOption(socket, IPPROTO_TCP, TCP_KEEPINTVL, KeepAliveIntervalSeconds);
#endif
#ifdef TCP_KEEPCNT
    SetOption(socket, IPPROTO_TCP, TCP_KEEPCNT, KeepAliveProbes);
#endif
#ifdef TCP_USER_TIMEOUT
    SetOption(socket, IPPROTO_TCP, TCP_USER_TIMEOUT,
              (KeepAliveIdleSeconds + KeepAliveIntervalSeconds * KeepAliveProbes) * 1000);
#endif
#ifdef SO_NOSIGPIPE
    SetOption(socket, SOL_SOCKET, SO_NOSIGPIPE, 1);
#endif
}

} // namespace

bool DirectLan::SendFrame(Connection& connection, std::span<const u8> frame) {
    if (connection.dropped) {
        return false;
    }
    const auto [sent, error] = connection.socket->Send(frame, Network::FLAG_MSG_DONTWAIT);
    if (sent == static_cast<s32>(frame.size())) {
        return true;
    }
    // A partial frame would desynchronise the stream, so the worker closes the connection
    connection.dropped = true;
    return false;
}

DirectLan::DirectLan(PacketHandler handler_) : handler{std::move(handler_)} {}

DirectLan::~DirectLan() {
    Close();
}

bool DirectLan::Open(Network::IPv4Address ip, Network::IPv4Address subnet_mask) {
    local_ip = ip;
    mask = subnet_mask;
    for (std::size_t i = 0; i < ip.size(); ++i) {
        broadcast_ip[i] = static_cast<u8>(ip[i] | ~subnet_mask[i]);
    }

    udp = std::make_unique<Network::Socket>();
    listener = std::make_unique<Network::Socket>();
    bool opened = Succeeded(udp->Initialize(Network::Domain::INET, Network::Type::DGRAM,
                                            Network::Protocol::UDP)) &&
                  Succeeded(udp->SetReuseAddr(true)) && Succeeded(udp->SetBroadcast(true)) &&
                  Succeeded(udp->Bind(LanAddress({}))) &&
                  Succeeded(listener->Initialize(Network::Domain::INET, Network::Type::STREAM,
                                                 Network::Protocol::TCP));
    if (opened) {
#ifdef _WIN32
        // SO_REUSEADDR would let a second process share the listening port
        SetOption(*listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#ifdef SIO_UDP_CONNRESET
        BOOL report_reset = FALSE;
        DWORD bytes_returned = 0;
        WSAIoctl(udp->GetFD(), SIO_UDP_CONNRESET, &report_reset, sizeof(report_reset), nullptr, 0,
                 &bytes_returned, nullptr, nullptr);
#endif
#else
        listener->SetReuseAddr(true);
#endif
        opened = Succeeded(listener->Bind(LanAddress(local_ip))) &&
                 Succeeded(listener->Listen(ListenBacklog)) &&
                 Succeeded(listener->SetNonBlock(true));
    }
    if (!opened) {
        LOG_ERROR(Service_LDN, "LAN play could not open port {} on {}", LanPort,
                  Network::IPv4AddressToString(local_ip));
        udp.reset();
        listener.reset();
        return false;
    }

    LOG_INFO(Service_LDN, "LAN play on {}, broadcasting to {}",
             Network::IPv4AddressToString(local_ip), Network::IPv4AddressToString(broadcast_ip));
    Network::SetDirectLanActive(true);
    worker = std::jthread([this](std::stop_token stop_token) { Run(stop_token); });
    return true;
}

void DirectLan::Close() {
    if (worker.joinable()) {
        worker.request_stop();
        worker.join();
        Network::SetDirectLanActive(false);
    }
    std::scoped_lock lock{mutex};
    host.reset();
    stations.clear();
    listener.reset();
    udp.reset();
}

bool DirectLan::ConnectToHost(Network::IPv4Address host_ip, std::span<const u8> node_info) {
    if (!IsOnSubnet(host_ip)) {
        LOG_ERROR(Service_LDN, "LAN play host {} is not on the local subnet",
                  Network::IPv4AddressToString(host_ip));
        return false;
    }
    auto connection = std::make_shared<Connection>();
    connection->socket = std::make_unique<Network::Socket>();
    connection->peer = host_ip;
    connection->to_host = true;
    auto& socket = *connection->socket;
    if (!Succeeded(socket.Initialize(Network::Domain::INET, Network::Type::STREAM,
                                     Network::Protocol::TCP)) ||
        !Succeeded(socket.Bind({Network::Domain::INET, local_ip, 0}))) {
        return false;
    }
    ConfigureStream(socket);

    const auto result = socket.Connect(LanAddress(host_ip));
    if (result == Network::Errno::INPROGRESS || result == Network::Errno::AGAIN) {
        std::vector<Network::PollFD> poll_fds{{&socket, Network::PollEvents::Out, {}}};
        Network::Poll(poll_fds, ConnectTimeoutMs);
        if (False(poll_fds[0].revents & Network::PollEvents::Out) ||
            socket.GetPendingError().first != Network::Errno::SUCCESS) {
            LOG_ERROR(Service_LDN, "LAN play could not reach the host at {}",
                      Network::IPv4AddressToString(host_ip));
            return false;
        }
    } else if (result != Network::Errno::SUCCESS) {
        return false;
    }

    std::scoped_lock lock{mutex};
    host = connection;
    if (!SendFrame(*connection, EncodeFrame(FrameType::Connect, node_info))) {
        host.reset();
        return false;
    }
    return true;
}

void DirectLan::Send(const Network::LDNPacket& packet) {
    switch (packet.type) {
    case Network::LDNPacketType::Scan:
    case Network::LDNPacketType::ScanResp: {
        const auto type =
            packet.type == Network::LDNPacketType::Scan ? FrameType::Scan : FrameType::ScanResponse;
        const auto destination = LanAddress(packet.broadcast ? broadcast_ip : packet.remote_ip);
        udp->SendTo(Network::FLAG_MSG_DONTWAIT, EncodeFrame(type, packet.data), &destination);
        break;
    }
    case Network::LDNPacketType::SyncNetwork: {
        const auto frame = EncodeFrame(FrameType::SyncNetwork, packet.data);
        std::scoped_lock lock{mutex};
        for (const auto& station : stations) {
            if (station->peer == packet.remote_ip && !station->node_info.empty()) {
                SendFrame(*station, frame);
            }
        }
        break;
    }
    case Network::LDNPacketType::Disconnect: {
        std::scoped_lock lock{mutex};
        host.reset();
        break;
    }
    case Network::LDNPacketType::DestroyNetwork: {
        std::scoped_lock lock{mutex};
        std::erase_if(stations,
                      [&](const auto& station) { return station->peer == packet.remote_ip; });
        break;
    }
    case Network::LDNPacketType::Connect:
        break;
    }
}

void DirectLan::Run(std::stop_token stop_token) {
    std::vector<std::shared_ptr<Connection>> connections;
    std::vector<std::shared_ptr<Connection>> dropped;
    std::vector<Network::PollFD> poll_fds;
    std::vector<Network::LDNPacket> packets;
    while (!stop_token.stop_requested()) {
        {
            std::scoped_lock lock{mutex};
            const auto now = std::chrono::steady_clock::now();
            std::erase_if(stations, [&](const auto& station) {
                return station->node_info.empty() && now - station->opened_at > ConnectDeadline;
            });
            connections = stations;
            if (host) {
                connections.push_back(host);
            }
            dropped.clear();
            std::erase_if(connections, [&](const auto& connection) {
                if (connection->dropped) {
                    dropped.push_back(connection);
                }
                return connection->dropped;
            });
        }
        for (const auto& connection : dropped) {
            OnClosed(connection, packets);
        }

        poll_fds.clear();
        poll_fds.push_back({udp.get(), Network::PollEvents::In, {}});
        poll_fds.push_back({listener.get(), Network::PollEvents::In, {}});
        for (const auto& connection : connections) {
            poll_fds.push_back({connection->socket.get(), Network::PollEvents::In, {}});
        }
        const auto [ready, error] = Network::Poll(poll_fds, PollIntervalMs);
        const bool any_ready = std::ranges::any_of(poll_fds, [](const Network::PollFD& fd) {
            return fd.revents != Network::PollEvents{};
        });
        if (ready != 0 && !any_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(PollIntervalMs));
        }
        if (ready > 0 && any_ready) {
            if (poll_fds[0].revents != Network::PollEvents{}) {
                ReceiveDatagrams(packets);
            }
            if (poll_fds[1].revents != Network::PollEvents{}) {
                Accept();
            }
            for (std::size_t i = 0; i < connections.size(); ++i) {
                if (poll_fds[i + 2].revents != Network::PollEvents{} &&
                    !ReceiveStream(*connections[i], packets)) {
                    OnClosed(connections[i], packets);
                }
            }
        }

        for (const auto& packet : packets) {
            handler(packet);
        }
        packets.clear();
    }
}

void DirectLan::ReceiveDatagrams(std::vector<Network::LDNPacket>& packets) {
    std::array<u8, FrameSizeMax> buffer;
    for (std::size_t received = 0; received < DatagramsPerWake; ++received) {
        Network::SockAddrIn source{};
        const auto [size, error] = udp->RecvFrom(Network::FLAG_MSG_DONTWAIT, buffer, &source);
        if (error == Network::Errno::CONNRESET) {
            continue;
        }
        if (size <= 0 || error != Network::Errno::SUCCESS) {
            return;
        }
        if (source.ip == local_ip || !IsOnSubnet(source.ip)) {
            continue;
        }
        std::vector<u8> datagram(buffer.begin(), buffer.begin() + size);
        Frame frame;
        if (TakeFrame(datagram, frame) != FrameResult::Complete ||
            (frame.type != FrameType::Scan && frame.type != FrameType::ScanResponse)) {
            continue;
        }
        packets.push_back({
            .type = frame.type == FrameType::Scan ? Network::LDNPacketType::Scan
                                                  : Network::LDNPacketType::ScanResp,
            .local_ip = source.ip,
            .remote_ip = local_ip,
            .broadcast = false,
            .data = std::move(frame.data),
        });
    }
}

void DirectLan::Accept() {
    auto [accepted, error] = listener->Accept();
    if (error != Network::Errno::SUCCESS || !accepted.socket) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (!IsOnSubnet(accepted.sockaddr_in.ip) || stations.size() >= StationCountMax) {
        LOG_WARNING(Service_LDN, "LAN play refused a connection from {}",
                    Network::IPv4AddressToString(accepted.sockaddr_in.ip));
        return;
    }
    auto connection = std::make_shared<Connection>();
    connection->peer = accepted.sockaddr_in.ip;
    connection->opened_at = std::chrono::steady_clock::now();
    connection->socket = std::move(accepted.socket);
    ConfigureStream(*connection->socket);
    stations.push_back(std::move(connection));
}

bool DirectLan::ReceiveStream(Connection& connection, std::vector<Network::LDNPacket>& packets) {
    std::array<u8, FrameSizeMax> buffer;
    const auto [size, error] = connection.socket->Recv(Network::FLAG_MSG_DONTWAIT, buffer);
    if (error == Network::Errno::AGAIN) {
        return true;
    }
    if (size <= 0) {
        return false;
    }
    connection.received.insert(connection.received.end(), buffer.begin(), buffer.begin() + size);

    Frame frame;
    while (true) {
        const auto result = TakeFrame(connection.received, frame);
        if (result != FrameResult::Complete) {
            return result == FrameResult::Incomplete;
        }
        Network::LDNPacket packet{
            .type = Network::LDNPacketType::SyncNetwork,
            .local_ip = connection.peer,
            .remote_ip = local_ip,
            .broadcast = false,
            .data = std::move(frame.data),
        };
        if (frame.type == FrameType::SyncNetwork && connection.to_host) {
            packets.push_back(std::move(packet));
            continue;
        }
        if (frame.type != FrameType::Connect || connection.to_host ||
            !connection.node_info.empty()) {
            continue;
        }
        // A station that reconnects replaces its old connection, which then closes silently
        std::scoped_lock lock{mutex};
        connection.node_info = packet.data;
        std::erase_if(stations, [&](const auto& station) {
            return station.get() != &connection && station->peer == connection.peer;
        });
        packet.type = Network::LDNPacketType::Connect;
        packets.push_back(std::move(packet));
    }
}

void DirectLan::OnClosed(const std::shared_ptr<Connection>& connection,
                         std::vector<Network::LDNPacket>& packets) {
    Network::LDNPacket packet{
        .type = Network::LDNPacketType::Disconnect,
        .local_ip = connection->peer,
        .remote_ip = local_ip,
        .broadcast = false,
        .data = connection->node_info,
    };
    std::scoped_lock lock{mutex};
    if (connection == host) {
        host.reset();
        packet.type = Network::LDNPacketType::DestroyNetwork;
        packets.push_back(std::move(packet));
        return;
    }
    const auto it = std::ranges::find(stations, connection);
    if (it == stations.end()) {
        return;
    }
    stations.erase(it);
    if (!packet.data.empty()) {
        packets.push_back(std::move(packet));
    }
}

bool DirectLan::IsOnSubnet(Network::IPv4Address ip) const {
    for (std::size_t i = 0; i < ip.size(); ++i) {
        if ((ip[i] & mask[i]) != (local_ip[i] & mask[i])) {
            return false;
        }
    }
    return true;
}

} // namespace Service::LDN
