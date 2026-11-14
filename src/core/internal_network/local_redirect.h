// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/socket_types.h"

/// Sends a title's connections to a server the user runs. Configured in Configure > Network and
/// read when a game starts; nothing changes while Settings::values.online_service_redirect is off:
///   online_redirect_hosts          these names and their subdomains resolve to 127.0.0.1 (and
///                                  are no longer blocked)
///   online_redirect_server         <ip>:<port>; TCP connections to 127.0.0.1:443 go there
///   online_redirect_loopback_only  other names do not resolve, and the guest's connections and
///                                  datagrams reach loopback only (and the server, through
///                                  127.0.0.1)
///   online_redirect_ca             a CA certificate (DER, at most 64 KiB) the guest's TLS trusts
///                                  too, through the system store and OpenSSL
/// The CA is trusted for every TLS connection the guest makes, not only those to the redirected
/// names: titles that verify with the system store's lists can't be told apart by host name.
/// Loopback only covers the guest's host sockets (BSD sockets, direct LAN play), not the traffic a
/// multiplayer room relays.
namespace Network::LocalRedirect {

/// Reads the settings (and the CA file) for the game that is starting, and logs the mapping when
/// the redirect is on.
void Configure();

/// Whether `host` is one of the redirected names or a subdomain of one (case and trailing dots
/// ignored).
[[nodiscard]] bool Matches(std::string_view host);

/// 127.0.0.1 for a redirected host, with the service's port (a number, "http" or "https").
[[nodiscard]] std::optional<std::vector<AddrInfo>> Resolve(
    std::string_view host, const std::optional<std::string>& service);

/// Where a TCP connection to `address` goes: the configured server for 127.0.0.1:443.
[[nodiscard]] SockAddrIn ConnectTarget(const SockAddrIn& address);

/// Whether the guest may reach `address` (always, unless loopback only is on). Checked before
/// ConnectTarget, so the configured server stays reachable.
[[nodiscard]] bool Allowed(const SockAddrIn& address);

[[nodiscard]] bool LoopbackOnly();

/// The extra CA certificate (DER), empty when unset or unreadable.
[[nodiscard]] std::vector<u8> TestCa();

} // namespace Network::LocalRedirect
