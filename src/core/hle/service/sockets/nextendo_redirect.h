// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "common/common_types.h"

// Sends Nintendo's online host names to Nextendo Network while the Nextendo setting is on,
// after citron-nextendo's GetNextendoRedirectIp (src/core/hle/service/sockets/sfdnsres.cpp).
namespace Service::Sockets::Nextendo {

enum class Target {
    Server,
    // The second NAT-check vantage point, which also serves Mario Kart 8 Deluxe.
    Nat,
};

// program_id selects the per-title publisher hosts (Epic, Ubisoft, brainCloud and others) that
// Nextendo serves only for the title that uses them.
std::optional<Target> Classify(std::string_view host, u64 program_id = 0);

// The address to resolve instead of host, or nullopt when host keeps its normal resolution.
std::optional<std::string> RedirectAddress(std::string_view host, u64 program_id = 0);

// NSD names carry "%" for the environment; retail consoles resolve it to "lp1".
std::string ExpandNsdHost(std::string host);

// Remembers which redirected name an address came from, so a TLS connection the guest opens by
// address still sends that name as SNI: Nextendo routes its auth servers on :443 by SNI.
void RememberHost(const std::string& address, const std::string& host);
std::string HostForAddress(const std::string& address);

} // namespace Service::Sockets::Nextendo
