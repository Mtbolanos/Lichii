// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>

#include "common/logging.h"
#include "common/nextendo.h"
#include "common/settings.h"
#include "core/hle/service/sockets/nextendo_redirect.h"

namespace Service::Sockets::Nextendo {

namespace {

bool IsDomainOrSubdomain(std::string_view host, std::string_view domain) {
    return host == domain ||
           (host.size() > domain.size() && host.ends_with(domain) &&
            host[host.size() - domain.size() - 1] == '.');
}

// "*" matches any run of characters.
bool MatchesPattern(std::string_view host, std::string_view pattern) {
    const auto star = pattern.find('*');
    if (star == std::string_view::npos) {
        return host == pattern;
    }
    const auto prefix = pattern.substr(0, star);
    const auto suffix = pattern.substr(star + 1);
    return host.size() >= prefix.size() + suffix.size() && host.starts_with(prefix) &&
           host.ends_with(suffix);
}

struct TitleHosts {
    std::array<u64, 4> titles;
    std::array<std::string_view, 5> hosts;
};

// Publisher services Nextendo stands in for, each only while its title runs, because other titles
// share the same hosts with their real owners (citron-nextendo sfdnsres.cpp GetNextendoRedirectIp,
// 5746dcff3, e75213fb3, 0b6d9d2ec, 2ea8993ac, 786662555, e9d371aeb, b5b4ce264).
constexpr std::array TitleRoutes{
    // Plants vs. Zombies: Battle for Neighborville (EA Blaze)
    TitleHosts{{0x0100C56010FD8000}, {"spring18.gosredirector.ea.com", "pvzgw3prodapp-*.ea.com"}},
    // Dragon Ball FighterZ
    TitleHosts{{0x0100A250097F0000}, {"dbf.channel.or.jp"}},
    // Friday the 13th: The Game
    TitleHosts{{0x010092A00C4B6000}, {"nswitch.f13.illfonic.com"}},
    // Sonic Racing CrossWorlds
    TitleHosts{{0x01006E001823C000},
               {"api.epicgames.dev", "prod-api.sora-online.net", "prod-cdn.sora-online.net",
                "turn.rtcp.on.epicgames.com"}},
    // Guilty Gear -Strive-
    TitleHosts{{0x0100AB401F136000, 0x01002F4020718000, 0x01002F402071A000, 0x010034A020A5C000},
               {"api.epicgames.dev", "switch-ggst-game.guiltygear.com",
                "prod-switch-match-ggst-game.guiltygear.com",
                "prod-to3match-ggst-game.guiltygear.com"}},
    // Just Dance 2022
    TitleHosts{{0x0100EA6014BB8000},
               {"public-ubiservices.ubi.com", "public-ws-ubiservices.ubi.com", "jd.ubisoft.com",
                "phonescoring.jd.ubisoft.com", "v2.phonescoring.jd.ubisoft.com"}},
    // UNO
    TitleHosts{{0x0100B6E012EBE000}, {"public-ubiservices.ubi.com"}},
    // SpeedRunners 2
    TitleHosts{{0x01004CC023582000},
               {"api.braincloudservers.com", "ns.photonengine.io", "api.gameanalytics.com"}},
    // Crypt of the NecroDancer
    TitleHosts{{0x0100CEA007D08000}, {"api.braincloudservers.com"}},
    // World War Z
    TitleHosts{{0x010099F013898000}, {"*.wwzapi.com", "*.hydrapi.net"}},
    // Minecraft Dungeons II (PlayFab and Xbox Live sign-in)
    TitleHosts{{0x0100A7C01B792000},
               {"83156.playfabapi.com", "*vex.minecraftservices.com", "xboxlive.com",
                "*.xboxlive.com", "login.live.com"}},
};

bool IsTitleHost(std::string_view host, u64 program_id) {
    if (program_id == 0) {
        return false;
    }
    for (const auto& route : TitleRoutes) {
        if (std::ranges::find(route.titles, program_id) == route.titles.end()) {
            continue;
        }
        for (const auto pattern : route.hosts) {
            if (!pattern.empty() && MatchesPattern(host, pattern)) {
                return true;
            }
        }
    }
    return false;
}

std::mutex host_mutex;
std::unordered_map<std::string, std::string> host_for_address;

} // namespace

std::optional<Target> Classify(std::string_view host, u64 program_id) {
    // Ryujinx-Nextendo's DnsMitmResolver and Prelude's hosts list put the NAT-check host and
    // Mario Kart 8 Deluxe's NEX server on the second address; everything else is on the first.
    if (host.starts_with("nncs2-") && host.ends_with(".n.n.srv.nintendo.net")) {
        return Target::Nat;
    }
    if (host == "g2b309e01-lp1.s.n.srv.nintendo.net" ||
        host == "g2df33d01-lp1.p.srv.nintendo.net") {
        return Target::Nat;
    }
    // Activision titles reach Demonware instead of Nintendo's services.
    for (const std::string_view domain :
         {"nintendo.net", "nintendo.com", "nintendowifi.net", "nintendo.co.jp", "demonware.net"}) {
        if (IsDomainOrSubdomain(host, domain)) {
            return Target::Server;
        }
    }
    if (IsTitleHost(host, program_id)) {
        return Target::Server;
    }
    return std::nullopt;
}

std::optional<std::string> RedirectAddress(std::string_view host, u64 program_id) {
    if (!Common::Nextendo::OnlinePlayActive()) {
        return std::nullopt;
    }
    const auto target = Classify(host, program_id);
    if (!target) {
        return std::nullopt;
    }
    std::string address = *target == Target::Nat ? Settings::values.nextendo_nat_ip.GetValue()
                                                 : Settings::values.nextendo_server_ip.GetValue();
    if (address.empty()) {
        return std::nullopt;
    }
    LOG_INFO(Service, "Nextendo: resolving {} to {}", host, address);
    return address;
}

std::string ExpandNsdHost(std::string host) {
    if (const auto pos = host.find('%'); pos != std::string::npos) {
        host.replace(pos, 1, "lp1");
    }
    return host;
}

void RememberHost(const std::string& address, const std::string& host) {
    std::scoped_lock lock{host_mutex};
    host_for_address[address] = host;
}

std::string HostForAddress(const std::string& address) {
    std::scoped_lock lock{host_mutex};
    const auto it = host_for_address.find(address);
    return it == host_for_address.end() ? std::string{} : it->second;
}

} // namespace Service::Sockets::Nextendo
