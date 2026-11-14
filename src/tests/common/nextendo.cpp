// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "common/nextendo.h"

namespace Nextendo = Common::Nextendo;

TEST_CASE("Nextendo: PKCE S256 matches RFC 7636 appendix B", "[common]") {
    REQUIRE(Nextendo::PkceChallenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") ==
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");

    const auto pkce = Nextendo::MakePkce();
    REQUIRE(pkce.has_value());
    REQUIRE(pkce->verifier.size() == 43);
    REQUIRE(pkce->challenge == Nextendo::PkceChallenge(pkce->verifier));
}

TEST_CASE("Nextendo: base64url has no padding", "[common]") {
    const auto encode = [](std::string_view text) {
        return Nextendo::Base64UrlEncode(
            std::span{reinterpret_cast<const u8*>(text.data()), text.size()});
    };
    REQUIRE(encode("f") == "Zg");
    REQUIRE(encode("fo") == "Zm8");
    REQUIRE(encode("foo") == "Zm9v");
    REQUIRE(encode("\xfb\xff") == "-_8");
}

TEST_CASE("Nextendo: the authorize URL carries the loopback redirect and PKCE", "[common]") {
    const std::string url = Nextendo::BuildAuthorizeUrl("nxc_test", "st", "ch");
    REQUIRE(url.starts_with("https://nextendo.network/oauth/authorize?response_type=code"));
    REQUIRE(url.find("&client_id=nxc_test") != std::string::npos);
    REQUIRE(url.find("&redirect_uri=http%3A%2F%2F127.0.0.1%3A47823%2Fcallback") !=
            std::string::npos);
    REQUIRE(url.find("&scope=identity%20game.matchmaking") != std::string::npos);
    REQUIRE(url.find("&state=st&code_challenge=ch&code_challenge_method=S256") !=
            std::string::npos);
}

TEST_CASE("Nextendo: callback parsing", "[common]") {
    using Params = std::multimap<std::string, std::string>;

    const auto ok = Nextendo::ParseCallback(Params{{"code", "abc"}, {"state", "s1"}}, "s1");
    REQUIRE(ok.matched);
    REQUIRE(ok.error.empty());
    REQUIRE(ok.code == "abc");

    const auto wrong_state =
        Nextendo::ParseCallback(Params{{"code", "abc"}, {"state", "other"}}, "s1");
    REQUIRE_FALSE(wrong_state.matched);
    REQUIRE(wrong_state.code.empty());

    // A stray local request with an error but not this sign-in's state must not end it.
    const auto stray = Nextendo::ParseCallback(Params{{"error", "access_denied"}}, "s1");
    REQUIRE_FALSE(stray.matched);

    const auto denied = Nextendo::ParseCallback(
        Params{{"error", "access_denied"}, {"error_description", "no"}, {"state", "s1"}}, "s1");
    REQUIRE(denied.matched);
    REQUIRE(denied.error == "access_denied: no");

    const auto no_code = Nextendo::ParseCallback(Params{{"state", "s1"}}, "s1");
    REQUIRE(no_code.code.empty());
    REQUIRE_FALSE(no_code.error.empty());
}

TEST_CASE("Nextendo: token responses", "[common]") {
    const auto emulator = Nextendo::ParseTokenResponse(
        R"({"token_type":"Bearer","nex_token":"nx2.a.b",)"
        R"("account":{"pid":1800000001,"username":"P","friend_code":"SW-1"}})");
    REQUIRE(emulator.has_value());
    REQUIRE(emulator->nex_token == "nx2.a.b");
    REQUIRE(emulator->pid == 1800000001);
    REQUIRE(emulator->username == "P");
    REQUIRE(emulator->friend_code == "SW-1");

    const auto third_party = Nextendo::ParseTokenResponse(
        R"({"access_token":"at","refresh_token":"rt","expires_in":3600})");
    REQUIRE(third_party.has_value());
    REQUIRE(third_party->nex_token.empty());
    REQUIRE(third_party->access_token == "at");
    REQUIRE(third_party->refresh_token == "rt");
    REQUIRE(third_party->fields.size() == 3);

    const auto wrong_types = Nextendo::ParseTokenResponse(
        R"({"access_token":"at","refresh_token":null,"nex_token":7,)"
        R"("account":{"pid":"42","username":null,"friend_code":[]}})");
    REQUIRE(wrong_types.has_value());
    REQUIRE(wrong_types->access_token == "at");
    REQUIRE(wrong_types->refresh_token.empty());
    REQUIRE(wrong_types->nex_token.empty());
    REQUIRE(wrong_types->pid == 42);
    REQUIRE(wrong_types->username.empty());
    REQUIRE(wrong_types->friend_code.empty());

    const auto game = Nextendo::ParseTokenResponse(
        R"({"nex_token":"nx2.g.s","pid":1800000002,"friend_code":"SW-2","username":"Q",)"
        R"("expires_in":86400})");
    REQUIRE(game.has_value());
    REQUIRE(game->nex_token == "nx2.g.s");
    REQUIRE(game->pid == 1800000002);
    REQUIRE(game->friend_code == "SW-2");
    REQUIRE(game->username == "Q");
    REQUIRE(game->expires_in == 86400);

    REQUIRE_FALSE(Nextendo::ParseTokenResponse("not json").has_value());
}

TEST_CASE("Nextendo: only invalid_grant ends the session", "[common]") {
    REQUIRE(Nextendo::RefreshTokenRejected(R"({"error":"invalid_grant"})"));
    REQUIRE(Nextendo::RefreshTokenRejected(
        R"({"error":"invalid_grant","error_description":"refresh token expired"})"));
    REQUIRE_FALSE(Nextendo::RefreshTokenRejected(R"({"error":"invalid_client"})"));
    REQUIRE_FALSE(Nextendo::RefreshTokenRejected(R"({"error":null})"));
    REQUIRE_FALSE(Nextendo::RefreshTokenRejected("<html>Bad Gateway</html>"));
}

TEST_CASE("Nextendo: online status reply", "[common]") {
    const auto refused = Nextendo::ParseOnlineStatus(R"({"allow":false,"reason":"unverified"})");
    REQUIRE(refused.has_value());
    REQUIRE_FALSE(refused->allow);
    REQUIRE(refused->reason == "unverified");

    const auto allowed = Nextendo::ParseOnlineStatus(R"({"allow":true,"reason":""})");
    REQUIRE(allowed.has_value());
    REQUIRE(allowed->allow);

    REQUIRE_FALSE(Nextendo::ParseOnlineStatus(R"({"error":"Session invalide"})").has_value());
    REQUIRE_FALSE(Nextendo::ParseOnlineStatus("not json").has_value());
}
