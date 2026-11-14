// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>

#include "common/hex_util.h"
#include "common/logging.h"
#include "common/uuid.h"
#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/hle/service/acc/nextendo_id_token.h"
#include "core/hle/service/hle_ipc.h"

namespace Service::Account::Nextendo {

namespace {

constexpr const char* BaasIssuer = "https://e0d67c509fb203858ebcb2fe3f88c2aa.baas.nintendo.com";
constexpr const char* BaasJku =
    "https://e0d67c509fb203858ebcb2fe3f88c2aa.baas.nintendo.com/1.0.0/certificates";
constexpr const char* BaasAudience = "ed9e2f05d286f7b8";
constexpr const char* BaasKeyId = "nextendo-baas-key-1";

std::string RandomHex(std::size_t bytes) {
    std::vector<u8> buffer(bytes);
    RAND_bytes(buffer.data(), static_cast<int>(buffer.size()));
    return Common::HexToString(buffer, false);
}

std::string Base64Url(std::string_view text) {
    return Common::Nextendo::Base64UrlEncode(
        std::span{reinterpret_cast<const u8*>(text.data()), text.size()});
}

// Nextendo's NEX servers don't check this signature, so a key made for the session will do.
// Titles that verify it against Nextendo's JWKS (Splatoon 2's membership check) need Nextendo's
// own key, which Lichii doesn't have.
std::string SignRs256(std::string_view input) {
    static EVP_PKEY* const key = EVP_RSA_gen(2048);
    if (key == nullptr) {
        return {};
    }
    EVP_MD_CTX* const ctx = EVP_MD_CTX_new();
    std::string signature;
    std::size_t length = 0;
    const auto* data = reinterpret_cast<const unsigned char*>(input.data());
    if (ctx != nullptr && EVP_DigestSignInit(ctx, nullptr, EVP_sha256(), nullptr, key) == 1 &&
        EVP_DigestSign(ctx, nullptr, &length, data, input.size()) == 1) {
        std::vector<u8> raw(length);
        if (EVP_DigestSign(ctx, raw.data(), &length, data, input.size()) == 1) {
            raw.resize(length);
            signature = Common::Nextendo::Base64UrlEncode(raw);
        }
    }
    EVP_MD_CTX_free(ctx);
    return signature;
}

std::string InstalledVersion(Core::System& system, u64 program_id) {
    if (program_id == 0) {
        return {};
    }
    const FileSys::PatchManager pm{program_id, system.GetFileSystemController(),
                                   system.GetContentProvider()};
    const auto metadata = pm.GetControlMetadata();
    return metadata.first != nullptr ? metadata.first->GetVersionString() : std::string{};
}

// Titles whose Nextendo servers are shared between several games and need the program id.
bool NeedsAppId(u64 program_id) {
    switch (program_id) {
    case 0x0100D870045B6000ULL: // NES - Nintendo Switch Online
    case 0x01008D300C50C000ULL: // SNES - Nintendo Switch Online
    case 0x0100C62011050000ULL: // Game Boy - Nintendo Switch Online
    case 0x010012F017576000ULL: // Game Boy Advance - Nintendo Switch Online
    case 0x0100C9A00ECE6000ULL: // Nintendo 64 - Nintendo Switch Online
    case 0x0100B3C014BDA000ULL: // SEGA Genesis - Nintendo Switch Online
    case 0x0100A3D008C5C000ULL: // Pokemon Scarlet
    case 0x0100559011740000ULL: // Monster Hunter Rise (Japan/Hong Kong)
        return true;
    default:
        return false;
    }
}

} // namespace

std::optional<Common::Nextendo::Account> LinkedAccount() {
    if (!Common::Nextendo::OnlinePlayActive()) {
        return std::nullopt;
    }
    return Common::Nextendo::LoadAccount();
}

std::string BuildIdToken(Core::System& system, const Common::Nextendo::Account& account) {
    static const std::string device_id = RandomHex(0x10);
    const u64 program_id = system.GetApplicationProcessProgramID();
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

    const nlohmann::ordered_json header{
        {"alg", "RS256"}, {"kid", BaasKeyId}, {"typ", "id_token"}, {"jku", BaasJku}};
    nlohmann::ordered_json payload{
        {"sub", RandomHex(0x10)},
        {"aud", BaasAudience},
        {"iss", BaasIssuer},
        {"typ", "id_token"},
        {"iat", now},
        {"exp", now + 3 * 60 * 60},
        {"jku", BaasJku},
        {"jti", Common::UUID::MakeRandom().FormattedString()},
        {"di", device_id},
        {"sn", "XAW10000000000"},
        {"bs:did", RandomHex(0x10)},
        // nnAccount rejects a token without the device claims (Ryujinx-Nextendo, citron-nextendo).
        {"nintendo",
         {{"dt", "NX Prod 1"}, {"pc", "HAC"}, {"di", device_id}, {"sn", "XAW10000000000"},
          {"ist", false}}},
    };
    if (!account.nex_token.empty()) {
        payload["nnex"] = account.nex_token;
    }
    if (const std::string version = InstalledVersion(system, program_id); !version.empty()) {
        payload["tv"] = version;
    }
    if (NeedsAppId(program_id)) {
        payload["app_id"] = fmt::format("{:016X}", program_id);
    }
    payload["hm"] = true;

    const std::string input = Base64Url(header.dump()) + "." + Base64Url(payload.dump());
    const std::string signature = SignRs256(input);
    if (signature.empty()) {
        LOG_ERROR(Service_ACC, "Could not sign the Nextendo id token");
    }
    if (account.nex_token.empty()) {
        LOG_WARNING(Service_ACC, "Nextendo: no nex_token for this account, the game server "
                                 "will refuse the login");
    }
    return input + "." + signature;
}

std::size_t WriteIdToken(HLERequestContext& ctx, std::string_view token) {
    const std::size_t capacity = ctx.GetWriteBufferSize();
    if (capacity == 0) {
        return 0;
    }
    if (token.size() >= capacity) {
        LOG_ERROR(Service_ACC, "Nextendo id token ({} bytes) does not fit the {}-byte buffer",
                  token.size(), capacity);
    }
    const std::size_t written = (std::min)(token.size(), capacity - 1);
    std::vector<u8> out(capacity, 0);
    std::memcpy(out.data(), token.data(), written);
    ctx.WriteBuffer(out);
    return written;
}

} // namespace Service::Account::Nextendo
