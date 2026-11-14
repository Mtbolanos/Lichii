// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "common/nextendo.h"

namespace Core {
class System;
}

namespace Service {
class HLERequestContext;
}

// The account answers a title gets while Nextendo Network is on, after citron-nextendo's
// src/core/hle/service/acc/acc.cpp (BuildIdToken, GetIdTokenBytes, WriteIdToken).
namespace Service::Account::Nextendo {

// The signed-in Nextendo account, while the Nextendo setting is on.
std::optional<Common::Nextendo::Account> LinkedAccount();

// The BAAS id_token a title forwards in its NEX login. Nextendo's game servers read the account's
// nx2 token from its "nnex" claim and the installed title version from "tv".
std::string BuildIdToken(Core::System& system, const Common::Nextendo::Account& account);

// Titles read the token as a C string, so the rest of their buffer is zeroed. Returns the token
// bytes written.
std::size_t WriteIdToken(HLERequestContext& ctx, std::string_view token);

} // namespace Service::Account::Nextendo
