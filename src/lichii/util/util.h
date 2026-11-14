// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2015 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <QFont>
#include <QIcon>
#include <QString>
#include "common/uuid.h"

/// Returns a QFont object appropriate to use as a monospace font for debugging widgets, etc.
[[nodiscard]] QFont GetMonospaceFont();

/// A menu icon from the Lichii icon set, white on the highlighted row. Other themes keep menus
/// without icons, so this returns a null icon unless Lichii Dark is active.
[[nodiscard]] QIcon MenuIcon(const char* name);

/**
 * Prompt the user for a profile ID. If there is only one valid profile, returns that profile.
 * @return The selected profile, or an std::nullopt if none were selected
 */
const std::optional<Common::UUID> GetProfileID();

/**
 * Prompt the user for a profile ID. If there is only one valid profile, returns that profile.
 * @return A string representation of the selected profile, or an empty string if none were seleeced
 */
std::string GetProfileIDString();
