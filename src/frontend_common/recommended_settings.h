// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>
#include "common/recommended_settings.h"

namespace FrontendCommon {

/// Applies the recommended settings to the global values (Settings::ApplyRecommendedSettings)
/// and resets every per-game config in user/config/custom that overrides one of them to the
/// global value, so the recommended value takes effect in every game. Other per-game settings
/// stay. Each section of a game's config keeps the overrides it reset under
/// recommended_settings_backup; a value backed up earlier stays until it is restored. Returns the
/// values that changed, the global ones first.
std::vector<Settings::RecommendedSettingChange> ApplyRecommendedSettings(
    Settings::Values& settings);

/// Puts back the per-game overrides ApplyRecommendedSettings reset and the global values
/// (Settings::RestorePreviousSettings), clears the backups and returns the values that changed,
/// the global ones first.
std::vector<Settings::RecommendedSettingChange> RestorePreviousSettings(Settings::Values& settings);

} // namespace FrontendCommon
