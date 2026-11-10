// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>
#include <QCoreApplication>
#include "common/recommended_settings.h"

class QWidget;

/// The message boxes around the developer's recommended settings (common/recommended_settings.h).
class RecommendedSettingsDialog {
    Q_DECLARE_TR_FUNCTIONS(RecommendedSettingsDialog)

public:
    /// Asks whether to use the recommended settings. True when the user accepts.
    static bool Ask(QWidget* parent);

    /// Says what ApplyRecommendedSettings changed and where to undo it.
    static void ShowApplied(QWidget* parent,
                            const std::vector<Settings::RecommendedSettingChange>& changes);

    /// Says what RestorePreviousSettings changed back.
    static void ShowRestored(QWidget* parent,
                             const std::vector<Settings::RecommendedSettingChange>& changes);
};
