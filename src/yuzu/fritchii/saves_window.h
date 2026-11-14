// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

class QWidget;

namespace FritchiiUi {

/// The Saves window (Fritchii v5.8.3): export the saves to a .zip, import them from one, or copy
/// them between Fritchii, Eden (%APPDATA%\eden) and Lichii 150+ (%APPDATA%\lichii). Saves that
/// get replaced are backed up first. `emulation_running` blocks it while a game runs.
void ShowSavesWindow(QWidget* parent, const std::function<bool()>& emulation_running);

} // namespace FritchiiUi
