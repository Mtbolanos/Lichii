// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <QString>
#include <QStringList>

class QIcon;
class QMainWindow;
class QPushButton;
class QWidget;

namespace Ui {
class MainWindow;
}

// The Fritchii look ("Graphite"): dark, one violet accent, Inter, a home screen for Ocarina of
// Time and plain menus. Everything here is switched on only for Fritchii builds, so a Lichii build
// keeps the upstream UI and merges from Lichii stay clean. Shared files only call in here.
namespace FritchiiUi {

/// Fritchii builds (g_build_name "Fritchii"). FRITCHII_CLASSIC_UI=1 shows the upstream UI.
[[nodiscard]] bool Enabled();

/// Fusion style, the Graphite palette, Inter and the stylesheet. Does nothing (and returns false)
/// when the Fritchii UI is off.
bool ApplyTheme();

/// Settings the Fritchii settings pages leave out (frame generation and its dev options).
[[nodiscard]] bool HidesSetting(const std::string& label);

/// A bundled line icon (fritchii/icons/<name>.svg): grey, and violet when checked.
[[nodiscard]] QIcon Icon(const QString& name, int px = 22);
/// The same icon in one fixed colour (e.g. the dark play glyph on the violet button).
[[nodiscard]] QIcon Glyph(const QString& name, unsigned rgb, int px);

struct MenuHooks {
    std::function<void()> export_log;
    std::function<void()> whats_new;
    std::function<void()> delete_shader_cache; ///< OoT's, after a confirmation
};
/// Plain names and Fritchii's grouping for the classic menus (File, Game, View, Tools, Help).
void RestructureMenus(Ui::MainWindow& ui, QMainWindow& window, const MenuHooks& hooks);

struct GraphicsHooks {
    std::function<void()> refresh;          ///< after a change: update the status bar state
    std::function<bool()> renderer_locked;  ///< the renderer can't change while a game runs
    std::function<void()> open_settings;
    std::function<void()> open_effects;     ///< empty without ReShade-style effects
};
/// The status bar's one "Graphics" button: renderer, accuracy, filter, anti-aliasing, effects.
[[nodiscard]] QPushButton* MakeGraphicsButton(QWidget* parent, GraphicsHooks hooks);

/// A settings page as the settings dialog shows it: inside a scroll area in Fritchii builds, so
/// the window keeps a normal size with the larger type; the page itself otherwise.
[[nodiscard]] QWidget* SettingsPage(QWidget* page);

/// Loading screen bar in the Graphite colours.
[[nodiscard]] const char* LoadingBarStyle();

/// The Fritchii Discord server and the owner's donation page, opened in the browser.
inline constexpr const char* DiscordInvite = "https://discord.gg/fritchii";
inline constexpr const char* DonateLink = "https://paypal.me/frizti";
void OpenDiscord();
void OpenDonate();
/// Help > Copy Data from Eden or Lichii (v5.8.1): asks, then restarts into the import.
void CopyDataFromOldFolder(QMainWindow& window);

/// Settings backups in config/backups/<yyyy-MM-dd HHmmss>: qt-config.ini, frizti.ini and the
/// per-game configs (custom/). Copies the current ones and keeps the newest 10; returns the
/// folder, empty when nothing could be saved.
QString BackupSettings();
/// The backups, newest first: {folder, label like "9 Nov 2026, 21:14"}.
[[nodiscard]] std::vector<std::pair<QString, QString>> SettingsBackups();
/// Puts a backup's files in place of the current ones (custom/ replaced as a whole).
bool RestoreSettingsFiles(const QString& backup);
/// The backup the settings dialog asked to restore; the main window reads and clears it.
[[nodiscard]] QString& PendingSettingsRestore();

/// Ocarina of Time's game-specific settings (Configure Game): the setting names whose value
/// beats the global one, read from config/custom/0400486024DEC000.ini.
[[nodiscard]] QStringList OcarinaGameOverrides();
/// Puts all of them back on the global settings (use_global=true in that file).
bool ClearOcarinaGameOverrides();

} // namespace FritchiiUi
