// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "common/common_types.h"
#include "common/settings.h"

// The settings the Frizti builds tune for The Legend of Zelda: Ocarina of Time, and how they are
// applied over a config shared with other Eden/Lichii builds (%APPDATA%\eden or ./user).
namespace Settings::Frizti {

// Bump when the tuned values change, so every player gets them once more.
inline constexpr u32 Revision = 6; // 6: Fritchii v5.6 (no FXAA on top of the game's AA)
inline constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;

// The tuned settings. Frizti's value is each setting's default in settings.h.
std::vector<BasicSetting*> TunedSettings(Values& values);

// What Frizti last wrote for the tuned settings (config/frizti.ini, which only Frizti writes).
struct Record {
    u32 revision = 0;
    std::map<std::string, std::string> global;   ///< label -> value written to qt-config.ini
    std::map<std::string, std::string> per_game; ///< label -> "global" or Ocarina's override
};

std::string SerializeRecord(const Record& record);
Record ParseRecord(std::string_view text);

struct Change {
    std::string label;
    std::string from;
    std::string to;
};

// `settings` hold global values. With `full`, sets each to Frizti's value; otherwise restores
// the value Frizti last wrote to those another build (or a hand edit) changed since.
std::vector<Change> ReconcileGlobal(std::span<BasicSetting* const> settings, const Record& record,
                                    bool full);

// `settings` hold Ocarina's per-game config. Resets to global the overrides Frizti did not write
// (all of them with `full`).
std::vector<Change> ReconcilePerGame(std::span<BasicSetting* const> settings, const Record& record,
                                     bool full);

// Docked mode can't use the handheld controller: player 1 becomes a Pro Controller, as the
// docked toggle does.
std::vector<Change> FixDockedController(Values& values);

std::map<std::string, std::string> GlobalValues(std::span<BasicSetting* const> settings);
std::map<std::string, std::string> PerGameValues(std::span<BasicSetting* const> settings);

// "label: from -> to, ..."
std::string FormatChanges(std::span<const Change> changes);

} // namespace Settings::Frizti
