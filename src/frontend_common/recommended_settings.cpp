// SPDX-License-Identifier: GPL-3.0-or-later

#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "frontend_common/config.h"
#include "frontend_common/recommended_settings.h"

namespace FrontendCommon {
namespace {

constexpr const char* BackupKey = "recommended_settings_backup";

// Calls func(game, section, ini) for every section of every per-game config and saves the configs
// for which it returned true.
template <typename Func>
void ForEachGameConfigSection(Func&& func) {
    const auto directory = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "custom";
    if (!Common::FS::IsDir(directory)) {
        return;
    }
    Common::FS::IterateDirEntries(
        directory,
        [&](const std::filesystem::directory_entry& entry) {
            const auto& path = entry.path();
            if (path.extension() != ".ini") {
                return true;
            }
            CSimpleIniA ini;
            ini.SetUnicode(true);
            ini.SetSpaces(false);
            if (ini.LoadData(
                    Common::FS::ReadStringFromFile(path, Common::FS::FileType::BinaryFile)) < 0) {
                LOG_ERROR(Config, "Could not read the game config {}",
                          Common::FS::PathToUTF8String(path));
                return true;
            }
            const std::string game = Common::FS::PathToUTF8String(path.stem());
            CSimpleIniA::TNamesDepend sections;
            ini.GetAllSections(sections);
            bool changed = false;
            for (const auto& section : sections) {
                changed |= func(game, section.pItem, ini);
            }
            std::string contents;
            if (changed && (ini.Save(contents) < 0 ||
                            Common::FS::WriteStringToFile(path, Common::FS::FileType::BinaryFile,
                                                          contents) != contents.size())) {
                LOG_ERROR(Config, "Could not write the game config {}",
                          Common::FS::PathToUTF8String(path));
            }
            return true;
        },
        Common::FS::DirEntryFilter::File);
}

// The game's own value of a setting in the way Config reads it, or nothing when the game uses the
// global value.
std::optional<std::string> GameValue(const CSimpleIniA& ini, const char* section,
                                     const Settings::BasicSetting& setting) {
    const std::string key = setting.GetLabel();
    if (ini.GetBoolValue(section, (key + "\\use_global").c_str(), true)) {
        return std::nullopt;
    }
    std::string value = ini.GetValue(section, key.c_str(), "");
    std::erase(value, '"');
    if (ini.GetBoolValue(section, (key + "\\default").c_str(), true) || value.empty()) {
        return setting.DefaultToString();
    }
    return value;
}

} // namespace

std::vector<Settings::RecommendedSettingChange> ApplyRecommendedSettings(
    Settings::Values& settings) {
    auto changes = Settings::ApplyRecommendedSettings(settings);
    ForEachGameConfigSection([&](const std::string& game, const char* section, CSimpleIniA& ini) {
        auto backup =
            Settings::ParseRecommendedSettingsBackup(ini.GetValue(section, BackupKey, ""));
        bool changed = false;
        Settings::ForEachRecommendedSetting(settings, [&](auto& setting, const auto&) {
            auto previous = GameValue(ini, section, setting);
            if (!previous) {
                return;
            }
            const std::string key = setting.GetLabel();
            ini.SetValue(section, (key + "\\use_global").c_str(), "true");
            ini.Delete(section, (key + "\\default").c_str());
            ini.Delete(section, key.c_str());
            backup.emplace(key, *previous);
            changed = true;
            if (*previous != setting.ToStringGlobal()) {
                changes.push_back({&setting, std::move(*previous), setting.ToStringGlobal(), game});
            }
        });
        if (changed) {
            ini.SetValue(section, BackupKey,
                         Settings::FormatRecommendedSettingsBackup(backup).c_str());
        }
        return changed;
    });
    return changes;
}

std::vector<Settings::RecommendedSettingChange> RestorePreviousSettings(
    Settings::Values& settings) {
    // The games first: until the global values are restored, a game that uses them runs with
    // the recommended ones.
    std::vector<Settings::RecommendedSettingChange> game_changes;
    ForEachGameConfigSection([&](const std::string& game, const char* section, CSimpleIniA& ini) {
        const char* const saved = ini.GetValue(section, BackupKey, nullptr);
        if (saved == nullptr) {
            return false;
        }
        const auto backup = Settings::ParseRecommendedSettingsBackup(saved);
        Settings::ForEachRecommendedSetting(settings, [&](auto& setting, const auto&) {
            const auto it = backup.find(setting.GetLabel());
            if (it == backup.end()) {
                return;
            }
            std::string previous =
                GameValue(ini, section, setting).value_or(setting.ToStringGlobal());
            const std::string key = setting.GetLabel();
            ini.SetValue(section, (key + "\\use_global").c_str(), "false");
            ini.SetValue(section, (key + "\\default").c_str(),
                         it->second == setting.DefaultToString() ? "true" : "false");
            ini.SetValue(section, key.c_str(), it->second.c_str());
            if (previous != it->second) {
                game_changes.push_back({&setting, std::move(previous), it->second, game});
            }
        });
        ini.Delete(section, BackupKey);
        return true;
    });
    auto changes = Settings::RestorePreviousSettings(settings);
    changes.insert(changes.end(), std::make_move_iterator(game_changes.begin()),
                   std::make_move_iterator(game_changes.end()));
    return changes;
}

} // namespace FrontendCommon
