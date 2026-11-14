// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <map>
#include <utility>
#include "common/recommended_settings.h"

namespace Settings {

std::map<std::string, std::string> ParseRecommendedSettingsBackup(const std::string& backup) {
    std::map<std::string, std::string> saved;
    for (std::size_t begin = 0; begin < backup.size();) {
        const std::size_t end = (std::min)(backup.find(',', begin), backup.size());
        const std::size_t colon = backup.find(':', begin);
        if (colon < end && colon + 1 < end) {
            saved.emplace(backup.substr(begin, colon - begin),
                          backup.substr(colon + 1, end - colon - 1));
        }
        begin = end + 1;
    }
    return saved;
}

std::string FormatRecommendedSettingsBackup(const std::map<std::string, std::string>& entries) {
    std::string backup;
    for (const auto& [label, value] : entries) {
        if (!backup.empty()) {
            backup += ',';
        }
        backup.append(label).append(":").append(value);
    }
    return backup;
}

std::vector<RecommendedSettingChange> ApplyRecommendedSettings(Values& settings) {
    const auto saved =
        ParseRecommendedSettingsBackup(settings.recommended_settings_backup.GetValue());
    std::vector<RecommendedSettingChange> changes;
    std::map<std::string, std::string> backup;
    ForEachRecommendedSetting(settings, [&](auto& setting, const auto& value) {
        std::string previous = setting.ToStringGlobal();
        // A value backed up earlier stays; a setting the backup does not hold yet, such as one a
        // later build added to the table, joins it.
        const auto it = saved.find(setting.GetLabel());
        backup.emplace(setting.GetLabel(), it == saved.end() ? previous : it->second);
        if (setting.GetValue(true) == value) {
            return;
        }
        const bool using_global = setting.UsingGlobal();
        setting.SetGlobal(true);
        setting.SetValue(value);
        setting.SetGlobal(using_global);
        changes.push_back({&setting, std::move(previous), setting.ToStringGlobal()});
    });
    settings.recommended_settings_backup = FormatRecommendedSettingsBackup(backup);
    return changes;
}

std::vector<RecommendedSettingChange> RestorePreviousSettings(Values& settings) {
    const auto saved =
        ParseRecommendedSettingsBackup(settings.recommended_settings_backup.GetValue());
    std::vector<RecommendedSettingChange> changes;
    ForEachRecommendedSetting(settings, [&](auto& setting, const auto&) {
        const auto it = saved.find(setting.GetLabel());
        if (it == saved.end()) {
            return;
        }
        std::string previous = setting.ToStringGlobal();
        const bool using_global = setting.UsingGlobal();
        setting.SetGlobal(true);
        setting.LoadString(it->second);
        setting.SetGlobal(using_global);
        if (setting.ToStringGlobal() != previous) {
            changes.push_back({&setting, std::move(previous), setting.ToStringGlobal()});
        }
    });
    settings.recommended_settings_backup = std::string{};
    return changes;
}

} // namespace Settings
