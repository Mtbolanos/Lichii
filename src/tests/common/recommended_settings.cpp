// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "common/recommended_settings.h"

namespace {

std::map<std::string, std::string> GlobalValues(const Settings::Values& settings) {
    std::map<std::string, std::string> values;
    for (const auto& [key, setting] : settings.linkage.by_key) {
        values.emplace(key, setting->ToStringGlobal());
    }
    return values;
}

// Gives every recommended setting another value than the recommended one and returns how many
// there are. The tests take the values from the table, so changing it does not break them.
std::size_t SetOtherPreferences(Settings::Values& settings) {
    std::size_t count = 0;
    Settings::ForEachRecommendedSetting(settings, [&](auto& setting, const auto& value) {
        using Type = std::remove_cvref_t<decltype(value)>;
        setting.SetValue(value == Type{} ? static_cast<Type>(1) : Type{});
        INFO(setting.GetLabel());
        REQUIRE(setting.GetValue() != value);
        ++count;
    });
    return count;
}

} // namespace

TEST_CASE("Recommended settings replace the global values and nothing else",
          "[recommended_settings]") {
    auto settings = std::make_unique<Settings::Values>();
    const std::size_t count = SetOtherPreferences(*settings);
    settings->resolution_setup.SetValue(Settings::ResolutionSetup::Res2X);
    settings->language_index.SetValue(Settings::Language::Dutch);
    const auto before = GlobalValues(*settings);

    const auto changes = Settings::ApplyRecommendedSettings(*settings);

    std::set<std::string> recommended;
    Settings::ForEachRecommendedSetting(*settings, [&](auto& setting, const auto& value) {
        INFO(setting.GetLabel());
        CHECK(setting.GetValue() == value);
        CHECK(recommended.insert(setting.GetLabel()).second);
    });
    REQUIRE(changes.size() == count);
    for (const auto& change : changes) {
        INFO(change.setting->GetLabel());
        CHECK(recommended.contains(change.setting->GetLabel()));
        CHECK(change.previous == before.at(change.setting->GetLabel()));
        CHECK(change.current == change.setting->ToStringGlobal());
        CHECK(change.previous != change.current);
    }
    for (const auto& [key, value] : GlobalValues(*settings)) {
        if (!recommended.contains(key) && key != settings->recommended_settings_backup.GetLabel()) {
            INFO(key);
            CHECK(value == before.at(key));
        }
    }
}

TEST_CASE("Previous settings come back after the recommended ones", "[recommended_settings]") {
    auto settings = std::make_unique<Settings::Values>();
    const std::size_t count = SetOtherPreferences(*settings);
    const auto before = GlobalValues(*settings);

    CHECK(Settings::ApplyRecommendedSettings(*settings).size() == count);
    const std::string backup = settings->recommended_settings_backup.GetValue();
    CHECK_FALSE(backup.empty());

    CHECK(Settings::ApplyRecommendedSettings(*settings).empty());
    CHECK(settings->recommended_settings_backup.GetValue() == backup);

    // Applying again after a change by hand keeps the first backup, so restoring still goes back
    // to the values from before the first time.
    bool first = true;
    Settings::ForEachRecommendedSetting(*settings, [&](auto& setting, const auto& value) {
        if (std::exchange(first, false)) {
            using Type = std::remove_cvref_t<decltype(value)>;
            setting.SetValue(value == Type{} ? static_cast<Type>(1) : Type{});
        }
    });
    CHECK(Settings::ApplyRecommendedSettings(*settings).size() == 1);
    CHECK(settings->recommended_settings_backup.GetValue() == backup);

    CHECK(Settings::RestorePreviousSettings(*settings).size() == count);
    CHECK(GlobalValues(*settings) == before);

    CHECK(Settings::RestorePreviousSettings(*settings).empty());
    CHECK(GlobalValues(*settings) == before);
}

TEST_CASE("A backup from a shorter table takes the settings it does not hold",
          "[recommended_settings]") {
    auto settings = std::make_unique<Settings::Values>();
    const std::size_t count = SetOtherPreferences(*settings);
    const auto before = GlobalValues(*settings);
    // As a build whose table held only the first setting leaves it after applying.
    Settings::ForEachRecommendedSetting(*settings, [&](auto& setting, const auto& value) {
        if (settings->recommended_settings_backup.GetValue().empty()) {
            settings->recommended_settings_backup =
                setting.GetLabel() + ":" + setting.ToStringGlobal();
            setting.SetValue(value);
        }
    });
    REQUIRE(count > 1);

    CHECK(Settings::ApplyRecommendedSettings(*settings).size() == count - 1);
    CHECK(Settings::RestorePreviousSettings(*settings).size() == count);
    CHECK(GlobalValues(*settings) == before);
}

TEST_CASE("Recommended settings leave per-game values alone", "[recommended_settings]") {
    auto settings = std::make_unique<Settings::Values>();
    const std::size_t count = SetOtherPreferences(*settings);
    const auto before = GlobalValues(*settings);
    const auto use_per_game_values = [&] {
        Settings::ForEachRecommendedSetting(*settings, [](auto& setting, const auto&) {
            setting.SetGlobal(false);
            setting.SetValue(setting.GetValue(true));
        });
    };
    const auto check_per_game_values = [&] {
        Settings::ForEachRecommendedSetting(*settings, [&](auto& setting, const auto&) {
            INFO(setting.GetLabel());
            CHECK_FALSE(setting.UsingGlobal());
            CHECK(setting.ToString() == before.at(setting.GetLabel()));
        });
    };
    use_per_game_values();

    CHECK(Settings::ApplyRecommendedSettings(*settings).size() == count);
    check_per_game_values();
    Settings::ForEachRecommendedSetting(*settings, [](auto& setting, const auto& value) {
        INFO(setting.GetLabel());
        CHECK(setting.GetValue(true) == value);
    });

    CHECK(Settings::RestorePreviousSettings(*settings).size() == count);
    check_per_game_values();
    CHECK(GlobalValues(*settings) == before);
}

TEST_CASE("A backup only restores recommended settings", "[recommended_settings]") {
    auto settings = std::make_unique<Settings::Values>();
    const auto before = GlobalValues(*settings);
    std::string label;
    std::string other;
    Settings::ForEachRecommendedSetting(*settings, [&](auto& setting, const auto& value) {
        if (!label.empty()) {
            return;
        }
        label = setting.GetLabel();
        using Type = std::remove_cvref_t<decltype(value)>;
        setting.SetValue(setting.GetDefault() == Type{} ? static_cast<Type>(1) : Type{});
        other = setting.ToStringGlobal();
        setting.SetValue(setting.GetDefault());
    });
    settings->recommended_settings_backup =
        "resolution_setup:0,unknown:7,broken," + label + ":" + other + ",speed_limit:,";

    const auto restored = Settings::RestorePreviousSettings(*settings);
    REQUIRE(restored.size() == 1);
    CHECK(restored[0].setting->GetLabel() == label);
    CHECK(restored[0].current == other);
    CHECK(settings->resolution_setup.GetValue() == settings->resolution_setup.GetDefault());
    CHECK(settings->recommended_settings_backup.GetValue().empty());
    auto after = GlobalValues(*settings);
    after[label] = before.at(label);
    CHECK(after == before);
}
