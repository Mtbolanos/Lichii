// SPDX-License-Identifier: GPL-3.0-or-later

#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"
#include "frontend_common/config.h"

namespace {

constexpr u64 OcarinaTitleId = 0x0400486024DEC000;
constexpr const char* DynamicFps = "Dynamic FPS by Dark (built-in)";
constexpr const char* MaxResolution = "Max Resolution (built-in)";

class AddonConfig final : public Config {
public:
    AddonConfig() {
        config = std::make_unique<CSimpleIniA>();
    }

    using Config::ReadDisabledAddOnValues;
    using Config::SaveDisabledAddOnValues;

    std::string Serialize() const {
        std::string data;
        REQUIRE(config->Save(data) >= 0);
        return data;
    }

    void Load(const std::string& data) {
        config = std::make_unique<CSimpleIniA>();
        REQUIRE(config->LoadData(data) >= 0);
        ReadDisabledAddOnValues();
    }

private:
    void ReloadAllValues() override {}
    void SaveAllValues() override {}
    void ReadHidbusValues() override {}
    void ReadDebugControlValues() override {}
    void ReadPathValues() override {}
    void ReadShortcutValues() override {}
    void ReadUIValues() override {}
    void ReadUIGamelistValues() override {}
    void ReadUILayoutValues() override {}
    void ReadMultiplayerValues() override {}
    void SaveHidbusValues() override {}
    void SaveDebugControlValues() override {}
    void SavePathValues() override {}
    void SaveShortcutValues() override {}
    void SaveUIValues() override {}
    void SaveUIGamelistValues() override {}
    void SaveUILayoutValues() override {}
    void SaveMultiplayerValues() override {}
    std::vector<Settings::BasicSetting*>& FindRelevantList(Settings::Category category) override {
        return Settings::values.linkage.by_category[category];
    }
};

struct AddonSettingsScope {
    AddonSettingsScope() : disabled{std::exchange(Settings::values.disabled_addons, {})} {}

    ~AddonSettingsScope() {
        Settings::values.disabled_addons = std::move(disabled);
    }

    decltype(Settings::values.disabled_addons) disabled;
};

} // namespace

TEST_CASE("Built-in Add-On disabling survives repeated config saves and loads", "[addon_config]") {
    AddonSettingsScope scope;
    AddonConfig config;
    Settings::values.disabled_addons[OcarinaTitleId] = {MaxResolution, DynamicFps};

    for (int i = 0; i < 3; ++i) {
        config.SaveDisabledAddOnValues();
        const auto saved = config.Serialize();
        Settings::values.disabled_addons.clear();
        config.Load(saved);
        REQUIRE(Settings::values.disabled_addons.at(OcarinaTitleId) ==
                std::vector<std::string>{MaxResolution, DynamicFps});
    }

    Settings::values.disabled_addons[OcarinaTitleId] = {DynamicFps};
    config.SaveDisabledAddOnValues();
    const auto disabled = config.Serialize();
    Settings::values.disabled_addons.clear();
    config.Load(disabled);
    REQUIRE(Settings::values.disabled_addons.at(OcarinaTitleId) ==
            std::vector<std::string>{DynamicFps});
    for (int i = 0; i < 3; ++i) {
        Settings::values.disabled_addons[OcarinaTitleId].clear();
        config.SaveDisabledAddOnValues();
        const auto enabled = config.Serialize();
        Settings::values.disabled_addons[OcarinaTitleId] = {MaxResolution, DynamicFps};
        config.Load(enabled);
        REQUIRE(Settings::values.disabled_addons.at(OcarinaTitleId).empty());
    }
}

TEST_CASE("Legacy Add-On configs retain default-on built-ins", "[addon_config]") {
    AddonSettingsScope scope;
    AddonConfig config;
    config.Load("[DisabledAddOns]\nsize=0\n");
    REQUIRE(Settings::values.disabled_addons.empty());
    config.SaveDisabledAddOnValues();
    config.Load(config.Serialize());
    REQUIRE(Settings::values.disabled_addons.empty());
}
