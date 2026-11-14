// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include "common/fs/path_util.h"
#include "common/recommended_settings.h"
#include "common/settings.h"
#include "frontend_common/config.h"
#include "frontend_common/recommended_settings.h"

namespace {

class FileConfig final : public Config {
public:
    explicit FileConfig(const std::string& name, ConfigType config_type = ConfigType::GlobalConfig)
        : Config{config_type} {
        Initialize(name);
    }

    explicit FileConfig(std::optional<std::string> path) {
        Initialize(path);
    }

    using Config::Reload;
    using Config::SaveValues;

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

class ConfigDirectory {
public:
    ConfigDirectory()
        : original{Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir)},
          path{std::filesystem::temp_directory_path() /
               fmt::format("lichii-config-{}", std::random_device{}())} {
        REQUIRE(std::filesystem::create_directory(path));
        Common::FS::SetEdenPath(Common::FS::EdenPath::ConfigDir, path);
        for (const auto& [key, setting] : Settings::values.linkage.by_key) {
            setting->SetGlobal(false);
            setting->LoadString("");
            setting->SetGlobal(true);
            setting->LoadString("");
        }
    }

    ~ConfigDirectory() {
        Common::FS::SetEdenPath(Common::FS::EdenPath::ConfigDir, original);
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    void Write(const std::string& name, const std::string& contents) const {
        std::filesystem::create_directories((path / name).parent_path());
        std::ofstream file{path / name};
        file << contents;
        REQUIRE(file.good());
    }

    const std::filesystem::path original;
    const std::filesystem::path path;
};

void CheckFreshProfile() {
    const auto& values = Settings::values;
    CHECK(values.scaling_filter.GetValue() == Settings::ScalingFilter::Fsr3);
    CHECK(values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK(values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu);
    CHECK(values.async_presentation.GetValue());
    CHECK(values.use_asynchronous_shaders.GetValue());
}

void CheckLegacyProfile() {
    const auto& values = Settings::values;
    CHECK(values.scaling_filter.GetValue() == Settings::ScalingFilter::ScaleForce);
    CHECK(values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::High);
    CHECK(values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Cpu);
    CHECK_FALSE(values.async_presentation.GetValue());
    CHECK_FALSE(values.use_asynchronous_shaders.GetValue());
}

void CheckUnchangedProfile() {
    const auto& values = Settings::values;
    CHECK(values.gpu_clock.GetValue() == Settings::GpuClock::Boost);
    CHECK(values.cpu_clock.GetValue() == Settings::CpuClock::Normal);
    CHECK(values.use_multi_core.GetValue());
    CHECK(values.use_speed_limit.GetValue());
    CHECK(values.speed_limit.GetValue() == 100);
    CHECK(values.slow_speed_limit.GetValue() == 50);
    CHECK(values.turbo_speed_limit.GetValue() == 200);
    CHECK_FALSE(values.sync_core_speed.GetValue());
    CHECK(values.memory_layout_mode.GetValue() == Settings::MemoryLayout::Memory_4Gb);
    CHECK(values.vsync_mode.GetValue() == Settings::VSyncMode::Fifo);
    CHECK(values.fsr_sharpening_slider.GetValue() == 25);
    CHECK(values.dma_accuracy.GetValue() == Settings::DmaAccuracy::Default);
    CHECK(values.gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Default);
    CHECK(values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Conservative);
    CHECK(values.max_anisotropy.GetValue() == Settings::AnisotropyMode::Automatic);
    CHECK(values.accelerate_astc.GetValue() == Settings::AstcDecodeMode::Gpu);
    CHECK(values.frame_pacing_mode.GetValue() == Settings::FramePacingMode::Target_Auto);
    CHECK(values.astc_recompression.GetValue() == Settings::AstcRecompression::Bc3);
    CHECK_FALSE(values.sync_memory_operations.GetValue());
    CHECK_FALSE(values.renderer_force_max_clock.GetValue());
    CHECK(values.use_disk_shader_cache.GetValue());
    CHECK(values.use_vulkan_driver_pipeline_cache.GetValue());
    CHECK_FALSE(values.use_video_framerate.GetValue());
    CHECK_FALSE(values.mkw_30fps.GetValue());
    CHECK_FALSE(values.mkw_60fps.GetValue());
    CHECK(values.mkw_60fps_fallback.GetValue());
    CHECK_FALSE(values.mkw_dynamic_fps.GetValue());
    CHECK_FALSE(values.hyrule_dynamic_fps.GetValue());
    CHECK_FALSE(values.hyrule_30fps.GetValue());
    CHECK(values.use_reactive_flushing.GetValue());
    CHECK(values.barrier_feedback_loops.GetValue());
    CHECK_FALSE(values.enable_buffer_history.GetValue());
    CHECK_FALSE(values.skip_cpu_inner_invalidation.GetValue());
    CHECK_FALSE(values.fix_bloom_effects.GetValue());
    CHECK(values.gpu_unswizzle_texture_size.GetValue() == Settings::GpuUnswizzleSize::Large);
    CHECK(values.gpu_unswizzle_stream_size.GetValue() == Settings::GpuUnswizzle::Medium);
    CHECK(values.gpu_unswizzle_chunk_size.GetValue() == Settings::GpuUnswizzleChunk::Medium);
    CHECK_FALSE(values.gpu_unswizzle_enabled.GetValue());
    CHECK(values.dyna_state.GetValue() == values.dyna_state.GetDefault());
    CHECK(values.sample_shading.GetValue() == 0);
    CHECK(values.vertex_input_dynamic_state.GetValue());
}

} // namespace

TEST_CASE("Absent global config receives the first-run profile and persists it", "[config]") {
    ConfigDirectory directory;
    std::unique_ptr<FileConfig> config;
    SECTION("Qt config name") {
        config = std::make_unique<FileConfig>(std::string{"qt-config"});
    }
    SECTION("SDL default path") {
        config = std::make_unique<FileConfig>(std::optional<std::string>{});
    }
    SECTION("SDL custom path") {
        config = std::make_unique<FileConfig>(
            std::make_optional(Common::FS::PathToUTF8String(directory.path / "explicit.ini")));
    }
    SECTION("SDL UTF-8 config path") {
        config = std::make_unique<FileConfig>(
            std::make_optional(Common::FS::PathToUTF8String(directory.path / u8"新用户-é.ini")));
    }
    CheckFreshProfile();
    CheckUnchangedProfile();
    CSimpleIniA saved;
    std::ifstream file{std::filesystem::path{Common::FS::ToU8String(config->GetConfigFilePath())}};
    REQUIRE(file.is_open());
    const std::string contents{std::istreambuf_iterator<char>{file},
                               std::istreambuf_iterator<char>{}};
    REQUIRE(saved.LoadData(contents) >= 0);
    file.close();
    for (const char* key : {"scaling_filter", "gpu_accuracy", "nvdec_emulation",
                            "async_presentation", "use_asynchronous_shaders"}) {
        CHECK_FALSE(saved.GetBoolValue("Renderer", (std::string{key} + "\\default").c_str(), true));
    }
    const auto filename = config->GetConfigFilePath();
    config.reset();
    FileConfig reopened{std::make_optional(filename)};
    CheckFreshProfile();
    reopened.Reload();
    CheckFreshProfile();
}

TEST_CASE("Existing explicit settings survive load save and reopen", "[config]") {
    ConfigDirectory directory;
    directory.Write(
        "qt-config.ini",
        fmt::format("[Renderer]\nscaling_filter={}\nscaling_filter\\default=false\n"
                    "gpu_accuracy={}\ngpu_accuracy\\default=false\n"
                    "nvdec_emulation={}\nnvdec_emulation\\default=false\n"
                    "async_presentation=false\nasync_presentation\\default=false\n"
                    "use_asynchronous_shaders=false\nuse_asynchronous_shaders\\default=false\n"
                    "fsr_sharpening_slider=83\nfsr_sharpening_slider\\default=false\n"
                    "[System]\nfast_gpu_time={}\nfast_gpu_time\\default=false\n",
                    u32(Settings::ScalingFilter::Bilinear), u32(Settings::GpuAccuracy::High),
                    u32(Settings::NvdecEmulation::Off), u32(Settings::GpuClock::Normal)));
    for (int i = 0; i < 3; ++i) {
        FileConfig config{std::string{"qt-config"}};
        CHECK(Settings::values.scaling_filter.GetValue() == Settings::ScalingFilter::Bilinear);
        CHECK(Settings::values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::High);
        CHECK(Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Off);
        CHECK_FALSE(Settings::values.async_presentation.GetValue());
        CHECK_FALSE(Settings::values.use_asynchronous_shaders.GetValue());
        CHECK(Settings::values.fsr_sharpening_slider.GetValue() == 83);
        CHECK(Settings::values.gpu_clock.GetValue() == Settings::GpuClock::Normal);
    }
}

TEST_CASE("Existing default markers and missing keys keep their historical values", "[config]") {
    ConfigDirectory directory;
    std::string contents;
    SECTION("Empty existing config") {}
    SECTION("Missing renderer keys") {
        contents = "[System]\nfast_gpu_time\\default=true\n";
    }
    SECTION("Stored values without default markers") {
        contents = "[Renderer]\nasync_presentation=true\nuse_asynchronous_shaders=true\n";
    }
    SECTION("Malformed existing config") {
        contents = "[incomplete section\nnot a setting\n";
    }
    SECTION("Default markers override conflicting stored values") {
        contents =
            fmt::format("[Renderer]\nscaling_filter={}\nscaling_filter\\default=true\n"
                        "gpu_accuracy={}\ngpu_accuracy\\default=true\n"
                        "nvdec_emulation={}\nnvdec_emulation\\default=true\n"
                        "async_presentation=true\nasync_presentation\\default=true\n"
                        "use_asynchronous_shaders=true\nuse_asynchronous_shaders\\default=true\n",
                        u32(Settings::ScalingFilter::Fsr3), u32(Settings::GpuAccuracy::Low),
                        u32(Settings::NvdecEmulation::Gpu));
    }
    directory.Write("qt-config.ini", contents);
    for (int i = 0; i < 3; ++i) {
        FileConfig config{std::string{"qt-config"}};
        CheckLegacyProfile();
    }
}

TEST_CASE("Resetting a seeded setting does not reapply the first-run profile", "[config]") {
    ConfigDirectory directory;
    {
        FileConfig config{std::string{"qt-config"}};
        CheckFreshProfile();
        Settings::values.scaling_filter.LoadString("");
        Settings::values.gpu_accuracy.LoadString("");
        Settings::values.nvdec_emulation.LoadString("");
        Settings::values.async_presentation.LoadString("");
        Settings::values.use_asynchronous_shaders.LoadString("");
        config.SaveValues();
        config.Reload();
        CheckLegacyProfile();
    }
    FileConfig reopened{std::string{"qt-config"}};
    CheckLegacyProfile();
}

TEST_CASE("Per-game overrides and use-global flags survive the new global profile", "[config]") {
    ConfigDirectory directory;
    FileConfig global{std::string{"qt-config"}};
    directory.Write(
        "custom/game.ini",
        fmt::format(
            "[Renderer]\nscaling_filter={}\nscaling_filter\\default=false\n"
            "scaling_filter\\use_global=false\n"
            "gpu_accuracy={}\ngpu_accuracy\\default=true\ngpu_accuracy\\use_global=false\n"
            "nvdec_emulation={}\nnvdec_emulation\\default=false\nnvdec_emulation\\use_global=true\n"
            "async_presentation=false\nasync_presentation\\default=false\n"
            "async_presentation\\use_global=false\n",
            u32(Settings::ScalingFilter::Bilinear), u32(Settings::GpuAccuracy::Low),
            u32(Settings::NvdecEmulation::Off)));
    for (int i = 0; i < 3; ++i) {
        FileConfig game{std::string{"game"}, Config::ConfigType::PerGameConfig};
        CHECK_FALSE(Settings::values.scaling_filter.UsingGlobal());
        CHECK(Settings::values.scaling_filter.GetValue() == Settings::ScalingFilter::Bilinear);
        CHECK(Settings::values.scaling_filter.GetValue(true) == Settings::ScalingFilter::Fsr3);
        CHECK_FALSE(Settings::values.gpu_accuracy.UsingGlobal());
        CHECK(Settings::values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::High);
        CHECK(Settings::values.gpu_accuracy.GetValue(true) == Settings::GpuAccuracy::Low);
        CHECK(Settings::values.nvdec_emulation.UsingGlobal());
        CHECK(Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu);
        CHECK_FALSE(Settings::values.async_presentation.UsingGlobal());
        CHECK_FALSE(Settings::values.async_presentation.GetValue());
        CHECK(Settings::values.async_presentation.GetValue(true));
        CHECK(Settings::values.use_asynchronous_shaders.UsingGlobal());
        CHECK(Settings::values.use_asynchronous_shaders.GetValue());
    }
}

TEST_CASE("Missing per-game and input profiles are not seeded", "[config]") {
    ConfigDirectory directory;
    FileConfig global{std::string{"qt-config"}};
    FileConfig game{std::string{"game"}, Config::ConfigType::PerGameConfig};
    CheckFreshProfile();
    CHECK(Settings::values.scaling_filter.UsingGlobal());
    CHECK_FALSE(game.Exists("Renderer", "scaling_filter\\default"));
    FileConfig input{std::string{"controller"}, Config::ConfigType::InputProfile};
    CHECK_FALSE(input.Exists("Renderer", "scaling_filter"));
    CheckFreshProfile();
}

namespace {

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream file{path};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

/// Temporary files a save left in the directory (qt-config.ini.tmp.<pid>.<n>).
size_t SaveTemporaries(const std::filesystem::path& directory) {
    return static_cast<size_t>(std::ranges::count_if(
        std::filesystem::directory_iterator{directory}, [](const auto& entry) {
            return entry.path().filename().string().starts_with("qt-config.ini.tmp.");
        }));
}

constexpr const char* SharpeningConfig =
    "[Renderer]\nfsr_sharpening_slider=83\nfsr_sharpening_slider\\default=false\n";

} // namespace

TEST_CASE("A config save replaces the file and ignores a leftover temporary file", "[config]") {
    ConfigDirectory directory;
    directory.Write("qt-config.ini", SharpeningConfig);
    directory.Write("qt-config.ini.tmp", "left by a save that was killed");
    FileConfig config{std::string{"qt-config"}};
    config.SaveValues();
    CHECK(ReadText(directory.path / "qt-config.ini").find("fsr_sharpening_slider=83") !=
          std::string::npos);
    CHECK(ReadText(directory.path / "qt-config.ini.tmp") == "left by a save that was killed");
    CHECK(SaveTemporaries(directory.path) == 0);
}

#ifndef _WIN32
TEST_CASE("A config save writes through a symlink and keeps it", "[config]") {
    ConfigDirectory directory;
    directory.Write("target.ini", SharpeningConfig);
    std::filesystem::create_symlink("target.ini", directory.path / "qt-config.ini");
    FileConfig config{std::string{"qt-config"}};
    config.SaveValues();
    CHECK(std::filesystem::is_symlink(directory.path / "qt-config.ini"));
    CHECK(ReadText(directory.path / "target.ini").find("fsr_sharpening_slider=83") !=
          std::string::npos);
}

TEST_CASE("A config save keeps a read-only config", "[config]") {
    ConfigDirectory directory;
    const auto filename = directory.path / "qt-config.ini";
    directory.Write("qt-config.ini", SharpeningConfig);
    std::filesystem::permissions(filename, std::filesystem::perms::owner_read);
    if (std::ofstream{filename, std::ios::app}.is_open()) {
        std::filesystem::permissions(filename, std::filesystem::perms::owner_read |
                                                   std::filesystem::perms::owner_write);
        SKIP("This user can write files despite restrictive permissions");
    }
    FileConfig config{std::string{"qt-config"}};
    Settings::values.fsr_sharpening_slider.SetValue(20);
    config.SaveValues();
    std::filesystem::permissions(filename, std::filesystem::perms::owner_read |
                                               std::filesystem::perms::owner_write);
    CHECK(ReadText(filename) == SharpeningConfig);
    CHECK(SaveTemporaries(directory.path) == 0);
}

TEST_CASE("A dangling config symlink is not a new user", "[config]") {
    ConfigDirectory directory;
    std::filesystem::create_symlink(directory.path / "target.ini",
                                    directory.path / "qt-config.ini");
    FileConfig config{std::string{"qt-config"}};
    CheckLegacyProfile();
}

TEST_CASE("An unreadable config is not seeded or overwritten", "[config]") {
    ConfigDirectory directory;
    const auto filename = directory.path / "qt-config.ini";
    const std::string contents = "[Renderer]\nasync_presentation=false\n";
    directory.Write("qt-config.ini", contents);
    std::filesystem::permissions(filename, std::filesystem::perms::none);
    if (std::ifstream{filename}.is_open()) {
        std::filesystem::permissions(filename, std::filesystem::perms::owner_read |
                                                   std::filesystem::perms::owner_write);
        SKIP("This user can read files despite restrictive permissions");
    }
    FileConfig config{std::string{"qt-config"}};
    CheckLegacyProfile();
    std::filesystem::permissions(filename, std::filesystem::perms::owner_read |
                                               std::filesystem::perms::owner_write);
    std::ifstream file{filename};
    const std::string saved{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    CHECK(saved == contents);
}

TEST_CASE("A denied config status lookup is not seeded", "[config]") {
    ConfigDirectory directory;
    const auto inaccessible = directory.path / "inaccessible";
    std::filesystem::create_directory(inaccessible);
    std::filesystem::permissions(inaccessible, std::filesystem::perms::none);
    const auto filename = inaccessible / "config.ini";
    std::error_code error;
    void(std::filesystem::symlink_status(filename, error));
    if (error != std::errc::permission_denied) {
        std::filesystem::permissions(inaccessible, std::filesystem::perms::owner_all);
        SKIP("This user can query paths despite restrictive permissions");
    }
    FileConfig config{std::make_optional(Common::FS::PathToUTF8String(filename))};
    CheckLegacyProfile();
    std::filesystem::permissions(inaccessible, std::filesystem::perms::owner_all);
    CHECK_FALSE(std::filesystem::exists(filename));
}
#endif

TEST_CASE("The recommended settings backup persists in the config", "[config]") {
    ConfigDirectory directory;
    SECTION("New install") {}
    SECTION("Config from an older build") {
        directory.Write("qt-config.ini",
                        "[Renderer]\nasync_presentation=false\nasync_presentation\\default=false\n");
    }
    SECTION("Config from a build that offered the recommended settings at startup") {
        directory.Write("qt-config.ini", "[Miscellaneous]\nrecommended_settings_prompt_shown=true\n"
                                         "recommended_settings_prompt_shown\\default=false\n");
    }
    std::string backup;
    std::size_t changed = 0;
    {
        FileConfig config{std::string{"qt-config"}};
        CHECK(Settings::values.recommended_settings_backup.GetValue().empty());
        Settings::values.speed_limit.SetValue(150);
        changed = Settings::ApplyRecommendedSettings(Settings::values).size();
        CHECK(changed > 0);
        backup = Settings::values.recommended_settings_backup.GetValue();
        config.SaveValues();
    }
    Settings::values.recommended_settings_backup = std::string{};
    {
        FileConfig config{std::string{"qt-config"}};
        CHECK(Settings::values.recommended_settings_backup.GetValue() == backup);
        CHECK(Settings::ApplyRecommendedSettings(Settings::values).empty());
        CHECK(Settings::RestorePreviousSettings(Settings::values).size() == changed);
        CHECK(Settings::values.speed_limit.GetValue() == 150);
        config.SaveValues();
    }
    Settings::values.speed_limit.SetValue(100);
    FileConfig config{std::string{"qt-config"}};
    CHECK(Settings::values.recommended_settings_backup.GetValue().empty());
    CHECK(Settings::values.speed_limit.GetValue() == 150);
}

namespace {

// Gives the game its own value, other than the recommended one, for every recommended setting
// and for the resolution, which is not one.
void WriteGameOverrides(const std::string& game) {
    FileConfig config{game, Config::ConfigType::PerGameConfig};
    Settings::ForEachRecommendedSetting(Settings::values, [](auto& setting, const auto& value) {
        using Type = std::remove_cvref_t<decltype(value)>;
        setting.SetGlobal(false);
        setting.SetValue(value == Type{} ? static_cast<Type>(1) : Type{});
    });
    Settings::values.resolution_setup.SetGlobal(false);
    Settings::values.resolution_setup.SetValue(Settings::ResolutionSetup::Res2X);
    config.SaveValues();
    Settings::RestoreGlobalState(false);
}

// Boots the game's config and returns the value it uses for every per-game setting, "global"
// where it uses the global value.
std::map<std::string, std::string> GameValues(const std::string& game) {
    std::map<std::string, std::string> values;
    {
        FileConfig config{game, Config::ConfigType::PerGameConfig};
        for (const auto& [key, setting] : Settings::values.linkage.by_key) {
            if (setting->Switchable()) {
                values.emplace(key, setting->UsingGlobal() ? "global" : setting->ToString());
            }
        }
    }
    Settings::RestoreGlobalState(false);
    return values;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file{path};
    std::stringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

} // namespace

TEST_CASE("Recommended settings reset the per-game values of recommended settings", "[config]") {
    ConfigDirectory directory;
    const std::string game = "0100152000022000";
    // The global values already match, so only the game changes.
    Settings::ApplyRecommendedSettings(Settings::values);
    Settings::values.recommended_settings_backup = std::string{};
    WriteGameOverrides(game);
    directory.Write("custom/0400C3F00006E000.ini",
                    "[Renderer]\nresolution_setup\\use_global=true\n");
    const auto before = GameValues(game);
    const std::string other_game = ReadFile(directory.path / "custom" / "0400C3F00006E000.ini");

    const auto changes = FrontendCommon::ApplyRecommendedSettings(Settings::values);
    REQUIRE_FALSE(changes.empty());
    CHECK(std::ranges::all_of(changes, [&](const auto& change) { return change.game == game; }));
    // Help > Restore My Previous Settings shows while the backup holds something.
    CHECK_FALSE(Settings::values.recommended_settings_backup.GetValue().empty());
    auto after = GameValues(game);
    Settings::ForEachRecommendedSetting(Settings::values, [&](auto& setting, const auto&) {
        INFO(setting.GetLabel());
        CHECK(after.at(setting.GetLabel()) == "global");
        after[setting.GetLabel()] = before.at(setting.GetLabel());
    });
    CHECK(after == before);
    CHECK(before.at("resolution_setup") != "global");
    CHECK(ReadFile(directory.path / "custom" / "0400C3F00006E000.ini") == other_game);

    const std::string applied = ReadFile(directory.path / "custom" / (game + ".ini"));
    CHECK(FrontendCommon::ApplyRecommendedSettings(Settings::values).empty());
    CHECK(ReadFile(directory.path / "custom" / (game + ".ini")) == applied);

    CHECK(FrontendCommon::RestorePreviousSettings(Settings::values).size() == changes.size());
    CHECK(GameValues(game) == before);
    CHECK(Settings::values.recommended_settings_backup.GetValue().empty());
    CHECK(FrontendCommon::RestorePreviousSettings(Settings::values).empty());
    CHECK(GameValues(game) == before);
}

TEST_CASE("A config path beneath a non-directory is not seeded", "[config]") {
    ConfigDirectory directory;
    directory.Write("file", "not a directory");
    FileConfig config{
        std::make_optional(Common::FS::PathToUTF8String(directory.path / "file" / "config.ini"))};
    CheckLegacyProfile();
}

TEST_CASE("DLSS old profiles default to M and preserve native quality values",
          "[config][dlss_mode]") {
    ConfigDirectory directory;
    directory.Write("qt-config.ini", "[Renderer]\nnvidia_dlss=true\nnvidia_dlss\\default=false\n");
    FileConfig config{std::string{"qt-config"}};
    CHECK(Settings::values.nvidia_dlss.GetValue());
    CHECK(Settings::values.nvidia_dlss_preset.GetValue() == Settings::NvidiaDlssPreset::M);
    CHECK(Settings::GetNvidiaDlssRuntimeSettings().preset == Settings::NvidiaDlssPreset::M);
    CHECK(Settings::GetNvidiaDlssRuntimeSettings().quality ==
          Settings::NvidiaDlssQuality::Automatic);
    directory.Write("qt-config.ini",
                    "[Renderer]\nnvidia_dlss_quality=1\nnvidia_dlss_quality\\default=false\n"
                    "nvidia_dlss_preset=0\nnvidia_dlss_preset\\default=false\n");
    FileConfig native_config{std::string{"qt-config"}};
    CHECK(Settings::GetNvidiaDlssRuntimeSettings().quality == Settings::NvidiaDlssQuality::Native);
    CHECK(Settings::GetNvidiaDlssRuntimeSettings().preset == Settings::NvidiaDlssPreset::M);
}

TEST_CASE("DLSS config round-trips choices and publishes per-game overrides",
          "[config][dlss_mode]") {
    ConfigDirectory directory;
    FileConfig config{std::string{"qt-config"}};
    for (const auto& [name, value] :
         Settings::EnumMetadata<Settings::NvidiaDlssPreset>::Canonicalizations()) {
        Settings::values.nvidia_dlss_preset.SetValue(value);
        config.SaveValues();
        Settings::values.nvidia_dlss_preset.LoadString("");
        FileConfig loaded{std::string{"qt-config"}};
        CHECK(Settings::values.nvidia_dlss_preset.GetValue() == value);
        CHECK(Settings::GetNvidiaDlssRuntimeSettings().preset == value);
    }
    for (const auto& [name, value] :
         Settings::EnumMetadata<Settings::NvidiaDlssQuality>::Canonicalizations()) {
        Settings::values.nvidia_dlss_quality.SetValue(value);
        config.SaveValues();
        Settings::values.nvidia_dlss_quality.LoadString("");
        FileConfig loaded{std::string{"qt-config"}};
        CHECK(Settings::values.nvidia_dlss_quality.GetValue() == value);
        CHECK(Settings::GetNvidiaDlssRuntimeSettings().quality == value);
    }
    Settings::values.nvidia_dlss_preset = Settings::NvidiaDlssPreset::K;
    config.SaveValues();
    directory.Write("custom/game.ini",
                    "[Renderer]\nnvidia_dlss_preset=0\nnvidia_dlss_preset\\default=false\n"
                    "nvidia_dlss_preset\\use_global=false\n");
    FileConfig game{std::string{"game"}, Config::ConfigType::PerGameConfig};
    CHECK_FALSE(Settings::values.nvidia_dlss_preset.UsingGlobal());
    CHECK(Settings::GetNvidiaDlssRuntimeSettings().preset == Settings::NvidiaDlssPreset::M);
    Settings::values.nvidia_dlss_preset.SetGlobal(true);
    Settings::RefreshNvidiaDlssRuntimeSettings();
    CHECK(Settings::GetNvidiaDlssRuntimeSettings().preset == Settings::NvidiaDlssPreset::K);
}
