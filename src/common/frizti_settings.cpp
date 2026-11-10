// SPDX-License-Identifier: GPL-3.0-or-later

#include <fmt/format.h>
#include "common/frizti_settings.h"

namespace Settings::Frizti {

namespace {
constexpr std::string_view RevisionKey = "revision";
constexpr std::string_view GlobalPrefix = "global.";
constexpr std::string_view PerGamePrefix = "game.";
constexpr std::string_view UseGlobal = "global";

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}
} // namespace

std::vector<BasicSetting*> TunedSettings(Values& values) {
    // One line per setting; bump Revision when this list or a default here changes.
    return {
        &values.gpu_accuracy,             // Low
        &values.dma_accuracy,             // Default
        &values.gpu_fence_behavior,       // Default
        &values.vsync_mode,               // Mailbox: the host frame pacer times presents
        &values.async_presentation,       // On: the host frame pacer runs on the present thread
        &values.use_asynchronous_shaders, // On: unbuilt pipelines skip draws, no freeze
        &values.use_docked_mode,          // Docked
        &values.memory_layout_mode,       // 8 GB: Ocarina's own memory size
        &values.nvdec_emulation,          // GPU: Ocarina's cutscene videos
        &values.scaling_filter,           // ScaleForce (newer Lichii builds seed FSR)
        &values.resolution_setup,         // 1x: a higher scale left over from other games costs fps
        &values.anti_aliasing,            // None: OoT's own AA (or DLSS) already smooths edges
        &values.cpu_accuracy,             // Auto
        &values.use_disk_shader_cache,    // On: no shader rebuild every boot
        &values.use_vulkan_driver_pipeline_cache, // On
        // EDS2 and dynamic vertex input: the precompiled OoT recipe is recorded with them, and
        // players who turned them off (an old AMD crash tip; the crash was shader libraries) had
        // every graphics pipeline skipped as "for another device" (2026-11-09, RX 9070 XT log).
        &values.dyna_state,                 // EDS2
        &values.vertex_input_dynamic_state, // On
    };
}

std::string SerializeRecord(const Record& record) {
    std::string text = "# Written by the Frizti builds: the tuned settings as they last saved "
                       "them. Other builds share qt-config.ini but not this file.\n";
    text += fmt::format("{}={}\n", RevisionKey, record.revision);
    for (const auto& [label, value] : record.global) {
        text += fmt::format("{}{}={}\n", GlobalPrefix, label, value);
    }
    for (const auto& [label, value] : record.per_game) {
        text += fmt::format("{}{}={}\n", PerGamePrefix, label, value);
    }
    return text;
}

Record ParseRecord(std::string_view text) {
    Record record;
    while (!text.empty()) {
        const size_t end = text.find('\n');
        const std::string_view line = Trim(text.substr(0, end));
        text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
        const size_t equals = line.find('=');
        if (line.empty() || line.front() == '#' || equals == std::string_view::npos) {
            continue;
        }
        const std::string_view key = Trim(line.substr(0, equals));
        const std::string value{Trim(line.substr(equals + 1))};
        if (key == RevisionKey) {
            try {
                record.revision = u32(std::stoul(value));
            } catch (...) {
                record.revision = 0;
            }
        } else if (key.starts_with(GlobalPrefix)) {
            record.global.insert_or_assign(std::string{key.substr(GlobalPrefix.size())}, value);
        } else if (key.starts_with(PerGamePrefix)) {
            record.per_game.insert_or_assign(std::string{key.substr(PerGamePrefix.size())}, value);
        }
    }
    return record;
}

std::vector<Change> ReconcileGlobal(std::span<BasicSetting* const> settings, const Record& record,
                                    bool full) {
    std::vector<Change> changes;
    for (BasicSetting* const setting : settings) {
        std::string target;
        if (full) {
            target = setting->DefaultToString();
        } else if (const auto it = record.global.find(setting->GetLabel());
                   it != record.global.end()) {
            target = it->second;
        } else {
            continue;
        }
        if (setting->ToStringGlobal() == target) {
            continue;
        }
        const std::string from = setting->Canonicalize();
        setting->LoadString(target);
        changes.push_back({setting->GetLabel(), from, setting->Canonicalize()});
    }
    return changes;
}

std::vector<Change> ReconcilePerGame(std::span<BasicSetting* const> settings, const Record& record,
                                     bool full) {
    std::vector<Change> changes;
    for (BasicSetting* const setting : settings) {
        if (setting->UsingGlobal()) {
            continue;
        }
        if (!full) {
            // A missing entry means Frizti never wrote an override for it.
            const auto it = record.per_game.find(setting->GetLabel());
            if (it != record.per_game.end() && it->second == setting->ToString()) {
                continue;
            }
        }
        const std::string from = setting->Canonicalize();
        setting->SetGlobal(true);
        changes.push_back({fmt::format("{:016X} {}", OcarinaOfTimeProgramId, setting->GetLabel()),
                           from, std::string{UseGlobal}});
    }
    return changes;
}

std::vector<Change> FixDockedController(Values& values) {
    auto& player = values.players.GetValue(true)[0];
    if (values.use_docked_mode.GetValue(true) != ConsoleMode::Docked ||
        player.controller_type != ControllerType::Handheld) {
        return {};
    }
    player.controller_type = ControllerType::ProController;
    return {{"player_0_type", "Handheld", "ProController"}};
}

std::map<std::string, std::string> GlobalValues(std::span<BasicSetting* const> settings) {
    std::map<std::string, std::string> values;
    for (const BasicSetting* const setting : settings) {
        values.emplace(setting->GetLabel(), setting->ToStringGlobal());
    }
    return values;
}

std::map<std::string, std::string> PerGameValues(std::span<BasicSetting* const> settings) {
    std::map<std::string, std::string> values;
    for (const BasicSetting* const setting : settings) {
        values.emplace(setting->GetLabel(),
                       setting->UsingGlobal() ? std::string{UseGlobal} : setting->ToString());
    }
    return values;
}

std::string FormatChanges(std::span<const Change> changes) {
    std::string text;
    for (const Change& change : changes) {
        if (!text.empty()) {
            text += ", ";
        }
        text += fmt::format("{}: {} -> {}", change.label, change.from, change.to);
    }
    return text;
}

} // namespace Settings::Frizti
