// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cwctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#ifndef SI_NO_CONVERSION
#define SI_NO_CONVERSION 1
#endif
#include <SimpleIni.h>
#include "common/fs/path_util.h"
#include "common/fs/profile_migration.h"
#include "frontend_common/profile_import.h"

namespace FrontendCommon {
namespace fs = std::filesystem;
namespace Migration = Common::FS::ProfileMigration;

fs::path ProfileImportStateDirectory() {
    const auto& root = Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir);
    return root.parent_path() / (root.filename().string() + ".migration");
}

bool NeedsLegacyProfileImport() {
#ifdef __ANDROID__
    return false;
#else
    using namespace Common::FS;
    if (IsPortableMode()) {
        return false;
    }
    const auto state = Migration::GetState(ProfileImportStateDirectory());
    if (state == Migration::State::Pending) {
        return true;
    }
    if (state != Migration::State::Unresolved) {
        return false;
    }
    constexpr std::array sources{LegacyEdenDir, CitronDir, SudachiDir, SuyuDir, YuzuDir};
    for (const auto source : sources) {
        if (fs::is_directory(GetLegacyPath(source))) {
            return true;
        }
    }
    return false;
#endif
}

void CheckLegacyProfileBeforeConfig() {
    if (NeedsLegacyProfileImport()) {
        throw std::runtime_error("Legacy emulator data or an unfinished import was found. "
                                 "Open the Lichii GUI to import it or explicitly start fresh "
                                 "before using eden-cli. No configuration was created.");
    }
}

fs::path RemapImportedPath(const fs::path& value, const fs::path& source,
                           const fs::path& destination) {
    if (value.empty() || !value.is_absolute()) {
        return value;
    }
    const auto normalized = value.lexically_normal();
    auto root = source.lexically_normal();
    if (root.filename().empty()) {
        root = root.parent_path();
    }
    auto current = normalized.begin();
    for (const auto& component : root) {
        if (current == normalized.end()) {
            return value;
        }
#ifdef _WIN32
        auto left = component.wstring();
        auto right = current->wstring();
        std::transform(left.begin(), left.end(), left.begin(), ::towlower);
        std::transform(right.begin(), right.end(), right.begin(), ::towlower);
        if (left != right) {
            return value;
        }
#else
        if (component != *current) {
            return value;
        }
#endif
        ++current;
    }
    auto mapped = destination;
    for (; current != normalized.end(); ++current) {
        mapped /= *current;
    }
    return mapped.lexically_normal();
}

void RemapImportedConfig(const fs::path& config_directory, const fs::path& source,
                         const fs::path& destination) {
    if (!fs::is_directory(config_directory)) {
        return;
    }
    constexpr std::array storage_keys{"nand_directory", "save_directory", "sdmc_directory",
                                      "load_directory", "tas_directory",  "dump_directory"};
    for (const auto& entry : fs::recursive_directory_iterator(config_directory)) {
        if (entry.path().extension() != ".ini" || !entry.is_regular_file()) {
            continue;
        }
        std::ifstream input{entry.path(), std::ios::binary};
        const std::string data{std::istreambuf_iterator<char>{input}, {}};
        if (!input || input.bad()) {
            throw std::runtime_error("Could not read imported configuration");
        }
        CSimpleIniA ini;
        ini.SetUnicode();
        if (ini.LoadData(data) < 0) {
            throw std::runtime_error("Could not parse imported configuration");
        }
        bool changed = false;
        const auto remap = [&](const char* section, const char* key) {
            const auto* value = ini.GetValue(section, key);
            if (!value) {
                return;
            }
            std::string decoded{value};
            std::erase(decoded, '"');
            const auto path = Common::FS::ToU8String(decoded);
            const auto mapped = RemapImportedPath(fs::path{path}, source, destination);
            if (mapped != fs::path{path}) {
                const auto text = Common::FS::PathToUTF8String(mapped);
                if (ini.SetValue(section, key, text.c_str()) < 0) {
                    throw std::runtime_error("Could not remap imported configuration");
                }
                changed = true;
            }
        };
        for (const auto* key : storage_keys) {
            remap("Data%20Storage", key);
        }
        remap("Screenshots", "screenshot_path");
        if (changed) {
            std::string output;
            if (ini.Save(output) < 0) {
                throw std::runtime_error("Could not serialize imported configuration");
            }
            std::ofstream file{entry.path(), std::ios::binary | std::ios::trunc};
            file << output;
            file.close();
            if (!file) {
                throw std::runtime_error("Could not write imported configuration");
            }
        }
    }
}

} // namespace FrontendCommon
