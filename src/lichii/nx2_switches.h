// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"

// The NX2 switches are read from the environment (EDEN_NX2_*), which a GUI launch has no way to
// set. nx2_switches.txt holds "NAME=VALUE" lines, one per line, '#' comments. A variable already
// in the real environment always wins, so a launcher script can still override one run; between
// files the later one wins, so the copy next to the executable (shipped with the build) can be
// overridden by %APPDATA%\lichii\nx2_switches.txt without editing the one that gets replaced.
static void ParseSwitchFile(const std::filesystem::path& path,
                            std::map<std::string, std::string>& entries) {
    std::ifstream file{path};
    if (!file) {
        return;
    }
    constexpr const char* blank = " \t\r";
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(blank));
        line.erase(line.find_last_not_of(blank) + 1);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string name = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        name.erase(name.find_last_not_of(blank) + 1);
        value.erase(0, value.find_first_not_of(blank));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }
        if (!name.empty()) {
            entries.insert_or_assign(std::move(name), std::move(value));
        }
    }
}

static std::filesystem::path GetSwitchExeDirectory(const char* argv0) {
#ifdef _WIN32
    return Common::FS::GetExeDirectory();
#else
    std::error_code ec;
#ifdef __linux__
    if (auto exe = std::filesystem::read_symlink("/proc/self/exe", ec); !ec) {
        return exe.parent_path();
    }
#endif
    const auto exe = std::filesystem::absolute(argv0, ec);
    return ec ? std::filesystem::path{} : exe.parent_path();
#endif
}

static void LoadNx2Switches(const char* argv0) {
    // Later files overwrite earlier entries; the real environment is checked only at the end.
    std::map<std::string, std::string> entries;
    if (const auto exe_dir = GetSwitchExeDirectory(argv0); !exe_dir.empty()) {
        ParseSwitchFile(exe_dir / "nx2_switches.txt", entries);
    }
#ifdef __linux__
    if (const char* appimage = std::getenv("APPIMAGE"); appimage && *appimage) {
        std::error_code ec;
        const std::filesystem::path path{appimage};
        if (path.is_absolute()) {
            const auto resolved = std::filesystem::canonical(path, ec);
            if (!ec) {
                ParseSwitchFile(resolved.parent_path() / "nx2_switches.txt", entries);
            }
        }
    }
#endif
    std::erase_if(entries, [](const auto& entry) {
        const auto& name = entry.first;
        return !(name.starts_with("EDEN_NX2_") || name == "EDEN_DKB_FRAME_TRACE" ||
                 name == "EDEN_DKB_INDIRECT_READBACK" || name == "EDEN_NATIVE_DRAW_CAPTURE" ||
                 name == "EDEN_NATIVE_DRAW_DIAGNOSTICS" || name == "EDEN_NATIVE_FRAME_CAPTURE" ||
                 name == "EDEN_NATIVE_SHADER_OVERRIDE_DIR" || name == "EDEN_PUNCH_HOLE" ||
                 name == "EDEN_STARFOX_WINDOW_TRACE" || name == "EDEN_VI2_MAX_RESOLUTION_MODE" ||
                 name == "EGLANTINE_EXPERIMENTAL_STATIC_VERTEX_REPACK" ||
                 name == "EGLANTINE_TEST_VERTEX_REPACK");
    });
#ifdef _WIN32
    const auto config_base = Common::FS::GetAppDataRoamingDirectory();
#else
    const auto config_base = Common::FS::GetDataDirectory("XDG_CONFIG_HOME");
#endif
    if (config_base.is_absolute()) {
        ParseSwitchFile(config_base / EDEN_DIR / "nx2_switches.txt", entries);
    }
    for (const auto& [name, value] : entries) {
        if (std::getenv(name.c_str()) != nullptr) {
            continue;
        }
#ifdef _WIN32
        _putenv_s(name.c_str(), value.c_str());
#else
        setenv(name.c_str(), value.c_str(), 1);
#endif
    }
}
