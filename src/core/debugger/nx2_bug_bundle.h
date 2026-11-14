// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Core::Nx2Debug {

/// A one-press bug report (docs/nx2/bug-bundle.md): one zip with the logs, the system, game and
/// settings report, a screenshot, a capture of the next frame with the shaders it used, and the
/// last seconds of frame times, kept under a size limit. Capture runs while the game does; Write
/// uses only what Capture kept, so the game may stop in between.
class BugBundle {
public:
    /// Under Discord's 20 MB limit for free accounts and GitHub's 25 MB attachments.
    static constexpr u64 DEFAULT_SIZE_LIMIT = 19ULL << 20;

    struct Options {
        /// "ui" (Help menu or hotkey) or "debug_server"
        std::string trigger;
        /// Where the zip goes; DefaultDirectory() when empty
        std::filesystem::path directory;
        u64 size_limit{DEFAULT_SIZE_LIMIT};
        std::chrono::seconds capture_timeout{90};
    };

    struct Result {
        std::filesystem::path path;
        u64 bytes{};
        /// What was left out to fit, one line each
        std::vector<std::string> dropped;
        /// Why the frame capture is missing, empty when it is in the zip
        std::string frame_error;
        /// bundle.json
        nlohmann::json manifest;
    };

    BugBundle(System& system, Options options);
    ~BugBundle();

    BugBundle(const BugBundle&) = delete;
    BugBundle& operator=(const BugBundle&) = delete;

    /// Takes the screenshot and the frame capture and collects the report. Blocks until the
    /// capture is written, the capture timeout passes or stop is set. Call it from a thread that
    /// does not present frames.
    void Capture(const std::atomic<bool>& stop);

    /// Writes the zip; description is the player's text. Throws std::runtime_error.
    [[nodiscard]] Result Write(std::string_view description);

    /// <log dir>/bug-reports
    [[nodiscard]] static std::filesystem::path DefaultDirectory();

private:
    System& system;
    Options options;
    std::filesystem::path work;
    std::string file_name;
    nlohmann::json report;
    nlohmann::json system_report;
    nlohmann::json mods;
    nlohmann::json frame_times;
    std::string settings;
    std::vector<u8> screenshot;
    std::string screenshot_error;
    std::string frame_error;
};

} // namespace Core::Nx2Debug
