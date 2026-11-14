// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>
#include <QImage>
#include <QObject>
#include <QStandardPaths>
#include "common/fs/path_util.h"

namespace QtCommon::Game {

enum class InstalledEntryType {
    Game,
    Update,
    AddOnContent,
};

enum class GameListRemoveTarget {
    GlShaderCache,
    VkShaderCache,
    AllShaderCache,
    CustomConfiguration,
    CacheStorage,
};

enum class ShortcutTarget {
    Desktop,
    Applications,
};

enum class ShortcutMessages { Fullscreen = 0, Success = 1, Volatile = 2, Failed = 3 };

bool CreateShortcutLink(const std::filesystem::path& shortcut_path, const std::string& comment,
                        const std::filesystem::path& icon_path,
                        const std::filesystem::path& command, const std::string& arguments,
                        const std::string& categories, const std::string& keywords,
                        const std::string& name);

bool MakeShortcutIcoPath(const u64 program_id, const std::string_view game_file_name,
                         std::filesystem::path& out_icon_path);

void OpenEdenFolder(const Common::FS::EdenPath& path);
void OpenRootDataFolder();
void OpenNANDFolder();
void OpenSaveFolder();
void OpenSDMCFolder();
void OpenModFolder();
void OpenLogFolder();

void RemoveBaseContent(u64 program_id, InstalledEntryType type);
void RemoveUpdateContent(u64 program_id, InstalledEntryType type);
void RemoveAddOnContent(u64 program_id, InstalledEntryType type);

void RemoveTransferableShaderCache(u64 program_id, GameListRemoveTarget target);
void RemoveVulkanDriverPipelineCache(u64 program_id);
void RemoveAllTransferableShaderCaches(u64 program_id);
void RemoveCustomConfiguration(u64 program_id, const std::string& game_path);
void RemoveCacheStorage(u64 program_id);

// Metadata //
void ResetMetadata(bool show_message = true);

// Shortcuts //
void CreateShortcut(const std::string& game_path, const u64 program_id,
                    const std::string& game_title_, const ShortcutTarget& target,
                    std::string arguments_, const bool needs_title);

std::string GetShortcutPath(ShortcutTarget target);
void CreateHomeMenuShortcut(ShortcutTarget target);

/**
 * Saves a windows icon to a file
 * @param path The icons path
 * @param image The image to save
 * @return bool If the operation succeeded
 */
[[nodiscard]] bool SaveIconToFile(const std::filesystem::path& icon_path, const QImage& image);

// Custom artwork //

/// Where the library artwork the user picked for a program is stored. It lives outside the game
/// file, so setting it never rewrites the game, and it works for every game format.
std::filesystem::path CustomArtworkPath(u64 program_id);

/// The custom artwork image file bytes for a program, or empty when none is set.
std::vector<u8> ReadCustomArtwork(u64 program_id);

/// Stores an image as the program's custom artwork (PNG, at most 1024 pixels per side).
[[nodiscard]] bool SaveCustomArtwork(u64 program_id, const QImage& image);

/// Deletes the program's custom artwork. Returns true when none is left.
bool RemoveCustomArtwork(u64 program_id);

// Custom name //

/// Where the name the user gave a program is stored (one UTF-8 line). Like custom artwork it
/// lives outside the game file, so a rename never rewrites the game.
std::filesystem::path CustomTitlePath(u64 program_id);

/// The program's custom name, or empty when none is set.
std::string ReadCustomTitle(u64 program_id);

/// Stores the program's custom name; an empty name removes it. Returns true on success.
[[nodiscard]] bool SaveCustomTitle(u64 program_id, const std::string& title);

} // namespace QtCommon::Game
