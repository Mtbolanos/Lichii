// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nx2_precompile.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <thread>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "core/file_sys/nx2_shader_extract.h"

namespace VideoCommon::Nx2Precompile {
namespace {

std::mutex g_game_files_mutex;
u64 g_game_files_title{};
GameFiles g_game_files;

/// Size and last write time of a file, as text, or empty when it cannot be read.
std::string FileStamp(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        return {};
    }
    const auto time = std::filesystem::last_write_time(path, error);
    if (error) {
        return {};
    }
    return fmt::format("{} {}", size, time.time_since_epoch().count());
}

/// <file>.recipe.txt: the recipe a precompiled file was assembled from, by its size and the
/// hash of its bytes. Not its path nor its time: a recipe beside the exe moves with every build
/// folder, and a rewritten file there would change the precompiled file's time, which the
/// prepared stamp holds (the whole preparation would run again for the same pipelines).
std::filesystem::path SourcePath(const std::filesystem::path& precompiled) {
    auto path = precompiled;
    path += ".recipe.txt";
    return path;
}

std::string SourceText(const Recipe& recipe) {
    return fmt::format("recipe {} {:016x}\n", recipe.size, recipe.content_hash);
}

/// The note as builds before 2026-11-09 wrote it: the recipe's path and its size and time.
std::string LegacySourceText(const std::filesystem::path& recipe) {
    return fmt::format("{}\n{}\n", Common::FS::PathToUTF8String(recipe), FileStamp(recipe));
}

/// Whether the file at `path` holds exactly `bytes`.
bool SameBytes(const std::filesystem::path& path, std::span<const u8> bytes) {
    std::error_code error;
    if (std::filesystem::file_size(path, error) != bytes.size() || error) {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    std::vector<char> chunk(size_t{1} << 20);
    size_t at = 0;
    while (at < bytes.size()) {
        const size_t wanted = std::min(chunk.size(), bytes.size() - at);
        if (!file.read(chunk.data(), static_cast<std::streamsize>(wanted)) ||
            std::memcmp(chunk.data(), bytes.data() + at, wanted) != 0) {
            return false;
        }
        at += wanted;
    }
    return true;
}

std::optional<std::vector<u8>> ReadWholeFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    std::vector<u8> bytes{std::istreambuf_iterator<char>(file), {}};
    if (file.bad()) {
        return std::nullopt;
    }
    return bytes;
}

/// The "recipes" folder shipped next to the executable.
std::filesystem::path RecipesDir() {
#ifdef _WIN32
    return Common::FS::GetExeDirectory() / "recipes";
#else
    std::error_code error;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::filesystem::path{"recipes"} : exe.parent_path() / "recipes";
#endif
}

} // Anonymous namespace

int Mode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE");
        if (value == nullptr || *value == '\0') {
            // On by default since 2026-11-09 (owner, with the Fritchii "Preparing shaders" screen):
            // OoT first visit with an empty cache, prepared vs cold driver cache: worst walking
            // freezes ~700-850 ms -> 40-70 ms, 35 -> 43 fps. Only runs when a recipe exists.
            // EDEN_NX2_PRECOMPILE=0 turns it off.
            return 1;
        }
        return std::clamp(std::atoi(value), 0, 2);
    }();
    return mode;
}

bool Forced() {
    const char* value = std::getenv("EDEN_NX2_PRECOMPILED_FORCE");
    return value != nullptr && value[0] == '1';
}

unsigned Threads() {
    static const unsigned threads = [] {
        if (const char* value = std::getenv("EDEN_NX2_PRECOMPILE_THREADS")) {
            return static_cast<unsigned>(std::clamp(std::atoi(value), 1, 16));
        }
        return std::clamp(std::thread::hardware_concurrency() / 4, 1U, 6U);
    }();
    return threads;
}

void SetGameFiles(u64 title_id, GameFiles files) {
    std::scoped_lock lock{g_game_files_mutex};
    g_game_files_title = files ? title_id : 0;
    g_game_files = std::move(files);
}

std::vector<FileSys::VirtualDir> OpenGameFiles(u64 title_id) {
    std::scoped_lock lock{g_game_files_mutex};
    if (!g_game_files || g_game_files_title != title_id) {
        return {};
    }
    return g_game_files();
}

std::filesystem::path PrecompiledPath(const std::filesystem::path& title_dir, u32 cache_version) {
    return title_dir / fmt::format("precompiled.fork-v{}.nx2prep", cache_version);
}

std::optional<Recipe> FindRecipe(u64 title_id, const std::filesystem::path& title_dir,
                                 u32 cache_version) {
    std::vector<std::filesystem::path> candidates;
    if (const char* forced = std::getenv("EDEN_NX2_PRECOMPILE_RECIPE");
        forced != nullptr && *forced != '\0') {
        candidates.emplace_back(forced);
    }
    for (const auto& folder : {title_dir, RecipesDir()}) {
        std::error_code error;
        std::vector<std::filesystem::path> found;
        for (std::filesystem::directory_iterator it{folder, error}, end; !error && it != end;
             it.increment(error)) {
            if (it->path().extension() == ".nx2recipe") {
                found.push_back(it->path());
            }
        }
        // The same order on every boot, whatever the directory listing gives.
        std::ranges::sort(found);
        candidates.insert(candidates.end(), found.begin(), found.end());
    }
    for (const auto& path : candidates) {
        const auto bytes = ReadWholeFile(path);
        if (!bytes) {
            LOG_WARNING(Render_Vulkan, "Precompile recipe {} cannot be read",
                        Common::FS::PathToUTF8String(path));
            continue;
        }
        const auto header = Nx2Recipe::ReadHeader(*bytes);
        if (!header || header->title_id != title_id) {
            continue;
        }
        if (header->cache_version != cache_version) {
            LOG_WARNING(Render_Vulkan,
                        "Precompile recipe {} is for pipeline cache version {}, this build uses "
                        "{}: it needs a recipe made for this build",
                        Common::FS::PathToUTF8String(path), header->cache_version, cache_version);
            continue;
        }
        return Recipe{path, *header, bytes->size(),
                      Common::CityHash64(reinterpret_cast<const char*>(bytes->data()),
                                         bytes->size())};
    }
    return std::nullopt;
}

std::filesystem::path EnsurePrecompiledFile(u64 title_id, const std::filesystem::path& title_dir,
                                            u32 cache_version, std::stop_token stop) {
    const auto precompiled = PrecompiledPath(title_dir, cache_version);
    std::error_code error;
    const bool have_file = std::filesystem::exists(precompiled, error);
    const auto recipe = FindRecipe(title_id, title_dir, cache_version);
    if (!recipe) {
        if (!have_file) {
            LOG_INFO(Render_Vulkan,
                     "Precompiled pipelines: no {} and no recipe for {:016X} (looked in {} and {})",
                     precompiled.filename().string(), title_id,
                     Common::FS::PathToUTF8String(title_dir),
                     Common::FS::PathToUTF8String(RecipesDir()));
            return {};
        }
        return precompiled;
    }
    if (have_file) {
        // A file without the note was put there by hand (or by tools/nx2-precompile): it is the
        // player's, and wins. One assembled from this very recipe is current.
        std::ifstream note(SourcePath(precompiled), std::ios::binary);
        if (!note) {
            return precompiled;
        }
        const std::string previous{std::istreambuf_iterator<char>(note), {}};
        if (previous == SourceText(*recipe)) {
            return precompiled;
        }
        if (previous == LegacySourceText(recipe->path)) {
            // The same recipe, noted the old way: noted again, the file stays.
            note.close();
            std::ofstream rewritten(SourcePath(precompiled), std::ios::binary | std::ios::trunc);
            rewritten << SourceText(*recipe);
            return precompiled;
        }
        LOG_INFO(Render_Vulkan, "Precompiled pipelines: {} changed, assembling it again",
                 Common::FS::PathToUTF8String(recipe->path));
    }

    const auto begin = std::chrono::steady_clock::now();
    const auto seconds = [&begin] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    };
    const auto roots = OpenGameFiles(title_id);
    if (roots.empty()) {
        LOG_WARNING(Render_Vulkan, "Precompiled pipelines: the game's files cannot be read, "
                                   "the recipe is not assembled");
        return have_file ? precompiled : std::filesystem::path{};
    }
    const auto recipe_bytes = ReadWholeFile(recipe->path);
    if (!recipe_bytes) {
        return have_file ? precompiled : std::filesystem::path{};
    }
    LOG_INFO(Render_Vulkan,
             "Precompiled pipelines: assembling {} (game version {}) with this copy of the game "
             "on {} threads",
             Common::FS::PathToUTF8String(recipe->path), recipe->header.game_version, Threads());
    std::vector<u8> assembled;
    Nx2Recipe::Report report;
    try {
        const auto programs = FileSys::Nx2ShaderExtract::Extract(roots, Threads(), nullptr, stop);
        if (stop.stop_requested()) {
            return {};
        }
        LOG_INFO(Render_Vulkan,
                 "Precompiled pipelines: {} shader programs found in {} files ({} MB) in {:.1f} s",
                 programs.programs.size(), programs.stats.files,
                 programs.stats.file_bytes >> 20, seconds());
        assembled = Nx2Recipe::Assemble(*recipe_bytes, programs, cache_version, report, Threads());
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Precompiled pipelines: assembling the recipe failed: {}",
                  e.what());
        return have_file ? precompiled : std::filesystem::path{};
    }
    if (stop.stop_requested() || assembled.empty()) {
        return {};
    }
    if (have_file && SameBytes(precompiled, assembled)) {
        // The same file again (the recipe was copied or moved, not changed): it is left as it
        // is, so its time and the preparation made for it stay valid.
        std::ofstream note(SourcePath(precompiled), std::ios::binary | std::ios::trunc);
        note << SourceText(*recipe);
        LOG_INFO(Render_Vulkan,
                 "Precompiled pipelines: {} assembled again in {:.1f} s, the same as before: kept",
                 precompiled.filename().string(), seconds());
        return precompiled;
    }
    // The file is large (Ocarina of Time: 1.17 GB); a full disk must not end with half of one.
    const auto space = std::filesystem::space(title_dir, error);
    if (!error && space.available < assembled.size() + (u64{512} << 20)) {
        LOG_WARNING(Render_Vulkan,
                    "Precompiled pipelines: {} MB needed and {} MB free in {}, not written",
                    assembled.size() >> 20, space.available >> 20,
                    Common::FS::PathToUTF8String(title_dir));
        return have_file ? precompiled : std::filesystem::path{};
    }
    auto temporary = precompiled;
    temporary += ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(assembled.data()),
                   static_cast<std::streamsize>(assembled.size()));
        file.close();
        if (!file) {
            std::filesystem::remove(temporary, error);
            LOG_ERROR(Render_Vulkan, "Precompiled pipelines: {} cannot be written",
                      Common::FS::PathToUTF8String(temporary));
            return have_file ? precompiled : std::filesystem::path{};
        }
    }
    std::filesystem::rename(temporary, precompiled, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        LOG_ERROR(Render_Vulkan, "Precompiled pipelines: {} cannot be renamed",
                  Common::FS::PathToUTF8String(temporary));
        return have_file ? precompiled : std::filesystem::path{};
    }
    {
        std::ofstream note(SourcePath(precompiled), std::ios::binary | std::ios::trunc);
        note << SourceText(*recipe);
    }
    LOG_INFO(Render_Vulkan,
             "Precompiled pipelines: assembled {} ({} MB) in {:.1f} s: {} pipelines, {} programs "
             "({} not in this copy of the game, {} not lowered), {} environments, {} render "
             "states, {} records dropped",
             precompiled.filename().string(), assembled.size() >> 20, seconds(), report.records,
             report.programs, report.programs_missing, report.programs_failed,
             report.environments, report.states, report.records_dropped);
    for (size_t i = 0; i < std::min<size_t>(report.errors.size(), 8); ++i) {
        LOG_DEBUG(Render_Vulkan, "Precompiled pipelines: {}", report.errors[i]);
    }
    return precompiled;
}

} // namespace VideoCommon::Nx2Precompile
