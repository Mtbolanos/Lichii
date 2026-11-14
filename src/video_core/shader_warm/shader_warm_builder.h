// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace Core {
class System;
}

namespace VideoCommon::ShaderWarm {

/// Where the shader precache file of a title lives: next to its pipeline cache
/// (<shader dir>/<title id>/warm-compute.bin). It holds the compute programs found in the game
/// files, in pipeline cache format with no recorded reads.
std::filesystem::path ComputeWarmPath(u64 title_id);

/// "Already precached" marker of a precache file (same folder, extension .done), written by the
/// emulator after the precache ran to the end. A matching marker skips the precache at boot.
std::filesystem::path WarmMarkerPath(const std::filesystem::path& warm_file);

/// Entry count of a precache file, written by the builder next to it (extension .count): the
/// entry count and the file's size. The file format has no count, so the loading screen shows
/// exact progress only with this.
std::filesystem::path WarmCountPath(const std::filesystem::path& warm_file);

/// The entry count recorded for `warm_file`, or 0 when there is none or it was written for a
/// file of another size.
u64 ReadWarmCount(const std::filesystem::path& warm_file);

/// What a marker records: the precache file's size and content hash, its entry count, the GPU
/// driver version and device ID, the pipeline cache version and the variant: a hash of the build's
/// SPIR-V emission sources (EmissionStamp) and of the settings that change the SPIR-V or pipeline
/// state of the precache builds (resolution scaling, for one).
/// The content, not the last-write time: building the file again with the same shaders ("Precache
/// Shaders" by hand) rebuilt every compute program (~3000, 8.5 min on a Ryzen 5 5600X). The
/// variant: after an update that changes the SPIR-V, or a change to one of those settings, the
/// programs are built again at boot, where before the marker still said "done" and they were
/// built during play.
struct WarmStamp {
    u64 size{};
    u64 content_hash{};
    u64 entries{};
    u32 driver{};
    u32 device_id{};
    u32 cache_version{};
    u64 variant{};

    /// True when `other` was written for the same file contents, driver, device, cache version
    /// and variant (the entry count is informative).
    [[nodiscard]] bool Matches(const WarmStamp& other) const noexcept {
        return size == other.size && content_hash == other.content_hash &&
               driver == other.driver && device_id == other.device_id &&
               cache_version == other.cache_version && variant == other.variant;
    }
};

/// A hash of this build's sources that shape the SPIR-V of guest shaders (the recompiler without
/// its OpenGL backends, the native lowering, the shader environment and the Vulkan wrapper that
/// calls the SPIR-V backend), made when it is built. It covers every source of the lowering stamp,
/// and changes only with a build that may emit other SPIR-V, unlike the revision.
[[nodiscard]] u64 EmissionStamp();

/// The stamp of `warm_file` as it is now, empty when it cannot be read.
std::optional<WarmStamp> CurrentWarmStamp(const std::filesystem::path& warm_file, u64 entries,
                                          u32 driver, u32 device_id, u32 cache_version,
                                          u64 variant = 0);

/// The marker of `warm_file`, empty when there is none or it cannot be read.
std::optional<WarmStamp> ReadWarmMarker(const std::filesystem::path& warm_file);

/// Writes the marker of `warm_file`. Returns false when it cannot be written.
bool WriteWarmMarker(const std::filesystem::path& warm_file, const WarmStamp& stamp);

/// Removes the marker of `warm_file` before `new_file` replaces it, unless the marker was written
/// for the same contents: a rebuild that comes out identical stays done.
void DropStaleWarmMarker(const std::filesystem::path& warm_file,
                         const std::filesystem::path& new_file);

/// True when the title's precache has never finished: no precache file yet, or no marker next
/// to it. A marker that no longer matches (new driver) is not seen here: the emulator checks that
/// at boot.
bool WarmPending(u64 title_id);

/// Whether this boot precaches before the game starts: Always, or Ask when the frontend asked and
/// the user agreed (`asked`). Off never does.
[[nodiscard]] constexpr bool PrecacheBeforeStart(Settings::PrecacheShaders mode, bool asked) {
    return mode == Settings::PrecacheShaders::Always ||
           (mode == Settings::PrecacheShaders::Ask && asked);
}

/// The launch's answer to "precache shaders before the game starts" (precache_shaders = Ask):
/// set by the frontend before booting, taken (and cleared) by the renderer when it loads. Setting
/// it also clears an earlier skip request.
void RequestWarmBeforeStart(bool before_start);
bool TakeWarmBeforeStart();

/// Skip button of the loading screen: the running precache stops after its current builds and the
/// game starts. Taken (and cleared) by the renderer.
void RequestSkipWarm();
bool TakeSkipWarm();

/// Removes the title's precache file, its marker and count, the list of precached graphics
/// pipelines older builds kept and a leftover temporary file (rebuild or delete from the UI). Returns false when a file exists but cannot be removed, with
/// the reason in `error` when given.
bool DeleteWarmFiles(u64 title_id, std::string* error = nullptr);

struct Progress {
    std::string stage;   ///< "Reading game files", "Extracting shaders", "Converting", "Writing"
    size_t done{};       ///< Items done in this stage
    size_t total{};      ///< Items in this stage, 0 when unknown
};

struct Result {
    bool ok{};
    std::string error;           ///< Empty when ok
    size_t archives{};           ///< Shader archives found in the game files
    size_t compute_programs{};   ///< Entries written to the precache file
    size_t skipped{};            ///< Programs that failed to extract or convert
};

/// Why a precache file must be built again: missing, of a cache version the emulator does not
/// load, written by an older builder (WarmBuilderVersion) or changed since. Empty when it is
/// current.
std::optional<std::string> WarmFilesStale(const std::filesystem::path& warm_file);

/// Builds the precache file for the game at `game_path` (any format the loader opens, Switch 2
/// .nup first), from the user's own copy and keys, with no external tools. Runs on the calling
/// thread; `progress` is called from it; stops early when `stop` is requested.
Result BuildWarmFiles(Core::System& system, const std::filesystem::path& game_path, u64 title_id,
                      const std::function<void(const Progress&)>& progress,
                      std::stop_token stop);

/// Overrides for BuildWarmFilesWith (tests and tools).
struct BuildOptions {
    /// Folder for warm-compute.bin; empty: <shader dir>/<title id>.
    std::filesystem::path output_dir;
    /// Conversion threads, 0: a quarter of the CPU threads, 2 to 6.
    size_t threads{};
};

/// BuildWarmFiles with an explicit output folder.
Result BuildWarmFilesWith(Core::System& system, const std::filesystem::path& game_path,
                          u64 title_id, const BuildOptions& options,
                          const std::function<void(const Progress&)>& progress,
                          std::stop_token stop);

} // namespace VideoCommon::ShaderWarm
