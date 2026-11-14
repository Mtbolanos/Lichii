// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/loader/loader.h"
#include "video_core/shader_warm/emission_stamp.h"
#include "video_core/shader_warm/shader_archives.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"

namespace VideoCommon::ShaderWarm {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view StageReading = "Reading game files";
constexpr std::string_view StageExtracting = "Extracting shaders";
constexpr std::string_view StageConverting = "Converting";
constexpr std::string_view StageWriting = "Writing";
/// Conversions in flight (each holds a copy of its program).
constexpr size_t MaxPendingJobs = 2048;

fs::path TitleDir(u64 title_id) {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) /
           fmt::format("{:016x}", title_id);
}

struct ArchiveFile {
    std::string path; ///< Inside the RomFS, '/' separated (the archive identity)
    FileSys::VirtualFile file;
};

/// Appends every file under `dir` to `out`, recursively, with its '/' separated path.
void ListFiles(const FileSys::VirtualDir& dir, const std::string& prefix,
               std::vector<ArchiveFile>& out) {
    for (const auto& file : dir->GetFiles()) {
        out.push_back({prefix + file->GetName(), file});
    }
    for (const auto& sub : dir->GetSubdirectories()) {
        ListFiles(sub, prefix + sub->GetName() + "/", out);
    }
}

/// The game's RomFS trees in the order the offline pipeline read them: a .nup gives its base
/// and newest update package (both are read, like the exported packages 0 and 1); other
/// formats give the loader's RomFS.
struct GameFiles {
    std::unique_ptr<FileSys::NupContainer> nup;
    std::unique_ptr<Loader::AppLoader> loader;
    std::vector<FileSys::VirtualDir> romfs;
};

/// Opens the game file and collects its RomFS trees into `game`. Returns the error text when the
/// file cannot be opened or has no RomFS.
std::optional<std::string> OpenGame(Core::System& system, const fs::path& game_path,
                                    GameFiles& game) {
    FileSys::VirtualFilesystem filesystem = system.GetFilesystem();
    if (filesystem == nullptr) {
        filesystem = std::make_shared<FileSys::RealVfsFilesystem>();
    }
    const auto file =
        filesystem->OpenFile(Common::FS::PathToUTF8String(game_path), FileSys::OpenMode::Read);
    if (file == nullptr) {
        return "Cannot open the game file";
    }
    if (FileSys::NupContainer::IsNupFile(file)) {
        // Small memory cache: every member is read once.
        game.nup = std::make_unique<FileSys::NupContainer>(file, 128 * 1024 * 1024);
        if (game.nup->GetStatus() != Loader::ResultStatus::Success) {
            return "The game file is not a valid package";
        }
        std::vector<size_t> packages;
        if (const auto base = game.nup->FindBasePackage()) {
            packages.push_back(*base);
        }
        if (const auto update = game.nup->FindBestUpdatePackage()) {
            packages.push_back(*update);
        }
        std::sort(packages.begin(), packages.end());
        for (const size_t package : packages) {
            if (auto dir = game.nup->GetRomFSDir(package)) {
                game.romfs.push_back(std::move(dir));
            }
        }
    } else {
        game.loader = Loader::GetLoader(system, file);
        if (game.loader == nullptr) {
            return "Unsupported game file format";
        }
        FileSys::VirtualFile romfs;
        if (game.loader->ReadRomFS(romfs) != Loader::ResultStatus::Success || romfs == nullptr) {
            return "Cannot read the game's RomFS (are the keys installed?)";
        }
        if (auto dir = FileSys::ExtractRomFS(romfs)) {
            game.romfs.push_back(std::move(dir));
        }
    }
    if (game.romfs.empty()) {
        return "The game has no RomFS";
    }
    return std::nullopt;
}

struct PairHash {
    u128 a{};
    u128 b{};
    bool operator==(const PairHash&) const = default;
};
struct PairHasher {
    size_t operator()(const PairHash& h) const noexcept {
        return static_cast<size_t>(h.a[0] ^ (h.b[0] * 0x9E3779B97F4A7C15ULL));
    }
};

u128 Hash128(std::span<const u8> bytes) {
    return Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

struct Job {
    size_t seq{};
    std::vector<u8> program;
    ComputeControl control{};
};

struct JobResult {
    bool ok{};
    std::vector<u8> bytes; ///< The full compute entry
    ComputeKey compute_key{};
};

class Converter {
public:
    /// Starts `threads` workers that convert pushed jobs until `stop` is requested.
    Converter(size_t threads, std::stop_token stop_) : stop{std::move(stop_)} {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] { Work(); });
        }
    }

    /// Lets the workers finish the queued jobs, then joins them.
    ~Converter() {
        {
            std::scoped_lock lock{mutex};
            closing = true;
        }
        jobs_cv.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }

    void Push(Job job) {
        {
            std::scoped_lock lock{mutex};
            jobs.push_back(std::move(job));
            ++pushed;
        }
        jobs_cv.notify_one();
    }

    /// Next result in submission order; waits when `wait` is set, else empty when not ready.
    std::optional<JobResult> Next(bool wait) {
        std::unique_lock lock{mutex};
        for (;;) {
            const auto it = results.find(next);
            if (it != results.end()) {
                JobResult result = std::move(it->second);
                results.erase(it);
                ++next;
                return result;
            }
            if (!wait || next == pushed || stop.stop_requested()) {
                return std::nullopt;
            }
            results_cv.wait_for(lock, std::chrono::milliseconds{100});
        }
    }

    size_t Pending() {
        std::scoped_lock lock{mutex};
        return pushed - next;
    }

private:
    /// Worker loop: takes jobs in order, converts each one (skipped once stopping) and files the
    /// result under its sequence number. Returns once closing with no job left.
    void Work() {
        Common::SetCurrentThreadName("ShaderWarmBuild");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        for (;;) {
            Job job;
            {
                std::unique_lock lock{mutex};
                jobs_cv.wait(lock, [this] { return closing || !jobs.empty(); });
                if (jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            JobResult result;
            if (!stop.stop_requested()) {
                result = Convert(job);
            }
            {
                std::scoped_lock lock{mutex};
                results.emplace(job.seq, std::move(result));
            }
            results_cv.notify_all();
        }
    }

    /// Lowers one program and makes its compute entry. The result is not ok when lowering or
    /// conversion failed.
    static JobResult Convert(const Job& job) {
        JobResult result;
        try {
            LoweredCode lowered;
            if (const auto error = LowerProgram(job.program, lowered)) {
                LOG_DEBUG(Render_Vulkan, "Shader precache: program not converted: {}", *error);
                return result;
            }
            result.bytes = ComputeEntry(lowered, job.control, result.compute_key);
            result.ok = true;
        } catch (const std::exception& e) {
            LOG_DEBUG(Render_Vulkan, "Shader precache: conversion failed: {}", e.what());
        }
        return result;
    }

    std::stop_token stop;
    std::mutex mutex;
    std::condition_variable jobs_cv;
    std::condition_variable results_cv;
    std::deque<Job> jobs;
    std::map<size_t, JobResult> results;
    size_t pushed{};
    size_t next{};
    bool closing{};
    std::vector<std::thread> workers;
};

template <typename T>
void Write(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void WriteHeader(std::ofstream& out) {
    out.write("yuzucach", 8);
    Write(out, WarmCacheVersion);
}

std::atomic<bool> warm_before_start{false};
std::atomic<bool> skip_warm{false};

/// Records `entries` for the finished precache file at `warm_file` (see WarmCountPath).
void WriteWarmCount(const fs::path& warm_file, u64 entries) {
    std::error_code ec;
    const auto size = fs::file_size(warm_file, ec);
    if (ec) {
        return;
    }
    std::ofstream out(WarmCountPath(warm_file), std::ios::trunc);
    out << entries << ' ' << size << ' ' << WarmBuilderVersion << '\n';
}

} // Anonymous namespace

fs::path ComputeWarmPath(u64 title_id) {
    return TitleDir(title_id) / "warm-compute.bin";
}

fs::path WarmMarkerPath(const fs::path& warm_file) {
    fs::path marker = warm_file;
    marker.replace_extension(".done");
    return marker;
}

fs::path WarmCountPath(const fs::path& warm_file) {
    fs::path count = warm_file;
    count.replace_extension(".count");
    return count;
}

u64 ReadWarmCount(const fs::path& warm_file) {
    std::ifstream in(WarmCountPath(warm_file));
    u64 entries{};
    u64 size{};
    if (!(in >> entries >> size)) {
        return 0;
    }
    std::error_code ec;
    const auto actual = fs::file_size(warm_file, ec);
    return !ec && actual == size ? entries : 0;
}

u64 EmissionStamp() {
    return GeneratedEmissionStamp;
}

std::optional<WarmStamp> CurrentWarmStamp(const fs::path& warm_file, u64 entries, u32 driver,
                                          u32 device_id, u32 cache_version, u64 variant) {
    std::error_code ec;
    const auto size = fs::file_size(warm_file, ec);
    if (ec) {
        return std::nullopt;
    }
    // Hashed in chunks (the file is ~70 MB): tens of milliseconds at boot.
    std::ifstream in(warm_file, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::vector<char> chunk(8u << 20);
    u64 hash = size;
    u64 read_total = 0;
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto got = static_cast<size_t>(in.gcount());
        if (got == 0) {
            break;
        }
        hash = Common::CityHash64WithSeed(chunk.data(), got, hash);
        read_total += got;
    }
    if (read_total != size) {
        return std::nullopt;
    }
    return WarmStamp{.size = size,
                     .content_hash = hash,
                     .entries = entries,
                     .driver = driver,
                     .device_id = device_id,
                     .cache_version = cache_version,
                     .variant = variant};
}

std::optional<WarmStamp> ReadWarmMarker(const fs::path& warm_file) {
    std::ifstream in(WarmMarkerPath(warm_file));
    WarmStamp stamp;
    // A marker of the older format (last-write time, no variant) fails here: the compute
    // programs not in the cache are built once more.
    std::string format;
    if (!(in >> format) || format != "v2" ||
        !(in >> stamp.size >> stamp.content_hash >> stamp.entries >> stamp.driver >>
          stamp.device_id >> stamp.cache_version >> stamp.variant)) {
        return std::nullopt;
    }
    return stamp;
}

bool WriteWarmMarker(const fs::path& warm_file, const WarmStamp& stamp) {
    std::ofstream out(WarmMarkerPath(warm_file), std::ios::trunc);
    out << "v2 " << stamp.size << ' ' << stamp.content_hash << ' ' << stamp.entries << ' '
        << stamp.driver << ' ' << stamp.device_id << ' ' << stamp.cache_version << ' '
        << stamp.variant << '\n';
    out.close();
    return static_cast<bool>(out);
}

void DropStaleWarmMarker(const fs::path& warm_file, const fs::path& new_file) {
    if (const auto marker = ReadWarmMarker(warm_file)) {
        const auto incoming = CurrentWarmStamp(new_file, 0, 0, 0, 0);
        if (incoming && incoming->size == marker->size &&
            incoming->content_hash == marker->content_hash) {
            return;
        }
    }
    std::error_code ignored;
    fs::remove(WarmMarkerPath(warm_file), ignored);
}

bool WarmPending(u64 title_id) {
    const auto file = ComputeWarmPath(title_id);
    std::error_code ec;
    return !fs::exists(file, ec) || !fs::exists(WarmMarkerPath(file), ec);
}

void RequestWarmBeforeStart(bool before_start) {
    skip_warm.store(false, std::memory_order_relaxed);
    warm_before_start.store(before_start, std::memory_order_relaxed);
}

bool TakeWarmBeforeStart() {
    return warm_before_start.exchange(false, std::memory_order_relaxed);
}

void RequestSkipWarm() {
    skip_warm.store(true, std::memory_order_relaxed);
}

bool TakeSkipWarm() {
    return skip_warm.exchange(false, std::memory_order_relaxed);
}

bool DeleteWarmFiles(u64 title_id, std::string* error) {
    const auto warm = ComputeWarmPath(title_id);
    fs::path tmp = warm;
    tmp += ".tmp";
    bool ok = true;
    for (const auto& path :
         {warm, WarmMarkerPath(warm), WarmCountPath(warm), tmp,
          TitleDir(title_id) / "precache-graphics.bin"}) {
        std::error_code ec;
        fs::remove(path, ec);
        if (ec) {
            ok = false;
            if (error != nullptr) {
                *error = ec.message();
            }
        }
    }
    return ok;
}

std::optional<std::string> WarmFilesStale(const fs::path& warm_file) {
    std::error_code ec;
    if (!fs::exists(warm_file, ec)) {
        return "none built yet";
    }
    {
        std::ifstream in(warm_file, std::ios::binary);
        std::array<char, 8> magic{};
        u32 version{};
        in.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        const auto& compatible = VideoCommon::CompatibleVulkanPipelineCacheVersions;
        if (!in || magic != VideoCommon::PipelineCacheMagic ||
            (version != WarmCacheVersion &&
             std::ranges::find(compatible, version) == compatible.end())) {
            return "built for another cache version";
        }
    }
    std::ifstream in(WarmCountPath(warm_file));
    u64 entries{};
    u64 size{};
    u32 builder{};
    if (!(in >> entries >> size >> builder) || builder != WarmBuilderVersion) {
        return "built by an older version";
    }
    if (fs::file_size(warm_file, ec) != size || ec) {
        return "changed since it was built";
    }
    return std::nullopt;
}

Result BuildWarmFiles(Core::System& system, const fs::path& game_path, u64 title_id,
                      const std::function<void(const Progress&)>& progress,
                      std::stop_token stop) {
    return BuildWarmFilesWith(system, game_path, title_id, BuildOptions{}, progress,
                              std::move(stop));
}

Result BuildWarmFilesWith(Core::System& system, const fs::path& game_path, u64 title_id,
                          const BuildOptions& options,
                          const std::function<void(const Progress&)>& progress,
                          std::stop_token stop) {
    Result result;
    const auto report = [&](std::string_view stage, size_t done, size_t total) {
        if (progress) {
            progress(Progress{std::string(stage), done, total});
        }
    };
    const auto begin = std::chrono::steady_clock::now();
    report(StageReading, 0, 0);

    const fs::path output_dir = options.output_dir.empty() ? TitleDir(title_id) : options.output_dir;
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    const fs::path compute_path = output_dir / "warm-compute.bin";
    fs::path compute_tmp = compute_path;
    compute_tmp += ".tmp";
    std::ofstream compute_out;
    const auto fail = [&](std::string error) {
        compute_out.close();
        std::error_code ignored;
        fs::remove(compute_tmp, ignored);
        result.ok = false;
        result.error = std::move(error);
        LOG_WARNING(Render_Vulkan, "Shader precache build for {:016x} failed: {}", title_id,
                    result.error);
        return result;
    };

    GameFiles game;
    if (const auto error = OpenGame(system, game_path, game)) {
        return fail(*error);
    }

    // Archive list, in the offline pipeline's order: every shader archive of each package, then
    // each package's effects, sorted by path inside a package (see ClassifyGameFile).
    std::vector<ArchiveFile> archives;
    std::vector<ArchiveFile> effects;
    ZstdDictionaries dictionaries;
    FileSys::VirtualFile dictionary_pack;
    for (const auto& romfs : game.romfs) {
        std::vector<ArchiveFile> files;
        ListFiles(romfs, "", files);
        std::sort(files.begin(), files.end(),
                  [](const ArchiveFile& a, const ArchiveFile& b) { return a.path < b.path; });
        std::vector<ArchiveFile> package_effects;
        for (auto& file : files) {
            switch (ClassifyGameFile(file.path)) {
            case GameFileKind::ShaderArchive:
                archives.push_back(file);
                break;
            case GameFileKind::Effect:
                package_effects.push_back(file);
                break;
            case GameFileKind::ZstdDictionaries:
                dictionary_pack = file.file; // the newest package wins
                break;
            case GameFileKind::Other:
                break;
            }
        }
        effects.insert(effects.end(), package_effects.begin(), package_effects.end());
    }
    archives.insert(archives.end(), effects.begin(), effects.end());
    if (dictionary_pack) {
        try {
            dictionaries.Load(dictionary_pack->ReadAllBytes());
        } catch (const ArchiveError& e) {
            LOG_WARNING(Render_Vulkan, "Shader precache: zstd dictionary pack: {}", e.what());
        }
    }
    // Without archives the file is written empty: the precache then has only the pipeline cache
    // of earlier sessions to build.
    result.archives = archives.size();
    LOG_INFO(Render_Vulkan, "Shader precache build: {} archives, {} zstd dictionaries",
             archives.size(), dictionaries.Count());

    compute_out.open(compute_tmp, std::ios::binary | std::ios::trunc);
    if (!compute_out) {
        return fail("Cannot write to " + Common::FS::PathToUTF8String(output_dir));
    }
    WriteHeader(compute_out);

    std::unordered_set<PairHash, PairHasher> compute_identities;
    std::unordered_set<std::string> compute_keys;
    size_t archive_errors = 0;
    size_t member_errors = 0;
    size_t seq = 0;

    const size_t threads =
        options.threads != 0
            ? options.threads
            : std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 2, 6);
    auto converter = std::make_unique<Converter>(threads, stop);

    // Consumes finished conversions in submission order (first program wins on equal keys).
    size_t consumed = 0;
    const auto consume = [&](JobResult&& r) {
        ++consumed;
        if (!r.ok) {
            ++result.skipped;
            return;
        }
        const std::string key(reinterpret_cast<const char*>(r.compute_key.data()),
                              r.compute_key.size());
        if (compute_keys.insert(key).second) {
            compute_out.write(reinterpret_cast<const char*>(r.bytes.data()),
                              static_cast<std::streamsize>(r.bytes.size()));
            ++result.compute_programs;
        }
    };
    const auto drain = [&] {
        while (auto r = converter->Next(false)) {
            consume(std::move(*r));
        }
    };

    for (size_t archive_index = 0; archive_index < archives.size(); ++archive_index) {
        if (stop.stop_requested()) {
            converter.reset();
            return fail("Stopped");
        }
        report(StageExtracting, archive_index, archives.size());
        const auto& archive = archives[archive_index];
        const auto data = archive.file->ReadAllBytes();
        const auto name = archive.path.substr(archive.path.rfind('/') + 1);
        const auto on_record = [&](const ShaderRecord& record) {
            if (record.stage != ProgramStage::Compute) {
                return;
            }
            if (!compute_identities.insert(PairHash{Hash128(record.code), Hash128(record.control)})
                     .second) {
                return;
            }
            const auto values = ReadComputeControl(record.control);
            if (!values || !values->workgroup[0] || !values->workgroup[1] ||
                !values->workgroup[2]) {
                ++result.skipped;
                return;
            }
            Job job;
            job.control = *values;
            job.seq = seq++;
            job.program.assign(record.code.begin(), record.code.end());
            converter->Push(std::move(job));
            while (converter->Pending() > MaxPendingJobs && !stop.stop_requested()) {
                if (auto r = converter->Next(true)) {
                    consume(std::move(*r));
                }
            }
        };
        const auto on_member_error = [&](const std::string& error) {
            ++member_errors;
            LOG_DEBUG(Render_Vulkan, "Shader precache: {}: {}", archive.path, error);
        };
        try {
            WalkArchive(data, name, &dictionaries, on_record, on_member_error);
        } catch (const ArchiveError& e) {
            ++archive_errors;
            LOG_WARNING(Render_Vulkan, "Shader precache: {}: {}", archive.path, e.what());
        }
        drain();
    }
    report(StageExtracting, archives.size(), archives.size());

    const size_t total_jobs = seq;
    while (consumed < total_jobs) {
        if (stop.stop_requested()) {
            converter.reset();
            return fail("Stopped");
        }
        report(StageConverting, consumed, total_jobs);
        if (auto r = converter->Next(true)) {
            consume(std::move(*r));
        }
    }
    report(StageConverting, total_jobs, total_jobs);
    converter.reset();

    report(StageWriting, 0, 1);
    compute_out.close();
    if (!compute_out) {
        return fail("Writing the precache file failed");
    }
    // A new file: the count of the old one no longer applies, and its marker only when the
    // contents changed (the marker also carries the driver and variant, checked at boot).
    DropStaleWarmMarker(compute_path, compute_tmp);
    std::error_code ignored;
    fs::remove(WarmCountPath(compute_path), ignored);
    fs::rename(compute_tmp, compute_path, ec);
    if (ec) {
        return fail("Cannot replace " + Common::FS::PathToUTF8String(compute_path));
    }
    WriteWarmCount(compute_path, result.compute_programs);
    report(StageWriting, 1, 1);
    result.ok = true;
    LOG_INFO(Render_Vulkan,
             "Shader precache build for {:016x}: {} compute entries, {} programs skipped, {} "
             "archive and {} member errors, {:.1f} s",
             title_id, result.compute_programs, result.skipped, archive_errors, member_errors,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return result;
}

} // namespace VideoCommon::ShaderWarm
