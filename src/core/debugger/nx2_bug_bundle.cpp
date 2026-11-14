// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <iterator>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>

#include <boost/algorithm/string/replace.hpp>
#include <fmt/format.h>

#include "common/cpu_features.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/debugger/nx2_bug_bundle.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/debugger/nx2_zip.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/loader/loader.h"
#include "video_core/gpu.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/renderer_base.h"

namespace Core::Nx2Debug {
namespace {

using nlohmann::json;
using namespace std::chrono_literals;

constexpr u64 MiB = 1ULL << 20;
/// Room kept for bundle.json and README.md, which are written last.
constexpr u64 MANIFEST_RESERVE = 256 * 1024;
/// A log over this keeps its first LOG_HEAD and last LOG_TAIL bytes.
constexpr u64 LOG_TRIM = 8 * MiB;
constexpr u64 LOG_HEAD = 1 * MiB;
constexpr u64 LOG_TAIL = 6 * MiB;
/// Frame times kept: the last this many seconds of guest presents.
constexpr u64 FRAME_TIME_SECONDS = 10;
/// Capture ids of bug bundles, apart from the debug server's capture_frame ids.
constexpr u64 CAPTURE_ID_BIT = 1ULL << 63;
std::atomic<u64> captures{};

constexpr std::string_view README = R"(# Lichii bug bundle

Written by Lichii's "Report a bug" (hotkey or Help menu) while the bug was on screen. It holds game
textures and shaders: keep it private to the maintainers.

- bundle.json: start here. Build, game, GPU, what the frame capture holds, frame times, and the
  parts left out to stay under the size limit ("dropped").
- system.json, settings.txt, mods.json: the machine, every setting (with this game's overrides),
  and the game's update, DLC and mods.
- screenshot.png: the frame on screen.
- frame/: a frame dump of the next frame (EDEN_NX2_FRAME_DUMP with _SAMPLED). manifest.jsonl lists
  every pass's colour targets, compute storage images, sampled textures and the present source,
  with shader hashes, formats and per-channel stats. Previews are JPEGs, downscaled; stats are
  from the full-precision data.
- frame/shaders/: shaders.jsonl lists each pipeline the frame used, with its key, SPIR-V and
  native programs (named as EDEN_NX2_SHADER_DUMP names them).
- frame_times.json: guest present intervals over the last seconds (Switch 2 games).
- logs/: eden_log.txt and eden_log.txt.old, trimmed in the middle when long.

Browse it with: python3 -I tools/nx2/bundle_index.py <this zip>
)";

std::string LocalStamp(std::time_t time) {
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    return fmt::format("{:04}{:02}{:02}-{:02}{:02}{:02}", local.tm_year + 1900, local.tm_mon + 1,
                       local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec);
}

std::string UtcStamp(std::time_t time) {
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    return fmt::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", utc.tm_year + 1900, utc.tm_mon + 1,
                       utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec);
}

/// The release tag (build-177), or dev-<sha> for a build without one.
std::string BuildTag() {
    const std::string_view tag{Common::g_build_release_tag};
    std::string out = !tag.empty() && !tag.starts_with('@')
                          ? std::string{tag}
                          : fmt::format("dev-{}", std::string_view{Common::g_scm_rev}.substr(0, 10));
    std::ranges::replace_if(
        out, [](char c) { return !std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '.'; },
        '_');
    return out;
}

/// Replaces the user's name the way the file log does.
std::string Redact(std::string text) {
    for (const char* const variable : {"LOGNAME", "USERNAME", "USER"}) {
        if (const char* const name = std::getenv(variable); name != nullptr && *name != '\0') {
            boost::replace_all(text, name, "user");
        }
    }
    return text;
}

std::string SettingsReport() {
    std::string out;
    for (const auto& [category, settings] : Settings::values.linkage.by_category) {
        if (category == Settings::Category::WebService ||
            category == Settings::Category::Multiplayer) {
            continue;
        }
        for (const Settings::BasicSetting* const setting : settings) {
            const std::string& label = setting->GetLabel();
            if (label.find("token") != std::string::npos ||
                label.find("password") != std::string::npos ||
                label.find("username") != std::string::npos) {
                continue;
            }
            out += fmt::format("[{}] {} = {}", Settings::TranslateCategory(category), label,
                               setting->ToString());
            if (setting->Switchable() && !setting->UsingGlobal()) {
                out += fmt::format(" (global: {})", setting->ToStringGlobal());
            }
            out += '\n';
        }
    }
    return Redact(std::move(out));
}

std::vector<u8> ReadFile(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

/// The text after "<key>" on the first line holding it, in the head of the log.
std::string LogValue(std::string_view head, std::string_view key) {
    const size_t at = head.find(key);
    if (at == std::string_view::npos) {
        return {};
    }
    const size_t begin = at + key.size();
    const size_t end = head.find_first_of("\r\n", begin);
    return std::string{head.substr(begin, end == std::string_view::npos ? end : end - begin)};
}

/// A log, or its first and last parts when it is long.
std::pair<std::vector<u8>, std::string> TrimmedLog(std::vector<u8> log) {
    if (log.size() <= LOG_TRIM) {
        return {std::move(log), {}};
    }
    // Cut at line ends, so no word (a redacted user name included) is split.
    auto head_end = log.begin() + LOG_HEAD;
    head_end = std::find(std::make_reverse_iterator(head_end), log.rend(), u8{'\n'}).base();
    auto tail_begin = std::find(log.end() - LOG_TAIL, log.end(), u8{'\n'});
    tail_begin = tail_begin == log.end() ? tail_begin : tail_begin + 1;
    const std::string marker = fmt::format(
        "[... {:.1f} MiB cut from the middle by the bug report ...]\n",
        static_cast<double>(tail_begin - head_end) / MiB);
    std::vector<u8> out(log.begin(), head_end);
    out.insert(out.end(), marker.begin(), marker.end());
    out.insert(out.end(), tail_begin, log.end());
    return {std::move(out), fmt::format("trimmed: first 1 MiB and last 6 MiB of {:.1f} MiB",
                                        static_cast<double>(log.size()) / MiB)};
}

std::string_view PatchTypeName(FileSys::PatchType type) {
    switch (type) {
    case FileSys::PatchType::Update:
        return "update";
    case FileSys::PatchType::DLC:
        return "dlc";
    case FileSys::PatchType::Mod:
        return "mod";
    }
    return "other";
}

std::vector<u8> Bytes(std::string_view text) {
    return {text.begin(), text.end()};
}

} // Anonymous namespace

std::filesystem::path BugBundle::DefaultDirectory() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "bug-reports";
}

BugBundle::BugBundle(System& system_, Options options_)
    : system{system_}, options{std::move(options_)} {
    if (options.directory.empty()) {
        options.directory = DefaultDirectory();
    }
    const std::time_t now = std::time(nullptr);
    const u64 program_id = system.GetApplicationProcessProgramID();
    file_name =
        fmt::format("lichii-bug-{}-{:016X}-{}.zip", BuildTag(), program_id, LocalStamp(now));
    work = options.directory / fmt::format(".work-{}-{}", LocalStamp(now), captures.load() + 1);
    report = {
        {"format", "lichii-bug-bundle"},
        {"version", 1},
        {"created", UtcStamp(now)},
        {"trigger", options.trigger},
        {"build",
         {{"tag", BuildTag()},
          {"name", std::string{Common::g_build_fullname}},
          {"version", std::string{Common::g_build_version}},
          {"scm_rev", std::string{Common::g_scm_rev}},
          {"branch", std::string{Common::g_scm_branch}}}},
    };
}

BugBundle::~BugBundle() {
    std::error_code error;
    std::filesystem::remove_all(work, error);
}

void BugBundle::Capture(const std::atomic<bool>& stop) {
    std::filesystem::create_directories(work);
    const u64 program_id = system.GetApplicationProcessProgramID();
    const bool vulkan =
        Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    const u64 id = CAPTURE_ID_BIT | ++captures;
    if (vulkan) {
        VideoCore::Nx2Debug::RequestCapture({.id = id,
                                             .directory = work / "frame",
                                             .sampled = true,
                                             .buffers = false,
                                             .shaders = true,
                                             .raw = false,
                                             .size_cap = 1ULL << 30,
                                             .preview_max = 512,
                                             .preview_jpeg = true});
    } else {
        frame_error = "frame capture needs the Vulkan renderer";
    }
    try {
        screenshot = CaptureScreenshot(system, 1080, 5s, stop).bytes;
    } catch (const std::exception& failure) {
        screenshot_error = failure.what();
    }

    // The report, while the capture runs.
    const FileSys::PatchManager patches{program_id, system.GetFileSystemController(),
                                        system.GetContentProvider()};
    std::string title;
    std::string version;
    if (const auto metadata = patches.GetControlMetadata(); metadata.first) {
        title = metadata.first->GetApplicationName();
        version = metadata.first->GetVersionString();
    }
    if (title.empty()) {
        void(system.GetGameName(title));
    }
    FileSys::VirtualFile update_raw;
    void(system.GetAppLoader().ReadUpdateRaw(update_raw));
    mods = json::array();
    for (const FileSys::Patch& patch : patches.GetPatches(update_raw)) {
        mods.push_back({{"name", patch.name},
                        {"version", patch.version},
                        {"type", PatchTypeName(patch.type)},
                        {"enabled", patch.enabled}});
    }
    report["game"] = {{"title", title},
                      {"program_id", fmt::format("{:016X}", program_id)},
                      {"version", version},
                      {"switch2", Settings::CurrentProgramIsSwitch2()}};

    Common::Log::Flush();
    std::string head(64 * 1024, '\0');
    {
        std::ifstream log{Common::Log::FilePath(), std::ios::binary};
        log.read(head.data(), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<size_t>(log.gcount()));
    }
    std::string cpu = LogValue(head, "Host CPU: ");
#ifdef ARCHITECTURE_x86_64
    if (cpu.empty()) {
        cpu = std::string{Common::g_cpu_caps.cpu_string,
                          strnlen(Common::g_cpu_caps.cpu_string, sizeof(Common::g_cpu_caps.cpu_string))};
    }
#endif
    const auto& device = system.GPU().Renderer().GetDeviceInfo();
    report["system"] = {{"os", LogValue(head, "Host OS: ")},
                        {"cpu", cpu},
                        {"threads", std::thread::hardware_concurrency()},
                        {"ram_mib", Common::GetMemInfo().TotalPhysicalMemory / MiB}};
    report["gpu"] = {{"renderer", vulkan ? "vulkan" : "other"},
                     {"device", device.model},
                     {"driver", device.driver},
                     {"api", device.api_version.empty() ? device.api : device.api_version}};
    json features = json::object();
    for (const auto& [feature, supported] : device.features) {
        features[feature] = supported;
    }
    system_report = {{"system", report["system"]},
                     {"gpu", report["gpu"]},
                     {"device_local_memory_mib", device.device_local_memory / MiB},
                     {"docked", Settings::IsDockedMode()},
                     {"features", std::move(features)},
                     {"extensions", device.extensions}};
    settings = SettingsReport();

    const u64 presents = g_nx2_present_count.load(std::memory_order_relaxed);
    const u64 capacity = g_nx2_present_times_ns.size();
    const u64 end_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
    frame_times = ReadFrameTimes(presents > capacity ? presents - capacity : 0, presents,
                                 end_ns - FRAME_TIME_SECONDS * 1'000'000'000ULL, end_ns, 60.0,
                                 capacity);
    frame_times["seconds"] = FRAME_TIME_SECONDS;

    if (!vulkan) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + options.capture_timeout;
    for (;;) {
        if (const auto finished = VideoCore::Nx2Debug::WaitCapture(id, 250ms)) {
            frame_error = finished->error;
            return;
        }
        if (stop.load(std::memory_order_relaxed) || std::chrono::steady_clock::now() >= deadline) {
            VideoCore::Nx2Debug::CancelCapture(id);
            frame_error = stop.load(std::memory_order_relaxed)
                              ? "the game stopped before the frame capture finished"
                              : fmt::format("no frame capture within {} s (paused or loading?)",
                                            options.capture_timeout.count());
            return;
        }
    }
}

BugBundle::Result BugBundle::Write(std::string_view description) {
    std::filesystem::create_directories(options.directory);
    Result result{.path = options.directory / file_name};
    auto partial = result.path;
    partial += ".partial";
    SCOPE_EXIT {
        std::error_code error;
        std::filesystem::remove(partial, error);
    };
    ZipWriter zip{partial};
    json parts = json::array();
    json dropped = json::array();
    const u64 budget =
        options.size_limit > MANIFEST_RESERVE ? options.size_limit - MANIFEST_RESERVE : 0;

    const auto add = [&](std::string name, std::span<const u8> data, bool compress,
                         std::string note = {}) {
        const ZipWriter::Entry entry = ZipWriter::Prepare(std::move(name), data, compress);
        if (zip.Size() + entry.ZipBytes() > budget) {
            return false;
        }
        zip.Add(entry);
        json part = {{"path", entry.name}, {"bytes", entry.data.size()}};
        if (!note.empty()) {
            part["note"] = std::move(note);
        }
        parts.push_back(std::move(part));
        return true;
    };
    const auto drop = [&](std::string what, size_t files, u64 bytes) {
        result.dropped.push_back(fmt::format("{}: {} files, {:.1f} MiB", what, files,
                                             static_cast<double>(bytes) / MiB));
        dropped.push_back({{"what", std::move(what)},
                           {"files", files},
                           {"bytes", bytes},
                           {"reason", "size limit"}});
    };
    /// Adds the files in order while they fit; the rest are listed as dropped.
    const auto add_group = [&](std::string_view what,
                               const std::vector<std::pair<std::string, std::filesystem::path>>& files,
                               bool compress) {
        size_t left = 0;
        u64 left_bytes = 0;
        for (const auto& [name, path] : files) {
            const std::vector<u8> data = ReadFile(path);
            if (left != 0 || !add(name, data, compress)) {
                ++left;
                left_bytes += data.size();
            }
        }
        if (left != 0) {
            drop(std::string{what}, left, left_bytes);
        }
    };

    add("system.json", Bytes(system_report.dump(2)), true);
    add("settings.txt", Bytes(settings), true);
    add("mods.json", Bytes(mods.dump(2)), true);
    add("frame_times.json", Bytes(frame_times.dump(2)), true);
    if (!screenshot.empty() && !add("screenshot.png", screenshot, false)) {
        drop("screenshot", 1, screenshot.size());
    }

    const auto frame_dir = work / "frame";
    json frame = {{"captured", false}, {"error", frame_error}};
    std::map<std::string, std::vector<std::pair<std::string, std::filesystem::path>>, std::less<>>
        images;
    if (frame_error.empty() && std::filesystem::exists(frame_dir / "manifest.jsonl")) {
        frame = {{"captured", true}, {"error", ""}, {"preview_max", 512}, {"preview", "jpeg"}};
        std::ifstream manifest{frame_dir / "manifest.jsonl"};
        for (std::string line; std::getline(manifest, line);) {
            const json entry = json::parse(line, nullptr, false);
            if (entry.is_discarded()) {
                continue;
            }
            const std::string kind = entry.value("kind", std::string{});
            if (kind == "frame") {
                frame["frame"] = entry.value("frame", u64{0});
            } else if (kind == "summary") {
                for (const char* const key :
                     {"images", "passes", "draws", "dispatches", "pipelines", "programs", "truncated"}) {
                    if (entry.contains(key)) {
                        frame[key] = entry[key];
                    }
                }
            } else if (entry.contains("preview") && entry["preview"].is_string()) {
                const std::string preview = entry["preview"].get<std::string>();
                images[kind].emplace_back("frame/" + preview, frame_dir / preview);
            }
        }
        add("frame/manifest.jsonl", ReadFile(frame_dir / "manifest.jsonl"), true);
        if (std::filesystem::exists(frame_dir / "shaders" / "shaders.jsonl")) {
            add("frame/shaders/shaders.jsonl", ReadFile(frame_dir / "shaders" / "shaders.jsonl"),
                true);
        }
    } else if (frame_error.empty()) {
        frame["error"] = "the frame capture wrote no manifest";
    }
    result.frame_error = frame["error"].get<std::string>();

    Common::Log::Flush();
    const auto log_path = Common::Log::FilePath();
    auto old_log_path = log_path;
    old_log_path += ".old.txt";
    const auto add_log = [&](const char* name, const std::filesystem::path& path) {
        if (path.empty() || !std::filesystem::exists(path)) {
            return;
        }
        auto [trimmed, note] = TrimmedLog(ReadFile(path));
        const std::vector<u8> log = Bytes(Redact(std::string{trimmed.begin(), trimmed.end()}));
        if (path.filename() != "eden_log.txt" && path.filename() != "eden_log.txt.old.txt") {
            note += fmt::format("{}from {}", note.empty() ? "" : "; ", path.filename().string());
        }
        if (!add(fmt::format("logs/{}", name), log, true, std::move(note))) {
            drop(fmt::format("logs/{}", name), 1, log.size());
        }
    };
    add_log("eden_log.txt", log_path);

    const auto shader_files = [&](std::string_view folder) {
        std::vector<std::pair<std::string, std::filesystem::path>> files;
        std::error_code error;
        for (const auto& item :
             std::filesystem::directory_iterator{frame_dir / "shaders" / folder, error}) {
            files.emplace_back(fmt::format("frame/shaders/{}/{}", folder,
                                           item.path().filename().generic_string()),
                               item.path());
        }
        std::ranges::sort(files);
        return files;
    };
    add_group("native programs", shader_files("native"), true);
    add_group("pipeline keys and QMDs", shader_files("pipelines"), true);
    add_group("SPIR-V", shader_files("spirv"), true);
    add_group("frame present source", images["present"], false);
    add_group("frame pass colour targets", images["pass"], false);
    add_group("frame compute storage images", images["dispatch"], false);
    add_group("frame sampled textures", images["sampled"], false);
    add_log("eden_log.txt.old", old_log_path);

    if (!screenshot_error.empty()) {
        report["screenshot_error"] = screenshot_error;
    }
    report["description"] = std::string{description};
    report["frame"] = std::move(frame);
    report["frame_times"] = {{"seconds", FRAME_TIME_SECONDS},
                             {"frames", frame_times.value("frames", 0)},
                             {"fps", frame_times.value("fps", 0.0)},
                             {"p99_ms", frame_times.value("p99_ms", 0.0)},
                             {"stutters", frame_times.value("stutters", 0)}};
    report["size_limit"] = options.size_limit;
    report["dropped"] = std::move(dropped);
    zip.Add(ZipWriter::Prepare("README.md", Bytes(README), true));
    report["parts"] = std::move(parts);
    // bundle.json states the zip's final size, which includes bundle.json itself.
    u64 total = zip.Size();
    for (u64 previous = 0; previous != total;) {
        previous = total;
        report["bytes"] = total;
        total = zip.Size() + ZipWriter::Prepare("bundle.json", Bytes(report.dump(2)), false).ZipBytes();
    }
    zip.Add(ZipWriter::Prepare("bundle.json", Bytes(report.dump(2)), false));
    zip.Finish();
    std::filesystem::rename(partial, result.path);
    result.bytes = std::filesystem::file_size(result.path);
    result.manifest = report;
    LOG_INFO(Debug, "Bug report written: {} ({:.1f} MiB){}", result.path.generic_string(),
             static_cast<double>(result.bytes) / MiB,
             result.frame_error.empty() ? "" : fmt::format(", no frame capture: {}", result.frame_error));
    return result;
}

} // namespace Core::Nx2Debug
