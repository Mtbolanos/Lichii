// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <iterator>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include "common/settings_enums.h"
#include "core/launch_params.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>

#include <fmt/ostream.h>

#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/cpu_manager.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/nup_writer.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/loader.h"
#include "frontend_common/config.h"
#include "frontend_common/profile_import.h"
#include "input_common/drivers/keyboard.h"
#include "input_common/main.h"
#include "network/network.h"
#include "sdl_config.h"
#include "video_core/gpu.h"
#include "video_core/gpu_trace/replay.h"
#include "video_core/gpu_trace/sample.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "lichii_cmd/emu_window/emu_window_sdl3.h"
#ifdef HAS_OPENGL
#include "lichii_cmd/emu_window/emu_window_sdl3_gl.h"
#endif
#include "lichii_cmd/emu_window/emu_window_sdl3_null.h"
#include "lichii_cmd/emu_window/emu_window_sdl3_vk.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#ifdef _WIN32
// windows.h needs to be included before shellapi.h
#include <windows.h>
#include <shellapi.h>
#include "common/windows/crash_handler.h"
#include "common/windows/timer_resolution.h"
#endif

#ifdef _WIN32
extern "C" {
// tells Nvidia and AMD drivers to use the dedicated GPU by default on laptops with switchable
// graphics
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

static void OnStateChanged(const Network::RoomMember::State& state) {
    switch (state) {
    case Network::RoomMember::State::Idle:
        LOG_DEBUG(Network, "Network is idle");
        break;
    case Network::RoomMember::State::Joining:
        LOG_DEBUG(Network, "Connection sequence to room started");
        break;
    case Network::RoomMember::State::Joined:
        LOG_DEBUG(Network, "Successfully joined to the room");
        break;
    case Network::RoomMember::State::Moderator:
        LOG_DEBUG(Network, "Successfully joined the room as a moderator");
        break;
    default:
        break;
    }
}

static void OnNetworkError(const Network::RoomMember::Error& error) {
    switch (error) {
    case Network::RoomMember::Error::LostConnection:
        LOG_DEBUG(Network, "Lost connection to the room");
        break;
    case Network::RoomMember::Error::CouldNotConnect:
        LOG_ERROR(Network, "Error: Could not connect");
        exit(1);
        break;
    case Network::RoomMember::Error::NameCollision:
        LOG_ERROR(
            Network,
            "You tried to use the same nickname as another user that is connected to the Room");
        exit(1);
        break;
    case Network::RoomMember::Error::IpCollision:
        LOG_ERROR(Network, "You tried to use the same fake IP-Address as another user that is "
                           "connected to the Room");
        exit(1);
        break;
    case Network::RoomMember::Error::WrongPassword:
        LOG_ERROR(Network, "Room replied with: Wrong password");
        exit(1);
        break;
    case Network::RoomMember::Error::WrongVersion:
        LOG_ERROR(Network,
                  "You are using a different version than the room you are trying to connect to");
        exit(1);
        break;
    case Network::RoomMember::Error::RoomIsFull:
        LOG_ERROR(Network, "The room is full");
        exit(1);
        break;
    case Network::RoomMember::Error::HostKicked:
        LOG_ERROR(Network, "You have been kicked by the host");
        break;
    case Network::RoomMember::Error::HostBanned:
        LOG_ERROR(Network, "You have been banned by the host");
        break;
    case Network::RoomMember::Error::UnknownError:
        LOG_ERROR(Network, "UnknownError");
        break;
    case Network::RoomMember::Error::PermissionDenied:
        LOG_ERROR(Network, "PermissionDenied");
        break;
    case Network::RoomMember::Error::NoSuchUser:
        LOG_ERROR(Network, "NoSuchUser");
        break;
    }
}

static void OnMessageReceived(const Network::ChatEntry& msg) {
    std::cout << std::endl << msg.nickname << ": " << msg.message << std::endl << std::endl;
}

static void OnStatusMessageReceived(const Network::StatusMessageEntry& msg) {
    std::string message = [&]() {
        switch (msg.type) {
        case Network::IdMemberJoin:
            return fmt::format("{} has joined", msg.nickname);
        case Network::IdMemberLeave:
            return fmt::format("{} has left", msg.nickname);
        case Network::IdMemberKicked:
            return fmt::format("{} has been kicked", msg.nickname);
        case Network::IdMemberBanned:
            return fmt::format("{} has been banned", msg.nickname);
        case Network::IdAddressUnbanned:
            return fmt::format("{} has been unbanned", msg.nickname);
        default:
            return std::string{};
        }
    }();
    if (!message.empty())
        std::cout << std::endl << "* " << message << std::endl << std::endl;
}

struct SdlState {
    Core::System system{};
    // Must outlive emu_window, which keeps a pointer to it.
    InputCommon::InputSubsystem input_subsystem{};
    std::unique_ptr<EmuWindow_SDL3> emu_window;
    bool network_initialized{};
};

constexpr int kRestartExitCode = 75;
static std::atomic_bool restart_requested{false};

// ExecuteProgram state for the restarted process: what the Qt frontend keeps in-process when it
// reboots a game. Only read when the launcher sets EDEN_CLI_RESTART_HANDOFF, so a leftover file
// never changes a normal launch.
struct RestartHandoff {
    u64 program_id{};
    s32 program_index{};
    s32 previous_program_index{-1};
    std::vector<std::vector<u8>> user_channel;
};

static std::filesystem::path RestartHandoffPath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir) / "restart-handoff.bin";
}

template <typename T>
static void PutRaw(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
static void GetRaw(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
}

// Bounds keep a damaged handoff from requesting huge allocations; the writer refuses anything the
// reader would reject.
constexpr u32 kHandoffMaxEntries = 256;
constexpr u64 kHandoffMaxTotalSize = 64u << 20;

static bool WriteRestartHandoff(const RestartHandoff& handoff) {
    u64 total = 0;
    for (const auto& entry : handoff.user_channel) {
        total += entry.size();
    }
    if (handoff.user_channel.size() > kHandoffMaxEntries || total > kHandoffMaxTotalSize) {
        return false;
    }
    std::ofstream out{RestartHandoffPath(), std::ios::binary | std::ios::trunc};
    PutRaw(out, handoff.program_id);
    PutRaw(out, handoff.program_index);
    PutRaw(out, handoff.previous_program_index);
    PutRaw(out, static_cast<u32>(handoff.user_channel.size()));
    for (const auto& entry : handoff.user_channel) {
        PutRaw(out, static_cast<u32>(entry.size()));
        out.write(reinterpret_cast<const char*>(entry.data()), std::streamsize(entry.size()));
    }
    out.close();
    return !out.fail();
}

// Leaves the file in place; the caller removes it once the game has loaded.
static std::optional<RestartHandoff> ReadRestartHandoff() {
    std::ifstream in{RestartHandoffPath(), std::ios::binary};
    RestartHandoff handoff;
    u32 count{};
    GetRaw(in, handoff.program_id);
    GetRaw(in, handoff.program_index);
    GetRaw(in, handoff.previous_program_index);
    GetRaw(in, count);
    if (in.fail() || count > kHandoffMaxEntries) {
        return std::nullopt;
    }
    u64 total = 0;
    for (u32 i = 0; i < count; ++i) {
        u32 size{};
        GetRaw(in, size);
        total += size;
        if (in.fail() || total > kHandoffMaxTotalSize) {
            return std::nullopt;
        }
        auto& entry = handoff.user_channel.emplace_back(size);
        in.read(reinterpret_cast<char*>(entry.data()), std::streamsize(size));
        if (in.fail()) {
            return std::nullopt;
        }
    }
    return handoff;
}

static std::atomic_bool pack_nup_cancel{false};
static std::atomic_bool unpack_nup_cancel{false};
static std::atomic_bool edit_nup_cancel{false};

// eden-cli --pack-nup <dump> [<out.nup>]; the output defaults to <dump>.nup next to the dump.
static bool PackNup(const std::string& source, const std::string& output) {
    const auto to_path = [](const std::string& utf8) {
        return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
    };
    const auto dump_root = to_path(source);
    const auto archive =
        output.empty() ? FileSys::Nup::DefaultArchivePath(dump_root) : to_path(output);
    std::signal(SIGINT, [](int) { pack_nup_cancel = true; });

    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;
    std::string error;
    const auto result = FileSys::Nup::PackDump(
        dump_root, archive, {},
        [&](const FileSys::Nup::PackProgress& p) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_report < std::chrono::seconds(1) && p.files_done != p.files_total) {
                return;
            }
            last_report = now;
            fmt::print(stderr, "\r{} / {} MiB, {} / {} files", p.bytes_done >> 20,
                       p.bytes_total >> 20, p.files_done, p.files_total);
        },
        &pack_nup_cancel, &error);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::print(stderr, "\n");
    switch (result) {
    case FileSys::Nup::PackResult::Success:
        fmt::print(stderr, "Wrote {} in {:.1f} s\n", archive.string(), seconds);
        return true;
    case FileSys::Nup::PackResult::Cancelled:
        fmt::print(stderr, "Cancelled\n");
        return false;
    default:
        fmt::print(stderr, "Packing failed: {}\n", error);
        return false;
    }
}

// eden-cli --unpack-nup <file.nup> [<out_dir>]; the output defaults to <name>/ next to the NUP.
static bool UnpackNup(const std::string& source, const std::string& output) {
    const auto to_path = [](const std::string& utf8) {
        return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
    };
    const auto archive = to_path(source);
    const auto out_dir =
        output.empty() ? FileSys::Nup::DefaultExtractPath(archive) : to_path(output);
    std::signal(SIGINT, [](int) { unpack_nup_cancel = true; });

    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;
    std::string error;
    const auto result = FileSys::Nup::ExtractNup(
        archive, out_dir,
        [&](const FileSys::Nup::PackProgress& p) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_report < std::chrono::seconds(1) && p.files_done != p.files_total) {
                return;
            }
            last_report = now;
            fmt::print(stderr, "\r{} / {} MiB, {} / {} files", p.bytes_done >> 20,
                       p.bytes_total >> 20, p.files_done, p.files_total);
        },
        &unpack_nup_cancel, &error);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::print(stderr, "\n");
    switch (result) {
    case FileSys::Nup::ExtractResult::Success:
        fmt::print(stderr, "Extracted to {} in {:.1f} s\n", out_dir.string(), seconds);
        return true;
    case FileSys::Nup::ExtractResult::Cancelled:
        fmt::print(stderr, "Cancelled\n");
        return false;
    default:
        fmt::print(stderr, "Extraction failed: {}\n", error);
        return false;
    }
}

static bool EditNup(const Core::LaunchParams& lp) {
    const auto to_path = [](const std::string& utf8) {
        return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
    };
    const auto archive = to_path(*lp.edit_nup_source);
    FileSys::Nup::MetadataEdit edit;
    if (lp.edit_nup_name) {
        edit.title_name = *lp.edit_nup_name;
    }
    if (lp.edit_nup_publisher) {
        edit.publisher = *lp.edit_nup_publisher;
    }
    if (lp.edit_nup_version) {
        edit.display_version = *lp.edit_nup_version;
    }
    if (lp.edit_nup_icon) {
        if (lp.edit_nup_icon->empty()) {
            edit.icon = std::vector<u8>{};
        } else {
            const auto icon_path = to_path(*lp.edit_nup_icon);
            std::ifstream in(icon_path, std::ios::binary);
            if (!in) {
                fmt::print(stderr, "Cannot read icon {}\n", icon_path.string());
                return false;
            }
            edit.icon = std::vector<u8>((std::istreambuf_iterator<char>(in)), {});
            if (edit.icon->empty()) {
                fmt::print(stderr, "Icon file is empty\n");
                return false;
            }
        }
    }
    if (!edit.title_name && !edit.icon && !edit.display_version && !edit.publisher) {
        fmt::print(stderr, "Nothing to edit: pass --nup-name, --nup-icon, --nup-version or "
                           "--nup-publisher (or a --clear-* flag)\n");
        return false;
    }
    std::signal(SIGINT, [](int) { edit_nup_cancel = true; });
    std::string error;
    switch (FileSys::Nup::EditNupMetadata(archive, edit, &error, &edit_nup_cancel)) {
    case FileSys::Nup::EditResult::Success:
        fmt::print(stderr, "Updated {}\n", archive.string());
        return true;
    case FileSys::Nup::EditResult::NoChange:
        fmt::print(stderr, "No changes to apply\n");
        return true;
    case FileSys::Nup::EditResult::Cancelled:
        fmt::print(stderr, "Cancelled\n");
        return false;
    default:
        fmt::print(stderr, "Edit failed: {}\n", error);
        return false;
    }
}

/// Hidden test hook (not in --help): builds a game's shader precache file without booting it.
///   lichii-cmd --build-warm <game file> <title id hex> <output dir>
static bool BuildWarm(Core::System& system, int argc, char** argv) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    if (argc < 5) {
        fmt::print(stderr, "usage: --build-warm <game> <title id> <output dir>\n");
        return false;
    }
    system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
    system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
    const u64 title_id = std::strtoull(argv[3], nullptr, 16);
    ShaderWarm::BuildOptions options;
    options.output_dir = Common::FS::ToU8String(argv[4]);
    std::string last_stage;
    auto last_print = std::chrono::steady_clock::now();
    const auto begin = last_print;
    const auto on_progress = [&](const ShaderWarm::Progress& progress) {
        const auto now = std::chrono::steady_clock::now();
        if (progress.stage == last_stage && now - last_print < std::chrono::seconds{5}) {
            return;
        }
        last_stage = progress.stage;
        last_print = now;
        fmt::print("{:7.1f} s  {}: {} / {}\n",
                   std::chrono::duration<double>(now - begin).count(), progress.stage,
                   progress.done, progress.total);
        std::fflush(stdout);
    };
    const auto result = ShaderWarm::BuildWarmFilesWith(
        system, Common::FS::ToU8String(argv[2]), title_id, options, on_progress, {});
    // The error can hold a Windows path: escape it so the summary stays valid JSON.
    std::string error;
    for (const char c : result.error) {
        switch (c) {
        case '"':
            error += "\\\"";
            break;
        case '\\':
            error += "\\\\";
            break;
        case '\n':
            error += "\\n";
            break;
        case '\r':
            error += "\\r";
            break;
        case '\t':
            error += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                error += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                error += c;
            }
            break;
        }
    }
    fmt::print("{{\"ok\": {}, \"error\": \"{}\", \"archives\": {}, \"compute\": {}, "
               "\"skipped\": {}, \"seconds\": {:.1f}}}\n",
               result.ok, error, result.archives, result.compute_programs, result.skipped,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    std::fflush(stdout);
    return result.ok;
}

/// GPU trace replay (docs/nx2/gpu-trace.md):
///   eden-cli --replay <trace> --out <dir> [--frames N] [--device N]
///                     [--record <trace> --record-from <present> --record-presents N]
///   eden-cli --gpu-trace-sample <trace>: writes the synthetic trace.
static bool ReplayGpuTrace(Core::System& system, int argc, char** argv) {
    namespace GpuTrace = VideoCommon::GpuTrace;
    if (std::string_view{argv[1]} == "--gpu-trace-sample") {
        if (argc < 3) {
            fmt::print(stderr, "usage: --gpu-trace-sample <trace>\n");
            return false;
        }
        return GpuTrace::WriteSampleTrace(Common::FS::ToU8String(argv[2]));
    }
    GpuTrace::ReplayOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string_view argument{argv[i]};
        const bool has_value = i + 1 < argc;
        if (argument == "--out" && has_value) {
            options.out = Common::FS::ToU8String(argv[++i]);
        } else if (argument == "--frames" && has_value) {
            options.max_frames = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        } else if (argument == "--record" && has_value) {
            options.record = Common::FS::ToU8String(argv[++i]);
        } else if (argument == "--record-from" && has_value) {
            options.record_from = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        } else if (argument == "--record-presents" && has_value) {
            options.record_presents = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        } else if (argument == "--device" && has_value) {
            Settings::values.vulkan_device.SetValue(std::atoi(argv[++i]));
        } else if (options.trace.empty()) {
            options.trace = Common::FS::ToU8String(argv[i]);
        } else {
            fmt::print(stderr, "unknown argument {}\n", argument);
            return false;
        }
    }
    if (options.trace.empty() || options.out.empty()) {
        fmt::print(stderr, "usage: --replay <trace> --out <dir> [--frames N] [--device N]\n");
        return false;
    }
    std::string error;
    if (!GpuTrace::Replay(system, options, error)) {
        fmt::print(stderr, "replay failed: {}\n", error);
        return false;
    }
    return true;
}

extern "C" SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {
    // SDL calls SDL_AppIterate back to back by default, which kept a host core at 100% polling
    // input; 240 Hz is plenty for input and the key script.
    SDL_SetHint(SDL_HINT_MAIN_CALLBACK_RATE, "240");

#ifdef _WIN32
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        freopen("CONOUT$", "wb", stdout);
        freopen("CONOUT$", "wb", stderr);
    }
    // Raise the CRT's open-file limit like the Qt frontend: loose RomFS folders keep thousands of
    // files open through the VFS cache.
    _setmaxstdio(8192);
#elif defined(__unix__) || defined(__APPLE__)
    // Same for the open file limit (macOS shells default to 256).
    if (struct rlimit rl; getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        rl.rlim_cur = std::min<rlim_t>(8192, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }
#endif

    try {
        FrontendCommon::CheckLegacyProfileBeforeConfig();
    } catch (const std::exception& exception) {
        fmt::print(stderr, "{}\n", exception.what());
        return SDL_APP_FAILURE;
    }

    SdlState* state = new SdlState();
    *appstate = state;

    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();
#ifdef _WIN32
    // EDEN_NX2_CRASH_DUMP=<dir> puts the dump somewhere other than the log directory.
    const char* const crash_dump_dir = std::getenv("EDEN_NX2_CRASH_DUMP");
    Common::Windows::InstallCrashHandler(
        crash_dump_dir ? std::filesystem::path{crash_dump_dir}
                       : Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir));
#endif

    if (argc > 1 && std::string_view{argv[1]} == "--build-warm") {
        return BuildWarm(state->system, argc, argv) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }
    if (argc > 1 && (std::string_view{argv[1]} == "--replay" ||
                     std::string_view{argv[1]} == "--gpu-trace-sample")) {
        return ReplayGpuTrace(state->system, argc, argv) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

    int option_index = 0;
#ifdef _WIN32
    int argc_w;
    auto argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    if (argv_w == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to get command line arguments");
        return SDL_APP_FAILURE;
    }
#else
    wchar_t **argv_w = nullptr;
#endif
    Core::LaunchParams lp = Core::ParseLaunchParams(state->system, argc, argv, argv_w);
#ifdef _WIN32
    LocalFree(argv_w);
#endif
    SdlConfig config{lp.config_path};

    Core::ApplyLaunchParams(lp);

    // Informational options already printed by ApplyLaunchParams; handle them before any
    // archive operation so --help/--version never packs, unpacks or edits.
    if (lp.print_help || lp.print_version) {
        return SDL_APP_SUCCESS;
    }

    if (lp.pack_nup_source) {
        return PackNup(*lp.pack_nup_source, lp.filepath) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

    if (lp.unpack_nup_source) {
        return UnpackNup(*lp.unpack_nup_source, lp.filepath) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

    if (lp.edit_nup_source) {
        return EditNup(lp) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

    if (lp.filepath.empty()) {
        LOG_CRITICAL(Frontend, "Failed to load ROM: No ROM specified");
        return SDL_APP_FAILURE;
    }

    state->network_initialized = Network::Init();
    if (!state->network_initialized) {
        LOG_CRITICAL(Network, "Failed to initialize room networking");
        return SDL_APP_FAILURE;
    }

    state->system.Initialize();


    // Apply the command line arguments
    state->system.ApplySettings();

    switch (Settings::values.renderer_backend.GetValue()) {
#ifdef HAS_OPENGL
    case Settings::RendererBackend::OpenGL_GLSL:
    case Settings::RendererBackend::OpenGL_GLASM:
    case Settings::RendererBackend::OpenGL_SPIRV:
        state->emu_window = std::make_unique<EmuWindow_SDL3_GL>(&state->input_subsystem, state->system, lp.fullscreen);
        break;
#endif
    case Settings::RendererBackend::Vulkan:
        state->emu_window = std::make_unique<EmuWindow_SDL3_VK>(&state->input_subsystem, state->system, lp.fullscreen);
        break;
    case Settings::RendererBackend::Null:
        state->emu_window = std::make_unique<EmuWindow_SDL3_Null>(&state->input_subsystem, state->system, lp.fullscreen);
        break;
    default:
        LOG_CRITICAL(Frontend, "Invalid renderer backend");
        return SDL_APP_FAILURE;
    }

#ifdef _WIN32
    Common::Windows::SetCurrentTimerResolutionToMaximum();
    state->system.CoreTiming().SetTimerResolutionNs(Common::Windows::GetCurrentTimerResolution());
#endif

    state->system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
    // Holds the updates and DLC an NX archive brings, as the Qt frontend does.
    static FileSys::ManualContentProvider manual_provider;
    state->system.RegisterContentProvider(FileSys::ContentProviderUnionSlot::FrontendManual,
                                          &manual_provider);
    state->system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
    state->system.GetFileSystemController().CreateFactories(*state->system.GetFilesystem());
    state->system.GetUserChannel().clear();

    Service::AM::FrontendAppletParameters load_parameters{
        .applet_id = Service::AM::AppletId::Application,
    };
    const char* const handoff_env = std::getenv("EDEN_CLI_RESTART_HANDOFF");
    const bool restarted = handoff_env != nullptr && handoff_env[0] != '\0';
    if (restarted) {
        auto handoff = ReadRestartHandoff();
        if (!handoff) {
            // Booting as a frontend launch would clear the game's temp storage.
            LOG_CRITICAL(Frontend, "Restart handoff {} is missing or damaged, not booting",
                         Common::FS::PathToUTF8String(RestartHandoffPath()));
            return SDL_APP_FAILURE;
        }
        state->system.GetUserChannel() = std::move(handoff->user_channel);
        load_parameters.program_id = handoff->program_id;
        load_parameters.launch_type = Service::AM::LaunchType::ApplicationInitiated;
        load_parameters.program_index = handoff->program_index;
        load_parameters.previous_program_index = handoff->previous_program_index;
        LOG_INFO(Frontend, "Restarted by the game: program index {}, {} user channel entries",
                 handoff->program_index, state->system.GetUserChannel().size());
    }
    const Core::SystemResultStatus load_result = state->system.Load(*state->emu_window, lp.filepath, load_parameters);
    switch (load_result) {
    case Core::SystemResultStatus::Success:
        if (restarted) {
            std::error_code ec;
            std::filesystem::remove(RestartHandoffPath(), ec);
        } else if (const auto ram = Common::GetHostRam();
                   Settings::CurrentProgramIsSwitch2() && Common::IsBelowSwitch2RamMinimum(ram)) {
            using namespace Common::Literals;
            fmt::print(stderr,
                       "Warning: this PC has {:.3g} GB of RAM. Switch 2 games need at least 16 GB, "
                       "and 32 GB is recommended.\nGames can crash, and can even freeze or crash "
                       "the whole PC.\nThis setup isn't supported, so please don't open issues or "
                       "ask for help about crashes on it.\n",
                       static_cast<f64>(ram.bytes) / f64{1_GiB});
        }
        break; // Expected case
    case Core::SystemResultStatus::ErrorGetLoader:
        LOG_CRITICAL(Frontend, "Failed to obtain loader for {}!", lp.filepath);
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorMemoryAllocation:
        LOG_CRITICAL(Frontend,
                     "Not enough host memory to start the game. Close other applications and "
                     "check available memory and paging-file space, then try again.");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorLoader:
        LOG_CRITICAL(Frontend, "Failed to load ROM!");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorNotInitialized:
        LOG_CRITICAL(Frontend, "CPUCore not initialized");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorVideoCore:
        LOG_CRITICAL(Frontend, "Failed to initialize VideoCore!");
        return SDL_APP_FAILURE;
    default:
        const u16 loader_id = u16(Core::SystemResultStatus::ErrorLoader);
        const u16 error_id = u16(load_result) - loader_id;
        LOG_CRITICAL(Frontend,
            "While attempting to load the ROM requested, an error occurred. Please "
            "refer to the Eden wiki for more information or the Eden discord for "
            "additional help.\n\nError Code: {:04X}-{:04X}\nError Description: {}",
            loader_id, error_id, Loader::ResultStatus(error_id));
        return SDL_APP_FAILURE;
    }

    if (lp.use_multiplayer) {
        if (auto member = Network::GetRoomMember().lock()) {
            member->BindOnChatMessageReceived(OnMessageReceived);
            member->BindOnStatusMessageReceived(OnStatusMessageReceived);
            member->BindOnStateChanged(OnStateChanged);
            member->BindOnError(OnNetworkError);
            LOG_DEBUG(Network, "Start connection to {}:{} with nickname {}", lp.address, lp.port, lp.nickname);
            member->Join(lp.nickname, lp.address.c_str(), lp.port, 0, Network::NoPreferredIP, lp.password);
        } else {
            LOG_ERROR(Network, "Could not access RoomMember");
            return SDL_APP_FAILURE;
        }
    }

    // Core is loaded, start the GPU (makes the GPU contexts current to this thread)
    state->system.GPU().Start();
    state->system.GetCpuManager().OnGpuReady();

    if (Settings::values.use_disk_shader_cache.GetValue()) {
        state->system.Renderer().ReadRasterizer()->LoadDiskResources(
            state->system.GetApplicationProcessProgramID(), std::stop_token{},
            [](VideoCore::LoadCallbackStage, size_t value, size_t total) {});
    }

    // don't do anything, SDL3 already exists for us :D
    state->system.RegisterExitCallback([] {});

    // A game executing a program (e.g. restarting after a settings change) writes its launch state
    // to the restart handoff and quits with kRestartExitCode; the launcher boots the same file again
    // with EDEN_CLI_RESTART_HANDOFF set. Mirrors MainWindow::OnExecuteProgram.
    state->system.RegisterExecuteProgramCallback([system = &state->system](std::size_t program_index) {
        RestartHandoff handoff{
            .program_index = static_cast<s32>(program_index),
            .user_channel = system->GetUserChannel(),
        };
        const u64 previous_program_id = system->GetApplicationProcessProgramID();
        if (previous_program_id > static_cast<u64>(Service::AM::AppletProgramId::MaxProgramId)) {
            handoff.previous_program_index =
                static_cast<s32>(previous_program_id - FileSys::GetBaseTitleID(previous_program_id));
            handoff.program_id = previous_program_id;
        }
        // Without the handoff, a relaunch would boot as a frontend launch and clear temp storage.
        if (!WriteRestartHandoff(handoff)) {
            LOG_CRITICAL(Frontend, "ExecuteProgram: could not write {}, not restarting",
                         Common::FS::PathToUTF8String(RestartHandoffPath()));
            return;
        }
        LOG_INFO(Frontend, "ExecuteProgram: program index {}, exiting with code {} to restart",
                 program_index, kRestartExitCode);
        restart_requested = true;
    });

    // QLaunch launched applications (should) now reload shader cache.
    static std::mutex shader_cache_reload_mutex;
    state->system.RegisterApplicationChangedCallback([&state](u64 changed_program_id) {
        if (!Settings::values.use_disk_shader_cache.GetValue()) {
            return;
        }

        std::scoped_lock lk{shader_cache_reload_mutex};

        LOG_INFO(Frontend, "Reloading disk shader cache for {:016X}", changed_program_id);

        state->system.Pause();
        state->system.GPU().WaitForIdle();
        state->system.GPU().ObtainContext();

        state->system.Renderer().ReadRasterizer()->LoadDiskResources(
            changed_program_id, std::stop_token{},
            [](VideoCore::LoadCallbackStage, size_t, size_t) {});

        state->system.GPU().ReleaseContext();
        state->system.Run();
    });

    void(state->system.Run());
    if (state->system.DebuggerEnabled())
        state->system.InitializeDebugger();
    return SDL_APP_CONTINUE;
}
// EDEN_NX2_KEY_SCRIPT="seconds:code,seconds:-code,..." (or "@<file>"): press (code) or release (-code) keyboard
// keys, by SDL scancode, that many seconds after start. Lets hidden test runs drive menus.
double Nx2KeyClockSeconds(); // emu_window_sdl3.cpp

static void RunKeyScript(InputCommon::InputSubsystem& input_subsystem) {
    struct KeyEvent {
        double seconds; // when frame is 0
        u64 frame;      // "f<frame>:code": after that many presents
        int code;
    };
    static const std::vector<KeyEvent> script = [] {
        std::vector<KeyEvent> events;
        if (const char* text = std::getenv("EDEN_NX2_KEY_SCRIPT")) {
            // "@<file>" reads the list from a file: long scripts in the environment slow down
            // every getenv in the process.
            std::string list{text};
            if (!list.empty() && list[0] == '@') {
                std::ifstream file{list.substr(1)};
                list.assign(std::istreambuf_iterator<char>{file}, {});
            }
            size_t start = 0;
            while (start < list.size()) {
                const size_t end = (std::min)(list.find(',', start), list.size());
                const std::string item = list.substr(start, end - start);
                if (const size_t colon = item.find(':'); colon != std::string::npos) {
                    const std::string when = item.substr(0, colon);
                    const int code = std::stoi(item.substr(colon + 1));
                    if (!when.empty() && when[0] == 'f') {
                        events.push_back({0.0, std::stoull(when.substr(1)), code});
                    } else {
                        events.push_back({std::stod(when), 0, code});
                    }
                }
                start = end + 1;
            }
        }
        return events;
    }();
    static size_t next = 0;
    const double elapsed = Nx2KeyClockSeconds();
    const u64 frame = Core::g_nx2_present_count.load();
    while (next < script.size() && (script[next].frame != 0 ? script[next].frame <= frame
                                                            : script[next].seconds <= elapsed)) {
        const int code = script[next].code;
        LOG_INFO(Frontend, "Key script: {} key {} at {:.1f} s (frame {})",
                 code < 0 ? "release" : "press", code < 0 ? -code : code, elapsed, frame);
        if (code < 0) {
            input_subsystem.GetKeyboard()->ReleaseKey(static_cast<std::size_t>(-code));
        } else {
            input_subsystem.GetKeyboard()->PressKey(static_cast<std::size_t>(code));
        }
        ++next;
    }
}

extern "C" SDL_AppResult SDL_AppIterate(void *appstate) {
    SdlState *state = (SdlState *)appstate;
    RunKeyScript(state->input_subsystem);
    if (state->emu_window) {
        state->emu_window->ApplyPendingTitle();
    }
    if (restart_requested) {
        return SDL_APP_SUCCESS;
    }
    return state->emu_window && state->emu_window->IsOpen() ? SDL_APP_CONTINUE : SDL_APP_SUCCESS;
}
extern "C" SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
    SdlState *state = (SdlState *)appstate;
    if (state->emu_window)
        state->emu_window->OnEvent(*event);
    return SDL_APP_CONTINUE;
}
extern "C" void SDL_AppQuit(void *appstate, SDL_AppResult result) {
    SdlState *state = (SdlState *)appstate;
    if (!state) {
        return;
    }
    state->system.DetachDebugger();
    // SDL calls this even when SDL_AppInit never powered the system on (no ROM given, --help or
    // --version): the kernel is uninitialised then, and Pause()/ShutdownMainProcess() would
    // dereference it.
    if (state->system.IsPoweredOn()) {
        void(state->system.Pause());
        state->system.ShutdownMainProcess();
    }
    const bool network_initialized = state->network_initialized;
    delete state;
    if (network_initialized) {
        Network::Shutdown();
    }
    if (restart_requested) {
        // SDL3 maps SDL_APP_SUCCESS to exit code 0; quit SDL ourselves to return the restart code.
        SDL_Quit();
        std::exit(kRestartExitCode);
    }
}

#define VMA_IMPLEMENTATION
#include "video_core/vulkan_common/vma.h"
