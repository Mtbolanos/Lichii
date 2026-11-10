// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <thread>

#include "common/fs/path_util.h"
#include "common/scope_exit.h"
#include "common/logging.h"
#include "common/nx2_sampler.h"
#include "common/nx2_timeline.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/stb.h"
#include "core/core.h"
#include "core/debugger/nx2_debug_server.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/file_sys/patch_manager.h"
#include "core/hle/kernel/k_process.h"
#include "core/memory.h"
#include "hid_core/hid_core.h"
#include "video_core/gpu.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_base.h"

namespace Core::Nx2Debug {

using nlohmann::json;
using namespace std::chrono_literals;

/// Lines the log ring keeps for log_tail.
constexpr size_t RING_LINES = 20000;
/// Bytes one read_memory or read_gpu_memory call returns at most.
constexpr u64 MAX_READ = 64 * 1024;
/// Presented frames one input or input_sequence call holds at most (10 minutes at 60 fps).
constexpr u64 MAX_INPUT_FRAMES = 36000;
constexpr size_t MAX_INPUT_STEPS = 1000;
/// Presented frames one input recording or input_replay holds at most (4 hours at 60 fps).
constexpr u64 MAX_RECORD_FRAMES = 864000;

constexpr std::array<std::pair<std::string_view, HID::NpadButton>, 16> INPUT_BUTTONS{{
    {"A", HID::NpadButton::A},         {"B", HID::NpadButton::B},
    {"X", HID::NpadButton::X},         {"Y", HID::NpadButton::Y},
    {"L", HID::NpadButton::L},         {"R", HID::NpadButton::R},
    {"ZL", HID::NpadButton::ZL},       {"ZR", HID::NpadButton::ZR},
    {"Plus", HID::NpadButton::Plus},   {"Minus", HID::NpadButton::Minus},
    {"Up", HID::NpadButton::Up},       {"Down", HID::NpadButton::Down},
    {"Left", HID::NpadButton::Left},   {"Right", HID::NpadButton::Right},
    {"StickL", HID::NpadButton::StickL}, {"StickR", HID::NpadButton::StickR},
}};

/// input_record's sampler: one sample per present, run-length encoded as it goes.
struct InputRecorder {
    std::mutex mutex;
    HID::EmulatedController* controller{};
    std::string controller_name;
    u64 first_present{};
    u64 last_present{};
    u64 frames{};
    u64 missed{};
    bool full{};
    /// A stopped recording whose file could not be written yet; stop retries it.
    bool unsaved{};
    HID::InjectedInput last{};
    std::vector<InputStep> steps;
    /// Last, so it is joined before the members it reads go.
    std::jthread thread;
};

struct ToolState {
    explicit ToolState(System& system_, std::string title_, u64 program_id_,
                       std::filesystem::path dump_directory_)
        : system{system_}, title{std::move(title_)}, program_id{program_id_},
          dump_directory{std::move(dump_directory_)} {}

    System& system;
    std::string title;
    u64 program_id;
    std::filesystem::path dump_directory;
    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};
    /// Names this session's captures apart from earlier ones in the same dump directory.
    s64 session{std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count()};
    std::string log_filter{Settings::values.log_filter.GetValue()};
    u64 captures{};
    u64 screenshots{};
    u64 samples{};
    u64 gpu_reads{};
    InputRecorder recorder;
    std::atomic<bool> stopping{};
};

namespace {

std::string Hex(u64 value) {
    return fmt::format("{:016x}", value);
}

json HexList(const std::vector<u64>& values) {
    json list = json::array();
    for (const u64 value : values) {
        list.push_back(Hex(value));
    }
    return list;
}

template <typename T>
T Param(const json& params, const char* name, T fallback, T low, T high) {
    const T value = params.value(name, fallback);
    return (std::clamp)(value, low, high);
}

std::string HexAddress(u64 value) {
    return fmt::format("{:#x}", value);
}

/// An address as a JSON number or a hex string (0x optional).
u64 AddressParam(const json& params, const char* name) {
    const json& value = params.at(name);
    if (value.is_number_unsigned()) {
        return value.get<u64>();
    }
    if (!value.is_string()) {
        throw std::runtime_error{fmt::format("{} must be a hex string or a non-negative number", name)};
    }
    const std::string text = value.get<std::string>();
    const std::string_view view{text};
    const std::string_view digits =
        view.starts_with("0x") || view.starts_with("0X") ? view.substr(2) : view;
    u64 address = 0;
    const auto [end, error] =
        std::from_chars(digits.data(), digits.data() + digits.size(), address, 16);
    if (digits.empty() || error != std::errc{} || end != digits.data() + digits.size()) {
        throw std::runtime_error{fmt::format("bad {} \"{}\": want hex, e.g. 0x8012345000", name, text)};
    }
    return address;
}

/// The size of a read: 1 to MAX_READ bytes, not wrapping past the end of the address space.
u64 ReadSize(const json& params, u64 address) {
    const json& value = params.at("size");
    if (!value.is_number_integer() || value.get<s64>() < 1 || value.get<u64>() > MAX_READ) {
        throw std::runtime_error{fmt::format("size must be 1 to {} bytes", MAX_READ)};
    }
    const u64 size = value.get<u64>();
    if (address + size < address) {
        throw std::runtime_error{"address + size wraps around"};
    }
    return size;
}

Kernel::KProcess& Application(ToolState& state) {
    Kernel::KProcess* const process = state.system.ApplicationProcess();
    if (process == nullptr) {
        throw std::runtime_error{"no application process is running"};
    }
    return *process;
}

/// A host pointer to the byte at address, or null when its page is not mapped. GetPointerSilent
/// reads the page entry as guest cores do, with no commit, log or GPU download; it masks the top
/// 16 bits, so addresses there would alias and are unmapped instead.
const u8* GuestByte(Memory::Memory& memory, u64 address) {
    return (address >> 48) == 0 ? memory.GetPointerSilent(address) : nullptr;
}

bool IsVulkan() {
    return Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
}

/// Waits for a future in short slices so a stopping server does not wait out the timeout.
template <typename T>
bool WaitReady(const ToolState& state, std::future<T>& future, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (future.wait_for(50ms) != std::future_status::ready) {
        if (state.stopping.load(std::memory_order_relaxed) ||
            std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
    return true;
}

void AppendPng(void* context, void* data, int size) {
    auto& png = *static_cast<std::vector<u8>*>(context);
    const u8* const bytes = static_cast<const u8*>(data);
    png.insert(png.end(), bytes, bytes + size);
}

json Status(ToolState& state, const json& params) {
    const auto window = std::chrono::milliseconds{Param<int>(params, "fps_window_ms", 500, 0, 5000)};
    const u64 before = g_nx2_present_count.load(std::memory_order_relaxed);
    const auto begin = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - begin < window &&
           !state.stopping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(20ms);
    }
    const u64 after = g_nx2_present_count.load(std::memory_order_relaxed);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    json frame_times = json::array();
    for (const auto& count : VideoCommon::Nx2Perf::g_frame_times) {
        frame_times.push_back(count.load(std::memory_order_relaxed));
    }
    return {
        {"frame_time_bucket_us", VideoCommon::Nx2Perf::FrameTimeBucketNs / 1000},
        {"frame_times", std::move(frame_times)},
        {"title", state.title},
        {"program_id", fmt::format("{:016X}", state.program_id)},
        {"build", std::string{Common::g_build_fullname}},
        {"scm_rev", std::string{Common::g_scm_rev}},
        {"switch2", Settings::CurrentProgramIsSwitch2()},
        {"renderer", IsVulkan() ? "vulkan" : "other"},
        {"powered_on", state.system.IsPoweredOn()},
        {"paused", state.system.IsPaused()},
        {"presents", after},
        {"fps", seconds > 0.0 ? static_cast<double>(after - before) / seconds : 0.0},
        {"uptime_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - state.start)
                         .count()},
        {"dump_dir", state.dump_directory.generic_string()},
        {"log_filter", state.log_filter},
        {"rejected", HexList(VideoCore::Nx2Debug::RejectedHashes())},
    };
}

json Screenshot(ToolState& state, const json& params) {
    auto& renderer = state.system.GPU().Renderer();
    if (renderer.IsScreenshotPending()) {
        throw std::runtime_error{"a screenshot is already pending: no frame was presented since"};
    }
    const u32 height = Param<u32>(params, "height", 720, 144, 2160);
    const auto layout = Layout::DefaultFrameLayout(height * 16 / 9, height);
    const auto pixels = std::make_shared<std::vector<u8>>(size_t{layout.width} * layout.height * 4);
    const auto done = std::make_shared<std::promise<bool>>();
    auto future = done->get_future();
    renderer.RequestScreenshot(
        pixels->data(), [pixels, done](bool invert_y) { done->set_value(invert_y); }, layout);
    const int timeout = Param<int>(params, "timeout_s", 5, 1, 30);
    if (!WaitReady(state, future, std::chrono::seconds{timeout})) {
        throw std::runtime_error{
            fmt::format("no frame was presented within {} s (paused or loading?)", timeout)};
    }
    const bool invert_y = future.get();
    std::vector<u8> rgb(size_t{layout.width} * layout.height * 3);
    for (u32 y = 0; y < layout.height; ++y) {
        const u8* source = pixels->data() + size_t{invert_y ? layout.height - 1 - y : y} * layout.width * 4;
        u8* target = rgb.data() + size_t{y} * layout.width * 3;
        for (u32 x = 0; x < layout.width; ++x, source += 4, target += 3) {
            target[0] = source[2];
            target[1] = source[1];
            target[2] = source[0];
        }
    }
    std::vector<u8> png;
    if (!stbi_write_png_to_func(AppendPng, &png, static_cast<int>(layout.width),
                                static_cast<int>(layout.height), 3, rgb.data(),
                                static_cast<int>(layout.width * 3))) {
        throw std::runtime_error{"PNG encoding failed"};
    }
    json result = {{"width", layout.width}, {"height", layout.height}};
    if (params.value("save", false)) {
        std::filesystem::create_directories(state.dump_directory);
        const auto path =
            state.dump_directory / fmt::format("screenshot-{}-{:04}.png", state.session, ++state.screenshots);
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
        if (!file) {
            throw std::runtime_error{fmt::format("cannot write {}", path.generic_string())};
        }
        result["path"] = path.generic_string();
    }
    if (params.value("inline", true)) {
        result["png_base64"] = Base64(png);
    }
    return result;
}

json LogTail(ToolState&, const json& params) {
    const u64 after = params.value("after", u64{0});
    const size_t limit = Param<size_t>(params, "limit", 200, 1, 5000);
    const std::string contains = params.value("contains", std::string{});
    const std::vector<Common::Log::RingLine> lines = Common::Log::ReadRing(after);
    std::vector<const Common::Log::RingLine*> matched;
    for (const auto& line : lines) {
        if (contains.empty() || line.text.find(contains) != std::string::npos) {
            matched.push_back(&line);
        }
    }
    // From the start the newest lines; after a sequence number the oldest ones, to page forward.
    const bool more = matched.size() > limit;
    if (more) {
        if (after == 0) {
            matched.erase(matched.begin(), matched.end() - static_cast<std::ptrdiff_t>(limit));
        } else {
            matched.resize(limit);
        }
    }
    json text = json::array();
    for (const auto* line : matched) {
        text.push_back(line->text);
    }
    const u64 next = more && after != 0 ? matched.back()->sequence
                     : lines.empty()    ? after
                                        : lines.back().sequence;
    return {{"lines", std::move(text)}, {"next", next}, {"more", more}};
}

json LogFilter(ToolState& state, const json& params) {
    const std::string filter = params.at("filter").get<std::string>();
    Common::Log::Filter parsed;
    parsed.ParseFilterString(filter);
    Common::Log::SetGlobalFilter(parsed);
    json result = {{"previous", state.log_filter}, {"filter", filter}};
    state.log_filter = filter;
    return result;
}

json CaptureFrame(ToolState& state, const json& params) {
    if (!IsVulkan()) {
        throw std::runtime_error{"frame capture needs the Vulkan renderer"};
    }
    const u64 id = ++state.captures;
    const auto directory = state.dump_directory / fmt::format("capture-{}-{:03}", state.session, id);
    std::filesystem::create_directories(directory);
    const bool sampled = params.value("sampled", false);
    const bool buffers = params.value("buffers", false);
    const int timeout = Param<int>(params, "timeout_s", 120, 1, 600);
    VideoCore::Nx2Debug::RequestCapture(
        {.id = id, .directory = directory, .sampled = sampled, .buffers = buffers});
    const auto finished = VideoCore::Nx2Debug::WaitCapture(id, std::chrono::seconds{timeout});
    if (!finished) {
        VideoCore::Nx2Debug::CancelCapture(id);
        throw std::runtime_error{fmt::format(
            "the capture did not finish within {} s (paused or loading?); if it started, its "
            "files are still being written to {}",
            timeout, directory.generic_string())};
    }
    if (!finished->error.empty()) {
        throw std::runtime_error{finished->error};
    }
    const size_t keep = Param<size_t>(params, "manifest_lines", 200, 0, 5000);
    json entries = json::array();
    json kinds = json::object();
    json frame;
    json summary;
    size_t total = 0;
    std::ifstream manifest{directory / "manifest.jsonl"};
    for (std::string line; std::getline(manifest, line);) {
        json entry = json::parse(line, nullptr, false);
        if (entry.is_discarded()) {
            continue;
        }
        const std::string kind = entry.value("kind", std::string{});
        if (kind == "frame") {
            frame = std::move(entry);
        } else if (kind == "summary") {
            summary = std::move(entry);
        } else {
            kinds[kind] = kinds.value(kind, 0) + 1;
            if (total++ < keep) {
                entries.push_back(std::move(entry));
            }
        }
    }
    return {
        {"directory", directory.generic_string()},
        {"sampled", sampled},
        {"buffers", buffers},
        {"frame", std::move(frame)},
        {"summary", std::move(summary)},
        {"kinds", std::move(kinds)},
        {"entries_total", total},
        {"entries", std::move(entries)},
        {"index_command",
         fmt::format("python3 -I tools/nx2/frame_dump_index.py \"{}\"", directory.generic_string())},
    };
}

json RejectShaders(ToolState&, const json& params) {
    if (!IsVulkan()) {
        throw std::runtime_error{"runtime rejection needs the Vulkan renderer"};
    }
    std::vector<u64> hashes;
    for (const json& hash : params.value("hashes", json::array())) {
        hashes.push_back(ParseShaderHash(hash.get<std::string>()));
    }
    const bool reject = params.value("reject", true);
    const u64 generation = VideoCore::Nx2Debug::SetRejected(hashes, reject);
    const int timeout = Param<int>(params, "timeout_s", 5, 0, 60);
    const bool applied = hashes.empty() ||
                         VideoCore::Nx2Debug::WaitRejectApplied(generation, std::chrono::seconds{timeout});
    json result = {{"rejected", HexList(VideoCore::Nx2Debug::RejectedHashes())}, {"applied", applied}};
    if (!applied) {
        result["note"] = "no frame was presented yet; the change applies at the next frame end";
    }
    return result;
}

// Records what the emulator's threads do and wait for (common/nx2_timeline.h) and returns once
// the Chrome trace is written.
#if defined(__LLVM_INSTR_PROFILE_GENERATE)
extern "C" int __llvm_profile_write_file(void);
#endif

// Writes the instrumented build's profile now (clang -fprofile-instr-generate builds only): it is
// otherwise written at a clean exit, which a training run can't always get.
json PgoDump(ToolState&, const json&) {
#if defined(__LLVM_INSTR_PROFILE_GENERATE)
    return {{"written", __llvm_profile_write_file() == 0}};
#else
    throw std::runtime_error{"not an instrumented (-fprofile-instr-generate) build"};
#endif
}

json Timeline(ToolState& state, const json& params) {
    const double seconds = Param<double>(params, "seconds", 5.0, 0.1, 30.0);
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    const auto path = state.dump_directory / fmt::format("timeline_{}.json", stamp);
    if (!Common::Nx2Timeline::Start(path, seconds, params.value("detail", false))) {
        throw std::runtime_error{"a timeline recording is already running"};
    }
    const std::string expected = Common::FS::PathToUTF8String(path);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds + 30.0);
    while (Common::Nx2Timeline::LastFile() != expected) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error{"the timeline was not written in time"};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    std::error_code error;
    return {{"file", expected},
            {"seconds", seconds},
            {"bytes", std::filesystem::file_size(path, error)}};
}

json SampleThreads(ToolState& state, const json& params) {
    const std::string threads = params.value("threads", std::string{"GPU,GPUBack"});
    const double seconds = Param<double>(params, "seconds", 10.0, 1.0, 60.0);
    std::filesystem::create_directories(state.dump_directory);
    const auto path =
        state.dump_directory / fmt::format("samples-{}-{:03}.txt", state.session, ++state.samples);
    std::string report = Nx2Sampler::Sample(threads.c_str(), seconds, path.string());
    report.resize((std::min)(report.size(), Param<size_t>(params, "max_chars", 60000, 0, 4000000)));
    return {{"file", path.generic_string()}, {"report", std::move(report)}};
}

json SetSwitch(ToolState&, const json& params) {
    const std::string name = params.at("name").get<std::string>();
    auto* const value = VideoCore::Nx2Debug::FindSwitch(name);
    if (value == nullptr) {
        throw std::runtime_error{fmt::format("unknown live switch {}", name)};
    }
    const int previous = value->load(std::memory_order_relaxed);
    if (params.contains("value")) {
        value->store(params.at("value").get<int>(), std::memory_order_relaxed);
    }
    return {{"name", name}, {"previous", previous}, {"value", value->load()}};
}

// Ocarina of Time's high refresh rate mode, switched live for tests without saving the setting
// (the Ctrl+U hotkey saves it into the player's config). Also returns main's code address, for
// read_memory of the patched cave (main + 0x3C46200) and the ripple hook (main + 0x475F8).
json OotHighRefresh(ToolState& state, const json& params) {
    Kernel::KProcess* const process = state.system.ApplicationProcess();
    json result = {{"patched", Settings::OotHighRefreshPatched()},
                   {"main", HexAddress(process ? process->GetOotMainCodeAddress() : 0)}};
    if (params.contains("on")) {
        const auto outcome =
            FileSys::SwitchOotHighRefreshLive(state.system, params.at("on").get<bool>());
        result["outcome"] = static_cast<int>(outcome);
        result["switched"] = outcome == FileSys::OotHighRefreshSwitch::Switched;
        result["patched"] = Settings::OotHighRefreshPatched();
    }
    return result;
}

// Lifts or restores the speed limit (test scripts boot through the logos and menus unlimited).
json SpeedLimit(ToolState&, const json& params) {
    auto& setting = Settings::values.use_speed_limit;
    const bool previous = setting.GetValue();
    if (params.contains("on")) {
        setting.SetValue(params.at("on").get<bool>());
    }
    return {{"previous", previous}, {"on", setting.GetValue()}};
}

json ReadMemory(ToolState& state, const json& params) {
    const u64 address = AddressParam(params, "address");
    const u64 size = ReadSize(params, address);
    const std::string as = params.value("as", std::string{"hex"});
    static_cast<void>(FormatMemory(std::span<const u8>{}, as));
    std::vector<u8> bytes(size);
    if (const auto unmapped = ReadGuestMemory(Application(state).GetMemory(), address, bytes)) {
        throw std::runtime_error{fmt::format(
            "guest address {:#x} is not mapped (reading {} bytes at {:#x})", *unmapped, size, address)};
    }
    json result = {{"address", HexAddress(address)}, {"size", size}};
    result.update(FormatMemory(bytes, as));
    return result;
}

json SearchMemory(ToolState& state, const json& params) {
    const std::vector<u8> pattern = ParseHexBytes(params.at("pattern").get<std::string>());
    if (pattern.size() > 256) {
        throw std::runtime_error{"the pattern is longer than 256 bytes"};
    }
    Kernel::KProcess& process = Application(state);
    const std::string regions = params.value("regions", std::string{"heap"});
    u64 region_start = 0;
    u64 region_size = 0;
    if (regions == "heap") {
        region_start = GetInteger(process.GetPageTable().GetHeapRegionStart());
        region_size = process.GetPageTable().GetHeapRegionSize();
    } else if (regions == "all") {
        region_start = GetInteger(process.GetPageTable().GetAddressSpaceStart());
        region_size = process.GetPageTable().GetAddressSpaceSize();
    } else {
        throw std::runtime_error{fmt::format("regions \"{}\": want heap or all", regions)};
    }
    const u64 start = params.contains("start") ? (std::max)(region_start, AddressParam(params, "start"))
                                               : region_start;
    const u64 end = params.contains("end")
                        ? (std::min)(region_start + region_size, AddressParam(params, "end"))
                        : region_start + region_size;
    const size_t max_hits = Param<size_t>(params, "max_hits", 64, 1, 256);
    const int budget = Param<int>(params, "budget_ms", 2000, 1, 10000);
    const auto begin = std::chrono::steady_clock::now();
    const GuestSearch search =
        SearchGuestMemory(process.GetMemory(), pattern, start, (std::max)(start, end), max_hits,
                          begin + std::chrono::milliseconds{budget});
    json hits = json::array();
    for (const u64 hit : search.hits) {
        hits.push_back(HexAddress(hit));
    }
    json result = {
        {"hits", std::move(hits)},
        {"complete", search.complete},
        {"start", HexAddress(start)},
        {"end", HexAddress((std::max)(start, end))},
        {"scanned_bytes", search.scanned},
        {"elapsed_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                 begin)
                           .count()},
    };
    if (!search.complete) {
        result["next"] = HexAddress(search.next);
    }
    return result;
}

json ReadGpuMemory(ToolState& state, const json& params) {
    if (!IsVulkan()) {
        throw std::runtime_error{"GPU memory reads need the Vulkan renderer"};
    }
    const u64 address = AddressParam(params, "gpu_address");
    const u64 size = ReadSize(params, address);
    const std::string as = params.value("as", std::string{"hex"});
    static_cast<void>(FormatMemory(std::span<const u8>{}, as));
    const u64 id = ++state.gpu_reads;
    const int timeout = Param<int>(params, "timeout_s", 5, 1, 60);
    VideoCore::Nx2Debug::RequestGpuRead({.id = id, .gpu_address = address, .size = size});
    const auto read = VideoCore::Nx2Debug::WaitGpuRead(id, std::chrono::seconds{timeout});
    if (!read) {
        VideoCore::Nx2Debug::CancelGpuRead(id);
        throw std::runtime_error{
            fmt::format("no frame was presented within {} s (paused or loading?)", timeout)};
    }
    if (!read->error.empty()) {
        throw std::runtime_error{read->error};
    }
    json result = {{"gpu_address", HexAddress(address)},
                   {"size", size},
                   {"address_space", read->address_space}};
    result.update(FormatMemory(read->bytes, as));
    return result;
}

/// "player1" or "handheld"; empty picks player 1, or the handheld controller when player 1 is
/// not connected.
std::pair<HID::EmulatedController*, std::string> PickController(ToolState& state,
                                                                std::string_view wanted) {
    auto& hid = state.system.HIDCore();
    if (wanted.empty()) {
        HID::EmulatedController* const player1 = hid.GetEmulatedController(HID::NpadIdType::Player1);
        if (player1->IsConnected()) {
            return {player1, "player1"};
        }
        HID::EmulatedController* const handheld =
            hid.GetEmulatedController(HID::NpadIdType::Handheld);
        if (handheld->IsConnected()) {
            return {handheld, "handheld"};
        }
        throw std::runtime_error{"neither player 1 nor the handheld controller is connected"};
    }
    if (wanted != "player1" && wanted != "handheld") {
        throw std::runtime_error{fmt::format("controller \"{}\": want player1 or handheld", wanted)};
    }
    HID::EmulatedController* const controller = hid.GetEmulatedController(
        wanted == "player1" ? HID::NpadIdType::Player1 : HID::NpadIdType::Handheld);
    if (!controller->IsConnected()) {
        throw std::runtime_error{fmt::format("the {} controller is not connected", wanted)};
    }
    return {controller, std::string{wanted}};
}

struct InputRun {
    u64 max_frames = MAX_INPUT_FRAMES;
    int max_timeout_s = 900;
    std::string_view controller;
    /// The present count the first step starts at; the next present when unset.
    std::optional<u64> at_present;
    /// Above 0: steps advance with elapsed time at this many frames a second instead of with
    /// presents. Games that move by elapsed time (Ocarina of Time) then follow the recorded path
    /// whatever frame rate the replay runs at.
    double frame_rate = 0.0;
};

/// Holds the steps on the run's controller from the next present (or at_present) on, and returns
/// once they are done. The injector stops by itself after the last step; leaving this function by
/// any path removes it.
json RunInput(ToolState& state, const std::vector<InputStep>& steps, const json& params,
              const ClientGone& client_gone, const InputRun& run = {}) {
    u64 total = 0;
    for (const InputStep& step : steps) {
        total += step.frames;
    }
    if (total > run.max_frames) {
        throw std::runtime_error{
            fmt::format("{} frames in total; at most {} per call", total, run.max_frames)};
    }
    const auto picked = PickController(state, run.controller);
    HID::EmulatedController* const controller = picked.first;
    const std::string& name = picked.second;
    const u64 current = g_nx2_present_count.load(std::memory_order_relaxed);
    const u64 start = run.at_present.value_or(current + 1);
    if (start <= current) {
        throw std::runtime_error{
            fmt::format("at_present {} has already been presented (now {})", start, current)};
    }
    const u64 wait_frames = start - current - 1 + total;
    const int timeout = Param<int>(
        params, "timeout_s",
        static_cast<int>((std::min)(wait_frames / 20 + 10, static_cast<u64>(run.max_timeout_s))), 1,
        run.max_timeout_s);
    const auto script = std::make_shared<const std::vector<InputStep>>(steps);
    const auto ends = std::make_shared<const std::vector<u64>>(InputStepEnds(steps, start));
    const double frame_rate = run.frame_rate;
    const auto timed_from = std::chrono::steady_clock::now();
    // The frame the replay is at: the present count, or with frame_rate the elapsed time.
    const auto frame_now = [frame_rate, timed_from, start] {
        if (frame_rate > 0.0) {
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - timed_from).count();
            return start + static_cast<u64>(seconds * frame_rate);
        }
        return g_nx2_present_count.load(std::memory_order_relaxed);
    };
    controller->SetInputInjector([script, ends, frame_now] {
        return InputAt(*script, *ends, frame_now());
    });
    SCOPE_EXIT {
        controller->SetInputInjector({});
    };
    const auto begin = std::chrono::steady_clock::now();
    const auto deadline = begin + std::chrono::seconds{timeout};
    for (;;) {
        const u64 now = frame_now();
        if (now >= start + total) {
            break;
        }
        if (state.stopping.load(std::memory_order_relaxed)) {
            throw std::runtime_error{"the game is shutting down; input released"};
        }
        if (client_gone()) {
            LOG_INFO(Debug, "Debug server client left during input; input released");
            throw std::runtime_error{"the client disconnected; input released"};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error{fmt::format(
                "{} of {} frames were presented within {} s (paused or loading?); input released",
                now >= start ? now - start : 0, total, timeout)};
        }
        std::this_thread::sleep_for(2ms);
    }
    return {
        {"controller", name},
        {"steps", steps.size()},
        {"frames", total},
        {"first_present", start},
        {"frame_rate", frame_rate},
        {"elapsed_s",
         std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()},
    };
}

json Input(ToolState& state, const json& params, const ClientGone& client_gone) {
    json step = params;
    step.erase("timeout_s");
    return RunInput(state, {ParseInputStep(step)}, params, client_gone);
}

json InputSequence(ToolState& state, const json& params, const ClientGone& client_gone) {
    const json& list = params.at("steps");
    if (!list.is_array() || list.empty() || list.size() > MAX_INPUT_STEPS) {
        throw std::runtime_error{fmt::format("steps must be an array of 1 to {}", MAX_INPUT_STEPS)};
    }
    std::vector<InputStep> steps;
    for (size_t index = 0; index < list.size(); ++index) {
        try {
            steps.push_back(ParseInputStep(list[index]));
        } catch (const std::exception& error) {
            throw std::runtime_error{fmt::format("step {}: {}", index, error.what())};
        }
    }
    return RunInput(state, steps, params, client_gone);
}

/// A UTF-8 path; relative ones are under the dump directory.
std::filesystem::path InputPath(const ToolState& state, const std::string& text) {
    const std::filesystem::path path{Common::FS::ToU8String(text)};
    return path.is_absolute() ? path : state.dump_directory / path;
}

void SampleInput(std::stop_token stop, InputRecorder& recorder) {
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(500us);
        const u64 now = g_nx2_present_count.load(std::memory_order_relaxed);
        std::scoped_lock lock{recorder.mutex};
        if (now == recorder.last_present || recorder.full) {
            continue;
        }
        if (now < recorder.last_present) {
            // The count restarted with a new boot; nothing was presented for this recording.
            recorder.last_present = now;
            continue;
        }
        const HID::InjectedInput input = RecordedInput(recorder.controller->GetNpadButtons().raw,
                                                       recorder.controller->GetSticks());
        const u64 skipped = now - recorder.last_present - 1;
        if (skipped > 0) {
            AppendRecordedInput(recorder.steps, recorder.frames > 0 ? recorder.last : input,
                                skipped);
            recorder.missed += skipped;
            recorder.frames += skipped;
        }
        AppendRecordedInput(recorder.steps, input, 1);
        ++recorder.frames;
        recorder.last = input;
        recorder.last_present = now;
        if (recorder.frames >= MAX_RECORD_FRAMES) {
            recorder.full = true;
            LOG_WARNING(Debug, "Input recording reached {} frames; sampling stopped",
                        MAX_RECORD_FRAMES);
        }
    }
}

json RecordingStatus(const InputRecorder& recorder) {
    return {
        {"recording", recorder.thread.joinable()},
        {"unsaved", recorder.unsaved},
        {"controller", recorder.controller_name},
        {"first_present", recorder.first_present},
        {"frames", recorder.frames},
        {"steps", recorder.steps.size()},
        {"missed", recorder.missed},
        {"full", recorder.full},
    };
}

json InputRecord(ToolState& state, const json& params) {
    InputRecorder& recorder = state.recorder;
    const std::string action = params.at("action").get<std::string>();
    if (action == "status") {
        std::scoped_lock lock{recorder.mutex};
        return RecordingStatus(recorder);
    }
    if (action == "start") {
        if (recorder.thread.joinable()) {
            throw std::runtime_error{"a recording is already running; stop it first"};
        }
        const auto [controller, name] =
            PickController(state, params.value("controller", std::string{}));
        if (controller->HasInputInjector()) {
            throw std::runtime_error{
                fmt::format("debug server input is being injected into {}; record after it", name)};
        }
        std::scoped_lock lock{recorder.mutex};
        recorder.controller = controller;
        recorder.controller_name = name;
        recorder.last_present = g_nx2_present_count.load(std::memory_order_relaxed);
        recorder.first_present = recorder.last_present + 1;
        recorder.frames = 0;
        recorder.missed = 0;
        recorder.full = false;
        recorder.unsaved = false;
        recorder.last = {};
        recorder.steps.clear();
        recorder.thread =
            std::jthread{[&recorder](std::stop_token stop) { SampleInput(stop, recorder); }};
        LOG_INFO(Debug, "Input recording of {} started at present {}", name,
                 recorder.first_present);
        return RecordingStatus(recorder);
    }
    if (action != "stop") {
        throw std::runtime_error{fmt::format("action \"{}\": want start, stop or status", action)};
    }
    if (recorder.thread.joinable()) {
        recorder.thread.request_stop();
        recorder.thread.join();
        recorder.unsaved = true;
    } else if (!recorder.unsaved) {
        throw std::runtime_error{"no recording is running"};
    }
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    const std::filesystem::path path =
        params.contains("path") ? InputPath(state, params.at("path").get<std::string>())
                                : state.dump_directory / fmt::format("input_{}.json", stamp);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::string text = fmt::format("{{\"version\":1,\"controller\":\"{}\",\"first_present\":{},"
                                   "\"frames\":{},\"missed\":{},\"steps\":[",
                                   recorder.controller_name, recorder.first_present,
                                   recorder.frames, recorder.missed);
    for (size_t index = 0; index < recorder.steps.size(); ++index) {
        text += index == 0 ? "\n" : ",\n";
        text += InputStepJson(recorder.steps[index]).dump();
    }
    text += "\n]}\n";
    {
        std::ofstream file{path, std::ios::binary | std::ios::trunc};
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!file) {
            throw std::runtime_error{fmt::format(
                "cannot write {}; the recording is kept, stop again with another path",
                Common::FS::PathToUTF8String(path))};
        }
    }
    recorder.unsaved = false;
    const std::string written = Common::FS::PathToUTF8String(path);
    LOG_INFO(Debug, "Input recording of {} frames ({} missed) written to {}", recorder.frames,
             recorder.missed, written);
    json result = RecordingStatus(recorder);
    result["path"] = written;
    return result;
}

/// Plays a recording's steps in one run, so recordings beyond input_sequence's limits play
/// without gaps between chunks.
json InputReplay(ToolState& state, const json& params, const ClientGone& client_gone) {
    const std::filesystem::path path = InputPath(state, params.at("path").get<std::string>());
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        throw std::runtime_error{fmt::format("cannot read {}", Common::FS::PathToUTF8String(path))};
    }
    const json recording = json::parse(file);
    if (recording.value("version", 1) != 1) {
        throw std::runtime_error{"unknown recording version; want 1"};
    }
    const json& list = recording.at("steps");
    if (!list.is_array() || list.empty()) {
        throw std::runtime_error{"steps must be a non-empty array"};
    }
    std::vector<InputStep> steps;
    steps.reserve(list.size());
    for (size_t index = 0; index < list.size(); ++index) {
        try {
            steps.push_back(ParseInputStep(list[index]));
        } catch (const std::exception& error) {
            throw std::runtime_error{fmt::format("step {}: {}", index, error.what())};
        }
    }
    const std::string controller =
        params.value("controller", recording.value("controller", std::string{}));
    std::optional<u64> at_present;
    if (params.contains("at_present")) {
        at_present = params.at("at_present").get<u64>();
    }
    // Timing: "time" (default when the recording has its rate) replays by elapsed time at the
    // recorded frame rate; "frames" replays one step frame per present.
    double frame_rate = 0.0;
    if (params.value("timing", std::string{"time"}) == "time") {
        if (params.contains("frame_rate")) {
            frame_rate = params.at("frame_rate").get<double>();
        } else if (recording.contains("frame_rate")) {
            frame_rate = recording.at("frame_rate").get<double>();
        } else if (recording.contains("seconds") && recording.value("seconds", 0.0) > 0.0) {
            frame_rate = recording.value("frames", 0.0) / recording.at("seconds").get<double>();
        }
    }
    json result = RunInput(state, steps, params, client_gone,
                           {.max_frames = MAX_RECORD_FRAMES,
                            .max_timeout_s = 86400,
                            .controller = controller,
                            .at_present = at_present,
                            .frame_rate = frame_rate});
    result["path"] = Common::FS::PathToUTF8String(path);
    return result;
}

Handlers MakeHandlers(const std::shared_ptr<ToolState>& state) {
    const auto handler = [&state](json (*tool)(ToolState&, const json&)) -> Handler {
        return [state, tool](const json& params, const ClientGone&) { return tool(*state, params); };
    };
    const auto waiting = [&state](json (*tool)(ToolState&, const json&, const ClientGone&)) -> Handler {
        return [state, tool](const json& params, const ClientGone& client_gone) {
            return tool(*state, params, client_gone);
        };
    };
    return {
        {"status", handler(Status)},
        {"screenshot", handler(Screenshot)},
        {"log_tail", handler(LogTail)},
        {"log_filter", handler(LogFilter)},
        {"capture_frame", handler(CaptureFrame)},
        {"reject_shaders", handler(RejectShaders)},
        {"sample", handler(SampleThreads)},
        {"timeline", handler(Timeline)},
        {"pgo_dump", handler(PgoDump)},
        {"switch", handler(SetSwitch)},
        {"speed_limit", handler(SpeedLimit)},
        {"oot_high_refresh", handler(OotHighRefresh)},
        {"read_memory", handler(ReadMemory)},
        {"search_memory", handler(SearchMemory)},
        {"read_gpu_memory", handler(ReadGpuMemory)},
        {"input", waiting(Input)},
        {"input_sequence", waiting(InputSequence)},
        {"input_record", handler(InputRecord)},
        {"input_replay", waiting(InputReplay)},
    };
}

} // Anonymous namespace

u64 ParseShaderHash(std::string_view text) {
    const std::string_view digits =
        text.starts_with("0x") || text.starts_with("0X") ? text.substr(2) : text;
    u64 value = 0;
    const auto [end, error] =
        std::from_chars(digits.data(), digits.data() + digits.size(), value, 16);
    if (digits.size() != 16 || error != std::errc{} || end != digits.data() + digits.size()) {
        throw std::runtime_error{fmt::format(
            "bad hash \"{}\": want all 16 hex digits of a native program key (shader dump file "
            "name) or shader hash (frame dump \"shaders\"); a shorter one would match nothing",
            text)};
    }
    return value;
}

std::vector<u8> ParseHexBytes(std::string_view text) {
    std::string digits;
    const std::string_view body =
        text.starts_with("0x") || text.starts_with("0X") ? text.substr(2) : text;
    for (const char c : body) {
        if (c != ' ') {
            digits.push_back(c);
        }
    }
    std::vector<u8> bytes(digits.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        const auto [end, error] =
            std::from_chars(digits.data() + i * 2, digits.data() + i * 2 + 2, bytes[i], 16);
        if (error != std::errc{} || end != digits.data() + i * 2 + 2) {
            bytes.clear();
            break;
        }
    }
    if (bytes.empty() || digits.size() % 2 != 0) {
        throw std::runtime_error{fmt::format(
            "bad pattern \"{}\": want hex byte pairs in memory order (little endian: 1.0f is "
            "0000803f)",
            text)};
    }
    return bytes;
}

json FormatMemory(std::span<const u8> bytes, std::string_view as) {
    if (as == "hex") {
        std::string hex;
        hex.reserve(bytes.size() * 2);
        for (const u8 byte : bytes) {
            fmt::format_to(std::back_inserter(hex), "{:02x}", byte);
        }
        return {{"hex", std::move(hex)}};
    }
    const size_t element = as == "u32" || as == "f32" ? 4 : as == "u64" ? 8 : 0;
    if (element == 0) {
        throw std::runtime_error{fmt::format("as \"{}\": want hex, u32, u64 or f32", as)};
    }
    if (bytes.size() % element != 0) {
        throw std::runtime_error{fmt::format("size must be a multiple of {} for {}", element, as)};
    }
    json values = json::array();
    for (size_t offset = 0; offset < bytes.size(); offset += element) {
        if (element == 8) {
            u64 value;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            values.push_back(HexAddress(value));
            continue;
        }
        u32 value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        if (as == "u32") {
            values.push_back(value);
            continue;
        }
        const float number = std::bit_cast<float>(value);
        if (std::isnan(number)) {
            values.push_back("nan");
        } else if (std::isinf(number)) {
            values.push_back(number < 0 ? "-inf" : "inf");
        } else {
            values.push_back(number);
        }
    }
    return {{std::string{as}, std::move(values)}};
}

std::optional<u64> ReadGuestMemory(Memory::Memory& memory, u64 address, std::span<u8> out) {
    for (size_t offset = 0; offset < out.size();) {
        const u64 current = address + offset;
        const size_t length = (std::min)(static_cast<size_t>(Memory::YUZU_PAGESIZE -
                                                             (current & Memory::YUZU_PAGEMASK)),
                                         out.size() - offset);
        const u8* const source = GuestByte(memory, current);
        if (source == nullptr) {
            return current;
        }
        std::memcpy(out.data() + offset, source, length);
        offset += length;
    }
    return std::nullopt;
}

GuestSearch SearchGuestMemory(Memory::Memory& memory, std::span<const u8> pattern, u64 start,
                              u64 end, size_t max_hits,
                              std::chrono::steady_clock::time_point deadline) {
    GuestSearch search{.hits = {}, .next = end, .complete = true, .scanned = 0};
    // The bytes of a run of mapped pages that a match starting there could still need.
    std::vector<u8> window;
    window.reserve(Memory::YUZU_PAGESIZE + pattern.size());
    for (u64 address = start; address < end;) {
        if (std::chrono::steady_clock::now() >= deadline) {
            search.next = address - window.size();
            search.complete = false;
            return search;
        }
        const u64 page_end = (std::min)((address | Memory::YUZU_PAGEMASK) + 1, end);
        const u8* const source = GuestByte(memory, address);
        if (source == nullptr) {
            window.clear();
            address = page_end;
            continue;
        }
        const u64 window_start = address - window.size();
        const size_t length = page_end - address;
        window.insert(window.end(), source, source + length);
        search.scanned += length;
        for (size_t offset = 0; offset + pattern.size() <= window.size(); ++offset) {
            const void* const found = std::memchr(window.data() + offset, pattern[0],
                                                  window.size() - pattern.size() + 1 - offset);
            if (found == nullptr) {
                break;
            }
            offset = static_cast<size_t>(static_cast<const u8*>(found) - window.data());
            if (std::memcmp(window.data() + offset, pattern.data(), pattern.size()) != 0) {
                continue;
            }
            search.hits.push_back(window_start + offset);
            if (search.hits.size() >= max_hits) {
                search.next = window_start + offset + 1;
                search.complete = false;
                return search;
            }
        }
        window.erase(window.begin(),
                     window.end() - static_cast<std::ptrdiff_t>(
                                        (std::min)(pattern.size() - 1, window.size())));
        address = page_end;
    }
    return search;
}

InputStep ParseInputStep(const json& step) {
    if (!step.is_object()) {
        throw std::runtime_error{"a step is an object"};
    }
    for (const auto& [key, value] : step.items()) {
        if (key != "buttons" && key != "left_stick" && key != "right_stick" && key != "frames") {
            throw std::runtime_error{fmt::format(
                "unknown key \"{}\": want buttons, left_stick, right_stick, frames", key)};
        }
    }
    InputStep result{.input = {}, .frames = 0};
    const json& frames = step.at("frames");
    if (!frames.is_number_integer() || frames.get<s64>() < 1 ||
        frames.get<u64>() > MAX_INPUT_FRAMES) {
        throw std::runtime_error{fmt::format("frames must be 1 to {}", MAX_INPUT_FRAMES)};
    }
    result.frames = frames.get<u64>();
    for (const json& button : step.value("buttons", json::array())) {
        const std::string text = button.is_string() ? button.get<std::string>() : button.dump();
        const auto it = std::ranges::find(INPUT_BUTTONS, std::string_view{text},
                                          &std::pair<std::string_view, HID::NpadButton>::first);
        if (it == INPUT_BUTTONS.end()) {
            std::string names;
            for (const auto& [known, _] : INPUT_BUTTONS) {
                names += names.empty() ? "" : ", ";
                names += known;
            }
            throw std::runtime_error{fmt::format("unknown button {}: want one of {}", text, names)};
        }
        result.input.buttons |= it->second;
    }
    const auto stick = [&step](const char* key) -> std::optional<HID::AnalogStickState> {
        if (!step.contains(key)) {
            return std::nullopt;
        }
        const json& value = step.at(key);
        const auto axis = [&](size_t index) {
            if (!value.is_array() || value.size() != 2 || !value[index].is_number() ||
                std::abs(value[index].get<double>()) > 1.0) {
                throw std::runtime_error{fmt::format("{} must be [x, y] with both in -1..1", key)};
            }
            return static_cast<s32>(std::lround(value[index].get<double>() * HID::HID_JOYSTICK_MAX));
        };
        return HID::AnalogStickState{.x = axis(0), .y = axis(1)};
    };
    result.input.left = stick("left_stick");
    result.input.right = stick("right_stick");
    return result;
}

std::optional<HID::InjectedInput> InputAt(std::span<const InputStep> steps, u64 start, u64 now) {
    if (now < start) {
        return std::nullopt;
    }
    u64 end = start;
    for (const InputStep& step : steps) {
        end += step.frames;
        if (now < end) {
            return step.input;
        }
    }
    return std::nullopt;
}

std::vector<u64> InputStepEnds(std::span<const InputStep> steps, u64 start) {
    std::vector<u64> ends;
    ends.reserve(steps.size());
    u64 end = start;
    for (const InputStep& step : steps) {
        end += step.frames;
        ends.push_back(end);
    }
    return ends;
}

std::optional<HID::InjectedInput> InputAt(std::span<const InputStep> steps,
                                          std::span<const u64> ends, u64 now) {
    if (steps.empty() || now < ends.front() - steps.front().frames) {
        return std::nullopt;
    }
    const auto it = std::ranges::upper_bound(ends, now);
    if (it == ends.end()) {
        return std::nullopt;
    }
    return steps[static_cast<size_t>(it - ends.begin())].input;
}

namespace {

constexpr double RECORD_STICK_STEPS = 1024.0;
constexpr double RECORD_STICK_DEADZONE = 0.02;

double QuantizedAxis(s32 raw) {
    const double value =
        std::clamp(static_cast<double>(raw) / HID::HID_JOYSTICK_MAX, -1.0, 1.0);
    if (std::abs(value) < RECORD_STICK_DEADZONE) {
        return 0.0;
    }
    return std::round(value * RECORD_STICK_STEPS) / RECORD_STICK_STEPS;
}

HID::AnalogStickState QuantizedStick(const HID::AnalogStickState& stick) {
    // The same scaling ParseInputStep applies, so a written and re-read step compares equal.
    const auto axis = [](s32 raw) {
        return static_cast<s32>(std::lround(QuantizedAxis(raw) * HID::HID_JOYSTICK_MAX));
    };
    return {.x = axis(stick.x), .y = axis(stick.y)};
}

bool SameStick(const std::optional<HID::AnalogStickState>& a,
               const std::optional<HID::AnalogStickState>& b) {
    return a.has_value() == b.has_value() && (!a || (a->x == b->x && a->y == b->y));
}

bool SameInput(const HID::InjectedInput& a, const HID::InjectedInput& b) {
    return a.buttons == b.buttons && SameStick(a.left, b.left) && SameStick(a.right, b.right);
}

} // Anonymous namespace

HID::InjectedInput RecordedInput(HID::NpadButton buttons, const HID::AnalogSticks& sticks) {
    HID::NpadButton kept = HID::NpadButton::None;
    for (const auto& [_, button] : INPUT_BUTTONS) {
        kept |= buttons & button;
    }
    return {.buttons = kept, .left = QuantizedStick(sticks.left), .right = QuantizedStick(sticks.right)};
}

void AppendRecordedInput(std::vector<InputStep>& steps, const HID::InjectedInput& input,
                         u64 frames) {
    while (frames > 0) {
        if (steps.empty() || !SameInput(steps.back().input, input) ||
            steps.back().frames >= MAX_INPUT_FRAMES) {
            steps.push_back({.input = input, .frames = 0});
        }
        const u64 added = (std::min)(frames, MAX_INPUT_FRAMES - steps.back().frames);
        steps.back().frames += added;
        frames -= added;
    }
}

json InputStepJson(const InputStep& step) {
    json result = {{"frames", step.frames}};
    json buttons = json::array();
    for (const auto& [name, button] : INPUT_BUTTONS) {
        if ((step.input.buttons & button) != HID::NpadButton::None) {
            buttons.push_back(std::string{name});
        }
    }
    if (!buttons.empty()) {
        result["buttons"] = std::move(buttons);
    }
    const auto stick = [&result](const char* key, const std::optional<HID::AnalogStickState>& value) {
        if (value) {
            result[key] = {QuantizedAxis(value->x), QuantizedAxis(value->y)};
        }
    };
    stick("left_stick", step.input.left);
    stick("right_stick", step.input.right);
    return result;
}

Service::Service(System& system, std::string title, u64 program_id, const Endpoint& endpoint,
                 std::string dump_directory)
    : state{std::make_shared<ToolState>(system, std::move(title), program_id,
                                        std::move(dump_directory))} {
    VideoCore::Nx2Debug::BeginSession();
    Common::Log::SetRingCapacity(RING_LINES);
    try {
        server = std::make_unique<Server>(endpoint, MakeHandlers(state));
    } catch (...) {
        VideoCore::Nx2Debug::EndSession();
        throw;
    }
    LOG_INFO(Debug, "Debug server listening on {}:{}, dump dir {}", endpoint.host, server->Port(),
             state->dump_directory.generic_string());
}

Service::~Service() {
    state->stopping.store(true, std::memory_order_relaxed);
    VideoCore::Nx2Debug::EndSession();
    server.reset();
}

std::unique_ptr<Service> MaybeStartService(System& system, std::string title, u64 program_id) {
    const char* const text = std::getenv("EDEN_NX2_DEBUG_SERVER");
    if (text == nullptr || *text == '\0') {
        return nullptr;
    }
    std::string error;
    const auto endpoint = ParseEndpoint(text, error);
    if (!endpoint) {
        LOG_ERROR(Debug, "EDEN_NX2_DEBUG_SERVER={}: {}; the debug server is off", text, error);
        return nullptr;
    }
    const char* const directory = std::getenv("EDEN_NX2_DEBUG_DIR");
    std::string dump_directory = directory != nullptr && *directory != '\0'
        ? std::string{directory}
        : (Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "debug-server").generic_string();
    try {
        return std::make_unique<Service>(system, std::move(title), program_id, *endpoint,
                                         std::move(dump_directory));
    } catch (const std::exception& failure) {
        LOG_ERROR(Debug, "Debug server cannot listen on {}:{}: {}", endpoint->host, endpoint->port,
                  failure.what());
        return nullptr;
    }
}

} // namespace Core::Nx2Debug
