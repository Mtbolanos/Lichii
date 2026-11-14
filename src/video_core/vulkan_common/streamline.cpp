// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA Streamline, with Streamline's manual hooking.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/display_hdr.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/frame_gen_choice.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#include "video_core/vulkan_common/streamline_frame_gen.h"

#include <sl_helpers.h>
#include <sl_security.h>

#include <softpub.h>
#include <wintrust.h>

namespace Vulkan::Streamline {
namespace {

// A custom engine names itself with its own project id; an application id from NVIDIA is only
// needed without an engine type. Keep it fixed: the driver may key behaviour off it.
constexpr const char* PROJECT_ID = "8d3e2f9a-6b41-4c7e-9a52-1f0c3b7d5e64";
constexpr const wchar_t* INTERPOSER = L"sl.interposer.dll";
constexpr const wchar_t* MODEL = L"nvngx_dlss.dll";
// The first DLSS with preset K.
constexpr u32 MIN_DLSS_MAJOR = 310;

// What DLSS super resolution loads from the loader's folder besides the loader and the model:
// sl.dlss.dll needs sl.common.dll.
constexpr std::array<const char*, 2> DLSS_PLUGIN_FILES{"sl.common.dll", "sl.dlss.dll"};

// Everything frame generation loads from the loader's folder. The SR plugins are here too: a local
// sl.dlss_g.dll has to come with the sl.common.dll of its own release. Reflex on Vulkan goes
// through NvLowLatencyVk.dll.
constexpr std::array<const char*, 7> FRAME_GEN_FILES{
    "sl.common.dll", "sl.dlss.dll",     "sl.dlss_g.dll",      "sl.reflex.dll",
    "sl.pcl.dll",    "nvngx_dlssg.dll", "NvLowLatencyVk.dll",
};

struct FrameGenFeature {
    sl::Feature feature;
    const char* name;
};

constexpr std::array<FrameGenFeature, 3> FRAME_GEN_FEATURES{{
    {sl::kFeatureDLSS_G, "DLSS-G (sl.dlss_g.dll)"},
    {sl::kFeatureReflex, "Reflex (sl.reflex.dll)"},
    {sl::kFeaturePCL, "PCL (sl.pcl.dll)"},
}};

using Version = std::array<u32, 4>;

struct State {
    HMODULE module{};
    Api api{};
    std::wstring plugin_dir;
    std::wstring model_dir;
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    std::vector<std::string> frame_gen_instance_extensions;
    std::vector<std::string> frame_gen_device_extensions;
    std::vector<std::string> frame_gen_device_features;
    QueueCounts frame_gen_queues{};
    /// OS, driver and GPU scheduling needs from the frame generation requirements, for the log.
    std::string frame_gen_needs;
    bool frame_gen_requested{};
    bool frame_gen_pending{};
    std::atomic<bool> frame_gen_available{};
    std::mutex frame_gen_reason_mutex;
    std::string frame_gen_reason;
    bool started{};
    bool attached{};
};

State& GetState() {
    static State state;
    return state;
}

int LogLevel() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_LOG");
    return value ? std::atoi(value) : 0;
}

void LogFromStreamline(sl::LogType type, const char* message) {
    std::string_view text{message ? message : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    switch (type) {
    case sl::LogType::eError:
        LOG_ERROR(Render_Vulkan, "Streamline: {}", text);
        break;
    case sl::LogType::eWarn:
        LOG_WARNING(Render_Vulkan, "Streamline: {}", text);
        break;
    default:
        if (LogLevel() > 0) {
            LOG_INFO(Render_Vulkan, "Streamline: {}", text);
        }
        break;
    }
}

/// Where the user may put sl.interposer.dll: EDEN_STREAMLINE_DIR, the user folder's "streamline"
/// folder, then a "streamline" folder next to the executable.
std::vector<std::filesystem::path> SearchDirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* const dir = std::getenv("EDEN_STREAMLINE_DIR"); dir && *dir) {
        dirs.emplace_back(dir);
    }
    dirs.push_back(UserFolder());
    dirs.push_back(Common::FS::GetExeDirectory() / "streamline");
    return dirs;
}

template <typename T>
bool Resolve(HMODULE module, const char* name, T*& function) {
    function = reinterpret_cast<T*>(GetProcAddress(module, name));
    if (!function) {
        LOG_ERROR(Render_Vulkan, "sl.interposer.dll has no {}", name);
    }
    return function != nullptr;
}

std::optional<Version> ReadVersion(const std::filesystem::path& path) {
    DWORD handle = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &handle);
    if (size == 0) {
        return std::nullopt;
    }
    std::vector<u8> data(size);
    VS_FIXEDFILEINFO* info{};
    UINT length = 0;
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data()) ||
        !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) || !info) {
        return std::nullopt;
    }
    return Version{HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS),
                   HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS)};
}

std::string VersionText(const std::filesystem::path& path) {
    const std::optional<Version> version = path.empty() ? std::nullopt : ReadVersion(path);
    if (!version) {
        return {};
    }
    const Version& v = *version;
    return v[3] == 0 ? fmt::format("{}.{}.{}", v[0], v[1], v[2])
                     : fmt::format("{}.{}.{}.{}", v[0], v[1], v[2], v[3]);
}

/// The NVIDIA app keeps every DLSS it downloads under ProgramData; pick the newest.
std::filesystem::path FindAppModel() {
    const wchar_t* const program_data = _wgetenv(L"ProgramData");
    if (!program_data) {
        return {};
    }
    const std::filesystem::path root =
        std::filesystem::path{program_data} / L"NVIDIA" / L"NGX" / L"models" / L"dlss" / L"versions";
    std::filesystem::path best_path;
    Version best{};
    std::error_code ec;
    for (const auto& version_dir : std::filesystem::directory_iterator(root, ec)) {
        std::error_code files_ec;
        for (const auto& file :
             std::filesystem::directory_iterator(version_dir.path() / L"files", files_ec)) {
            if (file.path().extension() != L".bin") {
                continue;
            }
            const std::optional<Version> version = ReadVersion(file.path());
            if (version && (*version)[0] >= MIN_DLSS_MAJOR && *version > best) {
                best = *version;
                best_path = file.path();
            }
        }
    }
    return best_path;
}

/// Copies the NVIDIA app's DLSS into our folder as nvngx_dlss.dll, the name Streamline loads.
std::filesystem::path UseAppModel(const DlssFiles& files) {
    const std::filesystem::path dir = UserFolder() / L"nvidia-app";
    const std::filesystem::path target = dir / MODEL;
    std::error_code ec;
    const bool current = std::filesystem::is_regular_file(target, ec) &&
                         std::filesystem::file_size(target, ec) ==
                             std::filesystem::file_size(files.app_model, ec) &&
                         ReadVersion(target) == ReadVersion(files.app_model);
    if (!current) {
        if (!IsSignedByNvidia(files.app_model)) {
            LOG_WARNING(Render_Vulkan, "The NVIDIA app's DLSS {} is not signed by NVIDIA",
                        Common::FS::PathToUTF8String(files.app_model));
            return {};
        }
        std::filesystem::create_directories(dir, ec);
        std::filesystem::copy_file(files.app_model, target,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            LOG_WARNING(Render_Vulkan, "Could not copy the NVIDIA app's DLSS to {}: {}",
                        Common::FS::PathToUTF8String(target), ec.message());
            return {};
        }
    }
    if (!IsSignedByNvidia(target)) {
        LOG_WARNING(Render_Vulkan, "Cached DLSS {} is not signed by NVIDIA",
                    Common::FS::PathToUTF8String(target));
        return {};
    }
    return dir;
}

bool Load(State& state, const std::filesystem::path& path) {
    if (!IsStreamlineModule(path)) {
        LOG_ERROR(Render_Vulkan, "{} is not signed by NVIDIA; not loading it",
                  Common::FS::PathToUTF8String(path));
        return false;
    }
    state.module = LoadLibraryW(path.c_str());
    if (!state.module) {
        LOG_ERROR(Render_Vulkan, "Could not load {} (error {})", Common::FS::PathToUTF8String(path),
                  GetLastError());
        return false;
    }
    Api& api = state.api;
    const HMODULE m = state.module;
    if (Resolve(m, "slInit", api.init) && Resolve(m, "slShutdown", api.shutdown) &&
        Resolve(m, "slIsFeatureSupported", api.is_feature_supported) &&
        Resolve(m, "slGetFeatureRequirements", api.get_feature_requirements) &&
        Resolve(m, "slGetFeatureVersion", api.get_feature_version) &&
        Resolve(m, "slGetFeatureFunction", api.get_feature_function) &&
        Resolve(m, "slGetNewFrameToken", api.get_new_frame_token) &&
        Resolve(m, "slSetConstants", api.set_constants) &&
        Resolve(m, "slEvaluateFeature", api.evaluate_feature) &&
        Resolve(m, "slFreeResources", api.free_resources) &&
        Resolve(m, "slSetVulkanInfo", api.set_vulkan_info)) {
        // Only frame generation uses these; it checks for them itself.
        api.set_tag_for_frame =
            reinterpret_cast<PFun_slSetTagForFrame*>(GetProcAddress(m, "slSetTagForFrame"));
        api.set_feature_loaded =
            reinterpret_cast<PFun_slSetFeatureLoaded*>(GetProcAddress(m, "slSetFeatureLoaded"));
        return true;
    }
    FreeLibrary(state.module);
    state.module = nullptr;
    state.api = {};
    return false;
}

std::vector<std::string> Names(uint32_t count, const char** names) {
    std::vector<std::string> result;
    for (uint32_t i = 0; i < count; ++i) {
        result.emplace_back(names[i]);
    }
    return result;
}

void AppendUnique(std::vector<std::string>& list, uint32_t count, const char** names) {
    for (uint32_t i = 0; i < count; ++i) {
        if (std::ranges::find(list, names[i]) == list.end()) {
            list.emplace_back(names[i]);
        }
    }
}

std::string ResultText(sl::Result result) {
    return fmt::format("{} ({})", sl::getResultAsStr(result), static_cast<int>(result));
}

void SetFrameGenReason(State& state, std::string reason) {
    std::scoped_lock lock{state.frame_gen_reason_mutex};
    state.frame_gen_reason = std::move(reason);
}

/// Why slIsFeatureSupported turned a frame generation feature down, in the user's terms.
std::string UnsupportedText(sl::Result result) {
    switch (result) {
    case sl::Result::eErrorOSDisabledHWS:
        return "Hardware-accelerated GPU scheduling is off in Windows' graphics settings";
    case sl::Result::eErrorNoSupportedAdapterFound:
    case sl::Result::eErrorAdapterNotSupported:
        return "this GPU has no DLSS Frame Generation (GeForce RTX 40 series or newer)";
    case sl::Result::eErrorDriverOutOfDate:
        return "the NVIDIA driver is too old";
    case sl::Result::eErrorOSOutOfDate:
        return "Windows is too old";
    default:
        return fmt::format("Streamline result {}", ResultText(result));
    }
}

/// What the frame generation plugins ask of the OS, the driver and Windows, for the log.
std::string NeedsText(const sl::FeatureRequirements& requirements) {
    std::string text;
    const auto add = [&text](const std::string& part) {
        text += (text.empty() ? "" : ", ") + part;
    };
    if (requirements.driverVersionRequired.major != 0) {
        add(fmt::format("NVIDIA driver {} (found {})", requirements.driverVersionRequired.toStr(),
                        requirements.driverVersionDetected.toStr()));
    }
    if (requirements.osVersionRequired.major != 0) {
        add(fmt::format("Windows {} (found {})", requirements.osVersionRequired.toStr(),
                        requirements.osVersionDetected.toStr()));
    }
    if (requirements.flags & sl::FeatureRequirementFlags::eHardwareSchedulingRequired) {
        add("Hardware-accelerated GPU scheduling on");
    }
    if (requirements.flags & sl::FeatureRequirementFlags::eVSyncOffRequired) {
        add("VSync off");
    }
    return text;
}

/// Collects what DLSS-G, Reflex and PCL need from the instance and device; false (logged) when
/// one of them can't be used.
bool LoadFrameGenRequirements(State& state) {
    const Api& api = state.api;
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    std::vector<std::string> device_features;
    QueueCounts queues{};
    std::string needs;
    for (const FrameGenFeature& entry : FRAME_GEN_FEATURES) {
        sl::FeatureRequirements requirements{};
        if (const sl::Result result = api.get_feature_requirements(entry.feature, requirements);
            result != sl::Result::eOk) {
            const std::vector<std::string> missing = FindDlssFiles().frame_gen_missing;
            std::string files;
            for (const std::string& name : missing) {
                files += (files.empty() ? "" : ", ") + name;
            }
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS Frame Generation is off: Streamline has no {}, result {}{}",
                        entry.name, ResultText(result),
                        files.empty() ? std::string{}
                                      : fmt::format(" (missing from the loader's folder: {}; the "
                                                    "Graphics settings can download them)",
                                                    files));
            SetFrameGenReason(state, files.empty()
                                         ? fmt::format("Streamline has no {}", entry.name)
                                         : fmt::format("missing {}", files));
            return false;
        }
        const std::string feature_needs = NeedsText(requirements);
        if (!(requirements.flags & sl::FeatureRequirementFlags::eVulkanSupported)) {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS Frame Generation is off: this {} has no Vulkan support",
                        entry.name);
            SetFrameGenReason(state, fmt::format("this {} has no Vulkan support", entry.name));
            return false;
        }
        queues.graphics += requirements.vkNumGraphicsQueuesRequired;
        queues.compute += requirements.vkNumComputeQueuesRequired;
        AppendUnique(instance_extensions, requirements.vkNumInstanceExtensions,
                     requirements.vkInstanceExtensions);
        AppendUnique(device_extensions, requirements.vkNumDeviceExtensions,
                     requirements.vkDeviceExtensions);
        AppendUnique(device_features, requirements.vkNumFeatures12, requirements.vkFeatures12);
        AppendUnique(device_features, requirements.vkNumFeatures13, requirements.vkFeatures13);
        if (!feature_needs.empty()) {
            needs += fmt::format("{}{}: {}", needs.empty() ? "" : "; ", entry.name, feature_needs);
        }
    }
    state.frame_gen_instance_extensions = std::move(instance_extensions);
    state.frame_gen_device_extensions = std::move(device_extensions);
    state.frame_gen_device_features = std::move(device_features);
    state.frame_gen_queues = queues;
    state.frame_gen_needs = std::move(needs);
    return true;
}

template <typename T>
bool ResolveFeature(const Api& api, sl::Feature feature, const char* name, T*& function) {
    void* address = nullptr;
    const sl::Result result = api.get_feature_function(feature, name, address);
    function = result == sl::Result::eOk ? reinterpret_cast<T*>(address) : nullptr;
    if (!function) {
        LOG_WARNING(Render_Vulkan, "Streamline has no {}, result {}", name, ResultText(result));
    }
    return function != nullptr;
}

/// Checks DLSS-G, Reflex and PCL on the attached device and fetches their functions.
bool AttachFrameGen(State& state, const sl::AdapterInfo& adapter) {
    Api& api = state.api;
    for (const FrameGenFeature& entry : FRAME_GEN_FEATURES) {
        if (const sl::Result result = api.is_feature_supported(entry.feature, adapter);
            result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS Frame Generation is off: {} is not supported on this GPU, "
                        "driver or Windows setup, result {}{}",
                        entry.name, ResultText(result),
                        state.frame_gen_needs.empty()
                            ? std::string{}
                            : fmt::format(" (needs {})", state.frame_gen_needs));
            SetFrameGenReason(state, UnsupportedText(result));
            return false;
        }
    }
    if (!api.set_tag_for_frame) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS Frame Generation is off: sl.interposer.dll has no slSetTagForFrame");
        SetFrameGenReason(state, "sl.interposer.dll has no slSetTagForFrame");
        return false;
    }
    if (ResolveFeature(api, sl::kFeatureDLSS_G, "slDLSSGSetOptions", api.dlssg_set_options) &&
        ResolveFeature(api, sl::kFeatureDLSS_G, "slDLSSGGetState", api.dlssg_get_state) &&
        ResolveFeature(api, sl::kFeatureReflex, "slReflexSetOptions", api.reflex_set_options) &&
        ResolveFeature(api, sl::kFeatureReflex, "slReflexSleep", api.reflex_sleep) &&
        ResolveFeature(api, sl::kFeatureReflex, "slReflexGetState", api.reflex_get_state) &&
        ResolveFeature(api, sl::kFeaturePCL, "slPCLSetMarker", api.pcl_set_marker)) {
        return true;
    }
    LOG_WARNING(Render_Vulkan, "NVIDIA DLSS Frame Generation is off: its functions are missing");
    SetFrameGenReason(state, "Streamline lacks DLSS-G, Reflex or PCL functions");
    return false;
}

void ClearFrameGenApi(Api& api) {
    api.dlssg_set_options = nullptr;
    api.dlssg_get_state = nullptr;
    api.reflex_set_options = nullptr;
    api.reflex_sleep = nullptr;
    api.reflex_get_state = nullptr;
    api.pcl_set_marker = nullptr;
}

} // Anonymous namespace

std::filesystem::path UserFolder() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::StreamlineDir);
}

bool IsStreamlineModule(const std::filesystem::path& path) {
    return sl::security::verifyEmbeddedSignature(path.c_str());
}

bool IsSignedByNvidia(const std::filesystem::path& path) {
    const std::wstring file_path = std::filesystem::absolute(path).wstring();
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof(file);
    file.pcwszFilePath = file_path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    bool valid = WinVerifyTrust(nullptr, &action, &data) == ERROR_SUCCESS;
    if (valid) {
        CRYPT_PROVIDER_DATA* const provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        CRYPT_PROVIDER_SGNR* const signer =
            provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        CRYPT_PROVIDER_CERT* const cert = signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert &&
                CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name,
                                   static_cast<DWORD>(std::size(name))) > 1 &&
                std::wstring_view{name} == L"NVIDIA Corporation";
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

DlssFiles FindDlssFiles() {
    DlssFiles files;
    files.folder = UserFolder();
    std::error_code ec;
    for (const std::filesystem::path& dir : SearchDirs()) {
        if (std::filesystem::is_regular_file(dir / INTERPOSER, ec)) {
            files.folder = dir;
            files.interposer = dir / INTERPOSER;
            break;
        }
    }
    if (std::filesystem::is_regular_file(files.folder / MODEL, ec)) {
        files.model = files.folder / MODEL;
    }
    files.app_model = FindAppModel();
    files.interposer_version = VersionText(files.interposer);
    files.model_version = VersionText(files.model);
    files.app_model_version = VersionText(files.app_model);
    // A Streamline module loads only with the sl.common.dll and loader of its own release.
    const auto check = [&files, &ec](std::span<const char* const> names,
                                     std::vector<std::string>& missing) {
        std::string streamline_version = files.interposer_version;
        bool mixed = false;
        for (const char* const name : names) {
            const std::filesystem::path path = files.folder / name;
            if (!std::filesystem::is_regular_file(path, ec)) {
                missing.emplace_back(name);
                continue;
            }
            if (!std::string_view{name}.starts_with("sl.")) {
                continue;
            }
            const std::string version = VersionText(path);
            if (streamline_version.empty()) {
                streamline_version = version;
            } else if (!version.empty() && version != streamline_version) {
                mixed = true;
            }
        }
        return mixed;
    };
    files.dlss_mixed = check(DLSS_PLUGIN_FILES, files.dlss_missing);
    files.streamline_mixed = check(FRAME_GEN_FILES, files.frame_gen_missing);
    return files;
}

std::span<const char* const> DlssPluginFiles() {
    return DLSS_PLUGIN_FILES;
}

std::span<const char* const> FrameGenFiles() {
    return FRAME_GEN_FILES;
}

bool FrameGenRequested() {
    return Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::NvidiaDlssG &&
           VideoCore::BlockedDlssg({
               .streamline = true,
               .dlss = Settings::values.nvidia_dlss.GetValue(),
               .program_id = Settings::GetCurrentProgramID(),
               .loaded = true,
           }) == VideoCore::DlssgBlock::None;
}

bool Start(void* window) {
    State& state = GetState();
    if (state.started) {
        return true;
    }
    if (!Settings::values.nvidia_dlss.GetValue()) {
        return false;
    }
    SetFrameGenReason(state, "NVIDIA Streamline did not start; the log above says why");
    if (!state.module) {
        const DlssFiles files = FindDlssFiles();
        if (!files.model.empty() && !IsSignedByNvidia(files.model)) {
            LOG_WARNING(Render_Vulkan, "DLSS model {} is not signed by NVIDIA; using FSR 3.1",
                        Common::FS::PathToUTF8String(files.model));
            return false;
        }
        if (files.interposer.empty()) {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS is on, but sl.interposer.dll (NVIDIA Streamline) is not in "
                        "{} (the Graphics settings can download it); using FSR 3.1",
                        Common::FS::PathToUTF8String(files.folder));
            return false;
        }
        if (!Load(state, files.interposer)) {
            return false;
        }
        state.plugin_dir = files.folder.wstring();
        state.model_dir.clear();
        if (files.model.empty() && !files.app_model.empty()) {
            if (const std::filesystem::path dir = UseAppModel(files); !dir.empty()) {
                state.model_dir = dir.wstring();
                LOG_INFO(Render_Vulkan, "Using the NVIDIA app's DLSS {}", files.app_model_version);
            }
        } else if (files.model.empty()) {
            LOG_WARNING(Render_Vulkan, "No nvngx_dlss.dll in {} and no DLSS from the NVIDIA app",
                        Common::FS::PathToUTF8String(files.folder));
        }
    }

    // Streamline's plugins come from the loader's folder or the NVIDIA app's downloads (OTA); the
    // DLSS model from the loader's folder, else the NVIDIA app's copy.
    const wchar_t* plugin_paths[] = {state.plugin_dir.c_str(), state.model_dir.c_str()};
    state.frame_gen_requested = FrameGenRequested();
    std::vector<sl::Feature> features{sl::kFeatureDLSS};
    if (state.frame_gen_requested) {
        for (const FrameGenFeature& entry : FRAME_GEN_FEATURES) {
            features.push_back(entry.feature);
        }
    }
    sl::Preferences preferences{};
    preferences.logLevel = LogLevel() >= 2 ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    preferences.logMessageCallback = LogFromStreamline;
    preferences.pathsToPlugins = plugin_paths;
    preferences.numPathsToPlugins = state.model_dir.empty() ? 1 : 2;
    preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                        sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eAllowOTA |
                        sl::PreferenceFlags::eLoadDownloadedPlugins;
    if (state.frame_gen_requested) {
        // DLSS-G's inputs are tagged per frame with slSetTagForFrame.
        preferences.flags |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    }
    preferences.featuresToLoad = features.data();
    preferences.numFeaturesToLoad = static_cast<uint32_t>(features.size());
    preferences.engine = sl::EngineType::eCustom;
    preferences.engineVersion = "1.0";
    preferences.projectId = PROJECT_ID;
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    const Api& api = state.api;
    if (const sl::Result result = api.init(preferences, sl::kSDKVersion);
        result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "NVIDIA Streamline failed to start (result {}); using FSR 3.1",
                  static_cast<int>(result));
        return false;
    }
    sl::FeatureRequirements requirements{};
    if (const sl::Result result = api.get_feature_requirements(sl::kFeatureDLSS, requirements);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan,
                    "Streamline found no DLSS plugin (result {}); put sl.common.dll and "
                    "sl.dlss.dll next to sl.interposer.dll (the Graphics settings can download "
                    "them) or install the NVIDIA app. Using FSR 3.1",
                    static_cast<int>(result));
        api.shutdown();
        return false;
    }
    if (requirements.vkNumComputeQueuesRequired != 0 ||
        requirements.vkNumGraphicsQueuesRequired != 0) {
        LOG_WARNING(Render_Vulkan, "This DLSS wants extra Vulkan queues; using FSR 3.1");
        api.shutdown();
        return false;
    }
    state.instance_extensions =
        Names(requirements.vkNumInstanceExtensions, requirements.vkInstanceExtensions);
    state.device_extensions =
        Names(requirements.vkNumDeviceExtensions, requirements.vkDeviceExtensions);
    Common::DisplayHdrMonitor::Instance().Watch(window);
    if (!state.frame_gen_requested) {
        SetFrameGenReason(state, "it was not the frame generation type when the game started");
    } else if (Settings::values.hdr_swapchain.GetValue() == Settings::HdrSwapchain::Scrgb &&
               Common::ResolveHdrOutput(Common::DisplayHdrMonitor::Instance().Current(),
                                        Settings::values.hdr_mode.GetValue(),
                                        Settings::values.hdr_peak_luminance.GetValue(),
                                        Settings::values.hdr_reference_white.GetValue())
                   .enabled) {
        state.frame_gen_requested = false;
        SetFrameGenReason(state, "the HDR format is scRGB; set it to HDR10 to use it");
    }
    state.frame_gen_pending = state.frame_gen_requested && LoadFrameGenRequirements(state);
    state.frame_gen_available = false;
    ResetFrameGen();
    sl::FeatureVersion version{};
    api.get_feature_version(sl::kFeatureDLSS, version);
    LOG_INFO(Render_Vulkan, "NVIDIA Streamline {} started from {} (DLSS {})",
             version.versionSL.toStr(),
             Common::FS::PathToUTF8String(std::filesystem::path{state.plugin_dir}),
             version.versionNGX.toStr());
    state.started = true;
    return true;
}

bool WantsDevice() {
    const State& state = GetState();
    return state.started && !state.attached;
}

std::vector<std::string> InstanceExtensions() {
    const State& state = GetState();
    if (!state.started) {
        return {};
    }
    std::vector<std::string> extensions = state.instance_extensions;
    if (state.frame_gen_pending) {
        for (const std::string& name : state.frame_gen_instance_extensions) {
            if (std::ranges::find(extensions, name) == extensions.end()) {
                extensions.push_back(name);
            }
        }
    }
    return extensions;
}

std::vector<std::string> DeviceExtensions() {
    const State& state = GetState();
    if (!state.started) {
        return {};
    }
    std::vector<std::string> extensions = state.device_extensions;
    if (state.frame_gen_pending) {
        for (const std::string& name : state.frame_gen_device_extensions) {
            if (std::ranges::find(extensions, name) == extensions.end()) {
                extensions.push_back(name);
            }
        }
    }
    // The KHR and EXT buffer device address extensions can't both be enabled.
    if (std::ranges::find(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) !=
        extensions.end()) {
        std::erase(extensions, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }
    return extensions;
}

bool FrameGenPending() {
    const State& state = GetState();
    return state.started && !state.attached && state.frame_gen_pending;
}

void DropFrameGen(std::string_view reason) {
    State& state = GetState();
    if (!state.frame_gen_pending) {
        return;
    }
    state.frame_gen_pending = false;
    LOG_WARNING(Render_Vulkan, "NVIDIA DLSS Frame Generation is off: {}", reason);
    SetFrameGenReason(state, std::string{reason});
}

QueueCounts ExtraQueues() {
    return FrameGenPending() ? GetState().frame_gen_queues : QueueCounts{};
}

std::vector<std::string> FrameGenDeviceFeatures() {
    return FrameGenPending() ? GetState().frame_gen_device_features : std::vector<std::string>{};
}

bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, u32 first_extra_queue, vk::DeviceDispatch& dld) {
    State& state = GetState();
    if (!WantsDevice()) {
        return false;
    }
    sl::VulkanInfo info{};
    info.device = device;
    info.instance = instance;
    info.physicalDevice = physical;
    info.computeQueueFamily = graphics_family;
    info.graphicsQueueFamily = graphics_family;
    const QueueCounts queues = ExtraQueues();
    if (queues.graphics != 0) {
        info.graphicsQueueIndex = first_extra_queue;
    }
    if (queues.compute != 0) {
        info.computeQueueIndex = first_extra_queue + queues.graphics;
    }
    if (const sl::Result result = state.api.set_vulkan_info(info); result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "Streamline did not take the Vulkan device (result {})",
                  static_cast<int>(result));
        Stop();
        return false;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physical;
    if (const sl::Result result = state.api.is_feature_supported(sl::kFeatureDLSS, adapter);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS is not supported on this GPU or driver (Streamline result {}); "
                    "using FSR 3.1",
                    static_cast<int>(result));
        Stop();
        return false;
    }
    // Manual hooking: Streamline does its per-frame work in these.
    const auto hook = [&](auto& function, const char* name) {
        if (const FARPROC proxy = GetProcAddress(state.module, name)) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(proxy);
        } else {
            LOG_WARNING(Render_Vulkan, "sl.interposer.dll has no {}", name);
        }
    };
    hook(dld.vkCreateSwapchainKHR, "vkCreateSwapchainKHR");
    hook(dld.vkDestroySwapchainKHR, "vkDestroySwapchainKHR");
    hook(dld.vkGetSwapchainImagesKHR, "vkGetSwapchainImagesKHR");
    hook(dld.vkAcquireNextImageKHR, "vkAcquireNextImageKHR");
    hook(dld.vkQueuePresentKHR, "vkQueuePresentKHR");
    hook(dld.vkDeviceWaitIdle, "vkDeviceWaitIdle");
    state.attached = true;
    LOG_INFO(Render_Vulkan, "NVIDIA DLSS is available through Streamline");

    if (state.frame_gen_pending && AttachFrameGen(state, adapter)) {
        state.frame_gen_available = true;
        QueryFrameGenMax();
        sl::FeatureVersion version{};
        state.api.get_feature_version(sl::kFeatureDLSS_G, version);
        LOG_INFO(Render_Vulkan,
                 "Streamline: DLSS Frame Generation ready (DLSS-G {}, Reflex, PCL; {} graphics + "
                 "{} compute queues; numFramesToGenerateMax {})",
                 version.versionNGX ? version.versionNGX.toStr() : version.versionSL.toStr(),
                 queues.graphics, queues.compute, FrameGenMaxGenerated());
        // DLSS-G refuses to run without Reflex on.
        sl::ReflexOptions reflex{};
        reflex.mode = sl::ReflexMode::eLowLatency;
        reflex.frameLimitUs = 0;
        if (const sl::Result result = state.api.reflex_set_options(reflex);
            result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan, "DLSS-G: slReflexSetOptions failed, result {}",
                        ResultText(result));
        }
    } else if (state.frame_gen_requested) {
        ClearFrameGenApi(state.api);
        // A loaded DLSS-G puts its own swapchain behind the hooks; without frame generation the
        // renderer keeps the driver's.
        if (state.api.set_feature_loaded) {
            state.api.set_feature_loaded(sl::kFeatureDLSS_G, false);
        }
        LOG_INFO(Render_Vulkan, "Streamline: DLSS Frame Generation is off; DLSS only");
    }
    state.frame_gen_pending = false;
    return true;
}

VkSurfaceKHR CreateFrameGenSurface(VkInstance instance, void* window) {
    const State& state = GetState();
    if (!FrameGenSession() || !window) {
        return VK_NULL_HANDLE;
    }
    const auto create = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        GetProcAddress(state.module, "vkCreateWin32SurfaceKHR"));
    if (!create) {
        LOG_WARNING(Render_Vulkan, "sl.interposer.dll has no vkCreateWin32SurfaceKHR");
        return VK_NULL_HANDLE;
    }
    const VkWin32SurfaceCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .hinstance = nullptr,
        .hwnd = static_cast<HWND>(window),
    };
    VkSurfaceKHR surface{};
    if (const VkResult result = create(instance, &info, nullptr, &surface);
        result != VK_SUCCESS) {
        LOG_WARNING(Render_Vulkan, "Streamline's vkCreateWin32SurfaceKHR failed ({})",
                    static_cast<int>(result));
        return VK_NULL_HANDLE;
    }
    return surface;
}

void DisableFrameGen(std::string_view reason) {
    State& state = GetState();
    {
        // The DLSS-G calls check FrameGenSession() under this lock, so none runs once the flag
        // is clear. The function pointers stay for the threads past an unlocked check.
        const auto lock = LockStreamline();
        if (!state.frame_gen_available.exchange(false)) {
            return;
        }
        if (state.api.set_feature_loaded) {
            state.api.set_feature_loaded(sl::kFeatureDLSS_G, false);
        }
    }
    LOG_WARNING(Render_Vulkan, "NVIDIA DLSS Frame Generation is off: {}", reason);
    SetFrameGenReason(state, std::string{reason});
}

bool IsDlssAvailable() {
    const State& state = GetState();
    return state.started && state.attached;
}

bool FrameGenSession() {
    const State& state = GetState();
    return IsDlssAvailable() && state.frame_gen_available;
}

bool FrameGenActive() {
    return FrameGenSession() && Settings::values.frame_gen.GetValue() &&
           Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::NvidiaDlssG;
}

std::string FrameGenUnavailableReason() {
    State& state = GetState();
    std::scoped_lock lock{state.frame_gen_reason_mutex};
    return state.frame_gen_reason;
}

void Stop() {
    State& state = GetState();
    if (!state.started) {
        return;
    }
    const sl::Result result = state.api.shutdown();
    state.started = false;
    state.attached = false;
    state.frame_gen_pending = false;
    state.frame_gen_available = false;
    ClearFrameGenApi(state.api);
    ResetFrameGen();
    LOG_INFO(Render_Vulkan, "NVIDIA Streamline stopped (result {})", static_cast<int>(result));
}

const Api& GetApi() {
    return GetState().api;
}

} // namespace Vulkan::Streamline
