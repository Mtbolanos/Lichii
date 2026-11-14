// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>

#include "common/display_hdr.h"

#ifdef _WIN32
#include <cwchar>
#include <vector>

#include <windows.h>

#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace Common {

#ifdef _WIN32
namespace {

// Nits the desktop shows SDR white at on the monitor whose GDI name is gdi_name: Windows' "SDR
// content brightness". DISPLAYCONFIG_SDR_WHITE_LEVEL counts in 1/1000 of 80 nits.
float QuerySdrWhiteLevel(const wchar_t* gdi_name) {
    UINT32 path_count{}, mode_count{};
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) !=
        ERROR_SUCCESS) {
        return 0.0f;
    }
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count,
                           modes.data(), nullptr) != ERROR_SUCCESS) {
        return 0.0f;
    }
    for (UINT32 i = 0; i < path_count; ++i) {
        const auto& path = paths[i];
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
        source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source.header.size = sizeof(source);
        source.header.adapterId = path.sourceInfo.adapterId;
        source.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
            std::wcscmp(source.viewGdiDeviceName, gdi_name) != 0) {
            continue;
        }
        DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
        white.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
        white.header.size = sizeof(white);
        white.header.adapterId = path.targetInfo.adapterId;
        white.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&white.header) != ERROR_SUCCESS) {
            return 0.0f;
        }
        return static_cast<float>(white.SDRWhiteLevel) * 80.0f / 1000.0f;
    }
    return 0.0f;
}

// Reads the monitor showing window through factory, which must be current.
std::optional<DisplayHdrInfo> QueryDisplayHdr(IDXGIFactory1* factory, HWND window) {
    const HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONULL);
    if (monitor == nullptr) {
        return std::nullopt;
    }
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        Microsoft::WRL::ComPtr<IDXGIOutput> output;
        for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
            Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
            DXGI_OUTPUT_DESC1 desc{};
            if (FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)) ||
                desc.Monitor != monitor) {
                continue;
            }
            // MaxLuminance comes from the display's current colour profile where it has one (a
            // Windows HDR Calibration profile), else from its EDID/DisplayID.
            return DisplayHdrInfo{
                .hdr_enabled = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
                .max_luminance = desc.MaxLuminance,
                .sdr_white_level = QuerySdrWhiteLevel(desc.DeviceName),
            };
        }
    }
    return std::nullopt;
}

} // Anonymous namespace

std::optional<DisplayHdrInfo> QueryDisplayHdr(void* window) {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (window == nullptr || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        return std::nullopt;
    }
    return QueryDisplayHdr(factory.Get(), static_cast<HWND>(window));
}

void DisplayHdrMonitor::Watch(void* new_window) {
    if (new_window == nullptr || window.exchange(new_window) == new_window) {
        return;
    }
    Update(QueryDisplayHdr(new_window));
    if (!thread.joinable()) {
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
    }
}

void DisplayHdrMonitor::Run(std::stop_token stop) {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    std::mutex wait_mutex;
    std::condition_variable_any wake;
    while (!stop.stop_requested()) {
        {
            std::unique_lock lock{wait_mutex};
            wake.wait_for(lock, stop, std::chrono::milliseconds{500}, [] { return false; });
        }
        if (stop.stop_requested()) {
            break;
        }
        if (!factory || !factory->IsCurrent()) {
            factory.Reset();
            if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
                continue;
            }
        }
        Update(QueryDisplayHdr(factory.Get(), static_cast<HWND>(window.load())));
    }
}
#else
std::optional<DisplayHdrInfo> QueryDisplayHdr(void*) {
    return std::nullopt;
}

void DisplayHdrMonitor::Watch(void*) {}

void DisplayHdrMonitor::Run(std::stop_token) {}
#endif

DisplayHdrMonitor::DisplayHdrMonitor() = default;

DisplayHdrMonitor::~DisplayHdrMonitor() = default;

DisplayHdrMonitor& DisplayHdrMonitor::Instance() {
    static DisplayHdrMonitor monitor;
    return monitor;
}

std::optional<DisplayHdrInfo> DisplayHdrMonitor::Current() const {
    std::scoped_lock lock{mutex};
    return current;
}

bool DisplayHdrMonitor::Update(const std::optional<DisplayHdrInfo>& info) {
    std::scoped_lock lock{mutex};
    if (info == current) {
        return false;
    }
    current = info;
    generation.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

void DisplayHdrMonitor::SetPresentingHdr(bool hdr) {
    std::scoped_lock lock{mutex};
    if (presenting_hdr.exchange(hdr, std::memory_order_acq_rel) != hdr) {
        generation.fetch_add(1, std::memory_order_acq_rel);
    }
}

HdrOutput ResolveHdrOutput(const std::optional<DisplayHdrInfo>& display, Settings::HdrMode mode,
                           u16 peak_setting, u16 reference_white_setting) {
    constexpr u32 DefaultPeak = 300; // the Switch 2 HDR calibration's starting value
    const auto nits = [](float value) { return static_cast<u32>(std::lround(value)); };
    HdrOutput output{};
    output.enabled = mode == Settings::HdrMode::On ||
                     (mode == Settings::HdrMode::Auto && display && display->hdr_enabled);
    u32 peak = peak_setting;
    if (peak == 0) {
        peak =
            display && display->max_luminance > 0.0f ? nits(display->max_luminance) : DefaultPeak;
    }
    peak = std::clamp<u32>(peak, 80, 10000);
    u32 white = reference_white_setting;
    if (white == 0 && display && display->sdr_white_level > 0.0f) {
        // The guest's paper white is the brightness its SDR-referred content (HUD, menus) takes;
        // the desktop's SDR white keeps it at the brightness of everything else on screen.
        white = nits(display->sdr_white_level);
    } else if (white == 0) {
        // Switch 2 calibration starts at ceil(peak / 1.5), clamped to 50..300 nits.
        white = std::clamp<u32>((peak * 2 + 2) / 3, 50, 300);
    }
    output.peak_luminance = static_cast<u16>(peak);
    output.reference_white = static_cast<u16>(std::min(white, peak));
    return output;
}

HdrOutput ResolveGuestHdrOutput(const std::optional<DisplayHdrInfo>& display,
                                Settings::HdrMode mode, u16 peak_setting,
                                u16 reference_white_setting, bool presenting_hdr) {
    HdrOutput output = ResolveHdrOutput(display, mode, peak_setting, reference_white_setting);
    output.enabled = output.enabled && presenting_hdr;
    return output;
}

} // namespace Common
