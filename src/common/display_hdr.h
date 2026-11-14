// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace Common {

/// What the host reports for the monitor showing a window.
struct DisplayHdrInfo {
    bool hdr_enabled{};      ///< The desktop runs the monitor in BT.2020 PQ (Windows HDR on).
    float max_luminance{};   ///< Peak luminance in nits from the monitor's EDID (0 if unknown).
    float sdr_white_level{}; ///< Nits the desktop shows SDR white at (0 if unknown).

    bool operator==(const DisplayHdrInfo&) const = default;
};

/// Queries the monitor that shows window (an HWND on Windows). Returns nullopt when the platform
/// has no such query or the window is on no known monitor.
std::optional<DisplayHdrInfo> QueryDisplayHdr(void* window);

/// Follows the HDR state of the monitor that shows a window. Win32 has no change notification for
/// it: Microsoft's guidance is to re-read DXGI_OUTPUT_DESC1 once IDXGIFactory1::IsCurrent turns
/// false, and to re-check when the window moves to another display. Windows HDR toggles and
/// output changes invalidate the factory; the SDR content brightness and a new HDR calibration
/// profile may not, so a background thread re-reads the whole state every 500 ms (keeping its
/// DXGI factory while it is current). Readers only load the cached state.
class DisplayHdrMonitor {
public:
    DisplayHdrMonitor();
    ~DisplayHdrMonitor();

    DisplayHdrMonitor(const DisplayHdrMonitor&) = delete;
    DisplayHdrMonitor& operator=(const DisplayHdrMonitor&) = delete;

    /// The process-wide monitor, shared by the display service and the presenter.
    static DisplayHdrMonitor& Instance();

    /// Follows window from now on (an HWND on Windows); reads its state at once the first time.
    /// Does nothing on hosts without a query.
    void Watch(void* window);

    /// The last reading; nullopt before Watch, on hosts without a query, or off any monitor.
    [[nodiscard]] std::optional<DisplayHdrInfo> Current() const;

    /// Counts the changes of Current(), so callers can compare it each frame for free.
    [[nodiscard]] u64 Generation() const {
        return generation.load(std::memory_order_acquire);
    }

    /// Stores a reading; returns true, and bumps the generation, when it differs from the last.
    bool Update(const std::optional<DisplayHdrInfo>& info);

    /// Whether the host swapchain currently presents HDR (HDR10 or scRGB). Set by the presenter
    /// whenever it creates a swapchain; a change bumps the generation.
    void SetPresentingHdr(bool hdr);
    [[nodiscard]] bool PresentingHdr() const {
        return presenting_hdr.load(std::memory_order_acquire);
    }

private:
    void Run(std::stop_token stop);

    mutable std::mutex mutex;
    std::optional<DisplayHdrInfo> current;
    std::atomic<u64> generation{};
    std::atomic<void*> window{};
    std::atomic<bool> presenting_hdr{};
    std::jthread thread;
};

/// HDR output as reported to a guest that asks for the display's HDR status.
struct HdrOutput {
    bool enabled{};
    u16 peak_luminance{};  ///< nits
    u16 reference_white{}; ///< nits, at most peak_luminance
};

/// Combines the HDR settings with the display state. Off and On decide alone; Auto turns HDR on
/// only while the host reports the monitor in HDR. Luminance settings of 0 follow the monitor.
HdrOutput ResolveHdrOutput(const std::optional<DisplayHdrInfo>& display, Settings::HdrMode mode,
                           u16 peak_setting, u16 reference_white_setting);

/// What the guest is told: ResolveHdrOutput, but HDR only while the host swapchain really
/// presents HDR, so a game never renders PQ into an SDR swapchain.
HdrOutput ResolveGuestHdrOutput(const std::optional<DisplayHdrInfo>& display,
                                Settings::HdrMode mode, u16 peak_setting,
                                u16 reference_white_setting, bool presenting_hdr);

} // namespace Common
