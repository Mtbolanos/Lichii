// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace Common {

struct MemoryInfo {
    u64 TotalPhysicalMemory{};
    u64 TotalSwapMemory{};
};

/**
 * Gets the memory info of the host system
 * @return Reference to a MemoryInfo struct with the physical and swap memory sizes in bytes
 */
[[nodiscard]] const MemoryInfo& GetMemInfo();

/// The host's RAM in bytes. installed: the amount fitted in the machine (Windows' firmware table,
/// macOS). Otherwise it is the memory the OS manages, short of what firmware and an integrated GPU
/// reserve.
struct HostRam {
    u64 bytes{};
    bool installed{};
};

[[nodiscard]] HostRam GetHostRam();

/// The sum of the MEMORY_DEVICE_<n>_SIZE entries in udev's data for the DMI device
/// (/run/udev/data/+dmi:id), or 0 without any.
[[nodiscard]] u64 InstalledRamFromUdev(std::string_view data);

/// The host's RAM from udev's data for the DMI device and the memory the OS manages. The udev total
/// counts as installed RAM only when it covers the managed memory; otherwise the result is the
/// managed memory.
[[nodiscard]] HostRam HostRamFromUdev(std::string_view data, u64 managed);

/// Whether the host has less than the 16 GiB of RAM Switch 2 titles need. Memory the OS manages
/// passes from 12 GiB, so a 16 GiB machine that reserves up to 4 GiB for its GPU still passes.
[[nodiscard]] bool IsBelowSwitch2RamMinimum(HostRam ram);

/// This process's memory in bytes. Zero where the host does not report a value.
struct ProcessMemoryUsage {
    u64 commit{};              ///< Private commit (Windows PrivateUsage, Linux VmData).
    u64 working_set{};         ///< Resident memory.
    u64 private_working_set{}; ///< Resident memory not shared with other mappings.
    u64 system_commit{};       ///< Commit charge of the whole system (Windows).
    u64 system_commit_limit{}; ///< Commit limit of the whole system (Windows).
};

[[nodiscard]] ProcessMemoryUsage GetProcessMemoryUsage();

/// Commit the system can still hand out, in bytes: its commit limit (RAM plus the page files)
/// minus what is committed. No value where the host reports no such limit (anything but Windows).
[[nodiscard]] std::optional<u64> GetSystemCommitRoom();

/// Physical memory the host can hand out now without paging anything out (0 when unknown).
[[nodiscard]] u64 GetAvailablePhysicalMemory();

/// Whether the host is short of RAM: less than a quarter of it, and less than 4 GiB, free.
[[nodiscard]] constexpr bool IsLowOnMemory(u64 available, u64 total) {
    return available != 0 && available < (std::min<u64>)(total / 4, u64{4} << 30);
}

/// IsLowOnMemory for this host, read again at most once a second.
[[nodiscard]] bool IsHostLowOnMemory();

} // namespace Common
