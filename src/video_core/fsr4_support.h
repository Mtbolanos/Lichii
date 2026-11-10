// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <string_view>

#include "common/common_types.h"

namespace VideoCommon {

/// Whether a GPU is one AMD's FSR 4 runtime supports: Radeon RX 6000, 7000 and 9000 (and the
/// matching Radeon PRO W cards), and the RDNA 3/3.5 integrated GPUs of Ryzen and Ryzen AI
/// processors, on AMD's own Windows driver. Told apart by the Vulkan device name, which AMD's driver
/// reports as for example "AMD Radeon RX 9070 XT" or "AMD Radeon(TM) 890M Graphics". AMD's runtime
/// still decides at run time, so a wrong guess only shows or hides the option.
constexpr bool IsFsr4CapableGpu(u32 vendor_id, bool amd_driver, std::string_view name) {
    constexpr u32 amd_vendor = 0x1002;
    if (vendor_id != amd_vendor || !amd_driver) {
        return false;
    }
    const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    const auto digits_at = [&](size_t at) {
        size_t count = 0;
        while (at + count < name.size() && is_digit(name[at + count])) {
            ++count;
        }
        return count;
    };
    // "RX 6800 XT", "RX 7600S", "PRO W7900": a four-digit model whose first digit is the series.
    for (const std::string_view prefix : {std::string_view{"RX "}, std::string_view{"PRO W"}}) {
        for (size_t at = name.find(prefix); at != std::string_view::npos;
             at = name.find(prefix, at + 1)) {
            const size_t model = at + prefix.size();
            if (digits_at(model) == 4) {
                const char series = name[model];
                if (series == '6' || series == '7' || series == '9') {
                    return true;
                }
            }
        }
    }
    // Integrated: "780M", "890M", "8060S". RDNA 3 starts at the 700 series; RDNA 2's 680M doesn't
    // run FSR 4.
    for (size_t at = 0; at < name.size(); ++at) {
        if (!is_digit(name[at]) || (at > 0 && is_digit(name[at - 1]))) {
            continue;
        }
        const size_t count = digits_at(at);
        const size_t end = at + count;
        if ((count == 3 || count == 4) && end < name.size() &&
            (name[end] == 'M' || name[end] == 'S') && (name[at] == '7' || name[at] == '8') &&
            (end + 1 == name.size() || name[end + 1] == ' ')) {
            return true;
        }
    }
    return false;
}

} // namespace VideoCommon
