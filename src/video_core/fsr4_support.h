// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <optional>
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

/// FSR 4 through fsr4-vk (github.com/Astyyyyy/fsr4-vk), the community Vulkan port of FSR 4.1.1
/// that the Linux packages ship. It runs AMD's INT8 network on RDNA 2 and newer Radeons and AMD's
/// FP8 network where the device has FP8 cooperative matrices (RDNA 4), on Mesa's RADV driver.
enum class Fsr4PortModel { Int8, Fp8 };

/// What fsr4-vk needs, as enabled on the logical device (or supported, before one exists).
struct Fsr4PortDevice {
    u32 vendor_id;
    u32 device_id;
    bool radv;
    /// Accelerated mixed-signedness 4x8-bit dot products: RDNA 3 and newer.
    bool mixed_dot4_accelerated;
    /// VK_EXT_descriptor_buffer with buffer device addresses and runtime descriptor arrays.
    bool descriptor_buffer;
    bool synchronization2;
    /// Compute pipelines can require full 32-lane subgroups.
    bool compute_wave32;
    /// shaderInt8, shaderInt16, shaderFloat16 with float16 denormals preserved, and 4x8-bit
    /// integer dot products.
    bool shader_arithmetic;
    /// Linear compute shader derivatives and quad subgroup operations in compute shaders.
    bool compute_derivatives;
    /// 16x16x16 FP16 and FP8 (E4M3) cooperative matrices with FP32 results, shaderFloat8, the
    /// cooperative matrix conversions of VK_NV_cooperative_matrix2, the Vulkan memory model and
    /// 8-bit storage buffers: what the FP8 network's shaders use.
    bool fp8_matrices;
};

struct Fsr4PortSupport {
    std::optional<Fsr4PortModel> model;
    /// Why that model, or why none.
    std::string_view reason;
};

/// RDNA 2 GPUs and APUs by PCI device ID, as fsr4-vk tells them apart (runtime/resources.hpp).
constexpr bool IsAmdRdna2DeviceId(u32 device_id) {
    return (device_id >= 0x73a0 && device_id <= 0x73ff) ||
           (device_id >= 0x7420 && device_id <= 0x743f) || device_id == 0x1506 ||
           device_id == 0x163f || device_id == 0x164e || device_id == 0x1681;
}

/// Whether fsr4-vk runs on a GPU: an RDNA 2 or newer Radeon on RADV. CheckFsr4Port also checks
/// the device's features.
constexpr bool IsFsr4PortGpu(u32 vendor_id, u32 device_id, bool radv, bool mixed_dot4_accelerated) {
    constexpr u32 amd_vendor = 0x1002;
    return vendor_id == amd_vendor && radv &&
           (mixed_dot4_accelerated || IsAmdRdna2DeviceId(device_id));
}

constexpr Fsr4PortSupport CheckFsr4Port(const Fsr4PortDevice& device) {
    if (!IsFsr4PortGpu(device.vendor_id, device.device_id, device.radv,
                       device.mixed_dot4_accelerated)) {
        return {std::nullopt, "needs an RDNA 2 or newer Radeon on Mesa's RADV driver"};
    }
    if (!device.descriptor_buffer) {
        return {std::nullopt, "no VK_EXT_descriptor_buffer with buffer device addresses"};
    }
    if (!device.synchronization2) {
        return {std::nullopt, "no VK_KHR_synchronization2"};
    }
    if (!device.compute_wave32) {
        return {std::nullopt, "compute shaders can't require 32-lane subgroups"};
    }
    if (!device.shader_arithmetic) {
        return {std::nullopt, "no 8/16-bit shader arithmetic or 4x8-bit integer dot products"};
    }
    if (!device.compute_derivatives) {
        return {std::nullopt, "no linear compute shader derivatives or quad subgroup operations"};
    }
    if (device.fp8_matrices) {
        return {Fsr4PortModel::Fp8, "FP8 cooperative matrices (RDNA 4)"};
    }
    return {Fsr4PortModel::Int8, "no FP8 cooperative matrices, so the INT8 network"};
}

} // namespace VideoCommon
