// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/fsr4_support.h"

TEST_CASE("FSR 4 is offered on RDNA 2 and newer Radeons on AMD's driver", "[video_core][fsr4]") {
    using VideoCommon::IsFsr4CapableGpu;
    constexpr u32 amd = 0x1002;
    for (const char* name :
         {"AMD Radeon RX 9070 XT", "AMD Radeon RX 9060 XT", "AMD Radeon RX 7900 XTX",
          "AMD Radeon RX 7600S", "AMD Radeon RX 6800 XT", "AMD Radeon RX 6700M",
          "AMD Radeon PRO W7900", "AMD Radeon(TM) 890M Graphics", "AMD Radeon(TM) 880M",
          "AMD Radeon(TM) 8060S Graphics", "AMD Radeon 780M Graphics"}) {
        INFO(name);
        CHECK(IsFsr4CapableGpu(amd, true, name));
    }
    for (const char* name :
         {"AMD Radeon RX 5700 XT", "AMD Radeon RX 580", "AMD Radeon RX Vega", "AMD Radeon(TM) 680M",
          "AMD Radeon(TM) Graphics", "AMD Radeon VII", "AMD Radeon RX 590 GME"}) {
        INFO(name);
        CHECK_FALSE(IsFsr4CapableGpu(amd, true, name));
    }
    CHECK_FALSE(IsFsr4CapableGpu(amd, false, "AMD Radeon RX 9070 XT (RADV GFX1201)"));
    CHECK_FALSE(IsFsr4CapableGpu(0x10DE, true, "NVIDIA GeForce RTX 3080"));
    CHECK_FALSE(IsFsr4CapableGpu(0x8086, true, "Intel(R) Arc(TM) A770 Graphics"));
}

TEST_CASE("The fsr4-vk FSR 4 port runs on RDNA 2 and newer Radeons on RADV with its features",
          "[video_core][fsr4]") {
    using VideoCommon::CheckFsr4Port;
    using VideoCommon::Fsr4PortDevice;
    using VideoCommon::Fsr4PortModel;
    constexpr Fsr4PortDevice rdna3{
        .vendor_id = 0x1002,
        .device_id = 0x744c, // Navi 31
        .radv = true,
        .mixed_dot4_accelerated = true,
        .descriptor_buffer = true,
        .synchronization2 = true,
        .compute_wave32 = true,
        .shader_arithmetic = true,
        .compute_derivatives = true,
        .fp8_matrices = false,
    };
    CHECK(CheckFsr4Port(rdna3).model == Fsr4PortModel::Int8);

    Fsr4PortDevice rdna4 = rdna3;
    rdna4.device_id = 0x7550; // Navi 48
    rdna4.fp8_matrices = true;
    CHECK(CheckFsr4Port(rdna4).model == Fsr4PortModel::Fp8);

    Fsr4PortDevice rdna2 = rdna3;
    rdna2.device_id = 0x73bf; // Navi 21
    rdna2.mixed_dot4_accelerated = false;
    CHECK(CheckFsr4Port(rdna2).model == Fsr4PortModel::Int8);
    rdna2.device_id = 0x1681; // Rembrandt
    CHECK(CheckFsr4Port(rdna2).model == Fsr4PortModel::Int8);

    Fsr4PortDevice rdna1 = rdna2;
    rdna1.device_id = 0x731f; // Navi 10
    CHECK_FALSE(CheckFsr4Port(rdna1).model);

    Fsr4PortDevice amdvlk = rdna4;
    amdvlk.radv = false;
    CHECK_FALSE(CheckFsr4Port(amdvlk).model);

    Fsr4PortDevice nvidia = rdna4;
    nvidia.vendor_id = 0x10DE;
    CHECK_FALSE(CheckFsr4Port(nvidia).model);

    for (bool Fsr4PortDevice::*feature :
         {&Fsr4PortDevice::descriptor_buffer, &Fsr4PortDevice::synchronization2,
          &Fsr4PortDevice::compute_wave32, &Fsr4PortDevice::shader_arithmetic,
          &Fsr4PortDevice::compute_derivatives}) {
        Fsr4PortDevice missing = rdna4;
        missing.*feature = false;
        const auto support = CheckFsr4Port(missing);
        CHECK_FALSE(support.model);
        CHECK_FALSE(support.reason.empty());
    }
}
