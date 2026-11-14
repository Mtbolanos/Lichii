// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu_split.h"
#include "video_core/macro.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"

namespace Tegra {

struct DmaPusherTestAccess {
    static bool IsMemoryWriter(const Engines::EngineInterface& engine, Engines::EngineTypes type,
                               u32 method) {
        return DmaPusher::IsMemoryWriter(engine, type, method);
    }

    static size_t Process(DmaPusher& pusher, std::span<const CommandHeader> words) {
        return pusher.ProcessCommands(words);
    }
};

} // namespace Tegra

TEST_CASE("Command writer classification excludes only plain compute and DMA state",
          "[video_core][command_writer]") {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Engines::KeplerCompute compute{memory};
    Tegra::Engines::MaxwellDMA dma{memory};
    Tegra::MacroEngine macros{true};
    using Access = Tegra::DmaPusherTestAccess;
    using Type = Tegra::Engines::EngineTypes;
    for (const bool native : {false, true}) {
        CAPTURE(native);
        compute.SetNativeMacroMode(native ? &macros : nullptr, {});
        for (u32 method = 0; method < 0xe00; ++method) {
            CAPTURE(method);
            const bool writer = Access::IsMemoryWriter(compute, Type::KeplerCompute, method);
            if (!native) {
                REQUIRE(writer == compute.execution_mask[method]);
            } else {
                const bool traced = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr &&
                                    (method == 0x5a || (method >= 0xc6 && method <= 0x108));
                const bool effect = traced || method == 0x45 || method == 0x46 || method == 0x47 ||
                                    method == 0x48 || method == 0xae || method == 0xaf ||
                                    method == 0xb0 || method == 0x107 || method == 0x6c3 ||
                                    method == KEPLER_COMPUTE_REG_INDEX(exec_upload) ||
                                    method == KEPLER_COMPUTE_REG_INDEX(data_upload);
                REQUIRE(writer == effect);
            }
        }
    }
    for (u32 method = 0; method < 0xe00; ++method) {
        CAPTURE(method);
        REQUIRE(Access::IsMemoryWriter(dma, Type::MaxwellDMA, method) ==
                dma.execution_mask[method]);
    }
    for (const u32 method : {0xe00U, 0xfffU, 0xffffU, std::numeric_limits<u32>::max()}) {
        CAPTURE(method);
        REQUIRE(Access::IsMemoryWriter(compute, Type::KeplerCompute, method));
        REQUIRE(Access::IsMemoryWriter(dma, Type::MaxwellDMA, method));
    }
}

namespace {

class SerialEngine final : public Tegra::Engines::EngineInterface {
public:
    std::vector<std::pair<u32, u32>> received;
    std::vector<u64> serials;

    void CallMethod(Core::System&, u32 method, u32 value, bool) override {
        received.emplace_back(method, value);
        serials.push_back(VideoCommon::Nx2FastDraw::Serial());
    }
    void CallMultiMethod(Core::System& system, u32 method, const u32* words, u32 amount,
                         u32) override {
        for (u32 i = 0; i < amount; ++i) {
            CallMethod(system, method, words[i], true);
        }
    }
};

} // namespace

TEST_CASE("Unsplit command parser invalidates reuse before other engine state and execution",
          "[video_core][fast_draw][unsplit_fast_draw]") {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    const bool previous = FastDraw::Enabled();
    const bool previous_reads_serial = FastDraw::t_reads_serial;
    const bool previous_serial_bumped = FastDraw::t_serial_bumped;
    FastDraw::SetEnabledForTest(true);
    SCOPE_EXIT {
        FastDraw::SetEnabledForTest(previous);
        FastDraw::t_reads_serial = previous_reads_serial;
        FastDraw::t_serial_bumped = previous_serial_bumped;
    };
    REQUIRE_FALSE(VideoCommon::GpuSplit::Get().Active());
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Control::ChannelState channel{0};
    Tegra::DmaPusher pusher{system, memory, channel};
    SerialEngine engine;
    engine.execution_mask.set(0x50);
    pusher.BindSubchannel(&engine, 0, Tegra::Engines::EngineTypes::Fermi2D);
    const auto header = [](u32 method, u32 count, Tegra::SubmissionMode mode) {
        return Tegra::BuildCommandHeader(static_cast<Tegra::BufferMethods>(method), count, mode);
    };
    const auto process = [&](std::span<const Tegra::CommandHeader> words, bool own_breaks = false) {
        engine.reports_fast_draw_breaks = own_breaks;
        const u64 before = FastDraw::Serial();
        REQUIRE(Tegra::DmaPusherTestAccess::Process(pusher, words) == words.size());
        REQUIRE((FastDraw::Serial() != before) == !own_breaks);
    };
    // A single non-executing method must already break before it is sunk.
    const std::array single{header(0x40, 7, Tegra::SubmissionMode::Inline)};
    process(single);
    REQUIRE(engine.received.empty());
    REQUIRE(engine.method_sink.size() == 1);
    // The increasing parser batches these without passing through CallMethod.
    const std::array increasing{header(0x41, 2, Tegra::SubmissionMode::Increasing),
                                Tegra::CommandHeader{.argument = 8},
                                Tegra::CommandHeader{.argument = 9}};
    process(increasing);
    REQUIRE(engine.method_sink.size() == 3);
    const std::array execute{header(0x50, 10, Tegra::SubmissionMode::Inline)};
    process(execute);
    REQUIRE(engine.method_sink.empty());
    REQUIRE(engine.received.size() == 4);
    REQUIRE(engine.received[0] == std::pair<u32, u32>{0x40, 7});
    REQUIRE(engine.received[1] == std::pair<u32, u32>{0x41, 8});
    REQUIRE(engine.received[2] == std::pair<u32, u32>{0x42, 9});
    REQUIRE(engine.received[3] == std::pair<u32, u32>{0x50, 10});
    for (const u64 serial : engine.serials) {
        REQUIRE(serial == FastDraw::Serial());
    }
    const std::array multi{header(0x50, 2, Tegra::SubmissionMode::NonIncreasing),
                           Tegra::CommandHeader{.argument = 11},
                           Tegra::CommandHeader{.argument = 12}};
    process(multi);
    REQUIRE(engine.received.size() == 6);
    REQUIRE(engine.received[4] == std::pair<u32, u32>{0x50, 11});
    REQUIRE(engine.received[5] == std::pair<u32, u32>{0x50, 12});
    REQUIRE(engine.serials.back() == FastDraw::Serial());
    // Engines such as Maxwell3D report their own breaks; light methods remain reusable.
    process(single, true);
    process(increasing, true);
    process(execute, true);
    process(multi, true);
}
