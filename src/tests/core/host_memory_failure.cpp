// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/cpu_features.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/service/am/applet_manager.h"
#include "tests/common/host_memory_test_access.h"
#include "tests/core/system_test_context.h"

namespace {
class MemoryFailureWindow final : public Core::Frontend::EmuWindow {
public:
    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        FAIL("Allocation failure must not initialize the renderer");
        return {};
    }

    bool IsShown() const override {
        return false;
    }
};
} // namespace

TEST_CASE("System rejects missing guest memory before kernel initialization and retries",
          "[host_memory_failure]") {
    ScopedSystemTestContext context;
    const bool previous_switch2 = Settings::CurrentProgramIsSwitch2();
    const auto previous_frequency = Common::g_wall_clock.GetGuestCNTFRQ();
    SCOPE_EXIT {
        Settings::SetCurrentProgramIsSwitch2(previous_switch2);
        Common::g_wall_clock.SetGuestCNTFRQ(previous_frequency);
    };
    Common::HostMemoryTestAccess allocations;
    Core::System system;
    SECTION("Initial allocation fails") {
        REQUIRE_FALSE(system.Initialize());
    }
    SECTION("A new memory layout cannot be allocated") {
        allocations.AllowFallback();
        REQUIRE(system.Initialize());
        allocations.FailFallback();
        Settings::values.memory_layout_mode.SetValue(Settings::MemoryLayout::Memory_6Gb);
    }
    REQUIRE(allocations.FallbackAttempts() == 1);

    MemoryFailureWindow window;
    Service::AM::FrontendAppletParameters params{};
    params.launch_type = Service::AM::LaunchType::ApplicationInitiated;
    CHECK(system.Load(window, "", params) == Core::SystemResultStatus::ErrorMemoryAllocation);
    CHECK_FALSE(system.IsPoweredOn());
    CHECK(allocations.FallbackAttempts() == 2);
    CHECK(system.Load(window, "", params) == Core::SystemResultStatus::ErrorMemoryAllocation);
    CHECK(allocations.FallbackAttempts() == 3);

    allocations.AllowFallback();
    REQUIRE(system.Initialize());
    REQUIRE(system.DeviceMemory().buffer.BackingBasePointer() != nullptr);
    CHECK(allocations.FallbackAttempts() == 4);
}
