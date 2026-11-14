// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "lichii_cmd/emu_window/sdl_window_state.h"

TEST_CASE("SDL hidden startup and visibility transitions use the current window flags",
          "[sdl_window_state]") {
    SdlWindowState state;
    CHECK_FALSE(state.IsShown());
    REQUIRE(state.Update(SDL_WINDOW_HIDDEN, true, 1280, 720));
    CHECK_FALSE(state.IsShown());

    REQUIRE(SdlWindowState::NeedsUpdate(SDL_EVENT_WINDOW_SHOWN));
    REQUIRE(state.Update(0, true, 1280, 720));
    CHECK(state.IsShown());

    REQUIRE(SdlWindowState::NeedsUpdate(SDL_EVENT_WINDOW_MINIMIZED));
    REQUIRE(state.Update(SDL_WINDOW_MINIMIZED, true, 1280, 720));
    CHECK_FALSE(state.IsShown());

    // Restore must resume presentation without requiring a later EXPOSED event.
    REQUIRE(SdlWindowState::NeedsUpdate(SDL_EVENT_WINDOW_RESTORED));
    REQUIRE(state.Update(0, true, 1280, 720));
    CHECK(state.IsShown());

    REQUIRE(SdlWindowState::NeedsUpdate(SDL_EVENT_WINDOW_HIDDEN));
    REQUIRE(state.Update(SDL_WINDOW_HIDDEN, true, 1280, 720));
    CHECK_FALSE(state.IsShown());

    // A queued EXPOSED event must not override the window's current hidden state.
    REQUIRE(SdlWindowState::NeedsUpdate(SDL_EVENT_WINDOW_EXPOSED));
    REQUIRE(state.Update(SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED, true, 1280, 720));
    CHECK_FALSE(state.IsShown());
}

TEST_CASE("SDL rejects unavailable or nonpositive pixel extents and recovers",
          "[sdl_window_state]") {
    SdlWindowState state;
    const auto original = state.Update(0, true, 1920, 1080);
    REQUIRE(original);
    CHECK(state.IsShown());

    const std::array<std::pair<int, int>, 5> invalid{{{0, 0}, {0, 720}, {1280, 0},
                                                    {-1, 720}, {1280, -1}}};
    for (const auto& [width, height] : invalid) {
        CAPTURE(width, height);
        CHECK_FALSE(state.Update(0, true, width, height));
        CHECK_FALSE(state.IsShown());
    }
    CHECK_FALSE(state.Update(0, false, 1920, 1080));
    CHECK_FALSE(state.IsShown());

    const auto restored = state.Update(SDL_WINDOW_MAXIMIZED, true, 2560, 1440);
    REQUIRE(restored);
    CHECK(restored->first == 2560U);
    CHECK(restored->second == 1440U);
    CHECK(state.IsShown());
}

TEST_CASE("SDL retains valid pixel dimensions independently of focus and visibility",
          "[sdl_window_state]") {
    SdlWindowState state;
    const auto hidden = state.Update(SDL_WINDOW_HIDDEN, true, 3840, 2160);
    REQUIRE(hidden);
    CHECK(hidden->first == 3840U);
    CHECK(hidden->second == 2160U);
    CHECK_FALSE(state.IsShown());
    REQUIRE(state.Update(SDL_WINDOW_NOT_FOCUSABLE, true, 1280, 720));
    CHECK(state.IsShown());
    const auto maximum = state.Update(0, true, std::numeric_limits<int>::max(), 1);
    REQUIRE(maximum);
    CHECK(maximum->first == static_cast<u32>(std::numeric_limits<int>::max()));
    CHECK(maximum->second == 1U);
}

TEST_CASE("SDL refreshes presentation state only for window state events",
          "[sdl_window_state]") {
    for (const auto event : {SDL_EVENT_WINDOW_SHOWN, SDL_EVENT_WINDOW_HIDDEN,
                             SDL_EVENT_WINDOW_MINIMIZED, SDL_EVENT_WINDOW_MAXIMIZED,
                             SDL_EVENT_WINDOW_RESTORED, SDL_EVENT_WINDOW_EXPOSED,
                             SDL_EVENT_WINDOW_RESIZED, SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED}) {
        CAPTURE(event);
        CHECK(SdlWindowState::NeedsUpdate(event));
    }
    for (const auto event : {SDL_EVENT_KEY_DOWN, SDL_EVENT_MOUSE_MOTION,
                             SDL_EVENT_WINDOW_CLOSE_REQUESTED, SDL_EVENT_QUIT}) {
        CAPTURE(event);
        CHECK_FALSE(SdlWindowState::NeedsUpdate(event));
    }
}
