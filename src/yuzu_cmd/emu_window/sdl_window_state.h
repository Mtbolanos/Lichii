// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <optional>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "common/common_types.h"

class SdlWindowState {
public:
    static constexpr bool NeedsUpdate(Uint32 event_type) {
        switch (event_type) {
        case SDL_EVENT_WINDOW_SHOWN:
        case SDL_EVENT_WINDOW_HIDDEN:
        case SDL_EVENT_WINDOW_MINIMIZED:
        case SDL_EVENT_WINDOW_MAXIMIZED:
        case SDL_EVENT_WINDOW_RESTORED:
        case SDL_EVENT_WINDOW_EXPOSED:
        case SDL_EVENT_WINDOW_RESIZED:
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            return true;
        default:
            return false;
        }
    }

    // Called on the SDL thread. No extent means the caller keeps its last valid layout.
    std::optional<std::pair<u32, u32>> Update(SDL_WindowFlags flags, bool size_available,
                                            int width, int height) {
        const bool valid_extent = size_available && width > 0 && height > 0;
        shown.store(valid_extent && (flags & (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED)) == 0,
                    std::memory_order_relaxed);
        if (!valid_extent) {
            return std::nullopt;
        }
        return std::pair{static_cast<u32>(width), static_cast<u32>(height)};
    }

    bool IsShown() const {
        return shown.load(std::memory_order_relaxed);
    }

private:
    std::atomic_bool shown{};
};
