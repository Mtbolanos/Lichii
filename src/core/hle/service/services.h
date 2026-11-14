// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/polyfill_thread.h"
#include "core/hle/service/sm/sm.h"

namespace Service {

/// @brief The purpose of this class is to own any objects that need to be shared across the other service
/// implementations. Will be torn down when the global system instance is shutdown.
class Services final {
public:
    explicit Services(std::shared_ptr<SM::ServiceManager>& sm, Core::System& system,
                      std::stop_token token);
    /// Joins the audio and vi threads. At shutdown this runs after ServiceManager::Close and
    /// CloseServices, so they leave their wait or server at once, and before the service manager
    /// they use is destroyed.
    ~Services() = default;

private:
    std::jthread audio_worker;
    std::jthread vi_worker;
};

} // namespace Service
