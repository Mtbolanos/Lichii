// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <utility>

#include "common/common_types.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Service {

/// Owns one guest-fence callback until it fires or its waiter leaves. Deregistration also
/// joins a callback that is already running.
class Nx2ScopedGuestFenceAction final {
public:
    template <typename Callback>
    Nx2ScopedGuestFenceAction(Tegra::Host1x::SyncpointManager& syncpoints_, u32 id_, u32 value,
                             Callback&& callback)
        : syncpoints{syncpoints_}, id{id_},
          handle{syncpoints.RegisterGuestAction(id, value, std::forward<Callback>(callback))} {}

    ~Nx2ScopedGuestFenceAction() {
        syncpoints.DeregisterGuestAction(id, handle);
    }

    Nx2ScopedGuestFenceAction(const Nx2ScopedGuestFenceAction&) = delete;
    Nx2ScopedGuestFenceAction& operator=(const Nx2ScopedGuestFenceAction&) = delete;

private:
    Tegra::Host1x::SyncpointManager& syncpoints;
    const u32 id;
    const Tegra::Host1x::SyncpointManager::ActionHandle handle;
};

} // namespace Service
