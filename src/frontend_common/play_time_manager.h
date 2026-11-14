// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <mutex>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include <common/polyfill_thread.h>

// TODO(crueter): Extract this into frontend_common
namespace Service::Account {
class ProfileManager;
}

namespace PlayTime {

using ProgramId = u64;
using PlayTime = u64;
using PlayTimeDatabase = std::map<ProgramId, PlayTime>;

class PlayTimeManager {
public:
    explicit PlayTimeManager();
    ~PlayTimeManager();

    LICHII_NON_COPYABLE(PlayTimeManager);
    LICHII_NON_MOVEABLE(PlayTimeManager);

    u64 GetPlayTime(u64 program_id) const;
    void ResetProgramPlayTime(u64 program_id);
    void SetProgramId(u64 program_id);
    void SetPlayTime(u64 program_id, u64 play_time);
    void Start();
    void Stop();

    static std::string GetReadablePlayTime(u64 time_seconds);
    static std::string GetPlayTimeHours(u64 time_seconds);
    static std::string GetPlayTimeMinutes(u64 time_seconds);
    static std::string GetPlayTimeSeconds(u64 time_seconds);

private:
    void AutoTimestamp(std::stop_token stop_token);
    /// Writes a copy of the database. Called with save_mutex held from the change being saved
    /// through the write, so saves land in the order of their changes. database_mutex guards the
    /// database alone, so GetPlayTime never waits on file I/O.
    void Save();

    std::mutex save_mutex;
    mutable std::mutex database_mutex;
    PlayTimeDatabase database;
    u64 running_program_id{};
    std::jthread play_time_thread;
};


} // namespace PlayTime
