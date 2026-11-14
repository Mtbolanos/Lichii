// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>

#include "common/polyfill_thread.h"

namespace Common::FS {

/// An exclusive advisory lock on a file (flock on POSIX, LockFileEx on Windows), held until the
/// object is destroyed. The system releases it when the process ends, so a crash never leaves it
/// held. It only keeps out others that take it too; the file it locks is created if missing and
/// holds no data.
class FileLock {
public:
    /// Takes the lock, waiting up to `wait` while another process holds it, unless `stop` is
    /// requested first. A crashed process holds it for as long as it is kept alive (to write a
    /// crash dump, for one): when the lock stays taken through a whole wait, that is logged, and
    /// until it is free again, a wait for it only tries once.
    explicit FileLock(const std::filesystem::path& path, std::chrono::milliseconds wait = {},
                      std::stop_token stop = {});
    ~FileLock();

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

    /// False when another process holds it, or the file can't be created or locked.
    [[nodiscard]] bool Held() const noexcept {
        return held;
    }

    /// True when it isn't held because another process held it at the last try.
    [[nodiscard]] bool Busy() const noexcept {
        return busy;
    }

private:
    std::intptr_t handle{-1}; ///< File descriptor, or a HANDLE on Windows
    bool held{};
    bool busy{};
};

} // namespace Common::FS
