// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <mutex>
#include <vector>

#include "common/fs/file_lock.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace Common::FS {

namespace {

#ifdef _WIN32

std::intptr_t OpenLockFile(const std::filesystem::path& path) {
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    return file == INVALID_HANDLE_VALUE ? -1 : reinterpret_cast<std::intptr_t>(file);
}

/// Takes the lock if it is free; `busy` tells whether another holds it.
bool TryLock(std::intptr_t handle, bool& busy) {
    OVERLAPPED overlapped{};
    if (LockFileEx(reinterpret_cast<HANDLE>(handle),
                   LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0,
                   &overlapped) != FALSE) {
        busy = false;
        return true;
    }
    busy = GetLastError() == ERROR_LOCK_VIOLATION;
    return false;
}

void CloseLockFile(std::intptr_t handle, bool held) {
    const HANDLE file = reinterpret_cast<HANDLE>(handle);
    if (held) {
        OVERLAPPED overlapped{};
        UnlockFileEx(file, 0, 1, 0, &overlapped);
    }
    CloseHandle(file);
}

#else

std::intptr_t OpenLockFile(const std::filesystem::path& path) {
    return ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
}

/// Takes the lock if it is free; `busy` tells whether another holds it.
bool TryLock(std::intptr_t handle, bool& busy) {
    int result;
    do {
        result = ::flock(static_cast<int>(handle), LOCK_EX | LOCK_NB);
    } while (result != 0 && errno == EINTR);
    busy = result != 0 && errno == EWOULDBLOCK;
    return result == 0;
}

void CloseLockFile(std::intptr_t handle, bool) {
    ::close(static_cast<int>(handle)); // Releases the lock
}

#endif

/// Locks that stayed taken through a whole wait, until they are taken here again
std::mutex stuck_mutex;
std::vector<std::filesystem::path> stuck_locks;

} // Anonymous namespace

FileLock::FileLock(const std::filesystem::path& path, std::chrono::milliseconds wait,
                   std::stop_token stop)
    : handle{OpenLockFile(path)} {
    if (handle == -1) {
        return;
    }
    const bool stuck = [&] {
        std::scoped_lock lock{stuck_mutex};
        return std::ranges::find(stuck_locks, path) != stuck_locks.end();
    }();
    const auto deadline = std::chrono::steady_clock::now() + wait;
    held = TryLock(handle, busy);
    while (!held && busy && !stuck && wait.count() > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::scoped_lock lock{stuck_mutex};
            if (std::ranges::find(stuck_locks, path) == stuck_locks.end()) {
                LOG_WARNING(Common_Filesystem,
                            "Another process has held {} for {} ms: going on without the lock "
                            "until it is free",
                            PathToUTF8String(path), wait.count());
                stuck_locks.push_back(path);
            }
            return;
        }
        if (!Common::StoppableTimedWait(stop, std::chrono::milliseconds{10})) {
            return;
        }
        held = TryLock(handle, busy);
    }
    if (held && stuck) {
        std::scoped_lock lock{stuck_mutex};
        std::erase(stuck_locks, path);
    }
}

FileLock::~FileLock() {
    if (handle != -1) {
        CloseLockFile(handle, held);
    }
}

} // namespace Common::FS
