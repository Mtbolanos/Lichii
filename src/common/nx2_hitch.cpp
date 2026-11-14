// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/host_memory.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace Common::Nx2Hitch {

namespace {

constexpr long long ThresholdUs = 50'000;
constexpr size_t Causes = static_cast<size_t>(Cause::Count);
constexpr size_t MaxThreads = 48;

struct ThreadEntry {
    char name[32]{};
#ifdef _WIN32
    HANDLE handle{};
#endif
};

std::array<ThreadEntry, MaxThreads> g_threads;
std::atomic<size_t> g_thread_count{};
std::mutex g_thread_mutex;

struct Snapshot {
    std::array<u64, Causes> ns{};
    std::array<u64, Causes> count{};
    std::array<u64, Causes> amount{};
    std::array<u64, Causes> longest{};
    u64 page_faults{};
    u64 zero_filled{};
    u64 user_100ns{};   ///< the process's CPU time in user mode, all threads
    u64 kernel_100ns{}; ///< and in kernel mode (page faults, VM calls)
    size_t threads{};
    std::array<u64, MaxThreads> thread_user{};
    std::array<u64, MaxThreads> thread_kernel{};
};

#ifdef _WIN32
u64 FileTimeValue(const FILETIME& time) {
    return (u64{time.dwHighDateTime} << 32) | time.dwLowDateTime;
}
#endif

Snapshot Take() {
    Snapshot snapshot;
    for (size_t i = 0; i < Causes; ++i) {
        snapshot.ns[i] = g_counters.ns[i].load(std::memory_order_relaxed);
        snapshot.count[i] = g_counters.count[i].load(std::memory_order_relaxed);
        snapshot.amount[i] = g_counters.amount[i].load(std::memory_order_relaxed);
        snapshot.longest[i] = g_counters.longest[i].exchange(0, std::memory_order_relaxed);
    }
    snapshot.zero_filled = HostMemory::ZeroFilledBytes();
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        snapshot.page_faults = counters.PageFaultCount;
    }
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        snapshot.user_100ns = FileTimeValue(user);
        snapshot.kernel_100ns = FileTimeValue(kernel);
    }
    snapshot.threads = g_thread_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < snapshot.threads; ++i) {
        if (GetThreadTimes(g_threads[i].handle, &created, &exited, &kernel, &user)) {
            snapshot.thread_user[i] = FileTimeValue(user);
            snapshot.thread_kernel[i] = FileTimeValue(kernel);
        }
    }
#endif
    return snapshot;
}

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_HITCH_LOG");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// The busiest named threads in the gap, CPU time summed per name (worker pools share one).
std::string BusiestThreads(const Snapshot& last, const Snapshot& now) {
    struct Busy {
        std::string name;
        u64 user{};
        u64 kernel{};
    };
    std::vector<Busy> busy;
    for (size_t i = 0; i < now.threads; ++i) {
        const u64 user = now.thread_user[i] - (i < last.threads ? last.thread_user[i] : 0);
        const u64 kernel = now.thread_kernel[i] - (i < last.threads ? last.thread_kernel[i] : 0);
        if (user + kernel == 0) {
            continue;
        }
        const std::string name{g_threads[i].name};
        const auto it = std::ranges::find(busy, name, &Busy::name);
        if (it != busy.end()) {
            it->user += user;
            it->kernel += kernel;
        } else {
            busy.push_back({name, user, kernel});
        }
    }
    std::ranges::sort(busy, [](const Busy& a, const Busy& b) {
        return a.user + a.kernel > b.user + b.kernel;
    });
    std::string text;
    for (size_t i = 0; i < (std::min)(busy.size(), size_t{6}); ++i) {
        text += fmt::format("{}{} {:.0f}+{:.0f}", i == 0 ? "" : ", ", busy[i].name,
                            static_cast<double>(busy[i].user) / 1e4,
                            static_cast<double>(busy[i].kernel) / 1e4);
    }
    return text.empty() ? "none measured" : text;
}

} // Anonymous namespace

namespace {
std::mutex g_compile_mutex;
std::unordered_map<u64, u8> g_compiled_on; ///< key -> mask of the cores that compiled it
u8 g_cores_seen{};
std::array<std::vector<u64>, 8> g_warm_queues;
} // Anonymous namespace

bool JitWarmEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_WARM");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

Cause CompileCause(u64 key, u32 core) {
    const u8 bit = static_cast<u8>(1U << (core & 7));
    std::scoped_lock lock{g_compile_mutex};
    g_cores_seen |= bit;
    u8& mask = g_compiled_on[key];
    const bool repeat = (mask & ~bit) != 0;
    mask |= bit;
    if (t_warming) {
        return Cause::WarmCompile;
    }
    // Single-stepping blocks (bit 57 of an A64 LocationDescriptor) are never worth warming.
    if (JitWarmEnabled() && ((key >> 57) & 1) == 0) {
        for (u32 other = 0; other < 8; ++other) {
            const u8 other_bit = static_cast<u8>(1U << other);
            if ((g_cores_seen & other_bit) != 0 && (mask & other_bit) == 0 &&
                g_warm_queues[other].size() < 400'000) {
                g_warm_queues[other].push_back(key);
            }
        }
    }
    return repeat ? Cause::RecompileRepeat : Cause::Recompile;
}

size_t TakeWarmKeys(u32 core, std::vector<u64>& out, size_t max) {
    out.clear();
    std::scoped_lock lock{g_compile_mutex};
    auto& queue = g_warm_queues[core & 7];
    const u8 bit = static_cast<u8>(1U << (core & 7));
    // Oldest first; skip keys this core compiled on demand meanwhile.
    size_t taken = 0;
    while (taken < queue.size() && out.size() < max) {
        const u64 key = queue[taken++];
        if ((g_compiled_on[key] & bit) == 0) {
            out.push_back(key);
        }
    }
    queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(taken));
    return out.size();
}

void RegisterThread(const char* name) {
    const std::string_view view{name};
    t_role = view.starts_with("CPUCore") ? Role::GuestCore
             : view == "GPUBack"         ? Role::GpuBack
             : view == "GPU"             ? Role::GpuFront
                                         : Role::Other;
#ifdef _WIN32
    std::scoped_lock lock{g_thread_mutex};
    const size_t index = g_thread_count.load(std::memory_order_relaxed);
    if (index >= MaxThreads) {
        return;
    }
    HANDLE handle{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0)) {
        return;
    }
    ThreadEntry& entry = g_threads[index];
    std::strncpy(entry.name, name, sizeof(entry.name) - 1);
    entry.handle = handle;
    g_thread_count.store(index + 1, std::memory_order_release);
#else
    static_cast<void>(name);
#endif
}

void OnPresent(long long gap_us) {
    static Snapshot last = Take();
    static u64 window_start_ns = NowNs();
    static int lines_in_window = 0;
    if (!Enabled()) {
        return;
    }
    const Snapshot now = Take();
    if (gap_us >= ThresholdUs) {
        // At most 60 lines a minute (a loading screen can stall many presents in a row; 20 hid a
        // 740 ms first load behind the title screen's gaps, 2026-11-10).
        const u64 now_ns = NowNs();
        if (now_ns - window_start_ns >= 60'000'000'000ULL) {
            window_start_ns = now_ns;
            lines_in_window = 0;
        }
        if (lines_in_window++ < 60) {
            std::string causes;
            for (size_t i = 0; i < Causes; ++i) {
                const u64 ns = now.ns[i] - last.ns[i];
                const u64 count = now.count[i] - last.count[i];
                if (count == 0) {
                    continue;
                }
                const u64 amount = now.amount[i] - last.amount[i];
                causes += fmt::format("{}{} {:.1f} ms ({}{}, longest {:.1f})",
                                      causes.empty() ? "" : ", ", CauseNames[i],
                                      static_cast<double>(ns) / 1e6, count,
                                      amount != 0 ? fmt::format(", {} MiB", amount >> 20) : "",
                                      static_cast<double>(now.longest[i]) / 1e6);
            }
            const u64 trim_since = g_counters.zero_trim_since.load(std::memory_order_relaxed);
            if (trim_since != 0) {
                causes += fmt::format("{}a zero-page pass running for {:.0f} ms",
                                      causes.empty() ? "" : ", ",
                                      static_cast<double>(now_ns - trim_since) / 1e6);
            }
            LOG_INFO(Render, // with the frame lines it lines up against
                     "Hitch: {:.1f} ms between presents; in it: {}; page faults {}, kernel "
                     "zero-filled {} MiB, CPU {:.0f} ms user + {:.0f} ms kernel; busiest threads "
                     "(user+kernel ms): {}",
                     static_cast<double>(gap_us) / 1e3, causes.empty() ? "nothing counted" : causes,
                     now.page_faults - last.page_faults, (now.zero_filled - last.zero_filled) >> 20,
                     static_cast<double>(now.user_100ns - last.user_100ns) / 1e4,
                     static_cast<double>(now.kernel_100ns - last.kernel_100ns) / 1e4,
                     BusiestThreads(last, now));
        }
    }
    last = now;
}

} // namespace Common::Nx2Hitch
