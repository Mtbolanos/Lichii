// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/adpf.h"
#include "common/core_pinning.h"
#include "common/error.h"
#include "common/event_wait.h"
#include "common/logging.h"
#include "common/assert.h"
#include "common/thread.h"
#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__HAIKU__)
#include <kernel/OS.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#include "common/windows/timer_resolution.h"
#else
#if defined(__FreeBSD__)
#include <sys/cpuset.h>
#include <sys/_cpuset.h>
#include <pthread_np.h>
// Compatibility with CPUset
#define cpu_set_t cpuset_t
#elif defined(__DragonFly__) || defined(__OpenBSD__) || defined(__Bitrig__)
#include <pthread_np.h>
#endif
#include <pthread.h>
#include <sched.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __linux__
#include <sys/resource.h>
#include <algorithm>

namespace {
constexpr int NICE_AUDIO = -16;
constexpr int NICE_URGENT_DISPLAY = -8;
constexpr int NICE_DISPLAY = -4;
constexpr int NICE_DEFAULT = 0;
constexpr int NICE_BACKGROUND = 10;

int LowestAllowedNice() {
    static const int lowest = [] {
        rlimit limit{};
        if (getrlimit(RLIMIT_NICE, &limit) != 0) {
            return 0;
        }
        if (limit.rlim_cur >= 40) {
            return -20;
        }
        return 20 - static_cast<int>(limit.rlim_cur);
    }();
    return lowest;
}

int NiceValueForPriority(Common::ThreadPriority priority) {
    const int wanted = [priority] {
        switch (priority) {
        case Common::ThreadPriority::Low: return NICE_BACKGROUND;
        case Common::ThreadPriority::Normal: return NICE_DEFAULT;
        case Common::ThreadPriority::High: return NICE_DISPLAY;
        case Common::ThreadPriority::VeryHigh: return NICE_URGENT_DISPLAY;
        case Common::ThreadPriority::Critical: return NICE_AUDIO;
        default: return NICE_DEFAULT;
        }
    }();
    return (std::max)(wanted, (std::min)(NICE_DEFAULT, LowestAllowedNice()));
}
} // Anonymous namespace
#endif

#ifdef __ANDROID__
#include <sys/utsname.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

namespace {
constexpr size_t ANDROID_MINIMUM_PERFORMANCE_CORES = 4;

// A core counts as a performance core while it is within this much of the fastest one.
constexpr s64 ANDROID_PERFORMANCE_CAPACITY_PERCENT = 50;

constexpr std::chrono::nanoseconds ANDROID_POLICY_POLL_INTERVAL = std::chrono::milliseconds{500};

enum class CoreGroup {
    Unrestricted,
    Performance,
    Efficiency,
};

struct ThreadPolicy {
    pid_t tid;
    CoreGroup group;
    s32 nice_value;
    bool has_nice;
};

struct CoreInfo {
    s64 weight;
    u64 midr;
    s32 cpu;
};

struct CpuTopologyState {
    std::mutex topology_mutex;
    cpu_set_t allowed{};
    cpu_set_t performance{};
    cpu_set_t efficiency{};
    bool separated = false;
    bool initialized = false;

    pid_t canary_tid = 0;
    cpu_set_t canary_mask{};
    bool canary_valid = false;

    std::atomic<s64> next_poll_ns{0};

    std::mutex policy_mutex;
    std::vector<ThreadPolicy> policies;
};

CpuTopologyState& State() {
    static CpuTopologyState* const state = new CpuTopologyState();
    return *state;
}

struct PolicyRegistration {
    ~PolicyRegistration() {
        const pid_t tid = gettid();
        ::Common::ADPF::RemoveCurrentThread();
        CpuTopologyState& state = State();
        std::scoped_lock lock{state.policy_mutex};
        std::erase_if(state.policies,
                      [tid](const ThreadPolicy& policy) { return policy.tid == tid; });
    }
};

thread_local PolicyRegistration t_policy_registration;

s32 PossibleCpuCount() {
    std::ifstream file("/sys/devices/system/cpu/possible");
    std::string list;
    if (file && std::getline(file, list) && !list.empty()) {
        s64 highest = -1;
        const char* cursor = list.c_str();
        while (*cursor != '\0') {
            char* end = nullptr;
            const s64 value = std::strtol(cursor, &end, 10);
            if (end == cursor) {
                break;
            }
            highest = (std::max)(highest, value);
            cursor = end;
            while (*cursor == '-' || *cursor == ',') {
                ++cursor;
            }
        }
        if (highest >= 0) {
            return static_cast<s32>((std::min<s64>)(highest + 1, CPU_SETSIZE));
        }
    }
    const s64 configured = sysconf(_SC_NPROCESSORS_CONF);
    if (configured > 0) {
        return static_cast<s32>((std::min<s64>)(configured, CPU_SETSIZE));
    }
    return static_cast<s32>((std::min<s64>)(std::thread::hardware_concurrency(), CPU_SETSIZE));
}

s64 ReadCpuScalar(s32 cpu, const char* node) {
    s64 value = 0;
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + node);
    if (!file || !(file >> value) || value <= 0) {
        return 0;
    }
    return value;
}

u64 ReadCpuMidr(s32 cpu) {
    u64 midr = 0;
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                       "/regs/identification/midr_el1");
    if (!file || !(file >> std::hex >> midr)) {
        return 0;
    }
    return midr;
}

std::vector<CoreInfo> CollectCores(const cpu_set_t& allowed, s32 total, const char* node,
                                   bool require_all) {
    std::vector<CoreInfo> cores;
    for (s32 cpu = 0; cpu < total; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const s64 weight = ReadCpuScalar(cpu, node);
        if (weight <= 0) {
            if (require_all) {
                return {};
            }
            LOG_WARNING(Common, "Could not read {} for CPU {}, treating it as an efficiency core",
                        node, cpu);
            continue;
        }
        cores.push_back(CoreInfo{weight, ReadCpuMidr(cpu), cpu});
    }
    return cores;
}

bool WeightsAreUniform(const std::vector<CoreInfo>& cores) {
    return std::all_of(cores.begin(), cores.end(),
                       [&](const CoreInfo& core) { return core.weight == cores.front().weight; });
}

bool MidrsAreDistinct(const std::vector<CoreInfo>& cores) {
    return std::none_of(cores.begin(), cores.end(),
                        [](const CoreInfo& core) { return core.midr == 0; }) &&
           std::any_of(cores.begin(), cores.end(),
                       [&](const CoreInfo& core) { return core.midr != cores.front().midr; });
}

void ComputeTopologyLocked(CpuTopologyState& state) {
    state.initialized = true;
    state.separated = false;
    CPU_ZERO(&state.allowed);
    CPU_ZERO(&state.performance);
    CPU_ZERO(&state.efficiency);

    if (sched_getaffinity(getpid(), sizeof(state.allowed), &state.allowed) != 0) {
        LOG_WARNING(Common, "Could not query process CPU affinity: {}",
                    ::Common::GetLastErrorMsg());
        return;
    }

    const s32 total = PossibleCpuCount();
    auto cores = CollectCores(state.allowed, total, "cpu_capacity", true);
    if (cores.empty() || (WeightsAreUniform(cores) && MidrsAreDistinct(cores))) {
        auto by_frequency = CollectCores(state.allowed, total, "cpufreq/cpuinfo_max_freq", false);
        if (!by_frequency.empty()) {
            cores = std::move(by_frequency);
        }
    }
    if (cores.empty()) {
        LOG_WARNING(Common, "Could not determine CPU topology, thread placement is disabled");
        return;
    }

    if (WeightsAreUniform(cores)) {
        if (MidrsAreDistinct(cores)) {
            LOG_WARNING(Common, "CPU clusters differ but rank identically, thread placement is "
                                "disabled");
        } else {
            LOG_INFO(Common, "CPU cores are symmetric, thread placement is disabled");
        }
        return;
    }

    std::sort(cores.begin(), cores.end(), [](const CoreInfo& lhs, const CoreInfo& rhs) {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        }
        return lhs.cpu < rhs.cpu;
    });

    const s64 fastest = cores.front().weight;
    size_t taken = 0;
    for (const auto& core : cores) {
        const bool fast_enough =
            core.weight * 100 >= fastest * ANDROID_PERFORMANCE_CAPACITY_PERCENT;
        if (!fast_enough && taken >= ANDROID_MINIMUM_PERFORMANCE_CORES) {
            break;
        }
        CPU_SET(core.cpu, &state.performance);
        ++taken;
    }
    if (taken == 0) {
        return;
    }

    for (s32 cpu = 0; cpu < total; ++cpu) {
        if (CPU_ISSET(cpu, &state.allowed) && !CPU_ISSET(cpu, &state.performance)) {
            CPU_SET(cpu, &state.efficiency);
        }
    }

    state.separated = CPU_COUNT(&state.efficiency) > 0;
    LOG_INFO(Common, "CPU topology: {} performance cores, {} efficiency cores, separation {}",
             CPU_COUNT(&state.performance), CPU_COUNT(&state.efficiency),
             state.separated ? "enabled" : "unavailable");
}

void EnsureTopologyLocked(CpuTopologyState& state) {
    if (!state.initialized) {
        ComputeTopologyLocked(state);
    }
}

void RefreshTopologyLocked(CpuTopologyState& state) {
    if (!state.initialized) {
        ComputeTopologyLocked(state);
        return;
    }
    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(getpid(), sizeof(current), &current) != 0) {
        return;
    }
    if (std::memcmp(&current, &state.allowed, sizeof(current)) != 0) {
        ComputeTopologyLocked(state);
    }
}

bool ApplyCoreGroupLocked(CpuTopologyState& state, pid_t tid, CoreGroup group,
                          bool* gone = nullptr) {
    const bool restrict_group = group != CoreGroup::Unrestricted && state.separated;
    const cpu_set_t* mask = &state.allowed;
    if (restrict_group) {
        mask = group == CoreGroup::Performance ? &state.performance : &state.efficiency;
    }
    if (CPU_COUNT(mask) == 0) {
        return false;
    }
    if (sched_setaffinity(tid, sizeof(*mask), mask) != 0) {
        if (gone != nullptr && errno == ESRCH) {
            *gone = true;
            return false;
        }
        LOG_WARNING(Common, "Could not restrict thread {} to its core group: {}", tid,
                    ::Common::GetLastErrorMsg());
        return false;
    }
    return true;
}

bool KernelPreservesRequestedAffinity() {
    utsname info{};
    if (uname(&info) != 0) {
        return false;
    }
    s32 major = 0;
    s32 minor = 0;
    if (std::sscanf(info.release, "%d.%d", &major, &minor) != 2) {
        return false;
    }
    return major > 6 || (major == 6 && minor >= 2);
}

void SnapshotCanaryLocked(CpuTopologyState& state) {
    state.canary_valid = false;
    if (!state.separated) {
        return;
    }
    for (const auto& policy : state.policies) {
        if (policy.group == CoreGroup::Unrestricted) {
            continue;
        }
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(policy.tid, sizeof(mask), &mask) != 0) {
            continue;
        }
        state.canary_tid = policy.tid;
        state.canary_mask = mask;
        state.canary_valid = true;
        return;
    }
}

bool DueForPoll(CpuTopologyState& state) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    s64 next = state.next_poll_ns.load(std::memory_order_relaxed);
    if (now_ns < next) {
        return false;
    }
    return state.next_poll_ns.compare_exchange_strong(
        next, now_ns + ANDROID_POLICY_POLL_INTERVAL.count(), std::memory_order_relaxed);
}

ThreadPolicy& AcquirePolicyLocked(CpuTopologyState& state, pid_t tid) {
    for (auto& policy : state.policies) {
        if (policy.tid == tid) {
            return policy;
        }
    }
    return state.policies.emplace_back(ThreadPolicy{tid, CoreGroup::Unrestricted, 0, false});
}

void SetCurrentThreadCoreGroup(CoreGroup group) {
    const pid_t tid = gettid();
    (void)&t_policy_registration;

    CoreGroup effective = group;
    if (tid == getpid() && group != CoreGroup::Unrestricted) {
        LOG_WARNING(Common, "Refusing to place the main thread: the CPU topology is read from it");
        effective = CoreGroup::Unrestricted;
    }

    CpuTopologyState& state = State();
    std::scoped_lock topology_lock{state.topology_mutex};
    EnsureTopologyLocked(state);
    ApplyCoreGroupLocked(state, tid, effective);

    std::scoped_lock policy_lock{state.policy_mutex};
    AcquirePolicyLocked(state, tid).group = effective;
    if (!state.canary_valid) {
        SnapshotCanaryLocked(state);
    }
}

void RememberCurrentThreadNice(pid_t tid, s32 nice_value) {
    (void)&t_policy_registration;
    CpuTopologyState& state = State();
    std::scoped_lock lock{state.policy_mutex};
    ThreadPolicy& policy = AcquirePolicyLocked(state, tid);
    policy.nice_value = nice_value;
    policy.has_nice = true;
}
} // Anonymous namespace
#endif

#include "common/cpu_features.h"
#ifdef ARCHITECTURE_x86_64
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "common/x64/rdtsc.h"
#endif

#ifdef _WIN32
namespace {
// Hybrid CPUs (Intel P/E cores): Windows alone may run the emulator's time-critical threads on an
// efficiency core, where a frame takes much longer. Keep them on the fastest cores with CPU sets
// (a soft affinity the power manager respects) and background workers on the slower ones.
// EDEN_HYBRID_CORES=0 turns this off.
struct WindowsCpuSets {
    std::vector<Common::LogicalProcessor> processors;
    std::vector<ULONG> performance;
    std::vector<ULONG> efficiency;
    bool hybrid = false;
};

const WindowsCpuSets& CpuSets() {
    static const WindowsCpuSets sets = [] {
        WindowsCpuSets result;
        if (const char* text = std::getenv("EDEN_HYBRID_CORES");
            text != nullptr && std::strtol(text, nullptr, 10) == 0) {
            LOG_INFO(Common, "Hybrid core placement off (EDEN_HYBRID_CORES=0)");
            return result;
        }
        using PFN_GetSystemCpuSetInformation = BOOL(WINAPI*)(PSYSTEM_CPU_SET_INFORMATION, ULONG,
                                                             PULONG, HANDLE, ULONG);
        const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        const auto get = reinterpret_cast<PFN_GetSystemCpuSetInformation>(
            reinterpret_cast<void*>(GetProcAddress(kernel32, "GetSystemCpuSetInformation")));
        if (get == nullptr) {
            return result;
        }
        ULONG length = 0;
        get(nullptr, 0, &length, GetCurrentProcess(), 0);
        if (length == 0) {
            return result;
        }
        std::vector<u8> buffer(length);
        if (!get(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), length, &length,
                 GetCurrentProcess(), 0)) {
            return result;
        }
        u8 best = 0;
        for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
            const auto* info =
                reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
            if (info->Size == 0) {
                break;
            }
            if (info->Type == CpuSetInformation) {
                const u32 core =
                    (static_cast<u32>(info->CpuSet.Group) << 8) | info->CpuSet.CoreIndex;
                result.processors.push_back({static_cast<u32>(info->CpuSet.Id), core,
                                             static_cast<u8>(info->CpuSet.EfficiencyClass)});
                best = (std::max)(best, static_cast<u8>(info->CpuSet.EfficiencyClass));
            }
            offset += info->Size;
        }
        for (const Common::LogicalProcessor& processor : result.processors) {
            (processor.efficiency_class == best ? result.performance : result.efficiency)
                .push_back(processor.id);
        }
        // About ten time-critical threads (six guest cores, GPU front and back, Vulkan worker,
        // timing, fencing) crowd fewer performance processors: an i5-1235U (2 P cores, 4 logical)
        // ran Ocarina of Time at 9.3 FPS with them kept there, 11.5 FPS with the placement off.
        constexpr size_t MinPerformanceProcessors = 8;
        const bool few_performance = result.performance.size() < MinPerformanceProcessors;
        result.hybrid = !result.efficiency.empty() && !few_performance;
        if (!result.efficiency.empty() && few_performance) {
            LOG_INFO(Common,
                     "Hybrid CPU with {} performance logical processors (fewer than {}): "
                     "threads are left to Windows",
                     result.performance.size(), MinPerformanceProcessors);
        }
        if (result.hybrid) {
            LOG_INFO(Common,
                     "Hybrid CPU: {} performance and {} efficiency logical processors, "
                     "time-critical threads stay on the performance cores",
                     result.performance.size(), result.efficiency.size());
        }
        return result;
    }();
    return sets;
}

void SelectCurrentThreadCpuSets(const std::vector<ULONG>& ids) {
    using PFN_SetThreadSelectedCpuSets = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
    static const auto set = reinterpret_cast<PFN_SetThreadSelectedCpuSets>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadSelectedCpuSets")));
    if (set == nullptr) {
        return;
    }
    if (!set(GetCurrentThread(), ids.empty() ? nullptr : ids.data(),
             static_cast<ULONG>(ids.size()))) {
        LOG_WARNING(Common, "SetThreadSelectedCpuSets failed: {}", Common::GetLastErrorMsg());
    }
}
} // Anonymous namespace
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#include <fstream>
#include <optional>

namespace {
// The CPUs of a sysfs CPU list file; empty when the file is missing.
cpu_set_t ReadCpuList(const char* path) {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::ifstream file{path};
    std::string list;
    if (!(file >> list)) {
        return set;
    }
    for (const u32 cpu : Common::ParseCpuList(list)) {
        if (cpu < CPU_SETSIZE) {
            CPU_SET(static_cast<int>(cpu), &set);
        }
    }
    return set;
}

// The logical processors this process may run on, with their physical core (package and core id)
// and, on hybrid CPUs, their relative capacity as the efficiency class.
std::vector<Common::LogicalProcessor> LinuxProcessors() {
    std::vector<Common::LogicalProcessor> processors;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return processors;
    }
    // Intel hybrid CPUs list their efficiency cores here (the cpu_atom PMU), also on kernels
    // that don't report cpu_capacity on x86. Without it, all cores rank equal and a CPU with
    // fewer than MinCoresForGpuThreadPinning performance cores would pass the check.
    const cpu_set_t efficiency = ReadCpuList("/sys/devices/cpu_atom/cpus");
    const auto read = [](int cpu, const char* node) -> std::optional<u32> {
        std::ifstream file{"/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + node};
        u32 value{};
        if (file >> value) {
            return value;
        }
        return std::nullopt;
    };
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const auto core = read(cpu, "topology/core_id");
        const auto package = read(cpu, "topology/physical_package_id");
        if (!core || !package) {
            return {};
        }
        // cpu_capacity is 1024 for the fastest cores; absent on CPUs that are not hybrid, and
        // often on x86 ones that are.
        const u32 capacity =
            read(cpu, "cpu_capacity").value_or(CPU_ISSET(cpu, &efficiency) ? 512U : 1024U);
        processors.push_back({static_cast<u32>(cpu), (*package << 16) | *core,
                              static_cast<u8>((std::min)(capacity, 1024U) >> 3)});
    }
    return processors;
}

void SetCurrentThreadAffinity(const std::vector<u32>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const u32 cpu : cpus) {
        CPU_SET(cpu, &set);
    }
    if (const int error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set); error != 0) {
        LOG_WARNING(Common, "pthread_setaffinity_np failed: {}", std::strerror(error));
    }
}
} // Anonymous namespace
#endif

namespace Common {

void SetCurrentThreadPriority(ThreadPriority new_priority) {
#ifdef _WIN32
    int windows_priority = [&]() {
        switch (new_priority) {
        case ThreadPriority::Low: return THREAD_PRIORITY_BELOW_NORMAL;
        case ThreadPriority::Normal: return THREAD_PRIORITY_NORMAL;
        case ThreadPriority::High: return THREAD_PRIORITY_ABOVE_NORMAL;
        case ThreadPriority::VeryHigh: return THREAD_PRIORITY_HIGHEST;
        case ThreadPriority::Critical: return THREAD_PRIORITY_TIME_CRITICAL;
        default: return THREAD_PRIORITY_NORMAL;
        }
    }();
    SetThreadPriority(GetCurrentThread(), windows_priority);
#elif defined(__HAIKU__)
    // TODO: We have priorities for 3D rendering applications - may help lavapipe?
    int priority = [&]() {
        switch (new_priority) {
        case ThreadPriority::Low: return B_LOW_PRIORITY;
        case ThreadPriority::Normal: return B_NORMAL_PRIORITY;
        case ThreadPriority::High: return B_DISPLAY_PRIORITY;
        case ThreadPriority::VeryHigh: return B_URGENT_DISPLAY_PRIORITY;
        case ThreadPriority::Critical: return B_URGENT_PRIORITY;
        default: return B_NORMAL_PRIORITY;
        }
    }();
    set_thread_priority(find_thread(NULL), priority);
#elif defined(__ANDROID__)
    const int nice_value = NiceValueForPriority(new_priority);
    const pid_t tid = gettid();
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice_value) != 0) {
        LOG_WARNING(Common, "Could not set thread nice value to {}: {}", nice_value,
                    GetLastErrorMsg());
        return;
    }
    RememberCurrentThreadNice(tid, nice_value);
#elif defined(__linux__)
    const int nice_value = NiceValueForPriority(new_priority);
    if (setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
        LOG_DEBUG(Common, "Could not set thread nice value to {}: {}", nice_value,
                  GetLastErrorMsg());
    }
#else
    const s32 max_prio = sched_get_priority_max(SCHED_OTHER);
    const s32 min_prio = sched_get_priority_min(SCHED_OTHER);
    if (max_prio > min_prio) {
        const u32 level = (std::min)(static_cast<u32>(new_priority), 4U);
        sched_param params{};
        params.sched_priority =
            min_prio + static_cast<s32>(static_cast<u32>(max_prio - min_prio) * level) / 4;
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &params);
    }
#endif
}

void SetCurrentThreadName(const char* name) {
#ifdef _WIN32
    // Sets the debugger-visible name of the current thread (Windows 10+). Looked up at run time,
    // which also covers MinGW, whose headers may not declare it.
    using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const auto set_description = reinterpret_cast<SetThreadDescriptionFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"KernelBase.dll"), "SetThreadDescription")));
    if (set_description) {
        set_description(GetCurrentThread(), UTF8ToUTF16W(name).c_str());
    }
#elif  defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__HAIKU__)
    rename_thread(find_thread(NULL), name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__sun__) || defined(__glibc__) || defined(__managarm__)
    int ret = pthread_setname_np(pthread_self(), name);
    if (ret == ERANGE) {
        // Linux limits thread names to 15 characters and will outright reject any
        // attempt to set a longer name with ERANGE.
        char buf[16];
        size_t const len = std::min<size_t>(std::strlen(name), sizeof(buf) - 1);
        std::memcpy(buf, name, len);
        buf[len] = '\0';
        pthread_setname_np(pthread_self(), buf);
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetCurrentThreadToPerformanceCores() {
#if defined(__ANDROID__)
    if (ADPF::AddCurrentThread(ADPF::Session::Render)) {
        SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
        return;
    }
    SetCurrentThreadCoreGroup(CoreGroup::Performance);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets(sets.performance);
    }
#endif
}

void PinCurrentThread(PinnedRole role) {
#ifdef _WIN32
    struct PinnedCpuSets {
        std::vector<ULONG> front;
        std::vector<ULONG> back;
        std::vector<ULONG> guest;
    };
    static const PinnedCpuSets pinned = [] {
        const GpuThreadPinning plan = PlanGpuThreadPinning(CpuSets().processors);
        if (plan.front.empty()) {
            LOG_WARNING(Common,
                        "GPU thread pinning skipped: {} physical performance cores, needs {}",
                        plan.performance_cores, MinCoresForGpuThreadPinning);
            return PinnedCpuSets{};
        }
        LOG_INFO(Common,
                 "GPU thread pinning: front and back on a physical core each, guest cores on the "
                 "other {} performance cores ({} logical processors)",
                 plan.performance_cores - 2, plan.guest.size());
        return PinnedCpuSets{{plan.front.begin(), plan.front.end()},
                             {plan.back.begin(), plan.back.end()},
                             {plan.guest.begin(), plan.guest.end()}};
    }();
    const std::vector<ULONG>& ids = role == PinnedRole::GpuFront  ? pinned.front
                                    : role == PinnedRole::GpuBack ? pinned.back
                                                                  : pinned.guest;
    if (!ids.empty()) {
        SelectCurrentThreadCpuSets(ids);
    }
#elif defined(__linux__) && !defined(__ANDROID__)
    static const GpuThreadPinning plan = [] {
        GpuThreadPinning result = PlanGpuThreadPinning(LinuxProcessors());
        if (result.front.empty()) {
            LOG_WARNING(Common,
                        "GPU thread pinning skipped: {} physical performance cores, needs {}",
                        result.performance_cores, MinCoresForGpuThreadPinning);
        } else {
            LOG_INFO(Common,
                     "GPU thread pinning: front and back on a physical core each, guest cores on "
                     "the other {} performance cores ({} logical processors)",
                     result.performance_cores - 2, result.guest.size());
        }
        return result;
    }();
    const std::vector<u32>& cpus = role == PinnedRole::GpuFront  ? plan.front
                                   : role == PinnedRole::GpuBack ? plan.back
                                                                 : plan.guest;
    if (!cpus.empty()) {
        SetCurrentThreadAffinity(cpus);
    }
#else
    (void)role;
    [[maybe_unused]] static const bool logged = [] {
        LOG_INFO(Common, "GPU thread pinning is only supported on Windows and Linux");
        return true;
    }();
#endif
}

void SetCurrentThreadToEfficiencyCores() {
#if defined(__ANDROID__)
    if (ADPF::AddCurrentThread(ADPF::Session::Background)) {
        SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
        return;
    }
    SetCurrentThreadCoreGroup(CoreGroup::Efficiency);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets(sets.efficiency);
    }
#endif
}

void SetCurrentThreadToBackgroundWork() {
#if defined(__ANDROID__)
    ADPF::AddCurrentThread(ADPF::Session::Background);
    SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets({});
    }
#endif
}

void SetCurrentThreadToAllCores() {
#if defined(__ANDROID__)
    ADPF::RemoveCurrentThread();
    SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets({});
    }
#endif
}

void RefreshThreadPolicies() {
#if defined(__ANDROID__)
    CpuTopologyState& state = State();
    std::scoped_lock topology_lock{state.topology_mutex};
    RefreshTopologyLocked(state);

    std::scoped_lock policy_lock{state.policy_mutex};
    std::erase_if(state.policies, [&state](const ThreadPolicy& policy) {
        bool gone = false;
        if (policy.has_nice &&
            setpriority(PRIO_PROCESS, static_cast<id_t>(policy.tid), policy.nice_value) != 0 &&
            errno == ESRCH) {
            gone = true;
        }
        if (!gone) {
            ApplyCoreGroupLocked(state, policy.tid, policy.group, &gone);
        }
        return gone;
    });
    SnapshotCanaryLocked(state);
#endif
}

void PollThreadPolicies() {
#if defined(__ANDROID__)
    static const bool needed = !KernelPreservesRequestedAffinity();
    if (!needed) {
        return;
    }

    CpuTopologyState& state = State();
    if (!DueForPoll(state)) {
        return;
    }

    pid_t tid;
    cpu_set_t expected;
    {
        std::scoped_lock lock{state.topology_mutex};
        if (!state.canary_valid) {
            return;
        }
        tid = state.canary_tid;
        expected = state.canary_mask;
    }

    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(tid, sizeof(current), &current) == 0 &&
        std::memcmp(&current, &expected, sizeof(current)) == 0) {
        return;
    }
    RefreshThreadPolicies();
#endif
}

#ifdef ARCHITECTURE_x86_64
// On Linux and UNIX systems, a futex would nominally be used to cover the costs
// the idea is that it's intuitivelly cheaper to use a direct instruction as opposed to a full futex call
// the underlying libc++ implementation uses pthread_cond_timedwait which MAY invoke a futex
// Let's pretend the OS is too expensive to jump into, and avoid ANY context switches
// this should *IN THEORY* lower CPU usage while just waiting for stuff effectively
// For windows the minimal quanta resolution is about 500us, and normal CRT cond var is 1.5ms(?)
// so may as well avoid that too
// Let's just give ALL platforms the same mechanisms (almost) for when they have umonitor OR waitpkg
#ifdef __clang__
__attribute__((target("waitpkg,mwaitx")))
#elif defined(__GNUC__)
#pragma GCC target("waitpkg")
#pragma GCC target("mwaitx")
#endif
bool Event::WaitFor(const std::chrono::nanoseconds time) {
#ifdef _WIN32
    auto const& caps = Common::g_cpu_caps;
    const auto deadline = EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()), time);
    const auto remaining = [deadline] {
        return deadline - std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch());
    };
    const u64 ticks_per_second = Common::g_wall_clock.NsToTicks(std::chrono::seconds{1});
    if (caps.monitorx) {
        while (true) {
            // Armed monitor, as per manual, MWAITX must be conditional if the condition isn't satisfied
            // to prevent a race condition.
            _mm_monitorx(reinterpret_cast<u64*>(std::addressof(is_set)), 0, 0);
            if (!is_set.load()) {
                const u32 cycles = MwaitxTimeoutCycles(remaining(), ticks_per_second);
                if (cycles != 0) {
                    // See here: https://github.com/torvalds/linux/blob/948a64995aca6820abefd17f1a4258f5835c5ad9/arch/x86/lib/delay.c#L93
                    // MWAITX accepts a 32-bit input timer which determines the total number of cycles to wait for
                    // NOT THE TOTAL ABSOLUTE TSC VALUE, it's just a delta
                    // BIT[1] = use a timer
                    // Hint = 0: Use C1 state when sleepy (means slower wakeup but better savings)
                    _mm_mwaitx(1 << 1, 0u, cycles);
                    continue;
                } else
                    return false; //timeout
            }
            bool expected = true;
            if (is_set.compare_exchange_weak(expected, false, std::memory_order_release))
                return true;
        }
    }
    if (caps.waitpkg) {
        // #UD If CPUID.7.0:ECX.WAITPKG[bit 5]=0.
        while (true) {
            _umonitor(std::addressof(is_set));
            if (!is_set.load()) {
                // UMWAIT also wakes early on interrupts, and its result (CF) only reports the OS
                // limit (IA32_UMWAIT_CONTROL). Wait again until the deadline, as for MWAITX. The
                // bounded budget keeps the absolute TSC deadline from wrapping.
                const u32 cycles = MwaitxTimeoutCycles(remaining(), ticks_per_second);
                if (cycles == 0)
                    return false; // timeout
                _umwait(0, Common::X64::FencedRDTSC() + cycles); // umwait is absolute time!!!
                continue;
            }
            bool expected = true;
            if (is_set.compare_exchange_weak(expected, false, std::memory_order_release))
                return true;
        }
    }
    while (!is_set.load() && remaining() > std::chrono::nanoseconds::zero())
        Common::Windows::SleepForOneTick();
    bool expected = true;
    return is_set.compare_exchange_strong(expected, false, std::memory_order_release);
#else
    std::unique_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()), time)};
    if (!condvar.wait_until(lk, deadline, [this] { return is_set.load(); }))
        return false;
    is_set = false;
    return true;
#endif
}
#else
bool Event::WaitFor(const std::chrono::nanoseconds time) {
#ifdef _WIN32
    auto const end = Common::g_wall_clock.GetTimeNS() + time;
    while (!is_set.load() && end > Common::g_wall_clock.GetTimeNS())
        Common::Windows::SleepForOneTick();
    if (is_set.load())
        Reset();
    return true;
#else
    std::unique_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()), time)};
    if (!condvar.wait_until(lk, deadline, [this] { return is_set.load(); }))
        return false;
    is_set = false;
    return true;
#endif
}
#endif

#if defined(ARCHITECTURE_x86_64) && defined(__clang__)
__attribute__((target("waitpkg,mwaitx")))
#endif
bool WaitForFlag(const std::atomic<bool>& flag, std::chrono::steady_clock::time_point deadline) {
#ifdef ARCHITECTURE_x86_64
    if (const auto& caps = Common::g_cpu_caps; caps.monitorx || caps.waitpkg) {
        auto* const line = const_cast<std::atomic<bool>*>(std::addressof(flag));
        const u64 ticks_per_second = Common::g_wall_clock.NsToTicks(std::chrono::seconds{1});
        while (true) {
            if (caps.monitorx) {
                _mm_monitorx(line, 0, 0);
            } else {
                _umonitor(line);
            }
            if (flag.load(std::memory_order_acquire)) {
                return true;
            }
            // Armed before the load, so a store after it ends the wait. Interrupts and stores to
            // the rest of the line end it early too: wait again until the deadline.
            const u32 cycles = MwaitxTimeoutCycles(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    deadline - std::chrono::steady_clock::now()),
                ticks_per_second);
            if (cycles == 0) {
                return false;
            }
            if (caps.monitorx) {
                _mm_mwaitx(1 << 1, 0u, cycles);
            } else {
                _umwait(0, Common::X64::FencedRDTSC() + cycles);
            }
        }
    }
#endif
    while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    return flag.load(std::memory_order_acquire);
}

bool Event::SpinFor(const std::chrono::nanoseconds time) {
    const auto deadline = std::chrono::steady_clock::now() + time;
    while (WaitForFlag(is_set, deadline)) {
        bool expected = true;
        if (is_set.compare_exchange_strong(expected, false, std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

Event::~Event() {
    if (sleep_wake) {
        CloseHandle(sleep_wake);
    }
}

void Event::WakeSleeper() {
    SetEvent(sleep_wake);
}

// Timer and wake event as in Eglantine #538 (a51de161ed, pretend7979): a high-resolution timer
// can expire between timer-resolution ticks.
bool Event::SleepFor(const std::chrono::nanoseconds time) {
    struct Timer {
        Timer() {
            handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                            TIMER_ALL_ACCESS);
            if (!handle) {
                handle = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
            }
        }
        ~Timer() {
            if (handle) {
                CloseHandle(handle);
            }
        }
        HANDLE handle;
    };
    static thread_local Timer timer;
    using TimerTicks = std::chrono::duration<s64, std::ratio<1, 10'000'000>>;

    std::unique_lock lk{mutex};
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()), time)};
    if (!sleep_wake) {
        sleep_wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }
    while (!is_set.load()) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::nanoseconds::zero()) {
            return false;
        }
        LARGE_INTEGER due{};
        due.QuadPart = -std::chrono::ceil<TimerTicks>(remaining).count();
        if (!sleep_wake || !timer.handle ||
            !SetWaitableTimerEx(timer.handle, &due, 0, nullptr, nullptr, nullptr, 0)) {
            condvar.wait_until(lk, deadline, [this] { return is_set.load(); });
            continue;
        }
        // Set() signals sleep_wake while a sleeper waits; the timeout only backs up the timer.
        ++sleepers;
        lk.unlock();
        const HANDLE handles[]{static_cast<HANDLE>(sleep_wake), timer.handle};
        const auto backstop = std::chrono::ceil<std::chrono::milliseconds>(remaining).count() + 1;
        const DWORD result = WaitForMultipleObjects(
            2, handles, FALSE, static_cast<DWORD>((std::min<s64>)(backstop, INFINITE - 1)));
        lk.lock();
        --sleepers;
        if (result == WAIT_FAILED) {
            condvar.wait_until(lk, deadline, [this] { return is_set.load(); });
        }
    }
    is_set = false;
    return true;
}
#else
bool Event::SleepFor(const std::chrono::nanoseconds time) {
    return WaitFor(time);
}
#endif

} // namespace Common
