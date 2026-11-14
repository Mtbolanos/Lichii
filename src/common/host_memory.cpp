// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32

#include <iterator>
#include "common/container/unordered_map.h"
#include <boost/icl/separate_interval_set.hpp>
#include <windows.h>
#include <psapi.h>
#include <vector>
#include "common/dynamic_library.h"

#else // ^^^ Windows ^^^ vvv POSIX vvv

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/icl/interval_set.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common/scope_exit.h"
#include "common/settings.h"

#if defined(__linux__)
#include <sys/random.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/random.h>
#include <mach/vm_map.h>
#include <mach/mach.h>
#elif defined(__FreeBSD__)
#include <sys/shm.h>
#elif defined(__OPENORBIS__)
#include <orbis/libkernel.h>
#endif

// FreeBSD
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
// Solaris 11 and illumos
#ifndef MAP_ALIGNED_SUPER
#define MAP_ALIGNED_SUPER 0
#endif
// macOS
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#endif // ^^^ POSIX ^^^

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <vector>

#include "common/alignment.h"
#include "common/nx2_hitch.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/free_region_manager.h"
#include "common/host_memory.h"
#include "common/logging.h"

#if defined(__ANDROID__) && __ANDROID_API__ < 30
#include <sys/syscall.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
static int memfd_create(const char* name, unsigned int flags) {
    return syscall(__NR_memfd_create, name, flags);
}
#endif

namespace Common {

[[maybe_unused]] constexpr size_t PageAlignment = 0x1000;
[[maybe_unused]] constexpr int BackingAttempts = 5; // fastmem backing allocation tries (Windows)
[[maybe_unused]] constexpr size_t HugePageSize = 0x200000;

#ifdef _WIN32

// Manually imported for MinGW compatibility
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

using PFN_CreateFileMapping2 = _Ret_maybenull_ HANDLE(WINAPI*)(
    _In_ HANDLE File, _In_opt_ SECURITY_ATTRIBUTES* SecurityAttributes, _In_ ULONG DesiredAccess,
    _In_ ULONG PageProtection, _In_ ULONG AllocationAttributes, _In_ ULONG64 MaximumSize,
    _In_opt_ PCWSTR Name,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_VirtualAlloc2 = _Ret_maybenull_ PVOID(WINAPI*)(
    _In_opt_ HANDLE Process, _In_opt_ PVOID BaseAddress, _In_ SIZE_T Size,
    _In_ ULONG AllocationType, _In_ ULONG PageProtection,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_MapViewOfFile3 = _Ret_maybenull_ PVOID(WINAPI*)(
    _In_ HANDLE FileMapping, _In_opt_ HANDLE Process, _In_opt_ PVOID BaseAddress,
    _In_ ULONG64 Offset, _In_ SIZE_T ViewSize, _In_ ULONG AllocationType, _In_ ULONG PageProtection,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_UnmapViewOfFile2 = BOOL(WINAPI*)(_In_ HANDLE Process, _In_ PVOID BaseAddress,
                                           _In_ ULONG UnmapFlags);

template <typename T>
static void GetFuncAddress(Common::DynamicLibrary& dll, const char* name, T& pfn) {
    if (!dll.GetSymbol(name, &pfn)) {
        LOG_CRITICAL(HW_Memory, "Failed to load {}", name);
    }
}

namespace {
/// EDEN_NX2_SEC_RESERVE (on by default, =0 off): the guest memory section is created reserved, and
/// HostMemory::Commit commits its pages as the kernel hands them out (see host_memory.h).
bool ReserveGuestMemory() {
    // On by default (2026-11-08: Castle Town 15.1 -> 12.8 GB working set with the prebuild
    // streaming, ~3 GB less commit, 0 fault commits); EDEN_NX2_SEC_RESERVE=0 turns it off.
    const char* value = std::getenv("EDEN_NX2_SEC_RESERVE");
    return value == nullptr || value[0] != '0';
}

/// The views of a reserved section, for ReservedPageHandler.
struct ReservedViews {
    std::atomic<bool> active{};
    u8* backing_base{};
    size_t backing_size{};
    u8* virtual_base{};
    size_t virtual_size{};
};
ReservedViews g_reserved_views;
std::atomic<u64> g_committed_bytes{};  ///< Committed by HostMemory::Commit
std::atomic<u64> g_net_commits{};      ///< Committed by ReservedPageHandler
std::atomic<u64> g_first_net_offset{}; ///< Its first: backing offset, or virtual offset | 1 << 63

/// The safety net of EDEN_NX2_SEC_RESERVE. Guest pages are committed when the kernel hands them
/// out, not here, but a page some path touched without that would fault: dynarmic takes a fault in
/// its code for a fastmem miss and moves the instruction to the slow path for good (which then
/// faults on the backing view), and other code would crash. Vectored handlers run before
/// dynarmic's (it handles faults through the function tables of its code), so this one sees it
/// first: on a reserved page of a guest memory view it commits the 64 KiB around the address,
/// counts it (the memory log prints the count, which must stay 0: anything else is a missed commit
/// to fix) and retries the access. Every other fault (an unmapped or protected guest page, a fault
/// elsewhere) is passed on as before.
LONG CALLBACK ReservedPageHandler(PEXCEPTION_POINTERS info) {
    const EXCEPTION_RECORD& record = *info->ExceptionRecord;
    if (record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2 ||
        !g_reserved_views.active.load(std::memory_order_acquire)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const ReservedViews& views = g_reserved_views;
    u8* const address = reinterpret_cast<u8*>(record.ExceptionInformation[1]);
    const bool in_backing =
        address >= views.backing_base && address < views.backing_base + views.backing_size;
    const bool in_virtual =
        address >= views.virtual_base && address < views.virtual_base + views.virtual_size;
    if (!in_backing && !in_virtual) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    MEMORY_BASIC_INFORMATION region{};
    if (VirtualQuery(address, &region, sizeof(region)) == 0 || region.State != MEM_RESERVE ||
        region.Type != MEM_MAPPED) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    constexpr uintptr_t Granule = 0x10000;
    u8* const granule = reinterpret_cast<u8*>(reinterpret_cast<uintptr_t>(address) & ~(Granule - 1));
    u8* const region_begin = static_cast<u8*>(region.BaseAddress);
    u8* const begin = (std::max)(granule, region_begin);
    u8* const end = (std::min)(granule + Granule, region_begin + region.RegionSize);
    if (VirtualAlloc(begin, static_cast<size_t>(end - begin), MEM_COMMIT, PAGE_READWRITE) ==
        nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (g_net_commits.fetch_add(1, std::memory_order_relaxed) == 0) {
        g_first_net_offset.store(in_backing ? static_cast<u64>(address - views.backing_base)
                                            : static_cast<u64>(address - views.virtual_base) |
                                                  (u64{1} << 63),
                                 std::memory_order_relaxed);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}
} // Anonymous namespace

class HostMemory::Impl {
public:
    explicit Impl(size_t backing_size_, size_t virtual_size_)
        : backing_size{backing_size_}
        , virtual_size{virtual_size_}
        , process{GetCurrentProcess()}
        , kernelbase_dll("Kernelbase")
    {}

    bool Init() {
        if (!kernelbase_dll.IsOpen()) {
            LOG_CRITICAL(HW_Memory, "Failed to load Kernelbase.dll");
            return false;
        }
        GetFuncAddress(kernelbase_dll, "CreateFileMapping2", pfn_CreateFileMapping2);
        GetFuncAddress(kernelbase_dll, "VirtualAlloc2", pfn_VirtualAlloc2);
        GetFuncAddress(kernelbase_dll, "MapViewOfFile3", pfn_MapViewOfFile3);
        GetFuncAddress(kernelbase_dll, "UnmapViewOfFile2", pfn_UnmapViewOfFile2);

        if (!pfn_CreateFileMapping2 || !pfn_VirtualAlloc2 || !pfn_MapViewOfFile3 || !pfn_UnmapViewOfFile2) {
            LOG_CRITICAL(HW_Memory, "Failed to find functions for virtual allocs");
            return false;
        }

        reserved = ReserveGuestMemory();
        // Allocate backing file map (SEC_RESERVE with EDEN_NX2_SEC_RESERVE). SEC_COMMIT charges all of it to the commit limit up front, and
        // when the free commit is only just above its size the call fails with
        // ERROR_NO_SYSTEM_RESOURCES without growing a system-managed page file. The emulator then
        // runs the whole session without fastmem (Ocarina of Time: 20-25% fewer fps). A private
        // commit of the same size does grow the page file, so make one, release it and try again.
        for (int attempt = 1; attempt <= BackingAttempts && !backing_handle; ++attempt) {
            backing_handle = pfn_CreateFileMapping2(INVALID_HANDLE_VALUE, nullptr, FILE_MAP_WRITE | FILE_MAP_READ, PAGE_READWRITE, reserved ? SEC_RESERVE : SEC_COMMIT, backing_size, nullptr, nullptr, 0);
            if (backing_handle) {
                if (attempt > 1) {
                    LOG_INFO(HW_Memory, "Allocated {} MiB of backing memory on attempt {}", backing_size >> 20, attempt);
                }
                break;
            }
            const DWORD error = GetLastError();
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            LOG_WARNING(HW_Memory, "Failed to allocate {} MiB of backing memory (attempt {}, error {}, {} MiB of commit free)",
                        backing_size >> 20, attempt, error, status.ullAvailPageFile >> 20);
            if (void* const probe = VirtualAlloc(nullptr, backing_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
                VirtualFree(probe, 0, MEM_RELEASE);
            }
            if (attempt < BackingAttempts) {
                Sleep(50);
            }
        }
        if (!backing_handle) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate {} MiB of backing memory", backing_size >> 20);
            return false;
        }
        // Allocate a virtual memory for the backing file map as placeholder
        backing_base = static_cast<u8*>(pfn_VirtualAlloc2(process, nullptr, backing_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
        if (!backing_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to reserve {} MiB of virtual memory", backing_size >> 20);
            return false;
        }
        // Map backing placeholder
        void* const ret = pfn_MapViewOfFile3(backing_handle, process, backing_base, 0, backing_size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
        if (ret != backing_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to map {} MiB of virtual memory", backing_size >> 20);
            return false;
        }
        // Allocate virtual address placeholder
        virtual_base = static_cast<u8*>(pfn_VirtualAlloc2(process, nullptr, virtual_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
        if (!virtual_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to reserve {} GiB of virtual memory", virtual_size >> 30);
            return false;
        }
        if (reserved) {
            g_reserved_views.backing_base = backing_base;
            g_reserved_views.backing_size = backing_size;
            g_reserved_views.virtual_base = virtual_base;
            g_reserved_views.virtual_size = virtual_size;
            g_reserved_views.active.store(true, std::memory_order_release);
            static std::once_flag handler_added;
            std::call_once(handler_added, [] { AddVectoredExceptionHandler(1, ReservedPageHandler); });
            LOG_INFO(HW_Memory,
                     "EDEN_NX2_SEC_RESERVE: {} MiB of guest memory reserved, committed as the "
                     "kernel hands it out",
                     backing_size >> 20);
        }
        return true;
    }

    ~Impl() {
        Release();
    }

    /// EDEN_NX2_SEC_RESERVE: commits [offset, end) of the section through the backing view.
    bool CommitBacking(size_t offset, size_t end) {
        end = (std::min)(end, backing_size);
        if (offset >= end) {
            return true;
        }
        if (VirtualAlloc(backing_base + offset, end - offset, MEM_COMMIT, PAGE_READWRITE) ==
            nullptr) {
            LOG_CRITICAL(HW_Memory,
                         "EDEN_NX2_SEC_RESERVE: committing {:#x} bytes of guest memory at {:#x} "
                         "failed, error {}",
                         end - offset, offset, GetLastError());
            return false;
        }
        return true;
    }

    void* Allocate(size_t size) {
        auto* ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (ptr == nullptr) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate fallback buffer with size {:#x}, error {}", size, GetLastError());
        }
        return ptr;
    }

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms) {
        const Common::Nx2Hitch::Scope timed{Common::Nx2Hitch::Cause::HostMap, length}; // before the lock
        std::unique_lock lock{placeholder_mutex};
        if (!IsNiechePlaceholder(virtual_offset, length)) {
            Split(virtual_offset, length);
        }
        ASSERT(placeholders.find({virtual_offset, virtual_offset + length}) == placeholders.end());
        TrackPlaceholder(virtual_offset, host_offset, length);

        MapView(virtual_offset, host_offset, length);
    }

    void Unmap(size_t virtual_offset, size_t length) {
        const Common::Nx2Hitch::Scope timed{Common::Nx2Hitch::Cause::HostUnmap, length}; // before the lock
        std::scoped_lock lock{placeholder_mutex};

        // Unmap until there are no more placeholders
        while (UnmapOnePlaceholder(virtual_offset, length)) {
        }
    }

    void Protect(size_t virtual_offset, size_t length, bool read, bool write, bool execute) {
        DWORD new_flags{};
        if (read && write) {
            new_flags = PAGE_READWRITE;
        } else if (read && !write) {
            new_flags = PAGE_READONLY;
        } else if (!read && !write) {
            new_flags = PAGE_NOACCESS;
        } else {
            UNIMPLEMENTED_MSG("Protection flag combination read={} write={}", read, write);
        }
        const size_t virtual_end = virtual_offset + length;

        const Common::Nx2Hitch::Scope timed{Common::Nx2Hitch::Cause::HostProtect, length}; // before the lock
        std::scoped_lock lock{placeholder_mutex};
        auto [it, end] = placeholders.equal_range({virtual_offset, virtual_end});
        while (it != end) {
            const size_t offset = (std::max)(it->lower(), virtual_offset);
            const size_t protect_length = (std::min)(it->upper(), virtual_end) - offset;
            DWORD old_flags{};
            if (!VirtualProtect(virtual_base + offset, protect_length, new_flags, &old_flags)) {
                LOG_CRITICAL(HW_Memory, "Failed to change virtual memory protect rules");
            }
            ++it;
        }
    }

    /// The views mapped now: {virtual offset, length, backing offset} each (SampleResident).
    std::vector<std::array<size_t, 3>> MappedViews() {
        std::scoped_lock lock{placeholder_mutex};
        std::vector<std::array<size_t, 3>> views;
        views.reserve(placeholders.iterative_size());
        for (const auto& interval : placeholders) {
            const auto it = placeholder_host_pointers.find(interval.lower());
            if (it != placeholder_host_pointers.end()) {
                views.push_back({interval.lower(), interval.upper() - interval.lower(), it->second});
            }
        }
        return views;
    }

    /// Backing offset the view page at virtual_offset maps, or nullopt when it is not mapped.
    std::optional<size_t> BackingOffset(size_t virtual_offset) {
        std::scoped_lock lock{placeholder_mutex};
        const auto it = placeholders.find(virtual_offset);
        if (it == placeholders.end()) {
            return std::nullopt;
        }
        const auto host_pointer_it = placeholder_host_pointers.find(it->lower());
        if (host_pointer_it == placeholder_host_pointers.end()) {
            return std::nullopt;
        }
        return host_pointer_it->second + virtual_offset - it->lower();
    }

    void EnableDirectMappedAddress() {
        // TODO
        UNREACHABLE();
    }

    const size_t backing_size; ///< Size of the backing memory in bytes
    const size_t virtual_size; ///< Size of the virtual address placeholder in bytes

    u8* backing_base{};
    u8* virtual_base{};
    bool reserved{}; ///< EDEN_NX2_SEC_RESERVE: the section was created with SEC_RESERVE

private:
    /// Release all resources in the object
    void Release() {
        if (reserved) {
            g_reserved_views.active.store(false, std::memory_order_release);
        }
        if (!placeholders.empty()) {
            for (const auto& placeholder : placeholders) {
                if (!pfn_UnmapViewOfFile2(process, virtual_base + placeholder.lower(),
                                          MEM_PRESERVE_PLACEHOLDER)) {
                    LOG_CRITICAL(HW_Memory, "Failed to unmap virtual memory placeholder");
                }
            }
            Coalesce(0, virtual_size);
        }
        if (virtual_base) {
            if (!VirtualFree(virtual_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(HW_Memory, "Failed to free virtual memory");
            }
        }
        if (backing_base) {
            if (!pfn_UnmapViewOfFile2(process, backing_base, MEM_PRESERVE_PLACEHOLDER)) {
                LOG_CRITICAL(HW_Memory, "Failed to unmap backing memory placeholder");
            }
            if (!VirtualFreeEx(process, backing_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(HW_Memory, "Failed to free backing memory");
            }
        }
        if (!CloseHandle(backing_handle)) {
            LOG_CRITICAL(HW_Memory, "Failed to free backing memory file handle");
        }
    }

    /// Unmap one placeholder in the given range (partial unmaps are supported)
    /// Return true when there are no more placeholders to unmap
    bool UnmapOnePlaceholder(size_t virtual_offset, size_t length) {
        const auto it = placeholders.find({virtual_offset, virtual_offset + length});
        const auto begin = placeholders.begin();
        const auto end = placeholders.end();
        if (it == end) {
            return false;
        }
        const size_t placeholder_begin = it->lower();
        const size_t placeholder_end = it->upper();
        const size_t unmap_begin = (std::max)(virtual_offset, placeholder_begin);
        const size_t unmap_end = (std::min)(virtual_offset + length, placeholder_end);
        ASSERT(unmap_begin >= placeholder_begin && unmap_begin < placeholder_end);
        ASSERT(unmap_end <= placeholder_end && unmap_end > placeholder_begin);

        const auto host_pointer_it = placeholder_host_pointers.find(placeholder_begin);
        ASSERT(host_pointer_it != placeholder_host_pointers.end());
        const size_t host_offset = host_pointer_it->second;

        const bool split_left = unmap_begin > placeholder_begin;
        const bool split_right = unmap_end < placeholder_end;

        if (!pfn_UnmapViewOfFile2(process, virtual_base + placeholder_begin,
                                  MEM_PRESERVE_PLACEHOLDER)) {
            LOG_CRITICAL(HW_Memory, "Failed to unmap placeholder");
        }
        // If we have to remap memory regions due to partial unmaps, we are in a data race as
        // Windows doesn't support remapping memory without unmapping first. Avoid adding any extra
        // logic within the panic region described below.

        // Panic region, we are in a data race right now
        if (split_left || split_right) {
            Split(unmap_begin, unmap_end - unmap_begin);
        }
        if (split_left) {
            MapView(placeholder_begin, host_offset, unmap_begin - placeholder_begin);
        }
        if (split_right) {
            MapView(unmap_end, host_offset + unmap_end - placeholder_begin,
                    placeholder_end - unmap_end);
        }
        // End panic region

        size_t coalesce_begin = unmap_begin;
        if (!split_left) {
            // Try to coalesce pages to the left
            coalesce_begin = it == begin ? 0 : std::prev(it)->upper();
            if (coalesce_begin != placeholder_begin) {
                Coalesce(coalesce_begin, unmap_end - coalesce_begin);
            }
        }
        if (!split_right) {
            // Try to coalesce pages to the right
            const auto next = std::next(it);
            const size_t next_begin = next == end ? virtual_size : next->lower();
            if (placeholder_end != next_begin) {
                // We can coalesce to the right
                Coalesce(coalesce_begin, next_begin - coalesce_begin);
            }
        }
        // Remove and reinsert placeholder trackers
        UntrackPlaceholder(it);
        if (split_left) {
            TrackPlaceholder(placeholder_begin, host_offset, unmap_begin - placeholder_begin);
        }
        if (split_right) {
            TrackPlaceholder(unmap_end, host_offset + unmap_end - placeholder_begin,
                             placeholder_end - unmap_end);
        }
        return true;
    }

    void MapView(size_t virtual_offset, size_t host_offset, size_t length) {
        if (!pfn_MapViewOfFile3(backing_handle, process, virtual_base + virtual_offset, host_offset,
                                length, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) {
            LOG_CRITICAL(HW_Memory, "Failed to map placeholder");
        }
    }

    void Split(size_t virtual_offset, size_t length) {
        if (!VirtualFreeEx(process, reinterpret_cast<LPVOID>(virtual_base + virtual_offset), length,
                           MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            LOG_CRITICAL(HW_Memory, "Failed to split placeholder");
        }
    }

    void Coalesce(size_t virtual_offset, size_t length) {
        if (!VirtualFreeEx(process, reinterpret_cast<LPVOID>(virtual_base + virtual_offset), length,
                           MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
            LOG_CRITICAL(HW_Memory, "Failed to coalesce placeholders");
        }
    }

    void TrackPlaceholder(size_t virtual_offset, size_t host_offset, size_t length) {
        placeholders.insert({virtual_offset, virtual_offset + length});
        placeholder_host_pointers.emplace(virtual_offset, host_offset);
    }

    void UntrackPlaceholder(boost::icl::separate_interval_set<size_t>::iterator it) {
        placeholder_host_pointers.erase(it->lower());
        placeholders.erase(it);
    }

    /// Return true when a given memory region is a "nieche" and the placeholders don't have to be
    /// split.
    bool IsNiechePlaceholder(size_t virtual_offset, size_t length) const {
        const auto it = placeholders.upper_bound({virtual_offset, virtual_offset + length});
        if (it != placeholders.end() && it->lower() == virtual_offset + length) {
            return it == placeholders.begin() ? virtual_offset == 0
                                              : std::prev(it)->upper() == virtual_offset;
        }
        return false;
    }

    HANDLE process{};        ///< Current process handle
    HANDLE backing_handle{}; ///< File based backing memory

    DynamicLibrary kernelbase_dll;
    PFN_CreateFileMapping2 pfn_CreateFileMapping2{};
    PFN_VirtualAlloc2 pfn_VirtualAlloc2{};
    PFN_MapViewOfFile3 pfn_MapViewOfFile3{};
    PFN_UnmapViewOfFile2 pfn_UnmapViewOfFile2{};

    std::mutex placeholder_mutex;                                 ///< Mutex for placeholders
    boost::icl::separate_interval_set<size_t> placeholders;       ///< Mapped placeholders
    ::Common::unordered_map<size_t, size_t> placeholder_host_pointers; ///< Placeholder backing offset
};

#elif defined(__OPENORBIS__) || defined(__managarm__)
// None of the luxuries of POSIX, all of the suffering
// For managarm: see https://github.com/managarm/managarm/issues/1370
#else // ^^^ Windows ^^^ vvv POSIX vvv

#ifndef MAP_NOCORE
#define MAP_NOCORE 0
#endif

#if defined(ARCHITECTURE_arm64) && !defined(__APPLE__)

static void* ChooseVirtualBase(size_t virtual_size) {
    constexpr uintptr_t Map39BitSize = (1ULL << 39);
    constexpr uintptr_t Map36BitSize = (1ULL << 36);

    // This is not a cryptographic application, we just want something random.
    std::mt19937_64 rng;

    // We want to ensure we are allocating at an address aligned to the L2 block size.
    // For Qualcomm devices, we must also allocate memory above 36 bits.
    const size_t lower = Map36BitSize / HugePageSize;
    const size_t upper = (Map39BitSize - virtual_size) / HugePageSize;
    const size_t range = upper - lower;

    // Try up to 64 times to allocate memory at random addresses in the range.
    for (int i = 0; i < 64; i++) {
        // Calculate a possible location.
        uintptr_t hint_address = ((rng() % range) + lower) * HugePageSize;

        // Try to map.
        // Note: we may be able to take advantage of MAP_FIXED_NOREPLACE here.
        void* map_pointer =
            mmap(reinterpret_cast<void*>(hint_address), virtual_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_NOCORE, -1, 0);

        // If we successfully mapped, we're done.
        if (reinterpret_cast<uintptr_t>(map_pointer) == hint_address) {
            return map_pointer;
        }

        // Unmap if necessary, and try again.
        if (map_pointer != MAP_FAILED) {
            munmap(map_pointer, virtual_size);
        }
    }

    return MAP_FAILED;
}

#else

static void* ChooseVirtualBase(size_t virtual_size) {
#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__OpenBSD__) || defined(__sun__) || defined(__HAIKU__) || defined(__managarm__) || defined(__AIX__)
    void* virtual_base = mmap(nullptr, virtual_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_ALIGNED_SUPER | MAP_NOCORE, -1, 0);
    if (virtual_base != MAP_FAILED)
        return virtual_base;
#endif
    return mmap(nullptr, virtual_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_NOCORE, -1, 0);
}

#endif

#if defined(__sun__) || defined(__HAIKU__) || defined(__NetBSD__) || defined(__DragonFly__)
/// Most Unices don't have a portable shm_open (AIX, OpenBSD, NetBSD, Solaris 11, OpenIndiana)
/// Portable implementation of shm_open(SHM_ANON, ...) - roughly equivalent but without
/// OS support - may fail sporadically, beware!
static int shm_open_anon(int flags, mode_t mode) {
    char name[16] = "/shm-";
    char *const limit = name + sizeof(name) - 1;
    *limit = '\0';
    char *start = name + strlen(name);
    for (int tries = 0; tries < 4; tries++) {
        struct timespec tv;
        clock_gettime(CLOCK_REALTIME, &tv);
        unsigned long r = (unsigned long)tv.tv_sec + (unsigned long)tv.tv_nsec;
        for (char *fill = start; fill < limit; r /= 8)
            *fill++ = '0' + (r % 8);
        int fd = shm_open(name, flags, mode);
        if (fd != -1) {
            if (shm_unlink(name) == -1) {
                int tmp = errno;
                close(fd);
                errno = tmp;
                return -1;
            }
            return fd;
        }
        if (errno != EEXIST)
            break;
    }
    return -1;
}
#elif defined(__OpenBSD__)
/// Except OpenBSD which explicitly uses shm_mkstemp instead (as a more secure alternative)
static int shm_open_anon(int flags, mode_t mode) {
    char name[16] = "/shm-XXXXXXXXXX";
    int fd;
    if ((fd = shm_mkstemp(name)) == -1)
        return -1;
    if (shm_unlink(name) == -1) {
        int tmp = errno;
        close(fd);
        errno = tmp;
        return -1;
    }
    return fd;
}
#endif

class HostMemory::Impl {
public:
    explicit Impl(size_t backing_size_, size_t virtual_size_)
        : backing_size{backing_size_}
        , virtual_size{virtual_size_}
    {}

#ifdef __linux__
    bool PunchHole(size_t offset, size_t length) {
        return fd >= 0 &&
               fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(offset),
                         static_cast<off_t>(length)) == 0;
    }
#endif

    bool Init() {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size < static_cast<long>(PageAlignment) ||
            page_size % static_cast<long>(PageAlignment) != 0) {
            LOG_WARNING(Common_Memory, "Host page size {:#x} does not support fastmem", page_size);
            return false;
        }
#ifdef HAS_NCE
        if (page_size != static_cast<long>(PageAlignment) && Settings::IsNceEnabled()) {
            LOG_WARNING(Common_Memory, "Host page size {:#x} does not support fastmem with NCE", page_size);
            return false;
        }
#endif
        host_page_size = static_cast<size_t>(page_size);
        // Backing memory initialization
#if defined(__sun__) || defined(__HAIKU__) || defined(__NetBSD__) || defined(__DragonFly__)
        fd = shm_open_anon(O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
#elif defined(__OpenBSD__)
        fd = shm_open_anon(O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
#elif defined(__FreeBSD__)
        fd = shm_open(SHM_ANON, O_RDWR, 0600);
#elif defined(__APPLE__)
        char shm_name[32];
        std::snprintf(shm_name, sizeof(shm_name), "/eden_mem_%d", int(getpid()));
        fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) {
            shm_unlink(shm_name);
        } else {
            char template_path[] = "/tmp/eden_mem_XXXXXX";
            fd = mkstemp(template_path);
            if (fd >= 0) {
                unlink(template_path);
            }
        }
#elif defined(__managarm__)
        // macOS doesn't have memfd_create, use anonymous temporary file
        char template_path[] = "/tmp/eden_mem_XXXXXX";
        fd = mkstemp(template_path);
        if (fd >= 0) {
            unlink(template_path);
        }
#else
        fd = memfd_create("HostMemory", 0);
#endif
        bool use_anon = false;
        if (fd <= 0) {
            LOG_WARNING(Common_Memory, "memfd_create: {}", strerror(errno));
            use_anon = true;
        }
        if (!use_anon) {
            // Defined to extend the file with zeros
            int ret = ftruncate(fd, backing_size);
            if (ret != 0) {
                LOG_WARNING(Common_Memory, "ftruncate: {} (likely out-of-emory)", strerror(errno));
                use_anon = true;
            }
        }
        if (use_anon) {
            LOG_WARNING(Common_Memory, "Using private mappings instead of shared ones");
            backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NOCORE, -1, 0));
            if (fd > 0) {
                fd = -1;
                close(fd);
            }
        } else {
            backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NOCORE, fd, 0));
        }
        if (backing_base == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
            return false;
        }

        // Virtual memory initialization
        virtual_base = virtual_map_base = static_cast<u8*>(ChooseVirtualBase(virtual_size));
        if (virtual_base == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
            return false;
        }
#if defined(__linux__)
        madvise(virtual_base, virtual_size, MADV_HUGEPAGE);
#endif
        free_manager.SetAddressSpace(virtual_base, virtual_size);

        if (host_page_size != PageAlignment) {
            if (fd < 0 || virtual_size % host_page_size != 0) {
                LOG_WARNING(Common_Memory, "Can't emulate {:#x} byte pages with host page size {:#x}",
                            PageAlignment, host_page_size);
                return false;
            }
            page_states_size = (virtual_size / PageAlignment) * sizeof(u32);
            void* const states = mmap(nullptr, page_states_size, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (states == MAP_FAILED) {
                LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
                return false;
            }
            page_states = static_cast<u32*>(states);
            LOG_INFO(Common_Memory, "Fastmem maps {:#x} byte host pages whose guest pages are contiguous",
                     host_page_size);
        }
        return true;
    }

    ~Impl() {
        Release();
    }

    void* Allocate(size_t size) {
        auto* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (ptr == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate fallback buffer with size {:#x}, {}", size, strerror(errno));
        }
        return ptr;
    }

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms) {
        // Intersect the range with our address space.
        const size_t intended_offset = virtual_offset;
        AdjustMap(&virtual_offset, &length);
        if (length != 0) {
            host_offset += virtual_offset - intended_offset;
        }

        if (page_states) {
            if (length == 0) {
                return;
            }
            std::scoped_lock lk{page_states_mutex};
            const size_t first = StateIndex(virtual_offset);
            const u32 state = PageMapped | PermissionBits(perms);
            for (size_t i = 0; i < length / PageAlignment; ++i) {
                const u32 host_page = static_cast<u32>(host_offset / PageAlignment + i);
                page_states[first + i] = state | (host_page << PageOffsetShift);
            }
            UpdateHostPages(virtual_offset, length, true);
            return;
        }

        // We are removing a placeholder.
        free_manager.AllocateBlock(virtual_base + virtual_offset, length);

        // Deduce mapping protection flags.
        int prot_flags = PROT_NONE;
        if (True(perms & MemoryPermission::Read))
            prot_flags |= PROT_READ;
        if (True(perms & MemoryPermission::Write))
            prot_flags |= PROT_WRITE;
#ifdef ARCHITECTURE_arm64
        if (True(perms & MemoryPermission::Execute))
            prot_flags |= PROT_EXEC;
#endif
        int flags = (fd >= 0 ? MAP_SHARED : MAP_PRIVATE) | MAP_FIXED;
        void* ret = mmap(virtual_base + virtual_offset, length, prot_flags, flags, fd, host_offset);
        ASSERT_MSG(ret != MAP_FAILED, "mmap: {} {}", strerror(errno), fd);
    }

    void Unmap(size_t virtual_offset, size_t length) {
        // The method name is wrong. We're still talking about the virtual range.
        // We don't want to unmap, we want to reserve this memory.

        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        if (page_states) {
            if (length == 0) {
                return;
            }
            std::scoped_lock lk{page_states_mutex};
            std::fill_n(page_states + StateIndex(virtual_offset), length / PageAlignment, 0u);
            UpdateHostPages(virtual_offset, length, true);
            return;
        }

        // Merge with any adjacent placeholder mappings.
        auto [merged_pointer, merged_size] =
            free_manager.FreeBlock(virtual_base + virtual_offset, length);

        void* ret = mmap(merged_pointer, merged_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
    }

    void Protect(size_t virtual_offset, size_t length, bool read, bool write, bool execute) {
        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        if (page_states) {
            if (length == 0) {
                return;
            }
            const u32 bits = (read ? PageRead : 0u) | (write ? PageWrite : 0u) |
                             (execute ? PageExecute : 0u);
            std::scoped_lock lk{page_states_mutex};
            u32* const states = page_states + StateIndex(virtual_offset);
            for (size_t i = 0; i < length / PageAlignment; ++i) {
                states[i] = (states[i] & ~PagePermissionMask) | bits;
            }
            UpdateHostPages(virtual_offset, length, false);
            return;
        }

        int flags = PROT_NONE;
        if (read) {
            flags |= PROT_READ;
        }
        if (write) {
            flags |= PROT_WRITE;
        }
#ifdef HAS_NCE
        if (execute) {
            flags |= PROT_EXEC;
        }
#endif
        int ret = mprotect(virtual_base + virtual_offset, length, flags);
        ASSERT_MSG(ret == 0, "mprotect failed: {}", strerror(errno));
    }

    void EnableDirectMappedAddress() {
        virtual_base = nullptr;
    }

    const size_t backing_size; ///< Size of the backing memory in bytes
    const size_t virtual_size; ///< Size of the virtual address placeholder in bytes

    u8* backing_base{reinterpret_cast<u8*>(MAP_FAILED)};
    u8* virtual_base{reinterpret_cast<u8*>(MAP_FAILED)};
    u8* virtual_map_base{reinterpret_cast<u8*>(MAP_FAILED)};

private:
    /// Release all resources in the object
    void Release() {
        if (page_states != nullptr) {
            munmap(page_states, page_states_size);
        }

        if (virtual_map_base != MAP_FAILED) {
            int ret = munmap(virtual_map_base, virtual_size);
            ASSERT_MSG(ret == 0, "munmap failed: {}", strerror(errno));
        }

        if (backing_base != MAP_FAILED) {
            int ret = munmap(backing_base, backing_size);
            ASSERT_MSG(ret == 0, "munmap failed: {}", strerror(errno));
        }

        if (fd != -1) {
            int ret = close(fd);
            ASSERT_MSG(ret == 0, "close failed: {}", strerror(errno));
        }
    }

    void AdjustMap(size_t* virtual_offset, size_t* length) {
        if (virtual_base != nullptr) {
            return;
        }

        // If we are direct mapped, we want to make sure we are operating on a region
        // that is in range of our virtual mapping.
        size_t intended_start = *virtual_offset;
        size_t intended_end = intended_start + *length;
        size_t address_space_start = reinterpret_cast<size_t>(virtual_map_base);
        size_t address_space_end = address_space_start + virtual_size;

        if (address_space_start > intended_end || intended_start > address_space_end) {
            *virtual_offset = 0;
            *length = 0;
        } else {
            *virtual_offset = (std::max)(intended_start, address_space_start);
            *length = (std::min)(intended_end, address_space_end) - *virtual_offset;
        }
    }

    static constexpr u32 PageRead = 1u << 0;
    static constexpr u32 PageWrite = 1u << 1;
    static constexpr u32 PageExecute = 1u << 2;
    static constexpr u32 PagePermissionMask = PageRead | PageWrite | PageExecute;
    static constexpr u32 PageMapped = 1u << 3;
    static constexpr u32 PageOffsetShift = 4;

    static u32 PermissionBits(MemoryPermission perms) {
        return (True(perms & MemoryPermission::Read) ? PageRead : 0u) |
               (True(perms & MemoryPermission::Write) ? PageWrite : 0u) |
               (True(perms & MemoryPermission::Execute) ? PageExecute : 0u);
    }

    static int ProtectionFlags(u32 bits) {
        int prot = PROT_NONE;
        if ((bits & PageRead) != 0) {
            prot |= PROT_READ;
        }
        if ((bits & PageWrite) != 0) {
            prot |= PROT_WRITE;
        }
#ifdef HAS_NCE
        if ((bits & PageExecute) != 0) {
            prot |= PROT_EXEC;
        }
#endif
        return prot;
    }

    struct HostPage {
        bool mapped;
        size_t host_offset;
        int prot;
    };

    size_t StateIndex(size_t virtual_offset) const {
        const size_t base = virtual_base != nullptr ? 0 : reinterpret_cast<size_t>(virtual_map_base);
        return (virtual_offset - base) / PageAlignment;
    }

    HostPage ResolveHostPage(size_t virtual_offset) const {
        const u32* const states = page_states + StateIndex(virtual_offset);
        const u32 first = states[0];
        const size_t host_offset = size_t(first >> PageOffsetShift) * PageAlignment;
        if ((first & PageMapped) == 0 || host_offset % host_page_size != 0) {
            return {};
        }
        u32 permissions = first & PagePermissionMask;
        for (size_t i = 1; i < host_page_size / PageAlignment; ++i) {
            const u32 state = states[i];
            if ((state & PageMapped) == 0 || (state >> PageOffsetShift) != (first >> PageOffsetShift) + i) {
                return {};
            }
            permissions &= state;
        }
        return {true, host_offset, ProtectionFlags(permissions)};
    }

    void UpdateHostPages(size_t virtual_offset, size_t length, bool remap) {
        const size_t begin = AlignDown(virtual_offset, host_page_size);
        const size_t end = AlignUp(virtual_offset + length, host_page_size);
        const auto apply = [&](size_t run_begin, size_t run_end, const HostPage& page) {
            u8* const address = virtual_base + run_begin;
            const size_t size = run_end - run_begin;
            if (!page.mapped) {
                if (remap) {
                    void* ret = mmap(address, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
                    ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
                }
            } else if (remap) {
                void* ret = mmap(address, size, page.prot, MAP_SHARED | MAP_FIXED, fd, off_t(page.host_offset));
                ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
            } else {
                int ret = mprotect(address, size, page.prot);
                ASSERT_MSG(ret == 0, "mprotect failed: {}", strerror(errno));
            }
        };
        size_t run_begin = begin;
        HostPage run = ResolveHostPage(begin);
        for (size_t offset = begin + host_page_size; offset < end; offset += host_page_size) {
            const HostPage page = ResolveHostPage(offset);
            const bool continues = page.mapped == run.mapped &&
                                   (!page.mapped || (page.prot == run.prot &&
                                                     page.host_offset == run.host_offset + (offset - run_begin)));
            if (!continues) {
                apply(run_begin, offset, run);
                run_begin = offset;
                run = page;
            }
        }
        apply(run_begin, end, run);
    }

    int fd{-1}; // memfd file descriptor, -1 is the error value of memfd_create
    FreeRegionManager free_manager{};
    size_t host_page_size{PageAlignment};
    u32* page_states{};
    size_t page_states_size{};
    std::mutex page_states_mutex;
};

#endif // ^^^ POSIX ^^^

HostMemory::HostMemory(size_t backing_size_, size_t virtual_size_)
    : backing_size(backing_size_)
    , virtual_size(virtual_size_)
    , num_granules(Common::DivCeil(backing_size_, size_t{1} << GranuleShift))
{
    granules = std::make_unique<std::atomic<u8>[]>(num_granules);
#if defined(__OPENORBIS__) || defined(__managarm__)
    LOG_WARNING(HW_Memory, "Platform doesn't support fastmem");
    backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    virtual_base = nullptr;
#else
    // Try to allocate a fastmem arena.
    impl = std::make_unique<HostMemory::Impl>(AlignUp(backing_size, PageAlignment), AlignUp(virtual_size, PageAlignment) + HugePageSize);
    if (impl->Init()) {
        backing_base = impl->backing_base;
        virtual_base = impl->virtual_base;
#ifdef _WIN32
        commit_on_use = impl->reserved;
#endif
        if (virtual_base) {
            // Ensure the virtual base is aligned to the L2 block size.
            virtual_base = reinterpret_cast<u8*>(Common::AlignUp(uintptr_t(virtual_base), HugePageSize));
            virtual_base_offset = virtual_base - impl->virtual_base;
        }
    } else {
        LOG_WARNING(HW_Memory, "Platform can support fastmem, but can't create it");
        fallback_buffer = true;
        backing_base = static_cast<u8*>(impl->Allocate(backing_size));
        virtual_base = nullptr;
        impl.reset();
    }
#endif
}

HostMemory::~HostMemory() {
#ifdef _WIN32
    if (fallback_buffer) {
        VirtualFree(backing_base, backing_size, MEM_RELEASE);
    }
#else
    if (fallback_buffer) {
        munmap(backing_base, backing_size);
    }
#endif
}

HostMemory::HostMemory(HostMemory&&) noexcept = default;

HostMemory& HostMemory::operator=(HostMemory&&) noexcept = default;

void HostMemory::Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms, bool separate_heap) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(host_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(virtual_offset + length <= virtual_size);
    ASSERT(host_offset + length <= backing_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    if (commit_on_use) {
        // Pages the kernel handed out are committed already; this covers any it did not.
        Commit(host_offset, length);
    }
    impl->Map(virtual_offset + virtual_base_offset, host_offset, length, perms);
#endif
}

void HostMemory::Unmap(size_t virtual_offset, size_t length, bool separate_heap) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(virtual_offset + length <= virtual_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    impl->Unmap(virtual_offset + virtual_base_offset, length);
#endif
}

void HostMemory::Protect(size_t virtual_offset, size_t length, MemoryPermission perm) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(virtual_offset + length <= virtual_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    const bool read = True(perm & MemoryPermission::Read);
    const bool write = True(perm & MemoryPermission::Write);
    const bool execute = True(perm & MemoryPermission::Execute);
    impl->Protect(virtual_offset + virtual_base_offset, length, read, write, execute);
#endif
}

void HostMemory::HintCold(size_t physical_offset, size_t length,
                          std::optional<size_t> virtual_offset) const {
    if (length == 0 || backing_base == nullptr) {
        return;
    }
    const auto hint = [length](u8* address) {
#ifdef _WIN32
        // Unlocking pages that are not locked takes them out of the working set.
        VirtualUnlock(address, length);
#elif defined(__linux__) && defined(MADV_PAGEOUT)
        madvise(address, length, MADV_PAGEOUT);
#else
        static_cast<void>(address);
#endif
    };
    hint(backing_base + physical_offset);
    if (virtual_offset && virtual_base != nullptr && *virtual_offset + length <= virtual_size) {
        hint(virtual_base + *virtual_offset);
    }
}

namespace {
std::atomic<u64> g_zero_filled_bytes{};

[[maybe_unused]] bool IsZeroPage(const u8* page) {
    // Stops at the first non-zero line. Read whole, the ~6 GB of resident guest pages took
    // 1.35-1.47 s a pass (fable52_answer_1), and most pages hold data in their first bytes.
    const auto* words = reinterpret_cast<const u64*>(page);
    constexpr size_t LineWords = 8; // 64 bytes
    for (size_t i = 0; i < PageAlignment / sizeof(u64); i += LineWords) {
        u64 bits = 0;
        for (size_t j = 0; j < LineWords; ++j) {
            bits |= words[i + j];
        }
        if (bits != 0) {
            return false;
        }
    }
    return true;
}
} // namespace

u64 HostMemory::ZeroFilledBytes() noexcept {
    return g_zero_filled_bytes.load(std::memory_order_relaxed);
}

namespace {
std::mutex g_resident_mutex;
HostMemory::ResidentSample g_resident_sample{};
} // Anonymous namespace

HostMemory::ResidentSample HostMemory::SampleResident(size_t stride_pages) const {
    ResidentSample sample{};
#ifdef _WIN32
    if (backing_base == nullptr || stride_pages == 0) {
        return sample;
    }
    const auto start = std::chrono::steady_clock::now();
    const size_t stride = stride_pages * PageAlignment;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries;
    entries.reserve(backing_size / stride + 1);
    for (size_t offset = 0; offset < backing_size; offset += stride) {
        entries.emplace_back().VirtualAddress = backing_base + offset;
    }
    const size_t backing_count = entries.size();
    // Each view sample point is followed by its backing page, on the same grid of virtual
    // offsets as everywhere else so the estimate stays unbiased.
    if (impl && impl->virtual_base != nullptr) {
        for (const auto& [virtual_offset, length, backing_offset] : impl->MappedViews()) {
            sample.view_mapped_bytes += length;
            const size_t end = virtual_offset + length;
            for (size_t offset = Common::AlignUp(virtual_offset, stride); offset < end;
                 offset += stride) {
                const size_t host_offset = backing_offset + (offset - virtual_offset);
                if (host_offset >= backing_size) {
                    break;
                }
                entries.emplace_back().VirtualAddress = impl->virtual_base + offset;
                entries.emplace_back().VirtualAddress = backing_base + host_offset;
            }
        }
    }
    // In slices, so page faults on the game's threads never wait long on the working set lock.
    constexpr size_t Slice = 8192;
    for (size_t first = 0; first < entries.size(); first += Slice) {
        const size_t count = (std::min)(Slice, entries.size() - first);
        if (!K32QueryWorkingSetEx(GetCurrentProcess(), entries.data() + first,
                                  static_cast<DWORD>(count * sizeof(entries[0])))) {
            return sample;
        }
    }
    for (size_t i = 0; i < backing_count; ++i) {
        if (entries[i].VirtualAttributes.Valid) {
            sample.backing_bytes += stride;
        }
    }
    for (size_t i = backing_count; i + 1 < entries.size(); i += 2) {
        if (entries[i].VirtualAttributes.Valid) {
            sample.view_bytes += stride;
            if (entries[i + 1].VirtualAttributes.Valid) {
                sample.both_bytes += stride;
            }
        }
    }
    sample.micros = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - start)
                                         .count());
    sample.valid = true;
    std::scoped_lock lock{g_resident_mutex};
    g_resident_sample = sample;
#endif
    return sample;
}

HostMemory::ResidentSample HostMemory::LastResidentSample() {
    std::scoped_lock lock{g_resident_mutex};
    return g_resident_sample;
}

namespace {
HostMemory::ZeroTrimPhases g_zero_trim_phases{};
[[maybe_unused]] u64 MicrosSince(std::chrono::steady_clock::time_point begin) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - begin)
                                .count());
}
} // Anonymous namespace

HostMemory::ZeroTrimPhases HostMemory::LastZeroTrimPhases() noexcept {
    return g_zero_trim_phases; // written and read by the ZeroTrim thread only
}

std::pair<size_t, size_t> HostMemory::TrimZeroPages() const {
#ifdef _WIN32
    if (backing_base == nullptr) {
        return {0, 0};
    }
    ZeroTrimPhases phases{};
    auto phase_start = std::chrono::steady_clock::now();
    std::vector<ULONG_PTR> entries(size_t{1} << 22);
    for (int attempt = 0;; ++attempt) {
        if (K32QueryWorkingSet(GetCurrentProcess(), entries.data(),
                               static_cast<DWORD>(entries.size() * sizeof(ULONG_PTR)))) {
            break;
        }
        if (GetLastError() != ERROR_BAD_LENGTH || attempt == 3) {
            return {0, 0};
        }
        entries.assign((std::max)(entries.size() * 2, size_t{entries[0]} + (size_t{1} << 18)), 0);
    }
    phases.query_us = MicrosSince(phase_start);
    phase_start = std::chrono::steady_clock::now();
    const size_t count = (std::min)(size_t{entries[0]}, entries.size() - 1);
    const auto backing_begin = reinterpret_cast<uintptr_t>(backing_base);
    const auto view_begin = reinterpret_cast<uintptr_t>(impl ? impl->virtual_base : nullptr);
    const size_t view_size = impl ? impl->virtual_size : 0;
    std::vector<uintptr_t> backing_pages;
    std::vector<uintptr_t> view_pages;
    for (size_t i = 1; i <= count; ++i) {
        const uintptr_t page = entries[i] & ~uintptr_t{PageAlignment - 1};
        if (page - backing_begin < backing_size) {
            backing_pages.push_back(page);
        } else if (view_begin != 0 && page - view_begin < view_size) {
            view_pages.push_back(page);
        }
    }
    std::sort(backing_pages.begin(), backing_pages.end());
    // Pages are only ever read through the backing, which stays mapped and writable: a view page
    // can be unmapped or protected by a guest thread meanwhile. Trimming keeps the contents, so a
    // write that lands after the check costs a soft fault and nothing else.
    std::vector<uintptr_t> trim;
    size_t zero_bytes = 0;
    for (const uintptr_t page : backing_pages) {
        if (IsZeroPage(reinterpret_cast<const u8*>(page))) {
            trim.push_back(page);
            zero_bytes += PageAlignment;
        }
    }
    for (const uintptr_t page : view_pages) {
        const auto offset = impl->BackingOffset(page - view_begin);
        if (!offset || *offset >= backing_size) {
            continue;
        }
        const uintptr_t backing_page = backing_begin + *offset;
        if (IsZeroPage(reinterpret_cast<const u8*>(backing_page))) {
            trim.push_back(page);
            zero_bytes += PageAlignment;
        }
        // Reading it put the backing page in the working set: take it out again.
        if (!std::binary_search(backing_pages.begin(), backing_pages.end(), backing_page)) {
            trim.push_back(backing_page);
        }
    }
    std::sort(trim.begin(), trim.end());
    trim.erase(std::unique(trim.begin(), trim.end()), trim.end());
    phases.scan_us = MicrosSince(phase_start);
    phase_start = std::chrono::steady_clock::now();
    // A run can span two views, which VirtualUnlock refuses as a whole: then page by page.
    const auto unlock = [](uintptr_t begin, uintptr_t end) {
        if (VirtualUnlock(reinterpret_cast<void*>(begin), end - begin) ||
            GetLastError() == ERROR_NOT_LOCKED) {
            return;
        }
        for (uintptr_t page = begin; page < end; page += PageAlignment) {
            VirtualUnlock(reinterpret_cast<void*>(page), PageAlignment);
        }
    };
    for (size_t i = 0; i < trim.size();) {
        size_t j = i + 1;
        while (j < trim.size() && trim[j] == trim[j - 1] + PageAlignment) {
            ++j;
        }
        const auto run_start = std::chrono::steady_clock::now();
        unlock(trim[i], trim[j - 1] + PageAlignment);
        phases.longest_unlock_us = (std::max)(phases.longest_unlock_us, MicrosSince(run_start));
        ++phases.unlock_runs;
        i = j;
    }
    phases.unlock_us = MicrosSince(phase_start);
    g_zero_trim_phases = phases;
    return {(backing_pages.size() + view_pages.size()) * PageAlignment, zero_bytes};
#else
    return {0, 0};
#endif
}

void HostMemory::ClearBackingRegion(size_t physical_offset, size_t length, u32 fill_value) {
    if (length == 0) {
        return;
    }
    MarkUsed(physical_offset, length);
    if (fill_value != 0) {
        std::memset(backing_base + physical_offset, fill_value, length);
        MarkDirty(physical_offset, length);
        return;
    }
    // Fresh backing memory is zero and the host gives it no RAM until it's touched: clear only
    // the granules that may hold data (title 1's heap made the host commit every page of it).
    // A granule cleared in full is zero again.
#ifdef __linux__
    // Handing a whole granule back to the kernel frees its RAM; a memset would commit it.
    static const bool punch = [] {
        const char* value = std::getenv("EDEN_PUNCH_HOLE");
        return value == nullptr || value[0] != '0';
    }();
#endif
    const size_t end = physical_offset + length;
    for (size_t offset = physical_offset; offset < end;) {
        const size_t index = offset >> GranuleShift;
        const size_t granule_begin = index << GranuleShift;
        const size_t granule_end = granule_begin + (size_t{1} << GranuleShift);
        const size_t clear_end = (std::min)(granule_end, end);
        if ((granules[index].load(std::memory_order_acquire) & GranuleDirty) != 0) {
            const bool whole = offset == granule_begin && clear_end == granule_end;
            bool cleared = false;
#ifdef __linux__
            cleared = whole && punch && impl && impl->PunchHole(granule_begin, granule_end - granule_begin);
#endif
            if (!cleared) {
                std::memset(backing_base + offset, 0, clear_end - offset);
                g_zero_filled_bytes.fetch_add(clear_end - offset, std::memory_order_relaxed);
            }
            if (whole) {
                granules[index].fetch_and(static_cast<u8>(~GranuleDirty), std::memory_order_release);
            }
        }
        offset = clear_end;
    }
}

void HostMemory::MarkUsed(size_t physical_offset, size_t length) {
    if (length == 0) {
        return;
    }
    if (commit_on_use) {
        Commit(physical_offset, length);
    }
    const size_t last = (physical_offset + length - 1) >> GranuleShift;
    for (size_t index = physical_offset >> GranuleShift; index <= last; ++index) {
        if ((granules[index].load(std::memory_order_relaxed) & GranuleUsed) == 0) {
            granules[index].fetch_or(GranuleUsed, std::memory_order_release);
        }
    }
}

void HostMemory::MarkDirty(size_t physical_offset, size_t length) {
    if (length == 0) {
        return;
    }
    const size_t last = (physical_offset + length - 1) >> GranuleShift;
    for (size_t index = physical_offset >> GranuleShift; index <= last; ++index) {
        granules[index].fetch_or(GranuleDirty, std::memory_order_release);
    }
}

void HostMemory::ForgetContents() {
    for (size_t index = 0; index < num_granules; ++index) {
        if ((granules[index].load(std::memory_order_relaxed) & GranuleUsed) != 0) {
            granules[index].fetch_or(GranuleDirty, std::memory_order_release);
        }
    }
}

void HostMemory::Commit(size_t physical_offset, size_t length) {
#ifdef _WIN32
    if (!commit_on_use || length == 0 || physical_offset >= backing_size) {
        return;
    }
    const size_t last = ((std::min)(physical_offset + length, backing_size) - 1) >> GranuleShift;
    size_t index = physical_offset >> GranuleShift;
    while (index <= last) {
        if ((granules[index].load(std::memory_order_acquire) & GranuleCommitted) != 0) {
            ++index;
            continue;
        }
        // One VirtualAlloc for each run of granules not committed yet.
        size_t run_end = index + 1;
        while (run_end <= last &&
               (granules[run_end].load(std::memory_order_acquire) & GranuleCommitted) == 0) {
            ++run_end;
        }
        if (impl->CommitBacking(index << GranuleShift, run_end << GranuleShift)) {
            u64 newly = 0;
            for (size_t granule = index; granule < run_end; ++granule) {
                if ((granules[granule].fetch_or(GranuleCommitted, std::memory_order_release) &
                     GranuleCommitted) == 0) {
                    ++newly;
                }
            }
            g_committed_bytes.fetch_add(newly << GranuleShift, std::memory_order_relaxed);
        }
        index = run_end;
    }
#else
    static_cast<void>(physical_offset);
    static_cast<void>(length);
#endif
}

void HostMemory::CommitAllBut(std::span<const std::pair<size_t, size_t>> lazy) {
    if (!commit_on_use) {
        return;
    }
    std::vector<std::pair<size_t, size_t>> ranges(lazy.begin(), lazy.end());
    std::ranges::sort(ranges);
    size_t offset = 0;
    for (const auto& [begin, length] : ranges) {
        if (begin > offset) {
            Commit(offset, begin - offset);
        }
        offset = (std::max)(offset, begin + length);
    }
    if (offset < backing_size) {
        Commit(offset, backing_size - offset);
    }
#ifdef _WIN32
    LOG_INFO(HW_Memory, "EDEN_NX2_SEC_RESERVE: {} MiB of guest memory committed (the kernel's own)",
             g_committed_bytes.load(std::memory_order_relaxed) >> 20);
#endif
}

HostMemory::CommitStats HostMemory::GetCommitStats() {
#ifdef _WIN32
    return CommitStats{
        .reserved = g_reserved_views.active.load(std::memory_order_relaxed),
        .committed_bytes = g_committed_bytes.load(std::memory_order_relaxed),
        .section_bytes = g_reserved_views.backing_size,
        .net_commits = g_net_commits.load(std::memory_order_relaxed),
        .first_net_offset = g_first_net_offset.load(std::memory_order_relaxed),
    };
#else
    return {};
#endif
}

void HostMemory::EnableDirectMappedAddress() {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    if (impl) {
        impl->EnableDirectMappedAddress();
        virtual_size += reinterpret_cast<uintptr_t>(virtual_base);
    }
#endif
}

} // namespace Common
