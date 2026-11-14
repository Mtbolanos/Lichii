// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "core/crypto/ncz_cache.h"

#ifdef __ANDROID__
#include "common/fs/path_util.h"
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include <sddl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Core::Crypto {

NczCache::NczCache() {
    std::error_code ec;
#ifdef __ANDROID__
    // Android sets no temporary directory variable, and the fallback isn't writable by apps.
    // Android's startup records the cache path without creating it.
    const auto& directory = Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir);
    std::filesystem::create_directories(directory, ec);
#else
    const auto directory = std::filesystem::temp_directory_path(ec);
#endif
    if (ec) {
        return;
    }
#ifdef _WIN32
    PSECURITY_DESCRIPTOR descriptor{};
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;OW)", SDDL_REVISION_1,
                                                              &descriptor, nullptr)) {
        return;
    }
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE};
    HANDLE handle = INVALID_HANDLE_VALUE;
    try {
        std::random_device random;
        for (unsigned attempt = 0; attempt < 16; ++attempt) {
            std::wstring name = L"lichii-ncz";
            for (unsigned part = 0; part < 4; ++part) {
                name += L"-" + std::to_wstring(random());
            }
            handle = CreateFileW((directory / name).c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
                                 0, &attributes, CREATE_NEW,
                                 FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
            if (handle != INVALID_HANDLE_VALUE || GetLastError() != ERROR_FILE_EXISTS) {
                break;
            }
        }
    } catch (const std::exception&) {
        // No insecure fallback if a private temporary file cannot be created.
    }
    LocalFree(descriptor);
    if (handle == INVALID_HANDLE_VALUE) {
        return;
    }
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_BINARY | _O_RDWR);
    if (fd == -1) {
        CloseHandle(handle);
        return;
    }
    file = _fdopen(fd, "w+b");
    if (!file) {
        _close(fd);
    }
#else
    const auto pattern = (directory / "lichii-ncz-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = mkstemp(name.data());
    if (fd == -1) {
        return;
    }
    const bool private_file =
        fchmod(fd, S_IRUSR | S_IWUSR) == 0 && fcntl(fd, F_SETFD, FD_CLOEXEC) == 0;
    const bool unlinked = unlink(name.data()) == 0;
    if (!private_file || !unlinked) {
        close(fd);
        return;
    }
    file = fdopen(fd, "w+b");
    if (!file) {
        close(fd);
    }
#endif
    if (file) {
        // Report failed disk writes immediately, not on a later buffered flush.
        if (std::setvbuf(file, nullptr, _IONBF, 0) != 0) {
            std::fclose(file);
            file = nullptr;
        }
    }
}

NczCache::~NczCache() {
    if (file) {
        std::fclose(file);
    }
}

bool NczCache::IsOpen() const {
    return file != nullptr;
}

bool NczCache::Seek(u64 offset) {
    if (!file || offset > static_cast<u64>((std::numeric_limits<s64>::max)())) {
        return false;
    }
#ifdef _WIN32
    return _fseeki64(file, static_cast<s64>(offset), SEEK_SET) == 0;
#else
    static_assert(sizeof(off_t) >= sizeof(s64));
    return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

bool NczCache::Write(std::span<const u8> bytes, u64 offset) {
    const auto max_offset = static_cast<u64>((std::numeric_limits<s64>::max)());
    return offset <= max_offset && bytes.size() <= max_offset - offset && Seek(offset) &&
           std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
}

std::size_t NczCache::Read(std::span<u8> bytes, u64 offset) {
    if (!Seek(offset)) {
        return 0;
    }
    return std::fread(bytes.data(), 1, bytes.size(), file);
}

} // namespace Core::Crypto
