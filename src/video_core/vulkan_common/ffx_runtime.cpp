// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <mutex>
#include <system_error>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/vulkan_common/ffx_runtime.h"

#ifdef _WIN32
#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <windows.h>
#include <bcrypt.h>
#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>
#endif

namespace Vulkan::FfxRuntime {

std::filesystem::path Folder() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::FidelityFxDir);
}

bool FilesPresent() {
    const std::filesystem::path folder = Folder();
    std::error_code ec;
    for (const File& file : FILES) {
        if (std::filesystem::file_size(folder / file.name, ec) != file.size || ec) {
            return false;
        }
    }
    return true;
}

#ifdef _WIN32

bool IsSignedByAmd(const std::filesystem::path& path) {
    const std::wstring file_path = std::filesystem::absolute(path).wstring();
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof(file);
    file.pcwszFilePath = file_path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    bool valid = WinVerifyTrust(nullptr, &action, &data) == ERROR_SUCCESS;
    if (valid) {
        CRYPT_PROVIDER_DATA* const provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        CRYPT_PROVIDER_SGNR* const signer =
            provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        CRYPT_PROVIDER_CERT* const cert = signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert &&
                CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name,
                                   static_cast<DWORD>(std::size(name))) > 1 &&
                std::wstring_view{name} == L"Advanced Micro Devices";
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

bool HasSha256(const std::filesystem::path& path, std::string_view sha256) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    BCRYPT_ALG_HANDLE algorithm{};
    if (!BCRYPT_SUCCESS(
            BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0))) {
        return false;
    }
    BCRYPT_HASH_HANDLE hash{};
    if (!BCRYPT_SUCCESS(BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0))) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return false;
    }
    std::vector<char> buffer(1 << 20);
    bool ok = true;
    while (ok && stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto read = static_cast<ULONG>(stream.gcount());
        ok = read == 0 || BCRYPT_SUCCESS(BCryptHashData(
                              hash, reinterpret_cast<PUCHAR>(buffer.data()), read, 0));
    }
    std::array<UCHAR, 32> digest{};
    ok = ok && stream.eof() &&
         BCRYPT_SUCCESS(BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0));
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    if (!ok || sha256.size() != digest.size() * 2) {
        return false;
    }
    std::array<char, 3> hex{};
    for (size_t i = 0; i < digest.size(); ++i) {
        std::snprintf(hex.data(), hex.size(), "%02x", digest[i]);
        if (sha256.substr(i * 2, 2) != std::string_view{hex.data(), 2}) {
            return false;
        }
    }
    return true;
}

namespace {

struct Loaded : Api {
    HMODULE module{};
};

std::mutex mutex;
std::weak_ptr<const Loaded> current;

/// Once every file checks out it isn't hashed again this session; a failed check is retried, so
/// files downloaded after a game ran are used by the next one.
bool FilesVerified() {
    static bool verified = false;
    if (verified) {
        return true;
    }
    const std::filesystem::path folder = Folder();
    for (const File& file : FILES) {
        const std::filesystem::path path = folder / file.name;
        std::error_code ec;
        const char* problem = nullptr;
        if (!std::filesystem::is_regular_file(path, ec)) {
            problem = "missing";
        } else if (std::filesystem::file_size(path, ec) != file.size ||
                   !HasSha256(path, file.sha256)) {
            problem = "not the pinned AMD FidelityFX SDK file";
        } else if (!IsSignedByAmd(path)) {
            problem = "not signed by AMD";
        }
        if (problem) {
            LOG_INFO(Render_Vulkan, "AMD FSR 4: {} is {}; using FSR 3.1",
                     Common::FS::PathToUTF8String(path), problem);
            return false;
        }
    }
    verified = true;
    return true;
}

} // Anonymous namespace

std::shared_ptr<const Api> Acquire() {
    std::scoped_lock lock{mutex};
    if (auto existing = current.lock()) {
        return existing;
    }
    if (!FilesVerified()) {
        return nullptr;
    }
    // The loader finds the upscaler next to itself.
    const std::wstring loader = (Folder() / FILES[0].name).wstring();
    const HMODULE module = LoadLibraryExW(loader.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        LOG_WARNING(Render_Vulkan, "AMD FSR 4: the FidelityFX loader doesn't load (error {})",
                    GetLastError());
        return nullptr;
    }
    auto loaded = std::make_unique<Loaded>();
    loaded->module = module;
    loaded->create_context =
        reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
    loaded->destroy_context =
        reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
    loaded->query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
    loaded->dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
    if (!loaded->create_context || !loaded->destroy_context || !loaded->query ||
        !loaded->dispatch) {
        LOG_WARNING(Render_Vulkan, "AMD FSR 4: the FidelityFX loader lacks its exports");
        FreeLibrary(module);
        return nullptr;
    }
    LOG_INFO(Render_Vulkan, "AMD FSR 4: FidelityFX SDK {} runtime loaded", SDK_VERSION);
    std::shared_ptr<const Loaded> shared{loaded.release(), [](const Loaded* l) {
                                             FreeLibrary(l->module);
                                             delete l;
                                         }};
    current = shared;
    return shared;
}

#else

bool IsSignedByAmd(const std::filesystem::path&) {
    return false;
}

bool HasSha256(const std::filesystem::path&, std::string_view) {
    return false;
}

std::shared_ptr<const Api> Acquire() {
    return nullptr;
}

#endif

} // namespace Vulkan::FfxRuntime
