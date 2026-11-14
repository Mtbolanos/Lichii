// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <utility>

#include "common/hex_util.h"
#include "common/logging.h"
#include <ranges>
#include "core/crypto/aes_util.h"
#include "core/crypto/ctr_encryption_layer.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/partition_filesystem.h"
#include "core/file_sys/vfs/vfs_offset.h"
#include "core/loader/loader.h"

#include "core/file_sys/fssystem/fssystem_compression_configuration.h"
#include "core/file_sys/fssystem/fssystem_crypto_configuration.h"
#include "core/file_sys/fssystem/fssystem_nca_file_system_driver.h"

namespace FileSys {

static u8 MasterKeyIdForKeyGeneration(u8 key_generation) {
    return std::max<u8>(key_generation, 1) - 1;
}

NCA::NCA(VirtualFile file_, const NCA* base_nca, bool allow_missing_base)
    : file(std::move(file_)), keys{Core::Crypto::KeyManager::Instance()} {
    if (file == nullptr) {
        status = Loader::ResultStatus::ErrorNullFile;
        return;
    }

    reader = std::make_shared<NcaReader>();
    if (Result rc = reader->Initialize(file, GetCryptoConfiguration(), GetNcaCompressionConfiguration()); R_FAILED(rc)) {
        // Without the header key every encrypted header reads as garbage, so a bad header
        // usually means the keys were never installed.
        status = keys.HasKey(Core::Crypto::S256KeyType::Header)
                     ? Loader::ResultStatus::ErrorBadNCAHeader
                     : Loader::ResultStatus::ErrorMissingHeaderKey;
        return;
    }

    // A decrypted NCA needs no keys.
    const bool decrypted = reader->GetMagic() == NcaHeader::MagicDecrypted;

    // Ensure we have the proper key area keys to continue.
    const u8 master_key_id = MasterKeyIdForKeyGeneration(reader->GetKeyGeneration());
    if (!decrypted && !keys.HasKey(Core::Crypto::S128KeyType::KeyArea, master_key_id, reader->GetKeyIndex())) {
        constexpr std::array<const char*, 3> key_area_names{"application", "ocean", "system"};
        const u8 key_index = reader->GetKeyIndex();
        LOG_ERROR(Loader,
                  "Cannot decrypt {}: key_area_key_{}_{:02x} is missing from prod.keys (key "
                  "generation {}). Keys dumped from older firmware than the game lack it.",
                  file->GetName(),
                  key_index < key_area_names.size() ? key_area_names[key_index] : "unknown",
                  master_key_id, reader->GetKeyGeneration());
        status = Loader::ResultStatus::ErrorMissingKeyAreaKey;
        return;
    }

    RightsId rights_id{};
    reader->GetRightsId(rights_id.data(), rights_id.size());
    if (!decrypted && rights_id != RightsId{}) {
        // External decryption key required; provide it here.
        u128 rights_id_u128;
        std::memcpy(rights_id_u128.data(), rights_id.data(), sizeof(rights_id));

        auto titlekey =
            keys.GetKey(Core::Crypto::S128KeyType::Titlekey, rights_id_u128[1], rights_id_u128[0]);
        if (titlekey == Core::Crypto::Key128{}) {
            LOG_ERROR(Loader, "Cannot decrypt {}: no title key for rights id {}", file->GetName(),
                      Common::HexToString(rights_id));
            status = Loader::ResultStatus::ErrorMissingTitlekey;
            return;
        }

        if (!keys.HasKey(Core::Crypto::S128KeyType::Titlekek, master_key_id)) {
            LOG_ERROR(Loader,
                      "Cannot decrypt {}: titlekek_{:02x} is missing from prod.keys (key "
                      "generation {}). Keys dumped from older firmware than the game lack it.",
                      file->GetName(), master_key_id, reader->GetKeyGeneration());
            status = Loader::ResultStatus::ErrorMissingTitlekek;
            return;
        }

        auto titlekek = keys.GetKey(Core::Crypto::S128KeyType::Titlekek, master_key_id);
        Core::Crypto::AESCipher<Core::Crypto::Key128> cipher(titlekek, Core::Crypto::Mode::ECB);
        cipher.Transcode(titlekey.data(), titlekey.size(), titlekey.data(),
                         Core::Crypto::Op::Decrypt);

        reader->SetExternalDecryptionKey(titlekey.data(), titlekey.size());
    }

    base_reader = base_nca ? base_nca->reader : nullptr;
    const s32 fs_count = reader->GetFsCount();
    for (s32 i = 0; i < fs_count; i++) {
        NcaFsHeaderReader header_reader;
        if (!reader->HasFsInfo(i) || R_FAILED(header_reader.Initialize(*reader, i))) {
            status = Loader::ResultStatus::ErrorBadNCAHeader;
            return;
        }

        auto& section = sections.emplace_back();
        section.index = i;
        section.partition = header_reader.GetFsType() == NcaFsHeader::FsType::PartitionFs;

        // Decrypted updates keep their patch tables with encryption type None.
        const auto& patch_info = header_reader.GetPatchInfo();
        if (header_reader.GetEncryptionType() == NcaFsHeader::EncryptionType::AesCtrEx ||
            (decrypted && patch_info.HasIndirectTable() && patch_info.HasAesCtrExTable())) {
            is_update = true;
        }
    }

    if (is_update && base_nca == nullptr && !allow_missing_base) {
        status = Loader::ResultStatus::ErrorMissingBKTRBaseRomFS;
    } else {
        status = Loader::ResultStatus::Success;
    }
}

NCA::~NCA() = default;

Loader::ResultStatus NCA::GetStatus() const {
    return status;
}

void NCA::OpenSection(Section& section) const {
    if (section.opened) {
        return;
    }
    section.opened = true;
    NcaFileSystemDriver fs(base_reader, reader);
    NcaFsHeaderReader header_reader;
    if (R_FAILED(fs.OpenStorage(&section.storage, &header_reader, section.index))) {
        LOG_ERROR(Loader, "Cannot open section {} of {}", section.index, file->GetName());
        section.storage = nullptr;
        return;
    }
    if (section.partition) {
        auto pfs = std::make_shared<PartitionFilesystem>(section.storage);
        if (pfs->GetStatus() == Loader::ResultStatus::Success) {
            section.partition_fs = std::move(pfs);
        }
    }
}

std::vector<VirtualFile> NCA::GetFiles() const {
    if (status != Loader::ResultStatus::Success) {
        return {};
    }
    std::scoped_lock lock{sections_mutex};
    std::vector<VirtualFile> files;
    for (auto& section : sections) {
        if (section.partition) {
            continue;
        }
        OpenSection(section);
        if (section.storage) {
            files.push_back(section.storage);
        }
    }
    return files;
}

std::vector<VirtualDir> NCA::GetSubdirectories() const {
    if (status != Loader::ResultStatus::Success) {
        return {};
    }
    std::scoped_lock lock{sections_mutex};
    std::vector<VirtualDir> dirs;
    for (auto& section : sections) {
        if (!section.partition) {
            continue;
        }
        OpenSection(section);
        if (section.partition_fs) {
            dirs.push_back(section.partition_fs);
        }
    }
    return dirs;
}

std::string NCA::GetName() const {
    return file->GetName();
}

VirtualDir NCA::GetParentDirectory() const {
    return file->GetContainingDirectory();
}

NCAContentType NCA::GetType() const {
    return static_cast<NCAContentType>(reader->GetContentType());
}

u64 NCA::GetTitleId() const {
    if (is_update) {
        return reader->GetProgramId() | 0x800;
    }
    return reader->GetProgramId();
}

RightsId NCA::GetRightsId() const {
    RightsId result;
    reader->GetRightsId(result.data(), result.size());
    return result;
}

u32 NCA::GetSDKVersion() const {
    return reader->GetSdkAddonVersion();
}

u8 NCA::GetKeyGeneration() const {
    return reader->GetKeyGeneration();
}

bool NCA::IsUpdate() const {
    return is_update;
}

VirtualFile NCA::GetRomFS() const {
    std::scoped_lock lock{sections_mutex};
    VirtualFile romfs;
    for (auto& section : sections) {
        if (section.partition) {
            continue;
        }
        OpenSection(section);
        if (section.storage) {
            romfs = section.storage;
        }
    }
    return romfs;
}

VirtualDir NCA::GetExeFS() const {
    std::scoped_lock lock{sections_mutex};
    for (auto& section : sections) {
        if (!section.partition) {
            continue;
        }
        OpenSection(section);
        if (section.partition_fs && IsDirectoryExeFS(section.partition_fs)) {
            return section.partition_fs;
        }
    }
    return nullptr;
}

VirtualFile NCA::GetBaseFile() const {
    return file;
}

VirtualDir NCA::GetLogoPartition() const {
    std::scoped_lock lock{sections_mutex};
    for (auto& section : sections) {
        if (!section.partition) {
            continue;
        }
        OpenSection(section);
        if (section.partition_fs && !IsDirectoryExeFS(section.partition_fs) &&
            IsDirectoryLogoPartition(section.partition_fs)) {
            return section.partition_fs;
        }
    }
    return nullptr;
}

} // namespace FileSys
