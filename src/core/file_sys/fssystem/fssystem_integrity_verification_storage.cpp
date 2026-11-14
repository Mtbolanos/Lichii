// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "core/file_sys/fssystem/fssystem_integrity_verification_storage.h"

namespace FileSys {

constexpr inline u32 ILog2(u32 val) {
    ASSERT(val > 0);
    return static_cast<u32>((sizeof(u32) * 8) - 1 - std::countl_zero<u32>(val));
}

void IntegrityVerificationStorage::Initialize(VirtualFile hs, VirtualFile ds, s64 verif_block_size,
                                              s64 upper_layer_verif_block_size, bool is_real_data) {
    // Validate preconditions.
    ASSERT(verif_block_size >= HashSize);

    // Set storages.
    m_hash_storage = hs;
    m_data_storage = ds;

    // Set verification block sizes.
    m_verification_block_size = verif_block_size;
    m_verification_block_order = ILog2(static_cast<u32>(verif_block_size));
    ASSERT(m_verification_block_size == 1ll << m_verification_block_order);

    // Set upper layer block sizes.
    upper_layer_verif_block_size = (std::max)(upper_layer_verif_block_size, HashSize);
    m_upper_layer_verification_block_size = upper_layer_verif_block_size;
    m_upper_layer_verification_block_order = ILog2(static_cast<u32>(upper_layer_verif_block_size));
    ASSERT(m_upper_layer_verification_block_size == 1ll << m_upper_layer_verification_block_order);

    // Validate sizes.
    {
        s64 hash_size = m_hash_storage->GetSize();
        s64 data_size = m_data_storage->GetSize();
        ASSERT(((hash_size / HashSize) * m_verification_block_size) >= data_size);
    }

    // Set data.
    m_is_real_data = is_real_data;
}

void IntegrityVerificationStorage::Finalize() {
    m_hash_storage = VirtualFile();
    m_data_storage = VirtualFile();
}

size_t IntegrityVerificationStorage::Read(u8* buffer, size_t size, size_t offset) const {
    // Succeed if zero size.
    if (size == 0) {
        return size;
    }

    // Validate arguments.
    ASSERT(buffer != nullptr);

    // Validate the offset.
    const size_t data_size = m_data_storage->GetSize();
    if (offset > data_size) {
        return 0;
    }

    // Validate the access range.
    if (R_FAILED(IStorage::CheckAccessRange(
            offset, size, Common::AlignUp(data_size, static_cast<size_t>(m_verification_block_size))))) {
        return 0;
    }

    // Determine the read extents.
    size_t read_size = size;
    if (offset + read_size > data_size) {
        // Zero the requested bytes past the end of the data.
        const size_t padding_offset = data_size - offset;
        std::memset(buffer + padding_offset, 0, size - padding_offset);

        // Set the new in-bounds size.
        read_size = static_cast<size_t>(data_size - offset);
    }

    // Perform the read.
    return m_data_storage->Read(buffer, read_size, offset);
}

size_t IntegrityVerificationStorage::GetSize() const {
    return m_data_storage->GetSize();
}

} // namespace FileSys
