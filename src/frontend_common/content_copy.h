// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <vector>
#include "core/file_sys/vfs/vfs.h"

namespace FrontendCommon {

inline bool CopyInstallFile(const FileSys::VirtualFile& source, const FileSys::VirtualFile& dest,
                            std::size_t block_size,
                            const std::function<bool(std::size_t, std::size_t)>& progress) {
    if (!source || !dest || block_size == 0)
        return false;
    const auto size = source->GetSize();
    if (!dest->Resize(size))
        return false;
    std::vector<u8> buffer(std::min<std::size_t>(block_size, 1024 * 1024));
    for (std::size_t offset = 0; offset < size; offset += buffer.size()) {
        if (progress && progress(size, offset)) {
            dest->Resize(0);
            return false;
        }
        const auto want = std::min(buffer.size(), size - offset);
        if (source->Read(buffer.data(), want, offset) != want ||
            dest->Write(buffer.data(), want, offset) != want) {
            dest->Resize(0);
            return false;
        }
    }
    return true;
}

} // namespace FrontendCommon
