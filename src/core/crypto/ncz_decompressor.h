// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <string_view>

#include "core/file_sys/vfs/vfs_types.h"

namespace Core::Crypto {

class NczCache;

/**
 * Opens block or solid NCZ as an NCA VirtualFile, preserving CTR encryption. Blocks are
 * decoded on demand; solid streams use a private, delete-on-close disk cache. Solid decoding
 * has bounded working memory and may require decoding the prefix up to the requested offset.
 *
 * @param ncz_file    The raw NCZ payload (e.g. "<id>.ncz" entry from an NSZ/XCZ PFS0).
 * @param output_name Name to give the returned VirtualFile (e.g. "<id>.nca").
 *
 * @return a VirtualFile for the NCA, or nullptr for unsupported headers or cache errors.
 */
[[nodiscard]] FileSys::VirtualFile DecompressNCZ(const FileSys::VirtualFile& ncz_file,
                                                 std::string_view output_name);

// An owned backing store permits host I/O policies and deterministic cache-failure testing.
[[nodiscard]] FileSys::VirtualFile DecompressNCZ(const FileSys::VirtualFile& ncz_file,
                                                 std::string_view output_name,
                                                 std::unique_ptr<NczCache> cache);

/**
 * Returns true if the given file is an NCZ payload in block-compressed (NCZBLOCK) form,
 * Solid NCZ returns false even though playback can use the disk-backed reader.
 */
[[nodiscard]] bool IsBlockCompressedNCZ(const FileSys::VirtualFile& ncz_file);

// Header-only eligibility check; actual NCA validation and decoding still happen on reads.
[[nodiscard]] bool IsSupportedNCZ(const FileSys::VirtualFile& ncz_file);

// Installation shares the same bounded decoder and private cache as playback.
[[nodiscard]] FileSys::VirtualFile DecompressNCZForInstall(
    const FileSys::VirtualFile& ncz_file, std::string_view output_name);

} // namespace Core::Crypto
