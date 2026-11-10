// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <utility>
#include <vector>
#include <span>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs.h"

namespace FileSys {

/// Byte ranges [begin, end) of the patched file (the NSO with its header) a patch writes.
using PatchWriteRanges = std::vector<std::pair<u64, u64>>;

VirtualFile PatchIPS(const VirtualFile& in, const VirtualFile& ips);

/// What PatchIPS would write, without the file it patches (nothing for a patch it rejects for
/// its format).
PatchWriteRanges GetIPSWriteRanges(const VirtualFile& ips);

class IPSwitchCompiler {
public:
    explicit IPSwitchCompiler(VirtualFile patch_text);
    ~IPSwitchCompiler();

    std::array<u8, 0x20> GetBuildID() const;
    VirtualFile Apply(const VirtualFile& in) const;
    /// What Apply writes (its enabled patches).
    PatchWriteRanges GetWriteRanges() const;

private:
    struct IPSwitchPatch;

    void ParseFlag(const std::string& flag);
    void Parse(std::span<u8 const> bytes);

    VirtualFile patch_text;
    std::vector<IPSwitchPatch> patches;
    std::array<u8, 0x20> nso_build_id{};
};

} // namespace FileSys
