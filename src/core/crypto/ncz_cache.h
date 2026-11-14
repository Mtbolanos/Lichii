// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdio>
#include <span>
#include "common/common_types.h"

namespace Core::Crypto {

// Private, exclusively created, delete-on-close scratch storage. The owner serializes access.
class NczCache {
public:
    NczCache();
    virtual ~NczCache();
    NczCache(const NczCache&) = delete;
    NczCache& operator=(const NczCache&) = delete;

    bool IsOpen() const;
    virtual bool Write(std::span<const u8> bytes, u64 offset);
    virtual std::size_t Read(std::span<u8> bytes, u64 offset);

private:
    bool Seek(u64 offset);
    std::FILE* file{};
};

} // namespace Core::Crypto
