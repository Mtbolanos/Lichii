// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

class InvalidationAccumulator {
public:
    InvalidationAccumulator() = default;
    ~InvalidationAccumulator() = default;

    void Add(GPUVAddr address, size_t size) noexcept {
        auto const end_address = start_address + accumulated_size;
        if (!(address >= start_address && address + size <= end_address)) {
            const GPUVAddr first_block = address & atomicity_mask;
            size = ((address + size + atomicity_size_mask) & atomicity_mask) - first_block;
            address = first_block;
            if (start_address == 0) {
                start_address = address;
                accumulated_size = size;
                // will now have collected as this point
            } else if (address != end_address) {
                buffer.emplace_back(start_address, accumulated_size);
                start_address = address;
                accumulated_size = size;
            } else {
                accumulated_size += size;
            }
        }
    }

    template <typename F>
    [[nodiscard]] bool InvalidateAll(F&& f) noexcept {
        if (start_address > 0) {
            for (auto [address, size] : buffer)
                f(address, size);
            f(start_address, accumulated_size);
            buffer.clear();
            start_address = 0;
            accumulated_size = 0;
            return true;
        }
        return false;
    }

private:
    // A range runs from the start of the first 32-byte block a write touches to the end of the
    // last one. The size used to be taken before the address was aligned down, so a write past
    // its block's start got a range ending short of it (a lone word at 0x54 gave [0x40, 0x4c)).
    static constexpr size_t atomicity_bits = 5;
    static constexpr size_t atomicity_size = 1ULL << atomicity_bits;
    static constexpr size_t atomicity_size_mask = atomicity_size - 1;
    static constexpr size_t atomicity_mask = ~atomicity_size_mask;
    std::vector<std::pair<VAddr, size_t>> buffer;
    GPUVAddr start_address = 0;
    size_t accumulated_size = 0;
};

} // namespace VideoCommon
