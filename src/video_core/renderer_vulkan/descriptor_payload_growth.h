// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace Vulkan {

enum class PayloadGrowthResult { Grown, AtLimit, AllocationFailed };

/// Grow within the configured ceiling, preserving old entries until drain() completes.
/// Allocation failure leaves storage and capacity unchanged; drained entries need no copy.
template <class Entry, class Drain, class Allocate>
PayloadGrowthResult GrowDescriptorPayload(std::unique_ptr<Entry[]>& storage, size_t& capacity,
                                          size_t maximum, size_t frames, size_t required,
                                          Drain&& drain, Allocate&& allocate) {
    if (capacity == 0 || capacity >= maximum || required > maximum || frames == 0) {
        return PayloadGrowthResult::AtLimit;
    }
    size_t next = capacity;
    do {
        next += (std::min)(next, maximum - next);
    } while (next < required && next < maximum);
    if (next == 0 || next > std::numeric_limits<size_t>::max() / frames) {
        return PayloadGrowthResult::AtLimit;
    }
    std::unique_ptr<Entry[]> replacement;
    try {
        replacement = allocate(next * frames);
    } catch (const std::bad_alloc&) {
        return PayloadGrowthResult::AllocationFailed;
    }
    if (!replacement) {
        return PayloadGrowthResult::AllocationFailed;
    }
    drain();
    storage = std::move(replacement);
    capacity = next;
    return PayloadGrowthResult::Grown;
}

} // namespace Vulkan
