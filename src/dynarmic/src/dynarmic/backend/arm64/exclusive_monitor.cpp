// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/interface/exclusive_monitor.h"

#include <algorithm>
#include <atomic>

#include "common/assert.h"

namespace Dynarmic {

ExclusiveMonitor::ExclusiveMonitor(std::size_t processor_count)
        : reservations(processor_count) {}

size_t ExclusiveMonitor::GetProcessorCount() const {
    return reservations.size();
}

void ExclusiveMonitor::Lock() {
    lock.Lock();
}

void ExclusiveMonitor::Unlock() {
    lock.Unlock();
}

bool ExclusiveMonitor::CheckAndClear(std::size_t processor_id, VAddr address) {
    const VAddr masked_address = address & RESERVATION_GRANULE_MASK;

#ifdef ARCHITECTURE_arm64
    std::atomic_ref<VAddr> own{reservations[processor_id].address};
    VAddr expected = masked_address;
    if (own.load(std::memory_order_relaxed) != masked_address ||
        !own.compare_exchange_strong(expected, INVALID_EXCLUSIVE_ADDRESS, std::memory_order_acq_rel)) {
        return false;
    }
    for (Reservation& reservation : reservations) {
        std::atomic_ref<VAddr> other{reservation.address};
        if (other.load(std::memory_order_relaxed) == masked_address) {
            expected = masked_address;
            other.compare_exchange_strong(expected, INVALID_EXCLUSIVE_ADDRESS,
                                          std::memory_order_acq_rel);
        }
    }
    return true;
#endif

    Lock();
    if (reservations[processor_id].address != masked_address) {
        Unlock();
        return false;
    }

    for (Reservation& other : reservations) {
        if (other.address == masked_address) {
            other.address = INVALID_EXCLUSIVE_ADDRESS;
        }
    }
    return true;
}

void ExclusiveMonitor::Clear() {
#ifdef ARCHITECTURE_arm64
    for (Reservation& other : reservations) {
        std::atomic_ref<VAddr>{other.address}.store(INVALID_EXCLUSIVE_ADDRESS, std::memory_order_release);
    }
#else
    Lock();
    for (Reservation& other : reservations) {
        other.address = INVALID_EXCLUSIVE_ADDRESS;
    }
    Unlock();
#endif
}

void ExclusiveMonitor::ClearProcessor(std::size_t processor_id) {
#ifdef ARCHITECTURE_arm64
    std::atomic_ref<VAddr>{reservations[processor_id].address}.store(INVALID_EXCLUSIVE_ADDRESS, std::memory_order_release);
#else
    Lock();
    reservations[processor_id].address = INVALID_EXCLUSIVE_ADDRESS;
    Unlock();
#endif
}

}  // namespace Dynarmic
