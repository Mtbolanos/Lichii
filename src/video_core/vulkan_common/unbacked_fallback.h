// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

/// Images larger than this are never backed (title 1's sparse texture arrays reach 4.5G texels,
/// ~23 GB): VMA would quietly place them in system memory once VRAM is full.
constexpr u64 MaxBackedImageTexels = u64{1} << 28;

/// Whether an image no memory is left for gets an unbacked sparse image (reads return zero,
/// writes are dropped) instead of failing. Only one too large to back, or one whose caller cannot
/// do without an image: the texture cache can (TryCreateImage), and leaves the image out until
/// the collector has freed memory. An unbacked render target is drawn into, which AMD's drivers
/// do not survive, and the texture cache would take it for an image over unwritten sparse memory.
constexpr bool UsesUnbackedFallback(u64 texels, bool may_fail) {
    return texels > MaxBackedImageTexels || !may_fail;
}

} // namespace Vulkan
