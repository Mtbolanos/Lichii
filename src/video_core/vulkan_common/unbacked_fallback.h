// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

/// Images over both limits are never backed (title 1's sparse texture arrays reach 4.5G texels,
/// ~23 GB): VMA would quietly place them in system memory once VRAM is full. Texels alone also
/// caught Metroid Prime 4's 1408x256x1008 BC1 and BC6H light volumes (173 and 346 MiB), which
/// then sampled as zero and drew the world black.
constexpr u64 MaxBackedImageTexels = u64{1} << 28;
constexpr u64 MaxBackedImageBytes = u64{1} << 30;

constexpr bool IsTooLargeToBack(u64 texels, u64 bytes) {
    return texels > MaxBackedImageTexels && bytes > MaxBackedImageBytes;
}

/// Whether an image no memory is left for gets an unbacked sparse image (reads return zero,
/// writes are dropped) instead of failing. Only one too large to back, or one whose caller cannot
/// do without an image: the texture cache can (TryCreateImage), and leaves the image out until
/// the collector has freed memory. An unbacked render target is drawn into, which AMD's drivers
/// do not survive, and the texture cache would take it for an image over unwritten sparse memory.
constexpr bool UsesUnbackedFallback(u64 texels, u64 bytes, bool may_fail) {
    return IsTooLargeToBack(texels, bytes) || !may_fail;
}

} // namespace Vulkan
