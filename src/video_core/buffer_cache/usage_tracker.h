// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <span>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_backx.h"
#include "video_core/nx2_perf.h"

namespace VideoCommon {

class UsageTracker {
    // PAGE_SHIFT is a macro on FreeBSD
    static constexpr size_t BUFFER_BYTES_PER_BITSHIFT = 6;
    static constexpr size_t BUFFER_PAGE_SHIFT = 6 + BUFFER_BYTES_PER_BITSHIFT;
    static constexpr size_t BUFFER_PAGE_BYTES = 1 << BUFFER_PAGE_SHIFT;
public:
    explicit UsageTracker(size_t size) {
        const size_t num_pages = (size >> BUFFER_PAGE_SHIFT) + 1;
        pages.resize(num_pages, 0ULL);
    }

    void Reset() noexcept {
        if (!tracked) {
            return;
        }
        tracked = false;
        // Only Track sets words, and only in [set_begin, set_end): a buffer of hundreds of MiB
        // bound in part cleared all its words (8 bytes a 4 KiB page) every frame.
        std::fill(pages.begin() + static_cast<std::ptrdiff_t>(set_begin),
                  pages.begin() + static_cast<std::ptrdiff_t>(set_end), 0ULL);
        if (BackX::g_verify && std::ranges::any_of(pages, [](u64 word) { return word != 0; })) {
            BackX::Mismatch("usage tracker word set outside the tracked extent", set_begin,
                            set_end);
            std::ranges::fill(pages, 0ULL);
        }
        set_begin = 0;
        set_end = 0;
        recent.fill({~u64{0}, 0});
        full_begin = 0;
        full_end = 0;
    }

    void Track(u64 offset, u64 size) noexcept {
        // Tracking is idempotent until Reset: skip ranges tracked just before. Draws bind the
        // same (often multi-megabyte) storage buffers again and again, and each Track wrote a
        // word per 4 KiB of the range.
        for (const auto& [recent_offset, recent_size] : recent) {
            if (recent_offset == offset && recent_size == size) {
                return;
            }
        }
        tracked = true;
        recent[recent_cursor++ % recent.size()] = {offset, size};
        const size_t page = offset >> BUFFER_PAGE_SHIFT;
        const size_t page_end = (offset + size) >> BUFFER_PAGE_SHIFT;
        if (page_end < page || page_end >= pages.size()) {
            return;
        }
        // TrackPage and FillPages below set words only in [page, page_end].
        set_begin = set_end == 0 ? page : (std::min)(set_begin, page);
        set_end = (std::max)(set_end, page_end + 1);
        TrackPage(page, offset, size);
        if (page == page_end) {
            return;
        }
        FillPages(page + 1, page_end);
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        TrackPage(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
    }

    [[nodiscard]] bool IsUsed(u64 offset, u64 size) const noexcept {
        const size_t page = offset >> BUFFER_PAGE_SHIFT;
        const size_t page_end = (offset + size) >> BUFFER_PAGE_SHIFT;
        if (page_end < page || page_end >= pages.size()) {
            return false;
        }
        if (IsPageUsed(page, offset, size)) {
            return true;
        }
        for (size_t i = page + 1; i < page_end; i++) {
            if (pages[i] != 0) {
                return true;
            }
        }
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        return IsPageUsed(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
    }

    [[nodiscard]] std::span<const u64> Pages() const noexcept {
        return pages;
    }

private:
    void FillPages(size_t begin, size_t end) noexcept {
        if (begin >= end) {
            return;
        }
        // Pages in [full_begin, full_end) are all ones already (only Reset clears a page): fill
        // around that run and grow it. Keeps tracking a multi-megabyte binding O(1) once filled.
        if (end < full_begin || begin > full_end) {
            std::fill(pages.begin() + begin, pages.begin() + end, ~u64{0});
            if (end - begin > full_end - full_begin) {
                full_begin = begin;
                full_end = end;
            }
            return;
        }
        if (Nx2Perf::Verify()) {
            static std::atomic<u32> logged{};
            const size_t check_begin = (std::max)(begin, full_begin);
            const size_t check_end = (std::min)(end, full_end);
            if (!std::all_of(pages.begin() + check_begin, pages.begin() + check_end,
                             [](u64 page) { return page == ~u64{0}; }) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf draw-binds: usage pages {}..{} not all tracked",
                          check_begin, check_end);
            }
            std::fill(pages.begin() + begin, pages.begin() + end, ~u64{0});
        } else {
            if (begin < full_begin) {
                std::fill(pages.begin() + begin, pages.begin() + full_begin, ~u64{0});
            }
            if (end > full_end) {
                std::fill(pages.begin() + full_end, pages.begin() + end, ~u64{0});
            }
        }
        full_begin = (std::min)(begin, full_begin);
        full_end = (std::max)(end, full_end);
    }

    void TrackPage(u64 page, u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        const size_t offset_in_page = offset % BUFFER_PAGE_BYTES;
        const size_t first_bit = offset_in_page >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t bytes = std::min<size_t>(size, BUFFER_PAGE_BYTES - offset_in_page);
        const size_t last_bit = (offset_in_page + bytes - 1) >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t num_bits = last_bit - first_bit + 1;
        const size_t mask = ~u64{0} >> (64 - num_bits);
        pages[page] |= (~u64{0} & mask) << first_bit;
    }

    bool IsPageUsed(u64 page, u64 offset, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const size_t offset_in_page = offset % BUFFER_PAGE_BYTES;
        const size_t first_bit = offset_in_page >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t bytes = std::min<size_t>(size, BUFFER_PAGE_BYTES - offset_in_page);
        const size_t last_bit = (offset_in_page + bytes - 1) >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t num_bits = last_bit - first_bit + 1;
        const size_t mask = ~u64{0} >> (64 - num_bits);
        const size_t mask2 = (~u64{0} & mask) << first_bit;
        return (pages[page] & mask2) != 0;
    }

private:
    std::vector<u64> pages;
    /// Ranges tracked last, (offset, size).
    std::array<std::pair<u64, u64>, 4> recent{{{~u64{0}, 0}, {~u64{0}, 0}, {~u64{0}, 0}, {~u64{0}, 0}}};
    size_t recent_cursor{};
    /// Pages known to be all ones (FillPages).
    size_t full_begin{};
    size_t full_end{};
    /// The words set since the last reset lie in [set_begin, set_end) (none when set_end is 0).
    size_t set_begin{};
    size_t set_end{};
    bool tracked{};
};

} // namespace VideoCommon
