// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/nx2_timeline.h"
#include "common/settings.h"
#include "video_core/nx2_debug_control.h"
#include "common/thread.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

class FenceBase {
public:
    explicit FenceBase(bool is_stubbed_) : is_stubbed{is_stubbed_} {}

    bool IsStubbed() const {
        return is_stubbed;
    }

protected:
    bool is_stubbed;
};

template <typename Traits>
class FenceManager {
    using TFence = typename Traits::FenceType;
    using TTextureCache = typename Traits::TextureCacheType;
    using TBufferCache = typename Traits::BufferCacheType;
    using TQueryCache = typename Traits::QueryCacheType;
    static constexpr bool can_async_check = Traits::HAS_ASYNC_CHECK;

public:
    /// Notify the fence manager about a new frame
    void TickFrame() {
        std::unique_lock lock(ring_guard);
        delayed_destruction_ring.Tick();
    }

    // Unlike other fences, this one doesn't
    void SignalOrdering() {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.AccumulateFlushes();
    }

    void SignalReference() {
        std::function<void()> do_nothing([] {});
        SignalFence(std::move(do_nothing));
    }

    void SyncOperation(std::function<void()>&& func) {
        // Not only the GPU thread queues these (query cache work runs elsewhere too), while
        // SignalFence moves the list into pending_operations: unguarded, the list was corrupted
        // and the fencing thread crashed walking it. Its own lock, held for no callbacks.
        {
            std::scoped_lock lock{uncommitted_guard};
            uncommitted_operations.emplace_back(std::move(func));
        }
        work_epoch.fetch_add(1, std::memory_order_release);
    }

    void SignalFence(std::function<void()>&& func) {
        Nx2Profile::Scope profile_scope{Nx2Profile::Zone::Fence};
        // The accuracy can change while the game runs (settings dialog, Settings::UpdateGPUAccuracy
        // at each command list end): read once, every decision of this fence uses one value.
        // EXP33 (zortos293/Lichii#337): a fence that writes a fence-flagged query report is never
        // skipped as empty, and its operation waits for the fence: a skipped fence runs its
        // operation at once, before the host GPU has run the work the report follows, and
        // Ocarina of Time then reads a wrong report (white flashes on leaves and walls at low GPU
        // accuracy). Only set by the query cache while EXP33 is on.
        const bool delay_fence = DelayFences() || VideoCore::g_signaling_query_report_fence;
        // EDEN_NX2_EXP28 (zortos293/Lichii#717 FENCE_TRIM, FENCE_FOLD), one value per fence.
        const int fence_trims = VideoCore::Nx2Debug::Experiment(28);
        const bool trim = (fence_trims & 1) != 0;
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        auto [should_flush, committed_flushes] = CommitCacheFlushes();
        if (Nx2Profile::Enabled()) [[unlikely]] {
            NoteFenceDecision(delay_fence, committed_flushes, trim);
        }
        if (!delay_fence && committed_flushes == 0 && SkipEmptyFence(trim)) {
            Nx2Profile::Scope skipped_scope{Nx2Profile::Zone::FenceSkipped};
            func();
            rasterizer.InvalidateGPUCache();
            return;
        }
        skipped_fences = 0;
        should_flush |= query_cache.HasUncommittedFlushes();
        query_cache.CommitAsyncFlushes();
        if constexpr (can_async_check) {
            // EXP28 & 2: a fence with nothing to wait for on the host GPU carries only queued
            // work (FoldStubbedFence).
            if ((fence_trims & 2) != 0 && !should_flush && committed_flushes == 0) {
                // As the made fence below: its callback runs now, before its queued work.
                if (!delay_fence) {
                    func();
                }
                FoldStubbedFence(delay_fence, func);
                if (delay_fence) {
                    SubmitDeferredWork();
                }
                rasterizer.InvalidateGPUCache();
                return;
            }
        }
        TFence new_fence = CreateFence(!should_flush);
        if constexpr (can_async_check) {
            guard.lock();
        }
        {
            std::scoped_lock lock{uncommitted_guard};
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(TakeUncommittedOperations());
        }
        pending_flushes.push_back(committed_flushes);
        if constexpr (can_async_check) {
            pending_query_pops.push_back(1);
        }
        QueueFence(new_fence);
        if (!delay_fence) {
            func();
        }
        fences.push(std::move(new_fence));
        fences_in_flight.fetch_add(1, std::memory_order_relaxed);
        work_epoch.fetch_add(1, std::memory_order_release);
        if (should_flush) {
            rasterizer.FlushCommands();
        }
        if constexpr (can_async_check) {
            guard.unlock();
            cv.notify_all();
        }
        if (delay_fence) {
            // Its callback runs on the fencing thread once the fence's work is done, and the
            // guest or WaitPendingFences may block on that callback: a submission left for
            // later (Scheduler::DeferFlush) would wait for the back to go idle, which it then
            // never does (GPU accuracy switched to High while the game ran, 24 s and more).
            SubmitDeferredWork();
        }
        rasterizer.InvalidateGPUCache();
    }

    void SignalSyncPoint(u32 value) {
        // The id comes from a command list (the 3D engine's sync method, the puller's fence
        // action). Past the syncpoint table, the increment would walk memory after it as an
        // action list; command words read wrong carry any id (Splatoon Raiders on Linux).
        if (value >= Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS) {
            LOG_ERROR(HW_GPU, "Signal of syncpoint {} past the table, ignored", value);
            return;
        }
        // The render-frame clock (video_core/nx2.h), before the increment: a flip its actions
        // wake must already see the bump.
        if (Tegra::g_nx2_render_clock_wanted.load(std::memory_order_relaxed)) {
            Tegra::Nx2NoteSyncpointSignal(
                value, static_cast<u32>(syncpoint_manager.GetGuestSyncpointValue(value)) + 1);
        }
        syncpoint_manager.IncrementGuest(value);
        std::function<void()> func([this, value] { syncpoint_manager.IncrementHost(value); });
        SignalFence(std::move(func));
    }

    void WaitPendingFences([[maybe_unused]] bool force) {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<true>();
        } else {
            if (!force) {
                return;
            }
            std::mutex wait_mutex;
            std::condition_variable wait_cv;
            std::atomic<bool> wait_finished{};
            std::function<void()> func([&] {
                std::scoped_lock lk(wait_mutex);
                wait_finished.store(true, std::memory_order_relaxed);
                wait_cv.notify_all();
            });
            SignalFence(std::move(func));
            // The fence's commands can be held back by a deferred flush that only this thread
            // submits. The fencing thread waits for that submission before releasing the fence,
            // so waiting here without submitting deadlocks (Ocarina of Time entering gameplay).
            SubmitDeferredWork();
            BackWaitScope wait_scope{"FenceManager::WaitPendingFences (fencing thread callback)"};
            std::unique_lock lk(wait_mutex);
            wait_cv.wait(
                lk, [&wait_finished] { return wait_finished.load(std::memory_order_relaxed); });
        }
    }

    /// GPU thread (the split's back between entries): whether WaitPendingFences(true) would only
    /// invalidate the GPU cache, as no download, query, fence operation or (`strict`, or with
    /// delayed fences) pending fence is left for it. A lock held elsewhere counts as work.
    /// `epoch`: FenceWorkEpoch() as of the check.
    bool FencesQuiet(bool strict, u64& epoch) {
        epoch = work_epoch.load(std::memory_order_acquire);
        if ((strict || DelayFences()) && fences_in_flight.load(std::memory_order_acquire) != 0) {
            return false;
        }
        {
            std::unique_lock lock{uncommitted_guard, std::try_to_lock};
            if (!lock || !uncommitted_operations.empty()) {
                return false;
            }
        }
        {
            std::unique_lock lock{texture_cache.mutex, std::try_to_lock};
            if (!lock || texture_cache.HasAsyncFlushWork()) {
                return false;
            }
        }
        {
            std::unique_lock lock{buffer_cache.mutex, std::try_to_lock};
            if (!lock || buffer_cache.HasUncommittedFlushes()) {
                return false;
            }
        }
        if (query_cache.HasUncommittedFlushes()) {
            return false;
        }
        if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            if (query_cache.HasPendingFenceWork()) {
                return false;
            }
        }
        return true;
    }

    /// Any thread: changes whenever a fence or a fence operation is added.
    u64 FenceWorkEpoch() const {
        return work_epoch.load(std::memory_order_acquire);
    }

protected:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                          TTextureCache& texture_cache_, TBufferCache& buffer_cache_,
                          TQueryCache& query_cache_)
        : rasterizer{rasterizer_}, gpu{gpu_}, syncpoint_manager{gpu.Host1x().GetSyncpointManager()},
          texture_cache{texture_cache_}, buffer_cache{buffer_cache_}, query_cache{query_cache_} {
        if constexpr (can_async_check) {
            fence_thread =
                std::jthread([this](std::stop_token token) { ReleaseThreadFunc(token); });
        }
    }

    virtual ~FenceManager() {
        if constexpr (can_async_check) {
            fence_thread.request_stop();
            cv.notify_all();
            fence_thread.join();
        }
    }

    /// Creates a Fence Interface, does not create a backend fence if 'is_stubbed' is
    /// true
    virtual TFence CreateFence(bool is_stubbed) = 0;
    /// Queues a fence into the backend if the fence isn't stubbed.
    virtual void QueueFence(TFence& fence) = 0;
    /// Notifies that the backend fence has been signaled/reached in host GPU.
    virtual bool IsFenceSignaled(TFence& fence) const = 0;
    /// Waits until a fence has been signalled by the host GPU.
    virtual void WaitFence(TFence& fence) = 0;
    /// Submits work whose submission the backend deferred (called before blocking on a fence).
    virtual void SubmitDeferredWork() {}

    VideoCore::RasterizerInterface& rasterizer;
    Tegra::GPU& gpu;
    Tegra::Host1x::SyncpointManager& syncpoint_manager;
    TTextureCache& texture_cache;
    TBufferCache& buffer_cache;
    TQueryCache& query_cache;

private:
    /// Vectors, not deques: MSVC's deque of std::function allocated a block per operation.
    using OperationList = std::vector<std::function<void()>>;
    static constexpr size_t MaxSpareOperationLists = 32;

    template <bool force_wait>
    void TryReleasePendingFences() {
        while (!fences.empty()) {
            TFence& current_fence = fences.front();
            if (ShouldWait() && !IsFenceSignaled(current_fence)) {
                if constexpr (force_wait) {
                    WaitFence(current_fence);
                } else {
                    return;
                }
            }
            PopAsyncFlushes(pending_flushes.front());
            pending_flushes.pop_front();
            auto operations = std::move(pending_operations.front());
            pending_operations.pop_front();
            for (auto& operation : operations) {
                operation();
            }
            operations.clear();
            {
                std::scoped_lock lock{uncommitted_guard};
                KeepSpareOperations(operations);
            }
            {
                std::unique_lock lock(ring_guard);
                delayed_destruction_ring.Push(std::move(current_fence));
            }
            fences.pop();
            fences_in_flight.fetch_sub(1, std::memory_order_release);
        }
    }

    void ReleaseThreadFunc(std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPUFencingThread");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        Common::SetCurrentThreadToPerformanceCores();

        // Every fence the host GPU has already passed is released together: Switch 2 titles
        // signal thousands a second and the host GPU completes them in bursts, so each cache's
        // lock is taken once per burst instead of once per fence (the GPU thread, which holds
        // them for nearly every command, spun, slept and woke this thread for each one).
        // EDEN_NX2_FENCE_BATCH=0: one fence at a time (for comparisons).
        static const size_t MaxBatch = [] {
            const char* value = std::getenv("EDEN_NX2_FENCE_BATCH");
            return value != nullptr && value[0] == '0' ? size_t{1} : size_t{64};
        }();
        struct Released {
            TFence fence;
            OperationList operations;
            u8 flushes;
            u32 query_pops; ///< Query cache entries: one per fence, folded ones included
        };
        std::vector<Released> batch;
        batch.reserve(MaxBatch);
        const auto take_front = [&] {
            batch.push_back({std::move(fences.front()), std::move(pending_operations.front()),
                             pending_flushes.front(), pending_query_pops.front()});
            fences.pop();
            pending_operations.pop_front();
            pending_flushes.pop_front();
            pending_query_pops.pop_front();
        };
        while (!stop_token.stop_requested()) {
            {
                std::unique_lock lock(guard);
                // Work the GPU thread runs inline (FoldStubbedFence) goes before any fence
                // queued meanwhile.
                {
                    // Fences come from SignalFence (the split's back).
                    Common::Nx2Timeline::Span wait_span{"fence: wait for work"};
                    cv.wait(lock, [&] {
                        return stop_token.stop_requested() || (!fences.empty() && !inline_running);
                    });
                }
                if (stop_token.stop_requested()) [[unlikely]] {
                    return;
                }
                take_front();
                fence_thread_busy = true;
            }
            if (!batch.front().fence->IsStubbed()) {
                // The host GPU reaching the fence's submission.
                Common::Nx2Timeline::Span wait_span{"fence: wait for fence"};
                WaitFence(batch.front().fence);
            }
            // The batch's landings, waiter latches and operations (host syncpoint increments,
            // which signal the guest's GPU events): the host GPU -> guest hand-off.
            Common::Nx2Timeline::ArgSpan release_span{"fence: release"};
            {
                std::unique_lock lock(guard);
                while (batch.size() < MaxBatch && !fences.empty() &&
                       (fences.front()->IsStubbed() || IsFenceSignaled(fences.front()))) {
                    take_front();
                }
            }
            release_span.Set(0, {"fences", batch.size()});
            size_t texture_pops = 0;
            size_t buffer_pops = 0;
            for (const Released& released : batch) {
                texture_pops += (released.flushes & TextureFlushes) != 0 ? 1 : 0;
                buffer_pops += (released.flushes & BufferFlushes) != 0 ? 1 : 0;
            }
            if (texture_pops != 0) {
                std::scoped_lock lock{texture_cache.mutex};
                for (size_t i = 0; i < texture_pops; ++i) {
                    texture_cache.PopAsyncFlushes();
                }
            }
            if (buffer_pops != 0) {
                std::scoped_lock lock{buffer_cache.mutex};
                for (size_t i = 0; i < buffer_pops; ++i) {
                    buffer_cache.PopAsyncFlushes();
                }
            }
            // Queries and operations keep their per-fence order.
            for (Released& released : batch) {
                for (u32 i = 0; i < released.query_pops; ++i) {
                    query_cache.PopAsyncFlushes();
                }
                for (auto& operation : released.operations) {
                    operation();
                }
                released.operations.clear();
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                for (Released& released : batch) {
                    KeepSpareOperations(released.operations);
                }
            }
            {
                std::unique_lock lock(ring_guard);
                for (Released& released : batch) {
                    delayed_destruction_ring.Push(std::move(released.fence));
                }
            }
            // Only once their operations ran: FencesQuiet must not see a fence as gone before.
            fences_in_flight.fetch_sub(static_cast<u32>(batch.size()), std::memory_order_release);
            batch.clear();
            {
                std::scoped_lock lock(guard);
                fence_thread_busy = false;
            }
        }
    }

    /// EDEN_NX2_EXP28 & 2 (zortos293/Lichii#717 FENCE_FOLD): a fence with nothing to wait for on
    /// the host GPU (no downloads, no query results) carries only queued work: the query cache's
    /// slot cleanup, operations, and with delayed fences its callback (a syncpoint's host
    /// increment). The fencing thread runs that work as soon as every fence before it is
    /// released, so instead of a fence of its own (an allocation, a queue entry and a wake of the
    /// fencing thread):
    /// - with fences queued, the work joins the newest one, released with it in order;
    /// - with nothing queued or being released, the split's back runs it at once (same order);
    /// - otherwise (the fencing thread is releasing fences it took) it gets a fence as before.
    void FoldStubbedFence(bool delay_fence, std::function<void()>& func) {
        std::unique_lock lock{guard};
        if (!fences.empty()) {
            {
                std::scoped_lock ops_lock{uncommitted_guard};
                auto& operations = pending_operations.back();
                for (auto& operation : uncommitted_operations) {
                    operations.emplace_back(std::move(operation));
                }
                uncommitted_operations.clear();
                if (delay_fence) {
                    operations.emplace_back(std::move(func));
                }
            }
            // This fence's query cache entry (no queries: CommitAsyncFlushes found none).
            ++pending_query_pops.back();
            work_epoch.fetch_add(1, std::memory_order_release);
            fold_counts[0].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (!fence_thread_busy && !inline_running && GpuSplit::OnBackThread()) {
            inline_running = true;
            OperationList operations;
            {
                std::scoped_lock ops_lock{uncommitted_guard};
                operations = TakeUncommittedOperations();
            }
            lock.unlock();
            query_cache.PopAsyncFlushes();
            for (auto& operation : operations) {
                operation();
            }
            if (delay_fence) {
                func();
            }
            operations.clear();
            {
                std::scoped_lock ops_lock{uncommitted_guard};
                KeepSpareOperations(operations);
            }
            lock.lock();
            inline_running = false;
            const bool queued = !fences.empty();
            work_epoch.fetch_add(1, std::memory_order_release);
            lock.unlock();
            if (queued) {
                cv.notify_all();
            }
            fold_counts[1].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        TFence new_fence = CreateFence(true);
        {
            std::scoped_lock ops_lock{uncommitted_guard};
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(TakeUncommittedOperations());
        }
        pending_flushes.push_back(0);
        pending_query_pops.push_back(1);
        QueueFence(new_fence);
        fences.push(std::move(new_fence));
        fences_in_flight.fetch_add(1, std::memory_order_relaxed);
        work_epoch.fetch_add(1, std::memory_order_release);
        lock.unlock();
        cv.notify_all();
        fold_counts[2].fetch_add(1, std::memory_order_relaxed);
    }

    /// Caller holds uncommitted_guard. Hands the uncommitted list over and puts a spare list in
    /// its place, keeping its capacity (else a new one reserved for 64): a list that started
    /// empty grew again at every fence, a report per draw adding one operation each.
    OperationList TakeUncommittedOperations() {
        OperationList taken = std::move(uncommitted_operations);
        if (!spare_operations.empty()) {
            uncommitted_operations = std::move(spare_operations.back());
            spare_operations.pop_back();
        } else {
            uncommitted_operations = OperationList{}; // moved-from: make it empty explicitly
            uncommitted_operations.reserve(64);
        }
        return taken;
    }

    /// Caller holds uncommitted_guard. `list`, its operations run and cleared (outside the lock:
    /// destroying one can run code), is kept for a later fence while few are spare.
    void KeepSpareOperations(OperationList& list) {
        if (spare_operations.size() < MaxSpareOperationLists && list.capacity() != 0) {
            spare_operations.push_back(std::move(list));
        }
    }

    /// The accuracy can change while the game runs (settings dialog, Settings::UpdateGPUAccuracy
    /// at each command list end): callers read it once per decision.
    static bool DelayFences() {
        return VideoCore::Nx2Debug::DelayedFences();
    }

    bool ShouldWait() const {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        return texture_cache.ShouldWaitAsyncFlushes() || buffer_cache.ShouldWaitAsyncFlushes() ||
               query_cache.ShouldWaitAsyncFlushes();
    }

    static constexpr u8 TextureFlushes = 1;
    static constexpr u8 BufferFlushes = 2;

    /// committed: what CommitAsyncFlushes returned for this fence.
    void PopAsyncFlushes(u8 committed) {
        if (committed & TextureFlushes) {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.PopAsyncFlushes();
        }
        if (committed & BufferFlushes) {
            std::scoped_lock lock{buffer_cache.mutex};
            buffer_cache.PopAsyncFlushes();
        }
        query_cache.PopAsyncFlushes();
    }

    struct CacheFlushes {
        bool should_flush;  ///< Downloads to wait for: the fence must reach the host GPU
        u8 committed;       ///< TextureFlushes, BufferFlushes: the caches that committed downloads
    };

    /// Commits the texture and buffer caches' downloads for a new fence (the query cache's are
    /// committed once a fence is made). Switch 2 titles signal a fence per compute dispatch
    /// (thousands a second), mostly with nothing or only buffer data to download: each cache's
    /// lock is taken once, alone, and the fencing thread later takes only the lock of the cache
    /// with work, instead of holding both while it copies (the GPU thread's draws and uploads
    /// waited for it).
    CacheFlushes CommitCacheFlushes() {
        CacheFlushes flushes{};
        {
            std::scoped_lock lock{texture_cache.mutex};
            flushes.should_flush = texture_cache.HasUncommittedFlushes();
            if (texture_cache.HasAsyncFlushWork()) {
                texture_cache.CommitAsyncFlushes();
                flushes.committed |= TextureFlushes;
            }
        }
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (buffer_cache.HasUncommittedFlushes()) {
                buffer_cache.CommitAsyncFlushes();
                flushes.should_flush = true;
                flushes.committed |= BufferFlushes;
            }
        }
        return flushes;
    }

    /// Whether a fence with no downloads would do nothing, so need not be made: with fast GPU
    /// fences its callback runs at once anyway, and most of the per-dispatch fences of Switch 2
    /// titles have no queries or queued operations either. Making, queueing and retiring them
    /// cost the GPU thread ~5% and woke the fencing thread for each. One in 64 is still made, so
    /// the operations fences carry along (query slot cleanup) keep running.
    // With the GPU profile on: why fences are made rather than skipped, logged every 5 s.
    // Called before SkipEmptyFence, which it mirrors without changing its state.
    void NoteFenceDecision(bool delay_fence, size_t committed_flushes, bool trim) {
        static std::array<u64, 6> counts{};
        static auto since = std::chrono::steady_clock::now();
        size_t reason = 5; // would be skipped
        if (delay_fence) {
            reason = 0;
        } else if (committed_flushes != 0) {
            reason = 1;
        } else if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            if (skipped_fences >= 64) {
                reason = 2;
            } else if (query_cache.HasPendingFenceWork(trim)) {
                reason = 3;
            } else {
                std::scoped_lock lock{uncommitted_guard};
                reason = uncommitted_operations.empty() ? 5 : 4;
            }
        } else {
            reason = 2;
        }
        ++counts[reason];
        const auto now = std::chrono::steady_clock::now();
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        // EXP28 & 2: what became of the made fences that had nothing to wait for.
        std::array<double, 3> folds{};
        for (size_t i = 0; i < folds.size(); ++i) {
            folds[i] =
                static_cast<double>(fold_counts[i].exchange(0, std::memory_order_relaxed)) / window;
        }
        LOG_INFO(HW_GPU,
                 "Fence decisions/s: {:.0f} made (delayed fences {:.0f}, committed flushes {:.0f}, "
                 "64-skip cap {:.0f}, query cache fence work {:.0f}, uncommitted operations "
                 "{:.0f}), {:.0f} skipped; EXP28 trim {}, fold: {:.0f} joined a queued fence, "
                 "{:.0f} run inline, {:.0f} fenced",
                 (counts[0] + counts[1] + counts[2] + counts[3] + counts[4]) / window,
                 counts[0] / window, counts[1] / window, counts[2] / window, counts[3] / window,
                 counts[4] / window, counts[5] / window, trim ? 1 : 0, folds[0], folds[1],
                 folds[2]);
        counts = {};
        since = now;
    }

    bool SkipEmptyFence(bool trim) {
        if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            // EDEN_NX2_EXP28 & 1 (zortos293/Lichii#717 FENCE_TRIM): payload releases (QMD
            // releases, ~4,300 a second in Ocarina) wrote guest memory at once anyway; their host
            // buffer syncs (a render pass end and two barriers each) go with the next fence made
            // or wait for idle.
            if (skipped_fences >= 64 || query_cache.HasPendingFenceWork(trim)) {
                return false;
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                if (!uncommitted_operations.empty()) {
                    return false;
                }
            }
            if (trim) {
                query_cache.NotePayloadSyncDeferred();
            }
            ++skipped_fences;
            return true;
        } else {
            return false;
        }
    }

    std::queue<TFence> fences;
    OperationList uncommitted_operations;
    std::deque<OperationList> pending_operations;
    /// Lists whose operations ran, kept with their capacity for the next fences.
    std::vector<OperationList> spare_operations; ///< Guarded by uncommitted_guard
    std::deque<u8> pending_flushes; ///< Per fence: the caches that committed downloads
    std::deque<u32> pending_query_pops; ///< Async check, per fence: query cache entries to pop
    bool fence_thread_busy{}; ///< Guarded by guard: the fencing thread releases fences it took
    bool inline_running{};    ///< Guarded by guard: FoldStubbedFence runs a fence's work inline
    /// EXP28 & 2 (FoldStubbedFence): fences joined to a queued one, run inline, made anyway.
    std::array<std::atomic<u64>, 3> fold_counts{};
    u32 skipped_fences{}; ///< Empty fences skipped since the last one made (GPU thread only)
    std::atomic<u32> fences_in_flight{}; ///< Fences made whose operations have not run yet
    std::atomic<u64> work_epoch{};       ///< Bumped by every fence and fence operation added

    std::mutex guard;
    std::mutex ring_guard;
    std::mutex uncommitted_guard; ///< Guards uncommitted_operations
    std::condition_variable cv;

    std::jthread fence_thread;

    DelayedDestructionRing<TFence, 8> delayed_destruction_ring;
};

} // namespace VideoCommon
