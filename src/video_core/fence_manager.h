// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/nx2_fast_draw.h"
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
        std::scoped_lock lock{uncommitted_guard};
        uncommitted_operations.emplace_back(std::move(func));
    }

    void SignalFence(std::function<void()>&& func) {
        Nx2Profile::Scope profile_scope{Nx2Profile::Zone::Fence};
        // A fence that writes a fence-flagged query report is never skipped as empty: a skipped
        // fence runs its operation at once, before the host GPU has run the work the report
        // follows, and Ocarina of Time then reads a wrong report (white flashes on leaves and walls
        // at low GPU accuracy).
        const bool delay_fence = VideoCore::g_signaling_query_report_fence ||
                                 (Settings::IsGPUFenceBehaviorDefault()
                                      ? Settings::IsGPULevelHigh()
                                      : Settings::IsGPUFenceBehaviorBalanced() ||
                                            Settings::IsGPUFenceBehaviorAccurate());
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        auto [should_flush, committed_flushes] = CommitCacheFlushes();
        if (Nx2Profile::Enabled()) {
            NoteFenceDecision(delay_fence, committed_flushes);
        }
        if (!delay_fence && committed_flushes == 0 && SkipEmptyFence()) {
            Nx2Profile::Scope skipped_scope{Nx2Profile::Zone::FenceSkipped};
            func();
            rasterizer.InvalidateGPUCache();
            return;
        }
        skipped_fences = 0;
        should_flush |= query_cache.HasUncommittedFlushes();
        query_cache.CommitAsyncFlushes();
        TFence new_fence = CreateFence(!should_flush);
        if constexpr (can_async_check) {
            guard.lock();
        }
        {
            std::scoped_lock lock{uncommitted_guard};
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(std::move(uncommitted_operations));
            if (!spare_operations.empty()) {
                uncommitted_operations = std::move(spare_operations.back());
                spare_operations.pop_back();
            } else {
                uncommitted_operations = OperationList{}; // moved-from: make it empty explicitly
                uncommitted_operations.reserve(64);
            }
        }
        pending_flushes.push_back(committed_flushes);
        QueueFence(new_fence);
        if (!delay_fence) {
            func();
        }
        fences.push(std::move(new_fence));
        if (should_flush) {
            rasterizer.FlushCommands();
        }
        if constexpr (can_async_check) {
            // The fencing thread waits for the host GPU while it has fences: it needs waking (a
            // system call) only when it waits for fences to be made.
            const bool wake = fence_thread_waiting;
            guard.unlock();
            if (wake) {
                cv.notify_all();
            }
        }
        if (delay_fence) {
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
            std::unique_lock lk(wait_mutex);
            wait_cv.wait(
                lk, [&wait_finished] { return wait_finished.load(std::memory_order_relaxed); });
        }
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
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            PopAsyncFlushes(pending_flushes.front());
            pending_flushes.pop_front();
            auto operations = std::move(pending_operations.front());
            pending_operations.pop_front();
            for (auto& operation : operations) {
                operation();
            }
            {
                std::unique_lock lock(ring_guard);
                delayed_destruction_ring.Push(std::move(current_fence));
            }
            fences.pop();
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
        };
        std::vector<Released> batch;
        batch.reserve(MaxBatch);
        const auto take_front = [&] {
            batch.push_back({std::move(fences.front()), std::move(pending_operations.front()),
                             pending_flushes.front()});
            fences.pop();
            pending_operations.pop_front();
            pending_flushes.pop_front();
        };
        while (!stop_token.stop_requested()) {
            {
                std::unique_lock lock(guard);
                cv.wait(lock, [&] {
                    fence_thread_waiting = !stop_token.stop_requested() && fences.empty();
                    return !fence_thread_waiting;
                });
                fence_thread_waiting = false;
                if (stop_token.stop_requested()) [[unlikely]] {
                    return;
                }
                take_front();
            }
            if (!batch.front().fence->IsStubbed()) {
                WaitFence(batch.front().fence);
            }
            {
                std::unique_lock lock(guard);
                while (batch.size() < MaxBatch && !fences.empty() &&
                       (fences.front()->IsStubbed() || IsFenceSignaled(fences.front()))) {
                    take_front();
                }
            }
            if (!batch.empty()) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            }
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
                query_cache.PopAsyncFlushes();
                for (auto& operation : released.operations) {
                    operation();
                }
                released.operations.clear();
            }
            if (!batch.empty()) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                for (Released& released : batch) {
                    if (spare_operations.size() < MaxSpareOperationLists &&
                        released.operations.capacity() != 0) {
                        spare_operations.push_back(std::move(released.operations));
                    }
                }
            }
            {
                std::unique_lock lock(ring_guard);
                for (Released& released : batch) {
                    delayed_destruction_ring.Push(std::move(released.fence));
                }
            }
            batch.clear();
        }
    }

    bool ShouldWait() const {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        return texture_cache.ShouldWaitAsyncFlushes() || buffer_cache.ShouldWaitAsyncFlushes() ||
               query_cache.ShouldWaitAsyncFlushes();
    }

    static constexpr u8 TextureFlushes = 1;
    static constexpr u8 BufferFlushes = 2;
    static constexpr size_t MaxSpareOperationLists = 32;

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
    void NoteFenceDecision(bool delay_fence, size_t committed_flushes) {
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
            } else if (query_cache.HasPendingFenceWork()) {
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
        LOG_INFO(HW_GPU,
                 "Fence decisions/s: {:.0f} made (delayed fences {:.0f}, committed flushes {:.0f}, "
                 "64-skip cap {:.0f}, query cache fence work {:.0f}, uncommitted operations "
                 "{:.0f}), {:.0f} skipped",
                 (counts[0] + counts[1] + counts[2] + counts[3] + counts[4]) / window,
                 counts[0] / window, counts[1] / window, counts[2] / window, counts[3] / window,
                 counts[4] / window, counts[5] / window);
        counts = {};
        since = now;
    }

    bool SkipEmptyFence() {
        if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            if (skipped_fences >= 64 || query_cache.HasPendingFenceWork()) {
                return false;
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                if (!uncommitted_operations.empty()) {
                    return false;
                }
            }
            ++skipped_fences;
            return true;
        } else {
            return false;
        }
    }

    std::queue<TFence> fences;
    // Vectors, not deques: MSVC's deque of std::function holds one element per block, so every
    // queued operation (one per query report, about 1000 a frame in busy scenes) allocated a block.
    using OperationList = std::vector<std::function<void()>>;
    OperationList uncommitted_operations;
    std::deque<OperationList> pending_operations;
    /// Lists the fencing thread ran, kept with their capacity for the next fences (a list grew
    /// from 64 operations at most fences: ~30 a fence at Ocarina of Time's water, in bursts).
    std::vector<OperationList> spare_operations; ///< Guarded by uncommitted_guard
    std::deque<u8> pending_flushes; ///< Per fence: the caches that committed downloads
    u32 skipped_fences{}; ///< Empty fences skipped since the last one made (GPU thread only)

    std::mutex guard;
    std::mutex ring_guard;
    std::mutex uncommitted_guard; ///< Guards uncommitted_operations
    std::condition_variable cv;
    bool fence_thread_waiting{}; ///< The fencing thread waits on cv for fences (guarded by guard)

    std::jthread fence_thread;

    DelayedDestructionRing<TFence, 8> delayed_destruction_ring;
};

} // namespace VideoCommon
