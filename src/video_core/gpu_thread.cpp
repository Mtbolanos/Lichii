// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "common/assert.h"
#include "common/dkb_cpu_trace.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_trace/recorder.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

namespace {

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuCommand};
        const VideoCommon::StutterTrace::ThreadScope stutter_thread{
            VideoCommon::StutterTrace::Thread::Front};
        Common::SetCurrentThreadName("GPU");
        VideoCommon::GpuSplit::MarkGpuThread();
#ifdef __linux__
        // The pending-channel waits below sleep 500 us; the default timer slack adds up to 50 more.
        prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        if (Settings::values.pin_gpu_threads.GetValue()) {
            Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        }
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the waits below).
        auto last_progress = std::chrono::steady_clock::now();
        // For this long after that the thread waits in hardware, then it blocks in the OS. In
        // hardware it looks at the words the channels wait on every WordCheckPeriod and polls the
        // channels in full every PendingPollPeriod.
        constexpr std::chrono::nanoseconds PendingPoll = std::chrono::microseconds{500};
        constexpr std::chrono::nanoseconds PendingPollPeriod = std::chrono::microseconds{20};
        constexpr std::chrono::nanoseconds WordCheckPeriod = std::chrono::microseconds{2};
        // How the last wait ended, counted when a channel moves after it.
        enum class WaitEnd { None, Woken, PollTimedOut, SleepTimedOut };
        WaitEnd last_wait = WaitEnd::None;
        auto& wake = Tegra::Nx2PendingWake::Get();
        auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
        auto& acquire_stats = VideoCommon::g_acquire_stats;
        namespace Profile = VideoCommon::Nx2Profile;
        auto& split = VideoCommon::GpuSplit::Get();
        namespace BackIdleTrace = VideoCommon::BackIdleTrace;
        using FrontState = BackIdleTrace::FrontState;
        while (!stop_token.stop_requested()) {
            Profile::Report();
            split.MaybeStart(system);
            if (VideoCommon::g_acquire_counting) {
                split.ReportAcquire();
            }
            if (scheduler.HasPendingChannels()) {
                // Suspended channels wait on semaphores and list copies: a new command, a GPU
                // release or a finished copy wakes the wait below at once (Nx2PendingWake), so
                // arm it before looking.
                wake.Arm();
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                const bool popped = state.queue.TryPop(next);
                if (popped) {
                    BackIdleTrace::SetFrontState(FrontState::Parse);
                }
                if (resumed || popped) {
                    if (last_wait == WaitEnd::PollTimedOut) {
                        VideoCommon::Count(acquire_stats.front_found_polling);
                    } else if (last_wait == WaitEnd::SleepTimedOut) {
                        VideoCommon::Count(acquire_stats.front_found_late);
                    }
                    last_wait = WaitEnd::None;
                    last_progress = std::chrono::steady_clock::now();
                }
                if (!popped) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    if (BackIdleTrace::Enabled()) {
                        BackIdleTrace::SetFrontState(scheduler.WaitsForListCopy()
                                                         ? FrontState::PendingCopy
                                                         : FrontState::PendingSemaphore);
                    }
                    // Mario Kart World's channels wait for semaphores its CPU releases, stores
                    // that send no wake, and the CPU usually releases soon after: for half a
                    // millisecond wait in hardware, watching the semaphore words, then block in
                    // the OS for 500 us at a time. A full poll every 20 us catches what sends no
                    // wake and changes no word, such as a release a shader wrote to the caches.
                    // Ocarina of Time waits here ~330 times a second and its present waits for the
                    // work behind them: with 1 ms sleeps the present waited ~17 ms a frame in
                    // Kokiri village, with 50 us ~1 ms.
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    split.SettleIdleBack();
                    const auto now = std::chrono::steady_clock::now();
                    bool woken = false;
                    if (now - last_progress < PendingPoll) {
                        VideoCommon::Count(acquire_stats.front_polls);
                        last_wait = WaitEnd::PollTimedOut;
                        const auto poll_end =
                            std::min(now + PendingPollPeriod, last_progress + PendingPoll);
                        auto until = now;
                        do {
                            until = std::min(std::chrono::steady_clock::now() + WordCheckPeriod,
                                             poll_end);
                            if (wake.Poll(until)) {
                                woken = true;
                                break;
                            }
                            if (waiters.WordStored()) {
                                VideoCommon::Count(acquire_stats.front_stored);
                                last_wait = WaitEnd::Woken;
                                break;
                            }
                        } while (until < poll_end);
                    } else {
                        VideoCommon::Count(acquire_stats.front_sleeps);
                        last_wait = WaitEnd::SleepTimedOut;
                        woken = wake.Sleep(std::chrono::microseconds{500});
                    }
                    if (woken) {
                        VideoCommon::Count(acquire_stats.front_woken);
                        last_wait = WaitEnd::Woken;
                    }
                    continue;
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                BackIdleTrace::SetFrontState(FrontState::QueueEmpty);
                state.queue.PopWait(next, stop_token);
                BackIdleTrace::SetFrontState(FrontState::Parse);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (VideoCommon::GpuTrace::g_requested.load(std::memory_order_relaxed) ||
                VideoCommon::GpuTrace::g_file_trigger) {
                const auto* const submit = std::get_if<SubmitListCommand>(&next.data);
                VideoCommon::GpuTrace::OnCommand(
                    system, scheduler.HasPendingChannels(), submit ? &submit->entries : nullptr,
                    submit ? submit->channel : -1, std::holds_alternative<GPUTickCommand>(next.data));
            }
            if (split.Active()) {
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (!std::holds_alternative<SubmitListCommand>(next.data)) {
                    // Maps and flushes may change what the snapshot copier reads.
                    Tegra::DmaPusher::WaitForSnapshotCopies();
                }
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    // The guest may have written pages the caches don't watch since the lists
                    // before it: uniform keeps end here, in order on the back.
                    split.Defer([] { VideoCommon::Nx2FastDraw::NoteSync(); },
                                VideoCommon::GpuSplit::FastDrawNeutral);
                    Profile::Scope scope{Profile::Zone::Push};
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    // TickWork ends fast draw reuse itself when it runs a request.
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); },
                                VideoCommon::GpuSplit::FastDrawNeutral);
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        // GPU memory maps unmap caches (UnmapMemory downloads): the back's state.
                        split.DeferFunction(std::move(run->func));
                        split.Drain(VideoCommon::DrainSite::Map);
                    } else {
                        split.DeferFunction(std::move(run->func));
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = flush->addr, size = flush->size] {
                        rasterizer_ptr->FlushRegion(addr, size);
                    });
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                        rasterizer_ptr->OnCacheInvalidation(addr, size);
                    });
                } else {
                    ASSERT(false);
                }
                split.Defer([state_ptr = &state, fence = next.fence, block = next.block] {
                    SignalCommand(*state_ptr, fence, block);
                }, VideoCommon::GpuSplit::FastDrawNeutral);
                split.Publish();
                last_progress = std::chrono::steady_clock::now();
                continue;
            }
            // Match split Call boundaries before callbacks, flushes and invalidations, and
            // conservatively end reuse at each submitted list, including channel changes.
            VideoCommon::Nx2FastDraw::BreakUnsplit(VideoCommon::Nx2FastDraw::Reason::Call);
            if (!std::holds_alternative<SubmitListCommand>(next.data)) {
                Tegra::DmaPusher::WaitForSnapshotCopies();
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                VideoCommon::Nx2FastDraw::NoteSync();
                Profile::Scope scope{Profile::Zone::Push};
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
            last_progress = std::chrono::steady_clock::now();
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads) {
    PushCommand(RunCommand{std::move(func), before_later_reads}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);
    Tegra::Nx2PendingWake::Get().Notify();

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    // Queued snapshot copies read the channels' memory managers, which services free next.
    Tegra::DmaPusher::WaitForSnapshotCopies();
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
