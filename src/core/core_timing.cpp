// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "common/cpu_features.h"
#include "common/cpu_features.h"
#include "common/logging.h"
#include "common/thread_cpu_clock.h"

#ifdef _WIN32
#include "common/windows/timer_resolution.h"
#endif

#include "common/settings.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"
#include "core/host_wait_margin.h"

namespace Core::Timing {

constexpr s64 MAX_SLICE_LENGTH = 10000;

constexpr u32 CPU_CLOCK_BASE_MHZ = 1020;
constexpr u32 CPU_CLOCK_BOOST_MHZ = 1734;
constexpr u32 CPU_CLOCK_OVERCLOCK_MHZ = 2040;

constexpr u32 CpuClockTargetMhz(Settings::CpuClock clock) {
    switch (clock) {
    case Settings::CpuClock::Boost:
        return CPU_CLOCK_BOOST_MHZ;
    case Settings::CpuClock::Overclock:
        return CPU_CLOCK_OVERCLOCK_MHZ;
    default:
        return CPU_CLOCK_BASE_MHZ;
    }
}

namespace {

// One second of the timer thread's waits, for EDEN_NX2_HOSTTIMING_STATS.
struct HostWaitWindow {
    s64 start{};
    std::optional<u64> cpu_start;
    u64 waits{};
    u64 woken{};
    s64 sleep_ns{};
    s64 precise_ns{};
    std::vector<s64> wake_late;
    std::vector<s64> event_late;
    std::vector<s64> one_shot_late;
};

thread_local HostWaitWindow t_host_wait;
thread_local HostWaitMargin t_host_wait_margin;

void LogHostWait(s64 now) {
    HostWaitWindow& w = t_host_wait;
    const auto percentiles = [](std::vector<s64>& late) -> std::array<s64, 3> {
        if (late.empty()) {
            return {};
        }
        std::sort(late.begin(), late.end());
        return {late[late.size() / 2] / 1000, late[late.size() * 99 / 100] / 1000,
                late.back() / 1000};
    };
    const auto wake = percentiles(w.wake_late);
    const auto event = percentiles(w.event_late);
    const auto one_shot = percentiles(w.one_shot_late);
    const f64 seconds = static_cast<f64>(now - w.start) / 1e9;
    const std::optional<u64> cpu = Common::CurrentThreadCpuTimeNs();
    LOG_INFO(Core_Timing,
             "HostTiming wait: {} waits, {} woken by new events, OS sleep {:.1f} ms/s, "
             "precise wait {:.1f} ms/s, margin {} us; wake late p50/p99/max {}/{}/{} us; {} events "
             "late p50/p99/max {}/{}/{} us, {} of them one-shot p50/p99/max {}/{}/{} us; thread "
             "CPU {}",
             w.waits, w.woken, static_cast<f64>(w.sleep_ns) / 1e6 / seconds,
             static_cast<f64>(w.precise_ns) / 1e6 / seconds,
             t_host_wait_margin.Estimate() / 1000, wake[0], wake[1], wake[2], w.event_late.size(),
             event[0], event[1], event[2], w.one_shot_late.size(), one_shot[0], one_shot[1],
             one_shot[2],
             cpu && w.cpu_start
                 ? fmt::format("{:.1f}%", static_cast<f64>(*cpu - *w.cpu_start) / 1e7 / seconds)
                 : std::string{"unknown"});
    w.start = now;
    w.cpu_start = cpu;
    w.waits = w.woken = 0;
    w.sleep_ns = w.precise_ns = 0;
    w.wake_late.clear();
    w.event_late.clear();
    w.one_shot_late.clear();
}

} // Anonymous namespace

std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback) {
    return std::make_shared<EventType>(std::move(callback), std::move(name));
}

struct CoreTiming::Event {
    s64 time;
    u64 fifo_order;
    std::weak_ptr<EventType> type;
    s64 reschedule_time;
    heap_t::handle_type handle{};

    // Sort by time, unless the times are the same, in which case sort by
    // the order added to the queue
    friend bool operator>(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) > std::tie(right.time, right.fifo_order);
    }

    friend bool operator<(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) < std::tie(right.time, right.fifo_order);
    }
};

CoreTiming::CoreTiming() = default;
CoreTiming::~CoreTiming() {
    Reset();
}

void CoreTiming::Initialize(std::function<void()>&& on_thread_init_) {
    Reset();
    on_thread_init = std::move(on_thread_init_);
    event_fifo_id = 0;
    cpu_ticks = 0;
    if (const char* stats = std::getenv("EDEN_NX2_HOSTTIMING_STATS")) {
        host_wait_stats = std::strtol(stats, nullptr, 10) != 0;
    }
    if (is_multicore) {
        timer_thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("HostTiming");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::VeryHigh);
            Common::SetCurrentThreadToPerformanceCores();
            on_thread_init();
            has_started = true;

            // base frequency in MHz: 1ns (10^-9) = 1GHz (10^9)
            while (!stop_token.stop_requested()) {
                while (!paused && !stop_token.stop_requested()) {
                    paused_set = false;
                    if (const NextEvents next = AdvanceEvents(); next.next) {
                        // There are more events left in the queue, wait until the next event.
                        WaitForEvent(*next.next, next.one_shot);
                    } else {
                        // Queue is empty, wait until another event is scheduled and signals us to
                        // continue.
                        wait_set = true;
                        event.Wait();
                    }
                    wait_set = false;
                }
                paused_set = true;
                pause_event.Wait();
            }
        });
    }
}

void CoreTiming::WaitForEvent(s64 next_time, std::optional<s64> one_shot_time) {
    const s64 start = GetGlobalTimeNs().count();
    if (next_time <= start) {
        return;
    }
    const s64 margin = t_host_wait_margin.Get(start);
    const bool precise = one_shot_time && *one_shot_time - margin <= start;
    const s64 wake = one_shot_time ? (std::min)(next_time, *one_shot_time - margin) : next_time;
    const bool signalled = precise ? event.SpinFor(std::chrono::nanoseconds{next_time - start})
                                   : event.SleepFor(std::chrono::nanoseconds{wake - start});
    const s64 end = GetGlobalTimeNs().count();
    if (precise) {
        t_host_wait_margin.OnPreciseWait(end - start);
    } else if (!signalled) {
        t_host_wait_margin.OnOversleep(end - wake);
    }
    if (!host_wait_stats.load(std::memory_order_relaxed)) {
        return;
    }
    HostWaitWindow& w = t_host_wait;
    ++w.waits;
    w.woken += signalled;
    (precise ? w.precise_ns : w.sleep_ns) += end - start;
    if (!precise && !signalled) {
        w.wake_late.push_back(end - wake);
    }
    if (w.start == 0) {
        w.start = end;
        w.cpu_start = Common::CurrentThreadCpuTimeNs();
    } else if (end - w.start >= 1'000'000'000) {
        LogHostWait(end);
    }
}

void CoreTiming::ClearPendingEvents() {
    std::scoped_lock lock{advance_lock, basic_lock};
    event_queue.clear();
    event.Set();
}

void CoreTiming::Pause(bool is_paused) {
    paused = is_paused;
    pause_event.Set();

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

void CoreTiming::SyncPause(bool is_paused) {
    if (is_paused == paused && paused_set == paused) {
        return;
    }

    Pause(is_paused);
    if (timer_thread.joinable()) {
        if (!is_paused) {
            pause_event.Set();
        }
        event.Set();
        while (paused_set != is_paused)
            ;
    }

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

bool CoreTiming::IsRunning() const {
    return !paused_set;
}

bool CoreTiming::HasPendingEvents() const {
    std::scoped_lock lock{basic_lock};
    return !(wait_set && event_queue.empty());
}

void CoreTiming::ScheduleEvent(std::chrono::nanoseconds ns_into_future,
                               const std::shared_ptr<EventType>& event_type, bool absolute_time) {
    {
        std::scoped_lock scope{basic_lock};
        const auto next_time{absolute_time ? ns_into_future : GetGlobalTimeNs() + ns_into_future};

        auto h{event_queue.emplace(Event{next_time.count(), event_fifo_id++, event_type, 0})};
        (*h).handle = h;
    }

    event.Set();
}

void CoreTiming::ScheduleLoopingEvent(std::chrono::nanoseconds start_time,
                                      std::chrono::nanoseconds resched_time,
                                      const std::shared_ptr<EventType>& event_type,
                                      bool absolute_time) {
    {
        std::scoped_lock scope{basic_lock};
        const auto next_time{absolute_time ? start_time : GetGlobalTimeNs() + start_time};

        auto h{event_queue.emplace(
            Event{next_time.count(), event_fifo_id++, event_type, resched_time.count()})};
        (*h).handle = h;
    }

    event.Set();
}

void CoreTiming::UnscheduleEvent(const std::shared_ptr<EventType>& event_type,
                                 UnscheduleEventType type) {
    {
        std::scoped_lock lk{basic_lock};

        std::vector<heap_t::handle_type> to_remove;
        for (auto itr = event_queue.begin(); itr != event_queue.end(); itr++) {
            const Event& e = *itr;
            if (e.type.lock().get() == event_type.get()) {
                to_remove.push_back(itr->handle);
            }
        }

        for (auto& h : to_remove) {
            event_queue.erase(h);
        }

        event_type->sequence_number++;
    }

    // Force any in-progress events to finish
    if (type == UnscheduleEventType::Wait) {
        std::scoped_lock lk{advance_lock};
    }
}

static u64 GetNextTickCount(u64 next_ticks) {
    if (Settings::values.use_custom_cpu_ticks.GetValue()) {
        return Settings::values.cpu_ticks.GetValue();
    }
    return next_ticks;
}

void CoreTiming::AddTicks(u64 ticks_to_add) {
    const u64 ticks = GetNextTickCount(ticks_to_add);
    cpu_ticks += ticks;
    downcount -= static_cast<s64>(ticks);
}

void CoreTiming::Idle() {
    AddTicks(1000U);
}

void CoreTiming::ResetTicks() {
    downcount = MAX_SLICE_LENGTH;
}

u64 CoreTiming::GetClockTicks() const {
    u64 fres = is_multicore ? Common::g_wall_clock.GetCNTPCT() : Common::g_wall_clock.CPUTickToGuestCNTPCT(cpu_ticks);
    if (const u32 target = CpuClockTargetMhz(Settings::values.cpu_clock.GetValue());
        target != CPU_CLOCK_BASE_MHZ) {
        fres = fres * target / CPU_CLOCK_BASE_MHZ;
    }
    if (::Settings::values.sync_core_speed.GetValue()) {
        auto const ticks = f64(fres);
        auto const speed_limit = f64(Settings::SpeedLimit()) * 0.01;
        return u64(ticks / speed_limit);
    }
    return fres;
}

u64 CoreTiming::GetGPUTicks() const {
    return is_multicore
        ? Common::g_wall_clock.GetGPUTick()
        : Common::WallClock::CPUTickToGPUTick(cpu_ticks);
}

std::optional<s64> CoreTiming::Advance() {
    return AdvanceEvents().next;
}

CoreTiming::NextEvents CoreTiming::AdvanceEvents() {
    std::scoped_lock lock{advance_lock, basic_lock};
    global_timer = GetGlobalTimeNs().count();

    while (!event_queue.empty() && event_queue.top().time <= global_timer) {
        const Event& evt = event_queue.top();

        if (const auto event_type{evt.type.lock()}) {
            const auto evt_time = evt.time;
            const auto evt_sequence_num = event_type->sequence_number;
            const std::chrono::nanoseconds late{GetGlobalTimeNs().count() - evt_time};
            if (is_multicore && host_wait_stats.load(std::memory_order_relaxed)) {
                t_host_wait.event_late.push_back(late.count());
                if (evt.reschedule_time == 0) {
                    t_host_wait.one_shot_late.push_back(late.count());
                }
            }

            if (evt.reschedule_time == 0) {
                event_queue.pop();

                basic_lock.unlock();

                event_type->callback(evt_time, late);

                basic_lock.lock();
            } else {
                basic_lock.unlock();

                // Time spent paused is not lateness: a callback that makes up for being late (the
                // sysmix pump) would otherwise wait out the pause again after it.
                const std::chrono::nanoseconds late_since_pause{
                    evt_time < pause_end_time
                        ? (std::min)(late.count(),
                                     GetGlobalTimeNs().count() - static_cast<s64>(pause_end_time))
                        : late.count()};
                const auto new_schedule_time{event_type->callback(evt_time, late_since_pause)};

                basic_lock.lock();

                if (evt_sequence_num != event_type->sequence_number) {
                    // Heap handle is invalidated after external modification.
                    continue;
                }

                const auto next_schedule_time{new_schedule_time.has_value()
                                                  ? new_schedule_time.value().count()
                                                  : evt.reschedule_time};

                // If this event was scheduled into a pause, its time now is going to be way
                // behind. Re-set this event to continue from the end of the pause.
                auto next_time{evt.time + next_schedule_time};
                if (evt.time < pause_end_time) {
                    next_time = pause_end_time + next_schedule_time;
                }

                event_queue.update(evt.handle, Event{next_time, event_fifo_id++, evt.type,
                                                     next_schedule_time, evt.handle});
            }
        }

        global_timer = GetGlobalTimeNs().count();
    }

    if (event_queue.empty()) {
        return {};
    }
    NextEvents next{.next = event_queue.top().time, .one_shot = std::nullopt};
    for (const Event& e : event_queue) {
        if (e.reschedule_time == 0 && (!next.one_shot || e.time < *next.one_shot)) {
            next.one_shot = e.time;
        }
    }
    return next;
}

void CoreTiming::Reset() {
    paused = true;
    // Request the stop before waking the thread: woken first, it could take the wake-up, see no
    // stop request yet, and go back to waiting on pause_event for good (the join below then hung
    // the frontend at boot).
    if (timer_thread.joinable()) {
        timer_thread.request_stop();
    }
    pause_event.Set();
    event.Set();
    if (timer_thread.joinable()) {
        timer_thread.join();
    }
    has_started = false;
}

/// @brief Returns current time in nanoseconds.
std::chrono::nanoseconds CoreTiming::GetGlobalTimeNs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeNS()
        : std::chrono::nanoseconds{Common::WallClock::CPUTickToNS(cpu_ticks)};
}

/// @brief Returns current time in microseconds.
std::chrono::microseconds CoreTiming::GetGlobalTimeUs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeUS()
        : std::chrono::microseconds{Common::WallClock::CPUTickToUS(cpu_ticks)};
}

#ifdef _WIN32
void CoreTiming::SetTimerResolutionNs(std::chrono::nanoseconds ns) {
    timer_resolution_ns = ns.count();
}
#endif

} // namespace Core::Timing
