// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <optional>
#include <utility>

#include "common/literals.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "common/thread.h"
#include "video_core/shader_jobs.h"

namespace VideoCommon {

struct ShaderJobBatchState {
    mutable std::mutex mutex;
    mutable std::condition_variable condition;
    size_t pending{};
    std::atomic<bool> stop{};
};

struct ShaderJobState {
    // Waiters lock only this mutex; the scheduler sets a final status under it.
    mutable std::mutex mutex;
    mutable std::condition_variable condition;
    std::atomic<ShaderJobStatus> status{ShaderJobStatus::Queued};
    std::atomic<ShaderJobPriority> priority{};
    std::atomic<bool> stop{};

    // Under the scheduler's mutex, or owned by the thread running a stage.
    ShaderJob work;
    std::shared_ptr<ShaderJobBatchState> batch;
    std::exception_ptr exception;
    ShaderJobPriority queued_priority{}; ///< Which queue holds the live entry
    u64 held_bytes{};                    ///< Counted in the in-flight bytes now
    std::optional<u64> reported_cpu_bytes;
    std::chrono::steady_clock::time_point submitted;
    std::chrono::steady_clock::time_point stage_begin;
    std::optional<std::chrono::steady_clock::time_point> stalled_since;
    bool started{};
};

namespace {

constexpr size_t Index(ShaderJobPriority priority) {
    return static_cast<size_t>(priority);
}

constexpr bool IsFinal(ShaderJobStatus status) {
    return status == ShaderJobStatus::Done || status == ShaderJobStatus::Cancelled ||
           status == ShaderJobStatus::Failed;
}

constexpr bool IsQueued(ShaderJobStatus status) {
    return status == ShaderJobStatus::Queued || status == ShaderJobStatus::DriverQueued;
}

u64 Nanoseconds(std::chrono::steady_clock::duration duration) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

bool StopRequested(const ShaderJobState& job) {
    return job.stop.load(std::memory_order_relaxed) ||
           (job.batch && job.batch->stop.load(std::memory_order_relaxed));
}

ShaderJob TakeStages(ShaderJobState& job) {
    ShaderJob stages;
    stages.cpu = std::move(job.work.cpu);
    stages.driver = std::move(job.work.driver);
    return stages;
}

void LogFailure(const std::exception_ptr& exception, const char* stage) {
    try {
        std::rethrow_exception(exception);
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Shader job {} stage failed: {}", stage, e.what());
    } catch (...) {
        LOG_ERROR(Render_Vulkan, "Shader job {} stage failed with an unknown exception", stage);
    }
}

} // Anonymous namespace

ShaderJobStatus ShaderJobHandle::Status() const {
    return state->status.load();
}

ShaderJobPriority ShaderJobHandle::Priority() const {
    return state->priority.load();
}

bool ShaderJobHandle::Ready() const {
    return IsFinal(Status());
}

void ShaderJobHandle::Wait() const {
    std::unique_lock lock{state->mutex};
    state->condition.wait(lock, [this] { return Ready(); });
}

bool ShaderJobHandle::WaitFor(std::chrono::nanoseconds timeout) const {
    std::unique_lock lock{state->mutex};
    return state->condition.wait_for(lock, timeout, [this] { return Ready(); });
}

std::exception_ptr ShaderJobHandle::Exception() const {
    return Ready() ? state->exception : nullptr;
}

void ShaderJobBatch::Wait() const {
    std::unique_lock lock{state->mutex};
    state->condition.wait(lock, [this] { return state->pending == 0; });
}

bool ShaderJobBatch::Idle() const {
    std::scoped_lock lock{state->mutex};
    return state->pending == 0;
}

bool ShaderJobBatch::StopRequested() const {
    return state->stop.load(std::memory_order_relaxed);
}

bool ShaderJobContext::StopRequested() const {
    return VideoCommon::StopRequested(job);
}

void ShaderJobContext::ReportCpuBytes(u64 bytes) {
    std::scoped_lock lock{scheduler.mutex};
    auto& stats = scheduler.stats;
    stats.in_flight_bytes = stats.in_flight_bytes - job.held_bytes + bytes;
    stats.peak_in_flight_bytes = std::max(stats.peak_in_flight_bytes, stats.in_flight_bytes);
    const bool released = bytes < job.held_bytes;
    job.held_bytes = bytes;
    job.reported_cpu_bytes = bytes;
    if (released) {
        scheduler.WakeHeldWorkers();
    }
}

void ShaderJobContext::SetDriverBytes(u64 bytes) {
    // Read under the scheduler's mutex only after this stage returns.
    job.work.driver_bytes = bytes;
}

u64 DefaultShaderJobBudget() {
    using namespace Common::Literals;
    return std::max<u64>(Common::GetMemInfo().TotalPhysicalMemory / 8, 512_MiB);
}

ShaderJobScheduler::ShaderJobScheduler(const ShaderJobConfig& config)
    : budget{config.memory_budget != 0 ? config.memory_budget : DefaultShaderJobBudget()},
      background_aging{config.background_aging} {
    stats.budget_bytes = budget;
    const auto start{[this](std::string name, bool driver) {
        threads.emplace_back([this, name = std::move(name), driver] {
            Common::SetCurrentThreadName(name.c_str());
            Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
            Common::SetCurrentThreadToBackgroundWork();
            Worker(driver);
        });
    }};
    for (size_t i = 0; i < std::max<size_t>(config.cpu_threads, 1); ++i) {
        start(config.name + "Cpu", false);
    }
    for (size_t i = 0; i < std::max<size_t>(config.driver_threads, 1); ++i) {
        start(config.name + "Driver", true);
    }
}

ShaderJobScheduler::~ShaderJobScheduler() {
    Shutdown();
}

ShaderJobHandle ShaderJobScheduler::Submit(ShaderJob job, const ShaderJobBatch* batch) {
    auto state{std::make_shared<ShaderJobState>()};
    state->priority = job.priority;
    state->queued_priority = job.priority;
    state->work = std::move(job);
    state->batch = batch ? batch->state : nullptr;
    state->submitted = Clock::now();
    const bool driver{!state->work.cpu};
    std::vector<ShaderJob> dead;
    {
        std::scoped_lock lock{mutex};
        if (state->batch) {
            std::scoped_lock batch_lock{state->batch->mutex};
            ++state->batch->pending;
        }
        ++stats.priorities[Index(state->work.priority)].queued;
        if (stopping || StopRequested(*state)) {
            Finish(*state, ShaderJobStatus::Cancelled, dead);
            return ShaderJobHandle{std::move(state)};
        }
        state->status = driver ? ShaderJobStatus::DriverQueued : ShaderJobStatus::Queued;
        (driver ? driver_queue : cpu_queue)[Index(state->work.priority)].push_back(state);
    }
    (driver ? driver_condition : cpu_condition).notify_one();
    return ShaderJobHandle{std::move(state)};
}

ShaderJobBatch ShaderJobScheduler::CreateBatch() {
    return ShaderJobBatch{std::make_shared<ShaderJobBatchState>()};
}

void ShaderJobScheduler::Promote(const ShaderJobHandle& handle, ShaderJobPriority priority) {
    ShaderJobState& job{*handle.state};
    std::scoped_lock lock{mutex};
    const ShaderJobStatus status{job.status.load()};
    const ShaderJobPriority old_priority{job.priority.load()};
    if (IsFinal(status) || Index(priority) >= Index(old_priority)) {
        return;
    }
    auto& from{stats.priorities[Index(old_priority)]};
    auto& to{stats.priorities[Index(priority)]};
    if (IsQueued(status)) {
        --from.queued;
        ++to.queued;
    } else {
        --from.running;
        ++to.running;
    }
    job.priority = priority;
    if (!IsQueued(status)) {
        return;
    }
    const bool driver{status == ShaderJobStatus::DriverQueued};
    job.queued_priority = priority;
    (driver ? driver_queue : cpu_queue)[Index(priority)].push_back(handle.state);
    // A worker the budget holds back may now take it.
    (driver ? driver_condition : cpu_condition).notify_all();
}

void ShaderJobScheduler::Cancel(const ShaderJobHandle& handle) {
    ShaderJobState& job{*handle.state};
    std::vector<ShaderJob> dead;
    std::scoped_lock lock{mutex};
    job.stop = true;
    if (IsQueued(job.status.load())) {
        Finish(job, ShaderJobStatus::Cancelled, dead);
    }
}

void ShaderJobScheduler::Cancel(const ShaderJobBatch& batch) {
    std::vector<ShaderJob> dead;
    std::scoped_lock lock{mutex};
    batch.state->stop = true;
    for (Queue* const queue : {&cpu_queue, &driver_queue}) {
        for (auto& entries : *queue) {
            for (const JobPtr& job : entries) {
                if (job->batch == batch.state && IsQueued(job->status.load())) {
                    Finish(*job, ShaderJobStatus::Cancelled, dead);
                }
            }
        }
    }
}

void ShaderJobScheduler::RunOrWait(const ShaderJobHandle& handle) {
    Promote(handle, ShaderJobPriority::NeedNow);
    std::unique_lock lock{mutex};
    switch (handle.state->status.load()) {
    case ShaderJobStatus::Queued:
        StartStage(*handle.state, false, Clock::now());
        lock.unlock();
        RunCpu(handle.state, true);
        return;
    case ShaderJobStatus::DriverQueued:
        StartStage(*handle.state, true, Clock::now());
        lock.unlock();
        RunDriver(handle.state);
        return;
    default:
        break;
    }
    lock.unlock();
    handle.Wait();
}

void ShaderJobScheduler::Shutdown() {
    std::vector<ShaderJob> dead;
    std::vector<std::thread> joining;
    {
        std::scoped_lock lock{mutex};
        stopping = true;
        for (Queue* const queue : {&cpu_queue, &driver_queue}) {
            for (auto& entries : *queue) {
                for (const JobPtr& job : entries) {
                    if (IsQueued(job->status.load())) {
                        Finish(*job, ShaderJobStatus::Cancelled, dead);
                    }
                }
                entries.clear();
            }
        }
        joining = std::move(threads);
    }
    cpu_condition.notify_all();
    driver_condition.notify_all();
    for (std::thread& thread : joining) {
        thread.join();
    }
}

ShaderJobStats ShaderJobScheduler::Stats() const {
    std::scoped_lock lock{mutex};
    return stats;
}

void ShaderJobScheduler::Worker(bool driver) {
    std::unique_lock lock{mutex};
    while (true) {
        JobPtr job;
        (driver ? driver_condition : cpu_condition).wait(lock, [&] {
            return stopping || (job = Pick(driver, Clock::now()));
        });
        if (!job) {
            return;
        }
        StartStage(*job, driver, Clock::now());
        lock.unlock();
        if (driver) {
            RunDriver(job);
        } else {
            RunCpu(job, false);
        }
        job.reset();
        lock.lock();
    }
}

ShaderJobScheduler::JobPtr ShaderJobScheduler::Pick(bool driver, Clock::time_point now) {
    Queue& queue{driver ? driver_queue : cpu_queue};
    const ShaderJobStatus queued{driver ? ShaderJobStatus::DriverQueued : ShaderJobStatus::Queued};
    const auto front{[&](ShaderJobPriority priority) -> std::deque<JobPtr>* {
        auto& entries{queue[Index(priority)]};
        while (!entries.empty() && (entries.front()->status.load() != queued ||
                                    entries.front()->queued_priority != priority)) {
            entries.pop_front();
        }
        return entries.empty() ? nullptr : &entries;
    }};
    bool& aged_last{driver ? driver_aged_last : cpu_aged_last};
    bool aged{aged_last};
    std::deque<JobPtr>* entries{front(ShaderJobPriority::NeedNow)};
    if (!entries) {
        std::deque<JobPtr>* const soon{front(ShaderJobPriority::Soon)};
        std::deque<JobPtr>* const background{front(ShaderJobPriority::Background)};
        // An aged Background job ahead of the Soon jobs takes every other pick, so a backlog
        // doesn't hold back the Soon jobs submitted after it.
        aged = background && soon && !aged_last &&
               now - background->front()->submitted >= background_aging &&
               background->front()->submitted < soon->front()->submitted;
        entries = aged ? background : (soon ? soon : background);
    }
    if (!entries) {
        return nullptr;
    }
    ShaderJobState& job{*entries->front()};
    const u64 bytes{driver ? job.work.driver_bytes : job.work.cpu_bytes};
    if (job.priority.load() != ShaderJobPriority::NeedNow && stats.in_flight_bytes != 0 &&
        stats.in_flight_bytes + bytes > budget) {
        if (!job.stalled_since) {
            job.stalled_since = now;
            ++stats.budget_stalls;
        }
        budget_held = true;
        return nullptr;
    }
    aged_last = aged;
    JobPtr picked{std::move(entries->front())};
    entries->pop_front();
    return picked;
}

void ShaderJobScheduler::StartStage(ShaderJobState& job, bool driver, Clock::time_point now) {
    auto& counters{stats.priorities[Index(job.priority.load())]};
    --counters.queued;
    ++counters.running;
    if (!job.started) {
        job.started = true;
        const u64 wait{Nanoseconds(now - job.submitted)};
        counters.queue_wait_ns += wait;
        counters.max_queue_wait_ns = std::max(counters.max_queue_wait_ns, wait);
    }
    if (job.stalled_since) {
        stats.budget_stall_ns += Nanoseconds(now - *job.stalled_since);
        job.stalled_since.reset();
    }
    job.held_bytes = driver ? job.work.driver_bytes : job.work.cpu_bytes;
    stats.in_flight_bytes += job.held_bytes;
    stats.peak_in_flight_bytes = std::max(stats.peak_in_flight_bytes, stats.in_flight_bytes);
    job.status = driver ? ShaderJobStatus::Driver : ShaderJobStatus::Cpu;
    job.stage_begin = now;
}

void ShaderJobScheduler::RunCpu(const JobPtr& job, bool run_driver_here) {
    ShaderDriverStage driver;
    std::exception_ptr exception;
    try {
        ShaderJobContext context{*this, *job};
        driver = job->work.cpu(context);
    } catch (...) {
        exception = std::current_exception();
    }
    // The CPU stage's captures go before the driver stage can start.
    job->work.cpu = {};
    if (exception) {
        LogFailure(exception, "CPU");
    }
    std::vector<ShaderJob> dead;
    bool run_driver{};
    {
        std::scoped_lock lock{mutex};
        auto& counters{stats.priorities[Index(job->priority.load())]};
        counters.cpu_run_ns += Nanoseconds(Clock::now() - job->stage_begin);
        if (job->reported_cpu_bytes) {
            stats.estimated_cpu_bytes += job->work.cpu_bytes;
            stats.reported_cpu_bytes += *job->reported_cpu_bytes;
        }
        ReleaseBytes(*job);
        if (exception) {
            job->exception = exception;
            Finish(*job, ShaderJobStatus::Failed, dead);
        } else if (!driver) {
            Finish(*job, StopRequested(*job) ? ShaderJobStatus::Cancelled : ShaderJobStatus::Done,
                   dead);
        } else if (stopping || StopRequested(*job)) {
            dead.emplace_back().driver = std::move(driver);
            Finish(*job, ShaderJobStatus::Cancelled, dead);
        } else {
            job->work.driver = std::move(driver);
            --counters.running;
            ++counters.queued;
            if (run_driver_here) {
                StartStage(*job, true, Clock::now());
                run_driver = true;
            } else {
                job->status = ShaderJobStatus::DriverQueued;
                job->queued_priority = job->priority.load();
                driver_queue[Index(job->queued_priority)].push_back(job);
                driver_condition.notify_one();
            }
        }
    }
    if (run_driver) {
        RunDriver(job);
    }
}

void ShaderJobScheduler::RunDriver(const JobPtr& job) {
    std::exception_ptr exception;
    try {
        if (job->work.driver) {
            job->work.driver();
        }
    } catch (...) {
        exception = std::current_exception();
    }
    job->work.driver = {};
    if (exception) {
        LogFailure(exception, "driver");
    }
    std::vector<ShaderJob> dead;
    std::scoped_lock lock{mutex};
    stats.priorities[Index(job->priority.load())].driver_run_ns +=
        Nanoseconds(Clock::now() - job->stage_begin);
    ReleaseBytes(*job);
    job->exception = exception;
    Finish(*job, exception ? ShaderJobStatus::Failed : ShaderJobStatus::Done, dead);
}

void ShaderJobScheduler::Finish(ShaderJobState& job, ShaderJobStatus status,
                                std::vector<ShaderJob>& dead) {
    auto& counters{stats.priorities[Index(job.priority.load())]};
    if (IsQueued(job.status.load())) {
        --counters.queued;
    } else {
        --counters.running;
    }
    switch (status) {
    case ShaderJobStatus::Done:
        ++counters.done;
        break;
    case ShaderJobStatus::Cancelled:
        ++counters.cancelled;
        break;
    default:
        ++counters.failed;
        break;
    }
    ReleaseBytes(job);
    dead.push_back(TakeStages(job));
    if (job.batch) {
        std::scoped_lock batch_lock{job.batch->mutex};
        if (--job.batch->pending == 0) {
            job.batch->condition.notify_all();
        }
    }
    {
        std::scoped_lock job_lock{job.mutex};
        job.status = status;
    }
    job.condition.notify_all();
}

void ParallelFor(ShaderJobScheduler& scheduler, size_t count,
                 const std::function<void(size_t)>& function) {
    std::vector<ShaderJobHandle> handles;
    handles.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        handles.push_back(scheduler.Submit({
            .priority = ShaderJobPriority::NeedNow,
            .cpu = [&function, index](ShaderJobContext&) {
                function(index);
                return ShaderDriverStage{};
            },
        }));
    }
    // Workers take NeedNow jobs oldest first, so the calling thread runs from the newest: it waits
    // only for calls already running.
    for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
        scheduler.RunOrWait(*it);
    }
}

void ShaderJobScheduler::ReleaseBytes(ShaderJobState& job) {
    if (job.held_bytes == 0) {
        return;
    }
    stats.in_flight_bytes -= job.held_bytes;
    job.held_bytes = 0;
    WakeHeldWorkers();
}

void ShaderJobScheduler::WakeHeldWorkers() {
    if (!budget_held) {
        return;
    }
    budget_held = false;
    cpu_condition.notify_all();
    driver_condition.notify_all();
}

} // namespace VideoCommon
