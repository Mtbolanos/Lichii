// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "common/unique_function.h"

namespace VideoCommon {

/// NeedNow: a draw or dispatch waits for it. Soon: bound but not drawn yet, or a boot build of a
/// used pipeline. Background: prebuild, precache, relowering, migration.
enum class ShaderJobPriority : u8 {
    NeedNow,
    Soon,
    Background,
};
inline constexpr size_t NumShaderJobPriorities = 3;

enum class ShaderJobStatus : u8 {
    Queued,       ///< Waiting for a CPU worker, or for the memory budget
    Cpu,          ///< CPU stage running (lowering, translation, emission)
    DriverQueued, ///< CPU stage finished, waiting for a driver worker or the budget
    Driver,       ///< Driver stage running (vkCreate*Pipelines)
    Done,
    Cancelled,
    Failed,
};

class ShaderJobContext;
using ShaderDriverStage = Common::UniqueFunction<void>;
using ShaderCpuStage = Common::UniqueFunction<ShaderDriverStage, ShaderJobContext&>;

/// One unit of shader work. The CPU stage returns the driver stage, so whatever the CPU stage
/// owned (IR pools, programs) is destroyed before the driver compile starts. A job without a CPU
/// stage starts in the driver queue with `driver`.
struct ShaderJob {
    ShaderJobPriority priority{ShaderJobPriority::Background};
    /// Estimated bytes each stage holds while it runs.
    u64 cpu_bytes{};
    u64 driver_bytes{};
    ShaderCpuStage cpu;
    ShaderDriverStage driver;
};

struct ShaderJobState;
struct ShaderJobBatchState;

/// Waiting on a handle needs no scheduler, so it also works once the scheduler is gone.
class ShaderJobHandle {
public:
    ShaderJobHandle() = default;

    explicit operator bool() const noexcept {
        return static_cast<bool>(state);
    }

    ShaderJobStatus Status() const;
    ShaderJobPriority Priority() const;
    /// Done, Cancelled or Failed.
    bool Ready() const;
    void Wait() const;
    /// @return True when the job reached a final state within `timeout`.
    bool WaitFor(std::chrono::nanoseconds timeout) const;
    /// What a stage threw, once the job failed.
    std::exception_ptr Exception() const;

private:
    friend class ShaderJobScheduler;
    explicit ShaderJobHandle(std::shared_ptr<ShaderJobState> state_) : state{std::move(state_)} {}

    std::shared_ptr<ShaderJobState> state;
};

/// Jobs submitted together, e.g. one boot load, to be waited for and cancelled together without
/// touching the scheduler's other jobs.
class ShaderJobBatch {
public:
    /// Waits until every job submitted with this batch has reached a final state.
    void Wait() const;
    bool Idle() const;
    bool StopRequested() const;

private:
    friend class ShaderJobScheduler;
    explicit ShaderJobBatch(std::shared_ptr<ShaderJobBatchState> state_)
        : state{std::move(state_)} {}

    std::shared_ptr<ShaderJobBatchState> state;
};

struct ShaderJobStats {
    /// By the job's priority at the time.
    struct Priority {
        u64 queued{};  ///< Now in either queue
        u64 running{}; ///< Now in either stage
        u64 done{};
        u64 cancelled{};
        u64 failed{};
        u64 queue_wait_ns{}; ///< Submission to the first stage's start, over started jobs
        u64 max_queue_wait_ns{};
        u64 cpu_run_ns{};
        u64 driver_run_ns{};
    };
    std::array<Priority, NumShaderJobPriorities> priorities{};
    u64 budget_stalls{};   ///< Stage starts the budget held back
    u64 budget_stall_ns{}; ///< Time they were held back
    u64 budget_bytes{};
    u64 in_flight_bytes{};
    u64 peak_in_flight_bytes{};
    /// Over the CPU stages that reported their size: their estimates and the reported sizes.
    u64 estimated_cpu_bytes{};
    u64 reported_cpu_bytes{};
};

struct ShaderJobConfig {
    size_t cpu_threads{1};
    size_t driver_threads{1};
    /// Zero means DefaultShaderJobBudget().
    u64 memory_budget{};
    /// A Background job queued this long takes every other pick from the Soon jobs submitted
    /// after it.
    std::chrono::nanoseconds background_aging{std::chrono::seconds{2}};
    /// Worker threads are named <name>Cpu and <name>Driver.
    std::string name{"ShaderJobs"};
};

/// An eighth of physical memory, at least 512 MiB.
u64 DefaultShaderJobBudget();

/// Runs shader jobs on two pools, CPU workers for the CPU stage and driver workers for the driver
/// stage, each taking the highest priority stage first. In-flight bytes are the estimates of the
/// stages now running. A Soon or Background stage starts only while they stay within the budget,
/// or when nothing else is in flight; a NeedNow stage always starts.
class ShaderJobScheduler {
public:
    explicit ShaderJobScheduler(const ShaderJobConfig& config);
    ~ShaderJobScheduler();

    ShaderJobScheduler(const ShaderJobScheduler&) = delete;
    ShaderJobScheduler& operator=(const ShaderJobScheduler&) = delete;

    /// A job submitted after Shutdown, or with a stopped batch, is cancelled at once.
    ShaderJobHandle Submit(ShaderJob job, const ShaderJobBatch* batch = nullptr);

    ShaderJobBatch CreateBatch();

    /// Raises a job's priority, wherever it is queued or running. A lower priority is ignored.
    void Promote(const ShaderJobHandle& handle, ShaderJobPriority priority);

    /// A queued job is cancelled at once. A running CPU stage sees StopRequested() and no driver
    /// stage follows it; a running driver stage completes.
    void Cancel(const ShaderJobHandle& handle);

    /// Cancel for every job of the batch, now and later submitted.
    void Cancel(const ShaderJobBatch& batch);

    /// Promotes the job to NeedNow and runs its queued stages on the calling thread, or waits for
    /// the worker running them.
    void RunOrWait(const ShaderJobHandle& handle);

    /// Cancels the queued jobs and joins the workers once their running stages end. A CPU stage
    /// that ends during shutdown gets no driver stage.
    void Shutdown();

    ShaderJobStats Stats() const;

private:
    friend class ShaderJobContext;
    using JobPtr = std::shared_ptr<ShaderJobState>;
    using Clock = std::chrono::steady_clock;
    using Queue = std::array<std::deque<JobPtr>, NumShaderJobPriorities>;

    void Worker(bool driver);
    JobPtr Pick(bool driver, Clock::time_point now);
    void StartStage(ShaderJobState& job, bool driver, Clock::time_point now);
    void RunCpu(const JobPtr& job, bool run_driver_here);
    void RunDriver(const JobPtr& job);
    void Finish(ShaderJobState& job, ShaderJobStatus status, std::vector<ShaderJob>& dead);
    void ReleaseBytes(ShaderJobState& job);
    void WakeHeldWorkers();

    const u64 budget;
    const std::chrono::nanoseconds background_aging;

    mutable std::mutex mutex;
    std::condition_variable cpu_condition;
    std::condition_variable driver_condition;
    /// Entries go stale on promotion, cancellation or a stage start and are skipped when reached.
    Queue cpu_queue;
    Queue driver_queue;
    bool stopping{};
    /// A worker waits for in-flight bytes to drop.
    bool budget_held{};
    /// The queue's last Soon or Background pick was an aged Background job ahead of Soon jobs.
    bool cpu_aged_last{};
    bool driver_aged_last{};
    ShaderJobStats stats;
    std::vector<std::thread> threads;
};

/// Calls `function` with every index below `count`, as NeedNow jobs that the calling thread helps
/// run, and returns once every call returned. Calls cancelled by a shutdown are left out.
/// `function` must not throw.
void ParallelFor(ShaderJobScheduler& scheduler, size_t count,
                 const std::function<void(size_t)>& function);

/// Given to a CPU stage while it runs.
class ShaderJobContext {
public:
    /// The job or its batch was cancelled; the stage may return early.
    bool StopRequested() const;
    /// Replaces the CPU stage's estimate with its measured size, e.g. the IR pools' capacity.
    void ReportCpuBytes(u64 bytes);
    /// Replaces the driver stage's estimate, e.g. once the SPIR-V size is known.
    void SetDriverBytes(u64 bytes);

private:
    friend class ShaderJobScheduler;
    ShaderJobContext(ShaderJobScheduler& scheduler_, ShaderJobState& job_)
        : scheduler{scheduler_}, job{job_} {}

    ShaderJobScheduler& scheduler;
    ShaderJobState& job;
};

} // namespace VideoCommon
