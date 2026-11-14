// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/scope_exit.h"
#include "common/unique_function.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/shader_jobs.h"

namespace Vulkan {

/// Shader cache v2: cached pipelines built in the background as Background jobs of one batch. A
/// lookup of one still queued or building raises it to NeedNow and waits for it, so that it is
/// built once.
template <typename Key, typename Pipeline>
class ScheduledPrebuild {
    struct Entry {
        VideoCommon::ShaderJobHandle job;
        std::unique_ptr<Pipeline> pipeline;
    };

public:
    /// Translates a pipeline on a CPU worker, unbuilt (null when it doesn't translate). Its
    /// Build() runs on a driver worker, and it is dropped when it Failed().
    using Translate = Common::UniqueFunction<std::unique_ptr<Pipeline>>;

    /// What became of the builds once every one has ended
    struct Outcome {
        size_t built;
        size_t failed;
        size_t taken_unfinished; ///< Taken by a lookup while still queued or building
        size_t unclaimed;        ///< Built, and no lookup has taken them yet
    };

    explicit ScheduledPrebuild(VideoCommon::ShaderJobScheduler& jobs_)
        : jobs{jobs_}, batch{jobs_.CreateBatch()} {}

    ~ScheduledPrebuild() {
        Stop();
    }

    ScheduledPrebuild(const ScheduledPrebuild&) = delete;
    ScheduledPrebuild& operator=(const ScheduledPrebuild&) = delete;

    /// Submits the build of a pipeline, unless one of its key is queued
    bool Add(const Key& key, u64 cpu_bytes, u64 driver_bytes, Translate translate) {
        std::scoped_lock lock{mutex};
        if (stopped || entries.contains(key)) {
            return false;
        }
        ++translating;
        auto entry = std::make_shared<Entry>();
        entry->job = jobs.Submit(
            {
                .priority = VideoCommon::ShaderJobPriority::Background,
                .cpu_bytes = cpu_bytes,
                .driver_bytes = driver_bytes,
                .cpu = [this, entry, translate = std::move(translate)](
                           VideoCommon::ShaderJobContext& context) mutable
                -> VideoCommon::ShaderDriverStage {
                    std::unique_ptr<Pipeline> pipeline;
                    {
                        SCOPE_EXIT {
                            translate = {};
                            Translated();
                        };
                        if (context.StopRequested()) {
                            return {};
                        }
                        pipeline = translate();
                    }
                    if (!pipeline) {
                        std::scoped_lock lock{mutex};
                        ++failed;
                        return {};
                    }
                    return [this, entry, pipeline = std::move(pipeline)]() mutable {
                        pipeline->Build();
                        std::scoped_lock entry_lock{mutex};
                        if (pipeline->Failed()) {
                            ++failed;
                            return;
                        }
                        ++built;
                        entry->pipeline = std::move(pipeline);
                    };
                },
            },
            &batch);
        entries.emplace(key, std::move(entry));
        return true;
    }

    /// The pipeline of `key`, built here or waited for while it builds (`waited` gets how long
    /// when it was still queued or building); null when there is none or it didn't build
    std::unique_ptr<Pipeline> Take(
        const Key& key, std::optional<std::chrono::steady_clock::duration>* waited = nullptr) {
        std::shared_ptr<Entry> entry;
        {
            std::scoped_lock lock{mutex};
            const auto it = entries.find(key);
            if (it == entries.end()) {
                return nullptr;
            }
            entry = std::move(it->second);
            entries.erase(it);
        }
        const bool unfinished = !entry->job.Ready();
        const auto begin = std::chrono::steady_clock::now();
        jobs.RunOrWait(entry->job);
        std::scoped_lock lock{mutex};
        if (unfinished) {
            ++taken_unfinished;
            if (waited) {
                *waited = std::chrono::steady_clock::now() - begin;
            }
        }
        return std::move(entry->pipeline);
    }

    /// Calls `done` once every build added so far was translated, on the thread that translated
    /// the last one, or now when none is left. Never once Stop() returned.
    void AfterTranslations(Common::UniqueFunction<void> done) {
        {
            std::scoped_lock lock{mutex};
            if (translating != 0) {
                after_translations = std::move(done);
                return;
            }
        }
        done();
    }

    /// The outcome once every build has ended, else nothing
    std::optional<Outcome> Finished() const {
        if (!batch.Idle()) {
            return std::nullopt;
        }
        std::scoped_lock lock{mutex};
        return Outcome{
            .built = built,
            .failed = failed,
            .taken_unfinished = taken_unfinished,
            .unclaimed = static_cast<size_t>(std::ranges::count_if(
                entries, [](const auto& entry) { return entry.second->pipeline != nullptr; })),
        };
    }

    /// The batch of the builds, to cancel them without waiting
    const VideoCommon::ShaderJobBatch& Batch() const {
        return batch;
    }

    /// Cancels the builds and drops every pipeline once the running ones end
    void Stop() {
        {
            std::scoped_lock lock{mutex};
            after_translations = {};
        }
        jobs.Cancel(batch);
        batch.Wait();
        std::unordered_map<Key, std::shared_ptr<Entry>> dropped;
        std::scoped_lock lock{mutex};
        stopped = true;
        dropped = std::move(entries);
        entries.clear();
    }

private:
    void Translated() {
        Common::UniqueFunction<void> done;
        {
            std::scoped_lock lock{mutex};
            if (--translating != 0 || !after_translations) {
                return;
            }
            done = std::move(after_translations);
            after_translations = {};
        }
        done();
    }

    VideoCommon::ShaderJobScheduler& jobs;
    VideoCommon::ShaderJobBatch batch;
    size_t translating{}; ///< Builds added and not translated yet
    Common::UniqueFunction<void> after_translations;
    size_t built{};
    size_t failed{};
    size_t taken_unfinished{};
    bool stopped{};
    mutable std::mutex mutex;
    std::unordered_map<Key, std::shared_ptr<Entry>> entries;
};

/// The skipped cached records the prebuild takes, from `skipped` in the order of the log: the
/// newest first (the log appends, so its end is what recent sessions recorded), at most `limit`.
template <typename Record>
std::vector<Record> NewestFirst(std::vector<Record> skipped, size_t limit) {
    std::ranges::reverse(skipped);
    skipped.erase(skipped.begin() + static_cast<std::ptrdiff_t>(std::min(limit, skipped.size())),
                  skipped.end());
    return skipped;
}

} // namespace Vulkan
