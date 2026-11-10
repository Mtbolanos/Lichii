// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <optional>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/query_cache/bank_base.h"
#include "video_core/query_cache/query_base.h"

namespace VideoCommon {

class StreamerInterface {
public:
    explicit StreamerInterface(size_t id_) : id{id_}, dependence_mask{}, dependent_mask{} {}
    virtual ~StreamerInterface() = default;

    virtual QueryBase* GetQuery(size_t id) = 0;

    virtual void StartCounter() {
        /* Do Nothing */
    }

    virtual void PauseCounter() {
        /* Do Nothing */
    }

    virtual void ResetCounter() {
        /* Do Nothing */
    }

    virtual void CloseCounter() {
        /* Do Nothing */
    }

    virtual bool HasPendingSync() const {
        return false;
    }

    virtual void PresyncWrites() {
        /* Do Nothing */
    }

    virtual void SyncWrites() {
        /* Do Nothing */
    }

    /// EDEN_NX2_EXP28 & 1: a fence went without this streamer's pending syncs; they wait for a
    /// later fence or wait for idle while their query slots may be reused.
    virtual void NoteSyncDeferred() {
        /* Do Nothing */
    }

    virtual size_t WriteCounter(VAddr address, bool has_timestamp, u32 value,
                                std::optional<u32> subreport = std::nullopt) = 0;

    virtual bool HasUnsyncedQueries() const {
        return false;
    }

    virtual void PushUnsyncedQueries() {
        /* Do Nothing */
    }

    virtual void PopUnsyncedQueries() {
        /* Do Nothing */
    }

    virtual void Free(size_t query_id) = 0;

    size_t GetId() const {
        return id;
    }

    u64 GetDependenceMask() const {
        return dependence_mask;
    }

    u64 GetDependentMask() const {
        return dependent_mask;
    }

    u64 GetAmendValue() const {
        return amend_value;
    }

    void SetAccumulationValue(u64 new_value) {
        accumulation_value = new_value;
    }

protected:
    void MakeDependent(StreamerInterface* depend_on) {
        dependence_mask |= 1ULL << depend_on->id;
        depend_on->dependent_mask |= 1ULL << id;
    }

    const size_t id;
    u64 dependence_mask;
    u64 dependent_mask;
    u64 amend_value{};
    u64 accumulation_value{};
};

template <typename QueryType>
class SimpleStreamer : public StreamerInterface {
public:
    explicit SimpleStreamer(size_t id_) : StreamerInterface{id_} {}
    virtual ~SimpleStreamer() = default;

protected:
    virtual QueryType* GetQuery(size_t query_id) override {
        if (query_id < slot_queries.size()) {
            return &slot_queries[query_id];
        }
        return nullptr;
    }

    virtual void Free(size_t query_id) override {
        std::scoped_lock lk(guard);
        ReleaseQuery(query_id);
    }

    template <typename... Args, typename = decltype(QueryType(std::declval<Args>()...))>
    size_t BuildQuery(Args&&... args) {
        std::scoped_lock lk(guard);
        if (old_queries_head != old_queries.size()) {
            size_t new_id = old_queries[old_queries_head++];
            if (old_queries_head == old_queries.size()) {
                old_queries.clear();
                old_queries_head = 0;
            }
            new (&slot_queries[new_id]) QueryType(std::forward<Args>(args)...);
            return new_id;
        }
        size_t new_id = slot_queries.size();
        slot_queries.emplace_back(std::forward<Args>(args)...);
        return new_id;
    }

    void ReleaseQuery(size_t query_id) {

        if (query_id < slot_queries.size()) {
            if (old_queries_head >= 1024 && old_queries_head * 2 >= old_queries.size()) {
                old_queries.erase(old_queries.begin(),
                                  old_queries.begin() + static_cast<std::ptrdiff_t>(old_queries_head));
                old_queries_head = 0;
            }
            old_queries.push_back(query_id);
            return;
        }
        UNREACHABLE();
    }

    std::mutex guard;
    std::deque<QueryType> slot_queries;
    /// Free ids, oldest first from old_queries_head: MSVC's deque of size_t allocated a block
    /// every two pushes, and a report per draw frees two queries.
    std::vector<size_t> old_queries;
    size_t old_queries_head{};
};

} // namespace VideoCommon
