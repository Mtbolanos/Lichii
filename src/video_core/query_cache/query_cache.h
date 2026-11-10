// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/query_cache/bank_base.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_cache/query_base.h"
#include "video_core/query_cache/query_cache_base.h"
#include "video_core/query_cache/query_stream.h"
#include "video_core/query_cache/types.h"

namespace VideoCommon {

using Maxwell = Tegra::Engines::Maxwell3D;

struct SyncValuesStruct {
    VAddr address;
    u64 value;
    u64 size;

    static constexpr bool GeneratesBaseBuffer = true;
};

template <typename Traits>
class GuestStreamer : public SimpleStreamer<GuestQuery> {
public:
    using RuntimeType = typename Traits::RuntimeType;

    GuestStreamer(size_t id_, RuntimeType& runtime_)
        : SimpleStreamer<GuestQuery>(id_), runtime{runtime_} {}

    virtual ~GuestStreamer() = default;

    size_t WriteCounter(VAddr address, bool has_timestamp, u32 value,
                        std::optional<u32> subreport = std::nullopt) override {
        auto new_id = BuildQuery(has_timestamp, address, static_cast<u64>(value));
        // The value is kept here too: with EXP28 & 1 the sync can come after fences that freed
        // the slot, which may hold another release by then (zortos293/Lichii#717).
        pending_sync.push_back(PendingSync{
            .id = new_id,
            .address = address,
            .value = static_cast<u64>(value),
            .has_timestamp = has_timestamp,
        });
        return new_id;
    }

    bool HasPendingSync() const override {
        return !pending_sync.empty();
    }

    void NoteSyncDeferred() override {
        if (!pending_sync.empty()) {
            sync_deferred = true;
        }
    }

    void SyncWrites() override {
        if (pending_sync.empty()) {
            sync_deferred = false;
            return;
        }
        std::vector<SyncValuesStruct> sync_values;
        sync_values.reserve(pending_sync.size());
        if (std::exchange(sync_deferred, false)) {
            SyncDeferredWrites(sync_values);
        } else {
            for (const PendingSync& pending : pending_sync) {
                auto& query = slot_queries[pending.id];
                if (True(query.flags & QueryFlagBits::IsRewritten) ||
                    True(query.flags & QueryFlagBits::IsInvalidated)) {
                    continue;
                }
                query.flags |= QueryFlagBits::IsHostSynced;
                sync_values.emplace_back(SyncValuesStruct{
                    .address = query.guest_address,
                    .value = query.value,
                    .size = static_cast<u64>(
                        True(query.flags & QueryFlagBits::HasTimestamp) ? 8 : 4)});
            }
        }
        pending_sync.clear();
        if (sync_values.size() > 0) {
            runtime.template SyncValues<SyncValuesStruct>(sync_values);
        }
    }

private:
    struct PendingSync {
        size_t id;
        VAddr address;
        u64 value;
        bool has_timestamp;
    };

    /// Syncs that waited past a fence (EXP28 & 1, zortos293/Lichii#717 SyncWrites): the values
    /// as reported, newest first, one per address (one copy command must not write a range
    /// twice); a slot's flags count only while it still holds that release.
    void SyncDeferredWrites(std::vector<SyncValuesStruct>& sync_values) {
        std::vector<VAddr> seen;
        seen.reserve(pending_sync.size());
        for (auto it = pending_sync.rbegin(); it != pending_sync.rend(); ++it) {
            if (std::find(seen.begin(), seen.end(), it->address) != seen.end()) {
                continue;
            }
            seen.push_back(it->address);
            auto& query = slot_queries[it->id];
            const bool same = query.guest_address == it->address && query.value == it->value;
            if (same && (True(query.flags & QueryFlagBits::IsRewritten) ||
                         True(query.flags & QueryFlagBits::IsInvalidated))) {
                continue;
            }
            if (same) {
                query.flags |= QueryFlagBits::IsHostSynced;
            }
            sync_values.emplace_back(SyncValuesStruct{
                .address = it->address,
                .value = it->value,
                .size = static_cast<u64>(it->has_timestamp ? 8 : 4)});
        }
    }

    RuntimeType& runtime;
    std::deque<PendingSync> pending_sync;
    /// EXP28 & 1: pending_sync waited past a fence (NoteSyncDeferred).
    bool sync_deferred{};
};

template <typename Traits>
class StubStreamer : public GuestStreamer<Traits> {
public:
    using RuntimeType = typename Traits::RuntimeType;

    StubStreamer(size_t id_, RuntimeType& runtime_, u32 stub_value_)
        : GuestStreamer<Traits>(id_, runtime_), stub_value{stub_value_} {}

    ~StubStreamer() override = default;

    size_t WriteCounter(VAddr address, bool has_timestamp, [[maybe_unused]] u32 value,
                        std::optional<u32> subreport = std::nullopt) override {
        size_t new_id =
            GuestStreamer<Traits>::WriteCounter(address, has_timestamp, stub_value, subreport);
        return new_id;
    }

private:
    u32 stub_value;
};

template <typename Traits>
struct QueryCacheBase<Traits>::QueryCacheBaseImpl {
    using RuntimeType = typename Traits::RuntimeType;

    QueryCacheBaseImpl(QueryCacheBase<Traits>* owner_, VideoCore::RasterizerInterface& rasterizer_,
                       Tegra::MaxwellDeviceMemoryManager& device_memory_, RuntimeType& runtime_,
                       Tegra::GPU& gpu_)
        : owner{owner_}, rasterizer{rasterizer_}, device_memory{device_memory_}, runtime{runtime_},
          gpu{gpu_} {
        streamer_mask = 0;
        for (size_t i = 0; i < static_cast<size_t>(QueryType::MaxQueryTypes); i++) {
            streamers[i] = runtime.GetStreamerInterface(static_cast<QueryType>(i));
            if (streamers[i]) {
                streamer_mask |= 1ULL << streamers[i]->GetId();
            }
        }
    }

    template <typename Func>
    void ForEachStreamerIn(u64 mask, Func&& func) {
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result<Func, StreamerInterface*>, bool>;
        while (mask != 0) {
            size_t position = std::countr_zero(mask);
            mask &= ~(1ULL << position);
            if constexpr (RETURNS_BOOL) {
                if (func(streamers[position])) {
                    return;
                }
            } else {
                func(streamers[position]);
            }
        }
    }

    template <typename Func>
    void ForEachStreamer(Func&& func) {
        ForEachStreamerIn(streamer_mask, func);
    }

    QueryBase* ObtainQuery(QueryCacheBase<Traits>::QueryLocation location) {
        size_t which_stream = location.stream_id.Value();
        auto* streamer = streamers[which_stream];
        if (!streamer) {
            return nullptr;
        }
        return streamer->GetQuery(location.query_id.Value());
    }

    QueryCacheBase<Traits>* owner;
    VideoCore::RasterizerInterface& rasterizer;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    RuntimeType& runtime;
    Tegra::GPU& gpu;
    std::array<StreamerInterface*, static_cast<size_t>(QueryType::MaxQueryTypes)> streamers;
    u64 streamer_mask;
    std::mutex flush_guard;
    std::deque<u64> flushes_pending;
    std::vector<QueryCacheBase<Traits>::QueryLocation> pending_unregister;
};

template <typename Traits>
QueryCacheBase<Traits>::QueryCacheBase(Tegra::GPU& gpu_,
                                       VideoCore::RasterizerInterface& rasterizer_,
                                       Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                       RuntimeType& runtime_)
    : cached_queries{} {
    impl = std::make_unique<QueryCacheBase<Traits>::QueryCacheBaseImpl>(
        this, rasterizer_, device_memory_, runtime_, gpu_);
}

template <typename Traits>
QueryCacheBase<Traits>::~QueryCacheBase() = default;

template <typename Traits>
void QueryCacheBase<Traits>::CounterEnable(QueryType counter_type, bool is_enabled) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNREACHABLE();
        return;
    }
    if (is_enabled) {
        streamer->StartCounter();
    } else {
        streamer->PauseCounter();
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterClose(QueryType counter_type) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNREACHABLE();
        return;
    }
    streamer->CloseCounter();
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterReset(QueryType counter_type) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNIMPLEMENTED();
        return;
    }
    streamer->ResetCounter();
}

template <typename Traits>
void QueryCacheBase<Traits>::BindToChannel(s32 id) {
    VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo>::BindToChannel(id);
    impl->runtime.Bind3DEngine(maxwell3d);
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterReport(GPUVAddr addr, QueryType counter_type,
                                           QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    const bool has_timestamp = True(flags & QueryPropertiesFlags::HasTimeout);
    const bool is_fence = True(flags & QueryPropertiesFlags::IsAFence);
    const auto requested_type = counter_type;
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (static const bool trace = std::getenv("EDEN_NX2_QUERY_TRACE") != nullptr;
        trace || (watch != 0 && (addr & ~GPUVAddr{0xf}) == watch)) {
        LOG_INFO(HW_GPU, "nx2 report addr={:#x} type={} flags={:#x} payload={:#x} as={} device={:#x}", addr,
                 static_cast<u32>(counter_type), static_cast<u32>(flags), payload, gpu_memory->GetID(),
                 gpu_memory->GpuToCpuAddress(addr).value_or(0));
    }
    size_t streamer_id = static_cast<size_t>(counter_type);
    auto* streamer = impl->streamers[streamer_id];
    if (streamer == nullptr) [[unlikely]] {
        counter_type = QueryType::Payload;
        payload = 1U;
        streamer_id = static_cast<size_t>(counter_type);
        streamer = impl->streamers[streamer_id];
    }
    auto cpu_addr_opt = gpu_memory->GpuToCpuAddress(addr);
    if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Request)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY request n={} as={} gpu={:#x} device={:#x} mapped={} type={} "
                 "effective_type={} flags={:#x} payload={:#x} subreport={}",
                 trace, gpu_memory->GetID(), addr, cpu_addr_opt.value_or(0),
                 cpu_addr_opt.has_value(), static_cast<u32>(requested_type),
                 static_cast<u32>(counter_type), static_cast<u32>(flags), payload, subreport);
    }
    if (!cpu_addr_opt) [[unlikely]] {
        return;
    }
    DAddr cpu_addr = *cpu_addr_opt;
    const size_t new_query_id = streamer->WriteCounter(cpu_addr, has_timestamp, payload, subreport);
    auto* query = streamer->GetQuery(new_query_id);
    if (is_fence) {
        query->flags |= QueryFlagBits::IsFence;
    }
    QueryLocation query_location{};
    query_location.stream_id.Assign(static_cast<u32>(streamer_id));
    query_location.query_id.Assign(static_cast<u32>(new_query_id));
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    // EXP33 (zortos293/Lichii#337): a fence-flagged report stays registered until its fence runs
    // (as Balanced already does), and its fence is made and waited for even when empty
    // (g_signaling_query_report_fence). Both halves: the fence alone, with the query freed at
    // once, left a stale operation behind (three channels waiting forever on a semaphore).
    const bool keep_fence_report = is_fence && VideoCore::Nx2Debug::Experiment(33) != 0;
    const bool is_synced = !keep_fence_report &&
                           (Settings::IsGPUFenceBehaviorDefault()
                                ? !VideoCore::Nx2Debug::AccuratePart(1)
                                : !Settings::IsGPUFenceBehaviorBalanced() &&
                                      !Settings::IsGPUFenceBehaviorAccurate()) &&
                           is_fence;
    // The report's operation, small enough for std::function to hold without allocating (a report
    // per draw allocated one each, zortos293/Lichii#1161): at most 48 bytes and a nothrow move,
    // or MSVC's std::function allocates whatever the size. Built only once the report is issued.
    // It owes the report's completion, counted for Nx2AsyncReportsSettled (SEM_FAST) and
    // FlushQueryResults' own-report wait (EXP30, zortos293/Lichii#559): completed on every way
    // out of the operation, written or discarded, and when the operation is dropped without
    // running (fences discarded when emulation stops), or the slot stays unequal and a later wait
    // on it never ends. A copy or move takes the debt from its source: exactly one completes it.
    struct ReportOperation {
        enum : u8 { Synced = 1, Owed = 2 };

        QueryCacheBase* cache;
        QueryBase* query_base;
        DAddr cpu_addr;
        GPUVAddr addr;
        QueryLocation query_location;
        u8 type;         ///< QueryType
        mutable u8 bits; ///< Synced, Owed
        u16 asid;        ///< For traces only

        ReportOperation(QueryCacheBase* cache_, QueryBase* query_base_, DAddr cpu_addr_,
                        GPUVAddr addr_, QueryLocation query_location_, QueryType type_,
                        bool synced_, size_t asid_) noexcept
            : cache{cache_}, query_base{query_base_}, cpu_addr{cpu_addr_}, addr{addr_},
              query_location{query_location_}, type{static_cast<u8>(type_)},
              bits{static_cast<u8>((synced_ ? Synced : 0) | Owed)},
              asid{static_cast<u16>(asid_)} {}
        ReportOperation(const ReportOperation& other) noexcept
            : cache{other.cache}, query_base{other.query_base}, cpu_addr{other.cpu_addr},
              addr{other.addr}, query_location{other.query_location}, type{other.type},
              bits{other.bits}, asid{other.asid} {
            other.bits &= static_cast<u8>(~Owed);
        }
        ReportOperation(ReportOperation&& other) noexcept
            : ReportOperation(static_cast<const ReportOperation&>(other)) {}
        ReportOperation& operator=(const ReportOperation&) = delete;
        ReportOperation& operator=(ReportOperation&&) = delete;
        ~ReportOperation() {
            Settle();
        }

        void Settle() noexcept {
            if ((bits & Owed) != 0) {
                bits &= static_cast<u8>(~Owed);
                Tegra::Nx2CompleteAsyncReport(cpu_addr);
            }
        }

        void operator()() {
            SCOPE_EXIT {
                Settle();
            };
            // Names apart from CounterReport's own (MSVC: /we4456).
            auto& state = *cache->impl;
            const bool synced = (bits & Synced) != 0;
            const auto report_type = static_cast<QueryType>(type);
            auto* const stream = state.streamers[query_location.stream_id.Value()];
            if (True(query_base->flags & QueryFlagBits::IsInvalidated)) {
                if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Discard)) {
                    LOG_INFO(HW_GPU,
                             "DKB_QUERY discard n={} as={} gpu={:#x} device={:#x} type={} "
                             "reason=invalidated flags={:#x}",
                             trace, asid, addr, cpu_addr, static_cast<u32>(report_type),
                             static_cast<u32>(query_base->flags));
                }
                if (!synced) [[likely]] {
                    state.pending_unregister.push_back(query_location);
                }
                return;
            }
            if (False(query_base->flags & QueryFlagBits::IsFinalValueSynced)) [[unlikely]] {
                if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Discard)) {
                    LOG_INFO(HW_GPU,
                             "DKB_QUERY discard n={} as={} gpu={:#x} device={:#x} type={} "
                             "reason=unsynchronized flags={:#x}",
                             trace, asid, addr, cpu_addr, static_cast<u32>(report_type),
                             static_cast<u32>(query_base->flags));
                }
                LOG_ERROR(HW_GPU,
                          "Query report value not synchronized. Consider increasing GPU accuracy.");
                if (!synced) [[likely]] {
                    state.pending_unregister.push_back(query_location);
                }
                return;
            }
            query_base->value += stream->GetAmendValue();
            stream->SetAccumulationValue(query_base->value);
            const bool wide = True(query_base->flags & QueryFlagBits::HasTimestamp);
            cache->FinishReads(cpu_addr, wide ? 16 : 4);
            // Looked up here, not when the report was made (two pointers less to carry).
            u8* const pointer = state.device_memory.template GetPointer<u8>(cpu_addr);
            if (pointer == nullptr) [[unlikely]] {
                // Unmapped since the report: nothing to write.
                if (!synced) [[likely]] {
                    state.pending_unregister.push_back(query_location);
                }
                return;
            }
            if (report_type == QueryType::Payload) {
                // EXP13=2: marked before the word is written (nx2.h).
                Tegra::Nx2NoteLanded(cpu_addr, static_cast<u32>(query_base->value),
                                     Tegra::Nx2Landing::QueryCompletion);
            }
            if (wide) {
                const u64 ticks = state.gpu.GetTicks();
                u8* const ticks_pointer = state.device_memory.template GetPointer<u8>(cpu_addr + 8);
                if (ticks_pointer != nullptr) {
                    std::memcpy(ticks_pointer, &ticks, sizeof(ticks));
                }
                std::memcpy(pointer, &query_base->value, sizeof(query_base->value));
            } else {
                const u32 word = static_cast<u32>(query_base->value);
                std::memcpy(pointer, &word, sizeof(word));
            }
            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(cpu_addr,
                                                      static_cast<u32>(query_base->value));
            Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::QueryCompletion, cpu_addr,
                                        addr, pointer, static_cast<u32>(query_base->value));
            if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Write)) {
                LOG_INFO(HW_GPU,
                         "DKB_QUERY write n={} path=completion as={} gpu={:#x} device={:#x} "
                         "type={} bytes={} value={:#x} flags={:#x}",
                         trace, asid, addr, cpu_addr, static_cast<u32>(report_type),
                         wide ? 8 : 4,
                         wide ? query_base->value : static_cast<u32>(query_base->value),
                         static_cast<u32>(query_base->flags));
            }
            if (!synced) [[likely]] {
                state.pending_unregister.push_back(query_location);
            }
        }
    };
    static_assert(sizeof(ReportOperation) <= 48,
                  "the report operation must fit std::function's own storage");
    static_assert(std::is_nothrow_move_constructible_v<ReportOperation>,
                  "MSVC's std::function allocates a callable without a nothrow move");
    // Issues the report, then builds its operation (which completes it).
    const auto issue_operation = [&] {
        Tegra::Nx2IssueAsyncReport(cpu_addr);
        return std::function<void()>(ReportOperation{this, query, cpu_addr, addr, query_location,
                                                     counter_type, is_synced,
                                                     gpu_memory->GetID()});
    };
    if (is_fence) {
        std::function<void()> operation = issue_operation();
        VideoCore::g_signaling_query_report_fence = keep_fence_report;
        impl->rasterizer.SignalFence(std::move(operation));
        VideoCore::g_signaling_query_report_fence = false;
    } else {
        if (!VideoCore::Nx2Debug::AccuratePart(1) && counter_type == QueryType::Payload) {
            u8* pointer = impl->device_memory.template GetPointer<u8>(cpu_addr);
            u8* pointer_timestamp = impl->device_memory.template GetPointer<u8>(cpu_addr + 8);
            FinishReads(cpu_addr, has_timestamp ? 16 : 4);
            // EXP13=2: marked before the word is written (nx2.h).
            Tegra::Nx2NoteLanded(cpu_addr, payload, Tegra::Nx2Landing::QueryImmediate);
            if (has_timestamp) {
                u64 timestamp = impl->gpu.GetTicks();
                u64 value = static_cast<u64>(payload);
                std::memcpy(pointer_timestamp, &timestamp, sizeof(timestamp));
                std::memcpy(pointer, &value, sizeof(value));
            } else {
                std::memcpy(pointer, &payload, sizeof(payload));
            }
            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(cpu_addr, payload);
            Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::QueryImmediate, cpu_addr, addr,
                                        pointer, payload);
            if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Write)) {
                LOG_INFO(HW_GPU,
                         "DKB_QUERY write n={} path=immediate as={} gpu={:#x} device={:#x} type={} "
                         "bytes={} value={:#x} flags={:#x}",
                         trace, gpu_memory->GetID(), addr, cpu_addr,
                         static_cast<u32>(counter_type), has_timestamp ? 8 : 4, payload,
                         static_cast<u32>(query->flags));
            }
            streamer->Free(new_query_id);
            return;
        }
        impl->rasterizer.SyncOperation(issue_operation());
    }
    if (is_synced) {
        streamer->Free(new_query_id);
        return;
    }
    auto [cont_addr, base] = gen_caching_indexing(cpu_addr);
    {
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.try_emplace(cont_addr);
        page_filter.Mark(cpu_addr, 16);
        auto& sub_container = it1.first->second;
        auto it_current = sub_container.find(base);
        if (it_current == sub_container.end()) {
            sub_container.insert_or_assign(base, query_location);
            return;
        }
        auto* old_query = impl->ObtainQuery(it_current->second);
        old_query->flags |= QueryFlagBits::IsRewritten;
        sub_container.insert_or_assign(base, query_location);
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::UnregisterPending() {
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    std::scoped_lock lock(cache_mutex);
    for (QueryLocation loc : impl->pending_unregister) {
        const auto [streamer_id, query_id] = loc.unpack();
        auto* streamer = impl->streamers[streamer_id];
        if (!streamer) [[unlikely]] {
            continue;
        }
        auto* query = streamer->GetQuery(query_id);
        auto [cont_addr, base] = gen_caching_indexing(query->guest_address);
        auto it1 = cached_queries.find(cont_addr);
        if (it1 != cached_queries.end()) {
            auto it2 = it1->second.find(base);
            if (it2 != it1->second.end()) {
                if (it2->second.raw == loc.raw) {
                    it1->second.erase(it2);
                }
            }
        }
        streamer->Free(query_id);
    }
    impl->pending_unregister.clear();
}

template <typename Traits>
void QueryCacheBase<Traits>::NotifyWFI() {
    bool should_sync = false;
    impl->ForEachStreamer(
        [&should_sync](StreamerInterface* streamer) { should_sync |= streamer->HasPendingSync(); });
    if (!should_sync) {
        return;
    }

    impl->ForEachStreamer([](StreamerInterface* streamer) { streamer->PresyncWrites(); });
    impl->runtime.Barriers(true);
    impl->ForEachStreamer([](StreamerInterface* streamer) { streamer->SyncWrites(); });
    impl->runtime.Barriers(false);
}

template <typename Traits>
void QueryCacheBase<Traits>::NotifySegment(bool resume) {
    if (resume) {
        impl->runtime.ResumeHostConditionalRendering();
    } else {
        impl->runtime.PauseHostConditionalRendering();
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::AccelerateHostConditionalRendering() {
    bool qc_dirty = false;
    const auto gen_lookup = [this, &qc_dirty](GPUVAddr address) -> VideoCommon::LookupData {
        auto cpu_addr_opt = gpu_memory->GpuToCpuAddress(address);
        if (!cpu_addr_opt) [[unlikely]] {
            return VideoCommon::LookupData{
                .address = 0,
                .found_query = nullptr,
            };
        }
        VAddr cpu_addr = *cpu_addr_opt;
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.find(cpu_addr >> Core::DEVICE_PAGEBITS);
        if (it1 == cached_queries.end()) {
            return VideoCommon::LookupData{
                .address = cpu_addr,
                .found_query = nullptr,
            };
        }
        auto& sub_container = it1->second;
        auto it_current = sub_container.find(cpu_addr & Core::DEVICE_PAGEMASK);

        if (it_current == sub_container.end()) {
            auto it_current_2 = sub_container.find((cpu_addr & Core::DEVICE_PAGEMASK) + 4);
            if (it_current_2 == sub_container.end()) {
                return VideoCommon::LookupData{
                    .address = cpu_addr,
                    .found_query = nullptr,
                };
            }
            it_current = it_current_2;
        }
        auto* query = impl->ObtainQuery(it_current->second);
        qc_dirty |= True(query->flags & QueryFlagBits::IsHostManaged) &&
                    False(query->flags & QueryFlagBits::IsGuestSynced);
        return VideoCommon::LookupData{
            .address = cpu_addr,
            .found_query = query,
        };
    };

    auto& regs = maxwell3d->regs;
    if (regs.render_enable_override != Maxwell::Regs::RenderEnable::Override::UseRenderEnable) {
        impl->runtime.EndHostConditionalRendering();
        return false;
    }
    const ComparisonMode mode = static_cast<ComparisonMode>(regs.render_enable.mode);
    const GPUVAddr address = regs.render_enable.Address();
    switch (mode) {
    case ComparisonMode::True:
        impl->runtime.EndHostConditionalRendering();
        return false;
    case ComparisonMode::False:
        impl->runtime.EndHostConditionalRendering();
        return false;
    case ComparisonMode::Conditional: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        return impl->runtime.HostConditionalRenderingCompareValue(object_1, qc_dirty);
    }
    case ComparisonMode::IfEqual: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        VideoCommon::LookupData object_2{gen_lookup(address + 16)};
        return impl->runtime.HostConditionalRenderingCompareValues(object_1, object_2, qc_dirty,
                                                                   true);
    }
    case ComparisonMode::IfNotEqual: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        VideoCommon::LookupData object_2{gen_lookup(address + 16)};
        return impl->runtime.HostConditionalRenderingCompareValues(object_1, object_2, qc_dirty,
                                                                   false);
    }
    default:
        return false;
    }
}

// Async downloads
template <typename Traits>
void QueryCacheBase<Traits>::CommitAsyncFlushes() {
    // Make sure to have the results synced in Host.
    NotifyWFI();

    u64 mask{};
    {
        std::scoped_lock lk(impl->flush_guard);
        impl->ForEachStreamer([&mask](StreamerInterface* streamer) {
            bool local_result = streamer->HasUnsyncedQueries();
            if (local_result) {
                mask |= 1ULL << streamer->GetId();
            }
        });
        impl->flushes_pending.push_back(mask);
    }
    std::function<void()> func([this] { UnregisterPending(); });
    impl->rasterizer.SyncOperation(std::move(func));
    if (mask == 0) {
        return;
    }
    u64 ran_mask = ~mask;
    while (mask) {
        impl->ForEachStreamerIn(mask, [&mask, &ran_mask](StreamerInterface* streamer) {
            u64 dep_mask = streamer->GetDependentMask();
            if ((dep_mask & ~ran_mask) != 0) {
                return;
            }
            u64 index = streamer->GetId();
            ran_mask |= (1ULL << index);
            mask &= ~(1ULL << index);
            streamer->PushUnsyncedQueries();
        });
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::HasUncommittedFlushes() const {
    bool result = false;
    impl->ForEachStreamer([&result](StreamerInterface* streamer) {
        result |= streamer->HasUnsyncedQueries();
        return result;
    });
    return result;
}

template <typename Traits>
bool QueryCacheBase<Traits>::HasPendingFenceWork(bool ignore_payload_sync) const {
    bool result = false;
    StreamerInterface* const payload =
        ignore_payload_sync ? impl->streamers[static_cast<size_t>(QueryType::Payload)] : nullptr;
    impl->ForEachStreamer([&result, payload](StreamerInterface* streamer) {
        result |= streamer->HasUnsyncedQueries() ||
                  (streamer != payload && streamer->HasPendingSync());
        return result;
    });
    return result;
}

template <typename Traits>
void QueryCacheBase<Traits>::NotePayloadSyncDeferred() {
    if (StreamerInterface* const payload =
            impl->streamers[static_cast<size_t>(QueryType::Payload)];
        payload != nullptr) {
        payload->NoteSyncDeferred();
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::ShouldWaitAsyncFlushes() {
    std::scoped_lock lk(impl->flush_guard);
    return !impl->flushes_pending.empty() && impl->flushes_pending.front() != 0ULL;
}

template <typename Traits>
void QueryCacheBase<Traits>::PopAsyncFlushes() {
    u64 mask;
    {
        std::scoped_lock lk(impl->flush_guard);
        mask = impl->flushes_pending.front();
        impl->flushes_pending.pop_front();
    }
    if (mask == 0) {
        return;
    }
    u64 ran_mask = ~mask;
    while (mask) {
        impl->ForEachStreamerIn(mask, [&mask, &ran_mask](StreamerInterface* streamer) {
            u64 dep_mask = streamer->GetDependenceMask();
            if ((dep_mask & ~ran_mask) != 0) {
                return;
            }
            u64 index = streamer->GetId();
            ran_mask |= (1ULL << index);
            mask &= ~(1ULL << index);
            streamer->PopUnsyncedQueries();
        });
    }
}

// Invalidation

template <typename Traits>
void QueryCacheBase<Traits>::InvalidateQuery(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return;
    }
    query_base->flags |= QueryFlagBits::IsInvalidated;
}

template <typename Traits>
bool QueryCacheBase<Traits>::IsQueryDirty(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return false;
    }
    return True(query_base->flags & QueryFlagBits::IsHostManaged) &&
           False(query_base->flags & QueryFlagBits::IsGuestSynced);
}

template <typename Traits>
bool QueryCacheBase<Traits>::SemiFlushQueryDirty(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return false;
    }
    if (True(query_base->flags & QueryFlagBits::IsFinalValueSynced) &&
        False(query_base->flags & QueryFlagBits::IsGuestSynced)) {
        FinishReads(query_base->guest_address,
                    True(query_base->flags & QueryFlagBits::HasTimestamp) ? 8 : 4);
        auto* ptr = impl->device_memory.template GetPointer<u8>(query_base->guest_address);
        if (True(query_base->flags & QueryFlagBits::HasTimestamp)) {
            std::memcpy(ptr, &query_base->value, sizeof(query_base->value));
        } else {
            u32 value_l = static_cast<u32>(query_base->value);
            std::memcpy(ptr, &value_l, sizeof(value_l));
        }
        Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::QuerySemiFlush,
                                    query_base->guest_address, 0, ptr,
                                    static_cast<u32>(query_base->value));
        if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Write)) {
            const bool wide = True(query_base->flags & QueryFlagBits::HasTimestamp);
            LOG_INFO(HW_GPU,
                     "DKB_QUERY write n={} path=semi-flush device={:#x} stream={} slot={} "
                     "bytes={} value={:#x} flags={:#x}",
                     trace, query_base->guest_address, location.stream_id.Value(),
                     location.query_id.Value(), wide ? 8 : 4,
                     wide ? query_base->value : static_cast<u32>(query_base->value),
                     static_cast<u32>(query_base->flags));
        }
        return false;
    }
    return True(query_base->flags & QueryFlagBits::IsHostManaged) &&
           False(query_base->flags & QueryFlagBits::IsGuestSynced);
}

template <typename Traits>
void QueryCacheBase<Traits>::FinishReads(DAddr address, size_t size) {
    // EDEN_NX2_EXP49 (zortos293/Lichii#1028): the query cache writes results to guest memory
    // through device memory pointers, not the memory manager: land a guest read's download still
    // on its way there first, or it writes the GPU's older copy over the result later.
    // SemiFlushQueryDirty asks with cache_mutex held, which no holder of the buffer cache's lock
    // takes.
    if (VideoCore::ReadDownloadsMayCover(address, size)) [[unlikely]] {
        impl->rasterizer.FinishRead(address, size);
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::RequestGuestHostSync() {
    impl->rasterizer.ReleaseFences();
}

} // namespace VideoCommon
