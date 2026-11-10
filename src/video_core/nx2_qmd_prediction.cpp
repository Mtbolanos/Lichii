// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_qmd_prediction.h"

namespace VideoCommon::QmdPrediction {
namespace {

using Clock = std::chrono::steady_clock;

struct Prediction {
    u64 device;
    u64 position; ///< Ring position after the launch; 0 = empty
    u64 qmd;
    u64 sequence; ///< Order of the records (positions repeat within one entry)
    u32 payload;
    u32 before;   ///< EXP13=2: the word in guest memory when it was sent
    Writer writer;
    Kind kind;
    u8 op;
    bool is_signed;
    bool stale;
    bool landed;  ///< EXP13=2: its write landed (NoteLanded, marked before the word is written)
    bool matched; ///< EXP13=2: the back made it (Verify)
    Source source;
    bool simple;      ///< Record::simple
    u32 landed_value; ///< The word it landed with
    u64 read_done;
    u64 overlay_position;
    u64 block_position;
};

struct Release {
    u64 device;
    u64 sequence;
    u32 payload;
    Writer writer;
    bool reduction;
};

constexpr size_t Buckets = 512;
/// 32 ways (16 before): a counter word many QMDs add to (Ocarina's 0x86010) can have more
/// records queued than 16 while the feed lets the front run a frame ahead, and an evicted
/// record looked like a back release nothing predicted.
constexpr size_t Ways = 32;
/// A bucket grows past Ways while none of its records is done (6h (a) 4): a hot word with a
/// frame of in-flight records no longer evicts unlanded ones. Evicting starts at MaxWays.
constexpr size_t MaxWays = 256;
/// No bound on a word with more unlanded records than this (Bail::Crowded, 6h fix 3).
constexpr size_t CrowdedRecords = 24;
constexpr size_t Releases = 1024;
constexpr size_t Writers = static_cast<size_t>(Writer::Count);

/// Why an acquire got no reduction lower bound (fable_answer_6 6c, section 5).
enum class Bail : u32 {
    NoRecords,     ///< No queued reduction or max release to the word
    MirrorOff,     ///< The records are not complete for the queued work
    GpuDirty,      ///< The word's page is GPU-written
    Evicted,       ///< A record of the word's bucket was evicted while queued
    PlainQueued,   ///< A plain release to the word is queued
    PlainUnlanded, ///< A plain release to the word ran but has not landed
    OtherWriter,   ///< Another noted write to the word is queued
    Unbounded,     ///< Queued work that may write anything
    NonMonotonic,  ///< MIN, DEC, AND or XOR queued
    SignedMax,     ///< A signed MAX queued
    IncPayload,    ///< INC with a payload other than 0xffffffff queued
    Wrap,          ///< The bound would pass 0x7fffffff
    Crowded,       ///< More unlanded records on the word than CrowdedRecords (6h)
    Count,
};

/// What a computed bound answers, by compare.
enum class Outcome : u32 {
    GeqMet,
    GeqShort,
    MaskMet,
    MaskShort,
    MaskNotOr, ///< Not every record folded is OR: no answer
    EqMet,
    EqDiffers,
    EqNotExact, ///< Run but unlanded records, or the word moved since they were sent
    OtherOp,
    Count,
};

/// Acquire compares (Tegra::Nx2SemaphoreWaiters' values).
enum class Compare : u32 { Equal, Gequal, Mask, Other, Count };

Compare CompareOf(u32 operation) {
    switch (operation) {
    case Tegra::Nx2SemaphoreWaiters::AcquireEqual:
        return Compare::Equal;
    case Tegra::Nx2SemaphoreWaiters::AcquireGequal:
        return Compare::Gequal;
    case Tegra::Nx2SemaphoreWaiters::AcquireMask:
        return Compare::Mask;
    default:
        return Compare::Other;
    }
}

bool Meets(Compare compare, u32 value, u32 current) {
    switch (compare) {
    case Compare::Equal:
        return current == value;
    case Compare::Gequal:
        return current >= value;
    case Compare::Mask:
        return (current & value) != 0;
    default:
        return true;
    }
}

/// A bound violation (fable_answer_6 6d): the word went below the word folded from (an
/// unrecorded writer: a plain release or the CPU), stayed short by the last records' payloads
/// (their landings missing), or anything else.
enum class Violation : u32 { WentDown, LandingsMissing, Other, Count };

size_t Hash(u64 device) {
    return static_cast<size_t>((device * 0x9e3779b97f4a7c15ULL) >> 32);
}

size_t Bucket(u64 device) {
    return Hash(device) % Buckets;
}

/// A single-producer single-consumer ring: the producer pushes, the front drains. A full ring
/// drops the event and counts it (the feed then turns off: its tables may be wrong).
template <typename T, size_t N>
struct SpscRing {
    static_assert((N & (N - 1)) == 0);
    std::array<T, N> slots{};
    alignas(64) std::atomic<u64> head{};
    alignas(64) std::atomic<u64> tail{};
    std::atomic<u64> lost{};

    void Push(const T& value) {
        const u64 at = head.load(std::memory_order_relaxed);
        if (at - tail.load(std::memory_order_acquire) >= N) {
            lost.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        slots[at % N] = value;
        head.store(at + 1, std::memory_order_release);
    }

    template <typename Func>
    void Drain(Func&& func) {
        u64 at = tail.load(std::memory_order_relaxed);
        const u64 end = head.load(std::memory_order_acquire);
        if (at == end) {
            return;
        }
        for (; at != end; ++at) {
            func(slots[at % N]);
        }
        tail.store(at, std::memory_order_release);
    }
};

/// A release the back made (Verify), checked on the front when it drains.
struct VerifyEvent {
    u64 device;
    u64 qmd;
    u64 entry_position;
    u32 payload;
    Writer writer;
    bool reduction;
    bool has_slots;
    std::array<u32, 6> slots;
};
/// A GPU semaphore write about to land (NoteLanded), attributed on the front when it drains.
struct LandingEvent {
    u64 device;
    u32 value;
    Tegra::Nx2Landing site;
};

/// Verify runs on the back only. Landings come from the back and the fence thread(s): one ring
/// per producing thread.
constexpr size_t LandingRings = 8;
struct Rings {
    SpscRing<VerifyEvent, 8192> verify;
    std::array<SpscRing<LandingEvent, 4096>, LandingRings> landings;
    std::atomic<u32> landing_threads{};
    std::atomic<u64> landing_no_ring{};

    SpscRing<LandingEvent, 4096>* ThreadLandings() {
        thread_local SpscRing<LandingEvent, 4096>* ring = nullptr;
        thread_local bool assigned = false;
        if (!assigned) {
            assigned = true;
            const u32 index = landing_threads.fetch_add(1, std::memory_order_acq_rel);
            ring = index < LandingRings ? &landings[index] : nullptr;
        }
        return ring;
    }

    [[nodiscard]] u64 Lost() const {
        u64 lost = verify.lost.load(std::memory_order_relaxed) +
                   landing_no_ring.load(std::memory_order_relaxed);
        for (const auto& ring : landings) {
            lost += ring.lost.load(std::memory_order_relaxed);
        }
        return lost;
    }
};

Rings& GetRings() {
    static Rings rings;
    return rings;
}

/// Why the EXP13=3 feed turned itself off.
enum class FeedOff : u32 {
    None,
    BoundViolated,   ///< A word below a bound (at its newest folded landing, or once met)
    PayloadMismatch, ///< A predicted release or reduction with another payload
    Unpredicted,     ///< A back release or reduction no record predicted
    Phantom,         ///< A record the back never made
    LostEvents,      ///< A full ring dropped back releases or landings
    Count,
};
/// What met an acquire in the feed.
enum class Fed : u32 { QueuedPlain, LandedPending, Bound, Count };

constexpr std::array<const char*, static_cast<size_t>(Source::Count)> source_names{
    "not a QMD", "inline QMD", "patched inline QMD", "overlay from memory", "upload shadow",
    "memory", "memory after a wait"};

/// Provenance gating (6f (b) 3): what the feed takes predictions from until the audit is clean
/// for the others.
bool FeedSource(Source source) {
    return source == Source::Other || source == Source::InlineOverlay ||
           source == Source::PatchedOverlay || source == Source::Shadow;
}

/// For the audit at a turn-off: the word, the offending record (sequence; position, decisions
/// at or after it also count) and, when known, the value the word really got.
struct AuditKey {
    u64 device;
    u64 sequence;
    u64 position;
    bool has_made;
    u32 made;
    /// An unpredicted back release at entry `position` (6g (b)): the decisions are judged by
    /// where they stand against it, for unmatched records of `kind` before it.
    Kind kind{Kind::Plain};
    bool positional{};
    /// A bound violated (6h fix 5): judged by whether the word reaches the bound when re-read
    /// (a late landing: the decision stands), not by its value at the violation.
    bool recovery{};
};

[[nodiscard]] bool IsFeedMode(int mode) {
    return mode == 3 || mode == 4;
}

template <typename Array>
std::string Rates(const Array& counts, double window) {
    std::string text;
    for (const u64 count : counts) {
        text += fmt::format("{}{:.0f}", text.empty() ? "" : "/",
                            static_cast<double>(count) / window);
    }
    return text;
}

/// Front-owned: only the front thread touches it (the back and fence threads only push to the
/// rings).
struct State {
    std::array<std::vector<Prediction>, Buckets> predictions{};
    std::array<u32, Buckets> next{};
    /// The newest position of a record evicted from each bucket (EXP13=2's bound).
    std::array<u64, Buckets> evicted{};
    std::array<Release, Releases> releases{};
    u64 sequence{};
    u64 record_sequence{};
    bool records_valid{};
    u64 records_valid_from{};

    std::array<u64, static_cast<size_t>(Signal::Count)> signals{};
    u64 reductions{};
    u64 unmapped_releases{};
    std::array<u64, Writers> predicted{};
    u64 predicted_stale{};
    u64 predicted_at_least{};
    std::array<u64, Writers> matched{};
    std::array<u64, Writers> differs{};
    u64 differs_stale{};
    std::array<u64, Writers> unpredicted{};
    u64 unpredicted_full{};
    u64 back_reductions{};
    std::array<u64, static_cast<size_t>(Coverage::Count)> coverage{};
    std::array<u64, static_cast<size_t>(MetBy::Count)> met_by{};
    std::array<u64, static_cast<size_t>(Mirror::Count)> mirror{};
    std::array<u64, static_cast<size_t>(Bail::Count)> bails{};
    std::array<u64, static_cast<size_t>(Outcome::Count)> outcomes{};
    std::array<u64, static_cast<size_t>(Compare::Count)> on_reduction_words{};
    /// Reduction records by op: QMD unsigned, QMD signed, compute reports.
    std::array<std::array<u64, 8>, 3> reduction_ops{};
    u64 bound_checked{};
    u64 bound_violated{};
    u64 bound_unlanded{};
    u32 differs_logged{};
    u32 unpredicted_logged{};
    u32 violated_logged{};
    u64 foreign_landings{};
    u32 foreign_logged{};
    /// Why a landing fit no record: every record of the word had landed; records of its kind
    /// waited but none with its value; none of its kind waited. Per landing site (Nx2Landing).
    std::array<std::array<u64, 3>, 5> foreign_by{};
    /// The newest plain record evicted before it landed, per bucket (its device; 0 for none):
    /// the word gets no bound until a plain landing there is attributed to it.
    std::array<u64, Buckets> evicted_plain{};
    /// Records of a back-checked writer evicted before the back made them, per bucket: (device,
    /// reduction or not, how many). A back release then finding no record is that one, not a
    /// divergence.
    struct EvictedUnmatched {
        u64 device;
        bool reduction;
        u32 count;
    };
    std::array<std::array<EvictedUnmatched, 4>, Buckets> evicted_unmatched{};
    u64 unpredicted_evicted{};
    u64 evicted_unmatched_lost{};
    u64 evicted_unlanded{};
    /// Back: reductions the copy predicted (Verify matches them too).
    u64 matched_reductions{};
    u64 differs_reductions{};
    u64 unpredicted_reductions{};
    u32 unpredicted_reductions_logged{};
    /// Records the back never made, swept once it ran well past them (Note).
    u64 phantoms{};
    u32 phantoms_logged{};
    size_t sweep_bucket{};
    /// Whole-session totals of the safety check (never reset by the 5 s report).
    u64 session_checked{};
    u64 session_violated{};
    u64 session_unlanded{};
    u64 session_foreign{};
    u64 session_phantoms{};
    /// Waits on writer ranges, by (kind, word offset in the QMD, size).
    struct RangeWait {
        u32 kind;
        s64 word;
        u64 size;
        u64 count;
    };
    std::array<RangeWait, 24> range_waits{};
    size_t range_wait_count{};
    u64 range_waits_other{};
    std::array<u64, static_cast<size_t>(GuestWriterCause::Count)> wait_causes{};
    std::array<std::pair<const char*, u64>, 8> wait_whats{};
    u64 wait_whats_other{};
    std::array<u64, static_cast<size_t>(Violation::Count)> violations{};
    std::array<u64, static_cast<size_t>(Violation::Count)> recovered{};
    std::array<u64, static_cast<size_t>(Violation::Count)> never_recovered{};
    /// Violations re-read later (at each Tick and the next CheckBound on the device).
    struct PendingViolation {
        u64 device;
        const u8* pointer;
        u32 bound;
        Violation kind;
        Clock::time_point since;
    };
    std::array<PendingViolation, 64> pending{};
    size_t pending_count{};
    Clock::time_point since{};

    /// EXP13=3: the feed's state, what it met, and the bounds it met with waiting for the
    /// landing of their newest folded record (checked against its value then).
    FeedOff feed_off{FeedOff::None};
    std::string feed_off_detail;
    int last_mode{};
    u64 lost_seen{};
    /// Record first, compare after (6g (a)): the back's releases of entries sent after the last
    /// Settle that find no record wait here for the next one.
    std::array<VerifyEvent, 64> deferred{};
    size_t deferred_count{};
    u64 horizon{};      ///< Settle's position: every release before it is recorded
    u64 signal_start{}; ///< The signal the last Settle ended: [signal_start, signal_end)
    u64 signal_end{};
    Source last_read_source{};
    QmdProbe probe{};
    u32 unpredicted_detail_logged{};
    u64 deferred_events{};
    /// EXP13=3: acquires the narrow feed left for records of dependency-waiting QMDs.
    u64 not_fed_nonsimple{};
    std::array<std::array<u64, 2>, static_cast<size_t>(Fed::Count)> fed{};
    u64 fed_checks_passed{};
    u64 fed_checks_violated{};
    u64 fed_checks_unverified{};
    u64 fed_checks_full{};
    struct FeedCheck {
        u64 device;
        u64 sequence; ///< The newest folded record
        u32 bound;
        Clock::time_point since;
        const u8* pointer; ///< The word's host memory, for the re-read (may be null)
        u32 folded;
        std::array<u64, 32> sequences; ///< The folded records
        u64 evicted_before;            ///< session_evicted_unlanded at the decision
        u64 foreign_before;            ///< session_foreign at the decision
    };
    std::array<FeedCheck, 256> feed_checks{};
    size_t feed_check_count{};
    u64 fed_checks_unverifiable{};
    /// Landing IOUs (6h fix 1): an unlanded record evicted still owes its word a landing, which
    /// must not be credited to the next live record (that shifted every later landing one
    /// record early).
    struct Iou {
        u64 device;
        u64 sequence;
        u32 payload;
        Kind kind;
    };
    std::array<std::vector<Iou>, Buckets> ious{};
    u64 ious_absorbed{};
    u64 ious_lost{};
    u64 session_evicted_unlanded{};
    /// Violated bounds whose word is re-read at 1 ms and 100 ms (6h fix 5, 6).
    struct Recheck {
        u64 device;
        const u8* pointer;
        u32 bound;
        u32 landed;
        Clock::time_point since;
        bool first_done;
        AuditKey key;
    };
    std::array<Recheck, 32> rechecks{};
    size_t recheck_count{};
    u64 rechecks_recovered{};
    u64 rechecks_never{};
    u32 violation_logged{};
    /// Where the feed's plain path stopped, by gate (6h observation: plain fed 0).
    enum class PlainGate : u32 {
        NewestNotPlain, ///< The word's newest record is a reduction, or landed
        EarlierToLand,  ///< A reduction or max release before it still to land
        RanAndMoved,    ///< Run, and the word no longer holds what it held when it was sent
        ValueShort,     ///< Its value does not meet the acquire
        LaterWriter,    ///< A write noted since it (write stamps, 256-byte blocks)
        Source,         ///< Its QMD source (provenance gate)
        NotSimple,      ///< The narrow feed: not a simple record
        Count,
    };
    std::array<u64, static_cast<size_t>(PlainGate::Count)> plain_gates{};
    std::array<u64, static_cast<size_t>(Source::Count)> not_fed_source{};
    /// EXP13=3: the last 1024 feed decisions, checked once their record landed (each report)
    /// and at a turn-off.
    struct Decision {
        u64 device;
        u64 sequence; ///< The record (the newest folded one for a bound)
        u64 position;
        Clock::time_point when;
        u32 value;    ///< The acquire's
        u32 operation;
        u32 expected; ///< The record's payload, or the bound
        Fed kind;
        bool checked;
    };
    std::array<Decision, 1024> audit{};
    u64 audit_count{};
    u64 audit_checked{};
    u64 audit_wrong{};
    u64 audit_unknown{};
    u64 session_audit_checked{};
    u64 session_audit_wrong{};
    u32 audit_logged{};

    /// Re-reads the pending violations (of `device` only, if given): recovered once the word
    /// reached the bound, never recovered after 100 ms.
    void ReRead(Clock::time_point now, const u64* device) {
        for (size_t i = 0; i < pending_count;) {
            PendingViolation& entry = pending[i];
            if (device != nullptr && entry.device != *device) {
                ++i;
                continue;
            }
            u32 current{};
            std::memcpy(&current, entry.pointer, sizeof(current));
            const size_t kind = static_cast<size_t>(entry.kind);
            if (current >= entry.bound) {
                ++recovered[kind];
            } else if (now - entry.since >= std::chrono::milliseconds{100}) {
                ++never_recovered[kind];
            } else {
                ++i;
                continue;
            }
            entry = pending[--pending_count];
        }
    }

    void Tick() {
        const auto now = Clock::now();
        if (pending_count != 0) {
            ReRead(now, nullptr);
        }
        if (recheck_count != 0) {
            RunRechecks(now);
        }
        const int mode = Mode();
        if (mode != last_mode) {
            // Leaving 3 re-arms the feed; entering it starts with what was lost so far.
            last_mode = mode;
            feed_off = FeedOff::None;
            feed_off_detail.clear();
            feed_check_count = 0;
            lost_seen = GetRings().Lost();
        }
        if (since == Clock::time_point{}) {
            since = now;
            return;
        }
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        CheckLost();
        CheckAudit(now);
        // Bounds whose newest folded record never landed (dropped from the table, or far late).
        for (size_t i = 0; i < feed_check_count;) {
            if (now - feed_checks[i].since >= std::chrono::seconds{2}) {
                ++fed_checks_unverified;
                feed_checks[i] = feed_checks[--feed_check_count];
            } else {
                ++i;
            }
        }
        const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
        const auto signal = [&](Signal s) { return rate(signals[static_cast<size_t>(s)]); };
        const auto covered = [&](Coverage c) { return rate(coverage[static_cast<size_t>(c)]); };
        const auto met = [&](MetBy m) { return rate(met_by[static_cast<size_t>(m)]); };
        const auto mirrored = [&](Mirror m) { return rate(mirror[static_cast<size_t>(m)]); };
        const auto bailed = [&](Bail b) { return rate(bails[static_cast<size_t>(b)]); };
        const auto outcome = [&](Outcome o) { return rate(outcomes[static_cast<size_t>(o)]); };
        const auto on_words = [&](Compare c) {
            return rate(on_reduction_words[static_cast<size_t>(c)]);
        };
        constexpr size_t qmd = static_cast<size_t>(Writer::Qmd);
        constexpr size_t dma = static_cast<size_t>(Writer::Dma);
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13={}): {:.0f} signals/s read ({:.0f} waiting on "
                 "dependencies, {:.0f} chained, {:.0f} unmapped QMD), {:.0f} releases predicted/s "
                 "({:.0f} max releases as lower bounds, {:.0f} maybe stale QMD; not predicted: "
                 "{:.0f} reductions, {:.0f} unmapped), {:.0f} copy semaphores predicted/s; back "
                 "QMD: {:.0f} matched, {:.0f} payload differs ({:.0f} maybe stale QMD), {:.0f} "
                 "unpredicted ({:.0f} table full), {:.0f} reductions ({:.0f} matched, {:.0f} "
                 "payload differs, {:.0f} unpredicted); back copies: {:.0f} "
                 "matched, {:.0f} payload differs, {:.0f} unpredicted; unmet acquires: {:.0f} "
                 "would be met by a predicted queued release, {:.0f} wrong value, {:.0f} already "
                 "ran ({:.0f} landed pending, {:.0f} really wrong), {:.0f} no prediction, {:.0f} "
                 "newest release a puller or report one; met after fences by: {:.0f} QMD "
                 "release, {:.0f} QMD reduction, {:.0f} copy semaphore, {:.0f} other writer "
                 "({:.0f} of the met by a release run before the check), {:.0f} still unmet; "
                 "QMD copy: QMD reads/s "
                 "{:.0f} from inline QMDs, {:.0f} from memory, {:.0f} after a wait; {:.0f} "
                 "inline QMDs patched by uploads/s; {:.0f} reads/s from upload shadows ({:.0f} "
                 "shadows made, {:.0f} evicted, {:.0f} patched, {:.0f} uploads over 4 KB, "
                 "{:.0f} not linear; {:.0f} overlays from memory and a patch); {:.0f} resyncs/s, "
                 "{:.0f} starts, {:.0f} auto-off; reduction bound would meet: {:.0f} "
                 "GEQ, {:.0f} EQ, {:.0f} MASK",
                 Mode(), signal(Signal::Read), signal(Signal::Waiting), signal(Signal::Chained),
                 signal(Signal::Unmapped), rate(predicted[qmd]), rate(predicted_at_least),
                 rate(predicted_stale), rate(reductions), rate(unmapped_releases),
                 rate(predicted[dma]), rate(matched[qmd]), rate(differs[qmd]),
                 rate(differs_stale), rate(unpredicted[qmd]), rate(unpredicted_full),
                 rate(back_reductions), rate(matched_reductions), rate(differs_reductions),
                 rate(unpredicted_reductions), rate(matched[dma]), rate(differs[dma]),
                 rate(unpredicted[dma]), covered(Coverage::WouldMeet),
                 covered(Coverage::WrongValue), covered(Coverage::AlreadyRan),
                 covered(Coverage::LandedPending), covered(Coverage::ReallyWrong),
                 covered(Coverage::None), covered(Coverage::OtherRelease),
                 met(MetBy::QmdRelease), met(MetBy::QmdReduction), met(MetBy::DmaRelease),
                 met(MetBy::Other), met(MetBy::BeforeCheck), met(MetBy::Unmet),
                 mirrored(Mirror::Overlay), mirrored(Mirror::Memory), mirrored(Mirror::Waited),
                 mirrored(Mirror::Patched), mirrored(Mirror::Shadow),
                 mirrored(Mirror::ShadowCreated), mirrored(Mirror::ShadowEvicted),
                 mirrored(Mirror::ShadowPatched), mirrored(Mirror::ShadowTooBig),
                 mirrored(Mirror::ShadowNotLinear), mirrored(Mirror::OverlayFromMemory),
                 mirrored(Mirror::Resync), mirrored(Mirror::SessionStart),
                 mirrored(Mirror::AutoOff), outcome(Outcome::GeqMet), outcome(Outcome::EqMet),
                 outcome(Outcome::MaskMet));
        if (IsFeedMode(mode)) {
            const auto fed_rate = [&](Fed kind, size_t resume) {
                return rate(fed[static_cast<size_t>(kind)][resume]);
            };
            u64 fed_total = not_fed_nonsimple;
            for (const auto& by_site : fed) {
                fed_total += by_site[0] + by_site[1];
            }
            const double narrow_share =
                fed_total != 0 ? 100.0 * static_cast<double>(not_fed_nonsimple) /
                                     static_cast<double>(fed_total)
                               : 0.0;
            constexpr std::array<const char*, static_cast<size_t>(FeedOff::Count)> off_names{
                "on",
                "off: a bound violated",
                "off: a payload mismatch",
                "off: an unpredicted back release",
                "off: a phantom record",
                "off: lost events"};
            LOG_INFO(HW_GPU,
                     "QMD release prediction feed (EXP13=3): acquires/s met by a queued plain "
                     "release {:.0f} (+{:.0f} at resume polls), by a run but unlanded one {:.0f} "
                     "(+{:.0f}), by the reduction bound {:.0f} (+{:.0f}); bounds checked at their "
                     "newest landing: {} passed, {} violated, {} unverified after 2 s, {} not fed "
                     "(check table full), {} waiting; not fed for their QMD source: {} from "
                     "memory, {} after a wait, {} overlays from memory; audit: {} checked, {} "
                     "wrongly met, {} unknown (session {} checked, {} wrongly met); {} feed: "
                     "{:.0f} acquires/s left for dependency-waiting QMDs' records ({:.0f}% of "
                     "those the predictions met); {} back releases waited for their record; bounds "
                     "unverifiable (record evicted) {}; landings owed by evicted records absorbed "
                     "{}, lost {}; violations re-read: {} recovered, {} never (session); plain "
                     "path stopped (/s) by: {:.0f} newest a reduction or landed, {:.0f} an "
                     "earlier one to land, {:.0f} ran with the word moved, {:.0f} value short, "
                     "{:.0f} a later writer (stamp), {:.0f} QMD source, {:.0f} not simple; feed "
                     "{}{}",
                     fed_rate(Fed::QueuedPlain, 0), fed_rate(Fed::QueuedPlain, 1),
                     fed_rate(Fed::LandedPending, 0), fed_rate(Fed::LandedPending, 1),
                     fed_rate(Fed::Bound, 0), fed_rate(Fed::Bound, 1), fed_checks_passed,
                     fed_checks_violated, fed_checks_unverified, fed_checks_full,
                     feed_check_count, not_fed_source[static_cast<size_t>(Source::Memory)],
                     not_fed_source[static_cast<size_t>(Source::Waited)],
                     not_fed_source[static_cast<size_t>(Source::OverlayFromMemory)],
                     audit_checked, audit_wrong, audit_unknown, session_audit_checked,
                     session_audit_wrong, mode == 3 ? "narrow" : "full",
                     rate(not_fed_nonsimple), narrow_share, deferred_events,
                     fed_checks_unverifiable, ious_absorbed, ious_lost, rechecks_recovered,
                     rechecks_never, rate(plain_gates[0]), rate(plain_gates[1]),
                     rate(plain_gates[2]), rate(plain_gates[3]), rate(plain_gates[4]),
                     rate(plain_gates[5]), rate(plain_gates[6]),
                     off_names[static_cast<size_t>(feed_off)],
                     feed_off_detail.empty() ? std::string{} : " (" + feed_off_detail + ")");
        }
        if (MirrorMode()) {
            LOG_INFO(HW_GPU,
                     "QMD release prediction (EXP13=2) reduction bound: acquires on reduction "
                     "words {:.0f} GEQ, {:.0f} EQ, {:.0f} MASK, {:.0f} other; GEQ {:.0f} met, "
                     "{:.0f} short; MASK {:.0f} met, {:.0f} short, {:.0f} not all OR; EQ {:.0f} "
                     "met, {:.0f} differs, {:.0f} not exact; {:.0f} other compares; no bound: "
                     "{:.0f} no records, {:.0f} copy off, {:.0f} GPU-written, {:.0f} evicted, "
                     "{:.0f} plain queued, {:.0f} plain not landed, {:.0f} other writer, {:.0f} "
                     "unbounded writer, {:.0f} non-monotonic op, {:.0f} signed max, {:.0f} inc "
                     "payload, {:.0f} wrap guard, {:.0f} crowded word; safety: {} checked, {} "
                     "bound violated, {} not "
                     "landed yet; violations: {} went down, {} landings missing, {} other; "
                     "re-read later: {}/{}/{} recovered, {}/{}/{} never recovered in 100 ms (went "
                     "down/landings missing/other); {} landings of unrecorded writes (all "
                     "landed/no value fits/none of the kind, by site query completion {}, query "
                     "immediate {}, QMD max {}, QMD reduction {}, report reduction {}), {} "
                     "phantom records swept, {} records evicted before landing ({} back "
                     "releases met an evicted record, {} evictions untracked); session totals: "
                     "{} checked, {} violated, {} not landed yet, {} landings of unrecorded "
                     "writes, {} phantoms; records/s "
                     "by op add/min/max/inc/dec/and/or/xor: QMD {}, QMD signed {}, compute report "
                     "{}",
                     on_words(Compare::Gequal), on_words(Compare::Equal), on_words(Compare::Mask),
                     on_words(Compare::Other), outcome(Outcome::GeqMet),
                     outcome(Outcome::GeqShort), outcome(Outcome::MaskMet),
                     outcome(Outcome::MaskShort), outcome(Outcome::MaskNotOr),
                     outcome(Outcome::EqMet), outcome(Outcome::EqDiffers),
                     outcome(Outcome::EqNotExact), outcome(Outcome::OtherOp),
                     bailed(Bail::NoRecords), bailed(Bail::MirrorOff), bailed(Bail::GpuDirty),
                     bailed(Bail::Evicted), bailed(Bail::PlainQueued),
                     bailed(Bail::PlainUnlanded), bailed(Bail::OtherWriter),
                     bailed(Bail::Unbounded), bailed(Bail::NonMonotonic),
                     bailed(Bail::SignedMax), bailed(Bail::IncPayload), bailed(Bail::Wrap),
                     bailed(Bail::Crowded),
                     bound_checked, bound_violated, bound_unlanded, violations[0],
                     violations[1], violations[2], recovered[0], recovered[1], recovered[2],
                     never_recovered[0], never_recovered[1], never_recovered[2],
                     foreign_landings, Rates(foreign_by[0], 1.0), Rates(foreign_by[1], 1.0),
                     Rates(foreign_by[2], 1.0), Rates(foreign_by[3], 1.0),
                     Rates(foreign_by[4], 1.0), phantoms, evicted_unlanded,
                     unpredicted_evicted, evicted_unmatched_lost, session_checked,
                     session_violated, session_unlanded, session_foreign, session_phantoms,
                     Rates(reduction_ops[0], window),
                     Rates(reduction_ops[1], window), Rates(reduction_ops[2], window));
            const auto waited = [&](GuestWriterCause c) {
                return rate(wait_causes[static_cast<size_t>(c)]);
            };
            std::string whats;
            for (const auto& [what, count] : wait_whats) {
                if (what != nullptr && count != 0) {
                    whats += fmt::format("{}{} {:.0f}", whats.empty() ? "" : ", ", what,
                                         rate(count));
                }
            }
            if (wait_whats_other != 0) {
                whats += fmt::format("{}others {:.0f}", whats.empty() ? "" : ", ",
                                     rate(wait_whats_other));
            }
            // GpuSplit::WriterKind's order.
            constexpr std::array<const char*, 5> range_kinds{"upload", "release", "compute",
                                                             "copy", "other"};
            std::string ranges;
            for (size_t i = 0; i < range_wait_count; ++i) {
                const RangeWait& entry = range_waits[i];
                ranges += fmt::format("{}{} @{} x{} {:.0f}", ranges.empty() ? "" : ", ",
                                      entry.kind < range_kinds.size() ? range_kinds[entry.kind]
                                                                      : "?",
                                      entry.word, entry.size, rate(entry.count));
            }
            if (range_waits_other != 0) {
                ranges += fmt::format("{}others {:.0f}", ranges.empty() ? "" : ", ",
                                      rate(range_waits_other));
            }
            LOG_INFO(HW_GPU,
                     "QMD release prediction (EXP13=2) QMD reads that waited, by what set the "
                     "target (/s): {:.0f} noted write, {:.0f} writer range, {:.0f} 3D ranged "
                     "write, {:.0f} unbounded compute/copy writer, {:.0f} unranged writer, {:.0f} "
                     "queued function, {:.0f} 3D method write, {:.0f} unknown noted write ({}), "
                     "{:.0f} noted-write scan limit, {:.0f} writer-range scan limit, {:.0f} "
                     "evicted inline QMD, {:.0f} none; writer ranges waited on by (kind, word "
                     "in the QMD, bytes): {}",
                     waited(GuestWriterCause::Pending), waited(GuestWriterCause::WriterRange),
                     waited(GuestWriterCause::Ranged), waited(GuestWriterCause::UnboundedWriter),
                     waited(GuestWriterCause::Unranged), waited(GuestWriterCause::Function),
                     waited(GuestWriterCause::MethodWrite),
                     waited(GuestWriterCause::UnknownPending), whats,
                     waited(GuestWriterCause::PendingScanLimit),
                     waited(GuestWriterCause::WriterScanLimit),
                     waited(GuestWriterCause::OverlayEvicted), waited(GuestWriterCause::None),
                     ranges);
        }
        signals = {};
        reductions = unmapped_releases = predicted_stale = predicted_at_least = 0;
        differs_stale = 0;
        predicted = matched = differs = unpredicted = {};
        unpredicted_full = back_reductions = 0;
        coverage = {};
        met_by = {};
        mirror = {};
        bails = {};
        outcomes = {};
        on_reduction_words = {};
        reduction_ops = {};
        bound_checked = bound_violated = bound_unlanded = 0;
        foreign_landings = 0;
        foreign_by = {};
        evicted_unlanded = 0;
        matched_reductions = differs_reductions = unpredicted_reductions = 0;
        phantoms = 0;
        range_wait_count = 0;
        range_waits_other = 0;
        wait_causes = {};
        wait_whats = {};
        wait_whats_other = 0;
        violations = recovered = never_recovered = {};
        fed = {};
        fed_checks_passed = fed_checks_violated = fed_checks_unverified = fed_checks_full = 0;
        fed_checks_unverifiable = 0;
        ious_absorbed = ious_lost = 0;
        plain_gates = {};
        not_fed_source = {};
        audit_checked = audit_wrong = audit_unknown = 0;
        not_fed_nonsimple = 0;
        deferred_events = 0;
        unpredicted_evicted = 0;
        since = now;
    }

    /// Whether a record has served its purpose: landed and (for the writers the back checks)
    /// matched.
    [[nodiscard]] static bool Done(const Prediction& record) {
        const bool checked = record.writer == Writer::Qmd || record.writer == Writer::Dma;
        return record.landed && (record.matched || !checked);
    }

    /// Stores a record (not landed, not matched yet), in a free way, else in the oldest done
    /// one, else in the oldest.
    void Insert(const Prediction& record) {
        const size_t bucket = Bucket(record.device);
        auto& ways = predictions[bucket];
        Prediction stored = record;
        stored.sequence = ++record_sequence;
        stored.landed = false;
        stored.matched = false;
        Prediction* slot = nullptr;
        Prediction* oldest_done = nullptr;
        Prediction* oldest_unchecked = nullptr; // Of a device without pending feed checks
        Prediction* oldest = nullptr;
        for (Prediction& way : ways) {
            if (way.position == 0) {
                slot = &way;
                break;
            }
            if (Done(way) && (oldest_done == nullptr || way.sequence < oldest_done->sequence)) {
                oldest_done = &way;
            }
            if (oldest == nullptr || way.sequence < oldest->sequence) {
                oldest = &way;
            }
        }
        if (slot == nullptr && oldest_done == nullptr && ways.size() < MaxWays) {
            // Nothing done to reuse: grow (pointers into the bucket are not kept across calls).
            if (ways.capacity() == 0) {
                ways.reserve(Ways);
            }
            ways.push_back(stored);
            return;
        }
        if (slot == nullptr && oldest_done == nullptr) {
            for (Prediction& way : ways) {
                if (!HasFeedChecks(way.device) &&
                    (oldest_unchecked == nullptr || way.sequence < oldest_unchecked->sequence)) {
                    oldest_unchecked = &way;
                }
            }
        }
        if (slot == nullptr) {
            slot = oldest_done != nullptr        ? oldest_done
                   : oldest_unchecked != nullptr ? oldest_unchecked
                                                 : oldest;
        }
        if (slot->position != 0 && !Done(*slot)) {
            if (!slot->landed) {
                evicted[bucket] = (std::max)(evicted[bucket], slot->position);
                ++evicted_unlanded;
                ++session_evicted_unlanded;
                if (slot->kind == Kind::Plain) {
                    evicted_plain[bucket] = slot->device;
                }
                // It still owes the word a landing (6h fix 1), and the device's pending bound
                // checks can no longer be attributed exactly (fix 2).
                if (ious[bucket].size() < 64) {
                    ious[bucket].push_back(
                        {slot->device, slot->sequence, slot->payload, slot->kind});
                } else {
                    ++ious_lost;
                }
                DropFeedChecks(slot->device);
            }
            if (!slot->matched && (slot->writer == Writer::Qmd || slot->writer == Writer::Dma)) {
                NoteEvictedUnmatched(bucket, slot->device, slot->kind == Kind::Reduction);
            }
        }
        *slot = stored;
    }

    [[nodiscard]] bool HasFeedChecks(u64 device) const {
        for (size_t i = 0; i < feed_check_count; ++i) {
            if (feed_checks[i].device == device) {
                return true;
            }
        }
        return false;
    }

    /// A device lost an unlanded record: its pending bound checks are unverifiable.
    void DropFeedChecks(u64 device) {
        for (size_t i = 0; i < feed_check_count;) {
            if (feed_checks[i].device == device) {
                ++fed_checks_unverifiable;
                feed_checks[i] = feed_checks[--feed_check_count];
            } else {
                ++i;
            }
        }
    }

    /// A landing of `kind` and `value` on `device` that an evicted record owes, older than the
    /// live record it would otherwise go to (`before`, 0 for none): absorbed.
    bool AbsorbIou(u64 device, Kind kind, u32 value, u64 before) {
        auto& owed = ious[Bucket(device)];
        size_t found = owed.size();
        for (size_t i = 0; i < owed.size(); ++i) {
            const Iou& iou = owed[i];
            const bool fits = iou.kind == Kind::Reduction ? true
                              : iou.kind == Kind::AtLeast ? iou.payload <= value
                                                          : iou.payload == value;
            if (iou.device != device || iou.kind != kind || !fits ||
                (before != 0 && iou.sequence > before)) {
                continue;
            }
            if (found == owed.size() || iou.sequence < owed[found].sequence) {
                found = i;
            }
        }
        if (found == owed.size()) {
            return false;
        }
        owed.erase(owed.begin() + static_cast<std::ptrdiff_t>(found));
        ++ious_absorbed;
        return true;
    }

    void NoteEvictedUnmatched(size_t bucket, u64 device, bool reduction) {
        EvictedUnmatched* free_entry = nullptr;
        for (EvictedUnmatched& entry : evicted_unmatched[bucket]) {
            if (entry.count != 0 && entry.device == device && entry.reduction == reduction) {
                ++entry.count;
                return;
            }
            if (entry.count == 0 && free_entry == nullptr) {
                free_entry = &entry;
            }
        }
        if (free_entry == nullptr) {
            ++evicted_unmatched_lost; // Then a later miss counts as unpredicted
            return;
        }
        *free_entry = {device, reduction, 1};
    }

    /// A back release found no record: one evicted before the back made it, if any.
    bool TakeEvictedUnmatched(u64 device, bool reduction) {
        for (EvictedUnmatched& entry : evicted_unmatched[Bucket(device)]) {
            if (entry.count != 0 && entry.device == device && entry.reduction == reduction) {
                --entry.count;
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] u32 EvictedUnmatchedOf(u64 device) const {
        u32 count = 0;
        for (const EvictedUnmatched& entry : evicted_unmatched[Bucket(device)]) {
            count += entry.device == device ? entry.count : 0;
        }
        return count;
    }

    [[nodiscard]] static bool Landed(const Prediction& record) {
        return record.landed;
    }

    /// The records of `device` in the table and how many of them landed.
    [[nodiscard]] std::pair<u32, u32> Counts(u64 device) const {
        std::pair<u32, u32> counts{};
        for (const Prediction& way : predictions[Bucket(device)]) {
            if (way.position != 0 && way.device == device) {
                ++counts.first;
                counts.second += way.landed ? 1 : 0;
            }
        }
        return counts;
    }

    /// Drops the records of `device`.
    void ClearWord(u64 device) {
        const size_t bucket = Bucket(device);
        for (Prediction& way : predictions[bucket]) {
            if (way.device == device) {
                way = {};
            }
        }
        if (evicted_plain[bucket] == device) {
            evicted_plain[bucket] = 0;
        }
    }

    /// Records of the writers the back checks (QMD and copy releases) that it never made, once
    /// it ran well past them (`consumed` is published every 4096 words at most): phantoms. A few
    /// buckets per call. Their words start over.
    void Sweep(u64 consumed) {
        if (!records_valid || consumed < 4096) {
            return;
        }
        for (size_t step = 0; step < 4; ++step) {
            const size_t bucket = sweep_bucket++ % Buckets;
            for (Prediction& way : predictions[bucket]) {
                if (way.position == 0 || way.matched || way.position + 4096 > consumed ||
                    way.position < records_valid_from ||
                    (way.writer != Writer::Qmd && way.writer != Writer::Dma)) {
                    continue;
                }
                ++phantoms;
                ++session_phantoms;
                const AuditKey key{way.device, way.sequence, way.position, false, 0};
                TurnFeedOff(FeedOff::Phantom,
                            fmt::format("QMD {:#x}, device {:#x}, payload {:#x}, from {}",
                                        way.qmd, way.device, way.payload,
                                        source_names[static_cast<size_t>(way.source)]),
                            &key);
                if (phantoms_logged < 16) {
                    ++phantoms_logged;
                    LOG_WARNING(HW_GPU,
                                "QMD release prediction (EXP13=2): phantom {} record (the back "
                                "never made it): QMD {:#x}, device {:#x}, kind {}, payload {:#x}, "
                                "sent at {}, back at {}",
                                way.writer == Writer::Qmd ? "QMD" : "copy", way.qmd, way.device,
                                static_cast<u32>(way.kind), way.payload, way.position, consumed);
                }
                ClearWord(way.device);
            }
        }
    }

    /// Drops the records.
    void Clear() {
        // In place: the table is large (a temporary would be on the stack).
        for (auto& ways : predictions) {
            ways.clear();
        }
        for (auto& owed : ious) {
            owed.clear();
        }
        next = {};
        evicted = {};
        evicted_plain = {};
        for (auto& entries : evicted_unmatched) {
            entries.fill(EvictedUnmatched{});
        }
        pending_count = 0;
        feed_check_count = 0;
    }

    /// EXP13=3: the feed stops for the rest of the session (until EXP13 leaves 3).
    void TurnFeedOff(FeedOff reason, std::string detail, const AuditKey* key = nullptr) {
        if (!IsFeedMode(last_mode) || feed_off != FeedOff::None) {
            return;
        }
        if (key != nullptr && key->recovery) {
            LOG_ERROR(HW_GPU, "QMD release prediction feed (EXP13=3/4) audit: device {:#x} is "
                              "judged when its word is re-read (1 ms, 100 ms)",
                      key->device);
        } else {
            ScanAudit(key);
        }
        feed_off = reason;
        feed_off_detail = std::move(detail);
        constexpr std::array<const char*, static_cast<size_t>(FeedOff::Count)> reasons{
            "", "a reduction bound was violated", "a predicted release's payload differs",
            "the back made a release or reduction nothing predicted",
            "a record the back never made", "a full ring dropped events"};
        LOG_ERROR(HW_GPU,
                  "QMD release prediction feed (EXP13=3) off for the rest of the session: {} "
                  "({})",
                  reasons[static_cast<size_t>(reason)], feed_off_detail);
    }

    /// EXP13=3, at a turn-off: the decisions on the offending word since its record that the
    /// value it really got does not meet (unknown value: every such decision is suspect).
    void ScanAudit(const AuditKey* key) {
        const u64 stored = (std::min<u64>)(audit_count, audit.size());
        if (key == nullptr) {
            LOG_ERROR(HW_GPU,
                      "QMD release prediction feed (EXP13=3) audit: {} decisions in the ring, "
                      "the cause names no word",
                      stored);
            return;
        }
        if (key->positional) {
            ScanAuditPositional(*key, stored);
            return;
        }
        u32 match_count = 0;
        u32 wrong = 0;
        std::string first;
        for (u64 i = 0; i < stored; ++i) {
            const Decision& decision = audit[i];
            if (decision.device != key->device ||
                (decision.sequence != key->sequence && decision.position < key->position)) {
                continue;
            }
            ++match_count;
            const bool bad = key->has_made ? !Meets(CompareOf(decision.operation), decision.value,
                                                    key->made)
                                           : true;
            if (bad) {
                if (++wrong <= 4) {
                    first += fmt::format(" (op {} value {:#x}, expected {:#x})",
                                         decision.operation, decision.value, decision.expected);
                }
            }
        }
        LOG_ERROR(HW_GPU,
                  "QMD release prediction feed (EXP13=3) audit: {} acquires {} met on device "
                  "{:#x} (of {} decisions on it since the record){}",
                  wrong, key->has_made ? "wrongly" : "possibly wrongly", key->device, match_count,
                  first);
    }

    /// An unpredicted back release of `key.kind` at entry E = key.position (6g (b)): a missed
    /// monotonic release only makes the word higher, so the decisions stand by value; what can
    /// be wrong is timing. With no unmatched record of that kind on the word before E the word
    /// was under-modelled (safe). Otherwise a decision d with record.position <= d.position < E
    /// that folded it (d.sequence >= record.sequence) met E - d.position ring words early.
    void ScanAuditPositional(const AuditKey& key, u64 stored) {
        std::array<const Prediction*, MaxWays> early_records{};
        size_t early_count = 0;
        for (const Prediction& way : predictions[Bucket(key.device)]) {
            if (way.position != 0 && way.device == key.device && way.kind == key.kind &&
                !way.matched && way.position < key.position) {
                early_records[early_count++] = &way;
            }
        }
        u32 on_word = 0;
        u32 early = 0;
        std::string first;
        for (u64 i = 0; i < stored; ++i) {
            const Decision& decision = audit[i];
            if (decision.device != key.device) {
                continue;
            }
            ++on_word;
            for (size_t r = 0; r < early_count; ++r) {
                const Prediction& record = *early_records[r];
                if (record.position <= decision.position && decision.position < key.position &&
                    decision.sequence >= record.sequence) {
                    if (++early <= 4) {
                        first += fmt::format(" (op {} value {:#x}, {} ring words early)",
                                             decision.operation, decision.value,
                                             key.position - decision.position);
                    }
                    break;
                }
            }
        }
        if (early_count == 0) {
            LOG_ERROR(HW_GPU,
                      "QMD release prediction feed (EXP13=3/4) audit: safe, the word was "
                      "under-modelled (no unmatched record before entry {} on device {:#x}; {} "
                      "decisions on it stand)",
                      key.position, key.device, on_word);
            return;
        }
        LOG_ERROR(HW_GPU,
                  "QMD release prediction feed (EXP13=3/4) audit: {} acquires wrongly met (early) "
                  "on device {:#x} against {} unmatched records before entry {} (of {} decisions "
                  "on it){}",
                  early, key.device, early_count, key.position, on_word, first);
    }

    /// EXP13=3, each report: the decisions whose record has landed since, against the value it
    /// landed with; those whose record left the table, or did not land in 2 s, are unknown.
    void CheckAudit(Clock::time_point now) {
        const u64 stored = (std::min<u64>)(audit_count, audit.size());
        for (u64 i = 0; i < stored; ++i) {
            Decision& decision = audit[i];
            if (decision.checked) {
                continue;
            }
            const Prediction* record = nullptr;
            for (const Prediction& way : predictions[Bucket(decision.device)]) {
                if (way.position != 0 && way.sequence == decision.sequence) {
                    record = &way;
                    break;
                }
            }
            if (record == nullptr || !record->landed) {
                if (record == nullptr || now - decision.when >= std::chrono::seconds{2}) {
                    decision.checked = true;
                    ++audit_unknown;
                }
                continue;
            }
            decision.checked = true;
            ++audit_checked;
            ++session_audit_checked;
            if (Meets(CompareOf(decision.operation), decision.value, record->landed_value)) {
                continue;
            }
            ++audit_wrong;
            ++session_audit_wrong;
            if (audit_logged < 8) {
                ++audit_logged;
                LOG_ERROR(HW_GPU,
                          "QMD release prediction feed (EXP13=3) audit: acquire wrongly met on "
                          "device {:#x}: op {} value {:#x}, fed by {} {:#x}, the word landed "
                          "{:#x}",
                          decision.device, decision.operation, decision.value,
                          decision.kind == Fed::Bound ? "the bound" : "a release",
                          decision.expected, record->landed_value);
            }
        }
    }

    void Audit(const Decision& decision) {
        audit[audit_count++ % audit.size()] = decision;
    }

    /// EXP13=3: events lost to full rings make the marks and matches unreliable.
    void CheckLost() {
        const u64 lost = GetRings().Lost();
        if (lost != lost_seen) {
            TurnFeedOff(FeedOff::LostEvents, fmt::format("{} events", lost - lost_seen));
            lost_seen = lost;
        }
    }

    /// EXP13=3: `record` just landed with `value`: a bound it was the newest folded record of
    /// must hold.
    void ResolveFeedChecks(const Prediction& record, u32 value) {
        for (size_t i = 0; i < feed_check_count;) {
            FeedCheck& check = feed_checks[i];
            if (check.sequence != record.sequence || check.device != record.device) {
                ++i;
                continue;
            }
            if (value >= check.bound) {
                ++fed_checks_passed;
            } else {
                ++fed_checks_violated;
                LogViolation(check, value);
                AuditKey key{check.device, check.sequence, ~u64{0}, true, value};
                key.recovery = true;
                AddRecheck(check.device, check.pointer, check.bound, value, key);
                TurnFeedOff(FeedOff::BoundViolated,
                            fmt::format("device {:#x}: bound {:#x}, landed {:#x}", check.device,
                                        check.bound, value),
                            &key);
            }
            check = feed_checks[--feed_check_count];
        }
    }

    /// A violated bound (6h fix 5): its folded records, the bucket, what happened since.
    void LogViolation(const FeedCheck& check, u32 value) {
        if (violation_logged >= 8) {
            return;
        }
        ++violation_logged;
        const u64 back = GpuSplit::Get().ConsumedNow();
        std::string folded;
        for (u32 i = 0; i < check.folded; ++i) {
            const Prediction* record = nullptr;
            for (const Prediction& way : predictions[Bucket(check.device)]) {
                if (way.position != 0 && way.sequence == check.sequences[i]) {
                    record = &way;
                    break;
                }
            }
            if (record == nullptr) {
                folded += fmt::format(" [{} gone]", check.sequences[i]);
                continue;
            }
            folded += fmt::format(" [{} QMD {:#x} from {} sent at {} ({} the back at {}), {}]",
                                  record->sequence, record->qmd,
                                  source_names[static_cast<size_t>(record->source)],
                                  record->position,
                                  record->position <= back ? "run by" : "ahead of",
                                  back,
                                  record->landed ? fmt::format("landed {:#x}", record->landed_value)
                                                 : std::string{"not landed"});
        }
        u32 in_use = 0;
        u32 unlanded = 0;
        u32 done = 0;
        u32 others = 0;
        for (const Prediction& way : predictions[Bucket(check.device)]) {
            if (way.position == 0) {
                continue;
            }
            ++in_use;
            unlanded += way.landed ? 0 : 1;
            done += Done(way) ? 1 : 0;
            others += way.device != check.device ? 1 : 0;
        }
        LOG_ERROR(HW_GPU,
                  "QMD release prediction feed (EXP13=3/4): bound {:#x} violated on device {:#x} "
                  "(landed {:#x}, {} short) at the landing of its newest folded record, {} us "
                  "after the decision; folded:{}; bucket: {} in use, {} unlanded, {} done, {} of "
                  "other devices; since the decision {} unlanded records evicted, {} foreign "
                  "landings; the word is re-read at 1 ms and 100 ms",
                  check.bound, check.device, value, check.bound - value,
                  std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - check.since)
                      .count(),
                  folded, in_use, unlanded, done, others,
                  session_evicted_unlanded - check.evicted_before,
                  session_foreign - check.foreign_before);
    }

    void AddRecheck(u64 device, const u8* pointer, u32 bound, u32 landed, const AuditKey& key) {
        if (pointer == nullptr) {
            // Nothing to re-read: by value, as before.
            AuditKey by_value = key;
            by_value.recovery = false;
            ScanAudit(&by_value);
            return;
        }
        if (recheck_count < rechecks.size()) {
            rechecks[recheck_count++] = {device, pointer, bound, landed, Clock::now(), false, key};
        }
    }

    /// The re-reads (6h fix 6): a word that reaches the bound was a late landing, the decision
    /// stands; one that never does in 100 ms is judged by value.
    void RunRechecks(Clock::time_point now) {
        for (size_t i = 0; i < recheck_count;) {
            Recheck& entry = rechecks[i];
            const auto age = now - entry.since;
            if (age < std::chrono::milliseconds{1} ||
                (entry.first_done && age < std::chrono::milliseconds{100})) {
                ++i;
                continue;
            }
            u32 word{};
            std::memcpy(&word, entry.pointer, sizeof(word));
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(age).count();
            if (word >= entry.bound) {
                ++rechecks_recovered;
                LOG_ERROR(HW_GPU,
                          "QMD release prediction feed (EXP13=3/4) audit: device {:#x} recovered "
                          "by +{} at {} us (word {:#x}, bound {:#x}): a late landing, the "
                          "decisions stand",
                          entry.device, word - entry.landed, us, word, entry.bound);
            } else if (!entry.first_done) {
                entry.first_done = true;
                ++i;
                continue;
            } else {
                ++rechecks_never;
                LOG_ERROR(HW_GPU,
                          "QMD release prediction feed (EXP13=3/4) audit: device {:#x} never "
                          "recovered in {} us (word {:#x}, bound {:#x})",
                          entry.device, us, word, entry.bound);
                AuditKey by_value = entry.key;
                by_value.recovery = false;
                by_value.made = word;
                ScanAudit(&by_value);
            }
            entry = rechecks[--recheck_count];
        }
    }

    /// The records of `device`, in send order.
    size_t Gather(u64 device, std::array<const Prediction*, MaxWays>& records) const {
        size_t count = 0;
        for (const Prediction& way : predictions[Bucket(device)]) {
            if (way.position != 0 && way.device == device) {
                records[count++] = &way;
            }
        }
        std::sort(records.begin(), records.begin() + static_cast<std::ptrdiff_t>(count),
                  [](const Prediction* a, const Prediction* b) {
                      return a->sequence < b->sequence;
                  });
        return count;
    }

    /// The reduction lower bound (6c section 3, 6e) of the word `check` reads.
    struct Folded {
        bool ok;
        Bail bail;
        bool reduction_word; ///< Unlanded reductions or max releases wait on the word
        bool all_or;
        u64 newest;          ///< The newest folded record
        Bound bound;
    };
    Folded Fold(const AcquireCheck& check, const std::array<const Prediction*, MaxWays>& records,
                size_t count) const {
        Folded folded{};
        const auto bail = [&folded](Bail reason) {
            folded.ok = false;
            folded.bail = reason;
            return folded;
        };
        // The word only grows under the records that have not landed, whatever their position
        // (`done` trails the back: a record past it may be inside the word already, and its
        // mark says so; the back marks before it writes).
        bool any_monotonic_record = false;
        bool plain_queued = false;
        bool plain_unlanded = false;
        for (size_t i = 0; i < count; ++i) {
            if (Landed(*records[i])) {
                continue;
            }
            any_monotonic_record |= records[i]->kind != Kind::Plain;
            if (records[i]->kind == Kind::Plain) {
                (records[i]->position > check.done ? plain_queued : plain_unlanded) = true;
            }
        }
        if (!any_monotonic_record) {
            return bail(Bail::NoRecords);
        }
        folded.reduction_word = true;
        size_t unlanded = 0;
        for (size_t i = 0; i < count; ++i) {
            unlanded += Landed(*records[i]) ? 0 : 1;
        }
        if (unlanded > CrowdedRecords) {
            return bail(Bail::Crowded); // About to overflow its bucket: no bound there
        }
        const size_t bucket = Bucket(check.device);
        if (check.gpu_dirty) {
            return bail(Bail::GpuDirty);
        }
        if (!records_valid || records_valid_from > check.done) {
            return bail(Bail::MirrorOff);
        }
        if (evicted[bucket] > check.done || evicted_plain[bucket] == check.device) {
            return bail(Bail::Evicted);
        }
        if (plain_queued) {
            return bail(Bail::PlainQueued);
        }
        if (plain_unlanded) {
            // It lands later and sets the word, maybe below the bound.
            return bail(Bail::PlainUnlanded);
        }
        // Other writers of the word sent after `done`; this table's report releases are noted
        // as writer ranges at their own positions.
        std::array<u64, MaxWays> own{};
        size_t own_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (records[i]->writer == Writer::Report && records[i]->position > check.done) {
                own[own_count++] = records[i]->position;
            }
        }
        const auto writers = GpuSplit::Get().QueuedGuestWriters(
            check.done, check.address, check.device, sizeof(u32),
            std::span<const u64>{own.data(), own_count});
        if (writers.noted != 0) {
            return bail(Bail::OtherWriter);
        }
        if (writers.unbounded != 0) {
            return bail(Bail::Unbounded);
        }
        // Fold the unlanded records in send order from the word read before the marks were
        // drained (a record marked after the read is folded, one marked before is in the
        // word). The 2^31 guard keeps the sum from wrapping.
        constexpr u64 Guard = 0x7fffffff;
        u64 bound = check.word;
        folded.all_or = true;
        if (bound > Guard) {
            return bail(Bail::Wrap);
        }
        Bound& result = folded.bound;
        for (size_t i = 0; i < count; ++i) {
            const Prediction& r = *records[i];
            if (Landed(r)) {
                continue;
            }
            folded.newest = r.sequence;
            result.sequences[result.folded] = r.sequence;
            result.payloads[result.folded++] = r.payload;
            result.payload_sum += r.payload;
            if (r.kind == Kind::AtLeast) {
                bound = (std::max)(bound, u64{r.payload});
                folded.all_or = false;
            } else {
                switch (r.op) {
                case 0: // ADD
                    bound += r.payload;
                    folded.all_or = false;
                    break;
                case 2: // MAX
                    if (r.is_signed) {
                        return bail(Bail::SignedMax);
                    }
                    bound = (std::max)(bound, u64{r.payload});
                    folded.all_or = false;
                    break;
                case 3: // INC: value >= payload ? 0 : value + 1
                    if (r.payload != 0xffffffff) {
                        return bail(Bail::IncPayload);
                    }
                    bound += 1;
                    folded.all_or = false;
                    break;
                case 6: // OR
                    bound |= r.payload;
                    break;
                default: // MIN, DEC, AND, XOR
                    return bail(Bail::NonMonotonic);
                }
            }
            if (bound > Guard) {
                return bail(Bail::Wrap);
            }
        }
        folded.ok = true;
        result.value = static_cast<u32>(bound);
        result.word = check.word;
        result.device = check.device;
        result.address = check.address;
        result.done = check.done;
        result.since = Clock::now();
        result.pointer = check.pointer;
        result.operation = check.operation;
        result.acquire_value = check.value;
        return folded;
    }
};

State& Get() {
    static State state;
    return state;
}

void ApplyLanded(State& state, u64 device, u32 value, Tegra::Nx2Landing site) {
    using Site = Tegra::Nx2Landing;
    const Kind kind = site == Site::QmdMax ? Kind::AtLeast
                      : site == Site::QmdReduction || site == Site::ReportReduction
                          ? Kind::Reduction
                          : Kind::Plain;
    // The record it belongs to: reductions land in send order among themselves (synchronously,
    // in back order), a plain write is known by its payload, a max by a payload up to it. Kinds
    // land out of order with each other (query completions run on the fencing thread).
    Prediction* oldest = nullptr;
    bool any = false;
    bool any_unlanded = false;
    bool kind_unlanded = false;
    const size_t bucket = Bucket(device);
    for (Prediction& way : state.predictions[bucket]) {
        if (way.position == 0 || way.device != device) {
            continue;
        }
        any = true;
        if (way.landed) {
            continue;
        }
        any_unlanded = true;
        if (way.kind != kind) {
            continue;
        }
        kind_unlanded = true;
        const bool fits = kind == Kind::Reduction ? true
                          : kind == Kind::AtLeast ? way.payload <= value
                                                  : way.payload == value;
        if (fits && (oldest == nullptr || way.sequence < oldest->sequence)) {
            oldest = &way;
        }
    }
    // A landing an evicted record owes, older than the live one (6h fix 1).
    if (!state.ious[bucket].empty() &&
        state.AbsorbIou(device, kind, value, oldest != nullptr ? oldest->sequence : 0)) {
        return;
    }
    if (oldest != nullptr) {
        oldest->landed = true;
        oldest->landed_value = value;
        if (state.feed_check_count != 0) {
            state.ResolveFeedChecks(*oldest, value);
        }
        return;
    }
    if (kind == Kind::Plain && state.evicted_plain[bucket] == device) {
        state.evicted_plain[bucket] = 0; // The evicted plain record's, most likely
        return;
    }
    if (!any) {
        return; // A word without records (counter reports, CPU-polled words)
    }
    ++state.foreign_landings;
    ++state.session_foreign;
    const size_t reason = !any_unlanded ? 0 : kind_unlanded ? 1 : 2;
    const size_t site_index = static_cast<size_t>(site);
    if (site_index < state.foreign_by.size()) {
        ++state.foreign_by[site_index][reason];
    }
    if (state.foreign_logged < 8) {
        ++state.foreign_logged;
        std::string kinds;
        for (const Prediction& way : state.predictions[bucket]) {
            if (way.position != 0 && way.device == device) {
                kinds += fmt::format(" {}{}={:#x}{}",
                                     way.kind == Kind::Plain     ? "plain"
                                     : way.kind == Kind::AtLeast ? "max"
                                                                 : "reduction",
                                     way.writer == Writer::Qmd      ? ""
                                     : way.writer == Writer::Dma    ? "(copy)"
                                     : way.writer == Writer::Puller ? "(puller)"
                                                                    : "(report)",
                                     way.payload, way.landed ? " landed" : "");
            }
        }
        constexpr std::array<const char*, 5> sites{"query completion", "query immediate",
                                                   "QMD max", "QMD reduction",
                                                   "report reduction"};
        constexpr std::array<const char*, 3> reasons{
            "every record of the word had landed", "no waiting record of its kind has its value",
            "no record of its kind waits"};
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13=2): a landing no record fits ({}): device {:#x}, "
                 "{} {:#x} through the {} path; the word's records:{}",
                 reasons[reason], device,
                 kind == Kind::Plain     ? "plain"
                 : kind == Kind::AtLeast ? "max"
                                         : "reduction",
                 value, site_index < sites.size() ? sites[site_index] : "?", kinds);
    }
}

/// `retry`: a deferred event applied again (its first pass counted it already); `last_try`: no
/// more deferring.
void ApplyVerify(State& state, const VerifyEvent& event, bool retry, bool last_try) {
    const Writer writer = event.writer;
    const u64 device = event.device;
    const u32 payload = event.payload;
    const u64 qmd = event.qmd;
    const u64 entry_position = event.entry_position;
    const bool reduction = event.reduction;
    // Made by an entry sent after the records were complete: nothing predicted it is a
    // divergence (EXP13=3 turns the feed off).
    const bool covered = state.records_valid && entry_position >= state.records_valid_from;
    const bool mirror = MirrorMode();
    if (!retry) {
        state.releases[Hash(device) % Releases] = {device, ++state.sequence, payload, writer,
                                                     reduction};
        if (reduction) {
            ++state.back_reductions;
        }
    }
    if (reduction && !mirror) {
        return;
    }
    // The launch the back runs is the first one sent after the entry it runs starts. With
    // EXP13=2 each record is the prediction of one release (matched once).
    auto& ways = state.predictions[Bucket(device)];
    Prediction* found = nullptr;
    u32 same_device = 0;
    u64 oldest = ~u64{0};
    for (Prediction& way : ways) {
        oldest = (std::min)(oldest, way.position);
        if (way.position <= entry_position || way.device != device || way.writer != writer ||
            (way.kind == Kind::Reduction) != reduction || (mirror && way.matched)) {
            continue;
        }
        if (way.qmd != qmd) {
            ++same_device;
            continue;
        }
        if (!found || way.position < found->position ||
            (way.position == found->position && way.payload == payload)) {
            found = &way;
        }
    }
    // The copy may not have recorded it yet (6g (a)): an entry after the last Settle waits for
    // the next one.
    if (found == nullptr && !last_try && state.records_valid && entry_position >= state.horizon &&
        state.deferred_count < state.deferred.size()) {
        state.deferred[state.deferred_count++] = event;
        state.deferred_events += retry ? 0 : 1;
        return;
    }
    const auto log_unpredicted = [&] {
        if (state.unpredicted_detail_logged >= 16) {
            return;
        }
        ++state.unpredicted_detail_logged;
        const bool inside =
            entry_position >= state.signal_start && entry_position < state.signal_end;
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13=2-4): unpredicted {} at entry {} (QMD {:#x}, "
                 "device {:#x}, payload {:#x}): the copy's last read was from {}; inside the "
                 "signal last recorded ({}..{}): {}; the back's release slots at run "
                 "(flags/payload): {}; records of the device evicted unmade: {}; the copy: {}",
                 reduction ? "reduction" : "release", entry_position, qmd, device, payload,
                 source_names[static_cast<size_t>(state.last_read_source)], state.signal_start,
                 state.signal_end, inside ? "yes" : "no",
                 event.has_slots
                     ? fmt::format("{:#x}/{:#x} {:#x}/{:#x} {:#x}/{:#x}", event.slots[0],
                                   event.slots[1], event.slots[2], event.slots[3],
                                   event.slots[4], event.slots[5])
                     : std::string{"none"},
                 state.EvictedUnmatchedOf(device),
                 state.probe != nullptr ? state.probe(qmd) : std::string{"?"});
    };
    // A record evicted before the back made it (a deep backlog on a hot word): not a
    // divergence. The fold missed it, which only lowered the bound.
    if (found == nullptr && state.TakeEvictedUnmatched(device, reduction)) {
        ++state.unpredicted_evicted;
        u64& unpredicted = reduction ? state.unpredicted_reductions
                                     : state.unpredicted[static_cast<size_t>(writer)];
        ++unpredicted;
        return;
    }
    const size_t index = static_cast<size_t>(writer);
    if (reduction) {
        if (!found) {
            ++state.unpredicted_reductions;
            log_unpredicted();
            if (covered) {
                const AuditKey key{device, 0, entry_position, false, 0, Kind::Reduction, true};
                state.TurnFeedOff(FeedOff::Unpredicted,
                                  fmt::format("reduction of QMD {:#x}, device {:#x}, payload {:#x}",
                                              qmd, device, payload),
                                  &key);
            }
            if (state.unpredicted_reductions_logged < 8) {
                ++state.unpredicted_reductions_logged;
                LOG_INFO(HW_GPU,
                         "QMD release prediction (EXP13=2): unpredicted QMD reduction, QMD "
                         "{:#x}, device {:#x} payload {:#x}, entry {}; queued predictions there "
                         "from other QMDs: {}",
                         qmd, device, payload, entry_position, same_device);
            }
        } else {
            found->matched = true;
            ++(found->payload == payload ? state.matched_reductions : state.differs_reductions);
            if (found->payload != payload) {
                const AuditKey key{device, found->sequence, found->position, false, 0};
                state.TurnFeedOff(FeedOff::PayloadMismatch,
                                  fmt::format("reduction of QMD {:#x}, device {:#x}: predicted "
                                              "{:#x} from {}, made {:#x}",
                                              qmd, device, found->payload,
                                              source_names[static_cast<size_t>(found->source)],
                                              payload),
                                  &key);
            }
        }
        return;
    }
    if (found) {
        found->matched = true;
    }
    if (!found) {
        ++state.unpredicted[index];
        log_unpredicted();
        if (covered && (writer == Writer::Qmd || writer == Writer::Dma)) {
            const AuditKey key{device, 0, entry_position, true, payload, Kind::Plain, true};
            state.TurnFeedOff(FeedOff::Unpredicted,
                              fmt::format("{} release, QMD {:#x}, device {:#x} = {:#x}",
                                          writer == Writer::Qmd ? "QMD" : "copy", qmd, device,
                                          payload),
                              &key);
        }
        // Every way was predicted after this entry: the prediction may have been evicted.
        state.unpredicted_full += oldest > entry_position && writer == Writer::Qmd ? 1 : 0;
        if (state.unpredicted_logged < 8) {
            ++state.unpredicted_logged;
            LOG_INFO(HW_GPU,
                     "QMD release prediction (EXP13): unpredicted {} release, QMD {:#x}, device "
                     "{:#x} = {:#x}, entry {}; queued predictions there from other QMDs: {}",
                     writer == Writer::Qmd ? "QMD" : "copy", qmd, device, payload, entry_position,
                     same_device);
        }
        return;
    }
    if (found->payload == payload) {
        ++state.matched[index];
        return;
    }
    ++state.differs[index];
    state.differs_stale += found->stale ? 1 : 0;
    const AuditKey key{device, found->sequence, found->position, true, payload};
    state.TurnFeedOff(FeedOff::PayloadMismatch,
                      fmt::format("{} release, QMD {:#x}, device {:#x}: predicted {:#x} from {}, "
                                  "made {:#x}",
                                  writer == Writer::Qmd ? "QMD" : "copy", qmd, device,
                                  found->payload, source_names[static_cast<size_t>(found->source)],
                                  payload),
                      &key);
    if (state.differs_logged < 16) {
        ++state.differs_logged;
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13): {} release payload differs, QMD {:#x}, device "
                 "{:#x}: predicted {:#x}, written {:#x}; QMD maybe stale {}, read from {} with "
                 "the back at {} (inline QMD at {}, upload shadow block at {}), sent at {}, run "
                 "at {}; the back's release slots (flags/payload): {}",
                 writer == Writer::Qmd ? "QMD" : "copy", qmd, device, found->payload, payload,
                 found->stale, source_names[static_cast<size_t>(found->source)],
                 found->read_done, found->overlay_position, found->block_position,
                 found->position, entry_position,
                 event.has_slots
                     ? fmt::format("{:#x}/{:#x} {:#x}/{:#x} {:#x}/{:#x}", event.slots[0],
                                   event.slots[1], event.slots[2], event.slots[3],
                                   event.slots[4], event.slots[5])
                     : std::string{"none"});
    }
}

/// The back's releases and the landings pushed since the last drain, applied in order.
void Drain(State& state) {
    auto& rings = GetRings();
    rings.verify.Drain(
        [&state](const VerifyEvent& event) { ApplyVerify(state, event, false, false); });
    const u32 threads =
        (std::min)(rings.landing_threads.load(std::memory_order_acquire), u32{LandingRings});
    for (u32 i = 0; i < threads; ++i) {
        rings.landings[i].Drain([&state](const LandingEvent& event) {
            ApplyLanded(state, event.device, event.value, event.site);
        });
    }
}

/// The front's state, with what the other threads pushed applied.
State& Front() {
    State& state = Get();
    Drain(state);
    return state;
}

} // Anonymous namespace

void NoteSignal(Signal signal) {
    auto& state = Front();
    ++state.signals[static_cast<size_t>(signal)];
    state.Tick();
}

void NoteReduction() {
    auto& state = Get();
    ++state.reductions;
}

void NoteUnmappedRelease() {
    auto& state = Get();
    ++state.unmapped_releases;
}

void Predict(Writer writer, u64 device, u32 payload, u64 position, u64 qmd, bool stale) {
    // Record first: the back's release of it is compared at the next drain.
    auto& state = Get();
    Prediction record{};
    record.device = device;
    record.position = position;
    record.qmd = qmd;
    record.payload = payload;
    record.writer = writer;
    record.kind = Kind::Plain;
    record.stale = stale;
    state.Insert(record);
    ++state.predicted[static_cast<size_t>(writer)];
    state.predicted_stale += stale ? 1 : 0;
    state.Tick();
}

void Note(u64 device, const Record& record, u32 before, u64 position, u64 qmd) {
    // Record first, compare after (6g (a)): no drain here (Settle, after the signal).
    auto& state = Get();
    Prediction stored{};
    stored.device = device;
    stored.position = position;
    stored.qmd = qmd;
    stored.payload = record.payload;
    stored.before = before;
    stored.writer = record.writer;
    stored.kind = record.kind;
    stored.op = static_cast<u8>(record.op & 7);
    stored.is_signed = record.is_signed;
    stored.source = record.source;
    stored.read_done = record.read_done;
    stored.overlay_position = record.overlay_position;
    stored.block_position = record.block_position;
    stored.simple = record.simple;
    static const bool hooked = [] {
        Tegra::g_nx2_landed_hook.store(&NoteLanded, std::memory_order_release);
        return true;
    }();
    (void)hooked;
    state.Insert(stored);
    switch (record.kind) {
    case Kind::Reduction:
        if (record.writer == Writer::Report) {
            ++state.reduction_ops[2][record.op & 7];
        } else {
            ++state.reductions;
            ++state.reduction_ops[record.is_signed ? 1 : 0][record.op & 7];
        }
        break;
    case Kind::AtLeast:
        ++state.predicted_at_least;
        [[fallthrough]];
    case Kind::Plain:
        ++state.predicted[static_cast<size_t>(record.writer)];
        break;
    }
    state.Tick();
}

void Settle(u64 position) {
    // The back's position first: every release of the entries it ran is in the rings then.
    const u64 consumed = GpuSplit::Get().ConsumedNow();
    auto& state = Front();
    state.signal_start = state.horizon;
    state.signal_end = position;
    state.horizon = (std::max)(state.horizon, position);
    if (state.deferred_count != 0) {
        std::array<VerifyEvent, 64> events = state.deferred;
        const size_t count = state.deferred_count;
        state.deferred_count = 0;
        for (size_t i = 0; i < count; ++i) {
            ApplyVerify(state, events[i], true, events[i].entry_position < state.horizon);
        }
    }
    state.Sweep(consumed);
}

void NoteReadSource(Source source) {
    Get().last_read_source = source;
}

void SetQmdProbe(QmdProbe probe) {
    Get().probe = probe;
}

void NoteMirror(Mirror event) {
    auto& state = Get();
    ++state.mirror[static_cast<size_t>(event)];
}

void SetRecordsValid(bool valid, u64 position) {
    auto& state = Front();
    state.records_valid = valid;
    state.records_valid_from = position;
    if (!valid) {
        // The copy stopped: its records and their landed counts start over with it.
        state.Clear();
    }
}

void NoteWait(GuestWriterCause cause, const char* what) {
    auto& state = Get();
    ++state.wait_causes[static_cast<size_t>(cause)];
    if (cause != GuestWriterCause::UnknownPending) {
        return;
    }
    for (auto& [name, count] : state.wait_whats) {
        if (name == nullptr || name == what ||
            (what != nullptr && std::strcmp(name, what) == 0)) {
            name = what != nullptr ? what : "unknown";
            ++count;
            return;
        }
    }
    ++state.wait_whats_other;
}

void NoteRangeWait(u32 kind, s64 offset, u64 size) {
    auto& state = Get();
    const s64 word = offset >= 0 ? offset / 4 : -((-offset + 3) / 4);
    for (size_t i = 0; i < state.range_wait_count; ++i) {
        State::RangeWait& entry = state.range_waits[i];
        if (entry.kind == kind && entry.word == word && entry.size == size) {
            ++entry.count;
            return;
        }
    }
    if (state.range_wait_count < state.range_waits.size()) {
        state.range_waits[state.range_wait_count++] = {kind, word, size, 1};
    } else {
        ++state.range_waits_other;
    }
}

void NoteLanded(u64 device, u32 value, Tegra::Nx2Landing site) {
    if (!MirrorMode()) {
        return;
    }
    auto& rings = GetRings();
    if (auto* const ring = rings.ThreadLandings()) {
        ring->Push({device, value, site});
    } else {
        rings.landing_no_ring.fetch_add(1, std::memory_order_relaxed);
    }
    // The caller writes the word next: a front that reads the new word must find the event.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Verify(Writer writer, u64 device, u32 payload, u64 qmd, u64 entry_position, bool reduction,
            const u32* slots) {
    VerifyEvent event{device, qmd, entry_position, payload, writer, reduction, slots != nullptr,
                      {}};
    if (slots != nullptr) {
        std::copy_n(slots, event.slots.size(), event.slots.begin());
    }
    GetRings().verify.Push(event);
}

Queued QueuedPayloads(u64 device, u64 consumed) {
    auto& state = Front();
    Queued queued{};
    for (const Prediction& way : state.predictions[Bucket(device)]) {
        if (way.position == 0 || way.device != device || way.kind == Kind::Reduction) {
            continue;
        }
        queued.any = true;
        if (way.position > consumed && queued.count < queued.payloads.size()) {
            queued.payloads[queued.count++] = way.payload;
        }
    }
    return queued;
}

void NoteAcquire(Coverage coverage) {
    auto& state = Front();
    ++state.coverage[static_cast<size_t>(coverage)];
    state.Tick();
}

std::optional<Bound> CheckAcquire(const AcquireCheck& check) {
    auto& state = Front();
    const Compare compare = CompareOf(check.operation);
    // The word's records, in send order.
    std::array<const Prediction*, MaxWays> records{};
    const size_t count = state.Gather(check.device, records);

    // Coverage by the plain predictions: the newest release recorded for the word decides.
    const auto would_meet = [&](const Prediction& r) {
        return r.kind == Kind::AtLeast
                   ? compare == Compare::Gequal && check.value <= r.payload
                   : Meets(compare, check.value, r.payload);
    };
    Coverage coverage = Coverage::None;
    if (count != 0 && (records[count - 1]->writer == Writer::Puller ||
                       (records[count - 1]->writer == Writer::Report &&
                        records[count - 1]->kind == Kind::Plain))) {
        coverage = Coverage::OtherRelease;
    } else if (count != 0 && records[count - 1]->kind != Kind::Reduction) {
        const Prediction& newest = *records[count - 1];
        if (newest.position > check.done) {
            coverage = would_meet(newest) ? Coverage::WouldMeet : Coverage::WrongValue;
        } else if (!state.Landed(newest) && would_meet(newest) &&
                   (check.word == newest.before || check.word == newest.payload)) {
            coverage = Coverage::LandedPending;
        } else {
            coverage = Coverage::ReallyWrong;
        }
    }
    ++state.coverage[static_cast<size_t>(coverage)];
    if (coverage == Coverage::LandedPending || coverage == Coverage::ReallyWrong) {
        ++state.coverage[static_cast<size_t>(Coverage::AlreadyRan)];
    }
    state.Tick();

    // The reduction lower bound (6c section 3, 6e).
    const State::Folded folded = state.Fold(check, records, count);
    if (folded.reduction_word) {
        ++state.on_reduction_words[static_cast<size_t>(compare)];
    }
    if (!folded.ok) {
        ++state.bails[static_cast<size_t>(folded.bail)];
        return std::nullopt;
    }
    const bool all_or = folded.all_or;
    Bound result = folded.bound;
    const u32 value = result.value;
    Outcome outcome = Outcome::OtherOp;
    switch (compare) {
    case Compare::Gequal:
        outcome = value >= check.value ? Outcome::GeqMet : Outcome::GeqShort;
        break;
    case Compare::Mask:
        outcome = !all_or                       ? Outcome::MaskNotOr
                  : (value & check.value) != 0 ? Outcome::MaskMet
                                               : Outcome::MaskShort;
        break;
    case Compare::Equal: {
        // With the marks every write still to land is folded (plain ones bailed above): the
        // fold of monotonic ops is the word's final value, unless an unrecorded writer joins.
        outcome = value == check.value ? Outcome::EqMet : Outcome::EqDiffers;
        break;
    }
    default:
        break;
    }
    ++state.outcomes[static_cast<size_t>(outcome)];
    return result;
}

bool Feed(const AcquireCheck& check, bool resume) {
    auto& state = Front();
    state.Tick();
    state.CheckLost();
    if (!IsFeedMode(state.last_mode) || state.feed_off != FeedOff::None || !state.records_valid ||
        state.records_valid_from > check.done || check.gpu_dirty) {
        return false;
    }
    std::array<const Prediction*, MaxWays> records{};
    const size_t count = state.Gather(check.device, records);
    if (count == 0) {
        return false;
    }
    const Compare compare = CompareOf(check.operation);
    const size_t resumed = resume ? 1 : 0;
    const bool narrow = state.last_mode == 3;
    // A plain or max release, the word's newest record: queued, or run with its landing still to
    // come (the word holds what it held when it was sent), with no reduction or max release
    // still to land before it and nothing queued after it that may write the word (6b slice B,
    // QueuedReleasePayload's miss 3).
    const Prediction& newest = *records[count - 1];
    using PlainGate = State::PlainGate;
    const auto stop = [&state](PlainGate gate) {
        ++state.plain_gates[static_cast<size_t>(gate)];
    };
    if (newest.kind == Kind::Reduction || newest.landed) {
        stop(PlainGate::NewestNotPlain);
    } else {
        bool earlier_monotonic = false;
        for (size_t i = 0; i + 1 < count; ++i) {
            earlier_monotonic |= !records[i]->landed && records[i]->kind != Kind::Plain;
        }
        const bool queued = newest.position > check.done;
        const bool pending = !queued && check.word == newest.before;
        const bool meets = newest.kind == Kind::AtLeast
                               ? compare == Compare::Gequal && check.value <= newest.payload
                               : Meets(compare, check.value, newest.payload);
        if (earlier_monotonic) {
            stop(PlainGate::EarlierToLand);
        } else if (!queued && !pending) {
            stop(PlainGate::RanAndMoved);
        } else if (!meets) {
            stop(PlainGate::ValueShort);
        } else if (GpuSplit::Get().WriteStamp(check.device, sizeof(u32), check.address) >
                   newest.position) {
            // Write stamps: a write the front noted since the release may have reached it.
            stop(PlainGate::LaterWriter);
        } else if (!FeedSource(newest.source)) {
            stop(PlainGate::Source);
            ++state.not_fed_source[static_cast<size_t>(newest.source)];
            return false;
        } else if (narrow && newest.writer == Writer::Qmd && !newest.simple) {
            stop(PlainGate::NotSimple);
            ++state.not_fed_nonsimple;
            return false;
        } else {
            const Fed kind = queued ? Fed::QueuedPlain : Fed::LandedPending;
            ++state.fed[static_cast<size_t>(kind)][resumed];
            state.Audit({check.device, newest.sequence, newest.position, Clock::now(),
                         check.value, check.operation, newest.payload, kind, false});
            return true;
        }
    }
    // The reduction lower bound, GEQ only (6c section 4), checked when its newest folded record
    // lands (6c section 6).
    if (compare != Compare::Gequal) {
        return false;
    }
    const State::Folded folded = state.Fold(check, records, count);
    if (!folded.ok || folded.bound.value < check.value) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!records[i]->landed && !FeedSource(records[i]->source)) {
            ++state.not_fed_source[static_cast<size_t>(records[i]->source)];
            return false;
        }
    }
    // The narrow feed: an unlanded record of a dependency-waiting QMD on the word, whose timing
    // is the uncertain thing, means no feed.
    for (size_t i = 0; narrow && i < count; ++i) {
        if (!records[i]->landed && records[i]->writer == Writer::Qmd && !records[i]->simple) {
            ++state.not_fed_nonsimple;
            return false;
        }
    }
    if (state.feed_check_count == state.feed_checks.size()) {
        ++state.fed_checks_full;
        return false;
    }
    State::FeedCheck& fed_check = state.feed_checks[state.feed_check_count++];
    fed_check = {check.device, folded.newest, folded.bound.value, Clock::now(), check.pointer,
                 folded.bound.folded, {}, state.session_evicted_unlanded, state.session_foreign};
    std::copy_n(folded.bound.sequences.begin(), folded.bound.folded, fed_check.sequences.begin());
    ++state.fed[static_cast<size_t>(Fed::Bound)][resumed];
    const Prediction* newest_folded = nullptr;
    for (size_t i = 0; i < count; ++i) {
        if (records[i]->sequence == folded.newest) {
            newest_folded = records[i];
        }
    }
    state.Audit({check.device, folded.newest,
                 newest_folded != nullptr ? newest_folded->position : 0, Clock::now(),
                 check.value, check.operation, folded.bound.value, Fed::Bound, false});
    return true;
}

void CheckBound(const Bound& bound, u32 word) {
    auto& state = Front();
    const auto now = Clock::now();
    if (state.pending_count != 0) {
        state.ReRead(now, &bound.device);
    }
    // Meaningful once every folded record landed (a record dropped since counts as not).
    u32 folded_landed = 0;
    for (u32 i = 0; i < bound.folded; ++i) {
        for (const Prediction& way : state.predictions[Bucket(bound.device)]) {
            if (way.position != 0 && way.sequence == bound.sequences[i]) {
                folded_landed += way.landed ? 1 : 0;
                break;
            }
        }
    }
    if (folded_landed < bound.folded) {
        ++state.bound_unlanded;
        ++state.session_unlanded;
        return;
    }
    ++state.bound_checked;
    ++state.session_checked;
    if (word >= bound.value) {
        return;
    }
    ++state.bound_violated;
    ++state.session_violated;
    AuditKey key{bound.device, bound.sequences[bound.folded - 1], ~u64{0}, true, word};
    key.recovery = true;
    state.AddRecheck(bound.device, bound.pointer, bound.value, word, key);
    state.TurnFeedOff(FeedOff::BoundViolated,
                      fmt::format("device {:#x}: bound {:#x}, word {:#x} once met", bound.device,
                                  bound.value, word),
                      &key);
    const auto [word_records, word_landed] = state.Counts(bound.device);
    const u32 deficit = bound.value - word;
    Violation kind = Violation::Other;
    if (word < bound.word) {
        kind = Violation::WentDown;
    } else {
        // Short by the payloads of the last j records folded: their landings are missing.
        u64 sum = 0;
        for (u32 j = bound.folded; j > 0; --j) {
            sum += bound.payloads[j - 1];
            if (sum == deficit) {
                kind = Violation::LandingsMissing;
                break;
            }
        }
    }
    ++state.violations[static_cast<size_t>(kind)];
    if (state.violated_logged < 32) {
        ++state.violated_logged;
        constexpr std::array<const char*, 3> kinds{"went down", "landings missing", "other"};
        LOG_WARNING(HW_GPU,
                    "QMD release prediction (EXP13=2): reduction bound violated ({}) at {:#x} "
                    "(device {:#x}): acquire op {} value {:#x}; folded from {:#x} at back "
                    "position {}: bound {:#x}, word {:#x} once met, deficit {:#x}; {} records "
                    "folded, payloads summing to {:#x}; records of the word: {}, {} landed; {} "
                    "us after the acquire",
                    kinds[static_cast<size_t>(kind)], bound.address, bound.device,
                    bound.operation, bound.acquire_value, bound.word, bound.done, bound.value,
                    word, deficit, bound.folded, bound.payload_sum, word_records, word_landed,
                    std::chrono::duration_cast<std::chrono::microseconds>(now - bound.since)
                        .count());
    }
    if (bound.pointer != nullptr && state.pending_count < state.pending.size()) {
        state.pending[state.pending_count++] = {bound.device, bound.pointer, bound.value, kind,
                                                now};
    }
}

u64 BackSequence() {
    auto& state = Front();
    return state.sequence;
}

std::optional<BackRelease> BackReleaseSince(u64 device, u64 since) {
    auto& state = Front();
    const Release& release = state.releases[Hash(device) % Releases];
    if (release.device != device || release.sequence <= since) {
        return std::nullopt;
    }
    return BackRelease{release.writer, release.payload, release.reduction};
}

std::optional<BackRelease> BackReleaseLatest(u64 device) {
    auto& state = Front();
    const Release& release = state.releases[Hash(device) % Releases];
    if (release.device != device || release.sequence == 0) {
        return std::nullopt;
    }
    return BackRelease{release.writer, release.payload, release.reduction};
}

void NoteMetBy(MetBy met_by) {
    auto& state = Get();
    ++state.met_by[static_cast<size_t>(met_by)];
}

} // namespace VideoCommon::QmdPrediction
