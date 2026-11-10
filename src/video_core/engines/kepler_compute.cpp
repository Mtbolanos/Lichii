// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "common/algorithm.h"
#include "common/assert.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/fire_emblem_upscale.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_qmd_prediction.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/engines/native_upscale.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/memory_manager.h"
#include "video_core/native_query_trace.h"
#include "video_core/oot_network.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Engines {

namespace {
/// Settings oot_movie_upscale: whether Ocarina's movies keep the game's own upscaler (the network)
/// instead of the bilinear copy. Automatic: where tensor operations run natively. Logged once.
bool OotMovieNetworkChosen() {
    static const char* const env = std::getenv("EDEN_NX2_OOT_MOVIE_NETWORK");
    bool chosen = false;
    const char* why = "";
    if (env != nullptr && (env[0] == '0' || env[0] == '1')) {
        chosen = env[0] == '1';
        why = "EDEN_NX2_OOT_MOVIE_NETWORK";
    } else {
        switch (Settings::values.oot_movie_upscale.GetValue()) {
        case Settings::OotMovieUpscale::Game:
            chosen = true;
            why = "setting";
            break;
        case Settings::OotMovieUpscale::Bilinear:
            why = "setting";
            break;
        case Settings::OotMovieUpscale::Automatic:
        default:
            chosen = VideoCommon::Nx2Perf::g_native_tensor_ops.load(std::memory_order_relaxed);
            why = chosen ? "Automatic: native tensor operations" : "Automatic: emulated tensor operations";
            break;
        }
    }
    static std::atomic<int> logged{-1};
    if (logged.exchange(chosen ? 1 : 0) != (chosen ? 1 : 0)) {
        LOG_INFO(HW_GPU, "Ocarina movies: {} ({})",
                 chosen ? "the game's own upscaler" : "bilinear copy", why);
    }
    return chosen;
}

// EDEN_NX2_EXP13=2 (diagnostic, counting only; from zortos293/Lichii#717's
// EDEN_NX2_QMD_PREDICT): the split's front runs its own copy of the QMD scheduler. All channels'
// native compute methods pass through the front in the order the back runs them, so fed the same
// signals and the same QMD words, the copy runs the same QMDs at the same places and makes the
// same releases, which it records for nx2_qmd_prediction (the back checks each one). It reads a
// QMD only where guest memory holds what the back will read (nothing queued writes it, or the
// inline QMD the front sent last over it, kept as an overlay); elsewhere it waits for that work
// only, and drains and copies the back's table on GPU-written pages, all-zero QMDs and at the
// start of a split session or of EXP13=2. Nothing it predicts meets an acquire.
struct FrontQmdMirror {
    struct Overlay {
        const void* space;
        GPUVAddr address;
        u64 position; ///< The end of the newest write it holds (0x107, upload); 0 = empty
        /// Made by a 0x107 (which QueuedGuestWriters doesn't see), not from guest memory and a
        /// patch: replacing it raises `evicted`.
        bool inline_qmd;
        bool patched; ///< An upload wrote over it since
        NativeQmdScheduler::Descriptor words;
    };
    static constexpr size_t MaxOverlays = 256; ///< #717 has 64; OoT keeps more in flight

    /// A known linear compute inline upload's bytes (Ocarina uploads a 4 KB page of 16 QMDs with
    /// one and patches their release words with small ones): a QMD read inside its filled bytes
    /// takes them as an overlay's. Uploads are writer ranges, so a dropped one needs no floor.
    /// Each 256-byte block has its own position (fable_answer_6 6f (a)): the end of the newest
    /// write it holds, its upload's fill, a patch or a 0x107 into it; one position for the page
    /// let a patch to one block pass a 0x107 into another.
    struct Shadow {
        const void* space;
        GPUVAddr address; ///< 256-byte aligned
        u64 size;     ///< The upload's bytes
        u64 filled;   ///< Of them, those its data words gave so far
        u64 position; ///< The newest of its blocks' positions; 0 = empty
        u64 id;       ///< Which upload made it (FrontState::upload_shadow)
        u64 last_use;
        std::array<u64, 16> block_position;
        std::array<u8, 4096> bytes;
    };
    static constexpr u64 BlockBytes = 256;
    static constexpr size_t MaxShadows = 64;
    static constexpr u64 ShadowBytes = 4096;

    FrontQmdMirror() {
        scheduler.table = &table;
    }

    NativeQmdScheduler scheduler;
    NativeQmdScheduler::StateTable table;
    u64 session{~u64{0}};
    bool synced{};   ///< The table matches the back's for the work sent
    bool disabled{}; ///< Auto-off, until EXP13 leaves 2
    std::array<Overlay, MaxOverlays> overlays{};
    std::array<Shadow, MaxShadows> shadows{};
    u64 shadow_ids{};
    u64 use_clock{};
    /// The newest position of an inline QMD overlay replaced by another QMD's: reads of a QMD
    /// without an overlay wait for it while it is queued.
    u64 evicted{};
    /// Where the copy last read each QMD, for the unpredicted lines (6g/6h discriminators).
    struct ReadNote {
        u64 position;
        const MemoryManager* space;
        VideoCommon::QmdPrediction::Source source;
    };
    std::unordered_map<u64, ReadNote> reads;
    void NoteRead(u64 address, u64 position, const MemoryManager* space,
                  VideoCommon::QmdPrediction::Source source) {
        if (reads.size() >= 16384) {
            reads.clear();
        }
        reads[address] = {position, space, source};
    }
    /// Reads in the current window, those that waited for queued writes, and resyncs; the
    /// reads of the session and the windows in a row over the auto-off limit.
    u32 window_reads{};
    u32 window_waits{};
    u32 window_resyncs{};
    u64 session_reads{};
    u32 strikes{};

    const Overlay* FindOverlay(const void* space, GPUVAddr address) const {
        for (const Overlay& overlay : overlays) {
            if (overlay.position != 0 && overlay.address == address && overlay.space == space) {
                return &overlay;
            }
        }
        return nullptr;
    }

    void AddOverlay(const void* space, GPUVAddr address, u64 position, const u32* words,
                    bool inline_qmd) {
        Overlay* slot = nullptr;
        for (Overlay& overlay : overlays) {
            if (overlay.position != 0 && overlay.address == address && overlay.space == space) {
                slot = &overlay;
                break;
            }
        }
        if (slot == nullptr) {
            // A free slot, else the oldest.
            slot = &overlays[0];
            for (Overlay& overlay : overlays) {
                if (overlay.position < slot->position) {
                    slot = &overlay;
                }
            }
            if (slot->position != 0 && slot->inline_qmd) {
                evicted = (std::max)(evicted, slot->position);
            }
        }
        slot->space = space;
        slot->address = address;
        slot->position = position;
        slot->inline_qmd = inline_qmd;
        slot->patched = false;
        std::memcpy(slot->words.data(), words, sizeof(slot->words));
    }

    /// A shadow for the upload of `size` bytes at `address` (older ones it overlaps go).
    /// `evicted_used`: an LRU shadow in use was replaced.
    Shadow& NewShadow(const void* space, GPUVAddr address, u64 size, bool& evicted_used) {
        for (Shadow& shadow : shadows) {
            if (shadow.id != 0 && shadow.space == space && shadow.address < address + size &&
                address < shadow.address + shadow.size) {
                shadow.id = 0; // Its bytes are this upload's from here on
                shadow.position = 0;
            }
        }
        // A free slot, else the least recently used.
        Shadow* slot = &shadows[0];
        for (Shadow& shadow : shadows) {
            if (shadow.id == 0) {
                slot = &shadow;
                break;
            }
            if (shadow.last_use < slot->last_use) {
                slot = &shadow;
            }
        }
        evicted_used = slot->id != 0;
        slot->space = space;
        slot->address = address;
        slot->size = size;
        slot->filled = 0;
        slot->position = 0;
        slot->block_position.fill(0);
        slot->id = ++shadow_ids;
        slot->last_use = ++use_clock;
        return *slot;
    }

    Shadow* FindShadowById(u64 id) {
        if (id == 0) {
            return nullptr;
        }
        for (Shadow& shadow : shadows) {
            if (shadow.id == id) {
                return &shadow;
            }
        }
        return nullptr;
    }

    /// The shadow whose filled bytes hold the 256-byte block at `address` with the newest
    /// position for it; `block` its index there.
    Shadow* FindShadow(const void* space, GPUVAddr address, size_t& block) {
        Shadow* found = nullptr;
        for (Shadow& shadow : shadows) {
            if (shadow.id == 0 || shadow.space != space || address < shadow.address ||
                shadow.address + shadow.filled < address + BlockBytes ||
                (address - shadow.address) % BlockBytes != 0) {
                continue;
            }
            const size_t index = static_cast<size_t>((address - shadow.address) / BlockBytes);
            if (shadow.block_position[index] != 0 &&
                (found == nullptr ||
                 shadow.block_position[index] > found->block_position[block])) {
                found = &shadow;
                block = index;
            }
        }
        return found;
    }

    /// `block` of `shadow` holds the write at `position` now.
    static void StampBlock(Shadow& shadow, size_t block, u64 position) {
        shadow.block_position[block] = position;
        shadow.position = (std::max)(shadow.position, position);
    }

    void Clear() {
        // In place: the tables are large (a temporary would be on the stack).
        for (Overlay& overlay : overlays) {
            overlay.position = 0;
        }
        for (Shadow& shadow : shadows) {
            shadow.id = 0;
            shadow.position = 0;
        }
        evicted = 0;
        reads.clear();
    }

    void ResetWindow() {
        window_reads = window_waits = window_resyncs = 0;
        session_reads = 0;
        strikes = 0;
    }
};
FrontQmdMirror g_front_qmd;

// The copy's scheduler state for a QMD, for the unpredicted lines (6g (a) discriminator).
std::string ProbeFrontQmd(u64 qmd) {
    const auto& table = g_front_qmd.table;
    const auto it = table.find(qmd);
    if (it == table.end()) {
        return "absent";
    }
    const NativeQmdScheduler::State& state = it->second;
    const auto& d = state.descriptor;
    std::string text = fmt::format(
        "table: loaded {}, scheduled {}, completed {}, dependencies {}, deferred {}, base count "
        "{}; its release slots (words 24-27, 28-31, 52-55) {:08x} {:08x} {:08x} {:08x} | {:08x} "
        "{:08x} {:08x} {:08x} | {:08x} {:08x} {:08x} {:08x}",
        state.loaded, state.scheduled, state.completed, state.dependencies, state.deferred,
        (d[4] >> 14) & 0xffff, d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[52],
        d[53], d[54], d[55]);
    const auto read = g_front_qmd.reads.find(qmd);
    if (read == g_front_qmd.reads.end() || read->second.space == nullptr) {
        return text + "; no read noted";
    }
    constexpr std::array<const char*, 7> sources{"not a QMD", "inline QMD", "patched inline QMD",
                                                 "overlay from memory", "upload shadow",
                                                 "memory", "memory after a wait"};
    constexpr std::array<const char*, 12> causes{
        "none", "noted write", "writer range", "3D ranged write", "unbounded writer",
        "unranged writer", "queued function", "3D method write", "unknown noted write",
        "noted-write scan limit", "writer-range scan limit", "evicted inline QMD"};
    const auto source = static_cast<size_t>(read->second.source);
    text += fmt::format("; read at {} from {}", read->second.position,
                        source < sources.size() ? sources[source] : "?");
    const auto device = read->second.space->GpuToCpuAddress(qmd);
    if (!device) {
        return text;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    const u64 stamp = split.WriteStamp(*device, sizeof(NativeQmdScheduler::Descriptor), qmd);
    text += fmt::format("; the front's newest write over it at {} ({} the read)", stamp,
                        stamp > read->second.position ? "after" : "not after");
    const u64 done = split.ConsumedNow();
    const auto writers =
        split.QueuedGuestWriters(done, qmd, *device, sizeof(NativeQmdScheduler::Descriptor));
    const bool noted = writers.noted >= writers.unbounded;
    const auto cause = static_cast<size_t>(noted ? writers.noted_cause : writers.unbounded_cause);
    if ((std::max)(writers.noted, writers.unbounded) > done) {
        text += fmt::format("; still queued: {} at {}{}",
                            cause < causes.size() ? causes[cause] : "?",
                            (std::max)(writers.noted, writers.unbounded),
                            !noted && writers.what != nullptr
                                ? fmt::format(" ({})", writers.what)
                                : std::string{});
    }
    return text;
}

// The game patches the release words of QMDs already written (native_qmd.h: "The SDK patches
// release payloads while a scheduled QMD is waiting"): an overlay or shadow block that still
// holds its bytes (nothing but this write since it took them) takes the new ones and counts as
// written at `position`, so a read takes it instead of waiting. One that holds newer bytes than
// this write (stream order crossed) is dropped: the read then waits rather than serving a mix.
// Returns which of the (at most two) 256-byte blocks from first's were taken.
std::array<bool, 2> PatchHolders(VideoCommon::GpuSplit& split, const MemoryManager& memory,
                                 GPUVAddr first, GPUVAddr last, const u8* data, u64 position,
                                 u64 tag, const FrontQmdMirror::Shadow* own, u64 done) {
    namespace Prediction = VideoCommon::QmdPrediction;
    auto& mirror = g_front_qmd;
    constexpr u64 block_bytes = FrontQmdMirror::BlockBytes;
    // Write stamps (6f (c) a): no write but this one's since `held_position`.
    const auto still_holds = [&](GPUVAddr address, u64 bytes, u64 held_position) {
        const auto device = memory.GpuToCpuAddress(address);
        if (!device || held_position <= done) {
            return false;
        }
        return split.WriteStamp(*device, bytes, address, tag) <= held_position;
    };
    std::array<bool, 2> block_patched{};
    const GPUVAddr first_block = first & ~GPUVAddr{block_bytes - 1};
    const auto note_patched = [&](GPUVAddr address, u64 bytes) {
        for (u64 block = 0; block < block_patched.size(); ++block) {
            const GPUVAddr block_address = first_block + block * block_bytes;
            if (address <= block_address && block_address + block_bytes <= address + bytes) {
                block_patched[block] = true;
            }
        }
    };
    for (FrontQmdMirror::Shadow& shadow : mirror.shadows) {
        if (&shadow == own || shadow.id == 0 || shadow.position == 0 ||
            shadow.space != &memory || last <= shadow.address ||
            shadow.address + shadow.filled <= first) {
            continue;
        }
        // Block by block: each holds its bytes as of its own position.
        const u64 first_index = (std::max)(first, shadow.address) - shadow.address;
        const u64 last_index = (std::min)(last, shadow.address + shadow.filled) - shadow.address;
        for (u64 block = first_index / block_bytes; block * block_bytes < last_index; ++block) {
            const GPUVAddr block_address = shadow.address + block * block_bytes;
            if (block_address + block_bytes > shadow.address + shadow.filled) {
                continue;
            }
            if (shadow.block_position[block] > position) {
                shadow.block_position[block] = 0; // Newer than this write: unknown mix
                continue;
            }
            if (!still_holds(block_address, block_bytes, shadow.block_position[block])) {
                continue;
            }
            const GPUVAddr from = (std::max)(first, block_address);
            const GPUVAddr to = (std::min)(last, block_address + block_bytes);
            std::memcpy(shadow.bytes.data() + (from - shadow.address), data + (from - first),
                        to - from);
            FrontQmdMirror::StampBlock(shadow, static_cast<size_t>(block), position);
            note_patched(block_address, block_bytes);
            Prediction::NoteMirror(Prediction::Mirror::ShadowPatched);
        }
        shadow.last_use = ++mirror.use_clock;
    }
    constexpr u64 size = sizeof(NativeQmdScheduler::Descriptor);
    for (FrontQmdMirror::Overlay& overlay : mirror.overlays) {
        if (overlay.position == 0 || overlay.space != &memory || last <= overlay.address ||
            overlay.address + size <= first || overlay.position <= done) {
            continue;
        }
        if (overlay.position > position) {
            // Newer than this write: unknown mix. A queued inline QMD dropped keeps its floor.
            if (overlay.inline_qmd) {
                mirror.evicted = (std::max)(mirror.evicted, overlay.position);
            }
            overlay.position = 0;
            continue;
        }
        if (!still_holds(overlay.address, size, overlay.position)) {
            continue; // Something else wrote it since: the read waits
        }
        const GPUVAddr from = (std::max)(first, overlay.address);
        const GPUVAddr to = (std::min)(last, overlay.address + size);
        std::memcpy(reinterpret_cast<u8*>(overlay.words.data()) + (from - overlay.address),
                    data + (from - first), to - from);
        overlay.position = position;
        overlay.patched = true;
        note_patched(overlay.address, size);
        Prediction::NoteMirror(Prediction::Mirror::Patched);
    }
    return block_patched;
}

// The back runs everything sent (the signal being noted too), then the front copies its table.
void ResyncFrontQmd(VideoCommon::GpuSplit& split, VideoCommon::QmdPrediction::Mirror why) {
    auto& mirror = g_front_qmd;
    split.Drain(VideoCommon::DrainSite::QmdRead);
    // Reads and patches compare write stamps (EnableWriteStamps counts what came before).
    split.EnableWriteStamps();
    VideoCommon::QmdPrediction::SetQmdProbe(&ProbeFrontQmd);
    // The back is idle: nothing else touches its table now.
    mirror.table = NativeQmdScheduler::SharedStates();
    mirror.scheduler.table = &mirror.table;
    mirror.Clear();
    mirror.session = split.Session();
    mirror.synced = true;
    if (why == VideoCommon::QmdPrediction::Mirror::SessionStart) {
        mirror.ResetWindow();
        // Records from before (another split session's ring positions, or EXP13=1's) go.
        VideoCommon::QmdPrediction::SetRecordsValid(false, 0);
    }
    VideoCommon::QmdPrediction::SetRecordsValid(true, split.SentPosition());
    VideoCommon::QmdPrediction::NoteMirror(why);
}

// EXP13 left 2, or the copy turned itself off: it starts over when EXP13=2 signals again.
void StopFrontQmd(bool disable) {
    auto& mirror = g_front_qmd;
    mirror.synced = false;
    mirror.disabled = disable;
    mirror.Clear();
    mirror.ResetWindow();
    VideoCommon::QmdPrediction::SetRecordsValid(false, 0);
}

} // Anonymous namespace

void Nx2PatchQmdCopy(VideoCommon::GpuSplit& split, const MemoryManager& memory, GPUVAddr address,
                     const u8* data, u64 size, u64 position, u64 tag) {
    if (size == 0 || !VideoCommon::QmdPrediction::MirrorMode() || !g_front_qmd.synced) {
        return;
    }
    (void)PatchHolders(split, memory, address, address + size, data, position, tag, nullptr,
                       split.ConsumedNow());
}

KeplerCompute::KeplerCompute(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}
    , upload_state{memory_manager, regs.upload}
{
    // EDEN_NX2_QMD_TRACE: log the QMD scheduler's loads, dependency counts and launches.
    if (std::getenv("EDEN_NX2_QMD_TRACE")) {
        native_qmd.trace = [this](std::string_view event, u64 address, s32 value) {
            LOG_INFO(HW_GPU, "QMD trace {} {:#x} {} (engine {} as {})", event, address, value,
                     fmt::ptr(this), memory_manager.GetID());
        };
    }
    native_qmd.skipped_release = [](u64 address, std::size_t slot, std::span<const u32, 4> words) {
        static std::unordered_set<u32> logged;
        if (!logged.insert(words[1]).second) return;
        LOG_WARNING(HW_GPU, "Native QMD {:#x} skipped release slot {} with unsupported format, words={:08x}",
                    address, slot, fmt::join(words, ","));
    };
    execution_mask.reset();
    execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
    // Everything ProcessNativeMethod acts on besides storing the value; the QMD and 0x5a
    // trace range too while native draw diagnostics trace it.
    native_plain_methods.set();
    for (const u32 method : {0x45U, 0x46U, 0x47U, 0x48U, 0xaeU, 0xafU, 0xb0U, 0x107U, 0x6c3U,
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(exec_upload)),
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(data_upload))}) {
        native_plain_methods[method] = false;
    }
    if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr) {
        native_plain_methods[0x5a] = false;
        for (u32 method = 0xc6; method <= 0x108; ++method) {
            native_plain_methods[method] = false;
        }
    }
}

KeplerCompute::~KeplerCompute() = default;

void KeplerCompute::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer);
}

void KeplerCompute::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        if ((method & SplitPlainTag) != 0) {
            // GpuSplit::SinkMethod (EDEN_NX2_COMPUTE_SINK): as CallMethod's native plain path.
            method &= ~SplitPlainTag;
            native_registers[method] = value;
        }
        if (method < Regs::NUM_REGS) {
            regs.reg_array[method] = value;
        }
    }
    method_sink.clear();
}

void KeplerCompute::ApplyPairs(Core::System& system, const u64* pairs, u32 count) {
    // EDEN_NX2_RING_PAIRS: native plain writes from the split's front (FrontPlainWrite), stored
    // as CallMethod's native plain path stores them.
    for (u32 i = 0; i < count; ++i) {
        const u32 method = static_cast<u32>(pairs[i] & 0xffff);
        const u32 value = static_cast<u32>(pairs[i] >> 32);
        if (method < native_plain_methods.size()) [[likely]] {
            native_registers[method] = value;
            if (method < Regs::NUM_REGS) {
                regs.reg_array[method] = value;
            }
        } else {
            CallMethod(system, method, value, true);
        }
    }
}

// SET_REPORT_SEMAPHORE_A-D (0x6c0-0x6c3): address high/low, payload, then operation in bits 1:0
// (release / acquire / report only / trap), reduction enable bit 3 with the op in bits 11:9, and
// a one-word (no timestamp) structure in bit 28. NVN2 releases queue syncs from compute this way.
void KeplerCompute::ProcessNativeReportSemaphore(u32 argument) {
    const GPUVAddr address = (GPUVAddr{native_registers[0x6c0] & 0x1ffffff} << 32) |
                             native_registers[0x6c1];
    const u32 payload = native_registers[0x6c2];
    const u32 operation = argument & 3;
    const bool one_word = (argument >> 28) & 1;
    static u32 logged{};
    if (logged++ < 32) {
        LOG_INFO(HW_GPU, "Native compute report semaphore {:#x} = {:#x}, control {:#x}", address,
                 payload, argument);
    }
    if (operation != 0 && operation != 2) {
        LOG_WARNING(HW_GPU, "Native compute report semaphore operation {} not supported", operation);
        return;
    }
    if ((argument >> 3) & 1) {
        const u32 reduction = (argument >> 9) & 7;
        rasterizer->SignalFence([this, address, payload, reduction] {
            const u32 value = ReduceSemaphore(reduction, false, memory_manager.Read<u32>(address),
                                              payload);
            // EXP13=2: marked before the word is written (nx2.h); counting only.
            if (Tegra::Nx2LandedHooked()) [[unlikely]] {
                if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                    Tegra::Nx2NoteLanded(*device, value, Tegra::Nx2Landing::ReportReduction);
                }
            }
            memory_manager.Write<u32>(address, value);
            if (Common::Nx2GpuWrite::g_listener.load(std::memory_order_relaxed) != nullptr) {
                Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::ComputeReduce,
                                            memory_manager.GpuToCpuAddress(address).value_or(0),
                                            address, memory_manager.GetPointer(address), value);
            }
        });
        return;
    }
    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                      one_word ? VideoCommon::QueryPropertiesFlags::IsAFence
                               : VideoCommon::QueryPropertiesFlags::HasTimeout,
                      payload, 0);
}

bool KeplerCompute::ProcessNativeMethod(Core::System& system, u32 method, u32 argument, bool is_last_call) {
    if (native_context && !front_macros) {
        if (method == 0x45 || method == 0x47) native_compute_uploaded = true;
        native_macro = native_compute_uploaded ? &native_context->compute_engine : &native_context->engine;
        native_macro_positions = native_compute_uploaded ? std::span<u32>{native_context->compute_positions} : std::span<u32>{native_context->positions};
    }
    if (method >= native_registers.size()) {
        LOG_ERROR(HW_GPU, "Native compute method out of range: {:#x}", method);
        return true;
    }
    if (method >= 0xe00) {
        if (!native_executing_macro) {
            if (method & 1) { LOG_ERROR(HW_GPU, "Native compute macro starts on argument method"); return true; }
            native_executing_macro = method;
        }
        native_parameters.push_back(argument);
        if (!is_last_call) return true;
        const u32 entry = ((*native_executing_macro - 0xe00) / 2) % native_macro_positions.size();
        native_executing_macro_logged = native_executing_macro;
        native_executing_macro.reset();
        static const u32 watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
        }();
        const bool watched = watch != 0 && std::ranges::any_of(native_parameters,
            [](u32 parameter) { return (parameter & ~0xfU) == watch; });
        if (watched) {
            LOG_INFO(HW_GPU, "Semaphore watch: compute macro start={:#x} parameters={:08x}",
                     native_macro_positions[entry], fmt::join(native_parameters, ","));
        }
        // EDEN_NX2_MME_DUMP: write the compute macro code once per engine, and log each macro
        // call with its entry and parameters (diagnostics).
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr; mme_dump) {
            if (!native_mme_dumped && native_macro_positions[entry] == 0xda) {
                native_mme_dumped = true;
                std::ofstream dump(fmt::format("native-compute-mme-{}.bin", fmt::ptr(this)),
                                   std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()),
                           sizeof(native_macro->native_code));
            }
            LOG_INFO(HW_GPU, "MME call engine {} method {:#x} start={:#x} parameters={:08x}",
                     fmt::ptr(this), *native_executing_macro_logged, native_macro_positions[entry],
                     fmt::join(native_parameters, ","));
        }
        if (native_data_ram) native_macro->native_interpreter.active_dram = native_data_ram;
        const u32 dram0_before = (*native_macro->native_interpreter.active_dram)[0x27];
        const u32 start = native_macro_positions[entry];
        const auto emit = [&](u32 address, u32 value) {
            static u32 diagnostic_writes{};
            static const bool log_emits = std::getenv("EDEN_NX2_SEM_WATCH_EMITS") != nullptr;
            if (watched && log_emits) LOG_INFO(HW_GPU, "Semaphore watch: compute macro emit {:#x} = {:#x}", address, value);
            if (diagnostic_writes++ < 4096) LOG_DEBUG(HW_GPU, "Native compute MME emit method={:#x}, value={:#x}", address, value);
            if (address >= 0xe00) return false;
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            CallMethod(system, address, value, true);
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        // Compiled macro code, as for 3D macros (macro_turing_jit.h); EDEN_NX2_MME_JIT=0 off.
        static const bool use_jit = [] {
            const char* value = std::getenv("EDEN_NX2_MME_JIT");
            return value == nullptr || value[0] != '0';
        }();
        if (use_jit && !watched) {
            if (!native_macro->native_jit) {
                native_macro->native_jit = std::make_shared<TuringMacroJit>();
            }
            native_macro->native_jit->Prepare(native_macro->native_code,
                                              native_macro->native_code_generation);
            const auto result = native_macro->native_jit->Execute(
                start, native_parameters,
                std::span<u32>{*native_macro->native_interpreter.active_dram},
                [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
                emit);
            if (!result.success) {
                native_macro->native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native MME (JIT) failed: compute start={:#x}, consumed={}/{}, error={}",
                          start, result.consumed, native_parameters.size(), result.error);
            }
            native_parameters.clear();
            return true;
        }
#endif
        // Repeated calls replay their writes, as for 3D macros (macro_memo.h).
        static const bool use_memo = [] {
            const char* value = std::getenv("EDEN_NX2_MME_MEMO");
            return value == nullptr || value[0] != '0';
        }();
        auto& memo = native_macro->native_memo;
        if (use_memo && !watched) {
            memo.Synchronize(native_macro->native_code_generation);
            if (std::remove_reference_t<decltype(memo)>::Effects effects;
                memo.Find(start, native_parameters, *native_macro->native_interpreter.active_dram,
                          effects)) {
                auto& data_ram = *native_macro->native_interpreter.active_dram;
                for (size_t i = 0; i < effects.data_ram_writes.size(); i += 2) {
                    data_ram[effects.data_ram_writes[i]] = effects.data_ram_writes[i + 1];
                }
                for (size_t i = 0; i < effects.writes.size(); i += 2) {
                    emit(effects.writes[i], effects.writes[i + 1]);
                }
                native_parameters.clear();
                return true;
            }
            native_macro->native_writes.clear();
        }
        // The interpreter consumes native_parameters as its FIFO: keep the call's own parameters.
        native_memo_parameters.assign(native_parameters.begin(), native_parameters.end());
        const u64 code_generation = native_macro->native_code_generation;
        const auto result = native_macro->native_interpreter.Execute(native_macro->native_code,
            start, native_parameters,
            [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
            [&](u32 address, u32 value) {
                if (use_memo) {
                    native_macro->native_writes.emplace_back(address, value);
                }
                return emit(address, value);
            });
        const auto& interpreter = native_macro->native_interpreter;
        if (use_memo && !watched && result.success && !interpreter.read_registers &&
            !interpreter.reread_data_ram && code_generation == native_macro->native_code_generation) {
            memo.Store(start, native_memo_parameters, interpreter.data_ram_reads,
                       interpreter.data_ram_writes, native_macro->native_writes);
        }
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr;
            mme_dump && (*native_macro->native_interpreter.active_dram)[0x27] != dram0_before) {
            LOG_INFO(HW_GPU, "MME data RAM[0x27] {:#x} -> {:#x} by engine {} macro start={:#x}",
                     dram0_before, (*native_macro->native_interpreter.active_dram)[0x27], fmt::ptr(this),
                     native_macro_positions[entry]);
        }
        if (!result.success) {
            native_macro->native_execution_failed = true;
            if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
                diagnostics) {
                std::ofstream dump("native-compute-mme.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            }
            LOG_ERROR(HW_GPU, "Native MME failed: compute start={:#x}, pc={:#x}, consumed={}/{}, error={}, parameters={:08x}",
                native_macro_positions[entry], result.pc, result.consumed, native_parameters.size(), result.error,
                fmt::join(native_parameters, ","));
        }
        native_parameters.clear();
        return true;
    }
    native_registers[method] = argument;
    if (method == 0x6c3) { // SET_REPORT_SEMAPHORE_D: the compute class's semaphore release
        ProcessNativeReportSemaphore(argument);
        return true;
    }
    if (method == 0x5a || (method >= 0xc6 && method <= 0x108)) {
        VideoCommon::TraceNativeQuery(12, memory_manager.GetID(), method, 0, 0, argument, 0);
    }
    if (method == 0x107) {
        const GPUVAddr address = ((GPUVAddr{native_registers[0xc6]} << 32) |
            native_registers[0xc7]) << 8;
        const GPUVAddr program = (GPUVAddr{native_registers[0xc8 + 49] & 0x1ffff} << 32) |
            native_registers[0xc8 + 48];
        const u32 control = native_registers[0xa7];
        LOG_DEBUG(HW_GPU, "Native inline QMD: address={:#x}, program={:#x}, control={:#x}, grid={}/{}/{}",
            address, program, control, native_registers[0xc8 + 12],
            native_registers[0xc8 + 13] & 0xffff, native_registers[0xc8 + 14] & 0xffff);
        if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
            diagnostics) {
            std::ofstream dump("native-inline-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(native_registers.data() + 0xc8), 256);
            std::ofstream registers("native-inline-compute-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
            if (memory_manager.IsFullyMappedRange(program, 0x10000)) {
                std::vector<u8> code(0x10000);
                memory_manager.ReadBlockUnsafe(program, code.data(), code.size());
                std::ofstream shader("native-inline-compute-shader.bin", std::ios::binary);
                shader.write(reinterpret_cast<const char*>(code.data()), code.size());
            }
        }
        if (!memory_manager.IsFullyMappedRange(address, 256)) {
            native_macro->native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Unmapped native inline QMD destination");
            return true;
        }
        // The QMD keeps grid placeholders for DispatchNativeCompute (nx2_indirect_grid.h).
        Tegra::Nx2IndirectGrid::g_writing_qmd = true;
        rasterizer->AccelerateInlineToMemory(address, 256,
            std::span<const u8>{reinterpret_cast<const u8*>(native_registers.data() + 0xc8), 256});
        Tegra::Nx2IndirectGrid::g_writing_qmd = false;
        if ((control & (1U << 24)) || !(native_registers[0xcc] & (1U << 12))) {
            const u32 previous = native_registers[0xad];
            native_registers[0xad] = static_cast<u32>(address >> 8);
            ProcessNativeMethod(system, 0xb0, 3, true);
            native_registers[0xad] = previous;
        }
        return true;
    }
    if (front_macros && method >= 0x45 && method <= 0x48) {
        return true; // Loaded and bound by the split GPU thread's front (FrontApply).
    }
    switch (method) {
    case 0x45: native_macro->ClearCode(argument); return true;
    case 0x46: native_macro->AddCode(0, argument); return true;
    case 0x47: native_bind_pointer = argument; return true;
    case 0x48:
        if (native_bind_pointer < native_macro_positions.size()) native_macro_positions[native_bind_pointer++] = argument;
        else LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        return true;
    case 0xae: case 0xaf: case 0xb0: {
        const GPUVAddr qmd_address = GPUVAddr{native_registers[0xad]} << 8;
        VideoCommon::TraceNativeQuery(9, memory_manager.GetID(), qmd_address,
            memory_manager.GpuToCpuAddress(qmd_address).value_or(0), method, argument, 0);
        if (method == 0xae || (method == 0xaf && !(argument & 2)) ||
            (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1))) return true;
        static const bool qmd_diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
        if ((argument & 15) == 12 && qmd_diagnostics) {
            static u32 captured{};
            if (captured++ == 0 && memory_manager.IsFullyMappedRange(qmd_address, 256)) {
                std::array<u32, 64> descriptor{};
                memory_manager.ReadBlockUnsafe(qmd_address, descriptor.data(), sizeof(descriptor));
                std::ofstream dump("native-increment-qmd.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(descriptor.data()), sizeof(descriptor));
                LOG_WARNING(HW_GPU, "Native increment dependency descriptor: address={:#x}, words={:08x}",
                    qmd_address, fmt::join(descriptor, ","));
            }
        }
        // EXP10: the split's front reads what the dispatches and releases this runs write.
        auto& split = VideoCommon::GpuSplit::Get();
        split.BeginDispatchRecord();
        SCOPE_EXIT {
            split.EndDispatchRecord();
        };
        u64 running_qmd = 0; // The QMD the releases below belong to (read last), for EXP13
        const NativeQmdScheduler::Descriptor* running_descriptor = nullptr;
        const auto error = native_qmd.Signal(qmd_address, argument,
            [&](u64 address, NativeQmdScheduler::Descriptor& descriptor) {
                running_qmd = address;
                running_descriptor = nullptr;
                if (!memory_manager.IsFullyMappedRange(address, sizeof(descriptor))) return false;
                memory_manager.ReadBlockUnsafe(address, descriptor.data(), sizeof(descriptor));
                running_descriptor = &descriptor;
                if (static u32 logged{}; (descriptor[53] & (1U << 23)) && logged < 8) {
                    ++logged;
                    LOG_INFO(HW_GPU, "Native QMD {:#x} releases: {:08x} | {:08x} | {:08x}", address,
                             fmt::join(std::span{descriptor}.subspan(24, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(28, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(52, 4), ","));
                }
                for (u32 slot = 0; slot < 2; ++slot) {
                    const u32 flags = descriptor[25 + slot * 4];
                    if (!(flags & (1U << 23))) continue;
                    const GPUVAddr destination = (GPUVAddr{flags & 0xff} << 32) | descriptor[24 + slot * 4];
                    VideoCommon::TraceNativeQuery(7 + slot, memory_manager.GetID(), destination,
                        memory_manager.GpuToCpuAddress(destination).value_or(0),
                        descriptor[4], descriptor[26 + slot * 4], flags);
                }
                return true;
            },
            [&](u64 address, u32 payload, u32 flags) {
                const bool four_words = (flags >> 30) == 0;
                if (!memory_manager.IsFullyMappedRange(address, four_words ? 16 : sizeof(payload))) return false;
                if (VideoCommon::QmdPrediction::Enabled() &&
                    VideoCommon::GpuSplit::OnBackThread()) {
                    if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                        std::array<u32, 6> slots{};
                        if (running_descriptor != nullptr) {
                            const auto& d = *running_descriptor;
                            slots = {d[25], d[26], d[29], d[30], d[53], d[54]};
                        }
                        VideoCommon::QmdPrediction::Verify(
                            VideoCommon::QmdPrediction::Writer::Qmd, *device, payload, running_qmd,
                            split.BackEntryPosition(), (flags & (1U << 26)) != 0,
                            running_descriptor != nullptr ? slots.data() : nullptr);
                    }
                }
                if (VideoCommon::GpuSplit::RecordingDispatch()) {
                    // A payload release writes up to a report's 16 bytes, at a later fence.
                    if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                        split.RecordDispatchWrite(*device, 16);
                    }
                }
                if (flags & (1U << 26)) {
                    rasterizer->SignalFence([this, address, payload, flags] {
                        const u32 value = ReduceSemaphore((flags >> 20) & 7, (flags >> 24) & 1,
                                                          memory_manager.Read<u32>(address), payload);
                        // EXP13=2: marked before the word is written (nx2.h); counting only.
                        if (Tegra::Nx2LandedHooked()) [[unlikely]] {
                            if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                                Tegra::Nx2NoteLanded(*device, value,
                                                     Tegra::Nx2Landing::QmdReduction);
                            }
                        }
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(*device, value);
                            Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::QmdReduce,
                                                        *device, address,
                                                        memory_manager.GetPointer(address), value);
                        }
                        static const u64 watch = [] {
                            const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                            return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                        }();
                        if (watch != 0 && (address & ~u64{0xf}) == watch) {
                            LOG_INFO(HW_GPU, "Semaphore watch: QMD release reduction {:#x} -> {:#x}", address, value);
                        }
                        VideoCommon::TraceNativeQuery(1, memory_manager.GetID(), address,
                            memory_manager.GpuToCpuAddress(address).value_or(0), 0, value, 1U << 26);
                    });
                } else if (four_words) {
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence | VideoCommon::QueryPropertiesFlags::HasTimeout,
                        payload, 0);
                } else if (flags & (1U << 19)) {
                    // NVN2 tracks QMD completion with a max release (bit 19) into a word other
                    // releases also advance: a plain write would move it backwards and leave
                    // GEQ acquires waiting forever.
                    rasterizer->SignalFence([this, address, payload] {
                        const u32 value = (std::max)(memory_manager.Read<u32>(address), payload);
                        // EXP13=2: marked before the word is written (nx2.h); counting only.
                        if (Tegra::Nx2LandedHooked()) [[unlikely]] {
                            if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                                Tegra::Nx2NoteLanded(*device, value, Tegra::Nx2Landing::QmdMax);
                            }
                        }
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(*device, value);
                            Common::Nx2GpuWrite::Notify(Common::Nx2GpuWrite::Path::QmdMax, *device,
                                                        address,
                                                        memory_manager.GetPointer(address), value);
                        }
                    });
                } else {
                    static const u64 watch = [] {
                        const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                        return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                    }();
                    if (watch != 0 && (address & ~u64{0xf}) == (watch & ~u64{0xf})) {
                        LOG_INFO(HW_GPU, "Semaphore watch: QMD release {:#x} = {:#x} flags {:#x}", address, payload, flags);
                    }
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
                }
                return true;
            }, [&](const auto& descriptor) { return DispatchNativeCompute(descriptor); });
        if (error.empty()) {
            LOG_DEBUG(HW_GPU, "Native QMD control action={:#x}, QMD={:#x}", argument, qmd_address);
            return true;
        }
        native_macro->native_execution_failed = true;
        std::array<u32, 128> qmd{};
        if (memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            memory_manager.ReadBlockUnsafe(qmd_address, qmd.data(), sizeof(qmd));
        }
        // Log a failing program's first few dispatches only (QMD words 48-49 hold its code
        // address); it fails the same way every frame.
        static std::unordered_map<u64, u32> failures;
        const u64 program = (u64{qmd[49]} << 32) | qmd[48];
        if (failures[program ^ std::hash<std::string>{}(std::string{error})]++ >= 2) {
            return true;
        }
        LOG_ERROR(HW_GPU, "Native QMD failed: {}", error);
        if (qmd_diagnostics && memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            std::ofstream dump("native-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(qmd.data()), sizeof(qmd));
            std::ofstream macros("native-dispatch-mme.bin", std::ios::binary);
            macros.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            std::ofstream registers("native-dispatch-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
        }
        LOG_ERROR(HW_GPU, "Native compute dispatch unsupported: method={:#x}, value={:#x}, QMD={:#x}, words={:08x}", method, argument,
                  qmd_address, fmt::join(qmd, ","));
        return true;
    }
    default:
        // The inline upload register block is shared with the older class.
        if (method < Regs::NUM_REGS) regs.reg_array[method] = argument;
        return method != KEPLER_COMPUTE_REG_INDEX(exec_upload) && method != KEPLER_COMPUTE_REG_INDEX(data_upload);
    }
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (native_mode) {
        // Most of the compute class's register writes (QMD fields replayed by its macros for
        // each of thousands of dispatches a second) only store the value: ~4% of title 1's GPU
        // thread went through ProcessNativeMethod's checks for them.
        if (method < native_plain_methods.size() && native_plain_methods[method]) {
            native_registers[method] = method_argument;
            if (method < Regs::NUM_REGS) {
                regs.reg_array[method] = method_argument;
            }
            return;
        }
        if (ProcessNativeMethod(system, method, method_argument, is_last_call)) return;
    }
    if (method >= Regs::NUM_REGS) {
        LOG_ERROR(HW_GPU, "Compute register out of range: {:#x}", method);
        return;
    }

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload): {
        // EDEN_NX2_EXP29 & 4 (zortos293/Lichii#717 EDEN_NX2_DROP_NATIVE_UPLOADS): only the
        // launch case below reads the list, and native mode never gets there
        // (ProcessNativeMethod takes 0xaf): the list grew by one entry per exec all session.
        if (!native_mode || (VideoCore::Nx2Debug::Experiment(29) & 4) == 0) {
            UploadInfo info{.upload_address = upload_address,
                            .exec_address = upload_state.ExecTargetAddress(),
                            .copy_size = upload_state.GetUploadSize(),
                            .was_dirty = upload_dirty};
            uploads.push_back(info);
        }
        upload_state.ProcessExec(regs.exec_upload.linear != 0);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();

        for (auto& data : uploads) {
            const GPUVAddr offset = data.exec_address - launch_desc_loc;
            if (offset / sizeof(u32) == LAUNCH_REG_INDEX(grid_dim_x)) {
                const bool source_dirty = memory_manager.IsMemoryDirty(data.upload_address, data.copy_size);
                if (data.was_dirty || source_dirty) {
                    indirect_compute = {data.upload_address};
                }
            }
        }
        uploads.clear();
        ProcessLaunch();
        indirect_compute = std::nullopt;
        break;
    }
    default:
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (native_mode) {
        for (u32 i = 0; i < amount; ++i) CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        return;
    }
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

bool KeplerCompute::DispatchNativeCompute(const NativeQmdScheduler::Descriptor& qmd) {
    launch_description = {};
    launch_description.grid_dim_x.Assign(qmd[12]);
    launch_description.grid_dim_y.Assign(qmd[13] & 0xffff);
    launch_description.grid_dim_z.Assign(qmd[14] & 0xffff);
    launch_description.shared_alloc.Assign(qmd[17] & 0x3ffff);
    launch_description.block_dim_x.Assign(qmd[18] >> 16);
    launch_description.block_dim_y.Assign(qmd[19] & 0xffff);
    launch_description.block_dim_z.Assign(qmd[19] >> 16);
    launch_description.const_buffer_enable_mask.Assign(qmd[20] & 0xff);
    launch_description.local_pos_alloc.Assign(qmd[23] & 0xffffff);
    launch_description.local_crs_alloc.Assign(qmd[50] & 0xffffff);
    regs.code_loc.address_low = qmd[48];
    regs.code_loc.address_high = qmd[49] & 0x1ffff;
    for (u32 slot = 0; slot < NumConstBuffers; ++slot) {
        const u32 high = qmd[33 + slot * 2];
        if (high & 0x1ff00) {
            LOG_ERROR(HW_GPU, "Native compute constant address exceeds supported GPU width");
            return false;
        }
        auto& buffer = launch_description.const_buffer_config[slot];
        buffer.address_low = qmd[32 + slot * 2];
        buffer.address_high.Assign(high & 0xff);
        buffer.size.Assign(((high >> 19) & 0x1fff) << 4);
    }
    if (!launch_description.block_dim_x || !launch_description.block_dim_y || !launch_description.block_dim_z)
        return false;
    // A grid of GPU-written placeholders: dispatch indirectly from the words they stand for.
    bool grid_placeholder = false;
    if (Tegra::Nx2IndirectGrid::AnyLive()) [[unlikely]] {
        const u32 x = qmd[12];
        const u32 y = qmd[13] & 0xffff;
        const u32 z = qmd[14] & 0xffff;
        Tegra::Nx2IndirectGrid::LogCounts();
        if (const auto match = Tegra::Nx2IndirectGrid::MatchX(x)) {
            const auto [slot, source] = *match;
            // y and z: the entry's words too, or the constant 1 the macro used instead (the
            // entry's then hold 1 as well: its sizes are counts of 1D work).
            const bool y_ok = y == Tegra::Nx2IndirectGrid::Placeholder(slot, 1) || y == 1;
            const bool z_ok = z == Tegra::Nx2IndirectGrid::Placeholder(slot, 2) || z == 1;
            static const bool read_back = [] {
                const char* value = std::getenv("EDEN_NX2_INDIRECT_GRID");
                return value != nullptr && value[0] == '2';
            }();
            if (!y_ok || !z_ok) {
                Tegra::Nx2IndirectGrid::Get().skipped.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1) < 8) {
                    LOG_ERROR(HW_GPU, "GPU-sized dispatch {:#x}: y {:#x} z {:#x} unsupported, skipped",
                              x, y, z);
                }
                return true;
            }
            if (read_back) {
                // Debug: wait for the words here and dispatch them directly.
                memory_manager.FlushRegion(source, 3 * sizeof(u32));
                std::array<u32, 3> words{};
                memory_manager.ReadBlock(source, words.data(), sizeof(words));
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1) < 40) {
                    LOG_INFO(HW_GPU, "GPU-sized dispatch {:#x} from {:#x}: {} {} {} (qmd y {:#x} z {:#x})",
                             x, source, words[0], words[1], words[2], y, z);
                }
                launch_description.grid_dim_x.Assign(words[0]);
                launch_description.grid_dim_y.Assign(words[1] & 0xffff);
                launch_description.grid_dim_z.Assign(words[2] & 0xffff);
            } else {
                indirect_compute = source;
            }
            grid_placeholder = true;
            Tegra::Nx2IndirectGrid::Get().dispatched.fetch_add(1, std::memory_order_relaxed);
        } else if (Tegra::Nx2IndirectGrid::LooksLikePlaceholder(x)) {
            Tegra::Nx2IndirectGrid::Get().skipped.fetch_add(1, std::memory_order_relaxed);
            // Never dispatch the placeholder itself: ~2 billion workgroups hang the GPU.
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1) < 8) {
                LOG_ERROR(HW_GPU, "GPU-sized dispatch {:#x} lost its source: skipped", x);
            }
            return true;
        }
    }
    SCOPE_EXIT {
        if (grid_placeholder) {
            indirect_compute = std::nullopt;
        }
    };
    dlss_chain.output_pass = false;
    // Fortune's Weave's evaluation is 13 dispatches. Bound incomplete evaluations, including
    // skipped dispatches, so a missing output pass cannot keep the chain active indefinitely.
    constexpr u32 max_dispatches = 64;
    if (dlss_chain.active && ++dlss_chain.dispatches > max_dispatches) {
        dlss_chain.active = false;
        dlss_chain.pending.clear();
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
    }
    // Ocarina of Time's movie network (port of Eglantine 92f119195f): a pass that reads one
    // colour image and stores none retains its colour and opens a skip window; the pass that
    // stores an image upscaling it is replaced by a plain copy and closes it. The gameplay
    // network is left alone: its pair (+0xc0 -> +0xd0) carries the tonemap.
    if (!native_upscale_disabled && !dlss_chain.active &&
        Settings::GetCurrentProgramID() == VideoCommon::OotNetwork::ProgramId &&
        !OotMovieNetworkChosen() &&
        (launch_description.const_buffer_enable_mask & 1)) {
        const auto& cbuf = launch_description.const_buffer_config[0];
        const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
        const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
            if (offset + sizeof(u32) > cbuf.size.Value()) return std::nullopt;
            const u32 handle = memory_manager.Read<u32>(cbuf_address + offset);
            if (handle == 0 || (handle & 0xfffff) > regs.tic.limit) return std::nullopt;
            return GetTICEntry(handle & 0xfffff);
        };
        const auto upscale_pair = [](const std::optional<Texture::TICEntry>& from,
                                     const std::optional<Texture::TICEntry>& to,
                                     bool same_format) {
            if (!from || !to || (same_format && from->format != to->format) ||
                from->Address() == 0 || to->Address() == 0 ||
                from->Address() == to->Address() || !from->IsBlockLinear() ||
                !to->IsBlockLinear())
                return false;
            const u64 from_width = from->Width(), from_height = from->Height();
            const u64 to_width = to->Width(), to_height = to->Height();
            const u64 cross_a = to_width * from_height, cross_b = to_height * from_width;
            return from_width >= 64 && to_width > from_width && to_width <= from_width * 4 &&
                   (cross_a > cross_b ? cross_a - cross_b : cross_b - cross_a) * 32 <= cross_a;
        };
        const auto [reads, stores] = NativeImageHandles(regs.code_loc.Address());
        std::optional<Texture::TICEntry> replacement_input;
        std::optional<Texture::TICEntry> replacement_output;
        if (native_upscale_input && network_skip_remaining > 0 && !stores.empty()) {
            if (const auto store = FindNativeUpscaleOutput(native_upscale_input, stores, image,
                                                           upscale_pair)) {
                replacement_input = native_upscale_input;
                replacement_output = image(*store);
            }
        }
        bool skipping{};
        if (reads.size() == 1 && stores.empty() && reads.front() == 0x58) {
            // Ocarina's movie network reads its colour at +0x58; anchoring the window there
            // keeps gameplay compute untouched.
            RememberNativeUpscaleInput();
            if (native_upscale_input) {
                network_skip_remaining = 32;
                skipping = true;
            }
        } else if (!replacement_input && network_skip_remaining > 0) {
            if (--network_skip_remaining == 0) {
                native_upscale_input.reset();
            }
            skipping = true;
        }
        if (replacement_input && replacement_output) {
            const auto surface = [](const Texture::TICEntry& tic)
                -> std::optional<Fermi2D::Surface> {
                using RTFormat = Tegra::RenderTargetFormat;
                Fermi2D::Surface result{};
                switch (tic.format) {
                case Texture::TextureFormat::B10G11R11:
                    result.format = RTFormat::B10G11R11_FLOAT;
                    break;
                case Texture::TextureFormat::R16G16B16A16:
                    result.format = RTFormat::R16G16B16A16_FLOAT;
                    break;
                case Texture::TextureFormat::A8B8G8R8:
                    result.format = RTFormat::A8B8G8R8_UNORM;
                    break;
                case Texture::TextureFormat::A2B10G10R10:
                    result.format = RTFormat::A2B10G10R10_UNORM;
                    break;
                default:
                    return std::nullopt;
                }
                if (tic.Address() == 0 || !tic.IsBlockLinear()) {
                    return std::nullopt;
                }
                result.linear = Fermi2D::MemoryLayout::BlockLinear;
                result.block_width.Assign(tic.block_width);
                result.block_height.Assign(tic.block_height);
                result.block_depth.Assign(tic.block_depth);
                result.depth = 1;
                result.width = tic.Width();
                result.height = tic.Height();
                result.addr_upper = static_cast<u32>(tic.Address() >> 32);
                result.addr_lower = static_cast<u32>(tic.Address());
                return result;
            };
            const auto source = surface(*replacement_input);
            const auto dest = surface(*replacement_output);
            if (source && dest) {
                // Does the program decode sRGB itself? (Ocarina's movie network does.)
                const bool decodes_srgb = [&] {
                    const GPUVAddr program = regs.code_loc.Address();
                    std::array<u32, 0x400> block{};
                    bool threshold = false, exponent = false;
                    for (u64 offset = 0; offset < 0x20000 && !(threshold && exponent);
                         offset += sizeof(block)) {
                        if (!memory_manager.IsFullyMappedRange(program + offset, sizeof(block)))
                            break;
                        memory_manager.ReadBlock(program + offset, block.data(), sizeof(block));
                        for (size_t word = 0; word < block.size(); word += 4) {
                            const u32 opcode = block[word] & 0xfff;
                            threshold |= opcode == 0x434 && block[word + 1] == 0x292d292d;
                            exponent |= opcode == 0x832 && block[word + 1] == 0x40cc40cc;
                        }
                    }
                    return threshold && exponent;
                }();
                const Fermi2D::Config config{
                    .operation = decodes_srgb ? Fermi2D::Operation::SrcCopySrgbToLinear
                                              : Fermi2D::Operation::SrcCopy,
                    .filter = Fermi2D::Filter::Bilinear,
                    .must_accelerate = true,
                    .dst_x0 = 0,
                    .dst_y0 = 0,
                    .dst_x1 = static_cast<s32>(replacement_output->Width()),
                    .dst_y1 = static_cast<s32>(replacement_output->Height()),
                    .src_x0 = 0,
                    .src_y0 = 0,
                    .src_x1 = static_cast<s32>(replacement_input->Width()),
                    .src_y1 = static_cast<s32>(replacement_input->Height()),
                };
                const bool done = rasterizer->AccelerateSurfaceCopy(*source, *dest, config);
                LOG_DEBUG(HW_GPU,
                          "Native network pass {:#x} replaced by input copy {}->{} ({}): {}",
                          regs.code_loc.Address(), replacement_input->Address(),
                          replacement_output->Address(), decodes_srgb ? "sRGB" : "plain", done);
                native_upscale_input.reset();
                network_skip_remaining = 0;
                if (done) {
                    return true;
                }
                // Run the real pass, and the whole network on later frames.
                native_upscale_disabled = true;
                skipping = false;
            }
        }
        if (skipping) {
            return true;
        }
    }
    bool dispatched = true;
    // EXP10 (GpuSplit::BeginDispatchRecord): what the dispatch writes. A rejected program writes
    // nothing besides what its bindings marked before (recorded as they were marked).
    auto& split = VideoCommon::GpuSplit::Get();
    const bool recording = VideoCommon::GpuSplit::RecordingDispatch();
    bool skipped = false;
    bool replaced = false;
    if (recording) {
        split.StartDispatch();
    }
    SCOPE_EXIT {
        if (recording) {
            split.FinishDispatch(!skipped, replaced);
        }
    };
    try {
        rasterizer->DispatchCompute();
    } catch (const VideoCommon::FireEmblemUpscale::DispatchReplaced&) {
        replaced = true;
        return true;
    } catch (const Shader::Exception& error) {
        skipped = true;
        if (Common::DkbFrameTrace::Enabled()) {
            static thread_local Common::DkbFrameTrace::RateLimit limit;
            static thread_local u64 rejected{};
            ++rejected;
            if (limit.Ready(std::chrono::steady_clock::now())) {
                LOG_INFO(HW_GPU,
                         "DKB shader rejected stage=compute address={:#x} count={} reason={}",
                         regs.code_loc.Address() + launch_description.program_start, rejected,
                         error.what());
                rejected = 0;
            }
        }
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        static std::unordered_set<std::string> logged;
        if (logged.insert(error.what()).second) {
            LOG_ERROR(HW_GPU, "Native compute shader rejected: {}", error.what());
        }
        dispatched = false;
        if (dlss_chain.active) {
            ++dlss_chain.rejected;
        } else if (std::string_view{error.what()}.find("matrix multiply") !=
                   std::string_view::npos) {
            DescribeNativeResources(qmd);
            LOG_ERROR(HW_GPU, "Native tensor dispatch is unsupported; no validated replacement was available");
        }
    }
    if (dlss_chain.output_pass) {
        replaced = true; // A replacement upscale may run
        return FinishDlssEvaluation() || dispatched;
    }
    return dispatched;
}

namespace {
constexpr u32 DLSS_NETWORK_CLEAN_EVALUATIONS = 8;
/// Health checks in a row the network must pass before it is shown (about a second: a network that
/// goes bad does so within its first few evaluations), and fail to be dropped.
constexpr u32 DLSS_NETWORK_GOOD_CHECKS = 30;
constexpr u32 DLSS_NETWORK_BAD_CHECKS = 10;
/// A check fails when more than this percentage of its blocks is mismatched (working networks:
/// at most 0.2% on Fortune's Weave's title and Kunlun Base; frozen ones: 6-10%), or when any output
/// sample is NaN or infinite.
constexpr u32 DLSS_CHECK_MISMATCH_PERCENT = 3;
} // Anonymous namespace

bool KeplerCompute::ProcessGameDlss(u64 program_id, u64 native_key) {
    using namespace VideoCommon::FireEmblemUpscale;
    const auto passes = CapturedPasses(program_id);
    if (!passes) {
        return false;
    }
    if (passes->spatial_only && (native_key == passes->prepass || native_key == passes->output)) {
        TraceDlssContract(native_key, false);
    }
    if (native_key == passes->prepass) {
        dlss_chain.spatial_only = passes->spatial_only;
        CaptureFireEmblemJitter();
        dlss_chain.active = true;
        dlss_chain.rejected = 0;
        dlss_chain.dispatches = 0;
        dlss_chain.pending.clear();
    }
    if (!dlss_chain.active) {
        return false;
    }
    dlss_chain.pending.insert(native_key);
    dlss_chain.output_pass = native_key == passes->output;
    if (dlss_chain.spatial_only) {
        return false;
    }
    const bool replacement = dlss_chain.failure != VideoCore::GameDlssReason::None ||
                             GameDlssBlocker() != VideoCore::GameDlssReason::None;
    if (dlss_chain.output_pass) {
        return replacement && FinishDlssEvaluation();
    }
    return dlss_chain.failure != VideoCore::GameDlssReason::None &&
           native_key != passes->prepass &&
           dlss_chain.programs.contains(native_key);
}

// The game's own DLSS network (its translated programs) is shown once every pass of an
// evaluation has run several times in a row and its output has passed several health checks in
// a row. Until then, when any pass can't run, and for the rest of the session once the network
// fails its checks, the replacement upscale draws over the output pass's result. The setting
// "Use the game's own DLSS" defaults off; an explicit native override enables it, while fsr/dlss
// overrides keep the replacement. The network's image sizes are fixed by the game, so it is used
// only without resolution scaling; its matrix multiplies need shaders 32 threads wide.
VideoCore::GameDlssReason KeplerCompute::GameDlssBlocker() const {
    using VideoCore::GameDlssReason;
    static const char* const option = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
    const bool forced_replacement = option && VideoCommon::FireEmblemUpscale::UseReplacement(
        VideoCommon::FireEmblemUpscale::ProgramId,
        VideoCommon::FireEmblemUpscale::OutputProgramKey, option);
    const bool forced_native = option && !forced_replacement;
    if (forced_replacement ||
        (!forced_native && !Settings::values.game_dlss_network.GetValue())) {
        return GameDlssReason::TurnedOff;
    }
    // NVIDIA DLSS, when turned on and running, comes first.
    if (rasterizer->UsesNvidiaDlss()) {
        return GameDlssReason::NvidiaDlss;
    }
    const auto& scale = Settings::values.resolution_info;
    if (scale.up_scale != 1 || scale.down_shift != 0) {
        return GameDlssReason::ResolutionScaling;
    }
    if (!rasterizer->SupportsGameDlssNetwork()) {
        return GameDlssReason::Unsupported;
    }
    return GameDlssReason::None;
}

// Checks the network's output against its input (on the GPU; the result arrives a few frames
// later) and drops the network for the session after DLSS_NETWORK_BAD_CHECKS failed checks in a
// row: a network that produces NaN, freezes on an old image or draws garbage would otherwise stay
// on screen.
void KeplerCompute::CheckDlssNetwork() {
    dlss_chain.checked = rasterizer->CanProbeGameDlssOutput();
    if (!dlss_chain.checked) {
        return;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x34c ||
        !memory_manager.IsFullyMappedRange(cbuf_address, 0x34c)) {
        return;
    }
    const auto surface = [&](u32 offset) -> std::optional<Fermi2D::Surface> {
        const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
        if (tic_index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{tic_index} * sizeof(Texture::TICEntry),
                sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        const auto tic = GetTICEntry(tic_index);
        using RTFormat = Tegra::RenderTargetFormat;
        Fermi2D::Surface result{};
        switch (tic.format) {
        case Texture::TextureFormat::B10G11R11:
            result.format = RTFormat::B10G11R11_FLOAT;
            break;
        case Texture::TextureFormat::R16G16B16A16:
            result.format = RTFormat::R16G16B16A16_FLOAT;
            break;
        case Texture::TextureFormat::A8B8G8R8:
            result.format = RTFormat::A8B8G8R8_UNORM;
            break;
        case Texture::TextureFormat::A2B10G10R10:
            result.format = RTFormat::A2B10G10R10_UNORM;
            break;
        default:
            return std::nullopt;
        }
        if (tic.Address() == 0 || !tic.IsBlockLinear()) {
            return std::nullopt;
        }
        result.linear = Fermi2D::MemoryLayout::BlockLinear;
        result.block_width.Assign(tic.block_width);
        result.block_height.Assign(tic.block_height);
        result.block_depth.Assign(tic.block_depth);
        result.depth = 1;
        result.width = tic.Width();
        result.height = tic.Height();
        result.addr_upper = static_cast<u32>(tic.Address() >> 32);
        result.addr_lower = static_cast<u32>(tic.Address());
        return result;
    };
    const auto input = surface(0x328);
    const auto output = surface(0x348);
    if (!input || !output) {
        return;
    }
    const auto probe = rasterizer->ProbeGameDlssOutput(*input, *output);
    if (!probe) {
        return;
    }
    const bool invalid = probe->invalid_samples != 0;
    const bool mismatched = probe->mismatched * 100 > probe->blocks * DLSS_CHECK_MISMATCH_PERCENT;
    if (!invalid && !mismatched) {
        ++dlss_chain.good_checks;
        dlss_chain.bad_checks = 0;
        return;
    }
    dlss_chain.good_checks = 0;
    if (++dlss_chain.bad_checks < DLSS_NETWORK_BAD_CHECKS) {
        return;
    }
    dlss_chain.failure = invalid ? VideoCore::GameDlssReason::InvalidOutput
                                 : VideoCore::GameDlssReason::OutputMismatch;
    LOG_WARNING(HW_GPU,
                "Game DLSS network: {} in {} checks in a row (last: {}/{} blocks mismatched, worst "
                "{:.3f}, {} invalid samples); the replacement upscale draws the output from now on",
                VideoCore::GameDlssReasonText(dlss_chain.failure), dlss_chain.bad_checks,
                probe->mismatched, probe->blocks, probe->worst, probe->invalid_samples);
}

// Logs what draws the game's DLSS evaluation, when that changes.
void KeplerCompute::UpdateGameDlssStatus(bool complete, VideoCore::GameDlssReason blocker) {
    using VideoCore::GameDlssReason;
    using VideoCore::GameDlssState;
    VideoCore::GameDlssStatus shown{.state = GameDlssState::Checking};
    if (dlss_chain.native) {
        shown.state = GameDlssState::Network;
    } else if (dlss_chain.failure != GameDlssReason::None) {
        shown = {.state = GameDlssState::Replacement, .reason = dlss_chain.failure};
    } else if (blocker != GameDlssReason::None) {
        shown = {.state = GameDlssState::Replacement, .reason = blocker};
    } else if (!complete) {
        shown = {.state = GameDlssState::Replacement,
                 .reason = GameDlssReason::PassesCannotRun,
                 .rejected_passes = dlss_chain.rejected};
    }
    if (shown.state == dlss_chain.status.state && shown.reason == dlss_chain.status.reason &&
        shown.rejected_passes == dlss_chain.status.rejected_passes) {
        return;
    }
    // A failed check was logged as it happened; the network's start is logged above.
    if (shown.state == GameDlssState::Checking) {
        LOG_INFO(HW_GPU, "Game DLSS network: checking its output; the replacement upscale draws "
                         "meanwhile");
    } else if (shown.reason == GameDlssReason::PassesCannotRun) {
        LOG_INFO(HW_GPU,
                 "Game DLSS network: {} pass(es) can't run, the replacement upscale "
                 "draws the output",
                 shown.rejected_passes);
    } else if (shown.state == GameDlssState::Replacement && shown.reason != dlss_chain.failure) {
        LOG_INFO(HW_GPU, "Game DLSS network: {}, the replacement upscale draws the output",
                 VideoCore::GameDlssReasonText(shown.reason));
    }
    dlss_chain.status = shown;
}

// Ends this frame's DLSS evaluation at its output pass: either the network's own result stays,
// or the replacement upscale draws over it. Returns whether the replacement ran.
bool KeplerCompute::FinishDlssEvaluation() {
    dlss_chain.active = false;
    dlss_chain.output_pass = false;
    if (dlss_chain.spatial_only) {
        dlss_chain.pending.clear();
        using namespace VideoCommon::FireEmblemUpscale;
        if (UseReplacement(StarFoxProgramId, StarFoxOutputProgramKey,
                           std::getenv("EDEN_NX2_DLSS_REPLACEMENT"))) {
            return TryFireEmblemUpscale(false);
        }
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        return false;
    }
    dlss_chain.programs.insert(dlss_chain.pending.begin(), dlss_chain.pending.end());
    dlss_chain.pending.clear();
    const bool complete = dlss_chain.rejected == 0;
    dlss_chain.clean_evaluations = complete ? dlss_chain.clean_evaluations + 1 : 0;
    ++dlss_chain.evaluations;
    using VideoCore::GameDlssReason;
    const GameDlssReason blocker = GameDlssBlocker();
    const bool can_show =
        blocker == GameDlssReason::None && dlss_chain.failure == GameDlssReason::None;
    // Checked before the replacement upscale draws over the output. Checks start over whenever
    // the network can't run.
    if (complete && can_show) {
        CheckDlssNetwork();
    } else {
        dlss_chain.good_checks = 0;
        dlss_chain.bad_checks = 0;
    }
    bool native = complete && can_show && dlss_chain.failure == GameDlssReason::None;
    if (!dlss_chain.native) {
        native = native && dlss_chain.clean_evaluations >= DLSS_NETWORK_CLEAN_EVALUATIONS &&
                 (!dlss_chain.checked || dlss_chain.good_checks >= DLSS_NETWORK_GOOD_CHECKS);
    }
    if (native != dlss_chain.native) {
        dlss_chain.native = native;
        if (native) {
            LOG_INFO(HW_GPU, "Game DLSS network: every pass runs{}, its own output is shown",
                     dlss_chain.checked ? " and its output checks out" : "");
        }
    }
    UpdateGameDlssStatus(complete, blocker);
    if (native) {
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        return false;
    }
    return TryFireEmblemUpscale();
}

// Logs the images a rejected program's handles (constant bank 0, bytes 0x80-0x11c) point to, so a
// substitute (e.g. an upscale blit for a tensor-core pass) can find its inputs and output.
void KeplerCompute::DescribeNativeResources(const NativeQmdScheduler::Descriptor& qmd) {
    static u32 described{};
    if (described++ >= 2) return;
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    LOG_INFO(HW_GPU, "Native resources: cbuf0 {:#x} size {:#x}, grid {}x{}x{}, block {}x{}x{}, tic pool {:#x}",
             cbuf_address, cbuf.size.Value(), qmd[12], qmd[13] & 0xffff, qmd[14] & 0xffff,
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value(), regs.tic.Address());
    for (u32 offset = 0x80; offset < 0x120; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(cbuf_address + offset);
        const u32 tic_index = handle & 0xfffff;
        if (handle == 0 || tic_index > regs.tic.limit) continue;
        const auto tic = GetTICEntry(tic_index);
        LOG_INFO(HW_GPU, "  handle @{:#x} = {:#x}: tic {} address {:#x} {}x{} format {:#x} types {}/{}/{}/{}",
                 offset, handle, tic_index, tic.Address(), tic.Width(), tic.Height(),
                 static_cast<u32>(tic.format.Value()), static_cast<u32>(tic.r_type.Value()),
                 static_cast<u32>(tic.g_type.Value()), static_cast<u32>(tic.b_type.Value()),
                 static_cast<u32>(tic.a_type.Value()));
    }
}

std::optional<Texture::TICEntry> KeplerCompute::ConstBuffer0Image(u32 offset) const {
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < offset + 4 ||
        !memory_manager.IsFullyMappedRange(cbuf_address + offset, 4)) {
        return std::nullopt;
    }
    const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
    if (tic_index > regs.tic.limit ||
        !memory_manager.IsFullyMappedRange(regs.tic.Address() +
                                               u64{tic_index} * sizeof(Texture::TICEntry),
                                           sizeof(Texture::TICEntry))) {
        return std::nullopt;
    }
    return GetTICEntry(tic_index);
}

bool KeplerCompute::BypassOotNetwork(u64 program_id, u64 native_key) {
    using namespace VideoCommon::OotNetwork;
    if (!BypassEnabled(program_id)) {
        return false;
    }
    auto* const state = rasterizer->OotNetworkBypass();
    if (!state) {
        return false;
    }
    // The tonemap pass draws before the network every frame; until it has been seen (menus, the
    // first frame) the network runs as usual. Its first passes are skipped only while the output
    // pass is being replaced, so a failed replacement brings the whole network back next frame.
    Maxwell3D::Regs::RenderTargetConfig tonemapped;
    {
        std::scoped_lock lock{state->mutex};
        tonemapped = state->tonemapped;
    }
    if (tonemapped.Address() == 0) {
        return false;
    }
    if (native_key == PrepassProgramKey || native_key == NetworkProgramKey) {
        return state->output_replaced.load(std::memory_order_relaxed);
    }
    if (native_key != OutputProgramKey) {
        return false;
    }
    // The copy needs the tonemapped image to be a B10G11R11 block-linear image of the output's size.
    const auto output = ConstBuffer0Image(OutputImageHandleOffset);
    if (!output || output->format != Texture::TextureFormat::B10G11R11 || output->Address() == 0 ||
        output->Address() == tonemapped.Address() || !output->IsBlockLinear() ||
        tonemapped.format != Tegra::RenderTargetFormat::B10G11R11_FLOAT ||
        tonemapped.tile_mode.is_pitch_linear || tonemapped.width != output->Width() ||
        tonemapped.height != output->Height()) {
        state->output_replaced.store(false, std::memory_order_relaxed);
        static bool warned = false;
        if (!std::exchange(warned, true)) {
            LOG_WARNING(HW_GPU,
                        "Ocarina of Time: unexpected network images; not replaced (tonemapped {}x{}, "
                        "output {})",
                        tonemapped.width, tonemapped.height,
                        output ? fmt::format("{}x{}", output->Width(), output->Height())
                               : std::string{"unreadable"});
        }
        return false;
    }
    Fermi2D::Surface destination{};
    destination.format = Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    destination.linear = Fermi2D::MemoryLayout::BlockLinear;
    destination.block_width.Assign(output->block_width);
    destination.block_height.Assign(output->block_height);
    destination.block_depth.Assign(output->block_depth);
    destination.depth = 1;
    destination.width = output->Width();
    destination.height = output->Height();
    destination.addr_upper = static_cast<u32>(output->Address() >> 32);
    destination.addr_lower = static_cast<u32>(output->Address());
    Fermi2D::Surface source = destination;
    source.block_width.Assign(tonemapped.tile_mode.block_width);
    source.block_height.Assign(tonemapped.tile_mode.block_height);
    source.block_depth.Assign(tonemapped.tile_mode.block_depth);
    source.addr_upper = tonemapped.address_high;
    source.addr_lower = tonemapped.address_low;
    const s32 width = static_cast<s32>(output->Width());
    const s32 height = static_cast<s32>(output->Height());
    const Fermi2D::Config config{
        .operation = Fermi2D::Operation::SrcCopy,
        .filter = Fermi2D::Filter::Point,
        .must_accelerate = true,
        .dst_x0 = 0,
        .dst_y0 = 0,
        .dst_x1 = width,
        .dst_y1 = height,
        .src_x0 = 0,
        .src_y0 = 0,
        .src_x1 = width,
        .src_y1 = height,
    };
    const bool copied = rasterizer->AccelerateSurfaceCopy(source, destination, config);
    state->output_replaced.store(copied, std::memory_order_relaxed);
    static bool logged = false;
    if (!std::exchange(logged, true)) {
        LOG_INFO(HW_GPU,
                 "Ocarina of Time: anti-aliasing network bypassed ({}): {:#x} -> {:#x} {}x{}",
                 copied ? "copy of the tonemapped image" : "copy failed", tonemapped.Address(),
                 output->Address(), width, height);
    }
    return copied;
}

std::atomic<u32> g_nx2_trace_draws_after_upscale{};

// The cbuf0 byte offsets of the texture handles a native program reads (TEX, TLD, TLD4, TXD) and
// the surface handles it stores to (SUST.P), from its code up to the first unconditional EXIT.
std::pair<std::vector<u32>, std::vector<u32>> KeplerCompute::NativeImageHandles(GPUVAddr program) {
    static std::unordered_map<GPUVAddr, std::pair<std::vector<u32>, std::vector<u32>>> cache;
    if (const auto it = cache.find(program); it != cache.end()) {
        return it->second;
    }
    std::pair<std::vector<u32>, std::vector<u32>> handles;
    const auto add = [](std::vector<u32>& list, u32 offset) {
        if (std::find(list.begin(), list.end(), offset) == list.end()) list.push_back(offset);
    };
    std::array<u32, 0x400> block{};
    bool exited = false;
    for (u64 offset = 0; offset < 0x40000 && !exited; offset += sizeof(block)) {
        if (!memory_manager.IsFullyMappedRange(program + offset, sizeof(block))) break;
        memory_manager.ReadBlock(program + offset, block.data(), sizeof(block));
        for (size_t word = 0; word < block.size() && !exited; word += 4) {
            const u32 opcode = block[word] & 0xfff;
            const u32 binding = (block[word + 1] >> 22) & 0x1f;
            const u32 handle = ((block[word + 1] >> 8) & 0x3fff) * 4;
            switch (opcode) {
            case 0xb60: case 0xb63: case 0xb66: case 0xb6d:
                if (binding == 0) add(handles.first, handle);
                break;
            case 0xb9b:
                if (binding == 0) add(handles.second, handle);
                break;
            case 0x94d: // EXIT, unconditional when guarded by PT
                exited = ((block[word] >> 12) & 0xf) == 7;
                break;
            default:
                break;
            }
        }
    }
    return cache.emplace(program, std::move(handles)).first->second;
}

void KeplerCompute::RememberNativeUpscaleInput() {
    native_upscale_input.reset();
    const auto [reads, stores] = NativeImageHandles(regs.code_loc.Address());
    // An image-fed matrix pass writes its intermediate results to buffers. Retain its colour
    // input: the final pass's sampled image is another intermediate that a skipped network never
    // produced (Ocarina's movie network reads its colour at +0x58).
    if (reads.size() != 1 || !stores.empty()) return;
    const auto& cbuf = launch_description.const_buffer_config[0];
    if (reads.front() + sizeof(u32) > cbuf.size.Value()) return;
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    const u32 handle = memory_manager.Read<u32>(address + reads.front());
    if (handle == 0 || (handle & 0xfffff) > regs.tic.limit) return;
    const auto input = GetTICEntry(handle & 0xfffff);
    if (input.Address() != 0 && input.IsBlockLinear() && input.Width() >= 64 &&
        input.Height() >= 64) {
        native_upscale_input = input;
    }
}

// Fire Emblem: Fortune's Weave's first DLSS pass (rejected, before the network) has the input colour
// at cbuf0+0x278, its motion vectors (R16G16 float, the input's size) at +0x290, its depth (Z32, the
// input's size) at +0x298 and this frame's camera jitter (x, y; input pixels, Halton bases 2 and 3
// less 0.5) at +0x1b0.
void KeplerCompute::RememberDlssMotionVectors() {
    dlss_motion_vectors.reset();
    dlss_depth.reset();
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x294 ||
        !memory_manager.IsFullyMappedRange(cbuf_address, cbuf.size.Value())) {
        return;
    }
    const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
        const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
        if (tic_index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{tic_index} * sizeof(Texture::TICEntry),
                sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        return GetTICEntry(tic_index);
    };
    const auto color = image(0x278), motion = image(0x290);
    if (!color || !motion || color->format != Texture::TextureFormat::B10G11R11 ||
        motion->format != Texture::TextureFormat::R16G16 || motion->Address() == 0 ||
        motion->Width() != color->Width() || motion->Height() != color->Height() ||
        !motion->IsBlockLinear())
        return;
    dlss_motion_vectors = *motion;
    dlss_motion_color = color->Address();
    const auto depth = cbuf.size.Value() >= 0x29c ? image(0x298) : std::nullopt;
    if (depth && depth->format == Texture::TextureFormat::Z32 && depth->Address() != 0 &&
        depth->Width() == color->Width() && depth->Height() == color->Height() &&
        depth->IsBlockLinear()) {
        dlss_depth = *depth;
    } else {
        dlss_depth.reset();
    }
}

void KeplerCompute::CaptureFireEmblemJitter() {
    using namespace VideoCommon::FireEmblemUpscale;
    (void)fe_upscale_jitter.Take();
    dlss_motion_vectors.reset();
    dlss_depth.reset();
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x27c ||
        !memory_manager.IsFullyMappedRange(address, 0x27c)) {
        return;
    }
    const u32 index = memory_manager.Read<u32>(address + PrepassColorOffset) & 0xfffff;
    if (index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
            regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry), sizeof(Texture::TICEntry))) {
        return;
    }
    const auto color = GetTICEntry(index);
    if (!ValidInput(color, dlss_chain.spatial_only)) {
        return;
    }
    fe_upscale_jitter.Capture(color.Address(), color.Width(), color.Height(),
        {std::bit_cast<f32>(memory_manager.Read<u32>(address + JitterOffset)),
         std::bit_cast<f32>(memory_manager.Read<u32>(address + JitterOffset + sizeof(f32)))});
    if (!dlss_chain.spatial_only) {
        RememberDlssMotionVectors();
    }
}

void KeplerCompute::TraceDlssContract(u64 native_key, bool warn_unvalidated) {
    static bool reported{};
    if (warn_unvalidated && !reported) {
        reported = true;
        LOG_WARNING(HW_GPU, "Fast Fusion temporal replacement is not selected: motion, depth, jitter and reset contract is unvalidated; native dispatches retained. EDEN_NX2_DLSS_CONTRACT_TRACE enables bounded evidence capture");
    }
    static const bool enabled = std::getenv("EDEN_NX2_DLSS_CONTRACT_TRACE") != nullptr;
    static const u32 skip = [] {
        const char* value = std::getenv("EDEN_NX2_DLSS_CONTRACT_SKIP");
        return value ? static_cast<u32>(std::min<unsigned long>(std::strtoul(value, nullptr, 10), 1000000UL)) : 0U;
    }();
    if (!enabled || dlss_contract_trace_count >= skip + 24) {
        return;
    }
    if (dlss_contract_trace_count++ < skip) {
        return;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    const u32 bytes = std::min<u32>(cbuf.size.Value(), 0x360U);
    if (!(launch_description.const_buffer_enable_mask & 1) || bytes < 0x100 ||
        !memory_manager.IsFullyMappedRange(address, bytes)) {
        return;
    }
    LOG_INFO(HW_GPU, "DLSS contract candidate {:016x}: cb0={:#x} bytes={:#x} grid={}x{}x{} block={}x{}x{} native dispatch retained",
             native_key, address, bytes, launch_description.grid_dim_x.Value(),
             launch_description.grid_dim_y.Value(), launch_description.grid_dim_z.Value(),
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value());
    for (u32 offset = 0x100; offset + 16 <= bytes; offset += 16) {
        std::array<u32, 4> words{};
        memory_manager.ReadBlock(address + offset, words.data(), sizeof(words));
        LOG_INFO(HW_GPU, "DLSS contract cb0+{:03x}: {:08x}", offset, fmt::join(words, " "));
    }
    for (u32 offset = 0x278; offset + 4 <= bytes; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(address + offset);
        const u32 index = handle & 0xfffff;
        const GPUVAddr tic_address = regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry);
        if (!handle || index > regs.tic.limit ||
            !memory_manager.IsFullyMappedRange(tic_address, sizeof(Texture::TICEntry))) {
            continue;
        }
        const auto tic = GetTICEntry(index);
        std::array<u32, sizeof(Texture::TICEntry) / sizeof(u32)> words{};
        std::memcpy(words.data(), &tic, sizeof(tic));
        LOG_INFO(HW_GPU, "DLSS contract handle+{:03x}={:08x}: tic={} addr={:#x} {}x{} raw={:08x}",
                 offset, handle, index, tic.Address(), tic.Width(), tic.Height(),
                 fmt::join(words, " "));
    }
}

bool KeplerCompute::TryFireEmblemUpscale(bool allow_history) {
    using namespace VideoCommon::FireEmblemUpscale;
    static const bool requested_history = [] {
        const char* const option = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return option && (std::string_view{option} == "fsr" || std::string_view{option} == "dlss");
    }();
    const bool use_history = allow_history && requested_history;
    const size_t game_index = dlss_chain.spatial_only ? 1 : 0;
    const char* const game = dlss_chain.spatial_only ? "Star Fox" : "Fire Emblem";
    const auto jitter = fe_upscale_jitter.Take();
    const auto motion = std::exchange(dlss_motion_vectors, std::nullopt);
    const auto depth = std::exchange(dlss_depth, std::nullopt);
    if (!jitter || (use_history && (!motion || !depth))) {
        static std::array<bool, 2> reported{};
        if (!reported[game_index]) {
            reported[game_index] = true;
            LOG_WARNING(HW_GPU, "{} replacement lacks current inputs: jitter={} motion={} depth={} temporal={}; retaining native dispatch",
                        game, jitter.has_value(), motion.has_value(), depth.has_value(), use_history);
        }
        return false;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x34c ||
        !memory_manager.IsFullyMappedRange(address, 0x34c)) {
        return false;
    }
    const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
        const u32 index = memory_manager.Read<u32>(address + offset) & 0xfffff;
        if (index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry), sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        return GetTICEntry(index);
    };
    const auto input = image(OutputColorInputOffset);
    const auto output = image(OutputColorOffset);
    if (!input || !output || !ValidPair(*input, *output, dlss_chain.spatial_only) ||
        jitter->color != input->Address() || (use_history && dlss_motion_color != input->Address()) ||
        jitter->width != input->Width() || jitter->height != input->Height()) {
        return false;
    }
    const auto surface = [](const Texture::TICEntry& tic, Tegra::RenderTargetFormat format) {
        Fermi2D::Surface result{};
        result.format = format;
        result.linear = Fermi2D::MemoryLayout::BlockLinear;
        result.block_width.Assign(tic.block_width);
        result.block_height.Assign(tic.block_height);
        result.block_depth.Assign(tic.block_depth);
        result.depth = 1;
        result.width = tic.Width();
        result.height = tic.Height();
        result.addr_upper = static_cast<u32>(tic.Address() >> 32);
        result.addr_lower = static_cast<u32>(tic.Address());
        return result;
    };
    const auto input_format = dlss_chain.spatial_only ? Tegra::RenderTargetFormat::A8B8G8R8_SRGB
                                                     : Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    const auto output_format = dlss_chain.spatial_only ? Tegra::RenderTargetFormat::R16G16B16A16_FLOAT
                                                      : Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    const bool done = rasterizer->AccelerateTemporalUpscale(
        surface(*input, input_format), surface(*output, output_format),
        use_history ? surface(*motion, Tegra::RenderTargetFormat::R16G16_FLOAT) : Fermi2D::Surface{},
        use_history ? &*depth : nullptr, jitter->offset,
        use_history ? std::array{1.0f, 1.0f} : std::array{0.0f, 0.0f}, use_history);
    static std::array<u32, 2> logged{};
    if (logged[game_index]++ < 8) {
        LOG_INFO(HW_GPU, "{} DLSS replacement: {} temporal={} input={:#x} {}x{} output={:#x} {}x{} jitter={},{}",
                 game, done ? "applied" : "unavailable; retaining native dispatch", use_history,
                 input->Address(), input->Width(), input->Height(), output->Address(),
                 output->Width(), output->Height(), jitter->offset[0], jitter->offset[1]);
    }
    return done;
}

void KeplerCompute::ProcessLaunch() {
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    rasterizer->DispatchCompute();
}

Texture::TICEntry KeplerCompute::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tic.Address() + tic_index * sizeof(Texture::TICEntry)};

    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry KeplerCompute::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tsc.Address() + tsc_index * sizeof(Texture::TSCEntry)};

    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2): the front
// runs compute macros against its own copy of the native registers (which hold the last value
// written to each, on the back too) and loads and binds the compute macro code itself.
struct KeplerCompute::FrontState {
    std::array<u32, 0x1000> registers{};
    std::optional<u32> executing_macro;
    std::vector<u32> parameters;
    bool compute_uploaded{};
    u32 bind_pointer{};
    MacroEngine* macro{}; ///< Without a native context: the one macro engine
    /// EDEN_NX2_COMPUTE_SINK: plain registers (native_plain_methods) this front sent a write to.
    /// The back stored that value in both native_registers and regs, so `registers` holds what
    /// both hold and a later write of the same value can be dropped.
    std::bitset<0xe00> plain_sent;
    u64 calls{}; ///< Macro calls, for EDEN_NX2_SPLIT_VERIFY
    /// The inline upload's destination from its last exec_upload (FrontNoteUpload), if known.
    GPUVAddr upload_address{};
    u64 upload_size{};
    bool upload_known{};
    /// EXP13=2 (FrontPatchOverlays): the bytes of the upload its data words filled so far, and
    /// whether it is linear and one line (its bytes land at upload_address as they are).
    u64 upload_offset{};
    bool upload_contiguous{};
    /// EXP13=2: the shadow of the upload (FrontQmdMirror::Shadow::id), 0 for none.
    u64 upload_shadow{};
    /// EXP13=2: the upload's write-stamp tag (its own stamps don't count against its patches),
    /// and its data words not yet given to the copy, with the position of the last one sent.
    u64 upload_tag{};
    std::vector<u32> upload_batch;
    u64 upload_batch_position{};
};

namespace {

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's compute registers with the back's (logs and resynchronizes differences).
bool ComputeFrontVerifyEnabled() {
    static const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    return verify;
}

} // Anonymous namespace

KeplerCompute::FrontState& KeplerCompute::Front(VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle.
        split.Drain(VideoCommon::DrainSite::FrontStart);
        front = std::make_unique<FrontState>();
        front->registers = native_registers;
        front->executing_macro = native_executing_macro;
        front->parameters = std::move(native_parameters);
        native_parameters.clear();
        native_executing_macro.reset();
        front->compute_uploaded = native_compute_uploaded;
        front->bind_pointer = native_bind_pointer;
        front->macro = native_macro;
        front_macros = true;
    }
    return *front;
}

void KeplerCompute::FrontApply(FrontState& fs, u32 method, u32 argument) {
    // As ProcessNativeMethod does with the macro code and bindings.
    if (method == 0x45 || method == 0x47) {
        fs.compute_uploaded = true;
    }
    fs.registers[method] = argument;
    switch (method) {
    case 0x107:
        // Keeps the copy exact (front macros read it; EDEN_NX2_COMPUTE_SINK relies on it): an
        // inline QMD whose destination is mapped launches through ProcessNativeMethod(0xb0, 3) on
        // the back, which leaves 3 in 0xb0 (0xad is restored). Without this the copy kept the
        // last 0xb0 the front sent.
        {
            const GPUVAddr address =
                ((GPUVAddr{fs.registers[0xc6]} << 32) | fs.registers[0xc7]) << 8;
            const u32 control = fs.registers[0xa7];
            if (memory_manager.IsFullyMappedRange(address, 256) &&
                ((control & (1U << 24)) || !(fs.registers[0xcc] & (1U << 12)))) {
                fs.registers[0xb0] = 3;
            }
        }
        break;
    case 0x45:
        FrontMacroEngine(fs).ClearCode(argument);
        break;
    case 0x46:
        FrontMacroEngine(fs).AddCode(0, argument);
        break;
    case 0x47:
        fs.bind_pointer = argument;
        break;
    case 0x48: {
        const auto positions = FrontMacroPositions(fs);
        if (fs.bind_pointer < positions.size()) {
            positions[fs.bind_pointer++] = argument;
        } else {
            LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        }
        break;
    }
    default:
        break;
    }
}

MacroEngine& KeplerCompute::FrontMacroEngine(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? native_context->compute_engine : native_context->engine;
    }
    return *fs.macro;
}

std::span<u32> KeplerCompute::FrontMacroPositions(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? std::span<u32>{native_context->compute_positions}
                                      : std::span<u32>{native_context->positions};
    }
    return native_macro_positions;
}

namespace {
// EDEN_NX2_MACRO_TRACE: the compute methods after a traced macro call (direct and macro writes).
int g_follow_left{};
std::string g_follow_text;
void FollowNote(const char* kind, u32 method, u32 value) {
    if (g_follow_left <= 0) [[likely]] {
        return;
    }
    g_follow_text += fmt::format(" {}{:x}={:x}", kind, method, value);
    if (--g_follow_left == 0) {
        LOG_INFO(HW_GPU, "after the traced macro:{}", g_follow_text);
        g_follow_text.clear();
    }
}
} // Anonymous namespace

void KeplerCompute::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                    u32 method, u32 argument, bool is_last_call,
                                    const GPUVAddr* segment) {
    auto& fs = Front(split);
    if (!fs.upload_batch.empty() && method != KEPLER_COMPUTE_REG_INDEX(data_upload)) {
        FrontFlushUploads(split, fs); // Before anything else may read or write a QMD
    }
    FollowNote("", method, argument);
    if (method >= fs.registers.size()) {
        split.Method(this, method, argument, is_last_call, segment);
        split.NoteUnboundedWriter(VideoCommon::GpuSplit::WriterKind::Compute);
        return;
    }
    if (method >= 0xe00) {
        if (!fs.executing_macro) {
            if (method & 1) {
                LOG_ERROR(HW_GPU, "Native compute macro starts on argument method");
                return;
            }
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    if (FrontPlainWrite(split, fs, method, argument)) {
        return;
    }
    FrontApply(fs, method, argument);
    split.Method(this, method, argument, is_last_call, segment, FrontEffect(method));
    FrontNoteWrites(split, fs, method);
    FrontNoteWriter(split, fs, method, argument);
    if (method == KEPLER_COMPUTE_REG_INDEX(data_upload) &&
        VideoCommon::QmdPrediction::MirrorMode()) {
        // EXP13=2: the copy takes an entry's words at once (6f (c) b).
        fs.upload_batch.push_back(argument);
        fs.upload_batch_position = split.SentPosition();
        if (is_last_call || fs.upload_batch.size() >= 1024) {
            FrontFlushUploads(split, fs);
        }
    }
    if (method == 0x6c3) {
        FrontNoteRelease(split, fs, argument);
    }
}

void KeplerCompute::FrontNoteRelease(VideoCommon::GpuSplit& split, const FrontState& fs,
                                     u32 control) const {
    // ProcessNativeReportSemaphore: a release (operation 0) without reduction writes the payload.
    if ((control & 3) != 0 || ((control >> 3) & 1) != 0) {
        return;
    }
    const GPUVAddr address =
        (GPUVAddr{fs.registers[0x6c0] & 0x1ffffff} << 32) | fs.registers[0x6c1];
    if (const auto device = memory_manager.GpuToCpuAddress(address)) {
        split.NoteRelease(*device, address, fs.registers[0x6c2]);
        if (VideoCommon::QmdPrediction::MirrorMode()) {
            // EXP13=2: recorded so the reduction bound sees it queued or run but not landed.
            namespace Prediction = VideoCommon::QmdPrediction;
            Prediction::Note(*device,
                             {.writer = Prediction::Writer::Report,
                              .kind = Prediction::Kind::Plain,
                              .payload = fs.registers[0x6c2],
                              .op = 0,
                              .is_signed = false},
                             memory_manager.Read<u32>(address), split.SentPosition(), 0);
        }
    }
}

VideoCommon::SplitWriteTracker::Effect KeplerCompute::FrontEffect(u32 method) const {
    using Effect = VideoCommon::SplitWriteTracker::Effect;
    if (method < native_plain_methods.size() && native_plain_methods[method]) {
        return Effect::None;
    }
    switch (method) {
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
        return Effect::None; // Macros: loaded and bound by the front
    case 0xae:
    case 0xaf:
    case 0xb0:
    case 0x107:
    case 0x6c3:
        return Effect::Gpu; // QMD launches and the semaphore releases after them
    default:
        return Effect::Unknown; // Inline uploads among them
    }
}

void KeplerCompute::FrontNoteWriter(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                    u32 argument) {
    // What ProcessNativeMethod writes memory for: QMD launches and inline QMDs dispatch, a report
    // semaphore up to 16 bytes, the inline upload its destination. The other registers
    // below the macros (plain or not), the macro code and bindings and the QMD actions it
    // ignores write nothing.
    using Kind = VideoCommon::GpuSplit::WriterKind;
    const auto predict = [&] {
        if (VideoCommon::QmdPrediction::Enabled()) {
            FrontPredictQmd(split, fs, method, argument);
        } else if (g_front_qmd.synced || g_front_qmd.disabled) [[unlikely]] {
            StopFrontQmd(false);
        }
    };
    switch (method) {
    case 0xaf:
        if ((argument & 2) != 0) {
            split.NoteDispatchWriter();
            predict();
        }
        return;
    case 0xb0:
        if ((argument & 15) > 1) {
            split.NoteDispatchWriter();
            predict();
        }
        return;
    case 0x107:
        split.NoteDispatchWriter();
        predict();
        return;
    case 0x6c3: {
        const GPUVAddr address =
            (GPUVAddr{fs.registers[0x6c0] & 0x1ffffff} << 32) | fs.registers[0x6c1];
        // EXP13=2: a reduction (bit 3, op in bits 11:9, always unsigned) recorded for the
        // reduction bound (ProcessNativeReportSemaphore; operations 0 and 2 write).
        if (const u32 report_operation = argument & 3;
            ((argument >> 3) & 1) != 0 && (report_operation == 0 || report_operation == 2) &&
            VideoCommon::QmdPrediction::MirrorMode()) {
            if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                namespace Prediction = VideoCommon::QmdPrediction;
                Prediction::Note(*device,
                                 {.writer = Prediction::Writer::Report,
                                  .kind = Prediction::Kind::Reduction,
                                  .payload = fs.registers[0x6c2],
                                  .op = (argument >> 9) & 7,
                                  .is_signed = false},
                                 memory_manager.Read<u32>(address), split.SentPosition(), 0);
            }
        }
        if ((VideoCore::Nx2Debug::Experiment(10) & 4) != 0) {
            split.NoteWriterRange(memory_manager, address, 16, Kind::Release); // As before (A/B)
            return;
        }
        // ProcessNativeReportSemaphore: operations 1 and 3 write nothing; a reduction writes 4
        // bytes from a fence callback, translated then; a release goes to the query cache, which
        // translates it at once (nothing when unmapped) and writes 4 bytes for a one-word report
        // (a fence), else the 8-byte value and its timestamp.
        const u32 operation = argument & 3;
        if (operation != 0 && operation != 2) {
            split.CountRelease(VideoCommon::GpuSplit::ReleaseNote::NoWrite);
            return;
        }
        const bool reduction = ((argument >> 3) & 1) != 0;
        const bool one_word = ((argument >> 28) & 1) != 0;
        split.NoteReleaseWriter(memory_manager, address, reduction || one_word ? 4 : 16,
                                !reduction);
        return;
    }
    case KEPLER_COMPUTE_REG_INDEX(exec_upload):
        FrontNoteUpload(fs, argument);
        fs.upload_tag = split.NewWriteStampTag();
        return;
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        if (fs.upload_known) {
            split.SetWriteStampTag(fs.upload_tag);
            split.NoteWriterRange(memory_manager, fs.upload_address, fs.upload_size, Kind::Upload);
            split.SetWriteStampTag(0);
        } else {
            split.NoteUnboundedWriter(Kind::Upload);
        }
        return;
    default:
        if (method >= 0xe00) {
            split.NoteUnboundedWriter(Kind::Compute);
        }
        return;
    }
}

void KeplerCompute::FrontPredictQmd(VideoCommon::GpuSplit& split, const FrontState& fs,
                                    u32 method, u32 argument) const {
    // The QMD Signal reads on the back (native_qmd.h): the one the action names, then its
    // dependent QMD (one level). Schedules (2, 3) run it once its dependency count is 0, a
    // decrement (7) when it reaches 0; increments and the rest run nothing.
    namespace Prediction = VideoCommon::QmdPrediction;
    if (Prediction::MirrorMode()) {
        FrontMirrorQmd(split, fs, method, argument);
        return;
    }
    if (g_front_qmd.synced || g_front_qmd.disabled) [[unlikely]] {
        StopFrontQmd(false);
    }
    using Descriptor = NativeQmdScheduler::Descriptor;
    const u64 position = split.SentPosition();
    const auto predict = [&](GPUVAddr qmd_address, const Descriptor& qmd, bool stale) {
        for (const u32 base : {24U, 28U, 52U}) {
            const u32 flags = qmd[base + 1];
            if (!(flags & (1U << 23))) {
                continue;
            }
            if (flags & (1U << 26)) {
                Prediction::NoteReduction();
                continue;
            }
            const GPUVAddr destination = (GPUVAddr{flags & 0xff} << 32) | qmd[base];
            if (const auto device = memory_manager.GpuToCpuAddress(destination)) {
                Prediction::Predict(Prediction::Writer::Qmd, *device, qmd[base + 2], position,
                                    qmd_address, stale);
            } else {
                Prediction::NoteUnmappedRelease();
            }
        }
    };
    const auto read = [&](GPUVAddr qmd_address, Descriptor& qmd, bool& stale) {
        if (!memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            Prediction::NoteSignal(Prediction::Signal::Unmapped);
            return false;
        }
        memory_manager.ReadBlockUnsafe(qmd_address, qmd.data(), sizeof(qmd));
        stale = split.HasQueuedMemoryWriters(qmd_address, qmd_address + sizeof(qmd));
        return true;
    };
    Descriptor qmd{};
    GPUVAddr qmd_address{};
    u32 action = argument & 15;
    bool stale = false;
    if (method == 0x107) {
        // An inline QMD: the back writes the registers' copy to memory, then schedules it.
        qmd_address = ((GPUVAddr{fs.registers[0xc6]} << 32) | fs.registers[0xc7]) << 8;
        if (!memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd)) ||
            !((fs.registers[0xa7] & (1U << 24)) || !(fs.registers[0xcc] & (1U << 12)))) {
            return;
        }
        std::copy_n(fs.registers.begin() + 0xc8, qmd.size(), qmd.begin());
        action = 3;
    } else {
        qmd_address = GPUVAddr{fs.registers[0xad]} << 8;
        if ((action != 2 && action != 3 && action != 7) || !read(qmd_address, qmd, stale)) {
            return;
        }
    }
    Prediction::NoteSignal(Prediction::Signal::Read);
    if (action != 7 && ((qmd[4] >> 14) & 0xffff) != 0) {
        Prediction::NoteSignal(Prediction::Signal::Waiting);
    }
    predict(qmd_address, qmd, stale);
    if (!(qmd[16] & 1)) {
        return;
    }
    const u32 dependent_action = (qmd[16] >> 1) & 7;
    const GPUVAddr dependent = GPUVAddr{qmd[15]} << 8;
    if ((dependent_action == 4 || dependent_action == 3 || dependent_action == 1) &&
        read(dependent, qmd, stale)) {
        Prediction::NoteSignal(Prediction::Signal::Chained);
        predict(dependent, qmd, stale);
    }
}

void KeplerCompute::FrontMirrorQmd(VideoCommon::GpuSplit& split, const FrontState& fs,
                                   u32 method, u32 argument) const {
    if (method != 0x107) {
        // 0xaf with bit 1, 0xb0 with an action past 1: what ProcessNativeMethod signals.
        FrontQmdSignal(split, GPUVAddr{fs.registers[0xad]} << 8, argument);
        return;
    }
    // An inline QMD goes to memory, then may launch (ProcessNativeMethod's 0x107); unmapped, it
    // neither writes nor launches.
    const GPUVAddr address = ((GPUVAddr{fs.registers[0xc6]} << 32) | fs.registers[0xc7]) << 8;
    if (!memory_manager.IsFullyMappedRange(address, sizeof(NativeQmdScheduler::Descriptor))) {
        return;
    }
    if (g_front_qmd.synced) {
        const u64 position = split.SentPosition();
        g_front_qmd.AddOverlay(&memory_manager, address, position, fs.registers.data() + 0xc8,
                               true);
        // A shadow holding the block takes the words too (6f (a)).
        constexpr u64 size = FrontQmdMirror::BlockBytes;
        for (FrontQmdMirror::Shadow& shadow : g_front_qmd.shadows) {
            if (shadow.id == 0 || shadow.space != &memory_manager || address < shadow.address ||
                shadow.address + shadow.size < address + size ||
                (address - shadow.address) % size != 0) {
                continue;
            }
            std::memcpy(shadow.bytes.data() + (address - shadow.address),
                        fs.registers.data() + 0xc8, size);
            FrontQmdMirror::StampBlock(shadow,
                                       static_cast<size_t>((address - shadow.address) / size),
                                       position);
        }
    }
    if ((fs.registers[0xa7] & (1U << 24)) == 0 && (fs.registers[0xcc] & (1U << 12)) != 0) {
        return; // Launched later, by 0xaf or 0xb0
    }
    FrontQmdSignal(split, address, 3);
}

void KeplerCompute::FrontQmdSignal(VideoCommon::GpuSplit& split, GPUVAddr qmd_address,
                                   u32 argument) const {
    namespace Prediction = VideoCommon::QmdPrediction;
    auto& mirror = g_front_qmd;
    if (mirror.disabled) {
        return;
    }
    if (!mirror.synced || mirror.session != split.Session()) {
        // First signal of this split session or of EXP13=2: start from the back's table, this
        // signal included.
        ResyncFrontQmd(split, Prediction::Mirror::SessionStart);
        return;
    }
    Prediction::NoteSignal(Prediction::Signal::Read);
    const u64 position = split.SentPosition();
    bool unknown = false;
    u64 running_qmd = 0; // The QMD the releases below belong to (read last), as on the back
    Prediction::Record read_info{}; // Where that read came from
    // Simple (6g (c)): a schedule of this QMD that loads it with no deferred decrements (action
    // 3 always loads; 2 only when not loaded yet), and the QMD read at its run has no base
    // dependency count; its releases then come from this very signal on the back too.
    const u32 action = argument & 15;
    bool schedule_simple = action == 3 || action == 2;
    if (const auto entry = mirror.table.find(qmd_address); entry != mirror.table.end()) {
        schedule_simple = schedule_simple && entry->second.deferred == 0 &&
                          (action == 3 || !entry->second.loaded);
    }
    u32 read_base = ~0U;
    const auto read = [&](u64 address, NativeQmdScheduler::Descriptor& descriptor) {
        running_qmd = address;
        read_info = {};
        const bool ok = FrontReadQmd(split, address, descriptor, unknown, read_info);
        read_base = ok ? (descriptor[4] >> 14) & 0xffff : ~0U;
        Prediction::NoteReadSource(read_info.source);
        mirror.NoteRead(address, position, &memory_manager, read_info.source);
        return ok;
    };
    // As ProcessNativeMethod's release callback: a reduction (bit 26), else a four-word release
    // (payload and time stamp), a max (bit 19) or a plain write of the payload.
    const auto release = [&](u64 address, u32 payload, u32 flags) {
        const bool four_words = (flags >> 30) == 0;
        if (!memory_manager.IsFullyMappedRange(address, four_words ? 16 : sizeof(payload))) {
            return false;
        }
        const auto device = memory_manager.GpuToCpuAddress(address);
        if (!device) {
            Prediction::NoteUnmappedRelease();
            return true;
        }
        Prediction::Record record{.writer = Prediction::Writer::Qmd,
                                  .kind = Prediction::Kind::Plain,
                                  .payload = payload,
                                  .op = 0,
                                  .is_signed = false,
                                  .source = read_info.source,
                                  .read_done = read_info.read_done,
                                  .overlay_position = read_info.overlay_position,
                                  .block_position = read_info.block_position,
                                  .simple = schedule_simple && running_qmd == qmd_address &&
                                            read_base == 0};
        if ((flags & (1U << 26)) != 0) {
            record.kind = Prediction::Kind::Reduction;
            record.op = (flags >> 20) & 7;
            record.is_signed = ((flags >> 24) & 1) != 0;
        } else if (!four_words && (flags & (1U << 19)) != 0) {
            record.kind = Prediction::Kind::AtLeast;
        }
        Prediction::Note(*device, record, memory_manager.Read<u32>(address), position,
                         running_qmd);
        return true;
    };
    const auto dispatch = [](const NativeQmdScheduler::Descriptor&) { return true; };
    (void)mirror.scheduler.Signal(qmd_address, argument, read, release, dispatch);
    // Every release of the signal is recorded: the back's ones are compared now.
    Prediction::Settle(position);
    if (unknown) {
        // The back's run of this signal may differ from the copy's: take its table.
        ResyncFrontQmd(split, Prediction::Mirror::Resync);
        ++mirror.window_resyncs;
    }
    // Reads that wait for the back (QMDs written by queued work other than inline QMDs) or that
    // drain would cost the front more than the copy saves: #717 turns it off then.
    // EDEN_NX2_QMD_AUTO_OFF=0 keeps it on however many reads wait (they still count).
    static const bool auto_off = [] {
        const char* const value = std::getenv("EDEN_NX2_QMD_AUTO_OFF");
        return value == nullptr || value[0] != '0';
    }();
    // A grace window (fable_answer_6 6e): no decision in the session's first 4096 reads (boot
    // schedules QMDs before any has an overlay), then two 1024-read windows in a row over 1 in 8.
    if (mirror.window_reads >= 1024 || mirror.window_resyncs > 8) {
        const bool over = (auto_off && mirror.window_waits * 8 > mirror.window_reads) ||
                          mirror.window_resyncs > 8;
        mirror.session_reads += mirror.window_reads;
        mirror.strikes = over ? mirror.strikes + 1 : 0;
        if (over && mirror.strikes >= 2 && mirror.session_reads >= 4096) {
            LOG_WARNING(HW_GPU,
                        "QMD release prediction (EXP13=2): {} of {} QMD reads waited for queued "
                        "writes, {} resyncs, the second window in a row; the QMD scheduler copy "
                        "is off until EXP13 changes",
                        mirror.window_waits, mirror.window_reads, mirror.window_resyncs);
            Prediction::NoteMirror(Prediction::Mirror::AutoOff);
            StopFrontQmd(true);
            return;
        }
        mirror.window_reads = 0;
        mirror.window_waits = 0;
        mirror.window_resyncs = 0;
    }
}

bool KeplerCompute::FrontReadQmd(VideoCommon::GpuSplit& split, u64 address,
                                 NativeQmdScheduler::Descriptor& descriptor, bool& unknown,
                                 VideoCommon::QmdPrediction::Record& info) const {
    namespace Prediction = VideoCommon::QmdPrediction;
    constexpr u64 size = sizeof(NativeQmdScheduler::Descriptor);
    if (!memory_manager.IsFullyMappedRange(address, size)) {
        return false; // As the back ("unmapped QMD")
    }
    // Downloads of GPU-written words may land on the back at any time.
    const auto device = memory_manager.GpuToCpuAddress(address);
    if (!device || Nx2MayBeGpuWritten(*device, size)) {
        unknown = true;
        return false;
    }
    auto& mirror = g_front_qmd;
    ++mirror.window_reads;
    const u64 done = split.ConsumedNow();
    // The newest write that may have reached the QMD (write stamps, 6f (c) a).
    u64 target = split.WriteStamp(*device, size, address);
    const FrontQmdMirror::Overlay* const overlay = mirror.FindOverlay(&memory_manager, address);
    // The inline QMD or upload written last over it, nothing queued after it that may write
    // there: the back finds exactly its words (0x107 writes all 256 bytes, the shadow's upload
    // all of its filled bytes; the patches since are applied to both).
    const bool overlay_ok =
        overlay != nullptr && overlay->position > done && target <= overlay->position;
    if (overlay == nullptr && mirror.evicted > done) {
        target = (std::max)(target, mirror.evicted); // Its inline QMD may have been evicted
    }
    // An inline QMD queued over it is no writer QueuedGuestWriters sees: newer than the shadow,
    // it wins (or, if its overlay can't serve the read, the read waits).
    const u64 inline_position =
        overlay != nullptr && overlay->inline_qmd && overlay->position > done ? overlay->position
                                                                              : 0;
    size_t block = 0;
    FrontQmdMirror::Shadow* const shadow = mirror.FindShadow(&memory_manager, address, block);
    const u64 block_position = shadow != nullptr ? shadow->block_position[block] : 0;
    const bool shadow_ok = shadow != nullptr && block_position > done &&
                           (std::max)(target, inline_position) <= block_position;
    info.read_done = done;
    info.overlay_position = overlay != nullptr ? overlay->position : 0;
    info.block_position = block_position;
    if (overlay_ok && (!shadow_ok || overlay->position >= block_position)) {
        descriptor = overlay->words;
        info.source = !overlay->inline_qmd ? Prediction::Source::OverlayFromMemory
                      : overlay->patched   ? Prediction::Source::PatchedOverlay
                                           : Prediction::Source::InlineOverlay;
        Prediction::NoteMirror(Prediction::Mirror::Overlay);
        return true;
    }
    if (shadow_ok) {
        std::memcpy(descriptor.data(), shadow->bytes.data() + (address - shadow->address), size);
        shadow->last_use = ++mirror.use_clock;
        info.source = Prediction::Source::Shadow;
        Prediction::NoteMirror(Prediction::Mirror::Shadow);
        return true;
    }
    info.source = target > done ? Prediction::Source::Waited : Prediction::Source::Memory;
    if (target > done) {
        // Diagnostics only (reads rarely wait now): what set the target.
        const auto writers = split.QueuedGuestWriters(done, address, *device, size);
        // What set the target, for the wait cause histogram.
        using Cause = VideoCommon::GuestWriterCause;
        Cause cause = writers.noted >= writers.unbounded ? writers.noted_cause
                                                         : writers.unbounded_cause;
        const char* what = writers.noted >= writers.unbounded ? nullptr : writers.what;
        if (overlay == nullptr && mirror.evicted > (std::max)(writers.noted, writers.unbounded)) {
            cause = Cause::OverlayEvicted;
            what = nullptr;
        }
        if (cause == Cause::WriterRange) {
            const s64 offset = static_cast<s64>(writers.range_begin) - static_cast<s64>(*device);
            const u64 range_size = writers.range_end - writers.range_begin;
            const u32 kind = static_cast<u32>(writers.range_kind);
            Prediction::NoteRangeWait(kind, offset, range_size);
            static u32 logged{};
            if (logged < 16) {
                ++logged;
                constexpr std::array<const char*, 5> kinds{"upload", "release", "compute",
                                                           "copy", "other"};
                LOG_INFO(HW_GPU,
                         "QMD release prediction (EXP13=2): QMD {:#x} read waits on a writer "
                         "range: {} at byte {} of the QMD, {} bytes, {} ring words ahead of the "
                         "back; overlay {}, upload shadow {}",
                         address, kind < kinds.size() ? kinds[kind] : "?", offset, range_size,
                         target - done,
                         overlay == nullptr               ? "none"
                         : overlay->position > done       ? "queued"
                                                          : "already run",
                         shadow == nullptr        ? "none"
                         : block_position > done  ? "queued, older than the writer"
                                                  : "already run");
            }
        }
        ++mirror.window_waits;
        split.WaitUntil(target, VideoCommon::DrainSite::QmdRead);
        Prediction::NoteMirror(Prediction::Mirror::Waited);
        Prediction::NoteWait(cause, what);
    } else {
        Prediction::NoteMirror(Prediction::Mirror::Memory);
    }
    memory_manager.ReadBlockUnsafe(address, descriptor.data(), size);
    // An all-zero QMD: the words may still be in the buffer cache.
    if (std::ranges::all_of(descriptor, [](u32 word) { return word == 0; })) {
        unknown = true;
        return false;
    }
    return true;
}

void KeplerCompute::FrontFlushUploads(VideoCommon::GpuSplit& split, FrontState& fs) const {
    if (fs.upload_batch.empty()) {
        return;
    }
    FrontPatchOverlays(split, fs, fs.upload_batch.data(), static_cast<u32>(fs.upload_batch.size()),
                       fs.upload_batch_position);
    fs.upload_batch.clear();
}

void KeplerCompute::FrontPatchOverlays(VideoCommon::GpuSplit& split, FrontState& fs,
                                       const u32* words, u32 count, u64 position) const {
    // As Upload::State::ProcessData: the data words fill the upload's bytes in order, four each
    // (the last one partly); a linear one-line upload writes them at its destination as they are.
    namespace Prediction = VideoCommon::QmdPrediction;
    const u64 begin = fs.upload_offset;
    const u64 end = (std::min)(fs.upload_size, begin + u64{count} * sizeof(u32));
    fs.upload_offset = (std::max)(begin, end);
    auto& mirror = g_front_qmd;
    if (!fs.upload_known || begin >= end || !mirror.synced) {
        return;
    }
    constexpr u64 size = sizeof(NativeQmdScheduler::Descriptor);
    if (begin == 0 && fs.upload_size >= size) {
        // An upload that may hold QMDs: shadowed when its bytes land as they are.
        if (!fs.upload_contiguous) {
            Prediction::NoteMirror(Prediction::Mirror::ShadowNotLinear);
        } else if (fs.upload_size > FrontQmdMirror::ShadowBytes) {
            Prediction::NoteMirror(Prediction::Mirror::ShadowTooBig);
        } else if (fs.upload_address % FrontQmdMirror::BlockBytes != 0) {
            Prediction::NoteMirror(Prediction::Mirror::ShadowNotLinear); // Not QMD-aligned
        } else {
            bool evicted_used = false;
            fs.upload_shadow =
                mirror.NewShadow(&memory_manager, fs.upload_address, fs.upload_size, evicted_used)
                    .id;
            Prediction::NoteMirror(Prediction::Mirror::ShadowCreated);
            if (evicted_used) {
                Prediction::NoteMirror(Prediction::Mirror::ShadowEvicted);
            }
        }
    }
    if (!fs.upload_contiguous) {
        return;
    }
    const GPUVAddr first = fs.upload_address + begin;
    const GPUVAddr last = fs.upload_address + end;
    const u64 done = split.ConsumedNow();
    const u8* const data = reinterpret_cast<const u8*>(words);
    constexpr u64 block_bytes = FrontQmdMirror::BlockBytes;
    // Its own shadow takes its words. Every filled block counts as written at `position`: the
    // upload's range is stamped whole at each of its data words, and the back's upload writes
    // its whole buffer at the end of each batch of them.
    FrontQmdMirror::Shadow* const own = mirror.FindShadowById(fs.upload_shadow);
    if (own != nullptr) {
        std::memcpy(own->bytes.data() + begin, data, end - begin);
        own->filled = end;
        for (size_t block = 0; block * block_bytes < end; ++block) {
            FrontQmdMirror::StampBlock(*own, block, position);
        }
        own->last_use = ++mirror.use_clock;
    }
    const std::array<bool, 2> block_patched = PatchHolders(
        split, memory_manager, first, last, data, position, fs.upload_tag, own, done);
    const GPUVAddr first_block = first & ~GPUVAddr{size - 1};
    // A small patch of a QMD nothing above holds: when guest memory holds the QMD now (nothing
    // else queued writes it, not GPU-written, no inline QMD of it queued or evicted), an overlay
    // from memory with the patch over it.
    if (own != nullptr || fs.upload_size >= size) {
        return;
    }
    for (u64 block = 0; block < block_patched.size(); ++block) {
        const GPUVAddr block_address = first_block + block * size;
        if (block_patched[block] || last <= block_address) {
            continue;
        }
        const auto device = memory_manager.GpuToCpuAddress(block_address);
        if (!device || !memory_manager.IsFullyMappedRange(block_address, size) ||
            Nx2MayBeGpuWritten(*device, size)) {
            continue;
        }
        const FrontQmdMirror::Overlay* const existing =
            mirror.FindOverlay(&memory_manager, block_address);
        if ((existing != nullptr && existing->position > done) ||
            (existing == nullptr && mirror.evicted > done)) {
            continue; // Its inline QMD may still be queued
        }
        if (split.WriteStamp(*device, size, block_address, fs.upload_tag) > done) {
            continue; // Another write may still land there
        }
        NativeQmdScheduler::Descriptor qmd{};
        memory_manager.ReadBlockUnsafe(block_address, qmd.data(), size);
        const GPUVAddr from = (std::max)(first, block_address);
        const GPUVAddr to = (std::min)(last, block_address + size);
        std::memcpy(reinterpret_cast<u8*>(qmd.data()) + (from - block_address),
                    data + (from - first), to - from);
        mirror.AddOverlay(&memory_manager, block_address, position, qmd.data(), false);
        Prediction::NoteMirror(Prediction::Mirror::OverlayFromMemory);
    }
}

void KeplerCompute::FrontNoteUpload(FrontState& fs, u32 exec) {
    // What Upload::State writes for the data after this exec_upload, from the same registers
    // (the upload block is shared with the older class).
    constexpr u32 base = KEPLER_COMPUTE_REG_INDEX(upload);
    const auto reg = [&fs](u32 offset) { return fs.registers[base + offset]; };
    const u64 line_length = reg(0);
    const u64 lines = reg(1);
    fs.upload_address = (GPUVAddr{reg(2)} << 32) | reg(3);
    fs.upload_size = 0;
    fs.upload_known = true;
    fs.upload_offset = 0;
    fs.upload_shadow = 0;
    // Linear, one line or lines back to back: its bytes land at upload_address as they come.
    fs.upload_contiguous = (exec & 1) != 0 && (lines == 1 || reg(4) == line_length);
    if (line_length == 0 || lines == 0) {
        return; // Writes nothing (copy_size 0)
    }
    if ((exec & 1) != 0) {
        fs.upload_size = u64{reg(4)} * (lines - 1) + line_length;
        return;
    }
    // Block linear: SwizzleSubrect writes back the whole surface it reads.
    const u32 width = reg(6);
    const u32 x_offset = reg(10);
    const u32 shift = Common::FoldRight(
        4U, [](u32 x, u32 y) { return (std::min)(x, static_cast<u32>(std::countr_zero(y))); },
        width, static_cast<u32>(line_length), x_offset, static_cast<u32>(fs.upload_address));
    const u32 block = reg(5);
    fs.upload_size = Tegra::Texture::CalculateSize(true, 1U << shift, width >> shift, reg(7),
                                                   reg(8), (block >> 4) & 15, (block >> 8) & 15);
    fs.upload_known = fs.upload_size != 0;
}

void KeplerCompute::FrontNoteWrites(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    if (method == 0x107) {
        // An inline QMD is written to memory before it launches.
        const GPUVAddr qmd = ((GPUVAddr{fs.registers[0xc6]} << 32) | fs.registers[0xc7]) << 8;
        split.NoteWrite(memory_manager, qmd, 256);
    } else if (method == 0x6c3) {
        split.NoteWrite(memory_manager,
                        (GPUVAddr{fs.registers[0x6c0] & 0x1ffffff} << 32) | fs.registers[0x6c1],
                        16);
    }
}

bool KeplerCompute::FrontPlainWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                    u32 argument) {
    // EDEN_NX2_COMPUTE_SINK: a write that only stores its value (CallMethod's native_plain_methods
    // path, which writes native_registers and regs) is dropped when both hold it already, else
    // sunk on the back and applied before the engine's next other write. The segment does not
    // matter: only data_upload reads current_dma_segment, and it is never plain.
    if (!VideoCommon::GpuSplit::ComputeSink() || method >= native_plain_methods.size() ||
        !native_plain_methods[method]) {
        return false;
    }
    if (fs.plain_sent[method] && fs.registers[method] == argument) {
        split.CountBatched(VideoCommon::GpuSplit::Batched::ComputeSame);
        return true;
    }
    fs.plain_sent[method] = true;
    fs.registers[method] = argument; // All FrontApply does for a plain register
    if (VideoCommon::GpuSplit::RingPairs()) {
        split.PairWrite(this, method, argument); // ApplyPairs
    } else {
        split.SinkMethod(this, method, argument);
    }
    split.CountBatched(VideoCommon::GpuSplit::Batched::ComputeSunk);
    return true;
}

void KeplerCompute::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                         u32 method, const u32* base_start, u32 amount,
                                         u32 methods_pending, const GPUVAddr* segment) {
    auto& fs = Front(split);
    if (!fs.upload_batch.empty() && method != KEPLER_COMPUTE_REG_INDEX(data_upload)) {
        FrontFlushUploads(split, fs);
    }
    // EXP13=2: methods that signal the QMD scheduler go one by one, so its copy sees each (the
    // back runs a multi-method word by word too). 0x6c3 stays a multi-method: one by one it would
    // also note its releases (FrontNoteRelease), which acquires use.
    const bool signals = method == 0xaf || method == 0xb0 || method == 0x107;
    if ((method >= 0xe00 && method < fs.registers.size()) ||
        (signals && amount > 1 && VideoCommon::QmdPrediction::MirrorMode())) {
        for (u32 i = 0; i < amount; ++i) {
            FrontCallMethod(system, split, method, base_start[i], methods_pending - i <= 1, segment);
        }
        return;
    }
    if (method >= fs.registers.size() || amount == 0) {
        split.MultiMethod(this, method, base_start, amount, methods_pending, segment);
        if (amount != 0) {
            split.NoteUnboundedWriter(VideoCommon::GpuSplit::WriterKind::Compute);
        }
        return;
    }
    for (u32 i = 0; i < amount; ++i) {
        FollowNote("x", method, base_start[i]);
        FrontApply(fs, method, base_start[i]);
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, segment,
                      FrontEffect(method));
    FrontNoteWrites(split, fs, method);
    // An inline upload's words all write its one range.
    const u32 notes = method == KEPLER_COMPUTE_REG_INDEX(data_upload) ? 1 : amount;
    for (u32 i = 0; i < notes; ++i) {
        FrontNoteWriter(split, fs, method, base_start[i]);
    }
    if (method == KEPLER_COMPUTE_REG_INDEX(data_upload) &&
        VideoCommon::QmdPrediction::MirrorMode()) {
        fs.upload_batch.insert(fs.upload_batch.end(), base_start, base_start + amount);
        fs.upload_batch_position = split.SentPosition();
        FrontFlushUploads(split, fs);
    }
}

void KeplerCompute::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                                  FrontState& fs) {
    const auto positions = FrontMacroPositions(fs);
    const u32 start = positions[((*fs.executing_macro - 0xe00) / 2) % positions.size()];
    // EDEN_NX2_MACRO_TRACE=<method>: the first calls of that macro, parameters and writes.
    static const u32 traced = [] {
        const char* value = std::getenv("EDEN_NX2_MACRO_TRACE");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 16)) : 0U;
    }();
    static u32 traced_calls{};
    const bool trace = traced != 0 && *fs.executing_macro == traced && traced_calls++ < 6;
    std::string trace_text;
    if (trace) [[unlikely]] {
        trace_text = fmt::format("compute macro {:#x} start {:#x} params", traced, start);
        for (const u32 p : fs.parameters) {
            trace_text += fmt::format(" {:x}", p);
        }
        trace_text += " writes";
    }
    fs.executing_macro.reset();
    auto& macro = FrontMacroEngine(fs);
    if (native_data_ram) {
        macro.native_interpreter.active_dram = native_data_ram;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
    const auto read = [&fs](u32 address) {
        return address < fs.registers.size() ? fs.registers[address] : 0U;
    };
    const auto emit = [&](u32 address, u32 value) {
        if (trace) [[unlikely]] {
            trace_text += fmt::format(" {:x}={:x}", address, value);
        }
        FollowNote("m", address, value);
        if (address >= 0xe00) {
            return false;
        }
        if (!fs.upload_batch.empty() && address != KEPLER_COMPUTE_REG_INDEX(data_upload)) {
            FrontFlushUploads(split, fs);
        }
        if (FrontPlainWrite(split, fs, address, value)) {
            return true;
        }
        FrontApply(fs, address, value);
        split.MacroWrite(this, address, value, FrontEffect(address));
        FrontNoteWrites(split, fs, address);
        FrontNoteWriter(split, fs, address, value);
        if (address == KEPLER_COMPUTE_REG_INDEX(data_upload) &&
            VideoCommon::QmdPrediction::MirrorMode()) {
            fs.upload_batch.push_back(value);
            fs.upload_batch_position = split.SentPosition();
        }
        if (address == 0x6c3) {
            FrontNoteRelease(split, fs, value);
        }
        return true;
    };
#ifdef ARCHITECTURE_x86_64
    if (!macro.native_jit) {
        macro.native_jit = std::make_shared<TuringMacroJit>();
    }
    macro.native_jit->Prepare(macro.native_code, macro.native_code_generation);
    const auto result = macro.native_jit->Execute(
        start, fs.parameters, std::span<u32>{*macro.native_interpreter.active_dram}, read, emit);
#else
    const auto result =
        macro.native_interpreter.Execute(macro.native_code, start, fs.parameters, read, emit);
#endif
    FrontFlushUploads(split, fs); // The macro's upload words, at once
    if (!result.success) {
        macro.native_execution_failed = true;
        LOG_ERROR(HW_GPU, "Native MME (split front) failed: compute start={:#x}, consumed={}, error={}",
                  start, result.consumed, result.error);
    }
    if (trace) [[unlikely]] {
        LOG_INFO(HW_GPU, "{}", trace_text);
        g_follow_left = 260;
        g_follow_text.clear();
    }
    fs.parameters.clear();
    if (VideoCommon::GpuSplit::ComputeSink() && ComputeFrontVerifyEnabled() &&
        (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

void KeplerCompute::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // Writes the back still holds in its method sink count as done (SinkMethod's are tagged).
    std::array<u32, 0x1000> expected = native_registers;
    std::array<u32, Regs::NUM_REGS> expected_regs;
    std::memcpy(expected_regs.data(), regs.reg_array.data(), sizeof(expected_regs));
    for (auto [method, value] : method_sink) {
        if ((method & SplitPlainTag) != 0) {
            method &= ~SplitPlainTag;
            expected[method] = value;
        }
        if (method < Regs::NUM_REGS) {
            expected_regs[method] = value;
        }
    }
    for (u32 i = 0; i < 0xe00; ++i) {
        // Dropped same-value writes rely on regs holding what native_registers holds.
        const bool regs_differ =
            fs.plain_sent[i] && i < Regs::NUM_REGS && expected_regs[i] != expected[i];
        if (fs.registers[i] == expected[i] && !regs_differ) {
            continue;
        }
        if (differences++ < 64) {
            LOG_WARNING(HW_GPU, "Split front compute register {:#x}: front {:#x}, back {:#x}{}", i,
                        fs.registers[i], expected[i],
                        regs_differ ? fmt::format(" (back regs {:#x})", expected_regs[i])
                                    : std::string{});
        }
        fs.registers[i] = expected[i];
        fs.plain_sent[i] = false;
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front compute register checks: {} done, {} differences", checks,
                 differences);
    }
}

} // namespace Tegra::Engines
