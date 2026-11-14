// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/settings.h"

/// Requests from the debug server (EDEN_NX2_DEBUG_SERVER, docs/nx2/debug-mcp.md) to the thread
/// that owns the rasterizer. The server posts them from its own thread; RasterizerVulkan takes
/// them at the end of each presented frame, so every cache they touch is used from its owner.
/// Without the server nothing is posted and the per-frame cost is one relaxed atomic load.
namespace VideoCore::Nx2Debug {

/// EXPERIMENT (not for nx2): EDEN_NX2_DMA_WAIT_WRITERS, read at startup and settable live by the
/// debug server's `switch` method, so one session can A/B it.
[[nodiscard]] inline int EnvSwitch(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? 0 : std::atoi(value); // whole number: EDEN_NX2_DMA_SNAPSHOT=10
}
/// The switch's value, read from the environment on first use (-1: not read yet). Static
/// initialization ran before the frontend applied nx2_switches.txt, so those settings were lost.
[[nodiscard]] inline int SwitchValue(std::atomic<int>& value, const char* name,
                                    int fallback = 0) {
    int current = value.load(std::memory_order_relaxed);
    if (current < 0) [[unlikely]] {
        int unset = -1;
        const int initial = std::getenv(name) != nullptr ? EnvSwitch(name) : fallback;
        value.compare_exchange_strong(unset, initial, std::memory_order_relaxed);
        current = value.load(std::memory_order_relaxed);
    }
    return current;
}
inline std::atomic<int> g_dma_wait_writers{-1};
/// EXPERIMENT, off by default: Ocarina of Time flickers white on the bottom half of Link's house
/// ladder with it (confirmed toggling it live, 2026-11-06), for +1.4 fps. EDEN_NX2_SEM_FAST=1,
/// semaphore acquires without draining the back
/// when the value is already met and no queued write overlaps it (puller.cpp).
inline std::atomic<int> g_sem_fast{-1};
/// TEST: EDEN_NX2_LIST_VERIFY=1 (live), each command list the split's front reads on GPU-written
/// pages is read again as before EXP40 and compared (video_core/nx2_list_verify.h; "list
/// verify" every 10 s). Changes timing: the front waits for every writer again for each list.
inline std::atomic<int> g_list_verify{-1};
/// EXPERIMENT (not for nx2): EDEN_NX2_LOOP_SKIP=1, the split's front does not send the instance
/// loop writes the back would ignore (draw.end in an instance loop, unchanged vertex and index
/// buffer firsts and counts) (maxwell_3d.cpp).
inline std::atomic<int> g_loop_skip{-1};

/// EDEN_NX2_DMA_SNAPSHOT (from EvilOOT's dma-snapshot fix): macro DMA reads of pages a drained
/// read flushed whole skip the drain while nothing sent since may write them (Maxwell3D::
/// FrontDmaRead). 9 = on (default: Lake Hylia 46.5 -> 52.3 fps; verify 0 differences in 235k
/// reads), 10 = on and verify every skipped read against the drained path, 0 = off.
inline std::atomic<int> g_dma_snapshot{-1};

/// EDEN_NX2_ACCURATE_PARTS (bit mask, 0 = none, read once at boot): parts of GPU accuracy High in
/// Fast mode, the split GPU thread kept, to find which one stops a flicker that Accurate mode
/// stops. Not live: switching the counter query mode mid-game crashed (queries opened in one
/// mode were closed in the other).
/// 1 = counter reports written once their host query completes (synced), 2 = precise occlusion
/// queries, 4 = CPU reads of GPU-written images download them, 8 = CPU writes flush GPU data
/// before invalidating it, 16 = fences wait for the GPU work before them.
/// Whether GPU accuracy High is on, or the EDEN_NX2_ACCURATE_PARTS bit `part` asks for it.
[[nodiscard]] inline bool AccuratePart(int part) {
    static const int parts = EnvSwitch("EDEN_NX2_ACCURATE_PARTS");
    return Settings::IsGPULevelHigh() || (parts & part) != 0;
}

/// Whether fences run their callbacks (fence-timed releases among them) once the host GPU passed
/// the work before them, not when made (FenceManager::DelayFences).
[[nodiscard]] inline bool DelayedFences() {
    return Settings::IsGPUFenceBehaviorDefault()
               ? AccuratePart(16)
               : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
}

/// EDEN_NX2_HOST_PACE: the host frame pacer (vk_host_pacer.h). 1 = on (default), 0 = only
/// measure how long each frame was shown.
inline std::atomic<int> g_host_pace{-1};

/// EDEN_NX2_ARB_SPIN (zortos293/Lichii#641; off by default here, #641 uses 8): microseconds a
/// guest thread about to sleep in WaitForAddress first watches the word (KAddressArbiter), 0 =
/// off. Live: read on every wait, so profile_toggle.py can A/B it in one session.
inline std::atomic<int> g_arb_spin{-1};
/// EDEN_NX2_IDLE_PAUSE=1 (zortos293/Lichii#641; off by default here, on in #641): the guest
/// core's idle poll (PhysicalCore::Idle, EDEN_NX2_IDLE_SPIN_US) runs PAUSE and reads the clock
/// every 16 pauses, instead of a SwitchToThread and a clock read per iteration. Live: read on
/// every Idle.
inline std::atomic<int> g_idle_pause{-1};

/// Numbered experiment switches EDEN_NX2_EXP0 .. EDEN_NX2_EXP63, live through the debug server:
/// slots for batches of fixes built together and A/B-tested in one session
/// (E:\OotRemakeEmulation\work\EXPERIMENTS.md lists which fix uses which slot). 0 = off.
/// EXP8 (texture upload hitches), bits: 1 = keep idle upload staging buffers up to 32 MiB
/// (StagingBufferPool::ReleaseLevel), 2 = no guest upload for a new image that GPU copies
/// overwrite whole (TextureCache::JoinImages), 4 = large 2D levels untiled on the transcode
/// workers too (UnswizzleImage), 8 = guest memory of uploaded textures not hinted cold
/// (TextureCache::TickFrame).
/// EXP10 = 1: macro DMA read snapshots (EDEN_NX2_DMA_SNAPSHOT) skip the drain for compute and
/// copies sent since whose write ranges miss the pages read (Maxwell3D::FrontDmaRead): copies,
/// inline uploads and releases as the front notes them, native dispatches as the back records
/// them once it ran them (the read waits for that much); 2 = the same, but a dispatch still
/// queued drains as before. +4 (5, 6): compute report semaphores noted as before, 16 bytes or
/// anything when unmapped (GpuSplit::NoteReleaseWriter).
/// EXP12 (EDEN_NX2_SEM_FAST acquires read current but unmet, puller.cpp ReleaseFencesForReadable):
/// 1 = their fence release waits only for the queued writers the read can't account for, full
/// drain only if still unmet; 2 = suspend then too, without the full drain; 3 = measure only.
/// EXP17 = 1: a small streamed uniform buffer binds again any stage or slot's copy of the same
/// guest range made in the current command buffer, when the bytes still match
/// (BindSmallUniformReusing; zortos293/Lichii#641 EDEN_NX2_UBO_RANGE_REUSE).
/// EXP18 (buffer binding memos, #641), bits: 1 = a constant buffer bind of the slot's current
/// GPU range keeps its translation and resolved buffer (UBO_REBIND), 2 = storage bindings reuse
/// their last FindBuffer result (STORAGE_MEMO), 4 = buffers touch the LRU once per frame
/// (default 4: exact, see LoadExperiments). All while the mappings and the set of buffers are
/// unchanged (BufferCache::binding_epoch).
/// EXP19 (graphics pipeline lookup without Refresh, PipelineCache::CurrentGraphicsPipelineClean):
/// 1 = as EDEN_NX2_PIPELINE_CLEAN=1, 2 = also #641's patches (blend registers left to the
/// dual-source bit and Dirty::Blending, attribute types redone alone under Dirty::VertexInput).
/// EXP50 (live; with EXP19 / EDEN_NX2_PIPELINE_CLEAN on; after the-cashman's fix in
/// zortos293/Lichii#717): 1 = the blend registers leave the clean lookup's word compare and
/// Dirty::Blending is not looked at; when they moved since the key was made, the attachments
/// RefreshBlending would build from them (FixedPipelineState::BlendingMatches) and the
/// dual-source bit are compared with the key's: equal keeps the pipeline without a Refresh,
/// different patches only the blend part of the key. Reads the registers, so #679's writes
/// without the flag are still caught. Logs "saved by the blend value check" and why the other
/// lookups refreshed in "EDEN_NX2_PIPELINE_CLEAN per second". Off by default, not measured yet.
/// EXP22 (split ring traffic, zortos293/Lichii#641; gpu_split.h DeltaWrite and DrawBatch), bits:
/// 1 = 3D state writes that went as Direct macro writes or (EDEN_NX2_RING_PAIRS) as pairs go in
/// StateDelta blocks, applied in one loop (FRONT_STATE, without its per-register merge),
/// 2 = repeated instance-loop groups go as one DrawBatch entry that counts the repeats
/// (DRAW_BATCH), 4 = same-value index_buffer.first writes stay on the front
/// (SKIP_SAME_DRAWREG, as EXP2), 8 = the back checks each delta and batch against the single
/// writes and keeps theirs (logs "EXP22 check"). 7 = all three. #641's FRONT_CB_ENCODE is not
/// ported (it holds constant buffer writes off the ring).
/// EXP23 = 1: trim unused pipeline libraries every 600 frames past 4096 (#579 stage B;
/// PipelineCache::TickFrame, GraphicsPipelineLibraryCache::TrimUnused). Off by default, not
/// measured yet.
/// EXP24 = 1: Ocarina of Time's present poll (`while (nvnEventGetValue(e) != 1) SleepThread(1 ms)`,
/// from Hookshot) sleeps until the GPU's next write of a 1 instead, at most 1 ms
/// (core/hle/kernel/svc/svc_nx2_oot_present.cpp; the site per build id is in its PresentSites
/// table, or EDEN_NX2_OOT_PRESENT_SITE=<ret>[:<blr>] in hex main offsets). Logs "OoT present
/// poll (EXP24)" counts every 10 s. EXP24 = 2: locate only, sleeps unchanged: short sleeps
/// (<= 2 ms) are counted by call chain and the top 8 logged every 10 s ("OoT present poll
/// locate"). EXP24 = 3: no GPU wake, the poll's 1 ms sleep is EDEN_NX2_OOT_PRESENT_SLEEP_US
/// instead (default 250 us, at most the time asked for). Off by default, not measured yet.
/// EXP25 (Ocarina of Time resource poll backoff, zortos293/Lichii#641; needs the built-in patch
/// placed at launch by EDEN_NX2_OOT_POLL_BACKOFF=1, core/hle/kernel/svc/svc_nx2_oot_poll.cpp):
/// the resource request thread sleeps before rescanning requests that are not ready.
/// 0 = no sleep (as without the patch, plus one SVC per rescan), 1 = 500 us (#641's default),
/// above 1 = that many us, at most 16000. Off by default, not measured yet.
/// EXP26 (native uniform bank check trims, zortos293/Lichii#717; BufferCache::
/// BindHostGraphicsUniformBuffer and CheckNativeBankPagesTracked), bits: 1 = a bind that streams
/// the buffer (at most 4 KiB, not GPU-modified) skips the check, the stream copies guest memory
/// itself (BUFCACHE_BANK), 2 = pages compared with an inline SSE2 loop instead of memcmp
/// (PAGE_CMP), 4 = changed pages copied to the host buffer from their snapshots instead of being
/// marked CPU-modified (two VirtualProtect calls each), for pages whole in the bound buffer with
/// no GPU-written data (UploadNativeBank's upload, without its once-a-frame bank shadows: the
/// check stays per bind, d79572d28a). Off by default; measure 1 and 2 first, then 4. Verify
/// with EXP9=2 ("missed" stays 0) and the flash watch.
/// EXP35 (guest writes of never-GPU-written cached pages, Core::Memory::HandleRasterizerWrite):
/// 1 = collected for the split's back (applied at its next FlushCaching) instead of OnCPUWrite
/// under both caches' mutexes. On by default (2026-11-08): Castle Town guest-core lock spinning
/// 11.6% -> 7.6% of busy (writes 6.1% -> 0.7%), standing 43.0 -> 45.3 fps, route 45.0 fps with
/// 175 flash reports (camera-motion noise ~270 either way).
/// EXP36 (live): 1 = guest reads of cached memory always take GetFlushArea's locked path (turns
/// off its lock-free answer for image-free, never-GPU-written pages). For the washed-out hunt.
/// EXP37 (live): 1 = nx2:gsv answers the driver's armed syncpoint waits right after each guest
/// syncpoint increment (SyncpointManager guest increment hook), not only at its 500 us poll.
/// EXP35=2 (live): every guest write of cached memory is collected for the back, not only writes to
/// never-GPU-written pages. EXP60 (live): 1 = a guest read whose cache mutexes stay taken after a
/// short spin is answered with a flush request served by the back (GetFlushArea).
/// EXP49 (live; async read-back, the core of zortos293/Lichii#1028 0b83798bf6): 1 = a guest read
/// of GPU-written buffer memory (GPU::OnCPURead) has the GPU thread record and submit the
/// downloads without waiting for the host GPU (RasterizerVulkan::FlushRegionForRead). The reading
/// thread waits for them and writes them to guest memory (BufferCache::FinishReadDownloads); a
/// read whose bytes are already on their way only waits. Until they land the bytes stay
/// GPU-modified; guest CPU writes keep their bytes (ClearReplacedDownload, pending writes at
/// landing); writes from the GPU's side (MemoryManager, query results, DMA clears and copies) land
/// them first. 0 = the synchronous flush (Scheduler::Finish on the GPU thread under the buffer
/// cache's lock). Both log "Read-back (EXP49) per second" every 10 s (reads, GPU thread time in
/// read flushes, reader waits). Not ported: #1028's early downloads and EDEN_NX2_READ_CHECK.
/// EXP27 (keep the split's back fed, fewer submits; zortos293/Lichii#717), bits: 1 = the front
/// publishes before it yields or sleeps polling suspended channels (PUBLISH_WAITS,
/// GpuSplit::PublishBeforeWait), 2 = retired (was PUBLISH_IDLE, each entry published at once
/// while the back found the ring empty; EXP39 & 1 replaces it), 4 = the back prefetches the ring 128 words
/// ahead (RING_PREFETCH), 8 = command list copies after a dispatch are submitted when the front
/// waits for them, when the back goes idle or with the next submission, instead of at once
/// (DISPATCH_BATCH, RasterizerVulkan::SubmitNx2ListCopies). Off by default; 1, 2 and 4 change
/// only when work becomes visible, 8 changes submission timing (the #161 family: flash watch).
/// EXP28 (fewer fences, zortos293/Lichii#717; FenceManager::SignalFence), bits: 1 = a fence whose
/// only work is syncing payload releases into host buffers is skipped like an empty one (the
/// guest word was written at once; the copy goes with the next fence made or wait for idle,
/// values kept per release since the query slot may be reused: GuestStreamer::
/// SyncDeferredWrites) (FENCE_TRIM), 2 = a made fence with nothing to wait for on the host GPU
/// joins the newest queued fence, or with none queued and the fencing thread idle runs its work
/// on the split's back at once (FENCE_FOLD, FoldStubbedFence). Off by default; timing family
/// (#161, boulder flashes): flash watch after each. "Fence decisions/s" (GPU profile) counts
/// both.
/// EXP29 (zortos293/Lichii#717's micro bundle), bits: 1 = Device::GetDeviceMemoryUsage reuses an
/// answer younger than 50 ms (MEMSTAT_MS; the buffer and texture caches ask twice per frame on
/// the back; zone "memory query" times the driver query), 2 = GPU::TickWork checks an atomic
/// before taking the sync request mutex (OPT_TICKWORK), 4 = native compute engines no longer
/// record inline upload execs for the old launch method they never run (DROP_NATIVE_UPLOADS).
/// Default 6: 2 and 4 are exact (2 only allows interleavings the mutex already allowed, 4 drops
/// a list nothing in native mode reads); 1 is off, since the caches' GC then sees usage up to
/// 50 ms old. EDEN_NX2_EXP29=0 is the behaviour before.
/// EXP30 = 1 (zortos293/Lichii#559, no env var of its own): a macro DMA read of report memory
/// with pending query results (#215's white flash fix, RasterizerVulkan::FlushQueryResults) waits
/// only until the counter reports in its range are written (Tegra::Nx2WaitAsyncReportsSettled,
/// slots from 12 bytes before the read), not for an operation queued behind every fence, which
/// also waited for all GPU work sent after those reports (8-15% of the back while walking in
/// Ocarina, ~11 waits/s of ~3.7 ms, and the front's FrontDmaRead drain with it). Vulkan only
/// (OpenGL keeps its FlushRegion). 0 = the operation wait as before. Off by default, not measured
/// yet: compare the back's "query DMA wait" profile zone (ms and waits per second) and the white
/// flash watch.
/// EXP31 (render target memo, RT_FAST from zortos293/Lichii#717; TextureCache::
/// UpdateRenderTargets), bits: 1 = a draw whose render targets are clean (no Dirty::RenderTargets)
/// skips preparing them again while the same views were prepared for a draw in the same frame,
/// no image was marked modified, created, deleted, rescaled or CPU written since (modification
/// tick and bindings_epoch unchanged), and every target image is GPU-modified with no CPU
/// modification, alias or sparse state, 2 = with 1: such draws run the loop anyway and check it
/// changed nothing but the targets' own modification ticks (flags, bindings_epoch; logs "RT_FAST
/// (EXP31) mismatch" and drops the memo on a difference). Logs skips, verified and mismatches
/// every 600 frames while on. Off by default; measure the back's "render targets" zone (3 = verify
/// first, then 1).
/// EXP32 (texture binding memo, the core of zortos293/Lichii#717's BINDMEMO for "the same textures
/// as the pipeline's previous draw"; Vulkan GraphicsPipeline::TextureMemo, graphics pipelines whose
/// image bindings are all sampled textures), bits: 1 = a draw that reads the same TIC and TSC
/// handles as its pipeline's previous draw, with the same pools, channel, GPU mappings and texture
/// cache epoch (no view removed: image deleted, joined or rescaled), the same guest TIC/TSC
/// entries and images without CPU writes to upload or aliases, takes that draw's view and sampler
/// ids and descriptor entries instead of GetSamplerId, FillImageViews and PushImageDescriptors
/// (handles are still read every draw: the EXP21 storage pointer rule and the per-bind uniform bank
/// check of d79572d28a are untouched), 2 = verify: every hit for 120 frames after the switch
/// changes, then one hit in 64, runs the full path instead and compares view ids, sampler ids,
/// descriptor entries and rescaling bits ("Texture memo mismatch (EXP32)" log, first 32), 4 = a
/// mismatch turns the memo off until the switch changes. "Texture memo (EXP32=n)" logs hits, misses
/// by cause, records and verify mismatches every 10 s while set. Off by default; 7 = memo, verify
/// and auto-off. Measure the back's "read handles", "fill image views" and "push descriptors"
/// zones (the sampler lookups move from the first to the second with the memo on).
/// EXP33 = 1 (zortos293/Lichii#337 by Inc0rporate; QueryCacheBase::CounterReport and
/// FenceManager::SignalFence): at fence behaviour Default without GPU accuracy High (or
/// EDEN_NX2_ACCURATE_PARTS bit 1), a fence-flagged query report is no longer treated as
/// synchronized: the query stays registered until its fence runs, and that fence is made even
/// when empty and runs the report's operation once the host GPU passed the work before it
/// (VideoCore::g_signaling_query_report_fence), as with delayed fences. Other fences are skipped
/// as before. Read once per report, live. #337's fix for Ocarina's white flashes on leaves and
/// walls at low accuracy; ours already has #215's FlushQueryResults wait (EXP30 trims it), so
/// off by default, not measured yet: white flash watch, and the front's semaphore waits (the
/// reports land later than with the fence skipped).
/// EXP34 (EDEN_NX2_ACQUIRE_PRECHECK=2's reset hole, fable_answer_6 6i (b); puller.cpp
/// PrecheckAudit): where level 2 would answer a GEQ acquire met from guest memory alone,
/// 1 = it takes the normal path instead when the front knows a write to the word is still queued
/// for the back: a plain release (puller, SEM_EXECUTE, 3D or compute 0x6c3) with any payload, a
/// noted write over the word (puller semaphores, 3D reports, compute 0x6c3 releases and
/// reductions, copies, uploads) or, with EXP13 on, a QMD prediction record, and when
/// SemaphoreReadableNow read the word current but below the value (the Castle Town route's only
/// cases, 5-13 per 5 s; most likely work it doesn't track ran between that read and level 2's,
/// the log's first 8 say); 2 = only when that
/// queued plain release writes less than the word read (a reset); 0 = count only. The counts
/// are logged either way ("Acquire precheck (level 2)", every 5 s). Default 1: it only turns met
/// answers into the path level 2 takes for a word below the value, never the other way, so it is
/// a strict tightening; its cost is the drains it puts back (compare 0, 1 and 2 with
/// profile_toggle and the AcquiresFast count).
/// EXP39 (live; keep the split's back fed, zortos293/Lichii#805 62db242b34), bits: 1 = the back
/// stores its position when it runs out of work (GpuSplit::back_idle_at) and, while that is the
/// position last published, the front publishes at its next Commit once it holds 8 words, not
/// after 256 (outside open Pairs and StateDelta blocks; replaces EXP27 & 2), 2 = when an acquire
/// resumes, the snapshot refresh (DmaPusher::RefreshSnapshot) skips the current list's entries
/// that ran already and the entry the acquire suspended inside, which reads memory anyway
/// (CanUseSubmittedCommands); the lists behind it are copied as before. Default 3: both only
/// change when the back sees work or which words nobody parses again are copied. Measure with
/// profile_toggle (EXP39 0 vs 3): back "split wait", front "resume pending".
/// EXP40 (live; command lists on GPU-written pages, zortos293/Lichii#805 34656e8225), bits:
/// 1 = each command list page keeps the 4-byte words GPU work wrote that guest memory lacks
/// (BufferCache::Nx2ListPage::gpu_words: set by every GPU write noted to the page, the whole range
/// a dispatch or copy may write, cleared for words the host writes whole and for all of them when
/// the front flushes the page); once the page was flushed, a list on none of those words reads
/// guest memory and waits for no copy, and a list on some reads those from the copy and the rest
/// from guest memory (the "list GPU wait" zone falls; a storage write mark that
/// EDEN_NX2_WRITTEN_SSBO_MARK_KIB capped counts as written to the end of its last page, since
/// the shader may write past the cap); 2 = a list on GPU-written pages waits
/// only for the compute and copy work sent before the last host sync ("list copy writers"
/// site), not for every one sent so far ("list writers"). Off by default: both read words
/// earlier than before. Verify with EDEN_NX2_LIST_VERIFY=1 ("list verify ... 0 differ"). Always
/// on (bug fixes): copy engine copies and clears tell the list pages what they write to guest
/// memory (their copies went stale), DMAClear checks its whole range (it passed words as bytes),
/// and a front waiting for a list copy publishes first.
/// EXP41 (live; command entries on pages no GPU work wrote yet, DmaPusher::Step): 1 = the writer
/// waits before the port of zortos293/Lichii#805 6229487c8c + d4e8d85d4d: EDEN_NX2_LIST_WAIT_SKIP=1
/// (built in) skips every entry's wait, =0 waits for every queued dispatch and copy. 0 (default)
/// = the new shape: an entry waits only for the compute and copy work sent before the last host
/// sync (WaitForSyncedWriters), and with LIST_WAIT_SKIP=1 an entry that begins between methods
/// (a command buffer the CPU writes) doesn't wait unless copy engine work is still queued;
/// entries inside a method's data (the parameters GPU work writes) still wait. Against the
/// built-in skip-everything this only adds waits. Entries that didn't wait for every writer are
/// checked on the back ("EDEN_NX2_LIST_WAIT_SKIP: command list at ... was GPU-written": stays 0).
/// EXP42 (live; VkImage pool, fable_answer_12 2B; vk_texture_cache.cpp): 1 = a deleted
/// single-level, single-layer render target's VkImage (and its rescaled copy), once the GPU is done
/// with it, is kept with its memory for the next image made with the same create info and view
/// formats, instead of vkDestroyImage + vkCreateImage. Only device-local, non-dedicated images of
/// 256 KiB or more; the pool holds at most 512 MiB, 180 frames each, and empties when the switch
/// goes off or an image does not fit in device memory. A taken image starts uninitialized
/// (UNDEFINED layout, contents discarded) as a new one does. Its bytes are left out of the
/// collector's device memory figure, as deleted images waiting for the GPU are. Logs "Image pool
/// (EXP42) last 600 frames" and counts pooled creates in "Slow clear texture cache". 1 is the default
/// (LoadExperiments); 0 = off.
/// EXP45 (live; 1 = off): the native compute write extent self-check (BufferCache::ExtentTrust,
/// fable_answer_25): a program is marked in full until a dispatch at its dims showed its proof
/// holds, and for good once one wrote past it (Ocarina of Time's glowing triangles, 2026-11-09).
/// EXP43 (live; render target relocation, fable_answer_12 2A; TextureCache::JoinImages): a render
/// target a clear overwrites whole, with no copies, aliases or remapped images among its overlaps
/// and exactly one predecessor among its bad overlaps (an image of the same single-level,
/// single-layer 2D shape it drifted off by less than its size, not written in this frame: Ocarina
/// of Time's bump-allocated per-frame targets while an NPC moves), takes that image's place;
/// bad overlaps of other shapes stay as for a new image. 2 = relocate: the image moves to the new
/// address (stale at the old one, untracked, unregistered, its overlap links dropped, registered
/// again; its VkImage, memory, views and framebuffers stay). 3 = as 2, and with several
/// predecessors the most recently modified one moves (lets a chain made before the switch went
/// on stop growing and age out). 1 = retire early: the predecessor is deleted at once and a new
/// image made, as with 0. 0 = off: candidates are only counted. 4 = verify (fable_answer_16 1c): as
/// 3 under the render-frame clock (as EDEN_NX2_RENDER_CLOCK=1), and each old range is kept as a
/// tombstone for 2 render frames; reads that land on one are counted ("Relocation tombstones
/// (EXP43=4) last 600 frames: N hits", first 8 logged), which must stay 0. Default 3 (LoadExperiments). Never for a shape
/// whose GPU-written contents were sampled or rendered to without a clear in a later frame, or
/// read by the CPU (sticky per shape for the session: "Render target relocation guard (EXP43)"),
/// nor for rescaled, sparse, converted or CPU-read images, ones with a download queued, bound as
/// another target of the clear or in another channel. Logs each shape's first candidate and
/// relocation and "Render target relocation (EXP43=n) last 600 frames".
/// EXP51 (live; RAM, fable_answer_26 C2c; Vulkan PipelineCache::EvictUnusedComputePipelines):
/// compute pipelines unused for 9000 frames are destroyed, least recently used first, while more
/// than 500 are held (checked every 600 frames). Their next dispatch translates them again on the
/// GPU thread, binding the pointers they bound, and builds them from the driver's pipeline cache
/// (no new disk cache record). 1 = only programs dispatched this session, 2 = also boot builds no
/// dispatch used yet. Never one still building or with a rebuild pending; destroyed through
/// retired_compute_pipelines once the GPU and their worker build are done. Off by default, not
/// measured yet: compare the 10 s memory line's compute count and the working set over a long
/// session, and the "Evicted N compute pipelines" log's "dispatched again" count (each is a
/// GPU-thread rebuild, a possible hitch).
/// EXP52 (live; with EXP19 / EDEN_NX2_PIPELINE_CLEAN on; fable52_answer_1 section 1; Vulkan
/// PipelineCache::CurrentGraphicsPipelineClean): 1 = when only the bound programs changed since
/// the last key (every other Refresh input stands), the key's state is kept and only what the
/// programs feed is redone (with dynamic vertex input, the attribute types masked by what the
/// programs read), then Next / the cache map looks the new key up, without
/// FixedPipelineState::Refresh. 2 = as 1, and each such key is also built by the full Refresh and
/// compared byte for byte; Refresh's is kept, the first 16 differences are logged ("Programs-only
/// key (EXP52=2) differs"). The "EDEN_NX2_PIPELINE_CLEAN per second" line counts "programs only"
/// and, at 2, "verified N, M differ". With EXP52 on its "programs" reason stays 0 (a change of
/// programs together with other state counts under that state). Off by default, not measured yet:
/// prove 2 shows 0 differ first (under EXP19=2 without EXP50, blend writes without
/// Dirty::Blending already leave the clean lookup's attachments stale, and verify reports that
/// too), then compare the lookup zone with profile_toggle (EXP52 0 vs 1).
inline constexpr int ExperimentCount = 64;
inline std::atomic<int> g_experiments[ExperimentCount]{};

inline void LoadExperiments() {
    static const bool loaded = [] {
        // Kept experiments: on unless their variable says otherwise. EXP8 = 15: Ocarina of Time
        // walking through Hyrule Field, fresh launches: 1% low 30.3 -> 35.1 fps, frames over
        // 25 ms 68 -> 38 of ~1650.
        g_experiments[8].store(15, std::memory_order_relaxed);
        // EXP9 = 1: native uniform bank pages compared only when they may have changed (verified
        // with EXP9=2: 0 missed in 120 s walking). Idle 54.5 -> 57.7 fps, walking 56.6 -> 58.9.
        // EXP9 = 3 (2026-11-08): the same without the once-a-frame compare of every page, which
        // never found a change ("frame check N / 0", "missed without the frame check 0"). Castle
        // Town, one session, 4 x 35 s each: uniform buffers 135.6 -> 97.9 ms/s on the split's back,
        // 40.8 -> 42.3 fps; flash watch on the Castle Town route 267 reports vs 273 with EXP9=1.
        g_experiments[9].store(3, std::memory_order_relaxed);
        // EXP10 = 1: macro DMA reads wait only for the dispatches, uploads, copies and releases
        // that may have written them (EDEN_NX2_DMA_SNAPSHOT=10 verify: 0 differed in 2 x 90 s
        // walking). GPU front idle 81 -> 105 ms/s, waits for the back 368 -> 338 ms/s.
        g_experiments[10].store(1, std::memory_order_relaxed);
        // EXP14: unchanged macro register writes dropped before FrontWrite's other checks.
        // Castle Town, profile toggle: front macro 250 -> 236 ms/s, push 77 -> 70 ms/s.
        g_experiments[14].store(1, std::memory_order_relaxed);
        // EXP18 = 4 (exact, zortos293/Lichii#717 wave 1): a buffer touches its LRU node once
        // per frame; a second touch in the same frame was already a no-op in LruCache::Touch
        // (its tick is not older), so only the node's cache miss goes. Bits 1 and 2 stay off.
        g_experiments[18].store(4, std::memory_order_relaxed);
        // EXP21 = 1: a texture whose handle word is a storage buffer's address gets no texture
        // (VideoCommon::HandleIsStoragePointer). Ocarina of Time's Zora's River waterfall
        // flickered every other frame in bursts without it.
        g_experiments[21].store(1, std::memory_order_relaxed);
        // EXP29 = 6: the exact parts of #717's micro bundle (TickWork's lock-free check, native
        // compute inline uploads not recorded); 1 (memory usage reuse) changes what the caches'
        // GC sees and stays off.
        g_experiments[29].store(6, std::memory_order_relaxed);
        // EXP34 = 1: ACQUIRE_PRECHECK=2 no longer answers met over a write to the word still
        // queued for the back (a reset there let a channel run before its producer). A strict
        // tightening: only met answers become the normal path.
        g_experiments[34].store(1, std::memory_order_relaxed);
        // EXP35 = 1: guest writes of never-GPU-written cached pages collected for the back instead
        // of OnCPUWrite under both caches' mutexes (see the EXP35 notes above).
        // EXP35 = 2 (2026-11-08): every guest write collected (guest-core write lock spinning 0.7% ->
        // 0, standing 45.3 -> 46.5 fps, route flash reports 253 = camera noise).
        g_experiments[35].store(2, std::memory_order_relaxed);
        // EXP42 = 1 and EXP43 = 3 (2026-11-08, moving-NPC dips): pooled VkImages for deleted
        // render targets, and drifted per-frame targets relocated in place (the newest of several
        // predecessors). Castle Town live toggles: EXP43 0/3 46.3 -> 49.3 fps, EXP42 0/1 on top
        // 44.7 -> 47.9 (each won all 4 pairs); 21 loads 0 washed out; owner playtest: no visual
        // bugs, rarely under 40 fps.
        g_experiments[42].store(1, std::memory_order_relaxed);
        g_experiments[43].store(3, std::memory_order_relaxed);
        // EXP56 = 1 (2026-11-09): a cleared depth target larger than its colour targets is not
        // uploaded from guest memory first (TextureCache::JoinImages, overwritten_by_clear).
        // Castle Town live toggle 41.3 -> 43.8 fps, clear zone 54.7 -> 36.3 ms/s; screenshots
        // off/on identical.
        g_experiments[56].store(1, std::memory_order_relaxed);
        // EXP58 = 1 (2026-11-10): guarded render targets that a newer target of another shape
        // superseded are deleted once 60 frames old (TextureCache::SweepColdImages). Castle Town
        // 150 s A/B: device memory 7.8-10.5 -> 5.2-5.7 GB, collector evictions 83-1915 a window
        // -> 0, worst frames 140-180 -> 30-48 ms, moving median 42.2 -> 46.9 fps; skeleton fight
        // 5.1-5.5 GB instead of 10.9 with 0 evictions; two owner playtests without glitches.
        g_experiments[58].store(1, std::memory_order_relaxed);
        // EXP61 = 1 (2026-11-10): FlushCommands submits nothing when nothing was recorded since
        // the last submit. Castle Town live toggle: empty submits 4,600 -> 145 a second, 43.0 ->
        // 45.6 fps (3 of 4 pairs, the 4th a tie), GPU thread split drain 432 -> 396 ms/s.
        g_experiments[61].store(1, std::memory_order_relaxed);
        // EXP49 = 1 (2026-11-10, owner's call for v5.7): the async read-back. Stalchild fight,
        // flipped every 30 s while the owner fought: 51.9 -> 56.9 fps, the back's read flushes
        // 35 -> 6 ms/s, guest lock waits halved. No beams in that session or two later ones
        // played with it on from boot.
        g_experiments[49].store(1, std::memory_order_relaxed);
        // EXP39 = 3: publish to a back that waits after 8 words, no snapshot refresh of entries
        // that ran (zortos293/Lichii#805 62db242b34; see the EXP39 notes above). Timing only.
        g_experiments[39].store(3, std::memory_order_relaxed);
        for (int n = 0; n < ExperimentCount; ++n) {
            const std::string name = "EDEN_NX2_EXP" + std::to_string(n);
            if (const char* value = std::getenv(name.c_str())) {
                g_experiments[n].store(std::atoi(value), std::memory_order_relaxed);
            }
        }
        return true;
    }();
    static_cast<void>(loaded);
}

/// The value of experiment switch `n` (EDEN_NX2_EXP<n>), 0 unless set.
[[nodiscard]] inline int Experiment(int n) {
    LoadExperiments();
    return g_experiments[n].load(std::memory_order_relaxed);
}

/// The live switch called `name`, or null.
[[nodiscard]] inline std::atomic<int>* FindSwitch(std::string_view name) {
    if (constexpr std::string_view prefix = "EDEN_NX2_EXP";
        name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix) {
        int n = 0;
        for (const char c : name.substr(prefix.size())) {
            if (c < '0' || c > '9') {
                return nullptr;
            }
            n = n * 10 + (c - '0');
        }
        if (n >= ExperimentCount) {
            return nullptr;
        }
        LoadExperiments();
        return &g_experiments[n];
    }
    if (name == "EDEN_NX2_LIST_VERIFY") {
        return &g_list_verify;
    }
    if (name == "EDEN_NX2_DMA_WAIT_WRITERS") {
        return &g_dma_wait_writers;
    }
    if (name == "EDEN_NX2_SEM_FAST") {
        return &g_sem_fast;
    }
    if (name == "EDEN_NX2_DMA_SNAPSHOT") {
        return &g_dma_snapshot;
    }
    if (name == "EDEN_NX2_LOOP_SKIP") {
        return &g_loop_skip;
    }
    if (name == "EDEN_NX2_HOST_PACE") {
        return &g_host_pace;
    }
    if (name == "EDEN_NX2_ARB_SPIN") {
        return &g_arb_spin;
    }
    if (name == "EDEN_NX2_IDLE_PAUSE") {
        return &g_idle_pause;
    }
    return nullptr;
}

struct CaptureRequest {
    u64 id;
    std::filesystem::path directory;
    bool sampled;
    /// The uniform and storage buffers of every draw and dispatch, and dispatch group counts.
    bool buffers;
};

struct CaptureResult {
    std::string error;
};

struct GpuReadRequest {
    u64 id;
    u64 gpu_address;
    u64 size;
};

struct GpuReadResult {
    std::vector<u8> bytes;
    /// Tegra::MemoryManager::GetID of the address space that maps the whole range.
    u64 address_space;
    std::string error;
};

struct RejectChange {
    u64 generation;
    /// Whether a hash left the list, so programs rejected before must be lowered again.
    bool unrejected;
};

/// Opens a debug server session (clears requests and rejections) or ends it (wakes waiters).
void BeginSession();
void EndSession();

/// Rasterizer thread: whether a capture or a reject change is waiting.
[[nodiscard]] bool HasRequests() noexcept;

/// Server thread: queues a frame capture; the rasterizer takes it at the next frame end.
void RequestCapture(CaptureRequest request);
/// Server thread: withdraws a capture that was not taken yet (it timed out).
void CancelCapture(u64 id);
/// Server thread: waits for the capture's manifest to be complete. Empty on timeout or when the
/// session ends.
[[nodiscard]] std::optional<CaptureResult> WaitCapture(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<CaptureRequest> TakeCapture();
/// Frame dump writer thread, once the manifest summary is written (or the capture failed).
void FinishCapture(u64 id, CaptureResult result);

/// Server thread: queues a read of GPU virtual memory; the rasterizer takes it at the next frame
/// end, where the GPU's own reads of its address spaces run.
void RequestGpuRead(GpuReadRequest request);
/// Server thread: withdraws a read that was not taken yet (it timed out).
void CancelGpuRead(u64 id);
/// Server thread: empty on timeout or when the session ends.
[[nodiscard]] std::optional<GpuReadResult> WaitGpuRead(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<GpuReadRequest> TakeGpuRead();
/// Rasterizer thread.
void FinishGpuRead(u64 id, GpuReadResult result);

/// Server thread: adds or removes native program keys or shader hashes (as EDEN_NX2_REJECT_HASH
/// takes them). Returns the generation to wait for.
u64 SetRejected(std::span<const u64> hashes, bool rejected);
[[nodiscard]] std::vector<u64> RejectedHashes();
/// Lowering: whether the debug server rejects this program. Cheap while the list is empty.
[[nodiscard]] bool IsRejected(u64 hash);
[[nodiscard]] bool AnyRejected() noexcept;
/// Rasterizer thread: the pending reject change, once.
[[nodiscard]] std::optional<RejectChange> TakeRejectChange();
/// Rasterizer thread: the cached programs of this generation were dropped.
void NoteRejectApplied(u64 generation);
/// Server thread: waits until the rasterizer applied the generation.
[[nodiscard]] bool WaitRejectApplied(u64 generation, std::chrono::milliseconds timeout);

} // namespace VideoCore::Nx2Debug
