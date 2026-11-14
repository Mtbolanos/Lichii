# Bananza fixes after the lava-cave profile

This pass integrates upstream merge `13f533027c41ce1fe5452ea990ee8f497fe2d418`,
whose upstream parent is `a666e4400b1a877145126411cce7ab49cf340793`. It retains
the atomic fix `ca6b8e3` and pacing fix `b785035`. The performance PRs come through
that merge, not a separate hand-port. No game was run in the cloud.

## Evidence and upstream coverage

The owner's `profile-run.txt` comes from `ba2d291`, before these changes. Late
samples show about 47–53 host CPU ms per frame on each GPU command/back thread,
less than 16 ms on each guest-core host thread, and about 8–14 ms on VulkanWorker.
The command/back path constrains the host in those samples. The thread times
overlap, and spinning counts as CPU execution.

At 104.648 s, the front reports 648.5 wall-ms/s in `split drain`. The complete
GPU-back window at 105.443 s reports:

| Zone | Wall-ms/s | Calls/s | Mean wall time per call |
| --- | ---: | ---: | ---: |
| Update buffers | 132.0 | 93,277 | 1.415 us |
| Engine methods | 110.8 | 1,313,858 | 0.084 us |
| Bind stage buffers | 81.4 | 210,578 | 0.387 us |
| Read handles | 33.3 | 93,277 | 0.357 us |
| Fill image views | 41.3 | 93,277 | 0.443 us |
| Split wait | 268.6 | 136,656 | 1.966 us |

These are instrumented wall-zone averages, not individual CPU timings. They
cannot establish the costs remaining after the merge and atomic fix.

- #256 covers descriptor-ID lookup in the handle/image-view path. Its array
  access preserves descriptor reads and invalidation; it does not optimize the
  geometry-buffer synchronization inside `update buffers`.
- #257 guards empty storage/texture bindings and changes pipeline equality.
  Populated bindings and uploads still do their work.
- #261 skips repeated mask assignments and empty unbinds. It does not change
  `SynchronizeBuffer()` or eliminate synchronization.
- #262 changes inline descriptor capacities and texture-buffer traversal.
  The old window spent only 3.9 wall-ms/s in `texture buffers`.
- #234 requires `!GpuSplit::Active()`. Its DMA batching cannot accelerate the
  split-active capture. The back thread already collects non-execution methods.
- #233 chunks large image uploads. The late profile has no texture-upload cost,
  so it is not evidence of a steady-state FPS improvement here.

The merge also retains framebuffer minimum layers, geometry, DEPBAR and the
`ad98585` narrow-storage CAS correction. OpenGL/Vulkan cache versions are 45/48.
MKW's dynamic-FPS gates coexist with the current-period present pacer; they do
not apply a title-specific FPS patch to Bananza.

## Additional correction beyond upstream

Upstream #256 still fills every reserved image-ID slot on each Debug deletion
or scale invalidation. The table reserves at least 524,289 slots even with a
limit of zero. `LiveBoundIds()` bounds poisoning to the addressable range:
one four-byte ID at limit zero, or 512 bytes at limit 127, instead of about 2 MiB
per initialized table. It does not allocate, change descriptor reads, or alter
Release clearing. A sparse table with a high limit can still cost more than the
old populated-map walk; this is not a general Debug FPS claim.

Additive tests cover ID hits and mutation, null IDs, compute/channel separation,
deletion invalidation, address/limit changes and bounded poisoning. The hidden
`[descriptor_lookup_bench]` compares the real previous flat-map lookup with the
merged array lookup. It measures synthetic lookup CPU time, not full cache calls
or game FPS; construction and invalidation are outside its timed region.

No further synchronization shortcut is selected. `update buffers` includes both
`UpdateGraphicsBuffers` and `BindHostGeometryBuffers`; the profile cannot isolate
the cost of clean-word tracking inside `SynchronizeBuffer`. A CPU-clean range can
still have untracked pages requiring notifications. A safe follow-up needs clean,
dirty and untracked-only word counts before considering an early return.

The 584,899 same-value register writes/s do not justify dropping executable
methods. That decision needs method-class, sink-length and payload counts.
The native pre-fetch drain remains before GPU-written classification, alongside
upstream's writer recheck. Snapshot and first-write ordering regressions remain
enabled. #157's unmap-unsafe pointer cache is not imported.

## Integrated verification

The resolved tree built `lichii-cmd` and `tests` with Clang 20, Ninja and all eight
available cores. All 529 test cases passed with frame tracing disabled and with
`EDEN_DKB_FRAME_TRACE=1`. The atomic, pacing, descriptor and command-fetch filter
passed 22 cases and 135,908 assertions.
Both Vulkan and OpenGL rasterizers also passed syntax checks with Debug validation
enabled. All seven descriptor cases passed AddressSanitizer and
UndefinedBehaviorSanitizer with `NDEBUG` unset, totaling 121,692 assertions.

The descriptor benchmark ran seven alternating-order trials per workload on
logical CPU 2. Across six workloads, merged array lookup took 0.697–0.860 times
the previous map lookup time, about 14–30% less. This confirms an isolated benefit
of inherited #256 on this shared Linux host, not an FPS improvement or a benefit
of the separate Debug correction.

The Vulkan test program was rebuilt against the merged shader library without
substitute backend objects. All six emitted SPIR-V fixtures passed `spirv-val`
for Vulkan 1.2 with uniform-buffer standard layout. All six native/NVN contention,
overflow and unmatched-address scenarios passed on llvmpipe, each using 256
invocations across eight workgroups. These are synthetic fixtures, not executions
of the nine rejected game shaders or a Windows/NVIDIA run.

Build, test, benchmark and Vulkan records are under
`~/.capy/work/dkb/integration/merged-*`. The source tests are committed; the Vulkan
test program is recorded under `~/.capy/work/bananza-compute-tests`.

## Check the Windows result

First compare the same lava-cave position and camera directions with
`EDEN_DKB_FRAME_TRACE=1`, `*:Info`, and `EDEN_NX2_GPU_PROFILE` unset. Keep resolution,
accuracy and speed settings unchanged. Leave `EDEN_DKB_INDIRECT_READBACK`,
`EDEN_NX2_SWAP_INTERVAL` and `EDEN_NX2_VSYNC_HZ` unset. Setting the last variable to
zero does not disable its override. Record any other environment overrides.
The incoming defaults use split mode 2 and batched fence release; leave
`EDEN_NX2_SPLIT` and `EDEN_NX2_FENCE_BATCH` unset for that baseline. These changes
are another reason not to treat the old profile as a post-merge measurement.

Check whether all nine failing shaders, including the six reported examples,
stop rejecting `GlobalAtomicIAdd32`, and whether the terrain or black screen
changes. A `no routable storage buffer` error identifies a remaining pointer
problem. Executing previously skipped work can increase the workload; assess
rendering correctness separately from FPS. See [atomic evidence](bananza-atomic-fix.md).

For a second run, enable `EDEN_NX2_GPU_PROFILE=1` and hold each camera direction
for at least 20 seconds. Save the complete startup log and mark visible speed
swings. Compare zone time per call, host CPU time per frame and frame durations.
The new `DKB vsync` records expose callback lateness/gaps and guest ticks, but
callback counts are not guest-consumed vblanks because signals can coalesce.
See [pacing evidence](bananza-pacing-followup.md).
