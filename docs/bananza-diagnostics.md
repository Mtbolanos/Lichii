# Bananza diagnosis build

Program ID: `040074A01BF12000`.

Neither the camera-dependent black screen nor the 30 fps drop has a demonstrated
cause in the supplied logs. The shader and indirect-reset ports fix independently
testable gaps, not a proven explanation of either symptom. See
[the upstream comparison](bananza-upstream-comparison.md) for the source commits
and retained fork behavior.

## Paths checked

`ShaderCache::RefreshStages` remembers native rejections and rejects later
draws using the same program. `RasterizerVulkan::PrepareDraw` also skips absent
pipelines, unavailable mesh support and failed resource configuration. Those paths
can depend on which objects a camera sees, so their stage addresses and counts are
logged rather than hidden behind the existing one-time warnings.

The asynchronous-pipeline heuristic is not a reason to discard native pending
draws: `PipelineCache::BuiltPipeline` uses `CanDiscardPendingDraw`, whose native
case returns false. Existing tests cover that policy. Unsupported indirect mesh
draws remain unsupported and now get bounded stage/address diagnostics.

`Maxwell3D::ProcessQueryCondition` can suppress a draw before the rasterizer sees
it. Vulkan can instead defer that decision to a GPU predicate. Both paths exist
and are traced without changing their decisions. Indirect commands can also have
zero live counts or zero-sized command records, which the optional readback samples.

The inspected first-use paths do not justify a blanket resource clear.
`QueryCacheRuntimeImpl` initializes its conditional-render flags, and the resolve
path writes its predicate before enabling it. `Image::UploadMemory` tracks the
first image transition; sparse-image initialization explicitly calls `ClearTiles`.
An initial `VK_IMAGE_LAYOUT_UNDEFINED` is not proof of an uninitialized read.
This inspection does not establish that every Bananza resource is initialized
correctly; no resource-clear workaround was added.

## Capture on Windows

Restart the emulator after setting environment variables. In PowerShell, launch
the diagnostic executable from the same shell:

```powershell
$env:EDEN_DKB_FRAME_TRACE = "1"
Remove-Item Env:EDEN_DKB_INDIRECT_READBACK -ErrorAction SilentlyContinue
& .\eden.exe
```

Use the executable's actual filename if it differs. Set the emulator's log filter
to `*:Info`. All new diagnostics are Info-level under `Service`, `HW.GPU`, or
`Render.Vulkan`; Debug logging is not required. Keep the same renderer, resolution,
GPU accuracy, speed limit and vsync settings between runs. Do not enable an
experimental shader override for the comparison.

Hold the problematic camera position for at least five seconds, then move the
camera until the image or frame rate recovers and hold it for another five seconds.
Record the approximate log times of these transitions. Preserve the complete log,
including shader rejection messages before gameplay. Do not clear shader caches
between these two positions.

For a second run focused on the black image, also enable:

```powershell
$env:EDEN_DKB_INDIRECT_READBACK = "1"
& .\eden.exe
```

This copies actual GPU indirect arguments, not potentially stale guest-memory
copies. It adds render-pass breaks and transfer barriers, so use the first run for
timing comparisons. If enabling readback changes the symptom, report that fact;
the readback is diagnostic, not a rendering fix.

Extract relevant lines while keeping the original log:

```powershell
Select-String -Path .\eden_log.txt -Pattern 'DKB|Rejected program|shader rejected|Draw skipped|swap interval|Native indirect reset'
```

## Interpret the evidence

- A request for swap interval 2 establishes that the guest asked for 30 Hz pacing
  at a nominal 60 Hz refresh. It does not establish why the guest chose it. Compare
  the preceding frame timings and the interval transitions, rather than treating
  every 33 ms present gap as a pacing bug.
- `DKB draw skipped` names the reason, render target and enabled stage addresses.
  Stage slots 0 through 5 are vertex-A, vertex-B, tessellation-control,
  tessellation-evaluation, geometry and fragment. Pair these with the existing
  `Rejected program` and `Native graphics shader rejected` lines. Compute
  rejections include `stage=compute`, their program address and the exception.
- `DKB draws` counts rasterizer attempts, skipped calls, indirect calls and zero-sized
  direct draws since its previous line. It logs the first four rasterizer frames
  and then at most once per second. These totals are not a count of visible objects;
  CPU-culled draws never reach the rasterizer and appear in condition diagnostics.
- With indirect readback enabled, `DKB indirect` samples at most four calls in each
  of those frames and at most eight command prefixes per call. `count_source=gpu`
  identifies a GPU count-buffer read; `command-parameter` is the fixed draw count.
  `requested_draws=0` means no indirect commands were requested by that sampled call.
  A nonzero value does not prove execution; a GPU predicate can suppress the call.
  `zero_commands` counts sampled commands with zero vertices/indices or instances.
  Commands beyond the sampled prefix are unknown. Transform-feedback byte-count
  draws report `gpu_byte_count` separately. `capture_dropped` means all eight
  readback slots were still busy or a buffer range could not be sampled.
- Readback never waits for the GPU. Results can appear several frames later;
  their `frame` identifies the originating rasterizer frame. Renderer frame IDs,
  query event counters and service present counters are different sequences.
- Query and conditional-rendering diagnostics distinguish actual CPU decisions,
  completed host query values, GPU-deferred predicates, and existing unconditional
  bypasses. See [query trace semantics](bananza-query-trace.md). A GPU-deferred
  predicate is not a measured pass or fail.

The timing fields and their limits are described in
[timing trace semantics](bananza-timing-trace.md). A large GPU duration points to
host GPU work, while a large host interval with a short GPU duration calls for CPU,
submission, or pacing investigation. Neither result alone proves that the guest's
dynamic frame-rate logic made the right decision.

If shader, draw, count and predicate observations remain normal while the screen
is black, this trace has not ruled out a resource initialization or shader-output
bug. The existing `EDEN_NATIVE_FRAME_CAPTURE` and `EDEN_NATIVE_DRAW_CAPTURE`
switches capture a bounded set of early resources, but produce larger, more
intrusive captures and are not required for the first run.

No game was run in the cloud. Windows Vulkan execution and symptom resolution
still require the owner's test.

## Cloud build

Use CMake 3.31 or newer and a Clang version that accepts `-mtls-dialect=gnu2`.
The cloud check uses Clang 20, since Ubuntu's Clang 18 rejects that existing flag.
The headless build configuration is:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_QT=OFF \
  -DLICHII_CMD=ON -DLICHII_TESTS=ON -DENABLE_LTO=OFF \
  -DENABLE_WEB_SERVICE=OFF -DUSE_DISCORD_PRESENCE=OFF -DENABLE_WERROR=OFF
cmake --build build --target lichii-cmd tests --parallel "$(nproc)"
ctest --test-dir build --output-on-failure
```

`ENABLE_WERROR=OFF` is needed for pre-existing Clang warnings in the audio tests,
including range-loop copies and signed comparisons. Those tests are not changed
or disabled by this work.
