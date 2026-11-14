# GPU traces

A GPU trace (`.lxtrace`) holds what the emulated GPU consumed over a window, in the order it consumed it, after a snapshot of the GPU state at the window's start. It records guest inputs (command words, guest memory, engine registers, maps and presents), never host Vulkan calls, so a replay runs the shader translation, the caches and the Vulkan backend of the build that replays it.

## Replay

```sh
eden-cli --replay <trace> --out <dir> [--frames N] [--device N]
```

The replay needs no game, no window and no keys. It brings up the GPU on a bare `Core::System` (`System::InitializeForGpuReplay`: kernel, Host1x and the GPU, no process, audio or services), restores the snapshot and feeds the records through the GPU in order. It writes per present:

- `frame-NNNN.png`: the composited frame at the size of its largest layer (crop rectangle), through the renderer's screenshot path.
- `frames.json`: per frame its file, size, XXH3-128 hash of the RGBA bytes and the wall time since the previous frame, plus the program id and the device that replayed it.

The window is a `VK_EXT_headless_surface` surface (`WindowSystemType::Headless`), so any Vulkan driver with that extension works, Mesa's lavapipe included.

`eden-cli --gpu-trace-sample <trace>` writes a synthetic trace that needs no game: one Switch 2 channel whose four frames are two CPU-written images and two sets of 3D engine clears (whole and scissored). The unit tests read it back (`[gpu_trace]`).

## Recording

A recording starts at the GPU thread's next command boundary where no channel waits on a semaphore, and ends after a number of presents (180 by default: 3 s at 60 fps). Start it with:

- the debug server: `trace_record` `{seconds?, presents?, path?}` (default 3 s, into the dump directory), `trace_stop`, `trace_status`;
- or `EDEN_NX2_GPU_TRACE=<dir>`: touching `<dir>/record.now` starts `<dir>/trace-<time>.lxtrace` (`EDEN_NX2_GPU_TRACE_PRESENTS=<n>` sets the length).

What happens (`video_core/gpu_trace/recorder.cpp`):

1. **Snapshot.** The GPU split drains and the 3D and compute engines take their state back from its front. Then the recorder writes every GPU address space and its mappings, from a map log each address space keeps, and the engine and global state. GPU-written data in the caches is flushed to guest memory, and every guest page the mappings reach is copied (zero pages skipped, identical pages stored once).
2. **CPU writes.** For the window, the recorder holds a cached-page reference on every page it copied, so each guest CPU write to them takes the rasterizer write path, where it is noted after the store. Before each later GPU command, the bytes written since the previous command are recorded with their current content, so GPU results on the rest of the page stay the replay's own. Recording needs the JIT CPU backend: NCE notes a write when it faults, before the store lands, so a recording under NCE fails to start. CPU-heavy scenes run slower while recording.
3. **Commands.** Each command list is recorded when the GPU thread takes it: its GPFIFO entries, the words copied at submit and the entries still deferred. GPU memory maps are recorded when they happen, with the SMMU mapping and content of any new pages.
4. **Presents.** A composite request is recorded at the GPU tick that runs it, in order with the commands.

`eden-cli --replay <trace> --out <dir> --record <new trace> --record-from <present>` records a replay from a present on. Replaying the new trace must give the same frames as the original from that present: that round trip checks the recorder, the snapshot and the restore against each other.

A channel that starts submitting after the snapshot ends the recording before its first submit, and a GPU shutdown ends it at once; both traces replay up to that point. Not recorded yet: SMMU unmaps (the replay keeps the stale mapping), sparse backing changes, the order in which suspended channels resume, and Switch 2 semaphore region changes after the snapshot. There is no Qt hotkey yet: use the debug server or the file trigger.

## File format

A 64-byte header (magic `LXGTRACE`, format version, header size), then one zstd stream of records: a 16-byte header (`type`, `size`) and `size` payload bytes. `video_core/gpu_trace/trace_file.h` defines every record. In order:

1. `Config`: JSON with the program id, whether it is a Switch 2 title, the memory layout and the output size.
2. Maps: `AddressSpace` (a GMMU with its page geometry), `GmmuMap`, `SmmuMap` (device pages to physical pages, so aliasing survives).
3. `Channel` per GPU channel, then `State` records: engine registers, the pusher's method state, the Switch 2 macro RAM, QMD scheduler state, syncpoints and semaphore regions (`gpu_trace/state.h`).
4. `Memory`: physical ranges and their content. Content is a `Blob` stored once and named by its XXH3-128 hash, so identical pages cost nothing after the first.
5. `SnapshotEnd`, then the stream: `Memory` (a CPU write the GPU sees from that point on), `Submit` (GPFIFO entries, prefetch words, submit-time word copies), map changes, `Present` (the layers and fences of a composite request) and `Resume`.
6. `End`.

Physical addresses are offsets into guest memory (`Core::DeviceMemory`). The replay maps them through a process page table whose addresses are those offsets.

## Lavapipe

Lavapipe creates the device (every required feature, extension and limit is there). Its gaps:

| Gap | What the replay does |
|---|---|
| Subgroups are 8 wide and can't be forced to 32 | The guest warp size is requested only where the device's subgroup size range includes 32. Warp operations (SHFL, VOTE, ballot) then run on 8 lanes and give other results than a GPU: frames that use them differ. |
| No depth bounds test | The existing fallback skips the test. |
| No ASTC | The existing CPU decode. |
| MSAA 1x and 4x only | 2x and 8x targets fail to create. |
| CPU rasterizer | Correct, slow: a trace replays at a few frames a second. |

Frames match bit for bit only between runs of the same build on the same driver. Across drivers they differ by precision, so expected frames are kept per device class.

## What a trace can't hold

- Guest CPU timing. A bug that comes from the order between the CPU and the GPU thread replays in the recorded order and may not reproduce.
- Content that never reaches guest memory: resolution-scaled images and MSAA targets that can't be downloaded. Record at 1x for exact traces.
- Driver bugs: those need the same driver to replay.
