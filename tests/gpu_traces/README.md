# GPU trace corpus

GPU traces (`.lxtrace`, see `docs/nx2/gpu-trace.md`) replay a scene without the game, so they make regression tests for rendering. Traces from games hold the game's shaders, textures and buffers, so they never go into this repository or a public place. The repository keeps only what checks them:

- `corpus.json`: per trace its `id`, `file` (the trace's file name in the private store), `sha256`, `size`, the game's `program_id`, the `scene`, the `bug` it covers and `expected`: per device class (`lavapipe` for Mesa's CPU driver) the XXH3-128 hash of every replayed frame, in order. `"source": "generated"` marks a trace `eden-cli` writes itself, which needs no store. `roundtrip_from` also records the replay from that present and checks that the new trace replays to the same frames.
- `expected/<id>/`: when a frame's look matters for review, a small PNG of it (only frames whose content may be public).

Check the corpus (CI runs this on lavapipe with the generated traces):

```sh
python3 -I tools/nx2/trace_replay.py check --cli build/bin/eden-cli \
  --corpus tests/gpu_traces/corpus.json --traces <directory with the private traces>
```

Compare two builds on one trace (an HTML page with the frames side by side, and the largest difference, mean difference and PSNR of every frame that differs):

```sh
python3 -I tools/nx2/trace_replay.py compare --base old/eden-cli --head new/eden-cli trace.lxtrace --out cmp
```

To add a trace: record it at 1x resolution, put it in the private store, replay it on lavapipe with `trace_replay.py replay` and add its entry with the hashes it printed. Hashes are only comparable on the same device class.
