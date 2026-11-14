# Offline shader sweep

`shader_sweep.py` runs native captures and serialized Vulkan shader environments through the
shader compiler and `spirv-val`. It does not start an emulator, access a GPU, or change input files.

## Run

Build the `shader-sweep` C++ target under `tools/shader-sweep`, then run:

```sh
python3 tools/nx2/shader_sweep.py \
    /path/to/native-dumps /path/to/cache/shader \
    --compiler /path/to/shader-sweep \
    --output /path/to/sweep-before
```

Install SPIRV-Tools so `spirv-val` is on `PATH`, or pass `--spirv-val /path/to/spirv-val`.
The script validates against Vulkan 1.2. `--jobs` controls concurrent compiler subprocesses and
`--timeout` bounds each compiler and validator invocation. Every shader runs in its own process,
so an assertion, crash, or hang becomes a result rather than stopping the rest of the sweep.

The output directory contains `results.json`, a grouped `summary.md`, and the individual
environment records, pipeline keys, lowered code, and SPIR-V modules in `records/`.
The command exits nonzero if any input, context, compiler, or validation failure remains.

After changing the lowering, preserve the original native-to-cache matches:

```sh
python3 tools/nx2/shader_sweep.py \
    /path/to/native-dumps /path/to/cache/shader \
    --compiler /path/to/shader-sweep \
    --contexts-from /path/to/sweep-before/results.json \
    --output /path/to/sweep-after
```

Keep the earlier output directory. Its extracted environments and keys supply context when the
new lowering no longer produces the exact bytes stored in the older cache.

## Inputs and coverage

The loader reads little-endian `yuzucach` Vulkan caches, fork versions 46 through 54. It verifies each
record boundary and rejects unknown versions or truncated input. Version 47 includes captured
storage-pointer classifications; versions 48 through 50 retain that layout and version 46 omits them.
Version 49 invalidates pipelines after tessellation patch-size clamping. Version 50 captures the
additional texture-table context needed by dynamic texture/static sampler composition. Neither
changes the serialized layout.
Version 53 stores the three mesh registers and the 80-byte mesh primitive header
for task/mesh environments. It uses a separate Vulkan environment-cache filename;
the runtime must support v53 to load those entries.
The offline reader preserves that context, but standalone task/mesh compilation
still reports an explicit context limitation rather than emitting a guessed module.
Version 54 only rebuilds pipelines emitted with the old float min/max instructions
(2e69168aae); its records, keys and task/mesh context are byte-for-byte the v53 layout,
which is why the runtime starts a v54 cache from a v53 or v52 file.
Graphics keys are the 1,056-byte layout
from these versions; compute keys contain 24 bytes. Driver pipeline blobs and `vulkan_used`
indexes do not contain replayable shader environments and are not treated as shader inputs.

An environment preserves its actual shader stage, header, constant-buffer reads, texture types,
formats, compute workgroup size, and shared/local memory sizes. Cache entries contain already
lowered Maxwell code, not the original native instructions. Replaying them tests translation,
optimization, and SPIR-V emission, but cannot recover native programs rejected before caching.

Normal `hash-gfx-address.bin` captures contain only native instructions. Their filename does not
identify a graphics stage. The sweep lowers these and matches the result against cached code,
excluding headers, scheduling words, padding, and the native address map. A match supplies the
captured stage and environment for a fresh lowering and compilation. It never labels every
graphics program as a vertex shader. `rejected-stage-N-address.bin` captures include a 128-byte
header; `N` is the Maxwell engine shader slot, not the `Shader::Stage` enumeration. Compute
captures use `hash-cs-address.bin`.

Missing context is reported separately from compiler failures. A missing stage, texture format,
constant-buffer value, or storage-pointer classification is not evidence of an unsupported GPU
instruction. Fixed-pipeline assumptions are included in the compiler's JSON diagnostics.
Validation establishes that the emitted module is structurally valid under the tested profile;
it does not establish correct pixels, synchronization, or behavior on every driver.

Native captures default to the game label `Mario Tennis Fever`; use `--dump-game` for another
game. Cache game labels come from their parent title-ID directory; unknown IDs remain visible
as IDs. Counts are distinct compile jobs, including different captured environments and pipeline
keys, not draw counts. Identical native files are deduplicated by content.

If a capture log establishes the stage of an uncached program, pass
`--native-stage 0123456789abcdef=tess-eval`, using its filename hash. This supplies only the
stage, not missing headers or resource data; full compilation still reports that context gap.

## Replay a recorded cache through two compilation paths

`replay` compiles every pipeline record of a recorded cache through path A and path B and
compares the SPIR-V of each stage word for word. It is the correctness harness for a new
translation path: the old path is A, the new one B.

```sh
python3 tools/nx2/shader_sweep.py replay /path/to/cache/shader/0400486024dec000 \
    --compiler-a /path/to/shader-sweep --path-a legacy \
    --compiler-b /path/to/other/shader-sweep --path-b legacy \
    --output /path/to/replay --jobs 8
```

Inputs are `vulkan.bin` or `vulkan.fork-vNN.bin` files, or directories searched for them.
`--compiler-b` defaults to `--compiler-a`, and both paths default to `legacy`. The paths are
`legacy` and `v2` (see [Compare with shader cache v2](#compare-with-shader-cache-v2)). A new
path is added to `shader-sweep --path` and needs no change here. The output directory must be
new or empty.

Each record is passed whole to `shader-sweep --pipeline` (see
[its README](../shader-sweep/README.md)), once per path. The legacy path translates and emits
the record as `PipelineCache::CreateGraphicsPipeline` does: stages are linked in order, so a
stage sees the varyings the previous one writes, descriptor bindings continue across stages,
vertex A merges into vertex B, and task and mesh programs use their recorded mesh registers
and primitive header. Compute records are emitted as `CreateComputePipeline` does.

Each stage gets one verdict:

- `match`: both paths emitted the same words, or neither emitted a module (vertex A).
- `mismatch`: the modules differ. The report gives both word counts and the first differing word.
- `one-sided`: one path compiled the pipeline and the other didn't.
- `unreplayable`: both paths failed. The summary groups these by cause.

Every stage also records the key it is filed under on each path (`key_a`, `key_b`) and the
pipeline hash. Legacy keys are `<pipeline hash>/<program slot>` for graphics and the pipeline
hash for compute, where the hash is `GraphicsPipelineCacheKey::Hash()` or
`ComputePipelineCacheKey::Hash()`. When two records in one cache have different SPIR-V under
the same path B key, the replay reports a key collision: a lookup by that key would return the
wrong shader.

The command exits nonzero on any mismatch, one-sided failure, key collision or unreadable
cache. The output holds `replay.json` with one row per stage (source, record id, record and
environment byte offsets, slot, stage, keys, verdict, errors) and `summary.md`. Records and
modules of stages that match are deleted; the rest stay under `records/` and `spirv/a|b/` for
rerunning `shader-sweep` by hand. `--keep-spirv` keeps everything.

The legacy keys alias by design in two cases, so a legacy path B reports collisions there:
compute records the runtime holds as alternative translations of one program (null-pointer
rebuilds, tried in order at boot), and later graphics records with the key of an earlier one,
which the runtime never builds. A new path's content keys must not collide.

### Compare with shader cache v2

`--path-b v2` compares the legacy cache with what shader cache v2 makes of it, using the
production code linked into `shader-sweep`:

1. `MigrateLegacyPipelineCache` migrates a copy of the cache into a new v2 log in the output
   directory, and `ReadLogFile` reads it back.
2. If a `lichii-cache.v2.bin` log is next to the cache, its Program records are used as the boot
   uses them. `LoadPrograms` and `FindProgram` find the records of this build's lowering, and
   `RelowerProgramRecord` lowers the records of other builds again. `RekeyPipelineRecord` then
   gives each pipeline record that uses one of these programs the program's code and new key.
3. `shader-sweep --pipeline --path v2` checks each record with `ReadPipelinePayload`, as the
   boot does, then compiles it.

Each side lists the cache itself (`shader-sweep --list`). The records are paired by the v2
record key, which leaves the driver out: `GraphicsRecordKey` for graphics and
`ComputeRecordKey` for compute. The legacy listing computes the same key, so a side with
`--path v2` needs a `shader-sweep` built in the tree, for both compilers. Within a key, the last
legacy record is paired, because the migration keeps the last one. Two more verdicts cover the
records the migration leaves out. They are counted but are not failures:

- `collapsed`: an earlier legacy record whose key a later one has. The migration kept only the
  later one.
- `dropped`: a legacy record with no v2 record of its key.

A v2 record with no legacy record is `one-sided`. Path B keys are `<record key>/<program slot>`,
and a key collision there is fatal as before. The summary gives, for each cache, the migration
counts (source version, graphics, compute, collapsed) and the Program records: how many were
found, lowered again or unusable, and how many pipeline records were moved to their code. A
moved record keeps its original key in `record_key`, and its row has `relinked: true`.

The fixed offline profile replaces the device: geometry, viewport-layer and mesh shader
support, not MoltenVK, no device-specific workarounds. Transform feedback pipelines are not
replayed and fail on both paths. A replay compares two paths under that same profile; it does
not reproduce the SPIR-V a particular GPU received.

### Run on the owner's PC

Recorded caches contain game shaders, so the replay runs on the Windows PC and its output stays
there. Build `shader-sweep.exe` with clang-cl in a separate build directory, so the main build
stays as it is. Configure it with the flags in `C:/lx/configure.sh` plus `-DENABLE_DEBUG_TOOLS=ON`,
then build only this target:

```sh
cmake --build C:/lx/build-tools --target shader-sweep --parallel 12
```

This in-tree build links `video_core` for `--path v2`, so the first build takes as long as an
emulator build.

Copy the cache while the emulator is closed, because the runtime appends to the file and
rewrites it. For `--path v2`, also copy `lichii-cache.v2.bin` from the same folder if it exists.
Then run the replay with native Windows Python:

```sh
PY=/c/Users/Zortos/AppData/Local/Programs/Python/Python314/python.exe
$PY -I C:/lx/src/tools/nx2/shader_sweep.py replay \
    C:/Users/Zortos/.capy/work/replay/caches/0400486024dec000 \
    --compiler-a C:/lx/build-tools/bin/shader-sweep.exe \
    --output C:/Users/Zortos/.capy/work/replay/oot-legacy --jobs 12
$PY -I C:/lx/src/tools/nx2/shader_sweep.py replay \
    C:/Users/Zortos/.capy/work/replay/caches/0400486024dec000 \
    --compiler-a C:/lx/build-tools/bin/shader-sweep.exe --path-a legacy --path-b v2 \
    --output C:/Users/Zortos/.capy/work/replay/oot-v2 --jobs 12
```

The script uses only the standard library and needs no `spirv-val`.

## Parser tests

```sh
python3 -m unittest discover -s tools/nx2 -p test_shader_sweep.py -v
```

Keep bulk game captures and generated modules outside the repository. The small regression
environments in `tools/shader-sweep/fixtures` preserve captured programs that exposed backend
defects. Each records its source cache hash and byte offset. Run them with:

```sh
python3 tools/shader-sweep/test_captured.py /path/to/shader-sweep
```

`tools/shader-sweep/test_replay.py /path/to/shader-sweep` builds v54 caches from the same
fixtures (single-stage, linked and compute records and a duplicate record) and replays them
with one binary as both paths, serially and with four jobs. With `--v2`, which the in-tree
`shader-sweep-replay-v2` test passes, it also replays them against `--path v2`. The migrated
cache must match on every stage, with the duplicate record collapsed. Records that differ only
in the driver fields must pair. A byte planted in one migrated record must give exactly one
mismatch.

## Playthrough hints

Use `nup-shader-export` and `extract_static_shaders.py` to produce the local
archive manifests. BAHS v11 and BFSHA are supported. For archives compressed
with a game dictionary, pass `--zstd-dictionary-pack` to the extractor
(Python 3.14 or `backports.zstd`).

Close the emulator before exporting. The hint file contains pipeline keys,
captured translation metadata, and shader fingerprints. It excludes shader
instruction bytes, saves, and driver cache blobs. Specify the exact game update
for both commands. Use hints from a trusted source: the pipeline metadata
comes from the recording even though the shader code is rebuilt locally.

```sh
python tools/nx2/playthrough_hints.py export \
    --title 0400486024dec000 --game-version 131072 \
    --cache /path/to/0400486024dec000/vulkan.fork-v52.bin \
    --output /path/to/zelda.hints.json

python tools/nx2/playthrough_hints.py import \
    --title 0400486024dec000 --game-version 131072 \
    --hints /path/to/zelda.hints.json --manifest /path/to/static/manifest.json \
    --compiler /path/to/shader-sweep --output /path/to/rebuilt-cache
```

Zelda's NVN2 `.vsetb.byml.zs` effect bundles may contain additional embedded
BNSH graphics and compute programs. Include the shared effects bundle and
relevant scene bundles in the local extraction inputs.

Pass each local extraction manifest with a separate `--manifest`. The importer
requires an exact packed shader match for every stage and reports incomplete
pipelines separately. NVN2 BNSH and BAHS task and mesh records can supply
the local code for v53 hints. The importer keeps the recorded mesh context and
requires an exact packed-code match for every stage, including the fragment
shader. The result is a new cache directory. For testing,
copy the resulting title directory into an empty portable profile's
`user/cache/shader` and launch with "Build all cached shaders before game start" on. The
cache format and pipeline state must be compatible with the recipient's build
and Vulkan features.
