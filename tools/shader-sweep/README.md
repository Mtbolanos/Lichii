# Offline shader compiler

`shader-sweep` runs captured instructions through Ampere lowering, Maxwell control-flow
analysis and translation, optimization passes, and SPIR-V emission. It does not launch
the emulator or a Vulkan device. Use `spirv-val` separately to validate its output.

## Build

The normal project build exposes the `shader-sweep` target with `ENABLE_DEBUG_TOOLS=ON`.
On Linux, a separate build needs only the common and shader-recompiler libraries:

```sh
sudo apt-get install cmake ninja-build g++ libboost-all-dev nlohmann-json3-dev \
  libssl-dev liblz4-dev libzstd-dev libcpp-httplib-dev libxbyak-dev python3 spirv-tools
cmake -S tools/shader-sweep -B build/shader-sweep -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/shader-sweep -j4
ctest --test-dir build/shader-sweep --output-on-failure
```

CMake 3.24 or newer is required. The standalone configuration downloads pinned fmt,
SPIRV-Headers, sirit, and ZBIC sources. It does not require system SPIRV-Headers.
The tests include synthetic CLI cases, parser bounds checks, and three small captured
regression programs. `spirv-val` is required when `BUILD_TESTING` is enabled.

## Inputs

Compile one extracted v46 to v54 serialized environment record:

```sh
shader-sweep --environment shader.env --cache-version 47 --output shader.spv
```

The record starts at its code-size field. Do not include a disk-cache header, pipeline
key, or neighboring records. Graphics records contain the program header and stage;
compute records contain the workgroup and shared-memory sizes. `--pipeline-key shader.key`
adds the captured 1056-byte graphics pipeline key. Compute keys are not accepted.

To rerun native lowering with captured specialization data:

```sh
shader-sweep capture.bin --environment shader.env --cache-version 47 \
  --pipeline-key shader.key --output shader.spv
```

To compile a whole pipeline record as the runtime does:

```sh
shader-sweep --pipeline pipeline.record --cache-version 54 --output-dir modules [--path legacy|v2]
```

The record is the cache's bytes for one pipeline: the environment count, the environments
and the pipeline key. Its stages are translated in program-slot order and emitted with the
runtime's linkage: each stage sees the previous stage's outputs, descriptor bindings continue
across stages, vertex A merges into vertex B, and task/mesh programs use their recorded mesh
context. `--path` selects the compilation path: `legacy`, or `v2`, which first checks the
record with the v2 reader (`ReadPipelinePayload`) and files modules under the v2 record key. One
`<slot>-<stage>.spv` module per emitted stage is written to the output directory, and only
when the whole pipeline compiles. The JSON result adds `path`, `kind` (`graphics` or
`compute`), `pipeline_hash` (the runtime's cache key hash) and `stages`: per environment its
`slot`, `stage`, `unique_hash`, `key`, `spirv` file name and `spirv_words`. Vertex A has no
module. A failure names the program slot in `failed_slot`. `shader_sweep.py replay` drives
this mode; see [shader-sweep.md](../nx2/shader-sweep.md).

To write the pipeline records of a whole cache as one path reads them:

```sh
shader-sweep --list vulkan.fork-v54.bin --output-dir records [--path legacy|v2] \
  [--programs-log lichii-cache.v2.bin]
```

`legacy` writes the records as they are in the file, and stops at the first damaged one, as
the runtime does. `v2` migrates the cache into a new v2 log under `records/migration`, using
the production migration, and writes the records that the v2 reader reads back. With
`--programs-log`, it gives the records the code of that log's Program records, as the v2 boot
does. The JSON result lists each record's file, offset, kind, `record_key` and stages, and
adds the migration, log and program counts for `v2`. `--path v2` and `record_key` need
`shader-sweep` built in the tree with `-DENABLE_DEBUG_TOOLS=ON`, which links `video_core`. The
standalone build has only `legacy`, and its `record_key` is null.

Raw dumps default to instruction-only format. Use `--format native-header` only for
captures with a 128-byte native graphics header followed by 16-byte instructions.
The header supplies the exact stage. Instruction-only compilation needs `--stage`
or a captured environment. Graphics compilation also requires a header or environment;
the tool does not invent graphics input/output metadata.

The six explicit stage names are `vertex`, `tess-control`, `tess-eval`, `geometry`,
`fragment`, and `compute`. Conflicting explicit, header, and environment stages are errors.
Task, mesh, and vertex-A cache records report missing pipeline context.

For matching a raw capture against a cache, lower without choosing a graphics stage:

```sh
shader-sweep capture.bin --lower-only lowered.bin
```

This writes packed little-endian 64-bit Maxwell instructions, without a header,
scheduling words, or native address map. The reported stage is null when unknown.
Geometry-specific lowering requires `--stage geometry`. `--lower-only` does not run
control-flow translation or SPIR-V emission and is not a full compilation check.

## Offline context and profile

Missing constant-buffer values, texture descriptors, or storage-pointer classifications
are errors, not zero values. v46 records do not contain storage-pointer classifications.
Native trap guards require captured constant-buffer values and must verify before lowering.

The compiler reports assumptions in each JSON result. Without an environment, compute
uses a 1x1x1 workgroup and zero local/shared memory. Without a graphics pipeline key,
the runtime uses point input topology, isolines with equal tessellation spacing, float
interface types, and no alpha test or transform feedback. With a key, vertex input types,
fragment attachment types, topology, tessellation modes, alpha test, and depth/fragment
flags come from the capture. Transform feedback is rejected. Previous-stage varying
linkage is not reconstructed; all previous-stage varying slots are enabled.

The fixed Vulkan-style profile emits SPIR-V 1.4 with unified descriptor bindings,
one-byte minimum storage-buffer alignment, 32-lane subgroups, integer types, subgroup
votes/shuffles, geometry streams/passthrough, viewport/layer output, and eight clip
distances. Descriptor aliasing and native 8/16-bit storage access stay disabled to
exercise the compiler's portable fallbacks. Integer fragment attachments always use
typed outputs. Default emulator shader settings apply, with resolution scaling inactive.
This is an offline compiler exercise, not a driver-specific golden shader or proof of
correct in-game rendering.

`--device-profile FILE` replaces the fixed profile with a device's. The file is a JSON object
with a `host` and a `profile` object of `HostTranslateInfo` and `Shader::Profile` fields by
name, as `PipelineCache`'s constructor sets them from a `Vulkan::Device`; fields it leaves out
keep the fixed values, and an unknown field is an error.
`tools/nx2/radv_offline/radv_pipe --profile` writes one for a RADV device.

## Results

Stdout contains one JSON object, except for `--help`. Stderr receives compiler warnings.
The object has `schema_version: 1`, `status`, `phase`, `stage`, `input_format`,
`source_bytes`, `spirv_words`, and `assumptions`. Lowering adds `lowered_words`.
Errors add `error.code` and `error.message`; lowering errors also include the native
instruction byte `error.offset`.

Exit status is 0 for successful emission, 1 for compilation/context/output failures,
and 2 for invalid arguments or input. Missing specialization data has phase `context`
and code `missing_context`. Compiler assertions routed through the common assertion
API become errors; a process-level crash in a dependency still needs to be detected
by the caller. Compilation failures do not overwrite the requested output.

```sh
spirv-val --target-env vulkan1.2 shader.spv
```

## Static archive inputs

`--capture-extent` accepts an allocated native code region and uses the runtime's
capture state machine to find its complete instruction extent. Padding is retained
in the extraction artifact, not treated as executable instructions. Missing
completion or invalid branch targets remain context failures.

The [all-library sweep report](../../docs/shader-sweep-all-games-20261104.md)
describes read-only inventory, NUP export, bounded archive extraction, preserved
control blobs, and static replay. Its cache associations are inferred, and its
synthetic-context validations are explicitly partial. Neither is a substitute for
capturing the original pipeline state.

## Replay extracted library shaders

Run inventory and extraction on the machine holding the library. Re-list it before
each source operation and skip files that disappeared. Use a fresh output directory
outside the library for every extraction. Examples use Bash variables.

```sh
python3 tools/nx2/inventory_games.py "$LIBRARY" --output "$WORK/inventory.json"
python3 tools/nx2/extract_static_shaders.py \
  --inventory "$WORK/inventory.json" --read-only-root "$LIBRARY" \
  --output "$WORK/static"
```

The Windows command is `py -3`; Python 3.14 supplies the zstd decoder there. Linux
uses the system `libzstd`. The bounded readers support NVN2 BNSH/BFSHA/SARC,
SHN/LZ4/SHDROUNC, and the observed DCX/BND4/CGSB layouts. They preserve complete
code and control allocations, SHA256s, container/member offsets and all origins.
Missing formats, unreadable sources and omitted inventory entries are errors.

The tools do not apply production fixes. To reproduce the coverage report's after column, build a
revision containing all three fixes listed in that report; the tooling branch keeps their code in
separate PRs.

Build the standalone compiler and exporter on Linux with clang and Ninja:

```sh
cmake -S tools/shader-sweep -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DBUILD_TESTING=ON
cmake --build "$WORK/build" --parallel 8
ctest --test-dir "$WORK/build" --output-on-failure
```

The exporter can catalog a NUP without exporting its asset payloads. Explicit
selectors include the package index; inspect the catalog before choosing them.
The broad `--candidates` heuristic can include large non-shader archives, so it
is not a claim that every candidate is a shader.

```sh
"$WORK/build/nup-shader-export" --input "$GAME_NUP" \
  --read-only-root "$LIBRARY" --output "$WORK/catalog" --list
"$WORK/build/nup-shader-export" --input "$GAME_NUP" \
  --read-only-root "$LIBRARY" --output "$WORK/export" \
  --path '0:/Shader/example.bfsha'
```

`tools/shader-sweep/build_nup_windows.py` builds this exporter into scratch using
an already configured clang-cl build's headers and libraries read-only. Its six
compile jobs and all object, PDB and executable outputs stay under `--work-dir`.

Transfer only the extracted shader corpus to the compiler machine, then run:

```sh
python3 tools/nx2/sweep_static_shaders.py "$CORPUS/manifest.json" \
  --read-only-root "$CORPUS" --compiler "$WORK/build/shader-sweep" \
  --cache-directory "$CACHES" --output "$WORK/replay" --jobs 8
```

The replay requires `spirv-val`. Its output directory must be new and outside all
declared read-only roots, input manifests and cache directories. It checks artifact
hashes, retains distinct control blobs, and writes `run.json`, `results.jsonl`,
`results.json`, `archive-errors.json`, and `summary.md`. `run.json` records executable,
runner and manifest hashes. Any failed or partial coverage returns a nonzero exit.

`--capture-extent` uses the runtime's `ProgramCaptureState`; it does not search for
an arbitrary EXIT or invent a missing header. Compute input is bounded by the
observed c9 control record's declared code region before runtime extent discovery.
All 683 complete compute captures in this corpus fit that bound. The other eight
remain incomplete. Workgroup dimensions, resource descriptors and mesh launch
state are not guessed.
