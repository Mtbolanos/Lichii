# Offline AMD shader compile (RADV)

Compile Lichii's shaders with AMD's real compiler, Mesa RADV with ACO, for chips we don't have,
and check their SPIR-V for constructs whose results differ between drivers. Nothing runs on a
GPU: RADV builds the pipelines on a fake amdgpu device, so this answers "what does AMD's
compiler make of this shader", not "what does it draw".

| Chip | Cards | drm-shim device |
|---|---|---|
| gfx1030 | RX 6800, 6900 | navi21 |
| gfx1032 | RX 6600, 6650 XT | navi23 (derived, see below) |
| gfx1100 | RX 7900 | navi31 |
| gfx1102 | RX 7600 | navi33 |
| gfx1200 | RX 9060 | gfx1200 (derived) |
| gfx1201 | RX 9070 | gfx1201 |

## Setup

On an Ubuntu 24.04 machine (x86_64 or aarch64), run:

```sh
bash tools/nx2/radv_offline/setup.sh
```

It installs the packages it needs and builds these under `~/.cache/lichii-radv-offline`
(`RADV_OFFLINE_HOME` moves it):

- Mesa 26.3.0 with only RADV, as a debugoptimized build with assertions, so NIR and ACO
  validation run. RADV compiles with ACO; LLVM 20 is linked only to disassemble the ISA.
- Mesa's amdgpu drm-shim. Mesa 26.3 has no null device behind `RADV_FORCE_FAMILY` any more; the shim
  (an `LD_PRELOAD` library that fakes the amdgpu kernel driver) replaces it.
  `add_shim_devices.py` adds navi23 and gfx1200, which the shim lacks, by copying their
  same-generation sibling (navi21, gfx1201) and changing the family, PCI id, revision and unit
  counts. The compiler keys on the family and gfx level, so the ISA is the real chip's.
- Fossilize at a pinned commit, with its capture layer.
- `radv_pipe`, the small host program described below.

Rerunning the script rebuilds only what changed. The first run takes about 10 minutes on 2 cores.
`compile.py` sets the driver environment itself; `source ~/.cache/lichii-radv-offline/env.sh` is
only for running the tools by hand.

The `record` and `cache` modes also need the standalone `shader-sweep`
([its README](../../shader-sweep/README.md)):

```sh
cmake -S tools/shader-sweep -B build/shader-sweep -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build/shader-sweep --target shader-sweep
```

## Which input to use

The pipeline the driver compiles is more than the SPIR-V: specialization constants, the
descriptor layout, colour formats (they decide how colour is exported), dynamic state and the
requested subgroup size all change the ISA. The inputs differ in how much of that is Lichii's.

1. **A Fossilize capture** (`foz`) has the best fidelity. The layer records every pipeline
   Lichii's own creation path made on the tester's machine, with all its state and the SPIR-V
   Lichii emitted for that device's profile (an AMD card gets the AMD profile). The replay builds
   exactly those pipelines for any chip. It needs one run of the game with the layer, on Linux:

   ```sh
   VK_LAYER_PATH=~/.cache/lichii-radv-offline/fossilize/share/vulkan/implicit_layer.d \
   FOSSILIZE=1 FOSSILIZE_DUMP_PATH=$HOME/oot-capture ./lichii
   ```

   Fossilize also builds on Windows, but this workflow was only tried on Linux.
2. **A pipeline cache record** (`record`, `cache`) needs no new run. `shader-sweep` translates
   the record with Lichii's own code, as `PipelineCache::CreateGraphicsPipeline` does, for the
   RADV device's profile: `radv_pipe --profile` reads the fake device's features and limits into
   the `Shader::Profile` and `HostTranslateInfo` fields `PipelineCache`'s constructor would set
   (SPIR-V 1.6, float controls, demote to helper, subgroup and quad operations, descriptor limits),
   and `shader-sweep --device-profile` translates with them (`device-profile.json` in the
   output). The six chips report the same profile, so one translation serves them all. Shader
   settings stay at their defaults and cooperative matrices off. `radv_pipe` then rebuilds the
   rest of the state: Lichii's dynamic state set, its robustness
   (`robustBufferAccess` on, `robustness2` access off), and its subgroup rule (32 lanes for
   stages with warp operations, where RADV can size the stage). Colour formats aren't in the
   SPIR-V: they default to `R16G16B16A16_SFLOAT`, so pass the real ones with `--color` (the frame
   dump's manifest has them) when colour export matters. Compute pipelines need no such state.
3. **SPIR-V modules** (`spirv`), for example from a frame dump, get the same rebuilt state.

Running Lichii's real `GraphicsPipeline::MakePipeline` against the fake device would need its
scheduler, render pass, buffer and texture caches and a full `Vulkan::Device` whose queue never
runs anything. Fossilize already records the output of that path without changing it, so the
tool records with Fossilize where it can and rebuilds the state from the record where it can't.

`radv_pipe` creates one pipeline with `VK_KHR_pipeline_executable_properties` and writes each
executable's statistics and internal representations. It reads a line-based description that
`compile.py` writes from SPIR-V reflection (`pipeline.txt` in the output):

| Line | Meaning |
|---|---|
| `kind compute` or `kind graphics` | the pipeline type |
| `stage <stage> <file.spv> <entry> [subgroup=32] [full]` | a stage, its required subgroup size, full subgroups |
| `spec <stage index> <id> <u32>` | a specialization constant |
| `binding <set> <binding> <VkDescriptorType> <count> <stage mask hex>` | a descriptor |
| `push <stage mask hex> <size>` | the push constant range |
| `color <VkFormat>`, `depth <VkFormat>` | attachments, one `color` line per location |
| `samples`, `topology`, `patch`, `vertex_dynamic 0`, `attribute <loc> <binding> <VkFormat> <offset>`, `robust none` | the rest |

## Run

```sh
RO=tools/nx2/radv_offline/compile.py
SW=build/shader-sweep/shader-sweep

# One pipeline from SPIR-V on two chips, plus a wave32 fragment variant
python3 -I $RO spirv vs.spv fs.spv --chip gfx1032 --chip gfx1102 \
    --variant wave32:RADV_PERFTEST=pswave32 --color R8G8B8A8_UNORM -o out

# One cache record
python3 -I $RO record 000123.record --shader-sweep $SW --cache-version 53 -o out

# A whole cache: check every program, compile the pipelines that use two programs
python3 -I $RO cache vulkan.fork-v53.bin --shader-sweep $SW --cache-version 53 \
    --check-all --program 59c1699f529e7c08 --program 4f5673eca06ebecb -o out

# A Fossilize capture
python3 -I $RO foz capture.0123456789abcdef.1.foz --chip gfx1201 -o out

# Only the SPIR-V checks
python3 -I $RO check *.spv -o out
```

The default is all six chips. `--variant NAME:VAR=VALUE,...` adds a compile with extra driver
variables, for example `RADV_PERFTEST=pswave32,cswave32`, `RADV_DEBUG=nodcc` or `ACO_DEBUG=novn`,
to A/B a suspected compiler path. Program hashes are the unique hashes Lichii logs and
`EDEN_NX2_REJECT_HASH` takes. The cache listing has none, so `cache` translates every record to
find them.

Output:

- `summary.md`: for each chip and variant, each executable's wave size, instructions, registers,
  spills, scratch and occupancy, the ISA notes, and every finding by shader hash.
- `findings.json`: the findings, machine-readable.
- `<chip>[+variant]/<n>-<stage>.{NIR-Shader-s-,ACO-IR,Assembly}.txt`: the final NIR, the ACO IR
  after register allocation, and the disassembly with encodings. Diff two directories to see what
  a chip or a variable changes. `foz` writes one file per pipeline stage with all three.
- `compile.json` and `pipeline.txt`: the raw results and the description.

Caches, captures and everything compiled from them are the game's shaders. Keep them on the
machine you work on (under `~/.capy/work` in the cloud), never commit or upload them, and delete
them when you're done.

## Checks

`spirv-val --target-env vulkan1.3 --uniform-buffer-standard-layout` (a feature Lichii's device
requires) runs on every module. Then each module gets these checks,
reported with its hash. They are static and conservative: a finding marks something whose result
the Vulkan spec leaves to the driver, not a proven bug. Compare its ISA and its frame dump before
calling it the cause.

| Check | Severity | What it flags |
|---|---|---|
| `nan-minmax` | warn | `FMin`/`FMax`/`FClamp`: undefined for NaN operands and for -0 against +0. Maxwell's FMNMX is `NMin`/`NMax`. |
| `nan-test-without-preserve` | warn | `IsNan`/`IsInf` without `SignedZeroInfNanPreserve` for that width: may fold to false. |
| `float-controls`, `denormals` | info | float widths with arithmetic but no signed-zero/Inf/NaN or denormal mode, and how many divisions, sign, atan2 and pow depend on it. |
| `fp16-overflow` | warn | fp32 to fp16 conversions or `PackHalf2x16` with no clamp in front: over 65504 gives Inf. |
| `fp16-arithmetic`, `relaxed-precision` | info | arithmetic in fp16, and values a driver may compute in fp16. |
| `undef` | warn | `OpUndef` used outside a phi. |
| `uninit-local` | warn | a Function variable loaded where no store dominates the load, or a Private variable never stored. |
| `uninit-shared`, `shared-without-barrier` | error, warn | Workgroup memory read but never written, or written and read with no barrier. |
| `output-never-written` | warn | a stage output in the entry point's interface that nothing stores. |
| `descriptor-oob` | error | a constant index past the end of a descriptor array. |
| `descriptor-divergent` | error | a non-uniform index into a descriptor array without `NonUniform`: AMD takes the first lane's descriptor for the whole wave. |
| `descriptor-unclamped` | warn | a dynamic descriptor index with no clamp in front: out of range is undefined even with robustness. |
| `derivative-divergent` | warn, error | implicit-LOD sampling or derivatives in non-uniform control flow; an error after a kill in a non-uniform branch, where the quad's other lanes are gone. |
| `helper-subgroup` | warn | subgroup operations in a fragment shader: helper invocations may or may not take part. |
| `subgroup-32-bits`, `subgroup-lane-mask` | warn, error | only the low word of a ballot or lane mask used, or the lane id masked to 0-31; an error where the chip runs that stage 64 wide. |
| `subgroup-ops` | info | which subgroup operations and built-ins the shader uses. |
| `checker-failed` | error | the checker itself failed on this module (`cache` mode keeps going). |

Uniformity is tracked through values, Function variables, phis at the merges of divergent
branches, and early exits. Fragment inputs, per-invocation built-ins, loads from Workgroup and
Output memory, atomics and non-uniform subgroup results are non-uniform. Constants, push
constants and buffer loads at uniform addresses are uniform.

The ISA notes count instructions whose results can differ from NVIDIA's: `v_cvt_pkrtz_f16_f32`
(16-bit packing rounds toward zero, so overflow gives 65504 instead of Inf), `v_readfirstlane_b32`
(one lane's value used for the whole wave), the `_legacy` min/max/mul instructions, and the
approximate `v_rcp_f32` and `v_rsq_f32`.

## Tests

```sh
python3 tools/nx2/radv_offline/test_checks.py
```

They run the checks on small hand-written modules and need neither Mesa nor SPIRV-Tools.
