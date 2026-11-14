# Bananza shader comparison

This comparison uses these pinned revisions, before the separate upstream merge:

| Ref | Commit |
| --- | --- |
| `origin/nx2` | `16e5ec00c7f154ab435187b020f0a066449d23fb` |
| `upstream/nx2` | `7adcecc6d23b86d84e158572ea6ddd261fc1cd14` |
| `upstream/pr363` | `044bb10c4534118e6666ade5219dc8d4af6b0e2e` |
| `upstream/pr330` | `22e0e41d6b9d23291fabdecfbe8641c59740d75c` |

The final integration base is merge commit `3483630ccbda157d8e29ebf80f94fa59b970a4b2`, on branch `fix/bananza-frame-diagnostics`. The selective ports are applied on top of that base. The original comparison and pre-merge test results below remain revision-specific; the merged native tracking behavior differs as described next.

PR363 includes the PR330 changes through `461015e`. Its current head enables geometry and indirect reset by default. Its earlier opt-in description is not an accurate description of the pinned code.

No game was run. The tests below establish compiler behavior, not the cause or resolution of Bananza's black screen or camera-dependent frame rate.

## Loaded pointers need a traversal boundary

On the final merge base, the PR330 traversal barrier fixes the **legacy fallback path**, not a newly exposed native Bananza path. `CollectStorageBuffers` already returns at `if (native && !storage_buffer)` before either cached `Track` search. Its pointer-pair walker and `NativePointerCandidates` do not follow global memory loads. For the regression's loaded pointer, with or without a constant offset, they find no candidate, so the native access already stays global without the PR330 barrier. If native tracking does find a buffer, the later `if (!storage_buffer)` fallback searches are skipped. This guard is present in `3483630` itself, not introduced by the selective port.

The merged regression now generates both native and legacy environments. The native cases check that the merge's protection is retained; the legacy cases exercise the additional PR330 fix. Do not describe this barrier as a new native Bananza rendering fix or infer that it explains the reported black screen.

In both nx2 baselines, `Track` in `global_memory_to_storage_buffer_pass.cpp` runs an unrestricted `BreadthFirstSearch`. Given a pointer loaded from global memory, it can walk through that load to the descriptor of the buffer containing the pointer. The pass then rebases an access through the loaded pointer against the wrong buffer.

PR330 adds `stop_at_memory` to stop this search at every global memory instruction. PR363 has the same fix. The port retains the fork's pointer-addend, iterator, and control-flow tracking and changes only the fallback search. It does not replace the pass with the older upstream implementation.

The original pre-merge `Loaded global pointers do not inherit the source buffer descriptor` regression constructs a load through `c[0][0x110]`, then writes through the loaded pointer. It tests both the pointer itself and a pointer plus a constant offset. On `16e5ec00c7` before the port, both cases produced `WriteStorage32`, removed `WriteGlobal32`, and incorrectly marked the source descriptor as written. All six failing assertions passed after the port. The source load still became `LoadStorage64`. These results do not establish a native regression on the later merge base.

The pinned upstream nx2 revision also has `NativePointerCandidates` and an environment-dependent pointer classification cache. The original fork instead has static pointer-pair and control-flow analysis. The merge combines both and adds the native early return described above. The PR330 port changes only `Track`, retaining that combined analysis.

## NVN descriptors need guest-slot indexing

Both nx2 baselines index `nvn_buffer_used` by the position in `storage_buffers_descriptors` in these functions:

- `SPIRV::EmitContext::DefineGlobalMemoryFunctions` in `backend/spirv/spirv_emit_context.cpp`.
- `GLSL::EmitContext::DefineGlobalMemoryFunctions` in `backend/glsl/glsl_emit_context.cpp`.
- `GlobalStorageOp` in `backend/glasm/emit_glasm_memory.cpp`.

That bitset describes guest slots, not the compact host binding list. For example, guest slots 3 and 15 can occupy bindings 1 and 2 when binding 0 belongs to another buffer. Testing bits 1 and 2 skips both valid descriptors. Indexing a 16-bit bitset by a descriptor index of 16 or greater is also outside its valid range.

PR330 and PR363 add `UsesNvnStorageDescriptor`. It checks the constant-buffer bank, the offset relative to `nvn_buffer_base`, 16-byte alignment, and the guest slot's used bit. The helper is ported without changing its semantics. The new regression puts an unrelated-bank descriptor before guest slots 3 and 15, checks rejected offsets, and reverses the list to prove selection is independent of binding order.

The port includes the include and predicate changes from `22e0e41` in all three functions above. It keeps the binding index for selecting the actual host buffer. The new backend regression emits a global store with guest slots 3 and 15 at host bindings 1 and 2, then checks the SPIR-V storage access chains, GLSL address-selection code, and GLASM descriptor loads.

PR330 also changes `AddNVNStorageBuffers` in `frontend/maxwell/translate_program.cpp`: deduplication must compare both `cbuf_index` and `cbuf_offset`. Both nx2 baselines compare only the offset, so an existing descriptor from bank 1 can incorrectly prevent adding the NVN descriptor from bank 0. The port includes this change. Its regression translates a five-instruction Maxwell program through the full frontend and checks that both descriptors survive, with the source descriptor read-only and the NVN descriptor writable.

### Compatibility with upstream PR464

Upstream nx2 includes PR464 through `9e56ed7` and `560b4c0`. Its SPIR-V `DefineGlobalMemoryFunctions` first handles `info.native_global_bounds` by selecting the closest candidate base, then returns. Only the separate NVN branch uses `nvn_buffer_used`. Apply the guest-slot predicate to that NVN branch, never to the native-candidate loops: native pointers need not have NVN descriptor sizes or 16-byte-aligned descriptor offsets.

PR464 does not change the GLSL or GLASM predicates or the bank-blind `AddNVNStorageBuffers` deduplication. Those fixes still apply directly. At the pinned upstream revision `7adcecc6d2`, the memory-load search barrier also applies after `NativePointerCandidates` declines an address loaded from memory. Unlike the final merge, that revision lacks the native early-return guard.

An isolated source snapshot of `7adcecc6d2` was compiled with these selective ports. The same backend and frontend regressions passed. Additional checks prove that a native select between pointers at cbuf offsets `0x168` and `0x178` still binds and accesses both buffers despite an empty NVN used-slot mask. Loaded-pointer destinations remain global for both native and non-native environments. The combined upstream tests passed with **4 cases and 5,663 assertions**. On unpatched upstream consumers and tracking, three cases failed with fourteen assertion failures, while the native-candidate case passed. This checks the compiler output and optimization behavior, not GPU execution. These results cover `7adcecc6d2`, not the final merge `3483630`.

In the current merged working tree, SPIR-V's `native_global_bounds` branch still returns before the NVN loop. Only that later loop calls `UsesNvnStorageDescriptor`, preserving the native candidate-selection path.

## Geometry and conversion differences

`GraphicsEnvironment` in the fork already accepts native geometry shaders and passes the geometry-stage flag to `LowerToMaxwell`. PR363 enables the same stage by default, with `EDEN_NX2_EXPERIMENTAL_GEOMETRY=0` as an override. Upstream nx2 lacks that geometry-stage adapter. There is no missing default-enable change to port into the fork.

The lowering implementations are not interchangeable:

- PR363 lowers LEPC into the native GPU address and treats exit setup opcode `0x357` as a no-op. The fork recognizes a validated `AnalyzeGeometryExit` prologue and handler, removes only that envelope's setup, and redirects body exits to the handler. Its existing geometry-capture and geometry-exit regressions exercise these guarantees. Do not replace them with general acceptance of LEPC or arbitrary exit setup.
- Both the fork and PR363 turn the geometry ISBERD map form into a register move. The fork also validates geometry output and finalization modes, predicates, streams, and invocation registers. The existing `ampere_geometry_lowering.cpp` tests cover those paths.
- PR363's LDC widths 0 through 5, VOTEU destination expansion, ceiling F2I, signed-byte conversion, and unsigned 16-bit truncation already have local implementations. The fork has tighter reserved-bit validation and broader conversion handling. Its byte conversion clamps before conversion rather than adopting PR363's two-step narrowing sequence.
- PR363 accepts signed 16-bit truncation mode `0x0021e900`. The fork rejects it, and two existing rejection tests explicitly cover that choice. No Bananza capture supplied to this task demonstrates that this extra mode is needed. It is recorded as a coverage difference, not silently enabled or used to weaken the existing tests.
- PR363 enables software HMMA by default through `EDEN_NX2_TENSOR`. The fork already lowers HMMA without that opt-in and requests its compact matrix path in `shader_environment.cpp`. Replacing it with PR363's large expansion would discard local work.
- PR363 extends `IsGeneric` through `Generic31W`; the fork already uses that bound. Upstream nx2 still ends at `Generic31X`.

No changes were made to `shader_environment.cpp` or `lower_to_maxwell.h`. The geometry, conversion, and vote tests passed against their existing implementations.

## Indirect reset differs in validation, not just enablement

The fork's `TryNativeIndirectReset` in `vk_rasterizer.cpp` already runs by default. It matches the complete 44-word captured program from `native_indirect_reset.h`, requires a non-indexed single-instance point draw, and rejects mesh draws. It reads the shader and pointer list through safe `ReadBlock` calls, then writes `{0, 1, 0, 0, 0}` to each validated target.

PR363's `TryNx2IndirectReset` is default-on with an `EDEN_NX2_RESETFIX=0` override. `PipelineCache::IsIndirectResetShader` identifies it by native-program hash `0x72fd62555e7196cd`, source size `128 + 11 * 16`, and an enabled vertex-B stage. It reads the list descriptor's address and byte size at `c[0][0x110]`, while the fork reads only the address.

Concrete differences for the rasterizer owner:

- PR363 requires a 0x120-byte cbuf and bounds `first + count` by the descriptor's list byte size. The fork requires only 0x118 bytes and checks mapped memory plus arithmetic overflow. A mapped range can still exceed the descriptor. Porting this guard requires reading the size and updating `GetListRange` and its caller together.
- PR363 rejects null list addresses and checks both ends of GPU address ranges. The fork's helper has no null-list check and its caller relies on mapped-range checks.
- PR363 validates that each record is a single exact `GetSubmappedRange` segment before `AccelerateInlineToMemory`. The fork checks full mapping, overflow, and list overlap but not the single-segment condition.
- Both implementations validate all targets before writing. PR363 skips null target pointers. The fork rejects the whole host fast path if any target is null. Do not change that policy without evidence that skipping a null target matches the captured shader.
- PR363 explicitly flushes a dirty list before `ReadBlockUnsafe`. The fork uses safe `ReadBlock`, which already preserves the GPU-write flush. Do not replace that with an unflushed read.
- The fork's draw-shape checks are stricter than PR363's. Preserve them when porting validation rather than replacing the function wholesale.

The port adds the descriptor-size, null-list, GPU-range, and single-segment checks to the local reset path. `GetListRange` takes `list_size` and its caller reads both descriptor words. Exact program matching, draw-shape guards, null-target rejection, and safe reads are preserved. A new regression covers descriptor bounds. The bullets above describe the pinned baselines, not the updated working tree.

## Verification

On the final merge base `3483630`, Clang 20 and Ninja built both `lichii-cmd` and `tests` using all eight cloud cores. The complete suite passed: **479 cases and 15,741,117 assertions**. The focused storage-buffer, Ampere, reset and asynchronous-draw run passed **149 cases and 7,915,720 assertions**. The native and legacy loaded-pointer variants both pass.

The trace opt-in, rate-limit and timestamp-wrap tests passed in six separate processes with the flag unset, empty, `0`, `1`, `10` and `true`: 3 cases and 15 assertions per process. An extracted behavior check also compared 5,832 CPU conditional-render decisions against the merge baseline under each of those settings; all matched. Vulkan predicate-operation bodies were unchanged after removing trace calls.

Before/after evidence was collected before integration using the real shader pass, frontend and backends, not substituted shader logic:

- The loaded-pointer regression failed six assertions before the traversal barrier.
- The identical NVN consumer regressions failed ten assertions before the port. SPIR-V accessed binding 3 instead of 1 and 2, GLSL and GLASM selected the wrong descriptors, and the frontend retained one descriptor instead of two. All passed after the port.
- An isolated snapshot of pinned upstream nx2 with PR464 failed fourteen assertions before the selective ports and passed 4 cases with 5,663 assertions afterward. Its native-candidate test retained both 8-byte-aligned pointers without NVN used bits.

Those historical failures are not evidence of either reported gameplay symptom. The final merge already protects native loaded pointers as described above. No game or Windows Vulkan execution was tested.

Use the build configuration in [the diagnostic guide](bananza-diagnostics.md), then rerun:

```sh
ctest --test-dir build --output-on-failure
build/bin/tests '[shader][storage_buffer],[native_reset],[video_core][ampere],[async_draw]'
EDEN_DKB_FRAME_TRACE=1 build/bin/tests '[dkb_frame_trace]'
```

The upstream changes can be inspected without moving the shared checkout:

```sh
git show 22e0e41d6b -- src/shader_recompiler
git show c31a22e -- src/shader_recompiler/frontend/ampere/lower_to_maxwell.h src/video_core/shader_environment.cpp
git show 044bb10c45 -- src/video_core/shader_environment.cpp src/video_core/renderer_vulkan/vk_rasterizer.cpp
git show upstream/pr363:src/video_core/indirect_reset.h
git diff origin/nx2 upstream/pr363 -- src/shader_recompiler/frontend/ampere/lower_to_maxwell.h
```
