# Bananza global atomic-add fix

The new profile identifies the missing operation as `GlobalAtomicIAdd32`. The native pointer pass deliberately retains this operation when its address can select between multiple buffers, but the SPIR-V emitter still throws instead of performing the atomic. This patch implements that one observed operation through the existing runtime buffer-selection code.

This fixes the established compiler gap. It does not establish that Bananza's black screen or frame-rate symptoms are resolved. No game was run during this work.

## Evidence and provenance

The source baseline is `ba2d291`, and the owner log is `~/.capy/work/dkb/profile-run.txt`. All nine distinct shader hashes with a compilation failure identify `GlobalAtomicIAdd32`:

```text
ceafb5fe9c1f007b  f92e1c9a711e8b7e  c7a1778ec38f628
eea4e264638313e5  9e5b0a839f3165bc  f523d0cf1b8025f8
e092e96153f6a627  655355bd2e34448c  4b6d62a0471c7106
```

The first occurs at 43.330675 seconds, log line 1181. The previous diagnostic patch now distinguishes this failure from the other SPIR-V stubs, so an implementation is justified rather than another speculative opcode expansion.

In `global_memory_to_storage_buffer_pass.cpp`, `CollectStorageBuffers` binds every candidate when `NativePointerCandidates` finds more than one, marks written descriptors, sets `native_runtime_bounds`, and returns without replacing the atomic. `GlobalMemoryToStorageBufferPass` transfers that state to `program.info.native_global_bounds`. Choosing one descriptor at compile time would lose writes from the other lanes or control-flow paths.

Upstream was refreshed before implementation:

- `upstream/nx2` was `d3a659982fff030e3e57ce15495765064d72e730`. Its `EmitGlobalAtomicIAdd32` is still the throwing stub.
- PR306 was refreshed to `770f3052bc7903be6e25baa9106ff93fa0b93f30`; its additional change concerns LEPC register handling, not global atomics. Its atomic emitter is also still a stub.
- The open PR list and repository issue/PR searches for BDA, physical-storage addressing, buffer device address, atomics, and `GlobalAtomicIAdd32` did not identify an applicable physical-storage atomic implementation. PR330 remains an address-tracking/NVN-slot fix, not a general guest-address mapping implementation. The BDA discussion in issue 164 describes capture limitations, not an atomic implementation that can be ported.
- Earlier local commit `62bc7538b119b3b5a8105ce9195b04ef626ec4ae` already contains descriptor-routed global atomics. This patch adapts only its IAdd operation pattern: device-scope `OpAtomicIAdd`, relaxed ordering, return the old value, and reject missing 64-bit addressing. It does not import its other atomic operations or replace the merged pointer-selection logic.

No guest GPU address is treated as a host Vulkan device address. Physical-storage atomics would require a valid guest-to-host address mapping and device-feature integration; casting the logged addresses to host pointers would be incorrect.

## Implementation

`EmitGlobalAtomicIAdd32` now accepts the guest address and increment. It calls a generated helper returning a 32-bit value. `DefineGlobalMemoryFunctions` emits that helper only when the program still contains `GlobalAtomicIAdd32`; ordinary global loads do not gain unused atomic functions.

The helper reuses `define_body` without changing either address-selection branch:

- For `native_global_bounds`, the existing code selects the closest aligned candidate base at or below the full 64-bit guest address. Native 8-byte parameter slots remain valid and do not need an NVN size word or used-slot bit.
- For the NVN fallback, the existing guest-slot predicate and descriptor size bounds choose the storage buffer. Unrelated descriptors and compact binding positions are not substituted for guest slots.

Once selected, the helper issues `OpAtomicIAdd` against that storage-buffer word with device scope and relaxed memory semantics, matching the existing storage-atomic path. It returns the atomic instruction's result directly, which is the value before the addition. It does not replace the operation with separate loads and stores or return the incremented value. The 32-bit addition wraps modulo 2^32.

Programs with no routable buffer still fail explicitly. No helper is generated if the descriptor list is empty, or if the NVN path has no descriptor matching a used guest slot. The emitter then throws `Global atomic add with no routable storage buffer`, with the existing IR-opcode annotation and cached native failure propagation. Missing 64-bit integer support also remains an explicit rejection. Failed dispatch is not converted into successful no-op compilation.

There is a separate runtime case: an otherwise routable program can supply an address outside all eligible ranges. For that case the helper follows the existing global-load fallback convention and returns zero without writing. This is not a claim that arbitrary guest addresses are mapped. The native selection policy and bound-buffer behavior are inherited from the existing global-memory implementation, not expanded by this patch.

Only four SPIR-V backend files change in production: the atomic emitter, its declaration, the emit context, and its header. The optimization pass, descriptors, renderer, cache policy, and other unsupported atomic opcodes are unchanged.

## Regression and Vulkan verification

The regression is in the already-registered `src/tests/video_core/global_memory_coherence.cpp`. Before the patch, a real native two-candidate IR program retained `GlobalAtomicIAdd32` correctly, then failed with the same unsupported-opcode exception as the owner log.

The tests now check:

- Both native candidates survive optimization, remain writable, and use offsets `0x168` and `0x178` with no NVN used-slot bits.
- Sparse NVN slots 3 and 15 route to their actual host bindings despite an unrelated descriptor before them.
- Every emitted atomic uses device scope and relaxed ordering, and the atomic result feeds `OpReturnValue` directly.
- Non-atomic global programs emit no atomic helper.
- Missing 64-bit addressing, empty native/non-native descriptor lists, foreign-bank descriptors, and unused NVN slots produce explicit failures rather than silent no-ops.

The focused Clang 20 runner rebuilt the SPIR-V backend with two jobs and passed **13 test cases with 20,935 assertions**, including adjacent coherence, NVN, opcode-diagnostic, and bitfield coverage.

Six actual emitted modules passed:

```sh
spirv-val --target-env vulkan1.2 --uniform-buffer-standard-layout module.spv
```

The uniform-buffer layout option matches the existing descriptor-aliasing profile. The test Vulkan device enabled `shaderInt64` and `uniformBufferStandardLayout`; no buffer-device-address feature was used.

The same emitted modules were executed on Mesa llvmpipe, LLVM 20.1.2, through Vulkan. Each dispatch ran 256 invocations across eight workgroups, with lanes alternating between two counters. The native and NVN variants both passed normal addition, overflow, and unmatched-address cases. The harness checked all returned old values, final counters, and unchanged neighboring words.

| Scenario | Native and NVN results |
| --- | --- |
| Counters initially 7 and 101 | Final values 135 and 229; each counter returned all 128 prior values exactly once. |
| Counters initially `0xfffffff0` and `0xfffffffc` | Final values 112 and 124; returned values wrapped correctly modulo 2^32. |
| Addresses below every eligible base | Counters remained 7 and 101; all returns were zero and guard words stayed unchanged. |

The synthetic guest bases have different high words, `0x12fffffff8` and `0x1300002000`. The first also has alignment padding, so the tests exercise 64-bit selection and correct rebasing rather than only zero-based offsets.

The Vulkan harness reuses the committed IR fixture. Its source, build script, emitted modules, and logs are under `~/.capy/work/bananza-compute-tests`, with `atomic-runtime.cpp`, `build-atomic-runtime.py`, `modules/`, and `runtime.log`. This is isolated compiler/GPU verification, not a Bananza gameplay run or an NVIDIA hardware test. The owner must still test the integrated build; other shader gaps or addresses outside the available descriptors can remain.

### Integrated release-build verification

The first integrated run failed the no-route regression despite the passing focused runner. This was an edit/build race, not a different release-mode policy. The emit-context source gained its no-route guard at 00:39:55, while an already-running compilation finished at 00:39:59. The resulting object still generated an empty atomic helper, and its newer timestamp kept Ninja from rebuilding it. Recompiling only the atomic emitter could not replace that helper generator.

A fresh emit-context object compiled with the exact integrated `-O2` flags, substituted into the same test link, passed all four global-atomic cases. Then the source timestamp was refreshed and the integrated emit-context object, shader library, and test executable were rebuilt. No production code or test assertion was changed to obtain the pass.

The rebuilt integration tree, which also contained pending descriptor and pacing changes, passed **500 test cases and 15,791,309 assertions**. Its focused atomic filter passed **four cases and 13,718 assertions**, including all no-route cases.

Finally, the Vulkan harness was relinked against the actual rebuilt `build/src/shader_recompiler/libshader_recompiler.a`, with no isolated backend object overrides. Its six emitted modules passed SPIR-V validation, and all six native/NVN contention, overflow, and unmatched-address scenarios passed again. These final-library logs and modules are under `~/.capy/work/bananza-release-check`.

For the standalone atomic commit, the pending descriptor and pacing changes were saved separately. Both `lichii-cmd` and `tests` were rebuilt with Clang 20 and Ninja using all eight available cores. That atomic-only tree passed **490 cases and 15,766,625 assertions**, plus the same four focused atomic cases. Its build and test logs are under `~/.capy/work/dkb/integration/atomic-only-*`. No game was run.
