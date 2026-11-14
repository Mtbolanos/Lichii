# Upstream video playback audit

Audit date: 2026-11-04. Upstream is `Djihads80/Eglantine`, remote `upstream`.
Initial base: `origin/nx2` at `9a435b584a`. Audited upstream nx2: `a666e4400b`.
The final pre-push fetch still found that origin base and no
`merge/upstream-nx2-20261104` branch.
The complete search result list is in [the inventory](VideoPlaybackUpstreamInventory.md).

## Port decisions

| Source | Commit | Decision |
| --- | --- | --- |
| #444 | `97351edbfc` | Port per-stream H.264 framing and its four regression tests. Also reset framing on decoder Reset/Drain and use size_t for prefix advances. This PR is merged upstream but absent from the local base. |
| #391 | `10d62f6661` | Port the environment-variable shadowing fix without changing EOS fence policy. |
| #123 | `bde691ac7a` | Port parent-memory and image-offset presentation lookup. Adapt it to the fork's shared device ranges rather than reading the active Maxwell channel from the service thread. Keep handle metadata in the existing mapping list so remaps and partial unmaps cannot leave stale ranges. |
| #123 | `9332056123` | Apply collected CPU writes before downloading GPU data. |
| #123 | `1e6cfd5b4e` | Port the bounded synchronous-submit wait as an opt-in diagnostic. Do not enable the old Yoshi-specific default. This fork already snapshots submitted command words and has newer GPU-write synchronization. |
| #118 | `2d3b628b2f` | Port only queued-EOS ready-event suppression. Keep the later #356 fence policy and this fork's existing stop/finalize and Mario Kart synthetic-EOS behavior. Do not port immediate release-fence completion. |
| #472 | `33ebba968f` | Initially ported the eight-packet admission limit. Reverted after the Mario Tennis CPU trace showed 606 rejected packets with no retry in one small-video session. The queue now retains all submitted packets and uses eight only as the readiness watermark. |
| #24 | `896af9f676`, `71630ffba8`, `bcaeef479d` | Decoder sessions, syncpoint cancellation, and the FFmpeg service implementation already exist. |
| #356 | `8d6cb5c0d4`, `701c91b402`, `3ffa971337`, `031e58e930`, `873cae5569` | VP9, H.264 detection, GPU invalidation, default pending EOS fences, and direct NV12 copies already exist. |
| #218 | `6f33896d8c`, `959973a76a` | AV1 codec and MSVC/Linux AV1 package selection already exist. |
| `fix/nx2-video-newer-sdk` | `1b726d93c5` | Commands 905, 201, 202, 210, 400 and descriptor-dependent input word offsets already exist. |
| `fix/nx2-gsv-912-decoder-image` | `574517814d` | Decoder image destruction already exists, including the fork's decoder-mapping cleanup. |
| #380 | `ea34492856` | Not ported. The PR explicitly says its MinGW selection requests an unpublished mingw-amd64 package. The release API did not resolve the release during this audit, so current availability was not independently established. An MSVC library is not a MinGW substitute. |
| #27, #149, #166, #455 | Bring-up and bulk integration branches | Do not import broad branches or their older decoder/fence implementations over newer local code. |
| #123 | Other device/shader changes | Device creation and decoder detection are already covered locally. General shader lowerings and the unrelated invalid-texel-buffer fallback are outside this decoder port. The author reports that the story intro still decodes black. |

These changes do not prove that the reported Yoshi or Kirby crash is fixed.
No game dump, save, or owner crash log was available for a local reproduction.
The new trace separates decoding, image writes, presentation, and skip/EOS handling.

## Verification evidence

The original packet converter fails 268 of 1,070 assertions from #444's four tests.
The port plus reset coverage passes 1,073 assertions across five tests in a standalone Catch2 build.

A generated ten-second, 640x360, 30 FPS H.264 MP4 with repeated parameter sets provides a separate FFmpeg check.
The original converter produces 268 of 300 frames and 32 rejected packets.
The port produces 300 of 300 frames, zero rejected packets, and identical per-frame pixel hashes to FFmpeg's normal MP4 decode.
Annex-B input also produces 300 matching frames without rejected packets.

Three registry tests cover parent offsets, partial unmaps, remaps, and session reset.
The full clang 20/Ninja RelWithDebInfo build passes with all eight available cores.
CTest passes all 479 cases and 15,726,880 assertions. The focused H.264 and mapping
selection passes eight cases and 1,089 assertions.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_QT=OFF -DLICHII_CMD=ON -DLICHII_TESTS=ON -DENABLE_WERROR=OFF
cmake --build build --parallel "$(nproc)"
ctest --test-dir build --output-on-failure -V
build/bin/tests '[h264],[nx2_mapping]'
```

The build used CMake 3.31.10 and system FFmpeg 6.1 with libdav1d. Clang 18 rejects
the existing `-mtls-dialect=gnu2` flag, so verification used clang 20 instead.
Existing `-Wno-error-all` warnings remain. The CLI's `--version` smoke test exits
successfully, though it emits empty-directory configuration diagnostics on this
fresh machine. No game-playback validation is implied by that smoke test.

## Related issue evidence

- #362 is the framing regression addressed by #444.
- #311 and #407 report AV1 cutscenes skipped in MinGW/PGO builds while MSVC works.
- #409 includes movie audio stutter and a later AV1 report where decoding advances faster than presentation. It does not establish that decoder pacing causes the audio problem.
- #457 reports a successful intro followed by a white screen after L+R, not a demonstrated decoder failure.

Diagnostic instructions are in [Debug video playback](VideoPlaybackDebug.md).
