# Star Fox spatial display workaround

This extends the guarded Fire Emblem workaround to the captured Star Fox DLSS
passes. Desktop testing confirms that it removes black rectangles in the opening
cutscene for the captured docked prepass. It does not repair the native temporal
network; later missions and motion-vector conventions remain unverified.

## Evidence

The `b980b75d67` log boots Star Fox (`0400E88025B1A000`) at 21.823 seconds.
The accompanying native compute dumps contain a tensor network, including a
prepass and final pass with the same relevant resource layout as Fire Emblem.
NVIDIA nvdisasm 12.8.90, using SM87, decodes these accesses:

| Capture key | Instruction offset | Access |
| --- | --- | --- |
| `d686ca0b9fad9e6d` | `0x3140` | `TLD` colour descriptor `0x9e`, cbuf0 byte `0x278` |
| `d686ca0b9fad9e6d` | `0x00e0`, `0x00f0` | Add cbuf0 `0x1b0`, `0x1b4` to input-pixel sampling positions |
| `7d3764ff101bec27` | `0x1c40` | `TLD` colour descriptor `0xca`, cbuf0 byte `0x328` |
| `7d3764ff101bec27` | `0x7e60` | `SUST` output descriptor `0xd2`, cbuf0 byte `0x348` |
| `7d3764ff101bec27` | `0x7d10` | `SUST` secondary output descriptor `0xd4`, cbuf0 byte `0x350` |

The dumps' filenames identify their exact `NativeProgramKey`. The prepass contains
zero-filled padding; replace only zero instructions with NOP in a scratch copy
when disassembling it, without changing the source used to calculate the key.

Upstream `401dfb9329` recognizes this family by resource slots rather than title.
Lichii deliberately retained the stricter Fire Emblem title/fingerprint gate when
merging it. That gate excludes Star Fox despite the matching captured accesses.
This change extends the strict recognition instead of reinstating the generic
heuristic. It reuses the spatial de-jitter path from `b9d6b4a71e`; motion-vector
conventions and the native network's numerical correctness remain unverified.

## Safety and behavior

Only Star Fox and the two exact keys select the new route. Every native dispatch
still runs, including the final pass, so its secondary output and other writes
are preserved. Afterward, the spatial replacement overwrites only the validated
display output. It does not read previous-frame history or motion vectors.

The prepass must supply finite jitter within half an input pixel. Its input
image must match the final pass's input address and dimensions. Both replacement
images must pass the existing mapped-descriptor, format, layout, layer, mip and
size checks. Missing or inconsistent inputs leave the native result untouched.
Samples are consumed once, and incomplete chains expire after 64 dispatches.
Fire Emblem's existing behavior is unchanged.

## Native-resolution cutscene capture

The next log, from `2c1314aa04`, reaches both of the same fingerprints twelve
times in its bounded trace. At 46.951708 seconds it reports
`Star Fox replacement lacks current inputs: jitter=false`. There is no
`Star Fox DLSS replacement: applied` message anywhere in that log. The reported
gameplay improvement cannot be attributed to an applied spatial replacement.

The rejected contract is the colour format and extent, not a different shader
variant or different slots. The input at `0x278`/`0x328` is a 1920x1080 RGBA8
UNORM image with sRGB sampling; the output at `0x348` is a different 1920x1080
RGBA16-float image. The old Fire Emblem checks require B10G11R11 at both ends
and a strictly larger output. The first check discards the otherwise valid
captured jitter, and the output and Vulkan format checks would also reject it.

Star Fox now accepts only this logged, equal-resolution format pair, with
identity RGBA swizzles and the recorded component types. Mipmapped, multisampled,
sparse, layered, aliasing or unequal-sized images are rejected. The source is
bound through an sRGB view, so sampling decodes it to linear colour before writing
the float output. The spatial and sharpening passes write only RGB, preserving
the alpha produced by the native pass and its separate output at `0x350`.
Fire Emblem keeps its original format and upscale restrictions. The colour-pair
change itself added no shader fingerprints or slot-search heuristics.

## Docked cutscene prepass validation

A November 11 desktop capture of Star Fox (`0400E88025B1A000`, main build
`A7730D0F74D168D90422354C8317F76FF78AD938`) uses prepass key
`ebab31dd6f0fc08d`. The old `d686ca0b9fad9e6d` gate therefore never opened the
replacement chain. Both keys now identify the same guarded prepass; the final
key remains `7d3764ff101bec27`.

Native disassembly confirms the existing resource contract: jitter loads at
instructions `0xe0` and `0xf0` read `cbuf0+0x1b0` and `+0x1b4`, while `TLD` at
`0x3140` reads colour through `cbuf0+0x278`. Captured jitter is
`(-0.4375, 0.388888955)`. The final pass reads that same 1920x1080 sRGB colour
image through `+0x328` and writes equal-sized RGBA16F output at `+0x348`, with
separate history at `+0x350`.

A sampled frame dump locates clean colour before the native temporal network and
black rectangles in its output. RenderDoc 1.46 independently confirms the chain
in a successfully replayed Vulkan capture: event 3215, translated shader
`25a620c965883cde`, produces the first corrupt output; event 3279
(`bccc6a99e0f227fa`) retains it; event 3308 (`a2884a1f4cd841e7`) enlarges it
into the final 4K image. Captures remain local.

On a desktop integration based on `e860139f53`, the patched contract trace reports
`Star Fox DLSS replacement: applied temporal=false`. Uncaptured room and cockpit
screenshots are free of the reproduced black rectangles. A later frame dump
also shows the native result repaired before downstream tone mapping. This
integration included separate audio and async-draw fixes; the native network,
secondary writes and output alpha still execute unchanged.

The regression checks both captured keys, every one-bit key mutation, the output
key, and Fire Emblem separation. This is a display repair, not an average-FPS
improvement. A possible trail in another moving-ship screenshot remains
unclassified; its exact frame was not captured, and later missions were not tested.

## Owner-side comparison

Start a fresh process with `EDEN_NX2_DLSS_CONTRACT_TRACE=1` and log filter
`*:Info HW.GPU:Debug Shader:Debug Render.Vulkan:Debug`.
Leave `EDEN_NX2_DLSS_REPLACEMENT` unset for the spatial workaround. Star Fox always
uses the spatial path, even if `fsr` or `dlss` is requested; its temporal resource
contract has not been validated. Set `EDEN_NX2_DLSS_REPLACEMENT=native` in a second
fresh process to retain the original output for comparison.

`Star Fox DLSS replacement: applied temporal=false` confirms application and logs
the actual image sizes, addresses and jitter. The bounded contract trace records
the relevant constant-buffer words and raw image descriptors, including the
secondary output. A missing application message is not evidence that the
workaround succeeded. Compare screenshots from the same camera motion and save.

The log also rejects a separate `SURED.D.3D.ADD` compute program at 117.963 seconds.
That is a real unsupported shader form, but the evidence does not connect it to
the temporal trails. This change does not hide or alter that rejection.
