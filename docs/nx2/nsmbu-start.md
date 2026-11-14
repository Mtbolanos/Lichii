# New Super Mario Bros. U Deluxe: black screen after Start

Game: `0100EA80032EA000`, version 1.0.0, 32-bit. Base: Lichii `649c59d3f23cc6f2eefc1e1f00abe591f97956b7`.

Without installed system archive `0100000000000802`, the MiiModel fallback supplied six 16-byte headers with no texture or shape parts. After L+R, the game read addresses zero and `0x14` and attempted to execute address zero. The captured guest stack included RedPro-S offsets `0x847884`, `0x848068`, `0x83f948`, and `0x83db94`; the screen went black.

The fix ports the generated blank textures and degenerate shapes from [ScarletNX PR 863](https://github.com/scarletnx/ScarletNX/pull/863), merge `05324b57629d0894506b31beb2c72869138b7ad1`. It supplies all part entries with valid sizes, alignments, and compressed payloads. It contains no firmware assets. Installed system archives continue to take precedence. Without those archives, Mii faces remain blank.

## Validation

All runs used a separate portable profile on Windows, Vulkan, and the same game. The input sequence held L+R for 30 frames, released for 180, held L+R for 30, and released for 120.

- Baseline without firmware: black screen and guest null-memory fault.
- Same baseline executable with the owner's existing local firmware copied into the isolated profile: Story Mode save selection.
- Patched executable with the original no-firmware profile restored: Story Mode save selection, character selection, opening sequence, Acorn Plains world map, and the first level; no corresponding guest fault.
- New Mii regression tests fail on the original headers. After the fix, the Mii and debug-server tests pass: 20 cases, 62,289 assertions. Tests independently decompress every generated part with zlib and check its layout and bounds.
- Native clang-cl 22.1.8 build of `tests` and `lichii-cmd` passed with 12 build jobs. The 17 Python debug-adapter tests also passed. Qt tests were unavailable in the bundled Qt package.

The automated runtime comparison used the separately submitted Switch 1 debug-frame counter fix. The Mii fallback change does not depend on it.

Private logs, screenshots, binaries, and build scripts remain outside the source changes.
