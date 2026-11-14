# NVIDIA DLSS presets and quality

On Windows, enable NVIDIA DLSS in Graphics, then select a quality mode and model
preset. Apply or OK updates these two choices while playing. The master DLSS
checkbox still takes effect when a game starts. These controls apply only to the
supported game-upscaler replacements, not every game or the window scaling filter.

## Checking that DLSS runs

DLSS is the "Use NVIDIA DLSS in place of a game's DLSS" checkbox, not a window
adapting filter. The filter button in the status bar (FSR 1, Bilinear and so on)
is a separate step that scales the finished image to the window. While DLSS
replaces a game's upscaler, the status bar and the performance overlay show
**DLSS**, the mode it runs in and the preset, for example `DLSS DLAA K`. When an
explicit mode doesn't fit the game's render size, they add the mode you chose,
for example `DLSS QUALITY K (PERFORMANCE DOESN'T FIT)`. Nothing DLSS-related is
shown when the game never calls its upscaler, or after DLSS has failed; a failure
keeps DLSS off until the game restarts, and the log says why.

## Model presets

Fritchii defaults to **K** for every quality mode (NVIDIA's own defaults for
Performance and Ultra Performance drew thin lines in Fire Emblem at 2x). The
choices are NVIDIA's default, J/K (DLSS 4 transformer models) and L/M (DLSS 4.5
transformer models). EDEN_NX2_DLSS_PRESET=J|K|L|M|default still overrides the setting.

[NVIDIA recommends](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/)
K for DLAA, Quality and Balanced, M for Performance, and L for Ultra Performance.
L/M also work with the other quality modes, but cost more, particularly on RTX 20
and 30 GPUs without native FP8 support. K is a useful alternative on those cards.
J can reduce ghosting compared with K at the expense of more flicker. If Fire
Emblem: Fortune's Weave shows horizontal lines with L/M at 2x, try K.

No NVIDIA DLL ships with Lichii. The Graphics download button obtains Streamline
2.14.16 and DLSS 310.9.16 from NVIDIA after license acceptance. That runtime supports
all the listed presets. L/M need DLSS 310.5 or newer; replace an older locally
installed runtime with NVIDIA's current files. The panel reports the discovered
versions. NVIDIA app overrides and downloaded models can supersede the requested
preset, so the emulator log reports the request, not proof of the model executed.

## Quality modes

- **Automatic** selects a mode supported by NVIDIA for the game's actual render
  and output sizes. This preserves the existing upscaling behavior.
- **Native (DLAA)** applies DLAA at the game's render resolution, then spatially
  scales that image to the guest destination if needed. It does not increase the
  game's render resolution to the display resolution. Ocarina's 1440p mode stays
  at 2560×1440 instead of reconstructing to 4K.
- **Quality, Balanced, Performance and Ultra Performance** request that NGX mode
  without changing guest rendering. The request is used only when both render
  dimensions fit NVIDIA's queried dynamic-resolution range. Otherwise Automatic
  chooses a compatible mode and the log records the request, actual mode and
  allowed size range. If no mode accepts the sizes, the existing FSR fallback runs.

Ultra Quality is not offered merely because an SDK enum exists; this integration
does not establish runtime support for it. A Performance request is not a promise
of higher frame rates, because it cannot lower the game's own rendering resolution.

Old profiles without these settings load Automatic and M. The Automatic/Native
values remain compatible with the quality settings in
[myshadowww's PR #840](https://github.com/zortos293/Lichii/pull/840), from which the
Native DLAA sizing and atomic runtime-settings publication were adapted. This
change does not import that PR's Linux backend or neural rendering.
