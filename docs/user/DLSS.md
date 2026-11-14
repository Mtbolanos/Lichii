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
**DLSS**, the mode it runs in and the preset, for example `DLSS DLAA M`. When an
explicit mode doesn't fit the game's render size, they add the mode you chose,
for example `DLSS QUALITY M (PERFORMANCE DOESN'T FIT)`. Nothing DLSS-related is
shown when the game never calls its upscaler, or after DLSS has failed; a failure
keeps DLSS off until the game restarts, and the log says why.

## Model presets

Lichii defaults to **M** for every quality mode. The available choices are NVIDIA
default, E/F, J/K and L/M. E/F are deprecated CNN models, J/K are DLSS 4 transformer
models, and L/M are DLSS 4.5 transformer models. Removed and reserved presets are
not offered.

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
- **Quality, Balanced, Performance and Ultra Performance** request that NGX mode.
  The request is used only when both render dimensions fit NVIDIA's queried
  dynamic-resolution range. Otherwise Automatic chooses a compatible mode and the
  log records the request, actual mode and allowed size range. If no mode accepts
  the sizes, the existing FSR fallback runs.

DLSS replaces a game's own upscaler, so it takes whatever resolution the game
renders. In most games these modes therefore don't lower the rendering
resolution, and a Performance request is not a promise of higher frame rates.

The Legend of Zelda: Ocarina of Time is the exception. Docked with the Max
Resolution add-on, it renders at 2560×1440 and DLSS reconstructs 4K, which is
NVIDIA's Quality ratio. Balanced, Performance and Ultra Performance patch the
game's fixed render scale so it renders at 2276×1280, 1920×1080 and 1280×720
(ProfessorTee's [#903](https://github.com/zortos293/Lichii/pull/903), with
QuincyForbes' follow-up for the Balanced and Performance upscale passes). Automatic and Quality keep 2560×1440. The render scale is
patched into the game when it starts, so a change takes effect at the next
start; until then DLSS applies the new mode to the old render size. Handheld,
DLSS off and a disabled Max Resolution add-on leave the game's scale alone. If
DLSS turns out to be unavailable when the game starts (no NVIDIA runtime, another
GPU), FSR 3.1 reconstructs the smaller scene instead of the game's own upscale;
at Balanced, whose padded scene only DLSS takes, the game's own upscale runs.

Ultra Quality is not offered merely because an SDK enum exists; this integration
does not establish runtime support for it.

Old profiles without these settings load Automatic and M. The Automatic/Native
values remain compatible with the quality settings in
[myshadowww's PR #840](https://github.com/zortos293/Lichii/pull/840), from which the
Native DLAA sizing and atomic runtime-settings publication were adapted. This
change does not import that PR's Linux backend or neural rendering.
