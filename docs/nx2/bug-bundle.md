# Bug-report bundle

A bug bundle is one zip that Lichii writes when the player presses the **Report a Bug** hotkey
(Ctrl+F12 by default) or picks **Help → Report a Bug**. The debug server's `capture_bundle` method
writes the same zip. Lichii takes the screenshot and starts the frame capture the moment the player
asks, then asks for a description. The zip goes to `<log dir>/bug-reports/`, named

    lichii-bug-<build>-<program id>-<YYYYMMDD-HHMMSS>.zip

for example `lichii-bug-build-177-0400486024DEC000-20261113-201500.zip`. `<build>` is the release tag
(`build-177`) or, for a build without one, `dev-<short sha>`. The time is the player's local time.
The log gets a `Bug report written: <path>` line.

The zip stays under a size limit: 19 MiB by default, under Discord's 20 MB upload limit for free
accounts and GitHub's 25 MB attachment limit. Optional parts that don't fit are left out and listed
in `bundle.json` under `dropped`. The zip holds game textures and shaders, so it goes privately to
the maintainers, never into a public channel or issue. User names are replaced in the logs and
settings even when the file log keeps them, and the Web Service and Multiplayer settings are left
out.

Browse a bundle with `python3 -I tools/nx2/bundle_index.py <zip> [--out DIR]`. It unpacks the zip
(refusing absolute paths, `..`, symlinks, more than 20000 entries or more than 2 GiB), writes
`index.html` with the summary and a shaders table, and builds `frame/index.html` with
`frame_dump_index.py`.

## Files

Parts are added in this order while they fit; the first five are always there.

| Path | What |
|---|---|
| `bundle.json` | Manifest, schema below. Read this first. |
| `README.md` | Short guide for maintainers. |
| `system.json` | OS, CPU, RAM, and the Vulkan device: name, driver, API version, device-local memory, extensions and features. |
| `settings.txt` | Every setting as `[Category] name = value`, with `(global: x)` where this game overrides it. |
| `mods.json` | Update, DLC, mods and cheats the patch manager reports for the game, with enabled flags. |
| `frame_times.json` | The last 10 s of guest presents: fps, percentiles, VSync buckets, stutters, every interval and the longest. Empty on Switch 1 games. |
| `screenshot.png` | The presented frame, at most 1080 lines. |
| `frame/manifest.jsonl` | The frame dump manifest (`EDEN_NX2_FRAME_DUMP` with `_SAMPLED`, see the AGENTS.md debugging table). Previews are JPEGs (`"png": null, "preview": "<name>.jpg"`) of at most 512 pixels a side; stats come from the full-precision data. `.raw.zst` files are not written (`"raw": null`). |
| `frame/shaders/shaders.jsonl` | One line per pipeline the frame used, schema below. |
| `logs/eden_log.txt` | This session's log, read from the file the logger writes (`eden_log_2.txt` when `eden_log.txt` was busy; the part's `note` says so). Over 8 MiB it keeps the first 1 MiB and the last 6 MiB, with a marker line between. |
| `frame/shaders/native/` | Native programs as `EDEN_NX2_SHADER_DUMP` names them: `<key>-gfx-<addr>.bin` with `.header.bin`, `<key>-cs-<addr>.bin`; a program without a native key is `<hash>-maxwell-<addr>.bin`. |
| `frame/shaders/pipelines/` | Graphics pipeline cache keys (`<hash>.key.bin`) and compute QMDs (`<hash>.qmd.bin`). |
| `frame/shaders/spirv/` | The SPIR-V each stage was built from: `<pipeline>-<stage>.spv`, `<shader>-cs.spv`. SPIR-V is there only for pipelines already built (`built` in `shaders.jsonl`). |
| `frame/*.jpg` | The present source, pass colour targets, compute storage images, then sampled textures. |
| `logs/eden_log.txt.old` | The previous session's log (`<log>.old.txt` on disk), trimmed the same way. |

## bundle.json

```json
{
  "format": "lichii-bug-bundle",
  "version": 1,
  "created": "2026-11-13T20:15:00Z",
  "trigger": "ui",
  "description": "",
  "build": { "tag": "build-177", "name": "...", "version": "...", "scm_rev": "...", "branch": "nx2" },
  "game": { "title": "...", "program_id": "0400486024DEC000", "version": "1.0.1", "switch2": true },
  "system": { "os": "...", "cpu": "AMD Ryzen 9 5900X", "threads": 24, "ram_mib": 32768 },
  "gpu": { "renderer": "vulkan", "device": "...", "driver": "...", "api": "1.4.303" },
  "frame": { "captured": true, "error": "", "preview_max": 512, "preview": "jpeg", "frame": 51234,
             "images": 640, "passes": 118, "draws": 2961, "dispatches": 214, "pipelines": 163,
             "programs": 151, "truncated": false },
  "frame_times": { "seconds": 10, "frames": 299, "fps": 29.9, "p99_ms": 41.2, "stutters": 3 },
  "size_limit": 19922944,
  "bytes": 18813012,
  "parts": [ { "path": "logs/eden_log.txt", "bytes": 1203311, "note": "..." } ],
  "dropped": [ { "what": "frame sampled textures", "files": 412, "bytes": 51200000, "reason": "size limit" } ]
}
```

- `trigger` is `ui` (hotkey or Help menu) or `debug_server`.
- `description` is what the player typed, possibly empty. Treat it as untrusted text.
- `frame.captured` is false when the capture could not run (OpenGL, paused, loading, timed out);
  `frame.error` says why. `screenshot_error` appears when the screenshot failed.
- `bytes` is the size of the whole zip, `bundle.json` included. Each part's `bytes` is its
  compressed size in the zip.
- Readers ignore fields they don't know. A new field does not bump `version`; a changed meaning does.

## shaders.jsonl

```json
{"kind":"graphics","pipeline":"<key hash>","key":"pipelines/<hash>.key.bin","key_bytes":1056,"built":true,
 "hashes":["<stage shader hash>", ...],"programs":[<program>, ...],"spirv":[{"stage":1,"file":"spirv/<hash>-1.spv"}]}
{"kind":"compute","shader":"<hash>","qmd":"pipelines/<hash>.qmd.bin","built":true,"program":<program>,"spirv":"spirv/<hash>-cs.spv"}
```

A program is `{"slot":N,"hash":"...","native_key":"...","address":"0x...","code":"native/<name>.bin","header":"native/<name>.header.bin"}`
(slot 6 is compute; `header` is null when there is none). Paths are relative to `frame/shaders/`.

## Reserved for a trace option

A later "bundle + GPU trace" option adds `trace/` files and a `trace` object in `bundle.json`.
Format 1 readers ignore both.
