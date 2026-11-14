# Renaming inherited yuzu names

Lichii inherits code from yuzu through Eden and Eglantine. The yuzu project name is replaced
with Lichii in code, build files, paths, user-visible text and documentation. Copyright and
licence notices stay exactly as they are, and so does anything that existing user data,
caches or build directories depend on.

The rename is a codemod, `tools/rename/rename.py`, so it can be run again after an upstream
sync. Upstream Eglantine still uses the yuzu names.

## Running it

```sh
python3 tools/rename/rename.py apply           # every area
python3 tools/rename/rename.py apply --area build
python3 tools/rename/rename.py check -v        # what still says yuzu, by class
python3 tools/rename/rename.py check --strict  # fails on anything not classified
```

`apply` moves tracked paths with `git mv` and rewrites text line by line. Both are idempotent:
a second run changes nothing.

Each file in `tools/rename/areas/` is one area and holds three lists:

- `PATHS`: `(regex, replacement, options)` applied to tracked paths.
- `TEXT`: `(regex, replacement, options)` applied to each line. `options` may set `files`
  and `exclude` (fnmatch globs) and `in_urls` (rules skip URLs unless it is set).
- `KEEP`: `(regex, class, reason[, files])`. Text a KEEP pattern matches is never rewritten,
  whatever area the rule comes from, and `check` reports it under its class.

Never rewritten, by any rule: lines with `SPDX-FileCopyrightText` or a `.reuse/dep5`
copyright, licence files, and `externals/` (only the build option names in our own CMake glue
there are renamed).

## Classes

- **attribution**: copyright notices and statements of where the code comes from. Removing
  them would break the GPL.
- **compat**: on-disk names, file magics, protocol strings and old option names that existing
  installs depend on. Lichii keeps reading them.
- **removed**: text that was deleted rather than renamed. A KEEP pattern stops `apply` from
  rewriting it if an upstream sync brings it back, and `check --strict` fails on it, with the
  reason saying what to delete.
- **pending**: anything else; `check --strict` fails while any remain.

## Build options

The CMake options are `LICHII_*` (`LICHII_TESTS`, `LICHII_CMD`, `LICHII_USE_BUNDLED_QT`, ...).
`CMakeModules/LegacyOptions.cmake` carries an old `-DYUZU_*` value, from the command line or
an existing cache, to the new name and removes the old entry, so existing build directories
and scripts keep their settings. The targets are `lichii` (output `lichii`), `lichii-cmd`
(output `eden-cli`), `lichii-room` and `lichii_room_standalone` (output `eden-room`).

## After an upstream sync

1. Merge upstream. Upstream files under `src/yuzu`, `src/yuzu_cmd` and the other old paths
   are renamed here, so git reports rename conflicts for files both sides changed: take
   upstream's change into the renamed file.
2. Run `python3 tools/rename/rename.py apply`, then `check --strict`.
3. Classify anything new that `check` lists: add a TEXT rule to the area that owns it, or a
   KEEP pattern when the name is attribution or compatibility data.
4. Build and commit the result as its own commit.

A branch that was opened before the rename can do the same: run `apply` on the branch, commit,
then merge nx2.

## Naming table

### Paths

| Old | New |
|---|---|
| `src/yuzu/` | `src/lichii/` (Qt frontend) |
| `src/yuzu_cmd/` | `src/lichii_cmd/` |
| `src/yuzu_room_standalone/yuzu_room_standalone.cpp` | `src/lichii_room_standalone/lichii_room_standalone.cpp` |
| `src/dedicated_room/yuzu_room.{cpp,h,rc}` | `src/dedicated_room/lichii_room.{cpp,h,rc}` |
| `src/tests/yuzu_cmd/` | `src/tests/lichii_cmd/` |
| `dist/yuzu.manifest` | `dist/lichii.manifest` |
| `src/android/app/src/main/java/org/yuzu/yuzu_emu/` | `src/android/app/src/main/java/io/github/zortos293/lichii/` |

### CMake targets

| Old | New | Output |
|---|---|---|
| `yuzu` | `lichii` | `lichii` (`lichii.exe`) |
| `yuzu-cmd` | `lichii-cmd` | `eden-cli` |
| `yuzu-room` | `lichii-room` | `eden-room` |
| `yuzu_room_standalone` | `lichii_room_standalone` | `eden-room` |
| `yuzu-android` | `lichii-android` | `liblichii-android.so` |

### CMake options

Every `YUZU_<name>` option and the compile definition it produces is `LICHII_<name>`, for
example `YUZU_TESTS` -> `LICHII_TESTS`, `YUZU_CMD` -> `LICHII_CMD`, `YUZU_ROOM` ->
`LICHII_ROOM`, `YUZU_USE_BUNDLED_QT` -> `LICHII_USE_BUNDLED_QT`. The full list is `OPTIONS` in
`tools/rename/areas/build.py`.

### Android

| Old | New |
|---|---|
| package and namespace `org.yuzu.yuzu_emu` | `io.github.zortos293.lichii` |
| JNI exports `Java_org_yuzu_yuzu_1emu_*` | `Java_io_github_zortos293_lichii_*` |
| `YuzuApplication`, `Yuzu*` classes, `Theme.Yuzu.*` | `LichiiApplication`, `Lichii*`, `Theme.Lichii.*` |
| `ic_yuzu`, `ic_yuzu_splash` | `ic_lichii`, `ic_lichii_splash` |

The applicationId `dev.eden.eden_emulator` was never a yuzu name and is unchanged.

### Identifiers and text

Identifiers become `lichii`, `Lichii` and `LICHII` in the case they had, or a neutral name where
Lichii reads wrong. In documentation and comments the project name yuzu or Yuzu becomes Lichii.

## Kept names

Each remaining yuzu is listed by `check -v` with its class and reason.

- `YUZU_<name>`, `-DYUZU_*`: old CMake option names `LegacyOptions.cmake` still reads (compat).
- `YuzuRoom-BanList-1`: ban list file magic of existing room ban lists (compat).
- `yuzucach`: shader cache file magic in existing caches (compat).
- Lineage statements naming yuzu among the projects Lichii builds on, and copyright notices
  (attribution).
- References to the separate yuzu emulator in the user docs: Steam ROM Manager's
  "Nintendo Switch - Yuzu" preset, the `ikuzen/yuzu-hdr-multiplayer-dedicated` Docker image,
  the profile bug in earlier versions of yuzu, and a link into Eden's `src/yuzu` (compat).
- `yuzu_meta` (NAND directory of installed raw CNMTs) and `.yuzu_save_size` (save-data size
  file) (compat).
- `YuzuFakeSsidForLdn`: the LDN SSID that other yuzu-based emulators match (compat).
- `"yuzu Emulator"` as the Vulkan application and engine name: AMD's driver applies a
  per-application profile to it (Eden #144 and #145) (compat).
- `YUZU_DIR`, `LEGACY_PATH(Yuzu, YUZU)`, `YuzuDir`, `YuzuConfigDir`, `YuzuCacheDir` and the
  "Yuzu" entry in `migration_worker.h`: the yuzu emulator's data directories that users import
  from (compat).
- `github.com/eden-emulator/yuzu-mod-archive`: the external mods URL (compat).
- Comments about amiibo data and input mappings the yuzu emulator wrote (compat), "upstream
  yuzu" and the yuzu PR #3173 link in the libraries, and Eden's old about text in the
  translations (attribution).

## Removed names

- The Android help links to `yuzu-mirror.github.io/help/quickstart` (and `yuzu-emu.org` in
  Arabic), a key and firmware dumping guide on another project's site. The five strings that
  only held a link (`add_games_warning_help`, `install_prod_keys_warning_help`,
  `install_firmware_warning_help`, `install_game_content_help_link`, `keys_missing_help`) are
  gone with their `helpLinkId`/`warningHelpLinkId` uses. `applets_error_description` and
  `loader_error_encrypted_keys_description` keep their text without the links, and
  `loader_error_encrypted_roms_description` reads "Please redump your game cartridges or
  installed titles." in English only, because its translations pointed at the guides. If an
  upstream sync brings such a link back, `check --strict` lists it as removed: delete it the
  same way. Don't point these strings at another site.
