# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# Documentation and the prose left outside the other areas: every Markdown file, plus comments
# in files no other area owns. Android package paths in the docs belong to the android area.

MD = ["*.md"]

# Not a package or file path segment (org.yuzu.yuzu_emu, org/yuzu/, yuzu-hdr-...), so tokens
# other areas own are left to them.
WORD = r"(?<![./\w-]){}(?![/\w-]|\.\w)"

TEXT = [
    # The Qt frontend's CMake target.
    (r"build target remains `yuzu`", "build target is `lichii`", {"files": MD}),
    (r"`yuzu`", "`lichii`", {"files": MD}),
    (r"--target yuzu\b", "--target lichii", {"files": MD}),
    # The project name in prose.
    (r"\bfor Yuzu Qt frontend\b", "for the Lichii Qt frontend", {"files": MD}),
    (WORD.format("yuzu"), "Lichii", {"files": MD}),
    (WORD.format("Yuzu"), "Lichii", {"files": MD}),
]

KEEP = [
    (r"Eglantine, Eden, yuzu and Sudachi", "attribution", "project lineage", "README.md"),
    (r"Eden and yuzu never had it", "attribution", "names the upstream projects the code comes from",
     "docs/nx2/debug-mcp.md"),
    (r"earlier versions of Eden and Yuzu", "compat", "interop: a bug in the separate yuzu emulator that users "
     "migrating from it can hit", "docs/user/Orphaned.md"),
    (r"Nintendo Switch - Yuzu", "compat", "interop: Steam ROM Manager's community preset for the yuzu emulator",
     "docs/user/SteamROM.md"),
    (r"ikuzen/yuzu-hdr-multiplayer-dedicated", "compat", "interop: third-party Docker image name",
     "docs/user/ServerHosting.md"),
    (r"https://git\.eden-emu\.dev/\S*?/src/yuzu/\S*?\.cpp", "compat",
     "interop: link into Eden's repository, whose Qt frontend is still src/yuzu", "docs/CPMUtil.md"),
    (r"\byuzucach\b", "compat", "shader cache file magic in existing caches", MD),
    (r"YUZU_<name>", "compat", "old -DYUZU_* CMake option names LegacyOptions.cmake reads",
     "CMakeModules/LegacyOptions.cmake"),
]
