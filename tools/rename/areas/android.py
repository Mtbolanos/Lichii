# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# The Android frontend, its JNI glue and the Android names other files refer to.
#
# The Kotlin package org.yuzu.yuzu_emu becomes io.github.zortos293.lichii, the id of the Linux
# desktop entry. The applicationId (dev.eden.eden_emulator) is not a yuzu name and stays, so
# the default SharedPreferences file, the provider authorities and the app's storage are the
# same as before.

ANDROID = ["src/android/*", "src/common/android/*"]

PATHS = [
    (r"/java/org/yuzu/yuzu_emu/", "/java/io/github/zortos293/lichii/", {"files": "src/android/*"}),
    (r"Yuzu", "Lichii", {"files": "src/android/*"}),
    (r"yuzu", "lichii", {"files": "src/android/*"}),
]

TEXT = [
    # The package in Kotlin, the manifest, layouts, navigation graphs and docs, as a JNI class
    # name or signature, and in JNI export names ("_" in a package segment mangles to "_1").
    (r"org\.yuzu\.yuzu_emu(?![A-Za-z0-9_])", "io.github.zortos293.lichii"),
    (r"org/yuzu/yuzu_emu(?![A-Za-z0-9_])", "io/github/zortos293/lichii"),
    (r"\bJava_org_yuzu_yuzu_1emu_", "Java_io_github_zortos293_lichii_"),
    # Android names used outside the Android tree: the launcher icons update-icons.sh writes,
    # the logcat tag in common/logging.cpp, and the input device accessors and Kotlin
    # interface the input driver uses.
    (r"\bic_yuzu", "ic_lichii"),
    (r"\bYuzuNative\b", "LichiiNative"),
    (r"Yuzu", "Lichii", {"files": ["src/input_common/drivers/android.*"]}),
    # Everything else in the Android tree: classes (YuzuApplication, YuzuInputDevice, ...),
    # themes and styles, colour and string resource names, and the JNI id cache.
    (r"Yuzu", "Lichii", {"files": ANDROID}),
    (r"yuzu", "lichii", {"files": ANDROID}),
    (r"YUZU", "LICHII", {"files": ANDROID}),
]

KEEP = [
    (r"https://yuzu-(mirror\.github\.io|emu\.org)/help/quickstart[^\s\"<\\]*", "removed",
     "link to the yuzu quickstart guide: delete the string and its helpLinkId use (docs/nx2/rename.md)",
     "src/android/app/src/main/res/values*/strings.xml"),
]
