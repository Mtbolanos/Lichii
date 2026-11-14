# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# The emulator libraries (common, core, video_core, audio_core, hid_core, input_common, network,
# web_service, shader_recompiler, frontend_common, tests) and the macros and functions they
# define for the rest of the tree.

LIBS = {
    "files": ["src/common/*", "src/core/*", "src/video_core/*", "src/audio_core/*", "src/hid_core/*",
              "src/input_common/*", "src/network/*", "src/web_service/*", "src/shader_recompiler/*",
              "src/frontend_common/*", "src/tests/*", "src/dynarmic/*", "src/updater/*"],
    # The JNI glue and the input driver's Android names belong to the android area.
    "exclude": ["src/common/android/*", "src/input_common/drivers/android.*", "src/tests/lichii_cmd/*",
                "*/CMakeLists.txt", "*.cmake"],
}

REGISTERED_CACHE = ["src/core/file_sys/registered_cache.*"]

MACROS = r"\bYUZU_(PAGESIZE|PAGEBITS|PAGEMASK|NON_COPYABLE|NON_MOVEABLE|NO_INLINE)\b"

TEXT = [
    # Macros and functions the libraries define, wherever they are used. In a macro body the
    # two characters the name gains come out of the padding before the line's backslash.
    (MACROS + r"(.*?)  \\$", r"LICHII_\1\2\\"),
    (MACROS, r"LICHII_\1"),
    (r"\bGetYuzuVersionData\b", "GetLichiiVersionData"),
    # The raw CNMTs installed into the NAND's yuzu_meta directory. The directory keeps its name.
    (r"\bAccumulateYuzuMeta\b", "AccumulateRawMeta", {"files": REGISTERED_CACHE}),
    (r"\bRawInstallYuzuMeta\b", "RawInstallMeta", {"files": REGISTERED_CACHE}),
    (r"\byuzu_meta(_iter)?\b", r"raw_meta\1", {"files": REGISTERED_CACHE}),
    (r"\bin yuzu meta\b", "in the yuzu_meta directory", {"files": REGISTERED_CACHE}),
    # The project name in prose, log text and user-visible strings, then identifiers.
    (r"(?<![A-Za-z0-9_])[yY]uzu(?![A-Za-z0-9_])", "Lichii", LIBS),
    (r"yuzu", "lichii", LIBS),
    (r"Yuzu", "Lichii", LIBS),
    (r"YUZU", "LICHII", LIBS),
]

KEEP = [
    (r"\byuzucach\b", "compat",
     "Vulkan pipeline cache file magic; existing caches and the nx2 shader tools read it",
     ["src/video_core/*", "tools/*.py"]),
    (r"[\"']yuzu_meta[\"']|\byuzu_meta(?= directory)|(?<=CNMT in )yuzu_meta\b", "compat",
     "NAND directory holding installed raw CNMTs in existing user NANDs", REGISTERED_CACHE),
    (r"\.yuzu_save_size\b", "compat", "save data size file stored in existing saves",
     ["src/core/file_sys/savedata_factory.h", "src/core/hle/service/acc/profile_manager.cpp"]),
    (r"\bYuzuFakeSsidForLdn\b", "compat",
     "LDN SSID sent in LAN discovery packets; other yuzu-based emulators match it",
     "src/core/hle/service/ldn/lan_discovery.h"),
    (r"\"yuzu Emulator\"", "compat",
     "Vulkan application and engine name; AMD's driver applies a per-application profile to it "
     "(Eden #144 and #145 restored it after RDNA3 line artifacts in TotK)",
     "src/video_core/vulkan_common/vulkan_wrapper.cpp"),
    (r"#define YUZU_DIR \"yuzu\"|\bLEGACY_PATH\(Yuzu, YUZU\)|\bYuzu(Config|Cache)?Dir\b", "compat",
     "interop: the yuzu emulator's data directories that users import from",
     ["src/common/fs/*", "src/frontend_common/*"]),
    (r"\bprevious instances of yuzu\b", "compat", "interop: amiibo data written by the yuzu emulator",
     "src/core/hle/service/nfc/common/device.cpp"),
    (r"\bold yuzu versions\b", "compat", "interop: input mappings saved by the yuzu emulator",
     "src/input_common/input_poller.cpp"),
    (r"\bupstream yuzu\b", "attribution", "names the upstream project the code comes from",
     "src/video_core/buffer_cache/buffer_cache_base.h"),
    (r"https://github\.com/yuzu-emu/yuzu/pull/\d+", "attribution",
     "link to the upstream yuzu discussion the code comes from", "src/common/threadsafe_queue.h"),
]
