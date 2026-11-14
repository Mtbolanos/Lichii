#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class SwitchDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.bin = self.root / "mount/usr/bin"
        self.bin.mkdir(parents=True)
        self.config = self.root / "config/lichii"
        self.config.mkdir(parents=True)
        source = self.root / "probe.cpp"
        source.write_text('''#include "lichii/nx2_switches.h"
#include <iostream>
namespace Common::FS {
std::filesystem::path GetDataDirectory(const std::string&) {
    return std::getenv("XDG_CONFIG_HOME");
}
}
int main(int argc, char** argv) {
    LoadNx2Switches(argv[0]);
    const char* value = std::getenv(argc > 1 ? argv[1] : "EDEN_NX2_PRELOAD_ALL_PIPELINES");
    std::cout << (value ? value : "unset");
}
''')
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "src"), str(source), "-o", str(self.bin / "lichii")], check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_probe(self, executable, name="EDEN_NX2_PRELOAD_ALL_PIPELINES", **env):
        clean = os.environ.copy()
        clean.pop("EDEN_NX2_PRELOAD_ALL_PIPELINES", None)
        clean.pop(name, None)
        clean.pop("APPIMAGE", None)
        clean["XDG_CONFIG_HOME"] = str(self.config.parent)
        clean.update(env)
        return subprocess.check_output([str(executable), name], env=clean, cwd=self.root, text=True)

    def test_untrusted_sidecars_cannot_set_host_environment(self):
        outer = self.root / "app"
        outer.mkdir()
        image = outer / "Lichii.AppImage"
        image.touch()
        forbidden = ("LD_PRELOAD", "LD_LIBRARY_PATH", "LIBVULKAN_PATH", "QT_PLUGIN_PATH",
                     "PATH", "HOME", "XDG_CONFIG_HOME", "LICHII_DO_STARTUP_CHECKS",
                     "EDEN_STREAMLINE_DIR")
        for folder in (self.bin, outer):
            sidecar = folder / "nx2_switches.txt"
            sidecar.write_text("".join(f"{name}=/untrusted\n" for name in forbidden))
            for name in forbidden:
                with self.subTest(folder=folder, name=name):
                    expected = str(self.config.parent) if name == "XDG_CONFIG_HOME" else "unset"
                    self.assertEqual(self.run_probe(self.bin / "lichii", name, APPIMAGE=str(image)), expected)
            sidecar.unlink()

    def test_diagnostics_and_explicit_user_environment(self):
        supported = ("EDEN_NX2_PRELOAD_ALL_PIPELINES", "EDEN_DKB_FRAME_TRACE",
                     "EDEN_DKB_INDIRECT_READBACK", "EDEN_NATIVE_DRAW_CAPTURE",
                     "EDEN_NATIVE_DRAW_DIAGNOSTICS", "EDEN_NATIVE_FRAME_CAPTURE",
                     "EDEN_NATIVE_SHADER_OVERRIDE_DIR", "EDEN_PUNCH_HOLE",
                     "EDEN_STARFOX_WINDOW_TRACE", "EDEN_VI2_MAX_RESOLUTION_MODE",
                     "EGLANTINE_EXPERIMENTAL_STATIC_VERTEX_REPACK", "EGLANTINE_TEST_VERTEX_REPACK")
        (self.bin / "nx2_switches.txt").write_text("".join(f"{name}=1\n" for name in supported))
        for name in supported:
            with self.subTest(name=name):
                self.assertEqual(self.run_probe(self.bin / "lichii", name), "1")
                self.assertEqual(self.run_probe(self.bin / "lichii", name, **{name: "2"}), "2")
        (self.config / "nx2_switches.txt").write_text("EDEN_STREAMLINE_DIR=/trusted\n")
        self.assertEqual(self.run_probe(self.bin / "lichii", "EDEN_STREAMLINE_DIR"), "/trusted")

    def test_invalid_config_base_does_not_search_cwd(self):
        cwd_config = self.root / "lichii"
        cwd_config.mkdir()
        (cwd_config / "nx2_switches.txt").write_text("EDEN_NX2_PRELOAD_ALL_PIPELINES=99\n")
        for base in ("", "."):
            with self.subTest(base=base):
                self.assertEqual(self.run_probe(self.bin / "lichii", XDG_CONFIG_HOME=base), "unset")

    def test_elf_path_symlink_and_precedence(self):
        (self.root / "nx2_switches.txt").write_text("EDEN_NX2_PRELOAD_ALL_PIPELINES=99\n")
        self.assertEqual(self.run_probe(self.bin / "lichii"), "unset")
        (self.bin / "nx2_switches.txt").write_text('EDEN_NX2_PRELOAD_ALL_PIPELINES="1"\n')
        linkdir = self.root / "links"
        linkdir.mkdir()
        (linkdir / "lichii").symlink_to(self.bin / "lichii")
        self.assertEqual(self.run_probe("lichii", PATH=str(linkdir)), "1")
        (self.config / "nx2_switches.txt").write_text("EDEN_NX2_PRELOAD_ALL_PIPELINES=2\n")
        self.assertEqual(self.run_probe(linkdir / "lichii"), "2")
        self.assertEqual(self.run_probe(linkdir / "lichii", EDEN_NX2_PRELOAD_ALL_PIPELINES="3"), "3")

    def test_appimage_external_overrides_internal(self):
        (self.bin / "nx2_switches.txt").write_text("EDEN_NX2_PRELOAD_ALL_PIPELINES=1\n")
        outer = self.root / "app"
        outer.mkdir()
        image = outer / "Lichii.AppImage"
        image.touch()
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE=str(image)), "1")
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE=str(outer / "missing.AppImage")), "1")
        (outer / "nx2_switches.txt").write_text("EDEN_NX2_PRELOAD_ALL_PIPELINES=2\n")
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE=str(image)), "2")
        link = self.root / "link.AppImage"
        link.symlink_to(image)
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE=str(link)), "2")
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE="app/Lichii.AppImage"), "1")
        (self.config / "nx2_switches.txt").write_text("EDEN_NX2_PRELOAD_ALL_PIPELINES=3\n")
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE=str(image)), "3")
        self.assertEqual(self.run_probe(self.bin / "lichii", APPIMAGE=str(image), EDEN_NX2_PRELOAD_ALL_PIPELINES="4"), "4")


if __name__ == "__main__":
    unittest.main()
