#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
# -I leaves the script's directory off sys.path.
sys.path.insert(0, HERE)

import bundle_index  # noqa: E402
import frame_dump_index  # noqa: E402

JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")
MANIFEST = {
    "format": "lichii-bug-bundle",
    "version": 1,
    "created": "2026-11-13T20:15:00Z",
    "trigger": "hotkey",
    "description": "<script>alert(1)</script> fog & bloom",
    "build": {"tag": "build-177", "scm_rev": "b8e3fd8", "branch": "nx2"},
    "game": {"title": "The Legend of Zelda: Ocarina of Time", "program_id": "0400486024DEC000", "version": "1.0.1"},
    "system": {"os": "Windows 11 (26200)", "cpu": "AMD Ryzen 9 5900X", "threads": 24, "ram_mib": 32768},
    "gpu": {"renderer": "vulkan", "device": "NVIDIA GeForce RTX 3080", "vendor": "NVIDIA", "driver": "566.36", "api": "1.4.303"},
    "frame": {"captured": True, "frame": 51234, "passes": 1, "draws": 2, "dispatches": 1, "pipelines": 2, "programs": 2,
              "truncated": False, "image_scale": 0.5, "error": ""},
    "frame_times": {"seconds": 10, "frames": 299, "fps": 29.9, "p99_ms": 41.2, "stutters": 3},
    "dropped": [{"what": "frame sampled textures", "files": 412, "bytes": 51200000, "reason": "size limit"}],
}
PROGRAM = {"slot": 1, "hash": "1111111111111111", "native_key": "2222222222222222", "address": "0x1000",
           "code": "native/2222222222222222-gfx-1000.bin", "header": "native/2222222222222222-gfx-1000.header.bin"}
GRAPHICS = {"kind": "graphics", "pipeline": "aaaaaaaaaaaaaaaa", "key": "pipelines/aaaaaaaaaaaaaaaa.key.bin", "key_bytes": 1056,
            "built": True, "hashes": ["1111111111111111"] + ["0000000000000000"] * 5, "programs": [PROGRAM],
            "spirv": [{"stage": 0, "file": "spirv/1111111111111111-0.spv"}]}
COMPUTE = {"kind": "compute", "shader": "cccccccccccccccc", "qmd": "pipelines/cccccccccccccccc.qmd.bin",
           "program": {**PROGRAM, "code": "native/3333333333333333-cs-2000.bin", "header": None},
           "spirv": "spirv/cccccccccccccccc-cs.spv"}


def write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def bundle_files():
    return {
        "bundle.json": json.dumps(MANIFEST),
        "settings.txt": "[Renderer] backend = vulkan\n",
        "system.json": "{}",
        "frame_times.json": json.dumps({"fps": 29.9, "longest_ms": [80.5, 41.2]}),
        "screenshot.png": b"\x89PNG fake",
        "logs/eden_log.txt": "Lichii Version: Lichii build-177 | nx2-build-177\n",
        "frame/manifest.jsonl": "\n".join(json.dumps(e) for e in [
            {"kind": "frame", "frame": 51234, "seconds": 60},
            {"kind": "pass", "name": "0001_pass0_rt0", "pass": 0, "slot": 0, "first_draw": 0, "draw_count": 2,
             "format": "R8G8B8A8_UNORM", "width": 1, "height": 1, "png": None, "preview": "0001_pass0_rt0.jpg"},
            {"kind": "summary", "images": 1, "passes": 1, "draws": 2, "dispatches": 1, "bytes": 1024},
        ]) + "\n",
        "frame/0001_pass0_rt0.jpg": JPEG,
        "frame/shaders/shaders.jsonl": json.dumps(GRAPHICS) + "\n" + json.dumps(COMPUTE) + "\n",
        "frame/shaders/native/2222222222222222-gfx-1000.bin": b"\0\1",
        "frame/shaders/native/2222222222222222-gfx-1000.header.bin": b"\0",
        "frame/shaders/spirv/1111111111111111-0.spv": b"\x03\x02\x23\x07",
    }


class BundleIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "bundle_index.py"), *map(str, args)],
                              capture_output=True, text=True)

    def test_builds_both_pages_and_escapes_untrusted_text(self):
        bundle = self.dir / "lichii-bug-build-177-0400486024DEC000-20261113-201500.zip"
        write_zip(bundle, bundle_files())
        result = self.run_cli(bundle)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.dir / "lichii-bug-build-177-0400486024DEC000-20261113-201500"
        self.assertEqual(result.stdout.strip().splitlines()[-1], str(out / "index.html"))

        frame = (out / "frame" / "index.html").read_text(encoding="utf-8")
        self.assertIn('src="0001_pass0_rt0.jpg"', frame)

        page = (out / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; fog &amp; bloom", page)
        for text in ("build-177", "0400486024DEC000", "1.0.1", "NVIDIA GeForce RTX 3080", "566.36", "1.4.303",
                     "AMD Ryzen 9 5900X", "32768 MiB RAM", "hotkey", "2026-11-13T20:15:00Z", "frame 51234",
                     "29.9 fps", "p99 41.2 ms", "3 stutters", "longest 80.5 41.2 ms", "frame sampled textures",
                     'src="screenshot.png"', 'href="frame/index.html"', 'href="settings.txt"', 'href="system.json"',
                     'href="logs/eden_log.txt"'):
            self.assertIn(text, page)
        self.assertNotIn('href="mods.json"', page)
        self.assertIn('href="frame/shaders/native/2222222222222222-gfx-1000.bin"', page)
        self.assertIn('href="frame/shaders/native/2222222222222222-gfx-1000.header.bin"', page)
        self.assertIn('href="frame/shaders/spirv/1111111111111111-0.spv"', page)
        self.assertIn("aaaaaaaaaaaaaaaa", page)
        self.assertIn("cccccccccccccccc", page)
        self.assertIn('<span class="missing">native/3333333333333333-cs-2000.bin</span>', page)
        self.assertIn("2 pipelines", page)

    def test_refuses_unsafe_paths(self):
        for name in ("../evil", "/abs/evil", "frame/../../evil"):
            with self.subTest(name=name):
                bundle = self.dir / "bad.zip"
                write_zip(bundle, {**bundle_files(), name: "x"})
                result = self.run_cli(bundle, "--out", self.dir / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe path", result.stderr)
                self.assertFalse((self.dir / "evil").exists())
                self.assertFalse((self.dir / "out").exists())

    def test_refuses_symlinks_and_too_many_entries(self):
        bundle = self.dir / "link.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("bundle.json", json.dumps(MANIFEST))
            info = zipfile.ZipInfo("logs/eden_log.txt")
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, "/etc/passwd")
        self.assertIn("symlink", self.run_cli(bundle).stderr)
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
        with self.assertRaisesRegex(bundle_index.BundleError, "entries"):
            bundle_index.check_members(infos * (bundle_index.MAX_ENTRIES // 2 + 1))

    def test_refuses_a_zip_that_is_not_a_bundle(self):
        plain = self.dir / "logs.zip"
        write_zip(plain, {"eden_log.txt": "log"})
        result = self.run_cli(plain)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a Lichii bug bundle", result.stderr)
        other = self.dir / "other.zip"
        write_zip(other, {"bundle.json": json.dumps({"format": "something-else"})})
        self.assertIn("not a Lichii bug bundle", self.run_cli(other).stderr)
        self.assertFalse((self.dir / "logs").exists())


    def test_checks_sizes_before_reading_bundle_json(self):
        bundle = self.dir / "big.zip"
        write_zip(bundle, {"bundle.json": "not json " * 100})
        limit = bundle_index.MAX_UNPACKED
        bundle_index.MAX_UNPACKED = 100
        try:
            with self.assertRaises(bundle_index.BundleError) as caught:
                bundle_index.build(bundle, self.dir / "out")
        finally:
            bundle_index.MAX_UNPACKED = limit
        self.assertNotIn("JSON", str(caught.exception))


class FrameDumpIndexTest(unittest.TestCase):
    def test_untrusted_kind_and_preview_stay_inert(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.jsonl").write_text(json.dumps(
                {"kind": '"><script>alert(1)</script>', "name": "x", "preview": "javascript:alert(1)"}) + "\n")
            with redirect_stdout(io.StringIO()):
                page = frame_dump_index.build_index(root).read_text(encoding="utf-8")
            self.assertNotIn("<script>", page)
            self.assertNotIn("javascript:", page)

    def test_plain_dump_with_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.jsonl").write_text(json.dumps(
                {"kind": "present", "name": "0001_present", "after_draws": 3, "png": "0001_present.png"}) + "\n")
            with redirect_stdout(io.StringIO()) as printed:
                out = frame_dump_index.build_index(root)
            self.assertEqual(out, root / "index.html")
            self.assertEqual(printed.getvalue(), f"{out}: 1 entries\n")
            self.assertIn('src="0001_present.png"', out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
