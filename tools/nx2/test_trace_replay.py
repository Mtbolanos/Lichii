#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests trace_replay.py's PNG reader and frame comparison."""

import math
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_replay  # noqa: E402


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def write_png(path, width, height, pixels, filters):
    """RGBA PNG whose rows use the given filter types (1-4 encoded properly, 0 raw)."""
    stride = width * 4
    raw = b""
    previous = bytes(stride)
    for y in range(height):
        line = pixels[y * stride:(y + 1) * stride]
        kind = filters[y % len(filters)]
        encoded = bytearray(stride)
        for i in range(stride):
            left = line[i - 4] if i >= 4 else 0
            up = previous[i]
            corner = previous[i - 4] if i >= 4 else 0
            if kind == 0:
                predictor = 0
            elif kind == 1:
                predictor = left
            elif kind == 2:
                predictor = up
            elif kind == 3:
                predictor = (left + up) >> 1
            else:
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else corner
            encoded[i] = (line[i] - predictor) & 0xFF
        raw += bytes([kind]) + bytes(encoded)
        previous = line
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                           chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class TraceReplayTest(unittest.TestCase):
    def test_png_filters_round_trip(self):
        width, height = 7, 5
        pixels = bytes((x * 37 + y * 11 + c * 5) & 0xFF
                       for y in range(height) for x in range(width) for c in range(4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            write_png(path, width, height, pixels, [0, 1, 2, 3, 4])
            self.assertEqual(trace_replay.read_png(path), (width, height, pixels))

    def test_difference(self):
        a = (2, 1, bytes([0, 0, 0, 255, 10, 10, 10, 255]))
        self.assertEqual(trace_replay.difference(a, a)["psnr"], math.inf)
        b = (2, 1, bytes([0, 0, 0, 255, 10, 10, 14, 255]))
        result = trace_replay.difference(a, b)
        self.assertEqual(result["max"], 4)
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertTrue(trace_replay.difference(a, (1, 1, bytes(4)))["size_differs"])

    def test_device_class(self):
        self.assertEqual(trace_replay.device_class("llvmpipe"), "lavapipe")
        self.assertEqual(trace_replay.device_class("NVIDIA GeForce RTX 4090"), "nvidia")

    def test_compare_runs_marks_differing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            base, head = Path(directory) / "base", Path(directory) / "head"
            base.mkdir()
            head.mkdir()
            write_png(base / "frame-0000.png", 1, 1, bytes([1, 2, 3, 255]), [0])
            write_png(head / "frame-0000.png", 1, 1, bytes([1, 2, 9, 255]), [0])
            frames = {"frames": [{"index": 0, "file": "frame-0000.png", "hash": "a"}]}
            changed = {"frames": [{"index": 0, "file": "frame-0000.png", "hash": "b"}]}
            rows = trace_replay.compare_runs(base, frames, head, changed)
            self.assertFalse(rows[0]["equal"])
            self.assertEqual(rows[0]["difference"]["max"], 6)
            self.assertTrue(trace_replay.compare_runs(base, frames, base, frames)[0]["equal"])


if __name__ == "__main__":
    unittest.main()
