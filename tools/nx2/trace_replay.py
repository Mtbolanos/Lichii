#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replays GPU traces (docs/nx2/gpu-trace.md) and compares their frames.

  trace_replay.py replay --cli <eden-cli> <trace> --out <dir>
  trace_replay.py compare --base <eden-cli> --head <eden-cli> <trace> --out <dir>
  trace_replay.py check --cli <eden-cli> --corpus tests/gpu_traces/corpus.json [--traces <dir>]

compare replays one trace on two builds and writes <out>/index.html with the frames side by
side and, per frame, the hashes, the largest channel difference, the mean difference and PSNR.
check replays every corpus trace it can find and compares the frame hashes with the expected
ones for the replaying device class; it exits 1 on a mismatch.
"""

import argparse
import hashlib
import html
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

SAMPLE_TRACE = "sample"


def replay(cli, trace, out, extra=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    command = [str(cli), "--replay", str(trace), "--out", str(out), *extra]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}):\n{tail}")
    return json.loads((out / "frames.json").read_text())


def device_class(renderer):
    """The expected-hash key for a replay's renderer string."""
    name = renderer.lower()
    if "llvmpipe" in name or "lavapipe" in name:
        return "lavapipe"
    return name.split()[0] if name else "unknown"


def read_png(path):
    """Decodes an 8-bit RGBA or RGB PNG (what eden-cli writes) to (width, height, rgba bytes)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    offset = 8
    idat = b""
    width = height = channels = 0
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or color not in (2, 6) or interlace != 0:
                raise ValueError(f"{path}: only 8-bit RGB(A), non-interlaced PNGs are read")
            channels = 4 if color == 6 else 3
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    raw = zlib.decompress(idat)
    stride = width * channels
    rows = []
    previous = bytearray(stride)
    position = 0
    for _ in range(height):
        kind = raw[position]
        line = bytearray(raw[position + 1:position + 1 + stride])
        position += 1 + stride
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = previous[i]
            corner = previous[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + left) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + up) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + ((left + up) >> 1)) & 0xFF
            elif kind == 4:
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else corner
                line[i] = (line[i] + predictor) & 0xFF
        rows.append(bytes(line))
        previous = line
    pixels = b"".join(rows)
    if channels == 3:
        pixels = b"".join(pixels[i:i + 3] + b"\xff" for i in range(0, len(pixels), 3))
    return width, height, pixels


def difference(a, b):
    """Largest channel difference, mean difference and PSNR (inf when equal) of two images."""
    if a[0] != b[0] or a[1] != b[1]:
        return {"size_differs": True}
    largest = 0
    total = 0
    squares = 0
    for x, y in zip(a[2], b[2]):
        d = abs(x - y)
        largest = max(largest, d)
        total += d
        squares += d * d
    count = len(a[2]) or 1
    mse = squares / count
    psnr = math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)
    return {"max": largest, "mean": total / count, "psnr": psnr}


def compare_runs(base_dir, base, head_dir, head):
    rows = []
    base_frames = {frame["index"]: frame for frame in base["frames"]}
    head_frames = {frame["index"]: frame for frame in head["frames"]}
    for index in sorted(set(base_frames) | set(head_frames)):
        row = {"index": index, "base": base_frames.get(index), "head": head_frames.get(index)}
        if row["base"] and row["head"]:
            row["equal"] = row["base"]["hash"] == row["head"]["hash"]
            if not row["equal"]:
                row["difference"] = difference(read_png(Path(base_dir) / row["base"]["file"]),
                                               read_png(Path(head_dir) / row["head"]["file"]))
        else:
            row["equal"] = False
        rows.append(row)
    return rows


def write_html(out, rows, base_label, head_label):
    lines = ["<!doctype html><meta charset=utf-8><title>GPU trace comparison</title>",
             "<style>body{font:14px sans-serif}td{padding:4px;vertical-align:top}"
             "img{max-width:640px;image-rendering:pixelated}.bad{background:#fdd}</style>",
             f"<p>base: {html.escape(base_label)}<br>head: {html.escape(head_label)}</p><table>"]
    for row in rows:
        cls = "" if row["equal"] else " class=bad"
        cells = []
        for side in ("base", "head"):
            frame = row[side]
            cells.append(f"<td><img src='{side}/{frame['file']}'><br>{frame['hash']}</td>"
                         if frame else "<td>missing</td>")
        note = "identical" if row["equal"] else html.escape(json.dumps(row.get("difference", {})))
        lines.append(f"<tr{cls}><td>{row['index']}</td>{''.join(cells)}<td>{note}</td></tr>")
    lines.append("</table>")
    (Path(out) / "index.html").write_text("\n".join(lines))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(cli, corpus_path, traces_dir, out, require):
    corpus = json.loads(Path(corpus_path).read_text())
    failures = 0
    for entry in corpus["traces"]:
        trace_id = entry["id"]
        work = Path(out) / trace_id
        work.mkdir(parents=True, exist_ok=True)
        if entry.get("source") == "generated":
            trace = work / "trace.lxtrace"
            subprocess.run([str(cli), "--gpu-trace-sample", str(trace)], check=True)
        else:
            trace = Path(traces_dir or ".") / entry["file"]
            if not trace.exists():
                print(f"{trace_id}: skipped, {trace} is not here")
                failures += 1 if require else 0
                continue
            if entry.get("sha256") and sha256(trace) != entry["sha256"]:
                print(f"{trace_id}: FAIL, {trace} does not match the corpus sha256")
                failures += 1
                continue
        start = entry.get("roundtrip_from")
        extra = [] if start is None else ["--record", str(work / "recorded.lxtrace"),
                                          "--record-from", str(start)]
        result = replay(cli, trace, work / "frames", extra)
        device = device_class(result.get("renderer", ""))
        if start is not None:
            again = replay(cli, work / "recorded.lxtrace", work / "recorded")
            tail = [frame["hash"] for frame in result["frames"]][start:]
            if [frame["hash"] for frame in again["frames"]] != tail:
                print(f"{trace_id}: FAIL, the trace recorded from present {start} replays to "
                      f"other frames")
                failures += 1
                continue
            print(f"{trace_id}: recorded from present {start}, {len(tail)} frames replay the same")
        expected = entry.get("expected", {}).get(device)
        hashes = [frame["hash"] for frame in result["frames"]]
        if expected is None:
            print(f"{trace_id}: no expected frames for {device}; got {json.dumps(hashes)}")
            continue
        if hashes != expected:
            print(f"{trace_id}: FAIL on {device}")
            for index, (got, want) in enumerate(zip(hashes, expected)):
                if got != want:
                    print(f"  frame {index}: {got}, expected {want}")
            if len(hashes) != len(expected):
                print(f"  {len(hashes)} frames, expected {len(expected)}")
            failures += 1
        else:
            print(f"{trace_id}: {len(hashes)} frames match on {device}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("replay")
    one.add_argument("--cli", required=True)
    one.add_argument("trace")
    one.add_argument("--out", required=True)
    two = commands.add_parser("compare")
    two.add_argument("--base", required=True)
    two.add_argument("--head", required=True)
    two.add_argument("trace")
    two.add_argument("--out", required=True)
    corpus = commands.add_parser("check")
    corpus.add_argument("--cli", required=True)
    corpus.add_argument("--corpus", required=True)
    corpus.add_argument("--traces", help="directory holding the corpus traces")
    corpus.add_argument("--out", help="work directory (default: a temporary one)")
    corpus.add_argument("--require", action="store_true",
                        help="fail when a corpus trace is missing instead of skipping it")
    args = parser.parse_args()

    if args.command == "replay":
        result = replay(args.cli, args.trace, args.out)
        for frame in result["frames"]:
            print(f"{frame['index']:4} {frame['hash']} {frame['ms']:8.1f} ms")
        return 0
    if args.command == "compare":
        out = Path(args.out)
        base = replay(args.base, args.trace, out / "base")
        head = replay(args.head, args.trace, out / "head")
        rows = compare_runs(out / "base", base, out / "head", head)
        write_html(out, rows, args.base, args.head)
        differing = [row["index"] for row in rows if not row["equal"]]
        print(f"{len(rows)} frames, {len(differing)} differ: {differing}; see {out / 'index.html'}")
        return 1 if differing else 0
    with tempfile.TemporaryDirectory() as temporary:
        failures = check(args.cli, args.corpus, args.traces, args.out or temporary, args.require)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
