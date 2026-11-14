#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rename inherited yuzu naming to Lichii; see docs/nx2/rename.md.

apply  rewrites tracked paths and text with every area under tools/rename/areas.
check  lists what still says yuzu, by class: attribution, compat, removed or pending.

Both are idempotent, so a branch or an upstream merge can run apply again.
"""

import argparse
import fnmatch
import re
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AREAS_DIR = Path(__file__).resolve().parent / "areas"
WORD = re.compile(r"yuzu", re.IGNORECASE)
URL = re.compile(r"https?://[^\s\"'<>)\]]+")

# Never rewritten: licence texts and this tool.
SKIP_PATHS = ["LICENSE*", "LICENSES/*", "*/LICENSE*", "COPYING*", "tools/rename/*", "docs/nx2/rename.md"]
# Vendored code is out of scope. Only rules that name an externals/ path touch it, and check
# only looks for build option names in our own CMake glue there.
EXTERNALS = "externals/*"
EXTERNALS_GLUE = ["externals/CMakeLists.txt", "externals/*/CMakeLists.txt", "externals/*/*.cmake"]
GLUE_TOKEN = re.compile(r"YUZU_[A-Z]")
# Lines that carry copyright attribution stay byte for byte.
ATTRIBUTION_LINE = re.compile(r"SPDX-FileCopyrightText|^Copyright:|^\s+\d{4}(-\d{4})? yuzu Emulator Project")


class Rule:
    def __init__(self, area, pattern, repl, files="*", exclude=(), in_urls=False):
        self.area = area
        self.pattern = re.compile(pattern)
        self.repl = repl
        self.files = [files] if isinstance(files, str) else list(files)
        self.exclude = list(exclude)
        self.in_urls = in_urls

    def applies_to(self, path):
        if fnmatch.fnmatchcase(path, EXTERNALS) and not any(g.startswith("externals/") for g in self.files):
            return False
        return match_any(path, self.files) and not match_any(path, self.exclude)


def match_any(path, globs):
    return any(fnmatch.fnmatchcase(path, g) for g in globs)


def load_areas(selected):
    areas = []
    for file in sorted(AREAS_DIR.glob("*.py")):
        name = file.stem
        if selected and name not in selected:
            continue
        ns = runpy.run_path(str(file))
        areas.append((name, ns))
    return areas


def collect(areas):
    path_rules, text_rules, keep = [], [], []
    for name, ns in areas:
        for spec in ns.get("PATHS", []):
            path_rules.append(Rule(name, *spec[:2], **(spec[2] if len(spec) > 2 else {})))
        for spec in ns.get("TEXT", []):
            text_rules.append(Rule(name, *spec[:2], **(spec[2] if len(spec) > 2 else {})))
        for spec in ns.get("KEEP", []):
            pattern, cls, reason = spec[:3]
            files = spec[3] if len(spec) > 3 else "*"
            keep.append((re.compile(pattern), cls, reason, [files] if isinstance(files, str) else files, name))
    return path_rules, text_rules, keep


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def tracked():
    return [p for p in git("ls-files", "-z").decode().split("\0") if p and not match_any(p, SKIP_PATHS)]


def read_text(path):
    data = (ROOT / path).read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def kept_spans(line, path, keep):
    spans = []
    for pattern, cls, reason, files, _ in keep:
        if match_any(path, files):
            spans += [(m.start(), m.end(), cls, reason) for m in pattern.finditer(line)]
    return spans


def rewrite_line(line, path, rules, keep):
    if ATTRIBUTION_LINE.search(line) or not WORD.search(line):
        return line
    for rule in rules:
        if not rule.applies_to(path):
            continue
        blocked = [(s, e) for s, e, *_ in kept_spans(line, path, keep)]
        if not rule.in_urls:
            blocked += [m.span() for m in URL.finditer(line)]

        def sub(m):
            if any(m.start() < e and s < m.end() for s, e in blocked):
                return m.group(0)
            return m.expand(rule.repl) if isinstance(rule.repl, str) else rule.repl(m)

        line = rule.pattern.sub(sub, line)
    return line


def apply(areas):
    path_rules, text_rules, _ = collect(areas)
    _, _, keep = collect(load_areas(None))
    moved = 0
    for path in tracked():
        if fnmatch.fnmatchcase(path, EXTERNALS):
            continue
        new = path
        for rule in path_rules:
            if rule.applies_to(new):
                new = rule.pattern.sub(rule.repl, new)
        if new != path:
            (ROOT / new).parent.mkdir(parents=True, exist_ok=True)
            git("mv", path, new)
            moved += 1
    changed = 0
    for path in tracked():
        text = read_text(path)
        if text is None or not WORD.search(text):
            continue
        rules = [r for r in text_rules if r.applies_to(path)]
        if not rules:
            continue
        lines = text.splitlines(keepends=True)
        out = "".join(rewrite_line(l, path, rules, keep) for l in lines)
        if out != text:
            (ROOT / path).write_bytes(out.encode("utf-8"))
            changed += 1
    print(f"moved {moved} paths, rewrote {changed} files")


def classify(line, path, start, end, keep):
    if ATTRIBUTION_LINE.search(line):
        return "attribution", "copyright notice"
    for s, e, cls, reason in kept_spans(line, path, keep):
        if s <= start and end <= e:
            return cls, reason
    return "pending", ""


def check(areas, verbose, strict):
    _, _, keep = collect(areas)
    counts, reasons, pending = Counter(), Counter(), []
    paths = [p for p in tracked() if WORD.search(p) and not fnmatch.fnmatchcase(p, EXTERNALS)]
    for path in tracked():
        external = fnmatch.fnmatchcase(path, EXTERNALS)
        if external and not match_any(path, EXTERNALS_GLUE):
            continue
        text = read_text(path)
        if text is None or not WORD.search(text):
            continue
        for no, line in enumerate(text.splitlines(), 1):
            for m in (GLUE_TOKEN if external else WORD).finditer(line):
                cls, reason = classify(line, path, m.start(), m.end(), keep)
                counts[cls] += 1
                if cls in ("pending", "removed"):
                    pending.append(f"{path}:{no}: {line.strip()[:160]}" + (f"  [{reason}]" if reason else ""))
                else:
                    reasons[(cls, reason)] += 1
    print(f"paths with yuzu: {len(paths)}")
    for p in paths if verbose else paths[:20]:
        print(f"  {p}")
    print("occurrences: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for (cls, reason), n in sorted(reasons.items()):
        print(f"  {cls:12} {n:6}  {reason}")
    if pending:
        print(f"pending lines: {len(pending)}")
        for p in pending if verbose else pending[:40]:
            print(f"  {p}")
    if strict and (pending or paths):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["apply", "check"])
    ap.add_argument("--area", action="append", help="only these areas (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--strict", action="store_true", help="check: fail on pending occurrences")
    args = ap.parse_args()
    areas = load_areas(args.area)
    if args.command == "apply":
        apply(areas)
    else:
        check(load_areas(None), args.verbose, args.strict)


if __name__ == "__main__":
    main()
