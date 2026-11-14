#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unpacks a Lichii bug bundle and builds index.html for it.

Usage: python3 -I tools/nx2/bundle_index.py <bundle.zip> [--out DIR]

DIR defaults to the zip's path without .zip. When the bundle has a frame capture, frame/index.html is built
with frame_dump_index.py as well. Everything read from the zip is shown as untrusted text.
"""
import argparse
import html
import importlib.util
import json
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

FORMAT = "lichii-bug-bundle"
MAX_ENTRIES = 20000
MAX_UNPACKED = 2 * 2**30
SHADER_LISTS = ("frame/shaders/shaders.jsonl", "shaders/shaders.jsonl")
FILES = ("settings.txt", "system.json", "mods.json", "frame_times.json", "README.md")


class BundleError(Exception):
    pass


def esc(value):
    return html.escape("-" if value is None or value == "" else str(value))


def load_frame_dump_index():
    path = Path(__file__).with_name("frame_dump_index.py")
    spec = importlib.util.spec_from_file_location("frame_dump_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(archive):
    try:
        data = archive.read("bundle.json")
    except KeyError:
        raise BundleError('no bundle.json at the zip root, so this is not a Lichii bug bundle') from None
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"bundle.json is not valid JSON ({error}), so this is not a Lichii bug bundle") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BundleError(f'bundle.json does not have "format": "{FORMAT}", so this is not a Lichii bug bundle')
    return manifest


def check_members(infos):
    if len(infos) > MAX_ENTRIES:
        raise BundleError(f"the zip has {len(infos)} entries, more than {MAX_ENTRIES}")
    total = 0
    for info in infos:
        name = info.filename.replace("\\", "/")
        parts = PurePosixPath(name).parts
        if name.startswith("/") or (len(name) > 1 and name[1] == ":") or ".." in parts:
            raise BundleError(f"unsafe path in the zip: {info.filename!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise BundleError(f"symlink in the zip: {info.filename!r}")
        total += info.file_size
    if total > MAX_UNPACKED:
        raise BundleError(f"the zip unpacks to {total / 2**30:.1f} GiB, more than {MAX_UNPACKED / 2**30:.0f} GiB")


def unpack(archive, out):
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    for info in archive.infolist():
        target = (root / info.filename.replace("\\", "/")).resolve()
        if target != root and root not in target.parents:
            raise BundleError(f"unsafe path in the zip: {info.filename!r}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)


def link(out, rel_base, path):
    """A link to path (relative to rel_base inside out), or the escaped path alone when it is missing or escapes out."""
    if not isinstance(path, str) or not path:
        return "-"
    rel = PurePosixPath(rel_base) / path
    target = (out / rel).resolve()
    if ".." in rel.parts or not target.is_file() or out.resolve() not in target.parents:
        return f'<span class="missing">{esc(path)}</span>'
    return f'<a href="{html.escape(quote(rel.as_posix()))}">{esc(path)}</a>'


def number(value, digits=1):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else esc(value)


def summary_rows(manifest, frame_times):
    def section(key):
        value = manifest.get(key)
        return value if isinstance(value, dict) else {}

    build, game, gpu, system = section("build"), section("game"), section("gpu"), section("system")
    frame, times = section("frame"), section("frame_times")
    rows = [
        ("build", f"{esc(build.get('tag') or build.get('name'))} · {esc(build.get('scm_rev'))} · {esc(build.get('branch'))}"),
        ("game", esc(game.get("title"))),
        ("program id", esc(game.get("program_id"))),
        ("version", esc(game.get("version"))),
        ("GPU", " · ".join(esc(gpu.get(k)) for k in ("device", "vendor", "driver", "api", "renderer"))),
        ("system", " · ".join((esc(system.get("os")), esc(system.get("cpu")),
                               f"{esc(system.get('threads'))} threads", f"{esc(system.get('ram_mib'))} MiB RAM"))),
        ("trigger", esc(manifest.get("trigger"))),
        ("created", esc(manifest.get("created"))),
    ]
    if frame.get("captured") is False:
        rows.append(("frame capture", "not captured: " + esc(frame.get("error"))))
    elif frame:
        stats = " · ".join(f"{esc(frame.get(k))} {k}" for k in ("passes", "draws", "dispatches", "pipelines", "programs"))
        extra = f" · images ×{esc(frame.get('image_scale'))}" + (" · TRUNCATED" if frame.get("truncated") else "")
        if frame.get("error"):
            extra += " · error: " + esc(frame.get("error"))
        rows.append(("frame capture", f"frame {esc(frame.get('frame'))}: {stats}{extra}"))
    else:
        rows.append(("frame capture", "-"))
    longest = frame_times.get("longest_ms") if isinstance(frame_times, dict) else None
    if isinstance(longest, list):
        longest = " ".join(number(v) for v in longest)
    rows.append(("frame times", f"{number(times.get('fps'))} fps · p99 {number(times.get('p99_ms'))} ms · "
                                f"{esc(times.get('stutters'))} stutters · {esc(times.get('frames'))} frames in "
                                f"{esc(times.get('seconds'))} s · longest {longest if isinstance(longest, str) else number(longest)} ms"))
    return rows


def dropped_rows(manifest):
    dropped = manifest.get("dropped")
    if not isinstance(dropped, list):
        return []
    return [f"<tr><td>{esc(d.get('what'))}</td><td>{esc(d.get('files'))}</td><td>{esc(d.get('bytes'))}</td>"
            f"<td>{esc(d.get('reason'))}</td></tr>" for d in dropped if isinstance(d, dict)]


def program_links(out, base, program):
    if not isinstance(program, dict):
        return "-", "-"
    return link(out, base, program.get("code")), link(out, base, program.get("header"))


def shader_rows(out):
    path = next((p for p in SHADER_LISTS if (out / p).is_file()), None)
    if not path:
        return None, []
    base = PurePosixPath(path).parent
    rows = []
    for line in (out / path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") == "compute":
            ident, key = entry.get("shader"), link(out, base, entry.get("qmd"))
            hashes = esc(entry.get("shader"))
            code, header = program_links(out, base, entry.get("program"))
            spirv = link(out, base, entry.get("spirv"))
        else:
            ident, key = entry.get("pipeline"), link(out, base, entry.get("key"))
            hashes = esc(" ".join(str(h) for h in entry.get("hashes") or []))
            programs = [p for p in entry.get("programs") or [] if isinstance(p, dict)]
            code = "<br>".join(f"{esc(p.get('slot'))}: {link(out, base, p.get('code'))}" for p in programs) or "-"
            header = "<br>".join(link(out, base, p.get("header")) for p in programs if p.get("header")) or "-"
            spirv = "<br>".join(f"{esc(s.get('stage'))}: {link(out, base, s.get('file'))}"
                                for s in entry.get("spirv") or [] if isinstance(s, dict)) or "-"
        rows.append(f"<tr><td>{esc(entry.get('kind'))}</td><td>{esc(ident)}</td><td>{key}</td><td>{esc(entry.get('built'))}</td>"
                    f"<td>{hashes}</td><td>{code}</td><td>{header}</td><td>{spirv}</td></tr>")
    return path, rows


STYLE = """
body{font:13px system-ui,sans-serif;background:#1b1b1f;color:#ddd;margin:16px;max-width:1400px}
h1{font-size:18px}h2{font-size:15px;margin-top:20px}a{color:#8bf}
table{border-collapse:collapse}th,td{text-align:left;padding:2px 8px;vertical-align:top;border-bottom:1px solid #333}
th{color:#999}td{word-break:break-all}.missing{color:#777}
pre.desc{background:#26262c;border-left:4px solid #a84;padding:8px;white-space:pre-wrap;max-width:900px}
img.shot{max-width:100%;max-height:540px;display:block}
"""


def build_page(out, manifest, frame_index):
    frame_times = {}
    try:
        frame_times = json.loads((out / "frame_times.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    game = manifest.get("game") if isinstance(manifest.get("game"), dict) else {}
    summary = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary_rows(manifest, frame_times))
    parts = [f"<!doctype html><meta charset=utf-8><title>Lichii bug bundle</title><style>{STYLE}</style>",
             f"<h1>Lichii bug bundle: {esc(game.get('title') or game.get('program_id'))}</h1>",
             f"<table>{summary}</table>",
             "<h2>Reporter's description (untrusted)</h2>",
             f'<pre class="desc">{html.escape(str(manifest.get("description") or "")) or "(empty)"}</pre>']
    if (out / "screenshot.png").is_file():
        parts.append('<h2>Screenshot</h2><a href="screenshot.png"><img class="shot" src="screenshot.png"></a>')
    files = ['<li><a href="frame/index.html">frame/index.html</a> (frame capture)</li>'] if frame_index else []
    logs = sorted(p.relative_to(out).as_posix() for p in (out / "logs").glob("*") if p.is_file()) if (out / "logs").is_dir() else []
    for name in [*FILES, *logs]:
        if (out / name).is_file():
            files.append(f'<li><a href="{html.escape(quote(name))}">{esc(name)}</a></li>')
    parts.append(f"<h2>Files</h2><ul>{''.join(files) or '<li>none</li>'}</ul>")
    dropped = dropped_rows(manifest)
    if dropped:
        parts.append("<h2>Dropped to fit the size limit</h2><table><tr><th>what</th><th>files</th><th>bytes</th><th>reason</th></tr>"
                     + "".join(dropped) + "</table>")
    shader_list, shaders = shader_rows(out)
    if shader_list:
        parts.append(f"<h2>Shaders ({len(shaders)} pipelines, {esc(shader_list)})</h2><table><tr><th>kind</th>"
                     "<th>pipeline / shader</th><th>key / QMD</th><th>built</th><th>stage hashes</th><th>native program</th>"
                     "<th>header</th><th>SPIR-V</th></tr>" + "".join(shaders) + "</table>")
    index = out / "index.html"
    index.write_text("".join(parts), encoding="utf-8")
    return index


def build(bundle, out):
    try:
        archive = zipfile.ZipFile(bundle)
    except (OSError, zipfile.BadZipFile) as error:
        raise BundleError(f"can't open the zip: {error}") from None
    with archive:
        check_members(archive.infolist())
        manifest = read_manifest(archive)
        unpack(archive, out)
    frame_index = None
    if (out / "frame" / "manifest.jsonl").is_file():
        frame_index = load_frame_dump_index().build_index(out / "frame")
    return build_page(out, manifest, frame_index)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unpack a Lichii bug bundle and build index.html for it.")
    parser.add_argument("bundle", type=Path, help="lichii-bug-*.zip")
    parser.add_argument("--out", type=Path, help="where to unpack it (default: the zip path without .zip)")
    args = parser.parse_args(argv)
    out = args.out or (args.bundle.with_suffix("") if args.bundle.suffix.lower() == ".zip" else Path(str(args.bundle) + ".d"))
    try:
        index = build(args.bundle, out)
    except BundleError as error:
        sys.exit(f"{args.bundle}: {error}")
    print(index)


if __name__ == "__main__":
    main()
