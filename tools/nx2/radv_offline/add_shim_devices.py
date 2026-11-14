#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adds navi23 (gfx1032) and gfx1200 to Mesa's amdgpu drm-shim device list.

The shim only knows chips someone dumped from real hardware. These two entries copy their
nearest dumped sibling with the same gfx level (navi21, gfx1201) and change what identifies
the chip: the family, PCI device id, external revision (inside the chip's ASICREV range in
amdgpu_asic_addr.h) and the shader engine and CU counts (8 CUs per shader array). Shader compilation depends on the
family and gfx level, so the ISA matches the real chip; memory and clock fields don't matter
for compiling. Idempotent.
"""
import re
import sys

DERIVED = [
    # name, copied from, CHIP_*, device id, external rev, shader engines, CUs
    ("navi23", "navi21", "CHIP_NAVI23", "0x73ff", "0x3d", 2, 32),
    ("gfx1200", "gfx1201", "CHIP_GFX1200", "0x7590", "0x41", 2, 32),
]


def entry(text: str, name: str) -> str:
    start = text.index(f'      .name = "{name}",')
    start = text.rindex("   {\n", 0, start)
    end = text.index("\n   },\n", start) + len("\n   },\n")
    return text[start:end]


def derive(block: str, name, chip, device_id, external_rev, engines, cus) -> str:
    block = re.sub(r'\.name = "[^"]+"', f'.name = "{name}"', block, count=1)
    block = re.sub(r"\.radeon_family = CHIP_\w+", f".radeon_family = {chip}", block, count=1)
    block = re.sub(r"\.device_id = 0x[0-9a-f]+", f".device_id = {device_id}", block, count=1)
    block = re.sub(r"\.external_rev = 0x[0-9a-f]+", f".external_rev = {external_rev}", block, count=1)
    block = re.sub(r"\.num_shader_engines = \d+", f".num_shader_engines = {engines}", block, count=1)
    block = re.sub(r"\.cu_active_number = \d+", f".cu_active_number = {cus}", block, count=1)

    def bitmap(match: re.Match) -> str:
        rows = [f"{match.group(2)}{{ {'0xff, 0xff' if i < engines else '0x0, 0x0'}, 0x0, 0x0, }},"
                for i in range(4)]
        return match.group(1) + "\n".join(rows) + match.group(3)

    return re.sub(r"(\.cu_(?:ao_)?bitmap = \{\n)(\s*)\{.*?(\n\s*\},)", bitmap, block, flags=re.S)


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    marker = "};\n\nconst size_t num_amdgpu_devices"
    if marker not in text:
        print(f"{path}: no amdgpu_devices table", file=sys.stderr)
        return 1
    added = ""
    for name, source, *fields in DERIVED:
        if f'.name = "{name}"' not in text:
            added += derive(entry(text, source), name, *fields)
    if added:
        text = text.replace(marker, added + marker, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
