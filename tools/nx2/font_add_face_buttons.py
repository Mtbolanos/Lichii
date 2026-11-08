"""Add the face-button position glyphs U+E0AA-U+E0AE to the synthesized standard font.

Usage: font_add_face_buttons.py <src/core/file_sys/system_archive/data>

Switch 2 titles look the controller button icons up in the standard font. Besides the A/B/X/Y
letters (U+E0A0-E0A3, E0E0-E0E3), Nintendo's extension font has a diamond of four face buttons
with none (U+E0AA) or one of them filled: right (E0AB), bottom (E0AC), top (E0AD) and left
(E0AE). These glyphs are drawn here from circles, in the box of the existing button glyphs, and
font_standard.cpp and font_standard.h are rewritten with the new font. Needs fontTools.
"""
import re
import sys
from io import BytesIO
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

FILLED = {0xE0AA: None, 0xE0AB: 'right', 0xE0AC: 'bottom', 0xE0AD: 'top', 0xE0AE: 'left'}
KAPPA = 0.5522847498


def circle(pen, cx, cy, r, clockwise):
    k = r * KAPPA
    if clockwise:
        segments = [((cx + k, cy + r), (cx + r, cy + k), (cx + r, cy)),
                    ((cx + r, cy - k), (cx + k, cy - r), (cx, cy - r)),
                    ((cx - k, cy - r), (cx - r, cy - k), (cx - r, cy)),
                    ((cx - r, cy + k), (cx - k, cy + r), (cx, cy + r))]
    else:
        segments = [((cx - k, cy + r), (cx - r, cy + k), (cx - r, cy)),
                    ((cx - r, cy - k), (cx - k, cy - r), (cx, cy - r)),
                    ((cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)),
                    ((cx + r, cy + k), (cx + k, cy + r), (cx, cy + r))]
    pen.moveTo((cx, cy + r))
    for segment in segments:
        pen.curveTo(*segment)
    pen.closePath()


def main(data_dir):
    cpp = data_dir / 'font_standard.cpp'
    header = data_dir / 'font_standard.h'
    text = cpp.read_text()
    body = text[text.index('{{') + 2:text.index('}};')]
    data = bytes(int(b, 16) for b in re.findall(r'0x([0-9a-fA-F]{2})', body))
    font = TTFont(BytesIO(data))
    cmap = font.getBestCmap()
    glyf, hmtx = font['glyf'], font['hmtx']
    box = glyf[cmap[0xE0A0]]
    x0, y0, x1, y1 = box.xMin, box.yMin, box.xMax, box.yMax
    advance = hmtx[cmap[0xE0A0]][0]
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
    r = round(half * 0.32)
    d = half - r
    stroke = round(r * 0.3)
    centres = {'top': (cx, cy + d), 'right': (cx + d, cy), 'bottom': (cx, cy - d), 'left': (cx - d, cy)}
    for cp, filled in FILLED.items():
        if cp in cmap:
            continue
        pen = TTGlyphPen(None)
        qpen = Cu2QuPen(pen, max_err=1.0, reverse_direction=False)
        for name, (px, py) in centres.items():
            circle(qpen, px, py, r, clockwise=True)
            if name != filled:
                circle(qpen, px, py, r - stroke, clockwise=False)
        name = 'uni%04X' % cp
        glyf[name] = pen.glyph()
        glyf[name].recalcBounds(glyf)
        hmtx[name] = (advance, glyf[name].xMin)
        for table in font['cmap'].tables:
            if table.isUnicode():
                table.cmap[cp] = name
    font.setGlyphOrder(list(glyf.glyphOrder))
    out = BytesIO()
    font.save(out)
    new = out.getvalue()
    lines = [', '.join('0x%02x' % b for b in new[i:i + 12]) + ',' for i in range(0, len(new), 12)]
    head = re.sub(r'std::array<unsigned char, \d+>', 'std::array<unsigned char, %d>' % len(new), text[:text.index('{{') + 2])
    cpp.write_text(head + '\n' + '\n'.join('    ' + l for l in lines) + '\n' + text[text.index('}};'):])
    header.write_text(re.sub(r'std::array<unsigned char, \d+>', 'std::array<unsigned char, %d>' % len(new), header.read_text()))
    print(len(data), '->', len(new))


main(Path(sys.argv[1]))
