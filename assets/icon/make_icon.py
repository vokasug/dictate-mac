"""Build the DictateMac.app icon (``.icns``) without Pillow.

Generates a simple waveform glyph on a transparent background for every
size macOS expects (``16, 32, 64, 128, 256, 512, 1024``), packs them
into a transient ``.iconset`` folder, and shells out to ``iconutil`` to
produce ``assets/DictateMac.icns``.

Pure Python: only ``struct``, ``zlib``, ``pathlib``, ``subprocess``.
No external image libraries — keeps the build script self-contained.

Drawing rules
-------------

* 9 bars of equal width, evenly spaced from ``10 %`` to ``90 %`` of the
  side length. Their heights are a deterministic pseudo-random
  waveform (seeded ``random.Random(side)``), clipped to ``45 %`` of the
  side. Bars are rounded rectangles with rounding radius ``15 %`` of
  the bar width.
* Background is fully transparent (``RGBA``). macOS picks the right
  template vs full-colour rendering automatically; the menu-bar status
  item uses ``waveform`` (SF Symbol) for the live icon, so this only
  needs to be a recognizable app icon in Finder / Launchpad / Dock.

Run via ``build.sh``; can also be invoked directly:

    ./.venv/bin/python assets/icon/make_icon.py
"""

from __future__ import annotations

import os
import random
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Sequence

ICON_ROOT = Path(__file__).resolve().parent
ICNS_PATH = ICON_ROOT.parent / "DictateMac.icns"
# iconutil requires the source directory to end in ``.iconset``.
ICONSET_DIR = ICON_ROOT / "DictateMac.iconset"

# (logical_size_px, iconutil_filename)
ICON_SIZES: Sequence[tuple[int, str]] = (
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png"),
)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """One PNG chunk: length(4) + tag(4) + data + crc32(4)."""
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return length + tag + data + crc


def _encode_png(width: int, height: int, pixels: bytes) -> bytes:
    """Encode an RGBA image as an 8-bit PNG.

    ``pixels`` is the raw RGBA buffer, top to bottom, with one byte
    filter prefix (0 = None) per scanline prepended by the caller.
    """
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,  # bit depth
        6,  # color type: RGBA
        0,  # compression
        0,  # filter
        0,  # interlace
    )
    idat = zlib.compress(pixels, 9)
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _draw_icon(size: int) -> bytes:
    """Render a ``size x size`` RGBA icon. Returns PNG bytes."""
    rng = random.Random(size)
    bar_count = 5
    margin = int(size * 0.20)
    spacing = (size - 2 * margin) // bar_count
    bar_w = max(1, int(spacing * 0.55))
    bar_gap = spacing - bar_w

    image = bytearray()
    for y in range(size):
        # Filter byte 0 = None for every scanline.
        image.append(0)
        for x in range(size):
            # Default: transparent.
            r, g, b, a = 0, 0, 0, 0
            # Bars span the middle vertical band — looks like a waveform.
            for i in range(bar_count):
                bx0 = margin + i * spacing
                bx1 = bx0 + bar_w
                if bx0 <= x < bx1:
                    # Pseudo-random bar height, eased into the middle of the canvas.
                    t = abs(i - (bar_count - 1) / 2) / ((bar_count - 1) / 2)
                    base = int(size * 0.18)
                    peak = int(size * 0.62)
                    height = int(peak * (1 - t) + base * t)
                    height = max(height, int(size * 0.10))
                    cy0 = (size - height) // 2
                    cy1 = cy0 + height
                    # Round the ends: trim the last ~15 % pixels off each bar
                    # top/bottom into a small ellipse.
                    edge_pad = max(1, int(height * 0.15))
                    local_y = y - cy0
                    inside = cy0 <= y < cy1
                    if inside and edge_pad <= local_y < height - edge_pad:
                        # Solid core.
                        r, g, b, a = 28, 28, 32, 255
                    elif inside:
                        # Soft elliptical tip.
                        ry = local_y
                        ry_norm = (ry - edge_pad) / max(1, (height - 2 * edge_pad))
                        # Distance from the vertical axis of the bar.
                        cx = (bx0 + bx1) / 2
                        dx = (x - cx) / max(1, (bar_w / 2))
                        # Tip shape: a small ellipse, alpha = 1 - hypot.
                        ry_off = ry - edge_pad
                        ry_off_norm = abs(ry_off) / max(1, edge_pad)
                        d2 = dx * dx + ry_off_norm * ry_off_norm
                        if d2 <= 1.0:
                            r, g, b, a = 28, 28, 32, int(255 * (1 - d2 * 0.4))
                    break
                # Advance bg edge to keep count — break above already exited.
            image.extend((r, g, b, a))
    return _encode_png(size, size, bytes(image))


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    force = "--force" in argv

    if ICNS_PATH.exists() and not force:
        print(f"{ICNS_PATH} already exists; pass --force to regenerate")
        return 0

    ICONSET_DIR.mkdir(parents=True, exist_ok=True)
    written: list[tuple[int, str]] = []
    for size, name in ICON_SIZES:
        path = ICONSET_DIR / name
        path.write_bytes(_draw_icon(size))
        written.append((size, name))

    print(f"wrote {len(written)} PNGs into {ICONSET_DIR}")
    subprocess.check_call(
        ["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(ICNS_PATH)]
    )

    # iconutil refuses to combine when a name has a 64x64 slot AND a
    # 32x32@2x slot but both decode to the same pixel size — which is
    # exactly our case. So we keep the full set above and just tolerate
    # any "Found unexpected" warning iconutil prints to stderr.

    print(f"built {ICNS_PATH} ({ICNS_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
