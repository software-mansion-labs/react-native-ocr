#!/usr/bin/env python3
"""Renders example/assets/ocr/sample.png, the page the example's parity screen reads.

It carries a title, two columns of short lines (reading order), one long line
in small type that is wider than the recognizer's maximum input (so the
wide-line split is exercised) and one tilted line. Needs Pillow and the macOS
system fonts.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "example" / "assets" / "ocr" / "sample.png"
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


page = Image.new("RGB", (800, 600), "white")
draw = ImageDraw.Draw(page)
draw.text((40, 30), "Parity check 2026", font=font(48), fill="black")

left = ["The quick brown fox", "jumps over the lazy dog", "while 12 geese watch", "from the river bank."]
right = ["Invoice no. 40713", "Total: 1,284.50 EUR", "Due: 14 October", "Thank you!"]
for i, line in enumerate(left):
    draw.text((40, 130 + i * 50), line, font=font(28), fill="black")
for i, line in enumerate(right):
    draw.text((460, 130 + i * 50), line, font=font(28), fill="black")

long_line = (
    "This single line of small type runs across the whole page so that it is far wider "
    "than the widest input the recognizer accepts"
)
draw.text((20, 400), long_line, font=font(13), fill="black")

# One line at an angle, so the detector returns an oriented quad.
tilted = Image.new("RGBA", (460, 50), (255, 255, 255, 0))
ImageDraw.Draw(tilted).text((5, 5), "Rotated lines keep their angle.", font=font(30), fill=(30, 30, 160))
tilted = tilted.rotate(6, expand=True, resample=Image.BICUBIC)
page.paste(tilted, (40, 460), tilted)

OUT.parent.mkdir(parents=True, exist_ok=True)
page.save(OUT, optimize=True)
print(f"wrote {OUT} {page.size}")
