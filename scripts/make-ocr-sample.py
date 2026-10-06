#!/usr/bin/env python3
"""Renders the pages the example app reads, plus the text each one carries.

- example/assets/ocr/sample.png: the page the parity and EasyOCR screens read.
  It carries a title, two columns of short lines (reading order), one long
  line in small type that is wider than the recognizer's maximum input (so the
  wide-line split is exercised) and one tilted line.
- example/assets/ocr/vertical.png: the page the EasyOCR screen reads with the
  `vertical` option. Three words are written as upright glyph stacks, one of
  them with the letters touching so the detector merges them into a single
  tall box (the char-level re-detection path); two horizontal lines check the
  vertical pass still reads ordinary text.
- example/assets/ocr/expected.json: the lines each page carries, so a screen
  can grade what it read.

Needs Pillow and the macOS system fonts.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "example" / "assets" / "ocr"
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def render_sample() -> list[str]:
    page = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(page)
    title = "Parity check 2026"
    draw.text((40, 30), title, font=font(48), fill="black")

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
    tilted_text = "Rotated lines keep their angle."
    tilted = Image.new("RGBA", (460, 50), (255, 255, 255, 0))
    ImageDraw.Draw(tilted).text((5, 5), tilted_text, font=font(30), fill=(30, 30, 160))
    tilted = tilted.rotate(6, expand=True, resample=Image.BICUBIC)
    page.paste(tilted, (40, 460), tilted)

    page.save(ASSETS / "sample.png", optimize=True)
    print(f"wrote {ASSETS / 'sample.png'} {page.size}")
    return [title, *left, *right, long_line, tilted_text]


def stack(draw: ImageDraw.ImageDraw, word: str, x: int, y: int, size: int, gap: int) -> None:
    """Writes `word` one upright letter under the other, `gap` px between them."""
    f = font(size)
    for letter in word:
        box = draw.textbbox((0, 0), letter, font=f)
        width = box[2] - box[0]
        # Centre every letter on `x` so the column is straight whatever the glyph width.
        draw.text((x - width / 2 - box[0], y - box[1]), letter, font=f, fill="black")
        y += (box[3] - box[1]) + gap


def render_vertical() -> list[str]:
    page = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(page)
    heading = "Vertical text sample"
    draw.text((40, 30), heading, font=font(36), fill="black")

    # Spaced letters: the detector returns one box per glyph and the pipeline
    # groups them into a stack.
    stack(draw, "HOTEL", 120, 130, 72, 28)
    stack(draw, "MENU", 300, 130, 72, 28)
    # Touching letters: the detector merges the column into one tall box that
    # the pipeline re-detects at glyph level.
    stack(draw, "EXIT", 480, 130, 72, 2)

    footer = "Platform 9"
    draw.text((40, 720), footer, font=font(36), fill="black")

    page.save(ASSETS / "vertical.png", optimize=True)
    print(f"wrote {ASSETS / 'vertical.png'} {page.size}")
    return [heading, "HOTEL", "MENU", "EXIT", footer]


ASSETS.mkdir(parents=True, exist_ok=True)
expected = {"sample": render_sample(), "vertical": render_vertical()}
(ASSETS / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
print(f"wrote {ASSETS / 'expected.json'}")
