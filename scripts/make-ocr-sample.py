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
- example/assets/ocr/layout.png: the page the Layout screen reads with the
  PP-DocLayout model: a paper-like page with a title, an author line, an
  abstract, two body columns with section headings, a figure and its caption,
  so that the layout detector has regions of several classes to find and the
  blocks come out in column order.
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


def paragraph(draw: ImageDraw.ImageDraw, lines: list[str], x: int, y: int, size: int, leading: int) -> int:
    """Writes `lines` one under the other and returns the y below the last one."""
    for line in lines:
        draw.text((x, y), line, font=font(size), fill="black")
        y += leading
    return y


def render_layout() -> list[str]:
    page = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(page)
    lines: list[str] = []

    title = "On-device Document Reading"
    draw.text((400 - draw.textlength(title, font=font(40)) / 2, 50), title, font=font(40), fill="black")
    authors = "A. Reader and B. Writer, Mobile Lab"
    draw.text((400 - draw.textlength(authors, font=font(20)) / 2, 110), authors, font=font(20), fill="black")
    lines += [title, authors]

    # Abstract across the full width.
    abstract_title = "Abstract"
    draw.text((60, 170), abstract_title, font=font(24), fill="black")
    abstract = [
        "We read printed pages on a phone without any server. A layout model finds",
        "the regions of the page, each region is read on its own crop and the blocks",
        "are returned in the order a person would read them, column by column.",
    ]
    paragraph(draw, abstract, 60, 210, 18, 28)
    lines += [abstract_title, *abstract]

    # Two body columns with section headings.
    left_title = "1 Introduction"
    draw.text((60, 330), left_title, font=font(24), fill="black")
    left = [
        "Scanned pages mix titles, body",
        "text, figures and tables. Reading",
        "them as one block loses the",
        "structure, so the pipeline first",
        "detects regions and then reads",
        "each one with the recognizer.",
        "Nested duplicates are merged by",
        "containment before reading.",
    ]
    paragraph(draw, left, 60, 370, 18, 28)
    lines += [left_title, *left]

    right_title = "2 Method"
    draw.text((420, 330), right_title, font=font(24), fill="black")
    right = [
        "The detector returns boxes with",
        "a class label and a confidence.",
        "Visual regions such as images",
        "carry no text and are skipped;",
        "every other region is cropped",
        "natively and read in place.",
    ]
    paragraph(draw, right, 420, 370, 18, 28)
    lines += [right_title, *right]

    # A figure in the right column: a framed line chart without text.
    x0, y0, x1, y1 = 420, 560, 740, 760
    draw.rectangle((x0, y0, x1, y1), fill=(235, 238, 245), outline=(90, 90, 90), width=2)
    draw.line((x0 + 30, y1 - 30, x1 - 20, y1 - 30), fill=(40, 40, 40), width=2)
    draw.line((x0 + 30, y1 - 30, x0 + 30, y0 + 20), fill=(40, 40, 40), width=2)
    points = [(x0 + 30 + i * 45, y1 - 30 - h) for i, h in enumerate((10, 60, 45, 110, 95, 150, 140))]
    draw.line(points, fill=(30, 90, 200), width=4)
    for px, py in points:
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(200, 60, 30))
    caption = "Figure 1. Throughput by batch size"
    draw.text((x0, y1 + 14), caption, font=font(17), fill="black")
    lines.append(caption)

    # The left column continues below the introduction.
    left_more_title = "3 Results"
    draw.text((60, 640), left_more_title, font=font(24), fill="black")
    left_more = [
        "Blocks come out in reading order",
        "on two-column pages, and each",
        "block keeps the label and the",
        "confidence the detector gave it.",
    ]
    paragraph(draw, left_more, 60, 680, 18, 28)
    lines += [left_more_title, *left_more]

    footer = "Preprint, October 2026"
    draw.text((400 - draw.textlength(footer, font=font(16)) / 2, 1040), footer, font=font(16), fill="black")
    lines.append(footer)

    page.save(ASSETS / "layout.png", optimize=True)
    print(f"wrote {ASSETS / 'layout.png'} {page.size}")
    return lines


ASSETS.mkdir(parents=True, exist_ok=True)
expected = {"sample": render_sample(), "vertical": render_vertical(), "layout": render_layout()}
(ASSETS / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
print(f"wrote {ASSETS / 'expected.json'}")
