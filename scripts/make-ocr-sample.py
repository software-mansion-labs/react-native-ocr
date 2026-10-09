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
- example/assets/ocr/rotated.png: `sample.png` turned 90 degrees clockwise,
  for the Document screen's orientation toggle.
- example/assets/ocr/curved.png: `sample.png` as a phone would photograph it
  lying curled on a dark table (bowed lines, shaded, with a border around the
  page), for the Document screen's dewarp toggle.
- example/assets/ocr/table.png: a page with a title and a ruled table, for
  the Document screen's table structure.
- example/assets/ocr/<script>.png, one per EasyOCR alphabet besides English
  (cyrillic, latin, japanese, zh_sim, korean, telugu, kannada): a heading,
  a few lines of running text in that script, a line with digits and
  punctuation, and on the CJK pages two columns of vertical text, for the
  EasyOCR screen's per-script check. Every character is checked against the
  recognizer's alphabet in `src/charsets.ts`, so a miss on device is the
  model's, never the page's.
- example/assets/ocr/expected.json: the lines each page carries, so a screen
  can grade what it read, plus the table's row and column counts.

Needs Pillow with raqm (for the Indic conjuncts), NumPy and the macOS system fonts.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "example" / "assets" / "ocr"
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


def render_rotated() -> None:
    """Turns the sample page 90 degrees clockwise (PIL's ROTATE_270 is counter-clockwise)."""
    page = Image.open(ASSETS / "sample.png").transpose(Image.Transpose.ROTATE_270)
    page.save(ASSETS / "rotated.png", optimize=True)
    print(f"wrote {ASSETS / 'rotated.png'} {page.size}")


def render_curved() -> None:
    """Photographs the sample page, so to speak: scaled down onto a dark table with a
    sinusoidal curl (every column shifted down by a bow that peaks mid-page), a brightness
    roll-off towards the curled edge and a soft shadow, so the page has borders a dewarping
    model can find and text lines that are visibly bent."""
    src = np.asarray(Image.open(ASSETS / "sample.png").convert("RGB"), dtype=np.float32)
    src_h, src_w = src.shape[:2]
    out_w, out_h = 900, 700
    scale = 0.84
    x0, y0 = 60, 80
    bow = 46.0  # px of downward shift at the middle of the page

    # Inverse map: for every output pixel, where in the page it comes from.
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    u = (xs - x0) / scale
    v = (ys - y0 - bow * np.sin(np.pi * np.clip(u, 0, src_w) / src_w)) / scale
    inside = (u >= 0) & (u < src_w - 1) & (v >= 0) & (v < src_h - 1)
    uc = np.clip(u, 0, src_w - 1.001)
    vc = np.clip(v, 0, src_h - 1.001)
    ui, vi = np.floor(uc).astype(int), np.floor(vc).astype(int)
    fu, fv = (uc - ui)[..., None], (vc - vi)[..., None]
    page = (
        src[vi, ui] * (1 - fu) * (1 - fv)
        + src[vi, ui + 1] * fu * (1 - fv)
        + src[vi + 1, ui] * (1 - fu) * fv
        + src[vi + 1, ui + 1] * fu * fv
    )
    # Shade the page as if lit from the left, darker towards the curled right edge.
    shade = (0.72 + 0.28 * np.cos(np.pi * np.clip(u, 0, src_w) / src_w / 2))[..., None]
    page *= shade

    table = np.empty((out_h, out_w, 3), dtype=np.float32)
    table[...] = (74, 70, 66)
    table += (ys / out_h * 18)[..., None]  # a little lighter towards the bottom
    shadow = np.roll(inside, (10, 12), axis=(0, 1))
    table[shadow] *= 0.6
    out = np.where(inside[..., None], page, table)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(ASSETS / "curved.png", optimize=True)
    print(f"wrote {ASSETS / 'curved.png'} {(out_w, out_h)}")


TABLE_ROWS = [
    ["Item", "Qty", "Unit price", "Total"],
    ["Paper, A4", "12", "4.20", "50.40"],
    ["Toner", "2", "61.00", "122.00"],
    ["Stapler", "1", "9.90", "9.90"],
    ["Envelopes", "30", "0.15", "4.50"],
]


def render_table() -> tuple[list[str], list[int]]:
    page = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(page)
    title = "Quarterly stock count"
    draw.text((60, 40), title, font=font(34), fill="black")
    lines = [title]

    x0, y0, x1, y1 = 60, 120, 740, 480
    cols = [x0, 320, 440, 590, x1]
    rows = [y0 + i * (y1 - y0) // len(TABLE_ROWS) for i in range(len(TABLE_ROWS) + 1)]
    draw.rectangle((x0, rows[0], x1, rows[1]), fill=(232, 236, 242))
    for x in cols:
        draw.line((x, y0, x, y1), fill="black", width=2)
    for y in rows:
        draw.line((x0, y, x1, y), fill="black", width=2)
    f = font(22)
    for r, row in enumerate(TABLE_ROWS):
        for c, text in enumerate(row):
            box = draw.textbbox((0, 0), text, font=f)
            cx = (cols[c] + cols[c + 1]) / 2 - (box[2] - box[0]) / 2
            cy = (rows[r] + rows[r + 1]) / 2 - (box[3] - box[1]) / 2 - box[1]
            draw.text((cx, cy), text, font=f, fill="black")
            lines.append(text)

    page.save(ASSETS / "table.png", optimize=True)
    print(f"wrote {ASSETS / 'table.png'} {page.size}")
    return lines, [len(TABLE_ROWS), len(TABLE_ROWS[0])]


def alphabets() -> dict[str, set[str]]:
    """The EasyOCR alphabets from `src/charsets.ts`, keyed as there."""
    source = (ROOT / "src" / "charsets.ts").read_text(encoding="utf8")
    body = re.search(r"export const alphabets = \{(.*?)\n\}", source, re.S).group(1)
    return {
        key: set(value.replace("\\'", "'").replace("\\\\", "\\"))
        for key, value in re.findall(r"\n  ([a-z_]+):\s*\n?\s*'((?:\\.|[^'])*)'", body)
    }


class Script:
    def __init__(
        self,
        key: str,
        font_path: str,
        heading: str,
        lines: list[str],
        columns: list[str] = (),
    ) -> None:
        self.key = key
        self.font_path = font_path
        self.heading = heading
        self.lines = lines
        # Vertical columns, written top to bottom one glyph under the other.
        self.columns = list(columns)


# Running text in each script, sized so that a line fits the page at 30 px.
# The CJK pages carry two vertical columns on the right, as a sign or a book
# spine would, because that is where vertical text occurs in practice.
SCRIPTS = [
    Script(
        "cyrillic",
        FONT,
        "Проверка распознавания",
        [
            "Быстрая коричневая лиса",
            "прыгает через ленивую собаку",
            "Счет 40713 на сумму 1 284,50 ₽",
            "Срок оплаты: 14 октября",
            "Київ, вулиця Хрещатик",
            "Спасибо!",
        ],
    ),
    Script(
        "latin",
        FONT,
        "Prüfung der Erkennung",
        [
            "Zażółć gęślą jaźń",
            "Ça coûte 1 284,50 € à Genève",
            "Año nuevo en São Paulo",
            "Straße, Köln, Ærø, Þórshöfn",
            "Thank you! Merci. Danke.",
        ],
    ),
    Script(
        "japanese",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "文字認識の確認",
        [
            "東京駅から新宿まで",
            "きつねが犬を飛び越える",
            "請求書番号 40713",
            "合計 1,284円",
            "ありがとうございました",
        ],
        ["春夏秋冬", "日本語の縦書き"],
    ),
    Script(
        "zh_sim",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "文字识别测试",
        [
            "北京市朝阳区建国路",
            "狐狸跳过了懒狗",
            "发票编号 40713",
            "总计 1284.50 元",
            "谢谢惠顾",
        ],
        ["春夏秋冬", "中文竖排文字"],
    ),
    Script(
        "korean",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "문자 인식 확인",
        [
            "서울특별시 강남구",
            "여우가 게으른 개 위로 뛰어넘는다",
            "청구서 번호 40713",
            "합계 1,284원",
            "감사합니다",
        ],
        ["봄여름가을겨울", "세로쓰기"],
    ),
    Script(
        "telugu",
        "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",
        "పాఠ్య గుర్తింపు పరీక్ష",
        [
            "హైదరాబాద్ నగరం",
            "నక్క కుక్కను దాటి దూకింది",
            "బిల్లు సంఖ్య 40713",
            "మొత్తం 1284.50",
            "ధన్యవాదాలు",
        ],
    ),
    Script(
        "kannada",
        "/System/Library/Fonts/Supplemental/Kannada Sangam MN.ttc",
        "ಪಠ್ಯ ಗುರುತಿಸುವಿಕೆ ಪರೀಕ್ಷೆ",
        [
            "ಬೆಂಗಳೂರು ನಗರ",
            "ನರಿ ನಾಯಿಯನ್ನು ದಾಟಿ ಹಾರಿತು",
            "ಬಿಲ್ ಸಂಖ್ಯೆ 40713",
            "ಒಟ್ಟು 1284.50",
            "ಧನ್ಯವಾದಗಳು",
        ],
    ),
]


def render_script(script: Script, alphabet: set[str]) -> list[str]:
    texts = [script.heading, *script.lines, *script.columns]
    for text in texts:
        unknown = sorted({c for c in text if c not in alphabet})
        if unknown:
            raise SystemExit(f"{script.key}: {text!r} uses characters outside the alphabet: {unknown}")

    def f(size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(script.font_path, size)

    page = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(page)
    draw.text((40, 30), script.heading, font=f(44), fill="black")
    # Indic vowel signs reach well above and below the base line: give them room.
    leading = 64 if script.key in ("telugu", "kannada") else 56
    for i, line in enumerate(script.lines):
        draw.text((40, 130 + i * leading), line, font=f(30), fill="black")

    # Two vertical columns down the right-hand side, one glyph under the other
    # with the leading of a printed column, so the detector sees separate
    # glyph boxes that the vertical pass groups into stacks.
    for i, column in enumerate(script.columns):
        x = 700 - i * 90
        y = 130
        glyph_font = f(44)
        for glyph in column:
            box = draw.textbbox((0, 0), glyph, font=glyph_font)
            draw.text((x - (box[2] - box[0]) / 2 - box[0], y - box[1]), glyph, font=glyph_font, fill="black")
            y += (box[3] - box[1]) + 12

    page.save(ASSETS / f"{script.key}.png", optimize=True)
    print(f"wrote {ASSETS / f'{script.key}.png'} {page.size}")
    return texts


ASSETS.mkdir(parents=True, exist_ok=True)
if not features.check("raqm"):
    raise SystemExit("Pillow was built without raqm: the Telugu and Kannada pages would not be shaped")
sample = render_sample()
render_rotated()
render_curved()
table, table_shape = render_table()
expected = {
    "sample": sample,
    "vertical": render_vertical(),
    "layout": render_layout(),
    "rotated": sample,
    "curved": sample,
    "table": table,
    "tableShape": table_shape,
}
known = alphabets()
for script in SCRIPTS:
    expected[script.key] = render_script(script, known[script.key])
(ASSETS / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
print(f"wrote {ASSETS / 'expected.json'}")
