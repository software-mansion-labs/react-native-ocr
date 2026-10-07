# @swmansion/react-native-ocr

Advanced on-device OCR for React Native, built on top of
[react-native-executorch](https://github.com/software-mansion/react-native-executorch): EasyOCR
and PaddleOCR text detection and recognition, vertical text, document layout, page orientation,
dewarping and table structure, all running on the device through ExecuTorch.

> **Status: pre-release.** `0.1.0` is being prepared and nothing is published to npm yet; the
> API below is what it will ship. `plan.md` holds the roadmap and the verification record.

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [EasyOCR alphabets](#easyocr-alphabets)
- [Known issue: the Vulkan exports](#known-issue-the-vulkan-exports)
- [API reference](https://software-mansion-labs.github.io/react-native-ocr/)

## Features

- Whole-page text detection and recognition with PaddleOCR PP-OCRv6 through `useOcr` and
  `createOcr`: oriented quads with text and confidence, in reading order, with the same results
  as react-native-executorch's `useOpticalCharacterRecognizer` on the same model.
- EasyOCR (CRAFT + CRNN) in eight alphabets (English, Cyrillic, Latin, Japanese, simplified
  Chinese, Korean, Telugu, Kannada), and vertical text (`vertical: true` per run): upright glyph
  stacks are clustered or re-detected glyph by glyph and read top to bottom.
- Document layout with PP-DocLayoutV3 (`layout` on the model config): the page is split into
  labelled regions, each region is read on its own crop, and the result carries the regions and
  reading-ordered blocks.
- Document helpers with PaddleHelpers (`documentModels` on the model config): page orientation,
  UVDoc dewarping and SLANet table structure (`tableHtml` per table block), each toggled per run.
- A model registry (`models`) with XNNPACK, Core ML and Vulkan exports and a per-platform
  `DEFAULT`.
- Pluggable detector strategies (`craftExtractBoxes`, `dbnetExtractBoxes`, or your own
  `TextBoxExtractor`), per-model recognizer options (charset, normalization, padding, a custom CTC
  decode) and per-run options.
- A small native module (C++ with OpenCV, JSI) for the kernels the pipeline needs and
  react-native-executorch does not expose: `rotateTensor`, `cropTensor`, `warpByGridTensor`,
  `padCornerMean`, `extractCraftTextBoxes`. Everything else runs on react-native-executorch's
  public API, and the whole pipeline runs on its worklet runtime.

## Installation

```sh
yarn add @swmansion/react-native-ocr react-native-executorch
```

Requirements:

- `react-native-executorch` 0.10.2 or newer, installed alongside this package, and
  `react-native-worklets` 0.10 to 0.12.
- iOS 17+, Android API 26+, the New Architecture. The package has native code, so an Expo app
  needs a development build (`expo run:ios` / `expo run:android`), not Expo Go; bare apps run
  `pod install` after installing.
- The app must opt into react-native-executorch's `ocr` feature so OpenCV is downloaded for both
  packages:

```json
{
  "react-native-executorch": {
    "features": ["ocr"]
  }
}
```

Models download from Hugging Face on first use and are cached by react-native-executorch;
`useOcr` reports `downloadProgress`.

## Usage

```tsx
import { models, useOcr } from '@swmansion/react-native-ocr';

function Screen() {
  const { isReady, downloadProgress, error, runOcr } = useOcr(
    models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT
  );

  const read = async (image: ImageBuffer) => {
    const { detections } = await runOcr!(image);
    for (const { text, confidence, quad } of detections) {
      console.log(text, confidence, quad);
    }
  };
  // ...
}
```

`ImageBuffer` is react-native-executorch's raw `uint8` HWC image (`rgb`, `rgba`, `bgr`, `bgra`
or `gray`). Without React, `createOcr(config, runtime?)` returns `{ runOcr, runOcrWorklet,
dispose }`; `runOcr` resolves on a worklet runtime, `runOcrWorklet` runs synchronously on the
calling worklet.

A result carries `detections` (flat, in reading order), `blocks` (one `'ungrouped'` block
without a layout model), `regions` (empty without one) and `image`, the frame the coordinates
refer to. Reading order is found by recursive cuts at column gutters and blank bands, so the
columns under a page-wide title still read left column first (react-native-executorch's PaddleOCR
task interleaves them).

### Layout

```tsx
const { isReady, runOcr } = useOcr<DocLayoutLabel>({
  ...models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT,
  layout: models.layoutDetection.PP_DOCLAYOUT.DEFAULT,
});

const { blocks, regions } = await runOcr!(image);
for (const { regionType, layoutConfidence, text, bbox } of blocks) {
  console.log(regionType, layoutConfidence, bbox, text);
}
```

With a `layout` model, PP-DocLayoutV3 (25 labels, `DOC_LAYOUT_LABELS`) finds the regions of the
page, nested duplicates are merged (a region 80% inside a larger one is dropped, except that
visual regions never swallow a caption), and every region that carries text is read on its own
crop; the lines are returned in page coordinates. `blocks` has one block per region with text
(`regionType`, `bbox`, `layoutConfidence`, `lines`, `text`) in reading order; `detections` follows
the blocks.
Regions labelled `image`, `chart` or `seal` are skipped (`visualLabels` on the config changes
the list); a `table` region (`tableLabel`) always yields a block, flagged `isTable`, and carries
`tableHtml` when the document helpers are loaded with a table config. Reading each region on its
own crop costs one detector pass per region: the example's paper-like page takes about 2 s with
layout against 0.8 s whole-page on the iOS simulator.

### Orientation, dewarp and tables

```tsx
const { isReady, runOcr } = useOcr<DocLayoutLabel>({
  ...models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT,
  layout: models.layoutDetection.PP_DOCLAYOUT.DEFAULT,
  documentModels: models.documentModels.PP_HELPERS.DEFAULT,
});

const { image, blocks } = await runOcr!(photo, { orientation: true, dewarp: true });
for (const block of blocks) {
  if (block.isTable) console.log(block.tableHtml);
}
```

`documentModels` loads the fused PaddleHelpers program alongside the OCR model. Per run,
`orientation` classifies the page's rotation (0, 90, 180 or 270 degrees) and turns it upright
when the classifier is at least `orientationMinConfidence` sure (default 0.85), then `dewarp`
flattens a curled or photographed page through the predicted UVDoc sampling grid, declining the
warp when it would wipe out most of the page's variance (`dewarpMinVarianceRatio`, default
0.5). A page photographed on a dark background keeps only about a third of its variance once
the background is cropped away, so set the ratio to about 0.1 for photos (the example's curved
page does); a collapsed page keeps close to none. Both are off unless the config or the run turns
them on; the result's `image` is the corrected frame, and every coordinate refers to it. With a
`layout` model, each `table` region is also run through the SLANet structure decoder and its
lines are placed into the recognized `<tr>`/`<td>` skeleton by nearest row and column centre
(`fillTableCells`, also exported for your own skeletons); that is on whenever the helpers carry a
table config, and `tables: false` skips it for a run.

### Vertical text

```tsx
const { detections } = await runOcr!(image, { vertical: true });
```

With EasyOCR (CRAFT), a tall box the detector merged from stacked letters is re-detected at glyph
level and each glyph is read on its own; spaced glyphs are clustered into a stack first. Reads
below `verticalMinConfidence` (default 0.4) are dropped. Two limits worth knowing: an isolated
glyph gives the recognizer no context, so `O` and `0` or `l` and `I` can swap, and letters that
touch are only approximately separated. With PaddleOCR (DBNet, no glyph-level mode) stacks are
read as clustered glyphs only.

## Models

Every entry of the registry carries `XNNPACK`, `COREML` and `VULKAN` exports plus a `DEFAULT`
that resolves to the fastest one the platform can run: Core ML on an iOS device, XNNPACK on the
simulator, Vulkan on Android when the app links that backend, else XNNPACK. The exports are
published on Hugging Face by Software Mansion.

| Registry entry                        | What it is                                                | XNNPACK | Core ML | Vulkan |
| ------------------------------------- | --------------------------------------------------------- | ------- | ------- | ------ |
| `models.ocr.PADDLE.PPOCRV6_SMALL`     | PP-OCRv6 small: DBNet detector + SVTR recognizer          | fp32    | int8    | fp16   |
| `models.ocr.EASYOCR.<ALPHABET>`       | EasyOCR: CRAFT detector + CRNN recognizer, one per script | int8    | int8    | fp16¹  |
| `models.layoutDetection.PP_DOCLAYOUT` | PP-DocLayoutV3, 25 region classes                         | fp32    | fp16    | fp16   |
| `models.documentModels.PP_HELPERS`    | PaddleHelpers: orientation, UVDoc dewarp, SLANet tables   | int8    | int8    | fp16   |

¹ the CRAFT detector on Vulkan, the CRNN recognizer on XNNPACK.

The int8 Core ML PP-OCRv6 export detects less small text than the fp32 XNNPACK one; that is a
property of the export, the same as under react-native-executorch.

## EasyOCR alphabets

The EasyOCR exports are int8 (XNNPACK and Core ML) and read a printed page as below. Each page
(`example/assets/ocr/<script>.png`, rendered by `scripts/make-ocr-sample.py`) carries a heading,
a few lines of running text and a line with digits; the Japanese, Chinese and Korean pages also
carry two vertical columns, read with `vertical: true`. "Lines" counts lines read exactly;
"chars" is the character accuracy over the page (edit distance per line). Results are from the
XNNPACK export, identical on the iOS simulator and the Android emulator.

| Alphabet | Lines | Chars | What goes wrong                                                      |
| -------- | ----- | ----- | -------------------------------------------------------------------- |
| English  | 7/11  | 97.8% | `snall`, `tlat`, `!` as `l`, a dropped period and a dropped space    |
| Cyrillic | 2/7   | 95.5% | `ра` as Latin `pa`, `и` as `ӏ`, `л` as `Jт`, `₽` as `П`, `ї` as `і`  |
| Latin    | 3/6   | 94.2% | `Ç` as `C`, `ö` as `Ö`, `,` as `;`, `M` as `^A`, a dropped period    |
| Japanese | 8/8   | 100%  | nothing, vertical columns included                                   |
| Chinese  | 0/8   | 77.6% | common characters dropped outright (文 字 京 国 了 顾 春), see below |
| Korean   | 6/8   | 94.4% | `뛰어넘는다` as `뒤어넘논다`, `을` as `올`, a dropped space          |
| Telugu   | 5/6   | 99.0% | one dropped space                                                    |
| Kannada  | 2/6   | 96.1% | `ಠ` as `7`, a dropped conjunct, a doubled space and a doubled virama |

For Latin text prefer PP-OCRv6: its fp32 export read every line of the English page exactly where
the int8 EasyOCR English model read about 98% of the characters. The simplified Chinese export
never emits some common characters (the recognizer's blank wins those timesteps on two different
fonts, horizontally and vertically) while it reads the rest of the line correctly; that is a
defect of the int8 export being reported to the exporters, not something this package can
correct. The other alphabets' misses are the usual int8 CRNN substitutions.

## Known issue: the Vulkan exports

`models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT` resolves to the Vulkan export on Android, and loading
it currently kills the app from inside react-native-executorch's `loadModel` (a crash in the
ExecuTorch Vulkan backend, reproduced on a Galaxy S23 and on the emulator). The fix is being
made in ExecuTorch and will arrive through a react-native-executorch release; this package needs
no change for it. Until then, use `models.ocr.PADDLE.PPOCRV6_SMALL.XNNPACK` on Android. The
EasyOCR Vulkan exports load and run on a Galaxy S23 (they keep the recognizer on XNNPACK), but
crash the Android emulator; use `XNNPACK` there. The PP-DocLayout Vulkan export
(`models.layoutDetection.PP_DOCLAYOUT.VULKAN`, also the Android `DEFAULT`) loads on the S23 but
fails at execute inside ExecuTorch's Vulkan backend (`Vulkan graph declares 3 inputs and 4
outputs, but the delegate call supplied 8 arguments`), an export defect being fixed outside this
package; use `PP_DOCLAYOUT.XNNPACK` on Android. The PaddleHelpers Vulkan export
(`models.documentModels.PP_HELPERS.VULKAN`, the Android `DEFAULT`) crashes at load on the S23
exactly like PP-OCRv6's; use `PP_HELPERS.XNNPACK` on Android until the fix ships.

## API reference

The full API reference, generated from the source, is at
[software-mansion-labs.github.io/react-native-ocr](https://software-mansion-labs.github.io/react-native-ocr/).

## Example app

`example/` is an Expo app with a screen per feature: **PaddleOCR parity** reads a bundled page
with this package and with react-native-executorch's own pipeline on the same export and compares
the results line by line; **EasyOCR + vertical text** reads the English pages and one page per
alphabet, with the vertical pass, and grades the result against the text the page was rendered
with (it can also save the raw CRAFT heatmap as a fixture for the native parity test); **Layout**
reads a paper-like page region by region; **Document helpers** runs orientation, dewarp and table
structure on bundled pages or a photo from the camera; **Vulkan diagnostics** (Android) loads
Vulkan exports one at a time to attribute a crash; **Native spike** self-tests the native module.

## Development

```sh
yarn install
yarn example ios       # or: yarn example android
yarn typecheck && yarn lint && yarn test
yarn test:native       # GoogleTest suites for the C++ kernels, against the host OpenCV
yarn docs              # the API reference, into docs/
```

See `CONTRIBUTING.md` for the repository layout, the native build, the sample pages and the
release process.

## License

MIT
