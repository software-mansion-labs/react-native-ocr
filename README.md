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
- iOS 17+, Android API 26+, the New Architecture. react-native-executorch's pods require an iOS
  deployment target of 17.0, which is above the default of a fresh app: in an Expo app set it
  through `expo-build-properties` (`"ios": { "deploymentTarget": "17.0" }`), in a bare app in the
  `Podfile` (`platform :ios, '17.0'`), or `pod install` stops with "required a higher minimum
  deployment target".
- The package has native code, so an Expo app needs a development build (`expo run:ios` /
  `expo run:android`), not Expo Go; bare apps run `pod install` after installing.
- The app must opt into react-native-executorch's `ocr` feature so OpenCV is downloaded for both
  packages, in the app's `package.json`:

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
"chars" is the character accuracy over the page (edit distance per line). XNNPACK results are
identical on the iOS simulator, the Android emulator and a Galaxy S23; Core ML results are from
an iPhone 17 (the iOS `DEFAULT`), where a page takes 105 to 225 ms against 360 to 720 ms on the
simulator's XNNPACK.

| Alphabet | XNNPACK lines | XNNPACK chars | Core ML lines | Core ML chars | What goes wrong on XNNPACK                                           |
| -------- | ------------- | ------------- | ------------- | ------------- | -------------------------------------------------------------------- |
| English  | 7/11          | 97.8%         | 7/11          | 97.8%         | `snall`, `tlat`, `!` as `l`, a dropped period and a dropped space    |
| Cyrillic | 2/7           | 95.5%         | 1/7           | 92.9%         | `ра` as Latin `pa`, `и` as `ӏ`, `л` as `Jт`, `₽` as `П`, `ї` as `і`  |
| Latin    | 3/6           | 94.2%         | 4/6           | 97.1%         | `Ç` as `C`, `ö` as `Ö`, `,` as `;`, `M` as `^A`, a dropped period    |
| Japanese | 8/8           | 100%          | 8/8           | 100%          | nothing, vertical columns included                                   |
| Chinese  | 0/8           | 77.6%         | 6/8           | 94.8%         | common characters dropped outright (文 字 京 国 了 顾 春), see below |
| Korean   | 6/8           | 94.4%         | 6/8           | 94.4%         | `뛰어넘는다` as `뒤어넘논다`, `을` as `올`, a dropped space          |
| Telugu   | 5/6           | 99.0%         | 6/6           | 100%          | one dropped space                                                    |
| Kannada  | 2/6           | 96.1%         | 4/6           | 93.1%         | `ಠ` as `7`, a dropped conjunct, a doubled space and a doubled virama |

For Latin text prefer PP-OCRv6: its fp32 export read every line of the English page exactly where
the int8 EasyOCR English model read about 98% of the characters. The **XNNPACK** simplified
Chinese export never emits some common characters (the recognizer's blank wins those timesteps on
two different fonts, horizontally and vertically) while it reads the rest of the line correctly;
the Core ML export of the same model reads every one of them (its two misses are dropped
spaces), so this is a defect of the XNNPACK int8 quantization, still to be looked into on the export
side, not something this package can correct; until then use the Core ML export on iOS and
expect the drops on Android. The EasyOCR Vulkan exports (Galaxy S23) read the English,
Cyrillic and Latin pages at 7/11 97.8%, 1/7 94.2% and 3/6 95.0%, the XNNPACK reads with one
fp16-detector difference each way. The other alphabets' misses are the usual int8 CRNN
substitutions, and they differ between the two int8 exports (Cyrillic and Kannada are a little
worse on Core ML, Latin, Chinese, Telugu and Kannada's heading better).

## Known issue: the Vulkan exports

On Android the registry's `DEFAULT` entries resolve to the Vulkan exports. They need the ExecuTorch
Vulkan backend from react-native-executorch's **`0.10.6-libs`** native release or newer: with the
`0.10.4-libs` that every react-native-executorch npm release up to 0.10.5 pins, loading the
PP-OCRv6 and PaddleHelpers exports kills the app inside `loadModel` (a crash in Qualcomm's shader
compiler, reproduced on a Galaxy S23). With `0.10.6-libs` both load and run on the S23: PP-OCRv6
reads the sample page identically to XNNPACK and to react-native-executorch's own pipeline
(11/11 lines, 0.00 px), and orientation, dewarping and table structure give the same results as on
XNNPACK in about 60% of the time. Until an npm release of react-native-executorch pins those
libraries, either pin them yourself the way this repository does (`CONTRIBUTING.md`, a Yarn patch
of `nativeLibsVersion`) or use the `XNNPACK` entries on Android. Two limits remain whatever the
libraries: the PP-DocLayout Vulkan export (`models.layoutDetection.PP_DOCLAYOUT.VULKAN`, also the
Android `DEFAULT`) loads but fails at execute inside the Vulkan backend (`Vulkan graph declares 3
inputs and 4 outputs, but the delegate call supplied 8 arguments`): five of its decoder-layer
partitions return the same sigmoid tensor twice (the reference points for the next layer and
their detached copy for the box head), and the ExecuTorch 1.3.1 Vulkan serializer the export used
collapsed the two output slots into one; the serializer was fixed on ExecuTorch `main` on
2026-09-21 (pytorch/executorch#22898), so the export needs re-cutting with it, and until then use
`PP_DOCLAYOUT.XNNPACK` on Android; and every Vulkan export crashes
the Android emulator (swiftshader), so use `XNNPACK` there. The EasyOCR Vulkan exports keep the
recognizer on XNNPACK and read exactly as the XNNPACK ones.

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

## React Native OCR is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in
building web and mobile apps. We are Core React Native Contributors and experts in dealing with
all kinds of React Native issues. We can help you build your next dream product –
[Hire us](https://swmansion.com/contact/projects?utm_source=react-native-ocr&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-ocr-github 'Software Mansion')](https://swmansion.com)
