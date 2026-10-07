# @swmansion/react-native-ocr

Advanced on-device OCR for React Native, built on top of
[react-native-executorch](https://github.com/software-mansion/react-native-executorch).

> **Status: pre-release.** The package is being built up phase by phase; see
> `plan.md` for the roadmap and the current state. Nothing here is published to npm yet.

## What works today

- Whole-page text detection and recognition with PaddleOCR PP-OCRv6 through `useOcr` and
  `createOcr`: oriented quads with text and confidence, in reading order, with the same
  results as react-native-executorch's `useOpticalCharacterRecognizer` on the same model.
- EasyOCR (CRAFT + CRNN) in eight alphabets, verified on device for English, and vertical text
  (`vertical: true` per run): upright glyph stacks are clustered or re-detected glyph by glyph
  and read top to bottom.
- A model registry (`models`) with XNNPACK, Core ML and Vulkan exports and a per-platform
  `DEFAULT`.
- Pluggable detector strategies (`craftExtractBoxes`, `dbnetExtractBoxes`, or your own
  `TextBoxExtractor`), per-model recognizer options (charset, normalization, padding, a custom
  CTC decode) and per-run options.
- A small native module (C++ with OpenCV, JSI) for the kernels the pipeline needs and
  react-native-executorch does not expose: `rotateTensor`, `cropTensor`, `warpByGridTensor`,
  `padCornerMean`, `extractCraftTextBoxes`. Everything else runs on react-native-executorch's
  public API.

- Document layout with PP-DocLayoutV3 (`layout` on the model config): the page is split into
  labelled regions, each region is read on its own crop, and the result carries the regions and
  reading-ordered blocks.

## Planned

- Document helpers: page orientation, UVDoc dewarping and SLANet table structure
  (`documentModels` on the model config).

`createOcr` rejects a config that sets `documentModels` until it lands.

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
the list); a `table` region (`tableLabel`) always yields a block, flagged `isTable`, and will
carry `tableHtml` once the document helpers land. Reading each region on its own crop costs one
detector pass per region: the example's paper-like page takes about 2 s with layout against
0.8 s whole-page on the iOS simulator.

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

### Known limits of the int8 EasyOCR exports

The EasyOCR XNNPACK and Core ML exports are int8. On the example's printed page the English model
read about 90% of the characters right, with substitutions such as `m`/`n` or `!`/`l`, where the
fp32 PP-OCRv6 export read every line exactly; prefer PP-OCRv6 for Latin text.

### Known issue: the Vulkan export crashes at load

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
package; use `PP_DOCLAYOUT.XNNPACK` on Android.

## Requirements

- `react-native-executorch` 0.10.2 or newer, installed alongside this package.
- The app must opt into react-native-executorch's `ocr` feature so OpenCV is downloaded for both
  packages:

```json
{
  "react-native-executorch": {
    "features": ["ocr"]
  }
}
```

- iOS 17+, Android API 26+, the New Architecture, `react-native-worklets` 0.10 to 0.12.

## Example app

`example/` is an Expo app with four screens: **PaddleOCR parity** reads a bundled page with
this package and with react-native-executorch's own pipeline on the same export and compares
the results line by line; **EasyOCR + vertical text** reads the same page or a vertical-text page
with EasyOCR English and grades the result against the text the page was rendered with (it can
also save the raw CRAFT heatmap as a fixture for the native parity test); **Vulkan diagnostics**
(Android) loads Vulkan exports one at a time to attribute a crash; **Native spike** self-tests the
native module.

## Development

```sh
yarn install
yarn example ios       # or: yarn example android
yarn typecheck && yarn lint && yarn test
yarn test:native       # GoogleTest suites for the C++ kernels, against the host OpenCV
```

See `CONTRIBUTING.md` for the repository layout and the native build.

## License

MIT
