# @swmansion/react-native-ocr

![React Native OCR](https://github.com/user-attachments/assets/a1944a2a-a366-43a7-8dcd-5d5dfbba8730)

Document OCR for React Native on top of
[react-native-executorch](https://github.com/software-mansion/react-native-executorch): layout,
tables, dewarping, page orientation, vertical text and EasyOCR alphabets, all on device.

For plain text detection and recognition, react-native-executorch's
[`useOpticalCharacterRecognizer`](https://docs.swmansion.com/react-native-executorch/) is enough.
Use this package when you need what it leaves out.

> **Status: pre-release.** Nothing is published to npm yet.

- [:rocket: Features](#rocket-features)
- [:package: Installation](#package-installation)
- [:books: Usage](#books-usage)
- [:brain: Models](#brain-models)
- [:warning: Known issues](#warning-known-issues)
- [:page_with_curl: API reference](https://software-mansion-labs.github.io/react-native-ocr/)
- [:handshake: Contributing](#handshake-contributing)
- [:page_facing_up: License](#page_facing_up-license)

## :rocket: Features

- **Document layout** with PP-DocLayoutV3: 25 region types, each region read on its own crop,
  results as reading-ordered blocks.
- **Tables** with SLANet: each table region comes back as HTML.
- **Orientation and dewarping**: upright a rotated page and flatten a curled or photographed one.
- **Vertical text**: upright glyph stacks read top to bottom.
- **EasyOCR** in eight alphabets: English, Cyrillic, Latin, Japanese, simplified Chinese, Korean,
  Telugu, Kannada.
- **Reading order** that handles multi-column pages under a page-wide title.
- XNNPACK, Core ML and Vulkan exports with a per-platform `DEFAULT`.

## :package: Installation

```sh
yarn add @swmansion/react-native-ocr react-native-executorch
```

Set up react-native-executorch first (`react-native-executorch` 0.10.6+, iOS 17+, Android API 26+,
New Architecture, a development build on Expo), then enable its `ocr` feature in the app's
`package.json` so OpenCV is downloaded for both packages:

```json
{
  "react-native-executorch": {
    "features": ["ocr"]
  }
}
```

## :books: Usage

```tsx
import { models, useOcr, type DocLayoutLabel } from '@swmansion/react-native-ocr';

const { isReady, downloadProgress, runOcr } = useOcr<DocLayoutLabel>({
  ...models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT,
  layout: models.layoutDetection.PP_DOCLAYOUT.DEFAULT,
  documentModels: models.documentModels.PP_HELPERS.DEFAULT,
});

const { image, blocks } = await runOcr!(photo, { orientation: true, dewarp: true });
for (const { regionType, text, isTable, tableHtml } of blocks) {
  console.log(regionType, isTable ? tableHtml : text);
}
```

`layout` and `documentModels` are optional; without them you get whole-page OCR. `image` is the
corrected frame that all coordinates refer to. For vertical text, use an EasyOCR model and pass
`{ vertical: true }` to `runOcr`. Without React, use `createOcr(config)`.

## :brain: Models

| Registry entry                        | What it is                                              | XNNPACK | Core ML | Vulkan |
| ------------------------------------- | ------------------------------------------------------- | ------- | ------- | ------ |
| `models.ocr.PADDLE.PPOCRV6_SMALL`     | PP-OCRv6 small, react-native-executorch's export        | int8²   | int8    | fp16   |
| `models.ocr.EASYOCR.<ALPHABET>`       | EasyOCR, one per alphabet                               | int8    | int8    | fp16¹  |
| `models.layoutDetection.PP_DOCLAYOUT` | PP-DocLayoutV3                                          | fp32    | fp16    | fp16   |
| `models.documentModels.PP_HELPERS`    | Orientation, UVDoc dewarp and SLANet tables in one file | int8    | int8    | fp16   |

¹ CRAFT detector on Vulkan, CRNN recognizer on XNNPACK.
² int8 detector, fp32 recognizer; the fp32 detector is `XNNPACK_FP32`.

`DEFAULT` picks Core ML on an iOS device, XNNPACK on the simulator, and Vulkan on Android when the
app links it, else XNNPACK.

## :warning: Known issues

- Vulkan exports crash in `loadModel` on react-native-executorch older than 0.10.6 (its
  `0.10.6-libs` native release fixes Adreno 720/740); on older versions use the `XNNPACK` entries on
  Android.
- Vulkan exports crash the Android emulator; use `XNNPACK` there.
- The XNNPACK simplified Chinese EasyOCR export drops some common characters; Core ML reads them.

## :handshake: Contributing

Contributions are welcome! See the [contributing guide](CONTRIBUTING.md) to learn about the
development workflow, the repository layout, the native build, the sample pages and the release
process.

## :page_facing_up: License

MIT

## React Native OCR is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in
building web and mobile apps. We are Core React Native Contributors and experts in dealing with
all kinds of React Native issues. We can help you build your next dream product –
[Hire us](https://swmansion.com/contact/projects?utm_source=react-native-ocr&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-ocr-github 'Software Mansion')](https://swmansion.com)
