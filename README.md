# @swmansion/react-native-ocr

Advanced on-device OCR for React Native, built on top of
[react-native-executorch](https://github.com/software-mansion/react-native-executorch).

> **Status: pre-release.** The package is being built up phase by phase; see
> `plan.md` for the roadmap. Nothing here is published to npm yet.

## What it will provide

- EasyOCR (CRAFT + CRNN, eight alphabets) and PaddleOCR PP-OCRv6 text detection and recognition
  as one fused ExecuTorch program per model.
- Vertical text: glyph-stack clustering and per-glyph recognition.
- Document layout with PP-DocLayoutV3 and per-region OCR.
- Document helpers: page orientation, UVDoc dewarping and SLANet table structure.
- A small native module (C++ with OpenCV, JSI) for the kernels the pipeline needs and
  react-native-executorch does not expose. Everything else runs on react-native-executorch's
  public API.

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

## Development

```sh
yarn install
yarn example ios       # or: yarn example android
yarn typecheck && yarn lint && yarn test
```

See `CONTRIBUTING.md` for the repository layout and the native build.

## License

MIT
