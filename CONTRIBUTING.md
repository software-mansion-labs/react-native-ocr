# Contributing

## Layout

```
src/            TypeScript pipeline (published)
src/native/     JSI bridge and typed wrappers over the native ops
cpp/            C++ kernels (OpenCV), JSI host functions, error guard
cpp/tests/      GoogleTest suites for the C++ kernels (host build)
ios/            TurboModule that installs the JSI object on iOS
android/        TurboModule, JNI adapter, Gradle and CMake for Android
example/        Expo app used for manual verification on devices
__tests__/      Jest suites, running against a JS fake of the native ops
```

## Setup

```sh
yarn install          # also runs react-native-executorch's native-lib download
yarn example ios      # builds and runs the example on a simulator
yarn example android  # builds and runs the example on a device/emulator
```

The example app is an Expo project; `expo run:*` generates `example/ios` and `example/android`,
which are git-ignored. Yarn creates no `node_modules` link for the root workspace, so
`example/react-native.config.js` points autolinking at the repository root explicitly and Metro
resolves the library through the Babel alias `react-native-builder-bob` adds in
`example/babel.config.js`. If another Metro already holds port 8081, pass `--port` to `expo run:*`.

## Checks

```sh
yarn typecheck
yarn lint
yarn test
yarn prepare          # bob build
```

Commits follow the Conventional Commits format (`feat:`, `fix:`, `chore:`, ...). Lefthook runs
lint, typecheck, tests and formatting on staged files; `clang-format` formats C++ when installed
(`brew install clang-format`).

## Native code

- OpenCV comes from react-native-executorch: the `opencv-rne` CocoaPod on iOS and the static
  libraries its postinstall downloads on Android. The app must declare the `ocr` feature.
- Every JSI host function takes and returns plain `ArrayBuffer`s with explicit shapes; the
  TypeScript wrappers in `src/native/ops.ts` bridge them to react-native-executorch tensors.
- Errors thrown from C++ reach JavaScript as `RnExecuTorchError`s with a `code`, so callers see one
  error type across both packages.
